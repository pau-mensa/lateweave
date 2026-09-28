# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#   "bm25s==0.3.11",
#   "numpy>=1.26,<3; python_version < '3.14'",
#   "numpy>=2.3,<3; python_version >= '3.14'",
#   "PyStemmer>=2.2,<4",
#   "scipy>=1.11",
# ]
# ///
"""BM25 gathering with an optional stored-MaxSim rerank.

Run from the lateweave package directory:

    uv run --with-editable . cookbook/bm25_stored_maxsim.py --help

The lexical stage is bm25s. It consumes only the query text, so it declares no
query features. What a caller can still get wrong is building with one analyzer
and querying with another, which loses terms silently. So the analyzer is
persisted next to the index and read back on every open and every mutation.

The rerank stage is lateweave's MaxSim reranker over a lateweave vector store.
The store carries the encoder representation it was built from, and a search
that supplies token embeddings must supply them from that encoder. A search
without embeddings is gather-only: BM25 scores rank.

Both stages index one segment: the external IDs of ``documents.jsonl`` in
order, at the generation ``corpus-manifest.json`` records. The vector store
keeps its own copy of the segment, so a pipeline pairing the lexical index with
a store built over other documents, or left at another generation, is refused.

bm25s has no incremental append or delete, so ``update`` and ``delete`` rebuild
the lexical index from ``documents.jsonl``, which this recipe maintains anyway.
The vector store keeps its incremental paths, and each mutation checks that the
store's next segment is the one the documents file now describes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import gc
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterator, Sequence
import unicodedata
import uuid

import numpy as np

from lateweave import (
    Candidate,
    CorpusManifest,
    Feature,
    Float32VectorStore,
    Int8VectorStore,
    MaxSimReranker,
    Query,
    Representation,
    ResourceBudget,
    SearchPipeline,
    Segment,
    open_vector_store,
)


CORPUS_MANIFEST = "corpus-manifest.json"
ANALYZER_FILE = "analyzer.json"
DOCUMENTS_FILE = "documents.jsonl"
VECTOR_DIRECTORY = "vectors"
LEXICAL_DIRECTORY = "bm25"
STORES = {"float32": Float32VectorStore, "int8": Int8VectorStore}

# Lucene's defaults, spelled out so a bm25s default change cannot move results.
LEXICAL_METHOD = "lucene"
LEXICAL_K1 = 1.5
LEXICAL_B = 0.75
LEXICAL_TOKENIZER = "unicode-fold"

_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class Analyzer:
    """Text to terms, identically for documents and queries.

    ``stemmer`` is a Snowball algorithm name (``"spanish"``, ``"english"``, ...)
    or ``None``. None is the default because it is language-agnostic.
    """

    stemmer: str | None = None

    def __post_init__(self) -> None:
        stem = None
        if self.stemmer is not None:
            import Stemmer

            if self.stemmer not in Stemmer.algorithms():
                raise ValueError(
                    f"unknown Snowball algorithm {self.stemmer!r}; available: "
                    f"{', '.join(sorted(Stemmer.algorithms()))}"
                )
            stem = Stemmer.Stemmer(self.stemmer).stemWords
        object.__setattr__(self, "_stem", stem)

    def tokens(self, text: str) -> list[str]:
        """Casefolded, accent-stripped ``\\w+`` runs, optionally stemmed."""
        normalized = unicodedata.normalize("NFKD", text.casefold())
        folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        terms = _TOKEN.findall(folded)
        if self._stem is not None:
            terms = self._stem(terms)
        return [sys.intern(term) for term in terms]

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "method": LEXICAL_METHOD,
                    "tokenizer": LEXICAL_TOKENIZER,
                    "stemmer": self.stemmer,
                    "k1": LEXICAL_K1,
                    "b": LEXICAL_B,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def read(cls, path: Path) -> "Analyzer":
        """Recover the chain an index was built with, refusing an unknown one."""
        parameters = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = parameters.get("tokenizer")
        if tokenizer != LEXICAL_TOKENIZER:
            raise ValueError(
                f"index was built with tokenizer {tokenizer!r}, which this "
                f"recipe cannot reproduce; expected {LEXICAL_TOKENIZER!r}"
            )
        return cls(stemmer=parameters.get("stemmer"))


def write_lexical_index(path: Path, texts: Sequence[str], analyzer: Analyzer) -> None:
    """Build the lexical index at ``path``, replacing anything already there."""
    import bm25s

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    index = bm25s.BM25(method=LEXICAL_METHOD, k1=LEXICAL_K1, b=LEXICAL_B)
    index.index([analyzer.tokens(text) for text in texts], show_progress=False)
    index.save(str(path), show_progress=False)
    del index
    gc.collect()


@contextmanager
def index_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path = path.parent / f".{path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def staged_index_copy(source: Path) -> Path:
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.name}.mutation.", dir=source.parent))
    shutil.copytree(source, temporary, dirs_exist_ok=True)
    return temporary


def publish_replacement(source: Path, replacement: Path) -> None:
    backup = source.parent / f".{source.name}.backup.{uuid.uuid4().hex}"
    source.replace(backup)
    try:
        replacement.replace(source)
    except BaseException:
        backup.replace(source)
        raise
    else:
        shutil.rmtree(backup)


def read_segment(source: Path) -> Segment:
    """The segment the manifest records, checked against ``documents.jsonl``."""
    documents = load_documents(source / DOCUMENTS_FILE)
    return Segment.from_manifest(
        CorpusManifest.read(source / CORPUS_MANIFEST), [row["id"] for row in documents]
    )


def require_same_segment(store_segment: Segment, expected: Segment) -> None:
    if store_segment != expected:
        raise RuntimeError(
            f"the vector store moved to {store_segment!r}, but the documents describe {expected!r}"
        )


class LexicalCandidateGenerator:
    """Cookbook adapter; the lexical index remains external to lateweave."""

    requires: dict[str, Representation] = {}
    score_semantics = "bm25s-lucene"

    def __init__(self, index: Any, segment: Segment, analyzer: Analyzer) -> None:
        self.index = index
        self.segment = segment
        self.segments = (segment,)
        self.analyzer = analyzer

    @classmethod
    def open(cls, path: Path, segment: Segment, analyzer: Analyzer) -> "LexicalCandidateGenerator":
        import bm25s

        index = bm25s.BM25.load(str(path), mmap=True, load_corpus=False, show_progress=False)
        stored = int(index.scores["num_docs"])
        if stored != len(segment):
            raise RuntimeError(
                f"lexical index holds {stored:,} documents but the segment "
                f"holds {len(segment):,}"
            )
        return cls(index, segment, analyzer)

    def gather(
        self, query: Query, limit: int, *, subset: dict[str, np.ndarray] | None = None
    ) -> tuple[Candidate, ...]:
        terms = self.analyzer.tokens(query.text)
        if not terms:
            return ()
        document_count = len(self.segment)
        weight_mask = None
        if subset is not None:
            # bm25s multiplies scores by the mask; masked documents score zero
            # and are dropped below with every other non-matching document.
            weight_mask = np.zeros(document_count, dtype=np.float32)
            weight_mask[subset[self.segment.corpus_id]] = 1.0
        documents, scores = self.index.retrieve(
            [terms],
            k=min(limit, document_count),
            show_progress=False,
            weight_mask=weight_mask,
        )
        candidates = []
        seen: set[int] = set()
        for raw_document_id, raw_score in zip(documents[0].tolist(), scores[0].tolist()):
            score = float(raw_score)
            # Under the Lucene idf a zero score shares no term with the query.
            if score != score or score <= 0.0:
                continue
            document_id = int(raw_document_id)
            if not 0 <= document_id < document_count:
                raise RuntimeError(f"bm25s returned out-of-range ID {document_id}")
            if document_id in seen:
                raise RuntimeError(f"bm25s returned duplicate ID {document_id}")
            seen.add(document_id)
            candidates.append(
                Candidate(self.segment, document_id, score, len(candidates), "bm25s")
            )
        return tuple(candidates)


def load_documents(path: Path) -> list[dict[str, str]]:
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                documents.append({"id": str(row["id"]), "text": str(row["text"])})
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid document at {path}:{line_number}") from error
    if not documents:
        raise ValueError("document input is empty")
    ids = [row["id"] for row in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("external document IDs must be unique")
    return documents


def load_packed_embeddings(embeddings_path: Path, lengths_path: Path) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    lengths = np.asarray(np.load(lengths_path), dtype=np.int64)
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        raise ValueError("document embeddings must be a float32 [tokens, dimension] array")
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("document lengths must be a positive int64 vector")
    if int(lengths.sum()) != len(embeddings):
        raise ValueError("document lengths do not match the packed embedding rows")
    return embeddings, lengths


def write_documents(path: Path, documents: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def build_index(args: argparse.Namespace) -> None:
    destination = args.index.expanduser().resolve()
    analyzer = Analyzer(stemmer=args.stemmer)
    documents = load_documents(args.documents)
    packed, lengths = load_packed_embeddings(args.embeddings, args.document_lengths)
    if len(documents) != len(lengths):
        raise ValueError("document and embedding counts differ")
    representation = Representation(
        encoder=args.encoder,
        encoder_revision=args.encoder_revision,
        dimension=int(packed.shape[1]),
        normalized=True,
        query_template=args.query_template,
        document_template=args.document_template,
    )
    segment = Segment(args.corpus_id, args.corpus_version, [row["id"] for row in documents])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with index_lock(destination, exclusive=True):
        if destination.exists():
            raise FileExistsError(f"index already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            write_documents(temporary / DOCUMENTS_FILE, documents)
            analyzer.write(temporary / ANALYZER_FILE)
            write_lexical_index(
                temporary / LEXICAL_DIRECTORY, [row["text"] for row in documents], analyzer
            )
            STORES[args.storage].create(
                temporary / VECTOR_DIRECTORY,
                segment,
                packed,
                lengths,
                representation,
                threads=args.threads,
            )
            segment.manifest.write(temporary / CORPUS_MANIFEST)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    print(f"built {len(documents):,}-document {args.storage} index at {destination}")


def update_index(args: argparse.Namespace) -> None:
    source = args.index.expanduser().resolve()
    additions = load_documents(args.documents)
    packed, lengths = load_packed_embeddings(args.embeddings, args.document_lengths)
    if len(additions) != len(lengths):
        raise ValueError("new document and embedding counts differ")

    with index_lock(source, exclusive=True):
        if not source.is_dir():
            raise FileNotFoundError(f"index not found: {source}")
        existing = load_documents(source / DOCUMENTS_FILE)
        # Refuses an external ID that already exists before anything is copied.
        segment = read_segment(source).appended([row["id"] for row in additions])
        analyzer = Analyzer.read(source / ANALYZER_FILE)
        replacement = staged_index_copy(source)
        try:
            store = open_vector_store(replacement / VECTOR_DIRECTORY)
            snapshot = store.append([row["id"] for row in additions], packed, lengths, threads=args.threads)
            require_same_segment(snapshot.segment, segment)
            documents = [*existing, *additions]
            write_lexical_index(
                replacement / LEXICAL_DIRECTORY, [row["text"] for row in documents], analyzer
            )
            write_documents(replacement / DOCUMENTS_FILE, documents)
            segment.manifest.write(replacement / CORPUS_MANIFEST)
            del store, snapshot
            gc.collect()
            publish_replacement(source, replacement)
        finally:
            if replacement.exists():
                shutil.rmtree(replacement)
    print(f"appended {len(additions):,} documents to {source}; generation {segment.generation}")


def delete_index(args: argparse.Namespace) -> None:
    source = args.index.expanduser().resolve()
    requested = list(dict.fromkeys(args.document_id))
    with index_lock(source, exclusive=True):
        if not source.is_dir():
            raise FileNotFoundError(f"index not found: {source}")
        documents = load_documents(source / DOCUMENTS_FILE)
        known = {row["id"] for row in documents}
        missing = [item for item in requested if item not in known]
        if missing:
            raise ValueError(f"external document ID not found: {missing[0]}")
        if len(requested) == len(documents):
            raise ValueError("delete cannot remove every document from the index")
        segment = read_segment(source).deleted(requested)
        deleted = set(requested)
        remaining = [row for row in documents if row["id"] not in deleted]
        analyzer = Analyzer.read(source / ANALYZER_FILE)
        replacement = staged_index_copy(source)
        try:
            store = open_vector_store(replacement / VECTOR_DIRECTORY)
            snapshot = store.delete(requested)
            require_same_segment(snapshot.segment, segment)
            # Rebuilding over the survivors compacts internal IDs to 0..n-1 in
            # document order, which is the order the segment compacts to.
            write_lexical_index(
                replacement / LEXICAL_DIRECTORY, [row["text"] for row in remaining], analyzer
            )
            write_documents(replacement / DOCUMENTS_FILE, remaining)
            segment.manifest.write(replacement / CORPUS_MANIFEST)
            del store, snapshot
            gc.collect()
            publish_replacement(source, replacement)
        finally:
            if replacement.exists():
                shutil.rmtree(replacement)
    print(f"deleted {len(requested):,} documents from {source}; generation {segment.generation}")


def search_index(args: argparse.Namespace) -> None:
    source = args.index.expanduser().resolve()
    with index_lock(source, exclusive=False):
        segment = read_segment(source)
        gatherer = LexicalCandidateGenerator.open(
            source / LEXICAL_DIRECTORY, segment, Analyzer.read(source / ANALYZER_FILE)
        )
        reranker = None
        features: dict[str, Feature] = {}
        if args.query_embeddings is not None:
            # The pipeline refuses the store while it holds another segment
            # than the lexical index's.
            store = open_vector_store(source / VECTOR_DIRECTORY)
            reranker = MaxSimReranker([store])
            features["multi_vector"] = Feature(
                store.representation,
                provider=lambda: np.ascontiguousarray(
                    np.load(args.query_embeddings), dtype=np.float32
                ),
            )
        subset = None
        if args.subset_id:
            subset = {segment.corpus_id: np.sort(segment.to_internal(args.subset_id))}
        result = SearchPipeline(gatherer, reranker).search(
            Query(args.query, **features),
            gather_limit=args.gather_limit,
            limit=args.limit,
            subset=subset,
            budget=ResourceBudget(
                max_batch_tokens=args.max_batch_tokens,
                max_documents_per_batch=args.max_documents_per_batch,
                threads=args.threads,
            ),
        )
        output = {
            "results": [
                {
                    "rank": row.rank,
                    "document_id": row.document_id,
                    "external_id": row.external_id,
                    "score": row.score,
                }
                for row in result.documents
            ],
            "timings": {
                "gather_seconds": result.timings.gather_seconds,
                "rerank_seconds": result.timings.rerank_seconds,
                "total_seconds": result.timings.total_seconds,
            },
            "diagnostics": result.diagnostics,
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build BM25 and a vector store")
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--documents", type=Path, required=True)
    build.add_argument("--embeddings", type=Path, required=True)
    build.add_argument("--document-lengths", type=Path, required=True)
    build.add_argument("--storage", choices=tuple(STORES), default="float32")
    build.add_argument("--corpus-id", required=True)
    build.add_argument("--corpus-version", required=True)
    build.add_argument("--encoder", required=True)
    build.add_argument("--encoder-revision", required=True)
    build.add_argument("--query-template", default="")
    build.add_argument("--document-template", default="")
    build.add_argument("--threads", type=int)
    build.add_argument(
        "--stemmer",
        default=None,
        help="Snowball algorithm for the lexical stage (e.g. spanish, english). "
        "Default: none. Persisted with the index and reused by every command.",
    )
    build.set_defaults(function=build_index)

    update = commands.add_parser("update", help="append documents and vectors")
    update.add_argument("--index", type=Path, required=True)
    update.add_argument("--documents", type=Path, required=True)
    update.add_argument("--embeddings", type=Path, required=True)
    update.add_argument("--document-lengths", type=Path, required=True)
    update.add_argument("--threads", type=int)
    update.set_defaults(function=update_index)

    delete = commands.add_parser("delete", help="delete external document IDs")
    delete.add_argument("--index", type=Path, required=True)
    delete.add_argument("--document-id", action="append", required=True)
    delete.set_defaults(function=delete_index)

    search = commands.add_parser("search", help="BM25 gather, MaxSim rerank when embeddings are given")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument(
        "--query-embeddings",
        type=Path,
        help="float32 [query_tokens, dimension] .npy from the index's encoder; omit for gather-only",
    )
    search.add_argument(
        "--subset-id", action="append", help="restrict the search to this external ID (repeatable)"
    )
    search.add_argument("--gather-limit", type=int, default=500)
    search.add_argument("--limit", type=int, default=100)
    search.add_argument("--max-batch-tokens", type=int, default=131_072)
    search.add_argument("--max-documents-per-batch", type=int, default=256)
    search.add_argument("--threads", type=int)
    search.set_defaults(function=search_index)
    return value


def main() -> int:
    args = parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
