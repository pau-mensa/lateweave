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

Two sides share one index directory and nothing else. ``build``, ``upsert``,
``delete``, and ``compact`` are the indexer: they keep ``documents.jsonl``, a
bm25s index, and a lateweave vector store up to date, each committed on its
own. ``search`` is the server: it follows whatever each index last committed,
ranks only the documents both hold, and reports how fresh the answer is.

The lexical stage is bm25s. It consumes only the query text, so it declares no
query features. What a caller can still get wrong is building with one analyzer
and querying with another, which loses terms silently. So the analyzer is
persisted next to the index and read back on every open and every mutation.
bm25s has no incremental append or delete, so each mutation builds the next
lexical generation from ``documents.jsonl`` and publishes it by renaming
``bm25/current.json``.

The rerank stage is lateweave's MaxSim reranker over a lateweave vector store,
which carries the encoder representation it was built from: a search that
supplies token embeddings must supply them from that encoder. A search without
embeddings is gather-only: BM25 scores rank.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterator, Sequence
import unicodedata

import numpy as np

from lateweave import (
    Candidate,
    Feature,
    Gathered,
    MaxSimReranker,
    Query,
    Representation,
    ResourceBudget,
    SearchPipeline,
    Subset,
    VectorStore,
    VectorStoreWriter,
)


ANALYZER_FILE = "analyzer.json"
DOCUMENTS_FILE = "documents.jsonl"
VECTOR_DIRECTORY = "vectors"
LEXICAL_DIRECTORY = "bm25"
CURRENT_FILE = "current.json"
IDS_FILE = "ids.json"

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


# -- the indexer --------------------------------------------------------------


def replace_file(path: Path, text: str) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def publish_lexical_index(
    index: Path, corpus: str, documents: Sequence[dict[str, str]], analyzer: Analyzer
) -> None:
    """Build the next lexical generation and make it current by rename."""
    import bm25s

    directory = index / LEXICAL_DIRECTORY
    current = directory / CURRENT_FILE
    generation = json.loads(current.read_text())["generation"] + 1 if current.exists() else 0
    target = directory / str(generation)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    lexical = bm25s.BM25(method=LEXICAL_METHOD, k1=LEXICAL_K1, b=LEXICAL_B)
    lexical.index([analyzer.tokens(row["text"]) for row in documents], show_progress=False)
    lexical.save(str(target), show_progress=False)
    (target / IDS_FILE).write_text(json.dumps([row["id"] for row in documents]))
    replace_file(
        current,
        json.dumps(
            {
                "corpus": corpus,
                "generation": generation,
                "committed_at": datetime.now(timezone.utc).timestamp(),
            }
        ),
    )
    for older in directory.iterdir():
        if older.is_dir() and older.name != str(generation):
            shutil.rmtree(older, ignore_errors=True)


@contextmanager
def writer_lock(path: Path) -> Iterator[None]:
    """Serializes indexers; searches never take it."""
    lock_path = path.parent / f".{path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_documents(path: Path, *, allow_empty: bool = False) -> list[dict[str, str]]:
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
    if not documents and not allow_empty:
        raise ValueError("document input is empty")
    ids = [row["id"] for row in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("document IDs must be unique")
    return documents


def write_documents(path: Path, documents: Sequence[dict[str, str]]) -> None:
    replace_file(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents))


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


def build_index(args: argparse.Namespace) -> None:
    index = args.index.expanduser().resolve()
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
    analyzer = Analyzer(stemmer=args.stemmer)
    with writer_lock(index):
        if index.exists():
            raise FileExistsError(f"index already exists: {index}")
        index.mkdir(parents=True)
        analyzer.write(index / ANALYZER_FILE)
        write_documents(index / DOCUMENTS_FILE, documents)
        publish_lexical_index(index, args.corpus_id, documents, analyzer)
        writer = VectorStoreWriter.create(
            index / VECTOR_DIRECTORY, args.corpus_id, representation, encoding=args.storage
        )
        writer.append([row["id"] for row in documents], packed, lengths, threads=args.threads)
        writer.commit()
    print(f"built {len(documents):,}-document {args.storage} index at {index}")


def upsert_index(args: argparse.Namespace) -> None:
    index = args.index.expanduser().resolve()
    additions = load_documents(args.documents)
    packed, lengths = load_packed_embeddings(args.embeddings, args.document_lengths)
    if len(additions) != len(lengths):
        raise ValueError("document and embedding counts differ")
    with writer_lock(index):
        replaced = {row["id"] for row in additions}
        existing = load_documents(index / DOCUMENTS_FILE, allow_empty=True)
        documents = [row for row in existing if row["id"] not in replaced] + additions
        writer = VectorStoreWriter(index / VECTOR_DIRECTORY)
        writer.append([row["id"] for row in additions], packed, lengths, threads=args.threads)
        writer.commit()
        write_documents(index / DOCUMENTS_FILE, documents)
        publish_lexical_index(index, writer.corpus, documents, Analyzer.read(index / ANALYZER_FILE))
    print(f"upserted {len(additions):,} documents into {index}")


def delete_index(args: argparse.Namespace) -> None:
    index = args.index.expanduser().resolve()
    deleted = set(args.document_id)
    with writer_lock(index):
        documents = load_documents(index / DOCUMENTS_FILE, allow_empty=True)
        remaining = [row for row in documents if row["id"] not in deleted]
        writer = VectorStoreWriter(index / VECTOR_DIRECTORY)
        count = writer.delete(sorted(deleted))
        writer.commit()
        write_documents(index / DOCUMENTS_FILE, remaining)
        publish_lexical_index(index, writer.corpus, remaining, Analyzer.read(index / ANALYZER_FILE))
    print(f"deleted {count:,} documents from {index}")


def compact_index(args: argparse.Namespace) -> None:
    index = args.index.expanduser().resolve()
    with writer_lock(index):
        writer = VectorStoreWriter(index / VECTOR_DIRECTORY)
        writer.compact()
        writer.commit()
    print(f"compacted the vector store of {index}")


# -- the server ---------------------------------------------------------------


@dataclass(frozen=True)
class LexicalGeneration:
    corpus: str
    generation: int
    committed_at: datetime
    index: Any
    ids: list[str]
    rows: dict[str, int]


class LexicalCandidateGenerator:
    """Cookbook adapter; the lexical index remains external to lateweave.

    Every search reads ``current.json`` and opens a generation only when it
    moved, so what the indexer publishes is served on the next search.
    """

    requires: dict[str, Representation] = {}
    score_semantics = "bm25s-lucene"

    def __init__(self, path: Path, analyzer: Analyzer) -> None:
        self.path = path
        self.analyzer = analyzer
        self.pinned: LexicalGeneration | None = None

    def current(self) -> LexicalGeneration:
        import bm25s

        state = json.loads((self.path / CURRENT_FILE).read_text())
        if self.pinned is None or self.pinned.generation != state["generation"]:
            directory = self.path / str(state["generation"])
            ids = json.loads((directory / IDS_FILE).read_text())
            self.pinned = LexicalGeneration(
                corpus=state["corpus"],
                generation=state["generation"],
                committed_at=datetime.fromtimestamp(state["committed_at"], timezone.utc),
                index=bm25s.BM25.load(str(directory), mmap=True, load_corpus=False, show_progress=False),
                ids=ids,
                rows={item: row for row, item in enumerate(ids)},
            )
        return self.pinned

    def gather(
        self, query: Query, limit: int, *, subset: Subset | None = None
    ) -> Gathered:
        lexical = self.current()
        terms = self.analyzer.tokens(query.text)
        if not terms or not lexical.ids:
            return Gathered([], lexical.committed_at)
        weight_mask = None
        if subset is not None:
            # bm25s multiplies scores by the mask; masked documents score zero
            # and are dropped below with every other non-matching document.
            restriction = subset.restriction(lexical.corpus)
            if restriction is None:
                weight_mask = np.zeros(len(lexical.ids), dtype=np.float32)
            else:
                kind, ids = restriction
                listed = [lexical.rows[item] for item in ids if item in lexical.rows]
                if kind == "only":
                    weight_mask = np.zeros(len(lexical.ids), dtype=np.float32)
                    weight_mask[listed] = 1.0
                else:
                    weight_mask = np.ones(len(lexical.ids), dtype=np.float32)
                    weight_mask[listed] = 0.0
        rows, scores = lexical.index.retrieve(
            [terms], k=min(limit, len(lexical.ids)), show_progress=False, weight_mask=weight_mask
        )
        candidates = []
        for row, raw_score in zip(rows[0].tolist(), scores[0].tolist()):
            score = float(raw_score)
            # Under the Lucene idf a zero score shares no term with the query.
            if score != score or score <= 0.0:
                continue
            candidates.append(Candidate(lexical.corpus, lexical.ids[int(row)], score, len(candidates), "bm25s"))
        return Gathered(candidates, lexical.committed_at)


def search_index(args: argparse.Namespace) -> None:
    index = args.index.expanduser().resolve()
    gatherer = LexicalCandidateGenerator(index / LEXICAL_DIRECTORY, Analyzer.read(index / ANALYZER_FILE))
    reranker = None
    features: dict[str, Feature] = {}
    if args.query_embeddings is not None:
        store = VectorStore(index / VECTOR_DIRECTORY)
        reranker = MaxSimReranker([store])
        features["multi_vector"] = Feature(
            store.representation,
            provider=lambda: np.ascontiguousarray(np.load(args.query_embeddings), dtype=np.float32),
        )
    corpus = gatherer.current().corpus
    result = SearchPipeline(gatherer, reranker).search(
        Query(args.query, **features),
        gather_limit=args.gather_limit,
        limit=args.limit,
        subset=Subset().including(corpus, args.subset_id) if args.subset_id else None,
        budget=ResourceBudget(
            max_batch_tokens=args.max_batch_tokens,
            max_documents_per_batch=args.max_documents_per_batch,
            threads=args.threads,
        ),
    )
    output = {
        "results": [
            {"rank": row.rank, "corpus": row.corpus, "document_id": row.document_id, "score": row.score}
            for row in result.documents
        ],
        "as_of": result.as_of.isoformat(),
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
    build.add_argument("--storage", choices=("float32", "int8"), default="float32")
    build.add_argument("--corpus-id", required=True)
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

    upsert = commands.add_parser("upsert", help="add documents, replacing any with the same ID")
    upsert.add_argument("--index", type=Path, required=True)
    upsert.add_argument("--documents", type=Path, required=True)
    upsert.add_argument("--embeddings", type=Path, required=True)
    upsert.add_argument("--document-lengths", type=Path, required=True)
    upsert.add_argument("--threads", type=int)
    upsert.set_defaults(function=upsert_index)

    delete = commands.add_parser("delete", help="delete document IDs")
    delete.add_argument("--index", type=Path, required=True)
    delete.add_argument("--document-id", action="append", required=True)
    delete.set_defaults(function=delete_index)

    compact = commands.add_parser("compact", help="reclaim the space of deleted and replaced vectors")
    compact.add_argument("--index", type=Path, required=True)
    compact.set_defaults(function=compact_index)

    search = commands.add_parser("search", help="BM25 gather, MaxSim rerank when embeddings are given")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument(
        "--query-embeddings",
        type=Path,
        help="float32 [query_tokens, dimension] .npy from the index's encoder; omit for gather-only",
    )
    search.add_argument(
        "--subset-id", action="append", help="restrict the search to this document ID (repeatable)"
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
