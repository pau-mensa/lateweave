# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "bm25s==0.3.11",
#   "numpy>=1.26,<3",
#   "PyStemmer>=2.2,<4",
#   "scipy>=1.11",
# ]
# ///
"""BM25 gathering + optional lateweave vector storage + MaxSim.

Run from the lateweave package directory:

    uv run --with-editable . cookbook/bm25_stored_maxsim.py --help

The lexical stage is bm25s. It owns no analysis chain of its own: queries arrive
as token strings mapped through the vocabulary the index itself persisted, so
nothing inside the index can fall out of step with the postings, and a persisted
index returns exactly what an in-memory one built from the same text returns.

What a caller can still get wrong is building with one analyzer and querying
with another, which loses terms silently. So the analyzer lives in the generator
manifest and is read back from it on every open and every mutation.

bm25s has no incremental append or delete, so ``update`` and ``delete`` rebuild
the lexical index from ``documents.jsonl``, which this recipe maintains anyway.
The vector store keeps its incremental paths. Rebuilding is O(corpus) rather
than O(delta) -- a few seconds for tens of thousands of documents -- and it
makes the internal ID compaction after a delete exact by construction rather
than by agreement between two implementations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
    IndexManifest,
    Int8VectorStore,
    JzipVectorStore,
    Query,
    ResourceBudget,
    SearchPipeline,
    StoredMaxSimScorer,
    TurboQuantVectorStore,
    document_ids_digest,
    open_vector_store,
)


GENERATOR_MANIFEST = "generator-manifest.json"
SCORER_MANIFEST = "scorer-manifest.json"
DOCUMENTS_FILE = "documents.jsonl"
VECTOR_DIRECTORY = "vectors"
LEXICAL_DIRECTORY = "bm25"

# Lucene's defaults, spelled out so a bm25s default change cannot move results.
LEXICAL_METHOD = "lucene"
LEXICAL_K1 = 1.5
LEXICAL_B = 0.75
#: Name recorded in the generator manifest for the tokenizer below.
LEXICAL_TOKENIZER = "unicode-fold"

_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class Analyzer:
    """Text to terms, identically for documents and queries.

    ``stemmer`` is a Snowball algorithm name (``"spanish"``, ``"portuguese"``,
    ``"english"``, ...) or ``None``. None is the default because it is
    language-agnostic: on a Spanish corpus it reached the same gold recall at
    the gather depth as a stemmed chain, and a corpus whose language varies per
    document has no single right answer.
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

    def build_parameters(self) -> dict[str, Any]:
        """The analysis chain, as it is recorded in the generator manifest."""
        return {
            "method": LEXICAL_METHOD,
            "tokenizer": LEXICAL_TOKENIZER,
            "stemmer": self.stemmer,
            "stopwords": None,
            "k1": LEXICAL_K1,
            "b": LEXICAL_B,
        }

    @classmethod
    def from_manifest(cls, manifest: IndexManifest) -> "Analyzer":
        """Recover the chain an index was built with, refusing an unknown one.

        Querying through a different chain than the postings were built with
        loses terms silently, so it is refused rather than tolerated.
        """
        parameters = manifest.build_parameters or {}
        tokenizer = parameters.get("tokenizer")
        if tokenizer != LEXICAL_TOKENIZER:
            raise ValueError(
                f"index was built with tokenizer {tokenizer!r}, which this "
                f"recipe cannot reproduce; expected {LEXICAL_TOKENIZER!r}"
            )
        return cls(stemmer=parameters.get("stemmer"))


def write_lexical_index(
    path: Path, texts: Sequence[str], analyzer: Analyzer
) -> None:
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
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source.name}.mutation.", dir=source.parent)
    )
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


def mutated_manifest(
    manifest: IndexManifest,
    documents: Sequence[dict[str, str]],
    *,
    operation: str,
    affected: int,
) -> IndexManifest:
    parameters = dict(manifest.build_parameters or {})
    history = list(parameters.get("cookbook_mutations", []))
    history.append(
        {
            "generation": manifest.generation + 1,
            "operation": operation,
            "affected_documents": affected,
        }
    )
    parameters["cookbook_mutations"] = history
    return replace(
        manifest,
        document_count=len(documents),
        document_ids_sha256=document_ids_digest([row["id"] for row in documents]),
        generation=manifest.generation + 1,
        build_parameters=parameters,
    )


class LexicalCandidateGenerator:
    """Cookbook adapter; the lexical index remains external to lateweave."""

    def __init__(
        self, index: Any, manifest: IndexManifest, analyzer: Analyzer
    ) -> None:
        self.index = index
        self.manifest = manifest
        self.analyzer = analyzer

    @classmethod
    def open(
        cls, path: Path, manifest: IndexManifest
    ) -> "LexicalCandidateGenerator":
        """Open the index, analyzing queries the way its postings were built."""
        import bm25s

        analyzer = Analyzer.from_manifest(manifest)
        index = bm25s.BM25.load(
            str(path), mmap=True, load_corpus=False, show_progress=False
        )
        stored = int(index.scores["num_docs"])
        if stored != manifest.document_count:
            raise RuntimeError(
                f"lexical index holds {stored:,} documents but the manifest "
                f"declares {manifest.document_count:,}"
            )
        return cls(index, manifest, analyzer)

    def gather(self, query: Query, limit: int) -> tuple[Candidate, ...]:
        terms = self.analyzer.tokens(query.text)
        if not terms:
            return ()
        # bm25s raises when the limit exceeds the corpus; it never pads.
        documents, scores = self.index.retrieve(
            [terms],
            k=min(limit, self.manifest.document_count),
            show_progress=False,
        )
        candidates = []
        seen: set[int] = set()
        for raw_document_id, raw_score in zip(
            documents[0].tolist(), scores[0].tolist()
        ):
            score = float(raw_score)
            # Under the Lucene idf a zero score shares no term with the query.
            if score != score or score <= 0.0:
                continue
            document_id = int(raw_document_id)
            if not 0 <= document_id < self.manifest.document_count:
                raise RuntimeError(f"bm25s returned out-of-range ID {document_id}")
            if document_id in seen:
                raise RuntimeError(f"bm25s returned duplicate ID {document_id}")
            seen.add(document_id)
            candidates.append(
                Candidate(document_id, score, len(candidates), "bm25s")
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
                document_id = str(row["id"])
                text = str(row["text"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid document at {path}:{line_number}") from error
            documents.append({"id": document_id, "text": text})
    if not documents:
        raise ValueError("document input is empty")
    ids = [row["id"] for row in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("external document IDs must be unique")
    return documents


def load_packed_embeddings(
    embeddings_path: Path, lengths_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    lengths = np.asarray(np.load(lengths_path), dtype=np.int64)
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        raise ValueError("document embeddings must be a float32 [tokens, dimension] array")
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("document lengths must be a positive int64 vector")
    if int(lengths.sum()) != len(embeddings):
        raise ValueError("document lengths do not match the packed embedding rows")
    for start in range(0, len(embeddings), 1_000_000):
        chunk = embeddings[start : start + 1_000_000]
        if not np.isfinite(chunk).all() or not np.allclose(
            np.linalg.norm(chunk, axis=1), 1.0, rtol=1e-3, atol=1e-4
        ):
            raise ValueError("document embeddings must be finite unit vectors")
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
    stores = {
        "int8": Int8VectorStore,
        "turboquant4": TurboQuantVectorStore,
        "jzip": JzipVectorStore,
    }
    store_type = stores[args.storage]

    common = IndexManifest(
        corpus_id=args.corpus_id,
        corpus_version=args.corpus_version,
        document_count=len(documents),
        document_ids_sha256=document_ids_digest([row["id"] for row in documents]),
        encoder=args.encoder,
        encoder_revision=args.encoder_revision,
        tokenizer=args.tokenizer,
        dimension=int(packed.shape[1]),
        dtype="float32",
        normalized=True,
        query_template=args.query_template,
        document_template=args.document_template,
    )
    generator_manifest = replace(
        common,
        representation="bm25s-sparse-index",
        score_semantics="bm25s-lucene",
        build_parameters=analyzer.build_parameters(),
    )
    scorer_manifest = replace(
        common,
        representation=store_type.format,
        score_semantics=store_type.score_semantics,
        build_parameters={
            "storage": args.storage,
            "chunk_tokens": args.chunk_tokens,
            **(
                {"compression_level": args.compression_level}
                if args.storage == "jzip"
                else {}
            ),
            "device": "cpu",
        },
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with index_lock(destination, exclusive=True):
        if destination.exists():
            raise FileExistsError(f"index already exists: {destination}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            write_documents(temporary / DOCUMENTS_FILE, documents)
            write_lexical_index(
                temporary / LEXICAL_DIRECTORY,
                [row["text"] for row in documents],
                analyzer,
            )
            store_options = {
                "chunk_tokens": args.chunk_tokens,
                "threads": args.threads,
            }
            if args.storage == "jzip":
                store_options["compression_level"] = args.compression_level
            store_type.create(
                temporary / VECTOR_DIRECTORY, packed, lengths, **store_options
            )
            generator_manifest.write(temporary / GENERATOR_MANIFEST)
            scorer_manifest.write(temporary / SCORER_MANIFEST)
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
        existing_ids = {row["id"] for row in existing}
        collisions = [row["id"] for row in additions if row["id"] in existing_ids]
        if collisions:
            raise ValueError(f"external document ID already exists: {collisions[0]}")
        generator_manifest = IndexManifest.read(source / GENERATOR_MANIFEST)
        scorer_manifest = IndexManifest.read(source / SCORER_MANIFEST)
        generator_manifest.assert_compatible(scorer_manifest)
        analyzer = Analyzer.from_manifest(generator_manifest)
        replacement = staged_index_copy(source)
        try:
            store = open_vector_store(replacement / VECTOR_DIRECTORY)
            store.append(
                packed,
                lengths,
                chunk_tokens=args.chunk_tokens,
                copy_chunk_tokens=args.copy_chunk_tokens,
                threads=args.threads,
            )
            updated_documents = [*existing, *additions]
            # bm25s has no incremental append, so the lexical index is rebuilt
            # over the whole corpus. The vector store still appends in place.
            write_lexical_index(
                replacement / LEXICAL_DIRECTORY,
                [row["text"] for row in updated_documents],
                analyzer,
            )
            write_documents(replacement / DOCUMENTS_FILE, updated_documents)
            mutated_manifest(
                generator_manifest,
                updated_documents,
                operation="append",
                affected=len(additions),
            ).write(replacement / GENERATOR_MANIFEST)
            mutated_manifest(
                scorer_manifest,
                updated_documents,
                operation="append",
                affected=len(additions),
            ).write(replacement / SCORER_MANIFEST)
            del store
            gc.collect()
            publish_replacement(source, replacement)
        finally:
            if replacement.exists():
                shutil.rmtree(replacement)
    print(
        f"appended {len(additions):,} documents to {source}; "
        f"generation {generator_manifest.generation + 1}"
    )


def delete_index(args: argparse.Namespace) -> None:
    source = args.index.expanduser().resolve()
    requested = list(dict.fromkeys(args.document_id))
    with index_lock(source, exclusive=True):
        if not source.is_dir():
            raise FileNotFoundError(f"index not found: {source}")
        documents = load_documents(source / DOCUMENTS_FILE)
        internal_by_external = {
            document["id"]: internal for internal, document in enumerate(documents)
        }
        missing = [item for item in requested if item not in internal_by_external]
        if missing:
            raise ValueError(f"external document ID not found: {missing[0]}")
        if len(requested) == len(documents):
            raise ValueError("delete cannot remove every document from the index")
        internal_ids = sorted(internal_by_external[item] for item in requested)
        deleted = set(internal_ids)
        remaining = [
            document for internal, document in enumerate(documents) if internal not in deleted
        ]
        generator_manifest = IndexManifest.read(source / GENERATOR_MANIFEST)
        scorer_manifest = IndexManifest.read(source / SCORER_MANIFEST)
        generator_manifest.assert_compatible(scorer_manifest)
        analyzer = Analyzer.from_manifest(generator_manifest)
        replacement = staged_index_copy(source)
        try:
            store = open_vector_store(replacement / VECTOR_DIRECTORY)
            store.delete(internal_ids, copy_chunk_tokens=args.copy_chunk_tokens)
            # Rebuilding over the survivors compacts internal IDs to 0..n-1 in
            # document order, which is the order the store compacts to as well.
            write_lexical_index(
                replacement / LEXICAL_DIRECTORY,
                [row["text"] for row in remaining],
                analyzer,
            )
            write_documents(replacement / DOCUMENTS_FILE, remaining)
            mutated_manifest(
                generator_manifest,
                remaining,
                operation="delete",
                affected=len(internal_ids),
            ).write(replacement / GENERATOR_MANIFEST)
            mutated_manifest(
                scorer_manifest,
                remaining,
                operation="delete",
                affected=len(internal_ids),
            ).write(replacement / SCORER_MANIFEST)
            del store
            gc.collect()
            publish_replacement(source, replacement)
        finally:
            if replacement.exists():
                shutil.rmtree(replacement)
    print(
        f"deleted {len(internal_ids):,} documents from {source}; "
        f"generation {generator_manifest.generation + 1}"
    )


def search_index(args: argparse.Namespace) -> None:
    source = args.index.expanduser().resolve()
    with index_lock(source, exclusive=False):
        documents = load_documents(source / DOCUMENTS_FILE)
        generator_manifest = IndexManifest.read(source / GENERATOR_MANIFEST)
        scorer_manifest = IndexManifest.read(source / SCORER_MANIFEST)
        generator = LexicalCandidateGenerator.open(
            source / LEXICAL_DIRECTORY, generator_manifest
        )
        scorer = StoredMaxSimScorer(
            open_vector_store(source / VECTOR_DIRECTORY), scorer_manifest
        )
        query_embeddings = np.ascontiguousarray(
            np.load(args.query_embeddings), dtype=np.float32
        )
        budget = ResourceBudget(
            max_memory_bytes=(
                int(args.max_memory_gb * 2**30)
                if args.max_memory_gb is not None
                else None
            ),
            max_batch_tokens=args.max_batch_tokens,
            max_documents_per_batch=args.max_documents_per_batch,
            threads=args.threads,
        )
        result = SearchPipeline(generator, scorer).search(
            Query(args.query, embeddings=query_embeddings),
            gather_limit=args.gather_limit,
            limit=args.limit,
            budget=budget,
        )
        output = {
            "results": [
                {
                    "rank": row.rank,
                    "document_id": row.document_id,
                    "external_id": documents[row.document_id]["id"],
                    "score": row.score,
                }
                for row in result.documents
            ],
            "timings": {
                "gather_seconds": result.timings.gather_seconds,
                "score_seconds": result.timings.score_seconds,
                "total_seconds": result.timings.total_seconds,
            },
            "diagnostics": result.diagnostics,
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _add_mutation_execution_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--chunk-tokens", type=int, default=131_072)
    command.add_argument("--copy-chunk-tokens", type=int, default=1_000_000)
    command.add_argument("--threads", type=int)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build BM25 and a vector store")
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--documents", type=Path, required=True)
    build.add_argument("--embeddings", type=Path, required=True)
    build.add_argument("--document-lengths", type=Path, required=True)
    build.add_argument(
        "--storage", choices=("int8", "turboquant4", "jzip"), default="int8"
    )
    build.add_argument("--compression-level", type=int, choices=range(1, 23), default=1)
    build.add_argument("--corpus-id", required=True)
    build.add_argument("--corpus-version", required=True)
    build.add_argument("--encoder", required=True)
    build.add_argument("--encoder-revision", required=True)
    build.add_argument("--tokenizer", required=True)
    build.add_argument("--query-template", default="")
    build.add_argument("--document-template", default="")
    build.add_argument("--chunk-tokens", type=int, default=131_072)
    build.add_argument("--threads", type=int)
    build.add_argument(
        "--stemmer",
        default=None,
        help="Snowball algorithm for the lexical stage (e.g. spanish, "
             "portuguese). Default: none, which is language-agnostic. Recorded "
             "in the generator manifest and reused by update, delete and "
             "search.",
    )
    build.set_defaults(function=build_index)

    update = commands.add_parser("update", help="append documents and vectors")
    update.add_argument("--index", type=Path, required=True)
    update.add_argument("--documents", type=Path, required=True)
    update.add_argument("--embeddings", type=Path, required=True)
    update.add_argument("--document-lengths", type=Path, required=True)
    _add_mutation_execution_arguments(update)
    update.set_defaults(function=update_index)

    delete = commands.add_parser("delete", help="delete external document IDs")
    delete.add_argument("--index", type=Path, required=True)
    delete.add_argument("--document-id", action="append", required=True)
    delete.add_argument("--copy-chunk-tokens", type=int, default=1_000_000)
    delete.set_defaults(function=delete_index)

    search = commands.add_parser("search", help="BM25 gather then stored MaxSim")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--query-embeddings", type=Path, required=True)
    search.add_argument("--gather-limit", type=int, default=500)
    search.add_argument("--limit", type=int, default=100)
    search.add_argument("--max-batch-tokens", type=int, default=131_072)
    search.add_argument("--max-documents-per-batch", type=int, default=256)
    search.add_argument("--max-memory-gb", type=float)
    search.add_argument("--threads", type=int)
    search.set_defaults(function=search_index)
    return value


def main() -> int:
    args = parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
