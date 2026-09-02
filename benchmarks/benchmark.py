# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "bm25s==0.3.11",
#   "numpy>=1.26,<3",
#   "scipy>=1.11",
# ]
# ///

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import tarfile
import time
from typing import Iterator, TextIO
import unicodedata

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = SCRIPT_DIRECTORY / ".work"
STORE_NAMES = ("int8", "turboquant4", "jzip")


@dataclass(frozen=True)
class BenchmarkPaths:
    workspace: Path

    @property
    def data(self) -> Path:
        return self.workspace / "data"

    @property
    def results(self) -> Path:
        return self.workspace / "results"

    @property
    def config(self) -> Path:
        return self.workspace / "benchmark-config.json"

    @property
    def bm25(self) -> Path:
        return self.workspace / "bm25"

    def store(self, name: str) -> Path:
        return self.workspace / f"store-{name}"


@dataclass(frozen=True)
class BenchmarkConfig:
    document_count: int
    document_tokens: int
    query_count: int
    query_tokens: int
    dimension: int
    candidates: int
    threads: int
    seed: int
    chunk_tokens: int
    max_batch_tokens: int
    compression_level: int
    source: str


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def write_result(
    paths: BenchmarkPaths, name: str, result: dict[str, object]
) -> None:
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / f"{name}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def save_config(paths: BenchmarkPaths, config: BenchmarkConfig) -> None:
    paths.workspace.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_config(paths: BenchmarkPaths) -> BenchmarkConfig:
    try:
        values = json.loads(paths.config.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"benchmark workspace is not prepared: {paths.workspace}\n"
            "run the prepare stage first"
        ) from error
    return BenchmarkConfig(**values)


def remove_generated(path: Path, workspace: Path) -> None:
    path = path.resolve()
    workspace = workspace.resolve()
    if path.parent != workspace:
        raise RuntimeError(f"refusing to remove path outside the workspace: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_output(path: Path, workspace: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise RuntimeError(f"output already exists: {path}; pass --overwrite to replace it")
    remove_generated(path, workspace)


def clear_prepared_workspace(paths: BenchmarkPaths, overwrite: bool) -> None:
    generated = [
        paths.data,
        paths.results,
        paths.config,
        paths.bm25,
        *(paths.store(name) for name in STORE_NAMES),
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            f"benchmark outputs already exist:\n{formatted}\n"
            "pass --overwrite to replace them"
        )
    for path in existing:
        remove_generated(path, paths.workspace)


def find_summaries_file(directory: Path) -> Path:
    matches = sorted(directory.rglob("summaries.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one summaries.jsonl below {directory}, found "
            f"{len(matches)}"
        )
    return matches[0]


@contextmanager
def open_summaries(source: Path) -> Iterator[TextIO]:
    source = source.expanduser().resolve()
    if source.is_dir():
        with find_summaries_file(source).open(encoding="utf-8") as handle:
            yield handle
        return
    if not source.is_file():
        raise FileNotFoundError(f"corpus source does not exist: {source}")
    if tarfile.is_tarfile(source):
        with tarfile.open(source, mode="r:*") as archive:
            matches = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name == "summaries.jsonl"
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one summaries.jsonl in {source}, found "
                    f"{len(matches)}"
                )
            raw = archive.extractfile(matches[0])
            if raw is None:
                raise RuntimeError(f"could not read {matches[0].name} from {source}")
            with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                yield handle
        return
    with source.open(encoding="utf-8") as handle:
        yield handle


def prepare(
    paths: BenchmarkPaths,
    source: Path,
    document_limit: int | None,
    document_tokens: int,
    query_count: int,
    query_tokens: int,
    dimension: int,
    candidates: int,
    threads: int,
    seed: int,
    chunk_tokens: int,
    max_batch_tokens: int,
    compression_level: int,
    overwrite: bool,
) -> None:
    clear_prepared_workspace(paths, overwrite)
    paths.data.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with open_summaries(source) as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "id": str(item["id"]),
                    "title": str(item["title"]),
                    "text": str(item["summary"]),
                }
            )
            if document_limit is not None and len(rows) == document_limit:
                break
    if not rows:
        raise RuntimeError("the corpus contains no documents")
    if document_limit is not None and len(rows) != document_limit:
        raise RuntimeError(
            f"requested {document_limit} documents, but the corpus contains {len(rows)}"
        )
    if query_count > len(rows):
        raise RuntimeError(
            f"query count ({query_count}) exceeds document count ({len(rows)})"
        )

    config = BenchmarkConfig(
        document_count=len(rows),
        document_tokens=document_tokens,
        query_count=query_count,
        query_tokens=query_tokens,
        dimension=dimension,
        candidates=min(candidates, len(rows)),
        threads=threads,
        seed=seed,
        chunk_tokens=chunk_tokens,
        max_batch_tokens=max_batch_tokens,
        compression_level=compression_level,
        source=str(source.expanduser().resolve()),
    )
    with (paths.data / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected = np.linspace(
        0, config.document_count - 1, config.query_count, dtype=np.int64
    )
    queries = [rows[int(index)]["title"] for index in selected]
    (paths.data / "queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.save(
        paths.data / "document-lengths.npy",
        np.full(config.document_count, config.document_tokens, np.int64),
    )

    random = np.random.default_rng(config.seed)
    total_tokens = config.document_count * config.document_tokens
    embeddings = np.lib.format.open_memmap(
        paths.data / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_tokens, config.dimension),
    )
    generation_chunk_tokens = min(config.chunk_tokens, 8_192)
    for first in range(0, total_tokens, generation_chunk_tokens):
        last = min(total_tokens, first + generation_chunk_tokens)
        chunk = random.standard_normal(
            (last - first, config.dimension), dtype=np.float32
        )
        chunk /= np.linalg.norm(chunk, axis=1, keepdims=True)
        embeddings[first:last] = chunk
    embeddings.flush()
    del embeddings

    query_embeddings = random.standard_normal(
        (config.query_count, config.query_tokens, config.dimension),
        dtype=np.float32,
    )
    query_embeddings /= np.linalg.norm(query_embeddings, axis=2, keepdims=True)
    np.save(paths.data / "query-embeddings.npy", query_embeddings)
    save_config(paths, config)
    write_result(
        paths,
        "prepare",
        {
            **asdict(config),
            "total_document_embeddings": total_tokens,
            "raw_embedding_bytes": total_tokens * config.dimension * 4,
            "peak_rss_mib": peak_rss_mib(),
        },
    )


# The lexical stage matches cookbook/bm25_stored_maxsim.py: casefold, strip
# combining marks, take \w+ runs, no stopwords and no stemming.
_TOKEN = re.compile(r"\w+")


def analyze(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return [sys.intern(token) for token in _TOKEN.findall(folded)]


def open_bm25(path: Path):
    """Reopen a persisted lexical index for searching."""
    import bm25s

    return bm25s.BM25.load(str(path), mmap=True, load_corpus=False, show_progress=False)


def bm25_rows(index, query: str, limit: int) -> list[tuple[int, float]]:
    """Top-``limit`` ``(document, score)`` pairs, matches only."""
    terms = analyze(query)
    if not terms:
        return []
    documents, scores = index.retrieve(
        [terms], k=min(limit, int(index.scores["num_docs"])), show_progress=False
    )
    return [
        (int(document), float(score))
        for document, score in zip(documents[0].tolist(), scores[0].tolist())
        if score > 0.0
    ]


def load_documents(paths: BenchmarkPaths) -> list[dict[str, str]]:
    with (paths.data / "documents.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def benchmark_bm25(
    paths: BenchmarkPaths, config: BenchmarkConfig, overwrite: bool
) -> None:
    import bm25s

    documents = load_documents(paths)
    queries = json.loads((paths.data / "queries.json").read_text(encoding="utf-8"))
    prepare_output(paths.bm25, paths.workspace, overwrite)
    started = time.perf_counter()
    builder = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    builder.index([analyze(row["text"]) for row in documents], show_progress=False)
    paths.bm25.mkdir(parents=True, exist_ok=True)
    builder.save(str(paths.bm25), show_progress=False)
    build_seconds = time.perf_counter() - started
    del builder
    gc.collect()

    # Search the reopened index, which is the shape a deployment runs.
    index = open_bm25(paths.bm25)
    bm25_rows(index, queries[0], config.candidates)
    candidate_ids = np.full(
        (config.query_count, config.candidates), -1, dtype=np.int64
    )
    counts = np.zeros(config.query_count, dtype=np.int64)
    timings_ms: list[float] = []
    for position, query in enumerate(queries):
        started = time.perf_counter()
        rows = bm25_rows(index, query, config.candidates)
        timings_ms.append((time.perf_counter() - started) * 1000)
        count = min(len(rows), config.candidates)
        counts[position] = count
        candidate_ids[position, :count] = [int(row[0]) for row in rows[:count]]
    np.save(paths.data / "candidate-ids.npy", candidate_ids)
    np.save(paths.data / "candidate-counts.npy", counts)
    np.save(
        paths.data / "bm25-latencies-ms.npy",
        np.asarray(timings_ms, dtype=np.float64),
    )
    write_result(
        paths,
        "bm25",
        {
            "build_seconds": build_seconds,
            "documents_per_second": config.document_count / build_seconds,
            "index_bytes": directory_bytes(paths.bm25),
            "candidate_limit": config.candidates,
            "mean_candidates": float(counts.mean()),
            "latency_mean_ms": float(np.mean(timings_ms)),
            "latency_p50_ms": percentile(timings_ms, 50),
            "latency_p95_ms": percentile(timings_ms, 95),
            "throughput_queries_per_second": 1000 / float(np.mean(timings_ms)),
            "peak_rss_mib": peak_rss_mib(),
        },
    )


def store_type(name: str):
    from lateweave import Int8VectorStore, JzipVectorStore, TurboQuantVectorStore

    return {
        "int8": Int8VectorStore,
        "turboquant4": TurboQuantVectorStore,
        "jzip": JzipVectorStore,
    }[name]


def build_store(
    paths: BenchmarkPaths,
    config: BenchmarkConfig,
    name: str,
    overwrite: bool,
) -> None:
    implementation = store_type(name)
    destination = paths.store(name)
    prepare_output(destination, paths.workspace, overwrite)
    embeddings = np.load(paths.data / "embeddings.npy", mmap_mode="r")
    lengths = np.load(paths.data / "document-lengths.npy")
    options: dict[str, object] = {
        "chunk_tokens": config.chunk_tokens,
        "threads": config.threads,
    }
    if name == "jzip":
        options["compression_level"] = config.compression_level
    started = time.perf_counter()
    store = implementation.create(destination, embeddings, lengths, **options)
    build_seconds = time.perf_counter() - started
    footprint = directory_bytes(destination)
    raw_bytes = int(embeddings.nbytes)
    result = {
        "store": name,
        "build_seconds": build_seconds,
        "embedding_vectors_per_second": len(embeddings) / build_seconds,
        "input_mib_per_second": raw_bytes / 2**20 / build_seconds,
        "store_bytes": footprint,
        "store_mib": footprint / 2**20,
        "raw_bytes": raw_bytes,
        "compression_ratio": raw_bytes / footprint,
        "encoded_bytes_per_token": store.encoded_bytes_per_token,
        "peak_rss_mib": peak_rss_mib(),
        "threads": config.threads,
    }
    del store, embeddings, lengths
    gc.collect()
    write_result(paths, f"build-{name}", result)


def manifest_for(store, config: BenchmarkConfig):
    from lateweave import IndexManifest

    return IndexManifest(
        corpus_id="lateweave-benchmark",
        corpus_version="1",
        document_count=config.document_count,
        document_ids_sha256="benchmark",
        encoder="random-normalized",
        encoder_revision=str(config.seed),
        tokenizer=f"fixed-{config.document_tokens}-token",
        dimension=config.dimension,
        dtype="float32",
        normalized=True,
        representation=store.format,
        score_semantics=store.score_semantics,
    )


def run_score_pass(
    scorer,
    query_embeddings,
    candidate_ids,
    counts,
    config: BenchmarkConfig,
) -> list[float]:
    from lateweave import Candidate, Query, ResourceBudget

    budget = ResourceBudget(
        max_batch_tokens=config.max_batch_tokens,
        max_documents_per_batch=config.candidates,
        threads=config.threads,
    )
    timings_ms: list[float] = []
    for position in range(config.query_count):
        ids = candidate_ids[position, : counts[position]]
        candidates = tuple(
            Candidate(int(document_id), 0.0, rank, "bm25s-benchmark")
            for rank, document_id in enumerate(ids)
        )
        query = Query(
            "benchmark",
            embeddings=np.ascontiguousarray(query_embeddings[position]),
        )
        started = time.perf_counter()
        scores = scorer.score(query, candidates, budget=budget)
        timings_ms.append((time.perf_counter() - started) * 1000)
        if len(scores) != len(candidates):
            raise RuntimeError("scorer changed candidate count")
    return timings_ms


def benchmark_search(
    paths: BenchmarkPaths, config: BenchmarkConfig, name: str
) -> None:
    from lateweave import (
        Candidate,
        Query,
        ResourceBudget,
        StoredMaxSimScorer,
        maxsim_scores_packed,
        open_vector_store,
    )

    store = open_vector_store(paths.store(name))
    scorer = StoredMaxSimScorer(store, manifest_for(store, config))
    query_embeddings = np.load(paths.data / "query-embeddings.npy", mmap_mode="r")
    candidate_ids = np.load(paths.data / "candidate-ids.npy", mmap_mode="r")
    counts = np.load(paths.data / "candidate-counts.npy", mmap_mode="r")
    bm25_ms = np.load(paths.data / "bm25-latencies-ms.npy")

    first_pass = run_score_pass(
        scorer, query_embeddings, candidate_ids, counts, config
    )
    second_pass = run_score_pass(
        scorer, query_embeddings, candidate_ids, counts, config
    )
    prepare_ms: list[float] = []
    transition_ms: list[float] = []
    maxsim_ms: list[float] = []
    for position in range(config.query_count):
        ids = sorted(int(item) for item in candidate_ids[position, : counts[position]])
        query = np.ascontiguousarray(query_embeddings[position])
        started = time.perf_counter()
        prepared_query = store.prepare_query(query, threads=config.threads)
        prepare_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        documents, lengths = store.transition(ids, threads=config.threads)
        transition_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        maxsim_scores_packed(
            prepared_query,
            documents,
            lengths,
            max_batch_tokens=config.max_batch_tokens,
            threads=config.threads,
        )
        maxsim_ms.append((time.perf_counter() - started) * 1000)

    end_to_end = bm25_ms + np.asarray(second_pass)
    queries = json.loads((paths.data / "queries.json").read_text(encoding="utf-8"))
    bm25 = open_bm25(paths.bm25)
    budget = ResourceBudget(
        max_batch_tokens=config.max_batch_tokens,
        max_documents_per_batch=config.candidates,
        threads=config.threads,
    )

    def integrated_pass() -> list[float]:
        timings: list[float] = []
        for position, query_text in enumerate(queries):
            started = time.perf_counter()
            rows = bm25_rows(bm25, query_text, config.candidates)
            candidates = tuple(
                Candidate(int(document_id), float(score), rank, "bm25s-benchmark")
                for rank, (document_id, score) in enumerate(rows)
            )
            scorer.score(
                Query(
                    query_text,
                    embeddings=np.ascontiguousarray(query_embeddings[position]),
                ),
                candidates,
                budget=budget,
            )
            timings.append((time.perf_counter() - started) * 1000)
        return timings

    integrated_first_ms = integrated_pass()
    integrated_ms = integrated_pass()
    mean_candidates = float(counts.mean())
    result = {
        "store": name,
        "queries": config.query_count,
        "mean_candidates": mean_candidates,
        "candidate_tokens_per_query_mean": mean_candidates * config.document_tokens,
        "first_pass_latency_mean_ms": float(np.mean(first_pass)),
        "first_pass_latency_p50_ms": percentile(first_pass, 50),
        "first_pass_latency_p95_ms": percentile(first_pass, 95),
        "warm_latency_mean_ms": float(np.mean(second_pass)),
        "warm_latency_p50_ms": percentile(second_pass, 50),
        "warm_latency_p95_ms": percentile(second_pass, 95),
        "warm_throughput_queries_per_second": 1000 / float(np.mean(second_pass)),
        "warm_embedding_vectors_per_second": (
            mean_candidates * config.document_tokens * 1000 / float(np.mean(second_pass))
        ),
        "end_to_end_bm25_plus_maxsim_mean_ms": float(np.mean(end_to_end)),
        "end_to_end_bm25_plus_maxsim_p50_ms": percentile(end_to_end.tolist(), 50),
        "end_to_end_bm25_plus_maxsim_p95_ms": percentile(end_to_end.tolist(), 95),
        "end_to_end_queries_per_second": 1000 / float(np.mean(end_to_end)),
        "integrated_warm_mean_ms": float(np.mean(integrated_ms)),
        "integrated_warm_p50_ms": percentile(integrated_ms, 50),
        "integrated_warm_p95_ms": percentile(integrated_ms, 95),
        "integrated_warm_queries_per_second": 1000 / float(np.mean(integrated_ms)),
        "integrated_first_pass_mean_ms": float(np.mean(integrated_first_ms)),
        "warm_query_prepare_mean_ms": float(np.mean(prepare_ms)),
        "warm_store_transition_mean_ms": float(np.mean(transition_ms)),
        "warm_maxsim_kernel_mean_ms": float(np.mean(maxsim_ms)),
        "peak_rss_mib": peak_rss_mib(),
        "threads": config.threads,
    }
    write_result(paths, f"search-{name}", result)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BM25 candidate retrieval and lateweave vector stores. "
            "Run each stage as a separate process for meaningful peak-RSS results."
        )
    )
    parser.add_argument(
        "mode", choices=("prepare", "bm25", "build", "search", "report")
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"generated data and results directory (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument("--store", choices=STORE_NAMES)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output for the selected stage",
    )
    prepare_group = parser.add_argument_group("prepare stage")
    prepare_group.add_argument(
        "--source",
        type=Path,
        help="summaries.jsonl, a directory containing it, or a tar archive",
    )
    prepare_group.add_argument(
        "--documents",
        type=positive_integer,
        help="limit the corpus; omitted means every document",
    )
    prepare_group.add_argument("--document-tokens", type=positive_integer, default=128)
    prepare_group.add_argument("--query-count", type=positive_integer, default=50)
    prepare_group.add_argument("--query-tokens", type=positive_integer, default=32)
    prepare_group.add_argument("--dimension", type=positive_integer, default=128)
    prepare_group.add_argument("--candidates", type=positive_integer, default=500)
    prepare_group.add_argument(
        "--threads", type=positive_integer, default=min(8, os.cpu_count() or 1)
    )
    prepare_group.add_argument("--seed", type=int, default=20260901)
    prepare_group.add_argument("--chunk-tokens", type=positive_integer, default=65_536)
    prepare_group.add_argument(
        "--max-batch-tokens", type=positive_integer, default=131_072
    )
    prepare_group.add_argument(
        "--compression-level", type=positive_integer, default=1
    )
    args = parser.parse_args()
    if args.mode == "prepare" and args.source is None:
        parser.error("prepare requires --source")
    if args.mode in {"build", "search"} and args.store is None:
        parser.error(f"{args.mode} requires --store")
    return args


def main() -> None:
    args = parse_args()
    paths = BenchmarkPaths(args.workspace.expanduser().resolve())
    if args.mode == "prepare":
        prepare(
            paths,
            source=args.source,
            document_limit=args.documents,
            document_tokens=args.document_tokens,
            query_count=args.query_count,
            query_tokens=args.query_tokens,
            dimension=args.dimension,
            candidates=args.candidates,
            threads=args.threads,
            seed=args.seed,
            chunk_tokens=args.chunk_tokens,
            max_batch_tokens=args.max_batch_tokens,
            compression_level=args.compression_level,
            overwrite=args.overwrite,
        )
    elif args.mode == "report":
        combined: dict[str, object] = {}
        if paths.config.exists():
            combined["benchmark_config"] = asdict(load_config(paths))
        combined.update(
            {
                item.stem: json.loads(item.read_text(encoding="utf-8"))
                for item in sorted(paths.results.glob("*.json"))
            }
        )
        print(json.dumps(combined, indent=2, sort_keys=True))
    else:
        config = load_config(paths)
        if args.mode == "bm25":
            benchmark_bm25(paths, config, args.overwrite)
        elif args.mode == "build":
            build_store(paths, config, args.store, args.overwrite)
        else:
            benchmark_search(paths, config, args.store)


if __name__ == "__main__":
    main()
