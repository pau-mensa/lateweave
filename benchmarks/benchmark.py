from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np


ROOT = Path("/private/tmp/lateweave-bench.WMhKh7")
SOURCE = (
    ROOT
    / "artifacts/law-summaries/runs/legalize-es-luna/summaries.jsonl"
)
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOCUMENTS = 12_289
DOCUMENT_TOKENS = 128
QUERY_COUNT = 50
QUERY_TOKENS = 32
DIMENSION = 128
CANDIDATES = 500
THREADS = 8


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def write_result(name: str, result: dict[str, object]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            rows.append(
                {
                    "id": str(source["id"]),
                    "title": str(source["title"]),
                    "text": str(source["summary"]),
                }
            )
            if len(rows) == DOCUMENTS:
                break
    if len(rows) != DOCUMENTS:
        raise RuntimeError(f"expected {DOCUMENTS} documents, got {len(rows)}")
    with (DATA / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected = np.linspace(0, DOCUMENTS - 1, QUERY_COUNT, dtype=np.int64)
    queries = [rows[int(index)]["title"] for index in selected]
    (DATA / "queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.save(DATA / "document-lengths.npy", np.full(DOCUMENTS, DOCUMENT_TOKENS, np.int64))

    random = np.random.default_rng(20260901)
    total_tokens = DOCUMENTS * DOCUMENT_TOKENS
    embeddings = np.lib.format.open_memmap(
        DATA / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_tokens, DIMENSION),
    )
    for first in range(0, total_tokens, 8192):
        last = min(total_tokens, first + 8192)
        chunk = random.standard_normal((last - first, DIMENSION), dtype=np.float32)
        chunk /= np.linalg.norm(chunk, axis=1, keepdims=True)
        embeddings[first:last] = chunk
    embeddings.flush()
    del embeddings

    query_embeddings = random.standard_normal(
        (QUERY_COUNT, QUERY_TOKENS, DIMENSION), dtype=np.float32
    )
    query_embeddings /= np.linalg.norm(query_embeddings, axis=2, keepdims=True)
    np.save(DATA / "query-embeddings.npy", query_embeddings)
    write_result(
        "prepare",
        {
            "documents": DOCUMENTS,
            "document_tokens": DOCUMENT_TOKENS,
            "total_document_embeddings": total_tokens,
            "query_count": QUERY_COUNT,
            "query_tokens": QUERY_TOKENS,
            "dimension": DIMENSION,
            "raw_embedding_bytes": total_tokens * DIMENSION * 4,
            "peak_rss_mib": peak_rss_mib(),
        },
    )


def load_documents() -> list[dict[str, str]]:
    with (DATA / "documents.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def benchmark_bm25() -> None:
    from bm25x import BM25

    documents = load_documents()
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    path = ROOT / "bm25x"
    if path.exists():
        shutil.rmtree(path)
    started = time.perf_counter()
    index = BM25(
        index=str(path),
        method="lucene",
        tokenizer="unicode",
        use_stopwords=False,
        cuda=False,
    )
    index.add([row["text"] for row in documents])
    build_seconds = time.perf_counter() - started

    index.search([queries[0]], k=CANDIDATES)
    candidate_ids = np.full((QUERY_COUNT, CANDIDATES), -1, dtype=np.int64)
    counts = np.zeros(QUERY_COUNT, dtype=np.int64)
    timings_ms: list[float] = []
    for position, query in enumerate(queries):
        started = time.perf_counter()
        rows = index.search([query], k=CANDIDATES)[0]
        timings_ms.append((time.perf_counter() - started) * 1000)
        counts[position] = len(rows)
        candidate_ids[position, : len(rows)] = [int(row[0]) for row in rows]
    np.save(DATA / "candidate-ids.npy", candidate_ids)
    np.save(DATA / "candidate-counts.npy", counts)
    np.save(DATA / "bm25-latencies-ms.npy", np.asarray(timings_ms, dtype=np.float64))
    write_result(
        "bm25",
        {
            "build_seconds": build_seconds,
            "documents_per_second": DOCUMENTS / build_seconds,
            "index_bytes": directory_bytes(path),
            "candidate_limit": CANDIDATES,
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


def build_store(name: str) -> None:
    implementation = store_type(name)
    destination = ROOT / f"store-{name}"
    if destination.exists():
        shutil.rmtree(destination)
    embeddings = np.load(DATA / "embeddings.npy", mmap_mode="r")
    lengths = np.load(DATA / "document-lengths.npy")
    options: dict[str, object] = {
        "chunk_tokens": 65_536,
        "threads": THREADS,
    }
    if name == "jzip":
        options["compression_level"] = 1
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
        "threads": THREADS,
    }
    del store, embeddings, lengths
    gc.collect()
    write_result(f"build-{name}", result)


def manifest_for(store):
    from lateweave import IndexManifest

    return IndexManifest(
        corpus_id="legalize-es-luna-benchmark",
        corpus_version="2026-09-01",
        document_count=DOCUMENTS,
        document_ids_sha256="benchmark",
        encoder="random-normalized",
        encoder_revision="20260901",
        tokenizer="fixed-128-token",
        dimension=DIMENSION,
        dtype="float32",
        normalized=True,
        representation=store.format,
        score_semantics=store.score_semantics,
    )


def run_score_pass(scorer, query_embeddings, candidate_ids, counts):
    from lateweave import Candidate, Query, ResourceBudget

    budget = ResourceBudget(
        max_batch_tokens=131_072,
        max_documents_per_batch=CANDIDATES,
        threads=THREADS,
    )
    timings_ms: list[float] = []
    for position in range(QUERY_COUNT):
        ids = candidate_ids[position, : counts[position]]
        candidates = tuple(
            Candidate(int(document_id), 0.0, rank, "bm25x-benchmark")
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


def benchmark_search(name: str) -> None:
    from bm25x import BM25
    from lateweave import (
        Candidate,
        Query,
        ResourceBudget,
        StoredMaxSimScorer,
        maxsim_scores_packed,
        open_vector_store,
    )

    store = open_vector_store(ROOT / f"store-{name}")
    scorer = StoredMaxSimScorer(store, manifest_for(store))
    query_embeddings = np.load(DATA / "query-embeddings.npy", mmap_mode="r")
    candidate_ids = np.load(DATA / "candidate-ids.npy", mmap_mode="r")
    counts = np.load(DATA / "candidate-counts.npy", mmap_mode="r")
    bm25_ms = np.load(DATA / "bm25-latencies-ms.npy")

    first_pass = run_score_pass(scorer, query_embeddings, candidate_ids, counts)
    second_pass = run_score_pass(scorer, query_embeddings, candidate_ids, counts)
    prepare_ms: list[float] = []
    transition_ms: list[float] = []
    maxsim_ms: list[float] = []
    for position in range(QUERY_COUNT):
        ids = sorted(int(item) for item in candidate_ids[position, : counts[position]])
        query = np.ascontiguousarray(query_embeddings[position])
        started = time.perf_counter()
        prepared_query = store.prepare_query(query, threads=THREADS)
        prepare_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        documents, lengths = store.transition(ids, threads=THREADS)
        transition_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        maxsim_scores_packed(
            prepared_query,
            documents,
            lengths,
            max_batch_tokens=131_072,
            threads=THREADS,
        )
        maxsim_ms.append((time.perf_counter() - started) * 1000)
    end_to_end = bm25_ms + np.asarray(second_pass)
    documents = load_documents()
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    bm25 = BM25(
        index=str(ROOT / "bm25x"),
        method="lucene",
        tokenizer="unicode",
        use_stopwords=False,
        cuda=False,
    )
    budget = ResourceBudget(
        max_batch_tokens=131_072,
        max_documents_per_batch=CANDIDATES,
        threads=THREADS,
    )
    def integrated_pass() -> list[float]:
        timings: list[float] = []
        for position, query_text in enumerate(queries):
            started = time.perf_counter()
            rows = bm25.search([query_text], k=CANDIDATES)[0]
            candidates = tuple(
                Candidate(int(document_id), float(score), rank, "bm25x-benchmark")
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
    result = {
        "store": name,
        "queries": QUERY_COUNT,
        "mean_candidates": float(counts.mean()),
        "candidate_tokens_per_query_mean": float(counts.mean() * DOCUMENT_TOKENS),
        "first_pass_latency_mean_ms": float(np.mean(first_pass)),
        "first_pass_latency_p50_ms": percentile(first_pass, 50),
        "first_pass_latency_p95_ms": percentile(first_pass, 95),
        "warm_latency_mean_ms": float(np.mean(second_pass)),
        "warm_latency_p50_ms": percentile(second_pass, 50),
        "warm_latency_p95_ms": percentile(second_pass, 95),
        "warm_throughput_queries_per_second": 1000 / float(np.mean(second_pass)),
        "warm_embedding_vectors_per_second": (
            float(counts.mean() * DOCUMENT_TOKENS) * 1000 / float(np.mean(second_pass))
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
        "threads": THREADS,
    }
    write_result(f"search-{name}", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("prepare", "bm25", "build", "search", "report")
    )
    parser.add_argument("--store", choices=("int8", "turboquant4", "jzip"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "bm25":
        benchmark_bm25()
    elif args.mode == "build":
        if args.store is None:
            parser.error("build requires --store")
        build_store(args.store)
    elif args.mode == "search":
        if args.store is None:
            parser.error("search requires --store")
        benchmark_search(args.store)
    else:
        combined = {
            item.stem: json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(RESULTS.glob("*.json"))
        }
        print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
