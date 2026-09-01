# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy>=1.26,<3",
# ]
# ///

"""Compare the candidate MaxSim kernels under serial and concurrent load.

The parent process starts one child per Rayon thread count. This is necessary
because Rayon's global pool is initialized once per process. BLAS remains
single-threaded so the experiment varies only MaxSim scheduling.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any


DEFAULT_THREAD_COUNTS = (1, 2)
DEFAULT_CONCURRENCY = (1, 2, 4)
DEFAULT_FIXED_DOCUMENT_TOKENS = (32, 128, 512)
BLAS_THREAD_ENVIRONMENT = (
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare old tiled-fused MaxSim, optimized packed MaxSim, and "
            "lateweave packed MaxSim in isolated thread configurations."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("maxsim-comparison.json"))
    parser.add_argument(
        "--thread-counts",
        type=positive_integer,
        nargs="+",
        default=DEFAULT_THREAD_COUNTS,
        help="Rayon worker counts, each measured in a fresh process",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_integer,
        nargs="+",
        default=DEFAULT_CONCURRENCY,
        help="closed-loop simultaneous query counts",
    )
    parser.add_argument(
        "--fixed-document-tokens",
        type=positive_integer,
        nargs="+",
        default=DEFAULT_FIXED_DOCUMENT_TOKENS,
    )
    parser.add_argument("--documents", type=positive_integer, default=500)
    parser.add_argument("--query-tokens", type=positive_integer, default=32)
    parser.add_argument("--dimension", type=positive_integer, default=128)
    parser.add_argument("--variable-min-tokens", type=positive_integer, default=32)
    parser.add_argument("--variable-max-tokens", type=positive_integer, default=256)
    parser.add_argument(
        "--concurrency-document-tokens", type=positive_integer, default=128
    )
    parser.add_argument("--max-batch-tokens", type=positive_integer, default=8_192)
    parser.add_argument("--warmups", type=positive_integer, default=4)
    parser.add_argument("--iterations", type=positive_integer, default=24)
    parser.add_argument("--requests-per-worker", type=positive_integer, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-threads", type=positive_integer, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.variable_min_tokens > args.variable_max_tokens:
        parser.error("--variable-min-tokens cannot exceed --variable-max-tokens")
    if args.worker and args.worker_threads is None:
        parser.error("internal worker mode requires --worker-threads")
    return args


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def cpu_model() -> str:
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def parent_metadata() -> dict[str, object]:
    return {
        "architecture": platform.machine(),
        "cpu_model": cpu_model(),
        "available_logical_cpus": available_cpus(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def worker_arguments(args: argparse.Namespace, threads: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-threads",
        str(threads),
        "--documents",
        str(args.documents),
        "--query-tokens",
        str(args.query_tokens),
        "--dimension",
        str(args.dimension),
        "--variable-min-tokens",
        str(args.variable_min_tokens),
        "--variable-max-tokens",
        str(args.variable_max_tokens),
        "--concurrency-document-tokens",
        str(args.concurrency_document_tokens),
        "--max-batch-tokens",
        str(args.max_batch_tokens),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--requests-per-worker",
        str(args.requests_per_worker),
        "--seed",
        str(args.seed),
        "--fixed-document-tokens",
        *(str(value) for value in args.fixed_document_tokens),
        "--concurrency",
        *(str(value) for value in args.concurrency),
    ]
    return command


def run_parent(args: argparse.Namespace) -> None:
    results: dict[str, object] = {
        "environment": parent_metadata(),
        "configuration": {
            "documents": args.documents,
            "query_tokens": args.query_tokens,
            "dimension": args.dimension,
            "fixed_document_tokens": args.fixed_document_tokens,
            "variable_document_tokens": [
                args.variable_min_tokens,
                args.variable_max_tokens,
            ],
            "concurrency_document_tokens": args.concurrency_document_tokens,
            "thread_counts": args.thread_counts,
            "concurrency": args.concurrency,
            "max_batch_tokens": args.max_batch_tokens,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "requests_per_worker": args.requests_per_worker,
            "seed": args.seed,
        },
        "runs": {},
    }
    for threads in args.thread_counts:
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(threads)
        for variable in BLAS_THREAD_ENVIRONMENT:
            environment[variable] = "1"
        completed = subprocess.run(
            worker_arguments(args, threads),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{threads}-thread benchmark worker failed:\n{completed.stderr}"
            )
        results["runs"][str(threads)] = json.loads(completed.stdout)

    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nwrote {destination}", file=sys.stderr)


def import_implementations():
    try:
        import lateweave
        import lateweave_maxsim_bench as comparison_kernels
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "the comparison requires editable installs of lateweave and its "
            "benchmark-only kernel crate; see benchmarks/README.md"
        ) from error
    return np, lateweave, comparison_kernels


def normalized(np, generator, shape: tuple[int, ...]):
    values = generator.standard_normal(shape, dtype=np.float32)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    return np.ascontiguousarray(values)


def timing_summary(np, values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "minimum_ms": float(array.min()),
        "queries_per_second": 1000.0 / float(array.mean()),
        "coefficient_of_variation": float(array.std() / array.mean()),
    }


def benchmark_interleaved(
    np,
    methods: dict[str, Callable[[], Any]],
    warmups: int,
    iterations: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    outputs: dict[str, Any] = {}
    for _ in range(warmups):
        for name, method in methods.items():
            outputs[name] = np.asarray(method())

    timings: dict[str, list[float]] = {name: [] for name in methods}
    order = list(methods)
    order_random = random.Random(seed)
    gc.disable()
    try:
        for _ in range(iterations):
            order_random.shuffle(order)
            for name in order:
                started = time.perf_counter_ns()
                outputs[name] = np.asarray(methods[name]())
                timings[name].append((time.perf_counter_ns() - started) / 1_000_000)
    finally:
        gc.enable()
    return (
        {name: timing_summary(np, values) for name, values in timings.items()},
        outputs,
    )


def correctness(np, outputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    reference = outputs["lateweave_packed"]
    return {
        name: {
            "maximum_absolute_error": float(np.max(np.abs(values - reference))),
            "mean_absolute_error": float(np.mean(np.abs(values - reference))),
        }
        for name, values in outputs.items()
    }


def fixed_case(
    np,
    lateweave,
    comparison_kernels,
    args: argparse.Namespace,
    document_tokens: int,
    seed: int,
) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    query = normalized(np, generator, (args.query_tokens, args.dimension))
    documents = normalized(
        np, generator, (args.documents, document_tokens, args.dimension)
    )
    packed = documents.reshape(-1, args.dimension)
    lengths = np.full(args.documents, document_tokens, dtype=np.int64)
    methods = {
        "old_fused_tiles": lambda: comparison_kernels.fused_scores(query, documents),
        "optimized_packed": lambda: comparison_kernels.packed_scores(
            query, packed, lengths
        ),
        "lateweave_packed": lambda: lateweave.maxsim_scores_packed(
            query,
            packed,
            lengths,
            max_batch_tokens=args.max_batch_tokens,
        ),
    }
    timings, outputs = benchmark_interleaved(
        np, methods, args.warmups, args.iterations, seed
    )
    return {
        "document_count": args.documents,
        "document_tokens": document_tokens,
        "total_document_tokens": args.documents * document_tokens,
        "timings": timings,
        "correctness_vs_lateweave": correctness(np, outputs),
        "optimized_packed_speedup_over_old_fused": (
            timings["old_fused_tiles"]["mean_ms"]
            / timings["optimized_packed"]["mean_ms"]
        ),
    }


def variable_case(
    np,
    lateweave,
    comparison_kernels,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    query = normalized(np, generator, (args.query_tokens, args.dimension))
    lengths = generator.integers(
        args.variable_min_tokens,
        args.variable_max_tokens + 1,
        size=args.documents,
        dtype=np.int64,
    )
    packed = normalized(np, generator, (int(lengths.sum()), args.dimension))
    methods = {
        "old_variable_fused_tiles": lambda: comparison_kernels.fused_scores_variable(
            query, packed, lengths
        ),
        "optimized_packed": lambda: comparison_kernels.packed_scores(
            query, packed, lengths
        ),
        "lateweave_packed": lambda: lateweave.maxsim_scores_packed(
            query,
            packed,
            lengths,
            max_batch_tokens=args.max_batch_tokens,
        ),
    }
    timings, outputs = benchmark_interleaved(
        np, methods, args.warmups, args.iterations, seed
    )
    return {
        "document_count": args.documents,
        "document_tokens_minimum": int(lengths.min()),
        "document_tokens_mean": float(lengths.mean()),
        "document_tokens_maximum": int(lengths.max()),
        "total_document_tokens": int(lengths.sum()),
        "timings": timings,
        "correctness_vs_lateweave": correctness(np, outputs),
        "optimized_packed_speedup_over_old_fused": (
            timings["old_variable_fused_tiles"]["mean_ms"]
            / timings["optimized_packed"]["mean_ms"]
        ),
    }


def closed_loop_concurrency(
    np,
    method: Callable[[], Any],
    concurrency: int,
    warmups: int,
    requests_per_worker: int,
) -> dict[str, object]:
    for _ in range(warmups):
        method()
    barrier = threading.Barrier(concurrency + 1)

    def worker() -> list[float]:
        latencies: list[float] = []
        barrier.wait()
        for _ in range(requests_per_worker):
            started = time.perf_counter_ns()
            method()
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        return latencies

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        barrier.wait()
        started = time.perf_counter()
        latencies = [value for future in futures for value in future.result()]
        elapsed = time.perf_counter() - started
    result = timing_summary(np, latencies)
    result.pop("queries_per_second")
    result["throughput_queries_per_second"] = len(latencies) / elapsed
    result["total_requests"] = len(latencies)
    return result


def concurrency_case(
    np,
    lateweave,
    comparison_kernels,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    query = normalized(np, generator, (args.query_tokens, args.dimension))
    documents = normalized(
        np,
        generator,
        (args.documents, args.concurrency_document_tokens, args.dimension),
    )
    packed = documents.reshape(-1, args.dimension)
    lengths = np.full(
        args.documents, args.concurrency_document_tokens, dtype=np.int64
    )
    methods = {
        "old_fused_tiles": lambda: comparison_kernels.fused_scores(query, documents),
        "optimized_packed": lambda: comparison_kernels.packed_scores(
            query, packed, lengths
        ),
        "lateweave_packed": lambda: lateweave.maxsim_scores_packed(
            query,
            packed,
            lengths,
            max_batch_tokens=args.max_batch_tokens,
        ),
    }
    results: dict[str, object] = {}
    method_order = list(methods)
    random.Random(seed).shuffle(method_order)
    for name in method_order:
        results[name] = {
            str(concurrency): closed_loop_concurrency(
                np,
                methods[name],
                concurrency,
                args.warmups,
                args.requests_per_worker,
            )
            for concurrency in args.concurrency
        }
    return {
        "document_count": args.documents,
        "document_tokens": args.concurrency_document_tokens,
        "results": results,
    }


def run_worker(args: argparse.Namespace) -> None:
    np, lateweave, comparison_kernels = import_implementations()
    workloads = {
        f"fixed_{args.documents}x{tokens}": fixed_case(
            np,
            lateweave,
            comparison_kernels,
            args,
            tokens,
            args.seed + position,
        )
        for position, tokens in enumerate(args.fixed_document_tokens)
    }
    workloads[
        f"variable_{args.documents}x{args.variable_min_tokens}_to_"
        f"{args.variable_max_tokens}"
    ] = variable_case(
        np,
        lateweave,
        comparison_kernels,
        args,
        args.seed + len(args.fixed_document_tokens),
    )
    result = {
        "rayon_threads": args.worker_threads,
        "blas_threads": {
            name: os.environ.get(name) for name in BLAS_THREAD_ENVIRONMENT
        },
        "modules": {
            "lateweave": lateweave.__file__,
            "comparison_kernels": comparison_kernels.__file__,
            "numpy": np.__version__,
        },
        "single_query": workloads,
        "concurrent_queries": concurrency_case(
            np,
            lateweave,
            comparison_kernels,
            args,
            args.seed + 10_000,
        ),
        "peak_rss_mib": peak_rss_mib(),
    }
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
