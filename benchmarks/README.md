# End-to-end storage benchmark

This benchmark measures BM25 candidate generation followed by MaxSim reranking
over the `int8`, `turboquant4`, and `jzip` vector stores. Each stage is a
separate command so its peak RSS is not inherited from an earlier operation.

The source may be the original tar archive, an extracted directory, or the
`summaries.jsonl` file itself. From the repository root, prepare the full corpus
with:

```console
uv run --with-editable . benchmarks/benchmark.py prepare \
  --source ../legal-encoder/full-es-luna.tar.gz
```

Generated files default to `benchmarks/.work`. Use `--workspace PATH` on every
command to put the potentially large benchmark data elsewhere. Preparation
stores the complete configuration in that workspace, so later stages only need
the workspace and store name:

```console
uv run --with-editable . benchmarks/benchmark.py bm25

uv run --with-editable . benchmarks/benchmark.py build --store int8
uv run --with-editable . benchmarks/benchmark.py search --store int8

uv run --with-editable . benchmarks/benchmark.py build --store turboquant4
uv run --with-editable . benchmarks/benchmark.py search --store turboquant4

uv run --with-editable . benchmarks/benchmark.py build --store jzip
uv run --with-editable . benchmarks/benchmark.py search --store jzip

uv run --with-editable . benchmarks/benchmark.py report
```

Pass `--overwrite` when rerunning `prepare`, `bm25`, or a store build. Preparing
again invalidates and removes all generated benchmark stages in that workspace.
For quick smoke tests, `prepare --documents 100` limits the corpus; the default
uses every document.

## MaxSim kernel comparison

`maxsim_kernel_comparison.py` compares three native implementations using the
same generated embeddings:

- the historical tiled-fused fixed/variable-length path;
- the optimized packed path added in `maxsim-cpu` commit `d0c9a1e`; and
- lateweave's current packed implementation.

The two historical implementations are copied into the benchmark-only native
crate at `benchmarks/maxsim-kernels`. They are versioned with lateweave for
reproducibility but are not compiled into, imported by, or exposed from the
production package. Their Apache-2.0 attribution is recorded beside the code.

The benchmark uses separate processes for every Rayon thread count because the
global Rayon pool cannot be resized after initialization. It fixes every known
BLAS thread setting to one, interleaves implementations during the single-query
test, checks their scores, and runs closed-loop concurrent queries. The default
shape matches the storage benchmark: 32 query tokens, 500 candidates, and 128
dimensions. Fixed document lengths of 32, 128, and 512 tokens plus variable
lengths from 32 through 256 are included.

On Debian or Ubuntu, install a compiler and the system OpenBLAS library before
running the benchmark:

```console
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libopenblas-dev
```

Run the comparison from the lateweave repository root:

```console
uv run --python 3.12 \
  --with-editable . \
  --with-editable benchmarks/maxsim-kernels \
  benchmarks/maxsim_kernel_comparison.py \
  --thread-counts 1 2 \
  --concurrency 1 2 4 \
  --output maxsim-comparison-x86.json
```

On a two-vCPU VPS, verify that the process can actually see two CPUs in the
result's `available_logical_cpus` field. Avoid running other CPU-intensive work
during the measurement. The output records CPU identity, platform, imported
module paths, configuration, peak RSS, correctness error, latency percentiles,
single-query throughput, and closed-loop concurrent throughput.

For a quick installation and API smoke test:

```console
uv run --python 3.12 \
  --with-editable . \
  --with-editable benchmarks/maxsim-kernels \
  benchmarks/maxsim_kernel_comparison.py \
  --thread-counts 1 \
  --concurrency 1 2 \
  --documents 32 \
  --fixed-document-tokens 16 \
  --variable-min-tokens 8 \
  --variable-max-tokens 24 \
  --warmups 1 \
  --iterations 2 \
  --requests-per-worker 2 \
  --output /tmp/maxsim-comparison-smoke.json
```
