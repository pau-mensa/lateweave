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
