# lateweave

`lateweave` is a Rust library, with Python bindings, for composing retrieval
pipelines out of engines it does not implement:

```text
Query -> CandidateGenerator -> [Reranker] -> deterministic top-k
```

A `Query` is raw text plus named features, each stamped with the
`Representation` (encoder identity) that produced it and materialized at most
once. A stage declares the features it consumes and the `CorpusManifest`
(document-set identity) it indexes. Stages must agree on the corpus; each stage
must agree with the query on the representation of every feature it uses. The
reranker is optional: without one, gather scores rank.

The package supplies:

- `Query`, `Feature`, `Candidate`, `Score`, `CorpusManifest`, `Representation`;
- the `CandidateGenerator`, `Reranker`, and `MultiVectorSource` contracts;
- `SearchPipeline`: compatibility checks, optional rerank, subset enforcement,
  deterministic ranking, provenance, timings;
- `MaxSimReranker` over any `MultiVectorSource`, backed by a packed CPU
  MaxSim kernel (SGEMM, SIMD maximum reduction, bounded token batches, worker
  pools);
- `Float32VectorStore` and `Int8VectorStore` (one `VectorStore` in Rust):
  memory-mapped multi-vector sources for gatherers, such as BM25, that keep no
  document vectors.

All of it is implemented once, in the `lateweave` crate, which builds without
PyO3 or NumPy. The Python package is a binding over that crate: a Python
gatherer, reranker, or source is adapted to the Rust contracts, and the
pipeline, MaxSim, and stores run natively.

Nothing in the package names an engine. bm25s, FastPLAID, NextPlaid and others
appear only in cookbooks.

## Layout

```text
Cargo.toml          the `lateweave` crate (workspace root)
src/                pipeline, stages, MaxSim kernel and reranker, stores
examples/           Rust usage
bindings/python/    `lateweave-python`: the PyO3 module `lateweave._native`
python/lateweave/   manifest dataclasses, protocols, type stubs
```

## Using it from Rust

```toml
[dependencies]
lateweave = { git = "https://github.com/pau-mensa/lateweave" }
```

```rust
use std::sync::Arc;
use lateweave::{
    Feature, MaxSimReranker, Query, SearchPipeline, SearchRequest, TokenMatrix, VectorStore,
    DEFAULT_FEATURE,
};

let store = Arc::new(VectorStore::open("index/vectors")?);
let reranker = MaxSimReranker::new(store.clone(), corpus.clone(), DEFAULT_FEATURE)?;
let pipeline = SearchPipeline::new(Arc::new(my_gatherer), Some(Arc::new(reranker)))?;

let query = Query::new("prescripción de una deuda tributaria").with_feature(
    DEFAULT_FEATURE,
    Feature::lazy(store.representation().clone(), move || {
        TokenMatrix::new(encode(&text), dimension)
    }),
);
let result = pipeline.search(&query, &SearchRequest::new(500, 100))?;
```

A gatherer implements `CandidateGenerator`; any reranker implements
`Reranker`. Stage and source errors from outside lateweave travel as
`Error::External`. [examples/stored_maxsim.rs](examples/stored_maxsim.rs) is a
complete program: `cargo run --example stored_maxsim`.

## Building

```bash
cargo test                 # the Rust library and its example
pip install maturin
maturin build --release    # the Python wheel
```

That default build has no external library dependency: the MaxSim kernel's
SGEMM comes from the bundled pure-Rust `matrixmultiply`, so the library links
and the extension imports on any host.

For a deployment, take the OpenBLAS kernel instead — about 1.75x faster on the
shapes this kernel sees, bit-identical results. Rust dependents enable the
crate's `openblas` feature; the wheel forwards the same feature:

```bash
maturin build --release --features pyo3/extension-module,openblas
```

`maturin` runs auditwheel's repair step, which **vendors** `libopenblas.so.0`
into `lateweave.libs/` inside the wheel. So this is self-contained too: it needs
OpenBLAS on the machine that *builds* the wheel, not on the machines that
install it. Set `OPENBLAS_LIB_DIR` if `libopenblas.so.0` lives outside the
linker's default search path.

Two things to know about that command line:

- `pyo3/extension-module` must be repeated. `--features` replaces the list in
  `[tool.maturin]` instead of extending it, and without `extension-module`
  pyo3 links `libpython`, which auditwheel then bundles into the wheel.
- The library lateweave needs exports the plain Fortran symbol `sgemm_`.
  `scipy-openblas32` on PyPI does not qualify -- it is a real threaded
  OpenBLAS, but every symbol carries a `scipy_` prefix.

macOS needs no feature flag: Accelerate ships with the OS and is used
automatically. [ARCHITECTURE.md](ARCHITECTURE.md#the-sgemm-dependency) explains
why Linux does not have an equivalent default.

## Queries and features

```python
from lateweave import Feature, Query, Representation

representation = Representation(
    encoder="lightonai/LateOn-Code", encoder_revision="main", dimension=128, normalized=True
)
query = Query(
    "CUDA_ERROR_ILLEGAL_ADDRESS after switching to bf16 attention",
    multi_vector=Feature(representation, provider=lambda: encode(text)),
)
```

`provider` runs the first time a stage asks for the feature, so a text-only
gatherer never pays for encoding, and a gatherer and reranker share one matrix.
A stage that needs the feature from another encoder is refused before anything
runs.

## Implementing a gatherer

```python
from lateweave import Candidate, CorpusManifest


class MyGatherer:
    requires = {}                      # consumes query.text only
    score_semantics = "my-gather-score"

    def __init__(self, index, corpus: CorpusManifest):
        self.index = index
        self.corpus = corpus

    def gather(self, query, limit, *, subset=None):
        rows = self.index.retrieve(query.text, limit=limit, allowed=subset)
        return tuple(
            Candidate(document_id, score, rank, "my-engine")
            for rank, (document_id, score) in enumerate(rows)
        )
```

`subset` is an ascending int64 array of internal IDs, or `None`. A gatherer
that cannot honour it must raise; the pipeline refuses a candidate outside the
subset. A gatherer that consumes a vector feature declares it:
`requires = {"multi_vector": representation}`. `corpus`, `requires`, and
`score_semantics` are read once, when the `SearchPipeline` is built.

## Implementing a reranker or a source

A reranker owns whatever it needs to score. When what it needs is the token
vectors of candidate documents, implement `MultiVectorSource` and let
`MaxSimReranker` do the scoring:

```python
from lateweave import MaxSimReranker


class EngineVectors:
    """Token vectors an engine already holds; no second copy."""

    representation = representation
    score_semantics = "engine-reconstructed-full-maxsim"

    def __init__(self, engine):
        self.engine = engine
        self.document_count = engine.document_count

    def document_lengths(self, document_ids):
        return {item: self.engine.length(item) for item in document_ids}

    def fetch(self, document_ids, *, threads=None):
        rows = [self.engine.vectors(item) for item in document_ids]   # float32 [tokens, D]
        return np.concatenate(rows), np.asarray([len(r) for r in rows], dtype=np.int64)


reranker = MaxSimReranker(EngineVectors(engine), corpus)
```

A lateweave store passed as the source is read natively, without calling back
into Python.

Or write a reranker directly:

```python
class MyReranker:
    requires = {"multi_vector": representation}
    score_semantics = "my-qualified-score-semantics"

    def __init__(self, corpus):
        self.corpus = corpus

    def rerank(self, query, candidates, *, budget):
        vectors = query.feature("multi_vector", representation)
        return tuple(Score(c.document_id, self.score_one(vectors, c.document_id)) for c in candidates)
```

The pipeline requires exactly one score per candidate, never NaN.

## Stores

`Float32VectorStore` and `Int8VectorStore` are `MultiVectorSource`
implementations for gatherers without document vectors. A store is created with
the `Representation` of its vectors and refuses queries from any other encoder.

```python
from lateweave import Float32VectorStore, MaxSimReranker, SearchPipeline

store = Float32VectorStore.create("index/vectors", packed_embeddings, lengths, representation)
pipeline = SearchPipeline(gatherer, MaxSimReranker(store, corpus))
result = pipeline.search(query, gather_limit=500, limit=100)
```

| Store | Bytes per token | Score semantics |
|---|---|---|
| `Float32VectorStore` | `4D` | `float32-exact-full-maxsim` |
| `Int8VectorStore` | `D + 4` | `int8-reconstructed-approximate-full-maxsim` |

Both append and delete in place; a delete compacts internal IDs to `0..n-1`.
The on-disk format is plain `.npy` files and `storage.json`, the same from
Rust and Python.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the ownership rules and
[cookbook/README.md](cookbook/README.md) for the BM25 recipe.
