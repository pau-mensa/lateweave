# lateweave

`lateweave` is a Rust library, with Python bindings, for composing retrieval
pipelines out of engines it does not implement:

```text
Query -> CandidateGenerator -> [Reranker] -> deterministic top-k
```

A `Query` is raw text plus named features, each stamped with the
`Representation` (encoder identity) that produced it and materialized at most
once. A document is an internal ID of a `Segment`: an immutable snapshot of one
corpus, its `CorpusManifest` plus the external IDs its internal IDs name. A
gatherer searches one or more segments, and a reranker must hold the same
snapshot of each; each stage must agree with the query on the representation of
every feature it uses. The reranker is optional: without one, gather scores
rank.

The package supplies:

- `Query`, `Feature`, `Segment`, `Candidate`, `CorpusManifest`, `Representation`;
- the `CandidateGenerator`, `Reranker`, and `MultiVectorSource` contracts;
- `SearchPipeline`: compatibility checks, optional rerank, subset enforcement,
  deterministic ranking, provenance, timings;
- `MaxSimReranker` over one `MultiVectorSource` per segment, backed by a packed
  CPU MaxSim kernel (SGEMM, SIMD maximum reduction, bounded token batches,
  worker pools);
- `Float32VectorStore` and `Int8VectorStore` (one `VectorStore` in Rust):
  memory-mapped multi-vector stores whose snapshots are sources for gatherers,
  such as BM25, that keep no document vectors.

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
    Feature, MaxSimReranker, MultiVectorSource, Query, SearchPipeline, SearchRequest,
    TokenMatrix, VectorStore, DEFAULT_FEATURE,
};

let snapshot = VectorStore::open("index/vectors")?.snapshot();
let gatherer = MyGatherer::open("index/lexical", snapshot.segment().clone())?;
let source: Arc<dyn MultiVectorSource> = snapshot.clone();
let reranker = MaxSimReranker::new([source], DEFAULT_FEATURE)?;
let pipeline = SearchPipeline::new(Arc::new(gatherer), Some(Arc::new(reranker)))?;

let query = Query::new("prescripción de una deuda tributaria").with_feature(
    DEFAULT_FEATURE,
    Feature::lazy(snapshot.representation().clone(), move || {
        TokenMatrix::new(encode(&text), dimension)
    }),
);
let result = pipeline.search(&query, &SearchRequest::new(500, 100))?;
for document in &result.documents {
    println!("{} {:?} {}", document.rank, document.external_id(), document.score);
}
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

## Segments

```python
from lateweave import Segment

segment = Segment("laws", "2026-09-01", external_ids)   # internal ID i names external_ids[i]
segment.internal("law-17"), segment.external(16)
next_segment = segment.deleted(["law-17"])              # generation + 1, compacted to 0..n-1
```

A segment never changes. `appended` and `deleted` return the next generation,
compacted exactly as a store mutation leaves it, and `segment.manifest` is the
`CorpusManifest` to persist. `Segment.from_manifest(manifest, external_ids)`
rebuilds it and refuses IDs the manifest was not computed over.

## Implementing a gatherer

```python
from lateweave import Candidate, Segment


class MyGatherer:
    requires = {}                      # consumes query.text only
    score_semantics = "my-gather-score"

    def __init__(self, index, segment: Segment):
        self.index = index
        self.segment = segment
        self.segments = (segment,)

    def gather(self, query, limit, *, subset=None):
        allowed = None if subset is None else subset[self.segment.corpus_id]
        rows = self.index.retrieve(query.text, limit=limit, allowed=allowed)
        return tuple(
            Candidate(self.segment, document_id, score, rank, "my-engine")
            for rank, (document_id, score) in enumerate(rows)
        )
```

`subset` is `None`, or maps every corpus ID in `segments` to an ascending int64
array of internal IDs, empty for a segment the search excludes. A gatherer that
cannot honour it must raise; the pipeline refuses a candidate outside the
subset or from a snapshot the gatherer did not declare. A gatherer that
consumes a vector feature declares it: `requires = {"multi_vector":
representation}`. `segments`, `requires`, and `score_semantics` are read once,
when the `SearchPipeline` is built.

One gatherer may search several segments, such as two indexes of unrelated
corpora whose hits it fuses; each candidate names its own segment, and nothing
needs a shared ID space. Search results carry `segment`, `document_id`, and
`external_id`.

## Implementing a reranker or a source

A reranker owns whatever it needs to score. When what it needs is the token
vectors of candidate documents, implement `MultiVectorSource` for each segment
and let `MaxSimReranker` do the scoring:

```python
from lateweave import MaxSimReranker


class EngineVectors:
    """Token vectors an engine already holds; no second copy."""

    representation = representation
    score_semantics = "engine-reconstructed-full-maxsim"

    def __init__(self, engine, segment):
        self.engine = engine
        self.segment = segment         # the snapshot these vectors belong to

    def document_lengths(self, document_ids):
        return {item: self.engine.length(item) for item in document_ids}

    def fetch(self, document_ids, *, threads=None):
        rows = [self.engine.vectors(item) for item in document_ids]   # float32 [tokens, D]
        return np.concatenate(rows), np.asarray([len(r) for r in rows], dtype=np.int64)


reranker = MaxSimReranker([EngineVectors(laws_engine, laws), EngineVectors(cases_engine, cases)])
```

A source is a snapshot: the vectors an internal ID names must not change for
its lifetime. The reranker routes each candidate to its segment's source and
refuses a candidate from any other snapshot. A lateweave store snapshot passed
as a source is read natively, without calling back into Python.

Or write a reranker directly:

```python
class MyReranker:
    requires = {"multi_vector": representation}
    score_semantics = "my-qualified-score-semantics"

    def __init__(self, *segments):
        self.segments = segments

    def rerank(self, query, candidates, *, budget):
        vectors = query.feature("multi_vector", representation)
        return [self.score_one(vectors, c.segment, c.document_id) for c in candidates]
```

The pipeline requires exactly one score per candidate, in candidate order,
never NaN.

## Stores

`Float32VectorStore` and `Int8VectorStore` hold the vectors of one segment for
gatherers without document vectors. A store is created with the segment and the
`Representation` of its vectors, and refuses queries from any other encoder.

```python
from lateweave import Float32VectorStore, MaxSimReranker, SearchPipeline

store = Float32VectorStore.create("index/vectors", segment, packed_embeddings, lengths, representation)
pipeline = SearchPipeline(gatherer, MaxSimReranker([store.snapshot()]))
result = pipeline.search(query, gather_limit=500, limit=100)
```

| Store | Bytes per token | Score semantics |
|---|---|---|
| `Float32VectorStore` | `4D` | `float32-exact-full-maxsim` |
| `Int8VectorStore` | `D + 4` | `int8-reconstructed-approximate-full-maxsim` |

`append(external_ids, embeddings, lengths)` and `delete(external_ids)` publish
the next generation and return its snapshot, whose segment is exactly
`segment.appended(...)` or `segment.deleted(...)`; a delete compacts internal
IDs to `0..n-1`. Snapshots taken earlier keep reading their own generation, so
a pipeline built over them keeps working until it is replaced by one built over
the new snapshot. The on-disk format is plain `.npy` files, the segment's
external IDs as JSON, and `storage.json`, the same from Rust and Python.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the ownership rules and
[cookbook/README.md](cookbook/README.md) for the BM25 recipe.
