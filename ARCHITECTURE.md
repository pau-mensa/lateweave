# lateweave architecture

## Ownership boundary

Lateweave owns the search algebra, the identity checks that make composition
safe, deterministic ranking, and the CPU MaxSim kernel. It does not own
retrieval algorithms, encoders, or engine index layouts.

```text
External engines                          lateweave

text / features -> (segment, ID)s ------>  CandidateGenerator contract
(segment, ID)s  -> qualified scores ---->  Reranker contract (optional)
one segment's document vectors   ------>  MultiVectorSource contract
                                            |
                                            +-- segment identity between stages
                                            +-- representation identity with the query
                                            +-- exact candidate-set validation
                                            +-- deterministic top-k, timings

Optional lateweave store snapshot ------->  MultiVectorSource -> MaxSimReranker
```

No engine is named in the package. Adapters live with their engine or in
cookbooks.

A service built on lateweave owns what surrounds a search: collections,
persistence of its own records, authorization, HTTP/MCP, quotas, billing, and
choosing which gatherer, reranker, and source make up a retrieval recipe.
Lateweave owns the recipe's execution: candidate generation contracts,
optional reranking, representation compatibility, search results with
provenance, the pipeline entrypoint, and the vector stores a MaxSim rerank
reads.

## One implementation, two languages

```text
lateweave (crate)            no PyO3, no NumPy; the only implementation
  ^
  |  Rust contracts
  |
lateweave-python             PyO3 module lateweave._native
  (bindings/python)           adapts Python stages, sources, and features
  ^
  |
lateweave (Python package)   manifest dataclasses, protocols, type stubs
```

A Rust service depends on the crate directly. The Python package is a binding:
`SearchPipeline`, `MaxSimReranker`, `Query`, `Feature`, and the stores are
native classes, and a Python gatherer, reranker, or `MultiVectorSource` is
wrapped so the Rust pipeline can call it. An exception raised by Python code
crosses the pipeline as `Error::External` and reaches the caller unchanged.
Search releases the GIL and reacquires it only to call back into Python, so a
pipeline built entirely from native stages runs without it.

A Python feature value is kept as the Python object for Python stages; a Rust
stage that needs a token matrix gets it converted, once, on first use.

## Two identities

**Segment identity** (`Segment`, identified by its `CorpusManifest`): which
documents, in which internal order, at which mutation generation. A segment is
an immutable snapshot of one corpus: its manifest plus the external IDs its
internal IDs name. Internal IDs are dense `0..n-1` and local to the segment, so
a document is `(segment, internal ID)` and corpora indexed separately never
share, or have to agree on, an ID space. Anything outside the pipeline that
must survive re-indexing refers to external IDs, never to internal ones.

A mutation never changes a segment; it yields the next generation.
`Segment.appended` and `Segment.deleted` compute it, compacted exactly as a
store mutation leaves it, so an engine that renumbers on delete can check it
agrees with the store rather than trust that it does.

**Representation identity** (`Representation`): which encoder, revision,
dimension, normalization, and templates produced a vector feature.
A stage declares the representation of each feature it consumes in `requires`;
a query declares the representation of each feature it carries. The pipeline
checks them against each other before any stage runs.

Keeping them apart is what lets a text gatherer and a vector reranker compose
without the gatherer pretending to have an encoder, and what lets two stages use
different encoders on purpose.

## Query features

A `Query` is raw text plus a mapping of named `Feature`s. A feature is a
representation plus either a value or a provider; the provider runs at most
once, when a stage first asks. The pipeline asks for every required feature up
front, so an unservable query fails before gathering and a feature from the
wrong encoder is refused before it is materialized.

Feature names are plain strings agreed between a query's producer and the
stages that consume them. The package fixes none; `MaxSimReranker` defaults to
`"multi_vector"`.

## Stages

`CandidateGenerator.segments` are the snapshots a gatherer searches, each under
a distinct corpus ID; one gatherer may search several, and fusing the hits of
several engines is a gatherer like any other. `gather(query, limit,
subset=None)` returns unique `(segment, internal ID)` candidates with gather
scores, dense zero-based ranks, and provenance. `subset` maps every searched
corpus ID to the strictly ascending internal IDs the search is restricted to;
a gatherer that cannot honour it raises rather than ignores it, and the
pipeline refuses any candidate outside it, outside its segment, or from a
snapshot the gatherer did not declare. The gatherer's `score_semantics`
qualifies its gather scores, which rank the results when no reranker follows.

`Reranker.segments` must hold the same snapshot of every segment the gatherer
searches; the pipeline checks that once, at construction. `rerank(query,
candidates, budget=...)` returns exactly one qualified score per candidate, in
candidate order. Gather scores never influence a reranked result.
`ResourceBudget` crosses the boundary because bounded execution is caller
policy; each reranker maps it onto its own representation.

A pipeline is bound to the snapshots its stages hold. After a mutation, build a
new pipeline over the new snapshots and swap it in: searches already running
finish on the old ones, and nothing can pair a gatherer's ID with another
generation's vectors.

## Multi-vector sources

A `MultiVectorSource` is a snapshot of one segment's vectors: it names its
`segment`, and the vectors an internal ID names never change for its lifetime.
`fetch(document_ids)` returns a packed float32 token matrix for the requested
documents in the requested order, plus their lengths, and the source declares
the representation of those vectors and the score semantics MaxSim over them
has. `MaxSimReranker` is the kernel plus one source per segment: it routes each
candidate to its segment's source, refuses a candidate from any other snapshot,
and requires every source to share one representation and one score semantics
so that scores across segments share a scale.

Snapshots of the two lateweave stores are sources for gatherers that hold no
document vectors. An engine that already reconstructs its own vectors implements the
protocol over them and stores nothing twice. When a rerank is meant to add
fidelity over a lossy engine index, a `Float32VectorStore` alongside it is the
deliberate second copy.

Sources are not generalized beyond multi-vector. Another kind of reranker brings
its own document representation behind the same `Reranker` protocol.

## Stores

```text
FixedRecordVectorStore
├── Float32VectorStore   exact
└── Int8VectorStore      symmetric INT8 per token, float32 row scale
```

Both use memory-mapped fixed-width token records. Each generation is its own
file set: one `.npy` per array, `document-offsets-G.npy`, and
`document-ids-G.json` holding the segment's external IDs. `storage.json` names
the live generation and carries the format, representation, corpus manifest,
and token count. In Rust they are one `VectorStore` parameterized by
`StoreFormat`.

A store holds one segment. Mutations take external IDs: append streams the
existing records and the new ones into the next generation's files; delete
copies the surviving runs of records and compacts IDs. Either loads the new
files and then publishes them by replacing `storage.json`, the single atomic
step, and removes the staged files if anything before it fails. Files are
never modified after they are written, so a `StoreSnapshot` keeps reading its
generation through its own maps however many mutations follow; the files of
superseded generations are removed on the next publish once nothing maps them.
Mutations run one at a time; one process writes a store and any number read
it. The files are ordinary little-endian `.npy`, readable by NumPy.

## Native scoring and execution

`maxsim_scores` (`maxsim_scores_packed` in Python) accepts a contiguous
float32 token matrix plus document lengths. Rust performs batched SGEMM, SIMD maximum reduction, and deterministic
document-order restoration. Token batch size and worker count are explicit.
The kernel knows nothing about where vectors came from.

Batches are scored in parallel with rayon and each batch performs its own
SGEMM, so the SGEMM itself is called single-threaded. A BLAS that parallelizes
internally nests inside that and oversubscribes: on a 16-core host, capping the
inner layer with `OMP_NUM_THREADS=4` is worth about 24% against leaving it to
spawn a thread per core.

The pipeline's ranking step enforces the reranker contract (one score per
candidate, no NaN) and orders by score, then gather rank.

## The SGEMM dependency

One SGEMM is the whole of the MaxSim kernel's floating-point cost, and it is the
only thing in lateweave with more than one implementation. Which one is used is
decided at compile time, never at run time.

| Build | SGEMM | External library |
| --- | --- | --- |
| default | bundled `matrixmultiply` | none |
| `--features openblas` | system OpenBLAS | `libopenblas.so.0` at build time |
| macOS | Accelerate | none; part of the OS |

The pure-Rust kernel is the default because it makes the crate link and the
extension module import everywhere with no system dependency at all. It is
roughly 1.75x slower than OpenBLAS on the shapes this kernel sees -- a tall,
skinny product where `k` is the embedding dimension and `n` is a batch of
document tokens -- and produces bit-identical results.

`--features openblas` is worth taking for a deployment. Combined with
auditwheel's repair step, which `maturin` runs by default, the resulting wheel
*vendors* `libopenblas.so.0` into `lateweave.libs/` and resolves it through an
`$ORIGIN` RPATH. That is the configuration to prefer: full kernel speed and no
runtime dependency on the host having a BLAS at all.

```bash
# Self-contained, full speed. OPENBLAS_LIB_DIR is only needed when the build
# machine keeps libopenblas.so.0 outside the linker's default search path.
OPENBLAS_LIB_DIR=/usr/lib \
  maturin build --release --features pyo3/extension-module,openblas
```

`pyo3/extension-module` has to be repeated on that command line. `--features`
replaces the list in `[tool.maturin]` rather than adding to it, and dropping
`extension-module` makes pyo3 link `libpython`, which auditwheel then bundles --
a second copy of the interpreter inside the wheel, and 35 MB of it.

### Why the choice is not deferred to run time

A dynamically resolved `sgemm_` is a correctness-shaped problem wearing
performance clothing. The symbol is satisfied by any BLAS the loader finds
first, and the netlib reference implementation satisfies it perfectly well:
same results, no error, no warning, and roughly 3.5x the latency. A build that
declares its kernel cannot be silently downgraded that way, and a vendored
library cannot be substituted at all.

This is also why Linux does not simply mirror the macOS arrangement. Accelerate
is part of macOS, so linking it is a fact about the platform. There is no
equivalent guarantee for OpenBLAS on Linux, so requiring it by default would
mean an extension that fails to import on some hosts and runs quietly slower on
others.

## Where purity can fail

- Some engines fuse gathering and scoring so tightly that an external
  candidate list destroys their defining optimization. They are gatherers with
  qualified scores and no reranker, which the pipeline supports directly.
- The CPU kernel reranks host-resident sources. A source on an accelerator
  pays a device-to-host copy to be reranked here; on such deployments the
  engine's own scoring is the reranker.
- Python protocols are structural. Conformance is established by tests
  against real adapters, not by type hints; the Rust traits are checked by the
  compiler.
