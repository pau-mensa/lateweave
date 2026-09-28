# lateweave architecture

## Ownership boundary

Lateweave owns the search algebra, the checks that make composition safe,
deterministic ranking, and the CPU MaxSim kernel. It does not own retrieval
algorithms, encoders, engine index layouts, or keeping any index up to date.

```text
External engines                          lateweave

text / features -> (corpus, ID)s, as_of -->  CandidateGenerator contract
(corpus, ID)s  -> scores or None, as_of -->  Reranker contract (optional)
one corpus's document vectors, as_of ---->  MultiVectorSource contract
                                            |
                                            +-- representation identity with the query
                                            +-- exact candidate-set validation
                                            +-- presence in every stage, freshness
                                            +-- deterministic top-k, timings

A lateweave vector store ---------------->  MultiVectorSource -> MaxSimReranker
```

No engine is named in the package. Adapters live with their engine or in
cookbooks.

A service built on lateweave owns what surrounds a search: collections,
persistence of its own records, authorization, HTTP/MCP, quotas, billing, and
choosing which gatherer, reranker, and source make up a retrieval recipe.
Lateweave owns the recipe's execution: candidate generation contracts,
optional reranking, representation compatibility, search results with
provenance and freshness, the pipeline entrypoint, and reading the vector
stores a MaxSim rerank scores. Writing indexes, stores included, belongs to
whatever indexes the corpus.

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
lateweave (Python package)   Representation, protocols, type stubs
```

A Rust service depends on the crate directly. The Python package is a binding:
`SearchPipeline`, `MaxSimReranker`, `Query`, `Feature`, and the store reader and
writer are native classes, and a Python gatherer, reranker, or `MultiVectorSource` is
wrapped so the Rust pipeline can call it. An exception raised by Python code
crosses the pipeline as `Error::External` and reaches the caller unchanged.
Search releases the GIL and reacquires it only to call back into Python, so a
pipeline built entirely from native stages runs without it.

A Python feature value is kept as the Python object for Python stages; a Rust
stage that needs a token matrix gets it converted, once, on first use.

## Two identities

**Document identity** (`DocumentKey`): the corpus a document belongs to and the
ID the system of record gives it. It is the only name for a document that
crosses a stage boundary. Every engine maps it to its own internal positions
privately and may renumber, rebuild, or compact whenever it likes, since no
other stage ever sees those positions; stages indexed separately never share,
or have to agree on, anything but the keys.

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

`CandidateGenerator.gather(query, limit, subset=None)` returns `Gathered`:
unique candidates, each a `DocumentKey` with a gather score, a dense zero-based
rank, and provenance, plus `as_of`. One gatherer may search several corpora,
and fusing the hits of several engines is a gatherer like any other. `subset`
maps corpora to the document IDs the search is restricted to; a corpus it does
not name contributes nothing, an ID the index does not hold is not a
candidate, and a gatherer that cannot honour it raises rather than ignores it.
The pipeline refuses a repeated candidate or one outside the subset. The
gatherer's `score_semantics` qualifies its gather scores, which rank the
results when no reranker follows.

`Reranker.rerank(query, candidates, budget=...)` returns `Scored`: exactly one
qualified score or `None` per candidate, in candidate order, plus `as_of`.
`None` means the reranker's index does not hold the document; a candidate from
a corpus the reranker cannot score at all is an error. Gather scores never
influence a reranked result. `ResourceBudget` crosses the boundary because
bounded execution is caller policy; each reranker maps it onto its own
representation.

Each call reads one consistent state of each index it touches, and `as_of` is
when that state was committed: every write committed to the index before it is
reflected in the result. Consistency is needed only there, inside a stage;
between stages, keys carry it.

## Freshness

Indexes move while a service searches them, often through writers in other
processes using other libraries, and no two of them move together. The
pipeline treats each as an independent, possibly lagging replica of the
corpus, and asks nothing of whoever maintains them:

- A document is ranked only when every stage holds it. The gatherer holds
  every candidate it returns; a candidate the reranker scores `None` is
  dropped and counted in the diagnostics. So a delete is served as soon as
  any index applies it, and an insert once every index has. An update is
  recalled from whatever text the gatherer indexed and scored on whatever the
  reranker holds, both of that document.
- `SearchResult.as_of` is the oldest `as_of` among the stages: every write
  committed to every index before it is reflected in the result. What an
  answer that old means, a failure, a warning, or nothing, is the caller's
  policy.

There is no snapshot to agree on, so there is nothing to wait for, pin, or
detect as stale beyond `as_of`: a pipeline is built without looking at any
index, and an index rebuilt from scratch, restored from a backup, or left
behind by a crashed writer is served as whatever it holds, with its age. A
search may return fewer than `limit` documents while an index lags; the
dropped count says how many.

The price is that `as_of` must be honest: it is the commit time of the state
read, not the time a stage reloaded it, and an idle index has to keep
committing so that its age stays true.

## Multi-vector sources

A `MultiVectorSource` serves one corpus's vectors: it names its `corpus`,
declares the representation of its vectors and the score semantics MaxSim over
them has, and hands out views. A `VectorView` is one consistent state with its
`as_of`: `document_lengths(ids)` gives the token count of each document it
holds and `None` for the rest, and `fetch(ids)` returns a packed float32 token
matrix for held documents in the requested order, plus their lengths.
`MaxSimReranker` is the kernel plus one source per corpus: each rerank takes
one view of every source, routes each candidate to its corpus's view, leaves
unscored what a view does not hold, and reports the oldest view's `as_of`.
Every source must share one representation and one score semantics so that
scores across corpora share a scale.

A lateweave `VectorStore` is a source for gatherers that hold no document
vectors. An engine that already reconstructs its own vectors implements the
protocol over them and stores nothing twice. When a rerank is meant to add
fidelity over a lossy engine index, a float32 store alongside it is the
deliberate second copy.

Sources are not generalized beyond multi-vector. Another kind of reranker brings
its own document representation behind the same `Reranker` protocol.

## Stores

```text
VectorStore          reads; follows manifest.json
VectorStoreWriter    one writer of the format; any other process may write it
```

The format, specified in [STORE_FORMAT.md](STORE_FORMAT.md), is the contract,
so the process that maintains a corpus need not use lateweave to maintain its
vectors. A store is a list of immutable segments, each `.npy` arrays with fixed
width token records (`float32`, or `int8` codes with a float32 row scale) plus
its document IDs, and per segment an immutable tombstone file of deleted rows.
`manifest.json`, replaced by rename, is the only file that changes: it names
the live segments and tombstones, the commit number, and `committed_at`, which
is the store's `as_of`. The newest row of a document decides whether it is
present.

So maintenance is as cheap and as deferred as the format allows. An append
writes one segment of the new documents, whether or not they were present; a
delete writes one small tombstone file; neither touches existing records, and
neither renumbers anything any other index depends on. Compaction, merging the
present documents into one segment, reclaims space and changes no read; it
runs whenever the writer likes, or never.

`VectorStore::view` rereads `manifest.json` and loads only segments and
tombstones it has not seen. A view maps its files and keeps reading them
however many commits follow; a reader that finds a file its manifest names
already removed rereads the manifest. One writer writes a store at a time and
any number read it. The files are ordinary little-endian `.npy`, readable and
writable by NumPy.

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

The pipeline's ranking step enforces the reranker contract (one score or
`None` per candidate, no NaN) and orders the scored candidates by score, then
gather rank.

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
