# lateweave architecture

## Ownership boundary

Lateweave owns search algebra, conformance, reusable execution policy, and the
CPU MaxSim kernel. It does not own retrieval algorithms or engine-specific
index layouts.

```text
External engine                         lateweave

query text/features -> gather IDs  ---> Candidate contract
private representation -> scores   ---> Score contract
                                         |
                                         +-- exact candidate-set validation
                                         +-- qualified score semantics
                                         +-- deterministic top-k
                                         +-- budgets, timing, diagnostics

Optional lateweave store ----------> StoredMaxSimScorer -> CPU MaxSim
```

bm25s, FastPLAID, WARP, and Tachiom are potential implementors or consumers of
the contracts, not dependencies of the package. Native engines that already
own document representations should implement `CandidateScorer` directly and
ignore optional storage.

The optional `DuckDBMetadataStore` is likewise an integration utility rather
than a search primitive. It maps a DuckDB expression to generator-internal
document IDs for one immutable index generation. It does not post-filter
candidates or define mutation semantics. Integrations rebuild it from the
generator's authoritative final ID bindings when publishing a new generation;
engines with native metadata ownership bypass it.

## Stable search interfaces

`CandidateGenerator.gather(query, limit)` returns ordered, unique internal
document IDs with gather scores, canonical zero-based ranks, and provenance.

`CandidateScorer.score(query, candidates, budget=...)` returns exactly one
qualified score for every supplied candidate. Representation access and
transition remain private. Gather scores do not affect final ranking unless an
external scorer explicitly declares different semantics.

Both components expose an `IndexManifest`. Composition validates corpus and
document-ID identity, encoder/tokenizer identity, query/document conventions,
and mutation generation before query execution. Generator and scorer storage
representations may intentionally differ.

## Storage is an optional implementation, not a search primitive

Some generators, notably BM25, do not retain token embeddings. For those
compositions, `StoredMaxSimScorer` uses a lateweave-owned `VectorStore`. The
store owns:

- persistent layout and representation metadata;
- query preparation and candidate-to-scoring-space transition;
- append and delete mechanics; and
- workspace estimates used by the scorer's resource policy.

There is no public codec interface in the search algebra. Compression is a
private detail of a store. The behavioral store base deliberately makes no
fixed-record assumption:

```text
VectorStore
├── FixedRecordVectorStore
│   ├── Int8VectorStore
│   └── TurboQuantVectorStore
└── JzipVectorStore
```

INT8 and TurboQuant use memory-mapped, fixed-width token records. Jzip uses a
variable-width directory and independently compressed document frames. This
separation lets future database, object-store, or codec-backed mechanics join
without changing `CandidateScorer`.

## Jzip document framing

The upstream jzip transform converts unit vectors to `D - 1` spherical angles,
transposes angles across vectors, byte-shuffles float32 lanes, and applies
zstd. Lateweave implements that algorithm natively in Rust but changes the
physical container:

```text
frame-directory.npy
  document ID -> byte offset, compressed length, token count

frames.bin
  [versioned document frame][versioned document frame]...
```

A candidate transition gathers only requested frames, decodes them in parallel
to packed float32 token rows, and invokes the shared MaxSim kernel. Appends add
new frames. Deletes copy live compressed frames into a compact replacement
without reconstruction or recompression.

Every frame contains magic, format version, normalization flags, token count,
and dimension in a little-endian header, and its zstd payload carries a content
checksum. The store checks these against its directory and manifest before
returning vectors. The format is
`lateweave-jzip-document-zstd-v1`, not the upstream CLI format.

The spherical round trip is near-lossless rather than bit-exact; its qualified
score semantics are `jzip-reconstructed-near-lossless-full-maxsim`.

## Native scoring and execution

`maxsim_scores_packed` accepts a contiguous float32 token matrix plus document
lengths. Rust performs batched SGEMM, SIMD maximum reduction, and deterministic
document-order restoration. Token batch size and worker count are explicit.
The kernel knows nothing about where vectors came from. See
[The SGEMM dependency](#the-sgemm-dependency) for which SGEMM it calls.

Batches are scored in parallel with rayon and each batch performs its own
SGEMM, so the SGEMM itself is called single-threaded. A BLAS that parallelizes
internally nests inside that and oversubscribes: on a 16-core host, capping the
inner layer with `OMP_NUM_THREADS=4` is worth about 24% against leaving it to
spawn a thread per core.

`ScorerCapabilities` records facts the runtime may rely upon, including mmap,
prefetch, candidate reordering, future CPU/GPU sharding, preferred batch size,
and score semantics. `ResourceBudget` crosses the scorer boundary because
bounded execution is caller policy; each scorer maps the budget to its private
representation.

## The SGEMM dependency

One SGEMM is the whole of the MaxSim kernel's floating-point cost, and it is the
only thing in lateweave with more than one implementation. Which one is used is
decided at compile time, never at run time.

| Build | SGEMM | External library |
| --- | --- | --- |
| default | bundled `matrixmultiply` | none |
| `--features openblas` | system OpenBLAS | `libopenblas.so.0` at build time |
| macOS | Accelerate | none; part of the OS |

The pure-Rust kernel is the default because it makes the extension module
importable everywhere with no system dependency at all. It is roughly 1.75x
slower than OpenBLAS on the shapes this kernel sees -- a tall, skinny product
where `k` is the embedding dimension and `n` is a batch of document tokens --
and produces bit-identical results.

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
  candidate list destroys their defining optimization. They should eventually
  satisfy a separate fused `SearchPlan` contract rather than fake a scorer
  boundary.
- Capability declarations need executable conformance tests; strings and type
  hints cannot guarantee behavior.
- Manifest fields must evolve conservatively. Backend-specific fields do not
  belong in the compatibility core.
- Engine adapters should live with their engine or in separately versioned
  integration packages. Cookbooks may demonstrate them without making them
  dependencies of lateweave.
