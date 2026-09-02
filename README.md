# lateweave

`lateweave` is a Rust/PyO3 package for composing late-interaction retrieval
systems. It does not implement BM25, PLAID, WARP, Tachiom, or their index
formats. Those libraries participate through two stable interfaces:

```text
Query -> CandidateGenerator -> CandidateScorer -> deterministic top-k
```

The package supplies:

- `Query`, `Candidate`, `Score`, and `IndexManifest` value contracts;
- `CandidateGenerator` and `CandidateScorer` protocols;
- orchestration, compatibility checks, timings, and resource budgets;
- a packed CPU `maxsim_scores_packed` kernel using BLAS, SIMD maximum
  reduction, bounded token batches, and optional worker pools; and
- optional vector stores for generators, such as BM25, that do not own a
  document-vector representation.

Native late-interaction engines can ignore lateweave storage entirely. A scorer
owns every representation-specific transition behind `CandidateScorer.score`.

## Optional vector stores

`StoredMaxSimScorer` composes the MaxSim kernel with one of three stores:

| Store | Physical representation | Fidelity |
|---|---|---|
| `Int8VectorStore` | symmetric INT8 row plus one float32 scale per token | lossy |
| `TurboQuantVectorStore` | training-free TurboQuant-MSE, four bits per coordinate | lossy |
| `JzipVectorStore` | spherical coordinates, byte shuffle, and one zstd frame per document | near-lossless, not bit-exact |

Each store owns its files, query preparation, candidate reconstruction,
append, and delete behavior. Encoding and decoding functions are private native
implementation details rather than public codec primitives.

```python
from lateweave import JzipVectorStore, StoredMaxSimScorer

store = JzipVectorStore.create(
    "index/vectors",
    packed_document_embeddings,
    document_lengths,
    compression_level=1,
    threads=8,
)
scorer = StoredMaxSimScorer(store, scorer_manifest)
```

Jzip frames are document-aligned for candidate-level random access. The format
is owned and versioned by lateweave; it is deliberately not byte-compatible
with the upstream monolithic jzip CLI. Deletes compact live frames without
decoding or recompressing them, while appends encode only new documents.

## Optional metadata filtering

`DuckDBMetadataStore` is a static, portable metadata utility for candidate
generators that accept a document subset but do not own metadata filtering. It
is immutable for one index generation: after an append, delete, or rebuild, the
integration creates a replacement store from the generator's final ID
bindings. Engines that already synchronize their own metadata should keep
using their native implementation.

Install the optional dependency with `pip install 'lateweave[metadata]'`:

```python
import duckdb
from lateweave import DuckDBMetadataStore, MetadataRecord

records = [
    MetadataRecord(0, "law-1", {"country": "ES", "year": 2024}),
    MetadataRecord(1, "law-2", {"country": "FR", "year": 2025}),
]
store = DuckDBMetadataStore.create("index/metadata.duckdb", records, manifest)

country = duckdb.ColumnExpression("country")
allowed_ids = store.select(country == duckdb.ConstantExpression("ES"))
```

The result is a sorted `int64` NumPy array of generator-internal IDs. An
adapter translates it to its native filter representation, such as a Boolean
`weight_mask` for bm25s. The store accepts DuckDB expression objects rather
than raw SQL strings, and validates corpus identity, document-ID digest,
generator representation, and generation whenever it opens. External-ID
digests are computed in ascending generator-ID order, including for sparse ID
spaces.

## Implementing an external engine

```python
from lateweave import Candidate, ResourceBudget, Score, ScorerCapabilities


class MyGenerator:
    def gather(self, query, limit):
        rows = self.index.retrieve(query.text, limit=limit)
        return tuple(
            Candidate(document_id, gather_score, rank, "my-generator")
            for rank, (document_id, gather_score) in enumerate(rows)
        )


class MyScorer:
    capabilities = ScorerCapabilities(
        preferred_batch_tokens=131_072,
        supports_candidate_reordering=True,
        score_semantics="my-qualified-score-semantics",
    )

    def score(self, query, candidates, *, budget: ResourceBudget):
        values = self.index.score_candidates(query, candidates, budget=budget)
        return tuple(
            Score(candidate.document_id, value)
            for candidate, value in zip(candidates, values, strict=True)
        )
```

The scorer may reconstruct vectors, evaluate compressed codes, or fuse access
and scoring. Lateweave only requires it to declare qualified semantics and
score exactly the candidate set it receives.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the ownership rules and
[cookbook/README.md](cookbook/README.md) for BM25 plus stored-vector examples.
