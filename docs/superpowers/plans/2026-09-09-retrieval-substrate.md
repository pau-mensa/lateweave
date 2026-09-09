# Retrieval Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape lateweave into a lean gather → optional rerank substrate whose
stages declare the query features and document representations they consume,
so any engine can plug in without duplicating storage or processing.

**Architecture:** A `Query` is text plus named, lazily materialized features,
each stamped with the `Representation` (encoder identity) that produced it.
Stages expose a `CorpusManifest` (identity of the indexed document set) and a
`requires` map of feature name → `Representation`. The pipeline checks corpus
identity between stages at construction and feature compatibility against the
query at search time. The reranker is optional; without one, gather scores rank.
The MaxSim reranker consumes any `MultiVectorSource`; the fixed-record stores
(float32 exact, int8 approximate) are two such sources.

**Tech Stack:** Rust (pyo3 0.24, numpy 0.24, rayon, matrixmultiply/BLAS),
Python 3.11+, numpy, pytest, maturin. Cookbook only: bm25s.

**Spec:** ../legal-encoder/docs/shared-agent-experience-pivot.md, section
"Follow-up: raw-text OSS protocol", plus the design agreed in conversation on
2026-09-09 (this plan's header records it).

## Global Constraints

- No backwards compatibility: greenfield, rename and delete freely.
- The package knows no engine by name in code (bm25s, fast-plaid, NextPlaid).
  Adapters live in cookbooks.
- No speculative flags, warnings, or capability declarations the pipeline does
  not read.
- Package dependency remains `numpy` only. `duckdb` and `zstd` leave.
- Every task ends with `.venv/bin/python -m pytest -q` green and a commit.
- Rebuild the extension after Rust edits: `uv pip install -q -e '.[dev]'`.

---

### Task 1: Delete the stale nested copy, benchmarks, and metadata store

**Files:**
- Delete: `lateweave/` (tracked duplicate of the repo at the initial commit)
- Delete: `benchmarks/` (measures stores that leave in Task 5; ships a `.so`)
- Delete: `python/lateweave/metadata.py`, `tests/test_metadata.py`
- Modify: `python/lateweave/__init__.py`, `pyproject.toml`, `.gitignore`

- [ ] **Step 1:** `git rm -r -q lateweave benchmarks python/lateweave/metadata.py tests/test_metadata.py`
- [ ] **Step 2:** Remove the `DuckDBMetadataStore, MetadataRecord` import and
      `__all__` entries from `__init__.py`; remove `metadata` extra and `duckdb`
      from `dev` in `pyproject.toml`; drop the two `/benchmarks/...` lines from
      `.gitignore`.
- [ ] **Step 3:** Run tests. Expected: all remaining tests pass.
- [ ] **Step 4:** Commit: `chore: remove nested repo copy, benchmarks, metadata store`

### Task 2: Manifests — corpus identity and representation identity

**Files:**
- Rewrite: `python/lateweave/manifest.py`
- Test: `tests/test_manifest.py` (new)

**Produces:**
```python
class IncompatibleIndexError(ValueError): ...
class IncompatibleQueryError(ValueError): ...
def document_ids_digest(document_ids: Sequence[str]) -> str

@dataclass(frozen=True)
class CorpusManifest:
    corpus_id: str; corpus_version: str; document_count: int
    document_ids_sha256: str; generation: int = 0
    def read(path) / write(path)
    def assert_compatible(self, other: CorpusManifest) -> None  # raises IncompatibleIndexError

@dataclass(frozen=True)
class Representation:
    encoder: str; encoder_revision: str; dimension: int; normalized: bool
    similarity: str = "dot"; query_template: str = ""; document_template: str = ""
    def assert_compatible(self, other: Representation) -> None  # raises IncompatibleQueryError
    def to_dict() / from_dict()
```

- [ ] **Step 1: Write failing tests**
```python
def test_corpus_manifests_compare_identity_fields_only():
    a = CorpusManifest("c", "1", 3, "abc")
    a.assert_compatible(CorpusManifest("c", "1", 3, "abc"))
    with pytest.raises(IncompatibleIndexError, match="generation"):
        a.assert_compatible(replace(a, generation=1))

def test_representation_mismatch_names_the_field():
    r = Representation("enc", "1", 8, True)
    with pytest.raises(IncompatibleQueryError, match="dimension"):
        r.assert_compatible(replace(r, dimension=4))

def test_manifests_round_trip_through_json(tmp_path):
    m = CorpusManifest("c", "1", 3, "abc", generation=2)
    m.write(tmp_path / "m.json"); assert CorpusManifest.read(tmp_path / "m.json") == m
    r = Representation("enc", "1", 8, True)
    assert Representation.from_dict(r.to_dict()) == r
```
- [ ] **Step 2:** Run: `pytest tests/test_manifest.py -v`. Expected: ImportError.
- [ ] **Step 3:** Implement `manifest.py` per the Produces block. Validation in
      `__post_init__`: non-empty strings, `document_count >= 0`,
      `generation >= 0`, `dimension > 0`.
- [ ] **Step 4:** Run tests. Expected: manifest tests pass; other suites still
      import `IndexManifest` and fail — that is expected until Tasks 3–6.
- [ ] **Step 5:** Commit: `feat: split manifest into corpus and representation identity`

### Task 3: Query features and stage protocols

**Files:**
- Rewrite: `python/lateweave/interfaces.py`
- Modify: `src/lib.rs` (delete `PyScorerCapabilities`; keep Candidate, Score,
  ResourceBudget without `max_memory_bytes`), `src/core.rs` (delete
  `ScorerCapabilities`, `CandidateGenerator`, `CandidateScorer` traits and the
  Rust `Candidate`/`Score`/`ResourceBudget` structs; keep `all_finite`,
  `validate_and_rank`, `RankingError`)
- Test: `tests/test_pipeline.py` (rewritten in Task 4; this task only needs
  `tests/test_query.py`)

**Produces:**
```python
class Feature:
    """One query representation, materialized at most once."""
    def __init__(self, representation: Representation, value=None, *, provider: Callable[[], Any] | None = None)
    representation: Representation
    @property value -> Any   # calls provider once; caches

class Query:
    def __init__(self, text: str, **features: Feature)
    text: str
    features: Mapping[str, Feature]
    def feature(self, name: str, representation: Representation) -> Any
        # KeyError-free: raises IncompatibleQueryError if missing or mismatched

@runtime_checkable
class CandidateGenerator(Protocol):
    corpus: CorpusManifest
    requires: Mapping[str, Representation]
    score_semantics: str
    def gather(self, query: Query, limit: int, *, subset: np.ndarray | None = None) -> Sequence[Candidate]

@runtime_checkable
class Reranker(Protocol):
    corpus: CorpusManifest
    requires: Mapping[str, Representation]
    score_semantics: str
    def rerank(self, query: Query, candidates: Sequence[Candidate], *, budget: ResourceBudget) -> Sequence[Score]

RankedDocument, SearchTimings(gather_seconds, rerank_seconds, total_seconds), SearchResult unchanged in shape.
```

- [ ] **Step 1: Write failing tests** (`tests/test_query.py`)
```python
def test_feature_provider_runs_once():
    calls = 0
    def encode():
        nonlocal calls; calls += 1; return np.ones((2, 2), np.float32)
    q = Query("q", multi_vector=Feature(REP, provider=encode))
    assert calls == 0
    assert q.feature("multi_vector", REP) is q.feature("multi_vector", REP)
    assert calls == 1

def test_missing_feature_is_an_incompatible_query():
    with pytest.raises(IncompatibleQueryError, match="multi_vector"):
        Query("q").feature("multi_vector", REP)

def test_feature_from_another_encoder_is_refused():
    q = Query("q", multi_vector=Feature(REP, np.ones((1, 8), np.float32)))
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        q.feature("multi_vector", replace(REP, encoder="other"))
```
- [ ] **Step 2:** Run. Expected: ImportError on `Feature`.
- [ ] **Step 3:** Implement `interfaces.py`; trim `lib.rs` and `core.rs`;
      rebuild with `uv pip install -q -e '.[dev]'`; `cargo test` must pass.
- [ ] **Step 4:** Run `tests/test_query.py`. Expected: pass.
- [ ] **Step 5:** Commit: `feat: query features with representation identity; rename scorer to reranker`

### Task 4: Pipeline with optional reranker and subset

**Files:**
- Rewrite: `python/lateweave/pipeline.py`
- Rewrite: `tests/test_pipeline.py`

**Produces:**
```python
class SearchPipeline:
    def __init__(self, gatherer: CandidateGenerator, reranker: Reranker | None = None)
        # gatherer.corpus.assert_compatible(reranker.corpus) when reranker given
    def search(self, query: Query | str, *, gather_limit: int, limit: int,
               subset: np.ndarray | None = None, budget: ResourceBudget | None = None) -> SearchResult
        # 1. for each stage, for name, rep in stage.requires: query.feature(name, rep)
        # 2. candidates = gatherer.gather(query, gather_limit, subset=subset); validate ranks
        # 3. scores = reranker.rerank(...) if reranker else gather scores as Score
        # 4. validate_and_rank; diagnostics = {candidate_count, gatherer, reranker (None ok), score_semantics}
```

- [ ] **Step 1: Write failing tests** covering: external gatherer + reranker
      compose; gather-only ranks by gather score with gather-rank tie-break;
      corpus mismatch raises at construction; missing query feature raises
      before gather (gatherer records it was never called); subset is passed
      through verbatim; reranker candidate drift and non-canonical gather ranks
      still rejected; the three kernel tests carry over unchanged.
- [ ] **Step 2:** Run. Expected: failures on the new signatures.
- [ ] **Step 3:** Implement `pipeline.py`.
- [ ] **Step 4:** Run `tests/test_pipeline.py`. Expected: pass.
- [ ] **Step 5:** Commit: `feat: optional reranker and subset filtering in the pipeline`

### Task 5: Multi-vector sources — float32 and int8 stores, MaxSim reranker

**Files:**
- Rewrite: `python/lateweave/storage.py` (drop TurboQuant and Jzip; base
  `FixedRecordVectorStore` with `Float32VectorStore` and `Int8VectorStore`;
  `create(path, embeddings, lengths, representation, *, chunk_tokens, threads)`;
  persist `representation` in `storage.json`; rename `transition` → `fetch`;
  delete `prepare_query`, `estimated_workspace_bytes`, `encoded_bytes_per_token`,
  `scoring_space`)
- Rewrite: `python/lateweave/scorers.py` → `python/lateweave/maxsim.py`
- Modify: `src/lib.rs`, `src/storage.rs` (keep int8 encode/decode, delete
  turboquant and jzip), `Cargo.toml` (drop `zstd`)
- Rewrite: `tests/test_storage.py`

**Produces:**
```python
@runtime_checkable
class MultiVectorSource(Protocol):
    representation: Representation
    score_semantics: str
    document_count: int
    def document_lengths(self, document_ids: Sequence[int]) -> dict[int, int]
    def fetch(self, document_ids: Sequence[int], *, threads: int | None = None) -> tuple[np.ndarray, np.ndarray]
        # packed float32 [tokens, dimension] in the requested order, int64 lengths

class MaxSimReranker:
    def __init__(self, source: MultiVectorSource, corpus: CorpusManifest, *, feature: str = "multi_vector")
    corpus; requires = {feature: source.representation}; score_semantics = source.score_semantics
    def rerank(self, query, candidates, *, budget) -> tuple[Score, ...]

Float32VectorStore.score_semantics == "float32-exact-full-maxsim"
Int8VectorStore.score_semantics == "int8-reconstructed-approximate-full-maxsim"
def open_vector_store(path) -> Float32VectorStore | Int8VectorStore
```

- [ ] **Step 1: Write failing tests**: both stores create/reopen/fetch in
      requested order; append and delete compact; reopen restores the
      representation; float32 fetch is bit-exact; `MaxSimReranker` scores an
      int8 store approximately and a float32 store exactly; reranker refuses a
      source whose document count differs from the corpus; an in-memory
      `MultiVectorSource` (a dict of arrays) works with the reranker without
      any store — this is the "borrowed source" case.
- [ ] **Step 2:** Run. Expected: import failures.
- [ ] **Step 3:** Implement storage, maxsim reranker, Rust trims; rebuild.
- [ ] **Step 4:** Run tests. Expected: pass; `cargo test` pass.
- [ ] **Step 5:** Commit: `feat: multi-vector sources with float32 and int8 stores; MaxSim reranker`

### Task 6: Package surface

**Files:**
- Rewrite: `python/lateweave/__init__.py`

- [ ] **Step 1:** Export exactly: `Candidate, CandidateGenerator, CorpusManifest,
      Feature, Float32VectorStore, IncompatibleIndexError, IncompatibleQueryError,
      Int8VectorStore, MaxSimReranker, MultiVectorSource, Query, RankedDocument,
      Representation, Reranker, ResourceBudget, Score, SearchPipeline,
      SearchResult, SearchTimings, document_ids_digest, maxsim_scores_packed,
      open_vector_store`.
- [ ] **Step 2:** Run the whole suite except the cookbook. Expected: pass.
- [ ] **Step 3:** Commit: `refactor: package surface for the retrieval substrate`

### Task 7: Cookbook on the new substrate

**Files:**
- Rewrite: `cookbook/bm25_stored_maxsim.py`, `cookbook/README.md`
- Rewrite: `tests/test_cookbook.py`

Changes: one `corpus-manifest.json` plus `analyzer.json` (the analyzer chain
leaves the manifest); store carries the representation; `--storage {float32,int8}`;
`search --query-embeddings` optional — without it the pipeline is gather-only;
`search --subset-id EXTERNAL_ID` (repeatable) translates to internal IDs and is
honored through bm25s `weight_mask`; `LexicalCandidateGenerator` exposes
`corpus`, `requires = {}`, `score_semantics = "bm25s-lucene"` and accepts
`subset`.

- [ ] **Step 1:** Rewrite tests for build/update/delete/search, gather-only
      search, and subset search.
- [ ] **Step 2:** Run. Expected: failures.
- [ ] **Step 3:** Rewrite the cookbook.
- [ ] **Step 4:** Run the full suite. Expected: pass.
- [ ] **Step 5:** Commit: `feat(cookbook): BM25 gather with optional MaxSim rerank and subsets`

### Task 8: Documentation

**Files:**
- Rewrite: `README.md`, `ARCHITECTURE.md`

- [ ] **Step 1:** README: what lateweave is (gather → optional rerank,
      features, sources), build instructions unchanged, stores table (two
      rows), implementing an external gatherer/reranker/source with the new
      signatures.
- [ ] **Step 2:** ARCHITECTURE: ownership boundary, query features and
      representation identity, corpus identity, multi-vector sources, kernel
      and SGEMM sections carried over.
- [ ] **Step 3:** Commit: `docs: describe the retrieval substrate`
