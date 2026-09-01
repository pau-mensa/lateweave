from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lateweave import (
    Candidate,
    IncompatibleIndexError,
    IndexManifest,
    Query,
    ResourceBudget,
    Score,
    ScorerCapabilities,
    SearchPipeline,
    maxsim_scores_packed,
)


def manifest(**changes: object) -> IndexManifest:
    value = IndexManifest(
        corpus_id="laws",
        corpus_version="v1",
        document_count=3,
        document_ids_sha256="abc",
        encoder="encoder",
        encoder_revision="rev",
        tokenizer="tokenizer",
        dimension=2,
        dtype="float32",
        normalized=True,
        representation="external-test-representation",
        score_semantics="external-test-score",
    )
    return replace(value, **changes)


class ExternalGenerator:
    def __init__(self, index_manifest: IndexManifest) -> None:
        self.manifest = index_manifest

    def gather(self, query: Query, limit: int) -> tuple[Candidate, ...]:
        assert query.text == "query"
        return tuple(
            Candidate(document_id, gather_score, rank, "external-generator")
            for rank, (document_id, gather_score) in enumerate(
                [(2, 100.0), (0, 10.0), (1, 1.0)][:limit]
            )
        )


class ExternalScorer:
    capabilities = ScorerCapabilities(
        preferred_batch_tokens=1024,
        supports_mmap=True,
        supports_candidate_reordering=True,
        score_semantics="external-test-score",
    )

    def __init__(self, index_manifest: IndexManifest) -> None:
        self.manifest = index_manifest
        self.received: list[int] = []

    def score(
        self,
        query: Query,
        candidates: tuple[Candidate, ...],
        *,
        budget: ResourceBudget,
    ) -> tuple[Score, ...]:
        self.received = [candidate.document_id for candidate in candidates]
        values = {0: 3.0, 1: 5.0, 2: 2.0}
        return tuple(Score(document_id, values[document_id]) for document_id in self.received)


def test_external_implementations_compose_without_backend_dependencies() -> None:
    index_manifest = manifest()
    scorer = ExternalScorer(index_manifest)
    result = SearchPipeline(ExternalGenerator(index_manifest), scorer).search(
        Query("query"), gather_limit=3, limit=2
    )

    assert scorer.received == [2, 0, 1]
    assert [row.document_id for row in result.documents] == [1, 0]
    assert result.diagnostics == {
        "candidate_count": 3,
        "scored_count": 3,
        "generator": "ExternalGenerator",
        "scorer": "ExternalScorer",
        "score_semantics": "external-test-score",
    }


def test_gather_scores_do_not_leak_into_final_ranking() -> None:
    index_manifest = manifest()
    result = SearchPipeline(
        ExternalGenerator(index_manifest), ExternalScorer(index_manifest)
    ).search(Query("query"), gather_limit=3, limit=3)

    assert [row.document_id for row in result.documents] == [1, 0, 2]


def test_pipeline_rejects_manifest_mismatch_before_search() -> None:
    with pytest.raises(IncompatibleIndexError, match="generation"):
        SearchPipeline(ExternalGenerator(manifest()), ExternalScorer(manifest(generation=1)))


def test_query_embedding_provider_is_lazy_and_cached() -> None:
    calls = 0

    def encode(query: Query) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones((2, 2), dtype=np.float32)

    query = Query("query", embedding_provider=encode)
    assert calls == 0
    assert query.embeddings is query.embeddings
    assert calls == 1


def test_native_ranking_rejects_scorer_candidate_drift() -> None:
    class BrokenScorer(ExternalScorer):
        def score(self, query, candidates, *, budget):  # type: ignore[no-untyped-def]
            return ()

    index_manifest = manifest()
    with pytest.raises(ValueError, match="omitted candidate"):
        SearchPipeline(ExternalGenerator(index_manifest), BrokenScorer(index_manifest)).search(
            Query("query"), gather_limit=3, limit=2
        )


def test_pipeline_rejects_noncanonical_gather_ranks() -> None:
    class BrokenGenerator(ExternalGenerator):
        def gather(self, query: Query, limit: int) -> tuple[Candidate, ...]:
            return (Candidate(0, 1.0, 4, "broken"),)

    index_manifest = manifest()
    with pytest.raises(ValueError, match="contiguous and zero-based"):
        SearchPipeline(BrokenGenerator(index_manifest), ExternalScorer(index_manifest)).search(
            Query("query"), gather_limit=1, limit=1
        )


def test_packed_maxsim_matches_reference_with_bounded_batches() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2**-0.5, 2**-0.5],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    lengths = np.asarray([2, 1, 1], dtype=np.int64)

    scores = maxsim_scores_packed(
        query,
        documents,
        lengths,
        max_batch_tokens=2,
        threads=2,
    )

    assert scores.tolist() == pytest.approx([2.0, 2**0.5, -1.0], abs=1e-6)


def test_packed_maxsim_validates_document_layout() -> None:
    with pytest.raises(ValueError, match="lengths sum"):
        maxsim_scores_packed(
            np.ones((1, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.asarray([1], dtype=np.int64),
        )


@pytest.mark.parametrize("threads", [1, 2])
def test_packed_maxsim_matches_numpy_for_variable_documents(threads: int) -> None:
    rng = np.random.default_rng(42)
    query = rng.standard_normal((5, 8), dtype=np.float32)
    lengths = np.asarray([1, 3, 7, 2, 5], dtype=np.int64)
    documents = rng.standard_normal((int(lengths.sum()), 8), dtype=np.float32)
    expected = []
    start = 0
    for length in lengths:
        document = documents[start : start + length]
        expected.append(float((document @ query.T).max(axis=0).sum()))
        start += int(length)

    observed = maxsim_scores_packed(
        query,
        documents,
        lengths,
        max_batch_tokens=6,
        threads=threads,
    )

    assert observed.tolist() == pytest.approx(expected, abs=2e-5)
