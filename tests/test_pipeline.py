from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lateweave import (
    Candidate,
    CorpusManifest,
    Feature,
    IncompatibleIndexError,
    IncompatibleQueryError,
    Query,
    Representation,
    ResourceBudget,
    Score,
    SearchPipeline,
    maxsim_scores_packed,
)


CORPUS = CorpusManifest("corpus", "1", 3, "abc")
REPRESENTATION = Representation("encoder", "1", 2, True)


class TextGatherer:
    requires: dict[str, Representation] = {}
    score_semantics = "external-gather"

    def __init__(self, corpus: CorpusManifest = CORPUS) -> None:
        self.corpus = corpus
        self.calls = 0
        self.subsets: list[np.ndarray | None] = []

    def gather(self, query: Query, limit: int, *, subset=None) -> tuple[Candidate, ...]:
        self.calls += 1
        self.subsets.append(subset)
        assert query.text == "query"
        rows = [(2, 100.0), (0, 10.0), (1, 10.0)]
        if subset is not None:
            rows = [row for row in rows if row[0] in subset]
        return tuple(
            Candidate(document_id, gather_score, rank, "external")
            for rank, (document_id, gather_score) in enumerate(rows[:limit])
        )


class VectorReranker:
    requires = {"multi_vector": REPRESENTATION}
    score_semantics = "external-rerank"

    def __init__(self, corpus: CorpusManifest = CORPUS) -> None:
        self.corpus = corpus
        self.received: list[int] = []

    def rerank(self, query, candidates, *, budget: ResourceBudget) -> tuple[Score, ...]:
        assert query.feature("multi_vector", REPRESENTATION).shape == (1, 2)
        self.received = [candidate.document_id for candidate in candidates]
        values = {0: 3.0, 1: 5.0, 2: 2.0}
        return tuple(Score(item, values[item]) for item in self.received)


def vector_query() -> Query:
    return Query(
        "query", multi_vector=Feature(REPRESENTATION, np.ones((1, 2), dtype=np.float32))
    )


def test_gatherer_and_reranker_compose_without_backend_dependencies() -> None:
    reranker = VectorReranker()
    result = SearchPipeline(TextGatherer(), reranker).search(
        vector_query(), gather_limit=3, limit=2
    )

    assert reranker.received == [2, 0, 1]
    assert [row.document_id for row in result.documents] == [1, 0]
    assert result.diagnostics == {
        "candidate_count": 3,
        "gatherer": "TextGatherer",
        "reranker": "VectorReranker",
        "score_semantics": "external-rerank",
    }


def test_gather_scores_do_not_leak_into_a_reranked_result() -> None:
    result = SearchPipeline(TextGatherer(), VectorReranker()).search(
        vector_query(), gather_limit=3, limit=3
    )
    assert [row.document_id for row in result.documents] == [1, 0, 2]


def test_without_a_reranker_gather_scores_rank_with_gather_rank_tie_break() -> None:
    result = SearchPipeline(TextGatherer()).search("query", gather_limit=3, limit=3)

    assert [row.document_id for row in result.documents] == [2, 0, 1]
    assert [row.score for row in result.documents] == [100.0, 10.0, 10.0]
    assert result.diagnostics["reranker"] is None
    assert result.diagnostics["score_semantics"] == "external-gather"


def test_stages_must_index_the_same_corpus() -> None:
    with pytest.raises(IncompatibleIndexError, match="generation"):
        SearchPipeline(TextGatherer(), VectorReranker(replace(CORPUS, generation=1)))


def test_an_unservable_query_fails_before_gathering() -> None:
    gatherer = TextGatherer()
    with pytest.raises(IncompatibleQueryError, match="'multi_vector'"):
        SearchPipeline(gatherer, VectorReranker()).search("query", gather_limit=3, limit=1)
    assert gatherer.calls == 0

    foreign = Query(
        "query",
        multi_vector=Feature(
            replace(REPRESENTATION, encoder="other"), np.ones((1, 2), dtype=np.float32)
        ),
    )
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        SearchPipeline(gatherer, VectorReranker()).search(foreign, gather_limit=3, limit=1)
    assert gatherer.calls == 0


def test_subset_reaches_the_gatherer_as_ascending_ids() -> None:
    gatherer = TextGatherer()
    result = SearchPipeline(gatherer).search(
        "query", gather_limit=3, limit=2, subset=np.asarray([0, 2], dtype=np.int64)
    )
    assert gatherer.subsets[0].tolist() == [0, 2]
    assert [row.document_id for row in result.documents] == [2, 0]


def test_subset_must_be_ascending_and_inside_the_corpus() -> None:
    gatherer = TextGatherer()
    for subset in ([2, 0], [0, 3]):
        with pytest.raises(ValueError, match="subset"):
            SearchPipeline(gatherer).search("query", gather_limit=3, limit=1, subset=subset)
    assert gatherer.calls == 0


def test_a_gatherer_that_ignores_the_subset_is_refused() -> None:
    class IgnoringGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return super().gather(query, limit)

    with pytest.raises(ValueError, match="outside the subset"):
        SearchPipeline(IgnoringGatherer()).search(
            "query", gather_limit=3, limit=1, subset=np.asarray([0], dtype=np.int64)
        )


def test_gatherer_exceptions_reach_the_caller_unchanged() -> None:
    class FailingGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            raise KeyError("engine unavailable")

    with pytest.raises(KeyError, match="engine unavailable"):
        SearchPipeline(FailingGatherer()).search("query", gather_limit=3, limit=1)


def test_native_ranking_rejects_reranker_candidate_drift() -> None:
    class BrokenReranker(VectorReranker):
        def rerank(self, query, candidates, *, budget):  # type: ignore[no-untyped-def]
            return ()

    with pytest.raises(ValueError, match="omitted candidate"):
        SearchPipeline(TextGatherer(), BrokenReranker()).search(
            vector_query(), gather_limit=3, limit=2
        )


def test_pipeline_rejects_noncanonical_gather_ranks() -> None:
    class BrokenGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return (Candidate(0, 1.0, 4, "broken"),)

    with pytest.raises(ValueError, match="contiguous and zero-based"):
        SearchPipeline(BrokenGatherer()).search("query", gather_limit=1, limit=1)


def test_packed_maxsim_matches_reference_with_bounded_batches() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5], [-1.0, 0.0]], dtype=np.float32
    )
    lengths = np.asarray([2, 1, 1], dtype=np.int64)

    scores = maxsim_scores_packed(query, documents, lengths, max_batch_tokens=2, threads=2)

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
        query, documents, lengths, max_batch_tokens=6, threads=threads
    )

    assert observed.tolist() == pytest.approx(expected, abs=2e-5)
