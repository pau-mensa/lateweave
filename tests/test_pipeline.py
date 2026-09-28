from __future__ import annotations

import numpy as np
import pytest

from lateweave import (
    Candidate,
    Feature,
    IncompatibleIndexError,
    IncompatibleQueryError,
    Query,
    Representation,
    ResourceBudget,
    SearchPipeline,
    Segment,
    maxsim_scores_packed,
)


SEGMENT = Segment("corpus", "1", ["x", "y", "z"])
REPRESENTATION = Representation("encoder", "1", 2, True)


class TextGatherer:
    requires: dict[str, Representation] = {}
    score_semantics = "external-gather"

    def __init__(self, segment: Segment = SEGMENT) -> None:
        self.segments = (segment,)
        self.calls = 0
        self.subsets: list[dict[str, np.ndarray] | None] = []

    def gather(self, query: Query, limit: int, *, subset=None) -> tuple[Candidate, ...]:
        self.calls += 1
        self.subsets.append(subset)
        assert query.text == "query"
        (segment,) = self.segments
        rows = [(2, 100.0), (0, 10.0), (1, 10.0)]
        if subset is not None:
            rows = [row for row in rows if row[0] in subset[segment.corpus_id]]
        return tuple(
            Candidate(segment, document_id, gather_score, rank, "external")
            for rank, (document_id, gather_score) in enumerate(rows[:limit])
        )


class VectorReranker:
    requires = {"multi_vector": REPRESENTATION}
    score_semantics = "external-rerank"

    def __init__(self, *segments: Segment) -> None:
        self.segments = segments or (SEGMENT,)
        self.received: list[int] = []

    def rerank(self, query, candidates, *, budget: ResourceBudget) -> list[float]:
        assert query.feature("multi_vector", REPRESENTATION).shape == (1, 2)
        self.received = [candidate.document_id for candidate in candidates]
        values = {0: 3.0, 1: 5.0, 2: 2.0}
        return [values[item] for item in self.received]


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
    assert [row.external_id for row in result.documents] == ["y", "x"]
    assert result.documents[0].segment == SEGMENT
    assert result.scores.tolist() == [2.0, 3.0, 5.0]
    assert result.diagnostics == {
        "candidate_count": 3,
        "gatherer": "TextGatherer",
        "reranker": "VectorReranker",
        "score_semantics": "external-rerank",
        "stale": False,
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


class FusedGatherer:
    """Appends the hits of two segments; their scores are not comparable."""

    requires: dict[str, Representation] = {}
    score_semantics = "unranked-union"

    def __init__(self, left: Segment, right: Segment) -> None:
        self.segments = (left, right)

    def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
        left, right = self.segments
        rows = [(left, 0), (right, 1), (left, 2), (right, 0)]
        return tuple(
            Candidate(segment, document_id, 0.0, rank, segment.corpus_id)
            for rank, (segment, document_id) in enumerate(rows[:limit])
        )


def test_one_gatherer_ranks_candidates_from_two_segments() -> None:
    laws = Segment("laws", "1", ["l0", "l1", "l2"])
    cases = Segment("cases", "1", ["c0", "c1"])

    class TableReranker(VectorReranker):
        def rerank(self, query, candidates, *, budget):  # type: ignore[no-untyped-def]
            table = {"l0": 1.0, "l2": 4.0, "c0": 3.0, "c1": 2.0}
            return np.asarray([table[candidate.external_id] for candidate in candidates])

    result = SearchPipeline(FusedGatherer(laws, cases), TableReranker(laws, cases)).search(
        vector_query(), gather_limit=4, limit=4
    )

    assert [(row.corpus_id, row.external_id) for row in result.documents] == [
        ("laws", "l2"),
        ("cases", "c0"),
        ("cases", "c1"),
        ("laws", "l0"),
    ]
    assert [candidate.provenance for candidate in result.candidates] == [
        "laws", "cases", "laws", "cases",
    ]


def test_the_reranker_must_hold_every_segment_the_gatherer_searches() -> None:
    laws = Segment("laws", "1", ["l0"])
    cases = Segment("cases", "1", ["c0"])
    with pytest.raises(IncompatibleIndexError, match="'cases'|\"cases\""):
        SearchPipeline(FusedGatherer(laws, cases), VectorReranker(laws))
    with pytest.raises(IncompatibleIndexError, match="generation"):
        SearchPipeline(FusedGatherer(laws, cases), VectorReranker(laws, cases.appended(["c1"])))


def test_candidates_must_come_from_a_declared_snapshot() -> None:
    later = SEGMENT.appended(["w"])

    class StaleGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return (Candidate(later, 3, 1.0, 0, "stale"),)

    with pytest.raises(IncompatibleIndexError):
        SearchPipeline(StaleGatherer()).search("query", gather_limit=1, limit=1)


def test_an_unservable_query_fails_before_gathering() -> None:
    gatherer = TextGatherer()
    with pytest.raises(IncompatibleQueryError, match="'multi_vector'"):
        SearchPipeline(gatherer, VectorReranker()).search("query", gather_limit=3, limit=1)
    assert gatherer.calls == 0

    foreign = Query(
        "query",
        multi_vector=Feature(
            Representation("other", "1", 2, True), np.ones((1, 2), dtype=np.float32)
        ),
    )
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        SearchPipeline(gatherer, VectorReranker()).search(foreign, gather_limit=3, limit=1)
    assert gatherer.calls == 0


def test_subset_reaches_the_gatherer_per_segment_as_ascending_ids() -> None:
    gatherer = TextGatherer()
    result = SearchPipeline(gatherer).search(
        "query", gather_limit=3, limit=2, subset={"corpus": np.asarray([0, 2], dtype=np.int64)}
    )
    assert {key: value.tolist() for key, value in gatherer.subsets[0].items()} == {"corpus": [0, 2]}
    assert [row.document_id for row in result.documents] == [2, 0]


def test_a_segment_the_subset_omits_contributes_nothing() -> None:
    laws = Segment("laws", "1", ["l0", "l1", "l2"])
    cases = Segment("cases", "1", ["c0", "c1"])
    received = []

    class Recording(FusedGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            received.append({key: value.tolist() for key, value in subset.items()})
            return ()

    SearchPipeline(Recording(laws, cases)).search(
        "query", gather_limit=3, limit=1, subset={"cases": [1]}
    )
    assert received == [{"cases": [1], "laws": []}]


def test_subset_must_be_ascending_inside_a_known_segment() -> None:
    gatherer = TextGatherer()
    for subset in ({"corpus": [2, 0]}, {"corpus": [0, 3]}, {"other": [0]}, {"corpus": [-1]}):
        with pytest.raises(ValueError, match="subset"):
            SearchPipeline(gatherer).search("query", gather_limit=3, limit=1, subset=subset)
    with pytest.raises(TypeError, match="corpus IDs"):
        SearchPipeline(gatherer).search("query", gather_limit=3, limit=1, subset=[0])
    assert gatherer.calls == 0


def test_a_repeated_subset_id_reaches_the_gatherer_once() -> None:
    gatherer = TextGatherer()
    SearchPipeline(gatherer).search("query", gather_limit=3, limit=1, subset={"corpus": [0, 0, 2]})
    assert gatherer.subsets[0]["corpus"].tolist() == [0, 2]


@pytest.mark.parametrize("limits", [(-1, 1), (3, -1)])
def test_negative_limits_are_value_errors(limits) -> None:
    gather_limit, limit = limits
    with pytest.raises(ValueError, match="must be positive"):
        SearchPipeline(TextGatherer()).search("query", gather_limit=gather_limit, limit=limit)


def test_a_gatherer_that_ignores_the_subset_is_refused() -> None:
    class IgnoringGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return super().gather(query, limit)

    with pytest.raises(ValueError, match="outside the subset"):
        SearchPipeline(IgnoringGatherer()).search(
            "query", gather_limit=3, limit=1, subset={"corpus": [0]}
        )


def test_gatherer_exceptions_reach_the_caller_unchanged() -> None:
    class FailingGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            raise KeyError("engine unavailable")

    with pytest.raises(KeyError, match="engine unavailable"):
        SearchPipeline(FailingGatherer()).search("query", gather_limit=3, limit=1)


def test_native_ranking_requires_one_score_per_candidate() -> None:
    class BrokenReranker(VectorReranker):
        def rerank(self, query, candidates, *, budget):  # type: ignore[no-untyped-def]
            return [1.0]

    with pytest.raises(ValueError, match="1 scores were returned for 3 candidates"):
        SearchPipeline(TextGatherer(), BrokenReranker()).search(
            vector_query(), gather_limit=3, limit=2
        )


def test_pipeline_rejects_noncanonical_gather_ranks() -> None:
    class BrokenGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return (Candidate(SEGMENT, 0, 1.0, 4, "broken"),)

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
