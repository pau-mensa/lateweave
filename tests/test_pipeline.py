from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lateweave import (
    Candidate,
    Feature,
    Gathered,
    IncompatibleQueryError,
    Query,
    Representation,
    ResourceBudget,
    Scored,
    SearchPipeline,
    StaleError,
    maxsim_scores_packed,
)


REPRESENTATION = Representation("encoder", "1", 2, True)
NOW = datetime.now(timezone.utc)
ROWS = (("corpus", "z", 100.0), ("corpus", "x", 10.0), ("corpus", "y", 10.0))


class TextGatherer:
    requires: dict[str, Representation] = {}
    score_semantics = "external-gather"

    def __init__(self, rows=ROWS, as_of=NOW) -> None:  # type: ignore[no-untyped-def]
        self.rows = rows
        self.as_of = as_of
        self.calls = 0
        self.subsets: list[dict[str, frozenset[str]] | None] = []

    def gather(self, query: Query, limit: int, *, subset=None) -> Gathered:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.subsets.append(subset)
        rows = [
            row for row in self.rows if subset is None or row[1] in subset.get(row[0], frozenset())
        ]
        return Gathered(
            [
                Candidate(corpus, document_id, score, rank, "external")
                for rank, (corpus, document_id, score) in enumerate(rows[:limit])
            ],
            self.as_of,
        )


class TableReranker:
    requires = {"multi_vector": REPRESENTATION}
    score_semantics = "table"

    def __init__(self, table: dict[str, float], as_of: datetime = NOW) -> None:
        self.table = table
        self.as_of = as_of
        self.received: list[str] = []

    def rerank(self, query, candidates, *, budget: ResourceBudget) -> Scored:  # type: ignore[no-untyped-def]
        assert query.feature("multi_vector", REPRESENTATION).shape == (1, 2)
        self.received = [candidate.document_id for candidate in candidates]
        return Scored([self.table.get(item) for item in self.received], self.as_of)


def vector_query() -> Query:
    return Query(
        "query", multi_vector=Feature(REPRESENTATION, np.ones((1, 2), dtype=np.float32))
    )


def ids(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [row.document_id for row in result.documents]


def test_gatherer_and_reranker_compose_by_document_id() -> None:
    reranker = TableReranker({"x": 3.0, "y": 5.0, "z": 2.0})
    result = SearchPipeline(TextGatherer(), reranker).search(vector_query(), gather_limit=3, limit=2)

    assert reranker.received == ["z", "x", "y"]
    assert ids(result) == ["y", "x"]
    assert [row.corpus for row in result.documents] == ["corpus", "corpus"]
    assert result.scores == [2.0, 3.0, 5.0]
    assert result.as_of == NOW
    assert result.diagnostics == {
        "candidate_count": 3,
        "dropped": 0,
        "gatherer": "TextGatherer",
        "reranker": "TableReranker",
        "score_semantics": "table",
    }


def test_a_document_the_reranker_lacks_is_dropped() -> None:
    result = SearchPipeline(TextGatherer(), TableReranker({"x": 3.0, "z": 2.0})).search(
        vector_query(), gather_limit=3, limit=3
    )
    assert ids(result) == ["x", "z"]
    assert result.scores == [2.0, 3.0, None]
    assert result.diagnostics["dropped"] == 1


def test_the_result_is_as_fresh_as_its_oldest_stage() -> None:
    older = NOW - timedelta(minutes=5)
    result = SearchPipeline(TextGatherer(), TableReranker({"x": 1.0}, as_of=older)).search(
        vector_query(), gather_limit=3, limit=1
    )
    assert result.as_of == older


def test_max_lag_refuses_indexes_older_than_it() -> None:
    gatherer = TextGatherer(as_of=datetime.now(timezone.utc) - timedelta(minutes=2))
    pipeline = SearchPipeline(gatherer)
    assert ids(pipeline.search("query", gather_limit=3, limit=1, max_lag=timedelta(minutes=5))) == ["z"]
    with pytest.raises(StaleError, match="allowed"):
        pipeline.search("query", gather_limit=3, limit=1, max_lag=timedelta(minutes=1))


def test_without_a_reranker_gather_scores_rank_with_gather_rank_tie_break() -> None:
    result = SearchPipeline(TextGatherer()).search("query", gather_limit=3, limit=3)
    assert ids(result) == ["z", "x", "y"]
    assert [row.score for row in result.documents] == [100.0, 10.0, 10.0]
    assert result.diagnostics["reranker"] is None
    assert result.diagnostics["score_semantics"] == "external-gather"


def test_one_gatherer_ranks_candidates_from_two_corpora() -> None:
    gatherer = TextGatherer(
        [("laws", "l0", 0.0), ("cases", "c1", 0.0), ("laws", "l2", 0.0), ("cases", "c0", 0.0)]
    )

    class ByKey(TableReranker):
        def rerank(self, query, candidates, *, budget):  # type: ignore[no-untyped-def]
            table = {("laws", "l0"): 1.0, ("laws", "l2"): 4.0, ("cases", "c0"): 3.0, ("cases", "c1"): 2.0}
            return Scored([table[(c.corpus, c.document_id)] for c in candidates], NOW)

    result = SearchPipeline(gatherer, ByKey({})).search(vector_query(), gather_limit=4, limit=4)
    assert [(row.corpus, row.document_id) for row in result.documents] == [
        ("laws", "l2"),
        ("cases", "c0"),
        ("cases", "c1"),
        ("laws", "l0"),
    ]


def test_an_unservable_query_fails_before_gathering() -> None:
    gatherer = TextGatherer()
    with pytest.raises(IncompatibleQueryError, match="'multi_vector'"):
        SearchPipeline(gatherer, TableReranker({})).search("query", gather_limit=3, limit=1)
    foreign = Query(
        "query",
        multi_vector=Feature(Representation("other", "1", 2, True), np.ones((1, 2), dtype=np.float32)),
    )
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        SearchPipeline(gatherer, TableReranker({})).search(foreign, gather_limit=3, limit=1)
    assert gatherer.calls == 0


def test_the_subset_reaches_the_gatherer_as_frozensets_per_corpus() -> None:
    gatherer = TextGatherer()
    result = SearchPipeline(gatherer).search(
        "query", gather_limit=3, limit=3, subset={"corpus": ["x", "z", "x", "unknown"]}
    )
    assert gatherer.subsets[0] == {"corpus": frozenset({"x", "z", "unknown"})}
    assert ids(result) == ["z", "x"]
    with pytest.raises(TypeError, match="single string"):
        SearchPipeline(gatherer).search("query", gather_limit=3, limit=1, subset={"corpus": "x"})
    with pytest.raises(TypeError, match="corpora"):
        SearchPipeline(gatherer).search("query", gather_limit=3, limit=1, subset=["x"])


def test_a_gatherer_that_ignores_the_subset_is_refused() -> None:
    class IgnoringGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return super().gather(query, limit)

    with pytest.raises(ValueError, match="outside the subset"):
        SearchPipeline(IgnoringGatherer()).search("query", gather_limit=3, limit=1, subset={"corpus": ["x"]})


@pytest.mark.parametrize("limits", [(-1, 1), (3, -1)])
def test_negative_limits_are_value_errors(limits) -> None:  # type: ignore[no-untyped-def]
    gather_limit, limit = limits
    with pytest.raises(ValueError, match="must be positive"):
        SearchPipeline(TextGatherer()).search("query", gather_limit=gather_limit, limit=limit)


def test_gatherer_exceptions_reach_the_caller_unchanged() -> None:
    class FailingGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            raise KeyError("engine unavailable")

    with pytest.raises(KeyError, match="engine unavailable"):
        SearchPipeline(FailingGatherer()).search("query", gather_limit=3, limit=1)


def test_stage_results_must_be_gathered_and_scored_values() -> None:
    class ListGatherer(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return super().gather(query, limit).candidates

    with pytest.raises(TypeError):
        SearchPipeline(ListGatherer()).search("query", gather_limit=3, limit=1)
    with pytest.raises(TypeError):
        Gathered([], datetime(2026, 1, 1))


def test_native_ranking_requires_one_score_per_candidate() -> None:
    class BrokenReranker(TableReranker):
        def rerank(self, query, candidates, *, budget):  # type: ignore[no-untyped-def]
            return Scored([1.0], NOW)

    with pytest.raises(ValueError, match="1 scores were returned for 3 candidates"):
        SearchPipeline(TextGatherer(), BrokenReranker({})).search(vector_query(), gather_limit=3, limit=2)


def test_pipeline_rejects_repeated_documents_and_noncanonical_ranks() -> None:
    class Repeating(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return Gathered([Candidate("corpus", "x", 1.0, 0, "t"), Candidate("corpus", "x", 1.0, 1, "t")], NOW)

    class Skipping(TextGatherer):
        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            return Gathered([Candidate("corpus", "x", 1.0, 4, "t")], NOW)

    with pytest.raises(ValueError, match="more than once"):
        SearchPipeline(Repeating()).search("query", gather_limit=3, limit=1)
    with pytest.raises(ValueError, match="contiguous and zero-based"):
        SearchPipeline(Skipping()).search("query", gather_limit=3, limit=1)


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

    observed = maxsim_scores_packed(query, documents, lengths, max_batch_tokens=6, threads=threads)

    assert observed.tolist() == pytest.approx(expected, abs=2e-5)
