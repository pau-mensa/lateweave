from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lateweave import (
    Candidate,
    Feature,
    Float32VectorStore,
    IncompatibleIndexError,
    IncompatibleQueryError,
    Int8VectorStore,
    MaxSimReranker,
    MultiVectorSource,
    Query,
    Representation,
    ResourceBudget,
    SearchPipeline,
    Segment,
    StoreSnapshot,
    open_vector_store,
)


REPRESENTATION = Representation("encoder", "1", 8, True)
STORES = [Float32VectorStore, Int8VectorStore]


def normalized(values: list[list[float]]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return output


def segment(*ids: str, corpus_id: str = "corpus") -> Segment:
    return Segment(corpus_id, "1", ids or ("a", "b", "c"))


DOCUMENTS = normalized(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0],
    ]
)
LENGTHS = np.asarray([2, 1, 1], dtype=np.int64)


def create(store_type, path, ids=("a", "b", "c"), documents=DOCUMENTS, lengths=LENGTHS):  # type: ignore[no-untyped-def]
    return store_type.create(path, segment(*ids), documents, lengths, REPRESENTATION)


@pytest.mark.parametrize("store_type", STORES)
def test_stores_reopen_and_fetch_in_requested_order(tmp_path, store_type) -> None:
    store_type.create(tmp_path / "vectors", segment(), DOCUMENTS, LENGTHS, REPRESENTATION, threads=1)
    store = open_vector_store(tmp_path / "vectors")

    assert type(store) is store_type
    assert store.segment == segment()
    snapshot = store.snapshot()
    assert isinstance(snapshot, StoreSnapshot) and isinstance(snapshot, MultiVectorSource)
    assert snapshot.representation == REPRESENTATION
    assert snapshot.score_semantics == store_type.score_semantics
    assert snapshot.document_lengths([2, 0]) == {2: 1, 0: 2}
    packed, lengths = snapshot.fetch([2, 0], threads=1)
    assert packed.shape == (3, 8) and packed.dtype == np.float32
    assert lengths.tolist() == [1, 2]
    assert np.linalg.norm(packed, axis=1) == pytest.approx(np.ones(3), abs=1e-6)
    assert packed[0] @ DOCUMENTS[3] > 0.99


def test_float32_store_is_bit_exact(tmp_path) -> None:
    store = create(Float32VectorStore, tmp_path / "vectors")
    packed, _ = store.snapshot().fetch([0, 1, 2])
    assert np.array_equal(packed, DOCUMENTS)


@pytest.mark.parametrize("store_type", STORES)
def test_mutations_take_external_ids_and_follow_the_segment(tmp_path, store_type) -> None:
    initial = normalized(
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]]
    )
    store = create(store_type, tmp_path / "vectors", documents=initial, lengths=[1, 2, 1])
    appended = store.append(["d"], normalized([[0, 0, 0, 0, 1, 0, 0, 0]]), [1])
    assert appended.segment == segment().appended(["d"])
    assert (appended.document_count, appended.token_count) == (4, 5)

    deleted = store.delete(["b"])
    assert deleted.segment == appended.segment.deleted(["b"])
    assert deleted.segment.document_ids == ["a", "c", "d"]
    assert (deleted.document_count, deleted.token_count) == (3, 3)
    packed, lengths = deleted.fetch([0, 1, 2])
    assert lengths.tolist() == [1, 1, 1]
    assert np.argmax(packed, axis=1).tolist() == [0, 3, 4]
    assert open_vector_store(tmp_path / "vectors").segment == deleted.segment

    with pytest.raises(ValueError, match="not in segment"):
        store.delete(["b"])
    with pytest.raises(ValueError, match="appears more than once"):
        store.append(["a"], DOCUMENTS[:1], [1])
    with pytest.raises(TypeError, match="single string"):
        store.delete("a")


def test_a_snapshot_keeps_reading_its_generation_after_mutations(tmp_path) -> None:
    store = create(Float32VectorStore, tmp_path / "vectors", ids=("a", "b"), documents=DOCUMENTS[:2], lengths=[1, 1])
    before = store.snapshot()
    store.delete(["a"])
    store.append(["c"], DOCUMENTS[2:3], [1])

    # Same document count, but internal ID 1 now names "c".
    after = store.snapshot()
    assert before.segment.external(1) == "b" and after.segment.external(1) == "c"
    np.testing.assert_array_equal(before.fetch([1])[0], DOCUMENTS[1:2])
    np.testing.assert_array_equal(after.fetch([1])[0], DOCUMENTS[2:3])


class InMemorySource:
    """A source an engine could expose from its own index; no store involved."""

    representation = REPRESENTATION
    score_semantics = "in-memory-exact-full-maxsim"

    def __init__(self, documents: list[np.ndarray], corpus_id: str = "corpus") -> None:
        self.documents = documents
        self.segment = Segment(corpus_id, "1", [str(item) for item in range(len(documents))])
        self.fetched: list[list[int]] = []

    def document_lengths(self, document_ids):  # type: ignore[no-untyped-def]
        return {item: len(self.documents[item]) for item in document_ids}

    def fetch(self, document_ids, *, threads=None):  # type: ignore[no-untyped-def]
        self.fetched.append(list(document_ids))
        rows = [self.documents[item] for item in document_ids]
        return np.concatenate(rows), np.asarray([len(row) for row in rows], dtype=np.int64)


def query(vectors: np.ndarray, representation: Representation = REPRESENTATION) -> Query:
    return Query("query", multi_vector=Feature(representation, vectors))


def test_reranker_scores_a_borrowed_source_without_a_store() -> None:
    source = InMemorySource([DOCUMENTS[:2], DOCUMENTS[2:3], DOCUMENTS[3:]])
    assert isinstance(source, MultiVectorSource)
    reranker = MaxSimReranker([source])
    assert reranker.requires == {"multi_vector": REPRESENTATION}
    assert reranker.score_semantics == "in-memory-exact-full-maxsim"
    assert reranker.segments == (source.segment,)
    assert reranker.sources == (source,)

    segment = source.segment
    scores = reranker.rerank(
        query(normalized([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])),
        (Candidate(segment, 1, 2.0, 0, "t"), Candidate(segment, 0, 1.0, 1, "t"), Candidate(segment, 2, 0.5, 2, "t")),
        budget=ResourceBudget(max_batch_tokens=2, threads=1),
    )

    assert scores.tolist() == pytest.approx([2**0.5, 2.0, -1.0], abs=1e-6)
    assert sorted(map(len, source.fetched)) == [1, 2]


@pytest.mark.parametrize(("store_type", "tolerance"), [(Float32VectorStore, 1e-6), (Int8VectorStore, 1e-2)])
def test_reranker_over_a_snapshot_reports_the_stores_fidelity(tmp_path, store_type, tolerance) -> None:
    snapshot = create(store_type, tmp_path / "vectors").snapshot()
    reranker = MaxSimReranker([snapshot])
    assert reranker.score_semantics == store_type.score_semantics

    scores = reranker.rerank(
        query(normalized([[1, 0, 0, 0, 0, 0, 0, 0]])),
        (Candidate(snapshot.segment, 1, 2.0, 0, "t"), Candidate(snapshot.segment, 0, 1.0, 1, "t")),
        budget=ResourceBudget(threads=1),
    )
    assert scores.tolist() == pytest.approx([2**-0.5, 1.0], abs=tolerance)


def test_one_reranker_scores_two_stores_through_a_fused_gatherer(tmp_path) -> None:
    laws = Float32VectorStore.create(
        tmp_path / "laws", segment("l0", "l1", corpus_id="laws"), DOCUMENTS[:2], [1, 1], REPRESENTATION
    ).snapshot()
    cases = Float32VectorStore.create(
        tmp_path / "cases", segment("c0", "c1", corpus_id="cases"), DOCUMENTS[2:], [1, 1], REPRESENTATION
    ).snapshot()

    class Fused:
        requires: dict[str, Representation] = {}
        score_semantics = "unranked-union"
        segments = (laws.segment, cases.segment)

        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            rows = [(laws.segment, 0), (cases.segment, 1), (laws.segment, 1), (cases.segment, 0)]
            return tuple(Candidate(s, i, 0.0, rank, s.corpus_id) for rank, (s, i) in enumerate(rows))

    result = SearchPipeline(Fused(), MaxSimReranker([laws, cases])).search(
        query(DOCUMENTS[:1]), gather_limit=4, limit=4
    )
    # l0 is the query itself; c0 = (1,1)/√2; l1 is orthogonal; c1 is its opposite.
    assert [row.external_id for row in result.documents] == ["l0", "c0", "l1", "c1"]


def test_reranker_refuses_a_query_feature_from_another_encoder(tmp_path) -> None:
    snapshot = create(Float32VectorStore, tmp_path / "vectors").snapshot()
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        MaxSimReranker([snapshot]).rerank(
            query(DOCUMENTS[:1], replace(REPRESENTATION, encoder="other")),
            (Candidate(snapshot.segment, 0, 1.0, 0, "t"),),
            budget=ResourceBudget(),
        )


def test_reranker_sources_must_agree(tmp_path) -> None:
    exact = create(Float32VectorStore, tmp_path / "a").snapshot()
    lossy = Int8VectorStore.create(
        tmp_path / "b", segment(corpus_id="other"), DOCUMENTS, LENGTHS, REPRESENTATION
    ).snapshot()
    with pytest.raises(IncompatibleIndexError, match="scores as"):
        MaxSimReranker([exact, lossy])
    with pytest.raises(ValueError, match="more than one source"):
        MaxSimReranker([exact, create(Float32VectorStore, tmp_path / "c").snapshot()])
    with pytest.raises(ValueError, match="at least one source"):
        MaxSimReranker([])


@pytest.mark.parametrize("store_type", STORES)
def test_stores_accept_non_contiguous_embeddings(tmp_path, store_type) -> None:
    doubled = np.repeat(DOCUMENTS, 2, axis=0)[::2]
    assert not doubled.flags.c_contiguous
    store = create(store_type, tmp_path / "vectors", documents=doubled)
    store.append(["d"], np.asfortranarray(DOCUMENTS[:1]), [1])
    expected = create(store_type, tmp_path / "expected")
    expected.append(["d"], DOCUMENTS[:1], [1])
    np.testing.assert_array_equal(
        store.snapshot().fetch([0, 1, 2, 3])[0], expected.snapshot().fetch([0, 1, 2, 3])[0]
    )


def test_a_reranker_holds_its_snapshot_across_store_mutations(tmp_path) -> None:
    store = create(Float32VectorStore, tmp_path / "vectors")
    snapshot = store.snapshot()
    reranker = MaxSimReranker([snapshot])
    vector_query = Query("query", multi_vector=Feature(REPRESENTATION, DOCUMENTS[2:3]))
    candidates = [Candidate(snapshot.segment, 1, 0.0, 0, "test")]
    before = reranker.rerank(vector_query, candidates, budget=ResourceBudget())

    store.delete(["a"])
    after = store.append(["d"], DOCUMENTS[:1], [1])
    np.testing.assert_array_equal(reranker.rerank(vector_query, candidates, budget=ResourceBudget()), before)
    with pytest.raises(IncompatibleIndexError, match="generation"):
        reranker.rerank(
            vector_query, [Candidate(after.segment, 1, 0.0, 0, "test")], budget=ResourceBudget()
        )


def test_store_refuses_vectors_that_contradict_the_representation_or_segment(tmp_path) -> None:
    with pytest.raises(ValueError, match="dimension"):
        Float32VectorStore.create(
            tmp_path / "a", segment(), DOCUMENTS, LENGTHS, replace(REPRESENTATION, dimension=4)
        )
    with pytest.raises(ValueError, match="normalized"):
        Float32VectorStore.create(tmp_path / "b", segment(), DOCUMENTS * 2, LENGTHS, REPRESENTATION)
    with pytest.raises(ValueError, match="segment holds 2"):
        Float32VectorStore.create(tmp_path / "c", segment("a", "b"), DOCUMENTS, LENGTHS, REPRESENTATION)
