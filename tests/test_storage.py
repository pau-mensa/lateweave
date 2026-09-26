from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lateweave import (
    Candidate,
    CorpusManifest,
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
    open_vector_store,
)


REPRESENTATION = Representation("encoder", "1", 8, True)
STORES = [Float32VectorStore, Int8VectorStore]


def normalized(values: list[list[float]]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return output


def corpus(document_count: int) -> CorpusManifest:
    return CorpusManifest("corpus", "1", document_count, "abc")


DOCUMENTS = normalized(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0],
    ]
)
LENGTHS = np.asarray([2, 1, 1], dtype=np.int64)


@pytest.mark.parametrize("store_type", STORES)
def test_stores_reopen_and_fetch_in_requested_order(tmp_path, store_type) -> None:
    store_type.create(tmp_path / "vectors", DOCUMENTS, LENGTHS, REPRESENTATION, threads=1)
    store = open_vector_store(tmp_path / "vectors")

    assert type(store) is store_type
    assert isinstance(store, MultiVectorSource)
    assert store.representation == REPRESENTATION
    assert store.document_lengths([2, 0]) == {2: 1, 0: 2}
    packed, lengths = store.fetch([2, 0], threads=1)
    assert packed.shape == (3, 8) and packed.dtype == np.float32
    assert lengths.tolist() == [1, 2]
    assert np.linalg.norm(packed, axis=1) == pytest.approx(np.ones(3), abs=1e-6)
    assert packed[0] @ DOCUMENTS[3] > 0.99


def test_float32_store_is_bit_exact(tmp_path) -> None:
    store = Float32VectorStore.create(tmp_path / "vectors", DOCUMENTS, LENGTHS, REPRESENTATION)
    packed, _ = store.fetch([0, 1, 2])
    assert np.array_equal(packed, DOCUMENTS)


@pytest.mark.parametrize("store_type", STORES)
def test_append_and_delete_compact_internal_ids(tmp_path, store_type) -> None:
    initial = normalized([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])
    store = store_type.create(
        tmp_path / "vectors", initial, np.asarray([1, 2, 1], dtype=np.int64), REPRESENTATION
    )
    store.append(normalized([[0, 0, 0, 0, 1, 0, 0, 0]]), np.asarray([1], dtype=np.int64))
    assert (store.document_count, store.token_count) == (4, 5)

    store.delete([1])
    assert (store.document_count, store.token_count) == (3, 3)
    packed, lengths = store.fetch([0, 1, 2])
    assert lengths.tolist() == [1, 1, 1]
    assert np.argmax(packed, axis=1).tolist() == [0, 3, 4]
    reopened = open_vector_store(tmp_path / "vectors")
    assert (reopened.document_count, reopened.token_count) == (3, 3)


def test_store_refuses_vectors_that_contradict_the_representation(tmp_path) -> None:
    with pytest.raises(ValueError, match="dimension"):
        Float32VectorStore.create(
            tmp_path / "a", DOCUMENTS, LENGTHS, replace(REPRESENTATION, dimension=4)
        )
    with pytest.raises(ValueError, match="normalized"):
        Float32VectorStore.create(tmp_path / "b", DOCUMENTS * 2, LENGTHS, REPRESENTATION)


class InMemorySource:
    """A source an engine could expose from its own index; no store involved."""

    representation = REPRESENTATION
    score_semantics = "in-memory-exact-full-maxsim"

    def __init__(self, documents: list[np.ndarray]) -> None:
        self.documents = documents
        self.document_count = len(documents)
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
    reranker = MaxSimReranker(source, corpus(3))
    assert reranker.requires == {"multi_vector": REPRESENTATION}
    assert reranker.score_semantics == "in-memory-exact-full-maxsim"

    scores = reranker.rerank(
        query(normalized([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])),
        (Candidate(1, 2.0, 0, "t"), Candidate(0, 1.0, 1, "t"), Candidate(2, 0.5, 2, "t")),
        budget=ResourceBudget(max_batch_tokens=2, threads=1),
    )

    assert [score.document_id for score in scores] == [1, 0, 2]
    assert [score.value for score in scores] == pytest.approx([2**0.5, 2.0, -1.0], abs=1e-6)
    assert sorted(map(len, source.fetched)) == [1, 2]


@pytest.mark.parametrize(("store_type", "tolerance"), [(Float32VectorStore, 1e-6), (Int8VectorStore, 1e-2)])
def test_reranker_over_a_store_reports_the_stores_fidelity(tmp_path, store_type, tolerance) -> None:
    store = store_type.create(tmp_path / "vectors", DOCUMENTS, LENGTHS, REPRESENTATION)
    reranker = MaxSimReranker(store, corpus(3))
    assert reranker.score_semantics == store_type.score_semantics

    scores = reranker.rerank(
        query(normalized([[1, 0, 0, 0, 0, 0, 0, 0]])),
        (Candidate(1, 2.0, 0, "t"), Candidate(0, 1.0, 1, "t")),
        budget=ResourceBudget(threads=1),
    )
    assert [score.value for score in scores] == pytest.approx([2**-0.5, 1.0], abs=tolerance)


def test_reranker_refuses_a_query_feature_from_another_encoder(tmp_path) -> None:
    store = Float32VectorStore.create(tmp_path / "vectors", DOCUMENTS, LENGTHS, REPRESENTATION)
    reranker = MaxSimReranker(store, corpus(3))
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        reranker.rerank(
            query(DOCUMENTS[:1], replace(REPRESENTATION, encoder="other")),
            (Candidate(0, 1.0, 0, "t"),),
            budget=ResourceBudget(),
        )


def test_reranker_refuses_a_source_that_disagrees_with_the_corpus(tmp_path) -> None:
    store = Float32VectorStore.create(tmp_path / "vectors", DOCUMENTS, LENGTHS, REPRESENTATION)
    with pytest.raises(ValueError, match="document counts"):
        MaxSimReranker(store, corpus(99))


@pytest.mark.parametrize("store_type", STORES)
def test_stores_accept_non_contiguous_embeddings(tmp_path, store_type) -> None:
    doubled = np.repeat(DOCUMENTS, 2, axis=0)[::2]
    assert not doubled.flags.c_contiguous
    store = store_type.create(tmp_path / "vectors", doubled, LENGTHS, REPRESENTATION)
    store.append(np.asfortranarray(DOCUMENTS[:1]), [1])
    expected = store_type.create(tmp_path / "expected", DOCUMENTS, LENGTHS, REPRESENTATION)
    expected.append(DOCUMENTS[:1], [1])
    np.testing.assert_array_equal(store.fetch([0, 1, 2, 3])[0], expected.fetch([0, 1, 2, 3])[0])


def test_a_reranker_refuses_a_store_mutated_after_it_was_built(tmp_path) -> None:
    store = Float32VectorStore.create(tmp_path / "vectors", DOCUMENTS, LENGTHS, REPRESENTATION)
    reranker = MaxSimReranker(store, corpus(3))
    query = Query("query", multi_vector=Feature(REPRESENTATION, DOCUMENTS[:1]))
    candidates = [Candidate(1, 0.0, 0, "test")]
    reranker.rerank(query, candidates, budget=ResourceBudget())
    store.delete([0])
    store.append(DOCUMENTS[:1], [1])
    with pytest.raises(IncompatibleIndexError, match="mutated"):
        reranker.rerank(query, candidates, budget=ResourceBudget())
