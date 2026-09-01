from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lateweave import (
    Candidate,
    IndexManifest,
    Int8VectorStore,
    JzipVectorStore,
    Query,
    ResourceBudget,
    StoredMaxSimScorer,
    TurboQuantVectorStore,
    open_vector_store,
)


def normalized(values: list[list[float]]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return output


def manifest(store, document_count: int) -> IndexManifest:  # type: ignore[no-untyped-def]
    return IndexManifest(
        corpus_id="laws",
        corpus_version="1",
        document_count=document_count,
        document_ids_sha256="abc",
        encoder="encoder",
        encoder_revision="1",
        tokenizer="tokenizer",
        dimension=store.dimension,
        dtype="float32",
        normalized=True,
        representation=store.format,
        score_semantics=store.score_semantics,
    )


@pytest.mark.parametrize(
    ("store_type", "expected_bytes"),
    [(Int8VectorStore, 12), (TurboQuantVectorStore, 4), (JzipVectorStore, None)],
)
def test_vector_stores_own_encoding_and_scoring_transition(
    tmp_path, store_type, expected_bytes  # type: ignore[no-untyped-def]
) -> None:
    documents = normalized(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    lengths = np.asarray([2, 1, 1], dtype=np.int64)
    store = store_type.create(tmp_path / "vectors", documents, lengths, threads=1)
    reopened = open_vector_store(tmp_path / "vectors")

    assert type(reopened) is store_type
    if expected_bytes is not None:
        assert reopened.encoded_bytes_per_token == expected_bytes
    else:
        assert reopened.encoded_bytes_per_token > 0
    assert reopened.document_lengths([2, 0]) == {2: 1, 0: 2}
    packed, packed_lengths = reopened.transition([2, 0], threads=1)
    assert packed.shape == (3, 8)
    assert packed_lengths.tolist() == [1, 2]
    assert np.linalg.norm(packed, axis=1) == pytest.approx(np.ones(3), abs=1e-6)

    query = normalized([[1, 0, 0, 0, 0, 0, 0, 0]])
    prepared = reopened.prepare_query(query, threads=1)
    assert prepared.shape == query.shape
    assert (prepared @ packed[1]).item() > (prepared @ packed[0]).item()


@pytest.mark.parametrize(
    "store_type", [Int8VectorStore, TurboQuantVectorStore, JzipVectorStore]
)
def test_vector_store_append_and_delete_are_representation_owned(
    tmp_path, store_type  # type: ignore[no-untyped-def]
) -> None:
    initial = normalized(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    store = store_type.create(
        tmp_path / "vectors",
        initial,
        np.asarray([1, 2, 1], dtype=np.int64),
        threads=1,
    )
    addition = normalized([[0, 0, 0, 0, 1, 0, 0, 0]])
    store.append(addition, np.asarray([1], dtype=np.int64), threads=1)

    assert store.document_count == 4
    assert store.token_count == 5
    store.delete([1])
    assert store.document_count == 3
    assert store.token_count == 3
    assert store.document_lengths([0, 1, 2]) == {0: 1, 1: 1, 2: 1}
    reopened = open_vector_store(tmp_path / "vectors")
    assert reopened.document_count == 3
    assert reopened.token_count == 3


def test_jzip_is_near_lossless_and_compresses_realistic_documents(tmp_path) -> None:
    random = np.random.default_rng(7)
    documents = random.normal(size=(200, 128)).astype(np.float32)
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    lengths = np.asarray([100, 100], dtype=np.int64)
    store = JzipVectorStore.create(
        tmp_path / "vectors",
        documents,
        lengths,
        compression_level=1,
        threads=2,
    )

    reconstructed, reconstructed_lengths = store.transition([1, 0], threads=2)
    expected = np.concatenate((documents[100:], documents[:100]), axis=0)
    cosine = np.sum(expected * reconstructed, axis=1)
    assert reconstructed_lengths.tolist() == [100, 100]
    assert cosine == pytest.approx(np.ones(200), abs=3e-7)
    assert store.encoded_bytes_per_token < documents.shape[1] * 4


def test_jzip_delete_compacts_frames_without_reencoding(tmp_path) -> None:
    random = np.random.default_rng(11)
    documents = random.normal(size=(30, 32)).astype(np.float32)
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    store = JzipVectorStore.create(
        tmp_path / "vectors",
        documents,
        np.asarray([10, 10, 10], dtype=np.int64),
        threads=1,
    )
    before = (tmp_path / "vectors" / "frames.bin").stat().st_size

    store.delete([1], copy_chunk_tokens=17)

    after = (tmp_path / "vectors" / "frames.bin").stat().st_size
    reconstructed, lengths = store.transition([0, 1], threads=1)
    assert after < before
    assert lengths.tolist() == [10, 10]
    assert np.sum(reconstructed[:10] * documents[:10], axis=1) == pytest.approx(
        np.ones(10), abs=3e-7
    )
    assert np.sum(reconstructed[10:] * documents[20:], axis=1) == pytest.approx(
        np.ones(10), abs=3e-7
    )


def test_jzip_rejects_corrupt_candidate_frame(tmp_path) -> None:
    values = normalized([[1, 0, 0, 0], [0, 1, 0, 0]])
    store = JzipVectorStore.create(
        tmp_path / "vectors", values, np.asarray([1, 1], dtype=np.int64)
    )
    with (tmp_path / "vectors" / "frames.bin").open("r+b") as handle:
        handle.write(b"BAD!")

    with pytest.raises(ValueError, match="magic"):
        store.transition([0], threads=1)


def test_stored_maxsim_scorer_is_optional_and_store_specific(tmp_path) -> None:
    documents = normalized(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    store = Int8VectorStore.create(
        tmp_path / "vectors",
        documents,
        np.asarray([2, 1], dtype=np.int64),
        threads=1,
    )
    scorer = StoredMaxSimScorer(store, manifest(store, 2))
    scores = scorer.score(
        Query("query", embeddings=normalized([[1, 0, 0, 0, 0, 0, 0, 0]])),
        (Candidate(1, 2.0, 0, "test"), Candidate(0, 1.0, 1, "test")),
        budget=ResourceBudget(threads=1),
    )

    assert [score.document_id for score in scores] == [1, 0]
    assert scores[0].value == pytest.approx(2**-0.5, abs=1e-2)
    assert scores[1].value == pytest.approx(1.0, abs=1e-6)


def test_scorer_rejects_store_manifest_representation_mismatch(tmp_path) -> None:
    values = normalized([[1, 0, 0, 0, 0, 0, 0, 0]])
    store = Int8VectorStore.create(
        tmp_path / "vectors", values, np.asarray([1], dtype=np.int64)
    )

    with pytest.raises(ValueError, match="representations"):
        StoredMaxSimScorer(
            store, replace(manifest(store, 1), representation="native-engine")
        )
