from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from lateweave import (
    Candidate,
    Feature,
    Gathered,
    IncompatibleIndexError,
    IncompatibleQueryError,
    MaxSimReranker,
    MultiVectorSource,
    Query,
    Representation,
    ResourceBudget,
    SearchPipeline,
    StoreView,
    VectorStore,
    VectorStoreWriter,
)


REPRESENTATION = Representation("encoder", "1", 8, True)
ENCODINGS = ["float32", "int8"]
TOLERANCE = {"float32": 1e-6, "int8": 1e-2}


def normalized(values: list[list[float]]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return output


def axes(*indices: int) -> np.ndarray:
    return np.eye(8, dtype=np.float32)[list(indices)]


DOCUMENTS = normalized(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0],
    ]
)
LENGTHS = [2, 1, 1]


def create(path, encoding="float32", corpus="corpus", ids=("a", "b", "c"), documents=DOCUMENTS, lengths=LENGTHS):  # type: ignore[no-untyped-def]
    writer = VectorStoreWriter.create(path, corpus, REPRESENTATION, encoding=encoding)
    writer.append(list(ids), documents, lengths)
    writer.commit()
    return writer


def query(vectors: np.ndarray, representation: Representation = REPRESENTATION) -> Query:
    return Query("query", multi_vector=Feature(representation, vectors))


def candidates(*keys: tuple[str, str]) -> list[Candidate]:
    return [Candidate(corpus, item, 0.0, rank, "t") for rank, (corpus, item) in enumerate(keys)]


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_a_store_reads_what_its_writer_commits_in_requested_order(tmp_path, encoding) -> None:  # type: ignore[no-untyped-def]
    committed = create(tmp_path / "vectors", encoding).commit()
    store = VectorStore(tmp_path / "vectors")

    assert (store.corpus, store.encoding, store.representation) == ("corpus", encoding, REPRESENTATION)
    view = store.view()
    assert isinstance(view, StoreView) and isinstance(store, MultiVectorSource)
    assert view.as_of == committed and view.as_of.tzinfo is not None
    assert view.document_ids == ["a", "b", "c"]
    assert "b" in view and "zzz" not in view
    assert view.document_lengths(["c", "zzz", "a"]) == {"c": 1, "a": 2}
    packed, lengths = view.fetch(["c", "a"], threads=1)
    assert packed.shape == (3, 8) and packed.dtype == np.float32
    assert lengths.tolist() == [1, 2]
    np.testing.assert_allclose(packed, DOCUMENTS[[3, 0, 1]], atol=TOLERANCE[encoding])
    with pytest.raises(ValueError, match="not in the vector store"):
        view.fetch(["zzz"])


def test_writes_are_invisible_until_committed_and_views_keep_their_commit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    writer = create(tmp_path / "vectors")
    store = VectorStore(tmp_path / "vectors")
    before = store.view()

    writer.append(["b", "d"], axes(4, 5), [1, 1])
    assert writer.delete(["a", "missing"]) == 1
    assert store.view().document_ids == ["a", "b", "c"]

    writer.commit()
    after = store.view()
    assert after.document_ids == ["b", "c", "d"]
    np.testing.assert_array_equal(after.fetch(["b"])[0], axes(4))
    np.testing.assert_array_equal(before.fetch(["b"])[0], DOCUMENTS[2:3])
    assert after.commit == before.commit + 1


def test_compaction_changes_nothing_a_reader_sees(tmp_path) -> None:  # type: ignore[no-untyped-def]
    writer = create(tmp_path / "vectors", "int8")
    writer.append(["c"], axes(6), [1])
    writer.delete(["a"])
    writer.commit()
    before = VectorStore(tmp_path / "vectors").view()
    writer.compact()
    writer.commit()
    after = VectorStore(tmp_path / "vectors").view()
    assert after.document_ids == before.document_ids == ["b", "c"]
    np.testing.assert_array_equal(after.fetch(["b", "c"])[0], before.fetch(["b", "c"])[0])
    assert sorted(path.name for path in (tmp_path / "vectors").iterdir()) == [
        "manifest.json",
        "segment-2.codes.npy",
        "segment-2.ids.json",
        "segment-2.offsets.npy",
        "segment-2.scales.npy",
    ]


def test_the_writer_refuses_vectors_that_contradict_the_representation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    writer = VectorStoreWriter.create(tmp_path / "vectors", "corpus", REPRESENTATION)
    with pytest.raises(ValueError, match="dimension"):
        writer.append(["a"], np.eye(4, dtype=np.float32)[:1], [1])
    with pytest.raises(ValueError, match="normalized"):
        writer.append(["a"], DOCUMENTS[:1] * 2, [1])
    with pytest.raises(ValueError, match="more than once"):
        writer.append(["a", "a"], DOCUMENTS[:2], [1, 1])
    with pytest.raises(TypeError, match="single string"):
        writer.delete("a")
    with pytest.raises(ValueError, match="encoding"):
        VectorStoreWriter.create(tmp_path / "other", "corpus", REPRESENTATION, encoding="float16")
    with pytest.raises(FileExistsError):
        VectorStoreWriter.create(tmp_path / "vectors", "corpus", REPRESENTATION)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_writer_accepts_non_contiguous_embeddings(tmp_path, encoding) -> None:  # type: ignore[no-untyped-def]
    doubled = np.repeat(DOCUMENTS, 2, axis=0)[::2]
    assert not doubled.flags.c_contiguous
    create(tmp_path / "strided", encoding, documents=doubled)
    create(tmp_path / "expected", encoding)
    ids = ["a", "b", "c"]
    np.testing.assert_array_equal(
        VectorStore(tmp_path / "strided").view().fetch(ids)[0],
        VectorStore(tmp_path / "expected").view().fetch(ids)[0],
    )


class InMemorySource:
    """A source an engine could expose from its own index; no store involved."""

    representation = REPRESENTATION
    score_semantics = "in-memory-exact-full-maxsim"
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __init__(self, documents: dict[str, np.ndarray], corpus: str = "corpus") -> None:
        self.corpus = corpus
        self.documents = documents
        self.fetched: list[list[str]] = []

    def view(self) -> "InMemorySource":
        return self

    def document_lengths(self, document_ids):  # type: ignore[no-untyped-def]
        return {item: len(self.documents[item]) for item in document_ids if item in self.documents}

    def fetch(self, document_ids, *, threads=None):  # type: ignore[no-untyped-def]
        self.fetched.append(list(document_ids))
        rows = [self.documents[item] for item in document_ids]
        return np.concatenate(rows), np.asarray([len(row) for row in rows], dtype=np.int64)


def test_reranker_scores_a_python_source_without_a_store() -> None:
    source = InMemorySource({"a": DOCUMENTS[:2], "b": DOCUMENTS[2:3], "c": DOCUMENTS[3:]})
    assert isinstance(source, MultiVectorSource)
    reranker = MaxSimReranker([source])
    assert reranker.requires == {"multi_vector": REPRESENTATION}
    assert reranker.score_semantics == "in-memory-exact-full-maxsim"
    assert reranker.sources == (source,)

    scored = reranker.rerank(
        query(axes(0, 1)),
        candidates(("corpus", "b"), ("corpus", "a"), ("corpus", "gone"), ("corpus", "c")),
        budget=ResourceBudget(max_batch_tokens=2, threads=1),
    )

    assert scored.scores == pytest.approx([2**0.5, 2.0, None, -1.0], abs=1e-6)
    assert scored.as_of == source.as_of
    assert sorted(map(len, source.fetched)) == [1, 2]


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_reranker_over_a_store_reports_the_stores_fidelity(tmp_path, encoding) -> None:  # type: ignore[no-untyped-def]
    create(tmp_path / "vectors", encoding)
    store = VectorStore(tmp_path / "vectors")
    reranker = MaxSimReranker([store])
    assert reranker.score_semantics == store.score_semantics

    scored = reranker.rerank(
        query(axes(0)), candidates(("corpus", "c"), ("corpus", "a")), budget=ResourceBudget(threads=1)
    )
    assert scored.scores == pytest.approx([-1.0, 1.0], abs=TOLERANCE[encoding])


def test_one_reranker_scores_two_stores_through_a_fused_gatherer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    create(tmp_path / "laws", corpus="laws", ids=("l0", "l1"), documents=DOCUMENTS[:2], lengths=[1, 1])
    create(tmp_path / "cases", corpus="cases", ids=("c0", "c1"), documents=DOCUMENTS[2:], lengths=[1, 1])

    class Fused:
        requires: dict[str, Representation] = {}
        score_semantics = "unranked-union"

        def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
            keys = [("laws", "l0"), ("cases", "c1"), ("laws", "l1"), ("cases", "c0")]
            return Gathered(candidates(*keys), datetime.now(timezone.utc))

    stores = [VectorStore(tmp_path / "laws"), VectorStore(tmp_path / "cases")]
    result = SearchPipeline(Fused(), MaxSimReranker(stores)).search(query(DOCUMENTS[:1]), gather_limit=4, limit=4)
    # l0 is the query itself; c0 = (1,1)/√2; l1 is orthogonal; c1 is its opposite.
    assert [row.document_id for row in result.documents] == ["l0", "c0", "l1", "c1"]


def test_reranker_refuses_a_query_from_another_encoder_and_an_unknown_corpus(tmp_path) -> None:  # type: ignore[no-untyped-def]
    create(tmp_path / "vectors")
    reranker = MaxSimReranker([VectorStore(tmp_path / "vectors")])
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        reranker.rerank(
            query(DOCUMENTS[:1], replace(REPRESENTATION, encoder="other")),
            candidates(("corpus", "a")),
            budget=ResourceBudget(),
        )
    with pytest.raises(IncompatibleIndexError, match="no source for corpus"):
        reranker.rerank(query(DOCUMENTS[:1]), candidates(("other", "a")), budget=ResourceBudget())


def test_reranker_sources_must_agree(tmp_path) -> None:  # type: ignore[no-untyped-def]
    create(tmp_path / "a")
    create(tmp_path / "b", "int8", corpus="other")
    create(tmp_path / "c")
    exact, lossy, duplicate = (VectorStore(tmp_path / name) for name in "abc")
    with pytest.raises(IncompatibleIndexError, match="scores as"):
        MaxSimReranker([exact, lossy])
    with pytest.raises(ValueError, match="more than one source"):
        MaxSimReranker([exact, duplicate])
    with pytest.raises(ValueError, match="at least one source"):
        MaxSimReranker([])
