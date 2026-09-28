"""Pipelines follow what writers publish, in this process or another."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from lateweave import (
    Candidate,
    Feature,
    Float32VectorStore,
    Live,
    MaxSimReranker,
    Query,
    Representation,
    SearchPipeline,
    Segment,
)


REPRESENTATION = Representation("encoder", "1", 4, True)
VECTORS = np.eye(4, dtype=np.float32)


class Moving:
    """A live stage another writer moves by replacing its snapshot."""

    def __init__(self, snapshot) -> None:  # type: ignore[no-untyped-def]
        self.snapshot = snapshot

    def current(self):  # type: ignore[no-untyped-def]
        return self.snapshot


class EveryDocument:
    """Gathers every document of its segments, in order."""

    requires: dict[str, Representation] = {}
    score_semantics = "every-document"

    def __init__(self, *segments: Segment) -> None:
        self.segments = segments

    def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
        rows = [(segment, item) for segment in self.segments for item in range(len(segment))]
        return tuple(
            Candidate(segment, item, 0.0, rank, "every")
            for rank, (segment, item) in enumerate(rows[:limit])
        )


class ByExternalId:
    requires: dict[str, Representation] = {}
    score_semantics = "by-external-id"

    def __init__(self, *segments: Segment) -> None:
        self.segments = segments

    def rerank(self, query, candidates, *, budget):  # type: ignore[no-untyped-def]
        return [-float(candidate.external_id[1:]) for candidate in candidates]


def ids(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [row.external_id for row in result.documents]


def test_a_live_pipeline_serves_each_publish_once_its_stages_agree() -> None:
    first = Segment("docs", "1", ["d0", "d1"])
    second = first.appended(["d2"])
    gatherer, reranker = Moving(EveryDocument(first)), Moving(ByExternalId(first))
    assert isinstance(gatherer, Live)
    pipeline = SearchPipeline(gatherer, reranker)
    frozen = pipeline.freeze()
    assert ids(pipeline.search("q", gather_limit=5, limit=5)) == ["d0", "d1"]

    gatherer.snapshot = EveryDocument(second)
    result = pipeline.search("q", gather_limit=5, limit=5)
    assert result.diagnostics["stale"] and ids(result) == ["d0", "d1"]

    reranker.snapshot = ByExternalId(second)
    result = pipeline.search("q", gather_limit=5, limit=5)
    assert not result.diagnostics["stale"] and ids(result) == ["d0", "d1", "d2"]
    assert ids(frozen.search("q", gather_limit=5, limit=5)) == ["d0", "d1"]


class LexicalIndex:
    """An adapter over an index another process rewrites: its engine state is
    a JSON file of external IDs, replaced by rename."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pinned: EveryDocument | None = None

    def current(self) -> EveryDocument:
        state = json.loads(self.path.read_text())
        segment = Segment(state["corpus_id"], "1", state["ids"], generation=state["generation"])
        if self.pinned is None or self.pinned.segments[0] != segment:
            self.pinned = EveryDocument(segment)
        return self.pinned


def publish_index(path: Path, segment: Segment) -> None:
    staged = path.with_suffix(".tmp")
    staged.write_text(
        json.dumps({"corpus_id": segment.corpus_id, "ids": segment.document_ids, "generation": segment.generation})
    )
    staged.replace(path)


WRITER = """
import sys
from lateweave import open_vector_store
open_vector_store(sys.argv[1]).delete([sys.argv[2]])
"""


def test_two_corpora_follow_a_writer_in_another_process(tmp_path: Path) -> None:
    stores, indexes = [], []
    for corpus_id, prefix in [("laws", "l"), ("cases", "c")]:
        segment = Segment(corpus_id, "1", [f"{prefix}{item}" for item in range(4)])
        stores.append(
            Float32VectorStore.create(tmp_path / corpus_id, segment, VECTORS, [1, 1, 1, 1], REPRESENTATION)
        )
        publish_index(tmp_path / f"{corpus_id}.json", segment)
        indexes.append(LexicalIndex(tmp_path / f"{corpus_id}.json"))

    class Fused:
        """Appends the hits of both corpora for the reranker to order."""

        def __init__(self, *indexes: LexicalIndex) -> None:
            self.indexes = indexes

        def current(self) -> EveryDocument:
            return EveryDocument(*(index.current().segments[0] for index in self.indexes))

    pipeline = SearchPipeline(Fused(*indexes), MaxSimReranker(stores))
    query = Query("q", multi_vector=Feature(REPRESENTATION, VECTORS[1:2]))

    def search():  # type: ignore[no-untyped-def]
        return pipeline.search(query, gather_limit=8, limit=2)

    assert sorted(ids(search())) == ["c1", "l1"]

    # The writer deletes l1 from the store; the lexical index has not moved yet.
    subprocess.run([sys.executable, "-c", WRITER, str(tmp_path / "laws"), "l1"], check=True)
    result = search()
    assert result.diagnostics["stale"] and sorted(ids(result)) == ["c1", "l1"]

    publish_index(tmp_path / "laws.json", indexes[0].current().segments[0].deleted(["l1"]))
    result = search()
    assert not result.diagnostics["stale"]
    assert ids(result)[0] == "c1" and "l1" not in ids(result)
    generations = {c.segment.corpus_id: c.segment.generation for c in result.candidates}
    assert generations == {"laws": 1, "cases": 0}
    assert len(result.candidates) == 7
