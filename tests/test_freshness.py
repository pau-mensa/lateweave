"""A serving process follows indexes that another process writes, each at its
own pace, and knows how fresh every answer is."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from lateweave import (
    Candidate,
    Feature,
    Gathered,
    MaxSimReranker,
    Query,
    Representation,
    SearchPipeline,
    VectorStore,
    VectorStoreWriter,
)


REPRESENTATION = Representation("encoder", "1", 4, True)
VECTORS = np.eye(4, dtype=np.float32)


def publish_index(path: Path, ids: list[str]) -> None:
    """What an engine outside lateweave does: replace its state by rename."""
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps({"ids": ids, "committed_at": datetime.now(timezone.utc).timestamp()}))
    staged.replace(path)


class LexicalIndex:
    """Gathers every document of an index another process rewrites."""

    requires: dict[str, Representation] = {}
    score_semantics = "every-document"

    def __init__(self, path: Path) -> None:
        self.path = path

    def gather(self, query, limit, *, subset=None):  # type: ignore[no-untyped-def]
        state = json.loads(self.path.read_text())
        return Gathered(
            [Candidate("docs", item, 0.0, rank, "lexical") for rank, item in enumerate(state["ids"][:limit])],
            datetime.fromtimestamp(state["committed_at"], timezone.utc),
        )


WRITER = """
import sys
from lateweave import VectorStoreWriter
writer = VectorStoreWriter(sys.argv[1])
writer.delete(sys.argv[2:])
writer.commit()
"""


def ids(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [row.document_id for row in result.documents]


def test_a_delete_is_served_once_either_index_applies_it(tmp_path: Path) -> None:
    writer = VectorStoreWriter.create(tmp_path / "vectors", "docs", REPRESENTATION)
    writer.append(["d0", "d1", "d2", "d3"], VECTORS, [1, 1, 1, 1])
    writer.commit()
    publish_index(tmp_path / "lexical.json", ["d0", "d1", "d2", "d3"])

    pipeline = SearchPipeline(LexicalIndex(tmp_path / "lexical.json"), MaxSimReranker([VectorStore(tmp_path / "vectors")]))
    query = Query("q", multi_vector=Feature(REPRESENTATION, VECTORS[1:2]))

    def search():  # type: ignore[no-untyped-def]
        return pipeline.search(query, gather_limit=8, limit=2)

    assert ids(search())[0] == "d1"

    # The vectors lose d1 first; the lexical index still returns it.
    subprocess.run([sys.executable, "-c", WRITER, str(tmp_path / "vectors"), "d1"], check=True)
    result = search()
    assert "d1" not in ids(result)
    assert result.diagnostics["dropped"] == 1

    # Then the lexical index loses d2; the vectors still hold it.
    publish_index(tmp_path / "lexical.json", ["d0", "d1", "d3"])
    result = search()
    assert "d1" not in ids(result) and "d2" not in ids(result)
    assert (result.diagnostics["candidate_count"], result.diagnostics["dropped"]) == (3, 1)


def test_an_insert_is_served_once_every_index_has_it(tmp_path: Path) -> None:
    writer = VectorStoreWriter.create(tmp_path / "vectors", "docs", REPRESENTATION)
    writer.append(["d0"], VECTORS[:1], [1])
    writer.commit()
    publish_index(tmp_path / "lexical.json", ["d0", "d1"])
    pipeline = SearchPipeline(LexicalIndex(tmp_path / "lexical.json"), MaxSimReranker([VectorStore(tmp_path / "vectors")]))
    query = Query("q", multi_vector=Feature(REPRESENTATION, VECTORS[1:2]))

    assert ids(pipeline.search(query, gather_limit=8, limit=2)) == ["d0"]
    writer.append(["d1"], VECTORS[1:2], [1])
    writer.commit()
    assert ids(pipeline.search(query, gather_limit=8, limit=2)) == ["d1", "d0"]


def test_the_answer_is_as_fresh_as_the_older_index_and_an_idle_writer_advances_it(tmp_path: Path) -> None:
    writer = VectorStoreWriter.create(tmp_path / "vectors", "docs", REPRESENTATION)
    writer.append(["d0"], VECTORS[:1], [1])
    first = writer.commit()
    time.sleep(0.01)
    publish_index(tmp_path / "lexical.json", ["d0"])
    pipeline = SearchPipeline(LexicalIndex(tmp_path / "lexical.json"), MaxSimReranker([VectorStore(tmp_path / "vectors")]))
    query = Query("q", multi_vector=Feature(REPRESENTATION, VECTORS[:1]))

    assert pipeline.search(query, gather_limit=1, limit=1).as_of == first
    heartbeat = writer.commit()
    result = pipeline.search(query, gather_limit=1, limit=1)
    assert ids(result) == ["d0"] and first < result.as_of < heartbeat
