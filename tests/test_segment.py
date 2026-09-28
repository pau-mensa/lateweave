from __future__ import annotations

import numpy as np
import pytest

from lateweave import CorpusManifest, IncompatibleIndexError, Segment, document_ids_digest


def test_internal_ids_follow_input_order() -> None:
    segment = Segment("docs", "1", ["b", "a", "c"])

    assert len(segment) == 3
    assert segment.corpus_id == "docs" and segment.generation == 0
    assert segment.document_ids == ["b", "a", "c"]
    assert segment.internal("a") == 1
    assert segment.external(2) == "c"
    assert "a" in segment and "z" not in segment
    with pytest.raises(KeyError):
        segment.internal("z")
    with pytest.raises(IndexError):
        segment.external(3)


def test_refuses_duplicates_and_bare_strings() -> None:
    with pytest.raises(ValueError, match="more than once"):
        Segment("docs", "1", ["a", "b", "a"])
    with pytest.raises(TypeError, match="single string"):
        Segment("docs", "1", "abc")


def test_translates_to_subset_arrays_and_back() -> None:
    segment = Segment("docs", "1", ["b", "a", "c"])

    internal = segment.to_internal(["c", "b"])
    assert internal.dtype == np.int64 and internal.tolist() == [2, 0]
    assert segment.to_external(internal) == ["c", "b"]
    with pytest.raises(ValueError, match="not in segment"):
        segment.to_internal(["z"])
    with pytest.raises(ValueError, match="outside segment"):
        segment.to_external([3])


def test_the_manifest_round_trips_through_from_manifest(tmp_path) -> None:
    segment = Segment("docs", "1", ["b", "a", "c"], generation=4)
    manifest = segment.manifest

    assert manifest == CorpusManifest("docs", "1", 3, document_ids_digest(["b", "a", "c"]), 4)
    manifest.write(tmp_path / "corpus.json")
    assert Segment.from_manifest(CorpusManifest.read(tmp_path / "corpus.json"), ["b", "a", "c"]) == segment
    with pytest.raises(IncompatibleIndexError, match="document_ids_sha256"):
        Segment.from_manifest(manifest, ["a", "b", "c"])


def test_mutations_return_the_next_generation() -> None:
    segment = Segment("docs", "1", ["a", "b", "c"])

    appended = segment.appended(["d"])
    assert (appended.document_ids, appended.generation) == (["a", "b", "c", "d"], 1)
    deleted = appended.deleted(["a", "c"])
    assert (deleted.document_ids, deleted.generation) == (["b", "d"], 2)
    assert segment.appended([]) == segment == segment.deleted([])
    assert segment != Segment("docs", "1", ["a", "b", "c"], generation=1)
    assert {segment: 1}[Segment("docs", "1", ["a", "b", "c"])] == 1
    with pytest.raises(ValueError, match="more than once"):
        segment.deleted(["b", "b"])
