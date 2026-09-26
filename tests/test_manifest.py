from __future__ import annotations

from dataclasses import replace

import pytest

from lateweave import (
    CorpusManifest,
    IncompatibleIndexError,
    IncompatibleQueryError,
    Representation,
    document_ids_digest,
)


def test_corpus_manifests_compare_every_identity_field() -> None:
    manifest = CorpusManifest("corpus", "1", 3, "abc")
    manifest.assert_compatible(CorpusManifest("corpus", "1", 3, "abc"))
    with pytest.raises(IncompatibleIndexError, match="generation: 0 != 1"):
        manifest.assert_compatible(replace(manifest, generation=1))


def test_representation_mismatch_names_the_field() -> None:
    representation = Representation("encoder", "1", 8, True)
    representation.assert_compatible(Representation("encoder", "1", 8, True))
    with pytest.raises(IncompatibleQueryError, match="dimension: 8 != 4"):
        representation.assert_compatible(replace(representation, dimension=4))


def test_manifests_round_trip(tmp_path) -> None:
    manifest = CorpusManifest("corpus", "1", 3, "abc", generation=2)
    manifest.write(tmp_path / "corpus.json")
    assert CorpusManifest.read(tmp_path / "corpus.json") == manifest

    representation = Representation("encoder", "1", 8, True, query_template="[Q] ")
    assert Representation.from_dict(representation.to_dict()) == representation


def test_manifests_reject_empty_identity() -> None:
    with pytest.raises(ValueError, match="corpus_id"):
        CorpusManifest("", "1", 3, "abc")
    with pytest.raises(ValueError, match="dimension"):
        Representation("encoder", "1", 0, True)


def test_document_ids_digest_depends_on_order_and_boundaries() -> None:
    assert document_ids_digest(["a", "b"]) != document_ids_digest(["b", "a"])
    assert document_ids_digest(["ab", "c"]) != document_ids_digest(["a", "bc"])
    assert document_ids_digest(name for name in ["a", "b"]) == document_ids_digest(("a", "b"))
