from __future__ import annotations

from dataclasses import replace

import pytest

from lateweave import IncompatibleQueryError, Representation


def test_representation_mismatch_names_the_field() -> None:
    representation = Representation("encoder", "1", 8, True)
    representation.assert_compatible(Representation("encoder", "1", 8, True))
    with pytest.raises(IncompatibleQueryError, match="dimension: 8 != 4"):
        representation.assert_compatible(replace(representation, dimension=4))


def test_representations_round_trip() -> None:
    representation = Representation("encoder", "1", 8, True, query_template="[Q] ")
    assert Representation.from_dict(representation.to_dict()) == representation


def test_representations_reject_empty_identity() -> None:
    with pytest.raises(ValueError, match="encoder"):
        Representation("", "1", 8, True)
    with pytest.raises(ValueError, match="dimension"):
        Representation("encoder", "1", 0, True)
