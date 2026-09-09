from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lateweave import Feature, IncompatibleQueryError, Query, Representation


REPRESENTATION = Representation("encoder", "1", 8, True)


def test_feature_provider_runs_once() -> None:
    calls = 0

    def encode() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones((2, 8), dtype=np.float32)

    query = Query("query", multi_vector=Feature(REPRESENTATION, provider=encode))
    assert calls == 0
    first = query.feature("multi_vector", REPRESENTATION)
    assert first is query.feature("multi_vector", REPRESENTATION)
    assert calls == 1


def test_feature_needs_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Feature(REPRESENTATION)
    with pytest.raises(ValueError, match="exactly one"):
        Feature(REPRESENTATION, np.ones(1), provider=lambda: np.ones(1))


def test_missing_feature_is_an_incompatible_query() -> None:
    with pytest.raises(IncompatibleQueryError, match="'multi_vector'"):
        Query("query").feature("multi_vector", REPRESENTATION)


def test_feature_from_another_encoder_is_refused_before_materializing() -> None:
    def encode() -> np.ndarray:
        raise AssertionError("must not encode for a stage that cannot use it")

    query = Query("query", multi_vector=Feature(REPRESENTATION, provider=encode))
    with pytest.raises(IncompatibleQueryError, match="encoder"):
        query.feature("multi_vector", replace(REPRESENTATION, encoder="other"))


def test_query_text_is_always_available() -> None:
    assert Query("query").text == "query"
    assert Query("query").features == {}
