from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

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


DEADLOCK_PROBE = """
import threading, time
from datetime import datetime, timezone
import numpy as np
from lateweave import (
    Candidate, Feature, Gathered, MaxSimReranker, Query, Representation, SearchPipeline,
)

representation = Representation("encoder", "1", 2, True)
rows = {"a": 0, "b": 1}
now = datetime.now(timezone.utc)

class View:
    as_of = now
    def document_lengths(self, document_ids):
        return {item: 1 for item in document_ids}
    def fetch(self, document_ids, *, threads=None):
        vectors = np.eye(2, dtype=np.float32)[[rows[item] for item in document_ids]]
        return vectors, np.ones(len(document_ids), dtype=np.int64)

class Source:
    corpus = "corpus"
    representation = representation
    score_semantics = "in-memory"
    def view(self):
        return View()

class Gatherer:
    requires = {}
    score_semantics = "gather"
    def gather(self, query, limit, *, subset=None):
        return Gathered(
            [Candidate("corpus", "a", 1.0, 0, "t"), Candidate("corpus", "b", 1.0, 1, "t")], now
        )

def encode():
    time.sleep(0.3)
    return np.asarray([[0.0, 1.0]], dtype=np.float32)

query = Query("query", multi_vector=Feature(representation, provider=encode))
pipeline = SearchPipeline(Gatherer(), MaxSimReranker([Source()]))
search = threading.Thread(target=lambda: pipeline.search(query, gather_limit=2, limit=2))
search.start()
time.sleep(0.05)
query.feature("multi_vector", representation)
search.join()
"""


def test_a_feature_shared_with_a_running_search_does_not_deadlock() -> None:
    # The search materializes the feature without the GIL while this thread
    # asks for it with the GIL held.
    subprocess.run([sys.executable, "-c", DEADLOCK_PROBE], check=True, timeout=30)
