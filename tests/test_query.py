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
import numpy as np
from lateweave import (
    Candidate, CorpusManifest, Feature, MaxSimReranker, Query, Representation, SearchPipeline,
)

representation = Representation("encoder", "1", 2, True)
corpus = CorpusManifest("corpus", "1", 2, "abc")

class Source:
    representation = representation
    score_semantics = "in-memory"
    document_count = 2
    def document_lengths(self, document_ids):
        return {item: 1 for item in document_ids}
    def fetch(self, document_ids, *, threads=None):
        rows = np.eye(2, dtype=np.float32)[list(document_ids)]
        return rows, np.ones(len(document_ids), dtype=np.int64)

class Gatherer:
    corpus = corpus
    requires = {}
    score_semantics = "gather"
    def gather(self, query, limit, *, subset=None):
        return (Candidate(0, 1.0, 0, "t"), Candidate(1, 1.0, 1, "t"))

def encode():
    time.sleep(0.3)
    return np.asarray([[0.0, 1.0]], dtype=np.float32)

query = Query("query", multi_vector=Feature(representation, provider=encode))
pipeline = SearchPipeline(Gatherer(), MaxSimReranker(Source(), corpus))
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
