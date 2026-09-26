"""Structural contracts for stages and sources implemented in Python.

The pipeline itself is native; these protocols describe what it calls on a
Python object passed as a gatherer, a reranker, or a ``MaxSimReranker`` source.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ._native import Candidate, Query, ResourceBudget, Score
from .manifest import CorpusManifest, Representation


@runtime_checkable
class CandidateGenerator(Protocol):
    """First stage: selects candidate documents from the whole corpus.

    ``requires`` maps feature names to the representation the gatherer was built
    with; a text-only gatherer declares an empty mapping. ``score_semantics``
    qualifies ``gather_score`` and ranks results when no reranker follows.
    ``subset`` is an ascending int64 array of the internal IDs the search is
    restricted to; a gatherer that cannot honour it must raise rather than
    ignore it. ``corpus``, ``requires``, and ``score_semantics`` are read once,
    when the pipeline is built.
    """

    corpus: CorpusManifest
    requires: Mapping[str, Representation]
    score_semantics: str

    def gather(
        self, query: Query, limit: int, *, subset: np.ndarray | None = None
    ) -> Sequence[Candidate]: ...


@runtime_checkable
class Reranker(Protocol):
    """Second stage: one qualified score for every candidate it is given."""

    corpus: CorpusManifest
    requires: Mapping[str, Representation]
    score_semantics: str

    def rerank(
        self,
        query: Query,
        candidates: Sequence[Candidate],
        *,
        budget: ResourceBudget,
    ) -> Sequence[Score]: ...


@runtime_checkable
class MultiVectorSource(Protocol):
    """Token vectors of documents, fetched by internal ID.

    ``fetch`` returns a float32 ``[tokens, dimension]`` matrix holding the
    requested documents in the requested order, plus their int64 lengths.
    ``score_semantics`` qualifies what MaxSim over those vectors means, since a
    source may reconstruct from a lossy code.
    """

    representation: Representation
    score_semantics: str
    document_count: int

    def document_lengths(self, document_ids: Sequence[int]) -> Mapping[int, int]: ...

    def fetch(
        self, document_ids: Sequence[int], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...
