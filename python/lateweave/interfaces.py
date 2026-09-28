"""Structural contracts for stages and sources implemented in Python.

The pipeline itself is native; these protocols describe what it calls on a
Python object passed as a gatherer, a reranker, or a ``MaxSimReranker`` source.

Any of them may instead be ``Live``: an object over state that writers, in
this process or another, move to new generations, whose ``current()`` returns
the snapshot object implementing the protocol. The pipeline calls it at the
start of every search.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

from ._native import Candidate, Query, ResourceBudget, Segment
from .manifest import Representation


Snapshot = TypeVar("Snapshot", covariant=True)


@runtime_checkable
class Live(Protocol[Snapshot]):
    """Hands out the current snapshot of a gatherer, reranker, or source.

    ``current()`` must return a snapshot whose segment and engine state were
    read together, and should return the same object while nothing moved, so
    the pipeline reads its declaration once per generation.
    """

    def current(self) -> Snapshot: ...


@runtime_checkable
class CandidateGenerator(Protocol):
    """First stage: selects candidate documents from its segments.

    ``segments`` are the snapshots the gatherer searches, each under a distinct
    corpus ID; every candidate names one of them and an internal ID inside it.
    ``requires`` maps feature names to the representation the gatherer was built
    with; a text-only gatherer declares an empty mapping. ``score_semantics``
    qualifies ``gather_score`` and ranks results when no reranker follows.
    ``subset`` maps every corpus ID in ``segments`` to an ascending int64 array
    of the internal IDs the search is restricted to, empty for a segment the
    search excludes; a gatherer that cannot honour it must raise rather than
    ignore it. ``segments``, ``requires``, and ``score_semantics`` are read
    once, when the pipeline is built.
    """

    segments: Sequence[Segment]
    requires: Mapping[str, Representation]
    score_semantics: str

    def gather(
        self, query: Query, limit: int, *, subset: Mapping[str, np.ndarray] | None = None
    ) -> Sequence[Candidate]: ...


@runtime_checkable
class Reranker(Protocol):
    """Second stage: one qualified score for every candidate, in candidate order.

    ``segments`` must hold the same snapshot of every segment the gatherer
    searches.
    """

    segments: Sequence[Segment]
    requires: Mapping[str, Representation]
    score_semantics: str

    def rerank(
        self,
        query: Query,
        candidates: Sequence[Candidate],
        *,
        budget: ResourceBudget,
    ) -> Sequence[float] | np.ndarray: ...


@runtime_checkable
class MultiVectorSource(Protocol):
    """Token vectors of one segment's documents, fetched by internal ID.

    A source is a snapshot: the vectors an internal ID of ``segment`` names
    never change for the life of the source. ``fetch`` returns a float32
    ``[tokens, dimension]`` matrix holding the requested documents in the
    requested order, plus their int64 lengths. ``score_semantics`` qualifies
    what MaxSim over those vectors means, since a source may reconstruct from a
    lossy code.
    """

    segment: Segment
    representation: Representation
    score_semantics: str

    def document_lengths(self, document_ids: Sequence[int]) -> Mapping[int, int]: ...

    def fetch(
        self, document_ids: Sequence[int], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...
