"""Structural contracts for stages and sources implemented in Python.

The pipeline itself is native; these protocols describe what it calls on a
Python object passed as a gatherer, a reranker, or a ``MaxSimReranker`` source.

Stages name documents by corpus and document ID, the ID the system of record
gives them, and each reads its own indexes, which writers anywhere move
independently. Every stage result carries ``as_of``, a timezone-aware
``datetime``: every write committed to the index the stage read before it is
reflected in the result.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ._native import Candidate, Gathered, Query, ResourceBudget, Scored, Subset
from .representation import Representation


@runtime_checkable
class CandidateGenerator(Protocol):
    """First stage: selects candidate documents.

    ``requires`` maps feature names to the representation the gatherer was built
    with; a text-only gatherer declares an empty mapping. ``score_semantics``
    qualifies ``gather_score`` and ranks results when no reranker follows.
    ``gather`` returns unique candidates with dense zero-based ranks, read from
    one consistent state of each index it searches. ``subset`` is a ``Subset``
    naming, per corpus, the documents the search may return: use
    ``subset.allows(corpus, id)``, or ``subset.restriction(corpus)`` to read an
    include list. A corpus it does not name contributes nothing, and a gatherer
    that cannot honour it must raise rather than ignore it. ``requires`` and ``score_semantics`` are read
    once, when the pipeline is built.
    """

    requires: Mapping[str, Representation]
    score_semantics: str

    def gather(
        self, query: Query, limit: int, *, subset: Subset | None = None
    ) -> Gathered: ...


@runtime_checkable
class Reranker(Protocol):
    """Second stage: one score, or ``None`` for a document its index does not
    hold, for every candidate, in candidate order.

    A candidate from a corpus the reranker cannot score at all must raise.
    """

    requires: Mapping[str, Representation]
    score_semantics: str

    def rerank(
        self,
        query: Query,
        candidates: Sequence[Candidate],
        *,
        budget: ResourceBudget,
    ) -> Scored: ...


@runtime_checkable
class VectorView(Protocol):
    """One consistent state of a ``MultiVectorSource``.

    ``document_lengths`` maps each requested document the view holds to its
    token count and leaves out the rest. ``fetch`` returns a float32
    ``[tokens, dimension]`` matrix holding the requested documents, which the
    view must hold, in the requested order, plus their int64 lengths.
    """

    as_of: datetime

    def document_lengths(self, document_ids: Sequence[str]) -> Mapping[str, int]: ...

    def fetch(
        self, document_ids: Sequence[str], *, threads: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class MultiVectorSource(Protocol):
    """Token vectors of one corpus's documents.

    ``view()`` returns the source's current state, which a rerank reads
    throughout. ``score_semantics`` qualifies what MaxSim over the fetched
    vectors means, since a source may reconstruct from a lossy code.
    """

    corpus: str
    representation: Representation
    score_semantics: str

    def view(self) -> VectorView: ...
