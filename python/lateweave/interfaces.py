from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ._native import Candidate, ResourceBudget, Score
from .manifest import CorpusManifest, IncompatibleQueryError, Representation


class Feature:
    """One query representation, materialized at most once.

    A feature is whatever an encoder produced for the query text: a token
    matrix, a dense vector, a sparse weighting. Its :class:`Representation`
    names that encoder so a stage can refuse a feature it was not built for.
    Pass ``value`` when it is already computed, or ``provider`` to defer the
    encoding until a stage asks for it.
    """

    __slots__ = ("representation", "_value", "_provider")

    def __init__(
        self,
        representation: Representation,
        value: Any = None,
        *,
        provider: Callable[[], Any] | None = None,
    ) -> None:
        if (value is None) == (provider is None):
            raise ValueError("a feature needs exactly one of value or provider")
        self.representation = representation
        self._value = value
        self._provider = provider

    @property
    def value(self) -> Any:
        if self._value is None:
            assert self._provider is not None
            self._value = self._provider()
            self._provider = None
        return self._value


class Query:
    """Raw text plus the named features stages may consume."""

    __slots__ = ("text", "features")

    def __init__(self, text: str, **features: Feature) -> None:
        self.text = text
        self.features: Mapping[str, Feature] = features

    def feature(self, name: str, representation: Representation) -> Any:
        """The value of feature ``name``, which must come from ``representation``."""
        try:
            feature = self.features[name]
        except KeyError:
            raise IncompatibleQueryError(
                f"query has no {name!r} feature; available: {sorted(self.features)}"
            ) from None
        representation.assert_compatible(feature.representation)
        return feature.value


@runtime_checkable
class CandidateGenerator(Protocol):
    """First stage: selects candidate documents from the whole corpus.

    ``requires`` maps feature names to the representation the gatherer was built
    with; a text-only gatherer declares an empty mapping. ``score_semantics``
    qualifies ``gather_score`` and ranks results when no reranker follows.
    ``subset`` restricts the search to those internal IDs; a gatherer that cannot
    honour it must raise rather than ignore it.
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


@dataclass(frozen=True)
class RankedDocument:
    document_id: int
    score: float
    rank: int


@dataclass(frozen=True)
class SearchTimings:
    gather_seconds: float
    rerank_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SearchResult:
    documents: tuple[RankedDocument, ...]
    candidates: tuple[Candidate, ...]
    scores: tuple[Score, ...]
    timings: SearchTimings
    diagnostics: dict[str, Any]
