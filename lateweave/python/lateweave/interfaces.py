from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from ._native import Candidate, ResourceBudget, Score, ScorerCapabilities
from .manifest import IndexManifest


EmbeddingProvider = Callable[["Query"], np.ndarray]


@dataclass
class Query:
    """A query with lazily shared features.

    `embedding_provider` is invoked at most once. A text-only gatherer therefore
    never pays query-encoding cost, while a gatherer and scorer can share the
    same materialized array.
    """

    text: str
    token_ids: np.ndarray | None = None
    _embeddings: np.ndarray | None = field(default=None, repr=False)
    embedding_provider: EmbeddingProvider | None = field(default=None, repr=False)

    def __init__(
        self,
        text: str,
        *,
        token_ids: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.text = text
        self.token_ids = token_ids
        self._embeddings = embeddings
        self.embedding_provider = embedding_provider

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            if self.embedding_provider is None:
                raise ValueError("query embeddings were requested but no provider is configured")
            self._embeddings = self.embedding_provider(self)
        value = np.asarray(self._embeddings, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError("query embeddings must have shape [tokens, dimension]")
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        self._embeddings = value
        return value


@runtime_checkable
class CandidateGenerator(Protocol):
    manifest: IndexManifest

    def gather(self, query: Query, limit: int) -> Sequence[Candidate]: ...


@runtime_checkable
class CandidateScorer(Protocol):
    manifest: IndexManifest
    capabilities: ScorerCapabilities

    def score(
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
    score_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SearchResult:
    documents: tuple[RankedDocument, ...]
    candidates: tuple[Candidate, ...]
    scores: tuple[Score, ...]
    timings: SearchTimings
    diagnostics: dict[str, Any]

