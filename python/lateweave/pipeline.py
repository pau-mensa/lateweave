from __future__ import annotations

import time

import numpy as np

from ._native import ResourceBudget, Score, validate_and_rank
from .interfaces import (
    CandidateGenerator,
    Query,
    RankedDocument,
    Reranker,
    SearchResult,
    SearchTimings,
)


class SearchPipeline:
    """Gather, optionally rerank, then deterministic top-k.

    Stages must index the same corpus; that is checked once, here. Each stage
    must be able to consume the query; that is checked per search, before any
    stage runs, so a query that cannot be served fails without gathering.
    Without a reranker the gather scores rank the results.
    """

    def __init__(
        self, gatherer: CandidateGenerator, reranker: Reranker | None = None
    ) -> None:
        if reranker is not None:
            gatherer.corpus.assert_compatible(reranker.corpus)
        self.gatherer = gatherer
        self.reranker = reranker

    def search(
        self,
        query: Query | str,
        *,
        gather_limit: int,
        limit: int,
        subset: np.ndarray | None = None,
        budget: ResourceBudget | None = None,
    ) -> SearchResult:
        if gather_limit <= 0:
            raise ValueError("gather_limit must be positive")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if limit > gather_limit:
            raise ValueError("limit cannot exceed gather_limit")
        if isinstance(query, str):
            query = Query(query)
        budget = budget or ResourceBudget()
        stages = [self.gatherer] if self.reranker is None else [self.gatherer, self.reranker]
        for stage in stages:
            for name, representation in stage.requires.items():
                query.feature(name, representation)

        started = time.perf_counter()
        candidates = tuple(self.gatherer.gather(query, gather_limit, subset=subset))
        gathered = time.perf_counter()
        if len(candidates) > gather_limit:
            raise ValueError("gatherer returned more candidates than requested")
        if [candidate.gather_rank for candidate in candidates] != list(
            range(len(candidates))
        ):
            raise ValueError("candidate gather ranks must be contiguous and zero-based")
        if self.reranker is None:
            scores = tuple(
                Score(candidate.document_id, candidate.gather_score)
                for candidate in candidates
            )
            score_semantics = self.gatherer.score_semantics
        else:
            scores = tuple(self.reranker.rerank(query, candidates, budget=budget))
            score_semantics = self.reranker.score_semantics
        reranked = time.perf_counter()

        positions = validate_and_rank(
            [item.document_id for item in candidates],
            [item.gather_rank for item in candidates],
            [item.document_id for item in scores],
            [item.value for item in scores],
            limit,
        )
        documents = tuple(
            RankedDocument(
                document_id=scores[position].document_id,
                score=scores[position].value,
                rank=rank,
            )
            for rank, position in enumerate(positions, 1)
        )
        finished = time.perf_counter()
        return SearchResult(
            documents=documents,
            candidates=candidates,
            scores=scores,
            timings=SearchTimings(
                gather_seconds=gathered - started,
                rerank_seconds=reranked - gathered,
                total_seconds=finished - started,
            ),
            diagnostics={
                "candidate_count": len(candidates),
                "gatherer": type(self.gatherer).__name__,
                "reranker": None if self.reranker is None else type(self.reranker).__name__,
                "score_semantics": score_semantics,
            },
        )
