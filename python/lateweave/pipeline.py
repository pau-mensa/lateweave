from __future__ import annotations

import time

from ._native import ResourceBudget, validate_and_rank
from .interfaces import (
    CandidateGenerator,
    CandidateScorer,
    Query,
    RankedDocument,
    SearchResult,
    SearchTimings,
)


class SearchPipeline:
    """Generic gather, score, and top-k orchestration."""

    def __init__(self, generator: CandidateGenerator, scorer: CandidateScorer) -> None:
        generator.manifest.assert_compatible(scorer.manifest)
        self.generator = generator
        self.scorer = scorer

    def search(
        self,
        query: Query | str,
        *,
        gather_limit: int,
        limit: int,
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

        started = time.perf_counter()
        candidates = tuple(self.generator.gather(query, gather_limit))
        gathered = time.perf_counter()
        if len(candidates) > gather_limit:
            raise ValueError("generator returned more candidates than requested")
        if [candidate.gather_rank for candidate in candidates] != list(
            range(len(candidates))
        ):
            raise ValueError("candidate gather ranks must be contiguous and zero-based")
        scores = tuple(self.scorer.score(query, candidates, budget=budget))
        scored = time.perf_counter()

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
                score_seconds=scored - gathered,
                total_seconds=finished - started,
            ),
            diagnostics={
                "candidate_count": len(candidates),
                "scored_count": len(scores),
                "generator": type(self.generator).__name__,
                "scorer": type(self.scorer).__name__,
                "score_semantics": self.scorer.capabilities.score_semantics,
            },
        )
