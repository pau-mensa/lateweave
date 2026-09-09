"""MaxSim reranking over any multi-vector source."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from ._native import Candidate, ResourceBudget, Score, maxsim_scores_packed
from .interfaces import Query
from .manifest import CorpusManifest
from .storage import MultiVectorSource


def _token_batches(
    document_ids: Sequence[int], lengths: dict[int, int], maximum_tokens: int
) -> Iterator[list[int]]:
    ordered = sorted(document_ids, key=lambda item: (lengths[item], item))
    batch: list[int] = []
    tokens = 0
    for document_id in ordered:
        length = lengths[document_id]
        if batch and tokens + length > maximum_tokens:
            yield batch
            batch = []
            tokens = 0
        batch.append(document_id)
        tokens += length
    if batch:
        yield batch


class MaxSimReranker:
    """Exact MaxSim between the query's token matrix and a source's documents.

    The source may be a lateweave store or an engine's own reconstruction; the
    reranker requires the query feature ``feature`` to carry the source's
    representation, so vectors from a different encoder are refused up front.
    """

    def __init__(
        self,
        source: MultiVectorSource,
        corpus: CorpusManifest,
        *,
        feature: str = "multi_vector",
    ) -> None:
        if source.document_count != corpus.document_count:
            raise ValueError("source and corpus manifest document counts differ")
        self.source = source
        self.corpus = corpus
        self.feature = feature
        self.requires = {feature: source.representation}
        self.score_semantics = source.score_semantics

    def rerank(
        self,
        query: Query,
        candidates: Sequence[Candidate],
        *,
        budget: ResourceBudget,
    ) -> tuple[Score, ...]:
        if not candidates:
            return ()
        vectors = np.ascontiguousarray(
            query.feature(self.feature, self.source.representation), dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[1] != self.source.representation.dimension:
            raise ValueError("query feature must be a [tokens, dimension] matrix")
        candidate_ids = [int(candidate.document_id) for candidate in candidates]
        lengths = self.source.document_lengths(candidate_ids)
        scores: dict[int, float] = {}
        for start in range(0, len(candidate_ids), budget.max_documents_per_batch):
            window = candidate_ids[start : start + budget.max_documents_per_batch]
            for batch_ids in _token_batches(window, lengths, budget.max_batch_tokens):
                documents, batch_lengths = self.source.fetch(batch_ids, threads=budget.threads)
                values = maxsim_scores_packed(
                    vectors,
                    documents,
                    batch_lengths,
                    max_batch_tokens=budget.max_batch_tokens,
                    threads=budget.threads,
                )
                scores.update(zip(batch_ids, values.tolist(), strict=True))
        return tuple(Score(item, scores[item]) for item in candidate_ids)
