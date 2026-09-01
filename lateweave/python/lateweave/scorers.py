"""Optional scorer implementations composed from lateweave-owned utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._native import Candidate, ResourceBudget, Score, ScorerCapabilities, maxsim_scores_packed
from .interfaces import Query
from .manifest import IndexManifest
from .storage import VectorStore


def _token_batches(
    document_ids: Sequence[int], lengths: dict[int, int], maximum_tokens: int
):
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


class StoredMaxSimScorer:
    """Optional MaxSim scorer for a lateweave-owned vector store.

    This is useful for BM25 and other generators without a document-vector
    representation. Native late-interaction engines should implement the
    CandidateScorer contract directly and keep their transition private.
    """

    def __init__(self, store: VectorStore, manifest: IndexManifest) -> None:
        if store.document_count != manifest.document_count:
            raise ValueError("vector store and scorer manifest document counts differ")
        if store.dimension != manifest.dimension:
            raise ValueError("vector store and scorer manifest dimensions differ")
        if store.format != manifest.representation:
            raise ValueError("vector store and scorer manifest representations differ")
        if store.score_semantics != manifest.score_semantics:
            raise ValueError("vector store and scorer score semantics differ")
        self.store = store
        self.manifest = manifest
        self.capabilities = ScorerCapabilities(
            preferred_batch_tokens=131_072,
            supports_mmap=True,
            supports_prefetch=False,
            supports_candidate_reordering=True,
            supports_cpu_gpu_sharding=False,
            score_semantics=store.score_semantics,
        )

    def score(
        self,
        query: Query,
        candidates: Sequence[Candidate],
        *,
        budget: ResourceBudget,
    ) -> tuple[Score, ...]:
        if not candidates:
            return ()
        query_embeddings = query.embeddings
        if query_embeddings.shape[1] != self.manifest.dimension:
            raise ValueError("query and scorer dimensions differ")
        if not np.isfinite(query_embeddings).all():
            raise ValueError("query embeddings contain a non-finite value")
        if self.manifest.normalized and not np.allclose(
            np.linalg.norm(query_embeddings, axis=1), 1.0, rtol=1e-3, atol=1e-4
        ):
            raise ValueError("scorer manifest requires normalized query embeddings")
        candidate_ids = [int(candidate.document_id) for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")

        prepared_query = self.store.prepare_query(
            query_embeddings, threads=budget.threads
        )
        lengths = self.store.document_lengths(candidate_ids)
        scores: dict[int, float] = {}
        for start in range(0, len(candidate_ids), budget.max_documents_per_batch):
            window = candidate_ids[start : start + budget.max_documents_per_batch]
            for batch_ids in _token_batches(window, lengths, budget.max_batch_tokens):
                token_count = sum(lengths[item] for item in batch_ids)
                workspace = self.store.estimated_workspace_bytes(
                    token_count, len(prepared_query)
                ) + prepared_query.nbytes
                if (
                    budget.max_memory_bytes is not None
                    and workspace > budget.max_memory_bytes
                ):
                    raise MemoryError(
                        f"estimated scoring workspace {workspace} exceeds "
                        f"budget {budget.max_memory_bytes}"
                    )
                documents, batch_lengths = self.store.transition(
                    batch_ids, threads=budget.threads
                )
                values = maxsim_scores_packed(
                    prepared_query,
                    documents,
                    batch_lengths,
                    max_batch_tokens=budget.max_batch_tokens,
                    threads=budget.threads,
                )
                scores.update(
                    (document_id, float(value))
                    for document_id, value in zip(batch_ids, values, strict=True)
                )
        return tuple(Score(item, scores[item]) for item in candidate_ids)
