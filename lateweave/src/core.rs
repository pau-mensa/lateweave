//! Language-independent search contracts.
//!
//! Backends own representation access and decoding.  The traits deliberately
//! meet only after document candidate selection, so a scorer may reconstruct
//! vectors, evaluate compressed codes, or fuse the complete operation.

use std::collections::HashSet;

use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub document_id: u64,
    pub gather_score: f32,
    pub gather_rank: usize,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub document_id: u64,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    pub max_memory_bytes: Option<u64>,
    pub max_batch_tokens: usize,
    pub max_documents_per_batch: usize,
    pub threads: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScorerCapabilities {
    pub preferred_batch_tokens: usize,
    pub supports_mmap: bool,
    pub supports_prefetch: bool,
    pub supports_candidate_reordering: bool,
    pub supports_cpu_gpu_sharding: bool,
    pub score_semantics: String,
}

pub trait CandidateGenerator<Q> {
    type Error;

    fn gather(&self, query: &Q, limit: usize) -> Result<Vec<Candidate>, Self::Error>;
}

pub trait CandidateScorer<Q> {
    type Error;

    fn capabilities(&self) -> &ScorerCapabilities;

    fn score(
        &self,
        query: &Q,
        candidates: &[Candidate],
        budget: &ResourceBudget,
    ) -> Result<Vec<Score>, Self::Error>;
}

#[derive(Debug, Error, PartialEq)]
pub enum RankingError {
    #[error("candidate document ID {0} occurs more than once")]
    DuplicateCandidate(u64),
    #[error("score document ID {0} occurs more than once")]
    DuplicateScore(u64),
    #[error("scorer omitted candidate document ID {0}")]
    MissingScore(u64),
    #[error("scorer returned document ID {0}, which was not a candidate")]
    UnexpectedScore(u64),
    #[error("score for document ID {0} is NaN")]
    NanScore(u64),
}

/// Validate the scorer contract and return score positions in final rank order.
///
/// Ties are stable by the original gather rank and then document ID. Gather
/// scores never participate in final ranking.
pub fn validate_and_rank(
    candidate_ids: &[u64],
    candidate_ranks: &[usize],
    score_ids: &[u64],
    score_values: &[f32],
    limit: usize,
) -> Result<Vec<usize>, RankingError> {
    let mut candidate_set = HashSet::with_capacity(candidate_ids.len());
    for &document_id in candidate_ids {
        if !candidate_set.insert(document_id) {
            return Err(RankingError::DuplicateCandidate(document_id));
        }
    }

    let mut score_set = HashSet::with_capacity(score_ids.len());
    for (&document_id, &value) in score_ids.iter().zip(score_values) {
        if !score_set.insert(document_id) {
            return Err(RankingError::DuplicateScore(document_id));
        }
        if !candidate_set.contains(&document_id) {
            return Err(RankingError::UnexpectedScore(document_id));
        }
        if value.is_nan() {
            return Err(RankingError::NanScore(document_id));
        }
    }

    for &document_id in candidate_ids {
        if !score_set.contains(&document_id) {
            return Err(RankingError::MissingScore(document_id));
        }
    }

    let rank_by_id = candidate_ids
        .iter()
        .copied()
        .zip(candidate_ranks.iter().copied())
        .collect::<std::collections::HashMap<_, _>>();
    let mut positions = (0..score_ids.len()).collect::<Vec<_>>();
    positions.sort_unstable_by(|&left, &right| {
        score_values[right]
            .total_cmp(&score_values[left])
            .then_with(|| rank_by_id[&score_ids[left]].cmp(&rank_by_id[&score_ids[right]]))
            .then_with(|| score_ids[left].cmp(&score_ids[right]))
    });
    positions.truncate(limit.min(positions.len()));
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_uses_gather_order_only_as_a_tie_breaker() {
        let order =
            validate_and_rank(&[9, 5, 7], &[0, 1, 2], &[5, 7, 9], &[2.0, 3.0, 2.0], 3).unwrap();
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn ranking_rejects_a_scorer_that_changes_the_candidate_set() {
        let error = validate_and_rank(&[1], &[0], &[2], &[1.0], 1).unwrap_err();
        assert_eq!(error, RankingError::UnexpectedScore(2));
    }
}
