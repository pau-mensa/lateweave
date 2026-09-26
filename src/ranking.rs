//! Deterministic ranking and the reranker-output contract it enforces.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::stage::{Candidate, Score};

/// Reject non-finite inputs without the per-element branch that stops the
/// autovectorizer.  A float is non-finite exactly when its exponent bits are
/// all set, so each chunk reduces with a branchless `|` and only the chunk
/// boundary short-circuits.
pub fn all_finite(values: &[f32]) -> bool {
    const EXPONENT_MASK: u32 = 0x7F80_0000;
    const CHUNK_VALUES: usize = 4096;

    values.chunks(CHUNK_VALUES).all(|chunk| {
        !chunk.iter().fold(false, |found, value| {
            found | (value.to_bits() & EXPONENT_MASK == EXPONENT_MASK)
        })
    })
}

#[derive(Debug, Error, PartialEq)]
pub enum RankingError {
    #[error("candidate document ID {0} occurs more than once")]
    DuplicateCandidate(u64),
    #[error("score document ID {0} occurs more than once")]
    DuplicateScore(u64),
    #[error("reranker omitted candidate document ID {0}")]
    MissingScore(u64),
    #[error("reranker returned document ID {0}, which was not a candidate")]
    UnexpectedScore(u64),
    #[error("score for document ID {0} is NaN")]
    NanScore(u64),
}

/// Validate the reranker contract and return score positions in final rank order.
///
/// Ties are stable by the original gather rank and then document ID. Gather
/// scores never participate in final ranking.
pub fn validate_and_rank(
    candidates: &[Candidate],
    scores: &[Score],
    limit: usize,
) -> Result<Vec<usize>, RankingError> {
    let mut rank_by_id = HashMap::with_capacity(candidates.len());
    for candidate in candidates {
        if rank_by_id
            .insert(candidate.document_id, candidate.gather_rank)
            .is_some()
        {
            return Err(RankingError::DuplicateCandidate(candidate.document_id));
        }
    }

    let mut score_set = HashSet::with_capacity(scores.len());
    for score in scores {
        if !score_set.insert(score.document_id) {
            return Err(RankingError::DuplicateScore(score.document_id));
        }
        if !rank_by_id.contains_key(&score.document_id) {
            return Err(RankingError::UnexpectedScore(score.document_id));
        }
        if score.value.is_nan() {
            return Err(RankingError::NanScore(score.document_id));
        }
    }

    for candidate in candidates {
        if !score_set.contains(&candidate.document_id) {
            return Err(RankingError::MissingScore(candidate.document_id));
        }
    }

    let mut positions = (0..scores.len()).collect::<Vec<_>>();
    positions.sort_unstable_by(|&left, &right| {
        let (left, right) = (&scores[left], &scores[right]);
        right
            .value
            .total_cmp(&left.value)
            .then_with(|| rank_by_id[&left.document_id].cmp(&rank_by_id[&right.document_id]))
            .then_with(|| left.document_id.cmp(&right.document_id))
    });
    positions.truncate(limit.min(positions.len()));
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(ids: &[u64]) -> Vec<Candidate> {
        ids.iter()
            .enumerate()
            .map(|(rank, &document_id)| Candidate {
                document_id,
                gather_score: 0.0,
                gather_rank: rank,
                provenance: "test".to_string(),
            })
            .collect()
    }

    fn scores(rows: &[(u64, f32)]) -> Vec<Score> {
        rows.iter()
            .map(|&(document_id, value)| Score { document_id, value })
            .collect()
    }

    #[test]
    fn ranking_uses_gather_order_only_as_a_tie_breaker() {
        let order = validate_and_rank(
            &candidates(&[9, 5, 7]),
            &scores(&[(5, 2.0), (7, 3.0), (9, 2.0)]),
            3,
        )
        .unwrap();
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn ranking_rejects_a_reranker_that_changes_the_candidate_set() {
        let error = validate_and_rank(&candidates(&[1]), &scores(&[(2, 1.0)]), 1).unwrap_err();
        assert_eq!(error, RankingError::UnexpectedScore(2));
    }
}
