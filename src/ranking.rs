//! Deterministic ranking and the score contract it enforces.

use thiserror::Error;

use crate::stage::Candidate;

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
    #[error("{scores} scores were returned for {candidates} candidates")]
    ScoreCount { candidates: usize, scores: usize },
    #[error("the score of candidate {0} is NaN")]
    NanScore(usize),
}

/// Validate one score or `None` per candidate and return the positions of
/// the scored candidates in final rank order.
///
/// `scores[i]` scores `candidates[i]`; unscored candidates are not ranked.
/// Ties keep gather order, which is total because gather ranks are the
/// candidates' positions. Gather scores never participate unless they are
/// the scores.
pub fn validate_and_rank(
    candidates: &[Candidate],
    scores: &[Option<f32>],
    limit: usize,
) -> Result<Vec<usize>, RankingError> {
    if scores.len() != candidates.len() {
        return Err(RankingError::ScoreCount {
            candidates: candidates.len(),
            scores: scores.len(),
        });
    }
    if let Some(position) = scores
        .iter()
        .position(|score| score.is_some_and(f32::is_nan))
    {
        return Err(RankingError::NanScore(position));
    }
    let mut ranked = scores
        .iter()
        .enumerate()
        .filter_map(|(position, score)| score.map(|score| (position, score)))
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|&(left, left_score), &(right, right_score)| {
        right_score.total_cmp(&left_score).then_with(|| {
            candidates[left]
                .gather_rank
                .cmp(&candidates[right].gather_rank)
        })
    });
    ranked.truncate(limit);
    Ok(ranked.into_iter().map(|(position, _)| position).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::DocumentKey;

    fn candidates(count: usize) -> Vec<Candidate> {
        (0..count)
            .map(|rank| Candidate {
                key: DocumentKey::new("docs", rank.to_string()),
                gather_score: 0.0,
                gather_rank: rank,
                provenance: "test".to_string(),
            })
            .collect()
    }

    #[test]
    fn ranking_uses_gather_order_only_as_a_tie_breaker() {
        let scores = [Some(2.0), Some(3.0), Some(2.0)];
        assert_eq!(
            validate_and_rank(&candidates(3), &scores, 3).unwrap(),
            [1, 0, 2]
        );
        assert_eq!(validate_and_rank(&candidates(3), &scores, 1).unwrap(), [1]);
    }

    #[test]
    fn unscored_candidates_are_not_ranked() {
        let scores = [None, Some(1.0), None, Some(4.0)];
        assert_eq!(
            validate_and_rank(&candidates(4), &scores, 4).unwrap(),
            [3, 1]
        );
    }

    #[test]
    fn ranking_requires_one_score_per_candidate() {
        assert_eq!(
            validate_and_rank(&candidates(2), &[Some(1.0)], 2).unwrap_err(),
            RankingError::ScoreCount {
                candidates: 2,
                scores: 1
            }
        );
        assert_eq!(
            validate_and_rank(&candidates(2), &[None, Some(f32::NAN)], 2).unwrap_err(),
            RankingError::NanScore(1)
        );
    }
}
