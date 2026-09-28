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

/// Validate one score per candidate and return candidate positions in final
/// rank order.
///
/// `scores[i]` scores `candidates[i]`. Ties keep gather order, which is total
/// because gather ranks are the candidates' positions. Gather scores never
/// participate unless they are the scores.
pub fn validate_and_rank(
    candidates: &[Candidate],
    scores: &[f32],
    limit: usize,
) -> Result<Vec<usize>, RankingError> {
    if scores.len() != candidates.len() {
        return Err(RankingError::ScoreCount {
            candidates: candidates.len(),
            scores: scores.len(),
        });
    }
    if let Some(position) = scores.iter().position(|score| score.is_nan()) {
        return Err(RankingError::NanScore(position));
    }
    let mut positions = (0..scores.len()).collect::<Vec<_>>();
    positions.sort_unstable_by(|&left, &right| {
        scores[right].total_cmp(&scores[left]).then_with(|| {
            candidates[left]
                .gather_rank
                .cmp(&candidates[right].gather_rank)
        })
    });
    positions.truncate(limit.min(positions.len()));
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::Segment;

    fn candidates(count: usize) -> Vec<Candidate> {
        let segment = Segment::new("docs", "1", 0, (0..count).map(|id| id.to_string())).unwrap();
        (0..count)
            .map(|rank| Candidate {
                segment: segment.clone(),
                document_id: rank as u64,
                gather_score: 0.0,
                gather_rank: rank,
                provenance: "test".to_string(),
            })
            .collect()
    }

    #[test]
    fn ranking_uses_gather_order_only_as_a_tie_breaker() {
        let order = validate_and_rank(&candidates(3), &[2.0, 3.0, 2.0], 3).unwrap();
        assert_eq!(order, vec![1, 0, 2]);
        assert_eq!(
            validate_and_rank(&candidates(3), &[2.0, 3.0, 2.0], 1).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn ranking_requires_one_score_per_candidate() {
        assert_eq!(
            validate_and_rank(&candidates(2), &[1.0], 2).unwrap_err(),
            RankingError::ScoreCount {
                candidates: 2,
                scores: 1
            }
        );
        assert_eq!(
            validate_and_rank(&candidates(2), &[1.0, f32::NAN], 2).unwrap_err(),
            RankingError::NanScore(1)
        );
    }
}
