//! Gather, optionally rerank, then deterministic top-k.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use crate::error::{Error, Result};
use crate::query::Query;
use crate::ranking::validate_and_rank;
use crate::stage::{
    Candidate, CandidateGenerator, DocumentKey, Requirements, Reranker, ResourceBudget, Subset,
};

/// Per-search parameters.
#[derive(Clone, Copy, Debug)]
pub struct SearchRequest<'a> {
    pub gather_limit: usize,
    pub limit: usize,
    pub subset: Option<&'a Subset>,
    pub budget: ResourceBudget,
}

impl<'a> SearchRequest<'a> {
    pub fn new(gather_limit: usize, limit: usize) -> Self {
        Self {
            gather_limit,
            limit,
            subset: None,
            budget: ResourceBudget::default(),
        }
    }

    pub fn with_subset(mut self, subset: &'a Subset) -> Self {
        self.subset = Some(subset);
        self
    }

    pub fn with_budget(mut self, budget: ResourceBudget) -> Self {
        self.budget = budget;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedDocument {
    pub key: DocumentKey,
    pub score: f32,
    /// One-based.
    pub rank: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchTimings {
    pub gather: Duration,
    pub rerank: Duration,
    pub total: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchDiagnostics {
    pub candidate_count: usize,
    /// Candidates the reranker's index does not hold, left out of the ranking.
    pub dropped: usize,
    pub gatherer: String,
    pub reranker: Option<String>,
    /// Qualifies the scores in [`SearchResult::documents`].
    pub score_semantics: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub documents: Vec<RankedDocument>,
    /// Everything the gatherer returned, with provenance, in gather order.
    pub candidates: Vec<Candidate>,
    /// `scores[i]` scores `candidates[i]`: the reranker's, `None` when its
    /// index does not hold the document, or the gather score.
    pub scores: Vec<Option<f32>>,
    /// Every write committed to every index the search read before this is
    /// reflected in the result.
    pub as_of: SystemTime,
    pub timings: SearchTimings,
    pub diagnostics: SearchDiagnostics,
}

/// A gatherer, optionally followed by a reranker.
///
/// Stages speak in [`DocumentKey`]s and each reads its own indexes, which
/// writers anywhere move independently. A document is ranked only when every
/// stage holds it, so a delete takes effect as soon as any index applies it
/// and an insert once all of them have. Each search reports, as
/// [`SearchResult::as_of`], the oldest commit among the indexes it read.
///
/// Each stage must be able to consume the query; that is checked before any
/// stage runs, so a query that cannot be served fails without gathering.
/// Without a reranker the gather scores rank the results.
pub struct SearchPipeline {
    gatherer: Arc<dyn CandidateGenerator>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl SearchPipeline {
    pub fn new(gatherer: Arc<dyn CandidateGenerator>, reranker: Option<Arc<dyn Reranker>>) -> Self {
        Self { gatherer, reranker }
    }

    pub fn search(&self, query: &Query, request: &SearchRequest<'_>) -> Result<SearchResult> {
        if request.gather_limit == 0 {
            return Err(Error::invalid("gather_limit must be positive"));
        }
        if request.limit == 0 {
            return Err(Error::invalid("limit must be positive"));
        }
        if request.limit > request.gather_limit {
            return Err(Error::invalid("limit cannot exceed gather_limit"));
        }
        require(query, self.gatherer.requires())?;
        if let Some(reranker) = &self.reranker {
            require(query, reranker.requires())?;
        }

        let started = Instant::now();
        let gathered = self
            .gatherer
            .gather(query, request.gather_limit, request.subset)?;
        let gathered_at = Instant::now();
        validate(&gathered.candidates, request.gather_limit, request.subset)?;
        let candidates = gathered.candidates;
        let (scores, as_of, score_semantics) = match &self.reranker {
            Some(reranker) => {
                let scored = reranker.rerank(query, &candidates, &request.budget)?;
                (
                    scored.scores,
                    gathered.as_of.min(scored.as_of),
                    reranker.score_semantics(),
                )
            }
            None => (
                candidates
                    .iter()
                    .map(|candidate| Some(candidate.gather_score))
                    .collect(),
                gathered.as_of,
                self.gatherer.score_semantics(),
            ),
        };
        let reranked = Instant::now();

        let documents = validate_and_rank(&candidates, &scores, request.limit)?
            .into_iter()
            .enumerate()
            .map(|(rank, position)| RankedDocument {
                key: candidates[position].key.clone(),
                score: scores[position].expect("only scored candidates are ranked"),
                rank: rank + 1,
            })
            .collect();
        let finished = Instant::now();
        Ok(SearchResult {
            documents,
            diagnostics: SearchDiagnostics {
                candidate_count: candidates.len(),
                dropped: scores.iter().filter(|score| score.is_none()).count(),
                gatherer: self.gatherer.name().to_string(),
                reranker: self
                    .reranker
                    .as_ref()
                    .map(|reranker| reranker.name().to_string()),
                score_semantics: score_semantics.to_string(),
            },
            candidates,
            scores,
            as_of,
            timings: SearchTimings {
                gather: gathered_at - started,
                rerank: reranked - gathered_at,
                total: finished - started,
            },
        })
    }
}

fn validate(candidates: &[Candidate], gather_limit: usize, subset: Option<&Subset>) -> Result<()> {
    if candidates.len() > gather_limit {
        return Err(Error::invalid(
            "gatherer returned more candidates than requested",
        ));
    }
    let mut seen = HashSet::with_capacity(candidates.len());
    for (rank, candidate) in candidates.iter().enumerate() {
        if candidate.gather_rank != rank {
            return Err(Error::invalid(
                "candidate gather ranks must be contiguous and zero-based",
            ));
        }
        if subset.is_some_and(|subset| !subset.contains(&candidate.key)) {
            return Err(Error::invalid(format!(
                "gatherer returned document {:?}, which is outside the subset",
                candidate.key
            )));
        }
        if !seen.insert(&candidate.key) {
            return Err(Error::invalid(format!(
                "gatherer returned document {:?} more than once",
                candidate.key
            )));
        }
    }
    Ok(())
}

/// Checks presence and representation only, so a lazy feature is still
/// materialized by the first stage that reads it, or never.
fn require(query: &Query, requirements: &Requirements) -> Result<()> {
    for (name, representation) in requirements {
        query.require_feature(name, representation)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::query::{Feature, TokenMatrix};
    use crate::representation::Representation;
    use crate::stage::{Gathered, Scored};

    fn representation() -> Representation {
        Representation::new("encoder", "1", 2, true).unwrap()
    }

    fn key(id: &str) -> DocumentKey {
        DocumentKey::new("docs", id)
    }

    fn seconds_ago(seconds: u64) -> SystemTime {
        SystemTime::now() - Duration::from_secs(seconds)
    }

    /// Returns `rows` in order, filtered by the subset.
    struct FixedGatherer {
        rows: Vec<(DocumentKey, f32)>,
        as_of: SystemTime,
        requires: Requirements,
        calls: AtomicUsize,
    }

    impl FixedGatherer {
        fn new() -> Self {
            Self::over(vec![(key("z"), 100.0), (key("x"), 10.0), (key("y"), 10.0)])
        }

        fn over(rows: Vec<(DocumentKey, f32)>) -> Self {
            Self {
                rows,
                as_of: seconds_ago(5),
                requires: BTreeMap::new(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl CandidateGenerator for FixedGatherer {
        fn requires(&self) -> &Requirements {
            &self.requires
        }

        fn score_semantics(&self) -> &str {
            "external-gather"
        }

        fn gather(&self, _: &Query, limit: usize, subset: Option<&Subset>) -> Result<Gathered> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let candidates = self
                .rows
                .iter()
                .filter(|(key, _)| subset.map_or(true, |subset| subset.contains(key)))
                .take(limit)
                .enumerate()
                .map(|(rank, (key, gather_score))| Candidate {
                    key: key.clone(),
                    gather_score: *gather_score,
                    gather_rank: rank,
                    provenance: "external".to_string(),
                })
                .collect();
            Ok(Gathered {
                candidates,
                as_of: self.as_of,
            })
        }
    }

    /// Scores by ID; an ID it does not hold is unscored.
    struct TableReranker {
        scores: BTreeMap<String, f32>,
        as_of: SystemTime,
        requires: Requirements,
        received: Mutex<Vec<DocumentKey>>,
    }

    impl TableReranker {
        fn new(scores: &[(&str, f32)]) -> Self {
            Self {
                scores: scores
                    .iter()
                    .map(|&(id, score)| (id.to_string(), score))
                    .collect(),
                as_of: seconds_ago(1),
                requires: BTreeMap::from([("multi_vector".to_string(), representation())]),
                received: Mutex::new(Vec::new()),
            }
        }
    }

    impl Reranker for TableReranker {
        fn requires(&self) -> &Requirements {
            &self.requires
        }

        fn score_semantics(&self) -> &str {
            "table"
        }

        fn rerank(
            &self,
            _: &Query,
            candidates: &[Candidate],
            _: &ResourceBudget,
        ) -> Result<Scored> {
            *self.received.lock().unwrap() = candidates.iter().map(|c| c.key.clone()).collect();
            Ok(Scored {
                scores: candidates
                    .iter()
                    .map(|candidate| self.scores.get(candidate.key.id()).copied())
                    .collect(),
                as_of: self.as_of,
            })
        }
    }

    fn vector_query() -> Query {
        Query::new("query").with_feature(
            "multi_vector",
            Feature::new(
                representation(),
                TokenMatrix::new(vec![1.0, 0.0], 2).unwrap(),
            ),
        )
    }

    fn ids(result: &SearchResult) -> Vec<&str> {
        result
            .documents
            .iter()
            .map(|document| document.key.id())
            .collect()
    }

    #[test]
    fn the_reranker_orders_the_gathered_keys() {
        let reranker = Arc::new(TableReranker::new(&[("x", 3.0), ("y", 5.0), ("z", 2.0)]));
        let pipeline = SearchPipeline::new(Arc::new(FixedGatherer::new()), Some(reranker.clone()));
        let result = pipeline
            .search(&vector_query(), &SearchRequest::new(3, 2))
            .unwrap();
        assert_eq!(
            *reranker.received.lock().unwrap(),
            [key("z"), key("x"), key("y")]
        );
        assert_eq!(ids(&result), ["y", "x"]);
        assert_eq!(result.scores, [Some(2.0), Some(3.0), Some(5.0)]);
        assert_eq!(result.as_of, result.as_of.min(reranker.as_of));
        assert_eq!(
            result.diagnostics.reranker.as_deref(),
            Some("TableReranker")
        );
        assert_eq!(result.diagnostics.dropped, 0);
    }

    #[test]
    fn a_document_one_stage_lacks_is_dropped() {
        let gatherer = Arc::new(FixedGatherer::new());
        let reranker = Arc::new(TableReranker::new(&[("x", 3.0), ("z", 2.0)]));
        let result = SearchPipeline::new(gatherer.clone(), Some(reranker))
            .search(&vector_query(), &SearchRequest::new(3, 3))
            .unwrap();
        assert_eq!(ids(&result), ["x", "z"]);
        assert_eq!(result.scores, [Some(2.0), Some(3.0), None]);
        assert_eq!(result.diagnostics.dropped, 1);
        assert_eq!(result.as_of, gatherer.as_of);
    }

    #[test]
    fn without_a_reranker_gather_scores_rank_with_gather_rank_tie_break() {
        let gatherer = Arc::new(FixedGatherer::new());
        let result = SearchPipeline::new(gatherer.clone(), None)
            .search(&Query::new("query"), &SearchRequest::new(3, 3))
            .unwrap();
        assert_eq!(ids(&result), ["z", "x", "y"]);
        assert_eq!(result.as_of, gatherer.as_of);
        assert_eq!(result.diagnostics.score_semantics, "external-gather");
    }

    #[test]
    fn an_unservable_query_fails_before_gathering() {
        let gatherer = Arc::new(FixedGatherer::new());
        let pipeline =
            SearchPipeline::new(gatherer.clone(), Some(Arc::new(TableReranker::new(&[]))));
        let error = pipeline
            .search(&Query::new("query"), &SearchRequest::new(3, 3))
            .unwrap_err();
        assert!(matches!(error, Error::IncompatibleQuery(_)));
        assert_eq!(gatherer.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn the_subset_reaches_the_gatherer_and_bounds_its_candidates() {
        let subset = Subset::new().with("docs", ["x", "unknown"]);
        let result = SearchPipeline::new(Arc::new(FixedGatherer::new()), None)
            .search(
                &Query::new("query"),
                &SearchRequest::new(3, 3).with_subset(&subset),
            )
            .unwrap();
        assert_eq!(ids(&result), ["x"]);

        struct Ignoring(FixedGatherer);
        impl CandidateGenerator for Ignoring {
            fn requires(&self) -> &Requirements {
                self.0.requires()
            }
            fn score_semantics(&self) -> &str {
                self.0.score_semantics()
            }
            fn gather(&self, query: &Query, limit: usize, _: Option<&Subset>) -> Result<Gathered> {
                self.0.gather(query, limit, None)
            }
        }
        let error = SearchPipeline::new(Arc::new(Ignoring(FixedGatherer::new())), None)
            .search(
                &Query::new("query"),
                &SearchRequest::new(3, 3).with_subset(&subset),
            )
            .unwrap_err();
        assert!(error.to_string().contains("outside the subset"));
    }

    #[test]
    fn candidates_must_be_unique_with_dense_ranks() {
        let repeated = FixedGatherer::over(vec![(key("x"), 1.0), (key("x"), 0.5)]);
        let error = SearchPipeline::new(Arc::new(repeated), None)
            .search(&Query::new("query"), &SearchRequest::new(3, 3))
            .unwrap_err();
        assert!(error.to_string().contains("more than once"));

        let same_id_elsewhere =
            FixedGatherer::over(vec![(key("x"), 1.0), (DocumentKey::new("other", "x"), 0.5)]);
        assert!(SearchPipeline::new(Arc::new(same_id_elsewhere), None)
            .search(&Query::new("query"), &SearchRequest::new(3, 3))
            .is_ok());
    }

    #[test]
    fn limits_are_validated() {
        let pipeline = SearchPipeline::new(Arc::new(FixedGatherer::new()), None);
        for (gather_limit, limit) in [(0, 1), (3, 0), (2, 3)] {
            assert!(pipeline
                .search(
                    &Query::new("query"),
                    &SearchRequest::new(gather_limit, limit)
                )
                .is_err());
        }
    }
}
