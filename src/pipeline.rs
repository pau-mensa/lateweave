//! Gather, optionally rerank, then deterministic top-k.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::query::Query;
use crate::ranking::validate_and_rank;
use crate::stage::{Candidate, CandidateGenerator, Requirements, Reranker, ResourceBudget, Score};

/// Per-search parameters.
#[derive(Clone, Copy, Debug)]
pub struct SearchRequest<'a> {
    pub gather_limit: usize,
    pub limit: usize,
    /// Ascending internal IDs the search is restricted to; a repeated ID
    /// counts once, and the gatherer receives each ID once.
    pub subset: Option<&'a [u64]>,
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

    pub fn with_subset(mut self, subset: &'a [u64]) -> Self {
        self.subset = Some(subset);
        self
    }

    pub fn with_budget(mut self, budget: ResourceBudget) -> Self {
        self.budget = budget;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankedDocument {
    pub document_id: u64,
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
    /// One score per candidate: the reranker's, or the gather scores.
    pub scores: Vec<Score>,
    pub timings: SearchTimings,
    pub diagnostics: SearchDiagnostics,
}

/// Stages must index the same corpus; that is checked once, at construction.
/// Each stage must be able to consume the query; that is checked per search,
/// before any stage runs, so a query that cannot be served fails without
/// gathering. Without a reranker the gather scores rank the results.
#[derive(Clone)]
pub struct SearchPipeline {
    gatherer: Arc<dyn CandidateGenerator>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl SearchPipeline {
    pub fn new(
        gatherer: Arc<dyn CandidateGenerator>,
        reranker: Option<Arc<dyn Reranker>>,
    ) -> Result<Self> {
        if let Some(reranker) = &reranker {
            gatherer.corpus().assert_compatible(reranker.corpus())?;
        }
        Ok(Self { gatherer, reranker })
    }

    pub fn gatherer(&self) -> &Arc<dyn CandidateGenerator> {
        &self.gatherer
    }

    pub fn reranker(&self) -> Option<&Arc<dyn Reranker>> {
        self.reranker.as_ref()
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
        let document_count = self.gatherer.corpus().document_count();
        let subset = request.subset.map(unique_subset).transpose()?;
        if let Some(subset) = &subset {
            if subset.last().is_some_and(|&last| last >= document_count) {
                return Err(Error::invalid(format!(
                    "subset reaches outside the corpus of {document_count} documents"
                )));
            }
        }
        let subset = subset.as_deref();
        require(query, self.gatherer.requires())?;
        if let Some(reranker) = &self.reranker {
            require(query, reranker.requires())?;
        }

        let started = Instant::now();
        let candidates = self.gatherer.gather(query, request.gather_limit, subset)?;
        let gathered = Instant::now();
        validate_candidates(&candidates, request.gather_limit, subset, document_count)?;
        let (scores, score_semantics) = match &self.reranker {
            Some(reranker) => (
                reranker.rerank(query, &candidates, &request.budget)?,
                reranker.score_semantics(),
            ),
            None => (
                candidates
                    .iter()
                    .map(|candidate| Score {
                        document_id: candidate.document_id,
                        value: candidate.gather_score,
                    })
                    .collect(),
                self.gatherer.score_semantics(),
            ),
        };
        let reranked = Instant::now();

        let documents = validate_and_rank(&candidates, &scores, request.limit)?
            .into_iter()
            .enumerate()
            .map(|(rank, position)| RankedDocument {
                document_id: scores[position].document_id,
                score: scores[position].value,
                rank: rank + 1,
            })
            .collect();
        let finished = Instant::now();
        Ok(SearchResult {
            documents,
            diagnostics: SearchDiagnostics {
                candidate_count: candidates.len(),
                gatherer: self.gatherer.name().to_string(),
                reranker: self
                    .reranker
                    .as_ref()
                    .map(|reranker| reranker.name().to_string()),
                score_semantics: score_semantics.to_string(),
            },
            candidates,
            scores,
            timings: SearchTimings {
                gather: gathered - started,
                rerank: reranked - gathered,
                total: finished - started,
            },
        })
    }
}

/// Checks presence and representation only, so a lazy feature is still
/// materialized by the first stage that reads it, or never.
fn require(query: &Query, requirements: &Requirements) -> Result<()> {
    for (name, representation) in requirements {
        query.require_feature(name, representation)?;
    }
    Ok(())
}

/// `subset` with repeats removed, borrowed when it has none.
fn unique_subset(subset: &[u64]) -> Result<Cow<'_, [u64]>> {
    if subset.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(Error::invalid("subset must be ascending"));
    }
    if subset.windows(2).all(|pair| pair[0] < pair[1]) {
        return Ok(Cow::Borrowed(subset));
    }
    let mut unique = subset.to_vec();
    unique.dedup();
    Ok(Cow::Owned(unique))
}

fn validate_candidates(
    candidates: &[Candidate],
    gather_limit: usize,
    subset: Option<&[u64]>,
    document_count: u64,
) -> Result<()> {
    if candidates.len() > gather_limit {
        return Err(Error::invalid(
            "gatherer returned more candidates than requested",
        ));
    }
    for (rank, candidate) in candidates.iter().enumerate() {
        if candidate.gather_rank != rank {
            return Err(Error::invalid(
                "candidate gather ranks must be contiguous and zero-based",
            ));
        }
        if candidate.document_id >= document_count {
            return Err(Error::invalid(format!(
                "gatherer returned document ID {}, outside the corpus of {document_count} documents",
                candidate.document_id
            )));
        }
        if let Some(subset) = subset {
            if subset.binary_search(&candidate.document_id).is_err() {
                return Err(Error::invalid(format!(
                    "gatherer returned document ID {}, which is outside the subset",
                    candidate.document_id
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::manifest::{CorpusManifest, Representation};
    use crate::query::{Feature, TokenMatrix};

    fn corpus() -> CorpusManifest {
        CorpusManifest::new("corpus", "1", 3, "abc").unwrap()
    }

    fn representation() -> Representation {
        Representation::new("encoder", "1", 2, true).unwrap()
    }

    struct TextGatherer {
        corpus: CorpusManifest,
        requires: Requirements,
        calls: AtomicUsize,
    }

    impl TextGatherer {
        fn new() -> Self {
            Self {
                corpus: corpus(),
                requires: BTreeMap::new(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl CandidateGenerator for TextGatherer {
        fn corpus(&self) -> &CorpusManifest {
            &self.corpus
        }

        fn requires(&self) -> &Requirements {
            &self.requires
        }

        fn score_semantics(&self) -> &str {
            "external-gather"
        }

        fn gather(
            &self,
            _: &Query,
            limit: usize,
            subset: Option<&[u64]>,
        ) -> Result<Vec<Candidate>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok([(2, 100.0), (0, 10.0), (1, 10.0)]
                .into_iter()
                .filter(|(document_id, _)| {
                    subset.map_or(true, |subset| subset.contains(document_id))
                })
                .take(limit)
                .enumerate()
                .map(|(rank, (document_id, gather_score))| Candidate {
                    document_id,
                    gather_score,
                    gather_rank: rank,
                    provenance: "external".to_string(),
                })
                .collect())
        }
    }

    struct VectorReranker {
        corpus: CorpusManifest,
        requires: Requirements,
    }

    impl VectorReranker {
        fn new(corpus: CorpusManifest) -> Self {
            Self {
                corpus,
                requires: BTreeMap::from([("multi_vector".to_string(), representation())]),
            }
        }
    }

    impl Reranker for VectorReranker {
        fn corpus(&self) -> &CorpusManifest {
            &self.corpus
        }

        fn requires(&self) -> &Requirements {
            &self.requires
        }

        fn score_semantics(&self) -> &str {
            "external-rerank"
        }

        fn rerank(
            &self,
            query: &Query,
            candidates: &[Candidate],
            _: &ResourceBudget,
        ) -> Result<Vec<Score>> {
            query.feature_as::<TokenMatrix>("multi_vector", &representation())?;
            Ok(candidates
                .iter()
                .map(|candidate| Score {
                    document_id: candidate.document_id,
                    value: [3.0, 5.0, 2.0][candidate.document_id as usize],
                })
                .collect())
        }
    }

    fn vector_query() -> Query {
        Query::new("query").with_feature(
            "multi_vector",
            Feature::new(
                representation(),
                TokenMatrix::new(vec![1.0, 1.0], 2).unwrap(),
            ),
        )
    }

    #[test]
    fn a_reranked_result_ignores_gather_scores() {
        let pipeline = SearchPipeline::new(
            Arc::new(TextGatherer::new()),
            Some(Arc::new(VectorReranker::new(corpus()))),
        )
        .unwrap();
        let result = pipeline
            .search(&vector_query(), &SearchRequest::new(3, 3))
            .unwrap();
        let ids = result
            .documents
            .iter()
            .map(|row| row.document_id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 0, 2]);
        assert_eq!(
            result.diagnostics,
            SearchDiagnostics {
                candidate_count: 3,
                gatherer: "TextGatherer".to_string(),
                reranker: Some("VectorReranker".to_string()),
                score_semantics: "external-rerank".to_string(),
            }
        );
    }

    #[test]
    fn without_a_reranker_gather_scores_rank() {
        let pipeline = SearchPipeline::new(Arc::new(TextGatherer::new()), None).unwrap();
        let result = pipeline
            .search(&Query::new("query"), &SearchRequest::new(3, 3))
            .unwrap();
        let ids = result
            .documents
            .iter()
            .map(|row| row.document_id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![2, 0, 1]);
        assert_eq!(result.diagnostics.score_semantics, "external-gather");
    }

    #[test]
    fn stages_must_index_the_same_corpus() {
        let error = SearchPipeline::new(
            Arc::new(TextGatherer::new()),
            Some(Arc::new(VectorReranker::new(corpus().with_generation(1)))),
        )
        .err()
        .unwrap();
        assert!(matches!(error, Error::IncompatibleIndex(_)));
    }

    #[test]
    fn an_unservable_query_fails_before_gathering() {
        let gatherer = Arc::new(TextGatherer::new());
        let pipeline = SearchPipeline::new(
            gatherer.clone(),
            Some(Arc::new(VectorReranker::new(corpus()))),
        )
        .unwrap();
        let error = pipeline
            .search(&Query::new("query"), &SearchRequest::new(3, 1))
            .unwrap_err();
        assert!(matches!(error, Error::IncompatibleQuery(_)));
        assert_eq!(gatherer.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn checking_requirements_leaves_lazy_features_unmaterialized() {
        let encodings = Arc::new(AtomicUsize::new(0));
        let counter = encodings.clone();
        let query = Query::new("query").with_feature(
            "multi_vector",
            Feature::lazy(representation(), move || {
                counter.fetch_add(1, Ordering::SeqCst);
                TokenMatrix::new(vec![1.0, 0.0], 2)
            }),
        );
        let mut gatherer = TextGatherer::new();
        gatherer.requires = BTreeMap::from([("multi_vector".to_string(), representation())]);
        let pipeline = SearchPipeline::new(Arc::new(gatherer), None).unwrap();
        pipeline.search(&query, &SearchRequest::new(3, 3)).unwrap();
        assert_eq!(encodings.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn subsets_must_be_ascending_and_inside_the_corpus() {
        let pipeline = SearchPipeline::new(Arc::new(TextGatherer::new()), None).unwrap();
        let query = Query::new("query");
        let result = pipeline
            .search(&query, &SearchRequest::new(3, 3).with_subset(&[0, 2]))
            .unwrap();
        assert_eq!(result.candidates.len(), 2);
        let repeated = pipeline
            .search(&query, &SearchRequest::new(3, 3).with_subset(&[0, 0, 2]))
            .unwrap();
        assert_eq!(repeated.candidates, result.candidates);
        for subset in [&[2, 0][..], &[0, 3][..]] {
            assert!(pipeline
                .search(&query, &SearchRequest::new(3, 3).with_subset(subset))
                .is_err());
        }
    }
}
