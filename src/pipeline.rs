//! Gather, optionally rerank, then deterministic top-k.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::live::{Fixed, Live};
use crate::query::Query;
use crate::ranking::validate_and_rank;
use crate::segment::Segment;
use crate::stage::{Candidate, CandidateGenerator, Requirements, Reranker, ResourceBudget, Subset};

/// Per-search parameters.
#[derive(Clone, Copy, Debug)]
pub struct SearchRequest<'a> {
    pub gather_limit: usize,
    pub limit: usize,
    /// Ascending internal IDs per segment the search is restricted to; a
    /// repeated ID counts once, and the gatherer receives each ID once.
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
    pub segment: Segment,
    pub document_id: u64,
    pub score: f32,
    /// One-based.
    pub rank: usize,
}

impl RankedDocument {
    /// `None` only for a document built outside a search.
    pub fn external_id(&self) -> Option<&str> {
        self.segment.external(self.document_id)
    }
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
    /// The stages' current snapshots disagreed, as while a writer is between
    /// publishing one resource and the next, so the search ran on the last
    /// snapshots that agreed.
    pub stale: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub documents: Vec<RankedDocument>,
    /// Everything the gatherer returned, with provenance, in gather order.
    pub candidates: Vec<Candidate>,
    /// `scores[i]` scores `candidates[i]`: the reranker's, or the gather score.
    pub scores: Vec<f32>,
    pub timings: SearchTimings,
    pub diagnostics: SearchDiagnostics,
}

/// A gatherer over one or more segments, optionally followed by a reranker
/// that covers each of them.
///
/// Stages are [`Live`]: at the start of every search the pipeline asks each
/// for its current snapshot, so a generation any process publishes is served
/// on the next search, and a running search finishes on the snapshots it
/// started with. The snapshots must agree: the reranker must hold the same
/// snapshot of every segment the gatherer searches. While they disagree, as
/// between a writer publishing one resource and the next, searches run on
/// the last snapshots that agreed and report
/// [`stale`](SearchDiagnostics::stale). [`freeze`](SearchPipeline::freeze)
/// pins one set for good.
///
/// Each stage must be able to consume the query; that is checked per search,
/// before any stage runs, so a query that cannot be served fails without
/// gathering. Without a reranker the gather scores rank the results.
pub struct SearchPipeline {
    gatherer: Arc<dyn Live<dyn CandidateGenerator>>,
    reranker: Option<Arc<dyn Live<dyn Reranker>>>,
    agreed: RwLock<Stages>,
}

impl SearchPipeline {
    /// Fails unless the stages' current snapshots agree.
    pub fn new(
        gatherer: Arc<dyn Live<dyn CandidateGenerator>>,
        reranker: Option<Arc<dyn Live<dyn Reranker>>>,
    ) -> Result<Self> {
        let stages = Stages::new(
            gatherer.current()?,
            reranker
                .as_ref()
                .map(|reranker| reranker.current())
                .transpose()?,
        )?;
        Ok(Self {
            gatherer,
            reranker,
            agreed: RwLock::new(stages),
        })
    }

    /// A pipeline over stages that never move.
    pub fn fixed(
        gatherer: Arc<dyn CandidateGenerator>,
        reranker: Option<Arc<dyn Reranker>>,
    ) -> Result<Self> {
        Self::new(
            Arc::new(Fixed(gatherer)),
            reranker.map(|reranker| Arc::new(Fixed(reranker)) as Arc<dyn Live<dyn Reranker>>),
        )
    }

    /// A pipeline fixed on the snapshots the next search would run on, for
    /// searches that must all see the same generations.
    pub fn freeze(&self) -> Result<Self> {
        let (stages, _) = self.stages()?;
        Self::fixed(stages.gatherer, stages.reranker)
    }

    pub fn search(&self, query: &Query, request: &SearchRequest<'_>) -> Result<SearchResult> {
        let (stages, stale) = self.stages()?;
        stages.search(query, request, stale)
    }

    /// The current snapshots when they agree, otherwise the last that did.
    fn stages(&self) -> Result<(Stages, bool)> {
        let gatherer = self.gatherer.current()?;
        let reranker = self
            .reranker
            .as_ref()
            .map(|reranker| reranker.current())
            .transpose()?;
        let agreed = self
            .agreed
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if agreed.holds(&gatherer, reranker.as_ref()) {
            return Ok((agreed, false));
        }
        match Stages::new(gatherer, reranker) {
            Ok(stages) => {
                *self.agreed.write().unwrap_or_else(PoisonError::into_inner) = stages.clone();
                Ok((stages, false))
            }
            Err(Error::IncompatibleIndex(_)) => Ok((agreed, true)),
            Err(error) => Err(error),
        }
    }
}

/// Stage snapshots checked to agree.
#[derive(Clone)]
struct Stages {
    gatherer: Arc<dyn CandidateGenerator>,
    reranker: Option<Arc<dyn Reranker>>,
    segments: Arc<BTreeMap<String, Segment>>,
}

impl Stages {
    fn new(
        gatherer: Arc<dyn CandidateGenerator>,
        reranker: Option<Arc<dyn Reranker>>,
    ) -> Result<Self> {
        let segments = by_corpus(gatherer.segments(), "gatherer")?;
        if let Some(reranker) = &reranker {
            let covered = by_corpus(reranker.segments(), "reranker")?;
            for (corpus_id, segment) in &segments {
                let other = covered.get(corpus_id).ok_or_else(|| {
                    Error::IncompatibleIndex(format!(
                        "the reranker cannot score segment {corpus_id:?}, which the gatherer searches"
                    ))
                })?;
                segment.assert_compatible(other)?;
            }
        }
        Ok(Self {
            gatherer,
            reranker,
            segments: Arc::new(segments),
        })
    }

    fn holds(
        &self,
        gatherer: &Arc<dyn CandidateGenerator>,
        reranker: Option<&Arc<dyn Reranker>>,
    ) -> bool {
        Arc::ptr_eq(&self.gatherer, gatherer)
            && match (&self.reranker, reranker) {
                (Some(held), Some(current)) => Arc::ptr_eq(held, current),
                (None, None) => true,
                _ => false,
            }
    }

    fn search(
        &self,
        query: &Query,
        request: &SearchRequest<'_>,
        stale: bool,
    ) -> Result<SearchResult> {
        if request.gather_limit == 0 {
            return Err(Error::invalid("gather_limit must be positive"));
        }
        if request.limit == 0 {
            return Err(Error::invalid("limit must be positive"));
        }
        if request.limit > request.gather_limit {
            return Err(Error::invalid("limit cannot exceed gather_limit"));
        }
        let subset = request
            .subset
            .map(|subset| self.normalize(subset))
            .transpose()?;
        require(query, self.gatherer.requires())?;
        if let Some(reranker) = &self.reranker {
            require(query, reranker.requires())?;
        }

        let started = Instant::now();
        let candidates = self
            .gatherer
            .gather(query, request.gather_limit, subset.as_ref())?;
        let gathered = Instant::now();
        self.validate(&candidates, request.gather_limit, subset.as_ref())?;
        let (scores, score_semantics) = match &self.reranker {
            Some(reranker) => (
                reranker.rerank(query, &candidates, &request.budget)?,
                reranker.score_semantics(),
            ),
            None => (
                candidates
                    .iter()
                    .map(|candidate| candidate.gather_score)
                    .collect(),
                self.gatherer.score_semantics(),
            ),
        };
        let reranked = Instant::now();

        let documents = validate_and_rank(&candidates, &scores, request.limit)?
            .into_iter()
            .enumerate()
            .map(|(rank, position)| RankedDocument {
                segment: candidates[position].segment.clone(),
                document_id: candidates[position].document_id,
                score: scores[position],
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
                stale,
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

    /// `subset` with repeats removed and an entry, possibly empty, for every
    /// segment the gatherer searches.
    fn normalize(&self, subset: &Subset) -> Result<Subset> {
        for (corpus_id, _) in subset.iter() {
            if !self.segments.contains_key(corpus_id) {
                return Err(Error::invalid(format!(
                    "subset names segment {corpus_id:?}, which the gatherer does not search"
                )));
            }
        }
        let mut normalized = Subset::new();
        for (corpus_id, segment) in self.segments.iter() {
            let ids = subset.ids(corpus_id);
            if ids.windows(2).any(|pair| pair[0] > pair[1]) {
                return Err(Error::invalid(format!(
                    "subset of segment {corpus_id:?} must be ascending"
                )));
            }
            if ids.last().is_some_and(|&last| !segment.contains(last)) {
                return Err(Error::invalid(format!(
                    "subset reaches outside segment {corpus_id:?} of {} documents",
                    segment.document_count()
                )));
            }
            let mut unique = ids.to_vec();
            unique.dedup();
            normalized = normalized.with(corpus_id.clone(), unique);
        }
        Ok(normalized)
    }

    fn validate(
        &self,
        candidates: &[Candidate],
        gather_limit: usize,
        subset: Option<&Subset>,
    ) -> Result<()> {
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
            let corpus_id = candidate.segment.corpus_id();
            let segment = self.segments.get(corpus_id).ok_or_else(|| {
                Error::invalid(format!(
                    "gatherer returned a candidate from segment {corpus_id:?}, which it does not declare"
                ))
            })?;
            candidate.segment.assert_compatible(segment)?;
            if !segment.contains(candidate.document_id) {
                return Err(Error::invalid(format!(
                    "gatherer returned document ID {}, outside segment {corpus_id:?} of {} documents",
                    candidate.document_id,
                    segment.document_count()
                )));
            }
            if let Some(subset) = subset {
                if subset
                    .ids(corpus_id)
                    .binary_search(&candidate.document_id)
                    .is_err()
                {
                    return Err(Error::invalid(format!(
                        "gatherer returned document ID {} of segment {corpus_id:?}, which is outside the subset",
                        candidate.document_id
                    )));
                }
            }
            if !seen.insert((corpus_id, candidate.document_id)) {
                return Err(Error::invalid(format!(
                    "gatherer returned document ID {} of segment {corpus_id:?} more than once",
                    candidate.document_id
                )));
            }
        }
        Ok(())
    }
}

fn by_corpus(segments: &[Segment], stage: &str) -> Result<BTreeMap<String, Segment>> {
    let mut by_corpus = BTreeMap::new();
    for segment in segments {
        let corpus_id = segment.corpus_id().to_string();
        if by_corpus.insert(corpus_id, segment.clone()).is_some() {
            return Err(Error::invalid(format!(
                "the {stage} declares segment {:?} more than once",
                segment.corpus_id()
            )));
        }
    }
    Ok(by_corpus)
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

    use super::*;
    use crate::manifest::Representation;
    use crate::query::{Feature, TokenMatrix};

    fn segment(corpus_id: &str) -> Segment {
        Segment::new(corpus_id, "1", 0, ["x", "y", "z"]).unwrap()
    }

    fn representation() -> Representation {
        Representation::new("encoder", "1", 2, true).unwrap()
    }

    /// Returns `rows` from its segments, in order, filtered by the subset.
    struct FixedGatherer {
        segments: Vec<Segment>,
        rows: Vec<(usize, u64, f32)>,
        requires: Requirements,
        calls: AtomicUsize,
    }

    impl FixedGatherer {
        fn new() -> Self {
            Self::over(
                vec![segment("docs")],
                vec![(0, 2, 100.0), (0, 0, 10.0), (0, 1, 10.0)],
            )
        }

        fn over(segments: Vec<Segment>, rows: Vec<(usize, u64, f32)>) -> Self {
            Self {
                segments,
                rows,
                requires: BTreeMap::new(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl CandidateGenerator for FixedGatherer {
        fn segments(&self) -> &[Segment] {
            &self.segments
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
            subset: Option<&Subset>,
        ) -> Result<Vec<Candidate>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .rows
                .iter()
                .map(|&(segment, document_id, score)| (&self.segments[segment], document_id, score))
                .filter(|(segment, document_id, _)| {
                    subset.map_or(true, |subset| {
                        subset.ids(segment.corpus_id()).contains(document_id)
                    })
                })
                .take(limit)
                .enumerate()
                .map(|(rank, (segment, document_id, gather_score))| Candidate {
                    segment: segment.clone(),
                    document_id,
                    gather_score,
                    gather_rank: rank,
                    provenance: "external".to_string(),
                })
                .collect())
        }
    }

    /// Scores by `(corpus_id, document_id)`.
    struct TableReranker {
        segments: Vec<Segment>,
        requires: Requirements,
        scores: BTreeMap<(String, u64), f32>,
    }

    impl TableReranker {
        fn new(segments: Vec<Segment>) -> Self {
            let scores = segments
                .iter()
                .flat_map(|segment| {
                    [3.0, 5.0, 2.0]
                        .into_iter()
                        .enumerate()
                        .map(|(document_id, score)| {
                            ((segment.corpus_id().to_string(), document_id as u64), score)
                        })
                })
                .collect();
            Self {
                segments,
                requires: BTreeMap::from([("multi_vector".to_string(), representation())]),
                scores,
            }
        }
    }

    impl Reranker for TableReranker {
        fn segments(&self) -> &[Segment] {
            &self.segments
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
        ) -> Result<Vec<f32>> {
            query.feature_as::<TokenMatrix>("multi_vector", &representation())?;
            Ok(candidates
                .iter()
                .map(|candidate| {
                    let key = (
                        candidate.segment.corpus_id().to_string(),
                        candidate.document_id,
                    );
                    self.scores.get(&key).copied().unwrap_or(1.0)
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

    fn ids(result: &SearchResult) -> Vec<(&str, u64)> {
        result
            .documents
            .iter()
            .map(|row| (row.segment.corpus_id(), row.document_id))
            .collect()
    }

    #[test]
    fn a_reranked_result_ignores_gather_scores() {
        let pipeline = SearchPipeline::fixed(
            Arc::new(FixedGatherer::new()),
            Some(Arc::new(TableReranker::new(vec![segment("docs")]))),
        )
        .unwrap();
        let result = pipeline
            .search(&vector_query(), &SearchRequest::new(3, 3))
            .unwrap();
        assert_eq!(ids(&result), [("docs", 1), ("docs", 0), ("docs", 2)]);
        assert_eq!(result.scores, [2.0, 3.0, 5.0]);
        assert_eq!(result.documents[0].external_id(), Some("y"));
        assert_eq!(
            result.diagnostics,
            SearchDiagnostics {
                candidate_count: 3,
                gatherer: "FixedGatherer".to_string(),
                reranker: Some("TableReranker".to_string()),
                score_semantics: "external-rerank".to_string(),
                stale: false,
            }
        );
    }

    #[test]
    fn without_a_reranker_gather_scores_rank() {
        let pipeline = SearchPipeline::fixed(Arc::new(FixedGatherer::new()), None).unwrap();
        let result = pipeline
            .search(&Query::new("query"), &SearchRequest::new(3, 3))
            .unwrap();
        assert_eq!(ids(&result), [("docs", 2), ("docs", 0), ("docs", 1)]);
        assert_eq!(result.diagnostics.score_semantics, "external-gather");
    }

    #[test]
    fn one_gatherer_ranks_candidates_from_several_segments() {
        let segments = vec![segment("a"), segment("b")];
        let gatherer = FixedGatherer::over(
            segments.clone(),
            vec![(0, 0, 1.0), (1, 0, 1.0), (1, 1, 1.0), (0, 2, 1.0)],
        );
        let pipeline = SearchPipeline::fixed(
            Arc::new(gatherer),
            Some(Arc::new(TableReranker::new(segments))),
        )
        .unwrap();
        let result = pipeline
            .search(&vector_query(), &SearchRequest::new(4, 4))
            .unwrap();
        assert_eq!(ids(&result), [("b", 1), ("a", 0), ("b", 0), ("a", 2)]);
    }

    #[test]
    fn the_reranker_must_hold_every_segment_the_gatherer_searches() {
        let gatherer = || {
            Arc::new(FixedGatherer::over(
                vec![segment("a"), segment("b")],
                vec![],
            ))
        };
        let missing = SearchPipeline::fixed(
            gatherer(),
            Some(Arc::new(TableReranker::new(vec![segment("a")]))),
        )
        .err()
        .unwrap();
        assert!(matches!(missing, Error::IncompatibleIndex(message) if message.contains("\"b\"")));

        let stale = segment("b").appended(["w"]).unwrap();
        let error = SearchPipeline::fixed(
            gatherer(),
            Some(Arc::new(TableReranker::new(vec![segment("a"), stale]))),
        )
        .err()
        .unwrap();
        assert!(matches!(error, Error::IncompatibleIndex(_)));

        let wider = SearchPipeline::fixed(
            Arc::new(FixedGatherer::over(vec![segment("a")], vec![])),
            Some(Arc::new(TableReranker::new(vec![
                segment("a"),
                segment("b"),
            ]))),
        );
        assert!(wider.is_ok());
    }

    #[test]
    fn a_segment_is_declared_once() {
        let gatherer = FixedGatherer::over(vec![segment("a"), segment("a")], vec![]);
        assert!(SearchPipeline::fixed(Arc::new(gatherer), None).is_err());
    }

    #[test]
    fn candidates_must_come_from_a_declared_snapshot() {
        let foreign = FixedGatherer {
            rows: vec![(1, 0, 1.0)],
            ..FixedGatherer::over(vec![segment("a"), segment("b")], vec![])
        };
        let mut declared = foreign.segments.clone();
        declared.truncate(1);
        struct Declares(FixedGatherer, Vec<Segment>);
        impl CandidateGenerator for Declares {
            fn segments(&self) -> &[Segment] {
                &self.1
            }
            fn requires(&self) -> &Requirements {
                self.0.requires()
            }
            fn score_semantics(&self) -> &str {
                self.0.score_semantics()
            }
            fn gather(
                &self,
                query: &Query,
                limit: usize,
                subset: Option<&Subset>,
            ) -> Result<Vec<Candidate>> {
                self.0.gather(query, limit, subset)
            }
        }
        let pipeline = SearchPipeline::fixed(Arc::new(Declares(foreign, declared)), None).unwrap();
        let error = pipeline
            .search(&Query::new("query"), &SearchRequest::new(1, 1))
            .unwrap_err();
        assert!(error.to_string().contains("does not declare"));

        let stale = FixedGatherer::over(
            vec![segment("a").appended(["w"]).unwrap()],
            vec![(0, 3, 1.0)],
        );
        let pipeline =
            SearchPipeline::fixed(Arc::new(Declares(stale, vec![segment("a")])), None).unwrap();
        let error = pipeline
            .search(&Query::new("query"), &SearchRequest::new(1, 1))
            .unwrap_err();
        assert!(matches!(error, Error::IncompatibleIndex(_)));
    }

    #[test]
    fn duplicate_candidates_are_refused() {
        let gatherer = FixedGatherer::over(vec![segment("a")], vec![(0, 1, 1.0), (0, 1, 1.0)]);
        let pipeline = SearchPipeline::fixed(Arc::new(gatherer), None).unwrap();
        let error = pipeline
            .search(&Query::new("query"), &SearchRequest::new(2, 2))
            .unwrap_err();
        assert!(error.to_string().contains("more than once"));
    }

    #[test]
    fn an_unservable_query_fails_before_gathering() {
        let gatherer = Arc::new(FixedGatherer::new());
        let pipeline = SearchPipeline::fixed(
            gatherer.clone(),
            Some(Arc::new(TableReranker::new(vec![segment("docs")]))),
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
        let mut gatherer = FixedGatherer::new();
        gatherer.requires = BTreeMap::from([("multi_vector".to_string(), representation())]);
        let pipeline = SearchPipeline::fixed(Arc::new(gatherer), None).unwrap();
        pipeline.search(&query, &SearchRequest::new(3, 3)).unwrap();
        assert_eq!(encodings.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn subsets_are_per_segment_ascending_and_inside_it() {
        let segments = vec![segment("a"), segment("b")];
        let gatherer = FixedGatherer::over(segments, vec![(0, 0, 3.0), (1, 0, 2.0), (1, 2, 1.0)]);
        let pipeline = SearchPipeline::fixed(Arc::new(gatherer), None).unwrap();
        let query = Query::new("query");

        let subset = Subset::new().with("b", vec![0, 0, 2]);
        let result = pipeline
            .search(&query, &SearchRequest::new(3, 3).with_subset(&subset))
            .unwrap();
        assert_eq!(ids(&result), [("b", 0), ("b", 2)]);

        for subset in [
            Subset::new().with("b", vec![2, 0]),
            Subset::new().with("b", vec![0, 3]),
            Subset::new().with("c", vec![0]),
        ] {
            assert!(pipeline
                .search(&query, &SearchRequest::new(3, 3).with_subset(&subset))
                .is_err());
        }
    }

    #[test]
    fn a_gatherer_that_ignores_the_subset_is_refused() {
        struct Ignores(FixedGatherer);
        impl CandidateGenerator for Ignores {
            fn segments(&self) -> &[Segment] {
                self.0.segments()
            }
            fn requires(&self) -> &Requirements {
                self.0.requires()
            }
            fn score_semantics(&self) -> &str {
                self.0.score_semantics()
            }
            fn gather(
                &self,
                query: &Query,
                limit: usize,
                _: Option<&Subset>,
            ) -> Result<Vec<Candidate>> {
                self.0.gather(query, limit, None)
            }
        }
        let pipeline =
            SearchPipeline::fixed(Arc::new(Ignores(FixedGatherer::new())), None).unwrap();
        let subset = Subset::new().with("docs", vec![0]);
        let error = pipeline
            .search(
                &Query::new("query"),
                &SearchRequest::new(3, 3).with_subset(&subset),
            )
            .unwrap_err();
        assert!(error.to_string().contains("outside the subset"));
    }

    /// A stage another writer moves by replacing what it holds.
    struct Moving<S: ?Sized>(std::sync::RwLock<Arc<S>>);

    impl<S: ?Sized + Send + Sync> Moving<S> {
        fn new(stage: Arc<S>) -> Arc<Self> {
            Arc::new(Self(std::sync::RwLock::new(stage)))
        }

        fn publish(&self, stage: Arc<S>) {
            *self.0.write().unwrap() = stage;
        }
    }

    impl<S: ?Sized + Send + Sync> Live<S> for Moving<S> {
        fn current(&self) -> Result<Arc<S>> {
            Ok(self.0.read().unwrap().clone())
        }
    }

    fn external_ids(result: &SearchResult) -> Vec<&str> {
        result
            .documents
            .iter()
            .map(|row| row.external_id().unwrap())
            .collect()
    }

    #[test]
    fn a_live_pipeline_serves_each_publish_once_its_stages_agree() {
        let first = segment("docs");
        let second = first.appended(["w"]).unwrap();
        let gatherer = |segment: &Segment, id: u64| -> Arc<dyn CandidateGenerator> {
            Arc::new(FixedGatherer::over(
                vec![segment.clone()],
                vec![(0, id, 1.0)],
            ))
        };
        let reranker = |segment: &Segment| -> Arc<dyn Reranker> {
            Arc::new(TableReranker::new(vec![segment.clone()]))
        };
        let live_gatherer = Moving::new(gatherer(&first, 0));
        let live_reranker = Moving::new(reranker(&first));
        let pipeline = SearchPipeline::new(
            live_gatherer.clone(),
            Some(live_reranker.clone() as Arc<dyn Live<dyn Reranker>>),
        )
        .unwrap();
        let frozen = pipeline.freeze().unwrap();
        let search = |pipeline: &SearchPipeline| {
            pipeline
                .search(&vector_query(), &SearchRequest::new(1, 1))
                .unwrap()
        };
        assert_eq!(external_ids(&search(&pipeline)), ["x"]);

        // The gatherer moved first: the last agreeing snapshots still serve.
        live_gatherer.publish(gatherer(&second, 3));
        let result = search(&pipeline);
        assert!(result.diagnostics.stale);
        assert_eq!(external_ids(&result), ["x"]);

        live_reranker.publish(reranker(&second));
        let result = search(&pipeline);
        assert!(!result.diagnostics.stale);
        assert_eq!(external_ids(&result), ["w"]);
        assert_eq!(result.documents[0].segment.generation(), 1);

        let result = search(&frozen);
        assert_eq!(external_ids(&result), ["x"]);
        assert_eq!(result.documents[0].segment.generation(), 0);
    }

    #[test]
    fn a_live_pipeline_must_agree_when_built() {
        let gatherer: Arc<dyn CandidateGenerator> = Arc::new(FixedGatherer::new());
        let reranker: Arc<dyn Reranker> = Arc::new(TableReranker::new(vec![segment("other")]));
        assert!(matches!(
            SearchPipeline::new(Moving::new(gatherer), Some(Moving::new(reranker))),
            Err(Error::IncompatibleIndex(_))
        ));
    }
}
