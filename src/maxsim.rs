//! MaxSim reranking over multi-vector sources, one per segment.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::kernel::maxsim_scores;
use crate::manifest::Representation;
use crate::query::{Query, TokenMatrix};
use crate::segment::Segment;
use crate::source::MultiVectorSource;
use crate::stage::{Candidate, Requirements, Reranker, ResourceBudget};
use crate::threads::install;

/// The feature name a [`MaxSimReranker`] reads unless told otherwise.
pub const DEFAULT_FEATURE: &str = "multi_vector";

/// MaxSim between the query's token matrix and each candidate's document,
/// read from the source of the candidate's segment.
///
/// Every source must carry the same representation and score semantics, so
/// scores from different segments share one scale; the reranker requires its
/// query feature to carry that representation, so vectors from a different
/// encoder are refused before anything is fetched. A candidate is scored only
/// against the snapshot of its segment the reranker holds.
pub struct MaxSimReranker {
    sources: BTreeMap<String, Arc<dyn MultiVectorSource>>,
    segments: Vec<Segment>,
    representation: Representation,
    score_semantics: String,
    feature: String,
    requires: Requirements,
}

impl MaxSimReranker {
    /// Sources must be for distinct segments.
    pub fn new(
        sources: impl IntoIterator<Item = Arc<dyn MultiVectorSource>>,
        feature: impl Into<String>,
    ) -> Result<Self> {
        let sources = sources.into_iter().collect::<Vec<_>>();
        let first = sources
            .first()
            .ok_or_else(|| Error::invalid("a MaxSim reranker needs at least one source"))?;
        let representation = first.representation().clone();
        let score_semantics = first.score_semantics().to_string();
        let mut by_corpus = BTreeMap::new();
        for source in &sources {
            let corpus_id = source.segment().corpus_id();
            source
                .representation()
                .assert_compatible(&representation)
                .map_err(|error| {
                    Error::IncompatibleIndex(format!(
                        "the source of segment {corpus_id:?} carries another representation: {error}"
                    ))
                })?;
            if source.score_semantics() != score_semantics {
                return Err(Error::IncompatibleIndex(format!(
                    "the source of segment {corpus_id:?} scores as {:?}, not {score_semantics:?}",
                    source.score_semantics()
                )));
            }
            if by_corpus
                .insert(corpus_id.to_string(), source.clone())
                .is_some()
            {
                return Err(Error::invalid(format!(
                    "more than one source for segment {corpus_id:?}"
                )));
            }
        }
        let feature = feature.into();
        Ok(Self {
            segments: sources
                .iter()
                .map(|source| source.segment().clone())
                .collect(),
            sources: by_corpus,
            requires: BTreeMap::from([(feature.clone(), representation.clone())]),
            representation,
            score_semantics,
            feature,
        })
    }

    /// In the order they were given.
    pub fn sources(&self) -> impl Iterator<Item = &Arc<dyn MultiVectorSource>> {
        self.segments
            .iter()
            .map(|segment| &self.sources[segment.corpus_id()])
    }

    pub fn feature(&self) -> &str {
        &self.feature
    }

    fn source(&self, segment: &Segment) -> Result<&Arc<dyn MultiVectorSource>> {
        let source = self.sources.get(segment.corpus_id()).ok_or_else(|| {
            Error::IncompatibleIndex(format!("no source for segment {:?}", segment.corpus_id()))
        })?;
        segment.assert_compatible(source.segment())?;
        Ok(source)
    }
}

/// Groups documents, shortest first, into batches of at most `maximum_tokens`
/// tokens; a document longer than that is a batch of its own.
fn token_batches(
    document_ids: &[u64],
    lengths: &HashMap<u64, usize>,
    maximum_tokens: usize,
) -> Vec<Vec<u64>> {
    let mut ordered = document_ids.to_vec();
    ordered.sort_unstable_by_key(|document_id| (lengths[document_id], *document_id));
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut tokens = 0;
    for document_id in ordered {
        let length = lengths[&document_id];
        if !batch.is_empty() && tokens + length > maximum_tokens {
            batches.push(std::mem::take(&mut batch));
            tokens = 0;
        }
        batch.push(document_id);
        tokens += length;
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

/// MaxSim of `vectors` against `candidate_ids` of one source, by ID.
fn score(
    source: &dyn MultiVectorSource,
    vectors: &TokenMatrix,
    candidate_ids: &[u64],
    budget: &ResourceBudget,
) -> Result<HashMap<u64, f32>> {
    let dimension = source.representation().dimension();
    let lengths = source.document_lengths(candidate_ids)?;
    if lengths.len() != candidate_ids.len() {
        return Err(Error::invalid(
            "source returned a different number of document lengths than requested",
        ));
    }
    let lengths = candidate_ids
        .iter()
        .copied()
        .zip(lengths)
        .collect::<HashMap<_, _>>();

    let mut scores = HashMap::with_capacity(candidate_ids.len());
    for window in candidate_ids.chunks(budget.max_documents_per_batch()) {
        for batch in token_batches(window, &lengths, budget.max_batch_tokens()) {
            let documents = source.fetch(&batch, budget.threads())?;
            let expected = batch.iter().map(|document_id| lengths[document_id]);
            if documents.dimension() != dimension
                || !documents.lengths().iter().copied().eq(expected)
            {
                return Err(Error::invalid(
                    "source fetched vectors that disagree with its declared lengths or dimension",
                ));
            }
            let values = maxsim_scores(
                vectors.values(),
                documents.vectors(),
                documents.lengths(),
                vectors.dimension(),
                Some(budget.max_batch_tokens()),
                budget.threads(),
            )?;
            scores.extend(batch.into_iter().zip(values));
        }
    }
    Ok(scores)
}

impl Reranker for MaxSimReranker {
    fn segments(&self) -> &[Segment] {
        &self.segments
    }

    fn requires(&self) -> &Requirements {
        &self.requires
    }

    fn score_semantics(&self) -> &str {
        &self.score_semantics
    }

    fn rerank(
        &self,
        query: &Query,
        candidates: &[Candidate],
        budget: &ResourceBudget,
    ) -> Result<Vec<f32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = query
            .feature(&self.feature, &self.representation)?
            .token_matrix()
            .ok_or_else(|| {
                Error::IncompatibleQuery(format!(
                    "query feature '{}' must be a [tokens, dimension] matrix",
                    self.feature
                ))
            })?;
        if vectors.dimension() != self.representation.dimension() {
            return Err(Error::IncompatibleQuery(format!(
                "query feature '{}' has dimension {}, but the representation declares {}",
                self.feature,
                vectors.dimension(),
                self.representation.dimension()
            )));
        }

        // Candidate positions per segment, each checked against the snapshot
        // this reranker holds before anything is fetched.
        let mut groups: BTreeMap<&str, (&Arc<dyn MultiVectorSource>, Vec<usize>)> = BTreeMap::new();
        for (position, candidate) in candidates.iter().enumerate() {
            let corpus_id = candidate.segment.corpus_id();
            let source = self.source(&candidate.segment)?;
            if !source.segment().contains(candidate.document_id) {
                return Err(Error::invalid(format!(
                    "document ID {} is outside segment {corpus_id:?}",
                    candidate.document_id
                )));
            }
            groups
                .entry(corpus_id)
                .or_insert_with(|| (source, Vec::new()))
                .1
                .push(position);
        }

        let mut output = vec![0.0; candidates.len()];
        install(budget.threads(), || {
            for (source, positions) in groups.values() {
                let ids = positions
                    .iter()
                    .map(|&position| candidates[position].document_id)
                    .collect::<Vec<_>>();
                let scores = score(source.as_ref(), vectors, &ids, budget)?;
                for (&position, document_id) in positions.iter().zip(ids) {
                    output[position] = scores[&document_id];
                }
            }
            Ok::<_, Error>(())
        })??;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::query::{Feature, TokenMatrix};
    use crate::source::PackedDocuments;

    struct InMemorySource {
        segment: Segment,
        representation: Representation,
        score_semantics: &'static str,
        documents: Vec<Vec<f32>>,
        fetched: Mutex<Vec<Vec<u64>>>,
    }

    impl InMemorySource {
        fn new(corpus_id: &str, documents: Vec<Vec<f32>>) -> Arc<Self> {
            Arc::new(Self {
                segment: Segment::new(
                    corpus_id,
                    "1",
                    0,
                    (0..documents.len()).map(|id| id.to_string()),
                )
                .unwrap(),
                representation: representation(),
                score_semantics: "in-memory-exact-full-maxsim",
                documents,
                fetched: Mutex::new(Vec::new()),
            })
        }
    }

    impl MultiVectorSource for InMemorySource {
        fn segment(&self) -> &Segment {
            &self.segment
        }

        fn representation(&self) -> &Representation {
            &self.representation
        }

        fn score_semantics(&self) -> &str {
            self.score_semantics
        }

        fn document_lengths(&self, document_ids: &[u64]) -> Result<Vec<usize>> {
            Ok(document_ids
                .iter()
                .map(|&document_id| self.documents[document_id as usize].len() / 2)
                .collect())
        }

        fn fetch(&self, document_ids: &[u64], _: Option<usize>) -> Result<PackedDocuments> {
            self.fetched.lock().unwrap().push(document_ids.to_vec());
            let vectors = document_ids
                .iter()
                .flat_map(|&document_id| self.documents[document_id as usize].iter().copied())
                .collect();
            PackedDocuments::new(vectors, self.document_lengths(document_ids)?, 2)
        }
    }

    fn representation() -> Representation {
        Representation::new("encoder", "1", 2, true).unwrap()
    }

    fn candidate(segment: &Segment, document_id: u64, gather_rank: usize) -> Candidate {
        Candidate {
            segment: segment.clone(),
            document_id,
            gather_score: 0.0,
            gather_rank,
            provenance: "test".to_string(),
        }
    }

    fn query(values: Vec<f32>) -> Query {
        Query::new("query").with_feature(
            DEFAULT_FEATURE,
            Feature::new(representation(), TokenMatrix::new(values, 2).unwrap()),
        )
    }

    #[test]
    fn scores_a_borrowed_source_in_candidate_order_within_the_token_budget() {
        let source = InMemorySource::new(
            "docs",
            vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.6, 0.8], vec![-1.0, 0.0]],
        );
        let reranker = MaxSimReranker::new(
            [source.clone() as Arc<dyn MultiVectorSource>],
            DEFAULT_FEATURE,
        )
        .unwrap();
        let segment = &source.segment;

        let scores = reranker
            .rerank(
                &query(vec![1.0, 0.0, 0.0, 1.0]),
                &[
                    candidate(segment, 1, 0),
                    candidate(segment, 0, 1),
                    candidate(segment, 2, 2),
                ],
                &ResourceBudget::new(2, 256, Some(1)).unwrap(),
            )
            .unwrap();

        for (score, expected) in scores.iter().zip([1.4, 2.0, -1.0]) {
            assert!((score - expected).abs() < 1e-6);
        }
        let mut sizes = source
            .fetched
            .lock()
            .unwrap()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 2]);
        assert_eq!(reranker.name(), "MaxSimReranker");
    }

    #[test]
    fn routes_each_candidate_to_its_segments_source() {
        let a = InMemorySource::new("a", vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let b = InMemorySource::new("b", vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let reranker = MaxSimReranker::new(
            [a.clone() as Arc<dyn MultiVectorSource>, b.clone()],
            DEFAULT_FEATURE,
        )
        .unwrap();
        assert_eq!(reranker.segments(), [a.segment.clone(), b.segment.clone()]);

        let scores = reranker
            .rerank(
                &query(vec![1.0, 0.0]),
                &[
                    candidate(&b.segment, 0, 0),
                    candidate(&a.segment, 0, 1),
                    candidate(&b.segment, 1, 2),
                ],
                &ResourceBudget::default(),
            )
            .unwrap();
        assert_eq!(scores, [0.0, 1.0, 1.0]);
        assert_eq!(*a.fetched.lock().unwrap(), [vec![0]]);
        assert_eq!(b.fetched.lock().unwrap().concat().len(), 2);
    }

    #[test]
    fn refuses_a_candidate_from_another_snapshot() {
        let source = InMemorySource::new("docs", vec![vec![1.0, 0.0]]);
        let reranker = MaxSimReranker::new(
            [source.clone() as Arc<dyn MultiVectorSource>],
            DEFAULT_FEATURE,
        )
        .unwrap();
        let later = source.segment.appended(["1"]).unwrap();
        let unknown = Segment::new("other", "1", 0, ["0"]).unwrap();
        for segment in [later, unknown] {
            let error = reranker
                .rerank(
                    &query(vec![1.0, 0.0]),
                    &[candidate(&segment, 0, 0)],
                    &ResourceBudget::default(),
                )
                .unwrap_err();
            assert!(matches!(error, Error::IncompatibleIndex(_)));
        }
        assert!(source.fetched.lock().unwrap().is_empty());
    }

    #[test]
    fn sources_must_share_representation_and_semantics() {
        let a = InMemorySource::new("a", vec![vec![1.0, 0.0]]);
        let other_semantics = Arc::new(InMemorySource {
            score_semantics: "lossy",
            ..Arc::try_unwrap(InMemorySource::new("b", vec![vec![1.0, 0.0]]))
                .ok()
                .unwrap()
        });
        assert!(matches!(
            MaxSimReranker::new(
                [a.clone() as Arc<dyn MultiVectorSource>, other_semantics],
                DEFAULT_FEATURE
            ),
            Err(Error::IncompatibleIndex(_))
        ));
        let duplicate = InMemorySource::new("a", vec![vec![1.0, 0.0]]);
        assert!(MaxSimReranker::new(
            [a as Arc<dyn MultiVectorSource>, duplicate],
            DEFAULT_FEATURE
        )
        .is_err());
        assert!(
            MaxSimReranker::new(Vec::<Arc<dyn MultiVectorSource>>::new(), DEFAULT_FEATURE).is_err()
        );
    }
}
