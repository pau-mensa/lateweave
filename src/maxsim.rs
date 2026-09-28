//! MaxSim reranking over multi-vector sources, one per corpus.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::kernel::maxsim_scores;
use crate::query::{Query, TokenMatrix};
use crate::representation::Representation;
use crate::source::{MultiVectorSource, VectorView};
use crate::stage::{Candidate, Requirements, Reranker, ResourceBudget, Scored};
use crate::threads::install;

/// The feature name a [`MaxSimReranker`] reads unless told otherwise.
pub const DEFAULT_FEATURE: &str = "multi_vector";

/// MaxSim between the query's token matrix and each candidate's document,
/// read from the source of the candidate's corpus.
///
/// Every source must carry the same representation and score semantics, so
/// scores from different corpora share one scale; the reranker requires its
/// query feature to carry that representation, so vectors from a different
/// encoder are refused before anything is fetched. Each rerank takes one
/// view of every source, and a candidate its source's view does not hold is
/// unscored.
pub struct MaxSimReranker {
    sources: BTreeMap<String, Arc<dyn MultiVectorSource>>,
    representation: Representation,
    score_semantics: String,
    feature: String,
    requires: Requirements,
}

impl MaxSimReranker {
    /// Sources must be for distinct corpora.
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
        for source in sources {
            let corpus = source.corpus().to_string();
            source
                .representation()
                .assert_compatible(&representation)
                .map_err(|error| {
                    Error::IncompatibleIndex(format!(
                        "the source of corpus {corpus:?} carries another representation: {error}"
                    ))
                })?;
            if source.score_semantics() != score_semantics {
                return Err(Error::IncompatibleIndex(format!(
                    "the source of corpus {corpus:?} scores as {:?}, not {score_semantics:?}",
                    source.score_semantics()
                )));
            }
            if by_corpus.contains_key(&corpus) {
                return Err(Error::invalid(format!(
                    "more than one source for corpus {corpus:?}"
                )));
            }
            by_corpus.insert(corpus, source);
        }
        let feature = feature.into();
        Ok(Self {
            sources: by_corpus,
            requires: BTreeMap::from([(feature.clone(), representation.clone())]),
            representation,
            score_semantics,
            feature,
        })
    }

    pub fn feature(&self) -> &str {
        &self.feature
    }

    fn query_vectors<'q>(&self, query: &'q Query) -> Result<&'q TokenMatrix> {
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
        Ok(vectors)
    }
}

/// Groups documents, shortest first, into batches of at most `maximum_tokens`
/// tokens; a document longer than that is a batch of its own.
fn token_batches<'a>(
    document_ids: &[&'a str],
    lengths: &HashMap<&str, usize>,
    maximum_tokens: usize,
) -> Vec<Vec<&'a str>> {
    let mut ordered = document_ids.to_vec();
    ordered.sort_unstable_by_key(|document_id| (lengths[document_id], *document_id));
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut tokens = 0;
    for document_id in ordered {
        let length = lengths[document_id];
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

/// MaxSim of `vectors` against the unique `document_ids` of one view, by ID;
/// an ID the view does not hold has no score.
fn score<'a>(
    view: &dyn VectorView,
    dimension: usize,
    vectors: &TokenMatrix,
    document_ids: &[&'a str],
    budget: &ResourceBudget,
) -> Result<HashMap<&'a str, f32>> {
    let lengths = view.document_lengths(document_ids)?;
    if lengths.len() != document_ids.len() {
        return Err(Error::invalid(
            "source returned a different number of document lengths than requested",
        ));
    }
    let lengths = document_ids
        .iter()
        .zip(lengths)
        .filter_map(|(&document_id, length)| Some((document_id, length?)))
        .collect::<HashMap<_, _>>();
    let held = document_ids
        .iter()
        .copied()
        .filter(|document_id| lengths.contains_key(document_id))
        .collect::<Vec<_>>();

    let mut scores = HashMap::with_capacity(held.len());
    for window in held.chunks(budget.max_documents_per_batch()) {
        for batch in token_batches(window, &lengths, budget.max_batch_tokens()) {
            let documents = view.fetch(&batch, budget.threads())?;
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
    ) -> Result<Scored> {
        let mut groups: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for candidate in candidates {
            let corpus = candidate.key.corpus();
            if !self.sources.contains_key(corpus) {
                return Err(Error::IncompatibleIndex(format!(
                    "the reranker has no source for corpus {corpus:?}"
                )));
            }
            groups.entry(corpus).or_default().push(candidate.key.id());
        }
        let views = self
            .sources
            .iter()
            .map(|(corpus, source)| Ok((corpus.as_str(), source.view()?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let as_of = views
            .values()
            .map(|view| view.as_of())
            .min()
            .expect("a MaxSim reranker has at least one source");
        if candidates.is_empty() {
            return Ok(Scored {
                scores: Vec::new(),
                as_of,
            });
        }
        let vectors = self.query_vectors(query)?;
        let dimension = self.representation.dimension();

        let scores = install(budget.threads(), || {
            groups
                .iter_mut()
                .map(|(corpus, document_ids)| {
                    document_ids.sort_unstable();
                    document_ids.dedup();
                    let scores = score(
                        views[corpus].as_ref(),
                        dimension,
                        vectors,
                        document_ids,
                        budget,
                    )?;
                    Ok((*corpus, scores))
                })
                .collect::<Result<BTreeMap<_, _>>>()
        })??;
        Ok(Scored {
            scores: candidates
                .iter()
                .map(|candidate| {
                    scores[candidate.key.corpus()]
                        .get(candidate.key.id())
                        .copied()
                })
                .collect(),
            as_of,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::{Duration, SystemTime};

    use super::*;
    use crate::query::Feature;
    use crate::source::PackedDocuments;
    use crate::stage::DocumentKey;

    struct InMemorySource {
        corpus: &'static str,
        representation: Representation,
        score_semantics: &'static str,
        view: Arc<InMemoryView>,
    }

    struct InMemoryView {
        documents: HashMap<&'static str, Vec<f32>>,
        as_of: SystemTime,
        fetched: Mutex<Vec<Vec<String>>>,
    }

    impl InMemorySource {
        fn new(corpus: &'static str, documents: &[(&'static str, Vec<f32>)]) -> Arc<Self> {
            Arc::new(Self {
                corpus,
                representation: representation(),
                score_semantics: "in-memory-exact-full-maxsim",
                view: Arc::new(InMemoryView {
                    documents: documents.iter().cloned().collect(),
                    as_of: SystemTime::UNIX_EPOCH + Duration::from_secs(1_000),
                    fetched: Mutex::new(Vec::new()),
                }),
            })
        }

        fn fetched(&self) -> Vec<Vec<String>> {
            self.view.fetched.lock().unwrap().clone()
        }
    }

    impl MultiVectorSource for InMemorySource {
        fn corpus(&self) -> &str {
            self.corpus
        }

        fn representation(&self) -> &Representation {
            &self.representation
        }

        fn score_semantics(&self) -> &str {
            self.score_semantics
        }

        fn view(&self) -> Result<Arc<dyn VectorView>> {
            Ok(self.view.clone())
        }
    }

    impl VectorView for InMemoryView {
        fn as_of(&self) -> SystemTime {
            self.as_of
        }

        fn document_lengths(&self, document_ids: &[&str]) -> Result<Vec<Option<usize>>> {
            Ok(document_ids
                .iter()
                .map(|document_id| {
                    self.documents
                        .get(document_id)
                        .map(|values| values.len() / 2)
                })
                .collect())
        }

        fn fetch(&self, document_ids: &[&str], _: Option<usize>) -> Result<PackedDocuments> {
            self.fetched
                .lock()
                .unwrap()
                .push(document_ids.iter().map(|id| id.to_string()).collect());
            let vectors = document_ids
                .iter()
                .flat_map(|document_id| self.documents[document_id].iter().copied())
                .collect();
            let lengths = self
                .document_lengths(document_ids)?
                .into_iter()
                .flatten()
                .collect();
            PackedDocuments::new(vectors, lengths, 2)
        }
    }

    fn representation() -> Representation {
        Representation::new("encoder", "1", 2, true).unwrap()
    }

    fn candidate(corpus: &str, id: &str, gather_rank: usize) -> Candidate {
        Candidate {
            key: DocumentKey::new(corpus, id),
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

    fn reranker(sources: &[Arc<InMemorySource>]) -> MaxSimReranker {
        MaxSimReranker::new(
            sources
                .iter()
                .map(|source| source.clone() as Arc<dyn MultiVectorSource>),
            DEFAULT_FEATURE,
        )
        .unwrap()
    }

    #[test]
    fn scores_in_candidate_order_within_the_token_budget() {
        let source = InMemorySource::new(
            "docs",
            &[
                ("a", vec![1.0, 0.0, 0.0, 1.0]),
                ("b", vec![0.6, 0.8]),
                ("c", vec![-1.0, 0.0]),
            ],
        );
        let scored = reranker(std::slice::from_ref(&source))
            .rerank(
                &query(vec![1.0, 0.0, 0.0, 1.0]),
                &[
                    candidate("docs", "b", 0),
                    candidate("docs", "a", 1),
                    candidate("docs", "c", 2),
                ],
                &ResourceBudget::new(2, 256, Some(1)).unwrap(),
            )
            .unwrap();

        for (score, expected) in scored.scores.iter().zip([1.4, 2.0, -1.0]) {
            assert!((score.unwrap() - expected).abs() < 1e-6);
        }
        let mut sizes = source.fetched().iter().map(Vec::len).collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, [1, 2]);
        assert_eq!(scored.as_of, source.view.as_of);
    }

    #[test]
    fn a_document_the_source_lacks_is_unscored_and_never_fetched() {
        let source = InMemorySource::new("docs", &[("a", vec![1.0, 0.0])]);
        let scored = reranker(std::slice::from_ref(&source))
            .rerank(
                &query(vec![1.0, 0.0]),
                &[candidate("docs", "gone", 0), candidate("docs", "a", 1)],
                &ResourceBudget::default(),
            )
            .unwrap();
        assert_eq!(scored.scores, [None, Some(1.0)]);
        assert_eq!(source.fetched(), [vec!["a".to_string()]]);
    }

    #[test]
    fn routes_each_candidate_to_its_corpus_and_reports_the_oldest_view() {
        let a = InMemorySource::new("a", &[("0", vec![1.0, 0.0]), ("1", vec![0.0, 1.0])]);
        let b = Arc::new(InMemorySource {
            view: Arc::new(InMemoryView {
                documents: [("0", vec![0.0, 1.0]), ("1", vec![1.0, 0.0])]
                    .into_iter()
                    .collect(),
                as_of: SystemTime::UNIX_EPOCH + Duration::from_secs(10),
                fetched: Mutex::new(Vec::new()),
            }),
            ..Arc::try_unwrap(InMemorySource::new("b", &[])).ok().unwrap()
        });
        let scored = reranker(&[a.clone(), b.clone()])
            .rerank(
                &query(vec![1.0, 0.0]),
                &[
                    candidate("b", "0", 0),
                    candidate("a", "0", 1),
                    candidate("b", "1", 2),
                ],
                &ResourceBudget::default(),
            )
            .unwrap();
        assert_eq!(scored.scores, [Some(0.0), Some(1.0), Some(1.0)]);
        assert_eq!(a.fetched(), [vec!["0".to_string()]]);
        assert_eq!(b.fetched().concat().len(), 2);
        assert_eq!(scored.as_of, b.view.as_of);
    }

    #[test]
    fn a_corpus_without_a_source_is_an_error() {
        let source = InMemorySource::new("docs", &[("a", vec![1.0, 0.0])]);
        let error = reranker(std::slice::from_ref(&source))
            .rerank(
                &query(vec![1.0, 0.0]),
                &[candidate("other", "a", 0)],
                &ResourceBudget::default(),
            )
            .unwrap_err();
        assert!(matches!(error, Error::IncompatibleIndex(_)));
        assert!(source.fetched().is_empty());
    }

    #[test]
    fn no_candidates_needs_no_query_vectors() {
        let source = InMemorySource::new("docs", &[]);
        let scored = reranker(std::slice::from_ref(&source))
            .rerank(&Query::new("query"), &[], &ResourceBudget::default())
            .unwrap();
        assert!(scored.scores.is_empty());
        assert_eq!(scored.as_of, source.view.as_of);
    }

    #[test]
    fn sources_must_share_representation_and_semantics() {
        let a = InMemorySource::new("a", &[]);
        let other_semantics = Arc::new(InMemorySource {
            score_semantics: "lossy",
            ..Arc::try_unwrap(InMemorySource::new("b", &[])).ok().unwrap()
        });
        assert!(matches!(
            MaxSimReranker::new(
                [a.clone() as Arc<dyn MultiVectorSource>, other_semantics],
                DEFAULT_FEATURE
            ),
            Err(Error::IncompatibleIndex(_))
        ));
        let duplicate = InMemorySource::new("a", &[]);
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
