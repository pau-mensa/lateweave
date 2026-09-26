//! MaxSim reranking over any multi-vector source.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::kernel::maxsim_scores;
use crate::manifest::CorpusManifest;
use crate::query::Query;
use crate::source::MultiVectorSource;
use crate::stage::{Candidate, Requirements, Reranker, ResourceBudget, Score};

/// The feature name a [`MaxSimReranker`] reads unless told otherwise.
pub const DEFAULT_FEATURE: &str = "multi_vector";

/// MaxSim between the query's token matrix and a source's documents.
///
/// The reranker requires its query feature to carry the source's
/// representation, so vectors from a different encoder are refused before
/// anything is fetched.
pub struct MaxSimReranker {
    source: Arc<dyn MultiVectorSource>,
    corpus: CorpusManifest,
    feature: String,
    requires: Requirements,
}

impl MaxSimReranker {
    pub fn new(
        source: Arc<dyn MultiVectorSource>,
        corpus: CorpusManifest,
        feature: impl Into<String>,
    ) -> Result<Self> {
        if source.document_count() != corpus.document_count() {
            return Err(Error::IncompatibleIndex(format!(
                "source and corpus manifest document counts differ ({} != {})",
                source.document_count(),
                corpus.document_count()
            )));
        }
        let feature = feature.into();
        let requires = BTreeMap::from([(feature.clone(), source.representation().clone())]);
        Ok(Self {
            source,
            corpus,
            feature,
            requires,
        })
    }

    pub fn source(&self) -> &Arc<dyn MultiVectorSource> {
        &self.source
    }

    pub fn feature(&self) -> &str {
        &self.feature
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

impl Reranker for MaxSimReranker {
    fn corpus(&self) -> &CorpusManifest {
        &self.corpus
    }

    fn requires(&self) -> &Requirements {
        &self.requires
    }

    fn score_semantics(&self) -> &str {
        self.source.score_semantics()
    }

    fn rerank(
        &self,
        query: &Query,
        candidates: &[Candidate],
        budget: &ResourceBudget,
    ) -> Result<Vec<Score>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let representation = self.source.representation();
        let vectors = query
            .feature(&self.feature, representation)?
            .token_matrix()
            .ok_or_else(|| {
                Error::IncompatibleQuery(format!(
                    "query feature '{}' must be a [tokens, dimension] matrix",
                    self.feature
                ))
            })?;
        if vectors.dimension() != representation.dimension() {
            return Err(Error::IncompatibleQuery(format!(
                "query feature '{}' has dimension {}, but the representation declares {}",
                self.feature,
                vectors.dimension(),
                representation.dimension()
            )));
        }

        let candidate_ids = candidates
            .iter()
            .map(|candidate| candidate.document_id)
            .collect::<Vec<_>>();
        let lengths = self.source.document_lengths(&candidate_ids)?;
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
                let documents = self.source.fetch(&batch, budget.threads())?;
                let expected = batch.iter().map(|document_id| lengths[document_id]);
                if documents.dimension() != representation.dimension()
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
        Ok(candidate_ids
            .into_iter()
            .map(|document_id| Score {
                document_id,
                value: scores[&document_id],
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::manifest::Representation;
    use crate::query::{Feature, TokenMatrix};
    use crate::source::PackedDocuments;

    struct InMemorySource {
        representation: Representation,
        documents: Vec<Vec<f32>>,
        fetched: Mutex<Vec<Vec<u64>>>,
    }

    impl MultiVectorSource for InMemorySource {
        fn representation(&self) -> &Representation {
            &self.representation
        }

        fn score_semantics(&self) -> &str {
            "in-memory-exact-full-maxsim"
        }

        fn document_count(&self) -> u64 {
            self.documents.len() as u64
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

    fn candidate(document_id: u64, gather_rank: usize) -> Candidate {
        Candidate {
            document_id,
            gather_score: 0.0,
            gather_rank,
            provenance: "test".to_string(),
        }
    }

    #[test]
    fn scores_a_borrowed_source_in_candidate_order_within_the_token_budget() {
        let representation = Representation::new("encoder", "1", 2, true).unwrap();
        let source = Arc::new(InMemorySource {
            representation: representation.clone(),
            documents: vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.6, 0.8], vec![-1.0, 0.0]],
            fetched: Mutex::new(Vec::new()),
        });
        let reranker = MaxSimReranker::new(
            source.clone(),
            CorpusManifest::new("corpus", "1", 3, "abc").unwrap(),
            DEFAULT_FEATURE,
        )
        .unwrap();
        let query = Query::new("query").with_feature(
            DEFAULT_FEATURE,
            Feature::new(
                representation,
                TokenMatrix::new(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap(),
            ),
        );

        let scores = reranker
            .rerank(
                &query,
                &[candidate(1, 0), candidate(0, 1), candidate(2, 2)],
                &ResourceBudget::new(2, 256, Some(1)).unwrap(),
            )
            .unwrap();

        let ids = scores
            .iter()
            .map(|score| score.document_id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 0, 2]);
        for (score, expected) in scores.iter().zip([1.4, 2.0, -1.0]) {
            assert!((score.value - expected).abs() < 1e-6);
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
}
