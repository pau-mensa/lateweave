//! The two stage contracts a pipeline composes, and the values crossing them.

use std::any::type_name;
use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::manifest::{CorpusManifest, Representation};
use crate::query::Query;

/// Feature name to the representation a stage was built for.
pub type Requirements = BTreeMap<String, Representation>;

/// One gathered document. `document_id` is a dense internal ID; `provenance`
/// names what produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub document_id: u64,
    pub gather_score: f32,
    pub gather_rank: usize,
    pub provenance: String,
}

/// A qualified score for one document.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub document_id: u64,
    pub value: f32,
}

/// Bounded execution is caller policy; each reranker maps it onto its own
/// representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    max_batch_tokens: usize,
    max_documents_per_batch: usize,
    threads: Option<usize>,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            max_batch_tokens: 131_072,
            max_documents_per_batch: 256,
            threads: None,
        }
    }
}

impl ResourceBudget {
    pub fn new(
        max_batch_tokens: usize,
        max_documents_per_batch: usize,
        threads: Option<usize>,
    ) -> Result<Self> {
        if max_batch_tokens == 0 {
            return Err(Error::invalid("max_batch_tokens must be positive"));
        }
        if max_documents_per_batch == 0 {
            return Err(Error::invalid("max_documents_per_batch must be positive"));
        }
        if threads == Some(0) {
            return Err(Error::invalid("threads must be positive"));
        }
        Ok(Self {
            max_batch_tokens,
            max_documents_per_batch,
            threads,
        })
    }

    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }

    pub fn max_documents_per_batch(&self) -> usize {
        self.max_documents_per_batch
    }

    pub fn threads(&self) -> Option<usize> {
        self.threads
    }
}

fn short_type_name<T: ?Sized>() -> &'static str {
    let full = type_name::<T>();
    let path = full.split('<').next().unwrap_or(full);
    path.rsplit("::").next().unwrap_or(path)
}

/// First stage: selects candidate documents from the whole corpus.
///
/// Returns ordered, unique internal IDs with gather scores, dense zero-based
/// ranks, and provenance. `subset`, when given, is a strictly ascending list
/// of internal IDs the search is restricted to; a gatherer that cannot honour
/// it must fail rather than ignore it.
pub trait CandidateGenerator: Send + Sync {
    fn corpus(&self) -> &CorpusManifest;

    /// A text-only gatherer requires nothing.
    fn requires(&self) -> &Requirements;

    /// Qualifies `gather_score`, which ranks results when no reranker follows.
    fn score_semantics(&self) -> &str;

    /// Reported in search diagnostics.
    fn name(&self) -> &str {
        short_type_name::<Self>()
    }

    fn gather(&self, query: &Query, limit: usize, subset: Option<&[u64]>)
        -> Result<Vec<Candidate>>;
}

/// Second stage: exactly one qualified score for every candidate it is given.
/// Gather scores never influence a reranked result.
pub trait Reranker: Send + Sync {
    fn corpus(&self) -> &CorpusManifest;

    fn requires(&self) -> &Requirements;

    fn score_semantics(&self) -> &str;

    /// Reported in search diagnostics.
    fn name(&self) -> &str {
        short_type_name::<Self>()
    }

    fn rerank(
        &self,
        query: &Query,
        candidates: &[Candidate],
        budget: &ResourceBudget,
    ) -> Result<Vec<Score>>;
}
