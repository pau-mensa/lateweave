//! The two stage contracts a pipeline composes, and the values crossing them.

use std::any::type_name;
use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::manifest::Representation;
use crate::query::Query;
use crate::segment::Segment;

/// Feature name to the representation a stage was built for.
pub type Requirements = BTreeMap<String, Representation>;

/// One gathered document: `document_id` is an internal ID of `segment`;
/// `provenance` names what produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub segment: Segment,
    pub document_id: u64,
    pub gather_score: f32,
    pub gather_rank: usize,
    pub provenance: String,
}

impl Candidate {
    /// `None` when `document_id` is outside the segment.
    pub fn external_id(&self) -> Option<&str> {
        self.segment.external(self.document_id)
    }
}

/// The documents a search is restricted to: internal IDs per segment, keyed
/// by corpus ID. A segment it does not name contributes no documents.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Subset {
    ids: BTreeMap<String, Vec<u64>>,
}

impl Subset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces any IDs already given for `corpus_id`.
    pub fn with(mut self, corpus_id: impl Into<String>, document_ids: impl Into<Vec<u64>>) -> Self {
        self.ids.insert(corpus_id.into(), document_ids.into());
        self
    }

    /// The segment's IDs; empty when the subset does not name it.
    pub fn ids(&self, corpus_id: &str) -> &[u64] {
        self.ids.get(corpus_id).map_or(&[], Vec::as_slice)
    }

    /// Named segments and their IDs, by corpus ID.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[u64])> {
        self.ids
            .iter()
            .map(|(corpus_id, ids)| (corpus_id.as_str(), ids.as_slice()))
    }
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

/// First stage: selects candidate documents from its segments.
///
/// Returns unique `(segment, document_id)` candidates from the segments it
/// declares, with gather scores, dense zero-based ranks, and provenance.
/// `subset`, when given, names strictly ascending internal IDs per segment
/// and restricts the search to them; a gatherer that cannot honour it must
/// fail rather than ignore it.
pub trait CandidateGenerator: Send + Sync {
    /// The snapshots this gatherer searches, each under a distinct corpus ID.
    fn segments(&self) -> &[Segment];

    /// A text-only gatherer requires nothing.
    fn requires(&self) -> &Requirements;

    /// Qualifies `gather_score`, which ranks results when no reranker follows.
    fn score_semantics(&self) -> &str;

    /// Reported in search diagnostics.
    fn name(&self) -> &str {
        short_type_name::<Self>()
    }

    fn gather(
        &self,
        query: &Query,
        limit: usize,
        subset: Option<&Subset>,
    ) -> Result<Vec<Candidate>>;
}

/// Second stage: exactly one qualified score for every candidate it is given,
/// in candidate order. Gather scores never influence a reranked result.
pub trait Reranker: Send + Sync {
    /// The snapshots this reranker can score; they must cover every segment
    /// the gatherer searches.
    fn segments(&self) -> &[Segment];

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
    ) -> Result<Vec<f32>>;
}
