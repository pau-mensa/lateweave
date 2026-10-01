//! The two stage contracts a pipeline composes, and the values crossing them.

use std::any::type_name;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;

use crate::error::{Error, Result};
use crate::query::Query;
use crate::representation::Representation;

/// Feature name to the representation a stage was built for.
pub type Requirements = BTreeMap<String, Representation>;

/// A document as every stage names it: the corpus it belongs to and the ID
/// the system of record gives it. Engines map keys to their own internal
/// positions privately, so they never have to agree on one.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DocumentKey {
    corpus: Arc<str>,
    id: Arc<str>,
}

impl DocumentKey {
    pub fn new(corpus: impl Into<Arc<str>>, id: impl Into<Arc<str>>) -> Self {
        Self {
            corpus: corpus.into(),
            id: id.into(),
        }
    }

    pub fn corpus(&self) -> &str {
        &self.corpus
    }

    pub fn id(&self) -> &str {
        &self.id
    }
}

impl fmt::Debug for DocumentKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}/{:?}", self.corpus, self.id)
    }
}

/// One gathered document; `provenance` names what produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub key: DocumentKey,
    pub gather_score: f32,
    pub gather_rank: usize,
    pub provenance: String,
}

/// Which documents of one corpus a search may return.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Restriction {
    /// Only these.
    Only(HashSet<Arc<str>>),
    /// Every document but these.
    Except(HashSet<Arc<str>>),
}

impl Restriction {
    pub fn allows(&self, id: &str) -> bool {
        match self {
            Restriction::Only(ids) => ids.contains(id),
            Restriction::Except(ids) => !ids.contains(id),
        }
    }
}

/// The documents a search is restricted to, by corpus. A corpus it does not
/// name contributes no documents; an ID an index does not hold is ignored.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Subset {
    corpora: BTreeMap<Arc<str>, Restriction>,
}

impl Subset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Only `ids` of `corpus`; replaces any restriction given for it.
    pub fn including<I>(self, corpus: impl Into<Arc<str>>, ids: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Arc<str>>,
    {
        self.restrict(
            corpus,
            Restriction::Only(ids.into_iter().map(Into::into).collect()),
        )
    }

    /// Every document of `corpus` but `ids`; replaces any restriction given
    /// for it.
    pub fn excluding<I>(self, corpus: impl Into<Arc<str>>, ids: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Arc<str>>,
    {
        self.restrict(
            corpus,
            Restriction::Except(ids.into_iter().map(Into::into).collect()),
        )
    }

    fn restrict(mut self, corpus: impl Into<Arc<str>>, restriction: Restriction) -> Self {
        self.corpora.insert(corpus.into(), restriction);
        self
    }

    /// `None` when the subset does not name `corpus`, which then contributes
    /// no documents.
    pub fn restriction(&self, corpus: &str) -> Option<&Restriction> {
        self.corpora.get(corpus)
    }

    pub fn contains(&self, key: &DocumentKey) -> bool {
        self.restriction(key.corpus())
            .is_some_and(|restriction| restriction.allows(key.id()))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Restriction)> {
        self.corpora
            .iter()
            .map(|(corpus, restriction)| (corpus.as_ref(), restriction))
    }
}

/// What a gatherer found, and how fresh the index it searched was: every
/// write committed to that index before `as_of` is reflected in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Gathered {
    pub candidates: Vec<Candidate>,
    pub as_of: SystemTime,
}

/// `scores[i]` scores candidate `i`, or is `None` when the reranker's index
/// does not hold that document: not written to it yet, or already deleted.
/// Every write committed to that index before `as_of` is reflected in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub scores: Vec<Option<f32>>,
    pub as_of: SystemTime,
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

/// First stage: selects candidate documents.
///
/// Returns unique candidates with gather scores and dense zero-based ranks,
/// read from one consistent state of each index it searches. `subset`, when
/// given, restricts the search to the documents it names; a gatherer that
/// cannot honour it must fail rather than ignore it.
pub trait CandidateGenerator: Send + Sync {
    /// A text-only gatherer requires nothing.
    fn requires(&self) -> &Requirements;

    /// Qualifies `gather_score`, which ranks results when no reranker follows.
    fn score_semantics(&self) -> &str;

    /// Reported in search diagnostics.
    fn name(&self) -> &str {
        short_type_name::<Self>()
    }

    fn gather(&self, query: &Query, limit: usize, subset: Option<&Subset>) -> Result<Gathered>;
}

/// Second stage: one qualified score, or `None`, for every candidate it is
/// given, in candidate order, read from one consistent state of each index it
/// scores. A candidate from a corpus the reranker cannot score at all is an
/// error, not `None`. Gather scores never influence a reranked result.
pub trait Reranker: Send + Sync {
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
    ) -> Result<Scored>;
}
