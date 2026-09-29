//! Document token vectors, fetched by document ID.

use std::sync::Arc;
use std::time::SystemTime;

use crate::error::{Error, Result};
use crate::representation::Representation;

/// Token vectors of documents: a row-major float32 `[sum(lengths), dimension]`
/// matrix holding each document's tokens contiguously, in `lengths` order.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedDocuments {
    vectors: Vec<f32>,
    lengths: Vec<usize>,
    dimension: usize,
}

impl PackedDocuments {
    pub fn new(vectors: Vec<f32>, lengths: Vec<usize>, dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::invalid("packed documents need a positive dimension"));
        }
        if lengths.contains(&0) {
            return Err(Error::invalid("document lengths must be positive"));
        }
        let tokens = lengths
            .iter()
            .try_fold(0usize, |total, &length| total.checked_add(length))
            .ok_or_else(|| Error::invalid("document lengths overflow usize"))?;
        if tokens.checked_mul(dimension) != Some(vectors.len()) {
            return Err(Error::invalid(format!(
                "document lengths sum to {tokens}, but {} values form {} rows of dimension {dimension}",
                vectors.len(),
                vectors.len() / dimension
            )));
        }
        Ok(Self {
            vectors,
            lengths,
            dimension,
        })
    }

    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Where a MaxSim reranker reads one corpus's document vectors from.
///
/// An engine that already holds document vectors implements this over them
/// and stores nothing twice; a [`VectorStore`](crate::VectorStore)
/// implements it for gatherers that keep no vectors. `score_semantics`
/// qualifies what MaxSim over the fetched vectors means, since a source may
/// reconstruct from a lossy code.
pub trait MultiVectorSource: Send + Sync {
    fn corpus(&self) -> &str;

    fn representation(&self) -> &Representation;

    fn score_semantics(&self) -> &str;

    /// The source's current state, which a rerank reads throughout.
    fn view(&self) -> Result<Arc<dyn VectorView>>;
}

/// One consistent state of a [`MultiVectorSource`]: the vectors a document
/// ID names never change for the life of the view.
pub trait VectorView: Send + Sync {
    /// Every write committed to the source before this is visible in the view.
    fn as_of(&self) -> SystemTime;

    /// Token counts of `document_ids`, in the same order; `None` for a
    /// document the view does not hold.
    fn document_lengths(&self, document_ids: &[&str]) -> Result<Vec<Option<usize>>>;

    /// The vectors of `document_ids`, which the view must hold, in order.
    fn fetch(&self, document_ids: &[&str], threads: Option<usize>) -> Result<PackedDocuments>;
}
