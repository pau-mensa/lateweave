//! Document token vectors, fetched by internal ID.

use crate::error::{Error, Result};
use crate::manifest::Representation;

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

/// Where a MaxSim reranker reads document vectors from.
///
/// An engine that already holds document vectors implements this over them
/// and stores nothing twice; the lateweave vector stores implement it for
/// gatherers that keep no vectors. `score_semantics` qualifies what MaxSim
/// over the fetched vectors means, since a source may reconstruct from a lossy
/// code.
pub trait MultiVectorSource: Send + Sync {
    fn representation(&self) -> &Representation;

    fn score_semantics(&self) -> &str;

    fn document_count(&self) -> u64;

    /// Changes whenever a mutation may have changed which vectors an ID
    /// holds. A reranker built over the source refuses to score once it has
    /// moved; a source that never mutates keeps the default.
    fn generation(&self) -> u64 {
        0
    }

    /// Token counts of `document_ids`, in the same order.
    fn document_lengths(&self, document_ids: &[u64]) -> Result<Vec<usize>>;

    /// The documents' vectors in the requested order.
    fn fetch(&self, document_ids: &[u64], threads: Option<usize>) -> Result<PackedDocuments>;
}
