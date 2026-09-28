use thiserror::Error;

use crate::ranking::RankingError;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum Error {
    /// Stages or sources that cannot be composed: another representation,
    /// another score scale, or a corpus the reranker has no source for.
    #[error("{0}")]
    IncompatibleIndex(String),
    /// A query lacks a feature a stage needs, or supplies it from another encoder.
    #[error("{0}")]
    IncompatibleQuery(String),
    #[error("{0}")]
    InvalidInput(String),
    /// A stage or source broke the contract the pipeline enforces on its output.
    #[error(transparent)]
    Ranking(#[from] RankingError),
    /// A vector store on disk is not one this version can read.
    #[error("{0}")]
    Storage(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Raised by a stage, source, or feature provider implemented outside
    /// lateweave; bindings use it to carry their own error back unchanged.
    #[error(transparent)]
    External(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    pub(crate) fn storage(message: impl Into<String>) -> Self {
        Self::Storage(message.into())
    }
}
