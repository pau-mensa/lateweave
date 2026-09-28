//! Composable candidate generation and optional late-interaction reranking.
//!
//! ```text
//! Query -> CandidateGenerator -> [Reranker] -> deterministic top-k
//! ```
//!
//! A [`Query`] is raw text plus named [`Feature`]s, each stamped with the
//! [`Representation`] that produced it. A document is an internal ID of a
//! [`Segment`], an immutable snapshot of one corpus; a gatherer searches one
//! or more segments and a reranker must hold the same snapshot of each.
//! [`SearchPipeline`] checks both identities before any stage runs, enforces
//! the stage contracts, and ranks deterministically. [`MaxSimReranker`]
//! scores [`MultiVectorSource`]s, one per segment, including snapshots of the
//! memory-mapped [`VectorStore`].

#[cfg(target_os = "macos")]
extern crate blas_src;

mod error;
mod kernel;
mod manifest;
mod maxsim;
mod pipeline;
mod query;
mod ranking;
mod segment;
mod source;
mod stage;
mod storage;
mod threads;

pub use error::{Error, Result};
pub use kernel::maxsim_scores;
pub use manifest::{document_ids_digest, CorpusManifest, Representation};
pub use maxsim::{MaxSimReranker, DEFAULT_FEATURE};
pub use pipeline::{
    RankedDocument, SearchDiagnostics, SearchPipeline, SearchRequest, SearchResult, SearchTimings,
};
pub use query::{Feature, FeatureValue, Query, TokenMatrix};
pub use ranking::RankingError;
pub use segment::Segment;
pub use source::{MultiVectorSource, PackedDocuments};
pub use stage::{Candidate, CandidateGenerator, Requirements, Reranker, ResourceBudget, Subset};
pub use storage::{StoreFormat, StoreSnapshot, VectorStore, METADATA_FILE};
