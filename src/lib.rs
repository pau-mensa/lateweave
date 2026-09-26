//! Composable candidate generation and optional late-interaction reranking.
//!
//! ```text
//! Query -> CandidateGenerator -> [Reranker] -> deterministic top-k
//! ```
//!
//! A [`Query`] is raw text plus named [`Feature`]s, each stamped with the
//! [`Representation`] that produced it. Stages declare the features they
//! consume and the [`CorpusManifest`] they index; [`SearchPipeline`] checks
//! both identities before any stage runs, enforces the stage contracts, and
//! ranks deterministically. [`MaxSimReranker`] scores any
//! [`MultiVectorSource`], including the memory-mapped [`VectorStore`].

#[cfg(target_os = "macos")]
extern crate blas_src;

mod error;
mod kernel;
mod manifest;
mod maxsim;
mod pipeline;
mod query;
mod ranking;
mod source;
mod stage;
mod storage;

pub use error::{Error, Result};
pub use kernel::maxsim_scores;
pub use manifest::{document_ids_digest, CorpusManifest, Representation};
pub use maxsim::{MaxSimReranker, DEFAULT_FEATURE};
pub use pipeline::{
    RankedDocument, SearchDiagnostics, SearchPipeline, SearchRequest, SearchResult, SearchTimings,
};
pub use query::{Feature, FeatureValue, Query, TokenMatrix};
pub use ranking::RankingError;
pub use source::{MultiVectorSource, PackedDocuments};
pub use stage::{Candidate, CandidateGenerator, Requirements, Reranker, ResourceBudget, Score};
pub use storage::{StoreFormat, VectorStore, METADATA_FILE, OFFSETS_FILE};
