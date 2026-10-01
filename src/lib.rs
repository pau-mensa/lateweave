//! Composable candidate generation and optional late-interaction reranking.
//!
//! ```text
//! Query -> CandidateGenerator -> [Reranker] -> deterministic top-k
//! ```
//!
//! A [`Query`] is raw text plus named [`Feature`]s, each stamped with the
//! [`Representation`] that produced it. Stages name documents by
//! [`DocumentKey`], the corpus and ID the system of record gives them, and
//! each reads its own indexes, which writers anywhere move independently:
//! every stage result says how fresh it is, and [`SearchPipeline`] ranks only
//! the documents every stage holds and reports the oldest commit it read.
//! [`MaxSimReranker`] scores [`MultiVectorSource`]s, one per corpus, such as
//! a [`VectorStore`], which reads what a [`VectorStoreWriter`] or any other
//! writer of its format commits.

#[cfg(target_os = "macos")]
extern crate blas_src;

mod error;
mod kernel;
mod maxsim;
mod pipeline;
mod query;
mod ranking;
mod representation;
mod source;
mod stage;
mod storage;
mod threads;

pub use error::{Error, Result};
pub use kernel::maxsim_scores;
pub use maxsim::{MaxSimReranker, DEFAULT_FEATURE};
pub use pipeline::{
    RankedDocument, SearchDiagnostics, SearchPipeline, SearchRequest, SearchResult, SearchTimings,
};
pub use query::{Feature, FeatureValue, Query, TokenMatrix};
pub use ranking::RankingError;
pub use representation::Representation;
pub use source::{MultiVectorSource, PackedDocuments, VectorView};
pub use stage::{
    Candidate, CandidateGenerator, DocumentKey, Gathered, Requirements, Reranker, ResourceBudget,
    Restriction, Scored, Subset,
};
pub use storage::{
    Encoding, StoreView, VectorStore, VectorStoreWriter, MANIFEST_FILE, STORE_FORMAT,
};
