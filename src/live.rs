//! Stages and sources whose snapshots move.

use std::sync::Arc;

use crate::error::Result;

/// Something that hands out immutable snapshots of `S`: a gatherer, reranker,
/// or source over state that writers, in this process or another, move to
/// new generations.
///
/// A [`SearchPipeline`](crate::SearchPipeline) asks for the current snapshot
/// at the start of every search, so a publish is served on the next search
/// and a running search finishes on what it started with. `current` must
/// return a snapshot whose segment and engine state were read together, and
/// should be cheap when nothing moved.
pub trait Live<S: ?Sized>: Send + Sync {
    fn current(&self) -> Result<Arc<S>>;
}

/// A snapshot that never moves.
pub struct Fixed<S: ?Sized>(pub Arc<S>);

impl<S: ?Sized + Send + Sync> Live<S> for Fixed<S> {
    fn current(&self) -> Result<Arc<S>> {
        Ok(self.0.clone())
    }
}
