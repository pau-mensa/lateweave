//! Bounded rayon execution shared by the kernel, the stores, and the reranker.

use rayon::ThreadPoolBuilder;

use crate::error::{Error, Result};

/// Runs `execute` on a pool of `threads` workers, or on the current pool when
/// `threads` is `None`.
///
/// A call made from inside a pool that already has `threads` workers runs
/// there, so an outer stage can build one pool and the calls it makes reuse it.
pub(crate) fn install<T: Send>(
    threads: Option<usize>,
    execute: impl FnOnce() -> T + Send,
) -> Result<T> {
    match threads {
        Some(0) => Err(Error::invalid("threads must be positive")),
        Some(threads)
            if rayon::current_thread_index().is_some()
                && rayon::current_num_threads() == threads =>
        {
            Ok(execute())
        }
        Some(threads) => ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| {
                Error::invalid(format!(
                    "could not create a pool of {threads} worker threads: {error}"
                ))
            })
            .map(|pool| pool.install(execute)),
        None => Ok(execute()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nested_call_of_the_same_size_reuses_the_pool() {
        let (outer, inner) = install(Some(3), || {
            let outer = rayon::current_thread_index().is_some();
            let inner = install(Some(3), rayon::current_num_threads).unwrap();
            (outer, inner)
        })
        .unwrap();
        assert!(outer);
        assert_eq!(inner, 3);
        assert_eq!(install(Some(2), rayon::current_num_threads).unwrap(), 2);
        assert!(install(Some(0), || ()).is_err());
    }
}
