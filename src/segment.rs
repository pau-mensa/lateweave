//! Segments: immutable snapshots of one corpus's document set.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, OnceLock};

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::manifest::{digest_document_id, digest_hex, CorpusManifest};

/// One corpus at one generation: its [`CorpusManifest`] and the external IDs
/// its internal IDs name.
///
/// Internal IDs are dense `0..n-1` and local to the segment: a document is
/// `(segment, document_id)`, so corpora indexed separately never share an ID
/// space. A segment never changes; [`Segment::appended`] and
/// [`Segment::deleted`] return the next generation, compacted exactly as a
/// [`VectorStore`](crate::VectorStore) mutation leaves it. Cloning is cheap.
#[derive(Clone)]
pub struct Segment {
    inner: Arc<Inner>,
}

struct Inner {
    manifest: CorpusManifest,
    external: Vec<Arc<str>>,
    /// Built on the first lookup by external ID.
    internal: OnceLock<HashMap<Arc<str>, u64>>,
}

impl Segment {
    /// Fails if an external ID appears more than once.
    pub fn new<I>(
        corpus_id: impl Into<String>,
        corpus_version: impl Into<String>,
        generation: u64,
        document_ids: I,
    ) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: Into<Arc<str>>,
    {
        let external = document_ids.into_iter().map(Into::into).collect::<Vec<_>>();
        let mut seen = HashSet::with_capacity(external.len());
        let mut digest = Sha256::new();
        for document_id in &external {
            if !seen.insert(document_id.clone()) {
                return Err(Error::invalid(format!(
                    "document ID {document_id:?} appears more than once"
                )));
            }
            digest_document_id(&mut digest, document_id);
        }
        let manifest = CorpusManifest::new(
            corpus_id,
            corpus_version,
            external.len() as u64,
            digest_hex(digest),
        )?
        .with_generation(generation);
        Ok(Self {
            inner: Arc::new(Inner {
                manifest,
                external,
                internal: OnceLock::new(),
            }),
        })
    }

    /// The segment `manifest` describes, which must have been computed over
    /// exactly `document_ids` in this order.
    pub fn from_manifest<I>(manifest: &CorpusManifest, document_ids: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: Into<Arc<str>>,
    {
        let segment = Self::new(
            manifest.corpus_id(),
            manifest.corpus_version(),
            manifest.generation(),
            document_ids,
        )?;
        segment.manifest().assert_compatible(manifest)?;
        Ok(segment)
    }

    pub fn manifest(&self) -> &CorpusManifest {
        &self.inner.manifest
    }

    pub fn corpus_id(&self) -> &str {
        self.inner.manifest.corpus_id()
    }

    pub fn generation(&self) -> u64 {
        self.inner.manifest.generation()
    }

    pub fn document_count(&self) -> u64 {
        self.inner.external.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.inner.external.is_empty()
    }

    /// External IDs in internal-ID order.
    pub fn document_ids(&self) -> impl ExactSizeIterator<Item = &str> {
        self.inner.external.iter().map(AsRef::as_ref)
    }

    pub fn contains(&self, document_id: u64) -> bool {
        document_id < self.document_count()
    }

    pub fn internal(&self, external: &str) -> Option<u64> {
        self.lookup().get(external).copied()
    }

    pub fn external(&self, internal: u64) -> Option<&str> {
        usize::try_from(internal)
            .ok()
            .and_then(|index| self.inner.external.get(index))
            .map(AsRef::as_ref)
    }

    /// Internal IDs of `external_ids`, in the same order; fails on an unknown ID.
    pub fn to_internal<I>(&self, external_ids: I) -> Result<Vec<u64>>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        external_ids
            .into_iter()
            .map(|external| {
                let external = external.as_ref();
                self.internal(external)
                    .ok_or_else(|| self.unknown(external))
            })
            .collect()
    }

    /// External IDs of `internal_ids`, in the same order; fails outside the
    /// segment.
    pub fn to_external(&self, internal_ids: &[u64]) -> Result<Vec<&str>> {
        internal_ids
            .iter()
            .map(|&internal| {
                self.external(internal)
                    .ok_or_else(|| self.outside(internal))
            })
            .collect()
    }

    /// The same snapshot: shared, or built independently over the same
    /// documents at the same generation.
    pub fn is(&self, other: &Segment) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || self.manifest() == other.manifest()
    }

    /// Fails with [`Error::IncompatibleIndex`] naming every differing field
    /// unless `other` is the same snapshot.
    pub fn assert_compatible(&self, other: &Segment) -> Result<()> {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return Ok(());
        }
        self.manifest().assert_compatible(other.manifest())
    }

    /// The next generation, with `document_ids` taking the next internal IDs;
    /// this segment itself when there are none. Fails if one is already in
    /// the segment or repeats.
    pub fn appended<I>(&self, document_ids: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: Into<Arc<str>>,
    {
        let mut added = document_ids.into_iter().map(Into::into).peekable();
        if added.peek().is_none() {
            return Ok(self.clone());
        }
        self.successor(self.inner.external.iter().cloned().chain(added))
    }

    /// The next generation without `document_ids`, compacted to `0..n-1` with
    /// the survivors in their existing order; this segment itself when there
    /// are none. Fails on an unknown or repeated ID.
    pub fn deleted<I>(&self, document_ids: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let deleted = self.deletion(document_ids)?;
        if deleted.is_empty() {
            return Ok(self.clone());
        }
        self.successor(
            self.inner
                .external
                .iter()
                .enumerate()
                .filter(|(internal, _)| !deleted.contains(&(*internal as u64)))
                .map(|(_, external)| external.clone()),
        )
    }

    /// Internal IDs of `document_ids` for a deletion: known and unique.
    pub(crate) fn deletion<I>(&self, document_ids: I) -> Result<HashSet<u64>>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut deleted = HashSet::new();
        for external in document_ids {
            let external = external.as_ref();
            let internal = self
                .internal(external)
                .ok_or_else(|| self.unknown(external))?;
            if !deleted.insert(internal) {
                return Err(Error::invalid(format!(
                    "document ID {external:?} is deleted more than once"
                )));
            }
        }
        Ok(deleted)
    }

    fn successor(&self, document_ids: impl Iterator<Item = Arc<str>>) -> Result<Self> {
        let manifest = self.manifest();
        Self::new(
            manifest.corpus_id(),
            manifest.corpus_version(),
            manifest.generation() + 1,
            document_ids,
        )
    }

    fn lookup(&self) -> &HashMap<Arc<str>, u64> {
        self.inner.internal.get_or_init(|| {
            self.inner
                .external
                .iter()
                .enumerate()
                .map(|(internal, external)| (external.clone(), internal as u64))
                .collect()
        })
    }

    fn unknown(&self, external: &str) -> Error {
        Error::invalid(format!(
            "document ID {external:?} is not in segment {:?}",
            self.corpus_id()
        ))
    }

    fn outside(&self, internal: u64) -> Error {
        Error::invalid(format!(
            "document ID {internal} is outside segment {:?} of {} documents",
            self.corpus_id(),
            self.document_count()
        ))
    }
}

impl PartialEq for Segment {
    fn eq(&self, other: &Self) -> bool {
        self.is(other)
    }
}

impl Eq for Segment {}

impl fmt::Debug for Segment {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let manifest = self.manifest();
        formatter
            .debug_struct("Segment")
            .field("corpus_id", &manifest.corpus_id())
            .field("corpus_version", &manifest.corpus_version())
            .field("generation", &manifest.generation())
            .field("document_count", &manifest.document_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document_ids_digest;

    fn segment(ids: &[&str]) -> Segment {
        Segment::new("docs", "v1", 0, ids.iter().copied()).unwrap()
    }

    #[test]
    fn internal_ids_follow_input_order() {
        let segment = segment(&["b", "a", "c"]);
        assert_eq!(segment.document_count(), 3);
        assert_eq!(segment.internal("a"), Some(1));
        assert_eq!(segment.internal("z"), None);
        assert_eq!(segment.external(2), Some("c"));
        assert_eq!(segment.external(3), None);
        assert!(segment.contains(2) && !segment.contains(3));
        assert_eq!(segment.document_ids().collect::<Vec<_>>(), ["b", "a", "c"]);
    }

    #[test]
    fn the_manifest_digests_the_external_ids() {
        let manifest = segment(&["b", "a", "c"]).manifest().clone();
        assert_eq!(manifest.document_count(), 3);
        assert_eq!(
            manifest.document_ids_sha256(),
            document_ids_digest(["b", "a", "c"])
        );
        assert!(Segment::from_manifest(&manifest, ["b", "a", "c"]).is_ok());
        assert!(matches!(
            Segment::from_manifest(&manifest, ["a", "b", "c"]),
            Err(Error::IncompatibleIndex(_))
        ));
    }

    #[test]
    fn duplicates_are_refused() {
        let error = Segment::new("docs", "v1", 0, ["a", "b", "a"]).unwrap_err();
        assert!(error.to_string().contains("\"a\" appears more than once"));
        assert!(segment(&["a"]).appended(["a"]).is_err());
    }

    #[test]
    fn translates_in_both_directions() {
        let segment = segment(&["b", "a", "c"]);
        assert_eq!(segment.to_internal(["c", "b"]).unwrap(), [2, 0]);
        assert!(segment.to_internal(["c", "z"]).is_err());
        assert_eq!(segment.to_external(&[1, 2]).unwrap(), ["a", "c"]);
        assert!(segment.to_external(&[3]).is_err());
    }

    #[test]
    fn snapshots_are_the_same_only_at_the_same_generation() {
        let first = segment(&["a", "b"]);
        assert!(first.is(&first.clone()));
        assert!(first.is(&segment(&["a", "b"])));
        assert!(!first.is(&segment(&["b", "a"])));
        assert!(Arc::ptr_eq(
            &first.appended(Vec::<String>::new()).unwrap().inner,
            &first.deleted(Vec::<String>::new()).unwrap().inner
        ));
        let next = first.appended(["c"]).unwrap();
        assert_eq!(next.generation(), 1);
        assert!(matches!(
            first.assert_compatible(&next),
            Err(Error::IncompatibleIndex(_))
        ));
    }

    #[test]
    fn appended_takes_the_next_ids() {
        let next = segment(&["b", "a"]).appended(["c", "d"]).unwrap();
        assert_eq!(next.generation(), 1);
        assert_eq!(
            next.document_ids().collect::<Vec<_>>(),
            ["b", "a", "c", "d"]
        );
        assert_eq!(next.internal("d"), Some(3));
    }

    #[test]
    fn deleted_compacts_preserving_order() {
        let next = segment(&["a", "b", "c", "d", "e"])
            .deleted(["d", "b"])
            .unwrap();
        assert_eq!(next.generation(), 1);
        assert_eq!(next.document_ids().collect::<Vec<_>>(), ["a", "c", "e"]);
        assert_eq!(next.internal("e"), Some(2));
        assert_eq!(next.internal("b"), None);
    }

    #[test]
    fn deleted_refuses_unknown_and_repeated_ids() {
        let segment = segment(&["a", "b"]);
        assert!(segment.deleted(["z"]).is_err());
        assert!(segment.deleted(["b", "b"]).is_err());
    }
}
