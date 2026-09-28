use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use lateweave::Segment;
use numpy::PyArray1;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyType};

use crate::convert::{corpus_manifest, corpus_manifest_to_py, to_py};

/// External document IDs from any iterable of strings. A bare string is
/// refused rather than read as one ID per character.
pub(crate) fn document_ids(document_ids: &Bound<'_, PyAny>) -> PyResult<Vec<Arc<str>>> {
    if document_ids.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "expected an iterable of document IDs, not a single string",
        ));
    }
    document_ids
        .try_iter()?
        .map(|document_id| Ok(document_id?.extract::<String>()?.into()))
        .collect()
}

/// One corpus at one generation: its manifest and the external IDs its
/// internal IDs name.
///
/// Internal IDs are dense ``0..n-1`` and local to the segment. A segment never
/// changes; ``appended`` and ``deleted`` return the next generation, compacted
/// exactly as a vector store mutation leaves it.
#[pyclass(name = "Segment", frozen, eq, module = "lateweave._native")]
#[derive(PartialEq)]
pub(crate) struct PySegment {
    pub(crate) inner: Segment,
}

impl From<Segment> for PySegment {
    fn from(inner: Segment) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySegment {
    #[new]
    #[pyo3(signature = (corpus_id, corpus_version, document_ids, *, generation=0))]
    fn new(
        corpus_id: String,
        corpus_version: String,
        document_ids: &Bound<'_, PyAny>,
        generation: u64,
    ) -> PyResult<Self> {
        let document_ids = self::document_ids(document_ids)?;
        Ok(
            Segment::new(corpus_id, corpus_version, generation, document_ids)
                .map_err(to_py)?
                .into(),
        )
    }

    /// The segment ``manifest`` describes; raises ``IncompatibleIndexError``
    /// unless it was computed over exactly ``document_ids`` in this order.
    #[classmethod]
    fn from_manifest(
        _class: &Bound<'_, PyType>,
        manifest: &Bound<'_, PyAny>,
        document_ids: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let manifest = corpus_manifest(manifest)?;
        let document_ids = self::document_ids(document_ids)?;
        Ok(Segment::from_manifest(&manifest, document_ids)
            .map_err(to_py)?
            .into())
    }

    #[getter]
    fn manifest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        corpus_manifest_to_py(py, self.inner.manifest())
    }

    #[getter]
    fn corpus_id(&self) -> &str {
        self.inner.corpus_id()
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    #[getter]
    fn document_ids(&self) -> Vec<&str> {
        self.inner.document_ids().collect()
    }

    /// Consistent with ``==``: equal segments share a manifest.
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.manifest().hash(&mut hasher);
        hasher.finish()
    }

    fn __len__(&self) -> usize {
        self.inner.document_count() as usize
    }

    fn __contains__(&self, external: &str) -> bool {
        self.inner.internal(external).is_some()
    }

    fn __repr__(&self) -> String {
        let manifest = self.inner.manifest();
        format!(
            "Segment({:?}, {:?}, <{} documents>, generation={})",
            manifest.corpus_id(),
            manifest.corpus_version(),
            manifest.document_count(),
            manifest.generation()
        )
    }

    /// Raises ``KeyError`` for an unknown ID.
    fn internal(&self, external: &str) -> PyResult<u64> {
        self.inner
            .internal(external)
            .ok_or_else(|| PyKeyError::new_err(external.to_string()))
    }

    /// Raises ``IndexError`` outside the segment.
    fn external(&self, internal: u64) -> PyResult<&str> {
        self.inner.external(internal).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "document ID {internal} is outside segment {:?} of {} documents",
                self.inner.corpus_id(),
                self.inner.document_count()
            ))
        })
    }

    /// Internal IDs as an int64 array, the form a ``subset`` takes.
    fn to_internal<'py>(
        &self,
        py: Python<'py>,
        external_ids: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let external_ids = document_ids(external_ids)?;
        let internal = self.inner.to_internal(&external_ids).map_err(to_py)?;
        Ok(PyArray1::from_iter(
            py,
            internal.into_iter().map(|id| id as i64),
        ))
    }

    fn to_external(&self, internal_ids: Vec<u64>) -> PyResult<Vec<&str>> {
        self.inner.to_external(&internal_ids).map_err(to_py)
    }

    fn appended(&self, document_ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let document_ids = self::document_ids(document_ids)?;
        Ok(self.inner.appended(document_ids).map_err(to_py)?.into())
    }

    fn deleted(&self, document_ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let document_ids = self::document_ids(document_ids)?;
        Ok(self.inner.deleted(&document_ids).map_err(to_py)?.into())
    }
}
