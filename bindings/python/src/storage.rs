use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::SystemTime;

use lateweave::{Encoding, StoreView, VectorStore, VectorStoreWriter};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::{
    document_ids, representation, representation_to_py, row_major_borrowed, to_py,
};
use crate::stages::lengths;

fn encoding(name: &str) -> PyResult<Encoding> {
    Encoding::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown encoding {name:?}; expected \"float32\" or \"int8\""
        ))
    })
}

/// Reads a lateweave vector store, following what its writer commits.
///
/// ``view()`` rereads ``manifest.json`` and loads only what it has not seen;
/// passed to ``MaxSimReranker`` the store hands every rerank its latest view.
#[pyclass(name = "VectorStore", frozen, module = "lateweave._native")]
pub(crate) struct PyVectorStore {
    pub(crate) inner: Arc<VectorStore>,
}

#[pymethods]
impl PyVectorStore {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| VectorStore::open(&path)).map_err(to_py)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path().to_path_buf()
    }

    #[getter]
    fn corpus(&self) -> &str {
        self.inner.corpus()
    }

    #[getter]
    fn encoding(&self) -> &'static str {
        self.inner.encoding().name()
    }

    #[getter]
    fn score_semantics(&self) -> &'static str {
        self.inner.encoding().score_semantics()
    }

    #[getter]
    fn representation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        representation_to_py(py, self.inner.representation())
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn view(&self, py: Python<'_>) -> PyResult<PyStoreView> {
        let inner = py.detach(|| self.inner.view()).map_err(to_py)?;
        Ok(PyStoreView { inner })
    }
}

/// One commit of a vector store. It stays readable after later commits.
#[pyclass(name = "StoreView", frozen, module = "lateweave._native")]
pub(crate) struct PyStoreView {
    inner: Arc<StoreView>,
}

#[pymethods]
impl PyStoreView {
    /// When the writer committed this view.
    #[getter]
    fn as_of(&self) -> SystemTime {
        self.inner.as_of()
    }

    #[getter]
    fn commit(&self) -> u64 {
        self.inner.commit()
    }

    /// The IDs of every document present, sorted.
    #[getter]
    fn document_ids(&self) -> Vec<&str> {
        self.inner.document_ids()
    }

    fn __contains__(&self, document_id: &str) -> bool {
        self.inner.contains(document_id)
    }

    /// Token counts of the documents present among ``document_ids``.
    fn document_lengths<'py>(
        &self,
        py: Python<'py>,
        document_ids: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = self::document_ids(document_ids)?;
        let ids = ids.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
        let output = PyDict::new(py);
        for (id, length) in ids.iter().zip(self.inner.document_lengths(&ids)) {
            if let Some(length) = length {
                output.set_item(id, length)?;
            }
        }
        Ok(output)
    }

    /// Packed float32 ``[tokens, dimension]`` vectors of ``document_ids``,
    /// which must be present, in order, plus their int64 lengths.
    #[pyo3(signature = (document_ids, *, threads=None))]
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        document_ids: &Bound<'py, PyAny>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let ids = self::document_ids(document_ids)?;
        let packed = py
            .detach(|| {
                let ids = ids.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
                self.inner.fetch(&ids, threads)
            })
            .map_err(to_py)?;
        let rows = packed.vectors().len() / packed.dimension();
        let vectors =
            PyArray1::from_slice(py, packed.vectors()).reshape([rows, packed.dimension()])?;
        let lengths = PyArray1::from_iter(py, packed.lengths().iter().map(|&length| length as i64));
        PyTuple::new(py, [vectors.into_any(), lengths.into_any()])
    }

    fn __repr__(&self) -> String {
        format!(
            "StoreView(corpus={:?}, commit={}, encoding={:?})",
            self.inner.corpus(),
            self.inner.commit(),
            self.inner.encoding().name()
        )
    }
}

/// Stages appends, deletes, and compactions of one vector store, and
/// publishes them together on ``commit()``.
///
/// Appending a document that is already present replaces it; deleting one
/// that is not present does nothing. A commit with nothing staged still
/// advances the time readers report as fresh.
#[pyclass(name = "VectorStoreWriter", frozen, module = "lateweave._native")]
pub(crate) struct PyVectorStoreWriter {
    inner: Mutex<VectorStoreWriter>,
}

impl PyVectorStoreWriter {
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        operation: impl FnOnce(&mut VectorStoreWriter) -> lateweave::Result<T> + Send,
    ) -> PyResult<T> {
        py.detach(|| operation(&mut self.inner.lock().unwrap_or_else(PoisonError::into_inner)))
            .map_err(to_py)
    }
}

#[pymethods]
impl PyVectorStoreWriter {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| VectorStoreWriter::open(&path))
            .map_err(to_py)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Writes an empty store at ``path``, which must not exist.
    #[staticmethod]
    #[pyo3(signature = (path, corpus, representation, *, encoding="float32"))]
    fn create(
        py: Python<'_>,
        path: PathBuf,
        corpus: String,
        representation: &Bound<'_, PyAny>,
        encoding: &str,
    ) -> PyResult<Self> {
        let encoding = self::encoding(encoding)?;
        let representation = self::representation(representation)?;
        let inner = py
            .detach(|| VectorStoreWriter::create(&path, encoding, corpus, representation))
            .map_err(to_py)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[getter]
    fn corpus(&self) -> String {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .corpus()
            .to_string()
    }

    /// Stages documents; ``embeddings`` is a float32 ``[tokens, dimension]``
    /// matrix packed by document in the order of ``document_ids``.
    #[pyo3(signature = (document_ids, embeddings, document_lengths, *, threads=None))]
    fn append(
        &self,
        py: Python<'_>,
        document_ids: &Bound<'_, PyAny>,
        embeddings: PyReadonlyArray2<'_, f32>,
        document_lengths: &Bound<'_, PyAny>,
        threads: Option<usize>,
    ) -> PyResult<()> {
        let ids = self::document_ids(document_ids)?;
        let dimension = embeddings.shape()[1];
        let values = row_major_borrowed(embeddings.as_array());
        let lengths = lengths(document_lengths)?;
        self.with(py, |writer| {
            writer.append(ids, &values, dimension, &lengths, threads)
        })
    }

    /// Stages the deletion of the documents present among ``document_ids``
    /// and returns how many there were.
    fn delete(&self, py: Python<'_>, document_ids: &Bound<'_, PyAny>) -> PyResult<usize> {
        let ids = self::document_ids(document_ids)?;
        self.with(py, |writer| Ok(writer.delete(&ids)))
    }

    /// Stages the present documents as one segment.
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, VectorStoreWriter::compact)
    }

    /// Publishes everything staged and returns the commit time readers
    /// report for it.
    fn commit(&self, py: Python<'_>) -> PyResult<SystemTime> {
        self.with(py, VectorStoreWriter::commit)
    }
}

#[pyfunction]
#[pyo3(signature = (query, documents, document_lengths, *, max_batch_tokens=None, threads=None))]
pub(crate) fn maxsim_scores_packed<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, f32>,
    documents: PyReadonlyArray2<'py, f32>,
    document_lengths: PyReadonlyArray1<'py, i64>,
    max_batch_tokens: Option<usize>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dimension = query.shape()[1];
    if documents.shape()[1] != dimension {
        return Err(PyValueError::new_err(format!(
            "query dimension {dimension} differs from document dimension {}",
            documents.shape()[1]
        )));
    }
    let lengths = lengths(document_lengths.as_any())?;
    let query = query.as_slice()?;
    let documents = documents.as_slice()?;
    let scores = py
        .detach(|| {
            lateweave::maxsim_scores(
                query,
                documents,
                &lengths,
                dimension,
                max_batch_tokens,
                threads,
            )
        })
        .map_err(to_py)?;
    Ok(PyArray1::from_vec(py, scores))
}
