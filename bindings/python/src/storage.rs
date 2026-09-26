use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use lateweave::{StoreFormat, VectorStore};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use crate::convert::{representation, representation_to_py, row_major_borrowed, to_py};
use crate::stages::lengths;

/// Any float32 matrix, copied only when it is not already C-contiguous.
fn embeddings<'a>(embeddings: &'a PyReadonlyArray2<'_, f32>) -> (Cow<'a, [f32]>, usize) {
    (
        row_major_borrowed(embeddings.as_array()),
        embeddings.shape()[1],
    )
}

fn wrap(py: Python<'_>, store: VectorStore) -> PyResult<Py<PyAny>> {
    let base = PyVectorStore {
        inner: Arc::new(store),
    };
    let format = base.inner.format();
    Ok(match format {
        StoreFormat::Float32 => Py::new(
            py,
            PyClassInitializer::from(base).add_subclass(PyFloat32VectorStore),
        )?
        .into_any(),
        StoreFormat::Int8 => Py::new(
            py,
            PyClassInitializer::from(base).add_subclass(PyInt8VectorStore),
        )?
        .into_any(),
    })
}

fn open_as(py: Python<'_>, path: PathBuf, format: StoreFormat) -> PyResult<PyVectorStore> {
    let store = py.detach(|| VectorStore::open(&path)).map_err(to_py)?;
    if store.format() != format {
        return Err(PyValueError::new_err(format!(
            "{} is not a {} store",
            path.display(),
            format.name()
        )));
    }
    Ok(PyVectorStore {
        inner: Arc::new(store),
    })
}

#[allow(clippy::too_many_arguments)]
fn create(
    py: Python<'_>,
    format: StoreFormat,
    path: PathBuf,
    embeddings: PyReadonlyArray2<'_, f32>,
    document_lengths: &Bound<'_, PyAny>,
    representation: &Bound<'_, PyAny>,
    threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let (values, dimension) = self::embeddings(&embeddings);
    let lengths = lengths(document_lengths)?;
    let representation = self::representation(representation)?;
    let store = py
        .detach(|| {
            VectorStore::create(
                &path,
                format,
                &values,
                dimension,
                &lengths,
                representation,
                threads,
            )
        })
        .map_err(to_py)?;
    wrap(py, store)
}

/// Memory-mapped fixed-width token records; a ``MultiVectorSource``.
#[pyclass(name = "VectorStore", subclass, frozen, module = "lateweave._native")]
pub(crate) struct PyVectorStore {
    pub(crate) inner: Arc<VectorStore>,
}

#[pymethods]
impl PyVectorStore {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path().to_path_buf()
    }

    #[getter]
    fn representation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        representation_to_py(py, self.inner.representation())
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn document_count(&self) -> u64 {
        self.inner.document_count()
    }

    #[getter]
    fn token_count(&self) -> u64 {
        self.inner.token_count()
    }

    fn document_lengths<'py>(
        &self,
        py: Python<'py>,
        document_ids: Vec<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lengths = self.inner.document_lengths(&document_ids).map_err(to_py)?;
        let output = PyDict::new(py);
        for (document_id, length) in document_ids.into_iter().zip(lengths) {
            output.set_item(document_id, length)?;
        }
        Ok(output)
    }

    /// Packed float32 ``[tokens, dimension]`` vectors of ``document_ids`` in
    /// the requested order, plus their int64 lengths.
    #[pyo3(signature = (document_ids, *, threads=None))]
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        document_ids: Vec<u64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let packed = py
            .detach(|| self.inner.fetch(&document_ids, threads))
            .map_err(to_py)?;
        let rows = packed.vectors().len() / packed.dimension();
        let vectors =
            PyArray1::from_slice(py, packed.vectors()).reshape([rows, packed.dimension()])?;
        let lengths = PyArray1::from_iter(py, packed.lengths().iter().map(|&length| length as i64));
        PyTuple::new(py, [vectors.into_any(), lengths.into_any()])
    }

    #[pyo3(signature = (embeddings, document_lengths, *, threads=None))]
    fn append(
        &self,
        py: Python<'_>,
        embeddings: PyReadonlyArray2<'_, f32>,
        document_lengths: &Bound<'_, PyAny>,
        threads: Option<usize>,
    ) -> PyResult<()> {
        let (values, dimension) = self::embeddings(&embeddings);
        let lengths = lengths(document_lengths)?;
        py.detach(|| self.inner.append(&values, dimension, &lengths, threads))
            .map_err(to_py)
    }

    /// Removes documents and compacts internal IDs to ``0..n-1``.
    fn delete(&self, py: Python<'_>, document_ids: Vec<u64>) -> PyResult<()> {
        py.detach(|| self.inner.delete(&document_ids))
            .map_err(to_py)
    }
}

/// Exact float32 token vectors.
#[pyclass(name = "Float32VectorStore", extends = PyVectorStore, frozen, module = "lateweave._native")]
pub(crate) struct PyFloat32VectorStore;

#[pymethods]
impl PyFloat32VectorStore {
    #[classattr]
    fn format() -> &'static str {
        StoreFormat::Float32.name()
    }

    #[classattr]
    fn score_semantics() -> &'static str {
        StoreFormat::Float32.score_semantics()
    }

    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<(Self, PyVectorStore)> {
        Ok((Self, open_as(py, path, StoreFormat::Float32)?))
    }

    #[classmethod]
    #[pyo3(signature = (path, embeddings, document_lengths, representation, *, threads=None))]
    fn create(
        _class: &Bound<'_, PyType>,
        py: Python<'_>,
        path: PathBuf,
        embeddings: PyReadonlyArray2<'_, f32>,
        document_lengths: &Bound<'_, PyAny>,
        representation: &Bound<'_, PyAny>,
        threads: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        create(
            py,
            StoreFormat::Float32,
            path,
            embeddings,
            document_lengths,
            representation,
            threads,
        )
    }
}

/// Symmetric INT8 per token with one float32 row scale; lossy.
#[pyclass(name = "Int8VectorStore", extends = PyVectorStore, frozen, module = "lateweave._native")]
pub(crate) struct PyInt8VectorStore;

#[pymethods]
impl PyInt8VectorStore {
    #[classattr]
    fn format() -> &'static str {
        StoreFormat::Int8.name()
    }

    #[classattr]
    fn score_semantics() -> &'static str {
        StoreFormat::Int8.score_semantics()
    }

    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<(Self, PyVectorStore)> {
        Ok((Self, open_as(py, path, StoreFormat::Int8)?))
    }

    #[classmethod]
    #[pyo3(signature = (path, embeddings, document_lengths, representation, *, threads=None))]
    fn create(
        _class: &Bound<'_, PyType>,
        py: Python<'_>,
        path: PathBuf,
        embeddings: PyReadonlyArray2<'_, f32>,
        document_lengths: &Bound<'_, PyAny>,
        representation: &Bound<'_, PyAny>,
        threads: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        create(
            py,
            StoreFormat::Int8,
            path,
            embeddings,
            document_lengths,
            representation,
            threads,
        )
    }
}

/// Opens a store as the class its format names.
#[pyfunction]
pub(crate) fn open_vector_store(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let store = py.detach(|| VectorStore::open(&path)).map_err(to_py)?;
    wrap(py, store)
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
