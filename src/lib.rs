pub mod core;
pub mod maxsim;
pub mod storage;

#[cfg(target_os = "macos")]
extern crate blas_src;

use numpy::{
    ndarray::{Array1, Array2},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type PyInt8StorageArrays<'py> = (Bound<'py, PyArray2<i8>>, Bound<'py, PyArray1<f32>>);

#[pyclass(name = "Candidate", frozen, get_all, module = "lateweave._native")]
#[derive(Clone, Debug)]
struct PyCandidate {
    document_id: u64,
    gather_score: f32,
    gather_rank: usize,
    provenance: String,
}

#[pymethods]
impl PyCandidate {
    #[new]
    fn new(document_id: u64, gather_score: f32, gather_rank: usize, provenance: String) -> Self {
        Self {
            document_id,
            gather_score,
            gather_rank,
            provenance,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Candidate(document_id={}, gather_score={}, gather_rank={}, provenance={:?})",
            self.document_id, self.gather_score, self.gather_rank, self.provenance
        )
    }
}

#[pyclass(name = "Score", frozen, get_all, module = "lateweave._native")]
#[derive(Clone, Debug)]
struct PyScore {
    document_id: u64,
    value: f32,
}

#[pymethods]
impl PyScore {
    #[new]
    fn new(document_id: u64, value: f32) -> Self {
        Self { document_id, value }
    }

    fn __repr__(&self) -> String {
        format!(
            "Score(document_id={}, value={})",
            self.document_id, self.value
        )
    }
}

#[pyclass(name = "ResourceBudget", frozen, get_all, module = "lateweave._native")]
#[derive(Clone, Debug)]
struct PyResourceBudget {
    max_batch_tokens: usize,
    max_documents_per_batch: usize,
    threads: Option<usize>,
}

#[pymethods]
impl PyResourceBudget {
    #[new]
    #[pyo3(signature = (*, max_batch_tokens=131_072, max_documents_per_batch=256, threads=None))]
    fn new(
        max_batch_tokens: usize,
        max_documents_per_batch: usize,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        if max_batch_tokens == 0 {
            return Err(PyValueError::new_err("max_batch_tokens must be positive"));
        }
        if max_documents_per_batch == 0 {
            return Err(PyValueError::new_err(
                "max_documents_per_batch must be positive",
            ));
        }
        if threads == Some(0) {
            return Err(PyValueError::new_err("threads must be positive"));
        }
        Ok(Self {
            max_batch_tokens,
            max_documents_per_batch,
            threads,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (candidate_ids, candidate_ranks, score_ids, score_values, limit))]
fn validate_and_rank(
    candidate_ids: Vec<u64>,
    candidate_ranks: Vec<usize>,
    score_ids: Vec<u64>,
    score_values: Vec<f32>,
    limit: usize,
) -> PyResult<Vec<usize>> {
    if candidate_ids.len() != candidate_ranks.len() {
        return Err(PyValueError::new_err(
            "candidate_ids and candidate_ranks must have equal lengths",
        ));
    }
    if score_ids.len() != score_values.len() {
        return Err(PyValueError::new_err(
            "score_ids and score_values must have equal lengths",
        ));
    }
    core::validate_and_rank(
        &candidate_ids,
        &candidate_ranks,
        &score_ids,
        &score_values,
        limit,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (query, documents, document_lengths, *, max_batch_tokens=None, threads=None))]
fn maxsim_scores_packed<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, f32>,
    documents: PyReadonlyArray2<'py, f32>,
    document_lengths: PyReadonlyArray1<'py, i64>,
    max_batch_tokens: Option<usize>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query_shape = query.shape();
    let document_shape = documents.shape();
    if query_shape[0] == 0 || query_shape[1] == 0 {
        return Err(PyValueError::new_err(
            "query must have shape [tokens, dimension] with non-zero axes",
        ));
    }
    if query_shape[0] > i32::MAX as usize || query_shape[1] > i32::MAX as usize {
        return Err(PyValueError::new_err(
            "query shape exceeds the 32-bit BLAS interface",
        ));
    }
    if query_shape[1] != document_shape[1] {
        return Err(PyValueError::new_err(format!(
            "query dimension {} differs from document dimension {}",
            query_shape[1], document_shape[1]
        )));
    }
    if max_batch_tokens == Some(0) {
        return Err(PyValueError::new_err("max_batch_tokens must be positive"));
    }
    if max_batch_tokens.is_some_and(|value| value > i32::MAX as usize) {
        return Err(PyValueError::new_err(
            "max_batch_tokens exceeds the 32-bit BLAS interface",
        ));
    }
    if threads == Some(0) {
        return Err(PyValueError::new_err("threads must be positive"));
    }

    // No non-finite scan here: it is a full streaming pass over every input
    // float, which costs more than the kernel itself at large candidate sets.
    // Non-finite embeddings propagate into the scores instead of raising.
    let query_values = query.as_slice()?;
    let document_values = documents.as_slice()?;
    let mut offsets = Vec::with_capacity(document_lengths.len() + 1);
    offsets.push(0);
    let mut total_tokens = 0usize;
    for (document, &length) in document_lengths.as_slice()?.iter().enumerate() {
        if length <= 0 {
            return Err(PyValueError::new_err(format!(
                "document length at position {document} must be positive"
            )));
        }
        if length > i32::MAX as i64 {
            return Err(PyValueError::new_err(format!(
                "document length at position {document} exceeds the 32-bit BLAS interface"
            )));
        }
        total_tokens = total_tokens
            .checked_add(length as usize)
            .ok_or_else(|| PyValueError::new_err("document lengths overflow usize"))?;
        offsets.push(total_tokens);
    }
    if total_tokens != document_shape[0] {
        return Err(PyValueError::new_err(format!(
            "document lengths sum to {total_tokens}, but documents contain {} rows",
            document_shape[0]
        )));
    }
    let scores = py
        .detach(|| {
            maxsim::maxsim_scores_packed(
                query_values,
                document_values,
                &offsets,
                query_shape[0],
                query_shape[1],
                max_batch_tokens,
                threads,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, scores))
}

#[pyfunction(name = "_storage_int8_encode")]
#[pyo3(signature = (embeddings, *, threads=None))]
fn storage_int8_encode<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<'py, f32>,
    threads: Option<usize>,
) -> PyResult<PyInt8StorageArrays<'py>> {
    let shape = embeddings.shape();
    let values = embeddings.as_slice()?;
    let (codes, scales) = py
        .detach(|| storage::int8_encode(values, shape[0], shape[1], threads))
        .map_err(PyValueError::new_err)?;
    let codes = Array2::from_shape_vec((shape[0], shape[1]), codes)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        codes.into_pyarray(py),
        Array1::from_vec(scales).into_pyarray(py),
    ))
}

#[pyfunction(name = "_storage_int8_decode")]
#[pyo3(signature = (codes, scales, *, normalize=true, threads=None))]
fn storage_int8_decode<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray2<'py, i8>,
    scales: PyReadonlyArray1<'py, f32>,
    normalize: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = codes.shape();
    let code_values = codes.as_slice()?;
    let scale_values = scales.as_slice()?;
    let output = py
        .detach(|| {
            storage::int8_decode(
                code_values,
                scale_values,
                shape[0],
                shape[1],
                normalize,
                threads,
            )
        })
        .map_err(PyValueError::new_err)?;
    let output = Array2::from_shape_vec((shape[0], shape[1]), output)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(output.into_pyarray(py))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCandidate>()?;
    module.add_class::<PyScore>()?;
    module.add_class::<PyResourceBudget>()?;
    module.add_function(wrap_pyfunction!(validate_and_rank, module)?)?;
    module.add_function(wrap_pyfunction!(maxsim_scores_packed, module)?)?;
    module.add_function(wrap_pyfunction!(storage_int8_encode, module)?)?;
    module.add_function(wrap_pyfunction!(storage_int8_decode, module)?)?;
    Ok(())
}
