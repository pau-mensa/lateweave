//! Errors and identity values crossing the Python boundary.

use lateweave::{CorpusManifest, Error, Representation, Requirements};
use numpy::ndarray::ArrayView2;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping};

create_exception!(
    lateweave,
    IncompatibleIndexError,
    PyValueError,
    "Two stages do not index the same documents."
);
create_exception!(
    lateweave,
    IncompatibleQueryError,
    PyValueError,
    "A query lacks a feature a stage needs, or supplies it from another encoder."
);

pub(crate) fn to_py(error: Error) -> PyErr {
    match error {
        Error::IncompatibleIndex(message) => IncompatibleIndexError::new_err(message),
        Error::IncompatibleQuery(message) => IncompatibleQueryError::new_err(message),
        Error::Io(error) => error.into(),
        Error::External(error) => match error.downcast::<PyErr>() {
            Ok(error) => *error,
            Err(error) => PyRuntimeError::new_err(error.to_string()),
        },
        error => PyValueError::new_err(error.to_string()),
    }
}

/// Carries an exception raised by Python code through the Rust pipeline so it
/// reaches the caller unchanged.
pub(crate) fn from_py(error: PyErr) -> Error {
    Error::External(Box::new(error))
}

fn attribute<'py, T: FromPyObject<'py>>(object: &Bound<'py, PyAny>, name: &str) -> PyResult<T> {
    object.getattr(name)?.extract()
}

fn non_negative(object: &Bound<'_, PyAny>, name: &str, message: &str) -> PyResult<u64> {
    let value: i64 = attribute(object, name)?;
    u64::try_from(value).map_err(|_| PyValueError::new_err(message.to_string()))
}

pub(crate) fn representation(object: &Bound<'_, PyAny>) -> PyResult<Representation> {
    let encoder: String = attribute(object, "encoder")?;
    let encoder_revision: String = attribute(object, "encoder_revision")?;
    let dimension: i64 = attribute(object, "dimension")?;
    let normalized: bool = attribute(object, "normalized")?;
    let similarity: String = attribute(object, "similarity")?;
    let query_template: String = attribute(object, "query_template")?;
    let document_template: String = attribute(object, "document_template")?;
    Representation::new(
        encoder,
        encoder_revision,
        usize::try_from(dimension).unwrap_or(0),
        normalized,
    )
    .and_then(|representation| representation.with_similarity(similarity))
    .map(|representation| representation.with_templates(query_template, document_template))
    .map_err(to_py)
}

pub(crate) fn corpus_manifest(object: &Bound<'_, PyAny>) -> PyResult<CorpusManifest> {
    let document_count = non_negative(
        object,
        "document_count",
        "corpus manifest document_count must not be negative",
    )?;
    let generation = non_negative(
        object,
        "generation",
        "corpus manifest generation must not be negative",
    )?;
    CorpusManifest::new(
        attribute::<String>(object, "corpus_id")?,
        attribute::<String>(object, "corpus_version")?,
        document_count,
        attribute::<String>(object, "document_ids_sha256")?,
    )
    .map(|manifest| manifest.with_generation(generation))
    .map_err(to_py)
}

pub(crate) fn requirements(object: &Bound<'_, PyAny>) -> PyResult<Requirements> {
    let mapping = object.downcast::<PyMapping>()?;
    mapping
        .items()?
        .iter()
        .map(|item| {
            let (name, value): (String, Bound<'_, PyAny>) = item.extract()?;
            Ok((name, representation(&value)?))
        })
        .collect()
}

fn manifest_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("lateweave.manifest")
}

pub(crate) fn representation_to_py<'py>(
    py: Python<'py>,
    representation: &Representation,
) -> PyResult<Bound<'py, PyAny>> {
    let arguments = PyDict::new(py);
    arguments.set_item("encoder", representation.encoder())?;
    arguments.set_item("encoder_revision", representation.encoder_revision())?;
    arguments.set_item("dimension", representation.dimension())?;
    arguments.set_item("normalized", representation.normalized())?;
    arguments.set_item("similarity", representation.similarity())?;
    arguments.set_item("query_template", representation.query_template())?;
    arguments.set_item("document_template", representation.document_template())?;
    manifest_module(py)?
        .getattr("Representation")?
        .call((), Some(&arguments))
}

pub(crate) fn corpus_manifest_to_py<'py>(
    py: Python<'py>,
    manifest: &CorpusManifest,
) -> PyResult<Bound<'py, PyAny>> {
    let arguments = PyDict::new(py);
    arguments.set_item("corpus_id", manifest.corpus_id())?;
    arguments.set_item("corpus_version", manifest.corpus_version())?;
    arguments.set_item("document_count", manifest.document_count())?;
    arguments.set_item("document_ids_sha256", manifest.document_ids_sha256())?;
    arguments.set_item("generation", manifest.generation())?;
    manifest_module(py)?
        .getattr("CorpusManifest")?
        .call((), Some(&arguments))
}

pub(crate) fn requirements_to_py<'py>(
    py: Python<'py>,
    requirements: &Requirements,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for (name, representation) in requirements {
        output.set_item(name, representation_to_py(py, representation)?)?;
    }
    Ok(output)
}

#[pyfunction(name = "_check_corpus_manifest")]
pub(crate) fn check_corpus_manifest(manifest: &Bound<'_, PyAny>) -> PyResult<()> {
    corpus_manifest(manifest).map(drop)
}

#[pyfunction(name = "_check_representation")]
pub(crate) fn check_representation(representation: &Bound<'_, PyAny>) -> PyResult<()> {
    self::representation(representation).map(drop)
}

#[pyfunction(name = "_assert_corpora_compatible")]
pub(crate) fn assert_corpora_compatible(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<()> {
    corpus_manifest(left)?
        .assert_compatible(&corpus_manifest(right)?)
        .map_err(to_py)
}

#[pyfunction(name = "_assert_representations_compatible")]
pub(crate) fn assert_representations_compatible(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<()> {
    representation(left)?
        .assert_compatible(&representation(right)?)
        .map_err(to_py)
}

#[pyfunction]
pub(crate) fn document_ids_digest(document_ids: Vec<String>) -> String {
    lateweave::document_ids_digest(document_ids)
}

/// Row-major values of a matrix: one `memcpy` when it is contiguous.
pub(crate) fn row_major(view: ArrayView2<'_, f32>) -> Vec<f32> {
    view.as_slice()
        .map_or_else(|| view.iter().copied().collect(), <[f32]>::to_vec)
}
