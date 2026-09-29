//! Errors and identity values crossing the Python boundary.

use std::borrow::Cow;
use std::sync::Arc;

use lateweave::{Error, Representation, Requirements};
use numpy::ndarray::ArrayView2;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyString};

create_exception!(
    lateweave,
    IncompatibleIndexError,
    PyValueError,
    "Stages or sources that cannot be composed: another representation, another score scale, or a corpus the reranker has no source for."
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

fn attribute<'py, T: FromPyObjectOwned<'py>>(
    object: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<T> {
    object.getattr(name)?.extract().map_err(Into::into)
}

pub(crate) fn representation(object: &Bound<'_, PyAny>) -> PyResult<Representation> {
    let encoder: String = attribute(object, "encoder")?;
    let encoder_revision: String = attribute(object, "encoder_revision")?;
    let dimension: i64 = attribute(object, "dimension")?;
    let normalized: bool = attribute(object, "normalized")?;
    let query_template: String = attribute(object, "query_template")?;
    let document_template: String = attribute(object, "document_template")?;
    Representation::new(
        encoder,
        encoder_revision,
        usize::try_from(dimension).unwrap_or(0),
        normalized,
    )
    .map(|representation| representation.with_templates(query_template, document_template))
    .map_err(to_py)
}

pub(crate) fn requirements(object: &Bound<'_, PyAny>) -> PyResult<Requirements> {
    let mapping = object.cast::<PyMapping>()?;
    mapping
        .items()?
        .iter()
        .map(|item| {
            let (name, value): (String, Bound<'_, PyAny>) = item.extract()?;
            Ok((name, representation(&value)?))
        })
        .collect()
}

fn representation_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("lateweave.representation")
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
    arguments.set_item("query_template", representation.query_template())?;
    arguments.set_item("document_template", representation.document_template())?;
    representation_module(py)?
        .getattr("Representation")?
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

#[pyfunction(name = "_check_representation")]
pub(crate) fn check_representation(representation: &Bound<'_, PyAny>) -> PyResult<()> {
    self::representation(representation).map(drop)
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

/// Document IDs from any iterable of strings. A bare string is refused
/// rather than read as one ID per character.
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

/// Row-major values of a matrix: one `memcpy` when it is contiguous.
pub(crate) fn row_major(view: ArrayView2<'_, f32>) -> Vec<f32> {
    view.as_slice()
        .map_or_else(|| view.iter().copied().collect(), <[f32]>::to_vec)
}

/// Row-major values of a matrix, borrowed when it is already C-contiguous.
pub(crate) fn row_major_borrowed<'a>(view: ArrayView2<'a, f32>) -> Cow<'a, [f32]> {
    match view.to_slice() {
        Some(values) => Cow::Borrowed(values),
        None => Cow::Owned(view.iter().copied().collect()),
    }
}
