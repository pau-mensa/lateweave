use std::any::Any;
use std::sync::{Arc, OnceLock};

use lateweave::{Feature, FeatureValue, Query, TokenMatrix};
use numpy::{AllowTypeChange, PyArray1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{from_py, representation, representation_to_py, to_py};

/// A feature value produced in Python. Python stages get the object back
/// unchanged; Rust stages that need a token matrix get it converted once.
struct PythonValue {
    object: Py<PyAny>,
    matrix: OnceLock<Option<TokenMatrix>>,
}

impl PythonValue {
    fn new(object: Py<PyAny>) -> Self {
        Self {
            object,
            matrix: OnceLock::new(),
        }
    }
}

impl FeatureValue for PythonValue {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn token_matrix(&self) -> Option<&TokenMatrix> {
        self.matrix
            .get_or_init(|| Python::attach(|py| token_matrix(self.object.bind(py)).ok()))
            .as_ref()
    }
}

fn token_matrix(object: &Bound<'_, PyAny>) -> PyResult<TokenMatrix> {
    let array = object.extract::<PyArrayLike2<'_, f32, AllowTypeChange>>()?;
    let view = array.as_array();
    TokenMatrix::new(view.iter().copied().collect(), view.ncols()).map_err(to_py)
}

pub(crate) fn value_to_py(py: Python<'_>, value: &dyn FeatureValue) -> PyResult<Py<PyAny>> {
    if let Some(value) = value.downcast_ref::<PythonValue>() {
        return Ok(value.object.clone_ref(py));
    }
    if let Some(matrix) = value.token_matrix() {
        return Ok(PyArray1::from_slice(py, matrix.values())
            .reshape([matrix.tokens(), matrix.dimension()])?
            .into_any()
            .unbind());
    }
    Err(PyTypeError::new_err(
        "feature value was produced in Rust and has no Python representation",
    ))
}

/// One query representation, materialized at most once.
///
/// Pass ``value`` when it is already computed, or ``provider`` to defer the
/// encoding until a stage asks for it.
#[pyclass(name = "Feature", frozen, module = "lateweave._native")]
pub(crate) struct PyFeature {
    inner: Arc<Feature>,
}

#[pymethods]
impl PyFeature {
    #[new]
    #[pyo3(signature = (representation, value=None, *, provider=None))]
    fn new(
        representation: &Bound<'_, PyAny>,
        value: Option<Py<PyAny>>,
        provider: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let representation = self::representation(representation)?;
        let feature = match (value, provider) {
            (Some(value), None) => Feature::new(representation, PythonValue::new(value)),
            (None, Some(provider)) => Feature::lazy(representation, move || {
                Python::attach(|py| provider.call0(py))
                    .map(PythonValue::new)
                    .map_err(from_py)
            }),
            _ => {
                return Err(PyValueError::new_err(
                    "a feature needs exactly one of value or provider",
                ))
            }
        };
        Ok(Self {
            inner: Arc::new(feature),
        })
    }

    #[getter]
    fn representation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        representation_to_py(py, self.inner.representation())
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, self.inner.value().map_err(to_py)?)
    }
}

/// Raw text plus the named features stages may consume.
#[pyclass(name = "Query", frozen, module = "lateweave._native")]
pub(crate) struct PyQuery {
    pub(crate) inner: Query,
}

impl PyQuery {
    pub(crate) fn from_python(query: &Bound<'_, PyAny>) -> PyResult<Query> {
        if let Ok(text) = query.extract::<String>() {
            return Ok(Query::new(text));
        }
        Ok(query.downcast::<PyQuery>()?.get().inner.clone())
    }
}

#[pymethods]
impl PyQuery {
    #[new]
    #[pyo3(signature = (text, **features))]
    fn new(text: String, features: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut query = Query::new(text);
        for (name, feature) in features.into_iter().flatten() {
            let feature = feature.downcast::<PyFeature>()?.get().inner.clone();
            query = query.with_feature(name.extract::<String>()?, feature);
        }
        Ok(Self { inner: query })
    }

    #[getter]
    fn text(&self) -> &str {
        self.inner.text()
    }

    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        for (name, feature) in self.inner.features() {
            output.set_item(
                name,
                PyFeature {
                    inner: feature.clone(),
                },
            )?;
        }
        Ok(output)
    }

    /// The value of feature ``name``, which must come from ``representation``.
    fn feature(
        &self,
        py: Python<'_>,
        name: &str,
        representation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let representation = self::representation(representation)?;
        value_to_py(
            py,
            self.inner.feature(name, &representation).map_err(to_py)?,
        )
    }

    fn __repr__(&self) -> String {
        let names = self.inner.features().keys().cloned().collect::<Vec<_>>();
        format!("Query({:?}, features={names:?})", self.inner.text())
    }
}
