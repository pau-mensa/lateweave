//! Python bindings for lateweave. Every search, rerank, and store operation
//! runs in the Rust library; this crate only converts at the boundary and
//! adapts Python-implemented stages and sources to the Rust contracts.

mod convert;
mod query;
mod stages;
mod storage;

use pyo3::prelude::*;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add(
        "IncompatibleIndexError",
        py.get_type::<convert::IncompatibleIndexError>(),
    )?;
    module.add(
        "IncompatibleQueryError",
        py.get_type::<convert::IncompatibleQueryError>(),
    )?;
    module.add_class::<query::PyFeature>()?;
    module.add_class::<query::PyQuery>()?;
    module.add_class::<stages::PyCandidate>()?;
    module.add_class::<stages::PyGathered>()?;
    module.add_class::<stages::PyScored>()?;
    module.add_class::<stages::PyResourceBudget>()?;
    module.add_class::<stages::PyMaxSimReranker>()?;
    module.add_class::<stages::PyRankedDocument>()?;
    module.add_class::<stages::PySearchTimings>()?;
    module.add_class::<stages::PySearchResult>()?;
    module.add_class::<stages::PySearchPipeline>()?;
    module.add_class::<storage::PyVectorStore>()?;
    module.add_class::<storage::PyStoreView>()?;
    module.add_class::<storage::PyVectorStoreWriter>()?;
    module.add_function(wrap_pyfunction!(storage::maxsim_scores_packed, module)?)?;
    module.add_function(wrap_pyfunction!(convert::check_representation, module)?)?;
    module.add_function(wrap_pyfunction!(
        convert::assert_representations_compatible,
        module
    )?)?;
    Ok(())
}
