use std::sync::Arc;

use lateweave::{
    Candidate, CandidateGenerator, MaxSimReranker, MultiVectorSource, PackedDocuments, Query,
    RankedDocument, Representation, Requirements, Reranker, ResourceBudget, SearchPipeline,
    SearchRequest, SearchResult, SearchTimings, Segment, Subset,
};
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyTuple};

use crate::convert::{from_py, representation, requirements, requirements_to_py, row_major, to_py};
use crate::query::PyQuery;
use crate::segment::PySegment;
use crate::storage::PyStoreSnapshot;

/// One gathered document: an internal ID of ``segment``.
#[pyclass(name = "Candidate", frozen, module = "lateweave._native")]
pub(crate) struct PyCandidate {
    inner: Candidate,
}

#[pymethods]
impl PyCandidate {
    #[new]
    fn new(
        segment: &Bound<'_, PySegment>,
        document_id: u64,
        gather_score: f32,
        gather_rank: usize,
        provenance: String,
    ) -> Self {
        Self {
            inner: Candidate {
                segment: segment.get().inner.clone(),
                document_id,
                gather_score,
                gather_rank,
                provenance,
            },
        }
    }

    #[getter]
    fn segment(&self) -> PySegment {
        self.inner.segment.clone().into()
    }

    #[getter]
    fn document_id(&self) -> u64 {
        self.inner.document_id
    }

    /// ``None`` when ``document_id`` is outside the segment.
    #[getter]
    fn external_id(&self) -> Option<&str> {
        self.inner.external_id()
    }

    #[getter]
    fn gather_score(&self) -> f32 {
        self.inner.gather_score
    }

    #[getter]
    fn gather_rank(&self) -> usize {
        self.inner.gather_rank
    }

    #[getter]
    fn provenance(&self) -> &str {
        &self.inner.provenance
    }

    fn __repr__(&self) -> String {
        format!(
            "Candidate(segment={:?}, document_id={}, gather_score={}, gather_rank={}, provenance={:?})",
            self.inner.segment.corpus_id(),
            self.inner.document_id,
            self.inner.gather_score,
            self.inner.gather_rank,
            self.inner.provenance
        )
    }
}

#[pyclass(name = "ResourceBudget", frozen, module = "lateweave._native")]
pub(crate) struct PyResourceBudget {
    inner: ResourceBudget,
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
        Ok(Self {
            inner: ResourceBudget::new(max_batch_tokens, max_documents_per_batch, threads)
                .map_err(to_py)?,
        })
    }

    #[getter]
    fn max_batch_tokens(&self) -> usize {
        self.inner.max_batch_tokens()
    }

    #[getter]
    fn max_documents_per_batch(&self) -> usize {
        self.inner.max_documents_per_batch()
    }

    #[getter]
    fn threads(&self) -> Option<usize> {
        self.inner.threads()
    }
}

fn candidates_to_py<'py>(
    py: Python<'py>,
    candidates: &[Candidate],
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        candidates.iter().map(|candidate| PyCandidate {
            inner: candidate.clone(),
        }),
    )
}

fn candidates_from_py(candidates: &Bound<'_, PyAny>) -> PyResult<Vec<Candidate>> {
    candidates
        .try_iter()?
        .map(|candidate| Ok(candidate?.downcast::<PyCandidate>()?.get().inner.clone()))
        .collect()
}

fn segments(stage: &Bound<'_, PyAny>) -> PyResult<Vec<Segment>> {
    stage
        .getattr("segments")?
        .try_iter()?
        .map(|segment| Ok(segment?.downcast::<PySegment>()?.get().inner.clone()))
        .collect()
}

fn segments_to_py<'py>(py: Python<'py>, segments: &[Segment]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, segments.iter().cloned().map(PySegment::from))
}

fn ids(values: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<u64>> {
    let values = values.extract::<PyArrayLike1<'_, i64, AllowTypeChange>>()?;
    values
        .as_array()
        .iter()
        .map(|&document_id| {
            u64::try_from(document_id).map_err(|_| {
                PyValueError::new_err(format!("{what} document IDs must not be negative"))
            })
        })
        .collect()
}

fn subset_to_py<'py>(py: Python<'py>, subset: &Subset) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for (corpus_id, ids) in subset.iter() {
        output.set_item(
            corpus_id,
            PyArray1::from_iter(py, ids.iter().map(|&document_id| document_id as i64)),
        )?;
    }
    Ok(output)
}

fn subset_from_py(subset: &Bound<'_, PyAny>) -> PyResult<Subset> {
    let mapping = subset.downcast::<PyMapping>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "subset must map corpus IDs to arrays of internal IDs",
        )
    })?;
    mapping
        .items()?
        .iter()
        .try_fold(Subset::new(), |subset, item| {
            let (corpus_id, values): (String, Bound<'_, PyAny>) = item.extract()?;
            Ok(subset.with(corpus_id, ids(&values, "subset")?))
        })
}

fn scores_from_py(scores: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    Ok(scores
        .extract::<PyArrayLike1<'_, f32, AllowTypeChange>>()?
        .as_array()
        .to_vec())
}

/// Identity a Python stage declares, read once when it joins a pipeline.
struct Declaration {
    segments: Vec<Segment>,
    requires: Requirements,
    score_semantics: String,
    name: String,
}

impl Declaration {
    fn read(stage: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            segments: segments(stage)?,
            requires: requirements(&stage.getattr("requires")?)?,
            score_semantics: stage.getattr("score_semantics")?.extract()?,
            name: stage.get_type().name()?.extract()?,
        })
    }
}

struct PythonGatherer {
    object: Py<PyAny>,
    declaration: Declaration,
}

impl CandidateGenerator for PythonGatherer {
    fn segments(&self) -> &[Segment] {
        &self.declaration.segments
    }

    fn requires(&self) -> &Requirements {
        &self.declaration.requires
    }

    fn score_semantics(&self) -> &str {
        &self.declaration.score_semantics
    }

    fn name(&self) -> &str {
        &self.declaration.name
    }

    fn gather(
        &self,
        query: &Query,
        limit: usize,
        subset: Option<&Subset>,
    ) -> lateweave::Result<Vec<Candidate>> {
        Python::attach(|py| {
            let arguments = PyDict::new(py);
            arguments.set_item(
                "subset",
                subset.map(|subset| subset_to_py(py, subset)).transpose()?,
            )?;
            let query = PyQuery {
                inner: query.clone(),
            };
            candidates_from_py(&self.object.bind(py).call_method(
                "gather",
                (query, limit),
                Some(&arguments),
            )?)
        })
        .map_err(from_py)
    }
}

struct PythonReranker {
    object: Py<PyAny>,
    declaration: Declaration,
}

impl Reranker for PythonReranker {
    fn segments(&self) -> &[Segment] {
        &self.declaration.segments
    }

    fn requires(&self) -> &Requirements {
        &self.declaration.requires
    }

    fn score_semantics(&self) -> &str {
        &self.declaration.score_semantics
    }

    fn name(&self) -> &str {
        &self.declaration.name
    }

    fn rerank(
        &self,
        query: &Query,
        candidates: &[Candidate],
        budget: &ResourceBudget,
    ) -> lateweave::Result<Vec<f32>> {
        Python::attach(|py| {
            let arguments = PyDict::new(py);
            arguments.set_item("budget", PyResourceBudget { inner: *budget })?;
            let query = PyQuery {
                inner: query.clone(),
            };
            scores_from_py(&self.object.bind(py).call_method(
                "rerank",
                (query, candidates_to_py(py, candidates)?),
                Some(&arguments),
            )?)
        })
        .map_err(from_py)
    }
}

/// A multi-vector source implemented in Python.
struct PythonSource {
    object: Py<PyAny>,
    segment: Segment,
    representation: Representation,
    score_semantics: String,
}

impl PythonSource {
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            object: source.clone().unbind(),
            segment: source
                .getattr("segment")?
                .downcast::<PySegment>()?
                .get()
                .inner
                .clone(),
            representation: representation(&source.getattr("representation")?)?,
            score_semantics: source.getattr("score_semantics")?.extract()?,
        })
    }
}

pub(crate) fn lengths(values: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let values = values.extract::<PyArrayLike1<'_, i64, AllowTypeChange>>()?;
    values
        .as_array()
        .iter()
        .enumerate()
        .map(|(position, &length)| {
            usize::try_from(length).map_err(|_| {
                PyValueError::new_err(format!(
                    "document length at position {position} must be positive"
                ))
            })
        })
        .collect()
}

impl MultiVectorSource for PythonSource {
    fn segment(&self) -> &Segment {
        &self.segment
    }

    fn representation(&self) -> &Representation {
        &self.representation
    }

    fn score_semantics(&self) -> &str {
        &self.score_semantics
    }

    fn document_lengths(&self, document_ids: &[u64]) -> lateweave::Result<Vec<usize>> {
        Python::attach(|py| {
            let lengths = self
                .object
                .bind(py)
                .call_method1("document_lengths", (document_ids,))?;
            document_ids
                .iter()
                .map(|&document_id| lengths.get_item(document_id)?.extract())
                .collect::<PyResult<Vec<usize>>>()
        })
        .map_err(from_py)
    }

    fn fetch(
        &self,
        document_ids: &[u64],
        threads: Option<usize>,
    ) -> lateweave::Result<PackedDocuments> {
        let (vectors, lengths, dimension) = Python::attach(|py| {
            let arguments = PyDict::new(py);
            arguments.set_item("threads", threads)?;
            let (vectors, document_lengths): (Bound<'_, PyAny>, Bound<'_, PyAny>) = self
                .object
                .bind(py)
                .call_method("fetch", (document_ids,), Some(&arguments))?
                .extract()?;
            let vectors = vectors.extract::<PyArrayLike2<'_, f32, AllowTypeChange>>()?;
            let view = vectors.as_array();
            Ok::<_, PyErr>((row_major(view), lengths(&document_lengths)?, view.ncols()))
        })
        .map_err(from_py)?;
        PackedDocuments::new(vectors, lengths, dimension)
    }
}

/// MaxSim between the query's token matrix and each candidate's document,
/// read from the source of the candidate's segment.
///
/// Each source is a vector-store snapshot, read natively, or any object
/// implementing the ``MultiVectorSource`` protocol, one per segment.
#[pyclass(name = "MaxSimReranker", frozen, module = "lateweave._native")]
pub(crate) struct PyMaxSimReranker {
    inner: Arc<MaxSimReranker>,
    sources: Py<PyTuple>,
}

#[pymethods]
impl PyMaxSimReranker {
    #[new]
    #[pyo3(signature = (sources, *, feature=lateweave::DEFAULT_FEATURE))]
    fn new(sources: &Bound<'_, PyAny>, feature: &str) -> PyResult<Self> {
        let sources = PyTuple::new(
            sources.py(),
            sources.try_iter()?.collect::<PyResult<Vec<_>>>()?,
        )?;
        let native = sources
            .iter()
            .map(|source| -> PyResult<Arc<dyn MultiVectorSource>> {
                Ok(match source.downcast::<PyStoreSnapshot>() {
                    Ok(snapshot) => snapshot.get().inner.clone(),
                    Err(_) => Arc::new(PythonSource::new(&source)?),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: Arc::new(MaxSimReranker::new(native, feature).map_err(to_py)?),
            sources: sources.unbind(),
        })
    }

    #[getter]
    fn sources(&self, py: Python<'_>) -> Py<PyTuple> {
        self.sources.clone_ref(py)
    }

    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        segments_to_py(py, self.inner.segments())
    }

    #[getter]
    fn requires<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        requirements_to_py(py, self.inner.requires())
    }

    #[getter]
    fn score_semantics(&self) -> &str {
        self.inner.score_semantics()
    }

    #[getter]
    fn feature(&self) -> &str {
        self.inner.feature()
    }

    /// One float32 score per candidate, in candidate order.
    #[pyo3(signature = (query, candidates, *, budget))]
    fn rerank<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyQuery>,
        candidates: &Bound<'py, PyAny>,
        budget: &Bound<'py, PyResourceBudget>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let query = query.get().inner.clone();
        let candidates = candidates_from_py(candidates)?;
        let budget = budget.get().inner;
        let scores = py
            .detach(|| self.inner.rerank(&query, &candidates, &budget))
            .map_err(to_py)?;
        Ok(PyArray1::from_vec(py, scores))
    }
}

#[pyclass(name = "RankedDocument", frozen, module = "lateweave._native")]
pub(crate) struct PyRankedDocument {
    inner: RankedDocument,
}

#[pymethods]
impl PyRankedDocument {
    #[getter]
    fn segment(&self) -> PySegment {
        self.inner.segment.clone().into()
    }

    #[getter]
    fn corpus_id(&self) -> &str {
        self.inner.segment.corpus_id()
    }

    #[getter]
    fn document_id(&self) -> u64 {
        self.inner.document_id
    }

    #[getter]
    fn external_id(&self) -> &str {
        self.inner
            .external_id()
            .expect("a search ranks only documents inside their segment")
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    fn __repr__(&self) -> String {
        format!(
            "RankedDocument(corpus_id={:?}, document_id={}, external_id={:?}, score={}, rank={})",
            self.corpus_id(),
            self.inner.document_id,
            self.external_id(),
            self.inner.score,
            self.inner.rank
        )
    }
}

#[pyclass(name = "SearchTimings", frozen, get_all, module = "lateweave._native")]
pub(crate) struct PySearchTimings {
    gather_seconds: f64,
    rerank_seconds: f64,
    total_seconds: f64,
}

impl From<&SearchTimings> for PySearchTimings {
    fn from(timings: &SearchTimings) -> Self {
        Self {
            gather_seconds: timings.gather.as_secs_f64(),
            rerank_seconds: timings.rerank.as_secs_f64(),
            total_seconds: timings.total.as_secs_f64(),
        }
    }
}

#[pyclass(name = "SearchResult", frozen, module = "lateweave._native")]
pub(crate) struct PySearchResult {
    inner: SearchResult,
}

#[pymethods]
impl PySearchResult {
    #[getter]
    fn documents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner
                .documents
                .iter()
                .map(|document| PyRankedDocument {
                    inner: document.clone(),
                }),
        )
    }

    #[getter]
    fn candidates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        candidates_to_py(py, &self.inner.candidates)
    }

    /// ``scores[i]`` scores ``candidates[i]``.
    #[getter]
    fn scores<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.scores)
    }

    #[getter]
    fn timings(&self) -> PySearchTimings {
        PySearchTimings::from(&self.inner.timings)
    }

    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let diagnostics = &self.inner.diagnostics;
        let output = PyDict::new(py);
        output.set_item("candidate_count", diagnostics.candidate_count)?;
        output.set_item("gatherer", &diagnostics.gatherer)?;
        output.set_item("reranker", &diagnostics.reranker)?;
        output.set_item("score_semantics", &diagnostics.score_semantics)?;
        Ok(output)
    }
}

/// Gather, optionally rerank, then deterministic top-k.
///
/// Stages are Python objects implementing the ``CandidateGenerator`` and
/// ``Reranker`` protocols, or native stages such as ``MaxSimReranker``.
#[pyclass(name = "SearchPipeline", frozen, module = "lateweave._native")]
pub(crate) struct PySearchPipeline {
    inner: SearchPipeline,
    gatherer: Py<PyAny>,
    reranker: Option<Py<PyAny>>,
}

#[pymethods]
impl PySearchPipeline {
    #[new]
    #[pyo3(signature = (gatherer, reranker=None))]
    fn new(gatherer: &Bound<'_, PyAny>, reranker: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let native_gatherer = Arc::new(PythonGatherer {
            object: gatherer.clone().unbind(),
            declaration: Declaration::read(gatherer)?,
        });
        let native_reranker = reranker
            .map(|reranker| -> PyResult<Arc<dyn Reranker>> {
                Ok(match reranker.downcast::<PyMaxSimReranker>() {
                    Ok(native) => native.get().inner.clone(),
                    Err(_) => Arc::new(PythonReranker {
                        object: reranker.clone().unbind(),
                        declaration: Declaration::read(reranker)?,
                    }),
                })
            })
            .transpose()?;
        Ok(Self {
            inner: SearchPipeline::new(native_gatherer, native_reranker).map_err(to_py)?,
            gatherer: gatherer.clone().unbind(),
            reranker: reranker.map(|reranker| reranker.clone().unbind()),
        })
    }

    #[getter]
    fn gatherer(&self, py: Python<'_>) -> Py<PyAny> {
        self.gatherer.clone_ref(py)
    }

    #[getter]
    fn reranker(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.reranker
            .as_ref()
            .map(|reranker| reranker.clone_ref(py))
    }

    /// ``query`` is a ``Query`` or plain text; ``subset`` maps corpus IDs to
    /// the ascending internal IDs the search is restricted to.
    #[pyo3(signature = (query, *, gather_limit, limit, subset=None, budget=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        gather_limit: i64,
        limit: i64,
        subset: Option<&Bound<'_, PyAny>>,
        budget: Option<&Bound<'_, PyResourceBudget>>,
    ) -> PyResult<PySearchResult> {
        let query = PyQuery::from_python(query)?;
        let subset = subset.map(subset_from_py).transpose()?;
        // A negative limit reaches the core as zero, which it rejects with
        // the same ValueError as any other non-positive limit.
        let request = SearchRequest {
            gather_limit: usize::try_from(gather_limit).unwrap_or(0),
            limit: usize::try_from(limit).unwrap_or(0),
            subset: subset.as_ref(),
            budget: budget.map_or_else(ResourceBudget::default, |budget| budget.get().inner),
        };
        let result = py
            .detach(|| self.inner.search(&query, &request))
            .map_err(to_py)?;
        Ok(PySearchResult { inner: result })
    }
}
