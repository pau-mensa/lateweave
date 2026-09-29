use std::sync::Arc;
use std::time::SystemTime;

use lateweave::{
    Candidate, CandidateGenerator, DocumentKey, Gathered, MaxSimReranker, MultiVectorSource,
    PackedDocuments, Query, RankedDocument, Representation, Requirements, Reranker, ResourceBudget,
    Scored, SearchPipeline, SearchRequest, SearchResult, SearchTimings, Subset, VectorView,
};
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyMapping, PyTuple};

use crate::convert::{
    document_ids, from_py, representation, requirements, requirements_to_py, row_major, to_py,
};
use crate::query::PyQuery;
use crate::storage::PyVectorStore;

/// One gathered document of ``corpus``.
#[pyclass(name = "Candidate", frozen, module = "lateweave._native")]
pub(crate) struct PyCandidate {
    inner: Candidate,
}

#[pymethods]
impl PyCandidate {
    #[new]
    fn new(
        corpus: &str,
        document_id: &str,
        gather_score: f32,
        gather_rank: usize,
        provenance: String,
    ) -> Self {
        Self {
            inner: Candidate {
                key: DocumentKey::new(corpus, document_id),
                gather_score,
                gather_rank,
                provenance,
            },
        }
    }

    #[getter]
    fn corpus(&self) -> &str {
        self.inner.key.corpus()
    }

    #[getter]
    fn document_id(&self) -> &str {
        self.inner.key.id()
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
            "Candidate(corpus={:?}, document_id={:?}, gather_score={}, gather_rank={}, provenance={:?})",
            self.inner.key.corpus(),
            self.inner.key.id(),
            self.inner.gather_score,
            self.inner.gather_rank,
            self.inner.provenance
        )
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
        .map(|candidate| Ok(candidate?.cast::<PyCandidate>()?.get().inner.clone()))
        .collect()
}

/// What a gatherer found, and when the index it searched last committed:
/// every write committed to that index before ``as_of`` is reflected in it.
#[pyclass(name = "Gathered", frozen, module = "lateweave._native")]
pub(crate) struct PyGathered {
    inner: Gathered,
}

#[pymethods]
impl PyGathered {
    #[new]
    fn new(candidates: &Bound<'_, PyAny>, as_of: SystemTime) -> PyResult<Self> {
        Ok(Self {
            inner: Gathered {
                candidates: candidates_from_py(candidates)?,
                as_of,
            },
        })
    }

    #[getter]
    fn candidates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        candidates_to_py(py, &self.inner.candidates)
    }

    #[getter]
    fn as_of(&self) -> SystemTime {
        self.inner.as_of
    }
}

/// One score per candidate, or ``None`` for a document the reranker's index
/// does not hold, and when that index last committed.
#[pyclass(name = "Scored", frozen, module = "lateweave._native")]
pub(crate) struct PyScored {
    inner: Scored,
}

#[pymethods]
impl PyScored {
    #[new]
    fn new(scores: &Bound<'_, PyAny>, as_of: SystemTime) -> PyResult<Self> {
        let scores = scores
            .try_iter()?
            .map(|score| score?.extract::<Option<f32>>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: Scored { scores, as_of },
        })
    }

    #[getter]
    fn scores(&self) -> Vec<Option<f32>> {
        self.inner.scores.clone()
    }

    #[getter]
    fn as_of(&self) -> SystemTime {
        self.inner.as_of
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

fn subset_to_py<'py>(py: Python<'py>, subset: &Subset) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for (corpus, ids) in subset.iter() {
        output.set_item(
            corpus,
            PyFrozenSet::new(py, ids.iter().map(AsRef::<str>::as_ref))?,
        )?;
    }
    Ok(output)
}

fn subset_from_py(subset: &Bound<'_, PyAny>) -> PyResult<Subset> {
    let mapping = subset.cast::<PyMapping>().map_err(|_| {
        PyTypeError::new_err("subset must map corpora to iterables of document IDs")
    })?;
    mapping
        .items()?
        .iter()
        .try_fold(Subset::new(), |subset, item| {
            let (corpus, ids): (String, Bound<'_, PyAny>) = item.extract()?;
            Ok(subset.with(corpus, document_ids(&ids)?))
        })
}

/// What a Python stage declares, read once when it joins a pipeline.
struct Declaration {
    requires: Requirements,
    score_semantics: String,
    name: String,
}

impl Declaration {
    fn read(stage: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
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
    ) -> lateweave::Result<Gathered> {
        Python::attach(|py| {
            let arguments = PyDict::new(py);
            arguments.set_item(
                "subset",
                subset.map(|subset| subset_to_py(py, subset)).transpose()?,
            )?;
            let query = PyQuery {
                inner: query.clone(),
            };
            let gathered =
                self.object
                    .bind(py)
                    .call_method("gather", (query, limit), Some(&arguments))?;
            Ok(gathered.cast::<PyGathered>()?.get().inner.clone())
        })
        .map_err(from_py)
    }
}

struct PythonReranker {
    object: Py<PyAny>,
    declaration: Declaration,
}

impl Reranker for PythonReranker {
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
    ) -> lateweave::Result<Scored> {
        Python::attach(|py| {
            let arguments = PyDict::new(py);
            arguments.set_item("budget", PyResourceBudget { inner: *budget })?;
            let query = PyQuery {
                inner: query.clone(),
            };
            let scored = self.object.bind(py).call_method(
                "rerank",
                (query, candidates_to_py(py, candidates)?),
                Some(&arguments),
            )?;
            Ok(scored.cast::<PyScored>()?.get().inner.clone())
        })
        .map_err(from_py)
    }
}

/// A multi-vector source implemented in Python.
struct PythonSource {
    object: Py<PyAny>,
    corpus: String,
    representation: Representation,
    score_semantics: String,
}

impl PythonSource {
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            object: source.clone().unbind(),
            corpus: source.getattr("corpus")?.extract()?,
            representation: representation(&source.getattr("representation")?)?,
            score_semantics: source.getattr("score_semantics")?.extract()?,
        })
    }
}

impl MultiVectorSource for PythonSource {
    fn corpus(&self) -> &str {
        &self.corpus
    }

    fn representation(&self) -> &Representation {
        &self.representation
    }

    fn score_semantics(&self) -> &str {
        &self.score_semantics
    }

    fn view(&self) -> lateweave::Result<Arc<dyn VectorView>> {
        Python::attach(|py| {
            let view = self.object.bind(py).call_method0("view")?;
            Ok(Arc::new(PythonView {
                as_of: view.getattr("as_of")?.extract()?,
                object: view.unbind(),
            }) as Arc<dyn VectorView>)
        })
        .map_err(from_py)
    }
}

struct PythonView {
    object: Py<PyAny>,
    as_of: SystemTime,
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

impl VectorView for PythonView {
    fn as_of(&self) -> SystemTime {
        self.as_of
    }

    fn document_lengths(&self, document_ids: &[&str]) -> lateweave::Result<Vec<Option<usize>>> {
        Python::attach(|py| {
            let lengths = self
                .object
                .bind(py)
                .call_method1("document_lengths", (PyList::new(py, document_ids)?,))?;
            document_ids
                .iter()
                .map(|&document_id| lengths.call_method1("get", (document_id,))?.extract())
                .collect::<PyResult<Vec<Option<usize>>>>()
        })
        .map_err(from_py)
    }

    fn fetch(
        &self,
        document_ids: &[&str],
        threads: Option<usize>,
    ) -> lateweave::Result<PackedDocuments> {
        let (vectors, lengths, dimension) = Python::attach(|py| {
            let arguments = PyDict::new(py);
            arguments.set_item("threads", threads)?;
            let (vectors, document_lengths): (Bound<'_, PyAny>, Bound<'_, PyAny>) = self
                .object
                .bind(py)
                .call_method("fetch", (PyList::new(py, document_ids)?,), Some(&arguments))?
                .extract()?;
            let vectors = vectors.extract::<PyArrayLike2<'_, f32, AllowTypeChange>>()?;
            let view = vectors.as_array();
            Ok::<_, PyErr>((row_major(view), lengths(&document_lengths)?, view.ncols()))
        })
        .map_err(from_py)?;
        PackedDocuments::new(vectors, lengths, dimension)
    }
}

fn source(source: &Bound<'_, PyAny>) -> PyResult<Arc<dyn MultiVectorSource>> {
    if let Ok(store) = source.cast::<PyVectorStore>() {
        return Ok(store.get().inner.clone());
    }
    Ok(Arc::new(PythonSource::new(source)?))
}

/// MaxSim between the query's token matrix and each candidate's document,
/// read from the source of the candidate's corpus.
///
/// Each source, one per corpus, is a ``VectorStore`` or any object
/// implementing the ``MultiVectorSource`` protocol. Every rerank takes one
/// view of each source; a document a view does not hold is unscored.
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
            .map(|item| source(&item))
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

    #[pyo3(signature = (query, candidates, *, budget))]
    fn rerank(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyQuery>,
        candidates: &Bound<'_, PyAny>,
        budget: &Bound<'_, PyResourceBudget>,
    ) -> PyResult<PyScored> {
        let query = query.get().inner.clone();
        let candidates = candidates_from_py(candidates)?;
        let budget = budget.get().inner;
        let inner = py
            .detach(|| self.inner.rerank(&query, &candidates, &budget))
            .map_err(to_py)?;
        Ok(PyScored { inner })
    }
}

#[pyclass(name = "RankedDocument", frozen, module = "lateweave._native")]
pub(crate) struct PyRankedDocument {
    inner: RankedDocument,
}

#[pymethods]
impl PyRankedDocument {
    #[getter]
    fn corpus(&self) -> &str {
        self.inner.key.corpus()
    }

    #[getter]
    fn document_id(&self) -> &str {
        self.inner.key.id()
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
            "RankedDocument(corpus={:?}, document_id={:?}, score={}, rank={})",
            self.inner.key.corpus(),
            self.inner.key.id(),
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

    /// ``scores[i]`` scores ``candidates[i]``; ``None`` when the reranker's
    /// index does not hold it.
    #[getter]
    fn scores(&self) -> Vec<Option<f32>> {
        self.inner.scores.clone()
    }

    /// Every write committed to every index the search read before this is
    /// reflected in the result.
    #[getter]
    fn as_of(&self) -> SystemTime {
        self.inner.as_of
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
        output.set_item("dropped", diagnostics.dropped)?;
        output.set_item("gatherer", &diagnostics.gatherer)?;
        output.set_item("reranker", &diagnostics.reranker)?;
        output.set_item("score_semantics", &diagnostics.score_semantics)?;
        Ok(output)
    }
}

/// Gather, optionally rerank, then deterministic top-k.
///
/// Stages are Python objects implementing the ``CandidateGenerator`` and
/// ``Reranker`` protocols, or native stages such as ``MaxSimReranker``. A
/// document is ranked only when every stage holds it.
#[pyclass(name = "SearchPipeline", frozen, module = "lateweave._native")]
pub(crate) struct PySearchPipeline {
    inner: SearchPipeline,
}

#[pymethods]
impl PySearchPipeline {
    #[new]
    #[pyo3(signature = (gatherer, reranker=None))]
    fn new(gatherer: &Bound<'_, PyAny>, reranker: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let gatherer = Arc::new(PythonGatherer {
            object: gatherer.clone().unbind(),
            declaration: Declaration::read(gatherer)?,
        });
        let reranker = reranker
            .map(|reranker| -> PyResult<Arc<dyn Reranker>> {
                match reranker.cast::<PyMaxSimReranker>() {
                    Ok(native) => Ok(native.get().inner.clone()),
                    Err(_) => Ok(Arc::new(PythonReranker {
                        object: reranker.clone().unbind(),
                        declaration: Declaration::read(reranker)?,
                    })),
                }
            })
            .transpose()?;
        Ok(Self {
            inner: SearchPipeline::new(gatherer, reranker),
        })
    }

    /// ``query`` is a ``Query`` or plain text; ``subset`` maps corpora to the
    /// document IDs the search is restricted to.
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
