use std::sync::Arc;

use lateweave::{
    Candidate, CandidateGenerator, CorpusManifest, MaxSimReranker, MultiVectorSource,
    PackedDocuments, Query, RankedDocument, Representation, Requirements, Reranker, ResourceBudget,
    Score, SearchPipeline, SearchRequest, SearchResult, SearchTimings,
};
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::{
    corpus_manifest, corpus_manifest_to_py, from_py, representation, requirements,
    requirements_to_py, to_py,
};
use crate::query::PyQuery;
use crate::storage::PyVectorStore;

#[pyclass(name = "Candidate", frozen, module = "lateweave._native")]
pub(crate) struct PyCandidate {
    inner: Candidate,
}

#[pymethods]
impl PyCandidate {
    #[new]
    fn new(document_id: u64, gather_score: f32, gather_rank: usize, provenance: String) -> Self {
        Self {
            inner: Candidate {
                document_id,
                gather_score,
                gather_rank,
                provenance,
            },
        }
    }

    #[getter]
    fn document_id(&self) -> u64 {
        self.inner.document_id
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
            "Candidate(document_id={}, gather_score={}, gather_rank={}, provenance={:?})",
            self.inner.document_id,
            self.inner.gather_score,
            self.inner.gather_rank,
            self.inner.provenance
        )
    }
}

#[pyclass(name = "Score", frozen, module = "lateweave._native")]
pub(crate) struct PyScore {
    inner: Score,
}

#[pymethods]
impl PyScore {
    #[new]
    fn new(document_id: u64, value: f32) -> Self {
        Self {
            inner: Score { document_id, value },
        }
    }

    #[getter]
    fn document_id(&self) -> u64 {
        self.inner.document_id
    }

    #[getter]
    fn value(&self) -> f32 {
        self.inner.value
    }

    fn __repr__(&self) -> String {
        format!(
            "Score(document_id={}, value={})",
            self.inner.document_id, self.inner.value
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

fn scores_to_py<'py>(py: Python<'py>, scores: &[Score]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, scores.iter().map(|&score| PyScore { inner: score }))
}

/// Identity a Python stage declares, read once when it joins a pipeline.
struct Declaration {
    corpus: CorpusManifest,
    requires: Requirements,
    score_semantics: String,
    name: String,
}

impl Declaration {
    fn read(stage: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            corpus: corpus_manifest(&stage.getattr("corpus")?)?,
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
    fn corpus(&self) -> &CorpusManifest {
        &self.declaration.corpus
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
        subset: Option<&[u64]>,
    ) -> lateweave::Result<Vec<Candidate>> {
        Python::attach(|py| {
            let arguments = PyDict::new(py);
            let subset = subset.map(|subset| {
                PyArray1::from_iter(py, subset.iter().map(|&document_id| document_id as i64))
            });
            arguments.set_item("subset", subset)?;
            let query = PyQuery {
                inner: query.clone(),
            };
            self.object
                .bind(py)
                .call_method("gather", (query, limit), Some(&arguments))?
                .try_iter()?
                .map(|candidate| Ok(candidate?.downcast::<PyCandidate>()?.get().inner.clone()))
                .collect::<PyResult<Vec<Candidate>>>()
        })
        .map_err(from_py)
    }
}

struct PythonReranker {
    object: Py<PyAny>,
    declaration: Declaration,
}

impl Reranker for PythonReranker {
    fn corpus(&self) -> &CorpusManifest {
        &self.declaration.corpus
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
    ) -> lateweave::Result<Vec<Score>> {
        Python::attach(|py| {
            let arguments = PyDict::new(py);
            arguments.set_item("budget", PyResourceBudget { inner: *budget })?;
            let query = PyQuery {
                inner: query.clone(),
            };
            self.object
                .bind(py)
                .call_method(
                    "rerank",
                    (query, candidates_to_py(py, candidates)?),
                    Some(&arguments),
                )?
                .try_iter()?
                .map(|score| Ok(score?.downcast::<PyScore>()?.get().inner))
                .collect::<PyResult<Vec<Score>>>()
        })
        .map_err(from_py)
    }
}

/// A multi-vector source implemented in Python.
struct PythonSource {
    object: Py<PyAny>,
    representation: Representation,
    score_semantics: String,
    document_count: u64,
}

impl PythonSource {
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            object: source.clone().unbind(),
            representation: representation(&source.getattr("representation")?)?,
            score_semantics: source.getattr("score_semantics")?.extract()?,
            document_count: source.getattr("document_count")?.extract()?,
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
    fn representation(&self) -> &Representation {
        &self.representation
    }

    fn score_semantics(&self) -> &str {
        &self.score_semantics
    }

    fn document_count(&self) -> u64 {
        self.document_count
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
            Ok::<_, PyErr>((
                view.iter().copied().collect(),
                lengths(&document_lengths)?,
                view.ncols(),
            ))
        })
        .map_err(from_py)?;
        PackedDocuments::new(vectors, lengths, dimension)
    }
}

/// MaxSim between the query's token matrix and a source's documents.
///
/// ``source`` is a lateweave vector store, read natively, or any object
/// implementing the ``MultiVectorSource`` protocol.
#[pyclass(name = "MaxSimReranker", frozen, module = "lateweave._native")]
pub(crate) struct PyMaxSimReranker {
    inner: Arc<MaxSimReranker>,
    source: Py<PyAny>,
}

#[pymethods]
impl PyMaxSimReranker {
    #[new]
    #[pyo3(signature = (source, corpus, *, feature=lateweave::DEFAULT_FEATURE))]
    fn new(source: &Bound<'_, PyAny>, corpus: &Bound<'_, PyAny>, feature: &str) -> PyResult<Self> {
        let native: Arc<dyn MultiVectorSource> = match source.downcast::<PyVectorStore>() {
            Ok(store) => store.get().inner.clone(),
            Err(_) => Arc::new(PythonSource::new(source)?),
        };
        Ok(Self {
            inner: Arc::new(
                MaxSimReranker::new(native, corpus_manifest(corpus)?, feature).map_err(to_py)?,
            ),
            source: source.clone().unbind(),
        })
    }

    #[getter]
    fn source(&self, py: Python<'_>) -> Py<PyAny> {
        self.source.clone_ref(py)
    }

    #[getter]
    fn corpus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        corpus_manifest_to_py(py, self.inner.corpus())
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
    fn rerank<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyQuery>,
        candidates: &Bound<'py, PyAny>,
        budget: &Bound<'py, PyResourceBudget>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let query = query.get().inner.clone();
        let candidates = candidates
            .try_iter()?
            .map(|candidate| Ok(candidate?.downcast::<PyCandidate>()?.get().inner.clone()))
            .collect::<PyResult<Vec<_>>>()?;
        let budget = budget.get().inner;
        let scores = py
            .detach(|| self.inner.rerank(&query, &candidates, &budget))
            .map_err(to_py)?;
        scores_to_py(py, &scores)
    }
}

#[pyclass(name = "RankedDocument", frozen, get_all, module = "lateweave._native")]
pub(crate) struct PyRankedDocument {
    document_id: u64,
    score: f32,
    rank: usize,
}

impl From<&RankedDocument> for PyRankedDocument {
    fn from(document: &RankedDocument) -> Self {
        Self {
            document_id: document.document_id,
            score: document.score,
            rank: document.rank,
        }
    }
}

#[pymethods]
impl PyRankedDocument {
    fn __repr__(&self) -> String {
        format!(
            "RankedDocument(document_id={}, score={}, rank={})",
            self.document_id, self.score, self.rank
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
        PyTuple::new(py, self.inner.documents.iter().map(PyRankedDocument::from))
    }

    #[getter]
    fn candidates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        candidates_to_py(py, &self.inner.candidates)
    }

    #[getter]
    fn scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        scores_to_py(py, &self.inner.scores)
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

    /// ``query`` is a ``Query`` or plain text; ``subset`` restricts the search
    /// to those internal IDs, in ascending order.
    #[pyo3(signature = (query, *, gather_limit, limit, subset=None, budget=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        gather_limit: usize,
        limit: usize,
        subset: Option<PyArrayLike1<'_, i64, AllowTypeChange>>,
        budget: Option<&Bound<'_, PyResourceBudget>>,
    ) -> PyResult<PySearchResult> {
        let query = PyQuery::from_python(query)?;
        let subset = subset
            .map(|subset| {
                subset
                    .as_array()
                    .iter()
                    .map(|&document_id| {
                        u64::try_from(document_id).map_err(|_| {
                            PyValueError::new_err("subset document IDs must not be negative")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let request = SearchRequest {
            gather_limit,
            limit,
            subset: subset.as_deref(),
            budget: budget.map_or_else(ResourceBudget::default, |budget| budget.get().inner),
        };
        let result = py
            .detach(|| self.inner.search(&query, &request))
            .map_err(to_py)?;
        Ok(PySearchResult { inner: result })
    }
}
