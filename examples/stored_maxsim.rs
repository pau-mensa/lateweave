//! A text gatherer implemented by the caller, reranked by MaxSim over a
//! lateweave vector store.
//!
//! ```bash
//! cargo run --example stored_maxsim
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use lateweave::{
    document_ids_digest, Candidate, CandidateGenerator, CorpusManifest, Feature, MaxSimReranker,
    Query, Representation, Requirements, ResourceBudget, Result, SearchPipeline, SearchRequest,
    StoreFormat, TokenMatrix, VectorStore, DEFAULT_FEATURE,
};

const DIMENSION: usize = 4;

/// Gathers every document containing a query term, scored by match count.
struct TermGatherer {
    corpus: CorpusManifest,
    requires: Requirements,
    documents: Vec<&'static str>,
}

impl CandidateGenerator for TermGatherer {
    fn corpus(&self) -> &CorpusManifest {
        &self.corpus
    }

    fn requires(&self) -> &Requirements {
        &self.requires
    }

    fn score_semantics(&self) -> &str {
        "term-match-count"
    }

    fn gather(
        &self,
        query: &Query,
        limit: usize,
        subset: Option<&[u64]>,
    ) -> Result<Vec<Candidate>> {
        let terms = query.text().split_whitespace().collect::<Vec<_>>();
        let mut matches = self
            .documents
            .iter()
            .enumerate()
            .map(|(document_id, text)| (document_id as u64, text))
            .filter(|(document_id, _)| {
                subset.map_or(true, |subset| subset.binary_search(document_id).is_ok())
            })
            .map(|(document_id, text)| {
                let count = text
                    .split_whitespace()
                    .filter(|word| terms.contains(word))
                    .count();
                (document_id, count as f32)
            })
            .filter(|&(_, count)| count > 0.0)
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        Ok(matches
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(rank, (document_id, gather_score))| Candidate {
                document_id,
                gather_score,
                gather_rank: rank,
                provenance: "term-gatherer".to_string(),
            })
            .collect())
    }
}

fn unit(axis: usize) -> [f32; DIMENSION] {
    let mut row = [0.0; DIMENSION];
    row[axis] = 1.0;
    row
}

fn main() -> Result<()> {
    let documents = vec!["red apple", "green apple", "red car"];
    let corpus = CorpusManifest::new(
        "fruit",
        "1",
        documents.len() as u64,
        document_ids_digest(["a", "b", "c"]),
    )?;
    let representation = Representation::new("toy-encoder", "1", DIMENSION, true)?;

    // One token per word; the axis stands in for what a real encoder produces.
    let embeddings = [unit(0), unit(2), unit(1), unit(2), unit(0), unit(3)].concat();
    let directory = std::env::temp_dir().join(format!("lateweave-example-{}", std::process::id()));
    let store = Arc::new(VectorStore::create(
        directory.join("vectors"),
        StoreFormat::Float32,
        &embeddings,
        DIMENSION,
        &[2, 2, 2],
        representation.clone(),
        None,
    )?);

    let gatherer = TermGatherer {
        corpus: corpus.clone(),
        requires: BTreeMap::new(),
        documents,
    };
    let reranker = MaxSimReranker::new(store, corpus, DEFAULT_FEATURE)?;
    let pipeline = SearchPipeline::new(Arc::new(gatherer), Some(Arc::new(reranker)))?;

    let query = Query::new("apple").with_feature(
        DEFAULT_FEATURE,
        Feature::lazy(representation, || {
            TokenMatrix::new(unit(1).to_vec(), DIMENSION)
        }),
    );
    let result = pipeline.search(
        &query,
        &SearchRequest::new(10, 2).with_budget(ResourceBudget::new(4096, 64, Some(1))?),
    )?;

    for document in &result.documents {
        println!(
            "#{} document {} score {}",
            document.rank, document.document_id, document.score
        );
    }
    println!("{:?}", result.diagnostics);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
