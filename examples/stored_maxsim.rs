//! A text gatherer implemented by the caller, reranked by MaxSim over a
//! lateweave vector store that a writer keeps moving.
//!
//! ```bash
//! cargo run --example stored_maxsim
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use lateweave::{
    Candidate, CandidateGenerator, DocumentKey, Encoding, Feature, Gathered, MaxSimReranker,
    MultiVectorSource, Query, Representation, Requirements, ResourceBudget, Result, SearchPipeline,
    SearchRequest, Subset, TokenMatrix, VectorStore, VectorStoreWriter, DEFAULT_FEATURE,
};

const DIMENSION: usize = 4;

/// Gathers every document containing a query term, scored by match count.
struct TermGatherer {
    requires: Requirements,
    documents: Vec<(&'static str, &'static str)>,
    indexed_at: SystemTime,
}

impl CandidateGenerator for TermGatherer {
    fn requires(&self) -> &Requirements {
        &self.requires
    }

    fn score_semantics(&self) -> &str {
        "term-match-count"
    }

    fn gather(&self, query: &Query, limit: usize, subset: Option<&Subset>) -> Result<Gathered> {
        let terms = query.text().split_whitespace().collect::<Vec<_>>();
        let mut matches = self
            .documents
            .iter()
            .map(|&(id, text)| (DocumentKey::new("fruit", id), text))
            .filter(|(key, _)| subset.map_or(true, |subset| subset.contains(key)))
            .map(|(key, text)| {
                let count = text
                    .split_whitespace()
                    .filter(|word| terms.contains(word))
                    .count();
                (key, count as f32)
            })
            .filter(|&(_, count)| count > 0.0)
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        Ok(Gathered {
            candidates: matches
                .into_iter()
                .take(limit)
                .enumerate()
                .map(|(rank, (key, gather_score))| Candidate {
                    key,
                    gather_score,
                    gather_rank: rank,
                    provenance: "term-gatherer".to_string(),
                })
                .collect(),
            as_of: self.indexed_at,
        })
    }
}

fn unit(axis: usize) -> [f32; DIMENSION] {
    let mut row = [0.0; DIMENSION];
    row[axis] = 1.0;
    row
}

fn main() -> Result<()> {
    let representation = Representation::new("toy-encoder", "1", DIMENSION, true)?;
    let directory = std::env::temp_dir().join(format!("lateweave-example-{}", std::process::id()));
    let path = directory.join("vectors");

    // The indexing side: one token per word, the axis standing in for what
    // a real encoder produces.
    let mut writer =
        VectorStoreWriter::create(&path, Encoding::Float32, "fruit", representation.clone())?;
    let embeddings = [unit(0), unit(2), unit(1), unit(2), unit(0), unit(3)].concat();
    writer.append(["a", "b", "c"], &embeddings, DIMENSION, &[2, 2, 2], None)?;
    writer.commit()?;

    // The serving side.
    let gatherer = TermGatherer {
        requires: BTreeMap::new(),
        documents: vec![("a", "red apple"), ("b", "green apple"), ("c", "red car")],
        indexed_at: SystemTime::now(),
    };
    let store = Arc::new(VectorStore::open(&path)?) as Arc<dyn MultiVectorSource>;
    let reranker = MaxSimReranker::new([store], DEFAULT_FEATURE)?;
    let pipeline = SearchPipeline::new(Arc::new(gatherer), Some(Arc::new(reranker)));

    let query = Query::new("apple").with_feature(
        DEFAULT_FEATURE,
        Feature::lazy(representation, || {
            TokenMatrix::new(unit(1).to_vec(), DIMENSION)
        }),
    );
    let request = SearchRequest::new(10, 2)
        .with_budget(ResourceBudget::new(4096, 64, Some(1))?)
        .with_max_lag(Duration::from_secs(60));

    let print = |label: &str| -> Result<()> {
        let result = pipeline.search(&query, &request)?;
        println!("{label}:");
        for document in &result.documents {
            println!(
                "  #{} {:?} score {}",
                document.rank, document.key, document.score
            );
        }
        println!("  {:?}", result.diagnostics);
        Ok(())
    };
    print("before the delete")?;

    // The indexing side deletes "b" from the vectors; the gatherer still
    // finds it, and the pipeline drops it.
    writer.delete(["b"]);
    writer.commit()?;
    print("after the delete")?;

    std::fs::remove_dir_all(directory)?;
    Ok(())
}
