//! Identity contracts checked before any search runs.
//!
//! Two identities are kept apart. A [`CorpusManifest`] says *which documents* a
//! stage indexes: every stage in a pipeline must agree on it. A
//! [`Representation`] says *which encoder* produced a vector feature: a stage
//! that consumes such a feature must agree on it with the query that supplies
//! it.

use std::fmt::Debug;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// SHA-256 over external IDs in internal-ID order; identifies an ID binding.
///
/// Each ID is length-prefixed so that `["ab", "c"]` and `["a", "bc"]` differ.
pub fn document_ids_digest<I>(document_ids: I) -> String
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut digest = Sha256::new();
    for document_id in document_ids {
        let encoded = document_id.as_ref().as_bytes();
        digest.update((encoded.len() as u64).to_le_bytes());
        digest.update(encoded);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn mismatch<T: PartialEq + Debug>(mismatches: &mut Vec<String>, name: &str, left: &T, right: &T) {
    if left != right {
        mismatches.push(format!("{name}: {left:?} != {right:?}"));
    }
}

fn require_non_empty(value: &str, message: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::invalid(message));
    }
    Ok(())
}

/// Identity of one indexed document set at one mutation generation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "CorpusManifestFields")]
pub struct CorpusManifest {
    corpus_id: String,
    corpus_version: String,
    document_count: u64,
    document_ids_sha256: String,
    generation: u64,
}

#[derive(Deserialize)]
struct CorpusManifestFields {
    corpus_id: String,
    corpus_version: String,
    document_count: u64,
    document_ids_sha256: String,
    #[serde(default)]
    generation: u64,
}

impl TryFrom<CorpusManifestFields> for CorpusManifest {
    type Error = Error;

    fn try_from(fields: CorpusManifestFields) -> Result<Self> {
        Ok(Self::new(
            fields.corpus_id,
            fields.corpus_version,
            fields.document_count,
            fields.document_ids_sha256,
        )?
        .with_generation(fields.generation))
    }
}

impl CorpusManifest {
    pub fn new(
        corpus_id: impl Into<String>,
        corpus_version: impl Into<String>,
        document_count: u64,
        document_ids_sha256: impl Into<String>,
    ) -> Result<Self> {
        let manifest = Self {
            corpus_id: corpus_id.into(),
            corpus_version: corpus_version.into(),
            document_count,
            document_ids_sha256: document_ids_sha256.into(),
            generation: 0,
        };
        require_non_empty(
            &manifest.corpus_id,
            "corpus manifest corpus_id must not be empty",
        )?;
        require_non_empty(
            &manifest.corpus_version,
            "corpus manifest corpus_version must not be empty",
        )?;
        require_non_empty(
            &manifest.document_ids_sha256,
            "corpus manifest document_ids_sha256 must not be empty",
        )?;
        Ok(manifest)
    }

    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    pub fn corpus_id(&self) -> &str {
        &self.corpus_id
    }

    pub fn corpus_version(&self) -> &str {
        &self.corpus_version
    }

    pub fn document_count(&self) -> u64 {
        self.document_count
    }

    pub fn document_ids_sha256(&self) -> &str {
        &self.document_ids_sha256
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_slice(&fs::read(path)?).map_err(|error| {
            Error::invalid(format!(
                "invalid corpus manifest {}: {error}",
                path.display()
            ))
        })
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut encoded = serde_json::to_string_pretty(self)
            .map_err(|error| Error::invalid(error.to_string()))?;
        encoded.push('\n');
        fs::write(path, encoded)?;
        Ok(())
    }

    /// Fails with [`Error::IncompatibleIndex`] naming every differing field.
    pub fn assert_compatible(&self, other: &Self) -> Result<()> {
        let mut mismatches = Vec::new();
        mismatch(
            &mut mismatches,
            "corpus_id",
            &self.corpus_id,
            &other.corpus_id,
        );
        mismatch(
            &mut mismatches,
            "corpus_version",
            &self.corpus_version,
            &other.corpus_version,
        );
        mismatch(
            &mut mismatches,
            "document_count",
            &self.document_count,
            &other.document_count,
        );
        mismatch(
            &mut mismatches,
            "document_ids_sha256",
            &self.document_ids_sha256,
            &other.document_ids_sha256,
        );
        mismatch(
            &mut mismatches,
            "generation",
            &self.generation,
            &other.generation,
        );
        if mismatches.is_empty() {
            return Ok(());
        }
        Err(Error::IncompatibleIndex(format!(
            "stages index different corpora ({})",
            mismatches.join("; ")
        )))
    }
}

/// Identity of the encoder that produced a vector feature.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RepresentationFields")]
pub struct Representation {
    encoder: String,
    encoder_revision: String,
    dimension: usize,
    normalized: bool,
    query_template: String,
    document_template: String,
}

#[derive(Deserialize)]
struct RepresentationFields {
    encoder: String,
    encoder_revision: String,
    dimension: usize,
    normalized: bool,
    #[serde(default)]
    query_template: String,
    #[serde(default)]
    document_template: String,
}

impl TryFrom<RepresentationFields> for Representation {
    type Error = Error;

    fn try_from(fields: RepresentationFields) -> Result<Self> {
        Ok(Self::new(
            fields.encoder,
            fields.encoder_revision,
            fields.dimension,
            fields.normalized,
        )?
        .with_templates(fields.query_template, fields.document_template))
    }
}

impl Representation {
    /// A representation with empty templates.
    pub fn new(
        encoder: impl Into<String>,
        encoder_revision: impl Into<String>,
        dimension: usize,
        normalized: bool,
    ) -> Result<Self> {
        let representation = Self {
            encoder: encoder.into(),
            encoder_revision: encoder_revision.into(),
            dimension,
            normalized,
            query_template: String::new(),
            document_template: String::new(),
        };
        require_non_empty(
            &representation.encoder,
            "representation encoder must not be empty",
        )?;
        require_non_empty(
            &representation.encoder_revision,
            "representation encoder_revision must not be empty",
        )?;
        if dimension == 0 {
            return Err(Error::invalid("representation dimension must be positive"));
        }
        Ok(representation)
    }

    pub fn with_templates(
        mut self,
        query_template: impl Into<String>,
        document_template: impl Into<String>,
    ) -> Self {
        self.query_template = query_template.into();
        self.document_template = document_template.into();
        self
    }

    pub fn encoder(&self) -> &str {
        &self.encoder
    }

    pub fn encoder_revision(&self) -> &str {
        &self.encoder_revision
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn normalized(&self) -> bool {
        self.normalized
    }

    pub fn query_template(&self) -> &str {
        &self.query_template
    }

    pub fn document_template(&self) -> &str {
        &self.document_template
    }

    /// Fails with [`Error::IncompatibleQuery`] naming every differing field.
    pub fn assert_compatible(&self, other: &Self) -> Result<()> {
        let mut mismatches = Vec::new();
        mismatch(&mut mismatches, "encoder", &self.encoder, &other.encoder);
        mismatch(
            &mut mismatches,
            "encoder_revision",
            &self.encoder_revision,
            &other.encoder_revision,
        );
        mismatch(
            &mut mismatches,
            "dimension",
            &self.dimension,
            &other.dimension,
        );
        mismatch(
            &mut mismatches,
            "normalized",
            &self.normalized,
            &other.normalized,
        );
        mismatch(
            &mut mismatches,
            "query_template",
            &self.query_template,
            &other.query_template,
        );
        mismatch(
            &mut mismatches,
            "document_template",
            &self.document_template,
            &other.document_template,
        );
        if mismatches.is_empty() {
            return Ok(());
        }
        Err(Error::IncompatibleQuery(format!(
            "representations differ ({})",
            mismatches.join("; ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_mismatch_names_the_field() {
        let manifest = CorpusManifest::new("corpus", "1", 3, "abc").unwrap();
        manifest.assert_compatible(&manifest.clone()).unwrap();
        let error = manifest
            .assert_compatible(&manifest.clone().with_generation(1))
            .unwrap_err();
        assert!(matches!(error, Error::IncompatibleIndex(_)));
        assert!(error.to_string().contains("generation: 0 != 1"));
    }

    #[test]
    fn representation_mismatch_names_the_field() {
        let representation = Representation::new("encoder", "1", 8, true).unwrap();
        let other = Representation::new("encoder", "1", 4, true).unwrap();
        let error = representation.assert_compatible(&other).unwrap_err();
        assert!(matches!(error, Error::IncompatibleQuery(_)));
        assert!(error.to_string().contains("dimension: 8 != 4"));
    }

    #[test]
    fn empty_identity_is_rejected() {
        assert!(CorpusManifest::new("", "1", 3, "abc").is_err());
        assert!(Representation::new("encoder", "1", 0, true).is_err());
        assert!(serde_json::from_str::<Representation>(
            r#"{"encoder": "", "encoder_revision": "1", "dimension": 8, "normalized": true}"#
        )
        .is_err());
    }

    #[test]
    fn manifests_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("corpus.json");
        let manifest = CorpusManifest::new("corpus", "1", 3, "abc")
            .unwrap()
            .with_generation(2);
        manifest.write(&path).unwrap();
        assert_eq!(CorpusManifest::read(&path).unwrap(), manifest);

        let representation = Representation::new("encoder", "1", 8, true)
            .unwrap()
            .with_templates("[Q] ", "");
        let encoded = serde_json::to_string(&representation).unwrap();
        assert_eq!(
            serde_json::from_str::<Representation>(&encoded).unwrap(),
            representation
        );
    }

    #[test]
    fn digest_depends_on_order_and_boundaries() {
        assert_ne!(
            document_ids_digest(["a", "b"]),
            document_ids_digest(["b", "a"])
        );
        assert_ne!(
            document_ids_digest(["ab", "c"]),
            document_ids_digest(["a", "bc"])
        );
    }
}
