//! Identity of the encoder behind a vector feature, checked before any stage
//! runs: a stage that consumes a feature must agree on it with the query that
//! supplies it.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

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
    fn representation_mismatch_names_the_field() {
        let representation = Representation::new("encoder", "1", 8, true).unwrap();
        let other = Representation::new("encoder", "1", 4, true).unwrap();
        let error = representation.assert_compatible(&other).unwrap_err();
        assert!(matches!(error, Error::IncompatibleQuery(_)));
        assert!(error.to_string().contains("dimension: 8 != 4"));
    }

    #[test]
    fn empty_identity_is_rejected() {
        assert!(Representation::new("", "1", 8, true).is_err());
        assert!(Representation::new("encoder", "1", 0, true).is_err());
        assert!(serde_json::from_str::<Representation>(
            r#"{"encoder": "", "encoder_revision": "1", "dimension": 8, "normalized": true}"#
        )
        .is_err());
    }

    #[test]
    fn representations_round_trip() {
        let representation = Representation::new("encoder", "1", 8, true)
            .unwrap()
            .with_templates("[Q] ", "");
        let encoded = serde_json::to_string(&representation).unwrap();
        assert_eq!(
            serde_json::from_str::<Representation>(&encoded).unwrap(),
            representation
        );
    }
}
