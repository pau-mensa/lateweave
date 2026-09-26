//! Queries: raw text plus named features, each materialized at most once.

use std::any::{type_name, Any};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::error::{Error, Result};
use crate::manifest::Representation;

/// Whatever an encoder produced for the query text: a token matrix, a dense
/// vector, a sparse weighting. Stages downcast to the type they consume.
pub trait FeatureValue: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// The value as a `[tokens, dimension]` matrix, for multi-vector stages.
    ///
    /// A value that holds its data in a foreign layout (a binding's array
    /// object, for example) overrides this instead of forcing every producer
    /// to convert eagerly.
    fn token_matrix(&self) -> Option<&TokenMatrix> {
        None
    }
}

impl dyn FeatureValue {
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }
}

/// A row-major float32 `[tokens, dimension]` matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenMatrix {
    values: Vec<f32>,
    dimension: usize,
}

impl TokenMatrix {
    pub fn new(values: Vec<f32>, dimension: usize) -> Result<Self> {
        if dimension == 0 || values.is_empty() {
            return Err(Error::invalid(
                "token matrix must have shape [tokens, dimension] with non-zero axes",
            ));
        }
        if values.len() % dimension != 0 {
            return Err(Error::invalid(format!(
                "{} values do not form rows of dimension {dimension}",
                values.len()
            )));
        }
        Ok(Self { values, dimension })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn tokens(&self) -> usize {
        self.values.len() / self.dimension
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl FeatureValue for TokenMatrix {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn token_matrix(&self) -> Option<&TokenMatrix> {
        Some(self)
    }
}

type Provider = Box<dyn Fn() -> Result<Arc<dyn FeatureValue>> + Send + Sync>;

/// One query representation stamped with the encoder that produced it.
///
/// A lazy feature runs its provider the first time a stage asks, so a
/// text-only gatherer never pays for encoding and every stage shares one
/// value. A provider that fails is kept and runs again on the next request.
pub struct Feature {
    representation: Representation,
    value: OnceLock<Arc<dyn FeatureValue>>,
    provider: Mutex<Option<Provider>>,
}

impl Feature {
    pub fn new(representation: Representation, value: impl FeatureValue) -> Self {
        Self::from_shared(representation, Arc::new(value))
    }

    pub fn from_shared(representation: Representation, value: Arc<dyn FeatureValue>) -> Self {
        Self {
            representation,
            value: OnceLock::from(value),
            provider: Mutex::new(None),
        }
    }

    pub fn lazy<V, F>(representation: Representation, provider: F) -> Self
    where
        V: FeatureValue,
        F: Fn() -> Result<V> + Send + Sync + 'static,
    {
        Self {
            representation,
            value: OnceLock::new(),
            provider: Mutex::new(Some(Box::new(move || {
                provider().map(|value| Arc::new(value) as Arc<dyn FeatureValue>)
            }))),
        }
    }

    pub fn representation(&self) -> &Representation {
        &self.representation
    }

    pub fn is_materialized(&self) -> bool {
        self.value.get().is_some()
    }

    pub fn value(&self) -> Result<&dyn FeatureValue> {
        if let Some(value) = self.value.get() {
            return Ok(value.as_ref());
        }
        let mut provider = self.provider.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(value) = self.value.get() {
            return Ok(value.as_ref());
        }
        let produce = provider
            .as_ref()
            .expect("an unmaterialized feature keeps its provider");
        let value = produce()?;
        let value = self.value.get_or_init(|| value);
        *provider = None;
        Ok(value.as_ref())
    }
}

impl fmt::Debug for Feature {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Feature")
            .field("representation", &self.representation)
            .field("materialized", &self.is_materialized())
            .finish()
    }
}

/// Raw text plus the named features stages may consume.
///
/// Feature names are plain strings agreed between a query's producer and the
/// stages that consume it. Cloning is cheap and shares materialized features.
#[derive(Clone, Debug)]
pub struct Query {
    text: Arc<str>,
    features: Arc<BTreeMap<String, Arc<Feature>>>,
}

impl Query {
    pub fn new(text: impl Into<Arc<str>>) -> Self {
        Self {
            text: text.into(),
            features: Arc::default(),
        }
    }

    pub fn with_feature(
        mut self,
        name: impl Into<String>,
        feature: impl Into<Arc<Feature>>,
    ) -> Self {
        Arc::make_mut(&mut self.features).insert(name.into(), feature.into());
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn features(&self) -> &BTreeMap<String, Arc<Feature>> {
        &self.features
    }

    /// The value of feature `name`, which must come from `representation`.
    ///
    /// The representation is checked before the feature is materialized, so a
    /// stage built for another encoder never triggers an encoding.
    pub fn feature(
        &self,
        name: &str,
        representation: &Representation,
    ) -> Result<&dyn FeatureValue> {
        let feature = self.features.get(name).ok_or_else(|| {
            let available = self
                .features
                .keys()
                .map(|name| format!("'{name}'"))
                .collect::<Vec<_>>()
                .join(", ");
            Error::IncompatibleQuery(format!(
                "query has no '{name}' feature; available: [{available}]"
            ))
        })?;
        representation.assert_compatible(&feature.representation)?;
        feature.value()
    }

    pub fn feature_as<T: Any>(&self, name: &str, representation: &Representation) -> Result<&T> {
        self.feature(name, representation)?
            .downcast_ref::<T>()
            .ok_or_else(|| {
                Error::IncompatibleQuery(format!(
                    "query feature '{name}' is not a {}",
                    type_name::<T>()
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn representation(encoder: &str) -> Representation {
        Representation::new(encoder, "1", 2, true).unwrap()
    }

    #[test]
    fn provider_runs_once_and_only_for_a_compatible_stage() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let query = Query::new("query").with_feature(
            "multi_vector",
            Feature::lazy(representation("encoder"), move || {
                counter.fetch_add(1, Ordering::SeqCst);
                TokenMatrix::new(vec![1.0, 0.0], 2)
            }),
        );

        let error = query
            .feature("multi_vector", &representation("other"))
            .err()
            .unwrap();
        assert!(error.to_string().contains("encoder"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        for _ in 0..2 {
            let matrix = query
                .feature_as::<TokenMatrix>("multi_vector", &representation("encoder"))
                .unwrap();
            assert_eq!(matrix.tokens(), 1);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_failed_provider_runs_again() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let feature = Feature::lazy(representation("encoder"), move || {
            if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(Error::invalid("transient"));
            }
            TokenMatrix::new(vec![1.0, 0.0], 2)
        });
        assert!(feature.value().is_err());
        assert!(feature.value().is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_missing_feature_names_what_is_available() {
        let query = Query::new("query").with_feature(
            "dense",
            Feature::new(
                representation("encoder"),
                TokenMatrix::new(vec![1.0, 0.0], 2).unwrap(),
            ),
        );
        let error = query
            .feature("multi_vector", &representation("encoder"))
            .err()
            .unwrap();
        assert!(matches!(error, Error::IncompatibleQuery(_)));
        assert_eq!(
            error.to_string(),
            "query has no 'multi_vector' feature; available: ['dense']"
        );
    }
}
