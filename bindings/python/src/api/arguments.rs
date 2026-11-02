use crate::{convert, requests};
use cofferdb::binding::{Argument, Input, Operation, SearchInput, ValueKind};
use cofferdb::{
    CollectionConfig, DistanceMetric, Error, ErrorCode, JsonValue, OnConflict, QueryEmbeddingInput,
    ResourceProfile, TokenizerConfig, VectorIndexConfig,
};
use convert::{Failure, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyString};

fn fields<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    allowed: &[&str],
) -> Result<&'a Bound<'py, PyDict>> {
    let object = convert::dict(value)?;
    for (key, _) in object.iter() {
        if !key.is_exact_instance_of::<PyString>() {
            return Err(Failure::Type(
                "configuration keys must be built-in str values",
            ));
        }
        let key = key.cast::<PyString>().map_err(PyErr::from)?;
        if !allowed.iter().any(|field| key == field) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "unsupported configuration field",
            )
            .into());
        }
    }
    Ok(object)
}

fn text(object: &Bound<'_, PyDict>, field: &'static str, input: &mut Input) -> Result<String> {
    convert::text(&requests::required(object, field)?, input)
}

pub(super) fn conflict(value: Option<&Bound<'_, PyAny>>, input: &mut Input) -> Result<OnConflict> {
    let Some(value) = value else {
        return Ok(OnConflict::Error);
    };
    let value = convert::text(value, input)?;
    match value.as_str() {
        "error" => Ok(OnConflict::Error),
        "skip" => Ok(OnConflict::Skip),
        "replace" => Ok(OnConflict::Replace),
        _ => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "unsupported conflict strategy",
        )
        .with_detail("field", "on_conflict")
        .into()),
    }
}

pub(super) fn collection(
    value: &Bound<'_, PyAny>,
    operation: &Operation,
) -> Result<Argument<CollectionConfig>> {
    let object = fields(
        value,
        &[
            "name",
            "embedding_dim",
            "distance_metric",
            "embedding_model",
            "vectors_normalized",
            "vector_index",
            "tokenizer",
        ],
    )?;
    let mut input = operation.input();
    let tokenizer = if let Some(value) = crate::output::lookup(object, "tokenizer")? {
        let mut tokenizer_input = operation.input();
        let value = convert::json(&value, &mut tokenizer_input, ValueKind::COLLECTION_CONFIG)?;
        operation.tokenizer(tokenizer_input.finish(value))?
    } else {
        TokenizerConfig::default_en()
    };
    let name = text(object, "name", &mut input)?;
    let embedding_dim = requests::native_integer(
        &requests::required(object, "embedding_dim")?,
        "embedding_dim",
    )?;
    let metric = text(object, "distance_metric", &mut input)?;
    let distance_metric = match metric.as_str() {
        "cosine" | "dot_product" | "euclidean" => DistanceMetric::parse(&metric)?,
        _ => {
            return Err(
                Error::new(ErrorCode::UnsupportedMetric, "unsupported distance metric").into(),
            )
        }
    };
    let embedding_model = crate::output::lookup(object, "embedding_model")?
        .filter(|value| !value.is_none())
        .map(|value| convert::text(&value, &mut input))
        .transpose()?;
    let vectors_normalized = crate::output::lookup(object, "vectors_normalized")?
        .map(|value| convert::boolean(&value))
        .transpose()?
        .unwrap_or(false);
    if let Some(value) = crate::output::lookup(object, "vector_index")? {
        if convert::text(&value, &mut input)? != "exact" {
            return Err(Error::new(
                ErrorCode::UnsupportedFeatureFlag,
                "only exact vector indexing is supported",
            )
            .with_detail("field", "vector_index")
            .into());
        }
    }
    Ok(input.finish(CollectionConfig {
        tokenizer,
        name,
        embedding_dim,
        distance_metric,
        embedding_model,
        vectors_normalized,
        vector_index: VectorIndexConfig::Exact,
    }))
}

pub(super) fn top_k(value: &Bound<'_, PyAny>) -> Result<usize> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(Failure::Type(
            "top_k requires a built-in int; bool and float are not accepted",
        ));
    }
    let value = match value.extract::<u64>() {
        Ok(value) => value,
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyOverflowError>(value.py()) => {
            return Err(Error::new(
                ErrorCode::InvalidSearchParams,
                "top_k is outside the supported exact integer range",
            )
            .with_detail("field", "top_k")
            .into())
        }
        Err(error) => return Err(Failure::Python(error)),
    };
    // Only reject values the Rust parameter cannot represent here. The shared
    // engine validates its supported range and preserves its complete error.
    if value > usize::MAX as u64 {
        return Err(Error::new(
            ErrorCode::InvalidSearchParams,
            "top_k is outside the supported exact integer range",
        )
        .with_detail("field", "top_k")
        .into());
    }
    Ok(value as usize)
}

pub(super) struct Query<'a, 'py> {
    pub text: Option<&'a Bound<'py, PyAny>>,
    pub embedding: Option<&'a Bound<'py, PyAny>>,
    pub filter: Option<&'a Bound<'py, PyAny>>,
    pub weights: Option<&'a Bound<'py, PyAny>>,
    pub metadata_boost: Option<&'a Bound<'py, PyAny>>,
    pub top_k: Option<&'a Bound<'py, PyAny>>,
    pub profile: Option<&'a Bound<'py, PyAny>>,
}

pub(super) fn query(value: Query<'_, '_>, operation: &Operation) -> Result<Argument<SearchInput>> {
    let top_k = value.top_k.map(top_k).transpose()?.unwrap_or(10);
    let mut input = operation.input();
    let text = value
        .text
        .filter(|value| !value.is_none())
        .map(|value| convert::text(value, &mut input))
        .transpose()?;
    let embedding = value
        .embedding
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            let object = fields(value, &["vector", "dim", "model_name"])?;
            Ok(QueryEmbeddingInput {
                vector: convert::vector(
                    &requests::required(object, "vector")?,
                    &mut input,
                    ValueKind::QUERY_EMBEDDING,
                )?,
                dim: requests::native_integer(&requests::required(object, "dim")?, "dim")?,
                model_name: crate::output::lookup(object, "model_name")?
                    .filter(|value| !value.is_none())
                    .map(|value| convert::text(&value, &mut input))
                    .transpose()?,
            })
        })
        .transpose()?;
    let filter = value
        .filter
        .filter(|value| !value.is_none())
        .map(|value| convert::json(value, &mut input, ValueKind::FILTER))
        .transpose()?
        .unwrap_or_else(JsonValue::object);
    let weights = value
        .weights
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            let mut weights_input = operation.input();
            let value = convert::json(value, &mut weights_input, ValueKind::WEIGHTS)?;
            Ok(operation.weights(weights_input.finish(value))?)
        })
        .transpose()?;
    let metadata_boost = value
        .metadata_boost
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            let mut boost_input = operation.input();
            let value = convert::json(value, &mut boost_input, ValueKind::METADATA_BOOST)?;
            Ok(boost_input.finish(value))
        })
        .transpose()?;
    let profile = value
        .profile
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            let text = convert::text(value, &mut input)?;
            ResourceProfile::parse(&text).ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidSearchParams,
                    "unsupported resource profile",
                )
                .with_detail("field", "profile")
                .into()
            })
        })
        .transpose()?;
    Ok(input.finish(SearchInput {
        text,
        embedding,
        filter,
        weights,
        metadata_boost,
        top_k,
        profile,
    }))
}
