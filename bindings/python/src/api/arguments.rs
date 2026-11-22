use crate::{convert, requests};
use cofferdb::binding::{Argument, Input, Operation, SearchInput, ValueKind};
use cofferdb::{
    CollectionConfig, DistanceMetric, Error, ErrorCode, JsonValue, OnConflict, QueryEmbeddingInput,
    ResourceProfile, TokenizerConfig, VectorIndexConfig,
};
use convert::{Failure, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyString};

// Use a caller-supplied field list for each fixed configuration shape. Collection
// settings and the query-embedding envelope have different keys even when both are
// ordinary Python dicts.
fn fields<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    allowed: &[&str],
) -> Result<&'a Bound<'py, PyDict>> {
    // Validate the exact dictionary and its keys before reading individual
    // settings. This keeps unknown input from being silently discarded when the
    // typed Rust request is assembled.
    let object = convert::dict(value)?;
    for (key, _) in object.iter() {
        // Restrict keys to exact built-in strings before equality checks, so
        // deciding whether a field exists cannot call a user-defined key's
        // comparison.
        if !key.is_exact_instance_of::<PyString>() {
            return Err(Failure::Type(
                "configuration keys must be built-in str values",
            ));
        }
        let key = key.cast::<PyString>().map_err(PyErr::from)?;
        // A string that lies outside the schema is a native InvalidJsonlSchema
        // error. A key with the wrong Python type fails the host type check above.
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
    // Look up the required key before decoding its text. Missing fields carry the
    // native field identifier, while a present value of the wrong type follows the
    // Python conversion path.
    convert::text(&requests::required(object, field)?, input)
}

pub(super) fn conflict(value: Option<&Bound<'_, PyAny>>, input: &mut Input) -> Result<OnConflict> {
    // Only an omitted argument selects the default conflict policy. An explicitly
    // supplied None is still an object and must pass text conversion.
    let Some(value) = value else {
        return Ok(OnConflict::Error);
    };
    let value = convert::text(value, input)?;
    // Match exact policy spellings. Writers and importers receive the same enum,
    // so error, skip and replace have the same dispatch independent of the Python
    // entry point.
    match value.as_str() {
        "error" => Ok(OnConflict::Error),
        "skip" => Ok(OnConflict::Skip),
        "replace" => Ok(OnConflict::Replace),
        // Include the public argument name on a rejected spelling. The engine can
        // expose a structured field error even though this check happened before
        // submitting the write.
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
    // Validate the collection's fixed field names before decoding its nested
    // tokenizer or retaining any input strings. This rejects a misspelled option
    // even when all required options are otherwise present.
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
        // Build the tokenizer JSON with its own guard on the same operation.
        // Converting it to the typed configuration consumes that temporary tree
        // while leaving the outer input builder available for the collection name
        // and model.
        let mut tokenizer_input = operation.input();
        let value = convert::json(&value, &mut tokenizer_input, ValueKind::COLLECTION_CONFIG)?;
        operation.tokenizer(tokenizer_input.finish(value))?
    } else {
        // Omitting the tokenizer selects the engine's default directly. A value
        // present in the mapping, including None, goes through tokenizer decoding
        // and must have the expected schema.
        TokenizerConfig::default_en()
    };
    let name = text(object, "name", &mut input)?;
    // Extract an exact native integer before constructing the collection request.
    // The core owns the positive u32 dimension rule, so a representable zero or
    // oversized width can still be rejected there.
    let embedding_dim = requests::native_integer(
        &requests::required(object, "embedding_dim")?,
        "embedding_dim",
    )?;
    let metric = text(object, "distance_metric", &mut input)?;
    // The Python collection API accepts the three canonical metric spellings. The
    // native parser also accepts aliases for other callers, so delegate only after
    // this public spelling check.
    let distance_metric = match metric.as_str() {
        "cosine" | "dot_product" | "euclidean" => DistanceMetric::parse(&metric)?,
        _ => {
            return Err(
                Error::new(ErrorCode::UnsupportedMetric, "unsupported distance metric").into(),
            )
        }
    };
    // Omission and Python None both leave the model unbound in the prepared
    // configuration. A supplied string is preserved for the core's later nonblank
    // and model-identity checks.
    let embedding_model = crate::output::lookup(object, "embedding_model")?
        .filter(|value| !value.is_none())
        .map(|value| convert::text(&value, &mut input))
        .transpose()?;
    // An omitted normalization flag gets false. Unlike the optional model, an
    // explicit None is an invalid flag, and a truthy integer is not a built-in
    // bool.
    let vectors_normalized = crate::output::lookup(object, "vectors_normalized")?
        .map(|value| convert::boolean(&value))
        .transpose()?
        .unwrap_or(false);
    // The Python create request accepts the literal string exact for this setting.
    // The tagged index object used in stored collection metadata has a separate
    // representation.
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
    // Move the final configuration together with the builder's reservation. The
    // detached collection-creation call checks that this argument belongs to the
    // operation that prepared it.
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
    // Accept only a built-in int. A bool or an integral float can represent the
    // same mathematical value but does not satisfy the Python calling contract for
    // the result limit.
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(Failure::Type(
            "top_k requires a built-in int; bool and float are not accepted",
        ));
    }
    // Convert before applying the engine's search policy. The native adapter
    // cannot receive a negative integer or an arbitrary-precision Python integer
    // beyond u64, so report these with the top_k field immediately.
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

// The query groups attached Python references only during argument preparation.
// Its converter below produces owned Rust values and operation reservations before
// releasing the interpreter for search.
pub(super) struct Query<'a, 'py> {
    // Borrowed Python lexical input during conversion. Missing or None leaves the
    // native text absent.
    pub text: Option<&'a Bound<'py, PyAny>>,
    // Borrowed embedding dictionary, including its vector, declared dimension and
    // optional model. Convert while attached before entering the engine.
    pub embedding: Option<&'a Bound<'py, PyAny>>,
    // An omitted or None predicate becomes the empty native filter object.
    pub filter: Option<&'a Bound<'py, PyAny>>,
    // Temporary Python weights object converted to validated numeric weights with a
    // separate input guard.
    pub weights: Option<&'a Bound<'py, PyAny>>,
    // Raw rule dictionary converted with its own reservation so later compilation
    // can discard the raw tree independently.
    pub metadata_boost: Option<&'a Bound<'py, PyAny>>,
    // Omission selects ten. An explicitly supplied value goes through the integer
    // checks, including an explicit None.
    pub top_k: Option<&'a Bound<'py, PyAny>>,
    // Optional resource preset name parsed while attached. Absence defers the
    // choice to the database.
    pub profile: Option<&'a Bound<'py, PyAny>>,
}

// Build the owned native search request while attached to Python. Group long-lived
// arguments under one operation and keep independently converted inputs in their
// own guards.
pub(super) fn query(value: Query<'_, '_>, operation: &Operation) -> Result<Argument<SearchInput>> {
    // Check the Python integer type and native size before admitting potentially
    // large arguments. The default applies only to omission, so explicit None
    // reaches the converter.
    let top_k = value.top_k.map(top_k).transpose()?.unwrap_or(10);
    let mut input = operation.input();
    // An absent or None text argument leaves the text channel unspecified. Copy a
    // supplied string as it arrived and leave normalization to core search.
    let text = value
        .text
        .filter(|value| !value.is_none())
        .map(|value| convert::text(value, &mut input))
        .transpose()?;
    // Treat absent or None embedding like an unused vector input. A present value
    // must have the fixed query-embedding field shape before coordinate
    // conversion.
    let embedding = value
        .embedding
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            // Accept vector, declared width and optional model as separate fields.
            // Collection validation later decides whether these inputs describe a
            // compatible query.
            let object = fields(value, &["vector", "dim", "model_name"])?;
            Ok(QueryEmbeddingInput {
                // Allocate the typed vector under the query embedding kind. The
                // core checks its own vector limits and compares dimension against
                // the selected collection.
                vector: convert::vector(
                    &requests::required(object, "vector")?,
                    &mut input,
                    ValueKind::QUERY_EMBEDDING,
                )?,
                // Read the dimension as an exact machine integer. It is a
                // declaration to validate, so it does not control the size of the
                // vector allocation above.
                dim: requests::native_integer(&requests::required(object, "dim")?, "dim")?,
                // Keep a model as an optional owned string. None or omission
                // leaves it unspecified rather than selecting a named default.
                model_name: crate::output::lookup(object, "model_name")?
                    .filter(|value| !value.is_none())
                    .map(|value| convert::text(&value, &mut input))
                    .transpose()?,
            })
        })
        .transpose()?;
    // Carry the converted filter for core validation and candidate eligibility.
    // Omission or None creates the empty object used for an unfiltered request.
    let filter = value
        .filter
        .filter(|value| !value.is_none())
        .map(|value| convert::json(value, &mut input, ValueKind::FILTER))
        .transpose()?
        .unwrap_or_else(JsonValue::object);
    // Absent weights defer score weights to the engine. Supplied weights use a
    // separate JSON reservation while becoming the typed coefficient value.
    let weights = value
        .weights
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            // Weights are converted from a JSON value to the engine's typed weights
            // here. Give that temporary value its own input guard so consuming it
            // leaves the other query arguments and their credit intact.
            let mut weights_input = operation.input();
            let value = convert::json(value, &mut weights_input, ValueKind::WEIGHTS)?;
            // Consume the temporary weight argument immediately. The resulting
            // coefficients have no owned JSON tree to keep through native search.
            Ok(operation.weights(weights_input.finish(value))?)
        })
        .transpose()?;
    // Unlike the numeric weights, boost rules are parsed by the native search.
    // Retain the raw tree as a separately guarded argument so compiled rules and
    // unrelated query inputs can have different lifetimes.
    // Store the prepared boost argument for the native adapter. The adapter checks
    // the argument's operation identity before compiling rules under that context.
    let metadata_boost = value
        .metadata_boost
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            let mut boost_input = operation.input();
            let value = convert::json(value, &mut boost_input, ValueKind::METADATA_BOOST)?;
            Ok(boost_input.finish(value))
        })
        .transpose()?;
    // An unspecified profile leaves candidate and cache policy to the database. A
    // string must select a recognized preset before the request can execute.
    let profile = value
        .profile
        .filter(|value| !value.is_none())
        .map(|value| -> Result<_> {
            // Apply the product profile parser to the exact admitted string.
            // Preserve a field-level error rather than treating an unrecognized
            // spelling as the default.
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
    // Transfer the request and its accounting together. Any nested boost argument
    // retains the separate lease needed until the core parses its rules.
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
