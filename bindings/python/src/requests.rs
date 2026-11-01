//! Shape checks and direct owned inputs for the shared native batch path.

use crate::convert::{self, Failure, Result};
use cofferdb::binding::{Batch, ChunkRecord, Input, Operation, ValueKind};
use cofferdb::{
    ChunkInput, DocumentInput, EmbeddingInput, Error, ErrorCode, JsonValue, SourceSpan,
};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyString};

pub(crate) fn object<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    fields: &[&str],
) -> Result<&'a Bound<'py, PyDict>> {
    let object = convert::dict(value)?;
    for (key, _) in object.iter() {
        if !key.is_exact_instance_of::<PyString>() {
            return Err(Failure::Type("record keys must be built-in str values"));
        }
        let key = key.cast::<PyString>().map_err(PyErr::from)?;
        if !fields.iter().any(|name| key == name) {
            return Err(
                Error::new(ErrorCode::InvalidJsonlSchema, "unsupported input field")
                    .with_detail("json_input_kind", "jsonl_record")
                    .into(),
            );
        }
    }
    Ok(object)
}

pub(crate) fn required<'py>(
    object: &Bound<'py, PyDict>,
    name: &'static str,
) -> Result<Bound<'py, PyAny>> {
    crate::output::lookup(object, name)?.ok_or_else(|| {
        Error::new(ErrorCode::MissingRequiredField, "required field is missing")
            .with_detail("field", name)
            .into()
    })
}

fn required_text(
    object: &Bound<'_, PyDict>,
    name: &'static str,
    input: &mut Input,
) -> Result<String> {
    convert::text(&required(object, name)?, input)
}

fn optional_text(
    object: &Bound<'_, PyDict>,
    name: &str,
    input: &mut Input,
) -> Result<Option<String>> {
    crate::output::lookup(object, name)?
        .filter(|value| !value.is_none())
        .map(|value| convert::text(&value, input))
        .transpose()
}

fn metadata(object: &Bound<'_, PyDict>, input: &mut Input, kind: ValueKind) -> Result<JsonValue> {
    match crate::output::lookup(object, "metadata")? {
        Some(value) => convert::json(&value, input, kind),
        None => Ok(JsonValue::Object(input.object(0, kind)?)),
    }
}

/// Keep integers exact before the core validates its narrower business range.
pub(crate) fn native_integer(value: &Bound<'_, PyAny>, field: &'static str) -> Result<usize> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(Failure::Type(
            "expected a built-in int; bool and float are not accepted",
        ));
    }
    value.extract::<usize>().map_err(|error| {
        if error.is_instance_of::<PyOverflowError>(value.py()) {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "integer is outside the native field range",
            )
            .with_detail("field", field)
            .into()
        } else {
            Failure::Python(error)
        }
    })
}

fn span(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<Option<SourceSpan>> {
    if value.is_none() {
        return Ok(None);
    }
    let object = object(value, &["start", "end", "line_start", "line_end"])?;
    let mut offset = |name: &'static str| -> Result<Option<u64>> {
        let Some(value) = crate::output::lookup(object, name)?.filter(|value| !value.is_none())
        else {
            return Ok(None);
        };
        let number = if value.is_exact_instance_of::<PyInt>() {
            Some(value.extract::<u64>().map_err(|error| {
                if error.is_instance_of::<PyOverflowError>(value.py()) {
                    Error::new(ErrorCode::InvalidJsonlSchema, "invalid source byte offset")
                        .with_detail("field", name)
                        .into()
                } else {
                    Failure::Python(error)
                }
            })?)
        } else if value.is_exact_instance_of::<PyString>() {
            let text = convert::text(&value, input)?;
            if !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()) {
                text.parse::<u64>().ok()
            } else {
                None
            }
        } else {
            return Err(Failure::Type(
                "source byte offsets require built-in int or decimal str",
            ));
        };
        number.map(Some).ok_or_else(|| {
            Error::new(ErrorCode::InvalidJsonlSchema, "invalid source byte offset")
                .with_detail("field", name)
                .into()
        })
    };
    let start = offset("start")?;
    let end = offset("end")?;
    let line = |name: &'static str| -> Result<Option<u32>> {
        let Some(value) = crate::output::lookup(object, name)?.filter(|value| !value.is_none())
        else {
            return Ok(None);
        };
        let number = native_integer(&value, name)?;
        u32::try_from(number).map(Some).map_err(|_| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "source line is outside u32 range",
            )
            .with_detail("field", name)
            .into()
        })
    };
    let span = SourceSpan {
        start,
        end,
        line_start: line("line_start")?,
        line_end: line("line_end")?,
    };
    span.validate()?;
    Ok(Some(span))
}

fn document(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<DocumentInput> {
    let object = object(value, &["external_id", "title", "source_uri", "metadata"])?;
    Ok(DocumentInput {
        external_id: required_text(object, "external_id", input)?,
        title: optional_text(object, "title", input)?,
        source_uri: optional_text(object, "source_uri", input)?,
        metadata: metadata(object, input, ValueKind::DOCUMENT_METADATA)?,
        chunks: Vec::new(),
    })
}

fn chunk(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<ChunkRecord> {
    let object = object(
        value,
        &[
            "document_external_id",
            "chunk_external_id",
            "ordinal",
            "text",
            "metadata",
            "source_span",
        ],
    )?;
    let source_span = crate::output::lookup(object, "source_span")?;
    Ok(ChunkRecord {
        document_external_id: required_text(object, "document_external_id", input)?,
        chunk: ChunkInput {
            chunk_external_id: required_text(object, "chunk_external_id", input)?,
            ordinal: native_integer(&required(object, "ordinal")?, "ordinal")?,
            text: required_text(object, "text", input)?,
            metadata: metadata(object, input, ValueKind::CHUNK_METADATA)?,
            source_span: source_span
                .as_ref()
                .map(|value| span(value, input))
                .transpose()?
                .flatten(),
        },
        source_span_provided: source_span.is_some(),
    })
}

fn embedding(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<EmbeddingInput> {
    let object = object(value, &["chunk_external_id", "model_name", "dim", "vector"])?;
    Ok(EmbeddingInput {
        chunk_external_id: required_text(object, "chunk_external_id", input)?,
        model_name: required_text(object, "model_name", input)?,
        dim: native_integer(&required(object, "dim")?, "dim")?,
        vector: convert::vector(&required(object, "vector")?, input, ValueKind::JSONL_RECORD)?,
    })
}

pub(crate) fn batch(
    operation: &Operation,
    documents: Option<&Bound<'_, PyAny>>,
    chunks: Option<&Bound<'_, PyAny>>,
    embeddings: Option<&Bound<'_, PyAny>>,
) -> Result<Batch> {
    let mut batch = operation.batch()?;
    for (kind, value) in [documents, chunks, embeddings].into_iter().enumerate() {
        let Some(value) = value else {
            continue;
        };
        let sequence = convert::Sequence::new(value)?;
        for index in 0..sequence.len() {
            let item = sequence.item(index)?;
            let mut input = operation.input();
            match kind {
                0 => {
                    let value = document(&item, &mut input)?;
                    batch.push_document(input.finish(value))?;
                }
                1 => {
                    let value = chunk(&item, &mut input)?;
                    batch.push_chunk(input.finish(value))?;
                }
                2 => {
                    let value = embedding(&item, &mut input)?;
                    batch.push_embedding(input.finish(value))?;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(batch)
}
