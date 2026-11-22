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
    // Check the entire set of field names before extracting any values. An
    // unexpected key is rejected even when all the required fields are present and
    // valid.
    let object = convert::dict(value)?;
    for (key, _) in object.iter() {
        // Require the built-in string type before comparing names. A str subclass
        // could supply its own equality during what should only be a schema
        // lookup.
        if !key.is_exact_instance_of::<PyString>() {
            return Err(Failure::Type("record keys must be built-in str values"));
        }
        let key = key.cast::<PyString>().map_err(PyErr::from)?;
        // A well-formed string outside the allowed field list is reported through
        // the native record schema error. A non-string key reaches the Python
        // type-error path above.
        if !fields.iter().any(|name| key == name) {
            return Err(
                Error::new(ErrorCode::InvalidJsonlSchema, "unsupported input field")
                    .with_detail("json_input_kind", "jsonl_record")
                    .into(),
            );
        }
    }
    // The caller receives the original dict under the same attached borrow, so
    // field extraction does not need a copied mapping. Values are converted into
    // owned inputs before the native batch is submitted.
    Ok(object)
}

pub(crate) fn required<'py>(
    object: &Bound<'py, PyDict>,
    name: &'static str,
) -> Result<Bound<'py, PyAny>> {
    // Only a successful dictionary lookup can establish that a required field is
    // absent. Propagate a lookup or key-allocation failure before constructing the
    // missing-field diagnostic.
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
    // Presence and content are separate checks. A present None reaches text
    // conversion and fails its type check instead of being diagnosed as a missing
    // key.
    convert::text(&required(object, name)?, input)
}

fn optional_text(
    object: &Bound<'_, PyDict>,
    name: &str,
    input: &mut Input,
) -> Result<Option<String>> {
    // Optional text treats an omitted key and a present None as the same absence.
    // A present empty string still reaches the shared text converter.
    crate::output::lookup(object, name)?
        .filter(|value| !value.is_none())
        .map(|value| convert::text(&value, input))
        .transpose()
}

fn metadata(object: &Bound<'_, PyDict>, input: &mut Input, kind: ValueKind) -> Result<JsonValue> {
    // Only omission supplies the empty object. A present Python None is converted
    // to JSON null, leaving the core to validate the actual metadata value
    // supplied.
    match crate::output::lookup(object, "metadata")? {
        Some(value) => convert::json(&value, input, kind),
        // An omitted metadata object has no nodes to reserve. Construct the empty
        // object through the same value kind and operation as explicit metadata.
        None => Ok(JsonValue::Object(input.object(0, kind)?)),
    }
}

/// Keep integers exact before the core validates its narrower business range.
pub(crate) fn native_integer(value: &Bound<'_, PyAny>, field: &'static str) -> Result<usize> {
    // Check the exact built-in type before extraction. Although bool is a Python
    // int subclass, accepting True here would silently turn an option flag into
    // ordinal or dimension 1.
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(Failure::Type(
            "expected a built-in int; bool and float are not accepted",
        ));
    }
    value.extract::<usize>().map_err(|error| {
        // Negative integers and positive integers too large for usize both cause
        // OverflowError here. Map that failure to the field-specific schema error
        // while carrying other Python exceptions unchanged.
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

// Byte offsets can use the full u64 range. Decode the Python integer or
// its unsigned decimal spelling directly. A round trip through a generic
// JSON number could lose precision before the core checks the bounds.
fn span(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<Option<SourceSpan>> {
    // An explicit Python None yields no coordinates. Its caller separately
    // remembers that the source_span key was present, which determines whether
    // metadata may supply a fallback.
    if value.is_none() {
        return Ok(None);
    }
    let object = object(value, &["start", "end", "line_start", "line_end"])?;
    let mut offset = |name: &'static str| -> Result<Option<u64>> {
        // An absent or null endpoint means that coordinate was not provided. Defer
        // deciding whether its partner is required until both endpoints and both
        // line fields have been decoded.
        let Some(value) = crate::output::lookup(object, name)?.filter(|value| !value.is_none())
        else {
            return Ok(None);
        };
        // Require an exact built-in int before u64 extraction. Negative or
        // excessively large values produce a field-specific schema error; an
        // unrelated Python exception remains the original exception.
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
            // Use the admitted text as an unsigned decimal spelling. Restrict it
            // to ASCII digits before parsing so whitespace, a sign and Unicode
            // decimal characters cannot quietly change the accepted offset
            // grammar.
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
        // A malformed or out-of-range decimal string belongs to the right Python
        // type but fails the native offset schema. This is separate from the type
        // error for an unsupported object.
        number.map(Some).ok_or_else(|| {
            Error::new(ErrorCode::InvalidJsonlSchema, "invalid source byte offset")
                .with_detail("field", name)
                .into()
        })
    };
    let start = offset("start")?;
    let end = offset("end")?;
    // Line numbers accept only built-in integers and must fit u32.
    let line = |name: &'static str| -> Result<Option<u32>> {
        let Some(value) = crate::output::lookup(object, name)?.filter(|value| !value.is_none())
        else {
            return Ok(None);
        };
        // Line numbers use the integer path without the decimal-string form
        // available to byte offsets. Narrow them to their stored width here; the
        // pair validator below enforces the one-based minimum.
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
    // Validate the relationships after decoding the four optional fields. An empty
    // dict is a provided empty span, while a lone or reversed endpoint is an error
    // and must not be filled from metadata.
    span.validate()?;
    Ok(Some(span))
}

fn document(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<DocumentInput> {
    // A document row carries only document fields. Chunks in the host batch are
    // independent rows with a parent external ID; accepting a nested chunks list
    // here would create a second submission path.
    let object = object(value, &["external_id", "title", "source_uri", "metadata"])?;
    Ok(DocumentInput {
        external_id: required_text(object, "external_id", input)?,
        title: optional_text(object, "title", input)?,
        source_uri: optional_text(object, "source_uri", input)?,
        // Use the document-metadata policy for this business value. Other document
        // text fields are ordinary owned strings, with separate workspace
        // accounting under the same input.
        metadata: metadata(object, input, ValueKind::DOCUMENT_METADATA)?,
        chunks: Vec::new(),
    })
}

// Convert a batch-level chunk into its document reference and owned chunk payload.
// Check the fixed record keys before admitting nested business values.
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
    // Keep the original lookup result alive so it records whether the caller
    // supplied the field. The decoded span can be absent for both omission and
    // explicit None.
    let source_span = crate::output::lookup(object, "source_span")?;
    Ok(ChunkRecord {
        // The outer document key binds this chunk to its parent in the batch or
        // collection. The chunk's own external ID remains inside its payload.
        document_external_id: required_text(object, "document_external_id", input)?,
        chunk: ChunkInput {
            chunk_external_id: required_text(object, "chunk_external_id", input)?,
            // Convert ordinal as an exact native integer. Later core validation
            // checks collisions within the selected parent.
            ordinal: native_integer(&required(object, "ordinal")?, "ordinal")?,
            text: required_text(object, "text", input)?,
            // Admit this arbitrary JSON value under the chunk metadata kind.
            // Structured limit errors identify that kind, while allocation shares
            // the operation's memory budget.
            metadata: metadata(object, input, ValueKind::CHUNK_METADATA)?,
            // Decode a supplied source-span field with the dedicated coordinate
            // parser. Flatten only after retaining the original presence for
            // precedence resolution.
            source_span: source_span
                .as_ref()
                .map(|value| span(value, input))
                .transpose()?
                .flatten(),
        },
        // Keep field presence independently of the decoded Option. Omission permits
        // metadata inheritance, while an explicit None is a request to clear the
        // source span.
        source_span_provided: source_span.is_some(),
    })
}

// Build a typed embedding row, keeping target ID, model and declared width
// separate from the vector allocation. The core checks that these values agree
// before admitting storage.
fn embedding(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<EmbeddingInput> {
    let object = object(value, &["chunk_external_id", "model_name", "dim", "vector"])?;
    Ok(EmbeddingInput {
        chunk_external_id: required_text(object, "chunk_external_id", input)?,
        model_name: required_text(object, "model_name", input)?,
        dim: native_integer(&required(object, "dim")?, "dim")?,
        // Read coordinates directly into a binary32 buffer charged to this row.
        // The vector parser does not allocate a generic JSON tree for those
        // numbers.
        vector: convert::vector(&required(object, "vector")?, input, ValueKind::JSONL_RECORD)?,
    })
}

// Convert the supplied sequences into one prepared Batch under the writer's
// operation. The native call is a separate step after conversion succeeds.
pub(crate) fn batch(
    operation: &Operation,
    documents: Option<&Bound<'_, PyAny>>,
    chunks: Option<&Bound<'_, PyAny>>,
    embeddings: Option<&Bound<'_, PyAny>>,
) -> Result<Batch> {
    // Convert all three sequences into one native batch before executing a write.
    // Each row transfers its owned data and reservation together. If a later
    // conversion fails, the partially built batch is dropped without submitting any
    // earlier rows.
    let mut batch = operation.batch()?;
    // Keep conversion order in document, chunk and embedding groups. The position
    // in this fixed array determines which record parser is used.
    for (kind, value) in [documents, chunks, embeddings].into_iter().enumerate() {
        // An omitted sequence has no rows to add. A supplied object must still
        // pass the exact sequence type check even when it is empty.
        let Some(value) = value else {
            continue;
        };
        let sequence = convert::Sequence::new(value)?;
        for index in 0..sequence.len() {
            let item = sequence.item(index)?;
            // Give each row its own input builder on the shared operation. A
            // successful push transfers the row and its lease to the batch.
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
