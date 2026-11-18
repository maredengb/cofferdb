//! Errors raised before manifest import starts.
//!
//! Callers that receive an error from core import must not pass it through
//! these helpers. Doing so would replace the core `suggested_action`.

use crate::error::{Error, ErrorCode};

pub(super) fn cli(message: impl Into<String>, field: Option<&str>, reason: Option<&str>) -> Error {
    let mut error = Error::new(ErrorCode::CliUsage, message);
    if let Some(field) = field {
        error = error.with_detail("field", field);
    }
    if let Some(reason) = reason {
        error = error.with_detail("reason", reason);
    }
    error.ingestion_boundary()
}

pub(super) fn missing(field: &str) -> Error {
    Error::new(
        ErrorCode::MissingRequiredField,
        format!("{field} is required"),
    )
    .with_detail("field", field)
    .ingestion_boundary()
}

pub(super) fn io_ingest(message: &str, stage: Option<&str>) -> Error {
    let mut error = Error::new(ErrorCode::IoError, message);
    if let Some(stage) = stage {
        error = error.with_detail("provider_stage", stage);
    }
    error.ingestion_boundary()
}

pub(super) fn schema(message: &str) -> Error {
    Error::new(ErrorCode::InvalidJsonlSchema, message).ingestion_boundary()
}

pub(super) fn vector_length(message: &str) -> Error {
    Error::new(ErrorCode::InvalidVectorLength, message).ingestion_boundary()
}

pub(super) fn with_host(error: Error, host: &str) -> Error {
    error.with_detail("endpoint_host", host)
}
