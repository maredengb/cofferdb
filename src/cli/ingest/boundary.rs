//! Errors raised before manifest import starts.
//!
//! Callers that receive an error from core import must not pass it through
//! these helpers. Doing so would replace the core `suggested_action`.

use crate::error::{Error, ErrorCode};

// Construct a CLI usage failure before core import. Optional detail keys name a
// specific flag or validation reason for callers that can handle it.
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

// Use the required-field code for missing input while retaining the ingestion
// phase marker. The caller supplies the field name used by script diagnostics.
pub(super) fn missing(field: &str) -> Error {
    Error::new(
        ErrorCode::MissingRequiredField,
        format!("{field} is required"),
    )
    .with_detail("field", field)
    .ingestion_boundary()
}

// Mark staging I/O failures before core import. Provider operations can supply a
// stage, while local file operations can omit it.
pub(super) fn io_ingest(message: &str, stage: Option<&str>) -> Error {
    let mut error = Error::new(ErrorCode::IoError, message);
    if let Some(stage) = stage {
        error = error.with_detail("provider_stage", stage);
    }
    error.ingestion_boundary()
}

// Keep malformed staged input in the existing JSONL schema category. The phase
// marker controls guidance without changing the public error code.
pub(super) fn schema(message: &str) -> Error {
    Error::new(ErrorCode::InvalidJsonlSchema, message).ingestion_boundary()
}

// Report invalid vector shape in the same category used by import. Applying the
// phase marker here identifies that the vector was still being staged.
pub(super) fn vector_length(message: &str) -> Error {
    Error::new(ErrorCode::InvalidVectorLength, message).ingestion_boundary()
}

// Attach the parsed endpoint host as a diagnostic detail. Adding the host
// preserves the error code, phase and any details already set.
pub(super) fn with_host(error: Error, host: &str) -> Error {
    error.with_detail("endpoint_host", host)
}
