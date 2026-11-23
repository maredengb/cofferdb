//! Structured errors shared by the database, CLI and binding adapters. Call paths
//! can attach a location or the outcome of a write as the error propagates; the
//! JSON projection preserves the stable category and selected detail types.
use std::fmt::{Display, Formatter};

use crate::json::{JsonBuilder, JsonValue};

/// Result of a fallible CofferDB operation, carrying a structured [Error] on failure.
pub type Result<T> = std::result::Result<T, Error>;

/// Machine-readable categories for database, input and operational failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    /// An underlying filesystem or I/O operation failed.
    IoError,
    /// Database or stored record bytes do not match the expected format.
    FileFormatError,
    /// The supplied format or schema version is unsupported.
    UnsupportedVersion,
    /// The requested feature or a required file feature is unsupported.
    UnsupportedFeatureFlag,
    /// The requested collection is absent from this view.
    CollectionNotFound,
    /// A visible collection already uses the requested name.
    CollectionAlreadyExists,
    /// The declared embedding dimension is outside the accepted range.
    InvalidEmbeddingDimension,
    /// A required embedding is absent.
    EmbeddingMissing,
    /// The metadata filter does not satisfy the supported filter syntax.
    InvalidMetadataFilter,
    /// Input does not satisfy the expected JSON or JSONL schema.
    InvalidJsonlSchema,
    /// A required input field or value was omitted.
    MissingRequiredField,
    /// A vector's component count does not match the required dimension.
    InvalidVectorLength,
    /// The model identity conflicts with the collection or stored embeddings.
    EmbeddingModelMismatch,
    /// The requested distance metric is unsupported.
    UnsupportedMetric,
    /// An external identifier conflicts with an existing or incoming record.
    DuplicateExternalId,
    /// A deleted collection still reserves the requested name.
    CollectionDeletedNameReserved,
    /// Required access to the database or a stable scan view could not be obtained.
    LockContention,
    /// A computed content checksum differs from its recorded or expected value.
    ChecksumMismatch,
    /// Benchmark artifact settings conflict with the requested run.
    BenchmarkArtifactModeConflict,
    /// Invalid trailing bytes were found beyond valid database records.
    CorruptTail,
    /// Derived index data failed validation.
    IndexCorruption,
    /// The requested recovery or database-state transition could not be completed safely.
    RecoveryError,
    /// The operation exceeded an available resource or configured budget.
    ResourceExhausted,
    /// The operation is forbidden for this handle, path or access mode.
    PermissionDenied,
    /// Query input or search configuration fails validation.
    InvalidSearchParams,
    /// Ranking weights fail their validation rules.
    InvalidSearchWeights,
    /// Metadata ranking boost configuration fails validation.
    InvalidMetadataBoost,
    /// The requested document is absent from the visible collection.
    DocumentNotFound,
    /// The requested chunk is absent from the visible collection.
    ChunkNotFound,
    /// Command-line arguments are missing, invalid or incompatible.
    CliUsage,
}

impl ErrorCode {
    /// Return the stable snake-case code used in JSON and diagnostics.
    pub fn as_str(self) -> &'static str {
        // Keep the wire spelling explicit for every variant. JSON consumers match
        // on these values, and adding a future category should force a decision
        // about its externally visible name.
        match self {
            ErrorCode::IoError => "io_error",
            ErrorCode::FileFormatError => "file_format_error",
            ErrorCode::UnsupportedVersion => "unsupported_version",
            ErrorCode::UnsupportedFeatureFlag => "unsupported_feature_flag",
            ErrorCode::CollectionNotFound => "collection_not_found",
            ErrorCode::CollectionAlreadyExists => "collection_already_exists",
            ErrorCode::InvalidEmbeddingDimension => "invalid_embedding_dimension",
            ErrorCode::EmbeddingMissing => "embedding_missing",
            ErrorCode::InvalidMetadataFilter => "invalid_metadata_filter",
            ErrorCode::InvalidJsonlSchema => "invalid_jsonl_schema",
            ErrorCode::MissingRequiredField => "missing_required_field",
            ErrorCode::InvalidVectorLength => "invalid_vector_length",
            ErrorCode::EmbeddingModelMismatch => "embedding_model_mismatch",
            ErrorCode::UnsupportedMetric => "unsupported_metric",
            ErrorCode::DuplicateExternalId => "duplicate_external_id",
            ErrorCode::CollectionDeletedNameReserved => "collection_deleted_name_reserved",
            ErrorCode::LockContention => "lock_contention",
            ErrorCode::ChecksumMismatch => "checksum_mismatch",
            ErrorCode::BenchmarkArtifactModeConflict => "benchmark_artifact_mode_conflict",
            ErrorCode::CorruptTail => "corrupt_tail",
            ErrorCode::IndexCorruption => "index_corruption",
            ErrorCode::RecoveryError => "recovery_error",
            ErrorCode::ResourceExhausted => "resource_exhausted",
            ErrorCode::PermissionDenied => "permission_denied",
            ErrorCode::InvalidSearchParams => "invalid_search_params",
            ErrorCode::InvalidSearchWeights => "invalid_search_weights",
            ErrorCode::InvalidMetadataBoost => "invalid_metadata_boost",
            ErrorCode::DocumentNotFound => "document_not_found",
            ErrorCode::ChunkNotFound => "chunk_not_found",
            ErrorCode::CliUsage => "cli_usage",
        }
    }
}

/// A database or input error with a machine-readable code, message and operation-specific details.
#[derive(Debug)]
pub struct Error {
    // Typed engine category used by Rust callers and by the public error
    // projection.
    code: ErrorCode,
    // Explanation of this failure instance, kept separately from the stable
    // machine-readable category.
    message: String,
    // Ordered detail fields captured while the error propagates. The host envelope
    // can use these to describe a stage, location or observed side effect.
    detail: Vec<(String, String)>,
    // Indices into detail for values deliberately recorded as numbers.
    // Numeric-looking identifiers and ordinary diagnostic strings must keep their
    // original type when exported.
    numeric_details: Vec<usize>,
    // Underlying operating-system error when one exists, preserved for Rust error
    // chaining.
    source: Option<std::io::Error>,
    /// Errors raised by v0.1.2 ingestion before manifest import starts.
    pre_import: bool,
}

impl Error {
    // Budget failure has two quantities with different ranges: the configured
    // limit is u64 and a requested total may already exceed that width. Keep the
    // whole requested amount for diagnosis.
    pub(crate) fn resource_exhausted(
        resource: &'static str,
        operation: &'static str,
        limit_bytes: u64,
        requested_bytes: u128,
    ) -> Self {
        // Decimal strings keep budget diagnostics exact even when summing overflows u64.
        Self::new(
            ErrorCode::ResourceExhausted,
            "operation resource budget exhausted",
        )
        .with_detail("resource", resource)
        .with_detail("operation", operation)
        .with_detail("limit_bytes", limit_bytes)
        .with_detail("requested_bytes", requested_bytes)
    }

    /// Construct an error with a category, a human-readable message and no detail entries.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        // An ordinary error starts outside the ingestion-specific envelope.
        // Callers add details as they learn where the failure occurred.
        Self {
            code,
            message: message.into(),
            detail: Vec::new(),
            numeric_details: Vec::new(),
            source: None,
            pre_import: false,
        }
    }

    /// Mark an ingestion failure that has not entered manifest import.
    ///
    /// The public code stays one of the existing codes. The envelope for this
    /// phase is recoverable and does not suggest a retry or a command-line secret.
    pub(crate) fn ingestion_boundary(mut self) -> Self {
        // Mark the pre-import phase and preserve the underlying Rust error's
        // category, message and I/O source.
        self.pre_import = true;
        self
    }

    pub(crate) fn index_retry_failure(mut self) -> Self {
        // Replace the lower layer's recorded effect in place. Keep any stage it
        // supplied to describe where the retry failed.
        if let Some((_, effect)) = self.detail.iter_mut().find(|(k, _)| k == "effect") {
            *effect = "index_may_need_rebuild".into();
            self
        } else {
            // With no prior effect, add both the retry phase and its possible
            // index consequence. A failed retry says the derived index may need
            // rebuilding, without making a new claim about the primary append.
            self.maintenance_phase("retry_checkpoint", "index_may_need_rebuild")
        }
    }
    pub(crate) fn checkpoint_verification_failed(mut self) -> Self {
        // A read-only scan can report its own unchanged effect after the writer
        // has published a full trailer. Keep that narrower scope as evidence.
        for (key, _) in &mut self.detail {
            match key.as_str() {
                "stage" => *key = "verification_stage".into(),
                "effect" => *key = "verification_effect".into(),
                _ => {}
            }
        }
        self.with_detail("stage", "verify_checkpoint")
            .with_detail("effect", "index_may_need_rebuild")
    }

    pub(crate) fn maintenance_phase(self, stage: &str, effect: &str) -> Self {
        // Preserve the details supplied by the operation that recorded an effect.
        // Skip adding the outer stage and a second effect.
        if self.detail.iter().any(|(key, _)| key == "effect") {
            return self;
        }
        self.with_detail("stage", stage)
            .with_detail("effect", effect)
    }

    /// Append a key and its stringified value to the diagnostic details and return the extended
    /// error.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        // Ordinary details are textual even when the caller supplies a number.
        // This also preserves exact resource sizes that cannot be represented by a
        // JSON binary64 number.
        self.detail.push((key.into(), value.to_string()));
        self
    }

    pub(crate) fn with_u64_detail(mut self, key: impl Into<String>, value: u64) -> Self {
        // JsonValue stores numbers as f64; parsed integer diagnostics must round-trip exactly.
        debug_assert_eq!((value as f64) as u128, u128::from(value));
        // Tag the position of this entry before appending it. The Rust detail
        // slice still exposes decimal text; only JSON projection uses the tag to
        // emit an actual number.
        self.numeric_details.push(self.detail.len());
        self.detail.push((key.into(), value.to_string()));
        self
    }

    /// Return the category for matching on this failure in Rust.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Return the same snake-case identifier used by [ErrorCode::as_str].
    pub fn code_str(&self) -> &'static str {
        self.code.as_str()
    }

    /// Borrow the human-readable explanation of this particular failure.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Borrow the ordered diagnostic key/value pairs. Inspect operation-specific fields such as
    /// `stage` and `effect` before retrying a maintenance write.
    pub fn detail(&self) -> &[(String, String)] {
        &self.detail
    }

    pub(crate) fn map_path_details(
        mut self,
        mut public_path: impl FnMut(&str) -> Option<String>,
    ) -> Self {
        // Numeric types are recorded by position in the detail vector. Save the
        // old positions before filtering so removing a private pathname cannot
        // change the JSON type of the next field.
        let numeric = std::mem::take(&mut self.numeric_details);
        let mut original = 0;
        let mut retained = 0;
        self.detail.retain_mut(|(key, value)| {
            // Only these two explicitly designated fields are passed to the
            // caller's path mapper. An unrecognized path can disappear while the
            // original error code, source and unrelated details stay intact.
            let keep = if matches!(key.as_str(), "path" | "manifest_path") {
                match public_path(value) {
                    Some(public) => {
                        *value = public;
                        true
                    }
                    None => false,
                }
            } else {
                true
            };
            // Advance the retained index only for a surviving pair. A mapped path
            // still occupies one output slot, whereas a removed path shifts every
            // later numeric index down by one.
            if keep {
                if numeric.contains(&original) {
                    self.numeric_details.push(retained);
                }
                retained += 1;
            }
            original += 1;
            keep
        });
        self
    }

    /// Build the structured error object with code, message, detail, recoverability and
    /// suggested action.
    pub fn to_json(&self) -> JsonValue {
        // Borrowing projection leaves the original error and its ordered details
        // intact. Use the same envelope builder as the consuming path so
        // recoverability, suggested actions and JSON field types agree.
        ErrorProjection::of(self)
            .build(
                self.message.clone(),
                self.detail.iter().cloned(),
                &self.numeric_details,
                &JsonBuilder::host(),
            )
            .expect("host error projection is infallible")
    }

    pub(crate) fn into_json_retained(
        self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        // Derive the recovery decision while the full error and its details are
        // still available. The message and detail strings can then move into a
        // projection charged to the caller's operation.
        let projection = ErrorProjection::of(&self);
        projection.build(
            self.message,
            self.detail.into_iter(),
            &self.numeric_details,
            &JsonBuilder::budgeted(&mut work.json, crate::json::JsonInputKind::StoredJson),
        )
    }

    fn recoverable(&self) -> bool {
        // A transient check-view failure is recoverable even when its broad
        // category also covers more serious storage errors. Consult this
        // operation-specific reason before the category fallback.
        if self
            .detail
            .iter()
            .any(|(k, v)| k == "reason" && v == "check_view_unavailable")
        {
            return true;
        }
        // Ingestion that stopped before manifest processing has a recoverable
        // envelope. The separate action selector can still decline to recommend
        // any retry.
        if self.pre_import {
            return true;
        }
        // JSON limit details refine several public categories. Validation and
        // input-limit failures may be recoverable, but a limit observed during
        // recovery alone does not establish a recoverable file.
        if crate::json::is_limit_error(self) {
            return self.code != ErrorCode::RecoveryError;
        }
        // Recognize the precise writer-reopen and manifest-vector conditions. A
        // generic recovery or schema error does not gain the same status merely by
        // sharing their category.
        if self.requires_writer_reopen() || self.is_manifest_vector_config_error() {
            return true;
        }
        // Use category defaults for the remaining failures. The recoverable flag
        // describes whether a recovery path exists; suggested_action separately
        // describes what the caller can actually do.
        matches!(
            self.code,
            ErrorCode::DuplicateExternalId
                | ErrorCode::CollectionAlreadyExists
                | ErrorCode::CollectionDeletedNameReserved
                | ErrorCode::LockContention
                | ErrorCode::IndexCorruption
                | ErrorCode::ResourceExhausted
                | ErrorCode::EmbeddingModelMismatch
        )
    }

    // Derive guidance from the observed effect and reason before applying
    // code-level defaults. Errors with the same code can require different next
    // steps after an operation changed durable state.
    fn suggested_action(&self) -> Option<&'static str> {
        // A content-lock deadline is a contention case with a specific retry
        // point. It can follow a check or a writer holding the content range.
        if self
            .detail
            .iter()
            .any(|(k, v)| k == "reason" && v == "content_lock_timeout")
        {
            return Some("retry after the active check or content change releases its lock");
        }
        // An unavailable check view calls for another consistent read attempt
        // after the writer finishes. This reason is more specific than general
        // writer admission failure.
        if self
            .detail
            .iter()
            .any(|(k, v)| k == "reason" && v == "check_view_unavailable")
        {
            return Some("retry check after the active write finishes");
        }
        // Inspect the reported file effect before general recovery guidance. A
        // visible replacement or published snapshot can need verification while
        // durability is uncertain.
        if let Some((_, effect)) = self.detail.iter().find(|(k, _)| k == "effect") {
            match effect.as_str() {
                // A published snapshot is identified by a target and receipt.
                // Guide the caller to verify it, since repeating a successful
                // publication can instead encounter an existing destination.
                "snapshot_published" => {
                    return Some(
                        "verify the existing target against the reported size, commit and digest",
                    )
                }
                // For a possibly changed primary path, reopen the current file and
                // inspect the last complete commit. The caller cannot infer the
                // file's state solely from command failure.
                "tail_truncated" | "main_replaced" | "unknown" => return Some(
                    "reopen the current path and check its last complete commit before retrying",
                ),
                // Treat stale or incomplete index state as a separate effect.
                // Verify the observed checkpoint before selecting an index
                // rebuild.
                "index_may_need_rebuild" => {
                    return Some("reopen and verify the checkpoint; rebuild the index if needed")
                }
                _ => {}
            }
        }
        // Lock/view reasons and maintenance effects take precedence. Other
        // pre-import failures have no suggested action.
        if self.pre_import {
            return None;
        }
        // Recognize memory and structural JSON limits by detail. Stored-file
        // recovery requires preserving the original bytes; a caller-supplied value
        // can be reduced or processed under less pressure.
        if crate::json::is_limit_error(self) {
            return Some(if self.code == ErrorCode::RecoveryError {
                "preserve the original file and use a compatible version for explicit migration"
            } else {
                "reduce the JSON value or processing footprint and retry"
            });
        }
        // An outdated writer must be replaced before another mutation. The
        // physical path may refer to a new generation after compaction.
        if self.requires_writer_reopen() {
            return Some("close the outdated writer if still open, confirm the current main database file after any compact replacement, and reopen a writer");
        }
        // Use the normalization mismatch guidance only for the manifest field that
        // caused it. Other schema failures need their own field-specific repair.
        if self.is_manifest_vector_config_error() {
            return Some("correct the manifest vectors_normalized declaration or choose a collection with matching configuration; do not rewrite existing vectors or change the collection configuration");
        }
        // Use code-only defaults when no more specific condition supplied
        // guidance. The effect and reason checks above take precedence over these
        // suggestions.
        match self.code {
            ErrorCode::DuplicateExternalId => {
                Some("rerun import with --on-conflict skip or replace")
            }
            ErrorCode::LockContention => Some("retry after the active writer finishes"),
            ErrorCode::IndexCorruption => {
                Some("run coffer rebuild-index for the affected collection")
            }
            // Resource refusal can follow partial progress on a write. Confirm
            // what committed before repeating the whole input.
            ErrorCode::ResourceExhausted => {
                Some("release resources or reduce the operation footprint; verify committed progress before retrying writes")
            }
            ErrorCode::EmbeddingModelMismatch => {
                Some("use the collection model, or explicitly export and clean up conflicting visible embeddings; rebuilding indexes or truncating the file does not change their models")
            }
            _ => None,
        }
    }

    // Identify the reopen requirement by both recovery code and reason. Other
    // recovery failures may require inspecting or preserving the file rather than
    // opening a replacement writer.
    fn requires_writer_reopen(&self) -> bool {
        self.code == ErrorCode::RecoveryError
            && self
                .detail
                .iter()
                .any(|(key, value)| key == "reason" && value == "writer_reopen_required")
    }

    // Distinguish a manifest normalization conflict from an arbitrary invalid row.
    // Both the schema code and field label are part of the match.
    fn is_manifest_vector_config_error(&self) -> bool {
        self.code == ErrorCode::InvalidJsonlSchema
            && self
                .detail
                .iter()
                .any(|(key, value)| key == "field" && value == "vectors_normalized")
    }
}

struct ErrorProjection {
    // Machine-readable error identifier to use in the projected envelope.
    code: &'static str,
    // Whether the documented recovery path is available for this particular
    // failure. Details can change the answer for two errors with the same code.
    recoverable: bool,
    // Suggested next action for callers, absent when the engine has no specific
    // action to recommend.
    suggested_action: Option<&'static str>,
}

impl ErrorProjection {
    // Capture code, recoverability and action from one error before moving any
    // owned fields into the builder. This keeps the JSON envelope consistent
    // across borrowed and consuming paths.
    fn of(error: &Error) -> Self {
        Self {
            code: error.code_str(),
            recoverable: error.recoverable(),
            suggested_action: error.suggested_action(),
        }
    }

    // Build the stable error envelope under the chosen result builder. Detail
    // field values retain their wire types and owned strings can move into the new
    // representation.
    fn build(
        self,
        message: String,
        detail: impl ExactSizeIterator<Item = (String, String)>,
        numeric: &[usize],
        builder: &JsonBuilder<'_>,
    ) -> Result<JsonValue> {
        // Track numeric details by the source entry's index before inserting into
        // the object. This preserves whether each original detail was added as
        // numeric or text.
        let detail = builder.object_entries(detail.enumerate().map(|(index, (key, value))| {
            let key = builder.owned_key(key)?;
            // Render a recorded numeric u64 as a JSON number. Text details keep
            // their exact spelling even when their characters are all digits.
            let value = if numeric.contains(&index) {
                JsonValue::Number(value.parse::<u64>().expect("numeric detail") as f64)
            } else {
                // Use the owned-string builder for nonnumeric detail values. A
                // consuming projection can transfer their current capacity into
                // the final JSON owner.
                builder.owned_string(value)?
            };
            Ok((key, value))
        }))?;
        // Include suggested_action with explicit null when the error has no
        // specific guidance. Consumers can use one envelope schema for all error
        // categories.
        builder.object([
            ("code", builder.string(self.code)?),
            ("message", builder.owned_string(message)?),
            ("recoverable", JsonValue::Bool(self.recoverable)),
            (
                "suggested_action",
                builder.optional_str(self.suggested_action)?,
            ),
            ("detail", detail),
        ])
    }
}

impl Display for Error {
    // The display form is for terminal or log text. It keeps details in insertion
    // order, while structured JSON consumers use the dedicated projection.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)?;
        for (key, value) in &self.detail {
            write!(f, " ({key}={value})")?;
        }
        Ok(())
    }
}

// Expose the original I/O cause through the standard Rust error chain when
// available. Serialization does not have to retain a live source error.
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|err| err as &(dyn std::error::Error + 'static))
    }
}

impl From<std::io::Error> for Error {
    // Translate the standard permission category separately from general I/O.
    // Preserve the original error as the source so later code can inspect its
    // native cause.
    fn from(source: std::io::Error) -> Self {
        let code = if source.kind() == std::io::ErrorKind::PermissionDenied {
            ErrorCode::PermissionDenied
        } else {
            ErrorCode::IoError
        };
        // Keep the source's message and store the full source value. Converting an
        // I/O error here does not invent artifact context; callers add it at the
        // boundary that knows it.
        Self {
            code,
            message: source.to_string(),
            detail: Vec::new(),
            numeric_details: Vec::new(),
            source: Some(source),
            pre_import: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_reopen_reason_is_conditionally_recoverable_without_changing_other_errors() {
        let error = Error::new(ErrorCode::RecoveryError, "retained old file")
            .with_detail("reason", "writer_reopen_required");
        let json = error.to_json();
        crate::test_error_contract::validate_writer_reopen(&json).unwrap();
        assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(true)));
        let action = json
            .get("suggested_action")
            .and_then(JsonValue::as_str)
            .unwrap();
        assert!(
            action.contains("close") && action.contains("current") && action.contains("reopen")
        );
        assert!(!action.contains("delete") && !action.contains("repair"));
        for other in [
            Error::new(ErrorCode::RecoveryError, "other recovery failure"),
            Error::new(ErrorCode::RecoveryError, "other reason").with_detail("reason", "other"),
            Error::new(ErrorCode::IoError, "IO failure")
                .with_detail("reason", "writer_reopen_required"),
        ] {
            let json = other.to_json();
            assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(false)));
            assert_eq!(json.get("suggested_action"), Some(&JsonValue::Null));
        }
    }

    #[test]
    fn path_redaction_preserves_core_fields_sources_and_numeric_detail_positions() {
        for seed in 0..100_u64 {
            let error = Error::from(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
                .with_u64_detail("line", seed + 1)
                .with_detail("path", "/private/generated.json")
                .with_u64_detail("expected_dim", 768)
                .with_detail("manifest_path", "user/dataset.json")
                .with_detail("requested_bytes", u128::MAX)
                .with_u64_detail("actual_vector_length", seed);
            let before = error.to_json();
            let mapped = error
                .map_path_details(|path| (path == "user/dataset.json").then(|| path.to_owned()));
            let after = mapped.to_json();
            for field in ["code", "message", "recoverable", "suggested_action"] {
                assert_eq!(before.get(field), after.get(field));
            }
            let detail = after.get("detail").unwrap();
            for field in [
                "line",
                "expected_dim",
                "manifest_path",
                "requested_bytes",
                "actual_vector_length",
            ] {
                assert_eq!(before.get("detail").unwrap().get(field), detail.get(field));
            }
            assert!(detail.get("path").is_none());
            assert!(std::error::Error::source(&mapped).is_some());
            assert!(!after
                .to_compact_string()
                .contains("/private/generated.json"));
        }
    }

    #[test]
    fn resource_errors_preserve_exact_budget_details_without_claiming_corruption() {
        for resource in ["memory", "catalog", "temporary_disk"] {
            let json = Error::resource_exhausted(resource, "new_stage_2", 0, u128::MAX).to_json();
            crate::test_error_contract::validate_error(&json).unwrap();
            assert_eq!(
                json.get("detail")
                    .unwrap()
                    .get("resource")
                    .and_then(JsonValue::as_str),
                Some(resource)
            );
        }
        for limit in [0, 1, 4096, (1 << 53) + 1, u64::MAX] {
            for additional in 1..=100 {
                let requested = u128::from(limit) + additional;
                let error = Error::resource_exhausted(
                    "temporary_disk",
                    "temporary_staging",
                    limit,
                    requested,
                );
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(!crate::json::is_limit_error(&error));
                let json = error.to_json();
                crate::test_error_contract::validate_error(&json).unwrap();
                assert_eq!(
                    json.get("code").and_then(JsonValue::as_str),
                    Some("resource_exhausted")
                );
                assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(true)));
                assert_eq!(
                    json.get("suggested_action").and_then(JsonValue::as_str),
                    Some("release resources or reduce the operation footprint; verify committed progress before retrying writes")
                );
                let detail = json.get("detail").unwrap().as_object().unwrap();
                assert_eq!(detail.len(), 4);
                for (key, expected) in [
                    ("resource", "temporary_disk".to_owned()),
                    ("operation", "temporary_staging".to_owned()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", requested.to_string()),
                ] {
                    assert_eq!(
                        detail.get(key).and_then(JsonValue::as_str),
                        Some(expected.as_str())
                    );
                }
                assert_eq!(crate::parse_json(&json.to_compact_string()).unwrap(), json);
            }
        }
    }

    #[test]
    fn numeric_details_keep_json_types_and_exactly_representable_large_integers() {
        let values = (0..100).chain([
            (1 << 53) - 1,
            1 << 53,
            (1 << 53) + 2,
            1 << 60,
            u64::MAX - 2047,
        ]);
        for value in values {
            let error = Error::new(ErrorCode::InvalidVectorLength, "test integer")
                .with_detail("query_id", "q")
                .with_u64_detail("row_dim", value)
                .with_detail("string_value", value);
            let wire = error.to_json().to_compact_string();
            let json = crate::parse_json(&wire).unwrap();
            let detail = json.get("detail").unwrap();
            assert_eq!(
                detail.get("row_dim").and_then(JsonValue::as_f64),
                Some(value as f64)
            );
            assert_eq!(
                detail.get("query_id").and_then(JsonValue::as_str),
                Some("q")
            );
            assert_eq!(
                detail.get("string_value").and_then(JsonValue::as_str),
                Some(value.to_string().as_str())
            );
            assert_eq!(error.detail()[1].1, value.to_string());
            let output = crate::binding::Operation::standalone()
                .error(error)
                .unwrap();
            assert_eq!(output.value(), &json);
        }
    }

    #[test]
    fn error_json_is_machine_readable() {
        let err = Error::new(ErrorCode::DuplicateExternalId, "external_id already exists")
            .with_detail("collection", "notes")
            .with_detail("external_id", "doc-1");

        let json = err.to_json();
        assert_eq!(
            json.get("code").and_then(JsonValue::as_str),
            Some("duplicate_external_id")
        );
        assert_eq!(
            json.get("message").and_then(JsonValue::as_str),
            Some("external_id already exists")
        );
        assert_eq!(
            json.get("recoverable").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            json.get("suggested_action").and_then(JsonValue::as_str),
            Some("rerun import with --on-conflict skip or replace")
        );
        let detail = json.get("detail").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            detail.get("collection").and_then(JsonValue::as_str),
            Some("notes")
        );
        assert_eq!(
            detail.get("external_id").and_then(JsonValue::as_str),
            Some("doc-1")
        );
    }

    #[test]
    fn manifest_vector_configuration_errors_are_conditionally_recoverable() {
        for code in [
            ErrorCode::InvalidJsonlSchema,
            ErrorCode::InvalidVectorLength,
        ] {
            for field in [None, Some("embedding_dim"), Some("vectors_normalized")] {
                let mut error = Error::new(code, "validation failed");
                if let Some(field) = field {
                    error = error.with_detail("field", field);
                }
                let expected =
                    code == ErrorCode::InvalidJsonlSchema && field == Some("vectors_normalized");
                let json = error.to_json();
                assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(expected)));
                if expected {
                    assert!(json
                        .get("suggested_action")
                        .and_then(JsonValue::as_str)
                        .unwrap()
                        .contains("do not rewrite existing vectors"));
                } else {
                    assert_eq!(json.get("suggested_action"), Some(&JsonValue::Null));
                }
            }
        }
    }
}
