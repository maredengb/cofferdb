use crate::error::{Error, ErrorCode};

// The versioned policy covers three separate limits. A small serialized value can
// still be too deeply nested, and constructing a valid value can require more live
// memory than its final encoded length.
pub(crate) struct JsonProcessingPolicyV1;

impl JsonProcessingPolicyV1 {
    // Count nested array and object levels. The depth check runs on entry to a
    // container, so an empty container consumes a level while a scalar leaf does
    // not add one.
    pub(crate) const MAX_CONTAINER_DEPTH: usize = 128;
    // Measure one schema-selected business value in its compact canonical form,
    // including escaping and punctuation. A JSONL file or wrapper can contain
    // several independently checked values.
    pub(crate) const MAX_UNIT_CANONICAL_BYTES: usize = 8 * 1024 * 1024;
    // This ceiling applies to JSON work in one operation. The database's shared
    // working-memory limit can reduce the amount available for that operation
    // before this ceiling is reached.
    pub(crate) const WORK_BUDGET_BYTES: usize = 32 * 1024 * 1024;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// Carry the business input kind through parsing, validation and allocation. The
// same structural limit can represent a bad search option, a malformed import or a
// failure processing stored JSON.
pub(crate) enum JsonInputKind {
    // An import or benchmark manifest, including configuration retained by the
    // benchmark runner.
    Manifest,
    // A collection's typed configuration presented for JSON validation or work
    // accounting.
    CollectionConfig,
    // Runtime settings validated while constructing the execution profile.
    RuntimeConfig,
    // Keep document and chunk metadata distinct in diagnostics. Both permit
    // application keys, but import and recovery can attribute a limit to the
    // specific embedded metadata field.
    DocumentMetadata,
    // Metadata owned by a chunk, including chunks read independently of their
    // parent document.
    ChunkMetadata,
    // A predicate, fusion weights and metadata-boost rules have separate public
    // errors. A limit inside one of these nested options should identify its own
    // input kind.
    Filter,
    // The weight configuration for combining search scores.
    Weights,
    // Rules for adjusting scores from document or chunk metadata.
    MetadataBoost,
    // Query vectors have a separate kind so the parser can charge the canonical
    // spelling of the finite f32 coordinates used by scoring, even though the
    // generic DOM retains f64 values.
    QueryEmbedding,
    // The general JSON input classification, also used for import record
    // envelopes.
    JsonlRecord,
    // A benchmark query, expected result or other runner artifact.
    BenchmarkArtifact,
    // Stored JSON can be encountered during recovery as well as retained output
    // processing. This classification reports a limit on such a value through the
    // recovery error category.
    StoredJson,
}

impl JsonInputKind {
    // These spellings appear in json_input_kind details. Keep them independent of
    // Rust's variant formatting so machines can match the same names across parser
    // and typed-value entry points.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::CollectionConfig => "collection_config",
            Self::RuntimeConfig => "runtime_config",
            Self::DocumentMetadata => "document_metadata",
            Self::ChunkMetadata => "chunk_metadata",
            Self::Filter => "filter",
            Self::Weights => "weights",
            Self::MetadataBoost => "metadata_boost",
            Self::QueryEmbedding => "query_embedding",
            Self::JsonlRecord => "jsonl_record",
            Self::BenchmarkArtifact => "benchmark_artifact",
            Self::StoredJson => "stored_json",
        }
    }

    // Choose a public error code from the input kind before attaching the limit
    // reason. Other schema errors can share that code, so recognizing a JSON limit
    // also requires its structured detail.
    pub(crate) fn error_code(self) -> ErrorCode {
        match self {
            Self::Filter => ErrorCode::InvalidMetadataFilter,
            Self::Weights => ErrorCode::InvalidSearchWeights,
            Self::MetadataBoost => ErrorCode::InvalidMetadataBoost,
            Self::StoredJson => ErrorCode::RecoveryError,
            _ => ErrorCode::InvalidJsonlSchema,
        }
    }

    pub(crate) fn limit_error(
        self,
        reason: JsonLimitReason,
        limit: usize,
        actual: Option<usize>,
    ) -> Error {
        // Report the active limit, its unit and the reason together. A depth limit
        // is measured in container levels, while canonical size and working credit
        // are measured in bytes.
        let mut error = Error::new(self.error_code(), reason.message())
            .with_detail("json_limit_reason", reason.as_str())
            .with_detail("json_input_kind", self.as_str())
            .with_u64_detail("limit", limit as u64)
            .with_detail("limit_unit", reason.unit());
        // The DOM uses binary64 for numeric details. Emit actual only through 2^53
        // inclusive, where every integer up to that boundary is exact. Omit larger
        // observations rather than reporting a rounded size.
        if let Some(actual) = actual.filter(|actual| (*actual as u128) <= (1u128 << 53)) {
            error = error.with_u64_detail("actual", actual as u64);
        }
        // Keep the common JSON reason alongside the option-specific reason used by
        // weight and boost diagnostics. Clients can classify the same failure at
        // either layer.
        match self {
            Self::Weights => error.with_detail("weights_reason", "json_processing_limit"),
            Self::MetadataBoost => {
                error.with_detail("metadata_boost_reason", "json_processing_limit")
            }
            _ => error,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum JsonLimitReason {
    // Depth and canonical bytes are intrinsic to the input's shape. A work-budget
    // refusal also depends on which other allocations are live when processing
    // reaches that value.
    Depth,
    UnitBytes,
    WorkBudget,
}

impl JsonLimitReason {
    // Use identifiers distinct from the human-readable messages below. Limit
    // detection matches these values in structured details, leaving prose free to
    // explain the particular failure.
    fn as_str(self) -> &'static str {
        match self {
            Self::Depth => "depth_exceeded",
            Self::UnitBytes => "unit_bytes_exceeded",
            Self::WorkBudget => "work_budget_exceeded",
        }
    }

    // Keep the human-readable message paired with the machine-readable identifier.
    // Callers that classify limits use the identifier stored in error detail.
    fn message(self) -> &'static str {
        match self {
            Self::Depth => "JSON container depth exceeds the processing limit",
            Self::UnitBytes => "JSON canonical bytes exceed the processing limit",
            Self::WorkBudget => "JSON processing exceeds the available work budget",
        }
    }

    // Depth is measured in container levels. Canonical size and transient work use
    // bytes, even though their limits account for different storage.
    fn unit(self) -> &'static str {
        match self {
            Self::Depth => "container_levels",
            _ => "bytes",
        }
    }
}

// Recognize a policy refusal across input kinds by its structured reason. The
// outer ErrorCode also describes malformed input in some of those kinds.
pub(crate) fn is_limit_error(error: &Error) -> bool {
    error.detail().iter().any(|(key, value)| {
        key == "json_limit_reason"
            && matches!(
                value.as_str(),
                "depth_exceeded" | "unit_bytes_exceeded" | "work_budget_exceeded"
            )
    })
}

// The caller may free optional cache or buffer space. Reclamation cannot raise a
// fixed depth or canonical-size limit or relieve an exhausted non-memory resource.
pub(crate) fn is_memory_budget_error(error: &Error) -> bool {
    (error.code() == ErrorCode::ResourceExhausted
        && error
            .detail()
            .iter()
            .any(|(key, value)| key == "resource" && value == "memory"))
        || error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded")
}

// The attempt must be safe to repeat before IO, retaining any live input credit.
pub(crate) fn retry_after_memory_reclaim<T>(
    mut attempt: impl FnMut() -> crate::Result<T>,
    reclaim: impl FnOnce() -> bool,
) -> crate::Result<T> {
    // Try once, then offer one reclamation on a recoverable memory refusal. Return
    // the original error when reclamation declines and the second result when
    // another attempt runs.
    match attempt() {
        Err(error) if is_memory_budget_error(&error) && reclaim() => attempt(),
        result => result,
    }
}

#[derive(Clone, Copy)]
// A schema selects record framing and routes errors to the appropriate input kind.
// Preserve it through descent until a field starts a separate JSON unit.
pub(crate) enum JsonSchema {
    // A standalone business value, with its own depth and canonical-size limits.
    Value(JsonInputKind),
    // An import document row, whose metadata gets its own input kind.
    DocumentRow,
    // An import chunk row, whose metadata uses the chunk-specific input kind.
    ChunkRow,
    // An import embedding row, with ordinary fields checked in the row's context.
    EmbeddingRow,
    // A benchmark query envelope. Route named business fields independently when
    // descending into the object.
    QueryRow,
    // A scalar query parameter with validation tied to its location in the query
    // record.
    QueryTopK,
    // A query-embedding record whose vector receives the query embedding context.
    QueryEmbeddingObject,
    // A benchmark artifact envelope using the benchmark error context for ordinary
    // fields.
    BenchmarkRecord,
    // The record envelope used for stored JSON. Treat recognized stored fields
    // individually during descent.
    StoredRecord,
    // An ordinary record member inherits the enclosing error kind and remains
    // subject to depth and workspace checks. It does not start a separate
    // canonical-size unit.
    RecordField(JsonInputKind),
}

impl JsonSchema {
    // Translate schema identity into the ErrorCode and details used for the
    // current unit. Several record schemas share a kind but select different
    // fields.
    pub(crate) fn kind(self) -> JsonInputKind {
        match self {
            Self::Value(kind) | Self::RecordField(kind) => kind,
            Self::DocumentRow | Self::ChunkRow | Self::EmbeddingRow => JsonInputKind::JsonlRecord,
            Self::QueryRow | Self::QueryTopK | Self::BenchmarkRecord => {
                JsonInputKind::BenchmarkArtifact
            }
            Self::QueryEmbeddingObject => JsonInputKind::QueryEmbedding,
            Self::StoredRecord => JsonInputKind::StoredJson,
        }
    }

    // Only envelopes get record framing. A Value, RecordField or top-k scalar has
    // a value layout even when it was reached through a record.
    pub(crate) fn is_record(self) -> bool {
        !matches!(
            self,
            Self::Value(_) | Self::RecordField(_) | Self::QueryTopK
        )
    }

    pub(crate) fn field(self, key: &str) -> Self {
        // Dispatch by both parent schema and field name so an identically named
        // key inside an arbitrary business value cannot acquire special record
        // semantics.
        let kind = match (self, key) {
            (Self::QueryRow, "top_k") => return Self::QueryTopK,
            (Self::DocumentRow, "metadata") => JsonInputKind::DocumentMetadata,
            (Self::ChunkRow, "metadata") => JsonInputKind::ChunkMetadata,
            (Self::QueryRow, "filter") => JsonInputKind::Filter,
            (Self::QueryRow, "weights") => JsonInputKind::Weights,
            (Self::QueryRow, "metadata_boost") => JsonInputKind::MetadataBoost,
            // Use the embedding kind for the direct query field and for a vector
            // inside the dedicated embedding object. Both represent query
            // embedding values.
            (Self::QueryRow, "embedding") | (Self::QueryEmbeddingObject, "vector") => {
                JsonInputKind::QueryEmbedding
            }
            (Self::StoredRecord, "metadata" | "config" | "tokenizer") => JsonInputKind::StoredJson,
            // Once a value's context has been selected, descendants inherit it.
            // Their structure contributes to the same enclosing value's checks.
            (Self::Value(_) | Self::RecordField(_), _) => return self,
            // Unknown fields still receive ordinary record-member treatment with
            // the parent kind. Consumers decide separately whether an unknown key
            // is allowed.
            _ => return Self::RecordField(self.kind()),
        };
        Self::Value(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonValue;

    #[test]
    fn memory_reclaim_attempts_are_bounded_and_preserve_errors() {
        for resource in ["memory", "catalog", "temporary_disk"] {
            for released in [false, true] {
                let mut attempts = 0;
                let mut reclaims = 0;
                let error = retry_after_memory_reclaim(
                    || {
                        attempts += 1;
                        Err::<(), _>(Error::resource_exhausted(
                            resource,
                            "batch_prepare",
                            100,
                            100 + attempts,
                        ))
                    },
                    || {
                        reclaims += 1;
                        released
                    },
                )
                .unwrap_err();
                assert_eq!(attempts, 1 + u128::from(resource == "memory" && released));
                assert_eq!(reclaims, usize::from(resource == "memory"));
                let expected =
                    Error::resource_exhausted(resource, "batch_prepare", 100, 100 + attempts);
                assert_eq!(error.code(), expected.code());
                assert_eq!(error.detail(), expected.detail());
            }
        }
        let value = retry_after_memory_reclaim(|| Ok(42), || panic!("success must retain cache"));
        assert_eq!(value.unwrap(), 42);
        for error in [
            Error::new(ErrorCode::IoError, "original IO failure"),
            JsonInputKind::DocumentMetadata.limit_error(JsonLimitReason::Depth, 128, Some(129)),
        ] {
            let expected = error.to_json();
            let mut error = Some(error);
            let actual = retry_after_memory_reclaim(
                || Err::<(), _>(error.take().expect("must not retry")),
                || panic!("must not reclaim"),
            )
            .unwrap_err();
            assert_eq!(actual.to_json(), expected);
        }
    }

    #[test]
    fn memory_reclaim_does_not_retry_other_resource_or_storage_failures() {
        for resource in ["memory", "catalog", "temporary_disk"] {
            let error = Error::resource_exhausted(resource, "catalog_build", 100, 101);
            assert_eq!(is_memory_budget_error(&error), resource == "memory");
        }
        for code in [
            ErrorCode::IoError,
            ErrorCode::ChecksumMismatch,
            ErrorCode::RecoveryError,
            ErrorCode::FileFormatError,
        ] {
            assert!(!is_memory_budget_error(&Error::new(
                code,
                "not a budget rejection"
            )));
        }
    }

    #[test]
    fn all_json_input_kinds_have_typed_limit_details_and_domain_codes() {
        for (kind, name, code) in [
            (JsonInputKind::Manifest, "manifest", "invalid_jsonl_schema"),
            (
                JsonInputKind::CollectionConfig,
                "collection_config",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::RuntimeConfig,
                "runtime_config",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::DocumentMetadata,
                "document_metadata",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::ChunkMetadata,
                "chunk_metadata",
                "invalid_jsonl_schema",
            ),
            (JsonInputKind::Filter, "filter", "invalid_metadata_filter"),
            (JsonInputKind::Weights, "weights", "invalid_search_weights"),
            (
                JsonInputKind::MetadataBoost,
                "metadata_boost",
                "invalid_metadata_boost",
            ),
            (
                JsonInputKind::QueryEmbedding,
                "query_embedding",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::JsonlRecord,
                "jsonl_record",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::BenchmarkArtifact,
                "benchmark_artifact",
                "invalid_jsonl_schema",
            ),
            (JsonInputKind::StoredJson, "stored_json", "recovery_error"),
        ] {
            for (reason, limit, unit) in [
                (JsonLimitReason::Depth, 128, "container_levels"),
                (
                    JsonLimitReason::UnitBytes,
                    JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                    "bytes",
                ),
                (JsonLimitReason::WorkBudget, 4096, "bytes"),
            ] {
                let error = kind.limit_error(reason, limit, Some(limit + 1));
                assert!(is_limit_error(&error));
                assert_eq!(
                    is_memory_budget_error(&error),
                    matches!(reason, JsonLimitReason::WorkBudget)
                );
                assert_eq!(error.code_str(), code);
                let json = error.to_json();
                crate::test_error_contract::validate_error(&json).unwrap();
                crate::test_error_contract::validate_error(
                    &kind.limit_error(reason, limit, None).to_json(),
                )
                .unwrap();
                let detail = json.get("detail").unwrap();
                assert_eq!(detail.get("limit"), Some(&JsonValue::Number(limit as f64)));
                assert_eq!(
                    detail.get("actual"),
                    Some(&JsonValue::Number((limit + 1) as f64))
                );
                assert_eq!(
                    detail.get("json_input_kind").and_then(JsonValue::as_str),
                    Some(name)
                );
                assert_eq!(
                    detail.get("limit_unit").and_then(JsonValue::as_str),
                    Some(unit)
                );
                let reason_field = match kind {
                    JsonInputKind::Weights => Some("weights_reason"),
                    JsonInputKind::MetadataBoost => Some("metadata_boost_reason"),
                    _ => None,
                };
                if let Some(field) = reason_field {
                    assert_eq!(
                        detail.get(field).and_then(JsonValue::as_str),
                        Some("json_processing_limit")
                    );
                }
            }
        }
    }
}
