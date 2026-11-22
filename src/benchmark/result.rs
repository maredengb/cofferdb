//! Validate the assembled benchmark report before the runner publishes it.
//! Individual rows describe measurements. The surrounding dataset, configuration
//! and baseline declarations determine which rows and units make sense together.
//!
//! This check operates on the report already in memory. The dataset loader
//! verifies source files against their digests. The execution journal records
//! which samples actually ran and ties its completion record to the serialized
//! result. Those checks cover evidence that cannot be recovered from the report's
//! field types.
use std::collections::{BTreeMap, BTreeSet};

use super::BenchmarkQueryMode;
use crate::error::{Error, ErrorCode, Result};
use crate::json::JsonValue;
use crate::storage::FORMAT_VERSION;

// The report is assembled by several producers. Validate the value that will
// actually be returned so a valid individual run cannot conceal a disagreement
// with the dataset or the baseline declarations.
pub(crate) fn validate_benchmark_result_contract(result: &JsonValue) -> Result<()> {
    let result_obj = result.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result must be a JSON object",
        )
        .with_detail("field", "benchmark_result")
    })?;
    // Check the top-level vocabulary before looking for missing fields. A misspelled
    // required field should be reported as the unknown key actually present, even
    // though its intended spelling is also missing.
    reject_unknown_benchmark_result_top_level_fields(result_obj)?;
    // This is a presence check. Some required members, such as the explanation for
    // a completed baseline, carry JSON null. Testing a projected string or number
    // here would collapse null, the wrong type and an absent key into the same
    // error.
    for field in [
        // The report schema, producing software and storage format are versioned
        // independently. A report assembled by the current binary still has to
        // identify all three.
        "schema_version",
        "cofferdb_version",
        "git_commit",
        "file_format_version",
        "hardware",
        // result_file is required even for an in-memory report, where the runner
        // supplies null. A path in this member describes the requested
        // destination. Publication occurs after this validator has returned.
        "result_file",
        "dataset_id",
        "dataset",
        // Record the selected profile separately from its numeric settings.
        // Query-level limits may differ from the profile defaults when a query
        // asks for more results.
        "profile",
        "profile_params",
        "engine_params",
        "candidate_counts",
        "durability",
        "index_build_mode",
        "import_context",
        // The summary is built from the primary warm run. The individual cold and
        // warm observations belong in runs, so the two reporting levels need
        // separate validation.
        "metrics",
        "runs",
        "baseline_configs",
        "baseline_results",
        "baseline_status",
        "baseline_status_reason",
    ] {
        if result.get(field).is_none() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark result missing required field {field}"),
            )
            .with_detail("field", field));
        }
    }
    // Validate the report-wide setting before nested import metadata. The nested
    // value is checked against the same vocabulary later. Neither vocabulary check
    // establishes that the two settings are identical.
    let index_build_mode = result
        .get("index_build_mode")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(
        index_build_mode,
        "inline_full_checkpoint" | "deferred_full_checkpoint" | "transient_only"
    ) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result index_build_mode must be a stable enum value",
        )
        .with_detail("field", "index_build_mode"));
    }
    // Establish the top-level types before examining source-dependent records. In
    // particular the text-only counter walk below borrows the metrics object on
    // the strength of the numeric checks here.
    validate_benchmark_version_fields(result)?;
    validate_benchmark_top_level_enums(result)?;
    validate_benchmark_engine_params(result)?;
    validate_benchmark_hardware(result)?;
    // This covers nested numeric and import settings before the validator tries to
    // interpret counts in terms of the dataset. A valid profile name alone does
    // not establish usable numeric parameters.
    validate_benchmark_numeric_surfaces(result)?;
    // Warnings have their own optional envelope. A missing warning or a warning
    // about baseline availability cannot establish whether measurements were
    // taken. That decision comes from the status, reason and result rows below.
    validate_benchmark_warnings(result)?;
    // Baseline status is the report-wide discriminator. The same decision governs
    // the reason field, whether measured rows may be present and which checks
    // require evidence of a completed baseline.
    let baseline_status = result
        .get("baseline_status")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // Only these two statuses have defined measurement and reason rules below.
    // Reject an unfamiliar spelling here so the fallback arm of the reason check
    // cannot admit it.
    if !matches!(baseline_status, "completed" | "not_run") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_status must be completed or not_run",
        )
        .with_detail("field", "baseline_status"));
    }
    // A skipped baseline must explain why no measurements follow. A completed one
    // uses a null reason. Keep both the value and its presence here so an absent
    // field or a non-string reason cannot masquerade as a valid explanation.
    match (
        baseline_status,
        result
            .get("baseline_status_reason")
            .and_then(JsonValue::as_str),
        result.get("baseline_status_reason"),
    ) {
        ("completed", _, Some(JsonValue::Null)) => {}
        ("completed", _, _) => {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "completed benchmark baselines must use null baseline_status_reason",
            )
            .with_detail("field", "baseline_status_reason"));
        }
        ("not_run", Some(reason), _) if !reason.trim().is_empty() => {}
        ("not_run", _, _) => {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "not_run benchmark baselines must include baseline_status_reason",
            )
            .with_detail("field", "baseline_status_reason"));
        }
        _ => {}
    }
    // The dataset description travels with the measurements. A consumer reading
    // just the report must be able to distinguish a missing identity member from
    // an optional artifact deliberately recorded as null.
    let dataset = result
        .get("dataset")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result dataset must be an object",
            )
            .with_detail("field", "dataset")
        })?;
    // Keep both corpus and query embedding identity members present in all modes.
    // The source-specific checks below decide which members carry values, so
    // text-only and vector reports retain the same keys.
    for field in [
        "dataset_id",
        "version",
        "chunking",
        "embedding_model",
        "embedding_dim",
        // Hash the input streams independently in the dataset loader. The report
        // keeps the identities separately so changing judgments or query text need
        // not change the recorded identity of the documents.
        "dataset_manifest_sha256",
        "documents_sha256",
        "chunks_sha256",
        "embeddings_sha256",
        "queries_sha256",
        "qrels_sha256",
        // These fields describe embeddings supplied to queries. They are distinct
        // from the corpus model and embeddings digest above, even when both use
        // vectors of the same width.
        "query_embedding_source",
        "query_embedding_model",
        "query_embedding_dim",
        "query_embeddings_sha256",
    ] {
        if dataset.get(field).is_none() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark result dataset missing required field {field}"),
            )
            .with_detail("field", format!("dataset.{field}")));
        }
    }
    // Validate identity before using the declared source to interpret the rows. An
    // unknown or incomplete source description must fail even if the report
    // happens to contain only text queries.
    validate_benchmark_dataset_identity(result, dataset)?;
    // Check all three arrays before iterating any of them. Lookup defaults in the
    // source-specific checks are convenient once the array type is known, but an
    // object in place of an array would otherwise disappear as though it were
    // empty.
    if !matches!(result.get("runs"), Some(JsonValue::Array(_))) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result runs must be an array",
        )
        .with_detail("field", "runs"));
    }
    if !matches!(result.get("baseline_configs"), Some(JsonValue::Array(_))) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_configs must be an array",
        )
        .with_detail("field", "baseline_configs"));
    }
    if !matches!(result.get("baseline_results"), Some(JsonValue::Array(_))) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_results must be an array",
        )
        .with_detail("field", "baseline_results"));
    }
    // A skipped baseline may retain its configuration. A row containing all zero
    // scores or timings would still claim a measurement, so not_run requires an
    // empty result array even when a reason is present.
    if baseline_status == "not_run"
        && result
            .get("baseline_results")
            .and_then(JsonValue::as_array)
            .map(|results| !results.is_empty())
            .unwrap_or(false)
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_results must be empty when baseline_status is not_run",
        )
        .with_detail("field", "baseline_results")
        .with_detail("baseline_status", "not_run"));
    }
    let query_embedding_source = result
        .get("dataset")
        .and_then(|dataset| dataset.get("query_embedding_source"))
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // A dataset with no query vectors can produce only text measurements. Check the
    // engine runs, baseline results and baseline declarations as well as the numeric
    // metrics; a plausible top-level summary could hide a vector run in one of those
    // arrays.
    if query_embedding_source == "none" {
        // The two vector-consuming modes are forbidden even when a run reports no
        // hits or zero distance operations. Row mode and the counters provide
        // independent evidence about the work represented.
        let has_vector_run = result
            .get("runs")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .any(|run| {
                matches!(
                    run.get("query_mode").and_then(JsonValue::as_str),
                    Some("vector_exact") | Some("hybrid")
                )
            });
        // Use a mode-conflict error when a recognized vector mode contradicts the
        // source. An unknown query-mode spelling falls through to row validation,
        // where it can be diagnosed as an enum error.
        if has_vector_run {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not contain vector or hybrid runs",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", "runs[].query_mode"));
        }
        // Baseline measurements live in their own array. A text-only native run
        // can be paired with a baseline row from another engine, so a scan of
        // native rows alone cannot establish the dataset constraint.
        let has_vector_baseline_result = result
            .get("baseline_results")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .any(|run| {
                matches!(
                    run.get("query_mode").and_then(JsonValue::as_str),
                    Some("vector_exact") | Some("hybrid")
                )
            });
        if has_vector_baseline_result {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not contain vector or hybrid baseline results",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", "baseline_results[].query_mode"));
        }
        // Configurations can survive a skipped baseline. Inspect their declared
        // input requirement as well as the rows that actually ran, including when
        // baseline_results is empty.
        let has_embedding_baseline = result
            .get("baseline_configs")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .any(|baseline| {
                baseline
                    .get("requires_query_embedding")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false)
            });
        if has_embedding_baseline {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not include embedding baselines",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", "baseline_configs[].requires_query_embedding"));
        }
        // A row labeled text can still contain copied vector counters. Inspect the
        // recorded work after checking the declared modes so these two forms of
        // inconsistency have distinct field paths.
        validate_none_source_has_no_vector_metrics(result)?;
    }
    // Validate each row's shape before checking relationships between arrays. The
    // link and coverage checks can then use a config name, engine and query count as
    // already validated fields.
    // A skipped baseline may still declare its requested modes and cache states,
    // with unavailable version or verification date represented by null.
    // Completion requires those provenance fields before any measured rows are
    // matched to the configuration.
    validate_benchmark_baseline_configs(result, baseline_status == "completed")?;
    validate_benchmark_run_array(result, "runs")?;
    validate_benchmark_run_array(result, "baseline_results")?;
    // Link by configuration name only after parsing both arrays. This checks the
    // engine and per-query count associated with that name, which a comparison of
    // array lengths cannot establish.
    validate_benchmark_baseline_links(result, baseline_status == "completed")?;
    // Pair the cache states within each observed group. An empty runs array
    // contributes no native group to this test. Baseline rows must exist when
    // completion is claimed. The runner's journal separately checks that all
    // planned native samples finished.
    validate_benchmark_cache_states(result, "runs", false)?;
    validate_benchmark_cache_states(result, "baseline_results", baseline_status == "completed")?;
    // Cold and warm observations in a single mode can form a valid pair while
    // leaving whole modes unmeasured. Completed SQLite coverage has to include the
    // required modes for the dataset's vector source as well.
    validate_benchmark_p0_sqlite_baseline_coverage(result, baseline_status == "completed")?;
    Ok(())
}

// The public report is a closed vocabulary. When introducing a measurement, update
// the producing code and the validation for its exact owning object. Adding a
// similarly named key to a nested metrics list does not admit a new top-level
// member.
fn reject_unknown_benchmark_result_top_level_fields(obj: &crate::json::JsonObject) -> Result<()> {
    for field in obj.keys() {
        if ![
            "schema_version",
            "cofferdb_version",
            "git_commit",
            "file_format_version",
            // Hardware and result-location fields describe the run environment.
            // The vocabulary includes conditional output metadata such as path and
            // dataset_path that are absent from the required-key list.
            "hardware",
            "status",
            "path",
            "result_file",
            "dataset_path",
            "dataset_id",
            "collection",
            "dataset",
            "profile",
            "profile_params",
            "engine_params",
            "candidate_counts",
            // The import description and ranking metrics report different phases.
            // The runner may report the import duration alongside retrieval work,
            // so field names are checked within their own nested objects.
            "durability",
            "index_build_mode",
            "import_context",
            "metrics",
            "warnings",
            "runs",
            "baseline_configs",
            "baseline_results",
            "baseline_status",
            "baseline_status_reason",
        ]
        .contains(&field.as_str())
        {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result contains an unknown top-level public field",
            )
            .with_detail("field", field));
        }
    }
    Ok(())
}

// A result can carry the same kind of work counter at report, run and per-query
// scope. Check each copy directly. A zero total at one scope does not establish
// that the detailed records agree with the declared text-only source.
fn validate_none_source_has_no_vector_metrics(result: &JsonValue) -> Result<()> {
    let metrics = result
        .get("metrics")
        .and_then(JsonValue::as_object)
        .expect("benchmark metrics object is validated before source contract checks");
    // Start with aggregate metrics, whose object shape the entry point has already
    // checked. Propagate the first conflicting counter with a path into the
    // report.
    reject_nonzero_none_source_vector_metrics(metrics, "metrics")?;

    // Apply the same counter restriction to the main and baseline result arrays.
    // Configuration records describe planned work and have no measurements to
    // inspect here.
    for field in ["runs", "baseline_results"] {
        let Some(runs) = result.get(field).and_then(JsonValue::as_array) else {
            continue;
        };
        // This pass only inspects recognized metric locations. A malformed run is
        // left to the row validator, which can report the actual type error
        // without treating the row as vector work.
        for run in runs {
            let Some(run_obj) = run.as_object() else {
                continue;
            };
            // A run's summary can conflict with its source even if it has no
            // per-query items. Inspect it before attempting the optional traversal
            // of the detail array.
            if let Some(metrics) = run_obj.get("metrics").and_then(JsonValue::as_object) {
                reject_nonzero_none_source_vector_metrics(metrics, &format!("{field}[].metrics"))?;
            }
            // A run need not provide objects at every nested location for this
            // traversal to continue. Missing or ill-typed detail cannot establish
            // a counter conflict, and the full row validator still runs afterward.
            let Some(per_query) = run_obj.get("per_query").and_then(JsonValue::as_array) else {
                continue;
            };
            for item in per_query {
                let Some(item_obj) = item.as_object() else {
                    continue;
                };
                // Some execution counters are published alongside the query
                // identity. The context can hold another copy, so checking either
                // location alone leaves a way for inconsistent measurements to
                // pass.
                if let Some(execution_stats) = item_obj
                    .get("execution_stats")
                    .and_then(JsonValue::as_object)
                {
                    reject_nonzero_none_source_vector_metrics(
                        execution_stats,
                        &format!("{field}[].per_query[].execution_stats"),
                    )?;
                }
                // Follow the recorded query context independently of the outer
                // execution_stats object. A missing outer object must not suppress
                // inspection of a populated inner one.
                if let Some(context_stats) = item_obj
                    .get("query_context")
                    .and_then(JsonValue::as_object)
                    .and_then(|context| context.get("execution_stats"))
                    .and_then(JsonValue::as_object)
                {
                    // For example, zero distance_ops in the run summary cannot
                    // excuse a positive distance_ops inside a query context for a
                    // text-only dataset. Report the path that contains the
                    // positive value.
                    reject_nonzero_none_source_vector_metrics(
                        context_stats,
                        &format!("{field}[].per_query[].query_context.execution_stats"),
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn reject_nonzero_none_source_vector_metrics(
    metrics: &crate::json::JsonObject,
    detail_prefix: &str,
) -> Result<()> {
    // These counters measure vector-specific work. Ordinary text candidates,
    // returned hits and latency are meaningful for a text-only dataset and are
    // outside this check.
    for metric in ["vector_scan_count", "distance_ops"] {
        let value = metrics
            .get(metric)
            .and_then(JsonValue::as_f64)
            .unwrap_or_default();
        // Only a positive numeric counter is evidence of vector work here. Missing
        // and nonnumeric values map to zero in this pass, leaving the shape and
        // range rules for that metric to their own validators.
        if value > 0.0 {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not include vector metrics",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", format!("{detail_prefix}.{metric}")));
        }
    }
    Ok(())
}

// The report has three different version concepts: its JSON schema, the producing
// build and the storage format used in the run. Each has its own comparison rule.
fn validate_benchmark_version_fields(result: &JsonValue) -> Result<()> {
    // The benchmark report schema stays at its own revision when the producing
    // crate changes version. Compare the serialized identifier as a string so a
    // numeric 0.1 cannot pass.
    if result.get("schema_version").and_then(JsonValue::as_str) != Some("0.1") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result schema_version must be 0.1",
        )
        .with_detail("field", "schema_version"));
    }
    // Treat producer version and commit as recorded provenance strings. This check
    // requires nonblank values without parsing a release number or looking up a
    // Git object.
    for field in ["cofferdb_version", "git_commit"] {
        require_nonempty_string(
            result
                .as_object()
                .expect("benchmark result object is validated by callers"),
            field,
            field,
        )?;
    }
    // The file format is a numeric member tied to the storage implementation's
    // format constant. A valid result schema string by itself says nothing about
    // the format used to create the measured database.
    let file_format_version = result
        .get("file_format_version")
        .and_then(JsonValue::as_f64)
        .unwrap_or_default();
    // Equality with the current format also rejects zero introduced by a missing
    // or nonnumeric value. Changing the storage format constant changes which
    // numeric provenance the active validator accepts.
    if file_format_version != FORMAT_VERSION as f64 {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result file_format_version must match the storage format version",
        )
        .with_detail("field", "file_format_version"));
    }
    Ok(())
}

fn validate_benchmark_top_level_enums(result: &JsonValue) -> Result<()> {
    // Use the profile parser shared with the runtime. A spelling a producer cannot
    // select as a resource profile must not appear as the label for a measured
    // configuration.
    let profile = result
        .get("profile")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !is_stable_profile_value(profile) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result profile must be a stable profile enum value",
        )
        .with_detail("field", "profile"));
    }
    // Durability uses the public report spellings. Keep this check independent of
    // the profile: neither a fast run nor a commit-durable run implies which
    // profile produced it.
    let durability = result
        .get("durability")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(durability, "commit" | "fast") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result durability must be commit or fast",
        )
        .with_detail("field", "durability"));
    }
    Ok(())
}

fn is_stable_profile_value(value: &str) -> bool {
    crate::ResourceProfile::parse(value).is_some()
}

// Score labels depend on both the retrieval mode and the engine. Preserve that
// distinction when the row validator compares the published score_kind,
// particularly for SQLite text ranking.
fn expected_benchmark_score_kind(
    engine: &str,
    query_mode: BenchmarkQueryMode,
) -> Option<&'static str> {
    match engine {
        // CofferDB and SQLite have known score labels for each mode. For another
        // baseline engine, the row still needs its own score label and a matching
        // named configuration. There is no built-in label to compare with here.
        "cofferdb" => Some(query_mode.score_kind()),
        "sqlite" => Some(query_mode.sqlite_score_kind()),
        _ => None,
    }
}

fn validate_benchmark_dataset_identity(
    result: &JsonValue,
    dataset: &crate::json::JsonObject,
) -> Result<()> {
    // Check nested keys independently of the top-level list. This object contains
    // the corpus inputs, query-vector provenance and tokenizer identifying how
    // text was prepared for the measured run.
    reject_unknown_benchmark_result_fields(
        dataset,
        "dataset",
        &[
            "dataset_id",
            "version",
            "chunking",
            "tokenizer",
            "embedding_model",
            "embedding_dim",
            "dataset_manifest_sha256",
            "documents_sha256",
            "chunks_sha256",
            "embeddings_sha256",
            "queries_sha256",
            "qrels_sha256",
            "query_embedding_source",
            "query_embedding_model",
            "query_embedding_dim",
            "query_embeddings_sha256",
        ],
    )?;
    // Tokenizer is required even though it is outside the caller's presence list.
    // Its validator parses the full tokenizer configuration and reports errors
    // against the dataset member.
    validate_tokenizer_field(dataset, "dataset")?;
    let top_dataset_id = result
        .get("dataset_id")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let dataset_id = dataset
        .get("dataset_id")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // Use trimming only to reject an empty identifier. The two stored spellings
    // must compare exactly so a grouping key at the report root identifies the
    // same dataset as the detailed metadata.
    if top_dataset_id.trim().is_empty() || top_dataset_id != dataset_id {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result dataset_id must match dataset.dataset_id",
        )
        .with_detail("field", "dataset_id"));
    }
    // The dataset version and these artifact digests are required even for a
    // text-only dataset.
    for field in [
        "version",
        // The loader compared file bytes with their declared SHA-256 values. This
        // report-only check requires nonblank identity strings. Verifying a saved
        // report against an external dataset requires comparing those digests with
        // the actual files.
        "dataset_manifest_sha256",
        "documents_sha256",
        "chunks_sha256",
        "queries_sha256",
        "qrels_sha256",
    ] {
        require_nonempty_string(dataset, field, format!("dataset.{field}"))?;
    }
    // Chunking describes how the corpus was prepared. It is independent of how
    // queries obtain embeddings and must be validated even when the embedding
    // source is none.
    validate_benchmark_dataset_chunking(dataset)?;
    let query_embedding_source = dataset
        .get("query_embedding_source")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // Validate the source enumeration before branching on combinations of nullable
    // metadata. Without that gate an unknown source could be confused with the
    // inline case when no sidecar digest exists.
    if !matches!(query_embedding_source, "none" | "sidecar" | "inline") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result dataset.query_embedding_source must be none, sidecar, or inline",
        )
        .with_detail("field", "dataset.query_embedding_source"));
    }
    // The collection width remains a positive integral field for text-only
    // datasets too. Only the query embedding dimension is nullable under the none
    // source.
    require_positive_integer(dataset, "embedding_dim", "dataset.embedding_dim")?;
    // The embedding source governs which identity fields have meaning. A sidecar has
    // its own artifact hash; inline query vectors are already part of the hashed
    // queries input. Require the corresponding fields to agree with that choice.
    let embedding_model_is_null = matches!(dataset.get("embedding_model"), Some(JsonValue::Null));
    let embeddings_hash_is_null = matches!(dataset.get("embeddings_sha256"), Some(JsonValue::Null));
    let model_is_null = matches!(dataset.get("query_embedding_model"), Some(JsonValue::Null));
    let dim_is_null = matches!(dataset.get("query_embedding_dim"), Some(JsonValue::Null));
    let hash_is_null = matches!(
        dataset.get("query_embeddings_sha256"),
        Some(JsonValue::Null)
    );
    // Nullness is tested against an actual JSON null for each member. An omitted
    // key or an empty string does not declare the absence of an embedding
    // artifact.
    match query_embedding_source {
        "none" => {
            // Text-only manifests omit corpus embedding provenance. The result
            // writes that absence as explicit nulls for both model and digest.
            if !(embedding_model_is_null && embeddings_hash_is_null) {
                return Err(Error::new(
                    ErrorCode::BenchmarkArtifactModeConflict,
                    "benchmark result query_embedding_source none must use null corpus embedding model and hash",
                )
                .with_detail("field", "dataset.embedding_model"));
            }
            if !(model_is_null && dim_is_null && hash_is_null) {
                return Err(Error::new(
                    ErrorCode::BenchmarkArtifactModeConflict,
                    "benchmark result query_embedding_source none must use null model, dim, and hash",
                )
                .with_detail("field", "dataset.query_embedding_model"));
            }
        }
        "sidecar" | "inline" => {
            // Vector queries require identity on both sides of the search. The
            // query and corpus models are recorded independently; their names are
            // each required, while compatibility of dimensions is checked below.
            let model = dataset
                .get("query_embedding_model")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            if model.trim().is_empty() {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result query embedding model must be non-empty",
                )
                .with_detail("field", "dataset.query_embedding_model"));
            }
            // The corpus embedding identity describes chunk vectors, independent
            // of where query vectors were supplied.
            require_nonempty_string(dataset, "embedding_model", "dataset.embedding_model")?;
            require_nonempty_string(dataset, "embeddings_sha256", "dataset.embeddings_sha256")?;
            let embedding_dim = dataset
                .get("embedding_dim")
                .and_then(JsonValue::as_f64)
                .unwrap_or_default();
            let dim = dataset
                .get("query_embedding_dim")
                .and_then(JsonValue::as_f64)
                .unwrap_or_default();
            // The JSON number representation also admits fractions. Reject an
            // invalid query width before testing its equality with the already
            // validated collection width, preserving which field caused the
            // failure.
            if !(dim.is_finite() && dim > 0.0 && dim.fract() == 0.0) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result query embedding dim must be a positive integer",
                )
                .with_detail("field", "dataset.query_embedding_dim"));
            }
            // A well-formed positive width can still describe vectors for a
            // different collection. Compare the two report fields before accepting
            // either inline or sidecar query vectors.
            if embedding_dim != dim {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result embedding_dim must match query_embedding_dim",
                )
                .with_detail("field", "dataset.embedding_dim"));
            }
            // The corpus digest identifies stored embeddings in either vector
            // mode. A sidecar additionally needs an identity for the independent
            // query-vector file.
            if query_embedding_source == "sidecar" {
                let hash = dataset
                    .get("query_embeddings_sha256")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default();
                // Report a missing sidecar digest against the sidecar field
                // itself. A valid queries digest records the query rows but cannot
                // identify a separate vector file.
                if hash.trim().is_empty() {
                    return Err(Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark result sidecar query embeddings hash is required",
                    )
                    .with_detail("field", "dataset.query_embeddings_sha256"));
                }
            } else if !hash_is_null {
                return Err(Error::new(
                    ErrorCode::BenchmarkArtifactModeConflict,
                    "benchmark result inline query embeddings must use null query_embeddings_sha256",
                )
                .with_detail("field", "dataset.query_embeddings_sha256"));
            }
        }
        _ => unreachable!("query_embedding_source enum validated above"),
    }
    Ok(())
}

// Callers supply the enclosing path because names such as metrics recur at several
// levels. The error must identify the particular public object that contains an
// extra field.
fn reject_unknown_benchmark_result_fields(
    obj: &crate::json::JsonObject,
    detail_prefix: &str,
    allowed: &[&str],
) -> Result<()> {
    // The allowlist describes keys that may occur, not which ones must occur.
    // Required, nullable and conditionally required members have distinct checks
    // at the call site.
    for field in obj.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result contains an unknown public field",
            )
            .with_detail("field", format!("{detail_prefix}.{field}")));
        }
    }
    Ok(())
}

fn validate_benchmark_dataset_chunking(dataset: &crate::json::JsonObject) -> Result<()> {
    // An explicit null records a dataset with no chunking description. Omitting
    // the member has a different meaning in the report contract and gets a
    // missing-field diagnostic.
    match dataset.get("chunking") {
        Some(JsonValue::Null) => Ok(()),
        Some(JsonValue::Object(chunking)) => {
            // The loader retains the producer's chunking description. Check the
            // two window settings used by this contract and permit other members
            // to travel with the description. These fields alone cannot establish
            // how the text was split.
            let chunk_size = chunking
                .get("chunk_size")
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark result dataset.chunking requires chunk_size",
                    )
                    .with_detail("field", "dataset.chunking.chunk_size")
                })?;
            // Sizes count whole units, although the JSON API exposes a
            // floating-point number. Check finiteness and integrality before using
            // the value as the upper bound for overlap.
            if !(chunk_size.is_finite() && chunk_size > 0.0 && chunk_size.fract() == 0.0) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result dataset.chunking.chunk_size must be a positive integer",
                )
                .with_detail("field", "dataset.chunking.chunk_size"));
            }
            // Validate overlap separately before comparing it against the size.
            // This keeps a negative or fractional value from being accepted just
            // because it is below the upper bound.
            let chunk_overlap = chunking
                .get("chunk_overlap")
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark result dataset.chunking requires chunk_overlap",
                    )
                    .with_detail("field", "dataset.chunking.chunk_overlap")
                })?;
            if !(chunk_overlap.is_finite() && chunk_overlap >= 0.0 && chunk_overlap.fract() == 0.0)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result dataset.chunking.chunk_overlap must be a non-negative integer",
                )
                .with_detail("field", "dataset.chunking.chunk_overlap"));
            }
            // A zero overlap is valid. Equal or larger overlap would leave no
            // forward step between successive chunks, even if both parameters
            // individually passed their integer checks.
            if chunk_overlap >= chunk_size {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result dataset.chunking.chunk_overlap must be smaller than chunk_size",
                )
                .with_detail("field", "dataset.chunking.chunk_overlap"));
            }
            Ok(())
        }
        // A null is handled by its own branch above. Other JSON scalar types and
        // arrays cannot represent chunk settings even if they happen to carry a
        // plausible count.
        Some(_) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result dataset.chunking must be an object or null",
        )
        .with_detail("field", "dataset.chunking")),
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            "benchmark result dataset missing required field chunking",
        )
        .with_detail("field", "dataset.chunking")),
    }
}

// Hardware describes the environment reported by the producing runner. A saved
// report can be read on a different host. This check operates on its recorded
// labels and capacity without probing the reader's CPU, filesystem or OS.
fn validate_benchmark_hardware(result: &JsonValue) -> Result<()> {
    let hardware = result
        .get("hardware")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result hardware must be an object",
            )
            .with_detail("field", "hardware")
        })?;
    // The public report has one description each for CPU, storage and OS. Exclude
    // split aliases here even when a combined field is present, so the same run
    // cannot carry conflicting descriptions.
    reject_unknown_benchmark_result_fields(
        hardware,
        "hardware",
        &["cpu", "memory_gb", "storage", "os"],
    )?;
    // CPU, storage and OS all need useful display text. The checks that follow
    // look for additional information in the CPU and OS strings only.
    for field in ["cpu", "storage", "os"] {
        let value = hardware
            .get(field)
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        if value.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result hardware.{field} must be a non-empty string"),
            )
            .with_detail("field", format!("hardware.{field}")));
        }
    }
    // The allowlist above already rejects these split names. For the current field
    // set, a report with cpu_model therefore fails as an unknown hardware field
    // before reaching this compatibility check.
    for field in [
        "cpu_model",
        "cpu_cores",
        "storage_type",
        "os_name",
        "os_version",
    ] {
        if hardware.get(field).is_some() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result hardware.{field} is not part of the v0.1 public hardware schema"),
            )
            .with_detail("field", format!("hardware.{field}")));
        }
    }
    // The runner combines a model description with the parallelism available to
    // its process. The schema checks for an ASCII digit and the fragment core. A
    // producer can satisfy that description even when the model probe fell back to
    // an unknown label.
    let cpu = hardware
        .get("cpu")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !(cpu.to_ascii_lowercase().contains("core") && cpu.chars().any(|ch| ch.is_ascii_digit())) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result hardware.cpu must include CPU model and core count",
        )
        .with_detail("field", "hardware.cpu"));
    }
    // The OS version convention is similarly permissive: one ASCII digit anywhere
    // in the description is enough. Do not infer that particular version
    // components have been parsed here.
    let os = hardware
        .get("os")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !os.chars().any(|ch| ch.is_ascii_digit()) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result hardware.os must include operating system name and version",
        )
        .with_detail("field", "hardware.os"));
    }
    // This is host capacity, separate from the runtime observations in
    // metrics.memory_peak_mb. The producer may leave capacity unknown even when it
    // measured a run. Accept a finite positive fractional capacity or explicit
    // null.
    match hardware.get("memory_gb") {
        Some(JsonValue::Null) => Ok(()),
        Some(value) => {
            let memory_gb = value.as_f64().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result hardware.memory_gb must be a number or null",
                )
                .with_detail("field", "hardware.memory_gb")
            })?;
            if memory_gb.is_finite() && memory_gb > 0.0 {
                Ok(())
            } else {
                Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result hardware.memory_gb must be positive when present",
                )
                .with_detail("field", "hardware.memory_gb"))
            }
        }
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            "benchmark result hardware missing memory_gb",
        )
        .with_detail("field", "hardware.memory_gb")),
    }
}

// Warnings carry explanatory diagnostics, such as an unavailable baseline. They
// give readers context even if no measurements could be taken. Later status and
// run checks determine whether baseline evidence exists.
fn validate_benchmark_warnings(result: &JsonValue) -> Result<()> {
    // Warnings are optional at the report root. Once the key is present it must
    // contain an array, including when the producer has no warnings to attach.
    let Some(warnings) = result.get("warnings") else {
        return Ok(());
    };
    // When present, keep warnings as a list of records. Empty is valid. A string
    // containing a human-readable explanation would lose the code and reason
    // fields that readers inspect.
    let warnings = warnings.as_array().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result warnings must be an array",
        )
        .with_detail("field", "warnings")
    })?;
    for warning in warnings {
        let obj = warning.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result warnings entries must be objects",
            )
            .with_detail("field", "warnings[]")
        })?;
        // Each warning has a closed set of keys. A reason key with null represents
        // an unavailable explanation. Omitting the key is a missing-field error
        // even if the accompanying message is useful.
        reject_unknown_benchmark_result_fields(obj, "warnings[]", &["code", "message", "reason"])?;
        require_nonempty_string(obj, "code", "warnings[].code")?;
        require_nonempty_string(obj, "message", "warnings[].message")?;
        require_string_or_null(obj, "reason", "warnings[].reason")?;
    }
    Ok(())
}

// Check the engine description independently of its measured results. The report
// must identify the retrieval algorithms and tokenizer well enough for readers to
// interpret the scores.
fn validate_benchmark_engine_params(result: &JsonValue) -> Result<()> {
    let params = benchmark_object(result, "engine_params")?;
    // The schema closes each object separately. Rejecting extra top-level engine
    // groups does not check the settings inside the text, vector or fusion groups.
    reject_unknown_benchmark_result_fields(
        params,
        "engine_params",
        &["text_index", "vector_index", "fusion"],
    )?;
    let text_index = require_object_field(params, "text_index", "engine_params.text_index")?;
    // Treat the text_index group as closed separately from its nested tokenizer
    // and BM25 settings. This detects a setting placed in the wrong object even if
    // its spelling is recognized elsewhere.
    reject_unknown_benchmark_result_fields(
        text_index,
        "engine_params.text_index",
        &["index_kind", "tokenizer", "bm25"],
    )?;
    // Require the stable text index identifier. A report cannot substitute a label
    // for another lexical algorithm while retaining the expected BM25 parameter
    // fields.
    require_string_in(
        text_index,
        "index_kind",
        "engine_params.text_index.index_kind",
        &["text_bm25"],
    )?;
    // The tokenizer belongs to the text index's reproducibility metadata. Check
    // its declared identity and let the core tokenizer parser validate its
    // configuration.
    let tokenizer = require_object_field(
        text_index,
        "tokenizer",
        "engine_params.text_index.tokenizer",
    )?;
    // Reject additional tokenizer keys at the report path before consulting the
    // product tokenizer parser. A later parser error can instead name a mismatch
    // inside the tokenizer configuration.
    reject_unknown_benchmark_result_fields(
        tokenizer,
        "engine_params.text_index.tokenizer",
        &["name", "version", "profile", "language_hint", "config"],
    )?;
    // Use the index's own supported name and version. Copying a benchmark-only
    // default here would let accepted reports diverge from the tokenizer the core
    // implements.
    require_string_in(
        tokenizer,
        "name",
        "engine_params.text_index.tokenizer.name",
        &[crate::index::text::DEFAULT_TOKENIZER_NAME],
    )?;
    // Version and name identify the tokenizer together. The full tokenizer
    // definition is checked against the product's immutable default immediately
    // afterward.
    require_string_in(
        tokenizer,
        "version",
        "engine_params.text_index.tokenizer.version",
        &[crate::index::text::DEFAULT_TOKENIZER_VERSION],
    )?;
    // Pass the complete object through the core parser. Valid name and version
    // tokens alone do not establish that the supplied profile, language or config
    // is usable.
    crate::TokenizerConfig::from_json(&JsonValue::Object(tokenizer.clone()))?;
    // The BM25 block is required even if a particular query mode uses no text. It
    // describes the engine configuration shared by the report's runs.
    let bm25 = require_object_field(text_index, "bm25", "engine_params.text_index.bm25")?;
    reject_unknown_benchmark_result_fields(bm25, "engine_params.text_index.bm25", &["k1", "b"])?;
    // Check finite, strictly positive numerical values for both parameters. This
    // report check imposes no upper bound on b and does not require locally
    // generated default values.
    require_positive_number(bm25, "k1", "engine_params.text_index.bm25.k1")?;
    require_positive_number(bm25, "b", "engine_params.text_index.bm25.b")?;

    // Validate vector configuration as report metadata, including the metric and
    // normalization description. It remains part of the engine declaration for
    // text-only runs.
    let vector_index = require_object_field(params, "vector_index", "engine_params.vector_index")?;
    reject_unknown_benchmark_result_fields(
        vector_index,
        "engine_params.vector_index",
        &[
            "index_kind",
            "distance_metric",
            "vectors_normalized",
            "normalization_method",
        ],
    )?;
    // An exact f32 index has its own report identifier. A different index type
    // cannot use this schema merely by naming an accepted distance metric.
    require_string_in(
        vector_index,
        "index_kind",
        "engine_params.vector_index.index_kind",
        &["vector_exact_f32"],
    )?;
    // The report uses canonical metric names. An input alias such as l2 can select
    // Euclidean distance at collection creation, but the result must spell that
    // metric euclidean.
    require_string_in(
        vector_index,
        "distance_metric",
        "engine_params.vector_index.distance_metric",
        &["cosine", "dot_product", "euclidean"],
    )?;
    // Require both a Boolean normalization flag and a nonblank description. This
    // is a shape check on each field; the method label is not compared against an
    // enum here.
    require_bool(
        vector_index,
        "vectors_normalized",
        "engine_params.vector_index.vectors_normalized",
    )?;
    // Keep the method field readable even when vectors_normalized is false. A
    // missing description is never inferred from the Boolean when reading a saved
    // result.
    require_nonempty_string(
        vector_index,
        "normalization_method",
        "engine_params.vector_index.normalization_method",
    )?;

    // An engine-local score scale allows different engines to calculate scores
    // differently. The report must name that scale before a consumer interprets the
    // numbers alongside weighted fusion.
    let fusion = require_object_field(params, "fusion", "engine_params.fusion")?;
    // The fusion object admits only an algorithm and a score-scale marker.
    // Query-specific weights belong to run context and are not accepted as
    // engine-wide properties here.
    reject_unknown_benchmark_result_fields(
        fusion,
        "engine_params.fusion",
        &["algorithm", "score_scale"],
    )?;
    // A weighted sum identifies the combination rule. Its presence does not make
    // numerical scores interchangeable across engines, hence the separate
    // score_scale member.
    require_string_in(
        fusion,
        "algorithm",
        "engine_params.fusion.algorithm",
        &["weighted_sum"],
    )?;
    // The per-query records carry an engine-specific score kind and mark scores as
    // noncomparable. For example, FTS5 and CofferDB can return the same ordered
    // chunk IDs with different numeric scores.
    require_string_in(
        fusion,
        "score_scale",
        "engine_params.fusion.score_scale",
        &["engine_local"],
    )?;
    Ok(())
}

// Distinguish configured limits, measured values and optional measurements. Each
// has a different definition of a valid zero or a valid null.
fn validate_benchmark_numeric_surfaces(result: &JsonValue) -> Result<()> {
    let profile_params = benchmark_object(result, "profile_params")?;
    reject_unknown_benchmark_result_fields(
        profile_params,
        "profile_params",
        &[
            "bm25_candidates",
            "vector_candidates",
            "fusion_candidates",
            "cache_budget_mb",
        ],
    )?;
    // Each profile value describes positive capacity, including the cache budget
    // expressed in megabytes. The result validator checks integral JSON numbers
    // here.
    for field in [
        "bm25_candidates",
        "vector_candidates",
        "fusion_candidates",
        "cache_budget_mb",
    ] {
        require_positive_integer(profile_params, field, format!("profile_params.{field}"))?;
    }

    // The top-level counts repeat the profile's three candidate limits under
    // shorter names. Validate this object's keys and numeric types before
    // reconciling those pairs.
    let candidate_counts = benchmark_object(result, "candidate_counts")?;
    reject_unknown_benchmark_result_fields(
        candidate_counts,
        "candidate_counts",
        &["bm25", "vector", "fusion"],
    )?;
    for field in ["bm25", "vector", "fusion"] {
        require_positive_integer(candidate_counts, field, format!("candidate_counts.{field}"))?;
    }
    // The top-level candidate counts describe the configured profile. Cross-check
    // the duplicate values here. Individual queries can raise their limits to
    // satisfy top_k, which is checked separately in the per-query context.
    // For example, a nominal BM25 limit of 50 stays 50 in both maps when a query
    // asks for 80 hits. Its enabled text route records the raised limit in that
    // query's context.
    for (candidate_field, profile_field) in [
        ("bm25", "bm25_candidates"),
        ("vector", "vector_candidates"),
        ("fusion", "fusion_candidates"),
    ] {
        // Both fields in this pair have already passed positive-integer
        // validation. The fallback in these lookups is not a default for a missing
        // candidate budget.
        let candidate_value = candidate_counts
            .get(candidate_field)
            .and_then(JsonValue::as_f64)
            .unwrap_or_default();
        // Read the corresponding value using the profile's longer key. The metric
        // name in the short candidate_counts map is not itself a profile_params
        // key.
        let profile_value = profile_params
            .get(profile_field)
            .and_then(JsonValue::as_f64)
            .unwrap_or_default();
        // Compare the stored numeric values after integral validation. The error
        // names both paths so either side of a copied configuration can be
        // corrected.
        if (candidate_value - profile_value).abs() > f64::EPSILON {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result candidate_counts must match profile_params",
            )
            .with_detail("field", format!("candidate_counts.{candidate_field}"))
            .with_detail("profile_field", format!("profile_params.{profile_field}")));
        }
    }

    // The runner copies its primary warm run's retrieval metrics here, choosing
    // hybrid when available. Import time and index size are added separately.
    // Averaging all cold and warm runs would give this section a different
    // meaning.
    let metrics = benchmark_object(result, "metrics")?;
    reject_unknown_benchmark_result_fields(
        metrics,
        "metrics",
        &[
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "p95_ms",
            "vector_scan_count",
            "distance_ops",
            "fusion_candidate_count",
            "import_time_ms",
            "index_size_mb",
            "memory_peak_mb",
        ],
    )?;
    // This pass checks finite, nonnegative observations. It neither recomputes
    // percentiles from individual timings nor bounds Recall and MRR at one. The
    // runner computes retrieval quality from ranked IDs and relevance judgments.
    for field in [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "p95_ms",
        "vector_scan_count",
        "distance_ops",
        "fusion_candidate_count",
        "import_time_ms",
        "index_size_mb",
    ] {
        // A measured zero is allowed for vector work when the text channel ran
        // alone. Another validation pass enforces the text-only dataset's
        // zero-vector-work constraint.
        require_nonnegative_number(metrics, field, format!("metrics.{field}"))?;
    }
    // A platform without a peak-memory reading still reports the field with a null
    // value. Numeric zero remains a measured value and is checked as such.
    require_nonnegative_number_or_null(metrics, "memory_peak_mb", "metrics.memory_peak_mb")?;

    // Validate the description of the import whose cost contributed to this
    // report. The tokenizer is parsed before checking the remaining import-context
    // fields.
    let import_context = benchmark_object(result, "import_context")?;
    // A benchmark can be read on a different machine than the one that imported
    // the dataset. Its tokenizer identity is validated against the known
    // definition rather than with a database read.
    validate_tokenizer_field(import_context, "import_context")?;
    // The nested import context admits its own tokenizer and summary. Fields
    // outside this vocabulary cannot introduce another source of timing or import
    // policy.
    reject_unknown_benchmark_result_fields(
        import_context,
        "import_context",
        &[
            "batch_size",
            "index_build_mode",
            "import_time_ms",
            "import_command",
            "manifest_path",
            "database_path",
            "on_conflict",
            "summary",
            "durability",
            "tokenizer",
        ],
    )?;
    // Keep the command, manifest and database locators as nonblank report strings.
    // Validation of a saved result need not resolve these paths on the machine
    // reading it.
    require_nonempty_string(
        import_context,
        "import_command",
        "import_context.import_command",
    )?;
    require_nonempty_string(
        import_context,
        "manifest_path",
        "import_context.manifest_path",
    )?;
    require_nonempty_string(
        import_context,
        "database_path",
        "import_context.database_path",
    )?;
    // Interpret conflict policy using the stable import spellings. Missing or
    // nonstring input cannot silently select the operational default for a
    // reconstructed report.
    let on_conflict = import_context
        .get("on_conflict")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(on_conflict, "error" | "skip" | "replace") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.on_conflict must be error, skip, or replace",
        )
        .with_detail("field", "import_context.on_conflict"));
    }
    // Batch size and import time are recorded numeric observations here. This
    // check accepts zero and does not apply the profile budget's strictly positive
    // integer rule.
    require_nonnegative_number(import_context, "batch_size", "import_context.batch_size")?;
    // The runner writes the same elapsed observation here and in the top-level
    // metrics. The fields are checked independently, so a consumer comparing
    // measurements cannot infer equality from successful schema validation.
    require_nonnegative_number(
        import_context,
        "import_time_ms",
        "import_context.import_time_ms",
    )?;
    // Durability records the write behavior used during import. Only the two
    // reported modes are accepted even when no later benchmark query writes to the
    // database.
    let durability = import_context
        .get("durability")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // This describes the import operation, which has a separate durability field
    // from the enclosing benchmark. Validate the embedded field where it is
    // recorded.
    if !matches!(durability, "commit" | "fast") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.durability must be commit or fast",
        )
        .with_detail("field", "import_context.durability"));
    }
    // Keep the index-build timing policy as a stable enum since import and
    // first-query costs can differ across these paths. Reject unrecognized
    // spellings before trusting the report.
    let import_index_build_mode = import_context
        .get("index_build_mode")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // The transient mode is still a valid report setting although it does not
    // produce the same persistent index as a full checkpoint. Preserve it as a
    // distinct spelling.
    if !matches!(
        import_index_build_mode,
        "inline_full_checkpoint" | "deferred_full_checkpoint" | "transient_only"
    ) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.index_build_mode must be a stable enum value",
        )
        .with_detail("field", "import_context.index_build_mode"));
    }
    // The import summary is an embedded object. Its counts and warnings have their
    // own producer. Even an empty object satisfies this check, which establishes
    // only the type at this path.
    if import_context
        .get("summary")
        .and_then(JsonValue::as_object)
        .is_none()
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.summary must be an object",
        )
        .with_detail("field", "import_context.summary"));
    }
    Ok(())
}

// Borrow a required top-level report section. Both a missing field and a nonobject
// section fail before its individual members are accessed.
fn benchmark_object<'a>(
    result: &'a JsonValue,
    field: &'static str,
) -> Result<&'a crate::json::JsonObject> {
    result
        .get(field)
        .and_then(JsonValue::as_object)
        // Use the section name as the diagnostic field before a nested path
        // exists. A scalar at metrics, for example, fails here rather than
        // appearing to be a missing metric.
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field} must be an object"),
            )
            .with_detail("field", field)
        })
}

// Accept zero for observations such as counters and elapsed times. Preserve a
// different error category for an absent measurement and a present value of the
// wrong type.
fn require_nonnegative_number(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj.get(field).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    // Use the JSON numeric type directly. A string containing decimal digits
    // cannot substitute for a measurement even when it would round to the same
    // binary64 value.
    let number = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a number"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    // Check finiteness even when validating an already constructed JsonValue. A
    // direct caller can supply values that could never be written as legal JSON
    // numbers.
    if number.is_finite() && number >= 0.0 {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be non-negative"),
        )
        .with_detail("field", detail_field))
    }
}

// Strict positivity is for continuous parameters such as BM25 tuning values.
// Fractional positive numbers are meaningful and pass this helper.
fn require_positive_number(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj.get(field).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    let number = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a number"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    // Finite is a separate condition from greater than zero: positive infinity
    // would otherwise pass the sign test for a programmatically constructed
    // result.
    if number.is_finite() && number > 0.0 {
        Ok(())
    } else {
        // Use the fully qualified field label supplied by the caller. The same
        // numeric helper is shared by settings whose leaf names are meaningful
        // only in their enclosing section.
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be positive"),
        )
        .with_detail("field", detail_field))
    }
}

// Require the member even when its measurement is unavailable. Only explicit null
// denotes absence of a reading; every other present value follows numeric
// validation.
fn require_nonnegative_number_or_null(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    match obj.get(field) {
        Some(JsonValue::Null) => Ok(()),
        Some(_) => require_nonnegative_number(obj, field, detail_field.clone()),
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field)),
    }
}

// Validate the JSON number already held by the report. This checks that its finite
// positive binary64 value has no fractional part; it does not recover an integer
// from the original token.
fn require_positive_integer(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj.get(field).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    // Read a JSON number without coercing a quoted count. Permitting
    // numeric-looking strings would change the wire type accepted for profile and
    // workload sizes.
    let number = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a number"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    // Check the fractional part along with the sign. A value such as 0.5 is a
    // legitimate magnitude but cannot specify a number of candidates.
    if number.is_finite() && number > 0.0 && number.fract() == 0.0 {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a positive integer"),
        )
        .with_detail("field", detail_field))
    }
}

// Match the recorded enum token exactly against this field's allowed spellings.
// Missing, nonstring and unknown values all fail the stable-value check.
fn require_string_in(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
    allowed: &[&str],
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj
        .get(field)
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // The allowed strings are compared without trimming or changing case. A near
    // match cannot identify a stable format value.
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a stable value"),
        )
        .with_detail("field", detail_field))
    }
}

// Borrow a nested object while retaining its qualified field name for diagnostics.
// Sibling groups can use the same raw key with different contracts.
fn require_object_field<'a>(
    obj: &'a crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<&'a crate::json::JsonObject> {
    let detail_field = detail_field.into();
    obj.get(field)
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {detail_field} must be an object"),
            )
            .with_detail("field", detail_field)
        })
}

// Check every declared baseline configuration independently of whether the run was
// completed. Completion determines which fields must carry verification evidence.
fn validate_benchmark_baseline_configs(result: &JsonValue, require_verified: bool) -> Result<()> {
    let configs = result
        .get("baseline_configs")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_configs array is validated before entry schema checks");
    for config in configs {
        let obj = config.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline_configs entries must be objects",
            )
            .with_detail("field", "baseline_configs[]")
        })?;
        // Keep the generic baseline descriptor and its engine-specific config
        // object separate. Each has its own allowlist and neither can absorb an
        // unknown field from the other.
        reject_unknown_benchmark_result_fields(
            obj,
            "baseline_configs[]",
            &[
                "name",
                "version",
                "run_mode",
                "mode",
                "requires_query_embedding",
                "top_k",
                "cache_states",
                "config_source",
                "checked_at",
                "config",
            ],
        )?;
        // The name acts as the join key for baseline_results. The embedded engine
        // is a separate property that is checked after the individual result rows
        // have been validated.
        require_nonempty_string(obj, "name", "baseline_configs[].name")?;
        // A result may describe a baseline it could not run. Validate the
        // configuration in either case, but require the verification evidence
        // appropriate to a completed run only when the baseline was executed.
        // This flag comes from overall baseline_status, not from an individual
        // config's self-reported status. An entry cannot avoid the evidence checks
        // by labeling itself not_run.
        if require_verified {
            require_nonempty_string(obj, "version", "baseline_configs[].version")?;
        } else {
            // For a baseline that was not run, an unavailable version is reported
            // with explicit null. If a version is supplied it must still be a
            // nonblank string.
            require_string_or_null(obj, "version", "baseline_configs[].version")?;
        }
        // The descriptor carries both a run-mode label and a benchmark-mode label.
        // This general shape check only requires nonblank strings; specialized
        // coverage rules follow.
        require_nonempty_string(obj, "run_mode", "baseline_configs[].run_mode")?;
        require_nonempty_string(obj, "mode", "baseline_configs[].mode")?;
        require_bool(
            obj,
            "requires_query_embedding",
            "baseline_configs[].requires_query_embedding",
        )?;
        // top_k records the configuration's requested retrieval bound. The
        // completed SQLite coverage check later relates it to the maximum bound
        // observed in the per-query rows.
        require_positive_integer(obj, "top_k", "baseline_configs[].top_k")?;
        // The cache-state field is a collection of declared labels. It is
        // independent of the rows that demonstrate a cold or warm measurement in
        // baseline_results.
        let cache_states = obj
            .get("cache_states")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result baseline_configs[].cache_states must be an array",
                )
                .with_detail("field", "baseline_configs[].cache_states")
            })?;
        // Treat the state declarations as a set for coverage. Order and repeated
        // cold or warm entries do not create additional evidence of a measured
        // run.
        let mut has_cold = false;
        let mut has_warm = false;
        for state in cache_states {
            match state.as_str() {
                Some("cold") => has_cold = true,
                Some("warm") => has_warm = true,
                // A third cache state or a nonstring entry is invalid even if the
                // array already contains both expected states.
                Some(_) | None => {
                    return Err(Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result baseline_configs[].cache_states entries must be cold or warm",
                    )
                    .with_detail("field", "baseline_configs[].cache_states"));
                }
            }
        }
        // A configuration must declare both cache states even if the report
        // records that the baseline did not run. Actual completed rows are checked
        // later.
        if !has_cold || !has_warm {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline_configs[].cache_states must include cold and warm",
            )
            .with_detail("field", "baseline_configs[].cache_states"));
        }
        // The runner records the source of the baseline configuration for readers,
        // for example its SQLite baseline documentation. Reading a report checks
        // the label without opening the referenced file.
        require_nonempty_string(obj, "config_source", "baseline_configs[].config_source")?;
        // An unverified configuration can leave checked_at null, but a nonnull
        // date is validated in either status. A completed configuration must
        // supply it.
        if require_verified {
            require_nonempty_string(obj, "checked_at", "baseline_configs[].checked_at")?;
        } else {
            require_string_or_null(obj, "checked_at", "baseline_configs[].checked_at")?;
        }
        // Check any supplied date even on a skipped run. The date records when the
        // producer checked the configuration and carries no expiry cutoff or
        // requirement to repeat that check when reading the report.
        if let Some(checked_at) = obj.get("checked_at").and_then(JsonValue::as_str) {
            validate_baseline_checked_at(checked_at)?;
        }
        // The enclosing baseline name is for joining runs, whereas this nested
        // object describes how the engine ran. Require the object even for a
        // configuration recorded without measurements.
        let Some(config) = obj.get("config").and_then(JsonValue::as_object) else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline_configs[].config must be an object",
            )
            .with_detail("field", "baseline_configs[].config"));
        };
        // Allow the known engine metadata keys while requiring only the common
        // subset below. A field's presence in this allowlist alone does not make
        // it mandatory.
        reject_unknown_benchmark_result_fields(
            config,
            "baseline_configs[].config",
            &[
                "engine",
                "sqlite_cli",
                "fts",
                "fts_table",
                "fts_tokenizer",
                "fts_score",
                "fusion",
                "filter",
                "status",
                "vector_extension",
                "vector_extension_version",
                "vector_mode",
                "query_count",
                "cache_policy",
                "latency_scope",
            ],
        )?;
        // Only these labels are mandatory strings in the generic engine
        // descriptor. Other allowed keys, such as cache_policy and latency_scope,
        // describe the run. Cold and warm coverage is established separately from
        // measurement rows.
        for field in [
            "engine",
            "sqlite_cli",
            "fts",
            "fts_tokenizer",
            "fts_score",
            "fusion",
            "filter",
            "status",
        ] {
            require_nonempty_string(config, field, format!("baseline_configs[].config.{field}"))?;
        }
        // Link each descriptor's nested status to the top-level baseline status.
        // Otherwise a report could describe an engine as completed inside an
        // overall skipped baseline.
        let expected_status = if require_verified {
            "completed"
        } else {
            "not_run"
        };
        // Report the status expected from the outer result along with the path of
        // the nested disagreement. This identifies contradictory declarations
        // without examining timing rows.
        if config.get("status").and_then(JsonValue::as_str) != Some(expected_status) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline config status must match baseline_status",
            )
            .with_detail("field", "baseline_configs[].config.status")
            .with_detail("expected", expected_status));
        }
        // Both vector capability keys must be present. A text-only baseline can
        // use explicit null, while the specialized SQLite check demands exact
        // labels when embeddings are required.
        require_string_or_null(
            config,
            "vector_extension",
            "baseline_configs[].config.vector_extension",
        )?;
        // A named vector mode describes the baseline's algorithm. Null still
        // reserves the member in the schema for a configuration with no vector
        // path.
        require_string_or_null(
            config,
            "vector_mode",
            "baseline_configs[].config.vector_mode",
        )?;
        // The number of queries must be a positive integer even in the declared
        // configuration of an unexecuted baseline. Completed runs later supply the
        // corresponding per-query rows.
        require_positive_integer(
            config,
            "query_count",
            "baseline_configs[].config.query_count",
        )?;
    }
    Ok(())
}

// checked_at is a calendar date. Append a fixed midnight UTC to reuse the existing
// calendar validator, while retaining the ten-byte requirement for the original
// date alone.
fn validate_baseline_checked_at(value: &str) -> Result<()> {
    let candidate = format!("{value}T00:00:00Z");
    // The date check rejects days beyond the specified month, including February
    // 29 in a non-leap year.
    if value.len() == 10 && crate::time::parse_rfc3339_seconds(&candidate).is_some() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_configs[].checked_at must be a YYYY-MM-DD date",
        )
        .with_detail("field", "baseline_configs[].checked_at"))
    }
}

// Cross-check previously validated objects by baseline name. A valid config row
// and a valid result row can still disagree about the engine or workload size.
fn validate_benchmark_baseline_links(result: &JsonValue, require_completed: bool) -> Result<()> {
    let configs = result
        .get("baseline_configs")
        // The earlier validator has established these array types. Borrow the
        // persisted containers to check their relationships without reconstructing
        // a benchmark run.
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_configs array is validated before baseline link checks");
    let results = result
        .get("baseline_results")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_results array is validated before baseline link checks");

    // Index the configured engine and query count once so cold and warm result
    // rows can reuse the same named configuration.
    let mut engines_by_name = BTreeMap::new();
    let mut query_counts_by_name = BTreeMap::new();
    for config in configs {
        let obj = config
            .as_object()
            .expect("benchmark baseline config objects are validated before link checks");
        let Some(name) = obj.get("name").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some(engine) = obj
            .get("config")
            .and_then(JsonValue::as_object)
            .and_then(|config| config.get("engine"))
            .and_then(JsonValue::as_str)
        else {
            continue;
        };
        // A repeated name replaces the prior engine and query count in these maps.
        // The SQLite coverage pass takes its first matching descriptor. Emit one
        // descriptor per name to keep both checks on the same configuration.
        engines_by_name.insert(name.to_owned(), engine.to_owned());
        // Use the positive numeric query count already checked in the
        // configuration pass. Save the converted count for comparison with array
        // lengths from each matching run.
        let query_count = obj
            .get("config")
            .and_then(JsonValue::as_object)
            .and_then(|config| config.get("query_count"))
            .and_then(JsonValue::as_f64)
            .unwrap_or_default() as usize;
        query_counts_by_name.insert(name.to_owned(), query_count);
    }

    // A completed status cannot be supported by an empty config set. A skipped
    // baseline does not incur this completion-only evidence requirement.
    if require_completed && engines_by_name.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "completed benchmark baselines must include baseline_configs entries",
        )
        .with_detail("field", "baseline_configs"));
    }

    // Resolve every result row to its named configuration and check the engine and
    // query count there. Matching the global dataset alone would allow a row from a
    // different baseline to pass.
    // Collect baseline names across runs. Multiple rows for one name are expected
    // because modes and cache states produce separate measurements.
    let mut result_names = BTreeSet::new();
    for run in results {
        let run_obj = run
            .as_object()
            .expect("benchmark baseline result objects are validated before link checks");
        // Resolve by name before comparing the engine. Multiple named
        // configurations can use the same engine executable with different
        // settings or workloads.
        let Some(name) = run_obj.get("name").and_then(JsonValue::as_str) else {
            continue;
        };
        // Every supplied run must resolve to a declared configuration. An unknown
        // name prevents the following engine and count comparisons from having a
        // meaningful reference.
        let Some(expected_engine) = engines_by_name.get(name) else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark baseline result name must match a baseline config",
            )
            .with_detail("field", "baseline_results[].name")
            .with_detail("baseline_name", name));
        };
        // The engine label is carried by both config and run. Check the exact
        // recorded spelling even if the run otherwise has the expected metrics and
        // query count.
        let engine = run_obj
            .get("engine")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        if engine != expected_engine {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark baseline result engine must match baseline config",
            )
            .with_detail("field", "baseline_results[].engine")
            .with_detail("baseline_name", name));
        }
        // The config's count applies to each run individually. Count the actual
        // per-query array elements instead of accepting a separate aggregate
        // number as evidence.
        if let Some(expected_query_count) = query_counts_by_name.get(name) {
            let actual_query_count = run_obj
                .get("per_query")
                .and_then(JsonValue::as_array)
                .map(|items| items.len())
                .unwrap_or(0);
            // A cold run and a warm run with ten query rows apiece can both
            // satisfy a count of ten even if their IDs differ. Aligning
            // measurements across runs requires the recorded query IDs.
            if actual_query_count != *expected_query_count {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark baseline result per_query count must match baseline config query_count",
                )
                .with_detail("field", "baseline_results[].per_query")
                .with_detail("baseline_name", name)
                .with_detail("expected_query_count", *expected_query_count)
                .with_detail("actual_query_count", actual_query_count));
            }
        }
        // A configuration can have several runs, so record only that this name was
        // observed. Per-mode cache coverage is checked separately from existence
        // of this link.
        result_names.insert(name.to_owned());
    }

    // Check the reverse link for completed reports. Configs without a measured run
    // are only admissible to this link check when completion is not claimed.
    if require_completed {
        for name in engines_by_name.keys() {
            // Check distinct configuration names, not equality of the numbers of
            // configs and results. One configuration can produce many result rows.
            if !result_names.contains(name) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "completed benchmark baseline config must have matching baseline results",
                )
                .with_detail("field", "baseline_results[].name")
                .with_detail("baseline_name", name));
            }
        }
    }
    Ok(())
}

// A completed benchmark has a required SQLite comparison selected by the dataset's
// query embedding source. Other valid baseline names may coexist with that
// required entry.
fn validate_benchmark_p0_sqlite_baseline_coverage(
    result: &JsonValue,
    require_completed: bool,
) -> Result<()> {
    // A skipped baseline has no runs to cover. For a completed baseline the required
    // SQLite modes depend on whether query embeddings exist; require both cache
    // states for each mode in that set.
    if !require_completed {
        return Ok(());
    }
    // The general cache-state check can account only for modes present in the
    // arrays. If a vector-capable comparison omitted both hybrid runs, the
    // remaining groups could all have cold and warm results. The required-mode
    // check below catches that omission.
    let query_embedding_source = result
        .get("dataset")
        .and_then(|dataset| dataset.get("query_embedding_source"))
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // Datasets without query embeddings require only the text comparison. Both
    // supported vector sources require the same three retrieval modes despite
    // storing their query vectors differently.
    let (expected_name, expected_requires_embedding, expected_modes): (&str, bool, &[&str]) =
        if query_embedding_source == "none" {
            ("sqlite-fts5-text-only", false, &["text_bm25"])
        } else {
            (
                "sqlite-fts5-sqlite-vec",
                true,
                &["text_bm25", "vector_exact", "hybrid"],
            )
        };

    // Use the config collection that already passed its general shape checks. This
    // pass adds the named SQLite requirements on top of the generic baseline
    // schema.
    let configs = result
        .get("baseline_configs")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_configs array is validated before P0 coverage checks");
    // Locate the specifically required config by its wire name. A fully measured
    // optional engine still cannot stand in for the required SQLite baseline.
    let Some(config) = configs.iter().find_map(|config| {
        let obj = config.as_object()?;
        (obj.get("name").and_then(JsonValue::as_str) == Some(expected_name)).then_some(obj)
    }) else {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "completed benchmark baselines must include the P0 SQLite baseline config",
        )
        .with_detail("field", "baseline_configs[].name")
        .with_detail("baseline_name", expected_name));
    };
    // Compare the capability flag with the dataset source. The vector-capable
    // SQLite configuration still runs text queries, so the flag does not mean that
    // every run uses a vector.
    if config
        .get("requires_query_embedding")
        .and_then(JsonValue::as_bool)
        != Some(expected_requires_embedding)
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 SQLite baseline requires_query_embedding must match dataset query embedding source",
        )
        .with_detail("field", "baseline_configs[].requires_query_embedding")
        .with_detail("baseline_name", expected_name));
    }
    // The required name and capability flag belong to the outer descriptor.
    // Engine, FTS and vector details are looked up in its inner config object.
    let config_obj = config
        .get("config")
        .and_then(JsonValue::as_object)
        .expect("benchmark baseline config object is validated before P0 coverage checks");
    // Check the nested engine identity in addition to the baseline name. The
    // desired label on an unrelated engine is insufficient to establish comparable
    // coverage.
    if config_obj.get("engine").and_then(JsonValue::as_str) != Some("sqlite") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 baseline config engine must be sqlite",
        )
        .with_detail("field", "baseline_configs[].config.engine")
        .with_detail("baseline_name", expected_name));
    }
    // Both SQLite configurations require the FTS5 text engine. Vector-capable mode
    // adds another channel rather than replacing the text comparison.
    if config_obj.get("fts").and_then(JsonValue::as_str) != Some("fts5") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 baseline config fts engine must be fts5",
        )
        .with_detail("field", "baseline_configs[].config.fts")
        .with_detail("baseline_name", expected_name));
    }
    // For the vector-capable configuration the extension, exact scan and fusion
    // rule are part of the baseline identity. Generic nonblank metadata would
    // permit a different workload here.
    if expected_requires_embedding {
        for (field, expected) in [
            ("vector_extension", "sqlite-vec"),
            ("vector_mode", "exact_scan"),
            ("fusion", "weighted_sum"),
        ] {
            if config_obj.get(field).and_then(JsonValue::as_str) != Some(expected) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "P0 sqlite-vec baseline config must record vector mode and fusion parameters",
                )
                .with_detail("field", format!("baseline_configs[].config.{field}"))
                .with_detail("baseline_name", expected_name));
            }
        }
    }
    // The comparison uses the configured maximum across queries, so retain this
    // bound once while traversing the individually reported query contexts.
    let config_top_k = config
        .get("top_k")
        .and_then(JsonValue::as_f64)
        .expect("benchmark baseline config top_k is validated before P0 coverage checks")
        as usize;

    let results = result
        .get("baseline_results")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_results array is validated before P0 coverage checks");
    // Gather cache states per mode for this specific baseline. A cold text run and a
    // warm vector run cannot together satisfy the cold/warm requirement for either
    // mode.
    let mut cache_states_by_mode: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut observed_max_top_k = 0usize;
    for run in results {
        let run_obj = run
            .as_object()
            .expect("benchmark baseline result objects are validated before P0 coverage checks");
        // Ignore other named baselines when building the required mode and
        // cache-state coverage. Their observations cannot fill a gap in the
        // selected SQLite configuration.
        if run_obj.get("name").and_then(JsonValue::as_str) != Some(expected_name) {
            continue;
        }
        // Fetch the mode and cache state from the same run element. Combining
        // independent global sets of observed modes and states would conceal a
        // missing combination.
        let query_mode = run_obj
            .get("query_mode")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        let cache_state = run_obj
            .get("cache_state")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        // Record observed cache states as a set per mode. Additional rows for a
        // mode do not substitute for one of its missing states.
        cache_states_by_mode
            .entry(query_mode)
            .or_default()
            .insert(cache_state);
        // Rows can carry different requested top_k values. Examine every query in
        // each selected run, not just the first query or the length of its hit
        // list.
        for item in run_obj
            .get("per_query")
            .and_then(JsonValue::as_array)
            .expect("benchmark baseline per_query array is validated before P0 coverage checks")
        {
            // Read top_k from the reported request context. The number of hits can
            // legitimately be smaller when a filter leaves fewer candidates.
            let top_k = item
                .get("query_context")
                .and_then(|context| context.get("top_k"))
                .and_then(JsonValue::as_f64)
                .expect(
                    "benchmark baseline query_context.top_k is validated before P0 coverage checks",
                ) as usize;
            // A repeated cold or warm result can raise the observed maximum but
            // never lowers it. Smaller per-query limits remain valid within this
            // configuration.
            observed_max_top_k = observed_max_top_k.max(top_k);
        }
    }
    // For requests of 5 and 10, the config must report 10 even if filtering leaves
    // fewer than five hits in both runs. A nonzero maximum mismatch is reported
    // here before looking for a missing required mode.
    if observed_max_top_k != 0 && observed_max_top_k != config_top_k {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 baseline config top_k must match the max per-query top_k",
        )
        // The largest per-query request provides the expected bound. Report the
        // separately declared config value as actual so the direction of the
        // comparison is visible.
        .with_detail("field", "baseline_configs[].top_k")
        .with_detail("baseline_name", expected_name)
        .with_detail("expected_top_k", observed_max_top_k)
        .with_detail("actual_top_k", config_top_k));
    }
    // Look for an entirely absent mode in the required SQLite comparison. In the
    // public validation sequence the generic cache-state check has already
    // rejected a reported group that contained only cold or only warm rows.
    for expected_mode in expected_modes {
        let Some(states) = cache_states_by_mode.get(expected_mode) else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "completed P0 SQLite baseline must include all required query modes",
            )
            .with_detail("field", "baseline_results[].query_mode")
            .with_detail("baseline_name", expected_name)
            .with_detail("expected_query_mode", *expected_mode));
        };
        // Cold and warm evidence must belong to this same mode and this same named
        // baseline. The config's cache-state declaration alone cannot supply a
        // measurement.
        if !(states.contains("cold") && states.contains("warm")) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "completed P0 SQLite baseline must include cold and warm results for every required query mode",
            )
            .with_detail("field", "baseline_results[].cache_state")
            .with_detail("baseline_name", expected_name)
            .with_detail("expected_query_mode", *expected_mode));
        }
    }
    Ok(())
}

// Validate the shareable run schema before treating a row as evidence for an
// engine or configuration. A native row explicitly names CofferDB as its engine. A
// baseline row also names a separately declared configuration. After this pass,
// other validators can join runs by configuration and check coverage across modes
// and cache states. Keep these relationships outside the per-query validator so an
// invalid inner row gets its own field path.
fn validate_benchmark_run_array(result: &JsonValue, field: &'static str) -> Result<()> {
    let runs = result
        .get(field)
        .and_then(JsonValue::as_array)
        .expect("benchmark result array fields are validated before run schema checks");
    // Validate each persisted run before inspecting its query list. A malformed
    // run element has an array-entry location and no deeper field context yet.
    for run in runs {
        let run_obj = run.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field} entries must be objects"),
            )
            .with_detail("field", format!("{field}[]"))
        })?;
        // Engine runs and baseline runs share the per-query shape. Baselines
        // additionally need a name to link them to a configuration; keep that field
        // out of the CofferDB run shape.
        let run_allowed: &[&str] = if field == "baseline_results" {
            &[
                "name",
                "engine",
                "query_mode",
                "cache_state",
                "metrics",
                "per_query",
            ]
        } else {
            &[
                "engine",
                "query_mode",
                "cache_state",
                "metrics",
                "per_query",
            ]
        };
        // Reject extra keys before reporting missing ones. This makes the field
        // path identify an unsupported run shape even if another required member
        // is also absent.
        reject_unknown_benchmark_result_fields(run_obj, &format!("{field}[]"), run_allowed)?;
        // Require the common members before checking their values. A present null
        // still goes through the later type and value checks.
        for required in [
            "engine",
            "query_mode",
            "cache_state",
            "metrics",
            "per_query",
        ] {
            // Separate absence from an invalid value at this layer. A metrics key
            // present as an array, for example, reaches the metric object
            // validator below.
            if run_obj.get(required).is_none() {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    format!("benchmark result {field} entry missing {required}"),
                )
                .with_detail("field", format!("{field}[].{required}")));
            }
        }
        // Require a baseline configuration name in the baseline array. Absence, an
        // empty string and a wrong type fail the name check.
        if field == "baseline_results" {
            require_nonempty_string(run_obj, "name", "baseline_results[].name")?;
        }
        // Read the declared engine once for score-label validation below. The
        // nonempty-string check ensures the fallback cannot be treated as a real
        // engine name.
        let engine = run_obj
            .get("engine")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        require_nonempty_string(run_obj, "engine", format!("{field}[].engine"))?;
        // Reserve the main runs array for CofferDB. Baseline engine and
        // configuration correspondence is checked separately.
        if field == "runs" && engine != "cofferdb" {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result runs[].engine must be cofferdb",
            )
            .with_detail("field", "runs[].engine"));
        }
        // Parse the declared run mode once. It identifies the inputs the engine
        // actually used, even when the source query supplied additional fields.
        // Reuse it for both score convention and channel consistency so changing a
        // mode cannot leave these checks using different route rules.
        let query_mode = run_obj
            .get("query_mode")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        // Use the parsed mode for all subsequent checks in this run. The exact
        // input token must be one of the shared mode names.
        let query_mode = BenchmarkQueryMode::parse(query_mode).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result query_mode must be a stable enum value",
            )
            .with_detail("field", format!("{field}[].query_mode"))
        })?;
        // Validate aggregate and individual evidence independently. A stored
        // latency percentile or Recall value can be well formed even when the
        // corresponding per-query list is missing. The overall result needs both,
        // and a shape error in the aggregate is surfaced before inspecting the
        // individual measurements.
        validate_benchmark_run_metrics(run_obj, field)?;
        // Each run must provide its own per-query rows as an array. The structural
        // check precedes the requirement that this array contain a measurement.
        let per_query = run_obj
            .get("per_query")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].per_query must be an array"),
                )
                .with_detail("field", format!("{field}[].per_query"))
            })?;
        // A run with no measured queries cannot provide benchmark evidence for its
        // mode and cache state. Require a nonempty array before inspecting
        // individual items.
        if per_query.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field}[].per_query must not be empty"),
            )
            .with_detail("field", format!("{field}[].per_query")));
        }
        // Per-query diagnostics share the CofferDB and baseline shape. Use the
        // engine and mode from this run to interpret fields without inferring them
        // from a row's result.
        for item in per_query {
            let item_obj = item.as_object().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].per_query entries must be objects"),
                )
                .with_detail("field", format!("{field}[].per_query[]"))
            })?;
            // Keep the allowed per-query field list closed. Nested structures such
            // as query_context and execution_stats get their own validation rather
            // than permitting arbitrary top-level payloads.
            reject_unknown_benchmark_result_fields(
                item_obj,
                &format!("{field}[].per_query[]"),
                &[
                    "query_id",
                    "latency_ms",
                    "ranked_chunk_ids",
                    "score_kind",
                    "score_scale",
                    "score_comparable",
                    "scores",
                    "hit",
                    "execution_stats",
                    "query_context",
                ],
            )?;
            // Check common keys for presence before validating each field's type.
            // A present but ill-typed value can then produce its field-specific
            // error.
            for required in [
                "query_id",
                "latency_ms",
                "ranked_chunk_ids",
                "score_kind",
                "score_scale",
                "score_comparable",
                "scores",
                "hit",
                "execution_stats",
                "query_context",
            ] {
                if item_obj.get(required).is_none() {
                    return Err(Error::new(
                        ErrorCode::MissingRequiredField,
                        format!("benchmark result per_query entry missing {required}"),
                    )
                    .with_detail("field", format!("{field}[].per_query[].{required}")));
                }
            }
            // Require a nonblank label while preserving its exact spelling.
            // Joining these IDs to the source workload and checking its coverage
            // requires a separate comparison.
            require_nonempty_string(
                item_obj,
                "query_id",
                format!("{field}[].per_query[].query_id"),
            )?;
            // Require finite nonnegative latency. Zero remains valid for the
            // result schema and does not by itself establish timer resolution.
            require_nonnegative_number(
                item_obj,
                "latency_ms",
                format!("{field}[].per_query[].latency_ms"),
            )?;
            // Require a score label before interpreting its meaning. The
            // engine/mode table below can then enforce its specific convention
            // where known.
            require_nonempty_string(
                item_obj,
                "score_kind",
                format!("{field}[].per_query[].score_kind"),
            )?;
            // Scores have engine- and mode-specific meanings. Check the declared
            // kind alongside its scale and comparability so a consumer cannot
            // mistake ranks from two engines for directly comparable scores.
            // Score semantics are part of the measurement. For example, an engine
            // can use a raw exact-vector score whose scale is different from that
            // of a fused lexical/vector score. A numeric array with the wrong kind
            // cannot be attributed to the declared mode.
            let score_kind = item_obj
                .get("score_kind")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            // Compare with an expected score label when this engine has a defined
            // mapping. Other engines still must report a nonblank kind and use
            // engine-local score metadata.
            if expected_benchmark_score_kind(engine, query_mode)
                .map(|expected| score_kind != expected)
                .unwrap_or(false)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query score_kind must match engine and query_mode",
                )
                .with_detail("field", format!("{field}[].per_query[].score_kind"))
                .with_detail("engine", engine)
                .with_detail("query_mode", query_mode.as_str()));
            }
            // Require an explicit score-scale string. The fixed engine_local value
            // is checked separately, after the per-query score kind has already
            // been classified.
            require_nonempty_string(
                item_obj,
                "score_scale",
                format!("{field}[].per_query[].score_scale"),
            )?;
            // Require the comparability flag as a boolean before checking its
            // value. A missing flag or a string spelling cannot implicitly mean
            // false.
            require_bool(
                item_obj,
                "score_comparable",
                format!("{field}[].per_query[].score_comparable"),
            )?;
            // The fixed scale marker is required even when scores happen to fall
            // between zero and one. An evaluation consumer may compare rankings or
            // quality metrics across engines without assigning the same meaning to
            // equal raw scores.
            if item_obj.get("score_scale").and_then(JsonValue::as_str) != Some("engine_local") {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query score_scale must be engine_local",
                )
                .with_detail("field", format!("{field}[].per_query[].score_scale")));
            }
            // Do not advertise raw scores as comparable across engines. The
            // required false value is separate from aligning scores with chunk IDs
            // within one run.
            if item_obj
                .get("score_comparable")
                .and_then(JsonValue::as_bool)
                != Some(false)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query score_comparable must be false",
                )
                .with_detail("field", format!("{field}[].per_query[].score_comparable")));
            }
            // The hit flag records the reported outcome against relevance labels.
            // A finite score is neither a relevance label nor a cutoff for this
            // check. The record's boolean must be present independently of the
            // length or content of its score array.
            require_bool(item_obj, "hit", format!("{field}[].per_query[].hit"))?;
            // Require per-query execution counters to be nested in one object. A
            // wrong type fails before individual metrics are read.
            let execution_stats = item_obj
                .get("execution_stats")
                .and_then(JsonValue::as_object)
                // If execution stats are absent or not an object, report the
                // entire nested field. Only an object can be checked for specific
                // missing counters.
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result per_query execution_stats must be an object",
                    )
                    .with_detail("field", format!("{field}[].per_query[].execution_stats"))
                })?;
            // Limit the execution counters to the current schema's vector scans,
            // distance operations and fusion candidates. Additional statistics
            // need an explicit schema change.
            reject_unknown_benchmark_result_fields(
                execution_stats,
                &format!("{field}[].per_query[].execution_stats"),
                &[
                    "vector_scan_count",
                    "distance_ops",
                    "fusion_candidate_count",
                ],
            )?;
            // Require finite nonnegative numeric observations for every supported
            // counter. This validator enforces numeric shape before run-level
            // comparisons.
            for metric in [
                "vector_scan_count",
                "distance_ops",
                "fusion_candidate_count",
            ] {
                require_nonnegative_number(
                    execution_stats,
                    metric,
                    format!("{field}[].per_query[].execution_stats.{metric}"),
                )?;
            }
            // Validate nested metadata before reading top_k or route flags from
            // it. A malformed context fails before rank/score alignment checks and
            // before a well-typed context can be checked against the run's actual
            // mode.
            validate_benchmark_query_context(item_obj, field)?;
            let context = item_obj
                // The previous check established that query_context is present and
                // an object. The mode check can now use its booleans without
                // duplicating field-type validation.
                .get("query_context")
                .and_then(JsonValue::as_object)
                .expect("benchmark per-query context is validated before mode consistency checks");
            // Bind the context's reported channels to this run's declared query
            // mode. A well-formed context can still describe the wrong execution
            // route.
            validate_benchmark_query_context_mode_consistency(context, field, query_mode)?;
            // A score is paired with its chunk by array position. Validate the two
            // arrays together and bound their length by this query's top_k, which
            // may differ from another query in the same run.
            // Check the rank and score vectors by position. For example, ranks [A,
            // B] and scores [0.8, 0.3] associate 0.8 with A. Sorting either array
            // independently changes that evidence even when both lengths stay
            // correct.
            let ranked_chunk_count = item_obj
                .get("ranked_chunk_ids")
                // Preserve array order because position defines association with
                // scores. The check reports a rank-list shape error before testing
                // score cardinality.
                .and_then(JsonValue::as_array)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result ranked_chunk_ids must be an array",
                    )
                    .with_detail("field", format!("{field}[].per_query[].ranked_chunk_ids"))
                })?
                // Check that each rank position holds a string identifier before
                // counting it. An empty result array is still a valid zero-hit
                // result.
                .iter()
                .map(|value| value.as_str())
                .collect::<Option<Vec<_>>>()
                // Entries use raw artifact identifiers, so numbers cannot stand in
                // for strings even if they could be displayed as text.
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result ranked_chunk_ids entries must be strings",
                    )
                    .with_detail("field", format!("{field}[].per_query[].ranked_chunk_ids"))
                })?
                .len();
            let score_count = item_obj
                .get("scores")
                // Validate the score array separately from identifier entries. A
                // present null or other scalar score collection cannot supply
                // aligned measurements.
                .and_then(JsonValue::as_array)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result scores must be an array",
                    )
                    .with_detail("field", format!("{field}[].per_query[].scores"))
                })?
                // Accept finite engine-local score numbers, including negative
                // ones where an engine uses them. Nonfinite entries cannot be part
                // of the persisted score array.
                .iter()
                .map(|value| value.as_f64().filter(|number| number.is_finite()))
                .collect::<Option<Vec<_>>>()
                // The score array's diagnostic applies to any entry that is
                // nonnumeric or nonfinite. The count is only reliable after every
                // element passes.
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result scores entries must be finite numbers",
                    )
                    .with_detail("field", format!("{field}[].per_query[].scores"))
                })?
                .len();
            // Finish validating all elements before comparing lengths. An empty
            // pair represents a query that returned nothing. One nonempty array
            // and one empty array would leave ranks without scores or scores
            // without identities.
            if ranked_chunk_count != score_count {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query scores must align with ranked_chunk_ids",
                )
                .with_detail("field", format!("{field}[].per_query[].scores")));
            }
            // Read the limit for this query rather than any command-wide default.
            // A run can measure different top_k values. The result bound is
            // independent of the wider retrieval and fusion windows recorded in
            // query_context.
            let top_k = context
                .get("top_k")
                .and_then(JsonValue::as_f64)
                .expect("benchmark query context top_k is validated before result count checks")
                as usize;
            if ranked_chunk_count > top_k {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result ranked_chunk_ids must not exceed query_context.top_k",
                )
                // Report both the requested count and the observed number of IDs.
                // The configured windows checked earlier do not replace this
                // result-length bound.
                .with_detail("field", format!("{field}[].per_query[].ranked_chunk_ids"))
                .with_detail("top_k", top_k)
                .with_detail("ranked_chunk_count", ranked_chunk_count));
            }
        }
    }
    Ok(())
}

// Check the channel flags against the selected mode after structural validation.
// Workload fields and executed retrieval are independent concerns.
fn validate_benchmark_query_context_mode_consistency(
    context: &crate::json::JsonObject,
    field: &'static str,
    query_mode: BenchmarkQueryMode,
) -> Result<()> {
    // A workload row can include text and a vector yet be measured by exact vector
    // alone. In that case text_present is false even though the source row
    // contained text. Attribute the result to the route that produced it.
    let expected_text_present = query_mode.has_text();
    if context.get("text_present").and_then(JsonValue::as_bool) != Some(expected_text_present) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.text_present must match query_mode",
        )
        // Include the context member and run mode on a channel mismatch. Consumers
        // can distinguish incorrect metadata from a malformed boolean value.
        .with_detail(
            "field",
            format!("{field}[].per_query[].query_context.text_present"),
        )
        .with_detail("query_mode", query_mode.as_str()));
    }
    // Use has_vector for the vector route. Both exact-vector and hybrid modes
    // require the context to report the vector channel.
    let expected_embedding_present = query_mode.has_vector();
    if context
        .get("embedding_present")
        .and_then(JsonValue::as_bool)
        != Some(expected_embedding_present)
    {
        // A mismatch is invalid even when the score label matched its engine and
        // mode. The executed-route flags are a distinct part of the saved
        // evidence.
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.embedding_present must match query_mode",
        )
        .with_detail(
            "field",
            format!("{field}[].per_query[].query_context.embedding_present"),
        )
        .with_detail("query_mode", query_mode.as_str()));
    }
    Ok(())
}

// The context preserves the request, effective choices and observed work. A native
// text run carries the product tokenizer. Candidate bounds are derived for
// CofferDB and taken as reported for baselines. The independent shape checks keep
// a valid-looking context from standing in for the run's mode, which is checked by
// the caller.
fn validate_benchmark_query_context(
    item_obj: &crate::json::JsonObject,
    field: &'static str,
) -> Result<()> {
    // Build the complete field prefix once for this nested validation. Errors from
    // either run family can point to the exact context member.
    let detail_prefix = format!("{field}[].per_query[].query_context");
    // Check the context container before inspecting its allowlist. A structurally
    // valid per-query item can still carry a scalar in this slot.
    let context = require_object_field(item_obj, "query_context", detail_prefix.clone())?;
    // Limit context keys to the versioned schema. This includes both summary
    // weight fields and the nested explanation, which are checked independently
    // below.
    reject_unknown_benchmark_result_fields(
        context,
        &detail_prefix,
        &[
            "text_present",
            "tokenizer",
            "embedding_present",
            "filter_present",
            "weights",
            "requested_weights",
            "effective_weights",
            "weights_source",
            "renormalization_reason",
            "metadata_boost_present",
            "metadata_boost_enabled",
            "boost_config_hash",
            "boost_rule_count",
            "top_k",
            "profile",
            "candidate_limits",
            "execution_stats",
            "index_source",
            "index_rebuild_reason",
            "transient_rebuild",
        ],
    )?;
    // Preserve the flags independently. In particular, a rule set can be provided
    // and left disabled. Requiring each boolean keeps the input and configuration
    // observations explicit, without making the presence flag imply that its route
    // contributed a score.
    for flag in [
        "text_present",
        "embedding_present",
        "filter_present",
        "metadata_boost_present",
        "metadata_boost_enabled",
    ] {
        require_bool(context, flag, format!("{detail_prefix}.{flag}"))?;
    }
    // Record tokenizer details only where the CofferDB text path ran. Vector-only
    // queries and baseline engines must not accidentally inherit CofferDB tokenizer
    // metadata.
    // The tokenizer key is required even when it must be null. A missing key would
    // leave the report ambiguous about whether the context was recorded.
    let tokenizer = context.get("tokenizer").ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            "query context requires tokenizer",
        )
        .with_detail("field", format!("{detail_prefix}.tokenizer"))
    })?;
    // For CofferDB text execution, validate the complete tokenizer identity
    // against the product parser. The same field is present with null for
    // exact-vector native and baseline rows. That distinction prevents downstream
    // comparison from crediting a different engine's lexical results to CofferDB
    // tokenization.
    if field == "runs" && context.get("text_present").and_then(JsonValue::as_bool) == Some(true) {
        crate::TokenizerConfig::from_json(tokenizer)?;
    } else if !tokenizer.is_null() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "non-CofferDB or vector-only query tokenizer must be null",
        )
        .with_detail("field", format!("{detail_prefix}.tokenizer")));
    }

    // The requested mix and effective mix answer different questions when a
    // supplied branch is unavailable. Keep both even if all three numbers are
    // identical. This validation checks their representation and leaves the
    // producer responsible for calculating effective weights.
    validate_weight_surface(context, "requested_weights", &detail_prefix)?;
    validate_weight_surface(context, "effective_weights", &detail_prefix)?;
    // Require a nonblank source label so the context identifies where its weights
    // came from. This helper validates the text shape, while upstream selection
    // establishes provenance.
    require_nonempty_string(
        context,
        "weights_source",
        format!("{detail_prefix}.weights_source"),
    )?;
    // Explicit null is used when there is no recorded renormalization reason. A
    // nonempty string records an applicable reason.
    require_string_or_null(
        context,
        "renormalization_reason",
        format!("{detail_prefix}.renormalization_reason"),
    )?;

    // Both the flat summary and the nested compatibility form are stored in the
    // result. Validate each shape because different clients read these forms. The
    // validator does not merge incomplete fields or use one form to fill omissions
    // in the other.
    let weights = require_object_field(context, "weights", format!("{detail_prefix}.weights"))?;
    reject_unknown_benchmark_result_fields(
        weights,
        &format!("{detail_prefix}.weights"),
        &[
            "present",
            "source",
            "requested",
            "effective",
            "renormalization_reason",
        ],
    )?;
    // Require an explicit presence flag in the nested object. Requested and
    // effective weights must still be shaped even if this flag is false.
    require_bool(
        weights,
        "present",
        format!("{detail_prefix}.weights.present"),
    )?;
    // Both summary and detailed weight-source labels are shape-checked as nonempty
    // strings. Neither is inferred from a present weight tuple.
    require_nonempty_string(weights, "source", format!("{detail_prefix}.weights.source"))?;
    // Use a qualified prefix for nested weight errors. Requested or effective
    // coefficients can also fail at the outer context level.
    let weights_prefix = format!("{detail_prefix}.weights");
    validate_weight_surface(weights, "requested", &weights_prefix)?;
    validate_weight_surface(weights, "effective", &weights_prefix)?;
    require_string_or_null(
        weights,
        "renormalization_reason",
        format!("{detail_prefix}.weights.renormalization_reason"),
    )?;

    // The nullable fingerprint slot can represent an absent rule identity. When an
    // identity was recorded, keep its exact string. This pass accepts a nonblank
    // label without reconstructing the rules or recomputing their digest.
    require_string_or_null(
        context,
        "boost_config_hash",
        format!("{detail_prefix}.boost_config_hash"),
    )?;
    // The rule count remains a numeric field regardless of whether boost is
    // enabled. This validates its finite nonnegative representation.
    require_nonnegative_number(
        context,
        "boost_rule_count",
        format!("{detail_prefix}.boost_rule_count"),
    )?;
    // Establish a positive integral quantity before using it as a rank bound. The
    // native branch additionally checks its exact JSON range before comparing
    // profile windows. Both checks use the value recorded with this query, not a
    // limit inferred from how many hits it returned.
    require_positive_integer(context, "top_k", format!("{detail_prefix}.top_k"))?;
    // Use the stable resource profile name recorded for the query. The CofferDB
    // branch below looks up its defaults only after this enum check succeeds.
    let profile = context
        .get("profile")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !is_stable_profile_value(profile) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.profile must be a stable profile enum value",
        )
        .with_detail("field", format!("{detail_prefix}.profile")));
    }

    // Require explicit limits for BM25, vector and fusion. Baselines also report
    // this shape even if some routes or limits do not apply to that engine.
    let candidate_limits = require_object_field(
        context,
        "candidate_limits",
        format!("{detail_prefix}.candidate_limits"),
    )?;
    // The candidate limits key admits the current three names only. When adding a
    // branch, the emitter, required fields and native reconstruction below must
    // agree.
    reject_unknown_benchmark_result_fields(
        candidate_limits,
        &format!("{detail_prefix}.candidate_limits"),
        &["bm25", "vector", "fusion"],
    )?;
    // Check each retrieval window as a finite nonnegative number first. The native
    // path then verifies that each value matches its effective limit.
    for metric in ["bm25", "vector", "fusion"] {
        require_nonnegative_number(
            candidate_limits,
            metric,
            format!("{detail_prefix}.candidate_limits.{metric}"),
        )?;
    }

    // Recompute CofferDB's effective window from the recorded profile and the
    // query's top_k. This also checks requests larger than a profile default.
    // Baseline engines can report their own retrieval windows.
    if field == "runs" {
        // Use the same query-specific top_k that limits the result arrays. This
        // covers a larger request changing the effective BM25, vector or fusion
        // limit independently.
        let top_k = context
            .get("top_k")
            .and_then(JsonValue::as_f64)
            .expect("top_k was validated above");
        // Apply CofferDB's exact JSON upper bound before converting or comparing
        // its requested top_k. The generic positive-integer check alone does not
        // impose that engine bound.
        if top_k > crate::query::MAX_SEARCH_TOP_K as f64 {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result CofferDB top_k exceeds the supported exact JSON range",
            )
            .with_detail("field", format!("{detail_prefix}.top_k")));
        }
        // Recover the defaults from the profile definition used by queries. This
        // keeps validation aligned when a profile's allowances change.
        let params = crate::ResourceProfile::parse(profile)
            .expect("profile was validated above")
            .params();
        // Read each default from the corresponding ResourceProfile parameter.
        // Comparing all branches avoids treating the profile as a single shared
        // window size.
        for (branch, default_limit) in [
            ("bm25", params.bm25_candidates),
            ("vector", params.vector_candidates),
            ("fusion", params.fusion_candidates),
        ] {
            // Use the greater of the query limit and the branch default,
            // independently for each component. A large request may widen one
            // window while leaving a larger branch default unchanged. This
            // expected value describes capacity even if the route produced zero
            // actual candidates.
            let expected = top_k.max(default_limit as f64);
            // A valid number can still misrepresent a native effective window.
            // Fail with the individual branch name when the reported limit differs
            // from what the profile and request imply.
            if candidate_limits.get(branch).and_then(JsonValue::as_f64) != Some(expected) {
                return Err(Error::new(ErrorCode::InvalidJsonlSchema,
                    "benchmark result CofferDB candidate limit must equal max(profile default, top_k)")
                    .with_detail("field", format!("{detail_prefix}.candidate_limits.{branch}")));
            }
        }
    }

    // The report has counters in two places so query_context can also stand on its
    // own. Validate each object independently here. A consumer reconstructing
    // totals or testing whether the two reports match needs a separate comparison.
    let execution_stats = require_object_field(
        context,
        "execution_stats",
        format!("{detail_prefix}.execution_stats"),
    )?;
    // Validate the context counter names on this surface too. Additional context
    // statistics cannot silently become aggregate or per-query summary fields.
    reject_unknown_benchmark_result_fields(
        execution_stats,
        &format!("{detail_prefix}.execution_stats"),
        &[
            "vector_scan_count",
            "distance_ops",
            "fusion_candidate_count",
        ],
    )?;
    // Require all three context counters to be finite and nonnegative. These count
    // work in the selected execution, not the configured candidate windows.
    for metric in [
        "vector_scan_count",
        "distance_ops",
        "fusion_candidate_count",
    ] {
        require_nonnegative_number(
            execution_stats,
            metric,
            format!("{detail_prefix}.execution_stats.{metric}"),
        )?;
    }

    // Use a common provenance vocabulary for native and SQLite results. This field
    // records the index source claimed by the producer. It conveys no comparison
    // of rank quality or raw score scale and is validated separately from mode and
    // tokenizer.
    let index_source = context
        .get("index_source")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // The allowed names cover persistent and transient native data as well as
    // SQLite FTS, vector and combined indexes. Their values identify origin, not
    // evidence of rank equivalence.
    if !matches!(
        index_source,
        "persistent_checkpoint"
            | "transient_rebuild"
            | "sqlite_fts5"
            | "sqlite_vec"
            | "sqlite_fts5_sqlite_vec"
    ) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.index_source must be a stable index source value",
        )
        .with_detail("field", format!("{detail_prefix}.index_source")));
    }
    // Require an explicit rebuild reason slot. A null value indicates the context
    // has no reason to report.
    require_string_or_null(
        context,
        "index_rebuild_reason",
        format!("{detail_prefix}.index_rebuild_reason"),
    )?;
    // A single object groups the rebuild attempt, budget and any measured work. A
    // baseline can carry the same shape with no rebuild. Use explicit null for an
    // unobserved scan count. Zero records an actual count and omitting the key
    // leaves the context incomplete.
    let transient_rebuild = require_object_field(
        context,
        "transient_rebuild",
        format!("{detail_prefix}.transient_rebuild"),
    )?;
    reject_unknown_benchmark_result_fields(
        transient_rebuild,
        &format!("{detail_prefix}.transient_rebuild"),
        // Require only the documented rebuild fields for start and completion,
        // cancellation and budget exhaustion, fixed budget, and optional measured
        // work.
        &[
            "started",
            "completed",
            "cancelled",
            "budget_mb",
            "chunks_scanned",
            "bytes_read",
            "budget_exceeded",
        ],
    )?;
    // Check started, completed, cancelled and budget exceeded as booleans. This
    // loop validates shape without inferring one state from another.
    for flag in ["started", "completed", "cancelled", "budget_exceeded"] {
        require_bool(
            transient_rebuild,
            flag,
            format!("{detail_prefix}.transient_rebuild.{flag}"),
        )?;
    }
    // The rebuild budget is a required finite nonnegative number. The
    // optional-in-value work observations below can instead be null.
    require_nonnegative_number(
        transient_rebuild,
        "budget_mb",
        format!("{detail_prefix}.transient_rebuild.budget_mb"),
    )?;
    // A null scan count means the report has no measured count. Preserve that
    // distinction from observing an actual zero.
    require_nonnegative_number_or_null(
        transient_rebuild,
        "chunks_scanned",
        format!("{detail_prefix}.transient_rebuild.chunks_scanned"),
    )?;
    // Treat bytes read the same way as scan count. A consumer can distinguish
    // missing observations from zero without looking at the rebuild flags.
    require_nonnegative_number_or_null(
        transient_rebuild,
        "bytes_read",
        format!("{detail_prefix}.transient_rebuild.bytes_read"),
    )?;
    Ok(())
}

// Validate a recorded coefficient object without applying query-time
// normalization. This helper serves both summary and nested detailed weights.
fn validate_weight_surface(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_prefix: &str,
) -> Result<()> {
    let weights = require_object_field(obj, field, format!("{detail_prefix}.{field}"))?;
    // Only the three known score components can appear. Each component is
    // mandatory in a persisted weight object even in single-route modes.
    reject_unknown_benchmark_result_fields(
        weights,
        &format!("{detail_prefix}.{field}"),
        &["bm25", "vector", "metadata"],
    )?;
    // Require finite nonnegative coefficients. The result validator checks their
    // representation here without recomputing weighting decisions.
    for weight in ["bm25", "vector", "metadata"] {
        require_nonnegative_number(weights, weight, format!("{detail_prefix}.{field}.{weight}"))?;
    }
    Ok(())
}

// Run-level aggregates can be consumed without opening every query row. Validate
// the required fields and finite ranges here. The persisted artifact contains
// observations from the runner; this shape pass does not replay the searches or
// derive the aggregates from its detailed rows.
fn validate_benchmark_run_metrics(
    run_obj: &crate::json::JsonObject,
    field: &'static str,
) -> Result<()> {
    // First resolve metrics as a single object. Working memory can be unavailable
    // by value, while a missing containing object leaves the whole run without
    // aggregates.
    let metrics = run_obj
        .get("metrics")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field}[].metrics must be an object"),
            )
            .with_detail("field", format!("{field}[].metrics"))
        })?;
    // Reject metrics outside the versioned surface before requiring the known
    // entries. Consumers can use the same expected keys for either run family.
    reject_unknown_benchmark_result_fields(
        metrics,
        &format!("{field}[].metrics"),
        &[
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "p95_ms",
            "vector_scan_count",
            "distance_ops",
            "fusion_candidate_count",
            "memory_peak_mb",
        ],
    )?;
    // Require a recorded finite nonnegative measurement for the ranking, latency
    // and execution aggregates. This check does not derive them from the per-query
    // rows.
    for metric in [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "p95_ms",
        "vector_scan_count",
        "distance_ops",
        "fusion_candidate_count",
    ] {
        require_nonnegative_number(metrics, metric, format!("{field}[].metrics.{metric}"))?;
    }
    // Use null when peak memory was unavailable or unreported. Zero denotes a
    // measured number. The key stays present in either case so consumers can
    // distinguish an unavailable observation from an older or incomplete result
    // shape.
    require_nonnegative_number_or_null(
        metrics,
        "memory_peak_mb",
        match field {
            "runs" => "runs[].metrics.memory_peak_mb",
            "baseline_results" => "baseline_results[].metrics.memory_peak_mb",
            _ => "metrics.memory_peak_mb",
        },
    )?;
    Ok(())
}

// Require a string with non-whitespace content. This shape check uses trim only to
// test emptiness and does not alter the original artifact.
fn require_nonempty_string(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj
        .get(field)
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if value.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a non-empty string"),
        )
        .with_detail("field", detail_field));
    }
    Ok(())
}

// Require a nullable string key. Null is an explicit result value, while omission
// is an incomplete record.
fn require_string_or_null(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    match obj.get(field) {
        Some(JsonValue::Null) => Ok(()),
        Some(JsonValue::String(value)) if !value.trim().is_empty() => Ok(()),
        Some(JsonValue::String(_)) | Some(_) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a non-empty string or null"),
        )
        .with_detail("field", detail_field.clone())),
        // Omission is a missing-field error. An explicit null above leaves the
        // required key in the result while carrying no selected text.
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field)),
    }
}

// Require the JSON boolean type. Numeric and string encodings of true or false are
// invalid for this persisted schema.
fn require_bool(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    if obj.get(field).and_then(JsonValue::as_bool).is_none() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a boolean"),
        )
        .with_detail("field", detail_field));
    }
    Ok(())
}

// Verify that every observed engine/configuration/mode group supplies cold and
// warm measurements. The caller decides whether the containing array may be empty.
fn validate_benchmark_cache_states(
    result: &JsonValue,
    field: &'static str,
    require_non_empty: bool,
) -> Result<()> {
    let runs = result
        .get(field)
        .and_then(JsonValue::as_array)
        .expect("benchmark result array fields are validated before cache-state checks");
    // The caller decides whether it needs any completed baseline evidence. Check
    // that requirement before grouping by cache state. The group walk only
    // validates runs that exist, so it cannot by itself detect a missing
    // configuration or mode.
    if require_non_empty && runs.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {field} must be non-empty when baselines completed"),
        )
        .with_detail("field", field));
    }
    // Group by engine, configuration name and query mode before checking cache
    // states. Looking for cold and warm anywhere in the array would pair
    // measurements from different configurations.
    let mut states_by_group: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // Use already validated run objects as input. Read only the grouping keys and
    // cache-state value during this pass.
    for run in runs {
        let run_obj = run
            .as_object()
            .expect("benchmark result run objects are validated before cache-state checks");
        let engine = run_obj
            .get("engine")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].engine must be a string"),
                )
                .with_detail("field", format!("{field}[].engine"))
            })?;
        // Keep the query mode in the grouping identity. Two modes under the same
        // baseline configuration measure different retrieval work.
        let query_mode = run_obj
            .get("query_mode")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].query_mode must be a string"),
                )
                .with_detail("field", format!("{field}[].query_mode"))
            })?;
        // Require the stable mode spelling before inserting a group key.
        // Unrecognized modes cannot form a separate, seemingly complete cold/warm
        // pair.
        if BenchmarkQueryMode::parse(query_mode).is_none() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result query_mode must be a stable enum value",
            )
            .with_detail("field", format!("{field}[].query_mode")));
        }
        // Interpret cache state as its declared string rather than deducing it
        // from run order or latency.
        let cache_state = run_obj
            .get("cache_state")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].cache_state must be a string"),
                )
                .with_detail("field", format!("{field}[].cache_state"))
            })?;
        // Only the benchmark's cold and warm states are admitted. These labels
        // refer to the runner's cache policy, not to latency thresholds.
        if !matches!(cache_state, "cold" | "warm") {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result cache_state must be cold or warm",
            )
            .with_detail("field", format!("{field}[].cache_state")));
        }
        // Native CofferDB entries carry no configuration name, so their grouping
        // name is empty. Baselines distinguish configurations using their names.
        let name = run_obj
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        // For example, a cold FTS run and a warm vector run under one baseline
        // name leave two incomplete groups. Within a group, repeating cold adds no
        // coverage. The set retains the two states needed for comparisons and
        // discards repetitions at this stage.
        states_by_group
            // Use a delimiter between the grouping components so adjacent ordinary
            // spellings do not form the same concatenation by accident.
            .entry(format!("{engine}\0{name}\0{query_mode}"))
            .or_default()
            .insert(cache_state.to_owned());
    }
    for states in states_by_group.values() {
        // Every observed group needs both states before it counts as a complete
        // pair. Configuration-to-run linkage and required mode coverage are
        // checked elsewhere, where an entirely absent group can be detected.
        if !(states.contains("cold") && states.contains("warm")) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field} must include cold and warm cache states"),
            )
            .with_detail("field", format!("{field}[].cache_state")));
        }
    }
    Ok(())
}

// Require the tokenizer metadata on the surfaces whose schema expects a complete
// configuration. Delegate the structure and value checks to the same tokenizer
// parser used by the product.
fn validate_tokenizer_field(obj: &crate::json::JsonObject, prefix: &str) -> Result<()> {
    let value = obj.get("tokenizer").ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            "complete tokenizer is required",
        )
        .with_detail("field", format!("{prefix}.tokenizer"))
    })?;
    // Use the tokenizer parser's error category and message while replacing the
    // field location with the caller's nested tokenizer path.
    crate::TokenizerConfig::from_json(value)
        .map(|_| ())
        .map_err(|err| {
            Error::new(err.code(), err.message())
                .with_detail("field", format!("{prefix}.tokenizer"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::object_from_pairs;

    #[test]
    fn result_validation_rejects_invalid_roots_without_mutation() {
        for seed in 0..100 {
            for value in [
                JsonValue::Null,
                JsonValue::Bool(seed % 2 == 0),
                JsonValue::Number(seed as f64),
                JsonValue::String(format!("artifact-{seed}")),
                JsonValue::Array(vec![JsonValue::Null; seed]),
            ] {
                let original = value.to_compact_string();
                let error = validate_benchmark_result_contract(&value).unwrap_err();
                assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                assert_eq!(
                    error.detail(),
                    &[("field".into(), "benchmark_result".into())]
                );
                assert_eq!(value.to_compact_string(), original);
            }
        }
        let error = validate_benchmark_result_contract(&JsonValue::object()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::MissingRequiredField);
        assert_eq!(error.detail(), &[("field".into(), "schema_version".into())]);
    }

    #[test]
    fn result_validation_rejects_unknown_top_level_fields_before_missing_fields() {
        for seed in 0..100 {
            let field = format!("unexpected_{seed}");
            let value = object_from_pairs([(field.as_str(), JsonValue::Null)]);
            let original = value.to_compact_string();
            let error = validate_benchmark_result_contract(&value).unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(error.detail(), &[("field".into(), field)]);
            assert_eq!(value.to_compact_string(), original);
        }
    }
}
