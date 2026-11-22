//! Command dispatch, validation, and result rendering for the binary.
//!
//! Each handler returns the result tree together with any operation context and
//! reservation needed to serialize it. Search and diagnosis release the database
//! handle and request-only buffers before allocating their final output trees. The
//! outer driver chooses presentation and exit status after the command returns.
//!
//! Maintenance commands can change files before encoding finishes. A receipt
//! carries the observed effect through later presentation or stdout failures. The
//! ingest and report modules have their own staging and publication boundaries,
//! whose errors can already identify an affected destination.
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::db::{
    CheckOptions, CompactOptions, Database, DatabaseOpenOptions, DiagnosisTarget, RecordKey,
};
use crate::error::{Error, ErrorCode, Result};
use crate::json::{
    object_from_pairs, parse, JsonBuilder, JsonInputKind, JsonOperationContext, JsonOutputSchema,
    JsonReservation, JsonSchema, JsonValue,
};
use crate::metadata::MetadataBoostConfig;
use crate::model::{
    CollectionConfig, DistanceMetric, OnConflict, QueryEmbeddingInput, SearchWeights,
    VectorIndexConfig,
};
use crate::profile::{DatabaseConfig, Durability, ResourceProfile};
use crate::storage::FORMAT_VERSION;

mod ingest;
mod report;

/// Run a CLI argument vector, including its program name, and return the JSON
/// output. A completed check can still have a nonzero command exit status. Call
/// run_with_exit_code when that distinction matters.
pub fn run(args: Vec<String>) -> Result<String> {
    run_with_exit_code(args).map(|output| output.0)
}

/// Run a command in JSON mode and return its rendered output and command exit
/// status. Argument, execution and rendering failures are returned as errors. The
/// caller owns delivery of the returned text.
pub fn run_with_exit_code(args: Vec<String>) -> Result<(String, i32)> {
    run_with_receipt(args).map(|(output, code, _)| (output, code))
}

/// File effects observed before rendering a successful maintenance operation or
/// published report. Keep this value until stdout delivery completes so a later
/// failure can report what already happened.
pub struct MaintenanceReceipt {
    // Observed maintenance effect attached to errors arising after the command has
    // acted.
    effect: Option<&'static str>,
    // Known result details that survive a subsequent output or cleanup failure,
    // such as the published target.
    fields: Vec<(&'static str, String)>,
}

impl MaintenanceReceipt {
    // A report can exist on disk even if the command's later output fails. Store
    // the published target separately from the large diagnosis result.
    fn published_report(output: &str) -> Self {
        Self {
            effect: Some("report_published"),
            fields: vec![("target", output.to_owned())],
        }
    }

    fn new(command: &str, result: &JsonValue) -> Self {
        // Diagnosis report publication has its own receipt because the generated
        // file, rather than the database, is the lasting effect. A result with its
        // published path can immediately supply the target for later delivery
        // errors.
        if command == "explain-report" {
            if let Some(path) = result.get("output_path").and_then(JsonValue::as_str) {
                return Self::published_report(path);
            }
        }
        // Derive the effect from the command's successful outcome. A repair that
        // found no tail or a rebuild that still reports work needed must take the
        // unchanged branch.
        let effect = match command {
            "repair" if result.get("repaired").and_then(JsonValue::as_bool) == Some(true) => {
                "tail_truncated"
            }
            "compact" => "main_replaced",
            "snapshot" => "snapshot_published",
            // The rebuild summary's flag describes the state after the operation.
            // A false value is the completed-index case, so interpreting this as
            // the original need to rebuild would reverse the reported effect.
            "rebuild-index"
                if result.get("rebuild_needed").and_then(JsonValue::as_bool) == Some(false) =>
            {
                "index_written"
            }
            "check" | "repair" | "rebuild-index" => "unchanged",
            // Most commands have no maintenance receipt, including some that write
            // data. An absent receipt says only that this layer has no maintenance
            // effect to attach.
            _ => {
                return Self {
                    effect: None,
                    fields: Vec::new(),
                }
            }
        };
        let mut fields = Vec::new();
        // Scope these identity fields to snapshots. A similarly named member in
        // another command's result does not establish that a new package was
        // published.
        if command == "snapshot" {
            if let Some(path) = result.get("output_path").and_then(JsonValue::as_str) {
                fields.push(("target", path.to_owned()));
            }
            // Take size and commit from the package's decimal strings. The
            // similarly named top-level counters are JSON numbers and can lose
            // precision above the exact integer range of f64. Keep the output path
            // as well so a delivery error identifies the package to inspect.
            if let Some(package) = result.get("package") {
                for key in ["sha256", "size_bytes", "last_valid_commit"] {
                    if let Some(value) = package.get(key).and_then(JsonValue::as_str) {
                        fields.push((key, value.to_owned()));
                    }
                }
            }
        }
        Self {
            effect: Some(effect),
            fields,
        }
    }
    /// Attach the observed maintenance effect and known output details to an error
    /// from the supplied stage, such as formatting or stdout delivery. A more
    /// specific maintenance effect already on the error takes precedence.
    pub fn error(&self, mut err: Error, stage: &str) -> Error {
        if let Some(effect) = self.effect {
            err = err.maintenance_phase(stage, effect);
        }
        // The target and snapshot details are useful even when the later failure
        // came from standard output and knows nothing about the published file.
        for (key, value) in &self.fields {
            err = err.with_detail(*key, value);
        }
        err
    }
}

/// Return JSON output, the CLI exit status and a receipt for errors that occur
/// while delivering the output. The argument vector includes the executable name.
pub fn run_with_receipt(args: Vec<String>) -> Result<(String, i32, MaintenanceReceipt)> {
    run_for_display(args, true)
}

/// Execute the argument vector, including the program name, and return the
/// requested presentation, command exit status and maintenance receipt. A
/// completed integrity check can return `Ok` with a nonzero status. The binary
/// delivers the text before applying that status.
///
/// # Errors
///
/// Argument parsing, command execution and result rendering can each fail. A
/// command can already have written database records or published files when
/// rendering fails. A caller that encounters an error writing or flushing the
/// returned text can attach the observed file effect with
/// [`MaintenanceReceipt::error`].
pub fn run_for_display(args: Vec<String>, json: bool) -> Result<(String, i32, MaintenanceReceipt)> {
    let started = Instant::now();
    // Args consumes the global JSON flag before subcommand parsing. The explicit
    // display-mode argument determines rendering even when this entry point is
    // used outside the executable.
    let mut args = Args::new(args);
    // Arguments follow std::env::args conventions. Consume the program name before
    // checking for the required subcommand so it cannot be misinterpreted as an
    // operation.
    let _program = args.next();
    let Some(command) = args.next() else {
        return Err(usage("missing command"));
    };
    // These commands supply a default for failures before a receipt exists. Report
    // generation tracks publication at its own file boundary. Applying an
    // unchanged default indiscriminately would also misdescribe data commands that
    // can commit a prefix.
    let maintenance = matches!(
        command.as_str(),
        "check" | "repair" | "compact" | "rebuild-index" | "snapshot"
    );
    // Supply a preparation-stage default only for maintenance errors. More
    // specific errors from the operation already carry an effect and
    // maintenance_phase preserves that information.
    let result = run_command(command.as_str(), args).map_err(|err| {
        if maintenance {
            err.maintenance_phase("prepare", "unchanged")
        } else {
            err
        }
    })?;
    // Capture file effects before rendering. Encoding and stdout delivery can
    // fail after a maintenance command has already changed the database.
    let receipt = MaintenanceReceipt::new(&command, &result.value);
    // Choose the status while the structured result is available. In the binary a
    // later stdout write or flush error reports delivery failure with its own
    // status, even if the completed check had already selected a different one.
    let exit_code = success_exit_code(&command, &result.value);
    // Capture elapsed time here, after command execution but before rendering.
    // Writing the returned string to stdout is a separate stage and is outside
    // this duration.
    let output = render_success(&command, result, started.elapsed().as_millis(), json)
        .map_err(|err| receipt.error(err, "encode_result"))?;
    Ok((output, exit_code, receipt))
}

/// Format an execution error as a CLI JSON envelope. A missing command name is
/// reported as unknown. There is no completed result or measured command duration
/// on this path.
pub fn error_output(command: Option<&str>, err: &Error) -> String {
    cli_envelope(
        command.unwrap_or("unknown"),
        false,
        JsonValue::object(),
        Some(err.to_json()),
        0,
    )
}

/// Format the result fields of a CLI envelope for terminal display. Input that
/// does not parse as JSON is returned verbatim. The JSON representation remains
/// available to callers that need all structured result fields.
pub fn human_output(output: &str) -> String {
    // Fallback applies to a parse failure. For a successfully parsed value with no
    // result object, the human form still contains the command heading.
    let Ok(envelope) = parse(output) else {
        return output.to_owned();
    };
    let command = envelope
        .get("command")
        .and_then(JsonValue::as_str)
        .unwrap_or("coffer");
    let mut summary = format!("{command}\n");
    if let Some(result) = envelope.get("result").and_then(JsonValue::as_object) {
        for (key, value) in result {
            // Search hits have a row-oriented display. Preserve hit order and read
            // public fields individually so a missing snippet can still leave an
            // identifier and score to display.
            if key == "results" {
                if let Some(items) = value.as_array() {
                    for item in items {
                        summary.push_str(&format!(
                            "  {}  score={}\n    {}\n",
                            // The search summary displays the external chunk
                            // identifier for the operator. Missing fields use
                            // local display defaults without modifying the stored
                            // JSON result.
                            item.get("chunk_external_id")
                                .and_then(JsonValue::as_str)
                                .unwrap_or("?"),
                            item.get("final_score")
                                .map(JsonValue::to_compact_string)
                                .unwrap_or_default(),
                            item.get("snippet")
                                .and_then(JsonValue::as_str)
                                .unwrap_or("")
                        ));
                    }
                }
            } else {
                // Render other fields as compact JSON so nested values and string
                // escaping survive inside the one-line field summary.
                summary.push_str(&format!("  {key}: {}\n", value.to_compact_string()));
            }
        }
    }
    summary
}

// These reason strings come from the structured diagnosis producer. A new reason
// needs a matching user-facing conclusion before it can reach this exhaustive
// display mapping.
fn diagnosis_conclusion(reason: Option<&str>) -> &'static str {
    match reason {
        Some("not_found") => "Target was not found.",
        Some("deleted") => "Target has been deleted.",
        Some("collection_mismatch") => "Target belongs to another collection.",
        Some("metadata_filter") => "Target was excluded by the metadata filter.",
        Some("no_candidate") => "Target did not enter the candidate set.",
        // The fusion-window and final top-k cutoffs represent two different losses
        // of a candidate. Keep their messages aligned with the stage recorded by
        // the query engine.
        Some("fusion_limit") => "Target was a candidate but ranked outside the fusion window.",
        Some("below_top_k") => "Target was a candidate but ranked below the final top-k.",
        Some("returned") => "Target was returned.",
        _ => unreachable!("unknown diagnosis reason"),
    }
}

// Route status is interpreted independently for each search method. For example, a
// vector route can be unavailable for a missing embedding while the text route
// still finds a candidate.
fn route_explanation(status: &str) -> &'static str {
    match status {
        "not_requested" => "route not requested",
        "unavailable" => "target unavailable",
        "filtered" => "excluded by the filter",
        "no_match" => "no matching query terms",
        "missing_embedding" => "target has no embedding",
        "embedding_dim_mismatch" => "embedding dimensions do not match",
        "outside_limit" => "outside the candidate limit",
        "candidate" => "entered the candidate set",
        _ => unreachable!("unknown diagnosis route status"),
    }
}

fn human_why_not(result: &JsonValue) -> String {
    use std::fmt::Write;
    // The specialized renderer consumes the diagnosis schema produced by the
    // command. Its required outcome and target are accessed directly; optional
    // ranks are displayed only when the corresponding stage recorded them.
    let outcome = result.get("outcome").expect("diagnosis has an outcome");
    fn text<'a>(field: &'a JsonValue, key: &str) -> Option<&'a str> {
        field.get(key).and_then(JsonValue::as_str)
    }
    let target = result
        .get("expected_chunk")
        .expect("diagnosis has a target");
    let id = text(target, "chunk_id").unwrap_or("?");
    let mut output = format!("why-not {id}");
    // An expected target may have only an internal ID. Add the external ID when
    // the resolved target includes one, retaining the internal spelling for
    // disambiguation.
    if let Some(external) = text(target, "chunk_external_id") {
        write!(output, " ({external})").unwrap();
    }
    let conclusion = diagnosis_conclusion(text(outcome, "reason"));
    writeln!(output, "\n{conclusion}").unwrap();
    // The exclusion expression and per-condition reasons belong to the filtering
    // stage. Other failure stages can still have route observations but do not
    // imply that a metadata filter rejected the target.
    if text(outcome, "stage") == Some("filtered") {
        let filter = result.get("filter_result").unwrap();
        writeln!(
            output,
            "  Filter: {}",
            // Print the exact structured filter expression as compact JSON
            // alongside the individual reasons. The human summary thereby
            // preserves the operator and operands used in this diagnosis.
            filter.get("expression").unwrap().to_compact_string()
        )
        .unwrap();
        for reason in filter.get("reasons").and_then(JsonValue::as_array).unwrap() {
            writeln!(output, "    {}", reason.as_str().unwrap_or("")).unwrap();
        }
    }
    // Walk the text and exact-vector routes in a fixed display order. Their ranks
    // come from separate candidate lists and cannot be substituted for the fused
    // or returned position.
    for (name, label) in [("bm25", "BM25"), ("exact", "Exact vector")] {
        let route = outcome.get(name).unwrap();
        let status = text(route, "status").unwrap();
        // Suppress routes the query never requested. A requested route that
        // produced no candidate still gets a line explaining the missing match or
        // unavailable input.
        if status == "not_requested" {
            continue;
        }
        let explanation = route_explanation(status);
        write!(output, "  {label}: {explanation}").unwrap();
        // A recorded rank is interpreted against this route's own window. The
        // fusion limit and final top-k below apply to later stages of the search.
        if let Some(rank) = route.get("rank").and_then(JsonValue::as_f64) {
            let limit = route.get("limit").unwrap().to_compact_string();
            write!(output, " (rank {rank}, limit {limit})").unwrap();
        }
        output.push('\n');
    }
    // A route candidate does not guarantee a fusion rank. Print the fusion
    // comparison only when the recorded candidate ranks include it.
    if let Some(rank) = result
        .get("candidate_ranks")
        .and_then(|v| v.get("fusion_rank"))
        .and_then(JsonValue::as_f64)
    {
        let limit = result
            .get("candidate_limits")
            .unwrap()
            .get("fusion")
            .unwrap()
            .to_compact_string();
        // Take top-k from the recorded query context. Rendering can be independent
        // of the original CLI flags and their defaults once the result has been
        // built.
        let k = result
            .get("query_context")
            .unwrap()
            .get("top_k")
            .unwrap()
            .to_compact_string();
        writeln!(
            output,
            "  Fusion rank {rank}, fusion limit {limit}, top-k {k}"
        )
        .unwrap();
    }
    // Use the recorded result position for the final rank. It can be absent even
    // for a target that entered one or more earlier candidate sets.
    if let Some(rank) = result.get("returned_rank").and_then(JsonValue::as_f64) {
        writeln!(output, "  Returned rank {rank}").unwrap();
    }
    output
}

// A command boundary separates database work from final formatting. Some results
// must carry their operation's JSON reservation and output schema across it
// because serialization allocates additional workspace.
struct CommandResult {
    // Structured command response to render for the selected output mode.
    value: JsonValue,
    // Optional context, memory reservation and schema kept with a budgeted result
    // until serialization.
    retained: Option<(JsonOperationContext, JsonReservation, JsonOutputSchema)>,
}

impl CommandResult {
    // The plain path has no retained output lease. Its value goes through the
    // general CLI envelope formatter.
    fn plain(value: JsonValue) -> Self {
        Self {
            value,
            retained: None,
        }
    }

    // Keep the value, live charge and schema together when the originating
    // database handle goes out of scope. The renderer can then admit envelope and
    // output memory against the same operation.
    fn retained(
        value: JsonValue,
        context: JsonOperationContext,
        work: JsonReservation,
        schema: JsonOutputSchema,
    ) -> Self {
        Self {
            value,
            retained: Some((context, work, schema)),
        }
    }
}

// Simple arms produce a value for the common argument check at the end. Arms
// transferring a budgeted result or delegating command parsing return early and
// must perform their own argument checks.
fn run_command(command: &str, mut args: Args) -> Result<CommandResult> {
    let result = match command {
        "init" => {
            let path = args.required("path")?;
            // Check all arguments before creating the database file. A misspelled
            // option must not leave a newly initialized database behind.
            args.reject_remaining()?;
            Database::create(&path, DatabaseConfig::default())?;
            // Return the format version the newly created database uses. The
            // success response is assembled only after creation has returned
            // without an error.
            Ok(object_from_pairs([
                ("path", JsonValue::String(path)),
                ("format_version", JsonValue::Number(FORMAT_VERSION as f64)),
                ("created", JsonValue::Bool(true)),
            ]))
        }
        "create-collection" => {
            let path = args.required("path")?;
            let name = args.required("name")?;
            let embedding_dim = parse_embedding_dim_flag(&mut args)?;
            let distance_metric = parse_distance_metric_flag(&mut args)?;
            let embedding_model = args.flag_value("--embedding-model")?;
            // Parse a supplied tokenizer by name and default only when the flag is
            // absent. A rejected name must not silently select the default
            // segmentation rules.
            let tokenizer = args
                .flag_value("--tokenizer")?
                .map(|name| crate::TokenizerConfig::from_name(&name))
                .transpose()?
                // Keep the resolved tokenizer so its serialized description
                // reflects the same defaults the new collection uses. The original
                // option need not still be present in Args.
                .unwrap_or_default();
            let vectors_normalized = args.has_flag("--vectors-normalized");
            args.reject_remaining()?;
            let db = Database::open(path)?;
            // Store distance, dimension, normalization and tokenizer in the
            // collection configuration. The response below describes that same
            // configuration after creation succeeds.
            db.create_collection(CollectionConfig {
                tokenizer: tokenizer.clone(),
                name: name.clone(),
                embedding_dim,
                distance_metric,
                embedding_model: embedding_model.clone(),
                vectors_normalized,
                // The CLI creates an exact vector index here. No index selection
                // flag reaches the configuration, even when an embedding model was
                // supplied.
                vector_index: VectorIndexConfig::Exact,
            })?;
            Ok(object_from_pairs([
                ("collection", JsonValue::String(name)),
                ("tokenizer", tokenizer.to_json()),
                ("embedding_dim", JsonValue::Number(embedding_dim as f64)),
                (
                    "distance_metric",
                    JsonValue::String(distance_metric.as_str().into()),
                ),
                // An omitted embedding model stays optional in the response. An
                // empty string would be a different model identifier.
                ("embedding_model", opt_string(embedding_model)),
                ("vectors_normalized", JsonValue::Bool(vectors_normalized)),
            ]))
        }
        "list-collections" => {
            let path = args.required("path")?;
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            // Collection enumeration can build a result whose size depends on
            // catalog contents. Obtain the context and reservation together so
            // final formatting still accounts for that catalog-derived JSON.
            let context = db.json_context();
            let (value, work) = db.list_collections_json(&context)?;
            // This early return bypasses the plain-value formatter. Retaining the
            // exact lease from list_collections_json avoids rebuilding a second
            // copy merely to hand it to the renderer.
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "collection-stats" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            // Index statistics are opt-in for this command. Keep the resolved
            // collection name when delegating to the same stats builder used for
            // the whole database.
            let include_index = args.has_flag("--include-index");
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let context = db.json_context();
            let (value, work) = db.stats_json(Some(&collection), include_index, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "import" => {
            let path = args.required("path")?;
            let manifest = args
                .flag_value("--manifest")?
                .ok_or_else(|| usage("--manifest is required"))?;
            // Parse durability before opening or creating the database. An absent
            // option selects commit durability, while an invalid supplied spelling
            // stops before either operation.
            let durability = parse_durability_flag(&mut args)?;
            // Manifest import defaults to reporting conflicting entries. The
            // separate embedding-only command defaults to replacing them, so
            // preserve each command's conflict policy when sharing argument
            // parsing.
            let on_conflict = args
                .flag_value("--on-conflict")?
                .map(|value| OnConflict::parse(&value))
                .transpose()?
                .unwrap_or(OnConflict::Error);
            args.reject_remaining()?;
            // Argument errors have been ruled out before this branch can create a
            // database. The manifest is read later, so an import error can still
            // leave a newly created database at the requested path.
            // Opening an existing file preserves its normal writer checks. The
            // existence test does not reserve a missing pathname. If another
            // process creates it first, Database::create refuses to overwrite it.
            let db = if PathBuf::from(&path).exists() {
                open_database_with_durability(&path, durability)?
            } else {
                Database::create(
                    &path,
                    DatabaseConfig {
                        durability,
                        ..DatabaseConfig::default()
                    },
                )?
            };
            let context = db.json_context();
            // Defer the full index checkpoint across import batches. The summary is
            // built afterward from the measured import and can be returned with the
            // workspace it still owns.
            // This call can make several primary commits before the final index
            // checkpoint. A late error can leave a committed prefix, so the CLI
            // must not recover from it by reopening the manifest and executing the
            // command again.
            let imported = db.import_manifest_measured_in(
                std::path::Path::new(&manifest),
                on_conflict,
                crate::db::ImportIndexBuildMode::DeferredFullCheckpoint,
                // Keep the same operation context for measuring the import and
                // building its output. The measured object retains its own data
                // while the summary is being copied into the response.
                context.clone(),
            )?;
            // Adding CLI-specific summary fields can itself need memory after the
            // import succeeded. Rebuild only the projected output through
            // import_output_in if its admission is retried.
            let (result, work) =
                import_output_in(&db, &imported.summary, &manifest, durability, &context)?;
            // Release the measured import after the new JSON result has its own
            // reservation. Rendering no longer needs to keep both representations
            // alive.
            drop(imported);
            return Ok(CommandResult::retained(
                result,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "import-embeddings" => {
            let path = args.required("path")?;
            let embeddings = args.required("embeddings.jsonl")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let durability = parse_durability_flag(&mut args)?;
            let on_conflict = args
                .flag_value("--on-conflict")?
                .map(|value| OnConflict::parse(&value))
                .transpose()?
                // Embedding-only import replaces existing embeddings by default. A
                // supplied conflict policy is parsed before the database is opened
                // or any embedding row is processed.
                .unwrap_or(OnConflict::Replace);
            args.reject_remaining()?;
            let db = open_database_with_durability(&path, durability)?;
            let col = db.collection(&collection)?;
            // Embedding-only import requires an existing database and collection.
            // The importer validates and stages this JSONL input before appending
            // its batches. An error after an append is a different retry decision
            // from failure to open the input.
            let summary = db.import_embeddings_jsonl(
                col.collection_id(),
                PathBuf::from(embeddings).as_path(),
                on_conflict,
            )?;
            // The embedding import has completed when the result builder runs.
            // Keep it outside the builder closure because summary construction may
            // be retried after reclaiming workspace.
            return retained_summary_output(&db, |builder| {
                let mut result = summary.build_json(builder)?;
                // Build from the importer's summary before adding the CLI
                // durability field. An existing field from the summary wins when
                // it already supplies that information.
                builder.insert_if_absent(result.as_object_mut().unwrap(), "durability", || {
                    builder.string(durability.as_str())
                })?;
                Ok(result)
            });
        }
        // These handlers take the remaining tokens into their own parsers. They
        // must validate flags and output paths before provider calls or
        // publication. The final unused-argument check below sees the consumed
        // slice and cannot protect those side effects.
        "embed-jsonl" => ingest::embed_jsonl(&mut args),
        "import-dir" => ingest::import_dir(&mut args),
        "search" => return search_command(args, false),
        "vector-search" => return search_command(args, true),
        "why-not" => return diagnosis_command(args, None),
        // An explanation report takes a destination that plain why-not does not
        // need. Consume the output option before handing the remaining search and
        // target arguments to the diagnosis command.
        "explain-report" => {
            let output = args
                .flag_value("--output")?
                .ok_or_else(|| usage("--output is required"))?;
            return diagnosis_command(args, Some(output));
        }
        "get-document" => {
            let path = args.required("path")?;
            // Record IDs and external IDs are interpreted in a collection. Resolve
            // that scope before turning the identifier into a query for a stored
            // document.
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let include_chunks = args.has_flag("--include-chunks");
            let id = args.flag_value("--document-id")?;
            // Consuming only the selected identifier lets reject_remaining catch the
            // two-identifier case before opening the database.
            let external = if id.is_none() {
                Some(
                    // If the internal ID is absent the external ID becomes
                    // mandatory. Giving neither produces a usage error without
                    // falling through to an empty-string lookup.
                    args.flag_value("--external-id")?
                        .ok_or_else(|| usage("--document-id or --external-id is required"))?,
                )
            } else {
                None
            };
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            let col = db.collection(&collection)?;
            // Resolve the selected string in its own namespace. An internal record
            // ID and a caller's external identifier with the same spelling need
            // not refer to the same document.
            let key = if let Some(id) = id.as_deref() {
                RecordKey::Id(id)
            } else {
                RecordKey::ExternalId(external.as_deref().unwrap())
            };
            let context = db.json_context();
            // The document builder can include its chunks, so response size can
            // exceed the document record alone. Return its reservation with the
            // value until the whole result has been formatted.
            let (value, work) =
                db.document_json_in_collection(col.collection_id(), key, include_chunks, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "get-chunk" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let id = args.flag_value("--chunk-id")?;
            // Consume just one key selector. If both are supplied the unused flag
            // remains for reject_remaining, before either a database open or
            // lookup takes place.
            let external = if id.is_none() {
                Some(
                    args.flag_value("--chunk-external-id")?
                        .ok_or_else(|| usage("--chunk-id or --chunk-external-id is required"))?,
                )
            } else {
                None
            };
            args.reject_remaining()?;
            // Read-only open avoids acquiring writer admission for a point lookup.
            // Collection selection still precedes the lookup so an external key is
            // interpreted in the right namespace.
            let db = Database::open_read_only(path)?;
            let col = db.collection(&collection)?;
            let key = if let Some(id) = id.as_deref() {
                RecordKey::Id(id)
            } else {
                RecordKey::ExternalId(external.as_deref().unwrap())
            };
            let context = db.json_context();
            // Get the chunk under the selected collection and keep its output
            // charge with the returned value. The CLI renderer will still need
            // that value after the read-only handle goes away.
            let (value, work) = db.chunk_json_in_collection(col.collection_id(), key, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "delete-document" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let db = Database::open(path)?;
            let col = db.collection(&collection)?;
            // Deletion uses the canonical document ID even when an external ID
            // selected it. Resolve and save the current document first so the
            // result can still report both identifiers after deletion.
            let document = if let Some(id) = args.flag_value("--document-id")? {
                col.get_document(&id)?
            } else {
                let external = args
                    .flag_value("--external-id")?
                    .ok_or_else(|| usage("--document-id or --external-id is required"))?;
                col.get_document_by_external_id(&external)?
            };
            // Measure dependent chunks and their present embedding references
            // before the delete changes visibility. Those counts describe this
            // document's captured state.
            let chunks = col.list_chunks(&document.document_id)?;
            let chunks_deleted = chunks.len();
            // Count only chunks with a recorded embedding reference. A document
            // can own chunks without embeddings, so the two deletion totals are
            // measured separately.
            let embeddings_deleted = chunks
                .iter()
                .filter(|chunk| chunk.embedding_id.is_some())
                .count();
            // Both the writer open and target lookup precede this check. With an
            // invalid extra selector, a missing document or lock failure may
            // surface first. This branch still rejects remaining arguments before
            // issuing the delete.
            args.reject_remaining()?;
            // One document tombstone hides its chunks and their embeddings. The
            // response counts those visibility changes, although no child
            // tombstone is written here. An error refreshing the index can follow
            // the committed delete.
            col.delete_document(&document.document_id)?;
            Ok(object_from_pairs([
                (
                    "document_id",
                    JsonValue::String(document.document_id.clone()),
                ),
                (
                    "external_id",
                    JsonValue::String(document.external_id.clone()),
                ),
                ("chunks_deleted", JsonValue::Number(chunks_deleted as f64)),
                (
                    "embeddings_deleted",
                    JsonValue::Number(embeddings_deleted as f64),
                ),
                (
                    "tombstone_id",
                    // The receipt-like tombstone identifier in the CLI result is
                    // derived from the target type and canonical ID. It is not a
                    // new storage sequence number.
                    tombstone_result_id("document", &document.document_id),
                ),
                ("deleted", JsonValue::Bool(true)),
            ]))
        }
        "delete-chunk" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let db = Database::open(path)?;
            let col = db.collection(&collection)?;
            // Resolve either key spelling into the current chunk before deleting
            // by canonical ID. External IDs are scoped by the collection selected
            // above.
            let chunk = if let Some(id) = args.flag_value("--chunk-id")? {
                col.get_chunk(&id)?
            } else {
                let external = args
                    .flag_value("--chunk-external-id")?
                    .ok_or_else(|| usage("--chunk-id or --chunk-external-id is required"))?;
                col.get_chunk_by_external_id(&external)?
            };
            // Capture whether this chunk referenced an embedding before deletion.
            // The resulting boolean would be unavailable from a later lookup of
            // the deleted chunk.
            let embedding_deleted = chunk.embedding_id.is_some();
            args.reject_remaining()?;
            // Use the resolved internal chunk ID for the mutation whichever
            // selector the caller supplied. The saved chunk is still available to
            // populate the external ID and tombstone fields.
            col.delete_chunk(&chunk.chunk_id)?;
            Ok(object_from_pairs([
                ("chunk_id", JsonValue::String(chunk.chunk_id.clone())),
                (
                    "chunk_external_id",
                    JsonValue::String(chunk.chunk_external_id.clone()),
                ),
                ("embedding_deleted", JsonValue::Bool(embedding_deleted)),
                (
                    "tombstone_id",
                    tombstone_result_id("chunk", &chunk.chunk_id),
                ),
                ("deleted", JsonValue::Bool(true)),
            ]))
        }
        "drop-collection" => {
            let path = args.required("path")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            args.reject_remaining()?;
            let db = Database::open(path)?;
            // Hold the collection's stable internal ID and capture its counts
            // before removing the collection. The response can then identify the
            // tombstone and dependent records after the name disappears.
            let col = db.collection(&collection)?;
            let collection_id = col.collection_id().to_owned();
            let stats = col.stats()?;
            // The collection drop writes a single collection record. Its
            // previously visible descendants become hidden through catalog lookup.
            // Their counts in the receipt are the saved statistics, independent of
            // how many records the drop appends.
            db.drop_collection(&collection)?;
            Ok(object_from_pairs([
                ("collection", JsonValue::String(collection)),
                (
                    "documents_deleted",
                    JsonValue::Number(stats.document_count as f64),
                ),
                (
                    "chunks_deleted",
                    JsonValue::Number(stats.chunk_count as f64),
                ),
                (
                    "embeddings_deleted",
                    JsonValue::Number(stats.embedding_count as f64),
                ),
                (
                    "tombstone_id",
                    tombstone_result_id("collection", &collection_id),
                ),
                ("dropped", JsonValue::Bool(true)),
            ]))
        }
        "stats" => {
            let path = args.required("path")?;
            let include_index = args.has_flag("--include-index");
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            // No collection filter selects aggregate database statistics. The same
            // builder accepts a named filter for collection-stats, keeping index
            // inclusion and output accounting consistent.
            let context = db.json_context();
            let (value, work) = db.stats_json(None, include_index, &context)?;
            return Ok(CommandResult::retained(
                value,
                context,
                work,
                JsonOutputSchema::Value(JsonInputKind::StoredJson),
            ));
        }
        "check" => {
            let path = args.required("path")?;
            // The CLI accepts the explicit verification flag, but checks indexes
            // in either spelling. A missing flag does not request a weaker
            // integrity check.
            let _verify_indexes = args.has_flag("--verify-indexes");
            args.reject_remaining()?;
            // Run the check using the path-level API so it can report findings
            // about the file itself. Later exit-code selection inspects the report
            // even when report generation succeeded.
            let report = Database::check_path(
                path,
                CheckOptions {
                    verify_indexes: true,
                },
            )?;
            // check_path is a path operation and returns a report without a live
            // Database handle for rendering. Give the report conversion a
            // standalone output context.
            return retained_output_in(JsonOperationContext::standalone(), |builder| {
                report.build_json(builder)
            });
        }
        "rebuild-index" => {
            let path = args.required("path")?;
            let collection = args.flag_value("--collection")?;
            // A named collection can still produce a checkpoint for the whole
            // database. If another collection has a conflicting model, the engine
            // can report transient validation of the requested collection. Read
            // the returned rebuild scope before claiming an index was written.
            args.reject_remaining()?;
            let db = Database::open(path)?;
            let summary = db.rebuild_index(collection.as_deref())?;
            // Encoding can fail after index work has finished. Preserve whether
            // the summary says a rebuild is still needed when reporting the effect
            // of that later failure.
            return retained_summary_output(&db, |builder| summary.build_json(builder)).map_err(
                |e| {
                    e.maintenance_phase(
                        "encode_result",
                        if summary.rebuild_needed {
                            "unchanged"
                        } else {
                            "index_written"
                        },
                    )
                },
            );
        }
        "repair" => {
            let path = args.required("path")?;
            // Require explicit truncation intent before invoking repair. Remaining
            // options are rejected before the database can trim its tail.
            if !args.has_flag("--truncate-tail") {
                return Err(usage("repair requires --truncate-tail"));
            }
            args.reject_remaining()?;
            // Keep the filesystem-changing repair outside the result builder.
            // Later JSON work operates on the returned summary and cannot repeat
            // the truncation.
            let summary = Database::repair_truncate_tail(path)?;
            return retained_output_in(JsonOperationContext::standalone(), |builder| {
                summary.build_json(builder)
            })
            .map_err(|e| {
                e.maintenance_phase(
                    "encode_result",
                    // Only a repair summary reporting an actual truncation
                    // licenses the tail_truncated effect. Formatting an unchanged
                    // report can still fail with the same stage.
                    if summary.repaired {
                        "tail_truncated"
                    } else {
                        "unchanged"
                    },
                )
            });
        }
        "snapshot" => {
            let path = args.required("path")?;
            let output_path = args.required("output-path")?;
            args.reject_remaining()?;
            // Snapshot writes to a separate destination. After publication, output
            // encoding must retain the target and identifying package details even
            // if no valid CLI response reaches the caller.
            let summary = Database::snapshot(path, output_path)?;
            return retained_output_in(JsonOperationContext::standalone(), |builder| {
                summary.build_json(builder)
            })
            .map_err(|e| {
                // Delivery diagnostics identify which snapshot is already present,
                // including its target, digest, byte count and last valid commit.
                // A caller can inspect that artifact after an output failure.
                e.maintenance_phase("encode_result", "snapshot_published")
                    .with_detail("target", summary.output_path.display())
                    .with_detail("sha256", &summary.sha256)
                    .with_detail("size_bytes", summary.bytes_copied)
                    .with_detail("last_valid_commit", summary.last_valid_commit)
            });
        }
        "compact" => {
            let path = args.required("path")?;
            // Compaction rebuilds indexes in either CLI spelling. The staging-disk
            // option below changes the quota for temporary files used by the
            // operation.
            let _rebuild_indexes = args.has_flag("--rebuild-indexes");
            let disk_budget = args
                .flag_value("--temporary-disk-bytes")?
                .map(|value| {
                    value
                        .parse::<u64>()
                        .map(crate::temporary::TemporaryDiskBudget::new)
                        .map_err(|_| usage("invalid --temporary-disk-bytes"))
                })
                // Only an omitted disk option selects the default quota. A
                // supplied zero remains zero, and a malformed or out-of-range
                // count is reported before opening the database.
                .transpose()?
                .unwrap_or_default();
            args.reject_remaining()?;
            // Install the chosen temporary disk budget when opening the database
            // so staging uses that budget throughout compaction, including cleanup
            // and replacement.
            let db = Database::open_with_disk_budget(
                std::path::Path::new(&path),
                DatabaseOpenOptions::default(),
                JsonOperationContext::standalone(),
                disk_budget,
            )?;
            // A successful compact can report a failed index rebuild after
            // replacing the primary file. This command returns status zero when it
            // delivers that summary. The index_rebuild field records whether the
            // new generation has a completed checkpoint.
            let summary = db.compact(CompactOptions {
                rebuild_indexes: true,
            })?;
            // If rendering the summary fails here, the main database has already
            // been replaced. Preserve that effect rather than presenting an
            // encoding error as proof that the compaction never happened.
            return retained_summary_output(&db, |builder| summary.build_json(builder))
                .map_err(|e| e.maintenance_phase("encode_result", "main_replaced"));
        }
        "export" => {
            let path = args.required("path")?;
            let out = args.required("output-dir")?;
            let collection = args
                .flag_value("--collection")?
                .ok_or_else(|| usage("--collection is required"))?;
            let format = args
                .flag_value("--format")?
                .unwrap_or_else(|| "jsonl".into());
            // Only JSONL has an exporter on this path. Reject an unsupported
            // format with the supplied value and expected spelling before opening
            // the read-only database.
            if format != "jsonl" {
                return Err(usage("--format must be jsonl")
                    .with_detail("field", "--format")
                    .with_detail("actual", &format)
                    .with_detail("expected", "jsonl"));
            }
            args.reject_remaining()?;
            let db = Database::open_read_only(path)?;
            // Export uses a read-only database handle but can create files in the
            // output directory. A subsequent failure constructing the result
            // occurs after the export has returned its summary.
            let summary = db.export_collection_jsonl(&collection, &out)?;
            // All export names have been published by this point. Only the summary
            // builder may be repeated after reclaiming memory. An encoding error
            // here leaves the completed export on disk.
            return retained_summary_output(&db, |builder| {
                let mut result = summary.build_json(builder)?;
                let obj = result.as_object_mut().unwrap();
                // Fill CLI-specific output fields only when absent. This preserves
                // authoritative values already produced by the export summary.
                builder.insert_if_absent(obj, "output_dir", || builder.string(&out))?;
                builder.insert_if_absent(obj, "collection", || builder.string(&collection))?;
                builder.insert_if_absent(obj, "format", || builder.string(&format))?;
                Ok(result)
            });
        }
        "help" => Ok(help_json()),
        "profile-help" => Ok(profile_help_json()),
        "benchmark" => return benchmark_command(args),
        // Unknown subcommands retain the rejected spelling and a machine-readable
        // reason. This classification occurs before the common remaining-arguments
        // check.
        _ => Err(usage(format!("unknown command {command}"))
            .with_detail("field", "command")
            .with_detail("actual", command)
            .with_detail("reason", "unknown_command")),
    }?;
    // This final check catches unconsumed flags for plain-value command arms.
    // Early-returning handlers must finish validation before transferring their
    // own result.
    args.reject_remaining()?;
    Ok(CommandResult::plain(result))
}

fn import_output_in(
    db: &Database,
    summary: &crate::import::ImportSummary,
    manifest: &str,
    durability: Durability,
    context: &JsonOperationContext,
) -> Result<(JsonValue, JsonReservation)> {
    // The import has already produced its summary. Cache reclamation may retry
    // construction of the CLI result, never the import that committed the records.
    db.admit_input_reclaiming(|| {
        // Use the import's existing context, which can already hold live summary
        // data. A second context here could give the two simultaneously resident
        // representations unrelated budgets.
        build_output_in(context, |builder| {
            let mut result = summary.build_json(builder)?;
            let obj = result.as_object_mut().unwrap();
            // Keep fields supplied by the summary. The CLI contributes its
            // manifest argument and chosen durability only when the summary does
            // not already describe them.
            builder.insert_if_absent(obj, "manifest", || builder.string(manifest))?;
            builder.insert_if_absent(obj, "durability", || builder.string(durability.as_str()))?;
            Ok(result)
        })
    })
}

fn build_output_in(
    context: &JsonOperationContext,
    build: impl FnOnce(&JsonBuilder<'_>) -> Result<JsonValue>,
) -> Result<(JsonValue, JsonReservation)> {
    // Create a fresh reservation for this output attempt. A failure drops the
    // partial tree and its charge together, leaving a retry with no fragments of
    // the first rendering.
    let mut work = context.reservation();
    // StoredJson classifies errors constructing these result fields. Warnings and
    // final serialization later use the schema carried with the retained result.
    let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
    let result = build(&builder)?;
    // Return the reservation alongside the tree. The value survives this helper
    // and must still be covered while the enclosing command response is assembled.
    Ok((result, work))
}

fn retained_summary_output(
    db: &Database,
    build: impl Fn(&JsonBuilder<'_>) -> Result<JsonValue>,
) -> Result<CommandResult> {
    // Get a context tied to the database's budget before handing the summary to
    // its renderer. This keeps other live work on the same database visible to
    // admission.
    let context = db.json_context();
    // The summary builder can be called again after optional cache memory is
    // released. Any command operation or file publication must already have
    // happened outside this closure.
    db.admit_input_reclaiming(|| retained_output_in(context.clone(), &build))
}

fn retained_output_in(
    context: JsonOperationContext,
    build: impl FnOnce(&JsonBuilder<'_>) -> Result<JsonValue>,
) -> Result<CommandResult> {
    // Repair and snapshot callers can also supply a standalone context after their
    // filesystem work. Keep the context chosen by the caller when handing the
    // result off for serialization.
    let (value, work) = build_output_in(&context, build)?;
    // Carry the original context, reservation and output schema together.
    // Serialization can then account for both the existing result and its new
    // public envelope.
    Ok(CommandResult::retained(
        value,
        context,
        work,
        JsonOutputSchema::Value(JsonInputKind::StoredJson),
    ))
}

#[cfg(test)]
fn success_output(command: &str, result: CommandResult, duration_ms: u128) -> Result<String> {
    render_success(command, result, duration_ms, true)
}

fn render_success(
    command: &str,
    result: CommandResult,
    duration_ms: u128,
    json: bool,
) -> Result<String> {
    // Plain commands have no attached result lease. Retained commands must use
    // their existing operation context for warnings, the envelope and its encoded
    // text.
    let mut retained = result.retained;
    let Some((context, work, schema)) = &mut retained else {
        let output = cli_envelope(command, true, result.value, None, duration_ms);
        return Ok(if json { output } else { human_output(&output) });
    };
    // Use the outer kind when building the envelope. Keep the full schema for the
    // serializer below: a filter expression inside a search explanation or a
    // why-not result has its own size and depth limits.
    let builder = JsonBuilder::budgeted(work, schema.kind());
    // The top-level warnings are public copies. Building them under the result's
    // reservation covers the period when both the original warnings and their
    // normalized copies are live.
    let warnings = match result.value.get("warnings").and_then(JsonValue::as_array) {
        Some(items) => builder.array(
            items
                .iter()
                // Normalize each warning with the budgeted builder that owns the
                // envelope. A failure partway through the array aborts this
                // rendering attempt before any text is returned.
                .map(|warning| public_warning_in(warning, &builder)),
        )?,
        // A missing or non-array warnings field yields an empty envelope array.
        // Leave the command's own result untouched for clients that read it
        // directly.
        None => JsonValue::Array(Vec::new()),
    };
    // Move the result into the envelope after reading its warnings. Large result
    // trees can keep their existing allocations rather than being copied field by
    // field.
    let envelope = build_cli_envelope(
        command,
        true,
        result.value,
        warnings,
        None,
        duration_ms,
        &builder,
    )?;
    // Pretty serialization validates the retained tree under its command schema
    // and returns the text with its own reservation. An encoding failure is still
    // possible after the underlying command succeeded.
    let (text, output_work) = envelope.pretty_retained_in(*schema, context, work)?;
    // The human diagnosis needs typed fields for its route and filter narrative.
    // Validate and encode under the same schema first, then format directly from
    // the checked result tree.
    if !json && matches!(schema, JsonOutputSchema::WhyNotEnvelope) {
        // The generic encoded form is no longer needed for the diagnosis. Drop it
        // before its output reservation and keep the envelope alive for the
        // field-by-field human renderer.
        drop(text);
        drop(output_work);
        return Ok(human_why_not(envelope.get("result").unwrap()));
    }
    // The ordinary paths have finished reading the envelope. Release the live tree
    // before its reservation, then hand the rendered text to the selected output
    // mode.
    drop(envelope);
    drop(retained);
    drop(output_work);
    // At this boundary the encoded string becomes caller-owned. The binary can
    // still fail to write or flush it, after the command has acted and the
    // result's budget has been released.
    Ok(if json { text } else { human_output(&text) })
}

fn success_exit_code(command: &str, result: &JsonValue) -> i32 {
    // A completed check comes with a usable report even when its process exit
    // status must be nonzero. Other completed commands use the ordinary success
    // code.
    if command != "check" {
        return 0;
    }
    if result
        .get("issues")
        // Check result issues only when the field is an array. Missing or
        // malformed issue metadata cannot by itself select the special mismatch
        // exit code.
        .and_then(JsonValue::as_array)
        .is_some_and(|issues| {
            // Reserve code 2 for a nonempty list consisting entirely of
            // embedding-model mismatches. With an empty list the all predicate
            // would otherwise match vacuously.
            !issues.is_empty()
                && issues.iter().all(|issue| {
                    // A missing or non-string code prevents the all-mismatch case
                    // just as a genuinely different issue does. The remaining
                    // status path then decides the exit code.
                    issue.get("code").and_then(JsonValue::as_str)
                        == Some("embedding_model_mismatch")
                })
        })
    {
        return 2;
    }
    // For other issue lists, use the string status. A missing or non-string status
    // follows the zero default, while a string other than ok yields code 3.
    match result.get("status").and_then(JsonValue::as_str) {
        Some("ok") | None => 0,
        Some(_) => 3,
    }
}

fn cli_envelope(
    command: &str,
    ok: bool,
    result: JsonValue,
    error: Option<JsonValue>,
    duration_ms: u128,
) -> String {
    // The CLI writes one pretty JSON envelope terminated by a newline. Keep that
    // framing consistent with the separately budgeted retained serializer.
    cli_envelope_value(command, ok, result, error, duration_ms).to_pretty_string() + "\n"
}

fn cli_envelope_value(
    command: &str,
    ok: bool,
    result: JsonValue,
    error: Option<JsonValue>,
    duration_ms: u128,
) -> JsonValue {
    // Host-built responses normalize warnings the same way as the retained path.
    // Their warning copies belong to the envelope, while result keeps the
    // command-specific originals.
    let warnings = result
        .get("warnings")
        .and_then(JsonValue::as_array)
        .map(|items| JsonValue::Array(items.iter().map(public_warning).collect()))
        .unwrap_or_else(|| JsonValue::Array(Vec::new()));
    build_cli_envelope(
        command,
        ok,
        result,
        warnings,
        error,
        duration_ms,
        // Use the host builder for the plain response and error presentation. A
        // retained command supplies a budgeted builder to the same field layout.
        &JsonBuilder::host(),
    )
    .expect("host envelope construction is infallible")
}

fn build_cli_envelope(
    command: &str,
    ok: bool,
    result: JsonValue,
    warnings: JsonValue,
    error: Option<JsonValue>,
    duration_ms: u128,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    let mut envelope = builder.object([
        // The schema_version belongs to this wire format. Keep its value
        // independent of the binary's package version and the storage format in
        // the database file.
        ("schema_version", builder.string("0.1")?),
        ("command", builder.string(command)?),
        ("ok", JsonValue::Bool(ok)),
        ("warnings", warnings),
        // The success path samples elapsed time before rendering the envelope. It
        // excludes JSON encoding and stdout delivery. Error envelopes pass zero
        // because that path has no elapsed measurement.
        ("duration_ms", JsonValue::Number(duration_ms as f64)),
        ("result", result),
    ])?;
    // Successful output omits the optional error member. On a failure, move the
    // caller's structured error into the envelope without rebuilding its nested
    // detail fields.
    if let Some(error) = error {
        builder.insert_if_absent(
            envelope
                .as_object_mut()
                .expect("CLI envelope is always an object"),
            "error",
            || Ok(error),
        )?;
    }
    Ok(envelope)
}

fn public_warning(warning: &JsonValue) -> JsonValue {
    // The host warning adapter shares its normalization with budgeted output. The
    // fallible builder form is needed when these extra copies count against a
    // command's retained JSON budget.
    public_warning_in(warning, &JsonBuilder::host())
        .expect("host warning construction is infallible")
}

fn public_warning_in(warning: &JsonValue, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    if warning.as_object().is_some() {
        // Copy the warning before adding public defaults. A command result can
        // also expose the original object, whose fields and representation must
        // remain intact.
        let mut value = builder.copy_validated_value(warning)?;
        let object = value.as_object_mut().expect("warning object copied above");
        // A supplied details member is authoritative even when empty. Only
        // synthesize it for object warnings using the older flat representation.
        if !object.contains_key("details") {
            let mut details = builder.object([])?;
            // Project the recognized diagnostic fields into details and retain
            // their original types. Unrecognized object fields remain available at
            // their original top-level keys.
            for key in ["field", "artifact_kind", "reason"] {
                if let Some(value) = object.get(key) {
                    // Copy the selected values because they remain in the original
                    // flat fields too. The nested details object must own
                    // independent JSON members.
                    builder.insert_if_absent(details.as_object_mut().unwrap(), key, || {
                        builder.copy_validated_value(value)
                    })?;
                }
            }
            builder.insert_if_absent(object, "details", || Ok(details))?;
        }
        // Defaults fill missing members only. Caller-supplied severity and path
        // survive normalization, including an explicit null path.
        builder.insert_if_absent(object, "severity", || builder.string("warning"))?;
        builder.insert_if_absent(object, "path", || Ok(JsonValue::Null))?;
        // Object warnings keep their own code and message, including their
        // absence. Only non-object warnings below receive a generic code and a
        // synthesized message.
        return Ok(value);
    }
    // A string warning supplies the public message verbatim. Other non-object
    // values use the generic message and the same empty details object, so callers
    // receive one warning shape.
    builder.object([
        ("code", builder.string("warning")?),
        ("severity", builder.string("warning")?),
        ("path", JsonValue::Null),
        (
            "message",
            builder.string(warning.as_str().unwrap_or("warning"))?,
        ),
        ("details", JsonValue::object()),
    ])
}

fn opt_string(value: Option<String>) -> JsonValue {
    // Keep optional fields present as JSON null when absent. Consumers can
    // distinguish this result shape from commands that omit the field entirely.
    value.map(JsonValue::String).unwrap_or(JsonValue::Null)
}

#[cfg(test)]
fn insert_object_field(value: &mut JsonValue, key: &str, item: JsonValue) {
    value
        .as_object_mut()
        .expect("CLI command result is always a JSON object")
        .insert(key.into(), item);
}

fn tombstone_result_id(target_type: &str, target_id: &str) -> JsonValue {
    // The CLI derives this label from the target type and internal ID. It
    // identifies a target, not a particular log record. Multiple events for the
    // same target would produce the same label.
    JsonValue::String(format!("tombstone:{target_type}:{target_id}"))
}

#[derive(Clone, Copy)]
enum ParsedProfile {
    // A profile help request is an action in its own right. The caller decides
    // whether that action is valid before opening a database or executing a
    // search.
    Help,
    Profile(ResourceProfile),
}

fn parse_profile_flag(args: &mut Args) -> Result<Option<ParsedProfile>> {
    args.flag_value("--profile")?
        .map(|value| {
            // Recognize help before parsing a concrete resource profile. Absence
            // remains a separate result so each command can apply its own profile
            // default.
            if value == "help" {
                Ok(ParsedProfile::Help)
            } else {
                ResourceProfile::parse(&value)
                    .map(ParsedProfile::Profile)
                    // Keep the invalid token in the CLI error and enumerate the
                    // accepted spellings, including help. A missing profile is
                    // handled by the command rather than being a parse error.
                    .ok_or_else(|| {
                        usage("invalid --profile")
                            .with_detail("field", "--profile")
                            .with_detail("actual", &value)
                            .with_detail("expected", "expanded|balanced|constrained|help")
                    })
            }
        })
        .transpose()
}

fn parse_durability_flag(args: &mut Args) -> Result<Durability> {
    args.flag_value("--durability")?
        .map(|value| {
            Durability::parse(&value).ok_or_else(|| {
                usage("invalid --durability")
                    .with_detail("field", "--durability")
                    .with_detail("actual", &value)
                    .with_detail("expected", "commit|fast")
            })
        })
        .transpose()
        // An omitted durability flag selects commit. A supplied invalid spelling
        // has already produced a usage error and must not silently choose that
        // default.
        .map(|value| value.unwrap_or(Durability::Commit))
}

fn parse_top_k_flag(args: &mut Args) -> Result<usize> {
    let value = args
        .flag_value("--top-k")?
        .unwrap_or_else(|| "10".to_owned());
    // This layer checks whether the token fits the native unsigned integer. Search
    // validation enforces the permitted nonzero range once the request is built.
    value.parse::<usize>().map_err(|_| {
        usage("--top-k must be an integer")
            .with_detail("field", "--top-k")
            .with_detail("actual", &value)
            .with_detail("expected", "positive_integer")
    })
}

fn parse_embedding_dim_flag(args: &mut Args) -> Result<usize> {
    let value = args
        .flag_value("--dim")?
        .ok_or_else(|| usage("--dim is required"))?;
    // Use the same structured flag diagnostic for a malformed number and an
    // explicitly zero dimension. A missing required flag remains a distinct usage
    // error.
    let dim = value
        .parse::<usize>()
        .map_err(|_| invalid_embedding_dim_usage(&value))?;
    if dim == 0 {
        return Err(invalid_embedding_dim_usage(&value));
    }
    Ok(dim)
}

// Use one error shape for overflow, nonnumeric input and zero. The raw spelling in
// actual is important when a parsed numeric value does not exist.
fn invalid_embedding_dim_usage(value: &str) -> Error {
    usage("--dim must be a positive integer")
        .with_detail("field", "--dim")
        .with_detail("actual", value)
        .with_detail("expected", "positive_integer")
}

fn parse_distance_metric_flag(args: &mut Args) -> Result<DistanceMetric> {
    let value = args
        .flag_value("--metric")?
        .unwrap_or_else(|| "cosine".to_owned());
    // The model parser defines the accepted distance metrics. Translate a
    // rejection into a CLI diagnostic that preserves the flag and the exact
    // spelling the user supplied.
    DistanceMetric::parse(&value).map_err(|_| {
        usage("invalid --metric")
            .with_detail("field", "--metric")
            .with_detail("actual", &value)
            // List the canonical names in the diagnostic. The parser also accepts
            // model-layer aliases, and successful collection output uses the
            // canonical name even when given an alias.
            .with_detail("expected", "cosine|dot_product|euclidean")
    })
}

fn open_database_with_durability(path: &str, durability: Durability) -> Result<Database> {
    Database::open_with_options(
        path,
        DatabaseOpenOptions {
            // Pass the CLI's durability choice as an explicit open override while
            // leaving all other database options to their normal defaults.
            durability: Some(durability),
            ..DatabaseOpenOptions::default()
        },
    )
}

fn help_json() -> JsonValue {
    // This command catalog is itself a machine-readable help result. Command
    // names, positional syntax, accepted flags and stability labels are maintained
    // alongside their dispatch branches.
    object_from_pairs([
        ("schema_version", JsonValue::String("0.1".into())),
        ("format", JsonValue::String("json".into())),
        (
            "commands",
            // These lists describe command-specific flags. The binary inspects
            // --json for the display mode and Args removes it before dispatch, so
            // each entry need not advertise it.
            JsonValue::Array(vec![
                help_command("help", "stable", "coffer help", &[]),
                help_command("init", "p0", "coffer init <path>", &[]),
                help_command(
                    "create-collection",
                    "p0",
                    "coffer create-collection <path> <name>",
                    &[
                        "--dim",
                        "--metric",
                        "--embedding-model",
                        "--tokenizer",
                        "--vectors-normalized",
                    ],
                ),
                help_command(
                    "list-collections",
                    "p0",
                    "coffer list-collections <path>",
                    &[],
                ),
                help_command(
                    "collection-stats",
                    "p0",
                    "coffer collection-stats <path>",
                    &["--collection", "--include-index"],
                ),
                // Manifest-driven import and direct embedding import have
                // different inputs. Advertise their actual selectors here so
                // scripts do not mistake manifest entries for flags accepted by
                // import.
                help_command(
                    "import",
                    "p0",
                    "coffer import <path>",
                    &["--manifest", "--on-conflict", "--durability"],
                ),
                help_command(
                    "import-embeddings",
                    "p0",
                    "coffer import-embeddings <path> <input-path>",
                    &["--collection", "--on-conflict", "--durability"],
                ),
                // The provider-generation entries have distinct side effects. Keep
                // their synopsis explicit about when chunk text leaves the machine
                // and when any database import starts.
                help_command(
                    "embed-jsonl",
                    "p0",
                    "coffer embed-jsonl <chunks.jsonl> --provider <openai|openai-compatible> --model <name> --out <embeddings.jsonl> --dimensions <n>. Uploads chunk text only after this command is configured. Does not open a database or store the API key. A failed generation batch does not write a database.",
                    &[
                        "--provider",
                        "--model",
                        "--out",
                        "--dimensions",
                        "--batch-size",
                        "--concurrency",
                        "--api-key-env",
                        "--base-url",
                        "--metadata-out",
                    ],
                ),
                // Directory import can write text artifacts without opening the
                // database. Its embedding mode adds a later import stage, so the
                // help text distinguishes a generation failure from already
                // committed batches.
                help_command(
                    "import-dir",
                    "p0",
                    "coffer import-dir <input-dir> <path> --collection <name> --chunk-size <bytes> --chunk-overlap <bytes> --emit-jsonl <dir> [--input-encoding <encoding>] [--embed <provider:model>]. Without --embed, writes text artifacts only and does not open the database. With --embed, uploads chunk text and imports only after validation. The API key is not stored. A failed generation batch does not change the database; core batches committed after import starts are kept.",
                    &[
                        "--collection",
                        "--chunk-size",
                        "--chunk-overlap",
                        "--format",
                        "--input-encoding",
                        "--include",
                        "--exclude",
                        "--emit-jsonl",
                        "--embed",
                        "--dimensions",
                        "--batch-size",
                        "--concurrency",
                        "--api-key-env",
                        "--base-url",
                        "--metadata-out",
                    ],
                ),
                help_command(
                    "search",
                    "p0",
                    // Use the query validator's maximum in public examples for
                    // top-k. The ceiling can depend on native size and on the
                    // representation used in externally reported limits.
                    &format!(
                        "coffer search <path> [<query>] [--top-k <1..={}>]",
                        crate::query::MAX_SEARCH_TOP_K
                    ),
                    &[
                        "--collection",
                        "--top-k",
                        "--profile",
                        "--filter",
                        "--weights",
                        "--metadata-boost",
                        "--embedding",
                    ],
                ),
                // Vector search takes its mandatory query vector from a flag. Keep
                // the vector-only synopsis distinct from the search command that
                // can accept positional query text.
                help_command(
                    "vector-search",
                    "p0",
                    &format!(
                        "coffer vector-search <path> [--top-k <1..={}>]",
                        crate::query::MAX_SEARCH_TOP_K
                    ),
                    &[
                        "--embedding",
                        "--collection",
                        "--top-k",
                        "--profile",
                        "--filter",
                    ],
                ),
                // Keep the document and chunk selector spellings aligned with
                // their respective commands. Each has its own external-ID argument
                // as well as an internal identifier.
                help_command(
                    "get-document",
                    "p0",
                    "coffer get-document <path>",
                    &["--collection", "--document-id", "--external-id"],
                ),
                help_command(
                    "get-chunk",
                    "p0",
                    "coffer get-chunk <path>",
                    &["--collection", "--chunk-id", "--chunk-external-id"],
                ),
                help_command(
                    "delete-document",
                    "p0",
                    "coffer delete-document <path>",
                    &["--collection", "--document-id", "--external-id"],
                ),
                help_command(
                    "delete-chunk",
                    "p0",
                    "coffer delete-chunk <path>",
                    &["--collection", "--chunk-id", "--chunk-external-id"],
                ),
                help_command(
                    "drop-collection",
                    "p0",
                    "coffer drop-collection <path>",
                    &["--collection"],
                ),
                // Statistics can optionally include index data. The check command
                // accepts the verification flag for compatibility and verifies
                // indexes even when it is omitted.
                help_command("stats", "p0", "coffer stats <path>", &["--include-index"]),
                help_command("check", "p0", "coffer check <path>", &["--verify-indexes"]),
                // Compaction accepts a staging-disk limit in bytes. This is
                // separate from collection or query profile controls and is
                // applied when opening the maintenance database.
                help_command(
                    "compact",
                    "p0",
                    "coffer compact <path>",
                    &["--rebuild-indexes","--temporary-disk-bytes"],
                ),
                // The benchmark accepts --output and --result-file as alternative
                // names for one destination. Listing both keeps the accepted
                // spellings discoverable, but specifying both is a usage error.
                help_command(
                    "benchmark",
                    "p0",
                    "coffer benchmark <path>",
                    &[
                        "--dataset",
                        "--profile",
                        "--durability",
                        "--collection",
                        "--output",
                        "--result-file",
                        "--require-baseline",
                    ],
                ),
                // The database is opened read-only for export, but existing export
                // files in the destination can be replaced. The returned result
                // identifies the collection, directory and JSONL format after
                // publication.
                help_command(
                    "export",
                    "p0",
                    "coffer export <path> <output-dir>",
                    &["--collection", "--format"],
                ),
                // Report generation and diagnosis share query controls. The report
                // also takes a new output pathname, a separate effect from
                // returning diagnosis data in the CLI response.
                help_command(
                    "explain-report", "debug", "coffer explain-report <path> --output <new.html>",
                    &["--output","--collection","--query","--chunk-id","--chunk-external-id","--top-k","--profile","--filter","--weights","--embedding","--metadata-boost"],
                ),
                help_command(
                    "why-not",
                    "debug",
                    &format!(
                        "coffer why-not <path> [--top-k <1..={}>]",
                        crate::query::MAX_SEARCH_TOP_K
                    ),
                    &[
                        "--collection",
                        "--query",
                        "--chunk-id",
                        "--chunk-external-id",
                        "--top-k",
                        "--profile",
                        "--filter",
                        "--weights",
                        "--embedding",
                        "--metadata-boost",
                    ],
                ),
                // The synopsis shows a named collection. Dispatch also accepts the
                // database path alone, in which case rebuild checks model
                // consistency across the database before writing a shared
                // checkpoint.
                help_command(
                    "rebuild-index",
                    "debug",
                    "coffer rebuild-index <path> --collection <name>",
                    &["--collection"],
                ),
                // Repair requires the explicit --truncate-tail flag; keep the
                // action visible in the flag list even though the command name
                // already identifies a maintenance operation.
                help_command(
                    "repair",
                    "debug",
                    "coffer repair <path>",
                    &["--truncate-tail"],
                ),
                // The stability string is metadata returned by help. Whether a
                // subcommand runs and what it does are defined by its actual
                // dispatcher.
                help_command(
                    "snapshot",
                    "future",
                    "coffer snapshot <path> <output-path>",
                    &[],
                ),
                help_command("profile-help", "stable", "coffer profile-help", &[]),
            ]),
        ),
        (
            // Keep these numeric meanings aligned with success_exit_code for
            // completed check reports and the binary's error_exit_code for
            // failures.
            "exit_codes",
            JsonValue::Array(vec![
                help_exit_code(0, "success"),
                help_exit_code(1, "user_or_input_error"),
                // This public label covers several paths. A completed check
                // containing only model mismatches, a resource exhaustion error,
                // and an output delivery failure can all produce code 2.
                help_exit_code(2, "file_or_corruption_error"),
                help_exit_code(3, "check_issues"),
                help_exit_code(4, "lock_contention"),
            ]),
        ),
    ])
}

// Build a command entry for machine-readable help. Keep the synopsis for display
// and the accepted flag spellings as a separate list so a caller does not have to
// parse the prose.
fn help_command(name: &str, stability: &str, synopsis: &str, flags: &[&str]) -> JsonValue {
    object_from_pairs([
        ("name", JsonValue::String(name.into())),
        ("stability", JsonValue::String(stability.into())),
        ("synopsis", JsonValue::String(synopsis.into())),
        (
            // Keep flags as a JSON array in the same order as the command
            // descriptor. The strings themselves are option spellings, separate
            // from any values shown in the synopsis.
            "flags",
            JsonValue::Array(
                flags
                    .iter()
                    .map(|flag| JsonValue::String((*flag).into()))
                    .collect(),
            ),
        ),
    ])
}

// Publish the numeric process exit status alongside its meaning. The platform
// status fits exactly in the JSON number representation.
fn help_exit_code(code: u8, meaning: &str) -> JsonValue {
    object_from_pairs([
        ("code", JsonValue::Number(code as f64)),
        ("meaning", JsonValue::String(meaning.into())),
    ])
}

// Profile help is a versioned JSON response distinct from executing a search.
// Expose the actual default preset through its CLI spelling so a consumer can use
// the same name in a command.
fn profile_help_json() -> JsonValue {
    object_from_pairs([
        // Version the help payload independently of the database and query output
        // schemas. Consumers can read these presets without issuing a query.
        ("schema_version", JsonValue::String("0.1".into())),
        (
            "default",
            JsonValue::String(ResourceProfile::Balanced.as_cli_str().into()),
        ),
        (
            "profiles",
            // List all supported presets in a stable presentation order. The
            // separate default key identifies the selected default regardless of
            // array order.
            JsonValue::Array(vec![
                profile_help_item(
                    ResourceProfile::Expanded,
                    "advanced preset for retrieval experiments, candidate inspection, and quality evaluation, with larger default candidate windows and cache budget; may increase CPU, memory, or latency without improving Recall/MRR. Per-query selection preserves explicit runtime cache-budget overrides; cache budget is not a process RSS limit",
                ),
                // Keep the help text's weight examples aligned with
                // default_weights_for in the query module. These are the defaults
                // for a route combination before any per-query adjustment, not
                // candidate counts or quality measurements.
                profile_help_item(
                    ResourceProfile::Balanced,
                    "default preset for everyday local RAG and code search; choose balanced when unsure. Without metadata boost, text+vector defaults to BM25/vector weights 0.5/0.5; text-only or vector-only uses weight 1 for its input. With metadata boost, defaults are 0.45/0.50/0.05 for both inputs, 0.95/0/0.05 for text-only, and 0/0.95/0.05 for vector-only (BM25/vector/metadata). --weights overrides these defaults; explain records requested/effective weights and their source. External vectors do not guarantee semantic relevance; the engine remains responsible for correct calculation, filtering and configuration. Score shares are not recall shares or probabilities. Reference-recipe quality requires separate reviewed evaluation; user overrides do not validate the default path. Candidate and cache budgets control investment, not an optimality guarantee. Per-query selection preserves explicit runtime cache-budget overrides; cache budget is not a process RSS limit",
                ),
                profile_help_item(
                    ResourceProfile::Constrained,
                    "advanced preset for applications with limited host resources, with smaller default candidate windows and cache budget. Queries may be slower, retain fewer candidates, or report resource_exhausted when the budget is insufficient. This preset does not guarantee lower RSS, CPU use, or energy use. Lower candidate limits do not reduce full exact vector distance work; explicit runtime cache-budget overrides still apply, and cache budget is not a process RSS limit",
                ),
            ]),
        ),
    ])
}

// Keep separate CLI and JSON fields even when their current spellings match. Each
// field comes from the accessor for its own input format.
fn profile_help_item(profile: ResourceProfile, description: &str) -> JsonValue {
    object_from_pairs([
        ("cli", JsonValue::String(profile.as_cli_str().into())),
        ("json", JsonValue::String(profile.as_json_str().into())),
        // Retain the alias member even when there is no alias. Consumers can keep
        // one item schema for every preset.
        ("alias", JsonValue::Null),
        ("description", JsonValue::String(description.into())),
        // Read nominal limits from the profile definition. A real search can widen
        // them to satisfy top_k or disable a route absent from the request, so
        // help reports the preset and never a live search limit.
        (
            "candidate_limits",
            object_from_pairs([
                (
                    "bm25",
                    JsonValue::Number(profile.params().bm25_candidates as f64),
                ),
                (
                    "vector",
                    JsonValue::Number(profile.params().vector_candidates as f64),
                ),
                (
                    "fusion",
                    JsonValue::Number(profile.params().fusion_candidates as f64),
                ),
            ]),
        ),
        // Expose the preset's cache budget separately from candidate limits. An
        // explicit runtime cache budget can still take precedence when running a
        // query.
        (
            "cache_budget_mb",
            JsonValue::Number(profile.params().cache_budget_mb as f64),
        ),
    ])
}

// Use the same result construction for general and vector-only searches. The
// vector-only command narrows accepted inputs at the boundary while both share
// collection resolution and retained result accounting.
fn search_command(mut args: Args, vector_only: bool) -> Result<CommandResult> {
    let path = args.required("path")?;
    // Parse before opening the database with an operation context that can
    // accompany execution later. Keep raw optional input and the filter in
    // separate reservations so their lifetimes can diverge in diagnostics.
    let context = JsonOperationContext::standalone();
    // Use the same parent context for embedding, weights and boost parsing. Peak
    // accounting includes simultaneously live raw and typed buffers for any option
    // being converted.
    let mut input_work = context.reservation();
    let mut filter_work = context.reservation();
    // Reject options outside the vector-only command surface before reading any
    // embedding file. This preserves usage-error precedence when both an
    // unsupported option and a bad file were supplied.
    if vector_only && args.args.iter().any(|arg| arg == "--weights") {
        return Err(usage("vector-search does not accept --weights")
            .with_detail("command", "vector-search")
            .with_detail("field", "--weights")
            .with_detail("reason", "unsupported_option"));
    }
    // The vector-only shorthand owns its score configuration. Reject a metadata
    // boost override here even though a general vector-only request through search
    // can accept one.
    if vector_only && args.args.iter().any(|arg| arg == "--metadata-boost") {
        return Err(usage("vector-search does not accept --metadata-boost")
            .with_detail("command", "vector-search")
            .with_detail("field", "--metadata-boost")
            .with_detail("reason", "unsupported_option"));
    }
    // An embedding occupies its own vector and model-name allocation. Transfer
    // that charge into input_work when taking the parsed object so the buffers are
    // covered through execution.
    let embedding = args
        .flag_value("--embedding")?
        .map(|path| -> Result<_> {
            let (value, work) = read_embedding_file_in(&path, &context)?;
            // Merge within the original operation rather than reserving the vector
            // twice. Moving the value without its charge would make later parsing
            // undercount live memory.
            input_work.merge(work);
            Ok(value)
        })
        .transpose()?;
    // The vector-only entry requires an explicit embedding even though a general
    // search may start from text alone.
    if vector_only && embedding.is_none() {
        return Err(usage("vector-search requires --embedding"));
    }
    let collection = args
        .flag_value("--collection")?
        .ok_or_else(|| usage("--collection is required"))?;
    // Top-k syntax is checked before the help sentinel. General search can return
    // profile help without query text. Vector-search has already required an
    // embedding.
    let top_k = parse_top_k_flag(&mut args)?;
    let profile = parse_profile_flag(&mut args)?;
    // Handle the help sentinel before parsing query JSON or opening the database.
    // Reject leftover tokens because this branch consumes no positional query
    // text.
    if matches!(profile, Some(ParsedProfile::Help)) {
        args.reject_remaining()?;
        // Return a plain help payload. No database-backed result or search-output
        // reservation is needed.
        return Ok(CommandResult::plain(profile_help_json()));
    }
    // Parse the remaining query options under their own JSON schemas. The filter
    // retains its charge separately from other inputs; diagnose can later transfer
    // it into its trace.
    let QueryJsonOptions {
        filter,
        weights,
        metadata_boost,
    } = parse_query_options_in(&mut args, &context, &mut input_work, &mut filter_work)?;
    // A general search takes at most one positional query value. Vector-only
    // leaves positional arguments untouched so the common leftover check can
    // reject them.
    let query = if vector_only { None } else { args.next() };
    // Trim for the presence check. Forward the original string to the search
    // builder, which applies the core normalization used for retrieval and the
    // reported query.
    if !vector_only
        && query
            .as_ref()
            .map(|query| query.trim().is_empty())
            .unwrap_or(true)
        && embedding.is_none()
    {
        return Err(usage("search requires query text or --embedding"));
    }
    // Reject unconsumed tokens before opening the database. That prevents invalid
    // CLI arguments from reaching reader admission or storage recovery paths.
    args.reject_remaining()?;
    // Open without write admission. The search builder is created only after the
    // request syntax has been accepted.
    let db = Database::open_read_only(path)?;
    // Find the requested collection under a read-only database handle. The builder
    // takes ownership of the parsed inputs for this single execution.
    let mut builder = db
        .collection(&collection)?
        .search()
        .top_k(top_k)
        .filter(filter);
    // Apply only options the caller supplied. Leaving absent overrides unset lets
    // the builder resolve defaults for the chosen modalities and resource profile.
    if let Some(metadata_boost) = metadata_boost {
        builder = builder.metadata_boost(metadata_boost);
    }
    // Set a resource profile independently of score weights. The profile selects
    // resource allowances while explicit weights control score composition.
    if let Some(ParsedProfile::Profile(profile)) = profile {
        builder = builder.resource_profile(profile);
    }
    if let Some(weights) = weights {
        builder = builder.weights(weights);
    }
    // Install the two input channels independently. If both are present the same
    // builder can prepare a hybrid request.
    if let Some(query) = query {
        builder = builder.text(query);
    }
    // Pass the typed embedding rather than reparsing its file from inside the
    // builder. Its model name and dimension remain available to collection
    // validation.
    if let Some(embedding) = embedding {
        builder = builder.query_embedding(embedding);
    }
    // Keep the CLI operation context attached to execution. The returned result
    // owns a separate workspace guard for data that will outlive the query.
    let (result, mut work) = builder.execute_retained(Some(context.clone()))?;
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::InputRelease, &context, &work);
    // Execution has finished using the parsed input. Release its reservations and
    // the read-only handle before allocating the output tree; the retained search
    // result carries the workspace needed for that conversion.
    drop(input_work);
    drop(filter_work);
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::DatabaseClose, &context, &work);
    drop(db);
    // The result can still fail during budgeted JSON materialization after search
    // succeeds. Only the retained result represents live output here. Dropping the
    // database and parsed request before conversion reduces the peak without
    // changing the selected rows.
    Ok(CommandResult::retained(
        result.into_json_retained(&mut work)?,
        context,
        work.into_json(),
        JsonOutputSchema::SearchEnvelope,
    ))
}

// Diagnose a specific chunk and optionally write a report containing its search
// evidence. Both parts run from a consistent read view before constructing the
// final response.
fn diagnosis_command(mut args: Args, output: Option<String>) -> Result<CommandResult> {
    let path = args.required("path")?;
    // Separate filter ownership from charged parsed inputs at the start. This lets
    // a successful diagnosis keep filter evidence without extending the lifetime
    // of the embedding or boost allocations.
    let context = JsonOperationContext::standalone();
    let mut input_work = context.reservation();
    let mut filter_work = context.reservation();
    let collection = args
        .flag_value("--collection")?
        .ok_or_else(|| usage("--collection is required"))?;
    // Hold the owned selector string in this scope while a DiagnosisTarget borrows
    // it. The branch records which ID namespace the lookup must use.
    let target;
    // Prefer the internal selector when supplied. The external selector, if also
    // present, remains available to the general leftover-argument check.
    let key = match args.flag_value("--chunk-id")? {
        Some(id) => {
            target = id;
            DiagnosisTarget::Internal(&target)
        }
        None => {
            // Require an external selector only when no internal selector was
            // provided. A missing selector is diagnosed before attempting to read
            // query inputs.
            target = args
                .flag_value("--chunk-external-id")?
                .ok_or_else(|| usage("--chunk-id or --chunk-external-id is required"))?;
            DiagnosisTarget::External(&target)
        }
    };
    // Diagnosis reads query text from a named flag. That leaves room for a chunk
    // selector and optionally an embedding without interpreting either as free
    // positional text.
    let query = args.flag_value("--query")?;
    let top_k = parse_top_k_flag(&mut args)?;
    let profile = parse_profile_flag(&mut args)?;
    // Profile help produces a response on its own and cannot stand in for a
    // written diagnosis report. Handle it before loading an embedding or opening
    // an output file.
    if matches!(profile, Some(ParsedProfile::Help)) {
        if output.is_some() {
            return Err(usage("choose a resource profile to generate a report"));
        }
        args.reject_remaining()?;
        return Ok(CommandResult::plain(profile_help_json()));
    }
    // Keep the embedding's charge with the other owned input. The later result can
    // outlive the query request and carries a separate retained reservation.
    let embedding = args
        .flag_value("--embedding")?
        .map(|path| -> Result<_> {
            let (value, work) = read_embedding_file_in(&path, &context)?;
            input_work.merge(work);
            Ok(value)
        })
        .transpose()?;
    let QueryJsonOptions {
        filter,
        weights,
        metadata_boost,
    } = parse_query_options_in(&mut args, &context, &mut input_work, &mut filter_work)?;
    // Finish all syntax checks before acquiring the read view. Missing or
    // unsupported options should not create a report file or contend with database
    // writers.
    args.reject_remaining()?;
    // Hold a read view through diagnosis and the search rows written to the optional
    // report. If a consistent view is unavailable, stop before creating the output
    // file; two independently observed database states would make a misleading
    // diagnosis.
    // A read lock can be unavailable during mutation, or the opened file can cease
    // to be the generation selected by the pathname. Stop before creating a draft
    // if the storage layer cannot establish a stable view.
    let view = crate::storage::try_read_view(std::path::Path::new(&path), None, &context)?
        // A missing view is different from an I/O error opening the database. Add
        // retry guidance only in the no-view case and preserve other storage
        // failures.
        .ok_or_else(|| {
            Error::new(
                ErrorCode::LockContention,
                "cannot acquire a consistent diagnosis view; retry after the active write",
            )
        })?;
    // Open the database while the read view already covers the source. Separate
    // open failure occurs before taking ownership of a draft report.
    let db = Database::open_read_only(path)?;
    // Create the optional report only after acquiring the view and opening the
    // database. Without a report, the row visitor below discards the search
    // evidence.
    let mut report = output.as_deref().map(report::Report::create).transpose()?;
    // Keep the read view alive across the diagnostic call and its visitor. This is
    // the common interval in which the diagnosis and accompanying search rows are
    // observed.
    let (result, mut trace_work) = db.why_not_debug_with_metadata_boost_in(
        &collection,
        key,
        query,
        embedding,
        filter,
        weights,
        // Pass explicit boost presence independently from its value. The
        // diagnostic path needs to retain whether that option was requested when
        // describing score decisions.
        metadata_boost.is_some(),
        metadata_boost,
        top_k,
        profile.map(|profile| match profile {
            ParsedProfile::Profile(profile) => profile,
            ParsedProfile::Help => unreachable!("handled above"),
        }),
        Some(context.clone()),
        // Stream search rows to the optional report before the diagnostic call
        // returns. A report write error propagates through the visitor instead of
        // returning a success summary.
        |rows| match report.as_mut() {
            Some(file) => file.write_search(rows),
            None => Ok(()),
        },
    )?;
    // After the diagnostic and row visitor finish, release the shared read view
    // before converting results to JSON.
    drop(view);
    // Transfer the filter reservation to the trace before dropping parsed inputs.
    // Diagnostic output can retain filter data after the search itself is over.
    trace_work.merge(filter_work);
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::DatabaseClose, &context, &trace_work);
    // Close the database before allocating the final output representation. The
    // result already owns the evidence needed for conversion.
    drop(db);
    #[cfg(test)]
    tests::before_query_dto(tests::QueryDtoStage::InputRelease, &context, &trace_work);
    // Release the remaining input charge before building the output tree. Only the
    // retained diagnostic work still participates in output conversion.
    drop(input_work);
    // If JSON materialization fails, the report draft is dropped. A successful
    // search visit alone does not publish an HTML report because the target's
    // trace still has to be written.
    let result = result.into_json_retained(&mut trace_work)?;
    // A report request returns its completed path after finish succeeds. Without a
    // report, return the diagnosis itself under the why-not serialization schema.
    if let Some(file) = report {
        // Finish can fail while finalizing or publishing the staged report. Return
        // the path only after publication succeeds; errors preserve any
        // publication details supplied by the report writer.
        let output = file.finish(&result, key)?;
        return Ok(CommandResult::plain(object_from_pairs([(
            "output_path",
            JsonValue::String(output),
        )])));
    }
    // This branch has no file publication step. The typed result is handed to the
    // why-not serializer with its retained context.
    Ok(CommandResult::retained(
        result,
        context,
        trace_work.into_json(),
        JsonOutputSchema::WhyNotEnvelope,
    ))
}

// Collect the benchmark inputs and output destination before invoking the runner.
// The CLI wraps the returned artifact with its retained memory accounting.
fn benchmark_command(mut args: Args) -> Result<CommandResult> {
    let path = args.required("path")?;
    let profile = parse_profile_flag(&mut args)?;
    // Profile help needs no dataset, collection or result path. Return before
    // validating the options used only by a real benchmark run.
    if matches!(profile, Some(ParsedProfile::Help)) {
        args.reject_remaining()?;
        return Ok(CommandResult::plain(profile_help_json()));
    }
    let dataset_path = args
        .flag_value("--dataset")?
        .ok_or_else(|| usage("--dataset is required"))?;
    let durability = parse_durability_flag(&mut args)?;
    let output_path = args.flag_value("--output")?;
    let result_file = args.flag_value("--result-file")?;
    // Carry this switch into the benchmark runner. It controls whether a missing
    // or unsupported baseline is an acceptance failure. The CLI still parses the
    // result path and validates remaining arguments before invoking the runner.
    let require_baseline = args.has_flag("--require-baseline");
    // Accept the two result-file flag spellings as alternatives. Reject both
    // together before opening any output, even when their values name the same
    // file.
    let result_file = match (output_path, result_file) {
        (Some(_), Some(_)) => return Err(usage("use only one of --output or --result-file")),
        (Some(path), None) | (None, Some(path)) => Some(path),
        (None, None) => None,
    };
    // Pick the benchmark collection name only when the caller did not select one.
    // The runner receives the final name alongside the dataset and database paths.
    let collection = args
        .flag_value("--collection")?
        .unwrap_or_else(|| "benchmark".into());
    // Neither malformed trailing arguments nor duplicate options should start a
    // benchmark. Check them before handing the dataset and database paths to the
    // runner.
    args.reject_remaining()?;
    // Resolve the profile sentinel before running. Help has already returned and
    // an omitted selection uses the balanced benchmark preset.
    let profile = profile
        .map(|profile| match profile {
            ParsedProfile::Profile(profile) => profile,
            ParsedProfile::Help => unreachable!("handled above"),
        })
        .unwrap_or(ResourceProfile::Balanced);
    let output = crate::benchmark::run_benchmark(crate::benchmark::BenchmarkOptions {
        path,
        dataset_path,
        collection,
        profile,
        durability,
        result_file,
        require_baseline,
    })?;
    // Preserve the runner's context and reservation with its artifact. The output
    // serializer can then reuse the retained charge and report limits under the
    // benchmark kind.
    Ok(CommandResult::retained(
        output.value,
        output.context,
        output.work,
        JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
    ))
}

// Charge the owned argument's raw capacity while its parser works. A successful
// parse can return a second guard for retained output, which stays charged after
// the input string is freed.
fn with_json_argument<T>(
    value: String,
    kind: JsonInputKind,
    context: &JsonOperationContext,
    parse: impl FnOnce(&str) -> Result<T>,
) -> Result<T> {
    let mut input = context.reservation();
    input.grow(value.capacity(), kind)?;
    // Run the callback while its borrowed input remains valid. Drop the raw buffer
    // explicitly before the temporary reservation releases its charge.
    let result = parse(&value);
    drop(value);
    result
}

struct QueryJsonOptions {
    // Parsed metadata filter JSON passed to the search request.
    filter: JsonValue,
    // Explicit component weights when supplied on the command line.
    weights: Option<SearchWeights>,
    // Optional metadata boost configuration parsed from its CLI argument.
    metadata_boost: Option<MetadataBoostConfig>,
}

// Keep JSON parsing shared between search and diagnosis. Both commands use the
// same budget and error classification for their optional filter, weights and
// boost.
fn parse_query_options_in(
    args: &mut Args,
    context: &JsonOperationContext,
    input_work: &mut JsonReservation,
    filter_work: &mut JsonReservation,
) -> Result<QueryJsonOptions> {
    // An absent filter is the empty object. An explicit argument is parsed as one
    // business value so its own canonical-size and depth checks apply.
    let filter = args
        .flag_value("--filter")?
        .map(|value| -> Result<_> {
            // Charge the raw string while parsing the owned filter tree. Extract
            // the tree and transfer its reservation before returning it from the
            // option callback.
            let (value, work) =
                with_json_argument(value, JsonInputKind::Filter, context, |value| {
                    crate::json::parse_with_schema_in(
                        value,
                        JsonSchema::Value(JsonInputKind::Filter),
                        context,
                    )
                })?
                .into_parts();
            // Keep the filter charge separate. Diagnosis can move it into retained
            // output while search can release it with its request.
            filter_work.merge(work);
            Ok(value)
        })
        .transpose()?
        .unwrap_or_else(JsonValue::object);
    // Convert the parsed weights into the typed score parameters while the source
    // tree is still in scope. The returned scalars do not need a persistent DOM
    // reservation.
    let weights = args
        .flag_value("--weights")?
        .map(|value| {
            with_json_argument(value, JsonInputKind::Weights, context, |value| {
                parse_weights_in(value, context)
            })
        })
        .transpose()?;
    // Dispatch by the first non-whitespace character. A value beginning with an
    // opening brace is always treated as an inline object. A parse failure on that
    // branch cannot fall back to reading it as a pathname.
    let metadata_boost = args
        .flag_value("--metadata-boost")?
        .map(|value| -> Result<_> {
            let (value, work) = if value.trim_start().starts_with('{') {
                with_json_argument(value, JsonInputKind::MetadataBoost, context, |value| {
                    parse_metadata_boost_in(value, context)
                })?
            } else {
                parse_metadata_boost_in(&value, context)?
            };
            // The typed boost configuration can own keys and rule data. Merge its
            // retained charge with the other query inputs for the duration of the
            // request.
            input_work.merge(work);
            Ok(value)
        })
        .transpose()?;
    Ok(QueryJsonOptions {
        filter,
        weights,
        metadata_boost,
    })
}

#[cfg(test)]
fn parse_weights(value: &str) -> Result<SearchWeights> {
    parse_weights_in(value, &JsonOperationContext::standalone())
}

// Parse raw JSON under the weight option's policy, then validate its typed
// content. The two stages distinguish malformed representation from a structurally
// valid but invalid weight set.
fn parse_weights_in(value: &str, context: &JsonOperationContext) -> Result<SearchWeights> {
    let value = crate::json::parse_with_schema_in(
        value,
        JsonSchema::Value(JsonInputKind::Weights),
        context,
    )
    // Preserve budget and I/O errors with their machine-readable details. Only the
    // remaining parse failures become InvalidSearchWeights with the invalid_json
    // reason.
    .map_err(|err| {
        // A fixed processing limit can be reported under the same option error
        // code as bad syntax. Recognize its structured reason before applying the
        // generic wrapper.
        if crate::json::is_limit_error(&err)
            || matches!(err.code(), ErrorCode::IoError | ErrorCode::PermissionDenied)
        {
            return err.with_detail("field", "weights");
        }
        // Record invalid_json under both the shared and weights-specific fields.
        // Later callers can identify this as parse failure without matching the
        // diagnostic text.
        Error::new(
            ErrorCode::InvalidSearchWeights,
            format!("invalid search weights JSON: {}", err.message()),
        )
        .with_detail("field", "weights")
        .with_detail("reason", "invalid_json")
        .with_detail("weights_reason", "invalid_json")
    })?;
    // The typed validator runs after parsing so it sees the original members of
    // one complete object. Return its field-specific errors directly.
    SearchWeights::from_validated_json(&value.value)
}

#[cfg(test)]
fn parse_metadata_boost(value: &str) -> Result<MetadataBoostConfig> {
    parse_metadata_boost_in(value, &JsonOperationContext::standalone()).map(|(value, _)| value)
}

// Accept a literal JSON object or a file containing one. Use the same input kind
// so both forms share limit classification and typed validation.
fn parse_metadata_boost_in(
    value: &str,
    context: &JsonOperationContext,
) -> Result<(MetadataBoostConfig, JsonReservation)> {
    // Use the same leading-brace rule as the caller's raw argument accounting. If
    // that test changes, update both paths so the temporary string is charged in
    // the inline case.
    let parsed = if value.trim_start().starts_with('{') {
        crate::json::parse_with_schema_in(
            value,
            JsonSchema::Value(JsonInputKind::MetadataBoost),
            context,
        )
    } else {
        // Open a path only on the file branch. Since File::open is evaluated
        // before parse_reader_in and the error mapper below, an open failure keeps
        // its original I/O classification.
        crate::json::parse_reader_in(
            fs::File::open(value)?,
            JsonSchema::Value(JsonInputKind::MetadataBoost),
            context,
        )
    };
    // Retain structured policy failures. Normalize other parser errors to the
    // boost-specific malformed-JSON reason without conflating them with failures
    // of the typed rules.
    let parsed = parsed.map_err(|err| {
        if crate::json::is_limit_error(&err)
            || matches!(err.code(), ErrorCode::IoError | ErrorCode::PermissionDenied)
        {
            return err.with_detail("field", "metadata_boost");
        }
        // Separate JSON syntax failure from configuration semantics. Only the
        // former gets this invalid_json wrapper; typed validation runs after it.
        Error::new(
            ErrorCode::InvalidMetadataBoost,
            format!("invalid metadata boost JSON: {}", err.message()),
        )
        .with_detail("field", "metadata_boost")
        .with_detail("metadata_boost_reason", "invalid_json")
        .with_detail("reason", "invalid_json")
    })?;
    // Build the owned configuration before the parsed DOM drops. Return the typed
    // allocation's reservation so the caller can keep it charged while executing.
    MetadataBoostConfig::parse_validated_in(&parsed.value, context)
}

#[cfg(test)]
fn read_embedding_file(path: &str) -> Result<QueryEmbeddingInput> {
    read_embedding_file_in(path, &JsonOperationContext::standalone()).map(|(value, _)| value)
}

// Read the CLI embedding file's object form. The parser accounts for the vector
// field as its own value; this layer checks the expected fields and narrows
// numbers into the scorer's precision.
fn read_embedding_file_in(
    path: &str,
    context: &JsonOperationContext,
) -> Result<(QueryEmbeddingInput, JsonReservation)> {
    // Stream the file under the embedding schema and attach its path to parse
    // failures. This is a separate contract from parsing free-form metadata or
    // query JSONL rows.
    let value = crate::json::parse_reader_in(
        fs::File::open(path)?,
        JsonSchema::QueryEmbeddingObject,
        context,
    )
    // Attach the file name to parsing errors rather than discarding their
    // structured path, kind and limit details.
    .map_err(|error| error.with_detail("path", path))?;
    // Require an object before inspecting names. Error details preserve the path
    // so a caller can identify which input file failed.
    let obj = value.value.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            "query_embedding.json must be a JSON object",
        )
        .with_detail("path", path)
    })?;
    // Reject unknown members before extracting the known fields. A typo in a field
    // name must not be silently accepted alongside valid values.
    for field in obj.keys() {
        if !matches!(field.as_str(), "model_name" | "dim" | "vector") {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "query_embedding.json contains unsupported field",
            )
            .with_detail("field", field)
            .with_detail("path", path));
        }
    }
    // Require the model identifier even when the vector could otherwise be scored.
    // It is carried to collection query validation with the typed embedding.
    let model_name = obj
        .get("model_name")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "query_embedding.json must include string model_name",
            )
            .with_detail("field", "model_name")
            .with_detail("path", path)
        })?;
    // Use trim only to detect an empty identifier. The original spelling is
    // preserved when it is copied into the query input.
    if model_name.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::MissingRequiredField,
            "query_embedding.json model_name must not be empty",
        )
        .with_detail("field", "model_name")
        .with_detail("path", path));
    }
    // Read dimension as JSON number first so missing or wrong-type input receives
    // field-level detail before length validation.
    let dim = obj.get("dim").and_then(JsonValue::as_f64).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            "query_embedding.json must include numeric dim",
        )
        .with_detail("field", "dim")
        .with_detail("path", path)
    })?;
    // Require a finite positive integer dimension. The later equality check
    // compares that declaration with the actual array length.
    if !dim.is_finite() || dim <= 0.0 || dim.fract() != 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidVectorLength,
            "query_embedding.json dim must be a positive integer",
        )
        .with_detail("field", "dim")
        .with_detail("path", path));
    }
    // The cast can saturate for a large JSON number. Do not allocate from this
    // declaration: the real array length controls storage, and the later equality
    // check rejects a declaration that cannot describe the supplied array.
    let dim = dim as usize;
    // Inspect the actual vector array before reserving its typed storage. The
    // declared dimension can never force an allocation larger than the array
    // supplied.
    let array = obj
        .get("vector")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "query_embedding.json must include vector array",
            )
            .with_detail("field", "vector")
            .with_detail("path", path)
        })?;
    // Keep this guard separate from the parsed file's guard. The array and owned
    // f32 output coexist while conversion runs, and only the typed charge leaves
    // this function.
    let mut work = context.reservation();
    let mut vector = Vec::new();
    // Reserve from the supplied array length. Allocation failure occurs before a
    // partially converted result can be inserted into the request.
    work.reserve_vec(&mut vector, array.len(), JsonInputKind::QueryEmbedding)?;
    // Check every element and propagate a field error before handing a partial
    // vector to the caller. The following narrowing check uses the value that
    // scoring will receive.
    let values = array.iter().map(|value| {
        let value = value.as_f64().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidVectorLength,
                "query_embedding.json vector entries must be numbers",
            )
            .with_detail("field", "vector")
            .with_detail("path", path)
        })?;
        // A finite JSON number can overflow when narrowed to an embedding component.
        // Test the value in the precision the scorer actually receives.
        let value = value as f32;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "query_embedding.json vector entries must be finite f32 values",
            )
            .with_detail("field", "vector")
            .with_detail("path", path))
        }
    });
    // Fill the pre-admitted vector in input order. If any element fails, the local
    // guard releases its charge with the partial result.
    for value in values {
        vector.push(value?);
    }
    // Compare with the actual number of finite f32 components. This check runs
    // after token conversion, so an invalid entry is reported before a mismatch
    // between the header and array length.
    if vector.len() != dim {
        return Err(Error::new(
            ErrorCode::InvalidVectorLength,
            "query_embedding.json dim must match vector length",
        )
        .with_detail("field", "dim")
        .with_detail("dim", dim)
        .with_detail("actual_vector_length", vector.len())
        .with_detail("path", path));
    }
    // Copy the model name into the same owned request before returning. The
    // original JSON object and its temporary accounting can then be discarded.
    Ok((
        QueryEmbeddingInput {
            model_name: Some(work.copy_string(model_name, JsonInputKind::JsonlRecord)?),
            dim,
            vector,
        },
        work,
    ))
}

// The CLI consumes options by removing tokens while positional arguments advance a
// cursor. Any removal before that cursor must preserve the identity of the next
// positional token.
#[derive(Clone)]
struct Args {
    // Remaining argument strings, with the global JSON display flag removed before
    // command parsing.
    args: Vec<String>,
    // Index of the next positional token, adjusted when earlier option/value pairs
    // are removed.
    pos: usize,
}

impl Args {
    // Remove the global JSON display token before subcommand parsing. The display
    // decision is supplied separately to the outer entry point.
    fn new(args: Vec<String>) -> Self {
        Self {
            args: args.into_iter().filter(|arg| arg != "--json").collect(),
            pos: 0,
        }
    }

    // Advance position only when a token exists. Returning an owned copy leaves
    // the vector available to later named-option scans.
    fn next(&mut self) -> Option<String> {
        let value = self.args.get(self.pos).cloned();
        if value.is_some() {
            self.pos += 1;
        }
        value
    }

    // Use a structured usage error for a missing positional argument. The caller's
    // name identifies the field expected at the current cursor.
    fn required(&mut self, name: &str) -> Result<String> {
        self.next().ok_or_else(|| {
            usage(format!("{name} is required"))
                .with_detail("field", name)
                .with_detail("reason", "missing_required_argument")
        })
    }

    // Find a named option across the argument vector, including options written
    // before a positional token consumed earlier. Remove just its first occurrence
    // and its following value.
    fn flag_value(&mut self, flag: &str) -> Result<Option<String>> {
        if let Some(index) = self.args.iter().position(|arg| arg == flag) {
            // Report a missing token only at the end of the argument list. A
            // following token that looks like a switch is still passed to the
            // option-specific parser as its value.
            if index + 1 >= self.args.len() {
                return Err(usage(format!("{flag} requires a value"))
                    .with_detail("field", flag)
                    .with_detail("reason", "missing_value"));
            }
            // Remove the following value first so the flag's index still addresses
            // the same token when the second removal occurs.
            let value = self.args.remove(index + 1);
            self.args.remove(index);
            // The consumed positional prefix may include a removed flag, its
            // value, or both. For example, removing an entirely consumed pair
            // moves a cursor at position three back to one. An unconsumed pair
            // leaves the cursor alone.
            if self.pos > index {
                self.pos = self.pos.saturating_sub(1);
            }
            if self.pos > index {
                self.pos = self.pos.saturating_sub(1);
            }
            // Return the removed value directly. A repeated flag remains in the
            // argument vector and can be caught by leftover validation.
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    // Remove one occurrence of a boolean switch without adjusting the positional
    // cursor. Callers expect the switch to follow any positional arguments already
    // consumed.
    fn has_flag(&mut self, flag: &str) -> bool {
        if let Some(index) = self.args.iter().position(|arg| arg == flag) {
            self.args.remove(index);
            true
        } else {
            false
        }
    }

    // Report the first token at the unconsumed positional cursor. Call after
    // consuming recognized options so a stray positional token or unknown flag
    // gets a usage failure.
    fn reject_remaining(&self) -> Result<()> {
        if let Some(arg) = self.args.get(self.pos) {
            return Err(usage(format!("unexpected argument {arg}"))
                .with_detail("field", "argument")
                .with_detail("actual", arg)
                .with_detail("reason", "unexpected_argument"));
        }
        Ok(())
    }
}

// Create the command-line usage category used by argument validators. They attach
// their own field and reason details when a script can act on the specific
// failure.
fn usage(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::CliUsage, message)
}

#[cfg(test)]
mod tests {
    mod catalog_output;
    mod diagnostic_output;
    mod embedding_import;
    mod import_output;
    mod query_dto;
    mod summary_output;
    use super::*;
    use crate::model::{ChunkInput, DocumentInput, EmbeddingInput};

    #[derive(Clone, Copy, Eq, PartialEq)]
    pub(super) enum QueryDtoStage {
        InputRelease,
        DatabaseClose,
    }

    type QueryDtoHook = Box<dyn FnOnce(&JsonOperationContext, &crate::storage::RecordReservation)>;
    thread_local! {
        static QUERY_DTO_READY: std::cell::RefCell<Option<(QueryDtoStage, QueryDtoHook)>> = const { std::cell::RefCell::new(None) };
    }

    pub(super) fn before_query_dto(
        stage: QueryDtoStage,
        context: &JsonOperationContext,
        work: &crate::storage::RecordReservation,
    ) {
        let hook = QUERY_DTO_READY.with(|slot| {
            let mut pending = slot.borrow_mut();
            if pending
                .as_ref()
                .is_some_and(|(expected, _)| *expected == stage)
            {
                pending.take().map(|(_, hook)| hook)
            } else {
                None
            }
        });
        if let Some(hook) = hook {
            hook(context, work);
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("cofferdb-cli-{name}-{}.coffer", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn owned_json_arguments_reserve_capacity_before_parsing() {
        let context = JsonOperationContext::standalone();
        let mut held = context.reservation();
        held.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 512,
            JsonInputKind::Filter,
        )
        .unwrap();
        let mut raw = String::with_capacity(1024);
        raw.push_str("{}");
        let invoked = std::cell::Cell::new(false);
        let error = with_json_argument(raw, JsonInputKind::Filter, &context, |_| {
            invoked.set(true);
            Ok(())
        })
        .unwrap_err();
        assert!(!invoked.get());
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        drop(held);
        with_json_argument("{}".into(), JsonInputKind::Filter, &context, |_| Ok(())).unwrap();
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Filter,
            )
            .unwrap();
    }

    #[test]
    fn search_preprocessing_shares_inputs_and_core_does_not_charge_filter_twice() {
        let path = temp_path("query-json-handoff");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
        drop(db);
        let filter = format!(
            "{{\"chunk.x\":{{\"$in\":[{}null]}}}}",
            "null,".repeat(450_000)
        );
        let args = |filter: String| {
            vec![
                "coffer".into(),
                "search".into(),
                path.display().to_string(),
                "missing".into(),
                "--collection".into(),
                "notes".into(),
                "--filter".into(),
                filter,
            ]
        };
        let output = run(args(filter.clone())).unwrap();
        assert!(parse(&output)
            .unwrap()
            .get("result")
            .unwrap()
            .get("results")
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty());
        let boost = format!(
            r#"{{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{{"when":{{"chunk.y":{{"$in":[{}null]}}}},"boost":0.1}}]}}"#,
            "null,".repeat(350_000)
        );
        parse_metadata_boost(&boost).unwrap();
        let why_not = vec![
            "coffer".into(),
            "why-not".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            "missing".into(),
            "--query".into(),
            "missing".into(),
            "--filter".into(),
            filter.clone(),
            "--metadata-boost".into(),
            boost.clone(),
        ];
        let error = run(why_not).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("metadata_boost")
        );
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        let mut combined = args(filter);
        combined.extend(["--metadata-boost".into(), boost]);
        let error = run(combined).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("metadata_boost")
        );
        std::fs::remove_file(path).unwrap();
    }

    fn create_cli_schema_fixture(name: &str) -> (PathBuf, String, String) {
        create_cli_schema_fixture_with_model(name, "m")
    }

    fn create_cli_schema_fixture_with_model(name: &str, model: &str) -> (PathBuf, String, String) {
        let path = temp_path(name);
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: Some("Doc".into()),
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "stable cli result schema".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: model.into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        let document_id = col.get_document_by_external_id("doc").unwrap().document_id;
        let chunk_id = col.get_chunk_by_external_id("doc#1").unwrap().chunk_id;
        drop(col);
        drop(db);
        (path, document_id, chunk_id)
    }

    #[test]
    fn retained_benchmark_envelopes_keep_warnings_and_release_budget() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let warning = object_from_pairs([
                (
                    "code",
                    JsonValue::String("benchmark_baseline_not_run".into()),
                ),
                ("message", JsonValue::String("baseline unavailable".into())),
                ("reason", JsonValue::String("x".repeat(seed))),
            ]);
            let result = object_from_pairs([
                ("warnings", JsonValue::Array(vec![warning])),
                (
                    "status",
                    JsonValue::String("completed_cofferdb_runs".into()),
                ),
            ]);
            let expected = cli_envelope("benchmark", true, result.clone(), None, 7);
            let work = result
                .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
                .unwrap();
            let actual = success_output(
                "benchmark",
                CommandResult::retained(
                    result,
                    context.clone(),
                    work,
                    JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
                ),
                7,
            )
            .unwrap();
            assert_eq!(actual, expected);
            let parsed = parse(&actual).unwrap();
            let warning = &parsed.get("warnings").unwrap().as_array().unwrap()[0];
            assert_eq!(warning.get("severity").unwrap().as_str(), Some("warning"));
            assert_eq!(
                warning
                    .get("details")
                    .unwrap()
                    .get("reason")
                    .unwrap()
                    .as_str(),
                Some("x".repeat(seed).as_str())
            );
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        let context = JsonOperationContext::standalone();
        let result = JsonValue::String("\"".repeat(20_000));
        let work = result
            .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
            .unwrap();
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes() - 4000,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        let error = success_output(
            "benchmark",
            CommandResult::retained(
                result,
                context.clone(),
                work,
                JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
            ),
            7,
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("benchmark_artifact")
        );
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        drop(busy);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
    }

    #[test]
    fn query_envelopes_reserve_construction_and_preserve_the_wire_format() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let result = object_from_pairs([("query", JsonValue::String(format!("query-{seed}")))]);
            let expected = cli_envelope("search", true, result.clone(), None, seed);
            let mut work = result
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            let envelope = build_cli_envelope(
                "search",
                true,
                result,
                JsonValue::Array(Vec::new()),
                None,
                seed,
                &JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson),
            )
            .unwrap();
            let measured = envelope
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            drop(measured);
            let (actual, _) = envelope
                .pretty_retained_in(JsonOutputSchema::SearchEnvelope, &context, &mut work)
                .unwrap();
            assert_eq!(actual, expected);
        }

        let context = JsonOperationContext::standalone();
        let mut occupied = context.reservation();
        occupied
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 64,
                JsonInputKind::StoredJson,
            )
            .unwrap();
        let mut work = context.reservation();
        let error = build_cli_envelope(
            "search",
            true,
            JsonValue::Null,
            JsonValue::Array(Vec::new()),
            None,
            0,
            &JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson),
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            error
                .detail()
                .iter()
                .find(|(key, _)| key == "json_input_kind")
                .map(|(_, value)| value.as_str()),
            Some("stored_json")
        );
        drop(work);
        drop(occupied);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::StoredJson,
            )
            .unwrap();
    }

    #[test]
    fn cli_search_keeps_result_json_live_while_budgeting_serialized_output() {
        let (path, _, _) = create_cli_schema_fixture("search-output-budget");
        let db = Database::open(&path).unwrap();
        let collection = db.collection("notes").unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "wide".into(),
                title: None,
                source_uri: None,
                metadata: object_from_pairs([(
                    "payload",
                    JsonValue::String("x".repeat(4 * 1024 * 1024)),
                )]),
                chunks: (0..5)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("wide#{ordinal}"),
                        text: "payload".into(),
                        ordinal,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        assert_eq!(
            collection
                .search()
                .text("payload")
                .top_k(5)
                .execute()
                .unwrap()
                .results
                .len(),
            5
        );
        drop(collection);
        drop(db);
        let before = fs::read(&path).unwrap();
        let args = |top_k: &str| {
            vec![
                "coffer".into(),
                "search".into(),
                path.display().to_string(),
                "payload".into(),
                "--collection".into(),
                "notes".into(),
                "--top-k".into(),
                top_k.into(),
            ]
        };
        let (output, code) = run_with_exit_code(args("2")).unwrap();
        assert_eq!(code, 0);
        assert!(output.len() > crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES);
        drop(output);
        let error = run(args("5")).unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        assert_eq!(detail_value(&error, "json_input_kind"), Some("stored_json"));
        assert_eq!(fs::read(&path).unwrap(), before);
        run(args("1")).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn init_output_uses_success_envelope() {
        let path = temp_path("init-envelope");
        let output = run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("init")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        let path_str = path.display().to_string();
        assert_eq!(
            result.get("path").and_then(JsonValue::as_str),
            Some(path_str.as_str())
        );
        assert_eq!(
            result.get("created").and_then(JsonValue::as_bool),
            Some(true)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rebuild_index_output_uses_success_envelope() {
        let path = temp_path("rebuild-index-envelope");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
        ])
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "rebuild-index".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("rebuild-index")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("collection").and_then(JsonValue::as_str),
            Some("notes")
        );
        assert_eq!(
            result
                .get("index_suffix_status")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );
        assert_eq!(
            result.get("rebuild_needed").and_then(JsonValue::as_bool),
            Some(false)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn create_collection_accepts_vectors_normalized_flag() {
        let path = temp_path("create-vectors-normalized");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
            "--metric".into(),
            "dot_product".into(),
            "--vectors-normalized".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result
                .get("vectors_normalized")
                .and_then(JsonValue::as_bool),
            Some(true)
        );

        let listed = run(vec![
            "coffer".into(),
            "list-collections".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&listed).unwrap();
        let collections = json
            .get("result")
            .and_then(|result| result.get("collections"))
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(
            collections[0]
                .get("vectors_normalized")
                .and_then(JsonValue::as_bool),
            Some(true)
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn create_collection_rejects_unknown_arguments() {
        let path = temp_path("create-rejects-unknown");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();

        let err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
            "--unknown".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("unexpected argument --unknown"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn invalid_create_collection_dim_reports_machine_readable_detail() {
        let text_err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            "notes.coffer".into(),
            "notes".into(),
            "--dim".into(),
            "wide".into(),
        ])
        .unwrap_err();
        assert_eq!(text_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&text_err, "field"), Some("--dim"));
        assert_eq!(detail_value(&text_err, "actual"), Some("wide"));
        assert_eq!(
            detail_value(&text_err, "expected"),
            Some("positive_integer")
        );

        let zero_err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            "notes.coffer".into(),
            "notes".into(),
            "--dim".into(),
            "0".into(),
        ])
        .unwrap_err();
        assert_eq!(zero_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&zero_err, "field"), Some("--dim"));
        assert_eq!(detail_value(&zero_err, "actual"), Some("0"));
        assert_eq!(
            detail_value(&zero_err, "expected"),
            Some("positive_integer")
        );
    }

    #[test]
    fn invalid_create_collection_metric_reports_machine_readable_detail() {
        let err = run(vec![
            "coffer".into(),
            "create-collection".into(),
            "notes.coffer".into(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
            "--metric".into(),
            "angular".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&err, "field"), Some("--metric"));
        assert_eq!(detail_value(&err, "actual"), Some("angular"));
        assert_eq!(
            detail_value(&err, "expected"),
            Some("cosine|dot_product|euclidean")
        );
    }

    #[test]
    fn query_option_parsers_preserve_native_and_read_errors_without_user_error_reasons() {
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let weights = r#"{"bm25":1,"vector":0,"metadata":0}"#;
        let error =
            crate::test_alloc::fail_nth_allocation_once(1, || parse_weights_in(weights, &context))
                .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "field"), Some("weights"));
        assert_eq!(detail_value(&error, "weights_reason"), None);
        assert_eq!(detail_value(&error, "json_limit_reason"), None);
        assert_eq!(context.available_workspace_bytes(), capacity);
        parse_weights_in(weights, &context).unwrap();
        let boost = r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.x":true},"boost":0.1}]}"#;
        let error = crate::test_alloc::fail_nth_allocation_once(1, || {
            parse_metadata_boost_in(boost, &context)
        })
        .err()
        .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "field"), Some("metadata_boost"));
        assert_eq!(detail_value(&error, "metadata_boost_reason"), None);
        assert_eq!(detail_value(&error, "json_limit_reason"), None);
        assert_eq!(context.available_workspace_bytes(), capacity);
        drop(parse_metadata_boost_in(boost, &context).unwrap());
        let directory = temp_path("boost-read-error");
        fs::create_dir(&directory).unwrap();
        let error = parse_metadata_boost_in(directory.to_str().unwrap(), &context)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(detail_value(&error, "metadata_boost_reason"), None);
        assert_eq!(detail_value(&error, "json_limit_reason"), None);
        let envelope = parse(&error_output(Some("search"), &error)).unwrap();
        assert_eq!(
            envelope.get("error").unwrap().get("code").unwrap().as_str(),
            Some("io_error")
        );
        assert_eq!(context.available_workspace_bytes(), capacity);
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn metadata_boost_parser_accepts_file_path() {
        let path = temp_path("metadata-boost-config-json");
        std::fs::write(
            &path,
            r#"{
                "schema_version":"0.1",
                "rules":[{"when":{"chunk.importance":{"$gte":0.8}},"boost":0.1}],
                "combine":"sum",
                "max_boost":0.2
            }"#,
        )
        .unwrap();

        let config = parse_metadata_boost(&path.display().to_string()).unwrap();
        assert_eq!(config.rule_count(), 1);
        assert_eq!(config.hash().len(), 64);

        let inline = parse_metadata_boost(
            r#"{"schema_version":"0.1","rules":[{"when":{"chunk.importance":{"$gte":0.8}},"boost":0.1}],"combine":"sum","max_boost":0.2}"#,
        )
        .unwrap();
        assert_eq!(inline.rule_count(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn metadata_boost_parser_reports_invalid_metadata_boost_for_bad_json() {
        let err = parse_metadata_boost("{bad json").unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidMetadataBoost);
        assert!(err.message().contains("invalid metadata boost JSON"));
        assert_eq!(detail_value(&err, "field"), Some("metadata_boost"));
        assert_eq!(detail_value(&err, "reason"), Some("invalid_json"));
    }

    #[test]
    fn search_reports_invalid_metadata_boost_for_bad_json() {
        let err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
            "notes".into(),
            "--metadata-boost".into(),
            "{bad json".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidMetadataBoost);
        assert!(err.message().contains("invalid metadata boost JSON"));
    }

    #[test]
    fn repair_truncate_tail_removes_corrupt_suffix_only() {
        use std::io::Write;

        let path = temp_path("repair-truncate-tail");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
        ])
        .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"broken-index-suffix")
            .unwrap();

        let before = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(before.index_suffix_status, "rebuild_needed");
        assert!(before.repair_available);
        assert!(before.issues.iter().any(|issue| {
            issue
                .suggested_command
                .as_deref()
                .map(|command| {
                    command.contains("coffer rebuild-index")
                        && command.contains("--collection notes")
                })
                .unwrap_or(false)
        }));

        let output = run(vec![
            "coffer".into(),
            "repair".into(),
            path.display().to_string(),
            "--truncate-tail".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("repair")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("action").and_then(JsonValue::as_str),
            Some("truncate_tail")
        );
        assert_eq!(
            result.get("repaired").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert!(result
            .get("bytes_removed")
            .and_then(JsonValue::as_f64)
            .map(|bytes| bytes > 0.0)
            .unwrap_or(false));

        let after = Database::check_path(
            &path,
            CheckOptions {
                verify_indexes: true,
            },
        )
        .unwrap();
        assert_eq!(after.status, "ok");
        assert_eq!(after.index_suffix_status, "missing");
        assert_eq!(after.live_documents, 0);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            after.critical_end_offset
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_command_uses_check_issue_exit_code() {
        use std::io::Write;

        let path = temp_path("check-issue-exit-code");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
        ])
        .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"corrupt-tail")
            .unwrap();

        let (output, exit_code) = run_with_exit_code(vec![
            "coffer".into(),
            "check".into(),
            path.display().to_string(),
        ])
        .unwrap();
        assert_eq!(exit_code, 3);
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("check")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("status").and_then(JsonValue::as_str),
            Some("warning")
        );
        assert_eq!(
            result.get("repair_available").and_then(JsonValue::as_bool),
            Some(true)
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_accepts_advertised_verify_indexes_flag() {
        let path = temp_path("check-verify-indexes-flag");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "check".into(),
            path.display().to_string(),
            "--verify-indexes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("check")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(result.get("status").and_then(JsonValue::as_str), Some("ok"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn snapshot_command_copies_consistent_database() {
        let path = temp_path("snapshot-source");
        let output_path = temp_path("snapshot-output");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
        ])
        .unwrap();
        let source_len = std::fs::metadata(&path).unwrap().len();

        let output = run(vec![
            "coffer".into(),
            "snapshot".into(),
            path.display().to_string(),
            output_path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("snapshot")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("bytes_copied").and_then(JsonValue::as_f64),
            Some(source_len as f64)
        );
        assert_eq!(
            result
                .get("index_suffix_status")
                .and_then(JsonValue::as_str),
            Some("persistent_checkpoint")
        );
        assert_eq!(std::fs::metadata(&output_path).unwrap().len(), source_len);

        let snapshot_collections = run(vec![
            "coffer".into(),
            "list-collections".into(),
            output_path.display().to_string(),
        ])
        .unwrap();
        let snapshot_json = parse(&snapshot_collections).unwrap();
        let collections = snapshot_json
            .get("result")
            .and_then(JsonValue::as_object)
            .unwrap()
            .get("collections")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(
            collections[0].get("name").and_then(JsonValue::as_str),
            Some("notes")
        );

        let err = run(vec![
            "coffer".into(),
            "snapshot".into(),
            path.display().to_string(),
            output_path.display().to_string(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchParams);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn snapshot_command_requires_writer_lock() {
        let path = temp_path("snapshot-lock-source");
        let output_path = temp_path("snapshot-lock-output");
        let _db = Database::create(&path, DatabaseConfig::default()).unwrap();

        let err = run(vec![
            "coffer".into(),
            "snapshot".into(),
            path.display().to_string(),
            output_path.display().to_string(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::LockContention);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn get_commands_use_public_result_schema() {
        let (path, _document_id, chunk_id) = create_cli_schema_fixture("get-schema");
        let output = run(vec![
            "coffer".into(),
            "get-document".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--external-id".into(),
            "doc".into(),
            "--include-chunks".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(result.get("document"), Some(JsonValue::Object(_))));
        assert_eq!(
            result.get("external_id").and_then(JsonValue::as_str),
            Some("doc")
        );
        assert_eq!(
            result.get("chunks_count").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result
                .get("chunks")
                .and_then(JsonValue::as_array)
                .map(|chunks| chunks.len()),
            Some(1)
        );

        let output = run(vec![
            "coffer".into(),
            "get-chunk".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            chunk_id,
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(result.get("chunk"), Some(JsonValue::Object(_))));
        assert!(matches!(result.get("document"), Some(JsonValue::Object(_))));
        assert_eq!(
            result.get("chunk_external_id").and_then(JsonValue::as_str),
            Some("doc#1")
        );
        assert_eq!(
            result.get("embedding_present").and_then(JsonValue::as_bool),
            Some(true)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn get_commands_serialize_full_depth_metadata_without_wrapper_penalty() {
        let (path, _, _) = create_cli_schema_fixture("get-deep-metadata");
        let db = Database::open(&path).unwrap();
        let col = db.collection("notes").unwrap();
        let mut nested = JsonValue::String("depth-boundary".into());
        for _ in 0..127 {
            nested = JsonValue::Array(vec![nested]);
        }
        let metadata = object_from_pairs([("nested", nested)]);
        col.insert_document(crate::DocumentInput {
            external_id: "deep".into(),
            title: None,
            source_uri: None,
            metadata: metadata.clone(),
            chunks: vec![crate::ChunkInput {
                chunk_external_id: "deep#0".into(),
                ordinal: 0,
                text: "payload".into(),
                metadata,
                source_span: None,
            }],
        })
        .unwrap();
        drop((col, db));
        for (command, flags) in [
            (
                "get-document",
                vec!["--external-id", "deep", "--include-chunks"],
            ),
            ("get-chunk", vec!["--chunk-external-id", "deep#0"]),
        ] {
            let mut args = vec![
                "coffer".into(),
                command.into(),
                path.display().to_string(),
                "--collection".into(),
                "notes".into(),
            ];
            args.extend(flags.into_iter().map(String::from));
            let (output, exit_code) = run_with_exit_code(args).unwrap();
            assert_eq!(exit_code, 0);
            assert_eq!(output.matches("depth-boundary").count(), 2);
            assert!(output.contains("\"ok\": true"));
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn delete_and_drop_commands_report_public_result_schema() {
        let (path, _document_id, chunk_id) = create_cli_schema_fixture("delete-chunk-schema");
        let output = run(vec![
            "coffer".into(),
            "delete-chunk".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            chunk_id,
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("chunk_external_id").and_then(JsonValue::as_str),
            Some("doc#1")
        );
        assert_eq!(
            result.get("embedding_deleted").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert!(result
            .get("tombstone_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .starts_with("tombstone:chunk:"));
        let _ = std::fs::remove_file(path);

        let (path, document_id, _chunk_id) = create_cli_schema_fixture("delete-document-schema");
        let output = run(vec![
            "coffer".into(),
            "delete-document".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--document-id".into(),
            document_id,
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("external_id").and_then(JsonValue::as_str),
            Some("doc")
        );
        assert_eq!(
            result.get("chunks_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("embeddings_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert!(result
            .get("tombstone_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .starts_with("tombstone:document:"));
        let _ = std::fs::remove_file(path);

        let (path, _document_id, _chunk_id) = create_cli_schema_fixture("drop-schema");
        let output = run(vec![
            "coffer".into(),
            "drop-collection".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("documents_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("chunks_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("embeddings_deleted").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert!(result
            .get("tombstone_id")
            .and_then(JsonValue::as_str)
            .unwrap()
            .starts_with("tombstone:collection:"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn why_not_output_uses_success_envelope_and_query_context() {
        let path = temp_path("why-not-envelope");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "wal recovery cli why not".into(),
                ordinal: 1,
                metadata: parse(r#"{"lang":"python"}"#).unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        drop(col);
        drop(db);

        let output = run(vec![
            "coffer".into(),
            "why-not".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--query".into(),
            "wal recovery".into(),
            "--chunk-id".into(),
            "doc#1".into(),
            "--filter".into(),
            r#"{"chunk.lang":"rust"}"#.into(),
            "--weights".into(),
            r#"{"bm25":1.0,"vector":0.0,"metadata":0.0}"#.into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("why-not")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        let context = result
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context.get("weights_source").and_then(JsonValue::as_str),
            Some("user_override")
        );
        let expected = result
            .get("expected_chunk")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            expected
                .get("chunk_external_id")
                .and_then(JsonValue::as_str),
            Some("doc#1")
        );
        let codes = result
            .get("diagnosis_codes")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert!(codes
            .iter()
            .any(|code| code.as_str() == Some("filtered_by_metadata")));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn stats_include_index_when_requested() {
        let path = temp_path("stats-include-index");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
        ])
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "stats".into(),
            path.display().to_string(),
            "--include-index".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(
            result.get("format_version"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("file_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("data_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("index_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert_eq!(
            result
                .get("background_compaction_enabled")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        let index_stats = result
            .get("index_stats")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            index_stats
                .get("rebuild_needed")
                .and_then(JsonValue::as_bool),
            Some(false)
        );

        let output = run(vec![
            "coffer".into(),
            "collection-stats".into(),
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--include-index".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(
            result.get("format_version"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("data_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("index_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert_eq!(
            result
                .get("background_compaction_enabled")
                .and_then(JsonValue::as_bool),
            Some(false)
        );
        assert!(matches!(
            result.get("index_stats"),
            Some(JsonValue::Object(_))
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compact_output_uses_public_size_aliases() {
        let path = temp_path("compact-schema");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let output = run(vec![
            "coffer".into(),
            "compact".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert!(matches!(
            result.get("before_file_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("after_file_size_bytes"),
            Some(JsonValue::Number(_))
        ));
        assert!(matches!(
            result.get("records_removed"),
            Some(JsonValue::Number(_))
        ));
        let output = run(vec![
            "coffer".into(),
            "compact".into(),
            path.display().to_string(),
            "--rebuild-indexes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("compact")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn import_commands_record_effective_durability() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("cofferdb-cli-durability-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("durability.coffer");
        std::fs::write(
            dir.join("documents.jsonl"),
            r#"{"external_id":"doc","metadata":{}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("chunks.jsonl"),
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"durability import","metadata":{}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_model":null,"embedding_dim":2,"distance_metric":"cosine"}"#,
        )
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "import".into(),
            db_path.display().to_string(),
            "--manifest".into(),
            dir.join("manifest.json").display().to_string(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let envelope_warnings = json.get("warnings").and_then(JsonValue::as_array).unwrap();
        assert_eq!(envelope_warnings.len(), 2);
        assert_eq!(
            envelope_warnings[0].get("code").and_then(JsonValue::as_str),
            Some("missing_reproducibility_hash")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result
                .get("warnings")
                .and_then(JsonValue::as_array)
                .map(|warnings| warnings.len()),
            Some(2)
        );
        assert_eq!(
            result.get("durability").and_then(JsonValue::as_str),
            Some("fast")
        );
        assert_eq!(
            result.get("documents_imported").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("chunks_imported").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(result.get("skipped").and_then(JsonValue::as_f64), Some(0.0));
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(result.get("errors").and_then(JsonValue::as_f64), Some(0.0));

        let output = run(vec![
            "coffer".into(),
            "import".into(),
            db_path.display().to_string(),
            "--manifest".into(),
            dir.join("manifest.json").display().to_string(),
            "--on-conflict".into(),
            "replace".into(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(2.0)
        );

        std::fs::write(
            dir.join("embeddings.jsonl"),
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        )
        .unwrap();
        let output = run(vec![
            "coffer".into(),
            "import-embeddings".into(),
            db_path.display().to_string(),
            dir.join("embeddings.jsonl").display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--on-conflict".into(),
            "replace".into(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("durability").and_then(JsonValue::as_str),
            Some("fast")
        );
        assert_eq!(
            result
                .get("embeddings_imported")
                .and_then(JsonValue::as_f64),
            Some(1.0)
        );
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            result.get("missing_chunks").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            result.get("model_mismatches").and_then(JsonValue::as_f64),
            Some(0.0)
        );

        let output = run(vec![
            "coffer".into(),
            "import-embeddings".into(),
            db_path.display().to_string(),
            dir.join("embeddings.jsonl").display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--durability".into(),
            "fast".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("replaced").and_then(JsonValue::as_f64),
            Some(1.0)
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn profile_help_reports_exact_names_and_candidate_limits_without_aliases() {
        let output = run(vec!["coffer".into(), "profile-help".into()]).unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("default").and_then(JsonValue::as_str),
            Some("balanced")
        );
        let profiles = result
            .get("profiles")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(profiles.len(), 3);
        for (profile, (name, candidates, cache)) in profiles.iter().zip([
            ("expanded", 500.0, 512.0),
            ("balanced", 200.0, 256.0),
            ("constrained", 50.0, 64.0),
        ]) {
            assert_eq!(profile.get("cli").and_then(JsonValue::as_str), Some(name));
            assert_eq!(profile.get("json").and_then(JsonValue::as_str), Some(name));
            assert_eq!(profile.get("alias"), Some(&JsonValue::Null));
            let description = profile
                .get("description")
                .and_then(JsonValue::as_str)
                .unwrap();
            let scenarios: &[&str] = match name {
                "expanded" => &[
                    "retrieval experiments",
                    "candidate inspection",
                    "quality evaluation",
                ],
                "balanced" => &["everyday local RAG", "code search"],
                "constrained" => &["limited host resources"],
                _ => unreachable!(),
            };
            for scenario in scenarios {
                assert!(
                    description.contains(scenario),
                    "{name} help must describe the applicable scenario: {scenario}"
                );
            }
            for field in ["bm25", "vector", "fusion"] {
                assert_eq!(
                    profile
                        .get("candidate_limits")
                        .unwrap()
                        .get(field)
                        .and_then(JsonValue::as_f64),
                    Some(candidates)
                );
            }
            assert_eq!(
                profile.get("cache_budget_mb").and_then(JsonValue::as_f64),
                Some(cache)
            );
        }
    }

    #[test]
    fn invalid_profile_and_durability_report_machine_readable_detail() {
        let profile_err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
            "notes".into(),
            "--profile".into(),
            "turbo".into(),
        ])
        .unwrap_err();
        assert_eq!(profile_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&profile_err, "field"), Some("--profile"));
        assert_eq!(detail_value(&profile_err, "actual"), Some("turbo"));
        assert_eq!(
            detail_value(&profile_err, "expected"),
            Some("expanded|balanced|constrained|help")
        );

        let durability_err = run(vec![
            "coffer".into(),
            "benchmark".into(),
            "notes.coffer".into(),
            "--dataset".into(),
            "dataset".into(),
            "--durability".into(),
            "unsafe".into(),
        ])
        .unwrap_err();
        assert_eq!(durability_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&durability_err, "field"), Some("--durability"));
        assert_eq!(detail_value(&durability_err, "actual"), Some("unsafe"));
        assert_eq!(
            detail_value(&durability_err, "expected"),
            Some("commit|fast")
        );
    }

    #[test]
    fn invalid_top_k_reports_machine_readable_detail() {
        let search_err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
            "notes".into(),
            "--top-k".into(),
            "many".into(),
        ])
        .unwrap_err();
        assert_eq!(search_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&search_err, "field"), Some("--top-k"));
        assert_eq!(detail_value(&search_err, "actual"), Some("many"));
        assert_eq!(
            detail_value(&search_err, "expected"),
            Some("positive_integer")
        );

        let why_not_err = run(vec![
            "coffer".into(),
            "why-not".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            "chunk_1".into(),
            "--top-k".into(),
            "ten".into(),
        ])
        .unwrap_err();
        assert_eq!(why_not_err.code(), ErrorCode::CliUsage);
        assert_eq!(detail_value(&why_not_err, "field"), Some("--top-k"));
        assert_eq!(detail_value(&why_not_err, "actual"), Some("ten"));
        assert_eq!(
            detail_value(&why_not_err, "expected"),
            Some("positive_integer")
        );
    }

    #[test]
    fn help_reports_public_cli_surface_and_flags() {
        let output = run(vec!["coffer".into(), "help".into()]).unwrap();
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("help")
        );
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        let commands = result
            .get("commands")
            .and_then(JsonValue::as_array)
            .unwrap();
        for name in [
            "init",
            "create-collection",
            "search",
            "vector-search",
            "stats",
            "check",
            "compact",
            "benchmark",
            "export",
            "snapshot",
            "embed-jsonl",
            "import-dir",
        ] {
            assert!(
                commands
                    .iter()
                    .any(|command| command.get("name").and_then(JsonValue::as_str) == Some(name)),
                "missing help command {name}"
            );
        }
        let search = help_command_by_name(commands, "search");
        assert_eq!(
            search.get("synopsis").and_then(JsonValue::as_str),
            Some(
                format!(
                    "coffer search <path> [<query>] [--top-k <1..={}>]",
                    crate::query::MAX_SEARCH_TOP_K
                )
                .as_str()
            )
        );
        for flag in [
            "--collection",
            "--top-k",
            "--profile",
            "--filter",
            "--weights",
            "--metadata-boost",
            "--embedding",
        ] {
            assert!(help_command_has_flag(search, flag), "search missing {flag}");
        }
        let create_collection = help_command_by_name(commands, "create-collection");
        for flag in [
            "--dim",
            "--metric",
            "--embedding-model",
            "--vectors-normalized",
        ] {
            assert!(
                help_command_has_flag(create_collection, flag),
                "create-collection missing {flag}"
            );
        }
        let benchmark = help_command_by_name(commands, "benchmark");
        for flag in [
            "--dataset",
            "--profile",
            "--durability",
            "--output",
            "--result-file",
            "--require-baseline",
        ] {
            assert!(
                help_command_has_flag(benchmark, flag),
                "benchmark missing {flag}"
            );
        }
        let embed_jsonl = help_command_by_name(commands, "embed-jsonl");
        let embed_synopsis = embed_jsonl
            .get("synopsis")
            .and_then(JsonValue::as_str)
            .unwrap();
        assert!(embed_synopsis.contains("Does not open a database or store the API key"));
        assert!(embed_synopsis.contains("A failed generation batch does not write a database"));
        let import_dir = help_command_by_name(commands, "import-dir");
        let import_dir_synopsis = import_dir
            .get("synopsis")
            .and_then(JsonValue::as_str)
            .unwrap();
        assert!(import_dir_synopsis
            .contains("writes text artifacts only and does not open the database"));
        assert!(import_dir_synopsis.contains(
            "A failed generation batch does not change the database; core batches committed after import starts are kept"
        ));
        let import = help_command_by_name(commands, "import");
        for flag in ["--manifest", "--on-conflict", "--durability"] {
            assert!(help_command_has_flag(import, flag), "import missing {flag}");
        }
        for flag in ["--collection", "--format", "--documents"] {
            assert!(
                !help_command_has_flag(import, flag),
                "import should not advertise unimplemented convenience flag {flag}"
            );
        }
        let check = help_command_by_name(commands, "check");
        assert!(help_command_has_flag(check, "--verify-indexes"));
        let compact = help_command_by_name(commands, "compact");
        assert!(help_command_has_flag(compact, "--rebuild-indexes"));
        let rebuild_index = help_command_by_name(commands, "rebuild-index");
        assert_eq!(
            rebuild_index.get("synopsis").and_then(JsonValue::as_str),
            Some("coffer rebuild-index <path> --collection <name>")
        );
        assert!(help_command_has_flag(rebuild_index, "--collection"));
        let import_embeddings = help_command_by_name(commands, "import-embeddings");
        assert!(help_command_has_flag(import_embeddings, "--on-conflict"));
        assert_eq!(
            help_command_by_name(commands, "why-not")
                .get("stability")
                .and_then(JsonValue::as_str),
            Some("debug")
        );
        assert_eq!(
            help_command_by_name(commands, "snapshot")
                .get("stability")
                .and_then(JsonValue::as_str),
            Some("future")
        );
        let exit_codes = result
            .get("exit_codes")
            .and_then(JsonValue::as_array)
            .unwrap();
        for (code, meaning) in [
            (0.0, "success"),
            (1.0, "user_or_input_error"),
            (2.0, "file_or_corruption_error"),
            (3.0, "check_issues"),
            (4.0, "lock_contention"),
        ] {
            assert!(
                exit_codes.iter().any(|entry| {
                    entry.get("code").and_then(JsonValue::as_f64) == Some(code)
                        && entry.get("meaning").and_then(JsonValue::as_str) == Some(meaning)
                }),
                "missing exit code {code} {meaning}"
            );
        }
    }

    fn help_command_by_name<'a>(commands: &'a [JsonValue], name: &str) -> &'a JsonValue {
        commands
            .iter()
            .find(|command| command.get("name").and_then(JsonValue::as_str) == Some(name))
            .unwrap_or_else(|| panic!("missing help command {name}"))
    }

    fn help_command_has_flag(command: &JsonValue, flag: &str) -> bool {
        command
            .get("flags")
            .and_then(JsonValue::as_array)
            .map(|flags| {
                flags
                    .iter()
                    .any(|value| value.as_str().map(|value| value == flag).unwrap_or(false))
            })
            .unwrap_or(false)
    }

    #[test]
    fn weights_require_all_fields() {
        let err = parse_weights(r#"{"bm25":1.0}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("numeric vector"));
        assert_eq!(detail_value(&err, "field"), Some("weights.vector"));
        assert_eq!(detail_value(&err, "expected"), Some("number"));
    }

    #[test]
    fn weights_must_sum_to_one() {
        let err = parse_weights(r#"{"bm25":1.0,"vector":1.0,"metadata":0.0}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("sum to 1.0"));
        assert_eq!(detail_value(&err, "field"), Some("weights"));
        assert_eq!(detail_value(&err, "reason"), Some("sum_mismatch"));
        assert_eq!(detail_value(&err, "expected_sum"), Some("1"));
        assert_eq!(detail_value(&err, "actual_sum"), Some("2"));
    }

    #[test]
    fn weights_reject_unknown_fields_with_machine_readable_detail() {
        let err =
            parse_weights(r#"{"bm25":1.0,"vector":0.0,"metadata":0.0,"debug":true}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("unsupported field debug"));
        assert_eq!(detail_value(&err, "field"), Some("weights.debug"));
        assert_eq!(detail_value(&err, "reason"), Some("unsupported_field"));
    }

    #[test]
    fn vector_search_rejects_weights_and_metadata_boost() {
        let missing_embedding = run(vec![
            "coffer".into(),
            "vector-search".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap_err();
        assert_eq!(missing_embedding.code(), ErrorCode::CliUsage);
        assert!(missing_embedding.message().contains("--embedding"));

        let err = run(vec![
            "coffer".into(),
            "vector-search".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
            "--weights".into(),
            r#"{"bm25":0.0,"vector":1.0,"metadata":0.0}"#.into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--weights"));
        assert_eq!(detail_value(&err, "command"), Some("vector-search"));
        assert_eq!(detail_value(&err, "field"), Some("--weights"));
        assert_eq!(detail_value(&err, "reason"), Some("unsupported_option"));

        let err = run(vec![
            "coffer".into(),
            "vector-search".into(),
            "notes.coffer".into(),
            "--collection".into(),
            "notes".into(),
            "--metadata-boost".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--metadata-boost"));
        assert_eq!(detail_value(&err, "command"), Some("vector-search"));
        assert_eq!(detail_value(&err, "field"), Some("--metadata-boost"));
        assert_eq!(detail_value(&err, "reason"), Some("unsupported_option"));
    }

    #[test]
    fn query_embedding_file_requires_public_object_schema() {
        let path = temp_path("query-embedding-schema");

        std::fs::write(&path, r#"[1.0,0.0]"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);

        std::fs::write(&path, r#"{"dim":2,"vector":[1.0,0.0]}"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("model_name"));

        std::fs::write(&path, r#"{"model_name":"m","dim":2,"vector":[1.0]}"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("1"));

        std::fs::write(&path, r#"{"model_name":"m","dim":2,"vector":[1.0,"bad"]}"#).unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "field"), Some("vector"));

        std::fs::write(
            &path,
            r#"{"model_name":"m","dim":2,"vector":[1.0,0.0],"chunk_external_id":"unexpected"}"#,
        )
        .unwrap();
        let err = read_embedding_file(&path.display().to_string()).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("chunk_external_id"));

        std::fs::write(&path, r#"{"model_name":"m","dim":2,"vector":[1.0,0.0]}"#).unwrap();
        let input = read_embedding_file(&path.display().to_string()).unwrap();
        assert_eq!(input.model_name.as_deref(), Some("m"));
        assert_eq!(input.dim, 2);
        assert_eq!(input.vector, vec![1.0, 0.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn search_accepts_embedding_without_query_text() {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "cofferdb-cli-queryless-search-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("queryless.coffer");
        let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                tokenizer: Default::default(),
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        col.insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: parse("{}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "doc#1".into(),
                text: "queryless vector search".into(),
                ordinal: 1,
                metadata: parse("{}").unwrap(),
                source_span: None,
            }],
        })
        .unwrap();
        col.put_embeddings(vec![crate::model::EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
        drop(col);
        drop(db);
        let embedding_path = dir.join("query_embedding.json");
        std::fs::write(
            &embedding_path,
            r#"{"model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        )
        .unwrap();

        let output = run(vec![
            "coffer".into(),
            "search".into(),
            db_path.display().to_string(),
            "--embedding".into(),
            embedding_path.display().to_string(),
            "--collection".into(),
            "notes".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        let context = result
            .get("query_context")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert_eq!(
            context.get("text_present").and_then(JsonValue::as_bool),
            Some(false)
        );
        assert_eq!(
            context
                .get("embedding_present")
                .and_then(JsonValue::as_bool),
            Some(true)
        );
        let candidate_counts = result
            .get("candidate_counts")
            .and_then(JsonValue::as_object)
            .unwrap();
        assert!(matches!(
            candidate_counts.get("vector"),
            Some(JsonValue::Number(_))
        ));
        let results = result.get("results").and_then(JsonValue::as_array).unwrap();
        assert_eq!(
            results[0]
                .get("chunk_external_id")
                .and_then(JsonValue::as_str),
            Some("doc#1")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn error_output_uses_failure_envelope() {
        let err = Error::new(ErrorCode::CliUsage, "missing command");
        let output = error_output(None, &err);
        let json = parse(&output).unwrap();
        assert_eq!(
            json.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        assert_eq!(
            json.get("command").and_then(JsonValue::as_str),
            Some("unknown")
        );
        assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(false));
        let error = json.get("error").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            error.get("code").and_then(JsonValue::as_str),
            Some("cli_usage")
        );
        assert_eq!(
            error.get("message").and_then(JsonValue::as_str),
            Some("missing command")
        );
        assert!(matches!(error.get("detail"), Some(JsonValue::Object(_))));
    }

    #[test]
    fn resource_exhaustion_preserves_the_failure_envelope_and_exact_details() {
        let error = Error::resource_exhausted(
            "temporary_disk",
            "temporary_staging",
            u64::MAX,
            u128::from(u64::MAX) + 1,
        );
        let output = parse(&error_output(Some("import"), &error)).unwrap();
        assert_eq!(
            output.get("schema_version").and_then(JsonValue::as_str),
            Some("0.1")
        );
        assert_eq!(
            output.get("command").and_then(JsonValue::as_str),
            Some("import")
        );
        assert_eq!(output.get("ok").and_then(JsonValue::as_bool), Some(false));
        let error = output.get("error").unwrap();
        crate::test_error_contract::validate_error(error).unwrap();
        assert_eq!(
            error.get("code").and_then(JsonValue::as_str),
            Some("resource_exhausted")
        );
        assert_eq!(error.get("detail").unwrap(), &parse(r#"{"resource":"temporary_disk","operation":"temporary_staging","limit_bytes":"18446744073709551615","requested_bytes":"18446744073709551616"}"#).unwrap());
    }

    #[test]
    fn cli_rejects_unexpected_arguments() {
        let path = temp_path("unexpected-argument");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        let err = run(vec![
            "coffer".into(),
            "stats".into(),
            path.display().to_string(),
            "--mystery".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("unexpected argument --mystery"));
        assert_eq!(detail_value(&err, "field"), Some("argument"));
        assert_eq!(detail_value(&err, "actual"), Some("--mystery"));
        assert_eq!(detail_value(&err, "reason"), Some("unexpected_argument"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unknown_command_reports_machine_readable_detail() {
        let err = run(vec!["coffer".into(), "frobnicate".into()]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("unknown command frobnicate"));
        assert_eq!(detail_value(&err, "field"), Some("command"));
        assert_eq!(detail_value(&err, "actual"), Some("frobnicate"));
        assert_eq!(detail_value(&err, "reason"), Some("unknown_command"));
    }

    #[test]
    fn missing_flag_value_reports_machine_readable_detail() {
        let err = run(vec![
            "coffer".into(),
            "search".into(),
            "notes.coffer".into(),
            "wal".into(),
            "--collection".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--collection requires a value"));
        assert_eq!(detail_value(&err, "field"), Some("--collection"));
        assert_eq!(detail_value(&err, "reason"), Some("missing_value"));
    }

    #[test]
    fn missing_required_argument_reports_machine_readable_detail() {
        let err = run(vec!["coffer".into(), "stats".into()]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("path is required"));
        assert_eq!(detail_value(&err, "field"), Some("path"));
        assert_eq!(
            detail_value(&err, "reason"),
            Some("missing_required_argument")
        );
    }

    #[test]
    fn export_accepts_explicit_jsonl_format() {
        let path = temp_path("export-format");
        run(vec![
            "coffer".into(),
            "init".into(),
            path.display().to_string(),
        ])
        .unwrap();
        run(vec![
            "coffer".into(),
            "create-collection".into(),
            path.display().to_string(),
            "notes".into(),
            "--dim".into(),
            "2".into(),
        ])
        .unwrap();
        let mut out = std::env::temp_dir();
        out.push(format!("cofferdb-cli-export-format-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let output = run(vec![
            "coffer".into(),
            "export".into(),
            path.display().to_string(),
            out.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--format".into(),
            "jsonl".into(),
        ])
        .unwrap();
        let json = parse(&output).unwrap();
        let result = json.get("result").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            result.get("format").and_then(JsonValue::as_str),
            Some("jsonl")
        );
        assert_eq!(
            result.get("documents_exported").and_then(JsonValue::as_f64),
            Some(0.0)
        );
        assert_eq!(
            result.get("manifest").and_then(JsonValue::as_str),
            Some("export_manifest.json")
        );
        assert!(out.join("export_manifest.json").exists());
        let err = run(vec![
            "coffer".into(),
            "export".into(),
            path.display().to_string(),
            out.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--format".into(),
            "csv".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert!(err.message().contains("--format must be jsonl"));
        assert_eq!(detail_value(&err, "field"), Some("--format"));
        assert_eq!(detail_value(&err, "actual"), Some("csv"));
        assert_eq!(detail_value(&err, "expected"), Some("jsonl"));
        let _ = std::fs::remove_dir_all(out);
        let _ = std::fs::remove_file(path);
    }
}
