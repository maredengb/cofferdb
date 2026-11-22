//! Run the native benchmark against a fixed dataset, compare its ranked results
//! with independently prepared expectations and assemble the report. The execution
//! journal can record a failed run even when there is no result document to
//! publish.
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use super::{
    validate_benchmark_result_contract, BenchmarkDataset, BenchmarkQuery, BenchmarkQueryMode,
    QueryEmbeddingSource,
};
use crate::catalog::Catalog;
use crate::db::{Database, DatabaseOpenOptions};
use crate::error::{Error, ErrorCode, Result};
use crate::hash::Sha256Reader;
use crate::import::{visit_chunks_jsonl_in, visit_documents_jsonl_in, visit_embeddings_jsonl_in};
use crate::index::hybrid::fuse;
use crate::index::text::{Bm25Hits, TextCandidates};
use crate::index::vector::{normalize_score, VectorCandidates, VectorHits};
use crate::json::{
    btree_workspace, copy_precharged_string, CatalogReservation, JsonBuilder, JsonInputKind,
    JsonOperationContext, JsonOutputSchema, JsonReservation, JsonValue, ReservedString,
};
use crate::metadata::FilterExpr;
use crate::model::{
    token_count_hint_in, Chunk, DistanceMetric, Document, OnConflict, SearchWeights,
};
use crate::profile::{DatabaseConfig, Durability, ResourceProfile};
use crate::query::{CandidateLimits, SearchResults};
use crate::storage::FORMAT_VERSION;
use crate::temporary::{create_temporary_file_in, TempFileGuard};

mod acceptance;
mod execution;
mod memory;
mod output;
#[cfg(test)]
mod source_probe;
mod sqlite;
#[cfg(test)]
use source_probe::Source as BaselineSource;
// Use an ordinary file for baseline reads in production.
// read_verified_baseline_input wraps that same stream in a digesting reader before
// passing it to the decoder.
#[cfg(not(test))]
type BaselineSource = File;
use memory::process_memory_peak_mb;

pub(crate) struct BenchmarkOptions {
    // Database path opened or created for the measured import and search runs.
    pub(crate) path: String,
    // Path of the manifest that selects and identifies the fixed source dataset.
    pub(crate) dataset_path: String,
    // Target collection name shared by the import and the benchmark queries.
    pub(crate) collection: String,
    // Resource profile applied to the database and used to select reference
    // candidate limits.
    pub(crate) profile: ResourceProfile,
    // Durability requested when opening or creating the measured database.
    pub(crate) durability: Durability,
    // Optional destination for the final result, distinct from the execution
    // journal.
    pub(crate) result_file: Option<String>,
    // Require a completed SQLite comparison for the overall benchmark to succeed.
    pub(crate) require_baseline: bool,
}

pub(crate) struct BenchmarkRun {
    // Structured benchmark result passed to publication or rendered by the CLI.
    pub(crate) value: JsonValue,
    // Operation context used for budgeted output and retained report data.
    pub(crate) context: JsonOperationContext,
    // Reservation accompanying the result's retained JSON allocations.
    pub(crate) work: JsonReservation,
}

// Pair a prepared reference with its mode index for one cold or warm pass. The
// shared expectation grid is immutable; the journal borrow is exclusive because
// each event advances the run's sequence.
struct QueryVerification<'a, 'run> {
    // Ranked source expectations prepared before timing the measured queries.
    expected: &'a acceptance::Acceptance,
    // Index into the mode-major expectations array for the current run.
    mode: usize,
    // When present, record this query's search and acceptance events in the
    // execution journal.
    journal: Option<&'a mut execution::ExecutionJournal<'run>>,
}

// Start the elapsed clock before loading the manifest. Failures during setup get
// the same elapsed-time origin as errors raised after a query, even when setup did
// not get far enough to open a journal.
pub(crate) fn run_benchmark(options: BenchmarkOptions) -> Result<BenchmarkRun> {
    // One continuous clock includes input verification, import, reference preparation,
    // every planned query, and verification. It is never reconstructed from phase sums.
    let import_started = Instant::now();
    // Attach elapsed time to the original error. This includes optional output
    // publication failures as well as failures before the import started, without
    // replacing their more specific error codes.
    run_benchmark_started(options, import_started).map_err(|error| {
        error.with_detail(
            "benchmark_elapsed_ms",
            import_started.elapsed().as_secs_f64() * 1000.0,
        )
    })
}

fn run_benchmark_started(
    options: BenchmarkOptions,
    import_started: Instant,
) -> Result<BenchmarkRun> {
    let BenchmarkOptions {
        path,
        dataset_path,
        collection,
        profile,
        durability,
        result_file,
        require_baseline,
    } = options;
    // Keep the loaded dataset and its reservation together for the whole run. Its
    // query rows and source identities are borrowed by reference preparation,
    // measured searches and the SQLite comparison.
    let dataset_context = JsonOperationContext::standalone();
    let (dataset, _dataset_work, input_paths) =
        BenchmarkDataset::load_with_sources_in(&dataset_path, &dataset_context)?;
    // Plan the modes before journal creation and reuse this exact order for
    // building expectations and executing observations. Recomputing it midway
    // would change the meaning of a recorded mode index.
    let modes = benchmark_query_modes(&dataset);
    // Resolve the optional output before opening the measured database. The output
    // object gets the protected input paths for its alias checks; an in-memory
    // result bypasses file publication.
    let output = result_file
        .as_ref()
        .map(|result_file| {
            // Protect the database, lock and resolved dataset inputs when a result
            // filename was requested. Even a workload that later fails before
            // publication must reject an already conflicting destination.
            output::BenchmarkOutput::new(Path::new(result_file), Path::new(&path), input_paths)
        })
        .transpose()?;
    // Allocate the journal after the mode plan is known so it can address each
    // expected query by mode, cache state and query index. The elapsed origin is
    // still the start of dataset loading.
    let mut journal = execution::ExecutionJournal::new(
        &dataset,
        &modes,
        execution::ExecutionConfig {
            database: &path,
            result_file: result_file.as_deref(),
            collection: &collection,
            profile,
            durability,
        },
        &dataset_context,
        import_started,
    )?;
    // Route all work after journal creation through one result. An error from
    // import, ranking verification or report construction can then be finalized
    // with the journal state reached at that point.
    let result = (|| {
        // An existing path is opened with explicit run settings rather than
        // replaced by a newly created database. Reopening later must apply the
        // same profile and durability for the measured searches.
        let db = if PathBuf::from(&path).exists() {
            Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    resource_profile: Some(profile),
                    durability: Some(durability),
                    ..DatabaseOpenOptions::default()
                },
            )?
        } else {
            Database::create(
                &path,
                DatabaseConfig {
                    durability,
                    resource_profile: profile,
                    ..DatabaseConfig::default()
                },
            )?
        };
        // Generate a manifest for the measured importer from the verified dataset.
        // The temporary file supplies corpus paths and hashes; queries and
        // relevance judgments are inputs to the benchmark itself.
        let import_manifest_file =
            write_benchmark_import_manifest(&dataset, &collection, &dataset_context)?;
        // Use a stable label in the report because the generated file is removed
        // after import. Recording its random path would leave a non-reusable
        // reference in the result.
        let import_manifest_label = "<generated-import-manifest>".to_string();
        // Measure the actual importer with replacement semantics and inline
        // checkpoints. The reported batch counts and summary come from this call,
        // including work on an existing target collection.
        let measured_import = db
            .import_manifest_measured_in(
                import_manifest_file.path(),
                OnConflict::Replace,
                crate::db::ImportIndexBuildMode::InlineFullCheckpoint,
                dataset_context.clone(),
            )
            // Translate errors raised against generated input paths to the
            // locations the benchmark dataset can expose. This mapping runs while
            // the temporary import manifest is still present.
            .map_err(|error| dataset.locate_input_error(error))?;
        // The importer has finished reading the temporary manifest. Release its
        // guard before journal and reference work so a later benchmark error does
        // not extend the lifetime of a staging file.
        drop(import_manifest_file);
        // Record import completion before preparing expected rankings. If a later
        // comparison fails, the journal can still identify the batches that had
        // already completed.
        journal.import_completed(&measured_import.batches)?;
        let import_summary = measured_import.summary;
        // Generate reference results from the fixed dataset before measuring
        // queries. The imported handle resolves identities; close it before the
        // first cold run so reference preparation cannot warm that run's handle
        // caches.
        let acceptance = acceptance::Acceptance::prepare(
            &db,
            &collection,
            &dataset,
            profile,
            &modes,
            &dataset_context,
        )
        .map_err(|error| dataset.locate_input_error(error))?;
        // Write the expected result identities before any timed search starts. The
        // mode indices used in that record must stay in the same order as the
        // measured loop.
        journal.expectations(&acceptance)?;
        // Release the import handle before opening a new measured handle. The
        // expectation table survives independently and is shared by both the cold
        // and warm passes.
        drop(db);

        // Keep a single primary warm run for the top-level summary. Comparisons
        // across modes and cache states still use the individual runs, including
        // their own per-query metrics.
        let primary_mode = modes
            .iter()
            .copied()
            .find(|mode| *mode == BenchmarkQueryMode::Hybrid)
            .unwrap_or_else(|| modes[0]);
        let mut runs = Vec::new();
        // This reservation accumulates the JSON that outlives each run. Per-query
        // scratch and temporary engine results have shorter lifetimes and should
        // not be charged for the life of the full report.
        let mut output_work = dataset_context.reservation();
        // Only the selected primary warm metrics survive in this slot; structured
        // JSON for every cache/mode pair is retained separately. An aggregate
        // taken across all pairs would describe a different measurement.
        let mut primary_metrics = None;
        let mut index_size_bytes = 0;
        // Each mode starts with a freshly opened database. Its warm run reuses
        // the handle populated by its cold run. The OS page cache can survive
        // both reopening and a change of mode, so cold describes the handle.
        for (mode_index, mode) in modes.iter().copied().enumerate() {
            // Open the same persisted database once per mode, after expectation
            // preparation and outside query latency timing. The handle then lives
            // across both passes of that mode.
            let bench_db = Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    resource_profile: Some(profile),
                    durability: Some(durability),
                    ..DatabaseOpenOptions::default()
                },
            )?;
            // Use the first nonzero measured index size for the report. Zero
            // leaves this check eligible on the next reopened handle rather than
            // treating an empty index as a cached measurement.
            if index_size_bytes == 0 {
                index_size_bytes = bench_db.index_stats()?.index_size_bytes;
            }
            // Run the cold observations first and send their search and
            // verification events to the journal. Only the structured cold values
            // are retained for the report's per-mode rows.
            let (cold_run, _cold_metrics, cold_work) = run_benchmark_queries(
                &bench_db,
                &collection,
                &dataset,
                profile,
                mode,
                "cold",
                &dataset_context,
                Some(QueryVerification {
                    expected: &acceptance,
                    mode: mode_index,
                    journal: Some(&mut journal),
                }),
            )?;
            // Transfer the cold row's reservation as soon as the row is retained.
            // The warm search may allocate substantially more data while that cold
            // result is still live.
            output_work.merge(cold_work);
            // Feed the same verification mode index to both cache states. Cold and
            // warm runs have distinct journal positions but are expected to return
            // the same ranking for a given query.
            let (warm_run, warm_metrics, warm_work) = run_benchmark_queries(
                &bench_db,
                &collection,
                &dataset,
                profile,
                mode,
                "warm",
                &dataset_context,
                Some(QueryVerification {
                    expected: &acceptance,
                    mode: mode_index,
                    journal: Some(&mut journal),
                }),
            )?;
            output_work.merge(warm_work);
            // Reserve space for the pair before appending either row. The public
            // array preserves cold then warm for each mode even though the summary
            // chooses just one warm row.
            output_work.reserve_vec(&mut runs, 2, JsonInputKind::BenchmarkArtifact)?;
            runs.push(cold_run);
            runs.push(warm_run);
            if mode == primary_mode {
                primary_metrics = Some(warm_metrics);
            }
        }
        // Despite the field name, this interval ends after both native searches
        // for every mode. It includes loading, import, reference preparation and
        // measured queries, and stops before the SQLite comparison.
        let import_time_ms = import_started.elapsed().as_secs_f64() * 1000.0;
        // The expected native rankings can now release their workspace. Baseline
        // execution uses the dataset independently and contributes its own
        // retained output to the report.
        drop(acceptance);
        let warm_metrics = primary_metrics.unwrap_or_default();
        let (baseline_outcome, baseline_work) =
            benchmark_baseline_outcome(&dataset, profile, &dataset_context);
        // Merge the baseline's retained charges before incorporating its rows into
        // the native report. A skipped baseline can carry a reason even though it
        // contributes no measured row.
        output_work.merge(baseline_work);
        // A skipped baseline normally has a reportable reason. When the caller
        // requires it, convert that outcome to a run error before building or
        // publishing a completed result.
        if require_baseline && baseline_outcome.status != "completed" {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark baseline is required but did not complete",
            )
            .with_detail("field", "baseline_status")
            .with_detail("baseline_status", &baseline_outcome.status)
            .with_detail(
                "baseline_status_reason",
                baseline_outcome.reason.as_deref().unwrap_or(""),
            ));
        }
        // From here the builder spends from the output reservation that already
        // owns the native and baseline JSON. Assembly can still fail under the
        // output budget after searches have completed.
        let builder = JsonBuilder::budgeted(&mut output_work, JsonInputKind::BenchmarkArtifact);
        let baseline_warnings = benchmark_baseline_warnings_in(&baseline_outcome, &builder)?;
        // Report the inputs actually used by the importer, together with the
        // measured batch summary. The command is a description with placeholders
        // because the generated manifest no longer exists.
        let import_context = builder.object([
        ("tokenizer", import_summary.tokenizer.build_json(&builder)?),
        (
            "batch_size",
            // Largest actual committed chunk batch, not the profile cap or manifest total.
            JsonValue::Number(measured_import.batches.max_batch_chunks as f64),
        ),
        (
            "index_build_mode",
            builder.string("inline_full_checkpoint")?,
        ),
        ("import_time_ms", JsonValue::Number(import_time_ms)),
        (
            "import_command",
            builder.owned_string(format!(
                "coffer import <benchmark-db> --manifest <generated-import-manifest> --on-conflict replace --durability {}",
                durability.as_str()
            ))?,
        ),
        ("manifest_path", builder.owned_string(import_manifest_label)?),
        ("database_path", builder.string(&path)?),
        ("on_conflict", builder.string("replace")?),
        ("summary", import_summary.build_json(&builder)?),
        ("durability", builder.string(durability.as_str())?),
    ])?;
        // Start from the selected warm run's metrics. Import elapsed time and the
        // on-disk index measurement are properties of the benchmark as a whole and
        // are added only to this summary.
        let mut metrics = benchmark_metrics_json_in(&warm_metrics, &builder)?;
        builder.insert_if_absent(metrics.as_object_mut().unwrap(), "import_time_ms", || {
            Ok(JsonValue::Number(import_time_ms))
        })?;
        // The index API reports bytes, while the public numeric field is measured
        // in 1,048,576-byte units. Keep the fractional part for small indexes.
        builder.insert_if_absent(metrics.as_object_mut().unwrap(), "index_size_mb", || {
            Ok(JsonValue::Number(index_size_bytes as f64 / 1_048_576.0))
        })?;
        // Build the report only after all required native and baseline work has
        // reached its stated outcome. The structured value is validated as a whole
        // before the publication path can use it.
        let result = builder.object([
            ("schema_version", builder.string("0.1")?),
            (
                "cofferdb_version",
                builder.string(env!("CARGO_PKG_VERSION"))?,
            ),
            ("git_commit", benchmark_git_commit_json_in(&builder)?),
            // Record the actual storage format constant alongside the software
            // version. Package and storage schema changes are independently
            // identifiable in a saved measurement.
            (
                "file_format_version",
                JsonValue::Number(FORMAT_VERSION as f64),
            ),
            ("hardware", benchmark_hardware_json_in(&builder)?),
            // This status attests completion of the native runs. Baseline
            // completion has a separate status and can legitimately be reported as
            // skipped here.
            ("status", builder.string("completed_cofferdb_runs")?),
            ("path", builder.owned_string(path)?),
            ("result_file", builder.optional_str(result_file.as_deref())?),
            ("dataset_path", builder.owned_string(dataset_path)?),
            ("dataset_id", builder.string(&dataset.manifest.dataset_id)?),
            ("collection", builder.owned_string(collection)?),
            (
                "dataset",
                dataset.manifest.build_result_dataset_json(&builder)?,
            ),
            ("profile", builder.string(profile.as_json_str())?),
            ("profile_params", profile_params_json_in(profile, &builder)?),
            ("engine_params", benchmark_engine_params_json_in(&builder)?),
            // Publish the profile's configured candidate limits. Separate query
            // contexts record the limits actually used, which can grow when a
            // particular query requests a larger top_k.
            (
                "candidate_counts",
                builder.object([
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
                ])?,
            ),
            ("durability", builder.string(durability.as_str())?),
            (
                "index_build_mode",
                builder.string("inline_full_checkpoint")?,
            ),
            // Place the import context beside ranking metrics so consumers can
            // attribute import costs and retrieval costs separately. Per-run rows
            // beneath runs carry their own search metrics.
            ("import_context", import_context),
            ("metrics", metrics),
            ("warnings", baseline_warnings),
            ("runs", JsonValue::Array(runs)),
            // Describe the comparison configuration even when it did not run. Its
            // verification status follows the actual outcome rather than the
            // presence of a configuration object.
            (
                "baseline_configs",
                sqlite_baseline_configs_in(
                    &dataset,
                    baseline_outcome.status == "completed",
                    &builder,
                )?,
            ),
            (
                "baseline_results",
                JsonValue::Array(baseline_outcome.results),
            ),
            (
                "baseline_status",
                builder.owned_string(baseline_outcome.status)?,
            ),
            (
                "baseline_status_reason",
                builder.optional_string(baseline_outcome.reason)?,
            ),
        ])?;
        // Check cross-record constraints on the assembled value, including mode,
        // cache pairing and baseline linkage. A completed search alone cannot
        // certify that these independent producers emitted a consistent report.
        validate_benchmark_result_contract(&result)?;
        let schema = JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact);
        // Reconcile the surviving JSON against its output schema after transient
        // values have been consumed. Return both the context and reservation so
        // publication can continue within the same accounting.
        result.reconcile_output_retained_in(schema, &mut output_work)?;
        Ok(BenchmarkRun {
            value: result,
            // Move the original dataset context out with the report data; an
            // output writer can serialize this value against the same accounting
            // without still borrowing the temporary dataset rows.
            context: dataset_context,
            work: output_work,
        })
    })();
    // The closure ends before publication. Construction failures follow the
    // journal failure path, and a successful value still passes through
    // publish_run, where optional output I/O can fail.
    match result {
        Ok(run) => journal.publish_run(run, output),
        Err(error) => Err(journal.fail(error)),
    }
}

#[derive(Clone, Debug, Default)]
struct BenchmarkRunMetrics {
    // Mean fraction of each query's relevant chunks found in its first five
    // results.
    recall_at_5: f64,
    // Mean fraction of each query's relevant chunks found in its first ten results.
    recall_at_10: f64,
    // Mean reciprocal rank of the first relevant result across this group of
    // queries.
    mrr: f64,
    // Nearest-rank 95th percentile of measured query latencies, in milliseconds.
    p95_ms: f64,
    // Process high-water RSS in MiB, when available for this run.
    memory_peak_mb: Option<f64>,
    // Sum of the per-query vector scan counts reported for this group.
    vector_scan_count: usize,
    // Sum of coordinate-level distance operations reported for the group.
    distance_ops: usize,
    // Sum of reported fusion candidate counts across this group's queries.
    fusion_candidate_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BenchmarkExecutionStats {
    // Number of vector rows reported as scanned for one query.
    vector_scan_count: usize,
    // Distance work counted in vector components for this query.
    distance_ops: usize,
    // Reported size of this query's candidate set at the fusion stage.
    fusion_candidate_count: usize,
}

impl BenchmarkExecutionStats {
    #[cfg(test)]
    fn to_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    // Keep the same three counters under per-query execution_stats as the
    // aggregation path reads. Converting them to JSON numbers here preserves their
    // native counter units.
    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            (
                "vector_scan_count",
                JsonValue::Number(self.vector_scan_count as f64),
            ),
            ("distance_ops", JsonValue::Number(self.distance_ops as f64)),
            (
                "fusion_candidate_count",
                JsonValue::Number(self.fusion_candidate_count as f64),
            ),
        ])
    }
}

// Choose modes that can run against every query. This keeps the cold/warm and
// cross-mode comparisons over a common query set rather than silently dropping
// input rows for just one mode.
fn benchmark_query_modes(dataset: &BenchmarkDataset) -> Vec<BenchmarkQueryMode> {
    // Compute text capability across the entire dataset, including whether a
    // supplied string is nonblank. A single unsuitable row precludes selecting an
    // ordinary all-query text pass through this branch.
    let all_have_text = dataset
        .queries
        .iter()
        .all(|query| !query.text.as_deref().unwrap_or("").trim().is_empty());
    let all_have_embedding = dataset
        .queries
        .iter()
        .all(|query| query.embedding.is_some());
    let mut modes = Vec::new();
    if all_have_text {
        modes.push(BenchmarkQueryMode::TextBm25);
    }
    if all_have_embedding {
        modes.push(BenchmarkQueryMode::VectorExact);
    }
    // Hybrid is available only when all rows supply both kinds of input. Push it
    // after the individual modes so acceptance and query execution use the same
    // stable indices.
    if all_have_text && all_have_embedding {
        modes.push(BenchmarkQueryMode::Hybrid);
    }
    // Keep the plan nonempty for callers that choose modes[0] as a fallback
    // summary. Selecting text here does not fabricate missing text; the measured
    // query path still validates each input.
    if modes.is_empty() {
        modes.push(BenchmarkQueryMode::TextBm25);
    }
    modes
}

// Translate only the corpus inputs into an import manifest. The source benchmark
// manifest also describes query and judgment files which must stay out of the
// measured database import.
fn write_benchmark_import_manifest(
    dataset: &BenchmarkDataset,
    collection: &str,
    context: &JsonOperationContext,
) -> Result<TempFileGuard> {
    let mut work = context.reservation();
    // The JSON being staged is an import manifest, with its own schema and size
    // budget. Constructing the final benchmark report later uses a separate
    // reservation and input kind.
    let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::Manifest);
    let mut manifest = builder.object([
        (
            "tokenizer",
            dataset.manifest.tokenizer.build_json(&builder)?,
        ),
        ("schema_version", builder.string("0.1")?),
        ("dataset_id", builder.string(&dataset.manifest.dataset_id)?),
        ("collection", builder.string(collection)?),
        // Use canonical paths because the generated manifest is placed in a
        // temporary directory with a different base from the dataset. Relative
        // corpus paths would otherwise resolve against the wrong parent.
        (
            "documents",
            builder.string(&canonical_path_string(&dataset.documents_path)?)?,
        ),
        (
            "chunks",
            builder.string(&canonical_path_string(&dataset.chunks_path)?)?,
        ),
        (
            "embedding_model",
            builder.optional_str(dataset.manifest.embedding_model.as_deref())?,
        ),
        (
            "embedding_dim",
            JsonValue::Number(dataset.manifest.embedding_dim as f64),
        ),
        // The measured collection uses cosine distance. Carry over the corpus
        // model and width alongside that setting so the importer interprets the
        // input vectors consistently.
        ("distance_metric", builder.string("cosine")?),
        // Carry the source hashes through to the importer. Reopening a named
        // corpus file for import must not silently change which contents the
        // benchmark dataset identified.
        (
            "hashes",
            builder.object([
                (
                    "documents_sha256",
                    builder.string(&dataset.manifest.documents_sha256)?,
                ),
                (
                    "chunks_sha256",
                    builder.string(&dataset.manifest.chunks_sha256)?,
                ),
            ])?,
        ),
    ])?;
    // Omit the embedding path and digest together for text-only corpora.
    // Query-side vectors, including a sidecar, are never corpus input to this
    // importer.
    if let Some(embeddings_path) = &dataset.embeddings_path {
        builder.insert_if_absent(manifest.as_object_mut().unwrap(), "embeddings", || {
            builder.string(&canonical_path_string(embeddings_path)?)
        })?;
        if let Some(obj) = manifest.as_object_mut() {
            let hashes = obj
                .get_mut("hashes")
                .and_then(JsonValue::as_object_mut)
                .expect("benchmark import manifest hashes is an object");
            // Nest the embeddings hash alongside the document and chunk hashes
            // when supplied. The optional path and digest are not top-level
            // siblings on the generated import schema.
            if let Some(hash) = &dataset.manifest.embeddings_sha256 {
                builder.insert_if_absent(hashes, "embeddings_sha256", || builder.string(hash))?;
            }
        }
    }
    // The buffered writer consumes workspace in addition to the retained manifest
    // tree. Keep both reservations live while serialization can still traverse the
    // JSON.
    let mut output_work = context.reservation();
    // Reserve the chosen buffer capacity before creating the file. That charge is
    // still needed while the separately charged manifest value is serialized into
    // the writer.
    output_work.grow(8192, JsonInputKind::Manifest)?;
    let (file, guard) = create_temporary_file("benchmark-import", "json")?;
    let mut file = std::io::BufWriter::with_capacity(8192, file);
    manifest.write_value_retained_in(
        &mut file,
        JsonInputKind::Manifest,
        context,
        &mut work,
        true,
    )?;
    // Finish and flush the buffered file before returning its guard. A flush
    // failure is an import setup error, and leaving through that path drops the
    // temporary-file guard.
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(guard)
}

fn canonical_path_string(path: &Path) -> Result<String> {
    Ok(fs::canonicalize(path)?.display().to_string())
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark_queries(
    db: &Database,
    collection: &str,
    dataset: &BenchmarkDataset,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    cache_state: &str,
    context: &JsonOperationContext,
    mut acceptance: Option<QueryVerification<'_, '_>>,
) -> Result<(JsonValue, BenchmarkRunMetrics, JsonReservation)> {
    // Resolve the collection once, outside query timing. All searches in this
    // invocation use the same open database and execute in dataset order for the
    // selected mode and cache state.
    let collection = db.collection(collection)?;
    let mut work = context.reservation();
    // The latency vector is temporary percentile workspace; the per-query JSON is
    // returned. Reserving them separately lets the scratch charge end without
    // releasing retained report data.
    let mut scratch = context.reservation();
    let mut per_query = Vec::new();
    let mut latencies = Vec::new();
    let mut recall_at_5_sum = 0.0;
    let mut recall_at_10_sum = 0.0;
    let mut reciprocal_rank_sum = 0.0;
    let mut vector_scan_count = 0;
    let mut distance_ops = 0;
    let mut fusion_candidate_count = 0;

    for (query_index, query) in dataset.queries.iter().enumerate() {
        // Select only the inputs defined by the mode, even when a dataset row
        // offers both. Passing the unused input would turn a text-only or
        // vector-only measurement into a different search.
        let (text, embedding) = match mode {
            BenchmarkQueryMode::TextBm25 => {
                let text = query.text.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "text benchmark mode requires query text",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                (Some(text.as_str()), None)
            }
            BenchmarkQueryMode::VectorExact => {
                let embedding = query.embedding.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "vector benchmark mode requires query embedding",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                (None, Some(embedding))
            }
            BenchmarkQueryMode::Hybrid => {
                let text = query.text.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "hybrid benchmark mode requires query text",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                // A hybrid search needs the source embedding even after text
                // validation succeeded. Tag missing-input errors with the dataset
                // query ID before any timed engine call or journal event.
                let embedding = query.embedding.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "hybrid benchmark mode requires query embedding",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                (Some(text.as_str()), Some(embedding))
            }
        };
        // Record the beginning before creating the optional trace or calling the
        // engine. The journal can then distinguish a query that started from one
        // that never reached its turn.
        if let Some(verification) = acceptance.as_mut() {
            if let Some(journal) = &mut verification.journal {
                journal.begin(verification.mode, cache_state, query_index)?;
            }
        }
        // Only a journal consumer needs a search-cost trace. Reference
        // verification can still inspect the completed results when there is no
        // journal attached.
        let trace = acceptance
            .as_ref()
            .and_then(|verification| verification.journal.as_ref())
            .map(|_| crate::search_cost::SearchCostTrace::new());
        let started = Instant::now();
        // Time the engine call alone. The journal records when search finished;
        // ranking verification can still fail afterward. Keep verification and
        // report construction outside the measured interval.
        let search = db.search_in_collection_traced(
            collection.collection_id(),
            text,
            embedding,
            &query.filter,
            None,
            false,
            None,
            query.top_k,
            Some(profile),
            Some(context.clone()),
            trace.as_ref(),
        );
        // Read the query clock immediately on return, including for an error.
        // Trace finalization, journal writes and ranking verification must not
        // inflate the measured search latency.
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        if let Some(trace) = &trace {
            trace.finish();
        }
        if let Some(verification) = acceptance.as_mut() {
            if let Some(journal) = &mut verification.journal {
                if let Some(trace) = trace {
                    // Preserve the engine error when recording its completion also
                    // fails, attaching the journal failure as detail. If the
                    // search succeeded, the journal failure is the error that
                    // prevents completion.
                    if let Err(record_error) =
                        journal.search_finished(trace, latency_ms, search.is_ok())
                    {
                        return Err(match search {
                            Err(error) => error
                                .with_detail("execution_record_error", record_error.to_string()),
                            Ok(_) => record_error,
                        });
                    }
                }
            }
        }
        // Propagate search failure after recording its outcome. A successful
        // engine return still needs to match the expected rankings before the
        // journal may mark the query complete.
        let (mut results, mut result_work) = search?;
        if let Some(verification) = acceptance.as_ref() {
            verification
                .expected
                .verify(verification.mode, query_index, &results)
                .map_err(|error| {
                    error
                        .with_detail("query_id", &query.query_id)
                        .with_detail("query_mode", mode.as_str())
                        .with_detail("cache_state", cache_state)
                })?;
        }
        // Write the query-completed event only after ranking verification
        // succeeds. A search-completed event can therefore be followed by a
        // verification failure without recording an accepted query.
        if let Some(verification) = acceptance.as_mut() {
            if let Some(journal) = &mut verification.journal {
                journal.complete_query(latency_ms, &results)?;
            }
        }
        // Iterate external chunk IDs in engine rank order. Both the reference
        // judgments and the published per-query identifiers use that namespace,
        // while independent passes compute the different rank cutoffs.
        let ranked_chunk_ids = || {
            results
                .results
                .iter()
                .map(|result| result.chunk_external_id.as_str())
        };
        // The hit flag considers every returned result. Recall uses only the
        // requested evaluation prefixes, and reciprocal rank depends on the first
        // relevant result, so none can be copied from the flag.
        let hit = ranked_chunk_ids().any(|chunk_id| query.relevant_chunk_ids.contains(chunk_id));
        let recall_at_5 = recall_at_k(ranked_chunk_ids(), query, 5);
        let recall_at_10 = recall_at_k(ranked_chunk_ids(), query, 10);
        let reciprocal_rank = reciprocal_rank(ranked_chunk_ids(), query);
        let stats = benchmark_execution_stats(&results);

        // Accumulate latency samples for the percentile, sums for per-query means
        // and total work counters across the dataset. The counter totals should
        // not be divided by the query count below.
        scratch.reserve_vec(&mut latencies, 1, JsonInputKind::BenchmarkArtifact)?;
        latencies.push(latency_ms);
        recall_at_5_sum += recall_at_5;
        recall_at_10_sum += recall_at_10;
        reciprocal_rank_sum += reciprocal_rank;
        vector_scan_count += stats.vector_scan_count;
        distance_ops += stats.distance_ops;
        fusion_candidate_count += stats.fusion_candidate_count;

        // Reserve the retained report row before constructing it. Its scores and
        // identifiers must refer to the same ranked sequence even after the engine
        // result is drained.
        work.reserve_vec(&mut per_query, 1, JsonInputKind::BenchmarkArtifact)?;
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        // Construct scores while ranked engine rows and their ordering are still
        // intact. Later the IDs can move into the report instead of cloning all
        // their strings.
        let scores = builder.array(
            results
                .results
                .iter()
                .map(|result| Ok(JsonValue::Number(result.final_score))),
        )?;
        // Move ranked external IDs into the report while their original reservation
        // still owns them. The score array above must be built first, since draining
        // the rows also discards their scores and explanations.
        let ranked_chunk_ids = {
            let (_, strings) = result_work.parts_mut();
            let builder = JsonBuilder::budgeted_owned_strings(
                &mut work,
                strings,
                JsonInputKind::BenchmarkArtifact,
            );
            builder.array(
                // Take ownership of the entire result vector in rank order. Each
                // external identifier is moved exactly once into the array paired
                // with the scores built above.
                std::mem::take(&mut results.results)
                    .into_iter()
                    .map(|result| builder.owned_string(result.chunk_external_id)),
            )?
        };
        // After extracting the ranked IDs, discard the remaining row allocation
        // and keep only the query-context workspace until it has been serialized.
        // The result's string reservation is still available for ownership
        // transfers.
        results.discard_rows(&mut result_work);
        let query_context = results.query_context;
        drop((results.schema_version, results.collection, results.query));
        // Narrow the residual charge to the remaining query context. Its own
        // strings may still need to move to the output builder after the original
        // result's top-level strings have been dropped.
        result_work.retain_workspace(query_context.workspace_bytes());
        let (_, strings) = result_work.parts_mut();
        let builder = JsonBuilder::budgeted_owned_strings(
            &mut work,
            strings,
            JsonInputKind::BenchmarkArtifact,
        );
        // Preserve both score order and score semantics. Numeric scores belong to
        // this engine and mode; consumers can compare relevance metrics across
        // engines without treating the raw score scales as interchangeable.
        per_query.push(builder.object([
            ("query_id", builder.string(&query.query_id)?),
            ("latency_ms", JsonValue::Number(latency_ms)),
            ("ranked_chunk_ids", ranked_chunk_ids),
            ("score_kind", builder.string(mode.score_kind())?),
            ("score_scale", builder.string("engine_local")?),
            ("score_comparable", JsonValue::Bool(false)),
            ("scores", scores),
            ("hit", JsonValue::Bool(hit)),
            ("execution_stats", stats.build_json(&builder)?),
            ("query_context", query_context.build_json(&builder)?),
        ])?);
    }

    // Every scheduled run iterates the same validated query set. Use its length
    // for macro-averaged recall and reciprocal rank, while the percentile consumes
    // the latency samples in place.
    let query_count = dataset.queries.len() as f64;
    let metrics = BenchmarkRunMetrics {
        recall_at_5: recall_at_5_sum / query_count,
        recall_at_10: recall_at_10_sum / query_count,
        mrr: reciprocal_rank_sum / query_count,
        p95_ms: percentile_95_in_place(&mut latencies),
        // Take the platform's process peak after the group has run. It can include
        // work before this cache/mode pass and may be unavailable, unlike the
        // per-query latency samples.
        memory_peak_mb: process_memory_peak_mb(),
        vector_scan_count,
        distance_ops,
        fusion_candidate_count,
    };
    // Return the retained JSON with its reservation and a separate numeric
    // summary. The caller can select the primary warm metrics without walking or
    // reparsing the per-query report.
    let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
    let value = builder.object([
        ("engine", builder.string("cofferdb")?),
        ("query_mode", builder.string(mode.as_str())?),
        ("cache_state", builder.string(cache_state)?),
        ("metrics", benchmark_metrics_json_in(&metrics, &builder)?),
        ("per_query", JsonValue::Array(per_query)),
    ])?;
    // Keep the numeric aggregate independent of the JSON value's lifetime. The
    // parent retains one aggregate for the summary and transfers the structured
    // row and its charge for every pass.
    Ok((value, metrics, work))
}

#[cfg(test)]
fn benchmark_metrics_json(metrics: &BenchmarkRunMetrics) -> JsonValue {
    benchmark_metrics_json_in(metrics, &JsonBuilder::host())
        .expect("host DTO construction is infallible")
}

// Use the same metric field names for native and SQLite run records. Retain each
// run's observed values even when the two engines do not produce comparable raw
// scores.
fn benchmark_metrics_json_in(
    metrics: &BenchmarkRunMetrics,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    // Build the nested metrics under the caller's budget. The same helper serves
    // per-mode output independently of which run the enclosing report selects for
    // its summary.
    builder.object([
        ("recall_at_5", JsonValue::Number(metrics.recall_at_5)),
        ("recall_at_10", JsonValue::Number(metrics.recall_at_10)),
        ("mrr", JsonValue::Number(metrics.mrr)),
        ("p95_ms", JsonValue::Number(metrics.p95_ms)),
        // A peak memory reading can be unavailable independently of the ranking
        // and latency metrics. Serialize that state explicitly so consumers can
        // distinguish it from a measured zero.
        (
            "memory_peak_mb",
            metrics
                .memory_peak_mb
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
        ),
        // The execution counters are totals for the queries in this run. They
        // retain their separate meanings even when a particular mode leaves one or
        // more counters at zero.
        (
            "vector_scan_count",
            JsonValue::Number(metrics.vector_scan_count as f64),
        ),
        (
            "distance_ops",
            JsonValue::Number(metrics.distance_ops as f64),
        ),
        (
            "fusion_candidate_count",
            JsonValue::Number(metrics.fusion_candidate_count as f64),
        ),
    ])
}

// Recall measures how much of the full positive judgment set appears among the
// first k returned ranks. The denominator stays fixed even if filtering leaves
// fewer than k hits.
fn recall_at_k<'a>(
    ranked_chunk_ids: impl IntoIterator<Item = &'a str>,
    query: &BenchmarkQuery,
    k: usize,
) -> f64 {
    // A query with judgments but no positive labels has no nonzero recall
    // denominator. Assign it a zero contribution to the run's average.
    if query.relevant_chunk_ids.is_empty() {
        return 0.0;
    }
    // Inspect at most k ranked entries in retrieval order. Membership uses the
    // benchmark's relevant external chunk IDs, which are shared by native and
    // SQLite runs.
    let hits = ranked_chunk_ids
        .into_iter()
        .take(k)
        .filter(|chunk_id| query.relevant_chunk_ids.contains(*chunk_id))
        .count();
    // Divide by all relevant chunks in the judgments, not by min(k,
    // relevant_count). A short cutoff can therefore have recall below one even
    // when every returned rank is relevant.
    hits as f64 / query.relevant_chunk_ids.len() as f64
}

// Only the first relevant rank contributes to reciprocal rank. Additional matching
// chunks affect recall but do not improve this metric once a hit has been found.
fn reciprocal_rank<'a>(
    ranked_chunk_ids: impl IntoIterator<Item = &'a str>,
    query: &BenchmarkQuery,
) -> f64 {
    ranked_chunk_ids
        .into_iter()
        .position(|chunk_id| query.relevant_chunk_ids.contains(chunk_id))
        // Convert the zero-based position to a one-based rank before division. An
        // empty ranking and a ranking with no relevant chunk both contribute zero.
        .map(|index| 1.0 / (index + 1) as f64)
        .unwrap_or(0.0)
}

// Use the nearest-rank percentile: select the first observed latency at or above
// the 95th percentile's rank. The operation may reorder this disposable sample
// buffer.
fn percentile_95_in_place(values: &mut [f64]) -> f64 {
    // Return the report's zero latency default for an empty sample set before
    // computing a final valid index.
    if values.is_empty() {
        return 0.0;
    }
    // Sort the retained observations themselves. The chosen percentile is an
    // actual sample, without interpolation between neighboring measurements.
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
    values[index.min(values.len() - 1)]
}

// Read the counters from the query context produced by search. Estimating work
// from the number of returned hits would miss scanned vectors or rejected
// candidates.
fn benchmark_execution_stats(results: &SearchResults) -> BenchmarkExecutionStats {
    let stats = results.query_context.execution_stats;
    BenchmarkExecutionStats {
        vector_scan_count: stats.vector_scan_count,
        distance_ops: stats.distance_ops,
        fusion_candidate_count: stats.fusion_candidate_count,
    }
}

#[cfg(test)]
fn benchmark_hardware_json() -> JsonValue {
    benchmark_hardware_json_in(&JsonBuilder::host()).expect("host DTO construction is infallible")
}

// Build a portable hardware description with one combined field each for CPU and
// OS. The contract expects names and counts or versions together rather than split
// aliases.
fn benchmark_hardware_json_in(builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    let cpu_model = hardware_cpu_model();
    // The core count uses available parallelism, which can be limited by the
    // execution environment. It is the concurrency visible to this benchmark
    // process.
    let cpu_cores = hardware_cpu_cores();
    let (os_name, os_version) = hardware_os_name_version();
    // Group model name and available core count into the single public cpu string.
    // Readers of the hardware schema do not expect separate cpu_model or cpu_cores
    // members.
    builder.object([
        (
            "cpu",
            builder.owned_string(format!("{cpu_model}; {cpu_cores} cores"))?,
        ),
        (
            "memory_gb",
            // An unavailable capacity probe stays null. Reporting zero would
            // describe a measurement and would violate the positive-capacity rule
            // for a known value.
            hardware_memory_gb()
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
        ),
        // This is the runner's storage category. It does not probe a disk model,
        // filesystem type or measured throughput on the host.
        ("storage", builder.string("local-storage")?),
        (
            "os",
            builder.owned_string(format!("{os_name} {os_version}"))?,
        ),
    ])
}

fn benchmark_git_commit_json_in(builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    // The caller's working directory and environment do not identify this binary's source.
    let commit = option_env!("COFFERDB_GIT_COMMIT")
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("unknown");
    builder.string(commit)
}

// Describe the configured SQLite comparison even when it could not run. The
// completion flag controls verification metadata, while the dataset controls what
// capabilities the baseline requires.
fn sqlite_baseline_configs_in(
    dataset: &BenchmarkDataset,
    baseline_completed: bool,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    // Inline and sidecar vectors enable the same baseline capabilities. Their
    // difference concerns the source of query coordinates rather than the SQLite
    // retrieval algorithms.
    let requires_query_embedding =
        dataset.manifest.query_embedding_source != QueryEmbeddingSource::None;
    let name = sqlite_baseline_name(dataset);
    // The filter label describes the vector path when present. SQLite text
    // candidates are fetched under their BM25 limit and the metadata predicate is
    // applied afterward.
    let filter_strategy = if requires_query_embedding {
        "metadata_filter_before_vector_scoring"
    } else {
        "metadata_filter_after_candidate_fetch"
    };
    // The baseline configuration describes a set of queries with potentially
    // different top_k values. Report the largest requested value here; the ranked
    // arrays and query contexts retain each individual request.
    let max_top_k = dataset
        .queries
        .iter()
        .map(|query| query.top_k)
        // Validated input has at least one query. The zero fallback is not a
        // requested top_k for a valid workload; the actual requests determine this
        // maximum.
        .max()
        .unwrap_or(0);
    // Emit one named SQLite configuration for this dataset. Its measured results
    // are separate records indexed by that name for each mode and cache state.
    builder.array(std::iter::once_with(|| {
        builder.object([
            ("name", builder.string(name)?),
            // Query the SQLite CLI for its reported version. A missing probe
            // result is represented as null, allowing the configuration of an
            // unexecuted baseline to be described.
            ("version", builder.optional_string(sqlite3_version())?),
            ("run_mode", builder.string("local_cli")?),
            ("mode", builder.string("local_cli")?),
            (
                "requires_query_embedding",
                JsonValue::Bool(requires_query_embedding),
            ),
            ("top_k", JsonValue::Number(max_top_k as f64)),
            // Both cache states belong to the intended workload. Their declaration
            // in this configuration is independent of the actual measurement rows
            // produced on success.
            (
                "cache_states",
                builder.array(
                    ["cold", "warm"]
                        .into_iter()
                        .map(|value| builder.string(value)),
                )?,
            ),
            // Identify the documented recipe independently of the installed SQLite
            // version. Changing the executable can change version metadata without
            // changing which baseline is named.
            (
                "config_source",
                builder.string("benchmarks/baselines/sqlite/README.md")?,
            ),
            (
                // Assign the UTC calendar date only to a completed baseline. A
                // skipped configuration should not claim the date as verification
                // evidence.
                "checked_at",
                if baseline_completed {
                    builder.owned_string(benchmark_checked_at_date())?
                } else {
                    JsonValue::Null
                },
            ),
            (
                "config",
                builder.object([
                    ("engine", builder.string("sqlite")?),
                    (
                        "sqlite_cli",
                        builder.string(sqlite_cli_label(&sqlite_cli())?)?,
                    ),
                    // Expose the FTS table, tokenizer and SQL scoring expression
                    // used to fetch text candidates. These describe SQLite's
                    // retrieval implementation, not CofferDB's tokenizer identity.
                    ("fts", builder.string("fts5")?),
                    ("fts_table", builder.string("chunks_fts")?),
                    ("fts_tokenizer", builder.string("default")?),
                    ("fts_score", builder.string("bm25(chunks_fts)")?),
                    // Each SQLite CLI invocation reopens the database. Cold leaves
                    // OS caches intact, and warm repeats the workload, so any OS
                    // cache state can carry across the passes.
                    (
                        "cache_policy",
                        builder.string(
                            "cold=reopened_handle_no_os_cache_eviction;warm=repeated_query",
                        )?,
                    ),
                    // SQLite runs through the CLI whereas CofferDB search is
                    // in-process. The measured latencies include different
                    // process-startup costs, so keep that scope with the
                    // configuration.
                    (
                        "latency_scope",
                        builder.string(
                            "sqlite_cli_process_startup_included;cofferdb_process_startup_excluded",
                        )?,
                    ),
                    (
                        // A text-only workload does not require an installed
                        // vector extension. Record explicit null for the extension
                        // and mode instead of attributing vector capabilities to
                        // that run.
                        "vector_extension",
                        if requires_query_embedding {
                            builder.string("sqlite-vec")?
                        } else {
                            JsonValue::Null
                        },
                    ),
                    (
                        "vector_mode",
                        if requires_query_embedding {
                            builder.string("exact_scan")?
                        } else {
                            JsonValue::Null
                        },
                    ),
                    (
                        // Probe the extension's version in an in-memory database
                        // only for vector workloads. A failed version probe leaves
                        // the reported version unknown.
                        "vector_extension_version",
                        if requires_query_embedding {
                            builder.optional_string(
                                sqlite3_script(Path::new(":memory:"), "SELECT vec_version();")
                                    .ok()
                                    .map(|value| value.trim().to_owned()),
                            )?
                        } else {
                            JsonValue::Null
                        },
                    ),
                    (
                        // Distinguish a vector-capable configuration's weighted
                        // fusion from text-only BM25. The configuration can still
                        // produce a separate text-mode run when vectors are
                        // present.
                        "fusion",
                        if requires_query_embedding {
                            builder.string("weighted_sum")?
                        } else {
                            builder.string("text_only_bm25")?
                        },
                    ),
                    // Report the filtering strategy beside fusion settings because
                    // postfetch text filtering and prescore vector filtering can
                    // yield different candidate populations at the same limit.
                    ("filter", builder.string(filter_strategy)?),
                    (
                        "query_count",
                        JsonValue::Number(dataset.queries.len() as f64),
                    ),
                    (
                        "status",
                        // Set the descriptor's status from the overall comparison
                        // outcome. Discovering an executable or obtaining a
                        // version alone cannot mark the baseline as completed.
                        builder.string(if baseline_completed {
                            "completed"
                        } else {
                            "not_run"
                        })?,
                    ),
                ])?,
            ),
        ])
    }))
}

// The baseline name is the join key between configuration and results. Choose it
// from the dataset source in the same way regardless of individual query mode.
fn sqlite_baseline_name(dataset: &BenchmarkDataset) -> &'static str {
    if dataset.manifest.query_embedding_source != QueryEmbeddingSource::None {
        "sqlite-fts5-sqlite-vec"
    } else {
        "sqlite-fts5-text-only"
    }
}

// The configuration field accepts a date rather than a full timestamp. Retain the
// date portion of the runner's UTC time at verification.
fn benchmark_checked_at_date() -> String {
    crate::time::now_rfc3339()
        .chars()
        .take(10)
        .collect::<String>()
}

fn benchmark_baseline_warnings_in(
    outcome: &BenchmarkBaselineOutcome,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    // A completed baseline emits an empty warnings array. A missing optional
    // baseline contributes a structured warning while the native benchmark rows
    // remain available.
    if outcome.status == "completed" {
        return Ok(JsonValue::Array(Vec::new()));
    }
    // Keep the warning code stable and place environment-specific failure details
    // in reason. A reader can group skipped comparisons without matching their
    // free-form messages.
    builder.array(std::iter::once_with(|| {
        builder.object([
            ("code", builder.string("benchmark_baseline_not_run")?),
            ("message", builder.string("benchmark baseline did not run")?),
            ("reason", builder.optional_str(outcome.reason.as_deref())?),
        ])
    }))
}

// Ask the selected executable itself for its SQLite version. Searching installed
// libraries would not identify the implementation used by this CLI benchmark.
fn sqlite3_version() -> Option<String> {
    sqlite3_script(Path::new(":memory:"), "SELECT sqlite_version();")
        .ok()
        .map(|version| version.trim().to_owned())
}

#[derive(Clone, Debug)]
struct BenchmarkBaselineOutcome {
    // Whether the comparison completed or was recorded as not_run.
    status: String,
    // Reason for a missing baseline, retained for the final report and the
    // required-baseline check.
    reason: Option<String>,
    // Result objects for completed baseline runs, empty when no comparison ran.
    results: Vec<JsonValue>,
}

impl BenchmarkBaselineOutcome {
    // On success transfer the measured rows into the outcome. Clearing the reason
    // satisfies the report's completed-state contract.
    fn completed(results: Vec<JsonValue>) -> Self {
        Self {
            status: "completed".into(),
            reason: None,
            results,
        }
    }

    // Represent a failed or unavailable comparison without attaching partially
    // measured runs. Its reason is used in warnings and, for a required baseline,
    // in a command failure.
    fn not_run(reason: impl Into<String>) -> Self {
        Self {
            status: "not_run".into(),
            reason: Some(reason.into()),
            results: Vec::new(),
        }
    }
}

struct SqliteBaselineData {
    // Source-derived document and chunk data used for baseline metadata filtering.
    catalog: Catalog,
    // Map from dataset chunk IDs to the internal IDs in the baseline catalog.
    chunk_ids_by_external_id: BTreeMap<String, String>,
    // Context shared by source decoding and baseline work.
    json_context: JsonOperationContext,
    // Reservation held for the catalog rows and associated lookup data during the
    // baseline.
    _rows_work: CatalogReservation,
}

// An unavailable optional comparison is reportable state. The caller applies
// require_baseline separately, after it can include the specific reason.
fn benchmark_baseline_outcome(
    dataset: &BenchmarkDataset,
    profile: ResourceProfile,
    context: &JsonOperationContext,
) -> (BenchmarkBaselineOutcome, JsonReservation) {
    // FTS5 is required for both configurations. Probe this common capability first
    // so a missing text engine is reported consistently on vector-capable datasets
    // too.
    if !sqlite3_fts5_available() {
        return (
            BenchmarkBaselineOutcome::not_run("sqlite3 with FTS5 is unavailable"),
            context.reservation(),
        );
    }
    // Only a dataset with query vectors needs sqlite-vec. A text-only baseline can
    // proceed even if the extension is missing.
    if dataset.manifest.query_embedding_source != QueryEmbeddingSource::None
        && !sqlite3_sqlite_vec_available()
    {
        return (BenchmarkBaselineOutcome::not_run("sqlite-vec is unavailable; set COFFERDB_SQLITE_VEC_EXTENSION to a trusted local extension"), context.reservation());
    }
    // Translate recognized input locations back to dataset labels before
    // converting a baseline error into its public not_run reason.
    match run_sqlite_baseline(dataset, profile, context)
        .map_err(|error| dataset.locate_input_error(error))
    {
        // Retain the result reservation on success. An unsuccessful comparison
        // returns an empty guard because no partial measured rows are attached to
        // the outcome.
        Ok((results, work)) => (BenchmarkBaselineOutcome::completed(results), work),
        // A failure after a few successful runs still produces a single skipped
        // outcome. Native measurements are handled by the caller, and a required
        // baseline turns this reason into an error.
        Err(err) => (
            BenchmarkBaselineOutcome::not_run(format!("sqlite baseline failed: {err}")),
            context.reservation(),
        ),
    }
}

// Build one temporary comparison database from the verified dataset, then reuse it
// for the mode and cache-state runs. Database construction is outside the
// per-query timer.
fn run_sqlite_baseline(
    dataset: &BenchmarkDataset,
    profile: ResourceProfile,
    context: &JsonOperationContext,
) -> Result<(Vec<JsonValue>, JsonReservation)> {
    // Rebuild filter metadata from the benchmark's own source artifacts. The
    // baseline should not take its candidate metadata or row identities from the
    // CofferDB index it is compared with.
    let baseline_data = read_sqlite_baseline_data(dataset, context)?;
    // The guard owns the temporary database's filesystem lifetime while SQLite
    // accesses it by path. Later build or query failures return through the same
    // guard.
    let (file, db_file) = create_temporary_file("sqlite-baseline", "sqlite")?;
    // Release the descriptor returned by exclusive temp-file creation. SQLite
    // opens that pathname itself; the guard remains alive to remove it when the
    // comparison ends.
    drop(file);
    build_sqlite_fts5_database(db_file.path(), &baseline_data.catalog)?;
    // Every dataset needs the FTS table. Add the vector table only for a source
    // mode that actually supplies query embeddings.
    if dataset.manifest.query_embedding_source != QueryEmbeddingSource::None {
        build_sqlite_vector_database(db_file.path(), dataset, context)?;
    }
    let modes = benchmark_query_modes(dataset);
    let mut work = context.reservation();
    let mut results = Vec::new();
    // Budget one cold and one warm result object slot for each selected mode. The
    // query runners charge their retained JSON to the same result reservation.
    work.reserve_vec(
        &mut results,
        modes.len() * 2,
        JsonInputKind::BenchmarkArtifact,
    )?;
    // Run the cold-labelled pass before the warm repeat on the same database. A
    // later failure discards this function's partial results instead of returning
    // a mixed completion status.
    for mode in modes {
        let cold = run_sqlite_baseline_queries(
            db_file.path(),
            dataset,
            &baseline_data,
            profile,
            mode,
            "cold",
            &mut work,
        )?;
        let warm = run_sqlite_baseline_queries(
            db_file.path(),
            dataset,
            &baseline_data,
            profile,
            mode,
            "warm",
            &mut work,
        )?;
        // Append the two records only after both passes succeeded. Their order in
        // the report matches the cold pass followed by its warm repeat.
        results.push(cold);
        results.push(warm);
    }
    Ok((results, work))
}

fn create_temporary_file(label: &str, extension: &str) -> Result<(File, TempFileGuard)> {
    create_temporary_file_in(&std::env::temp_dir(), label, extension)
}

fn read_sqlite_baseline_data(
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
) -> Result<SqliteBaselineData> {
    // The document map, chunk catalog and external-ID lookup remain live through
    // the SQLite runs. Keep their catalog charge separate from the short-lived
    // JSON row decoding budget.
    let mut work = context.catalog_reservation();
    let documents = read_verified_baseline_input(
        &dataset.documents_path,
        "documents",
        &dataset.manifest.documents_sha256,
        |source| {
            // Decode documents from the verified reader passed to the callback.
            // Using the pathname to reopen a separate reader here would disconnect
            // this scan from its checksum.
            read_sqlite_baseline_documents(&dataset.documents_path, source, context, &mut work)
        },
    )?;
    // Read chunks only after documents have been decoded and their input hash
    // verified. Each chunk resolves its parent using the source document's
    // external ID.
    let (mut catalog, chunk_ids_by_external_id) = read_verified_baseline_input(
        &dataset.chunks_path,
        "chunks",
        &dataset.manifest.chunks_sha256,
        |source| {
            read_sqlite_baseline_chunks(
                &dataset.chunks_path,
                source,
                // Supply the document map by reference while building chunks. It
                // cannot be moved into its final internal-ID table until all
                // parent links have been resolved.
                &documents,
                context,
                &mut work,
            )
        },
    )?;
    // A validated text-only dataset normally has no embeddings path. If one is
    // supplied to this layer, the no-op visitor still streams and verifies it;
    // vector-capable datasets do that work while constructing their SQLite vector
    // table.
    if dataset.manifest.query_embedding_source == QueryEmbeddingSource::None {
        visit_sqlite_baseline_embeddings(dataset, context, |_| Ok(()))?;
    }
    // Rekey the staged documents by their internal IDs only after chunks have used
    // the external-ID lookup. Both key spaces are needed at different stages of
    // construction.
    install_baseline_documents(&mut catalog, documents, &mut work)?;
    Ok(SqliteBaselineData {
        catalog,
        chunk_ids_by_external_id,
        json_context: context.clone(),
        // Keep the catalog reservation inside the returned baseline state.
        // Releasing it on exit from this helper would leave the source-derived
        // catalog live without its charge.
        _rows_work: work,
    })
}

// The digest covers only bytes consumed by the callback. Before it returns
// success it must consume the input through EOF. If it streams rows into an
// external transaction, that transaction must stay uncommitted until this
// function checks the digest and returns success.
fn read_verified_baseline_input<T>(
    path: &Path,
    kind: &str,
    expected: &str,
    read: impl FnOnce(&mut Sha256Reader<BaselineSource>) -> Result<T>,
) -> Result<T> {
    let source = File::open(path)?;
    #[cfg(test)]
    let source = source_probe::Source::new(source, kind);
    // Wrap the same open file used by the consumer. Reopening the path for a
    // separate verification pass could confirm a different file than the one used
    // to build the baseline.
    let mut source = Sha256Reader::new(source);
    // A reader or visitor error takes precedence over digest comparison. The
    // partial byte stream cannot establish the identity of the complete artifact.
    let value = read(&mut source)?;
    // Finish the digest after the consumer has completed its own shape checks. A
    // succeeding callback may have constructed a value, but that value is
    // provisional until this comparison passes.
    let actual = source.finish();
    // Withhold the callback result when the calculated digest disagrees with the
    // original dataset identity. This check applies even to files already
    // validated during dataset loading.
    if actual != expected {
        return Err(
            Error::new(ErrorCode::ChecksumMismatch, format!("{kind} hash mismatch"))
                .with_detail("field", format!("hashes.{kind}_sha256"))
                .with_detail("artifact_kind", kind)
                .with_detail("path", path.display())
                .with_detail("expected_sha256", expected)
                .with_detail("actual_sha256", actual),
        );
    }
    #[cfg(test)]
    source_probe::verified(kind);
    Ok(value)
}

fn visit_sqlite_baseline_embeddings(
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
    mut consume: impl FnMut(crate::EmbeddingInput) -> Result<()>,
) -> Result<()> {
    // An absent optional corpus artifact supplies no vector rows. Once a path
    // exists, require a known digest before handing any row to the downstream
    // consumer.
    let Some(path) = &dataset.embeddings_path else {
        return Ok(());
    };
    let expected = dataset
        .manifest
        .embeddings_sha256
        .as_deref()
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark embeddings require a hash",
            )
            .with_detail("field", "hashes.embeddings_sha256")
        })?;
    // Pass each decoded embedding directly to its consumer. The hash closes over
    // the same input stream regardless of whether the consumer checks, discards or
    // stages the vector.
    read_verified_baseline_input(path, "embeddings", expected, |source| {
        visit_embeddings_jsonl_in(path, source, context, u64::MAX, |_, embedding, work| {
            // Keep the row's workspace guard live during the callback. A streamed
            // SQL writer can consume this one vector without retaining the decoded
            // rows for the rest of the file.
            let (_work, embedding) = (work, embedding);
            consume(embedding)
        })
    })
}

fn read_sqlite_baseline_documents(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
    retained: &mut CatalogReservation,
) -> Result<BTreeMap<String, Box<Document>>> {
    // Save the incoming catalog charge as this file's rollback boundary. A later
    // row failure releases allocations accumulated by this pass without changing
    // reservations owned by its caller beforehand.
    let before = retained.reserved_bytes();
    let result = (|| {
        // The temporary map is keyed by external ID for chunk-parent resolution.
        // Its documents already carry the internal IDs used after final
        // installation.
        let mut documents = BTreeMap::new();
        visit_documents_jsonl_in(path, source, context, u64::MAX, |line_number, row, work| {
            let (mut work, row) = (work, row);
            // Check uniqueness before allocating a synthetic ID or transferring
            // the row's ownership. There must be a single document target for each
            // external spelling used by chunks.
            if documents.contains_key(&row.external_id) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate document external_id in benchmark documents",
                )
                .with_detail("external_id", &row.external_id));
            }
            // Use the source line for an internal ID while retaining the supplied
            // external ID as the join key. The benchmark can construct a
            // standalone catalog without creating CofferDB documents.
            let (document_id, id_work) = ReservedString::build(context, "catalog_build", |out| {
                write!(out, "doc_{line_number}")
            })?
            .into_parts();
            // The new internal string comes with its own workspace charge. Merge
            // that charge into the parsed row before transferring ownership of
            // both to the catalog.
            work.merge_workspace(id_work);
            // The map needs its own external key in addition to the String stored
            // on the document. Charge the key, boxed record, collection spelling
            // and growth of the tree before inserting them.
            let additional = btree_workspace::<String, Box<Document>>(documents.len() + 1)
                - btree_workspace::<String, Box<Document>>(documents.len())
                + std::mem::size_of::<Document>()
                + row.external_id.len()
                + "col_1".len();
            // Transfer the decoded fields' existing credit to the longer-lived
            // catalog, then reserve the extra structures. This avoids releasing
            // and reacquiring the same row storage in between.
            work.transfer_to_catalog(retained, work.json_bytes(), work.workspace_bytes())?;
            // Charge additional space for the map and duplicated buffers
            // separately from the decoded row. These are allocations the parser's
            // reservation never had to cover.
            retained.grow(additional)?;
            documents.insert(
                copy_precharged_string(&row.external_id)?,
                Box::new(Document {
                    document_id,
                    collection_id: copy_precharged_string("col_1")?,
                    // Move owned source fields into the baseline document.
                    // Internal timestamps are left empty because this
                    // reconstruction needs identity, text provenance and metadata
                    // for matching.
                    external_id: row.external_id,
                    title: row.title,
                    source_uri: row.source_uri,
                    metadata: row.metadata,
                    created_at: String::new(),
                    updated_at: String::new(),
                }),
            );
            Ok(())
        })?;
        // An artifact can be valid JSONL and still contain no document rows.
        // Reject the empty source before constructing relationships that depend on
        // a document catalog.
        if documents.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark documents must not be empty",
            )
            .with_detail("path", path.display()));
        }
        Ok(documents)
    })();
    if result.is_err() {
        // The failed file's partial map is gone; earlier source files stay owned.
        retained.retain(before);
    }
    result
}

fn read_sqlite_baseline_chunks(
    path: &Path,
    source: impl Read,
    documents: &BTreeMap<String, Box<Document>>,
    context: &JsonOperationContext,
    retained: &mut CatalogReservation,
) -> Result<(Catalog, BTreeMap<String, String>)> {
    // Rows transfer their storage into a catalog shared by the baseline query paths.
    // Remember its previous reservation so a failure anywhere in this load releases
    // only the credit added for these chunks.
    let before = retained.reserved_bytes();
    let result = (|| {
        // Build the catalog by synthetic chunk ID and maintain the reverse lookup
        // by external ID. SQLite returns external IDs, while metadata filters use
        // CofferDB catalog relationships.
        let mut catalog = Catalog::default();
        let mut chunk_ids_by_external_id = BTreeMap::new();
        visit_chunks_jsonl_in(path, source, context, u64::MAX, |line_number, row, work| {
            let (mut work, row) = (work, row);
            // Resolve the exact parent external ID before discarding it. The
            // baseline stores the parent's internal ID on the chunk so its
            // document metadata can participate in filter evaluation.
            let document = documents.get(&row.document_external_id).ok_or_else(|| {
                Error::new(
                    ErrorCode::DocumentNotFound,
                    "benchmark chunk references unknown document_external_id",
                )
                .with_detail("document_external_id", &row.document_external_id)
            })?;
            // After resolving the parent, release its source spelling. Keep
            // workspace credit for the external chunk ID and text buffers that are
            // moved into the two retained structures.
            drop(row.document_external_id);
            work.retain_workspace(
                row.chunk.chunk_external_id.capacity() + row.chunk.text.capacity(),
            );
            // Move text, metadata and source span from the decoded row. The
            // external-ID lookup and the owned catalog record require distinct
            // copies of only some fields.
            let crate::ChunkInput {
                chunk_external_id,
                text,
                ordinal,
                metadata,
                source_span,
            } = row.chunk;
            // Duplicate external chunk IDs would make SQLite result mapping
            // ambiguous even if the next synthetic ID were distinct. Detect the
            // collision before installing the row.
            if chunk_ids_by_external_id.contains_key(&chunk_external_id) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate chunk_external_id in benchmark chunks",
                )
                .with_detail("chunk_external_id", &chunk_external_id));
            }
            // Compute the token hint from the exact chunk text and generate a
            // line-based internal ID. The original external ID remains the
            // identity used in queries and judgments.
            let token_count = token_count_hint_in(&text, context)?;
            let (chunk_id, id_work) = ReservedString::build(context, "catalog_build", |out| {
                write!(out, "chk_{line_number}")
            })?
            .into_parts();
            // Preserve the ID buffer's charge alongside the original row
            // workspace. The reverse map ultimately owns this generated ID, while
            // the record and its catalog key keep copies.
            work.merge_workspace(id_work);
            // Hash the underlying UTF-8 text bytes and reserve the exact hex
            // width. This is the per-chunk hash, distinct from the SHA-256 of the
            // entire source artifact.
            let (text_hash, hash_work) =
                ReservedString::build_exact(64, context, "catalog_build", |out| {
                    crate::hash::write_digest_hex(out, crate::hash::sha256(text.as_bytes()))
                })?
                .into_parts();
            work.merge_workspace(hash_work);
            // Each admitted chunk grows both trees by one entry. In addition to
            // the moved row buffers, charge the Box and the identifier copies held
            // by the record, catalog key and reverse lookup.
            let entries = catalog.chunks.len();
            let additional = btree_workspace::<String, Box<Chunk>>(entries + 1)
                - btree_workspace::<String, Box<Chunk>>(entries)
                + btree_workspace::<String, String>(entries + 1)
                - btree_workspace::<String, String>(entries)
                + std::mem::size_of::<Chunk>()
                + 2 * chunk_id.len()
                + document.document_id.len()
                + "col_1".len()
                + chunk_external_id.len();
            // Move the row and constructed-string reservations into the catalog
            // before making the precharged copies. The enclosing error path
            // restores the previous charge if construction fails.
            work.transfer_to_catalog(retained, work.json_bytes(), work.workspace_bytes())?;
            retained.grow(additional)?;
            let chunk = Chunk {
                chunk_id: copy_precharged_string(&chunk_id)?,
                document_id: copy_precharged_string(&document.document_id)?,
                collection_id: copy_precharged_string("col_1")?,
                chunk_external_id: copy_precharged_string(&chunk_external_id)?,
                text_hash,
                token_count,
                text,
                ordinal,
                metadata,
                // The query filter catalog does not own embedding rows. SQLite's
                // separate vector table handles vector scoring, so reconstructed
                // chunks have no internal embedding ID.
                embedding_id: None,
                source_span,
                created_at: String::new(),
            };
            catalog
                .chunks
                .insert(copy_precharged_string(&chunk_id)?, Box::new(chunk));
            // Keep SQLite's external ID as the reverse-map key. The synthetic ID
            // it resolves to is the same one used for the catalog entry just
            // installed.
            chunk_ids_by_external_id.insert(chunk_external_id, chunk_id);
            Ok(())
        })?;
        // A valid documents artifact can coexist with an empty chunks file. Reject
        // that state before the baseline builds a searchable table containing no
        // retrieval units.
        if catalog.chunks.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark chunks must not be empty",
            )
            .with_detail("path", path.display()));
        }
        Ok((catalog, chunk_ids_by_external_id))
    })();
    if result.is_err() {
        retained.retain(before);
    }
    result
}

// Move the existing document boxes into the catalog without copying their
// contents. Only the map structure and its lookup keys change from source external
// IDs to internal IDs.
fn install_baseline_documents(
    catalog: &mut Catalog,
    documents: BTreeMap<String, Box<Document>>,
    retained: &mut CatalogReservation,
) -> Result<()> {
    // The old map owns key buffers with their original capacities. The replacement
    // keys are fresh copies of the internal document IDs and must be charged
    // separately.
    let old_keys = documents.keys().map(String::capacity).sum::<usize>();
    let new_keys = documents
        .values()
        .map(|doc| doc.document_id.len())
        .sum::<usize>();
    // The replacement table contains the same number of document boxes, but its
    // map nodes and key buffers temporarily overlap those of the source table.
    let tree = btree_workspace::<String, Box<Document>>(documents.len());
    // Both trees and their distinct keys coexist while the boxes are moved.
    retained.grow(tree + new_keys)?;
    for (_, document) in documents {
        catalog
            .documents
            .insert(copy_precharged_string(&document.document_id)?, document);
    }
    // The consumed map and all its external key buffers are gone. Release their
    // tree and key charges while leaving the new keys and moved document bodies
    // covered.
    retained.retain(retained.reserved_bytes() - tree - old_keys);
    Ok(())
}

// Populate the temporary comparison database through a single SQLite script.
// Journal and sync settings favor a disposable benchmark index, separate from the
// database under test.
fn build_sqlite_fts5_database(path: &Path, catalog: &Catalog) -> Result<()> {
    sqlite::execute(path, |output| {
        output.write_all(
            b"PRAGMA journal_mode=OFF;\n\
         PRAGMA synchronous=OFF;\n\
         CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text);\n\
         BEGIN;\n",
        )?;
        // Assign explicit rowids as the catalog is traversed, but store external
        // chunk IDs with the FTS rows. Returned rankings must join to the
        // benchmark's original judgments.
        for (index, chunk) in catalog.chunks.values().enumerate() {
            write!(
                output,
                // SQLite FTS needs a numeric rowid for storage. Keep the dataset's
                // external chunk identifier in a separate unindexed column so it
                // is returned unchanged and does not contribute terms.
                "INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES({}, ",
                index + 1
            )?;
            // Quote both IDs and text as SQL strings at the writer boundary. A
            // quote or newline in dataset text must remain part of one inserted
            // value.
            sqlite::write_string(output, &chunk.chunk_external_id)?;
            output.write_all(b", ")?;
            sqlite::write_string(output, &chunk.text)?;
            output.write_all(b");\n")?;
        }
        // Commit after every row has been emitted. If streaming fails beforehand,
        // the helper returns an error and this incomplete temporary database is
        // never used for queries.
        output.write_all(b"COMMIT;\n")?;
        Ok(())
    })
}

// Run one SQLite mode and cache-state pass over the benchmark workload. Keep the
// returned report charged to its owner; temporary latency samples have a separate
// reservation.
fn run_sqlite_baseline_queries(
    db_path: &Path,
    dataset: &BenchmarkDataset,
    baseline_data: &SqliteBaselineData,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    cache_state: &str,
    work: &mut JsonReservation,
) -> Result<JsonValue> {
    // Use scratch for values required only while aggregating the run. Per-query
    // records are retained in the artifact and grow under work instead.
    let mut scratch = baseline_data.json_context.reservation();
    let mut per_query = Vec::new();
    let mut latencies = Vec::new();
    // Accumulate per-query relevance measurements before dividing by query count.
    // This gives each query equal weight regardless of the number of relevant
    // chunks.
    let mut recall_at_5_sum = 0.0;
    let mut recall_at_10_sum = 0.0;
    let mut reciprocal_rank_sum = 0.0;
    // Aggregate execution counters independently of quality metrics. Counts
    // describe work performed, while recall and reciprocal rank depend on the
    // returned order.
    let mut vector_scan_count = 0;
    let mut distance_ops = 0;
    let mut fusion_candidate_count = 0;

    // Use the same workload order for native and baseline passes. The returned row
    // carries the query ID so consumers need not derive identity from its index.
    for query in &dataset.queries {
        // Measure the SQLite query routine, including its result and context
        // construction. Creating the enclosing JSON result row and computing
        // relevance occur after this timer is sampled.
        let started = Instant::now();
        let (ranked_chunk_ids, scores, stats, query_context) = sqlite_baseline_query(
            db_path,
            query,
            baseline_data,
            profile,
            mode,
            // Charge the query context to the eventual result before the per-query
            // record is assembled. It must remain owned with the rest of that
            // record through serialization.
            &JsonBuilder::budgeted(work, JsonInputKind::BenchmarkArtifact),
        )?;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        // A hit means at least one returned ID is relevant for this query. The
        // boolean complements, rather than replaces, the ranked metrics computed
        // below.
        let hit = ranked_chunk_ids
            .iter()
            .any(|chunk_id| query.relevant_chunk_ids.contains(chunk_id));
        // Use the same ranked IDs for recall at both cutoffs and reciprocal rank.
        // Keep those calculations independent of the engine-local score scale.
        recall_at_5_sum += recall_at_k(ranked_chunk_ids.iter().map(String::as_str), query, 5);
        recall_at_10_sum += recall_at_k(ranked_chunk_ids.iter().map(String::as_str), query, 10);
        reciprocal_rank_sum += reciprocal_rank(ranked_chunk_ids.iter().map(String::as_str), query);
        // Admit one latency sample before growing the temporary array. Percentile
        // calculation later uses that array in place.
        scratch.reserve_vec(&mut latencies, 1, JsonInputKind::BenchmarkArtifact)?;
        latencies.push(latency_ms);
        // Accumulate each query's reported work, including scores that never made
        // the final rank list. The aggregate does not derive work from the number
        // of returned IDs.
        vector_scan_count += stats.vector_scan_count;
        distance_ops += stats.distance_ops;
        fusion_candidate_count += stats.fusion_candidate_count;
        // Admit a retained array slot before assembling the next row. The builder
        // charges the row's keys, values and nested arrays to the same report
        // owner.
        work.reserve_vec(&mut per_query, 1, JsonInputKind::BenchmarkArtifact)?;
        // Reuse one builder per row for its nested values. Any budget refusal
        // aborts this run before it can return a partially built result.
        let builder = JsonBuilder::budgeted(work, JsonInputKind::BenchmarkArtifact);
        per_query.push(
            builder.object([
                ("query_id", builder.string(&query.query_id)?),
                ("latency_ms", JsonValue::Number(latency_ms)),
                (
                    // Move rank identifiers into the final array in ranking order.
                    // The scores array below uses the same order and length.
                    "ranked_chunk_ids",
                    builder.array(
                        ranked_chunk_ids
                            .into_iter()
                            .map(|id| builder.owned_string(id)),
                    )?,
                ),
                // Name the score kind using the SQLite table for this mode. Its
                // per-engine scale is required even when a normalized score lies
                // within a familiar interval.
                ("score_kind", builder.string(mode.sqlite_score_kind())?),
                ("score_scale", builder.string("engine_local")?),
                ("score_comparable", JsonValue::Bool(false)),
                (
                    // Retain the exact score array returned by the chosen route.
                    // The validator pairs values with corresponding ranked IDs by
                    // their positions.
                    "scores",
                    builder.array(scores.into_iter().map(|value| Ok(JsonValue::Number(value))))?,
                ),
                ("hit", JsonValue::Bool(hit)),
                ("execution_stats", stats.build_json(&builder)?),
                ("query_context", query_context),
            ])?,
        );
    }

    // Divide each relevance sum by the same workload count. Dataset preflight
    // supplies nonempty query input before a baseline is run.
    let query_count = dataset.queries.len() as f64;
    // The percentile array contains one observation per completed query, while
    // each relevance aggregate uses the same query count as denominator.
    let metrics = BenchmarkRunMetrics {
        recall_at_5: recall_at_5_sum / query_count,
        recall_at_10: recall_at_10_sum / query_count,
        mrr: reciprocal_rank_sum / query_count,
        // Calculate the latency percentile in place. The sample buffer is scratch
        // and need not survive the aggregate JSON conversion.
        p95_ms: percentile_95_in_place(&mut latencies),
        // SQLite runs in child processes; the parent's peak does not measure this baseline.
        memory_peak_mb: None,
        vector_scan_count,
        distance_ops,
        fusion_candidate_count,
    };
    // Publish enough run identity to link each row group to its SQLite
    // configuration and cache state in the result contract.
    let builder = JsonBuilder::budgeted(work, JsonInputKind::BenchmarkArtifact);
    builder.object([
        // Use the engine label and configuration name expected by baseline link
        // validation. The cache state applies to the entire repeated workload
        // pass.
        ("engine", builder.string("sqlite")?),
        ("name", builder.string(sqlite_baseline_name(dataset))?),
        ("query_mode", builder.string(mode.as_str())?),
        ("cache_state", builder.string(cache_state)?),
        ("metrics", benchmark_metrics_json_in(&metrics, &builder)?),
        ("per_query", JsonValue::Array(per_query)),
    ])
}

// Execute one baseline query and return ranks, scores, work counters and their
// context together. Each route retains its own score meaning.
fn sqlite_baseline_query(
    db_path: &Path,
    query: &BenchmarkQuery,
    baseline_data: &SqliteBaselineData,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    builder: &JsonBuilder<'_>,
) -> Result<(Vec<String>, Vec<f64>, BenchmarkExecutionStats, JsonValue)> {
    let params = profile.params();
    // Give the text route at least top_k slots when it runs. A mode without text
    // reports zero for its text window.
    let bm25_limit = if mode.has_text() {
        params.bm25_candidates.max(query.top_k)
    } else {
        0
    };
    // Use the same request floor for the vector route. Disabling the route gives
    // it zero capacity rather than assigning an unused profile allowance.
    let vector_limit = if mode.has_vector() {
        params.vector_candidates.max(query.top_k)
    } else {
        0
    };
    // Record a fusion candidate bound when vector search is present, even in
    // vector-only mode. Only the hybrid branch calls the fusion algorithm below.
    let fusion_limit = if mode.has_vector() {
        params.fusion_candidates.max(query.top_k)
    } else {
        0
    };
    // Text reports the converted FTS5 rank score, while exact vector reports
    // normalized cosine similarity. Hybrid reports the final weighted fusion score
    // even if the resulting chunk order matches a single-signal run.
    let (ranked_chunk_ids, scores, stats) = match mode {
        // Select the highest-ranked text hits up to top_k. This branch keeps the
        // converted FTS score and carries no vector counters.
        BenchmarkQueryMode::TextBm25 => {
            let hits = sqlite_fts5_hits(db_path, query, baseline_data, bm25_limit)?;
            // Limit only the outward list to top_k at this point. The FTS route
            // may have inspected a larger candidate prefix allowed by its window.
            let ranked_chunk_ids = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.chunk_id.clone())
                .collect::<Vec<_>>();
            // Read scores from the same ordered prefix used for IDs. Preserve
            // array alignment even when filtering returned fewer candidates than
            // the limit.
            let scores = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.score)
                .collect::<Vec<_>>();
            (ranked_chunk_ids, scores, BenchmarkExecutionStats::default())
        }
        // Use exact vector candidates when text retrieval is off. Report
        // normalized cosine similarity while retaining the branch's scan and
        // distance statistics.
        BenchmarkQueryMode::VectorExact => {
            let (hits, stats) = sqlite_vec_exact_hits(db_path, query, baseline_data, vector_limit)?;
            let ranked_chunk_ids = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.chunk_id.clone())
                .collect::<Vec<_>>();
            let scores = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.normalized_score)
                .collect::<Vec<_>>();
            (ranked_chunk_ids, scores, stats)
        }
        // Retrieve both independently bounded candidate sets before merging. Each
        // route can contribute IDs that are absent from the other set.
        BenchmarkQueryMode::Hybrid => {
            // The hybrid route evaluates the same filter in both index paths.
            // Their bounded outputs are joined by external chunk identity during
            // fusion.
            let bm25_hits = sqlite_fts5_hits(db_path, query, baseline_data, bm25_limit)?;
            let (vector_hits, mut stats) =
                sqlite_vec_exact_hits(db_path, query, baseline_data, vector_limit)?;
            // Use default weights for two active channels without metadata boost.
            // The baseline builds context from the same modality defaults.
            let weights = SearchWeights::default_for(true, true);
            // Pass the execution-fusion candidate limit rather than top_k. The
            // final rank list may still be shortened after fusion completes.
            let fused = fuse(
                &bm25_hits,
                &vector_hits,
                weights,
                fusion_limit,
                &baseline_data.json_context,
                // Supply a default metadata contribution for any candidate.
                // Baseline fusion uses the two retrieval scores only.
                |_| Default::default(),
            )?;
            let fused = fused.as_slice();
            // Record the size of the fused candidate list before truncating to the
            // requested output length. This is a count of retained fused
            // candidates.
            stats.fusion_candidate_count = fused.len();
            let ranked_chunk_ids = fused
                .iter()
                .take(query.top_k)
                .map(|hit| hit.chunk_id.to_owned())
                .collect::<Vec<_>>();
            let scores = fused
                .iter()
                .take(query.top_k)
                .map(|hit| hit.final_score)
                .collect::<Vec<_>>();
            (ranked_chunk_ids, scores, stats)
        }
    };
    // Build context from the same limits, mode and statistics that produced the
    // returned arrays. It must describe actual baseline behavior, including
    // disabled routes.
    let query_context = sqlite_baseline_query_context(
        query,
        profile,
        mode,
        CandidateLimits {
            bm25: bm25_limit,
            vector: vector_limit,
            fusion: fusion_limit,
        },
        stats,
        builder,
    )?;
    Ok((ranked_chunk_ids, scores, stats, query_context))
}

// Produce the shared query-context shape for a SQLite result. The values record
// applicable SQLite execution details and explicit absence for native-only
// features.
fn sqlite_baseline_query_context(
    query: &BenchmarkQuery,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    limits: CandidateLimits,
    execution_stats: BenchmarkExecutionStats,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    // Derive defaults from the declared active channels. No user weight override
    // is applied by this baseline path.
    let weights = SearchWeights::default_for(mode.has_text(), mode.has_vector());
    // Report filter presence from the parsed request shape. An empty object uses
    // no metadata predicate.
    let filter_present = query
        .filter
        .as_object()
        .map(|obj| !obj.is_empty())
        .unwrap_or(false);
    builder.object([
        // Set presence flags from the executed mode. A vector-only baseline can
        // start from a workload row that also contains text.
        ("text_present", JsonValue::Bool(mode.has_text())),
        // CofferDB tokenizer metadata is not applicable to the SQLite baseline
        // result. The shared schema represents that absence explicitly.
        ("tokenizer", JsonValue::Null),
        ("embedding_present", JsonValue::Bool(mode.has_vector())),
        ("filter_present", JsonValue::Bool(filter_present)),
        (
            // Keep nested and summary weight fields in agreement. Both views
            // describe the same default weights selected for this baseline mode.
            "weights",
            builder.object([
                ("present", JsonValue::Bool(false)),
                ("source", builder.string("default_no_boost")?),
                (
                    "requested",
                    benchmark_search_weights_json_in(weights, builder)?,
                ),
                (
                    "effective",
                    benchmark_search_weights_json_in(weights, builder)?,
                ),
                // Default coefficients are used directly here. An explicit null
                // records that no renormalization reason is present.
                ("renormalization_reason", JsonValue::Null),
            ])?,
        ),
        (
            // Keep requested and effective weights equal for this baseline. Query-
            // or metadata-driven adjustments are not applied here.
            "requested_weights",
            benchmark_search_weights_json_in(weights, builder)?,
        ),
        (
            "effective_weights",
            benchmark_search_weights_json_in(weights, builder)?,
        ),
        ("weights_source", builder.string("default_no_boost")?),
        ("renormalization_reason", JsonValue::Null),
        // The baseline runs without metadata boost. Emit both false flags, a null
        // configuration fingerprint and a zero rule count for the shared result
        // schema.
        ("metadata_boost_present", JsonValue::Bool(false)),
        ("metadata_boost_enabled", JsonValue::Bool(false)),
        ("boost_config_hash", JsonValue::Null),
        ("boost_rule_count", JsonValue::Number(0.0)),
        // Record the requested number of results and the selected profile
        // separately from the computed candidate windows.
        ("top_k", JsonValue::Number(query.top_k as f64)),
        ("profile", builder.string(profile.as_json_str())?),
        (
            // Expose separate capacity bounds for each path. A zero limit
            // distinguishes routes disabled by the mode from routes that ran and
            // returned no hits.
            "candidate_limits",
            builder.object([
                ("bm25", JsonValue::Number(limits.bm25 as f64)),
                ("vector", JsonValue::Number(limits.vector as f64)),
                ("fusion", JsonValue::Number(limits.fusion as f64)),
            ])?,
        ),
        // Mirror the execution counters from the route result. The index source
        // label identifies which SQLite index family produced the search evidence.
        ("execution_stats", execution_stats.build_json(builder)?),
        ("index_source", builder.string(sqlite_index_source(mode))?),
        // SQLite does not use CofferDB's transient rebuild path. Keep the nullable
        // observations empty and the rebuild state flags false.
        ("index_rebuild_reason", JsonValue::Null),
        (
            "transient_rebuild",
            builder.object([
                ("started", JsonValue::Bool(false)),
                ("completed", JsonValue::Bool(false)),
                ("cancelled", JsonValue::Bool(false)),
                ("budget_mb", JsonValue::Number(0.0)),
                // Leave native rebuild work measurements null. Zero would instead
                // imply that a measurement was available and found no work.
                ("chunks_scanned", JsonValue::Null),
                ("bytes_read", JsonValue::Null),
                ("budget_exceeded", JsonValue::Bool(false)),
            ])?,
        ),
    ])
}

// Emit all three weight components even when a mode leaves some coefficients at
// zero. The result schema uses identical tuple shapes at every weight location.
fn benchmark_search_weights_json_in(
    weights: SearchWeights,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        ("bm25", JsonValue::Number(weights.bm25)),
        ("vector", JsonValue::Number(weights.vector)),
        ("metadata", JsonValue::Number(weights.metadata)),
    ])
}

// Choose an index source from the actual baseline mode. Hybrid evidence identifies
// both SQLite families rather than claiming it came from one index.
fn sqlite_index_source(mode: BenchmarkQueryMode) -> &'static str {
    match mode {
        BenchmarkQueryMode::TextBm25 => "sqlite_fts5",
        BenchmarkQueryMode::VectorExact => "sqlite_vec",
        BenchmarkQueryMode::Hybrid => "sqlite_fts5_sqlite_vec",
    }
}

// Collect the SQLite text route's candidates using the active predicate. Keep
// ranking work bounded by the query's effective text window.
fn sqlite_fts5_hits(
    db_path: &Path,
    query: &BenchmarkQuery,
    baseline_data: &SqliteBaselineData,
    candidate_limit: usize,
) -> Result<Bm25Hits> {
    // Missing text produces an empty channel without opening a child process.
    // Present text can also produce no executable FTS terms.
    let Some(text) = query.text.as_deref() else {
        // Return the empty collection under the caller's context, so paths that
        // skip retrieval still use a consistent accounting owner.
        return Ok(Bm25Hits::empty(&baseline_data.json_context));
    };
    // Skip the FTS query when tokenization emitted no terms. An empty match
    // expression cannot yield text candidates.
    let Some(match_query) = sqlite_fts5_match_query_in(text, &baseline_data.json_context)? else {
        return Ok(Bm25Hits::empty(&baseline_data.json_context));
    };
    // Parse the metadata predicate once for the whole text route. The charged
    // expression remains available while score rows are streamed back.
    let (filter_expr, _filter_work) = FilterExpr::parse_validated_in(
        &query.filter,
        &baseline_data.json_context,
        JsonInputKind::Filter,
    )?;
    // Bound the retained text candidates while visiting SQLite output. Only rows
    // admitted by the filter and candidate order become hits.
    let mut hits = TextCandidates::new(candidate_limit, &baseline_data.json_context);
    sqlite::query(
        db_path,
        |output| {
            // Write the MATCH argument through the SQL string writer. The FTS
            // expression already has its own term quoting, and this additional
            // layer represents it as one SQL value.
            output.write_all(b"SELECT json_object('chunk_id',chunk_id,'score',bm25(chunks_fts)) FROM chunks_fts WHERE chunks_fts MATCH ")?;
            sqlite::write_string(output, match_query.as_ref())?;
            // Order by SQLite's BM25 column and then chunk ID before applying the
            // SQL limit. The later metadata predicate can remove items from that
            // limited prefix without requesting replacements.
            writeln!(
                output,
                " ORDER BY bm25(chunks_fts), chunk_id LIMIT {candidate_limit};"
            )?;
            Ok(())
        },
        |input| {
            // Stream rows into the CofferDB-side predicate evaluation. The SQLite
            // FTS index supplies score and external ID, while metadata remains in
            // the shared catalog.
            sqlite::visit_scores(input, &baseline_data.json_context, |chunk_id, raw_score| {
                if sqlite_baseline_candidate_matches_filter(&chunk_id, &filter_expr, baseline_data)
                {
                    hits.push(
                        &chunk_id,
                        // Translate the SQLite score direction before feeding the
                        // descending common candidate order. This path also
                        // supplies no matched-term evidence to the text hit.
                        sqlite_fts5_rank_score(raw_score),
                        std::iter::empty(),
                    )?;
                }
                Ok(())
            })
        },
    )?;
    // Finalize the bounded candidate structure only after the SQLite child and row
    // visitor completed successfully.
    Ok(hits.finish())
}

// Negate nonpositive FTS5 scores to make better matches larger. A positive input
// uses reciprocal scaling so better lower ranks still produce larger values.
fn sqlite_fts5_rank_score(raw_score: f64) -> f64 {
    if raw_score <= 0.0 {
        -raw_score
    } else {
        1.0 / (1.0 + raw_score)
    }
}

// Construct the baseline vector database through the SQLite child. The writer
// streams artifact rows through the same admitted context used by the baseline.
fn build_sqlite_vector_database(
    path: &Path,
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
) -> Result<()> {
    sqlite::execute(path, |output| {
        write_sqlite_vectors(output, dataset, context)
    })
}

// Write the vector table in one SQL transaction. Each row keeps its external chunk
// ID, actual width and binary32 vector payload.
fn write_sqlite_vectors(
    output: &mut impl Write,
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
) -> Result<()> {
    output.write_all(
        b"BEGIN; CREATE TABLE chunks_vec(chunk_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL);\n",
    )?;
    // Stream validated embedding rows individually. Their source hash is finalized
    // only after visiting the last row, before the transaction receives COMMIT.
    visit_sqlite_baseline_embeddings(dataset, context, |embedding| {
        // Use the SQLite string writer for the external identifier. The array
        // writer serializes numeric coordinates separately for vec_f32.
        output.write_all(b"INSERT OR REPLACE INTO chunks_vec VALUES(")?;
        sqlite::write_string(output, &embedding.chunk_external_id)?;
        // Pass numeric coordinates through the product's vector JSON writer.
        // SQLite's vec_f32 call consumes that representation as a vector blob.
        write!(output, ",{},vec_f32('", embedding.vector.len())?;
        crate::json::write_vector_json(output, &embedding.vector)?;
        output.write_all(b"'));\n")?;
        Ok(())
    })?;
    // The producing stream, including its trailing bytes, is verified before commit.
    output.write_all(b"COMMIT;\n")?;
    Ok(())
}

// Score dimension-matched SQLite vectors for the exact baseline route. Keep a
// bounded ranked list and separately account for every score row returned by
// SQLite.
fn sqlite_vec_exact_hits(
    db_path: &Path,
    query: &BenchmarkQuery,
    baseline_data: &SqliteBaselineData,
    candidate_limit: usize,
) -> Result<(VectorHits, BenchmarkExecutionStats)> {
    // A query with no embedding yields an empty vector route and zero execution
    // statistics. It does not open the SQLite child.
    let Some(embedding) = query.embedding.as_ref() else {
        return Ok((
            VectorHits::empty(&baseline_data.json_context),
            BenchmarkExecutionStats::default(),
        ));
    };
    // Resolve the optional predicate once and keep its reservation until both
    // request construction and row consumption finish.
    let (filter_expr, _filter_work) = FilterExpr::parse_validated_in(
        &query.filter,
        &baseline_data.json_context,
        JsonInputKind::Filter,
    )?;
    // Keep only the number of vector winners allowed by the candidate limit.
    // Stream score rows so the full exact scan need not become a large in-memory
    // list.
    let mut hits = VectorCandidates::new(candidate_limit, &baseline_data.json_context);
    let ((), vector_scan_count) = sqlite::query(
        db_path,
        |output| {
            // Use cosine distance and treat a null result as distance one. The
            // retained result is normalized only after the distance is read back
            // from SQLite.
            output.write_all(b"SELECT json_object('chunk_id',chunk_id,'score',COALESCE(vec_distance_cosine(vector,vec_f32('")?;
            crate::json::write_vector_json(output, &embedding.vector)?;
            // Restrict candidate rows to the query's actual vector width. The
            // stored dimension column prevents applying distance to incompatible
            // vectors.
            write!(
                output,
                "')),1.0)) FROM chunks_vec WHERE dim={}",
                embedding.vector.len()
            )?;
            // Apply the same metadata predicate to external chunk IDs before SQLite
            // scores vectors. An empty match set becomes IN (NULL), which yields no
            // rows while keeping the generated query valid.
            // When the predicate is absent, dimension remains the only SQL
            // eligibility condition in this branch.
            if filter_expr.is_some() {
                output.write_all(b" AND chunk_id IN (")?;
                let mut count = 0;
                // Use the baseline's external-ID map to enumerate IDs for the
                // predicate. The emitted SQL includes only those whose catalog
                // metadata passes.
                for chunk_external_id in baseline_data.chunk_ids_by_external_id.keys() {
                    if !sqlite_baseline_candidate_matches_filter(
                        chunk_external_id,
                        &filter_expr,
                        baseline_data,
                    ) {
                        continue;
                    }
                    // Stream accepted IDs directly to the child request and
                    // separate them with commas. No second materialized ID list is
                    // needed.
                    if count != 0 {
                        output.write_all(b",")?;
                    }
                    sqlite::write_string(output, chunk_external_id)?;
                    count += 1;
                }
                if count == 0 {
                    output.write_all(b"NULL")?;
                }
                output.write_all(b")")?;
            }
            output.write_all(b";\n")?;
            Ok(())
        },
        // Count returned score rows during the visit instead of inferring work
        // from the bounded list of winners.
        |input| {
            let mut count = 0;
            // The SQLite extension returns cosine distance. Convert it to similarity
            // before feeding the common candidate ordering, and count every row
            // scored even when the bounded candidate list discards it.
            sqlite::visit_scores(input, &baseline_data.json_context, |chunk_id, distance| {
                count += 1;
                // Convert distance to cosine similarity and clamp at the
                // mathematical endpoints. Normalize the clamped value using the
                // same cosine scale as the shared candidate representation.
                let raw = (1.0 - distance).clamp(-1.0, 1.0);
                // Pass raw and normalized similarities together. Shared ranking
                // can use the appropriate representation without recomputing the
                // distance.
                hits.push(
                    &chunk_id,
                    raw,
                    normalize_score(raw, DistanceMetric::Cosine, false),
                )
            })?;
            Ok(count)
        },
    )?;
    let hits = hits.finish();
    // Report scan count from score rows and estimate scalar distance work as count
    // times width. Candidate count describes the retained route result, which can
    // be smaller than the scan.
    let stats = BenchmarkExecutionStats {
        vector_scan_count,
        // Report distance_ops in component units. It is derived from matched-width
        // rows rather than a hardware instruction counter.
        distance_ops: vector_scan_count * embedding.dim,
        fusion_candidate_count: hits.len(),
    };
    Ok((hits, stats))
}

// Evaluate SQLite candidate metadata using the catalog's shared FilterExpr logic.
// This keeps document and chunk field handling consistent across retrieval routes.
fn sqlite_baseline_candidate_matches_filter(
    chunk_external_id: &str,
    filter_expr: &Option<FilterExpr>,
    baseline_data: &SqliteBaselineData,
) -> bool {
    // Without a predicate every supplied candidate is eligible. A present
    // predicate requires resolving the index's external ID back to a catalog row.
    let Some(expr) = filter_expr else {
        return true;
    };
    // Use the external-to-internal mapping from baseline setup. A stale or missing
    // mapping cannot satisfy a metadata predicate.
    let Some(chunk_id) = baseline_data
        .chunk_ids_by_external_id
        .get(chunk_external_id)
    else {
        return false;
    };
    // If the catalog lacks the mapped chunk, treat the candidate as a miss.
    // Evaluate metadata only with its live row and parent linkage.
    let Some(chunk) = baseline_data.catalog.chunks.get(chunk_id) else {
        return false;
    };
    // Evaluate the original predicate against the shared chunk and document
    // metadata. This is the same evaluator used by the native path.
    expr.matches_for_chunk(&baseline_data.catalog, chunk)
}

// Build a reserved FTS expression with a sizing pass and a rendering pass. The
// temporary allocation covers exactly its encoded bytes.
fn sqlite_fts5_match_query_in(
    text: &str,
    context: &JsonOperationContext,
) -> Result<Option<crate::json::ReservedString>> {
    let mut bytes = 0usize;
    // Visit terms in the same order as rendering. Each term adds quotes, its UTF-8
    // bytes, escaped quote bytes and a separator after the first term.
    crate::tokenizer::visit_tokens_in(text, context, |term| {
        // The separator is the four-byte token joining alternatives. Keep this
        // count and the literal in render_sqlite_fts5_match_query_in in sync.
        let separator = usize::from(bytes != 0) * 4;
        bytes = bytes
            .checked_add(separator)
            .and_then(|size| size.checked_add(2))
            .and_then(|size| size.checked_add(term.len()))
            .and_then(|size| size.checked_add(term.bytes().filter(|byte| *byte == b'"').count()))
            // Use checked size arithmetic so an oversized escaped expression
            // cannot wrap into a smaller reservation request.
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "SQLite match query length overflow",
                )
            })?;
        Ok(())
    })?;
    // Return no expression when no term was emitted. The caller treats this as an
    // empty candidate channel.
    if bytes == 0 {
        return Ok(None);
    }
    // Admit capacity before the second token visit. A changed rendering rule that
    // exceeds the estimate fails through the bounded writer.
    let mut query =
        crate::json::ReservedString::with_capacity(bytes, context, "operation_scratch")?;
    // Retokenize instead of retaining a second owned list of tokens. The length
    // pass releases each temporary term before the final expression allocation
    // grows.
    render_sqlite_fts5_match_query_in(text, &mut query, context)?;
    // The bounded writer prevents excess bytes, but a shorter result also
    // indicates divergence from the sizing pass. Verify exact agreement before
    // using the expression.
    if query.as_ref().len() != bytes {
        return Err(std::io::Error::other("SQLite match query sizing differs").into());
    }
    Ok(Some(query))
}

// Render quoted FTS terms joined by OR. This is FTS syntax inside a later SQL
// string, so its quote rules are applied separately.
fn render_sqlite_fts5_match_query_in(
    text: &str,
    query: &mut crate::json::ReservedString,
    context: &JsonOperationContext,
) -> Result<()> {
    // Track whether a separator is needed across token callbacks. The first
    // emitted term starts directly with its opening quote.
    let mut first = true;
    // Visit normalized terms directly in the renderer. This ties final output
    // order and escaping to the same token enumeration used by sizing.
    crate::tokenizer::visit_tokens_in(text, context, |term| {
        query.append(|output| {
            if !first {
                output.write_str(" OR ")?;
            }
            output.write_char('"')?;
            // Double a quote within the FTS term. Copy every other Unicode scalar
            // as emitted by the tokenizer.
            for ch in term.chars() {
                output.write_char(ch)?;
                if ch == '"' {
                    output.write_char('"')?;
                }
            }
            output.write_char('"')
        })?;
        first = false;
        Ok(())
    })
}

// Probe FTS5 on the configured SQLite executable using an in-memory database.
// Successful process exit is the capability signal; diagnostics are discarded for
// this check.
fn sqlite3_fts5_available() -> bool {
    Command::new(sqlite_cli())
        .arg("-batch")
        .arg(":memory:")
        .arg("CREATE VIRTUAL TABLE t USING fts5(x);")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        // A missing executable and an executable without FTS5 both yield an
        // unavailable result. The caller decides how to report a missing
        // capability.
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

// Check the vector extension by calling its version function on the same
// executable. The configured binary must expose the extension to the command used
// by the runner.
fn sqlite3_sqlite_vec_available() -> bool {
    sqlite3_script(Path::new(":memory:"), "SELECT vec_version();").is_ok()
}

// Configure the SQLite child for fail-fast batch execution and predictable text
// output. Pass the database path as its own native argument.
fn sqlite3_command(db_path: &Path) -> Command {
    let mut command = Command::new(sqlite_cli());
    command
        .arg("-bail")
        .arg("-batch")
        .arg("-noheader")
        .arg("-separator")
        // Select a fixed tab separator and suppress headers for scripts that read
        // small scalar outputs.
        .arg("\t")
        .arg(db_path);
    command
}

// Run a short SQL script through the SQLite wrapper. Capability code uses this
// path to probe extension support with the same execution setup.
fn sqlite3_script(db_path: &Path, sql: &str) -> Result<String> {
    sqlite::script(db_path, sql)
}

// Allow the SQLite executable to be overridden for builds carrying required
// extensions. Otherwise resolve the standard sqlite3 command from the environment.
fn sqlite_cli() -> std::ffi::OsString {
    std::env::var_os("COFFERDB_SQLITE3").unwrap_or_else(|| "sqlite3".into())
}

// Convert the configured executable into a nonblank UTF-8 report label. Reject NUL
// because the path also needs to be representable as a process argument.
fn sqlite_cli_label(command: &std::ffi::OsStr) -> Result<&str> {
    command
        .to_str()
        .filter(|value| !value.trim().is_empty() && !value.contains('\0'))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::CliUsage,
                "SQLite executable must be nonempty UTF-8 without NUL bytes",
            )
        })
}

#[cfg(test)]
fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

// Prefer the platform CPU brand string when available. Linux reports a model from
// procfs; absence of both leaves an explicit unknown label.
fn hardware_cpu_model() -> String {
    command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(linux_cpu_model)
        .unwrap_or_else(|| "unknown CPU".into())
}

// Use the parallelism available to this process. Fall back to one when the host
// cannot report it, avoiding a zero-core value in the result.
fn hardware_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
}

// Read total memory bytes from the platform probe or Linux procfs. Failure of both
// leaves the measurement unavailable.
fn hardware_memory_gb() -> Option<f64> {
    command_stdout("sysctl", &["-n", "hw.memsize"])
        .and_then(|value| value.parse::<u64>().ok())
        .map(bytes_to_gb)
        .or_else(linux_memory_gb)
}

// Use the platform marketing name and version when available together. Linux
// distribution values and platform constants provide fallback labels.
fn hardware_os_name_version() -> (String, String) {
    // Use both successful platform fields together. If either command fails, try
    // Linux metadata or the platform-constant fallback.
    if let Some(name) = command_stdout("sw_vers", &["-productName"]) {
        if let Some(version) = command_stdout("sw_vers", &["-productVersion"]) {
            return (name, version);
        }
    }
    // Prefer the Linux machine-readable VERSION_ID. A descriptive version or
    // architecture name supplies the fallback when that field is unavailable.
    if let Some(name) = linux_os_release_field("NAME") {
        let version = linux_os_release_field("VERSION_ID")
            .or_else(|| linux_os_release_field("VERSION"))
            .unwrap_or_else(|| std::env::consts::ARCH.into());
        return (name, version);
    }
    (std::env::consts::OS.into(), std::env::consts::ARCH.into())
}

// Use only successful, nonblank UTF-8 command output. Failed probes return None so
// hardware-report generation can try another source.
fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    // Ignore blank command responses. A successful process alone does not provide
    // a usable hardware label.
    (!value.is_empty()).then(|| value.to_owned())
}

// Find the first procfs model label and strip its key and surrounding whitespace.
// An empty or missing label lets the caller choose a fallback.
fn linux_cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("model name"))
        .and_then(|value| {
            value
                .split_once(':')
                .map(|(_, model)| model.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
}

// Read MemTotal and convert the kernel's KiB count to bytes before applying the
// report's binary memory unit.
fn linux_memory_gb() -> Option<f64> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| bytes_to_gb(kb * 1024))
}

// Read the first matching os-release key and strip enclosing double quotes. Only
// the naming fields needed for the fallback are consumed.
fn linux_os_release_field(field: &str) -> Option<String> {
    fs::read_to_string("/etc/os-release")
        .ok()?
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key == field).then(|| value.trim_matches('"').to_owned())
        })
        .filter(|value| !value.is_empty())
}

// Convert bytes to binary gigabytes and round to one decimal place. Keep the
// existing field's reporting convention consistent across host sources.
fn bytes_to_gb(bytes: u64) -> f64 {
    ((bytes as f64 / 1_073_741_824.0) * 10.0).round() / 10.0
}

// Emit the selected profile's defaults separately from per-query resolved limits.
// A larger top_k can increase the actual retrieval windows without changing these
// preset values.
fn profile_params_json_in(
    profile: ResourceProfile,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        (
            "bm25_candidates",
            JsonValue::Number(profile.params().bm25_candidates as f64),
        ),
        (
            "vector_candidates",
            JsonValue::Number(profile.params().vector_candidates as f64),
        ),
        (
            "fusion_candidates",
            JsonValue::Number(profile.params().fusion_candidates as f64),
        ),
        // Record cache allowance apart from the three candidate limits. The cache
        // setting describes retained resources rather than which candidates will
        // be scored.
        (
            "cache_budget_mb",
            JsonValue::Number(profile.params().cache_budget_mb as f64),
        ),
    ])
}

#[cfg(test)]
fn benchmark_engine_params_json() -> JsonValue {
    benchmark_engine_params_json_in(&JsonBuilder::host())
        .expect("host DTO construction is infallible")
}

// Publish the native engine recipe defaults independently of the selected dataset
// and collection metadata.
fn benchmark_engine_params_json_in(builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    builder.object([
        (
            // Record the default text-index recipe and BM25 constants. The
            // tokenizer actually selected for the workload appears in the dataset,
            // import and query contexts.
            "text_index",
            builder.object([
                ("index_kind", builder.string("text_bm25")?),
                (
                    "tokenizer",
                    crate::TokenizerConfig::default().build_json(builder)?,
                ),
                (
                    // Report the BM25 parameters from the implementation
                    // constants. This keeps the artifact aligned when the
                    // product's native scoring recipe changes.
                    "bm25",
                    builder.object([
                        ("k1", JsonValue::Number(crate::index::text::BM25_K1)),
                        ("b", JsonValue::Number(crate::index::text::BM25_B)),
                    ])?,
                ),
            ])?,
        ),
        (
            // Describe exact binary32 cosine retrieval with unnormalized stored
            // coordinates. The reported normalization method applies to the
            // similarity score.
            "vector_index",
            builder.object([
                ("index_kind", builder.string("vector_exact_f32")?),
                ("distance_metric", builder.string("cosine")?),
                ("vectors_normalized", JsonValue::Bool(false)),
                (
                    // Record score normalization separately from stored vector
                    // normalization. This recipe assumes stored values are not
                    // normalized but scales cosine scores for fusion.
                    "normalization_method",
                    builder.string("cosine_to_unit_interval")?,
                ),
            ])?,
        ),
        (
            // Record weighted-sum fusion and an engine-local score scale.
            // Identical ranking metrics do not imply that the underlying scores
            // share a scale.
            "fusion",
            builder.object([
                ("algorithm", builder.string("weighted_sum")?),
                ("score_scale", builder.string("engine_local")?),
            ])?,
        ),
    ])
}

#[cfg(test)]
mod tests;
