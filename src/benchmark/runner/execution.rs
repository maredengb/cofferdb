//! Record a native benchmark attempt as an ordered JSONL history. The plan,
//! source-derived expectations and actual search events let a reader distinguish a
//! completed measurement from a file left by interruption.
use super::*;
use std::io::BufWriter;

// Identifies the acceptance rules recorded in the protocol event. A change to how
// expectations, ordering or score tolerances are interpreted must remain
// distinguishable in historical journals.
const RULES_VERSION: &str = "cofferdb-builtin-source-acceptance-v4";

pub(super) struct ExecutionConfig<'a> {
    // Database path recorded in the protocol and used to locate the journal when no
    // result path is set.
    pub database: &'a str,
    // Optional result path, whose parent directory also hosts the journal.
    pub result_file: Option<&'a str>,
    // Collection named in the journal's reproducible run configuration.
    pub collection: &'a str,
    // Resource profile recorded in the run protocol.
    pub profile: ResourceProfile,
    // Requested database durability recorded with the execution plan.
    pub durability: Durability,
}

// One journal owns the cursor through a mode-major, cold-then-warm sample plan.
// Search completion freezes cost and latency; acceptance moves the cursor only
// after the measured result has been checked.
pub(super) struct ExecutionJournal<'a> {
    // Exclusively created JSONL file retained even if this attempt stops early.
    file: File,
    // Path to the retained execution evidence, attached to errors and published
    // results.
    path: PathBuf,
    // Fixed dataset that supplies query IDs and the sample ordering.
    dataset: &'a BenchmarkDataset,
    // Planned search modes in execution order for both cold and warm groups.
    modes: &'a [BenchmarkQueryMode],
    // Operation context for constructing and writing budgeted journal records.
    context: JsonOperationContext,
    // Start of the continuous benchmark clock, shared with the outer run.
    started: Instant,
    // Zero-based index of the next planned sample to complete or mark as failed.
    next: usize,
    // Start time for the currently active sample, used when search has no frozen
    // duration.
    active: Option<Instant>,
    // Cost trace captured at search completion or failure for the active sample.
    active_cost: Option<crate::search_cost::SearchCostTrace>,
    // Frozen duration of the active search call, excluding later acceptance checks.
    active_search_ms: Option<f64>,
    // Whether the active search succeeded with a valid cost trace, required before
    // acceptance can complete.
    active_search_successful: Option<bool>,
    // Total number of planned samples across modes, cold and warm groups, and
    // queries.
    planned: usize,
    // Latched after a journal write fails, preventing later events from implying a
    // complete history.
    write_failed: bool,
    // Successful query latencies retained for the current mode and cache group.
    latencies: Vec<f64>,
    // Reservation for that latency array until its group statistics have been
    // emitted.
    latency_work: JsonReservation,
}

impl<'a> ExecutionJournal<'a> {
    pub(super) fn new(
        dataset: &'a BenchmarkDataset,
        modes: &'a [BenchmarkQueryMode],
        config: ExecutionConfig<'_>,
        context: &JsonOperationContext,
        started: Instant,
    ) -> Result<Self> {
        // Place the evidence beside the requested result, falling back to the
        // database directory for an in-memory result. A bare filename has an empty
        // parent and uses the current directory.
        let parent = Path::new(config.result_file.unwrap_or(config.database))
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()?;
        // Count samples before creating a file. Each query runs once per selected
        // mode in each of two cache states, and the cursor cannot represent a plan
        // whose multiplication overflows usize.
        let planned = dataset
            .queries
            .len()
            .checked_mul(modes.len())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark sample count overflow",
                )
            })?;
        // Use an exclusively created filename for this attempt's JSONL evidence.
        // The optional result destination follows a separate publication path and
        // must not be replaced by creating the journal.
        let (file, guard) = create_temporary_file_in(&parent, "benchmark-execution", "jsonl")?;
        let path = guard.path().to_path_buf();
        // Publish a retained sidecar under its exclusive name before execution, so
        // interrupted runs survive without overwriting inputs or prior attempts.
        guard.persist(&path)?;
        // Initialize the cursor with no active search or accumulated latency
        // samples. A journal that stopped during plan creation cannot be mistaken
        // for one that has started executing its first query.
        let mut journal = Self {
            file,
            path,
            dataset,
            modes,
            context: context.clone(),
            started,
            next: 0,
            active: None,
            active_cost: None,
            active_search_ms: None,
            active_search_successful: None,
            planned,
            write_failed: false,
            latencies: Vec::new(),
            latency_work: context.reservation(),
        };
        // Once the file is retained, setup can fail after writing a subset of the
        // protocol or plan. Attach its basename so the caller can locate the
        // record.
        let result = (|| {
            let mut work = context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
            // Write the run conditions once, before any observations. Later sample
            // events refer to the declared order and dataset instead of repeating
            // the whole protocol for every query.
            let header = builder.object([
            ("event", builder.string("protocol")?),
            ("rules_version", builder.string(RULES_VERSION)?),
                ("git_commit", benchmark_git_commit_json_in(&builder)?),
                ("hardware", benchmark_hardware_json_in(&builder)?),
            ("dataset", dataset.manifest.build_result_dataset_json(&builder)?),
            ("database_path", builder.string(config.database)?),
            ("result_file", builder.optional_str(config.result_file)?),
            ("collection", builder.string(config.collection)?),
            ("profile", builder.string(config.profile.as_json_str())?),
            ("durability", builder.string(config.durability.as_str())?),
            ("index_build_mode", builder.string("inline_full_checkpoint")?),
            // The planned count includes both measured passes, with no separate
            // unreported warm-up group. A reader can therefore compare the final
            // sample count directly against the plan.
            ("planned_samples", JsonValue::Number(planned as f64)),
            ("order", builder.string("text_bm25,vector_exact,hybrid when available; cold then warm; artifact query order")?),
            ("repetitions_per_group", JsonValue::Number(1.0)),
            ("warmup_rounds", JsonValue::Number(0.0)),
            ("warmup_reason", builder.string("first pass is a measured cold-handle group; the measured warm group repeats it")?),
            ("cold_reset", builder.string("reopen handle; no OS page-cache eviction")?),
            ("concurrency", JsonValue::Number(1.0)),
            // Null deadlines record the absence of an internal timeout. A journal
            // ending in the middle of its plan still needs a terminal outcome to
            // qualify the run.
            ("query_timeout_ms", JsonValue::Null),
            ("run_timeout_ms", JsonValue::Null),
            ("timeout_policy", builder.string("no internal deadline; externally interrupted journals without a terminal event are incomplete, never successful")?),
            ("completion_rule", builder.string("CLI exit zero plus a durable final run_completed event linked to the published result; any failure or missing terminal event disqualifies the run")?),
            ("retry_policy", builder.string("abort on first error; no retries; unexecuted planned samples remain explicit")?),
            ("latency_scope", builder.string("continuous search call including preparation, recall, score completion, normalization/fusion and result materialization; excludes reference generation, verification and journal IO; failed samples use frozen search time when a search finished, otherwise started-event elapsed")?),
            ("search_cost_version", builder.string(crate::search_cost::VERSION)?),
            ("search_cost_scope", builder.string("costs are cumulative snapshots per sequence/attempt: use search_completed/search_failed once; completed/failed repeat the same snapshot solely to bind acceptance status and must not be added again; fixed per-search diagnostic counters; failed/reclaimed attempts remain counted; stage times are inclusive diagnostics, not a substitute for or sum defining end-to-end latency; preparation includes optional cache loads across collections; no engine candidate staging reads; source-reference/import/journal IO excluded")?),
            ("search_cost_work_rules", builder.string("text visits are consumer document/posting visits, not all validator CPU or binary-search comparisons; vector checks count inspected vectors before eligibility and distances count scored components; reuse counts each recalled branch score consumed directly in final candidate construction, including failed fusion attempts; logical bytes count typed payload or index-extent requests, application bytes count successful Read/read_at returns including partial failures, retries and checksum drains; physical disk IO unknown; unset candidate counts are null, never zero")?),
                ("percentile_method", builder.string("nearest rank ceil(p*n)-1 for P50/P95, unrounded successful query latencies, separately per mode/cache group; incomplete-group percentiles are diagnostic only")?),
            ("import_time_scope", builder.string("continuous first input read through all cold/warm queries and source verification; includes reference generation and journal IO in that interval")?),
            ("import_phase_scope", builder.string("commit includes commit preparation, old-index prefetch, critical append/durability and Catalog application; checkpoint includes postcommit index preparation, resident construction and suffix persistence/retry; cycles count logical calls, ready cycles count query-usable persistent suffixes after those calls; disjoint diagnostics, not the end-to-end total; omitted when import fails")?),
            // The scoring reference comes from input artifacts. Qrels grade the
            // relevance of the measured ranks separately; they are not the
            // reference ranked list checked for each timed search.
            ("reference", builder.string("hash-verified raw rows; full-collection streaming BM25 statistics; only the filtered recall union receives supplemental scores before normalization/fusion limits; shared vector metric and fusion primitives; imported IDs only for ties; qrels are quality labels, not a 100-percent-recall assertion")?),
            ("score_tolerance", JsonValue::Number(1e-12)),
            ("score_tolerance_rule", builder.string("absolute error <= tolerance * max(1, abs(expected)); exact ranked external IDs and result count")?),
            ("journal_durability", builder.string("flush each event; sync plan and terminal events")?),
        ])?;
            // Flush the protocol record before enumerating planned samples. Only
            // after all planned events are written is the combined header and plan
            // synchronized.
            journal.write(header, work)?;
            // Record every planned sample and sync the plan before starting the
            // workload. If the process stops later, the journal can distinguish
            // missing samples from a run that never planned them.
            for index in 0..planned {
                journal.sample("planned", index, None, None, None)?;
            }
            journal.file.sync_data().map_err(Error::from)
        })();
        // Surface the setup error with the journal's basename and rules version. A
        // retained file may contain a valid prefix even when no workload began.
        result.map_err(|error| journal.locate(error))?;
        Ok(journal)
    }

    // This record is emitted after the measured import succeeds. It captures
    // import-specific counters while the rest of the run, including reference
    // building and search, can still fail.
    pub(super) fn import_completed(
        &mut self,
        batches: &crate::db::ImportBatchMetrics,
    ) -> Result<()> {
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string("import_completed")?),
            (
                "committed_batches",
                JsonValue::Number(batches.committed_batches as f64),
            ),
            // A checkpoint cycle counts an attempt at the logical operation. Some
            // cycles end without a usable persistent index, so readiness has a
            // separate count and a final state.
            (
                "checkpoint_cycles",
                JsonValue::Number(batches.checkpoint_cycles as f64),
            ),
            (
                "checkpoint_ready_cycles",
                JsonValue::Number(batches.checkpoint_ready_cycles as f64),
            ),
            (
                "final_checkpoint_ready",
                JsonValue::Bool(batches.final_checkpoint_ready),
            ),
            // Report the largest committed chunk batch. A final short batch or a
            // profile's configured maximum alone would misstate the batch size
            // actually observed.
            (
                "max_batch_chunks",
                JsonValue::Number(batches.max_batch_chunks as f64),
            ),
            // The importer's separate commit and checkpoint intervals describe
            // where import time was spent. The elapsed field also includes earlier
            // run setup and is sampled from the continuous run clock.
            (
                "commit_time_ms",
                JsonValue::Number(batches.commit_time.as_secs_f64() * 1000.0),
            ),
            (
                "checkpoint_time_ms",
                JsonValue::Number(batches.checkpoint_time.as_secs_f64() * 1000.0),
            ),
            (
                "elapsed_ms",
                JsonValue::Number(self.started.elapsed().as_secs_f64() * 1000.0),
            ),
        ])?;
        self.write(value, work)?;
        // Synchronize the import record before the caller tries to generate source
        // expectations. If reference preparation later fails, this boundary still
        // records which import work completed.
        self.file.sync_data()?;
        Ok(())
    }

    // Persist the independent expected rankings before cold or warm measurements.
    // Both cache states for a mode share this same per-query reference.
    pub(super) fn expectations(&mut self, acceptance: &acceptance::Acceptance) -> Result<()> {
        for (mode, _) in self.modes.iter().enumerate() {
            for query in 0..self.dataset.queries.len() {
                // Scope the reservation to one expected event. A large query plan
                // need only charge the current JSON record in addition to the
                // already retained reference rankings.
                let mut work = self.context.reservation();
                let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
                let value = builder.object([
                    ("event", builder.string("expected")?),
                    ("query_mode", builder.string(self.modes[mode].as_str())?),
                    (
                        "query_id",
                        builder.string(&self.dataset.queries[query].query_id)?,
                    ),
                    // Read the ranked identity sequence and its scores from the
                    // same mode/query cell. The parallel arrays depend on
                    // preserving order across both traversals of expected_hits.
                    (
                        "ranked_chunk_ids",
                        builder.array(
                            acceptance
                                .expected_hits(mode, query)
                                .map(|(id, _)| builder.string(id)),
                        )?,
                    ),
                    (
                        "scores",
                        builder.array(
                            acceptance
                                .expected_hits(mode, query)
                                .map(|(_, score)| Ok(JsonValue::Number(score))),
                        )?,
                    ),
                ])?;
                self.write(value, work)?;
            }
        }
        // Synchronize after the complete expectation grid. An earlier write error
        // exits with a prefix on disk and prevents the caller from starting
        // measured search with incomplete reference evidence.
        self.file.sync_data()?;
        Ok(())
    }

    // Admit a sample only at the fixed position reserved for its mode, cache state
    // and workload query. Repeating or skipping a position would make later
    // evidence ambiguous.
    pub(super) fn begin(&mut self, mode: usize, cache: &str, query: usize) -> Result<()> {
        let cache = match cache {
            "cold" => 0,
            "warm" => 1,
            _ => return Err(invalid_order()),
        };
        // Flatten in mode-major order, then cold before warm, then query-file
        // order. The journal decodes this same sequence number when it writes a
        // sample's labels.
        let expected = (mode * 2 + cache) * self.dataset.queries.len() + query;
        if self.active.is_some() || self.next != expected || self.next >= self.planned {
            return Err(invalid_order());
        }
        // Clear state belonging to the preceding search before starting this slot.
        // An attempt that fails before search completion must not inherit another
        // query's timings or counters.
        self.active_cost = None;
        self.active_search_ms = None;
        self.active_search_successful = None;
        // Write the start event before marking the slot active in memory. A
        // failure while writing that event leaves no successfully recorded start
        // to pair with a later failed-sample event.
        self.sample("started", self.next, None, None, None)?;
        // This clock supplies a fallback duration only if search never froze one.
        // The search timer supplied to search_finished is measured by the runner
        // around the engine call.
        self.active = Some(Instant::now());
        Ok(())
    }

    // Record the engine's return separately from rank acceptance. A successful
    // call can still fail validation of its counters or its ranked results.
    pub(super) fn search_finished(
        &mut self,
        trace: crate::search_cost::SearchCostTrace,
        elapsed_ms: f64,
        successful: bool,
    ) -> Result<()> {
        // Permit one trace for an active sample. The elapsed value goes into JSON
        // and group metrics, so negative and nonfinite durations cannot be
        // accepted as search evidence.
        if self.active.is_none()
            || self.active_cost.is_some()
            || !elapsed_ms.is_finite()
            || elapsed_ms < 0.0
        {
            return Err(invalid_order());
        }
        // Freeze the caller's duration before emitting an event. A later
        // verification or publication error must not extend that already observed
        // search interval.
        self.active_search_ms = Some(elapsed_ms);
        // The trace's counters may overflow even when the engine returned a
        // result. Require both success conditions before complete_query is allowed
        // to advance.
        let valid = trace.valid();
        self.active_search_successful = Some(successful && valid);
        self.active_cost = Some(trace);
        // Choose the event name from the engine outcome. A completed search record
        // can therefore precede a failure for invalid counters; it is not the
        // query acceptance event.
        self.sample(
            if successful {
                "search_completed"
            } else {
                "search_failed"
            },
            self.next,
            Some(elapsed_ms),
            None,
            None,
        )?;
        // Surface trace overflow after attempting to write the observed search
        // outcome. A successful engine call can still leave no valid performance
        // sample.
        if !valid {
            return Err(Error::new(
                ErrorCode::InvalidSearchParams,
                "benchmark search cost counters overflowed; the measurement is invalid",
            ));
        }
        Ok(())
    }

    // The caller verifies expected ranks before reaching this step. Only a
    // verified query with a successful, valid search event can contribute a
    // latency to the group's percentiles.
    pub(super) fn complete_query(
        &mut self,
        latency_ms: f64,
        results: &SearchResults,
    ) -> Result<()> {
        if self.active.is_none()
            || self.active_cost.is_none()
            // Use the same duration that was recorded at search completion.
            // Recomputing it after reference comparisons would silently change the
            // latency from one event to the next.
            || self.active_search_ms != Some(latency_ms)
            || self.active_search_successful != Some(true)
        {
            return Err(invalid_order());
        }
        // Reserve the latency slot before recording completion. A budget failure
        // must not leave a completed event whose observation cannot be added to
        // the group.
        self.latency_work
            .reserve_vec(&mut self.latencies, 1, JsonInputKind::BenchmarkArtifact)?;
        // The final sample can now name the index source from accepted results.
        // Earlier search events carry the trace but do not establish rank
        // acceptance.
        self.sample(
            "completed",
            self.next,
            Some(latency_ms),
            Some(&results.query_context.index_source),
            None,
        )?;
        // Advance the sequence only after the completion record has been written.
        // Skipped or failed queries are counted separately when fail closes the
        // journal.
        self.latencies.push(latency_ms);
        self.next += 1;
        self.active = None;
        self.active_cost = None;
        self.active_search_ms = None;
        self.active_search_successful = None;
        // A group spans the entire validated query list for one mode and cache
        // state. The cursor now points just past its last query, hence the
        // subtraction when identifying the group to summarize.
        if self.next % self.dataset.queries.len() == 0 {
            let summary = self.group_summary(self.next / self.dataset.queries.len() - 1, 0, 0);
            // Drop both latency storage and its reservation at the group boundary,
            // including when writing the summary failed. A subsequent group must
            // not reuse the observations.
            self.latencies = Vec::new();
            self.latency_work = self.context.reservation();
            summary?;
        }
        Ok(())
    }

    pub(super) fn publish_run(
        &mut self,
        mut run: BenchmarkRun,
        output: Option<output::BenchmarkOutput>,
    ) -> Result<BenchmarkRun> {
        // A journal failure must not replace a previous result. Publication is
        // last; if it fails, the final journal event disqualifies this attempt.
        // Synchronize run completion before publishing to the requested output
        // path. If publication then fails, the failure path can append run_failed
        // after the completion record.
        let result = self.complete_run(&mut run).and_then(|()| {
            if let Some(output) = output {
                output.write(&run.value, &run.context, &mut run.work)?;
            }
            Ok(())
        });
        match result {
            Ok(()) => Ok(run),
            // Release the result's retained allocations before writing failure
            // evidence, so reporting the failure does not need to compete with the
            // full report for the same memory budget.
            Err(error) => {
                drop(run);
                Err(self.fail(error))
            }
        }
    }

    fn complete_run(&mut self, run: &mut BenchmarkRun) -> Result<()> {
        // Do not identify a result as complete until every planned sample has
        // advanced the cursor and no active search is left unresolved.
        if self.active.is_some() || self.next != self.planned {
            return Err(invalid_order());
        }
        // Hash the pretty-printed report on a sink, using the same output schema
        // as publication. The journal records the exact serialization without
        // requiring an extra copy of the result file.
        let mut digest = crate::hash::Sha256Writer::new(std::io::sink());
        run.value.write_pretty_retained_in(
            &mut digest,
            JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
            &run.context,
            &mut run.work,
        )?;
        let digest = digest.finish()?;
        // The journal event has its own reservation. The result remains live with
        // its original accounting for any subsequent file publication.
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string("run_completed")?),
            ("result_pretty_json_sha256", builder.string(&digest)?),
            (
                // The complete-run clock starts before dataset setup and can
                // include the SQLite baseline. It differs from the import_time_ms
                // value copied from the finished result below.
                "elapsed_ms",
                JsonValue::Number(self.started.elapsed().as_secs_f64() * 1000.0),
            ),
            (
                "import_time_ms",
                JsonValue::Number(
                    run.value
                        .get("metrics")
                        .unwrap()
                        .get("import_time_ms")
                        .unwrap()
                        .as_f64()
                        .unwrap(),
                ),
            ),
            // Bind the digest to the number of completed samples. This count
            // summarizes the full mode/cache schedule checked against planned at
            // the entrance to this function.
            ("completed_samples", JsonValue::Number(self.next as f64)),
        ])?;
        self.write(value, work)?;
        // Sync the journal's completion record before returning to the publisher.
        // The actual output write, when requested, still has its own failure path.
        self.file.sync_data()?;
        Ok(())
    }

    // Close an interrupted sample plan with best-effort diagnostic records. The
    // returned error must still identify what stopped the benchmark even if the
    // journal cannot be completed.
    pub(super) fn fail(&mut self, error: Error) -> Error {
        let record = (|| {
            // Locate the first incomplete group from the next sequence number.
            // When every scheduled query has terminated, there is no incomplete
            // group to summarize.
            let unfinished_group =
                (self.next < self.planned).then_some(self.next / self.dataset.queries.len().max(1));
            // Only a sample whose start was recorded can be marked failed. The
            // remaining planned slots are recorded as not_run so the journal
            // distinguishes an interrupted measurement from a query that was never
            // attempted.
            let failed = usize::from(self.active.is_some());
            // If a sample started, write its failure record at the same sequence
            // number before marking later slots unattempted. Use the search
            // duration when it is known.
            if let Some(started) = self.active {
                self.sample(
                    "failed",
                    self.next,
                    Some(
                        self.active_search_ms
                            .unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0),
                    ),
                    None,
                    Some(&error),
                )?;
                // Advance and clear active state only after the failure record is
                // written. This leaves the later planned slots available for
                // explicit not_run entries.
                self.next += 1;
                self.active = None;
                self.active_cost = None;
                self.active_search_ms = None;
                self.active_search_successful = None;
            }
            // Emit one terminal record for every later slot without an elapsed
            // time. Their query and group identity come from the same fixed plan
            // as successful records.
            for index in self.next..self.planned {
                self.sample("not_run", index, None, None, Some(&error))?;
            }
            // Summarize the incomplete group from its successful samples, the
            // active failure if any, and its remaining unattempted slots.
            if let Some(group) = unfinished_group {
                self.group_summary(
                    group,
                    failed,
                    self.dataset.queries.len() - self.latencies.len() - failed,
                )?;
                // Discard the current group's sample buffer before summarizing
                // later groups. Those groups have no observations because
                // execution never reached them.
                self.latencies = Vec::new();
                self.latency_work = self.context.reservation();
                // Emit a summary even for a group that was never entered. Its
                // planned count is still the workload size and every slot remains
                // not_run.
                for remaining in group + 1..self.modes.len() * 2 {
                    self.group_summary(remaining, 0, self.dataset.queries.len())?;
                }
            }
            // Build the overall failure event under a separate reservation. Its
            // code and reason come from the original error, with details
            // serialized alongside.
            let mut work = self.context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
            let value = builder.object([
                ("event", builder.string("run_failed")?),
                (
                    "elapsed_ms",
                    JsonValue::Number(self.started.elapsed().as_secs_f64() * 1000.0),
                ),
                ("error_code", builder.string(error.code_str())?),
                ("reason", builder.string(error.message())?),
                (
                    "error_detail",
                    builder.array(error.detail().iter().map(|(key, value)| {
                        builder.object([
                            ("field", builder.string(key)?),
                            ("value", builder.string(value)?),
                        ])
                    }))?,
                ),
            ])?;
            // Append the final failure record and request a data sync. A failure
            // to do either becomes context on the original benchmark error.
            self.write(value, work)?;
            self.file.sync_data()?;
            Ok::<_, Error>(())
        })();
        // Preserve the error that stopped the run. Failure to write or sync the
        // final journal entry is additional context for a reader reconciling the
        // incomplete journal.
        // Identify the journal by its published basename and rules version. This
        // lets the caller correlate the original run failure with any recorded
        // evidence.
        let error = self.locate(error);
        match record {
            Ok(()) => error,
            Err(record_error) => {
                error.with_detail("execution_record_error", record_error.to_string())
            }
        }
    }

    // Add the stable journal identity to the benchmark error without substituting
    // its internal filesystem path.
    fn locate(&self, error: Error) -> Error {
        // The sidecar is beside the supplied result (or database). Never expose
        // its internally canonicalized directory or a private staging location.
        error
            .with_detail(
                "benchmark_execution_record",
                self.path.file_name().unwrap().to_string_lossy(),
            )
            .with_detail("benchmark_rules_version", RULES_VERSION)
    }

    // Compute group statistics from completed sample latencies. Failure and
    // unattempted counts describe the rest of the fixed plan rather than
    // contributing artificial durations.
    fn group_summary(&mut self, group: usize, failed: usize, not_run: usize) -> Result<()> {
        // Sort the available latencies once. Both percentiles use the same order
        // and the nearest-rank position within it.
        self.latencies.sort_by(f64::total_cmp);
        let count = self.latencies.len();
        // A group with no successes has no percentile estimate. Null keeps this
        // distinct from a group whose measured durations rounded to zero.
        let percentile = |p: f64| {
            if count == 0 {
                JsonValue::Null
            } else {
                JsonValue::Number(self.latencies[(p * count as f64).ceil() as usize - 1])
            }
        };
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string("group_summary")?),
            // Derive mode and cache state by the group index. Every mode occupies
            // two consecutive groups, cold first and warm second.
            (
                "query_mode",
                builder.string(self.modes[group / 2].as_str())?,
            ),
            (
                "cache_state",
                builder.string(if group % 2 == 0 { "cold" } else { "warm" })?,
            ),
            (
                // Record the workload size as planned even if the group never
                // started. Completed, failed and not_run are separate observations
                // from that plan.
                "planned",
                JsonValue::Number(self.dataset.queries.len() as f64),
            ),
            ("completed", JsonValue::Number(count as f64)),
            ("failed", JsonValue::Number(failed as f64)),
            ("not_run", JsonValue::Number(not_run as f64)),
            // This event model records failed and unattempted samples separately.
            // The timed_out field stays zero because there is no independent
            // timeout event in this summary path.
            ("timed_out", JsonValue::Number(0.0)),
            ("p50_ms", percentile(0.5)),
            ("p95_ms", percentile(0.95)),
            (
                // Label percentiles from incomplete groups as diagnostic. They use
                // only successful observations and must not be treated as
                // comparable to a complete group.
                "statistics_scope",
                builder.string(if failed == 0 && not_run == 0 {
                    "complete_group"
                } else {
                    "diagnostic_successful_samples_only"
                })?,
            ),
        ])?;
        self.write(value, work)
    }

    // Write one lifecycle event for a planned sample. Every event for the same
    // slot carries the same sequence, query identity, mode and cache state.
    fn sample(
        &mut self,
        event: &str,
        index: usize,
        latency: Option<f64>,
        index_source: Option<&str>,
        error: Option<&Error>,
    ) -> Result<()> {
        // Map the flat sequence to a group and position within the workload. The
        // plan repeats the same query order in each cache-state group.
        let query_count = self.dataset.queries.len();
        let query = &self.dataset.queries[index % query_count];
        let group = index / query_count;
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string(event)?),
            ("sequence", JsonValue::Number(index as f64)),
            ("query_id", builder.string(&query.query_id)?),
            (
                "query_mode",
                builder.string(self.modes[group / 2].as_str())?,
            ),
            (
                "cache_state",
                builder.string(if group % 2 == 0 { "cold" } else { "warm" })?,
            ),
            // The plan makes a single pass per group and does not retry a sample.
            // Keep the round and attempt fields fixed for these lifecycle events.
            ("round", JsonValue::Number(1.0)),
            ("attempt", JsonValue::Number(1.0)),
            (
                // Use an explicit null when there was no measured interval. In
                // particular a not_run event cannot supply a latency sample.
                "elapsed_ms",
                latency.map(JsonValue::Number).unwrap_or(JsonValue::Null),
            ),
            ("index_source", builder.optional_str(index_source)?),
            (
                // Attach the active search trace only to event types where search
                // or sample outcome can be observed. Planned slots and their start
                // events have no trace evidence here.
                "search_cost",
                if matches!(
                    event,
                    "search_completed" | "search_failed" | "completed" | "failed"
                ) {
                    self.active_cost
                        .as_ref()
                        .map(|trace| trace.json(&builder))
                        .transpose()?
                        .unwrap_or(JsonValue::Null)
                } else {
                    JsonValue::Null
                },
            ),
            (
                // Keep error code and reason optional for lifecycle events. A
                // failed or unattempted slot can carry the error that caused
                // execution to stop.
                "error_code",
                builder.optional_str(error.map(Error::code_str))?,
            ),
            ("reason", builder.optional_str(error.map(Error::message))?),
        ])?;
        self.write(value, work)
    }

    // Write a single framed JSONL event. Once the stream may have an incomplete
    // event, refuse all later writes to preserve its last known boundary.
    fn write(&mut self, value: JsonValue, mut retained: JsonReservation) -> Result<()> {
        if self.write_failed {
            return Err(Error::new(
                ErrorCode::IoError,
                "execution journal contains an incomplete event; further writes are disabled",
            ));
        }
        // Admit the output buffer before wrapping the file. A budget refusal at
        // this point happens before emitting any part of the event.
        let mut buffer_work = self.context.reservation();
        buffer_work.grow(8192, JsonInputKind::BenchmarkArtifact)?;
        let mut output = BufWriter::with_capacity(8192, &mut self.file);
        // Serialize with the caller-owned retained value reservation. Keep the
        // separate I/O buffer charged only for this write.
        let result = value
            .write_value_retained_in(
                &mut output,
                JsonInputKind::BenchmarkArtifact,
                &self.context,
                &mut retained,
                false,
            )
            // Append the newline after the complete value and then drain the
            // buffer. Either failure can leave a partially emitted event.
            .and_then(|()| {
                output.write_all(b"\n")?;
                output.flush().map_err(Error::from)
            });
        // A partial event remains incomplete; never retry buffered bytes during Drop.
        let _ = output.into_parts();
        // Latch the result of the write and flush. The buffer has already been
        // dismantled, so leaving the function cannot retry its pending bytes.
        self.write_failed = result.is_err();
        result
    }
}

// Return an explicit ordering failure when observed execution does not match its
// planned sample sequence. Journal consumers depend on this invariant to reconcile
// events.
fn invalid_order() -> Error {
    Error::new(
        ErrorCode::RecoveryError,
        "benchmark execution differs from its fixed sample plan",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::text_only_benchmark_dataset;
    use super::*;

    #[test]
    fn journal_and_publication_failures_do_not_replace_previous_results() {
        let dir = text_only_benchmark_dataset("journal-publication-order");
        let context = JsonOperationContext::standalone();
        let (dataset, _work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
        let modes = [BenchmarkQueryMode::TextBm25];
        let database = dir.join("data.coffer");
        for stage in 0..3 {
            let result_path = dir.join(format!("result-{stage}.json"));
            fs::write(&result_path, b"prior result").unwrap();
            let output = output::BenchmarkOutput::new(&result_path, &database, Vec::new()).unwrap();
            let mut journal = ExecutionJournal::new(
                &dataset,
                &modes,
                ExecutionConfig {
                    database: database.to_str().unwrap(),
                    result_file: result_path.to_str(),
                    collection: "notes",
                    profile: ResourceProfile::Balanced,
                    durability: Durability::Fast,
                },
                &context,
                Instant::now(),
            )
            .unwrap();
            // Isolate terminal publication from query execution, tested separately.
            journal.next = journal.planned;
            if stage == 0 {
                journal.file = File::open(&journal.path).unwrap();
            } else if stage == 1 {
                fs::rename(&result_path, dir.join("prior-result.json")).unwrap();
                fs::create_dir(&result_path).unwrap();
            }
            let mut work = context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
            let value = builder
                .object([(
                    "metrics",
                    builder
                        .object([("import_time_ms", JsonValue::Number(1.0))])
                        .unwrap(),
                )])
                .unwrap();
            let result = journal.publish_run(
                BenchmarkRun {
                    value,
                    context: context.clone(),
                    work,
                },
                Some(output),
            );
            let records = fs::read_to_string(&journal.path).unwrap();
            match stage {
                0 => {
                    assert_eq!(result.err().unwrap().code(), ErrorCode::IoError);
                    assert_eq!(fs::read(&result_path).unwrap(), b"prior result");
                    assert!(!records.contains("\"run_completed\""));
                }
                1 => {
                    assert_eq!(result.err().unwrap().code(), ErrorCode::IoError);
                    assert_eq!(
                        fs::read(dir.join("prior-result.json")).unwrap(),
                        b"prior result"
                    );
                    assert!(records.lines().last().unwrap().contains("\"run_failed\""));
                }
                _ => {
                    let run = result.unwrap();
                    let mut expected = Vec::new();
                    let mut run = run;
                    run.value
                        .write_pretty_retained_in(
                            &mut expected,
                            JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
                            &run.context,
                            &mut run.work,
                        )
                        .unwrap();
                    assert_eq!(fs::read(&result_path).unwrap(), expected);
                    assert!(records
                        .lines()
                        .last()
                        .unwrap()
                        .contains("\"run_completed\""));
                }
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn journal_write_failure_is_sticky_and_does_not_corrupt_prior_events() {
        let dir = text_only_benchmark_dataset("journal-io-failure");
        let context = JsonOperationContext::standalone();
        let (dataset, _work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
        let modes = [BenchmarkQueryMode::TextBm25];
        let database = dir.join("data.coffer");
        let mut journal = ExecutionJournal::new(
            &dataset,
            &modes,
            ExecutionConfig {
                database: database.to_str().unwrap(),
                result_file: None,
                collection: "notes",
                profile: ResourceProfile::Balanced,
                durability: Durability::Fast,
            },
            &context,
            Instant::now(),
        )
        .unwrap();
        let before = fs::read(&journal.path).unwrap();
        journal.file = File::open(&journal.path).unwrap();
        let write_error = journal.begin(0, "cold", 0).unwrap_err();
        assert_eq!(write_error.code(), ErrorCode::IoError);
        assert!(journal.write_failed);
        journal.file = fs::OpenOptions::new()
            .append(true)
            .open(&journal.path)
            .unwrap();
        let error = journal.fail(Error::new(ErrorCode::ChecksumMismatch, "original failure"));
        assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "execution_record_error"
                && value.contains("further writes are disabled")));
        assert_eq!(fs::read(&journal.path).unwrap(), before);
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_initial_record_retains_location_and_shared_budget_can_be_reused() {
        let dir = text_only_benchmark_dataset("journal-budget-failure");
        let context = JsonOperationContext::standalone();
        let (dataset, work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
        let modes = [BenchmarkQueryMode::TextBm25];
        let database = dir.join("data.coffer");
        let new_journal = || {
            ExecutionJournal::new(
                &dataset,
                &modes,
                ExecutionConfig {
                    database: database.to_str().unwrap(),
                    result_file: None,
                    collection: "notes",
                    profile: ResourceProfile::Balanced,
                    durability: Durability::Fast,
                },
                &context,
                Instant::now(),
            )
        };
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes(),
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        let error = new_journal().err().unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        let path = error
            .detail()
            .iter()
            .find(|(key, _)| key == "benchmark_execution_record")
            .unwrap()
            .1
            .clone();
        assert_eq!(Path::new(&path).components().count(), 1);
        assert!(dir.join(&path).is_file());
        assert_eq!(fs::metadata(dir.join(&path)).unwrap().len(), 0);
        drop(busy);
        let mut journal = new_journal().unwrap();
        assert_ne!(journal.path, dir.join(path));
        assert!(journal.begin(0, "warm", 0).is_err());
        journal.begin(0, "cold", 0).unwrap();
        let interrupted = journal.path.clone();
        drop(journal);
        let records = fs::read_to_string(interrupted).unwrap();
        assert!(records.contains("\"started\""));
        assert!(!records.contains("\"run_completed\""));
        drop((dataset, work));
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
}
