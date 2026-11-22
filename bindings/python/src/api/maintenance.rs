//! Materialize full-width maintenance counters and preserve known file effects.
use super::*;
use cofferdb::{CompactSummary, RebuildIndexSummary, SnapshotSummary};

pub(super) enum Receipt {
    // Snapshot carries a retained catalog tree as well as the file summary. Its
    // slot can be taken during Python conversion and filled with the surviving
    // host lease after the collections list is built.
    Snapshot(SnapshotSummary, Option<Output>),
    // Rebuild keeps the native index summary beside the Output used to admit
    // construction of its Python fields.
    Rebuild(RebuildIndexSummary, Output),
    // Compaction uses the same lease lifetime even when index rebuilding was
    // skipped. The returned summary describes the main-file replacement and its
    // index outcome separately.
    Compact(CompactSummary, Output),
}

// PyO3's scalar convenience conversions may panic on allocator failure.
struct Fields<'py>(Bound<'py, PyDict>);

impl<'py> Fields<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        output::dictionary(py).map(Self)
    }

    // Prepare the Python key using a fallible allocation before inserting this one
    // field. An insertion failure propagates to the receipt builder, which may be
    // reporting an operation already completed on disk.
    fn put(&self, key: &str, value: &Bound<'py, PyAny>) -> PyResult<()> {
        self.0.set_item(output::name(self.0.py(), key)?, value)
    }

    fn text(&self, key: &str, value: &str) -> PyResult<()> {
        self.put(key, &output::string(self.0.py(), value)?)
    }

    fn optional_text(&self, key: &str, value: Option<&str>) -> PyResult<()> {
        // Represent an unavailable optional field as a present Python None where
        // the receipt schema calls for it. An empty string remains a supplied text
        // value.
        match value {
            Some(value) => self.text(key, value),
            None => self.put(key, self.0.py().None().bind(self.0.py())),
        }
    }

    fn number(&self, key: &str, value: u128) -> PyResult<()> {
        let py = self.0.py();
        // Avoid routing maintenance counters through binary64. Sizes and offsets
        // may exceed its consecutive-integer range, and native duration_ms is a
        // u128. Values that fit u64 need only the unsigned CPython constructor.
        if let Ok(small) = u64::try_from(value) {
            return self.put(key, &output::unsigned(py, small)?);
        }
        // The decimal representation of u128 fits in 39 digits and a terminator.
        let mut digits = [0u8; 40];
        // Leave the last byte zero and fill digits from the end. Only values above
        // u64::MAX reach this branch, so the parser always sees a nonempty decimal
        // string.
        let mut start = digits.len() - 1;
        let mut remaining = value;
        while remaining > 0 {
            start -= 1;
            digits[start] = b'0' + (remaining % 10) as u8;
            remaining /= 10;
        }
        // SAFETY: the GIL is held and the decimal buffer remains NUL-terminated
        // for the call. CPython returns a new reference or NULL with an error;
        // from_owned_ptr_or_err takes ownership or propagates that error.
        let integer = unsafe {
            Bound::from_owned_ptr_or_err(
                py,
                pyo3::ffi::PyLong_FromString(
                    // Pass the first significant digit rather than the start of
                    // the zero-filled array. All digits and the terminator stay on
                    // this stack until Python has built its own
                    // arbitrary-precision integer.
                    digits[start..].as_ptr().cast(),
                    std::ptr::null_mut(),
                    10,
                ),
            )
        }?;
        self.put(key, &integer)
    }

    fn attach(&self, error: &PyErr, key: &str) -> PyResult<()> {
        let py = self.0.py();
        // Attach the completed mapping in one attribute assignment. A field
        // allocation never exposes the half-built mapping through this exception.
        error.value(py).setattr(output::name(py, key)?, &self.0)
    }
}

impl Receipt {
    pub(super) fn dictionary(&mut self, py: Python<'_>, pid: u32) -> PyResult<Py<PyAny>> {
        // Check the PID before borrowing the output or its budget. A child can
        // inherit this receipt while the parent still owns both.
        check_pid(py, pid)?;
        // The fixed allowance covers dict slots, keys and full-width integers.
        // Variable Unicode fields may use up to four bytes per input byte.
        {
            let (lease, text_bytes) = match self {
                // All three variants retain an Output. For a snapshot the catalog
                // projection itself owns the budget used to build the surrounding
                // dictionary.
                Self::Rebuild(r, lease) => (lease, r.collection.as_ref().map_or(0, String::len)),
                Self::Compact(r, lease) => (
                    lease,
                    r.index_rebuild_reason.as_ref().map_or(0, String::len),
                ),
                Self::Snapshot(r, catalog) => (
                    catalog.as_mut().expect("unconsumed snapshot catalog"),
                    r.source_path.as_os_str().len() + r.output_path.as_os_str().len(),
                ),
            };
            // Admit the initial dictionary before creating any of its Python
            // fields. A budget error at this stage concerns delivery of an already
            // completed operation.
            lease
                .reserve_host(4096 + 4 * text_bytes as u128)
                .map_err(|error| materialization_error(py, output::Failure::Engine(error)))?;
        }
        #[cfg(feature = "native-test-support")]
        if std::env::var_os("COFFER_TEST_REFUSE_MAINTENANCE_RESULT").is_some() {
            return Err(errors::python::<errors::ResultMaterializationError>(
                py,
                "test result materialization refused",
            ));
        }
        let d = Fields::new(py)?;
        match self {
            Self::Snapshot(r, catalog) => {
                d.number("last_valid_commit", r.last_valid_commit as u128)?;
                // The digest and byte count cover the entire snapshot. Its
                // validated commit boundary is reported separately because derived
                // index data may follow it.
                d.number("size_bytes", r.bytes_copied as u128)?;
                d.text("sha256", &r.sha256)?;
                // Paths are display strings. Non-Unicode filesystem names can lose
                // information here; the digest still identifies the original file
                // bytes.
                d.text("source_path", &r.source_path.to_string_lossy())?;
                d.text("output_path", &r.output_path.to_string_lossy())?;
                d.number("critical_end_offset", r.critical_end_offset as u128)?;
                d.number(
                    "format_version",
                    cofferdb::binding::SNAPSHOT_FORMAT_VERSION as u128,
                )?;
                // Keep the package version and required feature IDs separate from
                // the database's on-disk format version. These describe different
                // reader requirements.
                d.number("package_version", 1)?;
                d.put("required_features", output::list(py, 0)?.as_any())?;
                d.text("index_suffix_status", &r.index_suffix_status)?;
                // The snapshot's collection list is a retained native projection.
                // Materialize it while its lease can still cover both copies, then
                // put the resulting host lease back into the receipt until final
                // delivery.
                let projection = catalog.take().expect("snapshot catalog delivered once");
                // Include the surrounding receipt in the estimate for the
                // converted catalog. reserve_host grows to an aggregate total on
                // the same projection.
                let items = output::Materialized::build(
                    py,
                    projection,
                    pid,
                    4096 + 4 * (r.source_path.as_os_str().len() + r.output_path.as_os_str().len()),
                )
                .map_err(|e| materialization_error(py, e))?;
                d.put("collections", items.value.bind(py))?;
                *catalog = Some(items.output);
            }
            Self::Rebuild(r, _) => {
                d.text("index_suffix_status", &r.index_suffix_status)?;
                // None means there is no reported checkpoint commit. Preserve it
                // as Python None so callers can distinguish it from a numeric
                // commit ID.
                match r.last_index_commit_id {
                    Some(id) => d.number("last_index_commit_id", id.into())?,
                    None => d.put("last_index_commit_id", py.None().bind(py))?,
                }
                d.text("schema_version", &r.schema_version)?;
                d.optional_text("collection", r.collection.as_deref())?;
                // The named collection records the request. The engine can do
                // wider work or build a transient index; the scope and remaining
                // rebuild flag describe the outcome.
                d.text("rebuild_scope", &r.rebuild_scope)?;
                d.put(
                    "rebuild_needed",
                    &pyo3::types::PyBool::new(py, r.rebuild_needed)
                        .to_owned()
                        .into_any(),
                )?;
                d.number("index_record_count", r.index_record_count as u128)?;
                d.number("index_extent_count", r.index_extent_count as u128)?;
                d.number("text_segment_count", r.text_segment_count as u128)?;
                d.number("vector_segment_count", r.vector_segment_count as u128)?;
                d.number("duration_ms", r.duration_ms)?;
            }
            Self::Compact(r, _) => {
                // Copy the optional index failure even on a successful compaction.
                // Replacement of the primary file precedes work on its derived
                // checkpoint.
                d.text("index_rebuild", &r.index_rebuild)?;
                d.optional_text("index_rebuild_reason", r.index_rebuild_reason.as_deref())?;
                d.number("before_file_size_bytes", r.before_file_size as u128)?;
                d.number("after_file_size_bytes", r.after_file_size as u128)?;
                d.number("records_removed", r.records_removed as u128)?;
                d.number("tombstones_removed", r.tombstones_removed as u128)?;
                d.number("segments_merged", r.segments_merged as u128)?;
                d.number("duration_ms", r.duration_ms)?;
            }
        }
        // The Python dictionary now owns its fields. The caller retains this
        // Receipt until its final process and signal checks have chosen what to
        // return.
        Ok(d.0.into_any().unbind())
    }
}

pub(super) fn execute_maintenance<I: Send>(
    py: Python<'_>,
    owner: &Arc<owner::Owner>,
    operation: &str,
    prepare: impl FnOnce(&Operation) -> PyResult<I>,
    run: impl FnOnce(Operation, &cofferdb::Database, I) -> cofferdb::Result<Receipt> + Send,
) -> PyResult<Py<PyAny>> {
    // Keep the engine Result inside Execution: even a failed maintenance call
    // may have published a file. finish_maintenance combines the engine's
    // error details with the effects observed before the call returned.
    let mut observation = None;
    // Maintenance has its own outcome fields. Do not request the ordinary
    // record-write receipt from the shared execution path.
    let execution = execute(py, owner, false, prepare, |mut op, db, input| {
        // Start observing only once input preparation and admission have reached
        // the native call. Earlier failures have no maintenance progress to
        // attach.
        observation = Some(op.observe_maintenance());
        Ok(run(op, db, input))
    })?;
    execution.finish_maintenance(py, operation, observation)
}

impl Execution<cofferdb::Result<Receipt>> {
    fn finish_maintenance(
        mut self,
        py: Python<'_>,
        operation: &str,
        observation: Option<cofferdb::binding::MaintenanceObservation>,
    ) -> PyResult<Py<PyAny>> {
        // Keep a successful native receipt alive through Python conversion and
        // exception delivery. Its reservation backs the owned outcome
        // independently of the dictionary built from it.
        let mut receipt = None;
        let mut native_fields = None;
        let built = match self.result {
            Ok(Ok(value)) => {
                // The native action has already returned a receipt at this point.
                // The protected closure covers Python marshalling only. Failure
                // here does not undo the completed action.
                let retained = receipt.insert(value);
                self.call
                    .protect(|| retained.dictionary(py, self.pid))
                    .map_err(|e| failure(py, e))
                    .and_then(|value| value)
            }
            Ok(Err(native)) => {
                // Build supplemental fields before converting the native error
                // into a Python exception. A field allocation failure leaves the
                // native error available for normal delivery.
                if self.pid == std::process::id() {
                    native_fields = observation.as_ref().and_then(|observation| {
                        outcome_fields(py, operation, observation, native.detail(), false).ok()
                    });
                }
                Err(engine_error(py, native))
            }
            Err(error) => Err(error),
        };
        // A signal can replace an otherwise successful result after the physical
        // maintenance action completed. Attach known effects to the exception
        // ultimately selected, using the original process's evidence.
        let result = errors::finish(py, built, None, self.pid).inspect_err(|error| {
            // Only the process that attempted the operation can use this
            // observation. A forked child must not present the parent's progress
            // as evidence of its own work.
            if self.pid == std::process::id() {
                if let Some(observation) = observation {
                    // When no prepared native fields survived, use operation
                    // progress alone. The panic class still triggers recovery
                    // advice even if no native error details were produced.
                    let fields = native_fields.or_else(|| {
                        outcome_fields(
                            py,
                            operation,
                            &observation,
                            &[],
                            error.is_instance_of::<errors::EnginePanicError>(py),
                        )
                        .ok()
                    });
                    if let Some(fields) = fields {
                        // Use the snapshot-specific attribute for target-file
                        // operations. Compaction and index rebuilding expose their
                        // progress through the general maintenance attribute.
                        let attribute = if operation == "snapshot" {
                            "snapshot_outcome"
                        } else {
                            "maintenance_outcome"
                        };
                        // Treat outcome delivery as best effort. An error while
                        // attaching the attribute must not replace the exception
                        // already selected by finish.
                        let _ = fields.attach(error, attribute);
                    }
                }
            }
        });
        // Hold the result's host-memory credit through signal and effect delivery.
        drop(receipt);
        result
    }
}

fn outcome_fields<'py>(
    py: Python<'py>,
    operation: &str,
    observation: &cofferdb::binding::MaintenanceObservation,
    native: &[(String, String)],
    panic: bool,
) -> PyResult<Fields<'py>> {
    let observed = observation.fields();
    // A signal or Python allocation failure can leave no engine error detail.
    // Start with observed progress. When both sources describe the same field,
    // the native detail takes precedence for the stage that actually failed.
    let mut fields: std::collections::BTreeMap<&str, &str> = observed
        .iter()
        .map(|(key, value)| (*key, value.as_str()))
        .collect();
    for (key, value) in native {
        fields.insert(key, value);
    }
    // Put the operation name on the outcome even when none of the observed or
    // native keys makes it into the Python field set.
    let receipt = Fields::new(py)?;
    receipt.text("operation", operation)?;
    // Expose only recognized receipt fields. Numeric entries remain integers in
    // Python and are skipped if the engine detail cannot be parsed as an unsigned
    // count.
    for (&key, &value) in &fields {
        match key {
            "last_valid_commit" | "size_bytes" | "last_index_commit_id" => {
                if let Ok(number) = value.parse::<u64>() {
                    receipt.number(key, number.into())?;
                }
            }
            "stage"
            | "effect"
            | "durability"
            | "target"
            | "sha256"
            | "index_suffix_status"
            | "index_rebuild"
            | "index_rebuild_reason" => receipt.text(key, value)?,
            _ => (),
        }
    }
    // Use None for companion fields omitted by the engine when their corresponding
    // status is available. This preserves the shape of a partial index observation
    // without inventing details.
    if fields.contains_key("index_rebuild") && !fields.contains_key("index_rebuild_reason") {
        receipt.optional_text("index_rebuild_reason", None)?;
    }
    // A known suffix status can coexist with an unavailable index commit. This
    // distinguished state lets the caller tell an unavailable commit from a
    // receipt with no suffix observation.
    if fields.contains_key("index_suffix_status") && !fields.contains_key("last_index_commit_id") {
        receipt.put("last_index_commit_id", py.None().bind(py))?;
    }
    // For a panic or unknown effect, distinguish the external snapshot target from
    // the current main file. Each requires inspecting a different durable artifact
    // before repeating the operation.
    if panic || fields.get("effect") == Some(&"unknown") {
        let recovery = if operation == "snapshot" {
            "Close the writer. Inspect the target path and verify the size, SHA-256 and commit before retrying."
        } else {
            "Close the writer, reopen the current main path and check its committed data and index state."
        };
        receipt.text("suggested_action", recovery)?;
    }
    Ok(receipt)
}
