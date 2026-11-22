//! Synchronous Python owners; native execution and host conversion are separate.

use crate::{convert, errors, output, owner, panic_guard, requests};
use cofferdb::binding::{Input, Operation, Output, WriteOutcome};
use cofferdb::{
    DatabaseConfig, DatabaseOpenOptions, Durability, Error, ErrorCode, IoMode, ResourceProfile,
};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::sync::Arc;
mod arguments;
mod collection;
mod maintenance;
#[cfg(test)]
mod tests;
pub(crate) use collection::Collection;

/// Preserve omission without constructing Python defaults during argument unpacking.
/// Explicit None remains Some and is checked by the ordinary input converter.
#[derive(Default)]
struct DefaultArgument(Option<Py<PyAny>>);

impl FromPyObject<'_, '_> for DefaultArgument {
    type Error = PyErr;

    // Only an actual Python argument reaches extraction. Keep an owned handle even
    // for Python None; the Rust Default is the distinct omitted case used by the
    // argument parser.
    fn extract(object: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self(Some(object.to_owned().unbind())))
    }
}

impl DefaultArgument {
    // Rebind the saved Python reference only while this thread is attached. The
    // converter can then apply the target parameter's own rule for an explicitly
    // supplied None.
    fn bind<'py>(&self, py: Python<'py>) -> Option<&Bound<'py, PyAny>> {
        self.0.as_ref().map(|value| value.bind(py))
    }
}

struct Execution<T> {
    // Owns admission to the database until final result or exception delivery,
    // including Python callbacks from signal checking.
    call: owner::Call,
    // Native result or an error selected during argument preparation or execution,
    // held for attached Python conversion.
    result: PyResult<T>,
    // Optional confirmation snapshot for a writing call. Constructing the Python
    // result can fail after this state was recorded.
    outcome: Option<WriteOutcome>,
    // Process that admitted the call. Check it again on Python handoff because a
    // signal handler can fork after native execution.
    pid: u32,
}

// An admission or preparation failure precedes the core write. Expose absent
// confirmation counters for that state; zero counters are used only after tracking
// has started.
fn not_started() -> WriteOutcome {
    WriteOutcome {
        state: "not_started",
        confirmed_committed_batches: None,
        confirmed_committed_records: None,
        last_confirmed_commit_id: None,
        last_confirmed_commit_sequence_no: None,
    }
}

// Separate preparation, detached native execution and attached delivery. The
// execution object carries the owner admission and any confirmed write outcome
// across the last boundary.
fn execute<I: Send, T: Send>(
    py: Python<'_>,
    owner: &Arc<owner::Owner>,
    writing: bool,
    prepare: impl FnOnce(&Operation) -> PyResult<I>,
    run: impl FnOnce(Operation, &cofferdb::Database, I) -> cofferdb::Result<T> + Send,
) -> PyResult<Execution<T>> {
    let pid = owner.pid();
    // Before admission there is no Call to hold through error delivery and no native
    // write to observe. Finish the error here so a rejected writer still reports
    // that it never started.
    let mut call = owner.enter(false).map_err(|error| {
        errors::finish::<()>(
            py,
            Err(failure(py, error)),
            writing.then(not_started).as_ref(),
            pid,
        )
        .unwrap_err()
    })?;
    // Admission has succeeded, so even failure to create the operation must retain
    // the Call through Python error delivery. No native mutation has begun at this
    // point.
    let operation = match call.run(|database| Ok(Operation::new(database))) {
        Ok(operation) => operation,
        Err(error) => {
            return Ok(Execution {
                call,
                result: Err(failure(py, error)),
                outcome: writing.then(not_started),
                pid,
            })
        }
    };
    // Take the observer before run consumes the operation. It can still read the
    // storage confirmations if native execution fails or unwinds before producing
    // its normal receipt.
    let observation = operation.observation();
    // Convert Python arguments while attached. Only owned input crosses into
    // the detached native operation, and Execution keeps the owner gate held
    // through result conversion and signal handling.
    // Argument conversion can execute Python code. Use the owner guard here too,
    // then let the branch below recheck which process resumed after preparation.
    let prepared = match call.protect(|| prepare(&operation)) {
        Ok(result) => result,
        Err(error) => Err(failure(py, error)),
    };
    let (result, outcome) = match prepared {
        // The argument conversion above ran while attached to Python. Recheck the
        // process before entering the detached call: a child must not use the
        // parent's database or inherit a claim that a write started.
        Ok(input) if pid == std::process::id() => {
            // The detached closure owns its inputs and freezes write progress
            // while Python callbacks still cannot intervene on this thread. Build
            // a Python exception only after reattaching.
            let (native, outcome) = py.detach(|| {
                let result = call.run(|database| run(operation, database, input));
                // This snapshot precedes attach, Python allocations and signals.
                let outcome = writing.then(|| observation.write_outcome());
                (result, outcome)
            });
            (native.map_err(|error| failure(py, error)), outcome)
        }
        // If argument preparation returned in a fork child, discard the prepared
        // input rather than touching the inherited database handle. Do not
        // attribute the parent's confirmed writes to the child.
        Ok(_) => (
            Err(errors::sdk(py, &owner::Failure::Forked)),
            writing.then(not_started),
        ),
        // A conversion error is still before the native execution boundary.
        // Preserve the original preparation exception and attach a not-started
        // outcome only for an operation declared to write.
        Err(error) => (Err(error), writing.then(not_started)),
    };
    // Move the Call into the result wrapper on both success and failure. Releasing
    // admission here would permit another operation or close to race host-side
    // conversion and signal checks.
    Ok(Execution {
        call,
        result,
        outcome,
        pid,
    })
}

impl Execution<Output> {
    fn finish(mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Materialization is still inside the owner's panic guard. If Python
        // allocation fails after a write, error delivery uses the captured
        // confirmation instead of inferring the write outcome from the failure.
        let result = self.result.and_then(|value| {
            self.call
                .protect(|| output::Materialized::build(py, value, self.pid, 0))
                .map_err(|error| failure(py, error))?
                .map_err(|error| materialization_error(py, error))
        });
        // Both paths perform the host handoff with the saved PID and optional
        // confirmation. A materialized success retains its output charge through
        // signal checking.
        match result {
            Ok(value) => value.finish(py, self.pid, self.outcome.as_ref()),
            Err(error) => errors::finish(py, Err(error), self.outcome.as_ref(), self.pid),
        }
    }
}

impl Execution<cofferdb::binding::CollectionIdentity> {
    fn finish(mut self, py: Python<'_>, owner: &Arc<owner::Owner>) -> PyResult<Py<Collection>> {
        // A collection result needs a Python owner-backed object as well as the
        // native identity's lease. Keep construction under the admitted Call just
        // like ordinary JSON result materialization.
        let result = self.result.and_then(|identity| {
            self.call
                .protect(|| {
                    collection_output(
                        py,
                        owner,
                        identity.into_parts(),
                        self.pid,
                        self.outcome.as_ref(),
                    )
                })
                .map_err(|error| failure(py, error))?
        });
        // The successful path already checked signals while its lease was live.
        match result {
            Ok(value) => Ok(value),
            Err(error) => errors::finish(py, Err(error), self.outcome.as_ref(), self.pid),
        }
    }
}

fn collection_output(
    py: Python<'_>,
    owner: &Arc<owner::Owner>,
    parts: (String, String, Output),
    pid: u32,
    outcome: Option<&WriteOutcome>,
) -> PyResult<Py<Collection>> {
    // On every early return the owned names must be destroyed before their
    // shared credit is returned. A successful Python handoff ends that credit.
    let mut lease;
    let (collection_id, name) = {
        let (collection_id, name, work) = parts;
        lease = work;
        (collection_id, name)
    };
    // Charge the host collection wrapper before allocating it. The identity
    // strings already have native credit; their existing Rust buffers will move
    // into the Python-owned Collection.
    lease
        .reserve_host(256)
        .map_err(|error| materialization_error(py, output::Failure::Engine(error)))?;
    // Recheck the admitted process after reserving and before constructing a
    // Python object that holds the database owner. A child cannot use its
    // fork-inherited owner as a new collection handle.
    check_pid(py, pid)?;
    let collection = Py::new(
        py,
        Collection {
            owner: Arc::clone(owner),
            collection_id,
            name,
        },
    )?;
    // Run signal handling while the identity lease is still live. A Python
    // exception at final handoff can coexist with an already completed native
    // collection operation.
    let result = errors::finish(py, Ok(collection), outcome, pid);
    drop(lease);
    result
}

// Use the common execution path for operations returning retained JSON. The
// writing flag follows the native operation, since serialization does not decide
// whether a failure needs write-progress metadata.
fn read_output<I: Send>(
    py: Python<'_>,
    owner: &Arc<owner::Owner>,
    writing: bool,
    prepare: impl FnOnce(&Operation) -> PyResult<I>,
    run: impl FnOnce(Operation, &cofferdb::Database, I) -> cofferdb::Result<Output> + Send,
) -> PyResult<Py<PyAny>> {
    execute(py, owner, writing, prepare, run)?.finish(py)
}

pub(crate) fn engine_error(py: Python<'_>, error: Error) -> PyErr {
    // Use a fresh operation to project the structured engine error. Creating the
    // exception can itself fail, so distinguish failure to build its Python fields
    // from the original engine error.
    match Operation::standalone().error(error) {
        // Allow for the Python exception wrapper in addition to its projected
        // native fields. Failure at this stage is handled by the materialization
        // path, even if the original core error carried a different category.
        Ok(value) => match output::Materialized::build(py, value, std::process::id(), 1024) {
            Ok(value) => value.exception(py).unwrap_or_else(|error| error),
            Err(error) => materialization_error(py, error),
        },
        // If projecting the original error to JSON itself failed, build a small
        // fallback exception directly. Trying to project that second core error
        // here would repeat the failing operation.
        Err(_) => errors::python::<errors::ResultMaterializationError>(
            py,
            "the native error could not be materialized",
        ),
    }
}

fn failure(py: Python<'_>, error: owner::Failure) -> PyErr {
    match error {
        // Owner errors and engine errors have different Python exception
        // contracts. Only the latter have a native structured error payload to
        // project.
        owner::Failure::Engine(error) => engine_error(py, error),
        other => errors::sdk(py, &other),
    }
}

fn shape(py: Python<'_>, error: convert::Failure) -> PyErr {
    // Shape failures can map directly to a Python type or value error. If
    // classification instead returns an engine error, preserve its ordinary
    // structured native error projection.
    match errors::conversion_shape(py, error) {
        Ok(error) => error,
        Err(error) => engine_error(py, error),
    }
}

fn materialization_error(py: Python<'_>, error: output::Failure) -> PyErr {
    let cause = match error {
        output::Failure::Python(error) => error,
        // An inherited owner has a dedicated fork error. Return it directly before
        // chaining ordinary output failures under a materialization exception.
        output::Failure::Forked => return errors::sdk(py, &owner::Failure::Forked),
        output::Failure::Engine(error) => engine_error(py, error),
    };
    // A core error can itself fail during host projection. Check the resulting
    // exception too, so a nested allocation failure keeps its host identity.
    if cause.is_instance_of::<PyMemoryError>(py) {
        return cause;
    }
    let error = errors::python::<errors::ResultMaterializationError>(
        py,
        "native work finished but its Python result could not be built",
    );
    // Keep the underlying Python or engine exception as the cause. Callers can
    // inspect why conversion failed while the outer class identifies the failure
    // as part of result delivery.
    error.set_cause(py, Some(cause));
    error
}

fn schema<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    fields: &[&str],
) -> PyResult<&'a Bound<'py, PyDict>> {
    // Validate the container and every key before converting any values.
    // Restricting this path to a built-in dict and built-in strings keeps
    // user-defined mapping and comparison behavior out of configuration decoding.
    let object = convert::dict(value).map_err(|error| shape(value.py(), error))?;
    for (key, _) in object.iter() {
        if !key.is_exact_instance_of::<PyString>() {
            return Err(errors::python::<PyTypeError>(
                value.py(),
                "configuration keys must be built-in str values",
            ));
        }
        // Once the key has the exact built-in type, compare it with the allowed
        // field names. A well-formed string outside the schema is an engine schema
        // error, distinct from a Python type error for the key itself.
        let key = key.cast::<PyString>()?;
        if !fields.iter().any(|name| key == name) {
            return Err(engine_error(
                value.py(),
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "unsupported configuration field",
                ),
            ));
        }
    }
    Ok(object)
}

fn profile(value: &Bound<'_, PyAny>, input: &mut Input) -> PyResult<ResourceProfile> {
    // Use ordinary admitted text for the enum name. The core parser supplies the
    // accepted spellings, allowing Python and Rust to interpret the same profile
    // consistently.
    let name = convert::text(value, input).map_err(|error| shape(value.py(), error))?;
    ResourceProfile::parse(&name).ok_or_else(|| {
        engine_error(
            value.py(),
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "unsupported resource profile",
            )
            .with_detail("field", "resource_profile"),
        )
    })
}
fn durability(value: &Bound<'_, PyAny>, input: &mut Input) -> PyResult<Durability> {
    let name = convert::text(value, input).map_err(|error| shape(value.py(), error))?;
    // Only a parsed durability mode can reach database configuration. A misspelled
    // explicit setting must surface with its field name because falling back could
    // change which commits are synchronized.
    Durability::parse(&name).ok_or_else(|| {
        engine_error(
            value.py(),
            Error::new(ErrorCode::InvalidJsonlSchema, "unsupported durability")
                .with_detail("field", "durability"),
        )
    })
}
fn io_mode(value: &Bound<'_, PyAny>, input: &mut Input) -> PyResult<IoMode> {
    let name = convert::text(value, input).map_err(|error| shape(value.py(), error))?;
    // Preserve the requested mode even where the current backend serves every mode
    // with file reads. Unsupported spellings are configuration errors
    // independently of backend capabilities.
    match name.as_str() {
        "auto" => Ok(IoMode::Auto),
        "read" => Ok(IoMode::Read),
        "mmap_disabled" => Ok(IoMode::MmapDisabled),
        _ => Err(engine_error(
            value.py(),
            Error::new(ErrorCode::InvalidJsonlSchema, "unsupported I/O mode")
                .with_detail("field", "io_mode"),
        )),
    }
}
fn boolean(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    // Require a built-in bool for configuration flags. In particular, accepting
    // Python truthiness would silently treat integers or arbitrary objects as
    // explicit engine settings.
    convert::boolean(value).map_err(|error| shape(value.py(), error))
}
fn cache(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let number = convert::unsigned(value).map_err(|error| shape(value.py(), error))?;
    // The host converter checks for a built-in nonnegative integer representable
    // in u64. This second conversion enforces the possibly narrower native
    // cache-size type before the value reaches the engine.
    usize::try_from(number).map_err(|_| {
        errors::python::<PyValueError>(
            value.py(),
            "cache budget is outside the native integer range",
        )
    })
}

fn database_config(
    value: Option<&Bound<'_, PyAny>>,
    input: &mut Input,
) -> PyResult<DatabaseConfig> {
    // A create request starts from the engine's complete defaults, whether the
    // caller omitted config or passed None. Only fields actually present in the
    // dict replace those defaults.
    let mut config = DatabaseConfig::default();
    if let Some(value) = value.filter(|value| !value.is_none()) {
        let object = schema(
            value,
            &[
                "resource_profile",
                "cache_budget_mb",
                "background_compaction",
                "durability",
                "io_mode",
            ],
        )?;
        if let Some(value) = crate::output::lookup(object, "resource_profile")? {
            config.resource_profile = profile(&value, input)?;
        }
        // An explicit None cache budget means use the selected profile's
        // allowance. Preserve an explicit zero, which disables the numeric
        // allowance instead of asking for the preset.
        if let Some(value) =
            crate::output::lookup(object, "cache_budget_mb")?.filter(|value| !value.is_none())
        {
            config.cache_budget_mb = Some(cache(&value)?);
        }
        // Check the flag's Python type here and leave the availability of
        // background compaction to the engine. A genuine True and an object that
        // merely tests true must not receive the same diagnostic.
        if let Some(value) = crate::output::lookup(object, "background_compaction")? {
            config.background_compaction = boolean(&value)?;
        }
        if let Some(value) = crate::output::lookup(object, "durability")? {
            config.durability = durability(&value, input)?;
        }
        if let Some(value) = crate::output::lookup(object, "io_mode")? {
            config.io_mode = io_mode(&value, input)?;
        }
    }
    Ok(config)
}

fn open_options(
    value: Option<&Bound<'_, PyAny>>,
    input: &mut Input,
    read_only: bool,
) -> PyResult<DatabaseOpenOptions> {
    // The separate read_only keyword determines access mode for the handle. It is
    // deliberately outside the options mapping and cannot be overridden by a field
    // of the same name.
    let mut options = DatabaseOpenOptions {
        read_only,
        ..Default::default()
    };
    if let Some(value) = value.filter(|value| !value.is_none()) {
        let object = schema(
            value,
            &[
                "resource_profile",
                "cache_budget_mb",
                "background_compaction",
                "durability",
                "io_mode",
            ],
        )?;
        // Optional open overrides preserve None as absence. An unset profile,
        // cache limit, compaction flag or durability setting therefore reaches the
        // core without forcing a concrete choice at this boundary.
        if let Some(value) =
            crate::output::lookup(object, "resource_profile")?.filter(|value| !value.is_none())
        {
            options.resource_profile = Some(profile(&value, input)?);
        }
        if let Some(value) =
            crate::output::lookup(object, "cache_budget_mb")?.filter(|value| !value.is_none())
        {
            options.cache_budget_mb = Some(cache(&value)?);
        }
        if let Some(value) =
            crate::output::lookup(object, "background_compaction")?.filter(|value| !value.is_none())
        {
            options.background_compaction = Some(boolean(&value)?);
        }
        if let Some(value) =
            crate::output::lookup(object, "durability")?.filter(|value| !value.is_none())
        {
            options.durability = Some(durability(&value, input)?);
        }
        // I/O mode is a concrete enum in DatabaseOpenOptions. Omitting it selects
        // its default, but an explicitly supplied None still has to pass the text
        // converter and is rejected.
        if let Some(value) = crate::output::lookup(object, "io_mode")? {
            options.io_mode = io_mode(&value, input)?;
        }
    }
    Ok(options)
}

/// Own a synchronous database connection. Use create or open, then close the owner
/// explicitly or use it as a context manager.
///
/// Only one writer across processes can own a physical database at a time. One operation at
/// a time may use this owner, including via its collections; concurrent or reentrant use
/// raises BusyError. A read-only owner keeps its fixed view. Close invalidates dependent
/// collections but leaves already returned Python values usable. An inherited owner raises
/// ForkedHandleError in a child process.
#[pyclass(module = "cofferdb", frozen)]
pub(crate) struct Database {
    // Shared admission gate and native database state, also used by every
    // collection obtained through this Python object.
    owner: Arc<owner::Owner>,
}

fn check_pid(py: Python<'_>, pid: u32) -> PyResult<()> {
    // The captured PID belongs to the operation before it entered Python argument
    // handling. Recheck it at transition points because an inherited database or
    // reservation cannot be used from a child process.
    if pid == std::process::id() {
        Ok(())
    } else {
        Err(errors::sdk(py, &owner::Failure::Forked))
    }
}

/// Scan an existing database path read-only and return a CheckReport.
///
/// The scan uses a consistent committed view, also beside an idle current-version writer.
/// It delays mutations while scanning and leaves file contents unchanged. A completed scan
/// can report warning or error: inspect status, issues, index_check and index_check_reason.
/// If a concurrent change prevents the scan, CofferError with code recovery_error and
/// detail reason check_view_unavailable can be retried after the write finishes. Pass a
/// built-in str path.
#[pyfunction]
pub(crate) fn check(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pid = std::process::id();
    let result = panic_guard::catch(|| {
        // A path-based check needs no existing Python database owner. Build the
        // input under a standalone operation so the detached native scan can
        // retain its own path and result budget.
        let operation = Operation::standalone();
        let mut input = operation.input();
        let path = convert::text(path, &mut input).map_err(|error| shape(py, error))?;
        // Finish all Python access to the path and check the originating process
        // before detaching. The scan receives only the owned Rust string and its
        // operation reservation.
        check_pid(py, pid)?;
        let report = py
            .detach(|| operation.check_path(input.finish(path)))
            .map_err(|error| engine_error(py, error))?;
        // A completed scan can still fail while allocating the Python report. Keep
        // that projection failure distinct from an error returned by the file
        // checker.
        output::Materialized::build(py, report, pid, 0)
            .map_err(|error| materialization_error(py, error))
    })
    .unwrap_or_else(|_| Err(errors::sdk(py, &owner::Failure::Panic)));
    // Both branches perform final Python error or signal handling with the
    // original PID. There is no write outcome to attach to a path-based read.
    match result {
        Ok(value) => value.finish(py, pid, None),
        Err(error) => errors::finish(py, Err(error), None, pid),
    }
}

#[pymethods]
impl Database {
    /// Create a new database at a built-in str path and return its writer owner.
    ///
    /// An existing path is never overwritten. config is an optional DatabaseConfigInput
    /// mapping; omission selects the balanced preset, its cache budget, commit durability,
    /// automatic I/O and manual compaction. Close the owner to release writer admission. If
    /// creation is interrupted, inspect the path before retrying.
    #[staticmethod]
    #[pyo3(signature = (path, *, config=None))]
    fn create(
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pid = std::process::id();
        let result = panic_guard::catch(|| {
            // The database ledger and host owner do not exist yet. Convert the
            // path and configuration under one standalone operation, then pass
            // their combined reservation with the owned values into create.
            let operation = Operation::standalone();
            let mut input = operation.input();
            let path = convert::text(path, &mut input).map_err(|error| shape(py, error))?;
            let config = database_config(config, &mut input)?;
            check_pid(py, pid)?;
            // Detach for native creation only after checking which process
            // prepared the arguments. File creation and writer admission can take
            // place without holding the Python interpreter attachment.
            let native = py
                .detach(|| operation.create(input.finish((path, config))))
                .map_err(|error| engine_error(py, error))?;
            // Install the owner gate only for a successfully created native
            // database. It supplies the shared call and close state to all
            // collection handles constructed from this Python object.
            Ok(Self {
                owner: owner::Owner::new(native),
            })
        });
        let result = result.unwrap_or_else(|_| Err(errors::sdk(py, &owner::Failure::Panic)));
        errors::finish(py, result, None, pid)
    }

    /// Open an existing database at a built-in str path. The default requests writer
    /// admission, exclusive across processes. A competing writer raises CofferError.
    ///
    /// Pass read_only=True for a fixed view that rejects writes. options is an optional
    /// OpenOptionsInput mapping whose runtime overrides leave stored collection settings
    /// unchanged. A missing file raises CofferError. Use close or a context manager to end
    /// ownership.
    #[staticmethod]
    #[pyo3(signature = (path, *, read_only=false, options=None))]
    fn open(
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = boolean)] read_only: bool,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Capture the creating process before running host conversion. A fork
        // during conversion must not pass an inherited native call to a different
        // process.
        let pid = std::process::id();
        // Catch unwinding around conversion, native open and owner construction.
        // Convert a caught panic into the SDK's Python error path.
        let result = panic_guard::catch(|| {
            // Open starts with a standalone operation because no database owner
            // exists yet. Path and options share the same prepared-input budget.
            let operation = Operation::standalone();
            let mut input = operation.input();
            // Consume Python values while attached to the interpreter. The
            // detached call below receives only the owned Rust input assembled
            // through this builder.
            let path = convert::text(path, &mut input).map_err(|error| shape(py, error))?;
            let options = open_options(options, &mut input, read_only)?;
            // Recheck PID immediately before releasing the interpreter for native
            // work. The owner created on success belongs to that process.
            check_pid(py, pid)?;
            // Release the GIL during file open and lock admission. Return to
            // Python before converting native errors or exposing an owner.
            let native = py
                .detach(|| operation.open(input.finish((path, options))))
                .map_err(|error| engine_error(py, error))?;
            Ok(Self {
                owner: owner::Owner::new(native),
            })
        });
        // Use the common return boundary for errors and pending signals. An open
        // that succeeded natively can still fail as a host call.
        let result = result.unwrap_or_else(|_| Err(errors::sdk(py, &owner::Failure::Panic)));
        errors::finish(py, result, None, pid)
    }

    /// Release the native file and any writer lock, invalidating handles to this owner's
    /// collections. Repeated close is harmless. If another call is active, BusyError leaves
    /// the owner open; close it after that call finishes. An inherited owner raises
    /// ForkedHandleError.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        // Closing accepts an already closed or poisoned owner. Process and busy
        // checks still run before taking the native handle.
        let mut call = self.owner.enter(true).map_err(|error| failure(py, error))?;
        // Run file and lock destruction detached from Python. The owner gate stays
        // held until the interpreter-side return boundary has been checked.
        let result = py
            .detach(|| call.close())
            .map_err(|error| failure(py, error));
        // Check pending host signals before releasing the gate. A signal can
        // replace a successful result after the native handle was consumed.
        let result = errors::finish(py, result, None, self.owner.pid());
        drop(call);
        result
    }

    /// Publish a standalone physical copy at an unused output_path and return its
    /// SnapshotReport.
    ///
    /// Call between operations on this writer. The source stays open under its existing
    /// writer lock and its bytes are unchanged. The destination directory must support hard
    /// links; an existing file is preserved. After interruption, inspect snapshot_outcome
    /// or CofferError.detail when available. If a target was published, verify its size,
    /// SHA-256 and last valid commit before using it or choosing another name. Physical
    /// backups can contain deleted data.
    fn snapshot(&self, py: Python<'_>, output_path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Use the shared maintenance path for admission, execution observation and
        // receipt materialization. Only the request parser and native operation
        // vary by method.
        maintenance::execute_maintenance(
            py,
            &self.owner,
            "snapshot",
            // Admit and validate the output path before detaching. The prepared
            // path and its lease move into the snapshot operation together.
            |op| {
                let mut input = op.input();
                let path = convert::text(output_path, &mut input).map_err(|e| shape(py, e))?;
                Ok(input.finish(path))
            },
            // Retain the native catalog projection in the receipt until Python can
            // materialize it. The scalar snapshot facts remain available at that
            // boundary.
            |op, db, path| {
                op.snapshot(db, path)
                    .map(|(report, catalog)| maintenance::Receipt::Snapshot(report, Some(catalog)))
            },
        )
    }

    /// Rebuild derived indexes on this writer and return a RebuildIndexReport.
    ///
    /// Omit collection for the whole database. A named collection can still produce a full
    /// checkpoint, or only an in-memory verification if another collection has conflicting
    /// models. Inspect rebuild_scope, rebuild_needed and last_index_commit_id. Committed
    /// business data is preserved. If interrupted while rewriting the suffix, inspect
    /// maintenance_outcome when available, reopen and check the checkpoint before
    /// rebuilding again.
    #[pyo3(signature = (collection=None))]
    fn rebuild_index(
        &self,
        py: Python<'_>,
        collection: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // Use the same maintenance boundary for named or database-wide rebuilds.
        // The caller's scope is carried as an optional owned string.
        maintenance::execute_maintenance(
            py,
            &self.owner,
            "rebuild_index",
            |op| {
                let mut input = op.input();
                // Preserve omission as None. Only a supplied collection name
                // passes through strict string conversion and uses string
                // workspace.
                let name = collection
                    .map(|name| convert::text(name, &mut input))
                    .transpose()
                    .map_err(|e| shape(py, e))?;
                Ok(input.finish(name))
            },
            // Return the typed native rebuild report with its projection lease.
            // Later Python allocation can fail after native index work has
            // completed.
            |op, db, name| {
                op.rebuild_index(db, name)
                    .map(|(report, lease)| maintenance::Receipt::Rebuild(report, lease))
            },
        )
    }

    /// Replace the main file with a validated compacted generation on this writer; return a
    /// CompactReport.
    ///
    /// Allow space in the database directory for both generations. Create a snapshot first
    /// to retain the old file. Existing readers can keep their old view and delay disk
    /// reclamation. rebuild_indexes defaults to True. If the replacement succeeds but index
    /// construction fails, the report carries index_rebuild='failed' and its reason. If
    /// interrupted, inspect maintenance_outcome when available; close, reopen and check the
    /// main path before retrying. If only the index failed, rebuild it on the verified
    /// file.
    #[pyo3(signature = (*, rebuild_indexes=DefaultArgument::default()), text_signature = "($self, *, rebuild_indexes=True)")]
    fn compact(&self, py: Python<'_>, rebuild_indexes: DefaultArgument) -> PyResult<Py<PyAny>> {
        maintenance::execute_maintenance(
            py,
            &self.owner,
            "compact",
            // Treat an omitted boolean as true. A supplied value must pass the
            // binding's boolean converter before compaction begins.
            |_| {
                rebuild_indexes
                    .bind(py)
                    .map(boolean)
                    .transpose()
                    .map(|v| v.unwrap_or(true))
            },
            // Preserve both the native file result and its conversion lease. The
            // maintenance observation can still describe a completed replacement
            // if allocating the Python receipt fails.
            |op, db, rebuild| {
                op.compact(db, rebuild)
                    .map(|(report, lease)| maintenance::Receipt::Compact(report, lease))
            },
        )
    }

    /// Return DatabaseStats for this owner's visible database view, including live-record
    /// counts, storage estimates and the effective cache budget. The result is owned by
    /// Python.
    // Select the general statistics branch. Both kinds hold the owner gate through
    // native inspection and Python result conversion.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.stats_call(py, false)
    }
    /// Return IndexStats for the derived indexes without rebuilding them. Inspect
    /// rebuild_needed and the covered commit before relying on persisted coverage.
    // Select the index-only statistics branch without attaching write progress or
    // triggering a rebuild.
    fn index_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.stats_call(py, true)
    }

    /// Create and persist a named collection on this writer; return its Collection handle.
    ///
    /// config requires name, embedding_dim and distance_metric. Omitted tokenizer uses
    /// default_en; omitted vector_index selects exact search. An existing name or a
    /// read-only owner raises CofferError. If interrupted after native work, check
    /// operation_outcome when available and look up the name before retrying.
    fn create_collection(
        &self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Collection>> {
        // Treat collection creation as a write-bearing call so the outcome
        // survives failures during handle construction.
        execute(
            py,
            &self.owner,
            true,
            |operation| arguments::collection(config, operation).map_err(|error| shape(py, error)),
            |operation, database, config| operation.create_collection(database, config),
        )?
        // Build the Python handle while the native identity reservation is still
        // held. Handoff releases the gate only after the host object and effect
        // receipt are ready.
        .finish(py, &self.owner)
    }

    /// Look up a visible collection by name and return a handle tied to this owner. Missing
    /// collections raise CofferError. This lookup does not write to the file.
    fn collection(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<Py<Collection>> {
        // Look up the collection through a read call on this owner. The resulting
        // handle must still point to this owner's lifetime.
        execute(
            py,
            &self.owner,
            false,
            |operation| {
                let mut input = operation.input();
                // Prepare the collection name as a native string under the call's
                // input budget. No Python string is borrowed during detached
                // lookup.
                let name = convert::text(name, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(name))
            },
            |operation, database, name| operation.collection(database, name),
        )?
        // Use the same handle materializer as creation. Lookup has no write
        // outcome, while the identity's accounting still spans host allocation.
        .finish(py, &self.owner)
    }

    /// Return owned CollectionInfo dictionaries for collections visible to this owner. Each
    /// entry includes the name, IDs and settings needed to interpret its data.
    fn list_collections(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Run the native list operation with no request payload. The shared output
        // path copies owned collection dictionaries into Python while admission is
        // active.
        read_output(
            py,
            &self.owner,
            false,
            |_| Ok(()),
            |operation, database, ()| operation.list_collections(database),
        )
    }

    /// Load the collection and JSONL inputs selected by manifest_path and return an
    /// ImportSummary.
    ///
    /// Requires this owner to hold the writer lock. Validates fixed inputs before bounded
    /// commits; an error or interruption may leave earlier batches committed. on_conflict
    /// defaults to error; skip and replace select alternative handling for conflicts.
    /// Inspect any operation_outcome and verify committed records before resuming.
    #[pyo3(signature = (manifest_path, *, on_conflict=DefaultArgument::default()),
           text_signature = "($self, manifest_path, *, on_conflict='error')")]
    fn import_jsonl(
        &self,
        py: Python<'_>,
        manifest_path: &Bound<'_, PyAny>,
        on_conflict: DefaultArgument,
    ) -> PyResult<Py<PyAny>> {
        // Use the retained-output path with write observation. A Python result
        // failure can happen after the manifest import confirmed some or all
        // planned records.
        read_output(
            py,
            &self.owner,
            true,
            |operation| {
                let mut input = operation.input();
                // Validate conflict policy and path under the same operation
                // context. Shape failures return before native import starts
                // reading the manifest.
                let on_conflict = arguments::conflict(on_conflict.bind(py), &mut input)
                    .map_err(|error| shape(py, error))?;
                let path =
                    convert::text(manifest_path, &mut input).map_err(|error| shape(py, error))?;
                Ok((input.finish(path), on_conflict))
            },
            // Pass only the prepared path and typed conflict policy into native
            // code. The shared wrapper handles any committed-progress receipt at
            // handoff.
            |operation, database, (path, on_conflict)| {
                operation.import_jsonl_output(database, path, on_conflict)
            },
        )
    }

    /// Enter a context for this open owner and return the same Database. Closed, busy,
    /// forked or poisoned owners raise their corresponding SDK exception.
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<Py<Database>> {
        // Validate the owner before returning another Python reference to the
        // existing object. Entering a context creates no new database or lock.
        let py = slf.py();
        let pid = slf.owner.pid();
        let call = slf.owner.enter(false).map_err(|error| failure(py, error))?;
        // Check the interpreter return boundary before releasing admission for
        // this enter call. The returned object preserves the same owner identity.
        let result = errors::finish(py, Ok(slf.into()), None, pid);
        drop(call);
        result
    }

    /// Close the owner on leaving the context and return False so a body exception
    /// propagates. If both the body and close fail, an exception group carries both errors.
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: &Bound<'_, PyAny>,
        exc: &Bound<'_, PyAny>,
        tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        // The exception value determines whether an earlier body error exists. The
        // type and traceback are part of the Python protocol signature.
        let _ = (exc_type, tb);
        // Close even when the body failed. Returning false preserves a body
        // exception after successful close, while two failures become a cleanup
        // group.
        match self.close(py) {
            Ok(()) => Ok(false),
            Err(error) if exc.is_none() => Err(error),
            Err(error) => Err(errors::cleanup_group(py, exc, error)),
        }
    }
}

impl Database {
    // Share owner admission, budget setup and result conversion for both
    // statistics surfaces. The index selector affects only which native read runs.
    fn stats_call(&self, py: Python<'_>, index: bool) -> PyResult<Py<PyAny>> {
        // Enter without closed-state access because statistics require a live
        // database owner. Hold the gate through materialization.
        let mut call = self
            .owner
            .enter(false)
            .map_err(|error| failure(py, error))?;
        let result = (|| {
            // Create the operation using the selected database's ledger. No input
            // staging is needed for this statistics request.
            let operation = call
                .run(|database| Ok(Operation::new(database)))
                .map_err(|error| failure(py, error))?;
            // Detach the native read while retaining the owner guard. Python work
            // resumes only once the native result or error has returned.
            let output = py
                .detach(|| {
                    call.run(|database| {
                        if index {
                            operation.index_stats(database)
                        } else {
                            operation.database_stats(database)
                        }
                    })
                })
                .map_err(|error| failure(py, error))?;
            // Protect host materialization against unwind separately. A native
            // read success can still produce a resource or conversion error at
            // this stage.
            call.protect(|| output::Materialized::build(py, output, self.owner.pid(), 0))
                .map_err(|error| failure(py, error))?
                .map_err(|error| materialization_error(py, error))
        })();
        // Deliver the materialized value or the converted error through the
        // interpreter return boundary. These reads carry no mutation observation.
        match result {
            Ok(value) => value.finish(py, self.owner.pid(), None),
            Err(error) => errors::finish(py, Err(error), None, self.owner.pid()),
        }
    }
}
