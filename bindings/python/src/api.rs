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
#[cfg(test)]
mod tests;
pub(crate) use collection::Collection;

/// Preserve omission without constructing Python defaults during argument unpacking.
/// Explicit None remains Some and is checked by the ordinary input converter.
#[derive(Default)]
struct DefaultArgument(Option<Py<PyAny>>);

impl FromPyObject<'_, '_> for DefaultArgument {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self(Some(object.to_owned().unbind())))
    }
}

impl DefaultArgument {
    fn bind<'py>(&self, py: Python<'py>) -> Option<&Bound<'py, PyAny>> {
        self.0.as_ref().map(|value| value.bind(py))
    }
}

struct Execution<T> {
    call: owner::Call,
    result: PyResult<T>,
    outcome: Option<WriteOutcome>,
    pid: u32,
}

fn not_started() -> WriteOutcome {
    WriteOutcome {
        state: "not_started",
        confirmed_committed_batches: None,
        confirmed_committed_records: None,
        last_confirmed_commit_id: None,
        last_confirmed_commit_sequence_no: None,
    }
}

fn execute<I: Send, T: Send>(
    py: Python<'_>,
    owner: &Arc<owner::Owner>,
    writing: bool,
    prepare: impl FnOnce(&Operation) -> PyResult<I>,
    run: impl FnOnce(Operation, &cofferdb::Database, I) -> cofferdb::Result<T> + Send,
) -> PyResult<Execution<T>> {
    let pid = owner.pid();
    let mut call = owner.enter(false).map_err(|error| {
        errors::finish::<()>(
            py,
            Err(failure(py, error)),
            writing.then(not_started).as_ref(),
            pid,
        )
        .unwrap_err()
    })?;
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
    let observation = operation.observation();
    let prepared = match call.protect(|| prepare(&operation)) {
        Ok(result) => result,
        Err(error) => Err(failure(py, error)),
    };
    let (result, outcome) = match prepared {
        Ok(input) if pid == std::process::id() => {
            let (native, outcome) = py.detach(|| {
                let result = call.run(|database| run(operation, database, input));
                // This snapshot precedes attach, Python allocations and signals.
                let outcome = writing.then(|| observation.write_outcome());
                (result, outcome)
            });
            (native.map_err(|error| failure(py, error)), outcome)
        }
        Ok(_) => (
            Err(errors::sdk(py, &owner::Failure::Forked)),
            writing.then(not_started),
        ),
        Err(error) => (Err(error), writing.then(not_started)),
    };
    Ok(Execution {
        call,
        result,
        outcome,
        pid,
    })
}

impl Execution<Output> {
    fn finish(mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = self.result.and_then(|value| {
            self.call
                .protect(|| output::Materialized::build(py, value, self.pid, 0))
                .map_err(|error| failure(py, error))?
                .map_err(|error| materialization_error(py, error))
        });
        match result {
            Ok(value) => value.finish(py, self.pid, self.outcome.as_ref()),
            Err(error) => errors::finish(py, Err(error), self.outcome.as_ref(), self.pid),
        }
    }
}

impl Execution<cofferdb::binding::CollectionIdentity> {
    fn finish(mut self, py: Python<'_>, owner: &Arc<owner::Owner>) -> PyResult<Py<Collection>> {
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
    lease
        .reserve_host(256)
        .map_err(|error| materialization_error(py, output::Failure::Engine(error)))?;
    check_pid(py, pid)?;
    let collection = Py::new(
        py,
        Collection {
            owner: Arc::clone(owner),
            collection_id,
            name,
        },
    )?;
    let result = errors::finish(py, Ok(collection), outcome, pid);
    drop(lease);
    result
}

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
    match Operation::standalone().error(error) {
        Ok(value) => match output::Materialized::build(py, value, std::process::id(), 1024) {
            Ok(value) => value.exception(py).unwrap_or_else(|error| error),
            Err(error) => materialization_error(py, error),
        },
        Err(_) => errors::python::<errors::ResultMaterializationError>(
            py,
            "the native error could not be materialized",
        ),
    }
}

fn failure(py: Python<'_>, error: owner::Failure) -> PyErr {
    match error {
        owner::Failure::Engine(error) => engine_error(py, error),
        other => errors::sdk(py, &other),
    }
}

fn shape(py: Python<'_>, error: convert::Failure) -> PyErr {
    match errors::conversion_shape(py, error) {
        Ok(error) => error,
        Err(error) => engine_error(py, error),
    }
}

fn materialization_error(py: Python<'_>, error: output::Failure) -> PyErr {
    let cause = match error {
        output::Failure::Python(error) => error,
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
    error.set_cause(py, Some(cause));
    error
}

fn schema<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    fields: &[&str],
) -> PyResult<&'a Bound<'py, PyDict>> {
    let object = convert::dict(value).map_err(|error| shape(value.py(), error))?;
    for (key, _) in object.iter() {
        if !key.is_exact_instance_of::<PyString>() {
            return Err(errors::python::<PyTypeError>(
                value.py(),
                "configuration keys must be built-in str values",
            ));
        }
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
    convert::boolean(value).map_err(|error| shape(value.py(), error))
}
fn cache(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let number = convert::unsigned(value).map_err(|error| shape(value.py(), error))?;
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
        if let Some(value) =
            crate::output::lookup(object, "cache_budget_mb")?.filter(|value| !value.is_none())
        {
            config.cache_budget_mb = Some(cache(&value)?);
        }
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
        if let Some(value) = crate::output::lookup(object, "io_mode")? {
            options.io_mode = io_mode(&value, input)?;
        }
    }
    Ok(options)
}

#[pyclass(module = "cofferdb", frozen)]
/// A synchronous native database owner. Construct with create or open.
///
/// Use a context manager or close explicitly. Retained collections do not
/// prevent explicit close from releasing this owner's native database.
pub(crate) struct Database {
    owner: Arc<owner::Owner>,
}

fn check_pid(py: Python<'_>, pid: u32) -> PyResult<()> {
    if pid == std::process::id() {
        Ok(())
    } else {
        Err(errors::sdk(py, &owner::Failure::Forked))
    }
}

#[pymethods]
impl Database {
    #[staticmethod]
    #[pyo3(signature = (path, *, config=None))]
    /// Create a new file using core defaults and return its writer owner.
    ///
    /// path must be a built-in str; an existing destination is not overwritten.
    fn create(
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pid = std::process::id();
        let result = panic_guard::catch(|| {
            let operation = Operation::standalone();
            let mut input = operation.input();
            let path = convert::text(path, &mut input).map_err(|error| shape(py, error))?;
            let config = database_config(config, &mut input)?;
            check_pid(py, pid)?;
            let native = py
                .detach(|| operation.create(input.finish((path, config))))
                .map_err(|error| engine_error(py, error))?;
            Ok(Self {
                owner: owner::Owner::new(native),
            })
        });
        let result = result.unwrap_or_else(|_| Err(errors::sdk(py, &owner::Failure::Panic)));
        errors::finish(py, result, None, pid)
    }

    #[staticmethod]
    #[pyo3(signature = (path, *, read_only=false, options=None))]
    /// Open an existing file, optionally using a fixed read-only view.
    ///
    /// Nullable option overrides inherit core defaults. This never creates a
    /// missing file or silently upgrades a read-only owner to a writer.
    fn open(
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = boolean)] read_only: bool,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pid = std::process::id();
        let result = panic_guard::catch(|| {
            let operation = Operation::standalone();
            let mut input = operation.input();
            let path = convert::text(path, &mut input).map_err(|error| shape(py, error))?;
            let options = open_options(options, &mut input, read_only)?;
            check_pid(py, pid)?;
            let native = py
                .detach(|| operation.open(input.finish((path, options))))
                .map_err(|error| engine_error(py, error))?;
            Ok(Self {
                owner: owner::Owner::new(native),
            })
        });
        let result = result.unwrap_or_else(|_| Err(errors::sdk(py, &owner::Failure::Panic)));
        errors::finish(py, result, None, pid)
    }

    /// Release the native database even when collection objects are retained.
    ///
    /// Repeated close is harmless; in-flight work raises BusyError. A forked
    /// inherited owner raises ForkedHandleError before touching native locks.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut call = self.owner.enter(true).map_err(|error| failure(py, error))?;
        let result = py
            .detach(|| call.close())
            .map_err(|error| failure(py, error));
        let result = errors::finish(py, result, None, self.owner.pid());
        drop(call);
        result
    }

    /// Return an independently owned projection of the core database statistics.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.stats_call(py, false)
    }
    /// Return core index statistics without forcing an index rebuild.
    fn index_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.stats_call(py, true)
    }

    /// Create a writable collection from a built-in configuration dict.
    ///
    /// Required keys are name, embedding_dim and distance_metric. An omitted
    /// tokenizer uses the core immutable default_en configuration.
    fn create_collection(
        &self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Collection>> {
        execute(
            py,
            &self.owner,
            true,
            |operation| arguments::collection(config, operation).map_err(|error| shape(py, error)),
            |operation, database, config| operation.create_collection(database, config),
        )?
        .finish(py, &self.owner)
    }

    /// Access an existing collection by name; never create it implicitly.
    fn collection(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<Py<Collection>> {
        execute(
            py,
            &self.owner,
            false,
            |operation| {
                let mut input = operation.input();
                let name = convert::text(name, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(name))
            },
            |operation, database, name| operation.collection(database, name),
        )?
        .finish(py, &self.owner)
    }

    /// Materialize the visible collection catalog using core fields and budgets.
    fn list_collections(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            false,
            |_| Ok(()),
            |operation, database, ()| operation.list_collections(database),
        )
    }

    #[pyo3(signature = (manifest_path, *, on_conflict=DefaultArgument::default()),
           text_signature = "($self, manifest_path, *, on_conflict='error')")]
    /// Validate a manifest's fixed inputs, then import with bounded core commits.
    ///
    /// The manifest selects the collection. Import is not whole-file atomic.
    /// On failure, operation_outcome preserves confirmed core commit progress;
    /// counts describe core records, including collection and tombstone records.
    fn import_jsonl(
        &self,
        py: Python<'_>,
        manifest_path: &Bound<'_, PyAny>,
        on_conflict: DefaultArgument,
    ) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            true,
            |operation| {
                let mut input = operation.input();
                let on_conflict = arguments::conflict(on_conflict.bind(py), &mut input)
                    .map_err(|error| shape(py, error))?;
                let path =
                    convert::text(manifest_path, &mut input).map_err(|error| shape(py, error))?;
                Ok((input.finish(path), on_conflict))
            },
            |operation, database, (path, on_conflict)| {
                operation.import_jsonl_output(database, path, on_conflict)
            },
        )
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<Py<Database>> {
        let py = slf.py();
        let pid = slf.owner.pid();
        let call = slf.owner.enter(false).map_err(|error| failure(py, error))?;
        let result = errors::finish(py, Ok(slf.into()), None, pid);
        drop(call);
        result
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: &Bound<'_, PyAny>,
        exc: &Bound<'_, PyAny>,
        tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let _ = (exc_type, tb);
        match self.close(py) {
            Ok(()) => Ok(false),
            Err(error) if exc.is_none() => Err(error),
            Err(error) => Err(errors::cleanup_group(py, exc, error)),
        }
    }
}

impl Database {
    fn stats_call(&self, py: Python<'_>, index: bool) -> PyResult<Py<PyAny>> {
        let mut call = self
            .owner
            .enter(false)
            .map_err(|error| failure(py, error))?;
        let result = (|| {
            let operation = call
                .run(|database| Ok(Operation::new(database)))
                .map_err(|error| failure(py, error))?;
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
            call.protect(|| output::Materialized::build(py, output, self.owner.pid(), 0))
                .map_err(|error| failure(py, error))?
                .map_err(|error| materialization_error(py, error))
        })();
        match result {
            Ok(value) => value.finish(py, self.owner.pid(), None),
            Err(error) => errors::finish(py, Err(error), None, self.owner.pid()),
        }
    }
}
