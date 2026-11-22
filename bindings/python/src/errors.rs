//! Python exception classes and attachment of confirmed native outcomes.

use crate::output;
use cofferdb::binding::WriteOutcome;
use pyo3::create_exception;
use pyo3::exceptions::{
    PyBaseException, PyBaseExceptionGroup, PyException, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::PyTypeInfo;

// Keep these two exception families as siblings. An except CofferError handler
// relies on the engine's structured fields; admission failures and failures
// constructing a Python result have different diagnostics.
create_exception!(
    cofferdb,
    CofferError,
    PyException,
    "Error reported by the native engine. Inspect code (machine identifier), message \
     (explanation), detail (structured context), recoverable (engine guidance) and \
     suggested_action (optional advice). For interrupted writes, inspect any attached outcome \
     before deciding whether to retry."
);
// Callers using ordinary Python argument validation can also receive TypeError or
// ValueError. SdkError is a common catch point for the extension's own
// diagnostics, not the root of every exception it can return.
create_exception!(
    cofferdb,
    SdkError,
    PyException,
    "Base class for errors at the Python ownership and result-construction boundary. Catch a \
     concrete subclass for the relevant recovery action. Engine errors are exposed separately \
     as CofferError."
);
// The Python-facing subclasses make the owner's recovery choices independently
// catchable. Keep their inheritance and the registrations below in step with the
// package's public type stub.
create_exception!(
    cofferdb,
    ClosedError,
    SdkError,
    "The owner was closed. Obtain a new owner and collection handle before continuing. \
     sdk_reason is closed."
);
// The busy gate is shared by a database and its collections. An operation using a
// different collection on the same database can therefore raise this class.
create_exception!(
    cofferdb,
    BusyError,
    SdkError,
    "Another call is active on this owner. Wait for it to finish before trying again. \
     sdk_reason is busy."
);
// A forked handle and a busy handle can both reject admission before native work.
// Only the busy condition can be resolved by waiting for this same owner.
create_exception!(
    cofferdb,
    ForkedHandleError,
    SdkError,
    "This handle belongs to the parent process. Open a fresh owner in the child, subject to the \
     file's writer lock. sdk_reason is forked_handle."
);
// Distinguish the call that panicked from later calls on the poisoned owner. The
// latter consult stored owner state without re-entering the engine.
create_exception!(
    cofferdb,
    PoisonedHandleError,
    SdkError,
    "A previous native panic poisoned this owner. Close it before opening a fresh owner. \
     sdk_reason is poisoned."
);
// Open or create can also panic before any Python owner is returned. The public
// diagnosis must make sense even when there is no handle for the caller to close.
create_exception!(
    cofferdb,
    EnginePanicError,
    SdkError,
    "The native call panicked. Its payload is hidden; close the affected owner if present and \
     verify any writes. sdk_reason is engine_panic."
);
// This class can be raised after a native write has returned. Catching an SDK
// error alone therefore does not establish that the operation is safe to repeat.
create_exception!(
    cofferdb,
    ResultMaterializationError,
    SdkError,
    "Native work could not be delivered as a Python result. A write or maintenance operation \
     may already have taken effect. Inspect any attached outcome and verify actual state before \
     retrying."
);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    // The public Python package re-exports these actual classes from the
    // extension. Register the same type objects used to construct PyErr so
    // catching a public name also catches the native exception.
    for (name, class) in [
        ("CofferError", py.get_type::<CofferError>()),
        ("SdkError", py.get_type::<SdkError>()),
        ("ClosedError", py.get_type::<ClosedError>()),
        ("BusyError", py.get_type::<BusyError>()),
        ("ForkedHandleError", py.get_type::<ForkedHandleError>()),
        ("PoisonedHandleError", py.get_type::<PoisonedHandleError>()),
        ("EnginePanicError", py.get_type::<EnginePanicError>()),
        (
            "ResultMaterializationError",
            py.get_type::<ResultMaterializationError>(),
        ),
    ] {
        // A failed name allocation or dict insertion aborts module initialization.
        // Publishing a partial hierarchy would let imports succeed while some
        // native errors are unavailable by name.
        module.dict().set_item(output::string(py, name)?, class)?;
    }
    Ok(())
}

/// Construct eagerly with fallible CPython calls. Lazy `new_err(&str)` would
/// otherwise allocate its diagnostic later, possibly outside the panic guard.
fn exception<T: PyTypeInfo>(py: Python<'_>, message: &str) -> PyResult<PyErr> {
    // Allocate the diagnostic text before invoking the exception type. If Python
    // cannot allocate either object, the caller receives that active Python
    // failure while still inside the conversion boundary.
    let message = output::string(py, message)?;
    // SAFETY: both owned objects remain live under the GIL for this call. The
    // registered class is a BaseException subclass; a null return has PyErr set.
    let value = unsafe {
        Bound::from_owned_ptr_or_err(
            py,
            pyo3::ffi::PyObject_CallOneArg(py.get_type::<T>().as_ptr(), message.as_ptr()),
        )
    }?;
    Ok(PyErr::from_value(value))
}

pub(crate) fn python<T: PyTypeInfo>(py: Python<'_>, message: &str) -> PyErr {
    // Exception construction itself is fallible. Preserve its Python error as the
    // returned exception instead of attempting a second diagnostic allocation.
    exception::<T>(py, message).unwrap_or_else(|error| error)
}

pub(crate) fn sdk(py: Python<'_>, error: &crate::owner::Failure) -> PyErr {
    // Each owner-state failure has both a catchable exception class and a stable
    // reason string. Changing the displayed message must not change the reason
    // exposed to Python.
    let (exception, reason) = match error {
        crate::owner::Failure::Closed => {
            (exception::<ClosedError>(py, "database is closed"), "closed")
        }
        crate::owner::Failure::Busy => (
            exception::<BusyError>(py, "database has an operation in flight"),
            "busy",
        ),
        crate::owner::Failure::Forked => (
            exception::<ForkedHandleError>(py, "open a new database handle in the child process"),
            "forked_handle",
        ),
        crate::owner::Failure::Poisoned => (
            exception::<PoisonedHandleError>(
                py,
                "database owner is poisoned; only close is allowed",
            ),
            "poisoned",
        ),
        crate::owner::Failure::Panic => (
            exception::<EnginePanicError>(
                py,
                "native operation panicked; close the poisoned owner",
            ),
            "engine_panic",
        ),
        // Native engine errors need the machine-readable core envelope. The caller
        // dispatches them before entering this mapping for SDK state and panic
        // failures.
        crate::owner::Failure::Engine(_) => {
            unreachable!("engine errors use the shared core projection")
        }
    };
    // If the class constructor failed, the intended SDK exception does not exist.
    // Return the construction error directly before trying to attach attributes to
    // a different exception.
    let exception = match exception {
        Ok(value) => value,
        Err(error) => return error,
    };
    // Attach sdk_reason only after the intended exception was constructed. If even
    // setting this attribute fails, retain the intended exception as the cause of
    // the new Python failure.
    let attached = (|| {
        exception
            .value(py)
            .setattr(output::name(py, "sdk_reason")?, output::string(py, reason)?)
    })();
    if let Err(error) = attached {
        error.set_cause(py, Some(exception));
        return error;
    }
    exception
}

/// The dict has already been built from the core's five-field projection.
pub(crate) fn engine(py: Python<'_>, fields: &Bound<'_, PyDict>) -> PyResult<PyErr> {
    // The Python exception message for the engine failure is separate from the
    // five fields built by the core. Copy those attributes from the prepared dict
    // so Python observes the engine's code, details and recovery guidance.
    let error = exception::<CofferError>(py, "CofferDB operation failed")?;
    for (key, value) in fields.iter() {
        // Assign the already materialized Python objects themselves. In particular
        // detail remains a dictionary and recoverable remains a bool, preserving
        // field types without formatting another JSON string.
        error.value(py).setattr(key.cast::<PyString>()?, value)?;
    }
    Ok(error)
}

pub(crate) fn conversion_shape(
    py: Python<'_>,
    error: crate::convert::Failure,
) -> Result<PyErr, cofferdb::Error> {
    // Shape errors are part of the Python calling contract. Preserve exceptions
    // already raised by Python and defer structured engine errors to the core
    // projection that supplies their machine-readable fields.
    match error {
        // The Result here distinguishes an already chosen Python exception from a
        // native error that still needs conversion. Its Ok variant does not
        // signify that the argument was accepted.
        crate::convert::Failure::Type(message) => Ok(python::<PyTypeError>(py, message)),
        crate::convert::Failure::Value(message) => Ok(python::<PyValueError>(py, message)),
        crate::convert::Failure::Python(error) => Ok(error),
        crate::convert::Failure::Engine(error) => Err(error),
    }
}

fn attach_outcome(py: Python<'_>, error: &PyErr, outcome: &WriteOutcome) -> PyResult<()> {
    // Construct the whole outcome dictionary before setting an attribute on the
    // selected exception. If building any of its fields fails, the caller can
    // chain the original exception under that reporting failure.
    let value = crate::output::dictionary(py)?;
    value.set_item(
        output::string(py, "state")?,
        output::string(py, outcome.state)?,
    )?;
    // A missing counter means the engine supplied no confirmation for that field.
    // Keep it distinct from a confirmed count of zero when projecting the write
    // outcome to Python.
    if let Some(count) = outcome.confirmed_committed_batches {
        value.set_item(
            output::string(py, "confirmed_committed_batches")?,
            output::unsigned(py, count as u64)?,
        )?;
    }
    // Use the native committed-record counter without recomputing it from user
    // rows. A replacement can commit a tombstone and a new value for one logical
    // input row.
    if let Some(count) = outcome.confirmed_committed_records {
        value.set_item(
            output::string(py, "confirmed_committed_records")?,
            output::unsigned(py, count as u64)?,
        )?;
    }
    // Report the last confirmed batch ID separately from the commit record's
    // sequence number. The state may still be unknown for a later append while
    // both fields identify an earlier confirmed batch.
    if let Some(id) = outcome.last_confirmed_commit_id {
        value.set_item(
            output::string(py, "last_confirmed_commit_id")?,
            output::unsigned(py, id)?,
        )?;
    }
    if let Some(id) = outcome.last_confirmed_commit_sequence_no {
        value.set_item(
            output::string(py, "last_confirmed_commit_sequence_no")?,
            output::unsigned(py, id)?,
        )?;
    }
    // Set one operation_outcome on the final exception, which may be a signal or a
    // projection failure. The observed counters need not match what the exception
    // class alone suggests.
    error
        .value(py)
        .setattr(output::name(py, "operation_outcome")?, value)
}

// Preserve both failures from context-manager exit. The exception raised by the
// body is first, and the error from database close follows it.
pub(crate) fn cleanup_group(py: Python<'_>, body: &Bound<'_, PyAny>, cleanup: PyErr) -> PyErr {
    // Validate the supplied body value before calling the exception-group
    // constructor. Direct calls to __exit__ can provide something that Python's
    // context protocol would not.
    let body = match body.cast::<PyBaseException>() {
        Ok(body) => body,
        Err(_) => {
            let error = python::<PyTypeError>(py, "context body must be a BaseException");
            // Attach the cleanup failure to the type error so manual misuse cannot
            // hide why closing the database failed.
            error.set_cause(py, Some(cleanup));
            return error;
        }
    };
    // Build the exception group under the active interpreter with fallible
    // allocations. If any step fails, the fallback retains references to the
    // original errors.
    let result: PyResult<PyErr> = (|| {
        let message = output::string(py, "context body and database close both failed")?;
        // Keep the original exception objects in argument order. Constructing a
        // new group preserves their traceback and other instance metadata.
        let values = output::list(py, 2)?;
        values.set_item(0, body)?;
        values.set_item(1, cleanup.value(py))?;
        // Call the base exception-group type because either failure can be a
        // BaseException outside the ordinary Exception hierarchy.
        // SAFETY: arguments and the class remain live under the GIL, and the
        // variadic list ends with a null PyObject pointer as required by CPython.
        let group = unsafe {
            // Adopt the owned reference returned by the constructor. A null result
            // becomes the Python error already set by CPython, which the fallback
            // can enrich.
            Bound::from_owned_ptr_or_err(
                py,
                pyo3::ffi::PyObject_CallFunctionObjArgs(
                    py.get_type::<PyBaseExceptionGroup>().as_ptr(),
                    message.as_ptr(),
                    values.as_ptr(),
                    std::ptr::null_mut::<pyo3::ffi::PyObject>(),
                ),
            )
        }?;
        Ok(PyErr::from_value(group))
    })();
    // Use the group when construction succeeds. A failure in error reporting
    // becomes the outer exception while the two original errors remain linked
    // below.
    match result {
        Ok(group) => group,
        Err(error) => {
            // Preserve both existing failures even if a group cannot be built.
            // SAFETY: both values are confirmed BaseExceptions. CPython steals
            // only this incremented reference to the original body exception.
            unsafe {
                pyo3::ffi::PyException_SetContext(
                    error.value(py).as_ptr(),
                    body.clone().into_ptr(),
                );
            }
            // Use cause for the close error and context for the original body
            // error. Both links remain available even without a grouped payload.
            error.set_cause(py, Some(cleanup));
            error
        }
    }
}

/// Called after native completion/cleanup and after preserving its observation.
pub(crate) fn finish<T>(
    py: Python<'_>,
    result: PyResult<T>,
    outcome: Option<&WriteOutcome>,
    pid: u32,
) -> PyResult<T> {
    // Signal handlers run Python and may even fork. Check the PID afterward,
    // preserving earlier failures as causes before attempting to attach the
    // write outcome to the selected exception.
    // Check pending signals after native cleanup regardless of the current result.
    // A selected signal takes precedence and carries any earlier operation failure
    // as its cause.
    let mut result = match py.check_signals() {
        Ok(()) => result,
        Err(signal) => {
            if let Err(original) = result {
                signal.set_cause(py, Some(original));
            }
            Err(signal)
        }
    };
    // Give a changed PID precedence over the result selected by signal handling. A
    // Python signal handler can fork before this boundary returns.
    if pid != std::process::id() {
        let error = sdk(py, &crate::owner::Failure::Forked);
        if let Err(original) = result {
            error.set_cause(py, Some(original));
        }
        result = Err(error);
    }
    // Attach the preserved write observation only to the selected error.
    // Successful return values already follow their own result schema.
    match result {
        Err(error) => {
            if let Some(outcome) = outcome {
                // Attaching a receipt can itself allocate and fail after native
                // completion. Surface that failure with the selected exception as
                // its cause so a Python-side reporting error cannot erase the
                // earlier diagnostic.
                if let Err(attachment) = attach_outcome(py, &error, outcome) {
                    attachment.set_cause(py, Some(error));
                    return Err(attachment);
                }
            }
            Err(error)
        }
        Ok(value) => Ok(value),
    }
}
