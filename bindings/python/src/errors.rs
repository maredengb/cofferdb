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

create_exception!(
    cofferdb,
    CofferError,
    PyException,
    "An error reported by the native CofferDB engine."
);
create_exception!(
    cofferdb,
    SdkError,
    PyException,
    "An error at the Python ownership or materialization boundary."
);
create_exception!(
    cofferdb,
    ClosedError,
    SdkError,
    "The database owner has been closed."
);
create_exception!(
    cofferdb,
    BusyError,
    SdkError,
    "Another operation is in flight on this owner."
);
create_exception!(
    cofferdb,
    ForkedHandleError,
    SdkError,
    "An inherited owner cannot be used after fork."
);
create_exception!(
    cofferdb,
    PoisonedHandleError,
    SdkError,
    "A caught panic poisoned the owner; close it."
);
create_exception!(
    cofferdb,
    EnginePanicError,
    SdkError,
    "A native panic was caught; its payload is private."
);
create_exception!(
    cofferdb,
    ResultMaterializationError,
    SdkError,
    "Native work finished but its Python result could not be built."
);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
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
        module.dict().set_item(output::string(py, name)?, class)?;
    }
    Ok(())
}

/// Construct eagerly with fallible CPython calls. Lazy `new_err(&str)` would
/// otherwise allocate its diagnostic later, possibly outside the panic guard.
fn exception<T: PyTypeInfo>(py: Python<'_>, message: &str) -> PyResult<PyErr> {
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
    exception::<T>(py, message).unwrap_or_else(|error| error)
}

pub(crate) fn sdk(py: Python<'_>, error: &crate::owner::Failure) -> PyErr {
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
        crate::owner::Failure::Engine(_) => {
            unreachable!("engine errors use the shared core projection")
        }
    };
    let exception = match exception {
        Ok(value) => value,
        Err(error) => return error,
    };
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
    let error = exception::<CofferError>(py, "CofferDB operation failed")?;
    for (key, value) in fields.iter() {
        error.value(py).setattr(key.cast::<PyString>()?, value)?;
    }
    Ok(error)
}

pub(crate) fn conversion_shape(
    py: Python<'_>,
    error: crate::convert::Failure,
) -> Result<PyErr, cofferdb::Error> {
    match error {
        crate::convert::Failure::Type(message) => Ok(python::<PyTypeError>(py, message)),
        crate::convert::Failure::Value(message) => Ok(python::<PyValueError>(py, message)),
        crate::convert::Failure::Python(error) => Ok(error),
        crate::convert::Failure::Engine(error) => Err(error),
    }
}

fn attach_outcome(py: Python<'_>, error: &PyErr, outcome: &WriteOutcome) -> PyResult<()> {
    let value = crate::output::dictionary(py)?;
    value.set_item(
        output::string(py, "state")?,
        output::string(py, outcome.state)?,
    )?;
    if let Some(count) = outcome.confirmed_committed_batches {
        value.set_item(
            output::string(py, "confirmed_committed_batches")?,
            output::unsigned(py, count as u64)?,
        )?;
    }
    if let Some(count) = outcome.confirmed_committed_records {
        value.set_item(
            output::string(py, "confirmed_committed_records")?,
            output::unsigned(py, count as u64)?,
        )?;
    }
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
    error
        .value(py)
        .setattr(output::name(py, "operation_outcome")?, value)
}

pub(crate) fn cleanup_group(py: Python<'_>, body: &Bound<'_, PyAny>, cleanup: PyErr) -> PyErr {
    let body = match body.cast::<PyBaseException>() {
        Ok(body) => body,
        Err(_) => {
            let error = python::<PyTypeError>(py, "context body must be a BaseException");
            error.set_cause(py, Some(cleanup));
            return error;
        }
    };
    let result: PyResult<PyErr> = (|| {
        let message = output::string(py, "context body and database close both failed")?;
        let values = output::list(py, 2)?;
        values.set_item(0, body)?;
        values.set_item(1, cleanup.value(py))?;
        // SAFETY: arguments and the class remain live under the GIL, and the
        // variadic list ends with a null PyObject pointer as required by CPython.
        let group = unsafe {
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
    let mut result = match py.check_signals() {
        Ok(()) => result,
        Err(signal) => {
            if let Err(original) = result {
                signal.set_cause(py, Some(original));
            }
            Err(signal)
        }
    };
    if pid != std::process::id() {
        let error = sdk(py, &crate::owner::Failure::Forked);
        if let Err(original) = result {
            error.set_cause(py, Some(original));
        }
        result = Err(error);
    }
    match result {
        Err(error) => {
            if let Some(outcome) = outcome {
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
