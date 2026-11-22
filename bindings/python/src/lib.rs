use pyo3::prelude::*;

mod api;
mod convert;
mod errors;
mod output;
mod owner;
mod panic_guard;
mod requests;
#[cfg(test)]
mod test_alloc;

// Declare the extension's dependence on the GIL. The converters borrow CPython
// object storage while attached; methods that detach do so only after turning host
// arguments into owned native inputs.
#[pymodule(gil_used = true)]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Set up the process-wide panic hook before exposing callable objects. The
    // guard tracks each thread separately and forwards panics outside guarded
    // extension operations to the previous hook.
    panic_guard::install();
    // Register the same exception and class objects that the Python package
    // re-exports. A failure setting up any of them must propagate from module
    // initialization so callers cannot import a partial interface.
    errors::register(module)?;
    module.add_function(wrap_pyfunction!(api::check, module)?)?;
    module.add_class::<api::Database>()?;
    module.add_class::<api::Collection>()?;
    // Read the extension package version and linked engine version from their own
    // crates. Package import checks the former against the installed Python files
    // and exposes the latter independently.
    module.dict().set_item(
        output::string(module.py(), "__version__")?,
        output::string(module.py(), env!("CARGO_PKG_VERSION"))?,
    )?;
    module.dict().set_item(
        output::string(module.py(), "engine_version")?,
        output::string(module.py(), cofferdb::binding::engine_version())?,
    )?;
    Ok(())
}
