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

#[pymodule(gil_used = true)]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    panic_guard::install();
    errors::register(module)?;
    module.add_class::<api::Database>()?;
    module.add_class::<api::Collection>()?;
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
