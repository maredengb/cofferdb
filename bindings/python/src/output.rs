//! One aggregate estimate and iterative construction of independent host values.

use cofferdb::binding::{Output, Stack, WriteOutcome};
use cofferdb::JsonValue;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};

// Use fallible CPython constructors: several convenience constructors panic
// on a null allocation result instead of returning the pending MemoryError.
pub(crate) fn string<'py>(py: Python<'py>, value: &str) -> PyResult<Bound<'py, PyAny>> {
    // Both passes use Unicode scalar indices. A non-ASCII Rust str may use several
    // UTF-8 bytes for one character; its byte length is not the number of slots
    // required by PyUnicode_New.
    let (count, highest) = string_layout(value);
    // SAFETY: the GIL is held; the core bound makes count fit Py_ssize_t. This
    // creates the final Unicode width once, without a UTF-8 decoder's widening
    // buffers. The new reference is private until every character is filled.
    let output = unsafe {
        Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyUnicode_New(count as isize, highest))
    }?;
    // The largest scalar measured above selects the storage width for the entire
    // object. Replaying the same immutable Rust text lets each write use its
    // character index without an intermediate Python bytes object.
    for (index, character) in value.chars().enumerate() {
        // SAFETY: this uniquely owned exact Unicode is not exposed, index is
        // in bounds, and each scalar is at most the admitted maximum codepoint.
        if unsafe {
            pyo3::ffi::PyUnicode_WriteChar(output.as_ptr(), index as isize, u32::from(character))
        } < 0
        {
            // Preserve the exception already set by the failed write. Dropping the
            // local Bound releases the partially initialized string before it can
            // reach the caller.
            return Err(PyErr::fetch(py));
        }
    }
    Ok(output)
}

pub(crate) fn name<'py>(
    py: Python<'py>,
    value: &str,
) -> PyResult<Bound<'py, pyo3::types::PyString>> {
    string(py, value)?.cast_into().map_err(PyErr::from)
}

/// A fixed field name can still require a Python allocation. Keep that failure
/// fallible rather than using the infallible Rust-str conversion convenience.
pub(crate) fn lookup<'py>(
    object: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // Return Ok(None) for a missing field. Failures constructing the key or
    // performing the lookup remain Python exceptions, so schema validation can
    // report absence only after a successful lookup.
    object.get_item(string(object.py(), key)?)
}

pub(crate) fn dictionary(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    // SAFETY: the GIL is held and a successful PyDict_New owns an exact dict.
    unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyDict_New()) }
        .and_then(|value| value.cast_into::<PyDict>().map_err(PyErr::from))
}

pub(crate) fn list(py: Python<'_>, length: usize) -> PyResult<Bound<'_, PyList>> {
    // SAFETY: the GIL is held and the core bound makes length fit Py_ssize_t.
    // PyList_New returns a new reference or NULL. Initialize its empty slots
    // with owned None references before handing the list to Python code.
    let value =
        unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyList_New(length as isize)) }?;
    let value = value.cast_into::<PyList>().map_err(PyErr::from)?;
    // Fill each slot with an owned None before returning the list to the tree
    // constructor. Interpreter inspection can safely encounter slots that the
    // traversal has yet to replace.
    for index in 0..length {
        value.set_item(index, py.None())?;
    }
    Ok(value)
}

fn integer(py: Python<'_>, value: i64) -> PyResult<Bound<'_, PyAny>> {
    // SAFETY: this is a scalar call with the GIL and returns an owned reference.
    unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyLong_FromLongLong(value)) }
}

// Commit identifiers and sequence numbers can exceed i64::MAX. Use CPython's
// unsigned constructor so projecting a confirmed write does not reinterpret its
// identifier as negative.
pub(crate) fn unsigned(py: Python<'_>, value: u64) -> PyResult<Bound<'_, PyAny>> {
    // SAFETY: with the GIL held, CPython returns a new integer reference or
    // NULL with an error. The Bound constructor handles both outcomes.
    unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyLong_FromUnsignedLongLong(value)) }
}

pub(crate) fn float(py: Python<'_>, value: f64) -> PyResult<Bound<'_, PyAny>> {
    // SAFETY: this is a scalar call with the GIL and returns an owned reference.
    unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyFloat_FromDouble(value)) }
}

#[derive(Debug)]
pub(crate) enum Failure {
    // Core reservation or traversal errors retain their engine diagnosis for the
    // result-conversion layer.
    Engine(cofferdb::Error),
    // Keep the actual host exception so conversion can preserve MemoryError and
    // chain other Python failures.
    Python(PyErr),
    // A PID change during construction invalidates the in-flight operation in this
    // process, even if the partially built Python object itself is still reachable
    // here.
    Forked,
}
impl From<cofferdb::Error> for Failure {
    fn from(error: cofferdb::Error) -> Self {
        Self::Engine(error)
    }
}
impl From<PyErr> for Failure {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}
type Result<T> = std::result::Result<T, Failure>;

fn check_process(pid: u32) -> Result<()> {
    // The saved PID belongs to the operation that produced the native tree.
    // Compare it again at allocation and delivery boundaries so a child cannot
    // finish a result on behalf of the parent.
    if pid != std::process::id() {
        return Err(Failure::Forked);
    }
    Ok(())
}

/// Include retained containers and direct final-width Unicode allocations.
/// This is a conservative construction estimate, not a process RSS limit.
fn estimate(output: &Output) -> Result<u128> {
    enum Children<'a> {
        Array(std::slice::Iter<'a, JsonValue>),
        Object(std::collections::btree_map::Iter<'a, String, JsonValue>),
    }
    // Walk the borrowed native tree with an operation-charged stack of iterators.
    // Only live ancestor positions need traversal workspace, even though every
    // value reached contributes to the eventual Python result.
    let mut stack: Stack<Children<'_>> = output.stack()?;
    let mut pending = Some(output.value());
    // The result retains every completed child, so container and Unicode sizes
    // accumulate. String conversion writes into the final-width object. Any
    // per-string scratch would peak at one conversion because the traversal visits
    // one child at a time.
    let mut retained = 0u128;
    let mut transient = 0u128;
    loop {
        if let Some(value) = pending.take() {
            match value {
                // None and bool reuse CPython singleton objects. An enclosing list
                // or dictionary still needs its own slots for references to them;
                // those slots are covered by the container allowance.
                JsonValue::Null | JsonValue::Bool(_) => (),
                JsonValue::Number(_) => retained += 64,
                JsonValue::String(value) => {
                    let (bytes, temporary) = string_estimate(value);
                    retained += bytes;
                    transient = transient.max(temporary);
                }
                // Account for the list object and its element slots before
                // visiting children. Traversal charges each child's Python
                // representation separately, so nesting cannot hide the cost of a
                // large subtree.
                JsonValue::Array(values) => {
                    retained += 128 + 16 * values.len() as u128;
                    stack.push(Children::Array(values.iter()))?;
                }
                // Dictionary entries require room for both a key and a value. This
                // allowance covers table storage; the Unicode storage for each
                // actual key is added as its entry is visited.
                JsonValue::Object(values) => {
                    retained += 256 + 96 * values.len() as u128;
                    stack.push(Children::Object(values.iter()))?;
                }
            }
        }
        let Some(children) = stack.last_mut() else {
            break;
        };
        pending = match children {
            Children::Array(values) => values.next(),
            Children::Object(values) => values.next().map(|(key, value)| {
                // Keys do not occur as JsonValue children. Account for their
                // final-width Python text here so long or non-ASCII property names
                // cannot bypass host admission.
                let (bytes, temporary) = string_estimate(key);
                retained += bytes;
                transient = transient.max(temporary);
                value
            }),
        };
        // Exhaustion finishes only the current iterator. Keep its parent's
        // iterator on the stack to resume at the next sibling; an empty stack
        // means the root's entire representation was measured.
        if pending.is_none() {
            stack.pop();
        }
    }
    Ok(retained + transient)
}

fn string_estimate(value: &str) -> (u128, u128) {
    let (count, highest) = string_layout(value);
    // CPython uses one code-unit width for the whole compact Unicode string. A
    // single code point above U+FFFF therefore increases the storage estimate for
    // every character in that string.
    let width = if highest <= 0xff {
        1
    } else if highest <= 0xffff {
        2
    } else {
        4
    };
    // Include an object allowance in addition to the final character buffer. The
    // direct final-width writer uses no separately sized intermediate string,
    // hence the zero transient component.
    (128 + count as u128 * width, 0)
}

fn string_layout(value: &str) -> (usize, u32) {
    // Count Unicode scalar values, independently of their Rust UTF-8 byte lengths.
    // The estimator and the CPython allocator both use this count and maximum code
    // point so their layout decisions agree.
    let mut count = 0usize;
    let mut highest = 0u32;
    for character in value.chars() {
        count += 1;
        highest = highest.max(u32::from(character));
    }
    (count, highest)
}

enum Frame<'a, 'py> {
    Array {
        // Iterator over the borrowed native array with each child's eventual list
        // index.
        source: std::iter::Enumerate<std::slice::Iter<'a, JsonValue>>,
        // Destination index saved when the current child was scheduled, used after
        // any nested frames have completed.
        position: usize,
        // Private Python list whose slots are filled as native children become
        // ready.
        value: Bound<'py, PyList>,
    },
    Object {
        // Iterator over native object members. Keys stay borrowed from the live
        // Output during traversal.
        source: std::collections::btree_map::Iter<'a, String, JsonValue>,
        // Native key saved for the in-flight child and consumed when the value is
        // installed in the Python dictionary.
        key: Option<&'a str>,
        // Private Python dictionary accumulating completed key/value pairs, exposed
        // only once the frame finishes.
        value: Bound<'py, PyDict>,
    },
}

impl<'a, 'py> Frame<'a, 'py> {
    fn accept(&mut self, value: Bound<'py, PyAny>) -> PyResult<()> {
        match self {
            Self::Array {
                value: array,
                position,
                ..
            } => array.set_item(*position, value),
            Self::Object {
                value: object, key, ..
            } => object.set_item(
                // Create the Python key only when attaching its completed child.
                // The native key was borrowed from Output and remains valid across
                // conversion of a nested value.
                string(value.py(), key.take().expect("pending projected key"))?,
                value,
            ),
        }
    }
    fn next(&mut self) -> Option<&'a JsonValue> {
        match self {
            Self::Array {
                source, position, ..
            } => {
                // Save the destination slot before yielding the child. Conversion
                // may push and pop additional frames before control returns to
                // fill this same Python list.
                let (index, value) = source.next()?;
                *position = index;
                Some(value)
            }
            Self::Object { source, key, .. } => {
                let (name, value) = source.next()?;
                *key = Some(name);
                Some(value)
            }
        }
    }
    // Expose the completed container as a Python object by transferring its
    // existing reference. The parent can then treat a finished container and a
    // scalar uniformly.
    fn finish(self) -> Bound<'py, PyAny> {
        match self {
            Self::Array { value, .. } => value.into_any(),
            Self::Object { value, .. } => value.into_any(),
        }
    }
}

// Frames keep Python containers private while their children are being
// materialized. Traversing one branch at a time makes nesting consume the
// operation's explicit stack. A conversion failure drops the incomplete
// root before the caller can observe it.
// Build the host tree while borrowing an admitted native projection. The frame
// stack stores only active ancestors, so each completed value has one place to
// attach.
fn tree<'py>(py: Python<'py>, output: &Output, pid: u32) -> Result<Bound<'py, PyAny>> {
    let mut stack: Stack<Frame<'_, 'py>> = output.stack()?;
    let mut pending = Some(output.value());
    let mut ready = None;
    loop {
        // Check the creating process before touching the retained output on each
        // traversal step. Python allocations and callbacks can reach this boundary
        // in a different process.
        check_process(pid)?;
        if let Some(value) = pending.take() {
            ready = match value {
                JsonValue::Null => Some(py.None().into_bound(py)),
                JsonValue::Bool(value) => Some(PyBool::new(py, *value).to_owned().into_any()),
                // Return Python ints only within the binary64 safe-integer
                // range. Beyond it, consecutive integers share a representation.
                JsonValue::Number(value)
                    if value.fract() == 0.0 && value.abs() <= 9_007_199_254_740_991.0 =>
                {
                    Some(integer(py, *value as i64)?)
                }
                // Keep fractional or larger binary64 values as Python floats.
                // Making a large float look like an exact arbitrary-precision
                // integer would change what the caller can infer about it.
                JsonValue::Number(value) => Some(float(py, *value)?),
                JsonValue::String(value) => Some(string(py, value)?),
                // Allocate the complete list length up front. A frame fills the
                // slots in order before the list is returned to the caller.
                JsonValue::Array(value) => {
                    let array = list(py, value.len())?;
                    // Recheck the PID after Python allocation and before adding
                    // Rust traversal workspace. The allocation can invoke
                    // interpreter callbacks.
                    check_process(pid)?;
                    stack.push(Frame::Array {
                        source: value.iter().enumerate(),
                        position: 0,
                        value: array,
                    })?;
                    None
                }
                // Allocate a dictionary and iterate borrowed native key/value
                // pairs. Keep each key with its frame until the corresponding
                // converted child is ready to attach.
                JsonValue::Object(value) => {
                    let object = dictionary(py)?;
                    check_process(pid)?;
                    stack.push(Frame::Object {
                        source: value.iter(),
                        key: None,
                        value: object,
                    })?;
                    None
                }
            };
        }
        // A ready child belongs to the top frame, whose key or array position was
        // saved by next(). Attach it before advancing the iterator. With no
        // enclosing frame the ready value is the completed root.
        if let Some(value) = ready.take() {
            if let Some(frame) = stack.last_mut() {
                frame.accept(value)?;
            } else {
                return Ok(value);
            }
        }
        // Check the process again after attaching the child. Insertion can run
        // Python allocation and must not resume inherited Rust traversal in a fork
        // child.
        check_process(pid)?;
        if let Some(frame) = stack.last_mut() {
            // Advance just the innermost native iterator while its Python
            // container remains private. On exhaustion, turn the frame into the
            // next ready child.
            if let Some(value) = frame.next() {
                pending = Some(value);
            } else {
                ready = Some(stack.pop().expect("nonempty output stack").finish());
            }
        }
    }
}

pub(crate) struct Materialized {
    // Completed host object. Keep it owned until final signal and process checks or
    // conversion to a Python exception.
    pub(crate) value: Py<PyAny>,
    // Retains the host allocation estimate through delivery even after the native
    // tree has been cleared.
    pub(crate) output: Output,
}

impl Materialized {
    pub(crate) fn build(
        py: Python<'_>,
        mut output: Output,
        pid: u32,
        extra_host_bytes: usize,
    ) -> Result<Self> {
        check_process(pid)?;
        // Both representations are live while building the Python tree. Admit
        // the host copy before allocating it, then release the native payload
        // while retaining host credit through result or exception delivery.
        // Include caller-supplied extra host bytes in the same wide admission
        // estimate. This covers additional wrapper allocations that coexist with
        // the projected tree.
        let bytes = estimate(&output)? + extra_host_bytes as u128;
        output.reserve_host(bytes)?;
        // Detach the Python reference from the lifetime of this attached borrow
        // without copying the object. It remains owned until final delivery.
        let value = tree(py, &output, pid)?.unbind();
        // Recheck the original process before clearing the retained native tree. A
        // fork child must not release the parent's database and budget owners
        // through this handoff.
        check_process(pid)?;
        output.clear_native();
        Ok(Self { value, output })
    }

    // Deliver the completed Python value through the common signal and PID
    // boundary. Retain the host estimate until that step returns or chooses an
    // error.
    pub(crate) fn finish(
        self,
        py: Python<'_>,
        pid: u32,
        outcome: Option<&WriteOutcome>,
    ) -> PyResult<Py<PyAny>> {
        let Self { value, output } = self;
        let result = crate::errors::finish(py, Ok(value), outcome, pid);
        drop(output);
        result
    }

    // Use the built dictionary to construct the engine exception. Its output lease
    // remains live through exception construction, just as for a successful host
    // value.
    pub(crate) fn exception(self, py: Python<'_>) -> PyResult<PyErr> {
        let fields = self.value.bind(py).cast::<PyDict>()?;
        let result = crate::errors::engine(py, fields);
        drop(self);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cofferdb::binding::Operation;
    use cofferdb::{Error, ErrorCode};

    #[test]
    fn non_ascii_construction_peak_stays_within_its_admitted_host_estimate() {
        const MARKER: &str = "COFFER_UNICODE_CONSTRUCTION_PEAK_TEST";
        const NAME: &str =
            "output::tests::non_ascii_construction_peak_stays_within_its_admitted_host_estimate";
        if std::env::var_os(MARKER).is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--nocapture"])
                .env(MARKER, "1")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let tracer = py.import("tracemalloc").unwrap();
            for message in [
                "\u{1f642}".repeat(2_000_000),
                "\u{4e2d}".repeat(1_000_000),
                format!(
                    "{}\u{4e2d}{}\u{1f642}",
                    "a".repeat(1_000_000),
                    "b".repeat(1_000_000)
                ),
            ] {
                let length = message.chars().count();
                let native = Operation::standalone()
                    .error(Error::new(ErrorCode::ChunkNotFound, message))
                    .unwrap();
                let admitted = estimate(&native).unwrap();
                tracer.call_method0("start").unwrap();
                let host = Materialized::build(py, native, std::process::id(), 0).unwrap();
                let (_, peak) = tracer
                    .call_method0("get_traced_memory")
                    .unwrap()
                    .extract::<(usize, usize)>()
                    .unwrap();
                tracer.call_method0("stop").unwrap();
                assert!(
                    peak as u128 <= admitted,
                    "host peak {peak} exceeds admitted {admitted}"
                );
                assert_eq!(
                    host.value
                        .bind(py)
                        .get_item("message")
                        .unwrap()
                        .len()
                        .unwrap(),
                    length
                );
                drop(host);
            }
        });
    }

    #[test]
    fn impossible_python_list_allocation_preserves_memory_error() {
        Python::initialize();
        Python::attach(|py| {
            // CPython rejects the overflowing pointer-array size before it can
            // allocate it; exercise a real null constructor result safely.
            let error = list(py, isize::MAX as usize).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyMemoryError>(py));
        });
    }

    #[test]
    fn list_projection_is_complete_owned_and_usable_after_native_close() {
        use cofferdb::{
            CollectionConfig, Database, DatabaseConfig, DistanceMetric, DocumentInput,
            TokenizerConfig, VectorIndexConfig,
        };
        let path =
            std::env::temp_dir().join(format!("coffer-output-lists-{}.coffer", std::process::id()));
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let collection = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: TokenizerConfig::default_en(),
            })
            .unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "document".into(),
                title: None,
                source_uri: None,
                chunks: vec![],
                metadata: cofferdb::parse_json(
                    "{\"values\":[null,true,3,0.25,\"\\u4e2d\",[],{\"nested\":[1,2]}]}",
                )
                .unwrap(),
            })
            .unwrap();
        let operation = Operation::new(&db);
        let mut input = operation.input();
        let id = input.text("document").unwrap();
        let output = operation
            .get_document(&db, collection.collection_id(), input.finish(id))
            .unwrap();
        Python::initialize();
        Python::attach(|py| {
            let materialized = Materialized::build(py, output, std::process::id(), 0).unwrap();
            db.close().unwrap();
            let value = materialized.finish(py, std::process::id(), None).unwrap();
            let array = value
                .bind(py)
                .get_item("metadata")
                .unwrap()
                .get_item("values")
                .unwrap();
            assert_eq!(array.len().unwrap(), 7);
            assert!(array.get_item(0).unwrap().is_none());
            assert!(array.get_item(1).unwrap().extract::<bool>().unwrap());
            assert_eq!(array.get_item(2).unwrap().extract::<u64>().unwrap(), 3);
            assert_eq!(array.get_item(3).unwrap().extract::<f64>().unwrap(), 0.25);
            assert_eq!(
                array.get_item(4).unwrap().extract::<String>().unwrap(),
                "\u{4e2d}"
            );
            assert_eq!(array.get_item(5).unwrap().len().unwrap(), 0);
            assert_eq!(
                array
                    .get_item(6)
                    .unwrap()
                    .get_item("nested")
                    .unwrap()
                    .get_item(1)
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                2
            );
        });
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn error_materialization_has_all_five_core_fields_and_a_null_action() {
        Python::initialize();
        Python::attach(|py| {
            let output = Operation::standalone()
                .error(
                    Error::new(ErrorCode::ChunkNotFound, "missing \u{4e2d}\u{1f642}")
                        .with_detail("field", "chunk_id"),
                )
                .unwrap();
            let materialized = Materialized::build(py, output, std::process::id(), 1024).unwrap();
            let error = materialized.exception(py).unwrap();
            assert_eq!(
                error
                    .value(py)
                    .getattr("code")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "chunk_not_found"
            );
            assert!(error
                .value(py)
                .getattr("suggested_action")
                .unwrap()
                .is_none());
            assert_eq!(
                error
                    .value(py)
                    .getattr("message")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "missing \u{4e2d}\u{1f642}"
            );
            assert!(!error
                .value(py)
                .getattr("recoverable")
                .unwrap()
                .extract::<bool>()
                .unwrap());
            let detail = error.value(py).getattr("detail").unwrap();
            assert_eq!(
                detail
                    .get_item("field")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "chunk_id"
            );
        });
    }

    #[test]
    fn host_estimate_covers_materialized_dictionary_and_unicode_samples() {
        Python::initialize();
        Python::attach(|py| {
            for message in [
                "",
                "ascii\0",
                "\u{e9}",
                "\u{4e2d}",
                "\u{1f642}",
                "mixed \u{4e2d}\u{1f642}",
            ] {
                let output = Operation::standalone()
                    .error(
                        Error::new(ErrorCode::InvalidJsonlSchema, message)
                            .with_detail("sample", message),
                    )
                    .unwrap();
                let bound = estimate(&output).unwrap();
                let materialized = Materialized::build(py, output, std::process::id(), 0).unwrap();
                let locals = PyDict::new(py);
                locals
                    .set_item("value", materialized.value.bind(py))
                    .unwrap();
                py.run(c"import sys\ndef size(value, seen):\n    identity = id(value)\n    if identity in seen:\n        return 0\n    seen.add(identity)\n    total = sys.getsizeof(value)\n    if type(value) is dict:\n        total += sum(size(k, seen) + size(v, seen) for k, v in value.items())\n    elif type(value) is list:\n        total += sum(size(v, seen) for v in value)\n    return total\nactual = size(value, set())", Some(&locals), Some(&locals)).unwrap();
                let actual = locals
                    .get_item("actual")
                    .unwrap()
                    .unwrap()
                    .extract::<u128>()
                    .unwrap();
                assert!(actual <= bound, "{actual} > {bound}");
                materialized.finish(py, std::process::id(), None).unwrap();
            }
        });
    }
}
