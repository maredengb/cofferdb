//! Exact built-in input conversion. No user-defined conversion protocols run.

use cofferdb::binding::{Input, Stack, ValueKind};
use cofferdb::{JsonObject, JsonValue};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyStringData, PyTuple};

// Keep core budget or schema errors separate from an existing Python exception and
// from newly diagnosed host type/value errors. The API wrapper translates each
// through a different exception path.
#[derive(Debug)]
pub(crate) enum Failure {
    // Preserve the core error, including its structured details, for native
    // error-envelope projection.
    Engine(cofferdb::Error),
    // Keep the original Python exception object so failures such as a Python
    // allocation error retain their identity.
    Python(PyErr),
    // The argument has the wrong built-in type. The binding will expose this as a
    // Python type error.
    Type(&'static str),
    // The host type was accepted but its contents cannot represent the requested
    // native value.
    Value(&'static str),
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

pub(crate) type Result<T> = std::result::Result<T, Failure>;

// Traverse exact list and tuple objects through their native accessors. No
// iterator supplied by the application is needed, and the references stay attached
// to Python during conversion.
pub(crate) enum Sequence<'py> {
    List(Bound<'py, PyList>),
    Tuple(Bound<'py, PyTuple>),
}

impl<'py> Sequence<'py> {
    pub(crate) fn new(value: &Bound<'py, PyAny>) -> Result<Self> {
        // Test the exact runtime type before casting: a Python subclass would pass
        // a normal downcast but could expose user-defined behavior the binding
        // does not accept.
        if value.is_exact_instance_of::<PyList>() {
            Ok(Self::List(
                value.cast::<PyList>().map_err(PyErr::from)?.clone(),
            ))
        } else if value.is_exact_instance_of::<PyTuple>() {
            Ok(Self::Tuple(
                value.cast::<PyTuple>().map_err(PyErr::from)?.clone(),
            ))
        } else {
            Err(Failure::Type("expected a built-in list or tuple"))
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::List(value) => value.len(),
            Self::Tuple(value) => value.len(),
        }
    }

    // Return an owned Python reference to the element at the requested position
    // without copying its contents. A failing indexed read remains a Python
    // exception.
    pub(crate) fn item(&self, index: usize) -> Result<Bound<'py, PyAny>> {
        match self {
            Self::List(value) => value.get_item(index).map_err(Failure::from),
            Self::Tuple(value) => value.get_item(index).map_err(Failure::from),
        }
    }
}

pub(crate) fn dict<'a, 'py>(value: &'a Bound<'py, PyAny>) -> Result<&'a Bound<'py, PyDict>> {
    // Require a built-in dict just as sequences require concrete built-in
    // containers. This prevents configuration and JSON conversion from invoking
    // arbitrary mapping methods.
    if !value.is_exact_instance_of::<PyDict>() {
        return Err(Failure::Type("expected a built-in dict"));
    }
    value
        .cast::<PyDict>()
        .map_err(|error| Failure::Python(error.into()))
}

pub(crate) fn boolean(value: &Bound<'_, PyAny>) -> Result<bool> {
    // Reject truthiness coercion. Flags require a Python bool; values such as 0, 1
    // and objects defining a truth test are not equivalent inputs here.
    if !value.is_exact_instance_of::<PyBool>() {
        return Err(Failure::Type("expected a built-in bool"));
    }
    value.extract().map_err(Failure::from)
}

pub(crate) fn unsigned(value: &Bound<'_, PyAny>) -> Result<u64> {
    // Python bool is an int subclass. Exact type matching excludes it and also
    // keeps a float containing a whole number from being accepted as a count.
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(Failure::Type(
            "expected a built-in int; bool and float are not accepted",
        ));
    }
    // Let the Python integer extractor check both sign and u64 range. Translate
    // only overflow to this value-range diagnosis; other Python failures keep
    // their original exception.
    value.extract().map_err(|error| {
        numeric_error(
            value.py(),
            error,
            "integer is outside the unsigned 64-bit range",
        )
    })
}

pub(crate) fn number(value: &Bound<'_, PyAny>) -> Result<f64> {
    // Allow native integer and float values for real-valued input. The exact-type
    // gate rejects bool and custom numeric conversion methods before extraction.
    if !(value.is_exact_instance_of::<PyInt>() || value.is_exact_instance_of::<PyFloat>()) {
        return Err(Failure::Type(
            "expected a built-in int or float; bool is not accepted",
        ));
    }
    // Conversion to binary64 can lose precision for a large integer that still
    // fits its range. This helper converts the type; callers with stricter
    // finiteness or representation constraints apply them separately.
    value.extract().map_err(|error| {
        numeric_error(
            value.py(),
            error,
            "number cannot be represented by the native numeric type",
        )
    })
}

fn numeric_error(py: Python<'_>, error: PyErr, message: &'static str) -> Failure {
    // Translate a Python integer overflow into the argument converter's
    // value-range failure. Keep unrelated Python exceptions intact so an
    // allocation or interpreter error does not appear to be bad numeric input.
    if error.is_instance_of::<PyOverflowError>(py) {
        Failure::Value(message)
    } else {
        Failure::Python(error)
    }
}

fn unicode<'a, 'py>(value: &'a Bound<'py, PyAny>) -> Result<Codepoints<'a>> {
    if !value.is_exact_instance_of::<PyString>() {
        return Err(Failure::Type("expected a built-in str"));
    }
    let string = value.cast::<PyString>().map_err(PyErr::from)?;
    // SAFETY: the Bound borrow keeps this exact built-in string alive while
    // its immutable code units are read with the GIL held. PyO3's data()
    // relies on the target's CPython Unicode bitfield layout. Each supported
    // little-endian ABI/platform must pass the conversion tests for UCS1,
    // UCS2, UCS4, embedded NULs and surrogates before distribution.
    let data = unsafe { string.data()? };
    let points = Codepoints { data, position: 0 };
    // Validate the immutable sequence before handing out a replayable cursor. A
    // Python string may contain a lone surrogate, which has no Rust char or UTF-8
    // encoding.
    if points.clone().any(|value| char::from_u32(value).is_none()) {
        return Err(Failure::Value("string is not valid UTF-8"));
    }
    Ok(points)
}

#[derive(Clone)]
struct Codepoints<'a> {
    // Borrowed immutable UCS1, UCS2 or UCS4 code units from the exact Python string
    // for the attached conversion.
    data: PyStringData<'a>,
    // Index of the next code unit, measured in the encoding's elements. Cloned
    // cursors can preflight and replay the same string.
    position: usize,
}

impl Iterator for Codepoints<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        // CPython selects one code-unit width for the whole string. Widen UCS1 and
        // UCS2 elements without interpreting them as UTF-8 bytes or combining
        // adjacent units; the validated iteration yields one scalar at a time.
        let value = match self.data {
            PyStringData::Ucs1(value) => value.get(self.position).copied().map(u32::from),
            PyStringData::Ucs2(value) => value.get(self.position).copied().map(u32::from),
            PyStringData::Ucs4(value) => value.get(self.position).copied(),
        };
        self.position += usize::from(value.is_some());
        value
    }
}

// Only unicode()'s validated immutable sequence reaches this conversion. Both
// sizing and encoding replay that sequence, so a surrogate cannot first appear on
// the second pass.
fn scalar(value: u32) -> char {
    char::from_u32(value).expect("validated Unicode scalar")
}

pub(crate) fn text(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<String> {
    // Names, paths and other ordinary text use the operation's workspace. Produce
    // the admitted UTF-8 buffer directly from borrowed code points instead of
    // allocating an intermediate Python encoding.
    input
        .text_from_chars(unicode(value)?.map(scalar))
        .map_err(Failure::from)
}

fn json_string(value: &Bound<'_, PyAny>, input: &mut Input, kind: ValueKind) -> Result<String> {
    // JSON keys and string values use the same Unicode traversal but charge their
    // retained bytes to the requested business-value kind.
    input
        .json_string_from_chars(unicode(value)?.map(scalar), kind)
        .map_err(Failure::from)
}

enum Frame<'py, I> {
    Array {
        // Python identity of this active sequence, used to detect descent through a
        // reference to an enclosing container.
        identity: usize,
        // Exact list or tuple retained while its elements are traversed with the
        // interpreter attached.
        source: Sequence<'py>,
        // Index of the next source element to schedule. The last visited child
        // still has to be attached to values.
        next: usize,
        // In-order native elements already constructed for this array, backed by
        // the operation's JSON reservation.
        values: Vec<JsonValue>,
    },
    Object {
        // Python identity of this active dictionary, checked alongside active array
        // frames for cycles.
        identity: usize,
        // Iterator retaining the traversal position in the exact Python dictionary
        // while nested children are converted.
        source: I,
        // Owned key for the value currently being visited. Consumed when the
        // completed child is inserted.
        key: Option<String>,
        // Native object containing completed key/value pairs. It is exposed to its
        // parent only after iteration finishes.
        values: JsonObject,
    },
}

impl<'py, I> Frame<'py, I>
where
    I: Iterator<Item = (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
{
    // Array and object frames use one Python identity namespace. Cycle detection
    // can therefore catch a list that returns to an enclosing dict as well as a
    // direct self-reference.
    fn identity(&self) -> usize {
        match self {
            Self::Array { identity, .. } | Self::Object { identity, .. } => *identity,
        }
    }

    // The caller supplies a completed native subtree. Array insertion uses the
    // capacity admitted when the frame was opened; an object consumes the pending
    // key associated with that particular child.
    fn accept(&mut self, value: JsonValue) {
        match self {
            Self::Array { values, .. } => values.push(value),
            Self::Object { key, values, .. } => {
                // A key is fixed before its value is visited; conversion cannot
                // invoke callbacks or release the GIL and mutate the dict.
                values.insert(key.take().expect("pending built-in dict key"), value);
            }
        }
    }

    fn next(&mut self, input: &mut Input, kind: ValueKind) -> Result<Option<Bound<'py, PyAny>>> {
        match self {
            Self::Array { source, next, .. } => {
                if *next == source.len() {
                    return Ok(None);
                }
                // Move the cursor past an element only after its retrieval
                // succeeds. The caller finishes that child before asking the frame
                // to schedule the following element.
                let value = source.item(*next)?;
                *next += 1;
                Ok(Some(value))
            }
            Self::Object { source, key, .. } => {
                let Some((name, value)) = source.next() else {
                    return Ok(None);
                };
                // Convert and charge the key before descending into its value. A
                // nested container may keep this frame suspended, so retain the
                // Rust key here until the child is ready to insert.
                *key = Some(json_string(&name, input, kind)?);
                Ok(Some(value))
            }
        }
    }

    // Move the completed native container out after its source iterator is
    // exhausted. The parent attaches one value regardless of whether the child was
    // a scalar or another frame.
    fn finish(self) -> JsonValue {
        match self {
            Self::Array { values, .. } => JsonValue::Array(values),
            Self::Object { values, .. } => JsonValue::Object(values),
        }
    }
}

/// Convert one free business value; fixed record wrappers use field-wise parsing.
// Each frame owns an unfinished Rust container and remembers where to attach
// its next child. Finish the child before advancing the parent. Until the
// root is complete the whole tree belongs to conversion: an error drops it
// without exposing a partially converted input to the engine.
pub(crate) fn json(
    value: &Bound<'_, PyAny>,
    input: &mut Input,
    kind: ValueKind,
) -> Result<JsonValue> {
    // Use an admitted stack for live ancestors. Pending owns an attached Python
    // reference to the next object, while ready owns its converted Rust value.
    let mut stack: Stack<Frame<'_, _>> = input.stack()?;
    let mut pending = Some(value.clone());
    let mut ready = None;
    loop {
        if let Some(value) = pending.take() {
            // Map exact built-in scalar types directly. Check bool separately from
            // int so a Boolean cannot enter the JSON number path.
            if value.is_none() {
                ready = Some(JsonValue::Null);
            } else if value.is_exact_instance_of::<PyBool>() {
                ready = Some(JsonValue::Bool(boolean(&value)?));
            } else if value.is_exact_instance_of::<PyInt>() {
                // JSON stores numbers as binary64. Compare the converted value with
                // the original Python integer: a large power of two can round-trip
                // exactly even when its immediate neighbors cannot.
                let converted = number(&value)?;
                if !converted.is_finite()
                    || !value.eq(crate::output::float(value.py(), converted)?)?
                {
                    return Err(Failure::Value(
                        "JSON integer is not exactly representable; use an explicit string",
                    ));
                }
                ready = Some(JsonValue::Number(converted));
            // Copy the exact Python float as a JSON number. Validation of the
            // completed business value still checks its numeric and size
            // invariants.
            } else if value.is_exact_instance_of::<PyFloat>() {
                ready = Some(JsonValue::Number(number(&value)?));
            } else if value.is_exact_instance_of::<PyString>() {
                // Encode a Python string into a native buffer charged under the
                // current value kind. Keys use the same accounting when an object
                // frame asks for a member.
                ready = Some(JsonValue::String(json_string(&value, input, kind)?));
            // Only exact dict, list and tuple instances enter the container path.
            // Other mappings and iterables must first be converted by the caller
            // to a supported built-in type.
            } else if value.is_exact_instance_of::<PyDict>()
                || value.is_exact_instance_of::<PyList>()
                || value.is_exact_instance_of::<PyTuple>()
            {
                // Use the live Python object's identity to detect cycles on the
                // current path. The frame keeps the source alive while that
                // identity is in the ancestor stack.
                let identity = value.as_ptr() as usize;
                // Only ancestors are active. A shared child in separate branches
                // is valid and is converted independently at each occurrence.
                if stack
                    .as_slice()
                    .iter()
                    .any(|frame| frame.identity() == identity)
                {
                    return Err(Failure::Value("cyclic containers are not supported"));
                }
                // Test one more container level before reserving its native node
                // or element slots. Scalar children do not consume another depth
                // level.
                input.check_container_depth(stack.as_slice().len(), kind)?;
                // Snapshot the expected dict size for admission, then fill it one
                // member at a time. The frame retains the pending key until its
                // value can be inserted.
                let frame = if value.is_exact_instance_of::<PyDict>() {
                    let dict = dict(&value)?;
                    Frame::Object {
                        identity,
                        source: dict.iter(),
                        key: None,
                        values: input.object(dict.len(), kind)?,
                    }
                } else {
                    // Borrow the supported sequence and reserve its native slot
                    // capacity before traversing elements. No temporary list of
                    // Python values is needed for traversal.
                    let sequence = Sequence::new(&value)?;
                    let values = input.array(sequence.len(), kind)?;
                    Frame::Array {
                        identity,
                        source: sequence,
                        next: 0,
                        values,
                    }
                };
                // Push the opened frame only after its state and allocation have
                // been prepared. The following iteration can obtain its first
                // child or finalize it as empty.
                stack.push(frame)?;
            } else {
                // Reject unsupported objects before any generic Python protocol
                // traversal. The converter has no path that coerces framework
                // scalars or container subclasses.
                return Err(Failure::Type(
                    "unsupported value; only built-in JSON types are accepted",
                ));
            }
        }
        // Validate the assembled root once its last frame is complete. Children can
        // belong to a larger, independently limited input value; validating each one
        // in isolation would miss the limit on the whole tree.
        if let Some(value) = ready.take() {
            // The last frame is the only unfinished parent of this child. Deeper
            // ancestors wait for that frame to become one complete native value.
            if let Some(parent) = stack.last_mut() {
                parent.accept(value);
            } else {
                // Apply business-value policy at the root after all children are
                // integrated. The same operation still owns both decoded capacity
                // and validation scratch.
                input.validate_json(&value, kind)?;
                return Ok(value);
            }
        }
        // Ask the innermost frame to advance. Exhaustion pops one completed
        // container and routes it through the same attachment step as a scalar.
        if let Some(frame) = stack.last_mut() {
            if let Some(value) = frame.next(input, kind)? {
                pending = Some(value);
            } else {
                ready = Some(stack.pop().expect("nonempty conversion stack").finish());
            }
        }
    }
}

/// Construct the core f32 vector directly, without an intermediate f64 tree.
pub(crate) fn vector(
    value: &Bound<'_, PyAny>,
    input: &mut Input,
    kind: ValueKind,
) -> Result<Vec<f32>> {
    // Determine the exact list or tuple length before admitting binary32 slots.
    // The typed converter does not build an intermediate JSON number array.
    let sequence = Sequence::new(value)?;
    let mut vector = input.vector(sequence.len(), kind)?;
    // Convert entries in source order and narrow immediately to the engine's
    // coordinate type. The core still validates the resulting finite range and
    // shape.
    for index in 0..sequence.len() {
        vector.push(number(&sequence.item(index)?)? as f32);
    }
    // Finite, declared dimension and model validation belong to the core
    // query/write path. Keeping these values does not admit them to storage.
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cofferdb::binding::Operation;
    use cofferdb::{Database, DatabaseConfig};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn with_input(run: impl FnOnce(Python<'_>, &mut Input)) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "coffer-convert-{}-{}.coffer",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let database = Database::create(&path, DatabaseConfig::default()).unwrap();
        let operation = Operation::new(&database);
        let mut input = operation.input();
        Python::initialize();
        Python::attach(|py| run(py, &mut input));
        drop(input);
        drop(operation);
        database.close().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_builtins_preserve_values_and_shared_children() {
        with_input(|py, input| {
            let value = py
                .eval(
                    c"{'key': [True, None, 3.5, ('é', 1152921504606846976)]}",
                    None,
                    None,
                )
                .unwrap();
            let value = json(&value, input, ValueKind::DOCUMENT_METADATA).unwrap();
            let array = value.get("key").unwrap().as_array().unwrap();
            assert_eq!(array[0], JsonValue::Bool(true));
            assert!(array[1].is_null());
            assert_eq!(
                array[3].as_array().unwrap()[1].as_f64(),
                Some(1152921504606846976.0)
            );
            let locals = PyDict::new(py);
            py.run(
                c"child = {'x': 1}\nvalue = [child, child]",
                None,
                Some(&locals),
            )
            .unwrap();
            let value = json(
                &locals.get_item("value").unwrap().unwrap(),
                input,
                ValueKind::CHUNK_METADATA,
            )
            .unwrap();
            assert_eq!(value.as_array().unwrap()[0], value.as_array().unwrap()[1]);
        });
    }

    #[test]
    fn implicit_protocols_and_subclasses_are_rejected_without_callbacks() {
        with_input(|py, input| {
            let locals = PyDict::new(py);
            py.run(c"class Sneaky(list):\n    def __iter__(self):\n        raise AssertionError('callback')\nclass Number:\n    def __float__(self):\n        raise AssertionError('callback')\nvalue = Sneaky([1])\nnumber = Number()", None, Some(&locals)).unwrap();
            assert!(matches!(
                json(
                    &locals.get_item("value").unwrap().unwrap(),
                    input,
                    ValueKind::FILTER
                ),
                Err(Failure::Type(_))
            ));
            assert!(matches!(
                number(&locals.get_item("number").unwrap().unwrap()),
                Err(Failure::Type(_))
            ));
            assert!(matches!(
                text(&py.eval(c"b'bytes'", None, None).unwrap(), input),
                Err(Failure::Type(_))
            ));
        });
    }

    #[test]
    fn integer_arguments_are_exact_and_json_integers_never_silently_round() {
        with_input(|py, input| {
            assert_eq!(
                unsigned(&py.eval(c"18446744073709551615", None, None).unwrap()).unwrap(),
                u64::MAX
            );
            for source in [c"True", c"1.0"] {
                assert!(matches!(
                    unsigned(&py.eval(source, None, None).unwrap()),
                    Err(Failure::Type(_))
                ));
            }
            for source in [c"-1", c"18446744073709551616"] {
                assert!(matches!(
                    unsigned(&py.eval(source, None, None).unwrap()),
                    Err(Failure::Value(_))
                ));
            }
            assert!(matches!(
                json(
                    &py.eval(c"9007199254740993", None, None).unwrap(),
                    input,
                    ValueKind::DOCUMENT_METADATA
                ),
                Err(Failure::Value(_))
            ));
        });
    }

    #[test]
    fn cyclic_unicode_and_native_json_boundaries_keep_distinct_failures() {
        with_input(|py, input| {
            let locals = PyDict::new(py);
            py.run(c"value = []\nvalue.append(value)", None, Some(&locals))
                .unwrap();
            assert!(matches!(
                json(
                    &locals.get_item("value").unwrap().unwrap(),
                    input,
                    ValueKind::FILTER
                ),
                Err(Failure::Value(_))
            ));
            assert!(matches!(
                text(&py.eval(c"'\\ud800'", None, None).unwrap(), input),
                Err(Failure::Value(_))
            ));
            let error = json(
                &py.eval(c"{'a': float('nan')}", None, None).unwrap(),
                input,
                ValueKind::DOCUMENT_METADATA,
            )
            .err()
            .unwrap();
            assert!(matches!(error, Failure::Engine(_)));
            py.run(
                c"value = None\nfor _ in range(128):\n    value = [value]",
                None,
                Some(&locals),
            )
            .unwrap();
            json(
                &locals.get_item("value").unwrap().unwrap(),
                input,
                ValueKind::FILTER,
            )
            .unwrap();
            py.run(c"value = [value]", None, Some(&locals)).unwrap();
            let error = json(
                &locals.get_item("value").unwrap().unwrap(),
                input,
                ValueKind::FILTER,
            )
            .err()
            .unwrap();
            assert!(
                matches!(error, Failure::Engine(error) if error.detail().iter().any(|(key,value)| key == "json_limit_reason" && value == "depth_exceeded"))
            );
        });
    }

    #[test]
    fn vector_goes_directly_to_f32_and_rejects_boolean_elements() {
        with_input(|py, input| {
            let result = vector(
                &py.eval(c"(1, 0.1, -2.5)", None, None).unwrap(),
                input,
                ValueKind::QUERY_EMBEDDING,
            )
            .unwrap();
            assert_eq!(result, vec![1.0_f32, 0.1_f32, -2.5_f32]);
            assert!(matches!(
                vector(
                    &py.eval(c"[True]", None, None).unwrap(),
                    input,
                    ValueKind::QUERY_EMBEDDING
                ),
                Err(Failure::Type(_))
            ));
        });
    }

    #[test]
    fn borrowed_cpython_unicode_storage_covers_all_widths_and_rejects_surrogates() {
        with_input(|py, input| {
            for expected in [
                "",
                "ascii\0value",
                "\u{e9}",
                "\u{4e2d}",
                "\u{1f642}",
                "\u{10ffff}",
                "a\0\u{e9}\u{4e2d}\u{1f642}",
            ] {
                let value = PyString::new(py, expected).into_any();
                assert_eq!(text(&value, input).unwrap(), expected);
                assert_eq!(
                    json(&value, input, ValueKind::DOCUMENT_METADATA)
                        .unwrap()
                        .as_str(),
                    Some(expected)
                );
            }
            for expression in [
                c"'\\ud800'",
                c"'\\udfff'",
                c"'\\U0010ffff\\ud800'",
                c"'\\u4e2d\\ud800'",
            ] {
                let value = py.eval(expression, None, None).unwrap();
                assert!(matches!(text(&value, input), Err(Failure::Value(_))));
            }
        });
    }
}
