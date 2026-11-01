//! Exact built-in input conversion. No user-defined conversion protocols run.

use cofferdb::binding::{Input, Stack, ValueKind};
use cofferdb::{JsonObject, JsonValue};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyStringData, PyTuple};

#[derive(Debug)]
pub(crate) enum Failure {
    Engine(cofferdb::Error),
    Python(PyErr),
    Type(&'static str),
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

pub(crate) enum Sequence<'py> {
    List(Bound<'py, PyList>),
    Tuple(Bound<'py, PyTuple>),
}

impl<'py> Sequence<'py> {
    pub(crate) fn new(value: &Bound<'py, PyAny>) -> Result<Self> {
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

    pub(crate) fn item(&self, index: usize) -> Result<Bound<'py, PyAny>> {
        match self {
            Self::List(value) => value.get_item(index).map_err(Failure::from),
            Self::Tuple(value) => value.get_item(index).map_err(Failure::from),
        }
    }
}

pub(crate) fn dict<'a, 'py>(value: &'a Bound<'py, PyAny>) -> Result<&'a Bound<'py, PyDict>> {
    if !value.is_exact_instance_of::<PyDict>() {
        return Err(Failure::Type("expected a built-in dict"));
    }
    value
        .cast::<PyDict>()
        .map_err(|error| Failure::Python(error.into()))
}

pub(crate) fn boolean(value: &Bound<'_, PyAny>) -> Result<bool> {
    if !value.is_exact_instance_of::<PyBool>() {
        return Err(Failure::Type("expected a built-in bool"));
    }
    value.extract().map_err(Failure::from)
}

pub(crate) fn unsigned(value: &Bound<'_, PyAny>) -> Result<u64> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(Failure::Type(
            "expected a built-in int; bool and float are not accepted",
        ));
    }
    value.extract().map_err(|error| {
        numeric_error(
            value.py(),
            error,
            "integer is outside the unsigned 64-bit range",
        )
    })
}

pub(crate) fn number(value: &Bound<'_, PyAny>) -> Result<f64> {
    if !(value.is_exact_instance_of::<PyInt>() || value.is_exact_instance_of::<PyFloat>()) {
        return Err(Failure::Type(
            "expected a built-in int or float; bool is not accepted",
        ));
    }
    value.extract().map_err(|error| {
        numeric_error(
            value.py(),
            error,
            "number cannot be represented by the native numeric type",
        )
    })
}

fn numeric_error(py: Python<'_>, error: PyErr, message: &'static str) -> Failure {
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
    // SAFETY: the package accepts only regular-GIL CPython on little-endian
    // arm64/x86_64, holds the GIL for this entire borrow, and never calls user
    // protocols or detaches while iterating. The six ABI/platform targets must
    // validate UCS1/UCS2/UCS4, surrogates and embedded NULs before delivery.
    let data = unsafe { string.data()? };
    let points = Codepoints { data, position: 0 };
    if points.clone().any(|value| char::from_u32(value).is_none()) {
        return Err(Failure::Value("string is not valid UTF-8"));
    }
    Ok(points)
}

#[derive(Clone)]
struct Codepoints<'a> {
    data: PyStringData<'a>,
    position: usize,
}

impl Iterator for Codepoints<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        let value = match self.data {
            PyStringData::Ucs1(value) => value.get(self.position).copied().map(u32::from),
            PyStringData::Ucs2(value) => value.get(self.position).copied().map(u32::from),
            PyStringData::Ucs4(value) => value.get(self.position).copied(),
        };
        self.position += usize::from(value.is_some());
        value
    }
}

fn scalar(value: u32) -> char {
    char::from_u32(value).expect("validated Unicode scalar")
}

pub(crate) fn text(value: &Bound<'_, PyAny>, input: &mut Input) -> Result<String> {
    input
        .text_from_chars(unicode(value)?.map(scalar))
        .map_err(Failure::from)
}

fn json_string(value: &Bound<'_, PyAny>, input: &mut Input, kind: ValueKind) -> Result<String> {
    input
        .json_string_from_chars(unicode(value)?.map(scalar), kind)
        .map_err(Failure::from)
}

enum Frame<'py, I> {
    Array {
        identity: usize,
        source: Sequence<'py>,
        next: usize,
        values: Vec<JsonValue>,
    },
    Object {
        identity: usize,
        source: I,
        key: Option<String>,
        values: JsonObject,
    },
}

impl<'py, I> Frame<'py, I>
where
    I: Iterator<Item = (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
{
    fn identity(&self) -> usize {
        match self {
            Self::Array { identity, .. } | Self::Object { identity, .. } => *identity,
        }
    }

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
                let value = source.item(*next)?;
                *next += 1;
                Ok(Some(value))
            }
            Self::Object { source, key, .. } => {
                let Some((name, value)) = source.next() else {
                    return Ok(None);
                };
                *key = Some(json_string(&name, input, kind)?);
                Ok(Some(value))
            }
        }
    }

    fn finish(self) -> JsonValue {
        match self {
            Self::Array { values, .. } => JsonValue::Array(values),
            Self::Object { values, .. } => JsonValue::Object(values),
        }
    }
}

/// Convert one free business value; fixed record wrappers use field-wise parsing.
pub(crate) fn json(
    value: &Bound<'_, PyAny>,
    input: &mut Input,
    kind: ValueKind,
) -> Result<JsonValue> {
    let mut stack: Stack<Frame<'_, _>> = input.stack()?;
    let mut pending = Some(value.clone());
    let mut ready = None;
    loop {
        if let Some(value) = pending.take() {
            if value.is_none() {
                ready = Some(JsonValue::Null);
            } else if value.is_exact_instance_of::<PyBool>() {
                ready = Some(JsonValue::Bool(boolean(&value)?));
            } else if value.is_exact_instance_of::<PyInt>() {
                let converted = number(&value)?;
                if !converted.is_finite()
                    || !value.eq(crate::output::float(value.py(), converted)?)?
                {
                    return Err(Failure::Value(
                        "JSON integer is not exactly representable; use an explicit string",
                    ));
                }
                ready = Some(JsonValue::Number(converted));
            } else if value.is_exact_instance_of::<PyFloat>() {
                ready = Some(JsonValue::Number(number(&value)?));
            } else if value.is_exact_instance_of::<PyString>() {
                ready = Some(JsonValue::String(json_string(&value, input, kind)?));
            } else if value.is_exact_instance_of::<PyDict>()
                || value.is_exact_instance_of::<PyList>()
                || value.is_exact_instance_of::<PyTuple>()
            {
                let identity = value.as_ptr() as usize;
                if stack
                    .as_slice()
                    .iter()
                    .any(|frame| frame.identity() == identity)
                {
                    return Err(Failure::Value("cyclic containers are not supported"));
                }
                input.check_container_depth(stack.as_slice().len(), kind)?;
                let frame = if value.is_exact_instance_of::<PyDict>() {
                    let dict = dict(&value)?;
                    Frame::Object {
                        identity,
                        source: dict.iter(),
                        key: None,
                        values: input.object(dict.len(), kind)?,
                    }
                } else {
                    let sequence = Sequence::new(&value)?;
                    let values = input.array(sequence.len(), kind)?;
                    Frame::Array {
                        identity,
                        source: sequence,
                        next: 0,
                        values,
                    }
                };
                stack.push(frame)?;
            } else {
                return Err(Failure::Type(
                    "unsupported value; only built-in JSON types are accepted",
                ));
            }
        }
        if let Some(value) = ready.take() {
            if let Some(parent) = stack.last_mut() {
                parent.accept(value);
            } else {
                input.validate_json(&value, kind)?;
                return Ok(value);
            }
        }
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
    let sequence = Sequence::new(value)?;
    let mut vector = input.vector(sequence.len(), kind)?;
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
