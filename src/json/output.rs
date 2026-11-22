use super::{JsonInputKind, JsonOperationContext, JsonReservation, JsonValue};
use crate::{Error, Result};

// These paths describe library-built DTOs, not arbitrary input wrappers. Business
// values have already been checked at their own roots before DTO construction.
#[derive(Clone, Copy)]
pub(crate) enum JsonOutputSchema {
    // A value fixes the error category for its subtree. The other variants mark
    // positions within library-built search and diagnostic envelopes until
    // traversal reaches an individually validated business value.
    Value(JsonInputKind),
    SearchEnvelope,
    WhyNotEnvelope,
    Search,
    SearchItems,
    SearchItem,
    Explain,
    WhyNot,
    FilterResult,
}

impl JsonOutputSchema {
    pub(crate) fn kind(self) -> JsonInputKind {
        match self {
            Self::Value(kind) => kind,
            _ => JsonInputKind::StoredJson,
        }
    }

    pub(super) fn field(self, key: &str) -> Self {
        // Interpret a member name together with the parent path. A user object
        // with a key named expression must not be classified as a filter unless it
        // is the expression of an actual filter-result wrapper.
        match (self, key) {
            (Self::SearchEnvelope, "result") => Self::Search,
            (Self::WhyNotEnvelope, "result") => Self::WhyNot,
            (Self::Search, "results") => Self::SearchItems,
            (Self::SearchItem, "explain") => Self::Explain,
            // Search explanations place the filter under explain.filter, while a
            // why-not report uses filter_result. Both carry the original predicate
            // under expression and need the same classification there.
            (Self::Explain, "filter") | (Self::WhyNot, "filter_result") => Self::FilterResult,
            (Self::FilterResult, "expression") => Self::Value(JsonInputKind::Filter),
            _ => Self::Value(self.kind()),
        }
    }

    pub(super) fn element(self) -> Self {
        match self {
            // Only the results array gives its elements search-item structure.
            // Unrelated arrays inherit the enclosing value kind even if their
            // elements happen to contain fields named explain.
            Self::SearchItems => Self::SearchItem,
            _ => Self::Value(self.kind()),
        }
    }
}

// Budgeted writers can use the active schema kind to report an output error
// against the correct business field. Unmetered formatting sinks need no per-field
// state transition.
pub(super) trait JsonWrite: std::fmt::Write {
    fn set_kind(&mut self, _kind: JsonInputKind) {}
}

impl JsonWrite for String {}

#[derive(Clone, Copy)]
// Borrow already validated fields for one output row. In particular, a vector can
// be streamed from native coordinates without constructing a JSON array of number
// nodes.
pub(crate) enum JsonFieldValue<'a> {
    Json(&'a JsonValue),
    String(Option<&'a str>),
    Vector(&'a [f32]),
}

// The fixed row wrapper is not a business root. Json fields must already have
// been validated and reserved at their own roots by the caller.
pub(crate) fn write_validated_jsonl_fields<W: std::io::Write, const N: usize>(
    output: &mut W,
    mut fields: [(&str, JsonFieldValue<'_>); N],
    context: &JsonOperationContext,
) -> Result<()> {
    use std::fmt::Write;
    // Sorting the fixed-size field list puts the row in canonical key order
    // without allocating a map. It also makes duplicate names adjacent for the
    // pre-write validation.
    fields.sort_unstable_by_key(|(key, _)| *key);
    // Reject duplicate names and nonfinite native coordinates before even the
    // opening brace is written. The output stream cannot roll back a row prefix
    // once it has reached the destination.
    if fields.windows(2).any(|pair| pair[0].0 == pair[1].0)
        || fields.iter().any(|(_, value)| {
            matches!(value, JsonFieldValue::Vector(values) if values.iter().any(|value| !value.is_finite()))
        })
    {
        return Err(super::json_error("invalid JSONL output fields"));
    }
    let mut scratch = context.reservation();
    let mut frames = Vec::new();
    // Reserve traversal frames before writing any bytes. The validated JSON fields
    // share this stack serially, so one deep field does not require each sibling
    // to reserve a stack of its own.
    scratch.reserve_vec(
        &mut frames,
        super::MAX_JSON_DEPTH + 1,
        JsonInputKind::StoredJson,
    )?;
    let mut writer = JsonIoWriter {
        output,
        error: None,
    };
    let written = (|| {
        writer.write_char('{')?;
        for (index, (key, value)) in fields.iter().enumerate() {
            if index != 0 {
                writer.write_char(',')?;
            }
            super::write_string(key, &mut writer)?;
            writer.write_char(':')?;
            match value {
                // Serialize the validated field's DOM with compact punctuation.
                // Its source allocations remain borrowed, and only the traversal
                // frames are shared with this row writer.
                JsonFieldValue::Json(value) => super::write_json_with_frames(
                    value,
                    &mut writer,
                    0,
                    false,
                    &mut frames,
                    JsonOutputSchema::Value(JsonInputKind::StoredJson),
                )?,
                JsonFieldValue::String(Some(value)) => super::write_string(value, &mut writer)?,
                // A missing borrowed string has a defined on-wire value. Emit null
                // while keeping the field name, preserving the distinction from
                // both an empty string and an omitted member.
                JsonFieldValue::String(None) => writer.write_str("null")?,
                JsonFieldValue::Vector(values) => {
                    write_f32_array(values, &mut writer)?;
                }
            }
        }
        // Emit the newline as part of the JSONL row, including for an empty
        // object. The artifact hash and the temporary file's disk accounting
        // include that byte.
        writer.write_str("}\n")
    })();
    // Finish through the adapter so a destination failure retains the original I/O
    // error.
    writer.finish(written)
}

fn write_f32_array(values: &[f32], writer: &mut impl std::fmt::Write) -> std::fmt::Result {
    writer.write_char('[')?;
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            writer.write_char(',')?;
        }
        // Widen each f32 exactly before applying the common JSON number spelling.
        // Formatting directly at f32 precision can choose a different decimal
        // representation for the same stored coordinate.
        super::write_number(f64::from(*value), writer)?;
    }
    writer.write_char(']')
}

pub(crate) fn write_vector_json(output: &mut impl std::io::Write, values: &[f32]) -> Result<()> {
    // Scan all coordinates before emitting the opening bracket. A non-finite value
    // near the end of the vector is a validation error and must not leave a
    // partially written JSON array.
    if values.iter().any(|value| !value.is_finite()) {
        return Err(super::json_error(
            "JSON vectors must contain finite numbers",
        ));
    }
    // This entry writes just a JSON array to the supplied stream. It adds no row
    // terminator, so a caller embedding the vector in an enclosing format owns the
    // surrounding punctuation.
    let mut writer = JsonIoWriter {
        output,
        error: None,
    };
    // Stream the borrowed f32 slice with the same numeric spelling used for JSONL
    // vector fields. An empty vector still emits a valid empty array without
    // allocating a JsonValue per entry.
    let written = write_f32_array(values, &mut writer);
    writer.finish(written)
}

struct JsonIoWriter<'a, W> {
    // Destination receiving serialized UTF-8. It may already contain a partial
    // document if an I/O operation fails.
    output: &'a mut W,
    // Original I/O error saved while returning fmt::Error through the formatter.
    // finish recovers its detailed cause.
    error: Option<std::io::Error>,
}

impl<W: std::io::Write> JsonIoWriter<'_, W> {
    // Formatting errors do not carry their I/O cause. Prefer the saved destination
    // error, including a partial-write failure, before classifying a plain
    // formatter failure as a JSON error.
    fn finish(self, written: std::fmt::Result) -> Result<()> {
        if let Some(error) = self.error {
            return Err(error.into());
        }
        written.map_err(|_| super::json_error("JSON serialization failed"))
    }
}

impl<W: std::io::Write> std::fmt::Write for JsonIoWriter<'_, W> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        // Once a destination write has failed, stop sending further fragments and
        // preserve the first error. The formatter may know only that a previous
        // call returned fmt::Error.
        if self.error.is_some() {
            return Err(std::fmt::Error);
        }
        // A sink can legally accept only a prefix of this UTF-8 fragment.
        // write_all drives the remaining bytes or records an error, so a short
        // success cannot truncate a string unnoticed.
        self.output.write_all(value.as_bytes()).map_err(|error| {
            self.error = Some(error);
            std::fmt::Error
        })
    }
}

impl<W: std::io::Write> JsonWrite for JsonIoWriter<'_, W> {}

// Batch admission validates immutable values first. This lease covers one
// serialization at a time, including repeated digest/output passes.
pub(crate) struct JsonEncodingWorkspace {
    // Maximum number of serializer stack entries covered by the preflight for these
    // immutable values.
    frames: usize,
    // Charge for that many frames, retained across a digest pass and an output pass
    // that use the workspace in succession.
    _work: JsonReservation,
}

impl JsonEncodingWorkspace {
    pub(crate) fn new(
        frames: usize,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let mut work = context.reservation();
        // The preflight supplies a frame count, while the shared budget is in
        // bytes. Reject overflow during conversion before creating a workspace
        // that would claim fewer bytes than one real stack requires.
        let bytes = frames
            .checked_mul(std::mem::size_of::<super::WriteFrame<'_>>())
            .ok_or_else(|| context.limit_error(kind))?;
        work.grow(bytes, kind)?;
        // Retain the measured count and its grant here, not a shared mutable
        // stack. Returning this workspace does not start a serialization or
        // allocate an output buffer.
        Ok(Self {
            frames,
            _work: work,
        })
    }

    pub(crate) fn write(
        &self,
        value: &JsonValue,
        kind: JsonInputKind,
        output: &mut impl std::io::Write,
    ) -> Result<()> {
        // The lease survives across passes, but each serialization allocates its
        // own frame Vec on entry. Its allocation is covered by the caller's
        // one-at-a-time use of this workspace.
        let mut frames = Vec::new();
        // Allocate the native frame storage before attaching the writer to the
        // destination. An allocator failure at this point has not emitted a
        // partial document.
        frames
            .try_reserve_exact(self.frames)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        // try_reserve_exact permits capacity larger than requested. Reject such an
        // allocation before writing the first byte because the reusable lease
        // covers only the preflight frame count.
        if frames.capacity() != self.frames {
            return Err(std::io::Error::other("JSON frame capacity exceeds reservation").into());
        }
        let mut writer = JsonIoWriter {
            output,
            error: None,
        };
        // Use bounded traversal for the previously measured tree. A caller
        // supplying a deeper tree can encounter a frame error during output, so
        // the workspace must correspond to the values being serialized.
        let written = super::write_json_frames(
            value,
            &mut writer,
            0,
            false,
            &mut frames,
            JsonOutputSchema::Value(kind),
            true,
        );
        // With a failed underlying write the destination can contain a prefix.
        // Surface that original I/O error rather than interpreting it as a
        // shortage of traversal frames.
        writer.finish(written)
    }
}

impl JsonValue {
    pub(crate) fn write_value_retained_in(
        &self,
        output: &mut impl std::io::Write,
        kind: JsonInputKind,
        context: &JsonOperationContext,
        retained: &mut JsonReservation,
        pretty: bool,
    ) -> Result<()> {
        // Treat this retained value as its own business unit. An enclosing library
        // result containing several prevalidated units is reconciled through the
        // separate output-schema path below.
        let schema = JsonOutputSchema::Value(kind);
        // This path already owns the value's JSON reservation. Reconcile that
        // owner while checking canonical size and nesting so validation does not
        // charge a second copy of the live tree.
        let layout = self.validate_tree_layout(
            Some(super::MAX_JSON_DEPTH),
            0,
            Some(schema),
            Some(retained),
            true,
        )?;
        // Traversal storage has a shorter lifetime than the retained JSON. Admit
        // it under a separate scratch reservation before sending any bytes to a
        // potentially non-rewindable destination.
        let mut scratch = context.reservation();
        let mut frames = Vec::new();
        // Reserve one frame beyond maximum container depth for the leaf value. The
        // same preflight suffices for compact and pretty output because
        // indentation does not change traversal depth.
        scratch.reserve_vec(&mut frames, layout.depth + 1, kind)?;
        let mut writer = JsonIoWriter {
            output,
            error: None,
        };
        // The compact or pretty writer reuses the prepared frames. Pretty
        // whitespace affects emitted byte count while the admission check above
        // measured the value's canonical compact size.
        let written =
            super::write_json_with_frames(self, &mut writer, 0, pretty, &mut frames, schema);
        writer.finish(written)
    }

    // Reconcile the retained tree after assembly using the output schema. This
    // path updates the tree's reservation without allocating a serialization
    // buffer.
    pub(crate) fn reconcile_output_retained_in(
        &self,
        schema: JsonOutputSchema,
        retained: &mut JsonReservation,
    ) -> Result<()> {
        self.validate_tree_layout(None, 0, Some(schema), Some(retained), false)?;
        Ok(())
    }

    // Returns a pretty representation and the reservation that owns its capacity.
    // Callers retaining the representation must retain both values for the same
    // lifetime.
    pub(crate) fn pretty_retained_in(
        &self,
        schema: JsonOutputSchema,
        context: &JsonOperationContext,
        retained: &mut JsonReservation,
    ) -> Result<(String, JsonReservation)> {
        use std::fmt::Write;
        // Validate the output tree before formatting. Bound the write stack from
        // observed depth and charge scratch separately from the serialized string,
        // which survives this call.
        let layout = self.validate_tree_layout(None, 0, Some(schema), Some(retained), false)?;
        let mut scratch = context.reservation();
        let mut frames = Vec::new();
        // Allow one more frame than the measured container depth for the
        // traversal's current value.
        scratch.reserve_vec(&mut frames, layout.depth + 1, schema.kind())?;
        // Use the same writer for the trailing newline. Its capacity must fit the
        // output budget just like the JSON text.
        let mut out = JsonTextWriter::new(context, schema.kind());
        let written = super::write_json_with_frames(self, &mut out, 0, true, &mut frames, schema)
            .and_then(|()| out.write_char('\n'));
        out.finish(written)
    }

    // Validates the tree and reserves traversal memory before touching the
    // destination. An I/O failure can leave a prefix in the caller's writer.
    pub(crate) fn write_pretty_retained_in(
        &self,
        output: &mut impl std::io::Write,
        schema: JsonOutputSchema,
        context: &JsonOperationContext,
        retained: &mut JsonReservation,
    ) -> Result<()> {
        use std::fmt::Write;
        let layout = self.validate_tree_layout(None, 0, Some(schema), Some(retained), false)?;
        let mut scratch = context.reservation();
        let mut frames = Vec::new();
        scratch.reserve_vec(&mut frames, layout.depth + 1, schema.kind())?;
        // Adapt std::io::Write to the formatter while retaining the original I/O
        // error for reporting. This path streams bytes and has no retained String
        // reservation.
        let mut writer = JsonIoWriter {
            output,
            error: None,
        };
        let written =
            super::write_json_with_frames(self, &mut writer, 0, true, &mut frames, schema)
                // Emit the record terminator only after the document was written.
                // A newline failure still goes through the same I/O error adapter.
                .and_then(|()| writer.write_char('\n'));
        writer.finish(written)
    }
}

pub(crate) struct JsonTextWriter {
    // Owned serialized text accumulated so far, whose capacity must remain admitted
    // as it grows.
    value: String,
    // JSON bytes backing the String. On success this reservation leaves the writer
    // alongside the result.
    work: JsonReservation,
    // Current output field's kind for attributing a growth refusal during
    // traversal.
    kind: JsonInputKind,
    // Structured failure retained behind a fmt::Error, returned to the caller
    // before any generic formatting failure.
    error: Option<Error>,
}

impl JsonTextWriter {
    // Start with zero output capacity. Growth is admitted as the serializer
    // requests each fragment.
    pub(crate) fn new(context: &JsonOperationContext, kind: JsonInputKind) -> Self {
        Self {
            value: String::new(),
            work: context.reservation(),
            kind,
            error: None,
        }
    }

    pub(super) fn reserve(&mut self, additional: usize) -> Result<()> {
        self.work
            .reserve_string(&mut self.value, additional, self.kind)
    }

    pub(crate) fn finish(self, written: std::fmt::Result) -> Result<(String, JsonReservation)> {
        // Prefer the saved failure when the formatter returns its generic error.
        // On success, move the text and its capacity charge out together.
        if let Some(error) = self.error {
            return Err(error);
        }
        // Map formatter failures lacking a saved structured error to the generic
        // serialization error. This also covers failures originating outside
        // String growth.
        written.map_err(|_| super::json_error("JSON serialization failed"))?;
        Ok((self.value, self.work))
    }

    // Used for diagnostic labels of values already validated at a business root.
    pub(crate) fn write_validated_value(&mut self, value: &JsonValue) -> std::fmt::Result {
        // Keep the first failure as the diagnostic. A second request on the same
        // writer must leave that error and the incomplete text in place.
        if self.error.is_some() {
            return Err(std::fmt::Error);
        }
        // Scratch frames are temporary even though the diagnostic string can
        // outlive this call. The input was already validated, so the policy
        // ceiling bounds the stack here.
        let mut scratch = self.work.scratch();
        let mut frames = Vec::new();
        // Reject the diagnostic before appending any of its bytes when the
        // traversal slots cannot be reserved.
        if let Err(error) = scratch.reserve_vec(&mut frames, super::MAX_JSON_DEPTH + 1, self.kind) {
            self.error = Some(error);
            return Err(std::fmt::Error);
        }
        super::write_json_with_frames(
            value,
            self,
            0,
            // Render diagnostic labels in compact form. Pretty printing is
            // reserved for the output paths that explicitly request it.
            false,
            &mut frames,
            JsonOutputSchema::Value(self.kind),
        )
    }
}

impl std::fmt::Write for JsonTextWriter {
    // Every fragment passes through the same fallible admission path. Append only
    // after capacity is available, then reuse that capacity on subsequent calls.
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        if self.error.is_some() {
            return Err(std::fmt::Error);
        }
        if let Err(error) = self.reserve(value.len()) {
            self.error = Some(error);
            return Err(std::fmt::Error);
        }
        self.value.push_str(value);
        Ok(())
    }
}

impl JsonWrite for JsonTextWriter {
    // The output traversal updates kind as it enters each schema field. A refusal
    // while growing the String uses the active field's error classification.
    fn set_kind(&mut self, kind: JsonInputKind) {
        self.kind = kind;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonProcessingPolicyV1, WorkLedger};
    use std::fmt::Write;
    use std::sync::Arc;

    #[test]
    fn streamed_vectors_match_canonical_json_and_propagate_io_failures() {
        struct Limited {
            bytes: Vec<u8>,
            limit: usize,
            calls: usize,
        }
        impl std::io::Write for Limited {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.calls += 1;
                if self.calls % 7 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self.bytes.len() == self.limit {
                    return Err(std::io::Error::other("injected vector write failure"));
                }
                let count = bytes.len().min(13).min(self.limit - self.bytes.len());
                self.bytes.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for seed in 0..100u32 {
            let mut state = seed + 1;
            let mut values = vec![0.0, -0.0, f32::from_bits(1), f32::MIN, f32::MAX];
            for _ in 0..seed * 7 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let value = f32::from_bits(state);
                if value.is_finite() {
                    values.push(value);
                }
            }
            let expected = JsonValue::Array(
                values
                    .iter()
                    .map(|value| JsonValue::Number(f64::from(*value)))
                    .collect(),
            )
            .to_compact_string();
            let mut output = Limited {
                bytes: Vec::new(),
                limit: usize::MAX,
                calls: 0,
            };
            write_vector_json(&mut output, &values).unwrap();
            assert_eq!(output.bytes, expected.as_bytes());
            for limit in [0, 1, expected.len() / 2, expected.len() - 1] {
                let mut output = Limited {
                    bytes: Vec::new(),
                    limit,
                    calls: 0,
                };
                assert_eq!(
                    write_vector_json(&mut output, &values).unwrap_err().code(),
                    crate::ErrorCode::IoError
                );
                assert_eq!(output.bytes, expected.as_bytes()[..limit]);
            }
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut output = Vec::new();
            assert!(write_vector_json(&mut output, &[1.0, value]).is_err());
            assert!(output.is_empty());
        }
        let mut empty = Vec::new();
        write_vector_json(&mut empty, &[]).unwrap();
        assert_eq!(empty, b"[]");
    }

    fn search_envelope(item: JsonValue) -> JsonValue {
        crate::json::object_from_pairs([(
            "result",
            crate::json::object_from_pairs([("results", JsonValue::Array(vec![item]))]),
        )])
    }

    #[test]
    fn admitted_encoding_reuses_fixed_credit_and_never_grows_past_its_frames() {
        use crate::json::{JsonProcessingPolicyV1, WorkLedger};
        for seed in 0..100 {
            let ledger = std::sync::Arc::new(WorkLedger::default());
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let mut value = crate::json::object_from_pairs([
                ("text", JsonValue::String(format!("{seed}\n\t\\\"\u{4e2d}"))),
                ("number", JsonValue::Number(seed as f64 / 3.0)),
            ]);
            let depth = if seed == 99 { 127 } else { seed };
            for _ in 0..depth {
                value = JsonValue::Array(vec![value]);
            }
            let expected = value.to_compact_string();
            let kind = JsonInputKind::DocumentMetadata;
            let (source, _) = value.guard_in(kind, &context).unwrap();
            let (frames, bytes) = value.encoding_shape_in(kind, &context).unwrap();
            assert_eq!(bytes, source.reserved_bytes());
            assert_eq!(frames, depth + 2);
            let workspace = JsonEncodingWorkspace::new(frames, kind, &context).unwrap();
            let credit = workspace._work.reserved_bytes();
            assert_eq!(
                credit,
                frames * std::mem::size_of::<super::super::WriteFrame<'_>>()
            );
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let mut held = other.workspace_reservation();
            held.grow_to(0, limit, "operation_scratch").unwrap();
            assert!(other
                .workspace_reservation()
                .grow_to(1, 1, "operation_scratch")
                .is_err());
            for _ in 0..3 {
                let mut output = Vec::new();
                workspace.write(&value, kind, &mut output).unwrap();
                assert_eq!(output, expected.as_bytes());
                assert_eq!(workspace._work.reserved_bytes(), credit);
            }
            // A mismatched internal plan is bounded too; it cannot silently grow.
            let oversized = JsonValue::Array(vec![value]);
            assert!(workspace
                .write(&oversized, kind, &mut std::io::sink())
                .is_err());
            assert_eq!(workspace._work.reserved_bytes(), credit);
            drop(oversized);
            drop(source);
            drop(workspace);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
        let context = JsonOperationContext::standalone();
        let mut deep = JsonValue::Null;
        for _ in 0..129 {
            deep = JsonValue::Array(vec![deep]);
        }
        let wide = JsonValue::String(
            "x".repeat(crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES),
        );
        for value in [deep, wide, JsonValue::Number(f64::NAN)] {
            assert!(value
                .encoding_shape_in(JsonInputKind::ChunkMetadata, &context)
                .is_err());
            drop(value);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
    }

    #[test]
    fn streamed_pretty_reports_preserve_bytes_and_release_scratch_on_io_failure() {
        struct Output {
            bytes: Vec<u8>,
            calls: usize,
            limit: usize,
        }
        impl std::io::Write for Output {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.calls += 1;
                if self.calls % 7 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self.bytes.len() == self.limit {
                    return Err(std::io::ErrorKind::BrokenPipe.into());
                }
                let count = bytes.len().min(13).min(self.limit - self.bytes.len());
                self.bytes.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let schema = JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact);
            let mut value = crate::json::object_from_pairs([
                ("text", JsonValue::String(format!("{seed}\n\t\\\"\u{4e2d}"))),
                (
                    "values",
                    JsonValue::Array(vec![
                        JsonValue::Number(seed as f64 / 3.0),
                        JsonValue::Number(-0.0),
                        JsonValue::Null,
                    ]),
                ),
            ]);
            for _ in 0..if seed == 99 { 131 } else { seed % 7 } {
                value = JsonValue::Array(vec![value]);
            }
            let expected = value.to_pretty_string() + "\n";
            let mut retained = context.reservation();
            value
                .reconcile_output_retained_in(schema, &mut retained)
                .unwrap();
            let retained_bytes = retained.reserved_bytes();
            for limit in [0, 1, 17, expected.len() - 1, usize::MAX] {
                let mut output = Output {
                    bytes: Vec::new(),
                    calls: 0,
                    limit,
                };
                let result =
                    value.write_pretty_retained_in(&mut output, schema, &context, &mut retained);
                if limit == usize::MAX {
                    result.unwrap();
                    assert_eq!(output.bytes, expected.as_bytes());
                } else {
                    assert_eq!(result.unwrap_err().code(), crate::ErrorCode::IoError);
                    assert_eq!(output.bytes, expected.as_bytes()[..limit]);
                }
                assert_eq!(retained.reserved_bytes(), retained_bytes);
                context
                    .reservation()
                    .grow(
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained_bytes,
                        JsonInputKind::BenchmarkArtifact,
                    )
                    .unwrap();
            }
        }
    }

    #[test]
    fn streamed_rows_preserve_canonical_bytes_without_copying_text_or_vectors() {
        use crate::json::object_from_pairs;
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let text = format!("\u{4e2d}\u{1f600}\"\\\n\u{0001}{seed}");
            let vector = [-0.0, seed as f32 / 7.0, f32::MIN_POSITIVE, f32::MAX];
            let metadata =
                object_from_pairs([("key", JsonValue::Array(vec![JsonValue::Bool(true); seed]))]);
            let _work = metadata
                .guard_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            let expected = object_from_pairs([
                ("text", JsonValue::String(text.clone())),
                ("metadata", metadata.clone()),
                (
                    "vector",
                    JsonValue::Array(
                        vector
                            .iter()
                            .map(|x| JsonValue::Number(f64::from(*x)))
                            .collect(),
                    ),
                ),
                ("optional", JsonValue::Null),
            ])
            .to_compact_string()
                + "\n";
            let mut output = Vec::new();
            write_validated_jsonl_fields(
                &mut output,
                [
                    ("vector", JsonFieldValue::Vector(&vector)),
                    ("text", JsonFieldValue::String(Some(&text))),
                    ("optional", JsonFieldValue::String(None)),
                    ("metadata", JsonFieldValue::Json(&metadata)),
                ],
                &context,
            )
            .unwrap();
            assert_eq!(output, expected.as_bytes());
        }

        struct Counter(usize);
        impl std::io::Write for Counter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 += bytes.len();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let context = JsonOperationContext::standalone();
        let mut busy = context.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 32 * 1024,
            JsonInputKind::StoredJson,
        )
        .unwrap();
        let text = "x".repeat(12 * 1024 * 1024);
        let vector = vec![1.0; 500_000];
        let mut output = Counter(0);
        write_validated_jsonl_fields(
            &mut output,
            [
                ("text", JsonFieldValue::String(Some(&text))),
                ("vector", JsonFieldValue::Vector(&vector)),
            ],
            &context,
        )
        .unwrap();
        assert!(output.0 > text.len() + vector.len());
        busy.grow(32 * 1024, JsonInputKind::StoredJson).unwrap();
    }

    #[test]
    fn streaming_output_handles_short_writes_interrupts_and_preserves_io_errors() {
        struct Writer {
            bytes: Vec<u8>,
            calls: usize,
            fail_after: Option<usize>,
        }
        impl std::io::Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.calls += 1;
                if self.calls % 2 == 1 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self
                    .fail_after
                    .map(|n| self.bytes.len() >= n)
                    .unwrap_or(false)
                {
                    return Err(std::io::ErrorKind::BrokenPipe.into());
                }
                let count = bytes.len().min(3);
                self.bytes.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let context = JsonOperationContext::standalone();
        let value = crate::json::object_from_pairs([(
            "unicode",
            JsonValue::String("\u{4e2d}\n\\\"".into()),
        )]);
        let mut retained = value
            .guard_in(JsonInputKind::StoredJson, &context)
            .unwrap()
            .0;
        for pretty in [false, true] {
            let mut writer = Writer {
                bytes: Vec::new(),
                calls: 0,
                fail_after: None,
            };
            value
                .write_value_retained_in(
                    &mut writer,
                    JsonInputKind::StoredJson,
                    &context,
                    &mut retained,
                    pretty,
                )
                .unwrap();
            assert_eq!(
                writer.bytes,
                if pretty {
                    value.to_pretty_string()
                } else {
                    value.to_compact_string()
                }
                .as_bytes()
            );
            let mut writer = Writer {
                bytes: Vec::new(),
                calls: 0,
                fail_after: Some(5),
            };
            assert_eq!(
                value
                    .write_value_retained_in(
                        &mut writer,
                        JsonInputKind::StoredJson,
                        &context,
                        &mut retained,
                        pretty
                    )
                    .unwrap_err()
                    .code(),
                crate::ErrorCode::IoError
            );
            assert!(writer.bytes.len() < value.to_compact_string().len());
        }
        let mut writer = Writer {
            bytes: Vec::new(),
            calls: 0,
            fail_after: Some(5),
        };
        assert_eq!(
            write_validated_jsonl_fields(
                &mut writer,
                [("metadata", JsonFieldValue::Json(&value))],
                &context
            )
            .unwrap_err()
            .code(),
            crate::ErrorCode::IoError
        );
        drop(retained);
        context
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::StoredJson,
            )
            .unwrap();
    }

    #[test]
    fn streaming_output_rejects_limits_before_writing_and_keeps_business_roots() {
        let context = JsonOperationContext::standalone();
        let too_large =
            JsonValue::String("x".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES));
        let mut retained = context.reservation();
        let mut output = Vec::new();
        let error = too_large
            .write_value_retained_in(
                &mut output,
                JsonInputKind::StoredJson,
                &context,
                &mut retained,
                false,
            )
            .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
        assert!(output.is_empty());
        drop(retained);
        let mut busy = context.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::StoredJson,
        )
        .unwrap();
        assert!(write_validated_jsonl_fields(
            &mut output,
            [("text", JsonFieldValue::String(Some("x")))],
            &context
        )
        .is_err());
        assert!(output.is_empty());
        drop(busy);
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                let context = JsonOperationContext::standalone();
                let mut metadata = JsonValue::Null;
                for _ in 0..128 {
                    metadata = JsonValue::Array(vec![metadata]);
                }
                let _work = metadata
                    .guard_in(JsonInputKind::StoredJson, &context)
                    .unwrap();
                let mut output = Vec::new();
                write_validated_jsonl_fields(
                    &mut output,
                    [("metadata", JsonFieldValue::Json(&metadata))],
                    &context,
                )
                .unwrap();
                assert_eq!(output.len(), 128 * 2 + 18);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn retained_pretty_output_preserves_canonical_tokens_and_shared_capacity() {
        use crate::json::object_from_pairs;
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut work = context.reservation();
            let metadata = object_from_pairs([
                (
                    "unicode",
                    JsonValue::String(format!("\u{4e2d}\u{1f600}\n\\\"{seed}")),
                ),
                (
                    "array",
                    JsonValue::Array(vec![JsonValue::Number(-0.0); seed]),
                ),
                ("fraction", JsonValue::Number(seed as f64 / 7.0)),
                ("empty", JsonValue::object()),
            ]);
            work.merge(
                metadata
                    .guard_in(JsonInputKind::StoredJson, &context)
                    .unwrap()
                    .0,
            );
            let value = search_envelope(object_from_pairs([("document_metadata", metadata)]));
            let (actual, output_work) = value
                .pretty_retained_in(JsonOutputSchema::SearchEnvelope, &context, &mut work)
                .unwrap();
            assert_eq!(actual, value.to_pretty_string() + "\n");
            assert_eq!(output_work.reserved_bytes(), actual.capacity());
            let measured = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            drop(measured);
            let mut remaining = context.reservation();
            remaining
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                        - work.reserved_bytes()
                        - output_work.reserved_bytes(),
                    JsonInputKind::StoredJson,
                )
                .unwrap();
            assert!(context
                .reservation()
                .grow(1, JsonInputKind::StoredJson)
                .is_err());
        }
    }

    #[test]
    fn fixed_output_envelope_does_not_consume_business_depth_or_unit_bytes() {
        use crate::json::object_from_pairs;
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                let context = JsonOperationContext::standalone();
                let mut metadata = JsonValue::Null;
                for _ in 0..127 {
                    metadata = JsonValue::Array(vec![metadata]);
                }
                let metadata = object_from_pairs([("nested", metadata)]);
                let mut work = metadata
                    .guard_in(JsonInputKind::StoredJson, &context)
                    .unwrap()
                    .0;
                let value = search_envelope(object_from_pairs([("document_metadata", metadata)]));
                let (text, _output_work) = value
                    .pretty_retained_in(JsonOutputSchema::SearchEnvelope, &context, &mut work)
                    .unwrap();
                assert_eq!(text, value.to_pretty_string() + "\n");
            })
            .unwrap()
            .join()
            .unwrap();

        let context = JsonOperationContext::standalone();
        let mut work = context.reservation();
        let metadata =
            object_from_pairs([("payload", JsonValue::String("x".repeat(5 * 1024 * 1024)))]);
        let other = metadata.clone();
        for value in [&metadata, &other] {
            work.merge(
                value
                    .guard_in(JsonInputKind::StoredJson, &context)
                    .unwrap()
                    .0,
            );
        }
        let value = search_envelope(object_from_pairs([
            ("document_metadata", metadata),
            ("chunk_metadata", other),
        ]));
        let (text, _output_work) = value
            .pretty_retained_in(JsonOutputSchema::SearchEnvelope, &context, &mut work)
            .unwrap();
        assert!(text.len() > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES);
        assert_eq!(text, value.to_pretty_string() + "\n");
    }

    #[test]
    fn output_budget_failures_keep_the_known_business_kind_and_release_scratch() {
        use crate::json::object_from_pairs;
        for (schema, filter) in [
            (JsonOutputSchema::SearchEnvelope, false),
            (JsonOutputSchema::SearchEnvelope, true),
            (JsonOutputSchema::WhyNotEnvelope, true),
        ] {
            let context = JsonOperationContext::standalone();
            let leaf = object_from_pairs([("payload", JsonValue::String("x".repeat(64 * 1024)))]);
            let value = match (schema, filter) {
                (JsonOutputSchema::WhyNotEnvelope, _) => object_from_pairs([(
                    "result",
                    object_from_pairs([(
                        "filter_result",
                        object_from_pairs([("expression", leaf)]),
                    )]),
                )]),
                (_, true) => search_envelope(object_from_pairs([(
                    "explain",
                    object_from_pairs([("filter", object_from_pairs([("expression", leaf)]))]),
                )])),
                (_, false) => search_envelope(object_from_pairs([("document_metadata", leaf)])),
            };
            let mut work = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            let before = value.to_compact_string();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes() - 4096,
                JsonInputKind::StoredJson,
            )
            .unwrap();
            let error = value
                .pretty_retained_in(schema, &context, &mut work)
                .err()
                .unwrap();
            assert_eq!(
                error.code(),
                if filter {
                    crate::ErrorCode::InvalidMetadataFilter
                } else {
                    crate::ErrorCode::RecoveryError
                }
            );
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("json_input_kind")
                    .and_then(JsonValue::as_str),
                Some(if filter { "filter" } else { "stored_json" })
            );
            assert_eq!(value.to_compact_string(), before);
            context
                .reservation()
                .grow(4096, JsonInputKind::StoredJson)
                .unwrap();
            drop(busy);
            drop(value);
            drop(work);
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::StoredJson,
                )
                .unwrap();
        }
    }

    #[test]
    fn text_output_uses_effective_budget_and_never_appends_a_failed_write() {
        let ledger = Arc::new(WorkLedger::default());
        let other = JsonOperationContext::new(Arc::clone(&ledger));
        let mut busy = other.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 4096,
            JsonInputKind::Filter,
        )
        .unwrap();
        let context = JsonOperationContext::new(ledger);
        let mut writer = JsonTextWriter::new(&context, JsonInputKind::StoredJson);
        writer.reserve(4096).unwrap();
        writer.write_str(&"x".repeat(4096)).unwrap();
        let written = writer.write_str("\u{1f600}");
        assert!(written.is_err());
        assert_eq!(writer.value.len(), 4096);
        // A caller ignoring fmt::Error still cannot receive truncated output.
        let error = writer.finish(Ok(())).err().unwrap();
        assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
        assert_eq!(
            error.to_json().get("detail").unwrap().get("limit"),
            Some(&crate::JsonValue::Number(4096.0))
        );
        context
            .reservation()
            .grow(4096, JsonInputKind::StoredJson)
            .unwrap();
    }

    #[test]
    fn native_text_output_failure_keeps_system_error_through_formatting_and_finish() {
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::Filter,
            JsonInputKind::StoredJson,
        ] {
            let payload = "x".repeat(8193);
            let value = JsonValue::String(payload.clone());
            let canonical = format!("\"{payload}\"");
            let error = crate::test_alloc::fail_byte_allocation_once(canonical.len(), || {
                value.try_to_compact_string_with_kind(kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(
                value.try_to_compact_string_with_kind(kind).unwrap(),
                canonical
            );
            let context = JsonOperationContext::standalone();
            let capacity = context.available_workspace_bytes();
            let mut writer = JsonTextWriter::new(&context, kind);
            writer.write_str("safe").unwrap();
            let old_capacity = writer.value.capacity();
            let written = crate::test_alloc::fail_byte_allocation_once(4 + payload.len(), || {
                writer.write_str(&payload)
            });
            assert!(written.is_err());
            assert_eq!(writer.value, "safe");
            assert_eq!(writer.value.capacity(), old_capacity);
            assert_eq!(context.available_workspace_bytes(), capacity - old_capacity);
            // Even a caller ignoring fmt::Error cannot receive partial output.
            let error = writer.finish(Ok(())).err().unwrap();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert!(!super::super::is_memory_budget_error(&error));
            assert_eq!(context.available_workspace_bytes(), capacity);
            let mut retry = JsonTextWriter::new(&context, kind);
            retry.write_str(&payload).unwrap();
            let (text, work) = retry.finish(Ok(())).unwrap();
            assert_eq!(text, payload);
            drop(text);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), capacity);
        }
    }

    #[test]
    fn text_output_retains_exact_capacity_until_the_consumer_releases_it() {
        let context = JsonOperationContext::standalone();
        let mut writer = JsonTextWriter::new(&context, JsonInputKind::Filter);
        for value in ["ascii", "\u{4e2d}\u{1f600}", "\n\\\""] {
            writer.write_str(value).unwrap();
        }
        let (value, work) = writer.finish(Ok(())).unwrap();
        assert_eq!(value, "ascii\u{4e2d}\u{1f600}\n\\\"");
        assert_eq!(work.reserved_bytes(), value.capacity());
        let mut remaining = context.reservation();
        remaining
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - value.capacity(),
                JsonInputKind::Filter,
            )
            .unwrap();
        assert!(context
            .reservation()
            .grow(1, JsonInputKind::Filter)
            .is_err());
        drop(value);
        drop(work);
        drop(remaining);
        context
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Filter,
            )
            .unwrap();
    }
}
