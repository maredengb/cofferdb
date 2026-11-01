use crate::json::{
    JsonInputKind, JsonLimitReason, JsonOperationContext, JsonProcessingPolicyV1, ReservedVec,
};
use crate::storage::RecordReservation;
use crate::{JsonObject, JsonValue, Result};

/// An existing core business-value category, without a separate host policy.
#[derive(Clone, Copy)]
pub struct ValueKind(JsonInputKind);

impl ValueKind {
    pub const DOCUMENT_METADATA: Self = Self(JsonInputKind::DocumentMetadata);
    pub const CHUNK_METADATA: Self = Self(JsonInputKind::ChunkMetadata);
    pub const FILTER: Self = Self(JsonInputKind::Filter);
    pub const WEIGHTS: Self = Self(JsonInputKind::Weights);
    pub const METADATA_BOOST: Self = Self(JsonInputKind::MetadataBoost);
    pub const QUERY_EMBEDDING: Self = Self(JsonInputKind::QueryEmbedding);
    pub const JSONL_RECORD: Self = Self(JsonInputKind::JsonlRecord);
    pub const RUNTIME_CONFIG: Self = Self(JsonInputKind::RuntimeConfig);
    pub const COLLECTION_CONFIG: Self = Self(JsonInputKind::CollectionConfig);
}

/// Aggregate credit for input constructed in place by a native binding.
pub struct Input {
    context: JsonOperationContext,
    work: RecordReservation,
}

/// An owned argument and its existing core reservation travel together.
pub struct Argument<T> {
    pub(crate) value: T,
    pub(crate) work: RecordReservation,
}

impl<T> Argument<T> {
    pub(super) fn map<U>(self, convert: impl FnOnce(T) -> U) -> Argument<U> {
        Argument {
            value: convert(self.value),
            work: self.work,
        }
    }
}

/// A bounded traversal stack using the existing core workspace container.
pub struct Stack<T>(ReservedVec<T>);

impl<T> Stack<T> {
    pub fn push(&mut self, value: T) -> Result<()> {
        self.0.try_push(value)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.0.as_mut_slice().last_mut()
    }

    pub fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl Input {
    pub(super) fn new(context: &JsonOperationContext) -> Self {
        Self {
            context: context.clone(),
            work: RecordReservation::new(context.reservation(), context.workspace_reservation()),
        }
    }

    pub fn stack<T>(&self) -> Result<Stack<T>> {
        ReservedVec::with_capacity(0, &self.context, "operation_scratch").map(Stack)
    }

    pub fn text(&mut self, text: &str) -> Result<String> {
        self.work.copy_string(text, &self.context)
    }

    /// Encode borrowed Unicode scalars directly into one admitted native string.
    pub fn text_from_chars(&mut self, chars: impl Iterator<Item = char> + Clone) -> Result<String> {
        let bytes = chars
            .clone()
            .try_fold(0usize, |sum, value| sum.checked_add(value.len_utf8()))
            .ok_or_else(|| std::io::Error::other("UTF-8 string length overflow"))?;
        let text = crate::json::ReservedString::build_exact(
            bytes,
            &self.context,
            "operation_scratch",
            |out| {
                for value in chars {
                    out.write_char(value)?;
                }
                Ok(())
            },
        )?;
        let (text, work) = text.into_parts();
        self.work.merge_workspace(work);
        Ok(text)
    }

    pub fn json_string(&mut self, text: &str, kind: ValueKind) -> Result<String> {
        self.work.json.copy_string(text, kind.0)
    }

    pub fn json_string_from_chars(
        &mut self,
        chars: impl Iterator<Item = char> + Clone,
        kind: ValueKind,
    ) -> Result<String> {
        self.work.json.format_string(kind.0, |out| {
            for value in chars.clone() {
                out.write_char(value)?;
            }
            Ok(())
        })
    }

    pub fn array(&mut self, length: usize, kind: ValueKind) -> Result<Vec<JsonValue>> {
        let mut values = Vec::new();
        self.work.json.reserve_vec(&mut values, length, kind.0)?;
        Ok(values)
    }

    pub fn object(&mut self, length: usize, kind: ValueKind) -> Result<JsonObject> {
        self.work.json.reserve_object_nodes(length, kind.0)?;
        Ok(JsonObject::new())
    }

    pub fn vector(&mut self, length: usize, kind: ValueKind) -> Result<Vec<f32>> {
        let mut vector = Vec::new();
        self.work.json.reserve_vec(&mut vector, length, kind.0)?;
        Ok(vector)
    }

    pub fn check_container_depth(&self, depth: usize, kind: ValueKind) -> Result<()> {
        if depth >= JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH {
            return Err(kind.0.limit_error(
                JsonLimitReason::Depth,
                JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH,
                Some(depth.saturating_add(1)),
            ));
        }
        Ok(())
    }

    /// Validate a completed business value while all owned capacity stays held.
    pub fn validate_json(&self, value: &JsonValue, kind: ValueKind) -> Result<()> {
        value.encoding_shape_in(kind.0, &self.context).map(|_| ())
    }

    /// Move arguments constructed through this builder into the core adapter.
    /// The caller must not mutate or replace admitted allocations before handoff.
    pub fn finish<T>(self, value: T) -> Argument<T> {
        Argument {
            value,
            work: self.work,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_and_json_inputs_share_credit_without_applying_the_unit_limit_to_text() {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut first = Input::new(&context);
        let text = first.text(&"t".repeat(9 * 1024 * 1024)).unwrap();
        let argument = first.finish(text);
        assert_eq!(
            context.available_workspace_bytes(),
            available - 9 * 1024 * 1024
        );
        let mut second = Input::new(&context);
        let error = second
            .array(
                available / std::mem::size_of::<JsonValue>(),
                ValueKind::FILTER,
            )
            .unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "filter"));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        drop(second);
        drop(argument);
        assert_eq!(context.available_workspace_bytes(), available);
    }

    #[test]
    fn business_depth_and_canonical_limits_keep_the_core_error_category() {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut input = Input::new(&context);
        let kind = ValueKind::DOCUMENT_METADATA;
        let mut value = JsonValue::Null;
        for depth in 0..JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH {
            input.check_container_depth(depth, kind).unwrap();
            let mut array = input.array(1, kind).unwrap();
            array.push(value);
            value = JsonValue::Array(array);
        }
        input.validate_json(&value, kind).unwrap();
        let error = input
            .check_container_depth(JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH, kind)
            .unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "depth_exceeded"));
        drop(value);
        drop(input);
        assert_eq!(context.available_workspace_bytes(), available);

        let mut input = Input::new(&context);
        let text = input
            .json_string(
                &"x".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES),
                kind,
            )
            .unwrap();
        let value = JsonValue::String(text);
        let error = input.validate_json(&value, kind).unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "unit_bytes_exceeded"));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "document_metadata"));
        drop(value);
        drop(input);
        assert_eq!(context.available_workspace_bytes(), available);
    }

    #[test]
    fn unicode_scalar_encoding_has_one_owned_buffer_and_shared_admission() {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let expected = "ascii\0\u{e9}\u{4e2d}\u{1f642}\u{10ffff}";
        let mut input = Input::new(&context);
        let text = input.text_from_chars(expected.chars()).unwrap();
        let json_text = input
            .json_string_from_chars(expected.chars(), ValueKind::DOCUMENT_METADATA)
            .unwrap();
        assert_eq!(text, expected);
        assert_eq!(json_text, expected);
        let encoded = JsonValue::String(json_text);
        input
            .validate_json(&encoded, ValueKind::DOCUMENT_METADATA)
            .unwrap();
        drop(encoded);
        drop(text);
        drop(input);
        assert_eq!(context.available_workspace_bytes(), available);
        let mut held = context.workspace_reservation();
        held.grow_to(available - 4, available - 4, "operation_scratch")
            .unwrap();
        let mut input = Input::new(&context);
        let error = input.text_from_chars(expected.chars()).unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        assert_eq!(context.available_workspace_bytes(), 4);
        drop(input);
        drop(held);
        assert_eq!(context.available_workspace_bytes(), available);
    }
}
