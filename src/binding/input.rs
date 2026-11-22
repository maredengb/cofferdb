use crate::json::{
    JsonInputKind, JsonLimitReason, JsonOperationContext, JsonProcessingPolicyV1, ReservedVec,
};
use crate::storage::RecordReservation;
use crate::{JsonObject, JsonValue, Result};

/// An existing core business-value category, without a separate host policy.
#[derive(Clone, Copy)]
pub struct ValueKind(JsonInputKind);

impl ValueKind {
    /// Apply the engine's document-metadata limits while converting this host
    /// value.
    pub const DOCUMENT_METADATA: Self = Self(JsonInputKind::DocumentMetadata);
    /// Apply the metadata limits for a chunk row, including its nested JSON
    /// values.
    pub const CHUNK_METADATA: Self = Self(JsonInputKind::ChunkMetadata);
    /// Use the filter input policy when constructing a search predicate.
    pub const FILTER: Self = Self(JsonInputKind::Filter);
    /// Use the search-weight policy for the host's requested scoring coefficients.
    pub const WEIGHTS: Self = Self(JsonInputKind::Weights);
    /// Use the metadata-boost policy for the rules supplied with a search request.
    pub const METADATA_BOOST: Self = Self(JsonInputKind::MetadataBoost);
    /// Apply query embedding limits to values constructed for vector search.
    pub const QUERY_EMBEDDING: Self = Self(JsonInputKind::QueryEmbedding);
    /// Use the core JSONL record policy when building a full input record.
    pub const JSONL_RECORD: Self = Self(JsonInputKind::JsonlRecord);
    /// Apply database runtime configuration limits to a converted settings value.
    pub const RUNTIME_CONFIG: Self = Self(JsonInputKind::RuntimeConfig);
    /// Apply collection configuration limits to values prepared for collection
    /// creation.
    pub const COLLECTION_CONFIG: Self = Self(JsonInputKind::CollectionConfig);
}

/// Aggregate credit for input constructed in place by a native binding.
pub struct Input {
    // Operation that admits each buffer built from host values and identifies which
    // native call may consume the completed argument.
    context: JsonOperationContext,
    // Aggregates JSON tree allocations and ordinary workspace strings that have
    // already been constructed for this input.
    work: RecordReservation,
}

/// An owned argument and its existing core reservation travel together.
pub struct Argument<T> {
    // Owned typed value or JSON tree to be moved to the engine without borrowing
    // any host object.
    pub(crate) value: T,
    // Credit for the allocations inside value, retained during validation and
    // transferred when a mapped argument changes shape.
    pub(crate) work: RecordReservation,
}

impl<T> Argument<T> {
    // Move the reservation with the converted value. Mapping an owned host record
    // to an import row changes the Rust type without changing which operation paid
    // for its buffers.
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
    /// Push a pending traversal item, reserving space against the operation before
    /// extending the stack.
    ///
    /// # Errors
    ///
    /// Returns an error if the stack cannot admit or allocate the required
    /// capacity.
    pub fn push(&mut self, value: T) -> Result<()> {
        self.0.try_push(value)
    }

    /// Remove the most recently pushed traversal item, or return `None` for an
    /// empty stack.
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    /// Borrows the top traversal item for in-place updates without removing it.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.0.as_mut_slice().last_mut()
    }

    /// Borrow the live traversal items in push order, from the bottom of the stack
    /// to the top.
    pub fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl Input {
    // Track native JSON allocations and ordinary workspace separately under the
    // same operation identity. A completed argument carries both portions when it
    // leaves this builder.
    pub(super) fn new(context: &JsonOperationContext) -> Self {
        Self {
            context: context.clone(),
            work: RecordReservation::new(context.reservation(), context.workspace_reservation()),
        }
    }

    /// Creates an empty traversal stack under this operation. The stack reserves
    /// backing storage when a push needs more capacity.
    pub fn stack<T>(&self) -> Result<Stack<T>> {
        ReservedVec::with_capacity(0, &self.context, "operation_scratch").map(Stack)
    }

    /// Copy native UTF-8 text and retain its workspace charge in the input
    /// builder. Use the JSON-string conversion when a business value's JSON limits
    /// also apply.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy cannot be admitted or allocated.
    pub fn text(&mut self, text: &str) -> Result<String> {
        self.work.copy_string(text, &self.context)
    }

    /// Encode borrowed Unicode scalars directly into one admitted native string.
    /// The cloned iterator must yield the same sequence. Its UTF-8 byte count
    /// sets the reservation before the second pass writes the string.
    pub fn text_from_chars(&mut self, chars: impl Iterator<Item = char> + Clone) -> Result<String> {
        let bytes = chars
            .clone()
            .try_fold(0usize, |sum, value| sum.checked_add(value.len_utf8()))
            .ok_or_else(|| std::io::Error::other("UTF-8 string length overflow"))?;
        // Use the exact-capacity renderer so the final length agrees with the
        // sizing pass. A different byte length on the second visit is an error.
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
        // Transfer the workspace lease into the builder before returning the
        // string. Its capacity remains charged until the completed argument
        // changes owner.
        let (text, work) = text.into_parts();
        self.work.merge_workspace(work);
        Ok(text)
    }

    /// Copies text into a JSON string allocation charged under the selected value
    /// kind.
    pub fn json_string(&mut self, text: &str, kind: ValueKind) -> Result<String> {
        self.work.json.copy_string(text, kind.0)
    }

    /// Encodes a borrowed scalar stream into a charged JSON string. Cloned
    /// iterators must yield the same sequence for size measurement and rendering.
    pub fn json_string_from_chars(
        &mut self,
        chars: impl Iterator<Item = char> + Clone,
        kind: ValueKind,
    ) -> Result<String> {
        // Use JSON allocation accounting here, unlike the ordinary text identifier
        // path. The kind selects limit details for this business input.
        self.work.json.format_string(kind.0, |out| {
            for value in chars.clone() {
                out.write_char(value)?;
            }
            Ok(())
        })
    }

    /// Returns an empty JSON array backing vector with room admitted for the
    /// requested number of elements. Fill it under the same builder before handing
    /// it to the operation.
    pub fn array(&mut self, length: usize, kind: ValueKind) -> Result<Vec<JsonValue>> {
        let mut values = Vec::new();
        self.work.json.reserve_vec(&mut values, length, kind.0)?;
        Ok(values)
    }

    /// Returns an empty JSON object after charging for its expected nodes. Keys
    /// and value payloads must be admitted separately.
    pub fn object(&mut self, length: usize, kind: ValueKind) -> Result<JsonObject> {
        self.work.json.reserve_object_nodes(length, kind.0)?;
        Ok(JsonObject::new())
    }

    /// Reserves the typed binary32 vector slots. The caller still validates
    /// coordinate values when building the host argument.
    pub fn vector(&mut self, length: usize, kind: ValueKind) -> Result<Vec<f32>> {
        let mut vector = Vec::new();
        self.work.json.reserve_vec(&mut vector, length, kind.0)?;
        Ok(vector)
    }

    // The caller supplies the number of enclosing containers before descending.
    // Report the proposed additional level so a root value assembled from nested
    // host containers obeys the same limit as parsed JSON.
    /// Checks whether another container can open beneath `depth` enclosing
    /// containers.
    ///
    /// # Errors
    ///
    /// Returns the selected kind's JSON depth error when the proposed level
    /// exceeds the production limit.
    pub fn check_container_depth(&self, depth: usize, kind: ValueKind) -> Result<()> {
        // Compare the number of existing ancestors before opening the next
        // container. The scalar child of a legal deepest container does not add
        // another level.
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
    /// # Errors
    ///
    /// Returns an error when traversal, canonical encoding, or JSON processing
    /// admission rejects the typed value.
    pub fn validate_json(&self, value: &JsonValue, kind: ValueKind) -> Result<()> {
        // Measure the typed tree through the same encoding rules used for decoded
        // JSON. A retained value's canonical-size limit is independent of its
        // allocation capacity.
        value.encoding_shape_in(kind.0, &self.context).map(|_| ())
    }

    /// Move arguments constructed through this builder into the core adapter.
    /// The caller must not mutate or replace admitted allocations before handoff.
    pub fn finish<T>(self, value: T) -> Argument<T> {
        // Return value and accounting as one argument. Operation ownership is
        // checked against this reservation during execution.
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
