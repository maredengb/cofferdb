use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Cursor, Read};

use crate::error::{Error, ErrorCode, Result};

mod build;
mod copy;
mod lines;
mod output;
mod policy;
mod work;
pub(crate) use build::JsonBuilder;
pub(crate) use lines::JsonLines;
use output::JsonWrite;
pub(crate) use output::{
    write_validated_jsonl_fields, write_vector_json, JsonEncodingWorkspace, JsonFieldValue,
    JsonOutputSchema, JsonTextWriter,
};
pub(crate) use policy::{
    is_limit_error, is_memory_budget_error, retry_after_memory_reclaim, JsonInputKind,
    JsonLimitReason, JsonProcessingPolicyV1, JsonSchema,
};
pub(crate) use work::{
    btree_workspace, copy_precharged_string, reserve_heap_slot, try_shrink_heap, CacheReservation,
    CatalogReservation, JsonOperationContext, JsonReservation, ReservedString, ReservedVec,
    WorkLedger, WorkspaceReservation,
};

/// JSON object members indexed by key in deterministic sorted order.
pub type JsonObject = BTreeMap<String, JsonValue>;

// The configured depth limit applies at bounded JSON entry points. Values built
// directly with the public enum can exceed it, so basic ownership operations must
// handle trees a parser would reject.
pub(crate) const MAX_JSON_DEPTH: usize = JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH;

/// An owned JSON value for metadata, filters and structured results. JSON numbers use binary64
/// precision.
#[derive(Default)]
pub enum JsonValue {
    /// The JSON literal `null`, also the default value.
    #[default]
    Null,
    /// A JSON boolean.
    Bool(
        /// Stored truth value.
        bool,
    ),
    /// A binary64 number. Serialization requires a finite value.
    Number(
        /// Numeric value before JSON encoding.
        f64,
    ),
    /// An owned Unicode string.
    String(
        /// Decoded text without JSON quotation marks or escapes.
        String,
    ),
    /// An ordered sequence of JSON values.
    Array(
        /// Elements in their original order.
        Vec<JsonValue>,
    ),
    /// An object mapping unique string keys to JSON values.
    Object(
        /// Member values indexed by their string keys.
        JsonObject,
    ),
}

impl std::fmt::Debug for JsonValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Debug formatting of containers delegates back to their children's Debug
        // implementations. Validate depth and finite numbers first so a hand-built
        // invalid tree can still be printed as a short diagnostic.
        if self.validate_tree(Some(MAX_JSON_DEPTH), 0).is_err() {
            return formatter.write_str("JsonValue(<invalid or over-depth>)");
        }
        match self {
            Self::Null => formatter.write_str("Null"),
            Self::Bool(value) => formatter.debug_tuple("Bool").field(value).finish(),
            Self::Number(value) => formatter.debug_tuple("Number").field(value).finish(),
            Self::String(value) => formatter.debug_tuple("String").field(value).finish(),
            // Only validated containers reach the recursive standard Debug
            // formatters. A validation failure above deliberately emits one
            // placeholder for the entire value being formatted.
            Self::Array(value) => formatter.debug_tuple("Array").field(value).finish(),
            Self::Object(value) => formatter.debug_tuple("Object").field(value).finish(),
        }
    }
}

impl Drop for JsonValue {
    fn drop(&mut self) {
        // Scalar and empty-container destruction needs no traversal. This fast
        // return also stops the shallow drops performed as the loop replaces a
        // fully drained current value.
        if !matches!(self, Self::Array(values) if !values.is_empty())
            && !matches!(self, Self::Object(values) if !values.is_empty())
        {
            return;
        }
        // Replace the caller's root with null before detaching its children. The
        // normal field destructor will see only that sentinel after this manual
        // walk finishes.
        let mut value = std::mem::take(self);
        // Null is the end marker for the chain of suspended parents. Every actual
        // parent put on that chain is an array or object with a last slot
        // available for the return link.
        let mut pending = Self::Null;
        loop {
            // Each suspended container holds the previous frame in its last
            // child slot. Remove that link on return, so every child is visited once.
            // Keep the container's last occupied slot until its detached child is
            // finished. The slot carries the link to the previous suspended
            // container and avoids allocating a separate descent stack during
            // Drop.
            let child = match &mut value {
                Self::Array(values) => values
                    .last_mut()
                    .map(|child| std::mem::replace(child, std::mem::take(&mut pending))),
                // Retain the B-tree key while descending into its value. Removing
                // the same last entry on return frees that key and recovers the
                // link stored in its value slot.
                Self::Object(values) => values.last_entry().map(|mut entry| {
                    std::mem::replace(entry.get_mut(), std::mem::take(&mut pending))
                }),
                _ => None,
            };
            // The child becomes the new current node and the container becomes its
            // pending parent. The previous pending parent is already inside the
            // overwritten slot, so no sibling or ancestor has been lost.
            if let Some(child) = child {
                pending = std::mem::replace(&mut value, child);
            // A null pending link means there is no suspended parent. A genuine
            // JSON null in a child position does not end traversal while a parent
            // still needs to pop that slot.
            } else if pending.is_null() {
                break;
            } else {
                // Replacing a drained child drops only a scalar or an empty
                // container. Keeping that invariant prevents the ordinary
                // destructor invoked by assignment from walking a remaining
                // subtree recursively.
                value = std::mem::take(&mut pending);
                // Remove the reused child slot before resuming its parent. Popping
                // a real sibling instead would either revisit a drained subtree or
                // skip a value's destruction.
                pending = match &mut value {
                    Self::Array(values) => values.pop().unwrap(),
                    Self::Object(values) => values.pop_last().unwrap().1,
                    _ => unreachable!(),
                };
            }
        }
    }
}

impl Clone for JsonValue {
    fn clone(&self) -> Self {
        // Cloning a scalar or empty container needs neither a task stack nor an
        // intermediate value array. Container capacity is not part of JSON value
        // semantics.
        match self {
            Self::Null => return Self::Null,
            Self::Bool(value) => return Self::Bool(*value),
            Self::Number(value) => return Self::Number(*value),
            Self::String(value) => return Self::String(value.clone()),
            Self::Array(values) if values.is_empty() => return Self::Array(Vec::new()),
            Self::Object(values) if values.is_empty() => return Self::object(),
            _ => {}
        }
        // Push children in reverse order. When the container task runs, the
        // completed clones form an ordered suffix that pairs with the original keys.
        // Value tasks visit borrowed input. Container tasks run after all of their
        // children and consume the corresponding suffix of the completed-value
        // stack.
        enum Task<'a> {
            Value(&'a JsonValue),
            Array(usize),
            Object(&'a JsonObject),
        }
        // The ordinary Clone trait has no operation context or error return. Its
        // explicit heap stacks remove the nesting dependency on the native call
        // stack but still allocate for pending children and the copy.
        let mut tasks = vec![Task::Value(self)];
        let mut values = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                Task::Value(value) => match value {
                    JsonValue::Null => values.push(JsonValue::Null),
                    JsonValue::Bool(value) => values.push(JsonValue::Bool(*value)),
                    JsonValue::Number(value) => values.push(JsonValue::Number(*value)),
                    JsonValue::String(value) => values.push(JsonValue::String(value.clone())),
                    JsonValue::Array(items) => {
                        tasks.push(Task::Array(items.len()));
                        // LIFO reverses the order of enqueued tasks. Push children
                        // from last to first so the resulting suffix can be moved
                        // into the destination array without another reversal.
                        tasks.extend(items.iter().rev().map(Task::Value));
                    }
                    JsonValue::Object(items) => {
                        // Keep a reference to the original map until its values
                        // have been cloned. Its ordered key iterator can then
                        // restore each key/value association without queuing
                        // separate owned keys.
                        tasks.push(Task::Object(items));
                        tasks.extend(items.values().rev().map(Task::Value));
                    }
                },
                Task::Array(len) => {
                    // Extract only this container's completed children. Earlier
                    // values on the stack belong to siblings of a parent still
                    // waiting for its own completion task.
                    let items = values.split_off(values.len() - len);
                    values.push(JsonValue::Array(items));
                }
                Task::Object(object) => {
                    // Object values finish in B-tree key order because the task
                    // scheduler pushed them in reverse. Zip with the keys from the
                    // same borrowed map before appending this object to its
                    // parent's value stack.
                    let items = values.split_off(values.len() - object.len());
                    values.push(JsonValue::Object(
                        object.keys().cloned().zip(items).collect(),
                    ));
                }
            }
        }
        // Exactly one completed value remains once the last task runs. The output
        // no longer borrows any part of the source, including strings originally
        // used as object keys.
        values.pop().unwrap()
    }
}

impl PartialEq for JsonValue {
    fn eq(&self, other: &Self) -> bool {
        // Compare the common scalar cases without allocating a work list. Numbers
        // use their native binary64 equality, so comparison does not validate or
        // canonicalize them.
        match (self, other) {
            (Self::Null, Self::Null) => return true,
            (Self::Bool(left), Self::Bool(right)) => return left == right,
            (Self::Number(left), Self::Number(right)) => return left == right,
            (Self::String(left), Self::String(right)) => return left == right,
            (Self::Array(_), Self::Array(_)) | (Self::Object(_), Self::Object(_)) => {}
            _ => return false,
        }
        // Start the work list only for a matching pair of container roots. Its
        // child pairs can contain any two values, including mismatched types.
        let mut pending = vec![(self, other)];
        while let Some((left, right)) = pending.pop() {
            match (left, right) {
                (JsonValue::Null, JsonValue::Null) => {}
                (JsonValue::Bool(a), JsonValue::Bool(b)) if a == b => {}
                (JsonValue::Number(a), JsonValue::Number(b)) if a == b => {}
                (JsonValue::String(a), JsonValue::String(b)) if a == b => {}
                // A container mismatch can fail before any more pairs are put on
                // the heap stack. Siblings may then be compared in reverse
                // encounter order because equality produces only a boolean, not a
                // diagnostic path.
                (JsonValue::Array(a), JsonValue::Array(b)) if a.len() == b.len() => {
                    // Check length before zipping the ordered arrays. Otherwise
                    // equal prefixes of different lengths could pass because zip
                    // stops at the shorter input.
                    pending.extend(a.iter().zip(b));
                }
                (JsonValue::Object(a), JsonValue::Object(b)) if a.len() == b.len() => {
                    // B-tree traversal yields both key sequences in sorted order.
                    // Comparing paired keys makes insertion order irrelevant while
                    // rejecting unequal member sets before comparing their values.
                    for ((key_a, value_a), (key_b, value_b)) in a.iter().zip(b) {
                        if key_a != key_b {
                            return false;
                        }
                        pending.push((value_a, value_b));
                    }
                }
                _ => return false,
            }
        }
        true
    }
}

impl JsonValue {
    /// Create an empty JSON object, suitable for empty metadata or filters.
    pub fn object() -> Self {
        JsonValue::Object(BTreeMap::new())
    }

    /// Borrow the object members, or return `None` for any other kind.
    pub fn as_object(&self) -> Option<&JsonObject> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Borrow the object members for mutation, or return `None` for any other kind.
    pub fn as_object_mut(&mut self) -> Option<&mut JsonObject> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Borrow the array elements in order, or return `None` for any other kind.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(values) => Some(values),
            _ => None,
        }
    }

    /// Borrow the string contents, or return `None` for any other kind.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(value) => Some(value),
            _ => None,
        }
    }

    /// Read a JSON number as an f64, or return `None` for any other kind.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Read a JSON boolean, or return `None` for any other kind.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Look up a field on an object. Returns `None` for missing keys and non-objects.
    // This helper collapses an absent key and a non-object into one missing
    // result. A key explicitly containing JSON null still yields a borrowed value.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|obj| obj.get(key))
    }

    /// Test whether this value is the JSON literal `null`.
    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    /// Encode as JSON with no optional whitespace.
    /// Use [Self::try_to_compact_string] for fallible, bounded encoding.
    ///
    /// # Panics
    ///
    /// Panics if any number in the value is non-finite.
    pub fn to_compact_string(&self) -> String {
        // This convenience encoder accepts hand-built finite trees without
        // applying a unit's business limits. The fallible encoder below validates
        // and budgets the tree before producing text.
        self.validate_finite_numbers()
            .expect("cannot serialize non-finite JSON number");
        self.compact_string_unchecked()
    }

    /// Encode as compact JSON subject to the JSON record limits.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid numbers or if the value exceeds the nesting,
    /// encoded-size or work budget limits.
    pub fn try_to_compact_string(&self) -> Result<String> {
        self.try_to_compact_string_with_kind(JsonInputKind::JsonlRecord)
    }

    pub(crate) fn try_to_compact_string_with_kind(&self, kind: JsonInputKind) -> Result<String> {
        // This public-style formatter creates a standalone operation for the
        // duration of encoding. Callers already holding source credit use the
        // retained writer paths to avoid charging that source to a second owner.
        let context = JsonOperationContext::standalone();
        // Validation computes the exact compact byte length including escaping and
        // punctuation while holding the source tree's admission. Use that size to
        // budget output during its overlap with encoder scratch.
        let (mut reservation, bytes) = self.guard_in(kind, &context)?;
        let mut frames = Vec::new();
        // Reserve frames for nested containers plus the initial value task. This
        // allocation coexists with the source and output string in the same
        // standalone operation.
        reservation.reserve_vec(&mut frames, MAX_JSON_DEPTH + 1, kind)?;
        let mut out = output::JsonTextWriter::new(&context, kind);
        // Reserve for canonical UTF-8 bytes, not the number of input characters.
        // Escaped keys, control characters, separators and numeric spellings are
        // already reflected in the validation result.
        out.reserve(bytes)?;
        let written = write_json_with_frames(
            self,
            &mut out,
            0,
            false,
            &mut frames,
            JsonOutputSchema::Value(kind),
        );
        // The text writer retains any detailed error hidden by fmt::Write. Finish
        // it before handing the owned String back to this convenience caller.
        out.finish(written).map(|(text, _)| text)
    }

    #[cfg(test)]
    fn canonical_digest(&self, kind: JsonInputKind) -> Result<(usize, String)> {
        let context = JsonOperationContext::standalone();
        let (_input, _) = self.guard_in(kind, &context)?;
        self.canonical_digest_validated_in(kind, &context)
            .map(|(bytes, hash, _)| (bytes, hash))
    }

    // The caller has already checked the input's finite numbers, depth and schema
    // limits. This path computes identity for that same value without making a
    // second owned representation.
    pub(crate) fn canonical_digest_validated_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<(usize, String, JsonReservation)> {
        struct Digest {
            // Digest of the emitted canonical UTF-8 bytes. Hashing the writer's
            // output keeps identity independent of the source formatting.
            hash: crate::hash::Sha256,
            // Length in bytes of exactly the same canonical representation fed to
            // the digest.
            bytes: usize,
        }
        impl std::fmt::Write for Digest {
            fn write_str(&mut self, value: &str) -> std::fmt::Result {
                // Count exactly the UTF-8 pieces fed to SHA-256. Hashing the
                // encoded text through this sink avoids building a second
                // canonical JSON string just to measure or digest it.
                self.hash.update(value.as_bytes());
                self.bytes += value.len();
                Ok(())
            }
        }
        impl JsonWrite for Digest {}
        let mut reservation = context.reservation();
        // Admit the digest state and its eventual 64 hexadecimal characters before
        // encoding. The traversal frame array has its own temporary portion of
        // this same reservation.
        reservation.grow(std::mem::size_of::<Digest>() + 64, kind)?;
        let mut frames = Vec::new();
        reservation.reserve_vec(&mut frames, MAX_JSON_DEPTH + 1, kind)?;
        let mut digest = Digest {
            hash: crate::hash::Sha256::default(),
            bytes: 0,
        };
        // Use the compact writer and the input's validated schema so digest
        // identity follows the same string escaping, numeric spelling and sorted
        // object keys as canonical output.
        write_json_with_frames(
            self,
            &mut digest,
            0,
            false,
            &mut frames,
            JsonOutputSchema::Value(kind),
        )
        .unwrap();
        // The returned byte count measures the canonical input, not the printed
        // digest. Take the traversal allocation's capacity before dropping it so
        // the correct grant can be refunded.
        let frame_bytes = frames.capacity() * std::mem::size_of::<WriteFrame<'_>>();
        drop(frames);
        let hash = crate::hash::digest_hex(digest.hash.finish());
        // The frame vector and digest state have finished their work. Return only
        // the hash-string credit with the digest and the canonical byte count.
        reservation.release(frame_bytes + std::mem::size_of::<Digest>());
        Ok((digest.bytes, hash, reservation))
    }

    fn compact_string_unchecked(&self) -> String {
        let mut out = String::new();
        write_json(self, &mut out, 0, false);
        out
    }

    /// Encode as indented, human-readable JSON.
    ///
    /// # Panics
    ///
    /// Panics if any number in the value is non-finite.
    pub fn to_pretty_string(&self) -> String {
        // Indented output can be longer than the canonical representation. This
        // convenience API formats the supplied tree after its finite-number check
        // and does not use its presentation length for a business-unit limit.
        self.validate_finite_numbers()
            .expect("cannot serialize non-finite JSON number");
        let mut out = String::new();
        write_json(self, &mut out, 0, true);
        out
    }

    /// Check all nested numbers and reject NaN or infinities before serialization.
    pub fn validate_finite_numbers(&self) -> Result<()> {
        // A finite-number check is meaningful for manually constructed trees of
        // any depth. Omitting the business schema here avoids imposing
        // encoded-byte or operation-budget limits on this predicate.
        self.validate_tree(None, 0)
    }

    pub(crate) fn validate_kind(&self, kind: JsonInputKind) -> Result<()> {
        // This entry point asks only whether the value passes. Using a standalone
        // operation applies the same checks and frees the temporary ownership as
        // soon as the answer is known.
        self.guard_in(kind, &JsonOperationContext::standalone())
            .map(|_| ())
    }

    pub(crate) fn guard_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<(JsonReservation, usize)> {
        let mut reservation = context.reservation();
        // Supplying both a schema and an owner requests the complete input check:
        // validate depth and canonical size, measure retained capacities and cover
        // those capacities in the supplied operation.
        let layout = self.validate_tree_layout(
            Some(MAX_JSON_DEPTH),
            0,
            Some(JsonOutputSchema::Value(kind)),
            Some(&mut reservation),
            true,
        )?;
        Ok((reservation, layout.canonical_bytes))
    }

    pub(crate) fn canonical_bytes_retained_in(
        &self,
        kind: JsonInputKind,
        retained: &mut JsonReservation,
    ) -> Result<usize> {
        // The caller already owns credit for a retained representation. Reconcile
        // that same owner while measuring its current encoding so existing storage
        // does not need an independent second reservation.
        self.validate_tree_layout(
            Some(MAX_JSON_DEPTH),
            0,
            Some(JsonOutputSchema::Value(kind)),
            Some(retained),
            true,
        )
        .map(|layout| layout.canonical_bytes)
    }

    // Structural checks use a schema-free walk. The same iterator machinery can
    // check finite numbers and optional enclosing depth without calculating a
    // canonical byte count or adopting any retained allocations.
    fn validate_tree(&self, depth_limit: Option<usize>, enclosing_depth: usize) -> Result<()> {
        self.validate_tree_layout(depth_limit, enclosing_depth, None, None, true)
            .map(|_| ())
    }

    // The source was already validated with its schema; do not turn an envelope into a value root.
    pub(crate) fn reserve_retained_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<JsonReservation> {
        let mut work = context.reservation();
        self.reconcile_retained_in(kind, &mut work)?;
        Ok(work)
    }

    // Reuse the reservation when owned JSON moves between representations. Scratch
    // is charged separately so traversal cannot spend credit backing live nodes.
    pub(crate) fn reconcile_retained_in(
        &self,
        kind: JsonInputKind,
        work: &mut JsonReservation,
    ) -> Result<()> {
        // The input's owning representation changed but its JSON was previously
        // validated. Recompute retained capacities without rerunning root-specific
        // wire-size or nesting limits on an extracted subtree.
        self.validate_tree_layout(
            None,
            0,
            Some(JsonOutputSchema::Value(kind)),
            Some(work),
            false,
        )?;
        Ok(())
    }

    fn validate_tree_layout(
        &self,
        depth_limit: Option<usize>,
        enclosing_depth: usize,
        schema: Option<JsonOutputSchema>,
        mut reservation: Option<&mut JsonReservation>,
        measure_canonical: bool,
    ) -> Result<JsonLayout> {
        // The traversal scratch and the resulting retained owner use the same
        // operation ledger, but have distinct lifetimes. A frame allocation must
        // not be left in the returned long-lived credit.
        let scratch = reservation.as_ref().map(|work| work.scratch());
        self.validate_tree_layout_with_scratch(
            depth_limit,
            enclosing_depth,
            schema,
            reservation.take(),
            scratch,
            measure_canonical,
        )
    }

    pub(crate) fn retained_bytes_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<usize> {
        // A sizing-only caller pays for the iterators needed to inspect the tree
        // but does not acquire the measured tree's credit here. It can use this
        // result to plan a later transfer.
        self.validate_tree_layout_with_scratch(
            None,
            0,
            Some(JsonOutputSchema::Value(kind)),
            None,
            Some(context.reservation()),
            false,
        )
        .map(|layout| layout.retained_bytes)
    }

    // The caller retains source ownership; only validation scratch is new work.
    pub(crate) fn encoding_shape_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<(usize, usize)> {
        // The encoder needs the maximum stack size and current retained layout.
        // Check canonical size and schema as well before allowing serialization to
        // begin, even though this helper returns no encoded bytes.
        self.validate_tree_layout_with_scratch(
            Some(MAX_JSON_DEPTH),
            0,
            Some(JsonOutputSchema::Value(kind)),
            None,
            Some(context.reservation()),
            true,
        )
        // Encoding needs a slot for the initial value in addition to one per open
        // container. A scalar has depth zero and still requires that initial
        // writer slot.
        .map(|layout| (layout.depth + 1, layout.retained_bytes))
    }

    fn validate_tree_layout_with_scratch(
        &self,
        depth_limit: Option<usize>,
        enclosing_depth: usize,
        schema: Option<JsonOutputSchema>,
        mut reservation: Option<&mut JsonReservation>,
        mut scratch: Option<JsonReservation>,
        measure_canonical: bool,
    ) -> Result<JsonLayout> {
        enum Children<'a> {
            /// An ordered sequence of JSON values.
            Array(std::slice::Iter<'a, JsonValue>),
            /// An object mapping unique string keys to JSON values.
            Object(std::collections::btree_map::Iter<'a, String, JsonValue>),
        }
        impl<'a> Iterator for Children<'a> {
            type Item = (Option<&'a str>, &'a JsonValue);
            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    // An array item supplies no field name. An object member
                    // carries its key into the next schema step so nested filter
                    // or metadata errors can retain the right classification.
                    Self::Array(values) => values.next().map(|value| (None, value)),
                    Self::Object(values) => values
                        .next()
                        .map(|(key, value)| (Some(key.as_str()), value)),
                }
            }
        }
        // Walk one open container per stack entry. Depth, canonical byte length and
        // retained allocation size answer different questions: an empty Vec with
        // spare capacity can be cheap on the wire and expensive to keep in memory.
        // Some callers need only structural or numeric validation. Measuring
        // allocation capacities is enabled by a scratch owner even if no retained
        // tree reservation is acquired by this walk.
        let measure_allocations = scratch.is_some();
        let mut stack = Vec::<(Children<'_>, Option<JsonOutputSchema>)>::new();
        let mut retained_bytes = 0usize;
        let mut next = Some((self, schema));
        let mut canonical_bytes = 0usize;
        let mut max_depth = 0;
        while let Some((value, schema)) = next {
            // The active output schema can vary by location. Fetch its input kind
            // for the current node before reporting a depth or unit-size error,
            // even when its parent used another kind.
            let kind = schema.map(JsonOutputSchema::kind);
            let children = match value {
                // Finiteness is checked for every number, including those in
                // schema-free validation. JSON has no representation for
                // infinities or NaN.
                JsonValue::Number(value) if !value.is_finite() => {
                    return Err(json_error("JSON numbers must be finite"));
                }
                JsonValue::Array(values) => Some(Children::Array(values.iter())),
                JsonValue::Object(values) => Some(Children::Object(values.iter())),
                _ => None,
            };
            if let Some(children) = children {
                // The next value is itself a container. Include the depth it will
                // occupy inside its enclosing output object when deciding whether
                // it fits. An empty array still opens one level.
                if depth_limit
                    .is_some_and(|limit| stack.len().saturating_add(enclosing_depth) >= limit)
                {
                    // When there is no business schema, a depth error needs a
                    // default JSON error category. The observed depth includes the
                    // new container and any enclosing levels supplied by the
                    // caller.
                    return Err(kind.unwrap_or(JsonInputKind::JsonlRecord).limit_error(
                        JsonLimitReason::Depth,
                        MAX_JSON_DEPTH,
                        Some(
                            stack
                                .len()
                                .saturating_add(enclosing_depth)
                                .saturating_add(1),
                        ),
                    ));
                }
                // The traversal stack and the retained value compete for the same
                // budget. Before failing to descend, return any unused reservation
                // above the bytes already covered. Dropping credit for live children
                // would undercount the tree.
                if let Some(work) = &mut scratch {
                    if let Err(error) = work.reserve_vec(&mut stack, 1, kind.unwrap()) {
                        // Admission of one more iterator can fail while an earlier
                        // tree reservation holds unused block headroom. Only the
                        // part beyond covered_bytes is safe to return before
                        // retrying the descent.
                        let can_trim = reservation.as_ref().is_some_and(|retained| {
                            retained.reserved_bytes() > retained.covered_bytes()
                        });
                        if !is_limit_error(&error) || !can_trim {
                            // Return the first admission error unless both
                            // conditions for reclaim hold. A successful trim gives
                            // the same pending descent one retry without consuming
                            // another child from its iterator.
                            return Err(error);
                        }
                        let retained = reservation.as_mut().unwrap();
                        retained.retain(retained.covered_bytes());
                        work.reserve_vec(&mut stack, 1, kind.unwrap())?;
                    }
                }
                // Push a cursor even for an empty container. Its opening level
                // contributes to the reported maximum depth before the next
                // iteration unwinds the empty cursor.
                stack.push((children, schema));
                // Depth used for writer sizing is local to the traversal stack.
                // The enclosing depth affects limit enforcement above, but those
                // already-open outer containers are not part of the value's own
                // stack size.
                max_depth = max_depth.max(stack.len());
            }
            // Carry the schema kind with each iterator frame. The enclosing JSON
            // unit can change at an object field, so attributing every error to the
            // root would report the wrong limit to the caller.
            if let Some(kind) = kind {
                if measure_canonical {
                    // For containers, the shallow length covers their delimiters,
                    // separators and object keys. Child values are visited
                    // separately, so their encoded bytes are counted exactly once.
                    canonical_bytes =
                        canonical_bytes.saturating_add(shallow_canonical_bytes(value));
                }
                // Check the running canonical size after each node. A saturating
                // byte count remains over the finite policy limit even when very
                // large subtrees would overflow ordinary size arithmetic.
                if measure_canonical
                    && canonical_bytes > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES
                {
                    return Err(kind.limit_error(
                        JsonLimitReason::UnitBytes,
                        JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                        Some(canonical_bytes),
                    ));
                }
                // Charge retained storage by allocation capacity. A String with
                // spare bytes and a mostly empty Vec both occupy more memory than
                // their canonical encodings. Object keys own separate String
                // buffers in addition to the tree nodes.
                if measure_allocations {
                    let bytes = match value {
                        JsonValue::String(value) => value.capacity(),
                        JsonValue::Array(values) => values
                            .capacity()
                            .saturating_mul(std::mem::size_of::<JsonValue>()),
                        JsonValue::Object(values) => values
                            .keys()
                            // For objects, node storage and owned key buffers
                            // belong to the map. The values in those slots are
                            // visited on their own turns, so only their separately
                            // retained allocations will be added later.
                            .fold(work::object_workspace(values.len()), |bytes, key| {
                                bytes.saturating_add(key.capacity())
                            }),
                        _ => 0,
                    };
                    // Add the child's retained capacity to the running total
                    // before asking the owner to cover it. Repeated calls cover
                    // cumulative bytes, rather than admitting the full prefix
                    // again for each node.
                    retained_bytes = retained_bytes.saturating_add(bytes);
                    if let Some(work) = &mut reservation {
                        work.cover(retained_bytes, kind)?;
                    }
                }
            }
            // Carry the output schema through wrappers and arrays so a failure
            // within a filter expression is attributed to the filter. The depth and
            // canonical byte counters still span the tree being measured.
            next = loop {
                let Some((children, schema)) = stack.last_mut() else {
                    break None;
                };
                if let Some((key, value)) = children.next() {
                    // Use the parent schema's field transition for object members
                    // and its element transition for arrays. This permits an
                    // embedded value to have a different error category without
                    // resetting the tree-wide layout counters.
                    let schema = schema.map(|schema| match key {
                        Some(key) => schema.field(key),
                        None => schema.element(),
                    });
                    break Some((value, schema));
                }
                // An exhausted child iterator has no further schema transitions.
                // Pop it and continue at the next sibling of its parent without
                // revisiting that parent's local byte contribution.
                stack.pop();
            };
        }
        // Once traversal is complete, retain the measured byte count and refund
        // any unused admission headroom. The scratch stack and its separate charge
        // expire on return.
        if let Some(work) = &mut reservation {
            work.retain(retained_bytes);
        }
        Ok(JsonLayout {
            canonical_bytes,
            depth: max_depth,
            retained_bytes,
        })
    }
}

struct JsonLayout {
    // Estimated retained allocation bytes for the in-memory tree, including
    // capacities that do not appear in serialized output.
    retained_bytes: usize,
    // Byte length of the canonical JSON representation. This is the size compared
    // with a business unit's wire limit.
    canonical_bytes: usize,
    // Maximum traversed container depth measured for this value. Serialization can
    // use it to plan a bounded traversal stack.
    depth: usize,
}

// Validate normalized typed objects without cloning their untrusted strings.
pub(crate) struct JsonObjectGuard {
    // Reservation for the typed object's nodes, keys and any field storage charged
    // through this guard.
    work: JsonReservation,
    // Business input kind used to classify validation and admission failures.
    kind: JsonInputKind,
    // Fields accounted for so far. The first field needs no comma and also changes
    // the shape of the object-node reservation.
    fields: usize,
    // Canonical length accumulated for the object, including its outer braces and
    // the syntax between keys and values.
    bytes: usize,
}

impl JsonObjectGuard {
    pub(crate) fn new(kind: JsonInputKind, context: &JsonOperationContext) -> Self {
        Self {
            work: context.reservation(),
            kind,
            fields: 0,
            // Even an empty object emits two braces. Starting with their byte cost
            // lets each subsequent field account for just its key, colon, value
            // and any separator from the preceding field.
            bytes: 2,
        }
    }

    pub(crate) fn string_field(&mut self, key: &str, value: &String) -> Result<()> {
        // The serialized string length includes quotes and escaping. Its owned
        // buffer is a separate memory charge based on actual capacity, so the two
        // limits can reject different inputs.
        self.field(key, string_canonical_bytes(value))?;
        self.work.grow(value.capacity(), self.kind)
    }

    // The enclosing typed object retains the same strings' admitted JSON credit.
    pub(crate) fn string_field_precharged(&mut self, key: &str, value: &str) -> Result<()> {
        // Skip charging the value buffer only when the enclosing input carries it
        // already. The key, separators and escaped value bytes still affect the
        // typed object's admission and size limit.
        self.field(key, string_canonical_bytes(value))
    }

    pub(crate) fn value_field(&mut self, key: &str, value: &JsonValue) -> Result<()> {
        // Validate this subtree as a child at depth one, using the enclosing
        // object's input kind. A tree valid as a root can already be too deep once
        // embedded under this field.
        let mut work = self.work.scratch();
        let layout = value.validate_tree_layout(
            Some(MAX_JSON_DEPTH),
            1,
            Some(JsonOutputSchema::Value(self.kind)),
            Some(&mut work),
            true,
        )?;
        // Keep subtree storage provisional until the enclosing field's canonical
        // size and node growth pass. An enclosing-unit failure then drops that
        // subtree's scratch reservation.
        self.field(key, layout.canonical_bytes)?;
        self.work.merge(work);
        Ok(())
    }

    fn field(&mut self, key: &str, value_bytes: usize) -> Result<()> {
        // Compute size in canonical UTF-8 bytes before admitting node storage.
        // This is independent of the enclosing struct's Rust layout or the number
        // of bytes the caller used to spell the input.
        self.bytes = self
            .bytes
            .saturating_add(string_canonical_bytes(key))
            .saturating_add(1 + usize::from(self.fields != 0))
            .saturating_add(value_bytes);
        // Check size before growing the reservation. On failure callers discard
        // the partially built guard. Reusing its updated byte count for a retry
        // would count the failing field twice.
        if self.bytes > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES {
            return Err(self.kind.limit_error(
                JsonLimitReason::UnitBytes,
                JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                Some(self.bytes),
            ));
        }
        // Only node growth above the previous field count is new work. Existing
        // keys and nodes retain their earlier charges, while this field adds one
        // key's bytes.
        self.work.grow(
            work::object_workspace(self.fields + 1) - work::object_workspace(self.fields)
                + key.len(),
            self.kind,
        )?;
        // Advance field count only after the key and its node growth are admitted.
        // This count controls both the next tree estimate and whether another
        // field will need a comma.
        self.fields += 1;
        Ok(())
    }

    // The caller may keep using its typed object after these checks. Return the
    // accumulated reservation so its admission can live as long as the operation
    // needs those fields.
    pub(crate) fn finish(self) -> JsonReservation {
        self.work
    }
}

// Build the ordered object from owned values. Its B-tree keys determine iteration
// and canonical output order independently of the order in which the caller
// supplied the pairs.
pub fn object_from_pairs<I, K>(pairs: I) -> JsonValue
where
    I: IntoIterator<Item = (K, JsonValue)>,
    K: Into<String>,
{
    JsonValue::Object(
        pairs
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

/// Parse one JSON value from text under the JSON record limits.
///
/// # Errors
///
/// Returns an error for malformed input or when a JSON processing limit is exceeded.
/// Integer tokens must be representable exactly as binary64. Supply larger inexact
/// integers as JSON strings when the application needs all their decimal digits.
pub fn parse(input: &str) -> Result<JsonValue> {
    parse_with_schema(input, JsonSchema::Value(JsonInputKind::JsonlRecord))
}

#[cfg(test)]
pub(crate) fn parse_with_kind(input: &str, kind: JsonInputKind) -> Result<JsonValue> {
    parse_with_schema(input, JsonSchema::Value(kind))
}

pub(crate) fn parse_with_schema(input: &str, schema: JsonSchema) -> Result<JsonValue> {
    // Standalone parsing returns only the DOM. Operation-aware callers use the
    // variant below to keep the tree's admission with it while they process or
    // transfer its allocations.
    parse_with_schema_in(input, schema, &JsonOperationContext::standalone())
        .map(ProcessedJson::into_value)
}

pub(crate) struct ProcessedJson {
    // Parsed or copied JSON tree whose allocations are kept alive by the
    // accompanying reservation.
    pub(crate) value: JsonValue,
    // Charge for the returned JSON storage. Moving the value across an operation
    // boundary must also move or transfer this credit.
    _reservation: JsonReservation,
    // Fixed-size parsed query parameter; never reconstruct it from the floating DOM.
    search_top_k: Option<usize>,
}

impl ProcessedJson {
    fn into_value(self) -> JsonValue {
        self.value
    }

    pub(crate) fn into_parts(self) -> (JsonValue, JsonReservation) {
        // Transfer both pieces when a typed reader adopts a parsed tree. Taking
        // only the value here would drop its reservation while the original
        // allocations were still live in the operation.
        (self.value, self._reservation)
    }

    // Query consumers need this third component before converting the DOM to typed
    // options. It records the exact top_k proven from decimal input, which a
    // rounded f64 alone cannot establish.
    pub(crate) fn into_parts_with_search_top_k(
        self,
    ) -> (JsonValue, JsonReservation, Option<usize>) {
        (self.value, self._reservation, self.search_top_k)
    }
}

pub(crate) fn parse_with_schema_in(
    input: &str,
    schema: JsonSchema,
    context: &JsonOperationContext,
) -> Result<ProcessedJson> {
    // The input string already supplies a contiguous borrowed buffer. A cursor
    // satisfies the buffered parser interface without another read buffer
    // competing with the parsed tree.
    parse_buffered(Cursor::new(input.as_bytes()), schema, context, false)
}

pub(crate) fn parse_reader(source: impl Read, schema: JsonSchema) -> Result<JsonValue> {
    parse_reader_in(source, schema, &JsonOperationContext::standalone())
        .map(ProcessedJson::into_value)
}

pub(crate) fn parse_reader_in(
    source: impl Read,
    schema: JsonSchema,
    context: &JsonOperationContext,
) -> Result<ProcessedJson> {
    // Stream readers use a fixed-size input buffer independently of the eventual
    // DOM size. A bounded canonical field below can reduce that read buffer to its
    // checked byte frame.
    read_json(source, schema, context, false, 8192)
}

#[cfg(test)]
pub(crate) fn parse_canonical_reader(source: impl Read, kind: JsonInputKind) -> Result<JsonValue> {
    parse_canonical_reader_in(
        &mut source.take(u64::MAX),
        kind,
        &JsonOperationContext::standalone(),
    )
    .map(ProcessedJson::into_value)
}

pub(crate) fn parse_canonical_reader_in(
    source: &mut std::io::Take<impl Read>,
    kind: JsonInputKind,
    context: &JsonOperationContext,
) -> Result<ProcessedJson> {
    // Stored fields already have a checked byte frame. Do not allocate an 8-KiB
    // buffer for every small metadata object in a primary-record scan.
    let buffer_bytes = source.limit().min(8192) as usize;
    read_json(source, JsonSchema::Value(kind), context, true, buffer_bytes)
}

fn read_json(
    source: impl Read,
    schema: JsonSchema,
    context: &JsonOperationContext,
    canonical: bool,
    buffer_bytes: usize,
) -> Result<ProcessedJson> {
    // Admit the reader buffer for the duration of parsing. Its bytes leave scope
    // when this function returns, while the parsed tree carries a distinct
    // reservation to the caller.
    let mut input_work = context.reservation();
    input_work.grow(buffer_bytes, schema.kind())?;
    let source = BufReader::with_capacity(buffer_bytes, source);
    parse_buffered(source, schema, context, canonical)
}

fn parse_buffered(
    source: impl BufRead,
    schema: JsonSchema,
    context: &JsonOperationContext,
    canonical: bool,
) -> Result<ProcessedJson> {
    let mut parser = Parser {
        source,
        canonical,
        pos: 0,
        unit: None,
        kind: schema.kind(),
        work: context.aggregate_reservation(),
        search_top_k: None,
    };
    // Parse one complete root and then inspect the remainder. Acceptable trailing
    // whitespace in ordinary input does not make a second adjacent JSON value part
    // of this call.
    let value = parser.parse_value(0, schema)?;
    // skip_ws enforces the active mode. Ordinary JSON accepts trailing whitespace,
    // while a canonical stored frame rejects it even if the whitespace is the only
    // data after a valid root.
    parser.skip_ws()?;
    if parser.peek()?.is_some() {
        return Err(json_error(format!(
            "unexpected trailing data at byte {}",
            parser.pos
        )));
    }
    // Finish aggregate accounting only after the end-of-input check. Return the
    // parser's retained coverage with the successful value, including any exact
    // query-parameter result.
    parser.work.finish_aggregate();
    Ok(ProcessedJson {
        value,
        _reservation: parser.work,
        search_top_k: parser.search_top_k,
    })
}

pub fn required_string(obj: &JsonObject, key: &str) -> Result<String> {
    // This borrowed-object helper returns an independently owned string. Callers
    // taking ownership of a decoded row can use take_required_string to reuse its
    // existing field allocation.
    obj.get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                format!("missing string field {key}"),
            )
        })
}

pub(crate) fn take_required_string(obj: &mut JsonObject, key: &str) -> Result<String> {
    // Move the String buffer out while leaving an empty string at the same object
    // key. An outer owner may safely drop the rest of the parsed object after
    // adopting the returned storage.
    match obj.get_mut(key) {
        Some(JsonValue::String(value)) => Ok(std::mem::take(value)),
        _ => Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing string field {key}"),
        )),
    }
}

// This convenience accessor collapses missing, null and other non-string values to
// None. Schema readers that must reject a present wrong type need to inspect that
// field explicitly.
pub fn optional_string(obj: &JsonObject, key: &str) -> Option<String> {
    obj.get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
}

pub fn required_usize(obj: &JsonObject, key: &str) -> Result<usize> {
    let value = obj.get(key).and_then(JsonValue::as_f64).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing numeric field {key}"),
        )
    })?;
    // The upper bound is exclusive. On a 64-bit host, converting usize::MAX itself
    // to f64 rounds up to the next power of two, so an inclusive float comparison
    // would accept that out-of-range value.
    let exclusive_max = (usize::MAX as u128 + 1) as f64;
    // Reject NaN, infinities, negative values and nonintegral numbers before
    // Rust's float-to-integer cast. Otherwise a saturating or truncating
    // conversion could turn invalid metadata into an apparently valid size.
    if !value.is_finite() || value < 0.0 || value >= exclusive_max || value.fract() != 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{key} must be a non-negative integer"),
        ));
    }
    Ok(value as usize)
}

// Borrow the array's slice from the source object. The caller can inspect large
// vector or option lists without a duplicate array allocation, but must retain the
// source while doing so.
pub fn required_array<'a>(obj: &'a JsonObject, key: &str) -> Result<&'a [JsonValue]> {
    obj.get(key).and_then(JsonValue::as_array).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing array field {key}"),
        )
    })
}

pub fn ensure_object(value: JsonValue) -> Result<JsonValue> {
    // Return the original object with its allocations untouched. This check
    // validates the outer variant only. Its caller determines which metadata or
    // configuration schema to apply.
    if matches!(value, JsonValue::Object(_)) {
        Ok(value)
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "metadata/config fields must be JSON objects",
        ))
    }
}

fn json_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidJsonlSchema, message)
}

#[cfg(test)]
pub(crate) fn is_depth_error(error: &Error) -> bool {
    error
        .detail()
        .iter()
        .any(|(key, value)| key == "json_limit_reason" && value == "depth_exceeded")
}

fn string_canonical_bytes(value: &str) -> usize {
    // Start with the quotes and account for each character in its emitted form. A
    // short escape and a six-byte control escape occupy different space even for a
    // one-byte source character.
    value.chars().fold(2usize, |len, ch| {
        len.saturating_add(char_canonical_bytes(ch))
    })
}

fn char_canonical_bytes(ch: char) -> usize {
    match ch {
        // Match the writer's short escapes explicitly. Other control characters
        // use a backslash-u escape, while ordinary Unicode characters retain their
        // raw UTF-8 byte width.
        '"' | '\\' | '\n' | '\r' | '\t' | '\u{08}' | '\u{0c}' => 2,
        ch if ch.is_control() => 6,
        ch => ch.len_utf8(),
    }
}

fn shallow_canonical_bytes(value: &JsonValue) -> usize {
    // Containers contribute only their own syntax here. The iterative tree walk
    // adds each child value separately, so this helper must not recursively count
    // contained values a second time.
    match value {
        JsonValue::Null => 4,
        JsonValue::Bool(value) => {
            if *value {
                4
            } else {
                5
            }
        }
        JsonValue::Number(value) => number_canonical_bytes(*value),
        JsonValue::String(value) => string_canonical_bytes(value),
        // The empty array still costs two brackets. A nonempty array adds exactly
        // one comma per gap regardless of the sizes or shapes of its children.
        JsonValue::Array(values) => 2usize.saturating_add(values.len().saturating_sub(1)),
        // Object keys belong to their containing node's contribution. Include
        // their quoted, escaped spellings and colons here because they are not
        // separate child values in the traversal.
        JsonValue::Object(values) => values.keys().fold(
            2usize.saturating_add(values.len().saturating_sub(1)),
            |len, key| {
                len.saturating_add(string_canonical_bytes(key))
                    .saturating_add(1)
            },
        ),
    }
}

// This count follows the formatter's emitted bytes, including unusually long
// integral f64 spellings. An estimate based on the in-memory width of f64 could
// understate the JSON limit.
fn number_canonical_bytes(value: f64) -> usize {
    use std::fmt::Write;
    struct Counter(usize);
    impl Write for Counter {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            self.0 += value.len();
            Ok(())
        }
    }
    // Measure the same formatter used for real output by giving it a byte-counting
    // sink. Updating this one spelling policy therefore changes both serialization
    // and its limit check together.
    let mut counter = Counter(0);
    write_number(value, &mut counter).unwrap();
    counter.0
}

fn write_number(value: f64, out: &mut impl std::fmt::Write) -> std::fmt::Result {
    // Positive and negative zero compare equal and deliberately share the
    // canonical spelling 0. Other integral floats omit a decimal suffix, including
    // integers larger than the exact contiguous range.
    if value == 0.0 {
        out.write_char('0')
    } else if value.fract() == 0.0 {
        write!(out, "{value:.0}")
    } else {
        write!(out, "{value}")
    }
}

fn is_canonical_number(value: f64, raw: &str) -> bool {
    struct Comparison<'a>(&'a str);
    impl std::fmt::Write for Comparison<'_> {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            // Compare successive formatter fragments against the unconsumed input.
            // A mismatch ends comparison immediately without allocating a second
            // string the size of the number token.
            self.0 = self.0.strip_prefix(text).ok_or(std::fmt::Error)?;
            Ok(())
        }
    }
    let mut comparison = Comparison(raw);
    // The formatted prefix alone cannot establish canonical form. Require full
    // consumption so an otherwise matching integer followed by a redundant decimal
    // or exponent is rejected.
    write_number(value, &mut comparison).is_ok() && comparison.0.is_empty()
}

fn canonical_error() -> Error {
    Error::new(
        ErrorCode::FileFormatError,
        "stored JSON is not in canonical form",
    )
}

pub(crate) fn validate_query_vector(vector: &[f32]) -> Result<()> {
    // Start with the array punctuation, then charge the canonical decimal spelling
    // of each widened f32. The JSON byte ceiling can be crossed even when the raw
    // float buffer is small.
    let mut bytes = 2usize.saturating_add(vector.len().saturating_sub(1));
    for &value in vector {
        // Reject a non-finite typed vector entry before formatting it. This path
        // accepts native f32 input, which can contain NaN or infinities that the
        // JSON text parser would already have rejected.
        if !value.is_finite() {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "query embedding vector entries must be finite f32 values",
            ));
        }
        bytes = bytes.saturating_add(shallow_canonical_bytes(&JsonValue::Number(f64::from(
            value,
        ))));
        if bytes > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES {
            return Err(JsonInputKind::QueryEmbedding.limit_error(
                JsonLimitReason::UnitBytes,
                JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                Some(bytes),
            ));
        }
    }
    Ok(())
}

// JSON syntax is already validated. Work on decimal digits before binary64 conversion.
// Keep the decimal spelling for this integer check. For example, the value
// 1.00000000000000001 rounds to 1.0 in binary64 but is not an integral top_k.
// Account for the decimal point and exponent before accepting the range.
fn exact_search_top_k(raw: &str) -> Option<usize> {
    // The search limit is strictly positive. Testing the sign before arithmetic
    // also rules out signed zero regardless of the exponent used to spell it.
    if raw.starts_with('-') {
        return None;
    }
    let (mantissa, exponent) = match raw.find(['e', 'E']) {
        Some(index) => {
            // An exponent too large for i64 cannot be a usable scale for a nonzero
            // top_k. Preserve its direction with a saturated value. The later digit
            // and integer-range checks reject it without expanding a string of
            // zeroes.
            let raw_exponent = &raw[index + 1..];
            let exponent = raw_exponent.parse::<i64>().unwrap_or_else(|_| {
                if raw_exponent.starts_with('-') {
                    i64::MIN
                } else {
                    i64::MAX
                }
            });
            (&raw[..index], exponent)
        }
        None => (raw, 0),
    };
    // Keep the scale as a count of decimal places. The accumulator later treats
    // decimal and exponent spellings such as 12.00 and 1200e-2 as the same
    // candidate integer.
    let fractional_digits = mantissa
        .find('.')
        .map_or(0, |index| mantissa.len() - index - 1);
    let mut first = None;
    let mut last = 0usize;
    let mut count = 0usize;
    // Work in digit positions with the point removed. Leading zeroes do not
    // increase the magnitude, and trailing zeroes can cancel a negative power of
    // ten without a fractional result.
    for byte in mantissa.bytes().filter(|&byte| byte != b'.') {
        if byte != b'0' {
            first.get_or_insert(count);
            last = count;
        }
        count += 1;
    }
    // For example, 1200e-2 has two trailing zeroes available, giving the positive
    // integer 12. If the exponent would remove a nonzero digit too, the input has a
    // fractional part and cannot be a count.
    let first = first?; // Zero is outside the positive range, regardless of exponent.
    let zeros = exponent
        .saturating_sub(fractional_digits as i64)
        .saturating_add((count - last - 1) as i64);
    // A negative residual scale means some nonzero digit would fall after the
    // point. Positive top_k bounds alone cannot detect such a fractional spelling
    // once rounded to f64.
    let zeros = usize::try_from(zeros).ok()?; // A negative scale is a noninteger.
    let digits = last - first + 1;
    // At most sixteen decimal digits can fit the supported exact JSON search
    // range. This quick rejection prevents work proportional to a huge positive
    // exponent. The exact limit is checked below.
    if digits.saturating_add(zeros) > 16 {
        return None;
    }
    // Accumulate only the significant digits, then apply the residual power of ten
    // with checked arithmetic. Both the host usize and the exact JSON integer range
    // constrain the final count.
    let mut value = 0usize;
    for byte in mantissa
        .bytes()
        .filter(|&byte| byte != b'.')
        .skip(first)
        .take(digits)
    {
        // The decimal accumulator uses checked native-size operations. On a 32-bit
        // target it can reject a value that would satisfy the same decimal-length
        // test on a wider host.
        value = value
            .checked_mul(10)?
            .checked_add(usize::from(byte - b'0'))?;
    }
    for _ in 0..zeros {
        value = value.checked_mul(10)?;
    }
    // Even when checked multiplication fits this host, reject a count beyond the
    // query contract's safe JSON integer ceiling. Smaller hosts already enforce
    // their own width during accumulation.
    (value <= crate::query::MAX_SEARCH_TOP_K).then_some(value)
}

fn validate_integer_precision(raw: &str) -> Result<()> {
    // Integer precision is symmetric about zero. Strip the sign and test the
    // unsigned source magnitude before conversion to f64 can hide an inexact
    // integer.
    let unsigned = raw.strip_prefix('-').unwrap_or(raw);
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(index) => {
            let exponent = &unsigned[index + 1..];
            // The magnitude test never has to materialize an exponent-sized
            // string. If the exponent does not fit i64, its direction suffices to
            // classify the bounded nonzero mantissa here.
            let value = exponent.parse::<i64>().unwrap_or_else(|_| {
                if exponent.starts_with('-') {
                    i64::MIN
                } else {
                    i64::MAX
                }
            });
            (&unsigned[..index], value)
        }
        None => (unsigned, 0),
    };
    let fractional_digits = mantissa
        .find('.')
        .map(|index| mantissa.len() - index - 1)
        .unwrap_or(0);
    let mut first_nonzero = None;
    let mut last_nonzero = 0;
    let mut digit_count = 0usize;
    for byte in mantissa.bytes().filter(|byte| *byte != b'.') {
        if byte != b'0' {
            first_nonzero.get_or_insert(digit_count);
            last_nonzero = digit_count;
        }
        digit_count += 1;
    }
    // With no nonzero digit the value is exactly zero at every decimal scale. This
    // general precision check accepts zero even though the query-specific top_k
    // check excludes it.
    let Some(first_nonzero) = first_nonzero else {
        return Ok(());
    };
    // Work from decimal digits before rounding to binary64. Moving the decimal point
    // and cancelling trailing zeros tells us whether the source denotes an integer
    // at all. For example, an exponent can turn a fractional spelling into a large
    // exact integer.
    let zeros = exponent
        .saturating_sub(fractional_digits as i64)
        .saturating_add((digit_count - last_nonzero - 1) as i64);
    // After cancelling decimal zeroes, a negative scale means the original value
    // is fractional. Exact-integer representability no longer applies. Ordinary
    // f64 parsing and the finite check still follow.
    if zeros < 0 {
        return Ok(());
    }
    let precision_error =
        || json_error("integer cannot be represented exactly as binary64; encode it as a string");
    let zeros = usize::try_from(zeros).map_err(|_| precision_error())?;
    let significant_digits = last_nonzero - first_nonzero + 1;
    let len = significant_digits.saturating_add(zeros);
    // A finite binary64 integer has at most 309 decimal digits. Reject longer
    // magnitudes before indexing the fixed-size digit buffer, even if the input
    // wrote them using a short exponent.
    if len > 309 {
        return Err(precision_error());
    }
    // All integers with at most fifteen decimal digits fit in the contiguous exact
    // binary64 range. Longer integers may still be exact and require the
    // significant-bit checks below.
    if len <= 15 {
        return Ok(());
    }
    // The zero-initialized tail supplies positive decimal powers after significant
    // digits are copied. The length check above bounds both this stack allocation
    // and later repeated division.
    let mut storage = [0u8; 309];
    for (slot, byte) in storage.iter_mut().zip(
        mantissa
            .bytes()
            .filter(|byte| *byte != b'.')
            .skip(first_nonzero)
            .take(significant_digits),
    ) {
        *slot = byte - b'0';
    }
    let digits = &mut storage[..len];
    // Try a native unsigned accumulator first. If it overflows, keep examining the
    // decimal digits using long division rather than rejecting large powers of two
    // that binary64 represents exactly.
    if let Some(value) = digits.iter().try_fold(0u64, |value, digit| {
        value.checked_mul(10)?.checked_add(u64::from(*digit))
    }) {
        // Removing binary trailing zeroes leaves the odd significand. It must fit
        // the 53-bit precision even if the full integer occupies more than 53
        // bits.
        return if 64 - value.leading_zeros() - value.trailing_zeros() <= 53 {
            Ok(())
        } else {
            Err(precision_error())
        };
    }

    // Divide the bounded decimal integer before any float conversion. Binary64
    // represents it exactly only if at most 53 bits remain after trailing zeros.
    let mut bits = 0;
    let mut trailing_zero_bits = 0;
    let mut seen_one = false;
    while digits.iter().any(|digit| *digit != 0) {
        let mut remainder = 0;
        // Divide the decimal digits in place. Each remainder exposes the next low
        // bit, so the first nonzero remainder ends the run of powers of two that
        // can be represented by the exponent.
        for digit in digits.iter_mut() {
            let value = remainder * 10 + *digit;
            *digit = value / 2;
            remainder = value % 2;
        }
        bits += 1;
        // Count trailing zero bits only until the first one is seen. A later zero
        // is internal to the significand and cannot be absorbed into binary64's
        // exponent.
        if !seen_one && remainder == 0 {
            trailing_zero_bits += 1;
        }
        seen_one |= remainder != 0;
    }
    // Significand width is only half the test for large values. Even a single
    // significant bit fails if its binary position would require an exponent
    // beyond finite binary64.
    if bits <= 1024 && bits - trailing_zero_bits <= 53 {
        Ok(())
    } else {
        Err(precision_error())
    }
}

enum WriteFrame<'a> {
    // Carry each pending value with its indentation and output schema. Resuming a
    // container after writing a child then uses that container's own formatting
    // and kind for subsequent children.
    Value(&'a JsonValue, usize, JsonOutputSchema),
    // The iterator remembers the next child. The boolean remembers whether a comma
    // is due, including when an empty array closes without any child frame ever
    // being pushed.
    Array(
        std::slice::Iter<'a, JsonValue>,
        usize,
        bool,
        JsonOutputSchema,
    ),
    // A B-tree iterator preserves canonical key order. Keep its first-field and
    // indentation state across descents so nested objects need no native recursive
    // calls.
    Object(
        std::collections::btree_map::Iter<'a, String, JsonValue>,
        usize,
        bool,
        JsonOutputSchema,
    ),
}

#[cfg(test)]
pub(crate) fn test_write_frame_bytes() -> usize {
    std::mem::size_of::<WriteFrame<'_>>()
}

fn write_json(value: &JsonValue, out: &mut String, indent: usize, pretty: bool) {
    // This growable entry also serves callers that pre-reserve frames. The fixed
    // encoding workspace calls write_json_frames with bounded=true to prohibit
    // growth during output.
    write_json_with_frames(
        value,
        out,
        indent,
        pretty,
        &mut Vec::new(),
        JsonOutputSchema::Value(JsonInputKind::JsonlRecord),
    )
    .unwrap();
}

fn write_json_with_frames<'a, W: JsonWrite>(
    value: &'a JsonValue,
    out: &mut W,
    indent: usize,
    pretty: bool,
    frames: &mut Vec<WriteFrame<'a>>,
    schema: JsonOutputSchema,
) -> std::fmt::Result {
    write_json_frames(value, out, indent, pretty, frames, schema, false)
}

fn write_json_frames<'a, W: JsonWrite>(
    value: &'a JsonValue,
    out: &mut W,
    indent: usize,
    pretty: bool,
    frames: &mut Vec<WriteFrame<'a>>,
    schema: JsonOutputSchema,
    bounded: bool,
) -> std::fmt::Result {
    // Use one explicit LIFO stack for opening values and resuming their children.
    // Carry schema and indentation in every frame, so pushing a nested unit
    // changes its accounting classification without changing the enclosing
    // siblings' classification.
    use WriteFrame as Frame;
    let newline = |out: &mut W, indent: usize| -> std::fmt::Result {
        // Compact output never emits this whitespace. Pretty formatting only
        // affects presentation here; scalar spelling and member traversal order
        // are shared by both paths.
        if pretty {
            out.write_char('\n')?;
            for _ in 0..indent {
                out.write_char(' ')?;
            }
        }
        Ok(())
    };
    // The bounded caller supplies the entire stack capacity before writing starts.
    // Verify room for the root frame without quietly falling back to an uncharged
    // allocation.
    if bounded && frames.capacity() == 0 {
        return Err(std::fmt::Error);
    }
    frames.push(Frame::Value(value, indent, schema));
    // An open container frame holds its iterator at the next unwritten element.
    // After popping it, push the rest of the container before the child so the
    // next iteration emits the child first.
    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Value(value, indent, schema) => {
                // Set the effective output unit before writing any of the current
                // value. Embedded metadata or filter nodes may have their own
                // policy even when they are nested within a larger record.
                out.set_kind(schema.kind());
                match value {
                    JsonValue::Null => out.write_str("null")?,
                    JsonValue::Bool(value) => {
                        out.write_str(if *value { "true" } else { "false" })?
                    }
                    // Formatting a number uses the shared canonical scalar
                    // encoder. Its spelling feeds both measured lengths and
                    // ordinary output, so a change here can alter a stored
                    // payload's hash and checksum.
                    JsonValue::Number(value) => write_number(*value, out)?,
                    JsonValue::String(value) => write_string(value, out)?,
                    JsonValue::Array(values) => {
                        // Turning a value frame into its array iterator consumes
                        // the slot just released by the pop. The later descent
                        // from the iterator needs two slots because the parent
                        // must remain resumable.
                        out.write_char('[')?;
                        frames.push(Frame::Array(values.iter(), indent, true, schema));
                    }
                    JsonValue::Object(values) => {
                        out.write_char('{')?;
                        frames.push(Frame::Object(values.iter(), indent, true, schema));
                    }
                }
            }
            Frame::Array(mut items, indent, first, schema) => {
                out.set_kind(schema.kind());
                if let Some(item) = items.next() {
                    // Descending needs two stack slots: one to resume the parent
                    // iterator and one for the child. Check before emitting the
                    // separator so bounded output cannot grow an uncharged stack in
                    // the middle of a container.
                    if bounded && frames.capacity() - frames.len() < 2 {
                        return Err(std::fmt::Error);
                    }
                    // Use the first flag to avoid a leading comma. Descendant
                    // indentation is two spaces deeper in pretty mode, while
                    // compact mode uses the same frame mechanics with no spaces.
                    if !first {
                        out.write_char(',')?;
                    }
                    newline(out, indent + 2)?;
                    // Push the continuation before its child. The child can
                    // allocate several inner frames, but the next sibling retains
                    // this container's original schema and prefix state.
                    frames.push(Frame::Array(items, indent, false, schema));
                    frames.push(Frame::Value(item, indent + 2, schema.element()));
                } else {
                    // An empty array is closed immediately after its opening
                    // bracket. Only a previously emitted child needs the closing
                    // newline and parent indentation.
                    if !first {
                        newline(out, indent)?;
                    }
                    out.write_char(']')?;
                }
            }
            Frame::Object(mut items, indent, first, schema) => {
                out.set_kind(schema.kind());
                if let Some((key, item)) = items.next() {
                    // Check both stack slots before touching the member's
                    // punctuation or key. After writing that key, the child frame
                    // must be able to resume under the field-specific schema.
                    if bounded && frames.capacity() - frames.len() < 2 {
                        return Err(std::fmt::Error);
                    }
                    if !first {
                        out.write_char(',')?;
                    }
                    newline(out, indent + 2)?;
                    // Keep the key and separator under the object frame's kind.
                    // The value switches to the possibly narrower field kind when
                    // that value is emitted.
                    write_string(key, out)?;
                    out.write_char(':')?;
                    // The pretty path puts a space after the key separator.
                    // Container indentation and grammar still pass through the
                    // same sink, which can report output limits independently of
                    // the stack bound.
                    if pretty {
                        out.write_char(' ')?;
                    }
                    frames.push(Frame::Object(items, indent, false, schema));
                    frames.push(Frame::Value(item, indent + 2, schema.field(key)));
                } else {
                    if !first {
                        newline(out, indent)?;
                    }
                    out.write_char('}')?;
                }
            }
        }
    }
    Ok(())
}

// JSON string encoding operates on UTF-8 character boundaries. Emit contiguous
// ordinary text as borrowed slices and encode only special characters
// individually, avoiding a second escaped allocation.
fn write_string(value: &str, out: &mut impl std::fmt::Write) -> std::fmt::Result {
    out.write_char('"')?;
    let mut start = 0;
    for (offset, ch) in value.char_indices() {
        if ch != '"' && ch != '\\' && !ch.is_control() {
            continue;
        }
        // Emit all bytes from the previous escape boundary up to but excluding
        // this character. The cursor advances by that character's UTF-8 length, so
        // multi-byte text after an escape starts at a valid character boundary.
        out.write_str(&value[start..offset])?;
        match ch {
            '"' => out.write_str("\\\"")?,
            '\\' => out.write_str("\\\\")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            '\t' => out.write_str("\\t")?,
            '\u{08}' => out.write_str("\\b")?,
            '\u{0c}' => out.write_str("\\f")?,
            // Use the same escape spelling that the canonical parser expects.
            // Common controls use the short escapes above, and other control
            // characters use a lowercase hexadecimal escape.
            ch => write!(out, "\\u{:04x}", ch as u32)?,
        }
        start = offset + ch.len_utf8();
    }
    // Flush the final ordinary-text slice after the last escape. A string made
    // solely of ordinary text is emitted as one slice between the quotes.
    out.write_str(&value[start..])?;
    out.write_char('"')
}

struct Parser<R> {
    // Buffered byte source at its current input position. Looking ahead must not
    // advance the byte count used in diagnostics.
    source: R,
    // Require canonical lexical form during decoding, including number spellings
    // and object-key order.
    canonical: bool,
    // Number of source bytes consumed, used as a byte offset for parse errors
    // independently of UTF-8 character count.
    pos: usize,
    // Active business unit's kind and canonical byte total, when the schema starts
    // a bounded unit. An enclosing transport object can contain several such units.
    unit: Option<(JsonInputKind, usize)>,
    // Input kind of the current schema position, used to attribute errors in
    // different parts of the same document.
    kind: JsonInputKind,
    // Operation reservation for parser frames and values under construction. The
    // surviving tree keeps a reservation after traversal finishes.
    work: JsonReservation,
    // Exact integer captured from the designated query top_k token before
    // conversion to f64. Numbers elsewhere in the tree cannot set it.
    search_top_k: Option<usize>,
}

impl<R: BufRead> Parser<R> {
    fn parse_value(&mut self, depth: usize, schema: JsonSchema) -> Result<JsonValue> {
        struct Frame {
            // Partially built container. It stays on the stack while a child is
            // parsed.
            value: JsonValue,
            // Pending object member name, retained until its value is ready to
            // attach. An array does not need a key.
            key: Option<String>,
            // Schema of this container, used to select the child schema by field
            // name or element position.
            schema: JsonSchema,
            // Container depth tracked while descending this unit. Siblings resume
            // at the saved parent depth.
            depth: usize,
        }
        let mut frames = Vec::<Frame>::new();
        // The parser alternates between one requested child and one completed
        // value. Future children live in the stack of open containers, avoiding
        // one Rust call frame per level of untrusted input.
        let mut next = Some((schema, depth));
        let mut ready = None;
        loop {
            // Taking the requested child clears that work item. A decoded scalar
            // attaches later in this same loop iteration. A newly opened container
            // supplies the next child through its frame.
            if let Some((schema, depth)) = next.take() {
                self.kind = schema.kind();
                self.skip_ws()?;
                // A unit starts at depth zero according to the selected schema. A
                // record's recognized fields can begin new units below, allowing a
                // small row to report a specific metadata, filter or embedding
                // limit.
                if depth == 0 {
                    self.unit = match schema {
                        JsonSchema::Value(kind) => Some((kind, 0)),
                        _ => None,
                    };
                }
                // Require an object at a record boundary before admitting
                // children. A plain value schema may still accept scalar and array
                // roots, so the same parser can serve both inputs.
                if schema.is_record() && self.peek()? != Some(b'{') {
                    return Err(json_error("JSON record must be an object"));
                }
                if matches!(self.peek()?, Some(b'[' | b'{')) {
                    // Check depth while looking at the byte that opens a
                    // container. Empty objects and arrays still add a nesting
                    // level. A rejected opening consumes no frame capacity.
                    if depth >= MAX_JSON_DEPTH {
                        return Err(schema
                            .kind()
                            .limit_error(JsonLimitReason::Depth, MAX_JSON_DEPTH, Some(depth + 1))
                            .with_detail("byte_offset", self.pos));
                    }
                    // Account for both delimiters when admitting the container.
                    // Any commas or separators will be charged as they are
                    // encountered, independent of how many bytes were read from
                    // the buffer.
                    self.charge_unit(2)?;
                    let value = if self.next()? == Some(b'[') {
                        JsonValue::Array(Vec::new())
                    } else {
                        JsonValue::object()
                    };
                    // Keep the schema and depth on the new frame while its
                    // children are parsed. The surrounding value does not become
                    // ready until the closing delimiter is consumed.
                    self.work.reserve_vec(&mut frames, 1, self.kind)?;
                    frames.push(Frame {
                        value,
                        key: None,
                        schema,
                        depth: depth + 1,
                    });
                } else {
                    // Choose the scalar parser from the next byte without
                    // consuming it here. Each helper owns its token grammar and
                    // input advancement, which keeps error positions tied to the
                    // actual parse.
                    let value = match self.peek()? {
                        Some(b'n') => {
                            self.expect_literal(b"null")?;
                            JsonValue::Null
                        }
                        Some(b't') => {
                            self.expect_literal(b"true")?;
                            JsonValue::Bool(true)
                        }
                        Some(b'f') => {
                            self.expect_literal(b"false")?;
                            JsonValue::Bool(false)
                        }
                        Some(b'"') => JsonValue::String(self.parse_string()?),
                        Some(b'-' | b'0'..=b'9') => self.parse_number(schema)?,
                        Some(_) => {
                            return Err(json_error(format!("unexpected byte at {}", self.pos)))
                        }
                        None => return Err(json_error("unexpected end of input")),
                    };
                    // A query-embedding coordinate must fit in a finite f32. Use
                    // the rounded coordinate to charge the canonical input size for
                    // the vector, but leave the parsed JSON number intact for the
                    // caller.
                    if let (Some((JsonInputKind::QueryEmbedding, _)), JsonValue::Number(number)) =
                        (self.unit, &value)
                    {
                        let number = *number as f32;
                        if !number.is_finite() {
                            return Err(Error::new(
                                ErrorCode::InvalidVectorLength,
                                "query embedding vector entries must be finite f32 values",
                            ));
                        }
                        self.charge_unit(shallow_canonical_bytes(&JsonValue::Number(f64::from(
                            number,
                        ))))?;
                    } else if !matches!(value, JsonValue::String(_)) {
                        self.charge_unit(shallow_canonical_bytes(&value))?;
                    }
                    ready = Some(value);
                }
            }
            // A completed scalar or closed container goes back to the most recent
            // unfinished frame. An empty stack hands ownership of the root to the
            // caller; the frame allocation is scratch and can be released at that
            // point.
            let Some(frame) = frames.last_mut() else {
                // The root is ready with no unfinished parent. Free the traversal
                // array before releasing its byte credit from the reservation that
                // accompanies the returned tree.
                let bytes = frames.capacity() * std::mem::size_of::<Frame>();
                drop(frames);
                self.work.release(bytes);
                return Ok(ready.unwrap());
            };
            // Use readiness before taking the completed value. That bit determines
            // whether punctuation after attachment must separate it from a
            // following child.
            let after_value = ready.is_some();
            self.kind = frame.schema.kind();
            if let Some(value) = ready.take() {
                match &mut frame.value {
                    // Capacity for a growing array belongs to the retained value,
                    // not to the traversal stack. Admit a slot before placing the
                    // already decoded child in the array.
                    JsonValue::Array(values) => {
                        self.work.reserve_vec(values, 1, self.kind)?;
                        values.push(value);
                    }
                    // Admit the map node before moving the pending key and
                    // completed child into a single entry.
                    JsonValue::Object(values) => {
                        self.work.grow(
                            work::object_workspace(values.len() + 1)
                                - work::object_workspace(values.len()),
                            self.kind,
                        )?;
                        values.insert(frame.key.take().unwrap(), value);
                    }
                    _ => unreachable!(),
                }
                // A record envelope can contain several independently limited
                // values. End this field's accounting before parsing its sibling.
                if frame.schema.is_record() {
                    self.unit = None;
                }
            }
            self.kind = frame.schema.kind();
            // Use the delimiter that matches this frame's container. A mismatched
            // delimiter cannot terminate the current value and will fail when a
            // child or separator is expected.
            let closing = if matches!(frame.value, JsonValue::Array(_)) {
                b']'
            } else {
                b'}'
            };
            self.skip_ws()?;
            if self.consume_if(closing)? {
                ready = Some(frames.pop().unwrap().value);
                continue;
            }
            // Distinguish a newly opened empty collection from one that already
            // accepted a child. Only the latter requires a comma before
            // continuing, so a trailing comma still requires a following child.
            if after_value {
                self.expect_byte(b',')?;
                self.charge_unit(1)?;
            }
            if let JsonValue::Object(values) = &frame.value {
                self.skip_ws()?;
                // Admit the key's storage under the enclosing kind. For a record
                // member the key and colon precede the start of the next field's
                // separate canonical-size unit.
                let key = self.parse_string()?;
                self.skip_ws()?;
                self.expect_byte(b':')?;
                self.charge_unit(1)?;
                // Reject duplicates while the original key order is still
                // observable. In canonical mode also reject out-of-order keys now:
                // inserting into the sorted map would otherwise hide that the source
                // was noncanonical.
                if values.contains_key(&key) {
                    return Err(json_error("duplicate object key"));
                }
                if self.canonical && values.last_key_value().is_some_and(|(last, _)| last > &key) {
                    return Err(canonical_error());
                }
                // Select the child's schema while the key is still borrowed.
                // Record fields restart the depth count for their unit, while a
                // nested object advances with the enclosing field's depth.
                next = Some((
                    frame.schema.field(&key),
                    if frame.schema.is_record() {
                        0
                    } else {
                        frame.depth
                    },
                ));
                // Keep the key in the frame until the corresponding value is
                // ready. The child parser may open several nested containers
                // before returning the value that belongs under this key.
                frame.key = Some(key);
            } else {
                next = Some((frame.schema, frame.depth));
            }
        }
    }

    fn charge_unit(&mut self, bytes: usize) -> Result<()> {
        // This counter measures canonical output size for the current input unit.
        // Escaped strings and values that have canonical numeric sizing can differ
        // from their original byte count.
        if let Some((kind, used)) = &mut self.unit {
            *used = used.saturating_add(bytes);
            if *used > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES {
                return Err(kind.limit_error(
                    JsonLimitReason::UnitBytes,
                    JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                    Some(*used),
                ));
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect_byte(b'"')?;
        // Charge the quotes once per string. The remaining loop adds the canonical
        // width of each decoded scalar or copies an ASCII run already known to use
        // one byte per character.
        self.charge_unit(2)?;
        let mut out = String::new();
        while self.peek()?.is_some() {
            // Most text is unescaped ASCII; copy each buffered run with one reservation.
            // The buffer can end inside the string even when this run is entirely
            // ASCII. Consume only this prefix and loop, letting the next fill
            // decide whether the following bytes are ASCII or need special
            // handling.
            let count = self
                .source
                .fill_buf()?
                .iter()
                .take_while(|&&byte| (0x20..=0x7e).contains(&byte) && !matches!(byte, b'"' | b'\\'))
                .count();
            // The ASCII scan stops before a quote, backslash or control. That
            // guarantee lets the next prefix be borrowed as UTF-8 without per-byte
            // decoding, while admission still precedes copying.
            if count != 0 {
                self.charge_unit(count)?;
                self.work.reserve_string(&mut out, count, self.kind)?;
                out.push_str(std::str::from_utf8(&self.source.fill_buf()?[..count]).unwrap());
                self.consume(count)?;
                continue;
            }
            // Process exactly one byte at the end of the run. A closing quote
            // returns the accumulated output. An escape or leading UTF-8 byte can
            // instead require more reads before appending another scalar.
            let byte = self.next()?.unwrap();
            let ch = match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = self
                        .next()?
                        .ok_or_else(|| json_error("unterminated escape"))?;
                    match escaped {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' if !self.canonical => '/',
                        b'b' => '\u{08}',
                        b'f' => '\u{0c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let ch = self.parse_unicode_escape()?;
                            // In canonical mode a Unicode escape can represent
                            // controls whose writer uses that form. A printable
                            // scalar or a control with a short named escape has
                            // another canonical spelling.
                            if self.canonical
                                && (!ch.is_control()
                                    || matches!(ch, '\n' | '\r' | '\t' | '\u{08}' | '\u{0c}'))
                            {
                                return Err(canonical_error());
                            }
                            ch
                        }
                        // Escaping a slash is legal for ordinary JSON. Canonical
                        // output writes the slash directly, so canonical decoding
                        // rejects the alternate escape spelling.
                        b'/' => return Err(canonical_error()),
                        other => {
                            return Err(json_error(format!(
                                "invalid escape '\\{}' at {}",
                                other as char, self.pos
                            )));
                        }
                    }
                }
                // Raw bytes below U+0020 are invalid inside JSON strings in both
                // modes. The canonical path also checks whether other control
                // scalars arrived through their required escape spelling.
                byte if byte < 0x20 => {
                    return Err(json_error(format!(
                        "control character in string at {}",
                        self.pos
                    )));
                }
                byte if byte < 0x80 => byte as char,
                _ => {
                    // The buffered ASCII run ended at the first non-ASCII byte.
                    // Consume exactly one scalar across buffer boundaries and
                    // validate all of it before appending; a continuation byte alone
                    // cannot start the next character.
                    let width = match byte {
                        0xc2..=0xdf => 2,
                        0xe0..=0xef => 3,
                        0xf0..=0xf4 => 4,
                        _ => return Err(json_error("invalid utf-8 leading byte")),
                    };
                    let mut bytes = [0u8; 4];
                    bytes[0] = byte;
                    // Read the remaining bytes of this UTF-8 scalar, including
                    // across input-buffer boundaries. Decoding the fixed slice
                    // then checks continuation bytes and excludes invalid code
                    // points.
                    for slot in &mut bytes[1..width] {
                        *slot = self
                            .next()?
                            .ok_or_else(|| json_error("incomplete utf-8 character"))?;
                    }
                    let s = std::str::from_utf8(&bytes[..width])
                        .map_err(|err| json_error(format!("invalid utf-8: {err}")))?;
                    let ch = s
                        .chars()
                        .next()
                        .ok_or_else(|| json_error("invalid utf-8"))?;
                    ch
                }
            };
            // A valid UTF-8 scalar can still require escaping in canonical output.
            // Compare how it arrived to the writer's rule before adding it to the
            // retained string.
            if self.canonical && byte != b'\\' && ch.is_control() {
                return Err(canonical_error());
            }
            // Canonical size and native string capacity can differ. For example, a
            // control scalar can occupy one retained byte but several encoded
            // bytes. Admit both costs before pushing it.
            self.charge_unit(char_canonical_bytes(ch))?;
            self.work
                .reserve_string(&mut out, ch.len_utf8(), self.kind)?;
            out.push(ch);
        }
        Err(json_error("unterminated string"))
    }

    fn parse_unicode_escape(&mut self) -> Result<char> {
        let first = self.parse_hex_escape()?;
        // Combine a high surrogate with a following escaped low surrogate. An
        // isolated low surrogate fails on its own, and the pair must resolve to
        // one valid scalar before it is retained.
        let value = match first {
            0xd800..=0xdbff => {
                self.expect_byte(b'\\')?;
                self.expect_byte(b'u')?;
                let second = self.parse_hex_escape()?;
                if !(0xdc00..=0xdfff).contains(&second) {
                    return Err(json_error("invalid unicode surrogate pair"));
                }
                // Use the ten payload bits from each member of the surrogate pair
                // and add the supplementary-plane base. This reproduces the scalar
                // that the UTF-8 string will store rather than keeping either
                // surrogate code unit.
                0x10000 + ((first - 0xd800) << 10) + second - 0xdc00
            }
            0xdc00..=0xdfff => return Err(json_error("unpaired unicode surrogate")),
            value => value,
        };
        char::from_u32(value).ok_or_else(|| json_error("invalid unicode code point"))
    }

    fn parse_hex_escape(&mut self) -> Result<u32> {
        let mut value = 0u32;
        // Consume exactly four hexadecimal digits for this escape. Canonical input
        // permits the writer's lowercase spelling, while ordinary input accepts
        // uppercase digits as well.
        for _ in 0..4 {
            let byte = self
                .next()?
                .ok_or_else(|| json_error("unterminated unicode escape"))?;
            value = value * 16
                + match byte {
                    b'0'..=b'9' => (byte - b'0') as u32,
                    b'a'..=b'f' => (byte - b'a' + 10) as u32,
                    b'A'..=b'F' if !self.canonical => (byte - b'A' + 10) as u32,
                    b'A'..=b'F' => return Err(canonical_error()),
                    _ => return Err(json_error("invalid unicode escape")),
                };
        }
        Ok(value)
    }

    fn parse_number(&mut self, schema: JsonSchema) -> Result<JsonValue> {
        // Keep the start offset for grammar errors. The temporary token collects
        // the original decimal spelling so precision rules and canonical checks
        // can inspect digits before and after f64 conversion.
        let start = self.pos;
        let mut raw = NumberToken::default();
        if self.peek()? == Some(b'-') {
            self.number_byte(&mut raw)?;
        }
        // A leading zero ends the integer part. A following digit remains unread,
        // so a padded integer cannot be consumed as one JSON number.
        match self.peek()? {
            Some(b'0') => self.number_byte(&mut raw)?,
            Some(b'1'..=b'9') => {
                while matches!(self.peek()?, Some(b'0'..=b'9')) {
                    self.number_byte(&mut raw)?;
                }
            }
            _ => return Err(json_error(format!("invalid number at {start}"))),
        }
        if self.peek()? == Some(b'.') {
            self.number_byte(&mut raw)?;
            // A decimal point opens a fraction and therefore requires at least one
            // following digit. This grammar check happens before applying the
            // general floating-point parser.
            if !matches!(self.peek()?, Some(b'0'..=b'9')) {
                return Err(json_error(format!("invalid number at {start}")));
            }
            while matches!(self.peek()?, Some(b'0'..=b'9')) {
                self.number_byte(&mut raw)?;
            }
        }
        if matches!(self.peek()?, Some(b'e' | b'E')) {
            self.number_byte(&mut raw)?;
            if matches!(self.peek()?, Some(b'+' | b'-')) {
                self.number_byte(&mut raw)?;
            }
            // The exponent marker with an optional sign still requires a digit. An
            // incomplete exponent is a syntax error at the token's saved start.
            if !matches!(self.peek()?, Some(b'0'..=b'9')) {
                return Err(json_error(format!("invalid exponent at {start}")));
            }
            while matches!(self.peek()?, Some(b'0'..=b'9')) {
                self.number_byte(&mut raw)?;
            }
        }
        // Keep the spelling until top_k has been checked. A fraction extremely
        // close to an integer can round to that integer as an f64. The query limit
        // must be exact in the input, before conversion to the general JSON number
        // representation.
        validate_integer_precision(raw.as_str())?;
        // The exact positive top_k becomes a separate parsed field. Keep it before
        // converting to f64 so the typed query can consume the validated integer
        // value.
        if matches!(schema, JsonSchema::QueryTopK) {
            self.search_top_k = Some(exact_search_top_k(raw.as_str()).ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "query top_k must be a positive integer in the supported exact JSON range",
                )
                .with_detail("field", "queries.top_k")
            })?);
        }
        let value = raw
            .as_str()
            .parse::<f64>()
            .map_err(|_| json_error("invalid JSON number"))?;
        // A decimal token can satisfy the grammar and overflow the available
        // numeric representation. Reject a nonfinite result even for inputs that
        // do not request canonical spelling.
        if !value.is_finite() {
            return Err(json_error("JSON number is not finite"));
        }
        // Compare the original token with the writer's representation after
        // numeric conversion. A valid floating-point value may have alternate
        // valid JSON spellings that the stored codec cannot accept as canonical.
        if self.canonical && !is_canonical_number(value, raw.as_str()) {
            return Err(canonical_error());
        }
        // Release only the temporary spelling's allocated capacity once the check
        // finishes. The inline buffer required no heap reservation, and the
        // returned numeric value fits directly in a JsonValue slot.
        let capacity = raw.spill.capacity();
        drop(raw);
        self.work.release(capacity);
        Ok(JsonValue::Number(value))
    }

    fn number_byte(&mut self, raw: &mut NumberToken) -> Result<()> {
        let byte = self
            .next()?
            .ok_or_else(|| json_error("incomplete number"))?;
        if raw.len < raw.inline.len() {
            raw.inline[raw.len] = byte;
        } else {
            // On the first spill, copy the already filled inline buffer into newly
            // admitted string capacity. Later spills grow that same string
            // incrementally, so the first digits stay available for the precision
            // test.
            if raw.spill.is_empty() {
                self.work
                    .reserve_string(&mut raw.spill, raw.len + 1, self.kind)?;
                raw.spill
                    .push_str(std::str::from_utf8(&raw.inline).unwrap());
            } else {
                // The first spill already copied the inline prefix. Reserve one
                // more byte for each subsequent byte of the spelling, including an
                // exponent marker or sign.
                self.work.reserve_string(&mut raw.spill, 1, self.kind)?;
            }
            raw.spill.push(char::from(byte));
        }
        raw.len += 1;
        Ok(())
    }

    fn expect_literal(&mut self, literal: &[u8]) -> Result<()> {
        // A literal is matched byte by byte and has no separate allocation. The
        // caller still checks whatever delimiter follows it, as with a numeric
        // token.
        for &byte in literal {
            if self.next()? != Some(byte) {
                return Err(json_error(format!("expected literal at {}", self.pos)));
            }
        }
        Ok(())
    }

    fn expect_byte(&mut self, expected: u8) -> Result<()> {
        // Report a mismatching consumed byte at its original zero-based offset,
        // since next has already advanced the position. EOF has no offending byte
        // to display.
        match self.next()? {
            Some(byte) if byte == expected => Ok(()),
            Some(byte) => Err(json_error(format!(
                "expected '{}', got '{}' at {}",
                expected as char,
                byte as char,
                self.pos.saturating_sub(1)
            ))),
            None => Err(json_error(format!("expected '{}'", expected as char))),
        }
    }

    fn consume_if(&mut self, expected: u8) -> Result<bool> {
        // Probe before consuming. When the caller is deciding whether a delimiter
        // is present, a mismatch must leave the byte for the next grammar check.
        if self.peek()? == Some(expected) {
            self.consume(1)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn skip_ws(&mut self) -> Result<()> {
        // Recognize only the JSON whitespace set. Canonical input fails at the
        // first such byte outside a value's string. Ordinary input consumes the
        // current buffered run in one step.
        while matches!(self.peek()?, Some(b' ' | b'\n' | b'\r' | b'\t')) {
            if self.canonical {
                return Err(canonical_error());
            }
            let count = self
                .source
                .fill_buf()?
                .iter()
                .take_while(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
                .count();
            self.consume(count)?;
        }
        Ok(())
    }

    fn peek(&mut self) -> Result<Option<u8>> {
        // Retry an interrupted fill without advancing the offset. Other reader
        // failures propagate, while an empty buffer produces EOF for the caller to
        // classify in context.
        loop {
            match self.source.fill_buf() {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => return Ok(result?.first().copied()),
            }
        }
    }

    fn consume(&mut self, count: usize) -> Result<()> {
        // Update the checked absolute input position before consuming from the
        // reader. A failure in position arithmetic therefore cannot leave the
        // physical cursor advanced beyond the diagnostic offset.
        self.pos = self
            .pos
            .checked_add(count)
            .ok_or_else(|| json_error("JSON input offset exceeds supported range"))?;
        self.source.consume(count);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<u8>> {
        let byte = self.peek()?;
        if byte.is_some() {
            self.consume(1)?;
        }
        Ok(byte)
    }
}

struct NumberToken {
    // Fixed storage for common numeric spellings, kept available until the token is
    // complete and precision can be validated.
    inline: [u8; 128],
    // Total token bytes consumed, including bytes stored in spill. Before the first
    // spill this also indexes the next free byte in inline.
    len: usize,
    // Heap-backed spelling used when the token outgrows inline storage. Retain the
    // digits so precision checks run on the source before binary64 rounding.
    spill: String,
}

impl Default for NumberToken {
    fn default() -> Self {
        Self {
            inline: [0; 128],
            len: 0,
            spill: String::new(),
        }
    }
}

impl NumberToken {
    fn as_str(&self) -> &str {
        // A token is kept entirely inline or entirely in the spill string. After
        // crossing that boundary the spill includes the inline prefix, so callers
        // need no concatenation before validation.
        if self.spill.is_empty() {
            std::str::from_utf8(&self.inline[..self.len]).unwrap()
        } else {
            &self.spill
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_top_k_token_validation_is_schema_scoped_and_releases_parser_credit() {
        let context = JsonOperationContext::standalone();
        let baseline = context.available_workspace_bytes();
        for raw in [
            "1.0000000000000001",
            "9007199254740991.1",
            "1e-100000000000000000000",
        ] {
            let input = format!(r#"{{"top_k":{raw},"filter":{{"top_k":1.0000000000000001}}}}"#);
            assert!(parse_with_schema_in(&input, JsonSchema::QueryRow, &context).is_err());
            assert_eq!(context.available_workspace_bytes(), baseline);
            // The ordinary numeric contract and a nested filter value are unchanged.
            drop(
                parse_with_schema_in(
                    &input,
                    JsonSchema::Value(JsonInputKind::JsonlRecord),
                    &context,
                )
                .unwrap(),
            );
            assert_eq!(context.available_workspace_bytes(), baseline);
        }
        for input in [
            r#"{"top_k":10.0,"filter":{"top_k":1.0000000000000001},"embedding":{"dim":2,"vector":[3,4]}}"#,
            r#"{"filter":{"top_k":1.0000000000000001},"embedding":{"dim":2,"vector":[3,4]},"top_k":1e1}"#,
        ] {
            let parsed = parse_with_schema_in(input, JsonSchema::QueryRow, &context).unwrap();
            assert_eq!(parsed.search_top_k, Some(10));
            drop(parsed);
            assert_eq!(context.available_workspace_bytes(), baseline);
        }
        assert!(
            parse_with_schema_in(r#"{"top_k":10,"top_k":20}"#, JsonSchema::QueryRow, &context)
                .is_err()
        );
        assert_eq!(context.available_workspace_bytes(), baseline);
        let input = r#"{"top_k":10.0,"filter":{"top_k":1.0000000000000001}}"#;
        let parsed = parse_with_schema_in(input, JsonSchema::QueryRow, &context).unwrap();
        assert_eq!(
            parsed.value.get("top_k").and_then(JsonValue::as_f64),
            Some(10.0)
        );
        assert_eq!(parsed.search_top_k, Some(10));
        drop(parsed);
        assert_eq!(context.available_workspace_bytes(), baseline);
    }

    #[test]
    fn native_json_parse_and_copy_failures_preserve_classification_input_and_credit() {
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::Filter,
            JsonInputKind::StoredJson,
        ] {
            let bytes = 131_071;
            let value = JsonValue::String("x".repeat(bytes));
            let wire = format!("\"{}\"", value.as_str().unwrap());
            let context = JsonOperationContext::standalone();
            let capacity = context.available_workspace_bytes();
            let input_work = value.reserve_retained_in(kind, &context).unwrap();
            let baseline = context.available_workspace_bytes();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                parse_with_schema_in(&wire, JsonSchema::Value(kind), &context)
            })
            .err()
            .unwrap();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert!(!is_memory_budget_error(&error));
            assert_eq!(context.available_workspace_bytes(), baseline);
            let parsed = parse_with_schema_in(&wire, JsonSchema::Value(kind), &context).unwrap();
            assert_eq!(parsed.value, value);
            drop(parsed);
            assert_eq!(context.available_workspace_bytes(), baseline);
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                value.try_clone_in(kind, &context)
            })
            .err()
            .unwrap();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(context.available_workspace_bytes(), baseline);
            let copy = value.try_clone_in(kind, &context).unwrap();
            assert_eq!(copy.value, value);
            drop(copy);
            let mut copy_work = context.reservation();
            copy_work.grow(17, kind).unwrap();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                copy_work.copy_validated_value(&value, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(copy_work.reserved_bytes(), 17);
            assert_eq!(context.available_workspace_bytes(), baseline - 17);
            let copy = copy_work.copy_validated_value(&value, kind).unwrap();
            assert_eq!(copy, value);
            assert_eq!(copy_work.reserved_bytes(), bytes + 17);
            drop(copy);
            drop(copy_work);
            assert_eq!(context.available_workspace_bytes(), baseline);
            assert_eq!(value.as_str().unwrap().len(), bytes);
            drop(value);
            drop(input_work);
            assert_eq!(context.available_workspace_bytes(), capacity);
        }
    }

    #[test]
    fn retained_layout_spare_does_not_block_necessary_stack_growth() {
        let mut value = JsonValue::Null;
        for _ in 0..5 {
            value = JsonValue::Array(vec![value]);
        }
        let bytes = 5 * std::mem::size_of::<JsonValue>();
        for parent in [false, true] {
            for limit in (4096..=5120).step_by(32) {
                let ledger = std::sync::Arc::new(WorkLedger::with_limits(
                    if parent { 8192 } else { limit },
                    0,
                    if parent { limit } else { 8192 },
                ));
                let context = JsonOperationContext::new(ledger);
                let (retained, canonical) =
                    value.guard_in(JsonInputKind::StoredJson, &context).unwrap();
                assert_eq!(retained.reserved_bytes(), bytes);
                assert_eq!(canonical, 14);
                assert_eq!(context.available_workspace_bytes(), limit - bytes);
                drop(retained);
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn retained_handoff_reuses_credit_and_releases_only_discarded_allocations() {
        let context = JsonOperationContext::standalone();
        let kind = JsonInputKind::StoredJson;
        let mut value = object_from_pairs([
            ("keep", JsonValue::String("x".repeat(6 * 1024 * 1024))),
            ("discard", JsonValue::String("y".repeat(1024 * 1024))),
        ]);
        let mut work = value.reserve_retained_in(kind, &context).unwrap();
        let mut other = context.reservation();
        other
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes() - 4096,
                kind,
            )
            .unwrap();
        // A second complete charge cannot fit, but transferring the same nodes can.
        assert!(value.reserve_retained_in(kind, &context).is_err());
        value.reconcile_retained_in(kind, &mut work).unwrap();
        let previous = work.reserved_bytes();
        drop(value.as_object_mut().unwrap().remove("discard"));
        value.reconcile_retained_in(kind, &mut work).unwrap();
        assert_eq!(
            previous - work.reserved_bytes(),
            1024 * 1024 + "discard".len()
        );
        let mut released = context.reservation();
        released.grow(1024 * 1024, kind).unwrap();
        drop(released);
        drop(other);
        let measured = value.reserve_retained_in(kind, &context).unwrap();
        assert_eq!(measured.reserved_bytes(), work.reserved_bytes());
    }

    #[test]
    fn retained_handoff_charges_scratch_and_does_not_release_credit_on_error() {
        let context = JsonOperationContext::standalone();
        let kind = JsonInputKind::StoredJson;
        let value = object_from_pairs([("payload", JsonValue::String("x".repeat(4096)))]);
        let mut work = value.reserve_retained_in(kind, &context).unwrap();
        let previous = work.reserved_bytes();
        let mut other = context.reservation();
        other
            .grow(JsonProcessingPolicyV1::WORK_BUDGET_BYTES - previous, kind)
            .unwrap();
        assert!(value.reconcile_retained_in(kind, &mut work).is_err());
        assert_eq!(work.reserved_bytes(), previous);
        drop(other);
        value.reconcile_retained_in(kind, &mut work).unwrap();
        assert_eq!(work.reserved_bytes(), previous);
        drop(value);
        drop(work);
        context
            .reservation()
            .grow(JsonProcessingPolicyV1::WORK_BUDGET_BYTES, kind)
            .unwrap();
    }

    struct Fragmented<'a> {
        bytes: &'a [u8],
        fragment: usize,
        interrupt: bool,
        consumed: usize,
    }

    impl Read for Fragmented<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            assert!(buffer.len() <= 8192);
            self.interrupt = !self.interrupt;
            if self.interrupt {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let count = buffer.len().min(self.fragment).min(self.bytes.len());
            buffer[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            self.consumed += count;
            Ok(count)
        }
    }

    #[test]
    fn typed_object_guard_reuses_precharged_strings_and_keeps_canonical_checks() {
        let context = JsonOperationContext::standalone();
        let initial = context.available_workspace_bytes();
        let kind = JsonInputKind::CollectionConfig;
        let mut strings = context.reservation();
        let text = "quote\"\n\u{85}\u{4e2d}".to_owned();
        strings.grow(text.capacity(), kind).unwrap();
        let mut guard = JsonObjectGuard::new(kind, &context);
        guard.string_field_precharged("name", &text).unwrap();
        let fields = work::object_workspace(1) + "name".len();
        assert_eq!(guard.work.reserved_bytes(), fields);
        assert_eq!(
            context.available_workspace_bytes(),
            initial - fields - text.capacity()
        );
        assert_eq!(
            guard.bytes,
            2 + string_canonical_bytes("name") + 1 + string_canonical_bytes(&text)
        );
        drop(guard);
        assert_eq!(
            context.available_workspace_bytes(),
            initial - text.capacity()
        );
        drop(text);
        drop(strings);
        assert_eq!(context.available_workspace_bytes(), initial);
    }

    #[test]
    fn typed_object_guard_counts_enclosing_depth_and_canonical_field_bytes() {
        let context = JsonOperationContext::standalone();
        let kind = JsonInputKind::CollectionConfig;
        let text = "quote\"\n\u{85}é".to_owned();
        let child = parse("[null,1,{}]").unwrap();
        let mut guard = JsonObjectGuard::new(kind, &context);
        guard.string_field("name", &text).unwrap();
        guard.value_field("child", &child).unwrap();
        let complete = object_from_pairs([("name", JsonValue::String(text)), ("child", child)]);
        let measured = complete.reserve_retained_in(kind, &context).unwrap();
        assert_eq!(guard.work.reserved_bytes(), measured.reserved_bytes());
        drop(measured);
        assert_eq!(guard.bytes, complete.to_compact_string().len());
        drop(guard);
        for (depth, accepted) in [(127, true), (128, false)] {
            let value = parse(&mixed_json(depth, "0")).unwrap();
            let mut guard = JsonObjectGuard::new(kind, &context);
            let result = guard.value_field("nested", &value);
            assert_eq!(result.is_ok(), accepted);
            if let Err(error) = result {
                assert!(is_depth_error(&error));
            }
        }
    }

    #[test]
    fn typed_object_guard_retains_every_fields_workspace() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let kind = JsonInputKind::CollectionConfig;
            let name = "n".repeat(seed * 17);
            let first = JsonValue::Array(vec![JsonValue::Null; seed]);
            let second = object_from_pairs([("text", JsonValue::String("x".repeat(seed * 31)))]);
            let mut guard = JsonObjectGuard::new(kind, &context);
            guard.string_field("name", &name).unwrap();
            guard.value_field("first", &first).unwrap();
            guard.value_field("second", &second).unwrap();
            let complete = object_from_pairs([
                ("name", JsonValue::String(name)),
                ("first", first),
                ("second", second),
            ]);
            let measured = complete.reserve_retained_in(kind, &context).unwrap();
            assert_eq!(guard.work.reserved_bytes(), measured.reserved_bytes());
        }
    }

    #[test]
    fn bounded_canonical_input_reserves_its_frame_size_and_preserves_the_next_field() {
        struct Checked<'a> {
            source: Cursor<&'a [u8]>,
            context: &'a JsonOperationContext,
            buffer_bytes: usize,
            fragment: usize,
            first: bool,
        }
        impl Read for Checked<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                assert!(output.len() <= self.buffer_bytes);
                if self.first {
                    self.first = false;
                    let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
                    let mut other = self.context.workspace_reservation();
                    other
                        .grow_to(
                            limit - self.buffer_bytes,
                            limit - self.buffer_bytes,
                            "operation_scratch",
                        )
                        .unwrap();
                    assert!(other
                        .grow_to(
                            limit - self.buffer_bytes + 1,
                            limit - self.buffer_bytes + 1,
                            "operation_scratch"
                        )
                        .is_err());
                }
                let count = output.len().min(self.fragment);
                self.source.read(&mut output[..count])
            }
        }

        for seed in 0..100usize {
            let value = if seed % 4 == 0 {
                JsonValue::object()
            } else {
                object_from_pairs([("value", JsonValue::String("\u{4e2d}\n".repeat(seed * 31)))])
            };
            let canonical = value.to_compact_string();
            let mut wire = canonical.as_bytes().to_vec();
            wire.extend_from_slice(b"next field");
            let context = JsonOperationContext::standalone();
            let mut source = Checked {
                source: Cursor::new(wire.as_slice()),
                context: &context,
                buffer_bytes: canonical.len().min(8192),
                fragment: 1 + seed * 17,
                first: true,
            };
            let mut limited = (&mut source).take(canonical.len() as u64);
            let parsed =
                parse_canonical_reader_in(&mut limited, JsonInputKind::StoredJson, &context)
                    .unwrap();
            assert_eq!(limited.limit(), 0);
            assert_eq!(parsed.value, value);
            assert_eq!(source.source.position(), canonical.len() as u64);
            assert!(!source.first);
            let mut other = context.workspace_reservation();
            let remaining =
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - parsed._reservation.reserved_bytes();
            other
                .grow_to(remaining, remaining, "operation_scratch")
                .unwrap();
            drop(parsed);
            other
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
    }

    #[test]
    fn bounded_canonical_input_keeps_json_limits_and_releases_failed_buffers() {
        let kind = JsonInputKind::StoredJson;
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let context = JsonOperationContext::standalone();
            let mut held = context.workspace_reservation();
            held.grow_to(limit - 512, limit - 512, "operation_scratch")
                .unwrap();
            let mut small = Cursor::new(b"{}next").take(2);
            assert!(parse_canonical_reader_in(&mut small, kind, &context)
                .unwrap()
                .value
                .as_object()
                .unwrap()
                .is_empty());
            assert_eq!(small.into_inner().position(), 2);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            for raw in [b"{}".as_slice(), b"{\"a\":1}".as_slice()] {
                let mut source = Cursor::new(raw).take(raw.len() as u64);
                let error = parse_canonical_reader_in(&mut source, kind, &context)
                    .err()
                    .unwrap();
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert_eq!(source.get_ref().position(), 0);
                assert!(is_limit_error(&error));
                assert!(error
                    .detail()
                    .iter()
                    .any(|(key, value)| key == "limit" && value == "0"));
            }
            held.retain(0);
            for raw in [
                "".to_owned(),
                "{\"a\":1".to_owned(),
                "{\"b\":1,\"a\":2}".to_owned(),
                "{\"a\":0,\"a\":1}".to_owned(),
                format!("{}0{}", "[".repeat(129 + seed), "]".repeat(129 + seed)),
            ] {
                let mut source = Cursor::new(raw.as_bytes()).take(raw.len() as u64);
                assert!(parse_canonical_reader_in(&mut source, kind, &context).is_err());
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(0);
            }
        }
    }

    #[test]
    fn streaming_parser_preserves_syntax_precision_and_split_unicode() {
        let long_number = format!("1.{}1", "0".repeat(150));
        for raw in [
            r#" {"a":[null,true,false,-1.25e-20],"b":"é存储\ud83d\ude80\n\u0085"} "#,
            r#"{"a":0,"\u0061":1}"#,
            "[1,]",
            "1e+",
            "9007199254740993",
            "9007199254740993.000000000000000000000000000000000000",
            "[\"\\ud800\"]",
            &long_number,
        ] {
            let expected = parse(raw);
            for fragment in [1, 2, 3, 7, 127, 8192] {
                let mut source = Fragmented {
                    bytes: raw.as_bytes(),
                    fragment,
                    interrupt: false,
                    consumed: 0,
                };
                let actual =
                    parse_reader(&mut source, JsonSchema::Value(JsonInputKind::JsonlRecord));
                match (&expected, actual) {
                    (Ok(expected), Ok(actual)) => assert_eq!(expected, &actual),
                    (Err(expected), Err(actual)) => assert_eq!(expected.code(), actual.code()),
                    (_, actual) => panic!("fragment={fragment}, input={raw:?}, result={actual:?}"),
                }
            }
        }
        for bytes in [
            &b"\"\xc0\x80\""[..],
            &b"\"\xed\xa0\x80\""[..],
            &b"\"\xf0\x90"[..],
        ] {
            assert!(parse_reader(bytes, JsonSchema::Value(JsonInputKind::Manifest)).is_err());
        }
    }

    #[test]
    fn streaming_parser_bounds_prefetch_and_does_not_count_wire_whitespace() {
        let whitespace = std::io::repeat(b' ').take(40 * 1_048_576);
        let source = whitespace.chain(Cursor::new(b"{\"a\":1}\n\t"));
        assert_eq!(
            parse_reader(source, JsonSchema::Value(JsonInputKind::Manifest)).unwrap(),
            parse("{\"a\":1}").unwrap()
        );
        let raw = "[".repeat(100_000);
        let mut source = Fragmented {
            bytes: raw.as_bytes(),
            fragment: 8192,
            interrupt: false,
            consumed: 0,
        };
        let error =
            parse_reader(&mut source, JsonSchema::Value(JsonInputKind::Manifest)).unwrap_err();
        assert!(is_depth_error(&error));
        assert_eq!(source.consumed, 8192);
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::Other.into())
            }
        }
        assert_eq!(
            parse_reader(Broken, JsonSchema::Value(JsonInputKind::Manifest))
                .unwrap_err()
                .code(),
            ErrorCode::IoError
        );
    }

    #[test]
    fn streamed_canonical_validation_and_digest_match_the_writer() {
        let mut seed = 0x6a09_e667_f3bc_c909u64;
        for round in 0..100 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let number = f64::from_bits(seed);
            let number = if number.is_finite() { number } else { 0.0 };
            let value = object_from_pairs([
                (
                    "a",
                    JsonValue::Array(vec![
                        JsonValue::Number(number),
                        JsonValue::Number(-0.0),
                        JsonValue::Bool(round % 2 == 0),
                    ]),
                ),
                (
                    "b\n\u{7f}存储",
                    JsonValue::String(format!(
                        "{round}:\"\\/\u{0}\u{08}\u{0c}\r\n\t\u{7f}\u{85}é存储"
                    )),
                ),
            ]);
            let canonical = value.to_compact_string();
            assert_eq!(
                value.canonical_digest(JsonInputKind::StoredJson).unwrap(),
                (
                    canonical.len(),
                    crate::hash::sha256_hex(canonical.as_bytes())
                )
            );
            for fragment in [1, 7, 8192] {
                let source = Fragmented {
                    bytes: canonical.as_bytes(),
                    fragment,
                    interrupt: false,
                    consumed: 0,
                };
                assert_eq!(
                    parse_canonical_reader(source, JsonInputKind::StoredJson).unwrap(),
                    value
                );
            }
        }
        for raw in [
            " {}",
            "{}\n",
            "{\"a\" :0}",
            "{\"b\":0,\"a\":1}",
            "-0",
            "0.0",
            "1e0",
            "1.00",
            "1E+1",
            r#""\/""#,
            r#""\u0061""#,
            r#""\u000a""#,
            r#""\u007F""#,
            r#""\ud83d\ude80""#,
            "\"\u{7f}\"",
            "\"\u{85}\"",
        ] {
            assert!(parse(raw).is_ok(), "{raw:?}");
            assert_eq!(
                parse_canonical_reader(raw.as_bytes(), JsonInputKind::StoredJson)
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError,
                "{raw:?}"
            );
        }
    }

    #[test]
    fn canonical_unit_bytes_count_escaping_keys_and_not_wire_whitespace() {
        let limit = JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES;
        let exact = JsonValue::String("x".repeat(limit - 2));
        assert!(exact.validate_kind(JsonInputKind::DocumentMetadata).is_ok());
        let raw = format!("  {}\n\t", exact.to_compact_string());
        assert_eq!(
            parse_with_kind(&raw, JsonInputKind::DocumentMetadata).unwrap(),
            exact
        );
        let too_large = JsonValue::String("x".repeat(limit - 1));
        for error in [
            too_large
                .validate_kind(JsonInputKind::DocumentMetadata)
                .unwrap_err(),
            parse_with_kind(
                &too_large.to_compact_string(),
                JsonInputKind::DocumentMetadata,
            )
            .unwrap_err(),
        ] {
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "unit_bytes_exceeded"));
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("actual")
                    .and_then(JsonValue::as_f64),
                Some((limit + 1) as f64)
            );
        }
        let escaped =
            object_from_pairs([("\"\\\t\u{7f}\u{85}", JsonValue::String("é\n\"".into()))]);
        let root = shallow_canonical_bytes(&escaped);
        let leaf = escaped.as_object().unwrap().values().next().unwrap();
        assert_eq!(
            root + shallow_canonical_bytes(leaf),
            escaped.to_compact_string().len()
        );
        assert!(parse_with_kind(
            &format!("{}0", " ".repeat(limit + 1)),
            JsonInputKind::Manifest
        )
        .is_ok());
    }

    #[test]
    fn record_schema_does_not_charge_outer_fields_to_metadata() {
        for (schema, kind) in [
            (JsonSchema::DocumentRow, JsonInputKind::DocumentMetadata),
            (JsonSchema::ChunkRow, JsonInputKind::ChunkMetadata),
            (JsonSchema::StoredRecord, JsonInputKind::StoredJson),
        ] {
            let raw = format!("{{\"metadata\":{}}}", mixed_json(MAX_JSON_DEPTH, "0"));
            assert!(parse_with_schema(&raw, schema).is_ok());
            let raw = format!("{{\"metadata\":{}}}", mixed_json(MAX_JSON_DEPTH + 1, "0"));
            let error = parse_with_schema(&raw, schema).unwrap_err();
            assert_eq!(error.code(), kind.error_code());
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == kind.as_str()));
        }
        let text = "x".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES + 1);
        assert!(parse_with_schema(
            &format!("{{\"text\":\"{text}\",\"metadata\":{{}}}}"),
            JsonSchema::ChunkRow
        )
        .is_ok());
        let raw = format!("{{\"metadata\":\"{text}\"}}");
        assert!(is_limit_error(
            &parse_with_schema(&raw, JsonSchema::ChunkRow).unwrap_err()
        ));
        // An unknown nested wrapper is not a recognized schema boundary.
        let raw = format!("{{\"unknown\":{}}}", mixed_json(MAX_JSON_DEPTH + 1, "0"));
        assert!(is_depth_error(
            &parse_with_schema(&raw, JsonSchema::ChunkRow).unwrap_err()
        ));
    }

    #[test]
    fn query_vector_canonical_bytes_use_the_same_f32_values_for_text_and_typed_inputs() {
        for raw in ["[0.1,0.2,-0.0,1e-50]", "[1.00000001,0.125,-8388608]"] {
            let mut parser = Parser {
                source: Cursor::new(raw.as_bytes()),
                canonical: false,
                pos: 0,
                unit: None,
                kind: JsonInputKind::QueryEmbedding,
                work: JsonOperationContext::standalone().reservation(),
                search_top_k: None,
            };
            let value = parser
                .parse_value(0, JsonSchema::Value(JsonInputKind::QueryEmbedding))
                .unwrap();
            let typed: Vec<_> = value
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            validate_query_vector(&typed).unwrap();
            let normalized = JsonValue::Array(
                typed
                    .into_iter()
                    .map(|value| JsonValue::Number(f64::from(value)))
                    .collect(),
            );
            assert_eq!(parser.unit.unwrap().1, normalized.to_compact_string().len());
        }
        let error = parse_with_kind(
            "[340282366920938463463374607431768211456]",
            JsonInputKind::QueryEmbedding,
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidVectorLength);
    }

    #[test]
    fn parser_uses_explicit_stack_on_small_host_threads_and_rejects_duplicates() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                assert!(parse(&mixed_json(MAX_JSON_DEPTH, "0")).is_ok());
                assert!(is_depth_error(&parse(&"[".repeat(30_000)).unwrap_err()));
                for raw in [
                    r#"{"a":null,"\u0061":1}"#,
                    "[1,]",
                    "{\"a\":0,}",
                    "[[] 0]",
                    "{}[]",
                ] {
                    assert!(parse(raw).is_err(), "{raw}");
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn work_budget_limits_wide_values_without_confusing_them_with_unit_bytes() {
        let values = JsonValue::Array(vec![JsonValue::Null; 1_100_000]);
        let error = values
            .validate_kind(JsonInputKind::DocumentMetadata)
            .unwrap_err();
        let detail = error.to_json();
        assert_eq!(
            detail
                .get("detail")
                .unwrap()
                .get("json_limit_reason")
                .unwrap()
                .as_str(),
            Some("work_budget_exceeded")
        );
        let raw = format!("[{}null]", "null,".repeat(1_100_000));
        let error = parse_with_kind(&raw, JsonInputKind::DocumentMetadata).unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert!(raw.len() < JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES);
    }

    #[test]
    fn jsonl_outer_text_work_limit_uses_record_kind_but_metadata_keeps_its_kind() {
        for (field, expected) in [
            ("text", JsonInputKind::JsonlRecord),
            ("metadata", JsonInputKind::ChunkMetadata),
        ] {
            let raw = if field == "text" {
                format!(
                    "{{\"metadata\":{{}},\"text\":\"{}\"}}",
                    "x".repeat(JsonProcessingPolicyV1::WORK_BUDGET_BYTES)
                )
            } else {
                format!(
                    "{{\"metadata\":[{}null],\"text\":\"x\"}}",
                    "null,".repeat(1_100_000)
                )
            };
            let error = parse_with_schema(&raw, JsonSchema::ChunkRow).unwrap_err();
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == expected.as_str()));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        }
    }

    #[test]
    fn parsed_values_hold_one_operation_budget_until_their_owned_memory_is_released() {
        let context = JsonOperationContext::standalone();
        let raw = format!("[{}null]", "null,".repeat(600_000));
        let first =
            parse_with_schema_in(&raw, JsonSchema::Value(JsonInputKind::Filter), &context).unwrap();
        let error = parse_with_schema_in(&raw, JsonSchema::Value(JsonInputKind::Filter), &context)
            .err()
            .unwrap();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        drop(first);
        assert!(
            parse_with_schema_in(&raw, JsonSchema::Value(JsonInputKind::Filter), &context).is_ok()
        );
    }

    #[test]
    fn number_byte_measurement_agrees_with_serialization_at_binary64_extremes() {
        for value in [
            0.0,
            -0.0,
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            0.1,
            -0.1,
        ] {
            let json = JsonValue::Number(value);
            assert_eq!(
                number_canonical_bytes(value),
                json.to_compact_string().len()
            );
        }
    }

    fn mixed_json(depth: usize, leaf: &str) -> String {
        let mut prefix = String::new();
        let mut suffix = String::new();
        for n in 0..depth {
            if n % 2 == 0 {
                prefix.push_str("{\"v\":");
                suffix.push('}');
            } else {
                prefix.push('[');
                suffix.push(']');
            }
        }
        format!("{prefix}{leaf}{}", suffix.chars().rev().collect::<String>())
    }

    #[test]
    fn container_depth_counts_empty_and_mixed_containers_but_not_scalars() {
        for leaf in ["0", "null", "\"unicode \\ud83d\\ude80\""] {
            let raw = mixed_json(MAX_JSON_DEPTH, leaf);
            let value = parse(&raw).unwrap();
            assert!(value.validate_kind(JsonInputKind::JsonlRecord).is_ok());
            assert!(value.try_to_compact_string().is_ok());
            let err = parse(&mixed_json(MAX_JSON_DEPTH + 1, leaf)).unwrap_err();
            assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
            assert!(err
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "depth_exceeded"));
            assert!(err
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "128"));
        }
        assert!(parse(&mixed_json(MAX_JSON_DEPTH - 1, "{}")).is_ok());
        assert!(parse(&mixed_json(MAX_JSON_DEPTH, "[]")).is_err());
        assert!(parse(&"[".repeat(30_000)).is_err());
        assert!(parse(&format!("{}!{}", "[".repeat(30_000), "]".repeat(30_000))).is_err());
        assert!(parse(&mixed_json(MAX_JSON_DEPTH, "!")).is_err());
    }

    #[test]
    fn wide_shallow_json_and_pretty_output_remain_compatible() {
        let obj = (0..20_000)
            .map(|n| {
                (
                    format!("key-{n:05}"),
                    JsonValue::Array(vec![JsonValue::Number(n as f64)]),
                )
            })
            .collect();
        let value = JsonValue::Object(obj);
        let raw = value.try_to_compact_string().unwrap();
        assert_eq!(parse(&raw).unwrap(), value);
        let value = parse(r#"{"a":[1,{"b":true}],"empty":{}}"#).unwrap();
        assert_eq!(
            value.to_pretty_string(),
            "{\n  \"a\": [\n    1,\n    {\n      \"b\": true\n    }\n  ],\n  \"empty\": {}\n}"
        );
        assert_eq!(value.clone(), value);
        assert_ne!(JsonValue::Number(f64::NAN), JsonValue::Number(f64::NAN));
    }

    #[test]
    fn fixed_seed_json_syntax_smoke_never_panics() {
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let alphabet = b"{}[],:\"\\ntf0123456789-+e abc";
        for _ in 0..2000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let len = seed as usize % 512;
            let mut raw = String::new();
            for _ in 0..len {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                raw.push(alphabet[seed as usize % alphabet.len()] as char);
            }
            if let Ok(value) = parse(&raw) {
                let canonical = value.try_to_compact_string().unwrap();
                assert_eq!(parse(&canonical).unwrap(), value);
            }
        }
    }

    #[test]
    fn json_round_trips_objects_arrays_and_unicode() {
        let value = parse(r#"{"lang":"rust","tags":["db",2],"ok":true,"u":"存储"}"#).unwrap();
        assert_eq!(value.get("lang").and_then(JsonValue::as_str), Some("rust"));
        let encoded = value.to_compact_string();
        assert_eq!(parse(&encoded).unwrap(), value);
    }

    #[test]
    fn rejects_non_finite_json_numbers() {
        assert!(parse(r#"{"too_big":1e999}"#).is_err());
        assert!(JsonValue::Number(f64::NAN).try_to_compact_string().is_err());
    }

    #[test]
    fn rejects_duplicate_object_keys() {
        let err = parse(r#"{"metadata":{"lang":"rust","lang":"go"}}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(err.message(), "duplicate object key");
    }

    #[test]
    fn rejects_integer_precision_loss_without_rejecting_exact_large_values() {
        let err = parse(r#"{"metadata":{"id":9007199254740993}}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("encode it as a string"));
        assert!(parse(r#"{"metadata":{"id":9007199254740991}}"#).is_ok());
        for raw in [
            "9007199254740992",
            "9007199254740994",
            "-9007199254740992",
            "9007199254740992.00",
            "90071992547409920e-1",
            "9.007199254740992E+15",
            "9223372036854775808",
            "18446744073709551616",
            "1267650600228229401496703205376",
        ] {
            let value = parse(raw).unwrap();
            assert_eq!(parse(&value.to_compact_string()).unwrap(), value, "{raw}");
        }
        for raw in [
            "-9007199254740993",
            "9007199254740993.0000",
            "90071992547409930e-1",
            "9.007199254740993e15",
            "18446744073709551615",
            "1267650600228229401496703205377",
            "1e309",
            "1e999999999999999999999",
        ] {
            assert!(parse(raw).is_err(), "{raw}");
        }
        assert!(parse("9007199254740993.5").is_ok());
        assert!(parse("1e-999999999999999999999").is_ok());
        assert_eq!(parse("-0e999999").unwrap(), JsonValue::Number(0.0));
        assert_eq!(
            parse(&format!("{:.0}", f64::MAX)).unwrap(),
            JsonValue::Number(f64::MAX)
        );
    }

    #[test]
    fn integer_precision_matches_independent_u128_bit_boundaries() {
        for exponent in 53..128 {
            let base = 1u128 << exponent;
            for offset in 0..8 {
                let value = base + offset;
                let exact = 128 - value.leading_zeros() - value.trailing_zeros() <= 53;
                for sign in ["", "-"] {
                    for suffix in ["", ".0", "00e-2"] {
                        let raw = format!("{sign}{value}{suffix}");
                        assert_eq!(parse(&raw).is_ok(), exact, "{raw}");
                    }
                }
            }
        }
    }

    #[test]
    fn canonical_bytes_have_independent_utf8_key_and_escaping_golden() {
        let input =
            "{\"\u{10000}\":1,\"\u{e000}\":2,\"a\":[-0,3],\"s\":\"a\\\"\\\\\\n\\t\\u0000\"}";
        let expected =
            "{\"a\":[0,3],\"s\":\"a\\\"\\\\\\n\\t\\u0000\",\"\u{e000}\":2,\"\u{10000}\":1}";
        assert_eq!(parse(input).unwrap().to_compact_string(), expected);
    }

    #[test]
    fn unicode_surrogate_pairs_decode_to_scalar_values_and_duplicate_keys() {
        for (raw, expected) in [
            (r#""\ud800\udc00""#, "\u{10000}"),
            (r#""\uDBFF\uDFFF""#, "\u{10ffff}"),
            (r#""\ud83d\ude80""#, "\u{1f680}"),
        ] {
            let value = parse(raw).unwrap();
            assert_eq!(value, JsonValue::String(expected.into()));
            assert_eq!(parse(&value.to_compact_string()).unwrap(), value);
        }
        for raw in [
            r#""\ud800""#,
            r#""\udc00""#,
            r#""\ud800\u0041""#,
            r#""\ud800\ud800""#,
            r#""\ud800x""#,
            r#""\ud800\u""#,
        ] {
            assert!(parse(raw).is_err(), "{raw}");
        }
        let raw = "{\"\u{10000}\":1,\"\\ud800\\udc00\":2}";
        assert!(parse(raw)
            .unwrap_err()
            .message()
            .contains("duplicate object key"));
    }

    #[test]
    fn usize_conversion_rejects_exclusive_bound_instead_of_saturating() {
        let exclusive_max = (usize::MAX as u128 + 1) as f64;
        let largest_integer = if usize::BITS <= 53 {
            exclusive_max - 1.0
        } else {
            f64::from_bits(exclusive_max.to_bits() - 1)
        };
        let value = object_from_pairs([("count", JsonValue::Number(largest_integer))]);
        assert_eq!(
            required_usize(value.as_object().unwrap(), "count").unwrap(),
            largest_integer as usize
        );
        for invalid in [exclusive_max, exclusive_max * 2.0, -1.0, 0.5, f64::NAN] {
            let value = object_from_pairs([("count", JsonValue::Number(invalid))]);
            assert_eq!(
                required_usize(value.as_object().unwrap(), "count")
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidJsonlSchema
            );
        }
    }

    #[test]
    fn long_unicode_strings_and_keys_preserve_every_scalar() {
        let text = "\u{10000}\u{e000}\u{20ac}e".repeat(16_384);
        let input = format!("{{\"{text}\":\"{text}\"}}");
        let value = parse(&input).unwrap();
        assert_eq!(
            value.get(&text).and_then(JsonValue::as_str),
            Some(text.as_str())
        );
        assert_eq!(value.to_compact_string(), input);
    }

    #[test]
    fn canonicalizes_negative_zero_numbers() {
        assert_eq!(JsonValue::Number(-0.0).to_compact_string(), "0");
        assert_eq!(parse("-0").unwrap().to_compact_string(), "0");
        assert_eq!(
            parse(r#"{"value":-0}"#).unwrap().to_compact_string(),
            r#"{"value":0}"#
        );
    }

    #[test]
    fn retained_footprint_uses_scratch_without_readmitting_the_resident_tree() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut text = String::with_capacity(65536 + seed);
            text.push_str("retained");
            let value =
                object_from_pairs([("payload", JsonValue::Array(vec![JsonValue::String(text)]))]);
            let retained = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            let expected = retained.reserved_bytes();
            let mut occupied = context.workspace_reservation();
            let bytes = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - expected - 4096;
            occupied.grow_to(bytes, bytes, "operation_scratch").unwrap();
            assert_eq!(
                value
                    .retained_bytes_in(JsonInputKind::StoredJson, &context)
                    .unwrap(),
                expected
            );
            assert!(value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .is_err());
            drop(occupied);
            drop(value);
            drop(retained);
            context
                .workspace_reservation()
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
    }
}
