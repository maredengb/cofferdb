use std::fmt::Write;

use super::*;
use crate::json::JsonTextWriter;

impl FilterExpr {
    // The compiled tree drives evaluation, while the caller supplies the original
    // JSON to show in the explanation. A shorthand equality expression, for
    // example, should retain its original spelling in the returned value.
    pub(crate) fn evaluate_for_chunk_in(
        &self,
        catalog: &Catalog,
        chunk: &Chunk,
        expression: JsonValue,
        context: &JsonOperationContext,
    ) -> Result<(FilterExplain, JsonReservation)> {
        // Resolve document and chunk fields with the same lookup used by the
        // boolean eligibility check. This path additionally visits individual
        // leaves to retain a reason for each condition.
        let (eval, work) =
            self.evaluate_lookup_in(&|field| resolve_field(catalog, chunk, field), context)?;
        Ok((
            FilterExplain {
                matched: eval.matched,
                expression,
                reasons: eval.reasons,
            },
            // The returned reservation accompanies the newly allocated reason
            // strings and their array. The expression moves in from the caller
            // with its existing ownership and is never copied by this wrapper.
            work,
        ))
    }

    fn evaluate_lookup_in<'a>(
        &self,
        lookup: &impl Fn(&str) -> Option<&'a JsonValue>,
        context: &JsonOperationContext,
    ) -> Result<(Eval, JsonReservation)> {
        // Keep the temporary traversal stack and the escaping explanation under
        // separate reservations. The stack only owns child iterators and can
        // release its allocation as soon as evaluation finishes.
        let kind = JsonInputKind::Filter;
        let mut scratch = context.reservation();
        let mut parents = Vec::<std::slice::Iter<'_, Self>>::new();
        let mut work = context.reservation();
        let mut reasons = Vec::new();
        let mut matched = true;
        let mut next = Some(self);
        while let Some(node) = next {
            // Walk nested conjunctions with an explicit iterator stack. This keeps
            // deep expression traversal from consuming the native call stack. An
            // empty conjunction opens an empty iterator and leaves matched true if
            // nothing else failed.
            if let Self::And(nodes) = node {
                scratch.reserve_vec(&mut parents, 1, kind)?;
                parents.push(nodes.iter());
            } else {
                // Each leaf supplies a stable operator label and a borrowed
                // expected value. Membership renders its list through a separate
                // branch, so the common path carries no expected scalar for that
                // variant.
                let (field, op, expected) = match node {
                    Self::Eq(field, value) => (field, "==", Some(value)),
                    Self::Ne(field, value) => (field, "!=", Some(value)),
                    Self::Gt(field, value) => (field, ">", Some(value)),
                    Self::Gte(field, value) => (field, ">=", Some(value)),
                    Self::Lt(field, value) => (field, "<", Some(value)),
                    Self::Lte(field, value) => (field, "<=", Some(value)),
                    Self::Contains(field, value) => (field, "contains", Some(value)),
                    Self::In(field, _) => (field, "in", None),
                    Self::And(_) => unreachable!(),
                };
                // Use the same borrowed lookup and comparison helper as the
                // boolean evaluator. Diagnostic formatting uses that already
                // selected value, so an absent key remains distinct from a JSON
                // null.
                let actual = lookup(field);
                let leaf_matched = node.matches_value(actual);
                // Unlike boolean filtering, an explanation includes every leaf in
                // expression order, even after the first failed conjunct.
                matched &= leaf_matched;
                // Reserve the new array slot before formatting the reason. If
                // string construction fails, the unfinished reason stays in the
                // writer and the caller receives the error without an incomplete
                // result.
                work.reserve_vec(&mut reasons, 1, kind)?;
                let mut out = JsonTextWriter::new(context, kind);
                let written = (|| {
                    write!(out, "{field} {op} ")?;
                    if let Self::In(_, values) = node {
                        // A successful membership reason shows the expected
                        // choices in their original order. A failed membership
                        // reason uses a compact placeholder for the list and
                        // proceeds to the same failed-condition suffix as other
                        // leaves.
                        if leaf_matched {
                            out.write_char('[')?;
                            for (index, value) in values.iter().enumerate() {
                                if index != 0 {
                                    out.write_str(", ")?;
                                }
                                write_label(&mut out, value)?;
                            }
                            return out.write_char(']');
                        }
                        out.write_str("[...]")?;
                    } else {
                        write_label(&mut out, expected.unwrap())?;
                    }
                    // Describe the actual slot on a failed comparison, including
                    // the literal word missing for a lookup that yielded no value.
                    // The inequality reason deliberately omits a separate
                    // actual-value suffix.
                    if !leaf_matched {
                        out.write_str(" failed")?;
                        if !matches!(node, Self::Ne(_, _)) {
                            out.write_str(": actual ")?;
                            match actual {
                                Some(value) => write_label(&mut out, value)?,
                                None => out.write_str("missing")?,
                            }
                        }
                    }
                    Ok(())
                })();
                // Finish the checked writer before inserting its string. The
                // returned reservation covers that string and transfers into the
                // same work object that keeps the reasons vector alive.
                let (reason, reserved) = out.finish(written)?;
                reasons.push(reason);
                work.merge(reserved);
            }
            // Resume the deepest open conjunction first. Once its iterator is
            // exhausted, pop it and resume its parent. This preserves the
            // left-to-right leaf order without a second walk.
            next = loop {
                let Some(parent) = parents.last_mut() else {
                    break None;
                };
                if let Some(node) = parent.next() {
                    break Some(node);
                }
                parents.pop();
            };
        }
        // Only the boolean outcome, reason strings and their reservation cross
        // this boundary. The traversal stack drops here even if the input
        // expression remains available for more candidates.
        Ok((Eval { matched, reasons }, work))
    }
}

fn write_label(out: &mut JsonTextWriter, value: &JsonValue) -> std::fmt::Result {
    // These labels are human-readable fragments of explanation strings. Strings
    // are shown directly, while compound actual values are formatted using the
    // validated JSON writer so nested metadata is rendered through the same
    // budgets.
    match value {
        JsonValue::String(value) => out.write_str(value),
        JsonValue::Null => out.write_str("null"),
        JsonValue::Bool(value) => out.write_str(if *value { "true" } else { "false" }),
        JsonValue::Number(value) => {
            // Integral numbers are shown without a decimal suffix. Retain the
            // ordinary formatter for fractional values so reasons distinguish the
            // operands a caller supplied.
            if value.fract() == 0.0 {
                write!(out, "{value:.0}")
            } else {
                write!(out, "{value}")
            }
        }
        value => out.write_validated_value(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{object_from_pairs, JsonProcessingPolicyV1};

    #[test]
    fn streamed_trace_matches_labels_for_scalars_and_nested_actual_values() {
        let expected = [
            JsonValue::Null,
            JsonValue::Bool(true),
            JsonValue::Number(-0.0),
            JsonValue::Number(1.25),
            JsonValue::String("\u{4e2d}\n\\\"".into()),
        ];
        for seed in 0..100 {
            let mut actual =
                object_from_pairs([("value", expected[seed % expected.len()].clone())]);
            for depth in 1..=seed % 127 {
                actual = if depth % 2 == 0 {
                    object_from_pairs([("nested", actual)])
                } else {
                    JsonValue::Array(vec![actual])
                };
            }
            for expected in &expected {
                let filter = FilterExpr::And(vec![
                    FilterExpr::Eq("chunk.x".into(), expected.clone()),
                    FilterExpr::In("chunk.x".into(), vec![expected.clone()]),
                ]);
                let lookup = |_: &str| Some(&actual);
                let legacy = filter.evaluate_lookup(&lookup);
                let context = JsonOperationContext::standalone();
                let (actual, work) = filter.evaluate_lookup_in(&lookup, &context).unwrap();
                assert_eq!(actual.matched, legacy.matched);
                assert_eq!(actual.reasons, legacy.reasons);
                let expected_bytes = actual.reasons.capacity() * std::mem::size_of::<String>()
                    + actual.reasons.iter().map(String::capacity).sum::<usize>();
                assert_eq!(work.reserved_bytes(), expected_bytes);
            }
        }
    }

    #[test]
    fn streamed_in_trace_avoids_per_literal_copies_and_releases_failed_output() {
        for (count, accepted) in [(200_000, true), (600_000, false)] {
            let context = JsonOperationContext::standalone();
            let filter = FilterExpr::In("chunk.x".into(), vec![JsonValue::Null; count]);
            let input = filter
                .reserve_json_work_in(&context, JsonInputKind::Filter)
                .unwrap();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - input.reserved_bytes()
                    - 3 * 1024 * 1024,
                JsonInputKind::Filter,
            )
            .unwrap();
            let actual = JsonValue::Null;
            let result = filter.evaluate_lookup_in(&|_| Some(&actual), &context);
            assert_eq!(result.is_ok(), accepted);
            match result {
                Ok((eval, work)) => {
                    assert!(eval.matched);
                    assert_eq!(eval.reasons.len(), 1);
                    assert!(eval.reasons[0].starts_with("chunk.x in [null, null"));
                    assert!(eval.reasons[0].ends_with("null]"));
                    assert_eq!(
                        eval.reasons[0].len(),
                        "chunk.x in [".len() + 4 * count + 2 * (count - 1) + 1
                    );
                    drop(eval);
                    drop(work);
                }
                Err(error) => {
                    assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "json_limit_reason"
                            && value == "work_budget_exceeded"));
                }
            }
            context
                .reservation()
                .grow(3 * 1024 * 1024, JsonInputKind::Filter)
                .unwrap();
            drop(busy);
            drop(input);
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::Filter,
                )
                .unwrap();
        }
    }

    #[test]
    fn trace_uses_iterative_and_walk_on_a_small_stack() {
        let mut filter = FilterExpr::Eq("chunk.x".into(), JsonValue::Number(-0.0));
        for _ in 0..63 {
            filter = FilterExpr::And(vec![filter]);
        }
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || {
                let context = JsonOperationContext::standalone();
                let (eval, _) = filter.evaluate_lookup_in(&|_| None, &context).unwrap();
                assert!(!eval.matched);
                assert_eq!(eval.reasons, ["chunk.x == -0 failed: actual missing"]);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
