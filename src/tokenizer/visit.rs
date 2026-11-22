use super::{
    normalized::{lowercase_layout, normalize, Token},
    symbol_parts, symbols,
};
use crate::json::{
    is_memory_budget_error, JsonOperationContext, ReservedVec, WorkspaceReservation,
};
use crate::Result;
use std::borrow::Cow;

struct SymbolTokens<'a> {
    // Normalized tokens unique within the current symbol, in first-encounter order,
    // with the array slots charged to its reservation.
    values: ReservedVec<Cow<'a, str>>,
    // Credit for separately owned lowercase strings in values. Entries borrowing
    // slices of the source symbol have no payload charge.
    payload: WorkspaceReservation,
}

pub(crate) fn visit_tokens_in(
    input: &str,
    context: &JsonOperationContext,
    mut visit: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    #[cfg(test)]
    super::TOKEN_VISITS.with(|count| count.set(count.get() + 1));
    // Deduplicate within each source symbol. The same normalized word in two
    // separate symbols must generate two visits so its lexical term frequency is
    // preserved.
    for symbol in symbols(input) {
        let mut parts = symbol_parts(symbol);
        let first = parts.next();
        // A symbol with no components, or one component identical to the original,
        // can produce at most one normalized spelling. Skip allocating a
        // uniqueness table in that common case.
        if first.is_none() || first == Some(symbol) {
            if let Some(token) = normalize(symbol, context)? {
                visit(token.value.as_ref())?;
            }
            continue;
        }
        // Collect a symbol's unique tokens before invoking callbacks. A failed
        // preparation can restart by scanning the original slices because no token
        // from this symbol has been emitted yet.
        let unique = match build_symbol(symbol, context) {
            Ok(unique) => unique,
            // Only a workspace-budget failure selects the rescan strategy. Other
            // preparation failures, including allocator or capacity-check errors,
            // reach the caller with their original causes.
            Err(error) if is_memory_budget_error(&error) => {
                drop(error);
                // This visitor has no external state reclamation hook. Its
                // fallback can trade repeated normalization for lower live memory,
                // but cannot retry an error by freeing a consumer's allocations.
                visit_without_array(
                    symbol,
                    context,
                    &mut (),
                    &mut |_, token| visit(token),
                    &mut |_| Ok(false),
                )?;
                continue;
            }
            Err(error) => return Err(error),
        };
        // The prepared table retains each spelling and its workspace grant through
        // the callback. A callback error propagates immediately because replay
        // would duplicate already observed tokens.
        for value in unique.values.as_slice() {
            visit(value.as_ref())?;
        }
    }
    Ok(())
}

// Stateful consumers may allocate in callbacks. Keep only the current payload,
// leaving required callback workspace available instead of retaining a symbol set.
pub(crate) fn visit_tokens_in_reclaiming<S>(
    input: &str,
    context: &JsonOperationContext,
    state: &mut S,
    mut visit: impl FnMut(&mut S, &str) -> Result<()>,
    mut reclaim: impl FnMut(&mut S) -> Result<bool>,
) -> Result<()> {
    #[cfg(test)]
    super::TOKEN_VISITS.with(|count| count.set(count.get() + 1));
    for symbol in symbols(input) {
        let mut parts = symbol_parts(symbol);
        let first = parts.next();
        // The simple-symbol path can still need workspace to lowercase the token.
        // Let the consumer free optional state before retrying that preparation,
        // with no side effects from a token visit yet.
        if first.is_none() || first == Some(symbol) {
            if let Some(token) =
                prepare_reclaiming(state, &mut reclaim, || normalize(symbol, context))?
            {
                visit(state, token.value.as_ref())?;
            }
        } else {
            // For a compound symbol, rescan the source slices even when memory is
            // currently available. A callback can retain new state, so an eagerly
            // held set of normalized strings would compete with it.
            visit_without_array(symbol, context, state, &mut visit, &mut reclaim)?;
        }
    }
    Ok(())
}

fn prepare_reclaiming<T, S>(
    state: &mut S,
    reclaim: &mut impl FnMut(&mut S) -> Result<bool>,
    mut prepare: impl FnMut() -> Result<T>,
) -> Result<T> {
    // Retry only preparation after the consumer has released optional work. Already
    // executed callbacks may have mutated the caller's state, so they are never
    // replayed by this helper.
    match prepare() {
        Err(error) if is_memory_budget_error(&error) => {
            // A false return means the consumer released no optional space.
            // Preserve the original budget error in that case. An error from the
            // reclamation itself propagates independently.
            if !reclaim(state)? {
                return Err(error);
            }
            drop(error);
            // The successful reclamation authorizes one new preparation attempt. A
            // second refusal is final here, preventing a nonproductive reclaim
            // loop from hiding the budget failure.
            prepare()
        }
        result => result,
    }
}

fn build_symbol<'a>(symbol: &'a str, context: &JsonOperationContext) -> Result<SymbolTokens<'a>> {
    // The staging array and separately allocated lowercase strings have different
    // owners. A failed symbol build drops both before the caller starts over using
    // the source slices.
    let mut unique = SymbolTokens {
        values: ReservedVec::with_capacity(0, context, "operation_scratch")?,
        payload: context.workspace_reservation(),
    };
    // Offer the whole identifier first, then its components in source order.
    // Deduplication keeps the first occurrence of each normalized spelling.
    for raw in std::iter::once(symbol).chain(symbol_parts(symbol)) {
        let Some(token) = normalize(raw, context)? else {
            continue;
        };
        // Test the normalized spelling, so differently cased components compete
        // for the same first occurrence. A duplicate's temporary lowercase buffer
        // is released on this path.
        if unique.values.as_slice().contains(&token.value) {
            continue;
        }
        let Token { value, work } = token;
        // Adding a distinct token can grow the slot array while its string still
        // holds a separate grant. Only merge that grant once the slot insertion
        // has succeeded.
        unique.values.try_push(value)?;
        unique.payload.merge(work);
    }
    Ok(unique)
}

// Under pressure, rescan source slices instead of retaining an array or payload set.
// Precheck normalization before callbacks; later allocations remain independently
// admitted. No already-emitted token is retried, even if a callback consumes credit.
fn visit_without_array<S>(
    symbol: &str,
    context: &JsonOperationContext,
    state: &mut S,
    visit: &mut impl FnMut(&mut S, &str) -> Result<()>,
    reclaim: &mut impl FnMut(&mut S) -> Result<bool>,
) -> Result<()> {
    // Probe the whole symbol and first-occurrence checks before issuing token
    // callbacks. A probe may ask the consumer to reclaim workspace. Later
    // callbacks can still consume credit after this pass.
    drop(prepare_reclaiming(state, reclaim, || {
        normalize(symbol, context)
    })?);
    for (index, raw) in symbol_parts(symbol).enumerate() {
        prepare_reclaiming(state, reclaim, || {
            is_first_part(symbol, index, raw, context)
        })?;
    }
    // The probes leave no reusable tokens or eligibility bitmap. Recreate the
    // whole token for the first callback, then repeat the part checks in their
    // original order.
    if let Some(token) = prepare_reclaiming(state, reclaim, || normalize(symbol, context))? {
        visit(state, token.value.as_ref())?;
    }
    for (index, raw) in symbol_parts(symbol).enumerate() {
        // An earlier callback can have consumed workspace. Reclaim and retry only
        // the duplicate check or normalization that is still pending, never an
        // earlier successful callback.
        if prepare_reclaiming(state, reclaim, || {
            is_first_part(symbol, index, raw, context)
        })? {
            if let Some(token) = prepare_reclaiming(state, reclaim, || normalize(raw, context))? {
                visit(state, token.value.as_ref())?;
            }
        }
    }
    Ok(())
}

fn is_first_part(
    symbol: &str,
    index: usize,
    raw: &str,
    context: &JsonOperationContext,
) -> Result<bool> {
    // A stop-word component has no first occurrence to emit. Test it with the same
    // normalizer as the buffered tokenizer, including its Unicode casing rules.
    let Some(token) = normalize(raw, context)? else {
        return Ok(false);
    };
    // Release this trial token before comparing any pair. Keeping it while
    // constructing both normalized operands would charge a third copy during a
    // low-memory fallback.
    drop(token);
    // Whole has nonempty output outside each proper part; compare only parts.
    for previous in symbol_parts(symbol).take(index) {
        // Identical raw text is already a duplicate. Two ASCII spellings can be
        // compared case-insensitively without allocating, while a mixed or Unicode
        // pair can require full string normalization.
        if raw == previous || (raw.is_ascii() && previous.is_ascii()) {
            if raw.eq_ignore_ascii_case(previous) {
                return Ok(false);
            }
            continue;
        }
        let (left_capacity, left_peak) = lowercase_layout(raw);
        let (right_capacity, right_peak) = lowercase_layout(previous);
        // Normalize the operand with the larger transient overhead first. Its peak
        // occurs while no other result buffer is retained, reducing the
        // simultaneous space needed to compare the pair.
        let (first, second) = if left_peak - left_capacity >= right_peak - right_capacity {
            (raw, previous)
        } else {
            (previous, raw)
        };
        let first = normalize(first, context)?;
        let second = normalize(second, context)?;
        // Only two retained token values can suppress the current component. A
        // preceding spelling that normalizes to a stop word has no emitted
        // occurrence to match.
        if matches!((&first, &second), (Some(left), Some(right)) if left.value == right.value) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use crate::{Error, ErrorCode};
    use std::sync::Arc;

    #[test]
    fn budgeted_visits_match_frozen_unicode_and_code_symbol_order() {
        let alphabet = [
            'a',
            'Z',
            '_',
            '/',
            '.',
            '\u{130}',
            '\u{3a3}',
            '\u{3c3}',
            '\u{c9}',
            '\u{e9}',
            '\u{307}',
            '\u{10400}',
            '\u{10428}',
            '\u{4e2d}',
        ];
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        for seed in 0..100u64 {
            let mut state = seed + 1;
            let mut input = String::from("HTTPServer __the_and__ fooFoo \u{39f}\u{3a3}Value ");
            for _ in 0..seed as usize * 17 + 1 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                input.push(alphabet[(state >> 32) as usize % alphabet.len()]);
            }
            let expected = super::super::tests::token_reference(&input);
            let mut position = 0;
            visit_tokens_in(&input, &context, |token| {
                assert_eq!(token, expected[position], "seed {seed}");
                position += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(position, expected.len());
            position = 0;
            visit_tokens_in_reclaiming(
                &input,
                &context,
                &mut position,
                |position, token| {
                    assert_eq!(token, expected[*position], "reclaiming seed {seed}");
                    *position += 1;
                    Ok(())
                },
                |_| panic!("ample workspace must not need reclamation"),
            )
            .unwrap();
            assert_eq!(position, expected.len());
            assert_eq!(context.available_workspace_bytes(), available);
        }
    }

    #[test]
    fn borrowed_visits_need_no_payload_and_repeated_parts_only_keep_unique_slots() {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let (result, allocated) = crate::test_alloc::allocations(|| {
            let mut count = 0;
            visit_tokens_in("rust 42 caf\u{e9} \u{10428} THE", &context, |token| {
                assert_eq!(token, ["rust", "42", "caf\u{e9}", "\u{10428}"][count]);
                count += 1;
                Ok(())
            })?;
            assert_eq!(count, 4);
            Ok::<_, Error>(())
        });
        result.unwrap();
        assert_eq!(allocated, 0);

        let bytes = 3 * std::mem::size_of::<Cow<'_, str>>();
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(bytes, bytes, bytes)));
        // Initialize platform mutex storage outside the tokenizer measurement.
        assert_eq!(context.available_workspace_bytes(), bytes);
        for count in [1usize, 100, 10000] {
            let input = "foo_bar_".repeat(count);
            let (result, allocated) = crate::test_alloc::allocation_bytes(|| {
                let mut position = 0;
                visit_tokens_in(&input, &context, |token| {
                    assert_eq!(token, [input.as_str(), "foo", "bar"][position]);
                    assert_eq!(context.available_workspace_bytes(), 0);
                    position += 1;
                    Ok(())
                })?;
                assert_eq!(position, 3);
                Ok::<_, Error>(())
            });
            result.unwrap();
            assert_eq!(allocated, bytes as u128);
            assert_eq!(context.available_workspace_bytes(), bytes);
        }
    }

    #[test]
    fn split_symbol_pressure_fallback_is_monotonic_and_releases_duplicate_payloads() {
        let slots = std::mem::size_of::<Cow<'_, str>>();
        let minimum = "foo_bar".len();
        for parent in [false, true] {
            for limit in minimum - 1..=200 {
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * 200 } else { limit },
                    2 * 200,
                    if parent { limit } else { 2 * 200 },
                )));
                let mut visits = 0;
                let result = visit_tokens_in("FOO_BAR", &context, |token| {
                    assert_eq!(token, ["foo_bar", "foo", "bar"][visits]);
                    let resident = if (85..=102).contains(&limit) {
                        3 * slots + 13
                    } else if limit >= 109 {
                        4 * slots + 13
                    } else {
                        token.len()
                    };
                    assert_eq!(context.available_workspace_bytes(), limit - resident);
                    visits += 1;
                    Ok(())
                });
                if limit < minimum {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", minimum.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                    }
                    assert_eq!(visits, 0);
                } else {
                    result.unwrap();
                    assert_eq!(visits, 3);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
        let limit = 512;
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)));
        let mut position = 0;
        visit_tokens_in("fooFoo Foo_foo", &context, |token| {
            assert_eq!(token, ["foofoo", "foo", "foo_foo", "foo"][position]);
            let payload = if position < 2 { 6 } else { 10 };
            assert_eq!(
                context.available_workspace_bytes(),
                limit - 4 * slots - payload
            );
            position += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(position, 4);
        assert_eq!(context.available_workspace_bytes(), limit);
    }

    #[test]
    fn owned_slot_growth_refusal_releases_both_payload_owners_before_fallback() {
        let minimum = 9 * std::mem::size_of::<Cow<'_, str>>() + 10;
        for parent in [false, true] {
            let limit = minimum - 1;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                if parent { 2 * minimum } else { limit },
                0,
                if parent { limit } else { 2 * minimum },
            )));
            let error = build_symbol("a_b_c_d_E", &context).err().unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .contains(&("requested_bytes".into(), minimum.to_string())));
            assert!(error
                .detail()
                .contains(&("limit_bytes".into(), limit.to_string())));
            assert_eq!(context.available_workspace_bytes(), limit);
            let mut position = 0;
            visit_tokens_in("a_b_c_d_E", &context, |token| {
                assert_eq!(token, ["a_b_c_d_e", "b", "c", "d", "e"][position]);
                position += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(position, 5);
            assert_eq!(context.available_workspace_bytes(), limit);
        }
    }

    #[test]
    fn unicode_pair_order_fits_independent_eighteen_byte_peak() {
        let minimum = 18;
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (minimum as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * minimum } else { limit },
                    0,
                    if parent { limit } else { 2 * minimum },
                )));
                let mut position = 0;
                let result = visit_tokens_in("\u{023a}X_\u{0130}", &context, |token| {
                    assert_eq!(
                        token,
                        ["\u{2c65}x_i\u{0307}", "\u{2c65}x", "i\u{0307}"][position]
                    );
                    let resident = if position == 0 { 12 } else { 8 };
                    assert_eq!(context.available_workspace_bytes(), limit - resident);
                    position += 1;
                    Ok(())
                });
                if delta < 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), minimum.to_string())));
                    assert!(error
                        .detail()
                        .contains(&("limit_bytes".into(), limit.to_string())));
                    assert_eq!(position, 0);
                } else {
                    result.unwrap();
                    assert_eq!(position, 3);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn unicode_pressure_fallback_preserves_order_and_budget_monotonicity() {
        for input in [
            "\u{0130}_\u{212a}".into(),
            "\u{212a}_K_k".into(),
            "\u{39f}\u{3a3}_\u{39f}\u{3c3}_\u{3bf}\u{3c3}".into(),
            "\u{39f}\u{0345}\u{3a3}_\u{39f}\u{3a3}".into(),
            format!("{}_{}_tail", "\u{0130}".repeat(9), "\u{0130}".repeat(9)),
            format!(
                "begin_{}_{}_end",
                "\u{0130}".repeat(33),
                "\u{212a}".repeat(33)
            ),
        ] {
            let expected = super::super::tests::token_reference(&input);
            let maximum = input.len() * 8 + 512;
            for parent in [false, true] {
                let mut succeeded = false;
                for limit in 0..=maximum {
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { 2 * maximum } else { limit },
                        0,
                        if parent { limit } else { 2 * maximum },
                    )));
                    let mut position = 0;
                    let result = visit_tokens_in(&input, &context, |token| {
                        assert_eq!(token, expected[position], "{input:?}, limit={limit}");
                        position += 1;
                        Ok(())
                    });
                    if let Err(error) = result {
                        assert!(
                            !succeeded,
                            "nonmonotonic refusal: {input:?}, limit={limit}, {error:?}"
                        );
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert!(error
                            .detail()
                            .contains(&("limit_bytes".into(), limit.to_string())));
                        assert_eq!(position, 0);
                    } else {
                        succeeded = true;
                        assert_eq!(position, expected.len());
                    }
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
                assert!(succeeded);
            }
        }
    }

    #[test]
    fn reclaiming_visits_retry_pure_unicode_work_without_replaying_callbacks() {
        let limit = 64;
        let input = "rust \u{23a}X_\u{130}";
        let expected = super::super::tests::token_reference(input);
        for parent in [false, true] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                if parent { 2 * limit } else { limit },
                0,
                if parent { limit } else { 2 * limit },
            )));
            let mut state = (context.workspace_reservation(), 0usize, 0usize);
            state.0.grow_to(47, 47, "operation_scratch").unwrap();
            visit_tokens_in_reclaiming(
                input,
                &context,
                &mut state,
                |(held, calls, _), token| {
                    assert_eq!(token, expected[*calls]);
                    if *calls == 1 {
                        assert_eq!(context.available_workspace_bytes(), limit - 12);
                        held.grow_to(47, 47, "operation_scratch")?;
                    }
                    *calls += 1;
                    Ok(())
                },
                |(held, calls, reclaims)| {
                    assert_eq!(*calls, [1, 3][*reclaims]);
                    // Both tokens from a failed comparison are already destroyed.
                    assert_eq!(context.available_workspace_bytes(), limit - 47);
                    held.retain(0);
                    *reclaims += 1;
                    Ok(true)
                },
            )
            .unwrap();
            assert_eq!((state.1, state.2), (expected.len(), 2));
            assert_eq!(context.available_workspace_bytes(), limit);
        }
    }

    #[test]
    fn reclaiming_visits_preserve_refusal_reclaim_failure_and_second_attempt_errors() {
        let limit = 64;
        for mode in 0..3 {
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, 0, limit)));
            let mut state = (context.workspace_reservation(), 0usize, 0usize);
            state.0.grow_to(47, 47, "operation_scratch").unwrap();
            let error = visit_tokens_in_reclaiming(
                "rust \u{23a}X_\u{130}",
                &context,
                &mut state,
                |(_, calls, _), token| {
                    assert_eq!(token, "rust");
                    *calls += 1;
                    Ok(())
                },
                |(held, _, reclaims)| {
                    *reclaims += 1;
                    assert_eq!(context.available_workspace_bytes(), limit - 47);
                    match mode {
                        0 => Ok(false),
                        1 => Err(Error::new(ErrorCode::IoError, "reclaim sentinel")),
                        _ => {
                            held.retain(0);
                            // Model new live work arriving between release and retry.
                            held.grow_to(48, 48, "operation_scratch")?;
                            Ok(true)
                        }
                    }
                },
            )
            .unwrap_err();
            assert_eq!((state.1, state.2), (1, 1));
            if mode == 1 {
                assert_eq!(error.code(), ErrorCode::IoError);
                assert_eq!(error.message(), "reclaim sentinel");
            } else {
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".into()),
                    ("operation", "operation_scratch".into()),
                    ("limit_bytes", limit.to_string()),
                    (
                        "requested_bytes",
                        if mode == 0 { "65" } else { "66" }.into(),
                    ),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                }
            }
            assert_eq!(
                context.available_workspace_bytes(),
                limit - state.0.reserved_bytes()
            );
            drop(state);
            assert_eq!(context.available_workspace_bytes(), limit);
        }
    }

    #[test]
    fn reclaiming_native_normalization_failure_never_reclaims_or_replays_prior_tokens() {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut calls = 0;
        let error = crate::test_alloc::fail_byte_allocation_once(7, || {
            visit_tokens_in_reclaiming(
                "rust PROJECT",
                &context,
                &mut calls,
                |calls, token| {
                    assert_eq!(token, "rust");
                    *calls += 1;
                    Ok(())
                },
                |_| panic!("native failure is not a budget refusal"),
            )
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(calls, 1);
        assert_eq!(context.available_workspace_bytes(), available);
    }

    #[test]
    fn reclaiming_callback_errors_and_unwinds_never_retry_or_reclaim() {
        let limit = 64;
        for input in ["PROJECT", "FOO_BAR", "\u{23a}X_\u{130}"] {
            let expected = super::super::tests::token_reference(input);
            for fail_at in 0..expected.len() {
                for mode in 0..3 {
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        limit, 0, limit,
                    )));
                    let mut visits = 0;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        visit_tokens_in_reclaiming(
                            input,
                            &context,
                            &mut visits,
                            |visits, token| {
                                assert_eq!(token, expected[*visits]);
                                let current = *visits;
                                *visits += 1;
                                if current == fail_at {
                                    return match mode {
                                        0 => {
                                            Err(Error::new(ErrorCode::IoError, "callback sentinel"))
                                        }
                                        1 => Err(Error::resource_exhausted(
                                            "memory",
                                            "operation_scratch",
                                            limit as u64,
                                            limit as u128 + 1,
                                        )),
                                        _ => panic!("callback unwind sentinel"),
                                    };
                                }
                                Ok(())
                            },
                            |_| panic!("callbacks must never be retried"),
                        )
                    }));
                    if mode == 2 {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(
                            result.unwrap().unwrap_err().code(),
                            if mode == 0 {
                                ErrorCode::IoError
                            } else {
                                ErrorCode::ResourceExhausted
                            }
                        );
                    }
                    assert_eq!(visits, fail_at + 1);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn visitor_errors_and_unwinds_release_symbol_owners_and_stop_visiting() {
        let limit = 512;
        let slots = std::mem::size_of::<Cow<'_, str>>();
        for (input, expected, payload, arrays) in [
            ("PROJECT", &["project"][..], 7, 0),
            ("FOO_BAR", &["foo_bar", "foo", "bar"][..], 13, 4 * slots),
        ] {
            for fail_at in 0..expected.len() {
                for unwind in [false, true] {
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        limit, limit, limit,
                    )));
                    let mut visits = 0;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        visit_tokens_in(input, &context, |token| {
                            assert_eq!(token, expected[visits]);
                            assert_eq!(
                                context.available_workspace_bytes(),
                                limit - payload - arrays
                            );
                            let current = visits;
                            visits += 1;
                            if current == fail_at {
                                if unwind {
                                    panic!("injected visitor unwind");
                                }
                                return Err(Error::new(
                                    ErrorCode::IoError,
                                    "injected visitor error",
                                ));
                            }
                            Ok(())
                        })
                    }));
                    if unwind {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(result.unwrap().unwrap_err().code(), ErrorCode::IoError);
                    }
                    assert_eq!(visits, fail_at + 1);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }
}
