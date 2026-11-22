use super::STOP_WORDS;
use crate::json::{JsonOperationContext, ReservedString, WorkspaceReservation};
use crate::Result;
use std::borrow::Cow;

pub(super) struct Token<'a> {
    // Borrow unchanged text from the input symbol. Lowercasing that changes its
    // bytes supplies a separately owned lowercase string.
    pub(super) value: Cow<'a, str>,
    // Workspace reservation for an owned string's capacity, empty when the
    // normalized value borrows its source.
    pub(super) work: WorkspaceReservation,
}

pub(super) fn normalize<'a>(
    raw: &'a str,
    context: &JsonOperationContext,
) -> Result<Option<Token<'a>>> {
    // ASCII casing preserves byte length. Test stop words case-insensitively
    // before allocating a lowercase copy, so even an all-uppercase stop word needs
    // no token workspace.
    if raw.is_ascii() {
        if STOP_WORDS.iter().any(|stop| stop.eq_ignore_ascii_case(raw)) {
            return Ok(None);
        }
        if raw.bytes().any(|byte| byte.is_ascii_uppercase()) {
            let (value, work) =
                // Every input byte produces exactly one output byte on this path.
                // The exact reservation can become the surviving token's charge
                // without allowing for UTF-8 expansion.
                ReservedString::build_exact(raw.len(), context, "operation_scratch", |out| {
                    for byte in raw.bytes() {
                        out.write_char(byte.to_ascii_lowercase() as char)?;
                    }
                    Ok(())
                })?
                .into_parts();
            return Ok(Some(Token {
                value: Cow::Owned(value),
                work,
            }));
        }
        // Already-lowercase ASCII can be passed directly to the visitor. Return an
        // empty reservation alongside it so borrowed and owned tokens follow the
        // same lifetime protocol.
        return Ok(Some(Token {
            value: Cow::Borrowed(raw),
            work: context.workspace_reservation(),
        }));
    }
    let mut work = context.workspace_reservation();
    // An unchanged Unicode symbol can borrow the original bytes. When any scalar
    // changes case, let the standard string operation produce the full result so
    // context-sensitive lowercase rules apply.
    let value = if raw
        .chars()
        .all(|ch| ch.to_lowercase().eq(std::iter::once(ch)))
    {
        Cow::Borrowed(raw)
    } else {
        // Unicode lowercasing can expand both the scalar count and the byte
        // length. For example, U+0130 lowercases to an ASCII i followed by a
        // combining dot, so raw.len() alone is insufficient admission.
        let (capacity, peak) = lowercase_layout(raw);
        work.grow_to_wide(peak, peak, "operation_scratch")?;
        // Use the whole-string conversion for the actual spelling. Per-scalar
        // lowercasing can predict byte growth here but cannot choose the correct
        // contextual form of Greek final sigma.
        let value = raw.to_lowercase();
        // The estimate covers peak growth while old and new buffers can coexist.
        // After conversion retain only the delivered capacity, and reject a size
        // outside the verified allocation envelope.
        // The runtime check sees only the final allocation. The bound on
        // simultaneous old and new buffers comes from the layout calculation, so
        // the check alone cannot validate a changed growth policy.
        if value.capacity() as u128 > capacity {
            return Err(std::io::Error::other(
                "Unicode normalization allocation exceeds verified capacity",
            )
            .into());
        }
        work.retain(value.capacity());
        Cow::Owned(value)
    };
    // Compare the completed normalized spelling with the stop list. If a
    // conversion is discarded here, its owned buffer and reservation both leave
    // scope before any token is exposed.
    if STOP_WORDS.contains(&value.as_ref()) {
        return Ok(None);
    }
    Ok(Some(Token { value, work }))
}

// Verified std String growth envelope; toolchain changes require revalidation.
// Contextual sigma changes content, not UTF-8 length. Std still creates content.
pub(super) fn lowercase_layout(raw: &str) -> (u128, u128) {
    let mut capacity = raw.len() as u128;
    let mut peak = capacity;
    let mut length = 0u128;
    // Count the UTF-8 width of every lowercase output scalar. An expansion of a
    // single input character can cross a growth boundary before the next input
    // character is examined.
    for ch in raw.chars().flat_map(char::to_lowercase) {
        length += ch.len_utf8() as u128;
        if length > capacity {
            // Track the greater of geometric growth, the current byte requirement
            // and the minimum capacity. During reallocation both buffers can be
            // live, so the peak includes the old capacity as well as the
            // replacement.
            let next = (2 * capacity).max(length).max(8);
            peak = peak.max(capacity + next);
            capacity = next;
        }
    }
    (capacity, peak)
}
