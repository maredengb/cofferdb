//! Hand-written JSON text for ingestion artifacts.
//!
//! Object key order is part of the artifact contract. `JsonValue` stores objects
//! in sorted maps, so these bytes are assembled directly. Strings follow RFC 8259
//! for quotes, backslashes, and U+0000..U+001F, without escaping `/`.

use crate::hash::{digest_hex, sha256};

// Use the lowercase textual form of the SHA-256 of the supplied bytes. Callers
// pass complete artifact contents or an explicitly defined dataset or endpoint
// preimage, each with a different identity.
pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    digest_hex(sha256(bytes))
}

// Append one complete JSON string to the artifact under construction. Escaping
// happens before hashing, so different valid escape spellings would change the
// artifact identity.
pub(super) fn push_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            // Write each JSON control character from U+0000 through U+001F as a
            // four-digit lowercase Unicode escape. This gives a newline a stable
            // spelling and prevents raw line breaks from splitting a JSONL row.
            c if (c as u32) <= 0x1f => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            // Other characters are emitted in their ordinary UTF-8 representation.
            // In particular a slash in a source path does not acquire an optional
            // JSON escape.
            c => out.push(c),
        }
    }
    out.push('"');
}

pub(super) fn push_u64(out: &mut String, value: u64) {
    // The largest permitted unquoted integer is 2^53 - 1. Use this stable boundary
    // even for a larger value that happens to be individually representable in
    // binary64.
    const MAX_NUMBER: u64 = 9_007_199_254_740_991;
    if value <= MAX_NUMBER {
        out.push_str(&value.to_string());
    } else {
        // Beyond the interoperable numeric range, encode the exact decimal digits
        // as a string. Source offsets can then retain all their bits when an
        // artifact passes through ordinary JSON readers.
        out.push('"');
        out.push_str(&value.to_string());
        out.push('"');
    }
}

pub(super) struct ArtifactBytes {
    // Serialized artifact bytes passed to publication without re-encoding.
    pub bytes: Vec<u8>,
    // SHA-256 computed over those same bytes, including their line endings.
    pub sha256: String,
}

impl ArtifactBytes {
    pub(super) fn from_string(text: String) -> Self {
        // Compute the digest before transferring the String's allocation to the
        // byte vector. Both the bytes and the digest then describe precisely the
        // data later written to disk.
        let sha256 = sha256_hex(text.as_bytes());
        Self {
            bytes: text.into_bytes(),
            sha256,
        }
    }
}

// The dataset preimage describes how named inputs are chunked. Document contents
// and produced vectors receive separate artifact digests, so this identifier need
// not read either payload.
pub(super) fn dataset_preimage(
    format: &str,
    chunk_size: u64,
    chunk_overlap: u64,
    paths: &[String],
) -> String {
    let mut out = String::new();
    // Keep field order, fixed labels, decimal option spellings and line separators
    // stable. They are input to a digest, not just a display representation.
    out.push_str("format=");
    out.push_str(format);
    out.push('\n');
    out.push_str("chunk_size=");
    out.push_str(&chunk_size.to_string());
    out.push('\n');
    out.push_str("chunk_overlap=");
    out.push_str(&chunk_overlap.to_string());
    out.push('\n');
    // The caller supplies paths in scanner order. Preserve each supplied path's
    // spelling and order when appending it and its final newline to the hash
    // input.
    for path in paths {
        out.push_str(path);
        out.push('\n');
    }
    out
}

// This literal is embedded verbatim in generated manifests. The property order and
// value spellings participate in their byte identity even when two JSON objects
// would parse to identical keys and values.
pub(super) fn tokenizer_json() -> &'static str {
    r#"{"name":"default_en","version":"0.1","profile":"latin_code_default","language_hint":"en","config":{"case_fold":"lower","split":"unicode_punctuation","stop_words":"default_en"}}"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_use_lowercase_hex_and_slash_stays_raw() {
        let mut out = String::new();
        push_string(&mut out, "a/b\"\\\n\u{0001}");
        assert_eq!(out, "\"a/b\\\"\\\\\\u000a\\u0001\"");
    }
}
