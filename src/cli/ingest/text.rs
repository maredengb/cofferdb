//! Hand-written JSON text for ingestion artifacts.
//!
//! Object key order is part of the artifact contract. `JsonValue` stores objects
//! in sorted maps, so these bytes are assembled directly. Strings follow RFC 8259
//! for quotes, backslashes, and U+0000..U+001F, without escaping `/`.

use crate::hash::{digest_hex, sha256};

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    digest_hex(sha256(bytes))
}

pub(super) fn push_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) <= 0x1f => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub(super) fn push_u64(out: &mut String, value: u64) {
    const MAX_NUMBER: u64 = 9_007_199_254_740_991;
    if value <= MAX_NUMBER {
        out.push_str(&value.to_string());
    } else {
        out.push('"');
        out.push_str(&value.to_string());
        out.push('"');
    }
}

pub(super) struct ArtifactBytes {
    pub bytes: Vec<u8>,
    pub sha256: String,
}

impl ArtifactBytes {
    pub(super) fn from_string(text: String) -> Self {
        let sha256 = sha256_hex(text.as_bytes());
        Self {
            bytes: text.into_bytes(),
            sha256,
        }
    }
}

pub(super) fn dataset_preimage(
    format: &str,
    chunk_size: u64,
    chunk_overlap: u64,
    paths: &[String],
) -> String {
    let mut out = String::new();
    out.push_str("format=");
    out.push_str(format);
    out.push('\n');
    out.push_str("chunk_size=");
    out.push_str(&chunk_size.to_string());
    out.push('\n');
    out.push_str("chunk_overlap=");
    out.push_str(&chunk_overlap.to_string());
    out.push('\n');
    for path in paths {
        out.push_str(path);
        out.push('\n');
    }
    out
}

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
