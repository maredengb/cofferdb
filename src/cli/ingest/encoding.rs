//! Explicit directory input encodings. All downstream stages consume UTF-8.
use super::boundary::cli;
use crate::error::Result;
use encoding_rs::{Encoding, GB18030, SHIFT_JIS, UTF_16BE, UTF_16LE, UTF_8, WINDOWS_1252};

#[derive(Clone, Copy)]
pub(super) struct InputEncoding {
    // Declared CLI spelling preserved in the scan and skip reports, including
    // utf-8-sig.
    pub label: &'static str,
    // Decoder used for strict conversion and matching any UTF byte-order mark.
    codec: &'static Encoding,
}

impl InputEncoding {
    pub fn parse(label: &str) -> Result<Self> {
        // Accept only the spellings advertised by directory import. A generic
        // codec-label resolver would also accept aliases and case variants,
        // changing which configuration the CLI considers valid.
        let (label, codec) = match label {
            "utf-8" => ("utf-8", UTF_8),
            // Keep the distinct reporting label even though both UTF-8 spellings
            // use the same decoder. A BOM is optional for either spelling and is
            // removed when present.
            "utf-8-sig" => ("utf-8-sig", UTF_8),
            "utf-16le" => ("utf-16le", UTF_16LE),
            "utf-16be" => ("utf-16be", UTF_16BE),
            "gb18030" => ("gb18030", GB18030),
            "shift_jis" => ("shift_jis", SHIFT_JIS),
            "windows-1252" => ("windows-1252", WINDOWS_1252),
            // Treat an unsupported declaration as an argument error. A malformed
            // file in a supported encoding instead yields None from decode,
            // allowing the scanner to report a per-file skip.
            _ => {
                return Err(cli(
                    "unsupported directory input encoding",
                    Some("--input-encoding"),
                    Some("unsupported_input_encoding"),
                ))
            }
        };
        Ok(Self { label, codec })
    }

    pub fn decode(self, raw: &[u8]) -> Option<String> {
        // Chunk offsets refer to text decoded with the selected codec. For UTF
        // input, strip a matching BOM and reject one that would switch codecs.
        let bytes = if self.codec == UTF_8 || self.codec == UTF_16LE || self.codec == UTF_16BE {
            match Encoding::for_bom(raw) {
                // Discard exactly the matching signature bytes. For UTF-16 their
                // order must agree with the declared endianness; otherwise the
                // scanner cannot use the resulting text and offsets.
                Some((bom, length)) if bom == self.codec => &raw[length..],
                Some(_) => return None,
                None => raw,
            }
        } else {
            // For an explicitly selected legacy encoding, pass the bytes to that
            // codec unchanged. A prefix that resembles a UTF signature does not
            // switch this file to a different decoder.
            raw
        };
        self.codec
            // Strict decoding preserves the byte-to-text decision for chunk
            // generation. Substituting a replacement character could turn
            // malformed input into a different document that appears valid.
            .decode_without_bom_handling_and_without_replacement(bytes)
            .map(|text| text.into_owned())
    }
}
