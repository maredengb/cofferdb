use std::io::{BufRead, BufReader, Read};

use super::{
    parse_with_schema_in, retry_after_memory_reclaim, JsonOperationContext, JsonReservation,
    JsonSchema, ProcessedJson,
};
use crate::{Error, ErrorCode, Result};

pub(crate) struct JsonLines<R> {
    // Buffered source. An unconsumed slice can be used again after releasing
    // optional memory.
    reader: BufReader<R>,
    // Raw bytes retained for the current physical line, which can span several
    // fills of the input buffer.
    line: Vec<u8>,
    // Operation shared by reading, per-line decoding and reclamation. Parsed
    // results can hold their own charges when iteration resumes.
    context: JsonOperationContext,
    // Schema applied independently to every nonblank record, including the input
    // kind used for limit errors.
    schema: JsonSchema,
    // One-based physical line being processed once reading begins. Blank lines
    // advance it even though they yield no record.
    line_number: usize,
    // Cumulative source bytes accounted for over the stream, including delimiters
    // and skipped blank lines.
    bytes_read: u64,
    // Maximum raw bytes accepted from this input across all lines. An individual
    // JSON unit also has its own size limit.
    byte_limit: u64,
    // Terminal state reached at EOF or on an error. A later next call must not
    // resume partway through a failed input.
    done: bool,
    // Charge for the BufReader allocation, held for the reader's lifetime
    // independently of the current line.
    _input_work: JsonReservation,
    // Charge for the separate line buffer, whose reusable capacity can outlast a
    // short line.
    line_work: JsonReservation,
}

impl<R: Read> JsonLines<R> {
    // Consuming this adapter returns the underlying reader at its own read
    // position. BufReader may have read ahead, so it cannot be used to resume at
    // the last yielded JSONL boundary simply by taking the reader early.
    pub(crate) fn into_inner(self) -> R {
        self.reader.into_inner()
    }

    pub(crate) fn new(
        source: R,
        schema: JsonSchema,
        context: &JsonOperationContext,
        byte_limit: u64,
    ) -> Result<Self> {
        // Both constructors use the same buffer and record limits. This form has
        // no caller-owned cache to release, so an initial memory refusal is
        // returned directly.
        Self::new_reclaiming(source, schema, context, byte_limit, || false)
    }

    pub(crate) fn new_reclaiming(
        source: R,
        schema: JsonSchema,
        context: &JsonOperationContext,
        byte_limit: u64,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<Self> {
        let mut input_work = context.reservation();
        // Admit the entire input buffer before building BufReader or reading from
        // the source. Only this reservation is retried after reclamation, leaving
        // a one-shot reader at its original position on failure.
        retry_after_memory_reclaim(|| input_work.grow(8192, schema.kind()), reclaim)
            .map_err(|error| error.with_detail("line", 1))?;
        Ok(Self {
            reader: BufReader::with_capacity(8192, source),
            context: context.clone(),
            schema,
            _input_work: input_work,
            // Start the physical-line reservation separately from the input-buffer
            // charge. A long record can require both allocations at once even when
            // most records fit in one input fill.
            line_work: context.reservation(),
            line: Vec::new(),
            // Zero is the pre-read sentinel. The reader advances the physical line
            // number before parsing or skipping a line, so its first diagnostic
            // and first yielded row are one-based.
            line_number: 0,
            bytes_read: 0,
            byte_limit,
            done: false,
        })
    }

    fn read_next(
        &mut self,
        reclaim: &mut impl FnMut() -> bool,
    ) -> Result<Option<(usize, ProcessedJson)>> {
        loop {
            // Number physical lines, including blank lines. Error details and
            // successful records use the same one-based numbering even when blank
            // lines were skipped.
            self.line_number = self.line_number.checked_add(1).ok_or_else(|| {
                Error::new(ErrorCode::InvalidJsonlSchema, "JSONL line count overflow")
            })?;
            // Clear the contents while keeping the charged capacity. Later lines
            // can reuse the buffer; a larger line must admit the additional
            // capacity before appending bytes.
            self.line.clear();
            loop {
                // Retry an interrupted buffer fill before counting or consuming
                // input. Other I/O errors end the iterator through the caller.
                let bytes = match self.reader.fill_buf() {
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    result => result?,
                };
                if bytes.is_empty() {
                    break;
                }
                // A buffer can hold several records or only a fragment of one
                // record. Copy through the first newline when present and leave
                // the following bytes in the reader.
                let count = bytes
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map_or(bytes.len(), |index| index + 1);
                self.bytes_read = self
                    .bytes_read
                    .checked_add(count as u64)
                    .filter(|bytes| *bytes <= self.byte_limit)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::InvalidJsonlSchema,
                            "JSONL input exceeds the byte budget",
                        )
                        .with_detail("reason", "import_budget_exceeded")
                        .with_detail("limit_bytes", self.byte_limit)
                    })?;
                // Count raw input, including newlines and skipped blank records. Consume
                // the reader only after admission so a reclaim retry sees the same bytes.
                let complete = bytes[count - 1] == b'\n';
                retry_after_memory_reclaim(
                    || {
                        self.line_work
                            .reserve_vec(&mut self.line, count, self.schema.kind())
                    },
                    &mut *reclaim,
                )?;
                self.line.extend_from_slice(&bytes[..count]);
                self.reader.consume(count);
                if complete {
                    break;
                }
            }
            // An empty accumulator at EOF means there is no next record. A
            // nonempty accumulator may hold the final line of a file without a
            // trailing newline.
            if self.line.is_empty() {
                return Ok(None);
            }
            // Validate UTF-8 after assembling the entire physical line. A
            // character may straddle two reader buffers.
            let line = std::str::from_utf8(&self.line).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stream did not contain valid UTF-8",
                )
            })?;
            // The UTF-8 and raw byte checks apply to blank lines too. Skip them
            // only after those checks so malformed input cannot disappear during
            // whitespace handling.
            if line.trim().is_empty() {
                continue;
            }
            // Retry the parse against this same buffered line. Reading the next line
            // during reclamation would skip the record that encountered the memory
            // limit.
            let value = retry_after_memory_reclaim(
                || parse_with_schema_in(line, self.schema, &self.context),
                &mut *reclaim,
            )?;
            // Transfer the parsed value together with its reservation. The reader
            // keeps its own separate reservation for the reusable line buffer.
            return Ok(Some((self.line_number, value)));
        }
    }

    pub(crate) fn next_reclaiming(
        &mut self,
        mut reclaim: impl FnMut() -> bool,
    ) -> Option<Result<(usize, ProcessedJson)>> {
        if self.done {
            return None;
        }
        let result = self.read_next(&mut reclaim);
        // End iteration on the first error. A failed read may have consumed part of
        // a record; a subsequent call cannot reliably establish the next record
        // boundary.
        self.done = matches!(&result, Ok(None) | Err(_));
        // Report the line associated with the error on the call that encounters
        // it. The done flag makes subsequent calls return None without attempting
        // to read again.
        result
            .map_err(|error| error.with_detail("line", self.line_number))
            .transpose()
    }
}

impl<R: Read> Iterator for JsonLines<R> {
    type Item = Result<(usize, ProcessedJson)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_reclaiming(|| false)
    }
}

impl<R: Read> std::iter::FusedIterator for JsonLines<R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonInputKind, JsonProcessingPolicyV1};

    struct RepeatedRows<'a> {
        bytes: &'a [u8],
        remaining: usize,
        offset: usize,
    }

    impl Read for RepeatedRows<'_> {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Ok(0);
            }
            let count = output.len().min(self.bytes.len() - self.offset);
            output[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
            self.offset += count;
            if self.offset == self.bytes.len() {
                self.offset = 0;
                self.remaining -= 1;
            }
            Ok(count)
        }
    }

    #[test]
    fn reclaiming_rows_retry_only_admission_or_retained_line_parsing() {
        use std::cell::Cell;
        struct Counted<'a> {
            input: &'a [u8],
            bytes: &'a Cell<usize>,
        }
        impl Read for Counted<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let count = self.input.read(out)?;
                self.bytes.set(self.bytes.get() + count);
                Ok(count)
            }
        }
        let input =
            b"{\"chunk_external_id\":\"c\",\"model_name\":\"m\",\"dim\":1,\"vector\":[1]}\n";
        for headroom in [1, input.len() + 8] {
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let bytes = Cell::new(0);
            let mut rows = JsonLines::new(
                Counted {
                    input,
                    bytes: &bytes,
                },
                JsonSchema::EmbeddingRow,
                &context,
                input.len() as u64,
            )
            .unwrap();
            let mut cache = Some(
                context
                    .cache_reservation(
                        (context.available_total_bytes() - headroom) as u128,
                        usize::MAX,
                    )
                    .unwrap(),
            );
            let mut reclaims = 0;
            let (line, parsed) = rows
                .next_reclaiming(|| {
                    reclaims += 1;
                    cache.take().is_some()
                })
                .unwrap()
                .unwrap();
            assert_eq!(line, 1);
            assert_eq!(
                parsed.value.get("chunk_external_id").unwrap().as_str(),
                Some("c")
            );
            assert_eq!(reclaims, 1);
            assert_eq!(bytes.get(), input.len());
            drop(parsed);
            assert!(rows.next_reclaiming(|| panic!("EOF reclaimed")).is_none());
            assert_eq!(bytes.get(), input.len());
            drop(rows);
            assert_eq!(context.available_workspace_bytes(), available);
        }
    }

    #[test]
    fn rows_reuse_released_capacity_but_charge_retained_values() {
        let context = JsonOperationContext::standalone();
        let mut busy = context.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 2 * 1024 * 1024,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let bytes = format!(
            "{{\"external_id\":\"doc\",\"metadata\":{{\"value\":\"{}\"}}}}\n",
            "x".repeat(512 * 1024)
        )
        .into_bytes();
        let source = || RepeatedRows {
            bytes: &bytes,
            remaining: 100,
            offset: 0,
        };
        let rows = JsonLines::new(source(), JsonSchema::DocumentRow, &context, u64::MAX).unwrap();
        let mut count = 0;
        for row in rows {
            let (line, parsed) = row.unwrap();
            count += 1;
            assert_eq!(line, count);
            assert_eq!(
                parsed
                    .value
                    .get("metadata")
                    .unwrap()
                    .get("value")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .len(),
                512 * 1024
            );
        }
        assert_eq!(count, 100);
        let mut rows =
            JsonLines::new(source(), JsonSchema::DocumentRow, &context, u64::MAX).unwrap();
        let first = rows.next().unwrap().unwrap();
        let error = rows.next().unwrap().err().unwrap();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert!(rows.next().is_none());
        drop((first, rows));
        busy.grow(2 * 1024 * 1024, JsonInputKind::JsonlRecord)
            .unwrap();
    }

    #[test]
    fn byte_limit_is_exact_and_errors_terminate_the_iterator() {
        for (bytes, limit, expected) in [
            (&b"{}\n{}\n"[..], 5, ErrorCode::InvalidJsonlSchema),
            (&b"{}\n\n{\"x\":\"\xff\"}"[..], u64::MAX, ErrorCode::IoError),
            (&b"{}\n\n{"[..], u64::MAX, ErrorCode::InvalidJsonlSchema),
        ] {
            let context = JsonOperationContext::standalone();
            let mut rows = JsonLines::new(bytes, JsonSchema::DocumentRow, &context, limit).unwrap();
            drop(rows.next().unwrap().unwrap());
            let error = rows.next().unwrap().err().unwrap();
            assert_eq!(error.code(), expected);
            let line = if limit == 5 { "2" } else { "3" };
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "line" && value == line));
            assert!(rows.next().is_none());
            assert!(rows.next().is_none());
            drop(rows);
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
        let context = JsonOperationContext::standalone();
        let rows = JsonLines::new(&b"{}\n{}\n"[..], JsonSchema::DocumentRow, &context, 6).unwrap();
        assert_eq!(rows.map(|row| row.unwrap().0).collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn line_buffer_failure_uses_the_enclosing_schema_kind() {
        for schema in [
            JsonSchema::DocumentRow,
            JsonSchema::ChunkRow,
            JsonSchema::EmbeddingRow,
            JsonSchema::QueryRow,
            JsonSchema::QueryEmbeddingObject,
            JsonSchema::BenchmarkRecord,
        ] {
            let context = JsonOperationContext::standalone();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 16384,
                schema.kind(),
            )
            .unwrap();
            let bytes = format!("\n{{\"text\":\"{}\"}}", "x".repeat(32768));
            let mut rows = JsonLines::new(bytes.as_bytes(), schema, &context, u64::MAX).unwrap();
            let error = rows.next().unwrap().err().unwrap();
            assert_eq!(error.code(), schema.kind().error_code());
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == schema.kind().as_str()));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "line" && value == "2"));
            drop(rows);
            busy.grow(16384, schema.kind()).unwrap();
        }
    }

    #[test]
    fn fragmented_unicode_and_interrupted_reads_keep_line_numbers() {
        struct Fragmented<'a> {
            bytes: &'a [u8],
            calls: usize,
            width: usize,
        }
        impl Read for Fragmented<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                self.calls += 1;
                if self.calls % 2 == 1 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = output.len().min(self.bytes.len()).min(self.width);
                output[..count].copy_from_slice(&self.bytes[..count]);
                self.bytes = &self.bytes[count..];
                Ok(count)
            }
        }
        let bytes = " \r\n{\"text\":\"\u{4e2d}\u{1f642}\"}\r\n\n{\"text\":\"last\"}";
        for width in 1..=100 {
            let context = JsonOperationContext::standalone();
            let rows = JsonLines::new(
                Fragmented {
                    bytes: bytes.as_bytes(),
                    calls: 0,
                    width,
                },
                JsonSchema::ChunkRow,
                &context,
                bytes.len() as u64,
            )
            .unwrap();
            let rows = rows
                .map(|row| {
                    let (line, parsed) = row.unwrap();
                    (
                        line,
                        parsed
                            .value
                            .get("text")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(rows, [(2, "\u{4e2d}\u{1f642}".into()), (4, "last".into())]);
        }
    }
}
