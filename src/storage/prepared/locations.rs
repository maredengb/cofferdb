use super::*;

// Every location occupies a fixed number of bytes. This lets the planner seek to
// an arbitrary record ordinal without retaining an in-memory list of all payload
// offsets.
pub(super) const LOCATION_BYTES: usize = 28;

pub(super) struct StagedRecord {
    // Critical frame type to write and, for a replacing tombstone, the document or
    // chunk type whose put it pairs with.
    pub(super) kind: BatchRecordKind,
    // Absolute byte offset, length and payload CRC in the separate staged data
    // file. These offsets belong to the temporary input until append supplies final
    // log locations.
    pub(super) source: PayloadRef,
}

impl StagedRecord {
    pub(super) fn encode(&self) -> Result<[u8; LOCATION_BYTES]> {
        // PayloadRef shares its checksum field with other storage formats. The
        // staging format stores a 32-bit CRC, so reject a larger value rather than
        // silently changing the expected digest.
        let checksum = u32::try_from(self.source.checksum).map_err(|_| invalid_location())?;
        let mut bytes = [0; LOCATION_BYTES];
        bytes[..2].copy_from_slice(&self.kind.record_type.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.kind.replaces.to_le_bytes());
        bytes[4..12].copy_from_slice(&self.source.offset.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.source.len.to_le_bytes());
        bytes[20..24].copy_from_slice(&checksum.to_le_bytes());
        // The reader checks the entry CRC before using its record kind, offset and
        // length. The embedded payload checksum is checked against the data itself.
        let checksum = checksum32(&bytes[..24]);
        bytes[24..].copy_from_slice(&checksum.to_le_bytes());
        Ok(bytes)
    }

    fn decode(bytes: &[u8; LOCATION_BYTES], source_bytes: u64) -> Result<Self> {
        if checksum32(&bytes[..24]) != read_u32(&bytes[24..]) {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "prepared import location checksum mismatch",
            ));
        }
        // Decode the two kind fields independently. The second describes which
        // immediately following put must share a commit with a replacement
        // tombstone; it is not another record type at this offset.
        let kind = BatchRecordKind {
            record_type: u16::from_le_bytes(bytes[..2].try_into().unwrap()),
            replaces: u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        };
        let source = PayloadRef {
            offset: read_u64(&bytes[4..12]),
            len: read_u64(&bytes[12..20]),
            checksum: u64::from(read_u32(&bytes[20..24])),
        };
        // A CRC protects the directory bytes from accidental changes but does not
        // make their values legal. Permit only importable data records; the main
        // writer creates the commit frame after the staged group is selected.
        if !matches!(
            kind.record_type,
            COLLECTION_CREATE
                | COLLECTION_DROP
                | DOCUMENT_PUT
                | CHUNK_PUT
                | EMBEDDING_PUT
                | TOMBSTONE
        ) || !(kind.replaces == 0
            || (kind.record_type == TOMBSTONE && matches!(kind.replaces, DOCUMENT_PUT | CHUNK_PUT)))
            // Validate the nonempty payload against the companion file's byte
            // length without wrapping its absolute end. The location reader has no
            // basis for extending a read past the accepted staging prefix.
            || source.len == 0
            || source
                .offset
                .checked_add(source.len)
                .filter(|end| *end <= source_bytes)
                .is_none()
        {
            return Err(invalid_location());
        }
        // Successful decoding proves the location is well formed. Its payload
        // still has to be read and compared with the embedded checksum before
        // these bytes can be committed to the database.
        Ok(Self { kind, source })
    }
}

pub(super) struct LocationReader<'a> {
    // Borrow of the location file positioned at the next logical entry. Read-ahead
    // can leave the underlying file cursor beyond that entry.
    input: BufReader<&'a mut File>,
    // Optional workspace credit for the input buffer. Zero permits direct reads
    // into the fixed-size entry array.
    work: WorkspaceReservation,
    // Count of entries left from the requested suffix. A read or decode failure
    // consumes the iterator so an uncertain boundary cannot be reused.
    remaining: usize,
    // Byte length of the staged payload file. Each decoded offset plus length must
    // lie within this bound.
    source_bytes: u64,
}

impl<'a> LocationReader<'a> {
    pub(super) fn new(
        file: &'a mut File,
        first: usize,
        remaining: usize,
        source_bytes: u64,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        // The requested suffix must consume the location file to its end. A short
        // file means an incomplete fixed-width entry or a missing entry. Surplus
        // bytes would represent locations the caller never counted.
        // Validate the entire suffix geometry before seeking to its first entry.
        // Checked arithmetic covers both the logical entry count and the
        // conversion to bytes. A corrupt count must never select a wrapped offset.
        let total = first.checked_add(remaining).ok_or_else(invalid_location)?;
        let expected_bytes = (total as u64)
            .checked_mul(LOCATION_BYTES as u64)
            .ok_or_else(invalid_location)?;
        let actual_bytes = file.metadata()?.len();
        // Distinguish a physically short location file from one with surplus data.
        // Trailing bytes indicate disagreement with the record count even if all
        // requested entries can be read.
        if actual_bytes < expected_bytes {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        if actual_bytes != expected_bytes {
            return Err(invalid_location());
        }
        // Entries have fixed width, so seeking directly to the suffix skips
        // already consumed records. The preceding byte-count proof also bounds
        // this starting offset.
        file.seek(SeekFrom::Start(first as u64 * LOCATION_BYTES as u64))?;
        let mut work = context.workspace_reservation();
        // Base optional buffering on the entries left to read, with a small upper
        // bound. The zero minimum leaves the reader usable under memory pressure.
        let preferred = remaining.saturating_mul(LOCATION_BYTES).min(8192);
        let capacity = work.grow_to(0, preferred, "operation_scratch")?;
        let mut input = BufReader::with_capacity(capacity, file);
        // Drop a buffer whose actual capacity exceeds the admitted grant before
        // returning its credit. No bytes have been read yet, so this fallback can
        // use the same cursor.
        if input.capacity() > capacity {
            let file = input.into_inner();
            work.retain(0);
            input = BufReader::with_capacity(0, file);
        }
        // Keep the bound on source payload bytes separately from the location file
        // length. Each decoded locator is validated against the source bound even
        // though this reader consumes only location records.
        Ok(Self {
            input,
            work,
            remaining,
            source_bytes,
        })
    }

    pub(super) fn is_buffered(&self) -> bool {
        self.work.reserved_bytes() > 0
    }

    pub(super) fn into_unbuffered(mut self) -> Result<Self> {
        // Discard read-ahead without skipping the locations already in the buffer.
        // Ask BufReader for the logical position. Its underlying file can be ahead
        // by unread entries. Seek back to the logical position before
        // relinquishing the buffer to keep the next unread entry available.
        let position = self.input.stream_position()?;
        self.input.seek(SeekFrom::Start(position))?;
        let Self {
            input,
            mut work,
            remaining,
            source_bytes,
        } = self;
        let file = input.into_inner();
        // Free the buffer before releasing its workspace. The new reader carries
        // the same record count and staged-data bound but reads without prefetch.
        work.retain(0);
        Ok(Self {
            input: BufReader::with_capacity(0, file),
            work,
            remaining,
            source_bytes,
        })
    }
}

impl Iterator for LocationReader<'_> {
    type Item = Result<StagedRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        // A zero remaining count ends the iterator without probing the file. The
        // constructor already ruled out extra entries, so reaching this state is
        // enough to finish a successful suffix.
        if self.remaining == 0 {
            return None;
        }
        // Read one fixed-width location and give the decoder the full staged file
        // bound. A valid width alone cannot establish that its claimed payload
        // range fits the source.
        let mut bytes = [0; LOCATION_BYTES];
        let result = self
            .input
            .read_exact(&mut bytes)
            .map_err(Error::from)
            .and_then(|_| StagedRecord::decode(&bytes, self.source_bytes));
        // Treat a failed entry as terminal. The cursor may have consumed only a
        // prefix of its bytes. Continuing would reinterpret an unknown boundary or
        // silently skip a malformed record.
        self.remaining = if result.is_ok() {
            self.remaining - 1
        } else {
            0
        };
        Some(result)
    }
}

fn invalid_location() -> Error {
    Error::new(
        ErrorCode::FileFormatError,
        "invalid prepared import location",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_buffers_share_headroom_and_preserve_read_ahead_on_release() {
        use crate::json::JsonProcessingPolicyV1;
        for seed in 0..100usize {
            let (mut file, temporary) = crate::temporary::create_temporary_file_in(
                &std::env::temp_dir(),
                "location-work",
                "tmp",
            )
            .unwrap();
            let count = 317 + seed;
            for offset in 0..count {
                file.write_all(
                    &StagedRecord {
                        kind: BatchRecordKind {
                            record_type: DOCUMENT_PUT,
                            replaces: 0,
                        },
                        source: PayloadRef {
                            offset: offset as u64,
                            len: 1,
                            checksum: seed as u64,
                        },
                    }
                    .encode()
                    .unwrap(),
                )
                .unwrap();
            }
            let context = JsonOperationContext::standalone();
            let available = [0, 1, 27, 28, 29, 4096, 8191, 8192, 10000][seed % 9];
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let mut reader =
                LocationReader::new(&mut file, 3, count - 3, count as u64, &context).unwrap();
            let capacity = reader.input.capacity();
            assert_eq!(capacity, available.min(8192));
            assert_eq!(reader.work.reserved_bytes(), capacity);
            for offset in 3..count {
                if offset == 8 {
                    reader = reader.into_unbuffered().unwrap();
                    assert!(!reader.is_buffered());
                    assert_eq!(reader.input.capacity(), 0);
                    context
                        .workspace_reservation()
                        .grow_to(available, available, "operation_scratch")
                        .unwrap();
                }
                let actual = reader.next().unwrap().unwrap();
                assert_eq!(actual.source.offset, offset as u64);
                assert_eq!(actual.source.checksum, seed as u64);
            }
            assert!(reader.next().is_none());
            drop(reader);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();

            file.seek(SeekFrom::Start((7 * LOCATION_BYTES) as u64))
                .unwrap();
            file.write_all(&[0xff]).unwrap();
            let mut reader =
                LocationReader::new(&mut file, 7, count - 7, count as u64, &context).unwrap();
            assert_eq!(
                reader.next().unwrap().err().unwrap().code(),
                ErrorCode::ChecksumMismatch
            );
            assert!(reader.next().is_none());
            drop(reader);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            file.set_len((count * LOCATION_BYTES - 1) as u64).unwrap();
            let error = LocationReader::new(&mut file, 0, count, count as u64, &context)
                .err()
                .unwrap();
            assert_eq!(error.code(), ErrorCode::IoError);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop(file);
            drop(temporary);
        }
    }

    #[test]
    fn locations_roundtrip_and_reject_every_corrupted_byte_across_100_seeds() {
        for seed in 0..100_u64 {
            let record = StagedRecord {
                kind: BatchRecordKind {
                    record_type: TOMBSTONE,
                    replaces: [0, DOCUMENT_PUT, CHUNK_PUT][seed as usize % 3],
                },
                source: PayloadRef {
                    offset: seed * 37,
                    len: seed + 1,
                    checksum: seed * 1234567,
                },
            };
            let bytes = record.encode().unwrap();
            let end = record.source.offset + record.source.len;
            assert_eq!(
                StagedRecord::decode(&bytes, end).unwrap().encode().unwrap(),
                bytes
            );
            assert!(StagedRecord::decode(&bytes, end - 1).is_err());
            for position in 0..LOCATION_BYTES {
                let mut bad = bytes;
                bad[position] ^= 0x80;
                assert_eq!(
                    StagedRecord::decode(&bad, end).err().unwrap().code(),
                    ErrorCode::ChecksumMismatch
                );
            }
            for (offset, length, kind, replaces) in [
                (u64::MAX, 1, TOMBSTONE, 0),
                (0, 0, TOMBSTONE, 0),
                (0, 1, COMMIT, 0),
                (0, 1, DOCUMENT_PUT, CHUNK_PUT),
                (0, 1, TOMBSTONE, EMBEDDING_PUT),
            ] {
                let bad = StagedRecord {
                    kind: BatchRecordKind {
                        record_type: kind,
                        replaces,
                    },
                    source: PayloadRef {
                        offset,
                        len: length,
                        checksum: 0,
                    },
                }
                .encode()
                .unwrap();
                assert_eq!(
                    StagedRecord::decode(&bad, u64::MAX).err().unwrap().code(),
                    ErrorCode::FileFormatError
                );
            }
        }
    }
}
