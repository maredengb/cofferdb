use super::*;
use crate::json::{JsonInputKind, JsonOperationContext, JsonReservation, WorkspaceReservation};

// The summary cache is optional. When it cannot fit, retain only the verified
// batch range and replay it after the commit marker has been validated.
pub(super) struct PendingBatch {
    // Decoded record summaries in log order for the currently uncommitted batch.
    // The entire vector is discarded if recovery switches to replay.
    records: Vec<CommittedRecord>,
    // JSON and string credit shared by these summaries. The visitor may transfer
    // credit to a longer-lived catalog entry when consuming a committed record.
    work: RecordReservation,
    // Workspace bytes covering the allocated record slots, released independently
    // of the payload allocations when the buffer is abandoned.
    array: WorkspaceReservation,
    // True while every record since the first has been retained. Once false, the
    // whole batch must be reread after its commit matches.
    buffered: bool,
    // Number of data frames seen before the commit. It continues to advance during
    // the unbuffered path and excludes the commit frame.
    count: usize,
    // Absolute file offset of the first data frame's header. Meaningful only for a
    // nonempty batch and used as the replay starting point.
    first_offset: u64,
    // Sequence number in the first data header. Together with count and the last
    // sequence it must agree with the commit payload.
    first_sequence: u64,
    // Sequence number of the most recent data frame. The caller separately verifies
    // sequence continuity as it scans.
    last_sequence: u64,
}

impl PendingBatch {
    pub(super) fn new(context: &JsonOperationContext) -> Self {
        Self {
            records: Vec::new(),
            // Keep the heap owned by decoded records separate from the vector of
            // record slots. Recovery can transfer each row's field credit to the
            // catalog while releasing the slot storage in a single operation.
            work: RecordReservation::new(context.reservation(), context.workspace_reservation()),
            array: context.workspace_reservation(),
            buffered: true,
            count: 0,
            first_offset: 0,
            first_sequence: 0,
            last_sequence: 0,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        // The decoded vector can be empty even when a commit has pending records.
        // Count accepted frames on both buffering paths; testing the vector would
        // mistake an unbuffered batch for an empty one.
        self.count == 0
    }

    pub(super) fn first_sequence(&self) -> Option<u64> {
        (!self.is_empty()).then_some(self.first_sequence)
    }

    pub(super) fn matches(&self, commit: &CommitPayload) -> bool {
        // This checks the record interval named by a commit. The scanner also
        // checks its batch ID and accumulated checksum before it can visit the
        // range or advance the verified end.
        self.count > 0
            && self.count == commit.record_count
            && self.first_sequence == commit.first_sequence_no
            && self.last_sequence == commit.last_sequence_no
    }

    pub(super) fn discard_buffer(&mut self) -> bool {
        // Only the initial transition to unbuffered mode may count as reclaiming
        // space. Reporting success again would let a caller repeat an allocation
        // with exactly the same available memory.
        if !self.buffered {
            return false;
        }
        // Retain the count, offsets and sequence endpoints. They are enough to match
        // the commit and locate the batch again. Dropping even the already buffered
        // records keeps the replay path responsible for the whole batch exactly
        // once.
        // Drop the decoded records themselves before returning any associated
        // credit to the ledger. Retaining vector capacity here would leave the
        // supposedly released slot allocation in the recovery buffer.
        self.records = Vec::new();
        self.work.json.retain(0);
        self.work.retain_workspace(0);
        self.array.retain(0);
        self.buffered = false;
        true
    }

    pub(super) fn push(
        &mut self,
        json: JsonReservation,
        strings: WorkspaceReservation,
        record: CommittedRecord,
    ) -> Result<()> {
        // Capture the first record's physical offset and sequence once. If the
        // summary buffer is later abandoned, these locate the start of the exact
        // batch that the commit must cover.
        if self.is_empty() {
            self.first_offset = record.record_ref.offset;
            self.first_sequence = record.record_ref.sequence_no;
        }
        // Update the endpoint and count even when no summaries are retained.
        // Recovery still verifies the commit's reported sequence range and record
        // count before revisiting any data frames.
        self.last_sequence = record.record_ref.sequence_no;
        self.count = self.count.checked_add(1).ok_or_else(|| {
            Error::new(ErrorCode::FileFormatError, "recovery batch count overflow")
        })?;
        // A missing buffer is a performance cost. Try to admit the next slot, but
        // preserve the verified extent for replay if the allocator or ledger cannot
        // provide the requested capacity.
        if self.buffered && self.records.len() == self.records.capacity() {
            let element = std::mem::size_of::<CommittedRecord>();
            // The minimum is an integral number of CommittedRecord slots large
            // enough for this record. Check both the slot count and the byte
            // multiplication before asking the ledger for more array space.
            let minimum = self
                .records
                .len()
                .checked_add(1)
                .and_then(|count| count.checked_mul(element))
                .ok_or_else(|| std::io::Error::other("recovery batch capacity overflow"))?;
            let preferred = minimum.saturating_mul(2);
            // Request spare slots as an optimization while allowing a smaller
            // grant. The exact capacity returned by the allocator must also fit
            // the bytes that the ledger admitted.
            match self.array.grow_to(minimum, preferred, "operation_scratch") {
                Ok(bytes) => {
                    let capacity = bytes / element;
                    if self
                        .records
                        .try_reserve_exact(capacity - self.records.len())
                        .is_err()
                        || self.records.capacity() > capacity
                    {
                        // An oversized allocation is as unusable as an allocation
                        // failure. Discard the whole buffer, including earlier
                        // records; a second pass must then reconstruct the batch
                        // from its first offset.
                        self.discard_buffer();
                    } else {
                        self.array.retain(self.records.capacity() * element);
                    }
                }
                // Among ledger errors, only ResourceExhausted falls back to
                // replay. Other ledger errors propagate.
                Err(error) if error.code() == ErrorCode::ResourceExhausted => {
                    self.discard_buffer();
                }
                Err(error) => return Err(error),
            }
        }
        // Retain the record with both its parsed JSON and native-field credit. In
        // replay mode none of these allocations must survive merely to remember
        // the record's physical position.
        if self.buffered {
            self.records.push(record);
            self.work.merge(json);
            self.work.merge_workspace(strings);
        } else {
            drop(record);
        }
        Ok(())
    }

    pub(super) fn visit(
        &mut self,
        file: &mut File,
        batch_id: u64,
        commit_offset: u64,
        context: &JsonOperationContext,
        visit: &mut dyn FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<()> {
        if self.buffered {
            // Deliver buffered records in original log order only after the
            // enclosing scanner accepts their commit. The common reservation lets
            // each visitor adopt the memory it retains in the catalog.
            for record in self.records.drain(..) {
                visit(record, &mut self.work)?;
            }
        } else {
            // Re-read the entire saved range on the same descriptor. Because the
            // buffer is discarded as a unit, a record seen on the initial scan is
            // never delivered once from memory and again from replay.
            self.replay(file, batch_id, commit_offset, context, visit)?;
        }
        // Start the next pending batch only after every visitor call succeeds. A
        // visitor failure propagates to the scanner and must not be followed by
        // continuing with a partially delivered batch.
        *self = Self::new(context);
        Ok(())
    }

    fn replay(
        &self,
        file: &mut File,
        batch_id: u64,
        commit_offset: u64,
        context: &JsonOperationContext,
        visit: &mut dyn FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<()> {
        // Replay borrows the scanner's cursor. Restore it on success so the
        // outer scan can continue from its position after commit validation.
        // That saved position becomes critical_end_offset. Bound the reread by the
        // start of the commit frame so replay cannot consume the commit as data.
        let resume = file.stream_position()?;
        let mut offset = self.first_offset;
        // Visit exactly the count admitted by the commit. Each header comes from
        // the tracked file range and identifies the next ordinal in that same
        // batch, so scanning stops on the first disagreement rather than searching
        // for a replacement frame.
        for index in 0..self.count {
            file.seek(SeekFrom::Start(offset))?;
            let mut header = [0; RECORD_HEADER_LEN];
            file.read_exact(&mut header)?;
            // Validate the fixed header before trusting any claimed payload
            // length. A short header read is an I/O failure, and a complete header
            // whose digest disagrees is a replay checksum error.
            if checksum32(&header[..46]) != read_u32(&header[46..50]) {
                return Err(Error::new(
                    ErrorCode::ChecksumMismatch,
                    "critical record changed during recovery replay",
                ));
            }
            let record_type = read_u16(&header[4..6]);
            // Build both absolute boundaries with checked arithmetic. An overflow
            // in either offset is equivalent to a range outside the batch, and a
            // record ending at the commit's start is the last possible data frame.
            let payload_offset = offset.checked_add(RECORD_HEADER_LEN as u64);
            let len = read_u64(&header[34..42]);
            let end = payload_offset.and_then(|start| start.checked_add(len));
            let sequence_no = read_u64(&header[24..32]);
            // The first pass established these identities before accepting the
            // commit. Check them again against the original batch and ordinal
            // before giving a reread record to the visitor.
            if &header[..4] != b"CFRL"
                || read_u16(&header[6..8]) != 1
                || read_u16(&header[32..34]) != RECORD_HEADER_LEN as u16
                || len.checked_add(RECORD_HEADER_LEN as u64) != Some(read_u64(&header[8..16]))
                || !(COLLECTION_CREATE..=COMPACT_MARKER).contains(&record_type)
                || record_type == COMMIT
                // Compare both the batch ID and exact sequence for this ordinal. A
                // checksummed frame from another batch or a frame duplicated
                // within this one cannot earn another visitor call.
                || read_u64(&header[16..24]) != batch_id
                || Some(sequence_no) != self.first_sequence.checked_add(index as u64)
                || end.map_or(true, |end| end > commit_offset)
            {
                return Err(Error::new(
                    ErrorCode::FileFormatError,
                    "invalid critical record during recovery replay",
                ));
            }
            let checksum = u64::from(read_u32(&header[42..46]));
            // Preserve the original frame and payload positions in the locator.
            // The visitor can retain that locator for a later primary read without
            // retaining the size of the whole recovery batch in memory.
            let record_ref = RecordRef {
                offset,
                record_type: record_type_name(record_type),
                batch_id,
                sequence_no,
                payload: PayloadRef {
                    offset: payload_offset.unwrap(),
                    len,
                    checksum,
                },
            };
            // Read a summary within its checked payload range. Text and vector
            // materialization can be deferred while the type, metadata and payload
            // checksum are still verified on this pass.
            let mut strings = context.workspace_reservation();
            let (payload, json) = payload::read_binary_payload_in(
                file,
                record_type,
                &record_ref.payload,
                false,
                JsonInputKind::StoredJson,
                context,
                Some(&mut strings),
            )
            .map_err(|error| {
                error
                    .with_detail("record_type", record_type_name(record_type))
                    .with_detail("record_offset", offset)
            })?;
            // Transfer only the memory the visitor elects to retain. Each record
            // has an independent reservation, so disposing of a scanned summary
            // frees its allocations before replay loads the next one.
            let mut work = RecordReservation::new(json, strings);
            visit(
                CommittedRecord {
                    payload,
                    record_ref,
                    payload_checksum: checksum,
                },
                &mut work,
            )?;
            // The next frame starts at the declared and verified end even if the
            // decode used a different underlying cursor. Move the tracked boundary
            // only after the visitor succeeds.
            offset = end.unwrap();
        }
        // Reaching the expected count is insufficient if a length changed during
        // replay. The byte after the last data frame must still be the commit we
        // matched.
        if offset != commit_offset {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "recovery batch range changed before replay completed",
            ));
        }
        // Only a successful pass restores the outer scan's position. Once replay
        // fails after visiting some records, the caller must abandon that recovery
        // attempt and any partially built view.
        file.seek(SeekFrom::Start(resume))?;
        Ok(())
    }
}
