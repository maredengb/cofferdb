use super::*;
#[cfg(test)]
use crate::json::JsonInputKind;
use crate::json::{JsonOperationContext, JsonReservation, WorkspaceReservation};
#[cfg(test)]
use crate::temporary::create_budgeted_temporary_file_in;
use crate::temporary::{
    create_budgeted_temporary_file_reclaiming, TempFileGuard, TemporaryDiskBudget, TemporaryFile,
};
use std::sync::Arc;

// Staging keeps raw binary payloads in one file and fixed-size location entries in
// another. The directory can therefore be scanned to plan commits without reading
// all the bodies. Only the group selected for the next batch needs decoded
// summaries in memory.
mod locations;
use locations::{LocationReader, StagedRecord};

// A completed staging pair. Both files remain private to this owner while batches
// borrow the payload source. Moving the logical cursor only when a batch is
// returned allows a later call to revisit a record that did not fit.
pub(crate) struct PreparedRecords {
    // Encoded payload bodies laid out consecutively, without main-log frame headers
    // or commit records.
    source: TemporaryFile,
    // Owns the temporary payload name and its disk reservation for as long as
    // source may be borrowed.
    _temporary: TempFileGuard,
    // Companion file of fixed-width entries recording each payload's kind, byte
    // range and checksum in source order.
    locations: TemporaryFile,
    // Owns the location file's temporary name and disk charge independently of the
    // payload file.
    _locations_temporary: TempFileGuard,
    // Final length of the payload file in bytes. Bounds every range decoded from a
    // location entry.
    source_bytes: u64,
    // Zero-based ordinal of the first location not yet returned in a prepared
    // batch. Read-ahead in the location file does not advance it.
    next_record: usize,
    // Number of staged records from next_record through the end. Decreased only for
    // records actually returned to the caller.
    remaining_records: usize,
    // Operation that charges directory reads, decoded summaries and append
    // workspace while staged records are consumed.
    context: JsonOperationContext,
}

// Builds the pair one record at a time. A record becomes available for planning
// only after both its complete body and its location entry have been written.
pub(crate) struct PreparedRecordsBuilder {
    // Writer for encoded payload bytes. Its current position gives the starting
    // offset of the next record.
    writer: StagingWriter,
    // Writer for the fixed-width location entries that refer into writer's file.
    locations: StagingWriter,
    // Count of records whose location write has succeeded, used to set the finished
    // reader's bounds.
    record_count: usize,
    // Per-file byte allowance passed to both writers. Their temporary file guards
    // separately share the caller's temporary-disk quota.
    byte_limit: u64,
    // Set before an attempted push and cleared only after the body, location and
    // count all succeed. A failed builder cannot yield a readable partial pair.
    failed: bool,
    // Operation used when encoding a source record and reserving temporary buffers.
    context: JsonOperationContext,
}

// One decoded summary accompanied by the location of the exact bytes destined for
// the main log. Catalog preparation can use the summary while the writer streams
// large fields directly from staging.
pub(super) struct PreparedRecord {
    // Decoded record summary retained for validation and catalog application. Large
    // payload fields can stay in the source file.
    pub(super) payload: RecordPayload,
    // Absolute byte offset, encoded length and checksum of this record's body
    // within the staging payload file.
    pub(super) source: PayloadRef,
    // Native workspace charged for this summary's decoded fields. Used to return
    // exactly its portion when a split leaves it for the next batch.
    string_bytes: usize,
}

// Classification stored alongside a staged range so planning can keep a replacement
// together without first decoding every body.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchRecordKind {
    // On-disk main-record type of the staged payload.
    record_type: u16,
    // Expected put type following a document or chunk replacement tombstone. Zero
    // means this record creates no pairing requirement.
    replaces: u16,
}

impl BatchRecordKind {
    // Only document and chunk tombstones participate in the adjacent replacement
    // rule. Their target types identify the put kind that would complete a
    // replacement. Other records cannot force the following entry into the same
    // batch.
    pub(crate) fn of(record: &RecordPayload) -> Self {
        let replaces = match record {
            RecordPayload::Tombstone(t) if t.target_type == "document" => DOCUMENT_PUT,
            RecordPayload::Tombstone(t) if t.target_type == "chunk" => CHUNK_PUT,
            _ => 0,
        };
        // Keep the physical kind and the following kind that would complete a
        // replacement as separate fields. A tombstone is still encoded as a
        // tombstone even when the planner groups it with the next put.
        Self {
            record_type: record_type_id(record),
            replaces,
        }
    }

    // This is an adjacency check on record kinds. Planning uses it at a proposed
    // cut. The semantic identities of the input records are dealt with when
    // preparing and applying the batch.
    pub(crate) fn pairs_with(self, next: Self) -> bool {
        self.replaces != 0 && self.replaces == next.record_type
    }
}

// A batch borrows the finalized source file for as long as its locations may be
// copied. It separately owns the decoded summaries and reservations that will
// accompany the records into the append path.
pub(crate) struct PreparedBatch<'a> {
    // Borrowed payload descriptor. Every range in records refers to this file and
    // stays readable throughout the append.
    pub(super) source: &'a File,
    // Records in staged order. This may be a complete prefix of a requested
    // split-eligible batch.
    pub(super) records: Vec<PreparedRecord>,
    // JSON reservation accumulated while decoding metadata for the retained
    // summaries.
    pub(super) metadata_work: JsonReservation,
    // General workspace for record slots and the streaming copy buffer. Its live
    // portion can later move to the returned committed-record array.
    pub(super) records_work: WorkspaceReservation,
    // Separate workspace for native allocations retained by decoded summaries. A
    // prefix split can return the credit belonging to a deferred record.
    pub(super) strings_work: WorkspaceReservation,
    // Reserved storage for the commit timestamp. Admitting it during preparation
    // leaves the commit writer with the space needed to produce its receipt.
    pub(super) commit_timestamp: crate::json::ReservedString,
}

pub(super) const PREPARED_COPY_BYTES: usize = 64 * 1024;
// Input summaries, returned committed summaries, and the streaming copy
// buffer coexist during append. Decoded strings have a separate lease;
// record kind names are static and persistent Catalog ownership is separate.
const RECORD_SLOT_BYTES: usize =
    std::mem::size_of::<PreparedRecord>() + std::mem::size_of::<CommittedRecord>();

// Ensure an input summary can be added while also accounting for the space that its
// committed counterpart will occupy. The streaming copy buffer is shared by all
// slots and is included once in the calculation.
fn reserve_record_slot(
    records: &mut Vec<PreparedRecord>,
    work: &mut WorkspaceReservation,
    has_complete_prefix: bool,
) -> Result<()> {
    if records.len() < records.capacity() {
        // Spare vector capacity already represents admitted slots for both the
        // incoming and committed records. A new range can borrow that space
        // without negotiating the shared copy buffer again.
        return Ok(());
    }
    let bytes_for = |slots: usize| {
        // Perform the slot multiplication before adding the one fixed buffer with
        // checked arithmetic. A count too large for this process must not look
        // like a small allocation suitable for splitting.
        slots
            .checked_mul(RECORD_SLOT_BYTES)
            .and_then(|bytes| bytes.checked_add(PREPARED_COPY_BYTES))
            .ok_or_else(|| std::io::Error::other("prepared record capacity overflow"))
    };
    let minimum = records.len() + 1;
    // Until one atomic group is complete, spare slots must not strand the
    // strings needed by that group's current record or replacement Put.
    let preferred = if has_complete_prefix {
        records.capacity().saturating_mul(2).max(4)
    } else {
        minimum
    };
    // Translate the byte allowance back into whole slots. Reserve the native array
    // only after obtaining credit, then reconcile the allocator's actual capacity.
    // An allocation failure before growth leaves the original array and charge
    // available.
    let previous = work.reserved_bytes();
    let granted = work.grow_to(bytes_for(minimum)?, bytes_for(preferred)?, "batch_prepare")?;
    let slots = (granted - PREPARED_COPY_BYTES) / RECORD_SLOT_BYTES;
    // Vec interprets its request as an addition to the current length. Convert the
    // total granted slot count accordingly so spare capacity is not accounted or
    // requested twice.
    if let Err(error) = records.try_reserve_exact(slots - records.len()) {
        work.retain(previous);
        return Err(std::io::Error::other(error.to_string()).into());
    }
    // The allocator can choose a capacity larger than requested. Reconcile the
    // reservation with that actual capacity before this array is allowed to offer
    // a usable prefix to the caller.
    let actual = bytes_for(records.capacity())?;
    if let Err(error) = work.grow_to(actual, actual, "batch_prepare") {
        // An allocator may grant extra capacity. Do not return a partially
        // admitted prefix whose backing allocation is not fully charged.
        *records = Vec::new();
        work.retain(0);
        return Err(error);
    }
    work.retain(actual);
    Ok(())
}

// An import replacement stages a tombstone followed by a put. When splitting a batch
// at a resource limit, leave both for the next commit if only the tombstone fits.
// Publishing half the pair would expose a deletion even if its replacement never commits.
fn complete_prefix(records: &[PreparedRecord], next: BatchRecordKind) -> usize {
    // A zero result means nothing can be split off yet. The caller can retry after
    // reclamation. An empty prefix cannot satisfy this request.
    records.len()
        - usize::from(
            records
                .last()
                .is_some_and(|record| BatchRecordKind::of(&record.payload).pairs_with(next)),
        )
}

// Splitting must make progress and release a constraint on the following attempt. A
// resource-exhausted error can justify returning a complete prefix. For a JSON
// memory error, require credit owned by the accumulated JSON that returning the
// prefix can make available.
fn can_split_batch(
    records: &[PreparedRecord],
    next: BatchRecordKind,
    work: &JsonReservation,
    error: &Error,
) -> bool {
    complete_prefix(records, next) > 0
        && (error.code() == ErrorCode::ResourceExhausted
            || (work.reserved_bytes() > 0 && crate::json::is_memory_budget_error(error)))
}

impl PreparedRecords {
    pub(crate) fn remaining_records(&self) -> usize {
        self.remaining_records
    }

    #[cfg(test)]
    pub(crate) fn create(
        records: Vec<RecordPayload>,
        directory: &Path,
        byte_limit: u64,
    ) -> Result<Self> {
        let mut builder =
            PreparedRecordsBuilder::new(directory, byte_limit, JsonOperationContext::standalone())?;
        for record in records {
            builder.push(record)?;
        }
        builder.finish()
    }

    // Expose the kinds and main-log byte sizes needed by batch planning. Seek by
    // logical ordinal in the location file and leave the persistent cursor
    // untouched. The frame header is added here because the staging file contains
    // only payload bodies.
    pub(crate) fn records(
        &mut self,
    ) -> Result<impl Iterator<Item = Result<(BatchRecordKind, u64)>> + '_> {
        // Each new traversal starts at the logical next record, regardless of how
        // far another planning pass read ahead. Payloads stay in their own staging
        // file until the selected batch is decoded for append.
        Ok(LocationReader::new(
            &mut self.locations,
            self.next_record,
            self.remaining_records,
            self.source_bytes,
            &self.context,
        )?
        .map(|record| {
            let record = record?;
            // The kind carries the replacement relationship alongside this length.
            // Budget the framed data record here; the batch planner separately
            // allows for the one commit frame shared by the group.
            let len = record
                .source
                .len
                .checked_add(RECORD_HEADER_LEN as u64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "prepared record length overflow",
                    )
                })?;
            Ok((record.kind, len))
        }))
    }

    #[cfg(test)]
    pub(crate) fn take_batch(&mut self, count: usize) -> Result<PreparedBatch<'_>> {
        self.take_batch_reclaiming(count, || false)
    }

    // Take up to the requested number of records. If the batch cannot be admitted,
    // a split is allowed only after a complete group, so a replacement put can
    // share its tombstone's commit.
    pub(crate) fn take_batch_reclaiming(
        &mut self,
        count: usize,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<PreparedBatch<'_>> {
        self.take_batch_with_split_reclaiming(count, true, reclaim)
    }

    // Request all remaining records as one atomic group. Reclamation may make room,
    // but a smaller committable prefix would violate the caller's all-or-nothing
    // requirement.
    pub(crate) fn take_atomic_batch_reclaiming(
        &mut self,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<PreparedBatch<'_>> {
        self.take_batch_with_split_reclaiming(self.remaining_records, false, reclaim)
    }

    // Decode only the summaries needed for this append while keeping the payload
    // bodies in staging. Preparation can retry without changing the persistent
    // cursor. In split mode it can also return an earlier complete prefix instead
    // of consuming a record whose resources were refused.
    fn take_batch_with_split_reclaiming(
        &mut self,
        count: usize,
        allow_split: bool,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<PreparedBatch<'_>> {
        // Validate the requested logical range before reserving a timestamp or
        // touching the temporary files. A zero-length result could make a caller
        // that drains batches loop without advancing.
        if count == 0 || count > self.remaining_records {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "invalid prepared batch length",
            ));
        }
        // One cache reclamation opportunity is shared by admitting the timestamp
        // and decoding all requested records. Consuming it through Option prevents
        // a failed allocator or a hostile input from repeatedly forcing eviction.
        let mut reclaim = Some(reclaim);
        let commit_timestamp = crate::json::ReservedString::with_capacity(
            time::MAX_TIMESTAMP_BYTES,
            &self.context,
            "batch_prepare",
        );
        let mut work = self.context.reservation();
        // Bound the directory reader by both the first unconsumed ordinal and the
        // payload file's actual byte extent. A location beyond either cannot be
        // used to copy bytes into a primary append.
        let mut locations = LocationReader::new(
            &mut self.locations,
            self.next_record,
            self.remaining_records,
            self.source_bytes,
            &self.context,
        )?;
        let mut first = None;
        // A failed timestamp reservation can coincide with a corrupt first location
        // or payload. Verify that first source range with fixed scratch before
        // returning a capacity error or attempting reclamation. This keeps broken
        // staging observable even at zero workspace headroom.
        let commit_timestamp = match commit_timestamp {
            Ok(timestamp) => timestamp,
            Err(error) => {
                // Preserve source corruption/IO precedence even at zero headroom.
                let staged = locations.next().ok_or_else(|| {
                    Error::new(ErrorCode::FileFormatError, "prepared locations ended early")
                })??;
                // Discard the bytes while checking the complete locator. If that
                // check fails, it supplies the error a retry would encounter even
                // with enough memory for a timestamp.
                copy_payload(
                    &self.source,
                    &staged.source,
                    &mut std::io::sink(),
                    &mut [0; PREPARED_COPY_BYTES],
                )?;
                // Only a memory-budget refusal can spend the caller's eviction
                // opportunity. Other timestamp errors and an unsuccessful eviction
                // return at this point without advancing the persistent batch
                // position.
                if !crate::json::is_memory_budget_error(&error)
                    || !reclaim.take().is_some_and(|reclaim| reclaim())
                {
                    return Err(error);
                }
                // The validated location was already consumed. Keep it for the
                // normal decode path rather than skipping the first record.
                first = Some(staged);
                crate::json::ReservedString::with_capacity(
                    time::MAX_TIMESTAMP_BYTES,
                    &self.context,
                    "batch_prepare",
                )?
            }
        };
        // Keep the summary array, the summaries' owned strings, and their JSON
        // trees on separate reservations. A split can remove individual rows while
        // retaining the array used by the returned prefix.
        let mut records_work = self.context.workspace_reservation();
        let mut strings_work = self.context.workspace_reservation();
        let mut records = Vec::new();
        for _ in 0..count {
            let staged = match first.take() {
                Some(staged) => staged,
                None => locations.next().ok_or_else(|| {
                    Error::new(ErrorCode::FileFormatError, "prepared locations ended early")
                })??,
            };
            // Remember the native string charge at the beginning of this record. If
            // a later record forces a prefix cut, this gives the exact boundary
            // between previously owned fields and the record still being decoded.
            let previous_strings = strings_work.reserved_bytes();
            // The last admitted row may be a replacement tombstone whose put is
            // next. Treat only rows preceding that pair as a cut point when
            // choosing how aggressively to grow the summary array.
            let has_complete_prefix = complete_prefix(&records, staged.kind) > 0;
            let decoded = loop {
                let decoded =
                    // Reserve the destination slot before decoding the summary and
                    // taking ownership of its fields. A retry can then re-read
                    // this same range without inserting a duplicate record.
                    reserve_record_slot(&mut records, &mut records_work, has_complete_prefix)
                        .and_then(|()| {
                            payload::read_staged_summary_in(
                                &mut self.source,
                                staged.kind.record_type,
                                &staged.source,
                                &self.context,
                                &mut strings_work,
                            )
                        });
                // Read-ahead in the location file is optional. On a limit error,
                // shed that buffer first and retry the current staged range.
                // Rebuilding the directory reader preserves its logical position
                // even when the old reader had already fetched later entries.
                if locations.is_buffered()
                    && decoded.as_ref().is_err_and(|error| {
                        crate::json::is_limit_error(error)
                            || error.code() == ErrorCode::ResourceExhausted
                    })
                {
                    // Drop directory read-ahead before considering eviction of
                    // database caches. The current location is already owned, and
                    // the replacement reader starts at the next unread logical
                    // location.
                    locations = locations.into_unbuffered()?;
                    continue;
                }
                break match decoded {
                    Err(error) => {
                        // A summary decode may run out of memory before reaching
                        // damaged payload bytes. Check the whole staged payload
                        // without retaining it before deciding whether to split the
                        // batch or retry under a smaller memory footprint.
                        let checked = copy_payload(
                            &self.source,
                            &staged.source,
                            &mut std::io::sink(),
                            &mut [0; PREPARED_COPY_BYTES],
                        );
                        // Prefer an error from checking the staged bytes
                        // themselves. A bad checksum or a short file cannot be
                        // repaired by returning a smaller batch or retrying the
                        // allocation.
                        let error = checked.err().unwrap_or(error);
                        // Keep the existing bounded-prefix fallback. If that
                        // cannot proceed, reclaim once without discarding the
                        // already decoded records or their reservations.
                        // In split mode prefer handing the caller a usable prefix
                        // when releasing its memory can help progress. The atomic
                        // path has no such option and can use the remaining
                        // eviction attempt on the whole request.
                        if !(allow_split && can_split_batch(&records, staged.kind, &work, &error))
                            && crate::json::is_memory_budget_error(&error)
                            && reclaim.take().is_some_and(|reclaim| reclaim())
                        {
                            continue;
                        }
                        Err(error)
                    }
                    result => result,
                };
            };
            match decoded {
                Ok((payload, retained)) => {
                    // The location file supplies the kind used for planning, while
                    // the decoded tombstone determines its actual replacement kind.
                    // Compare them before making the summary available to a caller
                    // that may commit the planned prefix.
                    if BatchRecordKind::of(&payload) != staged.kind {
                        return Err(Error::new(
                            ErrorCode::FileFormatError,
                            "prepared record kind differs from its location",
                        ));
                    }
                    // Keep the source locator with the decoded catalog summary.
                    // The eventual writer can copy the large encoded body from
                    // staging, including when the summary deliberately contains no
                    // text or coordinates.
                    records.push(PreparedRecord {
                        payload,
                        source: staged.source,
                        string_bytes: strings_work.reserved_bytes() - previous_strings,
                    });
                    // Transfer credit for this row's retained metadata into the
                    // batch only after its summary passed the kind check. The
                    // input location is still local to this attempt until the
                    // persistent cursor advances.
                    work.merge(retained);
                }
                Err(error) => {
                    let prefix = complete_prefix(&records, staged.kind);
                    // Neither workspace pressure nor JSON admission may split
                    // a replacement tombstone from its following Put.
                    if allow_split && can_split_batch(&records, staged.kind, &work, &error) {
                        // The boundary may move back one already-decoded record to
                        // keep a replacement whole. Drop those summaries, then
                        // shrink their native string credit using the amounts
                        // recorded when each one was decoded. The next batch will
                        // charge them again when it rereads them.
                        let removed_strings: usize = records[prefix..]
                            .iter()
                            .map(|record| record.string_bytes)
                            .sum();
                        // Truncating can remove a previously decoded tombstone as
                        // well as defer the failing put. Retain native credit only
                        // for the rows actually returned, with both members
                        // available to the next batch.
                        records.truncate(prefix);
                        strings_work.retain(previous_strings - removed_strings);
                        break;
                    }
                    return Err(error);
                }
            }
        }
        // Only the returned prefix is consumed. A split may have read a location for
        // the following record, but the next call must start with the first record
        // absent from this batch.
        self.next_record += records.len();
        self.remaining_records -= records.len();
        // The returned batch borrows the payload file. Rust keeps the owner alive
        // while the append reads those bytes by locator, even though the cursor
        // already points past the returned records.
        Ok(PreparedBatch {
            source: &self.source,
            records,
            metadata_work: work,
            records_work,
            strings_work,
            commit_timestamp,
        })
    }
}

impl PreparedRecordsBuilder {
    #[cfg(test)]
    pub(crate) fn new(
        directory: &Path,
        byte_limit: u64,
        context: JsonOperationContext,
    ) -> Result<Self> {
        Self::new_in(
            directory,
            byte_limit,
            context,
            Arc::new(TemporaryDiskBudget::new(byte_limit)),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_in(
        directory: &Path,
        byte_limit: u64,
        context: JsonOperationContext,
        budget: Arc<TemporaryDiskBudget>,
    ) -> Result<Self> {
        Self::new_in_reclaiming(directory, byte_limit, context, budget, || false)
    }

    // Both temporary files charge a common disk budget, but each needs its own file
    // handle and guard. If creating the locations file fails, the payload writer
    // assembled for the first field is dropped with its reservations.
    pub(crate) fn new_in_reclaiming(
        directory: &Path,
        byte_limit: u64,
        context: JsonOperationContext,
        budget: Arc<TemporaryDiskBudget>,
        mut reclaim: impl FnMut() -> bool,
    ) -> Result<Self> {
        Ok(Self {
            writer: StagingWriter::new(
                directory,
                "import-records",
                byte_limit,
                budget.clone(),
                &context,
                &mut reclaim,
            )?,
            // The small location stream and the potentially much larger payload
            // stream have independent limits and file handles. Both draw temporary
            // disk space from the shared budget passed to this builder.
            locations: StagingWriter::new(
                directory,
                "import-locations",
                byte_limit,
                budget,
                &context,
                &mut reclaim,
            )?,
            record_count: 0,
            byte_limit,
            failed: false,
            context,
        })
    }

    #[cfg(test)]
    pub(crate) fn push(&mut self, record: RecordPayload) -> Result<()> {
        self.push_retained(record, &mut self.context.reservation())
    }

    // Stage one payload and its corresponding directory entry. The guard being set
    // on entry is deliberate. A failure can leave a partial write in either private
    // file, so a later push cannot safely resume from the old logical record count.
    pub(crate) fn push_retained(
        &mut self,
        record: RecordPayload,
        work: &mut JsonReservation,
    ) -> Result<()> {
        if self.failed {
            return Err(failed_builder());
        }
        self.failed = true;
        // Accept only data records that an import can stage. The eventual main-log
        // writer chooses the commit ID, sequence numbers and batch checksum. A
        // supplied commit or compaction marker cannot be embedded as one of these
        // inputs.
        if matches!(
            record,
            RecordPayload::Commit(_) | RecordPayload::CompactMarker(_)
        ) {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "import staging cannot contain commit or compact markers",
            ));
        }
        // Share unused reservation credit so the two files cannot strand each other's capacity.
        self.locations.transfer_credit(&mut self.writer, u64::MAX)?;
        // Save the payload cursor before encoding. The per-record digest records
        // the precise byte count and CRC actually emitted. Together they supply a
        // bounded, checkable location without retaining the serialized body.
        let offset = self.byte_limit - self.writer.remaining;
        // Encode under the operation context while the input still owns its JSON.
        // The resulting length and checksum describe the bytes the staging writer
        // accepted rather than an estimate from the in-memory record.
        self.writer
            .encode_record(&record, &self.context, work, self.byte_limit)?;
        let source = PayloadRef {
            offset,
            len: self.writer.digest.len,
            checksum: u64::from(self.writer.digest.crc.finish()),
        };
        // Only fixed-size locations survive staging; the current batch is decoded on demand.
        let kind = BatchRecordKind::of(&record);
        // Extract the pairing kind before dropping the typed record. Once its
        // encoding is complete the builder needs only this classification and the
        // physical locator to construct the directory entry.
        drop(record);
        work.retain(0);
        // Publish the fixed-size location only after its matching body is complete.
        // Its record count is the last piece of state to advance. Any error in this
        // interval leaves the failed flag set and prevents callers from taking a
        // mismatched pair.
        let encoded = StagedRecord { kind, source }.encode()?;
        self.writer
            // Transfer only enough unused disk credit to cover this location
            // entry. The next payload can subsequently recover remaining unused
            // credit from the location writer.
            .transfer_credit(&mut self.locations, encoded.len() as u64)?;
        let written = self.locations.write_all(&encoded).map_err(Error::from);
        self.locations.checked(written, self.byte_limit)?;
        self.record_count = self.record_count.checked_add(1).ok_or_else(|| {
            Error::new(ErrorCode::FileFormatError, "prepared record count overflow")
        })?;
        self.failed = false;
        Ok(())
    }

    // Flush both writers and move their guards into the reader. Release any surplus
    // disk credit while keeping the bytes backing these two files reserved for
    // their lifetime. Only a builder with no incomplete record may be consumed.
    pub(crate) fn finish(self) -> Result<PreparedRecords> {
        if self.failed {
            return Err(failed_builder());
        }
        // Record the logical payload-file length before consuming its writer.
        // Readers will reject a forged location whose offset or range extends
        // beyond these staged bytes.
        let source_bytes = self.byte_limit - self.writer.remaining;
        let (source, temporary) = self.writer.finish()?;
        // Both files must finish successfully before a reader can escape. Failure
        // finalizing the location file also drops the completed payload file and
        // its temporary guard.
        let (locations, locations_temporary) = self.locations.finish()?;
        Ok(PreparedRecords {
            source,
            _temporary: temporary,
            locations,
            _locations_temporary: locations_temporary,
            source_bytes,
            next_record: 0,
            remaining_records: self.record_count,
            context: self.context,
        })
    }
}

fn failed_builder() -> Error {
    Error::new(
        ErrorCode::FileFormatError,
        "cannot reuse failed import staging",
    )
}

impl PreparedBatch<'_> {
    pub(crate) fn context(&self) -> &JsonOperationContext {
        self.records_work.context()
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    // Expose exactly the prepared prefix as immutable summaries. Catalog planning
    // and source copying must keep the same ordinal sequence when the primary
    // writer later supplies the committed locators.
    pub(crate) fn records(&self) -> impl ExactSizeIterator<Item = &RecordPayload> {
        self.records.iter().map(|record| &record.payload)
    }

    // Borrow the summaries and their metadata reservation together. Catalog
    // preparation can transfer already-admitted JSON while still looking at the
    // corresponding records, without constructing a second owning copy of the
    // entire batch.
    pub(crate) fn records_and_json(
        &mut self,
    ) -> (
        impl ExactSizeIterator<Item = &RecordPayload> + Clone,
        &mut JsonReservation,
    ) {
        // Catalog preparation traverses summaries while moving their existing
        // metadata credit. The cloned iterator can replay the summary order
        // without cloning the record payloads or transferring the reservation
        // itself.
        (
            self.records.iter().map(|record| &record.payload),
            &mut self.metadata_work,
        )
    }

    // Before a critical append, verify every source range with the same streaming
    // copy used by the writer. A verification sink discards the bytes, and all
    // records reuse one scratch buffer. A bad checksum is then discoverable before
    // the destination is truncated.
    pub(super) fn validate(&self) -> Result<()> {
        let mut buffer = [0u8; PREPARED_COPY_BYTES];
        // Verify every range before this batch can change the destination. Even a
        // later bad range must fail here, while all earlier bytes have only
        // reached the verification sink.
        for record in &self.records {
            copy_payload(
                self.source,
                &record.source,
                &mut std::io::sink(),
                &mut buffer,
            )?;
        }
        Ok(())
    }
}

// A writer tracks two bounds. The file has a logical byte allowance, and its guard
// holds credits from the shared temporary-disk budget. Surplus credit in one writer
// can be moved to its companion without briefly releasing the shared capacity.
struct StagingWriter {
    // Writable temporary descriptor containing bytes accepted so far.
    output: TemporaryFile,
    // Owns the temporary name and all disk credit held for this file, including
    // unused credit tracked below.
    temporary: TempFileGuard,
    // Byte count and CRC of the bytes accepted by this writer since its digest was
    // last reset. The payload writer resets it before each record attempt.
    digest: PayloadDigest,
    // Bytes still available under this file's own limit. Decreases by bytes
    // actually written.
    remaining: u64,
    // Size of a write refused by the per-file limit. checked uses it to reconstruct
    // the requested total after a generic I/O error.
    rejected_write_bytes: Option<u64>,
    // Original shared temporary-disk admission error, retained across the
    // std::io::Write interface so callers receive the precise resource diagnosis.
    budget_error: Option<Error>,
    // Reserved disk bytes not yet occupied by written content. Transferable to the
    // companion writer, refundable on finish.
    disk_credit: u64,
}

impl StagingWriter {
    // Move only idle disk credit. Previous bytes remain charged to the guard for
    // the file that contains them. Transfer the shared lease first, then adjust the
    // writers' local counters after the transfer succeeds.
    fn transfer_credit(&mut self, destination: &mut Self, requested: u64) -> Result<()> {
        // Cap the transfer at credit that has no corresponding file bytes yet.
        // Occupied capacity remains owned by the source guard even if its
        // companion needs more staging space.
        let bytes = self.disk_credit.min(requested);
        self.temporary
            .transfer_disk_to(&mut destination.temporary, bytes)?;
        self.disk_credit -= bytes;
        destination.disk_credit += bytes;
        Ok(())
    }

    fn new(
        directory: &Path,
        label: &str,
        byte_limit: u64,
        budget: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
        reclaim: &mut dyn FnMut() -> bool,
    ) -> Result<Self> {
        // Construction acquires the file and its cleanup guard together. Optional
        // memory reclamation can make the file's metadata admissible, while the
        // shared disk allowance still applies independently to every later write.
        let (file, temporary) = create_budgeted_temporary_file_reclaiming(
            directory,
            label,
            "tmp",
            budget,
            context,
            "batch_prepare",
            reclaim,
        )?;
        // The logical file limit starts at the full allowance. Shared disk credit
        // starts at zero and is acquired before writes, so simply opening two
        // companion files does not reserve either file's maximum size.
        Ok(Self {
            output: file,
            temporary,
            digest: PayloadDigest::default(),
            remaining: byte_limit,
            rejected_write_bytes: None,
            budget_error: None,
            disk_credit: 0,
        })
    }

    // The Write trait can carry only an I/O error. Recover the recorded quota error
    // if the failure was actually due to disk admission. For the per-file ceiling,
    // compute the requested total in a wider integer so the diagnostic itself
    // cannot wrap.
    fn checked<T>(&mut self, result: Result<T>, byte_limit: u64) -> Result<T> {
        if let Some(error) = self.budget_error.take() {
            return Err(error);
        }
        // Calculate the refused total from the already accepted prefix plus the
        // latest request. The diagnostic can exceed u64 even though the per-file
        // limit and offsets fit in u64.
        if let Some(bytes) = self.rejected_write_bytes {
            return Err(Error::resource_exhausted(
                "temporary_disk",
                "temporary_staging",
                byte_limit,
                u128::from(byte_limit - self.remaining) + u128::from(bytes),
            ));
        }
        result
    }

    // Encode one private payload, optionally buffering writes to improve
    // throughput. Save the file and reservation boundaries of the preceding
    // records. If optional buffering prevents JSON processing, only this unfinished
    // record may be rewound for a direct-write attempt.
    fn encode_record(
        &mut self,
        record: &RecordPayload,
        context: &JsonOperationContext,
        work: &mut JsonReservation,
        byte_limit: u64,
    ) -> Result<()> {
        let remaining = self.remaining;
        // Use accepted bytes to locate the beginning of this record. The buffer
        // may already have sent part of a failed encoding to the file, so retrying
        // from the current cursor could append a second copy after that prefix.
        let offset = byte_limit - remaining;
        let original_json = work.covered_bytes();
        let mut buffer_work = context.workspace_reservation();
        // Request the preferred buffer with a zero minimum. Direct encoding can
        // proceed while other operation owners hold most of the available
        // workspace.
        buffer_work.grow_to(0, 8192, "batch_prepare")?;
        loop {
            // Reset the digest for each whole attempt. After a rewind, its length
            // and CRC must represent the accepted bytes of the successful attempt
            // alone.
            self.digest = PayloadDigest::default();
            let mut output = BufWriter::with_capacity(buffer_work.reserved_bytes(), &mut *self);
            // The optional buffer can be used only if its native capacity fits the
            // credit that was granted. Discard the buffer and switch to direct
            // writes when the allocator rounds up. Refusing all staging in this
            // case would turn an optimization into a requirement.
            if output.capacity() > buffer_work.reserved_bytes() {
                let (writer, buffer) = output.into_parts();
                drop(buffer);
                buffer_work.retain(0);
                output = BufWriter::with_capacity(0, writer);
            }
            let encoded = codec::encode_retained_into(record, &mut output, context, work)
                .and_then(|()| output.flush().map_err(Error::from));
            // A failed record must not be flushed again by BufWriter's destructor.
            let (_, buffer) = output.into_parts();
            drop(buffer);
            let encoded = self.checked(encoded, byte_limit);
            // Retry only while there is buffer credit to release. A quota or file
            // error that survives checked() has no alternate buffer path here, and
            // a failed direct retry returns to the caller.
            if buffer_work.reserved_bytes() > 0
                && encoded.as_ref().is_err_and(crate::json::is_limit_error)
            {
                // Only this private, not-yet-indexed record is rolled back.
                self.output.set_len(offset)?;
                self.output.seek(SeekFrom::Start(offset))?;
                // Quota covering the truncated bytes is still owned by the guard.
                // Move it back to the unused-credit counter so the retry can reuse
                // it. Restore the per-file remaining count and release any extra
                // JSON and optional output-buffer charges before encoding the same
                // record again.
                self.disk_credit = self
                    .disk_credit
                    .checked_add(remaining - self.remaining)
                    .ok_or_else(|| std::io::Error::other("staging disk credit overflow"))?;
                self.remaining = remaining;
                work.retain(original_json.min(work.reserved_bytes()));
                buffer_work.retain(0);
                continue;
            }
            return encoded;
        }
    }

    // Flush data to the private descriptor and give back only unspent disk credit.
    // The file and its guard move together, so later reads continue to be backed by
    // the occupied portion of the reservation.
    fn finish(mut self) -> Result<(TemporaryFile, TempFileGuard)> {
        self.output.flush()?;
        self.temporary.release_disk(self.disk_credit);
        Ok((self.output, self.temporary))
    }
}

impl Write for StagingWriter {
    // Admit the requested write before allowing it to reach the descriptor. First
    // check the file's logical maximum, then fill any gap in its shared disk
    // credit. A quota refusal is saved so checked can return its structured error.
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            self.rejected_write_bytes = Some(bytes.len() as u64);
            return Err(std::io::Error::other("import staging budget exhausted"));
        }
        // The preferred reserve may admit more than this write needs. Unspent
        // bytes stay available on the guard for later records or transfer to a
        // companion file, while the logical file ceiling still limits every call.
        if bytes.len() as u64 > self.disk_credit {
            let needed = bytes.len() as u64 - self.disk_credit;
            match self.temporary.reserve_disk_buffer(needed, 64 * 1024) {
                Ok(granted) => self.disk_credit += granted,
                Err(error) => {
                    self.budget_error = Some(error);
                    return Err(std::io::Error::other(
                        "import temporary disk budget exhausted",
                    ));
                }
            }
        }
        // An underlying write can accept only a prefix. Debit and hash exactly that
        // prefix. The caller may send the remaining bytes again, so charging the
        // original request size here would overstate both the reservation spent and
        // the checksum length.
        let count = self.output.write(bytes)?;
        self.disk_credit -= count as u64;
        self.digest.write_all(&bytes[..count])?;
        self.remaining -= count as u64;
        Ok(count)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.output.flush()
    }
}

// Copy one previously recorded range, either to a verification sink or to its
// destination frame. A fixed buffer bounds scratch for large fields. The checksum
// is tested at the end, so an error can follow bytes already delivered to the
// output.
pub(super) fn copy_payload(
    source: &File,
    payload: &PayloadRef,
    output: &mut impl Write,
    buffer: &mut [u8],
) -> Result<()> {
    // Reject a zero-length transfer buffer before iterating. If the payload is
    // nonempty, each loop iteration needs scratch for at least one byte to make
    // progress.
    if buffer.is_empty() {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            "prepared payload copy buffer is empty",
        ));
    }
    let mut remaining = payload.len;
    let mut offset = payload.offset;
    let mut crc = Crc32c::default();
    // Use positioned reads on Unix to avoid disturbing another read of the same
    // staging file. On other systems explicitly seek to the absolute next offset on
    // the cloned descriptor before each chunk.
    #[cfg(not(unix))]
    let mut source = source.try_clone()?;
    // Accept the range in chunks with exact reads. Encountering EOF inside that
    // range fails even if the prefix matches the requested CRC so far.
    while remaining != 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            source.read_exact_at(&mut buffer[..count], offset)?;
        }
        #[cfg(not(unix))]
        {
            source.seek(SeekFrom::Start(offset))?;
            source.read_exact(&mut buffer[..count])?;
        }
        // Checksum the same bytes that are written and advance the absolute source
        // offset with checked arithmetic. After consuming exactly the recorded
        // length, compare the final CRC with the value saved on first encoding.
        crc.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
        offset = offset.checked_add(count as u64).ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "prepared payload offset overflow",
            )
        })?;
        remaining -= count as u64;
    }
    // Compare CRC after exactly the recorded length, including an empty body. The
    // output may already have received the full range when the comparison fails. A
    // verification caller discards it, and a writer must abandon its private or
    // uncommitted output.
    if u64::from(crc.finish()) != payload.checksum {
        return Err(Error::new(
            ErrorCode::ChecksumMismatch,
            "prepared import payload checksum mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCUMENT_IDENTITY_BYTES: usize = 2 * 20 + 2 * 20;
    const TOMBSTONE_IDENTITY_BYTES: usize = 20 + 20 + "document".len();

    fn management_bytes(directory: &Path) -> usize {
        2 * crate::temporary::test_file_management_bytes(directory)
    }

    fn assert_available_credit(context: &JsonOperationContext, expected: usize, limit: usize) {
        let mut work = context.workspace_reservation();
        work.grow_to(expected, expected, "operation_scratch")
            .unwrap();
        assert!(work
            .grow_to(expected + 1, expected + 1, "operation_scratch")
            .is_err());
        assert!(expected <= limit);
    }

    #[test]
    #[cfg(unix)]
    fn staging_file_creation_reclaims_before_copying_and_preserves_the_first_owner() {
        use std::os::unix::fs::MetadataExt;
        let dir = directory("staging-creation-reclaim");
        let management = management_bytes(&dir) / 2;
        let limit = 256 * 1024;
        for target in 1..=2 {
            let context = JsonOperationContext::new(Arc::new(
                crate::json::WorkLedger::with_limits(2 * limit, limit, limit),
            ));
            let occupied = limit - target * management - 128 + 1;
            let mut cache = Some(context.cache_reservation(occupied as u128, limit).unwrap());
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let mut calls = 0;
            let mut first = None;
            let mut builder = PreparedRecordsBuilder::new_in_reclaiming(
                &dir,
                4096,
                context.clone(),
                disk.clone(),
                || {
                    calls += 1;
                    assert_available_credit(&context, management + 127, limit);
                    assert_eq!(fs::read_dir(&dir).unwrap().count(), target - 1);
                    if target == 2 {
                        let path = fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
                        let metadata = fs::metadata(&path).unwrap();
                        first = Some((path, metadata.dev(), metadata.ino()));
                    }
                    drop(cache.take());
                    true
                },
            )
            .unwrap();
            assert_eq!(calls, 1);
            assert_eq!(builder.record_count, 0);
            assert_eq!(disk.used(), 0);
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
            if let Some((path, device, inode)) = first {
                assert_eq!(builder.writer.temporary.path(), path.as_path());
                let metadata = builder.writer.output.metadata().unwrap();
                assert_eq!((metadata.dev(), metadata.ino()), (device, inode));
            }
            builder.push(document()).unwrap();
            let mut prepared = builder.finish().unwrap();
            assert_eq!(
                fs::read(prepared._temporary.path()).unwrap(),
                codec::encode(&document()).unwrap()
            );
            assert_eq!(prepared.records().unwrap().count(), 1);
            assert_available_credit(&context, limit - 2 * management, limit);
            drop(prepared);
            assert_available_credit(&context, limit, limit);
            assert_eq!(disk.used(), 0);
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn staging_second_file_admission_is_atomic_and_keeps_original_context() {
        let dir = directory("second-file-admission");
        let management = management_bytes(&dir);
        let peak = management + 128;
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        for available in [peak - 1, peak, peak + 1] {
            let limit = peak * 2;
            let ledger = Arc::new(crate::json::WorkLedger::with_limits(
                2 * limit,
                limit,
                limit,
            ));
            let context = JsonOperationContext::new(ledger.clone());
            let sibling = JsonOperationContext::new(ledger);
            let mut busy = sibling.workspace_reservation();
            busy.grow_to(limit - available, limit - available, "operation_scratch")
                .unwrap();
            let result = PreparedRecordsBuilder::new_in(&dir, 4096, context.clone(), disk.clone());
            if available < peak {
                let error = result.err().unwrap();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, value) in [
                    ("resource", "memory".to_owned()),
                    ("operation", "batch_prepare".to_owned()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", (limit + 1).to_string()),
                ] {
                    assert!(
                        error.detail().contains(&(key.to_owned(), value)),
                        "{error:?}"
                    );
                }
                assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
            } else {
                let prepared = result.unwrap().finish().unwrap();
                drop(context);
                assert_available_credit(&sibling, available - management, limit);
                assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
                drop(prepared);
            }
            assert_available_credit(&sibling, available, limit);
            drop(busy);
            assert_available_credit(&sibling, limit, limit);
            assert_eq!(disk.used(), 0);
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn batch_reclamation_at_zero_headroom_keeps_locations_and_returns_credit() {
        let dir = directory("reclaim-first-location");
        let limit = 256 * 1024;
        for seed in 0..100 {
            for release in [0, 1, 128, limit] {
                let context = JsonOperationContext::new(Arc::new(
                    crate::json::WorkLedger::with_limits(limit, limit, limit),
                ));
                let mut builder =
                    PreparedRecordsBuilder::new(&dir, 1_048_576, context.clone()).unwrap();
                let input = [document(), replacement(), large_document(2, 100 + seed)];
                let expected = input
                    .iter()
                    .map(|record| codec::encode(record).unwrap())
                    .collect::<Vec<_>>();
                for record in input {
                    builder.push(record).unwrap();
                }
                let mut prepared = builder.finish().unwrap();
                let available = seed % time::MAX_TIMESTAMP_BYTES;
                let mut cached = context
                    .cache_reservation((limit - available - management_bytes(&dir)) as u128, limit)
                    .unwrap();
                let mut calls = 0;
                let result = prepared.take_batch_reclaiming(3, || {
                    calls += 1;
                    cached.retain(cached.reserved_bytes().saturating_sub(release));
                    release != 0
                });
                assert_eq!(calls, 1);
                if release == limit {
                    let batch = result.unwrap();
                    assert_eq!(batch.len(), 3);
                    for (record, expected) in batch.records.iter().zip(&expected) {
                        assert_eq!(codec::encode(&record.payload).unwrap(), *expected);
                    }
                    batch.validate().unwrap();
                    drop(batch);
                    assert_eq!(prepared.next_record, 3);
                    assert_eq!(prepared.remaining_records, 0);
                } else {
                    let error = result.err().unwrap();
                    assert!(crate::json::is_memory_budget_error(&error));
                    assert_eq!(prepared.next_record, 0);
                    assert_eq!(prepared.remaining_records, 3);
                }
                drop(cached);
                assert_available_credit(&context, limit - management_bytes(&dir), limit);
                if release != limit {
                    let batch = prepared.take_batch(3).unwrap();
                    for (record, expected) in batch.records.iter().zip(&expected) {
                        assert_eq!(codec::encode(&record.payload).unwrap(), *expected);
                    }
                }
                drop(prepared);
                assert_available_credit(&context, limit, limit);
                assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
            }
        }
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn explicit_atomic_batch_rejects_pressure_without_returning_a_committable_prefix() {
        let dir = directory("explicit-atomic-no-prefix");
        let limit = 512 * 1024;
        let context = JsonOperationContext::new(Arc::new(crate::json::WorkLedger::with_limits(
            limit, limit, limit,
        )));
        let mut builder = PreparedRecordsBuilder::new(&dir, 1_048_576, context.clone()).unwrap();
        for record in [
            large_document(1, 100),
            replacement(),
            large_document(2, 128 * 1024),
        ] {
            builder.push(record).unwrap();
        }
        let mut prepared = builder.finish().unwrap();
        let cached = context
            .cache_reservation((limit - 128 * 1024 - management_bytes(&dir)) as u128, limit)
            .unwrap();
        let available = context.available_workspace_bytes();
        let mut reclaim_calls = 0;
        let error = prepared
            .take_atomic_batch_reclaiming(|| {
                reclaim_calls += 1;
                false
            })
            .err()
            .unwrap();
        assert!(crate::json::is_memory_budget_error(&error));
        assert_eq!(reclaim_calls, 1);
        assert_eq!(prepared.next_record, 0);
        assert_eq!(prepared.remaining_records(), 3);
        assert_eq!(context.available_workspace_bytes(), available);

        // The existing manifest API can return this prefix; the atomic API did not.
        let prefix = prepared.take_batch_reclaiming(3, || false).unwrap();
        assert_eq!(prefix.len(), 1);
        drop(prefix);
        drop(cached);
        let remainder = prepared.take_atomic_batch_reclaiming(|| false).unwrap();
        assert_eq!(remainder.len(), 2);
        assert!(BatchRecordKind::of(&remainder.records[0].payload)
            .pairs_with(BatchRecordKind::of(&remainder.records[1].payload)));
        drop(remainder);
        drop(prepared);
        assert_eq!(context.available_workspace_bytes(), limit);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn batch_reclamation_prefers_complete_prefix_and_keeps_replacements_atomic() {
        let dir = directory("reclaim-atomic-prefix");
        let limit = 512 * 1024;
        for seed in 0..100 {
            let context = JsonOperationContext::new(Arc::new(
                crate::json::WorkLedger::with_limits(limit, limit, limit),
            ));
            let mut builder =
                PreparedRecordsBuilder::new(&dir, 1_048_576, context.clone()).unwrap();
            for record in [
                large_document(1, 100 + seed),
                replacement(),
                large_document(2, 128 * 1024 + seed),
            ] {
                builder.push(record).unwrap();
            }
            let mut prepared = builder.finish().unwrap();
            let mut cached = Some(
                context
                    .cache_reservation((limit - 128 * 1024 - management_bytes(&dir)) as u128, limit)
                    .unwrap(),
            );
            let first = prepared
                .take_batch_reclaiming(3, || {
                    panic!("a complete prefix can proceed without eviction")
                })
                .unwrap();
            assert_eq!(first.len(), 1);
            let path = dir.join(format!("data-{seed}.coffer"));
            let mut storage = Storage::create(&path).unwrap();
            let (append, work) = storage
                .append_prepared_batch(first, Durability::Fast)
                .unwrap();
            assert_eq!(append.records.len(), 1);
            drop((append, work));
            let committed = fs::read(&path).unwrap();
            assert_eq!(prepared.next_record, 1);
            assert_eq!(prepared.remaining_records, 2);
            let mut calls = 0;
            let second = prepared
                .take_batch_reclaiming(2, || {
                    calls += 1;
                    drop(cached.take());
                    true
                })
                .unwrap();
            assert_eq!(calls, 1);
            assert_eq!(second.len(), 2);
            assert_eq!(fs::read(&path).unwrap(), committed);
            assert!(BatchRecordKind::of(&second.records[0].payload)
                .pairs_with(BatchRecordKind::of(&second.records[1].payload)));
            let RecordPayload::DocumentPut(document) = &second.records[1].payload else {
                panic!("expected replacement document");
            };
            assert_eq!(
                document
                    .metadata
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .len(),
                128 * 1024 + seed
            );
            assert_eq!(
                second.metadata_work.reserved_bytes(),
                document
                    .metadata
                    .retained_bytes_in(JsonInputKind::DocumentMetadata, &context)
                    .unwrap()
            );
            assert_eq!(
                second.strings_work.reserved_bytes(),
                TOMBSTONE_IDENTITY_BYTES
                    + "replace".len()
                    + DOCUMENT_IDENTITY_BYTES
                    + "external-2".len()
            );
            let (append, work) = storage
                .append_prepared_batch(second, Durability::Fast)
                .unwrap();
            assert_eq!(append.records.len(), 2);
            assert_eq!(append.commit.batch_id, 2);
            drop((append, work));
            assert_eq!(prepared.remaining_records, 0);
            assert_available_credit(&context, limit - management_bytes(&dir), limit);
            drop((prepared, storage));
            assert_available_credit(&context, limit, limit);
            fs::remove_file(path).unwrap();
        }
        assert!(fs::read_dir(&dir).unwrap().all(|entry| entry
            .unwrap()
            .path()
            .extension()
            .is_some_and(|ext| ext == "lock")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn batch_reclamation_handles_json_pressure_without_prior_json_credit() {
        let dir = directory("reclaim-empty-json-prefix");
        let limit = 512 * 1024;
        for seed in 0..100 {
            let context = JsonOperationContext::new(Arc::new(
                crate::json::WorkLedger::with_limits(limit, limit, limit),
            ));
            let mut builder =
                PreparedRecordsBuilder::new(&dir, 1_048_576, context.clone()).unwrap();
            for record in [
                document(),
                replacement(),
                large_document(2, 128 * 1024 + seed),
            ] {
                builder.push(record).unwrap();
            }
            let mut prepared = builder.finish().unwrap();
            let mut cached = Some(
                context
                    .cache_reservation((limit - 128 * 1024 - management_bytes(&dir)) as u128, limit)
                    .unwrap(),
            );
            let mut calls = 0;
            let batch = prepared
                .take_batch_reclaiming(3, || {
                    calls += 1;
                    drop(cached.take());
                    true
                })
                .unwrap();
            assert_eq!(calls, 1);
            assert_eq!(batch.len(), 3);
            let RecordPayload::DocumentPut(first) = &batch.records[0].payload else {
                panic!("expected first document");
            };
            assert_eq!(first.document_id, "doc_0000000000000001");
            assert!(first.metadata.as_object().unwrap().is_empty());
            assert!(BatchRecordKind::of(&batch.records[1].payload)
                .pairs_with(BatchRecordKind::of(&batch.records[2].payload)));
            let RecordPayload::DocumentPut(last) = &batch.records[2].payload else {
                panic!("expected replacement document");
            };
            assert_eq!(
                last.metadata
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .len(),
                128 * 1024 + seed
            );
            batch.validate().unwrap();
            drop(batch);
            assert_eq!(prepared.remaining_records, 0);
            assert_available_credit(&context, limit - management_bytes(&dir), limit);
            drop(prepared);
            assert_available_credit(&context, limit, limit);
        }
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn first_atomic_record_uses_minimum_array_capacity_before_string_admission() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = directory("minimum-atomic-capacity");
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let reason = "r".repeat(128 * 1024 + seed);
            let strings = TOMBSTONE_IDENTITY_BYTES + reason.len();
            let mut record = replacement();
            if let RecordPayload::Tombstone(tombstone) = &mut record {
                tombstone.reason = Some(reason);
            }
            let mut builder =
                PreparedRecordsBuilder::new(&dir, 1_048_576, context.clone()).unwrap();
            builder.push(record).unwrap();
            let mut prepared = builder.finish().unwrap();
            let mut busy = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                - management_bytes(&dir)
                - PREPARED_COPY_BYTES
                - RECORD_SLOT_BYTES
                - time::MAX_TIMESTAMP_BYTES
                - strings;
            busy.grow_to(occupied + 1, occupied + 1, "operation_scratch")
                .unwrap();
            let error = prepared.take_batch(1).err().unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "requested_bytes"
                    && value == &(JsonProcessingPolicyV1::WORK_BUDGET_BYTES + 1).to_string()));
            assert_eq!(prepared.next_record, 0);
            busy.retain(occupied);
            let batch = prepared.take_batch(1).unwrap();
            assert_eq!(batch.records.capacity(), 1);
            assert_eq!(batch.strings_work.reserved_bytes(), strings);
            assert_eq!(
                batch.records_work.reserved_bytes(),
                PREPARED_COPY_BYTES + RECORD_SLOT_BYTES
            );
            drop(batch);
            drop(busy);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            drop(prepared);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
        }
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir(dir).unwrap();
    }

    fn staged_records(prepared: &mut PreparedRecords) -> Vec<StagedRecord> {
        LocationReader::new(
            &mut prepared.locations,
            prepared.next_record,
            prepared.remaining_records,
            prepared.source_bytes,
            &prepared.context,
        )
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap()
    }

    fn directory(label: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("cofferdb-prepared-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).unwrap();
        path
    }

    fn document() -> RecordPayload {
        RecordPayload::DocumentPut(Document {
            document_id: "doc_1".into(),
            collection_id: "col_1".into(),
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            created_at: "1970-01-01T00:00:00Z".into(),
            updated_at: "1970-01-01T00:00:00Z".into(),
        })
    }

    fn large_document(id: u64, bytes: usize) -> RecordPayload {
        let mut record = document();
        if let RecordPayload::DocumentPut(document) = &mut record {
            document.document_id = format!("doc_{id}");
            document.external_id = format!("external-{id}");
            document.metadata =
                object_from_pairs([("payload", JsonValue::String("x".repeat(bytes)))]);
        }
        record
    }

    fn replacement() -> RecordPayload {
        RecordPayload::Tombstone(Tombstone {
            target_type: "document".into(),
            target_id: "doc_2".into(),
            deleted_at: "1970-01-01T00:00:00Z".into(),
            reason: Some("replace".into()),
        })
    }

    #[test]
    fn string_pressure_preserves_atomic_prefixes_charge_lifetimes_and_checksum_priority() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = directory("string-pressure");
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut first = document();
            let title = "first-\u{e9}".repeat(1024 + seed);
            let first_bytes = DOCUMENT_IDENTITY_BYTES + "doc".len() + title.len();
            if let RecordPayload::DocumentPut(document) = &mut first {
                document.title = Some(title);
            }
            let mut second = large_document(2, 0);
            let source = "source/path/".repeat(512 * 1024 / 12 + seed);
            let second_bytes = DOCUMENT_IDENTITY_BYTES + "external-2".len() + source.len();
            if let RecordPayload::DocumentPut(document) = &mut second {
                document.source_uri = Some(source);
            }
            let mut builder =
                PreparedRecordsBuilder::new(&dir, 2 * 1_048_576, context.clone()).unwrap();
            for record in [first, replacement(), second] {
                builder.push(record).unwrap();
            }
            let mut prepared = builder.finish().unwrap();
            let last = staged_records(&mut prepared).pop().unwrap();
            let mut busy = context.workspace_reservation();
            let occupied =
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 256 * 1024 - management_bytes(&dir);
            busy.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let batch = prepared.take_batch(3).unwrap();
            assert_eq!(batch.len(), 1);
            assert_eq!(batch.strings_work.reserved_bytes(), first_bytes);
            let path = dir.join(format!("data-{seed}.coffer"));
            let mut storage = Storage::create(&path).unwrap();
            let (append, work) = storage
                .append_prepared_batch(batch, Durability::Fast)
                .unwrap();
            assert_eq!(
                work.1.reserved_bytes(),
                first_bytes + append.records.capacity() * std::mem::size_of::<CommittedRecord>()
            );
            let mut remaining = context.workspace_reservation();
            let free = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                - management_bytes(&dir)
                - occupied
                - work.0.reserved_bytes()
                - work.1.reserved_bytes()
                - time::MAX_TIMESTAMP_BYTES;
            remaining.grow_to(free, free, "operation_scratch").unwrap();
            assert!(remaining
                .grow_to(free + 1, free + 1, "operation_scratch")
                .is_err());
            drop(remaining);
            drop(append);
            drop(work);
            let committed = fs::read(&path).unwrap();
            assert_eq!(prepared.remaining_records, 2);
            let error = prepared.take_batch(2).err().unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(prepared.next_record, 1);
            assert_eq!(fs::read(&path).unwrap(), committed);
            // A budget rejection still drains and checks the entire source payload.
            let position = last.source.offset + last.source.len - 1;
            prepared.source.seek(SeekFrom::Start(position)).unwrap();
            let mut saved = [0];
            prepared.source.read_exact(&mut saved).unwrap();
            prepared.source.seek(SeekFrom::Start(position)).unwrap();
            prepared.source.write_all(&[saved[0] ^ 0x80]).unwrap();
            assert_eq!(
                prepared.take_batch(2).err().unwrap().code(),
                ErrorCode::ChecksumMismatch
            );
            assert_eq!(prepared.next_record, 1);
            assert_eq!(fs::read(&path).unwrap(), committed);
            prepared.source.seek(SeekFrom::Start(position)).unwrap();
            prepared.source.write_all(&saved).unwrap();
            drop(busy);
            let batch = prepared.take_batch(2).unwrap();
            assert_eq!(batch.len(), 2);
            assert_eq!(
                batch.strings_work.reserved_bytes(),
                TOMBSTONE_IDENTITY_BYTES + "replace".len() + second_bytes
            );
            let (append, work) = storage
                .append_prepared_batch(batch, Durability::Fast)
                .unwrap();
            assert_eq!(append.commit.record_count, 2);
            drop(append);
            drop(work);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            assert_eq!(prepared.remaining_records, 0);
            drop(storage);
            drop(prepared);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            fs::remove_file(path).unwrap();
        }
        assert!(fs::read_dir(&dir).unwrap().all(|entry| entry
            .unwrap()
            .path()
            .extension()
            .is_some_and(|ext| ext == "lock")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shared_memory_rejection_preserves_replacement_pair_for_retry_at_100_boundaries() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = directory("workspace-atomic-retry");
        for spare in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut builder = PreparedRecordsBuilder::new(&dir, 4096, context.clone()).unwrap();
            builder.push(replacement()).unwrap();
            builder.push(large_document(2, 0)).unwrap();
            let mut prepared = builder.finish().unwrap();
            let path = dir.join(format!("data-{spare}.coffer"));
            let mut storage = Storage::create(&path).unwrap();
            let before = fs::read(&path).unwrap();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - management_bytes(&dir)
                    - PREPARED_COPY_BYTES
                    - RECORD_SLOT_BYTES
                    - TOMBSTONE_IDENTITY_BYTES
                    - "replace".len()
                    - time::MAX_TIMESTAMP_BYTES
                    - spare,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            let error = prepared.take_batch(2).err().unwrap();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            for (key, expected) in [
                ("resource", "memory".to_owned()),
                ("operation", "batch_prepare".to_owned()),
                (
                    "limit_bytes",
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES.to_string(),
                ),
                (
                    "requested_bytes",
                    (JsonProcessingPolicyV1::WORK_BUDGET_BYTES + RECORD_SLOT_BYTES - spare)
                        .to_string(),
                ),
            ] {
                assert!(error
                    .detail()
                    .iter()
                    .any(|(name, value)| name == key && value == &expected));
            }
            assert_eq!(prepared.remaining_records, 2);
            assert_eq!(prepared.next_record, 0);
            assert_eq!(fs::read(&path).unwrap(), before);
            busy.grow(
                PREPARED_COPY_BYTES
                    + RECORD_SLOT_BYTES
                    + TOMBSTONE_IDENTITY_BYTES
                    + "replace".len()
                    + time::MAX_TIMESTAMP_BYTES
                    + spare,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            drop(busy);
            let batch = prepared.take_batch(2).unwrap();
            assert_eq!(batch.len(), 2);
            let string_bytes = batch.strings_work.reserved_bytes();
            assert_eq!(
                string_bytes,
                TOMBSTONE_IDENTITY_BYTES
                    + "replace".len()
                    + DOCUMENT_IDENTITY_BYTES
                    + "external-2".len()
            );
            let (append, work) = storage
                .append_prepared_batch(batch, Durability::Fast)
                .unwrap();
            assert_eq!(append.records.len(), 2);
            assert!(BatchRecordKind::of(&append.records[0].payload)
                .pairs_with(BatchRecordKind::of(&append.records[1].payload)));
            assert_eq!(
                work.1.reserved_bytes(),
                append.records.capacity() * std::mem::size_of::<CommittedRecord>() + string_bytes
            );
            drop(append);
            drop(work);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            drop(storage);
            drop(prepared);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            fs::remove_file(path).unwrap();
        }
        assert!(fs::read_dir(&dir).unwrap().all(|entry| entry
            .unwrap()
            .path()
            .extension()
            .is_some_and(|ext| ext == "lock")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shared_memory_splits_only_complete_atomic_prefixes_and_retains_output_arrays() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = directory("workspace-prefix");
        for seed in 0..100 {
            let slots = 32 + seed % 33;
            let context = JsonOperationContext::standalone();
            let mut builder =
                PreparedRecordsBuilder::new(&dir, 64 * 1024, context.clone()).unwrap();
            builder.push(document()).unwrap();
            for _ in 0..slots - 1 {
                builder.push(replacement()).unwrap();
            }
            builder.push(large_document(2, 0)).unwrap();
            let mut prepared = builder.finish().unwrap();
            let first = &staged_records(&mut prepared)[0];
            let metadata_bytes = payload::read_staged_summary_in(
                &mut prepared.source,
                first.kind.record_type,
                &first.source,
                &context,
                &mut context.workspace_reservation(),
            )
            .unwrap()
            .1
            .reserved_bytes();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - management_bytes(&dir)
                    - PREPARED_COPY_BYTES
                    - RECORD_SLOT_BYTES * slots
                    - DOCUMENT_IDENTITY_BYTES
                    - "doc".len()
                    - (TOMBSTONE_IDENTITY_BYTES + "replace".len()) * (slots - 1)
                    - time::MAX_TIMESTAMP_BYTES
                    - metadata_bytes,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
            let batch = prepared.take_batch(slots + 1).unwrap();
            let admitted = batch.len();
            assert!((1..slots).contains(&admitted));
            let string_bytes = DOCUMENT_IDENTITY_BYTES
                + "doc".len()
                + (TOMBSTONE_IDENTITY_BYTES + "replace".len()) * (admitted - 1);
            assert_eq!(batch.strings_work.reserved_bytes(), string_bytes);
            assert_eq!(
                batch.records_work.reserved_bytes(),
                PREPARED_COPY_BYTES + RECORD_SLOT_BYTES * batch.records.capacity()
            );
            let path = dir.join(format!("data-{seed}.coffer"));
            let mut storage = Storage::create(&path).unwrap();
            let (append, work) = storage
                .append_prepared_batch(batch, Durability::Fast)
                .unwrap();
            assert_eq!(append.records.len(), admitted);
            let remaining_records = slots + 1 - admitted;
            assert_eq!(prepared.remaining_records, remaining_records);
            assert_eq!(
                work.1.reserved_bytes(),
                append.records.capacity() * std::mem::size_of::<CommittedRecord>() + string_bytes
            );
            let mut remaining = context.reservation();
            remaining
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                        - management_bytes(&dir)
                        - busy.reserved_bytes()
                        - work.0.reserved_bytes()
                        - work.1.reserved_bytes()
                        - time::MAX_TIMESTAMP_BYTES,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
            assert!(remaining.grow(1, JsonInputKind::JsonlRecord).is_err());
            drop(remaining);
            drop(append);
            drop(work);
            let last = prepared.take_batch(remaining_records).unwrap();
            assert_eq!(last.len(), remaining_records);
            assert!(
                BatchRecordKind::of(&last.records[remaining_records - 2].payload).pairs_with(
                    BatchRecordKind::of(&last.records[remaining_records - 1].payload)
                )
            );
            let (append, work) = storage
                .append_prepared_batch(last, Durability::Fast)
                .unwrap();
            assert_eq!(append.commit.record_count, remaining_records);
            drop(append);
            drop(work);
            drop(busy);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            assert_eq!(prepared.remaining_records, 0);
            drop(storage);
            drop(prepared);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            fs::remove_file(path).unwrap();
        }
        assert!(fs::read_dir(&dir).unwrap().all(|entry| entry
            .unwrap()
            .path()
            .extension()
            .is_some_and(|ext| ext == "lock")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn metadata_staging_reuses_work_and_admits_only_complete_atomic_json_prefixes() {
        let dir = directory("metadata-spool");
        let context = JsonOperationContext::standalone();
        let mut busy = context.reservation();
        busy.grow(24 * 1_048_576, JsonInputKind::JsonlRecord)
            .unwrap();
        let mut builder =
            PreparedRecordsBuilder::new(&dir, 16 * 1_048_576, context.clone()).unwrap();
        for record in [
            large_document(1, 3 * 1_048_576),
            replacement(),
            large_document(2, 3 * 1_048_576),
        ] {
            builder.push(record).unwrap();
            context
                .reservation()
                .grow(
                    8 * 1_048_576 - management_bytes(&dir),
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
        let mut prepared = builder.finish().unwrap();
        assert!(std::mem::size_of::<StagedRecord>() <= 32);
        assert_eq!(prepared.remaining_records, 3);
        let mut storage = Storage::create(dir.join("data.coffer")).unwrap();
        let first = prepared.take_batch(3).unwrap();
        assert_eq!(first.len(), 1);
        assert!(first.metadata_work.reserved_bytes() >= 3 * 1_048_576);
        let (append, work) = storage
            .append_prepared_batch(first, Durability::Fast)
            .unwrap();
        assert_eq!(append.records.len(), 1);
        assert!(context
            .reservation()
            .grow(5 * 1_048_576, JsonInputKind::DocumentMetadata)
            .is_err());
        let RecordPayload::DocumentPut(document) = &append.records[0].payload else {
            panic!()
        };
        assert_eq!(
            document
                .metadata
                .get("payload")
                .and_then(JsonValue::as_str)
                .unwrap()
                .len(),
            3 * 1_048_576
        );
        drop(append);
        drop(work);
        context
            .reservation()
            .grow(
                8 * 1_048_576 - management_bytes(&dir),
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
        let second = prepared.take_batch(2).unwrap();
        assert_eq!(second.len(), 2);
        assert!(BatchRecordKind::of(&second.records[0].payload)
            .pairs_with(BatchRecordKind::of(&second.records[1].payload)));
        let (append, work) = storage
            .append_prepared_batch(second, Durability::Fast)
            .unwrap();
        assert_eq!(append.records.len(), 2);
        drop(append);
        drop(work);
        context
            .reservation()
            .grow(
                8 * 1_048_576 - management_bytes(&dir),
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
        assert!(prepared.records().unwrap().next().is_none());
        let spool_path = prepared._temporary.path().to_path_buf();
        drop(prepared);
        drop(storage);
        drop(busy);
        assert_available_credit(
            &context,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
        );
        assert!(!spool_path.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_metadata_admission_preserves_the_atomic_group_for_retry() {
        let dir = directory("metadata-retry");
        let context = JsonOperationContext::standalone();
        let mut builder =
            PreparedRecordsBuilder::new(&dir, 8 * 1_048_576, context.clone()).unwrap();
        builder.push(replacement()).unwrap();
        builder.push(large_document(2, 2 * 1_048_576)).unwrap();
        let mut prepared = builder.finish().unwrap();
        let mut storage = Storage::create(dir.join("data.coffer")).unwrap();
        let before = fs::read(storage.path()).unwrap();
        let mut busy = context.reservation();
        busy.grow(31 * 1_048_576, JsonInputKind::JsonlRecord)
            .unwrap();
        let error = prepared.take_batch(2).err().unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "document_metadata"));
        assert_eq!(prepared.remaining_records, 2);
        assert_eq!(fs::read(storage.path()).unwrap(), before);
        context
            .reservation()
            .grow(
                1_048_576 - management_bytes(&dir),
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
        drop(busy);
        let batch = prepared.take_batch(2).unwrap();
        assert_eq!(batch.len(), 2);
        let (append, work) = storage
            .append_prepared_batch(batch, Durability::Fast)
            .unwrap();
        assert_eq!(append.records.len(), 2);
        drop(append);
        drop(work);
        drop(prepared);
        drop(storage);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn corrupted_metadata_takes_checksum_precedence_and_clears_decoded_prefixes() {
        let dir = directory("metadata-checksum");
        let context = JsonOperationContext::standalone();
        let mut builder = PreparedRecordsBuilder::new(&dir, 4096, context.clone()).unwrap();
        builder.push(large_document(1, 100)).unwrap();
        builder.push(large_document(2, 100)).unwrap();
        let mut prepared = builder.finish().unwrap();
        let second = &staged_records(&mut prepared)[1];
        let bytes = codec::encode(&large_document(2, 100)).unwrap();
        let marker = b"{\"payload\":";
        let offset = second.source.offset
            + bytes
                .windows(marker.len())
                .position(|bytes| bytes == marker)
                .unwrap() as u64;
        prepared.source.seek(SeekFrom::Start(offset)).unwrap();
        prepared.source.write_all(&[255]).unwrap();
        assert_eq!(
            prepared.take_batch(2).err().unwrap().code(),
            ErrorCode::ChecksumMismatch
        );
        assert_eq!(prepared.remaining_records, 2);
        assert!(staged_records(&mut prepared)
            .iter()
            .all(|record| record.kind.record_type == DOCUMENT_PUT));
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                JsonInputKind::JsonlRecord,
            )
            .unwrap();
        drop(prepared);
        assert_available_credit(
            &context,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staged_payloads_release_large_fields_and_replay_exact_bytes() {
        let dir = directory("roundtrip");
        let mut records = vec![document()];
        for index in 1..=100 {
            let text = "x".repeat((index % 4 + 1) * 32768);
            records.push(RecordPayload::ChunkPut(Chunk {
                chunk_id: format!("chk_{index}"),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: format!("c-{index}"),
                text_hash: crate::model::chunk_text_hash(&text),
                token_count: 1,
                text,
                ordinal: index,
                metadata: object_from_pairs([("index", JsonValue::Number(index as f64))]),
                source_span: None,
                embedding_id: None,
                created_at: "1970-01-01T00:00:00Z".into(),
            }));
            records.push(RecordPayload::EmbeddingPut(Embedding {
                embedding_id: format!("emb_{index}"),
                chunk_id: format!("chk_{index}"),
                collection_id: "col_1".into(),
                chunk_external_id: format!("c-{index}"),
                dim: index,
                vector: vec![0.25; index],
                model_name: "m".into(),
                created_at: "1970-01-01T00:00:00Z".into(),
            }));
        }
        let expected = records
            .iter()
            .map(|record| codec::encode(record).unwrap())
            .collect::<Vec<_>>();
        let mut prepared = PreparedRecords::create(
            records,
            &dir,
            expected
                .iter()
                .map(|bytes| bytes.len() as u64 + locations::LOCATION_BYTES as u64)
                .sum(),
        )
        .unwrap();
        let mut copy_buffer = [0u8; 64 * 1024];
        let temporary = prepared._temporary.path().to_path_buf();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&temporary).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        for (record, expected) in staged_records(&mut prepared).iter().zip(&expected) {
            assert!(std::mem::size_of_val(record) <= 32);
            let mut actual = Vec::new();
            copy_payload(
                &prepared.source,
                &record.source,
                &mut actual,
                &mut copy_buffer,
            )
            .unwrap();
            assert_eq!(&actual, expected);
        }
        assert!(prepared.take_batch(0).is_err());
        assert!(prepared.take_batch(expected.len() + 1).is_err());
        let mut storage = Storage::create(dir.join("data.coffer")).unwrap();
        let mut offset = 0;
        while offset < expected.len() {
            let count = 13.min(expected.len() - offset);
            let (append, _metadata_work) = storage
                .append_prepared_batch(prepared.take_batch(count).unwrap(), Durability::Fast)
                .unwrap();
            let mut reader = storage.payload_reader().unwrap();
            for (record, expected) in append.records.iter().zip(&expected[offset..offset + count]) {
                let payload = reader.read_payload(&record.record_ref).unwrap();
                assert_eq!(&codec::encode(&payload).unwrap(), expected);
            }
            offset += count;
        }
        assert!(prepared.records().unwrap().next().is_none());
        assert!(scan_file(storage.path()).unwrap().findings.is_empty());
        drop(prepared);
        assert!(!temporary.exists());
        drop(storage);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staging_budget_and_encoding_errors_clean_all_temporary_files() {
        let dir = directory("budget");
        let payload_length = codec::encode(&document()).unwrap().len() as u64;
        let length = payload_length + locations::LOCATION_BYTES as u64;
        for limit in [0, length - 1, length, 2 * length - 1, 2 * length] {
            let staged = PreparedRecords::create(vec![document(), document()], &dir, limit);
            if limit == 2 * length {
                let staged = staged.unwrap();
                assert_eq!(staged.source.metadata().unwrap().len(), 2 * payload_length);
                assert_eq!(
                    staged.locations.metadata().unwrap().len(),
                    2 * locations::LOCATION_BYTES as u64
                );
                drop(staged);
            } else {
                let error = staged.err().unwrap();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(error
                    .detail()
                    .iter()
                    .any(|(key, value)| key == "resource" && value == "temporary_disk"));
            }
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        let mut invalid = document();
        if let RecordPayload::DocumentPut(record) = &mut invalid {
            record.updated_at = "invalid".into();
        }
        assert_eq!(
            PreparedRecords::create(vec![document(), invalid], &dir, 4096)
                .err()
                .unwrap()
                .code(),
            ErrorCode::FileFormatError
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staged_encoding_yields_its_buffer_and_rolls_back_only_the_current_record() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = directory("encoding-buffer-budget");
        let budget = Arc::new(TemporaryDiskBudget::new(4 * 1_048_576));
        for seed in 0..100usize {
            let context = JsonOperationContext::standalone();
            let mut builder = PreparedRecordsBuilder::new_in(
                &dir,
                2 * 1_048_576,
                context.clone(),
                budget.clone(),
            )
            .unwrap();
            let prefix = document();
            let mut expected = codec::encode(&prefix).unwrap();
            builder.push(prefix).unwrap();
            let mut record = document();
            let RecordPayload::DocumentPut(value) = &mut record else {
                panic!()
            };
            value.external_id = format!("record-{seed}");
            value.title = Some("title".repeat(4096 + seed));
            value.metadata = crate::json::object_from_pairs([(
                "payload",
                JsonValue::String("x".repeat(1024 + seed)),
            )]);
            let mut work = value
                .metadata
                .reserve_retained_in(JsonInputKind::DocumentMetadata, &context)
                .unwrap();
            let retained = work.reserved_bytes();
            expected.extend(codec::encode(&record).unwrap());
            let available = 4096 + seed;
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                - management_bytes(&dir)
                - retained
                - available;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let mut buffer = context.workspace_reservation();
            buffer
                .grow_to(available, available, "batch_prepare")
                .unwrap();
            let error =
                codec::encode_retained_into(&record, &mut std::io::sink(), &context, &mut work)
                    .unwrap_err();
            assert!(crate::json::is_limit_error(&error));
            drop(buffer);
            work.retain(retained.min(work.reserved_bytes()));

            // The large title reaches the private file before metadata needs scratch.
            // Success therefore requires truncation before the unbuffered retry.
            builder.push_retained(record, &mut work).unwrap();
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(builder.record_count, 2);
            assert_eq!(
                builder.writer.output.metadata().unwrap().len(),
                expected.len() as u64
            );
            assert_eq!(
                builder.writer.digest.len,
                expected.len() as u64 - codec::encode(&document()).unwrap().len() as u64
            );
            context
                .workspace_reservation()
                .grow_to(
                    retained + available,
                    retained + available,
                    "operation_scratch",
                )
                .unwrap();
            drop(held);
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                    JsonInputKind::Filter,
                )
                .unwrap();
            let mut prepared = builder.finish().unwrap();
            assert_eq!(fs::read(prepared._temporary.path()).unwrap(), expected);
            assert_eq!(
                budget.used(),
                expected.len() as u64 + 2 * locations::LOCATION_BYTES as u64
            );
            assert_eq!(prepared.records().unwrap().count(), 2);
            let batch = prepared.take_batch(2).unwrap();
            assert_eq!(batch.len(), 2);
            let RecordPayload::DocumentPut(second) = &batch.records[1].payload else {
                panic!()
            };
            assert_eq!(second.external_id, format!("record-{seed}"));
            assert_eq!(second.title.as_ref().unwrap().len(), 5 * (4096 + seed));
            drop(batch);
            drop(prepared);
            assert_available_credit(
                &context,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            assert_eq!(budget.used(), 0);
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn per_file_staging_limit_reports_exact_request_without_exhausting_shared_disk() {
        let dir = directory("file-budget");
        let budget = Arc::new(TemporaryDiskBudget::new(1_048_576));
        let mut writer = StagingWriter::new(
            &dir,
            "file-budget",
            5,
            budget.clone(),
            &JsonOperationContext::standalone(),
            &mut || false,
        )
        .unwrap();
        writer.write_all(b"abcd").unwrap();
        let rejected = writer.write_all(b"efg").map_err(Error::from);
        let error = writer.checked(rejected, 5).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "limit_bytes" && value == "5"));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "requested_bytes" && value == "7"));
        assert_eq!(writer.remaining, 1);
        assert!(budget.used() < budget.limit_bytes());
        drop(writer);
        assert_eq!(budget.used(), 0);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn incremental_staging_releases_vectors_and_cannot_finish_after_a_failed_row() {
        let dir = directory("incremental");
        for fail_after in 0..100 {
            let mut builder = PreparedRecordsBuilder::new(
                &dir,
                16 * 1_048_576,
                JsonOperationContext::standalone(),
            )
            .unwrap();
            for index in 0..fail_after {
                builder
                    .push(RecordPayload::EmbeddingPut(Embedding {
                        embedding_id: format!("emb_{}", index + 1),
                        chunk_id: format!("chk_{}", index + 1),
                        collection_id: "col_1".into(),
                        chunk_external_id: format!("c-{index}"),
                        dim: 768,
                        vector: vec![0.25; 768],
                        model_name: "m".into(),
                        created_at: "1970-01-01T00:00:00Z".into(),
                    }))
                    .unwrap();
                assert_eq!(builder.record_count, index + 1);
                assert_eq!(
                    builder.byte_limit - builder.locations.remaining,
                    (index as u64 + 1) * locations::LOCATION_BYTES as u64
                );
            }
            let mut invalid = document();
            if let RecordPayload::DocumentPut(record) = &mut invalid {
                record.updated_at = "invalid".into();
            }
            assert_eq!(
                builder.push(invalid).unwrap_err().code(),
                ErrorCode::FileFormatError
            );
            assert_eq!(builder.record_count, fail_after);
            assert!(builder.push(document()).is_err());
            assert!(builder.finish().is_err());
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        let builder =
            PreparedRecordsBuilder::new(&dir, 0, JsonOperationContext::standalone()).unwrap();
        let mut prepared = builder.finish().unwrap();
        assert_eq!(prepared.source.metadata().unwrap().len(), 0);
        assert!(prepared.records().unwrap().next().is_none());
        drop(prepared);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn exhausted_json_context_poisoning_removes_the_private_staging_file() {
        let dir = directory("shared-json-budget");
        let context = JsonOperationContext::standalone();
        let mut builder = PreparedRecordsBuilder::new(&dir, 4096, context.clone()).unwrap();
        builder.push(document()).unwrap();
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
            crate::json::JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let error = builder.push(document()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "document_metadata"));
        assert!(builder.finish().is_err());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        drop(busy);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                crate::json::JsonInputKind::JsonlRecord,
            )
            .unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staging_reuses_disk_credit_and_returns_unused_or_failed_reservations() {
        let dir = directory("shared-disk-budget");
        let payload_length = codec::encode(&document()).unwrap().len() as u64;
        let length = payload_length + locations::LOCATION_BYTES as u64;
        for seed in 0..100 {
            let other_bytes = 17 + seed;
            let exact = other_bytes + 2 * length;
            for limit in [exact - 1, exact, 1_048_576] {
                let budget = Arc::new(TemporaryDiskBudget::new(limit));
                let (file, mut held) =
                    create_budgeted_temporary_file_in(&dir, "input", "tmp", budget.clone())
                        .unwrap();
                held.reserve_disk(other_bytes).unwrap();
                let mut builder = PreparedRecordsBuilder::new_in(
                    &dir,
                    u64::MAX,
                    JsonOperationContext::standalone(),
                    budget.clone(),
                )
                .unwrap();
                builder.push(document()).unwrap();
                assert!(budget.used() > other_bytes + length);
                let result = builder.push(document());
                if limit < exact {
                    let error = result.unwrap_err();
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "resource" && value == "temporary_disk"));
                    assert!(builder.finish().is_err());
                    assert_eq!(budget.used(), other_bytes);
                } else {
                    result.unwrap();
                    let prepared = builder.finish().unwrap();
                    assert_eq!(budget.used(), exact);
                    assert_eq!(
                        prepared.source.metadata().unwrap().len(),
                        2 * payload_length
                    );
                    assert_eq!(
                        prepared.locations.metadata().unwrap().len(),
                        2 * locations::LOCATION_BYTES as u64
                    );
                    drop(prepared);
                    assert_eq!(budget.used(), other_bytes);
                }
                drop(file);
                drop(held);
                assert_eq!(budget.used(), 0);
                assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn damaged_staging_never_invalidates_the_committed_database_prefix() {
        for truncated in [false, true] {
            for after_admission in [false, true] {
                let dir = directory(if truncated { "truncated" } else { "corrupt" });
                let mut storage = Storage::create(dir.join("data.coffer")).unwrap();
                storage
                    .append_owned_batch(vec![document()], Durability::Commit)
                    .unwrap();
                let original = fs::read(storage.path()).unwrap();
                let mut prepared =
                    PreparedRecords::create(vec![document(), document()], &dir, 4096).unwrap();
                let damaged_offset = staged_records(&mut prepared)[1].source.offset;
                let mut changed = prepared.source.try_clone().unwrap();
                let damage = |file: &mut File| {
                    if truncated {
                        file.set_len(damaged_offset + 1).unwrap();
                    } else {
                        file.seek(SeekFrom::Start(damaged_offset)).unwrap();
                        file.write_all(&[0xa5]).unwrap();
                    }
                };
                let code = if truncated {
                    ErrorCode::IoError
                } else {
                    ErrorCode::ChecksumMismatch
                };
                if !after_admission {
                    damage(&mut changed);
                    assert_eq!(prepared.take_batch(2).err().unwrap().code(), code);
                    assert_eq!(prepared.remaining_records, 2);
                } else {
                    let batch = prepared.take_batch(2).unwrap();
                    damage(&mut changed);
                    assert_eq!(batch.validate().unwrap_err().code(), code);
                    let checked = PreparedBatch {
                        source: batch.source,
                        records: batch
                            .records
                            .iter()
                            .map(|record| PreparedRecord {
                                payload: record.payload.clone(),
                                source: record.source.clone(),
                                string_bytes: record.string_bytes,
                            })
                            .collect(),
                        metadata_work: JsonOperationContext::standalone().reservation(),
                        records_work: JsonOperationContext::standalone().workspace_reservation(),
                        strings_work: JsonOperationContext::standalone().workspace_reservation(),
                        commit_timestamp: crate::json::ReservedString::with_capacity(
                            time::MAX_TIMESTAMP_BYTES,
                            &JsonOperationContext::standalone(),
                            "batch_prepare",
                        )
                        .unwrap(),
                    };
                    assert_eq!(
                        storage
                            .append_prepared_batch(checked, Durability::Commit)
                            .err()
                            .unwrap()
                            .code(),
                        code
                    );
                    assert_eq!(fs::read(storage.path()).unwrap(), original);
                    assert!(!storage.write_failed);
                    let writer = BatchWriter::new(
                        &mut storage.file,
                        &mut storage.next_batch_id,
                        &mut storage.next_sequence_no,
                        2,
                        true,
                        &JsonOperationContext::standalone(),
                        Some(batch.commit_timestamp),
                    )
                    .unwrap();
                    let result = writer.append_all(
                        batch.records.into_iter().map(|record| {
                            Ok(WriteRecord {
                                payload: record.payload,
                                prepared: Some((batch.source, record.source)),
                                encoding: RecordEncoding::Batch,
                            })
                        }),
                        Durability::Commit,
                    );
                    assert!(result.is_err());
                }
                assert_eq!(fs::read(storage.path()).unwrap(), original);
                let scan = scan_file(storage.path()).unwrap();
                assert_eq!(scan.commits, 1);
                assert!(scan.findings.is_empty());
                drop(prepared);
                drop(changed);
                drop(storage);
                fs::remove_dir_all(dir).unwrap();
            }
        }
    }

    #[test]
    fn fixed_descriptors_decode_only_the_requested_batch_and_preserve_record_bytes() {
        let dir = directory("fixed-descriptors");
        assert!(!std::mem::needs_drop::<StagedRecord>());
        assert!(std::mem::size_of::<StagedRecord>() <= 32);
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut builder =
                PreparedRecordsBuilder::new(&dir, 1_048_576, context.clone()).unwrap();
            let mut expected = Vec::new();
            for ordinal in 0..7 {
                let mut record = large_document(ordinal + 1, seed * 31);
                if let RecordPayload::DocumentPut(document) = &mut record {
                    document.external_id = format!("external-{ordinal}-{}", "x".repeat(seed * 17));
                    document.title = Some(format!("title-{seed}-{ordinal}"));
                    document.source_uri = Some(format!("source/{seed}/{ordinal}"));
                }
                expected.push(codec::encode(&record).unwrap());
                builder.push(record).unwrap();
            }
            let mut prepared = builder.finish().unwrap();
            let mut offset = 0;
            while offset < expected.len() {
                let count = (seed % 3 + 1).min(expected.len() - offset);
                let batch = prepared.take_batch(count).unwrap();
                assert_eq!(batch.len(), count);
                for (record, expected) in
                    batch.records.iter().zip(&expected[offset..offset + count])
                {
                    assert_eq!(&codec::encode(&record.payload).unwrap(), expected);
                }
                assert!(batch.metadata_work.reserved_bytes() > 0);
                drop(batch);
                offset += count;
                assert_eq!(prepared.remaining_records, expected.len() - offset);
                context
                    .reservation()
                    .grow(
                        crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                            - management_bytes(&dir),
                        JsonInputKind::JsonlRecord,
                    )
                    .unwrap();
            }
            drop(prepared);
            assert_available_credit(
                &context,
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            );
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staged_summary_checksums_precede_parse_or_budget_errors_across_100_mutations() {
        let dir = directory("summary-mutations");
        let context = JsonOperationContext::standalone();
        let record = large_document(1, 257);
        let expected = codec::encode(&record).unwrap();
        let mut builder = PreparedRecordsBuilder::new(&dir, 4096, context.clone()).unwrap();
        builder.push(record).unwrap();
        let mut prepared = builder.finish().unwrap();
        for seed in 0..100 {
            let mut busy = context.reservation();
            if seed % 2 == 0 {
                busy.grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                        - management_bytes(&dir)
                        - seed % time::MAX_TIMESTAMP_BYTES,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
            }
            let position = (seed * 37) % expected.len();
            prepared
                .source
                .seek(SeekFrom::Start(position as u64))
                .unwrap();
            prepared
                .source
                .write_all(&[expected[position] ^ 0x80])
                .unwrap();
            assert_eq!(
                prepared
                    .take_batch_reclaiming(1, || panic!("checksum failure must not reclaim cache"))
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::ChecksumMismatch
            );
            assert_eq!(prepared.remaining_records, 1);
            prepared
                .source
                .seek(SeekFrom::Start(position as u64))
                .unwrap();
            prepared
                .source
                .write_all(&expected[position..position + 1])
                .unwrap();
            prepared.source.set_len(position as u64).unwrap();
            assert_eq!(
                prepared
                    .take_batch_reclaiming(1, || panic!("IO failure must not reclaim cache"))
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::IoError
            );
            assert_eq!(prepared.remaining_records, 1);
            prepared.source.seek(SeekFrom::Start(0)).unwrap();
            prepared.source.write_all(&expected).unwrap();
            drop(busy);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
        let batch = prepared.take_batch(1).unwrap();
        assert_eq!(codec::encode(&batch.records[0].payload).unwrap(), expected);
        drop(batch);
        drop(prepared);
        assert_available_credit(
            &context,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn damaged_locations_fail_before_admission_and_keep_prior_commits_unchanged() {
        let dir = directory("location-mutations");
        let context = JsonOperationContext::standalone();
        let mut builder = PreparedRecordsBuilder::new(&dir, 4096, context.clone()).unwrap();
        for id in 1..=3 {
            builder.push(large_document(id, 100)).unwrap();
        }
        let mut prepared = builder.finish().unwrap();
        let original = fs::read(prepared._locations_temporary.path()).unwrap();
        assert_eq!(original.len(), 3 * locations::LOCATION_BYTES);
        for seed in 0..100 {
            let offset = seed * 37 % original.len();
            prepared
                .locations
                .seek(SeekFrom::Start(offset as u64))
                .unwrap();
            prepared
                .locations
                .write_all(&[original[offset] ^ 0x80])
                .unwrap();
            assert_eq!(
                prepared
                    .records()
                    .unwrap()
                    .collect::<Result<Vec<_>>>()
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::ChecksumMismatch
            );
            assert_eq!(
                prepared.take_batch(3).err().unwrap().code(),
                ErrorCode::ChecksumMismatch
            );
            assert_eq!(prepared.next_record, 0);
            assert_eq!(prepared.remaining_records, 3);
            prepared
                .locations
                .seek(SeekFrom::Start(offset as u64))
                .unwrap();
            prepared
                .locations
                .write_all(&original[offset..offset + 1])
                .unwrap();
            prepared.locations.set_len(offset as u64).unwrap();
            assert_eq!(prepared.records().err().unwrap().code(), ErrorCode::IoError);
            assert_eq!(
                prepared.take_batch(3).err().unwrap().code(),
                ErrorCode::IoError
            );
            prepared.locations.seek(SeekFrom::Start(0)).unwrap();
            prepared.locations.write_all(&original).unwrap();
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - management_bytes(&dir),
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
        prepared.locations.seek(SeekFrom::End(0)).unwrap();
        prepared.locations.write_all(&[0]).unwrap();
        assert_eq!(
            prepared.records().err().unwrap().code(),
            ErrorCode::FileFormatError
        );
        prepared.locations.set_len(original.len() as u64).unwrap();
        let mut storage = Storage::create(dir.join("data.coffer")).unwrap();
        let (append, work) = storage
            .append_prepared_batch(prepared.take_batch(2).unwrap(), Durability::Fast)
            .unwrap();
        assert_eq!(append.records.len(), 2);
        drop(append);
        drop(work);
        let before = fs::read(storage.path()).unwrap();
        prepared
            .locations
            .seek(SeekFrom::Start(2 * locations::LOCATION_BYTES as u64))
            .unwrap();
        prepared.locations.write_all(&[0xff]).unwrap();
        assert_eq!(
            prepared.take_batch(1).err().unwrap().code(),
            ErrorCode::ChecksumMismatch
        );
        assert_eq!(prepared.remaining_records, 1);
        assert_eq!(fs::read(storage.path()).unwrap(), before);
        prepared.locations.seek(SeekFrom::Start(0)).unwrap();
        prepared.locations.write_all(&original).unwrap();
        let (append, work) = storage
            .append_prepared_batch(prepared.take_batch(1).unwrap(), Durability::Fast)
            .unwrap();
        assert_eq!(append.records.len(), 1);
        drop(append);
        drop(work);
        assert!(prepared.records().unwrap().next().is_none());
        drop(prepared);
        drop(storage);
        assert_available_credit(
            &context,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
        );
        assert_eq!(
            fs::read_dir(&dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<std::collections::BTreeSet<_>>(),
            ["data.coffer", "data.coffer.lock"]
                .into_iter()
                .map(std::ffi::OsString::from)
                .collect()
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
