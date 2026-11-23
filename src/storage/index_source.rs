use super::*;
use crate::index::text::TextStatsSummary;
use crate::json::{JsonOperationContext, WorkspaceReservation};

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub(crate) struct LazyVectorCheckpoint {
    #[cfg(test)]
    pub(crate) text_segments: Vec<TextIndexSegmentPayload>,
    // Location of both text and vector segments in the validated checkpoint. The
    // source owns the directory and its memory reservation.
    pub(crate) vectors: IndexSegmentSource,
}

// The directory is a map into one checkpoint, not a long-lived map of
// collection IDs to file offsets. An append can put new bytes at the same
// offset. Use the covered commit and generation when reusing entries or
// statistics; a collection name surviving the write proves nothing about
// the bytes at its old extent.
#[derive(Debug)]
pub(crate) struct IndexSegmentSource {
    // Absolute file position where the checkpoint header begins, immediately after
    // the covered critical log.
    offset: u64,
    // Length of the framed records in the checkpoint payload, excluding the outer
    // header, directory and trailer. Bounds each extent lookup.
    payload_len: u64,
    // Generation stamped on the checkpoint extents. The source also binds locally
    // retained text statistics to this value.
    generation: u64,
    // Visible commit sequence that each decoded segment must cover.
    commit_sequence_no: u64,
    // Only segment rows, ordered by (record type, numeric collection ID) for
    // query-time binary search.
    extents: IndexDirectory,
    // Optional statistics accompanied by the checkpoint generation at which they
    // were installed. A disk directory need not come with these statistics.
    text_stats: Option<(u64, TextStatsSummary)>,
}

impl IndexSegmentSource {
    pub(super) fn new(
        offset: u64,
        payload_len: u64,
        generation: u64,
        commit_sequence_no: u64,
        mut extents: IndexDirectory,
    ) -> Result<Self> {
        // The persisted table is ordered by record ID for verification. Once
        // verified, keep just the segment rows and order them by the key used by
        // queries. Adjacent duplicate keys would leave lookup unable to choose
        // between two representations of one collection.
        extents
            .retain(|entry| matches!(entry.record_type, INDEX_TEXT_SEGMENT | INDEX_VECTOR_SEGMENT));
        extents
            .as_mut_slice()
            .sort_unstable_by_key(|entry| (entry.record_type, entry.stream_id));
        // A collection may have both a text and a vector record with the same
        // stream ID. Reject only a repeated pair of record type and stream ID,
        // which would make selecting that kind of extent ambiguous.
        if extents.as_slice().windows(2).any(|pair| {
            (pair[0].record_type, pair[0].stream_id) == (pair[1].record_type, pair[1].stream_id)
        }) {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "duplicate collection in index snapshot directory",
            ));
        }
        Ok(Self {
            offset,
            payload_len,
            generation,
            commit_sequence_no,
            extents,
            // A parsed directory identifies where to read records. It does not by
            // itself supply collection term counts; the database attaches those
            // after a writer or validated text visit produces them.
            text_stats: None,
        })
    }

    pub(crate) fn with_text_stats(mut self, stats: TextStatsSummary) -> Self {
        // Preserve the generation alongside the summary at installation. A later
        // stats request compares the saved stamp with this directory as well as
        // checking the caller's primary boundary and commit position.
        self.text_stats = Some((self.generation, stats));
        self
    }

    pub(crate) fn text_stats(
        &self,
        critical_end_offset: u64,
        commit_sequence_no: u64,
    ) -> Option<&TextStatsSummary> {
        // The database supplies its current critical end and commit. Statistics
        // from a previous write can have a valid-looking collection number, so
        // check all three pieces of snapshot identity before returning them.
        let (generation, stats) = self.text_stats.as_ref()?;
        (self.offset == critical_end_offset
            && self.generation == *generation
            && self.commit_sequence_no == commit_sequence_no)
            .then_some(stats)
    }

    fn extent(&self, kind: u16, collection: u64) -> Option<&IndexExtentEntry> {
        let entries = self.extents.as_slice();
        // Search with the same composite key used at construction. A directory
        // miss means this source has no segment of the requested kind for the
        // collection; an equal stream ID under the other kind is irrelevant.
        entries
            .binary_search_by_key(&(kind, collection), |entry| {
                (entry.record_type, entry.stream_id)
            })
            .ok()
            .map(|position| &entries[position])
    }

    #[cfg(test)]
    pub(crate) fn assert_operation_context(&self, context: &JsonOperationContext) {
        self.extents.assert_operation_context(context);
        self.text_stats
            .as_ref()
            .unwrap()
            .1
            .assert_operation_context(context);
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.extents.reserved_bytes()
            + self
                .text_stats
                .as_ref()
                .map_or(0, |(_, stats)| stats.reserved_bytes())
    }

    #[cfg(test)]
    fn extent_mut(&mut self, kind: u16, collection: u64) -> &mut IndexExtentEntry {
        let entries = self.extents.as_mut_slice();
        let position = entries
            .binary_search_by_key(&(kind, collection), |entry| {
                (entry.record_type, entry.stream_id)
            })
            .unwrap();
        &mut entries[position]
    }

    #[cfg(test)]
    fn try_clone_in(&self, context: &JsonOperationContext) -> Result<Self> {
        let mut extents = IndexDirectory::with_capacity(
            self.extents.as_slice().len(),
            context,
            "operation_scratch",
        )?;
        for entry in self.extents.as_slice() {
            extents.push(*entry);
        }
        let mut source = Self::new(
            self.offset,
            self.payload_len,
            self.generation,
            self.commit_sequence_no,
            extents,
        )?;
        if let Some((generation, stats)) = &self.text_stats {
            source.text_stats = Some((*generation, stats.try_clone_in(context)?));
        }
        Ok(source)
    }

    // The size used to plan a vector read is the length of its framed record. The
    // text-side estimate below sums only text records across collections and
    // returns None if that sum overflows.
    pub(crate) fn encoded_len(&self, collection_id: &str) -> Option<u64> {
        // Convert the textual identity to its numeric wire key for this lookup. An
        // unparseable value supplies no length estimate and does not justify a
        // speculative read from another extent.
        self.extent(INDEX_VECTOR_SEGMENT, catalog_id_number(collection_id).ok()?)
            .map(|extent| extent.length)
    }

    pub(crate) fn text_encoded_len(&self) -> Option<u64> {
        // Text prewarming can cover several collections. Sum their frame lengths
        // with overflow detection so a wrapped estimate cannot admit a cache under
        // an artificially small byte budget.
        self.extents
            .as_slice()
            .iter()
            .filter(|entry| entry.record_type == INDEX_TEXT_SEGMENT)
            .try_fold(0_u64, |bytes, entry| bytes.checked_add(entry.length))
    }

    #[cfg(test)]
    pub(crate) fn load(
        &self,
        storage: &Storage,
        collection_id: &str,
    ) -> Result<Option<VectorIndexSegmentPayload>> {
        let mut file = storage.file.try_clone()?;
        self.read_from(&mut file, collection_id)
    }

    #[cfg(test)]
    pub(super) fn read_from<R: Read + Seek>(
        &self,
        reader: &mut R,
        collection_id: &str,
    ) -> Result<Option<VectorIndexSegmentPayload>> {
        self.read_with_mode(reader, collection_id, VectorDecodeMode::Retain)
            .map(|segment| segment.map(|segment| segment.payload.unwrap()))
    }

    pub(crate) fn load_cached(
        &self,
        storage: &Storage,
        collection_id: &str,
        reservation: crate::json::CacheReservation,
        context: &JsonOperationContext,
    ) -> Result<Option<CachedVectorSegment>> {
        // Use the open storage descriptor as the source of the bytes. Resolving
        // its pathname again could obtain a different file after a rewrite, even
        // if this directory still describes the original inode.
        let mut file = storage.file.try_clone()?;
        self.read_cached_from(&mut file, collection_id, reservation, context)
    }

    fn read_cached_from<R: Read + Seek>(
        &self,
        reader: &mut R,
        collection_id: &str,
        reservation: crate::json::CacheReservation,
        context: &JsonOperationContext,
    ) -> Result<Option<CachedVectorSegment>> {
        // The cache reservation is for the retained segment. Read-ahead has a
        // separate, temporary charge that ends after decoding. On an unusable
        // extent the cache reservation drops without a cache entry.
        let input = self.reserve_read_buffer(collection_id, context)?;
        let segment = self.read_with_capacity(
            reader,
            collection_id,
            VectorDecodeMode::Retain,
            input.reserved_bytes(),
            Some(context),
        )?;
        // Retain-mode decoding supplies an owned payload on success. Transfer the
        // cache reservation together with that payload only after the complete
        // record has validated.
        Ok(segment.map(|segment| CachedVectorSegment::new(segment.payload.unwrap(), reservation)))
    }

    pub(crate) fn reserve_read_buffer(
        &self,
        collection_id: &str,
        context: &JsonOperationContext,
    ) -> Result<WorkspaceReservation> {
        let mut input = context.workspace_reservation();
        // Choose a preferred input buffer from the encoded record length. Zero is a
        // valid admission, allowing the same read to proceed unbuffered under
        // memory pressure.
        let preferred = self.encoded_len(collection_id).unwrap_or(0).min(64 * 1024) as usize;
        input.grow_to(0, preferred, "operation_scratch")?;
        Ok(input)
    }

    // The visitor's provisional work is usable only when the entire extent validates.
    pub(crate) fn visit(
        &self,
        storage: &Storage,
        collection: &CollectionInfo,
        input: &WorkspaceReservation,
        visit: impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<bool> {
        // Visiting uses the open physical file that supplied the catalog view. A
        // reader can stream coordinate rows to scoring without first admitting a
        // complete resident vector segment.
        let mut file = storage.file.try_clone()?;
        self.visit_from(&mut file, collection, input, visit)
    }

    fn visit_from<R: Read + Seek>(
        &self,
        reader: &mut R,
        collection: &CollectionInfo,
        input: &WorkspaceReservation,
        mut visit: impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<bool> {
        // The decoder and the downstream visitor have different failure channels.
        // Save the callback's error separately while validation decides whether
        // the extent can be trusted at all.
        let mut error = None;
        let segment = self.read_with_mode(
            reader,
            &collection.collection_id,
            VectorDecodeMode::Visit {
                collection,
                input,
                error: &mut error,
                visit: &mut |chunk, _, values| {
                    // Reconstruct the externally formatted internal chunk ID from
                    // the numeric row key in stack storage. The visitor borrows
                    // this spelling and the decoded coordinates only for the
                    // duration of this row.
                    let mut id = [0; 24];
                    visit(catalog_id_text(&mut id, b"chk_", chunk), values)
                },
            },
        )?;
        // Return false for an unusable frame. The caller must discard any rows
        // already delivered to its visitor. Once the frame validates, propagate the
        // visitor's saved error.
        if segment.is_none() {
            return Ok(false);
        }
        match error {
            Some(error) => Err(error),
            None => Ok(true),
        }
    }

    fn read_with_mode<R: Read + Seek>(
        &self,
        reader: &mut R,
        collection_id: &str,
        mode: VectorDecodeMode<'_>,
    ) -> Result<Option<ParsedVectorIndexSegment>> {
        // A visiting caller can reduce its retained input buffer all the way to
        // zero under pressure. Use that actual grant as the read capacity so a
        // later retry does not silently restore the discarded buffer.
        let capacity = match &mode {
            VectorDecodeMode::Visit { input, .. } => input.reserved_bytes(),
            _ => 64 * 1024,
        };
        // Charge visitor reads to the operation that owns the buffer. The decoder
        // variants without an input guard do not provide a search context for the
        // index I/O counters on this path.
        let context = match &mode {
            VectorDecodeMode::Visit { input, .. } => Some(input.context()),
            _ => None,
        };
        self.read_with_capacity(reader, collection_id, mode, capacity, context)
    }

    fn read_with_capacity<R: Read + Seek>(
        &self,
        reader: &mut R,
        collection_id: &str,
        mode: VectorDecodeMode<'_>,
        capacity: usize,
        context: Option<&JsonOperationContext>,
    ) -> Result<Option<ParsedVectorIndexSegment>> {
        // Let the extent reader locate and verify the outer frame. The vector
        // decoder receives only the permitted body bytes and checks the embedded
        // segment against the checkpoint's covered commit.
        self.read_extent(
            reader,
            INDEX_VECTOR_SEGMENT,
            collection_id,
            capacity,
            context,
            |source, remaining| {
                let segment = decode_vector_index_segment_in(
                    &mut IndexPayloadReader::new(source, remaining, self.offset),
                    self.commit_sequence_no,
                    self.offset,
                    mode,
                )?;
                // Return the decoded collection number along with the segment. The
                // extent layer compares it with the directory key so a valid frame
                // belonging to another collection cannot satisfy this read.
                Ok((segment.collection_number, segment))
            },
        )
    }

    // Events are provisional until this whole snapshot frame has validated.
    // Consumer failure stops events, not validation, and never replays a visitor.
    pub(crate) fn visit_text(
        &self,
        storage: &Storage,
        collection_id: &str,
        input: &WorkspaceReservation,
        visit: impl FnMut(TextSegmentEntry<'_>) -> Result<()>,
    ) -> Result<bool> {
        // Read from the file already owned by Storage. The cloned descriptor
        // refers to that file even if its pathname has subsequently been replaced.
        let mut file = storage.file.try_clone()?;
        self.visit_text_from(&mut file, collection_id, input, visit)
    }

    #[cfg(test)]
    pub(crate) fn stage_text(
        &self,
        storage: &Storage,
        collection_id: &str,
        disk: &std::sync::Arc<crate::temporary::TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Option<TextSpool>> {
        let number = catalog_id_number(collection_id)?;
        if self.extent(INDEX_TEXT_SEGMENT, number).is_none() {
            return Ok(None);
        }
        let directory = storage.path().parent().unwrap_or_else(|| Path::new("."));
        let mut staging = TextStaging::new(directory, disk.clone(), context)?;
        self.stage_text_into(storage, collection_id, &mut staging, context)
            .map(|body| body.map(|extent| TextSpool { staging, extent }))
    }

    pub(crate) fn stage_text_into(
        &self,
        storage: &Storage,
        collection_id: &str,
        staging: &mut TextStaging,
        context: &JsonOperationContext,
    ) -> Result<Option<TextExtent>> {
        // A collection without an extent can be rebuilt from its catalog data.
        // Check for it before reserving input or output buffers or starting an
        // append to the staging file.
        let number = catalog_id_number(collection_id)?;
        if self.extent(INDEX_TEXT_SEGMENT, number).is_none() {
            return Ok(None);
        }
        // Keep input buffering and staging output buffering under separate grants.
        // A low-memory retry can surrender either or both while leaving the
        // staging owner and the previously accepted extents intact.
        let mut input = context.workspace_reservation();
        let mut output = context.workspace_reservation();
        let mut buffered = true;
        loop {
            // The caller has already admitted the necessary staging owner.
            if buffered {
                input.grow_to(0, 64 * 1024, "operation_scratch")?;
                output.grow_to(0, 64 * 1024, "operation_scratch")?;
            }
            // Each attempt seeks through the saved extent directory on the open
            // storage file. Reopening by pathname here could pair the old
            // directory offsets with an unrelated file.
            let mut file = storage.file.try_clone()?;
            // The decoder copies a candidate body to staging as it reads.
            // read_extent still has to check the outer record and embedded
            // collection. Only an accepted result may advance staging's committed
            // end.
            let staged = self.read_extent(
                &mut file,
                INDEX_TEXT_SEGMENT,
                collection_id,
                input.reserved_bytes(),
                Some(context),
                |source, remaining| {
                    // Pass the source checkpoint's covered sequence into staging.
                    // The copied text body must agree with its source view before
                    // it can be reused and retagged for a newer checkpoint.
                    let body = staging.copy_body_from(
                        source,
                        remaining,
                        self.commit_sequence_no,
                        context,
                        output.reserved_bytes(),
                    )?;
                    // The copied body is still tentative when its collection
                    // number is returned. Matching the directory's entry and
                    // passing the outer checksum checks are prerequisites for
                    // retaining this span in the staging file.
                    Ok((body.collection_number(), body))
                },
            );
            // Unlike visitor callbacks, this tentative file append has an explicit
            // rollback. A memory refusal can be retried after restoring staging's
            // previous end and releasing both optional buffers. Other unsuccessful
            // reads also roll back before returning.
            match staged {
                Err(error)
                    if crate::json::is_memory_budget_error(&error)
                        && input.reserved_bytes() + output.reserved_bytes() > 0 =>
                {
                    // Restore staging's last accepted length before attempting a
                    // second read. If rollback fails, return that failure because
                    // the shared file is no longer a valid basis for appending
                    // another candidate.
                    staging.rollback()?;
                    input.retain(0);
                    output.retain(0);
                    // Disable optional buffers for the rest of this operation. A
                    // second resource failure with nothing left to release
                    // propagates through the rollback-and-return branch.
                    buffered = false;
                }
                Ok(Some(body)) => {
                    // Advance the committed end of the temporary file only for an
                    // accepted extent. Return its descriptor after that step so
                    // later writers can safely append another collection after
                    // these bytes.
                    staging.finish_copy(&body)?;
                    return Ok(Some(body));
                }
                result => {
                    // Roll back on a missing or invalid extent as well as on a
                    // hard read error. A decoder can have copied a prefix before
                    // the outer checks decide that the candidate cannot be used.
                    staging.rollback()?;
                    return result;
                }
            }
        }
    }

    fn visit_text_from<R: Read + Seek>(
        &self,
        reader: &mut R,
        collection_id: &str,
        input: &WorkspaceReservation,
        mut visit: impl FnMut(TextSegmentEntry<'_>) -> Result<()>,
    ) -> Result<bool> {
        // Preserve the first consumer error independently of the decoder result.
        // The decoder can stop delivering events while still consuming the
        // remaining bytes needed to assess the segment.
        let mut error = None;
        let segment = self.read_extent(
            reader,
            INDEX_TEXT_SEGMENT,
            collection_id,
            input.reserved_bytes(),
            Some(input.context()),
            |source, remaining| {
                let segment = decode_text_index_segment_with(
                    &mut IndexPayloadReader::new(source, remaining, self.offset),
                    self.commit_sequence_no,
                    self.offset,
                    TextDecodeMode::Visit {
                        error: &mut error,
                        visit: &mut |entry| {
                            // Count visits when a chunk or posting event is
                            // delivered, before the consumer decides whether to
                            // accept it. Header and term events contribute neither
                            // counter.
                            match entry {
                                TextSegmentEntry::Chunk { .. } => input.context().search_cost_add(
                                    crate::search_cost::Counter::TextDocumentVisits,
                                    1,
                                ),
                                TextSegmentEntry::Posting { .. } => {
                                    input.context().search_cost_add(
                                        crate::search_cost::Counter::TextPostingVisits,
                                        1,
                                    )
                                }
                                _ => {}
                            }
                            visit(entry)
                        },
                    },
                    input.context(),
                )?;
                Ok((segment.collection_number, segment))
            },
        );
        // If decoder scratch is refused after a callback has failed, return the
        // recorded callback error. A corrupt or truncated frame instead gives the
        // caller a chance to search committed data.
        let segment = match segment {
            Err(failure) if crate::json::is_memory_budget_error(&failure) => {
                // A late budget failure can prevent validation from reaching the
                // end. Prefer an already recorded consumer failure in that case,
                // otherwise expose the resource refusal that stopped decoding.
                return Err(error.unwrap_or(failure));
            }
            result => result?,
        };
        // A missing, damaged, or truncated extent invalidates the provisional
        // events. Report false so callers can discard those results and choose
        // another index source before interpreting a consumer error.
        if segment.is_none() {
            return Ok(false);
        }
        // Reaching this branch establishes that the entire segment was accepted. A
        // previously recorded consumer failure now ends the call even if all
        // remaining bytes and checksums were valid.
        match error {
            Some(error) => Err(error),
            None => Ok(true),
        }
    }

    // Run the caller's decoder at most once on the selected extent. This matters
    // for a visitor that has already delivered rows or a copy that has written
    // temporary bytes. A successful return requires both those decoded fields and
    // the original frame checks.
    fn read_extent<R: Read + Seek, T>(
        &self,
        reader: &mut R,
        kind: u16,
        collection_id: &str,
        capacity: usize,
        context: Option<&JsonOperationContext>,
        decode: impl FnOnce(&mut dyn Read, u64) -> std::result::Result<(u64, T), IndexReadError>,
    ) -> Result<Option<T>> {
        // The directory is keyed by the numeric collection ID and record kind. A
        // miss can mean the snapshot has no segment of this kind for the
        // collection. Return before seeking or recording a read request.
        let collection_number = catalog_id_number(collection_id)?;
        let Some(extent) = self.extent(kind, collection_number) else {
            return Ok(None);
        };
        let mut counted = crate::search_cost::IndexReader {
            source: reader,
            context,
        };
        let reader = &mut counted;
        let loaded = (|| {
            // Resolve the stored relative offset against the checkpoint's own
            // origin and generation. An extent must fit within the segment range
            // before the reader may seek to it. The directory entry alone cannot
            // authorize an arbitrary file offset.
            let (offset, length) =
                index_extent_range(self.payload_len, extent, self.generation, self.offset)?;
            // Charge requested bytes once from the checked extent size. The wrapped
            // reader separately counts actual I/O, which can be smaller when the
            // frame fails early.
            if let Some(context) = context {
                context.search_cost_add(crate::search_cost::Counter::IndexRequestedBytes, length);
            }
            // An extent remembered at open time is a locator into a rebuildable
            // suffix. Validate the frame at that offset again before exposing the
            // decoded result: the file can still exist after its index bytes have
            // been replaced.
            reader.seek(SeekFrom::Start(offset))?;
            let mut header = [0; INDEX_RECORD_HEADER_LEN];
            reader.read_exact(&mut header)?;
            // Check the header against the locator before using its claimed
            // payload length. A different valid record at the same offset cannot
            // stand in for the requested extent.
            let mut record =
                parse_index_record_header(&header, extent, self.generation, offset, length)?;
            // The inner segment CRC occupies the last four bytes. Reject a payload
            // that cannot hold it before subtracting the footer from the decoder's
            // allowance.
            if record.payload.len < 4 {
                validate_record_segment_checksum(&record, self.offset)?;
            }
            let mut checked = SegmentChecksumReader::new(reader, &header, record.payload.len);
            let decoded = {
                // Use the caller's buffer capacity, including zero when optional
                // read-ahead was relinquished. Decoding and draining share this
                // one reader so prefetched bytes cannot disappear from the
                // integrity check.
                let mut source = BufReader::with_capacity(capacity, &mut checked);
                let decoded = decode(&mut source, record.payload.len - 4);
                // An actual I/O error from the decoder makes further reads in this
                // attempt unreliable. A semantic or memory failure can still be
                // compared with the outcome of the frame checks below after the
                // remaining bytes have been read.
                if matches!(&decoded, Err(IndexReadError::Io(_))) {
                    return decoded.map(|(_, value)| value);
                }
                // Finish checksumming this snapshot frame even after a semantic error.
                std::io::copy(&mut source, &mut std::io::sink())?;
                decoded
            };
            // Copying to a sink stops normally at EOF, including an EOF short of
            // the promised extent. Compare byte counts explicitly before
            // interpreting any digest or releasing a decoded result.
            if checked.bytes_read != record.payload.len {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            // Check the immutable directory's whole-record CRC first, then the
            // header's payload CRC, then the segment's own footer. Each covers a
            // different range of the same bytes read during this call.
            if checked.record_checksum() != extent.extent_record_checksum {
                return Err(index_extent_diagnostic(
                    "index_extent_record_checksum_mismatch",
                    if kind == INDEX_VECTOR_SEGMENT {
                        "vector extent no longer matches its snapshot checksum"
                    } else {
                        "text extent no longer matches its snapshot checksum"
                    },
                    record.offset,
                    record.index_record_id,
                )
                .into());
            }
            if checked.payload_checksum.finish() as u64 != record.payload.checksum {
                return Err(index_extent_diagnostic(
                    "index_record_payload_checksum_mismatch",
                    if kind == INDEX_VECTOR_SEGMENT {
                        "vector payload no longer matches its record checksum"
                    } else {
                        "text payload no longer matches its record checksum"
                    },
                    record.payload.offset,
                    record.index_record_id,
                )
                .into());
            }
            // Compare the segment footer with the digest captured at the segment
            // body boundary. The enclosing payload and whole-record checks can
            // both agree even if a stale inner checksum was written into the
            // footer.
            record.segment_checksum_valid = read_u32(&checked.tail) == checked.segment_checksum;
            validate_record_segment_checksum(&record, self.offset)?;
            // A valid CRC only establishes that these bytes match their recorded
            // values. Check the collection encoded in the segment against the
            // directory key as the last condition for returning it.
            let (decoded_collection, segment) = decoded?;
            if decoded_collection != collection_number {
                return Err(IndexReadError::Invalid(index_suffix_diagnostic(
                    "index_extent_table_mismatch",
                    if kind == INDEX_VECTOR_SEGMENT {
                        "vector segment collection does not match its snapshot reference"
                    } else {
                        "text segment collection does not match its snapshot reference"
                    },
                    record.offset,
                )));
            }
            Ok(segment)
        })();
        // An unusable or truncated extent gives the caller a chance to search
        // committed data. Other read failures and operation budget failures must
        // propagate; silently falling back would lose the reason this read could not
        // finish.
        match loaded {
            // Some is the only outcome that transfers a usable decoded result. A
            // callback can have already produced rows by the time the final
            // checksum fails, so its caller must discard any effects collected for
            // this attempt when it receives None.
            Ok(segment) => Ok(Some(segment)),
            Err(IndexReadError::Invalid(_)) => Ok(None),
            // A concurrent writer may truncate or replace this advisory suffix.
            Err(IndexReadError::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                Ok(None)
            }
            Err(IndexReadError::Io(error)) => Err(error.into()),
            Err(IndexReadError::Resource(error)) => Err(error),
        }
    }
}

#[cfg(test)]
impl Clone for IndexSegmentSource {
    fn clone(&self) -> Self {
        self.try_clone_in(&JsonOperationContext::standalone())
            .expect("test snapshot copy admission")
    }
}

struct SegmentChecksumReader<R> {
    // Underlying stream capped at the record's payload length so draining it stops
    // before the next header.
    source: std::io::Take<R>,
    // CRC of the previously read record header, saved for combination with the full
    // payload CRC.
    header_checksum: Crc32c,
    // Running CRC of all payload bytes delivered, including the segment footer when
    // reached.
    payload_checksum: Crc32c,
    // Snapshot of the running CRC at the end of the segment body, before the footer
    // is included.
    segment_checksum: u32,
    // Four footer bytes collected in order even when the underlying read splits
    // them.
    tail: [u8; 4],
    // Payload bytes preceding the stored segment CRC. Separates the body from its
    // footer in every read.
    body_len: u64,
    // Payload bytes consumed so far, used to find the footer boundary and detect a
    // short physical read.
    bytes_read: u64,
}

impl<R: Read> SegmentChecksumReader<R> {
    fn new(source: R, header: &[u8; INDEX_RECORD_HEADER_LEN], payload_len: u64) -> Self {
        // Seed the record checksum with the complete outer header. The payload
        // still starts at zero bytes read, so the two checks are independent until
        // record_checksum combines them using the exact delivered payload length.
        let mut header_checksum = Crc32c::default();
        header_checksum.update(header);
        Self {
            source: source.take(payload_len),
            header_checksum,
            payload_checksum: Crc32c::default(),
            segment_checksum: 0,
            tail: [0; 4],
            body_len: payload_len.saturating_sub(4),
            bytes_read: 0,
        }
    }

    // Append the payload CRC with its byte length to the header CRC. CRC
    // concatenation gives the checksum of the complete record without rereading
    // either part.
    fn record_checksum(&self) -> u32 {
        let mut checksum = self.header_checksum;
        checksum.append_crc(self.payload_checksum.finish(), self.bytes_read);
        checksum.finish()
    }
}

impl<R: Read> Read for SegmentChecksumReader<R> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let count = self.source.read(output)?;
        // One read may straddle the body/CRC boundary, or may contain only part of
        // the CRC. Hash the body portion first, capture its final CRC exactly at
        // the boundary, then collect the remaining bytes as both footer and payload
        // data.
        let body_count = self
            .body_len
            .saturating_sub(self.bytes_read)
            .min(count as u64) as usize;
        self.payload_checksum.update(&output[..body_count]);
        if self.bytes_read <= self.body_len && self.bytes_read + body_count as u64 == self.body_len
        {
            self.segment_checksum = self.payload_checksum.finish();
        }
        // The footer's destination index is the number of footer bytes already
        // accepted. Subtracting the body length with saturation gives zero on the
        // first crossing, and the limited source keeps every later index within
        // the four-byte footer.
        if body_count < count {
            let start = self.bytes_read.saturating_sub(self.body_len) as usize;
            self.tail[start..start + count - body_count]
                .copy_from_slice(&output[body_count..count]);
            self.payload_checksum.update(&output[body_count..count]);
        }
        self.bytes_read += count as u64;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn text_encoded_bytes_exclude_vector_extents_and_detect_overflow() {
        let context = JsonOperationContext::standalone();
        for (lengths, expected) in [
            ([0, 0], Some(0)),
            ([71, 83], Some(154)),
            ([u64::MAX, 1], None),
        ] {
            let mut entries =
                IndexDirectory::with_capacity(3, &context, "operation_scratch").unwrap();
            for (kind, id, length) in [
                (INDEX_TEXT_SEGMENT, 1, lengths[0]),
                (INDEX_TEXT_SEGMENT, 2, lengths[1]),
                (INDEX_VECTOR_SEGMENT, 1, u64::MAX),
            ] {
                entries.push(IndexExtentEntry {
                    index_record_id: id,
                    record_type: kind,
                    stream_id: id,
                    suffix_relative_offset: 0,
                    length,
                    checkpoint_generation: 1,
                    extent_record_checksum: 0,
                });
            }
            let source = IndexSegmentSource::new(0, 0, 1, 1, entries).unwrap();
            assert_eq!(source.text_encoded_len(), expected);
        }
    }

    #[test]
    fn snapshot_directory_retains_capacity_and_separates_segment_kinds() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;
        for seed in 0..100usize {
            let count = seed % 23;
            let capacity = count * 2 + 1;
            let needed = capacity * std::mem::size_of::<IndexExtentEntry>();
            let make = |context: &JsonOperationContext| {
                let mut entries =
                    IndexDirectory::with_capacity(capacity, context, "operation_scratch")?;
                entries.push(IndexExtentEntry {
                    index_record_id: 1,
                    record_type: INDEX_COMMIT_METADATA,
                    stream_id: 0,
                    suffix_relative_offset: 0,
                    length: 0,
                    checkpoint_generation: 17,
                    extent_record_checksum: 0,
                });
                for kind in [INDEX_VECTOR_SEGMENT, INDEX_TEXT_SEGMENT] {
                    for id in (0..count as u64).rev() {
                        entries.push(IndexExtentEntry {
                            index_record_id: id + 2,
                            record_type: kind,
                            stream_id: u64::MAX - id,
                            suffix_relative_offset: id,
                            length: id + u64::from(kind),
                            checkpoint_generation: 17,
                            extent_record_checksum: id as u32,
                        });
                    }
                }
                IndexSegmentSource::new(17, 1000, 17, 11, entries)
            };
            for available in [needed - 1, needed, needed + 1] {
                let total = needed + 4096;
                let ledger = Arc::new(WorkLedger::with_limits(total, total, total));
                let context = JsonOperationContext::new(ledger.clone());
                let other = JsonOperationContext::new(ledger);
                let mut blocker = other.workspace_reservation();
                blocker
                    .grow_to(total - available, total - available, "operation_scratch")
                    .unwrap();
                if available < needed {
                    assert_eq!(
                        make(&context).unwrap_err().code(),
                        ErrorCode::ResourceExhausted
                    );
                } else {
                    let source = make(&context).unwrap();
                    assert_eq!(source.extents.as_slice().len(), count * 2);
                    for id in 0..count as u64 {
                        for kind in [INDEX_TEXT_SEGMENT, INDEX_VECTOR_SEGMENT] {
                            let extent = source.extent(kind, u64::MAX - id).unwrap();
                            assert_eq!(extent.length, id + u64::from(kind));
                            assert_eq!(extent.checkpoint_generation, source.generation);
                        }
                    }
                    assert!(source.extent(INDEX_COMMIT_METADATA, 0).is_none());
                    let mut remaining = other.workspace_reservation();
                    remaining
                        .grow_to(available - needed, available - needed, "operation_scratch")
                        .unwrap();
                    assert!(context
                        .workspace_reservation()
                        .grow_to(1, 1, "operation_scratch")
                        .is_err());
                    if count > 0 {
                        assert_eq!(
                            source.try_clone_in(&context).unwrap_err().code(),
                            ErrorCode::ResourceExhausted
                        );
                    }
                    drop(remaining);
                    drop(source);
                }
                context
                    .workspace_reservation()
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
                drop(blocker);
                context
                    .workspace_reservation()
                    .grow_to(total, total, "operation_scratch")
                    .unwrap();
            }
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                needed, needed, needed,
            )));
            // The zero-collection fixture intentionally has room for only the metadata slot.
            if count > 0 {
                let mut duplicate =
                    IndexDirectory::with_capacity(2, &context, "operation_scratch").unwrap();
                let entry = IndexExtentEntry {
                    index_record_id: 2,
                    record_type: INDEX_TEXT_SEGMENT,
                    stream_id: 7,
                    suffix_relative_offset: 0,
                    length: 0,
                    checkpoint_generation: 17,
                    extent_record_checksum: 0,
                };
                duplicate.push(entry);
                duplicate.push(entry);
                assert_eq!(
                    IndexSegmentSource::new(17, 0, 17, 11, duplicate)
                        .unwrap_err()
                        .code(),
                    ErrorCode::FileFormatError
                );
            }
            context
                .workspace_reservation()
                .grow_to(needed, needed, "operation_scratch")
                .unwrap();
        }
    }

    fn fixture(
        seed: usize,
    ) -> (
        Vec<u8>,
        LazyVectorCheckpoint,
        Vec<VectorIndexSegmentPayload>,
    ) {
        fixture_with_model(seed, Some("m\u{4e2d}".repeat(seed % 5 + 1)))
    }

    fn fixture_with_model(
        seed: usize,
        model: Option<String>,
    ) -> (
        Vec<u8>,
        LazyVectorCheckpoint,
        Vec<VectorIndexSegmentPayload>,
    ) {
        fixture_with_generation(seed, model, 37)
    }

    fn fixture_with_generation(
        seed: usize,
        model: Option<String>,
        generation: u64,
    ) -> (
        Vec<u8>,
        LazyVectorCheckpoint,
        Vec<VectorIndexSegmentPayload>,
    ) {
        let count = seed % 4;
        let dim = [1, 17, 1025][seed % 3];
        let vectors = (1..=2)
            .map(|collection| VectorIndexSegmentPayload {
                collection_id: format!("col_{collection}"),
                covered_commit_sequence_no: 2,
                dim,
                distance_metric: DistanceMetric::DotProduct,
                embedding_model: model.clone(),
                vectors_normalized: false,
                vectors: (0..count)
                    .map(|index| VectorIndexEntry {
                        chunk_id: format!("chk_{}", collection * 100 + index),
                        embedding_id: format!("emb_{}", collection * 100 + index),
                        vector: (0..dim)
                            .map(|value| {
                                f32::from_bits(match value % 4 {
                                    0 => 0x8000_0000,
                                    1 => 1,
                                    2 => 0x7f7f_ffff,
                                    _ => (value as f32).to_bits(),
                                })
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let checkpoint = IndexCheckpointPayload {
            metadata: IndexCheckpointMetadata {
                covered_last_batch_id: 1,
                covered_last_commit_sequence_no: 2,
                coverage_hash: 17,
                collection_count: 2,
                live_document_count: usize::from(count > 0) * 2,
                live_chunk_count: count * 2,
                live_embedding_count: count * 2,
                text_segment_count: 2,
                vector_segment_count: 2,
                build_mode: "inline_full_checkpoint".into(),
            },
            text_segments: vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| TextIndexSegmentPayload {
                    collection_id: vector.collection_id.clone(),
                    covered_commit_sequence_no: 2,
                    chunks: vector
                        .vectors
                        .iter()
                        .map(|entry| TextIndexChunkEntry {
                            chunk_id: entry.chunk_id.clone(),
                            document_id: format!("doc_{}", index + 1),
                            token_count: 1,
                        })
                        .collect(),
                    terms: vec![TextIndexTermEntry {
                        term: "needle".into(),
                        postings: vector
                            .vectors
                            .iter()
                            .map(|entry| TextIndexPostingEntry {
                                chunk_id: entry.chunk_id.clone(),
                                term_frequency: 1,
                            })
                            .collect(),
                    }],
                })
                .collect(),
            vector_segments: vectors.clone(),
        };
        let mut bytes = vec![7; 37];
        bytes.extend_from_slice(
            &build_index_checkpoint_suffix_with_generation(&checkpoint, 37, generation).unwrap(),
        );
        let parsed = read_index_checkpoint_suffix(
            &mut Cursor::new(&bytes),
            37,
            bytes.len() as u64,
            37,
            1,
            2,
            IndexReadOptions {
                budget_bytes: u64::MAX,
                mode: IndexReadMode::LazyVectors,
            },
        )
        .unwrap()
        .unwrap();
        assert!(parsed.checkpoint.is_none());
        assert!(parsed.diagnostics.is_empty());
        (bytes, parsed.lazy_vectors.unwrap(), vectors)
    }

    #[test]
    fn text_statistics_follow_validated_generation_and_keep_snapshot_credit_until_drop() {
        use crate::catalog::Catalog;
        use crate::index::text::TextIndex;
        for seed in 0..20 {
            for generation in [0, 1, 37, u64::MAX - seed as u64] {
                let (bytes, checkpoint, expected) = fixture_with_generation(seed, None, generation);
                assert_eq!(checkpoint.vectors.generation, generation);
                assert!(checkpoint.vectors.text_stats(37, 2).is_none());
                let mut catalog = Catalog::default();
                for segment in &expected {
                    let collection = collection_for(segment);
                    catalog
                        .collections
                        .insert(collection.collection_id.clone(), collection);
                }
                let index = TextIndex::try_build_from_segments(checkpoint.text_segments).unwrap();
                let directory = checkpoint.vectors.reserved_bytes();
                let summary_bytes = 2 * std::mem::size_of::<(u64, usize)>();
                let owned = directory + summary_bytes;
                let total = 2 * owned + 8;
                let ledger = std::sync::Arc::new(crate::json::WorkLedger::with_limits(
                    2 * total,
                    total,
                    total,
                ));
                let context = JsonOperationContext::new(ledger.clone());
                let sibling = JsonOperationContext::new(ledger);
                let summary = TextStatsSummary::build(&index, &catalog, &context).unwrap();
                let source = checkpoint
                    .vectors
                    .try_clone_in(&context)
                    .unwrap()
                    .with_text_stats(summary);
                drop((checkpoint.vectors, index, catalog));
                assert_eq!(source.reserved_bytes(), owned);
                let stats = source.text_stats(37, 2).unwrap();
                assert_eq!(stats.term_count(), 1);
                for segment in &expected {
                    assert_eq!(
                        stats.term_count_in_collection(&segment.collection_id),
                        usize::from(!segment.vectors.is_empty()),
                    );
                    let actual = source
                        .read_from(&mut Cursor::new(&bytes), &segment.collection_id)
                        .unwrap()
                        .unwrap();
                    assert_eq!(actual.vectors.len(), segment.vectors.len());
                }
                assert!(source.text_stats(38, 2).is_none());
                assert!(source.text_stats(37, 3).is_none());
                let mut cloned = source.try_clone_in(&context).unwrap();
                assert_eq!(cloned.reserved_bytes(), owned);
                assert_eq!(cloned.text_stats(37, 2).unwrap().term_count(), 1);
                cloned.generation = generation.wrapping_add(1);
                assert!(cloned.text_stats(37, 2).is_none());
                cloned.generation = generation;
                cloned.offset += 1;
                assert!(cloned.text_stats(37, 2).is_none());
                cloned.offset -= 1;
                cloned.commit_sequence_no += 1;
                assert!(cloned.text_stats(37, 2).is_none());
                drop(context);
                let mut work = sibling.workspace_reservation();
                work.grow_to(8, 8, "operation_scratch").unwrap();
                assert!(work.grow_to(9, 9, "operation_scratch").is_err());
                drop(work);
                drop(cloned);
                let mut work = sibling.workspace_reservation();
                work.grow_to(owned + 8, owned + 8, "operation_scratch")
                    .unwrap();
                assert!(work
                    .grow_to(owned + 9, owned + 9, "operation_scratch")
                    .is_err());
                drop(work);
                drop(source);
                let mut work = sibling.workspace_reservation();
                work.grow_to(total, total, "operation_scratch").unwrap();
                assert!(work
                    .grow_to(total + 1, total + 1, "operation_scratch")
                    .is_err());
            }
        }
    }

    fn collection_for(segment: &VectorIndexSegmentPayload) -> CollectionInfo {
        CollectionInfo {
            collection_id: catalog_id_string(
                "col_",
                catalog_id_number(&segment.collection_id).unwrap(),
            ),
            name: "vectors".into(),
            embedding_dim: segment.dim,
            distance_metric: segment.distance_metric,
            embedding_model: segment.embedding_model.clone(),
            vectors_normalized: segment.vectors_normalized,
            tokenizer: Default::default(),
            vector_index: Default::default(),
            created_at: "test".into(),
        }
    }

    #[test]
    fn cached_vectors_hold_actual_capacity_until_drop_and_release_failed_loads() {
        for seed in 0..100 {
            let (bytes, checkpoint, expected) = fixture_with_model(
                seed,
                (seed % 2 == 0).then(|| "m\u{4e2d}\u{1f680}".repeat(seed * 101)),
            );
            let source = checkpoint.vectors;
            let id = &expected[0].collection_id;
            let encoded = source.encoded_len(id).unwrap();
            let probe = JsonOperationContext::standalone();
            let size = CachedVectorSegment::reserve(&probe, usize::MAX, encoded)
                .unwrap()
                .reserved_bytes();
            let context = JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(0, 0, size),
            ));
            assert!(CachedVectorSegment::reserve(&context, size - 1, encoded).is_none());
            let held = CachedVectorSegment::reserve(&context, size, encoded).unwrap();
            assert!(context.cache_reservation(1, usize::MAX).is_none());
            let cached = source
                .read_cached_from(&mut Cursor::new(&bytes), id, held, &context)
                .unwrap()
                .unwrap();
            assert_eq!(
                encode_vector_index_segment_payload(&cached).unwrap(),
                encode_vector_index_segment_payload(&expected[0]).unwrap()
            );
            let live = cached.reserved_bytes();
            assert!(live <= size);
            let remainder = context
                .cache_reservation((size - live) as u128, size)
                .unwrap();
            assert!(context.cache_reservation(1, size).is_none());
            drop(cached);
            let all_released = context.cache_reservation(live as u128, size).unwrap();
            drop(all_released);
            drop(remainder);
            let extent = *source
                .extent(INDEX_VECTOR_SEGMENT, catalog_id_number(id).unwrap())
                .unwrap();
            let offset = (source.offset + extent.suffix_relative_offset) as usize;
            for truncate in [false, true] {
                let mut damaged = bytes.clone();
                if truncate {
                    damaged.truncate(offset + extent.length as usize - 1);
                } else {
                    damaged[offset + INDEX_RECORD_HEADER_LEN] ^= 1;
                }
                let held = CachedVectorSegment::reserve(&context, size, encoded).unwrap();
                assert!(source
                    .read_cached_from(&mut Cursor::new(damaged), id, held, &context)
                    .unwrap()
                    .is_none());
                assert!(context.cache_reservation(size as u128, size).is_some());
            }
        }
    }

    #[test]
    fn lazy_vectors_match_full_decoding_and_reject_changed_snapshot_frames() {
        for seed in 0..100 {
            let (bytes, checkpoint, expected) = fixture(seed);
            assert_eq!(checkpoint.text_segments.len(), 2);
            let source = checkpoint.vectors;
            assert!(source.encoded_len("col_3").is_none());
            assert!(source
                .read_from(&mut Cursor::new(&bytes), "col_3")
                .unwrap()
                .is_none());
            for segment in expected {
                let loaded = source
                    .read_from(&mut Cursor::new(&bytes), &segment.collection_id)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    encode_vector_index_segment_payload(&loaded).unwrap(),
                    encode_vector_index_segment_payload(&segment).unwrap()
                );
                let extent = *source
                    .extent(
                        INDEX_VECTOR_SEGMENT,
                        catalog_id_number(&segment.collection_id).unwrap(),
                    )
                    .unwrap();
                let offset = (source.offset + extent.suffix_relative_offset) as usize;
                let length = extent.length as usize;
                assert_eq!(
                    source.encoded_len(&segment.collection_id),
                    Some(extent.length)
                );
                let mut damaged = bytes.clone();
                damaged[offset + seed % length] ^= 1;
                assert!(source
                    .read_from(&mut Cursor::new(&damaged), &segment.collection_id)
                    .unwrap()
                    .is_none());
                for end in [
                    offset,
                    offset + INDEX_RECORD_HEADER_LEN - 1,
                    offset + length - 1,
                ] {
                    assert!(source
                        .read_from(&mut Cursor::new(&bytes[..end]), &segment.collection_id)
                        .unwrap()
                        .is_none());
                }
                let mut changed = source.clone();
                changed.generation += 1;
                assert!(changed
                    .read_from(&mut Cursor::new(&bytes), &segment.collection_id)
                    .unwrap()
                    .is_none());
                changed = source.clone();
                changed.payload_len = 0;
                assert!(changed
                    .read_from(&mut Cursor::new(&bytes), &segment.collection_id)
                    .unwrap()
                    .is_none());
                changed = source.clone();
                changed.commit_sequence_no += 1;
                assert!(changed
                    .read_from(&mut Cursor::new(&bytes), &segment.collection_id)
                    .unwrap()
                    .is_none());
            }
        }
    }

    #[test]
    fn text_search_retries_without_read_ahead_and_uses_actual_explanation_dimensions() {
        use crate::index::text::{TextIndex, TextSearchOptions};
        use crate::json::WorkLedger;
        use std::sync::Arc;
        struct Fixture {
            storage: Option<Storage>,
            root: PathBuf,
        }
        impl Drop for Fixture {
            fn drop(&mut self) {
                drop(self.storage.take());
                let _ = std::fs::remove_dir_all(&self.root);
            }
        }
        let n = 4000usize;
        let checkpoint = IndexCheckpointPayload {
            metadata: IndexCheckpointMetadata {
                covered_last_batch_id: 1,
                covered_last_commit_sequence_no: 2,
                coverage_hash: 17,
                collection_count: 1,
                live_document_count: 1,
                live_chunk_count: n,
                live_embedding_count: 0,
                text_segment_count: 1,
                vector_segment_count: 1,
                build_mode: "inline_full_checkpoint".into(),
            },
            text_segments: vec![TextIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: 2,
                chunks: (1..=n)
                    .map(|chunk| TextIndexChunkEntry {
                        chunk_id: format!("chk_{chunk:016}"),
                        document_id: "doc_1".into(),
                        token_count: 1,
                    })
                    .collect(),
                terms: vec![
                    TextIndexTermEntry {
                        term: "alpha".into(),
                        postings: vec![TextIndexPostingEntry {
                            chunk_id: "chk_0000000000000001".into(),
                            term_frequency: 1,
                        }],
                    },
                    TextIndexTermEntry {
                        term: "empty".into(),
                        postings: Vec::new(),
                    },
                ],
            }],
            vector_segments: vec![VectorIndexSegmentPayload {
                collection_id: "col_1".into(),
                covered_commit_sequence_no: 2,
                dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vectors: Vec::new(),
            }],
        };
        let expected =
            TextIndex::try_build_from_segments(checkpoint.text_segments.clone()).unwrap();
        let mut bytes = vec![7; 37];
        bytes.extend(build_index_checkpoint_suffix_with_generation(&checkpoint, 37, 71).unwrap());
        let parsed = read_index_checkpoint_suffix(
            &mut Cursor::new(&bytes),
            37,
            bytes.len() as u64,
            37,
            1,
            2,
            IndexReadOptions {
                budget_bytes: 0,
                mode: IndexReadMode::LazyIndexes,
            },
        )
        .unwrap()
        .unwrap();
        let source = parsed.lazy_vectors.unwrap().vectors;
        let root =
            std::env::temp_dir().join(format!("cofferdb-text-read-ahead-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let mut fixture = Fixture {
            storage: None,
            root,
        };
        let mut storage = Storage::create(fixture.root.join("fixture.coffer")).unwrap();
        storage.file.set_len(0).unwrap();
        storage.file.seek(SeekFrom::Start(0)).unwrap();
        storage.file.write_all(&bytes).unwrap();
        fixture.storage = Some(storage);
        // First-pass query nodes, two usize fields per term, 24N scores,
        // two u64 ID arrays, and the five-byte reused term buffer coexist.
        let needed = crate::json::btree_workspace::<std::borrow::Cow<'_, str>, usize>(3)
            + 3 * 2 * std::mem::size_of::<usize>()
            + n * (std::mem::size_of::<(u64, usize, f64)>() + 2 * std::mem::size_of::<u64>())
            + 5;
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (needed as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * needed } else { limit },
                    0,
                    if parent { limit } else { 2 * needed },
                )));
                let options = TextSearchOptions::with_explanations(9);
                let (result, (inputs, matrices)) = TextIndex::with_segment_search_trace(|| {
                    TextIndex::search_from_segment(
                        &source,
                        fixture.storage.as_ref().unwrap(),
                        "col_1",
                        "alpha empty missing",
                        options,
                        &context,
                    )
                });
                assert_eq!(inputs, [64 * 1024, 0]);
                if delta < 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", needed.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                    }
                    assert!(matrices.is_empty());
                } else {
                    let hits = result.unwrap().unwrap();
                    let oracle = expected
                        .search(
                            "col_1",
                            "alpha empty missing",
                            options,
                            &JsonOperationContext::standalone(),
                        )
                        .unwrap();
                    assert_eq!(
                        format!("{:?}", hits.as_slice()),
                        format!("{:?}", oracle.as_slice())
                    );
                    assert_eq!(matrices, [(1, 1)]);
                    assert_eq!(hits.as_slice().len(), 1);
                    let hit = &hits.as_slice()[0];
                    let retained = std::mem::size_of_val(hit)
                        + 20
                        + std::mem::size_of_val(&hit.matched_terms[0])
                        + 5;
                    assert_eq!(context.available_workspace_bytes(), limit - retained);
                    let mut remaining = context.workspace_reservation();
                    remaining
                        .grow_to(limit - retained, limit - retained, "operation_scratch")
                        .unwrap();
                    let error = context
                        .workspace_reservation()
                        .grow_to(1, 1, "operation_scratch")
                        .unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), (limit + 1).to_string())));
                    drop(remaining);
                    drop(hits);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
        assert_eq!(
            std::fs::read(fixture.root.join("fixture.coffer")).unwrap(),
            bytes
        );
        let extent = source.extent(INDEX_TEXT_SEGMENT, 1).unwrap();
        let tail = source.offset + extent.suffix_relative_offset + extent.length - 1;
        let storage = fixture.storage.as_mut().unwrap();
        storage.file.seek(SeekFrom::Start(tail)).unwrap();
        storage.file.write_all(&[bytes[tail as usize] ^ 1]).unwrap();
        let limit = needed - 1;
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, 0, 2 * needed)));
        let (result, (inputs, matrices)) = TextIndex::with_segment_search_trace(|| {
            TextIndex::search_from_segment(
                &source,
                storage,
                "col_1",
                "alpha empty missing",
                TextSearchOptions::with_explanations(9),
                &context,
            )
        });
        assert!(result.unwrap().is_none());
        assert_eq!(inputs, [64 * 1024]);
        assert!(matrices.is_empty());
        assert_eq!(context.available_workspace_bytes(), limit);
        storage.file.seek(SeekFrom::Start(tail)).unwrap();
        storage.file.write_all(&[bytes[tail as usize]]).unwrap();
        assert_eq!(
            std::fs::read(fixture.root.join("fixture.coffer")).unwrap(),
            bytes
        );
    }

    #[test]
    fn shared_text_staging_rolls_back_source_miss_and_budget_failure_without_losing_prefix() {
        use crate::storage::checkpoint_writer::write_text_segment;
        use crate::temporary::TemporaryDiskBudget;
        use std::sync::Arc;

        let (original, checkpoint, _) = fixture(2);
        let mut prefix = Vec::new();
        write_text_segment(&checkpoint.text_segments[0], &mut prefix).unwrap();
        for mode in ["absent", "checksum", "truncated", "semantic", "memory"] {
            let root = std::env::temp_dir().join(format!(
                "cofferdb-source-staging-{}-{mode}",
                std::process::id()
            ));
            std::fs::create_dir(&root).unwrap();
            let mut storage = Storage::create(root.join("fixture.coffer")).unwrap();
            storage.file.set_len(0).unwrap();
            storage.file.seek(SeekFrom::Start(0)).unwrap();
            storage.file.write_all(&original).unwrap();
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let disk = Arc::new(TemporaryDiskBudget::new(4 * 1024 * 1024));
            let mut staging = TextStaging::new(&root, disk.clone(), &context).unwrap();
            checkpoint
                .vectors
                .stage_text_into(&storage, "col_1", &mut staging, &context)
                .unwrap()
                .unwrap();
            let path = std::fs::read_dir(&root)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .find(|path| {
                    path.file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .starts_with("cofferdb-text-checkpoint-")
                })
                .unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), prefix);
            let mut source = checkpoint.vectors.clone();
            let extent = *source.extent(INDEX_TEXT_SEGMENT, 2).unwrap();
            let offset = (source.offset + extent.suffix_relative_offset) as usize;
            let end = offset + extent.length as usize;
            let mut changed = original.clone();
            match mode {
                "checksum" => changed[end - 1] ^= 1,
                "semantic" => {
                    // Invalid TF with valid segment, envelope and snapshot checksums.
                    let start = offset + INDEX_RECORD_HEADER_LEN;
                    let frequency = start + 38 + 2 * 24 + 8 + 4 + 6 + 16 + 16 + 8;
                    changed[frequency..frequency + 8].fill(0);
                    let crc = checksum32(&changed[start..end - 4]);
                    changed[end - 4..end].copy_from_slice(&crc.to_le_bytes());
                    let crc = checksum32(&changed[start..end]);
                    changed[offset + 42..offset + 46].copy_from_slice(&crc.to_le_bytes());
                    let crc = checksum32(&changed[offset..offset + 46]);
                    changed[offset + 46..offset + 50].copy_from_slice(&crc.to_le_bytes());
                    source
                        .extent_mut(INDEX_TEXT_SEGMENT, 2)
                        .extent_record_checksum = checksum32(&changed[offset..end]);
                }
                _ => {}
            }
            storage.file.seek(SeekFrom::Start(0)).unwrap();
            storage.file.write_all(&changed).unwrap();
            if mode == "truncated" {
                storage.file.set_len(end as u64 - 1).unwrap();
            }
            let mut blocker = context.workspace_reservation();
            if mode == "memory" {
                let free = context.available_workspace_bytes();
                blocker.grow_to(free, free, "operation_scratch").unwrap();
            }
            let result = source.stage_text_into(
                &storage,
                if mode == "absent" { "col_3" } else { "col_2" },
                &mut staging,
                &context,
            );
            if mode == "memory" {
                assert!(crate::json::is_memory_budget_error(&result.err().unwrap()));
            } else {
                assert!(result.unwrap().is_none(), "{mode}");
            }
            drop(blocker);
            assert_eq!(disk.used(), prefix.len() as u64, "{mode}");
            assert_eq!(std::fs::read(&path).unwrap(), prefix, "{mode}");
            storage.file.seek(SeekFrom::Start(0)).unwrap();
            storage.file.write_all(&original).unwrap();
            storage.file.set_len(original.len() as u64).unwrap();
            checkpoint
                .vectors
                .stage_text_into(&storage, "col_1", &mut staging, &context)
                .unwrap()
                .unwrap();
            assert_eq!(disk.used(), 2 * prefix.len() as u64);
            assert_eq!(std::fs::read(&path).unwrap(), prefix.repeat(2));
            drop(staging);
            assert_eq!(disk.used(), 0);
            assert_eq!(context.available_workspace_bytes(), available);
            drop(storage);
            std::fs::remove_dir_all(&root).unwrap();
        }
    }

    #[test]
    fn text_visitors_preserve_first_errors_without_hiding_tail_integrity_or_io() {
        struct Input<'a> {
            bytes: Cursor<&'a [u8]>,
            fail_at: Option<u64>,
        }
        impl Read for Input<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                let count = if let Some(at) = self.fail_at {
                    if self.bytes.position() >= at {
                        return Err(std::io::Error::other("text tail read failure"));
                    }
                    output.len().min((at - self.bytes.position()) as usize)
                } else {
                    output.len()
                };
                self.bytes.read(&mut output[..count])
            }
        }
        impl Seek for Input<'_> {
            fn seek(&mut self, at: SeekFrom) -> std::io::Result<u64> {
                self.bytes.seek(at)
            }
        }
        let (original, checkpoint, _) = fixture(2);
        for header in [false, true] {
            for fault in 0..if header { 4 } else { 5 } {
                let mut source = checkpoint.vectors.clone();
                let extent = *source.extent(INDEX_TEXT_SEGMENT, 1).unwrap();
                let offset = (source.offset + extent.suffix_relative_offset) as usize;
                let end = offset + extent.length as usize;
                let mut bytes = original.clone();
                if fault == 1 {
                    bytes[end - 1] ^= 1;
                }
                if fault == 4 {
                    let start = offset + INDEX_RECORD_HEADER_LEN;
                    let frequency = start + 38 + 2 * 24 + 8 + 4 + "needle".len() + 16 + 16 + 8;
                    bytes[frequency..frequency + 8].fill(0);
                    let crc = checksum32(&bytes[start..end - 4]);
                    bytes[end - 4..end].copy_from_slice(&crc.to_le_bytes());
                    let crc = checksum32(&bytes[start..end]);
                    bytes[offset + 42..offset + 46].copy_from_slice(&crc.to_le_bytes());
                    let crc = checksum32(&bytes[offset..offset + 46]);
                    bytes[offset + 46..offset + 50].copy_from_slice(&crc.to_le_bytes());
                    source
                        .extent_mut(INDEX_TEXT_SEGMENT, 1)
                        .extent_record_checksum = checksum32(&bytes[offset..end]);
                }
                let limit = if header { 0 } else { 4096 };
                let context = JsonOperationContext::new(std::sync::Arc::new(
                    crate::json::WorkLedger::with_limits(limit, 0, limit),
                ));
                let trace = crate::search_cost::SearchCostTrace::new();
                let context = context
                    .with_search_cost(&trace)
                    .search_cost_stage(crate::search_cost::Stage::ScoreCompletion);
                let input = context.workspace_reservation();
                let mut reader = Input {
                    bytes: Cursor::new(if fault == 2 {
                        &bytes[..end - 1]
                    } else {
                        &bytes
                    }),
                    fail_at: (fault == 3).then_some(end as u64 - 1),
                };
                let mut calls = 0;
                let result = source.visit_text_from(&mut reader, "col_1", &input, |entry| {
                    calls += 1;
                    if (header && matches!(entry, TextSegmentEntry::Header { .. }))
                        || (!header && matches!(entry, TextSegmentEntry::Posting { .. }))
                    {
                        Err(
                            Error::new(ErrorCode::InvalidSearchParams, "first text consumer error")
                                .with_detail("sentinel", "text"),
                        )
                    } else {
                        Ok(())
                    }
                });
                assert_eq!(calls, if header { 1 } else { 5 });
                match fault {
                    0 => {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::InvalidSearchParams);
                        assert_eq!(error.detail(), &[("sentinel".into(), "text".into())]);
                    }
                    1 | 2 | 4 => assert!(!result.unwrap()),
                    _ => assert_eq!(
                        result.unwrap_err().to_string(),
                        "io_error: text tail read failure"
                    ),
                }
                assert_eq!(
                    reader.bytes.position(),
                    if fault == 2 || fault == 3 {
                        end as u64 - 1
                    } else {
                        end as u64
                    }
                );
                let expected = extent.length - u64::from(fault == 2 || fault == 3);
                assert_eq!(
                    trace.total(crate::search_cost::Counter::IndexReadBytes),
                    expected
                );
                assert_eq!(
                    trace.value(
                        crate::search_cost::Stage::ScoreCompletion,
                        crate::search_cost::Counter::IndexReadBytes
                    ),
                    expected
                );
                assert_eq!(
                    trace.total(crate::search_cost::Counter::IndexRequestedBytes),
                    extent.length
                );
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn text_stats_union_includes_empty_dictionary_keys_and_spills_outside_read_only_parent() {
        use crate::catalog::Catalog;
        use crate::temporary::TemporaryDiskBudget;
        use std::os::unix::fs::PermissionsExt;
        use std::sync::Arc;
        struct Directory {
            root: PathBuf,
            storage: Option<Storage>,
        }
        impl Drop for Directory {
            fn drop(&mut self) {
                let _ =
                    std::fs::set_permissions(&self.root, std::fs::Permissions::from_mode(0o700));
                drop(self.storage.take());
                let _ = std::fs::remove_dir_all(&self.root);
            }
        }
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        for wide in [false, true] {
            let (_, base, vectors) = fixture(1);
            let mut text_segments = base.text_segments;
            for (position, segment) in text_segments.iter_mut().enumerate() {
                let mut names = if position == 0 {
                    vec!["alpha".to_owned(), "empty".to_owned()]
                } else {
                    vec!["alpha".to_owned(), "beta".to_owned()]
                };
                if wide && position == 1 {
                    names.extend((0..4000).map(|term| format!("wide{term:06}")));
                }
                segment.terms = names
                    .into_iter()
                    .map(|term| TextIndexTermEntry {
                        postings: if term == "empty" {
                            Vec::new()
                        } else {
                            vec![TextIndexPostingEntry {
                                chunk_id: segment.chunks[0].chunk_id.clone(),
                                term_frequency: 1,
                            }]
                        },
                        term,
                    })
                    .collect();
            }
            let checkpoint = IndexCheckpointPayload {
                metadata: IndexCheckpointMetadata {
                    covered_last_batch_id: 1,
                    covered_last_commit_sequence_no: 2,
                    coverage_hash: 17,
                    collection_count: 2,
                    live_document_count: 2,
                    live_chunk_count: 2,
                    live_embedding_count: 2,
                    text_segment_count: 2,
                    vector_segment_count: 2,
                    build_mode: "inline_full_checkpoint".into(),
                },
                text_segments,
                vector_segments: vectors.clone(),
            };
            let mut bytes = vec![7; 37];
            bytes.extend(
                build_index_checkpoint_suffix_with_generation(&checkpoint, 37, 71).unwrap(),
            );
            let context = JsonOperationContext::standalone();
            let parsed = read_index_checkpoint_suffix_in(
                &mut Cursor::new(&bytes),
                37,
                bytes.len() as u64,
                37,
                1,
                2,
                (
                    IndexReadOptions {
                        budget_bytes: 0,
                        mode: IndexReadMode::LazyIndexes,
                    },
                    &context,
                ),
            )
            .unwrap()
            .unwrap();
            let source = parsed.lazy_vectors.unwrap();
            assert!(source.text_segments.is_empty());
            let source = source.vectors;
            let root = std::env::temp_dir().join(format!(
                "cofferdb-text-stats-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            let mut storage = Storage::create(root.join("fixture.coffer")).unwrap();
            storage.file.set_len(0).unwrap();
            storage.file.seek(SeekFrom::Start(0)).unwrap();
            storage.file.write_all(&bytes).unwrap();
            let directory = Directory {
                root,
                storage: Some(storage),
            };
            let original_names = std::fs::read_dir(&directory.root)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<BTreeSet<_>>();
            std::fs::set_permissions(&directory.root, std::fs::Permissions::from_mode(0o500))
                .unwrap();
            let mut catalog = Catalog::default();
            for vector in &vectors {
                let collection = collection_for(vector);
                catalog
                    .collections
                    .insert(collection.collection_id.clone(), collection);
            }
            let before = context.available_workspace_bytes();
            let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
            let stats = TextStatsSummary::from_source(
                &source,
                directory.storage.as_ref().unwrap(),
                &catalog,
                &disk,
                &context,
            )
            .unwrap()
            .unwrap();
            assert_eq!(stats.term_count(), if wide { 4003 } else { 3 });
            assert_eq!(stats.term_count_in_collection("col_1"), 1);
            assert_eq!(
                stats.term_count_in_collection("col_2"),
                if wide { 4002 } else { 2 }
            );
            assert_eq!(disk.used(), 0);
            assert_eq!(disk.peak_used() > 0, wide);
            drop(stats);
            assert_eq!(context.available_workspace_bytes(), before);
            if wide {
                let disk = Arc::new(TemporaryDiskBudget::new(0));
                let error = TextStatsSummary::from_source(
                    &source,
                    directory.storage.as_ref().unwrap(),
                    &catalog,
                    &disk,
                    &context,
                )
                .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(error
                    .detail()
                    .contains(&("resource".into(), "temporary_disk".into())));
                assert!(error.detail().contains(&("limit_bytes".into(), "0".into())));
                assert_eq!(disk.used(), 0);
                assert_eq!(context.available_workspace_bytes(), before);
            }
            assert_eq!(
                std::fs::read(directory.root.join("fixture.coffer")).unwrap(),
                bytes
            );
            assert_eq!(
                std::fs::read_dir(&directory.root)
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name())
                    .collect::<BTreeSet<_>>(),
                original_names
            );
        }
    }

    #[test]
    fn streaming_vectors_reuse_one_buffer_and_require_final_frame_validation() {
        for seed in 0..100 {
            let context = crate::json::JsonOperationContext::standalone();
            let input = context.workspace_reservation();
            let mut error = None;
            let (bytes, checkpoint, expected) = fixture(seed);
            let source = checkpoint.vectors;
            let collection = collection_for(&expected[0]);
            let mut pointers = BTreeSet::new();
            let mut seen = 0;
            let loaded = source
                .read_with_mode(
                    &mut Cursor::new(&bytes),
                    "col_1",
                    VectorDecodeMode::Visit {
                        collection: &collection,
                        input: &input,
                        error: &mut error,
                        visit: &mut |chunk, embedding, values| {
                            let entry = &expected[0].vectors[seen];
                            assert_eq!(chunk, catalog_id_number(&entry.chunk_id).unwrap());
                            assert_eq!(embedding, catalog_id_number(&entry.embedding_id).unwrap());
                            assert_eq!(
                                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                entry.vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                            pointers.insert(values.as_ptr() as usize);
                            seen += 1;
                            Ok(())
                        },
                    },
                )
                .unwrap()
                .unwrap();
            assert!(error.is_none());
            assert_eq!(seen, expected[0].vectors.len());
            assert_eq!(pointers.len(), usize::from(seen > 0));
            assert!(loaded.payload.is_none());
            let mut mismatched = collection.clone();
            mismatched.embedding_dim += 1;
            let mut called = false;
            assert!(source
                .read_with_mode(
                    &mut Cursor::new(&bytes),
                    "col_1",
                    VectorDecodeMode::Visit {
                        collection: &mismatched,
                        input: &input,
                        error: &mut error,
                        visit: &mut |_, _, _| {
                            called = true;
                            Ok(())
                        },
                    }
                )
                .unwrap()
                .is_none());
            assert!(!called);
            let extent = *source.extent(INDEX_VECTOR_SEGMENT, 1).unwrap();
            let end = (source.offset + extent.suffix_relative_offset + extent.length) as usize;
            let mut damaged = bytes.clone();
            damaged[end - 1] ^= 1;
            let mut provisional = 0;
            assert!(source
                .read_with_mode(
                    &mut Cursor::new(&damaged),
                    "col_1",
                    VectorDecodeMode::Visit {
                        collection: &collection,
                        input: &input,
                        error: &mut error,
                        visit: &mut |_, _, _| {
                            provisional += 1;
                            Ok(())
                        },
                    }
                )
                .unwrap()
                .is_none());
            assert_eq!(provisional, seen);
            assert!(error.is_none());
        }
    }

    #[test]
    fn streamed_vector_workspace_is_shared_exact_and_released_on_every_outcome() {
        use crate::json::{JsonOperationContext, WorkLedger};
        use std::sync::Arc;

        for seed in 0..100 {
            let (bytes, checkpoint, expected) = fixture(seed);
            let source = checkpoint.vectors;
            let collection = collection_for(&expected[0]);
            let count = expected[0].vectors.len();
            let needed = if count == 0 {
                0
            } else {
                collection.embedding_dim * 4
            };
            let occupied = 37 + seed;
            for short in 0..=usize::from(needed > 0) {
                let limit = occupied + needed - short;
                let ledger = Arc::new(WorkLedger::with_limits(
                    if seed % 2 == 0 { limit } else { limit * 2 },
                    limit,
                    if seed % 2 == 0 { limit * 2 } else { limit },
                ));
                let context = JsonOperationContext::new(ledger.clone());
                let input = context.workspace_reservation();
                let mut held = JsonOperationContext::new(ledger).workspace_reservation();
                held.grow_to(occupied, occupied, "operation_scratch")
                    .unwrap();
                let mut cursor = Cursor::new(&bytes);
                let mut seen = 0;
                let mut pointers = BTreeSet::new();
                let result = source.visit_from(&mut cursor, &collection, &input, |id, values| {
                    let expected = &expected[0].vectors[seen];
                    assert_eq!(
                        id,
                        catalog_id_string("chk_", catalog_id_number(&expected.chunk_id).unwrap())
                    );
                    assert_eq!(
                        values
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .vector
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    pointers.insert(values.as_ptr());
                    seen += 1;
                    let failure = context
                        .workspace_reservation()
                        .grow_to(1, 1, "operation_scratch")
                        .unwrap_err();
                    assert_eq!(failure.code(), ErrorCode::ResourceExhausted);
                    assert!(failure
                        .detail()
                        .contains(&("requested_bytes".into(), (limit + 1).to_string())));
                    Ok(())
                });
                if short == 0 {
                    assert!(result.unwrap());
                    assert_eq!(seen, count);
                    assert_eq!(pointers.len(), usize::from(count > 0));
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", (occupied + needed).to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)));
                    }
                    assert_eq!(seen, 0);
                }
                let extent = *source.extent(INDEX_VECTOR_SEGMENT, 1).unwrap();
                assert_eq!(
                    cursor.position(),
                    source.offset + extent.suffix_relative_offset + extent.length
                );
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(occupied);

                for field in 0..8 {
                    let mut mismatched = collection.clone();
                    match field {
                        0 => mismatched.embedding_dim += 1,
                        1 => mismatched.embedding_model = None,
                        2 => mismatched.distance_metric = DistanceMetric::Cosine,
                        3 => mismatched.vectors_normalized = true,
                        4 => mismatched.collection_id = "col_1".into(),
                        5 => mismatched.collection_id = "bad_0000000000000001".into(),
                        6 => mismatched.collection_id = "col_+000000000000001".into(),
                        _ => mismatched.collection_id = "col_00000000000000001".into(),
                    }
                    assert!(!source
                        .visit_from(&mut Cursor::new(&bytes), &mismatched, &input, |_, _| {
                            Err(Error::resource_exhausted(
                                "memory",
                                "operation_scratch",
                                1,
                                2,
                            ))
                        })
                        .unwrap());
                    held.grow_to(limit, limit, "operation_scratch").unwrap();
                    held.retain(occupied);
                }
                held.retain(0);
                assert!(source
                    .visit_from(&mut Cursor::new(&bytes), &collection, &input, |_, _| Ok(()))
                    .unwrap());
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
        }
    }

    #[test]
    fn streamed_vector_buffers_share_quota_and_models_need_no_owned_decode() {
        use std::sync::Arc;
        for seed in 0..100 {
            let model = format!(
                "{}{}",
                "x".repeat(4092 + seed % 9),
                "\u{a2}\u{4e2d}\u{1f642}".repeat(1000)
            );
            let (bytes, checkpoint, expected) = fixture_with_model(seed, Some(model));
            let source = checkpoint.vectors;
            let collection = collection_for(&expected[0]);
            let array = if expected[0].vectors.is_empty() {
                0
            } else {
                collection.embedding_dim * 4
            };
            let preferred = source
                .encoded_len(&collection.collection_id)
                .unwrap()
                .min(64 * 1024) as usize;
            let available = [
                0,
                1,
                array,
                array + preferred - 1,
                array + preferred,
                array + preferred + 64,
            ][seed % 6];
            let occupied = 37 + seed;
            let limit = occupied + available;
            let ledger = Arc::new(crate::json::WorkLedger::with_limits(limit, limit, limit));
            let context = JsonOperationContext::new(ledger.clone());
            let mut held = JsonOperationContext::new(ledger).workspace_reservation();
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let mut input = source
                .reserve_read_buffer(&collection.collection_id, &context)
                .unwrap();
            assert_eq!(input.reserved_bytes(), available.min(preferred));
            let mut cursor = Cursor::new(&bytes);
            let mut seen = 0;
            let result = source.visit_from(&mut cursor, &collection, &input, |_, _| {
                seen += 1;
                let remaining = available - input.reserved_bytes() - array;
                let error = context
                    .workspace_reservation()
                    .grow_to(remaining + 1, remaining + 1, "operation_scratch")
                    .unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(error
                    .detail()
                    .contains(&("requested_bytes".into(), (limit + 1).to_string())));
                Ok(())
            });
            if array <= available - input.reserved_bytes() {
                assert!(result.unwrap());
                assert_eq!(seen, expected[0].vectors.len());
            } else {
                assert_eq!(result.unwrap_err().code(), ErrorCode::ResourceExhausted);
                assert_eq!(seen, 0);
            }
            let extent = *source.extent(INDEX_VECTOR_SEGMENT, 1).unwrap();
            assert_eq!(
                cursor.position(),
                source.offset + extent.suffix_relative_offset + extent.length
            );
            input.retain(0);
            let result =
                source.visit_from(&mut Cursor::new(&bytes), &collection, &input, |_, _| Ok(()));
            if array <= available {
                assert!(result.unwrap());
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(error
                    .detail()
                    .contains(&("requested_bytes".into(), (occupied + array).to_string())));
            }
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
    }

    #[test]
    fn streamed_vector_ids_preserve_minimum_width_and_full_u64_range() {
        for number in [
            0,
            1,
            9,
            10,
            9_999_999_999_999_999,
            10_000_000_000_000_000,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let mut id = [0; 24];
            assert_eq!(
                catalog_id_text(&mut id, b"col_", number),
                catalog_id_string("col_", number)
            );
            let (mut bytes, checkpoint, mut expected) = fixture(1);
            let mut source = checkpoint.vectors;
            expected[0].vectors[0].chunk_id = format!("chk_{number}");
            let payload = encode_vector_index_segment_payload(&expected[0]).unwrap();
            let generation = source.generation;
            let offset = source.offset;
            let extent = source.extent_mut(INDEX_VECTOR_SEGMENT, 1);
            let record = build_index_record(
                INDEX_VECTOR_SEGMENT,
                generation,
                extent.index_record_id,
                &payload,
            );
            assert_eq!(record.len() as u64, extent.length);
            extent.extent_record_checksum = checksum32(&record);
            let start = (offset + extent.suffix_relative_offset) as usize;
            bytes[start..start + record.len()].copy_from_slice(&record);
            let context = crate::json::JsonOperationContext::standalone();
            let mut seen = 0;
            assert!(source
                .visit_from(
                    &mut Cursor::new(&bytes),
                    &collection_for(&expected[0]),
                    &context.workspace_reservation(),
                    |id, _| {
                        assert_eq!(id, format!("chk_{number:016}"));
                        seen += 1;
                        Ok(())
                    }
                )
                .unwrap());
            assert_eq!(seen, 1);
        }
    }

    #[test]
    fn lazy_open_admits_only_retained_text_and_directory_bytes() {
        for seed in 0..100 {
            let (bytes, checkpoint, _) = fixture(seed);
            let source = checkpoint.vectors;
            let directory_len =
                (source.extents.as_slice().len() + 1) as u64 * INDEX_EXTENT_ENTRY_LEN as u64;
            let vector_bytes = source
                .extents
                .as_slice()
                .iter()
                .filter(|extent| extent.record_type == INDEX_VECTOR_SEGMENT)
                .map(|extent| extent.length)
                .sum::<u64>();
            let retained = source.payload_len - vector_bytes + directory_len;
            let load = |mode, budget_bytes| {
                read_index_checkpoint_suffix(
                    &mut Cursor::new(&bytes),
                    37,
                    bytes.len() as u64,
                    37,
                    1,
                    2,
                    IndexReadOptions { mode, budget_bytes },
                )
            };
            let parsed = load(IndexReadMode::LazyVectors, retained).unwrap().unwrap();
            assert!(parsed.lazy_vectors.is_some());
            assert!(parsed.diagnostics.is_empty());
            for budget in [retained - 1, directory_len - 1] {
                assert!(
                    matches!(load(IndexReadMode::LazyVectors, budget), Err(error) if error.code() == ErrorCode::ResourceExhausted)
                );
            }
            assert!(
                matches!(load(IndexReadMode::Full, retained), Err(error) if error.code() == ErrorCode::ResourceExhausted)
            );
            assert!(
                load(IndexReadMode::Full, source.payload_len + directory_len)
                    .unwrap()
                    .unwrap()
                    .checkpoint
                    .is_some()
            );
        }
    }

    #[test]
    fn streamed_frame_checksums_match_wire_checksums_at_every_tail_alignment() {
        for seed in 0..100usize {
            let body_len = match seed % 5 {
                0 => seed,
                1 => 4093 - seed,
                2 => 4096 + seed,
                3 => 65536 - seed,
                _ => 65536 + seed,
            };
            let mut payload = (0..body_len)
                .map(|index| (index ^ seed) as u8)
                .collect::<Vec<_>>();
            let segment_checksum = checksum32(&payload);
            payload.extend_from_slice(&segment_checksum.to_le_bytes());
            let record = build_index_record(INDEX_VECTOR_SEGMENT, seed as u64, 2, &payload);
            let header = record[..INDEX_RECORD_HEADER_LEN].try_into().unwrap();
            payload.extend_from_slice(&[7; 17]);
            for fragment in [1, 3, 17, 4093, 65536] {
                let mut checked = SegmentChecksumReader::new(
                    Cursor::new(&payload),
                    &header,
                    (body_len + 4) as u64,
                );
                let mut output = vec![0; fragment];
                let mut first = true;
                while checked.read(&mut output).unwrap() != 0 {
                    if first {
                        assert_eq!(
                            checked.record_checksum(),
                            checksum32(
                                &record[..INDEX_RECORD_HEADER_LEN + checked.bytes_read as usize]
                            )
                        );
                        first = false;
                    }
                }
                assert_eq!(checked.bytes_read, (body_len + 4) as u64);
                assert_eq!(checked.source.get_ref().position(), checked.bytes_read);
                assert_eq!(checked.record_checksum(), checksum32(&record));
                assert_eq!(
                    checked.payload_checksum.finish(),
                    checksum32(&payload[..body_len + 4])
                );
                assert_eq!(checked.segment_checksum, segment_checksum);
                assert_eq!(read_u32(&checked.tail), segment_checksum);
            }
        }
    }

    #[test]
    fn lazy_vector_rejects_every_changed_byte_and_truncated_prefix() {
        let (bytes, checkpoint, _) = fixture(1);
        let source = checkpoint.vectors;
        let extent = *source.extent(INDEX_VECTOR_SEGMENT, 1).unwrap();
        let offset = (source.offset + extent.suffix_relative_offset) as usize;
        let end = offset + extent.length as usize;
        for changed in offset..end {
            let mut damaged = bytes.clone();
            damaged[changed] ^= 1;
            assert!(
                source
                    .read_from(&mut Cursor::new(&damaged), "col_1")
                    .unwrap()
                    .is_none(),
                "changed byte {changed}"
            );
            assert!(
                source
                    .read_from(&mut Cursor::new(&bytes[..changed]), "col_1")
                    .unwrap()
                    .is_none(),
                "truncated prefix {changed}"
            );
        }
    }

    #[test]
    fn lazy_vector_checks_inner_contracts_even_with_matching_extent_checksums() {
        for seed in 0..100 {
            let (bytes, checkpoint, expected) = fixture(seed);
            let source = checkpoint.vectors;
            let extent = *source.extent(INDEX_VECTOR_SEGMENT, 1).unwrap();
            let offset = (source.offset + extent.suffix_relative_offset) as usize;
            for case in [
                "collection",
                "commit",
                "segment_crc",
                "payload_crc",
                "nonfinite",
                "short",
            ] {
                if case == "nonfinite" && expected[0].vectors.is_empty() {
                    continue;
                }
                let mut payload = encode_vector_index_segment_payload(&expected[0]).unwrap();
                let body_len = payload.len() - 4;
                match case {
                    "collection" => payload[2..10].copy_from_slice(&99u64.to_le_bytes()),
                    "commit" => payload[10..18].copy_from_slice(&99u64.to_le_bytes()),
                    "nonfinite" => payload[body_len - 4..body_len]
                        .copy_from_slice(&f32::INFINITY.to_le_bytes()),
                    _ => {}
                }
                let crc = checksum32(&payload[..body_len]);
                payload[body_len..].copy_from_slice(&crc.to_le_bytes());
                if case == "segment_crc" {
                    payload[body_len] ^= 1;
                } else if case == "short" {
                    payload.truncate(seed % 4);
                }
                let mut record = build_index_record(
                    INDEX_VECTOR_SEGMENT,
                    source.generation,
                    extent.index_record_id,
                    &payload,
                );
                if case == "payload_crc" {
                    record[42] ^= 1;
                    let crc = checksum32(&record[..46]);
                    record[46..50].copy_from_slice(&crc.to_le_bytes());
                }
                let mut changed = source.clone();
                changed
                    .extent_mut(INDEX_VECTOR_SEGMENT, 1)
                    .extent_record_checksum = checksum32(&record);
                changed.extent_mut(INDEX_VECTOR_SEGMENT, 1).length = record.len() as u64;
                let mut damaged = bytes.clone();
                damaged[offset..offset + record.len()].copy_from_slice(&record);
                assert!(
                    changed
                        .read_from(&mut Cursor::new(&damaged), "col_1")
                        .unwrap()
                        .is_none(),
                    "seed {seed}, {case}"
                );
                let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                    crate::json::WorkLedger::with_limits(0, 0, 0),
                ));
                assert!(
                    !changed
                        .visit_from(
                            &mut Cursor::new(&damaged),
                            &collection_for(&expected[0]),
                            &context.workspace_reservation(),
                            |_, _| { panic!("an invalid frame cannot use an unreserved vector") }
                        )
                        .unwrap(),
                    "seed {seed}, {case}"
                );
            }
        }
    }

    #[test]
    fn lazy_vector_io_errors_propagate_and_decode_time_mutations_invalidate() {
        struct Stream {
            bytes: Cursor<Vec<u8>>,
            reads: usize,
            seeks: usize,
            fail_read: usize,
            fail_seek: usize,
            mutation: Option<u64>,
            bytes_read: u64,
        }
        impl Stream {
            fn new(bytes: &[u8]) -> Self {
                Self {
                    bytes: Cursor::new(bytes.to_vec()),
                    reads: 0,
                    seeks: 0,
                    fail_read: 0,
                    fail_seek: 0,
                    mutation: None,
                    bytes_read: 0,
                }
            }
        }
        impl Read for Stream {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                assert!(output.len() <= 64 * 1024);
                self.reads += 1;
                if self.reads == self.fail_read {
                    return Err(std::io::ErrorKind::Other.into());
                }
                if self.reads % 7 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let limit = output.len().min(4093);
                if let Some(position) = self.mutation {
                    if (self.bytes.position()..self.bytes.position() + limit as u64)
                        .contains(&position)
                    {
                        self.bytes.get_mut()[position as usize] ^= 1;
                        self.mutation = None;
                    }
                }
                let count = self.bytes.read(&mut output[..limit])?;
                self.bytes_read += count as u64;
                Ok(count)
            }
        }
        impl Seek for Stream {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                self.seeks += 1;
                if self.seeks == self.fail_seek {
                    return Err(std::io::ErrorKind::Other.into());
                }
                self.bytes.seek(position)
            }
        }
        let (bytes, checkpoint, expected) = fixture(11);
        let source = checkpoint.vectors;
        let collection = collection_for(&expected[0]);
        for mode in 0..5 {
            let load = |stream: &mut Stream| {
                if mode == 0 {
                    return source
                        .read_from(stream, "col_1")
                        .map(|value| value.is_some());
                }
                let context = if mode == 2 {
                    crate::json::JsonOperationContext::new(std::sync::Arc::new(
                        crate::json::WorkLedger::with_limits(0, 0, 0),
                    ))
                } else {
                    crate::json::JsonOperationContext::standalone()
                };
                let trace = crate::search_cost::SearchCostTrace::new();
                let context = context
                    .with_search_cost(&trace)
                    .search_cost_stage(crate::search_cost::Stage::Recall);
                if mode == 4 {
                    let held = CachedVectorSegment::reserve(
                        &context,
                        usize::MAX,
                        source.encoded_len("col_1").unwrap(),
                    )
                    .unwrap();
                    let reserved = held.reserved_bytes();
                    let result = source
                        .read_cached_from(stream, "col_1", held, &context)
                        .map(|value| value.is_some());
                    assert!(context
                        .cache_reservation(reserved as u128, reserved)
                        .is_some());
                    assert_eq!(
                        trace.total(crate::search_cost::Counter::IndexReadBytes),
                        stream.bytes_read
                    );
                    assert_eq!(
                        trace.total(crate::search_cost::Counter::IndexReadCalls),
                        stream.reads as u64
                    );
                    return result;
                }
                let mut calls = 0;
                let input = source
                    .reserve_read_buffer(&collection.collection_id, &context)
                    .unwrap();
                let result = source.visit_from(stream, &collection, &input, |_, _| {
                    calls += 1;
                    if mode == 3 {
                        Err(Error::resource_exhausted(
                            "memory",
                            "operation_scratch",
                            1,
                            2,
                        ))
                    } else {
                        Ok(())
                    }
                });
                drop(input);
                assert_eq!(
                    trace.total(crate::search_cost::Counter::IndexReadBytes),
                    stream.bytes_read
                );
                assert_eq!(
                    trace.total(crate::search_cost::Counter::IndexReadCalls),
                    stream.reads as u64
                );
                if mode == 2 {
                    assert_eq!(calls, 0);
                }
                if mode == 3 {
                    assert!(calls <= 1);
                }
                let limit = if mode == 2 {
                    0
                } else {
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                };
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
                result
            };
            let mut success = Stream::new(&bytes);
            if mode < 2 || mode == 4 {
                assert!(load(&mut success).unwrap());
            } else {
                assert_eq!(
                    load(&mut success).unwrap_err().code(),
                    ErrorCode::ResourceExhausted
                );
            }
            assert_eq!(
                success.bytes_read,
                source.extent(INDEX_VECTOR_SEGMENT, 1).unwrap().length
            );
            assert_eq!(success.seeks, 1);
            for read in 1..=success.reads {
                let mut stream = Stream::new(&bytes);
                stream.fail_read = read;
                assert_eq!(load(&mut stream).unwrap_err().code(), ErrorCode::IoError);
            }
            for seek in 1..=success.seeks {
                let mut stream = Stream::new(&bytes);
                stream.fail_seek = seek;
                assert_eq!(load(&mut stream).unwrap_err().code(), ErrorCode::IoError);
            }
            let extent = *source.extent(INDEX_VECTOR_SEGMENT, 1).unwrap();
            let mut stream = Stream::new(&bytes);
            stream.mutation = Some(
                source.offset + extent.suffix_relative_offset + INDEX_RECORD_HEADER_LEN as u64,
            );
            assert!(!load(&mut stream).unwrap());
            assert!(stream.mutation.is_none());
            let mutated_reads = stream.reads;
            let mut stream = Stream::new(&bytes);
            stream.mutation = Some(
                source.offset + extent.suffix_relative_offset + INDEX_RECORD_HEADER_LEN as u64,
            );
            stream.fail_read = mutated_reads;
            assert_eq!(load(&mut stream).unwrap_err().code(), ErrorCode::IoError);
            assert!(stream.mutation.is_none());
        }
    }
}
