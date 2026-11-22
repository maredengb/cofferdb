use super::*;
use crate::catalog::{Catalog, VisibleChunk};
use crate::index::text::TextStatsSummary;
use crate::json::{is_memory_budget_error, JsonOperationContext, ReservedVec};
use crate::temporary::{
    create_budgeted_temporary_file_in_context, ExternalTermFrequencies, TempFileGuard,
    TemporaryDiskBudget, TemporaryFile,
};
use std::sync::Arc;

#[cfg(test)]
mod tests;

// One file owner remains live across all collections and suffix-write attempts.
pub(crate) struct TextStaging {
    // Shared temporary file containing accepted bodies and, during an append, a
    // tentative tail.
    file: TemporaryFile,
    // Owns cleanup of the staging file and its temporary disk reservation.
    cleanup: TempFileGuard,
    // Same disk budget used by the external term sorter, whose runs can coexist
    // with this file.
    disk: Arc<TemporaryDiskBudget>,
    // End of the last accepted extent. Rollback truncates here after an
    // unsuccessful append.
    bytes: u64,
    // Absolute byte end covered by the guard's disk reservation. It can exceed
    // bytes while a write is in progress.
    reserved: u64,
    // Cleared after a failed rollback or compaction. The owner then refuses further
    // reads or appends from uncertain contents.
    healthy: bool,
}

pub(crate) struct TextExtent {
    // Numeric collection ID used as the directory key.
    collection: u64,
    // Covered commit in the staged body's original header, used when validating a
    // visit.
    commit: u64,
    // Byte offset in the staging file. In-place reclamation updates it if an
    // earlier body was discarded.
    offset: u64,
    // Length of the complete staged segment, including its stored four-byte CRC.
    length: u64,
    // Expected CRC of the segment body before its footer, also recorded in that
    // footer.
    checksum: u32,
}

pub(crate) struct StagedText<'a> {
    // Owner keeping both the temporary file and its disk reservation alive for this
    // view.
    staging: &'a TextStaging,
    // Location and independent checksum used to validate this particular
    // collection.
    pub(crate) extent: &'a TextExtent,
}

#[cfg(test)]
pub(crate) struct TextSpool {
    pub(in crate::storage) staging: TextStaging,
    pub(in crate::storage) extent: TextExtent,
}

pub(crate) struct TextCheckpoint {
    // One shared file owner for all retained collection bodies. It can be absent
    // when no staging file was needed.
    staging: Option<TextStaging>,
    // Budgeted directory of accepted extents. Collection order enables binary
    // search once prefetch or build is complete.
    bodies: ReservedVec<TextExtent>,
    // Statistics derived from the staged bodies. The prepared checkpoint transfers
    // this value to the published segment source.
    stats: Option<TextStatsSummary>,
}

impl TextStaging {
    pub(crate) fn new(
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        // The file guard and the term sorter's runs draw disk credit from the same
        // budget. Keep the guard with the file even while several collection
        // extents share its contents.
        let (file, cleanup) = create_budgeted_temporary_file_in_context(
            directory,
            "text-checkpoint",
            "tmp",
            disk.clone(),
            context,
            "operation_scratch",
        )?;
        Ok(Self {
            file,
            cleanup,
            disk,
            bytes: 0,
            reserved: 0,
            healthy: true,
        })
    }

    // Start an append only from an accepted boundary. The file length, reservation
    // and last accepted offset have to agree before more disk credit can be
    // acquired.
    fn begin(&mut self) -> Result<u64> {
        if !self.healthy || self.reserved != self.bytes || self.file.metadata()?.len() != self.bytes
        {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "text staging length changed",
            ));
        }
        // Restore the output cursor to the accepted end after checking the actual
        // file length. A prior read of an extent through a cloned descriptor may
        // have left the cursor somewhere inside that body.
        self.file.seek(SeekFrom::Start(self.bytes))?;
        Ok(self.bytes)
    }

    pub(crate) fn rollback(&mut self) -> Result<()> {
        // bytes ends the last accepted extent. A failed append can leave bytes and
        // reserved credit past that point. Release the excess only after restoring the file.
        let result = (|| {
            // Truncate first, then reposition the descriptor used for later
            // appends. If either operation fails, keep the outstanding disk credit
            // until the owner can release the file.
            self.file.set_len(self.bytes)?;
            self.file.seek(SeekFrom::Start(self.bytes))?;
            self.cleanup.release_disk(self.reserved - self.bytes);
            self.reserved = self.bytes;
            Ok(())
        })();
        // A failed truncate or seek leaves the placement of future writes
        // uncertain. Mark the whole staging owner unhealthy so later readers
        // cannot trust offsets that still look in bounds.
        if result.is_err() {
            self.healthy = false;
        }
        result
    }

    // Copy the decoder's input to the shared file while validating the text body.
    // The caller still has to verify the source record's outer checksums. Leave
    // this copy tentative until finish_copy is called.
    pub(in crate::storage) fn copy_body_from(
        &mut self,
        reader: &mut dyn Read,
        length: usize,
        commit: u64,
        context: &JsonOperationContext,
        capacity: usize,
    ) -> std::result::Result<TextExtent, IndexReadError> {
        if !self.healthy || self.reserved != self.bytes || self.file.metadata()?.len() != self.bytes
        {
            return Err(IndexReadError::Resource(staging_checksum_error()));
        }
        let offset = self.bytes;
        self.file.seek(SeekFrom::Start(offset))?;
        // The supplied length excludes the source's CRC word. Reserve room for the
        // body and the new staging footer together. The reservation remains with
        // the owner if the copy needs rollback.
        let bytes = (length as u64)
            .checked_add(4)
            .ok_or_else(|| IndexReadError::Resource(invalid_staging()))?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| IndexReadError::Resource(invalid_staging()))?;
        self.cleanup
            .reserve_disk(bytes)
            .map_err(IndexReadError::Resource)?;
        // Remember the full reserved end before feeding any source bytes into the
        // decoder. A partial read, encode or flush failure can leave fewer bytes
        // on disk than this reservation covers.
        self.reserved = end;
        let mut output = BufWriter::with_capacity(capacity, &mut self.file);
        let copied: std::result::Result<_, IndexReadError> = (|| {
            // The decoder consumes the source through a copying reader. The bytes
            // accepted into the staging writer and its running CRC refer to the
            // same body, without retaining another complete segment.
            let mut copying = CopyReader {
                source: reader,
                output: &mut output,
                crc: Crc32c::default(),
                bytes: 0,
            };
            // Summary mode validates the original collection and term data while
            // copying the body. The caller supplies the old covered commit here;
            // retagging belongs to the later checkpoint write.
            let parsed = decode_text_index_segment_with(
                &mut IndexPayloadReader::new(&mut copying, length, 0),
                commit,
                0,
                TextDecodeMode::Summary,
                context,
            )?;
            // A successfully decoded prefix is insufficient when the caller
            // declared more body bytes. Treat a short copy as a truncated extent
            // before returning its collection number.
            if copying.bytes != length as u64 {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            Ok((
                parsed.collection_number,
                copying.bytes,
                copying.crc.finish(),
            ))
        })()
        // A completed parse can coexist with an output write still buffered in
        // memory. Flush that output before handing back the descriptor so an I/O
        // refusal is attributed to this tentative copy.
        .and_then(|copied| {
            output.flush()?;
            Ok(copied)
        });
        // Failed reads and writes must not trigger another write from Drop.
        drop(output.into_parts().1);
        // Construct a tentative descriptor after the buffered copy flushes. The
        // caller still verifies the containing index record before asking this
        // file to append its independent footer.
        let (collection, bytes, checksum) = copied?;
        Ok(TextExtent {
            collection,
            commit,
            offset,
            length: bytes + 4,
            checksum,
        })
    }

    // An accepted source now has an independent CRC in the staging file. Advance
    // the accepted end only after writing and flushing the footer.
    pub(crate) fn finish_copy(&mut self, extent: &TextExtent) -> Result<()> {
        // Only the descriptor beginning at the current accepted end may be
        // finished. Finishing an older descriptor twice, or finishing another copy
        // out of sequence, would give one physical tail two owners.
        if !self.healthy || extent.offset != self.bytes {
            return Err(invalid_staging());
        }
        self.file.write_all(&extent.checksum.to_le_bytes())?;
        self.file.flush()?;
        self.bytes = extent.end()?;
        Ok(())
    }

    // Close holes left by removed collections in place. Visit surviving extents in
    // physical order so the destination of a move is always at or below the source.
    // Read each block before writing it, including when the two ranges overlap.
    fn reclaim(&mut self, bodies: &mut [TextExtent]) -> Result<()> {
        let result = (|| {
            self.begin()?;
            bodies.sort_unstable_by_key(|body| body.offset);
            let mut end = 0;
            let mut buffer = [0; 8192];
            for body in bodies.iter_mut() {
                let source_end = body.end()?;
                // Process retained extents in forward file order. Their new end
                // must not overtake the next unread source, and every source must
                // contain at least the fixed header and footer within the
                // previously accepted prefix.
                if end > body.offset || source_end > self.bytes || body.length < 22 {
                    return Err(invalid_staging());
                }
                let mut crc = Crc32c::default();
                let body_bytes = body.length - 4;
                let mut copied = 0;
                while copied != body_bytes {
                    let count = (body_bytes - copied).min(buffer.len() as u64) as usize;
                    self.file.seek(SeekFrom::Start(body.offset + copied))?;
                    self.file.read_exact(&mut buffer[..count])?;
                    crc.update(&buffer[..count]);
                    // When no earlier gap exists, this body need not move. Still
                    // feed every byte to the CRC so keeping a body in place cannot
                    // bypass the integrity checks applied to moved ones.
                    if end != body.offset {
                        self.file.seek(SeekFrom::Start(end + copied))?;
                        self.file.write_all(&buffer[..count])?;
                    }
                    copied += count as u64;
                }
                // Verify both the recomputed body CRC and the stored footer before
                // recording the moved offset. A failure can occur after earlier
                // bytes have moved, so the file owner must then be treated as
                // unhealthy.
                // Read the checksum at the original extent's end. For a downward
                // move, the bytes already written have not reached that old
                // footer, even when source and destination ranges overlap.
                self.file.seek(SeekFrom::Start(source_end - 4))?;
                let mut tail = [0; 4];
                self.file.read_exact(&mut tail)?;
                if crc.finish() != body.checksum || u32::from_le_bytes(tail) != body.checksum {
                    return Err(staging_checksum_error());
                }
                if end != body.offset {
                    self.file.seek(SeekFrom::Start(end + body_bytes))?;
                    self.file.write_all(&tail)?;
                }
                // Publish the new offset only after the checked footer is in
                // place. Offsets for earlier bodies may already have changed if a
                // later body fails, so the failure path invalidates the staging
                // file as a whole.
                body.offset = end;
                end = end.checked_add(body.length).ok_or_else(invalid_staging)?;
            }
            // Truncate before releasing the disk credit. Restore collection order
            // afterwards because callers look up extents by collection rather than
            // by their new file positions.
            // Only bytes below the compacted end are still in use. A failed length
            // change must leave the disk charge intact; releasing space first
            // would let another operation spend capacity the file still occupies.
            self.file.set_len(end)?;
            self.file.seek(SeekFrom::Start(end))?;
            self.cleanup.release_disk(self.reserved - end);
            self.reserved = end;
            self.bytes = end;
            bodies.sort_unstable_by_key(|body| body.collection);
            Ok(())
        })();
        if result.is_err() {
            // Compaction is in-place. Once a body or directory offset has moved
            // there is no original layout to rewind to, even if the file still has
            // its old length.
            self.healthy = false;
        }
        result
    }
}

impl TextExtent {
    fn end(&self) -> Result<u64> {
        // Offset and length originate in separate steps. Check their sum before
        // using it as an address or as a new accepted boundary, including for
        // bodies retained across a rebuild.
        self.offset
            .checked_add(self.length)
            .ok_or_else(invalid_staging)
    }

    pub(crate) fn collection_number(&self) -> u64 {
        self.collection
    }
}

impl StagedText<'_> {
    #[cfg(all(test, unix))]
    pub(crate) fn staging_identity(&self) -> Result<(u64, u64)> {
        use std::os::unix::fs::MetadataExt;
        let metadata = self.staging.file.metadata()?;
        Ok((metadata.dev(), metadata.ino()))
    }

    // A view may be read only within the accepted file prefix. The minimum covers
    // the schema and collection prefix, covered commit, and trailing CRC accessed
    // when rewriting the commit.
    fn reader(&self) -> Result<File> {
        let mut file = self.staging.file.try_clone()?;
        // Check the shared file's actual length, not just the end of this
        // particular extent. A truncated or extended staging owner cannot
        // establish that this collection belongs to the previously accepted set.
        if !self.staging.healthy
            || file.metadata()?.len() != self.staging.bytes
            || self.extent.end()? > self.staging.bytes
            || self.extent.length < 22
        {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "text staging length changed",
            ));
        }
        file.seek(SeekFrom::Start(self.extent.offset))?;
        Ok(file)
    }

    pub(super) fn collection_number(&self) -> u64 {
        self.extent.collection
    }

    // Reuse the encoded dictionary and posting lists across commits. Verify the
    // original header and body against the staged CRC while streaming a second CRC
    // over the new bytes with the replacement commit.
    pub(super) fn write(&self, output: &mut impl Write, commit: u64) -> Result<()> {
        let mut file = self.reader()?;
        let mut prefix = [0; 18];
        file.read_exact(&mut prefix)?;
        let mut old_crc = Crc32c::default();
        old_crc.update(&prefix);
        let mut writer = DigestWriter::new(output);
        // Preserve the schema and collection number from the original header, then
        // write the caller's covered commit. Hash the original prefix
        // independently so source validation still covers the commit being
        // replaced.
        writer.write_all(&prefix[..10])?;
        writer.write_all(&commit.to_le_bytes())?;
        // The already consumed prefix and the footer account for 22 bytes. Copy
        // only the intervening body into the new digest, leaving the old footer to
        // be checked rather than emitted.
        let mut remaining = self
            .extent
            .length
            .checked_sub(22)
            .ok_or_else(invalid_staging)?;
        let mut buffer = [0; 8192];
        while remaining != 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..count])?;
            old_crc.update(&buffer[..count]);
            writer.write_all(&buffer[..count])?;
            remaining -= count as u64;
        }
        // Only a verified source may supply the new checksum footer. A failure here
        // leaves any bytes already sent to the enclosing checkpoint as an
        // incomplete segment.
        self.check_crc(&mut file, old_crc.finish())?;
        writer.finish_segment()
    }

    fn check_crc(&self, file: &mut File, actual: u32) -> Result<()> {
        let mut tail = [0; 4];
        file.read_exact(&mut tail)?;
        // Require the externally retained checksum, the checksum read from the
        // file and the recomputed digest all to agree. An intact footer alone
        // gives no evidence about the bytes just streamed.
        if actual != self.extent.checksum || u32::from_le_bytes(tail) != self.extent.checksum {
            return Err(staging_checksum_error());
        }
        Ok(())
    }

    pub(crate) fn visit(
        &self,
        context: &JsonOperationContext,
        capacity: usize,
        visit: &mut TextSegmentVisitor<'_>,
    ) -> Result<bool> {
        let mut file = self.reader()?;
        // reader() has established the minimum framing length and positioned this
        // descriptor at the extent start. Exclude the stored four-byte CRC from
        // the body limit converted to the decoder's address size.
        let length = usize::try_from(self.extent.length - 4).map_err(|_| invalid_staging())?;
        let mut error = None;
        // Read only the body through CrcReader so the file is at the stored footer
        // afterwards. A callback may fail early, but the remainder of the body
        // still participates in the checksum before its error can be returned.
        let (parsed, checksum) = {
            let mut checked = CrcReader {
                // Place the byte limit below the buffer. Read-ahead can then fill
                // from the body only, leaving the fixed footer available on the
                // same descriptor after the buffered reader is dropped.
                source: BufReader::with_capacity(capacity, (&mut file).take(length as u64)),
                crc: Crc32c::default(),
            };
            let parsed = decode_text_index_segment_with(
                &mut IndexPayloadReader::new(&mut checked, length, 0),
                // The decoder checks the covered sequence recorded in this saved
                // extent. A reused body can describe an earlier commit until the
                // separate output writer replaces its coverage field for the new
                // checkpoint.
                self.extent.commit,
                0,
                TextDecodeMode::Visit {
                    error: &mut error,
                    visit,
                },
                context,
            );
            // A decoder I/O error means the source itself could not supply the
            // requested bytes. Return immediately instead of trying to classify
            // that failure by draining a reader that already failed.
            if matches!(&parsed, Err(IndexReadError::Io(_))) {
                return parsed.map(|_| true).map_err(staging_read_error);
            }
            std::io::copy(&mut checked, &mut std::io::sink())?;
            (parsed, checked.crc.finish())
        };
        // Validate against both the footer in the staging file and the checksum
        // recorded for this accepted extent. A consumer can observe rows before
        // these checks finish, so successful callbacks alone cannot establish
        // success.
        self.check_crc(&mut file, checksum)?;
        // If the decoder reports a resource error after the callback has already
        // failed, preserve that first callback error. A read or checksum failure
        // during the drain takes precedence.
        // Only a resource refusal from further decoding can yield to the stored
        // visitor error here. A malformed typed body takes the ordinary
        // staging-read error path even if an earlier callback also failed.
        if matches!(&parsed, Err(IndexReadError::Resource(_))) {
            if let Some(error) = error {
                return Err(error);
            }
        }
        parsed.map_err(staging_read_error)?;
        // If decoding and all physical checks succeeded, the visitor may still
        // have rejected an event. Preserve that failure even though the staged
        // body itself can be read correctly.
        match error {
            Some(error) => Err(error),
            None => Ok(true),
        }
    }
}

impl TextCheckpoint {
    // Capture unchanged bodies while the old suffix is still readable. Missing or
    // stale extents are left out of this directory for build_reclaiming to
    // regenerate from visible chunks.
    pub(crate) fn prefetch(
        storage: &Storage,
        catalog: &Catalog,
        source: Option<&IndexSegmentSource>,
        changed: &BTreeSet<String>,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        // Admit an extent slot for every live collection before copying any bytes
        // from the old suffix. The retained list can be shorter when the directory
        // did not contain a usable text record for some collections.
        let mut bodies = ReservedVec::with_capacity(
            catalog.live_collections().count(),
            context,
            "operation_scratch",
        )?;
        // Place the temporary copy beside the database and share one staging owner
        // across collections. The extent descriptors refer to offsets in that
        // owner's file and cannot be used with a replacement staging file.
        let directory = storage.path().parent().unwrap_or_else(|| Path::new("."));
        let mut staging = None;
        if let Some(source) = source {
            for collection in catalog.live_collections() {
                if !changed.contains(&collection.collection_id) {
                    // Open the shared temporary file only when an unchanged
                    // collection could supply a body. A missing source or a
                    // catalog containing only changed collections needs no staging
                    // file here.
                    if staging.is_none() {
                        staging = Some(TextStaging::new(directory, disk.clone(), context)?);
                    }
                    // A missing or invalid old extent produces no reusable entry.
                    // A hard read or disk-budget failure propagates, since it
                    // cannot establish whether the copy completed.
                    if let Some(body) = source.stage_text_into(
                        storage,
                        &collection.collection_id,
                        staging.as_mut().expect("created staging"),
                        context,
                    )? {
                        bodies.push(body);
                    }
                }
            }
        }
        // Store extents by numeric collection ID for the later reuse lookup.
        // Physical order in the temporary file and catalog iteration order do not
        // form the lookup contract.
        bodies
            .as_mut_slice()
            .sort_unstable_by_key(|body| body.collection);
        Ok(Self {
            staging,
            bodies,
            // Prefetch captures reusable bytes only. Statistics must be derived
            // later from the full set of both reused and regenerated bodies for
            // the target catalog view.
            stats: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn build<T: AsRef<str>>(
        storage: &Storage,
        catalog: &Catalog,
        commit: u64,
        reuse: Option<(Self, &BTreeSet<String>)>,
        load: impl FnMut(&VisibleChunk<'_>) -> Result<T>,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        Self::build_reclaiming(storage, catalog, commit, reuse, load, disk, context, || {
            false
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_reclaiming<T: AsRef<str>>(
        storage: &Storage,
        catalog: &Catalog,
        commit: u64,
        mut reuse: Option<(Self, &BTreeSet<String>)>,
        mut load: impl FnMut(&VisibleChunk<'_>) -> Result<T>,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
        mut reclaim: impl FnMut() -> bool,
    ) -> Result<Self> {
        // Numeric ordering also supports IDs crossing the minimum-width boundary.
        let mut collections = crate::json::retry_after_memory_reclaim(
            || {
                ReservedVec::with_capacity(
                    catalog.live_collections().count(),
                    context,
                    "operation_scratch",
                )
            },
            &mut reclaim,
        )?;
        // Keep references to the live collection definitions while building the
        // directory. Counting and collecting from this immutable catalog yield the
        // same number without copying names or configurations.
        for collection in catalog.live_collections() {
            collections.push(collection);
        }
        collections
            .as_mut_slice()
            .sort_unstable_by_key(|collection| {
                catalog_id_number(&collection.collection_id).unwrap_or(u64::MAX)
            });
        // A staged body can survive a new commit if its collection still exists and
        // none of that collection's text inputs changed. Reclaim the holes left by
        // discarded bodies before extending the staging file.
        if let Some((reuse, changed)) = &mut reuse {
            // Prefetch may have run before several deferred commits. Recheck the
            // changed-collection set against the catalog at build time so a
            // previously copied body can be discarded even if it was reusable when
            // captured.
            let old_count = reuse.bodies.as_slice().len();
            reuse.bodies.retain(|body| {
                collections
                    .as_slice()
                    .binary_search_by_key(&body.collection, |collection| {
                        catalog_id_number(&collection.collection_id).unwrap_or(u64::MAX)
                    })
                    .is_ok_and(|position| {
                        !changed.contains(&collections.as_slice()[position].collection_id)
                    })
            });
            // Reclaim holes only if the retained extent set actually shrank.
            // Moving bytes also updates their saved offsets, so every subsequent
            // reader must use the post-reclaim extent records.
            if old_count != reuse.bodies.as_slice().len() {
                if let Some(staging) = &mut reuse.staging {
                    staging.reclaim(reuse.bodies.as_mut_slice())?;
                }
            }
        }
        let directory = storage.path().parent().unwrap_or_else(|| Path::new("."));
        let (mut staging, old) = match reuse {
            Some((reuse, _)) => (reuse.staging, reuse.bodies),
            None => (
                None,
                ReservedVec::with_capacity(0, context, "operation_scratch")?,
            ),
        };
        let count = collections.as_slice().len();
        // Reuse the old directory allocation when it already covers every live
        // collection. For an empty target catalog the zero-capacity replacement
        // permits that allocation to be released as well.
        let mut bodies = if count != 0 && old.capacity() >= count {
            old
        } else {
            // Keep the old array and its credit together until growth succeeds.
            let mut bodies = crate::json::retry_after_memory_reclaim(
                || ReservedVec::with_capacity(count, context, "operation_scratch"),
                &mut reclaim,
            )?;
            // Move existing extent descriptors into the admitted replacement
            // array. Their old slot credit stays alive through the move while the
            // original backing allocation has not yet been released.
            let (old, old_work) = old.into_parts();
            for body in old {
                bodies.push(body);
            }
            // Keep the old slot ownership until every retained body has moved.
            // Returning its workspace before consuming the old allocation would
            // allow concurrent operations to spend bytes still in use.
            drop(old_work);
            bodies
        };
        // The retained prefix is sorted and continues to be searchable while new
        // extents accumulate at the end. Sorting the entire growing array on every
        // append would do needless work.
        let retained = bodies.as_slice().len();
        for collection in collections.as_slice() {
            // Sorting accepts a fallback key for malformed text, but no new body
            // may be emitted for such an ID. Parse it fallibly here before either
            // searching the retained prefix or starting a build.
            let number = catalog_id_number(&collection.collection_id)?;
            if bodies.as_slice()[..retained]
                .binary_search_by_key(&number, |body| body.collection)
                .is_err()
            {
                // A fully reusable nonempty checkpoint already owns its staging
                // file. Create a new one only on the first collection that
                // actually needs its encoded body built.
                if staging.is_none() {
                    staging = Some(crate::json::retry_after_memory_reclaim(
                        || TextStaging::new(directory, disk.clone(), context),
                        &mut reclaim,
                    )?);
                }
                // The loader supplies the actual visible chunk text for this
                // target collection. Rebuilt bodies use the new covered commit
                // directly, while the write path can supply that commit when
                // emitting a reused body.
                let body = build_collection_into(
                    catalog,
                    collection,
                    commit,
                    &mut load,
                    directory,
                    staging.as_mut().expect("created staging"),
                    context,
                );
                // The failed collection has rolled back; retained bodies stay in the same owner.
                let body = match body {
                    // Retry the failed collection only after a memory refusal and
                    // a successful reclaim callback. Already accepted extents stay
                    // in place; a disk, format or I/O failure must abort this
                    // build.
                    Err(error) if is_memory_budget_error(&error) && reclaim() => {
                        build_collection_into(
                            catalog,
                            collection,
                            commit,
                            &mut load,
                            directory,
                            staging.as_mut().expect("created staging"),
                            context,
                        )
                    }
                    result => result,
                }?;
                bodies.push(body);
            }
        }
        // Merge the ordering of reused and appended descriptors before the
        // statistics pass calls body() by binary search. The file itself need not
        // be rearranged to establish this order.
        bodies
            .as_mut_slice()
            .sort_unstable_by_key(|body| body.collection);
        // When the last collection has disappeared, release the old staging owner
        // as well as its extents. An empty checkpoint can still produce valid
        // statistics without retaining temporary file bytes.
        if count == 0 {
            staging = None;
        }
        drop(collections);
        let mut result = Self {
            staging,
            bodies,
            stats: None,
        };
        // Keep the statistics reader's optional buffer separate from the directory
        // and accepted bodies. Retrying with a smaller input buffer must not
        // invalidate anything needed to read the next collection.
        let mut input = context.workspace_reservation();
        if count != 0 {
            crate::json::retry_after_memory_reclaim(
                || input.grow_to(0, BUFFER_BYTES, "operation_scratch"),
                &mut reclaim,
            )?;
        }
        // Derive one set of statistics from the same accepted bodies that will be
        // written, including reused ones. Give up optional read-ahead first on a
        // budget refusal, then try other reclaimable memory.
        loop {
            let stats =
                TextStatsSummary::from_visitors(catalog, directory, disk, context, |id, visit| {
                    result
                        // Visit each collection through the final extent
                        // directory. Recomputing from catalog text alone could
                        // report values from a different body than the one that
                        // will be written for a reused collection.
                        .body(id)?
                        .visit(context, input.reserved_bytes(), visit)
                });
            match stats {
                // Discard read-ahead before invoking the wider reclaim callback.
                // On either retry the statistics pass starts again from the
                // accepted extents, without appending any new text bodies.
                Err(error) if is_memory_budget_error(&error) && input.reserved_bytes() > 0 => {
                    input.retain(0)
                }
                Err(error) if is_memory_budget_error(&error) && reclaim() => {}
                stats => {
                    result.stats = stats?;
                    break;
                }
            }
        }
        // The statistics builder can signal an unavailable input by returning no
        // summary. A completed private staging directory promises every live
        // collection has a valid body, so treat that outcome as staging failure.
        if result.stats.is_none() {
            return Err(invalid_staging());
        }
        Ok(result)
    }

    pub(crate) fn body(&self, id: &str) -> Result<StagedText<'_>> {
        let number = catalog_id_number(id)?;
        self.bodies
            .as_slice()
            .binary_search_by_key(&number, |body| body.collection)
            // The staged view borrows both the shared temporary file and this
            // extent descriptor. Returning it by reference preserves the lifetime
            // of the file offsets through a visit or checkpoint write.
            .map(|position| StagedText {
                staging: self.staging.as_ref().expect("nonempty staging"),
                extent: &self.bodies.as_slice()[position],
            })
            // At this point a missing body is a private directory inconsistency.
            // Unlike a prefetch miss against an optional old index, it cannot be
            // returned as a successfully built incomplete checkpoint.
            .map_err(|_| invalid_staging())
    }

    pub(crate) fn take_stats(&mut self) -> TextStatsSummary {
        // Transfer the finished statistics to their next owner exactly once.
        // Queries about zero-frequency terms must be made while this checkpoint
        // still owns the summary.
        self.stats.take().expect("prepared text statistics")
    }

    pub(crate) fn has_zero_frequency_terms(&self) -> bool {
        self.stats
            .as_ref()
            .expect("prepared text statistics")
            .has_zero_frequency_terms()
    }
}

#[cfg(test)]
impl TextSpool {
    pub(crate) fn new(
        collection: u64,
        commit: u64,
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        Ok(Self {
            staging: TextStaging::new(directory, disk, context)?,
            extent: TextExtent {
                collection,
                commit,
                offset: 0,
                length: 0,
                checksum: 0,
            },
        })
    }

    fn copy_body_from(
        &mut self,
        reader: &mut dyn Read,
        length: usize,
        commit: u64,
        context: &JsonOperationContext,
        capacity: usize,
    ) -> std::result::Result<u64, IndexReadError> {
        self.extent = self
            .staging
            .copy_body_from(reader, length, commit, context, capacity)?;
        Ok(self.extent.collection)
    }

    fn finish_copy(&mut self) -> Result<()> {
        self.staging.finish_copy(&self.extent)
    }

    fn view(&self) -> StagedText<'_> {
        StagedText {
            staging: &self.staging,
            extent: &self.extent,
        }
    }

    fn write(&self, output: &mut impl Write, commit: u64) -> Result<()> {
        self.view().write(output, commit)
    }

    fn visit(
        &self,
        context: &JsonOperationContext,
        capacity: usize,
        visit: &mut TextSegmentVisitor<'_>,
    ) -> Result<bool> {
        self.view().visit(context, capacity, visit)
    }
}

#[cfg(test)]
fn build_collection<T: AsRef<str>>(
    catalog: &Catalog,
    collection: &CollectionInfo,
    commit: u64,
    load: &mut impl FnMut(&VisibleChunk<'_>) -> Result<T>,
    directory: &Path,
    disk: &Arc<TemporaryDiskBudget>,
    context: &JsonOperationContext,
) -> Result<TextSpool> {
    let mut staging = TextStaging::new(directory, disk.clone(), context)?;
    let extent = build_collection_into(
        catalog,
        collection,
        commit,
        load,
        directory,
        &mut staging,
        context,
    )?;
    Ok(TextSpool { staging, extent })
}

// The segment puts chunk lengths before the postings, but both come from the
// same token stream. Accumulate lengths in numeric chunk order and spill
// (term, big-endian chunk number) keys for the external sort. Equal keys
// supply one frequency; sorting puts each term's postings in numeric order.
// The writer can then emit a term at a time and backpatch its counts.
fn build_collection_into<T: AsRef<str>>(
    catalog: &Catalog,
    collection: &CollectionInfo,
    commit: u64,
    load: &mut impl FnMut(&VisibleChunk<'_>) -> Result<T>,
    directory: &Path,
    staging: &mut TextStaging,
    context: &JsonOperationContext,
) -> Result<TextExtent> {
    let result = (|| {
        // Capture the absolute file offset at which this body starts. The writer
        // tracks patch positions relative to that base. A failed attempt can
        // truncate the unfinished body back to the last accepted extent.
        let offset = staging.begin()?;
        let collection_number = catalog_id_number(&collection.collection_id)?;
        let source = catalog
            .live_chunks()
            .filter(|chunk| chunk.collection_id == collection.collection_id);
        // Keep only a borrowed chunk and its future token count for each live row.
        // Sorting this compact directory makes the emitted IDs canonical without
        // retaining the whole collection's text.
        let mut chunks =
            ReservedVec::with_capacity(source.clone().count(), context, "operation_scratch")?;
        for chunk in source {
            chunks.push((chunk, 0usize));
        }
        // Use the numeric key for sort order even when the decimal spelling has
        // leading zeroes or variable width. The later canonical write still
        // validates each ID before storing it.
        chunks.as_mut_slice().sort_unstable_by_key(|(chunk, _)| {
            catalog_id_number(&chunk.chunk_id).unwrap_or(u64::MAX)
        });
        // The parser uses numeric IDs from the segment. Distinct decimal spellings
        // can collapse to the same number, so compare the sorted adjacent numeric
        // values before writing any entries.
        for pair in chunks.as_slice().windows(2) {
            if catalog_id_number(&pair[0].0.chunk_id)? == catalog_id_number(&pair[1].0.chunk_id)? {
                return Err(invalid_staging());
            }
        }
        let mut frequencies =
            ExternalTermFrequencies::new(directory, staging.disk.clone(), context);
        let mut max_term = 0;
        // Fetch the text of one live chunk at a time and accumulate its term
        // occurrences in the sorter. Both operate under the same memory budget. A
        // bounded spill can free sort space while the next text is being loaded.
        for (chunk, count) in chunks.as_mut_slice() {
            let visible = VisibleChunk {
                chunk,
                record_ref: catalog.record_ref_for_chunk(&chunk.chunk_id),
            };
            let text = after_spill(&mut frequencies, || load(&visible))?;
            // The sort key is the term followed by a separator and a big-endian
            // numeric chunk ID. Its byte order groups equal terms and orders their
            // postings by chunk, even when the decimal form of an ID grows in width.
            let id = catalog_id_number(&chunk.chunk_id)?.to_be_bytes();
            crate::tokenizer::visit_tokens_in_reclaiming(
                text.as_ref(),
                context,
                &mut frequencies,
                |frequencies, token| {
                    // Count every emitted token toward its chunk length. Multiple
                    // occurrences of the same term and chunk are later merged into
                    // one posting. The posting frequency records the original
                    // occurrence total.
                    max_term = max_term.max(token.len());
                    *count = count.checked_add(1).ok_or_else(invalid_staging)?;
                    frequencies.push(&[token.as_bytes(), &[0], &id], 1)
                },
                |frequencies| {
                    // Release an empty sort allocation first. Spill owned keys to
                    // disk only if that release did not make progress. This avoids
                    // writing a run solely to release capacity that contains no
                    // terms.
                    Ok(frequencies.release_empty_batch_capacity()? || frequencies.spill()?)
                },
            )?;
            // Release the chunk's loaded text before processing the next one. Empty
            // chunks have no postings, but a length of one is needed for the BM25
            // scoring convention.
            drop(text);
            *count = (*count).max(1);
        }
        // The longest token bounds a single reusable previous-term buffer during
        // the merge. A memory refusal may spill frequencies before allocating it,
        // so the sorter and comparator share the limit.
        let mut previous = after_spill(&mut frequencies, || {
            ReservedVec::zeroed(max_term, context, "operation_scratch")
        })?;
        let mut io = context.workspace_reservation();
        let capacity = io.grow_to(0, BUFFER_BYTES, "operation_scratch")?;
        let mut writer = StagingWriter {
            output: Some(BufWriter::with_capacity(capacity, &mut staging.file)),
            cleanup: &mut staging.cleanup,
            base: offset,
            bytes: 0,
            reserved: &mut staging.reserved,
        };
        // The stable text schema and tokenizer tags identify how to parse and
        // score this segment. Emit the actual BM25 parameters once for this body
        // before writing any per-chunk counts.
        writer.append(&TEXT_SEGMENT_SCHEMA_VERSION.to_le_bytes())?;
        writer.u64(collection_number)?;
        writer.u64(commit)?;
        writer.append(&TEXT_INDEX_KIND_BM25.to_le_bytes())?;
        writer.append(&DEFAULT_EN_TOKENIZER_ID.to_le_bytes())?;
        writer.append(&(crate::index::text::BM25_K1 as f32).to_le_bytes())?;
        writer.append(&(crate::index::text::BM25_B as f32).to_le_bytes())?;
        // The chunk table precedes the term dictionary. It includes every live
        // chunk, even an empty one, and stores the document ID to tie the row back
        // to the same catalog snapshot.
        writer.u64(chunks.as_slice().len() as u64)?;
        for (chunk, count) in chunks.as_slice() {
            writer.u64(catalog_id_number(&chunk.chunk_id)?)?;
            writer.u64(catalog_id_number(&chunk.document_id)?)?;
            writer.u64(*count as u64)?;
        }
        // Let the per-chunk array go before merging frequencies. The chunk table
        // is already serialized and numeric chunk IDs in the sort keys supply all
        // the posting identities still needed.
        drop(chunks);
        // Term and posting counts precede the entries they describe on disk. Reserve
        // their fields here and patch them at group boundaries; a term's final
        // posting count is unknown until the next key arrives.
        let term_count_offset = writer.bytes;
        writer.u64(0)?;
        let mut previous_len = 0;
        let mut count_offset = 0;
        let mut count = 0u64;
        let mut terms = 0u64;
        // Sorting may still admit merge heads. Release the optional writer buffer
        // until the first event, when all merge and previous-term storage is live.
        // Release the optional output buffer while the sorter admits its merge
        // heads. The first merged posting may request it again once the required
        // sorter and previous-term storage are live.
        writer.set_capacity(0)?;
        io.retain(0);
        writer.release_unused_disk();
        frequencies.finish_batch(|key, frequency| {
            // Peel off the fixed-width separator and big-endian ID at the end of
            // each key. Each merged key yields one posting. The merged frequency is
            // the number of occurrences, while advancing the posting count once
            // measures document frequency in chunks.
            // Remove the separator and numeric suffix from the end. Keeping the
            // suffix fixed-width makes extracting it independent of the term's
            // UTF-8 byte length.
            let split = key.len().checked_sub(9).ok_or_else(invalid_staging)?;
            if key[split] != 0 || frequency == 0 {
                return Err(invalid_staging());
            }
            let term = std::str::from_utf8(&key[..split]).map_err(|_| invalid_staging())?;
            // A change in the term opens a new posting group. Finalize the earlier
            // count first, then copy the new term into the reusable buffer. The
            // counter reset below applies only to the new group.
            if terms == 0 || &previous.as_slice()[..previous_len] != term.as_bytes() {
                if terms != 0 {
                    writer.patch_counts(count_offset, count)?;
                } else {
                    // Only the first emitted posting attempts to restore buffered
                    // output. For a collection with no terms this callback is
                    // never invoked, and the final count remains zero.
                    let capacity = io.grow_to(0, BUFFER_BYTES, "operation_scratch")?;
                    writer.set_capacity(capacity)?;
                }
                previous.as_mut_slice()[..term.len()].copy_from_slice(term.as_bytes());
                previous_len = term.len();
                writer.string(term)?;
                count_offset = writer.bytes;
                // The placeholder contains two adjacent u64 counts. patch_counts
                // fills them after the final posting for this term, preserving the
                // fixed-width positions of the already written bytes.
                writer.append(&[0; 16])?;
                count = 0;
                terms = terms.checked_add(1).ok_or_else(invalid_staging)?;
            }
            // The sorter's big-endian ID was chosen for lexical ordering. Convert
            // it back to a u64 because the final segment writes integer fields in
            // little-endian order.
            let chunk =
                u64::from_be_bytes(key[split + 1..].try_into().map_err(|_| invalid_staging())?);
            // Serialize the chunk identity followed by its occurrence frequency.
            // The per-term count increases once per unique key from the merge. A
            // posting's frequency can be larger than that increment.
            writer.u64(chunk)?;
            writer.u64(frequency)?;
            count = count.checked_add(1).ok_or_else(invalid_staging)?;
            Ok(())
        })?;
        // There is no next term to close the last group. Patch that group if
        // present, and always patch the total term count so a collection with no
        // tokens has a complete dictionary.
        if terms != 0 {
            writer.patch_counts(count_offset, count)?;
        }
        writer.patch(term_count_offset, &terms.to_le_bytes())?;
        // Flush before opening the same staged range for verification. Logical
        // byte offsets include buffered data, but the CRC read must see the
        // completed file image.
        writer.output().flush()?;
        let bytes = writer.bytes;
        writer.release_unused_disk();
        drop(writer);
        drop(io);
        // Patching counts changed bytes already written. Read the finished body back
        // for its CRC so the extent describes the final wire image, then append that
        // checksum before exposing the extent to reuse.
        staging.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = [0; 8192];
        let mut remaining = bytes;
        let mut crc = Crc32c::default();
        // Re-read exactly the patched body. CRC verification during later reuse
        // excludes the footer and confines the read to this extent, so earlier
        // collections cannot contaminate its checksum.
        while remaining != 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            staging.file.read_exact(&mut buffer[..count])?;
            crc.update(&buffer[..count]);
            remaining -= count as u64;
        }
        let length = bytes.checked_add(4).ok_or_else(invalid_staging)?;
        let end = offset.checked_add(length).ok_or_else(invalid_staging)?;
        let reserved = staging
            .reserved
            .checked_add(4)
            .ok_or_else(invalid_staging)?;
        // Reserve the footer separately after releasing unused body credit. The
        // extent remains provisional if this reservation or the footer write
        // fails.
        staging.cleanup.reserve_disk(4)?;
        staging.reserved = reserved;
        staging.file.write_all(&crc.finish().to_le_bytes())?;
        let body = TextExtent {
            collection: collection_number,
            commit,
            offset,
            length,
            checksum: crc.finish(),
        };
        // Publishing this extent changes the rollback boundary. Delay it until the
        // final CRC word is written, so failure in any earlier stage preserves
        // every previously accepted body.
        staging.bytes = end;
        Ok(body)
    })();
    // Return the spool to its last accepted boundary even when the last operation
    // already emitted some bytes. The caller can keep using prior extents only
    // after this rollback succeeds.
    if result.is_err() {
        staging.rollback()?;
    }
    result
}

struct StagingWriter<'a> {
    // Buffered view of the current collection's portion of the shared file. Option
    // allows replacing or disarming it without a drop-time flush.
    output: Option<BufWriter<&'a mut TemporaryFile>>,
    // File owner's disk accounting guard, used before extending the tentative
    // range.
    cleanup: &'a mut TempFileGuard,
    // Absolute staging-file offset of the first byte of this body.
    base: u64,
    // Body-relative end successfully accepted by append, unchanged by an in-place
    // count patch.
    bytes: u64,
    // Shared absolute end reserved against the disk budget. An append may reserve
    // extra space to amortize small writes.
    reserved: &'a mut u64,
}

impl<'a> StagingWriter<'a> {
    fn release_unused_disk(&mut self) {
        // Release only bytes beyond the end of the currently filled body. The
        // shared credit for earlier extents and this accepted prefix must remain
        // attached to the file.
        let end = self.base + self.bytes;
        self.cleanup.release_disk(*self.reserved - end);
        *self.reserved = end;
    }
    fn output(&mut self) -> &mut BufWriter<&'a mut TemporaryFile> {
        self.output.as_mut().expect("live staging writer")
    }
    fn set_capacity(&mut self, capacity: usize) -> Result<()> {
        // A buffer change must flush the bytes already accepted by StagingWriter.
        // The new buffer wraps the same file at that position, so changing its
        // capacity cannot create a gap or replay the pending prefix.
        self.output().flush()?;
        let (file, buffer) = self
            .output
            .take()
            .expect("live staging writer")
            .into_parts();
        drop(buffer);
        self.output = Some(BufWriter::with_capacity(capacity, file));
        Ok(())
    }
    // Reserve before extending the logical body, because write_all may already have
    // written a prefix when it fails. The owner retains the larger reservation
    // until rollback restores the physical file.
    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(invalid_staging)?;
        let absolute = self.base.checked_add(end).ok_or_else(invalid_staging)?;
        // Compare the absolute body end with the total disk grant, including
        // previous accepted bodies. Filling a preferred reservation can leave
        // surplus until release_unused_disk or rollback.
        if absolute > *self.reserved {
            *self.reserved += self
                .cleanup
                .reserve_disk_buffer(absolute - *self.reserved, BUFFER_BYTES as u64)?;
        }
        self.output().write_all(bytes)?;
        self.bytes = end;
        Ok(())
    }
    fn u64(&mut self, value: u64) -> Result<()> {
        self.append(&value.to_le_bytes())
    }
    fn string(&mut self, value: &str) -> Result<()> {
        let length = u32::try_from(value.len()).map_err(|_| invalid_staging())?;
        self.append(&length.to_le_bytes())?;
        self.append(value.as_bytes())
    }
    // Count placeholders are relative to the beginning of this body. Restrict
    // patches to its accepted written range, then restore the append position so
    // later terms do not overwrite earlier ones.
    fn patch(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        if offset
            .checked_add(bytes.len() as u64)
            .map_or(true, |end| end > self.bytes)
        {
            return Err(invalid_staging());
        }
        // Use the body's absolute start to translate the placeholder offset. The
        // patch overwrites part of this body, so it advances neither its length
        // nor the total reserved file extent.
        let absolute = self.base.checked_add(offset).ok_or_else(invalid_staging)?;
        self.output().seek(SeekFrom::Start(absolute))?;
        self.output().write_all(bytes)?;
        let end = self.base + self.bytes;
        self.output().seek(SeekFrom::Start(end))?;
        Ok(())
    }
    // Document frequency and posting count are identical here because the external
    // merge emits one pair per distinct (term, chunk). Both are explicit fields in
    // the segment format.
    fn patch_counts(&mut self, offset: u64, count: u64) -> Result<()> {
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&count.to_le_bytes());
        bytes[8..].copy_from_slice(&count.to_le_bytes());
        self.patch(offset, &bytes)
    }
}

// The normal path flushes explicitly. Disassemble the writer on drop so unwinding
// cannot retry a failed write. Rollback then restores the accepted staging
// boundary.
impl Drop for StagingWriter<'_> {
    fn drop(&mut self) {
        if let Some(output) = self.output.take() {
            drop(output.into_parts().1);
        }
    }
}

struct CopyReader<'a> {
    // Source segment body presented to the validator.
    source: &'a mut dyn Read,
    // Staging destination that receives precisely the bytes returned to the
    // decoder.
    output: &'a mut dyn Write,
    // CRC of the successfully forwarded body prefix.
    crc: Crc32c,
    // Length of that prefix for the final comparison with the declared body length.
    bytes: u64,
}
impl Read for CopyReader<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let count = self.source.read(bytes)?;
        // This reader copies the prefix actually obtained from the source before
        // returning it to the consumer. If the write fails, the enclosing copy
        // operation cannot publish a completed extent from its counters.
        self.output.write_all(&bytes[..count])?;
        self.crc.update(&bytes[..count]);
        self.bytes += count as u64;
        Ok(count)
    }
}
struct CrcReader<R> {
    // Bounded body stream. The stored CRC word is read separately by the owner.
    source: R,
    // Running checksum over the bytes that the typed decoder or its final drain
    // actually consumed.
    crc: Crc32c,
}
impl<R: Read> Read for CrcReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let count = self.source.read(bytes)?;
        self.crc.update(&bytes[..count]);
        Ok(count)
    }
}
fn invalid_staging() -> Error {
    Error::new(ErrorCode::FileFormatError, "invalid staged text checkpoint")
}

fn staging_checksum_error() -> Error {
    Error::new(ErrorCode::ChecksumMismatch, "text staging checksum changed")
}

fn after_spill<T>(
    frequencies: &mut ExternalTermFrequencies<'_>,
    mut attempt: impl FnMut() -> Result<T>,
) -> Result<T> {
    // Use this wrapper around allocation or reads that can safely be attempted
    // again. It handles a memory refusal by evicting sorter work. A file or schema
    // failure has no retry through this branch.
    match attempt() {
        Err(error) if is_memory_budget_error(&error) => {
            // Both spill and release of an empty batch are useful here. Evaluate
            // both even after a successful spill, then retry the refused allocation
            // once only if capacity was made available.
            if frequencies.spill()? | frequencies.release_empty_batch_capacity()? {
                attempt()
            } else {
                Err(error)
            }
        }
        result => result,
    }
}
fn staging_read_error(error: IndexReadError) -> Error {
    match error {
        IndexReadError::Io(error) => error.into(),
        IndexReadError::Resource(error) => error,
        IndexReadError::Invalid(_) => invalid_staging(),
    }
}
