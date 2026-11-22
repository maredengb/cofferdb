use super::*;
use crate::import::{embedding_from_json_in, with_jsonl_location};
use crate::json::{JsonLines, JsonSchema, ReservedString};
use crate::storage::PreparedRecordsBuilder;
use crate::temporary::ExternalUniqueKeys;

impl Database {
    pub(crate) fn import_embeddings_jsonl(
        &self,
        collection_id: &str,
        path: &Path,
        on_conflict: OnConflict,
    ) -> Result<InsertSummary> {
        let context = self.json_context();
        // Hold the connection lock across input preflight, ID allocation and the
        // eventual batch writes. Another handle must not replace a target chunk or
        // change its selected model between checking an early line and committing
        // it.
        let mut inner = self.lock()?;
        let mut collection_work = RecordReservation::from(context.reservation());
        // Resolve the existing destination by internal ID once. The owned settings
        // and their reservation can outlive individual catalog borrows while later
        // stages reclaim optional caches or build records.
        let (collection, work) = inner.read_reclaiming(|inner| {
            copy_collection_in(inner.catalog.collection_by_id(collection_id)?, &context)
        })?;
        collection_work.merge(work);
        let mut models = inner
            .read_reclaiming(|inner| ModelAdmission::new(&inner.catalog, &collection, &context))?;
        // Preserve the first target-dependent error while later input rows are
        // parsed and keyed. Source errors detected before or during the final key
        // merge take precedence.
        let mut preflight_error = None;
        let input = File::open(path)
            .map_err(|error| Error::from(error).with_detail("path", path.display()))?;
        // The external duplicate-key runs and the prepared records spool can exist
        // at the same time. Both draw from the database's shared temporary-disk
        // allowance so each cannot independently consume its full limit.
        let disk = self.temporary_disk_budget();
        let mut records = {
            let Inner {
                text_index,
                persistent_vector_segments,
                lazy_vector_segment,
                ..
            } = &mut *inner;
            // The builder privately stages encoded rows. Running out of memory or
            // disk while building it cannot expose a partially checked import. Its
            // reclaim callback can free resident query data without disturbing the
            // catalog used for lookup.
            PreparedRecordsBuilder::new_in_reclaiming(
                &std::env::temp_dir(),
                disk.limit_bytes(),
                context.clone(),
                disk.clone(),
                || {
                    Inner::release_optional_caches(
                        text_index,
                        persistent_vector_segments,
                        lazy_vector_segment,
                        std::iter::empty(),
                    )
                },
            )?
        };
        // Source keys need to remain unique across the entire file, including rows
        // skipped by the target conflict policy. Use the external key set so this
        // check is independent of how many embeddings fit in a commit.
        let reference_workspace = (inner.active_cache_budget_mb().saturating_mul(1_048_576) / 128)
            .clamp(64 * 1024, 1_048_576);
        // The bounded workspace is the preferred size for grouping keys into
        // sorted runs. A long individual alias may still be admitted against the
        // working-memory limit. Shrinking the cache profile must not silently
        // reject otherwise valid chunk IDs.
        let mut keys = inner.read_reclaiming(|_| {
            ExternalUniqueKeys::new_growing(
                &std::env::temp_dir(),
                disk.clone(),
                reference_workspace,
                &context,
                ErrorCode::DuplicateExternalId,
                "embedding for chunk appears more than once in the same import",
            )
        })?;
        // Capture the time once and format it under a reservation. All accepted
        // embeddings receive this same timestamp even if the source spans several
        // primary commits.
        let millis = time::unix_millis_now();
        let timestamp = inner.read_reclaiming(|_| {
            ReservedString::build(&context, "batch_prepare", |out| {
                time::write_unix_millis(out, millis)
            })
        })?;
        let mut summary = InsertSummary::default();
        // Read directly from the open input file. There is no manifest copy whose
        // size would bound the full JSONL stream here. The parser still enforces
        // the individual line, value and working-memory limits.
        let mut rows =
            JsonLines::new_reclaiming(input, JsonSchema::EmbeddingRow, &context, u64::MAX, || {
                reclaim_caches(&mut inner)
            })
            .map_err(|error| error.with_detail("path", path.display()))?;
        while let Some(row) = rows.next_reclaiming(|| reclaim_caches(&mut inner)) {
            let (line, parsed) = row.map_err(|error| error.with_detail("path", path.display()))?;
            // Keep the parser's reservation with the live JSON tree while
            // producing the typed row. The f32 vector can coexist with its
            // original JSON number array during conversion, so both allocations
            // must fit.
            let mut row_work = context.reservation();
            let (mut value, work) = parsed.into_parts();
            row_work.merge(work);
            let (input, conversion) = inner
                .read_reclaiming(|_| embedding_from_json_in(&mut value, &context))
                .map_err(|error| with_jsonl_location(error, path, line))?;
            row_work.merge(conversion);
            // The conversion has taken the row's strings and produced its vector.
            // Drop the remaining JSON containers before reducing credit to the
            // capacity still held by the typed input.
            drop(value);
            row_work.retain(input.workspace_bytes());
            // A row whose declared dimension disagrees with its own vector length
            // is malformed even if there is no matching chunk in the target. This
            // structural failure is diagnosed from the source before the catalog
            // is consulted.
            input
                .validate_structure()
                .map_err(|error| with_jsonl_location(error, path, line))?;
            // Register the key and its source line before looking up the target.
            // Two lines for the same chunk are a duplicate input even if both
            // would independently skip an already stored embedding.
            keys.push_value_reclaiming(
                &[input.chunk_external_id.as_bytes()],
                line as u64,
                0,
                || reclaim_caches(&mut inner),
            )
            .map_err(|error| with_jsonl_location(error, path, line))?;
            // Even after a target or model error, the rest of the input must still be parsed
            // and keyed: malformed input or duplicate source keys take precedence. Skip only
            // the checks and staging that depend on the target catalog.
            let consumed = (|| {
                if preflight_error.is_some() {
                    return Ok(());
                }
                // Validate the dimension and find a visible chunk in the named
                // collection before applying the stored-embedding policy. An
                // absent or deleted target cannot become a successful skip.
                input.validate_target_dimension(collection.embedding_dim)?;
                let chunk = inner
                    .catalog
                    .chunk_by_external_id(collection_id, &input.chunk_external_id)?;
                if inner.catalog.embedding_for_chunk(&chunk.chunk_id).is_some() {
                    match on_conflict {
                        OnConflict::Error => {
                            return Err(Error::new(
                                ErrorCode::DuplicateExternalId,
                                format!(
                                    "embedding for chunk {} already exists",
                                    input.chunk_external_id
                                ),
                            ))
                        }
                        // A skipped row already contributed its source key. It
                        // contributes neither a model to the batch nor a put
                        // record, so the next accepted row retains the next
                        // available embedding ID.
                        OnConflict::Skip => {
                            summary.embeddings_skipped += 1;
                            return Ok(());
                        }
                        OnConflict::Replace => summary.embeddings_replaced += 1,
                    }
                }
                // An unbound collection can first acquire its model from an
                // accepted row. Record a conflicting model with its line now, but
                // finish checking later source keys. Allocation and I/O errors are
                // handled separately below.
                match inner
                    .read_reclaiming(|_| models.validate_write(&input.model_name, None, &context))
                {
                    Err(error) if error.code() == ErrorCode::EmbeddingModelMismatch => {
                        preflight_error = Some(with_jsonl_location(error, path, line));
                        return Ok(());
                    }
                    result => result?,
                }
                // Derive IDs from successful insertions while the catalog is locked.
                // A skipped row must consume neither an embedding ID nor a model
                // binding for subsequent rows.
                // Keep the generated ID, internal chunk and collection IDs and
                // timestamp reserved until their record has been staged. Build
                // them as one retryable unit so an allocation refusal does not
                // consume another candidate ID.
                let ([embedding_id, chunk_id, collection_id, created_at], _fields_work) = inner
                    .read_reclaiming(|inner| {
                        let mut work = RecordReservation::from(context.reservation());
                        // Look up the chunk inside the retryable copy operation.
                        // This avoids keeping a catalog borrow across cache
                        // reclamation and ensures that each attempt takes all its
                        // field copies from the same locked view.
                        let chunk = inner
                            .catalog
                            .chunk_by_external_id(collection_id, &input.chunk_external_id)?;
                        let (embedding_id, id_work) = inner
                            .catalog
                            .next_id_in("emb_", summary.embeddings_inserted as u64, &context)?
                            .into_parts();
                        work.merge_workspace(id_work);
                        let [chunk_id, collection_id, created_at] = work.copy_strings(
                            [&chunk.chunk_id, collection_id, timestamp.as_ref()],
                            &context,
                        )?;
                        Ok(([embedding_id, chunk_id, collection_id, created_at], work))
                    })?;
                // Serialization transfers this row to the private spool. Move its
                // vector and model string into the record directly. A replacement
                // is encoded as a new EmbeddingPut that switches the chunk's
                // selected embedding on replay.
                records.push_retained(
                    RecordPayload::EmbeddingPut(Embedding {
                        embedding_id,
                        chunk_id,
                        collection_id,
                        chunk_external_id: input.chunk_external_id,
                        model_name: input.model_name,
                        dim: input.dim,
                        vector: input.vector,
                        created_at,
                    }),
                    &mut row_work,
                )?;
                // Increment the insert count only after the spool accepts the
                // record. A replacement counts as both an insertion of a new
                // embedding and a replacement of the previous selection.
                summary.embeddings_inserted += 1;
                Ok(())
            })();
            // Delay only the catalog-dependent errors listed here. Other failures,
            // including staging and resource errors, end the import immediately
            // with the source location and without appending its staged prefix.
            match consumed {
                Err(error)
                    if matches!(
                        error.code(),
                        ErrorCode::ChunkNotFound
                            | ErrorCode::DuplicateExternalId
                            | ErrorCode::InvalidEmbeddingDimension
                    ) =>
                {
                    preflight_error = Some(with_jsonl_location(error, path, line));
                }
                result => result.map_err(|error| with_jsonl_location(error, path, line))?,
            }
        }
        // Release the line reader before merging the duplicate-key runs. That
        // final merge may find equal keys in different runs and needs working
        // space while the prepared records are still retained.
        drop(rows);
        // Consume and validate the full input against one locked snapshot before any commit.
        keys.finish_reclaiming(|| reclaim_caches(&mut inner))
            .map_err(|error| error.with_detail("path", path.display()))?;
        // Once uniqueness has been established, return the earliest saved target
        // error. The record builder is still private on this path, so previously
        // accepted rows have not altered the catalog or primary file.
        if let Some(error) = preflight_error {
            return Err(error);
        }
        // Passing the finished spool crosses the preflight boundary. The import
        // may write multiple primary batches and build one final checkpoint, so a
        // failure during this phase can leave completed batches visible.
        inner.append_import_batches(
            records.finish()?,
            ImportIndexBuildMode::DeferredFullCheckpoint,
            &context,
        )?;
        Ok(summary)
    }
}

fn reclaim_caches(inner: &mut Inner) -> bool {
    // Parsing a row and sorting its external ID may both need additional workspace
    // under the same connection lock. Give those operations a chance to evict
    // resident search caches while preserving the staged records and the input's
    // current position.
    Inner::release_optional_caches(
        &mut inner.text_index,
        &mut inner.persistent_vector_segments,
        &mut inner.lazy_vector_segment,
        std::iter::empty(),
    )
}
