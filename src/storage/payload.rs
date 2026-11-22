use super::*;
use crate::json::{
    JsonInputKind, JsonOperationContext, JsonReservation, JsonSchema, ReservedString, ReservedVec,
    WorkspaceReservation,
};
use crate::model::VectorIndexConfig;

pub(crate) fn copy_document_in(
    document: &Document,
    context: &crate::json::JsonOperationContext,
) -> Result<(Document, RecordReservation)> {
    // A caller receiving a document owns a complete metadata snapshot. Only the
    // catalog-preparation path may request the field-only form whose metadata is
    // attached from the committed record later.
    copy_document_fields_in(document, context, true)
}

fn copy_json_in(
    value: &JsonValue,
    context: &JsonOperationContext,
    include: bool,
) -> Result<(JsonValue, JsonReservation)> {
    // Full copies own their metadata independently of the catalog or input buffer.
    // Charge the cloned tree as stored JSON in the receiving operation so it can
    // outlive the source reservation.
    if include {
        Ok(value
            .try_clone_in(JsonInputKind::StoredJson, context)?
            .into_parts())
    } else {
        // A provisional catalog summary uses an empty object of the same JSON
        // type. Its real metadata remains with the input until the append supplies
        // the surviving record reference.
        Ok((JsonValue::object(), context.reservation()))
    }
}

fn copy_document_fields_in(
    document: &Document,
    context: &JsonOperationContext,
    include_metadata: bool,
) -> Result<(Document, RecordReservation)> {
    let mut work = RecordReservation::from(context.reservation());
    // Perform a possibly deep metadata copy before copying the individual strings.
    // A failure to admit this document releases its temporary allocations without
    // changing the input document.
    let (metadata, json) = copy_json_in(&document.metadata, context, include_metadata)?;
    work.merge(json);
    // Admit the document's strings together; their bytes belong to native
    // workspace while the metadata tree remains under JSON accounting. Keep both
    // types of charge with the returned document.
    let [document_id, collection_id, external_id, title, source_uri, created_at, updated_at] = work
        .copy_strings(
            [
                &document.document_id,
                &document.collection_id,
                &document.external_id,
                document.title.as_deref().unwrap_or_default(),
                document.source_uri.as_deref().unwrap_or_default(),
                &document.created_at,
                &document.updated_at,
            ],
            context,
        )?;
    Ok((
        Document {
            document_id,
            collection_id,
            external_id,
            // The temporary copy uses an empty string for an absent optional
            // field. Consult the source presence bit again so Some("") stays
            // distinguishable from None in the owned document.
            title: document.title.as_ref().map(|_| title),
            source_uri: document.source_uri.as_ref().map(|_| source_uri),
            metadata,
            created_at,
            updated_at,
        },
        work,
    ))
}

pub(crate) fn copy_collection_in(
    collection: &CollectionInfo,
    context: &crate::json::JsonOperationContext,
) -> Result<(CollectionInfo, RecordReservation)> {
    copy_collection_fields_in(collection, context, true)
}

fn copy_collection_fields_in(
    collection: &CollectionInfo,
    context: &JsonOperationContext,
    include_metadata: bool,
) -> Result<(CollectionInfo, RecordReservation)> {
    let mut work = RecordReservation::from(context.reservation());
    // Exact indexing has no provider or parameter fields to retain. For an ANN
    // definition the provider belongs to native storage, and the nested parameters
    // are charged as JSON.
    let (provider, params) = match &collection.vector_index {
        VectorIndexConfig::Exact => ("", JsonValue::Null),
        VectorIndexConfig::Ann { provider, params } => {
            // A provisional ANN definition still needs the right provider and
            // variant. Its parameter tree can remain with the prepared record and
            // move into the selected catalog definition after append.
            let (params, json) = copy_json_in(params, context, include_metadata)?;
            work.merge(json);
            (provider.as_str(), params)
        }
    };
    // Copy the provider while the source configuration is borrowed. The resulting
    // descriptor can outlive the original collection and its operation, including
    // when a later write replaces the catalog entry.
    let [collection_id, name, embedding_model, created_at, provider] = work.copy_strings(
        [
            &collection.collection_id,
            &collection.name,
            collection.embedding_model.as_deref().unwrap_or_default(),
            &collection.created_at,
            provider,
        ],
        context,
    )?;
    // Restore the index variant after copying the strings. The scratch
    // placeholders used to copy an exact descriptor do not become provider or
    // parameter fields in the result.
    let vector_index = match collection.vector_index {
        VectorIndexConfig::Exact => VectorIndexConfig::Exact,
        VectorIndexConfig::Ann { .. } => VectorIndexConfig::Ann { provider, params },
    };
    Ok((
        CollectionInfo {
            // The tokenizer is a typed setting in the copied descriptor. JSON
            // charging here applies to the arbitrary ANN parameter tree; it does
            // not require serializing every scalar collection field to build a
            // copy.
            tokenizer: collection.tokenizer.clone(),
            collection_id,
            name,
            embedding_dim: collection.embedding_dim,
            distance_metric: collection.distance_metric,
            // A collection with no configured model can later derive it from its
            // live embeddings. Preserve that absence when copying the descriptor
            // instead of binding it to an empty model name.
            embedding_model: collection.embedding_model.as_ref().map(|_| embedding_model),
            vectors_normalized: collection.vectors_normalized,
            vector_index,
            created_at,
        },
        work,
    ))
}

pub(crate) struct RetainedEmbedding {
    // Owns the IDs, model and coordinate vector of one embedding. Individual
    // buffers can move into a vector index entry or a query result.
    embedding: Embedding,
    // Workspace charge for owned strings and f32 capacity. Moving the IDs and
    // coordinates into the cache transfers only their share.
    work: WorkspaceReservation,
}

impl RetainedEmbedding {
    pub(crate) fn copy(embedding: &Embedding, context: &JsonOperationContext) -> Result<Self> {
        // Measure the components and strings by the amount the clone needs to
        // copy. The source may have reserved spare capacity for its own writes;
        // that spare space does not have to be transferred to this new owner.
        let bytes = embedding.vector.len() * std::mem::size_of::<f32>()
            + embedding.embedding_id.len()
            + embedding.chunk_id.len()
            + embedding.collection_id.len()
            + embedding.chunk_external_id.len()
            + embedding.model_name.len()
            + embedding.created_at.len();
        let mut work = context.workspace_reservation();
        // Reserve the vector elements and string bytes together before cloning.
        // They will all be live in the returned embedding, even if the source
        // embedding remains in the catalog.
        work.grow_to(bytes, bytes, "operation_scratch")?;
        let embedding = embedding.clone();
        // Compare the actual capacities with the admitted amount before returning.
        // Allocator rounding or a change in how a field is cloned must not leave a
        // larger retained allocation uncharged.
        if embedding_capacity(&embedding) > bytes {
            return Err(
                std::io::Error::other("embedding allocation exceeds requested capacity").into(),
            );
        }
        Ok(Self { embedding, work })
    }

    pub(crate) fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    // A caller that keeps the complete embedding also takes its workspace charge.
    // Access that needs only coordinates can instead release the identifying
    // strings while keeping the vector.
    pub(crate) fn into_parts(self) -> (Embedding, WorkspaceReservation) {
        (self.embedding, self.work)
    }

    pub(super) fn try_into_cache_entry(
        mut self,
        cache: &mut crate::json::CacheReservation,
        limit: usize,
    ) -> Option<VectorIndexEntry> {
        let bytes = self.embedding.chunk_id.capacity()
            + self.embedding.embedding_id.capacity()
            + self.embedding.vector.capacity() * std::mem::size_of::<f32>();
        // The cache retains only the two identifiers and coordinates. Transfer
        // their capacity before taking those buffers so no allocation becomes
        // unowned if cache admission fails.
        // A cache refusal consumes this optional conversion and drops the retained
        // embedding. Nothing has been moved into the cache, so callers can
        // continue with the primary or a streamed segment.
        if !cache.absorb_workspace(&mut self.work, bytes, limit) {
            return None;
        }
        // These moves leave empty buffers on the source. The result can reuse both
        // ID allocations and the original coordinate array without copying vector
        // data into cache a second time.
        let entry = VectorIndexEntry {
            chunk_id: std::mem::take(&mut self.embedding.chunk_id),
            embedding_id: std::mem::take(&mut self.embedding.embedding_id),
            vector: std::mem::take(&mut self.embedding.vector),
        };
        // The cache now owns the two IDs and the vector. Dropping the remainder
        // releases the collection ID, external alias, model, timestamp and their
        // residual workspace together.
        drop(self);
        Some(entry)
    }

    fn into_vector(self) -> ReservedVec<f32> {
        let Self {
            mut embedding,
            work,
        } = self;
        // Move the existing coordinate array without copying it, then free the
        // string fields before shrinking the retained charge to the capacity of
        // the returned vector.
        let vector = std::mem::take(&mut embedding.vector);
        drop(embedding);
        ReservedVec::from_reserved(vector, work)
    }
}

fn embedding_capacity(embedding: &Embedding) -> usize {
    // Use allocated capacities for retained values. Multiplying vector capacity by
    // the f32 width places coordinates and UTF-8 string buffers in the same byte
    // accounting.
    embedding.vector.capacity() * std::mem::size_of::<f32>()
        + embedding.embedding_id.capacity()
        + embedding.chunk_id.capacity()
        + embedding.collection_id.capacity()
        + embedding.chunk_external_id.capacity()
        + embedding.model_name.capacity()
        + embedding.created_at.capacity()
}

pub(crate) struct RetainedChunk {
    // Owned chunk. A summary can retain identity, ordering and the text hash while
    // omitting the text and full metadata.
    chunk: Chunk,
    // JSON credit for the retained metadata. It follows the metadata if the caller
    // takes the chunk apart.
    json: JsonReservation,
    // Workspace credit for the chunk's owned strings, including text when loaded.
    // Consumers can keep just the credit for the field they extract.
    work: WorkspaceReservation,
}

impl RetainedChunk {
    pub(crate) fn copy(chunk: &Chunk, context: &JsonOperationContext) -> Result<Self> {
        Self::copy_in(chunk, context, true)
    }

    fn copy_in(
        chunk: &Chunk,
        context: &JsonOperationContext,
        include_payload: bool,
    ) -> Result<Self> {
        // A full copy needs text and metadata for callers reading a chunk. Batch
        // preparation can keep a summary with its identity, digest and ordering
        // fields and leave the original payload in the staged record.
        let (copied_metadata, json) = copy_json_in(&chunk.metadata, context, include_payload)?;
        let bytes = chunk.chunk_id.len()
            + chunk.document_id.len()
            // A summary still needs the owning document, collection and external
            // ID to resolve relationships. Text dominates many payloads and is
            // made optional here independently of those keys.
            + chunk.collection_id.len()
            + chunk.chunk_external_id.len()
            + if include_payload { chunk.text.len() } else { 0 }
            + chunk.text_hash.len()
            + chunk.embedding_id.as_ref().map_or(0, String::len)
            + chunk.created_at.len();
        let mut work = context.workspace_reservation();
        let metadata = copied_metadata;
        // Native strings and stored JSON occupy different reservations. Admit the
        // string copies while the cloned metadata remains live so the two do not
        // compete for credit already returned prematurely.
        work.grow_to(bytes, bytes, "operation_scratch")?;
        let chunk = Chunk {
            chunk_id: chunk.chunk_id.clone(),
            document_id: chunk.document_id.clone(),
            collection_id: chunk.collection_id.clone(),
            chunk_external_id: chunk.chunk_external_id.clone(),
            text: if include_payload {
                chunk.text.clone()
            } else {
                String::new()
            },
            // Keep the text digest, original ordinal and token count when primary
            // text is omitted. They describe the stored row and must not be
            // recomputed from the empty summary string.
            text_hash: chunk.text_hash.clone(),
            ordinal: chunk.ordinal,
            token_count: chunk.token_count,
            metadata,
            embedding_id: chunk.embedding_id.clone(),
            source_span: chunk.source_span.clone(),
            created_at: chunk.created_at.clone(),
        };
        // Check the capacity of what will leave this call. A successful string
        // clone is not enough to admit the result if any retained allocation
        // exceeds the total reserved for these native fields.
        if chunk_capacity(&chunk) > bytes {
            return Err(
                std::io::Error::other("chunk allocation exceeds requested capacity").into(),
            );
        }
        Ok(Self { chunk, json, work })
    }

    // Keep the two charges together when handing an owned chunk to a record or
    // JSON conversion. A caller can transfer its metadata independently of the
    // native buffers it wants to keep.
    pub(crate) fn into_parts(self) -> (Chunk, RecordReservation) {
        (self.chunk, RecordReservation::new(self.json, self.work))
    }

    pub(crate) fn with_embedding_id(mut self, id: Option<&str>) -> Result<Self> {
        // The current catalog link can differ from the one stored with a ChunkPut.
        // Preserve an equal link in place and pay for a replacement only if the
        // actual internal embedding ID changed.
        if self.chunk.embedding_id.as_deref() != id {
            let previous = self.chunk.embedding_id.take();
            let bytes = previous.as_ref().map_or(0, String::capacity);
            // Free the superseded string before returning its workspace. This
            // function owns the whole chunk, so a failed copy of the replacement
            // drops the partial result and its remaining credit.
            drop(previous);
            self.work.retain(self.work.reserved_bytes() - bytes);
            // An absent new link leaves the chunk with no selected embedding. A
            // present link uses this chunk's existing operation context, so
            // copying an ID cannot bypass the operation budget.
            if let Some(id) = id {
                let (id, work) = ReservedString::copy(id, self.work.context())?.into_parts();
                self.work.merge(work);
                self.chunk.embedding_id = Some(id);
            }
        }
        Ok(self)
    }

    // Borrow the chunk for inspection while allowing the caller to extend only its
    // JSON reservation. Its text and other native strings remain owned by the
    // retained chunk throughout that borrow.
    pub(crate) fn parts(&mut self) -> (&Chunk, &mut JsonReservation) {
        (&self.chunk, &mut self.json)
    }

    fn into_text(self) -> ReservedString {
        let Self {
            mut chunk,
            json,
            work,
        } = self;
        // Separate the text before dropping the materialized chunk and its JSON
        // metadata. Returning a ReservedString trims the remaining native credit
        // to the buffer that a text scan actually holds.
        let text = std::mem::take(&mut chunk.text);
        drop(chunk);
        // Drop the metadata reservation after dropping the chunk that owned the
        // JSON tree. The workspace passed into the returned text owner can then
        // release all spare bytes attributable to the discarded native fields.
        drop(json);
        ReservedString::from_reserved(text, work)
    }
}

fn chunk_capacity(chunk: &Chunk) -> usize {
    // The chunk's metadata is accounted for as JSON, and its ordinal and source
    // coordinates are inline. Count just the owned native string buffers when
    // transferring or validating its workspace.
    chunk.chunk_id.capacity()
        + chunk.document_id.capacity()
        + chunk.collection_id.capacity()
        + chunk.chunk_external_id.capacity()
        + chunk.text.capacity()
        + chunk.text_hash.capacity()
        + chunk.embedding_id.as_ref().map_or(0, String::capacity)
        + chunk.created_at.capacity()
}

impl PayloadReader {
    // Clone the already opened file together with its codec. A new reader can
    // follow locators into the same file if its pathname changes after the catalog
    // was built.
    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            codec: self.codec,
            // A clone failure leaves the existing reader usable. Only the new
            // descriptor and the codec selection move into the proposed reader;
            // the locator still belongs to its catalog view.
            file: self.file.try_clone()?,
        })
    }

    #[cfg(test)]
    pub fn read_payload(&mut self, record_ref: &RecordRef) -> Result<RecordPayload> {
        self.read_payload_in(record_ref, &JsonOperationContext::standalone())
            .map(|(record, _)| record)
    }

    #[cfg(test)]
    pub fn read_payload_in(
        &mut self,
        record_ref: &RecordRef,
        context: &JsonOperationContext,
    ) -> Result<(RecordPayload, JsonReservation)> {
        if self.codec == FileCodec::Legacy {
            return self
                .read_legacy_retained_payload_in(record_ref, context)
                .map(|(record, work, _)| (record, work));
        }
        let record_type = record_type_from_name(record_ref.record_type)?;
        read_binary_payload_in(
            &mut self.file,
            record_type,
            &record_ref.payload,
            true,
            JsonInputKind::StoredJson,
            context,
            None,
        )
    }

    pub(crate) fn read_embedding_vector_in(
        &mut self,
        record_ref: &RecordRef,
        context: &JsonOperationContext,
    ) -> Result<ReservedVec<f32>> {
        // Reading just the vector still goes through the typed embedding decoder.
        // A valid payload checksum alone cannot show that a locator actually names
        // an EmbeddingPut.
        self.read_embedding_in(record_ref, context)
            .map(RetainedEmbedding::into_vector)
    }

    pub(crate) fn read_embedding_in(
        &mut self,
        record_ref: &RecordRef,
        context: &JsonOperationContext,
    ) -> Result<RetainedEmbedding> {
        // Start a native reservation for the materialized values. The binary
        // decoder can grow it directly, while the legacy path initially accounts
        // for the typed copy against its parsed JSON representation.
        let mut retained = context.workspace_reservation();
        let (payload, json, mut legacy_copies) =
            self.read_working_payload_in(record_ref, context, &mut retained)?;
        // Record decoding follows the locator's recorded type. Check the resulting
        // variant before extracting any coordinates so asking for an embedding
        // through a chunk locator fails as a format error.
        let embedding = match payload {
            RecordPayload::EmbeddingPut(embedding) => embedding,
            other => {
                // Release the wrongly typed decoded heap before constructing the
                // error. Native and JSON guards remain in scope until the failure
                // returns and retire the corresponding credit.
                drop(other);
                return Err(Error::new(
                    ErrorCode::FileFormatError,
                    "embedding locator did not contain an EmbeddingPut payload",
                ));
            }
        };
        // Embeddings contain no business metadata tree to return. Only owned
        // strings and the coordinate allocation need to stay charged after
        // decoding succeeds.
        drop(json);
        if let Some(copies) = &mut legacy_copies {
            // The legacy typed copy was admitted under its equivalent JSON layout.
            // Transfer the same charge, without releasing/reacquiring shared credit.
            retained = copies.take_workspace(embedding_capacity(&embedding));
        }
        // Release any legacy copy credit left after assigning the native
        // allocations to their new owner. The returned embedding can now outlive
        // the JSON decoder and its working input.
        drop(legacy_copies);
        // Move the completed row and its native reservation together. Row strings
        // and vector components may coexist with the next catalog lookup after
        // this decoder has returned.
        Ok(RetainedEmbedding {
            embedding,
            work: retained,
        })
    }

    pub(crate) fn read_chunk_text_in(
        &mut self,
        record_ref: &RecordRef,
        context: &JsonOperationContext,
    ) -> Result<ReservedString> {
        // A text-only caller can discard the other materialized fields after their
        // record checks run. The resulting string keeps its allocation credit when
        // the chunk and its metadata are dropped.
        self.read_chunk_in(record_ref, context)
            .map(RetainedChunk::into_text)
    }

    pub(crate) fn read_chunk_in(
        &mut self,
        record_ref: &RecordRef,
        context: &JsonOperationContext,
    ) -> Result<RetainedChunk> {
        let mut retained = context.workspace_reservation();
        let (payload, json, mut legacy_copies) =
            self.read_working_payload_in(record_ref, context, &mut retained)?;
        // Keep the format check on the materialization path for both file codecs.
        // The requested record may be perfectly decodable yet still be the wrong
        // kind for the catalog locator.
        let chunk = match payload {
            RecordPayload::ChunkPut(chunk) => chunk,
            other => {
                drop(other);
                return Err(Error::new(
                    ErrorCode::FileFormatError,
                    "chunk locator did not contain a ChunkPut payload",
                ));
            }
        };
        // For a legacy chunk, move just the native string credit out of the
        // typed-copy allowance. Preserve the separate returned reservation for
        // business metadata, which remains owned by the chunk.
        if let Some(copies) = &mut legacy_copies {
            retained = copies.take_workspace(chunk_capacity(&chunk));
        }
        drop(legacy_copies);
        // A chunk owns native identifiers and text as well as JSON metadata. Keep
        // both lifetimes in the result so callers can later retain only the text
        // or build a complete output object.
        Ok(RetainedChunk {
            chunk,
            json,
            work: retained,
        })
    }

    fn read_working_payload_in(
        &mut self,
        record_ref: &RecordRef,
        context: &JsonOperationContext,
        retained: &mut WorkspaceReservation,
    ) -> Result<(RecordPayload, JsonReservation, Option<JsonReservation>)> {
        // Legacy payloads start as JSON and can temporarily own both the parse
        // tree and a typed copy. Return the extra guard so the caller can transfer
        // the precise workspace for the requested row kind.
        if self.codec == FileCodec::Legacy {
            let (payload, json, copies) =
                self.read_legacy_retained_payload_in(record_ref, context)?;
            Ok((payload, json, Some(copies)))
        } else {
            // Binary fields can be decoded directly into their final native types.
            // Request full payload retention for this materialization and let the
            // decoder charge the supplied workspace as it admits strings and
            // vectors.
            let (payload, json) = read_binary_payload_in(
                &mut self.file,
                // Use the locator's recorded type to select the binary schema. A
                // chunk or embedding accessor checks the variant it actually
                // receives after the common decoder has checked the underlying
                // bytes.
                record_type_from_name(record_ref.record_type)?,
                &record_ref.payload,
                true,
                JsonInputKind::StoredJson,
                context,
                Some(retained),
            )?;
            Ok((payload, json, None))
        }
    }

    fn read_legacy_retained_payload_in(
        &mut self,
        record_ref: &RecordRef,
        context: &JsonOperationContext,
    ) -> Result<(RecordPayload, JsonReservation, JsonReservation)> {
        // Count the requested wire extent before allocating or reading it. A
        // request that ultimately fails its memory or integrity check still
        // represents a primary read requested by this operation.
        context.search_cost_add(
            crate::search_cost::Counter::PrimaryRequestedBytes,
            record_ref.payload.len,
        );
        // The file length lives in a 64-bit locator, while the legacy parser needs
        // a contiguous native buffer. Reject an unrepresentable length before
        // computing the allocation or truncating the read size.
        let len = usize::try_from(record_ref.payload.len).map_err(|_| {
            Error::new(
                ErrorCode::FileFormatError,
                "record payload length does not fit in memory",
            )
        })?;
        // Keep the input bytes charged through conversion to a typed record.
        // During parsing those bytes coexist with the new JSON tree, even though
        // the raw buffer is discarded before returning.
        let mut input_work = context.reservation();
        let mut payload = Vec::new();
        if let Err(error) = input_work.reserve_vec(&mut payload, len, JsonInputKind::StoredJson) {
            if crate::json::is_memory_budget_error(&error) {
                // A payload too large to retain can still have an invalid checksum.
                // Stream it through a fixed-size buffer before returning the
                // reservation error so a damaged record is reported as such even
                // under a tight memory budget.
                let mut source = PositionedPayload {
                    file: &mut self.file,
                    offset: record_ref.payload.offset,
                    remaining: record_ref.payload.len,
                    crc: Crc32c::default(),
                    context,
                };
                // Use the legacy checksum width for comparison with this locator.
                // The positional reader's own CRC is not the integrity value
                // stored for a legacy JSON payload.
                let mut checksum = crate::hash::Checksum64::default();
                let mut buffer = [0; 8192];
                // The refused allocation rules out parsing the contiguous legacy
                // JSON but does not prevent checking its entire recorded extent.
                // Keep the last read within the remaining length and reject
                // truncation through read_exact.
                while source.remaining != 0 {
                    let count = source.remaining.min(buffer.len() as u64) as usize;
                    source.read_exact(&mut buffer[..count])?;
                    checksum.update(&buffer[..count]);
                }
                // A mismatch found on the bounded read takes precedence over the
                // saved memory refusal. A successful checksum establishes
                // integrity only; the requested decoded record still did not fit.
                if checksum.finish() != record_ref.payload.checksum {
                    return Err(checksum_error(&record_ref.payload));
                }
            }
            return Err(error);
        }
        // Resize only after admitting the full payload. The positional reader then
        // consumes this exact locator independently of any current cursor in a
        // cloned file descriptor.
        payload.resize(len, 0);
        PositionedPayload {
            file: &mut self.file,
            offset: record_ref.payload.offset,
            remaining: record_ref.payload.len,
            crc: Crc32c::default(),
            context,
        }
        .read_exact(&mut payload)?;
        // The successful allocation path checks the same legacy digest as the
        // bounded fallback. Only a verified byte slice is handed to text decoding,
        // keeping the two paths consistent on a wrong checksum.
        if checksum64(&payload) != record_ref.payload.checksum {
            return Err(checksum_error(&record_ref.payload));
        }
        let text = std::str::from_utf8(&payload).map_err(|err| {
            Error::new(
                ErrorCode::FileFormatError,
                format!(
                    "record payload is not utf-8 at offset {}: {err}",
                    record_ref.payload.offset
                ),
            )
        })?;
        // Dispatch the parsed text using the type from its enclosing record
        // reference. The JSON bytes by themselves do not select the Rust payload
        // variant that catalog replay must reconstruct.
        decode_legacy_retained_payload_in(
            record_type_from_name(record_ref.record_type)?,
            text,
            context,
        )
    }
}

pub(super) fn decode_legacy_summary_in(
    record_type: u16,
    text: &str,
    context: &JsonOperationContext,
) -> Result<(RecordPayload, RecordReservation)> {
    // Legacy JSON is decoded to a complete typed record before producing a
    // summary. Strip the two large payload fields before measuring what will
    // actually survive in the catalog.
    let (mut record, mut json, mut copies) =
        decode_legacy_retained_payload_in(record_type, text, context)?;
    match &mut record {
        RecordPayload::ChunkPut(chunk) => chunk.text = String::new(),
        RecordPayload::EmbeddingPut(embedding) => embedding.vector = Vec::new(),
        _ => {}
    }
    // Move credit for the remaining native allocations only after the text or
    // coordinates have been released. Measuring the complete typed copy here would
    // overstate the memory retained by summary replay.
    let strings = copies.take_workspace(record_workspace_bytes(&record));
    if let RecordPayload::CollectionCreate(CollectionInfo {
        vector_index: VectorIndexConfig::Ann { params, .. },
        ..
    }) = &record
    {
        // ANN parameters remain an owned JSON tree inside the typed settings.
        // Reconcile their actual retained allocation and merge that charge with
        // the returned metadata credit.
        params.reconcile_retained_in(JsonInputKind::StoredJson, &mut copies)?;
        json.merge(copies);
    }
    // Return a single record guard carrying the retained JSON and native strings
    // from the summary. Credit for any discarded large fields does not have to
    // persist through catalog application.
    Ok((record, RecordReservation::new(json, strings)))
}

impl RecordPayload {
    pub(crate) fn metadata_bytes_in(&self, context: &JsonOperationContext) -> Result<usize> {
        // Measure JSON that must survive a committed write. Most record kinds
        // retain only native fields, while document and chunk metadata and ANN
        // parameters can transfer pre-existing JSON ownership.
        let value = match self {
            Self::DocumentPut(value) => Some(&value.metadata),
            Self::ChunkPut(value) => Some(&value.metadata),
            Self::CollectionCreate(CollectionInfo {
                vector_index: VectorIndexConfig::Ann { params, .. },
                ..
            }) => Some(params),
            _ => None,
        };
        // Count retained heap shape rather than encoded JSON length. Even an empty
        // object is a field to inspect, and a failure measuring the tree must
        // reach the caller instead of becoming zero.
        value
            .map(|value| value.retained_bytes_in(JsonInputKind::StoredJson, context))
            .transpose()
            .map(|bytes| bytes.unwrap_or(0))
    }

    // Metadata stays in the writer's record until commit, with its ownership
    // credit already transferred to the catalog transaction. Do not clone it.
    pub(crate) fn copy_catalog_fields_in(
        &self,
        context: &JsonOperationContext,
    ) -> Result<(Self, RecordReservation)> {
        match self {
            // A collection still needs its tokenizer and vector-index variant for
            // provisional replay. The helper can omit the ANN parameter tree at
            // this stage because the committed put supplies it during locator
            // finalization.
            Self::CollectionCreate(value) => {
                return copy_collection_fields_in(value, context, false)
                    .map(|(value, work)| (Self::CollectionCreate(value), work))
            }
            // A provisional document must still supply its stored external
            // identity, title and source fields. Its metadata tree remains in the
            // writer's full payload until a successful append attaches it to the
            // catalog.
            Self::DocumentPut(value) => {
                return copy_document_fields_in(value, context, false)
                    .map(|(value, work)| (Self::DocumentPut(value), work))
            }
            // Use the summary copy for a tentative chunk. Its external key,
            // ownership and position can be applied before append without copying
            // the large text or duplicating deferred metadata.
            Self::ChunkPut(value) => {
                return RetainedChunk::copy_in(value, context, false).map(|value| {
                    let (value, work) = value.into_parts();
                    (Self::ChunkPut(value), work)
                })
            }
            _ => {}
        }
        let mut work = RecordReservation::from(context.reservation());
        let copied = match self {
            // The catalog keeps the embedding identity, model and dimension. Its
            // record reference locates the vector on disk; copying components here
            // would keep every embedding resident during catalog construction.
            Self::EmbeddingPut(value) => Self::EmbeddingPut(Embedding {
                embedding_id: work.copy_string(&value.embedding_id, context)?,
                chunk_id: work.copy_string(&value.chunk_id, context)?,
                collection_id: work.copy_string(&value.collection_id, context)?,
                chunk_external_id: work.copy_string(&value.chunk_external_id, context)?,
                // Keep the model identity even when leaving the coordinate array
                // empty. Visible-model checks examine the catalog summary
                // independently of any later vector materialization.
                model_name: work.copy_string(&value.model_name, context)?,
                created_at: work.copy_string(&value.created_at, context)?,
                dim: value.dim,
                vector: Vec::new(),
            }),
            // A catalog drop needs the internal ID and time. It resolves a known
            // collection name from the existing definition, so the provisional
            // payload does not have to copy the input's display name or reason.
            Self::CollectionDrop {
                collection_id,
                dropped_at,
                ..
            } => Self::CollectionDrop {
                collection_id: work.copy_string(collection_id, context)?,
                dropped_at: work.copy_string(dropped_at, context)?,
                name: String::new(),
                reason: None,
            },
            // The catalog's deletion transition uses the target kind and internal
            // ID. The writer still owns the full tombstone, including the reason
            // and timestamp for the primary record.
            Self::Tombstone(value) => Self::Tombstone(Tombstone {
                target_id: work.copy_string(&value.target_id, context)?,
                target_type: work.copy_string(&value.target_type, context)?,
                deleted_at: String::new(),
                reason: None,
            }),
            Self::CompactMarker(value) => {
                // Each watermark survives compaction even if the record that
                // supplied the highest ID was removed. Charge the map nodes
                // separately from their owned prefix strings when copying the
                // marker.
                let bytes =
                    crate::json::btree_workspace::<String, u64>(value.id_high_watermarks.len());
                // Reserve the exact tree estimate before inserting any watermark
                // keys. The reservation for tree structure and every separate
                // prefix buffer must coexist throughout copying.
                let mut nodes = context.workspace_reservation();
                nodes.grow_to(bytes, bytes, "catalog_build")?;
                work.merge_workspace(nodes);
                // Allocate each watermark's key under the same reservation as the
                // new map nodes. The numeric values travel unchanged even for
                // prefixes whose highest-numbered row did not survive compaction.
                let mut watermarks = BTreeMap::new();
                for (prefix, watermark) in &value.id_high_watermarks {
                    watermarks.insert(work.copy_string(prefix, context)?, *watermark);
                }
                Self::CompactMarker(CompactMarker {
                    compact_generation: value.compact_generation,
                    source_last_commit_sequence_no: value.source_last_commit_sequence_no,
                    compacted_at: String::new(),
                    records_rewritten: value.records_rewritten,
                    id_high_watermarks: watermarks,
                    compacted_main_checksum: value.compacted_main_checksum,
                })
            }
            // The catalog does not apply a commit frame as a business record. Keep
            // its scalar coverage fields in the provisional value without
            // allocating another timestamp string.
            Self::Commit(value) => Self::Commit(CommitPayload {
                batch_id: value.batch_id,
                first_sequence_no: value.first_sequence_no,
                last_sequence_no: value.last_sequence_no,
                record_count: value.record_count,
                batch_checksum: value.batch_checksum,
                committed_at: String::new(),
            }),
            // The collection, document and chunk copy helpers returned directly
            // above. Reaching one of those variants here would mean a new generic
            // copy path had bypassed its type-specific ownership rules.
            Self::CollectionCreate(_) | Self::DocumentPut(_) | Self::ChunkPut(_) => unreachable!(),
        };
        Ok((copied, work))
    }
}

// This is the heap storage of native fields owned by one record. Use capacities
// because allocating a longer input and then truncating it can leave its
// allocation large. JSON objects and owning array slots have separate accounting.
pub(crate) fn record_workspace_bytes(record: &RecordPayload) -> usize {
    let optional = |value: &Option<String>| value.as_ref().map_or(0, String::capacity);
    match record {
        RecordPayload::CollectionCreate(value) => {
            value.collection_id.capacity()
                + value.name.capacity()
                + optional(&value.embedding_model)
                + value.created_at.capacity()
                // The ANN provider string is native workspace. Its parameter
                // object is handled by the JSON ledger, so the two values follow
                // different owners when catalog replay retains the configuration.
                + match &value.vector_index {
                    VectorIndexConfig::Exact => 0,
                    VectorIndexConfig::Ann { provider, .. } => provider.capacity(),
                }
        }
        RecordPayload::CollectionDrop {
            collection_id,
            name,
            dropped_at,
            reason,
        } => collection_id.capacity() + name.capacity() + dropped_at.capacity() + optional(reason),
        // A native reservation for a document covers identifiers, optional title
        // and URI, and both timestamps. Metadata remains structured and is charged
        // by its retained JSON owner.
        RecordPayload::DocumentPut(value) => {
            value.document_id.capacity()
                + value.collection_id.capacity()
                + value.external_id.capacity()
                + optional(&value.title)
                + optional(&value.source_uri)
                + value.created_at.capacity()
                + value.updated_at.capacity()
        }
        // Chunk and embedding capacity include the large payload, if the caller
        // still owns it. A catalog summary can run the same measure after taking
        // out the text or vector and get the size of what actually survives.
        RecordPayload::ChunkPut(value) => chunk_capacity(value),
        RecordPayload::EmbeddingPut(value) => embedding_capacity(value),
        RecordPayload::Tombstone(value) => {
            value.target_type.capacity()
                + value.target_id.capacity()
                + value.deleted_at.capacity()
                + optional(&value.reason)
        }
        RecordPayload::Commit(value) => value.committed_at.capacity(),
        // A compact marker keeps allocation history as an ordered map. Count the
        // storage of map entries and the strings they own, even when no business
        // row corresponds to those IDs.
        RecordPayload::CompactMarker(value) => {
            value.compacted_at.capacity()
                + crate::json::btree_workspace::<String, u64>(value.id_high_watermarks.len())
                + value
                    .id_high_watermarks
                    .keys()
                    .map(String::capacity)
                    .sum::<usize>()
        }
    }
}

fn decode_legacy_retained_payload_in(
    record_type: u16,
    text: &str,
    context: &JsonOperationContext,
) -> Result<(RecordPayload, JsonReservation, JsonReservation)> {
    // Legacy frames store their logical shape as JSON, so the parsed tree and the
    // typed record can coexist temporarily. Decode with the stored-record context
    // from the start so oversized or deeply nested input fails under the same
    // operation's bounds.
    let (mut value, mut work) =
        crate::json::parse_with_schema_in(text, JsonSchema::StoredRecord, context)?.into_parts();
    // Only document and chunk records carry the retained metadata tree moved by
    // this path. Taking it out of the parsed object lets the subsequent typed
    // conversion build the remaining fields without making a second copy of that
    // tree.
    let metadata = if matches!(record_type, DOCUMENT_PUT | CHUNK_PUT) {
        value
            .as_object_mut()
            .and_then(|object| object.get_mut("metadata"))
            .map(|metadata| {
                // Validate the taken value while it is still held by the parse
                // reservation. Replacing the source entry with an empty object
                // preserves the shape the typed constructor expects before the
                // original metadata is installed on the result.
                crate::json::ensure_object(std::mem::replace(metadata, JsonValue::object()))
            })
            .transpose()?
    } else {
        None
    };
    // Reuse parsed metadata. The remaining typed fields need only a bounded copy of their source.
    // Keep enough credit for the copies the typed constructor will create while
    // the source tree is alive. That copy guard is returned separately from the
    // reservation that tracks what remains of the parsed JSON.
    let copies = value.reserve_retained_in(JsonInputKind::StoredJson, context)?;
    // Call the typed constructor while the input tree remains available for its
    // ordinary field lookups. The replaced metadata entry is only a placeholder.
    // Once installed in the typed result, the moved object keeps its original
    // allocation.
    let mut record = record_from_json(record_type, &value)?;
    if let Some(metadata) = metadata {
        match &mut record {
            RecordPayload::DocumentPut(document) => document.metadata = metadata,
            RecordPayload::ChunkPut(chunk) => chunk.metadata = metadata,
            _ => unreachable!(),
        }
    }
    // Drop the input tree before shrinking its reservation. The document or chunk
    // may still own the metadata that was moved out of it. Other records in this
    // path retain no part of that tree.
    drop(value);
    match &record {
        RecordPayload::DocumentPut(document) => document
            .metadata
            // Reconcile against the actual retained object after discarding the
            // rest of the parse tree. The object can have spare allocations that a
            // serialized byte count would miss.
            .reconcile_retained_in(JsonInputKind::StoredJson, &mut work)?,
        RecordPayload::ChunkPut(chunk) => chunk
            .metadata
            .reconcile_retained_in(JsonInputKind::StoredJson, &mut work)?,
        _ => work.retain(0),
    }
    Ok((record, work, copies))
}

pub(super) fn read_staged_summary_in(
    file: &mut File,
    record_type: u16,
    payload: &PayloadRef,
    context: &JsonOperationContext,
    strings: &mut WorkspaceReservation,
) -> Result<(RecordPayload, JsonReservation)> {
    // A staging file is read with the import-facing classification of the original
    // input. A size or depth failure therefore identifies the right document,
    // chunk or collection unit instead of looking like an unrelated failure on an
    // ordinary database read.
    let kind = match record_type {
        COLLECTION_CREATE => JsonInputKind::CollectionConfig,
        DOCUMENT_PUT => JsonInputKind::DocumentMetadata,
        CHUNK_PUT => JsonInputKind::ChunkMetadata,
        _ => JsonInputKind::JsonlRecord,
    };
    // Keep the caller's accumulated string reservation intact when decoding this
    // row fails. Successful summaries add their own storage to the same
    // reservation and return the JSON credit for the decoded record.
    let previous = strings.reserved_bytes();
    let result = read_binary_payload_in(
        file,
        record_type,
        payload,
        // Prepared replays require summaries and their identity fields. Validate
        // the complete staged extent while leaving text and coordinates available
        // in the staging file for the subsequent append.
        false,
        kind,
        context,
        Some(&mut *strings),
    );
    // Roll back only the credit obtained in this call. The caller can reuse its
    // reservation to stage further summaries, and earlier successfully retained
    // rows may still depend on its starting balance.
    if result.is_err() {
        strings.retain(previous);
    }
    result
}

pub(super) fn read_binary_payload_in(
    file: &mut File,
    record_type: u16,
    payload: &PayloadRef,
    retain_payloads: bool,
    kind: JsonInputKind,
    context: &JsonOperationContext,
    mut strings: Option<&mut WorkspaceReservation>,
) -> Result<(RecordPayload, JsonReservation)> {
    // Count the logical request once. A memory retry below can make a second
    // physical pass through the same locator. It still represents one requested
    // payload to the query that issued this read.
    context.search_cost_add(
        crate::search_cost::Counter::PrimaryRequestedBytes,
        payload.len,
    );
    // Optional read-ahead is admitted by record kind. Save an initial admission
    // error for comparison after the checksum pass, even though the native decoder
    // will not be called for that failed attempt.
    let (mut input_work, mut input_error) =
        match PayloadInputReservation::new(context, record_type, payload.len, kind) {
            Ok(work) => (Some(work), None),
            Err(error) => (None, Some(error)),
        };
    loop {
        let previous_strings = strings.as_ref().map_or(0, |work| work.reserved_bytes());
        // Restart every attempt at the original offset with a new CRC. The first
        // attempt may have consumed part or all of this frame before it
        // relinquished optional buffer space.
        let source = PositionedPayload {
            file: &mut *file,
            offset: payload.offset,
            remaining: payload.len,
            crc: Crc32c::default(),
            context,
        };
        let mut source = BufReader::with_capacity(
            input_work
                .as_ref()
                .map_or(0, PayloadInputReservation::capacity),
            source,
        );
        // Even if acquiring input workspace failed, read the complete range through
        // the checksum reader below. Reporting a resource limit first would hide
        // corruption in a payload the caller was trying to verify.
        let decoded = match input_error.take() {
            None => codec::decode_reader_with_kind_in(
                record_type,
                &mut source,
                payload.len,
                retain_payloads,
                kind,
                context,
                // Use the staging allocation phase for original import fields and
                // the normal scratch phase for stored JSON. Both can send native
                // string allocations to the caller's already shared reservation.
                strings.as_deref_mut().map(|work| {
                    let phase = if kind == JsonInputKind::StoredJson {
                        "operation_scratch"
                    } else {
                        "batch_prepare"
                    };
                    (work, phase)
                }),
            ),
            Some(error) => Err(error),
        };
        // Read, rather than fill_buf, also drains a zero-capacity BufReader.
        std::io::copy(&mut source, &mut std::io::sink())?;
        let source = source.into_inner();
        // Give incomplete or corrupt stored bytes precedence over the saved decode
        // error. The checksum covers the complete physical payload, including bytes
        // the decoder did not retain.
        if source.remaining != 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        // The checksum is read from the locator, whose common representation can
        // also describe older formats. This binary pass computes CRC32C and widens
        // its result for that comparison.
        if u64::from(source.crc.finish()) != payload.checksum {
            return Err(checksum_error(payload));
        }
        // Retry only a memory-budget error that actually released optional
        // buffering. A parser error, checksum mismatch or failed retry with no
        // buffer left has no alternate decode path.
        if decoded
            .as_ref()
            .is_err_and(crate::json::is_memory_budget_error)
            && input_work
                .as_mut()
                .is_some_and(PayloadInputReservation::disable_optional_buffering)
        {
            // Required decoded strings and JSON take precedence over input buffering.
            // Restore the string reservation to its level before this attempt. The
            // decoded value will be rebuilt from the start, and its temporary
            // string allocations cannot retain credit on the next pass.
            if let Some(work) = &mut strings {
                work.retain(previous_strings);
            }
            continue;
        }
        return decoded;
    }
}

// The diagnostic offset addresses the payload, which is the separately checked
// byte span. It can differ from the offset of the surrounding record header.
fn checksum_error(payload: &PayloadRef) -> Error {
    Error::new(
        ErrorCode::ChecksumMismatch,
        format!(
            "record payload checksum mismatch at offset {}",
            payload.offset
        ),
    )
}

struct PositionedPayload<'a> {
    // Borrowed data file shared with the sequential scanner. Positional reads let
    // this decoder stream a payload without consuming the scanner's place.
    file: &'a mut File,
    // Absolute byte position of the next read in the data file. It advances by the
    // number actually read, including short reads.
    offset: u64,
    // Bytes remaining inside this payload's declared extent. Zero presents EOF to
    // the decoder at the boundary with the next frame.
    remaining: u64,
    // Running CRC32C of bytes returned from this payload, checked by the caller
    // after decoding.
    crc: Crc32c,
    // Operation that receives primary-file read call and byte counters, shared by
    // the Unix and fallback paths.
    context: &'a JsonOperationContext,
}

impl Read for PositionedPayload<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        // Clamp every call to the remaining declared extent before borrowing the
        // caller's buffer. A zero result at exactly that boundary is intentional.
        // A zero result with bytes still remaining must be treated as truncation
        // by the enclosing reader.
        let len = buffer
            .len()
            .min(self.remaining.min(usize::MAX as u64) as usize);
        if len == 0 {
            return Ok(0);
        }
        #[cfg(unix)]
        // A Unix positional read operates on the file descriptor without moving
        // its cursor. Count the physical attempt just before reading, and account
        // for the accepted byte count independently of the requested span.
        let count = {
            use std::os::unix::fs::FileExt;
            self.context
                .search_cost_add(crate::search_cost::Counter::PrimaryReadCalls, 1);
            let count = self.file.read_at(&mut buffer[..len], self.offset)?;
            self.context
                .search_cost_add(crate::search_cost::Counter::PrimaryReadBytes, count as u64);
            count
        };
        #[cfg(not(unix))]
        let count = read_at_fallback_counted(
            self.file,
            self.offset,
            &mut buffer[..len],
            Some(self.context),
        )?;
        // Advance the position and checksum by the number actually read. A short
        // read can be followed by another call at this position, so counting the
        // requested bytes would skip data and include bytes the file never
        // supplied.
        self.offset = self.offset.checked_add(count as u64).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "payload offset overflow")
        })?;
        self.remaining -= count as u64;
        self.crc.update(&buffer[..count]);
        Ok(count)
    }
}

#[cfg(test)]
fn read_at_fallback(
    source: &mut (impl Read + Seek),
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<usize> {
    read_at_fallback_counted(source, offset, bytes, None)
}

#[cfg(any(not(unix), test))]
fn read_at_fallback_counted(
    source: &mut (impl Read + Seek),
    offset: u64,
    bytes: &mut [u8],
    context: Option<&JsonOperationContext>,
) -> std::io::Result<usize> {
    // The fallback borrows a mutable cursor to emulate a positional read. Capture
    // the original position before seeking, so the sequential consumer can resume
    // from the same boundary even if the payload read reached a different
    // position.
    let position = source.stream_position()?;
    let result = source.seek(SeekFrom::Start(offset)).and_then(|_| {
        if let Some(context) = context {
            context.search_cost_add(crate::search_cost::Counter::PrimaryReadCalls, 1);
        }
        // Count the physical read only after the seek succeeds. The seek and
        // cursor restoration can both fail, but neither operation by itself
        // transfers payload bytes.
        let count = source.read(bytes)?;
        if let Some(context) = context {
            context.search_cost_add(crate::search_cost::Counter::PrimaryReadBytes, count as u64);
        }
        Ok(count)
    });
    // Attempt to restore the original cursor even if the seek or read failed. The
    // original error takes priority, and an otherwise successful read must still
    // report a failed restoration.
    // Preserve the caller's cursor even if the seek to the requested extent
    // succeeded and the read failed. Returning the read outcome before attempting
    // restoration would leave sequential parsing at an unrelated offset.
    let restored = source.seek(SeekFrom::Start(position));
    let count = result?;
    restored?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    #[test]
    fn fallback_positional_read_restores_cursor_on_success_eof_and_failure() {
        use std::io::{Cursor, Read, Seek, SeekFrom};
        struct Source {
            input: Cursor<Vec<u8>>,
            fail_read: bool,
            fail_seek: Option<u64>,
        }
        impl Read for Source {
            fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
                if self.fail_read {
                    return Err(std::io::ErrorKind::PermissionDenied.into());
                }
                self.input.read(bytes)
            }
        }
        impl Seek for Source {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                if matches!(position, SeekFrom::Start(value) if self.fail_seek == Some(value)) {
                    return Err(std::io::ErrorKind::Other.into());
                }
                self.input.seek(position)
            }
        }
        for seed in 0..100u64 {
            let mut source = Source {
                input: Cursor::new((0..100u8).collect()),
                fail_read: false,
                fail_seek: None,
            };
            source.input.set_position(200);
            let mut byte = [0];
            assert_eq!(
                super::read_at_fallback(&mut source, seed, &mut byte).unwrap(),
                1
            );
            assert_eq!(byte, [seed as u8]);
            assert_eq!(source.input.position(), 200);
            assert_eq!(
                super::read_at_fallback(&mut source, 100, &mut byte).unwrap(),
                0
            );
            assert_eq!(source.input.position(), 200);
            source.fail_read = true;
            assert_eq!(
                super::read_at_fallback(&mut source, seed, &mut byte)
                    .unwrap_err()
                    .kind(),
                std::io::ErrorKind::PermissionDenied
            );
            assert_eq!(source.input.position(), 200);
            source.fail_read = false;
            source.fail_seek = Some(seed);
            assert!(super::read_at_fallback(&mut source, seed, &mut byte).is_err());
            assert_eq!(source.input.position(), 200);
            source.fail_seek = Some(200);
            let trace = crate::search_cost::SearchCostTrace::new();
            let context = crate::json::JsonOperationContext::standalone().with_search_cost(&trace);
            assert!(
                super::read_at_fallback_counted(&mut source, seed, &mut byte, Some(&context))
                    .is_err()
            );
            assert_eq!(
                trace.total(crate::search_cost::Counter::PrimaryReadBytes),
                1
            );
            assert_eq!(
                trace.total(crate::search_cost::Counter::PrimaryReadCalls),
                1
            );
        }
    }

    use super::*;
    use crate::json::JsonProcessingPolicyV1;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fixture(
        codec: FileCodec,
        record_type: &'static str,
        payload: &[u8],
    ) -> (PathBuf, PayloadReader, RecordRef) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "cofferdb-payload-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&[0xa5; 19]).unwrap();
        file.write_all(payload).unwrap();
        file.write_all(&[0x5a; 31]).unwrap();
        file.flush().unwrap();
        let record = RecordRef {
            offset: 0,
            record_type,
            batch_id: 1,
            sequence_no: 1,
            payload: PayloadRef {
                offset: 19,
                len: payload.len() as u64,
                checksum: match codec {
                    FileCodec::Binary => u64::from(checksum32(payload)),
                    FileCodec::Legacy => checksum64(payload),
                },
            },
        };
        let file = ManagedFile::open_in(&JsonOperationContext::standalone(), || Ok(file)).unwrap();
        (path, PayloadReader { codec, file }, record)
    }

    fn document(metadata: JsonValue) -> Document {
        Document {
            document_id: "doc_1".into(),
            collection_id: "col_1".into(),
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata,
            created_at: "1970-01-01T00:00:00Z".into(),
            updated_at: "1970-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn copied_string_groups_preserve_optional_empty_fields_and_metadata_selection() {
        let context = JsonOperationContext::standalone();
        let baseline = context.available_workspace_bytes();
        for (title, uri) in [
            (None, Some("")),
            (Some(""), None),
            (Some("title"), Some("uri")),
        ] {
            let mut source = document(crate::json::parse(r#"{"source":"original"}"#).unwrap());
            source.title = title.map(str::to_owned);
            source.source_uri = uri.map(str::to_owned);
            for include in [false, true] {
                let (copied, work) = copy_document_fields_in(&source, &context, include).unwrap();
                let mut expected = source.clone();
                if !include {
                    expected.metadata = JsonValue::object();
                }
                assert_eq!(copied.to_json(), expected.to_json());
                let record = RecordPayload::DocumentPut(copied);
                assert_eq!(
                    work.workspace_bytes(),
                    super::super::record_workspace_bytes(&record)
                );
                drop((record, work));
                assert_eq!(context.available_workspace_bytes(), baseline);

                for ann in [false, true] {
                    let collection = CollectionInfo {
                        tokenizer: Default::default(),
                        collection_id: "col_1".into(),
                        name: "notes".into(),
                        embedding_dim: 2,
                        distance_metric: crate::model::DistanceMetric::Cosine,
                        embedding_model: title.map(str::to_owned),
                        vectors_normalized: false,
                        vector_index: if ann {
                            VectorIndexConfig::Ann {
                                provider: "provider".into(),
                                params: source.metadata.clone(),
                            }
                        } else {
                            VectorIndexConfig::Exact
                        },
                        created_at: source.created_at.clone(),
                    };
                    let (copied, work) =
                        copy_collection_fields_in(&collection, &context, include).unwrap();
                    let mut expected = collection.clone();
                    if !include {
                        if let VectorIndexConfig::Ann { params, .. } = &mut expected.vector_index {
                            *params = JsonValue::object();
                        }
                    }
                    assert_eq!(copied.to_json(), expected.to_json());
                    let record = RecordPayload::CollectionCreate(copied);
                    assert_eq!(
                        work.workspace_bytes(),
                        super::super::record_workspace_bytes(&record)
                    );
                    drop((record, work));
                    assert_eq!(context.available_workspace_bytes(), baseline);
                }
            }
        }
    }

    #[test]
    fn copied_embeddings_reserve_live_content_not_unused_source_capacity() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let mut source = Embedding {
                embedding_id: "emb_1".into(),
                chunk_id: "chk_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "external".repeat(seed + 1),
                model_name: "model".into(),
                dim: seed + 1,
                vector: vec![0.5; seed + 1],
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            source.vector.reserve(1024);
            source.model_name.reserve(1024);
            let minimum = source.vector.len() * 4
                + source.embedding_id.len()
                + source.chunk_id.len()
                + source.collection_id.len()
                + source.chunk_external_id.len()
                + source.model_name.len()
                + source.created_at.len();
            assert!(embedding_capacity(&source) > minimum);
            let context = JsonOperationContext::standalone();
            let mut held = context.workspace_reservation();
            held.grow_to(
                limit - minimum + 1,
                limit - minimum + 1,
                "operation_scratch",
            )
            .unwrap();
            assert_eq!(
                RetainedEmbedding::copy(&source, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::ResourceExhausted
            );
            held.retain(limit - minimum);
            let copied = RetainedEmbedding::copy(&source, &context).unwrap();
            assert_eq!(copied.embedding().to_json(), source.to_json());
            let (value, work) = copied.into_parts();
            assert_eq!(work.reserved_bytes(), minimum);
            assert!(held
                .grow_to(
                    limit - minimum + 1,
                    limit - minimum + 1,
                    "operation_scratch"
                )
                .is_err());
            drop(value);
            drop(work);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
    }

    #[test]
    fn full_chunks_keep_metadata_and_owned_strings_charged_until_consumed() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let text = "text \u{4e2d}\n\"".repeat(20_000 + seed);
            let mut source = Chunk {
                chunk_id: "chk_0000000000000001".into(),
                document_id: "doc_0000000000000001".into(),
                collection_id: "col_0000000000000001".into(),
                chunk_external_id: "external".repeat(seed + 1),
                text_hash: crate::model::chunk_text_hash(&text),
                text,
                ordinal: seed,
                token_count: 1,
                metadata: object_from_pairs([("payload", JsonValue::String("x".repeat(1024)))]),
                source_span: Some(crate::SourceSpan {
                    start: Some(0),
                    end: Some(10),
                    line_start: Some(1),
                    line_end: Some(2),
                }),
                embedding_id: Some("emb_1".into()),
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            source.text.reserve(4096);
            for codec in [None, Some(FileCodec::Binary), Some(FileCodec::Legacy)] {
                let fixture = codec.map(|codec| {
                    let bytes = if codec == FileCodec::Binary {
                        codec::encode(&RecordPayload::ChunkPut(source.clone())).unwrap()
                    } else {
                        source.to_json().to_compact_string().into_bytes()
                    };
                    fixture(codec, "ChunkPut", &bytes)
                });
                let mut fixture = fixture;
                let context = JsonOperationContext::standalone();
                let mut read = || match &mut fixture {
                    Some((_, reader, record)) => reader.read_chunk_in(record, &context),
                    None => RetainedChunk::copy(&source, &context),
                };
                let mut held = context.workspace_reservation();
                held.grow_to(limit - 128 * 1024, limit - 128 * 1024, "operation_scratch")
                    .unwrap();
                assert_eq!(
                    read().err().unwrap().code(),
                    if codec == Some(FileCodec::Legacy) {
                        ErrorCode::RecoveryError
                    } else {
                        ErrorCode::ResourceExhausted
                    }
                );
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(held);
                let retained = read().unwrap();
                let mut expected = source.clone();
                if codec == Some(FileCodec::Binary) {
                    expected.embedding_id = None;
                }
                let actual = retained.chunk.to_json();
                for (key, value) in expected.to_json().as_object().unwrap() {
                    assert!(actual.get(key) == Some(value), "chunk field {key} differs");
                }
                assert_eq!(
                    retained.work.reserved_bytes(),
                    chunk_capacity(&retained.chunk)
                );
                assert!(retained.json.reserved_bytes() > 0);
                let bytes = retained.work.reserved_bytes() + retained.json.reserved_bytes();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                    .unwrap();
                assert!(held
                    .grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                    .is_err());
                if seed % 2 == 0 {
                    let (chunk, work) = retained.into_parts();
                    assert!(held.grow_to(limit, limit, "operation_scratch").is_err());
                    drop(chunk);
                    drop(work);
                } else {
                    drop(retained);
                }
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(held);
                let retained = read().unwrap();
                let bytes = retained.work.reserved_bytes() + retained.json.reserved_bytes();
                let text_pointer = retained.chunk.text.as_ptr();
                let (chunk, mut work) = retained.into_parts();
                let extra = if seed % 2 == 0 { 16 * 1024 } else { 0 };
                let mut held = context.workspace_reservation();
                held.grow_to(
                    limit - bytes - extra,
                    limit - bytes - extra,
                    "operation_scratch",
                )
                .unwrap();
                let value = chunk.into_json_retained(&mut work);
                if extra == 0 {
                    assert_eq!(value.err().unwrap().code(), ErrorCode::RecoveryError);
                } else {
                    let value = value.unwrap();
                    assert_eq!(
                        value.get("text").unwrap().as_str().unwrap().as_ptr(),
                        text_pointer
                    );
                    assert!(value.get("text").unwrap().as_str().unwrap() == source.text);
                    assert_eq!(
                        value.get("source_span"),
                        expected.to_json().get("source_span")
                    );
                    drop(value);
                }
                drop(work);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(held);
                let replacement = "embedding-id-".repeat(10 + seed);
                let retained = read().unwrap();
                let bytes = retained.work.reserved_bytes() + retained.json.reserved_bytes();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                    .unwrap();
                assert_eq!(
                    retained
                        .with_embedding_id(Some(&replacement))
                        .err()
                        .unwrap()
                        .code(),
                    ErrorCode::ResourceExhausted
                );
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(held);
                let id = (seed % 2 == 0).then_some(replacement.as_str());
                let retained = read().unwrap().with_embedding_id(id).unwrap();
                assert_eq!(retained.chunk.embedding_id.as_deref(), id);
                assert_eq!(
                    retained.work.reserved_bytes(),
                    chunk_capacity(&retained.chunk)
                );
                let bytes = retained.work.reserved_bytes() + retained.json.reserved_bytes();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                    .unwrap();
                let retained = retained.with_embedding_id(id).unwrap();
                drop(retained);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                if let Some((path, reader, _)) = fixture {
                    drop(reader);
                    fs::remove_file(path).unwrap();
                }
            }
        }
    }

    #[test]
    fn scanned_text_retains_only_its_buffer_and_preserves_source_errors() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let text = "text \u{4e2d} \n\"".repeat(16384 + seed);
            let chunk = Chunk {
                chunk_id: "chk_1".into(),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "external".repeat(seed + 1),
                text_hash: crate::model::chunk_text_hash(&text),
                text: text.clone(),
                ordinal: 0,
                token_count: 1,
                metadata: object_from_pairs([("payload", JsonValue::String("x".repeat(1024)))]),
                source_span: None,
                embedding_id: None,
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            for codec in [FileCodec::Binary, FileCodec::Legacy] {
                let bytes = if codec == FileCodec::Binary {
                    codec::encode(&RecordPayload::ChunkPut(chunk.clone())).unwrap()
                } else {
                    chunk.to_json().to_compact_string().into_bytes()
                };
                let (path, mut reader, record) = fixture(codec, "ChunkPut", &bytes);
                let context = JsonOperationContext::standalone();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - 128 * 1024, limit - 128 * 1024, "operation_scratch")
                    .unwrap();
                let error = reader.read_chunk_text_in(&record, &context).err().unwrap();
                assert_eq!(
                    error.code(),
                    if codec == FileCodec::Binary {
                        ErrorCode::ResourceExhausted
                    } else {
                        ErrorCode::RecoveryError
                    }
                );
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(held);
                if codec == FileCodec::Binary {
                    // The optional 64 KiB input buffer must yield to the live text.
                    let available = text.len() + 16384;
                    let mut held = context.workspace_reservation();
                    held.grow_to(limit - available, limit - available, "operation_scratch")
                        .unwrap();
                    let retained = reader.read_chunk_text_in(&record, &context).unwrap();
                    assert_eq!(retained.as_ref(), text);
                    drop(retained);
                    held.grow_to(limit, limit, "operation_scratch").unwrap();
                }
                let retained = reader.read_chunk_text_in(&record, &context).unwrap();
                assert_eq!(retained.as_ref(), text);
                let mut rest = context.workspace_reservation();
                rest.grow_to(limit - text.len(), limit - text.len(), "operation_scratch")
                    .unwrap();
                assert!(rest
                    .grow_to(
                        limit - text.len() + 1,
                        limit - text.len() + 1,
                        "operation_scratch"
                    )
                    .is_err());
                drop(retained);
                rest.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(rest);
                reader
                    .file
                    .seek(SeekFrom::Start(
                        record.payload.offset + record.payload.len - 1,
                    ))
                    .unwrap();
                reader
                    .file
                    .write_all(&[bytes[bytes.len() - 1] ^ 0x80])
                    .unwrap();
                assert_eq!(
                    reader
                        .read_chunk_text_in(&record, &context)
                        .err()
                        .unwrap()
                        .code(),
                    ErrorCode::ChecksumMismatch
                );
                reader
                    .file
                    .set_len(record.payload.offset + record.payload.len - 1)
                    .unwrap();
                assert_eq!(
                    reader
                        .read_chunk_text_in(&record, &context)
                        .err()
                        .unwrap()
                        .code(),
                    ErrorCode::IoError
                );
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
                drop(reader);
                fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn cache_materialization_keeps_codec_budget_and_corruption_boundaries() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..32usize {
            let embedding = Embedding {
                embedding_id: catalog_id_string("emb_", 1),
                chunk_id: catalog_id_string("chk_", 1),
                collection_id: catalog_id_string("col_", 1),
                chunk_external_id: "unused alias".repeat(seed + 1),
                model_name: "model".repeat(100 + seed * 129),
                dim: seed + 1,
                vector: vec![0.25; seed + 1],
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let collection = CollectionInfo {
                collection_id: embedding.collection_id.clone(),
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: embedding.dim,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                created_at: embedding.created_at.clone(),
            };
            for codec in [FileCodec::Binary, FileCodec::Legacy] {
                let bytes = if codec == FileCodec::Binary {
                    codec::encode(&RecordPayload::EmbeddingPut(embedding.clone())).unwrap()
                } else {
                    embedding.to_json().to_compact_string().into_bytes()
                };
                let (path, reader, reference) = fixture(codec, "EmbeddingPut", &bytes);
                for corruption in 0..3 {
                    let context = JsonOperationContext::standalone();
                    let mut blocker = context.workspace_reservation();
                    blocker
                        .grow_to(limit - 64, limit - 64, "operation_scratch")
                        .unwrap();
                    let mut reference = reference.clone();
                    if corruption == 1 {
                        reference.payload.checksum ^= 1;
                    }
                    if corruption == 2 {
                        reader
                            .file
                            .set_len(reference.payload.offset + reference.payload.len - 1)
                            .unwrap();
                    }
                    let mut source = VectorRecordSegment::new(
                        &collection,
                        1,
                        [(&embedding, Some(&reference))],
                        reader.try_clone().unwrap(),
                        &context,
                    )
                    .unwrap();
                    let result = source.try_materialize(limit);
                    if corruption != 0 {
                        assert_eq!(
                            result.unwrap_err().code(),
                            if corruption == 1 {
                                ErrorCode::ChecksumMismatch
                            } else {
                                ErrorCode::IoError
                            }
                        );
                    } else {
                        assert!(result.unwrap().is_none());
                        drop(blocker);
                        let cached = source.try_materialize(limit).unwrap().unwrap();
                        assert_eq!(cached.vectors.len(), 1);
                        assert_eq!(cached.vectors[0].vector, embedding.vector);
                        drop(cached);
                        blocker = context.workspace_reservation();
                    }
                    drop(source);
                    drop(blocker);
                    context
                        .workspace_reservation()
                        .grow_to(limit, limit, "operation_scratch")
                        .unwrap();
                }
                drop(reader);
                fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn embedding_cache_transfer_moves_only_retained_fields_for_both_codecs() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let embedding = Embedding {
                embedding_id: "emb_1".into(),
                chunk_id: "chk_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "external".repeat(seed + 1),
                model_name: "model".repeat(seed + 1),
                dim: seed + 2,
                vector: vec![0.25; seed + 2],
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            for codec in [FileCodec::Binary, FileCodec::Legacy] {
                let bytes = if codec == FileCodec::Binary {
                    codec::encode(&RecordPayload::EmbeddingPut(embedding.clone())).unwrap()
                } else {
                    embedding.to_json().to_compact_string().into_bytes()
                };
                let (path, mut reader, record) = fixture(codec, "EmbeddingPut", &bytes);
                for shortage in [true, false] {
                    let context = JsonOperationContext::standalone();
                    let retained = reader.read_embedding_in(&record, &context).unwrap();
                    let value = retained.embedding();
                    let pointers = (
                        value.chunk_id.as_ptr(),
                        value.embedding_id.as_ptr(),
                        value.vector.as_ptr(),
                    );
                    let bytes = value.chunk_id.capacity()
                        + value.embedding_id.capacity()
                        + value.vector.capacity() * 4;
                    let mut cache = context.cache_reservation(0, bytes).unwrap();
                    let entry =
                        retained.try_into_cache_entry(&mut cache, bytes - usize::from(shortage));
                    assert_eq!(entry.is_none(), shortage);
                    if let Some(entry) = &entry {
                        assert_eq!(
                            (
                                entry.chunk_id.as_ptr(),
                                entry.embedding_id.as_ptr(),
                                entry.vector.as_ptr()
                            ),
                            pointers
                        );
                        assert_eq!(entry.vector, embedding.vector);
                        assert_eq!(cache.reserved_bytes(), bytes);
                    } else {
                        assert_eq!(cache.reserved_bytes(), 0);
                    }
                    let mut all = context.workspace_reservation();
                    all.grow_to(limit, limit, "operation_scratch").unwrap();
                    drop(entry);
                    drop(cache);
                    drop(all);
                    assert!(context
                        .cache_reservation((3 * 1024 * 1024 * 1024usize) as u128, usize::MAX)
                        .is_some());
                }
                drop(reader);
                fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn full_embedding_reads_retain_strings_and_vector_until_owner_is_dropped() {
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            for codec in [FileCodec::Binary, FileCodec::Legacy] {
                let embedding = Embedding {
                    embedding_id: "emb_1".into(),
                    chunk_id: "chk_1".into(),
                    collection_id: "col_1".into(),
                    chunk_external_id: "external".repeat(seed + 1),
                    model_name: "model".repeat(seed + 1),
                    // Legacy diagnosis must observe both stored dim and actual vector length.
                    dim: seed + 2 + usize::from(codec == FileCodec::Legacy),
                    vector: vec![0.25; seed + 2],
                    created_at: "1970-01-01T00:00:00Z".into(),
                };
                let bytes = if codec == FileCodec::Binary {
                    codec::encode(&RecordPayload::EmbeddingPut(embedding.clone())).unwrap()
                } else {
                    embedding.to_json().to_compact_string().into_bytes()
                };
                let (path, mut reader, record) = fixture(codec, "EmbeddingPut", &bytes);
                let context = JsonOperationContext::standalone();
                let retained = reader.read_embedding_in(&record, &context).unwrap();
                let value = retained.embedding();
                assert_eq!(value.vector, embedding.vector);
                assert_eq!(value.dim, embedding.dim);
                assert_eq!(value.model_name, embedding.model_name);
                let strings = value.embedding_id.capacity()
                    + value.chunk_id.capacity()
                    + value.collection_id.capacity()
                    + value.chunk_external_id.capacity()
                    + value.model_name.capacity()
                    + value.created_at.capacity();
                let vector_bytes = value.vector.capacity() * std::mem::size_of::<f32>();
                let charged = strings + vector_bytes;
                assert_eq!(retained.work.reserved_bytes(), charged);
                let mut other = context.workspace_reservation();
                other
                    .grow_to(limit - charged, limit - charged, "operation_scratch")
                    .unwrap();
                assert!(other
                    .grow_to(
                        limit - charged + 1,
                        limit - charged + 1,
                        "operation_scratch"
                    )
                    .is_err());
                if seed % 2 == 0 {
                    let vector = retained.into_vector();
                    other
                        .grow_to(
                            limit - vector_bytes,
                            limit - vector_bytes,
                            "operation_scratch",
                        )
                        .unwrap();
                    assert_eq!(vector.as_slice(), embedding.vector);
                    drop(vector);
                } else {
                    drop(retained);
                }
                other.grow_to(limit, limit, "operation_scratch").unwrap();
                drop(other);
                drop(reader);
                fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn scanned_vectors_yield_optional_buffers_and_keep_live_values_charged() {
        for seed in 0..100usize {
            let embedding = Embedding {
                embedding_id: "emb_1".into(),
                chunk_id: "chk_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: String::new(),
                model_name: "model".repeat(seed + 1),
                dim: 32769 + seed,
                vector: vec![0.5; 32769 + seed],
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let bytes = codec::encode(&RecordPayload::EmbeddingPut(embedding.clone())).unwrap();
            let (path, mut reader, record) = fixture(FileCodec::Binary, "EmbeddingPut", &bytes);
            let context = JsonOperationContext::standalone();
            let vector_bytes = embedding.dim * std::mem::size_of::<f32>();
            // Three canonical 20-byte identifiers and one 20-byte timestamp are also live during decode.
            let minimum = 80 + embedding.model_name.len() + vector_bytes;
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            for available in [0, 19, minimum - 1, minimum, minimum + 65536] {
                let mut held = context.workspace_reservation();
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                reader.file.seek(SeekFrom::Start(7)).unwrap();
                let result = reader.read_embedding_vector_in(&record, &context);
                if available < minimum {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "operation" && value == "operation_scratch"));
                    if available == minimum - 1 {
                        assert!(error
                            .detail()
                            .iter()
                            .any(|(key, value)| key == "requested_bytes"
                                && value == &(limit + 1).to_string()));
                    }
                } else {
                    let vector = result.unwrap();
                    assert_eq!(vector.as_slice(), embedding.vector);
                    let mut other = context.workspace_reservation();
                    other
                        .grow_to(
                            available - vector_bytes,
                            available - vector_bytes,
                            "operation_scratch",
                        )
                        .unwrap();
                    assert!(other
                        .grow_to(
                            available - vector_bytes + 1,
                            available - vector_bytes + 1,
                            "operation_scratch"
                        )
                        .is_err());
                }
                #[cfg(unix)]
                assert_eq!(reader.file.stream_position().unwrap(), 7);
                context
                    .workspace_reservation()
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
            }
            let mut held = context.workspace_reservation();
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            reader
                .file
                .seek(SeekFrom::Start(
                    record.payload.offset + record.payload.len - 1,
                ))
                .unwrap();
            reader
                .file
                .write_all(&[bytes[bytes.len() - 1] ^ 0x80])
                .unwrap();
            assert_eq!(
                reader
                    .read_embedding_vector_in(&record, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::ChecksumMismatch
            );
            reader
                .file
                .set_len(record.payload.offset + record.payload.len - 1)
                .unwrap();
            assert_eq!(
                reader
                    .read_embedding_vector_in(&record, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::IoError
            );
            drop(held);
            context
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();
            drop(reader);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn legacy_scanned_vectors_transfer_typed_copy_credit_and_release_it_on_drop() {
        for seed in 0..100usize {
            let embedding = Embedding {
                embedding_id: "emb_1".into(),
                chunk_id: "chk_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "chunk".into(),
                model_name: "legacy".repeat(seed + 1),
                dim: seed + 2,
                vector: vec![0.25; seed + 2],
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let bytes = embedding.to_json().to_compact_string().into_bytes();
            let (path, mut reader, record) = fixture(FileCodec::Legacy, "EmbeddingPut", &bytes);
            let context = JsonOperationContext::standalone();
            let vector = reader.read_embedding_vector_in(&record, &context).unwrap();
            assert_eq!(vector.as_slice(), embedding.vector);
            // Legacy conversion grows a Vec, so retained capacity can exceed its length.
            let vector_bytes = Embedding::from_json(&embedding.to_json())
                .unwrap()
                .vector
                .capacity()
                * std::mem::size_of::<f32>();
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let mut rest = context.reservation();
            rest.grow(limit - vector_bytes, JsonInputKind::StoredJson)
                .unwrap();
            let error = rest.grow(1, JsonInputKind::StoredJson).unwrap_err();
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("limit")
                    .unwrap()
                    .as_f64(),
                Some((limit - vector_bytes) as f64)
            );
            drop(vector);
            rest.grow(vector_bytes, JsonInputKind::StoredJson).unwrap();
            drop(rest);
            reader
                .file
                .seek(SeekFrom::Start(
                    record.payload.offset + record.payload.len - 1,
                ))
                .unwrap();
            reader.file.write_all(b"!").unwrap();
            assert_eq!(
                reader
                    .read_embedding_vector_in(&record, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::ChecksumMismatch
            );
            context
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();
            drop(reader);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn non_json_positioned_reads_share_buffers_and_yield_to_staged_strings() {
        for seed in 0..100usize {
            let embedding = crate::model::Embedding {
                embedding_id: "emb_1".into(),
                chunk_id: "chk_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: String::new(),
                model_name: "model".repeat(1 + seed),
                dim: 32769 + seed,
                vector: vec![0.5; 32769 + seed],
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let bytes = codec::encode(&RecordPayload::EmbeddingPut(embedding.clone())).unwrap();
            let (path, mut reader, record) = fixture(FileCodec::Binary, "EmbeddingPut", &bytes);
            let context = JsonOperationContext::standalone();
            let available = [0, 1, 4096, 65535, 65536][seed % 5];
            let mut strings = context.workspace_reservation();
            strings.grow_to(17, 17, "batch_prepare").unwrap();
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available - 17;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            reader.file.seek(SeekFrom::Start(7)).unwrap();
            let (payload, work) = reader.read_payload_in(&record, &context).unwrap();
            let RecordPayload::EmbeddingPut(decoded) = payload else {
                panic!()
            };
            assert_eq!(decoded.vector, embedding.vector);
            assert_eq!(decoded.model_name, embedding.model_name);
            assert_eq!(work.reserved_bytes(), 0);
            drop(decoded);
            drop(work);
            let staged = read_staged_summary_in(
                &mut reader.file,
                EMBEDDING_PUT,
                &record.payload,
                &context,
                &mut strings,
            );
            if available < 4096 {
                assert_eq!(staged.err().unwrap().code(), ErrorCode::ResourceExhausted);
                assert_eq!(strings.reserved_bytes(), 17);
            } else {
                let (payload, _) = staged.unwrap();
                let RecordPayload::EmbeddingPut(decoded) = payload else {
                    panic!()
                };
                assert_eq!(decoded.model_name, embedding.model_name);
                assert!(decoded.vector.is_empty());
                assert!(strings.reserved_bytes() > 17);
                drop(decoded);
                strings.retain(17);
            }
            #[cfg(unix)]
            assert_eq!(reader.file.stream_position().unwrap(), 7);
            let mut probe = context.workspace_reservation();
            assert_eq!(
                probe
                    .grow_to(available, available, "operation_scratch")
                    .unwrap(),
                available
            );
            drop(probe);
            reader
                .file
                .seek(SeekFrom::Start(
                    record.payload.offset + record.payload.len - 1,
                ))
                .unwrap();
            reader
                .file
                .write_all(&[bytes[bytes.len() - 1] ^ 0x80])
                .unwrap();
            assert_eq!(
                reader
                    .read_payload_in(&record, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::ChecksumMismatch
            );
            assert_eq!(
                read_staged_summary_in(
                    &mut reader.file,
                    EMBEDDING_PUT,
                    &record.payload,
                    &context,
                    &mut strings
                )
                .err()
                .unwrap()
                .code(),
                ErrorCode::ChecksumMismatch
            );
            assert_eq!(strings.reserved_bytes(), 17);
            reader
                .file
                .set_len(record.payload.offset + record.payload.len - 1)
                .unwrap();
            assert_eq!(
                read_staged_summary_in(
                    &mut reader.file,
                    EMBEDDING_PUT,
                    &record.payload,
                    &context,
                    &mut strings
                )
                .err()
                .unwrap()
                .code(),
                ErrorCode::IoError
            );
            assert_eq!(strings.reserved_bytes(), 17);
            drop(reader);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn small_json_summaries_do_not_require_the_preferred_input_buffer() {
        for seed in 0..32usize {
            let text = "large text with a small summary ".repeat(4096 + seed);
            let chunk = Chunk {
                chunk_id: "chk_1".into(),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "external".into(),
                text_hash: crate::model::chunk_text_hash(&text),
                text,
                ordinal: seed,
                token_count: 1,
                metadata: JsonValue::object(),
                source_span: None,
                embedding_id: None,
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let bytes = codec::encode(&RecordPayload::ChunkPut(chunk.clone())).unwrap();
            let (path, mut reader, record) = fixture(FileCodec::Binary, "ChunkPut", &bytes);
            let database_path = path.with_extension("coffer");
            let mut storage = Storage::create(&database_path).unwrap();
            storage
                .append_batch(&[RecordPayload::ChunkPut(chunk.clone())], Durability::Fast)
                .unwrap();
            let original = fs::read(&database_path).unwrap();
            let context = JsonOperationContext::standalone();
            let available = 16 * 1024 + seed * 64;
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available;
            let mut held = context.workspace_reservation();
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let mut strings = context.workspace_reservation();
            let direct = codec::decode_reader_with_kind_in(
                CHUNK_PUT,
                &mut bytes.as_slice(),
                bytes.len() as u64,
                false,
                JsonInputKind::ChunkMetadata,
                &context,
                Some((&mut strings, "batch_prepare")),
            )
            .unwrap();
            drop(direct);
            strings.retain(0);
            let (payload, work) = read_staged_summary_in(
                &mut reader.file,
                CHUNK_PUT,
                &record.payload,
                &context,
                &mut strings,
            )
            .unwrap();
            let RecordPayload::ChunkPut(summary) = payload else {
                panic!()
            };
            assert!(summary.text.is_empty());
            assert_eq!(summary.text_hash, chunk.text_hash);
            assert_eq!(summary.ordinal, seed);
            assert_eq!(summary.metadata, chunk.metadata);
            drop((summary, work));
            strings.retain(0);
            let mut visits = 0;
            let scan = scan_file_with_records_in(&database_path, &context, |record| {
                visits += 1;
                let RecordPayload::ChunkPut(summary) = record.payload else {
                    panic!()
                };
                assert!(summary.text.is_empty());
                assert_eq!(summary.text_hash, chunk.text_hash);
                assert_eq!(summary.ordinal, seed);
                Ok(())
            })
            .unwrap();
            assert_eq!(visits, 1);
            assert_eq!(scan.commits, 1);
            assert!(scan.findings.is_empty());
            assert_eq!(fs::read(&database_path).unwrap(), original);
            let mut damaged = OpenOptions::new().write(true).open(&database_path).unwrap();
            let position = HEADER_LEN + RECORD_HEADER_LEN;
            damaged.seek(SeekFrom::Start(position as u64)).unwrap();
            damaged.write_all(&[original[position] ^ 0x80]).unwrap();
            let scan = scan_file_with_records_in(&database_path, &context, |_| {
                panic!("a damaged batch cannot be visible")
            })
            .unwrap();
            assert_eq!(scan.commits, 0);
            assert!(scan
                .findings
                .iter()
                .any(|finding| finding.code == "checksum_mismatch"));
            damaged
                .set_len((position + bytes.len() - 1) as u64)
                .unwrap();
            let scan = scan_file_with_records_in(&database_path, &context, |_| {
                panic!("an incomplete batch cannot be visible")
            })
            .unwrap();
            assert_eq!(scan.commits, 0);
            assert!(scan
                .findings
                .iter()
                .any(|finding| finding.code == "corrupt_tail"));
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop((damaged, reader, storage));
            fs::remove_file(path).unwrap();
            fs::remove_file(database_path).unwrap();
        }
    }

    #[test]
    fn positioned_json_reads_yield_optional_buffers_to_required_metadata() {
        for seed in 0..100usize {
            let metadata =
                object_from_pairs([("payload", JsonValue::String("x".repeat(128 * 1024 + seed)))]);
            let bytes =
                codec::encode(&RecordPayload::DocumentPut(document(metadata.clone()))).unwrap();
            let (path, mut reader, record) = fixture(FileCodec::Binary, "DocumentPut", &bytes);
            let context = JsonOperationContext::standalone();
            let available = 288 * 1024 + seed;
            let mut strings = context.workspace_reservation();
            strings.grow_to(17, 17, "batch_prepare").unwrap();
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available - 17;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            // The same record fits without optional input buffering.
            let direct = codec::decode_reader_with_kind_in(
                DOCUMENT_PUT,
                &mut bytes.as_slice(),
                bytes.len() as u64,
                false,
                JsonInputKind::DocumentMetadata,
                &context,
                Some((&mut strings, "batch_prepare")),
            )
            .unwrap();
            drop(direct);
            strings.retain(17);
            reader.file.seek(SeekFrom::Start(7)).unwrap();
            let (payload, work) = read_staged_summary_in(
                &mut reader.file,
                DOCUMENT_PUT,
                &record.payload,
                &context,
                &mut strings,
            )
            .unwrap();
            let RecordPayload::DocumentPut(decoded) = payload else {
                panic!()
            };
            assert_eq!(decoded.metadata, metadata);
            assert_eq!(
                strings.reserved_bytes(),
                17 + decoded.document_id.capacity()
                    + decoded.collection_id.capacity()
                    + decoded.external_id.capacity()
                    + decoded.created_at.capacity()
                    + decoded.updated_at.capacity()
            );
            #[cfg(unix)]
            assert_eq!(reader.file.stream_position().unwrap(), 7);
            drop((decoded, work));
            strings.retain(17);
            let mut probe = context.workspace_reservation();
            probe
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop(probe);
            reader
                .file
                .seek(SeekFrom::Start(record.payload.offset))
                .unwrap();
            reader.file.write_all(&[bytes[0] ^ 0x80]).unwrap();
            assert_eq!(
                read_staged_summary_in(
                    &mut reader.file,
                    DOCUMENT_PUT,
                    &record.payload,
                    &context,
                    &mut strings,
                )
                .err()
                .unwrap()
                .code(),
                ErrorCode::ChecksumMismatch
            );
            assert_eq!(strings.reserved_bytes(), 17);
            reader
                .file
                .set_len(record.payload.offset + record.payload.len - 1)
                .unwrap();
            assert_eq!(
                read_staged_summary_in(
                    &mut reader.file,
                    DOCUMENT_PUT,
                    &record.payload,
                    &context,
                    &mut strings,
                )
                .err()
                .unwrap()
                .code(),
                ErrorCode::IoError
            );
            assert_eq!(strings.reserved_bytes(), 17);
            drop(reader);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn required_json_admission_still_checks_damage_without_input_buffer() {
        let bytes = codec::encode(&RecordPayload::DocumentPut(document(object_from_pairs([
            ("payload", JsonValue::String("x".repeat(8192))),
        ]))))
        .unwrap();
        for available in 0..100 {
            let (path, mut reader, record) = fixture(FileCodec::Binary, "DocumentPut", &bytes);
            let context = JsonOperationContext::standalone();
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let error = reader.read_payload_in(&record, &context).err().unwrap();
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == &available.to_string()));
            let mut strings = context.workspace_reservation();
            let error = read_staged_summary_in(
                &mut reader.file,
                DOCUMENT_PUT,
                &record.payload,
                &context,
                &mut strings,
            )
            .err()
            .unwrap();
            // Two canonical IDs and the external ID precede the JSON metadata.
            if available < 43 {
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                let required = if available < 20 {
                    20
                } else if available < 40 {
                    40
                } else {
                    43
                };
                for (key, value) in [
                    ("resource", "memory".into()),
                    ("operation", "batch_prepare".into()),
                    (
                        "limit_bytes",
                        JsonProcessingPolicyV1::WORK_BUDGET_BYTES.to_string(),
                    ),
                    ("requested_bytes", (occupied + required).to_string()),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)));
                }
            } else {
                assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                for (key, value) in [
                    ("json_limit_reason", "work_budget_exceeded".into()),
                    ("json_input_kind", "document_metadata".into()),
                    ("limit", (available - 43).to_string()),
                ] {
                    assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                }
            }
            assert_eq!(strings.reserved_bytes(), 0);
            reader
                .file
                .seek(SeekFrom::Start(
                    record.payload.offset + record.payload.len - 1,
                ))
                .unwrap();
            reader
                .file
                .write_all(&[bytes[bytes.len() - 1] ^ 0x80])
                .unwrap();
            assert_eq!(
                reader
                    .read_payload_in(&record, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::ChecksumMismatch
            );
            reader
                .file
                .set_len(record.payload.offset + record.payload.len - 1)
                .unwrap();
            assert_eq!(
                reader
                    .read_payload_in(&record, &context)
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::IoError
            );
            let mut probe = context.workspace_reservation();
            assert_eq!(
                probe
                    .grow_to(available, available, "operation_scratch")
                    .unwrap(),
                available
            );
            drop(reader);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn binary_decode_transfers_retained_json_and_preserves_file_position() {
        let value = object_from_pairs([("payload", JsonValue::String("x".repeat(512 * 1024)))]);
        let bytes = codec::encode(&RecordPayload::DocumentPut(document(value.clone()))).unwrap();
        let (path, mut reader, record) = fixture(FileCodec::Binary, "DocumentPut", &bytes);
        let context = JsonOperationContext::standalone();
        reader.file.seek(SeekFrom::Start(7)).unwrap();
        let (payload, work) = reader.read_payload_in(&record, &context).unwrap();
        #[cfg(unix)]
        assert_eq!(reader.file.stream_position().unwrap(), 7);
        let RecordPayload::DocumentPut(document) = payload else {
            panic!()
        };
        assert_eq!(document.metadata, value);
        let (measured, _) = document
            .metadata
            .guard_in(JsonInputKind::StoredJson, &context)
            .unwrap();
        assert_eq!(measured.reserved_bytes(), work.reserved_bytes());
        drop(measured);
        drop(document);
        drop(work);
        let mut all = context.reservation();
        all.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::Filter,
        )
        .unwrap();
        drop(reader);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_budget_failure_drains_for_checksum_and_releases_temporary_work() {
        let bytes = codec::encode(&RecordPayload::DocumentPut(document(object_from_pairs([
            ("payload", JsonValue::String("x".repeat(512 * 1024))),
        ]))))
        .unwrap();
        let (path, mut reader, record) = fixture(FileCodec::Binary, "DocumentPut", &bytes);
        let before = fs::read(&path).unwrap();
        let trace = crate::search_cost::SearchCostTrace::new();
        let context = JsonOperationContext::standalone()
            .with_search_cost(&trace)
            .search_cost_stage(crate::search_cost::Stage::ResultMaterialization);
        let mut busy = context.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 128 * 1024,
            JsonInputKind::Filter,
        )
        .unwrap();
        let error = reader.read_payload_in(&record, &context).err().unwrap();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "stored_json"));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert_eq!(
            trace.total(crate::search_cost::Counter::PrimaryRequestedBytes),
            record.payload.len
        );
        assert_eq!(
            trace.total(crate::search_cost::Counter::PrimaryReadBytes),
            2 * record.payload.len
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        reader
            .file
            .seek(SeekFrom::Start(
                record.payload.offset + record.payload.len - 1,
            ))
            .unwrap();
        reader.file.write_all(&[1]).unwrap();
        reader.file.flush().unwrap();
        assert_eq!(
            reader
                .read_payload_in(&record, &context)
                .err()
                .unwrap()
                .code(),
            ErrorCode::ChecksumMismatch
        );
        assert_eq!(
            trace.total(crate::search_cost::Counter::PrimaryRequestedBytes),
            2 * record.payload.len
        );
        assert_eq!(
            trace.total(crate::search_cost::Counter::PrimaryReadBytes),
            3 * record.payload.len
        );
        drop(busy);
        let mut all = context.reservation();
        all.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::Filter,
        )
        .unwrap();
        drop(all);
        reader
            .file
            .set_len(record.payload.offset + record.payload.len / 2)
            .unwrap();
        assert_eq!(
            reader
                .read_payload_in(&record, &context)
                .err()
                .unwrap()
                .code(),
            ErrorCode::IoError
        );
        assert_eq!(
            trace.total(crate::search_cost::Counter::PrimaryRequestedBytes),
            3 * record.payload.len
        );
        assert_eq!(
            trace.total(crate::search_cost::Counter::PrimaryReadBytes),
            3 * record.payload.len + record.payload.len / 2
        );
        assert_eq!(
            trace.value(
                crate::search_cost::Stage::ResultMaterialization,
                crate::search_cost::Counter::PrimaryReadBytes
            ),
            trace.total(crate::search_cost::Counter::PrimaryReadBytes)
        );
        drop(reader);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_non_json_payload_does_not_consume_json_headroom() {
        let bytes = codec::encode(&RecordPayload::EmbeddingPut(Embedding {
            embedding_id: "emb_1".into(),
            chunk_id: "chk_1".into(),
            chunk_external_id: "doc#1".into(),
            collection_id: "col_1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
            created_at: "1970-01-01T00:00:00Z".into(),
        }))
        .unwrap();
        let (path, mut reader, record) = fixture(FileCodec::Binary, "EmbeddingPut", &bytes);
        let context = JsonOperationContext::standalone();
        let mut busy = context.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::Filter,
        )
        .unwrap();
        let (payload, work) = reader.read_payload_in(&record, &context).unwrap();
        assert!(matches!(payload, RecordPayload::EmbeddingPut(_)));
        assert_eq!(work.reserved_bytes(), 0);
        drop(reader);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_metadata_at_unit_limit_is_moved_without_charging_the_record_wrapper() {
        let size = JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES - 14;
        let doc = document(object_from_pairs([(
            "payload",
            JsonValue::String("x".repeat(size)),
        )]));
        let bytes = doc.to_json().to_compact_string().into_bytes();
        assert!(bytes.len() > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES);
        let (path, mut reader, record) = fixture(FileCodec::Legacy, "DocumentPut", &bytes);
        let context = JsonOperationContext::standalone();
        let (payload, work) = reader.read_payload_in(&record, &context).unwrap();
        let RecordPayload::DocumentPut(actual) = payload else {
            panic!()
        };
        assert_eq!(
            actual
                .metadata
                .get("payload")
                .unwrap()
                .as_str()
                .unwrap()
                .len(),
            size
        );
        let measured = actual
            .metadata
            .reserve_retained_in(JsonInputKind::StoredJson, &context)
            .unwrap();
        assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
        drop(measured);
        drop(actual);
        drop(work);
        let mut all = context.reservation();
        all.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::StoredJson,
        )
        .unwrap();
        drop(reader);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_adapter_keeps_metadata_business_root_below_record_wrapper() {
        let mut metadata = JsonValue::Null;
        for _ in 0..127 {
            metadata = JsonValue::Array(vec![metadata]);
        }
        let metadata = object_from_pairs([("nested", metadata)]);
        let doc = document(metadata.clone());
        let bytes = doc.to_json().to_compact_string().into_bytes();
        let (path, mut reader, record) = fixture(FileCodec::Legacy, "DocumentPut", &bytes);
        let context = JsonOperationContext::standalone();
        let (payload, _work) = reader.read_payload_in(&record, &context).unwrap();
        let RecordPayload::DocumentPut(actual) = payload else {
            panic!()
        };
        assert_eq!(actual.metadata, metadata);
        drop(reader);
        fs::remove_file(path).unwrap();
    }
}
