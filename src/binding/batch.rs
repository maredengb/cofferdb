use super::input::Argument;
use crate::db::{ImportDocument, ImportRow};
use crate::json::{JsonInputKind, JsonOperationContext, ReservedVec, WorkspaceReservation};
use crate::storage::RecordReservation;
use crate::{ChunkInput, DocumentInput, EmbeddingInput, Error, ErrorCode, InsertSummary, Result};

/// A host chunk row with its parent document key and the original presence of the
/// source-span field. The import consumer needs that presence bit after conversion
/// to the native chunk type.
pub struct ChunkRecord {
    /// External ID used to resolve this chunk's parent document in the target
    /// collection.
    pub document_external_id: String,
    /// Owned chunk data and the possible explicit source coordinates from the host
    /// record.
    pub chunk: ChunkInput,
    /// True when the host supplied the span key, including explicit null. Omission
    /// allows the import resolver to inherit a span from chunk metadata.
    pub source_span_provided: bool,
}

/// Mixed write input owned by one binding operation. Within a batch, documents are
/// presented first, followed by chunks and embeddings, so rows can reference
/// earlier kinds by external ID.
pub struct Batch {
    // Prepared document rows and their per-row credits, in the original document
    // sequence order.
    documents: ReservedVec<Argument<DocumentInput>>,
    // Prepared chunk rows. Parent references are resolved after the documents have
    // been presented to the same batch consumer.
    chunks: ReservedVec<Argument<ChunkRecord>>,
    // Prepared embedding rows, handed off after chunks so a vector can refer to a
    // chunk from the same batch.
    embeddings: ReservedVec<Argument<EmbeddingInput>>,
    // Operation shared by all three sequences. Rejecting a row from a different
    // context keeps its allocations on the correct budget.
    context: JsonOperationContext,
}

/// Native write counts and import warnings retained after a host batch succeeds.
/// The summary may be inspected directly or consumed to build an accounted host
/// result.
pub struct WriteSummary {
    // Core write summary returned after the mixed batch has been accepted, before
    // host output construction.
    pub(crate) counts: InsertSummary,
    // Warnings collected during the write, with owned strings kept until conversion
    // to the host report.
    pub(crate) warnings: Vec<crate::import::ImportWarning>,
    // Keeps warning storage charged through projection. Host conversion transfers
    // the credit for each owned warning string as it enters the result tree.
    pub(crate) _work: WorkspaceReservation,
}

impl WriteSummary {
    /// Borrow the native insertion counts without constructing the host-facing
    /// JSON result.
    pub fn counts(&self) -> &InsertSummary {
        &self.counts
    }
    /// Borrow the import warnings in their recorded order. The warning strings
    /// remain owned by this summary.
    pub fn warnings(&self) -> &[crate::import::ImportWarning] {
        &self.warnings
    }

    /// Consume the summary and build the native output tree for host conversion.
    /// The output retains a database handle for reclaiming cache space when
    /// admitting the host copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON values or their retained storage cannot be
    /// admitted. A projection error occurs after the original write has produced
    /// its summary.
    pub fn into_output(self, database: &crate::Database) -> Result<super::Output> {
        let Self {
            counts,
            warnings,
            _work,
        } = self;
        // Transfer the warning strings' existing workspace credit as they enter the
        // JSON tree. The summary owns the strings until then and should not need
        // room for a second copy of them.
        // Recover the operation from the summary's own reservation. Choosing a
        // fresh context would leave the existing warning buffers charged to a
        // different operation from their destination.
        let context = _work.context().clone();
        let mut work = RecordReservation::new(context.reservation(), _work);
        let (json, strings) = work.parts_mut();
        let builder = crate::json::JsonBuilder::budgeted_owned_strings(
            json,
            strings,
            JsonInputKind::StoredJson,
        );
        // Consume the warnings in their recorded order. The JSON builder can
        // transfer each owned string from the source workspace while constructing
        // this array.
        let warnings = builder.array(
            warnings
                .into_iter()
                .map(|warning| warning.into_json_owned(&builder)),
        )?;
        let mut value = counts.build_json(&builder)?;
        builder.insert_if_absent(
            value.as_object_mut().expect("summary object"),
            "warnings",
            || Ok(warnings),
        )?;
        // Only the JSON tree and its charge survive projection. Release residual
        // warning-vector workspace before handing the result to Output.
        work.retain_workspace(0);
        Ok(super::Output::new(value, work, Some(database.clone())))
    }
}

impl Batch {
    // Bind the three initially empty row sequences to one context. Each grows
    // separately as the host supplies rows of that kind.
    pub(super) fn new(context: &JsonOperationContext) -> Result<Self> {
        Ok(Self {
            documents: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            chunks: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            embeddings: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            context: context.clone(),
        })
    }

    // Check the reservation that owns the argument, not just its business-value
    // type. Two builders may produce the same kind of row while belonging to
    // different operations.
    fn check_context<T>(&self, input: &Argument<T>) -> Result<()> {
        self.check_operation(input.work.json.context())
    }

    pub(super) fn check_operation(&self, context: &JsonOperationContext) -> Result<()> {
        // Clones of the same JsonOperationContext share identity. Equal limits or
        // a common database alone do not make arguments interchangeable.
        if !self.context.same_operation(context) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "batch arguments must belong to the same operation",
            ));
        }
        Ok(())
    }

    /// Appends a document argument prepared by the operation that owns this batch.
    /// Put child chunks in the batch's separate chunk sequence.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument, nested chunks, inadmissible
    /// document metadata or insufficient buffer space. The row is appended only
    /// after these checks.
    pub fn push_document(&mut self, input: Argument<DocumentInput>) -> Result<()> {
        self.check_context(&input)?;
        // Documents and chunks have independent sequences and row numbers in the
        // import consumer. Accepting nested chunks here would give them no
        // unambiguous position in that stream.
        if !input.value.chunks.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "direct document records do not accept nested chunks",
            )
            .with_detail("field", "chunks"));
        }
        // Check metadata under its document-specific JSON policy while the
        // argument still owns it. Validation borrows the tree rather than
        // materializing a second metadata object.
        input
            .value
            .metadata
            .encoding_shape_in(JsonInputKind::DocumentMetadata, &self.context)?;
        self.documents.try_push(input)
    }

    /// Appends a chunk argument from the same operation. An omitted source span
    /// may be inferred from metadata. An explicitly supplied null suppresses that
    /// fallback.
    ///
    /// # Errors
    ///
    /// Returns an error if the argument belongs to another operation, the metadata
    /// fails JSON validation or its row slot cannot be reserved.
    pub fn push_chunk(&mut self, mut input: Argument<ChunkRecord>) -> Result<()> {
        self.check_context(&input)?;
        input
            .value
            .chunk
            .metadata
            .encoding_shape_in(JsonInputKind::ChunkMetadata, &self.context)?;
        // An omitted span inherits metadata. Preserve an explicitly null span
        // as empty until the import resolver checks for conflicting metadata.
        if input.value.source_span_provided && input.value.chunk.source_span.is_none() {
            input.value.chunk.source_span = Some(Default::default());
        }
        // Store the distinction between omission and explicit null in the queued
        // record. Comparison with any span found in metadata happens only when the
        // batch is visited.
        self.chunks.try_push(input)
    }

    /// Appends a prepared embedding for an external chunk ID.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument, blank chunk or model ID, a zero or
    /// mismatched width, nonfinite coordinates, or insufficient buffer space.
    /// Collection-specific checks occur when the batch is consumed.
    pub fn push_embedding(&mut self, input: Argument<EmbeddingInput>) -> Result<()> {
        self.check_context(&input)?;
        // Check the embedding's internal shape independently of any target
        // collection. This builder does not yet know which collection model and
        // dimension will be used.
        input.value.validate_structure()?;
        self.embeddings.try_push(input)
    }

    // Visit documents, then chunks, then embeddings, preserving each sequence's
    // original order. These dependencies allow a chunk to name a document, and a
    // vector to name a chunk, from this same batch.
    pub(super) fn visit(
        self,
        consume: &mut dyn FnMut(ImportRow, RecordReservation) -> Result<()>,
    ) -> Result<()> {
        // Each row hands its reservation to the consumer. The vector backing
        // that group stays allocated until its iterator is exhausted, so keep
        // the separate vector reservation alive for the whole loop.
        let (documents, documents_work) = self.documents.into_parts();
        for (index, input) in documents.into_iter().enumerate() {
            // Move only the flat document fields understood by the import
            // consumer. The push path has already established that the nested
            // chunks sequence is empty.
            let row = input.map(|document| ImportRow::Document {
                input: ImportDocument {
                    external_id: document.external_id,
                    title: document.title,
                    source_uri: document.source_uri,
                    metadata: document.metadata,
                },
                line: index + 1,
            });
            // Pass the row and its budget together. On a consumer error, the
            // remaining owned rows and their reservations drop as this consuming
            // traversal returns.
            consume(row.value, row.work)?;
        }
        // Each input array stays charged while its rows are consumed. Drop that
        // array's reservation before visiting the next kind. Row numbers are
        // one-based within each of the three host sequences.
        drop(documents_work);
        let (chunks, chunks_work) = self.chunks.into_parts();
        for (index, mut input) in chunks.into_iter().enumerate() {
            // Consult the explicit span only if the host actually supplied the
            // field. Without this presence bit an omitted field and an explicit
            // null would both look like None.
            let explicit = if input.value.source_span_provided {
                input.value.chunk.source_span.clone()
            } else {
                None
            };
            // Use the same span resolver as file import. An explicit span wins
            // even over malformed metadata and carries a conflict flag; an invalid
            // fallback with no explicit span fails the row.
            let (span, conflict) =
                crate::import::resolve_input_source_span(&input.value.chunk.metadata, explicit)?;
            input.value.chunk.source_span = span;
            let row = input.map(|row| ImportRow::Chunk {
                document_external_id: row.document_external_id,
                chunk: row.chunk,
                line: index + 1,
                // Keep the conflict alongside the resolved span for the importer
                // to report. The normalized span alone cannot show that the
                // metadata source disagreed.
                source_span_metadata_conflict: conflict,
            });
            consume(row.value, row.work)?;
        }
        drop(chunks_work);
        // An embedding row numbers from one in its own input sequence. It arrives
        // only after every queued chunk has been presented to the same consumer.
        let (embeddings, embeddings_work) = self.embeddings.into_parts();
        for (index, input) in embeddings.into_iter().enumerate() {
            let row = input.map(|input| ImportRow::Embedding {
                input,
                line: index + 1,
            });
            consume(row.value, row.work)?;
        }
        drop(embeddings_work);
        Ok(())
    }
}
