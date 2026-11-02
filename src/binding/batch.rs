use super::input::Argument;
use crate::db::{ImportDocument, ImportRow};
use crate::json::{JsonInputKind, JsonOperationContext, ReservedVec, WorkspaceReservation};
use crate::storage::RecordReservation;
use crate::{ChunkInput, DocumentInput, EmbeddingInput, Error, ErrorCode, InsertSummary, Result};

pub struct ChunkRecord {
    pub document_external_id: String,
    pub chunk: ChunkInput,
    pub source_span_provided: bool,
}

pub struct Batch {
    documents: ReservedVec<Argument<DocumentInput>>,
    chunks: ReservedVec<Argument<ChunkRecord>>,
    embeddings: ReservedVec<Argument<EmbeddingInput>>,
    context: JsonOperationContext,
}

pub struct WriteSummary {
    pub(crate) counts: InsertSummary,
    pub(crate) warnings: Vec<crate::import::ImportWarning>,
    pub(crate) _work: WorkspaceReservation,
}

impl WriteSummary {
    pub fn counts(&self) -> &InsertSummary {
        &self.counts
    }
    pub fn warnings(&self) -> &[crate::import::ImportWarning] {
        &self.warnings
    }

    pub fn into_output(self, database: &crate::Database) -> Result<super::Output> {
        let Self {
            counts,
            warnings,
            _work,
        } = self;
        let context = _work.context().clone();
        let mut work = RecordReservation::new(context.reservation(), _work);
        let (json, strings) = work.parts_mut();
        let builder = crate::json::JsonBuilder::budgeted_owned_strings(
            json,
            strings,
            JsonInputKind::StoredJson,
        );
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
        work.retain_workspace(0);
        Ok(super::Output::new(value, work, Some(database.clone())))
    }
}

impl Batch {
    pub(super) fn new(context: &JsonOperationContext) -> Result<Self> {
        Ok(Self {
            documents: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            chunks: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            embeddings: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            context: context.clone(),
        })
    }

    fn check_context<T>(&self, input: &Argument<T>) -> Result<()> {
        self.check_operation(input.work.json.context())
    }

    pub(super) fn check_operation(&self, context: &JsonOperationContext) -> Result<()> {
        if !self.context.same_operation(context) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "batch arguments must belong to the same operation",
            ));
        }
        Ok(())
    }

    pub fn push_document(&mut self, input: Argument<DocumentInput>) -> Result<()> {
        self.check_context(&input)?;
        if !input.value.chunks.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "direct document records do not accept nested chunks",
            )
            .with_detail("field", "chunks"));
        }
        input
            .value
            .metadata
            .encoding_shape_in(JsonInputKind::DocumentMetadata, &self.context)?;
        self.documents.try_push(input)
    }

    pub fn push_chunk(&mut self, mut input: Argument<ChunkRecord>) -> Result<()> {
        self.check_context(&input)?;
        input
            .value
            .chunk
            .metadata
            .encoding_shape_in(JsonInputKind::ChunkMetadata, &self.context)?;
        if input.value.source_span_provided && input.value.chunk.source_span.is_none() {
            input.value.chunk.source_span = Some(Default::default());
        }
        self.chunks.try_push(input)
    }

    pub fn push_embedding(&mut self, input: Argument<EmbeddingInput>) -> Result<()> {
        self.check_context(&input)?;
        input.value.validate_structure()?;
        self.embeddings.try_push(input)
    }

    pub(super) fn visit(
        self,
        consume: &mut dyn FnMut(ImportRow, RecordReservation) -> Result<()>,
    ) -> Result<()> {
        let (documents, documents_work) = self.documents.into_parts();
        for (index, input) in documents.into_iter().enumerate() {
            let row = input.map(|document| ImportRow::Document {
                input: ImportDocument {
                    external_id: document.external_id,
                    title: document.title,
                    source_uri: document.source_uri,
                    metadata: document.metadata,
                },
                line: index + 1,
            });
            consume(row.value, row.work)?;
        }
        drop(documents_work);
        let (chunks, chunks_work) = self.chunks.into_parts();
        for (index, mut input) in chunks.into_iter().enumerate() {
            let explicit = if input.value.source_span_provided {
                input.value.chunk.source_span.clone()
            } else {
                None
            };
            let (span, conflict) =
                crate::import::resolve_input_source_span(&input.value.chunk.metadata, explicit)?;
            input.value.chunk.source_span = span;
            let row = input.map(|row| ImportRow::Chunk {
                document_external_id: row.document_external_id,
                chunk: row.chunk,
                line: index + 1,
                source_span_metadata_conflict: conflict,
            });
            consume(row.value, row.work)?;
        }
        drop(chunks_work);
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
