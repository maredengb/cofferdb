use super::{arguments, convert, owner, read_output, requests, shape, DefaultArgument};
use pyo3::prelude::*;
use std::sync::Arc;

/// Access one named collection through its database owner. Obtain this handle from
/// Database.collection or Database.create_collection.
///
/// Calls share the owner's single-operation limit and its read view or writer lock. Closing
/// the owner invalidates this handle. Returned documents, chunks, results and statistics
/// are owned Python objects.
#[pyclass(module = "cofferdb", frozen)]
pub(crate) struct Collection {
    // Shares the parent's native handle, close state and non-queuing gate with
    // sibling collections.
    pub(super) owner: Arc<owner::Owner>,
    // Owned internal ID used to scope native reads, writes and searches after
    // argument conversion.
    pub(super) collection_id: String,
    // Owned name used by collection export, whose engine entry point resolves the
    // collection by name.
    pub(super) name: String,
}

#[pymethods]
impl Collection {
    /// Validate and commit one mixed batch on this collection's writer and return a
    /// WriteSummary.
    ///
    /// Each of documents, chunks and embeddings takes a finite built-in list or tuple,
    /// empty by default. An empty batch does not commit. Inputs are fixed before the first
    /// write, and a large call is never split. on_conflict selects error (default), skip or
    /// replace. Inspect per-kind counts and warnings. After a signal or conversion error,
    /// inspect operation_outcome if available and verify the data before retrying: an
    /// atomic batch may already have committed.
    #[pyo3(signature = (*, documents=DefaultArgument::default(), chunks=DefaultArgument::default(), embeddings=DefaultArgument::default(), on_conflict=DefaultArgument::default()),
           text_signature = "($self, *, documents=(), chunks=(), embeddings=(), on_conflict='error')")]
    fn write_batch(
        &self,
        py: Python<'_>,
        documents: DefaultArgument,
        chunks: DefaultArgument,
        embeddings: DefaultArgument,
        on_conflict: DefaultArgument,
    ) -> PyResult<Py<PyAny>> {
        // The same wrapper materializes both reads and writes. Mark this call as
        // writing so it can preserve native progress if construction of the Python
        // summary or a later signal check fails.
        read_output(
            py,
            &self.owner,
            true,
            |operation| {
                // Validate on_conflict before converting the row sequences. An
                // invalid policy fails without walking or allocating payloads from
                // an otherwise large batch.
                let mut input = operation.input();
                let conflict = arguments::conflict(on_conflict.bind(py), &mut input)
                    .map_err(|error| shape(py, error))?;
                // Keep Python objects in the attached preparation closure. Each
                // converted row enters one native Batch with its operation credit,
                // so the detached call needs no access to the caller's lists.
                let batch = requests::batch(
                    operation,
                    documents.bind(py),
                    chunks.bind(py),
                    embeddings.bind(py),
                )
                .map_err(|error| shape(py, error))?;
                Ok((batch, conflict))
            },
            // Writing and projecting the summary are separate fallible steps. The
            // native batch can already be committed if converting the counts or
            // warnings into an Output fails.
            |operation, database, (batch, conflict)| {
                operation
                    .write_batch(database, &self.collection_id, batch, conflict)?
                    .into_output(database)
            },
        )
    }

    /// Read full chunk content and its parent document by the chunk's internal ID. Both
    /// records come from one view in an owned ChunkContent mapping. A missing chunk raises
    /// CofferError.
    fn get_chunk(&self, py: Python<'_>, chunk_id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // The boolean chooses the native internal-ID lookup. An internal ID and an
        // application external ID are both strings, so keep this choice explicit
        // at the public entry points.
        self.chunk(py, chunk_id, false)
    }

    /// Read full chunk content and its parent document by the application-provided chunk ID
    /// in this collection. Both come from one view; a missing chunk raises CofferError.
    fn get_chunk_by_external_id(
        &self,
        py: Python<'_>,
        chunk_external_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.chunk(py, chunk_external_id, true)
    }

    /// Return the complete visible document for this collection's external_id, including
    /// its internal ID and stored metadata. A missing document raises CofferError. The
    /// source URI is not fetched.
    fn get_document_by_external_id(
        &self,
        py: Python<'_>,
        external_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            false,
            |operation| {
                let mut input = operation.input();
                // Copy the application's ID while Python is attached and transfer
                // the reservation with it. A conversion error reaches the caller
                // before any native document lookup.
                let id =
                    convert::text(external_id, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(id))
            },
            // Use the stored collection ID as part of the lookup. External
            // document IDs need only be unique within their collection, so an
            // unscoped lookup could return another collection's document.
            |operation, database, id| operation.get_document(database, &self.collection_id, id),
        )
    }

    /// Logically delete the document with this internal ID on the collection's writer. The
    /// commit also hides its chunks and embeddings. Returns None. Bytes may remain in the
    /// file. After interruption, inspect any operation_outcome and verify visibility before
    /// retrying.
    fn delete_document(
        &self,
        py: Python<'_>,
        document_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        // This selector chooses the native document tombstone path. Chunk deletion
        // follows a different path even though both accept an internal ID and
        // return Python None.
        self.delete(py, document_id, true)
    }

    /// Logically delete the chunk with this internal ID and its associated embedding on the
    /// collection's writer. Returns None; document and file bytes may remain. After
    /// interruption, inspect any operation_outcome and verify visibility before retrying.
    fn delete_chunk(&self, py: Python<'_>, chunk_id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.delete(py, chunk_id, false)
    }

    /// Return an owned CollectionStats mapping for this collection's visible data,
    /// including document, chunk, embedding and tombstone counts and index/storage
    /// estimates.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            false,
            // Statistics need no argument conversion. They still go through owner
            // admission and the common output path because reading collection
            // state and building the host mapping can each fail.
            |_| Ok(()),
            |operation, database, ()| operation.collection_stats(database, &self.collection_id),
        )
    }

    /// Write JSONL documents, chunks and embeddings from one visible view to output_dir and
    /// return the counts, checksums and manifest filename.
    ///
    /// This reads the collection and can replace previous export artifacts. The manifest is
    /// published last. If publication fails, check the files against the manifest and
    /// checksums before reimporting. Use a fresh export directory to keep a previous
    /// export.
    fn export_jsonl(&self, py: Python<'_>, output_dir: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Export reads the database rather than submitting a main-data write. Use
        // the read-result wrapper even though producing the export changes
        // destination directory entries.
        read_output(
            py,
            &self.owner,
            false,
            // Convert the destination path under this operation's input budget.
            // The detached exporter borrows only the native path, never the Python
            // object.
            |operation| {
                let mut input = operation.input();
                let path =
                    convert::text(output_dir, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(path))
            },
            // Pass the visible collection name because the exported manifest uses
            // it. Other operations on this handle normally resolve the internal
            // collection ID.
            |operation, database, path| operation.export_jsonl(database, &self.name, path),
        )
    }

    /// Return ranked SearchResults for text, an externally supplied query embedding, or
    /// both.
    ///
    /// filter restricts eligible records. weights and metadata_boost control score fusion.
    /// None uses input-dependent default weights and the owner's resource profile unless
    /// overridden. top_k defaults to 10 and accepts a built-in int from 1 to
    /// min(usize::MAX, 2**53-1), excluding bool. Inspect query_context and each hit's
    /// explain for actual candidate participation and scores. Results own their data; a
    /// later content lookup is a separate read.
    #[pyo3(signature = (text=None, *, embedding=None, filter=None, weights=None, metadata_boost=None, top_k=DefaultArgument::default(), profile=None),
           text_signature = "($self, text=None, *, embedding=None, filter=None, weights=None, metadata_boost=None, top_k=10, profile=None)")]
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        py: Python<'_>,
        text: Option<&Bound<'_, PyAny>>,
        embedding: Option<&Bound<'_, PyAny>>,
        filter: Option<&Bound<'_, PyAny>>,
        weights: Option<&Bound<'_, PyAny>>,
        metadata_boost: Option<&Bound<'_, PyAny>>,
        top_k: DefaultArgument,
        profile: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // Keep the shared owner gate held through argument conversion, detached
        // scoring and result materialization. Collection handles share the same
        // admission boundary as their database.
        read_output(
            py,
            &self.owner,
            false,
            // Collect the Python-facing options in one borrowed request while
            // attached. The converter returns an owned argument for the subsequent
            // native call.
            |operation| {
                arguments::query(
                    arguments::Query {
                        text,
                        embedding,
                        filter,
                        weights,
                        metadata_boost,
                        // The argument converter applies the default only on
                        // omission. A supplied limit must pass its Python integer
                        // check before building the native input.
                        top_k: top_k.bind(py),
                        profile,
                    },
                    operation,
                )
                .map_err(|error| shape(py, error))
            },
            |operation, database, input| operation.search(database, &self.collection_id, input),
        )
    }
}

impl Collection {
    // Share conversion and result accounting between both chunk lookup methods.
    // The caller selects the ID namespace explicitly rather than inferring it from
    // the string value.
    fn chunk(&self, py: Python<'_>, id: &Bound<'_, PyAny>, external: bool) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            false,
            |operation| {
                // Admit a single built-in string key before the detached lookup.
                // The typed Argument keeps its allocation charged for the duration
                // of the call.
                let mut input = operation.input();
                let id = convert::text(id, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(id))
            },
            |operation, database, id| {
                operation.get_chunk(database, &self.collection_id, id, external)
            },
        )
    }

    // Run a delete as a write-bearing call so an observed tombstone commit
    // survives a later Python error. Both forms resolve an internal ID in the
    // selected collection.
    fn delete(&self, py: Python<'_>, id: &Bound<'_, PyAny>, document: bool) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            true,
            |operation| {
                let mut input = operation.input();
                let id = convert::text(id, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(id))
            },
            // Choose the record family requested by the public method. Use the
            // same prepared argument and retained-output path for either mutation.
            |operation, database, id| {
                if document {
                    operation.delete_document(database, &self.collection_id, id)
                } else {
                    operation.delete_chunk(database, &self.collection_id, id)
                }
            },
        )
    }
}
