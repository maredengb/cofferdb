use super::{arguments, convert, owner, read_output, requests, shape, DefaultArgument};
use pyo3::prelude::*;
use std::sync::Arc;

/// A collection identity sharing its database's single synchronous owner.
///
/// Obtain from Database.create_collection or Database.collection. Closing the
/// owner invalidates all its collections; already returned data remains usable.
#[pyclass(module = "cofferdb", frozen)]
pub(crate) struct Collection {
    pub(super) owner: Arc<owner::Owner>,
    pub(super) collection_id: String,
    pub(super) name: String,
}

#[pymethods]
impl Collection {
    #[pyo3(signature = (*, documents=DefaultArgument::default(), chunks=DefaultArgument::default(), embeddings=DefaultArgument::default(), on_conflict=DefaultArgument::default()),
           text_signature = "($self, *, documents=(), chunks=(), embeddings=(), on_conflict='error')")]
    /// Validate and commit one finite mixed batch without automatic splitting.
    ///
    /// Each group requires a built-in list or tuple. None and generators are
    /// invalid. All records are fixed before the first write. An empty batch
    /// returns zero counts without a commit; warnings retain core diagnostics.
    fn write_batch(
        &self,
        py: Python<'_>,
        documents: DefaultArgument,
        chunks: DefaultArgument,
        embeddings: DefaultArgument,
        on_conflict: DefaultArgument,
    ) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            true,
            |operation| {
                let mut input = operation.input();
                let conflict = arguments::conflict(on_conflict.bind(py), &mut input)
                    .map_err(|error| shape(py, error))?;
                let batch = requests::batch(
                    operation,
                    documents.bind(py),
                    chunks.bind(py),
                    embeddings.bind(py),
                )
                .map_err(|error| shape(py, error))?;
                Ok((batch, conflict))
            },
            |operation, database, (batch, conflict)| {
                operation
                    .write_batch(database, &self.collection_id, batch, conflict)?
                    .into_output(database)
            },
        )
    }

    /// Read complete chunk and document projections in one core read view.
    fn get_chunk(&self, py: Python<'_>, chunk_id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.chunk(py, chunk_id, false)
    }

    /// Read complete content by a chunk external ID scoped to this collection.
    fn get_chunk_by_external_id(
        &self,
        py: Python<'_>,
        chunk_external_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.chunk(py, chunk_external_id, true)
    }

    /// Read one complete document, including its internal ID, without a search.
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
                let id =
                    convert::text(external_id, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(id))
            },
            |operation, database, id| operation.get_document(database, &self.collection_id, id),
        )
    }

    /// Delete by internal document ID using core cascade and visibility rules.
    fn delete_document(
        &self,
        py: Python<'_>,
        document_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.delete(py, document_id, true)
    }

    /// Delete by internal chunk ID using core embedding cascade rules.
    fn delete_chunk(&self, py: Python<'_>, chunk_id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.delete(py, chunk_id, false)
    }

    /// Return independently owned collection and index statistics.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            false,
            |_| Ok(()),
            |operation, database, ()| operation.collection_stats(database, &self.collection_id),
        )
    }

    /// Export this collection from one view, preserving core target safeguards.
    fn export_jsonl(&self, py: Python<'_>, output_dir: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            false,
            |operation| {
                let mut input = operation.input();
                let path =
                    convert::text(output_dir, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(path))
            },
            |operation, database, path| operation.export_jsonl(database, &self.name, path),
        )
    }

    #[pyo3(signature = (text=None, *, embedding=None, filter=None, weights=None, metadata_boost=None, top_k=DefaultArgument::default(), profile=None),
           text_signature = "($self, text=None, *, embedding=None, filter=None, weights=None, metadata_boost=None, top_k=10, profile=None)")]
    #[allow(clippy::too_many_arguments)]
    /// Search with the core's filtering, candidate fusion, scores and explain.
    ///
    /// Optional embedding is an externally supplied finite vector. No provider,
    /// model download or embedding generation runs here. None weights/profile
    /// inherit core defaults; each result includes its stable explain fields.
    /// top_k requires an exact int in 1..min(usize::MAX, 2**53-1), excluding bool.
    /// Results are owned dicts and remain usable after closing the database.
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
        read_output(
            py,
            &self.owner,
            false,
            |operation| {
                arguments::query(
                    arguments::Query {
                        text,
                        embedding,
                        filter,
                        weights,
                        metadata_boost,
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
    fn chunk(&self, py: Python<'_>, id: &Bound<'_, PyAny>, external: bool) -> PyResult<Py<PyAny>> {
        read_output(
            py,
            &self.owner,
            false,
            |operation| {
                let mut input = operation.input();
                let id = convert::text(id, &mut input).map_err(|error| shape(py, error))?;
                Ok(input.finish(id))
            },
            |operation, database, id| {
                operation.get_chunk(database, &self.collection_id, id, external)
            },
        )
    }

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
