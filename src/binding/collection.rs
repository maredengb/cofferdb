use super::{Argument, Operation, Output};
use crate::storage::RecordReservation;
use crate::{CollectionConfig, CollectionHandle, Database, Result};

/// Owned names retained through construction of the corresponding host handle.
pub struct CollectionIdentity {
    // Engine-assigned stable identifier moved from the native collection handle for
    // use in subsequent operations.
    collection_id: String,
    // Name returned by the native handle, retained for the host collection and
    // operations such as named export.
    name: String,
    // Lease covering the native identity strings until the host collection takes
    // ownership. Its JSON value is a placeholder.
    work: Output,
}

impl CollectionIdentity {
    // Consume the native handle without duplicating either identity string. Its
    // database reference travels with the output lease so host allocation can
    // still use the same reclamation path.
    fn new(handle: CollectionHandle, work: RecordReservation) -> Self {
        let CollectionHandle {
            db: database,
            collection_id,
            name,
        } = handle;
        Self {
            collection_id,
            name,
            // The strings themselves are the result. A null JSON payload lets the
            // shared Output type carry their native credit, the host-copy
            // allowance and the database reference through handoff.
            work: Output::new(crate::JsonValue::Null, work, Some(database)),
        }
    }

    /// Retain the returned lease until the host handle's ownership handoff.
    /// Returns the stable collection ID, display name and lease, in that order.
    /// Hosts can transfer the two owned strings without looking up the collection
    /// a second time.
    pub fn into_parts(self) -> (String, String, Output) {
        (self.collection_id, self.name, self.work)
    }
}

impl Operation {
    /// Creates a collection from a configuration prepared by this operation and
    /// returns its owned identity for a host handle.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument, an invalid configuration, a
    /// duplicate collection name, or a resource or storage failure. An observation
    /// retained before the call can report confirmed write progress.
    pub fn create_collection(
        self,
        database: &Database,
        input: Argument<CollectionConfig>,
    ) -> Result<CollectionIdentity> {
        self.check_input(&input)?;
        let Argument { value, work } = input;
        // Validation of ownership precedes the start of write tracking. Preserve
        // the consumed configuration's reservation through the engine call and
        // transfer the new identity's credit with the returned handle.
        self.progress.begin();
        database
            // Supply the prepared configuration's existing reservation to the
            // engine. Even on validation failure the owned buffers stay charged
            // until the engine releases them.
            .create_collection_retained_in(value, &self.context, Some(work))
            .map(|(handle, work)| CollectionIdentity::new(handle, work))
    }

    /// Looks up a collection by its prepared name and returns its owned ID, name
    /// and associated lease.
    ///
    /// # Errors
    ///
    /// Returns an error for a name prepared by another operation, a missing
    /// collection, or a failure to read or retain the identity.
    pub fn collection(
        self,
        database: &Database,
        name: Argument<String>,
    ) -> Result<CollectionIdentity> {
        self.check_input(&name)?;
        database
            // Borrow the input as a lookup key. The engine returns independently
            // owned catalog spellings under a new reservation, so the host handle
            // outlives the temporary name argument.
            .collection_retained_in(&name.value, &self.context)
            .map(|(handle, work)| CollectionIdentity::new(handle, work))
    }

    /// Lists collections under this operation's retained read context. Keep the
    /// returned Output alive while projecting the JSON array into host-owned
    /// values.
    pub fn list_collections(self, database: &Database) -> Result<Output> {
        let (value, work) = database.list_collections_json_in(&self.context)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    /// Reads statistics for the existing internal collection ID and returns a
    /// retained output. Missing collections and read/admission failures are
    /// reported by the database.
    pub fn collection_stats(self, database: &Database, collection_id: &str) -> Result<Output> {
        let (value, work) = database.collection_stats_json_in(collection_id, &self.context)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    /// Deletes the document with this internal ID from the selected collection.
    /// The operation consumes its prepared argument.
    ///
    /// # Errors
    ///
    /// Returns an error if the argument belongs to another operation, the document
    /// cannot be resolved in the collection, or the native write fails. A write
    /// observation can be retained before the call.
    pub fn delete_document(
        self,
        database: &Database,
        collection_id: &str,
        id: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&id)?;
        // Record the attempt only after checking ownership of the prepared input.
        // Actual committed progress comes from the append path.
        self.progress.begin();
        database.delete_document_in_collection_in(collection_id, &id.value, &self.context)?;
        // Use JSON null for a completed mutation with no result payload. The
        // resulting Output still participates in the host ownership handoff.
        Ok(Output::new(
            crate::JsonValue::Null,
            RecordReservation::from(self.context.reservation()),
            Some(database.clone()),
        ))
    }

    /// Deletes the chunk with this internal ID from the selected collection.
    ///
    /// # Errors
    ///
    /// Returns an error for foreign prepared input, an unknown chunk in this
    /// collection, or a native write failure.
    pub fn delete_chunk(
        self,
        database: &Database,
        collection_id: &str,
        id: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&id)?;
        self.progress.begin();
        // Pass the same context used for argument admission through to the native
        // delete. This preserves the budget and operation's write observation
        // together.
        database.delete_chunk_in_collection_in(collection_id, &id.value, &self.context)?;
        Ok(Output::new(
            crate::JsonValue::Null,
            RecordReservation::from(self.context.reservation()),
            Some(database.clone()),
        ))
    }

    /// Converts one prepared JSON value into a tokenizer configuration for this
    /// operation. A foreign argument or invalid tokenizer definition returns an
    /// error.
    pub fn tokenizer(&self, value: Argument<crate::JsonValue>) -> Result<crate::TokenizerConfig> {
        self.check_input(&value)?;
        crate::TokenizerConfig::from_validated_in(&value.value, &self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DatabaseConfig, DistanceMetric, ErrorCode, TokenizerConfig, VectorIndexConfig};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn config(name: String) -> CollectionConfig {
        CollectionConfig {
            name,
            tokenizer: TokenizerConfig::default_en(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        }
    }

    #[test]
    fn failed_collection_creation_keeps_owned_string_credit_until_actual_free() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        for preprocessed in [false, true] {
            let path = std::env::temp_dir().join(format!(
                "coffer-collection-credit-{}-{}.coffer",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            let name = "n".repeat(8193);
            db.create_collection(config(name.clone())).unwrap();
            let before = std::fs::read(&path).unwrap();
            let operation = Operation::new(&db);
            let context = operation.context.clone();
            let baseline = context.available_workspace_bytes();
            let mut input = operation.input();
            let value = config(if preprocessed {
                input.text(&name).unwrap()
            } else {
                name.clone()
            });
            let pointer = value.name.as_ptr();
            let bytes = value.name.capacity();
            let (error, available) =
                crate::test_alloc::available_before_deallocation(pointer, &context, || {
                    if preprocessed {
                        operation
                            .create_collection(&db, input.finish(value))
                            .err()
                            .unwrap()
                    } else {
                        db.create_collection(value).err().unwrap()
                    }
                });
            assert_eq!(error.code(), ErrorCode::CollectionAlreadyExists);
            assert_eq!(std::fs::read(&path).unwrap(), before);
            assert_eq!(context.available_workspace_bytes(), baseline);
            db.close().unwrap();
            std::fs::remove_file(path).unwrap();
            assert!(available.is_some_and(|available| available <= baseline - bytes),
                "preprocessed={preprocessed}: allocation still live, but its credit was returned: {available:?}");
        }
    }

    #[test]
    fn invalid_preprocessed_config_keeps_input_credit_through_validation_failure() {
        let path = std::env::temp_dir().join(format!(
            "coffer-invalid-config-credit-{}.coffer",
            std::process::id()
        ));
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        let before = std::fs::read(&path).unwrap();
        let operation = Operation::new(&db);
        let context = operation.context.clone();
        let baseline = context.available_workspace_bytes();
        let mut input = operation.input();
        let mut value = config(input.text(&"n".repeat(8193)).unwrap());
        value.embedding_dim = 0;
        let pointer = value.name.as_ptr();
        let bytes = value.name.capacity();
        let argument = input.finish(value);
        let (error, available) =
            crate::test_alloc::available_before_deallocation(pointer, &context, || {
                operation.create_collection(&db, argument).err().unwrap()
            });
        assert_eq!(error.code(), ErrorCode::InvalidEmbeddingDimension);
        assert!(available.is_some_and(|available| available <= baseline - bytes));
        assert_eq!(context.available_workspace_bytes(), baseline);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        db.close().unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
