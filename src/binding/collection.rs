use super::{Argument, Operation, Output};
use crate::storage::RecordReservation;
use crate::{CollectionConfig, CollectionHandle, Database, Result};

/// Owned names retained through construction of the corresponding host handle.
pub struct CollectionIdentity {
    collection_id: String,
    name: String,
    work: Output,
}

impl CollectionIdentity {
    fn new(handle: CollectionHandle, work: RecordReservation) -> Self {
        let CollectionHandle {
            db: database,
            collection_id,
            name,
        } = handle;
        Self {
            collection_id,
            name,
            work: Output::new(crate::JsonValue::Null, work, Some(database)),
        }
    }

    /// Retain the returned lease until the host handle's ownership handoff.
    pub fn into_parts(self) -> (String, String, Output) {
        (self.collection_id, self.name, self.work)
    }
}

impl Operation {
    pub fn create_collection(
        self,
        database: &Database,
        input: Argument<CollectionConfig>,
    ) -> Result<CollectionIdentity> {
        self.check_input(&input)?;
        let Argument { value, work } = input;
        self.progress.begin();
        database
            .create_collection_retained_in(value, &self.context, Some(work))
            .map(|(handle, work)| CollectionIdentity::new(handle, work))
    }

    pub fn collection(
        self,
        database: &Database,
        name: Argument<String>,
    ) -> Result<CollectionIdentity> {
        self.check_input(&name)?;
        database
            .collection_retained_in(&name.value, &self.context)
            .map(|(handle, work)| CollectionIdentity::new(handle, work))
    }

    pub fn list_collections(self, database: &Database) -> Result<Output> {
        let (value, work) = database.list_collections_json_in(&self.context)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    pub fn collection_stats(self, database: &Database, collection_id: &str) -> Result<Output> {
        let (value, work) = database.collection_stats_json_in(collection_id, &self.context)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    pub fn delete_document(
        self,
        database: &Database,
        collection_id: &str,
        id: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&id)?;
        self.progress.begin();
        database.delete_document_in_collection_in(collection_id, &id.value, &self.context)?;
        Ok(Output::new(
            crate::JsonValue::Null,
            RecordReservation::from(self.context.reservation()),
            Some(database.clone()),
        ))
    }

    pub fn delete_chunk(
        self,
        database: &Database,
        collection_id: &str,
        id: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&id)?;
        self.progress.begin();
        database.delete_chunk_in_collection_in(collection_id, &id.value, &self.context)?;
        Ok(Output::new(
            crate::JsonValue::Null,
            RecordReservation::from(self.context.reservation()),
            Some(database.clone()),
        ))
    }

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
