use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use cofferdb::binding::{Batch, ChunkRecord, Input, Operation, ValueKind};
use cofferdb::{
    ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric, DocumentInput,
    EmbeddingInput, ErrorCode, JsonValue, OnConflict, TokenizerConfig, VectorIndexConfig,
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "coffer-direct-batch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn database(directory: &Directory) -> (Database, cofferdb::CollectionHandle) {
    let db = Database::create(directory.0.join("data.coffer"), DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            tokenizer: TokenizerConfig::default_en(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    (db, collection)
}

fn document(operation: &Operation, batch: &mut Batch, id: &str) {
    let mut input = operation.input();
    let document = DocumentInput {
        external_id: input.text(id).unwrap(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: Vec::new(),
    };
    batch.push_document(input.finish(document)).unwrap();
}

#[test]
fn appended_ordinals_are_unique_against_surviving_chunks_and_reusable_after_delete() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "first");
    document(&operation, &mut batch, "second");
    chunk(&operation, &mut batch, "first", "old", 0);
    operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();

    for policy in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
        let before = fs::read(directory.0.join("data.coffer")).unwrap();
        let operation = Operation::new(&db);
        let observed = operation.observation();
        let mut batch = operation.batch().unwrap();
        chunk(&operation, &mut batch, "first", "new", 0);
        let error = operation
            .write_batch(&db, collection.collection_id(), batch, policy)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "field" && value == "ordinal"));
        assert_eq!(observed.write_outcome().state, "not_committed");
        assert_eq!(fs::read(directory.0.join("data.coffer")).unwrap(), before);
    }

    // A true skip contributes no new ordinal; another parent has its own space.
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    chunk(&operation, &mut batch, "first", "old", 0);
    chunk(&operation, &mut batch, "second", "other", 0);
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Skip)
        .unwrap();
    assert_eq!(summary.counts().chunks_skipped, 1);
    assert_eq!(summary.counts().chunks_inserted, 1);

    let old = collection.get_chunk_by_external_id("old").unwrap();
    collection.delete_chunk(&old.chunk_id).unwrap();
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    chunk(&operation, &mut batch, "first", "new", 0);
    operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    db.close().unwrap();
    let db = Database::open_read_only(directory.0.join("data.coffer")).unwrap();
    let collection = db.collection("notes").unwrap();
    let document = collection.get_document_by_external_id("first").unwrap();
    let chunks = collection.list_chunks(&document.document_id).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_external_id, "new");
    assert_eq!(chunks[0].ordinal, 0);
    db.close().unwrap();
}

#[test]
fn ordinal_replacement_uses_the_final_live_set_including_swaps_and_parent_cascades() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    chunk(&operation, &mut batch, "document", "a", 0);
    chunk(&operation, &mut batch, "document", "b", 1);
    operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();

    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    chunk(&operation, &mut batch, "document", "a", 0);
    assert_eq!(
        operation
            .write_batch(&db, collection.collection_id(), batch, OnConflict::Replace)
            .unwrap()
            .counts()
            .chunks_replaced,
        1
    );
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    chunk(&operation, &mut batch, "document", "a", 1);
    chunk(&operation, &mut batch, "document", "b", 0);
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Replace)
        .unwrap();
    assert_eq!(summary.counts().chunks_replaced, 2);
    assert_eq!(collection.get_chunk_by_external_id("a").unwrap().ordinal, 1);
    assert_eq!(collection.get_chunk_by_external_id("b").unwrap().ordinal, 0);

    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    chunk(&operation, &mut batch, "document", "replacement", 0);
    operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Replace)
        .unwrap();
    let document = collection.get_document_by_external_id("document").unwrap();
    let chunks = collection.list_chunks(&document.document_id).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_external_id, "replacement");
}

fn chunk(operation: &Operation, batch: &mut Batch, document: &str, id: &str, ordinal: usize) {
    let mut input = operation.input();
    let row = ChunkRecord {
        document_external_id: input.text(document).unwrap(),
        chunk: ChunkInput {
            chunk_external_id: input.text(id).unwrap(),
            ordinal,
            text: input.text("needle cache contents").unwrap(),
            metadata: JsonValue::object(),
            source_span: None,
        },
        source_span_provided: false,
    };
    batch.push_chunk(input.finish(row)).unwrap();
}

fn embedding(operation: &Operation, batch: &mut Batch, id: &str, dim: usize) {
    let mut input = operation.input();
    let mut vector = input.vector(dim, ValueKind::JSONL_RECORD).unwrap();
    vector.extend((0..dim).map(|index| f32::from(index == 0)));
    let row = EmbeddingInput {
        chunk_external_id: input.text(id).unwrap(),
        model_name: input.text("model").unwrap(),
        dim,
        vector,
    };
    batch.push_embedding(input.finish(row)).unwrap();
}

#[test]
fn documents_chunks_and_embedding_only_updates_use_one_commit_per_call() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let operation = Operation::new(&db);
    let observation = operation.observation();
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    assert_eq!(summary.counts().documents_inserted, 1);
    assert!(summary.warnings().is_empty());
    assert_eq!(observation.write_outcome().state, "committed");
    assert_eq!(
        observation.write_outcome().confirmed_committed_batches,
        Some(1)
    );
    let stored = collection.get_document_by_external_id("document").unwrap();
    assert!(collection
        .list_chunks(&stored.document_id)
        .unwrap()
        .is_empty());

    let previous_commit = db.stats().unwrap().last_commit_id;
    let operation = Operation::new(&db);
    let observation = operation.observation();
    let mut batch = operation.batch().unwrap();
    chunk(&operation, &mut batch, "document", "chunk", 0);
    embedding(&operation, &mut batch, "chunk", 2);
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    assert_eq!(
        (
            summary.counts().documents_inserted,
            summary.counts().chunks_inserted,
            summary.counts().embeddings_inserted
        ),
        (0, 1, 1)
    );
    assert_eq!(db.stats().unwrap().last_commit_id, previous_commit + 1);
    assert_eq!(
        observation.write_outcome().confirmed_committed_records,
        Some(2)
    );
    let content = collection
        .get_chunk_content_by_external_id("chunk")
        .unwrap();
    assert_eq!(content.document.document_id, stored.document_id);
    assert_eq!(content.chunk.text, "needle cache contents");

    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    embedding(&operation, &mut batch, "chunk", 2);
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Replace)
        .unwrap();
    assert_eq!(summary.counts().embeddings_inserted, 1);
    assert_eq!(summary.counts().embeddings_replaced, 1);
    assert_eq!(collection.stats().unwrap().embedding_count, 1);
    assert_eq!(
        collection
            .search()
            .text("needle")
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    db.close().unwrap();
    let reader = Database::open_read_only(directory.0.join("data.coffer")).unwrap();
    assert_eq!(
        reader
            .collection("notes")
            .unwrap()
            .get_chunk_content_by_external_id("chunk")
            .unwrap()
            .document
            .document_id,
        stored.document_id
    );
    reader.close().unwrap();
}

#[test]
fn late_dimension_and_duplicate_errors_never_commit_the_valid_prefix() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let before = fs::read(directory.0.join("data.coffer")).unwrap();
    let operation = Operation::new(&db);
    let observation = operation.observation();
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    chunk(&operation, &mut batch, "document", "first", 0);
    chunk(&operation, &mut batch, "document", "last", 1);
    embedding(&operation, &mut batch, "first", 2);
    embedding(&operation, &mut batch, "last", 3);
    let error = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .err()
        .unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidEmbeddingDimension);
    assert_eq!(observation.write_outcome().state, "not_committed");
    assert_eq!(
        observation.write_outcome().confirmed_committed_batches,
        Some(0)
    );
    assert_eq!(fs::read(directory.0.join("data.coffer")).unwrap(), before);
    assert_eq!(collection.stats().unwrap().document_count, 0);

    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "duplicate");
    document(&operation, &mut batch, "duplicate");
    chunk(&operation, &mut batch, "missing", "orphan", 0);
    let error = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .err()
        .unwrap();
    assert_eq!(error.code(), ErrorCode::DuplicateExternalId);
    assert_eq!(fs::read(directory.0.join("data.coffer")).unwrap(), before);
    db.close().unwrap();
}

#[test]
fn replace_cascade_cannot_leave_an_embedding_on_a_removed_old_chunk() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    chunk(&operation, &mut batch, "document", "chunk", 0);
    embedding(&operation, &mut batch, "chunk", 2);
    operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    let original_id = collection
        .get_chunk_by_external_id("chunk")
        .unwrap()
        .chunk_id;
    let before = fs::read(directory.0.join("data.coffer")).unwrap();
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    embedding(&operation, &mut batch, "chunk", 2);
    let error = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Replace)
        .err()
        .unwrap();
    assert_eq!(error.code(), ErrorCode::ChunkNotFound);
    assert_eq!(fs::read(directory.0.join("data.coffer")).unwrap(), before);

    let operation = Operation::new(&db);
    let observation = operation.observation();
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    chunk(&operation, &mut batch, "document", "chunk", 0);
    embedding(&operation, &mut batch, "chunk", 2);
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Replace)
        .unwrap();
    assert_eq!(summary.counts().documents_replaced, 1);
    assert_eq!(summary.counts().chunks_replaced, 1);
    assert_eq!(collection.stats().unwrap().chunk_count, 1);
    assert_eq!(collection.stats().unwrap().embedding_count, 1);
    assert_ne!(
        collection
            .get_chunk_by_external_id("chunk")
            .unwrap()
            .chunk_id,
        original_id
    );
    assert_eq!(
        observation.write_outcome().confirmed_committed_batches,
        Some(1)
    );
    db.close().unwrap();
}

#[test]
fn explicit_unknown_span_preserves_metadata_and_warning_and_empty_batch_does_not_write() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    let mut input = operation.input();
    let mut metadata = input.object(1, ValueKind::CHUNK_METADATA).unwrap();
    metadata.insert(
        input
            .json_string("byte_start", ValueKind::CHUNK_METADATA)
            .unwrap(),
        JsonValue::String(
            input
                .json_string("invalid", ValueKind::CHUNK_METADATA)
                .unwrap(),
        ),
    );
    let row = ChunkRecord {
        document_external_id: input.text("document").unwrap(),
        chunk: ChunkInput {
            chunk_external_id: input.text("chunk").unwrap(),
            text: input.text("needle").unwrap(),
            ordinal: 0,
            metadata: JsonValue::Object(metadata),
            source_span: None,
        },
        source_span_provided: true,
    };
    batch.push_chunk(input.finish(row)).unwrap();
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    assert_eq!(summary.warnings().len(), 1);
    assert_eq!(summary.warnings()[0].code, "source_span_metadata_conflict");
    let content = collection
        .get_chunk_content_by_external_id("chunk")
        .unwrap();
    assert_eq!(content.chunk.source_span, None);
    assert_eq!(
        content
            .chunk
            .metadata
            .get("byte_start")
            .and_then(JsonValue::as_str),
        Some("invalid")
    );
    let before = fs::read(directory.0.join("data.coffer")).unwrap();
    let operation = Operation::new(&db);
    let observation = operation.observation();
    let batch = operation.batch().unwrap();
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    assert_eq!(summary.counts().documents_inserted, 0);
    assert_eq!(observation.write_outcome().state, "not_committed");
    assert_eq!(fs::read(directory.0.join("data.coffer")).unwrap(), before);
    db.close().unwrap();
}

#[test]
fn foreign_operation_inputs_are_rejected_before_engine_work() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let first = Operation::new(&db);
    let second = Operation::new(&db);
    let mut batch = first.batch().unwrap();
    let mut input: Input = second.input();
    let row = DocumentInput {
        external_id: input.text("foreign").unwrap(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: Vec::new(),
    };
    assert_eq!(
        batch.push_document(input.finish(row)).unwrap_err().code(),
        ErrorCode::InvalidJsonlSchema
    );
    assert_eq!(
        second
            .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
            .err()
            .unwrap()
            .code(),
        ErrorCode::InvalidJsonlSchema
    );
    assert_eq!(collection.stats().unwrap().document_count, 0);
    db.close().unwrap();
}

#[test]
fn a_true_skip_keeps_ids_model_and_file_bytes_without_an_extra_commit() {
    let directory = Directory::new();
    let (db, collection) = database(&directory);
    let operation = Operation::new(&db);
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    chunk(&operation, &mut batch, "document", "chunk", 0);
    embedding(&operation, &mut batch, "chunk", 2);
    operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    let content = collection
        .get_chunk_content_by_external_id("chunk")
        .unwrap();
    let before = fs::read(directory.0.join("data.coffer")).unwrap();
    let operation = Operation::new(&db);
    let observation = operation.observation();
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    chunk(&operation, &mut batch, "document", "chunk", 0);
    embedding(&operation, &mut batch, "chunk", 2);
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Skip)
        .unwrap();
    assert_eq!(
        (
            summary.counts().documents_skipped,
            summary.counts().chunks_skipped,
            summary.counts().embeddings_skipped
        ),
        (1, 1, 1)
    );
    assert_eq!(
        observation.write_outcome().confirmed_committed_batches,
        Some(0)
    );
    assert_eq!(fs::read(directory.0.join("data.coffer")).unwrap(), before);
    assert_eq!(
        collection
            .get_chunk_content_by_external_id("chunk")
            .unwrap()
            .chunk
            .chunk_id,
        content.chunk.chunk_id
    );
    db.close().unwrap();
}

#[test]
fn direct_batch_above_the_profile_manifest_chunk_count_is_still_one_commit() {
    let directory = Directory::new();
    let db = Database::create(
        directory.0.join("data.coffer"),
        DatabaseConfig {
            resource_profile: cofferdb::ResourceProfile::Constrained,
            ..Default::default()
        },
    )
    .unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            tokenizer: TokenizerConfig::default_en(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    let previous = db.stats().unwrap().last_commit_id;
    let operation = Operation::new(&db);
    let observation = operation.observation();
    let mut batch = operation.batch().unwrap();
    document(&operation, &mut batch, "document");
    for index in 0..400 {
        chunk(
            &operation,
            &mut batch,
            "document",
            &format!("chunk-{index}"),
            index,
        );
    }
    let summary = operation
        .write_batch(&db, collection.collection_id(), batch, OnConflict::Error)
        .unwrap();
    assert_eq!(summary.counts().chunks_inserted, 400);
    assert_eq!(observation.write_outcome().state, "committed");
    assert_eq!(
        observation.write_outcome().confirmed_committed_batches,
        Some(1)
    );
    assert_eq!(
        observation.write_outcome().confirmed_committed_records,
        Some(401)
    );
    assert_eq!(db.stats().unwrap().last_commit_id, previous + 1);
    assert_eq!(collection.stats().unwrap().chunk_count, 400);
    db.close().unwrap();
}
