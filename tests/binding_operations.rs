use cofferdb::binding::{ChunkRecord, Operation, SearchInput, ValueKind};
use cofferdb::{
    ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric, DocumentInput,
    EmbeddingInput, ErrorCode, JsonValue, OnConflict, QueryEmbeddingInput, ResourceProfile,
    TokenizerConfig, VectorIndexConfig,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "coffer-binding-operations-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn create(db: &Database) -> (String, String, cofferdb::binding::Output) {
    let operation = Operation::new(db);
    let mut input = operation.input();
    let name = input.text("notes").unwrap();
    let config = CollectionConfig {
        name,
        tokenizer: TokenizerConfig::default_en(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    };
    let observed = operation.observation();
    let identity = operation
        .create_collection(db, input.finish(config))
        .unwrap();
    assert_eq!(
        observed.write_outcome().confirmed_committed_batches,
        Some(1)
    );
    identity.into_parts()
}

fn seed(db: &Database) {
    let collection = db.collection("notes").unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "document".into(),
            title: Some("Field notes".into()),
            source_uri: None,
            metadata: cofferdb::parse_json(r#"{"visible":true}"#).unwrap(),
            chunks: ["Jade leaves near the river.", "Granite is a durable stone."]
                .into_iter()
                .enumerate()
                .map(|(ordinal, text)| ChunkInput {
                    chunk_external_id: format!("part-{ordinal}"),
                    ordinal,
                    text: text.into(),
                    metadata: cofferdb::parse_json(r#"{"category":"sample"}"#).unwrap(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    collection
        .put_embeddings(
            [vec![1.0, 0.0], vec![0.0, 1.0]]
                .into_iter()
                .enumerate()
                .map(|(index, vector)| EmbeddingInput {
                    chunk_external_id: format!("part-{index}"),
                    model_name: "sample-2d".into(),
                    dim: 2,
                    vector,
                })
                .collect(),
        )
        .unwrap();
}

#[test]
fn retained_collection_catalog_stats_and_query_use_the_same_core_projections() {
    let directory = Directory::new();
    let db = Database::create(directory.0.join("data.coffer"), DatabaseConfig::default()).unwrap();
    let (id, name, mut lease) = create(&db);
    lease.reserve_host(256).unwrap();
    assert_eq!(name, "notes");
    seed(&db);
    let collection = db.collection("notes").unwrap();
    let catalog = Operation::new(&db).list_collections(&db).unwrap();
    assert_eq!(
        catalog.value(),
        &JsonValue::Array(
            db.list_collections()
                .unwrap()
                .into_iter()
                .map(|info| info.to_json())
                .collect()
        )
    );
    assert_eq!(
        Operation::new(&db)
            .collection_stats(&db, &id)
            .unwrap()
            .value(),
        &collection.stats().unwrap().to_json()
    );

    for profile in [
        ResourceProfile::Expanded,
        ResourceProfile::Balanced,
        ResourceProfile::Constrained,
    ] {
        for (text, vector) in [(true, false), (false, true), (true, true)] {
            let operation = Operation::new(&db);
            let mut input = operation.input();
            let text_input = text.then(|| input.text("jade").unwrap());
            let embedding = vector.then(|| {
                let mut values = input.vector(2, ValueKind::QUERY_EMBEDDING).unwrap();
                values.extend([1.0, 0.0]);
                QueryEmbeddingInput {
                    vector: values,
                    dim: 2,
                    model_name: None,
                }
            });
            let request = SearchInput {
                text: text_input,
                embedding,
                filter: JsonValue::object(),
                weights: None,
                metadata_boost: None,
                top_k: 2,
                profile: Some(profile),
            };
            let actual = operation.search(&db, &id, input.finish(request)).unwrap();
            let mut builder = collection.search().top_k(2).resource_profile(profile);
            if text {
                builder = builder.text("jade");
            }
            if vector {
                builder = builder.vector(vec![1.0, 0.0]);
            }
            assert_eq!(actual.value(), &builder.execute().unwrap().to_json());
        }
    }
    let operation = Operation::new(&db);
    let mut input = operation.input();
    let selected_name = input.text("notes").unwrap();
    let (selected_id, _, selected_lease) = operation
        .collection(&db, input.finish(selected_name))
        .unwrap()
        .into_parts();
    assert_eq!(selected_id, id);
    drop(selected_lease);
    drop(lease);
    db.close().unwrap();
    assert_eq!(catalog.value().as_array().unwrap().len(), 1);
}

#[test]
fn consumed_batch_warning_export_and_import_keep_core_fields_and_commit_observation() {
    let directory = Directory::new();
    let db = Database::create(directory.0.join("data.coffer"), DatabaseConfig::default()).unwrap();
    let (id, _, lease) = create(&db);
    drop(lease);
    let operation = Operation::new(&db);
    let observation = operation.observation();
    let mut batch = operation.batch().unwrap();
    let mut input = operation.input();
    let document = DocumentInput {
        external_id: input.text("document").unwrap(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: Vec::new(),
    };
    batch.push_document(input.finish(document)).unwrap();
    let mut input = operation.input();
    let mut metadata = input.object(2, ValueKind::CHUNK_METADATA).unwrap();
    for (key, value) in [("byte_start", 1.0), ("byte_end", 2.0)] {
        metadata.insert(
            input.json_string(key, ValueKind::CHUNK_METADATA).unwrap(),
            JsonValue::Number(value),
        );
    }
    let row = ChunkRecord {
        document_external_id: input.text("document").unwrap(),
        chunk: ChunkInput {
            chunk_external_id: input.text("part").unwrap(),
            ordinal: 0,
            text: input.text("Unknown source position.").unwrap(),
            metadata: JsonValue::Object(metadata),
            source_span: None,
        },
        source_span_provided: true,
    };
    batch.push_chunk(input.finish(row)).unwrap();
    let result = operation
        .write_batch(&db, &id, batch, OnConflict::Error)
        .unwrap();
    let expected = result.counts().to_json();
    let summary = result.into_output(&db).unwrap();
    for (key, value) in expected.as_object().unwrap() {
        assert_eq!(summary.value().get(key), Some(value));
    }
    assert_eq!(
        summary.value().get("warnings").unwrap().as_array().unwrap()[0]
            .get("code")
            .and_then(JsonValue::as_str),
        Some("source_span_metadata_conflict")
    );
    assert_eq!(
        observation.write_outcome().confirmed_committed_records,
        Some(2)
    );

    let operation = Operation::new(&db);
    let mut input = operation.input();
    let path = input
        .text(directory.0.join("export").to_str().unwrap())
        .unwrap();
    let exported = operation
        .export_jsonl(&db, "notes", input.finish(path))
        .unwrap();
    let reader = Database::create(
        directory.0.join("reimport.coffer"),
        DatabaseConfig::default(),
    )
    .unwrap();
    let operation = Operation::new(&reader);
    let observed = operation.observation();
    let mut input = operation.input();
    let path = input
        .text(
            directory
                .0
                .join("export/export_manifest.json")
                .to_str()
                .unwrap(),
        )
        .unwrap();
    let imported = operation
        .import_jsonl_output(&reader, input.finish(path), OnConflict::Error)
        .unwrap();
    assert_eq!(
        observed.write_outcome().confirmed_committed_batches,
        Some(1)
    );
    // The new collection is an additional committed core record during import.
    assert_eq!(
        observed.write_outcome().confirmed_committed_records,
        Some(3)
    );
    assert_eq!(
        imported
            .value()
            .get("chunks_inserted")
            .and_then(JsonValue::as_f64),
        Some(1.0)
    );
    assert_eq!(
        reader
            .collection("notes")
            .unwrap()
            .get_chunk_by_external_id("part")
            .unwrap()
            .text,
        "Unknown source position."
    );
    assert!(exported.value().get("manifest").is_some());
    reader.close().unwrap();
    db.close().unwrap();
    assert_eq!(
        summary
            .value()
            .get("warnings")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn foreign_arguments_and_invalid_weights_are_rejected_without_native_writes() {
    let directory = Directory::new();
    let db = Database::create(directory.0.join("data.coffer"), DatabaseConfig::default()).unwrap();
    let operation = Operation::new(&db);
    let foreign = Operation::new(&db);
    let mut input = foreign.input();
    let name = input.text("missing").unwrap();
    let before = std::fs::read(directory.0.join("data.coffer")).unwrap();
    let error = operation.collection(&db, input.finish(name)).err().unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(
        std::fs::read(directory.0.join("data.coffer")).unwrap(),
        before
    );
    for (text, expected) in [
        (r#"{"bm25":0.5,"vector":0.5,"metadata":0}"#, (0.5, 0.5, 0.0)),
        (r#"{"bm25":1,"vector":0,"metadata":0}"#, (1.0, 0.0, 0.0)),
    ] {
        let value = cofferdb::parse_json(text).unwrap();
        let operation = Operation::new(&db);
        let input = operation.input();
        let actual = operation.weights(input.finish(value)).unwrap();
        assert_eq!((actual.bm25, actual.vector, actual.metadata), expected);
    }
    let operation = Operation::new(&db);
    let input = operation.input();
    let value = cofferdb::parse_json(r#"{"bm25":-1,"vector":2,"metadata":0}"#).unwrap();
    assert_eq!(
        operation.weights(input.finish(value)).unwrap_err().code(),
        ErrorCode::InvalidSearchWeights
    );
    db.close().unwrap();
}
