use cofferdb::binding::Operation;
use cofferdb::{
    ChunkInput, CollectionConfig, DatabaseConfig, DatabaseOpenOptions, DistanceMetric,
    DocumentInput, Error, ErrorCode, JsonValue, SourceSpan, TokenizerConfig, VectorIndexConfig,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "coffer-projection-{}-{}",
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

#[test]
fn standalone_open_create_use_owned_inputs_and_shared_read_projections() {
    let directory = Directory::new();
    let path = directory.0.join("data-\u{4e2d}.coffer");
    let operation = Operation::standalone();
    let mut input = operation.input();
    let path_text = input
        .text_from_chars(path.to_str().unwrap().chars())
        .unwrap();
    let db = operation
        .create(input.finish((path_text, DatabaseConfig::default())))
        .unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: TokenizerConfig::default_en(),
        })
        .unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "document".into(),
            title: Some("title".into()),
            source_uri: Some("urn:sample:notes".into()),
            metadata: cofferdb::parse_json("{\"visible\":true}").unwrap(),
            chunks: vec![ChunkInput {
                chunk_external_id: "chunk".into(),
                ordinal: 0,
                text: "retained text\0\u{e9}\u{4e2d}\u{1f642}".repeat(512),
                metadata: cofferdb::parse_json("{\"label\":\"example\"}").unwrap(),
                source_span: Some(SourceSpan {
                    start: Some(9_007_199_254_740_992),
                    end: Some(u64::MAX),
                    line_start: Some(1),
                    line_end: Some(2),
                }),
            }],
        })
        .unwrap();
    let expected = collection
        .get_chunk_content_by_external_id("chunk")
        .unwrap();
    let operation = Operation::new(&db);
    let mut input = operation.input();
    let id = input.text("chunk").unwrap();
    let mut output = operation
        .get_chunk(&db, collection.collection_id(), input.finish(id), true)
        .unwrap();
    assert_eq!(output.value(), &expected.to_json());
    output.reserve_host(32 * 1024).unwrap();
    assert_eq!(
        output
            .value()
            .get("chunk")
            .unwrap()
            .get("source_span")
            .unwrap()
            .get("end")
            .and_then(JsonValue::as_str),
        Some("18446744073709551615")
    );

    let operation = Operation::new(&db);
    let mut input = operation.input();
    let id = input.text("document").unwrap();
    let document = operation
        .get_document(&db, collection.collection_id(), input.finish(id))
        .unwrap();
    assert_eq!(document.value(), &expected.document.to_json());
    assert_eq!(
        Operation::new(&db).database_stats(&db).unwrap().value(),
        &db.stats().unwrap().to_json()
    );
    assert_eq!(
        Operation::new(&db).index_stats(&db).unwrap().value(),
        &db.index_stats().unwrap().to_json()
    );
    db.close().unwrap();
    assert_eq!(
        output
            .value()
            .get("chunk")
            .unwrap()
            .get("text")
            .and_then(JsonValue::as_str),
        Some(expected.chunk.text.as_str())
    );
    output.clear_native();
    assert!(output.value().is_null());
    drop(output);
    drop(document);

    let operation = Operation::standalone();
    let mut input = operation.input();
    let path = input.text(path.to_str().unwrap()).unwrap();
    let db = operation
        .open(input.finish((
            path,
            DatabaseOpenOptions {
                read_only: true,
                ..Default::default()
            },
        )))
        .unwrap();
    assert_eq!(
        db.collection("notes")
            .unwrap()
            .get_chunk_by_external_id("chunk")
            .unwrap()
            .text,
        expected.chunk.text
    );
    db.close().unwrap();
}

#[test]
fn error_projection_retains_exact_fields_null_actions_and_numeric_details() {
    for code in [
        ErrorCode::InvalidJsonlSchema,
        ErrorCode::ResourceExhausted,
        ErrorCode::RecoveryError,
        ErrorCode::PermissionDenied,
        ErrorCode::LockContention,
    ] {
        let error = Error::new(code, "diagnostic \u{4e2d} \u{1f642}")
            .with_detail("key", "old")
            .with_detail("key", "new");
        let expected = error.to_json();
        let output = Operation::standalone().error(error).unwrap();
        assert_eq!(output.value(), &expected);
        assert_eq!(output.value().as_object().unwrap().len(), 5);
        assert_eq!(
            output
                .value()
                .get("detail")
                .unwrap()
                .get("key")
                .and_then(JsonValue::as_str),
            Some("new")
        );
    }
    let output = Operation::standalone()
        .error(Error::new(ErrorCode::ChunkNotFound, "missing chunk"))
        .unwrap();
    assert!(output.value().get("suggested_action").unwrap().is_null());
}

#[test]
fn foreign_bootstrap_input_never_creates_a_file() {
    let directory = Directory::new();
    let path = directory.0.join("not-created.coffer");
    let first = Operation::standalone();
    let second = Operation::standalone();
    let mut input = first.input();
    let text = input.text(path.to_str().unwrap()).unwrap();
    let error = second
        .create(input.finish((text, DatabaseConfig::default())))
        .err()
        .unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
    assert!(!path.exists());
}
