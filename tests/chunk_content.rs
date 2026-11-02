use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric,
    DocumentInput, ErrorCode, JsonValue, SourceSpan, TokenizerConfig, VectorIndexConfig,
};

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "cofferdb-content-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        Self(directory)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn config(name: &str) -> CollectionConfig {
    CollectionConfig {
        name: name.into(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
        tokenizer: TokenizerConfig::default_en(),
    }
}

fn document(text: &str) -> DocumentInput {
    let offset = 9_007_199_254_740_993;
    DocumentInput {
        external_id: "source".into(),
        title: Some("Cache source".into()),
        source_uri: Some("file:///unavailable-source/cache.rs".into()),
        metadata: parse_json(r#"{"language":"rust","revision":"fixed"}"#).unwrap(),
        chunks: vec![ChunkInput {
            chunk_external_id: "cache-body".into(),
            text: text.into(),
            ordinal: 0,
            metadata: parse_json(r#"{"symbol":"Cache","unicode":"缓存"}"#).unwrap(),
            source_span: Some(SourceSpan {
                start: Some(offset),
                end: Some(offset + text.len() as u64),
                line_start: Some(1),
                line_end: Some(2_000),
            }),
        }],
    }
}

#[test]
fn complete_content_is_owned_source_independent_and_survives_reopen() {
    let directory = Directory::new();
    let path = directory.0.join("content.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = db.create_collection(config("code")).unwrap();
    let text = "cache recovery 缓存\n".repeat(2_000);
    let input = document(&text);
    collection.insert_document(input.clone()).unwrap();
    let search = collection.search().text("cache").execute().unwrap();
    let chunk_id = search.results[0].chunk_id.clone();
    let mut content = collection.get_chunk_content(&chunk_id).unwrap();
    assert_eq!(content.chunk.text.as_bytes(), text.as_bytes());
    assert_eq!(content.chunk.document_id, content.document.document_id);
    assert_eq!(content.chunk.collection_id, content.document.collection_id);
    assert_eq!(content.document.metadata, input.metadata);
    assert_eq!(content.document.source_uri, input.source_uri);
    assert_eq!(content.chunk.metadata, input.chunks[0].metadata);
    assert_eq!(content.chunk.source_span, input.chunks[0].source_span);
    let projection = content.to_json();
    assert_eq!(
        projection
            .get("chunk")
            .unwrap()
            .get("source_span")
            .unwrap()
            .get("start")
            .and_then(JsonValue::as_str),
        Some("9007199254740993")
    );
    content.chunk.text.clear();
    content.document.metadata = JsonValue::Null;
    let unmodified = collection
        .get_chunk_content_by_external_id("cache-body")
        .unwrap();
    assert_eq!(unmodified.chunk.text, text);
    assert_eq!(unmodified.document.metadata, input.metadata);
    db.close().unwrap();
    assert_eq!(unmodified.chunk.text, text);
    let reader = Database::open_read_only(&path).unwrap();
    let restored = reader
        .collection("code")
        .unwrap()
        .get_chunk_content(&chunk_id)
        .unwrap();
    assert_eq!(restored.chunk.text, text);
    assert_eq!(restored.chunk.text_hash, unmodified.chunk.text_hash);
    assert_eq!(
        restored.document.document_id,
        unmodified.document.document_id
    );
    reader.close().unwrap();
}

#[test]
fn content_lookup_is_collection_scoped_and_deletion_keeps_reader_view() {
    let directory = Directory::new();
    let path = directory.0.join("content.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let first = db.create_collection(config("first")).unwrap();
    let second = db.create_collection(config("second")).unwrap();
    first.insert_document(document("cache first")).unwrap();
    second.insert_document(document("cache second")).unwrap();
    let content = first
        .get_chunk_content_by_external_id("cache-body")
        .unwrap();
    assert_eq!(content.chunk.text, "cache first");
    assert_eq!(
        second
            .get_chunk_content_by_external_id("cache-body")
            .unwrap()
            .chunk
            .text,
        "cache second"
    );
    assert_eq!(
        second
            .get_chunk_content(&content.chunk.chunk_id)
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    let reader = Database::open_read_only(&path).unwrap();
    first
        .delete_document(&content.document.document_id)
        .unwrap();
    assert_eq!(
        first
            .get_chunk_content(&content.chunk.chunk_id)
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    assert_eq!(
        first
            .get_chunk_content_by_external_id("cache-body")
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    let pinned = reader
        .collection("first")
        .unwrap()
        .get_chunk_content(&content.chunk.chunk_id)
        .unwrap();
    assert_eq!(pinned.chunk.text, "cache first");
    assert_eq!(pinned.document.document_id, content.document.document_id);
    reader.close().unwrap();
    db.close().unwrap();
}
