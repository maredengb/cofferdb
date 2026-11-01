//! A complete Rust lifecycle using APIs available in both 0.1.0 and 0.1.1.
//! Usage: programming_guide <create|read> <database.coffer>
//! The small vectors are explicit synthetic data, not a language model.

use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric,
    DocumentInput, EmbeddingInput, Error, ErrorCode, QueryEmbeddingInput, ResourceProfile,
    SearchWeights, SourceSpan, VectorIndexConfig,
};
use std::path::Path;

fn text() -> String {
    "Jade leaves grow in the north greenhouse after the spring rain.\n".repeat(16)
}

fn embedding() -> QueryEmbeddingInput {
    QueryEmbeddingInput {
        model_name: Some("example-2d".into()),
        dim: 2,
        vector: vec![1.0, 0.0],
    }
}

fn create(path: &Path) -> cofferdb::Result<()> {
    let db = Database::create(path, DatabaseConfig::default())?;
    let notes = db.create_collection(CollectionConfig {
        name: "notes".into(),
        tokenizer: Default::default(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: Some("example-2d".into()),
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    })?;
    let text = text();
    notes.insert_document(DocumentInput {
        external_id: "entry-1".into(),
        title: Some("Greenhouse log".into()),
        source_uri: Some("field-notes://greenhouse/entry-1".into()),
        metadata: parse_json(r#"{"site":"north","public":true}"#)?,
        chunks: vec![ChunkInput {
            chunk_external_id: "entry-1#leaf".into(),
            ordinal: 0,
            metadata: parse_json(r#"{"section":"leaf"}"#)?,
            source_span: Some(SourceSpan {
                start: Some(0),
                end: Some(text.len() as u64),
                line_start: Some(1),
                line_end: Some(16),
            }),
            text,
        }],
    })?;
    notes.put_embeddings(vec![EmbeddingInput {
        chunk_external_id: "entry-1#leaf".into(),
        model_name: "example-2d".into(),
        dim: 2,
        vector: vec![1.0, 0.0],
    }])?;
    drop(notes);
    db.close()
}

fn read(path: &Path) -> cofferdb::Result<()> {
    let db = Database::open_read_only(path)?;
    let notes = db.collection("notes")?;
    let lexical = notes.search().text("jade leaves").top_k(1).execute()?;
    let semantic = notes
        .search()
        .query_embedding(embedding())
        .top_k(1)
        .execute()?;
    let hybrid = notes
        .search()
        .text("jade leaves")
        .query_embedding(embedding())
        .filter(parse_json(r#"{"doc.site":"north"}"#)?)
        .weights(SearchWeights::new(0.7, 0.3, 0.0)?)
        .resource_profile(ResourceProfile::Balanced)
        .top_k(1)
        .execute()?;
    for result in [&lexical, &semantic, &hybrid] {
        assert_eq!(result.results.len(), 1);
        let hit = &result.results[0];
        assert_eq!(hit.chunk_external_id, "entry-1#leaf");
        assert!(hit.final_score.is_finite());
        assert!(hit.explain.get("weights").is_some());
        assert!(hit.explain.get("candidate_sources").is_some());
    }
    let chunk = notes.get_chunk(&hybrid.results[0].chunk_id)?;
    let document = notes.get_document(&chunk.document_id)?;
    assert_eq!(chunk.text.as_bytes(), text().as_bytes());
    assert!(chunk.text.len() > hybrid.results[0].snippet.len());
    assert_eq!(
        chunk.source_span.as_ref().unwrap().end,
        Some(chunk.text.len() as u64)
    );
    assert_eq!(document.external_id, "entry-1");
    assert_eq!(
        document.metadata,
        parse_json(r#"{"site":"north","public":true}"#)?
    );
    assert_eq!(
        notes
            .get_chunk_by_external_id("missing")
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    assert_eq!(chunk.metadata, parse_json(r#"{"section":"leaf"}"#)?);
    assert_eq!(
        SearchWeights::new(-0.1, 1.1, 0.0).unwrap_err().code(),
        ErrorCode::InvalidSearchWeights
    );
    let expected = notes.get_chunk_by_external_id("entry-1#leaf")?;
    assert_eq!(expected.text, chunk.text);
    println!("{}", hybrid.to_json().to_compact_string());
    drop(notes);
    db.close()?;
    // Owned values remain valid after close. The two reads above share the
    // read-only handle's fixed view; no original source document was opened.
    assert_eq!(chunk.text, text());
    assert_eq!(document.title.as_deref(), Some("Greenhouse log"));
    Ok(())
}

fn main() -> cofferdb::Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let usage = || {
        Error::new(
            ErrorCode::CliUsage,
            "usage: programming_guide <create|read> <database.coffer>",
        )
    };
    let action = arguments.next().ok_or_else(usage)?;
    let path = arguments.next().ok_or_else(usage)?;
    if arguments.next().is_some() {
        return Err(usage());
    }
    let path = Path::new(&path);
    if action == "create" {
        create(path)?;
    } else if action != "read" {
        return Err(usage());
    }
    read(path)
}
