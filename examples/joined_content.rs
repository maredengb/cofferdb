//! Complete-content reads and logical restore using the Rust 0.1.1 API.
//! First create the input with the programming_guide example.
//! Usage: joined_content <notes.coffer> <new-restored.coffer> <new-export-dir>

use cofferdb::{
    ChunkContent, Database, DatabaseConfig, DatabaseStats, Error, ErrorCode, ImportManifest,
    OnConflict, ProfileParams, ResourceProfile,
};
use std::path::Path;

fn equal_content(expected: &ChunkContent, actual: &ChunkContent) {
    assert_eq!(actual.chunk.text.as_bytes(), expected.chunk.text.as_bytes());
    assert_eq!(actual.chunk.text_hash, expected.chunk.text_hash);
    assert_eq!(actual.chunk.metadata, expected.chunk.metadata);
    assert_eq!(actual.chunk.source_span, expected.chunk.source_span);
    assert_eq!(actual.document.external_id, expected.document.external_id);
    assert_eq!(actual.document.metadata, expected.document.metadata);
    assert_eq!(actual.document.source_uri, expected.document.source_uri);
}

fn run(source: &Path, restored: &Path, export: &Path) -> cofferdb::Result<()> {
    let db = Database::open_read_only(source)?;
    let stats: DatabaseStats = db.stats()?;
    assert_eq!((stats.document_count, stats.chunk_count), (1, 1));
    let notes = db.collection("notes")?;
    let owned = notes.get_chunk_content_by_external_id("entry-1#leaf")?;
    assert_eq!(owned.chunk.document_id, owned.document.document_id);
    assert!(owned.chunk.text.len() > 240);
    let by_id = notes.get_chunk_content(&owned.chunk.chunk_id)?;
    equal_content(&owned, &by_id);
    let params: ProfileParams = ResourceProfile::Balanced.params();
    assert_eq!(params.bm25_candidates, 200);
    let exported = db.export_collection_jsonl("notes", export)?;
    assert_eq!(
        (exported.documents, exported.chunks, exported.embeddings),
        (1, 1, 1)
    );
    drop(notes);
    db.close()?;

    let manifest = export.join(&exported.manifest);
    assert_eq!(ImportManifest::load(&manifest)?.collection, "notes");
    let db = Database::create(restored, DatabaseConfig::default())?;
    let imported = db.import_manifest(&manifest, OnConflict::Error)?;
    assert_eq!(
        (imported.documents_inserted, imported.chunks_inserted),
        (1, 1)
    );
    assert_eq!(imported.embeddings_inserted, 1);
    let notes = db.collection("notes")?;
    equal_content(
        &owned,
        &notes.get_chunk_content_by_external_id("entry-1#leaf")?,
    );
    let skipped = db.import_manifest(&manifest, OnConflict::Skip)?;
    assert_eq!((skipped.documents_skipped, skipped.chunks_skipped), (1, 1));
    let replaced = db.import_manifest(&manifest, OnConflict::Replace)?;
    assert_eq!(replaced.documents_replaced, 1);
    equal_content(
        &owned,
        &notes.get_chunk_content_by_external_id("entry-1#leaf")?,
    );
    let error = db
        .import_manifest(&manifest, OnConflict::Error)
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::DuplicateExternalId);
    let stored = notes.get_chunk_content_by_external_id("entry-1#leaf")?;
    notes.delete_chunk(&stored.chunk.chunk_id)?;
    assert_eq!(
        notes
            .get_chunk_content(&stored.chunk.chunk_id)
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    notes.delete_document(&stored.document.document_id)?;
    let stats = notes.stats()?;
    assert_eq!(
        (
            stats.document_count,
            stats.chunk_count,
            stats.embedding_count
        ),
        (0, 0, 0)
    );
    drop(notes);
    db.close()?;
    // Owned content remains complete after both owners are closed. No original
    // source document or vector provider was used during reads or restore.
    assert_eq!(owned.chunk.text.len(), 1024);
    println!("joined content, export, reimport, conflict policies and deletion verified");
    Ok(())
}

fn main() -> cofferdb::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err(Error::new(
            ErrorCode::CliUsage,
            "usage: joined_content <notes.coffer> <new-restored.coffer> <new-export-dir>",
        ));
    }
    run(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    )
}
