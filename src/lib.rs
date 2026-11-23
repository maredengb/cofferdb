//! An embedded database for full-text and vector retrieval over document chunks.
//!
//! Use [Database] to create or open a local file. A [CollectionHandle] lets you
//! store documents and embeddings, search them and read the text behind a hit.
//! An open writer holds exclusive writer admission across processes. A read-only
//! connection sees a fixed view. Close it and reopen to observe later commits.
//!
//! See the matching [Rust programming guide](https://docs.rs/crate/cofferdb/0.2.1/source/docs/rust.md)
//! and [backup and recovery guide](https://docs.rs/crate/cofferdb/0.2.1/source/docs/backup.md).
//!
//! # Examples
//!
//! ## Write, search, compact and restore
//!
//! Rebuild and compact through the active writer. A path-based snapshot acquires
//! its own writer lock, so close the writer before copying to an unused filename.
//! Opening the copy read-only lets you verify application data as well as file integrity.
//!
//! ```
//! use cofferdb::{CheckOptions, ChunkInput, CollectionConfig, CompactOptions, Database,
//!     DatabaseConfig, DistanceMetric, DocumentInput, JsonValue, VectorIndexConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let dir = std::env::temp_dir().join(format!("cofferdb-doc-{}", std::process::id()));
//! # std::fs::create_dir(&dir)?;
//! # let path = dir.join("notes.coffer");
//! # let backup = dir.join("notes-backup.coffer");
//! let db = Database::create(&path, DatabaseConfig::default())?;
//! let notes = db.create_collection(CollectionConfig {
//!     name: "notes".into(),
//!     tokenizer: Default::default(),
//!     embedding_dim: 2,
//!     distance_metric: DistanceMetric::Cosine,
//!     embedding_model: None,
//!     vectors_normalized: false,
//!     vector_index: VectorIndexConfig::Exact,
//! })?;
//! notes.insert_document(DocumentInput {
//!     external_id: "first".into(), title: None, source_uri: None, metadata: JsonValue::object(),
//!     chunks: vec![ChunkInput {
//!         chunk_external_id: "first#1".into(), ordinal: 0, text: "River otters".into(),
//!         metadata: JsonValue::object(), source_span: None,
//!     }],
//! })?;
//! let export_dir = dir.join("export");
//! let exported = db.export_collection_jsonl("notes", &export_dir)?;
//! let restored = Database::create(dir.join("from-export.coffer"), DatabaseConfig::default())?;
//! let imported = restored.import_manifest(export_dir.join(exported.manifest), cofferdb::OnConflict::Error)?;
//! assert_eq!(imported.documents_inserted, 1);
//! restored.drop_collection("notes")?;
//! restored.close()?;
//!
//! db.rebuild_index(None)?;
//! let compact = db.compact(CompactOptions { rebuild_indexes: true })?;
//! assert_eq!(compact.index_rebuild, "full_checkpoint_written");
//! db.close()?;
//!
//! let copy = Database::snapshot(&path, &backup)?;
//! assert_eq!(copy.bytes_copied, std::fs::metadata(&backup)?.len());
//! let report = Database::check_path(&backup, CheckOptions { verify_indexes: true })?;
//! assert_eq!(report.status, "ok");
//! let reader = Database::open_read_only(&backup)?;
//! let restored = reader.collection("notes")?;
//! let hits = restored.search().text("otters").top_k(1).execute()?;
//! assert_eq!(hits.results.len(), 1);
//! assert_eq!(restored.get_chunk(&hits.results[0].chunk_id)?.text, "River otters");
//! reader.close()?;
//! # std::fs::remove_dir_all(dir)?;
//! # Ok(())
//! # }
//! ```

mod benchmark;
// The native host adapter must be callable from a separate crate so it can share
// engine types and operation budgets. Hide this integration surface from the
// application-facing crate documentation.
#[doc(hidden)]
pub mod binding;
mod catalog;
// The CLI binary is a separate Rust crate target. Keep its dispatch entry
// reachable while keeping command handling out of the documented library
// interface.
#[doc(hidden)]
pub mod cli;
mod db;
mod error;
mod hash;
mod import;
mod index;
mod json;
mod lock;
mod maintenance_progress;
mod metadata;
mod model;
mod physical_file;
mod process_state;
mod profile;
mod query;
mod search_cost;
mod storage;
mod temporary;
mod time;
mod tokenizer;
mod write_progress;

#[cfg(all(test, unix))]
mod maintenance_faults;

#[cfg(all(feature = "native-test-support", unix))]
mod maintenance_probe;

#[cfg(test)]
mod test_alloc;

#[cfg(test)]
#[path = "../tests/support/error_contract.rs"]
mod test_error_contract;

// These root re-exports are the paths used by applications and the crate examples.
// Implementation modules can remain private while their public input, result and
// diagnostic types stay nameable by callers.
pub use tokenizer::TokenizerConfig;

pub use db::{
    CheckOptions, CheckReport, CompactOptions, CompactSummary, Database, DatabaseOpenOptions,
    RebuildIndexSummary, RepairSummary, SnapshotSummary,
};
pub use error::{Error, ErrorCode, Result};
pub use import::{ExportSummary, ImportManifest, ImportSummary, ImportWarning};
pub use json::{parse as parse_json, JsonObject, JsonValue};
pub use metadata::MetadataBoostConfig;
pub use model::{
    Chunk, ChunkContent, ChunkInput, CollectionConfig, CollectionHandle, CollectionInfo,
    CollectionStats, DatabaseStats, DistanceMetric, Document, DocumentInput, Embedding,
    EmbeddingInput, IndexStats, InsertSummary, OnConflict, QueryEmbeddingInput, SearchWeights,
    SourceSpan, VectorIndexConfig,
};
pub use profile::{DatabaseConfig, Durability, IoMode, ProfileParams, ResourceProfile};
pub use query::{
    CandidateLimits, NormalizedScores, SearchBuilder, SearchExecutionStats, SearchQueryContext,
    SearchQueryContextInput, SearchResult, SearchResults, TransientRebuildContext, WeightContext,
    WhyNotDiagnosis,
};
pub use storage::{CheckIssue, IndexSuffixReport};
