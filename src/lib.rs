mod benchmark;
#[doc(hidden)]
pub mod binding;
mod catalog;
#[doc(hidden)]
pub mod cli;
mod db;
mod error;
mod hash;
mod import;
mod index;
mod json;
mod lock;
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

#[cfg(test)]
mod test_alloc;

#[cfg(test)]
#[path = "../tests/support/error_contract.rs"]
mod test_error_contract;

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
