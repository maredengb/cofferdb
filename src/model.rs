//! Core collection, record and search types shared by native operations and their
//! JSON representations. Collection-scoped external keys identify caller data;
//! stored IDs link records and order ties within a database snapshot.
use std::collections::BTreeMap;

use crate::error::{Error, ErrorCode, Result};
#[cfg(test)]
use crate::hash::sha256_hex;
use crate::json::{
    ensure_object, object_from_pairs, optional_string, required_array, required_string,
    required_usize, JsonBuilder, JsonValue,
};
use crate::profile::ResourceProfile;
use crate::time;

/// Metric used to compare a query vector with stored embeddings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceMetric {
    /// Compare vectors by cosine similarity.
    Cosine,
    /// Use the dot product. Fusion keeps the raw value unless vectors are declared normalized.
    DotProduct,
    /// Use Euclidean distance. Smaller raw distances produce larger fusion scores.
    Euclidean,
}

impl DistanceMetric {
    /// Return the canonical metric name used in JSON.
    pub fn as_str(self) -> &'static str {
        // Emit one canonical spelling even for metrics accepted under several
        // input aliases. Stored collection metadata and query diagnostics can then
        // use the same value regardless of which alias the caller supplied.
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::DotProduct => "dot_product",
            DistanceMetric::Euclidean => "euclidean",
        }
    }

    /// Parse `cosine`, `dot_product` (also `dot` or `dot-product`), or `euclidean` (also `l2`).
    /// Other names return `UnsupportedMetric`.
    pub fn parse(value: &str) -> Result<Self> {
        // Aliases are accepted at input only. Comparing canonical output with the
        // original input literally would reject supported spellings such as l2
        // after a round trip.
        match value {
            "cosine" => Ok(DistanceMetric::Cosine),
            "dot" | "dot_product" | "dot-product" => Ok(DistanceMetric::DotProduct),
            "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
            other => Err(Error::new(
                ErrorCode::UnsupportedMetric,
                format!("unsupported distance metric {other}"),
            )),
        }
    }
}

/// Vector retrieval strategy declared by the collection.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum VectorIndexConfig {
    /// Use exact vector comparisons; the default and supported index strategy.
    #[default]
    Exact,
    /// Reserved provider configuration. Creating a collection with this variant returns
    /// `UnsupportedFeatureFlag`.
    Ann {
        /// Name of the proposed approximate nearest-neighbor provider.
        provider: String,
        /// Provider-specific configuration.
        params: JsonValue,
    },
}

/// Settings for a new collection. Use a positive dimension even for text-only searches.
#[derive(Clone, Debug)]
pub struct CollectionConfig {
    /// Tokenizer shared by stored text and search queries.
    pub tokenizer: crate::TokenizerConfig,
    /// Collection name used for lookup.
    pub name: String,
    /// Required number of components in each vector.
    pub embedding_dim: usize,
    /// Metric used for vector retrieval.
    pub distance_metric: DistanceMetric,
    /// Optional model identity checked when embeddings are supplied.
    pub embedding_model: Option<String>,
    /// Declares whether the caller supplies normalized vectors.
    pub vectors_normalized: bool,
    /// Strategy for retrieving vector candidates.
    pub vector_index: VectorIndexConfig,
}

/// Stored identity and retrieval settings of a collection.
#[derive(Clone, Debug)]
pub struct CollectionInfo {
    /// Tokenizer shared by stored text and search queries.
    pub tokenizer: crate::TokenizerConfig,
    /// Database-assigned identifier of the collection.
    pub collection_id: String,
    /// Collection name used for lookup.
    pub name: String,
    /// Required number of components in each vector.
    pub embedding_dim: usize,
    /// Metric used for vector retrieval.
    pub distance_metric: DistanceMetric,
    /// Optional model identity checked when embeddings are supplied.
    pub embedding_model: Option<String>,
    /// Declares whether the caller supplies normalized vectors.
    pub vectors_normalized: bool,
    /// Strategy for retrieving vector candidates.
    pub vector_index: VectorIndexConfig,
    /// Creation timestamp in RFC 3339 format.
    pub created_at: String,
}

/// Optional byte and line coordinates in the original source. Supply both ends of each range.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceSpan {
    /// Zero-based offset of the first byte, paired with `end`.
    pub start: Option<u64>,
    /// Exclusive byte end, paired with `start`. Equal endpoints describe an empty range.
    pub end: Option<u64>,
    /// Starting line number, counted from one.
    pub line_start: Option<u32>,
    /// Inclusive ending line number, counted from one.
    pub line_end: Option<u32>,
}

/// A document and the chunks to insert with it.
#[derive(Clone, Debug)]
pub struct DocumentInput {
    /// Caller-assigned document identifier scoped to this collection.
    pub external_id: String,
    /// Optional display title for the document.
    pub title: Option<String>,
    /// Optional source location supplied by the caller.
    pub source_uri: Option<String>,
    /// Application-defined metadata as a JSON object.
    pub metadata: JsonValue,
    /// Passages to insert under this document, with distinct ordinals.
    pub chunks: Vec<ChunkInput>,
}

/// One searchable passage of a document. Its external ID must be unique in the collection.
#[derive(Clone, Debug)]
pub struct ChunkInput {
    /// Caller-assigned chunk identifier scoped to this collection.
    pub chunk_external_id: String,
    /// Full text of the chunk.
    pub text: String,
    /// Position assigned to the chunk within its document.
    pub ordinal: usize,
    /// Application-defined metadata as a JSON object.
    pub metadata: JsonValue,
    /// Explicit source coordinates. If absent, coordinates are read from the chunk metadata when
    /// present.
    pub source_span: Option<SourceSpan>,
}

/// A caller-supplied embedding for an existing chunk in the target collection.
#[derive(Clone, Debug)]
pub struct EmbeddingInput {
    /// Caller-assigned chunk identifier scoped to this collection.
    pub chunk_external_id: String,
    /// Identity of the model that produced the embedding.
    pub model_name: String,
    /// Declared number of vector components.
    pub dim: usize,
    /// Finite vector components in model output order.
    pub vector: Vec<f32>,
}

/// A query vector with a declared dimension and optional model identity.
#[derive(Clone, Debug)]
pub struct QueryEmbeddingInput {
    /// Optional identity of the query model, compared when the collection also declares a model.
    pub model_name: Option<String>,
    /// Declared number of vector components.
    pub dim: usize,
    /// Finite vector components in model output order.
    pub vector: Vec<f32>,
}

/// Stored document metadata. Use the collection handle to read its chunks.
#[derive(Clone, Debug)]
pub struct Document {
    /// Database-assigned identifier of the document.
    pub document_id: String,
    /// Database-assigned identifier of the collection.
    pub collection_id: String,
    /// Caller-assigned document identifier scoped to this collection.
    pub external_id: String,
    /// Optional display title for the document.
    pub title: Option<String>,
    /// Optional source location supplied by the caller.
    pub source_uri: Option<String>,
    /// Application-defined metadata as a JSON object.
    pub metadata: JsonValue,
    /// Creation timestamp in RFC 3339 format.
    pub created_at: String,
    /// Most recent update timestamp in RFC 3339 format.
    pub updated_at: String,
}

/// A stored passage with its full text, identifiers and optional source coordinates.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Database-assigned identifier of the chunk.
    pub chunk_id: String,
    /// Database-assigned identifier of the document.
    pub document_id: String,
    /// Database-assigned identifier of the collection.
    pub collection_id: String,
    /// Caller-assigned chunk identifier scoped to this collection.
    pub chunk_external_id: String,
    /// Full text of the chunk.
    pub text: String,
    /// SHA-256 digest of the UTF-8 chunk text.
    pub text_hash: String,
    /// Position assigned to the chunk within its document.
    pub ordinal: usize,
    /// Number of tokens produced by the collection tokenizer for this passage.
    pub token_count: usize,
    /// Application-defined metadata as a JSON object.
    pub metadata: JsonValue,
    /// Identifier of the embedding attached to this chunk, if present.
    pub embedding_id: Option<String>,
    /// Optional location of the passage in the original source.
    pub source_span: Option<SourceSpan>,
    /// Creation timestamp in RFC 3339 format.
    pub created_at: String,
}

/// A complete stored chunk and its owning document from one read view.
///
/// Both values are owned and remain usable after closing the database. This
/// does not pin a view across separate search/read calls or load a source file.
#[derive(Clone, Debug)]
pub struct ChunkContent {
    /// Full chunk read from the collection.
    pub chunk: Chunk,
    /// Owning document from the same read view as the chunk.
    pub document: Document,
}

/// Stored vector and the identities of its chunk, collection and model.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// Database-assigned identifier of the embedding.
    pub embedding_id: String,
    /// Database-assigned identifier of the chunk.
    pub chunk_id: String,
    /// Database-assigned identifier of the collection.
    pub collection_id: String,
    /// Caller-assigned chunk identifier scoped to this collection.
    pub chunk_external_id: String,
    /// Identity of the model that produced the embedding.
    pub model_name: String,
    /// Declared number of vector components.
    pub dim: usize,
    /// Finite vector components in model output order.
    pub vector: Vec<f32>,
    /// Creation timestamp in RFC 3339 format.
    pub created_at: String,
}

/// A deletion record in the primary log. Applying it changes live visibility for
/// the target while retaining history until compaction.
#[derive(Clone, Debug)]
pub struct Tombstone {
    /// Kind of object named by the tombstone. Collection, document and chunk
    /// targets affect different visibility sets.
    pub target_type: String,
    /// Internal ID of the deletion target. Its namespace identifies a stored object
    /// independently of any collection-scoped external name.
    pub target_id: String,
    /// Deletion timestamp recorded in the primary log. It belongs to this deletion
    /// event, including when several records target the same ID.
    pub deleted_at: String,
    /// Optional reason stored with the deletion. Visibility is determined by the
    /// target, independently of this descriptive text.
    pub reason: Option<String>,
}

/// How import handles existing external IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnConflict {
    /// Return an error when an input conflicts with an existing record.
    Error,
    /// Keep the existing record and count the input as skipped.
    Skip,
    /// Replace a conflicting record according to the import operation.
    Replace,
}

impl OnConflict {
    /// Parse the case-sensitive import policy `error`, `skip` or `replace`; unknown names return
    /// `CliUsage`.
    pub fn parse(value: &str) -> Result<Self> {
        // Conflict policy is an explicit caller choice. Unknown tokens return an
        // error instead of falling back to an action that might overwrite or skip
        // existing records.
        match value {
            "error" => Ok(OnConflict::Error),
            "skip" => Ok(OnConflict::Skip),
            "replace" => Ok(OnConflict::Replace),
            other => Err(Error::new(
                ErrorCode::CliUsage,
                format!("unsupported conflict strategy {other}"),
            )),
        }
    }
}

/// Relative contributions of BM25, vector retrieval and metadata rules to the final score.
#[derive(Clone, Copy, Debug)]
pub struct SearchWeights {
    /// Weight applied to the normalized BM25 score.
    pub bm25: f64,
    /// Weight applied to the vector score after metric-specific normalization.
    pub vector: f64,
    /// Weight applied to the metadata boost.
    pub metadata: f64,
}

impl SearchWeights {
    // The caller has admitted the JSON value. This pass validates the scoring
    // contract: only the three coefficient names are legal, and every one must
    // have a numeric value, even when a coefficient is zero.
    pub(crate) fn from_validated_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidSearchWeights,
                "weights must be a JSON object",
            )
            .with_detail("field", "weights")
            .with_detail("weights_reason", "invalid_type")
            .with_detail("expected", "object")
        })?;
        // Reject unknown keys before checking required ones. For a misspelling
        // such as an extra coefficient, preserve the supplied field name in the
        // error instead of silently substituting zero.
        for key in obj.keys() {
            if !matches!(key.as_str(), "bm25" | "vector" | "metadata") {
                return Err(Error::new(
                    ErrorCode::InvalidSearchWeights,
                    format!("weights contains unsupported field {key}"),
                )
                .with_detail("field", format!("weights.{key}"))
                .with_detail("weights_reason", "unsupported_field")
                .with_detail("reason", "unsupported_field"));
            }
        }
        // Use the typed constructor after extracting all three numbers. Inputs
        // from a host JSON object and callers constructing the native weight tuple
        // must pass the same sum and sign checks.
        Self::new(
            Self::required_json_weight(obj, "bm25")?,
            Self::required_json_weight(obj, "vector")?,
            Self::required_json_weight(obj, "metadata")?,
        )
    }

    fn required_json_weight(obj: &crate::json::JsonObject, field: &str) -> Result<f64> {
        // as_f64 maps both missing keys and values of the wrong type to None. Look
        // up key presence again for the diagnostic so a caller can distinguish an
        // omitted weight from a nonnumeric one.
        obj.get(field).and_then(JsonValue::as_f64).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidSearchWeights,
                format!("weights must include numeric {field}"),
            )
            .with_detail("field", format!("weights.{field}"))
            .with_detail(
                "weights_reason",
                // An explicitly null coefficient is a supplied value of the wrong
                // type. Treating it as missing would hide the difference between
                // an incomplete object and one with an invalid value.
                if obj.contains_key(field) {
                    "invalid_type"
                } else {
                    "missing_required_field"
                },
            )
            .with_detail("expected", "number")
        })
    }

    // Use an absolute tolerance around the required unit sum. This only decides
    // acceptance of caller-supplied coefficients; validation does not rescale
    // those coefficients.
    const SUM_EPSILON: f64 = 1e-9;

    /// Create validated fusion weights. Values must be finite, nonnegative and sum to one within
    /// `1e-9`.
    ///
    /// ```
    /// use cofferdb::SearchWeights;
    /// let weights = SearchWeights::new(0.75, 0.25, 0.0)?;
    /// assert_eq!(weights.bm25, 0.75);
    /// assert!(SearchWeights::new(1.0, 1.0, 0.0).is_err());
    /// # Ok::<(), cofferdb::Error>(())
    /// ```
    pub fn new(bm25: f64, vector: f64, metadata: f64) -> Result<Self> {
        let weights = Self {
            bm25,
            vector,
            metadata,
        };
        // Validate the complete tuple in one step. None of these fields has a
        // default when the caller explicitly supplies a weight object, even for a
        // branch absent from the eventual query.
        weights.validate()?;
        Ok(weights)
    }

    /// Give equal weight to available text and vector inputs and zero to metadata. With neither
    /// input, all three weights are zero.
    pub fn default_for(has_text: bool, has_vector: bool) -> Self {
        match (has_text, has_vector) {
            (true, true) => Self {
                bm25: 0.5,
                vector: 0.5,
                metadata: 0.0,
            },
            (true, false) => Self {
                bm25: 1.0,
                vector: 0.0,
                metadata: 0.0,
            },
            (false, true) => Self {
                bm25: 0.0,
                vector: 1.0,
                metadata: 0.0,
            },
            // The no-input default represents zero scoring channels. It is a
            // sentinel returned by mode selection; explicitly supplied all-zero
            // weights instead fail the validate method below.
            (false, false) => Self {
                bm25: 0.0,
                vector: 0.0,
                metadata: 0.0,
            },
        }
    }

    /// Return `InvalidSearchWeights` for non-finite or negative weights, all zeros, or a sum
    /// farther than `1e-9` from one.
    pub fn validate(self) -> Result<()> {
        let values = [
            ("bm25", self.bm25),
            ("vector", self.vector),
            ("metadata", self.metadata),
        ];
        // Check the components before calculating the sum. NaN would defeat a
        // comparison based only on distance from one, and a component error needs
        // to identify the offending field.
        if let Some((field, value)) = values
            .iter()
            .find(|(_, value)| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::new(
                ErrorCode::InvalidSearchWeights,
                "search weights must be finite and non-negative",
            )
            .with_detail("field", format!("weights.{field}"))
            .with_detail("reason", "finite_non_negative")
            .with_detail("weights_reason", "finite_non_negative")
            .with_detail("actual", value));
        }
        // Keep the all-zero case distinct from a general sum mismatch. It signals
        // that no scoring component was selected and has its own machine-readable
        // reason.
        if values.iter().all(|(_, value)| *value == 0.0) {
            return Err(Error::new(
                ErrorCode::InvalidSearchWeights,
                "at least one search weight must be positive",
            )
            .with_detail("field", "weights")
            .with_detail("reason", "all_zero")
            .with_detail("weights_reason", "all_zero")
            .with_detail("actual_sum", 0.0));
        }
        // Compare the sum with one only after individual finite, nonnegative
        // values are known. Preserve the original coefficients on success so
        // callers can reproduce the actual scoring parameters.
        let sum = self.bm25 + self.vector + self.metadata;
        // Use a strict greater-than comparison so sums exactly on the tolerance
        // boundary pass. The same tolerance applies whichever of the three
        // coefficients caused roundoff.
        if (sum - 1.0).abs() > Self::SUM_EPSILON {
            return Err(Error::new(
                ErrorCode::InvalidSearchWeights,
                "search weights must sum to 1.0",
            )
            .with_detail("field", "weights")
            .with_detail("reason", "sum_mismatch")
            .with_detail("weights_reason", "sum_mismatch")
            .with_detail("expected_sum", 1.0)
            .with_detail("actual_sum", sum));
        }
        Ok(())
    }
}

/// Counts of documents, chunks and embeddings affected by a write.
#[derive(Clone, Debug, Default)]
pub struct InsertSummary {
    /// Number of documents inserted by this operation.
    pub documents_inserted: usize,
    /// Number of chunks inserted by this operation.
    pub chunks_inserted: usize,
    /// Number of embeddings inserted by this operation.
    pub embeddings_inserted: usize,
    /// Number of documents skipped by this operation.
    pub documents_skipped: usize,
    /// Number of chunks skipped by this operation.
    pub chunks_skipped: usize,
    /// Number of embeddings skipped by this operation.
    pub embeddings_skipped: usize,
    /// Number of documents replaced by this operation.
    pub documents_replaced: usize,
    /// Number of chunks replaced by this operation.
    pub chunks_replaced: usize,
    /// Number of embeddings replaced by this operation.
    pub embeddings_replaced: usize,
}

/// Live record counts, index statistics and estimated storage usage for a collection.
#[derive(Clone, Debug, Default)]
pub struct CollectionStats {
    /// Tokenizer shared by stored text and search queries.
    pub tokenizer: crate::TokenizerConfig,
    /// Name of the collection described by this value.
    pub collection: String,
    /// Number of live documents.
    pub document_count: usize,
    /// Number of live chunks.
    pub chunk_count: usize,
    /// Number of live embeddings.
    pub embedding_count: usize,
    /// Number of recorded deletion markers.
    pub tombstone_count: usize,
    /// Number of distinct indexed text terms reported for this collection.
    pub text_index_terms: usize,
    /// Number of persisted text segments reported for this collection.
    pub text_index_segments: usize,
    /// Number of live vectors.
    pub vector_count: usize,
    /// Size in bytes of the usable database-wide index checkpoint, or zero when unavailable.
    pub index_size_bytes: u64,
    /// Estimated size of live data in bytes.
    pub live_bytes: u64,
    /// Estimated bytes associated with deleted records.
    pub dead_bytes_estimate: u64,
    /// Resource preset used for this operation.
    pub resource_profile: ResourceProfile,
    /// Cache budget in MiB.
    pub cache_budget_mb: usize,
    /// Identifier of the most recent committed batch in the database.
    pub last_commit_id: u64,
    /// Estimated bytes pending compaction, including an allowance for index storage.
    pub pending_compaction_bytes: u64,
    /// On-disk file format version.
    pub format_version: u16,
    /// Whether background compaction is active. Currently always false.
    pub background_compaction_enabled: bool,
}

/// Database-wide vector counts and information about the persisted index checkpoint.
#[derive(Clone, Debug, Default)]
pub struct IndexStats {
    /// Number of distinct terms reported by the persisted text checkpoint.
    pub text_terms: usize,
    /// Number of text segments in the usable checkpoint, or zero when unavailable.
    pub text_segments: usize,
    /// Number of live vectors.
    pub vector_count: usize,
    /// Number of vector segments in the usable checkpoint, or zero when unavailable.
    pub vector_segments: usize,
    /// Whether the persisted index needs rebuilding.
    pub rebuild_needed: bool,
    /// Size in bytes of the usable database-wide index checkpoint, or zero when unavailable.
    pub index_size_bytes: u64,
    /// Commit covered by the usable index checkpoint, when available.
    pub last_index_commit_id: Option<u64>,
}

/// Live record counts and storage statistics for the database.
#[derive(Clone, Debug)]
pub struct DatabaseStats {
    /// Number of live collections.
    pub collection_count: usize,
    /// Number of live documents.
    pub document_count: usize,
    /// Number of live chunks.
    pub chunk_count: usize,
    /// Number of live embeddings.
    pub embedding_count: usize,
    /// Number of recorded deletion markers.
    pub tombstone_count: usize,
    /// Size of the database file in bytes.
    pub file_size: u64,
    /// Size in bytes of the usable database-wide index checkpoint, or zero when unavailable.
    pub index_size_bytes: u64,
    /// Estimated size of live data in bytes.
    pub live_bytes: u64,
    /// Estimated bytes associated with deleted records.
    pub dead_bytes_estimate: u64,
    /// Resource preset active on the database handle.
    pub active_profile: ResourceProfile,
    /// Cache budget in MiB.
    pub cache_budget_mb: usize,
    /// Identifier of the most recent committed batch in the database.
    pub last_commit_id: u64,
    /// Estimated bytes pending compaction, including an allowance for index storage.
    pub pending_compaction_bytes: u64,
    /// On-disk file format version.
    pub format_version: u16,
    /// Whether background compaction is active. Currently always false.
    pub background_compaction_enabled: bool,
}

/// Access to one collection through the database that opened it.
///
/// Writes append batches through the database's exclusive writer handle. Input validation
/// errors leave the batch unapplied. An append or synchronization error can leave the commit
/// outcome uncertain. If the error detail says `writer_reopen_required`, close and reopen
/// the database and inspect the affected IDs before retrying. Deletion hides records;
/// [`Database::compact`](crate::Database::compact) reclaims space.
/// Separate read-only handles retain the view they opened.
///
/// The [Rust programming guide](https://docs.rs/crate/cofferdb/0.2.1/source/docs/rust.md)
/// explains how to reopen the database and read full chunk content.
///
/// # Examples
///
/// ```
/// use cofferdb::{
///     ChunkInput, CollectionConfig, Database, DistanceMetric, DocumentInput,
///     EmbeddingInput, ErrorCode, JsonValue,
/// };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
/// let dir = std::env::temp_dir().join(format!("coffer-notes-{}-{stamp}", std::process::id()));
/// std::fs::create_dir(&dir)?;
/// let db = Database::create(dir.join("notes.coffer"), Default::default())?;
/// let notes = db.create_collection(CollectionConfig {
///     name: "notes".into(), tokenizer: Default::default(), embedding_dim: 2,
///     distance_metric: DistanceMetric::Cosine, embedding_model: None,
///     vectors_normalized: false, vector_index: Default::default(),
/// })?;
/// notes.insert_document(DocumentInput {
///     external_id: "garden".into(), title: None, source_uri: None, metadata: JsonValue::object(),
///     chunks: vec![ChunkInput {
///         chunk_external_id: "leaf".into(), ordinal: 0, text: "A jade leaf".into(),
///         metadata: JsonValue::object(), source_span: None,
///     }],
/// })?;
/// notes.put_embeddings(vec![EmbeddingInput {
///     chunk_external_id: "leaf".into(), model_name: "demo".into(), dim: 2,
///     vector: vec![1.0, 0.0],
/// }])?;
/// let hits = notes.search().text("jade").vector(vec![1.0, 0.0]).top_k(1).execute()?;
/// let hit = &hits.results[0];
/// assert_eq!(hit.chunk_external_id, "leaf");
/// notes.delete_chunk(&hit.chunk_id)?;
/// assert_eq!(notes.get_chunk(&hit.chunk_id).unwrap_err().code(), ErrorCode::ChunkNotFound);
/// notes.delete_document(&hit.document_id)?;
/// db.close()?;
/// # std::fs::remove_dir_all(dir)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct CollectionHandle {
    // Shared database connection that resolves this handle's operations and close
    // state. Cloning a handle does not open another file.
    pub(crate) db: crate::db::Database,
    // Stable internal collection identity used for lookups and writes even when the
    // caller originally selected the collection by name.
    pub(crate) collection_id: String,
    // Collection name captured for the user-facing handle. Resolving current
    // visibility still goes through the database.
    pub(crate) name: String,
}

impl CollectionHandle {
    // Retain the connection and resolved ID when constructing the handle. Access
    // to data continues through the database, while name() can borrow the spelling
    // captured at lookup.
    pub(crate) fn new(db: crate::db::Database, collection_id: String, name: String) -> Self {
        Self {
            db,
            collection_id,
            name,
        }
    }

    /// Borrow the collection name for the lifetime of this handle reference.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the database-assigned collection identifier.
    pub fn collection_id(&self) -> &str {
        &self.collection_id
    }

    /// Commit a document and its chunks in one batch.
    ///
    /// # Errors
    ///
    /// Invalid input or an external ID conflict rejects the batch.
    /// Writes require a writable handle. See [`CollectionHandle`] for how to
    /// check the outcome of a failed commit before retrying.
    ///
    /// # Examples
    ///
    /// See the [collection example](CollectionHandle#examples).
    pub fn insert_document(&self, input: DocumentInput) -> Result<InsertSummary> {
        self.db
            .insert_document_in_collection(&self.collection_id, input, OnConflict::Error)
    }

    /// Commit a batch of embeddings for existing chunks.
    ///
    /// # Errors
    ///
    /// A missing chunk, invalid embedding, duplicate chunk ID or existing embedding rejects the batch.
    /// Writes require a writable handle. See [`CollectionHandle`] for how to
    /// check the outcome of a failed commit before retrying.
    ///
    /// # Examples
    ///
    /// See the [collection example](CollectionHandle#examples).
    pub fn put_embeddings(&self, embeddings: Vec<EmbeddingInput>) -> Result<InsertSummary> {
        // Resolve chunk external IDs within this handle's collection. The
        // containing database supplies the persistent identity and model
        // constraints when attaching the embeddings.
        self.db
            .put_embeddings_in_collection(&self.collection_id, embeddings, OnConflict::Error)
    }

    /// Read a live document by database-assigned ID. Returns `DocumentNotFound` when absent from
    /// this collection.
    pub fn get_document(&self, document_id: &str) -> Result<Document> {
        self.db
            .get_document_in_collection(&self.collection_id, document_id)
    }

    /// Read a live document by its caller-assigned ID in this collection.
    pub fn get_document_by_external_id(&self, external_id: &str) -> Result<Document> {
        self.db
            .get_document_by_external_id_in_collection(&self.collection_id, external_id)
    }

    /// Read the document's live chunks in ordinal order using its database-assigned ID.
    pub fn list_chunks(&self, document_id: &str) -> Result<Vec<Chunk>> {
        self.db
            .list_chunks_in_collection(&self.collection_id, document_id)
    }

    /// Read live chunks in ordinal order using the document's external ID.
    pub fn list_chunks_by_document_external_id(&self, external_id: &str) -> Result<Vec<Chunk>> {
        self.db
            .list_chunks_by_document_external_id_in_collection(&self.collection_id, external_id)
    }

    /// Read a live chunk by database-assigned ID. Returns `ChunkNotFound` when absent from this
    /// collection.
    pub fn get_chunk(&self, chunk_id: &str) -> Result<Chunk> {
        self.db
            .get_chunk_in_collection(&self.collection_id, chunk_id)
    }

    /// Read a live chunk by its caller-assigned ID in this collection.
    pub fn get_chunk_by_external_id(&self, chunk_external_id: &str) -> Result<Chunk> {
        self.db
            .get_chunk_by_external_id_in_collection(&self.collection_id, chunk_external_id)
    }

    /// Read the complete chunk and its document in one bounded read operation.
    /// Missing or deleted chunks return the existing `chunk_not_found` error.
    /// # Errors
    ///
    /// Returns an error if the chunk or its parent is not visible in this
    /// collection, or if their content cannot be read or admitted.
    pub fn get_chunk_content(&self, chunk_id: &str) -> Result<ChunkContent> {
        // Use the single content-read operation for the pair. Looking up the
        // document independently after returning the chunk could mix two different
        // views of a changing collection.
        self.db
            .get_chunk_content_in_collection(&self.collection_id, chunk_id, false)
    }

    /// Read complete content using an external ID scoped to this collection.
    pub fn get_chunk_content_by_external_id(&self, external_id: &str) -> Result<ChunkContent> {
        self.db
            .get_chunk_content_in_collection(&self.collection_id, external_id, true)
    }

    /// Commit a deletion marker that hides the document and its chunks from live reads.
    ///
    /// # Errors
    ///
    /// Returns an error if the document is absent from this collection.
    /// Writes require a writable handle. See [`CollectionHandle`] for how to
    /// check the outcome of a failed commit before retrying.
    ///
    /// # Examples
    ///
    /// See the [collection example](CollectionHandle#examples).
    pub fn delete_document(&self, document_id: &str) -> Result<()> {
        self.db
            .delete_document_in_collection(&self.collection_id, document_id)
    }

    /// Commit a deletion marker for a chunk. Its parent document remains available.
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk is absent from this collection.
    /// Writes require a writable handle. See [`CollectionHandle`] for how to
    /// check the outcome of a failed commit before retrying.
    ///
    /// # Examples
    ///
    /// See the [collection example](CollectionHandle#examples).
    pub fn delete_chunk(&self, chunk_id: &str) -> Result<()> {
        self.db
            .delete_chunk_in_collection(&self.collection_id, chunk_id)
    }

    /// Read counts and estimated storage usage for this collection.
    pub fn stats(&self) -> Result<CollectionStats> {
        self.db.collection_stats_by_id(&self.collection_id)
    }

    /// Start a search for this collection. Set the query inputs and call [`execute`](crate::SearchBuilder::execute) to obtain hits.
    pub fn search(&self) -> crate::query::SearchBuilder {
        // The builder keeps a cloned handle. Building a query can therefore end
        // this borrow while retaining the database and collection identity until
        // execution.
        crate::query::SearchBuilder::new(self.clone())
    }
}

impl CollectionConfig {
    /// Check the collection name, dimension, optional model, supported index strategy and JSON
    /// limits without creating a collection.
    /// # Errors
    ///
    /// Returns an error for a blank name, a zero or unrepresentable dimension, a
    /// blank supplied model, an unsupported index strategy, or a value that
    /// exceeds JSON limits.
    pub fn validate(&self) -> Result<()> {
        self.validate_in(&crate::json::JsonOperationContext::standalone())
            .map(|_| ())
    }

    // Retain JSON admission credit for callers that go on to materialize the
    // configuration. The public validation-only entry point drops this reservation
    // immediately after checking.
    pub(crate) fn validate_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::JsonReservation> {
        self.validate_object_in(context, false)
    }

    // Use this path when the name and optional model are already charged to a
    // known JSON reservation. Verify their allocated capacities before deciding
    // that validation need not charge their bytes another time.
    pub(crate) fn validate_retained_in(
        &self,
        strings: &crate::json::JsonReservation,
    ) -> Result<crate::json::JsonReservation> {
        // Count allocated capacity of the existing owned strings, including spare
        // capacity. The invariant here concerns memory already held, not the
        // shorter serialized lengths of their text.
        let bytes =
            self.name.capacity() + self.embedding_model.as_ref().map_or(0, String::capacity);
        assert!(bytes <= strings.covered_bytes());
        self.validate_object_in(strings.context(), true)
    }

    // Preprocessed host input can have its strings covered outside a single JSON
    // guard. The caller establishes that coverage and passes the same operation
    // context for validation of the remaining object.
    pub(crate) fn validate_preprocessed_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::JsonReservation> {
        self.validate_object_in(context, true)
    }

    fn validate_object_in(
        &self,
        context: &crate::json::JsonOperationContext,
        precharged_strings: bool,
    ) -> Result<crate::json::JsonReservation> {
        // Trimming is an emptiness test, not normalization. Leave a nonblank name
        // as supplied for the collection that will actually be created.
        if self.name.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "collection name cannot be empty",
            ));
        }
        // Vectors carry a host-sized count in memory but the collection accepts
        // widths representable as u32. Check this even for a collection that
        // currently has only text rows.
        if self.embedding_dim == 0 || u32::try_from(self.embedding_dim).is_err() {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "embedding dimension must be in 1..=u32::MAX",
            ));
        }
        // An unset model leaves the collection without a declared model name. A
        // supplied blank string cannot stand for that state, since it would
        // propagate an unusable model identity.
        if self
            .embedding_model
            .as_ref()
            .map(|model| model.trim().is_empty())
            .unwrap_or(false)
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding_model cannot be empty when provided",
            ));
        }
        // Accepting a provider name and parameters in the Rust representation does
        // not implement an ANN index. Reject this variant here before a caller
        // could persist an unsupported retrieval configuration.
        if matches!(self.vector_index, VectorIndexConfig::Ann { .. }) {
            return Err(Error::new(
                ErrorCode::UnsupportedFeatureFlag,
                "ANN vector index configuration is reserved for a future version",
            ));
        }
        // Validate the encoded configuration under the collection-config policy
        // after its semantic checks pass. Even strings whose storage is already
        // charged still contribute to the size and shape of this whole object.
        let mut guard = crate::json::JsonObjectGuard::new(
            crate::json::JsonInputKind::CollectionConfig,
            context,
        );
        // Select the precharged path only for strings covered by the prepared
        // input. It avoids a second charge for their storage.
        if precharged_strings {
            guard.string_field_precharged("name", &self.name)?;
        } else {
            guard.string_field("name", &self.name)?;
        }
        // An absent model is represented by JSON null rather than an empty
        // identity string. A supplied model uses the same admission rule as the
        // collection name.
        match &self.embedding_model {
            Some(model) if precharged_strings => {
                guard.string_field_precharged("embedding_model", model)?
            }
            Some(model) => guard.string_field("embedding_model", model)?,
            None => guard.value_field("embedding_model", &JsonValue::Null)?,
        }
        // The positive u32 width fits exactly in the JSON number representation.
        // Converting only after the range check preserves the configured component
        // count when stored or reported.
        guard.value_field(
            "embedding_dim",
            &JsonValue::Number(self.embedding_dim as f64),
        )?;
        guard.value_field(
            "distance_metric",
            &JsonValue::String(self.distance_metric.as_str().into()),
        )?;
        guard.value_field(
            "vectors_normalized",
            &JsonValue::Bool(self.vectors_normalized),
        )?;
        // The nested index and tokenizer objects are part of the same business
        // value. Include them in the aggregate guard so each field cannot pass
        // separately while the whole object exceeds its budget.
        guard.value_field("vector_index", &vector_index_to_json(&self.vector_index))?;
        guard.value_field("tokenizer", &self.tokenizer.to_json())?;
        // Return the finished reservation to callers that keep the checked object
        // alive. Functions that only validate discard it when the call returns.
        Ok(guard.finish())
    }
}

impl SourceSpan {
    /// Return the JSON representation of this source span.
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host span construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        // Serialize all four coordinate fields, using null for absent endpoints.
        // The integer writer switches large byte offsets to decimal strings so
        // adjacent u64 positions can be distinguished beyond the JSON number
        // precision limit.
        builder.object([
            ("start", builder.optional_u64(self.start)?),
            ("end", builder.optional_u64(self.end)?),
            (
                "line_start",
                builder.optional_u64(self.line_start.map(u64::from))?,
            ),
            (
                "line_end",
                builder.optional_u64(self.line_end.map(u64::from))?,
            ),
        ])
    }

    /// Read paired byte and line coordinates. A JSON null yields an empty span.
    ///
    /// # Errors
    ///
    /// Returns a schema error if a coordinate has the wrong type or range, one end
    /// of a pair is missing, or the start exceeds the end.
    pub fn from_json(value: &JsonValue) -> Result<Self> {
        if value.is_null() {
            return Ok(SourceSpan::default());
        }
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "source_span must be an object",
            )
        })?;
        // The byte fields accept exact unsigned decimal strings as well as JSON
        // integers. Line numbers have a narrower numeric representation. Keep the
        // field readers separate so a bad coordinate identifies its own field.
        let span = Self {
            start: optional_offset_field(obj, "start", "source_span")?,
            end: optional_offset_field(obj, "end", "source_span")?,
            line_start: optional_line_field(obj, "line_start", "source_span")?,
            line_end: optional_line_field(obj, "line_end", "source_span")?,
        };
        // Only after decoding each endpoint can the pair check distinguish an
        // omitted range from an incomplete one. Ascending equal endpoints are
        // permitted.
        span.validate()?;
        Ok(span)
    }

    /// Read coordinates from metadata, returning `None` for absent or invalid coordinates. Use [`try_from_metadata`](Self::try_from_metadata) to distinguish them.
    pub fn from_metadata(metadata: &JsonValue) -> Option<Self> {
        Self::try_from_metadata(metadata).ok().flatten()
    }

    /// Read `byte_start`/`byte_end` and `line_start`/`line_end`. Absent coordinates yield
    /// `Ok(None)`; malformed coordinates return an error.
    pub fn try_from_metadata(metadata: &JsonValue) -> Result<Option<Self>> {
        // Unrelated metadata need not provide coordinates. A nonobject has no keys
        // to project here; callers that require object metadata enforce that shape
        // separately.
        let Some(obj) = metadata.as_object() else {
            return Ok(None);
        };
        // Metadata uses byte_start and byte_end, whereas an explicit span uses
        // start and end. Both paths must agree on the numeric ranges accepted for
        // identical coordinates.
        let span = Self {
            start: optional_offset_field(obj, "byte_start", "metadata")?,
            end: optional_offset_field(obj, "byte_end", "metadata")?,
            line_start: optional_line_field(obj, "line_start", "metadata")?,
            line_end: optional_line_field(obj, "line_end", "metadata")?,
        };
        // Treat an all-absent or all-null set as no metadata span. The presence of
        // even one endpoint switches to strict pair validation; a partial range
        // must not disappear silently.
        if span.start.is_some()
            || span.end.is_some()
            || span.line_start.is_some()
            || span.line_end.is_some()
        {
            span.validate()?;
            Ok(Some(span))
        } else {
            Ok(None)
        }
    }

    /// Check that each range has both endpoints in ascending order (ties allowed) and that line
    /// numbers start at one.
    pub fn validate(&self) -> Result<()> {
        // Byte offsets are zero based, line numbers are one based. Check them as
        // independent pairs so a location may supply just one coordinate system.
        validate_span_pair(
            self.start,
            self.end,
            "source_span.start",
            "source_span.end",
            0,
        )?;
        validate_span_pair(
            self.line_start.map(u64::from),
            self.line_end.map(u64::from),
            "source_span.line_start",
            "source_span.line_end",
            1,
        )?;
        Ok(())
    }
}

impl CollectionInfo {
    /// Return the JSON representation of this collection info.
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host collection construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        // Collection information carries both the top-level tokenizer and the
        // nested config.tokenizer consumed by stored readers. Both values describe
        // the same tokenizer and must be built under the same output policy.
        builder.object([
            ("tokenizer", self.tokenizer.build_json(builder)?),
            (
                "config",
                builder.object([("tokenizer", self.tokenizer.build_json(builder)?)])?,
            ),
            ("collection_id", builder.string(&self.collection_id)?),
            ("name", builder.string(&self.name)?),
            (
                "embedding_dim",
                JsonValue::Number(self.embedding_dim as f64),
            ),
            (
                "distance_metric",
                builder.string(self.distance_metric.as_str())?,
            ),
            // Preserve an unbound collection model as JSON null. It is a different
            // state from a stored model string, including when emitting a
            // collection that has no live embeddings.
            (
                "embedding_model",
                builder.optional_str(self.embedding_model.as_deref())?,
            ),
            (
                "vectors_normalized",
                JsonValue::Bool(self.vectors_normalized),
            ),
            (
                // Delegate the index object to its tagged encoder. An existing
                // provider configuration can still be represented in collection
                // info even when new collections support only exact indexing.
                "vector_index",
                vector_index_build_json(&self.vector_index, builder)?,
            ),
            ("created_at", builder.string(&self.created_at)?),
        ])
    }

    /// Decode a stored collection info from JSON, returning an error for an invalid record.
    /// # Errors
    ///
    /// Returns an error if required fields or the timestamp are invalid, the
    /// tokenizer descriptions disagree, or present tokenizer, distance metric or
    /// vector-index settings cannot be decoded.
    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "collection record must be an object",
            )
        })?;
        // Older records can lack the configuration wrapper. A present wrapper must
        // be an object before reading its tokenizer; treating an invalid wrapper
        // as absent would silently change tokenization.
        let config_tokenizer = match obj.get("config") {
            Some(config) => config
                .as_object()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::FileFormatError,
                        "collection config must be an object",
                    )
                })?
                .get("tokenizer"),
            None => None,
        };
        // When both locations exist, compare their JSON values before selecting
        // one. Picking the first parsable tokenizer could reinterpret persisted
        // text with an inconsistent alternate definition.
        if let (Some(persisted), Some(public)) = (config_tokenizer, obj.get("tokenizer")) {
            if persisted != public {
                return Err(Error::new(
                    ErrorCode::FileFormatError,
                    "collection tokenizer identity mismatch",
                ));
            }
        }
        Ok(Self {
            collection_id: required_string(obj, "collection_id")?,
            // Prefer the nested tokenizer, accept the top-level form on its own,
            // and use the default only when both are absent. A present malformed
            // definition is an error.
            tokenizer: config_tokenizer
                .or_else(|| obj.get("tokenizer"))
                .map(crate::TokenizerConfig::from_json)
                .transpose()?
                .unwrap_or_default(),
            name: required_string(obj, "name")?,
            embedding_dim: required_usize(obj, "embedding_dim")?,
            distance_metric: DistanceMetric::parse(&required_string(obj, "distance_metric")?)?,
            // The model in this record is optional. Decoding collection metadata
            // on its own cannot determine the model identities of the currently
            // visible embeddings.
            embedding_model: optional_string(obj, "embedding_model"),
            // A missing or non-boolean normalization flag takes the historical
            // false value. The index configuration below defaults only when
            // absent; a present, invalid index still returns a decoding error.
            vectors_normalized: obj
                .get("vectors_normalized")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
            vector_index: obj
                .get("vector_index")
                .map(vector_index_from_json)
                .transpose()?
                .unwrap_or_default(),
            created_at: required_rfc3339_string(obj, "created_at")?,
        })
    }
}

impl Document {
    /// Return the JSON representation of this document.
    pub fn to_json(&self) -> JsonValue {
        // The public projection borrows its input and must leave all owned fields
        // usable. Clone here because the shared field encoder consumes its
        // document; the operation-aware borrowing path below admits the copy
        // first.
        self.clone()
            .build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn to_json_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<(JsonValue, crate::json::JsonReservation)> {
        // Start with an admitted copy while the caller keeps the original
        // document. Its metadata tree already carries JSON credit, and its owned
        // strings can transfer their workspace charge during projection.
        let (document, mut work) = crate::storage::copy_document_in(self, context)?;
        let value = document.into_json_retained(&mut work)?;
        Ok((value, work.json))
    }

    pub(crate) fn into_json_retained(
        self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        // Give the builder both parts of the record reservation. Moving a native
        // string to JSON transfers backing capacity from workspace, so the caller
        // need not reserve storage for a second copy.
        let (json, strings) = work.parts_mut();
        self.build_json(&JsonBuilder::budgeted_owned_strings(
            json,
            strings,
            crate::json::JsonInputKind::StoredJson,
        ))
    }

    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("document_id", builder.owned_string(self.document_id)?),
            ("collection_id", builder.owned_string(self.collection_id)?),
            ("external_id", builder.owned_string(self.external_id)?),
            ("title", builder.optional_string(self.title)?),
            ("source_uri", builder.optional_string(self.source_uri)?),
            // Metadata already has the output representation. Move the tree
            // directly and keep its original JSON accounting while the builder
            // adds the surrounding record.
            ("metadata", self.metadata),
            ("created_at", builder.owned_string(self.created_at)?),
            ("updated_at", builder.owned_string(self.updated_at)?),
        ])
    }

    /// Decode a stored document from JSON, returning an error for an invalid record.
    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "document record must be an object",
            )
        })?;
        Ok(Self {
            document_id: required_string(obj, "document_id")?,
            collection_id: required_string(obj, "collection_id")?,
            external_id: required_string(obj, "external_id")?,
            // The optional text decoder accepts only strings and maps other
            // representations to absence. Required identifiers and timestamps use
            // stricter readers.
            title: optional_string(obj, "title"),
            source_uri: optional_string(obj, "source_uri"),
            // An omitted metadata field represents an empty object in a stored
            // document. Once present, its value must be an object; an explicit
            // null is a schema error.
            metadata: obj
                .get("metadata")
                .cloned()
                .map(ensure_object)
                .transpose()?
                .unwrap_or_else(JsonValue::object),
            // Both timestamp fields must be parseable RFC 3339 text. Decoding
            // retains their supplied spelling and checks each independently of the
            // other timestamp.
            created_at: required_rfc3339_string(obj, "created_at")?,
            updated_at: required_rfc3339_string(obj, "updated_at")?,
        })
    }
}

impl ChunkContent {
    pub(crate) fn into_json_retained(
        self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        // One result owns both the chunk and its parent document. Project each
        // against the incoming reservation before charging the outer object's
        // members, so either child can transfer storage it already owns.
        let chunk = self.chunk.into_json_retained(work)?;
        let document = self.document.into_json_retained(work)?;
        let builder = JsonBuilder::budgeted(&mut work.json, crate::json::JsonInputKind::StoredJson);
        builder.object([("chunk", chunk), ("document", document)])
    }

    /// Project the complete owned content using the existing core JSON fields.
    pub fn to_json(&self) -> JsonValue {
        crate::json::object_from_pairs([
            ("chunk", self.chunk.to_json()),
            ("document", self.document.to_json()),
        ])
    }
}

impl Chunk {
    /// Return the JSON representation of this chunk.
    pub fn to_json(&self) -> JsonValue {
        self.clone()
            .build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn into_json_retained(
        self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        // A retained chunk can contain large source text as well as metadata. Its
        // native text buffer and its JSON tree are accounted in separate parts of
        // the incoming reservation before they are combined in the output.
        let (json, strings) = work.parts_mut();
        self.build_json(&JsonBuilder::budgeted_owned_strings(
            json,
            strings,
            crate::json::JsonInputKind::StoredJson,
        ))
    }

    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("chunk_id", builder.owned_string(self.chunk_id)?),
            ("document_id", builder.owned_string(self.document_id)?),
            ("collection_id", builder.owned_string(self.collection_id)?),
            (
                "chunk_external_id",
                builder.owned_string(self.chunk_external_id)?,
            ),
            // The stored text and its recorded hash are separate fields.
            // Projection moves both as supplied and never recomputes a hash from
            // the text.
            ("text", builder.owned_string(self.text)?),
            ("text_hash", builder.owned_string(self.text_hash)?),
            ("ordinal", JsonValue::Number(self.ordinal as f64)),
            ("token_count", JsonValue::Number(self.token_count as f64)),
            ("metadata", self.metadata),
            ("embedding_id", builder.optional_string(self.embedding_id)?),
            (
                // A missing source span becomes JSON null. A present span is
                // encoded as an object even when its individual coordinates are
                // absent.
                "source_span",
                self.source_span
                    .as_ref()
                    .map(|span| span.build_json(builder))
                    .transpose()?
                    .unwrap_or(JsonValue::Null),
            ),
            ("created_at", builder.owned_string(self.created_at)?),
        ])
    }

    /// Decode a stored chunk from JSON, returning an error for an invalid record.
    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(ErrorCode::FileFormatError, "chunk record must be an object")
        })?;
        Ok(Self {
            chunk_id: required_string(obj, "chunk_id")?,
            document_id: required_string(obj, "document_id")?,
            collection_id: required_string(obj, "collection_id")?,
            chunk_external_id: required_string(obj, "chunk_external_id")?,
            text: required_string(obj, "text")?,
            text_hash: required_string(obj, "text_hash")?,
            // Ordinal and token count come from numeric fields in the record,
            // independently of text length. Decoding this shape does not
            // retokenize the stored text.
            ordinal: required_usize(obj, "ordinal")?,
            token_count: required_usize(obj, "token_count")?,
            metadata: obj
                .get("metadata")
                .cloned()
                .map(ensure_object)
                .transpose()?
                .unwrap_or_else(JsonValue::object),
            embedding_id: optional_string(obj, "embedding_id"),
            // Accept an absent or null stored span as None. An object is parsed as
            // the explicit span; this decoder does not fill missing coordinates
            // from chunk metadata.
            source_span: obj
                .get("source_span")
                .filter(|value| !value.is_null())
                .map(SourceSpan::from_json)
                .transpose()?,
            created_at: required_rfc3339_string(obj, "created_at")?,
        })
    }
}

impl Embedding {
    /// Return the JSON representation of this embedding.
    /// The coordinate array contains JSON numbers in model output order. The
    /// stored model, dimensions and identifiers travel with the vector so its
    /// linkage is available without a second lookup.
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("embedding_id", JsonValue::String(self.embedding_id.clone())),
            ("chunk_id", JsonValue::String(self.chunk_id.clone())),
            (
                "collection_id",
                JsonValue::String(self.collection_id.clone()),
            ),
            (
                "chunk_external_id",
                JsonValue::String(self.chunk_external_id.clone()),
            ),
            ("model_name", JsonValue::String(self.model_name.clone())),
            ("dim", JsonValue::Number(self.dim as f64)),
            // Widen each native f32 to the JSON numeric representation. The
            // separate dim field preserves the recorded declaration independently
            // of how many coordinates are in the array.
            (
                "vector",
                JsonValue::Array(
                    self.vector
                        .iter()
                        .map(|value| JsonValue::Number(*value as f64))
                        .collect(),
                ),
            ),
            ("created_at", JsonValue::String(self.created_at.clone())),
        ])
    }

    /// Decode a stored embedding from JSON, returning an error for an invalid record.
    /// # Errors
    ///
    /// Returns an error for missing required fields, a nonnumeric or nonfinite
    /// binary32 coordinate, or an invalid recorded timestamp.
    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "embedding record must be an object",
            )
        })?;
        // Require a numeric array and finish its conversion before consuming the
        // record's identity fields. A malformed coordinate is reported as a vector
        // error, even when another field is also invalid.
        let vector = required_array(obj, "vector")?
            .iter()
            .map(|value| {
                let value = value.as_f64().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidVectorLength,
                        "embedding vector entries must be numbers",
                    )
                })?;
                // Check finiteness after narrowing. A finite JSON number can
                // overflow to infinity in f32, which would otherwise reach vector
                // comparisons as a seemingly valid coordinate.
                let value = value as f32;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(Error::new(
                        ErrorCode::InvalidVectorLength,
                        "embedding vector entries must be finite f32 values",
                    ))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding_id: required_string(obj, "embedding_id")?,
            chunk_id: required_string(obj, "chunk_id")?,
            collection_id: required_string(obj, "collection_id")?,
            chunk_external_id: required_string(obj, "chunk_external_id")?,
            model_name: required_string(obj, "model_name")?,
            // Read the declared dimension independently of vector length. This
            // record decoder reconstructs the stored fields; dimension agreement
            // with a collection is a separate validation step.
            dim: required_usize(obj, "dim")?,
            vector,
            created_at: required_rfc3339_string(obj, "created_at")?,
        })
    }
}

impl Tombstone {
    #[cfg(test)]
    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("target_type", JsonValue::String(self.target_type.clone())),
            ("target_id", JsonValue::String(self.target_id.clone())),
            ("deleted_at", JsonValue::String(self.deleted_at.clone())),
            ("reason", opt_string(&self.reason)),
        ])
    }

    /// Decodes the tombstone's target type and ID, deletion timestamp and optional
    /// reason from a stored record.
    ///
    /// # Errors
    ///
    /// Returns an error if the record is not an object, a required field is
    /// missing or invalid, or the deletion timestamp cannot be parsed.
    pub fn from_json(value: &JsonValue) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "tombstone record must be an object",
            )
        })?;
        Ok(Self {
            // Keep the target discriminator as stored. Catalog replay interprets
            // the kind and ID together to determine which visible object a
            // deletion affects.
            target_type: required_string(obj, "target_type")?,
            target_id: required_string(obj, "target_id")?,
            deleted_at: required_rfc3339_string(obj, "deleted_at")?,
            reason: optional_string(obj, "reason"),
        })
    }
}

impl DocumentInput {
    // Measure owned capacity, including spare slots in the chunk vector. A slot
    // holds the ChunkInput itself; count each live child's external ID and text
    // buffers separately. JSON metadata has its own reservation.
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.external_id.capacity()
            + self.title.as_ref().map_or(0, String::capacity)
            + self.source_uri.as_ref().map_or(0, String::capacity)
            + self.chunks.capacity() * std::mem::size_of::<ChunkInput>()
            + self
                .chunks
                .iter()
                .map(ChunkInput::workspace_bytes)
                .sum::<usize>()
    }

    /// Check the document ID, metadata and all chunks, including duplicate chunk IDs or ordinals
    /// within this input.
    /// # Errors
    ///
    /// Returns an error for an invalid document or chunk, duplicate chunk
    /// identifiers or ordinals within the document, or insufficient capacity for
    /// validation.
    pub fn validate(&self) -> Result<()> {
        self.validate_in(&crate::json::JsonOperationContext::standalone())
            .map(|_| ())
    }

    // Borrowed input keeps responsibility for its ordinary string and vector
    // allocations. This entry point guards the JSON metadata under the supplied
    // operation and returns that guard to its caller.
    pub(crate) fn validate_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::JsonReservation> {
        self.validate_with_workspace(context, None)
    }

    // The caller owns these inputs and supplies workspace for their native
    // buffers. This path still constructs JSON guards for metadata and returns
    // them for use after validation.
    pub(crate) fn validate_owned_in(
        &self,
        context: &crate::json::JsonOperationContext,
        workspace: &mut crate::json::WorkspaceReservation,
    ) -> Result<crate::json::JsonReservation> {
        self.validate_with_workspace(context, Some(workspace))
    }

    fn validate_with_workspace(
        &self,
        context: &crate::json::JsonOperationContext,
        workspace: Option<&mut crate::json::WorkspaceReservation>,
    ) -> Result<crate::json::JsonReservation> {
        // Keep one JSON guard alive across the entire document. Successfully
        // checked document and chunk metadata accumulate in it, so they cannot
        // each reuse budget that remains occupied by earlier values.
        let mut reservation = context.reservation();
        self.validate_with(
            context,
            &mut |value: &JsonValue, kind| {
                // Select the policy at each call site: a document's root metadata
                // and its child chunks are distinct business values. Their
                // retained allocations still contribute to one operation-wide
                // budget.
                reservation.merge(value.guard_in(kind, context)?.0);
                Ok(())
            },
            workspace,
        )?;
        Ok(reservation)
    }

    // Only the owned JSONL path may reuse its retained, already guarded metadata.
    pub(crate) fn validate_preprocessed(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        // Only skip the redundant guard allocation for previously admitted
        // metadata. Required IDs, metadata shape, per-chunk validation and
        // duplicate detection still run.
        self.validate_with(context, &mut |_, _| Ok(()), None)
    }

    fn validate_with(
        &self,
        context: &crate::json::JsonOperationContext,
        metadata_guard: &mut impl FnMut(&JsonValue, crate::json::JsonInputKind) -> Result<()>,
        workspace: Option<&mut crate::json::WorkspaceReservation>,
    ) -> Result<()> {
        // Trimming here tests for a missing identifier without replacing the
        // caller's spelling. Subsequent duplicate and lookup operations continue
        // to see the supplied string.
        if self.external_id.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "document external_id is required",
            ));
        }
        // Preserve the business field on metadata-limit failures, then check its
        // required object shape. A scalar within the size limit still cannot serve
        // as document metadata.
        metadata_guard(&self.metadata, crate::json::JsonInputKind::DocumentMetadata)
            .map_err(|err| err.with_detail("field", "document.metadata"))?;
        if self.metadata.as_object().is_none() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "metadata/config fields must be JSON objects",
            ));
        }
        // The owned-input path must also account for ordinary string and vector
        // capacity. Admit it before allocating the temporary order arrays needed
        // to detect duplicates.
        if let Some(workspace) = workspace {
            let bytes = self.workspace_bytes();
            workspace.grow_to(bytes, bytes, "batch_prepare")?;
        }
        // Find the first repeated ordinal and the first repeated external ID
        // separately. Each search uses its own short-lived array of positions,
        // leaving the chunk vector in input order.
        let duplicate_ordinal = first_duplicate_index_by(&self.chunks, context, |left, right| {
            left.ordinal.cmp(&right.ordinal)
        })?;
        let duplicate_external_id =
            first_duplicate_index_by(&self.chunks, context, |left, right| {
                left.chunk_external_id.cmp(&right.chunk_external_id)
            })?;
        // Report failures in input order. At a given position, check that chunk's
        // contents first, then prefer the duplicate ordinal error if the same
        // chunk also repeats an external ID.
        for (index, chunk) in self.chunks.iter().enumerate() {
            chunk.validate_with(metadata_guard)?;
            if duplicate_ordinal == Some(index) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "chunk ordinal must be unique within a document",
                ));
            }
            if duplicate_external_id == Some(index) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    format!(
                        "chunk_external_id {} is duplicated within the document input",
                        chunk.chunk_external_id
                    ),
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn first_duplicate_index_by<T>(
    values: &[T],
    context: &crate::json::JsonOperationContext,
    compare: impl Fn(&T, &T) -> std::cmp::Ordering,
) -> Result<Option<usize>> {
    // Singletons and empty inputs need no sort and consume no duplicate-check
    // scratch. This matters for single-chunk writes close to the operation's
    // workspace limit.
    if values.len() < 2 {
        return Ok(None);
    }
    // Sort indices instead of moving or copying the caller's chunks. Charge the
    // index array to this operation for the duration of the check, including when
    // the keys themselves are long strings.
    let mut order =
        crate::json::ReservedVec::with_capacity(values.len(), context, "catalog_build")?;
    for index in 0..values.len() {
        order.push(index);
    }
    // Tie-break by input position so the second member is its first duplicate.
    // Return the earliest duplicate across groups, preserving sequential validation.
    order.as_mut_slice().sort_unstable_by(|left, right| {
        compare(&values[*left], &values[*right]).then_with(|| left.cmp(right))
    });
    // Within an equal-key run every position after the first repeats an earlier
    // input. Taking the smallest original repeated position among runs yields the
    // row that a left-to-right scan would reject first.
    Ok(order
        .as_slice()
        .windows(2)
        .filter(|pair| compare(&values[pair[0]], &values[pair[1]]).is_eq())
        .map(|pair| pair[1])
        .min())
}

impl ChunkInput {
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.chunk_external_id.capacity() + self.text.capacity()
    }

    // The owned import path has already admitted the metadata tree. Reuse the
    // shared structural checks so preprocessed rows cannot bypass the ordinal,
    // text or source-span constraints.
    pub(crate) fn validate_preprocessed(&self) -> Result<()> {
        self.validate_with(&mut |_, _| Ok(()))
    }

    /// Check for a nonblank ID, nonempty text, an ordinal fitting in `u32`, object metadata and
    /// valid source coordinates.
    /// # Errors
    ///
    /// Returns an error for a missing identifier or text, an ordinal outside the
    /// stored range, invalid metadata or source coordinates, or a metadata
    /// resource-limit failure.
    pub fn validate(&self) -> Result<()> {
        let context = crate::json::JsonOperationContext::standalone();
        // This standalone chunk check needs the JSON guard only for validation. A
        // document insert that must retain multiple child values instead supplies
        // a callback that accumulates the guards.
        self.validate_with(&mut |value: &JsonValue, kind| {
            value.guard_in(kind, &context).map(|_| ())
        })
    }

    fn validate_with(
        &self,
        metadata_guard: &mut impl FnMut(&JsonValue, crate::json::JsonInputKind) -> Result<()>,
    ) -> Result<()> {
        // The API uses usize for an ordinal, but the stored value has a 32-bit
        // upper bound. Reject an oversized ordinal before inspecting the other
        // chunk fields.
        if u32::try_from(self.ordinal).is_err() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "chunk ordinal exceeds u32::MAX",
            )
            .with_detail("field", "ordinal"));
        }
        if self.chunk_external_id.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "chunk_external_id is required",
            ));
        }
        // Whitespace has meaning in source text and offsets refer to the supplied
        // content. This check rejects a zero-length string while preserving
        // whitespace-only text; identifiers above must contain a nonspace
        // character.
        if self.text.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "chunk text cannot be empty",
            ));
        }
        // Charge chunk metadata under its own business category so a limit error
        // retains the chunk field name. The object requirement also runs on inputs
        // whose existing reservation is reused.
        metadata_guard(&self.metadata, crate::json::JsonInputKind::ChunkMetadata)
            .map_err(|err| err.with_detail("field", "chunk.metadata"))?;
        if self.metadata.as_object().is_none() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "metadata/config fields must be JSON objects",
            ));
        }
        self.resolved_source_span()?;
        Ok(())
    }

    pub(crate) fn resolved_source_span(&self) -> Result<Option<SourceSpan>> {
        // An explicit source-span object takes precedence as a whole. Consult
        // metadata only when that field is absent; combining endpoints from both
        // sources could create a range never supplied by the caller.
        let span = match &self.source_span {
            Some(span) => {
                span.validate()?;
                Some(span.clone())
            }
            None => SourceSpan::try_from_metadata(&self.metadata)?,
        };
        // Normalize an explicitly empty span to None after applying precedence. It
        // can therefore suppress a populated metadata fallback without becoming a
        // stored all-null span.
        Ok(span.filter(|span| span != &SourceSpan::default()))
    }
}

impl EmbeddingInput {
    // Measure owned input capacity for batch workspace. The vector's storage is in
    // binary32 elements, while the IDs and model names occupy UTF-8 bytes.
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.chunk_external_id.capacity()
            + self.model_name.capacity()
            + self.vector.capacity() * std::mem::size_of::<f32>()
    }

    /// Check the target ID, model name, finite components and length against the collection's
    /// dimension and optional model.
    /// # Errors
    ///
    /// Returns an error for a missing ID or model, an invalid vector shape or
    /// coordinate, or a mismatch with the collection's declared dimension or bound
    /// model.
    pub fn validate(&self, collection: &CollectionInfo) -> Result<()> {
        // Check the input's standalone shape before comparing it with collection
        // settings. A malformed vector is actionable even when it also targets the
        // wrong collection.
        self.validate_structure()?;
        // Target comparison uses the declared width only after the structure check
        // proved it describes the vector. A wrong declaration cannot hide a wrong
        // element count.
        self.validate_target_dimension(collection.embedding_dim)?;
        // When the collection already names a model, require the exact incoming
        // spelling. An unbound collection leaves model admission to the write
        // path.
        if let Some(expected) = &collection.embedding_model {
            if expected != &self.model_name {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "embedding model does not match collection",
                )
                .with_detail("expected", expected)
                .with_detail("actual", &self.model_name));
            }
        }
        Ok(())
    }

    // Check invariants available without reading a collection. Import preflight
    // can reject malformed records before resolving their eventual target.
    pub(crate) fn validate_structure(&self) -> Result<()> {
        // Require a concrete chunk external key in each embedding input. Import
        // resolves that key before binding the payload to a stored chunk.
        if self.chunk_external_id.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding chunk_external_id is required",
            ));
        }
        // Trim only to check presence. The stored and compared model identity
        // remains the input's original string.
        if self.model_name.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding model_name is required",
            ));
        }
        // The declared dimension must be nonzero and match actual element count.
        // Report both values so callers can distinguish the shape from a later
        // target mismatch.
        if self.dim == 0 || self.vector.len() != self.dim {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "embedding dimension must be positive and match its vector length",
            )
            .with_detail("expected", self.dim)
            .with_detail("actual_dim", self.dim)
            .with_detail("actual_len", self.vector.len()));
        }
        // Reject any nonfinite binary32 coordinate before storage or scoring. A
        // valid length does not imply the vector is suitable for distance
        // calculations.
        if self.vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "embedding vector entries must be finite f32 values",
            ));
        }
        Ok(())
    }

    // Compare the embedding declaration against the selected collection width.
    // Callers using it for preflight also validate the vector's own shape.
    pub(crate) fn validate_target_dimension(&self, expected: usize) -> Result<()> {
        if self.dim != expected {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "embedding dimension does not match collection",
            )
            .with_detail("expected", expected)
            .with_detail("actual_dim", self.dim)
            .with_detail("actual_len", self.vector.len()));
        }
        Ok(())
    }
}

impl QueryEmbeddingInput {
    // Charge an already owned typed query value to the current JSON operation. The
    // reservation follows its actual allocated capacity, not just its initialized
    // entries.
    pub(crate) fn reserve_json_work_in(
        &self,
        context: &crate::json::JsonOperationContext,
    ) -> Result<crate::json::JsonReservation> {
        let mut work = context.reservation();
        // Convert element capacity to bytes using saturating arithmetic. A
        // capacity outside the supported work budget cannot wrap to a small
        // request.
        work.grow(
            self.vector
                .capacity()
                .saturating_mul(std::mem::size_of::<f32>()),
            crate::json::JsonInputKind::QueryEmbedding,
        )?;
        // Model text uses a separate string charge. An unspecified model adds no
        // allocation to the query input.
        if let Some(model) = &self.model_name {
            work.grow(model.capacity(), crate::json::JsonInputKind::JsonlRecord)?;
        }
        // Return one reservation for both typed allocations. A caller can move it
        // with the input while the vector and optional name remain owned.
        Ok(work)
    }

    /// Take ownership of a vector and infer its dimension. Leave the model identity unspecified.
    /// This constructor infers shape only. Collection-specific validation runs
    /// when the input is used for a query.
    pub fn from_vector(vector: Vec<f32>) -> Self {
        Self {
            dim: vector.len(),
            model_name: None,
            vector,
        }
    }

    /// Check finite components and length against the collection's dimension. Compare model
    /// names when both are present.
    /// # Errors
    ///
    /// Returns an error if the vector cannot be represented within JSON/vector
    /// policy, the dimension or length differs from the collection, or supplied
    /// model information is invalid or mismatched.
    pub fn validate(&self, collection: &CollectionInfo) -> Result<()> {
        self.validate_structure(collection)?;
        // Compare model names only if both sides supply one. An unspecified
        // request model does not assert a new model binding on a read.
        if let (Some(expected), Some(actual)) = (&collection.embedding_model, &self.model_name) {
            if actual != expected {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "query embedding model does not match collection",
                )
                .with_detail("expected", expected)
                .with_detail("actual", actual));
            }
        }
        Ok(())
    }

    // Apply typed query-vector validation whether the request came from Rust, a
    // host binding or the query parser. Stored embedding rows use a separate
    // validation path.
    pub(crate) fn validate_structure(&self, collection: &CollectionInfo) -> Result<()> {
        // Validate the typed vector under query JSON limits first. Those
        // processing errors can occur before reporting a dimension mismatch.
        crate::json::validate_query_vector(&self.vector)?;
        // Require both the caller's dimension and the actual vector width to agree
        // with the target. Checking either one alone would accept an inconsistent
        // request.
        if self.dim != collection.embedding_dim || self.vector.len() != collection.embedding_dim {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "query embedding dimension does not match collection",
            )
            .with_detail("expected", collection.embedding_dim)
            .with_detail("actual_dim", self.dim)
            .with_detail("actual_len", self.vector.len()));
        }
        if self.vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "query embedding vector entries must be finite f32 values",
            ));
        }
        // An optional model can be omitted, but a supplied name cannot consist
        // entirely of whitespace. This shape check does not normalize valid names.
        if self
            .model_name
            .as_ref()
            .map(|model| model.trim().is_empty())
            .unwrap_or(false)
        {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "query embedding model_name cannot be empty when provided",
            ));
        }
        Ok(())
    }
}

impl CollectionStats {
    /// Return the JSON representation of this collection stats.
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host statistics construction is infallible")
    }

    // For a borrowed result, copy the collection name under the destination
    // builder. The consuming form below can transfer its existing allocation
    // instead.
    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        self.build_json_with_collection(builder.string(&self.collection)?, builder)
    }

    // Move a native statistics summary into a budgeted output without copying the
    // collection name. Other fields use the same serializer as the borrowed path.
    pub(crate) fn into_json_retained(
        mut self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        // Take the owned name first, then give the builder access to its existing
        // workspace credit. Moving it through the JSON result should change its
        // owner without charging a duplicate string.
        let collection = std::mem::take(&mut self.collection);
        let (json, strings) = work.parts_mut();
        let builder = JsonBuilder::budgeted_owned_strings(
            json,
            strings,
            crate::json::JsonInputKind::StoredJson,
        );
        self.build_json_with_collection(builder.owned_string(collection)?, &builder)
    }

    // Keep the result field names and compatibility aliases shared between owned
    // and borrowed statistics. Only the caller's collection-name ownership
    // differs.
    fn build_json_with_collection(
        &self,
        collection: JsonValue,
        builder: &JsonBuilder<'_>,
    ) -> Result<JsonValue> {
        builder.object([
            ("tokenizer", self.tokenizer.build_json(builder)?),
            ("collection", collection),
            (
                "document_count",
                builder.unsigned(self.document_count as u128)?,
            ),
            ("chunk_count", builder.unsigned(self.chunk_count as u128)?),
            (
                "embedding_count",
                builder.unsigned(self.embedding_count as u128)?,
            ),
            // Expose the live counts and tombstone count as separate fields. An
            // alias for deletion remains tied to the tombstone counter, not to a
            // subtraction of live objects.
            (
                "tombstone_count",
                builder.unsigned(self.tombstone_count as u128)?,
            ),
            // Retain deleted_count as the legacy alias of tombstone_count. Derive
            // both from the same value to prevent two descriptions of one count
            // from diverging.
            (
                "deleted_count",
                builder.unsigned(self.tombstone_count as u128)?,
            ),
            (
                "text_index_terms",
                builder.unsigned(self.text_index_terms as u128)?,
            ),
            // Expose segment count separately from term count. A populated text
            // index can organize a different number of terms into a checkpoint's
            // index segments.
            (
                "text_index_segments",
                builder.unsigned(self.text_index_segments as u128)?,
            ),
            ("vector_count", builder.unsigned(self.vector_count as u128)?),
            (
                "index_size_bytes",
                builder.unsigned(u128::from(self.index_size_bytes))?,
            ),
            ("live_bytes", builder.unsigned(u128::from(self.live_bytes))?),
            // Use the live byte count for the compatibility data_size_bytes field.
            // Dead bytes and pending compaction are reported independently.
            (
                "data_size_bytes",
                builder.unsigned(u128::from(self.live_bytes))?,
            ),
            (
                "dead_bytes_estimate",
                builder.unsigned(u128::from(self.dead_bytes_estimate))?,
            ),
            (
                "resource_profile",
                builder.string(self.resource_profile.as_json_str())?,
            ),
            // Both profile fields use the same current resource profile. Existing
            // consumers can keep their preferred field name without changing the
            // reported preset.
            (
                "active_profile",
                builder.string(self.resource_profile.as_json_str())?,
            ),
            // Report the configured cache allowance with the active profile. This
            // field expresses database accounting rather than process memory
            // usage.
            (
                "cache_budget_mb",
                builder.unsigned(self.cache_budget_mb as u128)?,
            ),
            (
                "last_commit_id",
                builder.unsigned(u128::from(self.last_commit_id))?,
            ),
            // Keep the pending compaction estimate alongside the byte breakdown.
            // It is a separate reported quantity rather than the size of another
            // live object.
            (
                "pending_compaction_bytes",
                builder.unsigned(u128::from(self.pending_compaction_bytes))?,
            ),
            (
                "format_version",
                builder.unsigned(self.format_version as u128)?,
            ),
            (
                "background_compaction_enabled",
                JsonValue::Bool(self.background_compaction_enabled),
            ),
        ])
    }
}

impl InsertSummary {
    /// Return the JSON representation of this insert summary.
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    // Build the shared success shape for inserting or importing rows. Totals are
    // derived from the per-record-type counters rather than tracked separately.
    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        let skipped = self.documents_skipped as u128
            + self.chunks_skipped as u128
            + self.embeddings_skipped as u128;
        let replaced = self.documents_replaced as u128
            + self.chunks_replaced as u128
            + self.embeddings_replaced as u128;
        builder.object([
            (
                "documents_inserted",
                builder.unsigned(self.documents_inserted as u128)?,
            ),
            // Preserve the imported spellings as aliases for inserted counts in
            // the same success payload. An alias must always read the same source
            // counter.
            (
                "documents_imported",
                builder.unsigned(self.documents_inserted as u128)?,
            ),
            (
                "chunks_inserted",
                builder.unsigned(self.chunks_inserted as u128)?,
            ),
            (
                "chunks_imported",
                builder.unsigned(self.chunks_inserted as u128)?,
            ),
            (
                "embeddings_inserted",
                builder.unsigned(self.embeddings_inserted as u128)?,
            ),
            (
                "embeddings_imported",
                builder.unsigned(self.embeddings_inserted as u128)?,
            ),
            (
                "documents_skipped",
                builder.unsigned(self.documents_skipped as u128)?,
            ),
            (
                "chunks_skipped",
                builder.unsigned(self.chunks_skipped as u128)?,
            ),
            (
                "embeddings_skipped",
                builder.unsigned(self.embeddings_skipped as u128)?,
            ),
            ("skipped", builder.unsigned(skipped)?),
            (
                "documents_replaced",
                builder.unsigned(self.documents_replaced as u128)?,
            ),
            (
                "chunks_replaced",
                builder.unsigned(self.chunks_replaced as u128)?,
            ),
            (
                "embeddings_replaced",
                builder.unsigned(self.embeddings_replaced as u128)?,
            ),
            ("replaced", builder.unsigned(replaced)?),
            // Preserve the historical failure-counter keys in a successful insert
            // summary. Operational failures are propagated as errors instead of
            // incrementing these placeholder values.
            ("missing_chunks", JsonValue::Number(0.0)),
            ("model_mismatches", JsonValue::Number(0.0)),
            ("errors", JsonValue::Number(0.0)),
        ])
    }
}

impl IndexStats {
    /// Return the JSON representation of this index stats.
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host index statistics construction is infallible")
    }

    // The live-vector count does not depend on checkpoint availability. A
    // rebuild-needed index reports zero checkpoint bytes and no last index commit.
    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("text_terms", builder.unsigned(self.text_terms as u128)?),
            (
                "text_segments",
                builder.unsigned(self.text_segments as u128)?,
            ),
            ("vector_count", builder.unsigned(self.vector_count as u128)?),
            (
                "vector_segments",
                builder.unsigned(self.vector_segments as u128)?,
            ),
            // Expose the rebuild status directly. Consumers should read it
            // alongside the nullable commit field instead of inferring freshness
            // from the live-vector count.
            ("rebuild_needed", JsonValue::Bool(self.rebuild_needed)),
            (
                "index_size_bytes",
                builder.unsigned(u128::from(self.index_size_bytes))?,
            ),
            // Represent an absent index commit as null. Zero would look like an
            // actual commit identity rather than unavailable checkpoint
            // information.
            (
                "last_index_commit_id",
                builder.optional_u64(self.last_index_commit_id)?,
            ),
        ])
    }
}

impl DatabaseStats {
    /// Return the JSON representation of this database stats.
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host database statistics construction is infallible")
    }

    // Use one database statistics schema for typed, CLI and binding projection.
    // Compatibility aliases remain wired to the same underlying totals.
    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            (
                "collection_count",
                builder.unsigned(self.collection_count as u128)?,
            ),
            (
                "document_count",
                builder.unsigned(self.document_count as u128)?,
            ),
            ("chunk_count", builder.unsigned(self.chunk_count as u128)?),
            (
                "embedding_count",
                builder.unsigned(self.embedding_count as u128)?,
            ),
            (
                "tombstone_count",
                builder.unsigned(self.tombstone_count as u128)?,
            ),
            // Both tombstone and deleted counts name the same observation. Keep
            // them synchronized in the aggregate just as in collection statistics.
            (
                "deleted_count",
                builder.unsigned(self.tombstone_count as u128)?,
            ),
            // Preserve the original file_size spelling and its explicit byte-unit
            // alias. Neither key substitutes for live or index-owned bytes.
            ("file_size", builder.unsigned(u128::from(self.file_size))?),
            (
                "file_size_bytes",
                builder.unsigned(u128::from(self.file_size))?,
            ),
            (
                "index_size_bytes",
                builder.unsigned(u128::from(self.index_size_bytes))?,
            ),
            ("live_bytes", builder.unsigned(u128::from(self.live_bytes))?),
            // Use live_bytes as the value of data_size_bytes. Physical file size
            // also includes other extents such as dead or index data.
            (
                "data_size_bytes",
                builder.unsigned(u128::from(self.live_bytes))?,
            ),
            (
                "dead_bytes_estimate",
                builder.unsigned(u128::from(self.dead_bytes_estimate))?,
            ),
            // Serialize the active database resource profile separately from the
            // reported cache allowance. Callers can inspect the preset without
            // inferring a byte budget from it.
            (
                "active_profile",
                builder.string(self.active_profile.as_json_str())?,
            ),
            (
                "cache_budget_mb",
                builder.unsigned(self.cache_budget_mb as u128)?,
            ),
            (
                "last_commit_id",
                builder.unsigned(u128::from(self.last_commit_id))?,
            ),
            (
                "pending_compaction_bytes",
                builder.unsigned(u128::from(self.pending_compaction_bytes))?,
            ),
            (
                "format_version",
                builder.unsigned(self.format_version as u128)?,
            ),
            (
                "background_compaction_enabled",
                JsonValue::Bool(self.background_compaction_enabled),
            ),
        ])
    }
}

#[cfg(test)]
pub(crate) fn chunk_text_hash(text: &str) -> String {
    sha256_hex(text.as_bytes())
}

// Count tokenizer output without materializing a token list. The stored hint is a
// count, so the visitor can discard each token after increasing it.
pub(crate) fn token_count_hint_in(
    text: &str,
    context: &crate::json::JsonOperationContext,
) -> Result<usize> {
    let mut count = 0usize;
    crate::tokenizer::visit_tokens_in(text, context, |_| {
        // Check the counter before recording it. An overflowing token count must
        // not wrap and become a plausible but incorrect hint.
        count = count
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "text token count overflow"))?;
        Ok(())
    })?;
    Ok(count)
}

#[cfg(test)]
pub(crate) fn token_count_hint(text: &str) -> usize {
    crate::tokenizer::tokens(text).count()
}

#[cfg(test)]
fn opt_string(value: &Option<String>) -> JsonValue {
    value
        .as_ref()
        .map(|value| JsonValue::String(value.clone()))
        .unwrap_or(JsonValue::Null)
}

// Convert only finite integral JSON numbers inside the platform's native unsigned
// range. This helper validates numeric shape before the saturating Rust cast.
pub(crate) fn json_usize(value: &JsonValue) -> Option<usize> {
    value.as_f64().and_then(|value| {
        if value.is_finite()
            && value >= 0.0
            // Use an exclusive power-of-two ceiling. On a wide platform the
            // inclusive usize maximum can round to the first unrepresentable
            // integer in binary64.
            && value < (usize::MAX as u128 + 1) as f64
            // Reject fractional inputs before converting. Rust's float-to-integer
            // cast alone would truncate them and accept a different value than the
            // caller supplied.
            && value.fract() == 0.0
        {
            Some(value as usize)
        } else {
            None
        }
    })
}

// Read a source offset as an optional u64. Both the JSON number shape and a
// decimal string representing the full-width value are accepted.
fn optional_offset_field(
    obj: &BTreeMap<String, JsonValue>,
    key: &str,
    container: &str,
) -> Result<Option<u64>> {
    // Treat an absent field and explicit null as missing for span validation.
    // Their counterpart's presence is checked when validating the pair.
    let Some(value) = obj.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    // Decimal strings preserve offsets beyond the exact range of JSON doubles.
    // For a numeric offset use an exclusive upper bound: u64::MAX rounds to
    // 2^64 as f64, and casting that value would silently saturate.
    let number = match value {
        // Accept a nonempty digit string without trimming or a sign. Parsing the
        // string in u64 rejects overflow and preserves precision beyond binary64.
        JsonValue::String(text) if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) => {
            text.parse::<u64>().ok()
        }
        JsonValue::Number(number)
            if number.is_finite()
                && *number >= 0.0
                && *number < 18_446_744_073_709_551_616.0
                && number.fract() == 0.0 =>
        {
            // This cast runs only for an exactly integral numeric value strictly
            // below the upper boundary. The decimal string form handles full-u64
            // values that JSON numbers cannot carry exactly.
            Some(*number as u64)
        }
        _ => None,
    };
    // Reject a present invalid value with its complete qualified path. Returning
    // None is reserved for fields that are truly absent by value or by key.
    number.map(Some).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{container}.{key} must be a u64 integer or unsigned decimal string"),
        )
        .with_detail("field", format!("{container}.{key}"))
    })
}

// Read an optional u32 source line field. Use a JSON number because every u32
// integer is exactly representable in binary64.
fn optional_line_field(
    obj: &BTreeMap<String, JsonValue>,
    key: &str,
    container: &str,
) -> Result<Option<u32>> {
    // Treat both absence and null as no value. The caller later checks that line
    // start and end appear together.
    let Some(value) = obj.get(key).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    // Check the whole nonnegative u32 range before casting. The span validator
    // separately applies the one-based lower bound when lines are supplied.
    let number = value.as_f64().filter(|value| {
        value.is_finite() && *value >= 0.0 && *value <= u32::MAX as f64 && value.fract() == 0.0
    });
    // Keep invalid type, fractional line numbers and out-of-range values under the
    // same qualified schema field. A successful conversion still needs the span
    // pair's presence and ordering checks.
    number.map(|number| Some(number as u32)).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{container}.{key} must be a u32 integer"),
        )
        .with_detail("field", format!("{container}.{key}"))
    })
}

// Validate paired optional boundaries. Byte offsets and line positions share the
// same ordering rule while the caller supplies their different minimum values.
fn validate_span_pair(
    start: Option<u64>,
    end: Option<u64>,
    start_name: &str,
    end_name: &str,
    min_value: u64,
) -> Result<()> {
    match (start, end) {
        // When both endpoints exist, check each against the coordinate minimum
        // before comparing them. This preserves the diagnostic that points at the
        // malformed endpoint.
        (Some(start), Some(end)) => {
            if start < min_value {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{start_name} must be at least {min_value}"),
                ));
            }
            if end < min_value {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{end_name} must be at least {min_value}"),
                ));
            }
            // Allow equal endpoints. A byte range can be empty and a single-line
            // span naturally has the same start and end line.
            if start > end {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("{start_name} must be <= {end_name}"),
                ));
            }
            Ok(())
        }
        // Both endpoints may be omitted together. If just one is present, name the
        // missing counterpart so the caller can repair the incomplete span.
        (None, None) => Ok(()),
        (Some(_), None) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{end_name} is required when {start_name} is present"),
        )),
        (None, Some(_)) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{start_name} is required when {end_name} is present"),
        )),
    }
}

// Validate the stored timestamp with the engine's RFC3339 parser while retaining
// the original field text. Formatting can preserve the value's supplied spelling.
fn required_rfc3339_string(obj: &BTreeMap<String, JsonValue>, key: &str) -> Result<String> {
    let value = required_string(obj, key)?;
    if time::parse_rfc3339_seconds(&value).is_none() {
        return Err(Error::new(
            ErrorCode::FileFormatError,
            format!("{key} must be an RFC3339 timestamp"),
        ));
    }
    Ok(value)
}

// Build the host representation with a nonbudgeted JSON builder. Retained runtime
// outputs can use the same shape through a budgeted builder.
pub(crate) fn vector_index_to_json(config: &VectorIndexConfig) -> JsonValue {
    vector_index_build_json(config, &JsonBuilder::host())
        .expect("host vector configuration construction is infallible")
}

// Serialize vector configuration by variant. The exact index has a type tag only;
// ANN retains its provider and parameters for format compatibility.
fn vector_index_build_json(
    config: &VectorIndexConfig,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    match config {
        VectorIndexConfig::Exact => builder.object([("type", builder.string("exact")?)]),
        VectorIndexConfig::Ann { provider, params } => builder.object([
            ("type", builder.string("ann")?),
            ("provider", builder.string(provider)?),
            // Copy the already validated parameter value through the builder
            // rather than cloning around its reservation.
            ("params", builder.copy_validated_value(params)?),
        ]),
    }
}

// Decode the stored vector-index configuration independently of whether the engine
// can execute it. Feature and collection validation decide admission later.
pub(crate) fn vector_index_from_json(value: &JsonValue) -> Result<VectorIndexConfig> {
    let obj = value
        .as_object()
        .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "vector_index must be an object"))?;
    // Dispatch on the stored type string. A known ANN shape retains its provider
    // and opens an empty parameters object when the field is absent.
    match required_string(obj, "type")?.as_str() {
        "exact" => Ok(VectorIndexConfig::Exact),
        "ann" => Ok(VectorIndexConfig::Ann {
            provider: required_string(obj, "provider")?,
            // Use an empty parameters object for omitted parameters only. A
            // present value is passed through for subsequent configuration
            // validation.
            params: obj.get("params").cloned().unwrap_or_else(JsonValue::object),
        }),
        // An unknown type string is an unsupported feature. Missing or nonstring
        // type and provider values use the required-field error from the string
        // decoder.
        other => Err(Error::new(
            ErrorCode::UnsupportedFeatureFlag,
            format!("unsupported vector index type {other}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_collection_fields_preserve_ann_params_and_legacy_shape() {
        let params = object_from_pairs([
            ("ef_search", JsonValue::Number(31.0)),
            ("label", JsonValue::String("\u{4e2d}\"\\".into())),
        ]);
        let info = CollectionInfo {
            collection_id: "id".into(),
            name: "collection".into(),
            embedding_dim: 3,
            embedding_model: None,
            distance_metric: DistanceMetric::DotProduct,
            vectors_normalized: true,
            tokenizer: Default::default(),
            vector_index: VectorIndexConfig::Ann {
                provider: "legacy-provider".into(),
                params: params.clone(),
            },
            created_at: "2026-10-11T00:00:00Z".into(),
        };
        let context = crate::json::JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut work = context.reservation();
        let value = info
            .build_json(&JsonBuilder::budgeted(
                &mut work,
                crate::json::JsonInputKind::StoredJson,
            ))
            .unwrap();
        assert_eq!(value, info.to_json());
        assert_eq!(value.get("embedding_model"), Some(&JsonValue::Null));
        assert_eq!(
            value.get("vector_index").unwrap().get("params"),
            Some(&params)
        );
        assert_eq!(value.get("config").unwrap().as_object().unwrap().len(), 1);
        assert!(work.reserved_bytes() > 0);
        drop((value, work));
        assert_eq!(context.available_workspace_bytes(), available);
    }

    #[test]
    fn budgeted_token_hints_preserve_empty_stop_words_and_symbol_counts() {
        let context = crate::json::JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        for text in [
            "",
            "the and",
            "THE AND",
            "the_and",
            "fooFoo fooFoo",
            "FOO_BAR Foo_foo",
            "\u{23a}X_\u{130}",
            "\u{39f}\u{3a3}_\u{39f}\u{3c3}_\u{3bf}\u{3c3}",
        ] {
            assert_eq!(
                token_count_hint_in(text, &context).unwrap(),
                token_count_hint(text)
            );
            assert_eq!(context.available_workspace_bytes(), available);
        }
        assert_eq!(token_count_hint_in("the and", &context).unwrap(), 0);
        for text in ["", "the and", "rust rust", "foo_bar foo_bar"] {
            let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(0, 0, 0),
            ));
            assert_eq!(
                token_count_hint_in(text, &context).unwrap(),
                token_count_hint(text)
            );
            assert_eq!(context.available_workspace_bytes(), 0);
        }
    }

    #[test]
    fn budgeted_token_hints_admit_exact_payload_peak_on_both_axes_and_allow_retry() {
        let minimum = "PROJECT".len();
        for parent in [false, true] {
            for limit in [minimum - 1, minimum, minimum + 1] {
                let context = crate::json::JsonOperationContext::new(std::sync::Arc::new(
                    crate::json::WorkLedger::with_limits(
                        if parent { 2 * minimum } else { limit },
                        0,
                        if parent { limit } else { 2 * minimum },
                    ),
                ));
                let result = token_count_hint_in("rust PROJECT", &context);
                if limit < minimum {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", minimum.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                    }
                } else {
                    assert_eq!(result.unwrap(), 2);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
                assert_eq!(token_count_hint_in("rust", &context).unwrap(), 1);
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn owned_chunk_dto_moves_payloads_and_matches_public_contract() {
        use crate::json::{JsonInputKind, JsonOperationContext, JsonOutputSchema};
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let metadata =
                object_from_pairs([("value", JsonValue::String(format!("value-{seed}")))]);
            let mut work = metadata
                .guard_in(JsonInputKind::StoredJson, &context)
                .unwrap()
                .0;
            let mut text = String::with_capacity(128 + seed);
            text.push_str("payload\n\u{4e2d}");
            let pointer = text.as_ptr();
            let chunk = Chunk {
                chunk_id: format!("chk_{seed}"),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: format!("chunk-{seed}"),
                text,
                text_hash: "hash".into(),
                ordinal: seed,
                token_count: 2,
                metadata,
                embedding_id: (seed % 2 == 0).then(|| "emb_1".into()),
                source_span: Some(SourceSpan {
                    start: Some(u64::MAX - 1),
                    end: Some(u64::MAX),
                    line_start: Some(1),
                    line_end: Some(2),
                }),
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let expected = object_from_pairs([
                ("chunk_id", JsonValue::String(chunk.chunk_id.clone())),
                ("document_id", JsonValue::String("doc_1".into())),
                ("collection_id", JsonValue::String("col_1".into())),
                (
                    "chunk_external_id",
                    JsonValue::String(chunk.chunk_external_id.clone()),
                ),
                ("text", JsonValue::String(chunk.text.clone())),
                ("text_hash", JsonValue::String("hash".into())),
                ("ordinal", JsonValue::Number(seed as f64)),
                ("token_count", JsonValue::Number(2.0)),
                ("metadata", chunk.metadata.clone()),
                ("embedding_id", opt_string(&chunk.embedding_id)),
                (
                    "source_span",
                    object_from_pairs([
                        ("start", JsonValue::String("18446744073709551614".into())),
                        ("end", JsonValue::String("18446744073709551615".into())),
                        ("line_start", JsonValue::Number(1.0)),
                        ("line_end", JsonValue::Number(2.0)),
                    ]),
                ),
                ("created_at", JsonValue::String(chunk.created_at.clone())),
            ]);
            assert_eq!(chunk.to_json(), expected);
            let value = chunk
                .build_json(&JsonBuilder::budgeted(
                    &mut work,
                    crate::json::JsonInputKind::StoredJson,
                ))
                .unwrap();
            assert_eq!(value, expected);
            assert_eq!(
                value.get("text").unwrap().as_str().unwrap().as_ptr(),
                pointer
            );
            let measured = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            let (output, _) = value
                .pretty_retained_in(
                    JsonOutputSchema::Value(JsonInputKind::StoredJson),
                    &context,
                    &mut work,
                )
                .unwrap();
            assert_eq!(output, expected.to_pretty_string() + "\n");
        }
    }

    #[test]
    fn duplicate_scan_borrows_keys_and_reserves_only_its_index_array() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        use std::cell::Cell;
        use std::collections::BTreeSet;

        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let context = JsonOperationContext::standalone();
            let values = (0..seed + 2)
                .map(|index| format!("{}{}", "long-key".repeat(1024), (index * 37 + seed) % 23))
                .collect::<Vec<_>>();
            let mut seen = BTreeSet::new();
            let expected = values.iter().position(|value| !seen.insert(value.as_str()));
            let bytes = values.len() * std::mem::size_of::<usize>();
            let mut held = context.workspace_reservation();
            held.grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                .unwrap();
            let error = first_duplicate_index_by(&values, &context, |_, _| {
                panic!("comparison started before reserving the index array")
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert_eq!(error_detail_value(&error, "resource"), Some("memory"));
            assert_eq!(
                error_detail_value(&error, "operation"),
                Some("catalog_build")
            );
            assert_eq!(
                error_detail_value(&error, "limit_bytes"),
                Some(limit.to_string().as_str())
            );
            assert_eq!(
                error_detail_value(&error, "requested_bytes"),
                Some((limit + 1).to_string().as_str())
            );
            held.retain(limit - bytes);
            let checked = Cell::new(false);
            let actual = first_duplicate_index_by(&values, &context, |left, right| {
                if !checked.replace(true) {
                    assert!(context
                        .workspace_reservation()
                        .grow_to(1, 1, "operation_scratch")
                        .is_err());
                }
                left.cmp(right)
            })
            .unwrap();
            assert_eq!(actual, expected);
            assert!(checked.get());
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            for short in [&values[..0], &values[..1]] {
                assert_eq!(
                    first_duplicate_index_by(short, &context, String::cmp).unwrap(),
                    None
                );
            }
        }
    }

    #[test]
    fn sorted_duplicate_checks_preserve_sequential_document_error_precedence() {
        use std::collections::BTreeSet;

        for seed in 0..100usize {
            let mut input = DocumentInput {
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..12)
                    .map(|index| ChunkInput {
                        chunk_external_id: format!("chunk-{index}"),
                        text: "text".into(),
                        ordinal: index,
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            };
            input.chunks[3 + seed % 4].ordinal = 0;
            input.chunks[2 + (seed * 7) % 8].chunk_external_id = "chunk-0".into();
            if seed % 3 != 0 {
                input.chunks[(seed * 11) % 12].text.clear();
            }
            let sequential = (|| {
                let mut ordinals = BTreeSet::new();
                let mut external_ids = BTreeSet::new();
                for chunk in &input.chunks {
                    chunk.validate()?;
                    if !ordinals.insert(chunk.ordinal) {
                        return Err(Error::new(
                            ErrorCode::InvalidJsonlSchema,
                            "chunk ordinal must be unique within a document",
                        ));
                    }
                    if !external_ids.insert(chunk.chunk_external_id.as_str()) {
                        return Err(Error::new(
                            ErrorCode::DuplicateExternalId,
                            format!(
                                "chunk_external_id {} is duplicated within the document input",
                                chunk.chunk_external_id
                            ),
                        ));
                    }
                }
                Ok(())
            })()
            .unwrap_err();
            assert_eq!(
                input.validate().unwrap_err().to_json(),
                sequential.to_json()
            );
            assert_eq!(
                input
                    .validate_preprocessed(&crate::json::JsonOperationContext::standalone())
                    .unwrap_err()
                    .to_json(),
                sequential.to_json()
            );
        }
    }

    #[test]
    fn document_and_chunk_metadata_share_one_validation_work_budget() {
        let metadata =
            || object_from_pairs([("value", JsonValue::String("x".repeat(7 * 1024 * 1024)))]);
        let mut input = DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: metadata(),
            chunks: (0..3)
                .map(|ordinal| ChunkInput {
                    chunk_external_id: format!("chunk-{ordinal}"),
                    text: "text".into(),
                    ordinal,
                    metadata: metadata(),
                    source_span: None,
                })
                .collect(),
        };
        assert!(input.validate().is_ok());
        input.chunks.push(ChunkInput {
            chunk_external_id: "chunk-3".into(),
            text: "text".into(),
            ordinal: 3,
            metadata: metadata(),
            source_span: None,
        });
        let error = input.validate().unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_input_kind" && value == "chunk_metadata"));
        input.chunks.pop();
        assert!(input.validate().is_ok());
    }

    fn collection() -> CollectionInfo {
        CollectionInfo {
            tokenizer: Default::default(),
            collection_id: "col_1".into(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("test-model".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            created_at: "1970-01-01T00:00:00Z".into(),
        }
    }

    fn error_detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn collection_policy_counts_the_complete_normalized_object_before_cloning() {
        let mut config = CollectionConfig {
            tokenizer: Default::default(),
            name: "a".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("m".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        };
        let normalized = object_from_pairs([
            ("name", JsonValue::String(config.name.clone())),
            ("embedding_model", opt_string(&config.embedding_model)),
            ("embedding_dim", JsonValue::Number(2.0)),
            ("distance_metric", JsonValue::String("cosine".into())),
            ("vectors_normalized", JsonValue::Bool(false)),
            ("vector_index", vector_index_to_json(&config.vector_index)),
            ("tokenizer", config.tokenizer.to_json()),
        ]);
        let fixed_bytes = normalized.to_compact_string().len() - 2;
        let limit = crate::json::JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES;
        config.name = "n".repeat(4 * 1_048_576);
        config.embedding_model = Some("m".repeat(limit - fixed_bytes - config.name.len()));
        config.validate().unwrap();
        config.name.push('n');
        let error = config.validate().unwrap_err();
        assert_eq!(
            error_detail_value(&error, "json_limit_reason"),
            Some("unit_bytes_exceeded")
        );
        assert_eq!(
            error_detail_value(&error, "json_input_kind"),
            Some("collection_config")
        );
        assert_eq!(
            error_detail_value(&error, "actual"),
            Some((limit + 1).to_string().as_str())
        );
        config.name = "\n".repeat(limit / 2);
        config.name.push('n');
        config.embedding_model = None;
        assert_eq!(
            error_detail_value(&config.validate().unwrap_err(), "json_limit_reason"),
            Some("unit_bytes_exceeded")
        );
        config.name = String::with_capacity(crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES);
        config.name.push('n');
        assert_eq!(
            error_detail_value(&config.validate().unwrap_err(), "json_limit_reason"),
            Some("work_budget_exceeded")
        );
    }

    #[test]
    fn document_input_rejects_duplicate_chunk_external_ids() {
        let input = DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![
                ChunkInput {
                    chunk_external_id: "doc#1".into(),
                    text: "first".into(),
                    ordinal: 1,
                    metadata: JsonValue::object(),
                    source_span: None,
                },
                ChunkInput {
                    chunk_external_id: "doc#1".into(),
                    text: "second".into(),
                    ordinal: 2,
                    metadata: JsonValue::object(),
                    source_span: None,
                },
            ],
        };
        assert_eq!(
            input.validate().unwrap_err().code(),
            ErrorCode::DuplicateExternalId
        );
    }

    #[test]
    fn source_span_json_is_exact_at_safe_integer_and_u64_boundaries() {
        for value in [
            0,
            9_007_199_254_740_991,
            9_007_199_254_740_992,
            9_007_199_254_740_993,
            u64::MAX,
        ] {
            let span = SourceSpan {
                start: Some(value),
                end: Some(value),
                line_start: Some(u32::MAX),
                line_end: Some(u32::MAX),
            };
            let json = span.to_json();
            assert_eq!(
                json.get("start").unwrap().as_str().is_some(),
                value > 9_007_199_254_740_991
            );
            assert_eq!(
                SourceSpan::from_json(&crate::json::parse(&json.to_compact_string()).unwrap())
                    .unwrap(),
                span
            );
        }
        let numeric =
            crate::json::parse(r#"{"start":9223372036854775808,"end":9223372036854775808}"#)
                .unwrap();
        assert_eq!(
            SourceSpan::from_json(&numeric).unwrap().start,
            Some(1u64 << 63)
        );
        let padded = crate::json::parse(r#"{"start":"0001","end":"0002"}"#).unwrap();
        assert_eq!(
            SourceSpan::from_json(&padded)
                .unwrap()
                .to_json()
                .get("start"),
            Some(&JsonValue::Number(1.0))
        );
    }

    #[test]
    fn source_span_rejects_invalid_decimal_offsets_and_line_overflow() {
        for raw in [
            "",
            "+1",
            "-1",
            "1.0",
            "1e3",
            " 1",
            "1 ",
            "18446744073709551616",
            "not-a-number",
        ] {
            let value = object_from_pairs([
                ("start", JsonValue::String(raw.into())),
                ("end", JsonValue::String(raw.into())),
            ]);
            assert_eq!(
                SourceSpan::from_json(&value).unwrap_err().code(),
                ErrorCode::InvalidJsonlSchema,
                "{raw}"
            );
        }
        for raw in [
            r#"{"start":18446744073709551616,"end":18446744073709551616}"#,
            r#"{"line_start":4294967296,"line_end":4294967296}"#,
            r#"{"line_start":"1","line_end":"2"}"#,
            r#"{"start":1.5,"end":2}"#,
            r#"{"start":"3","end":"2"}"#,
            r#"{"start":"18446744073709551615"}"#,
        ] {
            let value = crate::json::parse(raw).unwrap();
            assert!(SourceSpan::from_json(&value).is_err(), "{raw}");
        }
        assert!(
            crate::json::parse(r#"{"start":9007199254740993,"end":9007199254740993}"#).is_err()
        );
    }

    #[test]
    fn source_span_from_json_rejects_invalid_boundaries() {
        let inverted_bytes = object_from_pairs([
            ("start", JsonValue::Number(20.0)),
            ("end", JsonValue::Number(10.0)),
        ]);
        let err = SourceSpan::from_json(&inverted_bytes).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("source_span.start"));

        let partial_lines = object_from_pairs([("line_end", JsonValue::Number(10.0))]);
        let err = SourceSpan::from_json(&partial_lines).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("source_span.line_start"));

        let zero_line = object_from_pairs([
            ("line_start", JsonValue::Number(0.0)),
            ("line_end", JsonValue::Number(10.0)),
        ]);
        let err = SourceSpan::from_json(&zero_line).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("at least 1"));
    }

    #[test]
    fn chunk_input_rejects_invalid_metadata_source_span() {
        let input = ChunkInput {
            chunk_external_id: "doc#1".into(),
            text: "first".into(),
            ordinal: 1,
            metadata: object_from_pairs([
                ("byte_start", JsonValue::Number(42.0)),
                ("byte_end", JsonValue::Number(7.0)),
            ]),
            source_span: None,
        };

        let err = input.validate().unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("source_span.start"));
    }

    #[test]
    fn source_span_accepts_valid_line_only_metadata() {
        let metadata = object_from_pairs([
            ("line_start", JsonValue::Number(3.0)),
            ("line_end", JsonValue::Number(8.0)),
        ]);

        let span = SourceSpan::try_from_metadata(&metadata)
            .unwrap()
            .expect("line metadata should produce source span");
        assert_eq!(span.start, None);
        assert_eq!(span.end, None);
        assert_eq!(span.line_start, Some(3));
        assert_eq!(span.line_end, Some(8));
    }

    #[test]
    fn embedding_input_rejects_non_finite_vectors() {
        let input = EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "test-model".into(),
            dim: 2,
            vector: vec![1.0, f32::NAN],
        };
        assert_eq!(
            input.validate(&collection()).unwrap_err().code(),
            ErrorCode::InvalidVectorLength
        );
    }

    #[test]
    fn embedding_input_rejects_blank_model_name() {
        let input = EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: " \t ".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        let mut collection = collection();
        collection.embedding_model = None;

        let err = input.validate(&collection).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("model_name"));
    }

    #[test]
    fn query_embedding_rejects_blank_model_name_when_provided() {
        let input = QueryEmbeddingInput {
            model_name: Some(" ".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        };
        let mut collection = collection();
        collection.embedding_model = None;

        let err = input.validate(&collection).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("model_name"));
    }

    #[test]
    fn search_weights_must_sum_to_one() {
        let err = SearchWeights::new(2.0, 1.0, 0.0).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert!(err.message().contains("sum to 1.0"));
        assert_eq!(error_detail_value(&err, "field"), Some("weights"));
        assert_eq!(error_detail_value(&err, "reason"), Some("sum_mismatch"));
        assert_eq!(
            error_detail_value(&err, "weights_reason"),
            Some("sum_mismatch")
        );
        assert_eq!(error_detail_value(&err, "expected_sum"), Some("1"));
        assert_eq!(error_detail_value(&err, "actual_sum"), Some("3"));
    }

    #[test]
    fn search_weights_validation_errors_include_machine_readable_detail() {
        let negative = SearchWeights::new(-0.1, 1.1, 0.0).unwrap_err();
        assert_eq!(negative.code(), ErrorCode::InvalidSearchWeights);
        assert_eq!(error_detail_value(&negative, "field"), Some("weights.bm25"));
        assert_eq!(
            error_detail_value(&negative, "reason"),
            Some("finite_non_negative")
        );
        assert_eq!(error_detail_value(&negative, "actual"), Some("-0.1"));

        let zero = SearchWeights::new(0.0, 0.0, 0.0).unwrap_err();
        assert_eq!(zero.code(), ErrorCode::InvalidSearchWeights);
        assert_eq!(error_detail_value(&zero, "field"), Some("weights"));
        assert_eq!(error_detail_value(&zero, "reason"), Some("all_zero"));
        assert_eq!(error_detail_value(&zero, "actual_sum"), Some("0"));
    }

    #[test]
    fn weights_reason_covers_non_finite_negative_zero_and_sum_errors() {
        for seed in 0..100 {
            let mut values = [0.25, 0.25, 0.5];
            let reason = match seed % 6 {
                0..=3 => {
                    values[seed / 6 % 3] = match seed % 6 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        _ => -(seed as f64 + 1.0),
                    };
                    "finite_non_negative"
                }
                4 => {
                    values = [0.0, -0.0, 0.0];
                    "all_zero"
                }
                _ => {
                    values[seed / 6 % 3] = seed as f64 + 1.0;
                    "sum_mismatch"
                }
            };
            let weights = SearchWeights {
                bm25: values[0],
                vector: values[1],
                metadata: values[2],
            };
            for error in [
                weights.validate().unwrap_err(),
                SearchWeights::new(values[0], values[1], values[2]).unwrap_err(),
            ] {
                assert_eq!(error.code(), ErrorCode::InvalidSearchWeights);
                assert_eq!(error_detail_value(&error, "weights_reason"), Some(reason));
                assert_eq!(error_detail_value(&error, "reason"), Some(reason));
                let json = crate::json::parse(&error.to_json().to_compact_string()).unwrap();
                assert_eq!(
                    json.get("detail")
                        .unwrap()
                        .get("weights_reason")
                        .and_then(JsonValue::as_str),
                    Some(reason)
                );
            }
        }
    }

    #[test]
    fn collection_config_rejects_blank_embedding_model() {
        let config = CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("  ".into()),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        };

        let err = config.validate().unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert!(err.message().contains("embedding_model"));
    }

    #[test]
    fn collection_config_rejects_reserved_ann_config() {
        let config = CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Ann {
                provider: "reserved".into(),
                params: JsonValue::object(),
            },
        };
        assert_eq!(
            config.validate().unwrap_err().code(),
            ErrorCode::UnsupportedFeatureFlag
        );
    }

    #[test]
    fn collection_record_rejects_invalid_timestamp() {
        let value = object_from_pairs([
            ("collection_id", JsonValue::String("col_1".into())),
            ("name", JsonValue::String("notes".into())),
            ("embedding_dim", JsonValue::Number(2.0)),
            ("distance_metric", JsonValue::String("cosine".into())),
            ("embedding_model", JsonValue::Null),
            ("vectors_normalized", JsonValue::Bool(false)),
            (
                "vector_index",
                vector_index_to_json(&VectorIndexConfig::Exact),
            ),
            ("created_at", JsonValue::String("not-a-time".into())),
        ]);
        assert_eq!(
            CollectionInfo::from_json(&value).unwrap_err().code(),
            ErrorCode::FileFormatError
        );
    }
}
