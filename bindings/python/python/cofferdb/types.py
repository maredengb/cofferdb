"""Public input and owned-result shapes for the synchronous native API.

These annotations do not coerce values. Runtime validation requires exact
built-in containers and scalar types, and rejects bool for numeric arguments.
"""

from typing import Literal, NotRequired, TypeAlias, TypedDict, TypeVar

#: Resource preset for cache and search budgets: expanded, balanced (default), or
#: constrained.
Profile: TypeAlias = Literal["expanded", "balanced", "constrained"]
#: How a write handles an existing external identity: reject it, skip it, or replace it.
Conflict: TypeAlias = Literal["error", "skip", "replace"]
#: Scoring rule for exact vector search: cosine, dot_product or euclidean.
Metric: TypeAlias = Literal["cosine", "dot_product", "euclidean"]
#: Write synchronization policy: commit (default) or fast.
Durability: TypeAlias = Literal["commit", "fast"]
#: File I/O selection for an owner: auto (default), read or mmap_disabled.
IoMode: TypeAlias = Literal["auto", "read", "mmap_disabled"]
# Python's type system treats bool as a subclass of int. The binding checks exact
# built-in types at runtime, so static acceptance of a boolean does not make it a
# valid numeric input.
#: Numeric input or result represented by a built-in int or float. Numeric inputs exclude
#: bool.
Number: TypeAlias = int | float
#: Recursive JSON value. Arrays accept built-in lists or tuples on input and return as owned
#: lists.
JsonValue: TypeAlias = None | bool | Number | str | list["JsonValue"] | tuple["JsonValue", ...] | dict[str, "JsonValue"]
#: JSON object with built-in str keys and recursively representable JSON values.
JsonObject: TypeAlias = dict[str, JsonValue]
_T = TypeVar("_T")
#: Finite built-in list or tuple of records. Convert generators explicitly before submitting
#: a batch.
BatchSeq: TypeAlias = list[_T] | tuple[_T, ...]
#: Finite built-in list or tuple of numeric components representable as finite f32 values.
Vector: TypeAlias = list[Number] | tuple[Number, ...]


# total=False controls whether a key is required. Each value annotation separately
# determines whether the caller can supply None, a distinction used by the native
# configuration parser.
class DatabaseConfigInput(TypedDict, total=False):
    """Runtime settings for a new writer. Omitted values use core defaults. Only
    cache_budget_mb accepts None.
    """

    #: Resource preset used to size cache and query budgets; the default is balanced.
    resource_profile: Profile
    #: Optional cache budget in MiB. None uses the preset budget; zero is an explicit budget.
    cache_budget_mb: int | None
    #: Background compaction setting. Leave false: this version rejects enabling it.
    background_compaction: bool
    #: Commit synchronization policy for the owner. The default is commit.
    durability: Durability
    #: File I/O mode for the owner. Omission selects auto; None is invalid.
    io_mode: IoMode


class OpenOptionsInput(TypedDict, total=False):
    """Per-owner settings when opening an existing file. Omitted or nullable overrides use
    runtime defaults and leave stored collection settings unchanged.
    """

    #: Resource preset used to size cache and query budgets; the default is balanced.
    resource_profile: Profile | None
    #: Optional cache budget in MiB. None uses the preset budget; zero is an explicit budget.
    cache_budget_mb: int | None
    #: Background compaction setting. Leave false: this version rejects enabling it.
    background_compaction: bool | None
    #: Commit synchronization policy for the owner. The default is commit.
    durability: Durability | None
    #: File I/O mode for the owner. Omission selects auto; None is invalid.
    io_mode: IoMode


# These literals describe one tokenizer identity. The native decoder checks the
# complete structure against its canonical configuration, so changing an individual
# rule is not a way to customize tokenization.
class TokenizerRules(TypedDict):
    """Required, fixed text normalization, splitting and stop-word rules for the supported
    tokenizer.
    """

    #: Lowercase normalization used when indexing and matching tokens.
    case_fold: Literal["lower"]
    #: Fixed rule used to divide input text into searchable terms.
    split: Literal["unicode_punctuation"]
    #: Built-in English stop-word list applied by this tokenizer.
    stop_words: Literal["default_en"]


class TokenizerConfig(TypedDict):
    """Complete immutable tokenizer identity shared by ingestion and search. An explicit
    configuration must supply every field.
    """

    #: Fixed tokenizer name default_en.
    name: Literal["default_en"]
    #: Version identifier of the stored tokenizer behavior.
    version: Literal["0.1"]
    #: Fixed latin_code_default tokenization profile.
    profile: Literal["latin_code_default"]
    #: Language hint recorded in the tokenizer identity.
    language_hint: Literal["en"]
    #: Complete tokenizer rule set required by this tokenizer identity.
    config: TokenizerRules


class CollectionConfigInput(TypedDict):
    """Schema for a named collection. Set the dimension and scoring metric before writing
    any embeddings.
    """

    #: Collection name used for lookup by the application.
    name: str
    #: Number of components required per vector, from 1 through 2**32-1.
    embedding_dim: int
    #: Metric used to compare query vectors to stored embeddings.
    distance_metric: Metric
    #: Explicit model name, or None if unspecified. Committed embeddings still determine the
    #: effective model binding.
    embedding_model: NotRequired[str | None]
    #: Whether the collection declares its supplied vectors normalized. Defaults to False.
    vectors_normalized: NotRequired[bool]
    #: Vector search implementation. Omission selects exact search.
    vector_index: NotRequired[Literal["exact"]]
    #: Omission selects the complete default_en tokenizer; explicit None is invalid.
    tokenizer: NotRequired[TokenizerConfig]


class SourceSpanInput(TypedDict, total=False):
    """Optional provenance supplied with a chunk. Supply both ends of each known range.
    Byte offsets accept unsigned 64-bit integers or decimal strings.
    """

    #: Zero-based first byte of the range, paired with end. None means unspecified.
    start: int | str | None
    #: Exclusive byte end, paired with start. Equal endpoints describe an empty range.
    end: int | str | None
    #: First source line, starting at 1, paired with line_end; None means unspecified.
    line_start: int | None
    #: Inclusive last source line, at least line_start. None means unspecified.
    line_end: int | None


# The input mapping can omit a coordinate pair, while the stored mapping publishes
# each coordinate key. Use decimal strings for wide offsets in the projected result
# so a round trip through the core's binary64 JSON numbers cannot lose a byte
# position.
class SourceSpan(TypedDict):
    """Stored provenance of a chunk. Unknown endpoints are None. Byte offsets above
    2**53-1 are returned as exact decimal strings.
    """

    #: Zero-based first byte of the range, paired with end. None means unspecified.
    start: int | str | None
    #: Exclusive byte end, paired with start. Equal endpoints describe an empty range.
    end: int | str | None
    #: First source line, starting at 1, paired with line_end; None means unspecified.
    line_start: int | None
    #: Inclusive last source line, at least line_start. None means unspecified.
    line_end: int | None


class DocumentRecord(TypedDict):
    """Document identity and descriptive data to write to a collection."""

    #: Application-provided document identifier within this collection.
    external_id: str
    #: Optional display title stored with the document.
    title: NotRequired[str | None]
    #: Optional source locator stored as text; reading a record does not fetch it.
    source_uri: NotRequired[str | None]
    #: User metadata stored with this record as a JSON object.
    metadata: NotRequired[JsonObject]


class ChunkRecord(TypedDict):
    """Searchable text associated with a document by external ID; each chunk has its own
    external ID.
    """

    #: Application identifier of the parent document in this collection.
    document_external_id: str
    #: Application identifier of the chunk in this collection.
    chunk_external_id: str
    #: Position of the chunk within its document, an unsigned 32-bit number unique within the
    #: document.
    ordinal: int
    #: Full searchable text of the chunk.
    text: str
    #: User metadata stored with this record as a JSON object.
    metadata: NotRequired[JsonObject]
    #: Explicit source positions override metadata span hints; None explicitly clears the
    #: span.
    source_span: NotRequired[SourceSpanInput | None]


class EmbeddingRecord(TypedDict):
    """Externally computed vector to attach to a chunk. Its length, dimension and model
    must agree with the collection and effective model binding.
    """

    #: Application identifier of the chunk in this collection.
    chunk_external_id: str
    #: Embedding model identity used to check compatibility with stored vectors.
    model_name: str
    #: Number of components, matching both the vector length and collection dimension.
    dim: int
    #: Embedding components supplied by the caller in model order.
    vector: Vector


# The query vector uses the same binary32 conversion as a stored embedding. Its
# optional model name is checked against effective collection binding during
# search, after the shape has been converted.
class QueryEmbeddingInput(TypedDict):
    """Externally computed query vector for exact search, with its expected dimension and
    optional model identity.
    """

    #: Embedding components supplied by the caller in model order.
    vector: Vector
    #: Number of components, matching both the vector length and collection dimension.
    dim: int
    #: Optional model identity checked against the collection's effective embedding model.
    model_name: NotRequired[str | None]


# Supplying this mapping is recorded as an explicit override even when its values
# match the default weights. Search distinguishes the requested coefficients from
# those effective for the available inputs.
class SearchWeightsInput(TypedDict):
    """Explicit weights for score fusion. Supply all three finite, non-negative values
    with a sum of one within engine tolerance.
    """

    #: Weight applied to the lexical component during fusion.
    bm25: Number
    #: Weight applied to the exact vector component during fusion.
    vector: Number
    #: Weight applied to the metadata boost component during fusion.
    metadata: Number


class Warning(TypedDict):
    """Non-fatal diagnostic returned alongside a successful write or import. Inspect the
    field and artifact to locate its source.
    """

    #: Machine-readable identifier for this diagnostic.
    code: str
    #: Always warning for these non-fatal input diagnostics.
    severity: Literal["warning"]
    #: None for the record warnings returned by this interface.
    path: None
    #: Structured details identifying the affected input.
    details: dict[str, str]
    #: Name of the input field that produced the warning.
    field: str
    #: Kind of input record that produced the warning.
    artifact_kind: str
    #: Human-readable description of the finding.
    message: str


# The per-kind inserted and imported spellings in this mapping are aliases. When
# comparing with user input, count each insertion once and look at skipped and
# replaced separately.
class WriteSummary(TypedDict):
    """Outcome of an atomic batch. Per-kind counts distinguish inserts, skips and
    replacements; warnings can accompany success.
    """

    #: Number of documents inserted.
    documents_inserted: Number | str
    #: Alias for documents_inserted.
    documents_imported: Number | str
    #: Number of chunks inserted.
    chunks_inserted: Number | str
    #: Alias for chunks_inserted.
    chunks_imported: Number | str
    #: Number of embeddings inserted.
    embeddings_inserted: Number | str
    #: Alias for embeddings_inserted.
    embeddings_imported: Number | str
    #: Number of documents skipped under the conflict policy.
    documents_skipped: Number | str
    #: Number of chunks skipped under the conflict policy.
    chunks_skipped: Number | str
    #: Number of embeddings skipped under the conflict policy.
    embeddings_skipped: Number | str
    #: Total skipped records across documents, chunks and embeddings.
    skipped: Number | str
    #: Number of documents replaced under the conflict policy.
    documents_replaced: Number | str
    #: Number of chunks replaced under the conflict policy.
    chunks_replaced: Number | str
    #: Number of embeddings replaced under the conflict policy.
    embeddings_replaced: Number | str
    #: Total replacements across documents, chunks and embeddings.
    replaced: Number | str
    #: Missing-chunk error count; zero in a successful write summary.
    missing_chunks: Number | str
    #: Model mismatch error count; zero in a successful write summary.
    model_mismatches: Number | str
    #: Error count in the successful summary. Failed calls raise exceptions.
    errors: Number | str
    #: Non-fatal diagnostics collected while processing input records.
    warnings: list[Warning]


# The manifest result reuses the write-summary count fields across batches. An
# interrupted import can have earlier committed batches even when no complete
# ImportSummary is returned.
class ImportSummary(WriteSummary):
    """Manifest import totals across bounded commits. Inherits per-kind write counts and
    warnings from WriteSummary.
    """

    #: Complete tokenizer identity used for text indexing and query tokenization.
    tokenizer: TokenizerConfig
    #: Skipped-file details; empty for a successful supported manifest import.
    skipped_files: list[JsonValue]
    #: Manifest dataset identifier, or None when absent.
    dataset_id: str | None
    #: Number of documents read from the manifest inputs.
    documents_seen: Number
    #: Number of chunks read from the manifest inputs.
    chunks_seen: Number
    #: Number of embeddings read from the manifest inputs.
    embeddings_seen: Number


class ExportSummary(TypedDict):
    """Counts and digests for the exported JSONL artifacts and the path of the manifest
    used for reimport.
    """

    #: Alias for documents_exported.
    documents: Number
    #: Number of visible documents written to the export.
    documents_exported: Number
    #: Alias for chunks_exported.
    chunks: Number
    #: Number of visible chunks written to the export.
    chunks_exported: Number
    #: Alias for embeddings_exported.
    embeddings: Number
    #: Number of visible embeddings written to the export.
    embeddings_exported: Number
    #: Manifest filename relative to the export directory; join it to that directory for reimport.
    manifest: str
    # The manifest records a separate digest for each data file. Reordering records
    # or changing line endings changes the file's digest even when parsed JSON
    # values compare equal.
    #: SHA-256 digest of the exported documents JSONL file.
    documents_sha256: str
    #: SHA-256 digest of the exported chunks JSONL file.
    chunks_sha256: str
    #: SHA-256 digest of the exported embeddings JSONL file.
    embeddings_sha256: str


class Document(TypedDict):
    """Complete owned document projection. The internal ID can be used to delete this
    document; source_uri is stored provenance.
    """

    #: Engine-assigned ID of the document, used by operations such as delete_document.
    document_id: str
    #: Engine-assigned ID of the containing collection.
    collection_id: str
    #: Application-provided document identifier within this collection.
    external_id: str
    #: Optional display title stored with the document.
    title: str | None
    #: Optional source locator stored as text; reading a record does not fetch it.
    source_uri: str | None
    #: User metadata stored with this record as a JSON object.
    metadata: JsonObject
    #: Creation time recorded for this object by the engine.
    created_at: str
    #: Last update time recorded for this document by the engine.
    updated_at: str


class Chunk(TypedDict):
    """Complete owned chunk projection, including full text, its parent identity and any
    known source position.
    """

    #: Engine-assigned ID accepted by get_chunk and delete_chunk.
    chunk_id: str
    #: Engine-assigned ID of the containing collection.
    collection_id: str
    #: Engine-assigned ID of the document, used by operations such as delete_document.
    document_id: str
    #: Application identifier of the chunk in this collection.
    chunk_external_id: str
    #: Position of the chunk within its document, an unsigned 32-bit number unique within the
    #: document.
    ordinal: Number
    #: Full searchable text of the chunk.
    text: str
    # Hash the stored UTF-8 text itself. The token-count hint comes from the
    # tokenizer: normalization or stop-word removal can change the count without
    # changing the stored text.
    #: SHA-256 digest of the chunk text encoded as UTF-8.
    text_hash: str
    #: Token-count hint stored with this chunk.
    token_count: Number
    #: User metadata stored with this record as a JSON object.
    metadata: JsonObject
    #: Internal ID of the associated embedding, or None if absent.
    embedding_id: str | None
    #: Optional source byte and line ranges for this chunk.
    source_span: SourceSpan | None
    #: Creation time recorded for this object by the engine.
    created_at: str


# Both lookups by chunk ID and by external ID build this pair under the same
# database view. A separate lookup after a search can observe a later write.
class ChunkContent(TypedDict):
    """Full chunk and parent document read together from a single visible database view."""

    #: The complete requested chunk, including its text.
    chunk: Chunk
    #: Parent document visible in the same view as the chunk.
    document: Document


class CollectionInfo(TypedDict):
    """Visible collection identity and stored settings returned by the catalog or
    snapshot.
    """

    #: Engine-assigned ID of the containing collection.
    collection_id: str
    #: Collection name used for lookup by the application.
    name: str
    #: Number of components required per vector, from 1 through 2**32-1.
    embedding_dim: Number
    #: Metric used to compare query vectors to stored embeddings.
    distance_metric: Metric
    #: Model name reported in this catalog projection, or None when unspecified.
    embedding_model: str | None
    #: Whether the collection declares its supplied vectors normalized. Defaults to False.
    vectors_normalized: bool
    #: Stored vector index configuration for the collection.
    vector_index: JsonObject
    # Both tokenizer entries describe the same stored configuration. Consumers
    # should get the same identity from the top-level tokenizer and the config
    # mapping.
    #: Complete tokenizer identity used for text indexing and query tokenization.
    tokenizer: TokenizerConfig
    #: Stored collection configuration, including the same tokenizer identity as tokenizer.
    config: dict[str, TokenizerConfig]
    #: Creation time recorded for this object by the engine.
    created_at: str


class DatabaseStats(TypedDict):
    """Counts and storage estimates across the owner's current view, with its effective
    resource configuration. Byte counts and commit IDs above 2**53-1 use decimal strings.
    """

    #: Resource preset active on this database owner.
    active_profile: Profile
    #: Whether background compaction is enabled for this owner.
    background_compaction_enabled: bool
    #: Effective cache budget in MiB for this owner; it does not cap process RSS.
    cache_budget_mb: Number | str
    #: Number of visible chunks in the reported scope.
    chunk_count: Number | str
    #: Number of visible collections covered by this report.
    collection_count: Number | str
    # File length is a filesystem measurement. Live and obsolete bytes are
    # estimates derived from the catalog; adding them does not reconstruct the
    # physical length.
    #: Alias for live_bytes, the estimated size of live data in bytes.
    data_size_bytes: Number | str
    #: Estimated obsolete bytes based on the number of recorded tombstones.
    dead_bytes_estimate: Number | str
    #: Alias for tombstone_count; counts deletion markers.
    deleted_count: Number | str
    #: Number of visible documents in the reported scope.
    document_count: Number | str
    #: Number of visible embeddings in the reported scope.
    embedding_count: Number | str
    #: Length of the database file in bytes; also returned as file_size_bytes.
    file_size: Number | str
    #: Length of the database file in bytes.
    file_size_bytes: Number | str
    #: On-disk format version of the database being reported.
    format_version: Number | str
    #: Encoded size in bytes of the usable database-wide index checkpoint, or zero
    #: when unavailable.
    index_size_bytes: Number | str
    #: Identifier of the last committed batch visible to this owner.
    last_commit_id: Number | str
    #: Estimated live chunk and embedding payload bytes across collections. Where
    #: available the engine uses the stored payload lengths.
    live_bytes: Number | str
    #: Estimate combining tombstone-based obsolete bytes with an allowance for index
    #: storage. Actual space reclaimed by compaction can differ.
    pending_compaction_bytes: Number | str
    #: Number of deletion markers in the reported scope.
    tombstone_count: Number | str


class CollectionStats(TypedDict):
    """Counts, derived index statistics and storage estimates reported for one visible
    collection. Byte counts and commit IDs above 2**53-1 use decimal strings.
    """

    #: Name of the collection to which this report applies.
    collection: str
    #: Complete tokenizer identity used for text indexing and query tokenization.
    tokenizer: TokenizerConfig
    #: Resource preset active on this database owner.
    active_profile: Profile
    #: Resource preset active on this owner; also exposed as active_profile.
    resource_profile: Profile
    #: Whether background compaction is enabled for this owner.
    background_compaction_enabled: bool
    #: Effective owner cache budget in MiB, shared across its collections.
    cache_budget_mb: Number | str
    #: Number of visible documents in the reported scope.
    document_count: Number | str
    #: Number of visible chunks in the reported scope.
    chunk_count: Number | str
    #: Number of visible embeddings in the reported scope.
    embedding_count: Number | str
    #: Number of deletion markers in the reported scope.
    tombstone_count: Number | str
    #: Alias for tombstone_count; counts deletion markers.
    deleted_count: Number | str
    # A usable checkpoint can contain terms that currently have no postings.
    # Collection statistics count terms with postings for this collection; the
    # database term count below includes retained empty keys.
    #: Distinct terms with postings in this collection's usable checkpoint. An
    #: unavailable checkpoint reports zero.
    text_index_terms: Number | str
    #: Number of lexical index segments reported for the collection.
    text_index_segments: Number | str
    #: Live embeddings in this collection, counted from the catalog even when the
    #: persisted index needs rebuilding.
    vector_count: Number | str
    #: Size in bytes of the entire usable database checkpoint, repeated for each
    #: collection. Zero when unavailable.
    index_size_bytes: Number | str
    #: Alias for live_bytes, the estimated size of live data in bytes.
    data_size_bytes: Number | str
    #: Estimated live chunk and embedding payload bytes for this collection.
    live_bytes: Number | str
    #: Estimated bytes attributable to obsolete data.
    dead_bytes_estimate: Number | str
    #: Identifier of the last committed batch visible to this owner.
    last_commit_id: Number | str
    #: Collection's estimated obsolete bytes plus an allowance for the database-wide
    #: checkpoint. Adding this field across collections can count the same allowance
    #: repeatedly.
    pending_compaction_bytes: Number | str
    #: On-disk format version of the database being reported.
    format_version: Number | str


class IndexStats(TypedDict):
    """Database-wide derived index coverage and usage. Reading these counters does not
    rebuild an index. Byte counts and commit IDs above 2**53-1 use decimal strings.
    """

    #: Encoded size in bytes of the usable database-wide index checkpoint, or zero
    #: when unavailable.
    index_size_bytes: Number | str
    #: Commit sequence number covered by the persisted index, or None when unavailable.
    last_index_commit_id: Number | str | None
    #: Whether the engine reports that a persistent index still needs rebuilding.
    rebuild_needed: bool
    # The engine reports one text and one vector segment per visible collection
    # when checkpoint coverage is available. A rebuild-needed report can still show
    # live vectors from the catalog.
    #: Number of lexical index segments reported for the database.
    text_segments: Number | str
    #: Distinct term keys across all collection dictionaries in the usable
    #: checkpoint, including keys with no postings. Zero when the checkpoint is
    #: unavailable.
    text_terms: Number | str
    #: Live embeddings counted from the catalog, including when the derived index
    #: needs rebuilding.
    vector_count: Number | str
    #: Number of vector index segments reported for the database.
    vector_segments: Number | str


# These counts describe recall and retention, not the origin of every score on the
# returned hit. Score completion can fill in the other route for a chunk already in
# the candidate set.
class CandidateCounts(TypedDict):
    """Candidate counts for the lexical route, vector route and combined pool before the
    final top-k selection.
    """

    #: Lexical candidates surviving the metadata filter on the bounded BM25 recall
    #: window.
    bm25: Number
    #: Candidates selected by bounded vector recall. A vector score completed later
    #: for a lexical hit adds no candidate here.
    vector: Number
    #: Number of distinct candidates retained after score fusion and before the
    #: final top-k cutoff.
    fusion: Number


class SearchExecutionStats(TypedDict):
    """Work observed during this query, including vector scans, score computations and the
    combined candidate pool.
    """

    #: Vector entries inspected during recall and score completion. A filtered entry
    #: can count without incurring a distance calculation.
    vector_scan_count: Number
    #: Vector components scored in recall and score completion. Scoring one
    #: 128-dimensional embedding adds 128.
    distance_ops: Number
    #: Number of distinct chunks retained by the fusion window, including each chunk
    #: recalled by both routes only once.
    fusion_candidate_count: Number


class NormalizedScores(TypedDict):
    """Per-hit components used in fusion. Their magnitudes are scoring inputs, not
    probabilities.
    """

    # Scale lexical scores by the largest positive raw score in the union of
    # recalled chunks, before the fusion cutoff. A vector component follows its
    # metric; an unnormalized dot product can exceed one.
    #: Normalized lexical component used by score fusion.
    bm25: Number
    #: Vector component used by score fusion, or None when unavailable.
    vector: Number | None
    #: Score contributed by matching metadata boost rules before weighting.
    metadata_boost: Number


class SearchHit(TypedDict):
    """Ranked matching chunk with identifiers, snippet, scores and an explanation. Fetch
    complete text using the chunk ID.
    """

    # The native ranker breaks equal final scores by ascending internal chunk ID.
    # Equal-score hits can therefore have distinct, reproducible ranks.
    #: One-based position in the sorted results.
    rank: Number
    #: Engine-assigned ID accepted by get_chunk and delete_chunk.
    chunk_id: str
    #: Application identifier of the chunk in this collection.
    chunk_external_id: str
    #: Engine-assigned ID of the document, used by operations such as delete_document.
    document_id: str
    #: Application identifier of the parent document in this collection.
    document_external_id: str
    #: Display excerpt from the matching chunk.
    snippet: str
    #: Owned metadata from the parent document.
    document_metadata: JsonObject
    #: Owned metadata from the matching chunk.
    chunk_metadata: JsonObject
    #: Fused score used for ordering results.
    final_score: Number
    # A raw score can be available even if the chunk entered through the other
    # recall route. Use the hit's explanation to tell which recall windows actually
    # selected it.
    #: Raw lexical score when available; None if this route supplied no score.
    bm25_score: Number | None
    #: Vector comparison score when available; None if no score was computed.
    vector_score: Number | None
    #: Score contributed by matching metadata boost rules before weighting.
    metadata_boost: Number
    #: Components used with the effective weights to compute the fused score.
    normalized_scores: NormalizedScores
    #: Structured diagnostic fields for this hit's candidate participation, scores and
    #: ranking.
    explain: JsonObject


class SearchResults(TypedDict):
    """One completed search, with the effective query context, work counters and ordered
    hits.
    """

    #: Version of the structured response schema.
    schema_version: str
    #: Name of the collection to which this report applies.
    collection: str
    #: Query text, or None for a vector-only search.
    query: str | None
    #: Requested maximum number of ranked hits.
    top_k: Number
    #: Resource preset used to execute this query.
    profile: Profile
    #: Effective query settings and diagnostics, including the routing and scoring context.
    query_context: JsonObject
    #: Sizes of the lexical, vector and fusion candidate pools.
    candidate_counts: CandidateCounts
    #: Work counters observed while running this query.
    execution_stats: SearchExecutionStats
    #: Returned hits in rank order, limited by top_k.
    results: list[SearchHit]


# The state describes the operation as a whole; the optional counters describe
# its confirmed prefix. For example, a later batch can have an unknown outcome
# after an earlier batch was confirmed. A missing count supplies no evidence
# that zero records were written.
class OperationOutcome(TypedDict):
    """Write progress attached to an exception when known. Counts refer to committed core
    records and can include collection or tombstone records.
    """

    #: Known commit disposition; unknown means the caller must verify the data before
    #: retrying.
    state: Literal["not_started", "not_committed", "partially_committed", "committed", "unknown"]
    #: Number of batches known to have committed, when established.
    confirmed_committed_batches: NotRequired[int]
    #: Number of core records known to have committed, when established.
    confirmed_committed_records: NotRequired[int]
    #: Identifier of the most recent confirmed commit, when known.
    last_confirmed_commit_id: NotRequired[int]
    #: Sequence number of the most recent confirmed commit, when known.
    last_confirmed_commit_sequence_no: NotRequired[int]


class CheckIssue(TypedDict):
    """One finding from a read-only scan, including location, impact and any suggested
    operator action.
    """

    #: Diagnostic level indicating the importance of the finding.
    severity: str
    #: Machine-readable identifier for this diagnostic.
    code: str
    #: Human-readable description of the finding.
    message: str
    #: Logical record stream in which the checker found this issue.
    stream: str
    #: How this issue affects the visible data or derived indexes.
    visibility_impact: str
    #: File byte offset of the finding, or None when unknown.
    offset: int | str | None
    #: Related batch identifier, or None when the checker cannot identify a batch.
    batch_id: int | str | None
    #: Identifier of the affected extent, if known.
    extent_id: str | None
    #: Kind of record that produced the finding, if known.
    record_type: str | None
    #: Suggested follow-up CLI command when one is available; the check does not execute it.
    suggested_command: str | None
    #: Whether the checker reports an available repair for this finding.
    repair_available: bool
    #: Alias of repair_available for this finding.
    repairable: bool


# The read-only checker sends its structured report through the shared JSON
# projection. Preserve decimal strings for counts or offsets that exceed the JSON
# integer range used by that report.
class CheckIndexSuffix(TypedDict):
    """Metadata for the derived-index checkpoint found by a scan. Large offset and count
    fields use exact decimal strings.
    """

    #: Status of the checkpoint metadata found in the file.
    status: str
    #: Method used to construct the stored checkpoint.
    build_mode: str
    #: Byte offset at which the checkpoint begins in the checked file.
    offset: int | str
    #: Generation identifier of the checkpoint.
    checkpoint_generation: int | str
    #: Byte boundary of the committed critical data covered by this checkpoint.
    covered_critical_end_offset: int | str
    #: Identifier of the last committed batch covered by the checkpoint.
    covered_last_batch_id: int | str
    #: Commit sequence number covered by the checkpoint.
    covered_last_commit_sequence_no: int | str
    #: Number of visible collections covered by this report.
    collection_count: int | str
    #: Number of distinct documents with at least one chunk in the checkpoint text segments.
    live_document_count: int | str
    #: Number of visible chunks recorded by the checkpoint.
    live_chunk_count: int | str
    #: Number of visible embeddings recorded by the checkpoint.
    live_embedding_count: int | str
    #: Number of lexical segments in the reported index.
    text_segment_count: int | str
    #: Number of vector segments in the reported index.
    vector_segment_count: int | str
    #: Number of records in the reported derived index.
    index_record_count: int | str
    #: Number of extents in the reported derived index.
    index_extent_count: int | str
    #: Length of the index payload in bytes.
    payload_len: int | str
    #: Total encoded checkpoint size in bytes.
    total_size_bytes: int | str
    #: 32-bit hash used to compare checkpoint coverage with the visible catalog.
    coverage_hash: int


class CheckCounts(TypedDict):
    """Visible data counts reconstructed by the scan. Counts above 2**53-1 use exact
    decimal strings.
    """

    #: All visible documents in committed data, including documents with no chunks.
    live_documents: int | str
    #: Visible chunks counted by the scan.
    live_chunks: int | str
    #: Deletion markers counted by the scan.
    tombstones: int | str


# The function can return an error-status report as data. A Python exception
# represents failure to obtain or deliver a report and is a separate outcome.
class CheckReport(TypedDict):
    """Diagnostics for one consistent file view. Inspect issues and the index-check
    disposition even when the call returned normally.
    """

    #: Version of the structured response schema.
    schema_version: str
    #: Path of the database file inspected by the checker.
    path: str
    #: Overall severity: ok, warning or error, determined by this scan's findings.
    status: str
    #: Engine classification of the on-disk derived-index suffix.
    index_suffix_status: str
    #: Whether index verification succeeded, was skipped or requires rebuilding.
    index_check: str
    #: On-disk format version of the database being reported.
    format_version: int
    #: ID of the last valid committed batch covered by this view.
    last_valid_commit: int | str
    #: Byte boundary through which committed critical data was validated.
    critical_end_offset: int | str
    #: Parsed checkpoint metadata if available, otherwise None.
    index_suffix: CheckIndexSuffix | None
    #: Findings in the scanned view; each describes its impact and any available action.
    issues: list[CheckIssue]
    #: True if at least one finding has an available repair.
    repair_available: bool
    #: Reason for skipped verification or a required index rebuild, when provided.
    index_check_reason: str | None
    #: Counts of visible documents, chunks and deletion markers reconstructed by the scan.
    result: CheckCounts


# Snapshot uses the native receipt projection. Its integer fields remain Python
# integers even where the checker represents the corresponding file metadata as
# decimal strings.
class SnapshotReport(TypedDict):
    """Receipt for a published standalone copy. Use its length, SHA-256 and committed view
    to verify the target.
    """

    #: Path of the source database from which the standalone snapshot was taken.
    source_path: str
    #: Path of the published snapshot file.
    output_path: str
    #: Snapshot package format version, distinct from the Python distribution version.
    package_version: int
    #: On-disk format version of the database being reported.
    format_version: int
    #: Numeric format feature IDs required to read the snapshot.
    required_features: list[int]
    #: ID of the last valid committed batch covered by this view.
    last_valid_commit: int
    #: Byte boundary through which committed critical data was validated.
    critical_end_offset: int
    #: Exact byte length of the snapshot file.
    size_bytes: int
    #: Hexadecimal SHA-256 digest of the published snapshot bytes.
    sha256: str
    #: Engine classification of the on-disk derived-index suffix.
    index_suffix_status: str
    #: Collection identities and configurations visible in the snapshot.
    collections: list[CollectionInfo]


class RebuildIndexReport(TypedDict):
    """Actual scope of derived-index work and the checkpoint left on disk. A requested
    collection can still produce a full checkpoint.
    """

    #: Version of the structured response schema.
    schema_version: str
    #: Requested collection name, or None for a database-wide request.
    collection: str | None
    #: Actual extent of index work; inspect this even when a collection was requested.
    rebuild_scope: str
    #: Engine classification of the on-disk derived-index suffix.
    index_suffix_status: str
    #: Whether the engine reports that a persistent index still needs rebuilding.
    rebuild_needed: bool
    #: Commit sequence number covered by the persisted index, or None when unavailable.
    last_index_commit_id: int | None
    #: Number of records in the reported derived index.
    index_record_count: int
    #: Number of extents in the reported derived index.
    index_extent_count: int
    #: Number of lexical segments in the reported index.
    text_segment_count: int
    #: Number of vector segments in the reported index.
    vector_segment_count: int
    #: Elapsed duration of this operation in milliseconds.
    duration_ms: int


class CompactReport(TypedDict):
    """File sizes and index outcome after a compacted generation replaces the main file. A
    later index failure is reported separately.
    """

    #: Main file length in bytes before compaction.
    before_file_size_bytes: int
    #: Replacement main file length in bytes after index handling.
    after_file_size_bytes: int
    #: Number of obsolete business records removed by compaction.
    records_removed: int
    #: Number of deletion markers removed from the compacted data.
    tombstones_removed: int
    #: Number of persistent index segments counted as merged by compaction.
    segments_merged: int
    #: Elapsed duration of this operation in milliseconds.
    duration_ms: int
    #: Outcome of the post-replacement index step: full_checkpoint_written, disabled or
    #: failed.
    index_rebuild: str
    #: Diagnostic for a failed index step, or None when no reason is reported.
    index_rebuild_reason: str | None


# Keep this export list in step with the public imports in __init__.py and
# __init__.pyi when adding or renaming a schema.
__all__ = [
    "SnapshotReport", "RebuildIndexReport", "CompactReport",
    "CheckIssue", "CheckIndexSuffix", "CheckCounts", "CheckReport",
    "Profile", "Conflict", "Metric", "Durability", "IoMode", "Number", "JsonValue",
    "JsonObject", "BatchSeq", "Vector", "DatabaseConfigInput", "OpenOptionsInput",
    "TokenizerRules", "TokenizerConfig", "CollectionConfigInput", "SourceSpanInput",
    "SourceSpan", "DocumentRecord", "ChunkRecord", "EmbeddingRecord",
    "QueryEmbeddingInput", "SearchWeightsInput", "Warning", "WriteSummary",
    "ImportSummary", "ExportSummary", "Document", "Chunk", "ChunkContent",
    "CollectionInfo", "DatabaseStats", "CollectionStats", "IndexStats", "CandidateCounts",
    "SearchExecutionStats", "NormalizedScores", "SearchHit", "SearchResults", "OperationOutcome",
]
