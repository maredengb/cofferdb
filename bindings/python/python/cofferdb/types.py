"""Public input and owned-result shapes for the synchronous native API.

These annotations do not coerce values. Runtime validation requires exact
built-in containers and scalar types, and rejects bool for numeric arguments.
"""

from typing import Literal, NotRequired, TypeAlias, TypedDict, TypeVar

Profile: TypeAlias = Literal["expanded", "balanced", "constrained"]
Conflict: TypeAlias = Literal["error", "skip", "replace"]
Metric: TypeAlias = Literal["cosine", "dot_product", "euclidean"]
Durability: TypeAlias = Literal["commit", "fast"]
IoMode: TypeAlias = Literal["auto", "read", "mmap_disabled"]
Number: TypeAlias = int | float
JsonValue: TypeAlias = None | bool | Number | str | list["JsonValue"] | tuple["JsonValue", ...] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
_T = TypeVar("_T")
BatchSeq: TypeAlias = list[_T] | tuple[_T, ...]
Vector: TypeAlias = list[Number] | tuple[Number, ...]


class DatabaseConfigInput(TypedDict, total=False):
    """Omitted values use core defaults; only cache_budget_mb accepts None."""

    resource_profile: Profile
    cache_budget_mb: int | None
    background_compaction: bool
    durability: Durability
    io_mode: IoMode


class OpenOptionsInput(TypedDict, total=False):
    """Nullable overrides inherit runtime defaults; io_mode cannot be None."""

    resource_profile: Profile | None
    cache_budget_mb: int | None
    background_compaction: bool | None
    durability: Durability | None
    io_mode: IoMode


class TokenizerRules(TypedDict):
    case_fold: Literal["lower"]
    split: Literal["unicode_punctuation"]
    stop_words: Literal["default_en"]


class TokenizerConfig(TypedDict):
    """The complete immutable tokenizer identity; partial objects are invalid."""

    name: Literal["default_en"]
    version: Literal["0.1"]
    profile: Literal["latin_code_default"]
    language_hint: Literal["en"]
    config: TokenizerRules


class CollectionConfigInput(TypedDict):
    name: str
    embedding_dim: int
    distance_metric: Metric
    embedding_model: NotRequired[str | None]
    vectors_normalized: NotRequired[bool]
    vector_index: NotRequired[Literal["exact"]]
    tokenizer: NotRequired[TokenizerConfig]


class SourceSpanInput(TypedDict, total=False):
    """Unsigned u64 byte offsets accept exact int or unsigned decimal str."""

    start: int | str | None
    end: int | str | None
    line_start: int | None
    line_end: int | None


class SourceSpan(TypedDict):
    """Byte offsets above 2**53-1 are returned as exact decimal strings."""

    start: int | str | None
    end: int | str | None
    line_start: int | None
    line_end: int | None


class DocumentRecord(TypedDict):
    external_id: str
    title: NotRequired[str | None]
    source_uri: NotRequired[str | None]
    metadata: NotRequired[JsonObject]


class ChunkRecord(TypedDict):
    document_external_id: str
    chunk_external_id: str
    ordinal: int
    text: str
    metadata: NotRequired[JsonObject]
    source_span: NotRequired[SourceSpanInput | None]


class EmbeddingRecord(TypedDict):
    chunk_external_id: str
    model_name: str
    dim: int
    vector: Vector


class QueryEmbeddingInput(TypedDict):
    vector: Vector
    dim: int
    model_name: NotRequired[str | None]


class SearchWeightsInput(TypedDict):
    bm25: Number
    vector: Number
    metadata: Number


class Warning(TypedDict):
    code: str
    severity: Literal["warning"]
    path: None
    details: dict[str, str]
    field: str
    artifact_kind: str
    message: str


class WriteSummary(TypedDict):
    documents_inserted: Number
    documents_imported: Number
    chunks_inserted: Number
    chunks_imported: Number
    embeddings_inserted: Number
    embeddings_imported: Number
    documents_skipped: Number
    chunks_skipped: Number
    embeddings_skipped: Number
    skipped: Number
    documents_replaced: Number
    chunks_replaced: Number
    embeddings_replaced: Number
    replaced: Number
    missing_chunks: Number
    model_mismatches: Number
    errors: Number
    warnings: list[Warning]


class ImportSummary(WriteSummary):
    tokenizer: TokenizerConfig
    skipped_files: list[JsonValue]
    dataset_id: str | None
    documents_seen: Number
    chunks_seen: Number
    embeddings_seen: Number


class ExportSummary(TypedDict):
    documents: Number
    documents_exported: Number
    chunks: Number
    chunks_exported: Number
    embeddings: Number
    embeddings_exported: Number
    manifest: str
    documents_sha256: str
    chunks_sha256: str
    embeddings_sha256: str


class Document(TypedDict):
    document_id: str
    collection_id: str
    external_id: str
    title: str | None
    source_uri: str | None
    metadata: JsonObject
    created_at: str
    updated_at: str


class Chunk(TypedDict):
    chunk_id: str
    collection_id: str
    document_id: str
    chunk_external_id: str
    ordinal: Number
    text: str
    text_hash: str
    token_count: Number
    metadata: JsonObject
    embedding_id: str | None
    source_span: SourceSpan | None
    created_at: str


class ChunkContent(TypedDict):
    chunk: Chunk
    document: Document


class CollectionInfo(TypedDict):
    collection_id: str
    name: str
    embedding_dim: Number
    distance_metric: Metric
    embedding_model: str | None
    vectors_normalized: bool
    vector_index: JsonObject
    tokenizer: TokenizerConfig
    config: dict[str, TokenizerConfig]
    created_at: str


class DatabaseStats(TypedDict):
    active_profile: Profile
    background_compaction_enabled: bool
    cache_budget_mb: Number
    chunk_count: Number
    collection_count: Number
    data_size_bytes: Number
    dead_bytes_estimate: Number
    deleted_count: Number
    document_count: Number
    embedding_count: Number
    file_size: Number
    file_size_bytes: Number
    format_version: Number
    index_size_bytes: Number
    last_commit_id: Number
    live_bytes: Number
    pending_compaction_bytes: Number
    tombstone_count: Number


class CollectionStats(TypedDict):
    collection: str
    tokenizer: TokenizerConfig
    active_profile: Profile
    resource_profile: Profile
    background_compaction_enabled: bool
    cache_budget_mb: Number
    document_count: Number
    chunk_count: Number
    embedding_count: Number
    tombstone_count: Number
    deleted_count: Number
    text_index_terms: Number
    text_index_segments: Number
    vector_count: Number
    index_size_bytes: Number
    data_size_bytes: Number
    live_bytes: Number
    dead_bytes_estimate: Number
    last_commit_id: Number
    pending_compaction_bytes: Number
    format_version: Number


class IndexStats(TypedDict):
    index_size_bytes: Number
    last_index_commit_id: Number | None
    rebuild_needed: bool
    text_segments: Number
    text_terms: Number
    vector_count: Number
    vector_segments: Number


class CandidateCounts(TypedDict):
    bm25: Number
    vector: Number
    fusion: Number


class SearchExecutionStats(TypedDict):
    vector_scan_count: Number
    distance_ops: Number
    fusion_candidate_count: Number


class NormalizedScores(TypedDict):
    bm25: Number
    vector: Number | None
    metadata_boost: Number


class SearchHit(TypedDict):
    rank: Number
    chunk_id: str
    chunk_external_id: str
    document_id: str
    document_external_id: str
    snippet: str
    document_metadata: JsonObject
    chunk_metadata: JsonObject
    final_score: Number
    bm25_score: Number | None
    vector_score: Number | None
    metadata_boost: Number
    normalized_scores: NormalizedScores
    explain: JsonObject


class SearchResults(TypedDict):
    schema_version: str
    collection: str
    query: str | None
    top_k: Number
    profile: Profile
    query_context: JsonObject
    candidate_counts: CandidateCounts
    execution_stats: SearchExecutionStats
    results: list[SearchHit]


class OperationOutcome(TypedDict):
    """Only confirmed facts are present; record counts include core records."""

    state: Literal["not_started", "not_committed", "partially_committed", "committed", "unknown"]
    confirmed_committed_batches: NotRequired[int]
    confirmed_committed_records: NotRequired[int]
    last_confirmed_commit_id: NotRequired[int]
    last_confirmed_commit_sequence_no: NotRequired[int]


__all__ = [
    "Profile", "Conflict", "Metric", "Durability", "IoMode", "Number", "JsonValue",
    "JsonObject", "BatchSeq", "Vector", "DatabaseConfigInput", "OpenOptionsInput",
    "TokenizerRules", "TokenizerConfig", "CollectionConfigInput", "SourceSpanInput",
    "SourceSpan", "DocumentRecord", "ChunkRecord", "EmbeddingRecord",
    "QueryEmbeddingInput", "SearchWeightsInput", "Warning", "WriteSummary",
    "ImportSummary", "ExportSummary", "Document", "Chunk", "ChunkContent",
    "CollectionInfo", "DatabaseStats", "CollectionStats", "IndexStats", "CandidateCounts",
    "SearchExecutionStats", "NormalizedScores", "SearchHit", "SearchResults", "OperationOutcome",
]
