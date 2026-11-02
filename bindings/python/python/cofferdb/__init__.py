"""Experimental in-process CofferDB binding for version 0.1.1."""

from ._runtime import check_runtime as _check_runtime

_check_runtime()
del _check_runtime

from ._native import (
    __version__ as _native_version, engine_version, Database, Collection, CofferError, SdkError, ClosedError, BusyError,
    ForkedHandleError, PoisonedHandleError, EnginePanicError,
    ResultMaterializationError,
)

__version__ = "0.1.1"
if _native_version != __version__:
    raise ImportError(
        "CofferDB Python and native extension versions do not match; "
        "reinstall a matching wheel."
    )
del _native_version

from . import types as _types
from .types import (
    Profile, Conflict, Metric, Durability, IoMode, Number, JsonValue, JsonObject,
    BatchSeq, Vector, DatabaseConfigInput, OpenOptionsInput, TokenizerRules,
    TokenizerConfig, CollectionConfigInput, SourceSpanInput, SourceSpan,
    DocumentRecord, ChunkRecord, EmbeddingRecord, QueryEmbeddingInput,
    SearchWeightsInput, Warning, WriteSummary, ImportSummary, ExportSummary,
    Document, Chunk, ChunkContent, CollectionInfo, DatabaseStats, CollectionStats,
    IndexStats, CandidateCounts, SearchExecutionStats, NormalizedScores,
    SearchHit, SearchResults, OperationOutcome,
)

__all__ = [
    "__version__", "engine_version", "Database", "Collection", "CofferError", "SdkError", "ClosedError",
    "BusyError", "ForkedHandleError", "PoisonedHandleError", "EnginePanicError",
    "ResultMaterializationError",
]
__all__ += _types.__all__
del _types
