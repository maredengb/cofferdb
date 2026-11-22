"""In-process CofferDB binding for version 0.2.0.

Create a private temporary database, write and retrieve one chunk, then verify
a snapshot after closing the writer:

>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> from cofferdb import Database, check
>>> with TemporaryDirectory() as directory:
...     main = str(Path(directory) / "notes.coffer")
...     backup = str(Path(directory) / "backup.coffer")
...     with Database.create(main) as db:
...         notes = db.create_collection({"name": "notes", "embedding_dim": 2,
...                                       "distance_metric": "cosine"})
...         written = notes.write_batch(
...             documents=[{"external_id": "leaf"}],
...             chunks=[{"document_external_id": "leaf", "chunk_external_id": "leaf#1",
...                      "ordinal": 0, "text": "Jade leaves grow near the river."}])
...         hit = notes.search("jade", top_k=1)["results"][0]
...         assert written["chunks_inserted"] == 1 and hit["chunk_external_id"] == "leaf#1"
...         saved = db.snapshot(backup)
...         assert not db.rebuild_index()["rebuild_needed"]
...         assert db.compact()["index_rebuild"] == "full_checkpoint_written"
...     verified = check(backup)
...     assert verified["status"] == "ok" and verified["last_valid_commit"] == saved["last_valid_commit"]
...     with Database.open(backup, read_only=True) as restored:
...         assert restored.collection("notes").get_chunk(hit["chunk_id"])["chunk"]["text"].startswith("Jade")

The same-version programming and recovery guides are docs/python.md and
docs/backup.md in the source distribution and offline reference.
"""

from ._runtime import check_runtime as _check_runtime

# Check support before the import below loads the native extension.
_check_runtime()
del _check_runtime

# Re-export the extension's Database, Collection and exception classes themselves.
# A Python wrapper or a second exception hierarchy here would change the type
# identities callers receive from native code.
from ._native import (
    check, __version__ as _native_version, engine_version, Database, Collection, CofferError, SdkError, ClosedError, BusyError,
    ForkedHandleError, PoisonedHandleError, EnginePanicError,
    ResultMaterializationError,
)

#: Version of the installed Python distribution.
__version__ = "0.2.0"
# A stale extension can still import after a partial reinstall. Check the
# extension version before exposing its API with this package's types.
# Compare the extension package version, which identifies the Python binding it was
# built with. The separately exported engine_version identifies the linked Rust
# engine and is not the installation compatibility check.
if _native_version != __version__:
    raise ImportError(
        "CofferDB Python and native extension versions do not match; "
        "reinstall a matching wheel."
    )
del _native_version

from . import types as _types
# Bind the public aliases and result schemas into this module so callers and the
# package stub can use the same top-level names. They describe ordinary Python
# arguments and the values returned by the extension.
from .types import (
    SnapshotReport, RebuildIndexReport, CompactReport,
    CheckIssue, CheckIndexSuffix, CheckCounts, CheckReport,
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
    "check", "__version__", "engine_version", "Database", "Collection", "CofferError", "SdkError", "ClosedError",
    "BusyError", "ForkedHandleError", "PoisonedHandleError", "EnginePanicError",
    "ResultMaterializationError",
]
# Wildcard imports use this combined list. Each public type name must be both bound
# above and listed in types.__all__, or direct imports and wildcard imports will
# expose different interfaces.
__all__ += _types.__all__
del _types
