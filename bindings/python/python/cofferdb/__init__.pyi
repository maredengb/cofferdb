from types import TracebackType
from typing import Self
# An explicit same-name alias re-exports the imported name in a stub. Keep these
# aliases aligned with the package initializer so a supported top-level import
# resolves in both Python and a type checker.
from .types import (
    SnapshotReport as SnapshotReport, RebuildIndexReport as RebuildIndexReport, CompactReport as CompactReport,
    CheckIssue as CheckIssue, CheckIndexSuffix as CheckIndexSuffix, CheckCounts as CheckCounts, CheckReport as CheckReport,
    Profile as Profile, Conflict as Conflict, Metric as Metric, Durability as Durability,
    IoMode as IoMode, Number as Number, JsonValue as JsonValue, JsonObject as JsonObject,
    BatchSeq as BatchSeq, Vector as Vector, DatabaseConfigInput as DatabaseConfigInput,
    OpenOptionsInput as OpenOptionsInput, TokenizerRules as TokenizerRules,
    TokenizerConfig as TokenizerConfig, CollectionConfigInput as CollectionConfigInput,
    SourceSpanInput as SourceSpanInput, SourceSpan as SourceSpan,
    DocumentRecord as DocumentRecord, ChunkRecord as ChunkRecord,
    EmbeddingRecord as EmbeddingRecord, QueryEmbeddingInput as QueryEmbeddingInput,
    SearchWeightsInput as SearchWeightsInput, Warning as Warning, WriteSummary as WriteSummary,
    ImportSummary as ImportSummary, ExportSummary as ExportSummary, Document as Document,
    Chunk as Chunk, ChunkContent as ChunkContent, CollectionInfo as CollectionInfo,
    DatabaseStats as DatabaseStats, CollectionStats as CollectionStats, IndexStats as IndexStats,
    CandidateCounts as CandidateCounts, SearchExecutionStats as SearchExecutionStats,
    NormalizedScores as NormalizedScores, SearchHit as SearchHit, SearchResults as SearchResults,
    OperationOutcome as OperationOutcome,
)

#: Version of the installed Python distribution.
__version__: str
#: Version of the native engine linked into the installed extension.
engine_version: str
#: Public names exported by the package.
__all__: list[str]

def check(path: str) -> CheckReport:
    """Scan an existing database path read-only and return a CheckReport.

    The scan uses a consistent committed view, also beside an idle current-version writer.
    It delays mutations while scanning and leaves file contents unchanged. A completed
    scan can report warning or error: inspect status, issues, index_check and
    index_check_reason. If a concurrent change prevents the scan, CofferError with code
    recovery_error and detail reason check_view_unavailable can be retried after the write
    finishes. Pass a built-in str path.
    """
    ...


# The native projector attaches these fields to engine exceptions. Write and
# maintenance observations are conditional attributes on the final exception, which
# may also be a Python signal or a result-construction failure.
class CofferError(Exception):
    """Error reported by the native engine. Inspect code (machine identifier), message
    (explanation), detail (structured context), recoverable (engine guidance) and
    suggested_action (optional advice). For interrupted writes, inspect any attached
    outcome before deciding whether to retry.
    """
    #: Machine-readable engine error identifier.
    code: str
    #: Human-readable engine explanation.
    message: str
    #: Structured context, including maintenance stage and effect when available.
    detail: JsonObject
    #: Whether the engine classifies this error as recoverable.
    recoverable: bool
    #: Suggested operator action, or None if no advice is available.
    suggested_action: str | None


class SdkError(Exception):
    """Base class for errors at the Python ownership and result-construction boundary. Catch
    a concrete subclass for the relevant recovery action. Engine errors are exposed
    separately as CofferError.
    """


# Declare sdk_reason on the owner-state and panic subclasses that actually receive
# it. Constructing a ResultMaterializationError does not attach that field, so the
# common base cannot promise it.
class ClosedError(SdkError):
    """The owner was closed. Obtain a new owner and collection handle before continuing.
    sdk_reason is closed.
    """
    #: Stable SDK reason for this exception: closed.
    sdk_reason: str


class BusyError(SdkError):
    """Another call is active on this owner. Wait for it to finish before trying again.
    sdk_reason is busy.
    """
    #: Stable SDK reason for this exception: busy.
    sdk_reason: str


class ForkedHandleError(SdkError):
    """This handle belongs to the parent process. Open a fresh owner in the child, subject to
    the file's writer lock. sdk_reason is forked_handle.
    """
    #: Stable SDK reason for this exception: forked_handle.
    sdk_reason: str


class PoisonedHandleError(SdkError):
    """A previous native panic poisoned this owner. Close it before opening a fresh owner.
    sdk_reason is poisoned.
    """
    #: Stable SDK reason for this exception: poisoned.
    sdk_reason: str


class EnginePanicError(SdkError):
    """The native call panicked. Its payload is hidden; close the affected owner if present
    and verify any writes. sdk_reason is engine_panic.
    """
    #: Stable SDK reason for this exception: engine_panic.
    sdk_reason: str


class ResultMaterializationError(SdkError):
    """Native work could not be delivered as a Python result. A write or maintenance
    operation may already have taken effect. Inspect any attached outcome and verify
    actual state before retrying.
    """


class Database:
    """Own a synchronous database connection. Use create or open, then close the owner
    explicitly or use it as a context manager.

    Only one writer across processes can own a physical database at a time. One operation
    at a time may use this owner, including via its collections; concurrent or reentrant
    use raises BusyError. A read-only owner keeps its fixed view. Close invalidates
    dependent collections but leaves already returned Python values usable. An inherited
    owner raises ForkedHandleError in a child process.
    """

    @staticmethod
    def create(path: str, *, config: DatabaseConfigInput | None = None) -> Database:
        """Create a new database at a built-in str path and return its writer owner.

        An existing path is never overwritten. config is an optional DatabaseConfigInput
        mapping; omission selects the balanced preset, its cache budget, commit
        durability, automatic I/O and manual compaction. Close the owner to release writer
        admission. If creation is interrupted, inspect the path before retrying.
        """
        ...

    @staticmethod
    def open(path: str, *, read_only: bool = False, options: OpenOptionsInput | None = None) -> Database:
        """Open an existing database at a built-in str path. The default requests writer
        admission, exclusive across processes. A competing writer raises CofferError.

        Pass read_only=True for a fixed view that rejects writes. options is an optional
        OpenOptionsInput mapping whose runtime overrides leave stored collection settings
        unchanged. A missing file raises CofferError. Use close or a context manager to
        end ownership.
        """
        ...

    # Only the bound receiver precedes the slash. The native method also accepts
    # config by keyword.
    def create_collection(self, /, config: CollectionConfigInput) -> Collection:
        """Create and persist a named collection on this writer; return its Collection
        handle.

        config requires name, embedding_dim and distance_metric. Omitted tokenizer uses
        default_en; omitted vector_index selects exact search. An existing name or a
        read-only owner raises CofferError. If interrupted after native work, check
        operation_outcome when available and look up the name before retrying.
        """
        ...

    def collection(self, /, name: str) -> Collection:
        """Look up a visible collection by name and return a handle tied to this owner.
        Missing collections raise CofferError. This lookup does not write to the file.
        """
        ...

    def snapshot(self, /, output_path: str) -> SnapshotReport:
        """Publish a standalone physical copy at an unused output_path and return its
        SnapshotReport.

        Call between operations on this writer. The source stays open under its existing
        writer lock and its bytes are unchanged. The destination directory must support
        hard links; an existing file is preserved. After interruption, inspect
        snapshot_outcome or CofferError.detail when available. If a target was published,
        verify its size, SHA-256 and last valid commit before using it or choosing another
        name. Physical backups can contain deleted data.
        """
        ...

    def rebuild_index(self, /, collection: str | None = None) -> RebuildIndexReport:
        """Rebuild derived indexes on this writer and return a RebuildIndexReport.

        Omit collection for the whole database. A named collection can still produce a
        full checkpoint, or only an in-memory verification if another collection has
        conflicting models. Inspect rebuild_scope, rebuild_needed and
        last_index_commit_id. Committed business data is preserved. If interrupted while
        rewriting the suffix, inspect maintenance_outcome when available, reopen and check
        the checkpoint before rebuilding again.
        """
        ...

    def compact(self, /, *, rebuild_indexes: bool = True) -> CompactReport:
        """Replace the main file with a validated compacted generation on this writer; return
        a CompactReport.

        Allow space in the database directory for both generations. Create a snapshot
        first to retain the old file. Existing readers can keep their old view and delay
        disk reclamation. rebuild_indexes defaults to True. If the replacement succeeds
        but index construction fails, the report carries index_rebuild='failed' and its
        reason. If interrupted, inspect maintenance_outcome when available; close, reopen
        and check the main path before retrying. If only the index failed, rebuild it on
        the verified file.
        """
        ...

    def list_collections(self, /) -> list[CollectionInfo]:
        """Return owned CollectionInfo dictionaries for collections visible to this owner.
        Each entry includes the name, IDs and settings needed to interpret its data.
        """
        ...

    def import_jsonl(self, /, manifest_path: str, *, on_conflict: Conflict = "error") -> ImportSummary:
        """Load the collection and JSONL inputs selected by manifest_path and return an
        ImportSummary.

        Requires this owner to hold the writer lock. Validates fixed inputs before bounded
        commits; an error or interruption may leave earlier batches committed. on_conflict
        defaults to error; skip and replace select alternative handling for conflicts.
        Inspect any operation_outcome and verify committed records before resuming.
        """
        ...

    def stats(self, /) -> DatabaseStats:
        """Return DatabaseStats for this owner's visible database view, including live-record
        counts, storage estimates and the effective cache budget. The result is owned by
        Python.
        """
        ...

    def index_stats(self, /) -> IndexStats:
        """Return IndexStats for the derived indexes without rebuilding them. Inspect
        rebuild_needed and the covered commit before relying on persisted coverage.
        """
        ...

    def close(self, /) -> None:
        """Release the native file and any writer lock, invalidating handles to this owner's
        collections. Repeated close is harmless. If another call is active, BusyError
        leaves the owner open; close it after that call finishes. An inherited owner
        raises ForkedHandleError.
        """
        ...

    # Use Self for context entry so type checkers preserve the caller's Database
    # type across the with statement.
    def __enter__(self, /) -> Self:
        """Enter a context for this open owner and return the same Database. Closed, busy,
        forked or poisoned owners raise their corresponding SDK exception.
        """
        ...

    def __exit__(self, /, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        """Close the owner on leaving the context and return False so a body exception
        propagates. If both the body and close fail, an exception group carries both
        errors.
        """
        ...


class Collection:
    """Access one named collection through its database owner. Obtain this handle from
    Database.collection or Database.create_collection.

    Calls share the owner's single-operation limit and its read view or writer lock.
    Closing the owner invalidates this handle. Returned documents, chunks, results and
    statistics are owned Python objects.
    """

    # Use BatchSeq here and for all three record kinds. A generic Sequence would
    # advertise custom sequence implementations the extension rejects.
    def write_batch(self, /, *, documents: BatchSeq[DocumentRecord] = (), chunks: BatchSeq[ChunkRecord] = (), embeddings: BatchSeq[EmbeddingRecord] = (), on_conflict: Conflict = "error") -> WriteSummary:
        """Validate and commit one mixed batch on this collection's writer and return a
        WriteSummary.

        Each of documents, chunks and embeddings takes a finite built-in list or tuple,
        empty by default. An empty batch does not commit. Inputs are fixed before the
        first write, and a large call is never split. on_conflict selects error (default),
        skip or replace. Inspect per-kind counts and warnings. After a signal or
        conversion error, inspect operation_outcome if available and verify the data
        before retrying: an atomic batch may already have committed.
        """
        ...

    def export_jsonl(self, /, output_dir: str) -> ExportSummary:
        """Write JSONL documents, chunks and embeddings from one visible view to output_dir
        and return the counts, checksums and manifest filename.

        This reads the collection and can replace previous export artifacts. The manifest
        is published last. If publication fails, check the files against the manifest and
        checksums before reimporting. Use a fresh export directory to keep a previous
        export.
        """
        ...

    def get_chunk(self, /, chunk_id: str) -> ChunkContent:
        """Read full chunk content and its parent document by the chunk's internal ID. Both
        records come from one view in an owned ChunkContent mapping. A missing chunk
        raises CofferError.
        """
        ...

    def get_chunk_by_external_id(self, /, chunk_external_id: str) -> ChunkContent:
        """Read full chunk content and its parent document by the application-provided chunk
        ID in this collection. Both come from one view; a missing chunk raises
        CofferError.
        """
        ...

    def get_document_by_external_id(self, /, external_id: str) -> Document:
        """Return the complete visible document for this collection's external_id, including
        its internal ID and stored metadata. A missing document raises CofferError. The
        source URI is not fetched.
        """
        ...

    def delete_document(self, /, document_id: str) -> None:
        """Logically delete the document with this internal ID on the collection's writer.
        The commit also hides its chunks and embeddings. Returns None. Bytes may remain in
        the file. After interruption, inspect any operation_outcome and verify visibility
        before retrying.
        """
        ...

    def delete_chunk(self, /, chunk_id: str) -> None:
        """Logically delete the chunk with this internal ID and its associated embedding on
        the collection's writer. Returns None; document and file bytes may remain. After
        interruption, inspect any operation_outcome and verify visibility before retrying.
        """
        ...

    def stats(self, /) -> CollectionStats:
        """Return an owned CollectionStats mapping for this collection's visible data,
        including document, chunk, embedding and tombstone counts and index/storage
        estimates.
        """
        ...

    # Keep text positional or keyword and the remaining search controls
    # keyword-only. The omission default for top_k is part of this signature and
    # must track the native method.
    def search(self, /, text: str | None = None, *, embedding: QueryEmbeddingInput | None = None, filter: JsonObject | None = None, weights: SearchWeightsInput | None = None, metadata_boost: JsonObject | None = None, top_k: int = 10, profile: Profile | None = None) -> SearchResults:
        """Return ranked SearchResults for text, an externally supplied query embedding, or
        both.

        filter restricts eligible records. weights and metadata_boost control score
        fusion. None uses input-dependent default weights and the owner's resource profile
        unless overridden. top_k defaults to 10 and accepts a built-in int from 1 to
        min(usize::MAX, 2**53-1), excluding bool. Inspect query_context and each hit's
        explain for actual candidate participation and scores. Results own their data; a
        later content lookup is a separate read.
        """
        ...
