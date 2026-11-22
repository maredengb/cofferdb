# Python 0.2.0 programming guide

## Install the matching package

Use regular-GIL CPython 3.12, 3.13 or 3.14 on macOS 15 or later on Apple
Silicon (arm64) or Intel (x86_64), or on 64-bit Linux (AArch64 or x86_64)
with glibc 2.28 or later. Install the matching wheel in a clean virtual environment.

```console
python -m pip install --only-binary=:all: cofferdb==0.2.0
python -c "import cofferdb; print(cofferdb.__version__, cofferdb.engine_version)"
```

Both version values should be `0.2.0`. For offline installation, download the
matching wheel and checksum list from the [0.2.0 release](https://github.com/maredengb/cofferdb/releases/tag/v0.2.0),
verify the wheel against that list and install the local file with
`python -m pip install --no-index --no-deps /path/to/the-matching-wheel.whl`.

The wheel includes the Rust engine. Binary use needs no Rust toolchain or CLI. Select the wheel for your CPython version and architecture. Windows, musl, free-threaded builds and PyPy are outside this package matrix.

## Write, search, reopen and check

Save this as `notes.py` in a fresh directory and run `python notes.py`.
It creates `notes.coffer`. Use another empty directory to repeat the example.
The two numbers below are a synthetic vector; the example needs no external
files, model, service or CLI.

```python
import cofferdb
from cofferdb import Database

path = "notes.coffer"
vector = {"model_name": "sample-2d", "dim": 2, "vector": [1.0, 0.0]}
with Database.create(path) as db:
    notes = db.create_collection({
        "name": "notes", "embedding_dim": 2, "distance_metric": "cosine",
        "embedding_model": "sample-2d",
    })
    counts = notes.write_batch(
        documents=[{"external_id": "field", "title": "Field notes"}],
        chunks=[{"document_external_id": "field", "chunk_external_id": "field#1",
                 "ordinal": 0, "text": "Jade leaves grow near the river."}],
        embeddings=[{"chunk_external_id": "field#1", **vector}],
    )
    assert counts["chunks_inserted"] == 1
    hit = notes.search("jade", top_k=1)["results"][0]
    assert hit["chunk_external_id"] == "field#1"

with Database.open(path, read_only=True) as db:
    notes = db.collection("notes")
    hit = notes.search(embedding=vector, top_k=1)["results"][0]
    content = notes.get_chunk(hit["chunk_id"])
    assert content["document"]["title"] == "Field notes"
    print(content["chunk"]["text"])

report = cofferdb.check(path)
assert report["status"] == "ok", report["issues"]
print(report["status"])
```

Expected output is `Jade leaves grow near the river.` followed by `ok`.
For a hybrid query, pass both inputs to the same call:
`notes.search("jade", embedding=vector, top_k=1)` inside the open context.
Use `hit["explain"]` to see which components contributed to the score.
The [larger example](../examples/code_retrieval.py) demonstrates filtered
hybrid search, replacement, logical export and restore with a bundled corpus.

## Data, configuration and ownership

Create with `Database.create(str(path), config=None)`; open an existing database
with `Database.open(str(path), read_only=False, options=None)`. Paths are built-in
strings, not arbitrary path protocols. `create` refuses existing files and
`open` does not create a missing file. The context manager closes the database;
explicit `close()` is also supported and can report a core error.

With no runtime overrides, create/open use these values:

| Runtime option | Default |
| --- | --- |
| `resource_profile` | `"balanced"` |
| `cache_budget_mb` | `None`: use the selected [resource preset's budget](../README.md#resource-presets-and-limits) |
| `background_compaction` | `False` |
| `durability` | `"commit"` |
| `io_mode` | `"auto"` |

Passing `config=None` or `options=None` is equivalent to omitting that whole
mapping. Within a create configuration, only `cache_budget_mb` accepts `None`.
Within open options, `None` leaves `resource_profile`, `cache_budget_mb`,
`background_compaction` and `durability` at their runtime defaults;
`io_mode=None` is invalid. An explicit zero cache budget and `False` are
preserved.

`create_collection(config)` requires the collection name, embedding
dimension and distance metric. Omitted optional fields select
`embedding_model=None` (no explicit model setting), `vectors_normalized=False`,
`vector_index="exact"` and the complete `default_en` tokenizer identity. The core
still derives and enforces the effective model binding
from committed embeddings. Only `embedding_model` accepts `None` among those optional
collection fields. `collection(name)`, `list_collections()`, `stats()` and `index_stats()`
observe existing state. Runtime configuration follows the core rules; it does
not rewrite persisted collection settings. The English/code `default_en`
tokenizer does not provide Chinese word segmentation.

Use exact built-in dictionaries for records/configuration, and built-in lists
or tuples for finite batches/vectors. Generators, NumPy coercion, subclasses
and arbitrary conversion protocols are not accepted. `bool` is not an integer
argument. Numeric values must be finite; exact integers must not silently round
through JSON. Cycles, invalid Unicode and excessive business JSON are rejected.
The installed package includes type definitions for public inputs and results.

The returned dictionaries/lists own their data. `get_document_by_external_id(id)`
returns complete document metadata, including its internal ID. `get_chunk(id)` and
`get_chunk_by_external_id(id)` return `{"chunk": ..., "document": ...}` from
one read view, with full text, IDs, metadata, hash and source span. They do not
read the source URI. Byte offsets above `2**53-1` through `u64::MAX` are exact
decimal strings; safe offsets and line numbers are integers; unknown values
are `None`. Explicit `source_span`, including `None`, takes precedence over
metadata span hints; retained conflicts produce warnings.

A separate search and read are not one transaction. A writer can change state
between them. A read-only database retains its fixed view; returned data stays
valid even after closing.

## Write, replace, export and delete

`collection.write_batch(documents=(), chunks=(), embeddings=(), on_conflict="error")`
validates all three groups and commits one atomic mixed batch. Documents can
exist without chunks, and embedding-only updates are supported. Missing groups
are empty; explicit `None` is invalid. A large batch is refused rather than
silently split. Use `skip` or `replace` explicitly and check all inserted,
skipped, replaced and warning fields in the summary.

`db.import_jsonl(manifest_path, on_conflict="error")` performs bounded commits;
the whole import is not atomic. `collection.export_jsonl(output_dir)` returns a
summary and writes a supported `export_manifest.json`. Reimport that generated
manifest through `db.import_jsonl` into a new database; the parser validates its
defined schema/configuration rather than accepting arbitrary export fields.
Keep data checksums, tokenizer/model identity and `vectors_normalized`.

`delete_chunk(chunk_id)` and `delete_document(document_id)` use native IDs;
document deletion cascades to visible child chunks/embeddings. Deletion is
logical and is not secure erasure. Use the [backup guide](backup.md) before
changing authoritative data.

## Search and failure handling

`collection.search(text=None, *, embedding=None, filter=None, weights=None,
metadata_boost=None, top_k=10, profile=None)` requires text, a valid query vector,
or both. `top_k` is an exact integer in
`1..min(usize::MAX, 9007199254740991)`. Three routes, filters, weights, metadata
boost and explain use the same engine as Rust. The three profile strings are
exactly `expanded`, `balanced` and `constrained`; `None` inherits runtime
configuration. Input-dependent default weights and their requested/effective
forms are visible in explain. Increasing `top_k` can change normalization and
ranking, and does not enlarge budgets or guarantee an unchanged prefix.

Catch `CofferError` for core failures and inspect `code`, `message`, `detail`,
`recoverable` and nullable `suggested_action`. SDK `ClosedError`, `BusyError`,
`ForkedHandleError`, `PoisonedHandleError` and `EnginePanicError` carry their defined
`sdk_reason`; the common SDK base does not promise that field. Type/value
conversion errors and genuine `MemoryError` keep their own categories.

After a failed write, inspect an available `operation_outcome`. `state` is one
of `not_started`, `not_committed`, `committed`, `partially_committed` or `unknown`;
only confirmed progress fields are included. Records count committed engine
records, not requested rows. Unknown does not mean rollback. A result-conversion
error can occur after commit; do not replay the write automatically.

## Threads, signals and resource limits

One owner admits one native operation at a time. Concurrent/reentrant work or
close raises `BusyError` without a per-owner queue. Native work releases the
GIL; conversion and result materialization hold it. There is no asyncio API,
forced cancellation or per-call timeout. A pending signal may raise
`KeyboardInterrupt` after native work committed; the exception preserves known
commit observations and any original core error.

Collections retain their owner but do not keep its writer open after explicit
close. Further access raises `ClosedError`. A context body and failed cleanup
are preserved together through `BaseExceptionGroup`; returned data is separate
from handle lifetime. A caught native panic poisons the owner and permits only
safe closing. Garbage collection is not a durable flush contract.

Inherited handles after `fork` are refused before accessing native locks;
finalization in the child does not unlock the parent's writer. Prefer `spawn`
or explicitly reopen in a safe child process. A fresh owner is still subject
to the parent's real file lock.

Conversion, execution and result construction share the aggregate 32 MiB
operation workspace. Business JSON retains its 128-level/8 MiB policy; full
text, a dataset or an entire batch is not assigned a new 8 MiB truncation cap.
Host inputs/results, native memory, caches and total process RSS are distinct.
The database file can contain sensitive text. Apply filesystem permissions
and protect exported files and backups.

## Check and restore a snapshot

`cofferdb.check(path)` reads one committed view. Inspect `status`, `issues`,
`index_check` and `index_check_reason` for findings and skipped checks. An idle
current-version writer may stay open. If a consistent view cannot be obtained,
inspect the `CofferError`: a busy check view has code `recovery_error` and
`detail["reason"] == "check_view_unavailable"`. Retry after the write finishes. The check leaves the database bytes unchanged.

Between batches, keep your writer open and choose an unused backup name:

```python
from cofferdb import Database
from pathlib import Path
import hashlib

with Database.open("notes.coffer") as db:
    backup = db.snapshot("notes-backup.coffer")
    with Path(backup["output_path"]).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == backup["sha256"]
    index = db.rebuild_index("notes")
    if index["rebuild_needed"]:
        print(index["rebuild_scope"], index["index_suffix_status"])
    result = db.compact()
    if result["index_rebuild"] == "failed":
        print(result["index_rebuild_reason"])
```

A collection-specific rebuild can persist a full checkpoint. If another
collection has conflicting models it can verify only the requested collection
in memory. Check the reported scope and covered commit. Compact replaces the
main file and rebuilds indexes by default. Pass `rebuild_indexes=False` to
skip that step. An old reader can keep disk blocks allocated until it closes.
To use the separate CLI for these operations, first release the writer.

A snapshot is a complete database file. To verify this sample with the wheel:

```python
from cofferdb import Database, check

path = "notes-backup.coffer"
assert check(path)["status"] == "ok"
with Database.open(path, read_only=True) as db:
    hit = db.collection("notes").search("jade", top_k=1)["results"][0]
    assert hit["chunk_external_id"] == "field#1"
```

For a writable restore, first copy the snapshot into a fresh directory and open
that copy. Use the CLI for explicit tail repair. The [backup guide](backup.md)
covers disk requirements and recovery. A physical backup may contain deleted
or sensitive data. SHA-256 verifies bytes.

If an exception follows publication, inspect `snapshot_outcome`, or the
engine error's `detail`, and verify the target's size, digest and commit.
Leave an existing target intact. For rebuilding or compaction, inspect
`maintenance_outcome` if present. After an uncertain replacement, close and
reopen the main path and run `check`. If only the index failed, verify the
current file and rebuild the index. A secondary cleanup error identifies a
staging file that could not be removed. Preserve unrelated files at that name,
resolve the cleanup error and reopen the main path before retrying. After interruption or a missing receipt,
verify the actual files before retrying.
