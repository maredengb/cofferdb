# Python 0.1.1 programming guide

This source contains an experimental synchronous, in-process Python API.
There is no 0.1.0 Python API. Use a matching 0.1.1 wheel and its checksum from
the [release record](https://github.com/maredengb/coffered/releases).
`cofferdb.__version__` identifies the SDK;
`cofferdb.engine_version` identifies the statically linked engine.

## Install a matching binary

Use regular-GIL CPython 3.12, 3.13 or 3.14 in a clean virtual environment and
`python -m pip` 25.3. The target matrix is:

| Platform | Matching platform tag | Native baseline |
| --- | --- | --- |
| macOS arm64 | `macosx_15_0_arm64` | macOS 15.0 or later, system libraries and the included engine |
| Linux x86_64 | `manylinux_2_28_x86_64` | glibc 2.28+, AlmaLinux 8 baseline |

Python and ABI tags must also match: `cp312-cp312`, `cp313-cp313` or
`cp314-cp314`. macOS 11–14, Intel macOS, Linux arm64/musl, Windows, PyPy,
free-threaded builds, subinterpreters and other Python versions are outside
this matrix. Higher macOS developer runs do not prove the macOS 15 minimum.
Actual platform results belong to the specific candidate artifact.

Prepare the installer and transfer the downloaded wheel and checksum manifest
before disconnecting an offline machine. Verify the checksum and install the
**actual filename**, with no source-build fallback:

```console
python -m pip install --no-index --no-deps --only-binary=:all: /path/to/the-matching-wheel.whl
python -c "import cofferdb; print(cofferdb.__version__, cofferdb.engine_version)"
```

The wheel has no third-party Python runtime dependencies. Rust, maturin, a CLI,
model downloads and provider keys are not required for binary use. An offline
source build is a different workflow and requires the toolchain and locked
build dependencies. See [source build details](../bindings/python/README.md).

## Run the complete source-backed example

Unpack the matching source/example material next to a new working directory.
With the wheel installed, run from the product root:

```console
python examples/code_retrieval.py /a/new/path/code.coffer
```

The [executable](../examples/code_retrieval.py) imports two source documents,
five chunks and five precomputed vectors; runs keyword, vector and hybrid
search with a metadata filter; checks explain; retrieves the complete text and
source coordinates; closes and reopens the database and compares owned values; then exports,
restores and deletes from a separate restored copy. It exercises explicit
replace with the unchanged source vector, skip, duplicate rejection and strict
`top_k` type handling.
At least one result is longer than its snippet. It verifies the bundled file
hashes and source spans before import and reads no original source file during
the reopened database read.

The immutable Rust by Example source commit, BAAI/bge-m3 model revision,
1,024-dimensional vectors and upstream MIT/Apache notices are recorded in
[provenance](../examples/data/real_corpus/provenance.json). The corpus is retrieval
data, not executable code. This small example checks functionality, not general
semantic recall. No example text is silently changed while keeping old vectors.

For finite direct writes, the [batch quickstart](../bindings/python/examples/quickstart.py)
creates a small collection and validates returned data and explain. Its small
two-dimensional vector is explicitly synthetic.

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
still derives and enforces the effective [model binding](compatibility.md#embedding-models-and-export-configuration)
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
See the generated return-shape/type reference rather than inventing optional
fields or replacing missing values with zero.

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
Local storage is not encryption; no telemetry, model/provider access or upload
runs implicitly. Python does not expose Rust compact/check/snapshot/repair or
why-not debug operations in this version.

Use the installed-package API/type reference in the matching
[offline archive](reference.md). Do not use newer Rust APIs or mutable `main`
documentation as a replacement for the actual installed SDK reference.
