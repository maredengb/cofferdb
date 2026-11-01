# CofferDB Python 0.1.1

This source contains the experimental in-process binding for CofferDB 0.1.1.
Use the [release record](https://github.com/maredengb/coffered/releases) to find
available artifacts, their checksums and package-specific validation results.
The native extension statically links the Rust engine from the same source tree.

The target runtimes are regular GIL builds of CPython 3.12, 3.13 and 3.14 on
macOS arm64 and Linux x86_64 with glibc. Source builds require the Rust toolchain
and maturin 1.10.2; the intended standard installation uses prebuilt wheels.

## Installation

For a matching wheel obtained from a release or a local build:

Use a clean virtual environment with `python -m pip` 25.3. Prepare that
installer version before transferring the environment to an offline machine.

```console
python -m pip install --no-index --no-deps --only-binary=:all: /path/to/cofferdb-0.1.1-<matching-tags>.whl
```

The wheel contains the native engine. Using it requires no Rust toolchain,
server process, CLI executable, model provider, or model download. Unsupported
interpreter/platform combinations fail at import instead of falling back to a
source build or a different execution route. The declared lower bounds are
macOS 15 on arm64 and glibc 2.28 (AlmaLinux 8) on Linux x86_64; macOS 11–14
is unsupported by this Python package. Wheel tags alone
do not establish that every minimum-environment acceptance run has passed.

The package uses Apache-2.0. The wheel includes the project license and the
selected Cargo dependencies' upstream notices. The source inventory at
[`licenses/inventory.json`](licenses/inventory.json) records their exact versions,
declared terms and notice hashes; it is not a claim of completed binary release
clearance for every platform. Python and native-extension SDK versions must
match; a mismatched installation fails at import.

For a source build, use the locked PyO3 0.27.2 and maturin 1.10.2 inputs:

```console
python -m pip install maturin==1.10.2
python tools/prepare_source.py /a/new/build-directory
cd /a/new/build-directory
MACOSX_DEPLOYMENT_TARGET=15.0 maturin build --locked --release
maturin sdist
```

Start source preparation from the product root, beside its canonical
`pyproject.toml`; the binding crate remains in `bindings/python`. The prepared
directory contains the engine, binding, both locks, notices, guides and examples.
Its static `cofferdb/_source.json` records source and lock identities without
performing a runtime network lookup. Without `--commit`, preparation marks the
source as an unversioned export and makes no product-commit claim. See the
[source/reference build guide](../../docs/reference.md).
An offline source build also requires
the exact locked Rust dependencies to be available locally.
The deployment variable is required for macOS builds; the binding rejects a
different or missing value instead of producing a misleading wheel tag. Linux
builds keep the separately declared glibc baseline.

## First database and query

```python
from cofferdb import Database

with Database.create("notes.coffer") as db:
    notes = db.create_collection({
        "name": "notes", "embedding_dim": 2, "distance_metric": "cosine",
    })
    summary = notes.write_batch(
        documents=[{"external_id": "field-notes", "title": "Field notes"}],
        chunks=[{
            "document_external_id": "field-notes",
            "chunk_external_id": "field-notes#1",
            "ordinal": 0,
            "text": "Jade leaves grow near the river.",
        }],
    )
    hits = notes.search("jade", top_k=10)
    first = notes.get_chunk(hits["results"][0]["chunk_id"])
    assert first["chunk"]["text"] == "Jade leaves grow near the river."

# Returned objects are ordinary owned Python dictionaries and lists.
assert summary["chunks_inserted"] == 1
assert first["document"]["external_id"] == "field-notes"
```

Use `Database.open(path, read_only=True)` for an existing fixed reader view.
Paths are built-in `str` values; convert a `Path` explicitly with `str(path)`.
`create` refuses existing destinations; `open` does not create missing files.
Use a fresh path when repeating the example.

## Vectors and hybrid retrieval

```python
from cofferdb import Database

with Database.open("notes.coffer") as db:
    notes = db.collection("notes")
    notes.write_batch(embeddings=[{
        "chunk_external_id": "field-notes#1", "model_name": "sample-2d",
        "dim": 2, "vector": [1.0, 0.0],
    }])
    results = notes.search(
        "jade", embedding={"vector": [1.0, 0.0], "dim": 2,
                           "model_name": "sample-2d"},
        weights={"bm25": 0.5, "vector": 0.5, "metadata": 0.0},
    )
    assert "explain" in results["results"][0]
```

The small vector above is a synthetic API example, not a language model.
CofferDB accepts externally generated vectors and uses its shared exact vector,
keyword, filter and fusion implementation. Embedding semantic quality and a
user's selected weights affect application retrieval quality; this package does
not guarantee a fixed application recall. Scores and component weights do not
measure causal recall contributions or relevance probabilities.

Omitted weights and profiles use the core's input-dependent defaults and the
database's runtime profile. Only `expanded`, `balanced` and `constrained` are
resource profile values. Explain includes requested/effective weights and their
source. Increasing `top_k` can change candidate normalization and ranking; it
does not promise an unchanged prefix or improved quality.

## Input, atomicity and failures

- Configurations and records require built-in `dict`; finite batches and vectors
  require built-in `list` or `tuple`. Generators, array coercions, container
  subclasses and arbitrary conversion protocols are not accepted. `bool` does
  not count as a numeric argument. Numbers must be finite; metadata integers
  cannot silently round when converted to the core's canonical JSON.
- `write_batch` validates all three groups before one atomic core commit and
  never splits a large batch. Empty groups default to `()`; explicit `None` is
  invalid. Documents may have no chunks, and embedding-only updates are valid.
  Conflict strategies are `error`, `skip` and `replace`.
- `db.import_jsonl(manifest_path)` validates fixed manifest inputs before bounded
  commits. An entire manifest import is not atomic. Use
  `collection.export_jsonl(output_dir)` for a consistent collection export and
  its reimportable manifest; existing target safeguards and cleanup follow the
  core implementation.
- Full chunk reads include the complete text and its document from one read
  view. They do not read the original source file or return only a snippet.
  Safe byte offsets are integers; offsets above `2**53-1` through `u64::MAX`
  are exact decimal strings. An explicit `source_span`, including `None`, takes
  priority over metadata span fields; conflicting metadata is retained and a
  warning is returned.
- Core `CofferError` exposes `code`, `message`, `detail`, `recoverable` and
  nullable `suggested_action`. SDK lifecycle failures have separate exception
  classes and `sdk_reason`. Type/value conversion errors and genuine Python
  `MemoryError` retain their own categories.
- A failed write call can expose `error.operation_outcome` with `state` and only
  known confirmed commit fields. Counts include committed core records,
  including collection and tombstone records; they are not input row counts.
  `unknown` does not mean rolled back. Do not blindly replay a failed write.

One call shares the core's aggregate working budget across conversion, execution
and result construction. Business JSON values have their existing size/depth
limits; a full text, entire batch or full result is not assigned an extra 8 MiB
cap. A finite batch can still exceed working capacity and be refused before a
commit. Host-owned input/results, Python heap, native memory and process RSS are
different quantities; the engine cache budget is not a process RSS limit.

## Lifecycle, threads and fork

All methods are synchronous. Native execution releases the GIL; Python input
validation and result materialization require it. One database owner admits one
operation at a time. Concurrent or reentrant work/close raises `BusyError`
instead of waiting in a per-owner queue. A caught native panic poisons the
owner, discloses no private payload and allows only safe closing.

Explicit `close` releases the owner's native database even if collections are
retained. Those collections then raise `ClosedError`; already returned data
remains usable. Context exit returns `False`; if the body and cleanup both fail,
a `BaseExceptionGroup` preserves both errors.

An owner inherited after `fork` raises `ForkedHandleError` before touching its
native locks, including on close. Its finalizer does not unlock the parent's
writer. A child may open a fresh owner, but the parent's real writer lock still
applies; other host extensions are outside this guarantee. Prefer `spawn`, or
close handles before forking when practical.

There is no forced cancellation or per-call timeout. A pending Python signal
may raise `KeyboardInterrupt` after native work has already committed. Known
commit observations and an original core error are preserved. Stopping the
caller from waiting does not stop or roll back the engine operation.

## API reference and versions

`cofferdb.__version__` identifies the SDK; `cofferdb.engine_version` reports the
actually linked Rust crate version. The package includes public
`TypedDict`/`Literal` shapes, `__init__.pyi`, `py.typed` and native docstrings.
These are type hints, not runtime coercion or schema bypasses. The native API
does not expose every Rust maintenance operation.

The [programming guide](../../docs/python-v0.1.1.md) and
[source-backed executable](../../examples/code_retrieval.py) cover the complete
retrieval workflow. The [reference source](docs/index.rst) documents the exported API. Generated
offline reference archives must be built against the matching installed wheel.
The [Rust product README](../../README.md) covers file/format and operational
boundaries, including backup, deletion, privacy and secure erasure. Local files
are not encrypted merely because they are embedded or opened read-only.
