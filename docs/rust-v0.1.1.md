# Rust 0.1.1 programming guide

This guide describes the experimental 0.1.1 implementation in this source tree.
Check the [release record](https://github.com/maredengb/coffered/releases) for
available packages and their validation results. The
[0.1.0 guide](rust-v0.1.0.md) covers the retained lifecycle, configuration,
direct writes, manifest import/export, searches, error and maintenance methods.
Those methods retain their signatures and defaults in this version.

Build this source with Rust 1.80 or later. Run the executable from the product
root with a fresh destination:

```console
cargo run --locked --example programming_guide -- create /a/new/path/notes.coffer
cargo run --locked --example programming_guide -- read /the/existing/path/notes.coffer
```

The [joined-content and logical-restore example](../examples/joined_content.rs)
reads the database created above, exports it, reimports into a new database,
checks skip/replace/error handling and deletes from the restored copy:

```console
cargo run --locked --example joined_content -- /the/existing/path/notes.coffer /a/new/path/restored.coffer /a/new/path/export
```

The [source-backed example](../examples/real_corpus_demo.rs) additionally uses
the fixed Rust by Example corpus and precomputed BGE-M3 vectors supplied with
this tree. The data, model revision and licenses are in
[provenance.json](../examples/data/real_corpus/provenance.json). It makes no
representative quality claim.

## Joined complete-content reads

The additive `CollectionHandle::get_chunk_content(id)` and
`get_chunk_content_by_external_id(external_id)` return `ChunkContent`, containing
an owned `chunk` and its owned `document` from **one** read view. The text is
complete; source URI, metadata, text hash and optional source span come from the
database, without opening the original file. Both values remain usable after
close. A missing object remains a core error.

This does not pin a transaction across an earlier search and a later read.
Another permitted write may occur between calls; the read can see updated data
or report that the object no longer exists. Use a fixed read-only owner for
comparisons. Source offsets are byte positions, not character indices.

## Python and the shared engine

The [Python API](python-v0.1.1.md) directly links this Rust engine. Its narrow
doc-hidden binding adapter is implementation support, not a separately stable
Rust API. Normal rustdoc excludes that adapter and CLI internals. The Rust
library does not gain Python types or require Python at runtime.

Python's finite mixed `write_batch` uses one shared atomic commit. It does not
change the older Rust `insert_document` plus `put_embeddings` calls into a
single transaction. Manifest import retains its bounded multi-batch behavior.
SDK host-input conversion, owner/PID/GIL behavior and Python exception outcomes
are described in the Python guide; they are not new Rust cancellation promises.

The file-format version, canonical JSON/hash rules, tokenizer/model semantics,
query scoring and default weights remain unchanged. New binding-owned resource
work uses the same aggregate operation budget. A local file is not encrypted;
logical deletion and compaction do not guarantee secure erasure.

## Reference and compatibility

Use the **0.1.1** rustdoc and public symbol index in the matching
[offline reference archive](reference.md), built from this exact source.
They include public types, methods, fields, variants and return structures;
doc-hidden modules are not presented as supported application APIs.

The new Python macOS 15 baseline applies to that package. It does not amend the
platform declaration of the old Rust/CLI release. Compatibility validation must
separately cover same-engine Rust/Python round trips and legitimate 0.1.0 files
in both directions. The code changes alone do not certify those experiments or
future releases. Preserve original files and use the documented backup process.
