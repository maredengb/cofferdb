# Rust 0.1.0 supplemental programming guide

This guide applies to the published Rust **0.1.0** API at
[`v0.1.0`](https://github.com/maredengb/coffered/tree/v0.1.0). It is supplied with newer documentation
without changing the original tag or package. The complete
[common lifecycle example](../examples/programming_guide.rs) was compiled and
run against that source. It uses only 0.1.0 APIs.

## Install and run

Use Rust 1.80 or later with a native toolchain, and pin the library when matching
this guide:

```toml
[dependencies]
cofferdb = "=0.1.0"
```

For the supplemental example, put `programming_guide.rs` in a small client's
`src/main.rs`, then run `cargo run -- create /a/new/path/notes.coffer`. A later
`cargo run -- read /the/existing/path/notes.coffer` opens it read-only. Create
refuses an existing file; the example does not remove an old database.
The synthetic two-dimensional vectors demonstrate the API, not semantic quality.

## Open, configure and close

`Database::create(path, DatabaseConfig::default())` creates a new file.
`Database::open(path)` opens an existing writer and
`Database::open_read_only(path)` opens a fixed reader view. Use
`open_with_options` for explicit runtime options; inspect `DatabaseConfig` in the
0.1.0 API reference in the [offline archive](reference.md).
Use one writer per database; do not interpret a file lock as distributed access
control. Drop temporary collection handles and call `db.close()` explicitly
when a fallible flush/release result matters.

Create a collection with `CollectionConfig`: name, tokenizer, dimensions,
metric, optional model identity, `vectors_normalized` and `VectorIndexConfig::Exact`.
The `default_en` tokenizer handles English and code identifiers, not Chinese
word segmentation. `Ann` in the enum does not make an ANN provider available.
Dimensions, tokenizer and model identity must match the stored collection.
Vectors come from the caller; the library does not generate or normalize them.

## Write and import

`insert_document(DocumentInput)` writes a document and its supplied chunks;
`put_embeddings(Vec<EmbeddingInput>)` is a separate operation. Do not describe
those two calls as one atomic mixed batch. Check each `InsertSummary`.
`DocumentInput` carries title, source URI and metadata; each chunk carries its
external ID, ordinal, complete text, metadata and optional byte/line source span.

For bounded JSONL ingestion, call
`db.import_manifest(path, OnConflict::Error)` with the manifest path.
`ImportManifest::load(path)` can separately inspect and validate its configuration;
the import method itself loads the path, not a preloaded `ImportManifest` object.
Use `Skip` or `Replace` explicitly when appropriate. Inspect inserted, skipped,
replaced and warning fields rather than assuming every input row was inserted.
A manifest import can commit multiple batches; failure does not imply rollback
of earlier batches. Do not blindly retry with a different conflict policy.

## Search, explain and retrieve full content

Start from `collection.search()`. Supply `.text(...)` for BM25,
`.query_embedding(...)` for exact vector retrieval, or both for hybrid.
`.filter(...)`, `.weights(...)`, `.metadata_boost(...)`, `.top_k(...)` and
`.resource_profile(...)` select the corresponding core options. End with
`.execute()?`. A result's `explain` records component scores, requested/effective
weights, their source and candidate participation.

The default resource profile is `balanced`; the other names are `expanded` and
`constrained`. They set resource defaults and do not rank relevance or promise
better performance. Default weights depend on supplied inputs and metadata
boost. See the [weight table](../README.md#search-inputs-and-weights); explicit
weights remain under application control. Recall cannot be attributed from score
shares, and external embeddings do not imply a relevance guarantee.

A search hit contains a snippet, not necessarily all stored text. In 0.1.0,
`collection.get_chunk(id)` or `get_chunk_by_external_id(id)` returns the full
owned `Chunk`; `get_document(&chunk.document_id)` retrieves its associated
`Document`. Use a fixed read-only handle when the two calls must observe one
stable view. The executable asserts complete text bytes, metadata, document ID,
source span and owned values after closing. It reads no original document file.
Do not call the 0.1.1 `get_chunk_content` method on a 0.1.0 dependency.

## Errors, deletion and maintenance

Methods return `cofferdb::Result<T>`. Inspect `Error::code()`/`code_str()` and the
structured `to_json()` projection. The fixed error keys include nullable
`suggested_action`; resource refusal is not syntax failure or proof of damaged
storage. A missing chunk uses `ChunkNotFound`. Keep original files when a read
is refused under resource or JSON policy.

Delete by native IDs with `delete_chunk` or `delete_document`; document deletion
also removes its visible children. This is logical deletion, not secure erasure.
Inspect collection and database stats. Rust also exposes integrity checking,
index rebuild, compaction and snapshot operations; use their actual options and
reports in the versioned reference. A snapshot or backup copy follows the
[quiescent-copy rules](backup.md), not an arbitrary live-file copy.

`db.export_collection_jsonl(collection_name, output_dir)` exports one consistent
collection view with its summary and manifest. The generated export manifest has
defined identity/configuration fields and is not interchangeable with arbitrary
user manifests. Use `ImportManifest::load` to validate supported input, preserve
the JSONL checksums, model/tokenizer and `vectors_normalized`, and import into a
new database. Never overwrite the authoritative file as a restore shortcut.

The 128-level/8 MiB business-JSON and aggregate 32 MiB workspace policies remain
applicable. They are not a full-text truncation rule or a bound on process RSS.
See [compatibility](compatibility.md), [backup/privacy](backup.md) and
[format/recovery](file-format.md) before deploying or migrating data.
