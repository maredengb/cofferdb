# Rust 0.2.0 programming guide

The Rust library and the `coffer` CLI use the same source crate. The maintenance
operations below support local filesystems on Linux and macOS. To install the CLI, run `cargo install cofferdb --version 0.2.0 --locked`.
Use Rust 1.80 or later. Pin the Rust crate in your application's `Cargo.toml`.

```toml
[dependencies]
cofferdb = "=0.2.0"
```

## Write, search and reopen

Create a binary application with `cargo new field-notes`, add the dependency
above under its existing `[dependencies]` heading, and put the following in
`src/main.rs`. Run `cargo run` from that application in a directory where
`notes.coffer` does not exist. It prints the stored sentence and reads the
same chunk after closing and reopening the database.

```rust
use cofferdb::{
    ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric, DocumentInput,
    JsonValue, VectorIndexConfig,
};

fn main() -> cofferdb::Result<()> {
    let db = Database::create("notes.coffer", DatabaseConfig::default())?;
    let notes = db.create_collection(CollectionConfig {
        name: "notes".into(),
        tokenizer: Default::default(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    })?;
    notes.insert_document(DocumentInput {
        external_id: "field-notes".into(),
        title: Some("Field notes".into()),
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: vec![ChunkInput {
            chunk_external_id: "field-notes#1".into(),
            ordinal: 0,
            text: "Jade leaves grow near the river.".into(),
            metadata: JsonValue::object(),
            source_span: None,
        }],
    })?;
    let found = notes.search().text("jade").top_k(1).execute()?;
    let hit = found.results.first().expect("search returned no results");
    println!("{}", notes.get_chunk(&hit.chunk_id)?.text);
    drop(notes);
    db.close()?;
    let reader = Database::open_read_only("notes.coffer")?;
    let text = reader.collection("notes")?
        .get_chunk_content_by_external_id("field-notes#1")?.chunk.text;
    assert_eq!(text, "Jade leaves grow near the river.");
    reader.close()?;
    Ok(())
}
```

## Open, configure and close

`Database::create(path, DatabaseConfig::default())` creates a new file.
`Database::open(path)` opens an existing writer and
`Database::open_read_only(path)` opens a fixed reader view. Use
`open_with_options` for explicit runtime options; inspect `DatabaseConfig` for resource profile, cache budget, I/O and durability options.
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
`put_embeddings(Vec<EmbeddingInput>)` is a separate operation. Each call commits separately. Check each `InsertSummary`.
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

A hit carries a snippet and a chunk ID. Call `get_chunk_content(&hit.chunk_id)`
or `get_chunk_content_by_external_id(...)` to read the complete chunk and its
document from one view. The result owns its text, metadata and optional source
coordinates, which remain usable after close. A separate search and read can
observe different commits on a writer. Use a read-only handle for a fixed view.

## Export, delete and handle errors

`db.export_collection_jsonl(collection_name, output_dir)` writes the live
documents, chunks, embeddings and a manifest for one collection. To restore
them into a new database, pass the generated `export_manifest.json` to
`import_manifest`. Keep the manifest and the JSONL files together.
Use `delete_chunk` or `delete_document` with native IDs. Document deletion
removes its visible children. Deleted content can remain on disk.

Core operations return `cofferdb::Result<T>`. Inspect `Error::code()` or
`code_str()` and the structured detail in `to_json()`. A missing chunk has
code `ChunkNotFound`. Inspect errors from writes and explicit close before
retrying. The [worked example](../examples/programming_guide.rs) demonstrates
vector and hybrid queries, filters, full reads and error handling.

## Check and recover

Use `Database::check_path(path, CheckOptions::default())` to examine an
existing file. For a snapshot, successfully close the writer and choose an
unused destination. Continuing with the database from the example:

```rust
use cofferdb::{CheckOptions, Database};

fn main() -> cofferdb::Result<()> {
    Database::snapshot("notes.coffer", "notes-backup.coffer")?;
    let report = Database::check_path("notes-backup.coffer", CheckOptions::default())?;
    assert_eq!(report.status, "ok");
    let db = Database::open_read_only("notes-backup.coffer")?;
    let row = db.collection("notes")?.get_chunk_content_by_external_id("field-notes#1")?;
    assert_eq!(row.chunk.text, "Jade leaves grow near the river.");
    db.close()
}
```

To resume writes, first copy the verified snapshot into a fresh directory.
The snapshot takes exclusive writer admission on its source and refuses an
existing destination. To keep the previous generation before compaction,
take a snapshot separately.

| Rust operation | When to use it |
| --- | --- |
| `Database::repair_truncate_tail(path)` | After closing writers, truncate a tail only when its commit boundary can be verified. Preserve a separate copy if the original bytes matter. |
| `db.rebuild_index(None)` or `db.rebuild_index(Some(name))` | On a writable database, rebuild derived index data. Read the reported scope when requesting a collection. |
| `db.compact(CompactOptions::default())` | With a writable handle, rewrite live data into a replacement file. Allow space for both generations. |

The CLI also provides `check`, `repair --truncate-tail`, `rebuild-index`,
`compact` and `snapshot`. The [backup guide](backup.md) explains their locks
and failure behaviour. For directory imports, see the [text ingestion
guide](ingestion.md). For search diagnostics, see [explain](explain.md).
