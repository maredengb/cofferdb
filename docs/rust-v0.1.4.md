# Rust 0.1.4 programming guide

Use Rust 1.80 or later. Pin the Rust crate in your application's `Cargo.toml`.

```toml
[dependencies]
cofferdb = "=0.1.4"
```

Follow the [Rust quickstart](../README.md#rust-and-python-quickstarts) to create
a collection, insert text and run a search. The [programming example](../examples/programming_guide.rs)
also covers batch writes, embeddings and ranked results. The CLI
[text ingestion guide](ingestion-v0.1.2.md) applies to this version.

## Create and restore a snapshot

Close application write handles, check that `Database::close()` succeeded, and
choose a new output path. From an application that uses `cofferdb`:

```rust
use cofferdb::{Database, Result};
use std::{fs, path::Path};

fn restore_example(source: &Path, snapshot: &Path, restored: &Path) -> Result<()> {
    // Use a fresh restore directory and a new destination file.
    Database::snapshot(source, snapshot)?;
    fs::copy(snapshot, restored)?;
    let db = Database::open_read_only(restored)?;
    let notes = db.collection("notes")?;
    let hits = notes.search().text("write ahead").top_k(1).execute()?;
    assert!(!hits.results.is_empty());
    drop(notes);
    db.close()?;
    Ok(())
}
```

The query above assumes the Markdown sample from the quickstart. Use a known
record or expected result from your own data when checking a real restore.
The snapshot can be transferred by itself to a compatible environment.
The [backup guide](backup.md) gives the equivalent CLI workflow, explains the
writer lock and distinguishes snapshot recovery from JSONL import and export.
