# CofferDB 0.2.1 command-line manual

Use `coffer` from a terminal to create a local database, load data, search it,
inspect results and perform maintenance. Install the CLI from the matching Rust
crate with `cargo install cofferdb --version 0.2.1 --locked`. From the matching
source tree, use `cargo build --locked --bin coffer` and run
`./target/debug/coffer`. The Python wheel has a separate in-process API.

## First search

From the product root, with `coffer` on your PATH, run:

```sh
work="$(mktemp -d)"
coffer init "$work/notes.coffer"
coffer import "$work/notes.coffer" --manifest examples/data/markdown/import_manifest.json
coffer list-collections "$work/notes.coffer"
coffer search "$work/notes.coffer" 'write ahead' --collection notes --top-k 1 --json
```

The bundled manifest supplies the collection configuration and sample data.
The collection list includes `notes`; the first hit identifies `wal#commit`.
To start with your own files, follow [Import a folder of text](ingestion.md).

## Commands by task

The examples below use `DB` for a database path, `NAME` for a collection,
and uppercase placeholders for values you supply. Prefix each command with
`coffer`. Use `coffer help --json` to inspect the command list on your installed
version. Use `coffer profile-help` for the resource profiles.

| Set up or inspect | What it does |
| --- | --- |
| `init DB` | Create a database at a new path. |
| `create-collection DB NAME --dim N --metric cosine` | Set vector dimensions and distance metric. Optionally supply `--embedding-model` and tokenizer settings. |
| `list-collections DB` | List collection names and their configurations. |
| `stats DB` / `collection-stats DB --collection NAME` | Inspect database or collection counts. Add `--include-index` for index details. |

| Load, export or remove data | What it does |
| --- | --- |
| `import-dir DIR DB --collection NAME --chunk-size N --chunk-overlap M --emit-jsonl OUT` | Prepare text files offline. Add `--input-encoding` when needed. With an explicit `--embed` provider it can also generate vectors and import. |
| `embed-jsonl CHUNKS --provider PROVIDER --model MODEL --dimensions N --out FILE` | Send chunks to the selected embedding service and write vectors. Provider setup and failure handling are in the [import tutorial](ingestion.md). |
| `import DB --manifest FILE` | Import the files described by a manifest. Default duplicate handling is `error`; choose `--on-conflict skip` or `replace` explicitly. An import may commit several batches. |
| `import-embeddings DB FILE --collection NAME` | Attach JSONL vectors to existing chunks. The default conflict policy is `replace`. |
| `export DB DIR --collection NAME` | Write a collection and its JSONL export manifest. Keep the files together for logical restore. |
| `delete-chunk DB --collection NAME --chunk-external-id ID` | Logically delete the chunk and its embedding. |
| `delete-document DB --collection NAME --external-id ID` | Logically delete a document and its visible children. |
| `drop-collection DB --collection NAME` | Remove the collection and its visible contents. |

| Retrieve and explain | What it does |
| --- | --- |
| `search DB TEXT --collection NAME` | Run BM25 keyword search. Add `--embedding FILE` for hybrid search; `--top-k N` limits returned hits. |
| `vector-search DB --collection NAME --embedding FILE` | Search by a query vector. The JSON file supplies the vector, dimension and matching model identity. |
| `get-document DB --collection NAME --external-id ID` | Read the stored document. Use `--document-id` for an internal ID. |
| `get-chunk DB --collection NAME --chunk-external-id ID` | Read a full chunk. Use `--chunk-id` for an internal ID. |
| `why-not DB --collection NAME --query TEXT --chunk-external-id ID` | Explain where the target ended up: filtered, outside the candidates, below the returned ranks, or returned. |
| `explain-report DB --collection NAME --query TEXT --chunk-external-id ID --output FILE` | Produce a local HTML report at a new path. Open it offline to inspect the results and the selected target. |
| `benchmark DB --dataset FILE` | Run the benchmark with a separate database path. See the [benchmark guide](../benchmarks/README.md) for datasets and result interpretation. |

| Maintain a database | What it does |
| --- | --- |
| `check DB --verify-indexes` | Read one committed view and report integrity findings. An idle current-version writer can stay open. |
| `repair DB --truncate-tail` | Obtain exclusive write access and truncate only a verifiable uncommitted tail. |
| `rebuild-index DB` | Rebuild derived indexes. For a collection request, add `--collection NAME` and inspect the actual scope in the result. |
| `compact DB` | Replace the file with a compacted generation. Take a snapshot separately if you need the old one. |
| `snapshot DB NEWFILE` | Close the writer first. Copy a committed database to an unused path; the copy can be opened on its own. |

See [backup and recovery](backup.md) for space requirements, concurrent
readers, interrupted operations and what to verify before resuming writes.

A check reports two document counts. Inside the check report, `result.live_documents`
counts all visible committed documents. `index_suffix.live_document_count` counts
distinct documents represented by at least one live chunk in the checkpoint text
segments. With two documents and chunks in only one of them, a healthy report
shows 2 and 1 respectively. Multiple chunks of one document count once.

## Output and exit status

File sizes, byte offsets, commit identifiers, counts and durations in maintenance reports use
ordinary numbers through `2^53 - 1` and decimal strings above that value.
The snapshot package always stores its size and commit identifier as decimal strings.

Add `--json` to consume the result in a script. Completed commands write a
report to stdout. Inspect `ok`, `result` and `warnings`. On an operation error,
inspect the error on stderr and its `code` and `detail`. A completed `check`
can report findings on stdout and still return a nonzero status.

| Exit | Meaning |
| --- | --- |
| 0 | Completed successfully. |
| 1 | Invalid command, arguments, input or permission denied. |
| 2 | File, I/O, corruption, resource exhaustion or recovery error (including an unavailable check view). A completed check with only embedding-model mismatches also uses 2. |
| 3 | Check completed with integrity findings, subject to the exception above. |
| 4 | Lock contention. |
