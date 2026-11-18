# Text ingestion

This guide describes command-line text ingestion in CofferDB 0.1.2.
For Rust integration and the matching dependency pin, see the
[Rust 0.1.2 guide](rust-v0.1.2.md).

Run commands from the product directory, the directory that contains
`Cargo.toml`. `coffer help` lists `embed-jsonl` and `import-dir` with their
flags. There is no separate `--help` switch.

```console
cargo run --locked --bin coffer -- help
```

## Three paths

`import-dir` without `--embed` writes text artifacts only. It walks the input
directory, splits files into chunks, and writes `documents.jsonl`,
`chunks.jsonl`, and `import_manifest.json` under `--emit-jsonl`. It does not
write embeddings or `embedding_artifact.json`. It does not read an API key,
open a network connection, or open, create, or modify the database path. The
command still succeeds when that path is missing.

`embed-jsonl` reads an existing chunks file and writes `embeddings.jsonl`.
When `--metadata-out` is omitted, it also writes `embedding_artifact.json` in
the same directory as `--out`. It does not write `import_manifest.json` and
does not open a database.

`import-dir --embed` writes the text artifacts, the embeddings, the sidecar,
and a manifest that copies the existing collection's dimension, distance
metric, and tokenizer. It then imports that manifest. `--embed` requires
`--emit-jsonl`. The value is `provider:model`, for example
`openai-compatible:example`.

`import-dir` without `--emit-jsonl` fails before it walks the input directory
or touches a database. Create the collection with `create-collection` before
an embedding import. This layer does not create a collection and does not
choose a distance metric.

## Chunks

`--chunk-size` and `--chunk-overlap` count UTF-8 bytes after one leading BOM
is removed. Both are integers from 1 through 1048576, with no sign and no
leading zeros. The overlap must be smaller than the chunk size. Cuts stay on
Unicode scalar boundaries. The same input, format, chunk size, and overlap
produce the same documents, chunks, and hashes.

`--format` is `text` by default or `markdown`. Repeated `--include` and
`--exclude` accept a closed gitignore-style subset. Dot-directories, and the
directories `.git`, `target`, `node_modules`, and `vendor`, are not read.
Symlinks are not followed. A symlink used as the input directory fails. A
non-UTF-8 or whitespace-only text file is skipped. Whitespace-only chunks are omitted; retained chunks preserve their original byte spans. A run with no eligible file, or with no chunk,
fails and does not leave a manifest that claims success.

## Providers

An embedding request uploads chunk text, the model name, and `--dimensions`.
Excluded file contents are not uploaded. The API key is sent only as an
`Authorization` header. It is not written into the artifacts, the sidecar, or
the database.

`openai` posts to `https://api.openai.com/v1/embeddings`. The default base is
`https://api.openai.com/v1`. `--base-url` is the only switch that changes the
base. `OPENAI_BASE_URL` is not consulted. The key is read from
`OPENAI_API_KEY` unless `--api-key-env` names another variable.

`openai-compatible` has no default host. `--base-url` and `--api-key-env` are
both required. This provider does not read `OPENAI_API_KEY`.

A missing, empty, whitespace-only, or control-character key fails before any
HTTP client or request is created. Any other value is sent unchanged. A provider name of
`local`, or a model-file path, is rejected. This command does not download a
model or run local inference.

`--dimensions` is required. It is an integer from 1 through 4294967295, with
no sign and no leading zeros, and it is sent on every request. The default
batch size is 32, from 1 through 2048. The default concurrency is 1, from 1
through 8. Each request waits at most 60 seconds. There are no retries, and
redirects are not followed.

The sidecar records the provider, model, dimension, batch size, artifact
hashes, host, and a SHA-256 fingerprint of the request URL. It does not record
the key. Its timestamp is not part of the JSONL or the manifest. Import does
not need the sidecar.

## Failures

A generation batch is one HTTP request. If a generation batch fails, later
batches are not started. The database is not created or modified. An existing
embeddings file is replaced only after the new bytes are complete, so a failed
run does not append to it. Artifact staging uses the output directory and
does not delete unrelated files outside it. HTTPS passes the authorization
header to curl through a pipe, not as a process argument. The request body is
a mode 0600 file in the system temporary directory and is removed when the
request finishes. If the process is killed during the request, that file can
remain until removed. `--out` and the sidecar must be different files, and the
sidecar must not use the same path as `documents.jsonl`, `chunks.jsonl`,
`embeddings.jsonl`, or `import_manifest.json`.

Committed core batches start when manifest import starts. That import uses
conflict handling `error` and commits each core batch. A failure after that
point keeps core batches that are already committed. The command can fail
while the database has changed. In JSON errors, `detail.import_stage` is
`core_import` when the manifest importer returns an error. The field identifies
the stage; it does not report a commit count. A second import of the same external ids
returns `duplicate_external_id`. Its suggested action is to rerun `import`
with `--on-conflict skip` or `--on-conflict replace`. `import-dir` does not
add its own conflict switch.

These two phases use those names. A generation-batch failure is not a
committed core batch, and a committed core batch is not rolled back when a
later batch fails.

User and input errors exit 1. That includes a refused command and permission
denied while opening an existing database. I/O errors, including provider
connection, timeout, and non-success HTTP status, exit 2. Lock contention
exits 4. `coffer help` lists the exit codes and the two command boundaries.

## Offline example

`examples/data/ingest/source/note.txt` is the eleven bytes `hello world`.
The files next to `import_manifest.json` are a fixed snapshot for collection
`notes`: one chunk, `file://note.txt#chunk-0001`, model `example`, dimension
2, cosine distance, `vectors_normalized` false, and vector `[1, 0]`. The
sidecar records the local compatible server that produced that vector. No
hosted key is required to import the snapshot.

```console
cargo run --locked --bin coffer -- init /tmp/notes.coffer
cargo run --locked --bin coffer -- create-collection /tmp/notes.coffer notes \
  --dim 2 --metric cosine --embedding-model example
cargo run --locked --bin coffer -- import /tmp/notes.coffer \
  --manifest examples/data/ingest/import_manifest.json
cargo run --locked --bin coffer -- search /tmp/notes.coffer hello --collection notes
cargo run --locked --bin coffer -- search /tmp/notes.coffer --collection notes \
  --embedding examples/data/ingest/query.json
cargo run --locked --bin coffer -- why-not /tmp/notes.coffer --collection notes \
  --query hello --chunk-external-id 'file://note.txt#chunk-0001'
```

To regenerate text artifacts without a provider, point `--emit-jsonl` at an
empty directory. The database path is not opened:

```console
cargo run --locked --bin coffer -- import-dir examples/data/ingest/source /tmp/missing.coffer \
  --collection notes --chunk-size 64 --chunk-overlap 1 \
  --emit-jsonl /tmp/notes-text
```

That manifest has no embeddings. The snapshot manifest does, because it was
produced with `--embed`.

To call a local compatible server yourself, put any non-blank value in the
variable you name. This example does not use a hosted key:

```console
export COFFER_EXAMPLE_KEY=local-only
cargo run --locked --bin coffer -- embed-jsonl examples/data/ingest/chunks.jsonl \
  --provider openai-compatible --model example --dimensions 2 \
  --out /tmp/embeddings.jsonl \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key-env COFFER_EXAMPLE_KEY
```

`import-dir --embed` uses the same provider flags after the collection exists.
The server must accept `POST /v1/embeddings` when the base ends in `/v1`.
