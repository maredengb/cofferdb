# Import a folder of text with the CofferDB 0.2.0 CLI

Use this workflow to make a directory of notes, Markdown or source text
searchable. For keyword search you need only the `coffer` executable and your
files. For vector search you also supply embeddings, either precomputed or
obtained from a provider you select. See the [CLI manual](cli.md) for installation.

## From a folder to your first search

Run the following in a POSIX shell. It creates a fresh directory and one input
file. To use your own notes, substitute their directory for `"$work/input"` in
the import command, keeping the output and database outside the input tree.

```sh
work="$(mktemp -d)"
mkdir "$work/input"
printf '%s\n' 'Jade leaves grow by the river.' > "$work/input/field.txt"
coffer import-dir "$work/input" "$work/notes.coffer" --collection notes \
  --chunk-size 256 --chunk-overlap 16 --emit-jsonl "$work/text" --json
coffer init "$work/notes.coffer"
coffer create-collection "$work/notes.coffer" notes --dim 2 --metric cosine
coffer import "$work/notes.coffer" --manifest "$work/text/import_manifest.json"
coffer search "$work/notes.coffer" jade --collection notes --json
```

The first command using `coffer` reports one document, one chunk and zero
embeddings. At that point only the text artifacts exist. The next three
commands create the database and collection, then import the saved manifest.
The final search returns the text from `field.txt`. The dimension is a required
collection setting even when storing text alone.

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
Symlinks are not followed. A symlink used as the input directory fails. With default UTF-8 input, invalid UTF-8 files are skipped. Whitespace-only text is skipped in every encoding. Whitespace-only chunks are omitted; retained chunks preserve their byte spans in the decoded UTF-8 text. A run with no eligible file, or with no chunk,
fails and does not leave a manifest that claims success.

## Read other text encodings

Specify one encoding for all selected files in a run. For example, to prepare Japanese text offline:

```sh
coffer import-dir ./notes ./notes.coffer --collection notes --chunk-size 1024 \
  --chunk-overlap 64 --emit-jsonl ./artifacts --input-encoding shift_jis
```

Accepted names are `utf-8`, `utf-8-sig`, `utf-16le`, `utf-16be`, `gb18030`,
`shift_jis` and `windows-1252`. A matching Unicode BOM is optional and is
removed once. A conflicting UTF-8, UTF-16LE or UTF-16BE BOM skips that Unicode input. Legacy encodings
use encoding_rs mappings. Use separate input directories or filters for a mixed
collection of encodings. The CLI cannot identify a wrong declaration if the
bytes are valid in the selected encoding.

Add `--json` and read `result.skipped_files` in the JSON summary for the path, original byte
length and declared encoding. Invalid input is skipped with
`invalid_input_encoding`. With the default UTF-8 reader the reason is
`non_utf8_text`. If every input is skipped the command fails with
`no_eligible_inputs`. The offline command leaves the database path untouched.

Chunk size, overlap and source offsets count UTF-8 bytes in the decoded text
after BOM removal. JSONL and database text remain UTF-8. With `--embed`, only
the decoded chunks selected for import are sent to the configured provider.

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
both required. It reads the named variable and has no implicit fallback to
`OPENAI_API_KEY`.

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

A generation batch is one HTTP request. After a failure is observed, workers
stop taking new batches. Requests already in flight may finish. The database is not created or modified. An existing
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

## Try a vector search without a service

The matching source tree includes a small ready-to-import dataset in
`examples/data/ingest`. Its one chunk contains `hello world` and already has
an example vector. From the product root, run:

```sh
work="$(mktemp -d)"
coffer init "$work/notes.coffer"
coffer create-collection "$work/notes.coffer" notes --dim 2 --metric cosine --embedding-model example
coffer import "$work/notes.coffer" --manifest examples/data/ingest/import_manifest.json
coffer vector-search "$work/notes.coffer" --collection notes --embedding examples/data/ingest/query.json
```

The result identifies `file://note.txt#chunk-0001`. For your own chunks, an
explicit provider invocation such as the following writes vectors to a new
directory. It requires a running compatible server at the supplied address.

```sh
vectors="$(mktemp -d)"
export COFFER_EXAMPLE_KEY=local-only
coffer embed-jsonl examples/data/ingest/chunks.jsonl --provider openai-compatible \
  --model example --dimensions 2 --out "$vectors/embeddings.jsonl" \
  --base-url http://127.0.0.1:8080/v1 --api-key-env COFFER_EXAMPLE_KEY
```
