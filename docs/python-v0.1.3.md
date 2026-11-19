# Python 0.1.3 programming guide

Use regular-GIL CPython 3.12, 3.13 or 3.14 on macOS 15 or later on arm64,
or Linux x86_64 with glibc 2.28 or later. Install the matching wheel in a clean
virtual environment.

```console
python -m pip install --only-binary=:all: cofferdb==0.1.3
python -c "import cofferdb; print(cofferdb.__version__, cofferdb.engine_version)"
```

Both version values should be `0.1.3`. For offline installation, download the
matching wheel and checksum list from the [0.1.3 release](https://github.com/maredengb/cofferdb/releases/tag/v0.1.3),
verify the wheel against that list and install the local file with
`python -m pip install --no-index --no-deps /path/to/the-matching-wheel.whl`.

The [Python quickstart](../README.md#rust-and-python-quickstarts) covers
collection setup, writes and search. The [0.1.2 programming guide](python-v0.1.2.md)
also applies to these operations and explains runtime, typing, error and
compatibility boundaries.

## Read a restored database

Exit the application's writable `Database` context before creating the snapshot
with the separate `coffer` CLI. Copy the snapshot to a fresh destination and
open that copy with the matching Python engine. For the Markdown sample:

```python
from cofferdb import Database

with Database.open("restored.coffer", read_only=True) as db:
    notes = db.collection("notes")
    hit = notes.search("write ahead", top_k=1)["results"][0]
    chunk = notes.get_chunk(hit["chunk_id"])["chunk"]
    assert chunk["chunk_external_id"] == "wal#commit"
```

This example uses an existing read-only handle after opening the copy. See the
[backup guide](backup.md) for the snapshot creation command, CLI integrity
check and conditions for making a physical copy.
