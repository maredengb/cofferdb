# Installed-package business checks

`driver.py` checks an already installed CofferDB Python distribution. The
adjacent `../install_driver.py` adapter performs the exact offline wheel
installation and verifies installed files before calling this driver. This
module does not install software, download data or provide process isolation.

The local fixture bundle is the repository's `examples/data/real_corpus`.
The driver retains its fixed Rust-by-Example source excerpts, provenance,
licenses and BAAI/bge-m3 vectors. It derives two equivalent import paths with
three documents (including one empty document), five chunks and five vectors;
it does not regenerate or reinterpret the embeddings. The expected fixture
map can be obtained without importing the SDK:

```sh
python3 -I -S -B driver.py --fixtures /absolute/real_corpus --fixture-hashes
```

After the outer installer verifies the actual distribution, execute the
business checks through its existing adapter interface. A direct invocation
on an already verified environment is:

```sh
/absolute/venv/bin/python -I -B driver.py \
  --fixtures /absolute/real_corpus \
  --contract /absolute/contract.json \
  --contract-sha256 ACTUAL_CONTRACT_SHA256 \
  --work /absolute/new-work-directory
```

The contract binds the actual attempt, cell, versions, fixture hashes and
wheel/native/distribution/source evidence. These identities are supplied by
the enclosing installer and execution record. A matching reference in this
driver is an input assertion, not independent proof of the file contents or
execution permission. The work directory must be new; failures are retained.

Both manifest and direct batch imports check counts, document/chunk IDs,
metadata, source spans, complete text, and empty documents. Vector, text and
hybrid queries retain their results and explanation fields. The checks compare
the selected stable fields across equivalent imports, verify snippet versus
full text, export and reimport, reject closed handles, and reopen after moving
the copied source files. The driver does not implement an independent scoring
oracle, performance benchmark, lifecycle stress test or platform matrix.

Output is newline-delimited JSON with immediate flushing; stderr retains the
traceback. Exit 0 from the business branch requires its final `pass` event.
The fixture-hash branch also exits 0, but emits `fixture_hashes` and performs no
business validation. Exit 1 indicates a business or SDK failure; exit 2 indicates
invalid inputs. The outer process recorder must distinguish these branches and
retain actual termination, raw output and all generated work files.

`static_check.py` checks the driver syntax and bounded control interface. Its
optional fake suite uses an in-memory substitute and temporary files to test
successful control flow and deliberate failures without loading CofferDB:

```sh
python3 -I -S -B static_check.py --fake /absolute/real_corpus
```

The source and this suite are copied without changes from the previously
reviewed driver. Their miniature results do not establish actual installation,
platform compatibility, package behavior or whole-version acceptance. Each
real run must bind the final source and package it actually uses.
