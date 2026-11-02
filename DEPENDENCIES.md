# Dependency And License Summary

The core crate declares no third-party Cargo dependencies. Core source follows the
existing repository Apache-2.0 LICENSE; Cargo metadata is synchronized to it.
The packaged LICENSE is byte-identical to the repository license text.

Compiled executables also use the Rust standard library and platform runtime.
The Rust project uses MIT/Apache-2.0 terms; binary release preparation must include
the applicable notices for the selected toolchain/system redistribution policy.
This file is a source dependency summary, not completed binary license clearance.

SQLite and sqlite-vec are externally selected optional benchmark dependencies,
not bundled core runtime crates or redistributed extensions. Record actual
versions, provenance and license documents in any benchmark/release that bundles
them. Never auto-load an extension named by imported data.

## Python binding and documentation

The separate `bindings/python` crate selects PyO3 0.27.2 and its locked Cargo
dependency graph. Its [source license inventory](bindings/python/licenses/inventory.json)
identifies the binding lock and retains upstream notice hashes. The wheel
statically includes the engine and binding; it declares no third-party Python
runtime requirements. Exact platform/toolchain binary notices remain part of
artifact-specific review.

Documentation tools are separately pinned in
`bindings/python/requirements-docs.txt`, including Sphinx 8.2.3 and
Python-Markdown 3.11.1 (BSD-3-Clause). They are build dependencies, not runtime
SDK dependencies. The offline reference builder retains available installed
tool notices and the selected Rust toolchain's upstream copyright/license
materials alongside generated assets. A toolchain inventory can list components
beyond the assets actually bundled; it is not a change to the core license.

## Bundled example sources

`examples/data/real_corpus/source_snapshot/rust-by-example/` contains unchanged
Markdown files from rust-lang/rust-by-example at commit
`dcc14f912bf8790792a2df082eddc82ab2539229`. These source files and the extracted
Rust snippets retain their upstream **MIT OR Apache-2.0** license, rather than
being relicensed by the crate's package metadata. Both upstream notices are
bundled alongside the snapshots. The example's `provenance.json` records source
URLs, hashes and the external BGE-M3 vector preparation. No model weights, Python
runtime or embedding service are included or needed to run the example.
