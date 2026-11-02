# Source and offline reference builds (0.1.1)

The source tree contains the engine, Python binding, locked inputs, notices,
guides and executable examples. The root `pyproject.toml` is the canonical
Python packaging entry; it points to `bindings/python/Cargo.toml`. Run maturin
from this root so the complete project layout and README paths are retained.
Published archives and their checksums are listed in the matching release record.

## Prepare source without carrying repository history

Use Python 3.12+ and Rust 1.80+ with the locked Cargo dependencies available.
The preparation tool uses offline Cargo file lists for the two crates and
verifies the binding lock against the dependency/notice inventory. It refuses
existing output directories and source links that could escape the tree.

```console
python tools/prepare_source.py /a/new/prepared-source
```

Without `--commit`, the result explicitly declares an **unversioned export**.
No enclosing repository's identity is substituted. For a clean product checkout,
`--commit FULL_PRODUCT_COMMIT` verifies that exact HEAD, repository root, clean
status and matching origin. The command only prepares local material; it does
not make the commit publicly reachable or approve publication.

`SOURCE-INPUTS.json` records relative input paths, sizes, modes and SHA-256.
The included `cofferdb/_source.json` records the source-input identity, optional
product commit/tree, actual SDK/engine versions and both lock identities.
It is static data: reading it uses `importlib.resources`, with no network call.
It describes source origin, not a claim that a user's later rebuild used the
original compiler. Exact wheel/archive checksums and the actual compiler,
target, interpreter, SDK, flags and native dependencies belong in that build's
artifact record. No archive can contain its own final checksum.

## Build the source distribution and binary

Install maturin **1.10.2** in a build environment, then use the prepared root:

```console
cd /a/new/prepared-source
maturin sdist --out /a/new/artifacts
MACOSX_DEPLOYMENT_TARGET=15.0 maturin build --release --locked -i python3.12 --out /a/new/artifacts
```

The macOS deployment variable is mandatory for macOS and must be `15.0`.
Linux builds use the declared manylinux_2_28 environment and compliance checks;
a normal build on a newer Linux host does not establish that baseline.
Portable release builds also remap the source, selected Rust sysroot and Cargo
dependency paths with Rust's `--remap-path-prefix` build flags, and audit the
resulting binary for private paths. Remapping only this source directory can
leave toolchain paths in macro-expanded diagnostics. Record the actual mappings
with the build; they do not replace native dependency or minimum-system checks.
An isolated rebuild can unpack the sdist and invoke the same PEP 517/maturin
backend there. The root and binding Cargo locks must both be present. Prepare
the build tools and Rust dependencies in advance for an offline source build;
binary users need neither.

The sdist excludes the large historical evaluation data, as do Cargo packages.
Its separate public archive is linked from the README. Source/API documentation,
examples, source snippets, matching example vectors and their provenance/notices
remain included. Keep source, artifacts, checksums and validation records together.

## Generate the human-readable reference archive

Use a separate documentation environment with the matching wheel installed and
all exact versions from `bindings/python/requirements-docs.txt`. These tools are
development dependencies, not runtime SDK dependencies. Python-Markdown renders
the guides; standard Sphinx autodoc imports the actual installed native API and
its packaged type hints. Rustdoc uses each Rust version's own source.

Obtain a clean Rust 0.1.0 checkout at the public `v0.1.0` tag to build the supplemental
old-version reference. It is not generated from newer source. From the prepared
0.1.1 root, with its wheel installed in this environment:

```console
python tools/build_reference.py /a/new/reference-output --rust-v010-source /path/to/clean-0.1.0-checkout
```

The tool rejects a mismatched installed package/source identity, changed input
files, unpinned documentation tools or the wrong legacy source. It generates:

- `cofferdb-0.1.1-reference/index.html`, linking the rendered guides, both Rust
  references, the Python reference and generated public symbol inventories.
- Rust indexes of types, functions, methods, fields, variants and return
  structures from normal public rustdoc, excluding doc-hidden implementation
  modules. Derived/trait methods are distinguished by the generated context.
- Python exports, native runtime signatures/docstrings, packaged stub method
  signatures, result fields, literal aliases and exception attributes from the
  actual installed distribution. No hand-maintained second API schema is used.
- A source/build identity manifest, per-file checksums, a `.tar.gz` archive and
  `SHA256SUMS`. Raw logs and temporary build targets stay outside the archive.

Unpack the archive and open `index.html` in a browser; no compiler or Python
environment is needed to read it. Keep the whole directory so relative source,
navigation, search and style assets remain available offline. Static local file
targets are checked; external upstream/standard-library links, JavaScript-created
links and arbitrary fragments are outside that check. The reference archive is
not proof of six-platform installation, compatibility, retrieval quality,
performance or full release acceptance.
