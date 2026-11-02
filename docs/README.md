# CofferDB programming documentation

Documentation follows the version of the library or installed Python package.
The 0.1.1 material in this source tree describes the experimental implementation.
Use the [release record](https://github.com/maredengb/coffered/releases) for
available artifacts, checksums and validation results for specific environments.

| Version | Guide | Executable example | Reference |
| --- | --- | --- | --- |
| Rust 0.1.0 | [Supplemental programming guide](rust-v0.1.0.md) | [Common Rust lifecycle](../examples/programming_guide.rs), also verified against the published 0.1.0 source | 0.1.0 rustdoc in the matching offline reference archive |
| Rust 0.1.1 | [Programming guide](rust-v0.1.1.md) | [Common lifecycle](../examples/programming_guide.rs) and [source-backed retrieval](../examples/real_corpus_demo.rs) | Generated rustdoc in the matching offline reference archive |
| Python 0.1.1 | [Programming guide](python-v0.1.1.md) and [installation/behavior details](../bindings/python/README.md) | [Offline code retrieval](../examples/code_retrieval.py) and [direct batch](../bindings/python/examples/quickstart.py) | Installed-package Sphinx reference in the matching offline archive |

The [reference build guide](reference.md) describes the sources, symbol indexes,
version checks and offline archive layout. Generated reference artifacts are
separate from the source files. Use the archive listed for the matching release
and keep it with its wheel and checksums. Do not substitute a mutable `main` checkout for a released API.

The Rust 0.1.0 guide is supplemental documentation delivered with this source
tree. It does not change that release's tag, package or platform declaration.
The shared executable uses APIs that exist in both Rust versions; the 0.1.1
joined full-content read is documented separately.

## Concepts and operating boundaries

- [Architecture](architecture.md), [file format and recovery](file-format.md).
- [Ranking and explain](explain.md), [benchmark guide](../benchmarks/README.md).
- [Backups, logical restore, privacy and deletion](backup.md).
- [0.1.0 compatibility notes](compatibility.md); the 0.1.1 guide describes the
  additive API and Python support differences.

The Python package targets regular-GIL CPython 3.12, 3.13 and 3.14 on macOS 15+
arm64 and Linux x86_64/glibc 2.28+. macOS 11–14, Intel macOS, musl, Windows,
other interpreters and free-threaded builds are outside this Python matrix.
The minimum deployment declaration and environments actually tested are
different facts; use the matching artifact's recorded results.
