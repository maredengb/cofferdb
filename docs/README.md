# CofferDB programming documentation

Documentation follows the version of the library or installed Python package.
Treat a source revision or
artifact as unreleased until it is identified in the matching release record.
Use the [release record](https://github.com/maredengb/cofferdb/releases) for
available artifacts, checksums and validation results for specific environments.

| Version | Guide | Executable example | Intended reference |
| --- | --- | --- | --- |
| Rust 0.1.0 | [Supplemental programming guide](rust-v0.1.0.md) | [Common Rust lifecycle](../examples/programming_guide.rs), also verified against the published 0.1.0 source | 0.1.0 rustdoc in the matching offline reference archive |
| Rust 0.1.1 | [Programming guide](rust-v0.1.1.md) | [Common lifecycle](../examples/programming_guide.rs) and [source-backed retrieval](../examples/real_corpus_demo.rs) | Generated rustdoc in the matching offline reference archive |
| Rust 0.1.2 | [Programming guide](rust-v0.1.2.md) and [CLI text ingestion](ingestion-v0.1.2.md) | [Common lifecycle](../examples/programming_guide.rs) | Rust API and CLI guides |
| Python 0.1.2 | [Programming guide](python-v0.1.2.md) | [Offline code retrieval](../examples/code_retrieval.py) | Check the matching release for wheel availability |
| Python 0.1.1 | [Programming guide](python-v0.1.1.md); installation and behavior details are in `bindings/python/README.md` in the complete source tree | [Offline code retrieval](../examples/code_retrieval.py); the direct batch example is in `bindings/python/examples/quickstart.py` in the complete source tree | Installed-package Sphinx reference in the matching offline archive |

The [0.1.1 reference build guide](reference.md) describes the sources, symbol indexes,
version checks and intended offline archive layout. Generated reference artifacts
are separate from the source files and are not provided by this source tree. If
a matching archive is published, keep it with its wheel and checksums. Do not
substitute a mutable `main` checkout for a released API.

See the [Rust 0.1.2 guide](rust-v0.1.2.md) for the Rust API and the
[Text ingestion guide](ingestion-v0.1.2.md) for the CLI workflow.

The Rust 0.1.0 guide is supplemental documentation delivered with this source
tree. It does not change that release's tag, package or platform declaration.
The shared executable uses APIs common to the Rust guides. The
[Rust 0.1.1 guide](rust-v0.1.1.md) documents the joined full-content read.

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
