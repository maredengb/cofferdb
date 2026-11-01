# Changelog

Entries describe version contents. The
[releases page](https://github.com/maredengb/coffered/releases) records published
artifacts and their validation results.

## 0.1.1

- Add a synchronous in-process Python API using the same Rust engine for
  collection lifecycle, finite atomic mixed batches, manifest import/export,
  keyword/vector/hybrid search, explain, deletion and statistics.
- Return complete owned chunk content with its document from one read view;
  keep the existing Rust read methods and file-format/canonical JSON contracts.
- Preserve native error fields, confirmed write outcomes, resource refusals,
  owner/close/fork rules and recoverable Python allocation failures at the host
  boundary. Native work releases the GIL; forced cancellation is not added.
- Provide versioned Rust/Python guides, typed Python results, installed-package
  reference sources and source-backed offline examples with pinned vectors.
- Name the existing Rust public return structures at the crate root so their
  fields and methods can be used and documented directly. Hidden binding and
  implementation helpers are outside the application API.
- Prepare source distributions from the complete product root, with both Cargo
  locks, upstream notices and static source identity included in built packages.

The Python target is regular-GIL CPython 3.12/3.13/3.14 on macOS 15+ arm64 and
Linux x86_64/glibc 2.28+. macOS 11–14 is unsupported by this Python package;
this replaces its earlier development target and does not amend the old
Rust/CLI platform declaration. Final platform qualification, exact artifact
identity and publication are separate from this source changelog.
This version introduces no fixed application recall target or default-weight
change. External vectors and user-selected weights retain their documented
effect on application relevance.

## 0.1.0

Initial experimental Rust library and CLI for local embedded retrieval.

- Store documents, chunks, metadata and externally generated vectors in a single database file.
- Search with BM25, exact cosine-vector scoring and weighted hybrid fusion, with metadata filters and explain output.
- Select expanded, balanced or constrained resource defaults; inspect effective settings and measured query work.
- Import and export JSONL data with explicit model and tokenizer configuration.
- Check stored data and indexes, rebuild indexes, compact files and retain existing reader snapshots across supported operations.
- Use self-contained Markdown, Rust-code and explain/troubleshooting examples.
- Reproduce representative retrieval observations and inspect historical costs, failures and regressions through the benchmark materials.

### Compatibility and upgrades

Compared with earlier experimental snapshots, 0.1.0 makes JSON limits and
conditional error diagnostics explicit, distinguishes controlled resource refusal,
and permits a nullable ordinary `suggested_action`. Compaction counters describe
committed business records; retained-file writers, embedding-model consistency,
export configuration and complete candidate scoring also have explicit contracts.
Clients and artifact tools must update strict enums, snapshots and restore logic,
including the supported preset names and safe-integer `top_k` range.

Read the version-specific [compatibility and upgrade notes](https://github.com/maredengb/coffered/blob/v0.1.0/docs/compatibility.md)
before moving data or reusing earlier configurations and benchmark results.
Keep original files and reports; these changes do not imply automatic repair,
data migration or universal relevance improvements.

External embeddings and application relevance judgments affect retrieval quality.
No universal Recall/MRR guarantee or hybrid superiority is claimed. Presets set
resource defaults rather than quality rankings. The file-format, backup and
security documentation describe compatibility and data-handling limitations.

This entry describes the 0.1.0 feature set. See the
[releases page](https://github.com/maredengb/coffered/releases) for publication
status and downloadable artifacts. Platform binaries and their checksums are
separate release artifacts; experimental status and the limitations above remain
applicable after publication.
