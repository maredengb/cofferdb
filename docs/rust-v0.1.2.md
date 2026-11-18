# Rust 0.1.2 programming guide

Use Rust 1.80 or later and pin the crate to this version:

```toml
[dependencies]
cofferdb = "=0.1.2"
```

The [Rust 0.1.1 guide](rust-v0.1.1.md) covers database and collection setup,
writes, reads, and querying. See the [compatibility notes](compatibility.md)
for file format and API expectations.

Version 0.1.2 adds a CLI workflow for turning a directory of text files into
import artifacts, generating embeddings with an HTTP provider, and importing
into an existing collection. The [text ingestion guide](ingestion-v0.1.2.md)
explains the commands, provider configuration, and offline import and search
example.
