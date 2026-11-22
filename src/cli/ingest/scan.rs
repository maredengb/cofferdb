//! Directory selection and the frozen UTF-8 byte chunk loop.
//!
//! Default ignored directories are not visited and cannot be brought back with
//! `--include`. Symlinks are never followed. Chunk bounds are codepoint
//! boundaries in the BOM-stripped text. Line numbers count only `\n`.

use std::fs;
use std::path::Path;

use crate::error::{Error, ErrorCode, Result};
use crate::json::JsonValue;

use super::boundary::{cli, io_ingest};
use super::encoding::InputEncoding;

// This directory set is pruned at traversal time. A later include match cannot
// recover a child of one of these directories because its entries were never
// visited.
const IGNORED_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor"];
// Known binary suffixes provide a fast file classification before trying a text
// decoder. A file without one of these suffixes still has to satisfy the scan's
// text checks.
const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "pdf", "zip", "gz", "tgz", "bz2", "7z", "wasm", "bin",
    "exe", "dll", "so", "dylib", "a", "o", "class", "jar", "pyc", "woff", "woff2", "ico", "mp3",
    "mp4", "mov", "avi", "sqlite", "db", "coffer",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    // The recursive wildcard consumes complete path components and can also
    // consume none. All other patterns run against the bytes of just one
    // component.
    Globstar,
    Pattern,
}

struct Segment {
    // Distinguishes a recursive globstar from a pattern matching one path
    // component.
    kind: SegmentKind,
    // Component pattern used by the ordinary wildcard matcher.
    pattern: String,
}

// An unanchored basename such as *.md is matched against the final component. A
// directory-only rule such as build/ is checked against ancestor components when
// deciding whether a file matches.
pub(super) struct Glob {
    // Match the entire root-relative component sequence when set.
    anchored: bool,
    // Apply the pattern to directories, including ancestors of a file being
    // considered.
    directory_only: bool,
    // Parsed slash-separated components used by the path matcher.
    segments: Vec<Segment>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
// Format selects which source files are eligible and is echoed into generated
// provenance. It is independent of the encoding used to turn raw bytes into UTF-8
// text.
pub(super) enum TextFormat {
    Text,
    Markdown,
}

impl TextFormat {
    // Validate the named format before discovery. An unsupported selector is a CLI
    // argument error; it does not become a reason attached to an individually
    // skipped file.
    pub(super) fn parse(value: &str) -> Result<Self> {
        match value {
            "text" => Ok(Self::Text),
            "markdown" => Ok(Self::Markdown),
            _ => Err(cli(
                "--format must be text or markdown",
                Some("--format"),
                Some("invalid_format"),
            )),
        }
    }

    // These spellings contribute to the dataset identity as well as displayed
    // metadata. Keep parsing and emission in step when a recognized format
    // changes.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Markdown => "markdown",
        }
    }
}

// A chunk has an identity within its document and two coordinate systems for its
// span: a half-open UTF-8 byte range and one-based source line numbers. Offsets
// can therefore differ from raw file positions after decoding.
pub(super) struct ChunkRecord {
    // Root-relative source path kept with the chunk for provenance.
    pub relative: String,
    // Owning document's external ID derived from the file's relative path.
    pub external_id: String,
    // Document ID plus the chunk ordinal, used to join embeddings to this chunk.
    pub chunk_external_id: String,
    // One-based position among emitted, non-whitespace chunks of this document.
    pub ordinal: u64,
    // Slice of the decoded file text carried into the chunks artifact.
    pub text: String,
    // Inclusive UTF-8 byte offset in the decoded source text.
    pub start: u64,
    // Exclusive UTF-8 byte offset immediately after the last included byte.
    pub end: u64,
    // One-based source line containing the first included byte.
    pub line_start: u32,
    // One-based source line containing the last included byte, even when the
    // exclusive end is on the next line.
    pub line_end: u32,
}

// Keep eligible sources separate from skipped diagnostics. Only eligible text
// participates in chunking and identification, while the skip report can also
// describe entries never opened for reading.
#[derive(Debug)]
pub(super) struct Scan {
    // Eligible decoded sources sorted by relative path bytes before chunking.
    pub files: Vec<SourceFile>,
    // Skipped entries in path order for reproducible diagnostics.
    pub skipped: Vec<SkippedFile>,
}

#[derive(Debug)]
pub(super) struct SourceFile {
    // Path relative to the scan root, used to derive document identity.
    pub relative: String,
    // Decoded text in UTF-8, used as the coordinate system for chunk spans.
    pub text: String,
}

#[derive(Debug)]
pub(super) struct SkippedFile {
    // Root-relative name of the skipped entry as reported by the scan.
    pub path: String,
    // Stable reason code used to classify this skipped path.
    pub reason: &'static str,
    // Source byte count, when the scan has file metadata for the entry.
    pub bytes: Option<u64>,
    // Human-readable explanation accompanying the reason code.
    pub message: &'static str,
}

impl SkippedFile {
    // Use a stable shape for skip diagnostics even when the entry was excluded
    // without reading its contents. Reader code can distinguish an unknown file
    // size from a known zero-byte file.
    pub(super) fn to_json(&self, encoding: Option<InputEncoding>) -> JsonValue {
        crate::json::object_from_pairs([
            ("path", JsonValue::String(self.path.clone())),
            ("reason", JsonValue::String(self.reason.into())),
            // Encoding is chosen by the caller or the default UTF-8 path. The
            // scanner does not infer a codec to report as detected, even when a
            // decoder recognized a matching byte-order mark.
            ("detected_encoding", JsonValue::Null),
            (
                "declared_encoding",
                // Preserve the exact supported CLI label (for example utf-8-sig)
                // when one was supplied, independently of the underlying decoder
                // it maps to.
                encoding.map_or(JsonValue::Null, |e| JsonValue::String(e.label.into())),
            ),
            (
                "bytes",
                // Use the metadata's source byte count when known. A skip can
                // happen before reading any file, and a decoded UTF-8 length would
                // describe a different quantity.
                match self.bytes {
                    Some(bytes) => JsonValue::Number(bytes as f64),
                    None => JsonValue::Null,
                },
            ),
            ("action", JsonValue::String("skipped".into())),
            ("message", JsonValue::String(self.message.into())),
        ])
    }
}

// Compile the complete option list before walking the input directory. Return the
// flag name from the first invalid pattern so --include and --exclude diagnostics
// stay attributable.
pub(super) fn compile_globs(patterns: &[String], field: &str) -> Result<Vec<Glob>> {
    patterns
        .iter()
        .map(|pattern| compile_glob(pattern, field))
        .collect()
}

fn compile_glob(pattern: &str, field: &str) -> Result<Glob> {
    // This syntax is intentionally narrower than shell or ignore-file matching.
    // Reject unsupported markers up front so a pattern cannot silently acquire
    // bracket, escape or negation behavior.
    if pattern.is_empty()
        || pattern.contains('!')
        || pattern.contains('[')
        || pattern.contains(']')
        || pattern.contains('\\')
    {
        return Err(cli(
            "include and exclude patterns are a closed glob subset",
            Some(field),
            Some("invalid_glob"),
        ));
    }
    let mut directory_only = false;
    let mut body = pattern;
    // A trailing slash changes the kind of entry matched. Record it as a directory
    // rule and strip it before splitting components so the trailing separator does
    // not become an empty component.
    if let Some(stripped) = body.strip_suffix('/') {
        directory_only = true;
        body = stripped;
        // Reject the root slash on its own. Removing a directory suffix must leave
        // an actual pattern for the matcher to compile.
        if body.is_empty() {
            return Err(cli(
                "include and exclude patterns are a closed glob subset",
                Some(field),
                Some("invalid_glob"),
            ));
        }
    }
    let mut anchored = false;
    // A leading slash anchors inside the input root. Remove the marker without
    // introducing a filesystem root into the normalized relative paths.
    if let Some(stripped) = body.strip_prefix('/') {
        anchored = true;
        body = stripped;
        if body.is_empty() {
            return Err(cli(
                "include and exclude patterns are a closed glob subset",
                Some(field),
                Some("invalid_glob"),
            ));
        }
    }
    // Slash-separated and recursive expressions match against root-relative
    // components. Only an ordinary single-component pattern remains an unanchored
    // basename match.
    if body.contains('/') || body.contains("**") {
        anchored = true;
    }
    let mut segments = Vec::new();
    // Use slash delimiters from the root-relative path representation on every
    // platform. The pattern does not use each operating system's native path
    // separator.
    for segment in body.split('/') {
        // An empty component inside the pattern is invalid. Collapsing repeated
        // separators would change which user-supplied rule is accepted.
        if segment.is_empty() {
            return Err(cli(
                "include and exclude patterns are a closed glob subset",
                Some(field),
                Some("invalid_glob"),
            ));
        }
        // Make a full ** component distinct from an ordinary component. The
        // matcher can then consume zero or more path components without giving
        // single-star patterns permission to cross a slash.
        if segment == "**" {
            segments.push(Segment {
                kind: SegmentKind::Globstar,
                pattern: segment.to_owned(),
            });
        // An embedded double star has no interpretation in this grammar. Reject it
        // here rather than treating the two characters as independent single-star
        // wildcards.
        } else if segment.contains("**") {
            return Err(cli(
                "include and exclude patterns are a closed glob subset",
                Some(field),
                Some("invalid_glob"),
            ));
        } else {
            // Store the component's original UTF-8 bytes for ordinary matching.
            // The runtime matcher handles * and ? inside this component after the
            // compiler has ruled out a misplaced **.
            segments.push(Segment {
                kind: SegmentKind::Pattern,
                pattern: segment.to_owned(),
            });
        }
    }
    Ok(Glob {
        anchored,
        directory_only,
        segments,
    })
}

// A bad scan root aborts the command. Individual entries have different treatment:
// selection or decoding can skip a file, while an I/O failure aborts the
// traversal.
pub(super) fn scan_dir(
    root: &Path,
    format: TextFormat,
    include: &[Glob],
    exclude: &[Glob],
    encoding: Option<InputEncoding>,
) -> Result<Scan> {
    // Inspect the directory entry itself. Following a root symlink here would
    // allow the entire source tree to come from a different location before
    // per-entry filtering even begins.
    let meta = fs::symlink_metadata(root).map_err(|_| {
        cli(
            "input directory is not a directory",
            Some("input-dir"),
            Some("not_a_directory"),
        )
    })?;
    // A recognized symlink has its own usage diagnostic, including a link whose
    // target is missing. No part of the input tree has been visited at this point.
    if meta.file_type().is_symlink() {
        return Err(cli(
            "input directory is a symbolic link",
            Some("input-dir"),
            Some("input_is_symlink"),
        ));
    }
    // A plain file can be a source entry beneath the root but cannot itself serve
    // as the root of a directory import.
    if !meta.is_dir() {
        return Err(cli(
            "input directory is not a directory",
            Some("input-dir"),
            Some("not_a_directory"),
        ));
    }
    // Keep the two result lists private to this scan until traversal succeeds. A
    // read error partway through a tree must not hand the importer a truncated
    // dataset.
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    // Start the logical path at the empty component list. Absolute and relative
    // spellings of the same root then yield the same document IDs for its
    // descendants.
    walk(
        root,
        &mut Vec::new(),
        format,
        include,
        exclude,
        encoding,
        &mut files,
        &mut skipped,
    )?;
    // Directory traversal visits one subtree at a time. Sort the full relative paths
    // again so sibling files and descendants have the same bytewise order regardless
    // of the directory layout.
    files.sort_by(|left, right| left.relative.as_bytes().cmp(right.relative.as_bytes()));
    skipped.sort_by(|left, right| left.path.as_bytes().cmp(right.path.as_bytes()));
    Ok(Scan { files, skipped })
}

// The component stack names the parent directory on entry and on successful
// return. Descend with the new component still on the stack, then remove it before
// visiting the next sibling.
fn walk(
    dir: &Path,
    rel: &mut Vec<String>,
    format: TextFormat,
    include: &[Glob],
    exclude: &[Glob],
    encoding: Option<InputEncoding>,
    files: &mut Vec<SourceFile>,
    skipped: &mut Vec<SkippedFile>,
) -> Result<()> {
    let mut entries = Vec::new();
    let read = fs::read_dir(dir).map_err(|_| io_ingest("input read failed", None))?;
    // Collect all sibling names before opening any child. Directory enumeration
    // supplies no portable order, and the first failing child is observable even
    // though successful results are sorted again.
    for entry in read {
        // An error pulling a directory entry is fatal even if other names were
        // already read. No filter decision can be made for an entry whose name is
        // unavailable.
        let entry = entry.map_err(|_| io_ingest("input read failed", None))?;
        let name = entry.file_name();
        // The relative name becomes a JSON path and contributes to the document
        // ID. Reject an unrepresentable name: lossy conversion could make two
        // distinct files acquire the same ID.
        let Some(name) = name.to_str() else {
            return Err(io_ingest("input read failed", None));
        };
        entries.push(name.to_owned());
    }
    // Sort the UTF-8 bytes directly so the visit order is independent of the
    // host's locale and of the order supplied by the filesystem.
    entries.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    for name in entries {
        let path = dir.join(&name);
        // Classify the entry using its own metadata. Consulting a symlink target
        // here could treat a directory link as another subtree, including a link
        // back to an ancestor.
        let meta = fs::symlink_metadata(&path).map_err(|_| io_ingest("input read failed", None))?;
        let file_type = meta.file_type();
        rel.push(name.clone());
        // Use slash-separated names for globs, reports and document identity. The
        // native filesystem path above is still used for metadata and reads.
        let relative = rel.join("/");
        // Record the link itself without probing whether its target is readable or
        // even exists. No size for the target belongs in this diagnostic.
        if file_type.is_symlink() {
            // A link has a skip record even if its name would have failed a glob.
            // The reason records the earliest classification that applies to the
            // entry.
            skipped.push(SkippedFile {
                path: relative,
                reason: "symlink",
                bytes: None,
                message: "symbolic link",
            });
            rel.pop();
            continue;
        }
        if file_type.is_dir() {
            // Prune a default-ignored directory before evaluating user patterns.
            // Its children never reach the include filter, and the skipped list
            // does not enumerate the hidden subtree.
            if ignored_dir(&name) {
                rel.pop();
                continue;
            }
            // A matching directory exclusion prunes traversal. Ordinary file globs
            // are tested later and never decide whether to enter a directory here.
            if directory_excluded(rel, exclude) {
                rel.pop();
                continue;
            }
            // No include-pattern match is required to descend. An unmatched parent
            // can still contain a selected descendant.
            walk(
                &path, rel, format, include, exclude, encoding, files, skipped,
            )?;
            rel.pop();
            continue;
        }
        // A nonregular entry can be reported without opening it. In particular,
        // reading a named pipe to determine its contents could block the run.
        if !file_type.is_file() {
            skipped.push(SkippedFile {
                path: relative,
                reason: "unsupported_file_type",
                bytes: None,
                // Here the entry is a special filesystem object. The same reason
                // on a binary-suffixed regular file has a size because ordinary
                // file metadata was available for classification.
                message: "unsupported file type",
            });
            rel.pop();
            continue;
        }
        // Only regular files reach the content classifier. The byte count for a
        // skipped file comes from this metadata observation rather than from a
        // possibly decoded string.
        classify_file(
            &path,
            relative,
            meta.len(),
            format,
            include,
            exclude,
            encoding,
            files,
            skipped,
        )?;
        rel.pop();
    }
    Ok(())
}

// The leading-dot rule applies to directory components. A dot-prefixed regular
// file still goes through the normal file filters.
fn ignored_dir(name: &str) -> bool {
    name.starts_with('.') || IGNORED_DIRS.contains(&name)
}

// Only patterns spelled as directory patterns can prune here. The match is
// evaluated against the complete stack up to the directory about to be entered.
fn directory_excluded(rel: &[String], exclude: &[Glob]) -> bool {
    exclude
        .iter()
        .any(|glob| glob.directory_only && directory_matches(glob, rel))
}

// Every successful classification appends exactly one record to one of the two
// lists. Failure to read a selected file propagates instead of turning that file
// into an ordinary exclusion.
fn classify_file(
    path: &Path,
    relative: String,
    bytes: u64,
    format: TextFormat,
    include: &[Glob],
    exclude: &[Glob],
    encoding: Option<InputEncoding>,
    files: &mut Vec<SourceFile>,
    skipped: &mut Vec<SkippedFile>,
) -> Result<()> {
    let name = relative.rsplit('/').next().unwrap_or(&relative);
    // The first matching exclusion supplies the reason in the ingest report.
    // Classify known binary extensions before glob matching or decoding so a path
    // with multiple reasons has a stable outcome.
    // This test uses the name alone. Admitting an unknown extension still requires
    // its bytes to decode successfully below.
    if binary_extension(name) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "unsupported_file_type",
            bytes: Some(bytes),
            message: "unsupported file type",
        });
        return Ok(());
    }
    // Match globs against the same root-relative components used during directory
    // traversal. Files in different directories can share a basename while
    // matching different anchored patterns.
    let segments: Vec<String> = relative.split('/').map(str::to_owned).collect();
    // An empty include list permits any otherwise eligible file. With explicit
    // includes, at least one has to match before exclusions and the format filter
    // are considered.
    if !include.is_empty() && !include.iter().any(|glob| file_matches(glob, &segments)) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "excluded_by_filter",
            bytes: Some(bytes),
            message: "excluded by filter",
        });
        return Ok(());
    }
    // An exclusion still wins over a matching include. Both a missed include and a
    // matched exclude use the same public filter reason.
    if exclude.iter().any(|glob| file_matches(glob, &segments)) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "excluded_by_filter",
            bytes: Some(bytes),
            message: "excluded by filter",
        });
        return Ok(());
    }
    // Markdown mode narrows the set of names after glob selection. An explicit
    // include alone cannot admit a non-Markdown extension in this mode.
    if format == TextFormat::Markdown && !markdown_extension(name) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "excluded_by_filter",
            bytes: Some(bytes),
            message: "excluded by filter",
        });
        return Ok(());
    }
    // Delay opening the file until the name and format checks pass. An unreadable
    // eligible file is an I/O error, whereas a name excluded above needs no
    // readable contents.
    let raw = fs::read(path).map_err(|_| io_ingest("input read failed", None))?;
    // All accepted sources enter the chunker as UTF-8 text. A declared encoding
    // controls conversion here, so a stored byte position later addresses this
    // string.
    let text = match encoding {
        Some(encoding) => encoding.decode(&raw),
        // Without an encoding option, accept only valid UTF-8. Strip one leading
        // BOM before computing any chunk positions, even when the file is
        // otherwise empty.
        None => std::str::from_utf8(&raw)
            .ok()
            .map(|s| s.strip_prefix('\u{feff}').unwrap_or(s).to_owned()),
    };
    let Some(text) = text else {
        // Distinguish invalid bytes under an explicit declaration from failure of
        // the default UTF-8 rule. This distinction survives in the per-file scan
        // report.
        let (reason, message) = if encoding.is_some() {
            (
                "invalid_input_encoding",
                "invalid text for declared encoding",
            )
        } else {
            ("non_utf8_text", "not valid UTF-8")
        };
        // Keep the source-byte count even when conversion fails before producing
        // any text. It can differ from the number of bytes the same file would
        // occupy as valid UTF-8.
        skipped.push(SkippedFile {
            path: relative,
            reason,
            bytes: Some(bytes),
            message,
        });
        return Ok(());
    };
    // Check whitespace only to decide whether to keep the source. Passing a
    // trimmed string downstream would change both chunk text and its source
    // offsets.
    if text.trim().is_empty() {
        // Whitespace-only input, including Unicode whitespace recognized by trim,
        // uses the same report entry as a zero-length file.
        skipped.push(SkippedFile {
            path: relative,
            reason: "empty_text",
            bytes: Some(bytes),
            message: "empty text",
        });
        return Ok(());
    }
    // Keep the original decoded whitespace and line endings. The chunker
    // calculates line numbers and byte ranges from these exact contents.
    files.push(SourceFile { relative, text });
    Ok(())
}

// Use the last suffix as a binary-type hint before attempting to decode a
// candidate. Compare ASCII case insensitively so the screen works the same for
// upper-case extensions.
fn binary_extension(name: &str) -> bool {
    let Some((stem, ext)) = name.rsplit_once('.') else {
        return false;
    };
    // Treat a bare dotfile as a name without an extension. A leading dot alone
    // does not make the rest of the name a binary type.
    if stem.is_empty() {
        return false;
    }
    BINARY_EXTENSIONS
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(ext))
}

// Recognize both supported Markdown suffixes, ignoring ASCII case. Bare dotfiles
// still lack a file stem and do not qualify.
fn markdown_extension(name: &str) -> bool {
    let Some((stem, ext)) = name.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty() && (ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown"))
}

// Match a candidate against the root-relative path components. Directory-only and
// anchored patterns use more of the path than a basename pattern.
fn file_matches(glob: &Glob, segments: &[String]) -> bool {
    if glob.directory_only {
        // For a directory pattern, omit the filename and test each ancestor
        // prefix. A matching directory can cover all its descendants.
        segments
            .split_last()
            .into_iter()
            .flat_map(|(_file, ancestors)| ancestor_prefixes(ancestors))
            .any(|prefix| directory_matches(glob, prefix))
    // An anchored pattern must consume the whole relative path. An unanchored
    // single-component pattern tests only the filename.
    } else if glob.anchored {
        match_segments(&glob.segments, segments)
    } else {
        segments
            .last()
            .is_some_and(|name| match_component(&glob.segments[0].pattern, name))
    }
}

// Generate nonempty ancestor prefixes starting at the scan root. Each prefix
// preserves the root-relative position for an anchored match.
fn ancestor_prefixes(ancestors: &[String]) -> impl Iterator<Item = &[String]> {
    (1..=ancestors.len()).map(|len| &ancestors[..len])
}

// The scan root is outside the candidate directory namespace. Check its descendant
// components when applying a directory-only pattern.
fn directory_matches(glob: &Glob, dir: &[String]) -> bool {
    if dir.is_empty() {
        return false;
    }
    if glob.anchored {
        match_segments(&glob.segments, dir)
    } else {
        // A basename directory pattern checks the final component at each
        // candidate depth. Callers walk ancestors when matching the whole path of
        // a file.
        dir.last()
            .is_some_and(|name| match_component(&glob.segments[0].pattern, name))
    }
}

// Match slash-separated patterns as whole components. Globstar can cross component
// boundaries; '*' and '?' are evaluated separately within one component.
fn match_segments(pattern: &[Segment], path: &[String]) -> bool {
    fn rec(pattern: &[Segment], path: &[String]) -> bool {
        // An exhausted pattern succeeds only if the relative path is also
        // exhausted. Extra path suffixes cannot satisfy an anchored match.
        if pattern.is_empty() {
            return path.is_empty();
        }
        // Try a zero-component globstar match first. Otherwise keep the globstar
        // active and discard one path component so it can absorb any depth.
        if pattern[0].kind == SegmentKind::Globstar {
            if rec(&pattern[1..], path) {
                return true;
            }
            return !path.is_empty() && rec(pattern, &path[1..]);
        }
        // A literal or wildcard component consumes one path component. Check both
        // are present before recursing with the shortened path.
        !path.is_empty()
            && match_component(&pattern[0].pattern, &path[0])
            && rec(&pattern[1..], &path[1..])
    }
    rec(pattern, path)
}

fn match_component(pattern: &str, text: &str) -> bool {
    match_bytes(pattern.as_bytes(), text.as_bytes())
}

// Use byte comparisons for literal text and '*' within a component. '?' consults
// the UTF-8 start byte so it consumes one character.
fn match_bytes(pattern: &[u8], text: &[u8]) -> bool {
    if pattern.is_empty() {
        return text.is_empty();
    }
    // Try the empty '*' match before letting it consume a byte. Keeping the same
    // pattern allows additional bytes without crossing the component boundary.
    if pattern[0] == b'*' {
        if match_bytes(&pattern[1..], text) {
            return true;
        }
        return !text.is_empty() && match_bytes(pattern, &text[1..]);
    }
    // A continuation byte is not the start of a character, so '?' cannot consume
    // it after a '*' branch lands there. A complete character must fit the
    // remaining slice.
    if pattern[0] == b'?' {
        let width = utf8_width(text.first().copied());
        return width > 0 && text.len() >= width && match_bytes(&pattern[1..], &text[width..]);
    }
    !text.is_empty() && pattern[0] == text[0] && match_bytes(&pattern[1..], &text[1..])
}

// Return a width only at a recognized UTF-8 leading byte. The original candidate
// is a str; this is a boundary check for the byte-oriented matcher.
fn utf8_width(first: Option<u8>) -> usize {
    match first {
        Some(byte) if byte < 0x80 => 1,
        Some(byte) if byte & 0xe0 == 0xc0 => 2,
        Some(byte) if byte & 0xf0 == 0xe0 => 3,
        Some(byte) if byte & 0xf8 == 0xf0 => 4,
        _ => 0,
    }
}

// Chunk the decoded UTF-8 text by byte limits, preserving slices on valid
// character boundaries. Store offsets in the decoded text's coordinates.
pub(super) fn chunk_files(
    files: &[SourceFile],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<ChunkRecord>> {
    let mut records = Vec::new();
    for file in files {
        // Restart offsets and the emitted ordinal for each file. The ordinal
        // advances only after a non-whitespace record is emitted.
        let mut start = 0usize;
        let mut ordinal = 1u64;
        let text = file.text.as_str();
        // Build the same document identity used in the documents artifact. The
        // chunk ID extends it with this file's emitted ordinal.
        let external_id = format!("file://{}", file.relative);
        while start < text.len() {
            // Clamp the tentative end to the available bytes and then round down
            // to a character boundary. No chunk exceeds the chosen byte cap.
            let mut end = start.saturating_add(chunk_size).min(text.len());
            end = floor_codepoint_boundary(text, end);
            // If flooring the end leaves no room for the first character,
            // advancing would violate either the cap or UTF-8. Return a usage
            // error before producing a record.
            if end <= start {
                return Err(cli(
                    "chunk size cannot hold a codepoint",
                    Some("--chunk-size"),
                    Some("codepoint_exceeds_chunk"),
                ));
            }
            // Whitespace-only ranges still advance the scan but produce no record.
            // Keep the source slice unchanged for nonempty chunks; trim is only
            // the admission test.
            if !text[start..end].trim().is_empty() {
                // Compute one-based line positions for both included ends. The end
                // offset is exclusive, so use end - 1 for the line containing the
                // last byte.
                let line_start = line_number(text, start)?;
                let line_end = line_number(text, end - 1)?;
                // Keep the full ownership and source location together. Artifacts
                // and embeddings join by chunk_external_id, while ordinal and
                // offsets describe the source slice.
                records.push(ChunkRecord {
                    relative: file.relative.clone(),
                    external_id: external_id.clone(),
                    chunk_external_id: format!("{external_id}#chunk-{ordinal:04}"),
                    ordinal,
                    text: text[start..end].to_owned(),
                    start: start as u64,
                    end: end as u64,
                    line_start,
                    line_end,
                });
                ordinal += 1;
            }
            // Stop once the chunk reaches the end of the input. Reapplying overlap
            // at EOF would produce another copy of the last suffix.
            if end == text.len() {
                break;
            }
            // Request overlap in bytes, then move to a legal character start.
            // Rounding down can make the actual overlap larger than the requested
            // number of bytes.
            let mut next = end.saturating_sub(chunk_overlap);
            next = floor_codepoint_boundary(text, next);
            // Rounding a byte overlap down to a character boundary can return to
            // the same start. Skip the overlap in that case to guarantee progress.
            if next <= start {
                next = end;
            }
            start = next;
        }
    }
    Ok(records)
}

// Find the UTF-8 boundary at or before a byte offset. Offsets at or beyond the end
// use the legal final boundary.
fn floor_codepoint_boundary(text: &str, mut index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

// Count line breaks strictly before the requested byte. The first byte of a line
// and later bytes in the same line share its one-based number.
fn line_number(text: &str, index: usize) -> Result<u32> {
    // `index` may sit inside a multibyte codepoint. Newlines are a single byte.
    let count = text.as_bytes()[..index]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count() as u64
        + 1;
    // The source-span format uses u32 line positions. Reject an out-of-range input
    // rather than narrowing it and pointing to the wrong line.
    u32::try_from(count).map_err(|_| {
        cli(
            "line number exceeds the supported range",
            Some("source_span"),
            Some("line_number_overflow"),
        )
    })
}

pub(super) fn no_eligible() -> Error {
    Error::new(ErrorCode::CliUsage, "no eligible inputs")
        .with_detail("reason", "no_eligible_inputs")
        .ingestion_boundary()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn temp_tree(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cofferdb-ingest-scan-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn newline_example_and_codepoint_boundary_follow_the_loop() {
        let files = vec![SourceFile {
            relative: "a.txt".into(),
            text: "a\nb".into(),
        }];
        let full = chunk_files(&files, 10, 1).unwrap();
        assert_eq!(full.len(), 1);
        assert_eq!(full[0].start, 0);
        assert_eq!(full[0].end, 3);
        assert_eq!(full[0].line_start, 1);
        assert_eq!(full[0].line_end, 2);
        assert_eq!(full[0].chunk_external_id, "file://a.txt#chunk-0001");

        let tail = chunk_files(&files, 1, 1).unwrap();
        let last = tail.last().unwrap();
        assert_eq!(last.start, 2);
        assert_eq!(last.end, 3);
        assert_eq!((last.line_start, last.line_end), (2, 2));

        let gaps = vec![SourceFile {
            relative: "g.txt".into(),
            text: "a    b".into(),
        }];
        let kept = chunk_files(&gaps, 2, 1).unwrap();
        assert_eq!(kept.len(), 2);
        assert_eq!((kept[1].ordinal, kept[1].start, kept[1].end), (2, 4, 6));
        assert_eq!(kept[1].text, " b");

        let wide = vec![SourceFile {
            relative: "e.txt".into(),
            text: "aé".into(),
        }];
        let split = chunk_files(&wide, 2, 1).unwrap();
        assert_eq!(split[0].text, "a");
        assert_eq!(split[1].text, "é");
        assert!(chunk_files(&wide, 1, 1).is_err());

        let gap = vec![SourceFile {
            relative: "gap.txt".into(),
            text: "hello world\n\n".into(),
        }];
        let kept_gap = chunk_files(&gap, 12, 1).unwrap();
        assert_eq!(kept_gap.len(), 1);
        assert_eq!(kept_gap[0].text, "hello world\n");
        assert_eq!(kept_gap[0].ordinal, 1);
    }

    #[test]
    fn whitespace_only_files_are_empty_text() {
        let root = temp_tree("blank-files");
        fs::write(root.join("spaces.txt"), "   ").unwrap();
        fs::write(root.join("nbsp.txt"), "\u{00a0}\n").unwrap();
        fs::write(root.join("gap.txt"), "hello").unwrap();
        let scan = scan_dir(&root, TextFormat::Text, &[], &[], None).unwrap();
        let paths: Vec<_> = scan
            .files
            .iter()
            .map(|file| file.relative.as_str())
            .collect();
        assert_eq!(paths, ["gap.txt"]);
        assert!(scan
            .skipped
            .iter()
            .any(|item| item.path == "spaces.txt" && item.reason == "empty_text"));
        assert!(scan
            .skipped
            .iter()
            .any(|item| item.path == "nbsp.txt" && item.reason == "empty_text"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn globs_ignore_hidden_dirs_binaries_and_symlinks() {
        let root = temp_tree("globs");
        fs::write(root.join("keep.txt"), "hello").unwrap();
        fs::write(root.join("note.md"), "title").unwrap();
        fs::write(root.join("pic.PNG"), [0, 1, 2, 3]).unwrap();
        fs::write(root.join(".hidden"), "secret-file").unwrap();
        fs::create_dir_all(root.join(".secret")).unwrap();
        fs::write(root.join(".secret").join("nope.txt"), "hidden-dir").unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::write(root.join("target").join("built.txt"), "build").unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(root.join("docs").join("a.md"), "docs").unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src").join("lib.rs"), "code").unwrap();
        std::os::unix::fs::symlink("keep.txt", root.join("link.txt")).unwrap();
        fs::write(root.join("bad.bintext"), [0xff, 0xfe]).unwrap();

        let include = compile_globs(&["*.md".into(), "src/".into()], "--include").unwrap();
        let exclude = compile_globs(&["docs/".into()], "--exclude").unwrap();
        let scan = scan_dir(&root, TextFormat::Text, &include, &exclude, None).unwrap();
        let paths: Vec<_> = scan
            .files
            .iter()
            .map(|file| file.relative.as_str())
            .collect();
        assert_eq!(paths, ["note.md", "src/lib.rs"]);
        assert!(scan
            .skipped
            .iter()
            .any(|item| item.path == "link.txt" && item.reason == "symlink"));
        assert!(scan
            .skipped
            .iter()
            .any(|item| item.path == "pic.PNG" && item.reason == "unsupported_file_type"));
        assert!(scan
            .skipped
            .iter()
            .all(|item| item.path != ".secret/nope.txt"));
        assert!(scan
            .skipped
            .iter()
            .all(|item| item.path != "target/built.txt"));
        assert!(scan
            .files
            .iter()
            .all(|file| !file.text.contains("hidden-dir")));
        assert!(scan
            .files
            .iter()
            .all(|file| !file.text.contains("secret-file") || file.relative == ".hidden"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn input_symlink_is_rejected_before_reading_the_target() {
        let root = temp_tree("root-link");
        let real = root.join("real");
        fs::create_dir(&real).unwrap();
        fs::write(real.join("a.txt"), "body").unwrap();
        let link = root.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let err = scan_dir(&link, TextFormat::Text, &[], &[], None).unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
        assert_eq!(
            err.detail()
                .iter()
                .find(|(key, _)| key == "reason")
                .map(|(_, value)| value.as_str()),
            Some("input_is_symlink")
        );
        let _ = fs::remove_dir_all(&root);
    }
}
