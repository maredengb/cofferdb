//! Directory selection and the frozen UTF-8 byte chunk loop.
//!
//! Default ignored directories are not visited and cannot be brought back with
//! `--include`. Symlinks are never followed. Chunk bounds are codepoint
//! boundaries in the BOM-stripped text. Line numbers count only `\n`.

use std::fs::{self, FileType};
use std::path::Path;

use crate::error::{Error, ErrorCode, Result};
use crate::json::JsonValue;

use super::boundary::{cli, io_ingest};

const IGNORED_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor"];
const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "pdf", "zip", "gz", "tgz", "bz2", "7z", "wasm", "bin",
    "exe", "dll", "so", "dylib", "a", "o", "class", "jar", "pyc", "woff", "woff2", "ico", "mp3",
    "mp4", "mov", "avi", "sqlite", "db", "coffer",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    Globstar,
    Pattern,
}

struct Segment {
    kind: SegmentKind,
    pattern: String,
}

pub(super) struct Glob {
    anchored: bool,
    directory_only: bool,
    segments: Vec<Segment>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TextFormat {
    Text,
    Markdown,
}

impl TextFormat {
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

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Markdown => "markdown",
        }
    }
}

pub(super) struct ChunkRecord {
    pub relative: String,
    pub external_id: String,
    pub chunk_external_id: String,
    pub ordinal: u64,
    pub text: String,
    pub start: u64,
    pub end: u64,
    pub line_start: u32,
    pub line_end: u32,
}

#[derive(Debug)]
pub(super) struct Scan {
    pub files: Vec<SourceFile>,
    pub skipped: Vec<SkippedFile>,
}

#[derive(Debug)]
pub(super) struct SourceFile {
    pub relative: String,
    pub text: String,
}

#[derive(Debug)]
pub(super) struct SkippedFile {
    pub path: String,
    pub reason: &'static str,
    pub bytes: Option<u64>,
    pub message: &'static str,
}

impl SkippedFile {
    pub(super) fn to_json(&self) -> JsonValue {
        crate::json::object_from_pairs([
            ("path", JsonValue::String(self.path.clone())),
            ("reason", JsonValue::String(self.reason.into())),
            ("detected_encoding", JsonValue::Null),
            ("declared_encoding", JsonValue::Null),
            (
                "bytes",
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

pub(super) fn compile_globs(patterns: &[String], field: &str) -> Result<Vec<Glob>> {
    patterns
        .iter()
        .map(|pattern| compile_glob(pattern, field))
        .collect()
}

fn compile_glob(pattern: &str, field: &str) -> Result<Glob> {
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
    if let Some(stripped) = body.strip_suffix('/') {
        directory_only = true;
        body = stripped;
        if body.is_empty() {
            return Err(cli(
                "include and exclude patterns are a closed glob subset",
                Some(field),
                Some("invalid_glob"),
            ));
        }
    }
    let mut anchored = false;
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
    if body.contains('/') || body.contains("**") {
        anchored = true;
    }
    let mut segments = Vec::new();
    for segment in body.split('/') {
        if segment.is_empty() {
            return Err(cli(
                "include and exclude patterns are a closed glob subset",
                Some(field),
                Some("invalid_glob"),
            ));
        }
        if segment == "**" {
            segments.push(Segment {
                kind: SegmentKind::Globstar,
                pattern: segment.to_owned(),
            });
        } else if segment.contains("**") {
            return Err(cli(
                "include and exclude patterns are a closed glob subset",
                Some(field),
                Some("invalid_glob"),
            ));
        } else {
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

pub(super) fn scan_dir(
    root: &Path,
    format: TextFormat,
    include: &[Glob],
    exclude: &[Glob],
) -> Result<Scan> {
    let meta = fs::symlink_metadata(root).map_err(|_| {
        cli(
            "input directory is not a directory",
            Some("input-dir"),
            Some("not_a_directory"),
        )
    })?;
    if meta.file_type().is_symlink() {
        return Err(cli(
            "input directory is a symbolic link",
            Some("input-dir"),
            Some("input_is_symlink"),
        ));
    }
    if !meta.is_dir() {
        return Err(cli(
            "input directory is not a directory",
            Some("input-dir"),
            Some("not_a_directory"),
        ));
    }
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    walk(
        root,
        &mut Vec::new(),
        format,
        include,
        exclude,
        &mut files,
        &mut skipped,
    )?;
    files.sort_by(|left, right| left.relative.as_bytes().cmp(right.relative.as_bytes()));
    skipped.sort_by(|left, right| left.path.as_bytes().cmp(right.path.as_bytes()));
    Ok(Scan { files, skipped })
}

fn walk(
    dir: &Path,
    rel: &mut Vec<String>,
    format: TextFormat,
    include: &[Glob],
    exclude: &[Glob],
    files: &mut Vec<SourceFile>,
    skipped: &mut Vec<SkippedFile>,
) -> Result<()> {
    let mut entries = Vec::new();
    let read = fs::read_dir(dir).map_err(|_| io_ingest("input read failed", None))?;
    for entry in read {
        let entry = entry.map_err(|_| io_ingest("input read failed", None))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return Err(io_ingest("input read failed", None));
        };
        entries.push(name.to_owned());
    }
    entries.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    for name in entries {
        let path = dir.join(&name);
        let meta = fs::symlink_metadata(&path).map_err(|_| io_ingest("input read failed", None))?;
        let file_type = meta.file_type();
        rel.push(name.clone());
        let relative = rel.join("/");
        if file_type.is_symlink() {
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
            if ignored_dir(&name) {
                rel.pop();
                continue;
            }
            if directory_excluded(rel, exclude) {
                rel.pop();
                continue;
            }
            walk(&path, rel, format, include, exclude, files, skipped)?;
            rel.pop();
            continue;
        }
        if !file_type.is_file() {
            skipped.push(SkippedFile {
                path: relative,
                reason: "unsupported_file_type",
                bytes: None,
                message: "unsupported file type",
            });
            rel.pop();
            continue;
        }
        classify_file(
            &path,
            relative,
            meta.len(),
            file_type,
            format,
            include,
            exclude,
            files,
            skipped,
        )?;
        rel.pop();
    }
    Ok(())
}

fn ignored_dir(name: &str) -> bool {
    name.starts_with('.') || IGNORED_DIRS.contains(&name)
}

fn directory_excluded(rel: &[String], exclude: &[Glob]) -> bool {
    exclude
        .iter()
        .any(|glob| glob.directory_only && directory_matches(glob, rel))
}

fn classify_file(
    path: &Path,
    relative: String,
    bytes: u64,
    _file_type: FileType,
    format: TextFormat,
    include: &[Glob],
    exclude: &[Glob],
    files: &mut Vec<SourceFile>,
    skipped: &mut Vec<SkippedFile>,
) -> Result<()> {
    let name = relative.rsplit('/').next().unwrap_or(&relative);
    if binary_extension(name) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "unsupported_file_type",
            bytes: Some(bytes),
            message: "unsupported file type",
        });
        return Ok(());
    }
    let segments: Vec<String> = relative.split('/').map(str::to_owned).collect();
    if !include.is_empty() && !include.iter().any(|glob| file_matches(glob, &segments)) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "excluded_by_filter",
            bytes: Some(bytes),
            message: "excluded by filter",
        });
        return Ok(());
    }
    if exclude.iter().any(|glob| file_matches(glob, &segments)) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "excluded_by_filter",
            bytes: Some(bytes),
            message: "excluded by filter",
        });
        return Ok(());
    }
    if format == TextFormat::Markdown && !markdown_extension(name) {
        skipped.push(SkippedFile {
            path: relative,
            reason: "excluded_by_filter",
            bytes: Some(bytes),
            message: "excluded by filter",
        });
        return Ok(());
    }
    let raw = fs::read(path).map_err(|_| io_ingest("input read failed", None))?;
    let (text, stripped_empty) = match std::str::from_utf8(&raw) {
        Ok(text) => {
            let text = text.strip_prefix('\u{feff}').unwrap_or(text);
            (text, text.trim().is_empty())
        }
        Err(_) => {
            skipped.push(SkippedFile {
                path: relative,
                reason: "non_utf8_text",
                bytes: Some(bytes),
                message: "not valid UTF-8",
            });
            return Ok(());
        }
    };
    if stripped_empty {
        skipped.push(SkippedFile {
            path: relative,
            reason: "empty_text",
            bytes: Some(bytes),
            message: "empty text",
        });
        return Ok(());
    }
    files.push(SourceFile {
        relative,
        text: text.to_owned(),
    });
    Ok(())
}

fn binary_extension(name: &str) -> bool {
    let Some((stem, ext)) = name.rsplit_once('.') else {
        return false;
    };
    if stem.is_empty() {
        return false;
    }
    BINARY_EXTENSIONS
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(ext))
}

fn markdown_extension(name: &str) -> bool {
    let Some((stem, ext)) = name.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty() && (ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown"))
}

fn file_matches(glob: &Glob, segments: &[String]) -> bool {
    if glob.directory_only {
        segments
            .split_last()
            .into_iter()
            .flat_map(|(_file, ancestors)| ancestor_prefixes(ancestors))
            .any(|prefix| directory_matches(glob, prefix))
    } else if glob.anchored {
        match_segments(&glob.segments, segments)
    } else {
        segments
            .last()
            .is_some_and(|name| match_component(&glob.segments[0].pattern, name))
    }
}

fn ancestor_prefixes(ancestors: &[String]) -> impl Iterator<Item = &[String]> {
    (1..=ancestors.len()).map(|len| &ancestors[..len])
}

fn directory_matches(glob: &Glob, dir: &[String]) -> bool {
    if dir.is_empty() {
        return false;
    }
    if glob.anchored {
        match_segments(&glob.segments, dir)
    } else {
        dir.last()
            .is_some_and(|name| match_component(&glob.segments[0].pattern, name))
    }
}

fn match_segments(pattern: &[Segment], path: &[String]) -> bool {
    fn rec(pattern: &[Segment], path: &[String]) -> bool {
        if pattern.is_empty() {
            return path.is_empty();
        }
        if pattern[0].kind == SegmentKind::Globstar {
            if rec(&pattern[1..], path) {
                return true;
            }
            return !path.is_empty() && rec(pattern, &path[1..]);
        }
        !path.is_empty()
            && match_component(&pattern[0].pattern, &path[0])
            && rec(&pattern[1..], &path[1..])
    }
    rec(pattern, path)
}

fn match_component(pattern: &str, text: &str) -> bool {
    match_bytes(pattern.as_bytes(), text.as_bytes())
}

fn match_bytes(pattern: &[u8], text: &[u8]) -> bool {
    if pattern.is_empty() {
        return text.is_empty();
    }
    if pattern[0] == b'*' {
        if match_bytes(&pattern[1..], text) {
            return true;
        }
        return !text.is_empty() && match_bytes(pattern, &text[1..]);
    }
    if pattern[0] == b'?' {
        let width = utf8_width(text.first().copied());
        return width > 0 && text.len() >= width && match_bytes(&pattern[1..], &text[width..]);
    }
    !text.is_empty() && pattern[0] == text[0] && match_bytes(&pattern[1..], &text[1..])
}

fn utf8_width(first: Option<u8>) -> usize {
    match first {
        Some(byte) if byte < 0x80 => 1,
        Some(byte) if byte & 0xe0 == 0xc0 => 2,
        Some(byte) if byte & 0xf0 == 0xe0 => 3,
        Some(byte) if byte & 0xf8 == 0xf0 => 4,
        _ => 0,
    }
}

pub(super) fn chunk_files(
    files: &[SourceFile],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<ChunkRecord>> {
    let mut records = Vec::new();
    for file in files {
        let mut start = 0usize;
        let mut ordinal = 1u64;
        let text = file.text.as_str();
        let external_id = format!("file://{}", file.relative);
        while start < text.len() {
            let mut end = start.saturating_add(chunk_size).min(text.len());
            end = floor_codepoint_boundary(text, end);
            if end <= start {
                return Err(cli(
                    "chunk size cannot hold a codepoint",
                    Some("--chunk-size"),
                    Some("codepoint_exceeds_chunk"),
                ));
            }
            if !text[start..end].trim().is_empty() {
                let line_start = line_number(text, start)?;
                let line_end = line_number(text, end - 1)?;
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
            if end == text.len() {
                break;
            }
            let mut next = end.saturating_sub(chunk_overlap);
            next = floor_codepoint_boundary(text, next);
            if next <= start {
                next = end;
            }
            start = next;
        }
    }
    Ok(records)
}

fn floor_codepoint_boundary(text: &str, mut index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn line_number(text: &str, index: usize) -> Result<u32> {
    // `index` may sit inside a multibyte codepoint. Newlines are a single byte.
    let count = text.as_bytes()[..index]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count() as u64
        + 1;
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
        let scan = scan_dir(&root, TextFormat::Text, &[], &[]).unwrap();
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
        let scan = scan_dir(&root, TextFormat::Text, &include, &exclude).unwrap();
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
        let err = scan_dir(&link, TextFormat::Text, &[], &[]).unwrap_err();
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
