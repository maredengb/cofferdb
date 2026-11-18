//! Optional v0.1.2 ingestion commands.
//!
//! `embed-jsonl` and `import-dir` stay inside the CLI. They do not add a public
//! Rust API. Provider failures are returned before manifest import and do not
//! create or modify the database. After import starts, core errors pass through
//! unchanged.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

use crate::db::Database;
use crate::error::{Error, ErrorCode, Result};
use crate::json::{parse, JsonValue};
use crate::model::OnConflict;
use crate::profile::Durability;

mod boundary;
mod float;
mod http;
mod publish;
mod scan;
mod text;

use boundary::{cli, io_ingest, missing, schema, vector_length};
use float::format_f32;
use http::{
    build_endpoint, embed_batch, BatchRequest, Endpoint, OPENAI_DEFAULT_BASE, REQUEST_TIMEOUT,
};
use publish::{publish, temporary_path, write_new, PlannedFile};
use scan::{
    chunk_files, compile_globs, no_eligible, scan_dir, ChunkRecord, SkippedFile, SourceFile,
    TextFormat,
};
use text::{dataset_preimage, push_string, push_u64, sha256_hex, tokenizer_json, ArtifactBytes};

pub(super) fn embed_jsonl(args: &mut super::Args) -> Result<JsonValue> {
    let flags = parse_flags(take_tokens(args), EMBED_FLAGS)?;
    let input = one_positional(&flags, "chunks.jsonl")?;
    let provider = provider_from_embed_flags(&flags, true)?;
    require_parent(Path::new(&provider.out), "--out")?;
    let sidecar_path = sidecar_path(&provider, Path::new(&provider.out).parent())?;
    require_parent(&sidecar_path, "--metadata-out")?;
    distinct_outputs(&[Path::new(&provider.out), sidecar_path.as_path()])?;
    let endpoint = provider.endpoint.clone();
    let key = read_key(&provider.api_key_env)?;
    let (chunks_sha, rows) = read_chunk_input(Path::new(&input))?;
    if rows.is_empty() {
        return Err(no_eligible());
    }
    let texts: Vec<String> = rows.iter().map(|row| row.text.clone()).collect();
    let vectors = embed_all(
        &endpoint,
        &key,
        &provider.model,
        provider.dimensions,
        provider.batch_size,
        provider.concurrency,
        &texts,
    )?;
    let ids: Vec<String> = rows.iter().map(|row| row.id.clone()).collect();
    let embeddings = embeddings_jsonl(&ids, &provider.model, provider.dimensions, &vectors);
    let generated = generated_at();
    let sidecar = sidecar_json(&SidecarInput {
        provider: provider.kind.as_str(),
        model: &provider.model,
        dimensions: provider.dimensions,
        embedding_input: &input,
        generated_at: &generated,
        batch_size: provider.batch_size,
        chunks_sha256: &chunks_sha,
        embeddings_sha256: &embeddings.sha256,
        endpoint: &endpoint,
    });
    let mut cleanup = Cleanup::default();
    let embeddings_temp =
        temporary_path(Path::new(&provider.out).parent().unwrap_or(Path::new(".")))?;
    let sidecar_temp = temporary_path(sidecar_path.parent().unwrap_or(Path::new(".")))?;
    cleanup.temps.push(embeddings_temp.clone());
    cleanup.temps.push(sidecar_temp.clone());
    write_new(&embeddings_temp, &embeddings.bytes)?;
    write_new(&sidecar_temp, &sidecar.bytes)?;
    self_check_embeddings(
        &embeddings_temp,
        &ids,
        &provider.model,
        provider.dimensions,
        &embeddings.sha256,
    )?;
    self_check_sidecar(
        &sidecar_temp,
        &sidecar.sha256,
        &provider,
        &input,
        &chunks_sha,
        &embeddings.sha256,
    )?;
    publish(vec![
        PlannedFile {
            temp: embeddings_temp,
            formal: PathBuf::from(&provider.out),
        },
        PlannedFile {
            temp: sidecar_temp,
            formal: sidecar_path,
        },
    ])?;
    cleanup.temps.clear();
    Ok(command_result(0, rows.len(), rows.len(), &[]))
}

pub(super) fn import_dir(args: &mut super::Args) -> Result<JsonValue> {
    let flags = parse_flags(take_tokens(args), IMPORT_FLAGS)?;
    let positionals = &flags.positionals;
    if positionals.len() != 2 {
        return Err(cli(
            "import-dir requires <input-dir> and <path>",
            Some("argument"),
            Some("missing_required_argument"),
        ));
    }
    let input_dir = PathBuf::from(&positionals[0]);
    let db_path = &positionals[1];
    let emit = flags.single("--emit-jsonl").ok_or_else(|| {
        cli(
            "import-dir requires --emit-jsonl",
            Some("--emit-jsonl"),
            Some("missing_emit_jsonl"),
        )
    })?;
    if flags.single("--embed").is_none() {
        reject_embed_only_flags(&flags)?;
    }
    let collection = required_text(flags.single("--collection"), "--collection")?;
    let chunk_size = required_u64(flags.single("--chunk-size"), "--chunk-size", 1, 1_048_576)?;
    let chunk_overlap = required_u64(
        flags.single("--chunk-overlap"),
        "--chunk-overlap",
        1,
        1_048_576,
    )?;
    if chunk_overlap >= chunk_size {
        return Err(cli(
            "chunk overlap must be smaller than chunk size",
            Some("--chunk-overlap"),
            Some("overlap_not_smaller"),
        ));
    }
    let format = match flags.single("--format") {
        Some(value) => TextFormat::parse(value)?,
        None => TextFormat::Text,
    };
    let include = compile_globs(&flags.repeated("--include"), "--include")?;
    let exclude = compile_globs(&flags.repeated("--exclude"), "--exclude")?;
    let embed = match flags.single("--embed") {
        Some(value) => Some(provider_from_embed_value(&flags, value)?),
        None => None,
    };
    if let Some(provider) = &embed {
        let emit_dir = Path::new(emit);
        let sidecar = sidecar_path_for(provider, emit_dir)?;
        distinct_outputs(&[
            &emit_dir.join("documents.jsonl"),
            &emit_dir.join("chunks.jsonl"),
            &emit_dir.join("embeddings.jsonl"),
            &emit_dir.join("import_manifest.json"),
            sidecar.as_path(),
        ])?;
        let _key = read_key(&provider.api_key_env)?;
    }
    let scan = scan_dir(&input_dir, format, &include, &exclude)?;
    let chunks = chunk_files(&scan.files, chunk_size as usize, chunk_overlap as usize)?;
    if scan.files.is_empty() || chunks.is_empty() {
        return Err(no_eligible());
    }
    let mut cleanup = Cleanup::default();
    let emit_path = PathBuf::from(emit);
    cleanup.created_dir = prepare_output_dir(&emit_path)?;
    let documents = documents_jsonl(&scan.files, format);
    let chunk_bytes = chunks_jsonl(&chunks);
    let dataset_id = sha256_hex(
        dataset_preimage(
            format.as_str(),
            chunk_size,
            chunk_overlap,
            &scan
                .files
                .iter()
                .map(|file| file.relative.clone())
                .collect::<Vec<_>>(),
        )
        .as_bytes(),
    );
    let doc_temp = temporary_path(&emit_path)?;
    let chunk_temp = temporary_path(&emit_path)?;
    cleanup.temps.push(doc_temp.clone());
    cleanup.temps.push(chunk_temp.clone());
    write_new(&doc_temp, &documents.bytes)?;
    write_new(&chunk_temp, &chunk_bytes.bytes)?;
    let mut planned = vec![
        PlannedFile {
            temp: doc_temp.clone(),
            formal: emit_path.join("documents.jsonl"),
        },
        PlannedFile {
            temp: chunk_temp.clone(),
            formal: emit_path.join("chunks.jsonl"),
        },
    ];
    if let Some(provider) = embed {
        let key = read_key(&provider.api_key_env)?;
        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        let vectors = embed_all(
            &provider.endpoint,
            &key,
            &provider.model,
            provider.dimensions,
            provider.batch_size,
            provider.concurrency,
            &texts,
        )?;
        let ids: Vec<String> = chunks
            .iter()
            .map(|chunk| chunk.chunk_external_id.clone())
            .collect();
        let embeddings = embeddings_jsonl(&ids, &provider.model, provider.dimensions, &vectors);
        let embedding_temp = temporary_path(&emit_path)?;
        cleanup.temps.push(embedding_temp.clone());
        write_new(&embedding_temp, &embeddings.bytes)?;
        let shape = load_collection(Path::new(db_path), &collection)?;
        if shape.embedding_dim != provider.dimensions as usize {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "collection embedding dimension does not match --dimensions",
            )
            .with_detail("field", "embedding_dim")
            .ingestion_boundary());
        }
        if shape
            .embedding_model
            .as_ref()
            .is_some_and(|model| model != &provider.model)
        {
            return Err(Error::new(
                ErrorCode::EmbeddingModelMismatch,
                "collection embedding model does not match --embed",
            )
            .with_detail("field", "embedding_model")
            .ingestion_boundary());
        }
        let manifest = embed_manifest(
            &dataset_id,
            &collection,
            &documents.sha256,
            &chunk_bytes.sha256,
            &embeddings.sha256,
            shape.embedding_model.as_deref(),
            shape.embedding_dim,
            &shape.distance_metric,
            shape.vectors_normalized,
        );
        let generated = generated_at();
        let sidecar_path = sidecar_path_for(&provider, &emit_path)?;
        let sidecar = sidecar_json(&SidecarInput {
            provider: provider.kind.as_str(),
            model: &provider.model,
            dimensions: provider.dimensions,
            embedding_input: "chunks.jsonl",
            generated_at: &generated,
            batch_size: provider.batch_size,
            chunks_sha256: &chunk_bytes.sha256,
            embeddings_sha256: &embeddings.sha256,
            endpoint: &provider.endpoint,
        });
        let manifest_temp = temporary_path(&emit_path)?;
        let sidecar_temp = temporary_path(sidecar_path.parent().unwrap_or(Path::new(".")))?;
        cleanup.temps.push(manifest_temp.clone());
        cleanup.temps.push(sidecar_temp.clone());
        write_new(&manifest_temp, &manifest.bytes)?;
        write_new(&sidecar_temp, &sidecar.bytes)?;
        self_check_text(
            &doc_temp,
            &chunk_temp,
            &manifest_temp,
            &collection,
            &documents.sha256,
            &chunk_bytes.sha256,
        )?;
        self_check_embeddings(
            &embedding_temp,
            &ids,
            &provider.model,
            provider.dimensions,
            &embeddings.sha256,
        )?;
        self_check_embed_manifest(
            &manifest_temp,
            &collection,
            provider.dimensions,
            &embeddings.sha256,
        )?;
        self_check_sidecar(
            &sidecar_temp,
            &sidecar.sha256,
            &provider,
            "chunks.jsonl",
            &chunk_bytes.sha256,
            &embeddings.sha256,
        )?;
        planned.push(PlannedFile {
            temp: embedding_temp,
            formal: emit_path.join("embeddings.jsonl"),
        });
        planned.push(PlannedFile {
            temp: sidecar_temp,
            formal: sidecar_path,
        });
        planned.push(PlannedFile {
            temp: manifest_temp,
            formal: emit_path.join("import_manifest.json"),
        });
        publish(planned)?;
        cleanup.temps.clear();
        let db = super::open_database_with_durability(db_path, Durability::Commit)?;
        db.import_manifest(emit_path.join("import_manifest.json"), OnConflict::Error)
            .map_err(|error| error.with_detail("import_stage", "core_import"))?;
        return Ok(command_result(
            scan.files.len(),
            chunks.len(),
            chunks.len(),
            &scan.skipped,
        ));
    }
    let manifest = text_manifest(
        &dataset_id,
        &collection,
        &documents.sha256,
        &chunk_bytes.sha256,
    );
    let manifest_temp = temporary_path(&emit_path)?;
    cleanup.temps.push(manifest_temp.clone());
    write_new(&manifest_temp, &manifest.bytes)?;
    self_check_text(
        &doc_temp,
        &chunk_temp,
        &manifest_temp,
        &collection,
        &documents.sha256,
        &chunk_bytes.sha256,
    )?;
    planned.push(PlannedFile {
        temp: manifest_temp,
        formal: emit_path.join("import_manifest.json"),
    });
    publish(planned)?;
    cleanup.temps.clear();
    Ok(command_result(
        scan.files.len(),
        chunks.len(),
        0,
        &scan.skipped,
    ))
}

struct Cleanup {
    created_dir: Option<PathBuf>,
    temps: Vec<PathBuf>,
}

impl Default for Cleanup {
    fn default() -> Self {
        Self {
            created_dir: None,
            temps: Vec::new(),
        }
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        for path in &self.temps {
            let _ = fs::remove_file(path);
        }
        if let Some(dir) = &self.created_dir {
            let empty = fs::read_dir(dir)
                .map(|mut entries| entries.next().is_none())
                .unwrap_or(false);
            if empty {
                let _ = fs::remove_dir(dir);
            }
        }
    }
}

struct FlagSet {
    values: Vec<(String, String)>,
    positionals: Vec<String>,
}

impl FlagSet {
    fn single(&self, flag: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(name, _)| name == flag)
            .map(|(_, value)| value.as_str())
    }

    fn repeated(&self, flag: &str) -> Vec<String> {
        self.values
            .iter()
            .filter(|(name, _)| name == flag)
            .map(|(_, value)| value.clone())
            .collect()
    }
}

const EMBED_FLAGS: &[&str] = &[
    "--provider",
    "--model",
    "--out",
    "--dimensions",
    "--batch-size",
    "--concurrency",
    "--api-key-env",
    "--base-url",
    "--metadata-out",
];

const IMPORT_FLAGS: &[&str] = &[
    "--collection",
    "--chunk-size",
    "--chunk-overlap",
    "--format",
    "--include",
    "--exclude",
    "--emit-jsonl",
    "--embed",
    "--dimensions",
    "--batch-size",
    "--concurrency",
    "--api-key-env",
    "--base-url",
    "--metadata-out",
];

const REPEATABLE: &[&str] = &["--include", "--exclude"];

fn take_tokens(args: &mut super::Args) -> Vec<String> {
    let rest = args.args.split_off(args.pos);
    args.pos = args.args.len();
    rest
}

fn parse_flags(mut tokens: Vec<String>, allowed: &[&str]) -> Result<FlagSet> {
    let mut values = Vec::new();
    let mut positionals = Vec::new();
    while !tokens.is_empty() {
        let token = tokens.remove(0);
        if token == "--" || token.starts_with('-') {
            if !allowed.contains(&token.as_str()) {
                return Err(cli(
                    format!("unexpected argument {token}"),
                    Some("argument"),
                    Some("unexpected_argument"),
                ));
            }
            if tokens.is_empty() || tokens[0].starts_with('-') {
                return Err(cli(
                    format!("{token} requires a value"),
                    Some(&token),
                    Some("missing_value"),
                ));
            }
            let value = tokens.remove(0);
            if !REPEATABLE.contains(&token.as_str())
                && values.iter().any(|(name, _)| name == &token)
            {
                return Err(cli(
                    format!("unexpected argument {token}"),
                    Some(&token),
                    Some("unexpected_argument"),
                ));
            }
            values.push((token, value));
        } else {
            positionals.push(token);
        }
    }
    Ok(FlagSet {
        values,
        positionals,
    })
}

fn one_positional(flags: &FlagSet, name: &str) -> Result<String> {
    if flags.positionals.len() != 1 {
        return Err(cli(
            format!("{name} is required"),
            Some(name),
            Some("missing_required_argument"),
        ));
    }
    Ok(flags.positionals[0].clone())
}

struct Provider {
    kind: ProviderKind,
    model: String,
    dimensions: u32,
    batch_size: usize,
    concurrency: usize,
    api_key_env: String,
    endpoint: Endpoint,
    out: String,
    metadata_out: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    Openai,
    Compatible,
}

impl ProviderKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Compatible => "openai-compatible",
        }
    }
}

fn provider_from_embed_flags(flags: &FlagSet, require_out: bool) -> Result<Provider> {
    let provider_name = required_text(flags.single("--provider"), "--provider")?;
    let kind = parse_provider(&provider_name)?;
    let model = match flags.single("--model") {
        Some(value) if !value.is_empty() => value.to_owned(),
        Some(_) => {
            return Err(cli(
                "model must not be empty",
                Some("--model"),
                Some("empty_model"),
            ))
        }
        None => return Err(missing("--model")),
    };
    finish_provider(flags, kind, model, require_out)
}

fn provider_from_embed_value(flags: &FlagSet, value: &str) -> Result<Provider> {
    let Some((kind_name, model)) = value.split_once(':') else {
        return Err(cli(
            "--embed must be provider:model",
            Some("--embed"),
            Some("invalid_embed"),
        ));
    };
    if model.is_empty() {
        return Err(cli(
            "--embed model must not be empty",
            Some("--embed"),
            Some("empty_model"),
        ));
    }
    let kind = parse_provider(kind_name)?;
    finish_provider(flags, kind, model.to_owned(), false)
}

fn finish_provider(
    flags: &FlagSet,
    kind: ProviderKind,
    model: String,
    require_out: bool,
) -> Result<Provider> {
    let dimensions = required_u64(
        flags.single("--dimensions"),
        "--dimensions",
        1,
        u64::from(u32::MAX),
    )? as u32;
    let batch_size = match flags.single("--batch-size") {
        Some(value) => required_u64(Some(value), "--batch-size", 1, 2048)? as usize,
        None => 32,
    };
    let concurrency = match flags.single("--concurrency") {
        Some(value) => required_u64(Some(value), "--concurrency", 1, 8)? as usize,
        None => 1,
    };
    let api_key_env = match kind {
        ProviderKind::Openai => match flags.single("--api-key-env") {
            Some(value) => validate_env_name(value)?,
            None => "OPENAI_API_KEY".to_owned(),
        },
        ProviderKind::Compatible => {
            let name = required_text(flags.single("--api-key-env"), "--api-key-env")?;
            validate_env_name(&name)?
        }
    };
    let base = match kind {
        ProviderKind::Openai => flags
            .single("--base-url")
            .unwrap_or(OPENAI_DEFAULT_BASE)
            .to_owned(),
        ProviderKind::Compatible => required_text(flags.single("--base-url"), "--base-url")?,
    };
    let endpoint = build_endpoint(&base)?;
    let out = if require_out {
        required_text(flags.single("--out"), "--out")?
    } else {
        flags.single("--out").unwrap_or("").to_owned()
    };
    Ok(Provider {
        kind,
        model,
        dimensions,
        batch_size,
        concurrency,
        api_key_env,
        endpoint,
        out,
        metadata_out: flags.single("--metadata-out").map(str::to_owned),
    })
}

fn parse_provider(name: &str) -> Result<ProviderKind> {
    match name {
        "openai" => Ok(ProviderKind::Openai),
        "openai-compatible" => Ok(ProviderKind::Compatible),
        _ => Err(cli(
            "provider must be openai or openai-compatible",
            Some("provider"),
            Some("unsupported_provider"),
        )),
    }
}

fn validate_env_name(name: &str) -> Result<String> {
    let mut chars = name.chars();
    let valid = matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
    if valid {
        Ok(name.to_owned())
    } else {
        Err(cli(
            "api key variable name is invalid",
            Some("--api-key-env"),
            Some("invalid_env_name"),
        ))
    }
}

fn required_text(value: Option<&str>, field: &str) -> Result<String> {
    match value {
        Some(value) if !value.is_empty() => Ok(value.to_owned()),
        Some(_) => Err(missing(field)),
        None => Err(missing(field)),
    }
}

fn required_u64(value: Option<&str>, field: &str, min: u64, max: u64) -> Result<u64> {
    let Some(text) = value else {
        return Err(missing(field));
    };
    if text.is_empty()
        || !text.bytes().all(|byte| byte.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return Err(cli(
            format!("{field} must be an integer without a sign or leading zero"),
            Some(field),
            Some("invalid_integer"),
        ));
    }
    let parsed = text.parse::<u64>().map_err(|_| {
        cli(
            format!("{field} must be an integer without a sign or leading zero"),
            Some(field),
            Some("invalid_integer"),
        )
    })?;
    if parsed < min || parsed > max {
        return Err(cli(
            format!("{field} is outside the supported range"),
            Some(field),
            Some("invalid_integer"),
        ));
    }
    Ok(parsed)
}

fn reject_embed_only_flags(flags: &FlagSet) -> Result<()> {
    for flag in [
        "--dimensions",
        "--batch-size",
        "--concurrency",
        "--api-key-env",
        "--base-url",
        "--metadata-out",
    ] {
        if flags.single(flag).is_some() {
            return Err(cli(
                format!("{flag} requires --embed"),
                Some(flag),
                Some("unexpected_argument"),
            ));
        }
    }
    Ok(())
}

fn read_key(name: &str) -> Result<String> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() && !value.chars().any(char::is_control) => Ok(value),
        _ => Err(missing(name)),
    }
}

fn distinct_outputs(paths: &[&Path]) -> Result<()> {
    let mut seen = Vec::with_capacity(paths.len());
    for path in paths {
        let logical = logical_path(path);
        if seen.iter().any(|item| item == &logical) {
            return Err(cli(
                "output paths must name different files",
                Some("argument"),
                Some("output_path_collision"),
            ));
        }
        seen.push(logical);
    }
    Ok(())
}

fn logical_path(path: &Path) -> PathBuf {
    // Join a relative path to the process directory so it matches an absolute alias.
    let cwd = if path.is_absolute() {
        None
    } else {
        std::env::current_dir().ok()
    };
    let mut out = PathBuf::new();
    for component in cwd
        .iter()
        .flat_map(|dir| dir.components())
        .chain(path.components())
    {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                // A parent of `/` is still `/` on this platform. Keep `..` only
                // for a relative path that could not be anchored.
                if !out.pop() && !out.has_root() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn require_parent(path: &Path, field: &str) -> Result<()> {
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    if parent.is_dir() {
        Ok(())
    } else {
        Err(cli(
            "output parent directory does not exist",
            Some(field),
            Some("missing_parent"),
        ))
    }
}

fn sidecar_path(provider: &Provider, out_parent: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = &provider.metadata_out {
        return Ok(PathBuf::from(path));
    }
    let parent = match out_parent {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    Ok(parent.join("embedding_artifact.json"))
}

fn sidecar_path_for(provider: &Provider, emit_dir: &Path) -> Result<PathBuf> {
    if let Some(path) = &provider.metadata_out {
        require_parent(Path::new(path), "--metadata-out")?;
        Ok(PathBuf::from(path))
    } else {
        Ok(emit_dir.join("embedding_artifact.json"))
    }
}

struct InputChunk {
    id: String,
    text: String,
}

fn read_chunk_input(path: &Path) -> Result<(String, Vec<InputChunk>)> {
    let bytes = fs::read(path).map_err(|_| io_ingest("input read failed", None))?;
    let digest = sha256_hex(&bytes);
    let text =
        std::str::from_utf8(&bytes).map_err(|_| schema("chunks input is not valid jsonl"))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if text.is_empty() {
        return Err(no_eligible());
    }
    if !text.ends_with('\n') {
        return Err(schema("chunks jsonl line is missing a newline"));
    }
    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    if lines.is_empty() {
        return Err(no_eligible());
    }
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() {
            return Err(schema("chunks jsonl contains an empty line"));
        }
        let value = parse(line).map_err(|_| schema("chunks jsonl is invalid"))?;
        let obj = value
            .as_object()
            .ok_or_else(|| schema("chunks jsonl row must be an object"))?;
        let id = match obj.get("chunk_external_id") {
            Some(JsonValue::String(id)) if !id.is_empty() => id.clone(),
            _ => return Err(schema("chunk_external_id must be a non-empty string")),
        };
        let text = match obj.get("text") {
            Some(JsonValue::String(text)) => text.clone(),
            _ => return Err(schema("text must be a string")),
        };
        if !seen.insert(id.clone()) {
            return Err(Error::new(
                ErrorCode::DuplicateExternalId,
                "duplicate chunk_external_id",
            )
            .with_detail("field", "chunk_external_id")
            .ingestion_boundary());
        }
        rows.push(InputChunk { id, text });
    }
    Ok((digest, rows))
}

struct CollectionShape {
    embedding_dim: usize,
    distance_metric: String,
    embedding_model: Option<String>,
    vectors_normalized: bool,
}

fn load_collection(path: &Path, name: &str) -> Result<CollectionShape> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(
                Error::new(ErrorCode::CollectionNotFound, "collection not found")
                    .with_detail("collection", name)
                    .ingestion_boundary(),
            );
        }
        Err(error) => return Err(Error::from(error).ingestion_boundary()),
        Ok(_) => {}
    }
    let db = Database::open_read_only(path).map_err(|error| error.ingestion_boundary())?;
    let info = db
        .list_collections()
        .map_err(|error| error.ingestion_boundary())?
        .into_iter()
        .find(|info| info.name == name);
    match info {
        Some(info) => Ok(CollectionShape {
            embedding_dim: info.embedding_dim,
            distance_metric: info.distance_metric.as_str().to_owned(),
            embedding_model: info.embedding_model,
            vectors_normalized: info.vectors_normalized,
        }),
        None => Err(
            Error::new(ErrorCode::CollectionNotFound, "collection not found")
                .with_detail("collection", name)
                .ingestion_boundary(),
        ),
    }
}

fn prepare_output_dir(path: &Path) -> Result<Option<PathBuf>> {
    if path.exists() {
        if path.is_dir() {
            Ok(None)
        } else {
            Err(io_ingest("output directory creation failed", None))
        }
    } else {
        fs::create_dir(path).map_err(|_| io_ingest("output directory creation failed", None))?;
        Ok(Some(path.to_owned()))
    }
}

fn documents_jsonl(files: &[SourceFile], format: TextFormat) -> ArtifactBytes {
    let mut out = String::new();
    for file in files {
        let title = file.relative.rsplit('/').next().unwrap_or(&file.relative);
        out.push_str("{\"external_id\":");
        push_string(&mut out, &format!("file://{}", file.relative));
        out.push_str(",\"title\":");
        push_string(&mut out, title);
        out.push_str(",\"source_uri\":");
        push_string(&mut out, &file.relative);
        out.push_str(",\"metadata\":{\"format\":");
        push_string(&mut out, format.as_str());
        out.push_str(",\"path\":");
        push_string(&mut out, &file.relative);
        out.push_str("}}\n");
    }
    ArtifactBytes::from_string(out)
}

fn chunks_jsonl(chunks: &[ChunkRecord]) -> ArtifactBytes {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str("{\"document_external_id\":");
        push_string(&mut out, &chunk.external_id);
        out.push_str(",\"chunk_external_id\":");
        push_string(&mut out, &chunk.chunk_external_id);
        out.push_str(",\"ordinal\":");
        out.push_str(&chunk.ordinal.to_string());
        out.push_str(",\"text\":");
        push_string(&mut out, &chunk.text);
        out.push_str(",\"metadata\":{\"path\":");
        push_string(&mut out, &chunk.relative);
        out.push_str("},\"source_span\":{\"start\":");
        push_u64(&mut out, chunk.start);
        out.push_str(",\"end\":");
        push_u64(&mut out, chunk.end);
        out.push_str(",\"line_start\":");
        out.push_str(&chunk.line_start.to_string());
        out.push_str(",\"line_end\":");
        out.push_str(&chunk.line_end.to_string());
        out.push_str("}}\n");
    }
    ArtifactBytes::from_string(out)
}

fn embeddings_jsonl(
    ids: &[String],
    model: &str,
    dimensions: u32,
    vectors: &[Vec<f32>],
) -> ArtifactBytes {
    let mut out = String::new();
    for (id, vector) in ids.iter().zip(vectors) {
        out.push_str("{\"chunk_external_id\":");
        push_string(&mut out, id);
        out.push_str(",\"model_name\":");
        push_string(&mut out, model);
        out.push_str(",\"dim\":");
        out.push_str(&dimensions.to_string());
        out.push_str(",\"vector\":[");
        for (index, value) in vector.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(&format_f32(*value));
        }
        out.push_str("]}\n");
    }
    ArtifactBytes::from_string(out)
}

fn text_manifest(
    dataset_id: &str,
    collection: &str,
    documents_sha: &str,
    chunks_sha: &str,
) -> ArtifactBytes {
    let mut out = String::from("{\"schema_version\":\"0.1\",\"dataset_id\":");
    push_string(&mut out, dataset_id);
    out.push_str(",\"collection\":");
    push_string(&mut out, collection);
    out.push_str(
        ",\"documents\":\"documents.jsonl\",\"chunks\":\"chunks.jsonl\",\"hashes\":{\"documents_sha256\":",
    );
    push_string(&mut out, documents_sha);
    out.push_str(",\"chunks_sha256\":");
    push_string(&mut out, chunks_sha);
    out.push_str("}}\n");
    ArtifactBytes::from_string(out)
}

fn embed_manifest(
    dataset_id: &str,
    collection: &str,
    documents_sha: &str,
    chunks_sha: &str,
    embeddings_sha: &str,
    model: Option<&str>,
    embedding_dim: usize,
    distance_metric: &str,
    vectors_normalized: bool,
) -> ArtifactBytes {
    let mut out = String::from("{\"schema_version\":\"0.1\",\"dataset_id\":");
    push_string(&mut out, dataset_id);
    out.push_str(",\"collection\":");
    push_string(&mut out, collection);
    out.push_str(
        ",\"documents\":\"documents.jsonl\",\"chunks\":\"chunks.jsonl\",\"embeddings\":\"embeddings.jsonl\"",
    );
    if let Some(model) = model {
        out.push_str(",\"embedding_model\":");
        push_string(&mut out, model);
    }
    out.push_str(",\"embedding_dim\":");
    out.push_str(&embedding_dim.to_string());
    out.push_str(",\"distance_metric\":");
    push_string(&mut out, distance_metric);
    out.push_str(",\"vectors_normalized\":");
    out.push_str(if vectors_normalized { "true" } else { "false" });
    out.push_str(",\"tokenizer\":");
    out.push_str(tokenizer_json());
    out.push_str(",\"hashes\":{\"documents_sha256\":");
    push_string(&mut out, documents_sha);
    out.push_str(",\"chunks_sha256\":");
    push_string(&mut out, chunks_sha);
    out.push_str(",\"embeddings_sha256\":");
    push_string(&mut out, embeddings_sha);
    out.push_str("}}\n");
    ArtifactBytes::from_string(out)
}

struct SidecarInput<'a> {
    provider: &'a str,
    model: &'a str,
    dimensions: u32,
    embedding_input: &'a str,
    generated_at: &'a str,
    batch_size: usize,
    chunks_sha256: &'a str,
    embeddings_sha256: &'a str,
    endpoint: &'a Endpoint,
}

fn sidecar_json(input: &SidecarInput<'_>) -> ArtifactBytes {
    let mut out = String::from("{\"embedding_provider\":");
    push_string(&mut out, input.provider);
    out.push_str(",\"embedding_provider_kind\":\"http\",\"embedding_model\":");
    push_string(&mut out, input.model);
    out.push_str(",\"embedding_dim\":");
    out.push_str(&input.dimensions.to_string());
    out.push_str(",\"embedding_dimensions_requested\":");
    out.push_str(&input.dimensions.to_string());
    out.push_str(",\"embedding_input\":");
    push_string(&mut out, input.embedding_input);
    out.push_str(",\"embedding_generated_at\":");
    push_string(&mut out, input.generated_at);
    out.push_str(",\"embedding_batch_size\":");
    out.push_str(&input.batch_size.to_string());
    out.push_str(",\"chunks_sha256\":");
    push_string(&mut out, input.chunks_sha256);
    out.push_str(",\"embeddings_sha256\":");
    push_string(&mut out, input.embeddings_sha256);
    out.push_str(",\"endpoint_host\":");
    push_string(&mut out, &input.endpoint.host);
    out.push_str(",\"endpoint_fingerprint\":");
    push_string(&mut out, &sha256_hex(input.endpoint.url.as_bytes()));
    out.push_str("}\n");
    ArtifactBytes::from_string(out)
}

fn generated_at() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    crate::time::format_unix_millis(seconds.saturating_mul(1000))
}

fn embed_all(
    endpoint: &Endpoint,
    key: &str,
    model: &str,
    dimensions: u32,
    batch_size: usize,
    concurrency: usize,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let batches: Vec<&[String]> = texts.chunks(batch_size).collect();
    if batches.is_empty() {
        return Ok(Vec::new());
    }
    if concurrency <= 1 || batches.len() == 1 {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in batches {
            vectors.extend(embed_batch(BatchRequest {
                endpoint,
                key,
                model,
                dimensions,
                inputs: batch,
                timeout: REQUEST_TIMEOUT,
            })?);
        }
        return Ok(vectors);
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let errors = Mutex::new(Vec::<(usize, Error)>::new());
    let outputs = Mutex::new(vec![None; batches.len()]);
    thread::scope(|scope| {
        for _ in 0..concurrency.min(batches.len()) {
            scope.spawn(|| loop {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= batches.len() || stop.load(Ordering::Acquire) {
                    break;
                }
                match embed_batch(BatchRequest {
                    endpoint,
                    key,
                    model,
                    dimensions,
                    inputs: batches[index],
                    timeout: REQUEST_TIMEOUT,
                }) {
                    Ok(vectors) => {
                        if let Ok(mut slot) = outputs.lock() {
                            slot[index] = Some(vectors);
                        }
                    }
                    Err(error) => {
                        stop.store(true, Ordering::Release);
                        if let Ok(mut stored) = errors.lock() {
                            stored.push((index, error));
                        }
                        break;
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap_or_default();
    if let Some((_, error)) = errors.into_iter().min_by_key(|(index, _)| *index) {
        return Err(error);
    }
    let outputs = outputs.into_inner().unwrap_or_default();
    let mut vectors = Vec::with_capacity(texts.len());
    for batch in outputs {
        vectors.extend(batch.ok_or_else(|| io_ingest("provider request failed", Some("request")))?);
    }
    if vectors.len() != texts.len() {
        return Err(vector_length(
            "provider response count does not match the request",
        ));
    }
    Ok(vectors)
}

fn self_check_text(
    documents: &Path,
    chunks: &Path,
    manifest: &Path,
    collection: &str,
    documents_sha: &str,
    chunks_sha: &str,
) -> Result<()> {
    let documents_bytes = read_nonempty(documents)?;
    let chunks_bytes = read_nonempty(chunks)?;
    let manifest_bytes = read_nonempty(manifest)?;
    if sha256_hex(&documents_bytes) != documents_sha || sha256_hex(&chunks_bytes) != chunks_sha {
        return Err(schema("artifact hash self-check failed"));
    }
    let manifest_text =
        std::str::from_utf8(&manifest_bytes).map_err(|_| schema("artifact self-check failed"))?;
    let value = parse(manifest_text).map_err(|_| schema("artifact self-check failed"))?;
    if value.get("collection").and_then(JsonValue::as_str) != Some(collection) {
        return Err(schema("artifact self-check failed"));
    }
    if value.get("documents").and_then(JsonValue::as_str) != Some("documents.jsonl")
        || value.get("chunks").and_then(JsonValue::as_str) != Some("chunks.jsonl")
    {
        return Err(schema("artifact self-check failed"));
    }
    let hashes = value
        .get("hashes")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| schema("artifact self-check failed"))?;
    if hashes.get("documents_sha256").and_then(JsonValue::as_str) != Some(documents_sha)
        || hashes.get("chunks_sha256").and_then(JsonValue::as_str) != Some(chunks_sha)
    {
        return Err(schema("artifact self-check failed"));
    }
    let _ = jsonl_values(&documents_bytes)?;
    for row in jsonl_values(&chunks_bytes)? {
        let text = row.get("text").and_then(JsonValue::as_str).unwrap_or("");
        if text.trim().is_empty() {
            return Err(schema("artifact self-check failed"));
        }
    }
    Ok(())
}

fn self_check_embeddings(
    path: &Path,
    ids: &[String],
    model: &str,
    dimensions: u32,
    expected_sha: &str,
) -> Result<()> {
    let bytes = read_nonempty(path)?;
    if sha256_hex(&bytes) != expected_sha {
        return Err(schema("artifact hash self-check failed"));
    }
    let rows = jsonl_values(&bytes)?;
    if rows.len() != ids.len() {
        return Err(schema("artifact self-check failed"));
    }
    for (row, id) in rows.iter().zip(ids) {
        if row.get("chunk_external_id").and_then(JsonValue::as_str) != Some(id.as_str())
            || row.get("model_name").and_then(JsonValue::as_str) != Some(model)
            || row.get("dim").and_then(JsonValue::as_f64) != Some(f64::from(dimensions))
        {
            return Err(schema("artifact self-check failed"));
        }
        let vector = row
            .get("vector")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| schema("artifact self-check failed"))?;
        if vector.len() != dimensions as usize {
            return Err(schema("artifact self-check failed"));
        }
    }
    Ok(())
}

fn self_check_embed_manifest(
    path: &Path,
    collection: &str,
    dimensions: u32,
    embeddings_sha: &str,
) -> Result<()> {
    let bytes = read_nonempty(path)?;
    let value =
        parse(std::str::from_utf8(&bytes).map_err(|_| schema("artifact self-check failed"))?)
            .map_err(|_| schema("artifact self-check failed"))?;
    if value.get("collection").and_then(JsonValue::as_str) != Some(collection) {
        return Err(schema("artifact self-check failed"));
    }
    if value.get("embedding_dim").and_then(JsonValue::as_f64) != Some(f64::from(dimensions)) {
        return Err(schema("artifact self-check failed"));
    }
    if value.get("embeddings").and_then(JsonValue::as_str) != Some("embeddings.jsonl") {
        return Err(schema("artifact self-check failed"));
    }
    let hashes = value
        .get("hashes")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| schema("artifact self-check failed"))?;
    if hashes.get("embeddings_sha256").and_then(JsonValue::as_str) != Some(embeddings_sha) {
        return Err(schema("artifact self-check failed"));
    }
    Ok(())
}

fn self_check_sidecar(
    path: &Path,
    expected_sha: &str,
    provider: &Provider,
    embedding_input: &str,
    chunks_sha: &str,
    embeddings_sha: &str,
) -> Result<()> {
    let bytes = read_nonempty(path)?;
    if sha256_hex(&bytes) != expected_sha {
        return Err(schema("artifact hash self-check failed"));
    }
    let value =
        parse(std::str::from_utf8(&bytes).map_err(|_| schema("artifact self-check failed"))?)
            .map_err(|_| schema("artifact self-check failed"))?;
    let timestamp = value
        .get("embedding_generated_at")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    if value.get("embedding_provider").and_then(JsonValue::as_str) != Some(provider.kind.as_str())
        || value
            .get("embedding_provider_kind")
            .and_then(JsonValue::as_str)
            != Some("http")
        || value.get("embedding_model").and_then(JsonValue::as_str) != Some(provider.model.as_str())
        || value.get("embedding_dim").and_then(JsonValue::as_f64)
            != Some(f64::from(provider.dimensions))
        || value
            .get("embedding_dimensions_requested")
            .and_then(JsonValue::as_f64)
            != Some(f64::from(provider.dimensions))
        || value.get("embedding_input").and_then(JsonValue::as_str) != Some(embedding_input)
        || value
            .get("embedding_batch_size")
            .and_then(JsonValue::as_f64)
            != Some(provider.batch_size as f64)
        || value.get("chunks_sha256").and_then(JsonValue::as_str) != Some(chunks_sha)
        || value.get("embeddings_sha256").and_then(JsonValue::as_str) != Some(embeddings_sha)
        || value.get("endpoint_host").and_then(JsonValue::as_str)
            != Some(provider.endpoint.host.as_str())
        || value
            .get("endpoint_fingerprint")
            .and_then(JsonValue::as_str)
            != Some(sha256_hex(provider.endpoint.url.as_bytes()).as_str())
        || timestamp.len() != 20
        || !timestamp.ends_with('Z')
        || timestamp.contains('.')
    {
        return Err(schema("artifact self-check failed"));
    }
    Ok(())
}

fn read_nonempty(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|_| schema("artifact self-check failed"))?;
    if bytes.is_empty() {
        return Err(schema("artifact self-check failed"));
    }
    Ok(bytes)
}

fn jsonl_values(bytes: &[u8]) -> Result<Vec<JsonValue>> {
    let text = std::str::from_utf8(bytes).map_err(|_| schema("artifact self-check failed"))?;
    if !text.ends_with('\n') {
        return Err(schema("artifact self-check failed"));
    }
    let mut rows = Vec::new();
    for line in text.split('\n') {
        if line.is_empty() {
            continue;
        }
        rows.push(parse(line).map_err(|_| schema("artifact self-check failed"))?);
    }
    if rows.is_empty() {
        return Err(schema("artifact self-check failed"));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests;

fn command_result(
    documents: usize,
    chunks: usize,
    embeddings: usize,
    skipped: &[SkippedFile],
) -> JsonValue {
    crate::json::object_from_pairs([
        ("documents", JsonValue::Number(documents as f64)),
        ("chunks", JsonValue::Number(chunks as f64)),
        ("embeddings", JsonValue::Number(embeddings as f64)),
        (
            "skipped_files",
            JsonValue::Array(skipped.iter().map(SkippedFile::to_json).collect()),
        ),
    ])
}
