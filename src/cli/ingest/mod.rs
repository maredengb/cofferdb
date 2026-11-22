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
mod encoding;
mod float;
mod http;
mod publish;
mod scan;
mod text;

use boundary::{cli, io_ingest, missing, schema, vector_length};
use encoding::InputEncoding;
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

// Generate embeddings for an existing chunk JSONL file. Provider success supplies
// the vectors; this command still verifies the serialized artifact and its sidecar
// before publication.
pub(super) fn embed_jsonl(args: &mut super::Args) -> Result<JsonValue> {
    let flags = parse_flags(take_tokens(args), EMBED_FLAGS)?;
    let input = one_positional(&flags, "chunks.jsonl")?;
    let provider = provider_from_embed_flags(&flags, true)?;
    // Both the embedding file and metadata file need usable parent directories.
    // Resolve the default sidecar beside the requested output before checking that
    // the two destinations differ.
    require_parent(Path::new(&provider.out), "--out")?;
    let sidecar_path = sidecar_path(&provider, Path::new(&provider.out).parent())?;
    require_parent(&sidecar_path, "--metadata-out")?;
    distinct_outputs(&[Path::new(&provider.out), sidecar_path.as_path()])?;
    let endpoint = provider.endpoint.clone();
    // Read the credential and validate the input rows before any provider request.
    // The batch texts and external IDs below are derived from this same validated
    // sequence.
    let key = read_key(&provider.api_key_env)?;
    // Hash the actual input file for the sidecar. The digest identifies the chunk
    // JSONL bytes, including their serialization, rather than a newly encoded list
    // of text values.
    let (chunks_sha, rows) = read_chunk_input(Path::new(&input))?;
    if rows.is_empty() {
        return Err(no_eligible());
    }
    // Form provider batches from chunk text alone. Keep the IDs local for joining
    // the vectors back to the original input after the provider has supplied all
    // batches.
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
    // Pair returned vectors with the retained input IDs in original order. Batches
    // may finish out of order, so embed_all must restore that order before handing
    // results back.
    let ids: Vec<String> = rows.iter().map(|row| row.id.clone()).collect();
    let embeddings = embeddings_jsonl(&ids, &provider.model, provider.dimensions, &vectors);
    let generated = generated_at();
    // Bind the provider metadata to the input and emitted embedding digests. The
    // generated time describes this run and is kept in the sidecar rather than the
    // embedding rows.
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
    // Keep both formal destinations untouched while their staged representations
    // are written and checked. The cleanup owner still covers temporary names if
    // either verification fails.
    let mut cleanup = Cleanup::default();
    let embeddings_temp =
        temporary_path(Path::new(&provider.out).parent().unwrap_or(Path::new(".")))?;
    let sidecar_temp = temporary_path(sidecar_path.parent().unwrap_or(Path::new(".")))?;
    cleanup.temps.push(embeddings_temp.clone());
    cleanup.temps.push(sidecar_temp.clone());
    write_new(&embeddings_temp, &embeddings.bytes)?;
    write_new(&sidecar_temp, &sidecar.bytes)?;
    // Reopen the staged vectors to check their IDs, dimensions, model and digest.
    // Provider success alone does not establish that the serialized output passes
    // these checks.
    self_check_embeddings(
        &embeddings_temp,
        &ids,
        &provider.model,
        provider.dimensions,
        &embeddings.sha256,
    )?;
    // Verify the sidecar independently against this request and the two known
    // digests. Only then are the output names eligible to replace their prior
    // contents under the publication routine.
    self_check_sidecar(
        &sidecar_temp,
        &sidecar.sha256,
        &provider,
        &input,
        &chunks_sha,
        &embeddings.sha256,
    )?;
    // Install the checked embeddings and sidecar through the same publication
    // plan. Each rename changes one destination, so a failure after the first can
    // require rollback.
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
    // Successful publication consumed the temporary files. Clear the remembered
    // paths so this guard cannot later act on a name the publisher already moved.
    cleanup.temps.clear();
    Ok(command_result(0, rows.len(), rows.len(), &[], None))
}

// Publish the checked inputs before handing them to the database. Text-only
// mode ends with reusable artifacts. With embeddings, opening the writer or
// importing the manifest can still fail after publication. The named inputs
// remain available; check for already committed batches before replaying them.
pub(super) fn import_dir(args: &mut super::Args) -> Result<JsonValue> {
    let flags = parse_flags(take_tokens(args), IMPORT_FLAGS)?;
    // Directory import needs a source directory and a database path even in its
    // text-only mode. The generation-only branch later completes without opening
    // that database.
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
    // Without --embed the operation produces text artifacts. Reject
    // provider-specific options here so a supplied dimension or endpoint cannot
    // appear to affect a text-only run.
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
    // A smaller overlap than chunk size is necessary for every new text window to
    // advance. Both values also have independent numeric bounds checked above.
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
    // An absent encoding keeps the scanner's default UTF-8 path. A supplied label
    // is validated before scanning, and its spelling is carried into skipped-file
    // diagnostics.
    let encoding = flags
        .single("--input-encoding")
        .map(InputEncoding::parse)
        .transpose()?;
    let include = compile_globs(&flags.repeated("--include"), "--include")?;
    let exclude = compile_globs(&flags.repeated("--exclude"), "--exclude")?;
    let embed = match flags.single("--embed") {
        Some(value) => Some(provider_from_embed_value(&flags, value)?),
        None => None,
    };
    // Validate the artifact destinations before sending any text to the provider. In
    // particular, a sidecar must not overwrite one of the inputs that its manifest
    // will identify.
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
        // Credential availability is a preflight for embedded directory import.
        // The value is read again at request time; this early read does not
        // contact a provider.
        let _key = read_key(&provider.api_key_env)?;
    }
    let scan = scan_dir(&input_dir, format, &include, &exclude, encoding)?;
    // Use the same ordered scan to build chunks and document rows. Reject a
    // directory with no usable documents or resulting chunks before creating an
    // output directory.
    let chunks = chunk_files(&scan.files, chunk_size as usize, chunk_overlap as usize)?;
    if scan.files.is_empty() || chunks.is_empty() {
        return Err(no_eligible());
    }
    let mut cleanup = Cleanup::default();
    let emit_path = PathBuf::from(emit);
    // Remember a newly created output directory separately from its files. An
    // error may remove it only while empty; an already existing destination
    // directory belongs to the caller.
    cleanup.created_dir = prepare_output_dir(&emit_path)?;
    let documents = documents_jsonl(&scan.files, format);
    let chunk_bytes = chunks_jsonl(&chunks);
    // The dataset identifier covers the format, chunk parameters and relative
    // paths. Actual document, chunk and embedding content is identified separately
    // by the digests stored in the manifest.
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
    // Create private staging files first. Validation below reads these paths rather
    // than any artifacts left at the formal names by a previous run.
    let doc_temp = temporary_path(&emit_path)?;
    let chunk_temp = temporary_path(&emit_path)?;
    // Track staging paths before writing their bytes so an early failure can clean
    // the files created so far. The publication plan below maps those temporary
    // names to their user-visible names.
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
    // The embedding mode has an external side effect before opening the database
    // for import. A later collection or artifact failure cannot undo a request
    // that has already sent chunk text.
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
        // Take IDs from the exact sequence sent for embedding. They are the join
        // keys for JSONL rows and are also passed to the validation that runs on
        // the staged file.
        let ids: Vec<String> = chunks
            .iter()
            .map(|chunk| chunk.chunk_external_id.clone())
            .collect();
        let embeddings = embeddings_jsonl(&ids, &provider.model, provider.dimensions, &vectors);
        let embedding_temp = temporary_path(&emit_path)?;
        cleanup.temps.push(embedding_temp.clone());
        write_new(&embedding_temp, &embeddings.bytes)?;
        // The provider request has already completed, but its vectors are still
        // staged. Check them against the target collection before publishing files
        // that claim to be importable into it.
        let shape = load_collection(Path::new(db_path), &collection)?;
        // Check the collection's required vector width independently from its
        // optional model constraint. A correctly shaped provider response can
        // still be incompatible with the chosen destination.
        if shape.embedding_dim != provider.dimensions as usize {
            return Err(Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "collection embedding dimension does not match --dimensions",
            )
            .with_detail("field", "embedding_dim")
            .ingestion_boundary());
        }
        // A missing collection model leaves this part of the import unconstrained.
        // A stored model name, when present, must match the provider's selected
        // model.
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
        // Build the manifest from the actual collection shape, including its
        // metric and normalization settings. The provider supplies vectors but
        // cannot decide those database properties.
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
        // The metadata output can live outside the main emit directory. Stage it
        // beside its own formal destination so its publication does not require
        // moving that file across filesystems.
        let sidecar_temp = temporary_path(sidecar_path.parent().unwrap_or(Path::new(".")))?;
        cleanup.temps.push(manifest_temp.clone());
        cleanup.temps.push(sidecar_temp.clone());
        write_new(&manifest_temp, &manifest.bytes)?;
        write_new(&sidecar_temp, &sidecar.bytes)?;
        // Verify the generated files as a set before renaming any of them. The
        // sidecar and manifest identify the staged data by digest; checking just the
        // provider response would miss a mismatch introduced while emitting JSONL.
        self_check_text(
            &doc_temp,
            &chunk_temp,
            &manifest_temp,
            &collection,
            &documents.sha256,
            &chunk_bytes.sha256,
        )?;
        // Validate the on-disk embedding rows against IDs in request order, after
        // validating the document and chunk inputs they reference.
        self_check_embeddings(
            &embedding_temp,
            &ids,
            &provider.model,
            provider.dimensions,
            &embeddings.sha256,
        )?;
        // A valid embeddings file is insufficient if the manifest identifies a
        // different artifact. Check its embedding digest, collection and
        // dimensions as a separate step.
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
        // Place data and sidecar in the plan before the manifest. Readers of the
        // formal manifest need the files named by it to have been installed first.
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
        // The formal manifest is now available to the core importer. Use commit
        // durability and conflict errors, retaining native import failure
        // information plus a stage that distinguishes it from generation failures.
        db.import_manifest(emit_path.join("import_manifest.json"), OnConflict::Error)
            .map_err(|error| error.with_detail("import_stage", "core_import"))?;
        return Ok(command_result(
            scan.files.len(),
            chunks.len(),
            chunks.len(),
            &scan.skipped,
            encoding,
        ));
    }
    // The text-only path ends by publishing reusable import inputs. It does not open
    // the target for a write, so the emitted manifest can be inspected or imported
    // separately.
    // A text-only manifest contains the document and chunk artifacts. There are no
    // provider rows to validate or import into the requested database on this
    // branch.
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
    // Publish only after reading and checking the staged text artifacts and
    // manifest. The result counts distinguish discovered input files, emitted
    // chunks and zero generated embeddings.
    publish(planned)?;
    cleanup.temps.clear();
    Ok(command_result(
        scan.files.len(),
        chunks.len(),
        0,
        &scan.skipped,
        encoding,
    ))
}

struct Cleanup {
    // Directory created by this command, eligible for removal on drop only while
    // empty.
    created_dir: Option<PathBuf>,
    // Temporary paths still owned by this command and removed on early return.
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
        // Try all tracked names even if one removal fails. Cleanup runs during
        // destruction and cannot replace the error that caused staging to stop.
        for path in &self.temps {
            let _ = fs::remove_file(path);
        }
        if let Some(dir) = &self.created_dir {
            // A directory read error cannot establish emptiness. Preserve the
            // directory in that case; even after a successful check, remove_dir
            // itself requires it still be empty.
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
    // Option names and values in input order, preserving repeated flags such as
    // include and exclude.
    values: Vec<(String, String)>,
    // Non-option arguments retained for the command's positional arity check.
    positionals: Vec<String>,
}

impl FlagSet {
    // The flag parser rejects duplicates for single-valued names. This accessor
    // may therefore borrow the first matching value without resolving competing
    // settings.
    fn single(&self, flag: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(name, _)| name == flag)
            .map(|(_, value)| value.as_str())
    }

    // Repeatable globs retain their CLI order. Clone the matching values so glob
    // compilation can use a standalone list while the rest of the flag set remains
    // available.
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
    "--input-encoding",
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

// Only selection globs accumulate multiple values. A repeated provider, output or
// sizing option must be diagnosed instead of silently replacing the first value.
const REPEATABLE: &[&str] = &["--include", "--exclude"];

fn take_tokens(args: &mut super::Args) -> Vec<String> {
    // Transfer only the suffix after the outer command and program name. Set the
    // outer cursor to the end so it does not parse the ingest arguments a second
    // time.
    let rest = args.args.split_off(args.pos);
    args.pos = args.args.len();
    rest
}

fn parse_flags(mut tokens: Vec<String>, allowed: &[&str]) -> Result<FlagSet> {
    let mut values = Vec::new();
    let mut positionals = Vec::new();
    // Read tokens in command-line order even though flags and positionals end up
    // in separate vectors. This lets parse errors identify the first invalid
    // option encountered.
    while !tokens.is_empty() {
        let token = tokens.remove(0);
        // A leading dash selects option parsing, even for the bare -- token. There
        // is no end-of-options mode here that would turn subsequent tokens into
        // arbitrary positional values.
        if token == "--" || token.starts_with('-') {
            // Check the allowlist for this command before looking for an option
            // value. Unknown options receive a different diagnostic from
            // recognized options missing an operand.
            if !allowed.contains(&token.as_str()) {
                return Err(cli(
                    format!("unexpected argument {token}"),
                    Some("argument"),
                    Some("unexpected_argument"),
                ));
            }
            // Do not consume another dash-prefixed token as this option's value.
            // This reports the missing operand at its flag instead of making the
            // following option look like input text.
            if tokens.is_empty() || tokens[0].starts_with('-') {
                return Err(cli(
                    format!("{token} requires a value"),
                    Some(&token),
                    Some("missing_value"),
                ));
            }
            // Consume the required operand before testing for a duplicate. A
            // repeated option with no operand is therefore reported as a missing
            // value first.
            let value = tokens.remove(0);
            // Only include and exclude can accumulate multiple values. Repeated
            // scalar options are rejected, so the configuration cannot depend on
            // choosing the first or last spelling.
            if !REPEATABLE.contains(&token.as_str())
                && values.iter().any(|(name, _)| name == &token)
            {
                return Err(cli(
                    format!("unexpected argument {token}"),
                    Some(&token),
                    Some("unexpected_argument"),
                ));
            }
            // Preserve insertion order for repeated glob options. Store positional
            // values separately in their original order even when options appear
            // between them.
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
    // This parser is for embed-jsonl, which takes exactly one input artifact. Both
    // an absent and an extra positional token fail the same argument-count check.
    if flags.positionals.len() != 1 {
        return Err(cli(
            format!("{name} is required"),
            Some(name),
            Some("missing_required_argument"),
        ));
    }
    // Return an owned input filename. Provider and output options will still be
    // read from the same FlagSet after positional validation.
    Ok(flags.positionals[0].clone())
}

// This configuration keeps the requested model and normalized endpoint together
// across batching, artifact generation and verification. It stores the name used
// to find a key, leaving the key itself with the request caller.
struct Provider {
    // Provider kind selecting credential and base-URL defaults and the metadata
    // label.
    kind: ProviderKind,
    // Model sent to the provider and recorded with its returned vectors.
    model: String,
    // Requested vector width checked against returned rows and any existing
    // collection.
    dimensions: u32,
    // Maximum number of source texts grouped into one provider request.
    batch_size: usize,
    // Maximum number of embedding batches run concurrently.
    concurrency: usize,
    // Name of the environment variable from which the credential is read.
    api_key_env: String,
    // Parsed embedding destination, reused for requests and output provenance.
    endpoint: Endpoint,
    // Requested embeddings JSONL output path for the embed command.
    out: String,
    // Optional metadata sidecar destination overriding the default path.
    metadata_out: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    // The built-in provider has a default base URL and credential variable. Its
    // compatible variant requires the caller to name both.
    Openai,
    Compatible,
}

impl ProviderKind {
    // Use stable provider spellings in provenance and validation, independent of
    // any endpoint override. The URL fingerprint is recorded separately.
    fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Compatible => "openai-compatible",
        }
    }
}

fn provider_from_embed_flags(flags: &FlagSet, require_out: bool) -> Result<Provider> {
    // The standalone command supplies provider and model as separate flags.
    // Validate their presence here before entering common request and output
    // configuration.
    let provider_name = required_text(flags.single("--provider"), "--provider")?;
    let kind = parse_provider(&provider_name)?;
    // Distinguish an omitted model from an explicitly empty model for this CLI.
    // Preserve every character of a nonempty model string for the exact
    // provider-echo check.
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
    // Both flag syntaxes eventually validate dimensions, concurrency, environment
    // and endpoint in the same function, so a provider's defaults do not depend on
    // which command selected it.
    finish_provider(flags, kind, model, require_out)
}

fn provider_from_embed_value(flags: &FlagSet, value: &str) -> Result<Provider> {
    // Split only on the first colon. The provider name selects the transport
    // configuration, while any further colons remain part of the model identifier.
    let Some((kind_name, model)) = value.split_once(':') else {
        return Err(cli(
            "--embed must be provider:model",
            Some("--embed"),
            Some("invalid_embed"),
        ));
    };
    // Reject an empty model before parsing the provider label. Keep this distinct
    // from a missing colon so the diagnostic identifies which part of the compound
    // option needs fixing.
    if model.is_empty() {
        return Err(cli(
            "--embed model must not be empty",
            Some("--embed"),
            Some("empty_model"),
        ));
    }
    // Directory import gets its embedding output path from the artifact directory.
    // It shares validation of model, credentials and scheduling with embed-jsonl
    // without requiring --out.
    let kind = parse_provider(kind_name)?;
    finish_provider(flags, kind, model.to_owned(), false)
}

fn finish_provider(
    flags: &FlagSet,
    kind: ProviderKind,
    model: String,
    require_out: bool,
) -> Result<Provider> {
    // Check the numeric width before narrowing to the request's u32 dimension. The
    // same value controls both provider output validation and compatibility with
    // an existing collection.
    let dimensions = required_u64(
        flags.single("--dimensions"),
        "--dimensions",
        1,
        u64::from(u32::MAX),
    )? as u32;
    // Batch size and concurrency are independent. One bounds the number of texts
    // in a request, the other bounds how many such requests can run alongside each
    // other.
    let batch_size = match flags.single("--batch-size") {
        Some(value) => required_u64(Some(value), "--batch-size", 1, 2048)? as usize,
        None => 32,
    };
    let concurrency = match flags.single("--concurrency") {
        Some(value) => required_u64(Some(value), "--concurrency", 1, 8)? as usize,
        None => 1,
    };
    // Keep only the environment variable's name in the Provider. Reading its
    // actual contents is a separate step, so endpoint setup and metadata never
    // need to own the secret.
    let api_key_env = match kind {
        ProviderKind::Openai => match flags.single("--api-key-env") {
            // An explicit environment name still goes through the normal
            // validation even for the built-in provider. Apply the default only
            // when the option was omitted.
            Some(value) => validate_env_name(value)?,
            None => "OPENAI_API_KEY".to_owned(),
        },
        ProviderKind::Compatible => {
            let name = required_text(flags.single("--api-key-env"), "--api-key-env")?;
            validate_env_name(&name)?
        }
    };
    // Choose the base URL from the declared provider. A compatible provider must
    // name its endpoint even when a default exists for the built-in provider.
    let base = match kind {
        ProviderKind::Openai => flags
            .single("--base-url")
            .unwrap_or(OPENAI_DEFAULT_BASE)
            .to_owned(),
        ProviderKind::Compatible => required_text(flags.single("--base-url"), "--base-url")?,
    };
    // Build one normalized endpoint before provider work. Both later transmission
    // and artifact provenance use this object, keeping the destination and its
    // recorded identity in agreement.
    let endpoint = build_endpoint(&base)?;
    // A standalone embeddings artifact needs an explicit output name. Directory
    // import constructs its formal artifact paths later and does not use this
    // field to select its output.
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
        // Retain an explicit metadata output separately from the main embedding
        // path. Its default depends on the caller's artifact layout and is
        // resolved later.
        metadata_out: flags.single("--metadata-out").map(str::to_owned),
    })
}

fn parse_provider(name: &str) -> Result<ProviderKind> {
    // Use the same two provider labels whether supplied through --provider or as
    // the prefix of --embed. Unknown labels cannot silently acquire the built-in
    // provider's defaults.
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
    // Accept an ASCII environment-variable name with a nondigit first character.
    // This argument identifies where to read a credential; it is never treated as
    // a credential value itself.
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
        // Required text preserves its nonempty spelling. A missing option and a
        // present empty string share the missing-field diagnostic here.
        Some(value) if !value.is_empty() => Ok(value.to_owned()),
        Some(_) => Err(missing(field)),
        None => Err(missing(field)),
    }
}

fn required_u64(value: Option<&str>, field: &str, min: u64, max: u64) -> Result<u64> {
    let Some(text) = value else {
        return Err(missing(field));
    };
    // Validate the CLI's decimal spelling in addition to its numeric value. Zero
    // is syntactically valid by itself; the caller's inclusive range decides
    // whether it is meaningful.
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
    // A long sequence of valid decimal digits can still overflow u64. Keep that as
    // an integer-parse failure before applying this option's narrower range.
    let parsed = text.parse::<u64>().map_err(|_| {
        cli(
            format!("{field} must be an integer without a sign or leading zero"),
            Some(field),
            Some("invalid_integer"),
        )
    })?;
    // Check the option-specific bounds only after conversion succeeds. The same
    // unsigned parser serves dimensions, batching and byte limits whose permitted
    // intervals differ.
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
    // Text-only directory scanning has no provider request or metadata sidecar.
    // Reject settings that could otherwise appear to have affected nonexistent
    // embedding work.
    for flag in [
        "--dimensions",
        "--batch-size",
        "--concurrency",
        "--api-key-env",
        "--base-url",
        "--metadata-out",
    ] {
        // These checks run in a fixed option order. If several provider-only flags
        // were given without --embed, report the first one in this list before
        // scanning any input.
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
    // Treat missing, non-Unicode, blank or control-containing environment values
    // as a missing credential. Only the variable name should become part of the
    // diagnostic.
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() && !value.chars().any(char::is_control) => Ok(value),
        _ => Err(missing(name)),
    }
}

fn distinct_outputs(paths: &[&Path]) -> Result<()> {
    let mut seen = Vec::with_capacity(paths.len());
    for path in paths {
        // Compare requested output names before generating any data. Logical
        // normalization can catch alternate spellings for targets that do not
        // exist yet.
        let logical = logical_path(path);
        if seen.iter().any(|item| item == &logical) {
            return Err(cli(
                "output paths must name different files",
                Some("argument"),
                Some("output_path_collision"),
            ));
        }
        // Record each normalized name only after verifying it differs from the
        // earlier entries. This is a collision check among the planned
        // destinations, separate from publication of those paths.
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
    // Normalize lexically because new output files may not exist. This comparison
    // does not ask the filesystem for symlink targets or inode identity.
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
    // Use the working directory as parent for a bare filename. A supplied output
    // parent must be a directory and is never created by this check.
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    // is_dir checks the parent at preflight time. The subsequent file creation or
    // publication can still report an I/O error if the directory changes.
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
    // A user sidecar path is used as given, including a path relative to the
    // working directory. The default instead places metadata beside the embeddings
    // output.
    if let Some(path) = &provider.metadata_out {
        return Ok(PathBuf::from(path));
    }
    let parent = match out_parent {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    // Choose the metadata filename independently of the embeddings filename.
    // Different output basenames in one directory therefore share the same default
    // metadata path unless it is overridden.
    Ok(parent.join("embedding_artifact.json"))
}

fn sidecar_path_for(provider: &Provider, emit_dir: &Path) -> Result<PathBuf> {
    // An explicit sidecar may be outside the directory this command creates. Check
    // that parent now; the default sidecar follows the normal emit-directory
    // setup.
    if let Some(path) = &provider.metadata_out {
        require_parent(Path::new(path), "--metadata-out")?;
        Ok(PathBuf::from(path))
    } else {
        Ok(emit_dir.join("embedding_artifact.json"))
    }
}

struct InputChunk {
    // External chunk ID kept with the provider response at this input position.
    id: String,
    // Text to embed for this chunk, read from the input artifact.
    text: String,
}

fn read_chunk_input(path: &Path) -> Result<(String, Vec<InputChunk>)> {
    let bytes = fs::read(path).map_err(|_| io_ingest("input read failed", None))?;
    // Fingerprint the exact input bytes before stripping a UTF-8 BOM or
    // interpreting lines. The sidecar can then identify the artifact on disk even
    // if two encodings yield the same rows.
    let digest = sha256_hex(&bytes);
    let text =
        std::str::from_utf8(&bytes).map_err(|_| schema("chunks input is not valid jsonl"))?;
    // Accept a single leading byte-order mark for parsing only. An empty file or
    // one containing only that mark has no eligible input for a provider call.
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if text.is_empty() {
        return Err(no_eligible());
    }
    // Require a complete newline-terminated artifact, including its last nonempty
    // row. A valid JSON object at end of file without the newline still fails this
    // input contract.
    if !text.ends_with('\n') {
        return Err(schema("chunks jsonl line is missing a newline"));
    }
    let mut lines: Vec<&str> = text.split('\n').collect();
    // Remove only the artificial empty suffix made by split after the final
    // newline. Empty lines between rows, including an extra trailing blank row,
    // remain visible to validation.
    if lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    if lines.is_empty() {
        return Err(no_eligible());
    }
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    for line in lines {
        // Treat an actual empty line as a malformed row. Its presence cannot be
        // ignored without changing the relationship between file rows and provider
        // result positions.
        if line.is_empty() {
            return Err(schema("chunks jsonl contains an empty line"));
        }
        // Finish validating every row before embedding begins. A bad later row can
        // reject the whole input without an earlier row having been submitted to
        // the provider.
        let value = parse(line).map_err(|_| schema("chunks jsonl is invalid"))?;
        let obj = value
            .as_object()
            .ok_or_else(|| schema("chunks jsonl row must be an object"))?;
        // The external ID must be a nonempty string for stable pairing with the
        // returned vector. Other row fields may be present but only ID and text
        // are carried to the request stage.
        let id = match obj.get("chunk_external_id") {
            Some(JsonValue::String(id)) if !id.is_empty() => id.clone(),
            _ => return Err(schema("chunk_external_id must be a non-empty string")),
        };
        // An empty string is still a text value here, unlike an absent or
        // non-string field. Preserve its exact contents for the request instead of
        // trimming it.
        let text = match obj.get("text") {
            Some(JsonValue::String(text)) => text.clone(),
            _ => return Err(schema("text must be a string")),
        };
        // Keep a set across all input lines rather than only adjacent records. A
        // repeated chunk ID would make two input positions compete for the same
        // artifact identity.
        if !seen.insert(id.clone()) {
            // Classify a repeated key with the external-ID error used by
            // ingestion, retaining the pre-import marker. The diagnostic
            // identifies the field without including chunk text.
            return Err(Error::new(
                ErrorCode::DuplicateExternalId,
                "duplicate chunk_external_id",
            )
            .with_detail("field", "chunk_external_id")
            .ingestion_boundary());
        }
        // Keep file order for batching. Returned provider indices refer to request
        // positions, so later reconstruction of IDs and texts must use this same
        // order.
        rows.push(InputChunk { id, text });
    }
    Ok((digest, rows))
}

struct CollectionShape {
    // Existing collection's vector width for import-configuration checks.
    embedding_dim: usize,
    // Distance metric recorded by the target collection.
    distance_metric: String,
    // Optional model identity already associated with the collection.
    embedding_model: Option<String>,
    // Whether this collection declares its stored vectors normalized.
    vectors_normalized: bool,
}

fn load_collection(path: &Path, name: &str) -> Result<CollectionShape> {
    // symlink_metadata checks the directory entry first. A dangling symlink can
    // pass this step and then fail at read-only open, instead of being treated as
    // an absent catalog here.
    match fs::symlink_metadata(path) {
        // A missing database file and an existing database without this collection
        // both report CollectionNotFound at the ingestion boundary. Other metadata
        // errors keep their original category.
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
    // Inspect the existing collection with a read-only handle during preparation.
    // This call cannot create the missing database as a side effect of checking
    // vector compatibility.
    let db = Database::open_read_only(path).map_err(|error| error.ingestion_boundary())?;
    let info = db
        .list_collections()
        .map_err(|error| error.ingestion_boundary())?
        .into_iter()
        .find(|info| info.name == name);
    match info {
        // Carry just the collection properties needed by the generated manifest.
        // The later core import opens its own writer after staged artifacts have
        // been checked and published.
        Some(info) => Ok(CollectionShape {
            embedding_dim: info.embedding_dim,
            distance_metric: info.distance_metric.as_str().to_owned(),
            embedding_model: info.embedding_model,
            // Retain the collection's normalization flag. The manifest must
            // describe the destination's vector interpretation regardless of the
            // coordinates returned by the provider.
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
        // An existing directory can be used, but it never becomes this
        // invocation's cleanup responsibility. Refuse an existing non-directory
        // before staging any files there.
        if path.is_dir() {
            Ok(None)
        } else {
            Err(io_ingest("output directory creation failed", None))
        }
    } else {
        // Create only the final directory, then return its path as a cleanup
        // token. The caller can later remove it if empty without touching a
        // directory that predated this run.
        fs::create_dir(path).map_err(|_| io_ingest("output directory creation failed", None))?;
        // Only record a directory after create_dir succeeds. If the directory
        // already existed there is no new directory to undo during error cleanup.
        Ok(Some(path.to_owned()))
    }
}

// Emit one document row per source file in the scanner's order. These bytes feed
// both the import bundle and the document digest recorded in the manifest.
fn documents_jsonl(files: &[SourceFile], format: TextFormat) -> ArtifactBytes {
    // Generate strings and keys in a stable order because the published content
    // digest identifies bytes rather than decoded object identity.
    let mut out = String::new();
    for file in files {
        // Use the last slash-separated component for the display title. Preserve
        // the complete relative path for identity and provenance.
        let title = file.relative.rsplit('/').next().unwrap_or(&file.relative);
        out.push_str("{\"external_id\":");
        // Construct the document external ID from the relative path within the
        // input tree. Chunk rows must reference the identical external ID for the
        // importer to associate them.
        push_string(&mut out, &format!("file://{}", file.relative));
        out.push_str(",\"title\":");
        push_string(&mut out, title);
        // Keep source_uri relative to the input root. Moving the whole input tree
        // therefore does not change the recorded source path.
        out.push_str(",\"source_uri\":");
        push_string(&mut out, &file.relative);
        // Expose the input format and path as ordinary document metadata. Both
        // values use the same JSON string escaping as the top-level fields.
        out.push_str(",\"metadata\":{\"format\":");
        push_string(&mut out, format.as_str());
        out.push_str(",\"path\":");
        push_string(&mut out, &file.relative);
        // Terminate every record with a newline, including the last. Hash the
        // encoded bytes in this form so a change to framing also changes the
        // manifest digest.
        out.push_str("}}\n");
    }
    ArtifactBytes::from_string(out)
}

// Emit one chunk row in scanner order. Preserve the decoded UTF-8 span alongside
// the text sent to retrieval.
fn chunks_jsonl(chunks: &[ChunkRecord]) -> ArtifactBytes {
    let mut out = String::new();
    for chunk in chunks {
        // Keep document and chunk external IDs separate. The first resolves the
        // parent during import and the second identifies the chunk itself.
        out.push_str("{\"document_external_id\":");
        push_string(&mut out, &chunk.external_id);
        out.push_str(",\"chunk_external_id\":");
        push_string(&mut out, &chunk.chunk_external_id);
        // Serialize the per-document ordinal independently of the chunk ID. Import
        // uses it to place the chunk without deriving order from the external ID
        // spelling.
        out.push_str(",\"ordinal\":");
        out.push_str(&chunk.ordinal.to_string());
        out.push_str(",\"text\":");
        push_string(&mut out, &chunk.text);
        // The relative path identifies the input whose decoded text supplies this
        // span.
        out.push_str(",\"metadata\":{\"path\":");
        push_string(&mut out, &chunk.relative);
        // Write exact byte offsets in the decoded UTF-8 text. They are independent
        // of the encoded file's original byte positions and of JSON escaping.
        out.push_str("},\"source_span\":{\"start\":");
        push_u64(&mut out, chunk.start);
        out.push_str(",\"end\":");
        push_u64(&mut out, chunk.end);
        // Keep start and end line numbers alongside byte offsets. Line positions
        // aid display while exact byte boundaries preserve source provenance.
        out.push_str(",\"line_start\":");
        out.push_str(&chunk.line_start.to_string());
        out.push_str(",\"line_end\":");
        out.push_str(&chunk.line_end.to_string());
        out.push_str("}}\n");
    }
    // Hash the completed UTF-8 JSONL text as an artifact. The same bytes are
    // written to staging and later verified by the reader.
    ArtifactBytes::from_string(out)
}

// Emit one embedding row for each chunk ID and vector pair already validated by
// the caller. The serialized path uses the canonical binary32 spelling selected by
// the artifact formatter.
fn embeddings_jsonl(
    ids: &[String],
    model: &str,
    dimensions: u32,
    vectors: &[Vec<f32>],
) -> ArtifactBytes {
    let mut out = String::new();
    // Zip in request order. The provider path reorders its response first, so row
    // order and IDs must both match before this formatter runs.
    for (id, vector) in ids.iter().zip(vectors) {
        out.push_str("{\"chunk_external_id\":");
        push_string(&mut out, id);
        // Each row carries the selected model and dimension. Core import can
        // validate a row against the manifest and collection independently of
        // other rows.
        out.push_str(",\"model_name\":");
        push_string(&mut out, model);
        out.push_str(",\"dim\":");
        out.push_str(&dimensions.to_string());
        // The array uses numeric coordinates. The artifact formatter must keep the
        // emitted binary32 spelling compatible with the importer's JSON number
        // rules.
        out.push_str(",\"vector\":[");
        for (index, value) in vector.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(&format_f32(*value));
        }
        // Write exactly one line for the row, including its final newline. The
        // self-check verifies the recorded digest over this exact byte stream.
        out.push_str("]}\n");
    }
    ArtifactBytes::from_string(out)
}

// Build the text-only manifest from sibling artifact names and their exact
// digests. Keep the manifest relocatable with its files by using these relative
// names.
fn text_manifest(
    dataset_id: &str,
    collection: &str,
    documents_sha: &str,
    chunks_sha: &str,
) -> ArtifactBytes {
    // Write schema 0.1 fields in their defined order. Diagnostic output can vary
    // independently of the byte layout consumed by import.
    let mut out = String::from("{\"schema_version\":\"0.1\",\"dataset_id\":");
    push_string(&mut out, dataset_id);
    out.push_str(",\"collection\":");
    push_string(&mut out, collection);
    // Use filenames that publication produces alongside this manifest. Changing
    // either side requires updating the self-check and importer-facing bundle
    // together.
    out.push_str(
        ",\"documents\":\"documents.jsonl\",\"chunks\":\"chunks.jsonl\",\"hashes\":{\"documents_sha256\":",
    );
    // Hash values come from the already encoded JSONL bytes. The manifest records
    // each digest alongside the corresponding file identity.
    push_string(&mut out, documents_sha);
    out.push_str(",\"chunks_sha256\":");
    push_string(&mut out, chunks_sha);
    out.push_str("}}\n");
    // Keep the manifest's own final newline as part of its artifact bytes, even
    // though its hashes refer only to members.
    ArtifactBytes::from_string(out)
}

// Build the vector manifest for a bundle whose document, chunk and embedding files
// have already been produced. Include the import configuration used to interpret
// the vectors.
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
    // Keep the vector manifest's common fields aligned with text_manifest.
    // Embedding support adds a member and interpretation settings to the same
    // artifact family.
    let mut out = String::from("{\"schema_version\":\"0.1\",\"dataset_id\":");
    push_string(&mut out, dataset_id);
    out.push_str(",\"collection\":");
    push_string(&mut out, collection);
    // Name the formal siblings installed by publication and then opened by core
    // import. Temporary writes use separately generated names.
    out.push_str(
        ",\"documents\":\"documents.jsonl\",\"chunks\":\"chunks.jsonl\",\"embeddings\":\"embeddings.jsonl\"",
    );
    // Include the model only when the target collection already has one bound. The
    // embedding rows still carry the selected provider model when this field is
    // omitted.
    if let Some(model) = model {
        out.push_str(",\"embedding_model\":");
        push_string(&mut out, model);
    }
    // Emit dimension, distance metric and normalization policy together. The same
    // vector coordinates can have a different interpretation under a different
    // collection configuration.
    out.push_str(",\"embedding_dim\":");
    out.push_str(&embedding_dim.to_string());
    out.push_str(",\"distance_metric\":");
    push_string(&mut out, distance_metric);
    // Write normalization as a JSON boolean. This is an interpretation flag passed
    // by the caller, so serializing the manifest does not normalize any
    // coordinates.
    out.push_str(",\"vectors_normalized\":");
    out.push_str(if vectors_normalized { "true" } else { "false" });
    // Write the frozen tokenizer metadata used by the artifact. Keep this in sync
    // with the corresponding import schema when a tokenizer field changes.
    out.push_str(",\"tokenizer\":");
    out.push_str(tokenizer_json());
    // Record the digest of each file's exact bytes after all manifest
    // configuration fields. The embedding digest identifies the encoded vectors,
    // not the provider response body.
    out.push_str(",\"hashes\":{\"documents_sha256\":");
    push_string(&mut out, documents_sha);
    out.push_str(",\"chunks_sha256\":");
    push_string(&mut out, chunks_sha);
    out.push_str(",\"embeddings_sha256\":");
    push_string(&mut out, embeddings_sha);
    out.push_str("}}\n");
    ArtifactBytes::from_string(out)
}

// Borrow staging parameters for sidecar emission. The sidecar is provenance for
// the generated vectors and is separate from the manifest consumed by core import.
struct SidecarInput<'a> {
    // Provider identifier to record for this HTTP embedding run.
    provider: &'a str,
    // Requested model name, carried alongside the embeddings artifact.
    model: &'a str,
    // Requested vector width recorded in both dimension fields of the sidecar.
    dimensions: u32,
    // Source chunks path or label recorded for this generation.
    embedding_input: &'a str,
    // Timestamp recorded for this set of generated vectors.
    generated_at: &'a str,
    // Configured request batch size published in the sidecar.
    batch_size: usize,
    // Digest binding this generation to the source chunks bytes.
    chunks_sha256: &'a str,
    // Digest of the exact embeddings output paired with the sidecar.
    embeddings_sha256: &'a str,
    // Parsed endpoint supplying the host label and the normalized URL used for the
    // fingerprint.
    endpoint: &'a Endpoint,
}

// Serialize the HTTP provenance associated with one generated embeddings file.
// Emit only the fields listed here rather than copying request or response
// structures.
fn sidecar_json(input: &SidecarInput<'_>) -> ArtifactBytes {
    // Use the configured provider label and the transport family as distinct
    // fields. Two provider labels can share the same HTTP transport.
    let mut out = String::from("{\"embedding_provider\":");
    push_string(&mut out, input.provider);
    out.push_str(",\"embedding_provider_kind\":\"http\",\"embedding_model\":");
    push_string(&mut out, input.model);
    // Record both the artifact's embedding dimension and the requested dimension.
    // The generator validates every returned vector against the same configured
    // width.
    out.push_str(",\"embedding_dim\":");
    out.push_str(&input.dimensions.to_string());
    out.push_str(",\"embedding_dimensions_requested\":");
    out.push_str(&input.dimensions.to_string());
    // Record the selected input and generation time alongside provider and model.
    // The timestamp identifies this production run, independent of the source
    // bytes' hashes.
    out.push_str(",\"embedding_input\":");
    push_string(&mut out, input.embedding_input);
    out.push_str(",\"embedding_generated_at\":");
    push_string(&mut out, input.generated_at);
    // Record the configured maximum batch size. The final request can contain
    // fewer items and this field remains the original setting.
    out.push_str(",\"embedding_batch_size\":");
    out.push_str(&input.batch_size.to_string());
    // Bind the sidecar to both chunks and embeddings bytes. Reordering rows or
    // changing the text changes the appropriate digest.
    out.push_str(",\"chunks_sha256\":");
    push_string(&mut out, input.chunks_sha256);
    out.push_str(",\"embeddings_sha256\":");
    push_string(&mut out, input.embeddings_sha256);
    // Expose the endpoint host and a digest of the normalized URL. The sidecar
    // contains neither the bearer credential nor the full endpoint URL.
    out.push_str(",\"endpoint_host\":");
    push_string(&mut out, &input.endpoint.host);
    out.push_str(",\"endpoint_fingerprint\":");
    // Hash the normalized operation URL supplied by endpoint construction. Base
    // URLs that resolve to the same normalized operation carry the same recorded
    // fingerprint.
    push_string(&mut out, &sha256_hex(input.endpoint.url.as_bytes()));
    out.push_str("}\n");
    ArtifactBytes::from_string(out)
}

// Record a UTC timestamp at whole-second precision. Use the Unix epoch as a
// fallback if the current clock precedes it.
fn generated_at() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    crate::time::format_unix_millis(seconds.saturating_mul(1000))
}

// Run provider requests in bounded batches and return one vector per text in input
// order. This function collects results before the caller can pair them with IDs
// and write artifacts.
fn embed_all(
    endpoint: &Endpoint,
    key: &str,
    model: &str,
    dimensions: u32,
    batch_size: usize,
    concurrency: usize,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    // Partition the source into borrowed slices so workers share the original text
    // order without cloning strings into a queue.
    let batches: Vec<&[String]> = texts.chunks(batch_size).collect();
    // With no input there is no provider request to make. Returning an empty list
    // lets the caller decide whether an empty artifact is valid.
    if batches.is_empty() {
        return Ok(Vec::new());
    }
    // Use sequential requests when there is only one batch or no parallelism
    // requested. Appending each completed batch preserves input order without
    // result slots.
    if concurrency <= 1 || batches.len() == 1 {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in batches {
            // Use the same timeout and shape validation for sequential batches as
            // for the worker path below. Return the first sequential error to the
            // caller.
            vectors.extend(embed_batch(BatchRequest {
                endpoint,
                key,
                model,
                dimensions,
                // Pass a borrowed contiguous batch unchanged to the transport.
                // Response indices therefore refer to the same local positions the
                // caller uses when flattening.
                inputs: batch,
                timeout: REQUEST_TIMEOUT,
            })?);
        }
        return Ok(vectors);
    }
    // Requests can finish out of order. Each result retains its input slot so
    // flattening pairs vectors with the correct chunk IDs. On failure, already
    // claimed requests may finish; report the earliest input batch among the
    // recorded errors after the workers have joined.
    // The atomic counter dispenses unique batch indices. It can use relaxed
    // ordering because the input list is immutable and workers publish other state
    // through their own primitives.
    let next = AtomicUsize::new(0);
    // Use the stop flag only to avoid starting more work. It does not cancel
    // requests that another worker already sent.
    let stop = AtomicBool::new(false);
    // Keep indexed errors and indexed results separately. Arrival order from
    // either mutex is not the order used for the final result.
    let errors = Mutex::new(Vec::<(usize, Error)>::new());
    let outputs = Mutex::new(vec![None; batches.len()]);
    // Scoped threads borrow endpoint settings, inputs and output slots. The scope
    // waits for all workers before the caller inspects those collections.
    thread::scope(|scope| {
        // Bound the number of workers by the number of available batches. More
        // workers could only contend for empty work slots.
        for _ in 0..concurrency.min(batches.len()) {
            scope.spawn(|| loop {
                // Check for a reported failure before claiming another batch. The
                // second check after claiming also avoids starting a request when
                // a worker has failed in the meantime.
                if stop.load(Ordering::Acquire) {
                    break;
                }
                let index = next.fetch_add(1, Ordering::Relaxed);
                // A claimed index can be beyond the final batch or unused after
                // stop. The final collection therefore checks every slot instead
                // of assuming all claims became results.
                if index >= batches.len() || stop.load(Ordering::Acquire) {
                    break;
                }
                // The index counter governs ownership of a request slot, while
                // stop is a separate advisory signal. It can change as another
                // request finishes.
                match embed_batch(BatchRequest {
                    endpoint,
                    key,
                    model,
                    dimensions,
                    inputs: batches[index],
                    timeout: REQUEST_TIMEOUT,
                }) {
                    Ok(vectors) => {
                        // Store each completed response by the input batch index.
                        // Even if all requests succeed, completion order can
                        // differ from the order needed for the artifact.
                        if let Ok(mut slot) = outputs.lock() {
                            slot[index] = Some(vectors);
                        }
                    }
                    // Publish the stop flag before recording the error so other
                    // workers can notice it as soon as possible. A worker already
                    // inside embed_batch may still complete.
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
    // Consider work complete only after every scoped thread returned. Choose the
    // lowest index among errors actually recorded, regardless of which arrival set
    // stop.
    let errors = errors.into_inner().unwrap_or_default();
    if let Some((_, error)) = errors.into_iter().min_by_key(|(index, _)| *index) {
        return Err(error);
    }
    // Move result slots out after joining. Each slot is consumed once so the final
    // vector can reuse the per-batch allocations' contents.
    let outputs = outputs.into_inner().unwrap_or_default();
    let mut vectors = Vec::with_capacity(texts.len());
    // Flatten by original batch index and preserve each batch's response order. A
    // missing slot after no recorded error still prevents publishing a partial
    // embedding set.
    for batch in outputs {
        vectors.extend(batch.ok_or_else(|| io_ingest("provider request failed", Some("request")))?);
    }
    // Check the assembled count against the input length before returning. Later
    // artifact formatting zips IDs and vectors and relies on exact alignment.
    if vectors.len() != texts.len() {
        return Err(vector_length(
            "provider response count does not match the request",
        ));
    }
    Ok(vectors)
}

// Read the staged text bundle back before using it in import. Verify byte
// identity, manifest wiring and basic row decoding as distinct checks.
fn self_check_text(
    documents: &Path,
    chunks: &Path,
    manifest: &Path,
    collection: &str,
    documents_sha: &str,
    chunks_sha: &str,
) -> Result<()> {
    // Read the actual staging files. This catches emission errors before
    // publication becomes possible.
    let documents_bytes = read_nonempty(documents)?;
    let chunks_bytes = read_nonempty(chunks)?;
    let manifest_bytes = read_nonempty(manifest)?;
    // Verify the document and chunk digests over the raw files. UTF-8 conversion
    // and row parsing are performed later so they cannot mask a byte mismatch.
    if sha256_hex(&documents_bytes) != documents_sha || sha256_hex(&chunks_bytes) != chunks_sha {
        return Err(schema("artifact hash self-check failed"));
    }
    // Decode and parse the manifest before inspecting individual fields. Reject
    // malformed content with the artifact self-check classification.
    let manifest_text =
        std::str::from_utf8(&manifest_bytes).map_err(|_| schema("artifact self-check failed"))?;
    let value = parse(manifest_text).map_err(|_| schema("artifact self-check failed"))?;
    // The expected collection came from the staging request. Check it in the
    // manifest before allowing core import to resolve that target.
    if value.get("collection").and_then(JsonValue::as_str) != Some(collection) {
        return Err(schema("artifact self-check failed"));
    }
    // Compare the two relative member names with the fixed bundle layout. Matching
    // content hashes alone would not establish which files the manifest points to.
    if value.get("documents").and_then(JsonValue::as_str) != Some("documents.jsonl")
        || value.get("chunks").and_then(JsonValue::as_str) != Some("chunks.jsonl")
    {
        return Err(schema("artifact self-check failed"));
    }
    // Read the recorded hashes as a JSON object and require them to match the
    // requested digests. This checks the binding between the manifest and already
    // verified file bytes.
    let hashes = value
        .get("hashes")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| schema("artifact self-check failed"))?;
    // The expected digests have already been compared against the files. A
    // mismatch in the manifest therefore indicates wrong recorded metadata even
    // when both member files are intact.
    if hashes.get("documents_sha256").and_then(JsonValue::as_str) != Some(documents_sha)
        || hashes.get("chunks_sha256").and_then(JsonValue::as_str) != Some(chunks_sha)
    {
        return Err(schema("artifact self-check failed"));
    }
    // Parse the document rows even though only chunk text requires a separate
    // nonempty check here. Import performs the complete row schema validation
    // afterward.
    let _ = jsonl_values(&documents_bytes)?;
    for row in jsonl_values(&chunks_bytes)? {
        // The generator must not emit a chunk with missing or whitespace-only
        // text. Check the parsed field rather than relying on the number of bytes
        // in a JSONL row.
        let text = row.get("text").and_then(JsonValue::as_str).unwrap_or("");
        if text.trim().is_empty() {
            return Err(schema("artifact self-check failed"));
        }
    }
    Ok(())
}

// Verify a generated embeddings file against its digest and requested row
// sequence. The check catches a complete file with the right count but the wrong
// association between chunks and vectors.
fn self_check_embeddings(
    path: &Path,
    ids: &[String],
    model: &str,
    dimensions: u32,
    expected_sha: &str,
) -> Result<()> {
    let bytes = read_nonempty(path)?;
    // Check raw bytes before validating row content. The digest represents the
    // exact embedding artifact, including formatting and final newlines.
    if sha256_hex(&bytes) != expected_sha {
        return Err(schema("artifact hash self-check failed"));
    }
    let rows = jsonl_values(&bytes)?;
    // Compare lengths before zipping rows with IDs. An unmatched suffix on either
    // side must fail instead of being silently ignored.
    if rows.len() != ids.len() {
        return Err(schema("artifact self-check failed"));
    }
    // Verify each row against its corresponding requested ID, the configured model
    // and dimension. A row reordered with its vector still violates the fixed byte
    // and ordering contract.
    for (row, id) in rows.iter().zip(ids) {
        // Compare the external ID as a string exactly as generated. A hash alone
        // does not explain which logical row association the artifact is expected
        // to preserve.
        if row.get("chunk_external_id").and_then(JsonValue::as_str) != Some(id.as_str())
            || row.get("model_name").and_then(JsonValue::as_str) != Some(model)
            || row.get("dim").and_then(JsonValue::as_f64) != Some(f64::from(dimensions))
        {
            return Err(schema("artifact self-check failed"));
        }
        // Check that the declared dimension agrees with the encoded vector array
        // length. The provider already validated coordinate values during
        // generation, and the digest protects the staged bytes here.
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

// Verify the vector-specific fields of the staged manifest before import. The
// text-bundle check handles the document and chunk fields separately.
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
    // Confirm target collection before validating the vector layout. A correct
    // vector file must still be imported into the intended collection.
    if value.get("collection").and_then(JsonValue::as_str) != Some(collection) {
        return Err(schema("artifact self-check failed"));
    }
    // Compare dimension against the requested width as a number. A string
    // containing those digits does not match the manifest contract.
    if value.get("embedding_dim").and_then(JsonValue::as_f64) != Some(f64::from(dimensions)) {
        return Err(schema("artifact self-check failed"));
    }
    // Require the exact relative embedding filename used by the bundle. The digest
    // alone cannot ensure the importer opens the intended artifact.
    if value.get("embeddings").and_then(JsonValue::as_str) != Some("embeddings.jsonl") {
        return Err(schema("artifact self-check failed"));
    }
    // Confirm that the manifest binds the staged vector file's digest under the
    // embeddings-specific hash field.
    let hashes = value
        .get("hashes")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| schema("artifact self-check failed"))?;
    if hashes.get("embeddings_sha256").and_then(JsonValue::as_str) != Some(embeddings_sha) {
        return Err(schema("artifact self-check failed"));
    }
    Ok(())
}

// Check retained provenance against the provider settings and both artifact
// digests. This sidecar is checked independently of the importer-facing manifest.
fn self_check_sidecar(
    path: &Path,
    expected_sha: &str,
    provider: &Provider,
    embedding_input: &str,
    chunks_sha: &str,
    embeddings_sha: &str,
) -> Result<()> {
    let bytes = read_nonempty(path)?;
    // First verify the sidecar's exact bytes, then decode its assertions. This
    // keeps accidental corruption separate from inconsistent provenance.
    if sha256_hex(&bytes) != expected_sha {
        return Err(schema("artifact hash self-check failed"));
    }
    let value =
        parse(std::str::from_utf8(&bytes).map_err(|_| schema("artifact self-check failed"))?)
            .map_err(|_| schema("artifact self-check failed"))?;
    // Read the generation timestamp as text for the format check below. A missing
    // or non-string value becomes empty and fails the same check.
    let timestamp = value
        .get("embedding_generated_at")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    // Verify that provider identity, transport kind and requested model describe
    // the actual staging configuration. These values come from the selected
    // Provider, not arbitrary response fields.
    if value.get("embedding_provider").and_then(JsonValue::as_str) != Some(provider.kind.as_str())
        || value
            .get("embedding_provider_kind")
            .and_then(JsonValue::as_str)
            != Some("http")
        || value.get("embedding_model").and_then(JsonValue::as_str) != Some(provider.model.as_str())
        // Check both dimension fields against the requested width. The producer
        // writes them from the same source and the sidecar preserves both
        // meanings.
        || value.get("embedding_dim").and_then(JsonValue::as_f64)
            != Some(f64::from(provider.dimensions))
        || value
            .get("embedding_dimensions_requested")
            .and_then(JsonValue::as_f64)
            != Some(f64::from(provider.dimensions))
        // Tie the source label and configured batch size back to the request. The
        // check is against configuration, not the number of rows in the last
        // batch.
        || value.get("embedding_input").and_then(JsonValue::as_str) != Some(embedding_input)
        || value
            .get("embedding_batch_size")
            .and_then(JsonValue::as_f64)
            != Some(provider.batch_size as f64)
        // Validate both byte digests so this sidecar names its exact input chunks
        // and output embeddings.
        || value.get("chunks_sha256").and_then(JsonValue::as_str) != Some(chunks_sha)
        || value.get("embeddings_sha256").and_then(JsonValue::as_str) != Some(embeddings_sha)
        // Compare the recorded host and recreate the digest from the normalized
        // endpoint. This verifies endpoint provenance without requiring the full
        // URL in the artifact.
        || value.get("endpoint_host").and_then(JsonValue::as_str)
            != Some(provider.endpoint.host.as_str())
        || value
            .get("endpoint_fingerprint")
            .and_then(JsonValue::as_str)
            != Some(sha256_hex(provider.endpoint.url.as_bytes()).as_str())
        // Check for the generated whole-second UTC form by length, terminal Z and
        // absence of fractional punctuation. This is a format check, not a
        // timestamp parser.
        || timestamp.len() != 20
        || !timestamp.ends_with('Z')
        || timestamp.contains('.')
    {
        return Err(schema("artifact self-check failed"));
    }
    Ok(())
}

// Read the artifact under the staging error category. An empty file cannot satisfy
// these generated-bundle self-checks even if it is readable.
fn read_nonempty(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|_| schema("artifact self-check failed"))?;
    if bytes.is_empty() {
        return Err(schema("artifact self-check failed"));
    }
    Ok(bytes)
}

// Decode the full artifact as UTF-8 and parse records one line at a time. Return
// the existing self-check error category for byte or syntax failures.
fn jsonl_values(bytes: &[u8]) -> Result<Vec<JsonValue>> {
    let text = std::str::from_utf8(bytes).map_err(|_| schema("artifact self-check failed"))?;
    // Require the line terminator written by the generator. A parseable final
    // object still fails this check when its expected newline is missing.
    if !text.ends_with('\n') {
        return Err(schema("artifact self-check failed"));
    }
    let mut rows = Vec::new();
    // Skip empty segments, including the one created by the final newline.
    // Nonempty lines must parse as complete JSON values.
    for line in text.split('\n') {
        // Only empty segments are skipped here. A whitespace-only segment is
        // passed to the normal parser and can fail the artifact check.
        if line.is_empty() {
            continue;
        }
        rows.push(parse(line).map_err(|_| schema("artifact self-check failed"))?);
    }
    // Reject a file containing only delimiters. The minimum is one decoded value,
    // not merely nonzero input bytes.
    if rows.is_empty() {
        return Err(schema("artifact self-check failed"));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests;

// Build the command summary for either artifact generation or completed import.
// The caller supplies the counts so this formatter does not infer import effects
// from filenames.
fn command_result(
    documents: usize,
    chunks: usize,
    embeddings: usize,
    skipped: &[SkippedFile],
    encoding: Option<InputEncoding>,
) -> JsonValue {
    // The same summary schema can describe text-only and vector-enabled
    // operations. A caller passes zero for stages that did not produce that record
    // type.
    crate::json::object_from_pairs([
        ("documents", JsonValue::Number(documents as f64)),
        ("chunks", JsonValue::Number(chunks as f64)),
        ("embeddings", JsonValue::Number(embeddings as f64)),
        // Keep skipped files as individually attributed results. The optional
        // selected input encoding allows each entry to report its own decoding
        // context.
        (
            "skipped_files",
            JsonValue::Array(skipped.iter().map(|file| file.to_json(encoding)).collect()),
        ),
    ])
}
