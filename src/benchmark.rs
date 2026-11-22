//! A benchmark dataset has two kinds of identity. The manifest and artifact
//! digests identify input bytes. Query, document and chunk IDs connect records
//! across those byte streams. A correctly hashed relevance judgment still needs to
//! name a chunk in this corpus.
//!
//! Validation builds the workload in query-file order. It retains the query inputs
//! and their positive judgments for repeated searches, while the corpus files
//! remain inputs to the generated import manifest. Inline and sidecar describe the
//! origin of a query vector. The runner selects the retrieval modes separately.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::error::{Error, ErrorCode, Result};
use crate::hash::{sha256_reader, Sha256Reader};
use crate::json::{
    parse_reader_in, required_string, required_usize, JsonBuilder, JsonInputKind, JsonLines,
    JsonObject, JsonOperationContext, JsonReservation, JsonSchema, JsonValue,
};
use crate::metadata::FilterExpr;
use crate::model::{json_usize, QueryEmbeddingInput};

mod query_mode;
use query_mode::BenchmarkQueryMode;
mod result;
pub(crate) use result::validate_benchmark_result_contract;
mod runner;
pub(crate) use runner::{run_benchmark, BenchmarkOptions};

// Allowed row fields are specific to each benchmark artifact. A query-side vector
// identifies a query, while a corpus vector identifies a chunk, even though both
// carry model, dimension and coordinates.
const BENCHMARK_DOCUMENT_FIELDS: &[&str] = &["external_id", "title", "source_uri", "metadata"];
const BENCHMARK_CHUNK_FIELDS: &[&str] = &[
    "document_external_id",
    "chunk_external_id",
    "ordinal",
    "text",
    "metadata",
    "source_span",
];
const BENCHMARK_EMBEDDING_FIELDS: &[&str] = &["chunk_external_id", "model_name", "dim", "vector"];
// The query file can carry an inline vector and a top-k token. Its row parser
// validates top-k from the source number token and preserves the resulting
// integer; a corpus embedding row instead carries its own model and declared
// dimension.
const BENCHMARK_QUERY_FIELDS: &[&str] = &["query_id", "text", "filter", "top_k", "embedding"];
const BENCHMARK_QREL_FIELDS: &[&str] = &["query_id", "chunk_external_id", "relevance"];
const BENCHMARK_QUERY_EMBEDDING_FIELDS: &[&str] = &["query_id", "model_name", "dim", "vector"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// One dataset chooses one source of query embeddings for all rows. The mode
// determines which paths and hashes are required in the manifest and how each
// retained query gets its vector.
pub enum QueryEmbeddingSource {
    // Match query IDs to vectors read from a separate query-embedding artifact.
    Sidecar,
    // Take each query's vector from its own JSONL row.
    Inline,
    // Run queries without embeddings and reject conflicting query-vector inputs.
    None,
}

#[derive(Clone, Debug)]
pub struct BenchmarkDatasetManifest {
    // Tokenizer configuration used when constructing the import manifest. Omitted
    // manifest settings use the default tokenizer.
    pub tokenizer: crate::TokenizerConfig,
    // Dataset identifier reported together with its declared version and artifact
    // digests.
    pub dataset_id: String,
    // Producer's version of the dataset, independent of the manifest schema
    // version.
    pub version: String,
    // Chunking description carried into the dataset identity. A missing description
    // is represented by JSON null.
    pub chunking: JsonValue,
    // Model name declared for corpus embeddings, when supplied.
    pub embedding_model: Option<String>,
    // Corpus vector width selected by dataset validation and used to configure the
    // benchmark collection.
    pub embedding_dim: usize,
    // Digest of the exact documents artifact bytes.
    pub documents_sha256: String,
    // Digest checked again when a reference pass reopens the chunk source.
    pub chunks_sha256: String,
    // Verified digest of the corpus embedding artifact when a hash was supplied.
    pub embeddings_sha256: Option<String>,
    // Digest of query rows, whose file order determines the order of samples.
    pub queries_sha256: String,
    // Digest of the relevance judgements that supply the quality labels.
    pub qrels_sha256: String,
    // Selects inline, sidecar or absent query vectors for all queries in the
    // dataset.
    pub query_embedding_source: QueryEmbeddingSource,
    // Model identity declared for query vectors, distinct from the corpus model
    // field.
    pub query_embedding_model: Option<String>,
    // Declared query-vector width when the manifest specifies one.
    pub query_embedding_dim: Option<usize>,
    // Verified digest of the separate query-vector artifact when present.
    pub query_embeddings_sha256: Option<String>,
    // Digest of the manifest file's original bytes, including its original
    // whitespace.
    pub dataset_manifest_sha256: String,
}

#[derive(Clone, Debug)]
// Keep the inputs needed for repeated scoring with the validated manifest and
// queries. Corpus inputs can be reopened from their resolved paths while the query
// and relevance data remain in memory.
pub struct BenchmarkDataset {
    // Validated dataset identity and index settings used by result construction and
    // import.
    pub manifest: BenchmarkDatasetManifest,
    // Resolved documents artifact path used by import and source-based
    // verification.
    pub documents_path: PathBuf,
    // Resolved chunk source reopened for scoring and for the baseline.
    pub chunks_path: PathBuf,
    // Resolved path to corpus vectors, when the dataset names an embeddings
    // artifact.
    pub embeddings_path: Option<PathBuf>,
    // Queries in their artifact order, joined to any sidecar vectors and positive
    // relevance judgements.
    pub queries: Vec<BenchmarkQuery>,
    // Manifest directory used to recover the source-relative path in benchmark
    // errors.
    base_dir: PathBuf,
    // Manifest path spelling included alongside artifact and line details in
    // diagnostics.
    manifest_path_label: String,
}

#[derive(Clone, Debug)]
pub struct BenchmarkQuery {
    // Source identifier used to join query embeddings and relevance judgements.
    pub query_id: String,
    // Text supplied to BM25-capable benchmark modes.
    pub text: Option<String>,
    // Validated metadata predicate to apply when selecting eligible chunks.
    pub filter: JsonValue,
    // Per-query maximum number of results to return and compare.
    pub top_k: usize,
    // Vector input resolved according to the dataset's single query-embedding
    // source.
    pub embedding: Option<QueryEmbeddingInput>,
    // External chunk IDs with positive relevance for this query. These are the
    // labels for recall and reciprocal rank.
    pub relevant_chunk_ids: BTreeSet<String>,
}

struct BenchmarkManifestInput {
    // Manifest file path retained in the list of input paths protected from result
    // publication.
    manifest_path: PathBuf,
    // Printable manifest path for errors raised before a dataset exists.
    // The manifest's source path serves I/O, whereas its requested label is
    // retained for diagnostics. Passing a directory as the dataset is why these
    // two spellings can differ.
    manifest_path_label: String,
    // Directory against which paths written in the manifest are resolved.
    base_dir: PathBuf,
    // SHA-256 computed while reading the manifest, before converting it into typed
    // fields.
    manifest_sha256: String,
    // Parsed manifest fields retained while related artifact declarations are
    // checked together.
    obj: JsonObject,
    // Source mode parsed before the checks for required and conflicting embedding
    // inputs.
    query_embedding_source: QueryEmbeddingSource,
    // Shared operation context for manifest parsing, artifact row conversion and
    // retained data.
    context: JsonOperationContext,
    // Reservation for the parsed manifest. Its retained portion follows data moved
    // into the final dataset.
    work: JsonReservation,
}

#[derive(Clone, Debug)]
struct Artifact {
    // Manifest spelling of the artifact path, used for source-facing diagnostics.
    raw_path: String,
    // Resolved filesystem path opened by digest and row readers.
    path: PathBuf,
    // Artifact category selecting the row schema, allowed fields and error label.
    kind: &'static str,
    // Manifest hash key corresponding to this artifact's raw bytes.
    hash_field: &'static str,
}

#[derive(Clone, Debug)]
struct QueryRow {
    // Join key shared with query-embedding and relevance rows.
    query_id: String,
    // One-based source line for reporting failures detected after reading the
    // query.
    line: usize,
    // Parsed text retained until the validated query is assembled.
    text: Option<String>,
    // Filter object after its predicate syntax and JSON limits were checked.
    filter: JsonValue,
    // Requested result count decoded from the query row.
    top_k: usize,
    // Inline vector held until the manifest's source-mode contract has been
    // checked.
    embedding: Option<QueryEmbeddingInput>,
    // Whether the original query row supplied a non-null embedding. Used to enforce
    // the source-mode rules.
    has_embedding: bool,
}

#[derive(Clone, Debug)]
struct SidecarRow {
    // First source line for this sidecar ID, retained to diagnose a duplicate.
    line: usize,
}

impl QueryEmbeddingSource {
    // These are the manifest and result wire spellings. Report the same source
    // choice that governed row validation rather than deriving it from the number
    // of vectors present.
    pub fn as_str(self) -> &'static str {
        match self {
            QueryEmbeddingSource::Sidecar => "sidecar",
            QueryEmbeddingSource::Inline => "inline",
            QueryEmbeddingSource::None => "none",
        }
    }
}

impl BenchmarkDatasetManifest {
    #[cfg(test)]
    pub fn load_and_validate(path: impl AsRef<Path>) -> Result<Self> {
        Ok(BenchmarkDataset::load_and_validate(path)?.manifest)
    }

    // Build the dataset section of a benchmark result from the manifest that was
    // actually validated. Include the exact raw artifact digests alongside the
    // declared version and preprocessing settings.
    pub(crate) fn build_result_dataset_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("dataset_id", builder.string(&self.dataset_id)?),
            ("tokenizer", self.tokenizer.build_json(builder)?),
            ("version", builder.string(&self.version)?),
            // The retained manifest continues to own its chunking value. Copy it
            // with the result's builder so the report can own its own nested data
            // and reservation.
            ("chunking", builder.copy_validated_value(&self.chunking)?),
            // Allow a missing corpus model to appear as null. This field describes
            // the dataset's declaration and is independent of whether timed
            // queries use text, vectors or both.
            (
                "embedding_model",
                builder.optional_str(self.embedding_model.as_deref())?,
            ),
            (
                "embedding_dim",
                JsonValue::Number(self.embedding_dim as f64),
            ),
            (
                // Include the digest of the original manifest itself alongside the
                // member digests. Editing only manifest metadata can change that
                // identity even when none of its artifact files changed.
                "dataset_manifest_sha256",
                builder.string(&self.dataset_manifest_sha256)?,
            ),
            // Corpus documents and chunks have independent content identities.
            // Keep their digests separate from both the dataset manifest's
            // checksum and any optional embedding artifact.
            ("documents_sha256", builder.string(&self.documents_sha256)?),
            ("chunks_sha256", builder.string(&self.chunks_sha256)?),
            (
                "embeddings_sha256",
                builder.optional_str(self.embeddings_sha256.as_deref())?,
            ),
            ("queries_sha256", builder.string(&self.queries_sha256)?),
            ("qrels_sha256", builder.string(&self.qrels_sha256)?),
            // Record how query vectors were supplied along with their optional
            // model, dimension and separate file hash. An inline dataset can
            // identify its query embeddings without a sidecar digest.
            (
                "query_embedding_source",
                builder.string(self.query_embedding_source.as_str())?,
            ),
            (
                "query_embedding_model",
                builder.optional_str(self.query_embedding_model.as_deref())?,
            ),
            (
                "query_embedding_dim",
                // A missing query-vector width is different from a zero-width
                // vector. The validation path requires any declared width to be
                // positive, so retain absence as JSON null.
                self.query_embedding_dim
                    .map(|dim| JsonValue::Number(dim as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "query_embeddings_sha256",
                builder.optional_str(self.query_embeddings_sha256.as_deref())?,
            ),
        ])
    }
}

impl BenchmarkDataset {
    // Translate import-facing diagnostic paths back to dataset inputs where
    // possible. The evaluator may have generated a private import manifest, so
    // arbitrary staging paths are not useful file locators in a benchmark report.
    fn locate_input_error(&self, error: Error) -> Error {
        // Only path detail fields are remapped. Keep the original error category
        // and other diagnostics intact so validation and runtime failures remain
        // distinguishable.
        let mut error = error.map_path_details(|reported| {
            if reported == self.manifest_path_label {
                return Some(reported.to_owned());
            }
            // Only the documents, chunks and optional corpus embeddings feed the
            // generated import manifest. Queries, qrels and query-side vectors are
            // read by the benchmark validators and already receive locations from
            // their own artifact descriptors.
            for path in [
                Some(&self.documents_path),
                Some(&self.chunks_path),
                self.embeddings_path.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                // Accept both the resolved path used to open the artifact and a
                // canonical spelling returned by lower layers. Failure to
                // canonicalize should still leave the direct path comparison
                // usable.
                if Path::new(reported) == path
                    || fs::canonicalize(path).ok().as_deref() == Some(Path::new(reported))
                {
                    return Some(
                        // Prefer paths relative to the manifest directory where
                        // possible. For a source outside that directory, keep the
                        // resolved path's spelling instead of inventing a
                        // misleading relative name.
                        path.strip_prefix(&self.base_dir)
                            .unwrap_or(path)
                            .display()
                            .to_string(),
                    );
                }
            }
            // Generated manifests and private staging paths have no public locator.
            None
        });
        // A lower layer may already have identified the dataset manifest. Fill the
        // public manifest locator only if it was absent rather than appending a
        // second value for the same key.
        if !error.detail().iter().any(|(key, _)| key == "manifest_path") {
            error = error.with_detail("manifest_path", &self.manifest_path_label);
        }
        error
    }

    #[cfg(test)]
    pub fn load_and_validate(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_and_validate_in(path, &JsonOperationContext::standalone())
            .map(|(dataset, _)| dataset)
    }

    #[cfg(test)]
    pub(crate) fn load_and_validate_in(
        path: impl AsRef<Path>,
        context: &JsonOperationContext,
    ) -> Result<(Self, JsonReservation)> {
        Self::load_with_sources_in(path, context).map(|(dataset, work, _)| (dataset, work))
    }

    // The caller retains the returned reservation while the dataset is in use. The
    // separate path list includes validated input files even when their rows have
    // already been folded into queries in memory.
    pub(crate) fn load_with_sources_in(
        path: impl AsRef<Path>,
        context: &JsonOperationContext,
    ) -> Result<(Self, JsonReservation, Vec<PathBuf>)> {
        // Validation has to finish before the runner opens or creates its target
        // database. A failure here leaves no benchmark database work to undo.
        // Paths returned on success also let the runner reject a result file that
        // aliases a dataset input.
        BenchmarkManifestInput::load(path, context)?.validate_dataset()
    }
}

impl BenchmarkManifestInput {
    fn load(path: impl AsRef<Path>, context: &JsonOperationContext) -> Result<Self> {
        let requested_path = path.as_ref();
        // Accept a manifest file or a dataset directory. Resolve sibling resources
        // relative to the actual manifest path, but preserve the originally
        // requested spelling as the diagnostic label.
        let manifest_path = if requested_path.is_dir() {
            requested_path.join("dataset_manifest.json")
        } else {
            requested_path.to_path_buf()
        };
        let manifest_path_label = requested_path.display().to_string();
        // Hash the manifest through the same reader that supplies the parser. The
        // recorded identity covers its original bytes, including whitespace, and
        // cannot drift because of a second open.
        let mut source = Sha256Reader::new(File::open(&manifest_path)?);
        let (mut value, work) = parse_reader_in(
            &mut source,
            JsonSchema::Value(JsonInputKind::Manifest),
            context,
        )?
        .into_parts();
        let manifest_sha256 = source.finish();
        // Take the parsed object's map before dropping its wrapper. Its memory is
        // still covered by work while the manifest fields are checked and any
        // retained members are moved out.
        let obj = value.as_object_mut().map(std::mem::take).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "dataset manifest must be a JSON object",
            )
            .with_detail("manifest_path", &manifest_path_label)
        })?;
        // Reject unknown top-level settings before interpreting the selected
        // source mode. The nested hashes object has its own closed set of member
        // names checked separately.
        reject_unknown_benchmark_manifest_fields(&obj, &manifest_path_label)?;
        // Validate the declared hash fields when the member exists and is
        // non-null. The mandatory-presence check is separate so an absent member
        // can receive its required-field diagnostic.
        if let Some(hashes) = obj.get("hashes").filter(|value| !value.is_null()) {
            let hashes = hashes.as_object().ok_or_else(|| {
                Error::new(ErrorCode::InvalidJsonlSchema, "hashes must be an object")
                    .with_detail("field", "hashes")
                    .with_detail("manifest_path", &manifest_path_label)
            })?;
            // A known hash key and a value of the expected type are separate
            // checks. Neither is yet evidence that the on-disk artifact actually
            // has the declared digest.
            reject_unknown_benchmark_hash_fields(hashes, &manifest_path_label)?;
            validate_benchmark_hash_field_values(hashes, &manifest_path_label)?;
        }
        // The manifest envelope is checked in two steps. Unknown keys, malformed
        // hash members and an unknown embedding source can be reported here before
        // validate_dataset examines the schema version or opens any artifact.
        let source = obj
            .get("query_embedding_source")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "missing string field query_embedding_source",
                )
                .with_detail("field", "query_embedding_source")
                .with_detail("manifest_path", &manifest_path_label)
            })?;
        // Reject unsupported embedding modes before opening artifacts. Silently
        // guessing a mode could change the meaning of a missing sidecar or inline
        // query vector.
        let query_embedding_source = match source {
            "sidecar" => QueryEmbeddingSource::Sidecar,
            "inline" => QueryEmbeddingSource::Inline,
            "none" => QueryEmbeddingSource::None,
            other => {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "query_embedding_source must be sidecar, inline, or none",
                )
                .with_detail("field", "query_embedding_source")
                .with_detail("query_embedding_source", other)
                .with_detail("manifest_path", &manifest_path_label));
            }
        };
        Ok(Self {
            base_dir: manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            manifest_path,
            manifest_path_label,
            manifest_sha256,
            obj,
            query_embedding_source,
            context: context.clone(),
            work,
        })
    }

    fn validate_dataset(mut self) -> Result<(BenchmarkDataset, JsonReservation, Vec<PathBuf>)> {
        // The benchmark dataset manifest version selects this input contract. The
        // producer's own dataset version is a separate required value carried into
        // the output.
        let schema_version = required_string(&self.obj, "schema_version")?;
        if schema_version != "0.1" {
            return Err(Error::new(
                ErrorCode::UnsupportedVersion,
                format!("unsupported dataset manifest schema_version {schema_version}"),
            )
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        let dataset_id = required_string(&self.obj, "dataset_id")?;
        // Reject blank dataset identity components before constructing an
        // externally reported dataset identity. Preserve the original nonblank
        // values, including their spelling.
        self.require_nonempty_manifest_string(&dataset_id, "dataset_id")?;
        let version = required_string(&self.obj, "version")?;
        self.require_nonempty_manifest_string(&version, "version")?;
        // Chunking describes how the supplied chunk rows were produced. The checks
        // validate the declared size and overlap, without re-chunking document
        // text to verify that the existing rows follow that recipe.
        self.validate_chunking()?;
        let hashes = self.hashes()?;
        // The four base artifacts are required across all query modes. Construct
        // their descriptors using the named paths and mandatory hash fields before
        // opening optional vector sources.
        let documents = self.required_artifact("documents", "documents", "documents_sha256")?;
        let chunks = self.required_artifact("chunks", "chunks", "chunks_sha256")?;
        let queries = self.required_artifact("queries", "queries", "queries_sha256")?;
        let qrels = self.required_artifact("qrels", "qrels", "qrels_sha256")?;
        // Collect optional paths and digests independently. A named query-vector
        // file and a digest for a missing file have different implications when
        // enforcing the source mode.
        let embeddings = self.optional_artifact("embeddings", "embeddings", "embeddings_sha256")?;
        let embeddings_hash = optional_string_field(hashes, "embeddings_sha256");
        let query_embeddings = self.optional_artifact(
            "query_embeddings",
            "query_embeddings",
            "query_embeddings_sha256",
        )?;
        let query_embedding_hash = optional_string_field(hashes, "query_embeddings_sha256");
        // A corpus row is checked against embedding_model. Sidecar rows are
        // checked against query_embedding_model, while inline vectors take that
        // name from the manifest. A successful shape check here does not establish
        // that both model declarations name the same model.
        let embedding_model = optional_string_field(&self.obj, "embedding_model");
        let query_embedding_model = optional_string_field(&self.obj, "query_embedding_model");
        self.reject_empty_optional_manifest_string(embedding_model.as_deref(), "embedding_model")?;
        self.reject_empty_optional_manifest_string(
            query_embedding_model.as_deref(),
            "query_embedding_model",
        )?;
        // JSON null means no declared query dimension. A present value must
        // convert to native unsigned size and must be positive before file rows
        // are reconciled with the manifest.
        let query_embedding_dim = self
            .obj
            .get("query_embedding_dim")
            .filter(|value| !value.is_null())
            .map(|value| {
                let dim = json_usize(value).ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "query_embedding_dim must be a positive integer",
                    )
                    .with_detail("field", "query_embedding_dim")
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?;
                // Preserve the distinction between a value that cannot be read as
                // a dimension and an explicitly zero vector width. The latter is
                // reported as a vector-length failure.
                if dim > 0 {
                    Ok(dim)
                } else {
                    Err(Error::new(
                        ErrorCode::InvalidVectorLength,
                        "query_embedding_dim must be greater than zero",
                    )
                    .with_detail("field", "query_embedding_dim")
                    .with_detail("manifest_path", &self.manifest_path_label))
                }
            })
            .transpose()?;
        // Check which combinations of paths, hashes and model metadata the mode
        // permits before inferring a dimension from any corpus artifact.
        self.validate_source_contract(
            query_embeddings.as_ref(),
            query_embedding_hash.as_ref(),
            query_embedding_model.as_ref(),
            query_embedding_dim,
            embeddings.as_ref(),
            embeddings_hash.as_ref(),
        )?;
        // Probe the first corpus embedding for the collection width before the
        // full hash sweep. A malformed first row can therefore fail ahead of a
        // checksum for a different artifact. If the probe reaches EOF, its row
        // reader verifies the digest even for an empty stream.
        let embedding_dim = self.infer_embedding_dim(embeddings.as_ref(), query_embedding_dim)?;

        // Bind the dataset identity to the artifact bytes. Row checks below answer a
        // different question: even a file with the expected hash can name a chunk or
        // query missing from another artifact.
        // Check raw digests in this order before validating the full row
        // relationships. Each later JSONL pass checks its own byte stream again at
        // EOF. A path replaced after this first pass cannot make a different valid
        // stream silently inherit the earlier digest.
        let documents_sha256 = self.validate_hash(&documents)?;
        let chunks_sha256 = self.validate_hash(&chunks)?;
        let queries_sha256 = self.validate_hash(&queries)?;
        let qrels_sha256 = self.validate_hash(&qrels)?;
        // Store a corpus embedding digest only for an artifact whose declared
        // checksum was verified. It is the raw file identity, not a checksum of
        // decoded floating-point values.
        let embeddings_sha256 = embeddings
            .as_ref()
            .filter(|_| embeddings_hash.is_some())
            .map(|artifact| self.validate_hash(artifact))
            .transpose()?;
        let query_embeddings_sha256 = query_embeddings
            .as_ref()
            .map(|artifact| self.validate_hash(artifact))
            .transpose()?;

        // Resolve references in dependency order. Qrels need both query and chunk
        // IDs, and chunks must already have been checked against their parent
        // documents. Keep the validated query data alive for the timed runs.
        // Query rows and selected filter or vector data will outlive validation.
        // Give their retained allocations an operation reservation before
        // validating references in other files.
        let mut retained = self.context.reservation();
        // Query validation runs before the document and chunk passes. Once the
        // digest checks above are complete, an invalid query row can take
        // precedence over a broken document reference in a chunk, even though
        // import will write the corpus first.
        let query_rows = self.read_queries(&queries, query_embedding_dim, &mut retained)?;
        let document_external_ids = self.read_document_external_ids(&documents)?;
        // Validate chunk-to-document references while the admitted document IDs
        // are still available. The resulting chunk IDs are then authoritative for
        // corpus embeddings and relevance judgements.
        let chunk_external_ids = self.read_chunk_external_ids(&chunks, &document_external_ids)?;
        // The corpus vector pass uses the whole chunk-ID set. It checks for
        // missing vectors as well as unknown or repeated chunk IDs. Merely
        // comparing row counts would accept a duplicate in place of a missing
        // chunk.
        self.validate_embeddings(
            embeddings.as_ref(),
            embedding_model.as_deref(),
            embedding_dim,
            &chunk_external_ids,
        )?;
        // Validate query and chunk references before grouping relevance data.
        // Timed queries can then consume prechecked labels without reopening the
        // qrels input.
        let mut qrels_rows = self.validate_qrels(&qrels, &query_rows, &chunk_external_ids)?;
        let mut sidecar_embeddings = match self.query_embedding_source {
            QueryEmbeddingSource::Sidecar => {
                let query_embeddings = query_embeddings
                    .as_ref()
                    .expect("sidecar contract validates query_embeddings is present");
                // Its validator checks coverage for every workload query and
                // validates extra sidecar rows, while retaining coordinates only
                // for query IDs used by the workload.
                self.validate_sidecar(
                    query_embeddings,
                    &query_rows,
                    query_embedding_model.as_deref(),
                    query_embedding_dim,
                    &mut retained,
                )?
            }
            // Inline vectors are already held by their query rows. Validate their
            // source contract without constructing a separate query-ID-to-vector
            // table.
            QueryEmbeddingSource::Inline => {
                self.validate_inline_queries(&query_rows)?;
                BTreeMap::new()
            }
            QueryEmbeddingSource::None => {
                self.validate_text_only_queries(&query_rows)?;
                BTreeMap::new()
            }
        };

        // Retain the ordering of the original query rows while consuming the
        // lookup tables by ID. A sidecar may list the same query vectors in a
        // different order.
        let benchmark_queries = query_rows
            .into_iter()
            .map(|row| {
                // Consume the chosen source of a query vector once. Removing a
                // sidecar entry by ID avoids cloning its coordinates when
                // assembling the benchmark workload.
                let embedding = match self.query_embedding_source {
                    QueryEmbeddingSource::Sidecar => sidecar_embeddings.remove(&row.query_id),
                    QueryEmbeddingSource::Inline => row.embedding,
                    QueryEmbeddingSource::None => None,
                };
                // Qrel validation required at least one judgment for each query.
                // An empty positive set here therefore means that every judgment
                // for this query had relevance zero.
                let relevant_chunk_ids = qrels_rows.remove(&row.query_id).unwrap_or_default();
                BenchmarkQuery {
                    query_id: row.query_id,
                    text: row.text,
                    filter: row.filter,
                    top_k: row.top_k,
                    embedding,
                    relevant_chunk_ids,
                }
            })
            .collect::<Vec<_>>();

        // Tokenizer parsing is deferred until file identities and cross-file
        // references are valid. A tokenizer error at this point discards the
        // assembled query data before the runner can consume it.
        let tokenizer = self
            .obj
            .get("tokenizer")
            .map(crate::TokenizerConfig::from_json)
            .transpose()?
            .unwrap_or_default();
        // Only the chunking subtree needs to survive as raw manifest JSON. Move it
        // out and drop the remaining manifest map before reconciling its charge to
        // the retained value.
        let chunking = self.obj.remove("chunking").unwrap_or(JsonValue::Null);
        drop(self.obj);
        // Reconcile the manifest charge after dropping discarded fields. Merging
        // that remainder into the query reservation makes the returned guard cover
        // the surviving raw chunking value.
        chunking.reconcile_retained_in(JsonInputKind::Manifest, &mut self.work)?;
        retained.merge(self.work);
        // Keep all source locations that participated in validation, including
        // optional query and corpus vectors. The benchmark runner can use them to
        // protect inputs from an output-path collision.
        let input_paths = [
            self.manifest_path,
            documents.path.clone(),
            chunks.path.clone(),
            queries.path,
            qrels.path,
        ]
        .into_iter()
        // The optional corpus vector file can be reopened during import, while the
        // query-side file may already have been consumed into memory. Both are
        // still protected dataset inputs.
        .chain(embeddings.as_ref().map(|artifact| artifact.path.clone()))
        .chain(query_embeddings.map(|artifact| artifact.path))
        .collect();
        Ok((
            BenchmarkDataset {
                manifest: BenchmarkDatasetManifest {
                    tokenizer,
                    dataset_id,
                    version,
                    chunking,
                    embedding_model,
                    embedding_dim,
                    documents_sha256,
                    chunks_sha256,
                    embeddings_sha256,
                    queries_sha256,
                    qrels_sha256,
                    query_embedding_source: self.query_embedding_source,
                    query_embedding_model,
                    query_embedding_dim,
                    query_embeddings_sha256,
                    dataset_manifest_sha256: self.manifest_sha256,
                },
                documents_path: documents.path,
                chunks_path: chunks.path,
                embeddings_path: embeddings.map(|artifact| artifact.path),
                queries: benchmark_queries,
                base_dir: self.base_dir,
                manifest_path_label: self.manifest_path_label,
            },
            retained,
            input_paths,
        ))
    }

    fn hashes(&self) -> Result<&JsonObject> {
        self.obj
            .get("hashes")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "missing object field hashes",
                )
                .with_detail("field", "hashes")
                .with_detail("manifest_path", &self.manifest_path_label)
            })
    }

    fn validate_chunking(&self) -> Result<()> {
        // An absent or null chunking declaration is accepted. For an object,
        // validate the two required coordinates and retain the entire object for
        // result provenance, including any producer-specific members.
        let Some(value) = self.obj.get("chunking").filter(|value| !value.is_null()) else {
            return Ok(());
        };
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest chunking must be a JSON object",
            )
            .with_detail("field", "chunking")
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        // Parse chunking coordinates as integer sizes. A missing or
        // non-convertible member is attributed to its nested field; zero is a
        // separate invalid chunk-size case below.
        let chunk_size = obj.get("chunk_size").and_then(json_usize).ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark manifest chunking requires positive integer chunk_size",
            )
            .with_detail("field", "chunking.chunk_size")
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        if chunk_size == 0 {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest chunking chunk_size must be greater than zero",
            )
            .with_detail("field", "chunking.chunk_size")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        // A present chunking object requires an overlap member even when it is
        // zero. Absence cannot be assumed to mean nonoverlapping chunks because
        // that would alter the declared preprocessing identity.
        let chunk_overlap = obj
            .get("chunk_overlap")
            .and_then(json_usize)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "benchmark manifest chunking requires integer chunk_overlap",
                )
                .with_detail("field", "chunking.chunk_overlap")
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
        // The overlap can be zero in a benchmark manifest. It must remain smaller
        // than the chunk width so a chunking description represents forward
        // progress.
        if chunk_overlap >= chunk_size {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest chunk_overlap must be smaller than chunk_size",
            )
            .with_detail("field", "chunking.chunk_overlap")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    // Whitespace is used only to decide whether this required label is present.
    // Preserve accepted strings for bytewise identity and diagnostics.
    fn require_nonempty_manifest_string(&self, value: &str, field: &'static str) -> Result<()> {
        if value.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark manifest field {field} must not be empty"),
            )
            .with_detail("field", field)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn reject_empty_optional_manifest_string(
        &self,
        value: Option<&str>,
        field: &'static str,
    ) -> Result<()> {
        // Missing optional strings remain absent. For labels that are present,
        // trim only for the emptiness test and retain the caller's spelling for
        // identity and model comparison.
        if matches!(value, Some(value) if value.trim().is_empty()) {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark manifest field {field} must not be empty when present"),
            )
            .with_detail("field", field)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    fn required_artifact(
        &self,
        path_field: &'static str,
        kind: &'static str,
        hash_field: &'static str,
    ) -> Result<Artifact> {
        // If the input path is missing, report the manifest location and artifact
        // kind even though there is no artifact filename to attach yet.
        let raw_path = required_string(&self.obj, path_field).map_err(|err| {
            Error::new(err.code(), err.message())
                .with_detail("field", path_field)
                .with_detail("artifact_kind", kind)
                .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        // A required path also needs its corresponding expected hash. At this
        // point the raw manifest spelling is available to identify which member
        // lacks byte identity.
        if optional_string_field(self.hashes()?, hash_field).is_none() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("missing string field hashes.{hash_field}"),
            )
            .with_detail("field", format!("hashes.{hash_field}"))
            .with_detail("path", &raw_path)
            .with_detail("artifact_kind", kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(self.artifact(raw_path, kind, hash_field))
    }

    // An optional artifact path can be absent at this stage. Its source mode
    // decides later whether absence is permitted, and whether a hash without that
    // path is a conflict.
    fn optional_artifact(
        &self,
        path_field: &'static str,
        kind: &'static str,
        hash_field: &'static str,
    ) -> Result<Option<Artifact>> {
        // Only a string supplies an optional artifact pathname. Other JSON types
        // yield no descriptor through this helper. The mode check later decides
        // whether the resulting absence violates a required input.
        Ok(optional_string_field(&self.obj, path_field)
            .map(|raw_path| self.artifact(raw_path, kind, hash_field)))
    }

    fn artifact(&self, raw_path: String, kind: &'static str, hash_field: &'static str) -> Artifact {
        Artifact {
            // Use the manifest directory to resolve relative input paths. Preserve
            // the original spelling in the descriptor so errors can refer to the
            // field's value instead of an internal working path.
            path: self.base_dir.join(&raw_path),
            raw_path,
            kind,
            hash_field,
        }
    }

    fn validate_source_contract(
        &self,
        query_embeddings: Option<&Artifact>,
        query_embedding_hash: Option<&String>,
        query_embedding_model: Option<&String>,
        query_embedding_dim: Option<usize>,
        embeddings: Option<&Artifact>,
        embeddings_hash: Option<&String>,
    ) -> Result<()> {
        // Corpus model presence and query model or dimension are distinct
        // conditions. The two vector modes require a corpus model as well as the
        // metadata that validates their query vectors.
        let embedding_model_present = self
            .obj
            .get("embedding_model")
            .and_then(JsonValue::as_str)
            .is_some();
        match self.query_embedding_source {
            QueryEmbeddingSource::Sidecar => {
                // A sidecar must identify a particular file and its bytes before
                // rows can be matched by query ID. A query model and width are
                // also required to interpret those rows.
                if query_embeddings.is_none() {
                    return Err(self.missing_source_field("query_embeddings"));
                }
                if query_embedding_hash.is_none() {
                    return Err(self.missing_source_field("hashes.query_embeddings_sha256"));
                }
                if query_embedding_model.is_none() {
                    return Err(self.missing_source_field("query_embedding_model"));
                }
                if query_embedding_dim.is_none() {
                    return Err(self.missing_source_field("query_embedding_dim"));
                }
                // Query-side vectors alone cannot define a vector benchmark. The
                // corpus must also declare embeddings and their digest so the
                // query has a compatible search space.
                if embeddings.is_none() {
                    return Err(self.missing_source_field("embeddings"));
                }
                if embeddings_hash.is_none() {
                    return Err(self.missing_source_field("hashes.embeddings_sha256"));
                }
                if !embedding_model_present {
                    return Err(self.missing_benchmark_field("embedding_model"));
                }
            }
            QueryEmbeddingSource::Inline => {
                // Inline vectors come from query rows. Reject both a named sidecar
                // and a stray sidecar digest even if one of them was supplied
                // without the other.
                if query_embeddings.is_some() || query_embedding_hash.is_some() {
                    return Err(self.mode_conflict(
                        "inline mode must not declare query_embeddings or hashes.query_embeddings_sha256",
                    ));
                }
                if query_embedding_model.is_none() {
                    return Err(self.missing_source_field("query_embedding_model"));
                }
                if query_embedding_dim.is_none() {
                    return Err(self.missing_source_field("query_embedding_dim"));
                }
                // Inline mode removes only the separate query artifact. The corpus
                // vector source is still required just as in sidecar mode.
                if embeddings.is_none() {
                    return Err(self.missing_source_field("embeddings"));
                }
                if embeddings_hash.is_none() {
                    return Err(self.missing_source_field("hashes.embeddings_sha256"));
                }
                if !embedding_model_present {
                    return Err(self.missing_benchmark_field("embedding_model"));
                }
            }
            QueryEmbeddingSource::None => {
                // Result provenance requires a null corpus model for text-only
                // datasets. Reject a declared model before the runner opens a
                // database or creates an execution journal.
                if self
                    .obj
                    .get("embedding_model")
                    .is_some_and(|value| !value.is_null())
                {
                    return Err(self
                        .mode_conflict("none mode requires embedding_model to be omitted or null")
                        .with_detail("field", "embedding_model"));
                }
                if query_embeddings.is_some()
                    || query_embedding_hash.is_some()
                    || query_embedding_model.is_some()
                    || query_embedding_dim.is_some()
                    || embeddings.is_some()
                    || embeddings_hash.is_some()
                {
                    return Err(self.mode_conflict(
                        "none mode must not declare embedding artifacts or query embedding fields",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_hash(&self, artifact: &Artifact) -> Result<String> {
        let expected = optional_string_field(self.hashes()?, artifact.hash_field)
            .expect("required hash presence was validated before hashing");
        // Hash the original file bytes before folding its rows into dataset state.
        // Report both actual and declared digests on mismatch so changing
        // serialization remains distinguishable from malformed row data.
        let actual = sha256_reader(File::open(&artifact.path)?)?;
        // The earlier manifest check required a nonblank string, without
        // normalizing its case or whitespace. Compare exactly with the computed
        // hexadecimal digest, so an uppercase spelling of an otherwise correct
        // hash is still a checksum mismatch.
        if actual != expected {
            return Err(artifact_hash_mismatch(
                artifact,
                &self.manifest_path_label,
                &expected,
                &actual,
            ));
        }
        Ok(actual)
    }

    // Choose the collection width before opening the benchmark database. The first
    // chunk embedding can supply that width, but the later traversal still has to
    // validate every vector against it.
    fn infer_embedding_dim(
        &self,
        embeddings: Option<&Artifact>,
        query_embedding_dim: Option<usize>,
    ) -> Result<usize> {
        // A text-only dataset may have no chunk embeddings and no query embedding
        // dimension. Supply one as the collection width in that case; no vector is
        // fabricated for a query.
        let Some(embeddings) = embeddings else {
            return Ok(query_embedding_dim.unwrap_or(1));
        };
        // Probe one record without retaining coordinates. This iterator normally
        // checks the stream digest at EOF, so a successful nonempty probe alone
        // cannot verify the file.
        let inferred = self
            .read_artifact_objects(embeddings)?
            // Consume just the first result from the row iterator. Propagate a
            // parse or schema error for that row before trying any fallback
            // dimension.
            .next()
            .transpose()?
            .map(
                |ArtifactObject {
                     line, obj, _work, ..
                 }| {
                    // Read the row's declared dimension even if the query manifest
                    // already gives one. A malformed first row is an artifact
                    // error and must not fall back to a plausible manifest value.
                    required_usize(&obj, "dim").map_err(|err| {
                        // Rebuild this field error with the embeddings file's
                        // source label and physical line. The query declaration is
                        // only an additional constraint and cannot locate the bad
                        // corpus row.
                        Error::new(err.code(), err.message())
                            .with_detail("field", "dim")
                            .with_detail("path", &embeddings.raw_path)
                            .with_detail("artifact_kind", embeddings.kind)
                            .with_detail("line", line)
                            .with_detail("manifest_path", &self.manifest_path_label)
                    })
                },
            )
            .transpose()?;
        // An empty embedding stream has no first dimension. The fallback only
        // completes this inference step; full validation below rejects a supplied
        // embeddings file with no rows.
        let inferred = inferred.or(query_embedding_dim).unwrap_or(1);
        // The first row and the query declaration constrain the same collection
        // width. Report both numbers here so disagreement is distinguishable from
        // a later short vector.
        if let Some(query_dim) = query_embedding_dim {
            // The dimension probe precedes the full-file hash checks. A
            // disagreement here can be reported before reading the rest of the
            // corpus, including a changed tail.
            if inferred != query_dim {
                return Err(Error::new(
                    ErrorCode::InvalidEmbeddingDimension,
                    "embedding dimension does not match query_embedding_dim",
                )
                // A conflicting query declaration names the manifest and both
                // widths. Unlike a malformed dim field, neither number is
                // ill-typed; it is their relationship that makes this dataset
                // invalid.
                .with_detail("embedding_dim", inferred)
                .with_detail("query_embedding_dim", query_dim)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        Ok(inferred)
    }

    // Read in query-file order, which becomes the order of the benchmark workload.
    // Source-mode validation later checks which rows were required or forbidden to
    // carry inline vectors.
    fn read_queries(
        &self,
        artifact: &Artifact,
        query_embedding_dim: Option<usize>,
        retained: &mut JsonReservation,
    ) -> Result<Vec<QueryRow>> {
        let mut rows = Vec::new();
        // Use a separate set for uniqueness so sorting or deduplicating the result
        // is unnecessary. Repeated query IDs would make sidecar and relevance
        // joins ambiguous.
        let mut seen_query_ids = BTreeSet::new();
        // The earlier checksum pass and this row stream open the pathname
        // separately. Exhaustion checks the digest of the bytes actually parsed. A
        // row error can end this pass before that second digest is available.
        for row in self.read_artifact_objects(artifact)? {
            // Take the source line from the iterator, not the number of accepted
            // queries. Empty lines in JSONL can make those numbers differ.
            let ArtifactObject {
                line,
                mut obj,
                _work: mut work,
                search_top_k,
            } = row?;
            // Take owned string fields out of the parsed object. The query can
            // outlive that object's workspace, and keeping a second copy of its
            // text would add another live allocation.
            let query_id =
                crate::json::take_required_string(&mut obj, "query_id").map_err(|err| {
                    Error::new(err.code(), err.message())
                        .with_detail("field", "query_id")
                        .with_detail("path", &artifact.raw_path)
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            // Test for a blank ID without normalizing it. The original spelling is
            // the key used by the sidecar, qrels and reported results.
            self.require_nonempty_artifact_string(&query_id, "queries.query_id", artifact, line)?;
            // Query IDs keep their exact spelling. Reject repeats before this
            // function checks the duplicate's text, filter or coordinates. The row
            // parser has already validated the envelope and captured top_k.
            if !seen_query_ids.insert(query_id.clone()) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "duplicate query_id in benchmark queries",
                )
                .with_detail("query_id", &query_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Text is required in every source mode, including runs that also use
            // a precomputed query vector. Keep the original text for the eventual
            // search.
            let text = crate::json::take_required_string(&mut obj, "text").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "queries.text")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Use trimming only for the emptiness test. Keep the accepted text as
            // supplied so each workload implementation receives the same input
            // before applying its own normalization.
            if text.trim().is_empty() {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    // A structurally valid vector does not exempt a query from
                    // supplying its text. This common workload schema keeps the
                    // same query rows usable by the text-based benchmark methods.
                    "benchmark query requires non-empty text",
                )
                .with_detail("field", "queries.text")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Remove a supplied filter so the validated query can take ownership
            // of the JSON subtree. An omitted filter defaults to an empty object,
            // while an explicit null is an invalid shape.
            let filter = obj
                .remove("filter")
                .map(|value| {
                    // Reject nonobject filters before passing them to the
                    // predicate parser. This distinguishes malformed artifact
                    // shape from a predicate whose operators or operands are
                    // invalid.
                    if matches!(value, JsonValue::Object(_)) {
                        Ok(value)
                    } else {
                        Err(Error::new(
                            ErrorCode::InvalidJsonlSchema,
                            "query filter must be a JSON object",
                        )
                        .with_detail("field", "queries.filter")
                        .with_detail("query_id", &query_id)
                        .with_detail("path", &artifact.raw_path)
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label))
                    }
                })
                .transpose()?
                // The default empty object represents an unrestricted predicate.
                // It uses the same validation and retention path as an explicitly
                // empty object.
                .unwrap_or_else(JsonValue::object);
            // Parsing the object also checks the filter language. Keep a JSON limit
            // failure in its original category and attach the artifact path and
            // line. Report an invalid predicate as a schema error in the queries
            // artifact.
            FilterExpr::parse_validated_in(&filter, &self.context, JsonInputKind::Filter).map_err(
                |err| {
                    if err
                        .detail()
                        .iter()
                        .any(|(key, _)| key == "json_limit_reason")
                    {
                        // A JSON-limit failure already carries measured resource
                        // details. Add the row location to that same error instead
                        // of losing the original reason and limits.
                        return err
                            .with_detail("path", &artifact.raw_path)
                            .with_detail("artifact_kind", artifact.kind)
                            .with_detail("line", line)
                            .with_detail("manifest_path", &self.manifest_path_label);
                    }
                    // Keep the predicate parser's own error code and message as
                    // details on the row-schema error. An invalid operand can then
                    // be traced back from the query artifact to its filter.
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "query filter must be a valid metadata filter",
                    )
                    .with_detail("field", "queries.filter")
                    .with_detail("query_id", &query_id)
                    .with_detail("filter_error_code", err.code().as_str())
                    .with_detail("filter_error", err.message())
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                },
            )?;
            // Treat omission and explicit null alike for this required numeric
            // field. Both differ from a present value whose type or exact numeric
            // spelling fails validation.
            let top_k_value = obj
                .get("top_k")
                .filter(|value| !value.is_null())
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark query requires top_k",
                    )
                    .with_detail("field", "queries.top_k")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            // The JSON number alone cannot distinguish all source spellings after
            // conversion to binary64. Require the integer captured by the row
            // parser from the original top_k token.
            let top_k = matches!(top_k_value, JsonValue::Number(_))
                .then_some(search_top_k)
                .flatten()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "query top_k must be a number with validated exact integer value",
                    )
                    .with_detail("field", "queries.top_k")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            // Even an exactly decoded integer has to fit the search API's
            // supported range. Zero would request an empty result set and is
            // rejected as a schema error here.
            if top_k == 0 || top_k > crate::query::MAX_SEARCH_TOP_K {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "query top_k must be in the supported exact JSON range",
                )
                .with_detail("field", "queries.top_k")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Keep the typed vector's allocation separate from the temporary JSON
            // representation. Their lifetimes diverge when the object is discarded
            // but the benchmark retains the query.
            let mut typed_work = self.context.reservation();
            // A present embedding field is fed to the array converter even when
            // its value is null. Only absence skips creation of the typed vector.
            let vector = obj
                .get("embedding")
                .map(|value| self.vector_values(value, &query_id, line, "queries", &mut typed_work))
                .transpose()
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            // Check actual vector length as soon as the manifest supplies an
            // expected dimension. A later check for presence in inline mode cannot
            // prove a present vector has the correct shape.
            if let (Some(vector), Some(expected)) = (&vector, query_embedding_dim) {
                let actual = vector.len();
                if actual != expected {
                    return Err(self.locate_artifact_error(
                        // Inline rows have no independent per-row dim member.
                        // Supply the manifest expectation for the declared-width
                        // field and carry the observed vector length separately.
                        invalid_vector_len(
                            &self.manifest_path_label,
                            &query_id,
                            expected,
                            expected,
                            actual,
                        ),
                        artifact,
                        line,
                    ));
                }
            }
            // An inline row supplies the coordinates only. Carry the model name
            // from the dataset declaration, and use the validated vector's own
            // element count as its dimension.
            let embedding = vector
                .map(|vector| -> Result<_> {
                    Ok(QueryEmbeddingInput {
                        model_name: self
                            .obj
                            .get("query_embedding_model")
                            .and_then(JsonValue::as_str)
                            // Each retained inline input owns the model spelling
                            // it reports. Account for that copy even though its
                            // bytes originated in the shared manifest.
                            .map(|model| {
                                typed_work.copy_string(model, JsonInputKind::QueryEmbedding)
                            })
                            .transpose()?,
                        dim: vector.len(),
                        vector,
                    })
                })
                .transpose()
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            // Release discarded row fields before recalculating the JSON charge
            // for the retained filter. The typed vector and any model copy have a
            // separate reservation.
            drop(obj);
            filter
                .reconcile_retained_in(JsonInputKind::Filter, &mut work)
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            // Transfer the remaining charges to the dataset owner. Both the filter
            // JSON and typed input survive the iteration that parsed this row.
            retained.merge(work);
            retained.merge(typed_work);
            // Remember inline presence separately from the option's contents. The
            // mode checks can identify the source of the vector before final
            // assembly moves it into the benchmark query.
            let has_embedding = embedding.is_some();
            rows.push(QueryRow {
                query_id,
                line,
                text: Some(text),
                // Keep the original filter JSON in the workload. The predicate
                // parsed above was a validation step. Both the reference search
                // and the measured engine receive this expression when they
                // execute the query.
                filter,
                top_k,
                embedding,
                has_embedding,
            });
        }
        // EOF with no queries is a missing workload. A successful parse of an
        // empty file cannot produce meaningful per-query benchmark results.
        if rows.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark queries must not be empty",
            )
            .with_detail("field", "queries")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(rows)
    }

    // Join a sidecar by query ID rather than position. It can list other IDs and
    // use a different order, provided its own rows are valid and every requested
    // query is covered.
    fn validate_sidecar(
        &self,
        artifact: &Artifact,
        queries: &[QueryRow],
        expected_model: Option<&str>,
        expected_dim: Option<usize>,
        retained: &mut JsonReservation,
    ) -> Result<BTreeMap<String, QueryEmbeddingInput>> {
        // The manifest selects one source of query vectors. With a sidecar, an
        // inline vector would give the query two possible values, even if both
        // happened to have the requested dimension.
        // Check source exclusivity before reopening the sidecar to decode its
        // rows. The earlier digest pass established the sidecar's identity but
        // cannot resolve an inline conflict in the queries file.
        if let Some(row) = queries.iter().find(|row| row.has_embedding) {
            return Err(self.mode_conflict(format!(
                "sidecar mode query {} at line {} must not inline embedding",
                row.query_id, row.line
            )));
        }

        // The caller enforces the manifest's sidecar contract before reading rows.
        // Missing model or dimension is handled there as a manifest error.
        let expected_model =
            expected_model.expect("sidecar contract validates query_embedding_model is present");
        let expected_dim = expected_dim.expect("sidecar contract validates query_embedding_dim");
        // Borrow IDs from the query slice for membership checks. The returned
        // vectors own the model names and coordinates needed after the parsed JSON
        // rows have been released.
        let query_ids = queries
            .iter()
            .map(|query| query.query_id.as_str())
            .collect::<BTreeSet<_>>();
        // Keep source lines independently of retained vectors. This lets a
        // duplicate name identify both occurrences even when the first row belongs
        // to a query outside the workload.
        let mut seen = BTreeMap::<String, SidecarRow>::new();
        let mut embeddings = BTreeMap::<String, QueryEmbeddingInput>::new();
        // Consume the entire sidecar before deciding which request IDs are absent.
        // A trailing invalid row or digest mismatch still fails a sidecar that
        // already covered the workload.
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line,
                mut obj,
                _work: mut work,
                ..
            } = row?;
            let query_id =
                crate::json::take_required_string(&mut obj, "query_id").map_err(|err| {
                    Error::new(err.code(), err.message())
                        .with_detail("field", "query_id")
                        .with_detail("path", &artifact.raw_path)
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            // A whitespace-only key cannot join to a query. As with the queries
            // file, retain the exact spelling of each accepted ID.
            self.require_nonempty_artifact_string(
                &query_id,
                "query_embeddings.query_id",
                artifact,
                line,
            )?;
            // Check for a repeated ID before interpreting its second vector. The
            // diagnostic identifies the first accepted row and the line that
            // conflicts with it.
            if let Some(first) = seen.get(&query_id) {
                return Err(self.locate_artifact_error(
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "duplicate query_id in query embeddings sidecar",
                    )
                    .with_detail("query_id", &query_id)
                    // Keep the two line numbers distinct from the location of the
                    // diagnostic itself. The first is the prior accepted record;
                    // the duplicate is the row currently being parsed.
                    .with_u64_detail("first_line", first.line as u64)
                    .with_u64_detail("duplicate_line", line as u64),
                    artifact,
                    line,
                ));
            }
            // A sidecar row carries its own model identifier. Read it from the row
            // so the declaration can be checked independently of the vector's
            // numerical dimension.
            let row_model =
                crate::json::take_required_string(&mut obj, "model_name").map_err(|err| {
                    self.locate_artifact_error(
                        Error::new(err.code(), err.message())
                            .with_detail("field", "model_name")
                            .with_detail("query_id", &query_id),
                        artifact,
                        line,
                    )
                })?;
            self.require_nonempty_artifact_string(
                &row_model,
                "query_embeddings.model_name",
                artifact,
                line,
            )?;
            // Check the sidecar row against query_embedding_model. Corpus
            // embedding rows use the separate embedding_model field. Equal
            // dimension alone supplies no model identity.
            if row_model != expected_model {
                return Err(self.locate_artifact_error(
                    Error::new(
                        ErrorCode::EmbeddingModelMismatch,
                        "query embedding model does not match dataset manifest",
                    )
                    .with_detail("query_id", &query_id)
                    .with_detail("manifest_query_embedding_model", expected_model)
                    .with_detail("row_model_name", row_model),
                    artifact,
                    line,
                ));
            }
            // The row declares its dimension separately from the array. Keep that
            // number until conversion succeeds so a mismatch can show the declared
            // and actual lengths together.
            let row_dim = required_usize(&obj, "dim").map_err(|err| {
                self.locate_artifact_error(
                    Error::new(err.code(), err.message())
                        .with_detail("field", "dim")
                        .with_detail("query_id", &query_id),
                    artifact,
                    line,
                )
            })?;
            // A parsed JSON array and the f32 input coexist during conversion.
            // Charge the typed allocation at the point it is created, including
            // for rows later discarded as unused.
            let mut typed_work = self.context.reservation();
            let vector = self
                // Use the same typed-coordinate checks as an inline vector. This
                // includes the operation's query-vector representation limit after
                // conversion to f32.
                .vector_values(
                    obj.get("vector").ok_or_else(|| {
                        self.locate_artifact_error(
                            Error::new(
                                ErrorCode::MissingRequiredField,
                                "missing array field vector",
                            )
                            .with_detail("field", "vector")
                            .with_detail("query_id", &query_id),
                            artifact,
                            line,
                        )
                    })?,
                    &query_id,
                    line,
                    artifact.kind,
                    &mut typed_work,
                )
                .map_err(|err| self.locate_artifact_error(err, artifact, line))?;
            // Compare three quantities: manifest width, row-declared width and
            // actual array length. Agreement between any two cannot establish that
            // the row is usable.
            let vector_len = vector.len();
            if row_dim != expected_dim || vector_len != expected_dim {
                return Err(self.locate_artifact_error(
                    invalid_vector_len(
                        &self.manifest_path_label,
                        &query_id,
                        expected_dim,
                        row_dim,
                        vector_len,
                    ),
                    artifact,
                    line,
                ));
            }
            // Validate every sidecar row, including IDs unused by this query set.
            // Keep vectors only for requested IDs, but keep the first line for every
            // key so a duplicate unused row is still diagnosed.
            if query_ids.contains(query_id.as_str()) {
                // Only the model string's capacity must survive from this row's
                // JSON reservation here. The converted coordinates are already
                // accounted for by typed_work.
                drop(obj);
                work.retain(row_model.capacity());
                typed_work.merge(work);
                retained.merge(typed_work);
                // Index the retained result by ID, not sidecar line. Later
                // assembly removes one entry per query while preserving the query
                // file's ordering.
                embeddings.insert(
                    query_id.clone(),
                    QueryEmbeddingInput {
                        model_name: Some(row_model),
                        dim: row_dim,
                        vector,
                    },
                );
            }
            seen.insert(query_id, SidecarRow { line });
        }

        // Check missing IDs in query-file order after consuming the sidecar. Extra
        // valid rows are permitted, so the two files' row counts alone cannot
        // establish coverage.
        if let Some(row) = queries
            .iter()
            .find(|query| !seen.contains_key(&query.query_id))
        {
            // There is no sidecar line for an absent ID. The diagnostic's source
            // names the expected artifact type, while path is the filename the
            // manifest actually supplied, which may differ.
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "query embedding sidecar row missing for query_id",
            )
            .with_detail("query_id", &row.query_id)
            .with_detail("field", "query_embeddings[query_id]")
            .with_detail("source", "query_embeddings.jsonl")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(embeddings)
    }

    // The general query reader validates coordinates when present. Inline mode
    // adds the separate requirement that every workload row actually supplied a
    // vector.
    fn validate_inline_queries(&self, queries: &[QueryRow]) -> Result<()> {
        if let Some(row) = queries.iter().find(|row| !row.has_embedding) {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "inline query embedding missing for query_id",
            )
            // A missing inline vector has a real source line in queries.jsonl. A
            // missing sidecar row, in contrast, can identify the expected query
            // but has no sidecar line of its own.
            .with_detail("query_id", &row.query_id)
            .with_detail("field", "queries.embedding")
            .with_detail("line", row.line)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    // Represent benchmark judgments as a set of relevant chunk IDs per query. Keep
    // all source pairs separately during validation because a zero grade still
    // counts as a judgment.
    fn validate_qrels(
        &self,
        artifact: &Artifact,
        queries: &[QueryRow],
        chunk_external_ids: &BTreeSet<String>,
    ) -> Result<BTreeMap<String, BTreeSet<String>>> {
        // Resolve references against the submitted queries. A relevance judgment
        // for a query outside this workload is a schema error even if its chunk
        // exists.
        let query_ids = queries
            .iter()
            .map(|query| query.query_id.clone())
            .collect::<BTreeSet<_>>();
        // These labels describe quality against the source corpus. A judged chunk
        // can be filtered out by a query or fall outside top_k. The benchmark
        // constructs its expected ranking independently of these sets.
        let mut qrels = BTreeMap::<String, BTreeSet<String>>::new();
        // Remember the first line for every pair, including nonrelevant pairs.
        // Otherwise two zero grades or a zero followed by a positive grade could
        // pass through the positive-only set.
        let mut seen_pairs = BTreeMap::<(String, String), usize>::new();
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            let query_id = required_string(&obj, "query_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "query_id")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(&query_id, "qrels.query_id", artifact, line)?;
            // The chunk key denotes one retrieval unit, so a document ID alone
            // cannot serve as a relevance target. Validate the second field with
            // the query ID already known.
            let chunk_external_id = required_string(&obj, "chunk_external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "chunk_external_id")
                    .with_detail("query_id", &query_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(
                &chunk_external_id,
                "qrels.chunk_external_id",
                artifact,
                line,
            )?;
            // Require a numeric relevance even though evaluation later uses only
            // the distinction between zero and positive. A missing grade cannot be
            // inferred from presence of the pair.
            let relevance = json_usize(obj.get("relevance").ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingRequiredField,
                    "missing numeric field relevance",
                )
                // A missing grade still has a source location. Report the query
                // whose judgment needs repair together with the physical qrels
                // artifact and line.
                .with_detail("field", "relevance")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label)
            })?)
            // A supplied grade that cannot become a nonnegative native integer is
            // malformed. A missing field reaches the distinct error branch above.
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "relevance must be a non-negative integer",
                )
                .with_detail("field", "relevance")
                .with_detail("query_id", &query_id)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Both references must resolve before recording the judgment. Check
            // the query first so errors on a row with two unknown IDs have a
            // consistent location.
            if !query_ids.contains(&query_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "qrels row references unknown query_id",
                )
                .with_detail("query_id", &query_id)
                .with_detail("field", "qrels.query_id")
                .with_detail("source", "qrels.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Check the chunk ID independently of the query ID. A positive grade
            // for an absent chunk would make the requested recall target
            // impossible to reach.
            if !chunk_external_ids.contains(&chunk_external_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "qrels row references unknown chunk_external_id",
                )
                .with_detail("query_id", &query_id)
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "qrels.chunk_external_id")
                .with_detail("source", "qrels.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Treat repeated pairs as an error even if their grades agree.
            // Silently unioning the judged chunks would hide a duplicate in the
            // source artifact.
            if let Some(first_line) =
                // Uniqueness is scoped to the ordered pair. The same chunk can be
                // relevant to many queries, and each query can have multiple
                // judged chunks.
                seen_pairs.insert((query_id.clone(), chunk_external_id.clone()), line)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "duplicate qrels row for query_id and chunk_external_id",
                )
                .with_detail("query_id", &query_id)
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "qrels.query_id+chunk_external_id")
                .with_detail("source", "qrels.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("first_line", first_line)
                .with_detail("duplicate_line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // A judgment with relevance zero still establishes coverage for the
            // query. Create the entry before testing relevance so an all-negative
            // query is distinguishable from a query missing its qrels altogether.
            let relevant = qrels.entry(query_id).or_default();
            // Positive grades contribute the same membership regardless of
            // magnitude. A query with only zero grades remains in the timed
            // workload and contributes zero Recall and reciprocal rank.
            if relevance > 0 {
                relevant.insert(chunk_external_id);
            }
        }
        // Coverage is per workload query. Report the first uncovered query in its
        // input order, even if other queries have many valid judgments.
        if let Some(query) = queries
            .iter()
            .find(|query| !qrels.contains_key(&query.query_id))
        {
            // There is no offending qrels row when an expected judgment is absent.
            // Identify the uncovered query and its expected input rather than
            // assigning a spurious row number.
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "qrels row missing for query_id",
            )
            .with_detail("query_id", &query.query_id)
            .with_detail("field", "qrels[query_id]")
            .with_detail("source", "qrels.jsonl")
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(qrels)
    }

    // Check chunk-vector rows against the chunk artifact before benchmark
    // execution. The runner needs their validated identity and shape here, while
    // import reads the vector artifact separately.
    fn validate_embeddings(
        &self,
        artifact: Option<&Artifact>,
        expected_model: Option<&str>,
        expected_dim: usize,
        chunk_external_ids: &BTreeSet<String>,
    ) -> Result<()> {
        // No supplied chunk-vector artifact requires no row checks here. Once a
        // file is supplied, empty or partially covered contents are errors.
        let Some(artifact) = artifact else {
            return Ok(());
        };
        // Track only chunk IDs and first source lines across rows. This validation
        // does not need an in-memory copy of every chunk vector.
        let mut seen = BTreeMap::<String, usize>::new();
        // Carry the path and digest into the generated import manifest. The
        // importer verifies the bytes it stages, so a changed file after preflight
        // still has to pass its own checksum check.
        for row in self.read_artifact_objects(artifact)? {
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            // A corpus row joins by chunk_external_id. Query-side vector records
            // have a different key and a different coverage rule.
            let chunk_external_id = required_string(&obj, "chunk_external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "embeddings.chunk_external_id")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            self.require_nonempty_artifact_string(
                &chunk_external_id,
                "embeddings.chunk_external_id",
                artifact,
                line,
            )?;
            // Validate identity before the model or coordinates. A row must name
            // an admitted chunk even if the rest of its embedding is well formed.
            if !chunk_external_ids.contains(&chunk_external_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "embedding row references unknown chunk_external_id",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "embeddings.chunk_external_id")
                .with_detail("source", "embeddings.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Duplicate chunk vectors use the same external-ID error class as
            // other duplicate corpus identities. Preserve both lines to locate
            // entries that might be far apart.
            if let Some(first_line) = seen.insert(chunk_external_id.clone(), line) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate chunk_external_id in benchmark embeddings",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "embeddings.chunk_external_id")
                .with_detail("source", "embeddings.jsonl")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("first_line", first_line)
                .with_detail("duplicate_line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            let row_model = required_string(&obj, "model_name").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "embeddings.model_name")
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // A chunk vector row always needs its own nonblank model label. The
            // presence of a manifest binding controls the additional equality
            // check that follows.
            self.require_nonempty_artifact_string(
                &row_model,
                "embeddings.model_name",
                artifact,
                line,
            )?;
            // Match model identifiers exactly when the manifest binds one.
            // Compatible dimensions alone do not show that the coordinates share
            // the dataset's embedding space.
            if let Some(expected_model) = expected_model {
                if row_model != expected_model {
                    return Err(Error::new(
                        ErrorCode::EmbeddingModelMismatch,
                        "embedding model_name does not match dataset manifest",
                    )
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("manifest_embedding_model", expected_model)
                    .with_detail("row_model_name", row_model)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label));
                }
            }
            // A missing or nonnumeric dim has the missing-field diagnostic. A
            // numeric fraction or a number outside the native unsigned range is a
            // schema error here, before the later comparison with the vector's
            // length.
            let row_dim = required_usize(&obj, "dim").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "embeddings.dim")
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Check every coordinate for conversion to finite f32 and return only
            // the element count. Keep the on-disk array for the later import
            // instead of constructing another typed vector here.
            let vector_len = self.validate_embedding_vector(
                // Report a missing vector field against the chunk key. An array
                // with invalid elements proceeds to coordinate validation, which
                // supplies a different error.
                obj.get("vector").ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "missing array field vector",
                    )
                    .with_detail("field", "embeddings.vector")
                    .with_detail("chunk_external_id", &chunk_external_id)
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
                })?,
                &chunk_external_id,
                line,
                artifact,
            )?;
            // An incorrect declared width and an incorrect array length are
            // independent faults. Report both observed values alongside the
            // inferred collection width.
            if row_dim != expected_dim || vector_len != expected_dim {
                return Err(Error::new(
                    ErrorCode::InvalidVectorLength,
                    "embedding dimension does not match dataset manifest",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("expected_dim", expected_dim)
                .with_detail("row_dim", row_dim)
                .with_detail("actual_vector_length", vector_len)
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        // A supplied file must yield at least one embedding. Report this
        // separately from the first particular missing chunk in an otherwise
        // nonempty file.
        if seen.is_empty() {
            // At this point the reader has reached EOF and checked the digest. A
            // matching hash for zero data rows still does not make the file a
            // valid corpus.
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark embeddings must not be empty",
            )
            .with_detail("field", "embeddings")
            .with_detail("source", "embeddings.jsonl")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        // When a chunk embedding artifact is supplied it must cover the full chunk
        // set. Per-row checks rule out unknown and duplicate keys; they do not
        // detect a missing last row in an otherwise valid file.
        // The chunk set is ordered by external ID. If several embeddings are
        // missing, this iteration selects a stable ID for the diagnostic.
        if let Some(missing) = chunk_external_ids
            .iter()
            .find(|chunk_id| !seen.contains_key(*chunk_id))
        {
            // A missing corpus row cannot supply its own input line. Return the
            // expected chunk ID and the named embeddings input so the omitted row
            // can be reconstructed.
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "embedding row missing for chunk_external_id",
            )
            .with_detail("chunk_external_id", missing)
            .with_detail("field", "embeddings[chunk_external_id]")
            .with_detail("source", "embeddings.jsonl")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    // Collect checked document IDs for the later parent join. Only the IDs survive
    // this scan. They are provisional until the iterator reaches EOF. An invalid
    // row stops the scan immediately, so its error can take precedence over a
    // digest mismatch in the rest of the file.
    fn read_document_external_ids(&self, artifact: &Artifact) -> Result<BTreeSet<String>> {
        // Retain a set to make later chunk ownership checks independent of
        // document row order. A set alone would hide duplicates, so reject failed
        // insertion at the source row.
        let mut ids = BTreeSet::new();
        // Drive the artifact iterator through EOF. Its final step compares the
        // digest of the same byte stream that produced these document rows.
        for row in self.read_artifact_objects(artifact)? {
            // Keep the row's reservation in scope while reading its object. Moving
            // fields out for validation must not release the charge before those
            // fields are finished.
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            // Read a string identifier and translate a bad field into an artifact
            // error with its physical line. Retain both the member and owning
            // manifest labels.
            let external_id = required_string(&obj, "external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "external_id")
                    .with_detail("path", &artifact.raw_path)
                    // The artifact kind identifies the row schema even when the
                    // input files share the same basename.
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Reject an empty or whitespace-only ID without trimming a valid ID.
            // Later joins compare the exact supplied spelling.
            self.require_nonempty_artifact_string(
                &external_id,
                "documents.external_id",
                artifact,
                line,
            )?;
            // Benchmark document metadata must be an object. Check its outer type
            // here before downstream code interprets individual keys.
            self.require_artifact_object_field(
                &obj,
                "metadata",
                "documents.metadata",
                artifact,
                line,
            )?;
            // IDs are global within the documents artifact. A second occurrence is
            // ambiguous even when other fields of the two rows differ.
            if !ids.insert(external_id.clone()) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate external_id in benchmark documents",
                )
                .with_detail("external_id", &external_id)
                .with_detail("field", "documents.external_id")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        // A digest can identify a completely empty stream, so verify semantic
        // nonemptiness separately after iteration. Blank lines do not count as
        // documents.
        if ids.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark documents must not be empty",
            )
            .with_detail("field", "documents")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        // Return the ID set only after successful iterator exhaustion. The final
        // hash failure must prevent later stages from receiving a partially
        // trusted document index.
        Ok(ids)
    }

    // Build the set of valid chunk IDs and check document ownership while
    // scanning. Chunk IDs support later artifact joins, while the ordinal table is
    // only needed during this check.
    fn read_chunk_external_ids(
        &self,
        artifact: &Artifact,
        document_external_ids: &BTreeSet<String>,
    ) -> Result<BTreeSet<String>> {
        // Keep the ID set global and the ordinal table keyed by document.
        // Different documents may use the same ordinal without any collision.
        let mut ids = BTreeSet::new();
        // Store the first line number with each document-local ordinal. The value
        // is diagnostic state, while the nested key models the uniqueness domain.
        let mut ordinals_by_document = BTreeMap::<String, BTreeMap<usize, usize>>::new();
        // Validate rows in physical source order. For failures tied to ordering,
        // retain the reader's line number rather than numbering only successfully
        // decoded rows.
        for row in self.read_artifact_objects(artifact)? {
            // The parser can fail before any artifact-specific validation can run.
            // Propagate that failure before looking up document identity or
            // counting this row.
            let ArtifactObject {
                line, obj, _work, ..
            } = row?;
            // Resolve the parent document before accepting any chunk-specific
            // fields. A chunk cannot manufacture a new document identity during
            // benchmark validation.
            let document_external_id =
                required_string(&obj, "document_external_id").map_err(|err| {
                    Error::new(err.code(), err.message())
                        .with_detail("field", "document_external_id")
                        .with_detail("path", &artifact.raw_path)
                        // Attribute a malformed parent ID to the chunks artifact
                        // even though the correct spelling must refer to
                        // documents. This makes the producer's invalid row the
                        // error target.
                        .with_detail("artifact_kind", artifact.kind)
                        .with_detail("line", line)
                        .with_detail("manifest_path", &self.manifest_path_label)
                })?;
            // Validate the raw parent string for presence before membership
            // lookup. Reporting a missing value helps distinguish it from a
            // well-formed but unknown reference.
            self.require_nonempty_artifact_string(
                &document_external_id,
                "chunks.document_external_id",
                artifact,
                line,
            )?;
            // Require membership in the already checked documents artifact. A
            // known chunk external ID cannot compensate for an unknown parent.
            if !document_external_ids.contains(&document_external_id) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark chunk references unknown document_external_id",
                )
                .with_detail("document_external_id", &document_external_id)
                .with_detail("field", "chunks.document_external_id")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Check the document-local position before later chunk fields and the
            // dataset-wide chunk ID. A row with both an ordinal collision and an
            // invalid chunk ID reports the ordinal first.
            let ordinal = required_usize(&obj, "ordinal").map_err(|err| {
                Error::new(err.code(), err.message())
                    // Use a qualified field label for ordinal validation. Manifest
                    // consumers use the same field key for invalid type, numeric
                    // value, and duplicate document position.
                    .with_detail("field", "chunks.ordinal")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // An ordinal identifies a position within its document; the external
            // chunk ID identifies a chunk across this dataset. Track the two
            // constraints independently and retain the first source line for an
            // ordinal collision.
            if let Some(first_line) = ordinals_by_document
                .entry(document_external_id.clone())
                .or_default()
                .insert(ordinal, line)
            {
                // Return immediately when a second row uses this document
                // position. Include the previous and new physical line so the
                // producer can locate both records.
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "duplicate chunk ordinal for document in benchmark chunks",
                )
                .with_detail("document_external_id", &document_external_id)
                .with_detail("ordinal", ordinal)
                .with_detail("field", "chunks.ordinal")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("first_line", first_line)
                .with_detail("duplicate_line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Require a string before checking whether the chunk has searchable
            // text. Keep field errors tied to the artifact and not to a later
            // retrieval failure.
            let text = required_string(&obj, "text").map_err(|err| {
                Error::new(err.code(), err.message())
                    .with_detail("field", "chunks.text")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Reject a blank chunk even if its ID and ordinal are valid. Trimming
            // is a validation predicate; the original text row is still the
            // artifact data.
            if text.trim().is_empty() {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    "benchmark chunk requires non-empty text",
                )
                // Keep a consistent qualified field for missing and blank text.
                // The reader has already separated malformed JSON from an unusable
                // text value.
                .with_detail("field", "chunks.text")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            // Check the chunk metadata object independently of the text. A legal
            // metadata value can be empty, but its enclosing field must still have
            // object type.
            self.require_artifact_object_field(
                &obj,
                "metadata",
                "chunks.metadata",
                artifact,
                line,
            )?;
            // Read the chunk identity separately from its parent linkage. This
            // identity is how later embeddings and relevance labels refer back to
            // the benchmark set.
            let chunk_external_id = required_string(&obj, "chunk_external_id").map_err(|err| {
                Error::new(err.code(), err.message())
                    // Attach the row location before testing cross-row uniqueness.
                    // This preserves the source position for a malformed ID even
                    // when no valid string is available to insert.
                    .with_detail("field", "chunk_external_id")
                    .with_detail("path", &artifact.raw_path)
                    .with_detail("artifact_kind", artifact.kind)
                    .with_detail("line", line)
                    .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Require a nonblank chunk ID before inserting it into the global set.
            // Preserve the exact spelling for cross-artifact lookup.
            self.require_nonempty_artifact_string(
                &chunk_external_id,
                "chunks.chunk_external_id",
                artifact,
                line,
            )?;
            // Duplicate chunk IDs are invalid across the whole dataset even if
            // their documents and ordinals differ. Return the duplicate row
            // location.
            if !ids.insert(chunk_external_id.clone()) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate chunk_external_id in benchmark chunks",
                )
                .with_detail("chunk_external_id", &chunk_external_id)
                .with_detail("field", "chunks.chunk_external_id")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        // The stream may contain only whitespace and still have its declared
        // digest. Require at least one checked chunk before validating references
        // from embeddings and relevance labels.
        if ids.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark chunks must not be empty",
            )
            .with_detail("field", "chunks")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(ids)
    }

    // Use the raw JSON key for lookup and the qualified field label for
    // diagnostics. Callers can identify the exact schema member without
    // reconstructing it from the error text.
    fn require_artifact_object_field(
        &self,
        obj: &JsonObject,
        field: &'static str,
        detail_field: &'static str,
        artifact: &Artifact,
        line: usize,
    ) -> Result<()> {
        // A missing member and a member of the wrong type are distinct failures.
        // An empty object remains legal here because the check only establishes
        // shape.
        match obj.get(field) {
            Some(JsonValue::Object(_)) => Ok(()),
            Some(_) => Err(Error::new(
                // A value of null reaches the wrong-type branch. A missing key
                // instead yields the distinct required-field error below.
                ErrorCode::InvalidJsonlSchema,
                format!("{detail_field} must be a JSON object"),
            )
            .with_detail("field", detail_field)
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)),
            // Preserve the same artifact context for a missing key as for a
            // wrong-type value. The error code tells callers which condition
            // occurred.
            None => Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("missing object field {detail_field}"),
            )
            .with_detail("field", detail_field)
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)),
        }
    }

    // Validate required content after the type check. Use trim only to decide
    // whether the supplied string has any content.
    fn require_nonempty_artifact_string(
        &self,
        value: &str,
        detail_field: &'static str,
        artifact: &Artifact,
        line: usize,
    ) -> Result<()> {
        if value.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark artifact field {detail_field} must not be empty"),
            )
            .with_detail("field", detail_field)
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    // Resolve the expected hash from the descriptor's manifest field before
    // opening a row stream. Every row and any final checksum error share that
    // descriptor.
    fn read_artifact_objects<'a>(&'a self, artifact: &'a Artifact) -> Result<ArtifactRows<'a>> {
        let expected = self
            // Use the root manifest's hashes object for member verification. It
            // was validated for known fields and types before artifact-specific
            // preflight.
            .hashes()?
            .get(artifact.hash_field)
            .and_then(JsonValue::as_str)
            // Use the full hashes path when an expected value was never declared.
            // A missing digest is a manifest error, independent of whether the
            // data file is readable.
            .ok_or_else(|| {
                self.missing_benchmark_field(&format!("hashes.{}", artifact.hash_field))
            })?;
        read_jsonl_objects(artifact, &self.manifest_path_label, &self.context, expected)
    }

    // Attach the physical line when a later validation stage provides one. The
    // shared helper retains a more specific location that the error already
    // carries.
    fn locate_artifact_error(&self, error: Error, artifact: &Artifact, line: usize) -> Error {
        locate_artifact_error(error, artifact, &self.manifest_path_label, Some(line))
    }

    // Text-only mode requires text and prohibits inline vectors. Check these rules
    // before any execution code can infer a modality from the query payload.
    fn validate_text_only_queries(&self, queries: &[QueryRow]) -> Result<()> {
        // Within this mode check, report vector presence before inspecting text.
        // The artifact loader also validates each row's text when it first parses
        // the workload.
        if let Some(row) = queries.iter().find(|row| row.has_embedding) {
            return Err(self.mode_conflict(format!(
                "none mode query {} at line {} must not inline embedding",
                row.query_id, row.line
            )));
        }
        // Treat an absent retained text value like blank text. Either leaves the
        // text-only request without searchable input.
        if let Some(row) = queries
            .iter()
            // Use trimming to test for a populated lexical channel. The core
            // search still applies its normal tokenization, so a nonblank string
            // can produce no searchable terms.
            .find(|row| row.text.as_deref().unwrap_or("").trim().is_empty())
        {
            // Include both query ID and source line in the validation failure so a
            // producer can locate the unusable row.
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "text-only benchmark query requires text",
            )
            .with_detail("field", "queries.text")
            .with_detail("query_id", &row.query_id)
            .with_detail("line", row.line)
            .with_detail("manifest_path", &self.manifest_path_label));
        }
        Ok(())
    }

    // Build an owned typed query vector and keep its allocation charged to the
    // caller's reservation. The result can survive deletion of the decoded
    // artifact row.
    fn vector_values(
        &self,
        value: &JsonValue,
        query_id: &str,
        line: usize,
        artifact_kind: &str,
        work: &mut JsonReservation,
    ) -> Result<Vec<f32>> {
        // Require the vector member to have array shape. Record which benchmark
        // artifact and query supplied it so this error remains actionable.
        let values = value.as_array().ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "missing array field vector",
            )
            .with_detail("field", "vector")
            .with_detail("artifact_kind", artifact_kind)
            .with_detail("query_id", query_id)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        let mut vector = Vec::new();
        // Charge typed capacity while the parsed JSON array is live. The actual
        // array length controls the reservation. The manifest and row dimensions
        // are declarations checked separately from allocation.
        work.reserve_vec(&mut vector, values.len(), JsonInputKind::QueryEmbedding)?;
        // Inspect every coordinate as a numeric JSON value. A valid prefix is
        // insufficient, since a later member can still invalidate the row.
        for entry in values {
            let value = entry.as_f64().ok_or_else(|| {
                Error::new(
                    // Use the vector validation category for a nonnumeric member.
                    // The array's presence was already established, so this
                    // failure describes its contents.
                    ErrorCode::InvalidVectorLength,
                    "query embedding vector entries must be finite numbers",
                )
                .with_detail("query_id", query_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Narrow before testing finiteness. JSON may hold a finite binary64
            // number that overflows when converted to a binary32 scoring
            // coordinate.
            let value = value as f32;
            if !value.is_finite() {
                return Err(Error::new(
                    ErrorCode::InvalidVectorLength,
                    "query embedding vector entries must be finite f32 values",
                )
                .with_detail("query_id", query_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
            vector.push(value);
        }
        // Check the library's typed query-vector processing rules after
        // conversion. This ensures the values passed to scoring also fit the
        // canonical spelling used for query limits.
        crate::json::validate_query_vector(&vector)?;
        // Transfer the typed vector out while its caller-held charge remains
        // active. The caller decides how long the query data must stay alive.
        Ok(vector)
    }

    // Validate a collection embedding row and return only its width. Preflight can
    // check its structure and binary32 range without keeping a second vector
    // allocation.
    fn validate_embedding_vector(
        &self,
        value: &JsonValue,
        chunk_external_id: &str,
        line: usize,
        artifact: &Artifact,
    ) -> Result<usize> {
        // Report the fully qualified embeddings.vector field and chunk identifier.
        // The caller can map this failure to the source row without interpreting
        // its position as a query index.
        let values = value.as_array().ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "missing array field vector",
            )
            .with_detail("field", "embeddings.vector")
            .with_detail("path", &artifact.raw_path)
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("chunk_external_id", chunk_external_id)
            .with_detail("line", line)
            .with_detail("manifest_path", &self.manifest_path_label)
        })?;
        // Check that every coordinate remains finite when narrowed to f32. Only
        // the length survives this pass. The importer builds its typed record when
        // it captures the artifact later.
        for entry in values {
            // Inspect the same binary64 values that import would narrow. The
            // preflight pass avoids allocating typed copies for an entire
            // embedding artifact.
            let value = entry.as_f64().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidVectorLength,
                    "embedding vector entries must be finite numbers",
                )
                .with_detail("field", "embeddings.vector")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("chunk_external_id", chunk_external_id)
                .with_detail("line", line)
                // Carry the manifest and member labels together. The chunk ID
                // identifies the embedding logically, and the physical line
                // identifies it in the member file.
                .with_detail("manifest_path", &self.manifest_path_label)
            })?;
            // Validate in the precision storage uses, since the original JSON
            // number's range may be wider. Return an artifact-specific error for
            // an unrepresentable coordinate.
            let value = value as f32;
            if !value.is_finite() {
                return Err(Error::new(
                    ErrorCode::InvalidVectorLength,
                    "embedding vector entries must be finite f32 values",
                )
                .with_detail("field", "embeddings.vector")
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("chunk_external_id", chunk_external_id)
                .with_detail("line", line)
                .with_detail("manifest_path", &self.manifest_path_label));
            }
        }
        // Return the array length after its contents passed. The caller compares
        // the actual width with manifest and row declarations independently.
        Ok(values.len())
    }

    // Report a field required by the manifest generally. Mode-dependent
    // requirements use the separate constructor so the selected source is visible
    // in the details.
    fn missing_benchmark_field(&self, field: &str) -> Error {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing required benchmark field {field}"),
        )
        .with_detail("field", field)
        .with_detail("manifest_path", &self.manifest_path_label)
    }

    // Record the mode that made this field mandatory. The same manifest key can be
    // required, optional or disallowed depending on how query vectors are
    // supplied.
    fn missing_source_field(&self, field: &'static str) -> Error {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing required field {field} for query_embedding_source"),
        )
        .with_detail("field", field)
        .with_detail(
            "query_embedding_source",
            self.query_embedding_source.as_str(),
        )
        .with_detail("manifest_path", &self.manifest_path_label)
    }

    // Use a mode-conflict error when the artifact supplies incompatible inputs.
    // Preserve the selected query embedding source and manifest label for
    // remediation.
    fn mode_conflict(&self, message: impl Into<String>) -> Error {
        Error::new(ErrorCode::BenchmarkArtifactModeConflict, message)
            .with_detail(
                // Expose the selected mode as data so a consumer need not parse it
                // out of the mode-conflict message.
                "query_embedding_source",
                self.query_embedding_source.as_str(),
            )
            .with_detail("manifest_path", &self.manifest_path_label)
    }
}

// Keep the manifest schema closed to the recognized top-level keys. Adding a new
// option elsewhere does not make a new manifest field accepted here.
fn reject_unknown_benchmark_manifest_fields(
    obj: &JsonObject,
    manifest_path_label: &str,
) -> Result<()> {
    // Inspect the entire root object for unknown keys. A recognizable subset
    // cannot excuse a typo or a field introduced by a newer unsupported schema.
    for field in obj.keys() {
        // Select the expected property family before checking mode combinations.
        // For example, a sidecar property can be known to this schema while being
        // disallowed for a text-only dataset.
        if !matches!(
            field.as_str(),
            "schema_version"
                | "dataset_id"
                | "version"
                | "documents"
                | "chunks"
                | "embeddings"
                | "queries"
                | "qrels"
                // Accept the fields used by any supported query embedding source
                // at this layer. Mode validation separately decides which of these
                // may coexist in one manifest.
                | "query_embedding_source"
                | "query_embeddings"
                | "embedding_model"
                | "query_embedding_model"
                | "query_embedding_dim"
                | "chunking"
                | "tokenizer"
                | "hashes"
        ) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark dataset manifest contains unsupported field",
            )
            // Use manifest as the error location for an unsupported root field.
            // This failure concerns the accepted shape of the root object.
            .with_detail("field", "manifest")
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

// Validate the hash object's field names independently of the manifest root. Each
// admitted name refers to one recognized artifact member.
fn reject_unknown_benchmark_hash_fields(obj: &JsonObject, manifest_path_label: &str) -> Result<()> {
    // Allow digest fields for artifacts that only some modes require. The selected
    // mode later checks required values and rejects conflicting members.
    for field in obj.keys() {
        if !matches!(
            field.as_str(),
            "documents_sha256"
                | "chunks_sha256"
                | "embeddings_sha256"
                | "queries_sha256"
                | "qrels_sha256"
                | "query_embeddings_sha256"
        ) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark dataset manifest hashes contains unsupported field",
            )
            // Name the hashes object for an unknown digest key. Known digest keys
            // receive a more specific hashes.member label when their associated
            // value is invalid.
            .with_detail("field", "hashes")
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

// Validate hash value types and presence before any file comparison. A string can
// still be wrong, so digest equality is checked against the actual artifact
// stream.
fn validate_benchmark_hash_field_values(obj: &JsonObject, manifest_path_label: &str) -> Result<()> {
    // Validate every digest value present in the manifest, including optional
    // ones. Later checks decide whether a recognized digest is applicable to the
    // selected mode.
    for (field, value) in obj {
        // Reject nonstring hash values before testing their content. Record the
        // fully qualified hashes member so the caller can repair the manifest.
        let Some(hash) = value.as_str() else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark manifest hash fields must be strings",
            )
            .with_detail("field", format!("hashes.{field}"))
            .with_detail("manifest_path", manifest_path_label));
        };
        // A blank hash is treated as missing information, while an actual digest
        // is taken verbatim for comparison later.
        if hash.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark manifest hash fields must not be empty",
            )
            .with_detail("field", format!("hashes.{field}"))
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

// Return an owned optional string when the value has that type. Callers that need
// to distinguish a missing key from a wrong type use explicit validation.
fn optional_string_field(obj: &JsonObject, field: &str) -> Option<String> {
    obj.get(field)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
}

// Pair the parsed row with its physical line and memory guard. The query parser
// carries the lexically captured top_k separately because the general JSON number
// cannot distinguish all adjacent integers at the boundary.
struct ArtifactObject {
    // Physical JSONL line used when locating schema and cross-artifact errors.
    line: usize,
    // Owned fields of this row after checking the artifact's allowed property
    // names.
    obj: JsonObject,
    // Memory charge kept with the parsed fields and transferable to retained
    // conversions.
    _work: JsonReservation,
    // Exact top_k captured by the query-row parser when present, before a JSON
    // number could lose integer precision.
    search_top_k: Option<usize>,
}

// Rows remain provisional until exhaustion verifies the digest of the parsed stream.
struct ArtifactRows<'a> {
    // Descriptor shared with every row and final digest error from this stream.
    artifact: &'a Artifact,
    // Identifies the owning manifest in failures surfaced by the iterator.
    manifest_path_label: &'a str,
    // Active JSONL parser over a hashing reader. Taking it marks the iterator
    // exhausted or failed.
    rows: Option<JsonLines<Sha256Reader<File>>>,
    // Expected digest checked at the end of the same byte stream that supplied the
    // rows.
    expected_sha256: &'a str,
}

// Enrich a row or stream error with the artifact location. Retain any already
// populated path or physical line supplied by a more specific layer.
fn locate_artifact_error(
    mut error: Error,
    artifact: &Artifact,
    manifest: &str,
    line: Option<usize>,
) -> Error {
    // Location is additive: core diagnostics, numeric types and original IO causes
    // survive unchanged. Only caller-supplied labels may enter public paths.
    // Use the source labels from the descriptor rather than resolved filesystem
    // paths. The benchmark identifies the caller's manifest and member by their
    // supplied spellings.
    for (key, value) in [
        ("path", artifact.raw_path.as_str()),
        ("artifact_kind", artifact.kind),
        ("manifest_path", manifest),
    ] {
        // Existing error details win when layers add context. A generic artifact
        // wrapper must not replace a more precise path or line produced by the
        // source failure.
        if !error.detail().iter().any(|(field, _)| field == key) {
            error = error.with_detail(key, value);
        }
    }
    // Only add line when this caller knows it. Open and end-of-file checksum
    // failures often belong to the whole artifact instead of a particular row.
    if let Some(line) = line {
        if !error.detail().iter().any(|(key, _)| key == "line") {
            error = error.with_detail("line", line);
        }
    }
    error
}

// Use a dedicated checksum mismatch error together with both digests. The manifest
// key and artifact label identify exactly which declared input failed
// verification.
fn artifact_hash_mismatch(
    artifact: &Artifact,
    manifest: &str,
    expected: &str,
    actual: &str,
) -> Error {
    Error::new(
        ErrorCode::ChecksumMismatch,
        format!("{} hash mismatch", artifact.kind),
    )
    // Point at the manifest's digest field, even though the bytes being checked
    // came from a member file.
    .with_detail("field", format!("hashes.{}", artifact.hash_field))
    .with_detail("path", &artifact.raw_path)
    .with_detail("artifact_kind", artifact.kind)
    // Report the expected and observed SHA-256 independently. A caller can verify
    // the manifest's recorded value without losing the digest computed from the
    // file.
    .with_detail("expected_sha256", expected)
    .with_detail("actual_sha256", actual)
    .with_detail("manifest_path", manifest)
}

// Hash the same descriptor that supplies the decoder. A successful digest covers
// the bytes consumed during this pass, including line separators. It does not say
// that a later open of the pathname will select the same file or contents.
fn read_jsonl_objects<'a>(
    artifact: &'a Artifact,
    manifest_path_label: &'a str,
    context: &JsonOperationContext,
    expected_sha256: &'a str,
) -> Result<ArtifactRows<'a>> {
    // Use the resolved path for I/O and the caller-supplied label for diagnostics.
    // An open failure occurs before there is a row number.
    let file = fs::File::open(&artifact.path)
        .map_err(|err| locate_artifact_error(err.into(), artifact, manifest_path_label, None))?;
    // Select the schema before reading rows. Query top-k and query embedding
    // arrays need their own parsing behavior, while qrels use the general
    // benchmark record form.
    let schema = match artifact.kind {
        "documents" => JsonSchema::DocumentRow,
        "chunks" => JsonSchema::ChunkRow,
        "embeddings" => JsonSchema::EmbeddingRow,
        "queries" => JsonSchema::QueryRow,
        // Use the dedicated query embedding record schema for sidecar rows, so its
        // vector member receives the same processing context as an inline query
        // vector.
        "query_embeddings" => JsonSchema::QueryEmbeddingObject,
        _ => JsonSchema::BenchmarkRecord,
    };
    // Apply the JSON line reader over the hashing input. This path uses the
    // maximum stream byte setting while row-level JSON limits still apply.
    let rows = JsonLines::new(Sha256Reader::new(file), schema, context, u64::MAX)
        // Construction can fail before the first decoded record. Add descriptor
        // context here and let the iterator add physical lines for errors raised
        // while reading.
        .map_err(|error| locate_artifact_error(error, artifact, manifest_path_label, None))?;
    Ok(ArtifactRows {
        artifact,
        manifest_path_label,
        rows: Some(rows),
        expected_sha256,
    })
}

// Yield one row at a time, with a terminal checksum error if all decoding finishes
// but the exact stream fails its expected digest.
impl Iterator for ArtifactRows<'_> {
    type Item = Result<ArtifactObject>;

    fn next(&mut self) -> Option<Self::Item> {
        // Give a row error its original precedence. Only normal exhaustion enters
        // final digest comparison.
        let row = match self.rows.as_mut()?.next() {
            Some(row) => row,
            // The normal EOF branch closes the digest over every raw byte read,
            // including blank lines and framing. Hashing reconstructed JSON would
            // lose this identity.
            None => {
                // Take the reader before comparing the digest. The final mismatch
                // is yielded once, and a successful comparison yields no
                // additional row. Consumers must exhaust the iterator to reach
                // either outcome.
                let actual = self.rows.take().unwrap().into_inner().finish();
                return (actual != self.expected_sha256).then(|| {
                    Err(artifact_hash_mismatch(
                        self.artifact,
                        self.manifest_path_label,
                        self.expected_sha256,
                        &actual,
                    ))
                });
            }
        };
        let result = (|| {
            // Use the same artifact labels for a parser failure, wrong root type
            // and unsupported key. A caller can identify the source consistently
            // across validation stages.
            let artifact = self.artifact;
            let manifest_path_label = self.manifest_path_label;
            // Preserve errors from the JSON line reader, including its physical
            // line, then supply missing artifact details. A row error aborts the
            // stream below.
            let (line, parsed) = row.map_err(|error| {
                locate_artifact_error(error, artifact, manifest_path_label, None)
            })?;
            // Keep the parsed reservation and exact top-k alongside the object
            // fields. Moving the map out avoids copying its keys and values.
            let (mut value, work, search_top_k) = parsed.into_parts_with_search_top_k();
            // Reject scalar or array rows before inspecting allowed fields. An
            // artifact can have valid JSON syntax without containing the required
            // object shape.
            let obj = value.as_object_mut().map(std::mem::take).ok_or_else(|| {
                Error::new(
                    // Distinguish valid JSON with a non-object root from
                    // unreadable JSON. Downstream artifact validators expect a map
                    // of named fields.
                    ErrorCode::InvalidJsonlSchema,
                    format!("{} row must be a JSON object", artifact.kind),
                )
                .with_detail("path", &artifact.raw_path)
                .with_detail("artifact_kind", artifact.kind)
                .with_detail("line", line)
                .with_detail("manifest_path", manifest_path_label)
            })?;
            // Check allowed keys before handing the object to artifact-specific
            // logic. Common row checks stay with the iterator so every consumer
            // receives the same validation.
            reject_unknown_benchmark_artifact_fields(&obj, artifact, line, manifest_path_label)?;
            // The caller can move individual fields out of this object. Carry the
            // original reservation through that transfer so handing off the map
            // cannot release credit for the still-live data.
            Ok(ArtifactObject {
                line,
                obj,
                _work: work,
                search_top_k,
            })
        })();
        // Discard the active reader on an error. Once validation stops, a later
        // next call must not surface a checksum for only the consumed prefix.
        if result.is_err() {
            self.rows = None;
        }
        Some(result)
    }
}

// The terminal state is represented by absence of the reader. Both successful EOF
// and any yielded error leave subsequent calls at None.
impl std::iter::FusedIterator for ArtifactRows<'_> {}

// Match allowed properties to the artifact's declared kind. Validation of required
// members and their values belongs to callers parsing that specific row type.
fn reject_unknown_benchmark_artifact_fields(
    obj: &JsonObject,
    artifact: &Artifact,
    line: usize,
    manifest_path_label: &str,
) -> Result<()> {
    // Select the row-level allowlist for known benchmark members. Unknown
    // descriptor kinds have no special property screening in this helper.
    let allowed = match artifact.kind {
        "documents" => BENCHMARK_DOCUMENT_FIELDS,
        "chunks" => BENCHMARK_CHUNK_FIELDS,
        "embeddings" => BENCHMARK_EMBEDDING_FIELDS,
        "queries" => BENCHMARK_QUERY_FIELDS,
        "qrels" => BENCHMARK_QREL_FIELDS,
        "query_embeddings" => BENCHMARK_QUERY_EMBEDDING_FIELDS,
        // Qrels uses the general parser schema and has an explicit allowlist
        // above. A new kind that falls through both matches would be decoded
        // without row-specific property checks.
        _ => return Ok(()),
    };
    // Check only top-level keys. Nested metadata and other fields have their own
    // validation rules where they are interpreted.
    for field in obj.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark artifact row contains unsupported field",
            )
            .with_detail("field", artifact.kind)
            .with_detail("path", &artifact.raw_path)
            // Report the artifact identity and physical row together. A disallowed
            // top-level field has the same validation category across artifacts.
            .with_detail("artifact_kind", artifact.kind)
            .with_detail("line", line)
            .with_detail("manifest_path", manifest_path_label));
        }
    }
    Ok(())
}

// Report mismatched dimensions with the manifest width, row declaration and actual
// vector length. Keeping all three values identifies which source contradicted the
// rest.
fn invalid_vector_len(
    manifest_path: &str,
    query_id: &str,
    expected_dim: usize,
    row_dim: usize,
    actual_vector_length: usize,
) -> Error {
    Error::new(
        ErrorCode::InvalidVectorLength,
        "query embedding dimension does not match dataset manifest",
    )
    .with_detail("query_id", query_id)
    // Keep independent dimensions in typed numeric details for callers that
    // compare them programmatically.
    .with_u64_detail("expected_dim", expected_dim as u64)
    .with_u64_detail("row_dim", row_dim as u64)
    .with_u64_detail("actual_vector_length", actual_vector_length as u64)
    .with_detail("manifest_path", manifest_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256_hex;

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("cofferdb-benchmark-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_file(dir: &Path, name: &str, content: &str) -> String {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        sha256_hex(&fs::read(path).unwrap())
    }

    #[test]
    fn streamed_manifest_hash_includes_the_exact_original_bytes() {
        let dir = temp_dir("streamed-manifest-hash");
        let path = dir.join("dataset_manifest.json");
        let raw = format!(
            "{}{{\"query_embedding_source\":\"none\"}}\n\t",
            " ".repeat(20_000)
        );
        fs::write(&path, &raw).unwrap();
        let input =
            BenchmarkManifestInput::load(&path, &JsonOperationContext::standalone()).unwrap();
        assert_eq!(input.manifest_sha256, sha256_hex(raw.as_bytes()));
        assert_ne!(
            input.manifest_sha256,
            sha256_hex(JsonValue::Object(input.obj).to_compact_string().as_bytes())
        );
        fs::remove_dir_all(dir).unwrap();
    }

    fn write_sidecar_fixture_with_artifacts(
        dir: &Path,
        documents_content: &str,
        chunks_content: &str,
        embeddings_content: &str,
        queries_content: &str,
        qrels_content: &str,
        query_embeddings_content: &str,
    ) {
        let documents = write_file(dir, "documents.jsonl", documents_content);
        let chunks = write_file(dir, "chunks.jsonl", chunks_content);
        let embeddings = write_file(dir, "embeddings.jsonl", embeddings_content);
        let queries = write_file(dir, "queries.jsonl", queries_content);
        let qrels = write_file(dir, "qrels.jsonl", qrels_content);
        let query_embeddings = write_file(dir, "query_embeddings.jsonl", query_embeddings_content);
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
    }

    fn detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    fn assert_artifact_error_is_additive(error: Error, line: Option<usize>) -> Error {
        let before = error.to_json();
        let had_source = std::error::Error::source(&error).is_some();
        let artifact = Artifact {
            raw_path: "inputs/queries.jsonl".into(),
            path: PathBuf::from("/private/internal-unpublished-artifact/queries.jsonl"),
            kind: "queries",
            hash_field: "queries_sha256",
        };
        let located = locate_artifact_error(error, &artifact, "dataset.json", line);
        let once = located.to_json();
        let located = locate_artifact_error(located, &artifact, "other.json", Some(999));
        let after = located.to_json();
        for key in ["code", "message", "recoverable", "suggested_action"] {
            assert_eq!(before.get(key), after.get(key), "{key}");
        }
        for (key, value) in before.get("detail").unwrap().as_object().unwrap() {
            assert_eq!(after.get("detail").unwrap().get(key), Some(value), "{key}");
        }
        for (key, value) in once.get("detail").unwrap().as_object().unwrap() {
            assert_eq!(after.get("detail").unwrap().get(key), Some(value), "{key}");
        }
        assert_eq!(std::error::Error::source(&located).is_some(), had_source);
        let unique = located
            .detail()
            .iter()
            .map(|(key, _)| key)
            .collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), located.detail().len());
        assert!(!after
            .to_compact_string()
            .contains("internal-unpublished-artifact"));
        located
    }

    #[test]
    fn artifact_location_preserves_resource_contract_and_open_operations_at_100_boundaries() {
        for seed in 0..100_u128 {
            for resource in ["memory", "catalog", "temporary_disk"] {
                for limit in [0, (1_u64 << 53) + 1, u64::MAX] {
                    let requested = if seed % 2 == 0 {
                        u128::from(limit) + seed + 1
                    } else {
                        u128::MAX - seed
                    };
                    let operation = format!("future_bounded_operation_{seed}");
                    let error =
                        Error::new(ErrorCode::ResourceExhausted, "resource budget exhausted")
                            .with_detail("resource", resource)
                            .with_detail("operation", &operation)
                            .with_detail("limit_bytes", limit)
                            .with_detail("requested_bytes", requested);
                    let error = assert_artifact_error_is_additive(error, Some(seed as usize + 1));
                    assert_eq!(detail_value(&error, "operation"), Some(operation.as_str()));
                    assert!(!crate::json::is_limit_error(&error));
                    let json = error.to_json();
                    assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(true)));
                    let detail = json.get("detail").unwrap();
                    assert_eq!(
                        detail.get("limit_bytes").unwrap().as_str(),
                        Some(limit.to_string().as_str())
                    );
                    assert_eq!(
                        detail.get("requested_bytes").unwrap().as_str(),
                        Some(requested.to_string().as_str())
                    );
                    assert_eq!(
                        detail.get("path").unwrap().as_str(),
                        Some("inputs/queries.jsonl")
                    );
                }
            }
        }
    }

    #[test]
    fn artifact_location_preserves_json_index_io_and_integrity_diagnostics() {
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::BenchmarkArtifact,
            JsonInputKind::Filter,
            JsonInputKind::Weights,
            JsonInputKind::MetadataBoost,
            JsonInputKind::StoredJson,
        ] {
            let context = JsonOperationContext::standalone();
            let error = context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES + 1,
                    kind,
                )
                .unwrap_err();
            let error = assert_artifact_error_is_additive(error, None);
            assert!(crate::json::is_limit_error(&error));
            assert!(detail_value(&error, "resource").is_none());
        }
        let errors = [
            Error::new(ErrorCode::IndexCorruption, "index unavailable")
                .with_detail(
                    "index_unavailable_reason",
                    "transient_rebuild_budget_exceeded",
                )
                .with_detail("index_source_attempted", "transient_rebuild")
                .with_detail("resource_profile", "constrained")
                .with_u64_detail("budget_mb", 64),
            Error::from(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
            Error::from(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
            Error::new(ErrorCode::ChecksumMismatch, "hash mismatch")
                .with_detail("expected_sha256", "a".repeat(64))
                .with_detail("actual_sha256", "b".repeat(64)),
            Error::new(ErrorCode::FileFormatError, "invalid header"),
        ];
        for error in errors {
            let error = assert_artifact_error_is_additive(error, Some(2));
            assert!(detail_value(&error, "resource").is_none());
        }
    }

    #[test]
    fn artifact_location_never_overwrites_existing_core_locator_or_integer_types() {
        let error = Error::new(ErrorCode::InvalidEmbeddingDimension, "dimension mismatch")
            .with_detail("field", "vector")
            .with_detail("path", "original/sidecar.jsonl")
            .with_detail("artifact_kind", "query_embeddings")
            .with_detail("manifest_path", "original/dataset.json")
            .with_u64_detail("line", 7)
            .with_u64_detail("expected_dim", 768)
            .with_u64_detail("actual_vector_length", 1);
        let before = error.to_json();
        let after = assert_artifact_error_is_additive(error, Some(19)).to_json();
        assert_eq!(before, after);
    }

    #[test]
    fn artifact_parsing_rechecks_the_bytes_after_initial_hash_validation() {
        let dir = temp_dir("artifact-stream-identity");
        write_sidecar_fixture_with_artifacts(
            &dir,
            r#"{"external_id":"doc","metadata":{}}"#,
            r#"{"document_external_id":"doc","chunk_external_id":"c","ordinal":0,"text":"needle","metadata":{}}"#,
            r#"{"chunk_external_id":"c","model_name":"m","dim":2,"vector":[1,0]}"#,
            r#"{"query_id":"q","text":"needle","filter":{},"top_k":1}"#,
            r#"{"query_id":"q","chunk_external_id":"c","relevance":1}"#,
            r#"{"query_id":"q","model_name":"m","dim":2,"vector":[1,0]}"#,
        );
        let kinds = [
            "documents",
            "chunks",
            "embeddings",
            "queries",
            "qrels",
            "query_embeddings",
        ];
        let context = JsonOperationContext::standalone();
        for seed in 0..100 {
            let kind = kinds[seed % kinds.len()];
            let artifact = Artifact {
                raw_path: format!("{kind}.jsonl"),
                path: dir.join(format!("{kind}.jsonl")),
                kind,
                hash_field: match kind {
                    "documents" => "documents_sha256",
                    "chunks" => "chunks_sha256",
                    "embeddings" => "embeddings_sha256",
                    "queries" => "queries_sha256",
                    "qrels" => "qrels_sha256",
                    _ => "query_embeddings_sha256",
                },
            };
            let before = fs::read(&artifact.path).unwrap();
            let input = BenchmarkManifestInput::load(&dir, &context).unwrap();
            input.validate_hash(&artifact).unwrap();
            let mut changed = before.clone();
            changed.extend(vec![b' '; seed * 211 + 1]);
            fs::write(&artifact.path, &changed).unwrap();
            let mut rows = input.read_artifact_objects(&artifact).unwrap();
            let parsed = rows.by_ref().collect::<Result<Vec<_>>>();
            let error = parsed
                .err()
                .expect("parsed artifact must match its frozen input hash");
            assert!(rows.next().is_none());
            assert!(rows.next().is_none());
            drop(rows);
            assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
            assert_eq!(detail_value(&error, "artifact_kind"), Some(kind));
            assert_eq!(
                detail_value(&error, "expected_sha256"),
                Some(sha256_hex(&before).as_str())
            );
            assert_eq!(
                detail_value(&error, "actual_sha256"),
                Some(sha256_hex(&changed).as_str())
            );
            fs::write(&artifact.path, before).unwrap();
            for row in input.read_artifact_objects(&artifact).unwrap() {
                row.unwrap();
            }
            drop(input);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn artifact_contract_errors_do_not_expose_later_rows() {
        let dir = temp_dir("artifact-contract-fused");
        let artifact = Artifact {
            raw_path: "queries.jsonl".into(),
            path: dir.join("queries.jsonl"),
            kind: "queries",
            hash_field: "queries_sha256",
        };
        for invalid in [r#"{"unexpected":"private_value"}"#, "[]", "{"] {
            let bytes = format!("{invalid}\n{{\"query_id\":\"must_not_be_exposed\"}}\n");
            fs::write(&artifact.path, &bytes).unwrap();
            let hash = sha256_hex(bytes.as_bytes());
            let context = JsonOperationContext::standalone();
            let mut rows =
                read_jsonl_objects(&artifact, "dataset_manifest.json", &context, &hash).unwrap();
            let error = rows.next().unwrap().err().unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(detail_value(&error, "line"), Some("1"));
            assert!(!error.to_string().contains("private_value"));
            assert!(rows.next().is_none());
            assert!(rows.next().is_none());
            drop(rows);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn artifact_rows_are_incremental_and_do_not_hide_later_errors() {
        let dir = temp_dir("incremental-jsonl");
        let path = dir.join("queries.jsonl");
        fs::write(&path, format!("{{\"query_id\":\"first\",\"text\":\"payload\",\"top_k\":1}}\n\n{{\"query_id\":\"second\",\"filter\":{}0{}}}\n", "[".repeat(30_000), "]".repeat(30_000))).unwrap();
        let artifact = Artifact {
            raw_path: "queries.jsonl".into(),
            path,
            kind: "queries",
            hash_field: "queries_sha256",
        };
        let hash = sha256_hex(&fs::read(&artifact.path).unwrap());
        let mut rows = read_jsonl_objects(
            &artifact,
            "dataset_manifest.json",
            &JsonOperationContext::standalone(),
            &hash,
        )
        .unwrap();
        let first = rows.next().unwrap().unwrap();
        assert_eq!(first.line, 1);
        assert_eq!(first.obj.get("query_id").unwrap().as_str(), Some("first"));
        assert!(first._work.reserved_bytes() > 0);
        let error = rows.next().unwrap().err().unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
        assert_eq!(detail_value(&error, "line"), Some("3"));
        assert_eq!(detail_value(&error, "json_input_kind"), Some("filter"));
        assert_eq!(detail_value(&error, "path"), Some("queries.jsonl"));
        assert!(rows.next().is_none());
        drop((first, rows));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn deep_jsonl_artifact_keeps_depth_reason_and_private_locator() {
        let dir = temp_dir("json-depth");
        let path = dir.join("queries.jsonl");
        fs::write(
            &path,
            format!(
                "{{\"filter\":{}0{}}}\n",
                "[".repeat(30_000),
                "]".repeat(30_000)
            ),
        )
        .unwrap();
        let artifact = Artifact {
            raw_path: "queries.jsonl".into(),
            path,
            kind: "queries",
            hash_field: "queries_sha256",
        };
        let err = read_jsonl_objects(
            &artifact,
            "dataset_manifest.json",
            &JsonOperationContext::standalone(),
            &sha256_hex(&fs::read(&artifact.path).unwrap()),
        )
        .unwrap()
        .next()
        .unwrap()
        .err()
        .unwrap();
        assert_eq!(err.code(), ErrorCode::InvalidMetadataFilter);
        for (field, expected) in [
            ("json_limit_reason", "depth_exceeded"),
            ("limit", "128"),
            ("line", "1"),
            ("path", "queries.jsonl"),
            ("artifact_kind", "queries"),
            ("manifest_path", "dataset_manifest.json"),
        ] {
            assert_eq!(detail_value(&err, field), Some(expected));
        }
        assert!(!err.message().contains(dir.to_str().unwrap()));
        fs::remove_dir_all(dir).unwrap();
    }

    fn write_sidecar_fixture(
        dir: &Path,
        queries_content: &str,
        qrels_content: &str,
        query_embeddings_content: &str,
    ) {
        let documents = write_file(
            dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(dir, "queries.jsonl", queries_content);
        let qrels = write_file(dir, "qrels.jsonl", qrels_content);
        let query_embeddings = write_file(dir, "query_embeddings.jsonl", query_embeddings_content);
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
    }

    fn write_sidecar_fixture_with_chunks_and_embeddings(
        dir: &Path,
        chunks_content: &str,
        embeddings_content: &str,
    ) {
        let documents = write_file(
            dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(dir, "chunks.jsonl", chunks_content);
        let embeddings = write_file(dir, "embeddings.jsonl", embeddings_content);
        let queries = write_file(
            dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
    }

    fn write_none_fixture(dir: &Path, queries_content: &str, qrels_content: &str) {
        let documents = write_file(
            dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{"lang":"rust"}}"#,
        );
        let queries = write_file(dir, "queries.jsonl", queries_content);
        let qrels = write_file(dir, "qrels.jsonl", qrels_content);
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();
    }

    #[test]
    fn benchmark_query_filters_share_work_across_rows_and_release_on_failure() {
        let dir = temp_dir("query-shared-work");
        let label = "x".repeat(256 * 1024);
        let queries = (0..2).map(|index| format!("{{\"query_id\":\"private_query_{index}\",\"text\":\"wal\",\"filter\":{{\"chunk.private_key\":\"{label}\"}},\"top_k\":10}}\n")).collect::<String>();
        let qrels = (0..2).map(|index| format!("{{\"query_id\":\"private_query_{index}\",\"chunk_external_id\":\"doc#1\",\"relevance\":1}}\n")).collect::<String>();
        write_none_fixture(&dir, &queries, &qrels);
        for available in [1280 * 1024, 2 * 1024 * 1024] {
            let context = JsonOperationContext::standalone();
            let mut busy = context.reservation();
            busy.grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
            let result = BenchmarkDataset::load_and_validate_in(&dir, &context);
            if available == 1280 * 1024 {
                let error = result.err().unwrap();
                assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
                assert_eq!(
                    detail_value(&error, "json_limit_reason"),
                    Some("work_budget_exceeded")
                );
                assert_eq!(detail_value(&error, "json_input_kind"), Some("filter"));
                assert_eq!(detail_value(&error, "line"), Some("2"));
                assert_eq!(detail_value(&error, "path"), Some("queries.jsonl"));
                assert_eq!(detail_value(&error, "artifact_kind"), Some("queries"));
                assert!(detail_value(&error, "query_id").is_none());
                assert!(!error.to_json().to_compact_string().contains("private_"));
            } else {
                let (dataset, work) = result.unwrap();
                assert_eq!(dataset.queries.len(), 2);
                let expected = dataset
                    .queries
                    .iter()
                    .map(|query| {
                        query
                            .filter
                            .reserve_retained_in(
                                JsonInputKind::Filter,
                                &JsonOperationContext::standalone(),
                            )
                            .unwrap()
                            .reserved_bytes()
                    })
                    .sum::<usize>();
                assert_eq!(work.reserved_bytes(), expected);
                drop(dataset);
                drop(work);
            }
            busy.grow(available, JsonInputKind::BenchmarkArtifact)
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn benchmark_query_vectors_and_manifest_values_retain_exact_owned_work() {
        let dir = temp_dir("query-vector-retained-work");
        let queries = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"text\":\"wal\",\"filter\":{{\"chunk.n\":{index}}},\"top_k\":10}}\n")).collect::<String>();
        let qrels = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"chunk_external_id\":\"doc#1\",\"relevance\":1}}\n")).collect::<String>();
        let mut sidecar = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[{index},0.1]}}\n")).collect::<String>();
        sidecar.push_str(
            "{\"query_id\":\"unused\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}\n",
        );
        for inline in [false, true] {
            write_sidecar_fixture(&dir, &queries, &qrels, &sidecar);
            if inline {
                let queries = (0..100).map(|index| format!("{{\"query_id\":\"q{index}\",\"text\":\"wal\",\"filter\":{{\"chunk.n\":{index}}},\"top_k\":10,\"embedding\":[{index},0.1]}}\n")).collect::<String>();
                let hash = write_file(&dir, "queries.jsonl", &queries);
                let path = dir.join("dataset_manifest.json");
                let mut manifest = crate::parse_json(&fs::read_to_string(&path).unwrap()).unwrap();
                let obj = manifest.as_object_mut().unwrap();
                obj.insert(
                    "query_embedding_source".into(),
                    JsonValue::String("inline".into()),
                );
                obj.remove("query_embeddings");
                let hashes = obj.get_mut("hashes").unwrap().as_object_mut().unwrap();
                hashes.remove("query_embeddings_sha256");
                hashes.insert("queries_sha256".into(), JsonValue::String(hash));
                fs::write(&path, manifest.to_compact_string()).unwrap();
            }
            let context = JsonOperationContext::standalone();
            let (dataset, work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
            let expected_context = JsonOperationContext::standalone();
            let mut expected = dataset
                .manifest
                .chunking
                .reserve_retained_in(JsonInputKind::Manifest, &expected_context)
                .unwrap();
            for (index, query) in dataset.queries.iter().enumerate() {
                expected.merge(
                    query
                        .filter
                        .reserve_retained_in(JsonInputKind::Filter, &expected_context)
                        .unwrap(),
                );
                let embedding = query.embedding.as_ref().unwrap();
                expected.merge(embedding.reserve_json_work_in(&expected_context).unwrap());
                assert_eq!(embedding.vector, vec![index as f32, 0.1]);
                assert_eq!(embedding.model_name.as_deref(), Some("m"));
                assert_eq!(
                    query.filter.get("chunk.n").and_then(JsonValue::as_f64),
                    Some(index as f64)
                );
            }
            assert_eq!(work.reserved_bytes(), expected.reserved_bytes());
            drop(dataset);
            drop(work);
            context
                .reservation()
                .grow(
                    crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn benchmark_typed_vector_reserves_before_copy_and_preserves_f32_validation() {
        let dir = temp_dir("query-vector-conversion-budget");
        let path = dir.join("dataset_manifest.json");
        fs::write(&path, r#"{"query_embedding_source":"none"}"#).unwrap();
        let context = JsonOperationContext::standalone();
        let input = BenchmarkManifestInput::load(&path, &context).unwrap();
        let source = JsonValue::Array(vec![JsonValue::Number(1.0); 256]);
        let mut busy = context.reservation();
        let used = input.work.reserved_bytes();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - used - 1023,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        let mut conversion = context.reservation();
        let error = input
            .vector_values(&source, "q", 3, "queries", &mut conversion)
            .unwrap_err();
        assert_eq!(
            detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        assert_eq!(
            detail_value(&error, "json_input_kind"),
            Some("query_embedding")
        );
        assert_eq!(conversion.reserved_bytes(), 0);
        drop(busy);
        for index in 0..100 {
            let source = JsonValue::Array(vec![
                JsonValue::Number(index as f64 / 10.0),
                JsonValue::Number(-0.0),
            ]);
            let mut conversion = context.reservation();
            let vector = input
                .vector_values(&source, "q", 3, "queries", &mut conversion)
                .unwrap();
            assert_eq!(vector[0].to_bits(), (index as f32 / 10.0).to_bits());
            assert_eq!(vector[1].to_bits(), (-0.0f32).to_bits());
            assert_eq!(conversion.reserved_bytes(), vector.capacity() * 4);
        }
        let mut conversion = context.reservation();
        let error = input
            .vector_values(
                &JsonValue::Array(vec![JsonValue::Number(f64::MAX)]),
                "q",
                3,
                "queries",
                &mut conversion,
            )
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidVectorLength);
        drop(conversion);
        drop(input);
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sidecar_dataset_manifest_validates_hashes_and_query_embeddings() {
        let dir = temp_dir("sidecar-ok");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""chunking":{{"chunk_size":800,"chunk_overlap":120}},"#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let (dataset, _work, paths) =
            BenchmarkDataset::load_with_sources_in(&dir, &JsonOperationContext::standalone())
                .unwrap();
        assert_eq!(
            paths.into_iter().collect::<BTreeSet<_>>(),
            [
                "dataset_manifest.json",
                "documents.jsonl",
                "chunks.jsonl",
                "embeddings.jsonl",
                "queries.jsonl",
                "qrels.jsonl",
                "query_embeddings.jsonl",
            ]
            .map(|name| dir.join(name))
            .into_iter()
            .collect()
        );
        let manifest = dataset.manifest;
        assert_eq!(manifest.dataset_id, "bench");
        assert_eq!(
            manifest
                .chunking
                .get("chunk_size")
                .and_then(JsonValue::as_f64),
            Some(800.0)
        );
        assert_eq!(
            manifest.query_embedding_source,
            QueryEmbeddingSource::Sidecar
        );
        assert_eq!(manifest.query_embedding_dim, Some(2));
        assert_eq!(manifest.query_embeddings_sha256, Some(query_embeddings));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_row_errors_report_the_actual_line_and_relative_artifact() {
        let dir = temp_dir("sidecar-error-locators");
        let rows = [
            (
                r#"{"query_id":"q1","model_name":"other","dim":2,"vector":[1,0]}"#,
                ErrorCode::EmbeddingModelMismatch,
            ),
            (
                r#"{"query_id":"q1","model_name":"m","dim":3,"vector":[1,0]}"#,
                ErrorCode::InvalidVectorLength,
            ),
            (
                r#"{"query_id":"q1","model_name":"m","dim":"private-dim","vector":[1,0]}"#,
                ErrorCode::MissingRequiredField,
            ),
            (
                concat!(
                    r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1,0]}"#,
                    "\n",
                    r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[0,1]}"#
                ),
                ErrorCode::InvalidJsonlSchema,
            ),
        ];
        for seed in 0..100 {
            let (row, code) = rows[seed % rows.len()];
            let blank_lines = seed / rows.len();
            write_sidecar_fixture(
                &dir,
                r#"{"query_id":"q1","text":"private query text","top_k":10}"#,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
                &format!("{}{row}\n", "\n".repeat(blank_lines)),
            );
            let error = BenchmarkDataset::load_and_validate(&dir).err().unwrap();
            assert_eq!(error.code(), code, "case {seed}");
            assert_eq!(detail_value(&error, "path"), Some("query_embeddings.jsonl"));
            assert_eq!(
                detail_value(&error, "artifact_kind"),
                Some("query_embeddings")
            );
            let expected_line = blank_lines + if seed % rows.len() == 3 { 2 } else { 1 };
            assert_eq!(
                detail_value(&error, "line"),
                Some(expected_line.to_string().as_str())
            );
            let wire = error.to_json().to_compact_string();
            assert!(!wire.contains("private-dim"));
            assert!(!wire.contains("private query text"));
            assert_eq!(detail_value(&error, "manifest_path"), dir.to_str());
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sidecar_rejects_duplicate_and_model_mismatch_rows() {
        let dir = temp_dir("sidecar-duplicate");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            concat!(
                r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
                "\n",
                r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[0.9,0.1]}"#
            ),
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("sidecar-model-mismatch");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"other","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::EmbeddingModelMismatch);
        assert_eq!(detail_value(&err, "row_model_name"), Some("other"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn inline_mode_requires_model_and_valid_embedding_dim() {
        let dir = temp_dir("inline-missing-model");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10,"embedding":[1.0,0.0]}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"inline","#,
                    r#""embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("query_embedding_model"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("inline-bad-dim");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10,"embedding":[1.0]}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"inline","#,
                    r#""embedding_model":"m","query_embedding_model":"m","#,
                    r#""query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();
        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("1"));
        assert_eq!(detail_value(&err, "path"), Some("queries.jsonl"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        assert_eq!(detail_value(&err, "line"), Some("1"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn query_embedding_dim_must_be_positive() {
        let dir = temp_dir("query-embedding-dim-zero");
        write_sidecar_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let manifest_path = dir.join("dataset_manifest.json");
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap()
            .replace(r#""query_embedding_dim":2"#, r#""query_embedding_dim":0"#);
        fs::write(&manifest_path, manifest).unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "field"), Some("query_embedding_dim"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_requires_query_embedding_for_each_query() {
        let dir = temp_dir("sidecar-missing-query-id");
        write_sidecar_fixture(
            &dir,
            concat!(
                r#"{"query_id":"q1","text":"wal","top_k":10}"#,
                "\n",
                r#"{"query_id":"q2","text":"log","top_k":10}"#
            ),
            concat!(
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
                "\n",
                r#"{"query_id":"q2","chunk_external_id":"doc#1","relevance":1}"#
            ),
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "query_id"), Some("q2"));
        assert_eq!(
            detail_value(&err, "field"),
            Some("query_embeddings[query_id]")
        );
        assert_eq!(detail_value(&err, "path"), Some("query_embeddings.jsonl"));
        assert_eq!(
            detail_value(&err, "artifact_kind"),
            Some("query_embeddings")
        );
        assert_eq!(detail_value(&err, "line"), None);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_rejects_query_embedding_dim_mismatch() {
        let dir = temp_dir("sidecar-dim-mismatch");
        write_sidecar_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
            r#"{"query_id":"q1","model_name":"m","dim":3,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "expected_dim"), Some("2"));
        assert_eq!(detail_value(&err, "row_dim"), Some("3"));
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn inline_dataset_manifest_loads_inline_query_embeddings() {
        let dir = temp_dir("inline-ok");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10,"embedding":[1.0,0.0]}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"inline","#,
                    r#""embedding_model":"m","query_embedding_model":"m","#,
                    r#""query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();

        let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
        assert_eq!(
            dataset.manifest.query_embedding_source,
            QueryEmbeddingSource::Inline
        );
        assert_eq!(dataset.manifest.query_embeddings_sha256, None);
        assert_eq!(dataset.queries.len(), 1);
        assert_eq!(dataset.queries[0].embedding.as_ref().unwrap().dim, 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn none_mode_rejects_query_embedding_fields() {
        let dir = temp_dir("none-conflict");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","query_embedding_model":"m","#,
                    r#""hashes":{{"documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
        assert_eq!(detail_value(&err, "query_embedding_source"), Some("none"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn none_mode_rejects_embedding_artifacts() {
        let dir = temp_dir("none-embedding-conflict");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let embeddings = write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::BenchmarkArtifactModeConflict);
        assert_eq!(detail_value(&err, "query_embedding_source"), Some("none"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_requires_embeddings_hash() {
        let dir = temp_dir("sidecar-missing-embedding-hash");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        write_file(
            &dir,
            "embeddings.jsonl",
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let query_embeddings = write_file(
            &dir,
            "query_embeddings.jsonl",
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}","#,
                    r#""query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(
            detail_value(&err, "field"),
            Some("hashes.embeddings_sha256")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_require_manifest_model() {
        let dir = temp_dir("embeddings-missing-model");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let documents = sha256_hex(&fs::read(dir.join("documents.jsonl")).unwrap());
        let chunks = sha256_hex(&fs::read(dir.join("chunks.jsonl")).unwrap());
        let embeddings = sha256_hex(&fs::read(dir.join("embeddings.jsonl")).unwrap());
        let queries = sha256_hex(&fs::read(dir.join("queries.jsonl")).unwrap());
        let qrels = sha256_hex(&fs::read(dir.join("qrels.jsonl")).unwrap());
        let query_embeddings = sha256_hex(&fs::read(dir.join("query_embeddings.jsonl")).unwrap());
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","#,
                    r#""query_embedding_model":"m","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("embedding_model"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_must_reference_declared_chunks() {
        let dir = temp_dir("embeddings-unknown-chunk");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"missing#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("missing#1"));
        assert_eq!(
            detail_value(&err, "field"),
            Some("embeddings.chunk_external_id")
        );
        assert_eq!(detail_value(&err, "source"), Some("embeddings.jsonl"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_reject_duplicate_chunk_rows() {
        let dir = temp_dir("embeddings-duplicate-chunk");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            concat!(
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
                "\n",
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[0.0,1.0]}"#
            ),
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_validate_model_dim_and_vector() {
        let dir = temp_dir("embeddings-model-mismatch");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"other","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::EmbeddingModelMismatch);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "manifest_embedding_model"), Some("m"));
        assert_eq!(detail_value(&err, "row_model_name"), Some("other"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("embeddings-dim-mismatch");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#2","ordinal":2,"text":"checkpoint","metadata":{}}"#
            ),
            concat!(
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
                "\n",
                r#"{"chunk_external_id":"doc#2","model_name":"m","dim":3,"vector":[0.0,1.0]}"#
            ),
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#2"));
        assert_eq!(detail_value(&err, "expected_dim"), Some("2"));
        assert_eq!(detail_value(&err, "row_dim"), Some("3"));
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("2"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("embeddings-vector-mismatch");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "expected_dim"), Some("2"));
        assert_eq!(detail_value(&err, "row_dim"), Some("2"));
        assert_eq!(detail_value(&err, "actual_vector_length"), Some("1"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_embeddings_require_complete_chunk_coverage() {
        let dir = temp_dir("embeddings-missing-chunk-row");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#2","ordinal":2,"text":"checkpoint","metadata":{}}"#
            ),
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#2"));
        assert_eq!(
            detail_value(&err, "field"),
            Some("embeddings[chunk_external_id]")
        );
        assert_eq!(detail_value(&err, "source"), Some("embeddings.jsonl"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hash_mismatch_reports_expected_and_actual_without_payload() {
        let dir = temp_dir("hash-mismatch");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"0000","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        assert_eq!(detail_value(&err, "expected_sha256"), Some("0000"));
        assert_eq!(detail_value(&err, "actual_sha256"), Some(queries.as_str()));
        assert!(err.to_string().contains("hash mismatch"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn jsonl_parse_errors_report_artifact_locator_without_payload() {
        let dir = temp_dir("query-malformed-json");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"do not echo private query text","top_k":10,"embedding":[1.0,"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "path"), Some("queries.jsonl"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        assert_eq!(detail_value(&err, "line"), Some("1"));
        let manifest_label = dir.display().to_string();
        assert_eq!(
            detail_value(&err, "manifest_path"),
            Some(manifest_label.as_str())
        );
        let public_json = err.to_json().to_compact_string();
        assert!(!public_json.contains("do not echo private query text"));
        assert!(!public_json.contains("embedding"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_top_k_checks_original_decimal_math_and_target_range() {
        let dir = temp_dir("query-exact-top-k");
        let maximum = crate::query::MAX_SEARCH_TOP_K;
        let max_token = maximum.to_string();
        let accepted = vec![
            ("1".to_owned(), 1),
            ("10.0".to_owned(), 10),
            ("1e1".to_owned(), 10),
            ("0.1e2".to_owned(), 10),
            ("1000e-2".to_owned(), 10),
            ((maximum - 1).to_string(), maximum - 1),
            (max_token.clone(), maximum),
            (format!("{max_token}000e-3"), maximum),
        ];
        for (raw, expected) in accepted {
            let query = format!(r#"{{"query_id":"q1","text":"wal","top_k":{raw}}}"#);
            write_none_fixture(
                &dir,
                &query,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#,
            );
            let original = fs::read(dir.join("queries.jsonl")).unwrap();
            let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
            assert_eq!(dataset.queries[0].top_k, expected, "{raw}");
            assert_eq!(fs::read(dir.join("queries.jsonl")).unwrap(), original);
        }
        for raw in [
            "0",
            "-1",
            "1.5",
            "1.0000000000000001",
            "1.0000000000000000000000000000000001",
            "9007199254740991.1",
            "9007199254740992",
            "9007199254740993",
            "9.007199254740992e15",
            "1e9999999999999999999999999",
            "1e-9999999999999999999999999",
            "true",
            "[]",
            "[10]",
            "{}",
            "{\"top_k\":10}",
            "\"10\"",
        ] {
            let query = format!(r#"{{"query_id":"q1","text":"wal","top_k":{raw}}}"#);
            write_none_fixture(
                &dir,
                &query,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#,
            );
            let original = fs::read(dir.join("queries.jsonl")).unwrap();
            let error = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema, "{raw}");
            assert_eq!(detail_value(&error, "artifact_kind"), Some("queries"));
            assert_eq!(detail_value(&error, "line"), Some("1"));
            assert_eq!(fs::read(dir.join("queries.jsonl")).unwrap(), original);
        }
        for field in ["", ",\"top_k\":null"] {
            let query = format!(r#"{{"query_id":"q1","text":"wal"{field}}}"#);
            write_none_fixture(
                &dir,
                &query,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#,
            );
            let error = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
            assert_eq!(error.code(), ErrorCode::MissingRequiredField);
            assert_eq!(detail_value(&error, "field"), Some("queries.top_k"));
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn benchmark_queries_require_text_and_top_k() {
        let dir = temp_dir("query-missing-text");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(&dir, "queries.jsonl", r#"{"query_id":"q1","top_k":10}"#);
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "field"), Some("queries.text"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("query-missing-top-k");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(&dir, "queries.jsonl", r#"{"query_id":"q1","text":"wal"}"#);
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "field"), Some("queries.top_k"));
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_artifacts_reject_blank_identifiers_and_models() {
        let dir = temp_dir("blank-query-id");
        write_none_fixture(
            &dir,
            r#"{"query_id":"","text":"wal","top_k":10}"#,
            r#"{"query_id":"","chunk_external_id":"doc#1","relevance":2}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("queries.query_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("blank-document-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("documents.external_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("blank-embedding-model");
        write_sidecar_fixture_with_chunks_and_embeddings(
            &dir,
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
            r#"{"chunk_external_id":"doc#1","model_name":"","dim":2,"vector":[1.0,0.0]}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("embeddings.model_name"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unknown_artifact_keys_are_not_reflected_as_public_schema_locations() {
        for seed in 0..100 {
            let secret = format!("private query or metadata value {seed}");
            let obj = [(secret.clone(), JsonValue::Null)]
                .into_iter()
                .collect::<JsonObject>();
            for error in [
                reject_unknown_benchmark_manifest_fields(&obj, "dataset.json").unwrap_err(),
                reject_unknown_benchmark_hash_fields(&obj, "dataset.json").unwrap_err(),
            ] {
                assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                assert!(!error.to_string().contains(&secret));
            }
            for kind in [
                "documents",
                "chunks",
                "embeddings",
                "queries",
                "qrels",
                "query_embeddings",
            ] {
                let artifact = Artifact {
                    raw_path: format!("{kind}.jsonl"),
                    path: PathBuf::from("/private/unused-resolved-path"),
                    kind,
                    hash_field: "unused",
                };
                let error =
                    reject_unknown_benchmark_artifact_fields(&obj, &artifact, 1, "dataset.json")
                        .unwrap_err();
                assert_eq!(detail_value(&error, "field"), Some(kind));
                assert!(!error.to_string().contains(&secret));
                assert!(!error.to_string().contains("/private/unused-resolved-path"));
            }
        }
    }

    #[test]
    fn benchmark_manifest_rejects_unknown_public_fields() {
        let dir = temp_dir("unknown-benchmark-manifest-field");
        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{"documents_sha256":"x","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"},"document":"typo.jsonl"}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("manifest"));

        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","embedding_dim":2,"hashes":{"documents_sha256":"x","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"}}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("manifest"));

        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{"documents_hash":"typo","documents_sha256":"x","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"}}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("hashes"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_artifact_rows_reject_unknown_public_fields() {
        let valid_documents =
            r#"{"external_id":"doc","title":"Doc","source_uri":"docs/wal.md","metadata":{}}"#;
        let valid_chunks = r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{},"source_span":{"start":0,"end":3}}"#;
        let valid_embeddings =
            r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#;
        let valid_queries = r#"{"query_id":"q1","text":"wal","top_k":10}"#;
        let valid_qrels = r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#;
        let valid_query_embeddings =
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#;

        for (
            name,
            expected_field,
            documents_content,
            chunks_content,
            embeddings_content,
            queries_content,
            qrels_content,
            query_embeddings_content,
        ) in [
            (
                "unknown-doc-row-field",
                "documents",
                r#"{"external_id":"doc","metadata":{},"alias":"doc"}"#,
                valid_chunks,
                valid_embeddings,
                valid_queries,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-chunk-row-field",
                "chunks",
                valid_documents,
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{},"vector":[1.0]}"#,
                valid_embeddings,
                valid_queries,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-embedding-row-field",
                "embeddings",
                valid_documents,
                valid_chunks,
                r#"{"chunk_external_id":"doc#1","model_name":"m","dim":2,"vector":[1.0,0.0],"extra":true}"#,
                valid_queries,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-query-row-field",
                "queries",
                valid_documents,
                valid_chunks,
                valid_embeddings,
                r#"{"query_id":"q1","text":"wal","top_k":10,"category":"hybrid"}"#,
                valid_qrels,
                valid_query_embeddings,
            ),
            (
                "unknown-qrel-row-field",
                "qrels",
                valid_documents,
                valid_chunks,
                valid_embeddings,
                valid_queries,
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2,"note":"gold"}"#,
                valid_query_embeddings,
            ),
            (
                "unknown-query-embedding-row-field",
                "query_embeddings",
                valid_documents,
                valid_chunks,
                valid_embeddings,
                valid_queries,
                valid_qrels,
                r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0],"score":1.0}"#,
            ),
        ] {
            let dir = temp_dir(name);
            write_sidecar_fixture_with_artifacts(
                &dir,
                documents_content,
                chunks_content,
                embeddings_content,
                queries_content,
                qrels_content,
                query_embeddings_content,
            );

            let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
            assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(detail_value(&err, "field"), Some(expected_field));
            assert_eq!(detail_value(&err, "line"), Some("1"));
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn benchmark_manifest_rejects_blank_hash_values() {
        let dir = temp_dir("blank-benchmark-hash");
        fs::write(
            dir.join("dataset_manifest.json"),
            r#"{"schema_version":"0.1","dataset_id":"bench","version":"0.1","documents":"documents.jsonl","chunks":"chunks.jsonl","queries":"queries.jsonl","qrels":"qrels.jsonl","query_embedding_source":"none","hashes":{"documents_sha256":"","chunks_sha256":"x","queries_sha256":"x","qrels_sha256":"x"}}"#,
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("hashes.documents_sha256"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_manifest_rejects_blank_identity_fields() {
        let dir = temp_dir("blank-dataset-id");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        let documents = sha256_hex(&fs::read(dir.join("documents.jsonl")).unwrap());
        let chunks = sha256_hex(&fs::read(dir.join("chunks.jsonl")).unwrap());
        let queries = sha256_hex(&fs::read(dir.join("queries.jsonl")).unwrap());
        let qrels = sha256_hex(&fs::read(dir.join("qrels.jsonl")).unwrap());
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("dataset_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("blank-query-embedding-model");
        write_sidecar_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
            r#"{"query_id":"q1","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
        );
        let documents = sha256_hex(&fs::read(dir.join("documents.jsonl")).unwrap());
        let chunks = sha256_hex(&fs::read(dir.join("chunks.jsonl")).unwrap());
        let embeddings = sha256_hex(&fs::read(dir.join("embeddings.jsonl")).unwrap());
        let queries = sha256_hex(&fs::read(dir.join("queries.jsonl")).unwrap());
        let qrels = sha256_hex(&fs::read(dir.join("qrels.jsonl")).unwrap());
        let query_embeddings = sha256_hex(&fs::read(dir.join("query_embeddings.jsonl")).unwrap());
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""embeddings":"embeddings.jsonl","queries":"queries.jsonl","#,
                    r#""qrels":"qrels.jsonl","query_embedding_source":"sidecar","#,
                    r#""query_embeddings":"query_embeddings.jsonl","embedding_model":"m","#,
                    r#""query_embedding_model":"","query_embedding_dim":2,"hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""embeddings_sha256":"{}","queries_sha256":"{}","#,
                    r#""qrels_sha256":"{}","query_embeddings_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, embeddings, queries, qrels, query_embeddings
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("query_embedding_model"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_queries_reject_invalid_metadata_filter() {
        let dir = temp_dir("query-invalid-filter");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","filter":{"lang":"rust"},"top_k":10}"#,
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "field"), Some("queries.filter"));
        assert_eq!(
            detail_value(&err, "filter_error_code"),
            Some("invalid_metadata_filter")
        );
        assert_eq!(detail_value(&err, "artifact_kind"), Some("queries"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn qrels_must_reference_declared_queries() {
        let dir = temp_dir("qrels-query-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q-missing","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q-missing"));
        assert_eq!(detail_value(&err, "field"), Some("qrels.query_id"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_must_reference_declared_documents() {
        let dir = temp_dir("chunks-document-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"missing-doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(
            detail_value(&err, "document_external_id"),
            Some("missing-doc")
        );
        assert_eq!(
            detail_value(&err, "field"),
            Some("chunks.document_external_id")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_documents_reject_duplicate_external_ids() {
        let dir = temp_dir("duplicate-document-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            concat!(
                r#"{"external_id":"doc","metadata":{}}"#,
                "\n",
                r#"{"external_id":"doc","metadata":{}}"#
            ),
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
        assert_eq!(detail_value(&err, "external_id"), Some("doc"));
        assert_eq!(detail_value(&err, "line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_documents_require_metadata_object() {
        let dir = temp_dir("document-missing-metadata");
        let documents = write_file(&dir, "documents.jsonl", r#"{"external_id":"doc"}"#);
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("documents.metadata"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_require_ordinal_text_and_metadata() {
        let dir = temp_dir("chunk-missing-ordinal");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("chunks.ordinal"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("chunk-missing-text");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("chunks.text"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("chunk-missing-metadata");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal"}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&err, "field"), Some("chunks.metadata"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn qrels_must_reference_declared_chunks() {
        let dir = temp_dir("qrels-chunk-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#missing","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#missing"));
        assert_eq!(detail_value(&err, "field"), Some("qrels.chunk_external_id"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn zero_relevance_queries_are_judged_without_positive_relevant_chunks() {
        let dir = temp_dir("qrels-judged-zero");
        let queries = (0..100)
            .map(|id| format!("{{\"query_id\":\"q-{id}\",\"text\":\"wal\",\"top_k\":10}}\n"))
            .collect::<String>();
        let qrels = (0..100).rev().map(|id| format!("{{\"query_id\":\"q-{id}\",\"chunk_external_id\":\"doc#1\",\"relevance\":{}}}\n", id % 3)).collect::<String>();
        write_none_fixture(&dir, &queries, &qrels);
        let dataset = BenchmarkDataset::load_and_validate(&dir).unwrap();
        assert_eq!(dataset.queries.len(), 100);
        for (id, query) in dataset.queries.iter().enumerate() {
            assert_eq!(query.query_id, format!("q-{id}"));
            assert_eq!(query.relevant_chunk_ids.len(), usize::from(id % 3 != 0));
            if id % 3 != 0 {
                assert!(query.relevant_chunk_ids.contains("doc#1"));
            }
        }
        write_none_fixture(
            &dir,
            &queries,
            &qrels.lines().skip(1).collect::<Vec<_>>().join("\n"),
        );
        let error = BenchmarkDataset::load_and_validate(&dir).unwrap_err();
        assert_eq!(error.code(), ErrorCode::MissingRequiredField);
        assert_eq!(detail_value(&error, "query_id"), Some("q-99"));
        assert_eq!(detail_value(&error, "field"), Some("qrels[query_id]"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_relevance_qrels_still_validate_identity_and_duplicates() {
        let dir = temp_dir("qrels-zero-unknown-query");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            r#"{"query_id":"q-missing","chunk_external_id":"doc#1","relevance":0}"#,
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q-missing"));
        assert_eq!(detail_value(&err, "field"), Some("qrels.query_id"));
        let _ = fs::remove_dir_all(dir);

        let dir = temp_dir("qrels-zero-duplicate-pair");
        write_none_fixture(
            &dir,
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
            concat!(
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":0}"#,
                "\n",
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#
            ),
        );

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn qrels_reject_duplicate_query_chunk_pairs() {
        let dir = temp_dir("qrels-duplicate-pair");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            concat!(
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
                "\n",
                r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":1}"#
            ),
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "query_id"), Some("q1"));
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_reject_duplicate_external_ids_before_qrels() {
        let dir = temp_dir("duplicate-chunk-id");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":2,"text":"recovery","metadata":{}}"#
            ),
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
        assert_eq!(detail_value(&err, "chunk_external_id"), Some("doc#1"));
        assert_eq!(detail_value(&err, "line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn benchmark_chunks_reject_duplicate_ordinals_within_document() {
        let dir = temp_dir("duplicate-chunk-ordinal");
        let documents = write_file(
            &dir,
            "documents.jsonl",
            r#"{"external_id":"doc","metadata":{}}"#,
        );
        let chunks = write_file(
            &dir,
            "chunks.jsonl",
            concat!(
                r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":1,"text":"wal","metadata":{}}"#,
                "\n",
                r#"{"document_external_id":"doc","chunk_external_id":"doc#2","ordinal":1,"text":"recovery","metadata":{}}"#
            ),
        );
        let queries = write_file(
            &dir,
            "queries.jsonl",
            r#"{"query_id":"q1","text":"wal","top_k":10}"#,
        );
        let qrels = write_file(
            &dir,
            "qrels.jsonl",
            r#"{"query_id":"q1","chunk_external_id":"doc#1","relevance":2}"#,
        );
        fs::write(
            dir.join("dataset_manifest.json"),
            format!(
                concat!(
                    r#"{{"schema_version":"0.1","dataset_id":"bench","version":"0.1","#,
                    r#""documents":"documents.jsonl","chunks":"chunks.jsonl","#,
                    r#""queries":"queries.jsonl","qrels":"qrels.jsonl","#,
                    r#""query_embedding_source":"none","hashes":{{"#,
                    r#""documents_sha256":"{}","chunks_sha256":"{}","#,
                    r#""queries_sha256":"{}","qrels_sha256":"{}"}}"#,
                    r#"}}"#
                ),
                documents, chunks, queries, qrels
            ),
        )
        .unwrap();

        let err = BenchmarkDatasetManifest::load_and_validate(&dir).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(detail_value(&err, "field"), Some("chunks.ordinal"));
        assert_eq!(detail_value(&err, "document_external_id"), Some("doc"));
        assert_eq!(detail_value(&err, "ordinal"), Some("1"));
        assert_eq!(detail_value(&err, "first_line"), Some("1"));
        assert_eq!(detail_value(&err, "duplicate_line"), Some("2"));
        let _ = fs::remove_dir_all(dir);
    }
}
