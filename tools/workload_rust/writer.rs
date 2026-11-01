//! Workload driver support, not a stable Rust application API or frozen protocol.
//! The direct-batch baseline deliberately names the existing host-binding adapter.
use super::*;
use cofferdb::{
    binding::{Batch, ChunkRecord, Input, Operation, ValueKind, WriteOutcome},
    ChunkInput, CollectionConfig, DatabaseConfig, DocumentInput, EmbeddingInput, Error, ErrorCode,
    ImportManifest, OnConflict, SourceSpan, VectorIndexConfig,
};
use std::io::{BufRead, BufReader};

fn fixture(message: &str) -> Error {
    Error::new(ErrorCode::InvalidJsonlSchema, message)
}
fn required<'a>(value: &'a JsonValue, key: &str) -> cofferdb::Result<&'a JsonValue> {
    value
        .get(key)
        .ok_or_else(|| fixture(&format!("missing workload field {key}")))
}
fn required_text(value: &JsonValue, key: &str, input: &mut Input) -> cofferdb::Result<String> {
    input.text(
        required(value, key)?
            .as_str()
            .ok_or_else(|| fixture("workload string required"))?,
    )
}
fn optional_text(
    value: &JsonValue,
    key: &str,
    input: &mut Input,
) -> cofferdb::Result<Option<String>> {
    match value.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::String(value)) => input.text(value).map(Some),
        _ => Err(fixture("workload optional string required")),
    }
}
fn copy_json(
    value: &JsonValue,
    input: &mut Input,
    kind: ValueKind,
    depth: usize,
) -> cofferdb::Result<JsonValue> {
    Ok(match value {
        JsonValue::Null => JsonValue::Null,
        JsonValue::Bool(value) => JsonValue::Bool(*value),
        JsonValue::Number(value) => JsonValue::Number(*value),
        JsonValue::String(value) => JsonValue::String(input.json_string(value, kind)?),
        JsonValue::Array(items) => {
            input.check_container_depth(depth, kind)?;
            let mut values = input.array(items.len(), kind)?;
            for value in items {
                values.push(copy_json(value, input, kind, depth + 1)?);
            }
            JsonValue::Array(values)
        }
        JsonValue::Object(items) => {
            input.check_container_depth(depth, kind)?;
            let mut values = input.object(items.len(), kind)?;
            for (key, value) in items {
                values.insert(
                    input.json_string(key, kind)?,
                    copy_json(value, input, kind, depth + 1)?,
                );
            }
            JsonValue::Object(values)
        }
    })
}
fn metadata(value: &JsonValue, input: &mut Input, kind: ValueKind) -> cofferdb::Result<JsonValue> {
    let value = required(value, "metadata")?;
    let value = copy_json(value, input, kind, 0)?;
    input.validate_json(&value, kind)?;
    Ok(value)
}
fn prepare_batch(operation: &Operation, kind: &str, rows: &[JsonValue]) -> cofferdb::Result<Batch> {
    let mut batch = operation.batch()?;
    for value in rows {
        let mut input = operation.input();
        match kind {
            "documents" => {
                let document = DocumentInput {
                    external_id: required_text(value, "external_id", &mut input)?,
                    title: optional_text(value, "title", &mut input)?,
                    source_uri: optional_text(value, "source_uri", &mut input)?,
                    metadata: metadata(value, &mut input, ValueKind::DOCUMENT_METADATA)?,
                    chunks: Vec::new(),
                };
                batch.push_document(input.finish(document))?;
            }
            "chunks" => {
                let source_span = value.get("source_span");
                let span = match source_span {
                    None | Some(JsonValue::Null) => None,
                    Some(v) => Some(SourceSpan::from_json(v)?),
                };
                let record = ChunkRecord {
                    document_external_id: required_text(value, "document_external_id", &mut input)?,
                    chunk: ChunkInput {
                        chunk_external_id: required_text(value, "chunk_external_id", &mut input)?,
                        text: required_text(value, "text", &mut input)?,
                        ordinal: exact_index(required(value, "ordinal")?)?,
                        metadata: metadata(value, &mut input, ValueKind::CHUNK_METADATA)?,
                        source_span: span,
                    },
                    source_span_provided: source_span.is_some(),
                };
                batch.push_chunk(input.finish(record))?;
            }
            "embeddings" => {
                let vector = required(value, "vector")?
                    .as_array()
                    .ok_or_else(|| fixture("workload vector array required"))?;
                let mut owned = input.vector(vector.len(), ValueKind::JSONL_RECORD)?;
                for number in vector {
                    owned.push(
                        number
                            .as_f64()
                            .ok_or_else(|| fixture("numeric workload vector required"))?
                            as f32,
                    );
                }
                let embedding = EmbeddingInput {
                    chunk_external_id: required_text(value, "chunk_external_id", &mut input)?,
                    model_name: required_text(value, "model_name", &mut input)?,
                    dim: exact_index(required(value, "dim")?)?,
                    vector: owned,
                };
                batch.push_embedding(input.finish(embedding))?;
            }
            _ => return Err(fixture("unsupported workload record kind")),
        }
    }
    Ok(batch)
}
pub(super) fn progress_value(value: WriteOutcome) -> JsonValue {
    // Counts and u64 IDs stay exact decimal strings in tool evidence, even if
    // they exceed this engine JSON value's exact f64 range. This is not SDK JSON.
    obj([
        ("state", string(value.state)),
        (
            "confirmed_committed_batches",
            value
                .confirmed_committed_batches
                .map_or(JsonValue::Null, count),
        ),
        (
            "confirmed_committed_records",
            value
                .confirmed_committed_records
                .map_or(JsonValue::Null, count),
        ),
        (
            "last_confirmed_commit_id",
            value
                .last_confirmed_commit_id
                .map_or(JsonValue::Null, |n| string(n.to_string())),
        ),
        (
            "last_confirmed_commit_sequence_no",
            value
                .last_confirmed_commit_sequence_no
                .map_or(JsonValue::Null, |n| string(n.to_string())),
        ),
        (
            "tool_integer_representation",
            string("exact_decimal_strings"),
        ),
    ])
}
fn direct(
    database: &Database,
    collection: &CollectionHandle,
    folder: &Path,
    size: usize,
    journal: &mut Journal,
) -> Result<()> {
    for kind in ["documents", "chunks", "embeddings"] {
        let mut lines = BufReader::new(File::open(folder.join(format!("{kind}.jsonl")))?).lines();
        let mut index = 0usize;
        loop {
            let mut rows = Vec::new();
            for _ in 0..size {
                match lines.next() {
                    Some(line) => rows.push(cofferdb::parse_json(&line?)?),
                    None => break,
                }
            }
            if rows.is_empty() {
                break;
            }
            let operation = Operation::new(database);
            let observation = operation.observation();
            journal.measure_observed(
                &format!("batch-{kind}-{index}"),
                || {
                    let batch = prepare_batch(&operation, kind, &rows)?;
                    operation.write_batch(
                        database,
                        collection.collection_id(),
                        batch,
                        OnConflict::Error,
                    )
                },
                |summary| {
                    let mut value = summary.counts().to_json();
                    value.as_object_mut().expect("summary object").insert(
                        "warnings".into(),
                        JsonValue::Array(
                            summary
                                .warnings()
                                .iter()
                                .map(|value| {
                                    obj([
                                        ("code", string(&value.code)),
                                        ("field", string(&value.field)),
                                        ("artifact_kind", string(&value.artifact_kind)),
                                        ("message", string(&value.message)),
                                    ])
                                })
                                .collect(),
                        ),
                    );
                    value
                },
                Some(&observation),
            )?;
            index += 1;
        }
    }
    Ok(())
}
pub(super) fn run(arguments: &[String]) -> Result<()> {
    if arguments.len() != 7 {
        return Err(invalid(
            "expected write database folder queries output profile mode batch_rows",
        )
        .into());
    }
    let profile =
        ResourceProfile::parse(&arguments[4]).ok_or_else(|| invalid("unsupported profile"))?;
    let size: usize = arguments[6].parse()?;
    if size == 0 {
        return Err(invalid("positive batch size required").into());
    }
    let mode = arguments[5].as_str();
    if !["manifest", "batch"].contains(&mode) {
        return Err(invalid("unsupported write mode").into());
    }
    let folder = Path::new(&arguments[1]);
    let manifest_path = folder.join("import_manifest.json");
    let manifest = ImportManifest::load(&manifest_path)?;
    if manifest.documents != folder.join("documents.jsonl")
        || manifest.chunks != folder.join("chunks.jsonl")
        || manifest.embeddings.as_deref() != Some(folder.join("embeddings.jsonl").as_path())
    {
        return Err(invalid(
            "workload manifest must select the inspected three local JSONL inputs",
        )
        .into());
    }
    let queries = cofferdb::parse_json(&fs::read_to_string(&arguments[2])?)?;
    let selected: Vec<_> = queries
        .as_array()
        .ok_or_else(|| invalid("query array required"))?
        .iter()
        .filter(|query| query.get("profile").and_then(JsonValue::as_str) == Some(&arguments[4]))
        .collect();
    if selected.is_empty() {
        return Err(invalid("no ready query for selected profile").into());
    }
    let mut journal=Journal::new(Path::new(&arguments[3]),obj([
        ("kind",string("DECLARED_RUST_WORKLOAD_DRIVER")),("engine_version",string(cofferdb::binding::engine_version())),
        ("declared_build_source_input_sha256",option_env!("COFFERDB_CORE_SOURCE_INPUT_SHA256").map_or(JsonValue::Null,string)),
        ("profile",string(&arguments[4])),("mode",string(mode)),
        ("functional_control_only",JsonValue::Bool(functional_control())),
        ("api",string("Rust host-binding Operation adapter; not stable idiomatic application API")),
        ("timing_scope",string("bounded predecoded rows; adapter argument conversion, engine call and owned typed summary inside interval; Operation setup and JSON reporting/observation outside; ready wall includes creation/import/readiness/reporting, not initial fixture/manifest/query preparation")),
        ("cache_scope",string("OS page cache uncontrolled; no cold-cache certification"))]))?;
    let ready = Instant::now();
    let database = Database::create(
        &arguments[0],
        DatabaseConfig {
            resource_profile: profile,
            ..Default::default()
        },
    )?;
    let body = (|| {
        if mode == "manifest" {
            let operation = Operation::new(&database);
            let observation = operation.observation();
            journal.measure_observed(
                "import-manifest",
                || operation.import_jsonl(&database, &manifest_path, OnConflict::Error),
                |summary| summary.to_json(),
                Some(&observation),
            )?;
        } else {
            let config = CollectionConfig {
                name: manifest.collection.clone(),
                tokenizer: manifest.tokenizer.clone(),
                embedding_dim: manifest
                    .embedding_dim
                    .ok_or_else(|| invalid("explicit fixture dimension required"))?,
                distance_metric: manifest
                    .distance_metric
                    .ok_or_else(|| invalid("explicit fixture metric required"))?,
                embedding_model: manifest.embedding_model.clone(),
                vectors_normalized: manifest
                    .vectors_normalized
                    .ok_or_else(|| invalid("explicit fixture normalization required"))?,
                vector_index: VectorIndexConfig::Exact,
            };
            let collection = database.create_collection(config)?;
            direct(&database, &collection, folder, size, &mut journal)?;
        }
        let collection = database.collection(&manifest.collection)?;
        for request in selected {
            journal.measure(
                &format!("import-ready-{}", text(request, "id")?),
                || search(&collection, request, profile),
                |result| result.to_json(),
            )?;
        }
        journal.append(obj([
            ("event", string("import_ready")),
            ("wall_ns", string(elapsed_ns(ready).to_string())),
            ("includes_observation_and_result_io", JsonValue::Bool(true)),
            ("observation", observe()),
        ]))?;
        journal.measure("import-stats", || database.stats(), |stats| stats.to_json())?;
        Ok(())
    })();
    let cleanup = database
        .close()
        .map_err(|error| Box::new(error) as Box<dyn std::error::Error>);
    finish(body, cleanup)?;
    journal.append(obj([
        ("event", string("case_complete")),
        ("observation", observe()),
    ]))?;
    Ok(())
}

/// A single bounded owned-read control. No latency or memory value is measured.
pub(super) fn content(arguments: &[String]) -> Result<()> {
    if arguments.len() != 3 {
        return Err(invalid("expected content database collection external_id").into());
    }
    let database = Database::open_read_only(&arguments[0])?;
    let body = (|| {
        let collection = database.collection(&arguments[1])?;
        collection.get_chunk_content_by_external_id(&arguments[2])
    })();
    let cleanup = database
        .close()
        .map_err(|error| Box::new(error) as Box<dyn std::error::Error>);
    match body {
        Ok(content) => {
            if let Err(tool_error) = cleanup {
                return Err(Box::new(EvidenceFailure {
                    call_error: None,
                    known_kind: "complete_json_value",
                    known: content.to_json(),
                    tool_error,
                }));
            }
            println!("{}", content.to_json().try_to_compact_string()?);
            Ok(())
        }
        Err(error) => finish(Err(Box::new(error)), cleanup),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn query_integers_reject_coercion_and_keep_exact_boundary() {
        for value in [
            JsonValue::Bool(true),
            JsonValue::Null,
            string("10"),
            JsonValue::Number(1.5),
            JsonValue::Number(-1.0),
            JsonValue::Number(9_007_199_254_740_992.0),
        ] {
            assert!(exact_index(&value).is_err());
        }
        assert_eq!(
            exact_index(&JsonValue::Number(9_007_199_254_740_991.0)).unwrap(),
            9_007_199_254_740_991
        );
    }
    #[test]
    fn invalid_vector_preflight_has_no_invented_commit_progress() {
        let operation = Operation::standalone();
        let observation = operation.observation();
        let row = cofferdb::parse_json(
            r#"{"chunk_external_id":"x","model_name":"m","dim":2,"vector":[1.0,true]}"#,
        )
        .unwrap();
        assert!(prepare_batch(&operation, "embeddings", &[row]).is_err());
        let value = progress_value(observation.write_outcome());
        assert_eq!(value.get("state").unwrap().as_str(), Some("not_started"));
        assert_eq!(
            value.get("confirmed_committed_records"),
            Some(&JsonValue::Null)
        );
    }
    #[test]
    fn admitted_batch_metadata_and_text_keep_explicit_source_fields() {
        let operation = Operation::standalone();
        let row=cofferdb::parse_json(r#"{"external_id":"row","title":"Jade 中","source_uri":"synthetic://row","metadata":{"number":7,"nested":[true,null,"中"]}}"#).unwrap();
        let batch = prepare_batch(&operation, "documents", &[row]).unwrap();
        drop(batch);
        assert_eq!(operation.observation().write_outcome().state, "not_started");
    }
}
