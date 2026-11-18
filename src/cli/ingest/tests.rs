use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::cli::{error_output, run};
use crate::db::Database;
use crate::error::{Error, ErrorCode};
use crate::hash::sha256_hex;
use crate::json::{parse, JsonValue};

use super::http::build_endpoint;
use super::http::{embed_batch, BatchRequest, REQUEST_TIMEOUT};

fn temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "cofferdb-ingest-{name}-{}-{}",
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

fn run_at(args: &[&str]) -> Result<String, Error> {
    let mut full = vec!["coffer".to_owned()];
    full.extend(args.iter().map(|arg| (*arg).to_owned()));
    run(full)
}

fn detail<'a>(error: &'a Error, key: &str) -> Option<&'a str> {
    error
        .detail()
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

fn envelope_error(command: &str, error: &Error) -> JsonValue {
    parse(&error_output(Some(command), error)).unwrap()
}

fn set_env(key: &str, value: &str) {
    // SAFETY: callers hold ENV_LOCK around process-global environment edits.
    unsafe { std::env::set_var(key, value) }
}

fn unset_env(key: &str) {
    unsafe { std::env::remove_var(key) }
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

struct Server {
    base: String,
    hits: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

struct Recorded {
    headers: String,
    body: String,
}

impl Server {
    fn spawn(handler: impl Fn(&Recorded) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let hits = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let hits_thread = Arc::clone(&hits);
        let stop_thread = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let handler = Arc::new(handler);
            while !stop_thread.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // The listener polls in nonblocking mode. Read each accepted
                        // request in blocking mode, even if its body arrives later.
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        let Some((headers, body)) = read_request(&mut stream) else {
                            continue;
                        };
                        let recorded = Recorded {
                            headers: headers.clone(),
                            body: String::from_utf8_lossy(&body).into_owned(),
                        };
                        let reply = handler(&recorded);
                        if !reply.delay.is_zero() {
                            thread::sleep(reply.delay);
                        }
                        hits_thread.lock().unwrap().push(recorded);
                        let _ = write_reply(&mut stream, &reply);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            base: format!("http://127.0.0.1:{port}/v1"),
            hits,
            stop,
            handle: Some(handle),
        }
    }

    fn hit_count(&self) -> usize {
        self.hits.lock().unwrap().len()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct Reply {
    status: u16,
    location: Option<String>,
    body: String,
    chunked: bool,
    delay: Duration,
}

impl Reply {
    fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            location: None,
            body: body.into(),
            chunked: false,
            delay: Duration::ZERO,
        }
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<(String, Vec<u8>)> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    let header_end = loop {
        let count = stream.read(&mut buf).ok()?;
        if count == 0 {
            return None;
        }
        raw.extend_from_slice(&buf[..count]);
        if let Some(index) = raw.windows(4).position(|window| window == b"\r\n\r\n") {
            break index;
        }
    };
    let headers = String::from_utf8_lossy(&raw[..header_end]).into_owned();
    let mut body = raw[header_end + 4..].to_vec();
    let length = content_length(&headers).unwrap_or(0);
    while body.len() < length {
        let count = stream.read(&mut buf).ok()?;
        if count == 0 {
            break;
        }
        body.extend_from_slice(&buf[..count]);
    }
    body.truncate(length);
    Some((headers, body))
}

fn content_length(headers: &str) -> Option<usize> {
    headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse().ok())
            .flatten()
    })
}

fn write_reply(stream: &mut std::net::TcpStream, reply: &Reply) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {} TEST\r\nConnection: close\r\n", reply.status);
    if let Some(location) = &reply.location {
        head.push_str("Location: ");
        head.push_str(location);
        head.push_str("\r\n");
    }
    if reply.chunked {
        head.push_str("Transfer-Encoding: chunked\r\n\r\n");
        stream.write_all(head.as_bytes())?;
        stream.write_all(format!("{:x}\r\n", reply.body.len()).as_bytes())?;
        stream.write_all(reply.body.as_bytes())?;
        stream.write_all(b"\r\n0\r\n\r\n")?;
    } else {
        head.push_str(&format!("Content-Length: {}\r\n\r\n", reply.body.len()));
        stream.write_all(head.as_bytes())?;
        stream.write_all(reply.body.as_bytes())?;
    }
    Ok(())
}

fn embeddings_json(model: &str, inputs: &[String]) -> String {
    let mut body = format!("{{\"model\":\"{model}\",\"data\":[");
    for (index, input) in inputs.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        let vector = match input.as_str() {
            "two" => "[0,1]",
            "three" => "[1,1]",
            _ => "[1,0]",
        };
        body.push_str(&format!("{{\"index\":{index},\"embedding\":{vector}}}"));
    }
    body.push_str("]}");
    body
}

fn parsed_inputs(body: &str) -> Vec<String> {
    let value = parse(body).unwrap();
    value
        .get("input")
        .and_then(JsonValue::as_array)
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

fn echo_embeddings(recorded: &Recorded) -> Reply {
    let inputs = parsed_inputs(&recorded.body);
    let model = parse(&recorded.body)
        .unwrap()
        .get("model")
        .and_then(JsonValue::as_str)
        .unwrap()
        .to_owned();
    Reply::json(200, embeddings_json(&model, &inputs))
}

#[test]
fn text_artifacts_are_stable_and_do_not_touch_the_database_or_key() {
    let _env = lock_env();
    set_env("OPENAI_API_KEY", "text-only-sentinel");
    set_env("OPENAI_BASE_URL", "http://127.0.0.1:9/v1");
    let root = temp_dir("text");
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("note.txt"), "a\nb").unwrap();
    fs::write(input.join("blank.txt"), "   ").unwrap();
    let bom = b"\xEF\xBB\xBFhi";
    fs::write(input.join("bom.txt"), bom).unwrap();
    let db = root.join("missing.coffer");
    let existing = root.join("existing.coffer");
    fs::write(&existing, b"ORIGINAL").unwrap();
    let out = root.join("out");
    let first = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "10",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        out.to_str().unwrap(),
    ])
    .unwrap();
    assert!(!db.exists());
    assert_eq!(fs::read(&existing).unwrap(), b"ORIGINAL");
    assert!(!out.join("embeddings.jsonl").exists());
    assert!(!out.join("embedding_artifact.json").exists());
    let chunks = fs::read_to_string(out.join("chunks.jsonl")).unwrap();
    assert!(chunks.contains("\"text\":\"a\\u000ab\""));
    assert!(!chunks.contains("\"text\":\"   \""));
    assert!(chunks.contains("\"text\":\"hi\""));
    assert!(chunks.contains("\"line_start\":1"));
    assert!(chunks.contains("\"line_end\":2"));
    assert!(!chunks.contains("text-only-sentinel"));
    let manifest = fs::read_to_string(out.join("import_manifest.json")).unwrap();
    assert!(manifest.starts_with("{\"schema_version\":\"0.1\",\"dataset_id\":"));
    assert!(manifest.contains("\"collection\":\"notes\""));
    assert!(!manifest.contains("embeddings"));
    assert!(!manifest.contains("distance_metric"));
    let preimage = "format=text\nchunk_size=10\nchunk_overlap=1\nbom.txt\nnote.txt\n";
    assert!(manifest.contains(&sha256_hex(preimage.as_bytes())));
    let second = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        existing.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "10",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        out.to_str().unwrap(),
    ])
    .unwrap();
    assert_eq!(
        fs::read(out.join("chunks.jsonl")).unwrap(),
        chunks.as_bytes()
    );
    assert_eq!(fs::read(&existing).unwrap(), b"ORIGINAL");
    assert_eq!(
        parse(&first)
            .unwrap()
            .get("result")
            .unwrap()
            .get("embeddings"),
        Some(&JsonValue::Number(0.0))
    );
    assert!(
        parse(&second)
            .unwrap()
            .get("ok")
            .and_then(JsonValue::as_bool)
            == Some(true)
    );
    let replay = root.join("replay.coffer");
    run_at(&["init", replay.to_str().unwrap()]).unwrap();
    run_at(&[
        "create-collection",
        replay.to_str().unwrap(),
        "notes",
        "--dim",
        "2",
        "--metric",
        "cosine",
    ])
    .unwrap();
    let before_emit = fs::read(&replay).unwrap();
    run_at(&[
        "import",
        replay.to_str().unwrap(),
        "--manifest",
        out.join("import_manifest.json").to_str().unwrap(),
    ])
    .unwrap();
    assert_ne!(fs::read(&replay).unwrap(), before_emit);
    let found = run_at(&[
        "search",
        replay.to_str().unwrap(),
        "hi",
        "--collection",
        "notes",
    ])
    .unwrap();
    assert!(found.contains("file://bom.txt#chunk-0001"));
    unset_env("OPENAI_API_KEY");
    unset_env("OPENAI_BASE_URL");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn filters_skip_without_reading_hidden_directories_or_link_targets() {
    let root = temp_dir("filter");
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("keep.txt"), "visible").unwrap();
    fs::write(input.join("pic.png"), b"png").unwrap();
    fs::write(input.join("bad.txt"), [0xff, 0xfe]).unwrap();
    fs::write(input.join("raw.txt"), [0xff, 0xfe, b'x']).unwrap();
    fs::create_dir(input.join(".secret")).unwrap();
    fs::write(input.join(".secret").join("hidden.txt"), "UNIQUE_HIDDEN").unwrap();
    fs::create_dir(input.join("target")).unwrap();
    fs::write(input.join("target").join("built.txt"), "UNIQUE_BUILD").unwrap();
    std::os::unix::fs::symlink("keep.txt", input.join("link.txt")).unwrap();
    let out = root.join("out");
    let output = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        root.join("missing.coffer").to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--include",
        "*.txt",
        "--exclude",
        "bad.txt",
        "--emit-jsonl",
        out.to_str().unwrap(),
    ])
    .unwrap();
    let chunks = fs::read_to_string(out.join("chunks.jsonl")).unwrap();
    assert!(chunks.contains("visible"));
    assert!(!chunks.contains("UNIQUE_HIDDEN"));
    assert!(!chunks.contains("UNIQUE_BUILD"));
    assert!(!chunks.contains("png"));
    let parsed = parse(&output).unwrap();
    let skipped = parsed
        .get("result")
        .unwrap()
        .get("skipped_files")
        .unwrap()
        .as_array()
        .unwrap()
        .to_vec();
    let reason = |path: &str| {
        skipped.iter().find_map(|item| {
            (item.get("path").and_then(JsonValue::as_str) == Some(path)).then(|| {
                item.get("reason")
                    .and_then(JsonValue::as_str)
                    .unwrap()
                    .to_owned()
            })
        })
    };
    assert_eq!(reason("pic.png").as_deref(), Some("unsupported_file_type"));
    assert_eq!(reason("link.txt").as_deref(), Some("symlink"));
    assert_eq!(reason("bad.txt").as_deref(), Some("excluded_by_filter"));
    assert_eq!(reason("raw.txt").as_deref(), Some("non_utf8_text"));
    assert!(reason(".secret/hidden.txt").is_none());
    let empty = root.join("empty");
    fs::create_dir(&empty).unwrap();
    let err = run_at(&[
        "import-dir",
        empty.to_str().unwrap(),
        root.join("missing.coffer").to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        root.join("empty-out").to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(detail(&err, "reason"), Some("no_eligible_inputs"));
    assert!(!root.join("empty-out").join("import_manifest.json").exists());
    let link = root.join("linked");
    std::os::unix::fs::symlink(&input, &link).unwrap();
    let err = run_at(&[
        "import-dir",
        link.to_str().unwrap(),
        root.join("missing.coffer").to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        root.join("link-out").to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(detail(&err, "reason"), Some("input_is_symlink"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn missing_emit_jsonl_fails_before_walking_or_creating_a_database() {
    let root = temp_dir("no-emit");
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    std::os::unix::fs::symlink(&input, root.join("link")).unwrap();
    let db = root.join("db.coffer");
    let err = run_at(&[
        "import-dir",
        root.join("link").to_str().unwrap(),
        db.to_str().unwrap(),
        "--chunk-size",
        "8",
        "--chunk-overlap",
        "1",
    ])
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::CliUsage);
    assert_eq!(detail(&err, "reason"), Some("missing_emit_jsonl"));
    assert!(!db.exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn embed_jsonl_writes_sidecar_and_does_not_open_a_database() {
    let _env = lock_env();
    let root = temp_dir("embed");
    let server = Server::spawn(echo_embeddings);
    let chunks = root.join("chunks.jsonl");
    fs::write(
        &chunks,
        "{\"chunk_external_id\":\"c1\",\"text\":\"hello\",\"ignored\":true}\n",
    )
    .unwrap();
    let out = root.join("vectors.jsonl");
    fs::write(&out, b"OLD\n").unwrap();
    let db = root.join("notes.coffer");
    set_env("COFFER_INGEST_TEST_KEY", "sekret-value");
    let output = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        out.to_str().unwrap(),
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_TEST_KEY",
        "--batch-size",
        "8",
    ])
    .unwrap();
    unset_env("COFFER_INGEST_TEST_KEY");
    let hit = &server.hits.lock().unwrap()[0];
    assert!(hit.headers.contains("Authorization: Bearer sekret-value"));
    assert!(hit.body.contains("\"model\":\"example\""));
    assert!(hit.body.contains("\"dimensions\":2"));
    assert!(hit.body.contains("\"input\":[\"hello\"]"));
    assert!(!hit.body.contains("ignored"));
    let written = fs::read_to_string(&out).unwrap();
    assert!(!written.starts_with("OLD"));
    assert!(written.contains("\"chunk_external_id\":\"c1\""));
    assert!(written.contains("\"vector\":[1,0]"));
    let sidecar = fs::read_to_string(root.join("embedding_artifact.json")).unwrap();
    assert!(sidecar.contains("\"embedding_provider\":\"openai-compatible\""));
    assert!(sidecar.contains("\"embedding_input\":"));
    assert!(sidecar.contains("chunks.jsonl") || sidecar.contains(&chunks.display().to_string()));
    assert!(!sidecar.contains("sekret-value"));
    assert!(!root.join("import_manifest.json").exists());
    assert!(!db.exists());
    let parsed = parse(&output).unwrap();
    assert_eq!(
        parsed.get("command").and_then(JsonValue::as_str),
        Some("embed-jsonl")
    );
    let result = parsed.get("result").unwrap();
    assert_eq!(
        result.get("documents").and_then(JsonValue::as_f64),
        Some(0.0)
    );
    assert_eq!(result.get("chunks").and_then(JsonValue::as_f64), Some(1.0));
    assert_eq!(
        result.get("embeddings").and_then(JsonValue::as_f64),
        Some(1.0)
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn missing_or_blank_keys_make_no_connection_and_ignore_base_env() {
    let _env = lock_env();
    let root = temp_dir("keys");
    let listener = Server::spawn(|_| Reply::json(500, "nope"));
    let other = Server::spawn(echo_embeddings);
    let chunks = root.join("chunks.jsonl");
    fs::write(
        &chunks,
        "{\"chunk_external_id\":\"c1\",\"text\":\"hello\"}\n",
    )
    .unwrap();
    unset_env("OPENAI_API_KEY");
    set_env("OPENAI_BASE_URL", &listener.base);
    let missing = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        root.join("out.jsonl").to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(missing.code(), ErrorCode::MissingRequiredField);
    assert_eq!(detail(&missing, "field"), Some("OPENAI_API_KEY"));
    assert_eq!(listener.hit_count(), 0);
    set_env("OPENAI_API_KEY", "   ");
    let blank = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        root.join("out.jsonl").to_str().unwrap(),
        "--base-url",
        &listener.base,
    ])
    .unwrap_err();
    assert_eq!(blank.code(), ErrorCode::MissingRequiredField);
    assert_eq!(listener.hit_count(), 0);
    set_env("OPENAI_API_KEY", "should-not-be-used");
    let compatible = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        root.join("out.jsonl").to_str().unwrap(),
        "--base-url",
        &listener.base,
    ])
    .unwrap_err();
    assert_eq!(compatible.code(), ErrorCode::MissingRequiredField);
    assert_eq!(detail(&compatible, "field"), Some("--api-key-env"));
    assert_eq!(listener.hit_count(), 0);
    set_env("COFFER_INGEST_EXPLICIT", "explicit-key");
    run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        root.join("out.jsonl").to_str().unwrap(),
        "--base-url",
        &other.base,
        "--api-key-env",
        "COFFER_INGEST_EXPLICIT",
    ])
    .unwrap();
    assert_eq!(listener.hit_count(), 0);
    assert_eq!(other.hit_count(), 1);
    assert!(other.hits.lock().unwrap()[0]
        .headers
        .contains("Authorization: Bearer explicit-key"));
    let suffix = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        root.join("out.jsonl").to_str().unwrap(),
        "--base-url",
        "https://api.openai.com/v1/embeddings",
        "--api-key-env",
        "COFFER_INGEST_EXPLICIT",
    ])
    .unwrap_err();
    assert_eq!(suffix.code(), ErrorCode::CliUsage);
    assert_eq!(REQUEST_TIMEOUT, Duration::from_secs(60));
    let endpoint = build_endpoint(super::http::OPENAI_DEFAULT_BASE).unwrap();
    assert_eq!(endpoint.url, super::http::OPENAI_EMBEDDINGS_URL);
    unset_env("OPENAI_API_KEY");
    unset_env("OPENAI_BASE_URL");
    unset_env("COFFER_INGEST_EXPLICIT");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn provider_failures_do_not_replace_artifacts_or_create_a_database() {
    let _env = lock_env();
    let root = temp_dir("fail");
    let chunks = root.join("chunks.jsonl");
    fs::write(
        &chunks,
        "{\"chunk_external_id\":\"c1\",\"text\":\"one\"}\n{\"chunk_external_id\":\"c2\",\"text\":\"two\"}\n",
    )
    .unwrap();
    let out = root.join("vectors.jsonl");
    fs::write(&out, b"OLD\n").unwrap();
    let db = root.join("notes.coffer");
    set_env("COFFER_INGEST_FAIL_KEY", "super-secret-value");
    let redirect_target = Server::spawn(|_| Reply::json(200, "stolen"));
    let redirect_base = redirect_target.base.clone();
    let redirects = AtomicUsize::new(0);
    let server = Server::spawn(move |recorded| {
        let count = redirects.fetch_add(1, Ordering::Relaxed);
        if count == 0 {
            Reply {
                status: 302,
                location: Some(format!("{redirect_base}/steal?key=super-secret-value")),
                body: "super-secret-value".into(),
                chunked: false,
                delay: Duration::ZERO,
            }
        } else if recorded.body.contains("two") {
            Reply::json(500, "super-secret-value")
        } else {
            echo_embeddings(recorded)
        }
    });
    let redirected = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--batch-size",
        "1",
        "--out",
        out.to_str().unwrap(),
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_FAIL_KEY",
    ])
    .unwrap_err();
    assert_eq!(redirected.code(), ErrorCode::IoError);
    assert_eq!(detail(&redirected, "provider_stage"), Some("redirect"));
    let rendered = error_output(Some("embed-jsonl"), &redirected);
    assert!(!rendered.contains("super-secret-value"));
    assert!(!rendered.contains("steal?key"));
    assert_eq!(fs::read(&out).unwrap(), b"OLD\n");
    assert_eq!(redirect_target.hit_count(), 0);
    assert!(!db.exists());
    let bad_dim = Server::spawn(|_| {
        Reply::json(
            200,
            r#"{"model":"example","data":[{"index":0,"embedding":[1]}]}"#,
        )
    });
    let mismatch = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--batch-size",
        "2",
        "--out",
        out.to_str().unwrap(),
        "--base-url",
        &bad_dim.base,
        "--api-key-env",
        "COFFER_INGEST_FAIL_KEY",
    ])
    .unwrap_err();
    assert_eq!(mismatch.code(), ErrorCode::InvalidVectorLength);
    assert_eq!(fs::read(&out).unwrap(), b"OLD\n");
    let non_finite = Server::spawn(|_| {
        Reply::json(
            200,
            r#"{"data":[{"index":0,"embedding":[340282366920938463463374607431768211456,0]}]}"#,
        )
    });
    let finite = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--batch-size",
        "1",
        "--out",
        out.to_str().unwrap(),
        "--base-url",
        &non_finite.base,
        "--api-key-env",
        "COFFER_INGEST_FAIL_KEY",
    ])
    .unwrap_err();
    assert_eq!(finite.code(), ErrorCode::InvalidVectorLength);
    let bad_index =
        Server::spawn(|_| Reply::json(200, r#"{"data":[{"index":1.5,"embedding":[1,0]}]}"#));
    let indexed = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--batch-size",
        "1",
        "--out",
        out.to_str().unwrap(),
        "--base-url",
        &bad_index.base,
        "--api-key-env",
        "COFFER_INGEST_FAIL_KEY",
    ])
    .unwrap_err();
    assert_eq!(indexed.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(fs::read(&out).unwrap(), b"OLD\n");
    let batch = Server::spawn(|recorded| {
        if recorded.body.contains("two") {
            Reply::json(500, "super-secret-value")
        } else {
            echo_embeddings(recorded)
        }
    });
    let input = root.join("docs");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("a.txt"), "one").unwrap();
    fs::write(input.join("b.txt"), "two").unwrap();
    let import_err = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        root.join("generated").to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--batch-size",
        "1",
        "--base-url",
        &batch.base,
        "--api-key-env",
        "COFFER_INGEST_FAIL_KEY",
    ])
    .unwrap_err();
    assert_eq!(import_err.code(), ErrorCode::IoError);
    assert_eq!(detail(&import_err, "provider_stage"), Some("response"));
    assert!(!db.exists());
    assert!(!error_output(Some("import-dir"), &import_err).contains("super-secret-value"));
    assert!(!root.join("generated").join("import_manifest.json").exists());
    unset_env("COFFER_INGEST_FAIL_KEY");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn local_provider_and_bad_flags_do_not_connect() {
    let _env = lock_env();
    let root = temp_dir("flags");
    let server = Server::spawn(echo_embeddings);
    let chunks = root.join("chunks.jsonl");
    fs::write(
        &chunks,
        "{\"chunk_external_id\":\"c1\",\"text\":\"hello\"}\n",
    )
    .unwrap();
    set_env("COFFER_INGEST_FLAG_KEY", "flag-secret");
    for args in [
        vec![
            "embed-jsonl",
            chunks.to_str().unwrap(),
            "--provider",
            "local",
            "--model",
            "file",
            "--dimensions",
            "2",
            "--out",
            root.join("out.jsonl").to_str().unwrap(),
            "--base-url",
            server.base.as_str(),
            "--api-key-env",
            "COFFER_INGEST_FLAG_KEY",
        ],
        vec![
            "embed-jsonl",
            chunks.to_str().unwrap(),
            "--provider",
            "openai-compatible",
            "--model",
            "example",
            "--dimensions",
            "02",
            "--out",
            root.join("out.jsonl").to_str().unwrap(),
            "--base-url",
            server.base.as_str(),
            "--api-key-env",
            "COFFER_INGEST_FLAG_KEY",
        ],
    ] {
        let err = run_at(&args).unwrap_err();
        assert_eq!(err.code(), ErrorCode::CliUsage);
    }
    assert_eq!(server.hit_count(), 0);
    let dup = root.join("dup.jsonl");
    fs::write(
        &dup,
        "{\"chunk_external_id\":\"c1\",\"text\":\"a\"}\n{\"chunk_external_id\":\"c1\",\"text\":\"b\"}\n",
    )
    .unwrap();
    let err = run_at(&[
        "embed-jsonl",
        dup.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        root.join("out.jsonl").to_str().unwrap(),
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_FLAG_KEY",
    ])
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::DuplicateExternalId);
    let envelope = envelope_error("embed-jsonl", &err);
    assert_eq!(
        envelope
            .get("error")
            .unwrap()
            .get("recoverable")
            .and_then(JsonValue::as_bool),
        Some(true)
    );
    assert!(envelope
        .get("error")
        .unwrap()
        .get("suggested_action")
        .unwrap()
        .is_null());
    assert_eq!(server.hit_count(), 0);
    unset_env("COFFER_INGEST_FLAG_KEY");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn import_embed_round_trip_preserves_order_and_keeps_core_import_errors() {
    let _env = lock_env();
    let root = temp_dir("round");
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("a.txt"), "one").unwrap();
    fs::write(input.join("b.txt"), "two").unwrap();
    fs::write(input.join("c.txt"), "three").unwrap();
    let server = Server::spawn(|recorded| {
        let mut reply = echo_embeddings(recorded);
        if recorded.body.contains("\"one\"") {
            reply.delay = Duration::from_millis(80);
        }
        reply
    });
    let db = root.join("notes.coffer");
    run_at(&["init", db.to_str().unwrap()]).unwrap();
    run_at(&[
        "create-collection",
        db.to_str().unwrap(),
        "notes",
        "--dim",
        "2",
        "--metric",
        "cosine",
        "--embedding-model",
        "example",
    ])
    .unwrap();
    let before = fs::read(&db).unwrap();
    set_env("COFFER_INGEST_ROUND_KEY", "round-secret");
    let generated = root.join("generated");
    let output = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        generated.to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--batch-size",
        "1",
        "--concurrency",
        "3",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_ROUND_KEY",
    ])
    .unwrap();
    let embeddings = fs::read_to_string(generated.join("embeddings.jsonl")).unwrap();
    let a = embeddings.find("file://a.txt#chunk-0001").unwrap();
    let b = embeddings.find("file://b.txt#chunk-0001").unwrap();
    let c = embeddings.find("file://c.txt#chunk-0001").unwrap();
    assert!(a < b && b < c, "{embeddings}");
    assert!(embeddings[a..b].contains("[1,0]"), "{embeddings}");
    assert!(embeddings[b..c].contains("[0,1]"), "{embeddings}");
    assert!(embeddings[c..].contains("[1,1]"), "{embeddings}");
    assert!(!embeddings.contains("round-secret"));
    assert!(
        fs::read_to_string(generated.join("embedding_artifact.json"))
            .unwrap()
            .contains("\"embedding_input\":\"chunks.jsonl\"")
    );
    assert_eq!(
        parse(&output)
            .unwrap()
            .get("result")
            .unwrap()
            .get("embeddings")
            .and_then(JsonValue::as_f64),
        Some(3.0)
    );
    let query = root.join("query.json");
    fs::write(&query, r#"{"model_name":"example","dim":2,"vector":[1,0]}"#).unwrap();
    let found = run_at(&[
        "search",
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--embedding",
        query.to_str().unwrap(),
    ])
    .unwrap();
    assert!(found.contains("file://a.txt#chunk-0001"));
    let again = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        generated.to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--batch-size",
        "8",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_ROUND_KEY",
    ])
    .unwrap_err();
    assert_eq!(again.code(), ErrorCode::DuplicateExternalId);
    assert_eq!(detail(&again, "import_stage"), Some("core_import"));
    assert_eq!(detail(&again, "provider_stage"), None);
    let envelope = envelope_error("import-dir", &again);
    let action = envelope
        .get("error")
        .unwrap()
        .get("suggested_action")
        .and_then(JsonValue::as_str)
        .unwrap();
    assert!(action.contains("on-conflict"));
    let still = run_at(&[
        "search",
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--embedding",
        query.to_str().unwrap(),
    ])
    .unwrap();
    assert!(still.contains("file://a.txt#chunk-0001"));
    let wrong_dim = root.join("dim.coffer");
    run_at(&["init", wrong_dim.to_str().unwrap()]).unwrap();
    run_at(&[
        "create-collection",
        wrong_dim.to_str().unwrap(),
        "notes",
        "--dim",
        "3",
        "--metric",
        "dot_product",
    ])
    .unwrap();
    let dim_before = fs::read(&wrong_dim).unwrap();
    let old_embeddings = fs::read(generated.join("embeddings.jsonl")).unwrap();
    let dim_err = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        wrong_dim.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        generated.to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_ROUND_KEY",
    ])
    .unwrap_err();
    assert_eq!(dim_err.code(), ErrorCode::InvalidEmbeddingDimension);
    assert_eq!(fs::read(&wrong_dim).unwrap(), dim_before);
    assert_eq!(
        fs::read(generated.join("embeddings.jsonl")).unwrap(),
        old_embeddings
    );
    let envelope = envelope_error("import-dir", &dim_err);
    assert!(envelope
        .get("error")
        .unwrap()
        .get("suggested_action")
        .unwrap()
        .is_null());
    let missing = root.join("absent.coffer");
    let missing_err = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        missing.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        root.join("absent-out").to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_ROUND_KEY",
    ])
    .unwrap_err();
    assert_eq!(missing_err.code(), ErrorCode::CollectionNotFound);
    assert!(!missing.exists());
    let garbage = root.join("garbage.coffer");
    fs::write(&garbage, b"not-a-database").unwrap();
    let direct = match Database::open_read_only(&garbage) {
        Ok(_) => panic!("garbage file opened as a database"),
        Err(error) => error,
    };
    let via = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        garbage.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        root.join("garbage-out").to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_ROUND_KEY",
    ])
    .unwrap_err();
    assert_eq!(via.code(), direct.code());
    assert_eq!(via.message(), direct.message());
    let garbage_envelope = envelope_error("import-dir", &via);
    assert_eq!(
        garbage_envelope
            .get("error")
            .unwrap()
            .get("recoverable")
            .and_then(JsonValue::as_bool),
        Some(true)
    );
    assert!(garbage_envelope
        .get("error")
        .unwrap()
        .get("suggested_action")
        .unwrap()
        .is_null());
    assert_eq!(fs::read(&garbage).unwrap(), b"not-a-database");
    assert_ne!(via.code(), ErrorCode::CollectionNotFound);
    assert_ne!(fs::read(&db).unwrap(), before);
    unset_env("COFFER_INGEST_ROUND_KEY");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn chunked_provider_body_and_short_timeout_use_the_client_contract() {
    let server = Server::spawn(|recorded| {
        let inputs = parsed_inputs(&recorded.body);
        let mut reply = Reply::json(200, embeddings_json("example", &inputs));
        reply.chunked = true;
        reply
    });
    let endpoint = build_endpoint(&server.base).unwrap();
    let vectors = embed_batch(BatchRequest {
        endpoint: &endpoint,
        key: "chunk-key",
        model: "example",
        dimensions: 2,
        inputs: &["hello".to_owned()],
        timeout: Duration::from_secs(2),
    })
    .unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0]]);
    let silent = Server::spawn(|_| Reply {
        status: 200,
        location: None,
        body: String::new(),
        chunked: false,
        delay: Duration::from_secs(5),
    });
    let endpoint = build_endpoint(&silent.base).unwrap();
    let err = embed_batch(BatchRequest {
        endpoint: &endpoint,
        key: "chunk-key",
        model: "example",
        dimensions: 2,
        inputs: &["hello".to_owned()],
        timeout: Duration::from_millis(200),
    })
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::IoError);
    assert_eq!(detail(&err, "provider_stage"), Some("request"));
}

#[test]
fn markdown_format_writes_text_artifacts_without_a_database() {
    let _env = lock_env();
    set_env("OPENAI_API_KEY", "markdown-sentinel");
    set_env("OPENAI_BASE_URL", "http://127.0.0.1:9/v1");
    let root = temp_dir("markdown");
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("note.MD"), "alpha").unwrap();
    fs::write(input.join("guide.markdown"), "beta").unwrap();
    fs::write(input.join(".notes.md"), "hidden-note").unwrap();
    fs::write(input.join(".markdown"), "DOTMARK").unwrap();
    fs::write(input.join("readme.txt"), "UNIQUE_TEXT_BODY").unwrap();
    fs::write(input.join("empty.md"), "").unwrap();
    fs::write(input.join("bom.md"), "\u{feff}").unwrap();
    let db = root.join("missing.coffer");
    let args = [
        "import-dir",
        input.to_str().unwrap(),
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "64",
        "--chunk-overlap",
        "1",
        "--format",
        "markdown",
        "--emit-jsonl",
    ];
    let out = root.join("out");
    let mut first = args.to_vec();
    first.push(out.to_str().unwrap());
    let output = run_at(&first).unwrap();
    assert!(!db.exists());
    assert!(!out.join("embeddings.jsonl").exists());
    assert!(!out.join("embedding_artifact.json").exists());
    let documents = fs::read_to_string(out.join("documents.jsonl")).unwrap();
    let chunks = fs::read_to_string(out.join("chunks.jsonl")).unwrap();
    let manifest = fs::read_to_string(out.join("import_manifest.json")).unwrap();
    assert!(documents.contains("\"format\":\"markdown\""));
    assert!(documents.contains("file://.notes.md"));
    assert!(document_contains(&documents, "guide.markdown"));
    assert!(document_contains(&documents, "note.MD"));
    assert!(!document_contains(&documents, "readme.txt"));
    assert!(!document_contains(&documents, ".markdown"));
    assert!(chunks.contains("hidden-note"));
    assert!(chunks.contains("alpha"));
    assert!(chunks.contains("beta"));
    assert!(!chunks.contains("UNIQUE_TEXT_BODY"));
    assert!(!chunks.contains("DOTMARK"));
    assert!(!chunks.contains("markdown-sentinel"));
    let preimage =
        "format=markdown\nchunk_size=64\nchunk_overlap=1\n.notes.md\nguide.markdown\nnote.MD\n";
    assert!(manifest.contains(&format!(
        "\"dataset_id\":\"{}\"",
        sha256_hex(preimage.as_bytes())
    )));
    let parsed = parse(&output).unwrap();
    let result = parsed.get("result").unwrap();
    assert_eq!(
        result.get("documents").and_then(JsonValue::as_f64),
        Some(3.0)
    );
    assert_eq!(result.get("chunks").and_then(JsonValue::as_f64), Some(3.0));
    assert_eq!(
        result.get("embeddings").and_then(JsonValue::as_f64),
        Some(0.0)
    );
    let skipped = result.get("skipped_files").unwrap().as_array().unwrap();
    let reasons: Vec<_> = skipped
        .iter()
        .map(|item| {
            (
                item.get("path").and_then(JsonValue::as_str).unwrap(),
                item.get("reason").and_then(JsonValue::as_str).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        reasons,
        [
            (".markdown", "excluded_by_filter"),
            ("bom.md", "empty_text"),
            ("empty.md", "empty_text"),
            ("readme.txt", "excluded_by_filter"),
        ]
    );
    assert!(skipped.iter().all(|item| {
        item.get("detected_encoding").unwrap().is_null()
            && item.get("declared_encoding").unwrap().is_null()
            && item.get("action").and_then(JsonValue::as_str) == Some("skipped")
    }));
    let out_again = root.join("out-again");
    let mut second = args.to_vec();
    second.push(out_again.to_str().unwrap());
    run_at(&second).unwrap();
    for name in ["documents.jsonl", "chunks.jsonl", "import_manifest.json"] {
        assert_eq!(
            fs::read(out.join(name)).unwrap(),
            fs::read(out_again.join(name)).unwrap()
        );
    }
    unset_env("OPENAI_API_KEY");
    unset_env("OPENAI_BASE_URL");
    let _ = fs::remove_dir_all(&root);
}

fn document_contains(documents: &str, name: &str) -> bool {
    documents.contains(&format!("file://{name}"))
}

#[test]
fn codepoint_wider_than_chunk_size_fails_before_artifacts() {
    let root = temp_dir("wide-codepoint");
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("euro.txt"), "€").unwrap();
    let out = root.join("out");
    let err = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        root.join("missing.coffer").to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "2",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        out.to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::CliUsage);
    assert_eq!(detail(&err, "reason"), Some("codepoint_exceeds_chunk"));
    assert!(!out.join("import_manifest.json").exists());
    assert!(!root.join("missing.coffer").exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn model_mismatch_does_not_publish_or_backfill() {
    let _env = lock_env();
    set_env("COFFER_INGEST_MODEL_KEY", "model-secret-value");
    let root = temp_dir("model-mismatch");
    let wrong = Server::spawn(|recorded| {
        let inputs = parsed_inputs(&recorded.body);
        Reply::json(200, embeddings_json("other", &inputs))
    });
    let chunks = root.join("chunks.jsonl");
    fs::write(
        &chunks,
        "{\"chunk_external_id\":\"c1\",\"text\":\"hello\"}\n",
    )
    .unwrap();
    let embeddings = root.join("embeddings.jsonl");
    let sidecar = root.join("embedding_artifact.json");
    fs::write(&embeddings, b"OLD\n").unwrap();
    fs::write(&sidecar, b"OLD_SIDE\n").unwrap();
    let err = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        embeddings.to_str().unwrap(),
        "--base-url",
        &wrong.base,
        "--api-key-env",
        "COFFER_INGEST_MODEL_KEY",
    ])
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::EmbeddingModelMismatch);
    assert_eq!(fs::read(&embeddings).unwrap(), b"OLD\n");
    assert_eq!(fs::read(&sidecar).unwrap(), b"OLD_SIDE\n");
    let envelope = envelope_error("embed-jsonl", &err);
    assert!(envelope
        .get("error")
        .unwrap()
        .get("suggested_action")
        .unwrap()
        .is_null());
    assert_eq!(
        envelope
            .get("error")
            .unwrap()
            .get("recoverable")
            .and_then(JsonValue::as_bool),
        Some(true)
    );
    let rendered = error_output(Some("embed-jsonl"), &err);
    assert!(!rendered.contains("model-secret-value"));
    assert_eq!(wrong.hit_count(), 1);

    let server = Server::spawn(echo_embeddings);
    let locked = root.join("locked.coffer");
    run_at(&["init", locked.to_str().unwrap()]).unwrap();
    run_at(&[
        "create-collection",
        locked.to_str().unwrap(),
        "notes",
        "--dim",
        "2",
        "--metric",
        "cosine",
        "--embedding-model",
        "other",
    ])
    .unwrap();
    let before = fs::read(&locked).unwrap();
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("note.txt"), "hello").unwrap();
    let generated = root.join("generated");
    fs::create_dir(&generated).unwrap();
    fs::write(generated.join("import_manifest.json"), b"KEEP_MANIFEST\n").unwrap();
    fs::write(generated.join("embeddings.jsonl"), b"KEEP_EMBEDDINGS\n").unwrap();
    let err = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        locked.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "32",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        generated.to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_MODEL_KEY",
    ])
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::EmbeddingModelMismatch);
    assert_eq!(fs::read(&locked).unwrap(), before);
    assert_eq!(
        fs::read(generated.join("import_manifest.json")).unwrap(),
        b"KEEP_MANIFEST\n"
    );
    assert_eq!(
        fs::read(generated.join("embeddings.jsonl")).unwrap(),
        b"KEEP_EMBEDDINGS\n"
    );
    assert!(!generated.join("documents.jsonl").exists());
    let rendered = error_output(Some("import-dir"), &err);
    assert!(!rendered.contains("model-secret-value"));
    assert!(server.hit_count() >= 1);

    let open = root.join("open.coffer");
    run_at(&["init", open.to_str().unwrap()]).unwrap();
    run_at(&[
        "create-collection",
        open.to_str().unwrap(),
        "notes",
        "--dim",
        "2",
        "--metric",
        "cosine",
    ])
    .unwrap();
    let emitted = root.join("emitted");
    run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        open.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "32",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        emitted.to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_MODEL_KEY",
    ])
    .unwrap();
    let manifest = fs::read_to_string(emitted.join("import_manifest.json")).unwrap();
    assert!(!manifest.contains("embedding_model"));
    assert!(!manifest.contains("model-secret-value"));
    let found = run_at(&[
        "search",
        open.to_str().unwrap(),
        "hello",
        "--collection",
        "notes",
    ])
    .unwrap();
    assert!(found.contains("file://note.txt#chunk-0001"));
    unset_env("COFFER_INGEST_MODEL_KEY");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn whitespace_windows_are_omitted_and_the_remaining_text_imports() {
    let root = temp_dir("ws-window");
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("gap.txt"), "hello world\n\n").unwrap();
    let out = root.join("out");
    run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        root.join("missing.coffer").to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "12",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        out.to_str().unwrap(),
    ])
    .unwrap();
    let chunks = fs::read_to_string(out.join("chunks.jsonl")).unwrap();
    assert!(chunks.contains("\"text\":\"hello world\\u000a\""));
    assert!(!chunks.contains("\"text\":\"\\u000a\\u000a\""));
    assert!(chunks.contains("file://gap.txt#chunk-0001"));
    assert!(!chunks.contains("chunk-0002"));
    let db = root.join("notes.coffer");
    run_at(&["init", db.to_str().unwrap()]).unwrap();
    run_at(&[
        "create-collection",
        db.to_str().unwrap(),
        "notes",
        "--dim",
        "2",
        "--metric",
        "cosine",
    ])
    .unwrap();
    run_at(&[
        "import",
        db.to_str().unwrap(),
        "--manifest",
        out.join("import_manifest.json").to_str().unwrap(),
    ])
    .unwrap();
    let found = run_at(&[
        "search",
        db.to_str().unwrap(),
        "hello",
        "--collection",
        "notes",
    ])
    .unwrap();
    assert!(found.contains("file://gap.txt#chunk-0001"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn colliding_output_paths_fail_before_http_or_a_database() {
    let _env = lock_env();
    let root = temp_dir("collide");
    let server = Server::spawn(echo_embeddings);
    let chunks = root.join("chunks.jsonl");
    fs::write(
        &chunks,
        "{\"chunk_external_id\":\"c1\",\"text\":\"hello\"}\n",
    )
    .unwrap();
    set_env("COFFER_INGEST_TEST_KEY", "sekret-value");
    let sidecar_name = root.join("embedding_artifact.json");
    let err = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        sidecar_name.to_str().unwrap(),
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_TEST_KEY",
    ])
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::CliUsage);
    assert_eq!(detail(&err, "reason"), Some("output_path_collision"));
    assert_eq!(server.hit_count(), 0);
    assert!(!sidecar_name.exists());
    let dir = root.join("out");
    fs::create_dir(&dir).unwrap();
    let formal = dir.join("vectors.jsonl");
    let alias = dir.join("..").join("out").join("vectors.jsonl");
    let aliased = run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        formal.to_str().unwrap(),
        "--metadata-out",
        alias.to_str().unwrap(),
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_TEST_KEY",
    ])
    .unwrap_err();
    assert_eq!(detail(&aliased, "reason"), Some("output_path_collision"));
    assert_eq!(server.hit_count(), 0);
    let input = root.join("in");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("note.txt"), "hello").unwrap();
    let emit = root.join("emit");
    fs::create_dir(&emit).unwrap();
    let db = root.join("notes.coffer");
    let collided = run_at(&[
        "import-dir",
        input.to_str().unwrap(),
        db.to_str().unwrap(),
        "--collection",
        "notes",
        "--chunk-size",
        "20",
        "--chunk-overlap",
        "1",
        "--emit-jsonl",
        emit.to_str().unwrap(),
        "--embed",
        "openai-compatible:example",
        "--dimensions",
        "2",
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_TEST_KEY",
        "--metadata-out",
        emit.join("embeddings.jsonl").to_str().unwrap(),
    ])
    .unwrap_err();
    assert_eq!(detail(&collided, "reason"), Some("output_path_collision"));
    assert_eq!(server.hit_count(), 0);
    assert!(!db.exists());
    assert!(!emit.join("documents.jsonl").exists());
    unset_env("COFFER_INGEST_TEST_KEY");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn logical_path_treats_relative_and_absolute_aliases_as_one_file() {
    let cwd = std::env::current_dir().unwrap();
    let absolute = cwd.join("out").join("vectors.jsonl");
    let relative = std::path::Path::new("out")
        .join("..")
        .join("out")
        .join("vectors.jsonl");
    assert_eq!(
        super::logical_path(&absolute),
        super::logical_path(&relative)
    );
    assert_eq!(
        super::logical_path(std::path::Path::new("/tmp/embeddings.jsonl")),
        super::logical_path(std::path::Path::new("/../tmp/embeddings.jsonl"))
    );
    assert!(super::distinct_outputs(&[
        std::path::Path::new("/tmp/embeddings.jsonl"),
        std::path::Path::new("/../tmp/embeddings.jsonl"),
    ])
    .is_err());
}

#[test]
fn embed_jsonl_still_sends_whitespace_text() {
    let _env = lock_env();
    let root = temp_dir("embed-ws");
    let server = Server::spawn(echo_embeddings);
    let chunks = root.join("chunks.jsonl");
    fs::write(&chunks, "{\"chunk_external_id\":\"c1\",\"text\":\"   \"}\n").unwrap();
    set_env("COFFER_INGEST_TEST_KEY", "sekret-value");
    let out = root.join("vectors.jsonl");
    run_at(&[
        "embed-jsonl",
        chunks.to_str().unwrap(),
        "--provider",
        "openai-compatible",
        "--model",
        "example",
        "--dimensions",
        "2",
        "--out",
        out.to_str().unwrap(),
        "--base-url",
        &server.base,
        "--api-key-env",
        "COFFER_INGEST_TEST_KEY",
    ])
    .unwrap();
    assert!(server.hits.lock().unwrap()[0].body.contains("\"   \""));
    unset_env("COFFER_INGEST_TEST_KEY");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn self_check_rejects_a_whitespace_only_chunk() {
    let root = temp_dir("self-check");
    let documents = "{\"external_id\":\"file://a.txt\"}\n";
    let chunks = "{\"text\":\"   \"}\n";
    let documents_sha = super::sha256_hex(documents.as_bytes());
    let chunks_sha = super::sha256_hex(chunks.as_bytes());
    let manifest = format!(
        "{{\"schema_version\":\"0.1\",\"dataset_id\":\"abc\",\"collection\":\"notes\",\"documents\":\"documents.jsonl\",\"chunks\":\"chunks.jsonl\",\"hashes\":{{\"documents_sha256\":\"{documents_sha}\",\"chunks_sha256\":\"{chunks_sha}\"}}}}\n"
    );
    fs::write(root.join("documents.jsonl"), documents).unwrap();
    fs::write(root.join("chunks.jsonl"), chunks).unwrap();
    fs::write(root.join("import_manifest.json"), manifest).unwrap();
    let err = super::self_check_text(
        &root.join("documents.jsonl"),
        &root.join("chunks.jsonl"),
        &root.join("import_manifest.json"),
        "notes",
        &documents_sha,
        &chunks_sha,
    )
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
    assert_eq!(err.message(), "artifact self-check failed");
    let _ = fs::remove_dir_all(&root);
}
