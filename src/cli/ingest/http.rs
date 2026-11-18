//! OpenAI embeddings HTTP client.
//!
//! `build_endpoint` is the only URL constructor. `OPENAI_BASE_URL` is never
//! read. The API key is not placed on a process argument list. Redirects are
//! not followed. There are no retries.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::{Error, ErrorCode, Result};
use crate::json::{parse, JsonValue};

use super::boundary::{cli, io_ingest, schema, vector_length, with_host};
use super::float::format_f32;
use super::text::push_string;

pub(super) const OPENAI_DEFAULT_BASE: &str = "https://api.openai.com/v1";
#[cfg(test)]
pub(super) const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";
pub(super) const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Endpoint {
    pub url: String,
    pub host: String,
    pub connect: String,
    pub path: String,
    pub https: bool,
}

pub(super) fn build_endpoint(base: &str) -> Result<Endpoint> {
    if base.is_empty()
        || base.contains('?')
        || base.contains('#')
        || base.contains('@')
        || base.contains(' ')
    {
        return Err(cli(
            "base url must be an http or https url without userinfo, query, or fragment",
            Some("--base-url"),
            Some("invalid_base_url"),
        ));
    }
    let trimmed = base.trim_end_matches('/');
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err(invalid_base());
    };
    if scheme != "http" && scheme != "https" {
        return Err(invalid_base());
    }
    if rest.is_empty() || rest.starts_with('/') {
        return Err(invalid_base());
    }
    let (host_port, path) = split_host_path(rest)?;
    if path.ends_with("/embeddings") {
        return Err(cli(
            "base url must not already end in /embeddings",
            Some("--base-url"),
            Some("embeddings_suffix"),
        ));
    }
    let (name, explicit_port) = split_port(&host_port)?;
    let default_port = if scheme == "http" { 80 } else { 443 };
    let port = explicit_port.unwrap_or(default_port);
    let mut url = String::new();
    url.push_str(scheme);
    url.push_str("://");
    url.push_str(&name);
    let mut host = name.clone();
    let mut connect = name.clone();
    if port != default_port {
        url.push(':');
        url.push_str(&port.to_string());
        host.push(':');
        host.push_str(&port.to_string());
    }
    connect.push(':');
    connect.push_str(&port.to_string());
    if path.is_empty() {
        url.push_str("/embeddings");
    } else {
        if !path.starts_with('/') {
            return Err(invalid_base());
        }
        url.push_str(path);
        url.push_str("/embeddings");
    }
    let request_path = if path.is_empty() {
        "/embeddings".to_owned()
    } else {
        format!("{path}/embeddings")
    };
    Ok(Endpoint {
        url,
        host,
        connect,
        path: request_path,
        https: scheme == "https",
    })
}

fn invalid_base() -> Error {
    cli(
        "base url must be an http or https url without userinfo, query, or fragment",
        Some("--base-url"),
        Some("invalid_base_url"),
    )
}

fn split_host_path(rest: &str) -> Result<(&str, &str)> {
    if let Some(body) = rest.strip_prefix('[') {
        let Some(end) = body.find(']') else {
            return Err(invalid_base());
        };
        let host_end = end + 2;
        if host_end > rest.len() {
            return Err(invalid_base());
        }
        if rest[host_end..].starts_with(':') || rest[host_end..].is_empty() {
            let path_at = rest[host_end..].find('/').map(|index| index + host_end);
            return match path_at {
                Some(index) => Ok((&rest[..index], &rest[index..])),
                None => Ok((rest, "")),
            };
        }
        if rest[host_end..].starts_with('/') {
            return Ok((&rest[..host_end], &rest[host_end..]));
        }
        return Err(invalid_base());
    }
    match rest.find('/') {
        Some(index) => Ok((&rest[..index], &rest[index..])),
        None => Ok((rest, "")),
    }
}

fn split_port(host_port: &str) -> Result<(String, Option<u16>)> {
    if let Some(body) = host_port.strip_prefix('[') {
        let Some(end) = body.find(']') else {
            return Err(invalid_base());
        };
        if end == 0 {
            return Err(invalid_base());
        }
        let name = format!("[{}]", &body[..end]);
        let after = &body[end + 1..];
        if after.is_empty() {
            return Ok((name, None));
        }
        let Some(port_text) = after.strip_prefix(':') else {
            return Err(invalid_base());
        };
        return Ok((name, Some(parse_port(port_text)?)));
    }
    if host_port.is_empty() || host_port.contains('[') || host_port.contains(']') {
        return Err(invalid_base());
    }
    match host_port.rsplit_once(':') {
        Some((host, port_text)) => {
            if host.is_empty() || host.contains(':') {
                return Err(invalid_base());
            }
            Ok((host.to_owned(), Some(parse_port(port_text)?)))
        }
        None => Ok((host_port.to_owned(), None)),
    }
}

fn parse_port(text: &str) -> Result<u16> {
    if text.is_empty()
        || !text.bytes().all(|byte| byte.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return Err(invalid_base());
    }
    let port = text.parse::<u16>().map_err(|_| invalid_base())?;
    if port == 0 {
        return Err(invalid_base());
    }
    Ok(port)
}

pub(super) struct BatchRequest<'a> {
    pub endpoint: &'a Endpoint,
    pub key: &'a str,
    pub model: &'a str,
    pub dimensions: u32,
    pub inputs: &'a [String],
    pub timeout: Duration,
}

pub(super) fn embed_batch(request: BatchRequest<'_>) -> Result<Vec<Vec<f32>>> {
    if request
        .key
        .bytes()
        .any(|byte| byte == b'\0' || byte == b'\n' || byte == b'\r')
    {
        return Err(with_host(
            io_ingest("provider request failed", Some("request")),
            &request.endpoint.host,
        ));
    }
    let body = request_body(request.model, request.inputs, request.dimensions);
    let limit = response_limit(request.inputs.len(), request.dimensions);
    let (status, payload) = if request.endpoint.https {
        https_post(request.endpoint, request.key, &body, request.timeout, limit)?
    } else {
        http_post(request.endpoint, request.key, &body, request.timeout, limit)?
    };
    interpret_status(status, &payload, request)
}

pub(super) fn response_limit(batch: usize, dimensions: u32) -> usize {
    let estimate = (batch as u128)
        .saturating_mul(u128::from(dimensions))
        .saturating_mul(64)
        .saturating_add(1024 * 1024);
    let capped = estimate.min(256 * 1024 * 1024);
    usize::try_from(capped).unwrap_or(usize::MAX)
}

fn request_body(model: &str, inputs: &[String], dimensions: u32) -> Vec<u8> {
    let mut body = String::new();
    body.push_str("{\"model\":");
    push_string(&mut body, model);
    body.push_str(",\"input\":[");
    for (index, input) in inputs.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        push_string(&mut body, input);
    }
    body.push_str("],\"dimensions\":");
    body.push_str(&dimensions.to_string());
    body.push('}');
    body.into_bytes()
}

fn interpret_status(
    status: u16,
    payload: &[u8],
    request: BatchRequest<'_>,
) -> Result<Vec<Vec<f32>>> {
    if (300..400).contains(&status) {
        return Err(with_host(
            io_ingest("provider returned a redirect", Some("redirect")),
            &request.endpoint.host,
        ));
    }
    if status != 200 {
        return Err(with_host(
            io_ingest("provider returned a non-success status", Some("response")),
            &request.endpoint.host,
        ));
    }
    let text = std::str::from_utf8(payload).map_err(|_| {
        with_host(
            schema("provider response is not utf-8 json"),
            &request.endpoint.host,
        )
    })?;
    parse_embeddings(
        text,
        request.model,
        request.dimensions,
        request.inputs.len(),
    )
    .map_err(|error| with_host(error, &request.endpoint.host))
}

fn http_post(
    endpoint: &Endpoint,
    key: &str,
    body: &[u8],
    timeout: Duration,
    limit: usize,
) -> Result<(u16, Vec<u8>)> {
    let deadline = Instant::now() + timeout;
    let mut last_error = io_ingest("provider request failed", Some("request"));
    let addresses = endpoint.connect.to_socket_addrs().map_err(|_| {
        with_host(
            io_ingest("provider request failed", Some("request")),
            &endpoint.host,
        )
    })?;
    let mut connected = None;
    for address in addresses {
        let remaining = remaining(deadline).map_err(|error| with_host(error, &endpoint.host))?;
        match TcpStream::connect_timeout(&address, remaining) {
            Ok(stream) => {
                connected = Some(stream);
                break;
            }
            Err(_) => {
                last_error = io_ingest("provider request failed", Some("request"));
            }
        }
    }
    let mut stream = connected.ok_or_else(|| with_host(last_error, &endpoint.host))?;
    let mut header = String::new();
    header.push_str("POST ");
    header.push_str(&endpoint.path);
    header.push_str(" HTTP/1.1\r\nHost: ");
    header.push_str(&endpoint.host);
    header.push_str("\r\nContent-Type: application/json\r\nAuthorization: Bearer ");
    header.push_str(key);
    header.push_str("\r\nContent-Length: ");
    header.push_str(&body.len().to_string());
    header.push_str("\r\nConnection: close\r\n\r\n");
    write_all_deadline(&mut stream, header.as_bytes(), deadline)
        .and_then(|_| write_all_deadline(&mut stream, body, deadline))
        .map_err(|error| with_host(error, &endpoint.host))?;
    read_response(&mut stream, deadline, limit).map_err(|error| with_host(error, &endpoint.host))
}

fn https_post(
    endpoint: &Endpoint,
    key: &str,
    body: &[u8],
    timeout: Duration,
    limit: usize,
) -> Result<(u16, Vec<u8>)> {
    let dir = std::env::temp_dir();
    let stamp = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
    let body_path = dir.join(format!(".coffer-ingest-body-{stamp}-{nanos}-{seq}"));
    let _request_file = write_private(&body_path, body)?;
    let seconds = timeout.as_secs().max(1).to_string();
    let body_arg = format!("@{}", body_path.display());
    let mut child = Command::new("curl")
        .arg("--disable")
        .arg("--http1.1")
        .arg("--silent")
        .arg("--max-filesize")
        .arg(limit.to_string())
        .arg("--max-time")
        .arg(&seconds)
        .arg("--path-as-is")
        .arg("--proto")
        .arg("=https")
        .arg("--proto-redir")
        .arg("-all")
        .arg("--max-redirs")
        .arg("0")
        .arg("--request")
        .arg("POST")
        .arg("--header")
        .arg("Accept:")
        .arg("--header")
        .arg("User-Agent:")
        .arg("--header")
        .arg("Expect:")
        .arg("--header")
        .arg("Content-Type: application/json")
        .arg("--header")
        .arg("@-")
        .arg("--data-binary")
        .arg(&body_arg)
        .arg("--write-out")
        .arg("%{stderr}%{http_code}")
        .env_remove("SSLKEYLOGFILE")
        .arg("--url")
        .arg(&endpoint.url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| {
            with_host(
                io_ingest("provider request failed", Some("request")),
                &endpoint.host,
            )
        })?;
    // curl reads the authorization header from a pipe and never receives it in argv.
    let write_result = child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(format!("Authorization: Bearer {key}\n").as_bytes());
    let failed = || {
        with_host(
            io_ingest("provider request failed", Some("request")),
            &endpoint.host,
        )
    };
    // Keep the HTTP status separate from the bounded, untrusted response body.
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut payload = Vec::new();
    let read_result = (&mut stdout)
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut payload)
        .and_then(|_| std::io::copy(&mut stdout, &mut std::io::sink()));
    if read_result.is_err() || write_result.is_err() {
        let _ = child.kill();
    }
    let output = child.wait_with_output().map_err(|_| failed())?;
    write_result.and(read_result).map_err(|_| failed())?;
    let status = std::str::from_utf8(&output.stderr)
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .filter(|s| (100..600).contains(s))
        .ok_or_else(failed)?;
    if status != 200 {
        return Ok((status, Vec::new()));
    }
    if payload.len() > limit || output.status.code() == Some(63) {
        return Err(with_host(
            io_ingest("provider response is too large", Some("response")),
            &endpoint.host,
        ));
    }
    if !output.status.success() {
        return Err(failed());
    }
    Ok((status, payload))
}

static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

fn write_private(path: &std::path::Path, bytes: &[u8]) -> Result<TempCleanup> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|_| io_ingest("provider request failed", Some("request")))?;
    let guard = TempCleanup(path.to_owned());
    file.write_all(bytes)
        .map_err(|_| io_ingest("provider request failed", Some("request")))?;
    Ok(guard)
}

struct TempCleanup(PathBuf);

impl Drop for TempCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn read_response(
    stream: &mut TcpStream,
    deadline: Instant,
    limit: usize,
) -> Result<(u16, Vec<u8>)> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    let header_end = loop {
        let count = read_deadline(stream, &mut buf, deadline)?;
        if count == 0 {
            return Err(io_ingest("provider request failed", Some("request")));
        }
        raw.extend_from_slice(&buf[..count]);
        if raw.len() > 64 * 1024 + limit {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
        if let Some(index) = find_header_end(&raw) {
            break index;
        }
        if raw.len() > 64 * 1024 {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
    };
    let header_bytes = raw[..header_end].to_vec();
    let mut body = raw[header_end + 4..].to_vec();
    let header_text = std::str::from_utf8(&header_bytes)
        .map_err(|_| io_ingest("provider returned a non-success status", Some("response")))?;
    let mut lines = header_text.split("\r\n");
    let status = parse_status_text(lines.next().unwrap_or(""))
        .ok_or_else(|| io_ingest("provider returned a non-success status", Some("response")))?;
    if status != 200 {
        return Ok((status, Vec::new()));
    }
    let mut content_length = None;
    let mut chunked = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse::<usize>().ok();
        } else if name.eq_ignore_ascii_case("transfer-encoding")
            && value.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    if chunked {
        let mut pending = body;
        let decoded = read_chunked(stream, &mut pending, deadline, limit)?;
        return Ok((status, decoded));
    }
    if let Some(length) = content_length {
        if length > limit {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
        while body.len() < length {
            let count = read_deadline(stream, &mut buf, deadline)?;
            if count == 0 {
                return Err(io_ingest("provider request failed", Some("request")));
            }
            let take = count.min(length - body.len());
            body.extend_from_slice(&buf[..take]);
        }
        body.truncate(length);
        return Ok((status, body));
    }
    loop {
        let count = read_deadline(stream, &mut buf, deadline)?;
        if count == 0 {
            break;
        }
        if body.len() + count > limit {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
        body.extend_from_slice(&buf[..count]);
    }
    Ok((status, body))
}

fn read_chunked(
    stream: &mut TcpStream,
    pending: &mut Vec<u8>,
    deadline: Instant,
    limit: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let line = read_line(stream, pending, &mut buf, deadline)?;
        let size_text = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| io_ingest("provider returned a non-success status", Some("response")))?;
        if size == 0 {
            loop {
                let trailer = read_line(stream, pending, &mut buf, deadline)?;
                if trailer.is_empty() {
                    break;
                }
            }
            break;
        }
        if out.len().saturating_add(size) > limit {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
        while pending.len() < size + 2 {
            let count = read_deadline(stream, &mut buf, deadline)?;
            if count == 0 {
                return Err(io_ingest("provider request failed", Some("request")));
            }
            pending.extend_from_slice(&buf[..count]);
        }
        out.extend_from_slice(&pending[..size]);
        if &pending[size..size + 2] != b"\r\n" {
            return Err(io_ingest(
                "provider returned a non-success status",
                Some("response"),
            ));
        }
        pending.drain(..size + 2);
    }
    Ok(out)
}

fn read_line(
    stream: &mut TcpStream,
    pending: &mut Vec<u8>,
    buf: &mut [u8],
    deadline: Instant,
) -> Result<String> {
    loop {
        if let Some(index) = pending.windows(2).position(|pair| pair == b"\r\n") {
            let line = std::str::from_utf8(&pending[..index])
                .map_err(|_| io_ingest("provider returned a non-success status", Some("response")))?
                .to_owned();
            pending.drain(..index + 2);
            return Ok(line);
        }
        let count = read_deadline(stream, buf, deadline)?;
        if count == 0 {
            return Err(io_ingest("provider request failed", Some("request")));
        }
        pending.extend_from_slice(&buf[..count]);
        if pending.len() > 64 * 1024 {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
    }
}

fn find_header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_status_text(line: &str) -> Option<u16> {
    let mut parts = line.split_whitespace();
    let version = parts.next()?;
    if !version.starts_with("HTTP/") {
        return None;
    }
    parts.next()?.parse().ok()
}

fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| io_ingest("provider request failed", Some("request")))
}

fn read_deadline(stream: &mut TcpStream, buf: &mut [u8], deadline: Instant) -> Result<usize> {
    stream
        .set_read_timeout(Some(remaining(deadline)?))
        .map_err(|_| io_ingest("provider request failed", Some("request")))?;
    match stream.read(buf) {
        Ok(count) => Ok(count),
        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
            read_deadline(stream, buf, deadline)
        }
        Err(_) => Err(io_ingest("provider request failed", Some("request"))),
    }
}

fn write_all_deadline(stream: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream
            .set_write_timeout(Some(remaining(deadline)?))
            .map_err(|_| io_ingest("provider request failed", Some("request")))?;
        match stream.write(bytes) {
            Ok(0) => return Err(io_ingest("provider request failed", Some("request"))),
            Ok(count) => bytes = &bytes[count..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(io_ingest("provider request failed", Some("request"))),
        }
    }
    Ok(())
}

pub(super) fn parse_embeddings(
    body: &str,
    model: &str,
    dimensions: u32,
    count: usize,
) -> Result<Vec<Vec<f32>>> {
    let body = normalize_provider_numbers(body)?;
    let value = match parse(&body) {
        Ok(value) => value,
        Err(error)
            if error.code() == ErrorCode::InvalidJsonlSchema
                && error.message() == "JSON number is not finite" =>
        {
            return Err(vector_length("embedding values must be finite numbers"));
        }
        Err(_) => return Err(schema("provider response is not a valid embeddings object")),
    };
    let obj = value
        .as_object()
        .ok_or_else(|| schema("provider response is not a valid embeddings object"))?;
    if let Some(returned) = obj.get("model") {
        match returned.as_str() {
            Some(returned) if returned == model => {}
            Some(_) => {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "provider model does not match the request",
                )
                .with_detail("field", "model")
                .ingestion_boundary());
            }
            None => return Err(schema("provider response model is not a string")),
        }
    }
    let data = obj
        .get("data")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| schema("provider response is missing data"))?;
    if data.len() != count {
        return Err(vector_length(
            "provider response count does not match the request",
        ));
    }
    let mut slots = vec![None; count];
    for item in data {
        let item = item
            .as_object()
            .ok_or_else(|| schema("provider response data item is not an object"))?;
        let index_value = item
            .get("index")
            .ok_or_else(|| schema("provider response data item is missing index"))?;
        let index = json_index(index_value)?;
        let embedding = item
            .get("embedding")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| schema("provider response data item is missing embedding"))?;
        if index >= count || slots[index].is_some() {
            return Err(vector_length(
                "provider response indexes do not match the request",
            ));
        }
        if embedding.len() != dimensions as usize {
            return Err(vector_length("embedding length does not match dimensions"));
        }
        let mut vector = Vec::with_capacity(embedding.len());
        for value in embedding {
            let Some(number) = value.as_f64() else {
                return Err(vector_length("embedding values must be numbers"));
            };
            let narrowed = number as f32;
            if !narrowed.is_finite() {
                return Err(vector_length("embedding values must be finite numbers"));
            }
            vector.push(narrowed);
        }
        slots[index] = Some(vector);
    }
    slots
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| vector_length("provider response indexes do not match the request"))
        })
        .collect()
}

fn normalize_provider_numbers(body: &str) -> Result<String> {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'"' {
            let start = index;
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index += 1;
                    if index < bytes.len() && bytes[index] == b'u' {
                        index = index.saturating_add(5).min(bytes.len());
                    } else {
                        index += 1;
                    }
                    continue;
                }
                index += 1;
                if bytes[index - 1] == b'"' {
                    break;
                }
            }
            out.push_str(&body[start..index]);
            continue;
        }
        if is_number_start(bytes, index) {
            let start = index;
            index = number_end(bytes, index);
            out.push_str(&rounded_number(&body[start..index])?);
            continue;
        }
        let next = index + utf8_len(bytes[index]);
        out.push_str(&body[index..next.min(bytes.len())]);
        index = next.min(bytes.len());
    }
    Ok(out)
}

fn is_number_start(bytes: &[u8], index: usize) -> bool {
    match bytes[index] {
        b'-' => bytes
            .get(index + 1)
            .is_some_and(|byte| byte.is_ascii_digit()),
        b'0'..=b'9' => true,
        _ => false,
    }
}

fn number_end(bytes: &[u8], mut index: usize) -> usize {
    if bytes.get(index) == Some(&b'-') {
        index += 1;
    }
    while bytes.get(index).is_some_and(|byte| byte.is_ascii_digit()) {
        index += 1;
    }
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        while bytes.get(index).is_some_and(|byte| byte.is_ascii_digit()) {
            index += 1;
        }
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        while bytes.get(index).is_some_and(|byte| byte.is_ascii_digit()) {
            index += 1;
        }
    }
    index
}

fn rounded_number(token: &str) -> Result<String> {
    let probe = format!("[{token}]");
    match parse(&probe) {
        Ok(_) => Ok(token.to_owned()),
        Err(error)
            if error.message() == "JSON number is not finite"
                || error.message()
                    == "integer cannot be represented exactly as binary64; encode it as a string" =>
        {
            match token.parse::<f64>() {
                Ok(value) if value.is_finite() => {
                    let narrowed = value as f32;
                    if !narrowed.is_finite() {
                        return Err(vector_length("embedding values must be finite numbers"));
                    }
                    Ok(format_f32(narrowed))
                }
                _ => Err(vector_length("embedding values must be finite numbers")),
            }
        }
        Err(_) => Ok(token.to_owned()),
    }
}

fn utf8_len(byte: u8) -> usize {
    match byte {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

fn json_index(value: &JsonValue) -> Result<usize> {
    let Some(number) = value.as_f64() else {
        return Err(schema("provider response index is not an integer"));
    };
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
        return Err(schema("provider response index is not an integer"));
    }
    if number > usize::MAX as f64 {
        return Err(vector_length(
            "provider response indexes do not match the request",
        ));
    }
    Ok(number as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_openai_url_is_the_frozen_endpoint() {
        let endpoint = build_endpoint(OPENAI_DEFAULT_BASE).unwrap();
        assert_eq!(endpoint.url, OPENAI_EMBEDDINGS_URL);
        assert_eq!(endpoint.host, "api.openai.com");
        assert!(endpoint.https);
        assert_eq!(endpoint.path, "/v1/embeddings");
    }

    #[test]
    fn base_url_rejects_suffix_userinfo_query_and_preserves_case() {
        assert!(build_endpoint("https://api.openai.com/v1/embeddings").is_err());
        assert!(build_endpoint("https://user:secret@127.0.0.1/v1").is_err());
        assert!(build_endpoint("http://127.0.0.1/v1?api_key=secret").is_err());
        assert!(build_endpoint("http://127.0.0.1/v1#frag").is_err());
        assert!(build_endpoint("file:///tmp/model.gguf").is_err());
        let endpoint = build_endpoint("HTTP://API.Example:443/V1").unwrap_err();
        assert_eq!(endpoint.code(), ErrorCode::CliUsage);
        let kept = build_endpoint("http://API.Example:8080/V1/").unwrap();
        assert_eq!(kept.url, "http://API.Example:8080/V1/embeddings");
        assert_eq!(kept.host, "API.Example:8080");
        let plain = build_endpoint("http://127.0.0.1:80").unwrap();
        assert_eq!(plain.url, "http://127.0.0.1/embeddings");
        assert_eq!(plain.host, "127.0.0.1");
        let v6 = build_endpoint("http://[::1]:9/v1").unwrap();
        assert_eq!(v6.url, "http://[::1]:9/v1/embeddings");
        assert_eq!(v6.host, "[::1]:9");
    }

    #[test]
    fn provider_numbers_round_in_range_and_reject_non_finite() {
        let rounded = parse_embeddings(
            r#"{"model":"example","data":[{"index":0,"embedding":[9007199254740993]}]}"#,
            "example",
            1,
            1,
        )
        .unwrap();
        let wide = 9007199254740993f64 as f32;
        assert!(wide.is_finite());
        assert_eq!(rounded, vec![vec![wide]]);

        let huge = parse_embeddings(
            r#"{"model":"example","data":[{"index":0,"embedding":[1e309]}]}"#,
            "example",
            1,
            1,
        )
        .unwrap_err();
        assert_eq!(huge.code(), ErrorCode::InvalidVectorLength);
        assert_eq!(huge.message(), "embedding values must be finite numbers");

        for token in ["1e23", "1e38"] {
            let finite = parse_embeddings(
                &format!(r#"{{"model":"example","data":[{{"index":0,"embedding":[{token}]}}]}}"#),
                "example",
                1,
                1,
            )
            .unwrap();
            let expected = token.parse::<f64>().unwrap() as f32;
            assert!(expected.is_finite(), "{token}");
            assert_eq!(finite, vec![vec![expected]]);
            let spelled = super::super::float::format_f32(expected);
            assert!(
                crate::json::parse(&format!("[{spelled}]")).is_ok(),
                "{spelled}"
            );
        }
        let overflow = parse_embeddings(
            r#"{"model":"example","data":[{"index":0,"embedding":[1e39]}]}"#,
            "example",
            1,
            1,
        )
        .unwrap_err();
        assert_eq!(overflow.code(), ErrorCode::InvalidVectorLength);

        let quoted = parse_embeddings(
            r#"{"model":"example","echo":"1e309","data":[{"index":0,"embedding":[1]}]}"#,
            "example",
            1,
            1,
        )
        .unwrap();
        assert_eq!(quoted, vec![vec![1.0]]);
    }
}
