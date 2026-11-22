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

// For the OpenAI provider, requests without an explicit base use this endpoint.
// The compatible provider requires a caller-supplied base.
pub(super) const OPENAI_DEFAULT_BASE: &str = "https://api.openai.com/v1";
#[cfg(test)]
pub(super) const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";
pub(super) const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Endpoint {
    // Complete normalized embeddings URL used for HTTPS requests and the endpoint
    // fingerprint.
    pub url: String,
    // HTTP Host value and diagnostic label, including an explicit non-default port.
    pub host: String,
    // Socket address spelling with a port even when the URL used the scheme
    // default.
    pub connect: String,
    // HTTP request target including the base path and appended embeddings route.
    pub path: String,
    // Selects HTTPS transport when true and the plain HTTP socket path otherwise.
    pub https: bool,
}

pub(super) fn build_endpoint(base: &str) -> Result<Endpoint> {
    // Treat the supplied string as a base endpoint with an optional path prefix.
    // Reject the disallowed components before splitting it into pieces so the same
    // input cannot produce different request and diagnostic hosts.
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
    // Normalize trailing slashes before testing the scheme and suffix. A base
    // ending in a slash then produces the same endpoint and fingerprint as its
    // unsuffixed spelling.
    let trimmed = base.trim_end_matches('/');
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err(invalid_base());
    };
    // Accept only the two implemented schemes. The transport decision stored in
    // Endpoint is made from this spelling after validation.
    if scheme != "http" && scheme != "https" {
        return Err(invalid_base());
    }
    if rest.is_empty() || rest.starts_with('/') {
        return Err(invalid_base());
    }
    // Split the authority from the optional path while the original slices are
    // still available. A colon inside a bracketed host must not be mistaken for a
    // port or path separator.
    let (host_port, path) = split_host_path(rest)?;
    // The caller supplies a base URL, not the complete operation route. Reject a
    // base already ending at the embeddings endpoint before appending that suffix
    // again.
    if path.ends_with("/embeddings") {
        return Err(cli(
            "base url must not already end in /embeddings",
            Some("--base-url"),
            Some("embeddings_suffix"),
        ));
    }
    let (name, explicit_port) = split_port(&host_port)?;
    let default_port = if scheme == "http" { 80 } else { 443 };
    // Keep three related spellings distinct: URL for HTTPS and identity, Host for
    // the HTTP header and error detail, and a socket address which always includes
    // its port.
    let port = explicit_port.unwrap_or(default_port);
    let mut url = String::new();
    url.push_str(scheme);
    url.push_str("://");
    url.push_str(&name);
    let mut host = name.clone();
    let mut connect = name.clone();
    // Omit a scheme's default port from both the normalized URL and Host.
    // Non-default ports identify different endpoints and remain visible in those
    // forms.
    if port != default_port {
        url.push(':');
        url.push_str(&port.to_string());
        host.push(':');
        host.push_str(&port.to_string());
    }
    // Always make the socket port explicit, including 80 and 443. Name resolution
    // for the socket receives this form rather than the normalized HTTP authority.
    connect.push(':');
    connect.push_str(&port.to_string());
    if path.is_empty() {
        url.push_str("/embeddings");
    } else {
        if !path.starts_with('/') {
            return Err(invalid_base());
        }
        // Keep the supplied base path before appending the same operation suffix
        // used by either transport.
        url.push_str(path);
        url.push_str("/embeddings");
    }
    // The socket transport uses an origin-form request target. Build it from the
    // same path prefix that was appended to the full HTTPS URL.
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
    // A colon inside brackets belongs to the host spelling. Locate the closing
    // bracket first so only a following colon can introduce the optional port.
    if let Some(body) = rest.strip_prefix('[') {
        let Some(end) = body.find(']') else {
            return Err(invalid_base());
        };
        // The search started after the opening bracket. Add both bracket bytes
        // when mapping the slice index back to the original authority.
        let host_end = end + 2;
        if host_end > rest.len() {
            return Err(invalid_base());
        }
        // Search for the path delimiter only after the bracketed host. A port,
        // when present, stays in the authority and receives its own numeric
        // validation.
        if rest[host_end..].starts_with(':') || rest[host_end..].is_empty() {
            let path_at = rest[host_end..].find('/').map(|index| index + host_end);
            return match path_at {
                Some(index) => Ok((&rest[..index], &rest[index..])),
                None => Ok((rest, "")),
            };
        }
        if rest[host_end..].starts_with('/') {
            // A slash immediately after the bracket begins the path without an
            // explicit port. Reject other trailing text rather than interpreting
            // it as part of the literal.
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
        // Brackets with no contents cannot identify a host. Retain the brackets
        // when forming the socket and Host spellings so a later port separator
        // stays unambiguous.
        if end == 0 {
            return Err(invalid_base());
        }
        let name = format!("[{}]", &body[..end]);
        // Keep the host and its suffix separate. With no suffix the scheme default
        // applies; with a suffix, require the colon and validate the entire
        // remaining port text.
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
    // For an unbracketed host there may be only one colon introducing the port.
    // Multi-colon host spellings must use brackets before reaching this branch.
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

// Require a nonzero decimal port in one unambiguous spelling. The parser rejects
// signs, leading zeroes and values beyond the u16 socket-port range.
fn parse_port(text: &str) -> Result<u16> {
    if text.is_empty()
        || !text.bytes().all(|byte| byte.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return Err(invalid_base());
    }
    // Parse into the exact type supported by socket addresses after checking the
    // digit spelling. Zero fits that type but does not designate a usable provider
    // port.
    let port = text.parse::<u16>().map_err(|_| invalid_base())?;
    if port == 0 {
        return Err(invalid_base());
    }
    Ok(port)
}

pub(super) struct BatchRequest<'a> {
    // Validated destination shared by the request and response diagnostics.
    pub endpoint: &'a Endpoint,
    // Borrowed bearer credential used for authorization. Errors identify the host.
    pub key: &'a str,
    // Model name sent in the request body.
    pub model: &'a str,
    // Requested coordinates per embedding, also used to bound and validate the
    // response.
    pub dimensions: u32,
    // Batch texts in input order. Provider response indices refer back to this
    // slice.
    pub inputs: &'a [String],
    // Transport timeout for this batch request.
    pub timeout: Duration,
}

pub(super) fn embed_batch(request: BatchRequest<'_>) -> Result<Vec<Vec<f32>>> {
    // Check the borrowed credential before constructing or transmitting a request.
    // Return the generic request-stage diagnostic with the endpoint host so the
    // credential itself never appears in this error.
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
    // Construct the payload before selecting transport. It includes the model,
    // texts and requested dimensions and can be passed unchanged to either
    // implementation.
    let body = request_body(request.model, request.inputs, request.dimensions);
    let limit = response_limit(request.inputs.len(), request.dimensions);
    // Both transports receive the same serialized request and response limit. A
    // transport returns the HTTP status and bounded body so one validator handles
    // provider-level errors and vector shape.
    let (status, payload) = if request.endpoint.https {
        https_post(request.endpoint, request.key, &body, request.timeout, limit)?
    } else {
        http_post(request.endpoint, request.key, &body, request.timeout, limit)?
    };
    interpret_status(status, &payload, request)
}

pub(super) fn response_limit(batch: usize, dimensions: u32) -> usize {
    // Estimate response bytes from the requested coordinate count with room for
    // decimal text and fixed JSON overhead. Saturating arithmetic preserves the
    // hard ceiling even when batch and dimension are large.
    let estimate = (batch as u128)
        .saturating_mul(u128::from(dimensions))
        .saturating_mul(64)
        .saturating_add(1024 * 1024);
    // Cap the estimate independently of the requested embedding size. The
    // transport enforces the resulting byte limit before JSON decoding can build a
    // vector of the declared shape.
    let capped = estimate.min(256 * 1024 * 1024);
    usize::try_from(capped).unwrap_or(usize::MAX)
}

fn request_body(model: &str, inputs: &[String], dimensions: u32) -> Vec<u8> {
    // Serialize directly to the HTTP request body. No field here needs to contain
    // the bearer credential, which is supplied separately to the transport.
    let mut body = String::new();
    body.push_str("{\"model\":");
    push_string(&mut body, model);
    body.push_str(",\"input\":[");
    // Encode texts in their input order and escape each as a separate JSON string.
    // Provider result indices are checked against this same batch when rows are
    // decoded.
    for (index, input) in inputs.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        push_string(&mut body, input);
    }
    // Request dimensions as an integer field. Every response vector is checked
    // against the same requested dimension.
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
    // Classify redirects separately and stop before decoding their bodies as
    // embeddings. No Location target is selected for another authenticated
    // request.
    if (300..400).contains(&status) {
        return Err(with_host(
            io_ingest("provider returned a redirect", Some("redirect")),
            &request.endpoint.host,
        ));
    }
    // Only status 200 has the embedding payload contract. Expose the original
    // endpoint host on other statuses without copying its response body into
    // diagnostics.
    if status != 200 {
        return Err(with_host(
            io_ingest("provider returned a non-success status", Some("response")),
            &request.endpoint.host,
        ));
    }
    // Validate UTF-8 before parsing JSON structure. Malformed bytes get a
    // response-schema failure while a transport error has already returned from
    // the chosen POST path.
    let text = std::str::from_utf8(payload).map_err(|_| {
        with_host(
            schema("provider response is not utf-8 json"),
            &request.endpoint.host,
        )
    })?;
    // Pass the request's model, dimension and input count into decoding. A
    // syntactically valid response still has to describe the embeddings requested
    // in this batch.
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
    // One deadline starts before resolving addresses and is rechecked before each
    // connection and each subsequent I/O stage. The synchronous address lookup
    // itself is not interrupted by these deadline checks.
    let deadline = Instant::now() + timeout;
    // Keep a request failure even when address resolution yields no candidates.
    // The caller attaches the chosen host to errors after all attempts have ended.
    let mut last_error = io_ingest("provider request failed", Some("request"));
    let addresses = endpoint.connect.to_socket_addrs().map_err(|_| {
        with_host(
            io_ingest("provider request failed", Some("request")),
            &endpoint.host,
        )
    })?;
    let mut connected = None;
    // Walk addresses until one connection succeeds. This is address selection for
    // the same request: no HTTP body has been sent and no response is replayed
    // after a later failure.
    for address in addresses {
        let remaining = remaining(deadline).map_err(|error| with_host(error, &endpoint.host))?;
        match TcpStream::connect_timeout(&address, remaining) {
            // Use the first successful socket for the single HTTP request. Only
            // connection failures before this point try another address.
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
    // Only construct the authorization header after connection selection. DNS and
    // connect errors can report the endpoint without retaining an attempted header
    // in the diagnostic.
    let mut header = String::new();
    header.push_str("POST ");
    header.push_str(&endpoint.path);
    header.push_str(" HTTP/1.1\r\nHost: ");
    header.push_str(&endpoint.host);
    header.push_str("\r\nContent-Type: application/json\r\nAuthorization: Bearer ");
    header.push_str(key);
    // Content-Length is the encoded byte count, not the number of input strings or
    // Unicode characters. The same body is written immediately after this header.
    header.push_str("\r\nContent-Length: ");
    header.push_str(&body.len().to_string());
    header.push_str("\r\nConnection: close\r\n\r\n");
    // Keep one deadline for writing the header and body and then reading the
    // response. If a write fails partway, return its stage error rather than
    // sending the batch again.
    write_all_deadline(&mut stream, header.as_bytes(), deadline)
        .and_then(|_| write_all_deadline(&mut stream, body, deadline))
        .map_err(|error| with_host(error, &endpoint.host))?;
    // Read the response with the same deadline and include only the validated host
    // when reporting parser or I/O errors from this connection.
    read_response(&mut stream, deadline, limit).map_err(|error| with_host(error, &endpoint.host))
}

fn https_post(
    endpoint: &Endpoint,
    key: &str,
    body: &[u8],
    timeout: Duration,
    limit: usize,
) -> Result<(u16, Vec<u8>)> {
    // curl needs a separately readable request body because its standard input
    // carries the authorization header. Keep the scratch file alive until the
    // child has finished with its pathname.
    let dir = std::env::temp_dir();
    let stamp = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    // Several batches may prepare files in the same process and timestamp. The
    // sequence distinguishes their candidate names; exclusive creation still
    // decides whether a name is available.
    let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
    let body_path = dir.join(format!(".coffer-ingest-body-{stamp}-{nanos}-{seq}"));
    // Create the body file before starting the child. If process creation fails,
    // the local guard can still remove the plain request text from disk.
    let _request_file = write_private(&body_path, body)?;
    // curl receives whole seconds for its timeout. Clamp subsecond values to one
    // because zero disables its transfer time limit.
    let seconds = timeout.as_secs().max(1).to_string();
    // Pass the file reference as one process argument. The command receives the
    // actual path directly without shell quoting, even when the temporary
    // directory contains spaces.
    let body_arg = format!("@{}", body_path.display());
    let mut child = Command::new("curl")
        // Keep --disable first so per-user curl configuration cannot change this
        // invocation's transport behavior.
        .arg("--disable")
        .arg("--http1.1")
        .arg("--silent")
        // Bound the transfer in curl as well as the bytes retained by this
        // process. The process status and the sentinel byte below cover limits
        // detected by different layers.
        .arg("--max-filesize")
        .arg(limit.to_string())
        .arg("--max-time")
        .arg(&seconds)
        // Use the validated URL path as supplied. This transport is for HTTPS
        // only, and a redirect must be reported to the caller without sending
        // another request.
        .arg("--path-as-is")
        .arg("--proto")
        .arg("=https")
        .arg("--proto-redir")
        .arg("-all")
        .arg("--max-redirs")
        .arg("0")
        .arg("--request")
        // Remove curl's normal Accept, User-Agent and Expect headers explicitly.
        // Set the content type for the JSON bytes that were written to the scratch
        // file.
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
        // Keep the two input channels distinct. @- reads header lines from stdin;
        // --data-binary reads the body file without rewriting line endings or
        // treating its content as options.
        .arg("@-")
        .arg("--data-binary")
        .arg(&body_arg)
        // Emit only the HTTP status to stderr while the body flows through stdout.
        // The code parses stderr as a single status, so extra curl diagnostics do
        // not belong in this channel.
        .arg("--write-out")
        .arg("%{stderr}%{http_code}")
        // Leave TLS session key logging disabled for the child even if the parent
        // environment enabled it.
        .env_remove("SSLKEYLOGFILE")
        .arg("--url")
        .arg(&endpoint.url)
        .stdin(Stdio::piped())
        // Take stdout for bounded streaming below. Leave stderr attached until
        // wait_with_output collects the separate transfer status.
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
    // Taking stdin makes the header pipe close after write_all finishes. curl sees
    // EOF on the header list without waiting for child completion.
    let write_result = child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(format!("Authorization: Bearer {key}\n").as_bytes());
    // Report a transport problem using the already selected host. Neither stderr
    // nor the request body becomes part of a user-facing failure.
    let failed = || {
        with_host(
            io_ingest("provider request failed", Some("request")),
            &endpoint.host,
        )
    };
    // Keep the HTTP status separate from the bounded, untrusted response body.
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut payload = Vec::new();
    // Save one byte beyond the cap to distinguish an exact fit from a response
    // that grew too large. Drain the rest to a sink so curl can finish even after
    // that decision is possible.
    let read_result = (&mut stdout)
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut payload)
        .and_then(|_| std::io::copy(&mut stdout, &mut std::io::sink()));
    // A broken pipe on either side invalidates the request. Attempt to stop the
    // child, then wait for it on both success and failure paths.
    if read_result.is_err() || write_result.is_err() {
        let _ = child.kill();
    }
    // Always reap the process after draining or terminating it. A pipe failure
    // still wins over an HTTP status printed by a child with an incomplete request
    // or response.
    let output = child.wait_with_output().map_err(|_| failed())?;
    write_result.and(read_result).map_err(|_| failed())?;
    // Accept a parsed HTTP status only in the protocol range. A child that
    // produced no usable status yields a request error even if its process exit
    // code is available.
    let status = std::str::from_utf8(&output.stderr)
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .filter(|s| (100..600).contains(s))
        .ok_or_else(failed)?;
    // Preserve a non-200 status for the common redirect and provider-error
    // classifier. The caller does not parse or surface an error response body.
    if status != 200 {
        return Ok((status, Vec::new()));
    }
    // curl can stop at its file-size limit before our extra-byte read detects
    // overflow. Treat its size-limit exit code as the same response failure.
    if payload.len() > limit || output.status.code() == Some(63) {
        return Err(with_host(
            io_ingest("provider response is too large", Some("response")),
            &endpoint.host,
        ));
    }
    // Even when the HTTP status was 200 and the body fit, require a successful
    // transfer before returning bytes to the embeddings parser.
    if !output.status.success() {
        return Err(failed());
    }
    Ok((status, payload))
}

static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

fn write_private(path: &std::path::Path, bytes: &[u8]) -> Result<TempCleanup> {
    let mut options = std::fs::OpenOptions::new();
    // Create the request file exclusively. A collision fails the request without
    // truncating an existing file or making that path this request's cleanup
    // responsibility.
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Set the Unix mode as part of creation so a newly visible file containing
        // user text is already owner-only, subject to a more restrictive umask.
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|_| io_ingest("provider request failed", Some("request")))?;
    // Install the guard only after this request creates the file, but before
    // writing its contents. A partial write then triggers the same removal attempt
    // as normal completion.
    let guard = TempCleanup(path.to_owned());
    file.write_all(bytes)
        .map_err(|_| io_ingest("provider request failed", Some("request")))?;
    Ok(guard)
}

// Owning this path means this request successfully created it. Keep it in the
// guard until the child has finished reading the upload file.
struct TempCleanup(PathBuf);

impl Drop for TempCleanup {
    fn drop(&mut self) {
        // File removal here is best effort. The caller's transport or result error
        // must survive an additional failure while dropping the scratch guard.
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
    // A single read can return the end of the headers and the start of the body.
    // Keep that prefix for the body decoder and apply the same deadline to
    // subsequent reads so a slow stream cannot renew its time allowance.
    let header_end = loop {
        // The header delimiter can span reads. A closed connection before the
        // complete header is a request failure even if some status-line bytes
        // arrived.
        let count = read_deadline(stream, &mut buf, deadline)?;
        if count == 0 {
            return Err(io_ingest("provider request failed", Some("request")));
        }
        raw.extend_from_slice(&buf[..count]);
        // Allow room for headers and body bytes that arrived in the same read.
        // Apply the separate unfinished-header limit below only while the
        // delimiter has not yet been found.
        if raw.len() > 64 * 1024 + limit {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
        // A single read can complete the delimiter and bring payload bytes with
        // it. Check for the delimiter before applying the separate cap to an
        // unfinished header.
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
    // Exclude the full CRLF delimiter and save every byte after it as the first
    // body fragment. Re-reading these bytes from the socket would lose the start
    // of the response.
    let mut body = raw[header_end + 4..].to_vec();
    // Parse the response head as UTF-8 before extracting status or framing.
    // Invalid head text has no response contract for the embedding decoder to
    // apply.
    let header_text = std::str::from_utf8(&header_bytes)
        .map_err(|_| io_ingest("provider returned a non-success status", Some("response")))?;
    let mut lines = header_text.split("\r\n");
    let status = parse_status_text(lines.next().unwrap_or(""))
        .ok_or_else(|| io_ingest("provider returned a non-success status", Some("response")))?;
    // Once the status is known to be non-200, skip body framing and parsing. The
    // shared status interpreter only needs the code on this path.
    if status != 200 {
        return Ok((status, Vec::new()));
    }
    let mut content_length = None;
    let mut chunked = false;
    for line in lines {
        // Ignore lines without a header separator. Split on its first occurrence
        // so later colons remain part of the value.
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        // Parse a native-sized decimal content length when available. An
        // unparseable value leaves no usable declared length, so any recognized
        // chunked framing or the EOF path governs.
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse::<usize>().ok();
        } else if name.eq_ignore_ascii_case("transfer-encoding")
            && value.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    // Chunked framing takes precedence when both headers are present. Without it,
    // the advertised length ends the response even if the server keeps the
    // connection open; an unframed body runs to EOF.
    if chunked {
        let mut pending = body;
        // Only decoded chunk bytes become provider JSON. Bytes already read after
        // the header must enter the chunk decoder before it reads more from the
        // stream.
        let decoded = read_chunked(stream, &mut pending, deadline, limit)?;
        return Ok((status, decoded));
    }
    // A declared success body can be rejected for size before any further read.
    // Wait only for the declared number of bytes, independent of when the peer
    // closes.
    if let Some(length) = content_length {
        if length > limit {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
        while body.len() < length {
            let count = read_deadline(stream, &mut buf, deadline)?;
            // EOF is a failure while any declared body bytes are still missing.
            // The unframed branch below gives EOF a different meaning.
            if count == 0 {
                return Err(io_ingest("provider request failed", Some("request")));
            }
            // A read can straddle the declared end. Append just the missing bytes
            // and also trim any excess already present in the original header
            // read.
            let take = count.min(length - body.len());
            body.extend_from_slice(&buf[..take]);
        }
        body.truncate(length);
        return Ok((status, body));
    }
    // For an unframed body, EOF is the message boundary. Subsequent socket reads
    // share the header's deadline and each addition is checked before it is
    // appended.
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
    // Keep unconsumed wire bytes separate from the decoded output. A single read
    // can cover part of a size line, a whole chunk, or several successive chunks.
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let line = read_line(stream, pending, &mut buf, deadline)?;
        // A chunk size is hexadecimal. Ignore extension text after the first
        // semicolon and keep framing bytes out of the decoded payload returned to
        // the JSON parser.
        let size_text = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| io_ingest("provider returned a non-success status", Some("response")))?;
        if size == 0 {
            // A zero-size chunk ends the data, but trailer fields can follow it.
            // Consume lines through the terminating empty line without adding
            // trailers to the embedding response.
            loop {
                let trailer = read_line(stream, pending, &mut buf, deadline)?;
                if trailer.is_empty() {
                    break;
                }
            }
            break;
        }
        // Apply the response cap to the cumulative decoded bytes before waiting
        // for this chunk. Chunk headers, extensions and trailing CRLF are outside
        // that count.
        if out.len().saturating_add(size) > limit {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
        // The payload can include CRLF of its own. Use the declared byte length,
        // then require the two framing bytes immediately following it.
        while pending.len() < size + 2 {
            let count = read_deadline(stream, &mut buf, deadline)?;
            if count == 0 {
                return Err(io_ingest("provider request failed", Some("request")));
            }
            pending.extend_from_slice(&buf[..count]);
        }
        out.extend_from_slice(&pending[..size]);
        // The chunk's trailing CRLF is required even if its declared payload
        // length fit the available bytes. A framing error discards the incomplete
        // response.
        if &pending[size..size + 2] != b"\r\n" {
            return Err(io_ingest(
                "provider returned a non-success status",
                Some("response"),
            ));
        }
        // Consume only this chunk's data and terminator. The same socket read can
        // also contain the start of the next size line, which must survive in
        // pending.
        pending.drain(..size + 2);
    }
    Ok(out)
}

// Read one CRLF-terminated response line from already buffered bytes or the
// socket. Leave later bytes in pending for the caller's next read.
fn read_line(
    stream: &mut TcpStream,
    pending: &mut Vec<u8>,
    buf: &mut [u8],
    deadline: Instant,
) -> Result<String> {
    loop {
        // Search the accumulated bytes before reading again. CR and LF can arrive
        // in different calls, and a successful delimiter identifies exactly the
        // consumed prefix.
        if let Some(index) = pending.windows(2).position(|pair| pair == b"\r\n") {
            // Interpret only the line before CRLF as text. A line containing
            // invalid UTF-8 becomes a response error without including provider
            // bytes in the message.
            let line = std::str::from_utf8(&pending[..index])
                .map_err(|_| io_ingest("provider returned a non-success status", Some("response")))?
                .to_owned();
            // Drain the delimiter as well as the line. Bytes after it may already
            // contain the next line or part of the body.
            pending.drain(..index + 2);
            return Ok(line);
        }
        // Read with the same absolute deadline used for earlier parts of this
        // response. A partial line does not start another timeout window.
        let count = read_deadline(stream, buf, deadline)?;
        // A zero-byte read before a delimiter means the line is incomplete. End
        // the request instead of retrying EOF.
        if count == 0 {
            return Err(io_ingest("provider request failed", Some("request")));
        }
        pending.extend_from_slice(&buf[..count]);
        // Cap accumulated framing bytes after each append. This bounds buffering
        // when a peer never supplies the expected delimiter.
        if pending.len() > 64 * 1024 {
            return Err(io_ingest(
                "provider response is too large",
                Some("response"),
            ));
        }
    }
}

// Locate the separator between the HTTP header block and body. Return its starting
// byte so the caller can retain any body bytes already read with the headers.
fn find_header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|window| window == b"\r\n\r\n")
}

// Parse only the version prefix and numeric status. Response classification
// depends on the code, so the reason phrase need not match a particular spelling.
fn parse_status_text(line: &str) -> Option<u16> {
    let mut parts = line.split_whitespace();
    // Check the protocol token before reading status so an arbitrary numeric
    // prefix cannot be treated as an HTTP status line.
    let version = parts.next()?;
    if !version.starts_with("HTTP/") {
        return None;
    }
    parts.next()?.parse().ok()
}

// Compute a fresh relative timeout from the request's absolute deadline. An
// elapsed or zero duration is already a request failure.
fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| io_ingest("provider request failed", Some("request")))
}

// Reset the socket timeout for each attempted read using only the time left. EINTR
// retries with that same deadline rather than extending it.
fn read_deadline(stream: &mut TcpStream, buf: &mut [u8], deadline: Instant) -> Result<usize> {
    stream
        .set_read_timeout(Some(remaining(deadline)?))
        .map_err(|_| io_ingest("provider request failed", Some("request")))?;
    // Pass successful read lengths, including EOF, to the framing caller. It
    // decides whether that count completes or truncates the requested response.
    match stream.read(buf) {
        Ok(count) => Ok(count),
        // An interrupted read consumed no bytes. Retry with the current output
        // buffer and recompute the timeout from the unchanged deadline.
        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
            read_deadline(stream, buf, deadline)
        }
        Err(_) => Err(io_ingest("provider request failed", Some("request"))),
    }
}

// Send a request across as many partial writes as needed within one deadline.
// Retry an interrupted call on the same unsent slice.
fn write_all_deadline(stream: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        stream
            .set_write_timeout(Some(remaining(deadline)?))
            .map_err(|_| io_ingest("provider request failed", Some("request")))?;
        match stream.write(bytes) {
            // Successful zero-byte writes make no progress. Fail immediately
            // rather than spinning on a request that never leaves the buffer.
            Ok(0) => return Err(io_ingest("provider request failed", Some("request"))),
            // Advance only by the acknowledged write count. The next iteration
            // recomputes the timeout before retrying the remaining suffix.
            Ok(count) => bytes = &bytes[count..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(io_ingest("provider request failed", Some("request"))),
        }
    }
    Ok(())
}

// Check the optional model echo, vector dimension and number of inputs. Return
// vectors in request order after all rows pass.
pub(super) fn parse_embeddings(
    body: &str,
    model: &str,
    dimensions: u32,
    count: usize,
) -> Result<Vec<Vec<f32>>> {
    // Run the provider-number compatibility pass before strict JSON decoding. This
    // admits numeric spellings the vector importer can represent while keeping one
    // product parser for the resulting object.
    let body = normalize_provider_numbers(body)?;
    let value = match parse(&body) {
        Ok(value) => value,
        // Report non-finite JSON numbers as a vector error. Other parse failures
        // are schema errors and omit raw provider response bytes.
        Err(error)
            if error.code() == ErrorCode::InvalidJsonlSchema
                && error.message() == "JSON number is not finite" =>
        {
            return Err(vector_length("embedding values must be finite numbers"));
        }
        Err(_) => return Err(schema("provider response is not a valid embeddings object")),
    };
    // Validate the envelope before inspecting model or response rows. A valid JSON
    // scalar is still an invalid embeddings response.
    let obj = value
        .as_object()
        .ok_or_else(|| schema("provider response is not a valid embeddings object"))?;
    // The provider may omit its model echo. If it supplies one, require a string
    // equal to the requested model and distinguish that mismatch from malformed
    // JSON.
    if let Some(returned) = obj.get("model") {
        match returned.as_str() {
            Some(returned) if returned == model => {}
            Some(_) => {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "provider model does not match the request",
                )
                // Use the model field as the error location without copying the
                // provider-supplied model spelling into the error.
                .with_detail("field", "model")
                .ingestion_boundary());
            }
            None => return Err(schema("provider response model is not a string")),
        }
    }
    // Require a data array before accepting any individual row. It must describe a
    // complete response for the current request.
    let data = obj
        .get("data")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| schema("provider response is missing data"))?;
    // Reject missing and surplus rows before creating the reorder slots. Matching
    // length alone is insufficient until every source index has been checked.
    if data.len() != count {
        return Err(vector_length(
            "provider response count does not match the request",
        ));
    }
    // The provider identifies inputs by index and may reorder its rows. Match
    // by index before pairing vectors with chunk IDs or writing any artifact.
    // Size the reorder slots by the request count already known to the caller. The
    // provider's highest index cannot cause this array to grow.
    let mut slots = vec![None; count];
    // Each row supplies both a source index and its vector. Keep that association
    // through validation, regardless of the provider's row order.
    for item in data {
        let item = item
            .as_object()
            .ok_or_else(|| schema("provider response data item is not an object"))?;
        let index_value = item
            .get("index")
            .ok_or_else(|| schema("provider response data item is missing index"))?;
        // Decode a numeric integer index before using it as a slot address. The
        // following bounds and duplicate checks still apply to every converted
        // index.
        let index = json_index(index_value)?;
        let embedding = item
            .get("embedding")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| schema("provider response data item is missing embedding"))?;
        // Reject indices outside the original request or already occupied.
        // Accepting the latter would overwrite a vector assigned to another
        // response row.
        if index >= count || slots[index].is_some() {
            return Err(vector_length(
                "provider response indexes do not match the request",
            ));
        }
        // Require each embedding to match the requested dimension before
        // allocating its typed vector. Count and dimensions are independent
        // checks.
        if embedding.len() != dimensions as usize {
            return Err(vector_length("embedding length does not match dimensions"));
        }
        let mut vector = Vec::with_capacity(embedding.len());
        for value in embedding {
            let Some(number) = value.as_f64() else {
                return Err(vector_length("embedding values must be numbers"));
            };
            // Validate the coordinate in the precision the importer stores. Even a
            // finite binary64 coordinate can become infinite after narrowing.
            let narrowed = number as f32;
            if !narrowed.is_finite() {
                return Err(vector_length("embedding values must be finite numbers"));
            }
            vector.push(narrowed);
        }
        // Install a vector in its original request slot only after all coordinates
        // pass. No partially converted row escapes this function on error.
        slots[index] = Some(vector);
    }
    // Collect by slot order, which is original request order. A remaining hole is
    // reported with the same index mismatch used for duplicates and out-of-range
    // indices.
    slots
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| vector_length("provider response indexes do not match the request"))
        })
        .collect()
}

// Scan JSON text for raw number tokens before the product parser. Keep quoted
// strings intact and leave final syntax and schema recognition to the normal
// parser.
fn normalize_provider_numbers(body: &str) -> Result<String> {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    // The scan owns the new response text. Keeping the source immutable lets token
    // spans borrow original spelling for the parse-or-preserve decision.
    let mut index = 0;
    while index < bytes.len() {
        // Copy strings as whole spans. Digits inside a model name, field name or
        // escape sequence are data and must never be interpreted as numeric
        // tokens.
        if bytes[index] == b'"' {
            let start = index;
            index += 1;
            while index < bytes.len() {
                // Skip an escaped byte before testing for a closing quote. For a
                // Unicode escape, skip the marker and four expected hex digits
                // while still leaving validation of those digits to the parser.
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
        // For a candidate numeric token, isolate its contiguous numeric spelling
        // and normalize only that span. Preserve all neighboring punctuation so
        // the strict decoder still sees the envelope.
        if is_number_start(bytes, index) {
            let start = index;
            index = number_end(bytes, index);
            out.push_str(&rounded_number(&body[start..index])?);
            continue;
        }
        // Copy the next UTF-8 scalar as a byte span outside number and string
        // branches. Advancing one byte at a time could split a multibyte
        // character.
        let next = index + utf8_len(bytes[index]);
        out.push_str(&body[index..next.min(bytes.len())]);
        index = next.min(bytes.len());
    }
    Ok(out)
}

// Start a candidate on a digit or on a minus immediately followed by a digit.
// Other minus bytes are copied through for syntax validation.
fn is_number_start(bytes: &[u8], index: usize) -> bool {
    match bytes[index] {
        b'-' => bytes
            .get(index + 1)
            .is_some_and(|byte| byte.is_ascii_digit()),
        b'0'..=b'9' => true,
        _ => false,
    }
}

// Find the end of a possible number by scanning sign, digits, fraction and
// exponent. The range finder locates tokens but does not decide whether their
// grammar is valid.
fn number_end(bytes: &[u8], mut index: usize) -> usize {
    if bytes.get(index) == Some(&b'-') {
        index += 1;
    }
    while bytes.get(index).is_some_and(|byte| byte.is_ascii_digit()) {
        index += 1;
    }
    // Include a decimal point and its following digits even when the input is
    // incomplete. The parser can then reject the original malformed token.
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        while bytes.get(index).is_some_and(|byte| byte.is_ascii_digit()) {
            index += 1;
        }
    }
    // Keep the exponent and optional sign with the mantissa. Normalizing either
    // piece separately could change the numeric value.
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

// Probe the strict decoder with the number as one array element. Preserve any
// spelling that already passes, including spellings that round-trip without
// conversion.
fn rounded_number(token: &str) -> Result<String> {
    // Brackets make this one token a complete JSON value for the shared parser.
    // They are only a probe and never enter the normalized response.
    let probe = format!("[{token}]");
    match parse(&probe) {
        Ok(_) => Ok(token.to_owned()),
        // Allow conversion only for non-finite or non-exact integer diagnostics.
        // The fallback uses binary64 parsing and then checks the narrower binary32
        // value.
        Err(error)
            if error.message() == "JSON number is not finite"
                || error.message()
                    == "integer cannot be represented exactly as binary64; encode it as a string" =>
        {
            match token.parse::<f64>() {
                // Convert finite values to the target precision before emitting a
                // replacement. The canonical artifact formatter chooses a spelling
                // that import accepts.
                Ok(value) if value.is_finite() => {
                    let narrowed = value as f32;
                    if !narrowed.is_finite() {
                        return Err(vector_length("embedding values must be finite numbers"));
                    }
                    // Emit the artifact's accepted spelling after narrowing. The
                    // final JSON parser then applies the same numeric rules as a
                    // direct artifact import.
                    Ok(format_f32(narrowed))
                }
                _ => Err(vector_length("embedding values must be finite numbers")),
            }
        }
        // Leave other rejected tokens intact. Rewriting malformed syntax could
        // hide a provider error from the final parse.
        Err(_) => Ok(token.to_owned()),
    }
}

// The body is already a Rust str. Determine the size of the current character from
// its first byte; this helper is not a general UTF-8 validator.
fn utf8_len(byte: u8) -> usize {
    match byte {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

// Convert the provider row's numeric index to a native offset. Keep malformed
// indices separate from offsets that cannot address a request slot.
fn json_index(value: &JsonValue) -> Result<usize> {
    let Some(number) = value.as_f64() else {
        return Err(schema("provider response index is not an integer"));
    };
    // Require a finite, nonnegative, integral number before converting. A
    // fractional source position must not silently round down to another row.
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
        return Err(schema("provider response index is not an integer"));
    }
    // Compare with the floating-point representation of the native maximum before
    // casting. The subsequent check against the actual request count remains
    // decisive near that rounded limit.
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
