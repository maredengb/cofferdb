// Complete-result workload driver. This executable is an optional maintainer
// tool, not an application API. Compilation does not run an evaluation.
use cofferdb::{
    CollectionHandle, Database, DatabaseOpenOptions, JsonObject, JsonValue, MetadataBoostConfig,
    QueryEmbeddingInput, ResourceProfile, SearchWeights,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};
mod writer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn obj<const N: usize>(fields: [(&str, JsonValue); N]) -> JsonValue {
    JsonValue::Object(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect::<JsonObject>(),
    )
}
fn string(value: impl Into<String>) -> JsonValue {
    JsonValue::String(value.into())
}
fn count(value: usize) -> JsonValue {
    string(value.to_string())
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[derive(Debug)]
struct EvidenceFailure {
    call_error: Option<cofferdb::Error>,
    known_kind: &'static str,
    known: JsonValue,
    tool_error: Box<dyn std::error::Error>,
}
impl std::fmt::Display for EvidenceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("call completed but evidence retention failed; known outcome retained")
    }
}
impl std::error::Error for EvidenceFailure {}
#[derive(Debug)]
struct CleanupFailure {
    body: Box<dyn std::error::Error>,
    cleanup: Box<dyn std::error::Error>,
}
impl std::fmt::Display for CleanupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("workload and database close both failed")
    }
}
impl std::error::Error for CleanupFailure {}
fn describe_error(error: &(dyn std::error::Error + 'static)) -> JsonValue {
    if let Some(value) = error.downcast_ref::<cofferdb::Error>() {
        return value.to_json();
    }
    if let Some(value) = error.downcast_ref::<EvidenceFailure>() {
        return obj([
            ("type", string("evidence_failure")),
            ("call_returned", JsonValue::Bool(value.call_error.is_none())),
            (
                "call_error",
                value
                    .call_error
                    .as_ref()
                    .map_or(JsonValue::Null, cofferdb::Error::to_json),
            ),
            ("known_kind", string(value.known_kind)),
            ("known", value.known.clone()),
            ("tool_error", describe_error(value.tool_error.as_ref())),
        ]);
    }
    if let Some(value) = error.downcast_ref::<CleanupFailure>() {
        return obj([
            ("type", string("body_and_cleanup_failure")),
            ("body", describe_error(value.body.as_ref())),
            ("cleanup", describe_error(value.cleanup.as_ref())),
        ]);
    }
    obj([("message", string(error.to_string()))])
}
fn finish(body: Result<()>, cleanup: Result<()>) -> Result<()> {
    match (body, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(body), Err(cleanup)) => Err(Box::new(CleanupFailure { body, cleanup })),
    }
}

#[cfg(target_os = "macos")]
fn observe_native() -> JsonValue {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    struct Malloc {
        blocks: u32,
        used: usize,
        maximum: usize,
        allocated: usize,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Task {
        virtual_size: u64,
        resident_size: u64,
        resident_max: u64,
        user_seconds: i32,
        user_microseconds: i32,
        system_seconds: i32,
        system_microseconds: i32,
        policy: i32,
        suspend_count: i32,
    }
    extern "C" {
        fn malloc_zone_statistics(zone: *mut c_void, result: *mut Malloc);
        fn task_info(task: u32, flavor: i32, info: *mut c_void, count: *mut u32) -> i32;
        static mach_task_self_: u32;
    }
    let mut malloc = Malloc::default();
    let mut task = Task::default();
    let mut words = (std::mem::size_of::<Task>() / 4) as u32;
    // C layouts and flavor match the native macOS APIs, with no borrowed pointers retained.
    unsafe {
        malloc_zone_statistics(std::ptr::null_mut(), &mut malloc);
        let status = task_info(
            mach_task_self_,
            20,
            (&mut task as *mut Task).cast(),
            &mut words,
        );
        obj([
            ("method", string("darwin_all_zone_malloc_and_mach_self")),
            ("allocator_used_bytes", count(malloc.used)),
            ("allocator_reserved_bytes", count(malloc.allocated)),
            ("allocator_blocks", count(malloc.blocks as usize)),
            ("allocator_high_water_raw_bytes", count(malloc.maximum)),
            (
                "allocator_high_water_bytes",
                if malloc.maximum >= malloc.used {
                    count(malloc.maximum)
                } else {
                    JsonValue::Null
                },
            ),
            ("mach_status", JsonValue::Number(status as f64)),
            (
                "rss_bytes",
                if status == 0 {
                    string(task.resident_size.to_string())
                } else {
                    JsonValue::Null
                },
            ),
            (
                "rss_high_water_raw_bytes",
                if status == 0 {
                    string(task.resident_max.to_string())
                } else {
                    JsonValue::Null
                },
            ),
            (
                "rss_high_water_bytes",
                if status == 0 && task.resident_max >= task.resident_size {
                    string(task.resident_max.to_string())
                } else {
                    JsonValue::Null
                },
            ),
        ])
    }
}
#[cfg(target_os = "linux")]
fn observe_native() -> JsonValue {
    use std::ffi::{c_int, c_void};
    extern "C" {
        fn open_memstream(buffer: *mut *mut u8, size: *mut usize) -> *mut c_void;
        fn malloc_info(options: c_int, stream: *mut c_void) -> c_int;
        fn fclose(stream: *mut c_void) -> c_int;
        fn free(value: *mut c_void);
    }
    let status = fs::read_to_string("/proc/self/status").unwrap_or_else(|e| e.to_string());
    let mut buffer = std::ptr::null_mut();
    let mut size = 0usize;
    // The bounded complete XML is retained for post-run parsing, not subtracted
    // from RSS or Python measurements to invent exclusive engine allocations.
    let xml = unsafe {
        let stream = open_memstream(&mut buffer, &mut size);
        if stream.is_null() {
            None
        } else {
            let result = malloc_info(0, stream);
            let closed = fclose(stream);
            let value = if result == 0
                && closed == 0
                && !buffer.is_null()
                && size <= 4 * 1024 * 1024
            {
                Some(String::from_utf8_lossy(std::slice::from_raw_parts(buffer, size)).into_owned())
            } else {
                None
            };
            free(buffer.cast());
            value
        }
    };
    obj([
        ("method", string("glibc_malloc_info_and_proc_self_status")),
        ("allocator_root_xml", xml.map_or(JsonValue::Null, string)),
        ("proc_status_raw", string(status)),
    ])
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn observe_native() -> JsonValue {
    obj([("method", string("unavailable"))])
}
fn functional_control() -> bool {
    std::env::var("COFFERDB_TOOL_FUNCTIONAL_CONTROL_ONLY").as_deref() == Ok("1")
}
fn elapsed_ns(start: Instant) -> u128 {
    if functional_control() {
        0
    } else {
        start.elapsed().as_nanos()
    }
}
fn observe() -> JsonValue {
    if functional_control() {
        return obj([
            ("disabled", JsonValue::Bool(true)),
            (
                "reason",
                string("single-record API control; no memory or performance measurement"),
            ),
        ]);
    }
    let fds = fs::read_dir(if cfg!(target_os = "macos") {
        "/dev/fd"
    } else {
        "/proc/self/fd"
    })
    .map(|entries| count(entries.count()))
    .unwrap_or(JsonValue::Null);
    obj([("native_process_and_rss",observe_native()),("open_fd_snapshot_count",fds),
         ("scope",string("overlapping process counters, not Rust-engine-exclusive; snapshots miss intervening peaks"))])
}

struct Journal {
    root: PathBuf,
    events: File,
    sequence: usize,
}
impl Journal {
    fn new(path: &Path, identity: JsonValue) -> Result<Self> {
        fs::create_dir(path)?;
        fs::create_dir(path.join("payloads"))?;
        let events = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join("events.jsonl"))?;
        let mut journal = Self {
            root: path.to_owned(),
            events,
            sequence: 0,
        };
        journal.append(obj([("event", string("identity")), ("identity", identity)]))?;
        Ok(journal)
    }
    fn append(&mut self, value: JsonValue) -> Result<()> {
        writeln!(self.events, "{}", value.try_to_compact_string()?)?;
        self.events.flush()?;
        self.events.sync_all()?;
        Ok(())
    }
    fn payload(&self, sequence: usize, value: &JsonValue) -> Result<JsonValue> {
        let relative = format!("payloads/{sequence:08}.json");
        let encoded = value.try_to_compact_string()?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join(&relative))?;
        file.write_all(encoded.as_bytes())?;
        file.sync_all()?;
        // SHA256 is independently indexed by the supervising Python process
        // after this worker exits; these are untouched complete original files.
        Ok(obj([
            ("relative_path", string(relative)),
            ("bytes", count(encoded.len())),
        ]))
    }
    fn measure<T>(
        &mut self,
        label: &str,
        call: impl FnOnce() -> cofferdb::Result<T>,
        project: impl FnOnce(&T) -> JsonValue,
    ) -> Result<()> {
        self.measure_observed(label, call, project, None)
    }
    fn measure_observed<T>(
        &mut self,
        label: &str,
        call: impl FnOnce() -> cofferdb::Result<T>,
        project: impl FnOnce(&T) -> JsonValue,
        progress: Option<&cofferdb::binding::WriteObservation>,
    ) -> Result<()> {
        self.sequence += 1;
        let seq = self.sequence;
        self.append(obj([
            ("event", string("started")),
            ("sequence", count(seq)),
            ("label", string(label)),
            ("before", observe()),
        ]))?;
        let start = Instant::now();
        let result = call();
        let elapsed = elapsed_ns(start);
        let after = observe();
        // Public Rust output is already an owned typed value. JSON projection
        // and persistence are reporting work outside the API interval.
        let mut value = match &result {
            Ok(v) => obj([("ok", project(v))]),
            Err(e) => obj([("error", e.to_json())]),
        };
        if let Some(progress) = progress {
            value.as_object_mut().expect("report object").insert(
                "operation_outcome".into(),
                writer::progress_value(progress.write_outcome()),
            );
        }
        let error = result.err();
        let payload = match self.payload(seq, &value) {
            Ok(payload) => payload,
            Err(tool_error) => {
                return Err(Box::new(EvidenceFailure {
                    call_error: error,
                    known_kind: "complete_json_value",
                    known: value,
                    tool_error,
                }))
            }
        };
        drop(value);
        if let Err(tool_error) = self.append(obj([
            ("event", string("completed")),
            ("sequence", count(seq)),
            ("label", string(label)),
            ("elapsed_ns", string(elapsed.to_string())),
            ("after_call", after),
            ("after_release", observe()),
            ("payload", payload.clone()),
            (
                "outcome",
                string(if error.is_some() { "error" } else { "ok" }),
            ),
        ])) {
            return Err(Box::new(EvidenceFailure {
                call_error: error,
                known_kind: "complete_payload_file",
                known: payload,
                tool_error,
            }));
        }
        if let Some(error) = error {
            return Err(Box::new(error));
        }
        Ok(())
    }
}
fn field<'a>(value: &'a JsonValue, key: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .ok_or_else(|| invalid(&format!("missing field: {key}")).into())
}
fn text<'a>(value: &'a JsonValue, key: &str) -> Result<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| invalid("string field required").into())
}
fn number(value: &JsonValue, key: &str) -> Result<f64> {
    field(value, key)?
        .as_f64()
        .ok_or_else(|| invalid("number field required").into())
}
fn exact_index(value: &JsonValue) -> cofferdb::Result<usize> {
    let error = || {
        cofferdb::Error::new(
            cofferdb::ErrorCode::InvalidSearchParams,
            "workload integer must be exact and representable",
        )
    };
    let number = value.as_f64().ok_or_else(error)?;
    if !number.is_finite()
        || number.fract() != 0.0
        || !(0.0..=9_007_199_254_740_991.0).contains(&number)
        || number > usize::MAX as f64
    {
        return Err(error());
    }
    Ok(number as usize)
}
fn search(
    collection: &CollectionHandle,
    request: &JsonValue,
    profile: ResourceProfile,
) -> cofferdb::Result<cofferdb::SearchResults> {
    let mut query = collection.search().resource_profile(profile);
    if let Some(text) = request.get("text").and_then(JsonValue::as_str) {
        query = query.text(text);
    }
    if let Some(value) = request.get("embedding").filter(|v| !v.is_null()) {
        let fixture_error = || {
            cofferdb::Error::new(
                cofferdb::ErrorCode::InvalidSearchParams,
                "invalid workload embedding fixture",
            )
        };
        let items = value
            .get("vector")
            .and_then(JsonValue::as_array)
            .ok_or_else(fixture_error)?;
        let dim = exact_index(value.get("dim").ok_or_else(fixture_error)?)?;
        let vector = items
            .iter()
            .map(|item| item.as_f64().map(|v| v as f32).ok_or_else(fixture_error))
            .collect::<cofferdb::Result<Vec<_>>>()?;
        let model_name = match value.get("model_name") {
            None | Some(JsonValue::Null) => None,
            Some(JsonValue::String(value)) => Some(value.clone()),
            _ => return Err(fixture_error()),
        };
        let parsed = QueryEmbeddingInput {
            dim,
            model_name,
            vector,
        };
        query = query.query_embedding(parsed);
    }
    if let Some(filter) = request.get("filter").filter(|v| !v.is_null()) {
        query = query.filter(filter.clone());
    }
    if let Some(weights) = request.get("weights").filter(|v| !v.is_null()) {
        let fixture_error = || {
            cofferdb::Error::new(
                cofferdb::ErrorCode::InvalidSearchParams,
                "invalid workload weights fixture",
            )
        };
        let values = ["bm25", "vector", "metadata"]
            .map(|key| number(weights, key).map_err(|_| fixture_error()));
        let [bm25, vector, metadata] = values;
        query = query.weights(SearchWeights::new(bm25?, vector?, metadata?)?);
    }
    if let Some(boost) = request.get("metadata_boost").filter(|v| !v.is_null()) {
        query = query.metadata_boost(MetadataBoostConfig::parse(boost.clone())?);
    }
    if let Some(value) = request.get("top_k") {
        query = query.top_k(exact_index(value)?);
    }
    query.execute()
}
fn run(arguments: &[String]) -> Result<()> {
    if arguments.len() != 7 {
        return Err(
            invalid("expected database collection queries output profile warmups repeats").into(),
        );
    }
    let profile =
        ResourceProfile::parse(&arguments[4]).ok_or_else(|| invalid("unsupported profile"))?;
    let warmups: usize = arguments[5].parse()?;
    let repeats: usize = arguments[6].parse()?;
    if warmups == 0 || repeats == 0 {
        return Err(invalid("positive repetition counts required").into());
    }
    let queries = cofferdb::parse_json(&fs::read_to_string(&arguments[2])?)?;
    let queries = queries
        .as_array()
        .ok_or_else(|| invalid("query array required"))?;
    let selected: Vec<_> = queries
        .iter()
        .filter(|q| q.get("profile").and_then(JsonValue::as_str) == Some(&arguments[4]))
        .collect();
    if selected.is_empty() {
        return Err(invalid("no declared queries for selected profile").into());
    }
    let identity=obj([("reader_tool_version",string(env!("CARGO_PKG_VERSION"))),
        ("engine_version",string(cofferdb::binding::engine_version())),
        ("declared_build_source_input_sha256",option_env!("COFFERDB_CORE_SOURCE_INPUT_SHA256").map_or(JsonValue::Null,string)),
        ("profile",string(&arguments[4])),("database",string(&arguments[0])),
        ("functional_control_only",JsonValue::Bool(functional_control())),
        ("timing_scope",string("Rust request conversion plus execute returning owned typed result; JSON report projection/IO outside interval")),
        ("cache_scope",string("OS page cache uncontrolled; new reader then same-reader warm passes"))]);
    let mut journal = Journal::new(Path::new(&arguments[3]), identity)?;
    for request in selected {
        let database = Database::open_with_options(
            &arguments[0],
            DatabaseOpenOptions {
                read_only: true,
                resource_profile: Some(profile),
                ..Default::default()
            },
        )?;
        let body = (|| {
            let collection = database.collection(&arguments[1])?;
            let id = text(request, "id")?;
            journal.measure(
                &format!("new-handle-{id}"),
                || search(&collection, request, profile),
                |v| v.to_json(),
            )?;
            for n in 0..warmups {
                journal.measure(
                    &format!("warmup-{id}-{n}"),
                    || search(&collection, request, profile),
                    |v| v.to_json(),
                )?;
            }
            for n in 0..repeats {
                journal.measure(
                    &format!("warm-{id}-{n}"),
                    || search(&collection, request, profile),
                    |v| v.to_json(),
                )?;
            }
            Ok(())
        })();
        // Consume close even after a body failure, retaining both errors.
        let cleanup = database
            .close()
            .map_err(|error| Box::new(error) as Box<dyn std::error::Error>);
        finish(body, cleanup)?;
        journal.append(obj([
            ("event", string("reader_closed")),
            ("observation", observe()),
        ]))?;
    }
    journal.append(obj([
        ("event", string("case_complete")),
        ("observation", observe()),
    ]))?;
    Ok(())
}
fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let result = if arguments.first().map(String::as_str) == Some("write") {
        writer::run(&arguments[1..])
    } else if arguments.first().map(String::as_str) == Some("content") {
        writer::content(&arguments[1..])
    } else {
        run(&arguments)
    };
    if let Err(error) = result {
        // A supervisor must preserve stderr even if the output filesystem fails.
        match describe_error(error.as_ref()).try_to_compact_string() {
            Ok(value) => eprintln!("{value}"),
            Err(serialization) => {
                eprintln!("workload failure: {error}; failure serialization: {serialization}")
            }
        }
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_payload_and_original_error_are_saved_before_propagation() {
        let root = std::env::temp_dir().join(format!(
            "cofferdb-reader-journal-no-engine-{}",
            std::process::id()
        ));
        let mut journal =
            Journal::new(&root, obj([("kind", string("NO_ENGINE_CONTROL"))])).unwrap();
        let input = obj([
            ("text", string("Jade 中\0\n")),
            ("span_end", string("18446744073709551615")),
            ("score", JsonValue::Number(0.123456789)),
        ]);
        journal
            .measure("complete", || Ok(input.clone()), Clone::clone)
            .unwrap();
        let error = cofferdb::Error::new(cofferdb::ErrorCode::DuplicateExternalId, "controlled")
            .with_detail("field", "fixture");
        assert!(journal
            .measure::<JsonValue>("error", || Err(error), Clone::clone)
            .is_err());
        drop(journal);
        let events = fs::read_to_string(root.join("events.jsonl")).unwrap();
        assert!(events.contains("\"outcome\":\"error\""));
        let value =
            cofferdb::parse_json(&fs::read_to_string(root.join("payloads/00000001.json")).unwrap())
                .unwrap();
        assert_eq!(value.get("ok"), Some(&input));
        let value =
            cofferdb::parse_json(&fs::read_to_string(root.join("payloads/00000002.json")).unwrap())
                .unwrap();
        assert_eq!(
            value.get("error").unwrap().get("code").unwrap().as_str(),
            Some("duplicate_external_id")
        );
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn native_allocator_observation_sees_and_releases_a_touched_buffer() {
        let used = || {
            observe_native()
                .get("allocator_used_bytes")
                .unwrap()
                .as_str()
                .unwrap()
                .parse::<usize>()
                .unwrap()
        };
        let before = used();
        let mut buffer = vec![0x31u8; 2 * 1024 * 1024];
        for value in &mut buffer {
            *value ^= 1;
        }
        std::hint::black_box(&buffer);
        let during = used();
        assert!(during >= before + 2 * 1024 * 1024 - 64 * 1024);
        drop(buffer);
        let after = used();
        assert!(after + 2 * 1024 * 1024 - 64 * 1024 <= during);
    }
    #[test]
    fn primitive_query_fields_do_not_silently_invent_numbers() {
        let value = cofferdb::parse_json("{\"top_k\":10,\"id\":\"a\",\"enabled\":true}").unwrap();
        assert_eq!(number(&value, "top_k").unwrap(), 10.0);
        assert!(number(&value, "enabled").is_err());
        assert!(text(&value, "missing").is_err());
    }
    #[test]
    fn payload_io_failure_retains_complete_returned_value_without_replay() {
        let root = std::env::temp_dir().join(format!(
            "cofferdb-reader-known-value-no-engine-{}",
            std::process::id()
        ));
        let mut journal =
            Journal::new(&root, obj([("kind", string("NO_ENGINE_CONTROL"))])).unwrap();
        fs::rename(root.join("payloads"), root.join("preserved-empty-payloads")).unwrap();
        let mut calls = 0;
        let value = obj([("committed", count(3)), ("body", string("Jade 中\0"))]);
        let error = journal
            .measure(
                "successful-call",
                || {
                    calls += 1;
                    Ok(value.clone())
                },
                Clone::clone,
            )
            .unwrap_err();
        assert_eq!(calls, 1);
        let description = describe_error(error.as_ref());
        assert_eq!(
            description.get("call_returned"),
            Some(&JsonValue::Bool(true))
        );
        assert_eq!(description.get("known").unwrap().get("ok"), Some(&value));
    }
    #[test]
    fn payload_io_failure_does_not_hide_the_original_core_error() {
        let root = std::env::temp_dir().join(format!(
            "cofferdb-reader-error-no-engine-{}",
            std::process::id()
        ));
        let mut journal =
            Journal::new(&root, obj([("kind", string("NO_ENGINE_CONTROL"))])).unwrap();
        fs::rename(root.join("payloads"), root.join("preserved-empty-payloads")).unwrap();
        let error = cofferdb::Error::new(
            cofferdb::ErrorCode::DuplicateExternalId,
            "controlled core error",
        )
        .with_detail("field", "fixture");
        let failure = journal
            .measure::<JsonValue>("core-failure", || Err(error), Clone::clone)
            .unwrap_err();
        let description = describe_error(failure.as_ref());
        assert_eq!(
            description.get("call_returned"),
            Some(&JsonValue::Bool(false))
        );
        assert_eq!(
            description
                .get("call_error")
                .unwrap()
                .get("code")
                .unwrap()
                .as_str(),
            Some("duplicate_external_id")
        );
        assert_eq!(
            description.get("known").unwrap().get("error"),
            description.get("call_error")
        );
    }
    #[test]
    fn explicit_cleanup_retains_both_body_and_close_failures() {
        let body: Result<()> = Err(Box::new(cofferdb::Error::new(
            cofferdb::ErrorCode::DuplicateExternalId,
            "controlled body",
        )));
        let cleanup: Result<()> = Err(Box::new(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "controlled close",
        )));
        let error = finish(body, cleanup).unwrap_err();
        let value = describe_error(error.as_ref());
        assert_eq!(
            value.get("body").unwrap().get("code").unwrap().as_str(),
            Some("duplicate_external_id")
        );
        assert_eq!(
            value
                .get("cleanup")
                .unwrap()
                .get("message")
                .unwrap()
                .as_str(),
            Some("controlled close")
        );
    }
}
