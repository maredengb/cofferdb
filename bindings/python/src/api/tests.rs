use super::*;
use cofferdb::{CollectionConfig, DistanceMetric, TokenizerConfig, VectorIndexConfig};
use pyo3::ffi;
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Allocator faults run in a fresh, single-test interpreter, never alongside
/// unrelated Python tests. The original allocator is restored before inspection.
fn isolated(name: &str, run: impl FnOnce()) {
    const MARKER: &str = "COFFER_HOST_FAILURE_TEST";
    if std::env::var(MARKER).ok().as_deref() == Some(name) {
        run();
    } else {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture", "--test-threads=1"])
            .env(MARKER, name)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

struct Fault {
    old: ffi::PyMemAllocatorEx,
    armed: AtomicBool,
    hits: AtomicUsize,
    size: usize,
}

extern "C" fn malloc(context: *mut c_void, size: usize) -> *mut c_void {
    // SAFETY: the stack context outlives the installed allocator; the previous
    // allocator's callbacks and context are retained unchanged. Shared mutation
    // is atomic, including if CPython invokes this wrapper from another thread.
    let fault = unsafe { &*context.cast::<Fault>() };
    if size == fault.size && fault.armed.swap(false, Ordering::SeqCst) {
        fault.hits.fetch_add(1, Ordering::SeqCst);
        std::ptr::null_mut()
    } else {
        (fault.old.malloc.unwrap())(fault.old.ctx, size)
    }
}
extern "C" fn calloc(context: *mut c_void, length: usize, size: usize) -> *mut c_void {
    let fault = unsafe { &*context.cast::<Fault>() };
    (fault.old.calloc.unwrap())(fault.old.ctx, length, size)
}
extern "C" fn realloc(context: *mut c_void, pointer: *mut c_void, size: usize) -> *mut c_void {
    let fault = unsafe { &*context.cast::<Fault>() };
    (fault.old.realloc.unwrap())(fault.old.ctx, pointer, size)
}
extern "C" fn free(context: *mut c_void, pointer: *mut c_void) {
    let fault = unsafe { &*context.cast::<Fault>() };
    (fault.old.free.unwrap())(fault.old.ctx, pointer)
}
struct Restore(ffi::PyMemAllocatorEx);
impl Drop for Restore {
    fn drop(&mut self) {
        // SAFETY: the GIL is still held; this restores the exact original domain.
        unsafe { ffi::PyMem_SetAllocator(ffi::PyMemAllocatorDomain::PYMEM_DOMAIN_OBJ, &mut self.0) }
    }
}

fn allocation_fault<T>(
    py: Python<'_>,
    size: usize,
    run: impl FnOnce() -> T,
) -> (std::thread::Result<T>, usize) {
    let _ = py;
    // SAFETY: this helper is called only in an isolated initialized interpreter
    // under the GIL. Restore is dropped before its stack context, also on unwind.
    unsafe {
        ffi::PyGC_Collect();
    }
    let mut previous = std::mem::MaybeUninit::uninit();
    unsafe {
        ffi::PyMem_GetAllocator(
            ffi::PyMemAllocatorDomain::PYMEM_DOMAIN_OBJ,
            previous.as_mut_ptr(),
        );
    }
    let previous = unsafe { previous.assume_init() };
    let fault = Fault {
        old: previous,
        armed: AtomicBool::new(true),
        hits: AtomicUsize::new(0),
        size,
    };
    let mut allocator = ffi::PyMemAllocatorEx {
        ctx: (&fault as *const Fault).cast_mut().cast(),
        malloc: Some(malloc),
        calloc: Some(calloc),
        realloc: Some(realloc),
        free: Some(free),
    };
    let restore = Restore(previous);
    unsafe {
        ffi::PyMem_SetAllocator(ffi::PyMemAllocatorDomain::PYMEM_DOMAIN_OBJ, &mut allocator);
    }
    let result = panic_guard::catch(run);
    drop(restore);
    (result, fault.hits.load(Ordering::SeqCst))
}

fn config(name: String) -> CollectionConfig {
    CollectionConfig {
        name,
        tokenizer: TokenizerConfig::default_en(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    }
}

fn database(name: &str) -> (std::path::PathBuf, Arc<owner::Owner>) {
    let path =
        std::env::temp_dir().join(format!("coffer-host-{name}-{}.coffer", std::process::id()));
    let db = cofferdb::Database::create(&path, DatabaseConfig::default()).unwrap();
    (path, owner::Owner::new(db))
}

fn collection(py: Python<'_>, owner: &Arc<owner::Owner>) -> Py<Collection> {
    let collection = owner
        .enter(false)
        .unwrap()
        .run(|db| db.create_collection(config("x".into())))
        .unwrap();
    Py::new(
        py,
        Collection {
            owner: Arc::clone(owner),
            collection_id: collection.collection_id().into(),
            name: "x".into(),
        },
    )
    .unwrap()
}

#[cfg(feature = "native-test-support")]
fn import_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    std::fs::create_dir(dir).unwrap();
    let mut documents = String::new();
    let mut chunks = String::new();
    let mut embeddings = String::new();
    for index in 0..100 {
        documents.push_str(&format!(
            "{{\"external_id\":\"d-{index}\",\"metadata\":{{}}}}\n"
        ));
        for ordinal in 0..2 {
            let id = index * 2 + ordinal;
            chunks.push_str(&format!("{{\"document_external_id\":\"d-{index}\",\"chunk_external_id\":\"c-{id:03}\",\"ordinal\":{ordinal},\"text\":\"needle haystack\",\"metadata\":{{}}}}\n"));
            embeddings.push_str(&format!("{{\"chunk_external_id\":\"c-{id:03}\",\"model_name\":\"m\",\"dim\":2,\"vector\":[1,0]}}\n"));
        }
    }
    for (name, contents) in [
        ("documents", documents),
        ("chunks", chunks),
        ("embeddings", embeddings),
    ] {
        std::fs::write(dir.join(format!("{name}.jsonl")), contents).unwrap();
    }
    let manifest = dir.join("manifest.json");
    std::fs::write(&manifest, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","embedding_model":"m","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl"}"#).unwrap();
    manifest
}

#[cfg(feature = "native-test-support")]
#[test]
fn native_import_failure_preserves_python_progress_and_reopen() {
    isolated(
        "api::tests::native_import_failure_preserves_python_progress_and_reopen",
        || {
            Python::initialize();
            Python::attach(|py| {
                errors::register(&PyModule::new(py, "import_fault_types").unwrap()).unwrap();
                for after in [0, 1, 2] {
                    let dir = std::env::temp_dir().join(format!(
                        "coffer-native-import-{}-{after}",
                        std::process::id()
                    ));
                    let manifest = import_fixture(&dir);
                    let path = dir.join("data.coffer");
                    let native = cofferdb::Database::create(
                        &path,
                        DatabaseConfig {
                            resource_profile: ResourceProfile::Constrained,
                            ..DatabaseConfig::default()
                        },
                    )
                    .unwrap();
                    let owner = owner::Owner::new(native.clone());
                    let object = Py::new(
                        py,
                        Database {
                            owner: Arc::clone(&owner),
                        },
                    )
                    .unwrap();
                    let guard = owner
                        .enter(false)
                        .unwrap()
                        .run(|db| cofferdb::binding::ImportBatchFailure::arm(db, after))
                        .unwrap();
                    let before = std::fs::read(&path).unwrap();
                    let error = object
                        .bind(py)
                        .call_method1("import_jsonl", (manifest.to_str().unwrap(),))
                        .unwrap_err();
                    assert!(error.is_instance_of::<errors::CofferError>(py));
                    let value = error.value(py);
                    assert_eq!(
                        value.getattr("code").unwrap().extract::<String>().unwrap(),
                        "io_error"
                    );
                    assert_eq!(
                        value
                            .getattr("message")
                            .unwrap()
                            .extract::<String>()
                            .unwrap(),
                        "injected import batch failure"
                    );
                    assert_eq!(value.getattr("detail").unwrap().len().unwrap(), 0);
                    assert!(!value
                        .getattr("recoverable")
                        .unwrap()
                        .extract::<bool>()
                        .unwrap());
                    assert!(value.getattr("suggested_action").unwrap().is_none());
                    let outcome = value.getattr("operation_outcome").unwrap();
                    assert_eq!(
                        outcome
                            .get_item("state")
                            .unwrap()
                            .extract::<String>()
                            .unwrap(),
                        if after == 0 {
                            "not_committed"
                        } else {
                            "partially_committed"
                        }
                    );
                    assert_eq!(
                        outcome
                            .get_item("confirmed_committed_batches")
                            .unwrap()
                            .extract::<usize>()
                            .unwrap(),
                        after
                    );
                    assert_eq!(
                        outcome
                            .get_item("confirmed_committed_records")
                            .unwrap()
                            .extract::<usize>()
                            .unwrap(),
                        after * 250
                    );
                    let stats = native.stats().unwrap();
                    let expected = match after {
                        0 => (0, 0, 0),
                        1 => (100, 149, 0),
                        _ => (100, 200, 199),
                    };
                    assert_eq!(
                        (
                            stats.document_count,
                            stats.chunk_count,
                            stats.embedding_count
                        ),
                        expected
                    );
                    if after == 0 {
                        assert_eq!(before, std::fs::read(&path).unwrap());
                        assert!(!outcome.contains("last_confirmed_commit_id").unwrap());
                        assert!(!outcome
                            .contains("last_confirmed_commit_sequence_no")
                            .unwrap());
                    } else {
                        assert_eq!(
                            outcome
                                .get_item("last_confirmed_commit_id")
                                .unwrap()
                                .extract::<u64>()
                                .unwrap(),
                            stats.last_commit_id
                        );
                        assert_eq!(
                            outcome
                                .get_item("last_confirmed_commit_sequence_no")
                                .unwrap()
                                .extract::<u64>()
                                .unwrap(),
                            stats.last_commit_id + (after * 250) as u64
                        );
                    }
                    // The error exposes only core progress, never a synthetic input line.
                    assert_eq!(outcome.len().unwrap(), if after == 0 { 3 } else { 5 });
                    assert!(
                        owner.enter(false).is_ok(),
                        "failed import must release the operation gate"
                    );
                    drop(guard);
                    object.bind(py).call_method0("close").unwrap();
                    let reopened = cofferdb::Database::open(&path).unwrap();
                    let restored = reopened.stats().unwrap();
                    assert_eq!(
                        (
                            restored.document_count,
                            restored.chunk_count,
                            restored.embedding_count
                        ),
                        expected
                    );
                    assert_eq!(restored.last_commit_id, stats.last_commit_id);
                    if after > 0 {
                        let notes = reopened.collection("notes").unwrap();
                        assert!(notes.get_document_by_external_id("d-99").is_ok());
                        assert!(notes.get_chunk_by_external_id("c-148").is_ok());
                        assert_eq!(notes.get_chunk_by_external_id("c-149").is_ok(), after == 2);
                    }
                    reopened.close().unwrap();
                    drop(object);
                    drop(owner);
                    drop(native);
                    std::fs::remove_dir_all(dir).unwrap();
                }
            });
        },
    );
}

#[cfg(feature = "native-test-support")]
#[test]
fn native_import_failure_guard_is_scoped_and_instance_local() {
    isolated(
        "api::tests::native_import_failure_guard_is_scoped_and_instance_local",
        || {
            let dir =
                std::env::temp_dir().join(format!("coffer-native-guard-{}", std::process::id()));
            let manifest = import_fixture(&dir);
            let first =
                cofferdb::Database::create(dir.join("first.coffer"), DatabaseConfig::default())
                    .unwrap();
            let second =
                cofferdb::Database::create(dir.join("second.coffer"), DatabaseConfig::default())
                    .unwrap();
            let owner = owner::Owner::new(first.clone());
            let guard = owner
                .enter(false)
                .unwrap()
                .run(|db| cofferdb::binding::ImportBatchFailure::arm(db, 0))
                .unwrap();
            assert!(cofferdb::binding::ImportBatchFailure::arm(&first, 1).is_err());
            second
                .import_manifest(&manifest, cofferdb::OnConflict::Error)
                .unwrap();
            assert_eq!(second.stats().unwrap().document_count, 100);
            drop(guard); // An unconsumed fault is cancelled, even on a cloned handle.
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = cofferdb::binding::ImportBatchFailure::arm(&first, 0).unwrap();
                panic!("controlled guard cancellation");
            }));
            assert!(unwind.is_err());
            let guard = cofferdb::binding::ImportBatchFailure::arm(&first, 0).unwrap();
            let error = first
                .import_manifest(&manifest, cofferdb::OnConflict::Error)
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(cofferdb::binding::ImportBatchFailure::arm(&first, 0).is_err());
            // Consumption is one-shot; the same live guard must not fault twice.
            first
                .import_manifest(&manifest, cofferdb::OnConflict::Error)
                .unwrap();
            assert_eq!(first.stats().unwrap().document_count, 100);
            drop(guard);
            let guard = cofferdb::binding::ImportBatchFailure::arm(&first, 0).unwrap();
            first.clone().close().unwrap();
            assert!(cofferdb::binding::ImportBatchFailure::arm(&first, 0).is_err());
            drop(guard); // Closed owners have no remaining fault state.
            second.close().unwrap();
            drop(owner);
            std::fs::remove_dir_all(dir).unwrap();
        },
    );
}

#[cfg(all(feature = "native-test-support", unix))]
#[test]
fn native_import_failure_fork_drop_preserves_parent_locks() {
    isolated(
        "api::tests::native_import_failure_fork_drop_preserves_parent_locks",
        || {
            use std::os::fd::AsRawFd;
            extern "C" {
                fn fork() -> i32;
                fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
                fn kill(pid: i32, signal: i32) -> i32;
                fn _exit(status: i32) -> !;
                fn flock(fd: i32, operation: i32) -> i32;
            }
            let dir =
                std::env::temp_dir().join(format!("coffer-native-fork-{}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();
            let path = dir.join("data.coffer");
            // The guard is deliberately the last reference to Inner at fork.
            let guard = cofferdb::binding::ImportBatchFailure::arm(
                &cofferdb::Database::create(&path, DatabaseConfig::default()).unwrap(),
                0,
            )
            .unwrap();
            let pid = unsafe { fork() };
            assert!(pid >= 0);
            if pid == 0 {
                drop(guard);
                // No inherited Python, Rust test harness or file destructor runs.
                unsafe { _exit(0) };
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let mut status = 0;
            loop {
                let reaped = unsafe { waitpid(pid, &mut status, 1) };
                if reaped == pid {
                    break;
                }
                assert_eq!(reaped, 0);
                if std::time::Instant::now() >= deadline {
                    unsafe {
                        kill(pid, 9);
                        waitpid(pid, &mut status, 0);
                    }
                    panic!("fork guard child exceeded its five-second limit");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(status, 0);
            let lock = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(dir.join("data.coffer.lock"))
                .unwrap();
            let data = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            for file in [&lock, &data] {
                assert_eq!(
                    unsafe { flock(file.as_raw_fd(), 2 | 4) },
                    -1,
                    "child must preserve both parent OS locks"
                );
                assert_eq!(
                    std::io::Error::last_os_error().kind(),
                    std::io::ErrorKind::WouldBlock
                );
            }
            drop(guard);
            for file in [&lock, &data] {
                assert_eq!(
                    unsafe { flock(file.as_raw_fd(), 2 | 4) },
                    0,
                    "parent drop must release both OS locks"
                );
                assert_eq!(unsafe { flock(file.as_raw_fd(), 8) }, 0);
            }
            drop(lock);
            drop(data);
            std::fs::remove_dir_all(dir).unwrap();
        },
    );
}

#[test]
fn default_arguments_do_not_allocate_during_python_unpacking() {
    isolated(
        "api::tests::default_arguments_do_not_allocate_during_python_unpacking",
        || {
            Python::initialize();
            Python::attach(|py| {
                panic_guard::install();
                let (path, owner) = database("default");
                let object = collection(py, &owner);
                let method = object.bind(py).getattr("write_batch").unwrap();
                // 46B was the exact allocation of the former Python "error" default.
                // If another 46B allocation occurs later, it must remain MemoryError.
                let (result, hits) = allocation_fault(py, 46, || method.call0());
                let result = result.expect("argument defaults must not cause a native panic");
                if hits == 0 {
                    assert!(result.is_ok());
                } else {
                    assert!(result.unwrap_err().is_instance_of::<PyMemoryError>(py));
                }
                assert!(owner.enter(false).is_ok());
                owner.enter(true).unwrap().close().unwrap();
                drop(method);
                drop(object);
                drop(owner);
                std::fs::remove_file(path).unwrap();
            });
        },
    );
}

#[test]
fn required_key_memory_error_does_not_poison_or_write() {
    isolated(
        "api::tests::required_key_memory_error_does_not_poison_or_write",
        || {
            Python::initialize();
            Python::attach(|py| {
                panic_guard::install();
                let (path, owner) = database("key");
                let object = collection(py, &owner);
                let method = object.bind(py).getattr("write_batch").unwrap();
                let kwargs = py
                    .eval(
                        c"{'on_conflict': 'error', 'documents': [{'external_id': 'one'}]}",
                        None,
                        None,
                    )
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap();
                let before = std::fs::read(&path).unwrap();
                let (result, hits) = allocation_fault(py, 52, || method.call((), Some(&kwargs)));
                assert_eq!(hits, 1, "required-key fault must fire");
                let error = result.expect("key lookup must not panic").unwrap_err();
                assert!(error.is_instance_of::<PyMemoryError>(py));
                assert_eq!(
                    error
                        .value(py)
                        .getattr("operation_outcome")
                        .unwrap()
                        .get_item("state")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "not_started"
                );
                assert!(owner.enter(false).is_ok());
                assert_eq!(before, std::fs::read(&path).unwrap());
                owner.enter(true).unwrap().close().unwrap();
                drop(method);
                drop(object);
                drop(owner);
                std::fs::remove_file(path).unwrap();
            });
        },
    );
}

#[test]
fn sdk_diagnostic_memory_error_is_preserved() {
    isolated(
        "api::tests::sdk_diagnostic_memory_error_is_preserved",
        || {
            Python::initialize();
            Python::attach(|py| {
                panic_guard::install();
                errors::register(&PyModule::new(py, "fault_types").unwrap()).unwrap();
                let (path, owner) = database("closed");
                owner.enter(true).unwrap().close().unwrap();
                let object = Py::new(
                    py,
                    Database {
                        owner: Arc::clone(&owner),
                    },
                )
                .unwrap();
                let method = object.bind(py).getattr("stats").unwrap();
                let (result, hits) = allocation_fault(py, 59, || method.call0());
                assert_eq!(hits, 1, "SDK diagnostic fault must fire");
                assert!(result
                    .expect("diagnostics must not panic")
                    .unwrap_err()
                    .is_instance_of::<PyMemoryError>(py));
                drop(method);
                drop(object);
                drop(owner);
                std::fs::remove_file(path).unwrap();
            });
        },
    );
}

#[test]
fn outcome_attachment_memory_error_keeps_original_cause() {
    isolated(
        "api::tests::outcome_attachment_memory_error_keeps_original_cause",
        || {
            Python::initialize();
            Python::attach(|py| {
                let original = errors::python::<PyValueError>(py, "controlled failure");
                let pointer = original.value(py).as_ptr();
                let outcome = WriteOutcome {
                    state: "committed",
                    confirmed_committed_batches: Some(1),
                    confirmed_committed_records: Some(2),
                    last_confirmed_commit_id: Some(3),
                    last_confirmed_commit_sequence_no: Some(4),
                };
                let (result, hits) = allocation_fault(py, 46, || {
                    errors::finish::<()>(py, Err(original), Some(&outcome), std::process::id())
                });
                assert_eq!(hits, 1, "outcome key fault must fire");
                let error = result
                    .expect("outcome attachment must not panic")
                    .unwrap_err();
                assert!(error.is_instance_of::<PyMemoryError>(py));
                assert_eq!(error.cause(py).unwrap().value(py).as_ptr(), pointer);
            });
        },
    );
}

#[test]
fn direct_cpython_allocation_control_returns_memory_error() {
    isolated(
        "api::tests::direct_cpython_allocation_control_returns_memory_error",
        || {
            Python::initialize();
            Python::attach(|py| {
                let (result, hits) = allocation_fault(py, 46, || unsafe {
                    Bound::from_owned_ptr_or_err(
                        py,
                        ffi::PyUnicode_FromStringAndSize(c"error".as_ptr(), 5),
                    )
                });
                assert_eq!(hits, 1);
                assert!(result
                    .unwrap()
                    .unwrap_err()
                    .is_instance_of::<PyMemoryError>(py));
            });
        },
    );
}

fn fill_budget(input: &mut Input) -> String {
    let host = "p".repeat(32 * 1024 * 1024);
    let error = input.text(&host).unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    let detail = |field: &str| {
        error
            .detail()
            .iter()
            .find(|(key, _)| key == field)
            .unwrap()
            .1
            .parse::<usize>()
            .unwrap()
    };
    let used = detail("requested_bytes") - host.len();
    let available = detail("limit_bytes") - used;
    input.text(&host[..available - 128]).unwrap()
}

#[test]
fn collection_refusal_keeps_owned_name_charged_until_actual_free() {
    isolated(
        "api::tests::collection_refusal_keeps_owned_name_charged_until_actual_free",
        || {
            Python::initialize();
            Python::attach(|py| {
                errors::register(&PyModule::new(py, "fault_types").unwrap()).unwrap();
                let (path, owner) = database("identity");
                let mut call = owner.enter(false).unwrap();
                let operation = call.run(|db| Ok(Operation::new(db))).unwrap();
                let observation = operation.observation();
                let mut probe = operation.input();
                let mut pressure = operation.input();
                let mut input = operation.input();
                let name = input.text(&"n".repeat(8193)).unwrap();
                let identity = call
                    .run(|db| operation.create_collection(db, input.finish(config(name))))
                    .unwrap();
                let parts = identity.into_parts();
                let pointer = parts.1.as_ptr();
                let held = fill_budget(&mut pressure);
                let before = std::fs::read(&path).unwrap();
                let outcome = observation.write_outcome();
                let observed = Cell::new(None);
                let guard = crate::test_alloc::watch(pointer, &mut probe, &observed);
                let result = call
                    .protect(|| {
                        collection_output(py, &owner, parts, std::process::id(), Some(&outcome))
                    })
                    .unwrap();
                let error =
                    errors::finish(py, result, Some(&outcome), std::process::id()).unwrap_err();
                drop(guard);
                assert_eq!(
                    observed.get(),
                    Some(false),
                    "owned name must still be charged before System.dealloc"
                );
                assert!(error.is_instance_of::<errors::ResultMaterializationError>(py));
                assert!(error
                    .cause(py)
                    .unwrap()
                    .is_instance_of::<errors::CofferError>(py));
                let observed_outcome = error.value(py).getattr("operation_outcome").unwrap();
                assert_eq!(
                    observed_outcome
                        .get_item("state")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "committed"
                );
                assert_eq!(
                    observed_outcome
                        .get_item("confirmed_committed_batches")
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    1
                );
                assert_eq!(
                    observed_outcome
                        .get_item("confirmed_committed_records")
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    1
                );
                assert_eq!(before, std::fs::read(&path).unwrap());
                assert!(
                    probe.text(&"q".repeat(512)).is_ok(),
                    "freed name must release its credit"
                );
                drop(held);
                drop(pressure);
                drop(probe);
                drop(call);
                owner.enter(true).unwrap().close().unwrap();
                drop(owner);
                let reopened = cofferdb::Database::open(&path).unwrap();
                assert!(reopened.collection(&"n".repeat(8193)).is_ok());
                reopened.close().unwrap();
                std::fs::remove_file(path).unwrap();
            });
        },
    );
}

#[test]
fn ordinary_argument_control_releases_credit_after_actual_free() {
    let operation = Operation::standalone();
    let mut input = operation.input();
    let mut probe = operation.input();
    let name = input.text(&"n".repeat(8193)).unwrap();
    let pointer = name.as_ptr();
    let argument = input.finish(name);
    let held = fill_budget(&mut probe);
    let observed = Cell::new(None);
    let guard = crate::test_alloc::watch(pointer, &mut probe, &observed);
    drop(argument);
    drop(guard);
    assert_eq!(observed.get(), Some(false));
    assert!(probe.text(&"q".repeat(512)).is_ok());
    drop(held);
}

#[test]
fn collection_nested_memory_error_keeps_confirmed_commit_and_owner() {
    isolated(
        "api::tests::collection_nested_memory_error_keeps_confirmed_commit_and_owner",
        || {
            Python::initialize();
            Python::attach(|py| {
                errors::register(&PyModule::new(py, "fault_types").unwrap()).unwrap();
                let (path, owner) = database("nested-memory");
                let mut call = owner.enter(false).unwrap();
                let operation = call.run(|db| Ok(Operation::new(db))).unwrap();
                let observation = operation.observation();
                let mut pressure = operation.input();
                let mut input = operation.input();
                let name = input.text(&"n".repeat(8193)).unwrap();
                let identity = call
                    .run(|db| operation.create_collection(db, input.finish(config(name))))
                    .unwrap();
                let held = fill_budget(&mut pressure);
                let before = std::fs::read(&path).unwrap();
                let execution = Execution {
                    call,
                    result: Ok(identity),
                    outcome: Some(observation.write_outcome()),
                    pid: std::process::id(),
                };
                // Refuse the exact diagnostic allocation while the host-budget
                // error's core cause is being materialized, after actual commit.
                let (result, hits) = allocation_fault(py, 66, || execution.finish(py, &owner));
                assert_eq!(hits, 1, "nested host allocation fault must fire");
                let error = result
                    .expect("fallible diagnostic must not panic")
                    .unwrap_err();
                let correct_type = error.is_instance_of::<PyMemoryError>(py);
                let outcome = error.value(py).getattr("operation_outcome").unwrap();
                assert_eq!(
                    outcome
                        .get_item("state")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "committed"
                );
                assert_eq!(
                    outcome
                        .get_item("confirmed_committed_batches")
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    1
                );
                assert_eq!(
                    outcome
                        .get_item("confirmed_committed_records")
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    1
                );
                assert_eq!(before, std::fs::read(&path).unwrap());
                drop(held);
                drop(pressure);
                assert!(owner.enter(false).is_ok());
                owner.enter(true).unwrap().close().unwrap();
                drop(owner);
                let reopened = cofferdb::Database::open(&path).unwrap();
                assert!(reopened.collection(&"n".repeat(8193)).is_ok());
                reopened.close().unwrap();
                std::fs::remove_file(path).unwrap();
                assert!(
                    correct_type,
                    "nested MemoryError must remain the top-level exception"
                );
            });
        },
    );
}

#[test]
fn invalid_context_body_never_enters_an_exception_context_slot() {
    isolated(
        "api::tests::invalid_context_body_never_enters_an_exception_context_slot",
        || {
            Python::initialize();
            Python::attach(|py| {
                errors::register(&PyModule::new(py, "context_types").unwrap()).unwrap();
                let (path, owner) = database("invalid-context");
                let object = Py::new(
                    py,
                    Database {
                        owner: Arc::clone(&owner),
                    },
                )
                .unwrap();
                let call = owner.enter(false).unwrap();
                let result =
                    object
                        .bind(py)
                        .getattr("__exit__")
                        .unwrap()
                        .call1((py.None(), 123, py.None()));
                let error = result.unwrap_err();
                let context = error.value(py).getattr("__context__").unwrap();
                let valid_context = context.is_none()
                    || context.is_instance_of::<pyo3::exceptions::PyBaseException>();
                let correct_type = error.is_instance_of::<PyTypeError>(py);
                let cleanup_retained = error
                    .cause(py)
                    .is_some_and(|cause| cause.is_instance_of::<errors::BusyError>(py));
                drop(call);
                assert!(owner.enter(false).is_ok());
                owner.enter(true).unwrap().close().unwrap();
                drop(object);
                drop(owner);
                std::fs::remove_file(path).unwrap();
                assert!(
                    valid_context,
                    "exception context must be None or BaseException"
                );
                assert!(
                    correct_type,
                    "an invalid body must be rejected as TypeError"
                );
                assert!(
                    cleanup_retained,
                    "the existing close failure must remain the cause"
                );
            });
        },
    );
}
