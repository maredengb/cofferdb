//! Maintenance commands must coordinate even when launched in a separate process.
#![cfg(any(target_os = "macos", target_os = "linux"))]

use cofferdb::{parse_json, CheckOptions, Database, JsonValue};
use std::fs::{self, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let dir = std::env::temp_dir().join(format!(
                "coffer-maintenance-process-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&dir) {
                Ok(()) => return Self(dir),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create fixture: {e}"),
            }
        }
    }

    fn create_db(&self) -> PathBuf {
        let path = self.0.join("source.coffer");
        json(&cli("init", &path, &[]), 0);
        json(
            &cli("create-collection", &path, &["notes", "--dim", "2"]),
            0,
        );
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cli(command: &str, path: &Path, args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .arg(command)
        .arg(path)
        .args(args)
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{command} remained blocked past the test deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

fn json(output: &Output, code: i32) -> JsonValue {
    assert_eq!(output.status.code(), Some(code), "{output:?}");
    let (payload, other) = if code == 0 {
        (&output.stdout, &output.stderr)
    } else {
        (&output.stderr, &output.stdout)
    };
    assert!(other.is_empty(), "{output:?}");
    parse_json(std::str::from_utf8(payload).unwrap()).unwrap()
}

fn reason<'a>(envelope: &'a JsonValue, expected_code: &str) -> &'a JsonValue {
    let error = envelope.get("error").unwrap();
    assert_eq!(error.get("code").unwrap().as_str(), Some(expected_code));
    error.get("detail").unwrap()
}

#[test]
fn idle_writer_allows_checks_but_excludes_all_maintenance_writers_by_inode() {
    let fixture = Fixture::new();
    let path = fixture.create_db();
    let alias = fixture.0.join("alias.coffer");
    let target = fixture.0.join("snapshot.coffer");
    fs::hard_link(&path, &alias).unwrap();
    let baseline = json(&cli("check", &path, &[]), 0);
    let writer = Database::open(&path).unwrap();
    let bytes = fs::read(&path).unwrap();

    for source in [&path, &alias] {
        let checked = json(&cli("check", source, &[]), 0);
        for key in ["last_valid_commit", "issues", "index_check"] {
            assert_eq!(
                checked.get("result").unwrap().get(key).unwrap(),
                baseline.get("result").unwrap().get(key).unwrap()
            );
        }
        for (command, args) in [
            ("repair", vec!["--truncate-tail"]),
            ("rebuild-index", vec![]),
            ("compact", vec![]),
            ("snapshot", vec![target.to_str().unwrap()]),
        ] {
            let blocked = json(&cli(command, source, &args), 4);
            reason(&blocked, "lock_contention");
            assert_eq!(fs::read(&path).unwrap(), bytes);
            assert!(!target.exists());
        }
    }
    writer.close().unwrap();
    for (command, args) in [
        ("repair", vec!["--truncate-tail"]),
        ("rebuild-index", vec![]),
        ("compact", vec![]),
        ("snapshot", vec![target.to_str().unwrap()]),
    ] {
        json(&cli(command, &path, &args), 0);
    }
    json(&cli("check", &target, &[]), 0);
}

#[test]
fn content_lock_prevents_an_unverified_report_and_times_out_before_mutation() {
    let fixture = Fixture::new();
    let path = fixture.create_db();
    let original = fs::read(&path).unwrap();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let set_content_lock = |mode| {
        // The file stays open in this process while an independent CLI tries
        // to read or change the database.
        let mut range: libc::flock = unsafe { std::mem::zeroed() };
        range.l_whence = libc::SEEK_SET as _;
        range.l_start = 1;
        range.l_len = 1;
        range.l_type = mode;
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_OFD_SETLK, &range) };
        assert_eq!(rc, 0, "{}", std::io::Error::last_os_error());
    };
    set_content_lock(libc::F_WRLCK as _);
    let unavailable = json(&cli("check", &path, &[]), 2);
    let detail = reason(&unavailable, "recovery_error");
    assert_eq!(
        detail.get("reason").unwrap().as_str(),
        Some("check_view_unavailable")
    );
    assert_eq!(
        detail.get("uncompleted_scope").unwrap().as_str(),
        Some("main_and_index")
    );
    let why_not = [
        "--collection",
        "notes",
        "--chunk-id",
        "missing",
        "--query",
        "needle",
    ];
    let busy = json(&cli("why-not", &path, &why_not), 4);
    assert_eq!(
        busy.get("error").unwrap().get("code").unwrap().as_str(),
        Some("lock_contention")
    );
    set_content_lock(libc::F_RDLCK as _);
    let diagnosis = json(&cli("why-not", &path, &why_not), 0);
    assert_eq!(
        diagnosis
            .get("result")
            .unwrap()
            .get("outcome")
            .unwrap()
            .get("reason")
            .unwrap()
            .as_str(),
        Some("not_found")
    );
    json(&cli("check", &path, &[]), 0);
    for (command, args) in [
        ("create-collection", vec!["blocked", "--dim", "2"]),
        ("rebuild-index", vec![]),
        ("compact", vec![]),
        ("repair", vec!["--truncate-tail"]),
    ] {
        let timed_out = json(&cli(command, &path, &args), 4);
        let detail = reason(&timed_out, "lock_contention");
        assert_eq!(
            detail.get("reason").unwrap().as_str(),
            Some("content_lock_timeout"),
            "{command}"
        );
        assert_eq!(
            detail.get("effect").unwrap().as_str(),
            Some("unchanged"),
            "{command}"
        );
        assert_eq!(fs::read(&path).unwrap(), original, "{command}");
    }
    drop(file);
    assert_eq!(fs::read(&path).unwrap(), original);
    json(&cli("check", &path, &[]), 0);
    json(&cli("repair", &path, &["--truncate-tail"]), 0);
}

#[test]
fn legacy_writer_flock_on_an_alias_excludes_new_checks_and_maintenance() {
    let fixture = Fixture::new();
    let path = fixture.create_db();
    let alias = fixture.0.join("legacy-alias.coffer");
    fs::hard_link(&path, &alias).unwrap();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(alias)
        .unwrap();
    // The previous release held this whole-inode lock until writer close.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(rc, 0, "{}", std::io::Error::last_os_error());
    let check_result = cli("check", &path, &[]);
    assert!(!check_result.status.success(), "{check_result:?}");
    assert!(check_result.stdout.is_empty());
    let blocked = json(&cli("compact", &path, &[]), 4);
    reason(&blocked, "lock_contention");
    drop(file);
    json(&cli("check", &path, &[]), 0);
}

#[test]
fn instance_check_stays_with_the_opened_database_after_replacement() {
    let fixture = Fixture::new();
    let path = fixture.create_db();
    let reader = Database::open_read_only(&path).unwrap();
    let checked = reader.check(CheckOptions::default()).unwrap();
    let old_bytes = fs::read(&path).unwrap();
    fs::rename(&path, fixture.0.join("original.coffer")).unwrap();
    json(&cli("init", &path, &[]), 0);
    let actual = reader.check(CheckOptions::default()).unwrap();
    assert_eq!(actual.last_valid_commit, checked.last_valid_commit);
    assert_eq!(actual.critical_end_offset, checked.critical_end_offset);
    assert_eq!(
        fs::read(fixture.0.join("original.coffer")).unwrap(),
        old_bytes
    );
    // An ordinary reader must not keep a legacy whole-file lock after check.
    let legacy = OpenOptions::new()
        .read(true)
        .write(true)
        .open(fixture.0.join("original.coffer"))
        .unwrap();
    let rc = unsafe { libc::flock(legacy.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(rc, 0, "{}", std::io::Error::last_os_error());
    let replacement = Database::check_path(&path, CheckOptions::default()).unwrap();
    assert_ne!(replacement.last_valid_commit, checked.last_valid_commit);
}

#[test]
fn child_dropping_an_inherited_writer_cannot_release_the_parent_locks() {
    const NAME: &str = "child_dropping_an_inherited_writer_cannot_release_the_parent_locks";
    if std::env::var_os("COFFER_TEST_ISOLATED_FORK").is_none() {
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--test-threads=1"])
            .env("COFFER_TEST_ISOLATED_FORK", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let fixture = Fixture::new();
    let path = fixture.create_db();
    let alias = fixture.0.join("alias.coffer");
    fs::hard_link(&path, &alias).unwrap();
    let writer = Database::open(&path).unwrap();
    // Only this test runs in this worker. Exercise the inherited handle and
    // avoid Rust test-harness teardown in the fork child.
    let child = unsafe { libc::fork() };
    assert!(child >= 0);
    if child == 0 {
        let refused = writer.stats().is_err();
        drop(writer);
        unsafe { libc::_exit(if refused { 0 } else { 1 }) }
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut status = 0;
    loop {
        let reaped = unsafe { libc::waitpid(child, &mut status, libc::WNOHANG) };
        if reaped == child {
            break;
        }
        assert_eq!(reaped, 0);
        if Instant::now() > deadline {
            unsafe {
                libc::kill(child, libc::SIGKILL);
                libc::waitpid(child, &mut status, 0);
            }
            panic!("inherited handle cleanup blocked after fork");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(status, 0);
    for source in [&path, &alias] {
        let result = json(&cli("compact", source, &[]), 4);
        reason(&result, "lock_contention");
    }
    writer.close().unwrap();
    json(&cli("compact", &path, &[]), 0);
}
