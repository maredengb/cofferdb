use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::{Error, ErrorCode, Result};
use crate::physical_file::ManagedFile;

// Compaction can put a new inode at the same path; a hard link can give one
// inode two names. The sidecar keeps the path locked across replacement and
// the data-file lock excludes writers through aliases. Acquire the replacement
// file's lock before releasing the old one.
pub(crate) struct FileLock {
    // PID that acquired the sidecar. An inherited guard skips explicit unlock and
    // registry removal in the fork child.
    pid: u32,
    // Key in the process-local held-lock set, canonicalized when possible. Drop
    // must remove the same key that acquisition inserted.
    path: PathBuf,
    // Open sidecar descriptor holding the path's exclusive writer lock across
    // replacement of the data file.
    file: File,
    // Writer admission for the current data inode. Initially absent after sidecar
    // acquisition, then installed or swapped during writer binding.
    data: Option<DataFileLock>,
}

impl FileLock {
    pub(crate) fn acquire_exclusive(data_path: &Path) -> Result<Self> {
        // Open a stable sidecar at a name derived from the requested database
        // path. The data inode may later be replaced while this sidecar descriptor
        // continues to serialize writers using that name.
        let display_path = lock_path(data_path);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&display_path)?;
        // Use the canonical sidecar name as the process-local key when resolution
        // succeeds. Keep the path spelling used to open it for diagnostics and as
        // the fallback registry key.
        let registry_path = display_path
            .canonicalize()
            .unwrap_or_else(|_| display_path.clone());
        reserve_process_lock(&registry_path, &display_path)?;
        // Reserve the pathname before trying the kernel lock. A second opener in
        // this process should see contention through the registry even while the
        // first is still attempting admission.
        if let Err(err) = lock_file_exclusive_nonblocking(&file, &display_path) {
            // Drop the provisional registry claim on any kernel lock error. The
            // descriptor then closes on return and a later attempt can retry
            // without a stale process-local entry.
            release_process_lock(&registry_path);
            return Err(err);
        }
        Ok(Self {
            pid: std::process::id(),
            path: registry_path,
            file,
            data: None,
        })
    }

    // This accessor is for callers that have already bound writer admission to the
    // live data inode. The sidecar-only state returned by acquire_exclusive is an
    // intermediate state and has no data descriptor yet.
    pub(crate) fn data_file(&self) -> &File {
        &self
            .data
            .as_ref()
            .expect("writer admission holds the data file")
            .file
    }

    // The caller acquires the new inode guard before replacing the old one.
    // Returning the old guard leaves its release to the caller, including when
    // publication must keep both generations locked.
    pub(crate) fn replace_data_lock(&mut self, data: DataFileLock) -> Option<DataFileLock> {
        self.data.replace(data)
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // A fork child inherits descriptors referring to the parent's open file
        // descriptions. Let field destruction close the child's copies without
        // issuing explicit unlocks or taking the inherited registry mutex here.
        if self.pid != std::process::id() {
            return;
        }
        self.data.take();
        // Remove the per-process key after the sidecar unlock. Until then another
        // writer in this process cannot be admitted through this registry entry.
        unlock_file(&self.file);
        release_process_lock(&self.path);
    }
}

pub(crate) struct DataFileLock {
    // Managed descriptor retaining the physical file generation. The writer uses it
    // when it needs an additional content lock.
    file: ManagedFile,
    // Exclusive byte-zero lock held throughout this inode's writer admission,
    // including intervals when no content lock is held.
    _admission: RangeLock,
}

impl DataFileLock {
    pub(crate) fn acquire(file: &ManagedFile, path: &Path) -> Result<Self> {
        // Retain the managed physical generation independently of the caller's
        // file owner. Failure to clone it must occur before either compatibility
        // or range locks are attempted.
        let retained = file.try_clone()?;
        // On Unix writer admission shares a registry mutex with replacement
        // publication. Holding it across retirement checks and nonblocking lock
        // acquisition keeps a newly retired inode from gaining writer admission in
        // the gap.
        let admission = file.with_writer_admission(|fd| {
            compatibility_lock(fd, path)?;
            // Byte 0 reserves writer admission for the inode. Byte 1 is acquired
            // separately for content access so idle writers can coexist with readers.
            RangeLock::acquire(fd, 0, true, path)
        })?;
        Ok(Self {
            file: retained,
            _admission: admission,
        })
    }
}

pub(crate) struct RangeLock {
    // Cloned descriptor for the same open file description. It stays live until the
    // guard can release this specific OFD range.
    file: File,
    // Zero-based byte offset of the one-byte lock. Admission uses byte zero and
    // readers or active writers take byte one for content access.
    start: i64,
    // Acquiring PID. A fork child can close its copy without explicitly unlocking
    // the parent's shared open file description.
    pid: u32,
}

#[cfg(target_os = "linux")]
// Older Linux writers took an exclusive flock on the entire data inode. Take a
// shared flock before entering the byte-range protocol so those writers still
// exclude both readers and new writers, even through a hard link.
fn compatibility_lock(file: &File, path: &Path) -> Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: file keeps its descriptor open and flock takes only scalar lock flags.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } == 0 {
        return Ok(());
    }
    // Capture errno immediately after the failed call. Only a conflicting lock
    // receives the database's contention diagnostic here. Other kernel errors need
    // their original I/O classification.
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock {
        Err(lock_contention(path))
    } else {
        Err(err.into())
    }
}

#[cfg(not(target_os = "linux"))]
// Platforms without this Linux compatibility check still go through change_range
// for actual content or writer admission.
fn compatibility_lock(_: &File, _: &Path) -> Result<()> {
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn change_range(file: &File, start: i64, write: Option<bool>) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: On both supported kernels, libc::flock has valid zero field values. Set the range,
    // reference point and lock type below, leaving l_pid zero as required for OFD locks.
    let mut range: libc::flock = unsafe { std::mem::zeroed() };
    // The one-byte ranges are relative to the beginning of the inode. The
    // descriptor's current seek position must have no effect on which admission or
    // content lock changes.
    range.l_start = start;
    range.l_len = 1;
    range.l_whence = libc::SEEK_SET as _;
    // The same operation acquires shared or exclusive access and releases a
    // previously acquired range. Unlock must use the original byte, leaving the
    // other protocol byte alone.
    range.l_type = match write {
        Some(true) => libc::F_WRLCK,
        Some(false) => libc::F_RDLCK,
        None => libc::F_UNLCK,
    } as _;
    // Use the nonblocking open-file-description operation. Cloned descriptors
    // share lock ownership, while independent opens of the same inode have
    // distinct ownership.
    // SAFETY: F_OFD_SETLK reads a flock. range is initialized and lives through the
    // call, and file owns the descriptor.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_OFD_SETLK, &range) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
// Fail before claiming that content is protected on a target without a supported
// range-lock implementation. A path-sidecar reservation by itself cannot protect
// aliases.
fn change_range(_: &File, _: i64, _: Option<bool>) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "file locking is unsupported",
    ))
}

impl RangeLock {
    fn acquire(file: &File, start: i64, write: bool, path: &Path) -> Result<Self> {
        // Clone before acquiring the lock. The guard can then release the same
        // open file description even if the caller closes its own descriptor in
        // the meantime.
        let file = file.try_clone()?;
        change_range(&file, start, Some(write)).map_err(|err| {
            // An existing conflicting record lock may be reported as EACCES or
            // EAGAIN. Normalize both acquisition results to the path-bearing
            // contention error so bounded retry handles them alike.
            if matches!(
                err.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::PermissionDenied
            ) {
                lock_contention(path)
            } else {
                err.into()
            }
        })?;
        Ok(Self {
            file,
            start,
            pid: std::process::id(),
        })
    }

    pub(crate) fn reader(file: &File, path: &Path) -> Result<Self> {
        // Check a legacy whole-inode writer first on Linux, then take byte one
        // shared. Byte zero can remain held by an idle writer while a content scan
        // proceeds.
        compatibility_lock(file, path)?;
        Self::acquire(file, 1, false, path)
    }

    pub(crate) fn writer(file: &File, path: &Path) -> Result<Self> {
        // Ordinary writes start a fresh wait interval. Callers locking more than
        // one inode can supply a shared start time to writer_since instead.
        Self::writer_since(file, path, std::time::Instant::now())
    }

    pub(crate) fn writer_since(
        file: &File,
        path: &Path,
        started: std::time::Instant,
    ) -> Result<Self> {
        // Use the caller's original start time for the entire content-lock
        // deadline. Retry only contention, so an unsupported lock or I/O failure is
        // returned immediately.
        loop {
            // Attempt acquisition before testing elapsed time. An uncontended lock
            // can therefore succeed immediately even when an earlier stage has
            // consumed the wait interval.
            match Self::acquire(file, 1, true, path) {
                Err(e) if e.code() == ErrorCode::LockContention => {
                    if started.elapsed() >= std::time::Duration::from_secs(1) {
                        // No content write has started at this point. Preserve the
                        // contention code and path while recording the timed-out
                        // stage and its unchanged effect.
                        return Err(e
                            .with_detail("reason", "content_lock_timeout")
                            .with_detail("effect", "unchanged")
                            .with_detail("stage", "content_lock"));
                    }
                    // Pause only after confirmed contention, then make a fresh
                    // nonblocking attempt. A real I/O error must not disappear
                    // behind a later timeout.
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                result => return result,
            }
        }
    }
}

impl Drop for RangeLock {
    fn drop(&mut self) {
        // Only the acquiring process may explicitly unlock the shared open file
        // description. Dropping an inherited descriptor in a fork child leaves the
        // parent's live range held.
        if self.pid == std::process::id() {
            let _ = change_range(&self.file, self.start, None);
        }
    }
}

// Append .lock to the native file name, using cofferdb when the path has no file
// name. This sidecar path survives replacement of the data inode.
pub(crate) fn lock_path(data_path: &Path) -> PathBuf {
    let mut name = data_path
        .file_name()
        .map(|value| value.to_os_string())
        .unwrap_or_else(|| "cofferdb".into());
    name.push(".lock");
    data_path.with_file_name(name)
}

// Publish admission in the process-local registry before asking the kernel for the
// sidecar lock. Both acquisition failure and owner destruction must remove the
// same key.
fn reserve_process_lock(path: &Path, display_path: &Path) -> Result<()> {
    // A poisoned registry cannot admit another writer safely. Map that failure to
    // the existing contention category so callers receive an admission failure.
    let mut locks = held_locks().lock().map_err(|_| {
        Error::new(
            ErrorCode::LockContention,
            "process lock registry is poisoned",
        )
    })?;
    if !locks.insert(path.to_path_buf()) {
        return Err(lock_contention(display_path));
    }
    Ok(())
}

// Release the process-local claim after failed kernel acquisition or on guard
// destruction. A poisoned mutex is left alone because it cannot admit new claims.
fn release_process_lock(path: &Path) {
    if let Ok(mut locks) = held_locks().lock() {
        locks.remove(path);
    }
}

// Use ProcessLocal so a child process starts with its own registry. Looking up a
// lock after fork must avoid a mutex that was held by a vanished thread.
fn held_locks() -> &'static Mutex<BTreeSet<PathBuf>> {
    static LOCKS: crate::process_state::ProcessLocal<Mutex<BTreeSet<PathBuf>>> =
        crate::process_state::ProcessLocal::new();
    LOCKS.get()
}

#[cfg(test)]
pub(crate) fn with_registry_locked_for_test(action: impl FnOnce()) {
    let _held = held_locks().lock().unwrap();
    action();
}

// Keep the caller-facing path in the contention diagnostic. The internally
// canonicalized key is only used to identify duplicate registrations.
fn lock_contention(path: &Path) -> Error {
    Error::new(
        ErrorCode::LockContention,
        "database writer lock is already held",
    )
    .with_detail("lock_path", path.display())
}

#[cfg(unix)]
fn lock_file_exclusive_nonblocking(file: &File, path: &Path) -> Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::raw::c_int;

    // These flags implement immediate exclusive sidecar admission. Waiting for
    // active content locks is handled separately by the range-lock writer path.
    const LOCK_EX: c_int = 2;
    const LOCK_NB: c_int = 4;

    extern "C" {
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    // SAFETY: file owns the descriptor throughout this call. LOCK_EX | LOCK_NB
    // requests a nonblocking exclusive lock through the native flock ABI.
    if unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) } == 0 {
        return Ok(());
    }
    // Read errno immediately after the failed flock. Only WouldBlock means another
    // holder is using the sidecar; other OS failures carry their original error.
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock {
        Err(lock_contention(path))
    } else {
        Err(Error::from(err))
    }
}

#[cfg(unix)]
fn unlock_file(file: &File) {
    use std::os::fd::AsRawFd;
    use std::os::raw::c_int;

    // Issue an explicit unlock on the descriptor used for acquisition. A drop path
    // does not retry this operation or propagate an error.
    const LOCK_UN: c_int = 8;

    extern "C" {
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    // SAFETY: the File still owns the descriptor. LOCK_UN releases the lock without closing it.
    let _ = unsafe { flock(file.as_raw_fd(), LOCK_UN) };
}

#[cfg(not(unix))]
// The non-Unix fallback only bypasses sidecar flock. The separate byte-range
// admission rejects platforms without its supported kernel implementation.
fn lock_file_exclusive_nonblocking(_file: &File, _path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock_file(_file: &File) {}
