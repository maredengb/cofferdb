// Temporary files can outlive the scope that opened them. The handle performs I/O
// and the guard controls publication or explicit removal. Until publication or
// cleanup settles ownership, their shared state keeps the pathname, open
// descriptor and disk charge available to the final owner.
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Result;

mod budget;
mod path;
mod sort;
pub(crate) use budget::{DiskReservation, TemporaryDiskBudget};
pub(crate) use path::TemporaryPath;
pub(crate) use sort::{ExternalKeyLookup, ExternalTermFrequencies, ExternalUniqueKeys};

// The process-local counter distinguishes successive candidate names even if the
// clock repeats. Actual exclusivity comes from create_new when opening the file,
// including in the event of a collision.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);
#[cfg(unix)]
const MIN_NAME_CAPACITY: usize = 128;
// Linux uses an inline futex. Other supported Unix builds lazily allocate
// native mutex storage; this allowance is conservative, not an ABI size.
#[cfg(unix)]
const CLEANUP_LOCK_WORK_BYTES: usize = if cfg!(target_os = "linux") { 0 } else { 256 };

// The guard can still publish after the file wrapper has been dropped. Conversely,
// dropping the guard early must leave a writer's cleanup and disk accounting
// attached until its handle goes away.
#[derive(Debug)]
pub(crate) struct TempFileGuard {
    // Path exposed to the guard's caller, for publication or diagnostics. Cleanup
    // keeps its own copy so it can outlive the guard.
    path: TemporaryPath,
    // Owner shared with a budgeted file handle. The last owner is responsible for
    // removal and for releasing or transferring the disk charge.
    cleanup: SharedCleanup,
}

#[derive(Clone, Debug)]
struct SharedCleanup {
    // Some while this owner participates in cleanup. Taking the Arc during Drop
    // allows exactly one of the owners to dismantle the shared state.
    state: Option<Arc<Mutex<Cleanup>>>,
}

impl std::ops::Deref for SharedCleanup {
    type Target = Mutex<Cleanup>;

    fn deref(&self) -> &Mutex<Cleanup> {
        // Taking the optional Arc is reserved for final destruction. A live file
        // or guard must always have a cleanup state whenever it locks through this
        // wrapper.
        self.state.as_ref().expect("live cleanup owner")
    }
}

impl Drop for SharedCleanup {
    fn drop(&mut self) {
        // All strong owners use into_inner: even concurrent drops elect one
        // winner and release the Arc allocation before returning its credit.
        if let Some(state) = self.state.take().and_then(Arc::into_inner) {
            // Once this owner has obtained the inner mutex, no other thread can
            // still lock it. Recovering its contents after poisoning preserves the
            // information needed to attempt cleanup.
            let mut cleanup = state
                .into_inner()
                .unwrap_or_else(|error| error.into_inner());
            // Keep the shared-state charge in a local while destroying Cleanup.
            // Return it after pathname, descriptor and any pending-node ownership
            // have been dealt with.
            let work = cleanup._state_work.take();
            drop(cleanup);
            drop(work);
        }
    }
}

#[derive(Debug)]
struct Cleanup {
    // Name still eligible for removal. A successful rename or explicit cleanup
    // clears it to prevent a later Drop from unlinking a newly created file at that
    // name.
    path: TemporaryPath,
    // Open descriptor transferred from a dropped TemporaryFile. On Unix it
    // identifies the original inode even after another file appears at the same
    // path.
    file: Option<File>,
    // Disk credit owned by a budgeted temporary file. Publication may transfer it
    // to a persistent owner, and an unlink failure can move it to the retry list.
    disk: Option<budget::DiskReservation>,
    // Preallocated retry-list node for a budgeted file. Drop can give it to the
    // disk budget without needing a fresh allocation during an error.
    pending: Option<budget::PendingCleanupOwner>,
    // Optional workspace charge for the shared allocation and native lock
    // allowance. The last owner takes it out before dismantling the cleanup state.
    _state_work: Option<crate::json::WorkspaceReservation>,
}

// The file and its guard share cleanup ownership, so either drop order keeps
// the disk reservation until the last handle is closed and cleanup succeeds.
#[derive(Debug)]
pub(crate) struct TemporaryFile {
    // Active descriptor used for I/O. Drop takes it and hands it to the shared
    // cleanup state before releasing this owner's reference.
    file: Option<File>,
    // Keeps the cleanup state alive even if the separate guard is dropped while the
    // caller still has the descriptor.
    cleanup: SharedCleanup,
}

impl std::ops::Deref for TemporaryFile {
    type Target = File;

    fn deref(&self) -> &File {
        self.file.as_ref().expect("live temporary file")
    }
}

impl std::ops::DerefMut for TemporaryFile {
    fn deref_mut(&mut self) -> &mut File {
        self.file.as_mut().expect("live temporary file")
    }
}

impl std::io::Read for TemporaryFile {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.file.as_mut().unwrap().read(bytes)
    }
}

impl std::io::Write for TemporaryFile {
    // This wrapper forwards writes to the open file. Callers that stage counted
    // bytes reserve disk through the matching guard before issuing their writes.
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.file.as_mut().unwrap().write(bytes)
    }

    // Flush delegates to the operating-system file. Publication and the
    // disposition of its disk reservation happen separately through the guard.
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.as_mut().unwrap().flush()
    }
}

impl std::io::Seek for TemporaryFile {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        self.file.as_mut().unwrap().seek(position)
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        // Hand the descriptor to the cleanup state before this file's shared owner
        // drops. The last owner can then check the original inode even if the
        // filename has been replaced since creation.
        self.cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .file = self.file.take();
    }
}

#[cfg(any(test, not(unix)))]
pub(crate) fn create_budgeted_temporary_file_in(
    directory: &Path,
    label: &str,
    extension: &str,
    budget: Arc<TemporaryDiskBudget>,
) -> Result<(TemporaryFile, TempFileGuard)> {
    // Do not accumulate more abandoned files while earlier cleanup still fails.
    budget.retry_cleanup()?;
    // Cleanup runs in Drop: reserve its transfer node before creating a file.
    let pending = budget::PendingCleanupOwner::default();
    // A newly created file starts with zero charged bytes. As callers extend the
    // staged data, they grow the reservation kept by this same cleanup state.
    let (file, guard) = create_temporary_file_in(directory, label, extension)?;
    {
        let mut cleanup = guard
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Install both the disk reservation and the preallocated retry entry
        // before returning the descriptor and guard. A failed deletion can then
        // pass ownership to the retry list without a new allocation in Drop.
        cleanup.disk = Some(budget.reservation());
        cleanup.pending = Some(pending);
    }
    // Keep the descriptor in one wrapper and share only its cleanup owner. Closing
    // either return value first must leave the other with a path back to the same
    // reservation.
    let file = TemporaryFile {
        file: Some(file),
        cleanup: guard.cleanup.clone(),
    };
    Ok((file, guard))
}

fn create_budgeted_sort_file(
    directory: &Path,
    label: &'static str,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
) -> Result<(TemporaryFile, TempFileGuard)> {
    // Attribute temporary sort-file creation to catalog_build. The phase
    // identifies which operation requested workspace even though the disk budget
    // is shared with other staging uses.
    create_budgeted_temporary_file_in_context(
        directory,
        label,
        "tmp",
        budget,
        context,
        "catalog_build",
    )
}

pub(crate) fn create_budgeted_temporary_file_in_context(
    directory: &Path,
    label: &str,
    extension: &str,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
) -> Result<(TemporaryFile, TempFileGuard)> {
    // The portable path keeps disk reservations and cleanup ownership. The Unix
    // path below additionally admits memory used by names, paths and shared
    // cleanup state to this operation.
    #[cfg(not(unix))]
    {
        let _ = (context, phase);
        return create_budgeted_temporary_file_in(directory, label, extension, budget);
    }
    #[cfg(unix)]
    {
        // The timestamp is part of a candidate spelling. Falling back to zero
        // before the epoch still leaves the process ID, counter and exclusive file
        // creation to disambiguate attempts.
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        create_file_with_names(
            directory,
            budget,
            context,
            phase,
            // Compute the longest permitted name before opening a path. Overflow
            // in the label or extension size must fail preparation before the
            // process creates anything to clean up.
            name_capacity(label, extension)?,
            || {
                // A collision asks this closure for another candidate while
                // preserving the original timestamp. The counter only needs unique
                // increments within this process, so it does not synchronize
                // access to file contents.
                let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
                budgeted_filename(
                    label,
                    extension,
                    (std::process::id(), nonce, id),
                    context,
                    phase,
                )
            },
        )
    }
}

// This wraps the creation of one staging file. Import or compaction may already
// hold earlier files and their charges when creating the next one causes pressure.
pub(crate) fn create_budgeted_temporary_file_reclaiming(
    directory: &Path,
    label: &str,
    extension: &str,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
    reclaim: impl FnOnce() -> bool,
) -> Result<(TemporaryFile, TempFileGuard)> {
    // Both attempts use the same operation context, phase and disk budget. A
    // second attempt must still satisfy the limits that rejected the first.
    let create = || {
        create_budgeted_temporary_file_in_context(
            directory,
            label,
            extension,
            budget.clone(),
            context,
            phase,
        )
    };
    match create() {
        // Only a memory admission failure can benefit from evicting the caller's
        // retained workspace. An older file's unlink error or a filesystem error
        // must reach the caller with its original details.
        Err(error) if crate::json::is_memory_budget_error(&error) => {
            // Failed admission owners are gone; this file has not been opened.
            if reclaim() {
                // Reclaim is a one-shot decision. Return the second attempt's own
                // result, even if the newly available memory still falls short.
                create()
            } else {
                // If reclamation frees nothing, return the original error with its
                // accounting details intact.
                Err(error)
            }
        }
        result => result,
    }
}

#[cfg(unix)]
fn name_capacity(label: &str, extension: &str) -> Result<usize> {
    // Prefix/separators plus maximum decimal u32/u128/u64 identifier widths.
    82usize
        // Count label and extension in bytes, the unit used by the reserved
        // String. Reject an unrepresentable combined length before requesting any
        // credit.
        .checked_add(label.len())
        .and_then(|bytes| bytes.checked_add(extension.len()))
        // Use the same minimum capacity for short names and the native paths
        // derived from them. Longer caller-supplied labels raise the bound instead
        // of forcing a reallocation during open.
        .map(|bytes| bytes.max(MIN_NAME_CAPACITY))
        .ok_or_else(|| std::io::Error::other("temporary name capacity overflow").into())
}

#[cfg(unix)]
// Label and extension become part of a single filename. Callers supply components
// suitable for that purpose, since formatting here copies their contents verbatim.
fn budgeted_filename(
    label: &str,
    extension: &str,
    (pid, nonce, id): (u32, u128, u64),
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
) -> Result<crate::json::ReservedString> {
    // Charge the maximum formatted name before constructing it. The caller keeps
    // this buffer live while it acquires both native path buffers.
    let mut name = crate::json::ReservedString::with_capacity(
        name_capacity(label, extension)?,
        context,
        phase,
    )?;
    // Write the identifiers directly into the admitted allocation. A change to the
    // name format must still fit the bound computed by name_capacity.
    name.append(|output| write!(output, "cofferdb-{label}-{pid}-{nonce}-{id}.{extension}"))?;
    Ok(name)
}

#[cfg(unix)]
fn create_file_with_names(
    directory: &Path,
    budget: Arc<TemporaryDiskBudget>,
    context: &crate::json::JsonOperationContext,
    phase: &'static str,
    name_capacity: usize,
    mut next_name: impl FnMut() -> Result<crate::json::ReservedString>,
) -> Result<(TemporaryFile, TempFileGuard)> {
    use std::os::unix::fs::OpenOptionsExt;
    // Retry abandoned removals before accepting a new file under the same disk
    // limit. A persistent failure has to surface here while the budget still owns
    // the old file's charge.
    budget.retry_cleanup()?;
    // A failed unlink may need to queue the descriptor and disk charge during
    // Drop. Allocate its queue node once, outside the name-collision loop and
    // before creating a filesystem entry.
    let pending = budget::PendingCleanupOwner::new(context, phase)?;
    // Use the directory's native encoded length, so non-UTF-8 names need no
    // conversion. One extra byte covers a separator even if this directory already
    // ends in one.
    let capacity = directory
        .as_os_str()
        .as_encoded_bytes()
        .len()
        .checked_add(1)
        .and_then(|bytes| bytes.checked_add(name_capacity))
        .ok_or_else(|| std::io::Error::other("temporary path capacity overflow"))?;
    // A single queue node covers the entire creation request. Each iteration
    // reacquires its own name, two paths and mutex storage, none of which belongs
    // to a previously opened file.
    loop {
        let name = next_name()?;
        let path = TemporaryPath::join_name(directory, name.as_ref(), capacity, context, phase)?;
        // The guard exposes one path to callers. The cleanup owner needs a
        // separately charged copy because it can outlive the guard or transfer
        // that name to a pending removal.
        let cleanup_path = path.copy_reserved(context, phase)?;
        // The size estimate includes the Cleanup inside its mutex, the Arc's
        // reference counters and the platform allowance for initializing a native
        // lock. Path buffers and the pending node already have reservations.
        let bytes = std::mem::size_of::<Mutex<Cleanup>>()
            + 2 * std::mem::size_of::<usize>()
            + CLEANUP_LOCK_WORK_BYTES;
        // Admit the shared allocation while the candidate name and both paths are
        // still live. This charge follows the final cleanup owner, which can
        // outlive the file handle.
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, phase)?;
        // Leave the state unarmed while probing a name. Dropping this attempt
        // after a collision must never unlink the file that already occupies that
        // path.
        let mut state = Mutex::new(Cleanup {
            path: TemporaryPath::default(),
            file: None,
            disk: None,
            pending: None,
            _state_work: Some(work),
        });
        // Initialize the native lock before opening a file or sharing owners,
        // so Drop never needs a cold-lock allocation or initialization race.
        drop(state.lock().unwrap_or_else(|error| error.into_inner()));
        let mut options = OpenOptions::new();
        // Exclusive creation makes the filesystem decide ownership even if two
        // processes generate the same name. Set owner-only permissions when the
        // file first becomes visible.
        options.read(true).write(true).create_new(true).mode(0o600);
        match options.open(&path) {
            Ok(file) => {
                // Only a successfully created file activates cleanup ownership.
                // The mutex is still private here. Install the path, empty disk
                // reservation and retry node before exposing the Arc to either
                // owner, so shared cleanup always sees a fully initialized file.
                let owner = state.get_mut().unwrap_or_else(|error| error.into_inner());
                owner.path = cleanup_path;
                // Start with a zero-byte disk reservation. Callers grow it for
                // file contents separately from the workspace already held for
                // managing the name and cleanup.
                owner.disk = Some(budget.reservation());
                owner.pending = Some(pending);
                let cleanup = SharedCleanup {
                    state: Some(Arc::new(state)),
                };
                // The open handle and the guard each keep the same cleanup state
                // alive. The caller may close the handle or discard the guard
                // first.
                let guard = TempFileGuard {
                    path,
                    cleanup: cleanup.clone(),
                };
                return Ok((
                    TemporaryFile {
                        file: Some(file),
                        cleanup,
                    },
                    guard,
                ));
            }
            // Discard this attempt's name, path copies and shared-state charge on
            // a collision. Keep the pending node for the next name. Other open
            // failures carry information a new name would hide.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

// Report and benchmark output also use this unbudgeted constructor. Its guard
// still controls removal and publication even though a plain descriptor bypasses
// staging-byte accounting.
pub(crate) fn create_temporary_file_in(
    directory: &Path,
    label: &str,
    extension: &str,
) -> Result<(File, TempFileGuard)> {
    // A clock before the epoch still permits a candidate with nonce zero. The
    // counter supplies more candidates and exclusive creation decides whether each
    // is free.
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    loop {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            "cofferdb-{label}-{}-{nonce}-{id}.{extension}",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        // This path returns a plain File. Only the guard owns cleanup, so code
        // handing the file elsewhere must arrange an appropriate guard lifetime as
        // well.
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // Request owner-only access when creating the Unix file. Applying
            // permissions after open would expose the initial mode during that
            // interval.
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => return Ok((file, TempFileGuard::new(path))),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err.into()),
        }
    }
}

impl TempFileGuard {
    pub(crate) fn new(path: PathBuf) -> Self {
        // This guard adopts a name without disk or workspace charges. Its private
        // copy can be cleared after publication while the caller's view still
        // holds the original path.
        Self {
            cleanup: SharedCleanup {
                state: Some(Arc::new(Mutex::new(Cleanup {
                    path: TemporaryPath::unreserved(path.clone()),
                    file: None,
                    disk: None,
                    pending: None,
                    _state_work: None,
                }))),
            },
            path: TemporaryPath::unreserved(path),
        }
    }

    /// Report a secondary cleanup failure without obscuring the operation error.
    pub(crate) fn cleanup_after_error(&self, error: crate::Error) -> crate::Error {
        // Serialize an explicit removal with descriptor handoff by
        // TemporaryFile::drop. Recover a poisoned lock because it can still
        // contain the sole record of the file needing cleanup.
        let mut cleanup = self.cleanup.lock().unwrap_or_else(|err| err.into_inner());
        // A cleared name records prior publication or cleanup. A later operation
        // error must not trigger a second removal through a name that can now be
        // reused.
        if cleanup.path.as_os_str().is_empty() {
            return error;
        }
        // A dropped budgeted handle leaves its descriptor here for identity checks
        // on Unix. Until that handoff, or for an unbudgeted guard, only the saved
        // name is available.
        let result = match &cleanup.file {
            Some(file) => remove_owned_file(file, &cleanup.path),
            None => fs::remove_file(&cleanup.path),
        };
        match result {
            Ok(()) => {
                // Disarm a successful removal before releasing the lock. Keep the
                // guard's public path available for diagnostics.
                cleanup.path.clear();
                error
            }
            // Preserve the operation's error code and attach the failed removal as
            // secondary detail. Leaving the cleanup name in place gives the last
            // owner a chance to retry.
            Err(failure) => error.with_detail(
                "cleanup_error",
                format!("{}: {failure}", cleanup.path.display()),
            ),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    // Reserve space before a writer grows the file. The charge belongs to Cleanup
    // so failures in a later write can keep the allocation accounted for until
    // removal.
    pub(crate) fn reserve_disk(&mut self, bytes: u64) -> Result<()> {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // The unbudgeted publication path leaves cleanup.disk empty, so it has no
        // shared ledger to update.
        match &mut cleanup.disk {
            Some(work) => work.grow(bytes),
            None => Ok(()),
        }
    }

    // Release an unused or removed portion of a budgeted file. This adjusts
    // accounting only; the caller must already know the bytes are no longer
    // needed.
    pub(crate) fn release_disk(&mut self, bytes: u64) {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(work) = &mut cleanup.disk {
            work.release(bytes);
        }
    }

    // Move an existing disk charge between two file owners. This does not copy
    // bytes or change the total charge on the shared budget.
    pub(crate) fn transfer_disk_to(&mut self, destination: &mut Self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // Guards are not cloneable, so mutable ownership excludes opposite transfers.
        let mut source = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut destination = destination
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Each owner must have a reservation so failed cleanup of either path can
        // retain its part of the charge.
        match (&mut source.disk, &mut destination.disk) {
            (Some(source), Some(destination)) => source.transfer_to(destination, bytes),
            _ => Err(crate::Error::new(
                crate::ErrorCode::FileFormatError,
                "temporary disk transfer requires budgeted files",
            )),
        }
    }

    // Admit the minimum needed for progress and take additional headroom if
    // available. Unbudgeted files can use the entire preferred grant without an
    // accounting entry.
    pub(crate) fn reserve_disk_buffer(&mut self, minimum: u64, preferred: u64) -> Result<u64> {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match &mut cleanup.disk {
            Some(work) => work.grow_at_least(minimum, preferred),
            None => Ok(preferred.max(minimum)),
        }
    }

    /// Publish a completed, unbudgeted file without replacing the destination.
    /// Once linked, a cleanup error must identify the complete destination.
    pub(crate) fn link_new(
        self,
        destination: &Path,
        after_link_error: impl FnOnce(std::io::Error) -> crate::Error,
    ) -> Result<()> {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // A hard link can retain two names simultaneously. Restrict this API to
        // unbudgeted publication rather than silently dropping the charge for a
        // still-reachable file.
        debug_assert!(cleanup.disk.is_none());
        // Create a second directory name for the completed file. Hard link
        // creation fails if the destination exists, giving publication its
        // no-replace behavior.
        fs::hard_link(self.path(), destination)?;
        // The destination is already complete when removal of the temporary name
        // starts. Route that distinct failure through the caller's
        // post-publication error mapping.
        fs::remove_file(self.path()).map_err(after_link_error)?;
        // Disarm cleanup only after the temporary link has been removed. Before
        // that point, destruction still owns the cleanup obligation for the source
        // name.
        cleanup.path.clear();
        Ok(())
    }

    // Publish by rename and drop any temporary disk charge once ownership leaves
    // this guard.
    pub(crate) fn persist(mut self, destination: &Path) -> Result<()> {
        self.publish(destination).map(drop)
    }

    // Bring the temporary reservation up to the expected retained size before
    // publication. A refusal leaves the temporary owner responsible for cleanup.
    pub(crate) fn prepare_retention(&mut self, bytes: u64) -> Result<()> {
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Keep existing credit when it already covers the retained size. Admit any
        // shortfall before the caller can proceed to publication.
        let disk = cleanup.disk.as_mut().expect("budgeted publication");
        disk.grow(bytes.saturating_sub(disk.reserved_bytes()))
    }

    // Return the disk guard with publication so the new owner can account for the
    // file after the temporary pathname is gone.
    pub(crate) fn persist_retaining_disk(&mut self, destination: &Path) -> Result<DiskReservation> {
        Ok(self.publish(destination)?.expect("budgeted publication"))
    }

    fn publish(&mut self, destination: &Path) -> Result<Option<DiskReservation>> {
        // Rename before disarming cleanup. A failed rename leaves the source path
        // and its reservation available for the normal cleanup path.
        fs::rename(&self.path, destination)?;
        // Do not remove a different file subsequently created at the temporary name.
        let mut cleanup = self
            .cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cleanup.path.clear();
        // Separate the accounting owner from the finished path. The caller can
        // retain the charge when the destination remains part of its disk budget.
        Ok(cleanup.disk.take())
    }
}

impl Cleanup {
    fn cleanup_with(&mut self, remove: impl FnOnce(&Path) -> std::io::Result<()>) {
        if !self.path.as_os_str().is_empty() {
            if let Err(error) = remove(&self.path) {
                // Treat NotFound from the removal path as success. On Unix,
                // remove_owned_file turns a missing name with a still-linked
                // original into an error that retains ownership for retry.
                if error.kind() != std::io::ErrorKind::NotFound {
                    // A budgeted file may still occupy storage after a failed
                    // removal. Keep the original descriptor, path and byte charge
                    // in one retry entry until ownership can be checked again.
                    if let Some(disk) = self.disk.take() {
                        disk.retain_after_cleanup_failure(
                            // Move ownership out of Cleanup while the original
                            // descriptor is pinned. The pending node was allocated
                            // before file creation so retry registration can run
                            // even under memory pressure.
                            std::mem::take(&mut self.path),
                            self.file.take().expect("budgeted temporary file owner"),
                            self.pending.take().expect("budgeted cleanup transfer node"),
                        );
                    }
                }
            }
        }
    }
}

impl Drop for Cleanup {
    // Take the descriptor only for the identity check, then put it back so a
    // cleanup failure can transfer the descriptor and budget together.
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let result = remove_owned_file(&file, &self.path);
            self.file = Some(file);
            self.cleanup_with(|_| result);
        } else {
            // Unbudgeted files lack a pinned identity and the pending cleanup
            // node. Their destructor attempts ordinary pathname removal.
            self.cleanup_with(|path| fs::remove_file(path));
        }
    }
}

// Remove the temporary file through its recorded path, checking the pinned
// identity on Unix. A still-linked original can require another cleanup attempt
// and must keep its charge.
fn remove_owned_file(file: &File, path: &Path) -> std::io::Result<()> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    // Unix supplies the link count and device/inode identity used by this cleanup
    // path. The following unlink remains a pathname operation; the identity check
    // alone cannot make those two steps atomic.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let owned = file.metadata()?;
        // The pinned descriptor prevents inode reuse. An already unlinked
        // original can be closed without touching a replacement at this name.
        if owned.nlink() == 0 {
            return Ok(());
        }
        // Inspect the entry itself. Following a replacement symlink would compare
        // the target's identity rather than the entry we would unlink.
        let named = match fs::symlink_metadata(path) {
            Ok(named) => named,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // A disappeared name is only sufficient when the pinned file has
                // no other links. A moved original can remain charged even though
                // this path no longer exists.
                if file.metadata()?.nlink() == 0 {
                    return Ok(());
                }
                return Err(std::io::Error::other(
                    "temporary file was moved but remains linked",
                ));
            }
            Err(error) => return Err(error),
        };
        // Match both device and inode before attempting unlink. On mismatch, leave
        // the current directory entry and retain the original for cleanup
        // accounting.
        if owned.dev() != named.dev() || owned.ino() != named.ino() {
            return Err(std::io::Error::other(
                "temporary file name has changed owner",
            ));
        }
    }
    // The path can disappear after the identity check. Recheck the pinned file on
    // NotFound to distinguish an unlinked original from a moved one.
    if let Err(error) = fs::remove_file(path) {
        #[cfg(unix)]
        if error.kind() == std::io::ErrorKind::NotFound {
            use std::os::unix::fs::MetadataExt;
            return if file.metadata()?.nlink() == 0 {
                Ok(())
            } else {
                Err(std::io::Error::other(
                    "temporary file was moved but remains linked",
                ))
            };
        }
        return Err(error);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // Successful unlink removed only this name. Keep the reservation when
        // another hard link still refers to the original allocation.
        if file.metadata()?.nlink() != 0 {
            return Err(std::io::Error::other(
                "temporary file still has another link",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_file_management_bytes(directory: &Path) -> usize {
    #[cfg(unix)]
    {
        let path = directory.as_os_str().as_encoded_bytes().len() + 1 + 128;
        2 * path
            + std::mem::size_of::<budget::PendingCleanupNode>()
            + std::mem::size_of::<Mutex<Cleanup>>()
            + 2 * std::mem::size_of::<usize>()
            + if cfg!(target_os = "linux") { 0 } else { 256 }
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonOperationContext, WorkLedger};
    use std::io::Write;

    #[cfg(unix)]
    fn sort_owner_sizes(directory: &Path) -> (usize, usize, usize) {
        (
            directory.as_os_str().as_encoded_bytes().len() + 1 + 128,
            std::mem::size_of::<budget::PendingCleanupNode>(),
            std::mem::size_of::<Mutex<Cleanup>>()
                + 2 * std::mem::size_of::<usize>()
                + if cfg!(target_os = "linux") { 0 } else { 256 },
        )
    }

    fn limited_context(limit: usize) -> JsonOperationContext {
        JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)))
    }

    fn assert_available(context: &JsonOperationContext, bytes: usize) {
        let mut spare = context.workspace_reservation();
        spare.grow_to(bytes, bytes, "catalog_build").unwrap();
        assert_eq!(
            spare
                .grow_to(bytes + 1, bytes + 1, "catalog_build")
                .unwrap_err()
                .code(),
            crate::ErrorCode::ResourceExhausted
        );
    }

    #[test]
    #[cfg(unix)]
    fn generic_file_management_uses_original_context_phase_and_parent_total() {
        let directory = sort::tests::TestDirectory::new("generic-management");
        let management = test_file_management_bytes(&directory.0);
        let peak = management + 128;
        for phase in ["catalog_build", "batch_prepare", "operation_scratch"] {
            for label in [
                "compact",
                "import-records",
                "import-input",
                "export-embeddings",
            ] {
                for available in [peak - 1, peak, peak + 1] {
                    let total = available + 7;
                    let ledger = Arc::new(WorkLedger::with_limits(3 * peak, 3 * peak, total));
                    let context = JsonOperationContext::new(ledger.clone());
                    let sibling = JsonOperationContext::new(ledger);
                    let cached = sibling.cache_reservation(7, total).unwrap();
                    let disk = Arc::new(TemporaryDiskBudget::new(4096));
                    let result = create_budgeted_temporary_file_in_context(
                        &directory.0,
                        label,
                        "tmp",
                        disk.clone(),
                        &context,
                        phase,
                    );
                    if available < peak {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".to_owned()),
                            ("operation", phase.to_owned()),
                            ("limit_bytes", total.to_string()),
                            ("requested_bytes", (peak + 7).to_string()),
                        ] {
                            assert!(
                                error.detail().contains(&(key.to_owned(), value)),
                                "{error:?}"
                            );
                        }
                        drop(context);
                    } else {
                        let owners = result.unwrap();
                        drop(context);
                        assert_available(&sibling, available - management);
                        drop(owners);
                    }
                    assert_available(&sibling, available);
                    drop(cached);
                    assert_available(&sibling, total);
                    directory.assert_clean(&disk);
                }
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn file_creation_reclaims_once_without_repeating_prior_files() {
        let directory = sort::tests::TestDirectory::new("creation-reclaim");
        let management = test_file_management_bytes(&directory.0);
        let limit = 4 * (management + 128);
        for phase in ["catalog_build", "batch_prepare", "operation_scratch"] {
            for target in 1..=4 {
                let ledger = Arc::new(WorkLedger::with_limits(2 * limit, limit, limit));
                let context = JsonOperationContext::new(ledger.clone());
                let sibling = JsonOperationContext::new(ledger);
                let occupied = limit - target * management - 128 + 1;
                let mut cache = Some(context.cache_reservation(occupied as u128, limit).unwrap());
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let mut owners = Vec::new();
                let mut calls = 0;
                for ordinal in 0..target {
                    let (mut file, mut guard) = create_budgeted_temporary_file_reclaiming(
                        &directory.0,
                        "reclaim",
                        "tmp",
                        disk.clone(),
                        &context,
                        phase,
                        || {
                            calls += 1;
                            // Failed name/path/node/state reservations cannot remain live.
                            assert_available(&sibling, management + 127);
                            drop(cache.take());
                            true
                        },
                    )
                    .unwrap();
                    assert_eq!(calls, usize::from(ordinal + 1 == target));
                    guard.reserve_disk(1).unwrap();
                    file.write_all(&[ordinal as u8]).unwrap();
                    owners.push((file, guard));
                }
                assert_eq!(calls, 1);
                assert!(cache.is_none());
                assert_eq!(disk.used(), target as u64);
                drop(context);
                assert_available(&sibling, limit - target * management);
                for (ordinal, (_, guard)) in owners.iter().enumerate() {
                    assert_eq!(fs::read(guard.path()).unwrap(), [ordinal as u8]);
                }
                drop(owners);
                assert_available(&sibling, limit);
                directory.assert_clean(&disk);
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn file_creation_declined_or_insufficient_reclaim_is_bounded_and_io_does_not_reclaim() {
        let directory = sort::tests::TestDirectory::new("creation-reclaim-failure");
        let management = test_file_management_bytes(&directory.0);
        let peak = management + 128;
        let limit = 2 * peak;
        for release in [0, 1] {
            let ledger = Arc::new(WorkLedger::with_limits(2 * limit, limit, limit));
            let context = JsonOperationContext::new(ledger);
            let occupied = limit - peak + 2;
            let mut cache = context.cache_reservation(occupied as u128, limit).unwrap();
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let mut calls = 0;
            let error = create_budgeted_temporary_file_reclaiming(
                &directory.0,
                "reclaim",
                "tmp",
                disk.clone(),
                &context,
                "operation_scratch",
                || {
                    calls += 1;
                    assert_available(&context, peak - 2);
                    cache.retain(occupied - release);
                    release != 0
                },
            )
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .contains(&("requested_bytes".into(), (limit + 2 - release).to_string())));
            assert_available(&context, peak - 2 + release);
            drop(cache);
            assert_available(&context, limit);
            directory.assert_clean(&disk);
        }
        let context = limited_context(limit);
        let cache = context.cache_reservation(1, limit).unwrap();
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let error = create_budgeted_temporary_file_reclaiming(
            &directory.0.join("missing"),
            "reclaim",
            "tmp",
            disk.clone(),
            &context,
            "operation_scratch",
            || panic!("IO must not reclaim optional cache"),
        )
        .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::IoError);
        assert_available(&context, limit - 1);
        drop(cache);
        assert_available(&context, limit);
        directory.assert_clean(&disk);
    }

    #[test]
    #[cfg(unix)]
    fn dynamic_name_capacity_and_open_failure_do_not_leak_or_remove_candidates() {
        let directory = sort::tests::TestDirectory::new("generic-open-failure");
        let context = limited_context(32 * 1024);
        let label = "a".repeat(200);
        let extension = "jsonl";
        let expected = format!(
            "cofferdb-{label}-{}-{}-{}.{extension}",
            u32::MAX,
            u128::MAX,
            u64::MAX
        );
        let name = budgeted_filename(
            &label,
            extension,
            (u32::MAX, u128::MAX, u64::MAX),
            &context,
            "operation_scratch",
        )
        .unwrap();
        assert_eq!(name.as_ref(), expected);
        assert_eq!(name_capacity(&label, extension).unwrap(), expected.len());
        drop(name);
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let missing = directory.0.join("missing");
        let error = create_budgeted_temporary_file_in_context(
            &missing,
            "compact",
            "tmp",
            disk.clone(),
            &context,
            "operation_scratch",
        )
        .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::IoError);
        assert!(!missing.exists());
        assert_available(&context, 32 * 1024);
        directory.assert_clean(&disk);
    }

    #[test]
    #[cfg(unix)]
    fn native_cleanup_mutex_allocation_fits_the_structure_allowance() {
        let lock = Mutex::new(());
        let (_, bytes) = crate::test_alloc::allocation_bytes(|| drop(lock.lock().unwrap()));
        assert!(
            bytes <= CLEANUP_LOCK_WORK_BYTES as u128,
            "native mutex allocated {bytes} bytes"
        );
        let (_, allocations) = crate::test_alloc::allocations(move || {
            // Keep native mutex cleanup inside the measured scope.
            let warmed_lock = lock;
            drop(warmed_lock.lock().unwrap());
        });
        assert_eq!(allocations, 0);
    }

    #[test]
    #[cfg(unix)]
    fn untouched_sort_file_drops_without_cold_lock_allocations() {
        let directory = sort::tests::TestDirectory::new("sort-cold-drop");
        let context = JsonOperationContext::standalone();
        for file_first in [false, true] {
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let (file, guard) =
                create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context)
                    .unwrap();
            let (_, allocations) = crate::test_alloc::allocations(|| {
                if file_first {
                    drop(file);
                    drop(guard);
                } else {
                    drop(guard);
                    drop(file);
                }
            });
            assert_eq!(allocations, 0);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    #[cfg(unix)]
    fn sort_file_admits_all_management_before_creation_and_releases_both_drop_orders() {
        let directory = sort::tests::TestDirectory::new("sort-owner-admission");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let live = 2 * path_bytes + node_bytes + state_bytes;
        let peak = live + 128;
        for limit in [peak - 1, peak, peak + 1] {
            for file_first in [false, true] {
                let context = limited_context(limit);
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let result =
                    create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context);
                if limit < peak {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|item| item.0 == "requested_bytes" && item.1 == peak.to_string()));
                } else {
                    let (mut file, mut guard) = result.unwrap();
                    let path = guard.path().to_path_buf();
                    guard.reserve_disk(13).unwrap();
                    file.write_all(b"owned payload").unwrap();
                    assert_available(&context, limit - live);
                    if file_first {
                        drop(file);
                        assert_available(&context, limit - live);
                        drop(guard);
                    } else {
                        drop(guard);
                        assert_available(&context, limit - live + path_bytes);
                        drop(file);
                    }
                    assert!(!path.exists());
                }
                assert_available(&context, limit);
                directory.assert_clean(&disk);
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn sort_cleanup_failure_keeps_original_path_and_node_credit_without_allocation() {
        let directory = sort::tests::TestDirectory::new("sort-owner-retry");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let limit = 128 + 2 * path_bytes + node_bytes + state_bytes;
        for seed in 0..100u64 {
            let context = limited_context(limit);
            let disk = Arc::new(TemporaryDiskBudget::new(4096));
            let (mut file, mut guard) =
                create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context)
                    .unwrap();
            let bytes = seed + 1;
            guard.reserve_disk(bytes).unwrap();
            file.write_all(&vec![17; bytes as usize]).unwrap();
            let path = guard.path().to_path_buf();
            drop(file);
            let (_, allocations) = crate::test_alloc::allocations(|| {
                guard
                    .cleanup
                    .lock()
                    .unwrap()
                    .cleanup_with(|_| Err(std::io::Error::from_raw_os_error(13)));
                drop(guard);
            });
            assert_eq!(allocations, 0);
            assert_available(&context, limit - path_bytes - node_bytes);
            assert_eq!(disk.used(), bytes);
            assert_eq!(Arc::strong_count(&disk), 1);
            assert!(path.exists());
            disk.retry_cleanup().unwrap();
            assert_available(&context, limit);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    #[cfg(unix)]
    fn concurrent_sort_owner_drops_return_state_credit_once_even_when_poisoned() {
        let directory = sort::tests::TestDirectory::new("sort-owner-concurrent");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let limit = 128 + 2 * path_bytes + node_bytes + state_bytes;
        for poison in [false, true] {
            for _ in 0..100 {
                let context = limited_context(limit);
                let disk = Arc::new(TemporaryDiskBudget::new(4096));
                let (file, mut guard) =
                    create_budgeted_sort_file(&directory.0, "import-sort", disk.clone(), &context)
                        .unwrap();
                guard.reserve_disk(1).unwrap();
                if poison {
                    let state = &guard.cleanup;
                    assert!(std::thread::scope(|scope| scope
                        .spawn(|| {
                            let _lock = state.lock().unwrap();
                            panic!("injected cleanup lock poison");
                        })
                        .join())
                    .is_err());
                }
                let barrier = std::sync::Barrier::new(2);
                std::thread::scope(|scope| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        drop(file);
                    });
                    scope.spawn(move || {
                        barrier.wait();
                        drop(guard);
                    });
                });
                assert_available(&context, limit);
                directory.assert_clean(&disk);
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn sort_name_collision_preserves_existing_file_and_retry_credit() {
        let directory = sort::tests::TestDirectory::new("sort-collision");
        let (path_bytes, node_bytes, state_bytes) = sort_owner_sizes(&directory.0);
        let limit = 128 + 2 * path_bytes + node_bytes + state_bytes;
        let context = limited_context(limit);
        let disk = Arc::new(TemporaryDiskBudget::new(4096));
        let existing =
            budgeted_filename("import-sort", "tmp", (0, 0, 0), &context, "catalog_build").unwrap();
        let existing_path = directory.0.join(existing.as_ref());
        drop(existing);
        fs::write(&existing_path, b"untouched").unwrap();
        let mut id = 0;
        let (file, guard) = create_file_with_names(
            &directory.0,
            disk.clone(),
            &context,
            "catalog_build",
            128,
            || {
                let result =
                    budgeted_filename("import-sort", "tmp", (0, 0, id), &context, "catalog_build");
                id += 1;
                result
            },
        )
        .unwrap();
        assert_eq!(id, 2);
        assert_eq!(fs::read(&existing_path).unwrap(), b"untouched");
        drop((file, guard));
        assert_available(&context, limit);
        assert_eq!(fs::read(&existing_path).unwrap(), b"untouched");
        fs::remove_file(existing_path).unwrap();
        directory.assert_clean(&disk);
        for label in ["import-sort", "import-lookup"] {
            let name = budgeted_filename(
                label,
                "tmp",
                (u32::MAX, u128::MAX, u64::MAX),
                &context,
                "catalog_build",
            )
            .unwrap();
            assert!(name.as_ref().len() <= 128);
            assert_eq!(
                name.as_ref(),
                format!(
                    "cofferdb-{label}-{}-{}-{}.tmp",
                    u32::MAX,
                    u128::MAX,
                    u64::MAX
                )
            );
        }
        assert_available(&context, limit);
    }

    #[test]
    fn cleanup_failure_transfers_existing_owners_without_new_allocations() {
        for seed in 0..100u64 {
            let bytes = seed + 1;
            let budget = Arc::new(TemporaryDiskBudget::new(4096));
            let (mut file, mut guard) = create_budgeted_temporary_file_in(
                &std::env::temp_dir(),
                "cleanup-no-allocation",
                "tmp",
                budget.clone(),
            )
            .unwrap();
            guard.reserve_disk(bytes).unwrap();
            file.write_all(&vec![31; bytes as usize]).unwrap();
            let path = guard.path().to_path_buf();
            drop(file);
            let (_, allocations) = crate::test_alloc::allocations(|| {
                guard
                    .cleanup
                    .lock()
                    .unwrap()
                    .cleanup_with(|_| Err(std::io::Error::from_raw_os_error(13)));
                drop(guard);
            });
            assert_eq!(allocations, 0, "seed {seed}");
            assert_eq!(budget.used(), bytes);
            assert_eq!(Arc::strong_count(&budget), 1);
            assert!(path.exists());
            budget.retry_cleanup().unwrap();
            assert_eq!(budget.used(), 0);
            assert!(!path.exists());
        }
    }

    #[test]
    #[cfg(unix)]
    fn mixed_cleanup_retry_preserves_only_failed_files_and_replacement_names() {
        for seed in 0..10usize {
            let budget = Arc::new(TemporaryDiskBudget::new(4096));
            let mut files = Vec::new();
            for ordinal in 0..8u64 {
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "mixed-cleanup",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                let bytes = ordinal + 1;
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![17; bytes as usize]).unwrap();
                files.push((file, guard, bytes));
            }
            let mut retained = Vec::new();
            let mut healthy = Vec::new();
            let mut total = 0;
            for (ordinal, (file, guard, bytes)) in files.into_iter().enumerate() {
                let path = guard.path().to_path_buf();
                drop(file);
                guard
                    .cleanup
                    .lock()
                    .unwrap()
                    .cleanup_with(|_| Err(std::io::Error::from_raw_os_error(13)));
                drop(guard);
                total += bytes;
                if ordinal % 3 == seed % 3 {
                    let original = path.with_extension("owned");
                    fs::rename(&path, &original).unwrap();
                    fs::write(&path, b"replacement").unwrap();
                    retained.push((path, original, bytes));
                } else {
                    healthy.push(path);
                }
            }
            assert_eq!(budget.used(), total);
            assert_eq!(Arc::strong_count(&budget), 1);
            assert_eq!(
                budget.retry_cleanup().unwrap_err().code(),
                crate::ErrorCode::IoError
            );
            assert_eq!(budget.used(), retained.iter().map(|entry| entry.2).sum());
            assert!(healthy.iter().all(|path| !path.exists()));
            let mut restored = Vec::new();
            for (path, original, bytes) in retained {
                assert_eq!(fs::read(&path).unwrap(), b"replacement");
                assert_eq!(fs::read(&original).unwrap(), vec![17; bytes as usize]);
                fs::remove_file(&path).unwrap();
                fs::rename(original, &path).unwrap();
                restored.push(path);
            }
            budget.retry_cleanup().unwrap();
            assert_eq!(budget.used(), 0);
            assert!(restored.iter().all(|path| !path.exists()));
        }
    }

    #[test]
    fn cleanup_failures_keep_ownership_until_retry_or_runtime_shutdown_across_100_seeds() {
        for seed in 0..100_u64 {
            for kind in [
                None,
                Some(std::io::ErrorKind::NotFound),
                Some(std::io::ErrorKind::PermissionDenied),
                Some(std::io::ErrorKind::Interrupted),
                Some(std::io::ErrorKind::Other),
            ] {
                let limit = 4096;
                let bytes = (seed * 127) % (limit + 1);
                let budget = Arc::new(TemporaryDiskBudget::new(limit));
                let weak = Arc::downgrade(&budget);
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "cleanup-retry",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![b'x'; bytes as usize]).unwrap();
                let path = guard.path().to_path_buf();
                drop(file);
                guard.cleanup.lock().unwrap().cleanup_with(|actual| {
                    assert_eq!(actual, path);
                    if matches!(kind, None | Some(std::io::ErrorKind::NotFound)) {
                        fs::remove_file(actual).unwrap();
                    }
                    match kind {
                        None => Ok(()),
                        Some(kind) => Err(std::io::Error::new(kind, "injected cleanup failure")),
                    }
                });
                drop(guard);
                let orphaned = match kind {
                    None | Some(std::io::ErrorKind::NotFound) => 0,
                    _ => bytes,
                };
                assert_eq!(budget.used(), orphaned, "seed {seed}, {kind:?}");
                assert_eq!(Arc::strong_count(&budget), 1);
                let mut next = budget.reservation();
                next.grow(limit - orphaned).unwrap();
                assert!(budget.check_additional(1).is_err());
                drop(next);
                assert_eq!(budget.used(), orphaned);
                if seed % 2 == 0 {
                    let (file, guard) = create_budgeted_temporary_file_in(
                        &std::env::temp_dir(),
                        "after-cleanup",
                        "tmp",
                        budget.clone(),
                    )
                    .unwrap();
                    assert_eq!(budget.used(), 0);
                    assert!(!path.exists());
                    drop((file, guard));
                }
                drop(budget);
                assert!(weak.upgrade().is_none());
                assert!(!path.exists());
            }
        }
    }

    #[test]
    fn disk_and_cleanup_ownership_outlive_either_drop_order() {
        for seed in 0..100 {
            for file_first in [false, true] {
                let bytes = seed + 1;
                let budget = Arc::new(TemporaryDiskBudget::new(bytes));
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "drop-order",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                let path = guard.path().to_path_buf();
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![b'x'; bytes as usize]).unwrap();
                if file_first {
                    drop(file);
                    assert!(path.exists());
                    assert_eq!(budget.used(), bytes);
                    drop(guard);
                } else {
                    drop(guard);
                    assert!(path.exists());
                    assert_eq!(budget.used(), bytes);
                    file.write_all(b"").unwrap();
                    drop(file);
                }
                assert!(!path.exists());
                assert_eq!(budget.used(), 0);
            }
        }
    }

    #[test]
    fn live_reservations_retry_cleanup_under_pressure_without_double_release() {
        let budget = Arc::new(TemporaryDiskBudget::new(4096));
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let budget = &budget;
                let barrier = &barrier;
                scope.spawn(move || {
                    let (mut file, mut guard) = create_budgeted_temporary_file_in(
                        &std::env::temp_dir(),
                        "concurrent-cleanup",
                        "tmp",
                        budget.clone(),
                    )
                    .unwrap();
                    guard.reserve_disk(512).unwrap();
                    file.write_all(&[b'x'; 512]).unwrap();
                    let path = guard.path().to_path_buf();
                    drop(file);
                    barrier.wait();
                    guard.cleanup.lock().unwrap().cleanup_with(|_| {
                        Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
                    });
                    drop(guard);
                    barrier.wait();
                    // All eight failed files retain their original charge.
                    assert_eq!(budget.used(), 4096);
                    barrier.wait();
                    let mut next = budget.reservation();
                    next.grow(512).unwrap();
                    barrier.wait();
                    assert!(!path.exists());
                    assert_eq!(budget.used(), 4096);
                    barrier.wait();
                    drop(next);
                });
            }
        });
        assert_eq!(budget.used(), 0);
        assert_eq!(Arc::strong_count(&budget), 1);
    }

    #[cfg(unix)]
    #[test]
    fn retry_never_removes_replacement_names_or_releases_still_linked_originals() {
        for seed in 0..100 {
            for hard_link in [false, true] {
                let bytes = seed + 1;
                let budget = Arc::new(TemporaryDiskBudget::new(bytes));
                let (mut file, mut guard) = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "cleanup-owner",
                    "tmp",
                    budget.clone(),
                )
                .unwrap();
                let path = guard.path().to_path_buf();
                guard.reserve_disk(bytes).unwrap();
                file.write_all(&vec![b'x'; bytes as usize]).unwrap();
                let (other, moved) =
                    create_temporary_file_in(&std::env::temp_dir(), "moved-owner", "tmp").unwrap();
                drop(other);
                fs::remove_file(moved.path()).unwrap();
                if hard_link {
                    fs::hard_link(&path, moved.path()).unwrap();
                } else {
                    fs::rename(&path, moved.path()).unwrap();
                    fs::write(&path, b"replacement").unwrap();
                }
                drop((file, guard));
                assert_eq!(budget.used(), bytes);
                assert!(budget.retry_cleanup().is_err());
                let error = create_budgeted_temporary_file_in(
                    &std::env::temp_dir(),
                    "refused-after-failure",
                    "tmp",
                    budget.clone(),
                )
                .unwrap_err();
                assert_eq!(error.code(), crate::ErrorCode::IoError);
                assert_eq!(budget.used(), bytes);
                assert_eq!(fs::read(moved.path()).unwrap(), vec![b'x'; bytes as usize]);
                if !hard_link {
                    assert_eq!(fs::read(&path).unwrap(), b"replacement");
                }
                drop(moved);
                budget.retry_cleanup().unwrap();
                assert_eq!(budget.used(), 0);
                if !hard_link {
                    assert_eq!(fs::read(&path).unwrap(), b"replacement");
                    fs::remove_file(&path).unwrap();
                }
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn already_unlinked_original_does_not_remove_a_replacement_symlink() {
        let budget = Arc::new(TemporaryDiskBudget::new(1));
        let (file, mut guard) = create_budgeted_temporary_file_in(
            &std::env::temp_dir(),
            "replaced-unlinked",
            "tmp",
            budget.clone(),
        )
        .unwrap();
        guard.reserve_disk(1).unwrap();
        let path = guard.path().to_path_buf();
        fs::remove_file(&path).unwrap();
        let (mut target_file, target) =
            create_temporary_file_in(&std::env::temp_dir(), "untouched-target", "tmp").unwrap();
        target_file.write_all(b"untouched").unwrap();
        drop(target_file);
        std::os::unix::fs::symlink(target.path(), &path).unwrap();
        drop((file, guard));
        assert_eq!(budget.used(), 0);
        assert!(fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(target.path()).unwrap(), b"untouched");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn disk_charge_is_released_after_actual_cleanup_absence_or_publication() {
        let directory = std::env::temp_dir();
        let budget = Arc::new(TemporaryDiskBudget::new(1024));
        for already_removed in [false, true] {
            let (mut file, mut guard) = create_budgeted_temporary_file_in(
                &directory,
                "cleanup-budget",
                "tmp",
                budget.clone(),
            )
            .unwrap();
            guard.reserve_disk(3).unwrap();
            file.write_all(b"abc").unwrap();
            drop(file);
            let path = guard.path().to_path_buf();
            if already_removed {
                fs::remove_file(&path).unwrap();
            }
            drop(guard);
            assert!(!path.exists());
            assert_eq!(budget.used(), 0);
        }

        let (destination_file, destination) =
            create_temporary_file_in(&directory, "budget-published", "tmp").unwrap();
        drop(destination_file);
        let (mut file, mut guard) =
            create_budgeted_temporary_file_in(&directory, "budget-source", "tmp", budget.clone())
                .unwrap();
        guard.reserve_disk(3).unwrap();
        file.write_all(b"abc").unwrap();
        drop(file);
        guard.persist(destination.path()).unwrap();
        assert_eq!(fs::read(destination.path()).unwrap(), b"abc");
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publish_replaces_only_after_complete_writes_and_cleans_up_failed_renames() {
        let directory = std::env::temp_dir();
        let (mut destination_file, destination) =
            create_temporary_file_in(&directory, "publish-destination", "tmp").unwrap();
        destination_file.write_all(b"old").unwrap();
        drop(destination_file);
        let (mut file, staged) =
            create_temporary_file_in(&directory, "publish-source", "tmp").unwrap();
        let staged_path = staged.path().to_path_buf();
        file.write_all(b"new complete output").unwrap();
        file.flush().unwrap();
        drop(file);
        assert_eq!(fs::read(destination.path()).unwrap(), b"old");
        staged.persist(destination.path()).unwrap();
        assert!(!staged_path.exists());
        assert_eq!(
            fs::read(destination.path()).unwrap(),
            b"new complete output"
        );

        let (file, failed) =
            create_temporary_file_in(&directory, "publish-failure", "tmp").unwrap();
        let failed_path = failed.path().to_path_buf();
        drop(file);
        assert!(failed.persist(&directory).is_err());
        assert!(!failed_path.exists());
        assert!(directory.is_dir());
    }
}
