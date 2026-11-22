use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::{Deref, DerefMut};
use std::path::Path;

use crate::json::JsonOperationContext;
use crate::temporary::TempFileGuard;
use crate::Result;

// The descriptor closes before its last local retention credit is returned.
pub(crate) struct ManagedFile {
    // Descriptor through which this handle performs I/O. Closing takes it out
    // before unregistering the file's last local owner.
    file: Option<File>,
    #[cfg(unix)]
    // Shared ownership of the open device/inode generation on Unix. Cloned
    // descriptors and separate opens through hard links can refer to the same
    // owner.
    owner: Option<std::sync::Arc<unix::FileOwner>>,
}

impl ManagedFile {
    // Admit the Unix ownership record before invoking the file-creation closure. A
    // catalog budget refusal must occur before the closure can create a filesystem
    // entry.
    pub(crate) fn create_in(
        context: &JsonOperationContext,
        create: impl FnOnce() -> Result<File>,
    ) -> Result<Self> {
        #[cfg(unix)]
        {
            let memory = unix::reserve_owner(context)?;
            // Pass the prepared credit into registration with the creation
            // closure. Device and inode are known only after opening the file.
            // Registration can adopt a new owner or reuse a live one for the same
            // identity.
            unix::open_prepared(context, Some(memory), create)
        }
        #[cfg(not(unix))]
        {
            Self::open_in(context, create)
        }
    }

    // Opening an existing file need not reserve a new catalog owner in advance.
    // The Unix registry checks the opened identity and can share an existing owner
    // through a different pathname.
    pub(crate) fn open_in(
        context: &JsonOperationContext,
        open: impl FnOnce() -> Result<File>,
    ) -> Result<Self> {
        #[cfg(unix)]
        {
            unix::open_in(context, open)
        }
        #[cfg(not(unix))]
        {
            // This path keeps the native descriptor directly on platforms without
            // Unix generation tracking. Errors from the supplied open operation
            // still propagate to the caller.
            let _ = context;
            Ok(Self {
                file: Some(open()?),
            })
        }
    }

    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            // Clone the existing descriptor rather than reopening a pathname that
            // might now name a replacement. The Unix owner and its retention
            // charge are shared across both handles.
            file: Some(self.deref().try_clone()?),
            #[cfg(unix)]
            owner: self.owner.clone(),
        })
    }

    pub(crate) fn require_writer(&self) -> Result<()> {
        // Publication attaches a disk reservation to the displaced Unix owner.
        // Even a zero-byte reservation marks the old generation retired, so test
        // its presence rather than its size.
        #[cfg(unix)]
        if self
            .owner
            .as_ref()
            .expect("managed file owner")
            .disk
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
        {
            // A caller holding an old descriptor can still read the old contents,
            // but writer admission cannot make that handle refer to the new file.
            // Expose the reopen reason with the recovery error.
            return Err(crate::Error::new(
                crate::ErrorCode::RecoveryError,
                "database file has been replaced; confirm the current main file and reopen the writer",
            )
            .with_detail("reason", "writer_reopen_required"));
        }
        Ok(())
    }

    pub(crate) fn with_writer_admission<T>(
        &self,
        acquire: impl FnOnce(&File) -> Result<T>,
    ) -> Result<T> {
        // Keep retirement checks and nonblocking lock acquisition atomic with publication.
        #[cfg(unix)]
        let _registry = unix::registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Test retirement while the publication lock is still held. If it
        // succeeds, acquire the caller's lock before allowing a replacement to
        // retire this owner.
        self.require_writer()?;
        acquire(self)
    }

    // On Unix the registry lock spans the identity check, rename and handoff
    // of writer admission. Open handles on the displaced inode share its
    // FileOwner in this process, even through a hard link. While that retired
    // owner is live it rejects new writers and existing readers can keep using it.
    pub(crate) fn publish_replacement(
        &self,
        temporary: &mut TempFileGuard,
        destination: &Path,
        bind_writer: &mut impl FnMut(),
    ) -> Result<()> {
        #[cfg(unix)]
        {
            // The old descriptor still names the generation readers may hold after
            // rename. Measure that generation when preparing its retention credit,
            // even if the compacted output is smaller.
            let old_bytes = self.metadata()?.len();
            // Cover the displaced file before entering the publication section. A
            // refusal can still return with the original name and writer bindings
            // intact.
            temporary.prepare_retention(old_bytes)?;
            let owner = self.owner.as_ref().expect("managed file owner");
            // Use the same process registry as open, final close and writer
            // admission. Participating handles must observe the old admitted
            // writer or the replacement with its handoff complete.
            let registry = unix::registry()
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            // Take the generation's disk mutex after the registry lock, the same
            // order used to check writer admission. Holding it protects the
            // transition from writable to retired.
            let mut retained = owner.disk.lock().unwrap_or_else(|error| error.into_inner());
            // An existing credit marks this generation retired even if a hard link
            // still names it. Do not retire it a second time or move a later
            // output's credit onto it.
            if retained.is_some() {
                return Err(crate::Error::new(
                    crate::ErrorCode::RecoveryError,
                    "database file has already been replaced; reopen the current file",
                )
                .with_detail("reason", "writer_reopen_required"));
            }
            use std::os::unix::fs::MetadataExt;
            // Compare the destination's current identity to the still-open old
            // file while publication is serialized locally. The descriptor remains
            // tied to its inode even when the pathname is reused.
            let named = std::fs::metadata(destination)?;
            let opened = self.metadata()?;
            // A mismatch means this writer no longer owns the named generation.
            // Return before consuming the staging guard or invoking the prepared
            // writer handoff.
            if (named.dev(), named.ino()) != (opened.dev(), opened.ino()) {
                return Err(crate::Error::new(
                    crate::ErrorCode::RecoveryError,
                    "database path now refers to another file",
                )
                .with_detail("reason", "writer_reopen_required"));
            }
            // A reader may still have the displaced inode open after rename. Keep the disk
            // reservation on its generation until the last lease drops.
            // Rename consumes the staged file's disk reservation only on success.
            // The old open generation takes that reservation while the new file
            // becomes reachable at the destination.
            let credit = temporary.persist_retaining_disk(destination)?;
            *retained = Some(credit);
            // Only swap prepared state here; managed handles remain owned by the caller.
            // The callback installs the writer lock already acquired on the
            // replacement. Move the previous managed handle out for its caller to
            // drop later because final close takes this registry lock.
            bind_writer();
            drop(registry);
            // The staged reservation may cover more than the displaced generation.
            // Return the surplus after releasing the registry lock, retaining
            // exactly the bytes still pinned by local handles.
            retained.as_mut().unwrap().retain(old_bytes);
            Ok(())
        }
        #[cfg(not(unix))]
        {
            // The portable path has no per-inode retention registry. Finish the
            // rename and writer handoff while the staging reservation is live,
            // then release it.
            let credit = temporary.persist_retaining_disk(destination)?;
            bind_writer();
            drop(credit);
            Ok(())
        }
    }
}

impl Deref for ManagedFile {
    type Target = File;

    fn deref(&self) -> &File {
        // Callers borrowing the descriptor still operate on the originally opened
        // generation after a path replacement. Only final close removes the File
        // from a live ManagedFile.
        self.file.as_ref().expect("live managed file")
    }
}

impl DerefMut for ManagedFile {
    fn deref_mut(&mut self) -> &mut File {
        self.file.as_mut().expect("live managed file")
    }
}

impl Read for ManagedFile {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.deref_mut().read(bytes)
    }
}

impl Write for ManagedFile {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.deref_mut().write(bytes)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.deref_mut().flush()
    }
}

impl Seek for ManagedFile {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.deref_mut().seek(position)
    }
}

#[cfg(unix)]
impl Drop for ManagedFile {
    fn drop(&mut self) {
        // Closing a Unix managed handle also updates process-local generation
        // ownership. A plain descriptor close would miss a final reader's retained
        // disk and catalog reservations.
        unix::close(self);
    }
}

#[cfg(all(test, unix))]
pub(crate) fn with_registry_locked_for_test(action: impl FnOnce()) {
    let _held = unix::registry().lock().unwrap();
    action();
}

#[cfg(unix)]
mod unix {
    use super::*;
    use crate::json::{btree_workspace, CatalogReservation};
    use crate::temporary::DiskReservation;
    use std::collections::BTreeMap;
    use std::os::unix::fs::MetadataExt;
    use std::sync::{Arc, Mutex, Weak};

    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
    pub(super) struct Identity {
        // Device number read from the open descriptor. Inode numbers are unique
        // only within a filesystem.
        device: u64,
        // Inode number on that device. A rename preserves this identity even when a
        // different file takes the original path.
        inode: u64,
    }

    pub(super) struct FileOwner {
        // Key under which this physical generation is registered. Final close
        // removes the entry only if it still points at this owner.
        identity: Identity,
        // None while this generation can admit writers. After replacement, holds
        // the disk credit for the displaced inode while local handles keep it open.
        pub(super) disk: Mutex<Option<DiskReservation>>,
        // Memory charge for the shared owner and its conservative registry-node and
        // lock allowance. Final close can drop the allocations before returning the
        // credit.
        memory: Option<CatalogReservation>,
    }

    // The registry's weak reference alone cannot keep a generation alive. A
    // successful open can upgrade it and hold the strong reference in its
    // ManagedFile.
    type Registry = BTreeMap<Identity, Weak<FileOwner>>;

    // Keep the registry process-local. A child that opens its own descriptor needs
    // independent ownership and must avoid inherited mutex state.
    pub(super) fn registry() -> &'static Mutex<Registry> {
        static REGISTRY: crate::process_state::ProcessLocal<Mutex<Registry>> =
            crate::process_state::ProcessLocal::new();
        REGISTRY.get()
    }

    // Open without a pre-reserved owner. Existing physical generations reuse their
    // registered owner; a newly observed generation must admit one under the
    // registry lock.
    pub(super) fn open_in(
        context: &JsonOperationContext,
        open: impl FnOnce() -> Result<File>,
    ) -> Result<ManagedFile> {
        open_prepared(context, None, open)
    }

    pub(super) fn reserve_owner(context: &JsonOperationContext) -> Result<CatalogReservation> {
        // One conservative table node per live identity covers shared map storage.
        let mut memory = context.catalog_reservation();
        // Charge the shared owner, Arc counters, registry storage and platform
        // lock allowance together. This estimate covers retained metadata,
        // independent of the descriptor's file length.
        memory.grow(
            std::mem::size_of::<FileOwner>()
                + 2 * std::mem::size_of::<usize>()
                + btree_workspace::<Identity, Weak<FileOwner>>(1)
                + if cfg!(target_os = "linux") { 0 } else { 256 },
        )?;
        Ok(memory)
    }

    pub(super) fn open_prepared(
        context: &JsonOperationContext,
        mut prepared: Option<CatalogReservation>,
        open: impl FnOnce() -> Result<File>,
    ) -> Result<ManagedFile> {
        // Open, registration, publication and final close share this boundary.
        let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
        // Obtain identity from the opened descriptor under the registry lock. A
        // pathname can name a different generation after a rename, while the
        // descriptor still refers to the original.
        let file = open()?;
        let metadata = file.metadata()?;
        let identity = Identity {
            device: metadata.dev(),
            inode: metadata.ino(),
        };
        // Upgrade the weak entry before leaving the lock. Hard links and
        // separately opened descriptors for one device/inode then share the same
        // generation and retained-disk owner.
        let owner = if let Some(owner) = registry.get(&identity).and_then(Weak::upgrade) {
            owner
        } else {
            // This admission only locks the memory ledger; it never runs disk cleanup.
            // A caller can reserve this catalog metadata before invoking the
            // file-creation closure. Consume that charge when installing a new
            // identity; an unused prepared guard refunds on drop.
            let memory = match prepared.take() {
                Some(memory) => memory,
                None => reserve_owner(context)?,
            };
            let owner = Arc::new(FileOwner {
                identity,
                disk: Mutex::new(None),
                memory: Some(memory),
            });
            // Expose the initialized owner to later opens using a weak entry. Only
            // the returned ManagedFile and its peers retain strong references.
            registry.insert(identity, Arc::downgrade(&owner));
            owner
        };
        Ok(ManagedFile {
            file: Some(file),
            owner: Some(owner),
        })
    }

    // Close the descriptor and unregister its owner under the same lock used by
    // opens. Refunds can follow outside the registry lock.
    pub(super) fn close(file: &mut ManagedFile) {
        let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
        drop(file.file.take());
        let owner = file.owner.take().expect("managed file owner");
        let pointer = Arc::as_ptr(&owner);
        // Use Arc::into_inner so exactly the final strong owner obtains the
        // retained state. Earlier closes leave the disk reservation attached to
        // surviving handles.
        let state = Arc::into_inner(owner);
        // A weak registry entry identifies an owner as well as an inode. Only the
        // last strong owner may remove its own entry, under the same lock used by
        // opens.
        if let Some(state) = &state {
            if registry
                .get(&state.identity)
                .is_some_and(|entry| entry.as_ptr() == pointer)
            {
                registry.remove(&state.identity);
                // Release the empty map's allocation before refunding the final
                // owner's memory charge. Removing one node from a nonempty
                // BTreeMap can leave shared tree storage.
                if registry.is_empty() {
                    drop(std::mem::take(&mut *registry));
                }
            }
        }
        // Release the registry lock before destroying budget owners. Returning
        // disk or catalog credit may take separate ledger locks; neither is needed
        // to make the identity invisible.
        drop(registry);
        if let Some(mut state) = state {
            // Keep the catalog reservation alive while dropping the physical owner
            // and its retained disk lease. Only then refund the owner's
            // accounting.
            let memory = state.memory.take();
            drop(state);
            drop(memory);
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use crate::temporary::{create_budgeted_temporary_file_in_context, TemporaryDiskBudget};
    use crate::ErrorCode;
    use std::fs::{self, OpenOptions};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cofferdb-file-owner-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn create(path: &Path, context: &JsonOperationContext, bytes: usize) -> ManagedFile {
        let mut file = ManagedFile::create_in(context, || {
            Ok(OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(path)?)
        })
        .unwrap();
        file.write_all(&vec![0x42; bytes]).unwrap();
        file
    }

    fn staging(
        root: &Path,
        context: &JsonOperationContext,
        budget: &Arc<TemporaryDiskBudget>,
        bytes: usize,
    ) -> (ManagedFile, TempFileGuard) {
        let (mut file, mut guard) = create_budgeted_temporary_file_in_context(
            root,
            "publish",
            "tmp",
            Arc::clone(budget),
            context,
            "operation_scratch",
        )
        .unwrap();
        guard.reserve_disk(bytes as u64).unwrap();
        file.write_all(&vec![0x19; bytes]).unwrap();
        drop(file);
        let replacement = ManagedFile::open_in(context, || {
            Ok(OpenOptions::new()
                .read(true)
                .write(true)
                .open(guard.path())?)
        })
        .unwrap();
        (replacement, guard)
    }

    #[test]
    fn identity_is_shared_without_recharging_and_create_refuses_before_side_effects() {
        let root = Directory::new();
        let path = root.0.join("data");
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(4096, 4096, 4096)));
        let file = create(&path, &context, 11);
        let available = context.available_workspace_bytes();
        assert!(available < 4096);
        let no_catalog =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(4096, 0, 4096)));
        let reader = ManagedFile::open_in(&no_catalog, || Ok(File::open(&path)?)).unwrap();
        assert!(Arc::ptr_eq(
            file.owner.as_ref().unwrap(),
            reader.owner.as_ref().unwrap()
        ));
        assert_eq!(context.available_workspace_bytes(), available);
        assert_eq!(no_catalog.available_workspace_bytes(), 4096);
        let absent = root.0.join("absent");
        let error = ManagedFile::create_in(&no_catalog, || Ok(File::create(&absent)?))
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert!(!absent.exists());
        drop(file);
        assert_eq!(context.available_workspace_bytes(), available);
        drop(reader);
        assert_eq!(context.available_workspace_bytes(), 4096);
    }

    #[test]
    fn publication_uses_maximum_overlap_and_retains_one_credit_until_the_last_reader() {
        for (old_bytes, new_bytes) in [(40, 80), (64, 64), (80, 40)] {
            let root = Directory::new();
            let path = root.0.join("data");
            let context = JsonOperationContext::standalone();
            let file = create(&path, &context, old_bytes);
            let reader = ManagedFile::open_in(&context, || Ok(File::open(&path)?)).unwrap();
            let mut clone = reader.try_clone().unwrap();
            let budget = Arc::new(TemporaryDiskBudget::new(
                old_bytes.max(new_bytes) as u64 + 7,
            ));
            let (mut outside_file, mut outside) = create_budgeted_temporary_file_in_context(
                &root.0,
                "held",
                "tmp",
                Arc::clone(&budget),
                &context,
                "operation_scratch",
            )
            .unwrap();
            outside.reserve_disk(7).unwrap();
            outside_file.write_all(&[0; 7]).unwrap();
            let (replacement, mut temporary) = staging(&root.0, &context, &budget, new_bytes);
            file.publish_replacement(&mut temporary, &path, &mut || ())
                .unwrap();
            assert_eq!(fs::read(&path).unwrap(), vec![0x19; new_bytes]);
            assert_eq!(budget.used(), old_bytes as u64 + 7);
            drop(file);
            drop(reader);
            assert_eq!(budget.used(), old_bytes as u64 + 7);
            clone.seek(SeekFrom::Start(0)).unwrap();
            let mut old = Vec::new();
            clone.read_to_end(&mut old).unwrap();
            assert_eq!(old, vec![0x42; old_bytes]);
            drop(clone);
            assert_eq!(budget.used(), 7);
            drop(replacement);
            drop(temporary);
            drop(outside_file);
            drop(outside);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn failed_admission_and_rename_leave_cleanup_and_the_original_file_covered() {
        for rename_failure in [false, true] {
            let root = Directory::new();
            let path = root.0.join("data");
            let context = JsonOperationContext::standalone();
            let file = create(&path, &context, 80);
            let budget = Arc::new(TemporaryDiskBudget::new(if rename_failure {
                80
            } else {
                79
            }));
            let (replacement, mut temporary) = staging(&root.0, &context, &budget, 40);
            let private = temporary.path().to_path_buf();
            let destination = if rename_failure { &root.0 } else { &path };
            let bound = std::cell::Cell::new(false);
            let error = file
                .publish_replacement(&mut temporary, destination, &mut || bound.set(true))
                .unwrap_err();
            assert!(!bound.get());
            assert_eq!(
                error.code(),
                if rename_failure {
                    ErrorCode::RecoveryError
                } else {
                    ErrorCode::ResourceExhausted
                }
            );
            assert_eq!(fs::read(&path).unwrap(), vec![0x42; 80]);
            file.require_writer().unwrap();
            assert!(private.exists());
            assert_eq!(budget.used(), if rename_failure { 80 } else { 40 });
            drop(replacement);
            drop(temporary);
            assert!(!private.exists());
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn writer_lock_admission_checks_the_open_identity_at_the_publication_boundary() {
        let root = Directory::new();
        let path = root.0.join("data");
        let alias = root.0.join("alias");
        let context = JsonOperationContext::standalone();
        let file = create(&path, &context, 80);
        fs::hard_link(&path, &alias).unwrap();
        let lock = crate::lock::DataFileLock::acquire(&file, &path).unwrap();
        let pending = ManagedFile::open_in(&context, || {
            Ok(OpenOptions::new().read(true).write(true).open(&alias)?)
        })
        .unwrap();
        // A writable descriptor opened before rename must still pass admission later.
        pending
            .with_writer_admission(|_| {
                assert!(matches!(
                    unix::registry().try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ));
                Ok(())
            })
            .unwrap();
        let budget = Arc::new(TemporaryDiskBudget::new(80));
        let (replacement, mut temporary) = staging(&root.0, &context, &budget, 40);
        let bound = std::cell::Cell::new(false);
        file.publish_replacement(&mut temporary, &path, &mut || {
            assert!(matches!(
                unix::registry().try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            assert_eq!(fs::read(&path).unwrap(), vec![0x19; 40]);
            bound.set(true);
        })
        .unwrap();
        assert!(bound.get());
        let error = crate::lock::DataFileLock::acquire(&pending, &alias)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert!(error
            .detail()
            .contains(&("reason".into(), "writer_reopen_required".into())));
        let current_lock = crate::lock::DataFileLock::acquire(&replacement, &path).unwrap();
        assert_eq!(fs::read(&alias).unwrap(), vec![0x42; 80]);
        assert_eq!(fs::read(&path).unwrap(), vec![0x19; 40]);
        assert_eq!(budget.used(), 80);
        drop((file, lock));
        assert_eq!(budget.used(), 80);
        drop(pending);
        assert_eq!(budget.used(), 0);
        drop((replacement, current_lock, temporary));
    }

    #[test]
    fn concurrent_final_close_and_open_preserve_identity_and_metadata_credit() {
        let root = Directory::new();
        let path = root.0.join("data");
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(4096, 4096, 4096)));
        drop(create(&path, &context, 13));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..256 {
                        let mut file =
                            ManagedFile::open_in(&context, || Ok(File::open(&path)?)).unwrap();
                        let mut bytes = [0; 13];
                        file.read_exact(&mut bytes).unwrap();
                        assert_eq!(bytes, [0x42; 13]);
                    }
                });
            }
        });
        assert_eq!(context.available_workspace_bytes(), 4096);
    }
}
