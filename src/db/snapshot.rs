use super::*;
#[cfg(unix)]
use std::{
    ffi::CString,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::ffi::OsStrExt,
    },
};

#[cfg(unix)]
fn name(path: &Path) -> Result<CString> {
    // Directory operations below take an open parent and a single filename. Strip
    // the parent here so resolving the leaf cannot reselect the directory by
    // walking the supplied path a second time.
    let component = path
        .file_name()
        .ok_or_else(|| Error::new(ErrorCode::InvalidSearchParams, "missing snapshot filename"))?;
    // A Unix filename need not be UTF-8. Pass its original bytes to libc,
    // rejecting a NUL byte that would otherwise terminate the C string before the
    // end of the requested name.
    CString::new(component.as_bytes())
        .map_err(|_| Error::new(ErrorCode::InvalidSearchParams, "invalid snapshot filename"))
}

#[cfg(unix)]
fn create_stage(dir: libc::c_int, target: &std::ffi::CStr) -> Result<CString> {
    use std::sync::atomic::{AtomicU64, Ordering};
    // The sequence only proposes names within this process. Relaxed ordering
    // suffices because mkdirat is the operation that claims a name in the
    // filesystem, including against stale or concurrent entries.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        // Numeric names avoid case-folding aliases and fit independently of the target length.
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        // The process ID separates the common case of concurrent snapshotters with
        // equal counters. A stale entry left by an earlier process or a counter
        // wrap is harmless to name selection because creation still has to
        // succeed.
        let temp = CString::new(format!(".{}.{}", std::process::id(), n)).expect("numeric name");
        // A caller may have requested a destination that looks like a generated
        // staging name. Leave that leaf free for publication and try the next
        // temporary name.
        if temp.as_c_str() == target {
            continue;
        }
        // SAFETY: The caller holds the parent directory open and temp owns the NUL-terminated name.
        if unsafe { libc::mkdirat(dir, temp.as_ptr(), 0o700) } == 0 {
            // On success this process has created a private subdirectory inside
            // the open destination parent. Its leaf name is returned for later
            // descriptor-relative operations and cleanup.
            return Ok(temp);
        }
        // Only an occupied name warrants another attempt. A permission or
        // filesystem error still applies to this directory and must reach the
        // caller with its original cause.
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(error.into());
        }
    }
}

#[cfg(unix)]
fn same_entry(
    parent: libc::c_int,
    component: &std::ffi::CStr,
    expected: &std::fs::Metadata,
) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    // Compare the current directory entry with metadata obtained from the already
    // opened staging object. Ask about the link itself so a symlink does not
    // inherit the identity of its target.
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: The caller holds the parent directory open, component is a C string, and
    // stat provides aligned, writable storage for one libc::stat.
    if unsafe {
        libc::fstatat(
            parent,
            component.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } < 0
    {
        // A failed lookup can mean that the name disappeared after the descriptor
        // was opened. Keep that filesystem error distinct from the successful stat
        // of a different object reported below.
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: fstatat initialized stat on success. Its error path returned above.
    let stat = unsafe { stat.assume_init() };
    // Device and inode together identify the object. Matching a pathname or
    // comparing file lengths would also accept a different file installed under
    // the same leaf during the copy.
    if stat.st_dev as u64 != expected.dev() || stat.st_ino != expected.ino() {
        return Err(Error::new(
            ErrorCode::RecoveryError,
            "snapshot staging entry changed identity",
        ));
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn publish(
    source: &Storage,
    output: &Path,
    commit: u64,
    sequence: u64,
    context: &crate::json::JsonOperationContext,
) -> Result<(SnapshotSummary, RecordReservation, PathBuf)> {
    use std::os::unix::fs::{FileExt, MetadataExt};
    // The caller supplies the commit ID and record sequence from its writer-locked
    // source. Keep the resolved destination path for recovery diagnostics even
    // though the public summary retains the caller's spelling.
    let recovery_path = resolve_snapshot_output(source.path(), output)?;
    let mut copy_space = context.workspace_reservation();
    const COPY_BLOCK: usize = 32 * 1024;
    // Reserve the block before creating anything in the destination directory.
    // Copying and independent read-back reuse this block so the byte workspace
    // stays bounded as the database grows.
    copy_space.grow_to(COPY_BLOCK, COPY_BLOCK, "operation_scratch")?;
    // Keep the opened parent directory through final sync. All staging,
    // publication and cleanup names are resolved relative to directory descriptors
    // once this descriptor is obtained.
    let directory = File::open(recovery_path.parent().expect("resolved snapshot parent"))?;
    let dir = directory.as_raw_fd();
    let target = name(output)?;
    let temp = create_stage(dir, &target)?;
    // SAFETY: directory owns dir and temp owns its C string throughout openat.
    // These flags open an existing directory, so openat needs no mode argument.
    let stage_fd = unsafe {
        libc::openat(
            dir,
            temp.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    // Failure to open the newly created directory leaves its identity unknown.
    // Include its name in the error so a caller can locate the remnant without
    // attempting removal through an unverified pathname.
    if stage_fd < 0 {
        return Err(Error::from(std::io::Error::last_os_error()).with_detail(
            "cleanup_error",
            format!(
                "could not verify staging directory {}",
                temp.to_string_lossy()
            ),
        ));
    }
    // SAFETY: openat returned a fresh, nonnegative descriptor. stage takes sole ownership.
    let stage = unsafe { File::from_raw_fd(stage_fd) };
    let stage_identity = stage
        .metadata()
        .map_err(|error| Error::from(error).with_detail("cleanup_error", temp.to_string_lossy()))?;
    // Compare the staging name in the open parent with the directory we actually
    // opened. If the staging entry was renamed or replaced, cleanup must fail the
    // identity check and leave the current occupant in place.
    let remove_stage = || -> Result<()> {
        same_entry(dir, &temp, &stage_identity)?;
        // SAFETY: directory still owns dir, and temp keeps its C string live through unlinkat.
        if unsafe { libc::unlinkat(dir, temp.as_ptr(), libc::AT_REMOVEDIR) } < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    };
    // Keep the data under a fixed component inside the private stage. The create
    // and read-back descriptors can then use exactly the same directory without
    // rebuilding the path from the public target.
    let payload = c"contents";
    let open = |flags| -> Result<File> {
        // SAFETY: stage owns the directory descriptor, payload is a static C string, and
        // the mode argument is supplied for callers that request O_CREAT.
        let fd = unsafe {
            libc::openat(
                stage.as_raw_fd(),
                payload.as_ptr(),
                flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600 as libc::c_int,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        // SAFETY: fd is the fresh descriptor from openat, checked for failure above.
        // The returned File takes sole responsibility for closing it.
        Ok(unsafe { File::from_raw_fd(fd) })
    };
    // Exclusive creation gives the copy a fresh payload. If creation fails, try to
    // remove the still-empty staging directory and preserve the creation error
    // when cleanup also fails.
    let mut file = open(libc::O_CREAT | libc::O_EXCL | libc::O_RDWR).map_err(|error| {
        match remove_stage() {
            Err(cleanup) => error.with_detail("cleanup_error", cleanup),
            Ok(()) => error,
        }
    })?;
    let owned = file
        .metadata()
        .map_err(|error| Error::from(error).with_detail("cleanup_error", temp.to_string_lossy()))?;
    // The writable descriptor remains open while cleanup verifies the named
    // payload. Remove that file first because removing the stage requires an empty
    // directory.
    let unlink = || -> Result<()> {
        same_entry(stage_fd, payload, &owned)?;
        // SAFETY: stage remains open during unlinkat and payload is a static C string.
        if unsafe { libc::unlinkat(stage.as_raw_fd(), payload.as_ptr(), 0) } < 0
            && std::io::Error::last_os_error().kind() != std::io::ErrorKind::NotFound
        {
            return Err(std::io::Error::last_os_error().into());
        }
        remove_stage()
    };
    let verified = (|| {
        // Copy the open file so a pathname replacement cannot redirect the snapshot.
        let input = source.descriptor();
        // Capture the full physical length, including a derived index suffix when
        // present. Restrict every positional read to that extent, then check the
        // input length again during verification.
        let size = input.metadata()?.len();
        let mut writer = crate::hash::Sha256Writer::new(&mut file);
        #[cfg(test)]
        crate::maintenance_faults::check(crate::maintenance_faults::Stage::CopySnapshot)?;
        let mut offset = 0;
        let mut block = [0u8; COPY_BLOCK];
        // Positioned reads leave the source descriptor's cursor available for
        // storage operations. A short source cannot be silently accepted as a
        // shorter snapshot because each block must be read in full.
        while offset < size {
            let length = (size - offset).min(block.len() as u64) as usize;
            input.read_exact_at(&mut block[..length], offset)?;
            std::io::Write::write_all(&mut writer, &block[..length])?;
            #[cfg(feature = "native-test-support")]
            if offset == 0 {
                crate::maintenance_probe::pause("copy_snapshot")?;
            }
            offset += length as u64;
        }
        // Finish the digest of the bytes written before syncing the file. This
        // sync covers the staged contents. Publication creates a separate
        // parent-directory entry that needs its own sync later.
        let digest = writer.finish()?;
        file.sync_all()?;
        let mut verified =
            crate::physical_file::ManagedFile::open_in(context, || open(libc::O_RDONLY))?;
        #[cfg(feature = "native-test-support")]
        crate::maintenance_probe::pause("verify_snapshot")?;
        // The read-back handle must name the file just written. Compare device and
        // inode before accepting its length and digest so a same-size entry
        // substituted at the staging name cannot satisfy verification by length
        // alone.
        let metadata = verified.metadata()?;
        let original = file.metadata()?;
        if (metadata.dev(), metadata.ino()) != (original.dev(), original.ino())
            || input.metadata()?.len() != size
            || metadata.len() != size
            || crate::hash::sha256_reader_with_buffer(&mut verified, &mut block)? != digest
        {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "snapshot verification failed",
            ));
        }
        // Matching digests show that the staged copy contains the bytes read.
        // Before publishing, replay the copy and check model consistency. Match
        // its commit ID, sequence number and end of the critical log to the
        // source, then verify the remaining tail. A byte-for-byte copy could
        // otherwise pass the digest check with the same damaged tail.
        let mut catalog = Catalog::new_in(context);
        let scan = crate::storage::scan_snapshot(&mut verified, context, |record, work| {
            catalog.apply_retained(record, work, context)
        })?;
        // Reject conflicting visible models before deriving coverage. An unbound
        // collection is valid.
        catalog.last_commit_id = scan.last_valid_commit;
        catalog.require_consistent_models()?;
        // The report records the index state derived from the copied file.
        // Matching the primary commit alone would omit whether the suffix covers
        // that view, so obtain the status while the replayed catalog and copy are
        // still available.
        let index = index_suffix_status_from_scan(&scan, &catalog, context)?;
        if scan.last_valid_commit != commit
            || scan.last_valid_commit_sequence_no != sequence
            || scan.critical_end_offset != source.critical_end_offset()
            || !crate::storage::verified_file_tail(&mut verified, &scan)?
        {
            return Err(Error::new(
                ErrorCode::RecoveryError,
                "snapshot does not match the source commit and index state",
            ));
        }
        // Build the return value from the replayed copy while publication can
        // still be declined. Allocation or model lookup failure here leaves the
        // requested destination unclaimed.
        let mut work = RecordReservation::from(context.reservation());
        let mut collections = crate::json::ReservedVec::with_capacity(
            catalog.live_collections().count(),
            context,
            "operation_scratch",
        )?;
        for info in catalog.live_collections() {
            let (mut info, reservation) = copy_collection_in(info, context)?;
            work.merge(reservation);
            // A collection may have acquired a model from visible embeddings
            // without explicitly naming one in its definition. Reflect that
            // binding in the returned descriptor and retain the copied name with
            // the result.
            if let crate::catalog::ModelState::Bound(model) =
                catalog.visible_model_state(&info.collection_id)?
            {
                info.embedding_model = Some(work.copy_string(model, context)?);
            }
            collections.push(info);
        }
        // Catalog iteration does not define the summary order. Sorting the owned
        // descriptors by name keeps clients from observing changes in the
        // catalog's internal iteration order.
        collections
            .as_mut_slice()
            .sort_unstable_by(|a, b| a.name.cmp(&b.name));
        let (collections, vector_credit) = collections.into_parts();
        work.merge_workspace(vector_credit);
        let summary = SnapshotSummary {
            schema_version: "0.1".into(),
            source_path: source.path().to_owned(),
            output_path: output.to_owned(),
            sha256: digest,
            collections,
            bytes_copied: size,
            last_valid_commit: commit,
            critical_end_offset: scan.critical_end_offset,
            index_suffix_status: index.as_str().into(),
        };
        // Store the recovery facts before linking the public name. After a
        // successful link, an error syncing or cleaning up must still identify
        // which target, bytes and source commit were published.
        context.maintenance.update(|state| {
            state.prepare_snapshot(crate::maintenance_progress::SnapshotFacts {
                target: recovery_path.to_string_lossy().into_owned(),
                sha256: summary.sha256.clone(),
                size,
                commit,
            })
        });
        // Linking publishes the verified inode and atomically refuses an existing target.
        same_entry(stage_fd, payload, &owned)?;
        // SAFETY: stage and directory own the two directory descriptors, and payload
        // and target keep both NUL-terminated component names live throughout linkat.
        if unsafe { libc::linkat(stage.as_raw_fd(), payload.as_ptr(), dir, target.as_ptr(), 0) } < 0
        {
            let err = std::io::Error::last_os_error();
            // Another creator can claim the target after the initial path check.
            // The link operation decides ownership at publication time and reports
            // an existing name through the same public error as the preflight
            // check.
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                return Err(Error::new(
                    ErrorCode::InvalidSearchParams,
                    "snapshot output path already exists",
                ));
            }
            return Err(err.into());
        }
        // At this point the target has its own link to the verified inode.
        // Removing the staging link later cannot retract publication, but the
        // parent-directory change still needs to reach durable storage.
        context.maintenance.update(|state| state.published());
        Ok((summary, work))
    })();
    // Attempt cleanup on both the successful and failed verification paths. If
    // verification failed, retain that primary error and attach the cleanup
    // failure so the caller can locate a leftover stage.
    let cleanup = unlink();
    let (summary, work) = verified.map_err(|mut err| {
        if let Err(failure) = &cleanup {
            err = err.with_detail("cleanup_error", failure);
        }
        err.with_detail("stage", "snapshot_publish")
            .with_detail("effect", "unchanged")
    })?;
    // Publication has already happened. Cleanup or directory sync can fail
    // with the target still visible and its durability unconfirmed.
    cleanup
        .and_then(|_| {
            #[cfg(test)]
            crate::maintenance_faults::check(
                crate::maintenance_faults::Stage::SyncSnapshotDirectory,
            )?;
            #[cfg(feature = "native-test-support")]
            crate::maintenance_probe::pause("snapshot_finalize")?;
            directory.sync_all().map_err(Error::from)
        })
        // A finalization error is observed after the target has become visible.
        // Attach the verified digest, size and commit so recovery can recognize
        // this snapshot without blindly retrying the create.
        .map_err(|err| {
            err.with_detail("stage", "snapshot_finalize")
                .with_detail("effect", "snapshot_published")
                .with_detail("durability", "unknown")
                .with_detail("target", recovery_path.display())
                .with_detail("sha256", &summary.sha256)
                .with_detail("size_bytes", summary.bytes_copied)
                .with_detail("last_valid_commit", commit)
        })?;
    // Mark durability confirmed only after both staging cleanup and the
    // parent-directory sync succeed. A caller receiving the summary can therefore
    // distinguish completion from the published error path.
    context
        .maintenance
        .update(|state| state.confirmed_snapshot());
    Ok((summary, work, recovery_path))
}

// Snapshot publication relies on Unix filesystem operations with explicit
// exclusive publication and directory synchronization. Keep the same entry point
// on other platforms so the caller receives a feature refusal before any output
// can be staged there.
#[cfg(not(unix))]
pub(super) fn publish(
    _: &Storage,
    _: &Path,
    _: u64,
    _: u64,
    _: &crate::json::JsonOperationContext,
) -> Result<(SnapshotSummary, RecordReservation, PathBuf)> {
    Err(Error::new(
        ErrorCode::UnsupportedFeatureFlag,
        "atomic snapshot publication is unsupported",
    ))
}
