use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::json::{JsonOperationContext, WorkspaceReservation};
use crate::Result;

#[derive(Debug, Default)]
// A temporary pathname can remain owned by a guard or a deferred cleanup after its
// creating function returns. Keep its workspace lease alongside the native path
// for that entire lifetime, including any unused buffer capacity.
pub(crate) struct TemporaryPath {
    // Owned native path used for filesystem calls. Clearing the name disables a
    // pending cleanup while retaining the allocation until drop.
    value: PathBuf,
    // Credit for the path buffer's allocated byte capacity. Absent for a default
    // path or a path explicitly adopted without an operation reservation.
    _work: Option<WorkspaceReservation>,
}

impl TemporaryPath {
    // Term-statistics collection uses the system temporary directory when it needs
    // spill files. Keeping the directory itself on the operation budget accounts
    // for it even before any run file exists.
    pub(crate) fn system_temp_in(context: &JsonOperationContext) -> Result<Self> {
        let mut work = context.workspace_reservation();
        // The OS/standard-library discovery allocation precedes adoption; its
        // resident capacity, not just its path length, remains charged here.
        let value = std::env::temp_dir();
        // If admission fails, the discovered path is still local and drops on
        // return. No path backed only by a failed workspace request escapes to the
        // caller.
        work.grow_to(value.capacity(), value.capacity(), "operation_scratch")?;
        Ok(Self {
            value,
            _work: Some(work),
        })
    }

    // The unbudgeted file constructor already owns its PathBuf. Move that
    // allocation into its guard without entering the operation ledger.
    pub(super) fn unreserved(value: PathBuf) -> Self {
        Self { value, _work: None }
    }

    pub(super) fn copy_in(value: &Path, context: &JsonOperationContext) -> Result<Self> {
        // External sorting owns the directory for the lifetime of its runs. Make
        // an independent catalog-build copy rather than keeping a borrow of the
        // caller's path.
        Self::copy_with_capacity(
            value,
            // A borrowed path carries no capacity information. Use its native
            // encoded length as the requested copy size so non-UTF-8 directory
            // names remain intact.
            value.as_os_str().as_encoded_bytes().len(),
            context,
            "catalog_build",
        )
    }

    #[cfg(unix)]
    pub(super) fn copy_reserved(
        &self,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        // Make the cleanup owner's copy independently payable for the full source
        // capacity. The original and copy can stay live for different lengths of
        // time and cannot share one buffer lease.
        Self::copy_with_capacity(&self.value, self.value.capacity(), context, phase)
    }

    fn copy_with_capacity(
        value: &Path,
        capacity: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        // Check the supplied bound before allocating. An oversized source must
        // never reach OsString::push, which would silently grow its buffer outside
        // the admitted capacity.
        if value.as_os_str().as_encoded_bytes().len() > capacity {
            return Err(std::io::Error::other("path copy exceeds admitted capacity").into());
        }
        // Obtain an empty native buffer together with its grant. The copy borrows
        // the source, so an allocation or capacity error can discard this attempt
        // without disturbing the directory the caller owns.
        let (mut bytes, work) = Self::allocate(capacity, context, phase)?;
        bytes.push(value.as_os_str());
        // Moving the OsString into a PathBuf should keep the admitted backing
        // allocation. Verify the observed capacity after the copy before returning
        // the new owner.
        Self::from_parts(bytes, capacity, work)
    }

    #[cfg(unix)]
    // Build a temporary path in a buffer charged for its full capacity. The caller
    // provides a bound for the directory and a single generated file name.
    pub(super) fn join_name(
        directory: &Path,
        name: &str,
        capacity: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        // Count the separator only when it will be written. Check native encoded
        // byte lengths before allocating so paths with non-UTF-8 directory
        // components use the same bound as the copy.
        let directory_bytes = directory.as_os_str().as_encoded_bytes();
        let separator = usize::from(directory_bytes.last().is_some_and(|&byte| byte != b'/'));
        let required = directory_bytes
            .len()
            .checked_add(separator)
            .and_then(|bytes| bytes.checked_add(name.len()));
        // Treat overflow like a path outside the admitted bound. Reject either
        // before creating an allocation or copying a prefix.
        if required.map_or(true, |bytes| bytes > capacity) {
            return Err(std::io::Error::other("path join exceeds admitted capacity").into());
        }
        let (mut bytes, work) = Self::allocate(capacity, context, phase)?;
        bytes.push(directory.as_os_str());
        // Insert a slash when a nonempty directory does not already end in one. An
        // empty directory leaves the generated name relative.
        if directory
            .as_os_str()
            .as_encoded_bytes()
            .last()
            .is_some_and(|&byte| byte != b'/')
        {
            // The write and its length check use the same separator rule. Changing
            // the join rule also requires changing the preflight count above.
            bytes.push("/");
        }
        bytes.push(name);
        Self::from_parts(bytes, capacity, work)
    }

    // Return an empty OsString with its workspace owner. Reserve the full bound
    // before the allocator is called, so budget refusal has no buffer to unwind.
    fn allocate(
        capacity: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<(OsString, WorkspaceReservation)> {
        let mut work = context.workspace_reservation();
        work.grow_to(capacity, capacity, phase)?;
        let mut bytes = OsString::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        // Reject any spare capacity beyond the grant. The local buffer and guard
        // then drop together before a pathname can escape.
        if bytes.capacity() != capacity {
            return Err(std::io::Error::other("path allocation exceeds admitted capacity").into());
        }
        Ok((bytes, work))
    }

    // Recheck capacity after copying or joining components. That prevents the
    // returned path from carrying an allocation larger than the reservation that
    // follows it.
    fn from_parts(bytes: OsString, capacity: usize, work: WorkspaceReservation) -> Result<Self> {
        if bytes.capacity() != capacity {
            // Free the buffer before returning its charge. Other operations can
            // consume that credit once the release reaches the shared ledger.
            drop(bytes);
            drop(work);
            return Err(std::io::Error::other("path growth exceeds admitted capacity").into());
        }
        Ok(Self {
            value: PathBuf::from(bytes),
            _work: Some(work),
        })
    }

    // Clear the visible pathname without discarding its backing allocation.
    // Cleanup can disarm the path while its workspace remains charged until drop.
    pub(super) fn clear(&mut self) {
        self.value.clear();
    }
}

impl std::ops::Deref for TemporaryPath {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.value
    }
}

impl AsRef<Path> for TemporaryPath {
    fn as_ref(&self) -> &Path {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use std::sync::Arc;

    fn context(limit: usize) -> JsonOperationContext {
        JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, limit, limit)))
    }

    #[test]
    fn system_temp_adoption_retains_actual_capacity_and_returns_it_on_rejection_or_drop() {
        let expected = std::env::temp_dir();
        let needed = expected.capacity();
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (needed as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    0,
                    if parent { limit } else { 2 * limit },
                )));
                let result = TemporaryPath::system_temp_in(&context);
                if delta < 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), needed.to_string())));
                } else {
                    let path = result.unwrap();
                    assert_eq!(path.as_ref(), expected.as_path());
                    assert_eq!(path._work.as_ref().unwrap().reserved_bytes(), needed);
                    assert_eq!(context.available_workspace_bytes(), limit - needed);
                }
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn path_copy_admits_exact_capacity_and_clear_keeps_ownership() {
        let input = Path::new("cofferdb-sort-directory");
        let bytes = input.as_os_str().as_encoded_bytes().len();
        for available in [bytes - 1, bytes, bytes + 1] {
            let context = context(available);
            let result = TemporaryPath::copy_in(input, &context);
            if available < bytes {
                assert_eq!(
                    result.unwrap_err().code(),
                    crate::ErrorCode::ResourceExhausted
                );
            } else {
                let mut path = result.unwrap();
                assert_eq!(path.as_ref(), input);
                assert_eq!(path.value.capacity(), bytes);
                path.clear();
                assert!(path.as_os_str().is_empty());
                assert_eq!(path.value.capacity(), bytes);
                let mut spare = context.workspace_reservation();
                spare
                    .grow_to(available - bytes, available - bytes, "catalog_build")
                    .unwrap();
                assert!(spare
                    .grow_to(
                        available - bytes + 1,
                        available - bytes + 1,
                        "catalog_build"
                    )
                    .is_err());
                drop(spare);
                drop(path);
            }
            context
                .workspace_reservation()
                .grow_to(available, available, "catalog_build")
                .unwrap();
        }
    }

    #[test]
    #[cfg(unix)]
    fn native_join_preserves_bytes_and_rejects_undersized_capacity_before_growth() {
        use std::os::unix::ffi::OsStringExt;
        for bytes in [
            b"".as_slice(),
            b"/",
            b"//",
            b"//tmp/",
            b"relative",
            b"/tmp/\xff",
        ] {
            let directory = PathBuf::from(OsString::from_vec(bytes.to_vec()));
            let expected = directory.join("run.tmp");
            let capacity = expected.as_os_str().as_encoded_bytes().len() + 19;
            let context = context(2 * capacity);
            let path = TemporaryPath::join_name(
                &directory,
                "run.tmp",
                capacity,
                &context,
                "catalog_build",
            )
            .unwrap();
            assert_eq!(path.as_ref(), expected);
            assert_eq!(
                path.as_os_str().as_encoded_bytes(),
                expected.as_os_str().as_encoded_bytes()
            );
            let copy = path.copy_reserved(&context, "catalog_build").unwrap();
            assert_eq!(copy.value.capacity(), capacity);
            assert_eq!(copy.as_ref(), path.as_ref());
            assert_eq!(
                copy.as_os_str().as_encoded_bytes(),
                expected.as_os_str().as_encoded_bytes()
            );
            assert!(context
                .workspace_reservation()
                .grow_to(1, 1, "catalog_build")
                .is_err());
            drop((copy, path));
            assert_eq!(
                TemporaryPath::join_name(
                    &directory,
                    "run.tmp",
                    expected.as_os_str().as_encoded_bytes().len() - 1,
                    &context,
                    "catalog_build"
                )
                .unwrap_err()
                .code(),
                crate::ErrorCode::IoError
            );
            context
                .workspace_reservation()
                .grow_to(2 * capacity, 2 * capacity, "catalog_build")
                .unwrap();
        }
    }
}
