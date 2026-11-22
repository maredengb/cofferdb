// A failed attempt to unlink a staged file must leave its bytes charged, even when
// the guard that created it is gone. The budget keeps the pathname, open file and
// charge as one pending entry until the original object can be released.
use super::TemporaryPath;
use std::fs::File;
use std::sync::{Arc, Mutex};

use crate::{Error, ErrorCode, Result};

#[derive(Debug)]
pub(crate) struct TemporaryDiskBudget {
    // Maximum temporary disk bytes that all reservations sharing this budget can
    // hold at once.
    limit: u64,
    // Serializes changes to the total charge and the list of files whose cleanup
    // must still release space.
    state: Mutex<DiskState>,
}

#[derive(Debug, Default)]
struct DiskState {
    // Bytes charged to active reservations and deferred cleanups. A failed unlink
    // leaves the total unchanged until a retry closes the original file.
    used: u64,
    #[cfg(test)]
    peak: u64,
    // Files awaiting another unlink attempt. Their descriptors and charges survive
    // the drop of the original guard.
    pending: PendingCleanups,
}

#[derive(Debug)]
struct PendingCleanup {
    // Owned name to check against the still-open file when retrying removal.
    path: TemporaryPath,
    // Pins the original file identity across retries and is closed before its disk
    // charge is refunded.
    file: File,
    // Bytes transferred from the failed file's reservation and still included in
    // DiskState::used.
    bytes: u64,
}

#[derive(Default)]
// A live file owns an empty retry node alongside its disk reservation. On failure
// it supplies both the cleanup information and the storage needed to queue it, so
// a destructor can transfer ownership under memory pressure.
pub(super) struct PendingCleanupNode {
    // Empty while the node is a spare owned by a live guard. Filled only when that
    // guard transfers a failed cleanup to the budget.
    cleanup: Option<PendingCleanup>,
    // Owns the next node in the pending list. Pop detaches the link so the removed
    // node can be retried or dropped independently.
    next: Option<PendingCleanupOwner>,
}

#[derive(Debug, Default)]
pub(super) struct PendingCleanupOwner {
    // Heap storage allocated before file creation, ready for insertion in the retry
    // list from Drop.
    node: Box<PendingCleanupNode>,
    // Optional charge for the node allocation. The node is dropped before the
    // workspace allowance is returned.
    _work: Option<crate::json::WorkspaceReservation>,
}

impl PendingCleanupOwner {
    #[cfg(unix)]
    pub(super) fn new(
        context: &crate::json::JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        // Charge the link node to the creating operation's workspace before
        // allocating it. The open file's contents are accounted against a separate
        // disk budget and may later have a different owner.
        let bytes = std::mem::size_of::<PendingCleanupNode>();
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, phase)?;
        Ok(Self {
            node: Box::default(),
            _work: Some(work),
        })
    }
}

#[derive(Default)]
struct PendingCleanups {
    // First deferred file. A missing head means no file is awaiting another
    // attempt.
    head: Option<PendingCleanupOwner>,
    // Number of linked nodes, maintained when pushing or detaching entries and used
    // for diagnostics.
    len: usize,
}

impl std::fmt::Debug for PendingCleanupNode {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print just this entry's cleanup payload. Following its next link from
        // here would recursively format every later file and could exhaust the
        // stack on a long retry list.
        out.debug_struct("PendingCleanupNode")
            .field("cleanup", &self.cleanup)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PendingCleanups {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Report queue size without walking through every retained path and
        // descriptor when a shared disk budget is formatted.
        out.debug_struct("PendingCleanups")
            .field("len", &self.len)
            .finish()
    }
}

impl PendingCleanups {
    fn push(&mut self, mut owner: PendingCleanupOwner) {
        // Insertion expects one detached node. Reusing a link already pointing to
        // another entry would splice an uncounted chain into the queue and break
        // the length invariant.
        assert!(owner.node.next.is_none());
        owner.node.next = self.head.take();
        self.head = Some(owner);
        self.len += 1;
    }

    fn pop(&mut self) -> Option<PendingCleanupOwner> {
        // Detach the next owner before handing the head to the caller. A failed
        // unlink can then requeue just this node, while a successful one can
        // destroy it without closing unrelated descriptors.
        let mut owner = self.head.take()?;
        self.head = owner.node.next.take();
        self.len -= 1;
        Some(owner)
    }
}

impl Drop for PendingCleanups {
    fn drop(&mut self) {
        // Unlink the list iteratively so dropping many failed cleanups does not
        // recurse through the entire chain of owned nodes.
        while self.pop().is_some() {}
    }
}

impl DiskState {
    fn retry_cleanup(&mut self) -> std::io::Result<()> {
        // An unlink failure affects just that file. Remember the first error for
        // the caller while still recovering space from every other removable
        // entry.
        let mut first_error = None;
        // Attempt each pending file once. A failed unlink keeps its descriptor,
        // path and disk charge together so a later cleanup can retry safely.
        let mut pending = std::mem::take(&mut self.pending);
        // Collect failures separately while draining the old list. Putting a
        // failed node straight back into the list being consumed could retry it
        // indefinitely on the same call.
        let mut failed = PendingCleanups::default();
        while let Some(node) = pending.pop() {
            // Only nodes transferred by retain_after_cleanup_failure are linked
            // here. Their descriptor pins the original file, so a reused pathname
            // alone cannot establish which inode is being released on Unix.
            let owner = node.node.cleanup.as_ref().expect("active cleanup node");
            match super::remove_owned_file(&owner.file, &owner.path) {
                Ok(()) => {
                    let bytes = owner.bytes;
                    drop(node); // Close the final descriptor before returning its charge.
                    self.used -= bytes;
                }
                Err(error) => {
                    // Keep the failed node intact, including its disk bytes and
                    // its workspace reservation. A later pass must be able to
                    // retry even if the original file guard has disappeared.
                    first_error.get_or_insert(error);
                    failed.push(node);
                }
            }
        }
        // Install the surviving list before returning the first error. A partially
        // successful pass has already reduced used for the nodes it removed.
        self.pending = failed;
        first_error.map_or(Ok(()), Err)
    }
}

impl Default for TemporaryDiskBudget {
    fn default() -> Self {
        // Disk staging is independent of resident cache; allow input and encoded copies
        // for the v0.1 2-GiB calibration while retaining one finite runtime-wide ceiling.
        Self::new(4 * 1024 * 1024 * 1024)
    }
}

impl TemporaryDiskBudget {
    pub(crate) fn limit_bytes(&self) -> u64 {
        self.limit
    }

    // Each budget starts with its own accounting and retry list. Cloning an Arc
    // lets multiple temporary files compete under this particular limit.
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            state: Mutex::new(DiskState::default()),
        }
    }

    // A successful check only observes the current headroom. Each writer still
    // needs a reservation to claim space before issuing its writes.
    pub(crate) fn check_additional(&self, bytes: u64) -> Result<()> {
        // Preflight does not issue filesystem I/O to reclaim space. Admission may
        // later retry pending cleanups under pressure and must account for
        // intervening grants.
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.available_after_check(state.used, bytes).map(|_| ())
    }

    pub(crate) fn retry_cleanup(&self) -> Result<()> {
        // Hold the disk-state lock across the unlink attempts and refunds. A
        // concurrent grant must not see the returned bytes until the matching
        // descriptor has been closed.
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retry_cleanup()?;
        Ok(())
    }

    fn available_after_check(&self, used: u64, minimum: u64) -> Result<u64> {
        // used includes pending removals as well as live reservations, and every
        // positive grant is bounded by remaining space. Subtract before comparing
        // so a request near u64::MAX cannot wrap the test.
        let available = self.limit - used;
        if minimum > available {
            return Err(Error::resource_exhausted(
                "temporary_disk",
                "temporary_staging",
                self.limit,
                // The diagnostic is the total the budget would need. Widen both
                // operands before adding, since an overflowing request must remain
                // reportable.
                u128::from(used) + u128::from(minimum),
            ));
        }
        // Return the entire available headroom when the minimum fits. A caller
        // taking a preferred buffer can then clamp its grant to the same locked
        // observation.
        Ok(available)
    }

    // Making an owner does not consume disk credit. Its Arc associates future
    // growth, release and deferred cleanup with this budget even if the caller
    // drops its own reference.
    pub(super) fn reservation(self: &Arc<Self>) -> DiskReservation {
        DiskReservation {
            budget: Arc::clone(self),
            bytes: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn used(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .used
    }

    #[cfg(test)]
    pub(crate) fn peak_used(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .peak
    }
}

impl Drop for TemporaryDiskBudget {
    fn drop(&mut self) {
        // Final best effort at runtime shutdown; pending entries contain no Arc
        // back to the ledger, so failures cannot keep a runtime alive forever.
        // No reservation can still hold an Arc when the budget itself is dropped.
        // Try the remaining entries using exclusive state access. Shutdown has no
        // caller to receive an unlink error.
        let _ = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .retry_cleanup();
    }
}

#[derive(Debug)]
pub(crate) struct DiskReservation {
    // Runtime-wide budget to which these bytes belong. Transfers between file
    // guards require both reservations to share this instance.
    budget: Arc<TemporaryDiskBudget>,
    // Bytes of the shared total for which this owner must eventually account.
    // Transfer adjusts the owners without changing the total.
    bytes: u64,
}

impl DiskReservation {
    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.bytes
    }

    // Reduce the caller's credit to a known retained size. A caller must establish
    // the retained size before releasing the unused portion.
    pub(crate) fn retain(&mut self, bytes: u64) {
        assert!(bytes <= self.bytes);
        self.release(self.bytes - bytes);
    }

    pub(super) fn grow(&mut self, bytes: u64) -> Result<()> {
        self.grow_at_least(bytes, bytes).map(|_| ())
    }

    // Return the actual grant for a buffer that can work with less than its
    // preferred size. The mutex covers both the retry decision and the final
    // admission, so another writer cannot take the freed bytes between checks.
    pub(super) fn grow_at_least(&mut self, minimum: u64, preferred: u64) -> Result<u64> {
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Only a shortage of required bytes justifies retrying failed file cleanups.
        // A caller asking for a larger buffer can use the available grant as long as
        // its minimum fits.
        if minimum > self.budget.limit - state.used {
            state.retry_cleanup()?;
        }
        // Recompute available credit after any pending cleanup. A refusal here
        // leaves this owner unchanged, even if cleanup already released bytes
        // belonging to another file.
        let available = self.budget.available_after_check(state.used, minimum)?;
        // Clamp the request between the required minimum and the latest headroom.
        // A preference smaller than the minimum still requests the minimum.
        let granted = preferred.max(minimum).min(available);
        state.used += granted;
        #[cfg(test)]
        {
            state.peak = state.peak.max(state.used);
        }
        // Record the same increment locally so Drop or publication can return
        // exactly this owner's contribution later.
        self.bytes += granted;
        Ok(granted)
    }

    // Return part of this owner's charge under the shared limit lock. The
    // assertion guards against turning someone else's reservation into available
    // capacity.
    pub(super) fn release(&mut self, bytes: u64) {
        assert!(bytes <= self.bytes);
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.used -= bytes;
        self.bytes -= bytes;
    }

    pub(super) fn retain_after_cleanup_failure(
        mut self,
        path: TemporaryPath,
        file: File,
        mut node: PendingCleanupOwner,
    ) {
        assert!(node.node.cleanup.is_none());
        // A failed unlink can leave real disk usage behind. Move the path, open file
        // and its charge to the pending list together. Dropping this guard must not
        // refund bytes that a later cleanup still has to reclaim.
        node.node.cleanup = Some(PendingCleanup {
            path,
            file,
            bytes: self.bytes,
        });
        // Link the fully initialized owner into the pending list under the budget
        // lock. The shared total stays charged until retry confirms removal,
        // regardless of when this consumed reservation drops.
        self.budget
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pending
            .push(node);
        // Clear ownership only after queue insertion. The pending entry now
        // carries the same charge, leaving this guard with nothing to refund.
        self.bytes = 0;
    }

    pub(super) fn transfer_to(&mut self, destination: &mut Self, bytes: u64) -> Result<()> {
        // Reject a move across ledgers or beyond the source's current balance
        // before mutating either guard. A cross-ledger move would require separate
        // destination admission.
        if !Arc::ptr_eq(&self.budget, &destination.budget) || bytes > self.bytes {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "invalid temporary disk reservation transfer",
            ));
        }
        // Both guards charge the same budget. Move responsibility for the eventual
        // refund between them without changing the shared total or opening a window
        // for another writer to spend those bytes.
        self.bytes -= bytes;
        destination.bytes += bytes;
        Ok(())
    }
}

// Ordinary destruction returns the remaining local charge. A transferred owner
// retains only the amount it still owns.
impl Drop for DiskReservation {
    fn drop(&mut self) {
        self.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_pending_lists_drop_iteratively_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                let mut pending = PendingCleanups::default();
                for _ in 0..100_000 {
                    pending.push(PendingCleanupOwner::default());
                }
                assert_eq!(pending.len, 100_000);
                let (_, allocations) = crate::test_alloc::allocations(|| drop(pending));
                assert_eq!(allocations, 0);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn preflight_is_read_only_and_actual_reservations_recheck_changed_usage() {
        for limit in [0, 1, 1024, u64::MAX] {
            let budget = Arc::new(TemporaryDiskBudget::new(limit));
            budget.check_additional(limit).unwrap();
            assert_eq!(budget.used(), 0);
            let mut other = budget.reservation();
            other.grow(limit).unwrap();
            budget.check_additional(0).unwrap();
            let preflight = budget.check_additional(1).unwrap_err();
            let mut actual = budget.reservation();
            let allocation = actual.grow(1).unwrap_err();
            assert_eq!(preflight.code(), ErrorCode::ResourceExhausted);
            assert_eq!(preflight.to_json(), allocation.to_json());
            assert!(preflight
                .detail()
                .iter()
                .any(|(key, value)| key == "requested_bytes"
                    && value == &(u128::from(limit) + 1).to_string()));
            assert_eq!(budget.used(), limit);
            drop(other);
            budget.check_additional(limit).unwrap();
            actual.grow(limit).unwrap();
            drop(actual);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn transfers_preserve_shared_usage_and_reject_invalid_owners_or_amounts() {
        for limit in [0, 1, 1024, u64::MAX] {
            for seed in 0..100 {
                let budget = Arc::new(TemporaryDiskBudget::new(limit));
                let other = Arc::new(TemporaryDiskBudget::new(limit));
                let mut source = budget.reservation();
                let mut destination = budget.reservation();
                let mut foreign = other.reservation();
                source.grow(limit).unwrap();
                let moved = (limit / 100) * seed;
                source.transfer_to(&mut destination, moved).unwrap();
                assert_eq!(source.bytes, limit - moved);
                assert_eq!(destination.bytes, moved);
                assert_eq!(budget.used(), limit);
                assert!(source.transfer_to(&mut foreign, 0).is_err());
                if let Some(excess) = source.bytes.checked_add(1) {
                    assert!(source.transfer_to(&mut destination, excess).is_err());
                }
                assert_eq!(budget.used(), limit);
                assert_eq!(other.used(), 0);
                drop(source);
                assert_eq!(budget.used(), moved);
                drop(destination);
                assert_eq!(budget.used(), 0);
            }
        }
    }

    #[test]
    fn reservations_share_the_limit_and_release_exactly_across_errors_and_overflow() {
        for limit in [0, 1, 1024, u64::MAX] {
            let budget = Arc::new(TemporaryDiskBudget::new(limit));
            let mut first = budget.reservation();
            let mut second = budget.reservation();
            first.grow(limit / 2).unwrap();
            second.grow(limit - limit / 2).unwrap();
            assert_eq!(budget.used(), limit);
            let error = first.grow(1).unwrap_err();
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "requested_bytes"
                    && value == &(u128::from(limit) + 1).to_string()));
            assert_eq!(budget.used(), limit);
            drop(first);
            assert_eq!(budget.used(), limit - limit / 2);
            second.grow(limit / 2).unwrap();
            assert_eq!(budget.used(), limit);
            drop(second);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn concurrent_operations_cannot_multiply_the_runtime_disk_limit() {
        let budget = Arc::new(TemporaryDiskBudget::new(4096));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let budget = Arc::clone(&budget);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    let mut work = budget.reservation();
                    work.grow(512).unwrap();
                    barrier.wait();
                    assert_eq!(budget.used(), 4096);
                    assert!(work.grow(1).is_err());
                    barrier.wait();
                    drop(work);
                    for iteration in 0..100 {
                        let mut work = budget.reservation();
                        let count = 1 + (worker * 17 + iteration * 31) % 512;
                        work.grow(count).unwrap();
                        assert!(budget.used() <= 4096);
                    }
                });
            }
        });
        assert_eq!(budget.used(), 0);
    }
}
