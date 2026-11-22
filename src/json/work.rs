use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

use super::{JsonInputKind, JsonLimitReason, JsonProcessingPolicyV1};
use crate::{Error, ErrorCode, Result};

pub(crate) struct WorkLedger {
    // Process in which the ledger and its mutex were created. A forked handle must
    // not interact with synchronization state inherited from its parent.
    pid: u32,
    // Shared counters and ceilings updated under the same lock, including when
    // ownership moves between budget categories.
    state: Mutex<LedgerState>,
}

// JSON and workspace share the working limit. Catalog allocations and cache
// entries have their own admission checks; all four also count toward the
// total. A buffer moving from scratch into retained storage is still the same
// allocation. Transfer its credit under the ledger lock so another operation
// can never spend the bytes in between release and reacquisition.
struct LedgerState {
    // Maximum combined JSON and general workspace bytes across operations attached
    // to this ledger.
    limit: usize,
    // Maximum combined bytes for JSON, workspace, retained catalog and cache.
    // Individual categories can still reject a request below this total.
    total_limit: usize,
    // Separate ceiling for bytes promoted to the retained catalog.
    catalog_limit: usize,
    // JSON bytes charged across attached operations, including spare bytes of
    // active reservations.
    used: usize,
    // General workspace bytes currently charged to attached operations. Together
    // with used, these consume the shared working limit.
    workspace_used: usize,
    // Resident catalog bytes charged to this ledger. Transferring storage into the
    // catalog changes its category without allocating it a second time.
    catalog_used: usize,
    // Resident cache bytes charged across entries and collections, also subject to
    // the cache limit supplied by the caller.
    cache_used: usize,
}

impl Default for WorkLedger {
    fn default() -> Self {
        // Working memory retains its existing ceiling. Persistent state has a
        // separate quota; neither limit is a promise about process RSS.
        Self::with_limits(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            Self::CATALOG_LIMIT_BYTES,
            Self::TOTAL_LIMIT_BYTES,
        )
    }
}

impl WorkLedger {
    pub(crate) const CATALOG_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
    const TOTAL_LIMIT_BYTES: usize = 3 * 1024 * 1024 * 1024;

    #[cfg(test)]
    fn new(limit: usize) -> Self {
        Self::with_limits(limit, limit, limit)
    }

    // Limits describe separate ceilings, not reserved pools of memory. Several
    // operations may share this ledger, so an otherwise small request can fail
    // because other owners already consumed its working or total headroom.
    pub(crate) fn with_limits(limit: usize, catalog_limit: usize, total_limit: usize) -> Self {
        Self {
            // Record the creating process before the first reservation exists.
            // Drop paths compare this identity without acquiring the mutex, which
            // can be inherited in a locked state after a fork.
            pid: std::process::id(),
            state: Mutex::new(LedgerState {
                limit,
                total_limit,
                catalog_limit,
                // A new ledger starts with no outstanding bytes in any category.
                // Limits may deliberately be zero, so a zero-byte owner can still
                // be meaningful without promising that its first growth will
                // succeed.
                used: 0,
                workspace_used: 0,
                catalog_used: 0,
                cache_used: 0,
            }),
        }
    }
}

impl LedgerState {
    fn try_admit_cache(&mut self, additional: u128, limit: usize) -> bool {
        // Requests can be wider than a host allocation size. Reject an
        // individually impossible request before adding it to resident cache
        // credit or narrowing it for the stored counter.
        if additional > limit as u128 {
            return false;
        }
        let cache = self.cache_used as u128 + additional;
        let total =
            cache + self.used as u128 + self.workspace_used as u128 + self.catalog_used as u128;
        // Enforce the caller's cache allowance against all resident cache bytes on
        // this ledger, then include JSON, workspace and catalog bytes in the
        // parent check. Optional admission reports a miss with no counter update.
        if cache > limit as u128 || total > self.total_limit as u128 {
            return false;
        }
        // Both cache and parent limits are host-sized. After the wide comparisons
        // pass, narrowing the accepted cache total to usize cannot discard any
        // part of the requested allocation.
        self.cache_used = cache as usize;
        true
    }

    fn working_available(&self) -> usize {
        // Working headroom can be limited by either the working pool itself or
        // other categories using the parent total. Taking the smaller remainder
        // enforces both invariants when a caller asks for a preferred buffer.
        let working = self.used + self.workspace_used;
        (self.limit - working).min(self.total_limit - working - self.catalog_used - self.cache_used)
    }

    fn admit_workspace(&self, extra: u128, phase: &'static str) -> Result<()> {
        // Saturating wide addition keeps an unrepresentably large request on the
        // rejection path. Resource errors carry the attempted combined usage and
        // the phase that made the request.
        let working = (self.used as u128 + self.workspace_used as u128).saturating_add(extra);
        // Workspace competes with JSON input and results for the same working
        // pool. Catalog and cache use distinct categories that enter only the
        // later check against the total ceiling.
        if working > self.limit as u128 {
            return Err(Error::resource_exhausted(
                "memory",
                phase,
                self.limit as u64,
                working,
            ));
        }
        // Only consult the parent ceiling after the working request fits. If both
        // ceilings would reject the same allocation, the first error identifies
        // the more local working limit.
        let total = working + self.catalog_used as u128 + self.cache_used as u128;
        if total > self.total_limit as u128 {
            return Err(Error::resource_exhausted(
                "memory",
                phase,
                self.total_limit as u64,
                total,
            ));
        }
        Ok(())
    }

    fn admit_catalog(&self, additional: u128, transferred: usize) -> Result<()> {
        let catalog = self.catalog_used as u128 + additional;
        // The catalog has its own retained quota even for memory that was already
        // admitted as working storage. Check the full destination amount before
        // discounting a transfer from total usage.
        if catalog > self.catalog_limit as u128 {
            return Err(Error::resource_exhausted(
                "catalog",
                "catalog_build",
                self.catalog_limit as u64,
                catalog,
            ));
        }
        // Subtract only bytes already held on the same ledger. A fresh catalog
        // allocation increases parent usage. Promoting a precharged allocation
        // changes its category while leaving that portion of parent usage
        // constant.
        let total =
            self.used as u128 + self.workspace_used as u128 + catalog + self.cache_used as u128
                - transferred as u128;
        // A catalog-specific refusal reports resource catalog. Fresh catalog
        // growth can fit that quota yet report resource memory when other
        // categories have filled the parent ledger. A pure ownership transfer
        // leaves total usage unchanged.
        if total > self.total_limit as u128 {
            return Err(Error::resource_exhausted(
                "memory",
                "catalog_build",
                self.total_limit as u64,
                total,
            ));
        }
        Ok(())
    }
}

pub(crate) struct CacheReservation {
    // Ledger against which this cache allocation was admitted. Dropping or
    // shrinking it must refund the same ledger.
    ledger: Arc<WorkLedger>,
    // Bytes this owner currently holds as cache credit, including transfers from a
    // workspace reservation.
    bytes: usize,
}

impl CacheReservation {
    // Two caches with identical configured limits can still belong to different
    // databases. Require ledger identity before moving credit between their local
    // owners.
    pub(crate) fn assert_same_ledger(&self, other: &Self) {
        assert!(Arc::ptr_eq(&self.ledger, &other.ledger));
    }

    pub(crate) fn try_grow(&mut self, additional: u128, limit: usize) -> bool {
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Global admission and the owner's byte count advance together. A false
        // return leaves the previous reservation intact so a cache can keep
        // serving from it or choose to evict it.
        if !ledger.try_admit_cache(additional, limit) {
            return false;
        }
        self.bytes += additional as usize;
        true
    }

    pub(crate) fn transfer_to(&mut self, target: &mut Self, bytes: usize) {
        // Moving bytes between two cache guards requires no new global admission.
        // The shared cache total already includes the source charge, and exclusive
        // access to both guards prevents a second transfer of the same bytes.
        self.assert_same_ledger(target);
        assert!(bytes <= self.bytes);
        // Both guards are exclusively borrowed; the shared total never changes.
        self.bytes -= bytes;
        target.bytes += bytes;
    }

    // Adopt an existing allocation without releasing or charging its shared
    // memory twice. Only the optional cache sub-limit needs new admission.
    pub(crate) fn absorb_workspace(
        &mut self,
        work: &mut WorkspaceReservation,
        bytes: usize,
        limit: usize,
    ) -> bool {
        assert!(bytes <= work.bytes);
        // Lock the workspace's operation before the shared ledger, as other
        // cross-category transfers do. Update both views under those guards so
        // concurrent admissions cannot spend an in-between credit.
        let mut operation = work
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Check the workspace's currently attached ledger while its operation
        // state is locked. Taking the ledger from an earlier unlocked snapshot
        // could assign a refund to a different database after an attachment.
        assert!(Arc::ptr_eq(&operation.ledger, &self.ledger));
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // This allocation is already inside the parent total as workspace. Check
        // only whether its new cache category can accept it, with no mutation to
        // either owner on refusal.
        if ledger.cache_used as u128 + bytes as u128 > limit as u128 {
            return false;
        }
        // The operation ceases to own these bytes as temporary work. Moving them
        // out of both workspace counters lets other operations use working
        // headroom even though total resident bytes stay constant.
        ledger.workspace_used -= bytes;
        operation.workspace_used -= bytes;
        ledger.cache_used += bytes;
        // The shared ledger counters and the two individual owners must describe
        // the same transfer. Update the outgoing owner too so dropping it later
        // cannot refund bytes now owned by the cache.
        work.bytes -= bytes;
        self.bytes += bytes;
        true
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn retain(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Refund only the difference between the old and retained sizes. The
        // remaining reservation still belongs to this cache and cannot be offered
        // to another owner.
        ledger.cache_used -= self.bytes - bytes;
        self.bytes = bytes;
    }
}

impl Drop for CacheReservation {
    fn drop(&mut self) {
        // A fork child has its own copy of the ledger memory but may have
        // inherited a locked mutex. Skip inherited refunds there before taking any
        // lock. The parent's original owner remains responsible for its
        // accounting.
        if self.bytes != 0 && self.ledger.pid == std::process::id() {
            self.retain(0);
        }
    }
}

pub(crate) struct CatalogReservation {
    // Owner of the catalog quota and total quota that backed this reservation.
    ledger: Arc<WorkLedger>,
    // Retained bytes charged to the catalog by this owner. Temporary input can
    // transfer into the charge before becoming a permanent row.
    bytes: usize,
}

impl CatalogReservation {
    // Attach incoming operations to this catalog's ledger so later promotions can
    // transfer already admitted input storage without crossing database budgets.
    pub(crate) fn attach_context(&self, context: &JsonOperationContext) -> Result<()> {
        context.attach(Arc::clone(&self.ledger), JsonInputKind::StoredJson)
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn grow(&mut self, additional: usize) -> Result<()> {
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Ordinary catalog growth has no source reservation to consume. Pass zero
        // transferred bytes so all of the new storage counts against both catalog
        // and total headroom.
        ledger.admit_catalog(additional as u128, 0)?;
        ledger.catalog_used += additional;
        self.bytes += additional;
        Ok(())
    }

    pub(crate) fn retain(&mut self, bytes: usize) {
        // Retaining catalog storage can only shrink an existing reservation. New
        // allocations must go through grow or absorb so the category and parent
        // limits get a chance to refuse them.
        assert!(bytes <= self.bytes);
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.catalog_used -= self.bytes - bytes;
        self.bytes = bytes;
    }

    // Move already admitted allocations, including their shared memory credit,
    // without briefly releasing it or charging the same allocation twice.
    pub(crate) fn absorb(
        &mut self,
        json: &mut JsonReservation,
        mut workspace: Option<&mut WorkspaceReservation>,
        json_bytes: usize,
        workspace_bytes: usize,
    ) -> Result<()> {
        // Only covered JSON bytes can travel with an allocation. A reservation may
        // also hold unused growth credit, but moving that portion as if it were
        // live input would undercharge future work in the source operation.
        assert!(json_bytes <= json.covered);
        assert!(workspace_bytes <= workspace.as_ref().map_or(0, |work| work.bytes));
        // Both input owners must belong to one operation because the transfer
        // decrements its single pair of JSON and workspace counters. Sharing a
        // database ledger alone is insufficient.
        if let Some(workspace) = &workspace {
            assert!(Arc::ptr_eq(&json.context.state, &workspace.context.state));
        }
        let mut operation = json
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert!(Arc::ptr_eq(&operation.ledger, &self.ledger));
        let mut ledger = self
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Add the JSON and ordinary workspace portions for the catalog admission
        // check. Debit the two source categories separately after the check
        // succeeds.
        let bytes = json_bytes + workspace_bytes;
        // Check the complete promotion before changing any counter. A refused
        // catalog admission leaves every input reservation valid and leaves the
        // destination catalog's previous charge intact.
        ledger.admit_catalog(bytes as u128, bytes)?;
        ledger.used -= json_bytes;
        operation.used -= json_bytes;
        ledger.workspace_used -= workspace_bytes;
        operation.workspace_used -= workspace_bytes;
        ledger.catalog_used += bytes;
        self.bytes += bytes;
        // Move the covered portion out of both JSON counters. Any spare grant
        // stays with the input owner. The catalog receives just the actual storage
        // passed across.
        json.bytes -= json_bytes;
        json.covered -= json_bytes;
        if let Some(workspace) = &mut workspace {
            workspace.bytes -= workspace_bytes;
        }
        Ok(())
    }
}

impl Drop for CatalogReservation {
    fn drop(&mut self) {
        // Only a reservation's originating process should touch its ledger on
        // destruction. Skipping a fork-child refund also avoids entering a mutex
        // whose owner thread did not survive the fork.
        if self.bytes != 0 && self.ledger.pid == std::process::id() {
            self.retain(0);
        }
    }
}

#[derive(Clone)]
pub(crate) struct JsonOperationContext {
    // Owning process for this operation. Fork checks must precede touching its
    // shared mutex.
    pid: u32,
    // State shared by clones of this operation context. Independent operations can
    // have separate local totals while sharing a database ledger.
    state: Arc<Mutex<OperationState>>,
    // Optional diagnostics are independent of the resource ledger. The stage
    // belongs to this clone; changing it cannot relabel other live owners.
    search_cost: Option<crate::search_cost::Handle>,
    // Optional observation of actual primary commit boundaries for a host
    // operation. Propagates to nested work without inferring commits from return
    // values.
    write_progress: Option<crate::write_progress::WriteProgress>,
    // Weak link to maintenance-effect tracking for this call. Retaining catalog or
    // cache state must not keep a finished observation alive.
    pub(crate) maintenance: crate::maintenance_progress::Tracking,
}

struct OperationState {
    // Ledger to which this operation currently contributes. Attaching a preexisting
    // context must account for the bytes it already holds.
    ledger: Arc<WorkLedger>,
    // This operation's portion of the ledger's JSON-byte charge, including credit
    // held by its cloned reservations.
    used: usize,
    // General scratch and retained workspace bytes owned by reservations from this
    // operation.
    workspace_used: usize,
}

impl JsonOperationContext {
    // This check is safe before taking the operation mutex because pid is stored
    // directly on the context. Destructors can use it to avoid touching inherited
    // synchronization state.
    pub(crate) fn is_current_process(&self) -> bool {
        self.pid == std::process::id()
    }

    // Clone the ledger currently attached to the operation under its lock. A
    // consumer needing a longer-lived catalog or cache owner can then hold that
    // ledger independently of later context clones.
    pub(crate) fn ledger(&self) -> Arc<WorkLedger> {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        Arc::clone(&operation.ledger)
    }

    // Pointer equality distinguishes two views of the same operation from
    // independently created operations using one shared database ledger. Only the
    // former can merge local JSON ownership directly.
    pub(crate) fn same_operation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    pub(crate) fn with_write_progress(
        &self,
        progress: &crate::write_progress::WriteProgress,
    ) -> Self {
        // Observation is attached to the returned clone. Nested work can report
        // actual append boundaries while both views continue charging the same
        // underlying operation.
        let mut context = self.clone();
        context.write_progress = Some(progress.clone());
        context
    }

    // A planner can declare the whole import's expected record count before the
    // first storage append. The observer retains its first declared total instead
    // of overwriting it with an individual batch size.
    pub(crate) fn expect_write_records(&self, count: usize) {
        if let Some(progress) = &self.write_progress {
            progress.expect_records(count);
        }
    }

    // Starting an append records that its result can be uncertain if the write
    // fails. Confirmation records the affected record count and the batch and
    // sequence numbers returned by storage.
    pub(crate) fn begin_append(&self, count: usize) {
        if let Some(progress) = &self.write_progress {
            progress.begin_append(count);
        }
    }

    // A batch ID and sequence number are available only at confirmation. Record
    // them on the observer here rather than inferring a successful commit from a
    // requested or started append.
    pub(crate) fn confirm_append(&self, records: usize, batch_id: u64, sequence_no: u64) {
        if let Some(progress) = &self.write_progress {
            progress.confirmed(records, batch_id, sequence_no);
        }
    }

    pub(crate) fn with_search_cost(&self, trace: &crate::search_cost::SearchCostTrace) -> Self {
        let mut context = self.clone();
        context.search_cost = Some(trace.handle());
        context
    }

    pub(crate) fn inherit_search_cost(&mut self, source: &Self) {
        // An index-cache load can need its own operation lifetime yet be
        // attributable to the search that triggered it. Copy just the diagnostic
        // handle, leaving both operations' resource counters independent.
        self.search_cost = source.search_cost.clone();
    }

    pub(crate) fn search_cost_stage(&self, stage: crate::search_cost::Stage) -> Self {
        // Change the stage on a cloned diagnostic handle. Earlier reservations may
        // still carry the old stage, but their accounting uses the same operation
        // and their measurements keep the original attribution.
        let mut context = self.clone();
        context.search_cost = context.search_cost.map(|handle| handle.stage(stage));
        context
    }

    pub(crate) fn search_cost_timer(&self) -> Option<crate::search_cost::Timer> {
        self.search_cost
            .as_ref()
            .map(crate::search_cost::Handle::timer)
    }

    pub(crate) fn search_cost_add(&self, counter: crate::search_cost::Counter, amount: u64) {
        if let Some(handle) = &self.search_cost {
            handle.add(counter, amount);
        }
    }

    // A stage counter is additive, but a window count records a cardinality
    // observation. Use the trace's count operation so repeated observations do not
    // inflate the reported size.
    pub(crate) fn search_cost_count(&self, count: crate::search_cost::Count, amount: usize) {
        if let Some(handle) = &self.search_cost {
            handle.count(count, amount);
        }
    }

    #[cfg(test)]
    pub(crate) fn available_total_bytes(&self) -> usize {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.total_limit
            - ledger.used
            - ledger.workspace_used
            - ledger.catalog_used
            - ledger.cache_used
    }

    #[cfg(test)]
    pub(crate) fn available_catalog_bytes(&self) -> usize {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.catalog_limit - ledger.catalog_used
    }

    #[cfg(test)]
    pub(crate) fn available_workspace_bytes(&self) -> usize {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        ledger.working_available()
    }

    pub(crate) fn new(ledger: Arc<WorkLedger>) -> Self {
        // Each new operation begins with zero local charges on the supplied
        // ledger. Clones share this operation state, whereas another call using
        // the same Arc starts independent local counters.
        Self {
            pid: ledger.pid,
            search_cost: None,
            write_progress: None,
            maintenance: Default::default(),
            state: Arc::new(Mutex::new(OperationState {
                ledger,
                used: 0,
                workspace_used: 0,
            })),
        }
    }

    pub(crate) fn standalone() -> Self {
        // A standalone entry point has its own default ledger. Database paths can
        // later attach a context to the handle's ledger and must account for any
        // bytes already retained during parsing.
        Self::new(Arc::new(WorkLedger::default()))
    }

    pub(crate) fn reservation(&self) -> JsonReservation {
        JsonReservation {
            context: self.clone(),
            bytes: 0,
            // Keep bytes granted and bytes actually covered separate. Aggregate
            // construction may reserve ahead of the next child and later return
            // the unclaimed remainder.
            covered: 0,
            aggregate: false,
        }
    }

    // Aggregate guards can reserve ahead for several small descendants. Start that
    // behavior only on aggregate construction, allowing ordinary reservations to
    // admit individual allocations without speculative growth.
    pub(super) fn aggregate_reservation(&self) -> JsonReservation {
        let mut work = self.reservation();
        work.start_aggregate();
        work
    }

    // This owner charges general working memory, including native buffers beside
    // the JSON tree. Sharing the operation keeps conversions between it and JSON
    // from counting the same live allocation twice.
    pub(crate) fn workspace_reservation(&self) -> WorkspaceReservation {
        WorkspaceReservation {
            context: self.clone(),
            bytes: 0,
        }
    }

    pub(crate) fn catalog_reservation(&self) -> CatalogReservation {
        // A catalog reservation stores a direct reference to the operation's
        // current ledger. Its retained bytes can outlive the short-lived operation
        // that built the catalog.
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        CatalogReservation {
            ledger: Arc::clone(&operation.ledger),
            bytes: 0,
        }
    }

    // Cache admission is optional, but its live allocations share the parent
    // memory ceiling with working memory and Catalog state.
    pub(crate) fn cache_reservation(&self, bytes: u128, limit: usize) -> Option<CacheReservation> {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let mut ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Refusing an optional cache is a normal fallback and does not mark a
        // source invalid. On success retain the exact ledger that accepted the
        // charge so later drops return it to the correct database.
        if !ledger.try_admit_cache(bytes, limit) {
            return None;
        }
        Some(CacheReservation {
            ledger: Arc::clone(&operation.ledger),
            bytes: bytes as usize,
        })
    }

    pub(crate) fn attach(&self, target: Arc<WorkLedger>, kind: JsonInputKind) -> Result<()> {
        // Hold the operation lock throughout the transfer. Its reservations
        // consult the current ledger when shrinking or dropping, so none can
        // refund the old owner between moving counters and publishing the new
        // owner.
        let mut operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let previous = Arc::clone(&operation.ledger);
        // Attaching to the same Arc is already complete. Return before trying to
        // lock the same ledger twice or readmit bytes the destination still owns.
        if Arc::ptr_eq(&previous, &target) {
            return Ok(());
        }
        // A parser can own JSON values and temporary workspace before it is attached
        // to a database. Check both charges against the destination while the
        // operation and ledgers are locked. On refusal the guards must still refund
        // the original ledger.
        let transfer = |source: &mut LedgerState, destination: &mut LedgerState| {
            // Check the operation's JSON charge against both its policy cap and
            // the space the target can supply. Workspace follows in a separate
            // combined check because its refusal carries the resource phase.
            let limit =
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES.min(destination.working_available());
            // A JSON transfer can exceed the target's effective JSON limit even if
            // the target ledger has not exhausted every category. Report that
            // category failure before the general shared-workspace preflight.
            if operation.used > limit {
                return Err(kind.limit_error(
                    JsonLimitReason::WorkBudget,
                    limit,
                    Some(operation.used),
                ));
            }
            // The target currently accounts for neither of this operation's
            // charges. Test their sum before subtracting anything from the source,
            // while the source still owns both on failure.
            destination.admit_workspace(
                operation.used as u128 + operation.workspace_used as u128,
                "operation_scratch",
            )?;
            // After the fallible checks, move JSON and workspace into their
            // corresponding destination categories. Their distinction still
            // matters for later per-operation JSON admission.
            source.used -= operation.used;
            destination.used += operation.used;
            // Only this operation's charges leave the source ledger. Other
            // operations sharing that ledger keep their own JSON, workspace,
            // catalog and cache counters.
            source.workspace_used -= operation.workspace_used;
            destination.workspace_used += operation.workspace_used;
            Ok(())
        };
        // A stable lock order also permits simultaneous transfers in opposite directions.
        // Order locks by ledger allocation identity rather than transfer
        // direction. Concurrent A-to-B and B-to-A attaches then acquire the same
        // two mutexes in the same order.
        if Arc::as_ptr(&previous) < Arc::as_ptr(&target) {
            let mut source = previous
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut destination = target
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            transfer(&mut source, &mut destination)?;
        // Reverse the acquisition sequence when the destination sorts first, but
        // pass source and destination to transfer in their semantic order.
        // Changing lock order must not reverse the direction of the accounting.
        } else {
            let mut destination = target
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut source = previous
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            transfer(&mut source, &mut destination)?;
        }
        // All outstanding guards share the operation, so switching this one ledger
        // pointer redirects their future release. Keep it pointed at the source
        // until the counter transfer has succeeded.
        operation.ledger = target;
        Ok(())
    }

    pub(crate) fn limit_error(&self, kind: JsonInputKind) -> crate::Error {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // When no reliable requested size is available, still report the effective
        // JSON allowance. The structured error can omit actual rather than
        // inventing a measurable allocation size.
        let limit = effective_limit(&operation, &ledger);
        kind.limit_error(JsonLimitReason::WorkBudget, limit, None)
    }

    fn available(&self) -> usize {
        let operation = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ledger = operation
            .ledger
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // The absolute limit includes bytes already held by this operation.
        // Subtract that usage when answering how much additional JSON work can
        // still be admitted.
        effective_limit(&operation, &ledger) - operation.used
    }
}

fn effective_limit(operation: &OperationState, ledger: &LedgerState) -> usize {
    // Add back this operation's existing JSON usage when computing its absolute
    // ceiling. The ledger's remaining headroom is already net of those bytes and
    // of other operations' live allocations.
    JsonProcessingPolicyV1::WORK_BUDGET_BYTES.min(ledger.working_available() + operation.used)
}

// This is a lease for bounded working buffers, not an allocator/RSS measurement.
// Keep it alive until the buffers are freed or transfer it with their ownership.
pub(crate) struct WorkspaceReservation {
    // Operation that must receive the refund or ownership transfer for this
    // workspace.
    context: JsonOperationContext,
    // Bytes of general workspace this reservation currently owns, independent of
    // any preferred size requested by the caller.
    bytes: usize,
}

impl std::fmt::Debug for WorkspaceReservation {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("WorkspaceReservation")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl WorkspaceReservation {
    pub(crate) fn take_json(
        &mut self,
        bytes: usize,
        kind: JsonInputKind,
    ) -> Result<JsonReservation> {
        // Transfer only bytes held by this workspace lease. The returned JSON
        // lease covers that same allocation, which must never become uncharged
        // between these two owners.
        assert!(bytes <= self.bytes);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // These bytes already count toward shared memory as workspace. Converting
        // them to retained JSON also subjects them to the JSON allowance. Check that
        // allowance before changing either view so refusal leaves the original
        // reservation intact.
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES
            // The shared ledger already counts the bytes as workspace. Add them
            // back for this category check because their conversion will not
            // consume new shared capacity.
            .min(ledger.working_available() + operation.used + bytes);
        // Compare existing JSON plus the conversion amount. A transfer of already
        // charged workspace can still cross the per-operation JSON ceiling even
        // though no new shared bytes were allocated.
        let requested = operation.used + bytes;
        if requested > limit {
            return Err(kind.limit_error(JsonLimitReason::WorkBudget, limit, Some(requested)));
        }
        // Move both the operation's and the ledger's category counters under the
        // same locks. Total working usage stays constant throughout a successful
        // ownership conversion.
        operation.workspace_used -= bytes;
        ledger.workspace_used -= bytes;
        operation.used += bytes;
        ledger.used += bytes;
        self.bytes -= bytes;
        Ok(JsonReservation {
            context: self.context.clone(),
            bytes,
            // Mark all transferred bytes covered immediately. A later aggregate
            // reservation must not treat live payload credit as spare admission it
            // could spend on another field.
            covered: bytes,
            aggregate: false,
        })
    }

    pub(crate) fn context(&self) -> &JsonOperationContext {
        &self.context
    }

    pub(crate) fn merge(&mut self, mut other: Self) {
        // Only guards belonging to the same operation can merge, even if that
        // operation shares a database ledger with others. Move ownership locally
        // and leave ledger totals as they were.
        assert!(Arc::ptr_eq(&self.context.state, &other.context.state));
        self.bytes += other.bytes;
        // Zero the consumed lease before it drops. Its bytes are now part of the
        // receiver, so allowing its destructor to refund them would leave live
        // allocations without credit.
        other.bytes = 0;
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn retain(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        // An unchanged grant needs no shared ledger access. Retaining fewer bytes
        // takes the normal refund path so another operation can see the freed
        // headroom.
        if bytes == self.bytes {
            return;
        }
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Use the ledger currently attached to this operation. It may differ from
        // the ledger that originally admitted the buffer, but attach transferred
        // the outstanding workspace with it.
        let released = self.bytes - bytes;
        operation.workspace_used -= released;
        ledger.workspace_used -= released;
        self.bytes = bytes;
    }

    pub(crate) fn grow_to(
        &mut self,
        minimum: usize,
        preferred: usize,
        phase: &'static str,
    ) -> Result<usize> {
        self.grow_to_wide(minimum as u128, preferred as u128, phase)
    }

    pub(crate) fn grow_to_wide(
        &mut self,
        minimum: u128,
        preferred: u128,
        phase: &'static str,
    ) -> Result<usize> {
        // A preferred size smaller than the minimum is a caller error. Both values
        // can exceed usize during admission so overflow is reported with the
        // requested size before selecting an implementable grant.
        assert!(minimum <= preferred);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // The arguments are absolute target sizes. Test only the extra minimum
        // needed beyond this lease's current bytes, and do not shrink a larger
        // existing reservation implicitly.
        let minimum_extra = minimum.saturating_sub(self.bytes as u128);
        ledger.admit_workspace(minimum_extra, phase)?;
        // Once the minimum fits, grant as much of the preferred size as current
        // headroom allows. The available count bounds the wide request before
        // converting to a native-size grant.
        let additional = preferred
            .saturating_sub(self.bytes as u128)
            .min(ledger.working_available() as u128) as usize;
        // Update the ledger, the operation and this individual lease while both
        // mutexes are held. Future release must remove the same additional amount
        // at each level.
        ledger.workspace_used += additional;
        operation.workspace_used += additional;
        self.bytes += additional;
        Ok(self.bytes)
    }
}

impl Drop for WorkspaceReservation {
    fn drop(&mut self) {
        // A fork child inherits the parent's memory image and locks but cannot
        // return the parent's credit. Check process identity before retain would
        // enter either inherited ledger mutex.
        if self.bytes != 0 && self.context.is_current_process() {
            self.retain(0);
        }
    }
}

pub(crate) fn reserve_heap_slot<T: Ord>(
    heap: &mut BinaryHeap<T>,
    work: &mut WorkspaceReservation,
    limit: usize,
) -> Result<()> {
    // A spare heap slot has already been paid for and can accept a candidate
    // without a new grant. Grow only at the boundary where insertion would ask the
    // allocator for more storage.
    if heap.len() < heap.capacity() {
        return Ok(());
    }
    let size = std::mem::size_of::<T>();
    let old_capacity = heap.capacity();
    let before = work.reserved_bytes();
    // The heap's owner can also cover the retained candidates' IDs or
    // explanations. Isolate those payload bytes before replacing the slots so
    // later capacity changes preserve their separate credit.
    let payload = before - old_capacity * size;
    // Count capacity in elements but admit bytes in u128. Growing to one more slot
    // must include the entire prospective backing buffer, even when just one new
    // candidate is about to be inserted.
    let minimum = heap.len() as u128 + 1;
    // Callers reject the incoming candidate or evict a loser when the result limit
    // is full. This makes room for one more slot. The cap here limits speculative
    // growth.
    let preferred = (old_capacity as u128 * 2).max(4).min(limit as u128);
    let granted = work.grow_to_wide(
        before as u128 + minimum * size as u128,
        before as u128 + preferred * size as u128,
        "operation_scratch",
    )?;
    // A byte grant need not contain an integral number of slots. Round down before
    // requesting native capacity and release any unusable remainder while both
    // buffers may still be live.
    let capacity = (granted - before) / size;
    // Keep both arrays charged until the old allocation has actually been replaced.
    work.retain(before + capacity * size);
    // Allocator refusal leaves the original heap intact. Return only the
    // speculative replacement budget and preserve the old array and payload
    // charge.
    if let Err(error) = heap.try_reserve_exact(capacity - heap.len()) {
        work.retain(before);
        return Err(std::io::Error::other(error.to_string()).into());
    }
    // An allocator that returned an unadmitted capacity cannot leave that heap
    // accessible. Discard it and remove slot credit, retaining the separately
    // tracked payload allowance on this error path.
    if heap.capacity() != capacity {
        drop(std::mem::take(heap));
        work.retain(payload);
        return Err(std::io::Error::other("candidate capacity exceeds reservation").into());
    }
    // Replace the old slot component of the grant with the final array size. Keep
    // the heap's separately measured payload even when the new capacity exactly
    // equals the minimum.
    work.retain(payload + capacity * size);
    Ok(())
}

pub(crate) fn try_shrink_heap<T: Ord>(
    heap: &mut BinaryHeap<T>,
    work: &mut WorkspaceReservation,
) -> Result<bool> {
    let capacity = heap.capacity();
    let count = heap.len();
    let size = std::mem::size_of::<T>();
    // A full heap has no spare slots to reclaim, and zero-sized entries have no
    // backing slot bytes. Report that the caller gained no workspace for its
    // pending retry.
    if count == capacity || size == 0 {
        return Ok(false);
    }
    let before = work.reserved_bytes();
    // Only the backing array gets smaller. Existing heap elements can own
    // additional bytes under this lease, which must still be covered after their
    // values move.
    let retained = before - capacity * size + count * size;
    // An empty heap can release its array without first allocating another one.
    // Keep any non-slot payload credit in the existing reservation.
    if count == 0 {
        drop(std::mem::take(heap));
        work.retain(retained);
        return Ok(true);
    }
    let peak = before as u128 + count as u128 * size as u128;
    // Shrinking a nonempty heap needs temporary space for both arrays. A resource
    // refusal means there was no usable opportunity to shrink, while a native
    // allocation failure below remains an error.
    match work.grow_to_wide(peak, peak, "operation_scratch") {
        Err(error) if error.code() == ErrorCode::ResourceExhausted => return Ok(false),
        result => {
            result?;
        }
    }
    // Build a separate replacement vector before touching the old heap. If
    // reserving exact space fails, dropping that vector restores the pre-shrink
    // state and existing candidate order.
    let mut values = Vec::new();
    if let Err(error) = values.try_reserve_exact(count) {
        drop(values);
        work.retain(before);
        return Err(std::io::Error::other(error.to_string()).into());
    }
    // Check replacement capacity while the original heap is still intact. A
    // mismatch can discard just the new Vec and roll back the speculative grant
    // without losing retained candidates.
    if values.capacity() != count {
        drop(values);
        work.retain(before);
        return Err(std::io::Error::other("candidate capacity exceeds reservation").into());
    }
    // Transfer the payloads without cloning; keep the old array charged until freed.
    let mut old = std::mem::take(heap).into_vec();
    values.append(&mut old);
    drop(old);
    // The exact-size replacement releases the difference in slot bytes.
    // Allocations owned by the elements stay with their moved values.
    *heap = BinaryHeap::from(values);
    work.retain(retained);
    Ok(true)
}

// The array must be freed before its lease. Only an explicit ownership transfer
// to a public result may detach it; growth beyond the admitted capacity is forbidden.
pub(crate) struct ReservedVec<T> {
    // Backing vector whose capacity is charged, including unused slots.
    values: Vec<T>,
    // Keeps the charge alive while the vector or its transferred storage can be
    // used.
    _work: WorkspaceReservation,
}

impl<T: std::fmt::Debug> std::fmt::Debug for ReservedVec<T> {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.values.fmt(out)
    }
}

impl<T> ReservedVec<T> {
    pub(crate) fn pop(&mut self) -> Option<T> {
        self.values.pop()
    }

    #[cfg(test)]
    pub(crate) fn assert_operation_context(&self, context: &JsonOperationContext) {
        let mut probe = self._work.context().workspace_reservation();
        probe.merge(context.workspace_reservation());
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self._work.reserved_bytes()
    }

    pub(crate) fn with_capacity(
        count: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        let mut work = context.workspace_reservation();
        // Derive the byte count in u128 so a product too large for the address
        // space still reaches the limit check at its real size before any
        // allocator request.
        let bytes = (count as u128) * (std::mem::size_of::<T>() as u128);
        // Ask the operation to admit the requested memory before trying the native
        // allocation. If the allocator then fails, the local lease drops without
        // exposing a partially usable vector.
        work.grow_to_wide(bytes, bytes, phase)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        // Zero-sized elements use Vec's special capacity representation and
        // require no backing allocation. For other types the measured slot grant
        // must match the capacity actually returned.
        if std::mem::size_of::<T>() != 0 && values.capacity() != count {
            return Err(
                std::io::Error::other("array allocation exceeds requested capacity").into(),
            );
        }
        Ok(Self {
            values,
            _work: work,
        })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub(crate) fn capacity(&self) -> usize {
        self.values.capacity()
    }

    pub(crate) fn context(&self) -> &JsonOperationContext {
        self._work.context()
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    pub(crate) fn retain(&mut self, keep: impl FnMut(&T) -> bool) {
        // Filtering changes the number of initialized elements while leaving the
        // backing allocation intact. Keep the entire slot reservation until
        // capacity is actually released or transferred.
        self.values.retain(keep);
    }

    pub(crate) fn push(&mut self, value: T) {
        // This method is for callers that already know the bound. Prevent Vec from
        // reallocating implicitly. Operations building an unknown number of
        // elements must use the fallible growth path.
        assert!(
            self.values.len() < self.values.capacity(),
            "reserved array capacity exceeded"
        );
        self.values.push(value);
    }

    pub(crate) fn try_push(&mut self, value: T) -> Result<()> {
        if self.values.len() == self.values.capacity() {
            let size = std::mem::size_of::<T>();
            let before = self._work.reserved_bytes();
            let minimum = self
                .values
                .len()
                .checked_add(1)
                .ok_or_else(|| std::io::Error::other("array capacity overflow"))?
                as u128;
            // Try geometric growth to amortize repeated appends, with one more
            // element as the true minimum. A partially granted preference can
            // still satisfy the insertion.
            let preferred = (self.values.capacity() as u128 * 2).max(4);
            let granted = self._work.grow_to_wide(
                before as u128 + minimum * size as u128,
                before as u128 + preferred * size as u128,
                "operation_scratch",
            )?;
            let capacity = (granted - before) / size;
            // The old array remains charged until its replacement is allocated.
            self._work.retain(before + capacity * size);
            // A failed allocator reserve leaves the old Vec unchanged. Restore the
            // pre-growth grant and let the caller distinguish that error from the
            // capacity-mismatch branch below.
            if let Err(error) = self.values.try_reserve_exact(capacity - self.values.len()) {
                self._work.retain(before);
                return Err(std::io::Error::other(error.to_string()).into());
            }
            // An unexpected capacity invalidates the bounded vector. Drop it
            // before refunding its slot grant. Unlike an allocator refusal above,
            // this path cannot keep the old elements accessible.
            if self.values.capacity() != capacity {
                drop(std::mem::take(&mut self.values));
                self._work.retain(0);
                return Err(std::io::Error::other("array capacity exceeds reservation").into());
            }
            // At this point the new capacity is exact and the old backing storage
            // is gone. Keep only the new slot grant before publishing the appended
            // element.
            self._work.retain(capacity * size);
        }
        self.values.push(value);
        Ok(())
    }

    // Taking just the Vec transfers it out of this reservation's lifetime. A
    // caller retaining the allocation within the operation needs into_parts so its
    // grant follows it.
    pub(crate) fn into_vec(self) -> Vec<T> {
        self.values
    }

    // Move backing storage and its existing workspace charge together without
    // changing the operation's total. Consumers that adopt a different
    // representation can then transfer that credit explicitly.
    pub(crate) fn into_parts(self) -> (Vec<T>, WorkspaceReservation) {
        (self.values, self._work)
    }

    pub(crate) fn from_reserved(values: Vec<T>, mut work: WorkspaceReservation) -> Self {
        // A reclaimed or sorted Vec can arrive with temporary headroom in its
        // lease. Trim the inherited credit to the actual slot capacity as the new
        // owner adopts the vector.
        work.retain(values.capacity() * std::mem::size_of::<T>());
        Self {
            values,
            _work: work,
        }
    }
}

impl<T> AsRef<[T]> for ReservedVec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl ReservedVec<u8> {
    pub(crate) fn zeroed(
        count: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        // Acquire the complete byte capacity before zero initialization. Resizing
        // to that exact count then fills already admitted storage without another
        // allocation.
        let mut bytes = Self::with_capacity(count, context, phase)?;
        bytes.values.resize(count, 0);
        Ok(bytes)
    }
}

pub(crate) struct ReservedString {
    // Owned UTF-8 bytes of the string, with unused capacity still charged.
    value: String,
    // Workspace for the string allocation. into_parts can move this credit with the
    // string to its next owner.
    _work: WorkspaceReservation,
}

impl ReservedString {
    pub(crate) fn into_parts(self) -> (String, WorkspaceReservation) {
        (self.value, self._work)
    }

    pub(crate) fn copy(value: &str, context: &JsonOperationContext) -> Result<Self> {
        Self::copy_in(value, context, "operation_scratch")
    }

    pub(crate) fn copy_in(
        value: &str,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        // UTF-8 byte length determines the required capacity for this copy. Admit
        // once before push_str so multibyte input does not trigger an unaccounted
        // growth.
        let mut owned = Self::with_capacity(value.len(), context, phase)?;
        owned.value.push_str(value);
        Ok(owned)
    }

    // Internal renderers must be deterministic. The second pass is bounded even
    // if a renderer changes its output after sizing.
    // Use the same formatter to measure the intended byte length and then produce
    // the text. Callers must supply a renderer whose output length is stable
    // across both calls.
    pub(crate) fn build(
        context: &JsonOperationContext,
        phase: &'static str,
        render: impl Fn(&mut dyn std::fmt::Write) -> std::fmt::Result,
    ) -> Result<Self> {
        Self::build_exact(rendered_string_bytes(&render)?, context, phase, render)
    }

    // Use this entry when the caller already knows the expected encoded size. The
    // writer uses the reserved capacity and completion checks the measured size
    // against the actual output.
    pub(crate) fn build_exact(
        bytes: usize,
        context: &JsonOperationContext,
        phase: &'static str,
        render: impl FnOnce(&mut dyn std::fmt::Write) -> std::fmt::Result,
    ) -> Result<Self> {
        let mut owned = Self::with_capacity(bytes, context, phase)?;
        owned.append(render)?;
        // A bounded write can still succeed with fewer bytes than promised. Reject
        // that mismatch before handing the string to a caller that may rely on the
        // exact length.
        if owned.value.len() != bytes {
            return Err(std::io::Error::other("reserved string formatting failed").into());
        }
        Ok(owned)
    }

    // Append using already admitted capacity. If a fragment exceeds the remaining
    // room, earlier fragments from the callback can still be present.
    pub(crate) fn append(
        &mut self,
        render: impl FnOnce(&mut dyn std::fmt::Write) -> std::fmt::Result,
    ) -> Result<()> {
        write_precharged_string(&mut self.value, render)
    }

    pub(crate) fn with_capacity(
        bytes: usize,
        context: &JsonOperationContext,
        phase: &'static str,
    ) -> Result<Self> {
        // Admit ordinary workspace before allocating the string. A failed
        // allocation or exact-capacity check drops the local guard and gives its
        // charge back.
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, phase)?;
        let mut owned = String::new();
        owned
            // Use a fallible allocation so allocator refusal is returned with an
            // I/O error. The workspace admission above reports a different failure
            // when the configured limit is exceeded.
            .try_reserve_exact(bytes)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        if owned.capacity() != bytes {
            return Err(
                std::io::Error::other("string allocation exceeds requested capacity").into(),
            );
        }
        Ok(Self::from_reserved(owned, work))
    }

    // Adopt a string backed by an existing workspace reservation. Refund the
    // excess while retaining coverage for its capacity, including empty space
    // available to append.
    pub(crate) fn from_reserved(value: String, mut work: WorkspaceReservation) -> Self {
        work.retain(value.capacity());
        Self { value, _work: work }
    }
}

impl AsRef<str> for ReservedString {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

// A guard owns part of the JSON work charge for a single operation. Coverage
// tracks live allocations. The charge can be larger during aggregation so child
// allocations can use the grant without locking the shared ledger every time.
pub(crate) struct JsonReservation {
    // Operation that accounts for this reservation and classifies any admission
    // failure.
    context: JsonOperationContext,
    // Charged blocks may include spare capacity; coverage only backs live work.
    bytes: usize,
    // Bytes already assigned to live allocations or validated work. The rest of the
    // charge, if any, is credit that can be reused or returned.
    covered: usize,
    // Whether growth can reserve in blocks for many small allocations. Finishing
    // the aggregate releases the unused difference between bytes and covered.
    aggregate: bool,
}

impl JsonReservation {
    // Charge the estimated map storage separately from the owned names and
    // payloads. This path is used when those values have already been admitted.
    pub(crate) fn reserve_object_nodes(
        &mut self,
        length: usize,
        kind: JsonInputKind,
    ) -> Result<()> {
        self.grow(object_workspace(length), kind)
    }

    // Round an aggregate grant to four KiB where budget allows. Near the byte
    // limit an exact grant can still succeed.
    const BLOCK_BYTES: usize = 4096;

    pub(crate) fn context(&self) -> &JsonOperationContext {
        &self.context
    }

    // Enable block precharging for a series of allocations made under this guard.
    pub(super) fn start_aggregate(&mut self) {
        self.aggregate = true;
    }

    // Return spare precharge before transferring the completed value out of a
    // traversal, and use exact charging on subsequent growth.
    pub(super) fn finish_aggregate(&mut self) {
        self.retain(self.covered);
        self.aggregate = false;
    }

    // Collect the estimate for a complete map's keys and nodes in a child guard.
    // The caller can merge the successful guard into the owner after obtaining the
    // charge.
    pub(super) fn reserve_object_keys<'a>(
        &mut self,
        keys: impl ExactSizeIterator<Item = &'a str>,
        kind: JsonInputKind,
    ) -> Result<Self> {
        let entries = keys.len();
        // Charge key lengths in UTF-8 bytes along with the estimated node cost.
        // Accumulate with saturation so an unrepresentable sum cannot become a
        // small allocation request.
        let bytes = keys.fold(object_workspace(entries), |bytes, key| {
            bytes.saturating_add(key.len())
        });
        // Transfer unused precharge from the parent first. Both guards point at
        // the same operation, so this part of the transfer does not change either
        // shared counter.
        let reused = bytes.min(self.bytes - self.covered);
        let mut work = self.scratch();
        self.bytes -= reused;
        work.bytes = reused;
        work.covered = reused;
        // Only the remainder needs a fresh grant. If it fails, dropping the child
        // refunds the precharge moved out of the parent.
        work.grow(bytes - reused, kind)?;
        Ok(work)
    }

    // Reclassify live JSON memory as ordinary workspace without reserving it
    // twice. Update both sets of counters while holding the operation lock
    // followed by its ledger lock.
    pub(crate) fn take_workspace(&mut self, bytes: usize) -> WorkspaceReservation {
        assert!(bytes <= self.covered);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Maintain the combined accounting across the two categories while
        // changing which limit will govern future growth. Neither lock is released
        // during the transfer.
        operation.used -= bytes;
        ledger.used -= bytes;
        operation.workspace_used += bytes;
        ledger.workspace_used += bytes;
        // Remove the transferred charge from this guard before constructing the
        // workspace owner. Destruction of either owner then releases only its own
        // share.
        self.bytes -= bytes;
        self.covered -= bytes;
        WorkspaceReservation {
            context: self.context.clone(),
            bytes,
        }
    }

    // Create an initially empty guard on the same operation for temporary
    // allocations that need an independent lifetime.
    pub(super) fn scratch(&self) -> Self {
        self.context.reservation()
    }

    // Bring coverage up to an absolute byte count. Repeated requests for the same
    // total are free, and larger ones can precharge spare capacity.
    pub(super) fn cover(&mut self, bytes: usize, kind: JsonInputKind) -> Result<()> {
        self.grow_with_granularity(bytes.saturating_sub(self.covered), kind, true)
    }

    // Keep the requested charge and refund the remainder. The retained bytes all
    // count as live coverage after this call.
    pub(crate) fn retain(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        self.refund(self.bytes - bytes);
        self.covered = bytes;
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn covered_bytes(&self) -> usize {
        self.covered
    }

    // Charge the actual UTF-8 allocation before copying. The returned String
    // relies on this reservation remaining alive with it.
    pub(crate) fn copy_string(&mut self, value: &str, kind: JsonInputKind) -> Result<String> {
        let mut copy = String::new();
        self.reserve_string(&mut copy, value.len(), kind)?;
        copy.push_str(value);
        Ok(copy)
    }

    pub(crate) fn format_string(
        &mut self,
        kind: JsonInputKind,
        render: impl Fn(&mut dyn std::fmt::Write) -> std::fmt::Result,
    ) -> Result<String> {
        // Measure before allocating, then take a coverage checkpoint for rollback.
        // A failed formatting pass must leave coverage for other live strings
        // intact.
        let length = rendered_string_bytes(&render)?;
        let covered = self.covered;
        let mut value = String::new();
        // Treat allocation and both render checks as one attempt. A caller can
        // reuse this guard after any of those steps fails because the error path
        // restores the prior coverage.
        let built = (|| {
            self.reserve_string(&mut value, length, kind)?;
            write_precharged_string(&mut value, render)?;
            // A too-short rendering also fails. The bounded writer catches excess
            // beyond the admitted capacity, while the final length check enforces
            // agreement with the sizing pass.
            if value.len() != length {
                return Err(std::io::Error::other("reserved string formatting failed").into());
            }
            Ok(())
        })();
        // Free the partial string before releasing its coverage. An aggregate can
        // keep the returned portion as unused credit for later work.
        if let Err(error) = built {
            drop(value);
            self.release(self.covered - covered);
            return Err(error);
        }
        Ok(value)
    }

    pub(super) fn copy_validated_value(
        &mut self,
        value: &super::JsonValue,
        kind: JsonInputKind,
    ) -> Result<super::JsonValue> {
        // Refund spare credit before the clone takes a new guard on this
        // operation. Otherwise precharge held here could reduce the memory
        // available to that guard.
        self.retain(self.covered);
        let (value, work) = value.clone_output_in(kind, &self.context)?.into_parts();
        self.merge(work);
        Ok(value)
    }

    // Combine ownership within one operation. Zero the donor before Drop so it
    // cannot refund bytes now accounted for by the receiver.
    pub(crate) fn merge(&mut self, mut other: Self) {
        assert!(Arc::ptr_eq(&self.context.state, &other.context.state));
        self.bytes += other.bytes;
        self.covered += other.covered;
        other.bytes = 0;
        other.covered = 0;
    }

    // Use the guard's current mode when admitting another allocation. The cover
    // path can request aggregate growth without changing that mode.
    pub(crate) fn grow(&mut self, additional: usize, kind: JsonInputKind) -> Result<()> {
        self.grow_with_granularity(additional, kind, self.aggregate)
    }

    fn grow_with_granularity(
        &mut self,
        additional: usize,
        kind: JsonInputKind,
        aggregate: bool,
    ) -> Result<()> {
        // A zero increment succeeds without inspecting the ledger. Other
        // operations may consume available headroom but cannot revoke bytes
        // already charged to this guard.
        if additional == 0 {
            return Ok(());
        }
        // An aggregate guard can have credit ahead of the bytes its children have
        // claimed. Consume that credit first. Charging the shared ledger again for
        // every child would count one allocation twice.
        let needed = self.covered.checked_add(additional);
        if let Some(needed) = needed.filter(|bytes| *bytes <= self.bytes) {
            self.covered = needed;
            return Ok(());
        }
        // Slow-path admission holds the operation lock and then its shared ledger,
        // matching refund and workspace transfer. That keeps the available limit
        // and the two counter changes consistent.
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Compare the operation's resulting charge with the effective limit, which
        // also reflects shared usage. Overflow counts as refusal rather than
        // underestimating the requested total.
        let limit = effective_limit(&operation, &ledger);
        let extra = needed.map(|bytes| bytes - self.bytes);
        let requested = extra.and_then(|extra| operation.used.checked_add(extra));
        if !requested.is_some_and(|bytes| bytes <= limit) {
            return Err(kind.limit_error(JsonLimitReason::WorkBudget, limit, requested));
        }
        // A successful request established a representable coverage count. Compute
        // optional precharge only after admitting the required bytes.
        let needed = needed.unwrap();
        // Round the total reservation rather than the increment. That aligns
        // successive small allocations to the same block boundaries.
        let preferred = if aggregate {
            needed
                .checked_add(Self::BLOCK_BYTES - 1)
                .map_or(needed, |bytes| {
                    bytes / Self::BLOCK_BYTES * Self::BLOCK_BYTES
                })
        } else {
            needed
        };
        // Near the budget, use the exact increment. Optional rounding changes only
        // the amount of spare credit, never whether required bytes are admissible.
        let granted = if preferred - self.bytes <= limit - operation.used {
            preferred - self.bytes
        } else {
            needed - self.bytes
        };
        // Coverage grows by the requested increment. Any excess from the larger
        // grant stays available to later allocations.
        operation.used += granted;
        ledger.used += granted;
        self.bytes += granted;
        self.covered = needed;
        Ok(())
    }

    // When the local credit alone covers the minimum, use it as the estimate and
    // avoid a ledger lookup. A caller sizing optional growth should still go
    // through grow to confirm the actual charge.
    fn available_for(&self, minimum: usize) -> usize {
        let spare = self.bytes - self.covered;
        if spare >= minimum {
            spare
        } else {
            spare + self.context.available()
        }
    }

    // Mark the released bytes as unused. Aggregate owners keep that credit for
    // later allocations, while exact owners return it immediately.
    pub(crate) fn release(&mut self, bytes: usize) {
        assert!(bytes <= self.covered);
        self.covered -= bytes;
        if !self.aggregate {
            self.refund(bytes);
        }
    }

    // Return a charged amount to the two shared counters. The caller updates the
    // live coverage according to whether it is releasing an allocation or trimming
    // spare credit.
    fn refund(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        assert!(bytes <= self.bytes);
        let mut operation = self
            .context
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let owner = Arc::clone(&operation.ledger);
        let mut ledger = owner
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.bytes -= bytes;
        operation.used -= bytes;
        ledger.used -= bytes;
    }

    pub(crate) fn reserve_vec<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
        kind: JsonInputKind,
    ) -> Result<()> {
        // Determine the required number of elements from the current length.
        // Already allocated slots can be reused without a new charge, and
        // zero-sized elements have no backed allocation.
        let needed = values
            .len()
            .checked_add(additional)
            .ok_or_else(|| self.context.limit_error(kind))?;
        if needed <= values.capacity() || std::mem::size_of::<T>() == 0 {
            return Ok(());
        }
        let old = values.capacity();
        // Prefer geometric growth to avoid reserving on every append. Keep
        // optional headroom within the estimated budget and make room for the
        // requested length regardless of that estimate.
        let mut target = needed
            .max(old.saturating_mul(2))
            .max(4)
            .min(old.saturating_add(
                // Divide available bytes by element size before computing
                // speculative capacity. The zero-size fast path above keeps that
                // divisor nonzero.
                self.available_for((needed - old).saturating_mul(std::mem::size_of::<T>()))
                    / std::mem::size_of::<T>(),
            ))
            .max(needed);
        let mut bytes = (target - old)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| self.context.limit_error(kind))?;
        // Growth beyond the requested length is optional. If the preferred capacity
        // is refused, admit just the required elements before asking the allocator.
        // Reconcile the actual capacity afterwards since the allocator can round it
        // up.
        if let Err(error) = self.grow(bytes, kind) {
            if target == needed {
                return Err(error);
            }
            // Retry only the admission for the exact required target. No allocator
            // call has happened yet, so there is no new capacity to retain or roll
            // back.
            target = needed;
            bytes = (target - old)
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| self.context.limit_error(kind))?;
            self.grow(bytes, kind)?;
        }
        // A failed allocation leaves the original vector in place. Release only
        // the increment admitted for this attempt.
        if let Err(error) = values.try_reserve_exact(target - values.len()) {
            self.release(bytes);
            return Err(std::io::Error::other(error.to_string()).into());
        }
        // Check the allocator's observed capacity before reporting success. If
        // admitting its extra capacity fails, propagate the refusal to the caller
        // handling the failed allocation.
        let extra = (values.capacity() - target) * std::mem::size_of::<T>();
        self.grow(extra, kind)
    }

    // Use the vector admission policy in byte units for String. Length, capacity
    // and the requested extension are UTF-8 byte counts.
    pub(crate) fn reserve_string(
        &mut self,
        value: &mut String,
        additional: usize,
        kind: JsonInputKind,
    ) -> Result<()> {
        let needed = value
            .len()
            .checked_add(additional)
            .ok_or_else(|| self.context.limit_error(kind))?;
        if needed <= value.capacity() {
            return Ok(());
        }
        let old = value.capacity();
        let mut target = needed
            .max(old.saturating_mul(2))
            .max(8)
            .min(old.saturating_add(self.available_for(needed - old)))
            .max(needed);
        // If the optional doubling cannot be admitted, try just the required
        // capacity. Do not ask the allocator until one of those grants succeeds.
        if let Err(error) = self.grow(target - old, kind) {
            if target == needed {
                return Err(error);
            }
            target = needed;
            self.grow(target - old, kind)?;
        }
        // A budget refusal precedes allocation. If the allocator itself rejects
        // the request, release the admitted increment before returning its error.
        if let Err(error) = value.try_reserve_exact(target - value.len()) {
            self.release(target - old);
            return Err(std::io::Error::other(error.to_string()).into());
        }
        // Reconcile any capacity above the requested target before the caller
        // appends into the string.
        self.grow(value.capacity() - target, kind)
    }
}

// Share sizing and bounded rendering across ordinary and JSON string owners.
fn rendered_string_bytes(
    render: &impl Fn(&mut dyn std::fmt::Write) -> std::fmt::Result,
) -> Result<usize> {
    // The measuring writer counts bytes without storing text. Overflow becomes a
    // formatting error before the allocation pass starts.
    struct Length(usize);
    impl std::fmt::Write for Length {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            self.0 = self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;
            Ok(())
        }
    }
    let mut length = Length(0);
    render(&mut length).map_err(|_| std::io::Error::other("reserved string sizing failed"))?;
    Ok(length.0)
}

fn write_precharged_string(
    value: &mut String,
    render: impl FnOnce(&mut dyn std::fmt::Write) -> std::fmt::Result,
) -> Result<()> {
    // Present the formatter with a writer that refuses any fragment larger than
    // the unused capacity. Appending therefore cannot trigger an implicit
    // reallocation.
    struct Bounded<'a>(&'a mut String);
    impl std::fmt::Write for Bounded<'_> {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            // An oversized fragment leaves earlier writes from the same callback
            // in place. The bounded adapter does not roll back the whole append.
            if value.len() > self.0.capacity() - self.0.len() {
                return Err(std::fmt::Error);
            }
            self.0.push_str(value);
            Ok(())
        }
    }
    render(&mut Bounded(value))
        .map_err(|_| std::io::Error::other("reserved string formatting failed").into())
}

// The enclosing aggregate reservation already covers this string's capacity.
pub(crate) fn copy_precharged_string(value: &str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    // The enclosing reservation covers the requested byte length exactly. Reject
    // an unexpectedly larger allocation before filling it and moving it into the
    // containing value.
    if owned.capacity() != value.len() {
        return Err(std::io::Error::other("string allocation exceeds requested capacity").into());
    }
    owned.push_str(value);
    Ok(owned)
}

// Return any unspent precharge along with live coverage when the guard is
// destroyed. The counters represent the full charge, which can exceed the sum of
// the copied value's allocations.
impl Drop for JsonReservation {
    fn drop(&mut self) {
        // A fork child cannot repair the parent's counters. In particular this
        // must precede access to either inherited mutex during FFI cleanup.
        if self.bytes != 0 && self.context.is_current_process() {
            self.refund(self.bytes);
        }
    }
}

#[cfg(all(test, unix))]
mod fork_cleanup_probe {
    use super::*;

    #[test]
    fn inherited_reservations_do_not_wait_for_parent_operation_or_ledger_locks() {
        const MARKER: &str = "COFFER_RESERVATION_FORK_PROBE";
        const NAME: &str = "json::work::fork_cleanup_probe::inherited_reservations_do_not_wait_for_parent_operation_or_ledger_locks";
        if std::env::var_os(MARKER).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--nocapture"])
                .env(MARKER, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        extern "C" {
            fn fork() -> i32;
            fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
            fn kill(pid: i32, signal: i32) -> i32;
            fn _exit(status: i32) -> !;
        }
        let context = JsonOperationContext::standalone();
        let mut json = context.reservation();
        json.grow(8, JsonInputKind::Filter).unwrap();
        let mut workspace = context.workspace_reservation();
        workspace.grow_to(8, 8, "operation_scratch").unwrap();
        let cache = context.cache_reservation(8, 1024).unwrap();
        let mut catalog = context.catalog_reservation();
        catalog.grow(8).unwrap();
        let mut output = crate::binding::Output::new(
            crate::JsonValue::Null,
            crate::storage::RecordReservation::from(context.reservation()),
            None,
        );
        let available = context.available_workspace_bytes();
        let holder = context.clone();
        let (ready, ready_rx) = std::sync::mpsc::channel();
        let (release, release_rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let operation = holder.state.lock().unwrap();
            let _ledger = operation.ledger.state.lock().unwrap();
            ready.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let child = unsafe { fork() };
        assert!(child >= 0);
        if child == 0 {
            let rejected = output.stack::<u8>().is_err() && output.reserve_host(8).is_err();
            output.clear_native();
            drop(output);
            drop((json, workspace, cache, catalog));
            let fresh = JsonOperationContext::standalone();
            let mut work = fresh.workspace_reservation();
            let passed = rejected && work.grow_to(8192, 8192, "operation_scratch").is_ok();
            drop(work);
            unsafe { _exit(if passed { 0 } else { 2 }) };
        }
        let start = std::time::Instant::now();
        let mut status = 0;
        let passed = loop {
            if unsafe { waitpid(child, &mut status, 1) } == child {
                break status == 0;
            }
            if start.elapsed() > std::time::Duration::from_secs(5) {
                unsafe {
                    kill(child, 9);
                    waitpid(child, &mut status, 0);
                }
                break false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        release.send(()).unwrap();
        thread.join().unwrap();
        assert_eq!(context.available_workspace_bytes(), available);
        drop(output);
        drop((json, workspace, cache, catalog));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
        assert!(
            passed,
            "fork child blocked while dropping inherited reservations"
        );
    }
}

// Reserve a full internal node for every possible minimally occupied B-tree
// node. This is workspace headroom, not an assertion about allocator/RSS bytes.
pub(super) fn object_workspace(entries: usize) -> usize {
    btree_workspace::<String, super::JsonValue>(entries)
}

pub(crate) fn btree_workspace<K, V>(entries: usize) -> usize {
    if entries == 0 {
        return 0;
    }
    let node = 11 * std::mem::size_of::<(K, V)>() + 16 * std::mem::size_of::<usize>();
    (1 + (entries - 1) / 5).saturating_mul(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_growth_reuses_blocks_and_falls_back_to_the_true_minimum() {
        let kind = JsonInputKind::Filter;
        for (limit, charged) in [(4095, 1), (4096, 4096), (4097, 4096)] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let mut work = context.aggregate_reservation();
            work.grow(1, kind).unwrap();
            assert_eq!((work.reserved_bytes(), work.covered_bytes()), (charged, 1));
            work.finish_aggregate();
            assert_eq!(work.reserved_bytes(), 1);
            drop(work);
            assert_eq!(context.available(), limit);
        }
        let ledger = Arc::new(WorkLedger::new(7000));
        let context = JsonOperationContext::new(Arc::clone(&ledger));
        let mut work = context.aggregate_reservation();
        work.grow(1, kind).unwrap();
        work.grow(4089, kind).unwrap();
        assert_eq!(work.reserved_bytes(), 4096);
        work.release(100);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 3990));
        work.grow(100, kind).unwrap();
        work.grow(16, kind).unwrap();
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4106, 4106));
        work.grow(7000 - 4106, kind).unwrap();
        let error = work.grow(1, kind).unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "actual" && value == "7001"));
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (7000, 7000));
        drop(work);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn aggregate_transfers_and_merges_preserve_live_coverage_and_spare_credit() {
        let ledger = Arc::new(WorkLedger::with_limits(12288, 32, 12288));
        let context = JsonOperationContext::new(Arc::clone(&ledger));
        let kind = JsonInputKind::StoredJson;
        let mut work = context.aggregate_reservation();
        work.grow(37, kind).unwrap();
        let mut workspace = work.take_workspace(13);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4083, 24));
        work.merge(workspace.take_json(13, kind).unwrap());
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 37));
        let mut catalog = context.catalog_reservation();
        assert!(catalog.absorb(&mut work, None, 37, 0).is_err());
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 37));
        catalog.absorb(&mut work, None, 24, 0).unwrap();
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4072, 13));
        let mut other = context.aggregate_reservation();
        other.grow(17, kind).unwrap();
        work.merge(other);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (8168, 30));
        work.grow(19, kind).unwrap();
        let live = work.take_workspace(49);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (8119, 0));
        drop(work);
        let state = ledger.state.lock().unwrap();
        assert_eq!(
            (state.used, state.workspace_used, state.catalog_used),
            (0, 49, 24)
        );
        drop(state);
        drop(live);
        drop(workspace);
        drop(catalog);
        assert_eq!(context.available(), 12288);
    }

    #[test]
    fn aggregate_native_failure_keeps_previous_data_and_retries_with_local_credit() {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::new(8192)));
        let kind = JsonInputKind::StoredJson;
        let mut work = context.aggregate_reservation();
        let previous = work.copy_string("previous payload", kind).unwrap();
        let input = "x".repeat(64);
        let covered = work.covered_bytes();
        let pointer = previous.as_ptr();
        let error =
            crate::test_alloc::fail_byte_allocation_once(64, || work.copy_string(&input, kind))
                .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(previous, "previous payload");
        assert_eq!(previous.as_ptr(), pointer);
        assert_eq!(work.covered_bytes(), covered);
        assert!(work.reserved_bytes() >= covered);
        let value = work.copy_string(&input, kind).unwrap();
        work.finish_aggregate();
        assert_eq!(
            work.reserved_bytes(),
            previous.capacity() + value.capacity()
        );
        drop(value);
        drop(previous);
        drop(work);
        assert_eq!(context.available(), 8192);
    }

    #[test]
    fn formatted_strings_reuse_aggregate_spare_and_roll_back_only_the_failed_value() {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::new(4096)));
        let mut work = context.aggregate_reservation();
        let kind = JsonInputKind::StoredJson;
        work.grow(1, kind).unwrap();
        assert_eq!(context.available(), 0);
        let value = work
            .format_string(kind, |out| out.write_str("abc"))
            .unwrap();
        assert_eq!(value, "abc");
        assert_eq!(work.reserved_bytes(), 4096);
        assert_eq!(work.covered_bytes(), 1 + value.capacity());
        let capacity = value.capacity();
        drop(value);
        work.release(capacity);
        for changed in ["", "too long for the allocated buffer"] {
            let calls = std::cell::Cell::new(0);
            let error = work
                .format_string(kind, |out| {
                    calls.set(calls.get() + 1);
                    out.write_str(if calls.get() == 1 { "abc" } else { changed })
                })
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 1));
            assert_eq!(context.available(), 0);
        }
        let input = "x".repeat(64);
        let error = crate::test_alloc::fail_byte_allocation_once(64, || {
            work.format_string(kind, |out| out.write_str(&input))
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!((work.reserved_bytes(), work.covered_bytes()), (4096, 1));
        work.finish_aggregate();
        assert_eq!(work.reserved_bytes(), 1);
        drop(work);
        assert_eq!(context.available(), 4096);
    }

    #[test]
    fn object_keys_share_one_reservation_and_preserve_the_existing_owner() {
        let keys = ["alpha", "beta beta", "\u{4e2d}"];
        let bytes = object_workspace(keys.len()) + keys.iter().map(|key| key.len()).sum::<usize>();
        let seed = 13;
        for parent in [false, true] {
            for available in [bytes - 1, bytes, bytes + 1] {
                let limit = seed + available;
                let ledger = Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    limit,
                    if parent { limit } else { 2 * limit },
                ));
                let context = JsonOperationContext::new(ledger);
                let mut existing = context.reservation();
                existing.grow(seed, JsonInputKind::Filter).unwrap();
                let result = existing.reserve_object_keys(keys.into_iter(), JsonInputKind::Filter);
                if available < bytes {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
                    assert!(error.detail().iter().any(|(key, value)| {
                        key == "actual" && value == &(seed + bytes).to_string()
                    }));
                    assert_eq!(context.available(), available);
                } else {
                    let keys_work = result.unwrap();
                    assert_eq!(keys_work.reserved_bytes(), bytes);
                    assert_eq!(context.available(), available - bytes);
                    existing.merge(keys_work);
                    assert_eq!(existing.reserved_bytes(), seed + bytes);
                }
                drop(existing);
                assert_eq!(context.available(), limit);
            }
        }
    }

    #[test]
    fn heap_shrink_admits_overlap_and_moves_payloads_without_dropping_them() {
        use std::cell::Cell;
        use std::collections::BTreeMap;
        struct Owned<'a> {
            key: String,
            drops: &'a Cell<usize>,
        }
        impl PartialEq for Owned<'_> {
            fn eq(&self, other: &Self) -> bool {
                self.key == other.key
            }
        }
        impl Eq for Owned<'_> {}
        impl PartialOrd for Owned<'_> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Owned<'_> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.key.cmp(&other.key)
            }
        }
        impl Drop for Owned<'_> {
            fn drop(&mut self) {
                self.drops.set(self.drops.get() + 1);
            }
        }
        let size = std::mem::size_of::<Owned<'_>>();
        for seed in 0..100usize {
            let count = 1 + seed % 13;
            let capacity = count + 1 + seed;
            let active = 13 + seed;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let drops = Cell::new(0);
                    let mut heap = BinaryHeap::new();
                    heap.try_reserve_exact(capacity).unwrap();
                    for index in 0..count {
                        heap.push(Owned {
                            key: format!("key-{seed}-{index:03}"),
                            drops: &drops,
                        });
                    }
                    let payload = heap.iter().map(|item| item.key.capacity()).sum::<usize>();
                    let before = capacity * size + payload;
                    let needed = active + before + count * size;
                    let limit = (needed as isize + delta) as usize;
                    let ledger = Arc::new(WorkLedger::with_limits(
                        if parent { 2 * needed } else { limit },
                        0,
                        if parent { limit } else { 2 * needed },
                    ));
                    let context = JsonOperationContext::new(ledger.clone());
                    let other = JsonOperationContext::new(ledger);
                    let mut outside = other.workspace_reservation();
                    outside
                        .grow_to(active, active, "operation_scratch")
                        .unwrap();
                    let mut work = context.workspace_reservation();
                    work.grow_to(before, before, "operation_scratch").unwrap();
                    assert_eq!(heap.capacity(), capacity);
                    let array = heap.as_slice().as_ptr();
                    let snapshot = || {
                        heap.iter()
                            .map(|item| (item.key.clone(), item.key.as_ptr() as usize))
                            .collect::<BTreeMap<_, _>>()
                    };
                    let previous = snapshot();
                    let (changed, allocations) =
                        crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
                    assert_eq!(changed.unwrap(), delta >= 0);
                    assert_eq!(drops.get(), 0);
                    let actual = heap
                        .iter()
                        .map(|item| (item.key.clone(), item.key.as_ptr() as usize))
                        .collect::<BTreeMap<_, _>>();
                    assert_eq!(actual, previous);
                    let resident = if delta < 0 {
                        assert_eq!(heap.as_slice().as_ptr(), array);
                        assert_eq!(heap.capacity(), capacity);
                        before
                    } else {
                        assert_eq!(allocations, 1);
                        assert_eq!(heap.capacity(), count);
                        count * size + payload
                    };
                    assert_eq!(work.reserved_bytes(), resident);
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - resident
                    );
                    let sorted = heap.into_sorted_vec();
                    assert_eq!(drops.get(), 0);
                    assert_eq!(
                        sorted
                            .iter()
                            .map(|item| item.key.as_str())
                            .collect::<Vec<_>>(),
                        previous.keys().map(String::as_str).collect::<Vec<_>>()
                    );
                    drop(sorted);
                    assert_eq!(drops.get(), count);
                    drop(work);
                    drop(outside);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn heap_shrink_allocation_failure_keeps_old_array_and_allows_retry() {
        let size = std::mem::size_of::<[u8; 17]>();
        for count in 1..=100usize {
            let capacity = count + 7;
            let payload = 13;
            let before = payload + capacity * size;
            let needed = before + count * size;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(needed, 0, needed)));
            let mut work = context.workspace_reservation();
            work.grow_to(before, before, "operation_scratch").unwrap();
            let prefix = vec![12u8; payload];
            let mut heap = BinaryHeap::new();
            heap.try_reserve_exact(capacity).unwrap();
            for value in 0..count {
                heap.push([value as u8; 17]);
            }
            let array = heap.as_slice().as_ptr();
            let previous = heap.as_slice().to_vec();
            let error = crate::test_alloc::fail_byte_allocation_once(count * size, || {
                try_shrink_heap(&mut heap, &mut work)
            })
            .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert_eq!(heap.as_slice().as_ptr(), array);
            assert_eq!(heap.as_slice(), previous);
            assert_eq!(heap.capacity(), capacity);
            assert_eq!(work.reserved_bytes(), before);
            let (changed, allocations) = crate::test_alloc::key_allocations(count * size, || {
                try_shrink_heap(&mut heap, &mut work)
            });
            assert!(changed.unwrap());
            assert_eq!(allocations.allocated, 1);
            assert!(!allocations.overflow);
            assert_eq!(heap.capacity(), count);
            assert_eq!(work.reserved_bytes(), payload + count * size);
            assert_eq!(
                heap.into_sorted_vec(),
                previous
                    .into_iter()
                    .collect::<BinaryHeap<_>>()
                    .into_sorted_vec()
            );
            drop(prefix);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), needed);
        }
    }

    #[test]
    fn heap_shrink_empty_full_and_zero_sized_paths_do_not_allocate() {
        let size = std::mem::size_of::<[u8; 17]>();
        for count in 0..100usize {
            let payload = 13;
            let before = payload + count * size;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(before, 0, before)));
            let mut work = context.workspace_reservation();
            work.grow_to(before, before, "operation_scratch").unwrap();
            let prefix = vec![12u8; payload];
            let mut heap = BinaryHeap::new();
            heap.try_reserve_exact(count).unwrap();
            for value in 0..count {
                heap.push([value as u8; 17]);
            }
            let pointer = heap.as_slice().as_ptr();
            let (changed, allocated) =
                crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
            assert!(!changed.unwrap());
            assert_eq!(allocated, 0);
            assert_eq!(heap.as_slice().as_ptr(), pointer);
            assert_eq!(work.reserved_bytes(), before);
            heap.clear();
            let (changed, allocated) =
                crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
            assert_eq!(changed.unwrap(), count != 0);
            assert_eq!(allocated, 0);
            assert_eq!(heap.capacity(), 0);
            assert_eq!(work.reserved_bytes(), payload);
            drop(heap);
            drop(prefix);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), before);
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let mut work = context.workspace_reservation();
        let mut heap = BinaryHeap::new();
        heap.push(());
        let (changed, allocated) =
            crate::test_alloc::allocations(|| try_shrink_heap(&mut heap, &mut work));
        assert!(!changed.unwrap());
        assert_eq!(allocated, 0);
        assert_eq!(heap.len(), 1);
        assert_eq!(work.reserved_bytes(), 0);
    }

    #[test]
    fn heap_growth_admits_old_and_new_arrays_before_reallocation() {
        let size = std::mem::size_of::<[u8; 17]>();
        for seed in 0..100usize {
            let old = 1 + seed;
            let payload = 13 + seed;
            let active = 7 + seed * 3;
            let needed = active + payload + (old + old + 1) * size;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (needed as isize + delta) as usize;
                    let ledger = Arc::new(WorkLedger::with_limits(
                        if parent { needed * 2 } else { limit },
                        0,
                        if parent { limit } else { needed * 2 },
                    ));
                    let context = JsonOperationContext::new(ledger.clone());
                    let competing = JsonOperationContext::new(ledger);
                    let mut outside = competing.workspace_reservation();
                    outside
                        .grow_to(active, active, "operation_scratch")
                        .unwrap();
                    let mut work = context.workspace_reservation();
                    work.grow_to(
                        payload + old * size,
                        payload + old * size,
                        "operation_scratch",
                    )
                    .unwrap();
                    let prefix = vec![11u8; payload];
                    let mut heap = BinaryHeap::new();
                    heap.try_reserve_exact(old).unwrap();
                    for value in 0..old {
                        heap.push([value as u8; 17]);
                    }
                    assert_eq!(heap.capacity(), old);
                    let pointer = heap.as_slice().as_ptr();
                    let values = heap.as_slice().to_vec();
                    let (result, allocations) =
                        crate::test_alloc::key_allocations((old + 1) * size, || {
                            reserve_heap_slot(&mut heap, &mut work, usize::MAX)
                        });
                    if delta < 0 {
                        assert_memory_error(
                            &result.unwrap_err(),
                            "operation_scratch",
                            limit,
                            needed as u128,
                        );
                        assert_eq!(allocations.allocated, 0);
                        assert_eq!(heap.capacity(), old);
                        assert_eq!(heap.as_slice().as_ptr(), pointer);
                        assert_eq!(work.reserved_bytes(), payload + old * size);
                    } else {
                        result.unwrap();
                        assert_eq!(allocations.allocated, 1);
                        assert_eq!(heap.capacity(), old + 1);
                        assert_eq!(work.reserved_bytes(), payload + (old + 1) * size);
                    }
                    assert!(!allocations.overflow);
                    assert_eq!(heap.as_slice(), values);
                    if delta >= 0 {
                        heap.push([240; 17]);
                        assert_eq!(heap.peek(), Some(&[240; 17]));
                    }
                    assert_eq!(
                        context.available_workspace_bytes(),
                        limit - active - work.reserved_bytes()
                    );
                    drop(heap);
                    drop(prefix);
                    drop(work);
                    drop(outside);
                    assert_eq!(context.available_workspace_bytes(), limit);
                    let mut all = context.workspace_reservation();
                    all.grow_to(limit, limit, "operation_scratch").unwrap();
                    assert_memory_error(
                        &competing
                            .workspace_reservation()
                            .grow_to(1, 1, "operation_scratch")
                            .unwrap_err(),
                        "operation_scratch",
                        limit,
                        limit as u128 + 1,
                    );
                }
            }
        }
    }

    #[test]
    fn heap_growth_preference_and_spare_slots_keep_only_resident_credit() {
        let size = std::mem::size_of::<[u8; 17]>();
        for old in [0usize, 1, 4, 17, 99] {
            let payload = 23;
            let preferred = (old * 2).max(4);
            for new in [old + 1, (old + 1 + preferred) / 2, preferred] {
                let needed = payload + (old + new) * size;
                for remainder in [0, size - 1] {
                    let limit = needed + remainder;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        limit, 0, limit,
                    )));
                    let mut work = context.workspace_reservation();
                    work.grow_to(
                        payload + old * size,
                        payload + old * size,
                        "operation_scratch",
                    )
                    .unwrap();
                    let prefix = vec![12u8; payload];
                    let mut heap = BinaryHeap::new();
                    heap.try_reserve_exact(old).unwrap();
                    for value in 0..old {
                        heap.push([value as u8; 17]);
                    }
                    reserve_heap_slot(&mut heap, &mut work, usize::MAX).unwrap();
                    assert_eq!(heap.capacity(), new);
                    assert_eq!(work.reserved_bytes(), payload + new * size);
                    let pointer = heap.as_slice().as_ptr();
                    let mut remaining = context.workspace_reservation();
                    let free = limit - work.reserved_bytes();
                    remaining.grow_to(free, free, "operation_scratch").unwrap();
                    let (result, allocated) = crate::test_alloc::allocations(|| {
                        reserve_heap_slot(&mut heap, &mut work, usize::MAX)
                    });
                    result.unwrap();
                    assert_eq!(allocated, 0);
                    assert_eq!(heap.as_slice().as_ptr(), pointer);
                    assert_eq!(work.reserved_bytes(), payload + new * size);
                    heap.push([250; 17]);
                    assert_eq!(heap.peek(), Some(&[250; 17]));
                    drop(heap);
                    drop(prefix);
                    drop(work);
                    drop(remaining);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let mut work = context.workspace_reservation();
        let mut heap = BinaryHeap::<()>::new();
        let (result, allocated) =
            crate::test_alloc::allocations(|| reserve_heap_slot(&mut heap, &mut work, usize::MAX));
        result.unwrap();
        assert_eq!(allocated, 0);
        heap.push(());
        assert_eq!(heap.len(), 1);
        assert_eq!(work.reserved_bytes(), 0);
    }

    #[test]
    fn heap_reallocation_failure_preserves_old_owner_and_allows_retry() {
        let size = std::mem::size_of::<[u8; 17]>();
        for old in 1..=100usize {
            let payload = 13;
            let needed = payload + (old + old + 1) * size;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(needed, 0, needed)));
            let mut work = context.workspace_reservation();
            work.grow_to(
                payload + old * size,
                payload + old * size,
                "operation_scratch",
            )
            .unwrap();
            let prefix = vec![11u8; payload];
            let mut heap = BinaryHeap::new();
            heap.try_reserve_exact(old).unwrap();
            for value in 0..old {
                heap.push([value as u8; 17]);
            }
            let pointer = heap.as_slice().as_ptr();
            let values = heap.as_slice().to_vec();
            let error = crate::test_alloc::fail_byte_allocation_once((old + 1) * size, || {
                reserve_heap_slot(&mut heap, &mut work, usize::MAX)
            })
            .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert_eq!(heap.capacity(), old);
            assert_eq!(heap.as_slice().as_ptr(), pointer);
            assert_eq!(heap.as_slice(), values);
            assert_eq!(work.reserved_bytes(), payload + old * size);
            reserve_heap_slot(&mut heap, &mut work, usize::MAX).unwrap();
            heap.push([250; 17]);
            assert_eq!(heap.peek(), Some(&[250; 17]));
            assert_eq!(heap.capacity(), old + 1);
            assert_eq!(work.reserved_bytes(), payload + (old + 1) * size);
            drop(heap);
            drop(prefix);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), needed);
        }
    }

    #[test]
    fn workspace_cache_transfer_keeps_shared_credit_and_is_atomic_on_refusal() {
        for seed in 1..=100usize {
            let bytes = seed * 17;
            let ledger = Arc::new(WorkLedger::with_limits(bytes, bytes, bytes));
            let context = JsonOperationContext::new(ledger.clone());
            let competing = JsonOperationContext::new(ledger.clone());
            let mut work = context.workspace_reservation();
            work.grow_to(bytes, bytes, "operation_scratch").unwrap();
            let mut cache = context.cache_reservation(0, bytes).unwrap();
            assert!(!cache.absorb_workspace(&mut work, bytes, bytes - 1));
            assert_eq!(work.reserved_bytes(), bytes);
            assert_eq!(cache.reserved_bytes(), 0);
            let partial = bytes / 3;
            assert!(cache.absorb_workspace(&mut work, partial, bytes));
            assert_eq!(work.reserved_bytes(), bytes - partial);
            assert!(competing.cache_reservation(1, bytes + 1).is_none());
            assert!(competing
                .workspace_reservation()
                .grow_to(1, 1, "operation_scratch")
                .is_err());
            assert!(cache.absorb_workspace(&mut work, bytes - partial, bytes));
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(cache.reserved_bytes(), bytes);
            drop(work);
            assert!(competing.cache_reservation(1, bytes + 1).is_none());
            drop(cache);
            assert!(competing.cache_reservation(bytes as u128, bytes).is_some());
            let state = ledger.state.lock().unwrap();
            assert_eq!(
                (state.used, state.workspace_used, state.cache_used),
                (0, 0, 0)
            );
        }
    }

    #[test]
    fn workspace_cache_transfer_rejects_a_foreign_or_reattached_ledger() {
        for reattach in [false, true] {
            let first = Arc::new(WorkLedger::with_limits(1024, 1024, 1024));
            let second = Arc::new(WorkLedger::with_limits(1024, 1024, 1024));
            let cache_context = JsonOperationContext::new(first.clone());
            let source = JsonOperationContext::new(if reattach {
                first.clone()
            } else {
                second.clone()
            });
            let mut cache = cache_context.cache_reservation(0, 1024).unwrap();
            let mut work = source.workspace_reservation();
            work.grow_to(512, 512, "operation_scratch").unwrap();
            if reattach {
                source
                    .attach(second.clone(), JsonInputKind::StoredJson)
                    .unwrap();
            }
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cache.absorb_workspace(&mut work, 512, 1024)
            }))
            .is_err());
            assert_eq!(cache.reserved_bytes(), 0);
            assert_eq!(work.reserved_bytes(), 512);
            assert_eq!(second.state.lock().unwrap().workspace_used, 512);
            drop(work);
            assert_eq!(first.state.lock().unwrap().workspace_used, 0);
            assert_eq!(second.state.lock().unwrap().workspace_used, 0);
        }
    }

    #[test]
    fn concurrent_workspace_cache_transfers_never_double_charge_or_release_credit() {
        let limit = 4096;
        let ledger = Arc::new(WorkLedger::with_limits(limit, limit, limit));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let ledger = ledger.clone();
                scope.spawn(move || {
                    let context = JsonOperationContext::new(ledger.clone());
                    for _ in 0..1000 {
                        let mut work = context.workspace_reservation();
                        work.grow_to(512, 512, "operation_scratch").unwrap();
                        let mut cache = context.cache_reservation(0, limit).unwrap();
                        assert!(cache.absorb_workspace(&mut work, 512, limit));
                        let state = ledger.state.lock().unwrap();
                        assert!(state.workspace_used + state.cache_used <= limit);
                    }
                });
            }
        });
        let state = ledger.state.lock().unwrap();
        assert_eq!((state.workspace_used, state.cache_used), (0, 0));
    }

    #[test]
    fn cache_ownership_shares_parent_without_consuming_work_or_catalog_quota() {
        for seed in 0..100usize {
            let ledger = Arc::new(WorkLedger::with_limits(200, 600, 1000 + seed));
            let context = JsonOperationContext::new(Arc::clone(&ledger));
            let mut catalog = context.catalog_reservation();
            catalog.grow(500).unwrap();
            let mut work = context.workspace_reservation();
            work.grow_to(100, 100, "operation_scratch").unwrap();
            assert!(context.cache_reservation(u128::MAX, usize::MAX).is_none());
            assert!(context
                .cache_reservation(401 + seed as u128, 1000)
                .is_none());
            assert!(context
                .cache_reservation(400 + seed as u128, 399 + seed)
                .is_none());
            let mut cache = context
                .cache_reservation(400 + seed as u128, 400 + seed)
                .unwrap();
            assert_eq!(cache.reserved_bytes(), 400 + seed);
            assert_eq!(ledger.state.lock().unwrap().working_available(), 0);
            assert_memory_error(
                &work.grow_to(101, 101, "operation_scratch").unwrap_err(),
                "operation_scratch",
                1000 + seed,
                1001 + seed as u128,
            );
            assert_memory_error(
                &catalog.grow(1).unwrap_err(),
                "catalog_build",
                1000 + seed,
                1001 + seed as u128,
            );
            let json_error = context
                .reservation()
                .grow(1, JsonInputKind::StoredJson)
                .unwrap_err();
            assert_eq!(json_error.code(), crate::ErrorCode::RecoveryError);
            cache.retain(300 + seed);
            work.grow_to(200, 200, "operation_scratch").unwrap();
            assert!(context.cache_reservation(1, usize::MAX).is_none());
            drop(work);
            let other = JsonOperationContext::new(Arc::clone(&ledger));
            let second = other.cache_reservation(200, 500 + seed).unwrap();
            assert!(context.cache_reservation(1, 500 + seed).is_none());
            let target = Arc::new(WorkLedger::with_limits(200, 600, 1000));
            context
                .attach(Arc::clone(&target), JsonInputKind::StoredJson)
                .unwrap();
            assert_eq!(ledger.state.lock().unwrap().cache_used, 500 + seed);
            assert_eq!(target.state.lock().unwrap().cache_used, 0);
            drop(context);
            drop(cache);
            assert_eq!(ledger.state.lock().unwrap().cache_used, 200);
            drop(second);
            drop(catalog);
            let state = ledger.state.lock().unwrap();
            assert_eq!(
                (
                    state.used,
                    state.workspace_used,
                    state.catalog_used,
                    state.cache_used
                ),
                (0, 0, 0, 0)
            );
        }
    }

    #[test]
    fn simultaneous_cache_admission_cannot_exceed_the_shared_limit() {
        let ledger = Arc::new(WorkLedger::with_limits(1024, 1024, 1024));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let ledger = Arc::clone(&ledger);
                scope.spawn(move || {
                    let context = JsonOperationContext::new(Arc::clone(&ledger));
                    for _ in 0..1000 {
                        let cache = context.cache_reservation(256, 768);
                        let mut work = context.workspace_reservation();
                        let _ = work.grow_to(256, 256, "operation_scratch");
                        {
                            let state = ledger.state.lock().unwrap();
                            assert!(state.cache_used <= 768);
                            assert!(state.cache_used + state.workspace_used <= 1024);
                        }
                        drop(cache);
                    }
                });
            }
        });
        let state = ledger.state.lock().unwrap();
        assert_eq!((state.cache_used, state.workspace_used), (0, 0));
    }

    #[test]
    fn catalog_quota_and_shared_memory_rejections_identify_the_actual_ledger() {
        for seed in 0..100 {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                1024,
                600 + seed,
                800 + seed,
            )));
            let mut catalog = context.catalog_reservation();
            catalog.grow(600 + seed).unwrap();
            let error = catalog.grow(1).unwrap_err();
            let detail = error.to_json();
            let detail = detail.get("detail").unwrap();
            assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
            assert_eq!(detail.get("resource").unwrap().as_str(), Some("catalog"));
            assert_eq!(
                detail.get("operation").unwrap().as_str(),
                Some("catalog_build")
            );
            assert_eq!(
                detail.get("limit_bytes").unwrap().as_str(),
                Some((600 + seed).to_string().as_str())
            );
            assert_eq!(
                detail.get("requested_bytes").unwrap().as_str(),
                Some((601 + seed).to_string().as_str())
            );
            let mut workspace = context.workspace_reservation();
            assert_eq!(
                workspace.grow_to(200, 1024, "operation_scratch").unwrap(),
                200
            );
            assert_memory_error(
                &workspace
                    .grow_to(201, 1024, "operation_scratch")
                    .unwrap_err(),
                "operation_scratch",
                800 + seed,
                801 + seed as u128,
            );
            catalog.retain(500 + seed);
            workspace.grow_to(300, 300, "operation_scratch").unwrap();
            assert_memory_error(
                &catalog.grow(1).unwrap_err(),
                "catalog_build",
                800 + seed,
                801 + seed as u128,
            );
            let error = context
                .reservation()
                .grow(1, JsonInputKind::StoredJson)
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "0"));
            drop(workspace);
            drop(catalog);
            context
                .workspace_reservation()
                .grow_to(800 + seed, 800 + seed, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn catalog_transfer_is_atomic_preserves_memory_and_survives_context_reattachment() {
        for seed in 0..100 {
            let ledger = Arc::new(WorkLedger::with_limits(1024, 900, 800));
            let context = JsonOperationContext::new(Arc::clone(&ledger));
            let mut catalog = context.catalog_reservation();
            catalog.grow(400).unwrap();
            let mut json = context.reservation();
            json.grow(200 + seed, JsonInputKind::StoredJson).unwrap();
            let mut workspace = context.workspace_reservation();
            workspace
                .grow_to(200 - seed, 200 - seed, "operation_scratch")
                .unwrap();
            catalog
                .absorb(&mut json, Some(&mut workspace), 100 + seed, 100 - seed)
                .unwrap();
            assert_eq!(catalog.reserved_bytes(), 600);
            assert_eq!(json.reserved_bytes(), 100);
            assert_eq!(workspace.reserved_bytes(), 100);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!(
                    (state.used, state.workspace_used, state.catalog_used),
                    (100, 100, 600)
                );
            }
            assert_memory_error(&catalog.grow(1).unwrap_err(), "catalog_build", 800, 801);
            let target = Arc::new(WorkLedger::with_limits(1024, 900, 800));
            context
                .attach(Arc::clone(&target), JsonInputKind::StoredJson)
                .unwrap();
            {
                let source = ledger.state.lock().unwrap();
                let target = target.state.lock().unwrap();
                assert_eq!(
                    (source.used, source.workspace_used, source.catalog_used),
                    (0, 0, 600)
                );
                assert_eq!(
                    (target.used, target.workspace_used, target.catalog_used),
                    (100, 100, 0)
                );
            }
            drop(json);
            drop(workspace);
            drop(catalog);
            assert_eq!(ledger.state.lock().unwrap().catalog_used, 0);
            assert_eq!(target.state.lock().unwrap().working_available(), 800);
        }
    }

    #[test]
    fn failed_catalog_transfer_keeps_source_credit_and_shared_json_error_priority() {
        for seed in 0..100 {
            let ledger = Arc::new(WorkLedger::with_limits(1024, 200 + seed, 300 + seed));
            let context = JsonOperationContext::new(Arc::clone(&ledger));
            let mut catalog = context.catalog_reservation();
            catalog.grow(150 + seed).unwrap();
            let mut json = context.reservation();
            json.grow(120, JsonInputKind::StoredJson).unwrap();
            let mut workspace = context.workspace_reservation();
            workspace.grow_to(30, 30, "operation_scratch").unwrap();
            let error = catalog
                .absorb(&mut json, Some(&mut workspace), 120, 30)
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "resource" && value == "catalog"));
            assert_eq!(catalog.reserved_bytes(), 150 + seed);
            assert_eq!(json.reserved_bytes(), 120);
            assert_eq!(workspace.reserved_bytes(), 30);
            drop(workspace);
            let error = json.grow(31, JsonInputKind::StoredJson).unwrap_err();
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "150"));
            let target = Arc::new(WorkLedger::with_limits(1024, 200, 200));
            let target_context = JsonOperationContext::new(Arc::clone(&target));
            let mut occupied = target_context.catalog_reservation();
            occupied.grow(190).unwrap();
            let error = context
                .attach(Arc::clone(&target), JsonInputKind::StoredJson)
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "10"));
            assert_eq!(ledger.state.lock().unwrap().used, 120);
            assert_eq!(target.state.lock().unwrap().used, 0);
        }
    }

    fn assert_memory_error(error: &Error, phase: &str, limit: usize, requested: u128) {
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        let json = error.to_json();
        let detail = json.get("detail").unwrap();
        assert_eq!(detail.as_object().unwrap().len(), 4);
        for (key, expected) in [
            ("resource", "memory".to_owned()),
            ("operation", phase.to_owned()),
            ("limit_bytes", limit.to_string()),
            ("requested_bytes", requested.to_string()),
        ] {
            assert_eq!(detail.get(key).unwrap().as_str(), Some(expected.as_str()));
        }
    }

    #[test]
    fn reserved_strings_charge_utf8_bytes_before_copying_and_release_on_drop() {
        let limit = 4096;
        for seed in 0..100usize {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let value = "x\u{4e2d}\n".repeat(seed);
            let mut other = context.workspace_reservation();
            other
                .grow_to(
                    limit - value.len(),
                    limit - value.len(),
                    "operation_scratch",
                )
                .unwrap();
            let owned = ReservedString::copy(&value, &context).unwrap();
            assert_eq!(owned.as_ref(), value);
            assert!(other
                .grow_to(
                    limit - value.len() + 1,
                    limit - value.len() + 1,
                    "operation_scratch"
                )
                .is_err());
            drop(owned);
            other.grow_to(limit, limit, "operation_scratch").unwrap();
            if !value.is_empty() {
                other.retain(limit - value.len() + 1);
                let error = ReservedString::copy(&value, &context).err().unwrap();
                assert_memory_error(&error, "operation_scratch", limit, limit as u128 + 1);
                other.grow_to(limit, limit, "operation_scratch").unwrap();
            }
        }
    }

    #[test]
    fn generated_strings_admit_exact_bytes_and_bound_the_rendering_pass() {
        use std::cell::Cell;
        for seed in 0..100usize {
            let limit = 4096;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let text = "x\u{4e2d}".repeat(seed);
            let expected = format!("{text}:{seed:016}");
            let calls = Cell::new(0);
            let render = |out: &mut dyn std::fmt::Write| {
                calls.set(calls.get() + 1);
                write!(out, "{text}:{seed:016}")
            };
            let mut held = context.workspace_reservation();
            let occupied = limit - expected.len();
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let result = ReservedString::build(&context, "batch_prepare", render).unwrap();
            assert_eq!(result.as_ref(), expected);
            assert_eq!(result.value.capacity(), expected.len());
            assert_eq!(calls.get(), 2);
            drop(result);
            calls.set(0);
            held.grow_to(occupied + 1, occupied + 1, "operation_scratch")
                .unwrap();
            let error = ReservedString::build(&context, "batch_prepare", render)
                .err()
                .unwrap();
            assert_memory_error(&error, "batch_prepare", limit, limit as u128 + 1);
            assert_eq!(
                calls.get(),
                1,
                "admission must precede the heap rendering pass"
            );
            held.retain(0);
            for changed in ["", "longer"] {
                let calls = Cell::new(0);
                let error = ReservedString::build(&context, "batch_prepare", |out| {
                    calls.set(calls.get() + 1);
                    out.write_str(if calls.get() == 1 { "x" } else { changed })
                })
                .err()
                .unwrap();
                assert_eq!(error.code(), crate::ErrorCode::IoError);
                assert_eq!(calls.get(), 2);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(0);
            }
            for fail_at in [1, 2] {
                let calls = Cell::new(0);
                let error = ReservedString::build(&context, "batch_prepare", |out| {
                    calls.set(calls.get() + 1);
                    out.write_str("x")?;
                    if calls.get() == fail_at {
                        Err(std::fmt::Error)
                    } else {
                        Ok(())
                    }
                })
                .err()
                .unwrap();
                assert_eq!(error.code(), crate::ErrorCode::IoError);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(0);
            }
            assert!(ReservedString::build(&context, "batch_prepare", |_| Ok(()))
                .unwrap()
                .as_ref()
                .is_empty());
        }
    }

    #[test]
    fn typed_copy_transfer_preserves_shared_usage_without_reacquiring_credit() {
        for seed in 0..100usize {
            let limit = 2048 + seed;
            let ledger = Arc::new(WorkLedger::new(limit));
            let context = JsonOperationContext::new(ledger.clone());
            let mut json = context.reservation();
            json.grow(limit, JsonInputKind::StoredJson).unwrap();
            let moved = 1 + seed * 7;
            let work = json.take_workspace(moved);
            assert_eq!(json.reserved_bytes(), limit - moved);
            assert_eq!(work.reserved_bytes(), moved);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (limit - moved, moved));
            }
            let competing = JsonOperationContext::new(ledger.clone());
            assert_memory_error(
                &competing
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .unwrap_err(),
                "operation_scratch",
                limit,
                limit as u128 + 1,
            );
            assert!(competing
                .reservation()
                .grow(1, JsonInputKind::StoredJson)
                .is_err());
            let destination = Arc::new(WorkLedger::new(limit));
            context
                .attach(destination.clone(), JsonInputKind::StoredJson)
                .unwrap();
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (0, 0));
            }
            drop(json);
            let mut other = context.reservation();
            other
                .grow(limit - moved, JsonInputKind::StoredJson)
                .unwrap();
            assert!(other.grow(1, JsonInputKind::StoredJson).is_err());
            drop(work);
            other.grow(moved, JsonInputKind::StoredJson).unwrap();
            drop(other);
            let state = destination.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn owned_string_transfer_never_reacquires_credit_and_keeps_json_limit_on_rejection() {
        for seed in 0..100usize {
            let limit = 2048 + seed;
            let ledger = Arc::new(WorkLedger::new(limit));
            let context = JsonOperationContext::new(ledger.clone());
            let mut work = context.workspace_reservation();
            work.grow_to(limit, limit, "operation_scratch").unwrap();
            let moved = 1 + seed * 7;
            let json = work.take_json(moved, JsonInputKind::StoredJson).unwrap();
            assert_eq!(json.reserved_bytes(), moved);
            assert_eq!(work.reserved_bytes(), limit - moved);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (moved, limit - moved));
            }
            let other = JsonOperationContext::new(ledger.clone());
            assert_memory_error(
                &other
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .unwrap_err(),
                "operation_scratch",
                limit,
                limit as u128 + 1,
            );
            let destination = Arc::new(WorkLedger::new(limit));
            context
                .attach(destination.clone(), JsonInputKind::StoredJson)
                .unwrap();
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (0, 0));
            }
            drop(json);
            drop(work);
            {
                let state = destination.state.lock().unwrap();
                assert_eq!((state.used, state.workspace_used), (0, 0));
            }

            let json_limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let ledger = Arc::new(WorkLedger::new(json_limit + 1024));
            let context = JsonOperationContext::new(ledger.clone());
            let mut json = context.reservation();
            json.grow(json_limit - seed, JsonInputKind::StoredJson)
                .unwrap();
            let mut work = context.workspace_reservation();
            work.grow_to(seed + 1, seed + 1, "operation_scratch")
                .unwrap();
            let error = work
                .take_json(seed + 1, JsonInputKind::StoredJson)
                .err()
                .unwrap();
            assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == &json_limit.to_string()));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "actual" && value == &(json_limit + 1).to_string()));
            assert_eq!(work.reserved_bytes(), seed + 1);
            {
                let state = ledger.state.lock().unwrap();
                assert_eq!(
                    (state.used, state.workspace_used),
                    (json_limit - seed, seed + 1)
                );
            }
            drop(json);
            let json = work.take_json(seed + 1, JsonInputKind::StoredJson).unwrap();
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(json.reserved_bytes(), seed + 1);
            drop(json);
            let state = ledger.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn reserved_arrays_admit_exact_capacity_and_preserve_wide_error_counts() {
        for seed in 0..100usize {
            let count = seed + 1;
            let bytes = count * std::mem::size_of::<u64>();
            let limit = 2048 + seed;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let mut held = context.workspace_reservation();
            held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                .unwrap();
            let mut values =
                ReservedVec::with_capacity(count, &context, "operation_scratch").unwrap();
            for index in 0..count {
                values.push(index as u64);
            }
            assert_eq!(values._work.bytes, bytes);
            assert_eq!(values.values.capacity(), count);
            let error = ReservedVec::<u8>::with_capacity(1, &context, "operation_scratch")
                .err()
                .unwrap();
            assert_memory_error(&error, "operation_scratch", limit, limit as u128 + 1);
            let public_values = values.into_vec();
            assert_eq!(public_values, (0..count as u64).collect::<Vec<_>>());
            let wide =
                ReservedVec::<[u8; 16]>::with_capacity(usize::MAX, &context, "operation_scratch")
                    .err()
                    .unwrap();
            assert_memory_error(
                &wide,
                "operation_scratch",
                limit,
                (limit - bytes) as u128 + (usize::MAX as u128) * 16,
            );
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            let empty =
                ReservedVec::<u64>::with_capacity(0, &context, "operation_scratch").unwrap();
            assert!(empty.as_slice().is_empty());
            let mut zero_sized =
                ReservedVec::<()>::with_capacity(count, &context, "operation_scratch").unwrap();
            for _ in 0..count {
                zero_sized.push(());
            }
            assert_eq!(zero_sized.as_slice().len(), count);
            assert_eq!(zero_sized._work.bytes, 0);
        }
    }

    #[test]
    fn reserved_arrays_growth_admits_coexisting_arrays_on_both_ledgers() {
        let size = std::mem::size_of::<[u8; 257]>();
        for old in [0usize, 1, 4, 17, 99] {
            let active = 23;
            let minimum = active + (2 * old + 1) * size;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (minimum as isize + delta) as usize;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { 2 * minimum } else { limit },
                        0,
                        if parent { limit } else { 2 * minimum },
                    )));
                    let mut held = context.workspace_reservation();
                    held.grow_to(active, active, "operation_scratch").unwrap();
                    let mut values =
                        ReservedVec::with_capacity(old, &context, "operation_scratch").unwrap();
                    for index in 0..old {
                        values.push([index as u8; 257]);
                    }
                    let pointer = values.as_slice().as_ptr();
                    let (result, allocations) =
                        crate::test_alloc::key_allocations((old + 1) * size, || {
                            let result = values.try_push([240; 257]);
                            if let Err(error) = &result {
                                assert_memory_error(
                                    error,
                                    "operation_scratch",
                                    limit,
                                    minimum as u128,
                                );
                                assert_eq!(values.capacity(), old);
                                assert_eq!(values.as_slice().as_ptr(), pointer);
                                assert_eq!(values._work.reserved_bytes(), old * size);
                                assert_eq!(values.as_slice().len(), old);
                            } else {
                                assert_eq!(values.capacity(), old + 1);
                                assert_eq!(values._work.reserved_bytes(), (old + 1) * size);
                                assert_eq!(values.as_slice().last(), Some(&[240; 257]));
                            }
                            for (index, value) in values.as_slice()[..old].iter().enumerate() {
                                assert_eq!(*value, [index as u8; 257]);
                            }
                            drop(values);
                            result.map_err(drop)
                        });
                    assert_eq!(result.is_ok(), delta >= 0);
                    assert_eq!(allocations.allocated, usize::from(delta >= 0));
                    assert_eq!(allocations.live, 0);
                    assert!(!allocations.overflow);
                    assert_eq!(context.available_workspace_bytes(), limit - active);
                    drop(held);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn reserved_arrays_native_growth_failure_keeps_data_and_credit_for_retry() {
        let old = 17;
        let size = std::mem::size_of::<[u8; 257]>();
        let limit = 3 * old * size;
        let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
        let mut values = ReservedVec::with_capacity(old, &context, "operation_scratch").unwrap();
        for index in 0..old {
            values.push([index as u8; 257]);
        }
        let pointer = values.as_slice().as_ptr();
        let error = crate::test_alloc::fail_byte_allocation_once(2 * old * size, || {
            values.try_push([240; 257])
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(values.capacity(), old);
        assert_eq!(values.as_slice().as_ptr(), pointer);
        assert_eq!(values._work.reserved_bytes(), old * size);
        for (index, value) in values.as_slice().iter().enumerate() {
            assert_eq!(*value, [index as u8; 257]);
        }
        values.try_push([240; 257]).unwrap();
        assert_eq!(values.capacity(), 2 * old);
        assert_eq!(values._work.reserved_bytes(), 2 * old * size);
        assert_eq!(values.as_slice().last(), Some(&[240; 257]));
        drop(values);
        assert_eq!(context.available_workspace_bytes(), limit);

        let context = JsonOperationContext::new(Arc::new(WorkLedger::new(0)));
        let mut units = ReservedVec::with_capacity(0, &context, "operation_scratch").unwrap();
        for _ in 0..100 {
            units.try_push(()).unwrap();
        }
        assert_eq!(units.as_slice().len(), 100);
        assert_eq!(units._work.reserved_bytes(), 0);
    }

    #[test]
    fn reserved_arrays_keep_the_lease_until_all_elements_are_dropped() {
        struct Probe {
            context: JsonOperationContext,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                assert!(self
                    .context
                    .workspace_reservation()
                    .grow_to(1, 1, "operation_scratch")
                    .is_err());
            }
        }
        for count in 1..=100 {
            let bytes = count * std::mem::size_of::<Probe>();
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(bytes)));
            let mut values =
                ReservedVec::with_capacity(count, &context, "operation_scratch").unwrap();
            for _ in 0..count {
                values.push(Probe {
                    context: context.clone(),
                });
            }
            drop(values);
            context
                .workspace_reservation()
                .grow_to(bytes, bytes, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn json_and_workspace_share_one_limit_at_100_exact_boundaries() {
        for extra in 0..100 {
            let ledger = Arc::new(WorkLedger::new(1024 + extra));
            let context = JsonOperationContext::new(ledger.clone());
            let mut json = context.reservation();
            json.grow(512, JsonInputKind::JsonlRecord).unwrap();
            let mut work = context.workspace_reservation();
            assert_eq!(
                work.grow_to(512, usize::MAX, "catalog_build").unwrap(),
                512 + extra
            );
            let error = work
                .grow_to(513 + extra, usize::MAX, "catalog_build")
                .unwrap_err();
            assert_memory_error(&error, "catalog_build", 1024 + extra, 1025 + extra as u128);
            assert_eq!(work.bytes, 512 + extra);
            assert_eq!(context.available(), 0);
            let error = json.grow(1, JsonInputKind::JsonlRecord).unwrap_err();
            assert!(super::super::is_limit_error(&error));
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("limit")
                    .unwrap()
                    .as_f64(),
                Some(512.0)
            );
            drop(work);
            json.grow(512 + extra, JsonInputKind::JsonlRecord).unwrap();
            drop(json);
            let state = ledger.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn workspace_transfer_is_atomic_and_keeps_both_ownership_views() {
        let source = Arc::new(WorkLedger::new(1024));
        let target = Arc::new(WorkLedger::new(1024));
        let context = JsonOperationContext::new(source.clone());
        let mut json = context.reservation();
        json.grow(200, JsonInputKind::Filter).unwrap();
        let mut work = context.workspace_reservation();
        work.grow_to(500, 500, "catalog_build").unwrap();
        let mut busy = JsonOperationContext::new(target.clone()).reservation();
        busy.grow(400, JsonInputKind::Filter).unwrap();
        let error = context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap_err();
        assert_memory_error(&error, "operation_scratch", 1024, 1100);
        assert_eq!(source.state.lock().unwrap().workspace_used, 500);
        assert_eq!(target.state.lock().unwrap().workspace_used, 0);
        busy.release(100);
        context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap();
        context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap();
        assert_eq!(source.state.lock().unwrap().workspace_used, 0);
        assert_eq!(source.state.lock().unwrap().used, 0);
        let mut later = context.clone().workspace_reservation();
        assert_eq!(later.grow_to(1, 100, "catalog_build").unwrap(), 24);
        drop(work);
        drop(json);
        drop(context);
        assert_eq!(target.state.lock().unwrap().workspace_used, 24);
        drop(later);
        drop(busy);
        let state = target.state.lock().unwrap();
        assert_eq!((state.used, state.workspace_used), (0, 0));
    }

    #[test]
    fn json_transfer_rejection_keeps_priority_with_live_workspace() {
        let source = Arc::new(WorkLedger::new(1024));
        let target = Arc::new(WorkLedger::new(1024));
        let context = JsonOperationContext::new(source.clone());
        let mut json = context.reservation();
        json.grow(600, JsonInputKind::Filter).unwrap();
        let mut work = context.workspace_reservation();
        work.grow_to(100, 100, "catalog_build").unwrap();
        let mut busy = JsonOperationContext::new(target.clone()).workspace_reservation();
        busy.grow_to(500, 500, "catalog_build").unwrap();
        let error = context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap_err();
        assert!(super::super::is_limit_error(&error));
        assert_eq!(context.available(), 324);
        drop(busy);
        context
            .attach(target.clone(), JsonInputKind::Filter)
            .unwrap();
        assert_eq!(context.available(), 324);
        drop(json);
        drop(work);
        assert_eq!(target.state.lock().unwrap().workspace_used, 0);
    }

    #[test]
    fn workspace_minimum_totals_are_wider_than_usize_and_preference_never_rejects() {
        for limit in [0, 1, 1024, usize::MAX] {
            let context = JsonOperationContext::new(Arc::new(WorkLedger::new(limit)));
            let mut first = context.workspace_reservation();
            assert_eq!(
                first.grow_to(0, usize::MAX, "operation_scratch").unwrap(),
                limit
            );
            assert_eq!(first.grow_to(0, 0, "operation_scratch").unwrap(), limit);
            let mut second = context.workspace_reservation();
            let error = second
                .grow_to(usize::MAX, usize::MAX, "operation_scratch")
                .unwrap_err();
            assert_memory_error(
                &error,
                "operation_scratch",
                limit,
                limit as u128 + usize::MAX as u128,
            );
            assert_eq!(second.bytes, 0);
            drop(first);
            assert_eq!(
                second.grow_to(0, usize::MAX, "operation_scratch").unwrap(),
                limit
            );
        }
    }

    #[test]
    fn mixed_concurrent_reservations_and_transfers_never_overcommit() {
        let ledgers = [
            Arc::new(WorkLedger::new(1024)),
            Arc::new(WorkLedger::new(1024)),
        ];
        let workers: Vec<_> = (0..8)
            .map(|index| {
                let ledgers = ledgers.clone();
                std::thread::spawn(move || {
                    let context = JsonOperationContext::new(ledgers[index % 2].clone());
                    for step in 0..1000 {
                        let mut json = if step % 2 == 0 {
                            context.aggregate_reservation()
                        } else {
                            context.reservation()
                        };
                        let mut work = context.workspace_reservation();
                        let _ = json.grow(64, JsonInputKind::Filter);
                        let _ = work.grow_to(128, 192, "catalog_build");
                        let _ = context
                            .attach(ledgers[(index + step) % 2].clone(), JsonInputKind::Filter);
                        for ledger in &ledgers {
                            let state = ledger.state.lock().unwrap();
                            assert!(state.used + state.workspace_used <= state.limit);
                        }
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        for ledger in ledgers {
            let state = ledger.state.lock().unwrap();
            assert_eq!((state.used, state.workspace_used), (0, 0));
        }
    }

    #[test]
    fn concurrent_catalog_promotions_share_the_actual_memory_ceiling() {
        let ledger = Arc::new(WorkLedger::with_limits(1024, 1536, 2048));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let ledger = Arc::clone(&ledger);
                std::thread::spawn(move || {
                    let context = JsonOperationContext::new(Arc::clone(&ledger));
                    for _ in 0..1000 {
                        let mut json = context.reservation();
                        let mut work = context.workspace_reservation();
                        let mut catalog = context.catalog_reservation();
                        let _ = json.grow(80, JsonInputKind::StoredJson);
                        let _ = work.grow_to(96, 144, "catalog_build");
                        let json_bytes = json.reserved_bytes();
                        let work_bytes = work.reserved_bytes();
                        let _ = catalog.absorb(&mut json, Some(&mut work), json_bytes, work_bytes);
                        let _ = catalog.grow(64);
                        let state = ledger.state.lock().unwrap();
                        assert!(state.used + state.workspace_used <= state.limit);
                        assert!(state.catalog_used <= state.catalog_limit);
                        assert!(
                            state.used + state.workspace_used + state.catalog_used
                                <= state.total_limit
                        );
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let state = ledger.state.lock().unwrap();
        assert_eq!(
            (state.used, state.workspace_used, state.catalog_used),
            (0, 0, 0)
        );
    }

    #[test]
    fn context_transfer_keeps_live_usage_and_rolls_back_a_failed_adoption() {
        let source = Arc::new(WorkLedger::new(1024));
        let target = Arc::new(WorkLedger::new(1024));
        let context = JsonOperationContext::new(Arc::clone(&source));
        let clone = context.clone();
        let mut input = context.reservation();
        input.grow(700, JsonInputKind::Filter).unwrap();
        let mut busy = JsonOperationContext::new(Arc::clone(&target)).reservation();
        busy.grow(400, JsonInputKind::Filter).unwrap();
        let error = context
            .attach(Arc::clone(&target), JsonInputKind::Filter)
            .unwrap_err();
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("limit")
                .unwrap()
                .as_f64(),
            Some(624.0)
        );
        assert_eq!(source.state.lock().unwrap().used, 700);
        assert_eq!(target.state.lock().unwrap().used, 400);
        busy.release(100);
        context
            .attach(Arc::clone(&target), JsonInputKind::Filter)
            .unwrap();
        assert_eq!(source.state.lock().unwrap().used, 0);
        assert_eq!(target.state.lock().unwrap().used, 1000);
        context
            .attach(Arc::clone(&target), JsonInputKind::Filter)
            .unwrap();
        let mut next = clone.reservation();
        next.grow(24, JsonInputKind::Filter).unwrap();
        assert!(next.grow(1, JsonInputKind::Filter).is_err());
        drop(input);
        assert_eq!(target.state.lock().unwrap().used, 324);
        drop(next);
        drop(busy);
        assert_eq!(target.state.lock().unwrap().used, 0);
    }

    #[test]
    fn opposite_concurrent_context_transfers_preserve_accounting() {
        let ledgers = [
            Arc::new(WorkLedger::new(1024)),
            Arc::new(WorkLedger::new(1024)),
        ];
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let threads: Vec<_> = (0..2)
            .map(|index| {
                let ledgers = ledgers.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let context = JsonOperationContext::new(Arc::clone(&ledgers[index]));
                    let mut held = context.reservation();
                    held.grow(64, JsonInputKind::Filter).unwrap();
                    barrier.wait();
                    for step in 0..1000 {
                        context
                            .attach(
                                Arc::clone(&ledgers[(index + step) % 2]),
                                JsonInputKind::Filter,
                            )
                            .unwrap();
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        for ledger in ledgers {
            assert_eq!(ledger.state.lock().unwrap().used, 0);
        }
    }

    #[test]
    fn simultaneous_steps_share_one_quota_and_released_capacity_is_reusable() {
        let ledger = Arc::new(WorkLedger::new(1024));
        let operation = JsonOperationContext::new(Arc::clone(&ledger));
        let mut input = operation.reservation();
        let mut copy = operation.clone().reservation();
        input.grow(700, JsonInputKind::Filter).unwrap();
        copy.grow(324, JsonInputKind::Filter).unwrap();
        let error = copy.grow(1, JsonInputKind::Filter).unwrap_err();
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("limit")
                .unwrap()
                .as_f64(),
            Some(1024.0)
        );
        assert_eq!(ledger.state.lock().unwrap().used, 1024);
        drop(input);
        copy.grow(700, JsonInputKind::Filter).unwrap();
        drop(copy);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn merging_reservations_preserves_ownership_without_charging_twice() {
        let ledger = Arc::new(WorkLedger::new(1024));
        let operation = JsonOperationContext::new(Arc::clone(&ledger));
        let mut first = operation.reservation();
        let mut second = operation.reservation();
        first.grow(512, JsonInputKind::DocumentMetadata).unwrap();
        second.grow(512, JsonInputKind::ChunkMetadata).unwrap();
        first.merge(second);
        assert_eq!(ledger.state.lock().unwrap().used, 1024);
        drop(operation);
        drop(first);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn different_operations_observe_effective_available_limit_without_double_charging() {
        let ledger = Arc::new(WorkLedger::new(2000));
        let first = JsonOperationContext::new(Arc::clone(&ledger));
        let second = JsonOperationContext::new(Arc::clone(&ledger));
        let mut held = first.reservation();
        held.grow(800, JsonInputKind::DocumentMetadata).unwrap();
        let mut other = second.reservation();
        other.grow(1200, JsonInputKind::StoredJson).unwrap();
        let error = other.grow(1, JsonInputKind::StoredJson).unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("limit")
                .unwrap()
                .as_f64(),
            Some(1200.0)
        );
        drop(held);
        other.grow(800, JsonInputKind::StoredJson).unwrap();
        drop(other);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn operation_limit_applies_even_when_the_shared_ledger_is_larger() {
        let operation = JsonOperationContext::new(Arc::new(WorkLedger::new(usize::MAX)));
        let mut reservation = operation.reservation();
        reservation
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::Weights,
            )
            .unwrap();
        let error = reservation
            .grow(usize::MAX, JsonInputKind::Weights)
            .unwrap_err();
        let json = error.to_json();
        let detail = json.get("detail").unwrap();
        assert_eq!(
            detail.get("limit").unwrap().as_f64(),
            Some(JsonProcessingPolicyV1::WORK_BUDGET_BYTES as f64)
        );
        assert!(detail.get("actual").is_none());
        assert_eq!(
            detail.get("weights_reason").unwrap().as_str(),
            Some("json_processing_limit")
        );
    }

    #[test]
    fn rejected_buffer_growth_does_not_allocate_or_consume_a_reservation() {
        let ledger = Arc::new(WorkLedger::new(80));
        let operation = JsonOperationContext::new(Arc::clone(&ledger));
        let mut reservation = operation.reservation();
        let mut values = Vec::<u64>::new();
        reservation
            .reserve_vec(&mut values, 10, JsonInputKind::JsonlRecord)
            .unwrap();
        values.resize(10, 0);
        assert!(reservation
            .reserve_vec(&mut values, 1, JsonInputKind::JsonlRecord)
            .is_err());
        assert_eq!(values.capacity(), 10);
        assert_eq!(ledger.state.lock().unwrap().used, 80);
        drop(reservation);
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }

    #[test]
    fn json_byte_buffers_native_failure_preserves_data_and_refunds_only_new_credit() {
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::Filter,
            JsonInputKind::Weights,
            JsonInputKind::MetadataBoost,
            JsonInputKind::StoredJson,
        ] {
            let context = JsonOperationContext::standalone();
            let capacity = context.available_workspace_bytes();
            let mut other = context.reservation();
            other.grow(17, kind).unwrap();
            let baseline = context.available_workspace_bytes();
            let bytes = 8193;
            let mut work = context.reservation();
            let mut values = Vec::<u8>::new();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                work.reserve_vec(&mut values, bytes, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert!(!super::super::is_memory_budget_error(&error));
            assert!(values.is_empty());
            assert_eq!(values.capacity(), 0);
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(context.available_workspace_bytes(), baseline);
            work.reserve_vec(&mut values, bytes, kind).unwrap();
            values.resize(bytes, 23);
            let pointer = values.as_ptr();
            let error = crate::test_alloc::fail_byte_allocation_once(2 * bytes, || {
                work.reserve_vec(&mut values, 1, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(values.as_ptr(), pointer);
            assert_eq!(values.capacity(), bytes);
            assert_eq!(values.len(), bytes);
            assert!(values.iter().all(|byte| *byte == 23));
            assert_eq!(work.reserved_bytes(), bytes);
            assert_eq!(context.available_workspace_bytes(), baseline - bytes);
            work.reserve_vec(&mut values, 1, kind).unwrap();
            values.push(24);
            assert_eq!(values.last(), Some(&24));
            drop(values);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), baseline);
            drop(other);
            assert_eq!(context.available_workspace_bytes(), capacity);
        }
    }

    #[test]
    fn json_strings_native_failure_preserves_data_and_refunds_only_new_credit() {
        let payload = "x".repeat(8193);
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::Filter,
            JsonInputKind::Weights,
            JsonInputKind::MetadataBoost,
            JsonInputKind::StoredJson,
        ] {
            let context = JsonOperationContext::standalone();
            let capacity = context.available_workspace_bytes();
            let mut other = context.reservation();
            other.grow(17, kind).unwrap();
            let baseline = context.available_workspace_bytes();
            let bytes = payload.len();
            let mut work = context.reservation();
            let mut value = String::new();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                work.reserve_string(&mut value, bytes, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert!(!super::super::is_memory_budget_error(&error));
            assert!(value.is_empty());
            assert_eq!(value.capacity(), 0);
            assert_eq!(work.reserved_bytes(), 0);
            assert_eq!(context.available_workspace_bytes(), baseline);
            work.reserve_string(&mut value, bytes, kind).unwrap();
            value.push_str(&payload);
            let pointer = value.as_ptr();
            let error = crate::test_alloc::fail_byte_allocation_once(2 * bytes, || {
                work.reserve_string(&mut value, 1, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(value.as_ptr(), pointer);
            assert_eq!(value.capacity(), bytes);
            assert_eq!(value, payload);
            assert_eq!(work.reserved_bytes(), bytes);
            assert_eq!(context.available_workspace_bytes(), baseline - bytes);
            work.reserve_string(&mut value, 1, kind).unwrap();
            value.push('y');
            assert!(value.ends_with('y'));
            drop(value);
            drop(work);
            assert_eq!(context.available_workspace_bytes(), baseline);
            drop(other);
            assert_eq!(context.available_workspace_bytes(), capacity);
        }
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_shared_ledger() {
        let ledger = Arc::new(WorkLedger::new(1024));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let ledger = Arc::clone(&ledger);
            workers.push(std::thread::spawn(move || {
                let operation = JsonOperationContext::new(Arc::clone(&ledger));
                for _ in 0..1000 {
                    let mut reservation = operation.reservation();
                    let _ = reservation.grow(256, JsonInputKind::ChunkMetadata);
                    assert!(ledger.state.lock().unwrap().used <= 1024);
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(ledger.state.lock().unwrap().used, 0);
    }
}
