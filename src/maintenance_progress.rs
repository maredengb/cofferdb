//! File effects already observed on one handle-bound maintenance call.
use std::sync::{Arc, Mutex, Weak};

/// Shared record of the file changes observed during a maintenance operation.
#[derive(Clone, Default)]
pub struct MaintenanceObservation(Arc<Mutex<State>>);

// The call's observer owns the state. Store only a weak reference in operation
// contexts so a cached or long-lived context cannot keep diagnostic state alive
// after the host drops its observation.
#[derive(Clone, Default)]
pub(crate) struct Tracking(Weak<Mutex<State>>);

#[derive(Default)]
pub(crate) struct State {
    // Latest maintenance stage observed by the handle, used to locate a later
    // failure even if the native operation has already affected the file.
    stage: Option<&'static str>,
    // Known file effect, such as a published snapshot. A later failure does not by
    // itself imply that the old file state was restored.
    effect: Option<&'static str>,
    // Durability information for the observed effect, distinct from whether a file
    // name or update is already visible.
    durability: Option<&'static str>,
    // Facts for a verified snapshot, staged before publication and reported only
    // after the publication boundary is known to have been crossed.
    snapshot: Option<SnapshotFacts>,
    // Observed index-suffix status after an index maintenance operation, when
    // available.
    index_status: Option<&'static str>,
    // Confirmed index commit identifier paired with the index status. Absence
    // indicates that the operation supplied no ID.
    index_commit: Option<u64>,
    // Observed state of the index rebuild associated with compaction.
    compact_index: Option<&'static str>,
    // Optional explanation for the observed compaction index state, returned
    // alongside that state when present.
    compact_reason: Option<String>,
}

// Prepare these facts together while the snapshot identity is known. Their
// presence in memory alone is insufficient evidence that the destination was
// installed.
pub(crate) struct SnapshotFacts {
    // Destination path recorded for the snapshot so a caller can find a published
    // copy even after finalization fails.
    pub(crate) target: String,
    // Digest of the validated snapshot bytes. The path becomes externally useful
    // once publication is observed.
    pub(crate) sha256: String,
    // Verified snapshot file size in bytes.
    pub(crate) size: u64,
    // Main-data commit ID captured in the snapshot being published.
    pub(crate) commit: u64,
}

impl MaintenanceObservation {
    pub(crate) fn tracking(&self) -> Tracking {
        Tracking(Arc::downgrade(&self.0))
    }

    /// Materialize diagnostic strings only after the native operation finishes.
    pub fn fields(&self) -> Vec<(&'static str, String)> {
        // Read all fields under one lock so a stage, effect and their optional
        // details reflect the same observation. A panic in the updating thread can
        // poison the lock without erasing transitions that were already recorded.
        let state = self.0.lock().unwrap_or_else(|err| err.into_inner());
        // An observation with no transitions still has an explicit initial stage
        // and effect. Optional details are emitted only when the corresponding
        // operation branch recorded them.
        let mut fields = vec![
            ("stage", state.stage.unwrap_or("prepare_maintenance").into()),
            ("effect", state.effect.unwrap_or("unchanged").into()),
        ];
        // A missing durability or rebuild value means the state did not provide
        // that detail. Omitting it avoids turning lack of evidence into a claim
        // about persistence or index status.
        for (key, value) in [
            ("durability", state.durability),
            ("index_suffix_status", state.index_status),
            ("index_rebuild", state.compact_index),
            ("index_rebuild_reason", state.compact_reason.as_deref()),
        ] {
            if let Some(value) = value {
                fields.push((key, value.into()));
            }
        }
        // Keep a confirmed index commit separate from a snapshot's last valid
        // commit. Both identify commit boundaries but belong to different
        // operations and are reported under different keys.
        if let Some(commit) = state.index_commit {
            fields.push(("last_index_commit_id", commit.to_string()));
        }
        // A prepared copy already has a target and digest. Report them only once
        // publication made that copy visible to the caller.
        if let Some(snapshot) = state
            .snapshot
            .as_ref()
            .filter(|_| state.effect == Some("snapshot_published"))
        {
            // Clone identifying strings only when assembling the host-facing
            // diagnostics. A failed finalization may still expose the published
            // target and its content identity, even while durability remains
            // unknown.
            fields.extend([
                ("target", snapshot.target.clone()),
                ("sha256", snapshot.sha256.clone()),
                ("size_bytes", snapshot.size.to_string()),
                ("last_valid_commit", snapshot.commit.to_string()),
            ]);
        }
        fields
    }
}

impl Tracking {
    // Retained catalog and cache contexts do not extend the call observation's lifetime.
    // Upgrade the weak observation only for this update. If the host dropped it,
    // retained execution state can proceed without recording anything.
    pub(crate) fn update(&self, visit: impl FnOnce(&mut State)) {
        if let Some(owner) = self.0.upgrade() {
            // Serialize each observation update under one mutex. Recover the
            // existing fields after poisoning so completed file effects remain
            // reportable during unwinding.
            visit(&mut owner.lock().unwrap_or_else(|err| err.into_inner()));
        }
    }
}

impl State {
    // Keep the prepared receipt facts available before publication changes the
    // target. The later effect flag distinguishes a prepared receipt from one
    // whose target actually became visible.
    pub(crate) fn prepare_snapshot(&mut self, facts: SnapshotFacts) {
        self.snapshot = Some(facts);
    }
    // Record that snapshot publication reached the target name. Durability remains
    // unknown until the separate confirmation updates it.
    pub(crate) fn published(&mut self) {
        self.stage = Some("snapshot_finalize");
        self.effect = Some("snapshot_published");
        self.durability = Some("unknown");
    }
    // Mark the visible snapshot as durable after confirmation. Prepared size,
    // digest and commit facts remain available to the host if later output fails.
    pub(crate) fn confirmed_snapshot(&mut self) {
        self.stage = Some("complete");
        self.durability = Some("confirmed");
    }
    // Mark the checkpoint write in progress. During compaction, retain the more
    // consequential main_replaced effect because it already describes a published
    // primary-file change.
    pub(crate) fn writing_index(&mut self) {
        self.stage = Some("write_checkpoint");
        if self.effect != Some("main_replaced") {
            self.effect = Some("index_may_need_rebuild");
        }
    }
    // Describe uncertainty only while index synchronization is pending. Clearing
    // that flag does not invent any additional file effect.
    pub(crate) fn syncing_index(&mut self, pending: bool) {
        self.durability = pending.then_some("unknown");
    }
    // Record the verification stage before inspecting the checkpoint. The existing
    // effect remains whichever file change was previously observed.
    pub(crate) fn verifying_index(&mut self) {
        self.stage = Some("verify_checkpoint");
    }
    // Record the successful rebuild outcome and its checkpoint identity. The
    // caller supplies both the verified disposition and whether a write was
    // needed.
    pub(crate) fn finished_rebuild(
        &mut self,
        status: &'static str,
        commit: Option<u64>,
        written: bool,
    ) {
        self.stage = Some("complete");
        // A rebuild can finish after transient validation without writing a
        // checkpoint. Report unchanged while retaining the supplied index status.
        self.effect = Some(if written {
            "index_written"
        } else {
            "unchanged"
        });
        self.index_status = Some(status);
        self.index_commit = commit;
    }
    // Switch to the replacement-confirmation stage once compaction installs the
    // new main file. The new name is visible before its durability can be assumed.
    pub(crate) fn replaced(&mut self) {
        self.stage = Some("confirm_replacement");
        self.effect = Some("main_replaced");
        self.durability = Some("unknown");
    }
    // Continue to index preparation after confirming the replacement. Clear the
    // earlier sync uncertainty while keeping main_replaced available for later
    // errors.
    pub(crate) fn confirmed_replacement(&mut self) {
        self.stage = Some("prepare_index");
        self.durability = None;
    }
    // Save what happened to the index while compacting, including any reason
    // supplied by the engine. The file replacement effect remains independent of
    // index outcome.
    pub(crate) fn processed_index(&mut self, index: &'static str, reason: Option<&str>) {
        self.stage = Some("measure_replacement");
        self.compact_index = Some(index);
        self.compact_reason = reason.map(str::to_owned);
    }
    // Only move the observed stage to completion. Preserve the previous effect and
    // receipt fields for host projection.
    pub(crate) fn complete(&mut self) {
        self.stage = Some("complete");
    }
}
