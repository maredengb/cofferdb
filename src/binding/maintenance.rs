use super::{Argument, Operation, Output};
use crate::storage::RecordReservation;
use crate::{
    CompactOptions, CompactSummary, Database, RebuildIndexSummary, Result, SnapshotSummary,
};

/// Storage format version of a snapshot produced by this engine. Bindings can
/// include it in a receipt without maintaining an independent format-version
/// constant.
pub const SNAPSHOT_FORMAT_VERSION: u16 = crate::storage::FORMAT_VERSION;

impl Operation {
    /// Track effects even if the native operation unwinds before creating its receipt.
    pub fn observe_maintenance(&mut self) -> super::MaintenanceObservation {
        // Return the strong observation to the caller and install its weak tracker
        // on this operation. The caller must keep the observation alive through
        // the native call to collect progress after an error or unwind.
        let observation = super::MaintenanceObservation::default();
        self.context.maintenance = observation.tracking();
        observation
    }

    /// Return scalar receipt fields and a budgeted projection of the collection list.
    /// Collection payloads are transferred to the second return value.
    /// # Errors
    ///
    /// Returns an error for a target prepared by another operation or a failure in
    /// snapshot creation, publication or receipt construction. A receipt error
    /// after publication carries the target, digest, byte count and last valid
    /// commit.
    pub fn snapshot(
        &self,
        database: &Database,
        target: Argument<String>,
    ) -> Result<(SnapshotSummary, Output)> {
        // Reject a foreign path before creating the target. Its string is held by
        // the prepared argument while the snapshot call borrows the path.
        self.check_input(&target)?;
        // The core returns the scalar receipt and collection list together, with
        // catalog workspace and the recovery path for the published target. Keep
        // all three until host projection has succeeded.
        let (mut summary, catalog_work, recovery_path) =
            database.snapshot_in(target.value.as_ref(), &self.context)?;
        // The target is already durable. If the receipt outgrows the remaining
        // budget, return enough information to identify the published snapshot.
        // Mark only failures from projecting the collection list as a durable
        // published snapshot here. Errors returned earlier by snapshot_in carry
        // the effects known at their own failure boundary.
        let catalog = self.snapshot_collections(&summary).map_err(|err| {
            err.with_detail("effect", "snapshot_published")
                .with_detail("stage", "materialize_snapshot")
                .with_detail("durability", "confirmed")
                .with_detail("target", recovery_path.display())
                .with_detail("sha256", &summary.sha256)
                .with_detail("size_bytes", summary.bytes_copied)
                .with_detail("last_valid_commit", summary.last_valid_commit)
        })?;
        // After the collection list has been projected, keep only scalar receipt
        // data in the native summary. Destroy the duplicate native list before
        // returning its catalog reservation.
        drop(std::mem::take(&mut summary.collections));
        drop(catalog_work);
        Ok((summary, catalog))
    }

    // Project collections in the snapshot receipt's collection-name order.
    fn snapshot_collections(&self, summary: &SnapshotSummary) -> Result<Output> {
        let mut work = RecordReservation::from(self.context.reservation());
        let builder = crate::json::JsonBuilder::budgeted(
            &mut work.json,
            crate::json::JsonInputKind::StoredJson,
        );
        // Build each entry under the same result reservation. If a later entry
        // exceeds the budget, the partial array is dropped and the caller can
        // still read the native receipt's recovery fields.
        let value = builder.array(
            summary
                .collections
                .iter()
                .map(|item| item.build_json(&builder)),
        )?;
        // The returned JSON array owns its native payload. With no database
        // reference, later host allocation uses direct admission without
        // database-cache reclamation.
        Ok(Output::new(value, work, None))
    }

    /// Request an index rebuild for the named collection, or a database-wide
    /// rebuild when the argument is `None`. Returns the native report and a lease
    /// for constructing the host receipt.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument, an unknown collection, or a native
    /// rebuild failure. A retained maintenance observation can identify stages and
    /// effects reached before an error.
    pub fn rebuild_index(
        &self,
        database: &Database,
        collection: Argument<Option<String>>,
    ) -> Result<(RebuildIndexSummary, Output)> {
        // Validate the optional name's operation identity even for None. That
        // argument still carries the reservation used as the output lease on
        // success.
        self.check_input(&collection)?;
        let report =
            database.rebuild_index_in(collection.value.as_deref(), self.context.clone())?;
        // Return the host projection lease alongside the typed report. The host
        // uses that lease to admit the representation it builds after the index
        // operation has finished.
        let lease = Output::new(crate::JsonValue::Null, collection.work, None);
        Ok((report, lease))
    }

    /// Compact the database with the requested index rebuild policy. Returns the
    /// native compaction report and a lease for constructing its host
    /// representation.
    ///
    /// # Errors
    ///
    /// Returns a storage, resource or validation error if the native operation
    /// fails. A retained maintenance observation can distinguish effects that were
    /// already observed.
    pub fn compact(
        &self,
        database: &Database,
        rebuild_indexes: bool,
    ) -> Result<(CompactSummary, Output)> {
        // Let the core carry the maintenance tracker through compaction and any
        // requested index rebuild. The host projection lease is constructed only
        // after the core returns a report.
        let report =
            database.compact_in(CompactOptions { rebuild_indexes }, self.context.clone())?;
        // Compaction has no prepared input string to recycle as a lease. Start one
        // from this call's context with a null JSON placeholder; the typed report
        // carries the actual values.
        let lease = Output::new(
            crate::JsonValue::Null,
            self.context.reservation().into(),
            None,
        );
        Ok((report, lease))
    }
}
