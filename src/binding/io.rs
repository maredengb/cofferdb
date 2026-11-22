use super::{Argument, Operation, Output};
use crate::storage::RecordReservation;
use crate::{Database, OnConflict, Result};
use std::path::Path;

impl Operation {
    /// Imports the manifest named by a prepared path and returns the retained host
    /// result. The conflict policy applies to the entire import request.
    ///
    /// # Errors
    ///
    /// Returns errors from argument ownership, manifest validation, admission,
    /// storage, or result projection. A failed call can still have confirmed
    /// earlier commits; retain a write observation before execution to inspect
    /// them.
    pub fn import_jsonl_output(
        self,
        database: &Database,
        path: Argument<String>,
        on_conflict: OnConflict,
    ) -> Result<Output> {
        self.check_input(&path)?;
        // Start tracking after the argument ownership check. Opening or validating
        // the manifest can now fail with zero confirmed batches; an argument from
        // another operation is rejected before tracking begins.
        self.progress.begin();
        // Use the prepared path as a native Path without a second owned copy.
        // Request deferred index checkpointing as part of the same import call.
        database
            .import_manifest_measured_in(
                Path::new(&path.value),
                on_conflict,
                crate::db::ImportIndexBuildMode::DeferredFullCheckpoint,
                self.context.clone(),
            )?
            // Convert the measured result only after the native call returns. This
            // conversion can fail while the earlier writes remain committed.
            .into_output(database, &self.context)
    }

    /// Exports an existing collection by name into the directory named by a
    /// prepared argument. Returns the engine's export summary as retained output.
    ///
    /// # Errors
    ///
    /// Returns an error if the prepared path is foreign, the collection cannot be
    /// read, writing the artifacts fails, or the summary cannot be converted after
    /// export.
    pub fn export_jsonl(
        self,
        database: &Database,
        collection_name: &str,
        path: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&path)?;
        // Keep the argument reservation live during the engine call. The retained
        // summary has a separate lease for the strings moved into the host result.
        let (summary, retained) = database.export_collection_jsonl_retained_in(
            collection_name,
            Path::new(&path.value),
            &self.context,
        )?;
        // The export has already produced its native summary. Transfer its retained
        // strings when building the host output, then release workspace left by the
        // consumed native wrapper.
        let mut work = RecordReservation::new(self.context.reservation(), retained);
        let value = summary.into_json_retained(&mut work)?;
        // Release only the remaining workspace once the JSON payload owns its
        // allocations. Output retains the resulting JSON guard and database
        // handle.
        work.retain_workspace(0);
        Ok(Output::new(value, work, Some(database.clone())))
    }
}
