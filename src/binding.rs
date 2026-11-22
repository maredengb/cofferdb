//! Narrow host-binding adapters. These are not a stable application API.

use std::path::Path;

use crate::json::JsonOperationContext;
use crate::write_progress::WriteProgress;
use crate::{Database, ImportSummary, OnConflict, Result};

// Expose the core receipt type directly to hosts. Its state describes confirmed
// write progress and remains meaningful when a later step fails before returning
// the normal result.
pub use crate::write_progress::WriteOutcome;
mod batch;
mod collection;
mod input;
mod io;
mod maintenance;
pub use crate::maintenance_progress::MaintenanceObservation;
pub use maintenance::SNAPSHOT_FORMAT_VERSION;
mod output;
mod read;
mod search;
pub use batch::{Batch, ChunkRecord, WriteSummary};
pub use collection::CollectionIdentity;
pub use input::{Argument, Input, Stack, ValueKind};
pub use output::Output;
pub use search::SearchInput;

/// Scoped, instance-local import failure injection for native integration tests.
///
/// This Rust-only surface is absent unless `native-test-support` is enabled.
/// Arm it through the host's ordinary owner gate, release that gate, then call
/// the real binding. The guard must stay in the creating process and outlive
/// the call. Dropping it cancels any unconsumed fault. No Python API is added.
#[cfg(feature = "native-test-support")]
#[doc(hidden)]
pub struct ImportBatchFailure {
    database: std::mem::ManuallyDrop<Database>,
    pid: u32,
}

#[cfg(feature = "native-test-support")]
impl ImportBatchFailure {
    pub fn arm(database: &Database, after_batches: usize) -> Result<Self> {
        database.arm_import_batch_failure(Some(after_batches))?;
        Ok(Self {
            database: std::mem::ManuallyDrop::new(database.clone()),
            pid: std::process::id(),
        })
    }
}

#[cfg(feature = "native-test-support")]
impl Drop for ImportBatchFailure {
    fn drop(&mut self) {
        if self.pid == std::process::id() {
            self.database.clear_import_batch_failure();
            // SAFETY: Drop runs once in the creating process. A fork child
            // quarantines the inherited handle, including its file locks.
            unsafe { std::mem::ManuallyDrop::drop(&mut self.database) };
        }
    }
}

/// Return the version of the native CofferDB crate compiled into this binding.
/// Hosts can report it separately from the version of their language package.
pub fn engine_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// One shared resource context and commit observation for one host call.
pub struct Operation {
    // Per-call admission context shared by argument conversion, native execution
    // and result projection, and attached to this call's progress observation.
    pub(crate) context: JsonOperationContext,
    // Shared record of confirmed write progress. An observer can read it after
    // execution consumes the operation or a native call unwinds.
    pub(crate) progress: WriteProgress,
}

/// Read-only progress retained even when the executing operation unwinds.
#[derive(Clone)]
pub struct WriteObservation {
    // Read-only handle to the executing write's confirmation state. A snapshot does
    // not depend on the result having reached the host.
    progress: WriteProgress,
}

impl WriteObservation {
    /// Snapshot the write progress currently confirmed for this call. Retain the
    /// observation before executing a consuming operation to inspect progress
    /// after a failure.
    pub fn write_outcome(&self) -> WriteOutcome {
        self.progress.snapshot()
    }
}

impl Operation {
    /// Start an operation with its own resource context and an empty write
    /// observation. Use this form for host work that has no open database to
    /// supply a context.
    pub fn standalone() -> Self {
        let progress = WriteProgress::default();
        Self {
            // Attach the same tracker that the observation handle will clone.
            // Argument conversion and native write execution must agree about
            // which call owns the progress record.
            context: JsonOperationContext::standalone().with_write_progress(&progress),
            progress,
        }
    }

    /// Create an input builder that charges converted host values to this
    /// operation. The builder retains the context while its owned arguments are
    /// assembled.
    pub fn input(&self) -> Input {
        Input::new(&self.context)
    }

    /// Create an empty batch for documents, chunks and embeddings prepared under
    /// this operation. The writer checks that the batch belongs to the consuming
    /// call.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation cannot create the batch's reserved row
    /// buffers.
    pub fn batch(&self) -> Result<Batch> {
        Batch::new(&self.context)
    }

    /// Runs one prepared host batch against an existing collection and consumes
    /// this operation.
    ///
    /// The batch must have been prepared for the same operation. `on_conflict`
    /// controls handling of duplicate external IDs. The returned summary retains
    /// any warnings and the reservation that accounts for them.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign batch, invalid input, a rejected write, or a
    /// storage failure. An observation obtained before the call can still describe
    /// confirmed progress when the operation returns an error.
    pub fn write_batch(
        self,
        database: &Database,
        collection_id: &str,
        batch: Batch,
        on_conflict: OnConflict,
    ) -> Result<WriteSummary> {
        // Reject arguments prepared for a different operation before recording a
        // write attempt. Later failures may occur after the progress tracker began,
        // even before the first batch commits.
        batch.check_operation(&self.context)?;
        self.progress.begin();
        // Allocate warning storage under the same call context before visiting
        // batch rows. Warning data can outlive the consumed operation when
        // execution succeeds.
        let mut warnings = crate::import::warnings::ImportWarnings::new(&self.context)?;
        let counts = database.write_preprocessed_batch(
            collection_id,
            on_conflict,
            self.context.clone(),
            &mut warnings,
            |consume| batch.visit(consume),
        )?;
        // Move the warning buffer and its retained workspace together. The summary
        // remains safe to project into host output after the batch is gone.
        let (warnings, work) = warnings.into_parts();
        Ok(WriteSummary {
            counts,
            warnings,
            _work: work,
        })
    }

    /// Begins a host call using the database's resource ledger and a fresh write
    /// observation.
    pub fn new(database: &Database) -> Self {
        let progress = WriteProgress::default();
        Self {
            context: database.json_context().with_write_progress(&progress),
            progress,
        }
    }

    /// Clones a read-only observation of the write. Keep it outside the executing
    /// call to inspect confirmed progress after it returns or unwinds.
    pub fn observation(&self) -> WriteObservation {
        WriteObservation {
            progress: self.progress.clone(),
        }
    }

    /// Consume the operation so its input credit and receipt cannot be reused.
    ///
    /// ```compile_fail
    /// use cofferdb::{binding::Operation, Database, OnConflict};
    /// use std::path::Path;
    /// fn twice(operation: Operation, database: &Database, path: &Path) {
    ///     let _ = operation.import_jsonl(database, path, OnConflict::Error);
    ///     let _ = operation.import_jsonl(database, path, OnConflict::Error);
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns a manifest, validation, budget, or storage error if import stops.
    /// Earlier batches can already have committed before a later batch fails.
    /// Obtain an observation before consuming the operation if the host needs to
    /// report that progress.
    pub fn import_jsonl(
        self,
        database: &Database,
        manifest_path: &Path,
        on_conflict: OnConflict,
    ) -> Result<ImportSummary> {
        // Announce that the operation began before reading the manifest.
        // Confirmation is recorded only by the storage append path when commit
        // identity is known.
        self.progress.begin();
        database
            .import_manifest_measured_in(
                manifest_path,
                on_conflict,
                // Request deferred checkpointing for the manifest. Pending index
                // work is attempted after the primary batches, whereas a direct
                // host batch requests inline checkpoints.
                crate::db::ImportIndexBuildMode::DeferredFullCheckpoint,
                self.context.clone(),
            )
            // Return the import summary without retaining runner-specific timing
            // measurements.
            .map(|measured| measured.summary)
    }
}
