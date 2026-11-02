//! Narrow host-binding adapters. These are not a stable application API.

use std::path::Path;

use crate::json::JsonOperationContext;
use crate::write_progress::WriteProgress;
use crate::{Database, ImportSummary, OnConflict, Result};

pub use crate::write_progress::WriteOutcome;
mod batch;
mod collection;
mod input;
mod io;
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

pub fn engine_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// One shared resource context and commit observation for one host call.
pub struct Operation {
    pub(crate) context: JsonOperationContext,
    pub(crate) progress: WriteProgress,
}

/// Read-only progress retained even when the executing operation unwinds.
#[derive(Clone)]
pub struct WriteObservation {
    progress: WriteProgress,
}

impl WriteObservation {
    pub fn write_outcome(&self) -> WriteOutcome {
        self.progress.snapshot()
    }
}

impl Operation {
    pub fn standalone() -> Self {
        let progress = WriteProgress::default();
        Self {
            context: JsonOperationContext::standalone().with_write_progress(&progress),
            progress,
        }
    }

    pub fn input(&self) -> Input {
        Input::new(&self.context)
    }

    pub fn batch(&self) -> Result<Batch> {
        Batch::new(&self.context)
    }

    pub fn write_batch(
        self,
        database: &Database,
        collection_id: &str,
        batch: Batch,
        on_conflict: OnConflict,
    ) -> Result<WriteSummary> {
        batch.check_operation(&self.context)?;
        self.progress.begin();
        let mut warnings = crate::import::warnings::ImportWarnings::new(&self.context)?;
        let counts = database.write_preprocessed_batch(
            collection_id,
            on_conflict,
            self.context.clone(),
            &mut warnings,
            |consume| batch.visit(consume),
        )?;
        let (warnings, work) = warnings.into_parts();
        Ok(WriteSummary {
            counts,
            warnings,
            _work: work,
        })
    }

    pub fn new(database: &Database) -> Self {
        let progress = WriteProgress::default();
        Self {
            context: database.json_context().with_write_progress(&progress),
            progress,
        }
    }

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
    pub fn import_jsonl(
        self,
        database: &Database,
        manifest_path: &Path,
        on_conflict: OnConflict,
    ) -> Result<ImportSummary> {
        self.progress.begin();
        database
            .import_manifest_measured_in(
                manifest_path,
                on_conflict,
                crate::db::ImportIndexBuildMode::DeferredFullCheckpoint,
                self.context.clone(),
            )
            .map(|measured| measured.summary)
    }
}
