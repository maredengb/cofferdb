use super::{Argument, Operation, Output};
use crate::storage::RecordReservation;
use crate::{Database, OnConflict, Result};
use std::path::Path;

impl Operation {
    pub fn import_jsonl_output(
        self,
        database: &Database,
        path: Argument<String>,
        on_conflict: OnConflict,
    ) -> Result<Output> {
        self.check_input(&path)?;
        self.progress.begin();
        database
            .import_manifest_measured_in(
                Path::new(&path.value),
                on_conflict,
                crate::db::ImportIndexBuildMode::DeferredFullCheckpoint,
                self.context.clone(),
            )?
            .into_output(database, &self.context)
    }

    pub fn export_jsonl(
        self,
        database: &Database,
        collection_name: &str,
        path: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&path)?;
        let (summary, retained) = database.export_collection_jsonl_retained_in(
            collection_name,
            Path::new(&path.value),
            &self.context,
        )?;
        let mut work = RecordReservation::new(self.context.reservation(), retained);
        let value = summary.into_json_retained(&mut work)?;
        work.retain_workspace(0);
        Ok(Output::new(value, work, Some(database.clone())))
    }
}
