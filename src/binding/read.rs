use std::path::Path;

use super::{Argument, Operation, Output};
use crate::storage::RecordReservation;
use crate::{Database, DatabaseConfig, DatabaseOpenOptions, Error, ErrorCode, Result};

impl Operation {
    /// Convert a native error into retained JSON for the host. This path can be
    /// used without a database handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the diagnostic value cannot be constructed under this
    /// operation's budget.
    pub fn error(self, error: Error) -> Result<Output> {
        let mut work = RecordReservation::from(self.context.reservation());
        // Consume the error's owned fields under the new reservation. The
        // diagnostic output has no database handle, so later host allocation
        // cannot reclaim database caches through it.
        let value = error.into_json_retained(&mut work)?;
        Ok(Output::new(value, work, None))
    }

    pub(super) fn check_input<T>(&self, input: &Argument<T>) -> Result<()> {
        // Calls on the same database can have distinct reservations. Accepting
        // another call's input would charge its conversion to the wrong budget.
        if !self.context.same_operation(input.work.json.context()) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "arguments must belong to the same operation",
            ));
        }
        Ok(())
    }

    /// Create a database using the path and configuration prepared for this
    /// binding operation.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument or a failure creating the database
    /// with the supplied configuration and path.
    pub fn create(self, input: Argument<(String, DatabaseConfig)>) -> Result<Database> {
        self.check_input(&input)?;
        // Retain both argument reservations while create_in borrows the path and
        // consumes the database configuration.
        let Argument { value, work: _work } = input;
        let (path, config) = value;
        Database::create_in(Path::new(&path), config, self.context.clone())
    }

    /// Open a database using the prepared path and native open options.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument or if opening the database under
    /// these options fails.
    pub fn open(self, input: Argument<(String, DatabaseOpenOptions)>) -> Result<Database> {
        self.check_input(&input)?;
        // The options and path are read under the operation that paid for their
        // conversion. Even when opening fails, their reservation remains live for
        // the duration of the native call.
        let Argument { value, work: _work } = input;
        let (path, options) = value;
        Database::open_in(Path::new(&path), options, self.context.clone())
    }

    /// Read the chunk and its parent document from the named internal collection
    /// ID. `external_id` chooses whether `id` identifies the chunk by its external
    /// key or by its internal ID.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument, an unresolved chunk or parent
    /// document, or a failure admitting, reading or projecting the content.
    pub fn get_chunk(
        self,
        database: &Database,
        collection_id: &str,
        id: Argument<String>,
        external_id: bool,
    ) -> Result<Output> {
        self.check_input(&id)?;
        // Both identifier forms are scoped by the native collection ID. Keep the
        // returned records and their reservation together through JSON projection
        // so an owned chunk can outlive its read call.
        let (payload, mut work) = database.get_chunk_content_retained_in(
            collection_id,
            &id.value,
            external_id,
            &self.context,
        )?;
        let content = payload;
        // Consume the retained chunk and document into one output value. The
        // returned work guard owns that projection, and the database clone can
        // support cache reclamation while the host allocates its copy.
        let value = content.into_json_retained(&mut work)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    /// Reads the visible document for an external ID within one internal
    /// collection ID. The returned output retains the native JSON while the host
    /// copies it.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign argument, a missing document in that
    /// collection, or a failure to read or project the result.
    pub fn get_document(
        self,
        database: &Database,
        collection_id: &str,
        external_id: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&external_id)?;
        // Resolve this key in the requested collection. The retained copy lets the
        // catalog lookup release its lock while the document's owned fields are
        // converted to JSON.
        let (payload, mut work) = database.get_document_by_external_id_retained_in(
            collection_id,
            &external_id.value,
            &self.context,
        )?;
        let document = payload;
        // Move the metadata subtree with its existing JSON charge. The builder can
        // transfer owned-string capacity from workspace into the destination JSON
        // reservation.
        let value = document.into_json_retained(&mut work)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    /// Checks the database at a prepared path and requests verification of its
    /// indexes. Inspect the returned report's status and issues even when this
    /// call succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error for an argument from a different operation or if a stable
    /// check report cannot be obtained or projected.
    pub fn check_path(self, input: Argument<String>) -> Result<Output> {
        self.check_input(&input)?;
        let report = crate::db::build_check_report(
            Path::new(&input.value),
            crate::CheckOptions {
                verify_indexes: true,
            },
            &self.context,
        )?;
        // This path-based check has no open database handle to retain. Its
        // independent result reservation accounts for the JSON report while the
        // host reads its findings.
        let mut work = RecordReservation::from(self.context.reservation());
        let builder = crate::json::JsonBuilder::budgeted(
            &mut work.json,
            crate::json::JsonInputKind::StoredJson,
        );
        let value = report.build_json(&builder)?;
        Ok(Output::new(value, work, None))
    }

    /// Returns a retained view of database-wide live counts, file sizes and
    /// effective resource settings, as sampled by `Database::stats`.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot supply the statistics or their JSON
    /// allocation cannot be admitted.
    pub fn database_stats(self, database: &Database) -> Result<Output> {
        let mut work = RecordReservation::from(self.context.reservation());
        let builder = crate::json::JsonBuilder::budgeted(
            &mut work.json,
            crate::json::JsonInputKind::StoredJson,
        );
        // Materialize a single statistics sample. Holding the resulting JSON does
        // not hold the database lock or refresh these values after another commit.
        let value = database.stats()?.build_json(&builder)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    /// Returns live vector counts and persisted index coverage for the database. A
    /// successful result can indicate that the index needs rebuilding.
    ///
    /// # Errors
    ///
    /// Returns an error if the statistics cannot be read or the result cannot be
    /// admitted.
    pub fn index_stats(self, database: &Database) -> Result<Output> {
        let mut work = RecordReservation::from(self.context.reservation());
        let builder = crate::json::JsonBuilder::budgeted(
            &mut work.json,
            crate::json::JsonInputKind::StoredJson,
        );
        // Preserve the core report's nullable checkpoint commit and rebuild flag
        // in the host projection. A database with live vectors can lack a usable
        // persisted checkpoint.
        let value = database.index_stats()?.build_json(&builder)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }
}

#[cfg(test)]
mod tests {
    use crate::{CheckIssue, IndexSuffixReport, JsonValue};

    #[test]
    fn check_diagnostics_keep_full_width_ids_and_nullable_offsets() {
        let issue = CheckIssue {
            severity: "warning".into(),
            code: "corrupt_tail".into(),
            message: String::new(),
            stream: "critical".into(),
            visibility_impact: "tail_only".into(),
            offset: None,
            extent_id: None,
            record_type: None,
            batch_id: Some(u64::MAX),
            repair_available: false,
            suggested_command: None,
        };
        let suffix = IndexSuffixReport {
            status: "persistent_checkpoint".into(),
            offset: (1 << 53) - 1,
            checkpoint_generation: (1 << 53) + 1,
            covered_critical_end_offset: 1 << 53,
            covered_last_batch_id: u64::MAX,
            covered_last_commit_sequence_no: u64::MAX - 1,
            coverage_hash: 0,
            collection_count: 0,
            live_document_count: 0,
            live_chunk_count: 0,
            live_embedding_count: 0,
            text_segment_count: 0,
            vector_segment_count: 0,
            index_record_count: 0,
            index_extent_count: 0,
            payload_len: 0,
            build_mode: "full_visible_snapshot".into(),
        };
        let issue = issue.to_json();
        assert_eq!(issue.get("offset"), Some(&JsonValue::Null));
        assert_eq!(
            issue.get("batch_id").unwrap().as_str(),
            Some("18446744073709551615")
        );
        let suffix = suffix.to_json();
        assert_eq!(
            suffix.get("offset").unwrap().as_f64(),
            Some(9007199254740991.0)
        );
        assert_eq!(
            suffix.get("checkpoint_generation").unwrap().as_str(),
            Some("9007199254740993")
        );
        assert_eq!(
            suffix.get("covered_critical_end_offset").unwrap().as_str(),
            Some("9007199254740992")
        );
    }
}
