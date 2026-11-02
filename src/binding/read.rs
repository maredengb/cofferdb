use std::path::Path;

use super::{Argument, Operation, Output};
use crate::storage::RecordReservation;
use crate::{Database, DatabaseConfig, DatabaseOpenOptions, Error, ErrorCode, Result};

impl Operation {
    pub fn error(self, error: Error) -> Result<Output> {
        let mut work = RecordReservation::from(self.context.reservation());
        let value = error.into_json_retained(&mut work)?;
        Ok(Output::new(value, work, None))
    }

    pub(super) fn check_input<T>(&self, input: &Argument<T>) -> Result<()> {
        if !self.context.same_operation(input.work.json.context()) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "arguments must belong to the same operation",
            ));
        }
        Ok(())
    }

    pub fn create(self, input: Argument<(String, DatabaseConfig)>) -> Result<Database> {
        self.check_input(&input)?;
        let Argument { value, work: _work } = input;
        let (path, config) = value;
        Database::create_in(Path::new(&path), config, self.context.clone())
    }

    pub fn open(self, input: Argument<(String, DatabaseOpenOptions)>) -> Result<Database> {
        self.check_input(&input)?;
        let Argument { value, work: _work } = input;
        let (path, options) = value;
        Database::open_in(Path::new(&path), options, self.context.clone())
    }

    pub fn get_chunk(
        self,
        database: &Database,
        collection_id: &str,
        id: Argument<String>,
        external_id: bool,
    ) -> Result<Output> {
        self.check_input(&id)?;
        let (payload, mut work) = database.get_chunk_content_retained_in(
            collection_id,
            &id.value,
            external_id,
            &self.context,
        )?;
        let content = payload;
        let value = content.into_json_retained(&mut work)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    pub fn get_document(
        self,
        database: &Database,
        collection_id: &str,
        external_id: Argument<String>,
    ) -> Result<Output> {
        self.check_input(&external_id)?;
        let (payload, mut work) = database.get_document_by_external_id_retained_in(
            collection_id,
            &external_id.value,
            &self.context,
        )?;
        let document = payload;
        let value = document.into_json_retained(&mut work)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    pub fn database_stats(self, database: &Database) -> Result<Output> {
        let mut work = RecordReservation::from(self.context.reservation());
        let builder = crate::json::JsonBuilder::budgeted(
            &mut work.json,
            crate::json::JsonInputKind::StoredJson,
        );
        let value = database.stats()?.build_json(&builder)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }

    pub fn index_stats(self, database: &Database) -> Result<Output> {
        let mut work = RecordReservation::from(self.context.reservation());
        let builder = crate::json::JsonBuilder::budgeted(
            &mut work.json,
            crate::json::JsonInputKind::StoredJson,
        );
        let value = database.index_stats()?.build_json(&builder)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }
}
