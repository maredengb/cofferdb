use super::{Input, Stack};
use crate::json::WorkspaceReservation;
use crate::storage::RecordReservation;
use crate::{Database, JsonValue, Result};
use std::mem::ManuallyDrop;

/// An owned core projection retained until native destruction and host handoff.
pub struct Output {
    // Owned native JSON tree traversed by the host converter, replaced with null
    // once its contents have been copied.
    value: JsonValue,
    // JSON and workspace credit for the retained native projection, released when
    // that tree is cleared.
    work: RecordReservation,
    // Separate workspace estimate for the host copy while both representations are
    // live, held through the final handoff.
    host_work: WorkspaceReservation,
    // Optional database reference for cache reclamation during host admission. Its
    // destructor may run only in the creating process.
    database: ManuallyDrop<Option<Database>>,
    // Creating process ID used at destruction so a fork child cannot release the
    // parent's database reference and file locks.
    pid: u32,
}

impl Output {
    pub(crate) fn new(
        value: JsonValue,
        work: RecordReservation,
        database: Option<Database>,
    ) -> Self {
        // Native and host representations may coexist while the binding converts
        // the result. Give the host copy its own initially empty reservation on
        // the native result's ledger so releasing either representation leaves the
        // other accounted for.
        let host_work = work.json.context().workspace_reservation();
        Self {
            value,
            work,
            host_work,
            database: ManuallyDrop::new(database),
            pid: std::process::id(),
        }
    }

    /// Borrows the native JSON projection for sizing or conversion. After
    /// `clear_native` releases the projection in its creating process, this
    /// returns JSON null.
    pub fn value(&self) -> &JsonValue {
        &self.value
    }

    /// Creates an empty, separately accounted traversal stack on the output's
    /// operation. Its backing storage is released when the traversal stack is
    /// dropped.
    ///
    /// # Errors
    ///
    /// An output inherited across a fork cannot create a traversal stack.
    pub fn stack<T>(&self) -> Result<Stack<T>> {
        self.check_process()?;
        Input::new(self.work.json.context()).stack()
    }

    fn check_process(&self) -> Result<()> {
        // Check the ledger owner's process before creating workspace or touching
        // the associated database. A fork child can inherit these Rust values
        // while their locks and accounting still belong to the parent.
        if self.work.json.context().is_current_process() {
            Ok(())
        } else {
            Err(crate::Error::new(
                crate::ErrorCode::UnsupportedFeatureFlag,
                "an inherited binding output cannot be materialized after fork",
            ))
        }
    }

    /// Admit the binding's aggregate host construction estimate before allocating.
    /// Pass the total bytes estimated for the host representation. The reservation
    /// can grow across calls and is held while the host value is delivered.
    ///
    /// # Errors
    ///
    /// Returns an error if this is an inherited output or the host representation
    /// exceeds the available budget. The native value remains available after an
    /// admission failure.
    pub fn reserve_host(&mut self, bytes: u128) -> Result<()> {
        self.check_process()?;
        let work = &mut self.host_work;
        // Request the aggregate as a wide integer. Only growth beyond previously
        // reserved host bytes is charged, and an estimate larger than the machine
        // address space must reach admission without truncation.
        let admit = || {
            work.grow_to_wide(bytes, bytes, "result_materialize")
                .map(|_| ())
        };
        // A database-associated output may reclaim optional cache space and retry
        // the host reservation. Standalone diagnostic outputs have no database to
        // reclaim from, but still use the same admission rule.
        match &*self.database {
            Some(database) => database.admit_input_reclaiming(admit),
            None => {
                let mut admit = admit;
                admit()
            }
        }
    }

    /// Destroy the native tree before returning its credit. Host credit remains
    /// held until this adapter is dropped at the final host ownership handoff.
    pub fn clear_native(&mut self) {
        // A fork child skips the parent's ledger and database reference. In the
        // creating process clearing is repeatable because the substitutes are
        // null, zero credit and an empty database slot.
        if !self.work.json.context().is_current_process() {
            return;
        }
        drop(std::mem::replace(&mut self.value, JsonValue::Null));
        self.work.json.retain(0);
        self.work.retain_workspace(0);
        // Take the optional reference out of its slot. This releases the database
        // now and leaves an initialized None for the creator's final ManuallyDrop
        // cleanup.
        drop(self.database.take());
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if self.pid == std::process::id() {
            // SAFETY: normal destruction runs once. A fork child quarantines
            // its inherited database reference rather than unlocking files.
            unsafe { ManuallyDrop::drop(&mut self.database) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};

    #[test]
    fn native_projection_and_host_capacity_share_credit_until_last_use_and_handoff() {
        let context = JsonOperationContext::standalone();
        let mut input = Input::new(&context);
        let text = input
            .text_from_chars(std::iter::repeat('x').take(9 * 1024 * 1024))
            .unwrap();
        let argument = input.finish(text);
        let mut output = Output::new(JsonValue::String(argument.value), argument.work, None);
        let available = context.available_workspace_bytes();
        output.reserve_host(9 * 1024 * 1024).unwrap();
        let after = context.available_workspace_bytes();
        let error = output
            .reserve_host((9 * 1024 * 1024 + after + 1) as u128)
            .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        assert_eq!(context.available_workspace_bytes(), after);
        assert!(output.value().as_str().unwrap().len() > 8 * 1024 * 1024);
        output.clear_native();
        assert_eq!(context.available_workspace_bytes(), available);
        drop(output);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }

    #[test]
    fn wide_host_request_is_rejected_without_panic_or_budget_mutation() {
        let context = JsonOperationContext::standalone();
        let mut input = Input::new(&context);
        let text = input.text("retained native allocation").unwrap();
        let argument = input.finish(text);
        let mut output = Output::new(JsonValue::String(argument.value), argument.work, None);
        let available = context.available_workspace_bytes();
        let error = output.reserve_host(u128::MAX).unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        assert_eq!(context.available_workspace_bytes(), available);
        assert_eq!(output.value().as_str(), Some("retained native allocation"));
        output.reserve_host(1024).unwrap();
        drop(output);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }
}
