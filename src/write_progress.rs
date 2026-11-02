//! Operation-local observations of real main-data commit boundaries.

use std::sync::{Arc, Mutex};

/// Confirmed progress, independent of any particular host exception type.
///
/// Records include main-data puts, tombstones and collection records, not a
/// difference of database statistics or a count of caller input rows. An
/// uncertain append may have committed beyond this confirmed prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WriteOutcome {
    pub state: &'static str,
    pub confirmed_committed_batches: Option<usize>,
    pub confirmed_committed_records: Option<usize>,
    pub last_confirmed_commit_id: Option<u64>,
    pub last_confirmed_commit_sequence_no: Option<u64>,
}

#[derive(Clone, Default)]
pub(crate) struct WriteProgress(Arc<Mutex<State>>);

#[derive(Default)]
struct State {
    started: bool,
    append_in_flight: bool,
    expected_records: Option<usize>,
    batches: usize,
    records: usize,
    last_commit: Option<(u64, u64)>,
}

impl WriteProgress {
    pub(crate) fn begin(&self) {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .started = true;
    }

    pub(crate) fn expect_records(&self, count: usize) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.started = true;
        // A manifest sets its complete plan before the first append; an
        // individual storage append must not replace that aggregate total.
        state.expected_records.get_or_insert(count);
    }

    pub(crate) fn begin_append(&self, count: usize) {
        self.expect_records(count);
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .append_in_flight = true;
    }

    pub(crate) fn confirmed(&self, records: usize, batch_id: u64, sequence_no: u64) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.batches += 1;
        state.records += records;
        state.last_commit = Some((batch_id, sequence_no));
        state.append_in_flight = false;
    }

    pub(crate) fn snapshot(&self) -> WriteOutcome {
        let state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let name = if state.append_in_flight {
            "unknown"
        } else if !state.started {
            "not_started"
        } else if state.batches == 0 {
            "not_committed"
        } else if state.expected_records == Some(state.records) {
            "committed"
        } else {
            "partially_committed"
        };
        WriteOutcome {
            state: name,
            confirmed_committed_batches: state.started.then_some(state.batches),
            confirmed_committed_records: state.started.then_some(state.records),
            last_confirmed_commit_id: state.last_commit.map(|(id, _)| id),
            last_confirmed_commit_sequence_no: state.last_commit.map(|(_, sequence)| sequence),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_preflight_partial_commit_and_completion_are_distinct() {
        let progress = WriteProgress::default();
        assert_eq!(progress.snapshot().state, "not_started");
        assert_eq!(progress.snapshot().confirmed_committed_batches, None);
        progress.begin();
        assert_eq!(progress.snapshot().state, "not_committed");
        progress.expect_records(5);
        progress.begin_append(2);
        assert_eq!(progress.snapshot().state, "unknown");
        progress.confirmed(2, 7, 20);
        let partial = progress.snapshot();
        assert_eq!(partial.state, "partially_committed");
        assert_eq!(partial.confirmed_committed_batches, Some(1));
        assert_eq!(partial.confirmed_committed_records, Some(2));
        assert_eq!(partial.last_confirmed_commit_id, Some(7));
        progress.begin_append(3);
        assert_eq!(progress.snapshot().state, "unknown");
        assert_eq!(progress.snapshot().last_confirmed_commit_id, Some(7));
        progress.confirmed(3, 8, 24);
        assert_eq!(progress.snapshot().state, "committed");
        assert_eq!(
            progress.snapshot().last_confirmed_commit_sequence_no,
            Some(24)
        );
    }

    #[test]
    fn a_single_confirmed_batch_remains_committed_after_later_output_failure() {
        let progress = WriteProgress::default();
        progress.begin_append(3);
        progress.confirmed(3, 4, 11);
        assert_eq!(progress.snapshot().state, "committed");
        // There is no error-driven reset or implicit retry of an append.
        assert_eq!(progress.clone().snapshot(), progress.snapshot());
    }
}
