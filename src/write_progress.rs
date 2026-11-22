//! Operation-local observations of real main-data commit boundaries.

use std::sync::{Arc, Mutex};

/// Confirmed progress, independent of any particular host exception type.
///
/// Records include main-data puts, tombstones and collection records, not a
/// difference of database statistics or a count of caller input rows. An
/// uncertain append may have committed beyond this confirmed prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WriteOutcome {
    /// Write state: not started, not committed, partially committed, committed or unknown.
    pub state: &'static str,
    /// Batches confirmed during this operation, when progress tracking began.
    pub confirmed_committed_batches: Option<usize>,
    /// Main-data records in those confirmed batches, when progress tracking began.
    pub confirmed_committed_records: Option<usize>,
    /// Last batch identifier confirmed by this operation, if any.
    pub last_confirmed_commit_id: Option<u64>,
    /// Commit record sequence number for the last confirmed batch, if any.
    pub last_confirmed_commit_sequence_no: Option<u64>,
}

// Clones observe one logical request. The executing operation and any host receipt
// reader share this one state even if the call returns an error or unwinds.
#[derive(Clone, Default)]
pub(crate) struct WriteProgress(Arc<Mutex<State>>);

#[derive(Default)]
struct State {
    // Whether this operation started observing writes. Before tracking begins the
    // public outcome has no confirmed counters.
    started: bool,
    // An append began and its commit has not been confirmed here. Errors in this
    // interval leave the outcome of that append unknown.
    append_in_flight: bool,
    // Planned number of main-data records for the logical request. A manifest can
    // set it across batches before its first append.
    expected_records: Option<usize>,
    // Batches this operation has positively confirmed, excluding an in-flight batch
    // of uncertain outcome.
    batches: usize,
    // Main-data records in those confirmed batches, not the number of caller rows
    // or net change in live object count.
    records: usize,
    // Identifier and commit-record sequence number of the most recently confirmed
    // batch, absent until the first confirmation.
    last_commit: Option<(u64, u64)>,
}

impl WriteProgress {
    // Mark that native write processing began, including validation and preflight.
    // Counter presence reflects this moment, while positive counts require a
    // confirmed append.
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

    // Record an expected total before marking the append uncertain. A later
    // snapshot can expose confirmed earlier batches while leaving the current
    // append outcome unknown.
    pub(crate) fn begin_append(&self, count: usize) {
        self.expect_records(count);
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .append_in_flight = true;
    }

    // Update counts and last identity together under one lock. Only storage
    // confirmation can close the current in-flight interval.
    pub(crate) fn confirmed(&self, records: usize, batch_id: u64, sequence_no: u64) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.batches += 1;
        state.records += records;
        state.last_commit = Some((batch_id, sequence_no));
        state.append_in_flight = false;
    }

    pub(crate) fn snapshot(&self) -> WriteOutcome {
        // Read a coherent snapshot of flags, counts and commit identity under the
        // same mutex. The outcome describes known write effects rather than the
        // return value received by a caller.
        let state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        // Earlier batches can be confirmed even when the most recent append has an
        // unknown outcome. Keep its uncertainty separate from the confirmed counts.
        let name = if state.append_in_flight {
            "unknown"
        } else if !state.started {
            "not_started"
        } else if state.batches == 0 {
            "not_committed"
        // Use the planned main-data record total to distinguish a fully confirmed
        // request from a prefix. The host may still report an error after all
        // planned records have been confirmed.
        } else if state.expected_records == Some(state.records) {
            "committed"
        } else {
            "partially_committed"
        };
        WriteOutcome {
            state: name,
            // Expose zero after tracking has started and no batch has been
            // confirmed. Before tracking begins, absent counters carry a different
            // meaning.
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
