//! Fixed, optional diagnostics for one measured search. This owns no resource
//! credit, changes no admission/retry policy, and never enters public query JSON.
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::json::{JsonBuilder, JsonOperationContext, JsonValue};
use crate::Result;

// This string identifies the diagnostic report schema. Stage names, counter
// meanings and candidate-count boundaries are interpreted under this version by
// report readers.
pub(crate) const VERSION: &str = "cofferdb-search-cost-v1";

// Each stage selects a fixed slot in the trace. A stage records work through the
// contexts attributed to it, even when several stages call the same storage or
// scoring helper.
#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Preparation,
    Recall,
    ScoreCompletion,
    NormalizationFusion,
    ResultMaterialization,
}
impl Stage {
    // Emit stages in execution order for report readers, including stages the
    // search never entered. Each slot's entered flag distinguishes an absent stage
    // from one that recorded no measurable work.
    const ALL: [Self; 5] = [
        Self::Preparation,
        Self::Recall,
        Self::ScoreCompletion,
        Self::NormalizationFusion,
        Self::ResultMaterialization,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Preparation => "preparation",
            Self::Recall => "recall",
            Self::ScoreCompletion => "score_completion",
            Self::NormalizationFusion => "normalization_fusion",
            Self::ResultMaterialization => "result_materialization",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Counter {
    // A visit counts a lexical document examined by the active path. Two scoring
    // passes over the same chunk can contribute two visits, so this is a work
    // measure rather than a count of unique documents.
    TextDocumentVisits,
    // Posting work and document work advance independently. Several matching term
    // lists can be examined for the same chunk, and a streamed text scan can visit
    // documents without a posting list.
    TextPostingVisits,
    // Count candidate vectors examined even when a filter rejects one before
    // distance scoring. The companion component count measures only dimensions
    // actually passed to the scorer.
    VectorChecks,
    DistanceComponents,
    // Fusion can use the BM25 or vector score already attached to a recall hit.
    // Record each route separately when consuming its rows, including a chunk
    // recalled by both routes.
    Bm25ScoreReuse,
    VectorScoreReuse,
    // An empty unfiltered lexical result can prove that vector-only candidates
    // have zero BM25 score. Count each requested row completed from that proof
    // without another text search.
    ZeroMatchProofReuse,
    // Normalization visits BM25 scores to determine the lexical maximum. The
    // fusion loop then visits each distinct candidate ID, so overlapping retrieval
    // windows affect these two counts differently.
    NormalizationScoreVisits,
    FusionCandidateVisits,
    // Requested bytes describe the logical payload or index extent chosen by a
    // reader. Count the actual read calls and returned byte counts separately,
    // including cases where the full extent could not be consumed.
    PrimaryRequestedBytes,
    IndexRequestedBytes,
    PrimaryReadCalls,
    IndexReadCalls,
    PrimaryReadBytes,
    IndexReadBytes,
}
impl Counter {
    // Use one counter vocabulary for every stage and for the trace totals. New
    // metrics must appear here as well as in the enum and its report-name mapping
    // to be included in the JSON report.
    const ALL: [Self; 15] = [
        Self::TextDocumentVisits,
        Self::TextPostingVisits,
        Self::VectorChecks,
        Self::DistanceComponents,
        Self::Bm25ScoreReuse,
        Self::VectorScoreReuse,
        Self::ZeroMatchProofReuse,
        Self::NormalizationScoreVisits,
        Self::FusionCandidateVisits,
        Self::PrimaryRequestedBytes,
        Self::IndexRequestedBytes,
        Self::PrimaryReadCalls,
        Self::IndexReadCalls,
        Self::PrimaryReadBytes,
        Self::IndexReadBytes,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::TextDocumentVisits => "text_document_visits",
            Self::TextPostingVisits => "text_posting_visits",
            Self::VectorChecks => "vector_checks",
            Self::DistanceComponents => "distance_components",
            Self::Bm25ScoreReuse => "bm25_score_reuse",
            Self::VectorScoreReuse => "vector_score_reuse",
            Self::ZeroMatchProofReuse => "zero_match_proof_reuse",
            Self::NormalizationScoreVisits => "normalization_score_visits",
            Self::FusionCandidateVisits => "fusion_candidate_visits",
            Self::PrimaryRequestedBytes => "primary_requested_bytes",
            Self::IndexRequestedBytes => "index_requested_bytes",
            Self::PrimaryReadCalls => "primary_read_calls",
            Self::IndexReadCalls => "index_read_calls",
            Self::PrimaryReadBytes => "primary_read_bytes",
            Self::IndexReadBytes => "index_read_bytes",
        }
    }
}

// These cardinalities are observations at specific boundaries. An early return may
// leave a boundary unobserved, which has a different meaning from an observed
// empty result.
#[derive(Clone, Copy)]
pub(crate) enum Count {
    // The two recall sizes are captured after metadata filtering. Their sum can
    // exceed the union because the same chunk may survive both retrieval routes.
    Bm25Recall,
    VectorRecall,
    RecallUnion,
    // Fusion retains its ranked working window before final result construction.
    // Record returned separately so a report need not infer the user-visible count
    // from an intermediate cap.
    FusionRetained,
    Returned,
}
impl Count {
    const ALL: [Self; 5] = [
        Self::Bm25Recall,
        Self::VectorRecall,
        Self::RecallUnion,
        Self::FusionRetained,
        Self::Returned,
    ];
    // These names identify observed result windows in the report. A count is the
    // last measured cardinality at that boundary, so summing repeated observations
    // would double-count candidates.
    fn name(self) -> &'static str {
        match self {
            Self::Bm25Recall => "bm25_recall_count",
            Self::VectorRecall => "vector_recall_count",
            Self::RecallUnion => "recall_union_count",
            Self::FusionRetained => "fusion_retained_count",
            Self::Returned => "returned_count",
        }
    }
}

#[derive(Default)]
struct Slot {
    // Per-stage work totals indexed by Counter. They accumulate events such as
    // visits and bytes across repeated work in the same stage.
    counters: [AtomicU64; 15],
    // Accumulated stage timer durations in nanoseconds. Conversion to milliseconds
    // is deferred to the diagnostic report.
    elapsed_ns: AtomicU64,
    // Set when an active stage timer starts, distinguishing an entered stage from
    // an untouched slot with zero elapsed time.
    entered: AtomicBool,
}
struct State {
    // Fixed slots indexed by Stage so recording a search phase requires no
    // dynamically allocated map.
    slots: [Slot; 5],
    // Last observed candidate cardinality at each named boundary, indexed by Count.
    // These values are observations rather than additive work.
    counts: [AtomicU64; 5],
    // A separate presence bit per cardinality lets the report distinguish an
    // observed empty window from a boundary the search never reached.
    count_known: [AtomicBool; 5],
    // Ends recording when the measured search finishes, including for diagnostic
    // handles carried by longer-lived resources.
    active: AtomicBool,
    // Sticky flag for overflowing a counter, timer or cross-stage total. A
    // saturated trace cannot supply exact measurements.
    saturated: AtomicBool,
}

// The trace owns the lifetime of recording. Handles and timers may retain the
// shared Arc after the measured search ends, which is why dropping the last handle
// cannot define the measurement boundary.
pub(crate) struct SearchCostTrace(Arc<State>);
#[derive(Clone)]
pub(crate) struct Handle {
    // Shared state for this one measured search. Cloning a handle keeps all stages
    // in the same trace.
    state: Arc<State>,
    // Stage to which this handle attributes future work. Creating a new stage
    // handle leaves older handles' attribution intact.
    stage: Stage,
}
pub(crate) struct Timer {
    // Handle fixed to the stage that started the timer, even if later work switches
    // stages.
    handle: Handle,
    // Monotonic instant captured when the timer is created. Drop adds elapsed
    // nanoseconds to the counter only while the trace is active.
    started: Instant,
}
// This borrowed guard ends recording on early returns as well as successful exits.
// The trace itself remains available for building a report after the guard has
// dropped.
pub(crate) struct Finish<'a>(&'a SearchCostTrace);
impl Drop for Finish<'_> {
    fn drop(&mut self) {
        self.0.finish();
    }
}

impl SearchCostTrace {
    pub(crate) fn new() -> Self {
        // Slots begin at zero, but a zero in the cardinality array has no meaning
        // until its presence bit is set. That distinction matters when a search
        // fails before a later stage.
        Self(Arc::new(State {
            slots: std::array::from_fn(|_| Slot::default()),
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
            count_known: std::array::from_fn(|_| AtomicBool::new(false)),
            active: AtomicBool::new(true),
            saturated: AtomicBool::new(false),
        }))
    }
    // Unqualified operation work starts in preparation. Stage-specific clones
    // share the same totals while routing future events to their own fixed slot.
    pub(crate) fn handle(&self) -> Handle {
        Handle {
            state: Arc::clone(&self.0),
            stage: Stage::Preparation,
        }
    }
    // Cached reservations can outlive this search. Their diagnostic handles must
    // stop recording before verification/journal IO or another operation begins.
    pub(crate) fn finish(&self) {
        // Closing a trace is idempotent. The guard, an explicit finish, and the
        // trace's own destructor can all reach this store without clearing
        // measurements already available for reporting.
        self.0.active.store(false, Ordering::Relaxed);
    }
    pub(crate) fn scope(&self) -> Finish<'_> {
        Finish(self)
    }
    pub(crate) fn valid(&self) -> bool {
        // Even individually valid stage counters can overflow when added. Form
        // every total before accepting the trace's saturation flag as its validity
        // result.
        for counter in Counter::ALL {
            self.total(counter);
        }
        !self.0.saturated.load(Ordering::Relaxed)
    }
    pub(crate) fn value(&self, stage: Stage, counter: Counter) -> u64 {
        self.0.slots[stage as usize].counters[counter as usize].load(Ordering::Relaxed)
    }
    pub(crate) fn total(&self, counter: Counter) -> u64 {
        // A per-stage value may be representable even when their sum is not. Add
        // in integer space and retain the saturated value so report readers can
        // distinguish it through the shared overflow flag.
        Stage::ALL.into_iter().fold(0u64, |n, stage| {
            n.checked_add(self.value(stage, counter))
                .unwrap_or_else(|| {
                    self.0.saturated.store(true, Ordering::Relaxed);
                    u64::MAX
                })
        })
    }
    pub(crate) fn json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        // Force every cross-stage total to be checked before serializing the
        // saturation flag. An overflow found only during aggregation must still be
        // visible in this same report.
        let _valid = self.valid();
        let counters = |stage: Option<Stage>| {
            // Build the same work fields for every stage and for the aggregate.
            // Reading through a common counter list keeps zeros and units
            // comparable across those objects.
            let mut fields = Counter::ALL.map(|counter| (counter.name(), JsonValue::Null));
            for (counter, field) in Counter::ALL.into_iter().zip(&mut fields) {
                let n =
                    stage.map_or_else(|| self.total(counter), |stage| self.value(stage, counter));
                // Use the exact-counter encoder for both individual slots and
                // totals. A raw conversion to binary64 would merge distinct counts
                // above JSON's safe integer range.
                field.1 = builder.optional_u64(Some(n))?;
            }
            builder.object(fields)
        };
        // Emit all stages in their fixed schema order. A consumer can distinguish
        // a stage never entered from a short or idle stage by its presence flag,
        // even when both show zero work.
        let stages = builder.array(Stage::ALL.into_iter().map(|stage| {
            let slot = &self.0.slots[stage as usize];
            builder.object([
                ("stage", builder.string(stage.name())?),
                (
                    "entered",
                    JsonValue::Bool(slot.entered.load(Ordering::Relaxed)),
                ),
                (
                    "elapsed_ms",
                    // Stage durations use fractional milliseconds for the report.
                    // This conversion applies only to timing. Work and cardinality
                    // counters retain exact integer or decimal-text encoding.
                    JsonValue::Number(slot.elapsed_ns.load(Ordering::Relaxed) as f64 / 1e6),
                ),
                ("work", counters(Some(stage))?),
            ])
        }))?;
        // Keep unknown candidate counts as null. An operation may have performed
        // work in an earlier stage and failed before observing the later window
        // sizes.
        let mut fields = Count::ALL.map(|count| (count.name(), JsonValue::Null));
        for (count, field) in Count::ALL.into_iter().zip(&mut fields) {
            field.1 = builder.optional_u64(
                self.0.count_known[count as usize]
                    .load(Ordering::Relaxed)
                    .then(|| self.0.counts[count as usize].load(Ordering::Relaxed)),
            )?;
        }
        // Cardinalities describe different points in the search and are already
        // observations. Emit this object separately from the per-stage work
        // totals.
        let counts = builder.object(fields)?;
        builder.object([
            ("version", builder.string(VERSION)?),
            ("counter_encoding", builder.string("exact JSON integers through 2^53-1; unsigned decimal strings above it")?),
            ("counter_saturated", JsonValue::Bool(self.0.saturated.load(Ordering::Relaxed))),
            ("stages", stages),
            ("totals", counters(None)?),
            ("candidate_counts", counts),
            ("staging_read_bytes", JsonValue::Number(0.0)),
            // Application reads can be served by the OS cache. The report exposes
            // their call and byte counters while leaving the physical-disk
            // measurement unavailable.
            ("physical_read_bytes", JsonValue::Null),
            ("physical_read_reason", builder.string("physical disk/cache IO is unavailable; application Read/read_at bytes are not physical IO")?),
        ])
    }
}
impl Drop for SearchCostTrace {
    fn drop(&mut self) {
        self.finish();
    }
}
impl Handle {
    // Cloning a handle for a stage shares the measurement state but changes only
    // this handle's label. Work still holding a previous handle continues to
    // belong to its original stage.
    pub(crate) fn stage(&self, stage: Stage) -> Self {
        Self {
            state: Arc::clone(&self.state),
            stage,
        }
    }
    pub(crate) fn add(&self, counter: Counter, amount: u64) {
        // Handles can outlive a finished search. Skip an update when the handle
        // observes an inactive trace. This check does not wait for a recorder
        // already in flight.
        if self.state.active.load(Ordering::Relaxed) {
            self.add_atomic(
                &self.state.slots[self.stage as usize].counters[counter as usize],
                amount,
            );
        }
    }
    fn add_atomic(&self, counter: &AtomicU64, amount: u64) {
        // Clamp at the integer maximum instead of wrapping a large measurement to a
        // plausible small value. Flag that loss of precision for consumers of the
        // benchmark report.
        // An atomic read-modify-write retries a contested increment against the
        // updated value, so concurrent stage events cannot overwrite one another.
        let previous = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_add(amount))
            })
            .unwrap();
        // Use the pre-update value from the successful atomic operation to
        // determine whether this increment saturated. A counter exactly equal to
        // the maximum can still be an exact measurement.
        if previous.checked_add(amount).is_none() {
            self.state.saturated.store(true, Ordering::Relaxed);
        }
    }
    pub(crate) fn count(&self, count: Count, amount: usize) {
        // A candidate count is an observed window size, not a cumulative workload.
        // Store the size and mark it known so a real zero differs from a stage
        // that stopped before producing a count.
        if self.state.active.load(Ordering::Relaxed) {
            self.state.counts[count as usize].store(amount as u64, Ordering::Relaxed);
            self.state.count_known[count as usize].store(true, Ordering::Relaxed);
        }
    }
    pub(crate) fn timer(&self) -> Timer {
        // Mark the stage entered when creating its timer. A stage can receive
        // elapsed time without performing any counted reads or scoring work. Final
        // report fields can still distinguish that stage from one never entered.
        if self.state.active.load(Ordering::Relaxed) {
            self.state.slots[self.stage as usize]
                .entered
                .store(true, Ordering::Relaxed);
        }
        Timer {
            handle: self.clone(),
            started: Instant::now(),
        }
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        // Use the same activity boundary as the counters. A timer dropped after
        // trace completion contributes no late time, while an active timer
        // converts its monotonic elapsed duration to nanoseconds.
        if self.handle.state.active.load(Ordering::Relaxed) {
            let nanos = self.started.elapsed().as_nanos();
            // The elapsed duration is wider than the reported counter. Mark
            // truncation before converting it, and let the atomic addition
            // independently detect any overflow from combining multiple timers for
            // this stage.
            if nanos > u64::MAX as u128 {
                self.handle.state.saturated.store(true, Ordering::Relaxed);
            }
            self.handle.add_atomic(
                &self.handle.state.slots[self.handle.stage as usize].elapsed_ns,
                nanos.min(u64::MAX as u128) as u64,
            );
        }
    }
}

// Counts every attempted application read and each successfully returned byte,
// including short reads, checksum drains and bytes preceding a later error.
pub(crate) struct IndexReader<'a, R> {
    // Underlying index reader. Seeks are forwarded, and the wrapper records only
    // actual application read calls and returned bytes.
    pub(crate) source: &'a mut R,
    // Optional context carrying the search stage. Reads without a traced context
    // preserve the same underlying I/O behavior.
    pub(crate) context: Option<&'a JsonOperationContext>,
}
impl<R: Read> Read for IndexReader<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        // Count attempts separately from successes. The underlying read can return
        // an error, EOF or a partial buffer, and only the returned prefix
        // contributes bytes to the stage.
        if let Some(context) = self.context {
            context.search_cost_add(Counter::IndexReadCalls, 1);
        }
        let count = self.source.read(bytes)?;
        if let Some(context) = self.context {
            context.search_cost_add(Counter::IndexReadBytes, count as u64);
        }
        Ok(count)
    }
}
impl<R: Seek> Seek for IndexReader<'_, R> {
    fn seek(&mut self, offset: SeekFrom) -> std::io::Result<u64> {
        // Forward seeks as positioning operations. A later read can still charge
        // application bytes at the new offset, while seeks themselves add no
        // payload bytes.
        self.source.seek(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonInputKind, JsonOperationContext};

    #[test]
    fn trace_stage_clones_share_budget_without_relabeling_or_recording_after_finish() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<JsonOperationContext>();
        send_sync::<Handle>();
        let original = JsonOperationContext::standalone();
        let available = original.available_workspace_bytes();
        let trace = SearchCostTrace::new();
        let recall = original
            .with_search_cost(&trace)
            .search_cost_stage(Stage::Recall);
        let completion = recall.search_cost_stage(Stage::ScoreCompletion);
        let mut reservation = recall.workspace_reservation();
        reservation.grow_to(17, 17, "operation_scratch").unwrap();
        assert_eq!(completion.available_workspace_bytes(), available - 17);
        recall.search_cost_add(Counter::VectorChecks, 3);
        completion.search_cost_add(Counter::VectorChecks, 2);
        assert_eq!(trace.value(Stage::Recall, Counter::VectorChecks), 3);
        assert_eq!(
            trace.value(Stage::ScoreCompletion, Counter::VectorChecks),
            2
        );
        drop(trace.scope());
        let frozen = trace.json(&JsonBuilder::host()).unwrap();
        drop(
            completion
                .search_cost_stage(Stage::ResultMaterialization)
                .search_cost_timer(),
        );
        assert_eq!(trace.json(&JsonBuilder::host()).unwrap(), frozen);
        recall.search_cost_add(Counter::VectorChecks, 100);
        completion.search_cost_count(Count::Returned, 9);
        assert_eq!(trace.total(Counter::VectorChecks), 5);
        drop(reservation);
        assert_eq!(original.available_workspace_bytes(), available);
        let json = trace.json(&JsonBuilder::host()).unwrap();
        assert!(json
            .get("candidate_counts")
            .unwrap()
            .get("returned_count")
            .unwrap()
            .is_null());
        assert!(json.get("physical_read_bytes").unwrap().is_null());
    }

    #[test]
    fn trace_exact_large_integers_and_overflow_invalidation_are_budgeted() {
        let trace = SearchCostTrace::new();
        let context = JsonOperationContext::standalone().with_search_cost(&trace);
        context.search_cost_add(Counter::PrimaryReadBytes, u64::MAX - 1);
        assert!(trace.valid());
        let mut work = context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let json = trace.json(&builder).unwrap();
        assert_eq!(
            json.get("totals")
                .unwrap()
                .get("primary_read_bytes")
                .unwrap()
                .as_str(),
            Some("18446744073709551614")
        );
        assert!(work.reserved_bytes() > 0);
        context
            .search_cost_stage(Stage::Recall)
            .search_cost_add(Counter::PrimaryReadBytes, 2);
        assert!(!trace.valid()); // Aggregate overflow also invalidates a measurement.
        context.search_cost_add(Counter::PrimaryReadBytes, 2);
        assert!(!trace.valid()); // Per-stage overflow remains explicit.
        let json = trace.json(&JsonBuilder::host()).unwrap();
        assert_eq!(json.get("counter_saturated").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn index_reads_count_partial_success_and_attempts_before_error_without_seek_bytes() {
        struct Input {
            remaining: usize,
        }
        impl Read for Input {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return Err(std::io::ErrorKind::PermissionDenied.into());
                }
                let n = output.len().min(self.remaining).min(3);
                output[..n].fill(1);
                self.remaining -= n;
                Ok(n)
            }
        }
        impl Seek for Input {
            fn seek(&mut self, _: SeekFrom) -> std::io::Result<u64> {
                Ok(0)
            }
        }
        let trace = SearchCostTrace::new();
        let context = JsonOperationContext::standalone().with_search_cost(&trace);
        let mut source = Input { remaining: 7 };
        let mut reader = IndexReader {
            source: &mut source,
            context: Some(&context),
        };
        reader.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(
            reader.read_exact(&mut [0; 10]).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(trace.total(Counter::IndexReadBytes), 7);
        assert_eq!(trace.total(Counter::IndexReadCalls), 4);
        assert_eq!(trace.total(Counter::PrimaryReadBytes), 0);
    }
}
