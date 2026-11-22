//! Build benchmark expectations from the fixed source documents, chunk rows and
//! embeddings. Use the imported catalog to resolve identifiers and the native tie
//! order, then compare each measured hit against the expected rank and score.
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::*;
use crate::index::text::{BM25_B, BM25_K1};
use crate::index::vector::raw_score;
use crate::json::ReservedVec;
use crate::temporary::{ExternalKeyLookup, ExternalUniqueKeys};

// Bound the number of queries whose candidate heaps and filter workspace coexist
// during a source pass. This is a preparation window, independent of candidate
// limits and the number of queries in the final report.
const QUERY_WINDOW: usize = 4;

// Expectations come from hash-verified source rows, never search results or index bytes.
// Stored IDs are used only to bind the documented internal-ID tie break to this import.
pub(super) struct Acceptance {
    // Expected hits organized by mode index, then source-query index, then rank.
    expected: Vec<Vec<Vec<ExpectedHit>>>,
    // Reservation accompanying the stored expectations through verification.
    _work: JsonReservation,
}

struct ExpectedHit {
    // Dataset chunk identifier expected at this rank.
    external_id: String,
    // Expected final score for this mode, checked with the benchmark's score
    // tolerance.
    score: f64,
}

impl Acceptance {
    // The journal iterates the same stored ranks used by verification. Borrow the
    // external IDs and copy the scores so recording the expectations does not
    // rebuild or reorder the reference list.
    pub(super) fn expected_hits(
        &self,
        mode: usize,
        query: usize,
    ) -> impl ExactSizeIterator<Item = (&str, f64)> {
        self.expected[mode][query]
            .iter()
            .map(|hit| (hit.external_id.as_str(), hit.score))
    }

    // Use a fixed window for ordinary runs. The worker can retry a
    // capacity-limited group one query at a time without changing the order or the
    // contents expected for any query.
    pub(super) fn prepare(
        db: &Database,
        collection: &str,
        dataset: &BenchmarkDataset,
        profile: ResourceProfile,
        modes: &[BenchmarkQueryMode],
        context: &JsonOperationContext,
    ) -> Result<Self> {
        Self::prepare_windowed(
            db,
            collection,
            dataset,
            profile,
            modes,
            context,
            QUERY_WINDOW,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_windowed(
        db: &Database,
        collection: &str,
        dataset: &BenchmarkDataset,
        profile: ResourceProfile,
        modes: &[BenchmarkQueryMode],
        context: &JsonOperationContext,
        window: usize,
    ) -> Result<Self> {
        // Keep the decoded source documents charged throughout all query windows.
        // Their metadata is reused when evaluating chunk filters; a new window
        // should not require another document map.
        let mut source_work = context.catalog_reservation();
        // Hash the same stream that populates the document lookup. A digest
        // mismatch invalidates the decoded map before any imported row count or
        // ranking is considered.
        let documents = read_verified_baseline_input(
            &dataset.documents_path,
            "documents",
            &dataset.manifest.documents_sha256,
            |source| {
                read_sqlite_baseline_documents(
                    &dataset.documents_path,
                    source,
                    context,
                    &mut source_work,
                )
            },
        )?;
        // The source catalog's document map accepts external document IDs from the
        // chunk artifact. It is separate from the database snapshot, whose
        // internal IDs determine native tie breaks.
        let mut source_catalog = Catalog::default();
        source_catalog.documents = documents;
        let mut work = context.reservation();
        let mut expected = Vec::new();
        // Reserve one destination vector per mode, each large enough for all
        // dataset queries. Window results can then transfer their hit vectors
        // without growing the destination during append.
        work.reserve_vec(&mut expected, modes.len(), JsonInputKind::BenchmarkArtifact)?;
        for _ in modes {
            let mut queries = Vec::new();
            work.reserve_vec(
                &mut queries,
                dataset.queries.len(),
                JsonInputKind::BenchmarkArtifact,
            )?;
            expected.push(queries);
        }
        // Use one read snapshot for the whole preparation. Once a source external
        // ID is matched to an imported row, the internal tie-break ID must remain
        // stable across windows and modes.
        db.with_read_snapshot(|catalog, _| {
            let collection_id = &catalog.collection_by_name(collection)?.collection_id;
            // Count only live documents in the named collection. A matching count
            // is a prerequisite for reference building; it does not by itself
            // identify the matching chunks or establish a ranking.
            if catalog
                .live_documents()
                .filter(|document| &document.collection_id == collection_id)
                .count()
                != source_catalog.documents.len()
            {
                return Err(mismatch(
                    "imported document count differs from fixed dataset",
                ));
            }
            // Build expectations a window at a time so the working set scales with
            // simultaneous queries. Each result still needs collection-wide
            // statistics from the fixed source; slicing the queries must not slice
            // the source documents.
            for queries in dataset.queries.chunks(window) {
                let attempt = || {
                    prepare_queries(
                        db,
                        catalog,
                        collection_id,
                        &source_catalog,
                        dataset,
                        queries,
                        profile,
                        modes,
                        context,
                    )
                };
                // Append only a wholly prepared window. A partially accumulated
                // mode or query list would shift the journal's mode/query index on
                // the next append.
                match attempt() {
                    Ok(part) => append_expected(&mut expected, &mut work, part),
                    // Fallback is reserved for capacity pressure and requires a
                    // group that can actually shrink. Parse errors, hash
                    // mismatches and errors from a single-query window propagate
                    // with their original diagnosis.
                    Err(error) if queries.len() > 1 && window_capacity_error(&error) => {
                        // Failed window data and credit are gone before one bounded fallback.
                        for query in queries {
                            // Reuse the same snapshot, source and mode order for
                            // each retry. The fallback changes the simultaneous
                            // working set, not the collection statistics or filter
                            // semantics of a query.
                            let part = prepare_queries(
                                db,
                                catalog,
                                collection_id,
                                &source_catalog,
                                dataset,
                                std::slice::from_ref(query),
                                profile,
                                modes,
                                context,
                            )?;
                            append_expected(&mut expected, &mut work, part);
                        }
                    }
                    // Do not retry unrelated errors. In particular, rereading a
                    // mismatched source under a smaller window could mask a change
                    // to the fixed input.
                    Err(error) => return Err(error),
                }
            }
            Ok(())
        })?;
        // The verified reference outlives the source document workspace. Only
        // ranked external IDs, their scores and the reservation for those retained
        // values escape preparation.
        Ok(Self {
            expected,
            _work: work,
        })
    }

    // Compare in measured rank order using external chunk IDs and final scores.
    // Passing a set of IDs here would discard both tie order and a score
    // regression that leaves the order unchanged.
    pub(super) fn verify(&self, mode: usize, query: usize, results: &SearchResults) -> Result<()> {
        verify_hits(
            &self.expected[mode][query],
            results
                .results
                .iter()
                .map(|hit| (hit.chunk_external_id.as_str(), hit.final_score)),
        )
    }

    #[cfg(test)]
    pub(super) fn assert_serial_source_candidates(
        db: &Database,
        collection: &str,
        dataset: &BenchmarkDataset,
        profile: ResourceProfile,
        context: &JsonOperationContext,
    ) {
        let mut work = context.catalog_reservation();
        let mut documents = Catalog::default();
        documents.documents = read_verified_baseline_input(
            &dataset.documents_path,
            "documents",
            &dataset.manifest.documents_sha256,
            |source| {
                read_sqlite_baseline_documents(&dataset.documents_path, source, context, &mut work)
            },
        )
        .unwrap();
        db.with_read_snapshot(|catalog, _| {
            let collection = &catalog.collection_by_name(collection)?.collection_id;
            for queries in dataset.queries.chunks(QUERY_WINDOW) {
                let window = source_candidates_window(
                    db, catalog, collection, &documents, dataset, queries, profile, false, context,
                )?;
                for (query, source) in queries.iter().zip(window.as_slice()) {
                    let (serial_text, serial_vector) = source_candidates(
                        db, catalog, collection, &documents, dataset, query, profile, context,
                    )?;
                    assert_eq!(
                        source
                            .text
                            .as_slice()
                            .iter()
                            .map(|hit| (&hit.chunk_id, hit.score.to_bits()))
                            .collect::<Vec<_>>(),
                        serial_text
                            .as_slice()
                            .iter()
                            .map(|hit| (&hit.chunk_id, hit.score.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        source
                            .vector
                            .as_slice()
                            .iter()
                            .map(|hit| (
                                &hit.chunk_id,
                                hit.raw_score.to_bits(),
                                hit.normalized_score.to_bits()
                            ))
                            .collect::<Vec<_>>(),
                        serial_vector
                            .as_slice()
                            .iter()
                            .map(|hit| (
                                &hit.chunk_id,
                                hit.raw_score.to_bits(),
                                hit.normalized_score.to_bits()
                            ))
                            .collect::<Vec<_>>()
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    }
}

// Only memory-budget errors and exhausted temporary-disk resources can benefit
// from reducing the simultaneous query window. Other ResourceExhausted errors need
// not improve on retry.
fn window_capacity_error(error: &Error) -> bool {
    crate::json::is_memory_budget_error(error)
        || (error.code() == ErrorCode::ResourceExhausted
            && error
                .detail()
                .iter()
                .any(|(key, value)| key == "resource" && value == "temporary_disk"))
}

// Move one group's query vectors into preallocated mode slots. Their owned hit
// lists and ID strings survive; only the group's outer and per-mode vector buffers
// are released.
fn append_expected(
    expected: &mut [Vec<Vec<ExpectedHit>>],
    work: &mut JsonReservation,
    mut part: Acceptance,
) {
    // Compute the discarded buffer charge from capacity, not length. The temporary
    // per-mode vectors may have reserved unused slots, while the nested hit
    // vectors move intact into their destinations.
    let released = part.expected.capacity() * std::mem::size_of::<Vec<Vec<ExpectedHit>>>()
        + part
            .expected
            .iter()
            .map(|queries| queries.capacity() * std::mem::size_of::<Vec<ExpectedHit>>())
            .sum::<usize>();
    // Append in mode order so query positions remain relative to the original
    // dataset across windows. The initial full-dataset reservation makes this
    // ownership transfer allocation-free for the destination slots.
    for (target, mut queries) in expected.iter_mut().zip(part.expected) {
        debug_assert!(target.capacity() - target.len() >= queries.len());
        target.append(&mut queries);
    }
    // Remove the charge for the consumed container buffers, then merge the
    // remaining charge for the retained hit vectors and strings into the result
    // owner.
    part._work.retain(part._work.covered_bytes() - released);
    work.merge(part._work);
}

#[allow(clippy::too_many_arguments)]
fn prepare_queries(
    db: &Database,
    catalog: &Catalog,
    collection: &str,
    documents: &Catalog,
    dataset: &BenchmarkDataset,
    queries: &[BenchmarkQuery],
    profile: ResourceProfile,
    modes: &[BenchmarkQueryMode],
    context: &JsonOperationContext,
) -> Result<Acceptance> {
    // This reservation is local to one attempt. A source or capacity error drops
    // the whole partial result before the caller retries with a narrower window.
    let mut work = context.reservation();
    let mut expected = Vec::new();
    work.reserve_vec(&mut expected, modes.len(), JsonInputKind::BenchmarkArtifact)?;
    for _ in modes {
        let mut items = Vec::new();
        work.reserve_vec(&mut items, queries.len(), JsonInputKind::BenchmarkArtifact)?;
        expected.push(items);
    }
    // Build the expected results from source candidates once per query window. Keep
    // the extra scores for hybrid fusion when requested; single-signal modes must
    // receive only the signal they would use in a real search.
    let candidates = source_candidates_window(
        db,
        catalog,
        collection,
        documents,
        dataset,
        queries,
        profile,
        // Collect the extra component scores only when hybrid is requested. They
        // allow fusion to score a candidate admitted by one route with evidence
        // from the other.
        modes.contains(&BenchmarkQueryMode::Hybrid),
        context,
    )?;
    // The source pass returns one candidate set per window query in order. Build
    // all requested modes from that set before moving on so each mode sees the
    // same source evidence.
    for (query, source) in queries.iter().zip(candidates.as_slice()) {
        for (mode_index, &mode) in modes.iter().enumerate() {
            let ranked = crate::index::hybrid::fuse_completed(
                // An unavailable retrieval route is an empty candidate slice. This
                // preserves the scoring rule for a single-channel run even if the
                // source row also has usable data for the other route.
                if mode.has_text() {
                    source.text.as_slice()
                } else {
                    &[]
                },
                if mode.has_vector() {
                    source.vector.as_slice()
                } else {
                    &[]
                },
                if mode == BenchmarkQueryMode::Hybrid {
                    source.text_scores.as_slice()
                } else {
                    &[]
                },
                if mode == BenchmarkQueryMode::Hybrid {
                    source.vector_scores.as_slice()
                } else {
                    &[]
                },
                mode.has_text(),
                // Use mode defaults and give fusion at least the requested top_k
                // slots. Candidate breadth can exceed the final number of expected
                // hits retained for comparison.
                SearchWeights::default_for(mode.has_text(), mode.has_vector()),
                profile.params().fusion_candidates.max(query.top_k),
                context,
                // No metadata boost is part of this benchmark request. Source
                // metadata affects candidate eligibility through filters without
                // adding score at this fusion step.
                |_| Default::default(),
            )?
            .hits;
            // Reserve only the ranks that will be compared. The fused list can
            // reflect a larger profile candidate window than the query's requested
            // output count.
            let mut hits = Vec::new();
            work.reserve_vec(
                &mut hits,
                ranked.as_slice().len().min(query.top_k),
                JsonInputKind::BenchmarkArtifact,
            )?;
            for hit in ranked.as_slice().iter().take(query.top_k) {
                // Translate the tie-broken internal rank to the stable external
                // identifier used by dataset judgments and recorded search
                // results. Verification needs no borrow into the database
                // snapshot.
                let chunk = catalog.chunk_by_id(collection, hit.chunk_id)?;
                hits.push(ExpectedHit {
                    external_id: work
                        .copy_string(&chunk.chunk_external_id, JsonInputKind::BenchmarkArtifact)?,
                    score: hit.final_score,
                });
            }
            // Store this query under its mode rather than transposing the outer
            // array. Journal iteration later indexes the completed grid by mode
            // and source-query position.
            expected[mode_index].push(hits);
        }
    }
    Ok(Acceptance {
        expected,
        _work: work,
    })
}

// Compare the complete ranking in its existing order. Equal hit sets are
// insufficient: a tie broken differently or the same chunk at a different rank
// changes the result being checked.
fn verify_hits<'a>(
    expected: &[ExpectedHit],
    actual: impl IntoIterator<Item = (&'a str, f64)>,
) -> Result<()> {
    let mut actual = actual.into_iter();
    // Walk the measured iterator once alongside the reference slice. This needs no
    // second ranked buffer and can identify an unexpectedly short result at the
    // first unmatched rank.
    for expected in expected {
        let Some((id, score)) = actual.next() else {
            return Err(mismatch("acceptance query returned too few results"));
        };
        // IDs must match exactly. Apply a tolerance only to the numerical score,
        // with an absolute floor for scores near zero and relative scaling for
        // larger expected magnitudes.
        if id != expected.external_id
            // Reject a nonfinite measured score explicitly. In particular, testing
            // the tolerance alone would fail to reject a NaN because its
            // comparisons are unordered.
            || !score.is_finite()
            || (score - expected.score).abs() > 1e-12 * expected.score.abs().max(1.0)
        {
            return Err(
                mismatch("acceptance query rank or score differs from source reference")
                    .with_detail("expected_chunk_external_id", &expected.external_id)
                    .with_detail("actual_chunk_external_id", id),
            );
        }
    }
    // A matching reference prefix is insufficient. One final probe distinguishes a
    // complete match from a measured ranking with an unexpected extra hit,
    // including when the reference is empty.
    if actual.next().is_some() {
        return Err(mismatch("acceptance query returned too many results"));
    }
    Ok(())
}

// Report an acceptance discrepancy with one stable reason to distinguish
// disagreement with the source reference from I/O, resource or query failures
// earlier in the run.
fn mismatch(message: &'static str) -> Error {
    Error::new(ErrorCode::RecoveryError, message)
        .with_detail("reason", "benchmark_acceptance_mismatch")
}

#[derive(Debug)]
struct TextCandidate<'a> {
    // Imported internal chunk ID used to break ties in the same order as search.
    id: &'a str,
    // Positive BM25 score computed directly from the source chunk and collection
    // statistics.
    score: f64,
    // Whether this chunk passes the query filter. Applied after bounding the text
    // heap.
    matches: bool,
}

// Heap equality uses the same score and ID ordering as heap replacement. The
// filter flag travels with the retained candidate but must not change which
// candidates enter the bounded text window.
impl PartialEq for TextCandidate<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for TextCandidate<'_> {}
impl PartialOrd for TextCandidate<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TextCandidate<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // The maximum is the worst retained candidate.
        // Reverse only the score comparison: a lower score sorts as worse. With
        // equal scores, the larger internal ID is worse and is the first tied
        // candidate the max-heap can evict.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(other.id))
    }
}

// One state lives for every query in the active window. It holds per-query term
// statistics across corpus passes, followed by the bounded candidates and any
// scores needed to complete their union.
struct SourceQuery<'a> {
    // Query whose source-based scores are being built in this window.
    query: &'a BenchmarkQuery,
    // Compiled metadata predicate shared by this query's text and vector reference
    // passes.
    filter: Option<FilterExpr>,
    // Memory retained for the compiled predicate while scanning source artifacts.
    _filter_work: JsonReservation,
    // Tokenized search terms indexing the parallel frequency and weight arrays.
    terms: crate::tokenizer::QueryTerms<'a>,
    // Per-term occurrence counts overwritten for each source chunk.
    frequency: ReservedVec<usize>,
    // Number of source chunks containing each query term, counted before filtering.
    document_frequency: ReservedVec<usize>,
    // BM25 inverse document frequency for each query term over the whole source
    // collection.
    idf: ReservedVec<f64>,
    // Sum of chunk lengths with a minimum of one per chunk, used to derive the BM25
    // average.
    total_length: usize,
    // Bounded text candidate heap, with its worst retained score and tie-break ID
    // at the root.
    text: BinaryHeap<TextCandidate<'a>>,
    // Workspace reservation following the capacity of the text candidate heap.
    text_work: crate::json::WorkspaceReservation,
    // Builds the disk-backed set of eligible external chunk IDs for a filtered
    // vector query.
    allowed: Option<ExternalUniqueKeys>,
    // Finished membership lookup for the embeddings pass, released after initial
    // vector recall.
    lookup: Option<ExternalKeyLookup>,
    // Finalized BM25 recall hits after filtering the bounded text window.
    hits: Option<Bm25Hits>,
    // Accumulator for top vector candidates during the initial embeddings scan.
    vector: VectorCandidates,
    // Finalized initial vector recall results, used to identify missing text
    // scores.
    vectors: Option<VectorHits>,
    // Supplemental BM25 scores for chunks recalled only by the vector branch,
    // including zero.
    text_scores: Option<Bm25Hits>,
    // Accumulator for supplemental vector scores of text-only candidates that have
    // source embeddings.
    vector_scores: Option<VectorCandidates>,
}

// Keep initial recall separate from scores filled in afterward. A supplemented
// score belongs to a chunk already recalled by the other branch and cannot
// introduce a new candidate by itself.
struct SourceCandidates {
    // Initial text recall results after applying eligibility.
    text: Bm25Hits,
    // Initial vector recall results from source embeddings.
    vector: VectorHits,
    // BM25 scores completing the vector-only portion of the recalled union.
    text_scores: Bm25Hits,
    // Vector scores available for the text-only portion of that union.
    vector_scores: VectorHits,
}

#[allow(clippy::too_many_arguments)]
// Build the reference candidates from source files for one window of queries.
// First count terms across all chunks so filtering cannot change a term's
// document frequency. A second pass ranks text candidates and remembers which
// chunk IDs pass each vector filter. That lets the separate embeddings stream
// use the same filter without retaining every source chunk.
fn source_candidates_window<'a>(
    db: &Database,
    catalog: &'a Catalog,
    collection: &str,
    documents: &Catalog,
    dataset: &BenchmarkDataset,
    queries: &'a [BenchmarkQuery],
    profile: ResourceProfile,
    complete_hybrid: bool,
    context: &JsonOperationContext,
) -> Result<ReservedVec<SourceCandidates>> {
    // Reserve one query state per item in the current window. Each state retains
    // its own filter, working term counts and bounded candidate structures.
    let mut states = ReservedVec::with_capacity(queries.len(), context, "operation_scratch")?;
    for query in queries {
        // Hold the parsed predicate with its reservation. The same predicate is
        // consulted during text scoring and when building the eligible IDs for the
        // vector stream.
        let (filter, filter_work) =
            FilterExpr::parse_validated_in(&query.filter, context, JsonInputKind::Filter)?;
        // Keep a single normalized query term order. Every frequency and weight
        // vector uses that order so an index identifies the same term across all
        // source passes.
        let terms = crate::tokenizer::query_terms_in(query.text.as_deref().unwrap_or(""), context)?;
        // Use the frequency array as reusable per-chunk scratch. Document
        // frequency accumulates independently across source rows and is never
        // reset while this query window is evaluated.
        let mut frequency = ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        let mut document_frequency =
            ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        for _ in 0..terms.len() {
            frequency.push(0usize);
            document_frequency.push(0usize);
        }
        // Reserve room for one inverse document frequency per query term. The
        // actual weights cannot be computed until the first source pass finishes.
        let idf = ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        states.push(SourceQuery {
            query,
            filter,
            _filter_work: filter_work,
            terms,
            frequency,
            document_frequency,
            idf,
            total_length: 0,
            text: BinaryHeap::new(),
            text_work: context.workspace_reservation(),
            allowed: None,
            lookup: None,
            hits: None,
            // Initialize the vector window from the effective profile limit. A
            // large top_k must be able to retain at least that many eligible
            // candidates.
            vector: VectorCandidates::new(
                profile.params().vector_candidates.max(query.top_k),
                context,
            ),
            vectors: None,
            text_scores: None,
            vector_scores: None,
        });
    }
    // Count rows in the verified raw chunks stream. This provides a common corpus
    // denominator for all the query states in the window.
    let mut count = 0usize;
    // Count a term once per chunk for document frequency, however often the token
    // occurs in that chunk. All chunks contribute to length statistics before any
    // query-specific metadata filter is applied.
    visit_source_chunks(dataset, context, |row| {
        count += 1;
        for state in states.as_mut_slice() {
            // With no query terms, skip tokenization and use the minimum
            // denominator contribution. Such a state has no term weights to
            // contribute a text score.
            let length = if state.terms.len() == 0 {
                0
            } else {
                term_counts_in(
                    &row.chunk.text,
                    &state.terms,
                    state.frequency.as_mut_slice(),
                    context,
                )?
            };
            // Use a minimum length of one for every chunk. It keeps the same
            // positive averaging denominator when some source text has no
            // searchable tokens.
            state.total_length += length.max(1);
            // Repeated occurrences increase term frequency while this chunk
            // contributes once to document frequency. The scoring pass recomputes
            // each chunk's term counts.
            for (df, &tf) in state
                .document_frequency
                .as_mut_slice()
                .iter_mut()
                .zip(state.frequency.as_slice())
            {
                *df += usize::from(tf != 0);
            }
        }
        Ok(())
    })?;
    // Require a nonempty source stream and the same number of live stored chunks
    // for this collection. Later ID lookup checks associations row by row.
    if count == 0
        || count
            != catalog
                .live_chunks()
                .filter(|chunk| chunk.collection_id == collection)
                .count()
    {
        return Err(mismatch("imported chunk count differs from fixed dataset"));
    }
    // Compute all weights once per window after the whole verified corpus pass.
    // The smoothed formula gives a finite contribution even when a query term is
    // absent.
    for state in states.as_mut_slice() {
        for &df in state.document_frequency.as_slice() {
            state
                .idf
                .push(((count as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln());
        }
        // Create a disk-backed membership builder only for vector queries that
        // have a filter. Text can carry its decision directly on each retained
        // candidate.
        if state.query.embedding.is_some() && state.filter.is_some() {
            state.allowed = Some(ExternalUniqueKeys::new(
                &std::env::temp_dir(),
                db.temporary_disk_budget(),
                1024 * 1024,
                context,
                ErrorCode::DuplicateExternalId,
                "duplicate source chunk in benchmark acceptance",
            )?);
        }
    }
    // Replay the verified chunks for ranking now that corpus statistics are known.
    // The raw source supplies text and metadata while the database catalog
    // supplies the stored chunk identity.
    visit_source_chunks(dataset, context, |row| {
        // Resolve the imported internal ID for ranking and tie-breaking. The
        // comparison later converts it back to the external ID reported for the
        // measured hit.
        let stored = catalog.chunk_by_external_id(collection, &row.chunk.chunk_external_id)?;
        let chunk = source_chunk(row);
        // Require the source document before evaluating its metadata. A missing
        // reference would otherwise look like a filter rejection.
        if !documents.documents.contains_key(&chunk.document_id) {
            return Err(mismatch("source chunk references an absent document"));
        }
        for state in states.as_mut_slice() {
            // The same filter has different positions in the two recall paths.
            // Record eligibility now for the vector stream, but let every
            // positive-scoring text chunk compete for the bounded text window.
            // Filtering that window later can leave it short.
            // Evaluate one source predicate per chunk and query state. Its
            // decision travels with the text candidate and also feeds the external
            // key sorter if vector filtering is active.
            let matches = state
                .filter
                .as_ref()
                .map_or(true, |filter| filter.matches_for_chunk(documents, &chunk));
            // Store external IDs for vector eligibility. The embeddings input
            // references chunks by those keys rather than the internal IDs used by
            // ranking.
            if let Some(allowed) = &mut state.allowed {
                if matches {
                    allowed.push(&[chunk.chunk_external_id.as_bytes()], 0)?;
                }
            }
            // Use the same minimum length and token-counting procedure as the
            // corpus pass. Per-row frequencies can be recomputed without retaining
            // a full source matrix.
            let length = if state.terms.len() == 0 {
                1
            } else {
                term_counts_in(
                    &chunk.text,
                    &state.terms,
                    state.frequency.as_mut_slice(),
                    context,
                )?
                .max(1)
            };
            // Use the average length of the full corpus, including chunks outside
            // the filter. Only the later candidate materialization applies text
            // eligibility.
            let average = (state.total_length as f64 / count as f64).max(1.0);
            // Sum BM25 contributions for matching terms in query-term order. Terms
            // with zero frequency contribute no weight to this source chunk.
            let score = state
                .frequency
                .as_slice()
                .iter()
                .zip(state.idf.as_slice())
                .filter(|(&tf, _)| tf != 0)
                .map(|(&tf, &idf)| {
                    let tf = tf as f64;
                    idf * (tf * (BM25_K1 + 1.0))
                        / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length as f64 / average))
                })
                .sum::<f64>();
            // Only positive text scores enter the source window. Preserve the
            // filter decision on the candidate for the later bounded-window step.
            if score > 0.0 {
                let candidate = TextCandidate {
                    id: &stored.chunk_id,
                    score,
                    matches,
                };
                // Raise the profile window to top_k and cap it by corpus count.
                // The retained heap never needs more slots than there are source
                // chunks.
                let limit = profile
                    .params()
                    .bm25_candidates
                    .max(state.query.top_k)
                    .min(count);
                // Admit a slot before pushing while the heap has room. Once full,
                // reuse the existing root slot for a better candidate.
                if state.text.len() < limit {
                    crate::json::reserve_heap_slot(&mut state.text, &mut state.text_work, limit)?;
                    state.text.push(candidate);
                // The heap root is the worst retained candidate under the score
                // and ID ordering. A lower-ranked source row cannot displace it.
                } else if state.text.peek().is_some_and(|worst| candidate < *worst) {
                    *state.text.peek_mut().unwrap() = candidate;
                }
            }
        }
        Ok(())
    })?;
    for state in states.as_mut_slice() {
        // Turn the streamed eligible keys into a bounded lookup before scanning
        // embeddings. An unfiltered query has no membership file to construct or
        // retain.
        state.lookup = state
            .allowed
            .take()
            .map(ExternalUniqueKeys::finish_lookup)
            .transpose()?;
        // Try to return spare heap storage before allocating the final text
        // result. Charge the replacement according to the number of eligible
        // winners.
        crate::json::try_shrink_heap(&mut state.text, &mut state.text_work)?;
        let mut hits =
            Bm25Hits::with_capacity(state.text.iter().filter(|hit| hit.matches).count(), context)?;
        // Sort the already bounded text window before dropping ineligible
        // candidates. Filtering earlier would change which chunks could enter that
        // window.
        for hit in std::mem::take(&mut state.text)
            .into_sorted_vec()
            .into_iter()
            .filter(|hit| hit.matches)
        {
            // Carry the source BM25 score and stored ID. The acceptance comparison
            // needs ordering and scores but does not reconstruct per-hit term
            // lists.
            hits.push(hit.id, hit.score, std::iter::empty())?;
        }
        // The heap has been consumed. Return its remaining capacity charge after
        // moving the hits to their independently reserved result.
        state.text_work.retain(0);
        state.hits = Some(hits);
    }
    // Read the embeddings source only if this window contains a vector query. A
    // text-only window has no reason to open or verify that member here.
    if queries.iter().any(|query| query.embedding.is_some()) {
        let path = dataset
            .embeddings_path
            .as_ref()
            .ok_or_else(|| mismatch("vector acceptance requires fixed source embeddings"))?;
        // Use the declared digest to bind the source embedding pass to the dataset
        // validated by the runner.
        let expected = dataset
            .manifest
            .embeddings_sha256
            .as_deref()
            .ok_or_else(|| mismatch("vector acceptance requires a source embedding hash"))?;
        // Verify the bytes consumed by this vector pass. A valid prefix cannot
        // become a verified candidate set if the remaining source has changed.
        read_verified_baseline_input(path, "embeddings", expected, |source| {
            visit_embeddings_jsonl_in(path, source, context, u64::MAX, |_, row, _work| {
                for state in states.as_mut_slice() {
                    // Skip vector scoring for states without a query embedding.
                    // They retain an empty channel in the final result.
                    let Some(embedding) = &state.query.embedding else {
                        continue;
                    };
                    // Only eligible chunks enter the vector heap. Filtering after
                    // taking its prefix would let an ineligible high score displace
                    // a lower eligible hit, yielding a different reference result.
                    // Apply the external key membership before looking up the
                    // stored ID and calculating similarity. Filter-ineligible
                    // vectors consume no ranking slot.
                    if let Some(allowed) = &mut state.lookup {
                        if allowed.get(row.chunk_external_id.as_bytes())?.is_none() {
                            continue;
                        }
                    }
                    // Convert the artifact's external ID into the stored ID used
                    // by both recall lists. Hybrid completion needs a common
                    // identity for candidates found by either route.
                    let chunk = catalog.chunk_by_external_id(collection, &row.chunk_external_id)?;
                    // Calculate raw cosine similarity from the source vectors,
                    // then pass both raw and normalized scores to the vector
                    // window.
                    let raw = raw_score(&embedding.vector, &row.vector, DistanceMetric::Cosine);
                    state.vector.push(
                        &chunk.chunk_id,
                        raw,
                        normalize_score(raw, DistanceMetric::Cosine, false),
                    )?;
                }
                Ok(())
            })
        })?;
    }

    // Finalize initial vector recall for each query while retaining the already
    // finalized text list. Completion works against these fixed lists.
    for state in states.as_mut_slice() {
        let vector = std::mem::replace(&mut state.vector, VectorCandidates::new(0, context));
        state.vectors = Some(vector.finish());
        // U already satisfies the source metadata filter. Its supplemental
        // lookups no longer need the external filter-key staging file.
        state.lookup = None;
        // Complete scores only for requested hybrid expectations with both inputs.
        // The candidate union remains fixed, so completion cannot introduce new
        // recall members.
        if complete_hybrid && state.query.text.is_some() && state.query.embedding.is_some() {
            let text = state.hits.as_ref().unwrap();
            let vector = state.vectors.as_ref().unwrap();
            // Count vector hits absent from the text window. They still need the
            // actual source text score, which can be below the cutoff or zero.
            let missing_text = vector
                .iter()
                .filter(|v| !text.iter().any(|b| b.chunk_id == v.chunk_id))
                .count();
            // Count text hits absent from the vector window. These may have a
            // lower vector score or may have no embedding in the source at all.
            let missing_vector = text
                .iter()
                .filter(|b| !vector.iter().any(|v| v.chunk_id == b.chunk_id))
                .count();
            // Reserve capacity for supplemental scores from the set difference.
            // This keeps completion workspace proportional to the original union
            // rather than corpus size.
            if missing_text > 0 {
                state.text_scores = Some(Bm25Hits::with_capacity(missing_text, context)?);
            }
            if missing_vector > 0 {
                state.vector_scores = Some(VectorCandidates::new(missing_vector, context));
            }
        }
    }
    // These extra source streams are independently hash verified before any
    // expectation is published. Retain scores only for the bounded original U;
    // never obtain reference scores from Database search or persistent indexes.
    // Open another source pass only when a recalled vector hit needs a text score.
    // This pass shares the retained corpus statistics already calculated for
    // initial recall.
    if states.as_slice().iter().any(|s| s.text_scores.is_some()) {
        visit_source_chunks(dataset, context, |row| {
            let stored = catalog.chunk_by_external_id(collection, &row.chunk.chunk_external_id)?;
            for state in states.as_mut_slice() {
                // Skip states that need no completion, IDs with a text score
                // already present, and IDs outside the vector list. Only the
                // vector-only part of the union participates.
                if state.text_scores.is_none()
                    || state
                        .hits
                        .as_ref()
                        .unwrap()
                        .iter()
                        .any(|b| b.chunk_id == stored.chunk_id)
                    || !state
                        .vectors
                        .as_ref()
                        .unwrap()
                        .iter()
                        .any(|v| v.chunk_id == stored.chunk_id)
                {
                    continue;
                }
                // Recompute term counts only for supplemental candidates. Use the
                // original scoring length and average conventions so a
                // supplemental value means the same as an initial one.
                let length = if state.terms.len() == 0 {
                    1
                } else {
                    term_counts_in(
                        &row.chunk.text,
                        &state.terms,
                        state.frequency.as_mut_slice(),
                        context,
                    )?
                    .max(1)
                };
                let average = (state.total_length as f64 / count as f64).max(1.0);
                let score = state
                    .frequency
                    .as_slice()
                    .iter()
                    .zip(state.idf.as_slice())
                    .filter(|(&tf, _)| tf != 0)
                    .map(|(&tf, &idf)| {
                        let tf = tf as f64;
                        idf * (tf * (BM25_K1 + 1.0))
                            / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length as f64 / average))
                    })
                    .fold(0.0, |sum, part| sum + part);
                // Retain the computed score even when it is zero. A completed zero
                // must remain distinguishable from an ID never found in the source
                // stream.
                state.text_scores.as_mut().unwrap().push(
                    &stored.chunk_id,
                    score,
                    std::iter::empty(),
                )?;
            }
            Ok(())
        })?;
        for state in states.as_mut_slice() {
            // Each vector-only hit needs a text score, including a genuine zero.
            // Verify coverage before using the sorted lookup; otherwise a missing
            // source row and a computed zero could become indistinguishable.
            if let Some(scores) = &mut state.text_scores {
                let missing = state
                    .vectors
                    .as_ref()
                    .unwrap()
                    .iter()
                    .filter(|v| {
                        !state
                            .hits
                            .as_ref()
                            .unwrap()
                            .iter()
                            .any(|b| b.chunk_id == v.chunk_id)
                    })
                    .count();
                // Require every missing BM25 candidate to be covered by the
                // replay. An incomplete source association cannot silently count
                // as a zero-score omission.
                if scores.len() != missing {
                    return Err(mismatch(
                        "source text completion does not cover recalled union",
                    ));
                }
                // Sort the supplemental text values by ID for later lookup. They
                // are supplied as scores for existing candidates, not as a new
                // ranked recall stream.
                scores.sort_by_id();
            }
        }
    }
    // Open another embeddings pass when a text hit can gain a supplemental vector
    // score. This pass leaves a hit unscored if it finds no matching embedding
    // row.
    if states.as_slice().iter().any(|s| s.vector_scores.is_some()) {
        let path = dataset
            .embeddings_path
            .as_ref()
            .ok_or_else(|| mismatch("vector completion requires fixed source embeddings"))?;
        let expected = dataset
            .manifest
            .embeddings_sha256
            .as_deref()
            .ok_or_else(|| mismatch("vector completion requires a source embedding hash"))?;
        // Verify this second read as a separate source operation. An earlier
        // successful scan cannot prove that the member content stayed identical.
        read_verified_baseline_input(path, "embeddings", expected, |source| {
            visit_embeddings_jsonl_in(path, source, context, u64::MAX, |_, row, _work| {
                let chunk = catalog.chunk_by_external_id(collection, &row.chunk_external_id)?;
                for state in states.as_mut_slice() {
                    // Score only text recall hits missing from the initial vector
                    // list. An embedding row outside that set must not enlarge the
                    // original hybrid union.
                    if state.vector_scores.is_none()
                        || !state
                            .hits
                            .as_ref()
                            .unwrap()
                            .iter()
                            .any(|b| b.chunk_id == chunk.chunk_id)
                        || state
                            .vectors
                            .as_ref()
                            .unwrap()
                            .iter()
                            .any(|v| v.chunk_id == chunk.chunk_id)
                    {
                        continue;
                    }
                    // Use the same raw cosine and normalization as initial recall.
                    // A candidate omitted by the vector window still gets its
                    // actual available vector score when found.
                    let raw = raw_score(
                        &state.query.embedding.as_ref().unwrap().vector,
                        &row.vector,
                        DistanceMetric::Cosine,
                    );
                    state.vector_scores.as_mut().unwrap().push(
                        &chunk.chunk_id,
                        raw,
                        normalize_score(raw, DistanceMetric::Cosine, false),
                    )?;
                }
                Ok(())
            })
        })?;
    }
    // Admit final result slots before unpacking the per-query state. Existing
    // retained hits move to the result without recreating their allocations.
    let mut result = ReservedVec::with_capacity(queries.len(), context, "operation_scratch")?;
    // Keep the original state-array charge alive while consuming it. Individual
    // scratch owners drop with their state, and the state buffer's reservation
    // drops after the loop.
    let (states, state_work) = states.into_parts();
    for state in states {
        // Turn the supplemental vector accumulator into an ID-indexed score
        // source. An empty source means no additional scores were available,
        // rather than no original vector hits.
        let mut vector_scores = state
            .vector_scores
            .map(VectorCandidates::finish)
            .unwrap_or_else(|| VectorHits::empty(context));
        vector_scores.sort_by_id();
        // Keep initial recall and supplemental scores in separate fields. A
        // consumer can enrich scores for recalled IDs without treating completion
        // as additional recall.
        result.push(SourceCandidates {
            text: state.hits.unwrap(),
            vector: state.vectors.unwrap(),
            text_scores: state
                .text_scores
                .unwrap_or_else(|| Bm25Hits::empty(context)),
            vector_scores,
        });
    }
    drop(state_work);
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
// Serial source oracle detects arithmetic changes independently of window traversal.
fn source_candidates(
    db: &Database,
    catalog: &Catalog,
    collection: &str,
    documents: &Catalog,
    dataset: &BenchmarkDataset,
    query: &BenchmarkQuery,
    profile: ResourceProfile,
    context: &JsonOperationContext,
) -> Result<(Bm25Hits, VectorHits)> {
    let (filter, _filter_work) =
        FilterExpr::parse_validated_in(&query.filter, context, JsonInputKind::Filter)?;
    let terms = crate::tokenizer::query_terms_in(query.text.as_deref().unwrap_or(""), context)?;
    let mut frequency =
        crate::json::ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
    let mut document_frequency =
        crate::json::ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
    for _ in 0..terms.len() {
        frequency.push(0usize);
        document_frequency.push(0usize);
    }
    let mut count = 0usize;
    let mut total_length = 0usize;
    visit_source_chunks(dataset, context, |row| {
        let length = if terms.len() == 0 {
            0
        } else {
            term_counts_in(&row.chunk.text, &terms, frequency.as_mut_slice(), context)?
        };
        count += 1;
        total_length += length.max(1);
        for (df, &tf) in document_frequency
            .as_mut_slice()
            .iter_mut()
            .zip(frequency.as_slice())
        {
            *df += usize::from(tf != 0);
        }
        Ok(())
    })?;
    if count == 0
        || count
            != catalog
                .live_chunks()
                .filter(|chunk| chunk.collection_id == collection)
                .count()
    {
        return Err(mismatch("imported chunk count differs from fixed dataset"));
    }
    let average = (total_length as f64 / count as f64).max(1.0);
    let mut idf =
        crate::json::ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
    for &df in document_frequency.as_slice() {
        idf.push(((count as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln());
    }
    let limit = profile.params().bm25_candidates.max(query.top_k).min(count);
    let mut text_work = context.workspace_reservation();
    let mut text: BinaryHeap<TextCandidate<'_>> = BinaryHeap::new();
    let mut allowed = if query.embedding.is_some() && filter.is_some() {
        Some(ExternalUniqueKeys::new(
            &std::env::temp_dir(),
            db.temporary_disk_budget(),
            1024 * 1024,
            context,
            ErrorCode::DuplicateExternalId,
            "duplicate source chunk in benchmark acceptance",
        )?)
    } else {
        None
    };
    visit_source_chunks(dataset, context, |row| {
        let stored = catalog.chunk_by_external_id(collection, &row.chunk.chunk_external_id)?;
        let chunk = source_chunk(row);
        if !documents.documents.contains_key(&chunk.document_id) {
            return Err(mismatch("source chunk references an absent document"));
        }
        let matches = filter
            .as_ref()
            .map_or(true, |filter| filter.matches_for_chunk(documents, &chunk));
        if let Some(allowed) = &mut allowed {
            if matches {
                allowed.push(&[chunk.chunk_external_id.as_bytes()], 0)?;
            }
        }
        let length = if terms.len() == 0 {
            1
        } else {
            term_counts_in(&chunk.text, &terms, frequency.as_mut_slice(), context)?.max(1)
        };
        let score = frequency
            .as_slice()
            .iter()
            .zip(idf.as_slice())
            .filter(|(&tf, _)| tf != 0)
            .map(|(&tf, &idf)| {
                let tf = tf as f64;
                idf * (tf * (BM25_K1 + 1.0))
                    / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length as f64 / average))
            })
            .sum::<f64>();
        if score > 0.0 {
            let candidate = TextCandidate {
                id: &stored.chunk_id,
                score,
                matches,
            };
            if text.len() < limit {
                crate::json::reserve_heap_slot(&mut text, &mut text_work, limit)?;
                text.push(candidate);
            } else if text.peek().is_some_and(|worst| candidate < *worst) {
                *text.peek_mut().unwrap() = candidate;
            }
        }
        Ok(())
    })?;
    let mut allowed = allowed.map(ExternalUniqueKeys::finish_lookup).transpose()?;
    crate::json::try_shrink_heap(&mut text, &mut text_work)?;
    let mut hits = Bm25Hits::with_capacity(text.iter().filter(|hit| hit.matches).count(), context)?;
    for hit in text.into_sorted_vec().into_iter().filter(|hit| hit.matches) {
        hits.push(hit.id, hit.score, std::iter::empty())?;
    }
    drop(text_work);
    let mut vector =
        VectorCandidates::new(profile.params().vector_candidates.max(query.top_k), context);
    if let Some(embedding) = &query.embedding {
        let path = dataset
            .embeddings_path
            .as_ref()
            .ok_or_else(|| mismatch("vector acceptance requires fixed source embeddings"))?;
        let expected = dataset
            .manifest
            .embeddings_sha256
            .as_deref()
            .ok_or_else(|| mismatch("vector acceptance requires a source embedding hash"))?;
        read_verified_baseline_input(path, "embeddings", expected, |source| {
            visit_embeddings_jsonl_in(path, source, context, u64::MAX, |_, row, _work| {
                if let Some(allowed) = &mut allowed {
                    if allowed.get(row.chunk_external_id.as_bytes())?.is_none() {
                        return Ok(());
                    }
                }
                let chunk = catalog.chunk_by_external_id(collection, &row.chunk_external_id)?;
                let raw = raw_score(&embedding.vector, &row.vector, DistanceMetric::Cosine);
                vector.push(
                    &chunk.chunk_id,
                    raw,
                    normalize_score(raw, DistanceMetric::Cosine, false),
                )
            })
        })?;
    }
    Ok((hits, vector.finish()))
}

// Visit the chunks member through the digest verifier on every pass. The caller
// can borrow or move one decoded row at a time without retaining the whole
// dataset.
fn visit_source_chunks(
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
    mut visit: impl FnMut(crate::import::JsonlChunk) -> Result<()>,
) -> Result<()> {
    // Use the manifest hash for the same descriptor that feeds the row parser. A
    // validation pass only succeeds after the underlying stream's final digest
    // agrees.
    read_verified_baseline_input(
        &dataset.chunks_path,
        "chunks",
        &dataset.manifest.chunks_sha256,
        |source| {
            visit_chunks_jsonl_in(
                &dataset.chunks_path,
                source,
                context,
                u64::MAX,
                // Forward the decoded row and keep its parsing reservation in
                // scope for the callback. The acceptance visitor does not need the
                // physical line in its scoring state.
                |_, row, _work| visit(row),
            )
        },
    )
}

// Count all emitted tokens for length and update only the selected query terms.
// The returned length includes tokens with no matching query term.
fn term_counts_in(
    text: &str,
    terms: &crate::tokenizer::QueryTerms<'_>,
    counts: &mut [usize],
    context: &JsonOperationContext,
) -> Result<usize> {
    // Reset the caller's reusable array for this chunk. Leaving previous values
    // would inflate both BM25 term frequencies and corpus document counts.
    counts.fill(0);
    let mut length = 0usize;
    crate::tokenizer::visit_tokens_in(text, context, |token| {
        // Check total token count on each callback. Per-term counts are bounded by
        // that same total since a token increments at most its one selected term.
        length = length
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "text token count overflow"))?;
        if let Some(index) = terms.position(token) {
            counts[index] += 1;
        }
        Ok(())
    })?;
    Ok(length)
}

// Represent raw source fields as a temporary Chunk so the common metadata
// predicate can evaluate them. Stored IDs and index metadata come from the catalog
// separately.
fn source_chunk(row: crate::import::JsonlChunk) -> Chunk {
    Chunk {
        // Leave storage-only fields empty because this value is used for source
        // filtering and text scoring. Its document_id carries the external source
        // key used by the source catalog.
        chunk_id: String::new(),
        collection_id: String::new(),
        document_id: row.document_external_id,
        chunk_external_id: row.chunk.chunk_external_id,
        text: row.chunk.text,
        // The source oracle tokenizes the supplied text directly. It neither needs
        // a stored hash nor reuses a cached token count.
        text_hash: String::new(),
        token_count: 0,
        ordinal: row.chunk.ordinal,
        metadata: row.chunk.metadata,
        source_span: row.chunk.source_span,
        // Vector evidence comes from the separate verified embeddings stream. This
        // source-chunk view does not manufacture a stored embedding reference.
        embedding_id: None,
        created_at: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_window_capacity_fallback_is_bounded_and_native_failure_is_not_retried() {
        let directory = super::super::tests::text_only_benchmark_dataset("source-window-fallback");
        let setup = JsonOperationContext::standalone();
        let (mut dataset, input_work) =
            BenchmarkDataset::load_and_validate_in(&directory, &setup).unwrap();
        let db =
            Database::create(directory.join("data.coffer"), DatabaseConfig::default()).unwrap();
        let manifest = write_benchmark_import_manifest(&dataset, "notes", &setup).unwrap();
        db.import_manifest(manifest.path(), OnConflict::Error)
            .unwrap();
        drop(manifest);
        let text = (0..10_000)
            .map(|index| format!("unique{index} "))
            .collect::<String>();
        dataset.queries = (0..5)
            .map(|index| BenchmarkQuery {
                query_id: format!("window-{index}"),
                text: Some(text.clone()),
                filter: JsonValue::Null,
                embedding: None,
                top_k: 1,
                relevant_chunk_ids: Default::default(),
            })
            .collect();
        let modes = benchmark_query_modes(&dataset);
        let limit = 1536 * 1024;
        for parent in [false, true] {
            let context = JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    crate::json::WorkLedger::CATALOG_LIMIT_BYTES,
                    if parent { limit } else { 2 * limit },
                ),
            ));
            let (result, scans) = super::super::source_probe::measure(None, || {
                Acceptance::prepare(
                    &db,
                    "notes",
                    &dataset,
                    ResourceProfile::Balanced,
                    &modes,
                    &context,
                )
            });
            let acceptance = result.unwrap();
            assert_eq!(scans.opened, [1, 10, 0]);
            assert_eq!(scans.verified, scans.opened);
            for mode in 0..modes.len() {
                for query in 0..5 {
                    assert_eq!(acceptance.expected_hits(mode, query).len(), 0);
                }
            }
            drop(acceptance);
            assert_eq!(context.available_workspace_bytes(), limit);
            assert_eq!(db.temporary_disk_budget().used(), 0);
        }
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let text = dataset.queries[0].text.replace("P".repeat(265));
        let (result, scans) = super::super::source_probe::measure(None, || {
            crate::test_alloc::fail_byte_allocation_once(265, || {
                Acceptance::prepare(
                    &db,
                    "notes",
                    &dataset,
                    ResourceProfile::Balanced,
                    &modes,
                    &context,
                )
            })
        });
        dataset.queries[0].text = text;
        assert_eq!(result.err().unwrap().code(), ErrorCode::IoError);
        assert_eq!(scans.opened, [1, 0, 0]);
        assert_eq!(context.available_workspace_bytes(), capacity);
        let limit = 64 * 1024;
        let context =
            JsonOperationContext::new(std::sync::Arc::new(crate::json::WorkLedger::with_limits(
                limit,
                crate::json::WorkLedger::CATALOG_LIMIT_BYTES,
                limit,
            )));
        let (result, scans) = super::super::source_probe::measure(None, || {
            Acceptance::prepare(
                &db,
                "notes",
                &dataset,
                ResourceProfile::Balanced,
                &modes,
                &context,
            )
        });
        assert!(window_capacity_error(&result.err().unwrap()));
        assert_eq!(scans.opened, [1, 0, 0]);
        assert_eq!(context.available_workspace_bytes(), limit);
        drop((db, dataset, input_work));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn stop_word_only_acceptance_keeps_source_validation_but_skips_token_scratch() {
        let directory =
            super::super::tests::text_only_benchmark_dataset("source-stop-word-scratch");
        let part = "P".repeat(131_071);
        let rows = format!("{{\"document_external_id\":\"doc\",\"chunk_external_id\":\"doc#1\",\"ordinal\":0,\"text\":\"{part}_{part}\",\"metadata\":{{}}}}\n");
        fs::write(directory.join("chunks.jsonl"), &rows).unwrap();
        let manifest_path = directory.join("dataset_manifest.json");
        let mut manifest =
            crate::json::parse(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .get_mut("hashes")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "chunks_sha256".into(),
                JsonValue::String(crate::hash::sha256_hex(rows.as_bytes())),
            );
        fs::write(&manifest_path, manifest.to_compact_string()).unwrap();
        let setup = JsonOperationContext::standalone();
        let (mut dataset, input_work) =
            BenchmarkDataset::load_and_validate_in(&directory, &setup).unwrap();
        let db =
            Database::create(directory.join("data.coffer"), DatabaseConfig::default()).unwrap();
        let manifest = write_benchmark_import_manifest(&dataset, "notes", &setup).unwrap();
        db.import_manifest(manifest.path(), OnConflict::Error)
            .unwrap();
        drop(manifest);
        let mut source_work = setup.catalog_reservation();
        let mut documents = Catalog::default();
        documents.documents = read_sqlite_baseline_documents(
            &dataset.documents_path,
            File::open(&dataset.documents_path).unwrap(),
            &setup,
            &mut source_work,
        )
        .unwrap();
        let query = BenchmarkQuery {
            query_id: "stop-word".into(),
            text: Some("THE and".into()),
            filter: JsonValue::Null,
            embedding: None,
            top_k: 1,
            relevant_chunk_ids: Default::default(),
        };
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let (result, keys) = crate::test_alloc::key_allocations(131_071, || {
            db.with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
        });
        let (text, vector) = result.unwrap();
        assert!(text.as_slice().is_empty());
        assert!(vector.as_slice().is_empty());
        assert_eq!(keys.allocated, 0);
        drop((text, vector));
        assert_eq!(context.available_workspace_bytes(), capacity);
        let expected_hash = std::mem::replace(&mut dataset.manifest.chunks_sha256, "0".repeat(64));
        let error = db
            .with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        assert_eq!(context.available_workspace_bytes(), capacity);
        dataset.manifest.chunks_sha256 = expected_hash;
        documents.documents.clear();
        let error = db
            .with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            error.message(),
            "source chunk references an absent document"
        );
        assert_eq!(context.available_workspace_bytes(), capacity);
        drop((documents, source_work, dataset, input_work, db));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn source_term_counts_preserve_fixed_tf_df_and_partial_failure_is_discarded() {
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let terms =
            crate::tokenizer::query_terms_in("Beta alpha ALPHA fooFoo missing", &context).unwrap();
        assert_eq!(
            terms.iter().collect::<Vec<_>>(),
            ["alpha", "beta", "foo", "foofoo", "missing"]
        );
        let mut counts = vec![0; terms.len()];
        let mut df = vec![0; terms.len()];
        let rows = [
            ("ALPHA alpha beta fooFoo", 5, [2, 1, 1, 1, 0]),
            ("beta", 1, [0, 1, 0, 0, 0]),
            ("the and", 0, [0, 0, 0, 0, 0]),
        ];
        let mut total_length = 0;
        for (text, length, expected) in rows {
            assert_eq!(
                term_counts_in(text, &terms, &mut counts, &context).unwrap(),
                length
            );
            assert_eq!(counts, expected);
            total_length += length.max(1);
            for (df, &tf) in df.iter_mut().zip(&counts) {
                *df += usize::from(tf != 0);
            }
        }
        assert_eq!(df, [1, 2, 1, 1, 0]);
        assert_eq!(total_length, 7);
        let available = context.available_workspace_bytes();
        let mut held = context.workspace_reservation();
        held.grow_to(available, available, "operation_scratch")
            .unwrap();
        let error = term_counts_in("alpha PROJECT", &terms, &mut counts, &context).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
        assert_eq!(counts, [1, 0, 0, 0, 0]);
        held.retain(0);
        assert_eq!(
            term_counts_in("beta", &terms, &mut counts, &context).unwrap(),
            1
        );
        assert_eq!(counts, [0, 1, 0, 0, 0]);
        drop(terms);
        assert_eq!(context.available_workspace_bytes(), capacity);
    }

    #[test]
    fn source_acceptance_admits_frequency_df_and_idf_arrays_before_their_allocation() {
        use crate::json::WorkLedger;
        let directory = super::super::tests::text_only_benchmark_dataset("source-array-admission");
        let setup = JsonOperationContext::standalone();
        let (dataset, input_work) =
            BenchmarkDataset::load_and_validate_in(&directory, &setup).unwrap();
        let db =
            Database::create(directory.join("data.coffer"), DatabaseConfig::default()).unwrap();
        let manifest = write_benchmark_import_manifest(&dataset, "notes", &setup).unwrap();
        db.import_manifest(manifest.path(), OnConflict::Error)
            .unwrap();
        drop(manifest);
        let mut source_work = setup.catalog_reservation();
        let mut documents = Catalog::default();
        documents.documents = read_sqlite_baseline_documents(
            &dataset.documents_path,
            File::open(&dataset.documents_path).unwrap(),
            &setup,
            &mut source_work,
        )
        .unwrap();
        let count = 10_000usize;
        let text = (0..count)
            .map(|index| format!("unique{index} "))
            .collect::<String>();
        let mut query = BenchmarkQuery {
            query_id: "array-quota".into(),
            text: Some(text),
            filter: JsonValue::Null,
            embedding: None,
            top_k: 1,
            relevant_chunk_ids: Default::default(),
        };
        let nodes = count.div_ceil(5)
            * (11 * std::mem::size_of::<(std::borrow::Cow<'_, str>, usize)>()
                + 16 * std::mem::size_of::<usize>());
        let array = count * std::mem::size_of::<usize>();
        for parent in [false, true] {
            for arrays in 1usize..=3 {
                let requested = nodes
                    + arrays.min(2) * array
                    + usize::from(arrays == 3) * count * std::mem::size_of::<f64>();
                let limit = requested - 1;
                let context =
                    JsonOperationContext::new(std::sync::Arc::new(WorkLedger::with_limits(
                        if parent { 2 * requested } else { limit },
                        0,
                        if parent { limit } else { 2 * requested },
                    )));
                let error = db
                    .with_read_snapshot(|catalog, _| {
                        source_candidates(
                            &db,
                            catalog,
                            &catalog.collection_by_name("notes")?.collection_id,
                            &documents,
                            &dataset,
                            &query,
                            ResourceProfile::Balanced,
                            &context,
                        )
                    })
                    .err()
                    .unwrap();
                assert_eq!(
                    error.code(),
                    ErrorCode::ResourceExhausted,
                    "array {arrays}: {error}"
                );
                assert_eq!(
                    error
                        .detail()
                        .iter()
                        .find(|item| item.0 == "limit_bytes")
                        .map(|item| item.1.as_str()),
                    Some(limit.to_string().as_str())
                );
                assert_eq!(
                    error
                        .detail()
                        .iter()
                        .find(|item| item.0 == "requested_bytes")
                        .map(|item| item.1.as_str()),
                    Some(requested.to_string().as_str())
                );
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
        query.text = Some("benchmark query".into());
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let (text, vector) = db
            .with_read_snapshot(|catalog, _| {
                source_candidates(
                    &db,
                    catalog,
                    &catalog.collection_by_name("notes")?.collection_id,
                    &documents,
                    &dataset,
                    &query,
                    ResourceProfile::Balanced,
                    &context,
                )
            })
            .unwrap();
        assert_eq!(text.as_slice().len(), 2);
        assert!(vector.as_slice().is_empty());
        for hit in text.as_slice() {
            assert!((hit.score - 2.0 * 1.2f64.ln()).abs() < 1e-12);
        }
        drop((text, vector));
        assert_eq!(context.available_workspace_bytes(), capacity);
        drop((query, documents, source_work, dataset, input_work, db));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn verification_checks_empty_counts_rank_scores_and_nonfinite_values() {
        for seed in 0..100 {
            let expected = (0..seed % 9)
                .map(|i| ExpectedHit {
                    external_id: format!("chunk-{i}"),
                    score: 1.0 / (i + 1) as f64,
                })
                .collect::<Vec<_>>();
            let mut actual = expected
                .iter()
                .map(|hit| (hit.external_id.as_str(), hit.score))
                .collect::<Vec<_>>();
            verify_hits(&expected, actual.iter().copied()).unwrap();
            actual.push(("unexpected", 0.0));
            assert!(verify_hits(&expected, actual.iter().copied()).is_err());
            actual.pop();
            if !actual.is_empty() {
                let saved = actual[0];
                for score in [f64::NAN, f64::INFINITY, -f64::INFINITY, saved.1 + 1e-6] {
                    actual[0].1 = score;
                    assert!(verify_hits(&expected, actual.iter().copied()).is_err());
                }
                actual[0] = ("wrong", saved.1);
                assert!(verify_hits(&expected, actual.iter().copied()).is_err());
                actual[0] = saved;
                actual.pop();
                assert!(verify_hits(&expected, actual.iter().copied()).is_err());
            }
        }
    }

    #[test]
    fn bounded_text_candidates_keep_score_order_then_internal_id_ties() {
        let budget = 1024 * 1024;
        let mut shrink_cases = [false; 3];
        let mut output_cases = [false; 3];
        for seed in 0..100 {
            let ids = (0..37).map(|i| format!("chk_{i}")).collect::<Vec<_>>();
            let mut all = (0..37)
                .map(|i| TextCandidate {
                    id: &ids[i],
                    score: ((i * 17 + seed) % 11) as f64,
                    matches: seed % 10 != 0 && i % 3 != 0,
                })
                .collect::<Vec<_>>();
            let limit = 1 + seed % 53;
            let context = JsonOperationContext::new(std::sync::Arc::new(
                crate::json::WorkLedger::with_limits(budget, 0, budget),
            ));
            let mut work = context.workspace_reservation();
            let mut bounded = BinaryHeap::new();
            for hit in &all {
                let hit = TextCandidate {
                    id: hit.id,
                    score: hit.score,
                    matches: hit.matches,
                };
                if bounded.len() < limit {
                    crate::json::reserve_heap_slot(&mut bounded, &mut work, limit).unwrap();
                    bounded.push(hit);
                } else if hit < *bounded.peek().unwrap() {
                    *bounded.peek_mut().unwrap() = hit;
                }
            }
            all.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(b.id)));
            let expected = all
                .iter()
                .take(limit)
                .filter(|hit| hit.matches)
                .map(|hit| &hit.id)
                .collect::<Vec<_>>();
            let capacity = bounded.capacity();
            let count = bounded.len();
            let size = std::mem::size_of::<TextCandidate<'_>>();
            assert_eq!(work.reserved_bytes(), capacity * size);
            let mut blocker = context.workspace_reservation();
            let available = count * size - 1 + seed % 3;
            let blocked = budget - work.reserved_bytes() - available;
            blocker
                .grow_to(blocked, blocked, "operation_scratch")
                .unwrap();
            let changed = crate::json::try_shrink_heap(&mut bounded, &mut work).unwrap();
            assert_eq!(changed, capacity > count && seed % 3 != 0);
            shrink_cases[if capacity == count {
                0
            } else if changed {
                1
            } else {
                2
            }] = true;
            assert_eq!(bounded.capacity(), if changed { count } else { capacity });
            assert_eq!(work.reserved_bytes(), bounded.capacity() * size);
            let actual = bounded
                .into_sorted_vec()
                .into_iter()
                .filter(|hit| hit.matches)
                .map(|hit| hit.id)
                .collect::<Vec<_>>();
            assert_eq!(actual.iter().collect::<Vec<_>>(), expected);
            output_cases[if actual.is_empty() {
                0
            } else if actual.len() == count {
                1
            } else {
                2
            }] = true;
            drop(work);
            drop(blocker);
            assert_eq!(context.available_workspace_bytes(), budget);
        }
        assert_eq!(shrink_cases, [true; 3]);
        assert_eq!(output_cases, [true; 3]);
    }
}
