use super::{contribution, Bm25Hits, MatchedTerm, TextIndex, TextSearchOptions};
use crate::json::{JsonOperationContext, ReservedVec};
use crate::storage::{
    catalog_id_number, catalog_id_text, IndexSegmentSource, Storage, TextSegmentEntry,
    TextSegmentVisitor,
};
use crate::{Error, ErrorCode, Result};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct ScoredChunk {
    // Numeric chunk ID in the checkpoint. Used to join postings and converted back
    // to the public ID string for tie breaking and returned hits.
    id: u64,
    // Indexed chunk length in tokens, retained while term postings are accumulated
    // and reproduced for the winning term explanations.
    length: usize,
    // Sum of raw BM25 contributions from the segment's matching term postings for
    // this chunk.
    score: f64,
}

// Equality here describes a rank key. The stored length participates in scoring
// earlier but has no place in the tie once two chunks have a final score and ID.
impl PartialEq for ScoredChunk {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ScoredChunk {}
impl PartialOrd for ScoredChunk {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredChunk {
    fn cmp(&self, other: &Self) -> Ordering {
        // A greater chunk wins. The bounded heap wraps these keys in Reverse so a
        // new contender can be compared with the least desirable survivor at its
        // root.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                // Numeric order differs from the frozen string tie-break across ID widths.
                // Use each canonical ID spelling in its own stack buffer. Once the
                // numeric suffix exceeds its minimum padded width, the lexically
                // earlier ID can correspond to the larger number.
                catalog_id_text(&mut [0; 24], b"chk_", other.id).cmp(catalog_id_text(
                    &mut [0; 24],
                    b"chk_",
                    self.id,
                ))
            })
    }
}

// Dictionary entries can have zero postings. Like absent query terms, they consume
// no column of the winner explanation matrix, so the default column value is
// meaningful only after testing df.
#[derive(Default)]
struct TermStats {
    // Count of chunks containing this query term according to its collection
    // segment. Zero leaves the term out of the explanation matrix.
    df: usize,
    // Dense column assigned only to query terms with positive document frequency.
    // Readers test df before indexing this column.
    column: usize,
}

impl TextIndex {
    pub(crate) fn search_from_segment(
        source: &IndexSegmentSource,
        storage: &Storage,
        collection_id: &str,
        query: &str,
        options: TextSearchOptions,
        context: &JsonOperationContext,
    ) -> Result<Option<Bm25Hits>> {
        // Unrestricted recall competes across chunks that score positively and
        // keeps the caller's cutoff. Passing None leaves the visitor free to build
        // a score row for every chunk in this collection.
        Self::search_selected_from_segment(
            source,
            storage,
            collection_id,
            query,
            options,
            None,
            context,
        )
    }

    pub(crate) fn score_ids_from_segment(
        source: &IndexSegmentSource,
        storage: &Storage,
        collection_id: &str,
        query: &str,
        ids: &[&str],
        context: &JsonOperationContext,
    ) -> Result<Option<Bm25Hits>> {
        // Score completion is keyed by a caller-supplied set of IDs. Give every
        // requested row room and request its explanation, including valid rows
        // whose query terms made no contribution.
        Self::search_selected_from_segment(
            source,
            storage,
            collection_id,
            query,
            TextSearchOptions::with_explanations(ids.len()),
            Some(ids),
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn search_selected_from_segment(
        source: &IndexSegmentSource,
        storage: &Storage,
        collection_id: &str,
        query: &str,
        options: TextSearchOptions,
        ids: Option<&[&str]>,
        context: &JsonOperationContext,
    ) -> Result<Option<Bm25Hits>> {
        // Reading the segment and scoring its contents draw from the same
        // operation budget. A zero minimum allows the reader to take only the
        // input buffer space left under that budget.
        let mut input = context.workspace_reservation();
        input.grow_to(0, 64 * 1024, "operation_scratch")?;
        loop {
            #[cfg(test)]
            tests::record_input(input.reserved_bytes());
            // Each attempt invokes a fresh segment visit, so any score arrays
            // built on the failed pass have dropped before the same input is tried
            // with a different buffer grant.
            match Self::search_selected_segment_with(
                query,
                options,
                ids,
                input.context(),
                |visit| source.visit_text(storage, collection_id, &input, visit),
            ) {
                Err(error)
                    if crate::json::is_memory_budget_error(&error)
                        && input.reserved_bytes() > 0 =>
                {
                    // The read buffer is optional. Release it before rerunning
                    // scoring so mandatory query arrays can use the same credit.
                    input.retain(0)
                }
                // Only a memory-budget failure with a nonzero read buffer takes
                // another pass. Once the zero-buffer attempt returns, its error or
                // source availability goes directly to the caller.
                result => return result,
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn search_segment_with(
        query: &str,
        options: TextSearchOptions,
        context: &JsonOperationContext,
        visit: impl FnMut(&mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<Option<Bm25Hits>> {
        Self::search_selected_segment_with(query, options, None, context, visit)
    }

    fn search_selected_segment_with(
        query: &str,
        options: TextSearchOptions,
        ids: Option<&[&str]>,
        context: &JsonOperationContext,
        mut visit: impl FnMut(&mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<Option<Bm25Hits>> {
        // A caller asking for no rows needs neither a source visit nor query-term
        // workspace. This successful empty result is available even when the
        // segment itself cannot serve other requests.
        if options.limit == 0 {
            return Ok(Some(Bm25Hits::empty(context)));
        }
        let terms = crate::tokenizer::query_terms_in(query, context)?;
        // Lexical recall needs a matching token to produce any hits. Score
        // completion still owes a row, potentially with a zero score, for every
        // supplied candidate ID.
        if terms.len() == 0 && ids.is_none() {
            return Ok(Some(Bm25Hits::empty(context)));
        }
        // Keep one statistics slot per query term, including terms missing from
        // the segment. Later passes address the slot by query vocabulary position
        // even when only some terms receive matrix columns.
        let mut stats = ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        for _ in 0..terms.len() {
            stats.push(TermStats::default());
        }
        let mut scores =
            ReservedVec::<ScoredChunk>::with_capacity(0, context, "operation_scratch")?;
        let selected_ids = ids
            .map(|ids| -> Result<_> {
                let mut numbers =
                    ReservedVec::with_capacity(ids.len(), context, "operation_scratch")?;
                // Use the numeric IDs stored in the segment for membership checks.
                // Their directory order can differ from the caller's formatted
                // strings.
                for id in ids {
                    numbers.push(catalog_id_number(id)?);
                }
                // Sort a private numeric copy. The caller's original slice can
                // still serve other retrieval branches, and its textual order need
                // not match the segment's numeric chunk order.
                numbers.as_mut_slice().sort_unstable();
                Ok(numbers)
            })
            .transpose()?;
        // The segment lists chunks before postings. Even an ID lookup needs the
        // lengths of all chunks to compute the same average length as a full search.
        // Keep the selected rows in numeric ID order for the postings lookup.
        let mut collection_chunks = 0;
        let mut total_length = 0usize;
        let mut average = 1.0;
        let mut current = None;
        if !visit(&mut |entry| {
            match entry {
                TextSegmentEntry::Header { chunks } => {
                    collection_chunks = chunks;
                    // Full recall can accumulate one score per chunk. For a
                    // completion request the same header gives the corpus
                    // population, but only the requested number of score slots
                    // needs workspace.
                    scores = ReservedVec::with_capacity(
                        ids.map_or(chunks, |ids| ids.len()),
                        context,
                        "operation_scratch",
                    )?;
                }
                TextSegmentEntry::Chunk { id, length } => {
                    // Accumulate every decoded chunk's token length before testing
                    // membership. Dropping unrequested rows from this sum would
                    // make a chosen chunk's score depend on the other requested
                    // IDs.
                    total_length = total_length.checked_add(length).ok_or_else(|| {
                        Error::new(ErrorCode::FileFormatError, "text token count overflow")
                    })?;
                    if selected_ids
                        .as_ref()
                        .map_or(true, |ids| ids.as_slice().binary_search(&id).is_ok())
                    {
                        // Seed selected chunks with zero. A candidate whose
                        // postings never match the query is still a valid
                        // completion row, including when the query vocabulary is
                        // empty.
                        scores.push(ScoredChunk {
                            id,
                            length,
                            score: 0.0,
                        });
                    }
                }
                TextSegmentEntry::Term {
                    text,
                    document_frequency,
                } => {
                    // Term headers follow all chunk lengths, so the collection
                    // average is complete before the first contribution. The empty
                    // collection uses one as its fallback normalization length.
                    let n = collection_chunks;
                    average = if n == 0 {
                        1.0
                    } else {
                        total_length as f64 / n as f64
                    };
                    // Replace the current posting-group state on every dictionary
                    // entry. An unrelated term must clear the preceding term's IDF
                    // or its postings would score as matches for that earlier
                    // term.
                    current = terms.position(text).map(|position| {
                        stats.as_mut_slice()[position].df = document_frequency;
                        let df = document_frequency as f64;
                        ((n as f64 - df + 0.5) / (df + 0.5) + 1.0).ln()
                    });
                }
                TextSegmentEntry::Posting {
                    chunk, frequency, ..
                } => {
                    if let Some(idf) = current {
                        // The chunk directory visits IDs in numeric order. Pushing
                        // only wanted rows preserves that order, so postings can
                        // join directly to their accumulated scores by binary
                        // search.
                        let position = match scores
                            .as_slice()
                            .binary_search_by_key(&chunk, |entry| entry.id)
                        {
                            Ok(position) => position,
                            // A selected-ID lookup deliberately omitted the other
                            // chunks from the score table. Their postings still
                            // belong to the term's collection-wide statistics. In a
                            // full scan every posting needs a score row.
                            Err(_) if ids.is_some() => return Ok(()),
                            Err(_) => {
                                return Err(Error::new(
                                    ErrorCode::IndexCorruption,
                                    "text posting has no chunk",
                                ))
                            }
                        };
                        // Use the score row's chunk length and the collection-wide
                        // average with this term's IDF. The same inputs are kept
                        // for explaining each winning contribution after
                        // selection.
                        let score = &mut scores.as_mut_slice()[position];
                        score.score += contribution(frequency, score.length, average, idf);
                    }
                }
            }
            Ok(())
        })? {
            // The visitor's false result describes the whole segment. Even if
            // earlier closures filled some score rows, none of them can be
            // returned from an incomplete source.
            return Ok(None);
        }

        // A missing requested ID makes the segment unusable for score completion.
        // Let the caller try its other source. In contrast, scoring the available
        // rows and selecting no matches below is a successful empty result.
        // The selected table stores one row per encountered chunk. A repeated
        // requested ID, just like an absent one, prevents this row count from
        // satisfying the completion request.
        if ids.is_some_and(|ids| scores.as_slice().len() != ids.len()) {
            return Ok(None);
        }
        // Empty query still reports the collection's true lengths and zero scores.
        // A segment may contain chunks but no term headers. Derive the final
        // average from the accumulated lengths here as well, independent of
        // whether scoring encountered a posting group.
        if collection_chunks != 0 {
            average = total_length as f64 / collection_chunks as f64;
        }
        let chunks = collection_chunks;
        // Completion allows zero scores into selection. Ordinary recall admits
        // only positive scores so an unrelated chunk cannot fill an otherwise
        // unused result slot.
        let selected =
            select_candidates_with_zeros(scores.as_slice(), options.limit, ids.is_some(), context)?;
        // The bounded selection owns copies of its score rows. Drop the larger
        // accumulator before reserving the lookup and winner-by-term matrix, which
        // can compete for the same budget.
        drop(scores);
        if selected.as_slice().is_empty() {
            return Ok(Some(Bm25Hits::empty(context)));
        }

        // An absent query term has no explanation entries in any winning row.
        // Assign columns only to present terms, keeping their order from the sorted
        // query vocabulary.
        let mut columns = 0usize;
        // The default column value is also a legitimate first column. Document
        // frequency determines whether a slot received a column before any lookup
        // may use that value.
        for term in stats.as_mut_slice().iter_mut().filter(|term| term.df != 0) {
            term.column = columns;
            columns += 1;
        }
        // Only the selected rows need term-by-term explanations. Revisit the
        // segment for their frequencies after ranking, avoiding a table of term
        // counts for every chunk. Score-only queries finish without that pass.
        let frequencies = if options.explanations {
            let mut lookup = ReservedVec::with_capacity(
                selected.as_slice().len(),
                context,
                "operation_scratch",
            )?;
            // Winners are in score order, while postings refer to numeric chunk
            // IDs. Sort a separate ID-to-row mapping for lookup so filling the
            // frequency matrix does not disturb the output ranking.
            for (position, entry) in selected.as_slice().iter().enumerate() {
                lookup.push((entry.0.id, position));
            }
            lookup.as_mut_slice().sort_unstable_by_key(|entry| entry.0);
            #[cfg(test)]
            tests::record_matrix(selected.as_slice().len(), columns);
            // With no present query terms the matrix can have rows and zero
            // columns. Its empty backing array is valid because extraction checks
            // document frequency before indexing a cell.
            let mut frequencies = frequency_matrix(selected.as_slice().len(), columns, context)?;
            let mut current = None;
            // The explanation pass reads the segment a second time. If that visit
            // cannot finish against a valid segment, discard the first pass's
            // winners too. Returning them with missing frequencies would make the
            // scores and their explanations disagree.
            if !visit(&mut |entry| {
                match entry {
                    TextSegmentEntry::Term { text, .. } => {
                        // The column belongs to the current term group only.
                        // Replace it with None for an irrelevant dictionary entry
                        // before visiting the postings that follow.
                        current = terms
                            .position(text)
                            .filter(|&position| stats.as_slice()[position].df != 0)
                            .map(|position| stats.as_slice()[position].column);
                    }
                    TextSegmentEntry::Posting {
                        chunk, frequency, ..
                    } => {
                        if let Some(column) = current {
                            if let Ok(position) = lookup
                                .as_slice()
                                .binary_search_by_key(&chunk, |entry| entry.0)
                            {
                                // Map the matched numeric ID back to the winner's
                                // original row. The lookup's own binary-search
                                // position is in ID order and cannot address the
                                // score-ranked matrix directly.
                                let row = lookup.as_slice()[position].1;
                                frequencies.as_mut_slice()[row * columns + column] = frequency;
                            }
                        }
                    }
                    _ => {}
                }
                Ok(())
            })? {
                return Ok(None);
            }
            Some(frequencies)
        } else {
            None
        };

        // The lookup, score rows and frequency cells hold intermediate data. The
        // returned hits retain only the selected IDs, scores and individually
        // matched term details, with their own workspace owner.
        let mut hits = Bm25Hits::with_capacity(selected.as_slice().len(), context)?;
        for (row, entry) in selected.as_slice().iter().enumerate() {
            let chunk = &entry.0;
            let matched = terms.iter().enumerate().filter_map(|(position, term)| {
                // A score-only result has no matrix. Its matched-term iterator
                // stays empty while the hit below still receives the score
                // computed by the first pass.
                let frequencies = frequencies.as_ref()?;
                let stats = &stats.as_slice()[position];
                if stats.df == 0 {
                    return None;
                }
                // A term may be present in this collection without appearing in
                // this particular winner. Its zero cell contributes neither an
                // explanation entry nor an additional term score.
                let frequency = frequencies.as_slice()[row * columns + stats.column];
                if frequency == 0 {
                    return None;
                }
                // Reconstruct each matched term from the same collection
                // population, document frequency and chunk length used for its raw
                // score. These details remain per term even when several
                // contribute to one hit.
                let n = chunks as f64;
                let df = stats.df as f64;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                Some(MatchedTerm {
                    document_frequency: stats.df,
                    chunk_length: chunk.length,
                    avg_chunk_length: average,
                    term,
                    frequency,
                    bm25_contribution: contribution(frequency, chunk.length, average, idf),
                })
            });
            hits.push(
                // Decode the numeric ID to a stack buffer only for this call. The
                // result builder takes an owned copy before the temporary buffer
                // expires.
                catalog_id_text(&mut [0; 24], b"chk_", chunk.id),
                chunk.score,
                matched,
            )?;
        }
        // Recall returns ranked rows. Selected-ID scoring instead returns a lookup
        // table for fusion, whose binary search expects lexical chunk-ID order.
        if ids.is_some() {
            hits.sort_by_id();
        }
        Ok(Some(hits))
    }
}

#[cfg(test)]
fn select_candidates(
    scores: &[ScoredChunk],
    limit: usize,
    context: &JsonOperationContext,
) -> Result<ReservedVec<Reverse<ScoredChunk>>> {
    select_candidates_with_zeros(scores, limit, false, context)
}

fn select_candidates_with_zeros(
    scores: &[ScoredChunk],
    limit: usize,
    include_zero: bool,
    context: &JsonOperationContext,
) -> Result<ReservedVec<Reverse<ScoredChunk>>> {
    // The predicate is shared between the count and the heap pass. A zero-score
    // candidate survives in selected-ID completion only, where an explicit zero is
    // part of the result for a requested ID.
    let matches = || {
        scores
            .iter()
            .filter(|chunk| include_zero || chunk.score > 0.0)
            .copied()
    };
    // Count eligible scores first so the heap reserves exactly what can survive. A
    // selected-ID lookup can retain real zeros, whereas ordinary recall has no
    // lexical hit to emit for them.
    let count = matches().count().min(limit);
    // Carry the byte calculation in the ledger's wide representation. The heap
    // takes the same number of slots that will survive ranking, without reserving
    // one for every row in the score table.
    let bytes = count as u128 * std::mem::size_of::<ScoredChunk>() as u128;
    let mut work = context.workspace_reservation();
    work.grow_to_wide(bytes, bytes, "operation_scratch")?;
    let mut heap = BinaryHeap::new();
    heap.try_reserve_exact(count)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    // Reject an allocator overgrant before adding hits. The array that survives in
    // the result must remain within the byte count reserved above, even if the
    // heap stores fewer rows later.
    if heap.capacity() != count {
        return Err(std::io::Error::other("text candidate capacity exceeds admission").into());
    }
    // Use Reverse so the least valuable retained score stays at the root. Ties are
    // resolved by ScoredChunk ordering, so a candidate at the boundary has the
    // same chance to survive as in a full deterministic sort.
    for chunk in matches() {
        if heap.len() < count {
            heap.push(Reverse(chunk));
        } else if heap.peek().is_some_and(|worst| chunk > worst.0) {
            *heap.peek_mut().unwrap() = Reverse(chunk);
        }
    }
    // Return the sorted vector using the heap's allocation. The reservation must
    // follow that allocation until the caller finishes materializing owned
    // matches.
    Ok(ReservedVec::from_reserved(heap.into_sorted_vec(), work))
}

fn frequency_matrix(
    rows: usize,
    columns: usize,
    context: &JsonOperationContext,
) -> Result<ReservedVec<usize>> {
    // Size the dense frequency array with checked arithmetic before converting to
    // a host allocation length. Each selected ID receives one cell per query term
    // with a positive document frequency, including pairs absent from that row's
    // postings.
    let count = rows as u128 * columns as u128;
    let bytes = count
        .checked_mul(std::mem::size_of::<usize>() as u128)
        .ok_or_else(|| std::io::Error::other("text frequency array size overflow"))?;
    let count = match usize::try_from(count) {
        Ok(count) => count,
        // Ask the ledger about the full byte requirement before reporting an
        // address-space error, preserving the resource error and its details.
        Err(_) => {
            context
                .workspace_reservation()
                .grow_to_wide(bytes, bytes, "operation_scratch")?;
            return Err(std::io::Error::other("text frequency array exceeds address space").into());
        }
    };
    let mut values = ReservedVec::with_capacity(count, context, "operation_scratch")?;
    // Initialize the entire matrix to zero. Posting visits fill only observed
    // pairs, so a missing term for a candidate retains its zero count without
    // another sparse lookup.
    for _ in 0..count {
        values.push(0);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use crate::storage::{
        catalog_id_number, TextIndexChunkEntry, TextIndexPostingEntry, TextIndexSegmentPayload,
        TextIndexTermEntry,
    };
    use std::sync::Arc;

    type Trace = (Vec<usize>, Vec<(usize, usize)>);
    #[derive(Default)]
    struct Capture {
        inputs: [usize; 3],
        input_count: usize,
        matrices: [(usize, usize); 3],
        matrix_count: usize,
        overflow: bool,
    }
    thread_local! {
        static TRACE: std::cell::RefCell<Option<Capture>> = const { std::cell::RefCell::new(None) };
    }

    pub(super) fn record_input(bytes: usize) {
        TRACE.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                if let Some(slot) = trace.inputs.get_mut(trace.input_count) {
                    *slot = bytes;
                    trace.input_count += 1;
                } else {
                    trace.overflow = true;
                }
            }
        });
    }

    pub(super) fn record_matrix(rows: usize, columns: usize) {
        TRACE.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                if let Some(slot) = trace.matrices.get_mut(trace.matrix_count) {
                    *slot = (rows, columns);
                    trace.matrix_count += 1;
                } else {
                    trace.overflow = true;
                }
            }
        });
    }

    impl TextIndex {
        pub(crate) fn with_segment_search_trace<T>(run: impl FnOnce() -> T) -> (T, Trace) {
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    TRACE.with(|trace| *trace.borrow_mut() = None);
                }
            }
            TRACE.with(|trace| {
                let mut trace = trace.borrow_mut();
                assert!(trace.is_none());
                *trace = Some(Capture::default());
            });
            let _reset = Reset;
            let result = run();
            let trace = TRACE.with(|trace| trace.borrow_mut().take().unwrap());
            assert!(!trace.overflow);
            (
                result,
                (
                    trace.inputs[..trace.input_count].to_vec(),
                    trace.matrices[..trace.matrix_count].to_vec(),
                ),
            )
        }
    }

    fn segment(seed: usize) -> TextIndexSegmentPayload {
        let ids = [
            1,
            2,
            10,
            9_999_999_999_999_999,
            10_000_000_000_000_000,
            u64::MAX,
        ];
        TextIndexSegmentPayload {
            collection_id: "col_1".into(),
            covered_commit_sequence_no: 7,
            chunks: ids
                .iter()
                .enumerate()
                .map(|(position, id)| TextIndexChunkEntry {
                    chunk_id: format!("chk_{id:016}"),
                    document_id: "doc_1".into(),
                    token_count: if position == 3 {
                        0
                    } else {
                        3 + seed % 11 + position
                    },
                })
                .collect(),
            terms: ["alpha", "beta", "empty", "gamma"]
                .iter()
                .enumerate()
                .map(|(term, text)| TextIndexTermEntry {
                    term: (*text).into(),
                    postings: ids
                        .iter()
                        .enumerate()
                        .filter(|(position, _)| {
                            term != 2 && *position != 3 && (position + term + seed) % 3 != 0
                        })
                        .map(|(position, id)| TextIndexPostingEntry {
                            chunk_id: format!("chk_{id:016}"),
                            term_frequency: 1 + (seed + position + term) % 3,
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    fn emit(segment: &TextIndexSegmentPayload, visit: &mut TextSegmentVisitor<'_>) -> Result<bool> {
        visit(TextSegmentEntry::Header {
            chunks: segment.chunks.len(),
        })?;
        for chunk in &segment.chunks {
            visit(TextSegmentEntry::Chunk {
                id: catalog_id_number(&chunk.chunk_id)?,
                length: chunk.token_count.max(1),
            })?;
        }
        for term in &segment.terms {
            visit(TextSegmentEntry::Term {
                text: &term.term,
                document_frequency: term.postings.len(),
            })?;
            for posting in &term.postings {
                visit(TextSegmentEntry::Posting {
                    term: &term.term,
                    chunk: catalog_id_number(&posting.chunk_id)?,
                    frequency: posting.term_frequency,
                })?;
            }
        }
        Ok(true)
    }

    #[test]
    fn persistent_text_scores_and_explanations_match_cached_oracle_across_id_widths() {
        for seed in 0..100 {
            let segment = segment(seed);
            let index = TextIndex::try_build_from_segments(vec![segment.clone()]).unwrap();
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            for query in [
                "alpha beta gamma",
                "ALPHA alpha",
                "alpha missing",
                "the and",
            ] {
                for explanations in [false, true] {
                    let options = TextSearchOptions {
                        limit: 1 + seed % 9,
                        explanations,
                    };
                    let mut calls = 0;
                    let actual =
                        TextIndex::search_segment_with(query, options, &context, |visit| {
                            calls += 1;
                            emit(&segment, visit)
                        })
                        .unwrap()
                        .unwrap();
                    let expected = index.search("col_1", query, options, &context).unwrap();
                    assert_eq!(
                        calls,
                        if query == "the and" {
                            0
                        } else {
                            1 + usize::from(explanations)
                        }
                    );
                    assert_eq!(
                        format!("{:?}", actual.as_slice()),
                        format!("{:?}", expected.as_slice())
                    );
                    for (actual, expected) in actual.iter().zip(expected.iter()) {
                        assert_eq!(actual.score.to_bits(), expected.score.to_bits());
                        for (actual, expected) in
                            actual.matched_terms.iter().zip(&expected.matched_terms)
                        {
                            assert_eq!(
                                actual.avg_chunk_length.to_bits(),
                                expected.avg_chunk_length.to_bits()
                            );
                            assert_eq!(
                                actual.bm25_contribution.to_bits(),
                                expected.bm25_contribution.to_bits()
                            );
                        }
                    }
                }
            }
            assert_eq!(context.available_workspace_bytes(), available);
        }
    }

    #[test]
    fn empty_terms_and_zero_limit_do_not_visit_or_admit_segment_arrays() {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        for (query, limit) in [("the and !!!", usize::MAX), ("actual term", 0)] {
            let hits = TextIndex::search_segment_with(
                query,
                TextSearchOptions::with_explanations(limit),
                &context,
                |_| panic!("no segment may be visited"),
            )
            .unwrap()
            .unwrap();
            assert!(hits.is_empty());
            assert_eq!(context.available_workspace_bytes(), 0);
        }
    }

    #[test]
    fn second_pass_failure_discards_all_provisional_candidates_and_credit() {
        let segment = segment(1);
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        for fault in 0..3 {
            let mut calls = 0;
            let result = TextIndex::search_segment_with(
                "alpha beta",
                TextSearchOptions::with_explanations(4),
                &context,
                |visit| {
                    calls += 1;
                    let valid = emit(&segment, visit)?;
                    if calls != 2 {
                        return Ok(valid);
                    }
                    match fault {
                        0 => Ok(false),
                        1 => Err(std::io::Error::other("second pass read failure").into()),
                        _ => Err(Error::resource_exhausted(
                            "memory",
                            "operation_scratch",
                            7,
                            8,
                        )),
                    }
                },
            );
            assert_eq!(calls, 2);
            match fault {
                0 => assert!(result.unwrap().is_none()),
                1 => assert_eq!(
                    result.unwrap_err().to_string(),
                    "io_error: second pass read failure"
                ),
                _ => {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), "8".into())));
                }
            }
            assert_eq!(context.available_workspace_bytes(), available);
        }
    }

    #[test]
    fn frequency_matrix_reserves_actual_dimensions_and_handles_wide_counts() {
        for rows in 1..=100 {
            let columns = rows % 11 + 1;
            let needed = rows * columns * std::mem::size_of::<usize>();
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (needed as isize + delta) as usize;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { needed * 2 } else { limit },
                        0,
                        if parent { limit } else { needed * 2 },
                    )));
                    let result = frequency_matrix(rows, columns, &context);
                    if delta < 0 {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".into()),
                            ("operation", "operation_scratch".into()),
                            ("limit_bytes", limit.to_string()),
                            ("requested_bytes", needed.to_string()),
                        ] {
                            assert!(error.detail().contains(&(key.into(), value)));
                        }
                    } else {
                        let values = result.unwrap();
                        assert_eq!(values.reserved_bytes(), needed);
                        assert_eq!(values.as_slice(), vec![0; rows * columns]);
                        assert_eq!(context.available_workspace_bytes(), limit - needed);
                    }
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
        let context = JsonOperationContext::standalone();
        assert_eq!(
            frequency_matrix(usize::MAX, usize::MAX, &context)
                .unwrap_err()
                .code(),
            ErrorCode::IoError
        );
        assert!(frequency_matrix(0, usize::MAX, &context)
            .unwrap()
            .as_slice()
            .is_empty());
    }

    #[test]
    fn candidate_selection_failure_returns_score_array_and_all_credit() {
        let mut segment = segment(1);
        segment.terms[0].postings = segment
            .chunks
            .iter()
            .map(|chunk| TextIndexPostingEntry {
                chunk_id: chunk.chunk_id.clone(),
                term_frequency: 1,
            })
            .collect();
        let nodes = crate::json::btree_workspace::<std::borrow::Cow<'_, str>, usize>(1);
        let base = nodes
            + std::mem::size_of::<TermStats>()
            + segment.chunks.len() * std::mem::size_of::<ScoredChunk>();
        let required = base + 5 * std::mem::size_of::<ScoredChunk>();
        for parent in [false, true] {
            let limit = required - 1;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                if parent { 2 * limit } else { limit },
                0,
                if parent { limit } else { 2 * limit },
            )));
            let mut calls = 0;
            let error = TextIndex::search_segment_with(
                "alpha",
                TextSearchOptions::scores_only(5),
                &context,
                |visit| {
                    calls += 1;
                    emit(&segment, visit)
                },
            )
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
            assert!(error
                .detail()
                .contains(&("limit_bytes".into(), limit.to_string())));
            assert!(error
                .detail()
                .contains(&("requested_bytes".into(), required.to_string())));
            assert_eq!(context.available_workspace_bytes(), limit);
        }
    }

    #[test]
    fn selection_admits_only_actual_matches_once_at_both_budget_boundaries() {
        for count in 0..100usize {
            let scores = (0..count)
                .map(|id| ScoredChunk {
                    id: id as u64,
                    length: 1,
                    score: if id % 3 == 0 { 0.0 } else { 1.0 },
                })
                .collect::<Vec<_>>();
            for limit in [0, 1, 17, usize::MAX] {
                let count = scores
                    .iter()
                    .filter(|chunk| chunk.score > 0.0)
                    .count()
                    .min(limit);
                let needed = count * std::mem::size_of::<ScoredChunk>();
                for parent in [false, true] {
                    for available in [needed.saturating_sub(1), needed, needed + 1] {
                        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                            if parent { needed * 2 + 1 } else { available },
                            0,
                            if parent { available } else { needed * 2 + 1 },
                        )));
                        let result = select_candidates(&scores, limit, &context);
                        if available < needed {
                            let error = result.unwrap_err();
                            assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                            assert!(error
                                .detail()
                                .contains(&("requested_bytes".into(), needed.to_string())));
                            assert!(error
                                .detail()
                                .contains(&("limit_bytes".into(), available.to_string())));
                        } else {
                            let selected = result.unwrap();
                            assert_eq!(selected.as_slice().len(), count);
                            assert_eq!(selected.reserved_bytes(), needed);
                            assert_eq!(context.available_workspace_bytes(), available - needed);
                        }
                        assert_eq!(context.available_workspace_bytes(), available);
                    }
                }
            }
        }
    }

    #[test]
    fn equal_scores_use_string_ids_at_the_decimal_width_boundary() {
        let ids = [9_999_999_999_999_999u64, 10_000_000_000_000_000, u64::MAX];
        let segment = TextIndexSegmentPayload {
            collection_id: "col_1".into(),
            covered_commit_sequence_no: 7,
            chunks: ids
                .iter()
                .map(|id| TextIndexChunkEntry {
                    chunk_id: format!("chk_{id:016}"),
                    document_id: "doc_1".into(),
                    token_count: 3,
                })
                .collect(),
            terms: vec![TextIndexTermEntry {
                term: "alpha".into(),
                postings: ids
                    .iter()
                    .map(|id| TextIndexPostingEntry {
                        chunk_id: format!("chk_{id:016}"),
                        term_frequency: 1,
                    })
                    .collect(),
            }],
        };
        let context = JsonOperationContext::standalone();
        let expected = [ids[1], ids[2], ids[0]].map(|id| format!("chk_{id:016}"));
        for limit in 1..=3 {
            for explanations in [false, true] {
                let hits = TextIndex::search_segment_with(
                    "alpha",
                    TextSearchOptions {
                        limit,
                        explanations,
                    },
                    &context,
                    |visit| emit(&segment, visit),
                )
                .unwrap()
                .unwrap();
                assert_eq!(
                    hits.iter()
                        .map(|hit| hit.chunk_id.as_str())
                        .collect::<Vec<_>>(),
                    expected[..limit]
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                );
                assert!(hits
                    .as_slice()
                    .windows(2)
                    .all(|pair| pair[0].score.to_bits() == pair[1].score.to_bits()));
            }
        }
    }
    #[test]
    fn selected_scores_preserve_full_collection_statistics_zero_matches_and_id_widths() {
        for seed in 0..100 {
            let segment = segment(seed);
            let index = TextIndex::try_build_from_segments(vec![segment.clone()]).unwrap();
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let mut ids = segment
                .chunks
                .iter()
                .enumerate()
                .filter(|(i, _)| (*i + seed) % 2 == 0 || *i == 3)
                .map(|(_, row)| row.chunk_id.as_str())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            for query in [
                "alpha beta gamma",
                "ALPHA alpha missing",
                "the and",
                "missing",
            ] {
                let mut expected = index.search_reference("col_1", query, usize::MAX);
                expected.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
                let resident = index.score_ids("col_1", query, &ids, &context).unwrap();
                let persisted = TextIndex::search_selected_segment_with(
                    query,
                    TextSearchOptions::with_explanations(ids.len()),
                    Some(&ids),
                    &context,
                    |visit| emit(&segment, visit),
                )
                .unwrap()
                .unwrap();
                assert_eq!(resident.len(), ids.len());
                assert_eq!(persisted.len(), ids.len());
                for ((a, b), id) in resident.iter().zip(persisted.iter()).zip(&ids) {
                    let expected = expected
                        .binary_search_by(|row| row.chunk_id.as_str().cmp(id))
                        .map(|i| expected[i].clone())
                        .unwrap_or_else(|_| super::super::Bm25Hit {
                            chunk_id: (*id).into(),
                            ..Default::default()
                        });
                    super::super::tests::assert_same_hits(
                        std::slice::from_ref(a),
                        std::slice::from_ref(&expected),
                    );
                    super::super::tests::assert_same_hits(
                        std::slice::from_ref(b),
                        std::slice::from_ref(&expected),
                    );
                }
                drop((resident, persisted));
                assert_eq!(context.available_workspace_bytes(), available);
            }
            let absent = ["chk_0000000000000003"];
            let unavailable = TextIndex::search_selected_segment_with(
                "alpha",
                TextSearchOptions::with_explanations(1),
                Some(&absent),
                &context,
                |visit| emit(&segment, visit),
            )
            .unwrap();
            assert!(unavailable.is_none());
            assert_eq!(context.available_workspace_bytes(), available);
        }
    }
}
