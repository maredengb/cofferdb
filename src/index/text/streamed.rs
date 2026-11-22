use super::{contribution, Bm25Hits, MatchedTerm, TextCandidates, TextIndex, TextSearchOptions};
use crate::catalog::{VisibleChunk, VisibleSnapshot};
use crate::error::Result;
use crate::json::{JsonOperationContext, ReservedVec};
use crate::tokenizer::QueryTerms;
#[cfg(test)]
use std::collections::BTreeSet;

#[derive(Default)]
struct TermCounts {
    // Occurrences of this query term in the chunk currently being tokenized. Reset
    // for each chunk in both passes.
    frequency: usize,
    // Number of collection chunks containing this term at least once, accumulated
    // during the first pass and reused throughout scoring.
    document_frequency: usize,
}

// Both the statistics pass and the scoring pass call this same tokenizer. A change
// in what counts as a token can alter document length as well as term frequency,
// even when the extra token is absent from the query.
fn count_terms(
    text: &str,
    terms: &QueryTerms<'_>,
    counts: &mut [TermCounts],
    context: &JsonOperationContext,
) -> Result<usize> {
    // Keep the collection-level counts across chunks. Only the per-chunk
    // frequencies are reset here, so a repeated term can increase term frequency
    // several times while contributing only one document to its document frequency.
    for count in counts.iter_mut() {
        count.frequency = 0;
    }
    // Count all emitted tokens toward normalization. Only a member of the query
    // vocabulary needs an occurrence counter, so the per-chunk work array depends
    // on the query's size.
    let mut length = 0;
    crate::tokenizer::visit_tokens_in(text, context, |token| {
        length += 1;
        if let Some(index) = terms.position(token) {
            counts[index].frequency += 1;
        }
        Ok(())
    })?;
    // A chunk that produces no tokens still contributes one unit to the corpus
    // length and keeps a positive scoring length. This matches the length stored
    // in the text index.
    Ok(length.max(1))
}

impl TextIndex {
    // The snapshot supplies visibility while the callback supplies text for each
    // selected version. Holding one snapshot across both passes keeps their chunk
    // population consistent as corpus statistics become scores.
    pub(crate) fn search_from_snapshot<F, T>(
        snapshot: &VisibleSnapshot<'_>,
        collection_id: &str,
        query: &str,
        options: TextSearchOptions,
        context: &JsonOperationContext,
        load_text: F,
    ) -> Result<Bm25Hits>
    where
        F: FnMut(&VisibleChunk<'_>) -> Result<T>,
        T: AsRef<str>,
    {
        // Leave the ID filter absent for lexical recall. After corpus statistics
        // are fixed, the second pass may choose any matching chunk in this
        // collection for the bounded result.
        Self::search_snapshot_selected(
            snapshot,
            collection_id,
            query,
            options,
            None,
            context,
            load_text,
        )
    }

    pub(crate) fn score_ids_from_snapshot<F, T>(
        snapshot: &VisibleSnapshot<'_>,
        collection_id: &str,
        query: &str,
        ids: &[&str],
        context: &JsonOperationContext,
        load_text: F,
    ) -> Result<Bm25Hits>
    where
        F: FnMut(&VisibleChunk<'_>) -> Result<T>,
        T: AsRef<str>,
    {
        // Score completion needs term contributions for the rows already named by
        // another retrieval route. Keep the caller's row count as the limit so one
        // requested chunk's low or zero lexical score cannot displace it.
        Self::search_snapshot_selected(
            snapshot,
            collection_id,
            query,
            TextSearchOptions::with_explanations(ids.len()),
            Some(ids),
            context,
            load_text,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn search_snapshot_selected<F, T>(
        snapshot: &VisibleSnapshot<'_>,
        collection_id: &str,
        query: &str,
        options: TextSearchOptions,
        ids: Option<&[&str]>,
        context: &JsonOperationContext,
        mut load_text: F,
    ) -> Result<Bm25Hits>
    where
        F: FnMut(&VisibleChunk<'_>) -> Result<T>,
        T: AsRef<str>,
    {
        let limit = options.limit;
        // Skip both passes for a zero output limit. In particular, the fallback
        // does not need to open chunk payloads or allocate query vocabulary
        // storage just to return an empty list.
        if limit == 0 {
            return Ok(Bm25Hits::empty(context));
        }
        let terms = crate::tokenizer::query_terms_in(query, context)?;
        // The scan has no postings list. Its first pass computes document frequencies
        // and average length across the whole collection, even when the caller asks
        // for scores for a few IDs. Restricting these statistics to those IDs would
        // give them different BM25 scores from the ordinary retrieval path.
        // The table scales with the query vocabulary. Each slot holds the current
        // chunk's count alongside its accumulating document frequency, avoiding
        // storage proportional to all chunks or all corpus terms.
        let mut counts = ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        for _ in 0..terms.len() {
            counts.push(TermCounts::default());
        }
        let mut chunks = 0;
        let mut total_length = 0;
        // Both passes iterate the same visible snapshot. Ignore foreign collection
        // rows before loading any text so their terms and lengths cannot change
        // this collection's scores.
        for visible in snapshot.live_chunks() {
            if visible.chunk.collection_id != collection_id {
                continue;
            }
            // Charge a text visit only after the collection filter and before
            // attempting the payload read. The second pass counts any repeated
            // visit to that same document separately.
            context.search_cost_add(crate::search_cost::Counter::TextDocumentVisits, 1);
            total_length += count_terms(
                load_text(&visible)?.as_ref(),
                &terms,
                counts.as_mut_slice(),
                context,
            )?;
            // Every visible chunk in the collection contributes to the population,
            // including a chunk with no matching terms. Its effective length still
            // influences the average used for other chunks.
            chunks += 1;
            for count in counts.as_mut_slice() {
                // A chunk contributes at most one document even if the term
                // repeats. This pass uses the occurrence count only as a presence
                // test. Scoring counts it again when the chunk is loaded.
                count.document_frequency += usize::from(count.frequency != 0);
            }
        }
        // Ordinary recall with no matching term is already complete. A selected-ID
        // lookup must still visit the requested chunks and return their real zero
        // scores if they exist.
        // An empty visible collection returns immediately for either mode.
        // Dividing by chunks and checking requested-ID completeness below both
        // require reaching the nonempty-collection path.
        if chunks == 0
            || (ids.is_none()
                && counts
                    .as_slice()
                    .iter()
                    .all(|count| count.document_frequency == 0))
        {
            return Ok(Bm25Hits::empty(context));
        }
        // count_terms gives even a tokenless chunk an effective length of one.
        // Thus a nonempty collection has a positive average here even if none of
        // its texts produced a token.
        let avg_len = total_length as f64 / chunks as f64;
        // Compute each term's inverse document frequency once from the complete
        // corpus. Keep its index aligned with the query terms and counts as each
        // second-pass chunk is retokenized.
        let mut idfs = ReservedVec::with_capacity(terms.len(), context, "operation_scratch")?;
        for count in counts.as_slice() {
            let df = count.document_frequency as f64;
            idfs.push(((chunks as f64 - df + 0.5) / (df + 0.5) + 1.0).ln());
        }
        let idfs = idfs.as_slice();
        // With collection statistics fixed, load text again to score each eligible
        // chunk. Ordinary recall retains a bounded ranked window. Score completion
        // needs an entry for every requested ID, including zero matches, in ID order
        // for the later merge.
        let mut candidates = TextCandidates::new(limit, context);
        // Allocating the selected table can fail before the first eligible
        // second-pass row. Its known bound avoids growing the result in response
        // to each payload that the scan happens to encounter.
        let mut selected = ids
            .map(|ids| Bm25Hits::with_capacity(ids.len(), context))
            .transpose()?;
        for visible in snapshot.live_chunks() {
            if visible.chunk.collection_id != collection_id
                // Completion supplies chunk ID strings in lexical order.
                // Binary-search that slice before opening a payload on this second
                // pass. The first pass still visited every collection member.
                || ids
                    .is_some_and(|ids| ids.binary_search(&visible.chunk.chunk_id.as_str()).is_err())
            {
                continue;
            }
            context.search_cost_add(crate::search_cost::Counter::TextDocumentVisits, 1);
            let length = count_terms(
                load_text(&visible)?.as_ref(),
                &terms,
                counts.as_mut_slice(),
                context,
            )?;
            let mut score = 0.0;
            let mut matched_count = 0;
            // The counts now describe this chunk's occurrences while the IDFs
            // still describe the first pass's collection. Accumulate one
            // contribution per distinct matching term, using this chunk's own
            // length.
            for (count, idf) in counts.as_slice().iter().zip(idfs) {
                if count.frequency != 0 {
                    matched_count += 1;
                    score += contribution(count.frequency, length, avg_len, *idf);
                }
            }
            // Omit an unmatched chunk from the lexical recall window. Score
            // completion was explicitly asked about its ID, so keep a zero-scoring
            // row with an empty term list in that case.
            if matched_count == 0 && ids.is_none() {
                continue;
            }
            let matched_terms = terms
                .iter()
                // A score-only query takes no explanation entries. Ranking has
                // already used the full occurrence counts, so suppressing term
                // details changes neither the score nor the admission order.
                .take(if options.explanations { terms.len() } else { 0 })
                .enumerate()
                .filter(|(index, _)| counts.as_slice()[*index].frequency != 0)
                // This iterator borrows counts that will be overwritten for the
                // next chunk. A retained hit copies the matched fields while they
                // still describe the row just scored.
                .map(|(index, term)| MatchedTerm {
                    document_frequency: counts.as_slice()[index].document_frequency,
                    chunk_length: length,
                    avg_chunk_length: avg_len,
                    term,
                    frequency: counts.as_slice()[index].frequency,
                    bm25_contribution: contribution(
                        counts.as_slice()[index].frequency,
                        length,
                        avg_len,
                        idfs[index],
                    ),
                });
            // Completion pre-admitted enough output slots for the supplied IDs.
            // Ordinary recall hands the borrowed details to the bounded heap,
            // which can reject a losing row before copying them.
            if let Some(selected) = &mut selected {
                selected.push(&visible.chunk.chunk_id, score, matched_terms)?;
            } else {
                candidates.push(&visible.chunk.chunk_id, score, matched_terms)?;
            }
        }
        // Completion joins by ID in fusion. Check that every requested visible
        // chunk supplied a row before sorting the table, or a missing chunk could
        // be mistaken for a genuine zero match.
        if let Some(mut selected) = selected {
            // The snapshot supplies at most one row per visible chunk. A missing
            // ID or a repeated requested ID consequently leaves the nonempty
            // collection's completion table short.
            if selected.len() != ids.unwrap().len() {
                return Err(crate::Error::new(
                    crate::ErrorCode::RecoveryError,
                    "text candidate is not visible in collection",
                ));
            }
            selected.sort_by_id();
            Ok(selected)
        } else {
            // The candidate heap finishes in ranking order for ordinary recall.
            // Completion instead returns the ID-ordered table consumed by score
            // fusion.
            Ok(candidates.finish())
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::model::{Chunk, CollectionInfo, DistanceMetric, Document, VectorIndexConfig};
    use crate::{Error, ErrorCode, JsonValue};

    pub(in crate::index::text) fn fixture(seed: usize) -> Catalog {
        let mut catalog = Catalog::default();
        for owner in ["first", "second"] {
            catalog.collections.insert(
                owner.into(),
                CollectionInfo {
                    collection_id: owner.into(),
                    name: owner.into(),
                    embedding_dim: 2,
                    distance_metric: DistanceMetric::Cosine,
                    embedding_model: None,
                    tokenizer: Default::default(),
                    vectors_normalized: false,
                    vector_index: VectorIndexConfig::Exact,
                    created_at: String::new(),
                },
            );
            catalog.documents.insert(
                owner.into(),
                Box::new(Document {
                    document_id: owner.into(),
                    collection_id: owner.into(),
                    external_id: owner.into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    created_at: String::new(),
                    updated_at: String::new(),
                }),
            );
            for ordinal in 0..20 + seed % 17 {
                let id = format!("{owner}-{ordinal}");
                let text = if (ordinal + seed) % 7 == 0 {
                    "the and".into()
                } else {
                    format!(
                        "shared {} {}",
                        "alphaBeta ".repeat((ordinal + seed) % 4),
                        ["storage_path 42", "delta delta gamma", "\u{0130} \u{03a3}"][ordinal % 3]
                    )
                };
                catalog.chunks.insert(
                    id.clone(),
                    Box::new(Chunk {
                        chunk_id: id.clone(),
                        document_id: owner.into(),
                        collection_id: owner.into(),
                        chunk_external_id: id.clone(),
                        text,
                        text_hash: String::new(),
                        ordinal,
                        token_count: 0,
                        metadata: JsonValue::object(),
                        embedding_id: None,
                        source_span: None,
                        created_at: String::new(),
                    }),
                );
                if (ordinal + seed) % 11 == 0 {
                    catalog.deleted_chunks.insert(id);
                }
            }
        }
        if seed % 13 == 0 {
            catalog.deleted_documents.insert("first".into());
        }
        if seed % 17 == 0 {
            catalog.deleted_collections.insert("second".into());
        }
        catalog
    }

    #[test]
    fn rebuilt_postings_share_collection_identity_through_replacement_and_drop() {
        use std::collections::BTreeMap;
        use std::sync::{Arc, Weak};
        for seed in 0..100 {
            let catalog = fixture(seed);
            let snapshot = catalog.visible_snapshot();
            let mut index = TextIndex::try_build_matching(
                &snapshot,
                |_| true,
                |visible| Ok(visible.chunk.text.clone()),
            )
            .unwrap();
            let mut owners: BTreeMap<&str, (Weak<str>, usize)> = BTreeMap::new();
            for posting in index.postings.values().flatten() {
                let (identity, count) = owners
                    .entry(posting.collection_id.as_ref())
                    .or_insert_with(|| (Arc::downgrade(&posting.collection_id), 0));
                assert!(Arc::ptr_eq(
                    &identity.upgrade().unwrap(),
                    &posting.collection_id
                ));
                *count += 1;
            }
            let owners = owners
                .into_iter()
                .map(|(id, value)| (id.to_owned(), value))
                .collect::<BTreeMap<_, _>>();
            assert!(owners
                .values()
                .all(|(identity, count)| identity.strong_count() == *count));
            let Some((target, (previous, count))) = owners.first_key_value() else {
                continue;
            };
            let retained = index.clone();
            let replacement = TextIndex::try_build_matching(
                &snapshot,
                |id| id == target,
                |visible| Ok(visible.chunk.text.clone()),
            )
            .unwrap();
            let next =
                Arc::downgrade(&replacement.postings.values().next().unwrap()[0].collection_id);
            assert!(!Weak::ptr_eq(previous, &next));
            let old_chunk_ids = snapshot
                .live_chunks()
                .filter(|visible| &visible.chunk.collection_id == target)
                .map(|visible| visible.chunk.chunk_id.as_str());
            index.replace_collections(
                &BTreeSet::from([target.clone()]),
                old_chunk_ids,
                replacement,
            );
            assert_eq!(previous.strong_count(), *count);
            assert_eq!(next.strong_count(), *count);
            for id in owners.keys() {
                for query in ["shared", "alphaBeta storage_path", "delta gamma", "the and"] {
                    super::super::tests::assert_same_hits(
                        &index
                            .search(
                                id,
                                query,
                                TextSearchOptions::with_explanations(usize::MAX),
                                &JsonOperationContext::standalone(),
                            )
                            .unwrap(),
                        &retained
                            .search(
                                id,
                                query,
                                TextSearchOptions::with_explanations(usize::MAX),
                                &JsonOperationContext::standalone(),
                            )
                            .unwrap(),
                    );
                }
            }
            drop(retained);
            assert!(previous.upgrade().is_none());
            assert!(next.upgrade().is_some());
            drop(index);
            assert!(next.upgrade().is_none());
            assert!(owners
                .values()
                .all(|(identity, _)| identity.upgrade().is_none()));
        }
    }

    #[test]
    fn streamed_scores_terms_and_ties_equal_full_index_across_100_seeds() {
        for seed in 0..100 {
            let catalog = fixture(seed);
            let snapshot = catalog.visible_snapshot();
            let full = TextIndex::try_build_matching(
                &snapshot,
                |_| true,
                |visible| Ok(visible.chunk.text.clone()),
            )
            .unwrap();
            for owner in ["first", "second", "missing"] {
                let chunks = snapshot
                    .live_chunks()
                    .filter(|visible| visible.chunk.collection_id == owner)
                    .count();
                for query in [
                    "shared",
                    "alpha ALPHA storage",
                    "path 42 gamma",
                    "the and",
                    "absent",
                    "\u{0130} alphaBeta",
                    "alphaBeta_alphaBeta storage_path_storage_path",
                    "\u{0130}_\u{0130} \u{03a3}_\u{03a3} \u{39f}\u{3a3}Value",
                ] {
                    for limit in [0, 1, 9, usize::MAX] {
                        for options in [
                            TextSearchOptions::with_explanations(limit),
                            TextSearchOptions::scores_only(limit),
                        ] {
                            let mut expected = full.search_reference(owner, query, limit);
                            if !options.explanations {
                                for hit in &mut expected {
                                    hit.matched_terms.clear();
                                }
                            }
                            let mut reads = 0;
                            let actual = TextIndex::search_from_snapshot(
                                &snapshot,
                                owner,
                                query,
                                options,
                                &JsonOperationContext::standalone(),
                                |visible| {
                                    reads += 1;
                                    assert_eq!(visible.chunk.collection_id, owner);
                                    Ok(visible.chunk.text.clone())
                                },
                            )
                            .unwrap();
                            let cached = full
                                .search(owner, query, options, &JsonOperationContext::standalone())
                                .unwrap();
                            super::super::tests::assert_same_hits(&actual, &expected);
                            super::super::tests::assert_same_hits(&cached, &expected);
                            if !options.explanations {
                                assert!(actual
                                    .iter()
                                    .chain(cached.iter())
                                    .all(|hit| hit.matched_terms.capacity() == 0));
                            }
                            assert_eq!(
                                reads,
                                if limit == 0 {
                                    0
                                } else if expected.is_empty() {
                                    chunks
                                } else {
                                    2 * chunks
                                },
                                "seed={seed}, owner={owner}, query={query}, limit={limit}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn query_arrays_are_admitted_before_use_and_released_on_each_failure() {
        use super::super::QueryTerm;
        use crate::json::WorkLedger;
        use std::borrow::Cow;
        use std::mem::size_of;
        use std::sync::Arc;

        for (count, owned) in (1..=100).flat_map(|count| [(count, false), (count, true)]) {
            let query = (0..count)
                .map(|i| {
                    if owned {
                        format!("TERM{i}")
                    } else {
                        format!("term{i}")
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            let catalog = fixture(1);
            let snapshot = catalog.visible_snapshot();
            let full =
                TextIndex::try_build_matching(&snapshot, |_| true, |_| Ok(query.as_str())).unwrap();
            let chunks = snapshot
                .live_chunks()
                .filter(|visible| visible.chunk.collection_id == "first")
                .count();
            let owned_bytes = if owned {
                (0..count).map(|i| format!("term{i}").len()).sum()
            } else {
                0
            };
            let terms = crate::json::btree_workspace::<Cow<'_, str>, usize>(count) + owned_bytes;
            let counts = count * size_of::<TermCounts>();
            let idfs = count * size_of::<f64>();
            let plans = count * size_of::<QueryTerm<'_>>();
            let pending = count * size_of::<(&str, usize, usize)>();
            let limit = 65536;
            let ledger = Arc::new(WorkLedger::with_limits(limit, limit, limit));
            let context = JsonOperationContext::new(Arc::clone(&ledger));
            let other = JsonOperationContext::new(ledger);
            let options = TextSearchOptions::scores_only(1);
            for (cached, required, expected_reads) in [
                (true, terms, 0),
                (true, terms + plans, 0),
                (true, terms + plans + pending, 0),
                (false, terms, 0),
                (false, terms + counts, 0),
                (false, terms + counts + idfs, chunks),
            ] {
                let mut held = other.workspace_reservation();
                held.grow_to(
                    limit - required + 1,
                    limit - required + 1,
                    "operation_scratch",
                )
                .unwrap();
                let mut reads = 0;
                let result = if cached {
                    full.search("first", &query, options, &context)
                } else {
                    TextIndex::search_from_snapshot(
                        &snapshot,
                        "first",
                        &query,
                        options,
                        &context,
                        |_| {
                            reads += 1;
                            Ok(query.as_str())
                        },
                    )
                };
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                let detail = error
                    .detail()
                    .iter()
                    .cloned()
                    .collect::<std::collections::BTreeMap<_, _>>();
                assert_eq!(detail["resource"], "memory");
                assert_eq!(detail["operation"], "operation_scratch");
                assert_eq!(detail["limit_bytes"], limit.to_string());
                assert_eq!(detail["requested_bytes"], (limit + 1).to_string());
                assert_eq!(reads, expected_reads);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
            }
            let mut expected = full.search_reference("first", &query, 1);
            expected[0].matched_terms.clear();
            let actual = full.search("first", &query, options, &context).unwrap();
            super::super::tests::assert_same_hits(&actual, &expected);
            drop(actual);
            let actual = TextIndex::search_from_snapshot(
                &snapshot,
                "first",
                &query,
                options,
                &context,
                |_| Ok(query.as_str()),
            )
            .unwrap();
            super::super::tests::assert_same_hits(&actual, &expected);
            drop(actual);
            other
                .workspace_reservation()
                .grow_to(limit, limit, "operation_scratch")
                .unwrap();
        }
    }

    #[test]
    fn token_budget_refusals_in_either_pass_discard_candidates_and_allow_retry() {
        use crate::json::WorkLedger;
        use std::sync::Arc;
        let catalog = fixture(1);
        let snapshot = catalog.visible_snapshot();
        let chunks = snapshot
            .live_chunks()
            .filter(|visible| visible.chunk.collection_id == "first")
            .count();
        let denied = "A".repeat(65537);
        let reference =
            TextIndex::try_build_matching(&snapshot, |_| true, |_| Ok("shared")).unwrap();
        let limit = 8192;
        let nodes = 11 * std::mem::size_of::<(std::borrow::Cow<'_, str>, usize)>()
            + 16 * std::mem::size_of::<usize>();
        let mut retained_ids = snapshot
            .live_chunks()
            .filter(|visible| visible.chunk.collection_id == "first")
            .take(chunks - 1)
            .map(|visible| visible.chunk.chunk_id.as_str())
            .collect::<Vec<_>>();
        retained_ids.sort_unstable();
        for parent in [false, true] {
            for fail_at in [0, chunks - 1, chunks, 2 * chunks - 1] {
                for options in [
                    TextSearchOptions::with_explanations(3),
                    TextSearchOptions::scores_only(3),
                ] {
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { 2 * denied.len() } else { limit },
                        0,
                        if parent { limit } else { 2 * denied.len() },
                    )));
                    let mut reads = 0;
                    let second_pass = fail_at >= chunks;
                    let candidates = if fail_at == 2 * chunks - 1 {
                        3 * std::mem::size_of::<super::super::Bm25Hit>()
                            + retained_ids
                                .iter()
                                .take(3)
                                .map(|id| id.len())
                                .sum::<usize>()
                            + usize::from(options.explanations)
                                * 3
                                * (std::mem::size_of::<MatchedTerm>() + "shared".len())
                    } else {
                        0
                    };
                    let requested = nodes
                        + std::mem::size_of::<TermCounts>()
                        + usize::from(second_pass) * std::mem::size_of::<f64>()
                        + candidates
                        + denied.len();
                    let (result, allocations) =
                        crate::test_alloc::key_allocations(denied.len(), || {
                            TextIndex::search_from_snapshot(
                                &snapshot,
                                "first",
                                "shared",
                                options,
                                &context,
                                |_| {
                                    let current = reads;
                                    reads += 1;
                                    if current == fail_at {
                                        Ok(denied.as_str())
                                    } else {
                                        Ok("shared")
                                    }
                                },
                            )
                        });
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", requested.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                    }
                    assert_eq!(reads, fail_at + 1);
                    assert_eq!(allocations.allocated, 0);
                    assert_eq!(allocations.live, 0);
                    assert!(!allocations.overflow);
                    assert_eq!(context.available_workspace_bytes(), limit);
                    let actual = TextIndex::search_from_snapshot(
                        &snapshot,
                        "first",
                        "shared",
                        options,
                        &context,
                        |_| Ok("shared"),
                    )
                    .unwrap();
                    let expected = reference
                        .search("first", "shared", options, &context)
                        .unwrap();
                    super::super::tests::assert_same_hits(&actual, &expected);
                    drop(actual);
                    drop(expected);
                    assert_eq!(context.available_workspace_bytes(), limit);
                }
            }
        }
    }

    #[test]
    fn read_errors_in_either_pass_do_not_return_partial_candidates() {
        let catalog = fixture(1);
        let snapshot = catalog.visible_snapshot();
        let chunks = snapshot
            .live_chunks()
            .filter(|visible| visible.chunk.collection_id == "first")
            .count();
        for fail_at in [0, chunks - 1, chunks, 2 * chunks - 1] {
            for options in [
                TextSearchOptions::with_explanations(3),
                TextSearchOptions::scores_only(3),
            ] {
                let context = JsonOperationContext::standalone();
                let mut reads = 0;
                let error = TextIndex::search_from_snapshot(
                    &snapshot,
                    "first",
                    "shared",
                    options,
                    &context,
                    |visible| {
                        let current = reads;
                        reads += 1;
                        if current == fail_at {
                            Err(Error::new(ErrorCode::IoError, "injected text read failure"))
                        } else {
                            Ok(visible.chunk.text.clone())
                        }
                    },
                )
                .unwrap_err();
                assert_eq!(error.code(), ErrorCode::IoError);
                assert_eq!(reads, fail_at + 1);
                let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
                context
                    .workspace_reservation()
                    .grow_to(limit, limit, "operation_scratch")
                    .unwrap();
            }
        }
    }
}
