use crate::catalog::Catalog;
use crate::error::Result;
#[cfg(test)]
use crate::index::hybrid::{fuse, CandidateScore};
use crate::index::text::{Bm25Hit, Bm25Hits};
use crate::index::vector::{VectorHit, VectorHits};
use crate::json::{JsonOperationContext, ReservedVec};
use crate::metadata::{FilterExpr, MetadataBoostConfig};
use crate::model::SearchWeights;

pub(super) fn filter_candidates(
    catalog: &Catalog,
    bm25_hits: &mut Bm25Hits,
    vector_hits: &mut VectorHits,
    filter: Option<&FilterExpr>,
) {
    // Both retrieval branches use the same metadata predicate before their scores
    // are combined. A chunk returned by both branches must not survive solely
    // through its other score.
    if let Some(filter) = filter {
        // Hit lists contain internal chunk IDs. A missing catalog row has no
        // metadata with which to satisfy the predicate, so remove that hit along
        // with rows that explicitly fail it.
        let matches = |chunk_id: &str| {
            catalog
                .chunks
                .get(chunk_id)
                .is_some_and(|chunk| filter.matches_for_chunk(catalog, chunk))
        };
        // Retaining in place preserves each branch's existing order. This step
        // does not fetch another candidate to replace a filtered-out hit from a
        // bounded recall window.
        bm25_hits.retain(|hit| matches(&hit.chunk_id));
        vector_hits.retain(|hit| matches(&hit.chunk_id));
    }
}

// Retrieval limits remain caller-owned: diagnostics deliberately pass full candidates.
#[cfg(test)]
pub(super) fn fuse_candidates<'a>(
    catalog: &Catalog,
    bm25_hits: &'a [Bm25Hit],
    vector_hits: &'a [VectorHit],
    boost: Option<&MetadataBoostConfig>,
    weights: SearchWeights,
    fusion_limit: usize,
    context: &JsonOperationContext,
) -> Result<ReservedVec<CandidateScore<'a>>> {
    fuse(
        bm25_hits,
        vector_hits,
        weights,
        fusion_limit,
        context,
        |id| {
            boost
                .and_then(|config| {
                    catalog
                        .chunks
                        .get(id)
                        .map(|chunk| config.score_for_chunk(catalog, chunk))
                })
                .unwrap_or_default()
        },
    )
}

pub(super) struct SupplementaryScores {
    // Text scores computed for chunk IDs obtained through vector recall. These rows
    // supply missing components without expanding the original text window.
    pub text: Bm25Hits,
    // Vector scores computed for eligible text hits whose vectors were missing from
    // the vector window.
    pub vector: VectorHits,
    // Additional vector checks attributed to score completion in the accumulated
    // vector work.
    pub vector_scans: usize,
    // Number of vector components examined in the accumulated distance
    // calculations, rather than the number of completed result rows.
    pub distance_ops: usize,
    // Primary payload bytes read while completing scores, including reads from
    // earlier attempts after a budget retry. The caller adds recall bytes
    // separately.
    pub bytes_read: u64,
    // Whether this query must keep using the transient text source after checkpoint
    // validation failed, including on a retry.
    pub invalidated_text: bool,
}

// This operation owns only IDs from U and one loaded payload at a time. A failed
// index visit or allocation drops its provisional tables before central reclaim.
#[allow(clippy::too_many_arguments)]
pub(super) fn supplement_scores(
    inner: &super::Inner,
    collection: &crate::model::CollectionInfo,
    text: Option<&str>,
    text_has_matches: bool,
    query: Option<&crate::model::QueryEmbeddingInput>,
    bm25_hits: &[Bm25Hit],
    vector_hits: &[VectorHit],
    query_context: &crate::query::SearchQueryContext,
    context: &JsonOperationContext,
    vector_work: &mut crate::index::vector::VectorWork,
    bytes_read: &mut u64,
    invalidated_text: &mut bool,
) -> Result<SupplementaryScores> {
    // BM25 hits are ordered by chunk ID for the membership checks. Fill in the other score only
    // for IDs already recalled by one route, preserving the original candidate set.
    let missing = vector_hits
        .iter()
        .filter(|hit| {
            bm25_hits
                .binary_search_by(|row| row.chunk_id.cmp(&hit.chunk_id))
                .is_err()
        })
        .count();
    // Counting before allocating keeps this workspace proportional to vector-only
    // hits. The entries borrow IDs from the recall window, which stays alive until
    // fusion has consumed both windows.
    let mut ids = ReservedVec::with_capacity(missing, context, "operation_scratch")?;
    for hit in vector_hits {
        if bm25_hits
            .binary_search_by(|row| row.chunk_id.cmp(&hit.chunk_id))
            .is_err()
        {
            ids.push(hit.chunk_id.as_str());
        }
    }
    // The selected-ID text scorers take their lookup keys in chunk-ID order.
    // Vector recall orders by score, so its surviving IDs cannot be passed
    // directly even when every vector hit needs a text score.
    ids.as_mut_slice().sort_unstable();
    let text_scores = if let Some(text) = text.filter(|_| !ids.as_slice().is_empty()) {
        if !text_has_matches {
            // Empty unfiltered recall proves no query term matched anywhere in
            // this collection/snapshot. Reuse that proof without reparsing terms.
            let mut zero_scores = Bm25Hits::with_capacity(ids.as_slice().len(), context)?;
            for id in ids.as_slice() {
                // Materialize a zero for each requested ID, with an empty term
                // explanation. Fusion can then distinguish a known zero for a text
                // query from an absent text modality.
                context.search_cost_add(crate::search_cost::Counter::ZeroMatchProofReuse, 1);
                zero_scores.push(id, 0.0, std::iter::empty())?;
            }
            zero_scores
        // A resident text index can supply scores against its collection-wide
        // statistics without reopening the checkpoint segments. A previous
        // invalidation applies even if this attempt still sees a resident cache.
        } else if !*invalidated_text
            && inner.index_suffix_status == super::IndexSuffixState::PersistentCheckpoint
            && inner.text_index.has_resident_cache()
        {
            inner
                .text_index
                .score_ids(&collection.collection_id, text, ids.as_slice(), context)?
        } else {
            // The segment directory serves text sections as well as vectors. If it
            // carries no usable text segment for this collection, scoring can
            // still use the committed chunk payloads.
            let indexed = if !*invalidated_text
                && inner.index_suffix_status == super::IndexSuffixState::PersistentCheckpoint
            {
                inner
                    .lazy_vector_source
                    .as_ref()
                    .map(|source| {
                        crate::index::text::TextIndex::score_ids_from_segment(
                            source,
                            &inner.storage,
                            &collection.collection_id,
                            text,
                            ids.as_slice(),
                            context,
                        )
                    })
                    // A read or allocation error aborts this attempt. Only a
                    // missing source or successful lookup with no usable indexed
                    // scores falls through to the snapshot below.
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            if let Some(scores) = indexed {
                scores
            } else {
                // Keep the fallback flag outside this function's provisional
                // results. If scoring later runs out of memory, the next attempt
                // must remember that the checkpoint could not answer this request.
                *invalidated_text |=
                    inner.index_suffix_status == super::IndexSuffixState::PersistentCheckpoint;
                // Pin this query to its committed snapshot once the checkpoint
                // fails, even if an external writer restores it before retry.
                if *invalidated_text {
                    // Compare the estimated text pass, including the requested
                    // IDs, with the query profile's transient rebuild limit before
                    // scanning the collection.
                    super::ensure_transient_rebuild_budget(
                        super::transient_query_scratch_bytes(
                            &inner.catalog,
                            &collection.collection_id,
                            Some(text),
                            crate::index::text::TextSearchOptions::with_explanations(
                                ids.as_slice().len(),
                            ),
                            false,
                            0,
                        ),
                        query_context.transient_rebuild.budget_mb,
                        query_context.profile,
                    )?;
                }
                // Selecting a few chunk IDs narrows the returned scores. Computing
                // comparable BM25 values still needs collection-wide document
                // frequency and average length from this visible snapshot.
                let mut reader = inner.storage.payload_reader()?;
                crate::index::text::TextIndex::score_ids_from_snapshot(
                    &inner.catalog.visible_snapshot(),
                    &collection.collection_id,
                    text,
                    ids.as_slice(),
                    context,
                    |visible| {
                        // Pass the caller's running byte counter through each
                        // materialization. The reader charges an on-disk payload
                        // when requested, so a budget retry can account for the
                        // same payload more than once.
                        super::read_visible_chunk_text(&mut reader, visible, context, bytes_read)
                    },
                )?
            }
        }
    } else {
        Bm25Hits::empty(context)
    };
    // The text scores own the IDs they will publish. Return the borrowed-ID
    // array's workspace now, before vector materialization starts competing for
    // the same operation budget.
    drop(ids);
    // Fill in vector scores only for text hits missing from the vector candidate
    // list. A text hit without a visible embedding stays in the text list and
    // contributes no supplementary vector score.
    // At most one new vector score can be needed per text hit. Use that window as
    // the capacity bound even when the independent vector recall limit is smaller.
    let mut vector_scores = crate::index::vector::VectorCandidates::new(bm25_hits.len(), context);
    if let Some(query) = query {
        // Construct the scorer once for this set of text-only candidates. For
        // cosine similarity it can reuse the query norm in every equal-dimension
        // comparison.
        let scorer =
            crate::index::vector::VectorScorer::new(&query.vector, collection.distance_metric);
        let mut reader = inner.storage.payload_reader()?;
        // The vector window retains recall order. Test membership by identity here
        // so a low-ranked retained hit is never charged a second distance
        // calculation solely because its text rank differs.
        for hit in bm25_hits {
            if vector_hits.iter().any(|row| row.chunk_id == hit.chunk_id) {
                continue;
            }
            let Some(embedding) = inner.catalog.embedding_for_chunk(&hit.chunk_id) else {
                continue;
            };
            // Pair the catalog entry with its current record locator. A live
            // embedding may be represented by an in-memory vector or by an on-disk
            // body, and the reader accounts for those paths separately.
            let visible = crate::catalog::VisibleEmbedding {
                embedding,
                record_ref: inner
                    .catalog
                    .record_ref_for_embedding(&embedding.embedding_id),
            };
            // Count a vector check before attempting to load its components. The
            // distance counter below measures only a completed calculation, so a
            // read failure or malformed vector can separate the two counts.
            vector_work.scans += 1;
            context.search_cost_add(crate::search_cost::Counter::VectorChecks, 1);
            let vector =
                super::read_visible_embedding_vector(&mut reader, &visible, context, bytes_read)?;
            // Query admission has already established the expected dimension. A
            // different materialized length indicates inconsistent stored data and
            // must reach the caller as a recovery error.
            if vector.as_slice().len() != query.vector.len() {
                return Err(crate::Error::new(
                    crate::ErrorCode::RecoveryError,
                    "visible embedding dimension does not match query",
                ));
            }
            let raw = scorer.score(vector.as_slice());
            vector_work.distance_ops += vector.as_slice().len();
            context.search_cost_add(
                crate::search_cost::Counter::DistanceComponents,
                vector.as_slice().len() as u64,
            );
            // Only the scalar score and chunk identity must survive this
            // iteration. Return the vector payload's workspace before admitting
            // the candidate's retained ID.
            drop(vector);
            vector_scores.push(
                &hit.chunk_id,
                raw,
                // Use the same metric and normalization flag as vector recall.
                // Fusion needs both the raw distance or similarity and its
                // larger-is-better value to compare hits obtained by either route.
                crate::index::vector::normalize_score(
                    raw,
                    collection.distance_metric,
                    collection.vectors_normalized,
                ),
            )?;
        }
    }
    let mut vector_scores = vector_scores.finish();
    // The bounded candidate builder returns rank order. Fusion binary-searches
    // supplementary rows by chunk ID and must see this independent ordering even
    // if just one side supplied the original hit.
    vector_scores.sort_by_id();
    Ok(SupplementaryScores {
        text: text_scores,
        vector: vector_scores,
        // The caller owns these accumulators across budget retries. Publish the
        // accumulated checks, component operations and payload reads together with
        // the successful set of scores.
        vector_scans: vector_work.scans,
        distance_ops: vector_work.distance_ops,
        bytes_read: *bytes_read,
        invalidated_text: *invalidated_text,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fuse_completed_candidates<'a>(
    catalog: &Catalog,
    bm25_hits: &'a [Bm25Hit],
    vector_hits: &'a [VectorHit],
    text_scores: &[Bm25Hit],
    vector_scores: &[VectorHit],
    text_present: bool,
    boost: Option<&MetadataBoostConfig>,
    weights: SearchWeights,
    fusion_limit: usize,
    context: &JsonOperationContext,
) -> Result<crate::index::hybrid::FusionResult<'a>> {
    // Pass recall windows and score-completion rows separately. The recall windows
    // establish membership and candidate identity. The supplementary rows can
    // supply the missing component of one of those candidates without expanding
    // the union of recalled IDs.
    crate::index::hybrid::fuse_completed(
        bm25_hits,
        vector_hits,
        text_scores,
        vector_scores,
        // A present text query and a nonempty text window describe different
        // conditions. Preserve the first even if it found no candidates, so the
        // fusion helper can apply the corresponding scoring rules without
        // inferring input presence from the hit count.
        text_present,
        weights,
        fusion_limit,
        context,
        // Calculate metadata boost against the same catalog view as the
        // candidates. An absent configuration or a missing chunk summary yields
        // the neutral score and matches no rule. Normalize and bound the final
        // fused candidate list in the shared hybrid implementation.
        |id| {
            boost
                .and_then(|config| {
                    catalog
                        .chunks
                        .get(id)
                        .map(|chunk| config.score_for_chunk(catalog, chunk))
                })
                .unwrap_or_default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{parse, JsonValue};
    use crate::model::Chunk;
    use std::collections::BTreeMap;

    #[test]
    fn candidate_pipeline_matches_independent_filter_and_score_oracle() {
        let mut catalog = Catalog::default();
        for i in 0..9 {
            let id = format!("c{i}");
            catalog.chunks.insert(
                id.clone(),
                Box::new(Chunk {
                    chunk_id: id.clone(),
                    chunk_external_id: id,
                    collection_id: "collection".into(),
                    document_id: "document".into(),
                    text: String::new(),
                    text_hash: String::new(),
                    ordinal: i,
                    token_count: 0,
                    metadata: parse(&format!(
                        r#"{{"keep":{},"boost":{}}}"#,
                        i % 2 == 0,
                        i % 3 == 0
                    ))
                    .unwrap(),
                    embedding_id: None,
                    source_span: None,
                    created_at: String::new(),
                }),
            );
        }
        let boost = MetadataBoostConfig::parse(parse(r#"{"schema_version":"0.1","combine":"sum","max_boost":0.3,"rules":[{"when":{"chunk.boost":true},"boost":0.25},{"when":{"chunk.keep":true},"boost":0.125}]}"#).unwrap()).unwrap();
        for seed in 0..100 {
            let filter = FilterExpr::parse(match seed % 4 {
                0 => JsonValue::object(),
                1 => parse(r#"{"chunk.keep":true}"#).unwrap(),
                2 => parse(r#"{"chunk.keep":false}"#).unwrap(),
                _ => parse(r#"{"chunk.absent":"value"}"#).unwrap(),
            })
            .unwrap();
            let weights = match seed % 5 {
                0 => SearchWeights::new(1.0, 0.0, 0.0),
                1 => SearchWeights::new(0.0, 1.0, 0.0),
                2 => SearchWeights::new(0.25, 0.5, 0.25),
                3 => SearchWeights::new(0.0, 0.0, 1.0),
                _ => SearchWeights::new(0.5, 0.5, 0.0),
            }
            .unwrap();
            let boost_enabled = seed % 3 != 0;
            let included = |i| match seed % 4 {
                0 => true,
                1 => i % 2 == 0,
                2 => i % 2 != 0,
                _ => false,
            };
            let text_score = |i| ((seed + i * 3) % 11) as f64;
            let vector_score = |i| ((seed + i) % 9) as f64 / 8.0;
            let has_text = |i| (seed + i) % 3 != 0;
            let has_vector = |i| (seed + i) % 4 != 0;
            let mut bm25 = Bm25Hits::with_capacity(8, &JsonOperationContext::standalone()).unwrap();
            for i in (0..8).filter(|&i| has_text(i)) {
                bm25.push(&format!("c{i}"), text_score(i), std::iter::empty())
                    .unwrap();
            }
            let mut vector = crate::index::vector::VectorCandidates::new(
                8,
                &crate::json::JsonOperationContext::standalone(),
            );
            for i in (0..8).filter(|&i| has_vector(i)) {
                vector
                    .push(&format!("c{i}"), 1.0 - vector_score(i), vector_score(i))
                    .unwrap();
            }
            let mut vector = vector.finish();
            let maximum = (0..8)
                .filter(|&i| included(i) && has_text(i))
                .map(text_score)
                .fold(0.0_f64, f64::max);
            let mut expected = Vec::new();
            let mut expected_counts = BTreeMap::new();
            for i in 0..8 {
                if !included(i) || (!has_text(i) && !has_vector(i)) {
                    continue;
                }
                let bm25_norm = if has_text(i) && maximum > 0.0 {
                    text_score(i) / maximum
                } else {
                    0.0
                };
                let vector_norm = if has_vector(i) { vector_score(i) } else { 0.0 };
                let boost_score = if boost_enabled {
                    let count = usize::from(i % 3 == 0) + usize::from(i % 2 == 0);
                    if count > 0 {
                        expected_counts.insert(format!("c{i}"), count);
                    }
                    ((if i % 3 == 0 { 0.25_f64 } else { 0.0 })
                        + if i % 2 == 0 { 0.125 } else { 0.0 })
                    .min(0.3)
                } else {
                    0.0
                };
                expected.push((
                    format!("c{i}"),
                    weights.bm25 * bm25_norm
                        + weights.vector * vector_norm
                        + weights.metadata * boost_score,
                ));
            }
            expected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            expected.truncate(seed % 10);
            filter_candidates(&catalog, &mut bm25, &mut vector, filter.as_ref());
            let ranked = fuse_candidates(
                &catalog,
                &bm25,
                &vector,
                boost_enabled.then_some(&boost),
                weights,
                seed % 10,
                &JsonOperationContext::standalone(),
            )
            .unwrap();
            for score in ranked.as_slice() {
                assert_eq!(
                    score.matched_boost_rule_count,
                    expected_counts
                        .get(score.chunk_id)
                        .copied()
                        .unwrap_or_default(),
                    "seed={seed}"
                );
            }
            let actual = ranked
                .as_slice()
                .iter()
                .map(|score| (score.chunk_id.to_owned(), score.final_score))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "seed={seed}");
            assert_eq!(
                bm25.len(),
                (0..8).filter(|&i| included(i) && has_text(i)).count()
            );
            assert_eq!(
                vector.len(),
                (0..8).filter(|&i| included(i) && has_vector(i)).count()
            );
        }
    }
}
