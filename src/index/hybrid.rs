use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::Result;
use crate::index::text::Bm25Hit;
use crate::index::vector::VectorHit;
use crate::json::{JsonOperationContext, ReservedVec};
use crate::metadata::MetadataBoostScore;
use crate::model::SearchWeights;

#[derive(Clone, Copy, Debug, Default)]
pub struct CandidateScore<'a> {
    /// Identifier borrowed from one of the recall windows. Those windows must
    /// outlive the ranked rows.
    pub chunk_id: &'a str,
    /// The BM25 recall window contained this chunk before score completion, even if
    /// its lexical weight was zero.
    pub bm25_recalled: bool,
    /// The vector window admitted this chunk. Completing a vector score for a text
    /// hit leaves this flag unchanged.
    pub vector_recalled: bool,
    /// Lexical score before normalization. With score completion, a text query can
    /// produce a real zero for a vector-recalled chunk with no term matches.
    pub bm25_raw: Option<f64>,
    /// Raw BM25 divided by the candidate union's positive maximum and clamped to
    /// the unit interval. Zero if the maximum is zero.
    pub bm25_norm: f64,
    /// The collection metric's original value, absent if the chunk has no available
    /// vector score.
    pub vector_raw: Option<f64>,
    /// Metric-specific vector component before fusion weighting. An absent
    /// component contributes zero to the sum.
    pub vector_norm: Option<f64>,
    /// Contribution supplied by the metadata scorer after the rule cap and before
    /// applying the metadata weight.
    pub metadata_boost: f64,
    /// Number of metadata rules that matched, including matches that add nothing
    /// once the cap is reached.
    pub matched_boost_rule_count: usize,
    /// Weighted sum used for ranking. A larger score ranks first, with the internal
    /// chunk ID breaking ties.
    pub final_score: f64,
}

impl PartialEq for CandidateScore<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Use the same ranking comparison for equality and ordering. Component
        // scores and recall flags can differ without changing a row's position
        // when its final score and chunk ID are the same.
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for CandidateScore<'_> {}
impl PartialOrd for CandidateScore<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
// Best-first order leaves the worst retained candidate at the max-heap root.
impl Ord for CandidateScore<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse only the score comparison. On equal scores the smaller internal
        // ID sorts first, making the larger ID the first tied candidate displaced
        // from a full heap.
        other
            .final_score
            .partial_cmp(&self.final_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.chunk_id.cmp(other.chunk_id))
    }
}

// Use the scores already present in the two recall windows when the caller supplies
// no supplementary lookup tables. A component available only through the other
// route then remains absent.
pub fn fuse<'a>(
    bm25_hits: &'a [Bm25Hit],
    vector_hits: &'a [VectorHit],
    weights: SearchWeights,
    fusion_limit: usize,
    context: &JsonOperationContext,
    metadata_boost: impl FnMut(&str) -> MetadataBoostScore,
) -> Result<ReservedVec<CandidateScore<'a>>> {
    // Both recall windows own the strings borrowed by the resulting rows. This
    // entry point uses their measured scores directly and returns only the bounded
    // rows, discarding the union size and normalization maximum from the full
    // fusion report.
    Ok(fuse_scored(
        bm25_hits,
        vector_hits,
        None,
        weights,
        fusion_limit,
        context,
        metadata_boost,
    )?
    .hits)
}

pub(crate) struct FusionResult<'a> {
    /// Best-first rows, limited by the fusion window. Their IDs continue to borrow
    /// from the recall inputs.
    pub hits: ReservedVec<CandidateScore<'a>>,
    /// Number of distinct IDs presented to fusion before the heap cutoff. A zero
    /// fusion limit skips the union and reports zero.
    pub candidate_count: usize,
    /// Maximum raw BM25 over the normalization candidates, including completed
    /// lexical scores. Zero when there is no positive lexical measurement.
    pub bm25_max: f64,
}

// Supplementary score tables are not recall tables: only B/V determine U and provenance.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fuse_completed<'a>(
    bm25_hits: &'a [Bm25Hit],
    vector_hits: &'a [VectorHit],
    text_scores: &[Bm25Hit],
    vector_scores: &[VectorHit],
    text_present: bool,
    weights: SearchWeights,
    fusion_limit: usize,
    context: &JsonOperationContext,
    metadata_boost: impl FnMut(&str) -> MetadataBoostScore,
) -> Result<FusionResult<'a>> {
    // The two recall slices select which IDs compete. The supplementary slices must
    // be ordered by chunk ID for lookup and may cover a wider population, as in an
    // exact diagnosis. Their extra rows may supply neither new candidates nor
    // out-of-window normalization scores.
    fuse_scored(
        bm25_hits,
        vector_hits,
        // Keep the text-present flag alongside the lookup slices. A requested text
        // query with no match can contribute a known zero, whereas a vector-only
        // query leaves the lexical component absent.
        Some((text_scores, vector_scores, text_present)),
        weights,
        // This limit bounds the fused output after both score components and the
        // metadata boost have been evaluated. The input slices already embody the
        // independent retrieval cutoffs.
        fusion_limit,
        context,
        metadata_boost,
    )
}

#[allow(clippy::too_many_arguments)]
fn fuse_scored<'a>(
    bm25_hits: &'a [Bm25Hit],
    vector_hits: &'a [VectorHit],
    completion: Option<(&[Bm25Hit], &[VectorHit], bool)>,
    weights: SearchWeights,
    fusion_limit: usize,
    context: &JsonOperationContext,
    mut metadata_boost: impl FnMut(&str) -> MetadataBoostScore,
) -> Result<FusionResult<'a>> {
    // A caller requesting no retained rows gets an empty result before allocating
    // the index permutation or invoking the metadata scorer.
    if fusion_limit == 0 {
        return Ok(FusionResult {
            hits: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            candidate_count: 0,
            bm25_max: 0.0,
        });
    }
    let count = bm25_hits
        .len()
        .checked_add(vector_hits.len())
        .ok_or_else(|| std::io::Error::other("candidate count overflow"))?;
    // Index the two input slices together rather than copying their IDs. Entries
    // with the same chunk ID become adjacent after sorting, so a chunk recalled by
    // both routes occupies one fusion slot.
    let mut indices = ReservedVec::with_capacity(count, context, "operation_scratch")?;
    for index in 0..count {
        indices.push(index);
    }
    // The permutation uses a single integer domain. Values below the text window
    // length index that window, and the rest are vector indices shifted by that
    // length. This keeps both scores borrowed throughout fusion.
    let chunk_id = |index: usize| {
        if index < bm25_hits.len() {
            bm25_hits[index].chunk_id.as_str()
        } else {
            vector_hits[index - bm25_hits.len()].chunk_id.as_str()
        }
    };
    // Preserve last-hit-wins within each branch without a stable-sort allocation.
    indices
        .as_mut_slice()
        .sort_unstable_by(|a, b| chunk_id(*a).cmp(chunk_id(*b)).then_with(|| a.cmp(b)));
    let indices = indices.as_slice();
    // Count distinct IDs before reserving the ranked array. The two windows may
    // overlap completely, and both the output capacity and the reported
    // normalization population depend on that overlap.
    let unique = indices
        .iter()
        .enumerate()
        .filter(|(offset, index)| {
            *offset == 0 || chunk_id(indices[*offset - 1]) != chunk_id(**index)
        })
        .count();
    let capacity = unique.min(fusion_limit);
    let (storage, work) =
        ReservedVec::with_capacity(capacity, context, "operation_scratch")?.into_parts();
    let mut candidates = BinaryHeap::<CandidateScore<'a>>::from(storage);
    // Normalize over the recalled union, including text scores filled in for
    // vector-only candidates. Scores outside that union must not dilute the hits.
    let bm25_max = bm25_hits
        .iter()
        .map(|hit| hit.score)
        // A diagnostic caller can supply text scores for the whole collection while
        // testing a narrower recall window. Consult the union index for each such
        // score so a stronger out-of-window hit cannot rescale every returned row.
        .chain(completion.into_iter().flat_map(|(scores, _, _)| {
            scores
                .iter()
                .filter(|hit| {
                    indices
                        .binary_search_by(|index| chunk_id(*index).cmp(hit.chunk_id.as_str()))
                        .is_ok()
                })
                .map(|hit| hit.score)
        }))
        .inspect(|_| {
            context.search_cost_add(crate::search_cost::Counter::NormalizationScoreVisits, 1)
        })
        // Seed the maximum at zero. When the recall union has no positive lexical
        // score, normalization below returns zero without dividing by that
        // maximum.
        .fold(0.0_f64, f64::max)
        .max(0.0);
    let mut offset = 0;
    while offset < indices.len() {
        context.search_cost_add(crate::search_cost::Counter::FusionCandidateVisits, 1);
        let id = chunk_id(indices[offset]);
        let mut score = CandidateScore {
            chunk_id: id,
            ..CandidateScore::default()
        };
        // Consume the entire run for one ID before applying weights. Entries from
        // the two branches fill separate components. Repeated entries from the same
        // branch are visited in input order, leaving its last measurement in the
        // row.
        while offset < indices.len() && chunk_id(indices[offset]) == id {
            let index = indices[offset];
            if index < bm25_hits.len() {
                let hit = &bm25_hits[index];
                context.search_cost_add(crate::search_cost::Counter::Bm25ScoreReuse, 1);
                score.bm25_recalled = true;
                score.bm25_raw = Some(hit.score);
                score.bm25_norm = if bm25_max > 0.0 {
                    (hit.score / bm25_max).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            } else {
                let hit = &vector_hits[index - bm25_hits.len()];
                context.search_cost_add(crate::search_cost::Counter::VectorScoreReuse, 1);
                score.vector_recalled = true;
                score.vector_raw = Some(hit.raw_score);
                score.vector_norm = Some(hit.normalized_score);
            }
            offset += 1;
        }
        // Keep the recall flags from the input windows when filling a missing
        // component. A text-only hit may acquire a real vector score here; explain
        // must still report that text was the route that retrieved it.
        if let Some((text_scores, vector_scores, text_present)) = completion {
            if text_present && score.bm25_raw.is_none() {
                // Full-score lookup may omit genuine zero-match rows (debug full scan).
                score.bm25_raw = Some(
                    text_scores
                        .binary_search_by(|hit| hit.chunk_id.as_str().cmp(id))
                        .map(|index| text_scores[index].score)
                        .unwrap_or(0.0),
                );
            }
            // A missing vector measurement has no numeric interpretation for the
            // collection's metric. Leave its raw and normalized fields absent and
            // supply zero only when forming the sum.
            if score.vector_raw.is_none() {
                if let Ok(index) =
                    vector_scores.binary_search_by(|hit| hit.chunk_id.as_str().cmp(id))
                {
                    score.vector_raw = Some(vector_scores[index].raw_score);
                    score.vector_norm = Some(vector_scores[index].normalized_score);
                }
            }
            // Recompute the lexical fraction after completion as a vector-recalled
            // chunk may just have acquired its BM25 measurement. Use the same union
            // maximum for every candidate, including those that will later be
            // evicted.
            score.bm25_norm = if bm25_max > 0.0 {
                (score.bm25_raw.unwrap_or(0.0) / bm25_max).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        // Rank only after score completion and the metadata boost. A candidate near
        // the edge of the retained heap can beat the current worst only after its
        // other component has been taken into account.
        let boost = metadata_boost(id);
        score.metadata_boost = boost.boost;
        score.matched_boost_rule_count = boost.matched_rule_count;
        // This is the first point at which the configured weights are applied to
        // all available components. A missing vector contributes zero to the
        // weighted sum, while output fields retain its absence for diagnostics.
        score.final_score = weights.bm25 * score.bm25_norm
            + weights.vector * score.vector_norm.unwrap_or(0.0)
            + weights.metadata * score.metadata_boost;
        // The ordering makes the lowest score the heap maximum. At equal scores a
        // smaller chunk ID can displace the current worst, preserving the same
        // cutoff and order as a full sort.
        if candidates.len() < capacity {
            candidates.push(score);
        // The cutoff is deterministic even across equal scores. Once the heap is
        // full, compare against its current worst entry and replace only if the
        // new candidate would precede it under the same ordering used for the
        // returned array.
        } else if candidates.peek().is_some_and(|worst| score < *worst) {
            *candidates.peek_mut().unwrap() = score;
        }
    }
    // Reuse the admitted heap allocation for the best-first array. Only that
    // allocation escapes with the scores. The index permutation was temporary and
    // its reservation ends here.
    Ok(FusionResult {
        hits: ReservedVec::from_reserved(candidates.into_sorted_vec(), work),
        candidate_count: unique,
        bm25_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn assert_fields(actual: &CandidateScore<'_>, expected: &CandidateScore<'_>) {
        assert_eq!(actual.chunk_id, expected.chunk_id);
        assert_eq!(
            actual.bm25_raw.map(f64::to_bits),
            expected.bm25_raw.map(f64::to_bits)
        );
        assert_eq!(actual.bm25_norm.to_bits(), expected.bm25_norm.to_bits());
        assert_eq!(
            actual.vector_raw.map(f64::to_bits),
            expected.vector_raw.map(f64::to_bits)
        );
        assert_eq!(
            actual.vector_norm.map(f64::to_bits),
            expected.vector_norm.map(f64::to_bits)
        );
        assert_eq!(
            actual.metadata_boost.to_bits(),
            expected.metadata_boost.to_bits()
        );
        assert_eq!(actual.final_score.to_bits(), expected.final_score.to_bits());
        assert_eq!(
            actual.matched_boost_rule_count,
            expected.matched_boost_rule_count
        );
    }

    #[test]
    fn fusion_preserves_duplicate_branch_order_and_borrows_ids() {
        for seed in 0..100 {
            let count = 1 + seed % 31;
            let text = (0..count * 3)
                .rev()
                .map(|i| Bm25Hit {
                    chunk_id: format!("c{}", (i * 7 + seed) % count),
                    score: ((i + seed) % 23) as f64 - 2.0,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let vector = (0..count * 4)
                .map(|i| VectorHit {
                    chunk_id: format!("c{}", (i * 11 + seed) % (count + 2)),
                    raw_score: i as f64,
                    normalized_score: match (i + seed) % 5 {
                        0 => -0.0,
                        1 => 0.0,
                        _ => (i % 7) as f64 - 3.0,
                    },
                })
                .collect::<Vec<_>>();
            let weights = match seed % 4 {
                0 => SearchWeights::new(1.0, 0.0, 0.0),
                1 => SearchWeights::new(0.0, 1.0, 0.0),
                2 => SearchWeights::new(0.0, 0.0, 1.0),
                _ => SearchWeights::new(0.25, 0.5, 0.25),
            }
            .unwrap();
            let maximum = text.iter().map(|hit| hit.score).fold(0.0_f64, f64::max);
            let mut reference = BTreeMap::<&str, CandidateScore<'_>>::new();
            for hit in &text {
                let score = reference.entry(&hit.chunk_id).or_default();
                score.chunk_id = &hit.chunk_id;
                score.bm25_raw = Some(hit.score);
                score.bm25_norm = if maximum > 0.0 {
                    (hit.score / maximum).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
            for hit in &vector {
                let score = reference.entry(&hit.chunk_id).or_default();
                score.chunk_id = &hit.chunk_id;
                score.vector_raw = Some(hit.raw_score);
                score.vector_norm = Some(hit.normalized_score);
            }
            let boost = |id: &str| {
                let number = id[1..].parse::<usize>().unwrap();
                MetadataBoostScore {
                    boost: ((number + seed) % 7) as f64 / 4.0,
                    matched_rule_count: number % 3,
                }
            };
            let mut expected = reference
                .into_values()
                .map(|mut score| {
                    let metadata = boost(score.chunk_id);
                    score.metadata_boost = metadata.boost;
                    score.matched_boost_rule_count = metadata.matched_rule_count;
                    score.final_score = weights.bm25 * score.bm25_norm
                        + weights.vector * score.vector_norm.unwrap_or_default()
                        + weights.metadata * score.metadata_boost;
                    score
                })
                .collect::<Vec<_>>();
            expected.sort_by(|a, b| {
                b.final_score
                    .partial_cmp(&a.final_score)
                    .unwrap()
                    .then_with(|| a.chunk_id.cmp(b.chunk_id))
            });
            for limit in [0, 1, 3, count, usize::MAX] {
                let mut visited = Vec::new();
                let actual = fuse(
                    &text,
                    &vector,
                    weights,
                    limit,
                    &JsonOperationContext::standalone(),
                    |id| {
                        visited.push(id.to_owned());
                        boost(id)
                    },
                )
                .unwrap();
                assert_eq!(actual.as_slice().len(), expected.len().min(limit));
                if limit == 0 {
                    assert!(visited.is_empty());
                } else {
                    let mut ids = expected
                        .iter()
                        .map(|score| score.chunk_id)
                        .collect::<Vec<_>>();
                    ids.sort_unstable();
                    assert_eq!(visited, ids);
                }
                for (actual, expected) in actual.as_slice().iter().zip(&expected) {
                    assert_fields(actual, expected);
                    assert!(text
                        .iter()
                        .map(|hit| &hit.chunk_id)
                        .chain(vector.iter().map(|hit| &hit.chunk_id))
                        .any(|id| id.as_ptr() == actual.chunk_id.as_ptr()
                            && id.len() == actual.chunk_id.len()));
                }
            }
        }
    }

    #[test]
    fn fusion_admits_exact_scratch_and_retains_only_the_ranked_array() {
        use crate::json::WorkLedger;
        for seed in 0..100 {
            let count = 2 + seed % 19;
            let text = (0..count)
                .map(|i| Bm25Hit {
                    chunk_id: format!("{}-{i}", "long-borrowed-id".repeat(seed + 1)),
                    score: i as f64,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let vector = text
                .iter()
                .rev()
                .map(|hit| VectorHit {
                    chunk_id: hit.chunk_id.clone(),
                    raw_score: 1.0,
                    normalized_score: 1.0,
                })
                .collect::<Vec<_>>();
            let keep = 1 + seed % count;
            let indices_bytes = (text.len() + vector.len()) * std::mem::size_of::<usize>();
            let result_bytes = keep * std::mem::size_of::<CandidateScore<'_>>();
            let needed = indices_bytes + result_bytes;
            let limit = needed + 73;
            let ledger = std::sync::Arc::new(WorkLedger::with_limits(limit, limit, limit));
            let context = JsonOperationContext::new(ledger.clone());
            let other = JsonOperationContext::new(ledger);
            let mut held = other.workspace_reservation();
            for available in [indices_bytes - 1, needed - 1, needed] {
                held.retain(0);
                held.grow_to(limit - available, limit - available, "operation_scratch")
                    .unwrap();
                let mut visits = 0;
                let result = fuse(
                    &text,
                    &vector,
                    SearchWeights::default_for(true, true),
                    keep,
                    &context,
                    |_| {
                        visits += 1;
                        MetadataBoostScore::default()
                    },
                );
                if available < needed {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    let detail = error.detail().iter().cloned().collect::<BTreeMap<_, _>>();
                    assert_eq!(detail["resource"], "memory");
                    assert_eq!(detail["operation"], "operation_scratch");
                    assert_eq!(detail["limit_bytes"], limit.to_string());
                    assert_eq!(detail["requested_bytes"], (limit + 1).to_string());
                    assert_eq!(visits, 0);
                } else {
                    let hits = result.unwrap();
                    assert_eq!(visits, count);
                    assert_eq!(hits.as_slice().len(), keep);
                    let mut spare = other.workspace_reservation();
                    assert_eq!(
                        spare.grow_to(0, limit, "operation_scratch").unwrap(),
                        indices_bytes
                    );
                    let error = spare
                        .grow_to(indices_bytes + 1, indices_bytes + 1, "operation_scratch")
                        .unwrap_err();
                    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
                    drop(hits);
                    assert_eq!(
                        spare.grow_to(needed, needed, "operation_scratch").unwrap(),
                        needed
                    );
                }
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                assert!(fuse(
                    &text,
                    &vector,
                    SearchWeights::default_for(true, true),
                    0,
                    &context,
                    |_| panic!("zero limit must not evaluate boosts")
                )
                .unwrap()
                .as_slice()
                .is_empty());
                assert!(fuse(
                    &[],
                    &[],
                    SearchWeights::default_for(true, true),
                    usize::MAX,
                    &context,
                    |_| panic!("empty input must not evaluate boosts")
                )
                .unwrap()
                .as_slice()
                .is_empty());
            }
        }
    }

    #[test]
    fn bounded_fusion_matches_independent_scores_and_full_order_across_100_seeds() {
        for seed in 0..100 {
            let count = seed % 41;
            let ids = (0..count)
                .map(|id| format!("chunk-{id}"))
                .collect::<Vec<_>>();
            let weights = match seed % 4 {
                0 => SearchWeights::new(1.0, 0.0, 0.0),
                1 => SearchWeights::new(0.0, 1.0, 0.0),
                2 => SearchWeights::new(0.0, 0.0, 1.0),
                _ => SearchWeights::new(0.25, 0.5, 0.25),
            }
            .unwrap();
            let has_text = |id| (id + seed) % 3 != 0;
            let has_vector = |id| (id + seed) % 4 != 0;
            let has_boost = |id| (id + seed) % 5 != 0;
            let text_score = |id| ((id * 7 + seed) % 9) as f64;
            let vector_score = |id| ((id * 3 + seed) % 9) as f64 / 8.0;
            let boost_score = |id| ((id + seed) % 7) as f64 / 4.0;
            let text = (0..count)
                .rev()
                .filter(|&id| has_text(id))
                .map(|id| Bm25Hit {
                    chunk_id: format!("chunk-{id}"),
                    score: text_score(id),
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let vector = (0..count)
                .filter(|&id| has_vector(id))
                .map(|id| VectorHit {
                    chunk_id: format!("chunk-{id}"),
                    raw_score: 2.0 * vector_score(id) - 1.0,
                    normalized_score: vector_score(id),
                })
                .collect::<Vec<_>>();
            let boosts: BTreeMap<_, _> = (0..count)
                .filter(|&id| has_boost(id))
                .map(|id| (format!("chunk-{id}"), boost_score(id)))
                .collect();
            let maximum = (0..count)
                .filter(|&id| has_text(id))
                .map(text_score)
                .fold(0.0_f64, f64::max);
            let mut expected = (0..count)
                .filter(|&id| has_text(id) || has_vector(id))
                .map(|id| {
                    let bm25_norm = if has_text(id) && maximum > 0.0 {
                        text_score(id) / maximum
                    } else {
                        0.0
                    };
                    let vector_norm = has_vector(id).then(|| vector_score(id));
                    let metadata_boost = if has_boost(id) { boost_score(id) } else { 0.0 };
                    CandidateScore {
                        chunk_id: &ids[id],
                        bm25_recalled: has_text(id),
                        vector_recalled: has_vector(id),
                        bm25_raw: has_text(id).then(|| text_score(id)),
                        bm25_norm,
                        vector_raw: has_vector(id).then(|| 2.0 * vector_score(id) - 1.0),
                        vector_norm,
                        metadata_boost,
                        matched_boost_rule_count: usize::from(has_boost(id)),
                        final_score: weights.bm25 * bm25_norm
                            + weights.vector * vector_norm.unwrap_or(0.0)
                            + weights.metadata * metadata_boost,
                    }
                })
                .collect::<Vec<_>>();
            expected.sort_by(|a, b| {
                b.final_score
                    .total_cmp(&a.final_score)
                    .then_with(|| a.chunk_id.cmp(b.chunk_id))
            });
            for limit in [0, 1, 7, count, usize::MAX] {
                let actual = fuse(
                    &text,
                    &vector,
                    weights,
                    limit,
                    &JsonOperationContext::standalone(),
                    |id| MetadataBoostScore {
                        boost: boosts.get(id).copied().unwrap_or_default(),
                        matched_rule_count: usize::from(boosts.contains_key(id)),
                    },
                )
                .unwrap();
                let actual = actual.as_slice();
                assert_eq!(actual.len(), expected.len().min(limit), "seed={seed}");
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_fields(actual, expected);
                }
            }
        }
    }
}
