use super::{rank_to_json, ExpectedChunkLookup, WhyNotRanks};
use crate::error::{Error, ErrorCode, Result};
use crate::json::{JsonBuilder, JsonValue};
use crate::query::SearchQueryContext;

#[derive(Clone, Copy)]
// Carry the caller's choice of key namespace through both the ordinary result
// lookup and diagnostic replay. An external alias can have the spelling of a
// generated internal ID and resolve to a different chunk.
pub(crate) enum DiagnosisTarget<'a> {
    // Prefer the internal identity when resolving an explicitly typed target. A
    // record found outside the requested collection can still provide a
    // collection-mismatch diagnosis.
    Internal(&'a str),
    // Give an external alias priority, including a diagnostic lookup outside the
    // requested collection if no local alias matches.
    External(&'a str),
    // The untyped interface retains its collection-local lookup order. Remember
    // that the caller supplied no namespace when reconciling the two possible
    // ranks.
    Legacy(&'a str),
}

impl<'a> DiagnosisTarget<'a> {
    // Copying the target copies only the namespace tag and the borrowed string
    // slice. Use the unmodified spelling when displaying a missing target or
    // matching the ordinary search page, then apply the lookup rules when
    // resolving its identity.
    pub(super) fn value(self) -> &'a str {
        match self {
            Self::Internal(id) | Self::External(id) | Self::Legacy(id) => id,
        }
    }
}

/// Route ranks come from the point where that route actually truncates its input.
fn route(
    requested: bool,
    blocked: Option<&str>,
    rank: Option<usize>,
    limit: usize,
    absent: &str,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    // A rank is meaningful as a cutoff explanation only for a requested route with
    // an eligible target. An unavailable or filtered target keeps the shared
    // blocking reason even if the caller supplies a rank.
    let status = match (requested, blocked, rank) {
        (false, _, _) => "not_requested",
        (true, Some(reason), _) => reason,
        // Ranks are one-based and the boundary itself is eligible. The absent
        // label is route-specific because a text row with no match and a vector
        // row with no usable embedding have different causes.
        (true, None, Some(n)) if n > limit => "outside_limit",
        (true, None, Some(_)) => "candidate",
        (true, None, None) => absent,
    };
    builder.object([
        ("status", builder.string(status)?),
        // Keep the measured rank independent of the displayed status. A disabled
        // route has no applicable limit, which is represented by null rather than
        // the numeric cutoff from the active profile.
        ("rank", rank_to_json(rank)),
        ("limit", rank_to_json(requested.then_some(limit))),
    ])
}

pub(super) fn outcome(
    target: &ExpectedChunkLookup<'_>,
    matched: bool,
    vector_absence: Option<&str>,
    returned: Option<usize>,
    ranks: &WhyNotRanks,
    query: &SearchQueryContext,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    let limits = query.candidate_limits;
    // Use the ordinary search result as the authority for membership. A replay rank
    // that would have returned an absent target signals inconsistent diagnostics.
    // Check the resolved target's availability before interpreting any rank. The
    // ordinary search ran before this lookup, so a saved position cannot explain a
    // chunk that now belongs to another collection or has been deleted.
    let (stage, reason) = if !target.exists {
        ("unavailable", "not_found")
    } else if !target.collection_matches {
        ("unavailable", "collection_mismatch")
    } else if target.deleted {
        ("unavailable", "deleted")
    // Apply the metadata verdict before reporting retrieval or ranking. A filtered
    // target can match terms or have a valid embedding and still be ineligible for
    // the returned page.
    } else if !matched {
        ("filtered", "metadata_filter")
    // Use the rank recorded for the resolved ID from the actual result page. A
    // returned row does not need the diagnostic search to reproduce its position
    // to establish membership.
    } else if returned.is_some() {
        ("returned", "returned")
    } else {
        // These ranks are one-based. Attribute a missing row to the earliest
        // limiting stage: the fusion window cuts the ranked union first, then
        // top_k selects the public page from the retained candidates.
        match ranks.fusion_rank {
            // No fusion rank means the target never entered the ranked candidate
            // union. Without that rank, neither the fusion cutoff nor the final
            // output limit explains the omission.
            None => ("not_candidate", "no_candidate"),
            Some(n) if n > limits.fusion => ("not_returned", "fusion_limit"),
            Some(n) if n > query.top_k => ("not_returned", "below_top_k"),
            // The ordinary result omitted an otherwise eligible target ranked
            // inside both cutoffs. Surface the inconsistency rather than returning
            // a plausible-looking explanation that contradicts the actual page.
            Some(_) => {
                return Err(Error::new(
                    ErrorCode::RecoveryError,
                    "diagnostic rank conflicts with ordinary search results",
                ))
            }
        }
    };
    // An unavailable or filtered target blocks both retrieval routes. The other
    // overall outcomes leave each route free to report its own rank, cutoff, or
    // reason for supplying no candidate.
    let blocked = match stage {
        "unavailable" | "filtered" => Some(stage),
        _ => None,
    };
    // Exact vector search can rank every eligible visible embedding of the correct
    // dimension. For a requested route, absence of a rank needs a concrete cause,
    // such as a missing embedding or a dimension mismatch.
    if query.embedding_present
        && blocked.is_none()
        && ranks.vector_rank.is_none()
        && vector_absence.is_none()
    {
        return Err(Error::new(
            ErrorCode::RecoveryError,
            "no diagnostic rank for a visible target with a valid embedding",
        ));
    }
    builder.object([
        ("stage", builder.string(stage)?),
        ("reason", builder.string(reason)?),
        (
            "bm25",
            // Text retrieval applies its limit before the common metadata filter.
            // Use the rank at that boundary when explaining whether the text
            // window could have supplied this target.
            route(
                query.text_present,
                blocked,
                ranks.bm25_retrieval_rank,
                limits.bm25,
                "no_match",
                builder,
            )?,
        ),
        (
            "exact",
            // Vector eligibility includes the filter before the vector limit.
            // Supply the exact route's own rank and the caller's embedding
            // diagnosis so a missing vector cannot be mistaken for a rank below
            // the retrieval window.
            route(
                query.embedding_present,
                blocked,
                ranks.vector_rank,
                limits.vector,
                vector_absence.unwrap_or("unavailable"),
                builder,
            )?,
        ),
    ])
}
