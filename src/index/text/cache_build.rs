use super::{cache::TextCacheOwner, TextCacheBuild, TextIndex};
use crate::json::JsonOperationContext;
use crate::storage::{catalog_id_text, TextSegmentEntry, TextSegmentVisitor};
use crate::{Error, ErrorCode, Result};
use std::sync::Arc;

impl TextIndex {
    pub(crate) fn try_cache_from_visitors<'a>(
        collections: impl IntoIterator<Item = &'a str>,
        context: &JsonOperationContext,
        limit: usize,
        mut visit: impl FnMut(&str, &mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<TextCacheBuild> {
        // Build into a private candidate so a later collection cannot leave an
        // earlier collection's postings installed after a validation error. An
        // empty input still yields a valid resident index.
        let mut index = Self {
            cache: Some(TextCacheOwner::new(context)),
            ..Self::default()
        };
        for id in collections {
            // Share one collection ID across its postings. Keep this handle local
            // to the visit until its corpus statistics can take ownership of the
            // same allocation.
            let mut owner = None::<Arc<str>>;
            let mut expected = None;
            let mut chunks = 0usize;
            let mut total = 0usize;
            if !visit(id, &mut |entry| {
                match entry {
                    // The header's count is a claim about this collection.
                    // Accumulate the observed count and token lengths from chunk
                    // events, independently of how many terms or postings appear.
                    TextSegmentEntry::Header { chunks } => expected = Some(chunks),
                    TextSegmentEntry::Chunk { length, .. } => {
                        chunks = chunks.checked_add(1).ok_or_else(cache_count_overflow)?;
                        total = total
                            // Give an empty chunk a length of one both here and in
                            // the length row below. Otherwise the corpus average
                            // and the individual length used to score that same
                            // chunk would disagree.
                            .checked_add(length.max(1))
                            .ok_or_else(cache_count_overflow)?;
                    }
                    _ => {}
                }
                // Even after cache admission fails, finish visiting the source and
                // checking its declared chunk count before reporting a cache miss.
                if !index.has_resident_cache() {
                    return Ok(());
                }
                let keep = match entry {
                    TextSegmentEntry::Header { .. } => true,
                    TextSegmentEntry::Chunk { id, length } => {
                        // Segment events carry compact numeric chunk IDs. Expand
                        // them into the catalog's canonical text in stack storage
                        // for the duration of the insertion; the cache makes its
                        // own charged key copy.
                        let mut bytes = [0; 24];
                        index.cache_chunk(
                            catalog_id_text(&mut bytes, b"chk_", id),
                            length.max(1),
                            limit,
                        )?
                    }
                    // The dictionary declares the number of postings still to come
                    // for this term in this collection. Reserve for that
                    // contribution even when other collections already use the
                    // same term.
                    TextSegmentEntry::Term {
                        text,
                        document_frequency,
                    } => index.cache_term(text, document_frequency, limit)?,
                    TextSegmentEntry::Posting {
                        term,
                        chunk,
                        frequency,
                    } => {
                        // Wait for the first posting before allocating a shared
                        // collection ID. A collection containing only empty or
                        // unsearchable text still needs lengths and statistics but
                        // has no posting that needs this Arc.
                        if owner.is_none() {
                            owner = index.cache_collection_owner(id, limit);
                        }
                        if let Some(owner) = &owner {
                            let mut bytes = [0; 24];
                            index.cache_posting(
                                term,
                                catalog_id_text(&mut bytes, b"chk_", chunk),
                                frequency,
                                owner,
                                limit,
                            )?
                        } else {
                            false
                        }
                    }
                };
                if !keep {
                    // A collection-local Arc must drop before the candidate's credit.
                    owner = None;
                    // A refusal in one collection discards the entire candidate,
                    // including previously visited collections. The remaining
                    // events still contribute to validation without allocating
                    // replacement cache entries.
                    index.clear_cache();
                }
                Ok(())
            })? {
                // An unsuccessful source visit invalidates the candidate even if
                // an earlier event exhausted its cache allowance. Propagated
                // visitor errors retain their own error code.
                return Ok(TextCacheBuild::Invalid);
            }
            // Checking against Some also catches a missing header when zero chunks
            // were observed. Matching two numeric zeros alone would accept that
            // incomplete visit.
            if expected != Some(chunks) {
                return Err(Error::new(
                    ErrorCode::IndexCorruption,
                    "text segment chunk count mismatch",
                ));
            }
            // Publish statistics only for a nonempty collection. Its positive
            // chunk count supplies the divisor for the average; the cache's lookup
            // already counts an absent collection as empty.
            if chunks != 0
                && index.has_resident_cache()
                && !index.cache_collection(id, chunks, total, owner.take(), limit)?
            {
                index.clear_cache();
            }
        }
        // A valid segment can exceed the optional resident-cache allowance. Return
        // Miss so queries can still stream it. Invalid above means the segment visit
        // itself could not be used as an index source.
        if index.has_resident_cache() {
            // Collections arrive in caller-supplied order. Order each term's
            // combined postings by collection and chunk before exposing the
            // binary-searchable resident ranges.
            index.sort_postings();
            Ok(TextCacheBuild::Complete(index))
        } else {
            Ok(TextCacheBuild::Miss)
        }
    }

    pub(crate) fn restore_empty_terms_from_visitors<'a>(
        &mut self,
        collections: impl IntoIterator<Item = &'a str>,
        limit: usize,
        mut visit: impl FnMut(&str, &mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<bool> {
        let result = (|| {
            // Replacing a collection's postings can remove the last entry for a
            // term. Recover zero-frequency terms recorded in the prepared checkpoint
            // as well, so the cache retains the same vocabulary as the source.
            // Scan every live collection in the prepared checkpoint. Removing the
            // last posting can erase a term shared with a dictionary-only entry in
            // an untouched collection.
            for id in collections {
                if !visit(id, &mut |entry| {
                    if let TextSegmentEntry::Term {
                        text,
                        document_frequency: 0,
                    } = entry
                    {
                        // A zero declaration requests no additional posting slots.
                        // A term absent from the resident dictionary can still
                        // need a key and a tree node. Evict the optional cache if
                        // those allocations cannot fit.
                        if self.has_resident_cache() && !self.cache_term(text, 0, limit)? {
                            self.clear_cache();
                        }
                    }
                    // Keep consuming the source after any cache miss. A valid body
                    // with no resident result differs from a truncated or invalid
                    // prepared checkpoint, and the visitor's final validity result
                    // still decides that distinction.
                    Ok(())
                })? {
                    return Err(Error::new(
                        ErrorCode::IndexCorruption,
                        "incomplete prepared text checkpoint",
                    ));
                }
            }
            Ok(self.has_resident_cache())
        })();
        // Parts of the vocabulary may have been restored before a later visitor
        // fails. Clear that partial cache as a unit rather than exposing a
        // vocabulary for only the successfully visited collections.
        if result.is_err() {
            self.clear_cache();
        }
        result
    }
}

fn cache_count_overflow() -> Error {
    Error::new(ErrorCode::IndexCorruption, "text token count overflow")
}
