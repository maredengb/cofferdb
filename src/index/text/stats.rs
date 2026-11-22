#[cfg(test)]
use super::TextIndex;
use crate::catalog::Catalog;
use crate::error::{Error, ErrorCode, Result};
use crate::json::{JsonOperationContext, ReservedVec};
use crate::storage::catalog_id_number;
use crate::storage::{IndexSegmentSource, Storage, TextSegmentEntry, TextSegmentVisitor};
use crate::temporary::{ExternalTermFrequencies, TemporaryDiskBudget};
use std::sync::Arc;

// The database-wide vocabulary cannot be recovered by summing the collection
// counts. A term can occur in several collections or appear in a dictionary with
// an empty posting list.
#[derive(Debug)]
pub(crate) struct TextStatsSummary {
    // Count of distinct term strings across all live collections' segment
    // dictionaries, including terms with empty posting lists.
    terms: usize,
    // True if any visited dictionary lists a term with zero postings. Cache
    // replacement uses this fact when restoring empty dictionary entries.
    zero_frequency_terms: bool,
    // Operation-owned array sorted by numeric collection ID. Each entry counts only
    // the distinct terms with postings in that collection.
    collections: ReservedVec<(u64, usize)>,
}

impl TextStatsSummary {
    // Publish the summary only if every live collection could be visited. A
    // missing source makes even the partial global term count unusable, while
    // failures in decoding or the external accumulator propagate to the caller.
    pub(crate) fn from_source(
        source: &IndexSegmentSource,
        storage: &Storage,
        catalog: &Catalog,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
    ) -> Result<Option<Self>> {
        // Keep the operation-owned system temp path across attempts. The external
        // accumulator can spill intermediate term keys there while charging its
        // temporary files to the supplied disk budget.
        let directory = crate::temporary::TemporaryPath::system_temp_in(context)?;
        let mut input = context.workspace_reservation();
        input.grow_to(0, 64 * 1024, "operation_scratch")?;
        // The optional read buffer and mandatory statistics work share the same
        // operation budget. A retry without the buffer can still read a valid
        // source when the buffered pass runs out of memory.
        loop {
            // A retry starts with a fresh term accumulator and collection-count
            // vector. The failed read has already dropped both, so terms seen
            // before a budget error cannot be counted twice.
            let result = Self::read_source(source, storage, catalog, &directory, disk, &input);
            match result {
                Err(error)
                    if crate::json::is_memory_budget_error(&error)
                        && input.reserved_bytes() > 0 =>
                {
                    // Release only the read reservation. The next attempt may
                    // still require space for its collection array, temporary path
                    // and accumulator even when decoding runs without a buffer.
                    input.retain(0);
                }
                // A missing segment and a temporary-disk error bypass the buffer
                // retry. Releasing memory cannot supply the absent source or
                // enlarge the disk budget.
                result => return result,
            }
        }
    }

    fn read_source(
        source: &IndexSegmentSource,
        storage: &Storage,
        catalog: &Catalog,
        directory: &std::path::Path,
        disk: &Arc<TemporaryDiskBudget>,
        input: &crate::json::WorkspaceReservation,
    ) -> Result<Option<Self>> {
        // All collection visits borrow the same read allowance. Pass its operation
        // context into the accumulator so decoding and global deduplication share
        // the same workspace budget.
        Self::from_visitors(catalog, directory, disk, input.context(), |id, visit| {
            source.visit_text(storage, id, input, visit)
        })
    }

    pub(crate) fn from_visitors(
        catalog: &Catalog,
        directory: &std::path::Path,
        disk: &Arc<TemporaryDiskBudget>,
        context: &JsonOperationContext,
        mut read: impl FnMut(&str, &mut TextSegmentVisitor<'_>) -> Result<bool>,
    ) -> Result<Option<Self>> {
        // Keep one compact count per live collection. The dictionary itself can be
        // much larger, so use an external sorter for the global vocabulary rather
        // than holding every term in the returned summary.
        let mut collections = ReservedVec::with_capacity(
            catalog.live_collections().count(),
            context,
            "operation_scratch",
        )?;
        let mut frequencies = ExternalTermFrequencies::new(directory, disk.clone(), context);
        let mut zero_frequency_terms = false;
        // Reset the local count on each live collection. The sorter stays open
        // across those visits so identical term bytes from several collections
        // collapse to one global vocabulary key.
        for collection in catalog.live_collections() {
            let mut count = 0usize;
            if !read(&collection.collection_id, &mut |entry| {
                if let TextSegmentEntry::Term {
                    text,
                    document_frequency,
                } = entry
                {
                    // The global count is a union of dictionary keys, including empty
                    // posting lists. Each collection counts only terms with postings.
                    frequencies.push(&[text.as_bytes()], 1)?;
                    if document_frequency != 0 {
                        count = count.checked_add(1).ok_or_else(stats_overflow)?;
                    } else {
                        zero_frequency_terms = true;
                    }
                }
                Ok(())
            })? {
                // A visitor returning false leaves an unusable checkpoint source.
                // The sorter can already hold provisional dictionary keys, so
                // discard this entire operation and let the caller choose another
                // path.
                return Ok(None);
            }
            // Attach the count only after validating that collection's body.
            // Record the numeric ID that the readers and scanners use on the wire
            // instead of retaining another copy of the formatted catalog ID.
            collections.push((catalog_id_number(&collection.collection_id)?, count));
        }
        // Sort once for the later per-collection binary search. Repeated numeric
        // IDs would make the lookup ambiguous, so report a format error before
        // publishing the summary.
        let counts = collections.as_mut_slice();
        counts.sort_unstable_by_key(|&(id, _)| id);
        // If two collection spellings encode the same numeric ID, no deterministic
        // binary search can tell them apart. Reject the ambiguous summary before
        // using it to qualify a resident cache.
        if counts.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "duplicate collection in text statistics",
            ));
        }
        // The external frequency accumulator groups identical term bytes across
        // collections. Count each emitted key once regardless of the accumulated
        // frequency.
        let mut terms = 0usize;
        frequencies.finish_batch(|_, _| {
            terms = terms.checked_add(1).ok_or_else(stats_overflow)?;
            Ok(())
        })?;
        Ok(Some(Self {
            terms,
            zero_frequency_terms,
            collections,
        }))
    }

    #[cfg(test)]
    pub(crate) fn build(
        index: &TextIndex,
        catalog: &Catalog,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let mut collections = ReservedVec::with_capacity(
            catalog.live_collections().count(),
            context,
            "operation_scratch",
        )?;
        for collection in catalog.live_collections() {
            collections.push((catalog_id_number(&collection.collection_id)?, 0));
        }
        let counts = collections.as_mut_slice();
        counts.sort_unstable_by_key(|&(id, _)| id);
        if counts.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "duplicate collection in text statistics",
            ));
        }
        for postings in index.postings.values() {
            let mut remaining = postings.as_slice();
            while let Some(posting) = remaining.first() {
                let id = posting.collection_id.as_ref();
                if let Ok(position) =
                    counts.binary_search_by_key(&catalog_id_number(id)?, |&(id, _)| id)
                {
                    counts[position].1 += 1;
                }
                let end = remaining.partition_point(|posting| posting.collection_id.as_ref() == id);
                remaining = &remaining[end..];
            }
        }
        Ok(Self {
            // Empty posting lists still belong to the global dictionary count.
            terms: index.term_count(),
            zero_frequency_terms: index.postings().values().any(Vec::is_empty),
            collections,
        })
    }

    pub(crate) fn term_count(&self) -> usize {
        self.terms
    }

    pub(crate) fn has_zero_frequency_terms(&self) -> bool {
        self.zero_frequency_terms
    }

    // An unparseable numeric suffix or an absent numeric ID yields zero. Public
    // collection statistics separately resolve the actual catalog collection.
    pub(crate) fn term_count_in_collection(&self, collection_id: &str) -> usize {
        let Ok(id) = catalog_id_number(collection_id) else {
            return 0;
        };
        let counts = self.collections.as_slice();
        counts
            .binary_search_by_key(&id, |&(id, _)| id)
            .map_or(0, |position| counts[position].1)
    }

    #[cfg(test)]
    pub(crate) fn assert_operation_context(&self, context: &JsonOperationContext) {
        self.collections.assert_operation_context(context);
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.collections.reserved_bytes()
    }

    #[cfg(test)]
    pub(crate) fn try_clone_in(&self, context: &JsonOperationContext) -> Result<Self> {
        let mut collections = ReservedVec::with_capacity(
            self.collections.as_slice().len(),
            context,
            "operation_scratch",
        )?;
        for &entry in self.collections.as_slice() {
            collections.push(entry);
        }
        Ok(Self {
            terms: self.terms,
            zero_frequency_terms: self.zero_frequency_terms,
            collections,
        })
    }
}

fn stats_overflow() -> Error {
    Error::new(ErrorCode::FileFormatError, "text statistic count overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::text::Posting;
    use crate::json::WorkLedger;
    use crate::model::{CollectionInfo, DistanceMetric};
    use std::sync::Arc;

    fn collection(id: &str) -> CollectionInfo {
        CollectionInfo {
            collection_id: id.into(),
            name: id.into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            tokenizer: Default::default(),
            vectors_normalized: false,
            vector_index: Default::default(),
            created_at: String::new(),
        }
    }

    fn fixture(seed: usize) -> (TextIndex, Catalog) {
        let mut index = TextIndex::default();
        let mut catalog = Catalog::default();
        let ids = [
            1,
            2,
            10,
            9_999_999_999_999_999,
            10_000_000_000_000_000,
            u64::MAX,
        ];
        for (owner, id) in ids.into_iter().enumerate() {
            let id = format!("col_{id:016}");
            catalog.collections.insert(id.clone(), collection(&id));
            for term in 0..seed {
                for occurrence in 0..(seed + term * (owner + 1)) % 4 {
                    index
                        .postings
                        .entry(format!("term-{term}"))
                        .or_default()
                        .push(Posting {
                            collection_id: id.as_str().into(),
                            chunk_id: format!("{owner}-{occurrence}"),
                            term_frequency: 1,
                        });
                }
            }
        }
        let empty = "col_0000000000000003";
        catalog.collections.insert(empty.into(), collection(empty));
        catalog
            .deleted_collections
            .insert("col_0000000000000002".into());
        index
            .postings
            .insert("empty dictionary key".into(), Vec::new());
        index.sort_postings();
        (index, catalog)
    }

    fn assert_available(context: &JsonOperationContext, bytes: usize) {
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, "operation_scratch").unwrap();
        assert!(work
            .grow_to(bytes + 1, bytes + 1, "operation_scratch")
            .is_err());
    }

    #[test]
    fn summary_counts_dictionary_union_and_live_collection_groups_across_id_widths() {
        for seed in 0..100 {
            let (index, catalog) = fixture(seed);
            let context = JsonOperationContext::standalone();
            let summary = TextStatsSummary::build(&index, &catalog, &context).unwrap();
            assert_eq!(summary.term_count(), index.postings.len());
            assert_eq!(
                summary.reserved_bytes(),
                catalog.live_collections().count() * std::mem::size_of::<(u64, usize)>()
            );
            assert!(summary
                .collections
                .as_slice()
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0));
            for collection in catalog.collections.values() {
                let expected = if catalog.is_collection_live(&collection.collection_id) {
                    index
                        .postings
                        .values()
                        .filter(|postings| {
                            postings.iter().any(|posting| {
                                posting.collection_id.as_ref() == collection.collection_id
                            })
                        })
                        .count()
                } else {
                    0
                };
                assert_eq!(
                    summary.term_count_in_collection(&collection.collection_id),
                    expected
                );
            }
            for absent in [
                "col_0000000000000000",
                "col_0000000000000011",
                "invalid",
                "",
            ] {
                assert_eq!(summary.term_count_in_collection(absent), 0);
            }
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let empty =
            TextStatsSummary::build(&TextIndex::default(), &Catalog::default(), &context).unwrap();
        assert_eq!((empty.term_count(), empty.reserved_bytes()), (0, 0));
    }

    #[test]
    fn shared_terms_and_zero_posting_key_are_not_summed_across_collections() {
        let mut catalog = Catalog::default();
        for id in ["col_10", "col_2", "col_1"] {
            catalog.collections.insert(id.into(), collection(id));
        }
        let mut index = TextIndex::default();
        for term in ["shared-one", "shared-two"] {
            for id in ["col_10", "col_1"] {
                for chunk in ["chk_1", "chk_2"] {
                    index
                        .postings
                        .entry(term.into())
                        .or_default()
                        .push(Posting {
                            collection_id: id.into(),
                            chunk_id: chunk.into(),
                            term_frequency: 1,
                        });
                }
            }
        }
        index.postings.insert("zero-posting".into(), Vec::new());
        index.sort_postings();
        let summary =
            TextStatsSummary::build(&index, &catalog, &JsonOperationContext::standalone()).unwrap();
        drop((index, catalog));
        assert_eq!(summary.term_count(), 3);
        assert_eq!(summary.collections.as_slice(), &[(1, 2), (2, 0), (10, 2)]);
    }

    #[test]
    fn summary_admission_holds_exact_capacity_after_context_and_postings_drop() {
        for seed in 0..100 {
            let (index, catalog) = fixture(seed);
            let bytes = catalog.live_collections().count() * std::mem::size_of::<(u64, usize)>();
            let limit = bytes + 64;
            for parent in [false, true] {
                for available in [bytes - 1, bytes, bytes + 1] {
                    let ledger = Arc::new(WorkLedger::with_limits(
                        if parent { 2 * limit } else { limit },
                        limit,
                        if parent { limit } else { 2 * limit },
                    ));
                    let context = JsonOperationContext::new(ledger.clone());
                    let sibling = JsonOperationContext::new(ledger);
                    let occupied = limit - available;
                    let cache =
                        parent.then(|| context.cache_reservation(occupied as u128, limit).unwrap());
                    let mut working = context.workspace_reservation();
                    if !parent {
                        working
                            .grow_to(occupied, occupied, "operation_scratch")
                            .unwrap();
                    }
                    let result = TextStatsSummary::build(&index, &catalog, &context);
                    drop(context);
                    if available < bytes {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".to_owned()),
                            ("operation", "operation_scratch".to_owned()),
                            ("limit_bytes", limit.to_string()),
                            ("requested_bytes", (limit + 1).to_string()),
                        ] {
                            assert!(error.detail().contains(&(key.into(), value)));
                        }
                        assert_available(&sibling, available);
                    } else {
                        let summary = result.unwrap();
                        assert_eq!(summary.reserved_bytes(), bytes);
                        assert_eq!(summary.term_count(), index.term_count());
                        assert_available(&sibling, available - bytes);
                        drop(summary);
                        assert_available(&sibling, available);
                    }
                    drop((cache, working));
                    assert_available(&sibling, limit);
                }
            }
            let context = JsonOperationContext::standalone();
            let summary = TextStatsSummary::build(&index, &catalog, &context).unwrap();
            let expected = summary.term_count();
            drop((index, catalog, context));
            assert_eq!(summary.term_count(), expected);
        }
    }

    #[test]
    fn summary_rejects_invalid_or_duplicate_numeric_ids_without_stranding_credit() {
        for ids in [["col_bad", "col_2"], ["col_1", "col_0000000000000001"]] {
            let mut catalog = Catalog::default();
            for id in ids {
                catalog.collections.insert(id.into(), collection(id));
            }
            let limit = 2 * std::mem::size_of::<(u64, usize)>();
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(limit, 0, limit)));
            let error =
                TextStatsSummary::build(&TextIndex::default(), &catalog, &context).unwrap_err();
            assert_eq!(error.code(), ErrorCode::FileFormatError);
            assert_available(&context, limit);
        }
    }
}
