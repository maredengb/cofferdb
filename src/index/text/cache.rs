use super::{CollectionCounts, Posting, TextIndex};
use crate::json::{btree_workspace, CacheReservation, JsonOperationContext};
use crate::{Error, ErrorCode, Result};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

#[cfg(test)]
mod tests;

// The slot selects the node estimate for one concrete map type. An unused
// allowance in the lengths map cannot cover node growth in the term directory
// while both trees remain allocated.
const TERMS: usize = 0;
const LENGTHS: usize = 1;
const COUNTS: usize = 2;
const AVERAGES: usize = 3;

pub(super) struct TextCacheOwner {
    // Bytes reserved for B-tree nodes in the term, length, count and average maps,
    // indexed by the constants above. Each entry keeps its high-water mark until
    // the cache is dropped.
    nodes: [usize; 4],
    // Bytes charged for allocations outside those nodes, including owned keys,
    // posting buffers and shared collection IDs.
    payload: usize,
    // Holds the sum of the four node estimates and live payload credit against the
    // database cache budget.
    reservation: CacheReservation,
}

impl TextCacheOwner {
    pub(super) fn new(context: &JsonOperationContext) -> Self {
        Self {
            nodes: [0; 4],
            payload: 0,
            // An empty resident index still needs an owner on the database's
            // ledger. The zero-byte reservation distinguishes a usable empty cache
            // from an index with no resident representation.
            reservation: context
                .cache_reservation(0, usize::MAX)
                .expect("empty cache admission"),
        }
    }

    fn admit_node<K, V>(&mut self, tree: usize, count: usize, key: usize, limit: usize) -> bool {
        // Removing a key can free its buffer without immediately freeing a B-tree
        // node. Charge only growth above the existing node estimate and account for
        // the key separately so payload removal can return exactly its share.
        let nodes = btree_workspace::<K, V>(count).max(self.nodes[tree]);
        // Attempt the node and key growth together before changing either
        // in-memory counter. A refusal therefore leaves the existing cache and its
        // accounting intact for the caller.
        if !self
            .reservation
            .try_grow((nodes - self.nodes[tree]) as u128 + key as u128, limit)
        {
            return false;
        }
        self.nodes[tree] = nodes;
        self.payload += key;
        true
    }

    // Posting arrays and shared collection identifiers can grow independently of
    // their B-tree nodes. Charge those allocations as payload so removal can
    // release them without guessing which nodes the tree retained.
    fn admit_payload(&mut self, bytes: u128, limit: usize) -> bool {
        if !self.reservation.try_grow(bytes, limit) {
            return false;
        }
        // Successful admission bounds the u128 request and the owner's total to
        // what the ledger can represent in usize. Keep the narrowing conversion
        // after that check.
        self.payload += bytes as usize;
        true
    }

    fn release_payload(&mut self, bytes: usize) {
        // Retain the node high-water estimates even after a posting or key is
        // freed. Releasing those estimates on every removal would advertise
        // reusable memory while the B-tree could still own the same nodes.
        self.payload -= bytes;
        self.reservation
            .retain(self.nodes.iter().sum::<usize>() + self.payload);
    }

    fn transfer_payload(&mut self, target: &mut Self, bytes: usize) {
        // During a collection replacement both caches already hold credit on the
        // same ledger. Transfer the charge with the allocation to avoid briefly
        // paying twice or making it available to another allocation.
        self.reservation.transfer_to(&mut target.reservation, bytes);
        self.payload -= bytes;
        target.payload += bytes;
    }

    pub(super) fn reserved_bytes(&self) -> usize {
        self.reservation.reserved_bytes()
    }
}

fn arc_bytes(id: &str) -> u128 {
    // A posting clones the Arc rather than allocating another ID. Account for the
    // string bytes, the two reference counts and alignment once per shared
    // collection allocation.
    // Use native word size and alignment here. This is an estimate of the
    // allocation held by the running process, independent of the on-disk encoding
    // of collection IDs.
    let align = std::mem::align_of::<usize>() as u128;
    let bytes = 2 * std::mem::size_of::<usize>() as u128 + id.len() as u128;
    bytes.div_ceil(align) * align
}

fn copy_key(value: &str) -> Result<String> {
    let mut key = String::new();
    key.try_reserve_exact(value.len())
        .map_err(std::io::Error::other)?;
    // The directory charged exactly the requested byte length. Reject a larger
    // returned capacity before publishing the key so later removal can refund the
    // same number of bytes that were charged.
    if key.capacity() != value.len() {
        return Err(std::io::Error::other("text cache key capacity differs").into());
    }
    key.push_str(value);
    Ok(key)
}

impl TextIndex {
    // A resident collection can have no searchable tokens. Test for the cache
    // owner instead of testing whether its postings map is empty when deciding
    // which source serves the query.
    pub(crate) fn has_resident_cache(&self) -> bool {
        self.cache.is_some()
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, TextCacheOwner::reserved_bytes)
    }

    pub(crate) fn clear_cache(&mut self) -> bool {
        // Drop the postings and their shared IDs, the length rows, and both
        // statistics maps before releasing the aggregate cache reservation. The
        // return value reports whether clearing these owners freed charged bytes.
        let bytes = self.reserved_bytes();
        self.postings = BTreeMap::new();
        self.chunk_lengths = BTreeMap::new();
        self.total_chunks_by_collection = BTreeMap::new();
        self.avg_len_by_collection = BTreeMap::new();
        self.cache = None;
        bytes != 0
    }

    pub(crate) fn collection_chunk_count(&self, id: &str) -> usize {
        // This count is the collection's BM25 population. A collection absent from
        // the statistics contributes zero chunks, even when other collections
        // share this resident index.
        self.total_chunks_by_collection
            .get(id)
            .map_or(0, |counts| counts.chunks)
    }

    pub(super) fn cache_chunk(&mut self, id: &str, length: usize, limit: usize) -> Result<bool> {
        // The length table has one row per indexed chunk, even for chunks without
        // terms. A second occurrence is corrupt input and must remain
        // distinguishable from a valid entry that cannot fit in cache.
        if self.chunk_lengths.contains_key(id) {
            return Err(Error::new(
                ErrorCode::IndexCorruption,
                "duplicate text chunk",
            ));
        }
        if !self
            .cache
            .as_mut()
            .expect("resident cache owner")
            .admit_node::<String, usize>(LENGTHS, self.chunk_lengths.len() + 1, id.len(), limit)
        {
            return Ok(false);
        }
        let key = match copy_key(id) {
            Ok(key) => key,
            Err(error) => {
                // Node admission may already have raised its high-water estimate.
                // If the key copy fails, release only its reserved payload bytes
                // and leave that conservative node estimate for future insertions.
                self.cache.as_mut().unwrap().release_payload(id.len());
                return Err(error);
            }
        };
        self.chunk_lengths.insert(key, length);
        Ok(true)
    }

    pub(super) fn cache_term(
        &mut self,
        term: &str,
        additional: usize,
        limit: usize,
    ) -> Result<bool> {
        let owner = self.cache.as_mut().expect("resident cache owner");
        // A term shared by multiple collections has one posting list. Admit the
        // map entry only on first sight, then grow that same list by the declared
        // number of additional postings.
        if !self.postings.contains_key(term) {
            if !owner.admit_node::<String, Vec<Posting>>(
                TERMS,
                self.postings.len() + 1,
                term.len(),
                limit,
            ) {
                return Ok(false);
            }
            // The node admission includes the prospective term string. If
            // allocating that string fails, return its share while leaving the
            // tree's conservative node headroom under the cache owner.
            let key = match copy_key(term) {
                Ok(key) => key,
                Err(error) => {
                    owner.release_payload(term.len());
                    return Err(error);
                }
            };
            self.postings.insert(key, Vec::new());
        }
        let postings = self.postings.get_mut(term).expect("text term installed");
        // Use the existing posting count when appending another collection. Adding
        // only the new declaration to a fresh capacity estimate would undercount
        // terms that already have entries.
        let count = postings
            .len()
            .checked_add(additional)
            .ok_or_else(|| Error::new(ErrorCode::IndexCorruption, "text posting count overflow"))?;
        // Existing spare slots are already charged. A zero-frequency term can
        // enter the term table and require no posting allocation of its own.
        if count <= postings.capacity() {
            return Ok(true);
        }
        // The existing array remains live until the fully admitted new array is installed.
        // Overflow while doubling is harmless here. The checked required count
        // above is still a valid allocation target when a speculative doubling
        // would exceed usize.
        let preferred = postings
            .capacity()
            .checked_mul(2)
            .unwrap_or(count)
            .max(count);
        // Charge the full new backing array while the old allocation still holds
        // its postings. During replacement those two arrays coexist even though
        // the existing chunk ID strings move without being cloned.
        let width = std::mem::size_of::<Posting>() as u128;
        let capacity = if owner.admit_payload(preferred as u128 * width, limit) {
            preferred
        // Geometric growth is optional. If the shared cache limit permits only the
        // number of entries actually promised for this term, use exactly that size
        // and still admit the collection.
        } else if preferred != count && owner.admit_payload(count as u128 * width, limit) {
            count
        } else {
            return Ok(false);
        };
        // The multiplication was done in u128 for admission. A successful request
        // fits the usize byte limit, so the error paths can return that exact
        // charge in the owner's native units.
        let bytes = capacity as u128 * width;
        let mut grown = Vec::new();
        // A failed allocator request differs from a deliberate cache refusal. Undo
        // only the new buffer's reservation and leave existing postings intact as
        // the allocation error propagates.
        if let Err(error) = grown.try_reserve_exact(capacity) {
            owner.release_payload(bytes as usize);
            return Err(std::io::Error::other(error).into());
        }
        // try_reserve_exact promises at least the requested count. Reject an
        // allocator's larger capacity before installing the array because the
        // excess slots have no credit on this ledger.
        if grown.capacity() != capacity {
            drop(grown);
            owner.release_payload(bytes as usize);
            return Err(std::io::Error::other("text cache array capacity differs").into());
        }
        // Move the posting values into the replacement and drop the drained old
        // array before releasing its slot credit. The ID strings and shared
        // collection owners stay live throughout.
        let old = postings.capacity() * std::mem::size_of::<Posting>();
        grown.append(postings);
        drop(std::mem::replace(postings, grown));
        owner.release_payload(old);
        Ok(true)
    }

    pub(super) fn cache_collection_owner(&mut self, id: &str, limit: usize) -> Option<Arc<str>> {
        self.cache
            .as_mut()
            .expect("resident cache owner")
            // One collection string allocation is shared by all its postings.
            // Charge its Arc payload and reference counts once before cloning that
            // same owner into each individual posting.
            .admit_payload(arc_bytes(id), limit)
            .then(|| Arc::from(id))
    }

    pub(super) fn cache_posting(
        &mut self,
        term: &str,
        id: &str,
        frequency: usize,
        collection: &Arc<str>,
        limit: usize,
    ) -> Result<bool> {
        // The term declaration must precede its posting events. An unknown term is
        // a segment-format error, while exceeding the optional cache allowance
        // below can be handled as a cache miss.
        let postings = self
            .postings
            .get_mut(term)
            .ok_or_else(|| Error::new(ErrorCode::IndexCorruption, "text posting has no term"))?;
        // The declared term allocated slots for its posting group. A zero
        // occurrence count or a posting that would force an unplanned growth
        // invalidates the event stream at this point.
        if postings.len() == postings.capacity() || frequency == 0 {
            return Err(Error::new(
                ErrorCode::IndexCorruption,
                "invalid text posting",
            ));
        }
        let owner = self.cache.as_mut().expect("resident cache owner");
        // Only the incoming chunk ID needs a new string allocation here. The array
        // slot is already admitted and the collection identifier is shared by the
        // caller's Arc.
        if !owner.admit_payload(id.len() as u128, limit) {
            return Ok(false);
        }
        // Copying the chunk ID can fail after its bytes are admitted. The posting
        // list has not been extended yet, so the failure releases just this
        // prospective ID.
        let key = match copy_key(id) {
            Ok(key) => key,
            Err(error) => {
                owner.release_payload(id.len());
                return Err(error);
            }
        };
        // Store the internal chunk ID and occurrence count together. Cloning the
        // collection Arc preserves this posting's scope without another allocation
        // for its text.
        postings.push(Posting {
            collection_id: Arc::clone(collection),
            chunk_id: key,
            term_frequency: frequency,
        });
        Ok(true)
    }

    pub(super) fn cache_collection(
        &mut self,
        id: &str,
        chunks: usize,
        total: usize,
        collection: Option<Arc<str>>,
        limit: usize,
    ) -> Result<bool> {
        // A complete collection supplies its corpus size and average once.
        // Accepting a second entry under the same internal ID would mix ownership
        // and BM25 statistics from different visits.
        if self.total_chunks_by_collection.contains_key(id) {
            return Err(Error::new(
                ErrorCode::IndexCorruption,
                "duplicate text collection",
            ));
        }
        let owner = self.cache.as_mut().expect("resident cache owner");
        if !owner.admit_node::<String, CollectionCounts>(
            COUNTS,
            self.total_chunks_by_collection.len() + 1,
            id.len(),
            limit,
        ) {
            return Ok(false);
        }
        // The counts and average maps retain distinct owned keys. If admission of
        // the second fails, return the first prospective key's credit before
        // reporting an optional miss.
        if !owner.admit_node::<String, f64>(
            AVERAGES,
            self.avg_len_by_collection.len() + 1,
            id.len(),
            limit,
        ) {
            owner.release_payload(id.len());
            return Ok(false);
        }
        // Make both key copies before inserting either statistic. A failure
        // copying the second then leaves neither map with half of this
        // collection's statistics.
        let keys = (|| -> Result<_> { Ok((copy_key(id)?, copy_key(id)?)) })();
        let (count_key, average_key) = match keys {
            Ok(keys) => keys,
            Err(error) => {
                owner.release_payload(2 * id.len());
                return Err(error);
            }
        };
        // The optional owner keeps a posting collection's shared identifier alive
        // with its counts. A nonempty collection with no postings can still
        // contribute corpus statistics without that allocation.
        self.total_chunks_by_collection.insert(
            count_key,
            CollectionCounts {
                chunks,
                owner: collection,
            },
        );
        self.avg_len_by_collection
            // Callers finish only nonempty collections through this path. Use the
            // same sum of admitted chunk lengths and count as the resident index
            // will use for BM25 normalization.
            .insert(average_key, total as f64 / chunks as f64);
        Ok(true)
    }

    pub(crate) fn replace_cached_collections<'a>(
        &mut self,
        collections: &BTreeSet<String>,
        old_chunk_ids: impl IntoIterator<Item = &'a str>,
        mut replacement: Self,
        limit: usize,
    ) -> Result<bool> {
        // A partial replacement requires cache owners on both sides. If either is
        // absent, discard the old resident state so it cannot continue answering
        // with outdated collection data.
        if self.cache.is_none() || replacement.cache.is_none() {
            self.clear_cache();
            return Ok(false);
        }
        self.cache
            .as_ref()
            .unwrap()
            .reservation
            // Payload credit for unchanged or moved allocations can be handed
            // between cache owners only when both belong to the same ledger. A
            // mismatched ledger would double charge one operation or release
            // another's memory.
            .assert_same_ledger(&replacement.cache.as_ref().unwrap().reservation);
        // A partial merge may have moved postings whose shared collection
        // allocation still belongs to the incoming cache. Keep the source alive
        // until a failed destination has been cleared.
        let result = self.merge_cached(collections, old_chunk_ids, &mut replacement, limit);
        // A cache-limit refusal and a hard merge error both invalidate the
        // destination. Some posting groups may already have moved, making the
        // previous resident view unusable even though primary data remains
        // available.
        if !matches!(result, Ok(true)) {
            // Moved postings may still share allocations credited by replacement.
            self.clear_cache();
        }
        drop(replacement);
        result
    }

    fn merge_cached<'a>(
        &mut self,
        collections: &BTreeSet<String>,
        old_chunk_ids: impl IntoIterator<Item = &'a str>,
        replacement: &mut Self,
        limit: usize,
    ) -> Result<bool> {
        // Evict only postings for the collections being rebuilt. A term shared with
        // an untouched collection keeps its posting array, including spare capacity.
        // Refund the removed ID strings now and the array only when the term itself
        // disappears.
        // The merge mutates the destination as it goes. A budget refusal may
        // arrive after the previous collection's rows have been removed, so the
        // caller owns eviction of the whole failed result.
        let mut removed = 0usize;
        self.postings.retain(|key, postings| {
            postings.retain(|posting| {
                if collections.contains(posting.collection_id.as_ref()) {
                    removed += posting.chunk_id.capacity();
                    false
                } else {
                    true
                }
            });
            // Keep the existing term key and posting vector when any untouched row
            // remains. Their spare capacity is still allocated after retain and
            // must stay in the destination account.
            if postings.is_empty() {
                removed += key.capacity() + postings.capacity() * std::mem::size_of::<Posting>();
                false
            } else {
                true
            }
        });
        // Use the caller's old chunk IDs to remove lengths, including rows absent
        // from the replacement. Looking only at the incoming IDs would leave old
        // lengths for deleted chunks.
        for id in old_chunk_ids {
            if let Some((key, _)) = self.chunk_lengths.remove_entry(id) {
                removed += key.capacity();
                drop(key);
            }
        }
        // Collection anchors own the shared ID allocation's credit. Remove their
        // postings first so returning that credit cannot precede destruction of a
        // still-referenced string.
        self.total_chunks_by_collection.retain(|key, counts| {
            if collections.contains(key) {
                removed +=
                    key.capacity() + counts.owner.as_ref().map_or(0, |id| arc_bytes(id) as usize);
                false
            } else {
                true
            }
        });
        self.avg_len_by_collection.retain(|key, _| {
            if collections.contains(key) {
                removed += key.capacity();
                false
            } else {
                true
            }
        });
        // Return only the removed payload capacity once those entries have been
        // destroyed. Map node reservations remain because clearing selected keys
        // does not tell us how much backing storage the trees still hold.
        self.cache.as_mut().unwrap().release_payload(removed);

        while let Some((term, mut postings)) = replacement.postings.pop_first() {
            let array = postings.capacity() * std::mem::size_of::<Posting>();
            // Separate the size of the incoming posting array from the sum of the
            // child ID allocations. The shared-term branch frees the old array but
            // transfers ownership of those IDs.
            let ids = postings
                .iter()
                .map(|posting| posting.chunk_id.capacity())
                .sum::<usize>();
            // When a term is shared, grow the destination array before moving its
            // postings. The chunk ID allocations move with them; the incoming term
            // string and emptied backing array can be freed. For a new term the key
            // and the whole array change owners together.
            if self.postings.contains_key(&term) {
                // Ask the same ledger for destination array room while the
                // incoming array is still live. If this fails, the source and
                // destination can be partly consumed, and the outer merge wrapper
                // discards the optional cache.
                if !self.cache_term(&term, postings.len(), limit)? {
                    return Ok(false);
                }
                let target = self.postings.get_mut(&term).unwrap();
                target.append(&mut postings);
                // Sorting at this boundary preserves the ordering that
                // collection_postings uses to find the range of one collection,
                // followed by chunk IDs searched within that range.
                target.sort_unstable_by(super::posting_order);
                let released = term.capacity() + array;
                drop((term, postings));
                // Refund the source term and empty array after dropping them.
                // Transfer only the chunk strings' existing credit to the
                // destination.
                let source = replacement.cache.as_mut().unwrap();
                source.release_payload(released);
                source.transfer_payload(self.cache.as_mut().unwrap(), ids);
            } else {
                // A new term needs a destination tree entry. Its key string and
                // posting array already have owners in the source, so move their
                // credit only after the node is admitted.
                if !self
                    .cache
                    .as_mut()
                    .unwrap()
                    .admit_node::<String, Vec<Posting>>(TERMS, self.postings.len() + 1, 0, limit)
                {
                    return Ok(false);
                }
                replacement
                    .cache
                    .as_mut()
                    .unwrap()
                    .transfer_payload(self.cache.as_mut().unwrap(), term.capacity() + array + ids);
                self.postings.insert(term, postings);
            }
        }
        // Move the length table as well as the postings. Scoring a moved chunk with
        // its old length would change BM25 even when the new postings and collection
        // average are correct.
        // A length is keyed independently of the term dictionary, even for a
        // zero-token chunk. Retain the admitted key with its length so score
        // completion can look it up later.
        while let Some((key, length)) = replacement.chunk_lengths.pop_first() {
            if !self.cache.as_mut().unwrap().admit_node::<String, usize>(
                LENGTHS,
                self.chunk_lengths.len() + 1,
                0,
                limit,
            ) {
                return Ok(false);
            }
            // The length table takes ownership of the original key, rather than
            // cloning it for a different map. Transfer its existing payload grant
            // after reserving the destination map node.
            replacement
                .cache
                .as_mut()
                .unwrap()
                .transfer_payload(self.cache.as_mut().unwrap(), key.capacity());
            self.chunk_lengths.insert(key, length);
        }
        // Collection anchors move only after all their postings and ID storage.
        // The common collection ID is shared by multiple postings. Transfer its
        // credit with the collection anchor once its postings are installed,
        // leaving recovery of a partial merge to the outer caller.
        while let Some((key, counts)) = replacement.total_chunks_by_collection.pop_first() {
            if !self
                .cache
                .as_mut()
                .unwrap()
                .admit_node::<String, CollectionCounts>(
                    COUNTS,
                    self.total_chunks_by_collection.len() + 1,
                    0,
                    limit,
                )
            {
                return Ok(false);
            }
            // The anchor can carry an optional Arc owner in addition to the map
            // key. The transfer covers both that allocation and the key even if
            // only a subset of this collection's postings was nonempty.
            let bytes =
                key.capacity() + counts.owner.as_ref().map_or(0, |id| arc_bytes(id) as usize);
            replacement
                .cache
                .as_mut()
                .unwrap()
                .transfer_payload(self.cache.as_mut().unwrap(), bytes);
            self.total_chunks_by_collection.insert(key, counts);
        }
        // Publish averages from the replacement collection together with its
        // postings and length counts. Scoring must observe this set as a unit
        // because the same raw counts can produce a different BM25 score under a
        // different average.
        while let Some((key, average)) = replacement.avg_len_by_collection.pop_first() {
            if !self.cache.as_mut().unwrap().admit_node::<String, f64>(
                AVERAGES,
                self.avg_len_by_collection.len() + 1,
                0,
                limit,
            ) {
                // A refusal here can occur after every posting has been moved. The
                // caller still clears the incomplete cache, since it cannot safely
                // rank with new postings and an incomplete average-length table.
                return Ok(false);
            }
            replacement
                .cache
                .as_mut()
                .unwrap()
                .transfer_payload(self.cache.as_mut().unwrap(), key.capacity());
            self.avg_len_by_collection.insert(key, average);
        }
        // The successful path has moved every required table to the destination.
        // The remaining reservation in the incoming cache covers only storage
        // still owned there and will be released with that cache.
        Ok(true)
    }
}
