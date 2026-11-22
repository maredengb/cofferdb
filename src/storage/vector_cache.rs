use std::collections::{btree_map, BTreeMap};
use std::ops::Deref;

use super::{CollectionInfo, RetainedEmbedding, VectorIndexEntry, VectorIndexSegmentPayload};
use crate::json::{btree_workspace, CacheReservation, JsonOperationContext};

// Payload must be dropped before its credit. This type is deliberately not
// Clone: reusing a segment moves its existing allocation and reservation.
pub(crate) struct CachedVectorSegment {
    // Decoded segment for one collection. Its covered commit and vector
    // configuration travel with the entries when the segment moves between caches.
    payload: VectorIndexSegmentPayload,
    // Cache credit for the segment's owned identifiers, entry slots and coordinate
    // buffers. The map's copy of the collection ID has separate credit.
    reservation: CacheReservation,
}

impl CachedVectorSegment {
    pub(super) fn with_capacity(
        collection: &CollectionInfo,
        commit: u64,
        count: usize,
        context: &JsonOperationContext,
        limit: usize,
    ) -> crate::Result<Option<Self>> {
        // Admit the segment header's variable-length fields and the entire array
        // of row slots up front. Chunk IDs, embedding IDs and coordinate buffers
        // arrive with each decoded row and are transferred separately.
        let bytes = collection.collection_id.len() as u128
            + collection.embedding_model.as_ref().map_or(0, String::len) as u128
            + count as u128 * std::mem::size_of::<VectorIndexEntry>() as u128;
        // A cache limit refusal means this segment can still be written or
        // streamed without a resident copy. An allocator failure after the
        // reservation is granted remains a real error to the caller.
        let Some(reservation) = context.cache_reservation(bytes, limit) else {
            return Ok(None);
        };
        let mut vectors = Vec::new();
        vectors
            .try_reserve_exact(count)
            .map_err(std::io::Error::other)?;
        // Copy the collection's scoring settings with the directory identity.
        // Reusing these vectors later requires comparing them against the
        // collection and checkpoint that a new request expects.
        let payload = VectorIndexSegmentPayload {
            collection_id: collection.collection_id.clone(),
            covered_commit_sequence_no: commit,
            dim: collection.embedding_dim,
            distance_metric: collection.distance_metric,
            embedding_model: collection.embedding_model.clone(),
            vectors_normalized: collection.vectors_normalized,
            vectors,
        };
        // The reservation was made for exact capacities. If the allocator supplies
        // extra slots or string space, drop this provisional payload instead of
        // exposing a cache that understates its memory.
        if payload.vectors.capacity() != count
            || payload.collection_id.capacity() != collection.collection_id.len()
            || payload.embedding_model.as_ref().map_or(0, String::capacity)
                != collection.embedding_model.as_ref().map_or(0, String::len)
        {
            return Err(
                std::io::Error::other("cache allocation exceeds requested capacity").into(),
            );
        }
        Ok(Some(Self::new(payload, reservation)))
    }

    pub(super) fn push_retained(&mut self, embedding: RetainedEmbedding, limit: usize) -> bool {
        // The producer counted the candidate rows before creating this array. A
        // full array indicates that its count and traversal disagree, not that the
        // cache can grow by silently allocating another row slot.
        assert!(self.payload.vectors.len() < self.payload.vectors.capacity());
        // Moving a retained row may still be refused under the active limit. Do
        // not add a partial entry when its owned IDs and coordinate buffer could
        // not be transferred together.
        let Some(entry) = embedding.try_into_cache_entry(&mut self.reservation, limit) else {
            return false;
        };
        self.payload.vectors.push(entry);
        true
    }

    pub(crate) fn reserve(
        context: &JsonOperationContext,
        cache_limit: usize,
        encoded_len: u64,
    ) -> Option<CacheReservation> {
        // Keep the existing 8x decoder headroom, including owned IDs, vector
        // slots and strings. Directories hold their own allocation credit.
        // The encoded byte count includes framing, while the retained
        // representation owns native strings and a row array. Use a wide product
        // for the admission estimate before comparing with the effective cache
        // limit.
        context.cache_reservation(encoded_len as u128 * 8, cache_limit)
    }

    pub(super) fn new(
        payload: VectorIndexSegmentPayload,
        mut reservation: CacheReservation,
    ) -> Self {
        // The serialized extent length was only an admission estimate. Measure the
        // decoded payload by actual capacities, including unused vector slots,
        // before releasing excess cache credit.
        let bytes = payload.collection_id.capacity()
            + payload.embedding_model.as_ref().map_or(0, String::capacity)
            + payload.vectors.capacity() * std::mem::size_of::<VectorIndexEntry>()
            + payload
                .vectors
                .iter()
                // The vector-entry array holds String and Vec headers inline. Each
                // initialized entry also owns both ID buffers and f32 storage,
                // whose allocations must be counted separately from that array.
                .map(|entry| {
                    entry.chunk_id.capacity()
                        + entry.embedding_id.capacity()
                        + entry.vector.capacity() * std::mem::size_of::<f32>()
                })
                .sum::<usize>();
        // Trim the reservation only after the payload's final representation is
        // known. This owner subsequently moves with the segment and also covers
        // any spare component capacity that will remain resident.
        reservation.retain(bytes);
        Self {
            payload,
            reservation,
        }
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reservation.reserved_bytes()
    }

    pub(crate) fn set_covered_commit(&mut self, commit: u64) {
        // Reusing an unchanged collection can carry its same entries into a newer
        // checkpoint. Only the sequence stamp changes here; the configuration and
        // coordinate data must already be valid for the new view.
        self.payload.covered_commit_sequence_no = commit;
    }
}

pub(crate) struct VectorCache {
    // Maps collection ID to its current in-memory segment. A second segment for the
    // same ID replaces the first while keeping the existing map key.
    entries: BTreeMap<String, CachedVectorSegment>,
    // Shared cache credit assigned to this B-tree's nodes and heap-owned keys. Each
    // segment carries its own payload credit.
    directory: CacheReservation,
    // High-water byte allowance for B-tree nodes. Removing keys can release their
    // strings without establishing that any particular node was freed.
    node_bytes: usize,
    // Bytes held in heap allocations for the current map keys. A key moved between
    // directories transfers its credit alongside the String.
    key_bytes: usize,
}

impl VectorCache {
    pub(crate) fn new(context: &JsonOperationContext) -> Self {
        Self {
            // Create the directory reservation on the operation's ledger even
            // while the map is empty. Later cache merges can transfer
            // already-owned key storage only between compatible ledgers.
            entries: BTreeMap::new(),
            directory: context
                .cache_reservation(0, usize::MAX)
                .expect("empty cache admission"),
            node_bytes: 0,
            key_bytes: 0,
        }
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        // Combine the directory's actual charge with all the independently owned
        // segment reservations. Summing vector lengths alone would omit spare
        // capacity, identifiers and the B-tree's admitted nodes.
        self.directory.reserved_bytes()
            + self
                .entries
                .values()
                .map(CachedVectorSegment::reserved_bytes)
                .sum::<usize>()
    }

    pub(crate) fn values_mut(&mut self) -> btree_map::ValuesMut<'_, String, CachedVectorSegment> {
        self.entries.values_mut()
    }

    pub(crate) fn retain(
        &mut self,
        mut keep: impl FnMut(&String, &mut CachedVectorSegment) -> bool,
    ) {
        let mut removed = 0;
        // The predicate can adjust a retained segment in place, for example when
        // its covered commit is updated. Evicting an entry drops that segment's
        // payload reservation independently of the directory.
        self.entries.retain(|id, segment| {
            if keep(id, segment) {
                true
            } else {
                // Refund the discarded key's owned capacity after the map releases
                // it. Only the directory's key bytes are accumulated here;
                // charging the segment payload again would subtract it twice.
                removed += id.capacity();
                false
            }
        });
        self.key_bytes -= removed;
        // Retain node headroom: removal does not promise that all empty nodes
        // have been deallocated. The entire tree drops before this reservation.
        // Even an empty map reached by retaining no entries can still carry node
        // headroom. Clearing the entire cache takes a different path that drops
        // the tree before returning all directory credit.
        self.directory.retain(self.node_bytes + self.key_bytes);
    }

    // Admit the directory growth before allocating a new B-tree entry. Existing
    // node credit is a lower bound even after removals, because the allocator need
    // not immediately return empty nodes.
    fn admit_node(&mut self, key_bytes: usize, limit: usize) -> bool {
        let nodes = btree_workspace::<String, CachedVectorSegment>(self.entries.len() + 1)
            .max(self.node_bytes);
        // Only the increase in tree capacity needs a new grant. A brand-new
        // collection key also needs its own allocation, distinct from the
        // collection ID already stored inside the segment.
        if !self
            .directory
            .try_grow((nodes - self.node_bytes) as u128 + key_bytes as u128, limit)
        {
            return false;
        }
        self.node_bytes = nodes;
        true
    }

    pub(crate) fn insert(&mut self, segment: CachedVectorSegment, limit: usize) -> bool {
        // The directory and the retained segment must participate in the same
        // budget. A separate ledger could make their combined live memory
        // invisible to the connection's cache accounting.
        self.directory.assert_same_ledger(&segment.reservation);
        if let Some(existing) = self.entries.get_mut(&segment.collection_id) {
            // Replacement can reuse the existing directory slot. A zero-byte
            // growth check still tests the current shared cache limit before
            // dropping the old segment and accepting the new one.
            if !self.directory.try_grow(0, limit) {
                return false;
            }
            *existing = segment;
            return true;
        }
        // Reserve space for a new directory key before cloning it. If admission
        // fails, returning drops the incoming segment and leaves the destination
        // with its existing entries.
        let bytes = segment.collection_id.len();
        if !self.admit_node(bytes, limit) {
            return false;
        }
        let id = segment.collection_id.clone();
        self.key_bytes += bytes;
        self.entries.insert(id, segment);
        true
    }

    pub(crate) fn append(&mut self, mut other: Self, limit: usize) {
        self.directory.assert_same_ledger(&other.directory);
        // An empty destination can take ownership of the whole tree. Otherwise move
        // entries individually so duplicate keys can keep the existing node and new
        // keys can claim destination node space.
        // Swapping an empty destination makes the other tree and all of its key
        // and node reservations move as one unit. The limit check after both paths
        // is still required.
        if self.entries.is_empty() {
            std::mem::swap(self, &mut other);
        } else {
            while let Some((id, segment)) = other.entries.pop_first() {
                let bytes = id.capacity();
                other.key_bytes -= bytes;
                // Duplicate keys reuse the destination's key allocation. Drop the
                // transfer key and the displaced segment through their original
                // owners so that only the surviving identity and new vector remain
                // charged.
                if let Some(existing) = self.entries.get_mut(&id) {
                    *existing = segment;
                    drop(id);
                    other.directory.retain(other.node_bytes + other.key_bytes);
                } else if self.admit_node(0, limit) {
                    // The existing key allocation moves without release or a second
                    // charge; both tree allocations remain credited until freed.
                    other.directory.transfer_to(&mut self.directory, bytes);
                    self.key_bytes += bytes;
                    self.entries.insert(id, segment);
                // A refused directory node does not prevent movement of later
                // entries. Release this candidate and its key space while
                // retaining the source tree's node headroom until the entire
                // source tree drops.
                } else {
                    drop((id, segment));
                    other.directory.retain(other.node_bytes + other.key_bytes);
                }
            }
        }
        // Drop the remaining source directory before trying the final limit check.
        // A rejected cache is optional and can be cleared as a unit, so it never
        // prevents access through the underlying index.
        drop(other);
        if !self.directory.try_grow(0, limit) {
            self.clear();
        }
    }

    pub(crate) fn clear(&mut self) {
        // Drop the tree and all retained segments before releasing their remaining
        // directory credit. Reset both capacities so a later insertion budgets its
        // cold-start allocation.
        self.entries = BTreeMap::new();
        self.directory.retain(0);
        self.node_bytes = 0;
        self.key_bytes = 0;
    }
}

impl Deref for VectorCache {
    type Target = BTreeMap<String, CachedVectorSegment>;

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl Deref for CachedVectorSegment {
    type Target = VectorIndexSegmentPayload;

    fn deref(&self) -> &Self::Target {
        &self.payload
    }
}

#[cfg(test)]
impl std::fmt::Debug for CachedVectorSegment {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.payload.fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use std::sync::Arc;

    const LIMIT: usize = 1024 * 1024;

    fn context() -> JsonOperationContext {
        JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, LIMIT)))
    }

    fn segment(context: &JsonOperationContext, id: usize, seed: usize) -> CachedVectorSegment {
        let payload = VectorIndexSegmentPayload {
            collection_id: format!("col_{id:016}"),
            covered_commit_sequence_no: seed as u64,
            dim: seed % 17 + 1,
            distance_metric: crate::model::DistanceMetric::DotProduct,
            embedding_model: None,
            vectors_normalized: false,
            vectors: vec![VectorIndexEntry {
                chunk_id: format!("chk_{id:016}"),
                embedding_id: format!("emb_{id:016}"),
                vector: vec![seed as f32; seed % 17 + 1],
            }],
        };
        let held =
            CachedVectorSegment::reserve(context, LIMIT, payload.encoded_len().unwrap()).unwrap();
        CachedVectorSegment::new(payload, held)
    }

    fn assert_used(context: &JsonOperationContext, bytes: usize) {
        let rest = context
            .cache_reservation((LIMIT - bytes) as u128, LIMIT)
            .unwrap();
        assert!(context.cache_reservation(1, LIMIT).is_none());
        drop(rest);
    }

    #[test]
    fn directory_admits_before_insert_and_holds_nodes_after_payload_removal() {
        for seed in 0..100 {
            let context = context();
            let mut cache = VectorCache::new(&context);
            let count = seed % 12;
            for id in 0..count {
                assert!(cache.insert(segment(&context, id, seed), LIMIT));
            }
            let original = cache.reserved_bytes();
            let nodes =
                btree_workspace::<String, CachedVectorSegment>(count + 1).max(cache.node_bytes);
            let additional = nodes - cache.node_bytes + 20;
            for shortage in [true, false] {
                let incoming = segment(&context, 1000, seed);
                let payload_bytes = incoming.reserved_bytes();
                let pointer = incoming.vectors[0].vector.as_ptr();
                let blocker = context
                    .cache_reservation(
                        (LIMIT - original - payload_bytes - additional + usize::from(shortage))
                            as u128,
                        LIMIT,
                    )
                    .unwrap();
                assert_eq!(cache.insert(incoming, LIMIT), !shortage);
                if shortage {
                    assert_eq!(cache.len(), count);
                    assert_eq!(cache.reserved_bytes(), original);
                } else {
                    assert_eq!(cache.len(), count + 1);
                    assert_eq!(
                        cache.reserved_bytes(),
                        original + payload_bytes + additional
                    );
                    assert_eq!(
                        cache["col_0000000000001000"].vectors[0].vector.as_ptr(),
                        pointer
                    );
                }
                drop(blocker);
                assert_eq!(
                    cache.key_bytes,
                    cache.keys().map(String::capacity).sum::<usize>()
                );
                assert_used(&context, cache.reserved_bytes());
            }
            cache.retain(|_, _| false);
            assert!(cache.is_empty());
            assert_eq!(cache.directory.reserved_bytes(), nodes);
            assert_used(&context, nodes);
            cache.clear();
            assert_used(&context, 0);
        }
    }

    #[test]
    fn cache_append_moves_existing_keys_and_payloads_without_double_charging() {
        for seed in 0..100 {
            let context = context();
            let mut left = VectorCache::new(&context);
            let mut right = VectorCache::new(&context);
            for id in 0..seed % 12 {
                assert!(left.insert(segment(&context, id, seed), LIMIT));
            }
            for id in seed % 7..seed % 7 + seed % 9 + 1 {
                assert!(right.insert(segment(&context, id, seed + 1), LIMIT));
            }
            let mut expected = left
                .iter()
                .map(|(id, value)| (id.clone(), (id.as_ptr(), value.vectors[0].vector.as_ptr())))
                .collect::<BTreeMap<_, _>>();
            for (id, value) in right.iter() {
                let key = expected.get(id).map_or(id.as_ptr(), |entry| entry.0);
                expected.insert(id.clone(), (key, value.vectors[0].vector.as_ptr()));
            }
            let extra = btree_workspace::<String, CachedVectorSegment>(expected.len())
                .saturating_sub(left.node_bytes);
            let blocker = context
                .cache_reservation(
                    (LIMIT - left.reserved_bytes() - right.reserved_bytes() - extra) as u128,
                    LIMIT,
                )
                .unwrap();
            left.append(right, LIMIT);
            assert_eq!(left.len(), expected.len());
            for (id, value) in left.iter() {
                assert_eq!(
                    (id.as_ptr(), value.vectors[0].vector.as_ptr()),
                    expected[id]
                );
            }
            drop(blocker);
            assert_eq!(
                left.key_bytes,
                left.keys().map(String::capacity).sum::<usize>()
            );
            assert_used(&context, left.reserved_bytes());
            drop(left);
            assert_used(&context, 0);
        }
    }

    #[test]
    fn cache_append_releases_misses_and_reuses_an_empty_directory_at_a_lower_limit() {
        for seed in 0..100 {
            let context = context();
            let mut left = VectorCache::new(&context);
            for id in 0..5 {
                assert!(left.insert(segment(&context, id, seed), LIMIT));
            }
            let original = left.reserved_bytes();
            let mut right = VectorCache::new(&context);
            assert!(right.insert(segment(&context, 1000, seed), LIMIT));
            let blocker = context
                .cache_reservation(
                    (LIMIT - left.reserved_bytes() - right.reserved_bytes()) as u128,
                    LIMIT,
                )
                .unwrap();
            left.append(right, LIMIT);
            assert_eq!(left.len(), 5);
            assert_eq!(left.reserved_bytes(), original);
            drop(blocker);
            assert_used(&context, original);
            left.retain(|_, _| false);
            let mut right = VectorCache::new(&context);
            assert!(right.insert(segment(&context, 1000, seed), LIMIT));
            let right_bytes = right.reserved_bytes();
            let pointer = right["col_0000000000001000"].vectors.as_ptr();
            left.append(right, right_bytes);
            assert_eq!(left.len(), 1);
            assert_eq!(left["col_0000000000001000"].vectors.as_ptr(), pointer);
            assert_used(&context, right_bytes);
            left.append(VectorCache::new(&context), 0);
            assert!(left.is_empty());
            assert_used(&context, 0);
        }
    }

    #[test]
    fn cache_cannot_adopt_another_runtimes_unaccounted_payload_or_directory() {
        let context = context();
        let foreign = JsonOperationContext::standalone();
        let mut cache = VectorCache::new(&context);
        for directory in [false, true] {
            let incoming = segment(&foreign, 1, 0);
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if directory {
                    let mut source = VectorCache::new(&foreign);
                    assert!(source.insert(incoming, LIMIT));
                    cache.append(source, LIMIT);
                } else {
                    cache.insert(incoming, LIMIT);
                }
            }));
            assert!(failed.is_err());
            assert!(cache.is_empty());
            assert_used(&context, 0);
            assert_used(&foreign, 0);
        }
    }
}
