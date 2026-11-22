use super::*;

// The optional snapshot cache owns no heap storage; long historical IDs use the live scan.
#[derive(Clone, Copy, Debug)]
struct InlineId {
    // Inline UTF-8 bytes of a complete collection or embedding ID. Returning no
    // InlineId for a longer name keeps the optional cache allocation-free.
    bytes: [u8; 32],
    // Used prefix in bytes, including all bytes of the final character. The
    // remaining array contents do not participate in comparisons.
    len: u8,
}

impl InlineId {
    fn new(value: &str) -> Option<Self> {
        // Long historical IDs remain valid catalog keys. Refuse a cache entry that
        // cannot hold the complete byte sequence rather than truncating distinct
        // IDs to the same key.
        if value.len() > 32 {
            return None;
        }
        let mut bytes = [0; 32];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        Some(Self {
            bytes,
            len: value.len() as u8,
        })
    }

    fn as_str(&self) -> &str {
        // Construction copied a complete &str and saved its byte length. The
        // zero-filled tail is padding and must be excluded when recovering the
        // original ID.
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("inline ID preserves complete UTF-8 bytes")
    }
}

#[derive(Clone, Copy, Debug)]
enum CachedModelState {
    // No visible embedding supplied a witness. A declaration on the collection may
    // still bind its model.
    Empty,
    // The witness identifies a visible embedding whose name can supply the
    // effective model when the collection has no explicit declaration.
    Unique(InlineId),
    // Keep the offending embedding ID for diagnostics. A generic conflict bit
    // would lose the row needed by a consistency check to report its location.
    Conflict(InlineId),
}

#[derive(Clone, Debug, Default)]
pub(super) struct ModelStates {
    // Small fixed ring of collection IDs and cached model conclusions. Entries hold
    // embedding IDs as witnesses so the model name comes from the current catalog.
    entries: [Option<(InlineId, CachedModelState)>; 8],
    // Slot to replace on the next insertion. Eviction changes the cost of the next
    // lookup but not the result of resolving the collection model.
    next: usize,
}

impl ModelStates {
    fn get(&self, collection_id: &str) -> Option<&CachedModelState> {
        self.entries
            .iter()
            .flatten()
            .find(|(id, _)| id.as_str() == collection_id)
            .map(|(_, state)| state)
    }

    fn insert(&mut self, collection_id: InlineId, state: CachedModelState) {
        // Callers avoid reinserting a collection already cached. Replacing the
        // next slot bounds memory independently of catalog size, and an evicted
        // collection can recover its state by scanning visible embeddings.
        self.entries[self.next] = Some((collection_id, state));
        self.next = (self.next + 1) % self.entries.len();
    }

    fn invalidate(&mut self, collection_id: &str) {
        // A write that changes visibility can change the effective model without
        // modifying the collection definition. Remove any witness for this owner
        // before carrying the cache into the new view.
        for entry in &mut self.entries {
            if entry
                .as_ref()
                .is_some_and(|(id, _)| id.as_str() == collection_id)
            {
                *entry = None;
            }
        }
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ModelState<'a> {
    // Neither the collection nor its visible embeddings impose a model. Checking a
    // caller-supplied name against this state leaves the catalog unmodified.
    Unbound,
    // The effective name borrows the collection's declaration or a visible
    // embedding. It remains meaningful only for this catalog view.
    Bound(&'a str),
    // Retain the actual conflicting row so queries, consistency checks and repair
    // reports can identify the embedding that disagreed.
    Conflict(&'a Embedding),
}

impl ModelState<'_> {
    pub(crate) fn validate(self, actual: Option<&str>) -> Result<()> {
        match self {
            // Historical conflicts fail even when the caller supplied no model.
            // Whole-catalog checks use this case to validate stored relationships
            // independently of a particular query.
            Self::Conflict(witness) => Err(Error::new(
                ErrorCode::EmbeddingModelMismatch,
                "collection has conflicting visible embedding models; export and explicitly clean up its data",
            )
            .with_detail("collection_id", &witness.collection_id)
            .with_detail("embedding_id", &witness.embedding_id)),
            // An absent caller declaration requests consistency only. For an
            // explicit mismatching name, report the effective collection model as
            // expected and the supplied name as actual.
            Self::Bound(expected) if actual.is_some_and(|actual| actual != expected) => {
                Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "input embedding model does not match the collection snapshot",
                )
                .with_detail("expected", expected)
                .with_detail("actual", actual.unwrap()))
            }
            _ => Ok(()),
        }
    }
}

impl Catalog {
    // A collection with no declared model can still be bound by its live embeddings.
    // Keep one matching embedding as evidence, or return the first conflicting one
    // so diagnostics can identify the row that prevented a common model.
    fn model_witness(&self, collection_id: &str) -> Result<(Option<&Embedding>, bool)> {
        let collection = self.collection_by_id(collection_id)?;
        // The collection declaration constrains even the first observed vector.
        // With no declaration, the first visible model supplies the comparison for
        // the rest of this scan.
        let mut expected = collection.embedding_model.as_deref();
        let mut witness = None;
        // Use the live embedding iterator so an old vector hidden by a chunk
        // replacement cannot make the collection appear to contain two models.
        // Other collections may use any model independently.
        for embedding in self
            .live_embeddings()
            .filter(|embedding| embedding.collection_id == collection_id)
        {
            // A mismatch against either the declaration or the first visible model
            // is sufficient. Preserve the offending embedding itself so validation
            // can identify its collection and stored ID.
            if expected.is_some_and(|model| model != embedding.model_name) {
                return Ok((Some(embedding), true));
            }
            if witness.is_none() {
                witness = Some(embedding);
                expected = Some(&embedding.model_name);
            }
        }
        // An empty iterator returns no witness and no conflict. The caller can
        // still bind that collection to an explicit declaration without any stored
        // vectors.
        Ok((witness, false))
    }

    pub(crate) fn visible_model_state(&self, collection_id: &str) -> Result<ModelState<'_>> {
        // Resolve the collection before consulting the optional ring. A missing or
        // dropped collection remains an error even if a cache slot happens to
        // contain its ID.
        let collection = self.collection_by_id(collection_id)?;
        if let Some(cached) = self.model_states.get(collection_id) {
            return Ok(match cached {
                // Empty describes the set of visible embeddings at the time of the
                // scan. The declaration can independently bind a model, even when
                // the cache has no witness ID to follow.
                CachedModelState::Empty => collection
                    .embedding_model
                    .as_deref()
                    .map_or(ModelState::Unbound, ModelState::Bound),
                CachedModelState::Unique(id) => {
                    // Prefer the declared model. Otherwise recover the name
                    // through the saved internal embedding ID, borrowing it from
                    // the catalog without retaining another model-name string in
                    // the ring.
                    ModelState::Bound(collection.embedding_model.as_deref().unwrap_or_else(|| {
                        &self
                            .embeddings
                            .get(id.as_str())
                            .expect("validated model witness")
                            .model_name
                    }))
                }
                // A conflict entry identifies the actual offending row. Mutations
                // that remove or supersede a witness must invalidate the cached
                // conclusion before another caller follows this ID.
                CachedModelState::Conflict(id) => ModelState::Conflict(
                    self.embeddings
                        .get(id.as_str())
                        .expect("validated conflict witness"),
                ),
            });
        }
        // On a miss, compute the same answer with a read-only scan. Resolving a
        // model does not require changing the catalog or successfully admitting a
        // cache entry.
        let (witness, conflict) = self.model_witness(collection_id)?;
        if conflict {
            return Ok(ModelState::Conflict(
                witness.expect("conflicting model witness"),
            ));
        }
        Ok(collection
            .embedding_model
            .as_deref()
            .or_else(|| witness.map(|embedding| embedding.model_name.as_str()))
            .map_or(ModelState::Unbound, ModelState::Bound))
    }

    pub(crate) fn cache_model_state(&mut self, collection_id: &str) -> Result<()> {
        // An existing slot already represents the current catalog until mutation
        // invalidates it. Do not advance the eviction pointer merely because
        // another query requests the same collection.
        if self.model_states.get(collection_id).is_some() {
            return Ok(());
        }
        let (witness, conflict) = self.model_witness(collection_id)?;
        // Historical collection and embedding IDs can exceed the inline slot. Skip
        // caching such a result so a later lookup can rescan it exactly, without
        // truncating or allocating an ID.
        let Some(id) = InlineId::new(collection_id) else {
            return Ok(());
        };
        let state = match witness {
            Some(embedding) => {
                let Some(witness_id) = InlineId::new(&embedding.embedding_id) else {
                    return Ok(());
                };
                // Retain the first disagreeing witness as a distinct cache state.
                // Treating it as a single-model witness would allow later reads to
                // accept mixed-model data.
                if conflict {
                    CachedModelState::Conflict(witness_id)
                } else {
                    CachedModelState::Unique(witness_id)
                }
            }
            // A cached empty scan avoids repeating work for a collection with no
            // current vector. The declaration is still read on every resolution
            // and can therefore supply the returned model.
            None => CachedModelState::Empty,
        };
        self.model_states.insert(id, state);
        Ok(())
    }

    pub(crate) fn invalidate_model_state(&mut self, collection_id: &str) {
        self.model_states.invalidate(collection_id);
    }

    pub(super) fn clear_model_states(&mut self) {
        self.model_states.clear();
    }

    pub(crate) fn require_consistent_models(&self) -> Result<()> {
        // Validate each live collection with no incoming model assertion. This
        // rejects conflicting visible data while accepting both an unbound
        // collection and one consistently bound to its own model.
        for collection in self.live_collections() {
            self.visible_model_state(&collection.collection_id)?
                .validate(None)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::{chunk, collection, document};

    fn fixture(explicit: Option<&str>, models: &[&str]) -> Catalog {
        fixture_with_ids(explicit, models, "col_1", "emb_")
    }

    fn fixture_with_ids(
        explicit: Option<&str>,
        models: &[&str],
        collection_id: &str,
        embedding_prefix: &str,
    ) -> Catalog {
        let mut catalog = Catalog::new_in(&crate::json::JsonOperationContext::standalone());
        let mut info = collection(1);
        info.collection_id = collection_id.to_owned();
        info.embedding_model = explicit.map(str::to_owned);
        catalog
            .apply_with_ref(RecordPayload::CollectionCreate(info), None)
            .unwrap();
        for (index, model) in models.iter().enumerate() {
            let number = index as u64 + 1;
            catalog
                .apply_with_ref(
                    RecordPayload::DocumentPut(document(number, collection_id)),
                    None,
                )
                .unwrap();
            catalog
                .apply_with_ref(RecordPayload::ChunkPut(chunk(number, collection_id)), None)
                .unwrap();
            catalog
                .apply_with_ref(
                    RecordPayload::EmbeddingPut(Embedding {
                        embedding_id: format!("{embedding_prefix}{number}"),
                        chunk_id: format!("chk_{number}"),
                        collection_id: collection_id.into(),
                        chunk_external_id: format!("chunk-{number}"),
                        model_name: (*model).into(),
                        dim: 2,
                        vector: Vec::new(),
                        created_at: "1970-01-01T00:00:00Z".into(),
                    }),
                    None,
                )
                .unwrap();
        }
        catalog
    }

    #[test]
    fn model_state_matrix_uses_exact_visible_identity() {
        for (explicit, models, bound, conflict) in [
            (None, vec![], None, false),
            (Some("M"), vec![], Some("M"), false),
            (None, vec!["M", "M"], Some("M"), false),
            (Some("M"), vec!["M"], Some("M"), false),
            (None, vec!["M", "m"], None, true),
            (Some("M"), vec!["m"], None, true),
        ] {
            let mut catalog = fixture(explicit, &models);
            let before = catalog.state_signature();
            catalog.cache_model_state("col_1").unwrap();
            match catalog.visible_model_state("col_1").unwrap() {
                ModelState::Unbound => assert!(bound.is_none() && !conflict),
                ModelState::Bound(actual) => assert_eq!(Some(actual), bound),
                ModelState::Conflict(witness) => {
                    assert!(conflict);
                    assert!(catalog.is_chunk_live(&witness.chunk_id));
                }
            }
            assert_eq!(catalog.state_signature(), before);
            let bytes = catalog.reserved_bytes();
            catalog.cache_model_state("col_1").unwrap();
            assert_eq!(catalog.reserved_bytes(), bytes);
        }
    }

    #[test]
    fn model_cache_has_no_heap_storage_or_catalog_budget_growth() {
        let mut catalog = fixture(None, &["M"]);
        let before = catalog.reserved_bytes();
        let (_, allocations) = crate::test_alloc::allocations(|| {
            catalog.cache_model_state("col_1").unwrap();
            assert!(matches!(
                catalog.visible_model_state("col_1").unwrap(),
                ModelState::Bound("M")
            ));
            catalog.cache_model_state("col_1").unwrap();
            catalog.invalidate_model_state("absent");
            catalog.invalidate_model_state("col_1");
            catalog.cache_model_state("col_1").unwrap();
            catalog.clear_model_states();
        });
        assert_eq!(allocations, 0);
        assert!(std::mem::size_of::<ModelStates>() <= 1024);
        assert_eq!(catalog.reserved_bytes(), before);
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Bound("M")
        ));
    }

    #[test]
    fn model_cache_eviction_and_invalidation_preserve_other_collections() {
        let mut catalog = Catalog::new_in(&crate::json::JsonOperationContext::standalone());
        for number in 1..=9 {
            let mut info = collection(number);
            info.embedding_model = Some(format!("M{number}"));
            catalog
                .apply_with_ref(RecordPayload::CollectionCreate(info), None)
                .unwrap();
        }
        let before = catalog.reserved_bytes();
        for number in 1..=9 {
            catalog.cache_model_state(&format!("col_{number}")).unwrap();
        }
        assert!(catalog.model_states.get("col_1").is_none());
        assert!(catalog.model_states.get("col_9").is_some());
        catalog.invalidate_model_state("col_9");
        assert!(catalog.model_states.get("col_9").is_none());
        assert!(catalog.model_states.get("col_8").is_some());
        for number in 1..=9 {
            let id = format!("col_{number}");
            let ModelState::Bound(actual) = catalog.visible_model_state(&id).unwrap() else {
                panic!("configured collection must remain bound");
            };
            assert_eq!(actual, format!("M{number}"));
            catalog.cache_model_state(&id).unwrap();
        }
        assert_eq!(catalog.reserved_bytes(), before);
    }

    #[test]
    fn long_utf8_ids_fall_back_without_restricting_model_identity() {
        let long = "x".repeat(64);
        for (collection_id, prefix) in [
            (long.as_str(), "emb_"),
            ("col_1", long.as_str()),
            (long.as_str(), long.as_str()),
        ] {
            for models in [vec!["Model"], vec!["Model", "model"]] {
                let mut catalog = fixture_with_ids(None, &models, collection_id, prefix);
                let (_, allocations) = crate::test_alloc::allocations(|| {
                    catalog.cache_model_state(collection_id).unwrap();
                });
                assert_eq!(allocations, 0);
                assert!(catalog.model_states.get(collection_id).is_none());
                match catalog.visible_model_state(collection_id).unwrap() {
                    ModelState::Bound(actual) => assert_eq!(actual, "Model"),
                    ModelState::Conflict(witness) => assert_eq!(witness.model_name, "model"),
                    ModelState::Unbound => panic!("live models cannot be unbound"),
                }
            }
        }
        let utf8 = "\u{e9}".repeat(16);
        assert_eq!(InlineId::new(&utf8).unwrap().as_str(), utf8);
        assert!(InlineId::new(&(utf8.clone() + "\u{e9}")).is_none());
    }

    #[test]
    fn deleted_and_replaced_embeddings_do_not_bind() {
        let mut catalog = fixture(None, &["M", "N"]);
        catalog.cache_model_state("col_1").unwrap();
        catalog
            .apply_with_ref(
                RecordPayload::Tombstone(Tombstone {
                    target_type: "document".into(),
                    target_id: "doc_2".into(),
                    deleted_at: "now".into(),
                    reason: None,
                }),
                None,
            )
            .unwrap();
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Bound("M")
        ));
        catalog
            .apply_with_ref(
                RecordPayload::EmbeddingPut(Embedding {
                    embedding_id: "emb_3".into(),
                    chunk_id: "chk_1".into(),
                    collection_id: "col_1".into(),
                    chunk_external_id: "chunk-1".into(),
                    model_name: "N".into(),
                    dim: 2,
                    vector: Vec::new(),
                    created_at: "now".into(),
                }),
                None,
            )
            .unwrap();
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Bound("N")
        ));
        catalog
            .apply_with_ref(
                RecordPayload::Tombstone(Tombstone {
                    target_type: "chunk".into(),
                    target_id: "chk_1".into(),
                    deleted_at: "now".into(),
                    reason: None,
                }),
                None,
            )
            .unwrap();
        assert!(matches!(
            catalog.visible_model_state("col_1").unwrap(),
            ModelState::Unbound
        ));
    }
}
