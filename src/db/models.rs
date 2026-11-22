use super::*;
use crate::catalog::ModelState;
use crate::json::ReservedString;

impl Inner {
    // Cache the catalog's model witness for repeated lookups under this
    // connection. The witness describes already visible embeddings. The in-flight
    // model choice for rows in a single write lives in ModelAdmission.
    pub(super) fn cache_collection_model(&mut self, collection_id: &str) -> Result<()> {
        self.catalog.cache_model_state(collection_id)
    }
}

// Use one admission state for the entire logical request. A manifest can span
// several storage batches, but crossing such a boundary cannot give the remaining
// rows permission to select another embedding model.
pub(super) struct ModelAdmission {
    // Model name this request must continue to use. It can come from the
    // collection's existing binding or from the first accepted input in an unbound
    // collection.
    expected: Option<ReservedString>,
    // Visible embedding models disagree with one another or with the collection's
    // declared model. New embeddings must wait until that conflict is resolved.
    historical_conflict: bool,
}

impl ModelAdmission {
    pub(super) fn new(
        catalog: &Catalog,
        collection: &CollectionInfo,
        context: &crate::json::JsonOperationContext,
    ) -> Result<Self> {
        // A collection proposed by a manifest has an ID before its create record
        // is committed. In that case a catalog lookup would report a missing
        // collection. Its proposed configuration supplies the initial constraint
        // instead.
        let state = if catalog.collections.contains_key(&collection.collection_id) {
            catalog.visible_model_state(&collection.collection_id)?
        } else {
            collection
                .embedding_model
                .as_deref()
                .map_or(ModelState::Unbound, ModelState::Bound)
        };
        // Copy a bound name into operation-owned storage. Catalog replay owns the
        // original strings and a newly proposed collection may itself be
        // temporary. The request needs an independent name to check later rows.
        Ok(Self {
            expected: match state {
                ModelState::Bound(model) => Some(ReservedString::copy(model, context)?),
                _ => None,
            },
            // Both a conflict and an unbound collection lack a single expected
            // name. Carry the conflict separately so the first new embedding
            // cannot silently pick one side of an existing disagreement.
            historical_conflict: matches!(state, ModelState::Conflict(_)),
        })
    }

    // Call this only for a row that will produce an embedding put. The conflict
    // policy may skip a row for an existing chunk. Letting that row choose a model
    // would reject a later row that actually reaches the database.
    pub(super) fn validate_write(
        &mut self,
        model: &str,
        declared: Option<&str>,
        context: &crate::json::JsonOperationContext,
    ) -> Result<()> {
        // No incoming name can resolve disagreement among live stored rows. Report
        // that historical conflict before comparing the current row against a
        // manifest declaration or a name inferred earlier in this request.
        if self.historical_conflict {
            return Err(Error::new(
                ErrorCode::EmbeddingModelMismatch,
                "collection has conflicting visible embedding models; export and explicitly clean up its data before adding embeddings",
            ));
        }
        // A supplied manifest declaration and the request's current binding are
        // independent constraints. The row must satisfy each present name.
        // Checking only the first would allow a manifest to override an
        // established collection.
        for expected in [declared, self.expected.as_ref().map(AsRef::as_ref)]
            .into_iter()
            .flatten()
        {
            if expected != model {
                return Err(Error::new(
                    ErrorCode::EmbeddingModelMismatch,
                    "input embedding model does not match the logical request",
                )
                .with_detail("expected", expected)
                .with_detail("actual", model));
            }
        }
        // When no model was established before this request, the first named model
        // becomes the comparison point for the remaining rows in the batch.
        if self.expected.is_none() {
            // Assign only after the copy succeeds. A budget retry for this first
            // name must see the same unbound state, with no partial change to the
            // catalog or to earlier accepted rows.
            self.expected = Some(ReservedString::copy(model, context)?);
        }
        Ok(())
    }
}
