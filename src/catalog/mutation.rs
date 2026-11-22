use std::mem::size_of;

use super::*;
use crate::json::{
    btree_workspace, CatalogReservation, JsonInputKind, JsonOperationContext, JsonReservation,
    ReservedString, ReservedVec, WorkspaceReservation,
};
use crate::model::VectorIndexConfig;
use crate::storage::RecordReservation;

pub(super) struct CatalogMemory {
    // Catalog's charge against the shared ledger. During a transaction it can also
    // cover payloads awaiting transfer or values needed for rollback.
    reservation: CatalogReservation,
    // Bytes attributed to the current catalog contents, excluding temporary journal
    // storage and unconsumed input credit.
    live: usize,
}

impl std::fmt::Debug for CatalogMemory {
    // The live total can fall before the reservation does. Diagnostic output shows
    // the two quantities independently while a transaction still owns displaced
    // rows or credit for a retry.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogMemory")
            .field("live", &self.live)
            .field("reserved", &self.reservation.reserved_bytes())
            .finish()
    }
}

#[cfg(test)]
impl Clone for CatalogMemory {
    fn clone(&self) -> Self {
        // Test reference catalogs are independent snapshots, not production copies.
        let mut reservation = JsonOperationContext::standalone().catalog_reservation();
        reservation.grow(self.live).unwrap();
        Self {
            reservation,
            live: self.live,
        }
    }
}

type Map<V> = BTreeMap<String, V>;
// The higher-ranked lifetime ties the selected map to the catalog passed at
// rollback. Store a selector and an owned scope in the journal so later edits can
// still borrow other parts of that catalog.
type Select<V> = for<'a> fn(&'a mut Catalog, &str) -> &'a mut Map<V>;
type SelectSet = fn(&mut Catalog) -> &mut BTreeSet<String>;

struct MapUndo<V> {
    // Selects the map to restore, including collection-specific alias maps. An undo
    // entry cannot keep a mutable reference into the catalog across later edits.
    select: Select<V>,
    // Collection key for a nested map. Selectors for top-level maps ignore this
    // argument.
    scope: String,
    // Lookup key used to find the entry changed by this transaction, even if that
    // entry no longer exists.
    key: String,
    // Ownership needed to undo insertion, replacement or removal. A removed map key
    // is retained with its value for reinsertion.
    previous: Previous<V>,
}

enum Previous<V> {
    // There was no entry before insertion. The forward operation owns the
    // installed key and value until rollback removes them.
    Missing,
    // The original key allocation stayed in the tree when its value changed.
    // Restore the displaced value in that same entry.
    Replaced(V),
    // Removal took both allocations out of the tree. Reinsert the original key and
    // value together so their capacities still agree with the catalog's saved
    // live-byte count.
    Removed(String, V),
}

impl<V> MapUndo<V> {
    fn rollback(self, catalog: &mut Catalog) {
        // Resolve the map at rollback time. When this edits an alias inside a
        // collection, reverse journal order keeps the outer collection entry
        // available until its inner edits have been undone.
        let map = (self.select)(catalog, &self.scope);
        match self.previous {
            // Remove by the journal's copy of the lookup key. Taking the installed
            // entry also drops the value introduced by the failed transaction.
            Previous::Missing => {
                map.remove(&self.key);
            }
            Previous::Replaced(value) => {
                // Later edits on this key have already been reversed. The key must
                // therefore still exist for this replacement and the saved value
                // can be moved back without growing the tree.
                *map.get_mut(&self.key).unwrap() = value;
            }
            Previous::Removed(key, value) => {
                map.insert(key, value);
            }
        }
        // Return an empty tree's remaining storage before restoring the saved
        // memory charge. A map that just lost its last entry can otherwise retain
        // a root allocation.
        if map.is_empty() {
            *map = Map::new();
        }
    }
}

// Each typed inverse owns the old value until the transaction commits or unwinds.
// The concrete variants keep map restoration type-checked for rows, auxiliary
// counts, locators and nested lookup tables alike.
enum Undo {
    Collection(MapUndo<CollectionInfo>),
    Document(MapUndo<Box<Document>>),
    Chunk(MapUndo<Box<Chunk>>),
    Embedding(MapUndo<Box<Embedding>>),
    String(MapUndo<String>),
    // Restore the locator together with the value it describes. Otherwise a
    // rolled-back row could select bytes from a record in the abandoned batch.
    Reference(MapUndo<RecordRef>),
    // An aborted batch must restore the old allocation high watermark for its ID
    // prefix. Keeping the provisional advance would affect which IDs the next
    // attempt assigns.
    Watermark(MapUndo<u64>),
    // Per-target tombstone counters participate in individual map edits.
    // Whole-catalog historical totals are restored once from the transaction's
    // saved baselines.
    Count(MapUndo<usize>),
    Children(MapUndo<Vec<String>>),
    // A collection's first external alias can create a new inner map. Journal the
    // outer entry as well so removing that alias on rollback also restores the
    // previous outer tree.
    AliasGroup(MapUndo<Map<String>>),
    Set {
        // Selects the membership set in the catalog to which the inverse operation
        // applies.
        select: SelectSet,
        // Key whose tentative membership must be removed before restoring any
        // preexisting entry.
        key: String,
        // Owned original key if the set contained it before the edit. None means
        // rollback should leave the key absent.
        previous: Option<String>,
    },
    ChildAppend {
        // Parent whose membership vector received the final element. Reverse
        // journal order makes a pop restore the old list.
        document: String,
    },
    ChildRemove {
        // Parent whose membership vector lost the child.
        document: String,
        // Original slot in the parent's list. Restoring the same position preserves
        // the order of equal-ordinal children.
        index: usize,
        // Owned child ID to put back into the parent's vector without allocating a
        // replacement string.
        child: String,
    },
    ChildCapacity {
        // Parent whose child vector was replaced by a larger backing allocation.
        document: String,
        // Old vector kept for its original backing allocation. Rollback moves the
        // children back into it to restore its capacity as well as its contents.
        previous: Vec<String>,
    },
    ChunkEmbedding {
        // Internal chunk ID identifying the summary whose embedding link changed.
        chunk: String,
        // Original optional embedding ID moved out of the chunk summary. Restore
        // this owned link even if the tentative value referred to another
        // embedding.
        previous: Option<String>,
    },
}

impl Undo {
    fn rollback(self, catalog: &mut Catalog) {
        // Undo records own the keys, values or backing allocations displaced by the
        // forward edit. Applying them in reverse mutation order restores the
        // pre-edit representation even when the same key was changed more than
        // once.
        match self {
            Self::Collection(edit) => edit.rollback(catalog),
            Self::Document(edit) => edit.rollback(catalog),
            Self::Chunk(edit) => edit.rollback(catalog),
            Self::Embedding(edit) => edit.rollback(catalog),
            Self::String(edit) => edit.rollback(catalog),
            Self::Reference(edit) => edit.rollback(catalog),
            Self::Watermark(edit) => edit.rollback(catalog),
            Self::Count(edit) => edit.rollback(catalog),
            Self::Children(edit) => edit.rollback(catalog),
            // Undo creation of an outer alias group after undoing its contained
            // key edits. Reversing that order would lose the nested map selected
            // by their saved scopes.
            Self::AliasGroup(edit) => edit.rollback(catalog),
            Self::Set {
                select,
                key,
                previous,
            } => {
                // The currently installed key and the displaced key may have equal
                // text but different owned storage. Removing the installed copy
                // first lets the original allocation become the set member again.
                let set = select(catalog);
                set.remove(&key);
                if let Some(previous) = previous {
                    set.insert(previous);
                }
                if set.is_empty() {
                    *set = BTreeSet::new();
                }
            }
            // Appending is undone at the tail because later edits have already been
            // reversed. Removing a child, by contrast, saved its index and must
            // restore the original ordering.
            Self::ChildAppend { document } => {
                catalog.document_chunks.get_mut(&document).unwrap().pop();
            }
            Self::ChildRemove {
                document,
                index,
                child,
            } => {
                catalog
                    .document_chunks
                    .get_mut(&document)
                    .unwrap()
                    .insert(index, child);
            }
            Self::ChildCapacity {
                document,
                mut previous,
            } => {
                // At this point the child-level edits have already been undone.
                // Move the contents back into the saved allocation, then return it
                // to the map. Restoring just the elements would leave the wrong
                // capacity charged.
                let children = catalog.document_chunks.get_mut(&document).unwrap();
                previous.append(children);
                *children = previous;
            }
            // The link carried by a chunk summary is separate from the
            // chunk-to-embedding lookup map. Restore the original optional ID to
            // the summary without allocating a second string.
            Self::ChunkEmbedding { chunk, previous } => {
                catalog.chunks.get_mut(&chunk).unwrap().embedding_id = previous;
            }
        }
    }
}

// Heap measurements here are for the value beyond its slot in a tree node. The
// mutation path accounts separately for the map nodes and the owned lookup key.
trait Entry: Sized {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize>;
    // Use a typed conversion from the displaced map entry to its inverse. The
    // updater can journal different catalog maps while preserving each value's
    // original owner until commit or rollback.
    fn undo(edit: MapUndo<Self>) -> Undo;
}

macro_rules! scalar_entry {
    ($type:ty, $variant:ident) => {
        impl Entry for $type {
            fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
                Ok(0)
            }
            fn undo(edit: MapUndo<Self>) -> Undo {
                Undo::$variant(edit)
            }
        }
    };
}
// A locator and the scalar counters have fixed-size representations. They require
// a map slot and possibly a key allocation, but no additional heap allocation for
// the stored value.
scalar_entry!(RecordRef, Reference);
scalar_entry!(u64, Watermark);
scalar_entry!(usize, Count);

impl Entry for String {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(self.capacity())
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::String(edit)
    }
}
// A vector of child IDs owns both its array of String headers and each child's
// byte buffer. Count spare capacity in the array and in the strings when
// transferring or restoring its charge.
impl Entry for Vec<String> {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(
            self.capacity() * size_of::<String>()
                + self.iter().map(String::capacity).sum::<usize>(),
        )
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Children(edit)
    }
}
// An external-ID group owns its own B-tree inside the outer collection map. Its
// inner keys and values each own a string buffer, independent of the allocation
// charged for the outer collection key.
impl Entry for Map<String> {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(btree_workspace::<String, String>(self.len())
            + self
                .iter()
                .map(|(key, value)| key.capacity() + value.capacity())
                .sum::<usize>())
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::AliasGroup(edit)
    }
}

fn json_bytes(value: &crate::JsonValue, context: &JsonOperationContext) -> Result<usize> {
    // Use the stored JSON accounting rules when measuring retained metadata. The
    // serialized length cannot stand in for keys, nested arrays and object
    // allocations held in memory.
    value.retained_bytes_in(JsonInputKind::StoredJson, context)
}
// An absent optional string carries no separate buffer. For a present one, spare
// capacity belongs to the value just as it does for a required string.
fn optional_bytes(value: &Option<String>) -> usize {
    value.as_ref().map_or(0, String::capacity)
}

impl Entry for CollectionInfo {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize> {
        Ok(self.collection_id.capacity()
            + self.name.capacity()
            + self.created_at.capacity()
            + optional_bytes(&self.embedding_model)
            + match &self.vector_index {
                // Exact indexing carries no provider configuration. An ANN
                // declaration retains both its provider name and the parameter
                // tree and must include both in the collection's live storage.
                VectorIndexConfig::Exact => 0,
                VectorIndexConfig::Ann { provider, params } => {
                    provider.capacity() + json_bytes(params, context)?
                }
            })
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Collection(edit)
    }
}
impl Entry for Box<Document> {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize> {
        // A document value lives behind a Box. Include the allocation for the
        // Document itself as well as its owned strings and metadata; the map node
        // holds only the pointer.
        Ok(size_of::<Document>()
            + self.document_id.capacity()
            + self.collection_id.capacity()
            + self.external_id.capacity()
            + optional_bytes(&self.title)
            + optional_bytes(&self.source_uri)
            + self.created_at.capacity()
            + self.updated_at.capacity()
            + json_bytes(&self.metadata, context)?)
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Document(edit)
    }
}
impl Entry for Box<Chunk> {
    fn heap(&self, context: &JsonOperationContext) -> Result<usize> {
        Ok(size_of::<Chunk>()
            + self.chunk_id.capacity()
            + self.document_id.capacity()
            + self.collection_id.capacity()
            + self.chunk_external_id.capacity()
            // Stored summaries can omit the original chunk text. Charge the
            // capacity actually present in this catalog row rather than the
            // payload length recorded by the storage locator.
            + self.text.capacity()
            + self.text_hash.capacity()
            + optional_bytes(&self.embedding_id)
            + self.created_at.capacity()
            + json_bytes(&self.metadata, context)?)
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Chunk(edit)
    }
}
impl Entry for Box<Embedding> {
    fn heap(&self, _: &JsonOperationContext) -> Result<usize> {
        Ok(size_of::<Embedding>()
            + self.embedding_id.capacity()
            + self.chunk_id.capacity()
            + self.collection_id.capacity()
            + self.chunk_external_id.capacity()
            + self.model_name.capacity()
            + self.created_at.capacity()
            // A vector's spare components still consume workspace. Charge capacity
            // in f32 elements converted to bytes, even when fewer coordinates are
            // currently initialized.
            + self.vector.capacity() * size_of::<f32>())
    }
    fn undo(edit: MapUndo<Self>) -> Undo {
        Undo::Embedding(edit)
    }
}

// Mutations are private under the database lock until the critical append succeeds.
// The journal owns only changed entries; it never clones an entire catalog or child list.
pub(crate) struct CatalogTransaction<'a> {
    // Exclusively borrowed catalog being edited. Its maps may contain tentative
    // changes until the transaction is consumed or dropped.
    catalog: &'a mut Catalog,
    // Operation that admits journal storage, durable catalog growth and transfers
    // of previously charged inputs.
    context: &'a JsonOperationContext,
    // Inverse edits in application order. Rollback consumes them from the end so
    // dependent map entries and memberships are restored in reverse order.
    undo: Vec<Undo>,
    // Charge for the journal and scratch retained while earlier values may still be
    // needed for rollback.
    journal_work: WorkspaceReservation,
    // Live catalog byte count on entry. Rollback restores it while handling any
    // retained input credit separately.
    initial_bytes: usize,
    // Business record count before tentative edits. Undoing map changes alone
    // cannot restore this scalar.
    initial_business: usize,
    // Historical tombstone count on entry, restored together with the per-target
    // counters.
    initial_tombstones: usize,
    // Set after finalizing the transaction so Drop does not undo a published
    // mutation.
    committed: bool,
    // Whether this catalog needs to remain usable on failure. Replaying into a new
    // catalog that will be discarded can avoid recording inverse edits.
    rollback_on_drop: bool,
    // Per-record JSON bytes transferred ahead of the durable append. Keep their
    // order so finish_references can attach the corresponding charge, or a
    // pre-append retry can reuse it.
    pending_json: Option<ReservedVec<usize>>,
}

impl Catalog {
    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.memory
            .as_ref()
            .map_or(0, |memory| memory.reservation.reserved_bytes())
    }

    pub(crate) fn attach_context(&self, context: &JsonOperationContext) -> Result<()> {
        // Binding an operation to a managed catalog can fail before any records
        // are changed. A catalog constructed without memory accounting carries no
        // shared reservation to attach.
        match &self.memory {
            Some(memory) => memory.reservation.attach_context(context),
            None => Ok(()),
        }
    }

    pub(crate) fn apply_retained(
        &mut self,
        record: CommittedRecord,
        work: &mut RecordReservation,
        context: &JsonOperationContext,
    ) -> Result<()> {
        let mut transaction = self.transaction(context);
        // Every caller is building a new catalog and discards it on recovery
        // failure. Only mutations of an already published catalog need undo.
        transaction.rollback_on_drop = false;
        transaction.apply(record.payload, Some(record.record_ref), work)?;
        transaction.commit();
        // A replayed record can change which embedding or parent is live. Drop
        // cached model witnesses after committing the replay step so the next
        // lookup derives them from this view.
        self.clear_model_states();
        Ok(())
    }

    pub(crate) fn new_in(context: &JsonOperationContext) -> Self {
        Self {
            // A managed catalog starts with a zero-byte live footprint and a
            // reservation tied to the caller's ledger. Map and payload growth
            // during replay will be attributed to that same owner.
            memory: Some(CatalogMemory {
                reservation: context.catalog_reservation(),
                live: 0,
            }),
            ..Self::default()
        }
    }

    pub(crate) fn transaction<'a>(
        &'a mut self,
        context: &'a JsonOperationContext,
    ) -> CatalogTransaction<'a> {
        CatalogTransaction {
            // Save the live-byte and scalar-count baselines before tentative
            // mutation. Map inverses preserve displaced allocations, while these
            // snapshots restore totals that have no independent row identity.
            initial_bytes: self.memory.as_ref().map_or(0, |memory| memory.live),
            initial_business: self.business_record_count,
            initial_tombstones: self.tombstone_count,
            catalog: self,
            context,
            undo: Vec::new(),
            // The journal's vector capacity and any copied keys are temporary
            // workspace. Keeping their owner separate lets rollback release them
            // before retrying the same operation on the restored catalog.
            journal_work: context.workspace_reservation(),
            committed: false,
            rollback_on_drop: true,
            pending_json: None,
        }
    }
}

impl CatalogTransaction<'_> {
    pub(crate) fn prepare_reclaiming<'b>(
        &mut self,
        records: impl ExactSizeIterator<Item = &'b RecordPayload> + Clone,
        json_work: &mut JsonReservation,
        reclaim: impl FnOnce() -> bool,
    ) -> Result<()> {
        assert!(
            self.rollback_on_drop,
            "reclaiming preparation requires rollback"
        );
        // Prepare from a clone of the iterator so progress on the first attempt
        // cannot advance the retry cursor. Reuse the original payload owners and
        // input credit across both passes.
        match self.prepare(records.clone(), json_work) {
            // Only a budget refusal may be retried, and only if the one-shot
            // callback actually reclaimed space. Other errors unwind through the
            // ordinary transaction drop path.
            Err(error) if crate::json::is_memory_budget_error(&error) && reclaim() => {
                // Input metadata already transferred here must stay charged while
                // the same immutable batch is prepared again, without another IO.
                self.rollback_changes(true);
                // Call the internal preparation path after rewinding. The ordinary
                // entry point expects no outstanding per-row JSON credit, but this
                // retry deliberately retained credit absorbed on the first pass.
                self.prepare_records(records, json_work)
            }
            result => result,
        }
    }

    pub(crate) fn prepare<'b>(
        &mut self,
        records: impl ExactSizeIterator<Item = &'b RecordPayload>,
        json_work: &mut JsonReservation,
    ) -> Result<()> {
        // A new preparation cannot inherit a partially charged prefix from another
        // input. Only the dedicated retry path may reuse that prefix with the same
        // records in the same order.
        assert!(self.pending_json.is_none());
        self.prepare_records(records, json_work)
    }

    fn prepare_records<'b>(
        &mut self,
        records: impl ExactSizeIterator<Item = &'b RecordPayload>,
        json_work: &mut JsonReservation,
    ) -> Result<()> {
        // Reserve one accounting slot per input record before transferring the
        // first row's metadata credit. Subsequent pushes can record a transfer
        // without another fallible array growth.
        if self.pending_json.is_none() {
            self.pending_json = Some(ReservedVec::with_capacity(
                records.len(),
                self.context,
                "catalog_build",
            )?);
        }
        for (index, record) in records.enumerate() {
            // A retry can revisit a prefix whose input credit was already absorbed
            // into the catalog. pending_json records exactly that prefix in input
            // order. Charge only newly reached rows, even though their catalog
            // changes may have to be applied again.
            if index >= self.pending_json.as_ref().unwrap().as_slice().len() {
                let bytes = record.metadata_bytes_in(self.context)?;
                // The input may already carry credit for its live JSON. Grow it
                // only by the shortfall before transferring this row's bytes to
                // the catalog reservation.
                json_work.grow(
                    bytes.saturating_sub(json_work.covered_bytes()),
                    JsonInputKind::StoredJson,
                )?;
                if let Some(memory) = &mut self.catalog.memory {
                    memory.reservation.absorb(json_work, None, bytes, 0)?;
                }
                // Record the transferred amount only after admission succeeds. If
                // a later summary or map operation fails, the retry can recover
                // how much credit is already owned for this input position.
                self.pending_json.as_mut().unwrap().push(bytes);
            }
            // Copy just the catalog fields while the append still needs ownership
            // of the caller's record. These provisional rows can be undone if a
            // later record or the primary write fails.
            let (summary, mut work) = record.copy_catalog_fields_in(self.context)?;
            // The data frame does not exist yet. Store the one-based batch position
            // as a provisional sequence so publication can find the real record. A
            // zero offset and the pending type must never be used to read this
            // payload from disk.
            // Only record kinds that supply an object ID need a pending locator.
            // Recovery metadata such as a delete can affect catalog visibility
            // without creating a new materializable payload.
            let reference = record_id(record).map(|_| RecordRef {
                offset: 0,
                record_type: "pending",
                batch_id: 0,
                sequence_no: index as u64 + 1,
                payload: crate::storage::PayloadRef {
                    offset: 0,
                    len: 0,
                    checksum: 0,
                },
            });
            // The summary and provisional locator are installed together. The
            // record's final offset and sequence become available only after
            // storage has emitted the batch successfully.
            self.apply(summary, reference, &mut work)?;
        }
        Ok(())
    }

    fn rollback_changes(&mut self, keep_input_credit: bool) {
        if self.rollback_on_drop {
            // One transaction may move an alias more than once. Undo the last move first
            // so each preceding edit sees the owner it originally replaced.
            while let Some(edit) = self.undo.pop() {
                edit.rollback(self.catalog);
            }
            // The business and tombstone totals can both change in a single batch.
            // Restore them after reversing the per-key changes so history counters
            // also describe the pre-transaction view.
            self.catalog.business_record_count = self.initial_business;
            self.catalog.tombstone_count = self.initial_tombstones;
        }
        // A retry discards the used journal allocation as well as its entries. The
        // next preparation must compete for clean journal workspace while the
        // input metadata remains charged.
        if keep_input_credit {
            self.undo = Vec::new();
            self.journal_work.retain(0);
        }
        if let Some(memory) = &mut self.catalog.memory {
            if self.rollback_on_drop {
                memory.live = self.initial_bytes;
            }
            // On a retry, transferred metadata still occupies memory even after
            // tentative rows have been undone. Keep that prefix's credit alongside
            // the restored live catalog until the next preparation uses it.
            let input_bytes = if keep_input_credit {
                self.pending_json
                    .as_ref()
                    .map_or(0, |bytes| bytes.as_slice().iter().sum())
            } else {
                0
            };
            memory.reservation.retain(memory.live + input_bytes);
        }
    }

    pub(crate) fn finish_references(&mut self, records: &mut Vec<CommittedRecord>) {
        // Preparation and append see records in the same order. Drain the
        // committed payloads so their stored metadata can move directly into the
        // catalog, using the input index to find both the provisional reference
        // and its previously admitted JSON bytes.
        for (index, record) in records.drain(..).enumerate() {
            let current =
                record_id(&record.payload).and_then(|id| self.catalog.record_refs.get_mut(id));
            // Several records in this batch may refer to the same ID. Only its surviving
            // provisional slot can adopt the on-disk offset and the metadata of this record.
            // A zero batch ID marks a placeholder. Comparing its one-based
            // position prevents an earlier put for an ID from overwriting the
            // locator installed for a later put in the same append.
            let Some(reference) = current.filter(|reference| {
                reference.batch_id == 0 && reference.sequence_no == index as u64 + 1
            }) else {
                // A tombstone, a record without its own locator, or an ID
                // overwritten later in the batch can reach this path. Discard its
                // committed payload without transferring its metadata into the
                // surviving summary.
                continue;
            };
            // Install the physical reference supplied by storage only after append
            // has supplied the actual offsets. Until this point the proposed row's
            // placeholder cannot be used for a payload read.
            *reference = record.record_ref;
            match record.payload {
                // The provisional document already owns its catalog fields. Fill
                // just the retained metadata from the committed payload and keep
                // the compact summary in place.
                RecordPayload::DocumentPut(value) => {
                    self.catalog
                        .documents
                        .get_mut(&value.document_id)
                        .unwrap()
                        .metadata = value.metadata
                }
                // Keep chunk metadata in the catalog so metadata filters need no
                // body read. The text and other large fields can still be
                // materialized from the physical locator installed above.
                RecordPayload::ChunkPut(value) => {
                    self.catalog
                        .chunks
                        .get_mut(&value.chunk_id)
                        .unwrap()
                        .metadata = value.metadata
                }
                // Only ANN configuration carries parameter JSON to transfer. The
                // provisional definition was built from this same put, so the
                // variant must still match when its parameters are installed.
                RecordPayload::CollectionCreate(value) => {
                    if let VectorIndexConfig::Ann { params, .. } = value.vector_index {
                        let target = &mut self
                            .catalog
                            .collections
                            .get_mut(&value.collection_id)
                            .unwrap()
                            .vector_index;
                        // The committed ANN parameters belong to the prepared
                        // definition selected by the provisional sequence check.
                        // If the same collection ID was written twice, only the
                        // surviving put can reach this assignment.
                        let VectorIndexConfig::Ann { params: target, .. } = target else {
                            unreachable!()
                        };
                        *target = params;
                    }
                }
                _ => {}
            }
            // Preparation transferred each input's JSON credit before storage
            // wrote the batch. Count the part belonging to a surviving locator as
            // live now. Superseded rows never reach this accounting step.
            self.change_live(0, self.pending_json.as_ref().unwrap().as_slice()[index]);
        }
        // Clearing this table signals that attachment is complete. Commit checks
        // this state before returning excess catalog credit and dropping the
        // journal.
        self.pending_json = None;
    }

    fn catalog_key(
        &mut self,
        select: impl FnOnce(&Catalog) -> Option<&str>,
    ) -> Result<Option<String>> {
        // Copy any selected key while the catalog borrow is valid. The caller may
        // subsequently mutate the same map, and needs a stable name even when
        // moving its previous value.
        let copied = select(self.catalog)
            .map(|value| ReservedString::copy(value, self.context))
            .transpose()?;
        // No selected key needs no workspace and returns None. A copy failure is a
        // resource error, not evidence that the relationship was absent.
        Ok(copied.map(|copied| {
            let (value, work) = copied.into_parts();
            // Keep the copied lookup key on journal workspace. It serves the
            // transaction while looking up or removing an existing relationship
            // and does not itself become a live catalog entry.
            self.journal_work.merge(work);
            value
        }))
    }

    fn charge(&mut self, bytes: usize) -> Result<()> {
        // A catalog can be constructed without an attached memory reservation.
        // Admitted catalogs must extend that reservation before allocating
        // additional retained storage.
        if let Some(memory) = &mut self.catalog.memory {
            memory.reservation.grow(bytes)?;
        }
        Ok(())
    }

    fn change_live(&mut self, before: usize, after: usize) {
        if let Some(memory) = &mut self.catalog.memory {
            // Live bytes describe only the currently installed representation. The
            // reservation may remain larger while inverse edits own old strings or
            // array allocations that could be needed by rollback.
            memory.live = memory.live - before + after;
        }
    }

    fn copy(&mut self, value: &str) -> Result<String> {
        // Catalog-owned strings and rollback keys use different reservations.
        // Charge this copy to the catalog because the copied key or ID may remain
        // installed after this transaction ends.
        self.charge(value.len())?;
        let value = value.to_owned();
        // The ledger admitted exactly the UTF-8 byte length for this new owned
        // key. Reject an allocation with extra capacity instead of quietly
        // retaining more catalog memory than was reserved.
        if value.len() != value.capacity() {
            return Err(
                std::io::Error::other("catalog string allocation exceeds reservation").into(),
            );
        }
        Ok(value)
    }

    fn journal_key(&mut self, value: &str) -> Result<String> {
        // Journal lookup keys need their own charge because they must survive
        // mutation of the map they name. Keep the reservation even if a later edit
        // removes that key from the live tree.
        let (key, work) = ReservedString::copy(value, self.context)?.into_parts();
        self.journal_work.merge(work);
        Ok(key)
    }

    fn undo_key(&mut self, value: &str) -> Result<String> {
        // Replay into a disposable catalog will discard the entire partial result
        // on failure. An empty rollback key allows those edits to avoid allocating
        // strings that would never be used.
        if self.rollback_on_drop {
            self.journal_key(value)
        } else {
            Ok(String::new())
        }
    }

    fn save_undo(&mut self, edit: Undo) {
        // Every reversible caller reserves the vector slot before editing the
        // catalog. Appending its inverse here must be infallible once the forward
        // mutation has taken effect.
        if self.rollback_on_drop {
            self.undo.push(edit);
        }
    }

    fn journal_slot(&mut self) -> Result<()> {
        // A journal slot is needed only when edits must be reversible and the undo
        // vector is full. Already admitted spare slots can accept more edits
        // without another workspace request.
        if !self.rollback_on_drop || self.undo.len() < self.undo.capacity() {
            return Ok(());
        }
        let old = self.undo.capacity();
        let minimum = old + 1;
        // Ask for geometric growth to amortize repeated edits, but require only
        // one additional slot. Under memory pressure a transaction can continue
        // using a smaller grant.
        let preferred = minimum.max(old.saturating_mul(2)).max(8);
        // The journal reservation also covers owned names. Subtract only the old
        // vector slots to preserve that separate charge while negotiating a new
        // array capacity.
        let strings = self.journal_work.reserved_bytes() - old * size_of::<Undo>();
        let granted = self.journal_work.grow_to_wide(
            strings as u128 + minimum as u128 * size_of::<Undo>() as u128,
            strings as u128 + preferred as u128 * size_of::<Undo>() as u128,
            "catalog_build",
        )?;
        // Convert the byte grant back to whole entries before asking the
        // allocator. Any bytes left after rounding cannot be used for a partial
        // undo record.
        let capacity = (granted - strings) / size_of::<Undo>();
        self.undo
            // try_reserve_exact takes a number of additional elements relative to
            // length. Translate the total admitted capacity accordingly while
            // existing records are still in the array.
            .try_reserve_exact(capacity - self.undo.len())
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        // Allocator capacity is authoritative. Check it against the grant before
        // keeping the array, then retain only the slots actually obtained along
        // with the still-live strings.
        if self.undo.capacity() > capacity {
            return Err(
                std::io::Error::other("catalog journal allocation exceeds reservation").into(),
            );
        }
        self.journal_work
            .retain(strings + self.undo.capacity() * size_of::<Undo>());
        Ok(())
    }

    fn update<V: Entry>(
        &mut self,
        select: Select<V>,
        scope: &str,
        key: &str,
        value: Option<V>,
    ) -> Result<()> {
        let map = select(self.catalog, scope);
        let before_len = map.len();
        let previous_key = map.get_key_value(key).map(|(key, _)| key.capacity());
        // Removing a nonexistent key changes neither the tree nor its accounting.
        // In particular it must not create an inverse edit that would later assume
        // a previous allocation.
        if previous_key.is_none() && value.is_none() {
            return Ok(());
        }
        // Measure the owned key actually in the tree and the old value's retained
        // heap. The caller's lookup string can have a different capacity even when
        // their bytes match.
        let before = map
            .get_key_value(key)
            .map(|(key, value)| value.heap(self.context).map(|bytes| bytes + key.capacity()))
            .transpose()?
            .unwrap_or(0);
        // Tree workspace depends on entry count. A replacement leaves the count
        // unchanged even when the new value has a different heap footprint.
        // Account for those two changes separately.
        let after_len =
            before_len - usize::from(map.contains_key(key)) + usize::from(value.is_some());
        // Measure the incoming value before moving it into the tree. A fallible
        // inspection of nested JSON must return while the original map entry can
        // still be read.
        let after_value = value
            .as_ref()
            .map(|value| value.heap(self.context))
            .transpose()?
            .unwrap_or(0);
        // Reserve net growth in tree nodes, but retain the full owned keys needed
        // for rollback. Replacing a value under an existing key reuses the map key.
        // Inserting a new value needs an additional owned key.
        let before_tree = btree_workspace::<String, V>(before_len);
        let after_tree = btree_workspace::<String, V>(after_len);
        self.charge(after_tree.saturating_sub(before_tree))?;
        self.journal_slot()?;
        // A nested map needs both its collection scope and its leaf key during
        // undo. Save those names independently of the tree nodes that this
        // mutation can remove.
        let journal_scope = self.undo_key(scope)?;
        let journal_key = self.undo_key(key)?;
        // Admit a catalog-owned key only for insertion. Journal names remain
        // transaction scratch and cannot be moved into the live map as a
        // substitute for its key.
        let next_key = (value.is_some() && previous_key.is_none())
            .then(|| self.copy(key))
            .transpose()?;
        // Replacement keeps the original stored key allocation. Insertion uses the
        // newly admitted key and deletion leaves no live key, even though the
        // journal may hold the old allocation.
        let after_key = if value.is_some() {
            previous_key.unwrap_or_else(|| next_key.as_ref().unwrap().capacity())
        } else {
            0
        };
        let after = after_value + after_key;
        // The tree growth, undo slot and owned keys have all been admitted before
        // changing the map. An error while reserving one of them must leave the
        // original key and value available to rollback.
        let map = select(self.catalog, scope);
        let previous = match value {
            Some(value) => match map.get_mut(key) {
                // Replace the value in its existing slot and move the old object
                // into the journal. Rollback can restore its original owned
                // allocation without cloning the row.
                Some(current) => Previous::Replaced(std::mem::replace(current, value)),
                // The inverse of inserting a previously absent key is its removal.
                // Mark that distinction even when another edit to the same key
                // will be recorded later in the batch.
                None => {
                    map.insert(next_key.unwrap(), value);
                    Previous::Missing
                }
            },
            None => match map.remove_entry(key) {
                // Removal must retain the owned map key as well as its value. On
                // rollback, reinserting those allocations restores the original
                // representation used to compute live bytes.
                Some((key, value)) => Previous::Removed(key, value),
                None => Previous::Missing,
            },
        };
        // Release the empty tree's remaining allocation. The estimated node
        // workspace for zero entries assumes the catalog no longer retains an
        // empty root.
        if map.is_empty() {
            *map = Map::new();
        }
        self.save_undo(V::undo(MapUndo {
            select,
            scope: journal_scope,
            key: journal_key,
            previous,
        }));
        // Update the live estimate after the mutation and inverse edit have been
        // prepared. Reservation for the old representation may have to remain
        // until commit discards the undo entry.
        self.change_live(before + before_tree, after + after_tree);
        Ok(())
    }

    fn set(&mut self, select: SelectSet, key: &str, present: bool) -> Result<()> {
        let set = select(self.catalog);
        // Setting a deletion or reservation flag twice is idempotent. Avoid
        // charging another owned key or recording an inverse operation when
        // membership already matches the requested state.
        if set.contains(key) == present {
            return Ok(());
        }
        // Use stored capacity for a removed set key and the separate tree-size
        // estimate for node storage. Key length alone cannot reconstruct how much
        // catalog credit the set currently retains.
        let old_len = set.len();
        let old_bytes = set.get(key).map_or(0, String::capacity);
        let new_len = if present { old_len + 1 } else { old_len - 1 };
        let before_tree = btree_workspace::<String, ()>(old_len);
        let after_tree = btree_workspace::<String, ()>(new_len);
        self.charge(after_tree.saturating_sub(before_tree))?;
        self.journal_slot()?;
        let lookup = self.undo_key(key)?;
        let next = present.then(|| self.copy(key)).transpose()?;
        let new_bytes = next.as_ref().map_or(0, String::capacity);
        let set = select(self.catalog);
        // Taking the actual key lets rollback restore its original allocation. The
        // lookup key remains independently owned, including on an insertion for
        // which no previous key existed.
        let previous = set.take(key);
        if let Some(next) = next {
            set.insert(next);
        }
        if set.is_empty() {
            *set = BTreeSet::new();
        }
        self.save_undo(Undo::Set {
            select,
            key: lookup,
            previous,
        });
        self.change_live(before_tree + old_bytes, after_tree + new_bytes);
        Ok(())
    }

    fn alias(
        &mut self,
        document: bool,
        collection: &str,
        external: &str,
        id: Option<&str>,
    ) -> Result<()> {
        // The document and chunk alias maps share this implementation but have
        // separate namespaces. Choose both selectors from the same namespace so
        // the parent group and its entries can be undone together.
        let groups: Select<Map<String>> = if document {
            |catalog, _| &mut catalog.document_external_ids.collections
        } else {
            |catalog, _| &mut catalog.chunk_external_ids.collections
        };
        // Inner selectors expect the outer collection group to exist. Forward
        // edits below create that group before its first alias, and rollback
        // restores it before replaying any dependent inner edit.
        let ids: Select<String> = if document {
            |catalog, collection| {
                catalog
                    .document_external_ids
                    .collections
                    .get_mut(collection)
                    .unwrap()
            }
        } else {
            |catalog, collection| {
                catalog
                    .chunk_external_ids
                    .collections
                    .get_mut(collection)
                    .unwrap()
            }
        };
        // A missing group can represent removal of an already absent alias. Create
        // the per-collection map only when an owner will actually be installed.
        if !groups(self.catalog, "").contains_key(collection) {
            if id.is_none() {
                return Ok(());
            }
            self.update(groups, "", collection, Some(Map::new()))?;
        }
        // An unchanged owner needs no second string or journal entry. This applies
        // to an absent mapping being removed as well as to a repeated put of the
        // same internal ID.
        if ids(self.catalog, collection)
            .get(external)
            .map(String::as_str)
            == id
        {
            return Ok(());
        }
        // The new internal owner becomes a catalog value distinct from the
        // external lookup key. Copy it under catalog credit before changing the
        // association.
        let next = id.map(|id| self.copy(id)).transpose()?;
        self.update(ids, collection, external, next)?;
        // Remove an emptied collection group through the same journal.
        // Reverse-order replay restores the group before an earlier undo tries to
        // reinstall its external key.
        if ids(self.catalog, collection).is_empty() {
            self.update(groups, "", collection, None)?;
        }
        Ok(())
    }

    fn remove_alias(
        &mut self,
        document: bool,
        collection: &str,
        external: &str,
        id: &str,
    ) -> Result<()> {
        let aliases = if document {
            &self.catalog.document_external_ids
        } else {
            &self.catalog.chunk_external_ids
        };
        // A later put may already own this external name. Moving an older row to
        // another alias must leave the newer owner's mapping intact.
        if aliases
            .get(collection, external)
            .is_some_and(|owner| owner == id)
        {
            self.alias(document, collection, external, None)?;
        }
        Ok(())
    }

    fn append_child(&mut self, document: &str, chunk: &str) -> Result<()> {
        // Create the parent list as a reversible map edit. If adding the first
        // child later fails, rollback can remove the otherwise empty list as part
        // of the same transaction.
        if !self.catalog.document_chunks.contains_key(document) {
            self.update(
                |catalog, _| &mut catalog.document_chunks,
                "",
                document,
                Some(Vec::new()),
            )?;
        }
        let children = &self.catalog.document_chunks[document];
        // A child array may have to grow before the transaction commits. Keep the
        // previous allocation in the undo record even after moving its strings into
        // the bigger array. Rollback can put the children back into that allocation
        // and restore its original capacity.
        if children.len() == children.capacity() {
            let minimum = children.len() + 1;
            let mut capacity = minimum.max(children.capacity().saturating_mul(2)).max(4);
            // Try to reserve credit for geometric growth first, then fall back to
            // the exact minimum under pressure. Either choice needs room for one
            // replacement while the old buffer is kept for rollback.
            if self.charge(capacity * size_of::<String>()).is_err() {
                capacity = minimum;
                self.charge(capacity * size_of::<String>())?;
            }
            // Admit the capacity-change inverse and its owned document ID before
            // allocating or swapping buffers. A failed journal allocation can then
            // leave the original child list intact.
            self.journal_slot()?;
            let key = self.undo_key(document)?;
            let mut replacement = Vec::new();
            replacement
                .try_reserve_exact(capacity)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            // A larger capacity would leave unaccounted array slots in the
            // catalog. Verify the allocation before moving any children out of
            // their old buffer.
            if replacement.capacity() != capacity {
                return Err(
                    std::io::Error::other("catalog child array exceeds reservation").into(),
                );
            }
            let children = self.catalog.document_chunks.get_mut(document).unwrap();
            let old_capacity = children.capacity();
            // Move the existing String handles without copying their contents. The
            // emptied vector still owns its original capacity, which the inverse
            // edit can refill if publication fails.
            replacement.append(children);
            let previous = std::mem::replace(children, replacement);
            self.save_undo(Undo::ChildCapacity {
                document: key,
                previous,
            });
            // The old buffer remains charged during rollback-capable preparation,
            // but it is no longer the live membership array. Replace its slot
            // footprint in the live counter without counting the child strings a
            // second time.
            self.change_live(
                old_capacity * size_of::<String>(),
                capacity * size_of::<String>(),
            );
        }
        // Capacity growth and appending the child are separate inverse edits. A
        // failure to obtain the child or its journal key can roll back just the
        // earlier growth.
        self.journal_slot()?;
        let key = self.undo_key(document)?;
        let child = self.copy(chunk)?;
        let bytes = child.capacity();
        self.catalog
            .document_chunks
            .get_mut(document)
            .unwrap()
            .push(child);
        // Record a tail removal as the inverse. Later appends are undone first, so
        // this child will again be the last element when its own rollback runs.
        self.save_undo(Undo::ChildAppend { document: key });
        self.change_live(0, bytes);
        Ok(())
    }

    fn remove_child(&mut self, document: &str, chunk: &str) -> Result<()> {
        let Some(index) = self
            .catalog
            .document_chunks
            .get(document)
            .and_then(|children| children.iter().position(|child| child == chunk))
        else {
            return Ok(());
        };
        // Capture the parent's name and admit a journal slot before removing the
        // child. Its original array index is needed to reconstruct iteration order
        // when a later operation fails.
        self.journal_slot()?;
        let key = self.undo_key(document)?;
        // Use an order-preserving removal so all surviving children keep their
        // relative order. Undo retains the removed string itself and can insert it
        // at the saved position without copying the ID.
        let child = self
            .catalog
            .document_chunks
            .get_mut(document)
            .unwrap()
            .remove(index);
        let bytes = child.capacity();
        self.save_undo(Undo::ChildRemove {
            document: key,
            index,
            child,
        });
        // Removing a child changes the live string bytes but leaves the array
        // capacity in place. If this was its last child, journal the separate
        // removal of the parent entry and its buffer below.
        self.change_live(bytes, 0);
        if self.catalog.document_chunks[document].is_empty() {
            // Because this outer-map removal is logged after the child removal,
            // rollback reinstates the empty buffer first. The preceding inverse
            // can then reinsert the child at its original index.
            self.update(
                |catalog, _| &mut catalog.document_chunks,
                "",
                document,
                None,
            )?;
        }
        Ok(())
    }

    fn track_id(&mut self, id: &str) -> Result<()> {
        // Only the generated ID namespaces participate in allocation. An
        // unrecognized spelling may still be present in historical data without
        // advancing a numeric sequence.
        for prefix in ["col_", "doc_", "chk_", "emb_"] {
            if let Some(rest) = id.strip_prefix(prefix) {
                // Advance the watermark only after parsing the entire suffix as an
                // unsigned number. A recognized prefix with a nonnumeric suffix
                // provides no sequence value to reserve.
                if let Ok(value) = rest.parse::<u64>() {
                    self.watermark(prefix, value)?;
                }
                break;
            }
        }
        Ok(())
    }

    fn watermark(&mut self, prefix: &str, value: u64) -> Result<()> {
        if self
            .catalog
            .id_high_watermarks
            .get(prefix)
            // Watermarks never retreat when replay visits older or out-of-order
            // numbers. Journal only a new maximum so a failed batch cannot reserve
            // IDs solely through its tentative catalog edits.
            .map_or(true, |old| *old < value)
        {
            self.update(
                |catalog, _| &mut catalog.id_high_watermarks,
                "",
                prefix,
                Some(value),
            )?;
        }
        Ok(())
    }

    fn reference(&mut self, id: &str, reference: Option<RecordRef>) -> Result<()> {
        // A materialized record can carry its physical location, and a prepared
        // put can carry a provisional location until publication. Preserve the
        // locator when one is supplied. A separate deletion path removes a retired
        // locator, since an absent reference argument alone makes no edit.
        if let Some(reference) = reference {
            self.update(
                |catalog, _| &mut catalog.record_refs,
                "",
                id,
                Some(reference),
            )?;
        }
        Ok(())
    }

    pub(crate) fn apply(
        &mut self,
        mut record: RecordPayload,
        reference: Option<RecordRef>,
        work: &mut RecordReservation,
    ) -> Result<()> {
        // Compute the next record count before editing the catalog. A transaction
        // can replay the same kinds of records as a live write, including
        // administrative records that must not raise the business record count.
        let business = !matches!(
            record,
            RecordPayload::Commit(_) | RecordPayload::CompactMarker(_)
        );
        // Calculate the next total while the current row is still provisional. The
        // checked value is assigned at the end of the operation and the
        // transaction restores its initial total if later work rolls back.
        let next_business = self
            .catalog
            .business_record_count
            .checked_add(usize::from(business))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "catalog business record count overflow",
                )
            })?;
        // Large primary payloads are read lazily, never retained by the catalog.
        match &mut record {
            RecordPayload::ChunkPut(chunk) => chunk.text = String::new(),
            RecordPayload::EmbeddingPut(embedding) => embedding.vector = Vec::new(),
            _ => {}
        }
        // Measure the JSON that will live in the catalog. Chunk and document rows
        // retain metadata, and ANN configuration can retain a parameter object.
        // Embedding coordinates have already been dropped from the catalog
        // summary.
        let json = match &record {
            RecordPayload::DocumentPut(document) => json_bytes(&document.metadata, self.context)?,
            RecordPayload::ChunkPut(chunk) => json_bytes(&chunk.metadata, self.context)?,
            RecordPayload::CollectionCreate(CollectionInfo {
                vector_index: VectorIndexConfig::Ann { params, .. },
                ..
            }) => json_bytes(params, self.context)?,
            _ => 0,
        };
        // Retained JSON becomes catalog-owned along with the row. Transfer the
        // incoming reservation before installing it so an admission failure can
        // still unwind the provisional edits.
        if let Some(memory) = &mut self.catalog.memory {
            work.transfer_to_catalog(
                &mut memory.reservation,
                json,
                crate::storage::record_workspace_bytes(&record),
            )?;
        }
        match record {
            RecordPayload::CollectionCreate(collection) => {
                let id = self.journal_key(&collection.collection_id)?;
                self.track_id(&id)?;
                self.reference(&id, reference)?;
                // The collection-name map owns a separate copy of the internal ID
                // as its value. Replacing a name mapping can leave its former
                // value in the undo journal, so account for that copy
                // independently of the ID stored in the definition.
                let owner = self.copy(&id)?;
                self.update(
                    |catalog, _| &mut catalog.collection_names,
                    "",
                    &collection.name,
                    Some(owner),
                )?;
                // Clear the old deletion state for this identity in the same
                // transaction that installs its name and definition. A read after
                // publication should use the visibility associated with the new
                // put.
                self.set(|catalog| &mut catalog.deleted_collections, &id, false)?;
                self.update(
                    |catalog, _| &mut catalog.collections,
                    "",
                    &id,
                    Some(collection),
                )?;
            }
            RecordPayload::CollectionDrop {
                collection_id,
                dropped_at,
                ..
            } => {
                // Look for a known definition before reserving its name. A drop
                // for an unknown ID can still establish deletion state and a drop
                // time, but this branch has no name it can safely reserve.
                if let Some(name) = self.catalog_key(|catalog| {
                    catalog
                        .collections
                        .get(&collection_id)
                        .map(|value| value.name.as_str())
                })? {
                    self.set(
                        |catalog| &mut catalog.reserved_collection_names,
                        &name,
                        true,
                    )?;
                }
                // Retain the drop time beside the ID. Compaction can reconstruct
                // the drop from the stored definition and this timestamp even
                // though normal readers exclude the collection and its contents.
                self.update(
                    |catalog, _| &mut catalog.collection_drop_times,
                    "",
                    &collection_id,
                    Some(dropped_at),
                )?;
                self.set(
                    |catalog| &mut catalog.deleted_collections,
                    &collection_id,
                    true,
                )?;
            }
            RecordPayload::DocumentPut(document) => {
                let id = self.journal_key(&document.document_id)?;
                self.track_id(&id)?;
                // The internal ID is stable across an update, but the externally
                // visible lookup key can change. Remove the old alias while the
                // previous row still supplies its collection and external ID.
                if self.catalog.documents.get(&id).is_some_and(|old| {
                    old.collection_id != document.collection_id
                        || old.external_id != document.external_id
                }) {
                    // Copies of the old scope and external key are needed because
                    // changing an alias mutably borrows the catalog. Keep the old
                    // owner's identity in the undo sequence so restoring this row
                    // can restore the previous mapping.
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(&id)
                                .map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(&id)
                                .map(|old| old.external_id.as_str())
                        })?
                        .unwrap();
                    self.remove_alias(true, &collection, &external, &id)?;
                }
                // Install the put's new locator before making the alias point to
                // its row. These edits are journaled together, so a failure in the
                // later alias or row allocation can restore the earlier locator as
                // part of rollback.
                self.reference(&id, reference)?;
                // Deleting and later replaying a put for the same internal ID
                // requires clearing the deletion marker. The collection's own
                // liveness remains a separate condition, so making the document
                // row current cannot resurrect a dropped collection.
                self.set(|catalog| &mut catalog.deleted_documents, &id, false)?;
                self.alias(
                    true,
                    &document.collection_id,
                    &document.external_id,
                    Some(&id),
                )?;
                // Reserve the boxed summary separately from its strings. The box
                // remains in the catalog and its heap cost must survive transfer
                // of the incoming record's variable-sized allocations.
                self.charge(size_of::<Document>())?;
                self.update(
                    |catalog, _| &mut catalog.documents,
                    "",
                    &id,
                    Some(Box::new(document)),
                )?;
            }
            RecordPayload::ChunkPut(chunk) => {
                let id = self.journal_key(&chunk.chunk_id)?;
                self.track_id(&id)?;
                // A chunk participates in two independent lookups: external ID
                // within its collection, and membership in its parent document.
                // Reconcile both against the old row before replacing it. A move to
                // a different document must disappear from the old parent's
                // enumeration.
                if self.catalog.chunks.get(&id).is_some_and(|old| {
                    old.collection_id != chunk.collection_id
                        || old.chunk_external_id != chunk.chunk_external_id
                }) {
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog
                                .chunks
                                .get(&id)
                                .map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .chunks
                                .get(&id)
                                .map(|old| old.chunk_external_id.as_str())
                        })?
                        .unwrap();
                    self.remove_alias(false, &collection, &external, &id)?;
                }
                // Parent membership is independent of the collection-scoped alias.
                // If the document changes, remove the child from its previous
                // enumeration and append it under the new parent. An in-place
                // update within one document keeps the existing child position.
                if self.catalog.chunks.get(&id).map(|old| &old.document_id)
                    != Some(&chunk.document_id)
                {
                    // A new row has no previous parent. A moved row can use the
                    // old document ID still recorded in its summary, even if the
                    // new put will overwrite that summary later in this arm.
                    if let Some(document) = self.catalog_key(|catalog| {
                        catalog.chunks.get(&id).map(|old| old.document_id.as_str())
                    })? {
                        self.remove_child(&document, &id)?;
                    }
                    self.append_child(&chunk.document_id, &id)?;
                }
                // The updated chunk can replace its physical locator while
                // preserving its logical identity. Visibility, the scoped alias
                // and the summary are published as part of that same catalog
                // transaction.
                self.reference(&id, reference)?;
                self.set(|catalog| &mut catalog.deleted_chunks, &id, false)?;
                self.alias(
                    false,
                    &chunk.collection_id,
                    &chunk.chunk_external_id,
                    Some(&id),
                )?;
                // Charge the new chunk box before replacing the old one. Updating
                // the map can carry the old box in the undo journal until this
                // transaction commits, so the final live size alone understates
                // the transient requirement.
                self.charge(size_of::<Chunk>())?;
                self.update(
                    |catalog, _| &mut catalog.chunks,
                    "",
                    &id,
                    Some(Box::new(chunk)),
                )?;
            }
            RecordPayload::EmbeddingPut(mut embedding) => {
                let id = self.journal_key(&embedding.embedding_id)?;
                self.track_id(&id)?;
                self.reference(&id, reference)?;
                // When the chunk is already known, use the chunk's current
                // external ID. Caller or earlier frame data can carry a spelling
                // that predates a chunk update. An embedding supplied before its
                // chunk has no current spelling to copy here.
                if let Some(external) = self.catalog_key(|catalog| {
                    catalog
                        .chunks
                        .get(&embedding.chunk_id)
                        .map(|chunk| chunk.chunk_external_id.as_str())
                })? {
                    embedding.chunk_external_id = self.copy(&external)?;
                    // Reserve an undo slot and prepare the key and new ID before
                    // changing the reverse link. Then replace returns the old string
                    // for the journal with no fallible allocation in between.
                    // Account for the change in the live string's capacity.
                    self.journal_slot()?;
                    let key = self.undo_key(&embedding.chunk_id)?;
                    let next = self.copy(&id)?;
                    let after = next.capacity();
                    // This change updates the embedding link in the chunk summary
                    // itself. The separate chunk-to-embedding map still contains
                    // the old ID until the later update, leaving a lookup for the
                    // displaced embedding during this replay.
                    let previous = self
                        .catalog
                        .chunks
                        .get_mut(&embedding.chunk_id)
                        .unwrap()
                        .embedding_id
                        .replace(next);
                    let before = optional_bytes(&previous);
                    // The old link string remains owned by the undo record until
                    // the transaction commits. Adjust the live-byte total
                    // separately so aborted writes can restore the displaced
                    // allocation and the earlier count.
                    self.save_undo(Undo::ChunkEmbedding {
                        chunk: key,
                        previous,
                    });
                    self.change_live(before, after);
                }
                // A chunk has at most one current embedding. When its ID changes,
                // retire the old row and disk reference together. Updating the chunk
                // lookup alone would leave the old embedding addressable by ID.
                // Consult the reverse table to find the embedding displaced by the
                // new link. Retire its summary and old record reference when its
                // ID differs. A put of the same ID updates that record in place.
                if let Some(previous) = self.catalog_key(|catalog| {
                    catalog
                        .chunk_embeddings
                        .get(&embedding.chunk_id)
                        .map(String::as_str)
                })? {
                    if previous != id {
                        self.update(|catalog, _| &mut catalog.embeddings, "", &previous, None)?;
                        self.update(|catalog, _| &mut catalog.record_refs, "", &previous, None)?;
                    }
                }
                // Install the new embedding ID in the reverse map even when the
                // chunk row does not yet exist. Visibility still derives from a
                // live chunk and excludes an orphan until the enclosing
                // relationships qualify.
                let owner = self.copy(&id)?;
                self.update(
                    |catalog, _| &mut catalog.chunk_embeddings,
                    "",
                    &embedding.chunk_id,
                    Some(owner),
                )?;
                // The embedding locator has its own journal entry. Rollback
                // restores it alongside the displaced summary, including its
                // model.
                self.charge(size_of::<Embedding>())?;
                self.update(
                    |catalog, _| &mut catalog.embeddings,
                    "",
                    &id,
                    Some(Box::new(embedding)),
                )?;
            }
            RecordPayload::Tombstone(tombstone) => {
                self.tombstone(&tombstone.target_type, &tombstone.target_id)?
            }
            RecordPayload::CompactMarker(marker) => {
                // Apply marker watermarks for retired identities that are absent
                // from the rewritten data. Assigning IDs from visible rows alone
                // after compaction could reuse a deleted object's identifier.
                for (prefix, value) in marker.id_high_watermarks {
                    self.watermark(&prefix, value)?;
                }
            }
            RecordPayload::Commit(_) => {}
        }
        // Publish the checked business total only after the selected record's
        // edits succeed. The count includes delete frames whose targets were not
        // found, so use the record kind and completed replay rather than row
        // occupancy to derive it.
        self.catalog.business_record_count = next_business;
        Ok(())
    }

    fn tombstone(&mut self, target: &str, id: &str) -> Result<()> {
        let count = self.catalog.tombstone_count.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::FileFormatError,
                "catalog tombstone count overflow",
            )
        })?;
        // Advance allocation history even for a delete whose target has no
        // retained row. The target ID still occurred on disk and must be
        // considered when allocating the next ID of its namespace.
        self.track_id(id)?;
        let counters: Select<usize> = match target {
            "collection" => |catalog, _| &mut catalog.collection_tombstones,
            "document" => |catalog, _| &mut catalog.document_tombstones,
            "chunk" => |catalog, _| &mut catalog.chunk_tombstones,
            // An unrecognized target type has no per-kind map or deletion set
            // here. Its record still contributes to the aggregate tombstone total.
            _ => {
                self.catalog.tombstone_count = count;
                return Ok(());
            }
        };
        // Repeated delete records contribute separately to the count for their
        // target. Visibility only needs a set membership bit, while statistics
        // preserve the number of tombstone records for the same object.
        let next = counters(self.catalog, "")
            .get(id)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "catalog target tombstone count overflow",
                )
            })?;
        // Keep the counter edit in the same journal as the deletion state. If
        // fitting a reduced document or chunk fails, rollback restores the counter
        // with the previous rows, locators and flags.
        self.update(counters, "", id, Some(next))?;
        // Keep the deletion history separate from the retained row. A tombstone can
        // precede lookup of an old identity; the deleted sets control visibility,
        // while the reduced rows below preserve identifiers needed to resolve it.
        match target {
            // A collection tombstone hides its descendants through the
            // collection's deletion flag. The separate CollectionDrop record
            // carries the drop time and can reserve the collection's name.
            "collection" => self.set(|catalog| &mut catalog.deleted_collections, id, true)?,
            "document" => {
                if self.catalog.documents.contains_key(id) {
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(id)
                                .map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .documents
                                .get(id)
                                .map(|old| old.external_id.as_str())
                        })?
                        .unwrap();
                    // Replace a known document with a shell that keeps the
                    // identity and scoped external key. The hidden document no
                    // longer needs title, source URI, metadata or its stored
                    // locator. Child visibility follows the parent deletion flag.
                    self.charge(size_of::<Document>())?;
                    let document = Box::new(Document {
                        document_id: self.copy(id)?,
                        collection_id: self.copy(&collection)?,
                        external_id: self.copy(&external)?,
                        title: None,
                        source_uri: None,
                        metadata: crate::JsonValue::object(),
                        created_at: String::new(),
                        updated_at: String::new(),
                    });
                    self.update(|catalog, _| &mut catalog.documents, "", id, Some(document))?;
                    self.update(|catalog, _| &mut catalog.record_refs, "", id, None)?;
                }
                // Set deletion state even when this replay prefix contained no
                // document body. The lookup above only controls whether there is
                // an old summary and locator to reduce.
                self.set(|catalog| &mut catalog.deleted_documents, id, true)?;
            }
            "chunk" => {
                if self.catalog.chunks.contains_key(id) {
                    let document = self
                        .catalog_key(|catalog| {
                            catalog.chunks.get(id).map(|old| old.document_id.as_str())
                        })?
                        .unwrap();
                    let collection = self
                        .catalog_key(|catalog| {
                            catalog.chunks.get(id).map(|old| old.collection_id.as_str())
                        })?
                        .unwrap();
                    let external = self
                        .catalog_key(|catalog| {
                            catalog
                                .chunks
                                .get(id)
                                .map(|old| old.chunk_external_id.as_str())
                        })?
                        .unwrap();
                    // Keep the current link on the retained chunk shell. The
                    // visibility test rejects the deleted chunk and prevents that
                    // link from making its embedding searchable.
                    let embedding = self.catalog_key(|catalog| {
                        catalog
                            .chunks
                            .get(id)
                            .and_then(|old| old.embedding_id.as_deref())
                    })?;
                    // A deleted chunk still carries its parent, position and span
                    // for inspection. Discard the stored payload reference and large
                    // fields while retaining those relationships behind the deletion
                    // flag.
                    let old = &self.catalog.chunks[id];
                    let (ordinal, token_count, source_span) =
                        (old.ordinal, old.token_count, old.source_span.clone());
                    // Both the previous summary and the reduced shell can exist
                    // while undo is possible. Charge the shell before replacing
                    // the old value and release the historical payload only when
                    // the edit is committed.
                    self.charge(size_of::<Chunk>())?;
                    let chunk = Box::new(Chunk {
                        chunk_id: self.copy(id)?,
                        document_id: self.copy(&document)?,
                        collection_id: self.copy(&collection)?,
                        chunk_external_id: self.copy(&external)?,
                        embedding_id: embedding.as_deref().map(|id| self.copy(id)).transpose()?,
                        ordinal,
                        token_count,
                        source_span,
                        text: String::new(),
                        text_hash: String::new(),
                        metadata: crate::JsonValue::object(),
                        created_at: String::new(),
                    });
                    // Replace the summary and clear its obsolete locator as one
                    // journaled operation. The retained shell carries its original
                    // relationships for diagnostics, while primary lookups must
                    // continue to enforce the deletion state.
                    self.update(|catalog, _| &mut catalog.chunks, "", id, Some(chunk))?;
                    self.update(|catalog, _| &mut catalog.record_refs, "", id, None)?;
                }
                self.set(|catalog| &mut catalog.deleted_chunks, id, true)?;
            }
            _ => unreachable!(),
        }
        // The aggregate total is set after the per-target accounting and any
        // retained summary change have succeeded. A failure before this point
        // still goes through transaction rollback, which restores the starting
        // counters.
        self.catalog.tombstone_count = count;
        Ok(())
    }

    pub(crate) fn commit(mut self) {
        // Prepared metadata has a separate period of ownership while a transaction
        // is waiting for primary locators. Publication must finish transferring
        // that ownership before this point, since commit drops the old values
        // along with the undo records.
        assert!(
            self.pending_json.is_none(),
            "prepared metadata has not been transferred"
        );
        self.committed = true;
        // Free replaced payloads and old array buffers before returning their credit.
        self.undo.clear();
        // At this point the remaining retained credit can shrink to the final live
        // accounting. Shrinking while the undo journal still held replaced values
        // would have let another operation reserve the same memory prematurely.
        if let Some(memory) = &mut self.catalog.memory {
            memory.reservation.retain(memory.live);
        }
    }
}

fn record_id(record: &RecordPayload) -> Option<&str> {
    // These are the record kinds for which the prepared path associates a put with
    // a record locator. Administrative and deletion frames are replayed through
    // their own state changes.
    match record {
        RecordPayload::CollectionCreate(value) => Some(&value.collection_id),
        RecordPayload::DocumentPut(value) => Some(&value.document_id),
        RecordPayload::ChunkPut(value) => Some(&value.chunk_id),
        RecordPayload::EmbeddingPut(value) => Some(&value.embedding_id),
        _ => None,
    }
}

impl Drop for CatalogTransaction<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // An early return may have journaled only a prefix of the changes. Unwind
        // that prefix in reverse order and release unused pending credit back to
        // the shared ledger.
        self.rollback_changes(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::{chunk, collection, document};
    use crate::json::{JsonProcessingPolicyV1, WorkLedger};
    use std::sync::Arc;

    fn funded(record: &RecordPayload, context: &JsonOperationContext) -> RecordReservation {
        let mut json = context.reservation();
        json.grow(
            record.metadata_bytes_in(context).unwrap(),
            JsonInputKind::StoredJson,
        )
        .unwrap();
        let mut strings = context.workspace_reservation();
        let bytes = crate::storage::record_workspace_bytes(record);
        strings.grow_to(bytes, bytes, "catalog_build").unwrap();
        RecordReservation::new(json, strings)
    }

    fn apply(
        catalog: &mut Catalog,
        record: RecordPayload,
        context: &JsonOperationContext,
    ) -> Result<()> {
        let mut work = funded(&record, context);
        let mut transaction = catalog.transaction(context);
        transaction.apply(record, None, &mut work)?;
        transaction.commit();
        catalog.clear_model_states();
        Ok(())
    }

    fn replay(
        catalog: &mut Catalog,
        record: RecordPayload,
        context: &JsonOperationContext,
    ) -> Result<()> {
        let mut work = funded(&record, context);
        let mut transaction = catalog.transaction(context);
        transaction.rollback_on_drop = false;
        let result = transaction.apply(record, None, &mut work);
        assert_eq!(transaction.undo.capacity(), 0);
        result?;
        transaction.commit();
        catalog.clear_model_states();
        Ok(())
    }

    fn footprint(catalog: &Catalog, context: &JsonOperationContext) -> usize {
        fn map<V: Entry>(values: &Map<V>, context: &JsonOperationContext) -> usize {
            btree_workspace::<String, V>(values.len())
                + values
                    .iter()
                    .map(|(key, value)| key.capacity() + value.heap(context).unwrap())
                    .sum::<usize>()
        }
        fn set(values: &BTreeSet<String>) -> usize {
            btree_workspace::<String, ()>(values.len())
                + values.iter().map(String::capacity).sum::<usize>()
        }
        map(&catalog.collections, context)
            + map(&catalog.collection_names, context)
            + set(&catalog.reserved_collection_names)
            + map(&catalog.documents, context)
            + map(&catalog.document_external_ids.collections, context)
            + map(&catalog.chunks, context)
            + map(&catalog.chunk_external_ids.collections, context)
            + map(&catalog.document_chunks, context)
            + map(&catalog.embeddings, context)
            + map(&catalog.chunk_embeddings, context)
            + map(&catalog.record_refs, context)
            + map(&catalog.id_high_watermarks, context)
            + map(&catalog.collection_tombstones, context)
            + map(&catalog.document_tombstones, context)
            + map(&catalog.chunk_tombstones, context)
            + set(&catalog.deleted_collections)
            + set(&catalog.deleted_documents)
            + set(&catalog.deleted_chunks)
            + map(&catalog.collection_drop_times, context)
    }

    fn history(seed: usize) -> Vec<RecordPayload> {
        let mut records = vec![
            RecordPayload::CollectionCreate(collection(1)),
            RecordPayload::CollectionCreate(collection(2)),
        ];
        for id in 1..=2 {
            records.push(RecordPayload::DocumentPut(document(id, "col_1")));
        }
        for step in 0..80usize {
            let id = (step * 7 + seed) % 13 + 1;
            let owner = if step % 3 == 0 { "col_2" } else { "col_1" };
            let record = match step % 8 {
                0 | 1 | 2 | 6 => {
                    let mut value = chunk(id as u64, owner);
                    value.document_id = format!("doc_{}", step % 2 + 1);
                    value.chunk_external_id = format!("alias-{}", id % 5);
                    value.metadata = crate::json::object_from_pairs([(
                        "payload",
                        crate::JsonValue::String("x".repeat(seed + 1)),
                    )]);
                    RecordPayload::ChunkPut(value)
                }
                3 | 4 => RecordPayload::EmbeddingPut(Embedding {
                    embedding_id: format!("emb_{}", step + 1),
                    chunk_id: format!("chk_{id}"),
                    collection_id: owner.into(),
                    chunk_external_id: "old-external".into(),
                    model_name: "model".into(),
                    dim: 2,
                    vector: vec![1.0, 2.0],
                    created_at: "time".into(),
                }),
                5 => RecordPayload::Tombstone(crate::model::Tombstone {
                    target_type: "chunk".into(),
                    target_id: format!("chk_{id}"),
                    deleted_at: "time".into(),
                    reason: None,
                }),
                _ => {
                    let mut value = document((id % 2 + 1) as u64, owner);
                    value.external_id = format!("alias-{}", id % 3);
                    RecordPayload::DocumentPut(value)
                }
            };
            records.push(record);
        }
        records.push(RecordPayload::Tombstone(crate::model::Tombstone {
            target_type: "document".into(),
            target_id: "doc_1".into(),
            deleted_at: "time".into(),
            reason: None,
        }));
        records.push(RecordPayload::CollectionDrop {
            collection_id: "col_1".into(),
            name: "collection-1".into(),
            dropped_at: "time".into(),
            reason: None,
        });
        records
    }

    #[test]
    fn retained_catalog_matches_reference_and_exact_live_accounting() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut catalog = Catalog::new_in(&context);
            let mut recovered = Catalog::new_in(&context);
            let mut reference = Catalog::default();
            for record in history(seed) {
                let previous_key = match &record {
                    RecordPayload::DocumentPut(value) => catalog
                        .documents
                        .get_key_value(&value.document_id)
                        .map(|(key, _)| (value.document_id.clone(), key.as_ptr())),
                    _ => None,
                };
                reference.apply_with_ref(record.clone(), None).unwrap();
                replay(&mut recovered, record.clone(), &context).unwrap();
                apply(&mut catalog, record, &context).unwrap();
                if let Some((id, pointer)) = previous_key {
                    assert_eq!(
                        catalog.documents.get_key_value(&id).unwrap().0.as_ptr(),
                        pointer
                    );
                }
                assert_eq!(
                    catalog.state_signature(),
                    reference.state_signature(),
                    "seed={seed}"
                );
                let memory = catalog.memory.as_ref().unwrap();
                assert_eq!(memory.live, footprint(&catalog, &context));
                assert_eq!(memory.reservation.reserved_bytes(), memory.live);
                assert_eq!(recovered.state_signature(), reference.state_signature());
                assert_eq!(recovered.reserved_bytes(), footprint(&recovered, &context));
            }
            // Persistent allocations must not consume the JSON/operation child quota.
            context
                .workspace_reservation()
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
    }

    #[test]
    fn preparation_rollback_keeps_input_credit_until_retry_or_cancel() {
        let mut partial_metadata = 0;
        let mut partial_changes = 0;
        for seed in 0..100 {
            const TOTAL: usize = 2 * 1024 * 1024;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                TOTAL,
                TOTAL,
            )));
            let baseline = || {
                vec![
                    RecordPayload::CollectionCreate(collection(1)),
                    RecordPayload::DocumentPut(document(1, "col_1")),
                    RecordPayload::ChunkPut(chunk(1, "col_1")),
                ]
            };
            let mut catalog = Catalog::new_in(&context);
            for record in baseline() {
                apply(&mut catalog, record, &context).unwrap();
            }
            let original = catalog.state_signature();
            let initial = catalog.reserved_bytes();
            let records = history(seed);
            let mut json = context.reservation();
            let metadata = records
                .iter()
                .map(|record| record.metadata_bytes_in(&context).unwrap())
                .sum();
            json.grow(metadata, JsonInputKind::StoredJson).unwrap();
            let mut cache = Some(
                context
                    .cache_reservation(
                        (TOTAL - initial - metadata - seed * 1024) as u128,
                        usize::MAX,
                    )
                    .unwrap(),
            );
            let mut transaction = catalog.transaction(&context);
            if let Err(error) = transaction.prepare(records.iter(), &mut json) {
                assert!(crate::json::is_memory_budget_error(&error), "{error:?}");
                let input: usize = transaction
                    .pending_json
                    .as_ref()
                    .map_or(0, |pending| pending.as_slice().iter().sum());
                partial_metadata += usize::from(input > 0);
                partial_changes += usize::from(!transaction.undo.is_empty());
                drop(cache.take());
                transaction.rollback_changes(true);
                assert_eq!(transaction.catalog.state_signature(), original);
                assert_eq!(transaction.catalog.reserved_bytes(), initial + input);
                assert_eq!(json.reserved_bytes(), metadata - input);
                transaction
                    .prepare_records(records.iter(), &mut json)
                    .unwrap();
            }
            drop(cache);
            assert_eq!(json.reserved_bytes(), 0);
            assert_eq!(
                transaction.pending_json.as_ref().unwrap().as_slice().len(),
                records.len()
            );
            let expected_context = JsonOperationContext::standalone();
            let mut expected = Catalog::new_in(&expected_context);
            for record in baseline() {
                apply(&mut expected, record, &expected_context).unwrap();
            }
            let expected_records = history(seed);
            let mut expected_json = expected_context.reservation();
            expected_json
                .grow(metadata, JsonInputKind::StoredJson)
                .unwrap();
            let mut expected_transaction = expected.transaction(&expected_context);
            expected_transaction
                .prepare(expected_records.iter(), &mut expected_json)
                .unwrap();
            assert_eq!(
                transaction.catalog.state_signature(),
                expected_transaction.catalog.state_signature()
            );
            assert_eq!(
                transaction.catalog.business_record_count,
                expected_transaction.catalog.business_record_count
            );
            assert_eq!(
                transaction.catalog.tombstone_count,
                expected_transaction.catalog.tombstone_count
            );
            drop(expected_records);
            drop(expected_transaction);
            drop(records);
            drop(transaction);
            drop(json);
            assert_eq!(catalog.state_signature(), original);
            assert_eq!(catalog.reserved_bytes(), initial);
            let remainder = context
                .cache_reservation((TOTAL - initial) as u128, usize::MAX)
                .unwrap();
            assert!(context.cache_reservation(1, usize::MAX).is_none());
            drop(remainder);
        }
        assert!(partial_metadata > 0 && partial_changes > 0);
    }

    #[test]
    fn insufficient_cache_reclaim_retries_once_and_returns_all_transaction_credit() {
        for seed in 0..100 {
            const TOTAL: usize = 1024 * 1024;
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(TOTAL, TOTAL, TOTAL)));
            let mut catalog = Catalog::new_in(&context);
            let original = catalog.state_signature();
            let records = history(seed);
            let mut json = context.reservation();
            let metadata: usize = records
                .iter()
                .map(|record| record.metadata_bytes_in(&context).unwrap())
                .sum();
            json.grow(metadata, JsonInputKind::StoredJson).unwrap();
            let mut optional = Some(context.cache_reservation(16, usize::MAX).unwrap());
            let busy = context
                .cache_reservation((TOTAL - metadata - seed * 32 - 16) as u128, usize::MAX)
                .unwrap();
            let mut calls = 0;
            let mut transaction = catalog.transaction(&context);
            let error = transaction
                .prepare_reclaiming(records.iter(), &mut json, || {
                    calls += 1;
                    drop(optional.take());
                    true
                })
                .unwrap_err();
            assert_eq!(calls, 1);
            assert!(crate::json::is_memory_budget_error(&error), "{error:?}");
            drop(records);
            drop(transaction);
            drop(json);
            drop(busy);
            assert_eq!(catalog.state_signature(), original);
            assert_eq!(catalog.reserved_bytes(), 0);
            assert!(context
                .cache_reservation(TOTAL as u128, usize::MAX)
                .is_some());
        }
    }

    #[test]
    fn failed_or_cancelled_batches_restore_catalog_children_and_budget() {
        let baseline = || {
            vec![
                RecordPayload::CollectionCreate(collection(1)),
                RecordPayload::DocumentPut(document(1, "col_1")),
                RecordPayload::ChunkPut(chunk(1, "col_1")),
            ]
        };
        let context = JsonOperationContext::standalone();
        let mut initial = Catalog::new_in(&context);
        for record in baseline() {
            apply(&mut initial, record, &context).unwrap();
        }
        let initial_bytes = initial.memory.as_ref().unwrap().live;
        let mut refusals = 0;
        let mut prepared = 0;
        for seed in 0..100 {
            let catalog_limit = initial_bytes + seed * 4096;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                catalog_limit,
                2 * JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            )));
            let mut catalog = Catalog::new_in(&context);
            for record in baseline() {
                apply(&mut catalog, record, &context).unwrap();
            }
            let expected = catalog.state_signature();
            let pointer = catalog.document_chunks["doc_1"].as_ptr();
            let capacity = catalog.document_chunks["doc_1"].capacity();
            let records = history(seed);
            let mut json = context.reservation();
            for record in &records {
                json.grow(
                    record.metadata_bytes_in(&context).unwrap(),
                    JsonInputKind::StoredJson,
                )
                .unwrap();
            }
            let mut transaction = catalog.transaction(&context);
            match transaction.prepare_reclaiming(records.iter(), &mut json, || {
                panic!("the independent Catalog limit cannot be relieved by cache eviction")
            }) {
                Ok(()) => prepared += 1,
                Err(error) => {
                    refusals += 1;
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "resource" && value == "catalog"));
                }
            }
            drop(records);
            drop(transaction);
            drop(json);
            assert_eq!(catalog.state_signature(), expected);
            assert_eq!(catalog.document_chunks["doc_1"].as_ptr(), pointer);
            assert_eq!(catalog.document_chunks["doc_1"].capacity(), capacity);
            assert_eq!(
                catalog
                    .memory
                    .as_ref()
                    .unwrap()
                    .reservation
                    .reserved_bytes(),
                initial_bytes
            );
            assert_eq!(footprint(&catalog, &context), initial_bytes);
            let mut remainder = context.catalog_reservation();
            remainder.grow(catalog_limit - initial_bytes).unwrap();
            assert_eq!(
                remainder.grow(1).unwrap_err().code(),
                ErrorCode::ResourceExhausted
            );
        }
        assert!(refusals > 0 && prepared > 0);
    }

    #[test]
    fn failed_recovery_drops_partial_state_and_all_catalog_credit() {
        let mut failures = 0;
        for seed in 0..100 {
            let limit = (seed + 1) * 512;
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                limit,
                2 * JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            )));
            let mut catalog = Catalog::new_in(&context);
            for record in history(seed) {
                if let Err(error) = replay(&mut catalog, record, &context) {
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "resource" && value == "catalog"));
                    failures += 1;
                    break;
                }
            }
            assert_eq!(catalog.reserved_bytes(), footprint(&catalog, &context));
            drop(catalog);
            context.catalog_reservation().grow(limit).unwrap();
            context
                .workspace_reservation()
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
        assert!(failures > 0);
    }
}
