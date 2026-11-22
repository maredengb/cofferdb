use crate::catalog::Catalog;
use crate::error::{Error, ErrorCode, Result};
use crate::hash::Crc32c;
use crate::json::{JsonOperationContext, ReservedVec};
use crate::model::{CollectionInfo, DistanceMetric};

struct CoverageEntry<'a> {
    // Numeric part of the collection's internal ID. Sorting by it makes coverage
    // serialization independent of map iteration order.
    id: u64,
    // Borrowed collection supplies parameters and the full ID. Full-ID comparison
    // distinguishes unusual spellings with the same numeric suffix.
    collection: &'a CollectionInfo,
    // Live document, chunk and embedding counts in that order. Deleted or
    // superseded rows cannot establish coverage for the current view.
    counts: [usize; 3],
}

// The format writes the numeric suffix as an integer. Parsing removes decimal
// padding, so a spelling such as col_10 sorts after col_2 even when the catalog's
// string keys sort them in the opposite order.
fn collection_number(id: &str) -> Option<u64> {
    id.strip_prefix("col_")?.parse().ok()
}

// Both checkpoint creation and recovery feed this routine a primary catalog. The
// fingerprint ties a checkpoint's declared coverage to that catalog at a
// particular committed byte boundary and sequence. Segment payload integrity is
// checked independently.
pub(super) fn catalog_index_coverage_hash(
    catalog: &Catalog,
    critical_end_offset: u64,
    last_commit_sequence_no: u64,
    context: &JsonOperationContext,
) -> Result<u32> {
    #[cfg(test)]
    super::tests::record_coverage_hash();
    // Keep one scratch entry per live collection, including an empty one. Borrow
    // names and search configuration from the catalog so long strings do not also
    // have to fit in a temporary copy of the entire hash input.
    let mut collections = ReservedVec::with_capacity(
        catalog.live_collections().count(),
        context,
        "operation_scratch",
    )?;
    for collection in catalog.live_collections() {
        // A live collection with a malformed ID has no numeric wire
        // representation. Fail before publishing the checksum. All scratch entries
        // admitted so far are owned by this call and released on return.
        let id = collection_number(&collection.collection_id)
            .ok_or_else(|| Error::new(ErrorCode::FileFormatError, "invalid collection ID"))?;
        collections.push(CoverageEntry {
            id,
            collection,
            counts: [0; 3],
        });
    }
    let collections = collections.as_mut_slice();
    // Numeric collection order is part of the checkpoint coverage hash. Keep the order
    // independent of the catalog traversal used to gather live rows.
    collections.sort_unstable_by_key(|entry| entry.id);
    // Populate the counters in one pass over each live record family. A deleted
    // parent hides its children here, and a replaced vector contributes only its
    // currently selected embedding.
    for (id, kind) in catalog
        .live_documents()
        .map(|row| (row.collection_id.as_str(), 0))
        .chain(
            catalog
                .live_chunks()
                .map(|row| (row.collection_id.as_str(), 1)),
        )
        .chain(
            catalog
                .live_embeddings()
                .map(|row| (row.collection_id.as_str(), 2)),
        )
    {
        // A row whose owner cannot be parsed cannot contribute to any of the
        // validated collection entries. An otherwise valid owner number absent
        // from the live table likewise contributes no count.
        let Some(number) = collection_number(id) else {
            continue;
        };
        // Find the start of the equal-number run once for each row. Numeric lookup
        // narrows the search, while comparing the complete original ID below
        // prevents rows from being charged to a differently spelled owner.
        let start = collections.partition_point(|entry| entry.id < number);
        // Preserve exact string ownership even for noncanonical IDs sharing a number.
        for entry in collections[start..]
            .iter_mut()
            .take_while(|entry| entry.id == number)
        {
            if entry.collection.collection_id == id {
                entry.counts[kind] = entry.counts[kind].checked_add(1).ok_or_else(|| {
                    Error::new(ErrorCode::FileFormatError, "coverage count overflow")
                })?;
            }
        }
    }
    // Stream the fixed-width little-endian fields directly into CRC32C. No
    // intermediate byte vector is needed, even when many collection names and
    // model names enter the same checksum.
    let mut checksum = Crc32c::default();
    checksum.update(&1u16.to_le_bytes());
    checksum.update(&critical_end_offset.to_le_bytes());
    checksum.update(&last_commit_sequence_no.to_le_bytes());
    // Include the number of collections even if every per-collection count is
    // zero. Adding an empty collection changes the eligible text and vector
    // segment set.
    hash_count(&mut checksum, collections.len());
    for entry in collections {
        let collection = entry.collection;
        checksum.update(&entry.id.to_le_bytes());
        hash_string(&mut checksum, &collection.name)?;
        // The in-memory dimension uses the host's usize. Reject a value that
        // cannot fit the four-byte field in the coverage format instead of
        // producing the same checksum as its truncated value.
        let dim = u32::try_from(collection.embedding_dim).map_err(|_| {
            Error::new(
                ErrorCode::InvalidEmbeddingDimension,
                "dimension exceeds u32",
            )
        })?;
        checksum.update(&dim.to_le_bytes());
        checksum.update(&[distance_metric_id(collection.distance_metric)]);
        hash_optional_string(&mut checksum, collection.embedding_model.as_deref())?;
        checksum.update(&[u8::from(collection.vectors_normalized)]);
        for count in entry.counts {
            hash_count(&mut checksum, count);
        }
        // These fixed trailing bytes are part of each collection's coverage
        // encoding. Keep their values and order compatible with existing
        // checkpoint writers when extending the layout.
        checksum.update(&1u16.to_le_bytes());
        checksum.update(&2u16.to_le_bytes());
        checksum.update(&[1, 1]);
    }
    Ok(checksum.finish())
}

// Frame the UTF-8 bytes with their byte length so adjacent variable-length fields
// cannot share an encoding. Reject an unrepresentable length before adding any
// part of this field to the coverage CRC.
fn hash_string(checksum: &mut Crc32c, value: &str) -> Result<()> {
    let len = u32::try_from(value.len())
        .map_err(|_| Error::new(ErrorCode::FileFormatError, "coverage string exceeds u32"))?;
    checksum.update(&len.to_le_bytes());
    checksum.update(value.as_bytes());
    Ok(())
}

fn hash_optional_string(checksum: &mut Crc32c, value: Option<&str>) -> Result<()> {
    // An unset model and an explicitly empty name are different collection
    // settings. The presence byte preserves that distinction even when the
    // following string length would be zero.
    match value {
        Some(value) => {
            checksum.update(&[1]);
            hash_string(checksum, value)?;
        }
        None => checksum.update(&[0]),
    }
    Ok(())
}

// Use the checkpoint's fixed-width little-endian count for every catalog
// cardinality. Native usize width or byte order must not make reopening the same
// file produce a different fingerprint.
fn hash_count(checksum: &mut Crc32c, count: usize) {
    checksum.update(&(count as u64).to_le_bytes());
}

// These numeric tags enter persisted coverage. Keep them stable across enum
// changes so the same collection metric continues to produce the same fingerprint.
fn distance_metric_id(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 1,
        DistanceMetric::DotProduct => 2,
        DistanceMetric::Euclidean => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use std::sync::Arc;

    fn fixture(seed: usize) -> Catalog {
        let mut catalog = Catalog::default();
        for id in 1..=seed % 17 + 1 {
            let collection = CollectionInfo {
                collection_id: format!("col_{id:016}"),
                name: "\u{4e2d}".repeat(4096 + seed),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: Some("model".repeat(8192 + seed)),
                tokenizer: Default::default(),
                vectors_normalized: false,
                vector_index: Default::default(),
                created_at: String::new(),
            };
            catalog
                .collections
                .insert(collection.collection_id.clone(), collection);
        }
        catalog
    }

    fn assert_available(context: &JsonOperationContext, bytes: usize) {
        let mut work = context.workspace_reservation();
        work.grow_to(bytes, bytes, "operation_scratch").unwrap();
        assert!(work
            .grow_to(bytes + 1, bytes + 1, "operation_scratch")
            .is_err());
    }

    #[test]
    fn coverage_rows_obey_independent_working_and_parent_boundaries_and_release() {
        for seed in 0..100 {
            let catalog = fixture(seed);
            let required = catalog.live_collections().count()
                * (8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>());
            let expected = catalog_index_coverage_hash(
                &catalog,
                58,
                seed as u64,
                &JsonOperationContext::standalone(),
            )
            .unwrap();
            for parent in [false, true] {
                for limit in [required - 1, required, required + 1] {
                    let total = if parent { limit + 8 } else { 2 * limit + 8 };
                    let working = if parent { 2 * limit + 8 } else { limit };
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        working, total, total,
                    )));
                    let cache = context.cache_reservation(8, 8).unwrap();
                    let result = catalog_index_coverage_hash(&catalog, 58, seed as u64, &context);
                    if limit < required {
                        let error = result.unwrap_err();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        for (key, value) in [
                            ("resource", "memory".to_owned()),
                            ("operation", "operation_scratch".to_owned()),
                            (
                                "limit_bytes",
                                if parent { total } else { working }.to_string(),
                            ),
                            (
                                "requested_bytes",
                                if parent { 8 + required } else { required }.to_string(),
                            ),
                        ] {
                            assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                        }
                    } else {
                        assert_eq!(result.unwrap(), expected);
                    }
                    assert_available(&context, limit);
                    drop(cache);
                }
            }
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                required, required, required,
            )));
            // Initialize the caller's ledger locks outside the array measurement.
            assert_eq!(context.available_workspace_bytes(), required);
            let (actual, allocated) = crate::test_alloc::allocation_bytes(|| {
                catalog_index_coverage_hash(&catalog, 58, seed as u64, &context).unwrap()
            });
            assert_eq!(actual, expected);
            assert_eq!(allocated, required as u128, "seed {seed}");
            assert_available(&context, required);
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        let catalog = Catalog::default();
        assert_eq!(context.available_workspace_bytes(), 0);
        let (_, allocated) = crate::test_alloc::allocation_bytes(|| {
            catalog_index_coverage_hash(&catalog, 58, 0, &context).unwrap()
        });
        assert_eq!(allocated, 0);
        assert_available(&context, 0);
    }

    #[test]
    fn coverage_rows_release_credit_on_id_or_dimension_validation_failure() {
        for invalid_id in [false, true] {
            let mut catalog = fixture(0);
            let (key, mut collection) = catalog.collections.pop_first().unwrap();
            if invalid_id {
                collection.collection_id = "doc_0000000000000001".into();
                catalog
                    .collections
                    .insert(collection.collection_id.clone(), collection);
            } else {
                collection.embedding_dim = u32::MAX as usize + 1;
                catalog.collections.insert(key, collection);
            }
            let required =
                8 + std::mem::size_of::<&CollectionInfo>() + 3 * std::mem::size_of::<usize>();
            let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                required, required, required,
            )));
            let error = catalog_index_coverage_hash(&catalog, 58, 0, &context).unwrap_err();
            assert_eq!(
                error.code(),
                if invalid_id {
                    ErrorCode::FileFormatError
                } else {
                    ErrorCode::InvalidEmbeddingDimension
                }
            );
            assert_available(&context, required);
        }
    }
}
