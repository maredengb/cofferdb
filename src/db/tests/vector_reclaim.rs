use super::*;

fn signature(hits: &VectorHits) -> Vec<(String, u64, u64)> {
    hits.iter()
        .map(|hit| {
            (
                hit.chunk_id.clone(),
                hit.raw_score.to_bits(),
                hit.normalized_score.to_bits(),
            )
        })
        .collect()
}

#[test]
fn cached_vector_candidates_reclaim_parent_pressure_without_losing_prior_hits() {
    let path = cached_vector_batch_fixture("vector-parent-reclaim");
    let db = Database::open_read_only(&path).unwrap();
    let before = fs::read(&path).unwrap();
    let context = db.json_context();
    let mut inner = db.lock().unwrap();
    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
    assert!(cached.reserved_bytes() > 32 * 1024);
    inner.persistent_vector_segments = Some(cached);
    let (collection, collection_work) = copy_collection_in(
        inner.catalog.collection_by_name("cached").unwrap(),
        &context,
    )
    .unwrap();
    let mut query = vec![0.0; collection.embedding_dim];
    query[0] = 1.0;
    let (prior, prior_count) = inner
        .search_checkpoint_vector_index(&collection, &query, 1, None, &context)
        .unwrap()
        .unwrap();
    assert_eq!(prior_count, 8);
    let prior_signature = signature(&prior);
    let prior_pointer = prior[0].chunk_id.as_ptr();
    let (baseline, count) = inner
        .search_checkpoint_vector_index(&collection, &query, 3, None, &context)
        .unwrap()
        .unwrap();
    assert_eq!(count, 8);
    let expected = signature(&baseline);
    drop(baseline);
    let cached_bytes = inner.optional_cache_bytes();
    let catalog_signature = inner.catalog.state_signature();
    let allowance = 3 * (std::mem::size_of::<VectorHit>() + prior[0].chunk_id.len()) - 1;
    let blocker = context
        .cache_reservation(
            (context.available_total_bytes() - allowance) as u128,
            usize::MAX,
        )
        .unwrap();
    let occupied = blocker.reserved_bytes();
    assert_eq!(context.available_workspace_bytes(), allowance);
    let first = search_persistent_vector_segment(
        inner
            .persistent_vector_segments
            .as_ref()
            .unwrap()
            .get(&collection.collection_id)
            .unwrap(),
        &query,
        collection.distance_metric,
        3,
        |_| true,
        &context,
    )
    .unwrap_err();
    assert!(crate::json::is_memory_budget_error(&first));
    assert_eq!(inner.optional_cache_bytes(), cached_bytes);
    assert_eq!(context.available_workspace_bytes(), allowance);
    let mut actual_work = VectorWork::default();
    let hits = inner
        .search_checkpoint_vector_index_with_work(
            &collection,
            &query,
            3,
            None,
            &context,
            &mut actual_work,
        )
        .unwrap()
        .unwrap();
    assert_eq!(signature(&hits), expected);
    // The third vector was scored before failed candidate admission; all eight
    // are then revisited after cache reclaim. Actual work cannot be rolled back.
    assert_eq!(actual_work.scans, 11);
    assert_eq!(actual_work.distance_ops, 11 * query.len());
    assert_eq!(signature(&prior), prior_signature);
    assert_eq!(prior[0].chunk_id.as_ptr(), prior_pointer);
    assert_eq!(blocker.reserved_bytes(), occupied);
    assert_eq!(inner.optional_cache_bytes(), 0);
    assert_eq!(inner.catalog.state_signature(), catalog_signature);
    assert_eq!(
        inner.index_suffix_status,
        IndexSuffixState::PersistentCheckpoint
    );
    drop(hits);
    assert_eq!(
        context.available_workspace_bytes(),
        allowance + cached_bytes
    );
    drop((blocker, prior, collection, collection_work, inner));
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn cached_vector_native_failure_does_not_reclaim_or_retry() {
    let path = cached_vector_batch_fixture("vector-native-no-reclaim");
    let db = Database::open_read_only(&path).unwrap();
    let before = fs::read(&path).unwrap();
    let context = db.json_context();
    let mut inner = db.lock().unwrap();
    let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
    inner.persistent_vector_segments = Some(cached);
    let (collection, collection_work) = copy_collection_in(
        inner.catalog.collection_by_name("cached").unwrap(),
        &context,
    )
    .unwrap();
    let id_bytes = inner
        .persistent_vector_segments
        .as_ref()
        .unwrap()
        .get(&collection.collection_id)
        .unwrap()
        .vectors[0]
        .chunk_id
        .len();
    let cached_bytes = inner.optional_cache_bytes();
    let available = context.available_workspace_bytes();
    let mut query = vec![0.0; collection.embedding_dim];
    query[0] = 1.0;
    let error = crate::test_alloc::fail_byte_allocation_once(id_bytes, || {
        inner.search_checkpoint_vector_index(&collection, &query, 3, None, &context)
    })
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(inner.optional_cache_bytes(), cached_bytes);
    assert_eq!(context.available_workspace_bytes(), available);
    assert_eq!(
        inner.index_suffix_status,
        IndexSuffixState::PersistentCheckpoint
    );
    assert_eq!(
        inner
            .search_checkpoint_vector_index(&collection, &query, 3, None, &context)
            .unwrap()
            .unwrap()
            .0
            .len(),
        3
    );
    drop((collection, collection_work, inner));
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
fn cached_hybrid_finishes_absent_text_terms_before_retaining_vector_candidates() {
    let path = cached_vector_batch_fixture("cached-hybrid-term-peak");
    let db = Database::open_read_only(&path).unwrap();
    let before = fs::read(&path).unwrap();
    let context = db.json_context();
    {
        let mut inner = db.lock().unwrap();
        let TextCacheBuild::Complete(index) = build_text_cache_in(
            &inner.catalog,
            |_| true,
            &context,
            64 * 1024 * 1024,
            |id, input, visit| {
                inner.lazy_vector_source.as_ref().unwrap().visit_text(
                    &inner.storage,
                    id,
                    input,
                    visit,
                )
            },
        )
        .unwrap() else {
            panic!("resident text fixture required")
        };
        inner.text_index = index;
        let cached = inner.prefetch_unchanged_vector_segments(&BTreeSet::new());
        inner.persistent_vector_segments = Some(cached);
    }
    let words = 1000usize;
    let text = (0..words)
        .map(|index| format!("nevermatch{index} "))
        .collect::<String>();
    let run = || {
        db.collection("cached")
            .unwrap()
            .search()
            .text(&text)
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 1024,
                vector: std::iter::once(1.0)
                    .chain(std::iter::repeat(0.0).take(1023))
                    .collect(),
            })
            .top_k(3)
            .execute()
    };
    let baseline = run().unwrap();
    assert!(baseline
        .results
        .iter()
        .all(|row| row.bm25_score == Some(0.0)));
    let signature = |results: &SearchResults| {
        results
            .results
            .iter()
            .map(|row| (row.chunk_id.clone(), row.final_score.to_bits()))
            .collect::<Vec<_>>()
    };
    let expected = signature(&baseline);
    drop(baseline);
    let nodes = words.div_ceil(5)
        * (11 * std::mem::size_of::<(std::borrow::Cow<'_, str>, usize)>()
            + 16 * std::mem::size_of::<usize>());
    let held = Arc::new(Mutex::new(None));
    let hook_held = held.clone();
    let hook_context = context.clone();
    db.lock().unwrap().query_output_hook = Some((
        QueryOutputStage::CheckpointCandidates,
        Box::new(move || {
            let available = hook_context.available_workspace_bytes();
            let mut work = hook_context.workspace_reservation();
            work.grow_to(available - nodes, available - nodes, "operation_scratch")
                .unwrap();
            *hook_held.lock().unwrap() = Some(work);
        }),
    ));
    let result = run().unwrap();
    assert_eq!(signature(&result), expected);
    assert_eq!(result.query_context.index_source, "persistent_checkpoint");
    assert_eq!(result.query_context.execution_stats.vector_scan_count, 8);
    assert_eq!(result.query_context.execution_stats.distance_ops, 8192);
    assert!(held.lock().unwrap().is_some());
    drop(result);
    held.lock().unwrap().take();
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(db);
    fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires COFFERDB_LARGE_CHECKPOINT_FIXTURE containing the bulk axis-zero reference vector"]
fn large_checkpoint_query_reports_required_workspace_and_recovers_on_reopen() {
    use crate::json::{JsonOperationContext, WorkLedger};

    let path = PathBuf::from(std::env::var_os("COFFERDB_LARGE_CHECKPOINT_FIXTURE").unwrap());
    let digest = || crate::hash::sha256_reader(std::fs::File::open(&path).unwrap()).unwrap();
    let before = digest();
    let verified = Database::check_path(
        &path,
        CheckOptions {
            verify_indexes: true,
        },
    )
    .unwrap();
    assert_eq!(verified.index_check, "ok");
    let suffix = verified.index_suffix.unwrap();
    assert!(suffix.payload_len > 512 * 1024 * 1024);

    let work_limit = 32 * 1024 * 1024;
    let cache_mib = 256;
    let open = || {
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
            work_limit, work_limit, work_limit,
        )));
        Database::open_in(
            &path,
            DatabaseOpenOptions {
                read_only: true,
                cache_budget_mb: Some(cache_mib),
                ..Default::default()
            },
            context,
        )
        .unwrap()
    };
    let db = open();
    let info = db
        .list_collections()
        .unwrap()
        .into_iter()
        .find(|c| c.name == "bulk")
        .unwrap();
    let search = |collection: &CollectionHandle| {
        let mut axis = vec![0.0; info.embedding_dim];
        axis[0] = 1.0;
        collection
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: info.embedding_model.clone(),
                dim: info.embedding_dim,
                vector: axis,
            })
            .top_k(1)
            .execute()
    };
    let assert_hit = |collection: &CollectionHandle| {
        let result = search(collection).unwrap();
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].chunk_external_id, "bulk#000");
        assert_eq!(result.results[0].rank, 1);
        assert_eq!(result.query_context.index_source, "persistent_checkpoint");
        assert!(!result.query_context.transient_rebuild.started);
    };
    let bulk = db.collection("bulk").unwrap();
    assert_hit(&bulk);

    // Admit the full query embedding and leave just half a row for scanning
    // the file. The complete encoded segment exceeds the optional cache, so
    // this exact search needs a second vector's worth of scratch.
    let context = db.json_context();
    let row_bytes = info.embedding_dim * std::mem::size_of::<f32>();
    let remaining = row_bytes + row_bytes / 2;
    let available = context.available_workspace_bytes();
    assert!(available > remaining);
    let mut pressure = context.workspace_reservation();
    let hold = available - remaining;
    pressure.grow_to(hold, hold, "operation_scratch").unwrap();
    let error = search(&bulk).unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted, "{error:?}");
    assert!(error
        .detail()
        .contains(&("resource".into(), "memory".into())));
    let stats = db.index_stats().unwrap();
    assert!(!stats.rebuild_needed);
    assert_eq!(
        stats.last_index_commit_id,
        Some(suffix.covered_last_commit_sequence_no)
    );
    eprintln!("P={} cache_mib={cache_mib} work_limit={work_limit} row_bytes={row_bytes} available_before={available} pressure={hold} remaining={remaining} error={:?}",
        suffix.payload_len, error.detail());
    drop((pressure, bulk, db));

    let reopened = open();
    assert_hit(&reopened.collection("bulk").unwrap());
    drop(reopened);
    assert_eq!(digest(), before);
    eprintln!("initial and fresh reads: rank=1 bulk#000 persistent_checkpoint transient_rebuild.started=false unchanged_sha256={before}");
}
