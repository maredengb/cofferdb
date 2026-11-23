//! I/O faults injected once on the calling test thread.
use crate::{
    CheckOptions, ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric,
    DocumentInput, Error, JsonValue, VectorIndexConfig,
};
use std::{
    cell::Cell,
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    CopySnapshot,
    SyncSnapshotDirectory,
    SyncRepair,
    MeasureReplacement,
    BeforeCheckpoint,
    WriteCheckpoint,
    PartialTrailer,
    CleanCheckpoint,
    SyncCheckpointCleanup,
    SyncCheckpoint,
    VerifyCheckpoint,
    VerifyViewUnavailable,
    VerifySequence,
    VerifyStatus,
}
thread_local! {
    static NEXT: Cell<Option<Stage>> = const { Cell::new(None) };
    static AFTER: Cell<Option<Stage>> = const { Cell::new(None) };
}

pub(crate) fn fail_next(stage: Stage) {
    NEXT.with(|next| assert!(next.replace(Some(stage)).is_none()));
}

pub(crate) fn take(stage: Stage) -> bool {
    NEXT.with(|next| {
        if next.get() != Some(stage) {
            return false;
        }
        next.set(AFTER.with(Cell::take));
        true
    })
}

pub(crate) fn check(stage: Stage) -> io::Result<()> {
    if !take(stage) {
        return Ok(());
    }
    let errno = match stage {
        Stage::CopySnapshot | Stage::WriteCheckpoint | Stage::PartialTrailer => libc::ENOSPC,
        _ => libc::EIO,
    };
    Err(io::Error::from_raw_os_error(errno))
}

pub(crate) fn readback(mut checked: crate::CheckReport) -> crate::CheckReport {
    if take(Stage::VerifySequence) {
        checked
            .index_suffix
            .as_mut()
            .unwrap()
            .covered_last_commit_sequence_no -= 1;
    }
    if take(Stage::VerifyStatus) {
        checked.index_suffix_status = "rebuild_needed".into();
    }
    checked
}

struct Fixture(PathBuf);

impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "coffer-maintenance-io-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let db = Database::create(root.join("source"), DatabaseConfig::default()).unwrap();
        let col = db
            .create_collection(CollectionConfig {
                name: "notes".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        for label in ["one", "two"] {
            col.insert_document(DocumentInput {
                external_id: label.into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: label.into(),
                    text: format!("find {label}"),
                    ordinal: 0,
                    metadata: JsonValue::object(),
                    source_span: None,
                }],
            })
            .unwrap();
        }
        db.close().unwrap();
        Self(root)
    }
    fn source(&self) -> PathBuf {
        self.0.join("source")
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        NEXT.with(|next| next.set(None));
        AFTER.with(|next| next.set(None));
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn detail<'a>(err: &'a Error, key: &str) -> &'a str {
    err.detail()
        .iter()
        .find(|(name, _)| name == key)
        .unwrap_or_else(|| panic!("{key} in {err:?}"))
        .1
        .as_str()
}
fn query_ids(path: &Path) -> Vec<String> {
    let db = Database::open_read_only(path).unwrap();
    let result = db
        .collection("notes")
        .unwrap()
        .search()
        .text("find")
        .execute()
        .unwrap();
    let mut ids: Vec<_> = result
        .results
        .into_iter()
        .map(|hit| hit.chunk_external_id)
        .collect();
    ids.sort();
    ids
}

#[test]
fn snapshot_disk_full_before_publish_and_sync_error_after_publish_are_distinguishable() {
    let fixture = Fixture::new("snapshot");
    let source = fixture.source();
    let target = fixture.0.join("copy");
    let bytes = fs::read(&source).unwrap();
    NEXT.with(|slot| slot.set(Some(Stage::CopySnapshot)));
    let before = Database::snapshot(&source, &target).unwrap_err();
    assert_eq!(detail(&before, "effect"), "unchanged");
    assert!(!target.exists());
    assert!(!fs::read_dir(&fixture.0).unwrap().any(|p| p
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
    NEXT.with(|slot| slot.set(Some(Stage::SyncSnapshotDirectory)));
    let after = Database::snapshot(&source, &target).unwrap_err();
    assert_eq!(detail(&after, "effect"), "snapshot_published");
    assert_eq!(detail(&after, "durability"), "unknown");
    assert_eq!(
        detail(&after, "sha256"),
        crate::hash::sha256_reader(&bytes[..]).unwrap()
    );
    assert_eq!(detail(&after, "size_bytes"), bytes.len().to_string());
    let copied = Database::check_path(
        &target,
        CheckOptions {
            verify_indexes: true,
        },
    )
    .unwrap();
    assert_eq!(
        detail(&after, "last_valid_commit"),
        copied.last_valid_commit.to_string()
    );
    assert_eq!(fs::read(&target).unwrap(), bytes);
    assert_eq!(fs::read(&source).unwrap(), bytes);
    assert_eq!(query_ids(&target), ["one", "two"]);
}

#[test]
fn full_disk_during_index_write_preserves_commits_and_permits_recovery() {
    let fixture = Fixture::new("checkpoint");
    let source = fixture.source();
    let before = Database::check_path(
        &source,
        CheckOptions {
            verify_indexes: true,
        },
    )
    .unwrap();
    let bytes = fs::read(&source).unwrap();
    let db = Database::open(&source).unwrap();
    NEXT.with(|slot| slot.set(Some(Stage::WriteCheckpoint)));
    let error = db.rebuild_index(None).unwrap_err();
    assert_eq!(detail(&error, "effect"), "index_may_need_rebuild");
    db.close().unwrap();
    let damaged = fs::read(&source).unwrap();
    let prefix = before.critical_end_offset as usize;
    assert_eq!(damaged[..prefix], bytes[..prefix]);
    assert_eq!(damaged.len(), prefix);
    assert_ne!(damaged, bytes);
    assert_eq!(query_ids(&source), ["one", "two"]);
    let restarted = Database::open(&source).unwrap();
    let summary = restarted.rebuild_index(None).unwrap();
    assert!(!summary.rebuild_needed);
    restarted.close().unwrap();
    let checked = Database::check_path(
        &source,
        CheckOptions {
            verify_indexes: true,
        },
    )
    .unwrap();
    assert_eq!(checked.last_valid_commit, before.last_valid_commit);
    assert_eq!(checked.index_check, "ok");
    assert_eq!(query_ids(&source), ["one", "two"]);
}

#[test]
fn checkpoint_failures_distinguish_unchanged_partial_and_complete_suffixes() {
    for (stage, second, expected_disk) in [
        (Stage::BeforeCheckpoint, None, "persistent_checkpoint"),
        (Stage::WriteCheckpoint, None, "missing"),
        (Stage::PartialTrailer, None, "missing"),
        (
            Stage::PartialTrailer,
            Some(Stage::CleanCheckpoint),
            "rebuild_needed",
        ),
        (
            Stage::WriteCheckpoint,
            Some(Stage::SyncCheckpointCleanup),
            "missing",
        ),
        (Stage::SyncCheckpoint, None, "persistent_checkpoint"),
        (Stage::VerifyCheckpoint, None, "persistent_checkpoint"),
        (Stage::VerifyViewUnavailable, None, "persistent_checkpoint"),
        (Stage::VerifySequence, None, "persistent_checkpoint"),
        (Stage::VerifyStatus, None, "persistent_checkpoint"),
    ] {
        let fixture = Fixture::new("checkpoint-stages");
        let path = fixture.source();
        let checked = |path: &Path| {
            Database::check_path(
                path,
                CheckOptions {
                    verify_indexes: true,
                },
            )
            .unwrap()
        };
        let writer = Database::open(&path).unwrap();
        writer
            .collection("notes")
            .unwrap()
            .put_embeddings(vec![crate::EmbeddingInput {
                chunk_external_id: "one".into(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            }])
            .unwrap();
        let before = checked(&path);
        let bytes = fs::read(&path).unwrap();
        let old = writer.index_stats().unwrap();
        assert_eq!(old.vector_count, 1);
        NEXT.with(|slot| slot.set(Some(stage)));
        AFTER.with(|slot| slot.set(second));
        let error = writer.rebuild_index(None).unwrap_err();
        assert!(NEXT.with(Cell::get).is_none());
        let current = writer.index_stats().unwrap();
        if stage == Stage::BeforeCheckpoint {
            assert_eq!(fs::read(&path).unwrap(), bytes);
            assert!(!current.rebuild_needed);
            assert_eq!(current.last_index_commit_id, old.last_index_commit_id);
            assert_eq!(current.index_size_bytes, old.index_size_bytes);
        } else {
            assert_eq!(detail(&error, "effect"), "index_may_need_rebuild");
            assert!(current.rebuild_needed);
            assert_eq!(
                (
                    current.index_size_bytes,
                    current.text_segments,
                    current.vector_segments
                ),
                (0, 0, 0)
            );
            assert_eq!(current.last_index_commit_id, None);
            assert_eq!(current.vector_count, old.vector_count);
            let phase = match stage {
                Stage::WriteCheckpoint => "write_checkpoint",
                Stage::PartialTrailer => "write_checkpoint_trailer",
                Stage::SyncCheckpoint => "sync_checkpoint",
                _ => "verify_checkpoint",
            };
            assert_eq!(detail(&error, "stage"), phase);
        }
        if let Some(secondary) = second {
            assert_eq!(detail(&error, "durability"), "unknown");
            assert!(!detail(&error, "cleanup_error").is_empty());
            assert!(error
                .message()
                .contains(&io::Error::from_raw_os_error(libc::ENOSPC).to_string()));
            let cleanup = if secondary == Stage::CleanCheckpoint {
                "truncate_checkpoint"
            } else {
                assert_eq!(detail(&error, "cleanup_effect"), "suffix_truncated");
                "sync_checkpoint_cleanup"
            };
            assert_eq!(detail(&error, "cleanup_stage"), cleanup);
        }
        if stage == Stage::SyncCheckpoint {
            assert_eq!(detail(&error, "durability"), "unknown");
        }
        if stage == Stage::VerifyViewUnavailable {
            assert_eq!(detail(&error, "reason"), "check_view_unavailable");
            assert_eq!(detail(&error, "uncompleted_scope"), "main_and_index");
            assert_eq!(detail(&error, "verification_effect"), "unchanged");
            assert_eq!(detail(&error, "verification_stage"), "check_view");
        }
        if stage == Stage::VerifySequence || stage == Stage::VerifyStatus {
            assert_eq!(error.code(), crate::ErrorCode::IndexCorruption);
        }
        writer.close().unwrap();
        let after = checked(&path);
        let on_disk = fs::read(&path).unwrap();
        let prefix = before.critical_end_offset as usize;
        assert_eq!(&on_disk[..prefix], &bytes[..prefix]);
        assert_eq!(after.last_valid_commit, before.last_valid_commit);
        assert_eq!(after.index_suffix_status, expected_disk);
        match expected_disk {
            "missing" => assert_eq!(on_disk.len(), prefix),
            "rebuild_needed" => assert!(on_disk.ends_with(b"CFOK")),
            _ => assert_eq!(after.index_check, "ok"),
        }
        assert_eq!(query_ids(&path), ["one", "two"]);
    }
}

#[test]
fn automatic_checkpoint_failure_retains_the_acknowledged_business_commit() {
    for (stage, expected_disk) in [
        (Stage::WriteCheckpoint, "missing"),
        (Stage::SyncCheckpoint, "persistent_checkpoint"),
    ] {
        let fixture = Fixture::new("ordinary-checkpoint");
        let path = fixture.source();
        let writer = Database::open(&path).unwrap();
        let check = || {
            writer
                .check(CheckOptions {
                    verify_indexes: true,
                })
                .unwrap()
        };
        let before = check();
        let old_bytes = fs::read(&path).unwrap();
        let notes = writer.collection("notes").unwrap();
        let id = notes
            .get_document_by_external_id("two")
            .unwrap()
            .document_id;
        NEXT.with(|slot| slot.set(Some(stage)));
        notes.delete_document(&id).unwrap();
        assert!(NEXT.with(Cell::get).is_none());
        let current = writer.index_stats().unwrap();
        assert!(current.rebuild_needed);
        assert_eq!(current.last_index_commit_id, None);
        let after = check();
        assert!(after.last_valid_commit > before.last_valid_commit);
        assert_eq!(after.live_documents, 1);
        assert_eq!(after.index_suffix_status, expected_disk);
        let bytes = fs::read(&path).unwrap();
        let old_end = before.critical_end_offset as usize;
        assert_eq!(&bytes[..old_end], &old_bytes[..old_end]);
        if stage == Stage::WriteCheckpoint {
            assert_eq!(bytes.len() as u64, after.critical_end_offset);
        }
        writer.close().unwrap();
        let restarted = Database::open_read_only(&path).unwrap();
        assert_eq!(
            restarted.stats().unwrap().last_commit_id,
            after.last_valid_commit
        );
        assert_eq!(
            restarted.index_stats().unwrap().rebuild_needed,
            stage == Stage::WriteCheckpoint
        );
        assert_eq!(query_ids(&path), ["one"]);
    }
}

#[test]
fn repair_sync_failure_retains_valid_prefix_and_reports_completed_truncation() {
    let fixture = Fixture::new("repair");
    let source = fixture.source();
    let healthy = Database::check_path(
        &source,
        CheckOptions {
            verify_indexes: true,
        },
    )
    .unwrap();
    let bytes = fs::read(&source).unwrap();
    let mut offset = 58;
    let mut commits = Vec::new();
    while offset < healthy.critical_end_offset as usize {
        let kind = u16::from_le_bytes(bytes[offset + 4..offset + 6].try_into().unwrap());
        let size = u64::from_le_bytes(bytes[offset + 34..offset + 42].try_into().unwrap()) as usize;
        let end = offset + 50 + size;
        if kind == 7 {
            commits.push((offset, end));
        }
        offset = end;
    }
    let (last_start, _) = commits.pop().unwrap();
    let (_, boundary) = commits.pop().unwrap();
    fs::write(&source, &bytes[..last_start]).unwrap();
    NEXT.with(|slot| slot.set(Some(Stage::SyncRepair)));
    let err = Database::repair_truncate_tail(&source).unwrap_err();
    assert_eq!(detail(&err, "effect"), "tail_truncated");
    assert_eq!(detail(&err, "durability"), "unknown");
    assert_eq!(fs::read(&source).unwrap(), bytes[..boundary]);
    assert_eq!(query_ids(&source), ["one"]);
    assert!(!Database::repair_truncate_tail(&source).unwrap().repaired);
}

#[test]
fn compact_reports_post_publication_index_failure_and_preserves_both_reader_views() {
    let fixture = Fixture::new("compact-checkpoint");
    let source = fixture.source();
    let old = Database::open_read_only(&source).unwrap();
    let writer = Database::open(&source).unwrap();
    let collection = writer.collection("notes").unwrap();
    let id = collection
        .get_document_by_external_id("two")
        .unwrap()
        .document_id;
    collection.delete_document(&id).unwrap();
    NEXT.with(|slot| slot.set(Some(Stage::WriteCheckpoint)));
    let compact = writer
        .compact(crate::CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
    assert_eq!(compact.index_rebuild, "failed");
    assert!(compact
        .index_rebuild_reason
        .unwrap()
        .contains("index_may_need_rebuild"));
    assert!(writer.index_stats().unwrap().rebuild_needed);
    let checked = writer
        .check(CheckOptions {
            verify_indexes: true,
        })
        .unwrap();
    assert_eq!(checked.index_suffix_status, "missing");
    assert_eq!(checked.critical_end_offset, compact.after_file_size);
    assert_eq!(checked.live_documents, 1);
    writer.close().unwrap();
    let old_rows = old
        .collection("notes")
        .unwrap()
        .search()
        .text("find")
        .execute()
        .unwrap();
    let mut old_ids: Vec<_> = old_rows
        .results
        .into_iter()
        .map(|r| r.chunk_external_id)
        .collect();
    old_ids.sort();
    assert_eq!(old_ids, ["one", "two"]);
    assert_eq!(query_ids(&source), ["one"]);
    let reopened = Database::open(&source).unwrap();
    assert!(!reopened.rebuild_index(None).unwrap().rebuild_needed);
}

#[test]
fn compact_measurement_failure_retains_known_index_outcome() {
    for (rebuild_indexes, expected) in [(false, "disabled"), (true, "full_checkpoint_written")] {
        let fixture = Fixture::new("compact-measure");
        let writer = Database::open(fixture.source()).unwrap();
        NEXT.with(|slot| slot.set(Some(Stage::MeasureReplacement)));
        let err = writer
            .compact(crate::CompactOptions { rebuild_indexes })
            .unwrap_err();
        assert_eq!(detail(&err, "effect"), "main_replaced");
        assert_eq!(detail(&err, "stage"), "measure_replacement");
        assert_eq!(detail(&err, "index_rebuild"), expected);
        writer.close().unwrap();
        assert_eq!(query_ids(&fixture.source()), ["one", "two"]);
    }
}

#[test]
fn snapshot_uses_a_short_staging_name_for_a_long_target_name() {
    let fixture = Fixture::new("long-output");
    let output = fixture.0.join("x".repeat(250));
    let report = Database::snapshot(fixture.source(), &output).unwrap();
    assert_eq!(report.output_path, output);
    assert_eq!(
        fs::read(output).unwrap(),
        fs::read(fixture.source()).unwrap()
    );
}

#[test]
fn statistics_and_maintenance_receipts_round_trip_wide_sizes_and_coverage() {
    let fixture = Fixture::new("wide-receipts");
    let path = fixture.source();
    let db = Database::open(&path).unwrap();
    let mut index = db.index_stats().unwrap();
    let mut stats = db.stats().unwrap();
    let mut collection = db.collection("notes").unwrap().stats().unwrap();
    let mut rebuild = db.rebuild_index(None).unwrap();
    let mut compact = db.compact(crate::CompactOptions::default()).unwrap();
    db.close().unwrap();
    let mut snapshot = Database::snapshot(&path, fixture.0.join("backup")).unwrap();
    let mut repair = Database::repair_truncate_tail(&path).unwrap();

    for n in [(1 << 53) - 1, (1 << 53) + 1, u64::MAX] {
        // Exercise the actual DTO projections and a serialize/parse round trip.
        let verify = |report: JsonValue, keys: &[&str]| {
            let decoded = crate::parse_json(&report.to_compact_string()).unwrap();
            for key in keys {
                let value = decoded.get(key).unwrap();
                if n < 1 << 53 {
                    assert_eq!(value.as_f64(), Some(n as f64), "{key}");
                } else {
                    assert_eq!(value.as_str().unwrap().parse::<u64>().unwrap(), n, "{key}");
                }
            }
        };
        index.index_size_bytes = n;
        index.last_index_commit_id = Some(n);
        verify(
            index.to_json(),
            &["index_size_bytes", "last_index_commit_id"],
        );
        stats.file_size = n;
        stats.index_size_bytes = n;
        stats.last_commit_id = n;
        verify(
            stats.to_json(),
            &[
                "file_size",
                "file_size_bytes",
                "index_size_bytes",
                "last_commit_id",
            ],
        );
        collection.index_size_bytes = n;
        collection.last_commit_id = n;
        verify(
            collection.to_json(),
            &["index_size_bytes", "last_commit_id"],
        );
        rebuild.last_index_commit_id = Some(n);
        verify(rebuild.to_json(), &["last_index_commit_id"]);
        compact.before_file_size = n;
        compact.after_file_size = n;
        let sizes = [
            "before_file_size",
            "after_file_size",
            "before_file_size_bytes",
            "after_file_size_bytes",
        ];
        verify(compact.to_json(), &sizes);
        repair.before_file_size = n;
        repair.after_file_size = n;
        repair.bytes_removed = n;
        repair.critical_end_offset = n;
        verify(repair.to_json(), &sizes);
        verify(repair.to_json(), &["critical_end_offset", "bytes_removed"]);
        if let Ok(counter) = usize::try_from(n) {
            compact.records_removed = counter;
            rebuild.index_extent_count = counter;
            verify(compact.to_json(), &["records_removed"]);
            verify(rebuild.to_json(), &["index_extent_count"]);
        }
        compact.duration_ms = u128::MAX;
        let duration = compact.to_json();
        assert_eq!(
            duration.get("duration_ms").unwrap().as_str(),
            Some(u128::MAX.to_string().as_str())
        );
        snapshot.bytes_copied = n;
        snapshot.critical_end_offset = n;
        snapshot.last_valid_commit = n;
        verify(
            snapshot.to_json(),
            &["bytes_copied", "last_valid_commit", "critical_end_offset"],
        );
    }
}
