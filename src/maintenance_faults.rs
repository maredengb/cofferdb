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
    WriteCheckpoint,
}
thread_local! {
    static NEXT: Cell<Option<Stage>> = const { Cell::new(None) };
}

pub(crate) fn check(stage: Stage) -> io::Result<()> {
    if NEXT.with(Cell::get) != Some(stage) {
        return Ok(());
    }
    NEXT.with(|next| next.set(None));
    let errno = match stage {
        Stage::CopySnapshot | Stage::WriteCheckpoint => libc::ENOSPC,
        Stage::SyncSnapshotDirectory | Stage::SyncRepair | Stage::MeasureReplacement => libc::EIO,
    };
    Err(io::Error::from_raw_os_error(errno))
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
    assert!(damaged.len() > prefix);
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
    assert!(compact.index_rebuild_reason.is_some());
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
