//! Maintenance through the installed CLI contract, checked against independent file bytes.
#![cfg(unix)]

use cofferdb::{
    parse_json, CollectionConfig, Database, DatabaseConfig, DistanceMetric, DocumentInput,
    JsonValue, VectorIndexConfig,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, Barrier};
use std::thread;

struct Directory(PathBuf);

impl Directory {
    fn new(test: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "coffer-maintenance-{test}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn fixture(path: &Path, label: &str) {
    let db = Database::create(path, DatabaseConfig::default()).unwrap();
    db.create_collection(CollectionConfig {
        name: "notes".into(),
        tokenizer: Default::default(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    })
    .unwrap();
    for external_id in ["stable", label] {
        db.collection("notes")
            .unwrap()
            .insert_document(DocumentInput {
                external_id: external_id.into(),
                title: Some(format!("title of {external_id}")),
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![],
            })
            .unwrap();
    }
    db.close().unwrap();
}

fn command(name: &str, path: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
    command.arg(name).arg(path).arg("--json");
    command
}

fn report(output: &Output) -> JsonValue {
    let bytes = if output.status.success() {
        &output.stdout
    } else {
        &output.stderr
    };
    parse_json(std::str::from_utf8(bytes).unwrap()).unwrap()
}

fn result(output: Output) -> JsonValue {
    assert!(output.status.success(), "{output:?}");
    report(&output).get("result").unwrap().clone()
}

fn repair(path: &Path) -> Output {
    command("repair", path)
        .arg("--truncate-tail")
        .output()
        .unwrap()
}

#[test]
fn safe_pending_batch_is_removed_but_later_committed_bytes_are_never_discarded() {
    let dir = Directory::new("truncate");
    let path = dir.0.join("notes.coffer");
    fixture(&path, "later");
    let complete = fs::read(&path).unwrap();
    // Parse the format-1 record boundaries independently of the recovery scanner.
    let mut frames = Vec::new();
    let mut offset = 58;
    while complete.get(offset..offset + 4) == Some(b"CFRL") {
        let record_type = u16::from_le_bytes(complete[offset + 4..offset + 6].try_into().unwrap());
        let size = u64::from_le_bytes(complete[offset + 34..offset + 42].try_into().unwrap());
        frames.push((offset, record_type));
        offset += 50 + usize::try_from(size).unwrap();
    }
    let final_commit = frames.iter().rposition(|(_, kind)| *kind == 7).unwrap();
    let prior_commit = frames[..final_commit]
        .iter()
        .rposition(|(_, kind)| *kind == 7)
        .unwrap();
    let last_batch = frames[prior_commit + 1].0;
    assert!(last_batch < frames[final_commit].0);

    // The entire final data frame reached disk, its commit did not.
    fs::write(&path, &complete[..frames[final_commit].0]).unwrap();
    let fixed = result(repair(&path));
    assert_eq!(
        fixed.get("repaired").and_then(JsonValue::as_bool),
        Some(true)
    );
    assert_eq!(fs::read(&path).unwrap(), complete[..last_batch]);
    let second = result(repair(&path));
    assert_eq!(
        second.get("repaired").and_then(JsonValue::as_bool),
        Some(false)
    );
    let reader = Database::open_read_only(&path).unwrap();
    let notes = reader.collection("notes").unwrap();
    assert!(notes.get_document_by_external_id("stable").is_ok());
    assert!(notes.get_document_by_external_id("later").is_err());
    reader.close().unwrap();

    // A damaged frame followed by a physically complete commit is ambiguous.
    let mut corrupt = complete.clone();
    corrupt[last_batch + 46] ^= 1;
    fs::write(&path, &corrupt).unwrap();
    let rejected = repair(&path);
    assert!(!rejected.status.success(), "{rejected:?}");
    assert_eq!(
        report(&rejected).get("ok").and_then(JsonValue::as_bool),
        Some(false)
    );
    assert_eq!(fs::read(&path).unwrap(), corrupt);

    // Magic alone does not establish the identity or extent of an index suffix.
    // A later commit is still present when the first eight bytes have changed.
    let mut disguised = complete;
    disguised[last_batch..last_batch + 8].copy_from_slice(b"CFIDXCHK");
    fs::write(&path, &disguised).unwrap();
    let rejected = repair(&path);
    assert!(!rejected.status.success(), "{rejected:?}");
    assert_eq!(fs::read(&path).unwrap(), disguised);
    let destination = dir.0.join("ambiguous-snapshot");
    let copy = command("snapshot", &path)
        .arg(&destination)
        .output()
        .unwrap();
    assert!(!copy.status.success(), "{copy:?}");
    assert!(!destination.exists());
}

#[test]
fn concurrent_snapshot_selects_one_complete_source_and_preserves_existing_aliases() {
    let directory = Directory::new("snapshot");
    let source_a = directory.0.join("a.coffer");
    let source_b = directory.0.join("b.coffer");
    let destination = directory.0.join("selected.coffer");
    fixture(&source_a, "only-a");
    fixture(&source_b, "only-b");
    let bytes_a = fs::read(&source_a).unwrap();
    let bytes_b = fs::read(&source_b).unwrap();
    let start = Arc::new(Barrier::new(2));
    let workers = [&source_a, &source_b].map(|source| {
        let start = Arc::clone(&start);
        let dest = destination.clone();
        let mut cmd = command("snapshot", source);
        thread::spawn(move || {
            start.wait();
            cmd.arg(dest).output().unwrap()
        })
    });
    let [a, b] = workers.map(|worker| worker.join().unwrap());
    assert_ne!(a.status.success(), b.status.success(), "{a:?} {b:?}");
    let (label, winner, loser, expected) = if a.status.success() {
        ("only-a", a, b, &bytes_a)
    } else {
        ("only-b", b, a, &bytes_b)
    };
    assert_eq!(loser.status.code(), Some(1));
    let details = result(winner);
    let package = details.get("package").unwrap();
    let digest = package.get("sha256").and_then(JsonValue::as_str).unwrap();
    assert_eq!(digest.len(), 64);
    assert!(digest
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    assert_eq!(
        package
            .get("size_bytes")
            .and_then(JsonValue::as_str)
            .unwrap(),
        expected.len().to_string()
    );
    assert_eq!(fs::read(&destination).unwrap(), *expected);
    let reader = Database::open_read_only(&destination).unwrap();
    assert_eq!(
        reader
            .collection("notes")
            .unwrap()
            .get_document_by_external_id(label)
            .unwrap()
            .title
            .unwrap(),
        format!("title of {label}")
    );
    reader.close().unwrap();

    let hard_link = directory.0.join("hard");
    let symlink = directory.0.join("symbolic");
    fs::hard_link(&source_a, &hard_link).unwrap();
    std::os::unix::fs::symlink(&source_b, &symlink).unwrap();
    for existing in [&destination, &source_a, &hard_link, &symlink] {
        let output = command("snapshot", &source_a)
            .arg(existing)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{output:?}");
    }
    assert_eq!(fs::read(&source_a).unwrap(), bytes_a);
    assert_eq!(fs::read(&source_b).unwrap(), bytes_b);
    assert_eq!(fs::read(&destination).unwrap(), *expected);
}
