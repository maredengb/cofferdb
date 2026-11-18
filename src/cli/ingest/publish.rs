//! Replace formal artifact names only after the temporary bytes are durable.
//!
//! A failure puts the previous formal bytes back on their original names and
//! removes this run's temporary files. New bytes are never appended to an
//! existing formal file.

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use crate::error::Result;

use super::boundary::io_ingest;

const RESERVED: &[&str] = &[
    "documents.jsonl",
    "chunks.jsonl",
    "embeddings.jsonl",
    "import_manifest.json",
    "embedding_artifact.json",
];

pub(super) struct PlannedFile {
    pub temp: PathBuf,
    pub formal: PathBuf,
}

struct Slot {
    temp: PathBuf,
    formal: PathBuf,
    backup: Option<PathBuf>,
    moved_aside: bool,
    published: bool,
}

pub(super) fn publish(files: Vec<PlannedFile>) -> Result<()> {
    let mut slots = Vec::with_capacity(files.len());
    for file in files {
        sync_file(&file.temp)?;
        slots.push(Slot {
            temp: file.temp,
            formal: file.formal,
            backup: None,
            moved_aside: false,
            published: false,
        });
    }
    let published = (|| {
        for slot in &mut slots {
            if slot.formal.exists() {
                let backup = backup_path(slot.formal.parent().unwrap_or_else(|| Path::new(".")))?;
                fs::rename(&slot.formal, &backup)
                    .map_err(|_| io_ingest("artifact publish failed", None))?;
                slot.backup = Some(backup);
                slot.moved_aside = true;
            }
        }
        for slot in &mut slots {
            fs::rename(&slot.temp, &slot.formal)
                .map_err(|_| io_ingest("artifact publish failed", None))?;
            slot.published = true;
        }
        Ok(())
    })();
    if let Err(error) = published {
        restore(&slots);
        return Err(error);
    }
    for slot in &slots {
        if let Some(backup) = &slot.backup {
            let _ = fs::remove_file(backup);
        }
    }
    Ok(())
}

fn restore(slots: &[Slot]) {
    for slot in slots {
        if slot.published {
            let _ = fs::remove_file(&slot.formal);
        }
        if let Some(backup) = &slot.backup {
            if slot.moved_aside {
                let _ = fs::rename(backup, &slot.formal);
            }
        }
        let _ = fs::remove_file(&slot.temp);
    }
}

fn sync_file(path: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|_| io_ingest("artifact publish failed", None))?;
    file.sync_all()
        .map_err(|_| io_ingest("artifact publish failed", None))?;
    Ok(())
}

pub(super) fn temporary_path(directory: &Path) -> Result<PathBuf> {
    let pid = std::process::id();
    for seq in 0..10_000u32 {
        let name = format!(".coffer-ingest-{pid}-{seq}");
        if RESERVED.contains(&name.as_str()) {
            continue;
        }
        let path = directory.join(&name);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(io_ingest("artifact publish failed", None)),
        }
    }
    Err(io_ingest("artifact publish failed", None))
}

pub(super) fn backup_path(directory: &Path) -> Result<PathBuf> {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    for seq in 0..10_000u32 {
        let name = format!(".coffer-ingest-bak-{pid}-{nanos}-{seq}");
        if RESERVED.contains(&name.as_str()) {
            continue;
        }
        let path = directory.join(&name);
        if !path.exists() {
            return Ok(path);
        }
    }
    Err(io_ingest("artifact publish failed", None))
}

pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|_| io_ingest("artifact publish failed", None))?;
    std::io::Write::write_all(&mut file, bytes)
        .map_err(|_| io_ingest("artifact publish failed", None))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cofferdb-ingest-publish-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn success_replaces_formal_bytes_instead_of_appending() {
        let dir = temp_dir();
        let formal = dir.join("embeddings.jsonl");
        fs::write(&formal, b"OLD\n").unwrap();
        let temp = temporary_path(&dir).unwrap();
        write_new(&temp, b"NEW\n").unwrap();
        publish(vec![PlannedFile {
            temp: temp.clone(),
            formal: formal.clone(),
        }])
        .unwrap();
        assert_eq!(fs::read(&formal).unwrap(), b"NEW\n");
        assert!(!temp.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_publish_restores_previous_formal_bytes() {
        let dir = temp_dir();
        let first = dir.join("documents.jsonl");
        let second = dir.join("chunks.jsonl");
        fs::write(&first, b"A").unwrap();
        fs::write(&second, b"B").unwrap();
        let temp = temporary_path(&dir).unwrap();
        write_new(&temp, b"NEW").unwrap();
        let missing = dir.join(".coffer-ingest-missing");
        let error = publish(vec![
            PlannedFile {
                temp,
                formal: first.clone(),
            },
            PlannedFile {
                temp: missing,
                formal: second.clone(),
            },
        ])
        .unwrap_err();
        assert_eq!(error.code(), crate::error::ErrorCode::IoError);
        assert_eq!(fs::read(&first).unwrap(), b"A");
        assert_eq!(fs::read(&second).unwrap(), b"B");
        let _ = fs::remove_dir_all(&dir);
    }
}
