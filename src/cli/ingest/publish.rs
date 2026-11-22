//! Replace formal artifact names only after the temporary bytes are durable.
//!
//! Each rename publishes one file. A reader can see intermediate names while
//! the set is being replaced. On a rename failure, restore uses the per-file
//! progress flags to undo completed moves. Rollback is best effort: a second
//! filesystem error or process termination can leave backups or partial output.

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use crate::error::Result;

use super::boundary::io_ingest;

// Keep canonical artifact names out of the scratch and backup namespaces. A
// staging path is safe to truncate or discard only while it is distinct from every
// formal member of the publication set.
const RESERVED: &[&str] = &[
    "documents.jsonl",
    "chunks.jsonl",
    "embeddings.jsonl",
    "import_manifest.json",
    "embedding_artifact.json",
];

pub(super) struct PlannedFile {
    // Staged bytes to sync before moving any existing destination.
    pub temp: PathBuf,
    // Destination name that this artifact should occupy after publication.
    pub formal: PathBuf,
}

struct Slot {
    // Staged name still available until it has been renamed into place.
    temp: PathBuf,
    // Destination name for this member of the publication set.
    formal: PathBuf,
    // Chosen backup name if a previous destination must be preserved for rollback.
    backup: Option<PathBuf>,
    // True only after the old destination was successfully renamed to backup.
    moved_aside: bool,
    // True only after this invocation installed its staged file at the destination.
    published: bool,
}

pub(super) fn publish(files: Vec<PlannedFile>) -> Result<()> {
    let mut slots = Vec::with_capacity(files.len());
    // Sync every staged file before moving any existing name. A sync failure in this
    // pass leaves the formal names in place. Directory entries change in the two
    // rename passes below.
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
    // After preflight, either pass may fail with an arbitrary prefix already
    // moved. Use the flags on each slot to decide what the restoration pass
    // actually owns.
    let published = (|| {
        // Move all old names aside before installing the first new file. Record each
        // successful move immediately: a failure on the next file needs to restore
        // even a backup whose replacement was never published.
        for slot in &mut slots {
            if slot.formal.exists() {
                // Choose backup storage beside each destination. The set can span
                // directories, so using one common staging directory could turn a
                // local rename into a cross-filesystem operation.
                let backup = backup_path(slot.formal.parent().unwrap_or_else(|| Path::new(".")))?;
                fs::rename(&slot.formal, &backup)
                    .map_err(|_| io_ingest("artifact publish failed", None))?;
                slot.backup = Some(backup);
                slot.moved_aside = true;
            }
        }
        // Install files in the caller's order. An artifact set becomes visible one
        // name at a time, so callers that use a manifest to announce the set put it
        // last.
        for slot in &mut slots {
            fs::rename(&slot.temp, &slot.formal)
                .map_err(|_| io_ingest("artifact publish failed", None))?;
            // Mark new ownership only after the staged file has been installed. A
            // subsequent error can remove this formal name without mistaking an
            // untouched pre-existing file for new output.
            slot.published = true;
        }
        Ok(())
    })();
    // Return the publication error even when best-effort restoration encounters a
    // second problem. Success in undoing some earlier moves does not complete the
    // requested output set.
    if let Err(error) = published {
        restore(&slots);
        return Err(error);
    }
    // The new outputs are now installed. Cleanup of leftover backups is best
    // effort and cannot turn successful installation into a request to roll back
    // the whole set.
    for slot in &slots {
        if let Some(backup) = &slot.backup {
            let _ = fs::remove_file(backup);
        }
    }
    Ok(())
}

// Walk all slots on rollback. Each slot records which name changes succeeded, so
// restoring one file does not depend on whether earlier files were already
// published.
fn restore(slots: &[Slot]) {
    // Only unlink a formal name if this call installed it. A failed backup rename
    // leaves the original at that name. Try the remaining restores even if one
    // fails; the caller reports the original publication error.
    for slot in slots {
        if slot.published {
            let _ = fs::remove_file(&slot.formal);
        }
        // Choosing a backup pathname leaves the original in place. Restore it only
        // if the earlier rename set moved_aside.
        if let Some(backup) = &slot.backup {
            if slot.moved_aside {
                let _ = fs::rename(backup, &slot.formal);
            }
        }
        // Discard the staging name after attempting restoration. It may already
        // have moved to the formal name, so a cleanup failure remains secondary.
        let _ = fs::remove_file(&slot.temp);
    }
}

// Reopen the staged file by name and sync its contents before it can become a
// formal artifact. Use a writable descriptor for that sync.
fn sync_file(path: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|_| io_ingest("artifact publish failed", None))?;
    // Treat a sync failure as a publication error so the caller can follow the
    // same stage-specific cleanup as an open failure.
    file.sync_all()
        .map_err(|_| io_ingest("artifact publish failed", None))?;
    Ok(())
}

// Reserve a staging name by creating the file in the publication directory.
// Returning an existing empty file lets the writer open it later without creating
// another name.
pub(super) fn temporary_path(directory: &Path) -> Result<PathBuf> {
    let pid = std::process::id();
    // Limit attempts when earlier files already use a sequence. Only a name
    // collision selects the next attempt; other open errors are reported
    // immediately.
    for seq in 0..10_000u32 {
        let name = format!(".coffer-ingest-{pid}-{seq}");
        if RESERVED.contains(&name.as_str()) {
            continue;
        }
        let path = directory.join(&name);
        // Let create_new establish exclusive ownership even if attempts share the
        // same PID and sequence. The accepted pathname names the file this call
        // just created.
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(io_ingest("artifact publish failed", None)),
        }
    }
    Err(io_ingest("artifact publish failed", None))
}

// Choose an unused sibling name for preserving an original during publication.
// Keep it in the same directory so rename stays within the bundle's filesystem.
pub(super) fn backup_path(directory: &Path) -> Result<PathBuf> {
    let pid = std::process::id();
    // The timestamp distinguishes backup searches in one process. The sequence
    // also permits progress when the clock repeats or a candidate is already
    // present.
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
        // This branch selects a currently unused name without creating it. The
        // original remains at its formal path until the caller attempts the backup
        // rename.
        if !path.exists() {
            return Ok(path);
        }
    }
    Err(io_ingest("artifact publish failed", None))
}

// Write into a staging file already reserved by temporary_path. Open without
// create so a missing staging entry causes a failure rather than silently getting
// a new owner.
pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|_| io_ingest("artifact publish failed", None))?;
    // Write the complete artifact bytes before returning success. Durability and
    // publication are separate steps driven by the bundle owner.
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
