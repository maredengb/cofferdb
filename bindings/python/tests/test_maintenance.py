"""Installed-wheel maintenance and single-file recovery, without an installed CLI."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cofferdb
from cofferdb import CofferError, ClosedError, Database


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_notes(db):
    notes = db.create_collection(
        {
            "name": "notes",
            "embedding_dim": 2,
            "distance_metric": "cosine",
            "embedding_model": "plants",
        }
    )
    notes.write_batch(documents=[{"external_id": "one"}, {"external_id": "obsolete"}])
    notes.write_batch(
        documents=[{"external_id": "one", "metadata": {"revision": 2}}],
        chunks=[
            {
                "document_external_id": "one",
                "chunk_external_id": "fern",
                "ordinal": 0,
                "text": "Jade fern",
            }
        ],
        embeddings=[
            {
                "chunk_external_id": "fern",
                "model_name": "plants",
                "dim": 2,
                "vector": [1.0, 0.0],
            }
        ],
        on_conflict="replace",
    )
    notes.delete_document(notes.get_document_by_external_id("obsolete")["document_id"])
    return notes


def verify_restored(path, commit):
    before = digest(path)
    assert int(cofferdb.check(str(path))["last_valid_commit"]) == commit
    with Database.open(str(path), read_only=True) as db:
        notes = db.collection("notes")
        (info,) = db.list_collections()
        assert (
            info["embedding_dim"],
            info["distance_metric"],
            info["embedding_model"],
        ) == (2, "cosine", "plants")
        assert notes.get_document_by_external_id("one")["metadata"] == {"revision": 2}
        assert notes.get_chunk_by_external_id("fern")["chunk"]["text"] == "Jade fern"
        assert notes.stats()["document_count"] == 1
        results = notes.search(
            embedding={"model_name": "plants", "dim": 2, "vector": [1.0, 0.0]}
        )["results"]
        (hit,) = results
        assert (hit["chunk_external_id"], hit["vector_score"]) == ("fern", 1.0)
        notes.export_jsonl("export")
        (vector,) = Path("export/embeddings.jsonl").read_text().splitlines()
        assert json.loads(vector)["vector"] == [1.0, 0.0]
    assert digest(path) == before


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.files = ExitStack()
        self.addCleanup(self.files.close)
        self.root = Path(self.files.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / "source"
        self.target = self.root / "backup"
        self.db = Database.create(str(self.source))
        self.files.callback(self.db.close)

    def test_snapshot_restore(self):
        notes = setup_notes(self.db)
        before = digest(self.source)
        commit = int(self.db.stats()["last_commit_id"])
        report = self.db.snapshot(str(self.target))
        self.assertEqual(digest(self.source), before)
        self.assertEqual(report.keys(), cofferdb.SnapshotReport.__required_keys__)
        self.assertEqual(
            (report["sha256"], report["size_bytes"], report["last_valid_commit"]),
            (digest(self.target), self.target.stat().st_size, commit),
        )
        notes.write_batch(documents=[{"external_id": "later"}])
        self.db.close()
        with tempfile.TemporaryDirectory() as restore:
            copy = Path(restore) / "only.coffer"
            shutil.copyfile(self.target, copy)
            self.target.unlink()
            self.source.unlink()
            subprocess.run(
                [sys.executable, "-I", __file__, str(copy), str(commit)],
                cwd=restore,
                check=True,
            )

    def test_rebuild_reopen(self):
        setup_notes(self.db)
        for name in (None, "notes"):
            report = self.db.rebuild_index(name)
            self.assertEqual(
                report.keys(), cofferdb.RebuildIndexReport.__required_keys__
            )
            self.assertEqual(
                (report["collection"], report["rebuild_scope"]),
                (name, "full_visible_snapshot"),
            )
            self.assertFalse(report["rebuild_needed"])
            self.assertEqual(
                report["last_index_commit_id"],
                int(
                    cofferdb.check(str(self.source))["index_suffix"][
                        "covered_last_commit_sequence_no"
                    ]
                ),
            )
        with self.assertRaises(CofferError) as caught:
            self.db.rebuild_index("missing")
        self.assertEqual(caught.exception.code, "collection_not_found")
        self.db.close()
        with Database.open(str(self.source), read_only=True) as reader:
            self.assertEqual(
                reader.index_stats()["last_index_commit_id"],
                report["last_index_commit_id"],
            )

    def test_compact_generations(self):
        notes = setup_notes(self.db)
        alias = self.root / "old"
        os.link(self.source, alias)
        with Database.open(str(self.source), read_only=True) as old:
            notes.write_batch(documents=[{"external_id": "later"}])
            before = digest(alias)
            size = alias.stat().st_size
            report = self.db.compact()
            self.assertEqual(report.keys(), cofferdb.CompactReport.__required_keys__)
            self.assertEqual(
                (report["before_file_size_bytes"], report["after_file_size_bytes"]),
                (size, self.source.stat().st_size),
            )
            self.assertEqual(report["index_rebuild"], "full_checkpoint_written")
            self.assertGreater(report["records_removed"], 0)
            self.assertEqual(old.collection("notes").stats()["document_count"], 1)
            with self.assertRaises(CofferError) as caught:
                Database.open(str(alias))
            self.assertEqual(
                (caught.exception.code, caught.exception.detail["reason"]),
                ("recovery_error", "writer_reopen_required"),
            )
            self.assertEqual(digest(alias), before)
        skipped = self.db.compact(rebuild_indexes=False)
        self.assertEqual(
            (skipped["index_rebuild"], skipped["index_rebuild_reason"]),
            ("disabled", None),
        )
        notes.write_batch(documents=[{"external_id": "after"}])
        self.db.close()
        with Database.open(str(self.source)) as reopened:
            self.assertEqual(reopened.collection("notes").stats()["document_count"], 3)

    def test_snapshot_publish_race(self):
        contender = """
import json, sys, time
from pathlib import Path
from cofferdb import Database, CofferError
name, root = sys.argv[1], Path(sys.argv[2])
with Database.create(str(root / name)) as writer:
    writer.create_collection({"name":name, "embedding_dim":2, "distance_metric":"cosine"})
    (root / (name + ".ready")).touch()
    deadline = time.monotonic() + 8
    while not all((root / (other+".ready")).exists() for other in ("north","south")):
        if time.monotonic() > deadline: raise TimeoutError(name)
        time.sleep(.005)
    try:
        print(json.dumps(writer.snapshot(str(root / "winner"))))
    except CofferError as error:
        print(json.dumps({"error":error.code}))
"""
        with ExitStack() as processes:
            children = [
                processes.enter_context(
                    subprocess.Popen(
                        [sys.executable, "-I", "-c", contender, name, str(self.root)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
                for name in ("north", "south")
            ]
            replies = []
            for child in children:
                out, err = child.communicate(timeout=12)
                self.assertEqual(child.returncode, 0, err)
                replies.append(json.loads(out))
        (winner,) = (r for r in replies if "sha256" in r)
        (loser,) = (r for r in replies if "error" in r)
        self.assertEqual(loser["error"], "invalid_search_params")
        self.assertEqual(digest(self.root / "winner"), winner["sha256"])
        with Database.open(str(self.root / "winner"), read_only=True) as reader:
            self.assertEqual(reader.list_collections(), winner["collections"])

    def test_maintenance_rejections(self):
        for target in (self.source, self.target):
            if target == self.target:
                target.write_bytes(b"existing backup")
            before = digest(target)
            with self.assertRaises(CofferError):
                self.db.snapshot(str(target))
            self.assertEqual(digest(target), before)
        dangling = self.root / "link"
        dangling.symlink_to("uncreated")
        alias = self.root / "alias"
        os.link(self.source, alias)
        for existing in (dangling, alias):
            with self.assertRaises(CofferError) as caught:
                self.db.snapshot(str(existing))
            self.assertEqual(caught.exception.code, "invalid_search_params")
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / "uncreated").exists())
        with self.assertRaises(TypeError):
            self.db.compact(rebuild_indexes=1)
        with self.assertRaises(TypeError):
            self.db.rebuild_index(5)
        self.db.close()
        with Database.open(str(self.source), read_only=True) as reader:
            for db, error in ((reader, CofferError), (self.db, ClosedError)):
                for method, args in (
                    ("snapshot", (str(self.root / "absent"),)),
                    ("compact", ()),
                    ("rebuild_index", ()),
                ):
                    with self.assertRaises(error):
                        getattr(db, method)(*args)
        self.assertFalse((self.root / "absent").exists())


if __name__ == "__main__":
    verify_restored(Path(sys.argv[1]), int(sys.argv[2]))
