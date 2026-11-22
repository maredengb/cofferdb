"""Exercise path-based diagnostics through the installed extension."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cofferdb


class CheckTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "notes.coffer"

    def compare_cli(self, report, code):
        cli = os.environ.get("COFFER_TEST_CLI")
        if cli:
            run = subprocess.run(
                [cli, "check", str(self.path), "--json"], capture_output=True,
                text=True, timeout=30,
            )
            self.assertEqual(run.returncode, code, run.stderr)
            self.assertEqual(report, json.loads(run.stdout)["result"])

    def test_idle_external_writer_and_no_cli(self):
        program = """
import cofferdb, sys
with cofferdb.Database.create(sys.argv[1]) as db:
    c = db.create_collection({"name":"notes", "embedding_dim":2, "distance_metric":"cosine"})
    c.write_batch(documents=[{"external_id":"one"}])
    print("ready", flush=True)
    input()
"""
        with subprocess.Popen([sys.executable, "-c", program, str(self.path)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as writer:
            try:
                self.assertEqual(writer.stdout.readline(), "ready\n")
                with self.assertRaises(cofferdb.CofferError) as competing_writer:
                    cofferdb.Database.open(str(self.path))
                self.assertEqual(competing_writer.exception.code, "lock_contention")
                before = hashlib.sha256(self.path.read_bytes()).digest()
                with patch.dict(os.environ, {"PATH": ""}):
                    report = cofferdb.check(str(self.path))
                self.assertEqual(report["result"]["live_documents"], 1)
                self.assertEqual(report.keys(), cofferdb.CheckReport.__required_keys__)
                self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), before)
                self.compare_cli(report, 0)
            finally:
                if writer.poll() is None:
                    with contextlib.suppress(BrokenPipeError):
                        writer.stdin.write("\n")
                        writer.stdin.flush()
                    writer.wait(timeout=30)

    def test_missing_path_and_invalid_type(self):
        with self.assertRaises(cofferdb.CofferError) as caught:
            cofferdb.check(str(self.path))
        self.assertFalse(self.path.exists())
        self.assertTrue(caught.exception.code)
        self.assertIsInstance(caught.exception.detail, dict)
        self.assertIsInstance(caught.exception.recoverable, bool)
        with self.assertRaises(TypeError):
            cofferdb.check(self.path)

    def test_unavailable_view_retains_core_diagnostic(self):
        cofferdb.Database.create(str(self.path)).close()
        # A v0.1 writer holds this lock over the entire data inode.
        with self.path.open("r+b") as old_writer:
            fcntl.flock(old_writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(cofferdb.CofferError) as caught:
                cofferdb.check(str(self.path))
            error = caught.exception
            self.assertEqual(error.code, "recovery_error")
            self.assertEqual(error.detail["reason"], "check_view_unavailable")
            self.assertEqual(error.detail["uncompleted_scope"], "main_and_index")
            self.assertTrue(error.recoverable)
            self.assertIn("retry", error.suggested_action)
            cli = os.environ.get("COFFER_TEST_CLI")
            if cli:
                run = subprocess.run([cli, "check", str(self.path), "--json"],
                                     capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 2)
                self.assertEqual(run.stdout, "")
                diagnostic = json.loads(run.stderr)["error"]
                for field in ("code", "detail", "recoverable", "suggested_action"):
                    self.assertEqual(getattr(error, field), diagnostic[field])
        recovered = cofferdb.check(str(self.path))
        self.assertEqual(recovered["status"], "ok")
        self.assertEqual(recovered["index_check"], "skipped")
        self.assertEqual(recovered["index_check_reason"], "persistent_index_checkpoint_missing")
        self.compare_cli(recovered, 0)

    def test_completed_report_contains_issues(self):
        cofferdb.Database.create(str(self.path)).close()
        with self.path.open("ab") as stream:
            stream.write(b"incomplete")
        expected = self.path.read_bytes()
        report = cofferdb.check(str(self.path))
        self.assertTrue(report["issues"])
        self.assertEqual(report["issues"][0].keys(), cofferdb.CheckIssue.__required_keys__)
        self.assertEqual(self.path.read_bytes(), expected)
        self.compare_cli(report, 3)
