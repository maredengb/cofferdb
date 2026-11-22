"""Run with a native-test-support wheel, using python -m unittest discover."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import sys
import tempfile
import time
import unittest

from cofferdb import (
    BusyError,
    CofferError,
    Database,
    EnginePanicError,
    PoisonedHandleError,
    check,
)
from unittest.mock import patch

OPERATIONS = {
    "copy_snapshot": "snapshot",
    "snapshot_finalize": "snapshot",
    "prepare_replacement": "compact",
    "confirm_replacement": "compact",
    "write_checkpoint": "rebuild_index",
}


def wait_for(path, process=None):
    deadline = time.monotonic() + 12
    while not path.exists():
        if process and process.poll() is not None:
            raise RuntimeError(process.communicate())
        if time.monotonic() >= deadline:
            raise TimeoutError(str(path))
        time.sleep(0.005)


def worker(root, stage, operation):
    with Database.open(str(root / "main")) as db:
        notes = db.collection("notes")
        os.environ.update(
            COFFER_TEST_MAINTENANCE_STAGE=stage,
            COFFER_TEST_MAINTENANCE_DIRECTORY=str(root),
        )
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            args = (str(root / "backup"),) if operation == "snapshot" else ()
            future = pool.submit(getattr(db, operation), *args)
            wait_for(root / "ready")
            for call in (
                notes.stats,
                db.close,
                db.compact,
                lambda: db.compact(rebuild_indexes=1),
            ):
                try:
                    call()
                except BusyError:
                    continue
                raise AssertionError("shared owner admitted overlapping work")
            (root / "busy").touch()
            try:
                report = future.result(timeout=15)
            except CofferError as error:
                report = {
                    "error": error.code,
                    "detail": error.detail,
                    "outcome": getattr(
                        error,
                        "maintenance_outcome",
                        getattr(error, "snapshot_outcome", None),
                    ),
                }
        os.environ.pop("COFFER_TEST_MAINTENANCE_STAGE")
        detail = report.get("detail", {})
        if (
            detail.get("effect") != "main_replaced"
            and "cleanup_error" not in detail
            and detail.get("durability") != "unknown"
        ):
            notes.write_batch(documents=[{"external_id": "after"}])
        print(json.dumps(report))


class BoundaryTests(unittest.TestCase):
    def exercise(self, stage, response, operation=None):
        operation = operation or OPERATIONS[stage]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "main"
            backup = root / "backup"
            with Database.create(str(main)) as db:
                c = db.create_collection(
                    {"name": "notes", "embedding_dim": 2, "distance_metric": "cosine"}
                )
                c.write_batch(
                    documents=[{"external_id": "keep"}, {"external_id": "deleted"}]
                )
                c.delete_document(
                    c.get_document_by_external_id("deleted")["document_id"]
                )
            if operation == "compact":
                with Database.open(str(main)) as writer:
                    original_backup = writer.snapshot(str(backup))["sha256"]
            commit = int(check(str(main))["last_valid_commit"])
            original = main.read_bytes()
            primary_end = int(check(str(main))["critical_end_offset"])
            with Database.open(str(main), read_only=True) as old:
                with subprocess.Popen(
                    [sys.executable, "-I", __file__, directory, stage, operation],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as run:
                    try:
                        wait_for(root / "busy", run)
                        os.link(main, root / "current-alias")
                        for path in (main, root / "current-alias"):
                            with self.assertRaises(CofferError) as caught:
                                Database.open(str(path))
                            self.assertEqual(caught.exception.code, "lock_contention")
                        self.assertEqual(
                            old.collection("notes").stats()["document_count"], 1
                        )
                        if response == "cleanup-failure":
                            (staging,) = root.glob("cofferdb-compact-*.tmp")
                            staging.rename(root / "moved")
                            staging.write_bytes(b"unrelated file")
                        if response == "kill":
                            run.kill()
                        else:
                            (root / "reply").write_text(
                                "enospc" if response == "cleanup-failure" else response
                            )
                            (root / "reply").rename(root / "resume")
                        stdout, stderr = run.communicate(timeout=20)
                    finally:
                        if run.poll() is None:
                            run.kill()
                if response != "kill":
                    self.assertEqual(run.returncode, 0, stderr)
                    report = json.loads(stdout)
                    if response == "cleanup-failure":
                        self.assertIn("cleanup_error", report["detail"])
                        self.assertEqual(staging.read_bytes(), b"unrelated file")
                    elif response in ("enospc", "eio"):
                        if operation == "compact" and stage == "write_checkpoint":
                            self.assertEqual(report["index_rebuild"], "failed")
                        else:
                            self.assertIn("error", report)
                            if stage in (
                                "confirm_replacement",
                                "snapshot_finalize",
                                "write_checkpoint",
                                "verify_checkpoint",
                                "sync_checkpoint",
                                "verify_snapshot",
                            ):
                                effect = {
                                    "confirm_replacement": "main_replaced",
                                    "snapshot_finalize": "snapshot_published",
                                    "write_checkpoint": "index_may_need_rebuild",
                                    "verify_checkpoint": "index_may_need_rebuild",
                                    "sync_checkpoint": "index_may_need_rebuild",
                                    "verify_snapshot": "unchanged",
                                }[stage]
                                self.assertEqual(report["detail"]["effect"], effect)
                                self.assertEqual(report["outcome"]["effect"], effect)
                                if stage == "sync_checkpoint":
                                    self.assertEqual(
                                        report["outcome"]["durability"], "unknown"
                                    )
                                if stage == "verify_checkpoint":
                                    self.assertEqual(
                                        report["outcome"]["stage"], "verify_checkpoint"
                                    )
                                if stage == "snapshot_finalize":
                                    self.assertEqual(
                                        report["detail"]["sha256"],
                                        hashlib.sha256(backup.read_bytes()).hexdigest(),
                                    )
                    else:
                        self.assertNotIn("error", report)
            if operation == "compact":
                self.assertEqual(
                    hashlib.sha256(backup.read_bytes()).hexdigest(), original_backup
                )
            if operation == "snapshot":
                self.assertEqual(
                    backup.exists(),
                    response == "continue" or stage == "snapshot_finalize",
                )
                if backup.exists():
                    self.assertEqual(backup.read_bytes(), original)
                    self.assertEqual(check(str(backup))["last_valid_commit"], commit)
            if operation == "rebuild_index":
                self.assertEqual(
                    main.read_bytes()[:primary_end], original[:primary_end]
                )
            with Database.open(str(main)) as db:
                notes = db.collection("notes")
                self.assertEqual(
                    notes.get_document_by_external_id("keep")["external_id"], "keep"
                )
                with self.assertRaises(CofferError):
                    notes.get_document_by_external_id("deleted")
                if operation != "snapshot":
                    db.rebuild_index()
            self.assertEqual(check(str(main))["status"], "ok")

    def test_python_delivery_after_native_success(self):
        for operation, stage in (
            ("snapshot", "snapshot_finalize"),
            ("compact", "confirm_replacement"),
            ("rebuild_index", "write_checkpoint"),
        ):
            for fail in ("materialize", "signal"):
                with self.subTest(
                    operation=operation, fail=fail
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with Database.create(str(root / "main")) as db:
                        c = db.create_collection(
                            {
                                "name": "notes",
                                "embedding_dim": 2,
                                "distance_metric": "cosine",
                            }
                        )
                        c.write_batch(documents=[{"external_id": "keep"}])
                        args = (
                            (str(root / "backup"),) if operation == "snapshot" else ()
                        )

                        def interrupt():
                            wait_for(root / "ready")
                            os.kill(os.getpid(), signal.SIGINT)
                            (root / "reply").write_text("continue")
                            (root / "reply").rename(root / "resume")

                        if fail == "signal":
                            os.environ.update(
                                COFFER_TEST_MAINTENANCE_STAGE=stage,
                                COFFER_TEST_MAINTENANCE_DIRECTORY=directory,
                            )
                            thread = threading.Thread(target=interrupt)
                            thread.start()
                        else:
                            os.environ["COFFER_TEST_REFUSE_MAINTENANCE_RESULT"] = "1"
                        try:
                            from cofferdb import ResultMaterializationError

                            with self.assertRaises(
                                KeyboardInterrupt
                                if fail == "signal"
                                else ResultMaterializationError
                            ) as caught:
                                getattr(db, operation)(*args)
                        finally:
                            for key in (
                                "COFFER_TEST_MAINTENANCE_STAGE",
                                "COFFER_TEST_MAINTENANCE_DIRECTORY",
                                "COFFER_TEST_REFUSE_MAINTENANCE_RESULT",
                            ):
                                os.environ.pop(key, None)
                            if fail == "signal":
                                thread.join(15)
                        if operation == "snapshot":
                            receipt = caught.exception.snapshot_outcome
                            self.assertEqual(receipt["durability"], "confirmed")
                            self.assertEqual(
                                hashlib.sha256(
                                    (root / "backup").read_bytes()
                                ).hexdigest(),
                                receipt["sha256"],
                            )
                        elif operation == "compact":
                            self.assertEqual(
                                caught.exception.maintenance_outcome["effect"],
                                "main_replaced",
                            )
                        else:
                            self.assertEqual(
                                caught.exception.maintenance_outcome["effect"],
                                "index_written",
                            )
                        self.assertEqual(check(str(root / "main"))["status"], "ok")

    def test_native_panic_preserves_last_known_effect(self):
        for operation, stage, effect in (
            ("snapshot", "snapshot_finalize", "snapshot_published"),
            ("compact", "confirm_replacement", "main_replaced"),
            ("rebuild_index", "write_checkpoint", "index_may_need_rebuild"),
            ("compact", "measure_replacement", "main_replaced"),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                main, backup = root / "main", root / "backup"
                db = Database.create(str(main))
                db.create_collection(
                    {"name": "notes", "embedding_dim": 2, "distance_metric": "cosine"}
                )
                (root / "resume").write_text("panic")
                with patch.dict(
                    os.environ,
                    {
                        "COFFER_TEST_MAINTENANCE_STAGE": stage,
                        "COFFER_TEST_MAINTENANCE_DIRECTORY": directory,
                    },
                ), self.assertRaises(EnginePanicError) as caught:
                    getattr(db, operation)(
                        *((str(backup),) if operation == "snapshot" else ())
                    )
                receipt = getattr(
                    caught.exception,
                    "snapshot_outcome"
                    if operation == "snapshot"
                    else "maintenance_outcome",
                )
                self.assertEqual(receipt["effect"], effect)
                self.assertTrue(receipt["suggested_action"])
                if operation == "snapshot":
                    self.assertEqual(
                        receipt["sha256"],
                        hashlib.sha256(backup.read_bytes()).hexdigest(),
                    )
                if stage == "measure_replacement":
                    self.assertEqual(
                        receipt["index_rebuild"], "full_checkpoint_written"
                    )
                with self.assertRaises(PoisonedHandleError):
                    db.stats()
                try:
                    db.close()
                except CofferError as error:
                    self.assertEqual(error.code, "lock_contention")
                with Database.open(str(main)) as reopened:
                    self.assertEqual(reopened.list_collections()[0]["name"], "notes")

    def test_known_index_result_on_size_measurement_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with Database.create(str(root / "main")) as db:
                db.create_collection(
                    {"name": "notes", "embedding_dim": 2, "distance_metric": "cosine"}
                )
                (root / "resume").write_text("enospc")
                with patch.dict(
                    os.environ,
                    {
                        "COFFER_TEST_MAINTENANCE_STAGE": "measure_replacement",
                        "COFFER_TEST_MAINTENANCE_DIRECTORY": directory,
                    },
                ), self.assertRaises(CofferError) as caught:
                    db.compact()
                receipt = caught.exception.maintenance_outcome
                self.assertEqual(receipt["effect"], "main_replaced")
                self.assertEqual(receipt["index_rebuild"], "full_checkpoint_written")
                self.assertIsNone(receipt["index_rebuild_reason"])
                self.assertEqual(check(str(root / "main"))["status"], "ok")

    def test_relative_snapshot_target_survives_changed_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = root / "other"
            other.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(root)
                with Database.create(str(root / "main")) as db, patch.dict(
                    os.environ,
                    {
                        "COFFER_TEST_MAINTENANCE_STAGE": "snapshot_finalize",
                        "COFFER_TEST_MAINTENANCE_DIRECTORY": directory,
                        "COFFER_TEST_REFUSE_MAINTENANCE_RESULT": "1",
                    },
                ), concurrent.futures.ThreadPoolExecutor(1) as pool:
                    future = pool.submit(db.snapshot, "backup")
                    wait_for(root / "ready")
                    os.chdir(other)
                    (root / "resume").write_text("continue")
                    from cofferdb import ResultMaterializationError

                    with self.assertRaises(ResultMaterializationError) as caught:
                        future.result(timeout=15)
                    receipt = caught.exception.snapshot_outcome
                    self.assertEqual(
                        Path(receipt["target"]).resolve(), (root / "backup").resolve()
                    )
                    self.assertEqual(
                        hashlib.sha256(
                            Path(receipt["target"]).read_bytes()
                        ).hexdigest(),
                        receipt["sha256"],
                    )
            finally:
                os.chdir(previous)

    def test_cross_process_and_failure_boundaries(self):
        for stage in OPERATIONS:
            for action in ("continue", "enospc", "kill"):
                with self.subTest(stage=stage, action=action):
                    self.exercise(stage, action)
        self.exercise("verify_snapshot", "eio", operation="snapshot")
        self.exercise("sync_checkpoint", "eio", operation="rebuild_index")
        self.exercise("verify_checkpoint", "eio", operation="rebuild_index")
        self.exercise("write_checkpoint", "enospc", operation="compact")
        self.exercise("prepare_replacement", "cleanup-failure")


if __name__ == "__main__":
    worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
