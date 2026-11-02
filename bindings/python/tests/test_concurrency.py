"""Real installed-extension GIL, owner-gate and pending-signal checks."""

import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest

from cofferdb import BusyError, CofferError, Database


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = str(Path(directory.name) / "active.coffer")
        self.database = Database.create(self.path)
        self.addCleanup(self.database.close)
        self.collection = self.database.create_collection({
            "name": "notes", "embedding_dim": 2, "distance_metric": "cosine",
        })
        self.records = [{
            "document_external_id": "document", "chunk_external_id": f"part-{i}",
            "ordinal": i, "text": "River reeds grow beside the water. " * 16,
        } for i in range(1500)]
        self.started = threading.Event()
        self.finished = threading.Event()
        self.observed_busy = threading.Event()
        self.worker_errors = []

    def observe(self, action):
        def work():
            try:
                if not self.started.wait(5):
                    raise AssertionError("writer did not reach its start point")
                deadline = time.monotonic() + 15
                while not self.finished.is_set() and time.monotonic() < deadline:
                    try:
                        self.database.stats()
                    except BusyError as error:
                        assert error.sdk_reason == "busy"
                        action()
                        self.observed_busy.set()
                        return
                    time.sleep(0.001)
            except BaseException as error:
                self.worker_errors.append(error)
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        return worker

    def finish_worker(self, worker):
        self.finished.set()
        worker.join(5)
        self.assertFalse(worker.is_alive(), "observer exceeded its bounded join")
        if self.worker_errors:
            raise self.worker_errors[0]
        self.assertTrue(self.observed_busy.is_set(),
                        "no Python progress was observed while the native owner was busy")

    def test_native_execution_releases_gil_and_busy_rejects_without_closing(self):
        with tempfile.TemporaryDirectory() as directory:
            with Database.create(str(Path(directory) / "independent.coffer")) as other:
                def probe():
                    with self.assertRaises(BusyError):
                        self.database.close()
                    with self.assertRaises(BusyError) as failure:
                        self.collection.write_batch(documents=[{"external_id": "rejected"}])
                    self.assertEqual(failure.exception.operation_outcome, {"state": "not_started"})
                    body = ValueError("controlled body failure")
                    with self.assertRaises(BaseExceptionGroup) as cleanup:
                        self.database.__exit__(ValueError, body, None)
                    self.assertEqual(len(cleanup.exception.exceptions), 2)
                    self.assertIs(cleanup.exception.exceptions[0], body)
                    self.assertIsInstance(cleanup.exception.exceptions[1], BusyError)
                    self.assertIsInstance(other.stats(), dict)

                worker = self.observe(probe)
                self.started.set()
                try:
                    summary = self.collection.write_batch(
                        documents=[{"external_id": "document"}], chunks=self.records)
                finally:
                    self.finish_worker(worker)
        self.assertEqual(summary["chunks_inserted"], len(self.records))
        self.assertIsInstance(self.database.stats(), dict)
        with self.assertRaises(CofferError) as absent:
            self.collection.get_document_by_external_id("rejected")
        self.assertEqual(absent.exception.code, "document_not_found")

    @unittest.skipUnless(hasattr(signal, "SIGINT"), "requires a process interrupt signal")
    def test_interrupt_after_native_write_preserves_confirmed_commit(self):
        worker = self.observe(lambda: os.kill(os.getpid(), signal.SIGINT))
        self.started.set()
        try:
            with self.assertRaises(KeyboardInterrupt) as interrupted:
                self.collection.write_batch(
                    documents=[{"external_id": "document"}], chunks=self.records)
        finally:
            self.finish_worker(worker)
        outcome = interrupted.exception.operation_outcome
        self.assertEqual(outcome["state"], "committed")
        self.assertEqual(outcome["confirmed_committed_batches"], 1)
        self.assertEqual(outcome["confirmed_committed_records"], len(self.records) + 1)
        self.assertGreater(outcome["last_confirmed_commit_id"], 0)
        self.assertGreater(outcome["last_confirmed_commit_sequence_no"], 0)
        self.database.close()
        with Database.open(self.path, read_only=True) as database:
            collection = database.collection("notes")
            self.assertEqual(collection.stats()["chunk_count"], len(self.records))
            content = collection.get_chunk_by_external_id(f"part-{len(self.records) - 1}")
            self.assertEqual(content["chunk"]["text"], self.records[-1]["text"])


if __name__ == "__main__":
    unittest.main()
