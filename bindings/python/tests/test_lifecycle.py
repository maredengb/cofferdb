"""Native lifecycle smoke against an installed wheel, without CLI subprocesses."""

import gc
import os
from pathlib import Path
import tempfile
import time
import unittest

from cofferdb import CofferError, ClosedError, Database, ForkedHandleError


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "lifecycle.coffer")

    def test_close_reopen_and_owned_stats(self):
        with Database.create(self.path) as database:
            statistics = database.stats()
            self.assertIsInstance(statistics, dict)
            self.assertIsInstance(database.index_stats(), dict)
        self.assertIsInstance(statistics, dict)
        database.close()
        with self.assertRaises(ClosedError) as failure:
            database.stats()
        self.assertEqual(failure.exception.sdk_reason, "closed")
        with Database.open(self.path) as reopened:
            self.assertIsInstance(reopened.stats(), dict)

    def test_create_config_and_nullable_open_inheritance(self):
        with Database.create(self.path, config={
            "resource_profile": "constrained", "cache_budget_mb": 0,
            "background_compaction": False, "durability": "commit",
            "io_mode": "read",
        }):
            pass
        with Database.open(self.path, read_only=True, options={
            "resource_profile": None, "cache_budget_mb": None,
            "background_compaction": None, "durability": None,
        }) as database:
            self.assertIsInstance(database.stats(), dict)

    def test_type_rejection_and_core_error_fields(self):
        with self.assertRaises(TypeError):
            Database.create(Path(self.path))
        with self.assertRaises(TypeError):
            Database.open(self.path, read_only=1)
        with Database.create(self.path):
            with self.assertRaises(CofferError) as failure:
                Database.open(self.path)
        error = failure.exception
        self.assertEqual(error.code, "lock_contention")
        self.assertIsInstance(error.message, str)
        self.assertIsInstance(error.detail, dict)
        self.assertIsInstance(error.recoverable, bool)
        self.assertTrue(error.suggested_action is None or
                        isinstance(error.suggested_action, str))

    def test_body_exception_and_gc_release(self):
        original = ValueError("body failed")
        with self.assertRaises(ValueError) as failure:
            with Database.create(self.path):
                raise original
        self.assertIs(failure.exception, original)
        database = Database.open(self.path)
        del database
        gc.collect()
        Database.open(self.path).close()

    @unittest.skipUnless(hasattr(os, "fork"), "requires a real POSIX fork")
    def test_inherited_owner_rejects_and_child_gc_does_not_unlock_parent(self):
        database = Database.create(self.path)
        self.addCleanup(database.close)
        child = os.fork()
        if child == 0:
            try:
                for operation in (database.stats, database.close):
                    try:
                        operation()
                    except ForkedHandleError as error:
                        assert error.sdk_reason == "forked_handle"
                    else:
                        raise AssertionError("inherited owner was accepted")
                del operation, database
                gc.collect()
                with Database.open(self.path, read_only=True) as fresh:
                    assert isinstance(fresh.stats(), dict)
                try:
                    Database.open(self.path)
                except CofferError as error:
                    assert error.code == "lock_contention"
                else:
                    raise AssertionError("child released the parent's writer lock")
            except BaseException:
                os._exit(2)
            os._exit(0)
        deadline = time.monotonic() + 5
        while True:
            completed, status = os.waitpid(child, os.WNOHANG)
            if completed:
                self.assertEqual(status, 0)
                break
            if time.monotonic() > deadline:
                os.kill(child, 9)
                os.waitpid(child, 0)
                self.fail("fork child did not finish in the bounded wait")
            time.sleep(0.005)
        self.assertIsInstance(database.stats(), dict)


if __name__ == "__main__":
    unittest.main()
