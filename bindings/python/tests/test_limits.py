"""Functional resource boundaries; no latency or application-quality gate."""

from pathlib import Path
import tempfile
import unittest

from cofferdb import CofferError, Database


class LimitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "limits.coffer"
        self.database = Database.create(str(self.path))
        self.addCleanup(self.database.close)
        self.collection = self.database.create_collection({
            "name": "notes", "embedding_dim": 2, "distance_metric": "cosine",
        })

    def test_large_text_is_not_a_business_json_value(self):
        text = "x" * (9 * 1024 * 1024)
        summary = self.collection.write_batch(documents=[{"external_id": "doc"}], chunks=[{
            "document_external_id": "doc", "chunk_external_id": "large",
            "ordinal": 0, "text": text,
        }])
        self.assertEqual(summary["chunks_inserted"], 1)
        result = self.collection.get_chunk_by_external_id("large")
        self.database.close()
        self.assertEqual(result["chunk"]["text"], text)

    def test_business_unit_limit_rejects_metadata_before_commit(self):
        before = self.path.read_bytes()
        with self.assertRaises(CofferError) as caught:
            self.collection.write_batch(documents=[{
                "external_id": "too-large", "metadata": {"value": "x" * (8 * 1024 * 1024)},
            }])
        self.assertEqual(caught.exception.detail["json_input_kind"], "document_metadata")
        self.assertEqual(caught.exception.detail["json_limit_reason"], "unit_bytes_exceeded")
        self.assertEqual(caught.exception.operation_outcome, {"state": "not_started"})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.collection.write_batch()["documents_inserted"], 0)

    def test_one_batch_cannot_split_around_the_shared_work_budget(self):
        before = self.path.read_bytes()
        text = "x" * (9 * 1024 * 1024)
        chunks = [{"document_external_id": "doc", "chunk_external_id": str(index),
                   "ordinal": index, "text": text} for index in range(4)]
        with self.assertRaises(CofferError) as caught:
            self.collection.write_batch(documents=[{"external_id": "doc"}], chunks=chunks)
        self.assertEqual(caught.exception.code, "resource_exhausted")
        self.assertEqual(caught.exception.detail["resource"], "memory")
        self.assertEqual(int(caught.exception.detail["limit_bytes"]), 32 * 1024 * 1024)
        self.assertEqual(caught.exception.operation_outcome, {"state": "not_started"})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.database.stats()["document_count"], 0)
        self.collection.write_batch(documents=[{"external_id": "after-failure"}])
        self.assertEqual(self.collection.get_document_by_external_id("after-failure")["external_id"], "after-failure")

    def test_json_precision_depth_cycles_and_no_implicit_protocols(self):
        shared = {"label": "same"}
        self.collection.write_batch(documents=[{"external_id": "shared", "metadata": {"left": shared, "right": shared}}])
        self.assertEqual(self.collection.get_document_by_external_id("shared")["metadata"], {"left": shared, "right": shared})
        cycle = {}
        cycle["self"] = cycle
        invalid_inputs = [cycle, {"rounded": 9007199254740993}, {"surrogate": "\ud800"}]
        for value in invalid_inputs:
            with self.subTest(shape=list(value)), self.assertRaises(ValueError):
                self.collection.write_batch(documents=[{"external_id": "invalid", "metadata": value}])
        nested = {}
        for _ in range(128):
            nested = {"child": nested}
        with self.assertRaises(CofferError) as caught:
            self.collection.write_batch(documents=[{"external_id": "deep", "metadata": nested}])
        self.assertEqual(caught.exception.detail["json_limit_reason"], "depth_exceeded")
        self.assertEqual(self.database.stats()["document_count"], 1)


if __name__ == "__main__":
    unittest.main()
