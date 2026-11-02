"""Installed native API checks using neutral records and exact observed state."""

import gc
import inspect
from pathlib import Path
import tempfile
import unittest

from cofferdb import CofferError, ClosedError, Database


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "records.coffer")
        self.database = Database.create(self.path)
        self.addCleanup(self.database.close)
        self.collection = self.database.create_collection({
            "name": "notes", "embedding_dim": 2, "distance_metric": "cosine",
        })

    def seed(self):
        return self.collection.write_batch(
            documents=[{"external_id": "doc", "title": "Field notes",
                        "metadata": {"public": True}}],
            chunks=[{"document_external_id": "doc", "chunk_external_id": "first",
                     "ordinal": 0, "text": "Jade leaves grow near the river.",
                     "metadata": {"category": "plants"}},
                    {"document_external_id": "doc", "chunk_external_id": "second",
                     "ordinal": 1, "text": "Granite forms a durable stone.",
                     "metadata": {"category": "rocks"}}],
            embeddings=[{"chunk_external_id": "first", "model_name": "sample-2d",
                         "dim": 2, "vector": [1.0, 0.0]},
                        {"chunk_external_id": "second", "model_name": "sample-2d",
                         "dim": 2, "vector": (0.0, 1.0)}])

    def test_mixed_batch_read_and_visible_collections(self):
        previous_commit = self.database.stats()["last_commit_id"]
        summary = self.seed()
        self.assertEqual(self.database.stats()["last_commit_id"], previous_commit + 1)
        self.assertEqual(summary["documents_inserted"], 1)
        self.assertEqual(summary["chunks_inserted"], 2)
        self.assertEqual(summary["embeddings_inserted"], 2)
        self.assertEqual(summary["warnings"], [])
        self.assertEqual([item["name"] for item in self.database.list_collections()], ["notes"])
        other = self.database.collection("notes")
        first = other.get_chunk_by_external_id("first")
        self.assertEqual(first["chunk"]["text"], "Jade leaves grow near the river.")
        self.assertEqual(first["document"]["external_id"], "doc")
        self.assertEqual(other.get_chunk(first["chunk"]["chunk_id"]), first)
        self.assertEqual(other.get_document_by_external_id("doc"), first["document"])
        self.assertIsInstance(other.stats(), dict)

    def test_search_defaults_single_routes_and_owned_results(self):
        self.seed()
        text = self.collection.search("jade", top_k=2)
        self.assertEqual(text["results"][0]["chunk_external_id"], "first")
        vector = self.collection.search(embedding={"vector": [0.0, 1.0], "dim": 2,
                                                  "model_name": "sample-2d"})
        self.assertEqual(vector["results"][0]["chunk_external_id"], "second")
        hybrid = self.collection.search("jade", embedding={"vector": [1.0, 0.0], "dim": 2},
                                        weights={"bm25": 0.5, "vector": 0.5, "metadata": 0.0},
                                        profile="constrained")
        self.assertEqual(hybrid["profile"], "constrained")
        self.assertIn("explain", hybrid["results"][0])
        self.database.close()
        self.assertEqual(text["results"][0]["chunk_external_id"], "first")
        text["results"][0]["chunk_external_id"] = "local change"
        with self.assertRaises(ClosedError):
            self.collection.stats()
        with Database.open(self.path, read_only=True) as reopened:
            self.assertEqual(reopened.collection("notes").search("jade")["results"][0]["chunk_external_id"], "first")

    def test_failed_mixed_batch_is_atomic_and_reports_no_commit(self):
        with self.assertRaises(CofferError) as caught:
            self.collection.write_batch(documents=[{"external_id": "orphan"}], chunks=[{
                "document_external_id": "missing", "chunk_external_id": "bad",
                "ordinal": 0, "text": "Cannot refer to an absent parent.",
            }])
        self.assertEqual(caught.exception.operation_outcome["state"], "not_committed")
        with self.assertRaises(CofferError):
            self.collection.get_document_by_external_id("orphan")
        self.assertEqual(self.collection.write_batch()["documents_inserted"], 0)

    def test_invalid_final_record_does_not_commit_earlier_records(self):
        with self.assertRaises(TypeError) as caught:
            self.collection.write_batch(documents=[{"external_id": "valid"},
                                                   {"external_id": "invalid", "metadata": {"x": object()}}])
        self.assertEqual(caught.exception.operation_outcome["state"], "not_started")
        with self.assertRaises(CofferError):
            self.collection.get_document_by_external_id("valid")

    def test_batch_rejects_none_generators_and_unknown_fields(self):
        for field in ("documents", "chunks", "embeddings"):
            for invalid in (None, iter(())):
                with self.subTest(field=field, invalid=type(invalid).__name__):
                    with self.assertRaises(TypeError):
                        self.collection.write_batch(**{field: invalid})
        with self.assertRaises(CofferError) as caught:
            self.collection.write_batch(documents=[{"external_id": "doc", "chunks": []}])
        self.assertEqual(caught.exception.code, "invalid_jsonl_schema")

    def test_document_only_embedding_only_and_conflicts(self):
        self.collection.write_batch(documents=[{"external_id": "solo"}])
        document = self.collection.get_document_by_external_id("solo")
        self.assertEqual(document["external_id"], "solo")
        self.seed()
        skipped = self.collection.write_batch(documents=[{"external_id": "doc"}], on_conflict="skip")
        self.assertEqual(skipped["documents_skipped"], 1)
        replaced = self.collection.write_batch(embeddings=[{
            "chunk_external_id": "first", "model_name": "sample-2d", "dim": 2,
            "vector": [0.6, 0.8],
        }], on_conflict="replace")
        self.assertEqual(replaced["embeddings_replaced"], 1)
        self.assertIsNone(self.collection.delete_document(document["document_id"]))
        with self.assertRaises(CofferError):
            self.collection.get_document_by_external_id("solo")

    def test_delete_cascades_and_retained_collection_does_not_keep_writer(self):
        self.seed()
        first = self.collection.get_chunk_by_external_id("first")
        self.assertIsNone(self.collection.delete_chunk(first["chunk"]["chunk_id"]))
        with self.assertRaises(CofferError):
            self.collection.get_chunk_by_external_id("first")
        self.assertIsNone(self.collection.delete_document(first["document"]["document_id"]))
        with self.assertRaises(CofferError):
            self.collection.get_chunk_by_external_id("second")
        # Explicit close releases the writer while live collection references
        # remain and cyclic garbage collection is disabled.
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            self.database.close()
            with Database.open(self.path) as reopened:
                self.assertIsInstance(reopened.stats(), dict)
            with self.assertRaises(ClosedError):
                self.collection.stats()
        finally:
            if was_enabled:
                gc.enable()

    def test_export_and_manifest_reimport_with_owned_summary(self):
        self.seed()
        target = Path(self.directory.name) / "export"
        summary = self.collection.export_jsonl(str(target))
        manifest = target / "export_manifest.json"
        self.assertTrue(manifest.is_file())
        self.assertIsInstance(summary, dict)
        with Database.create(str(Path(self.directory.name) / "restored.coffer")) as restored:
            imported = restored.import_jsonl(str(manifest))
            self.assertEqual(imported["documents_inserted"], 1)
            content = restored.collection("notes").get_chunk_by_external_id("second")
            self.assertEqual(content["chunk"]["text"], "Granite forms a durable stone.")
        self.assertIsInstance(summary, dict)

    def test_source_span_preserves_full_u64_and_explicit_none_warning(self):
        self.collection.write_batch(documents=[{"external_id": "span"}], chunks=[{
            "document_external_id": "span", "chunk_external_id": "wide", "ordinal": 0,
            "text": "Encoding fixture: \0é中🙂", "source_span": {
                "start": "9007199254740992", "end": 18446744073709551615,
                "line_start": 1, "line_end": 2,
            },
        }])
        record = self.collection.get_chunk_by_external_id("wide")["chunk"]
        self.assertEqual(record["text"].encode("utf-8"), "Encoding fixture: \0é中🙂".encode("utf-8"))
        span = record["source_span"]
        self.assertEqual(span, {"start": "9007199254740992", "end": "18446744073709551615",
                                "line_start": 1, "line_end": 2})
        result = self.collection.write_batch(chunks=[{
            "document_external_id": "span", "chunk_external_id": "explicit", "ordinal": 1,
            "text": "Unknown source position.", "source_span": None,
            "metadata": {"byte_start": 1, "byte_end": 2},
        }])
        self.assertEqual(len(result["warnings"]), 1)
        content = self.collection.get_chunk_by_external_id("explicit")["chunk"]
        self.assertIsNone(content["source_span"])
        self.assertEqual(content["metadata"], {"byte_start": 1, "byte_end": 2})

    def test_exact_top_k_and_published_defaults(self):
        for invalid in (True, 1.0, None):
            with self.subTest(value=invalid):
                with self.assertRaises(TypeError):
                    self.collection.search("jade", top_k=invalid)
        for invalid in (0, -1, 9007199254740992, 10 ** 100):
            with self.subTest(value=str(invalid)):
                with self.assertRaises(CofferError) as caught:
                    self.collection.search("jade", top_k=invalid)
                self.assertEqual(caught.exception.code, "invalid_search_params")
        self.assertEqual(inspect.signature(self.collection.search).parameters["top_k"].default, 10)
        self.assertEqual(inspect.signature(self.collection.write_batch).parameters["documents"].default, ())
        with self.assertRaises(CofferError) as caught:
            self.collection.search()
        self.assertEqual(caught.exception.code, "invalid_search_params")

    def test_read_only_mutations_are_core_failures(self):
        self.seed()
        self.database.close()
        with Database.open(self.path, read_only=True) as database:
            collection = database.collection("notes")
            before_stats = database.stats()
            before_collection_stats = collection.stats()
            before_bytes = Path(self.path).read_bytes()
            with self.assertRaises(CofferError) as caught:
                collection.write_batch(documents=[{"external_id": "denied"}])
            self.assertEqual(caught.exception.code, "permission_denied")
            self.assertEqual(caught.exception.operation_outcome["state"], "not_committed")
            self.assertEqual(database.stats(), before_stats)
            self.assertEqual(collection.stats(), before_collection_stats)
            self.assertEqual(Path(self.path).read_bytes(), before_bytes)
            self.assertEqual(collection.get_document_by_external_id("doc")["external_id"], "doc")
            with self.assertRaises(CofferError):
                collection.get_document_by_external_id("denied")
        self.assertEqual(Path(self.path).read_bytes(), before_bytes)

    def test_full_content_survives_source_move_and_reopen(self):
        # This neutral multilingual body exercises exact UTF-8 ownership,
        # including embedded NULs; it is not a retrieval-quality fixture.
        body = ("Field note: jade river. \0é中🙂\n" * 32768) + "Final boundary."
        expected_bytes = body.encode("utf-8")
        self.assertGreater(len(expected_bytes), 1024 * 1024)
        sources = Path(self.directory.name) / "sources"
        sources.mkdir()
        original = sources / "body.txt"
        original.write_bytes(expected_bytes)
        span = {"start": 0, "end": len(expected_bytes), "line_start": 1, "line_end": 32769}
        summary = self.collection.write_batch(
            documents=[{"external_id": "full", "title": "Complete field note",
                        "source_uri": original.as_uri(), "metadata": {"purpose": "encoding"}}],
            chunks=[{"document_external_id": "full", "chunk_external_id": "full-body",
                     "ordinal": 0, "text": body, "source_span": span,
                     "metadata": {"complete": True}}])
        self.assertEqual((summary["documents_inserted"], summary["chunks_inserted"]), (1, 1))
        before = self.collection.get_chunk_by_external_id("full-body")
        self.database.close()
        sources.rename(Path(self.directory.name) / "moved-sources")
        self.assertFalse(original.exists())
        with Database.open(self.path, read_only=True) as reopened:
            collection = reopened.collection("notes")
            after = collection.get_chunk_by_external_id("full-body")
            self.assertEqual(after, before)
            self.assertEqual(after["chunk"]["text"].encode("utf-8"), expected_bytes)
            self.assertEqual(after["chunk"]["source_span"],
                             span)
            self.assertEqual(after["chunk"]["metadata"], {"complete": True})
            self.assertEqual(after["document"]["metadata"], {"purpose": "encoding"})
            self.assertEqual(after["document"]["source_uri"], original.as_uri())
            self.assertEqual(collection.get_chunk(after["chunk"]["chunk_id"]), after)
            retained = after
        self.assertEqual(retained["chunk"]["text"].encode("utf-8"), expected_bytes)


if __name__ == "__main__":
    unittest.main()
