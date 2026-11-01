"""Check the actual installed package, its stubs and its public owned shapes."""

import ast
import importlib
import importlib.metadata
import importlib.resources
import inspect
from pathlib import Path
import tempfile
import unittest
from typing import get_args, get_origin

import cofferdb


class PublicTypesTests(unittest.TestCase):
    def test_recursive_tuple_input_is_declared_and_returns_owned_lists(self):
        self.assertTrue(any(get_origin(member) is tuple for member in get_args(cofferdb.JsonValue)))
        with tempfile.TemporaryDirectory() as directory:
            with cofferdb.Database.create(str(Path(directory) / "tuples.coffer")) as db:
                collection = db.create_collection({"name": "notes", "embedding_dim": 2, "distance_metric": "cosine"})
                metadata = {"nested": ({"enabled": True, "values": (1, "jade")},)}
                collection.write_batch(documents=[{"external_id": "doc", "metadata": metadata}])
                document = collection.get_document_by_external_id("doc")
            metadata["nested"][0]["enabled"] = False
            self.assertEqual(document["metadata"], {"nested": [{"enabled": True, "values": [1, "jade"]}]})

    def test_sdk_reason_is_declared_only_on_the_diagnostic_subclasses(self):
        package = importlib.resources.files("cofferdb")
        classes = {node.name: node for node in ast.parse(package.joinpath("__init__.pyi").read_text()).body
                   if isinstance(node, ast.ClassDef)}
        for name in ("SdkError", "ResultMaterializationError"):
            self.assertFalse(any(isinstance(node, ast.AnnAssign) and node.target.id == "sdk_reason"
                                 for node in classes[name].body))
            self.assertFalse(hasattr(getattr(cofferdb, name)("controlled"), "sdk_reason"))
        with tempfile.TemporaryDirectory() as directory:
            db = cofferdb.Database.create(str(Path(directory) / "closed.coffer"))
            db.close()
            with self.assertRaises(cofferdb.ClosedError) as raised:
                db.stats()
            self.assertEqual(raised.exception.sdk_reason, "closed")
            self.assertTrue(any(isinstance(node, ast.AnnAssign) and node.target.id == "sdk_reason"
                                for node in classes["ClosedError"].body))

    def test_mismatched_native_version_is_rejected(self):
        native = importlib.import_module("cofferdb._native")
        original = native.__version__
        try:
            native.__version__ = "0.0.0"
            with self.assertRaisesRegex(ImportError, "versions do not match"):
                importlib.reload(cofferdb)
        finally:
            native.__version__ = original
            importlib.reload(cofferdb)

    def test_distribution_contains_the_project_and_dependency_licenses(self):
        metadata = importlib.metadata.metadata("cofferdb")
        self.assertEqual(metadata["Version"], cofferdb.__version__)
        self.assertEqual(metadata["License-Expression"], "Apache-2.0")
        licenses = metadata.get_all("License-File") or []
        self.assertIn("LICENSE", licenses)
        self.assertTrue(any("pyo3-0.27.2/" in name for name in licenses))
        files = importlib.metadata.files("cofferdb")
        inventories = [path for path in files if str(path).endswith("licenses/inventory.json")]
        self.assertEqual(len(inventories), 1)
        self.assertTrue(inventories[0].locate().is_file())

    def test_public_types_and_packaged_typing_files(self):
        package = importlib.resources.files("cofferdb")
        self.assertTrue(package.joinpath("py.typed").is_file())
        stub = ast.parse(package.joinpath("__init__.pyi").read_text())
        self.assertEqual(cofferdb.__version__, "0.1.1")
        self.assertIsInstance(cofferdb.engine_version, str)
        for name in cofferdb.__all__:
            self.assertTrue(hasattr(cofferdb, name), name)
        self.assertEqual(cofferdb.CollectionConfigInput.__required_keys__, {"name", "embedding_dim", "distance_metric"})
        for node in stub.body:
            if isinstance(node, ast.ClassDef) and node.name in ("Database", "Collection"):
                native = getattr(cofferdb, node.name)
                for method in node.body:
                    if not isinstance(method, ast.FunctionDef):
                        continue
                    self.assertTrue(hasattr(native, method.name), (node.name, method.name))
                    signature = inspect.signature(getattr(native, method.name))
                    expected = [arg.arg for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs]
                    self.assertEqual(list(signature.parameters), expected, (node.name, method.name))

    def test_core_owned_shapes_match_the_published_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            with cofferdb.Database.create(str(Path(directory) / "types.coffer")) as db:
                collection = db.create_collection({"name": "notes", "embedding_dim": 2, "distance_metric": "cosine"})
                summary = collection.write_batch(documents=[{"external_id": "doc"}], chunks=[{
                    "document_external_id": "doc", "chunk_external_id": "chunk", "ordinal": 0,
                    "text": "Jade leaves grow.",
                }])
                content = collection.get_chunk_by_external_id("chunk")
                search = collection.search("jade")
                shapes = [(cofferdb.DatabaseStats, db.stats()), (cofferdb.IndexStats, db.index_stats()),
                          (cofferdb.CollectionInfo, db.list_collections()[0]), (cofferdb.CollectionStats, collection.stats()),
                          (cofferdb.Document, content["document"]), (cofferdb.Chunk, content["chunk"]),
                          (cofferdb.ChunkContent, content), (cofferdb.WriteSummary, summary),
                          (cofferdb.SearchResults, search), (cofferdb.SearchHit, search["results"][0]),
                          (cofferdb.CandidateCounts, search["candidate_counts"]),
                          (cofferdb.SearchExecutionStats, search["execution_stats"]),
                          (cofferdb.NormalizedScores, search["results"][0]["normalized_scores"])]
                for shape, value in shapes:
                    with self.subTest(shape=shape.__name__):
                        self.assertEqual(set(value), shape.__required_keys__)


if __name__ == "__main__":
    unittest.main()
