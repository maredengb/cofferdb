"""AST audit and optional --fake control tests; never imports the real SDK."""

import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile


root = Path(__file__).resolve().parent
tree = ast.parse((root / "driver.py").read_text(encoding="utf-8"))
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.append(node.module)
allowed = {"argparse", "hashlib", "json", "pathlib", "sys", "traceback", "cofferdb"}
if set(imports) - allowed:
    raise SystemExit("unexpected import")
if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
    raise SystemExit("optimization-sensitive assert")
calls = {node.func.attr for node in ast.walk(tree)
         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
expected = {"create", "open", "create_collection", "collection", "write_batch", "import_jsonl",
            "search", "get_document_by_external_id", "get_chunk_by_external_id", "get_chunk",
            "export_jsonl", "stats"}
if not expected <= calls:
    raise SystemExit("missing business API coverage")
for name in ("driver.py", "static_check.py"):
    text = (root / name).read_text(encoding="utf-8")
    ast.parse(text, filename=name)
    if any(line.rstrip() != line for line in text.splitlines()):
        raise SystemExit("trailing whitespace: " + name)
print("PASS_STATIC: 2 AST parses, import allowlist, explicit checks, 12 SDK API call names, whitespace")
print("NOT RUN: real SDK import, native loading, real-package business, installation, platform and network tests")


def fake_checks(bundle):
    driver = runpy.run_path(str(root / "driver.py"), run_name="fake_control_only")
    documents, chunks, embeddings, files = driver["fixture"](bundle)
    require = driver["require"]
    require("cofferdb" not in sys.modules, "real SDK must not be imported")

    class ClosedError(Exception):
        pass

    def fake_database(fault):
        stores = {}
        ids = {}

        class Collection:
            def __init__(self, db):
                self.db = db

            def stats(self):
                if self.db.closed and fault != "close":
                    raise ClosedError()
                return {}

            def write_batch(self, *, documents, chunks, embeddings):
                self.db.rows = copy.deepcopy((documents, chunks, embeddings))
                stores[self.db.path] = self.db.rows
                return {"documents_inserted": len(documents), "chunks_inserted": len(chunks),
                        "embeddings_inserted": 0 if fault == "write_count" else len(embeddings)}

            def get_document_by_external_id(self, external_id):
                self.stats()
                document = next(row for row in self.db.rows[0] if row["external_id"] == external_id)
                return {**copy.deepcopy(document), "document_id": self.db.uid + external_id}

            def get_chunk_by_external_id(self, external_id):
                self.stats()
                chunk = copy.deepcopy(next(row for row in self.db.rows[1]
                                           if row["chunk_external_id"] == external_id))
                document = self.get_document_by_external_id(chunk.pop("document_external_id"))
                chunk.update(chunk_id=self.db.uid + external_id, document_id=document["document_id"])
                if fault == "body":
                    chunk["text"] = "truncated"
                if fault == "span":
                    chunk["source_span"]["end"] += 1
                if fault == "source_fallback" and self.db.read_only:
                    source = Path(self.db.path).parent / document["source_uri"]
                    require(source.exists(), "source fallback attempted")
                return {"document": document, "chunk": chunk}

            def get_chunk(self, chunk_id):
                require(chunk_id.startswith(self.db.uid), "fake cross-database ID leak")
                return self.get_chunk_by_external_id(chunk_id[len(self.db.uid):])

            def search(self, *, text=None, embedding=None, top_k, profile, weights):
                self.stats()
                require(top_k == 5 and profile == "balanced" and
                        weights == {"bm25": 0.5, "vector": 0.5, "metadata": 0.0}, "query config")
                if embedding is not None:
                    require(embedding == {key: embeddings[0][key] for key in ("vector", "dim", "model_name")},
                            "query must reuse frozen self vector")
                require(text in (None, "pangram"), "query text")
                hits = []
                selected = self.db.rows[1] if embedding is not None else self.db.rows[1][:1]
                for index, row in enumerate(selected):
                    content = self.get_chunk_by_external_id(row["chunk_external_id"])
                    if fault == "hit_get_mismatch" and index == 0:
                        content = self.get_chunk_by_external_id(self.db.rows[1][1]["chunk_external_id"])
                    score = 1.0 / (index + 1)
                    if fault == "score" and Path(self.db.path).name == "batch.coffer":
                        score += 0.1
                    explain = {"schema_version": "0.1", "collection": "notes", "final_score": score,
                               "chunk_id": content["chunk"]["chunk_id"],
                               "document_id": content["document"]["document_id"],
                               "chunk_external_id": row["chunk_external_id"],
                               "document_external_id": content["document"]["external_id"],
                               "scores": {"final_score": score}, "weights": copy.deepcopy(weights),
                               "normalization": {"scope": "candidate_set", "candidate_count": len(selected)},
                               "matched_terms": ["pangram"] if text else [], "bm25": {},
                               "matched_term_details": [], "tokenizer": {}, "fusion": "weighted_sum",
                               "candidate_sources": (["bm25"] if text else []) + (["vector"] if embedding else []),
                               "resource_profile": "balanced", "candidate_limits": {"fusion": 5},
                               "distance_metric": "cosine", "vector_raw_kind": "similarity",
                               "normalization_method": "cosine", "filter": None,
                               "metadata_boost_context": {"present": False, "config_hash": None},
                               "chunk_boundary": row["source_span"],
                               "source": {"source_uri": content["document"].get("source_uri")},
                               "index_source": self.db.uid, "index_rebuild_reason": self.db.uid,
                               "transient_rebuild": {"fake_instance": self.db.uid},
                               "execution_stats": {"fake_instance": self.db.uid}}
                    if fault == "explain":
                        del explain["weights"]
                    if fault == "explain_id":
                        explain["document_id"] = "wrong-database-id"
                    hits.append({"chunk_id": content["chunk"]["chunk_id"],
                                 "document_id": content["document"]["document_id"],
                                 "document_external_id": content["document"]["external_id"],
                                 "chunk_external_id": row["chunk_external_id"], "final_score": score,
                                 "snippet": row["text"] if fault == "snippet" else row["text"][:80],
                                 "explain": explain})
                return {"profile": profile, "results": hits}

            def export_jsonl(self, path):
                self.stats()
                target = Path(path)
                target.mkdir()
                manifest = json.loads(files["import_manifest.json"])
                for name, rows in zip(("documents.jsonl", "chunks.jsonl", "embeddings.jsonl"), self.db.rows):
                    (target / name).write_bytes(b"".join(map(driver["encoded"], rows)))
                (target / "export_manifest.json").write_bytes(driver["encoded"](manifest))
                return {"documents": len(self.db.rows[0]), "chunks": len(self.db.rows[1]),
                        "embeddings": 0 if fault == "export_count" else len(self.db.rows[2])}

        class Database:
            def __init__(self, path, read_only=False):
                self.path, self.read_only, self.closed = path, read_only, False
                self.uid = ids.setdefault(path, "database-" + str(len(ids)) + ":")
                instances.append(self)
                self.rows = copy.deepcopy(stores.get(path, ([], [], [])))

            @classmethod
            def create(cls, path):
                require(path not in stores, "fake create must be new")
                return cls(path)

            @classmethod
            def open(cls, path, read_only=False):
                require(path in stores, "fake open must exist")
                return cls(path, read_only)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def collection(self, name):
                require(name == "notes", "collection name")
                return Collection(self)

            def create_collection(self, config):
                require(config == {"name": "notes", "embedding_dim": 1024, "embedding_model": "BAAI/bge-m3",
                                   "vectors_normalized": False, "distance_metric": "cosine"},
                        "collection config")
                return self.collection(config["name"])

            def import_jsonl(self, path):
                manifest = json.loads(Path(path).read_bytes())
                rows = [[json.loads(line) for line in (Path(path).parent / manifest[key]).read_bytes().splitlines()]
                        for key in ("documents", "chunks", "embeddings")]
                return self.collection(manifest["collection"]).write_batch(
                    documents=rows[0], chunks=rows[1], embeddings=rows[2])

        instances = []
        return Database

    cases = {None: None, "write_count": "embeddings_inserted", "export_count": "export embeddings",
             "body": "chunk text", "span": "source_span", "explain": "explain fields",
             "explain_id": "explain document id", "snippet": "body must exceed actual snippet",
             "score": "manifest/batch query equivalence", "close": "collection remains usable",
             "source_fallback": "source fallback attempted",
             "hit_get_mismatch": "hit/get chunk external id"}
    # Run only fake in-memory APIs; filesystem effects stay in disposable test roots.
    for fault, expected_error in cases.items():
        captured = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="f-smoke-fake-") as directory:
            error = None
            with contextlib.redirect_stdout(captured):
                try:
                    driver["run"](fake_database(fault), ClosedError, Path(directory) / "attempt",
                                  copy.deepcopy(documents), copy.deepcopy(chunks), copy.deepcopy(embeddings), files)
                except AssertionError as caught:
                    error = str(caught)
            events = [json.loads(line) for line in captured.getvalue().splitlines()]
            passed = any(event["event"] == "pass" for event in events)
            if expected_error is None:
                require(error is None and passed, "fake success: " + str(error))
                require(sum(event["event"] == "query" for event in events) == 18, "all query routes")
                require(sum(event["event"] == "reimport" for event in events) == 2, "both reimports")
                require(not (Path(directory) / "attempt/source_snapshot").exists(), "source not moved")
            else:
                require(error is not None and expected_error in error and not passed,
                        "fault not caught at expected boundary: " + str(fault) + ": " + str(error))
        print("PASS_FAKE_CONTROL: " + (fault or "success-with-distinct-internal-IDs-and-index-state"))
    require("cofferdb" not in sys.modules, "real SDK imported")
    print("PASS_FAKE_CONTROL: 12 cases; not SDK/API compatibility or engine correctness evidence")


if len(sys.argv) > 1:
    if len(sys.argv) != 3 or sys.argv[1] != "--fake":
        raise SystemExit("usage: static_check.py [--fake LOCAL_REAL_CORPUS]")
    fake_checks(Path(sys.argv[2]).resolve())
