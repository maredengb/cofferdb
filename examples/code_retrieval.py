"""Offline retrieval of pinned Rust source using an installed CofferDB wheel.

Run from a source distribution with a new database path. Bundled source and
precomputed vectors are checked before import; no model or provider is needed.
The retrieved snippets are data and are never executed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from cofferdb import CofferError, ClosedError, Database


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_bundle(fixtures):
    provenance = json.loads((fixtures / "provenance.json").read_text(encoding="utf-8"))
    for relative, expected in provenance["files_sha256"].items():
        assert hashlib.sha256((fixtures / relative).read_bytes()).hexdigest() == expected, relative
    documents = {row["external_id"]: row for row in read_jsonl(fixtures / "code/documents.jsonl")}
    chunks = {row["chunk_external_id"]: row for row in read_jsonl(fixtures / "code/chunks.jsonl")}
    for chunk in chunks.values():
        document = documents[chunk["document_external_id"]]
        source = fixtures / "source_snapshot/rust-by-example" / document["metadata"]["path"]
        span = chunk["source_span"]
        assert source.read_bytes()[span["start"]:span["end"]] == chunk["text"].encode("utf-8")
    query = json.loads((fixtures / "code-query.json").read_text(encoding="utf-8"))
    assert query["embedding"]["model_name"] == provenance["embedding"]["model"]
    assert query["embedding"]["dim"] == provenance["embedding"]["dimension"]
    return provenance, documents, chunks, query


def check_content(content, chunks, documents):
    chunk = content["chunk"]
    expected = chunks[chunk["chunk_external_id"]]
    document = documents[expected["document_external_id"]]
    assert chunk["text"].encode("utf-8") == expected["text"].encode("utf-8")
    assert chunk["metadata"] == expected["metadata"]
    assert chunk["source_span"] == expected["source_span"]
    assert chunk["document_id"] == content["document"]["document_id"]
    for field in ("external_id", "title", "source_uri", "metadata"):
        assert content["document"][field] == document[field]


def run(path, fixtures):
    exported_path = path.with_name(path.name + ".export")
    restored_path = path.with_name(path.name + ".restored.coffer")
    if any(candidate.exists() for candidate in (path, exported_path, restored_path)):
        raise FileExistsError("use a new database, export directory and restored database path")
    provenance, documents, chunks, query = load_bundle(fixtures)
    original_embedding = read_jsonl(fixtures / "code/embeddings.jsonl")[0]
    retained = {}
    runs = []
    with Database.create(str(path)) as database:
        imported = database.import_jsonl(str(fixtures / "code/import_manifest.json"))
        assert imported["documents_inserted"] == len(documents)
        assert imported["chunks_inserted"] == imported["embeddings_inserted"] == len(chunks)
        collection = database.collection("real_code")
        # Exercise explicit replace with the identical source vector. Text and
        # vector provenance stay paired; no embedding is regenerated here.
        replaced = collection.write_batch(embeddings=[{
            field: original_embedding[field]
            for field in ("chunk_external_id", "model_name", "dim", "vector")
        }], on_conflict="replace")
        assert replaced["embeddings_replaced"] == 1
        for mode, arguments in (
            ("keyword", {"text": query["text"]}),
            ("vector", {"embedding": query["embedding"]}),
            ("hybrid", {"text": query["text"], "embedding": query["embedding"]}),
        ):
            result = collection.search(**arguments, filter={"doc.lang": "rust"}, top_k=5)
            assert result["results"]
            assert result["profile"] == "balanced"
            assert result["execution_stats"]["fusion_candidate_count"] >= len(result["results"])
            for hit in result["results"]:
                assert hit["document_metadata"]["lang"] == "rust"
                assert hit["explain"]["final_score"] == hit["final_score"]
                assert "weights" in hit["explain"] and "candidate_sources" in hit["explain"]
                content = collection.get_chunk(hit["chunk_id"])
                check_content(content, chunks, documents)
                assert len(content["chunk"]["text"]) > len(hit["snippet"])
                retained[hit["chunk_external_id"]] = content
            runs.append({"mode": mode, "hits": len(result["results"]),
                         "first_id": result["results"][0]["chunk_external_id"]})
        try:
            collection.search(query["text"], top_k=True)
        except TypeError:
            pass
        else:
            raise AssertionError("bool must not be accepted as top_k")
        previous_commit = database.stats()["last_commit_id"]
        try:
            collection.write_batch(documents=[{"external_id": next(iter(documents))}])
        except CofferError as error:
            assert error.code == "duplicate_external_id"
            assert error.operation_outcome["state"] == "not_committed"
        else:
            raise AssertionError("duplicate document must be rejected")
        assert database.stats()["last_commit_id"] == previous_commit
        skipped = database.import_jsonl(str(fixtures / "code/import_manifest.json"), on_conflict="skip")
        assert skipped["documents_skipped"] == len(documents)
        assert skipped["chunks_skipped"] == skipped["embeddings_skipped"] == len(chunks)
        exported = collection.export_jsonl(str(exported_path))
        assert exported["documents"] == len(documents)
        assert exported["chunks"] == exported["embeddings"] == len(chunks)
    try:
        collection.stats()
    except ClosedError as error:
        assert error.sdk_reason == "closed"
    else:
        raise AssertionError("collection access must reject a closed owner")
    # Only the database is read below. The source snapshot and vectors are not
    # reopened; complete content and source coordinates come from stored data.
    with Database.open(str(path), read_only=True) as database:
        restored = database.collection("real_code")
        for external_id, expected in retained.items():
            content = restored.get_chunk_by_external_id(external_id)
            assert content == expected
            check_content(content, chunks, documents)
    # Restore uses only the supported exported manifest and JSONL files. No
    # original source snapshot, vector service or model is consulted here.
    with Database.create(str(restored_path)) as database:
        restored_import = database.import_jsonl(str(exported_path / exported["manifest"]))
        assert restored_import["documents_inserted"] == len(documents)
        assert restored_import["chunks_inserted"] == restored_import["embeddings_inserted"] == len(chunks)
        restored = database.collection("real_code")
        for external_id in retained:
            check_content(restored.get_chunk_by_external_id(external_id), chunks, documents)
        first = restored.get_chunk_by_external_id(next(iter(retained)))
        restored.delete_chunk(first["chunk"]["chunk_id"])
        try:
            restored.get_chunk(first["chunk"]["chunk_id"])
        except CofferError as error:
            assert error.code == "chunk_not_found"
        else:
            raise AssertionError("deleted chunk must not remain visible")
        restored.delete_document(first["document"]["document_id"])
        assert restored.stats()["document_count"] == len(documents) - 1
    return {"source_commit": provenance["source_commit"],
            "model": provenance["embedding"]["model"],
            "model_revision": provenance["embedding"]["revision"],
            "runs": runs, "complete_chunks_checked_after_reopen": len(retained),
            "import_warnings": imported["warnings"],
            "export_and_restore_verified": True, "deletion_verified_on_restored_copy": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a new database file")
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parent / "data/real_corpus")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.path, arguments.fixtures), indent=2))


if __name__ == "__main__":
    main()
