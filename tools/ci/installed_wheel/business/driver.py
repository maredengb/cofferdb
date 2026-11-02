"""Candidate installed-final-wheel F-SMOKE business checks; no installation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=True, allow_nan=False), flush=True)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def fixture(bundle):
    provenance_bytes = (bundle / "provenance.json").read_bytes()
    require(hashlib.sha256(provenance_bytes).hexdigest() ==
            "c8100b36276b4fff07cee483f4e1331e2c8498eda2e4b6624cbe4e837ef6a5e2", "pinned provenance")
    provenance = json.loads(provenance_bytes)
    names = ["code/" + name for name in ("documents.jsonl", "chunks.jsonl", "embeddings.jsonl")]
    names += [name for name in provenance["files_sha256"] if name.startswith("source_snapshot/")]
    original = {}
    for name in names:
        data = (bundle / name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == provenance["files_sha256"][name], name)
        original[name] = data
    documents, chunks, embeddings = [[json.loads(line) for line in original["code/" + name].splitlines()]
                                      for name in ("documents.jsonl", "chunks.jsonl", "embeddings.jsonl")]
    require((len(documents), len(chunks), len(embeddings)) == (2, 5, 5), "pinned code counts")
    documents.append({"external_id": "empty", "title": "Empty source catalogue entry",
                      "metadata": {"empty": True}})
    parents = {item["external_id"]: item for item in documents}
    for chunk, embedding in zip(chunks, embeddings):
        source = original[parents[chunk["document_external_id"]]["source_uri"]]
        span = chunk["source_span"]
        require(source[span["start"]:span["end"]] == chunk["text"].encode("utf-8"), "source slice")
        require(embedding["chunk_external_id"] == chunk["chunk_external_id"], "vector/text pairing")
        require(embedding["model_name"] == "BAAI/bge-m3" and embedding["dim"] == 1024
                and len(embedding["vector"]) == 1024, "pinned embedding definition")
    manifest = {"schema_version": "0.1", "dataset_id": "final-package-f-smoke-v1",
                "collection": "notes", "embedding_dim": 1024, "embedding_model": "BAAI/bge-m3",
                "vectors_normalized": False, "distance_metric": "cosine", "documents": "documents.jsonl",
                "chunks": "chunks.jsonl", "embeddings": "embeddings.jsonl"}
    files = {"import_manifest.json": encoded(manifest),
             "documents.jsonl": b"".join(map(encoded, documents)),
             "chunks.jsonl": b"".join(map(encoded, chunks)),
             "embeddings.jsonl": b"".join(map(encoded, embeddings)),
             "provenance.json": provenance_bytes}
    files.update({name: data for name, data in original.items() if name.startswith("source_snapshot/")})
    return documents, chunks, embeddings, files


def hashes(files):
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def validate_contract(contract, files):
    require(contract["schema_version"] == 1, "contract schema")
    for key in ("attempt_id", "cell_id", "authorization_reference", "preflight_evidence",
                "sdk_version", "engine_version", "budget_reference"):
        require(isinstance(contract[key], str) and contract[key].strip(), key)
    require(contract["files_sha256"] == hashes(files), "frozen fixture bytes differ")
    # These are caller attestations, not a replacement for the preflight tool.
    for key in ("wheel", "native", "dist_info", "source"):
        item = contract["identity"][key]
        require(isinstance(item["reference"], str) and item["reference"].strip(), key + " reference")
        expected = item["expected_sha256"]
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in "0123456789abcdef" for c in expected), key + " digest")
        require(item["observed_sha256"] == expected, key + " preflight mismatch")


def content_check(collection, documents, chunks, label):
    contents = {}
    parents = {document["external_id"]: document for document in documents}
    for expected in documents:
        actual = collection.get_document_by_external_id(expected["external_id"])
        emit("document", label=label, result=actual)
        for key, value in expected.items():
            require(actual[key] == value, label + " document " + key)
    for expected in chunks:
        actual = collection.get_chunk_by_external_id(expected["chunk_external_id"])
        emit("content", label=label, result=actual)
        for key in ("chunk_external_id", "ordinal", "text", "metadata", "source_span"):
            require(actual["chunk"][key] == expected[key], label + " chunk " + key)
        require(actual["document"]["external_id"] == expected["document_external_id"], "parent")
        for key, value in parents[expected["document_external_id"]].items():
            require(actual["document"][key] == value, label + " content document " + key)
        require(actual["chunk"]["document_id"] == actual["document"]["document_id"], "parent id")
        require(collection.get_chunk(actual["chunk"]["chunk_id"]) == actual, "get by id")
        contents[expected["chunk_external_id"]] = actual
    return contents


def query_check(collection, label, chunks, embeddings):
    embedding = {key: embeddings[0][key] for key in ("vector", "dim", "model_name")}
    first_id = chunks[0]["chunk_external_id"]
    signatures = {}
    for mode, kwargs in (("bm25", {"text": "pangram"}),
                         ("vector", {"embedding": embedding}),
                         ("hybrid", {"text": "pangram", "embedding": embedding})):
        result = collection.search(**kwargs, top_k=5, profile="balanced",
                                   weights={"bm25": 0.5, "vector": 0.5, "metadata": 0.0})
        emit("query", label=label, mode=mode, result=result)
        require(result["profile"] == "balanced", "profile")
        hits = result["results"]
        require(bool(hits) and hits[0]["chunk_external_id"] == first_id, mode + " first hit")
        require(len({hit["chunk_external_id"] for hit in hits}) == len(hits), "duplicate hit")
        for hit in hits:
            require(hit["chunk_external_id"] in {item["chunk_external_id"] for item in chunks}, "unknown hit")
            require(hit["explain"]["final_score"] == hit["final_score"], "explain score")
            require("weights" in hit["explain"] and "candidate_sources" in hit["explain"], "explain fields")
            content = collection.get_chunk(hit["chunk_id"])
            require(hit["chunk_id"] == content["chunk"]["chunk_id"], "hit/get chunk id")
            require(hit["chunk_external_id"] == content["chunk"]["chunk_external_id"],
                    "hit/get chunk external id")
            require(hit["document_id"] == content["chunk"]["document_id"] ==
                    content["document"]["document_id"], "hit/get document id")
            require(hit["document_external_id"] == content["document"]["external_id"],
                    "hit/get document external id")
            require(hit["explain"]["chunk_id"] == content["chunk"]["chunk_id"], "explain chunk id")
            require(hit["explain"]["document_id"] == content["document"]["document_id"], "explain document id")
            require(hit["explain"]["chunk_external_id"] == hit["chunk_external_id"], "explain external id")
            require(hit["explain"]["document_external_id"] == content["document"]["external_id"],
                    "explain parent external id")
        full = collection.get_chunk(hits[0]["chunk_id"])
        require(len(full["chunk"]["text"]) > len(hits[0]["snippet"]), "body must exceed actual snippet")
        # Never compare database-local IDs or mutable index lifecycle diagnostics.
        fields = ("schema_version", "chunk_external_id", "document_external_id", "collection",
                  "final_score", "scores", "weights", "normalization", "matched_terms", "bm25",
                  "matched_term_details", "tokenizer", "fusion", "candidate_sources", "resource_profile",
                  "candidate_limits", "distance_metric", "vector_raw_kind", "normalization_method",
                  "filter", "metadata_boost_context", "chunk_boundary", "source")
        signatures[mode] = [(hit["chunk_external_id"], hit["final_score"],
                             {key: hit["explain"][key] for key in fields}) for hit in hits]
    return signatures


def run(Database, ClosedError, work, documents, chunks, embeddings, files):
    work.mkdir(parents=False, exist_ok=False)
    for name, data in files.items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    baseline_queries = None
    retained = {}
    for route in ("manifest", "batch"):
        path = work / (route + ".coffer")
        with Database.create(str(path)) as database:
            if route == "manifest":
                summary = database.import_jsonl(str(work / "import_manifest.json"))
                collection = database.collection("notes")
            else:
                collection = database.create_collection({"name": "notes", "embedding_dim": 1024,
                                                         "embedding_model": "BAAI/bge-m3",
                                                         "vectors_normalized": False,
                                                         "distance_metric": "cosine"})
                summary = collection.write_batch(documents=documents, chunks=chunks, embeddings=embeddings)
            emit("write", route=route, result=summary)
            for key, count in zip(("documents_inserted", "chunks_inserted", "embeddings_inserted"),
                                  (len(documents), len(chunks), len(embeddings))):
                require(summary[key] == count, route + " " + key)
            retained[route] = content_check(collection, documents, chunks, route)
            queries = query_check(collection, route, chunks, embeddings)
            if baseline_queries is None:
                baseline_queries = queries
            else:
                require(queries == baseline_queries, "manifest/batch query equivalence")
            output = work / (route + "-export")
            exported = collection.export_jsonl(str(output))
            emit("export", route=route, result=exported)
            for key, count in zip(("documents", "chunks", "embeddings"),
                                  (len(documents), len(chunks), len(embeddings))):
                require(exported[key] == count, route + " export " + key)
        try:
            collection.stats()
        except ClosedError:
            emit("closed", route=route)
        else:
            raise AssertionError("collection remains usable after close")
    (work / "source_snapshot").rename(work / "moved-sources")
    require(not (work / "source_snapshot").exists(), "original source remains")
    emit("source_moved", original_exists=False)
    for route in ("manifest", "batch"):
        with Database.open(str(work / (route + ".coffer")), read_only=True) as database:
            collection = database.collection("notes")
            require(content_check(collection, documents, chunks, route + "-reopen") == retained[route],
                    "reopen full content differs")
            require(query_check(collection, route + "-reopen", chunks, embeddings) == baseline_queries,
                    "reopen queries differ")
        with Database.create(str(work / (route + "-restored.coffer"))) as database:
            summary = database.import_jsonl(str(work / (route + "-export/export_manifest.json")))
            emit("reimport", route=route, result=summary)
            for key, count in zip(("documents_inserted", "chunks_inserted", "embeddings_inserted"),
                                  (len(documents), len(chunks), len(embeddings))):
                require(summary[key] == count, "reimport " + key)
            collection = database.collection("notes")
            content_check(collection, documents, chunks, route + "-restored")
            require(query_check(collection, route + "-restored", chunks, embeddings) == baseline_queries,
                    "export query equivalence")
    emit("pass", scope="F-SMOKE business candidate only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-hashes", action="store_true")
    parser.add_argument("--fixtures", required=True, type=Path, help="pinned local real_corpus bundle")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    try:
        documents, chunks, embeddings, files = fixture(args.fixtures.resolve())
    except Exception:
        traceback.print_exc()
        emit("input_rejected")
        return 2
    if args.fixture_hashes:
        emit("fixture_hashes", files_sha256=hashes(files))
        return 0
    try:
        require(args.contract is not None and args.work is not None and args.contract_sha256,
                "contract, contract-sha256 and work required")
        raw = args.contract.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == args.contract_sha256, "contract SHA mismatch")
        contract = json.loads(raw)
        validate_contract(contract, files)
        require(not args.work.exists() and args.work.parent.is_dir(), "work must be new with existing parent")
        emit("input", contract_sha256=args.contract_sha256, contract=contract)
    except Exception:
        traceback.print_exc()
        emit("input_rejected")
        return 2
    try:
        # Import is deliberately after the frozen caller preflight contract.
        import cofferdb
        from cofferdb import ClosedError, Database
        require(cofferdb.__version__ == contract["sdk_version"], "SDK version")
        require(cofferdb.engine_version == contract["engine_version"], "engine version")
        emit("loaded", module_file=cofferdb.__file__, sdk_version=cofferdb.__version__,
             engine_version=cofferdb.engine_version)
        run(Database, ClosedError, args.work.resolve(), documents, chunks, embeddings, files)
        return 0
    except Exception:
        traceback.print_exc()
        emit("fail", scope="F-SMOKE business candidate only")
        return 1


if __name__ == "__main__":
    sys.exit(main())
