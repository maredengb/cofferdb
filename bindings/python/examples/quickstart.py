"""Run with an installed wheel; creates only files under the supplied new path."""

import argparse
from pathlib import Path

from cofferdb import Database


def run(path: str):
    with Database.create(path) as database:
        notes = database.create_collection({
            "name": "notes", "embedding_dim": 2, "distance_metric": "cosine",
        })
        summary = notes.write_batch(documents=[{"external_id": "doc", "title": "Field notes"}], chunks=[{
            "document_external_id": "doc", "chunk_external_id": "doc#1", "ordinal": 0,
            "text": "Jade leaves grow near the river.",
        }], embeddings=[{
            "chunk_external_id": "doc#1", "model_name": "sample-2d", "dim": 2,
            "vector": [1.0, 0.0],
        }])
        hits = notes.search("jade", embedding={"vector": [1.0, 0.0], "dim": 2,
                                                "model_name": "sample-2d"})
        assert summary["documents_inserted"] == summary["chunks_inserted"] == summary["embeddings_inserted"] == 1
        assert hits["profile"] == "balanced" and len(hits["results"]) == 1
        hit = hits["results"][0]
        assert hit["chunk_external_id"] == "doc#1"
        assert hit["explain"]["final_score"] == hit["final_score"]
        assert "weights" in hit["explain"] and "candidate_sources" in hit["explain"]
        content = notes.get_chunk(hit["chunk_id"])
        assert content["chunk"]["text"] == "Jade leaves grow near the river."
        assert content["document"]["title"] == "Field notes"
    assert content["chunk"]["chunk_external_id"] == "doc#1"
    with Database.open(path, read_only=True) as database:
        restored = database.collection("notes").get_chunk_by_external_id("doc#1")
        assert restored == content
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    content = run(str(args.path))
    print(content["chunk"]["text"])


if __name__ == "__main__":
    main()
