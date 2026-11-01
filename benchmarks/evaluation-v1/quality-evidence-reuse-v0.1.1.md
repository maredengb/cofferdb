# Quality evidence for v0.1.1

The archived [representative study](representative-quality-report.md) provides
relevance evidence for the unchanged default retrieval strategy on its recorded
inputs. Reusing that evidence does not establish the quality of a new Python
wheel, a different platform or an application dataset.

## Inputs and results

The supplied passages, queries, vectors and relevance judgments are documented in
the [dataset](public-quality-v1/dataset_manifest.json),
[source provenance](public-quality-v1/source-provenance.json) and
[embedding provenance](public-quality-v1/embedding-provenance.json). The inputs
are authored evaluation material. Replay uses supplied vectors and needs no inference service.
Source and model licenses are retained with the materials.

The study compares keyword, exact-vector and hybrid retrieval with unchanged
default settings. Hybrid matched keyword retrieval on overall Recall@10, but had
lower MRR@10. Individual queries also favored different modes. Hybrid and larger
resource presets therefore do not guarantee better relevance.
The [full results](representative-quality-v1/v2/quality-report.json) retain the
per-query measurements, comparisons and methodology. Repeated timing samples do
not increase the number of independent relevance tasks. SQLite's tokenizer,
filtering and scoring differ; its results are not a matched-work speed ratio.

## Applicability

Source review found no change to default scoring, candidate selection, tokenizer
or fixed-input semantics requiring a new historical relevance evaluation for
this release. A separate Rust/Python comparison checked selected tasks and
configurations; it does not replace the full study or establish behavior for
every input combination, missing embeddings, other platforms or the final wheel.

Historical latency, memory and work measurements describe their original source
and environment. They are not measurements of the current Python package or
proof of its platform, scale, lifecycle or threading behavior.

Consult the [evaluation archive](README.md) for the studies and their distinct
scopes, and the [replay guide](representative-replay.md) for reproduction details.
Results from different datasets, vectors or judgments are not an improvement
comparison. Changes to scoring, candidate selection, tokenizer, defaults or
evaluation inputs require a new assessment. No universal retrieval-quality or
whole-release acceptance claim is made.
