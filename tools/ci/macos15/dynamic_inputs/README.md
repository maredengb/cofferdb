# Mac15 observed inputs and exact plan derivation

These maintenance tools collect declared interpreter inputs and derive technical plans for the existing Mac15 helpers. They do not build, install, import, or test the SDK. The default action is `verify`, which does not probe the platform, create directories, or start processes.

The companion files are `dynamic_collect.py`, `dynamic_plan.py`, and `test_dynamic_inputs.py`. Use Python with `-I -S -B`. The collector verifies the exact companion module digest before compiling it. Explicit collection requires native Darwin 15 / arm64 and a matching package manifest; another macOS version stops before output creation or interpreter queries.

## Configuration and inputs

A canonical JSON configuration has exactly these fields: `schema_version` (integer 1), `kind` (`mac15_dynamic_input_config_1`), `source_repository` (canonical `maredengb/cofferdb` or its registered legacy alias `maredengb/coffered`), fixed `source_commit` (40 lowercase hexadecimal characters), `helper_sha256`, `actions`, `python_providers`, `selection_method`, `native_closure_gate`, `driver_contract_sha256`, `candidate_manifest_sha256`, and `native_closure_validator_sha256`.

These two repository names refer to the same repository; actual checkout evidence must independently bind repository ID 1422246026 and the exact source commit. No other repository is accepted. This configuration compatibility does not change the existing prepare_source.py origin check or its clean-HEAD/source-commit checks; a workflow may retain repository=maredengb/coffered to match that existing origin contract.

The fixed helper digests and action commits are declared in `dynamic_plan.py`. The five existing helpers are runner.py, probe.py, workflow_build.py, venv_prepare.py, and q_process_supervisor.py. This extraction does not change them. Checkout and upload-artifact action pins describe workflow dependencies; this code never invokes either action. A manifest verifies these four companion files. Its digest identifies technical bytes; it grants no execution capability.

Collection requires `--config`, `--helper-root`, and a new `--output` directory under an existing caller-owned 0700 parent. The helper root contains macos15/{runner,probe,workflow_build,venv_prepare}.py and q_process_supervisor.py. The output must not exist. All five exact helper digests are checked. Only the fixed supervisor is loaded, to capture bounded interpreter identity queries; no shell or arbitrary query command is accepted.

The deterministic provider priority is hostedtoolcache/Python, Python.framework/Versions, then the three fixed Homebrew Cellar python@3.12/3.13/3.14 roots. There is no download, setup-python call, Homebrew command, or PATH fallback. At most 128 names per provider and eight candidate interpreters per minor are considered. Identity queries choose the highest final micro version within the first usable provider priority, rejecting ambiguous equal versions. Micro versions are observed integers, not predicted hashes. CP312, CP313 and CP314 must be ordinary GIL-enabled native CPython interpreters.

Each binary is hashed before and after its fixed `-I -S -B -c` query. Raw stdout/stderr, actual process status, EOF and process-group cleanup are retained through the supervisor. Each query is limited to 20 seconds and 64 KiB per raw stream, with a two-second cleanup reserve; the overall collection has a cooperative 180-second deadline. Each declared stdlib is indexed without following directory or file links, with a 32 MiB file limit, 1 GiB per-tree byte limit, 24,000-entry limit and depth 64. File and directory metadata drift stops collection, including replacement of a provider ancestor. The final named root is reopened with NoFollow and compared to its initial signature. These observations are not an atomic snapshot against a malicious concurrent writer.

Outputs are INTENT.json, bounded query requests/captures, READONLY-INDEX.json, OBSERVATIONS.json and SNAPSHOT.json. The snapshot records actual image metadata when supplied by the runner environment, interpreter facts and file digests. Its source commit is the configured expectation; it is not evidence of a clean checkout or actual Git HEAD. The existing source preparation and checkout validation remain necessary.

## Plan derivation

`dynamic_plan.py instantiate` reads canonical `--config`, `--snapshot`, `--layout` and optional `--driver` / `--closure` JSON, verifies the package manifest, and returns canonical JSON. Duplicate keys, nonfinite numbers, boolean-as-integer version fields, unexpected fields, wrong source/config binding and altered helper pins are rejected. Inputs use bounded regular NoFollow files and their original canonical bytes.

The layout selects checkout, cargo_home, work_parent, evidence_parent and venv_parent. The three mutable parent roots must be pairwise disjoint and separate from the checkout, cargo home, helper files, observed interpreters, stdlibs and tool inputs. Build scripts must be exactly checkout/tools/prepare_source.py and checkout/tools/build_python.py, matching the existing workflow_build validator.

A complete independently produced native-closure record must bind the configured validator digest, exact snapshot digest and readonly index digest, with complete=true and an evidence reference. Supplying a JSON assertion does not establish its provenance or prove dependency completeness. The caller must validate the fixed validator and its actual evidence. A stdlib index alone cannot prove the native closure.

The external runtime driver contract has exact runtime_drivers (three entries), network_probe and command_seconds fields and must match its configured digest. This tool does not invent runtime assertions or execute that driver. Missing tool rows, validator identity, closure evidence or driver produce INPUT_MISSING and no plans.

When these inputs are present, the tool derives the existing three-ABI supervised build plan and three empty-venv plan. They include exact source, tool, interpreter and supervisor identities. Actual new empty-venv bindings and actual wheel/driver inputs do not exist before the preceding steps execute; runtime runner plans therefore remain INPUT_MISSING in this version. No SDK helper is invoked automatically.

## Required integration

This is an input-capture and initial-plan component, not a complete workflow. A workflow still needs exact tool observations, a fixed native-closure validator and evidence, fresh closure readback immediately before each SDK action, actual build/venv outputs, exact runtime-plan instantiation, and the external runtime driver. Existing runner file checks do not validate the complete stdlib index. No hook is added to the five helpers here; the integration must close that gap explicitly.

Preserve the existing Mac15 same-profile requirements: compiler execution and reads and network access must be rejected in the actual sandbox used by the runtime driver. A synthetic plan, directory hash or successful JSON parse does not establish these capabilities. The intended platform matrix is unchanged; this tool does not add a private build prerequisite or claim platform acceptance.

The local test file uses miniature filesystem fixtures and process/network tripwires. Its synthetic closure and driver data are not observed runner evidence. No platform collection or SDK action is part of these tests.
