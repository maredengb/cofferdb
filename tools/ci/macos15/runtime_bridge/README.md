# Mac15 runtime plan bridge

`runtime_bridge.py` joins the results of the existing three-ABI build and empty-venv helpers to the existing sandbox runner. It does not build, install, execute an interpreter, import the SDK or driver, start a subprocess, access the network, or create output directories. Its default `verify` action performs no input reads. The explicit `instantiate` action reads only declared local inputs and prints three plans with their exact raw JSON digests.

This is one component of the Mac15 execution chain. It is **not an executable workflow** or a platform acceptance result. Native closure collection, fresh input checks at every actual SDK command boundary, tool acquisition, installer/driver behavior and workflow dispatch remain separate components. No successful JSON assertion supplies its own provenance.

## Existing helpers

Pass `--helper-root` as the directory containing the unchanged `runner.py`, `probe.py`, `workflow_build.py`, `venv_prepare.py` and `dynamic_inputs/`. The supervisor is its sibling `../q_process_supervisor.py`. Each loaded module is checked against its fixed digest before compilation. The bridge uses the existing dynamic plan derivation, build/venv validators, command constructors, capture predicates and runtime schema/profile; it does not relax their rules. The four-file dynamic input manifest is also checked.

## Bound input bundle

Use `instantiate --inputs INPUTS.json --inputs-sha256 SHA256 --helper-root MAC15_HELPERS`. The top-level object has exactly `schema_version: 1`, `kind: "mac15_runtime_bridge_inputs_1"`, and the eight fields below. Each is an exact `{ "path": ABSOLUTE_PATH, "sha256": RAW_FILE_SHA256 }` reference. The top-level digest and every referenced digest bind the original bytes, including whitespace. Duplicate keys, nonfinite JSON values, links, group/world-writable files, nonregular or oversized inputs and observed read-time changes are rejected. JSON inputs are limited to 2 MiB and other files to 128 MiB.

| Field | Original object |
| --- | --- |
| `config` | Fixed `mac15_dynamic_input_config_1` configuration |
| `snapshot` | Observed `observed_mac15_technical_inputs_1` snapshot, including tool identities and native files |
| `layout` | Existing checkout/cargo/work/evidence/venv parent layout |
| `driver_contract` | Exact three-ABI external driver templates, network probe and command budget |
| `native_closure` | Independently produced native closure record bound to the snapshot and readonly index |
| `build_result` | Original `workflow_build` `RESULT.json` |
| `build_manifest` | Original `build_python` `build-manifest.json` |
| `empty_venvs_result` | Original `venv_prepare` `RESULT.json` |

The existing initial plans are derived again from the first five objects. Results must be at those plans' exact output paths and bind their exact plan bytes. Every expected preparation/build/venv capture must be present, have integer exit 0, complete raw EOF and confirmed process-group cleanup, and bind the exact supervisor and command request. Partial or failed captures are rejected. These checks do not independently authenticate a caller-supplied receipt: the caller must preserve and verify the actual captures and their origin.

The build manifest must identify the same fixed clean public source, repository, builder and SDK/engine 0.1.1. Three ordered ordinary native CPython 3.12/3.13/3.14 records must match the observed micro versions. Each actual wheel is hashed again and its native member and source metadata are compared to the successful builder record and actual prepared source metadata. This reuses the builder's complete archive, license and native inspection; it is not a second independent build or a claim that an arbitrary archive is a valid wheel merely because its digest matches.

Each original empty-venv binding is checked against its plan, base interpreter and actual copy, `pyvenv.cfg`, owned 0700 root and still-empty `site-packages`. The selected invocation must still resolve inside that venv. Existing roots are never repaired. Runtime work/evidence destinations must not exist. Original helper output files are not modified.

## Driver template contract

The existing external driver contract has `runtime_drivers` (three ordered entries), `network_probe`, and `command_seconds`. Each entry has exactly `minor`, `path`, `sha256`, `arguments`, `inputs`, and `no_detach_reviewed: true`. `inputs` is a list of exact file rows. The caller supplies the reviewed installer/runtime driver and its supporting files; the bridge never imports it or adds business assertions. A driver that only tests an already installed package still needs the separate installation adapter supplied by the execution chain.

Only whole argument tokens are substituted:

- `{wheel}` and `{wheel_sha256}` are mandatory and select the actual matching artifact.
- `{native_sha256}`, `{source_metadata}`, `{source_input_sha256}`, `{sdk_version}`, and `{engine_version}` bind actual build identities.
- `{venv}` selects the actual new environment; `{smoke_work}` selects a new child of the runner's future work directory.

Unknown or embedded brace expressions are rejected. There is no shell interpolation. Literal arguments are retained. The actual wheel and prepared source metadata are added to the exact driver input list; a conflicting predeclared digest is rejected. Any additional pip wheel, fixture, adapter or assertion file must already be in the external template's input set. The bridge does not grant directory-wide reads for these files.

Runtime plans retain the existing 300-second whole limit and external command limit, exact compiler/sandbox/supervisor identities, and the unchanged renderer. Standard-library access uses the observed per-interpreter root. Non-system native dependencies must be present as the snapshot's exact `runtime_files`; no additional broad dependency roots are granted. The closure producer must therefore provide a complete file list compatible with this plan, or integration must stop.

## Result and freshness

`RUNTIME_PLANS_DERIVED_NOT_EXECUTED` contains three plan objects and digests, the source commit/tree and actual artifact/configuration observations. The result explicitly sets `sdk_executed: false`, `product_acceptance: false`, and `fresh_native_closure_gate_required_before_execution: true`. It is not a runtime capability, approval or acceptance certificate.

The bridge checks selected files and rereads the bound documents at its end. This is not an atomic snapshot against a concurrent writer. The execution integration must revalidate the complete relevant readonly index, tools, native closure, actual wheel/venv and selected driver immediately before each actual SDK helper command, and retain failures. It must then use the original runner and same-profile compiler execution/read and network denial probes. This component does not implement those missing launch hooks or claim they have run.

## Local controls

Run `python3 -I -S -B test_runtime_bridge.py --helper-root MAC15_HELPERS`. The controls use miniature files, synthetic capture records, fake native bytes, and process/network tripwires. The unchanged runner's system-file readback is stubbed; the new bridge's wheel/source/venv checks are exercised on real temporary files. No SDK, platform collection, sandbox, compiler, installation or real native loading occurs. These controls establish plan and rejection behavior only.
