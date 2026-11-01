# Mac15 build and runtime orchestration

`workflow_bridge.py` joins the existing observation, native dependency, build,
empty-venv, and runtime components. Each next input is produced by the actual
preceding function. The script never substitutes a prewritten success receipt
for a build, installation or platform check. Default `verify` reads no inputs,
installs no signal handler and executes nothing.

An explicit execution uses native macOS 15 arm64, a fixed clean GitHub checkout
and already prepared tools/Cargo inputs:

```sh
python3 -I -S -B workflow_bridge.py run \
  --helper-root /absolute/checkout/tools/ci/macos15 \
  --inputs /absolute/workflow-inputs.json \
  --inputs-sha256 ACTUAL_INPUT_SHA256
```

The input has exactly `schema_version: 1`,
`kind: "mac15_complete_workflow_inputs_1"`, `source_commit`, `repository_id`,
`checkout`, `cargo_home`, `selected_tools`, `driver_contract` and `output`.
`repository_id` is the existing repository's numeric ID 1422246026.
`source_commit` is the selected full public SHA. `driver_contract` is an exact
absolute-path/raw-SHA reference to the existing three-ABI driver template.
`selected_tools` supplies the seven explicit paths consumed by the tool observer.
The output must be new, separate from inputs, below an existing owned 0700
parent. The entry creates separate private data/work/evidence/venv/gate roots.

The job environment must identify the selected repository ID, source SHA and
actual run/attempt. These environment observations are not proof of a clean
checkout. The unchanged `prepare_source.py` command independently enforces its
origin, clean-HEAD and source-commit checks before the build. The retained
`coffered` repository spelling is compatible with that original origin contract.

The entry pins the adopted component bytes and original helper package. The
external driver contract binds its own installer, assertion driver, contexts
and fixture files. Its actual authorization, budget and fixture references must
be supplied by the selected execution; this code never invents them. Incorrect
template ordering or placeholders stop before interpreter collection.

The sequence is:

1. Collect the three declared ordinary CPython runtimes and complete stdlib
   indices using the existing bounded observer.
2. Observe the selected tool versions and retain original command receipts.
3. Inspect the declared native dependency graph and save the real amended
   snapshot, report and bound closure record.
4. Derive and validate the existing build and empty-venv plans. Run them through
   the fresh launch bridge, retaining its checks before each original capture.
5. Derive three runtime plans from actual build artifacts and actual new venv
   bindings. Run all three through the same bridge and unchanged sandbox runner.
   Its compiler execution/read and network-denial probes precede the external
   installation/business driver in the same sandbox profile.
6. Recheck the actual wheel bytes, original stdout/stderr/status seals, bound
   documents and component sources before and after saving the provisional result.

The build and venv result paths must be the original plans' outputs and their
returned results must match the exact original serialized bytes. Generated records
are hashed before writing. Collector and tool snapshots retain their producer
digests, and the build manifest must match the builder's originally sealed stdout.
These expected identities are never replaced by a later file hash. Runtime-plan derivation uses the existing
validator for actual source metadata, wheel/native digests and venv identities.
Original child receipts still require actual exit 0, EOF and process-group
cleanup. The orchestrator records eleven original build/venv/runtime captures;
this count describes the fixed call sequence, not acceptance coverage.

The in-process job can supply `retain_check` to retain the same final verifier.
It calls that verifier after its own source readback and final result write,
while the original parent timer remains active. The callback retains the
original document, stream and wheel digests; it does not reseal bytes or run
any build, installation or process again. A mismatch fails the enclosing job.

The parent has one 7200-second timer beginning before input/helper reads. The
unchanged stages retain their smaller original budgets: collection/tool/native
observation 180 seconds each, build 3900 seconds, venv preparation 300 seconds,
and each runtime 300 seconds. The parent limit does not extend a stage. The
one-shot POSIX alarm raises an exception and latches expiry; catching the alarm
inside a child supervisor cannot make a later stage run. The existing supervisor
still owns cleanup of the child process group it created. Timer restoration is
explicit and an existing process timer is never replaced. The entry requires
the main thread and rejects an inherited blocked SIGALRM without changing the
inherited signal mask.

This timer is a Python process limit, not a kernel guarantee for uninterruptible
IO or an abruptly terminated host. The invoking GitHub job must additionally
have a finite timeout and preserve/upload failure evidence. An interrupted job
without final child cleanup evidence remains incomplete. Cleanup after the
parent deadline is not additional product execution time, and a leftover
`RESULT.json` is provisional until the actual entry/job exits successfully.

First failure preserves the output prefix and writes HOLD when possible; no
existing directory is repaired, deleted, resumed or retried. Product acceptance
is always false in this technical result. The test owner still evaluates the
actual assertions, platform matrix, final packages and version-level evidence.

This entry does not acquire Rust, maturin, Python or Cargo registry/vendor
inputs. The final workflow must bind those inputs, the source/driver upload
set and execution scope before dispatch. It also needs checkout and artifact
actions pinned to the versions already declared by `dynamic_plan.py`. Creating
this local component does not publish source or execute a GitHub job.

Development controls use the actual initial-plan validators and derivation,
with miniature filesystem data and substituted collector/native/build/runtime
calls. They check ordering, returned-output binding, failure propagation,
first-read and final-write drift, original-stream tampering, timer expiry and
inherited blocked alarms. Process/network tripwires prevent
real execution. Synthetic wheel/receipt data is explicitly test-only:

```sh
python3 -I -S -B test_workflow_bridge.py --helper-root /absolute/tools/ci/macos15
```

These controls are not real builds, installs, SDK calls, Mac15 observations or
platform acceptance. The component is only one part of the final GitHub job.
