# Fresh inputs at original macOS helper launch boundaries

`launch_bridge.py` connects the declared-native-graph fresh check to the
unchanged macOS build, empty-venv, and runtime helpers. It loads their fixed
source bytes into isolated maintenance modules and wraps only the returned
supervisor's `capture` call. The original plan validators, host gate, raw-output
capture, sandbox profile, denial probes, exit checks, and cleanup logic remain
in use. No original helper file or global installed module is modified.

The default `verify` action only prints `LAUNCH_BRIDGE_NOT_EXECUTED`. It does not
read inputs, import maintenance helpers, probe a platform, create directories,
or start any process.

## One explicit step

An execution selects one of `build`, `venvs`, or `runtime`, an exact original
plan and its raw-byte SHA-256, and the same plan digest in the existing explicit
execution-binding argument. That digest binds the command's data; it does not
create authorization or replace the caller's execution decision.

```sh
python3 -I -S -B launch_bridge.py run \
  --step runtime \
  --plan /absolute/runtime-plan.json --plan-sha256 ACTUAL_PLAN_SHA256 \
  --approved-plan-sha256 ACTUAL_PLAN_SHA256 \
  --fresh-inputs /absolute/fresh-inputs.json \
  --fresh-inputs-sha256 ACTUAL_FRESH_BUNDLE_SHA256 \
  --gate-evidence /absolute/new-fresh-evidence \
  --helper-root /absolute/tools/ci/macos15 \
  --runtime-bridge /absolute/runtime_bridge.py \
  --native-validator /absolute/native_closure.py
```

The fresh bundle contains exactly `snapshot`, `index`, and `report`, each an
object with an absolute `path` and actual raw-byte `sha256`. These are the
outputs of actual input/native observation, not constructed success records.
The native graph source and runtime bridge source are pinned by this entry's
constants. All original helper digests are also rechecked.

The caller supplies a new evidence path under an existing owned `0700` parent.
It must be separate from source/maintenance inputs and the original helper's
mutable directories. Fresh input documents may not be placed inside a directory
the runtime can write. No existing output directory is repaired or reused.

The bridge checks the complete readonly/native/tool inputs at entry, **before
each actual supervisor capture**, and after successful completion. It rereads
the digest-bound input documents around each observation. Before checking that
graph, it binds the selected plan to the snapshot's actual helper paths, base
interpreters, source commit/build tools, or runtime standard library, exact
native files and selected sandbox/compiler, as appropriate for that step.
A different valid plan cannot borrow an unrelated fresh observation. The original build
loop makes two captures, empty-venv loop three, and runtime loop two; extra or
missing captures and repeated supervisor loading are errors. Runtime probes and
the install/smoke driver therefore retain their common original sandbox profile
and each receive a fresh gate. A failed gate or failed evidence write prevents
that child launch. A later failure retains earlier output and does not retry.

Gate work consumes the original deadlines: the wrapper forwards the exact
deadline and capture arguments rather than extending them. It checks remaining
time before and after observation. The enclosing workflow must retain bounded
supervision of the entire entry as well: a filesystem read or a fresh check
already in progress is not an independently cancellable job. The original
helper's timeout, group cleanup, EOF, and raw-output rules still apply to its
children. This bridge does not certify termination of an interrupted parent.

The new evidence contains an intent, one `launch-NN.json` observation for each
entered capture, and a final result or failure record. Each observation binds
the original request digest and deadline. Original raw output and command
receipts remain in their original evidence directory; the bridge result refers
to that original helper result. If a capture raises after entry, the failure's
`captures_entered` is not a completed-command count.

The boundary check is a fresh observation, not an atomic filesystem snapshot.
It covers the exact declared interpreter/library/tool set and standard-library
index. It does not claim arbitrary `dlopen` completeness. Actual installation,
native loading, fixed driver assertions, same-profile compiler/network denial,
and platform behavior must still be observed on the selected host.

## Development controls

```sh
python3 -I -S -B test_launch_bridge.py \
  --helper-root /absolute/tools/ci/macos15 \
  --runtime-bridge /absolute/runtime_bridge.py \
  --native-validator /absolute/native_closure.py
```

Controls exercise ordering, unchanged deadline forwarding, expired budgets,
failed gates and records, extra launch rejection, loader restoration, and the
original build helper's two-command loop. The original-loop controls replace
the host observation, native fresh check, and supervisor capture with fixtures;
they use tiny files and never execute either build command. Process spawning
and network connections have tripwires. These results test the new connection,
not a real build, installation, platform, or product acceptance.

The tool-version observer, artifact installer/business-driver connection, and
outer workflow supply the other parts of a complete run. Having this entry or
a successful component control does not establish those parts or close any
platform acceptance item. Results always retain `product_acceptance: false`.
