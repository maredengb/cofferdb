# Source, native wheel and reference tools

These tools are optional maintainer commands. Using the installed SDK does not
require them, Rust, maturin, Sphinx or a compiler. The tools do not upload files.
Choose new output directories outside the source tree; failed outputs and logs
are retained for diagnosis.

For supervised macOS 15 build and isolated runtime commands, see the
[CI helper guide](ci/README.md).

## Prepare the source and build native wheels

Use Python 3.12+, Rust with the selected target's linker, maturin **1.10.2**,
and the locked Cargo dependencies available locally. Select regular GIL builds
of CPython 3.12, 3.13 or 3.14 on native macOS arm64 or glibc Linux x86_64.
Free-threaded builds and cross-compilation are outside this command's scope.

From the product repository root:

```sh
python3.12 tools/prepare_source.py /tmp/cofferdb-prepared
python3.12 tools/build_python.py /tmp/cofferdb-wheels \
  --source /tmp/cofferdb-prepared \
  --python /absolute/path/to/python3.12 \
  --maturin /absolute/path/to/maturin
```

Repeat `--python` to select other supported interpreter minors. Interpreter
arguments are filesystem paths. The source preparer records a complete input
map, versions, locks and dependency notice identities. A versioned export can
add `--commit` only for that clean product repository's exact HEAD and origin;
an ordinary unversioned export keeps its Git identity null.
The builder rejects unlisted files and symbolic links in the prepared export,
including unlisted Cargo configuration. Keep caches and outputs outside it.

The builder uses the repository-root `pyproject.toml`, a separate Cargo target,
locked/offline Cargo resolution and the selected interpreter. It preserves
explicit compiler flags and adds source, Cargo and Rust-toolchain path remaps.
On macOS it sets deployment target `15.0` and removes build object path prefixes;
a conflicting deployment target is rejected. On Linux it requests
`manylinux_2_28` compatibility. Build in a suitable toolchain environment:
requesting a wheel tag does not make a newer native dependency compatible.

Each actual wheel is checked against its complete expected file set, including
Python modules, typing files, the selected interpreter's exact native module
path, product license and dependency notices. RECORD, internal tags, package
version and exact wheel filename, source bytes and selected absolute build paths
are checked too. macOS inspection
checks the actual thin arm64 dynamic-library header, macOS platform identifier,
minimum deployment version, system dependencies and absence of
runtime library search paths. Linux inspection uses `readelf` for x86_64, GLIBC
symbol requirements and runtime search paths; maturin performs its own
manylinux audit. An image's Mach-O `LC_ID_DYLIB` is its identity, not an external
runtime dependency.

The output includes raw build logs, a partial manifest after each inspected
wheel and a final build manifest with actual tool/interpreter versions, flags,
source identity and artifact hashes. The build context records detailed Rust
tool versions, native compiler driver/system linker inventory, the selected
macOS SDK and Cargo environment configuration file hashes. The binary's actual
Mach-O SDK version is recorded separately. Explicit linker selectors remain in
the recorded flags, environment and configuration; default tool inventory does
not prove an override used those default tools. These build records can contain local
paths; they are separate from the wheel. Use an external process timeout when
supervising long builds. An absent final manifest is not a complete build.

Static inspection and a successful build do not prove installation on macOS 15
or the declared minimum Linux distribution. Install and exercise the exact
artifact in each actual target environment. See the
[versioned installation guide](../docs/python-v0.1.1.md).

After building, exercise malformed source and artifact rejection paths without
installing a package or loading its native extension:

```sh
python3.12 -B tools/check_python_builder.py /tmp/cofferdb-builder-checks \
  --source /tmp/cofferdb-prepared \
  --wheel /tmp/cofferdb-wheels/cp312/cofferdb-0.1.1-cp312-cp312-macosx_15_0_arm64.whl \
  --python /absolute/path/to/python3.12
```

Select the actual matching wheel and interpreter for the host. The checker
retains malformed copies with rebuilt RECORD files, including individual
missing members and displaced native modules, plus source configuration,
symlink and mode controls. The original source and wheel remain unchanged.
On macOS the controls also inspect modified native headers, platform identifiers
and deployment-version fields without
loading or executing the modified native binaries.

## Observe process memory

```sh
python3.12 tools/memory_observer.py --trace-python
python3.12 tools/check_memory_observer.py
```

The first command observes its own process. The second touches and frees a
known 2 MiB allocation, checking that supported allocator counters observe the
change. It does not load CofferDB or run a database workload.

`ProcessMemoryObserver` can also be imported by benchmark scripts. Start
`tracemalloc` explicitly when Python heap observations are needed. Sample
outside the interval whose API latency is being measured, distinguish
instrumented runs, and record whether outputs are retained or released.

These observations overlap and must be reported separately:

- Python's `tracemalloc` reports traced Python allocations and the peak since
  its last reset. It is not an engine allocator or whole-process meter.
- Darwin's all-zone malloc statistics include other native and Python
  consumers. A high-water counter below current use is unavailable, with its
  raw value retained; a reported zero is not a measured zero engine peak.
- glibc's `malloc_info` reports root arena/free-chunk/mmap counters. The tool
  reports them individually and does not manufacture an exclusive Rust
  in-use total by subtraction.
- Current RSS and the process-lifetime RSS peak come from Mach on Darwin or
  `/proc/self/status` on Linux. They include memory beyond the allocator and
  can remain high after objects are freed. They are not per-call peaks.

Snapshots miss intervening peaks. The descriptor count is a raw snapshot that
can include the directory enumeration itself. Unsupported observations remain
null or explicitly unavailable. This tool does not prove bounded lifecycle
growth, attribute native bytes exclusively to Rust, or replace a matched
workload and its complete results.

## Build offline API references

See [reference preparation](../docs/reference.md) for the installed-package,
versioned Rust source and pinned documentation dependencies required by
`build_reference.py`. Its output is separate from compiler logs and caches.

## Prepare workload inputs and retain complete results

```sh
python3.12 tools/prepare_python_workload.py /tmp/cofferdb-workload-inputs
python3.12 -B tools/check_result_journal.py /tmp/cofferdb-journal-checks
```

The preparer creates deterministic 1k, 10k and 100k chunk inputs plus a separate
bounded vector-batch and long-body fixture. The 100k JSONL bytes reproduce the
existing public lifecycle reconstruction; smaller cases use document prefixes.
The ownership fixture has explicit required JSONL metadata, a distinct synthetic
model identity and a body of 1,277,952 UTF-8 bytes. Its query file and the scale
queries are hashed separately. Synthetic vectors exercise mechanics and provide
no evidence of semantic relevance. This command creates files only; it does not
open a database or run an evaluation.

`result_journal.ResultJournal` is an optional helper for a separately reviewed
workload. It times its callable, samples outside that interval, and retains each
complete JSON result in a verified content-addressed file. Reporting IO, hashing,
serialization and collection can affect later cache and memory observations;
they are excluded from the callable interval and must be declared in a protocol.
No conversion/execution sub-times are inferred from the total.

The journal keeps all core error fields, `operation_outcome`, `sdk_reason`, and
the cause/context/exception-group graph. It persists the error before propagating
it. Serialization failures leave partial payloads and an explicit evidence error;
they do not become successful calls. Started events and completed events are
flushed separately, so an incomplete invocation remains visible. Results retained
by the harness and their later release are distinct events. The retention flag
records an actual harness reference, including on failure paths. Results must be
JSON values; resource handles need separate lifecycle management by the workload.

A clock or observer failure after the callable returns still attempts to save
its complete result. A post-call tool error exposes `call_returned` and
`call_record`: the latter identifies any complete payload already written.
When a result could not be released safely, `call_result_available` is true and
`call_result` holds the exact value for the in-process supervisor. These fields
belong to this helper, not to the SDK. They do not infer a database commit state
or replay an operation. Preserve the original error and this context before
ending the worker. A tool failure makes the case incomplete even if an event
was written before a flush or later observation failed; a visible row alone
is not a successful run. Files and flushes do not provide a separate crash- or
power-loss recovery protocol.

The checker exercises full Unicode/numeric results, errors and commit progress,
exception groups/cycles, retention/release, unavailable observations and damaged
outputs without importing an SDK or database engine. Original outputs are kept.
These controls and prepared inputs do not constitute a frozen protocol or prove
database quality, performance or lifecycle stability.

## Execute a selected workload

`python_workload.py` consumes the files produced by the input preparer and an
explicit JSON plan. A matching plan hash identifies bytes; it does not establish
independent review or approval. Select and review the plan before an evaluation.
No measured pass threshold or speed ratio is supplied by these tools.

The plan uses schema version 1 and exactly these fields:

- `source_input_sha256`, `wheel_sha256`, `input_manifest_sha256`, `sdk_version`
  and `engine_version` identify the actual installed wheel and prepared inputs.
- `profiles`, `scales` and `modes` select nonempty subsets of the three supported
  resource profiles, the 1000/10000/100000 chunk inputs and `manifest`/`batch`.
- `rounds`, `query_repeats`, `warmup_repeats`, `batch_rows`, `retain_repeats`,
  `lifecycle_cycles`, `case_timeout_seconds` and `total_timeout_seconds` are
  positive integers. `lifecycle_scale` selects one declared scale.
- `trace_python` is a boolean. `tool_sha256` maps exactly
  `python_workload.py`, `result_journal.py` and `memory_observer.py` to their
  SHA-256 values. Freeze the actual tool bytes with the selected plan.

With the matching wheel already installed in the chosen environment:

```sh
python3.12 -B tools/python_workload.py \
  --protocol /tmp/selected-plan.json --protocol-sha256 SHA256_OF_SELECTED_PLAN \
  --inputs /tmp/cofferdb-workload-inputs --wheel /absolute/path/to/actual.whl \
  --output /tmp/cofferdb-python-run
python3.12 -B tools/check_python_workload.py /tmp/cofferdb-workload-controls
```

Use a new output directory outside both source and input trees. The supervisor
starts a fresh process for each case, preserves both output streams and stops
at the first failure or timeout without retrying. Exit zero alone is insufficient:
the supervisor checks matching completed-call events and every full payload.
Every completed call, expected refusal and recorded pre-refusal state must carry
a valid content-addressed reference with the matching result/error envelope.
One missing result makes a case incomplete, even when other results remain.
An interrupted or incomplete run retains its partial manifest and active case.
The checker uses a fake SDK and small files; it does not import or exercise an
engine, certify a platform or run the selected database evaluation.

Before terminating on failure, the worker also attempts to stream a
`COFFERDB_WORKLOAD_FAILURE_JSON` record to the separately captured stderr,
including the complete error graph and known returned values. This preserves
available outcomes when the payload directory fails. The original exception
still propagates; the call is not replayed. If both result storage and the
reporting channel fail, complete evidence cannot be promised: the case remains
incomplete, and a missing result does not imply that a write was uncommitted.

The worker exercises import readiness, reopened/warm readers, retained-result
release, duplicate-refusal cleanup and owned long content after closing. Counts,
complete results, full errors and confirmed write progress remain in the journal.
Synthetic inputs do not measure semantic quality. Lifecycle observations and
successful cases still require the selected analysis and acceptance criteria.
OS page cache is uncontrolled; a reopened handle is not a cold-cache claim.

The optional Rust driver builds from the same product source without additional
dependencies or an installed Python package:

```sh
cargo build --locked --offline --example workload-driver
cargo test --locked --offline --example workload-driver -- --test-threads=1
```

Its executable is `target/debug/examples/workload-driver` unless a separate Cargo
target directory was selected. It accepts these argument forms:

```text
write DATABASE INPUT_CASE QUERIES OUTPUT PROFILE MODE BATCH_ROWS
DATABASE COLLECTION QUERIES OUTPUT PROFILE WARMUPS REPEATS
content DATABASE COLLECTION CHUNK_EXTERNAL_ID
```

The first form creates a new database and executes import-readiness queries;
the second reopens a read-only handle per query and retains all new/warm results;
the third closes the database before serializing a bounded owned full-content
result. `QUERIES` is a JSON array of requests with `id` and `profile` plus the
ordinary search fields. The write fixture names exactly the three local JSONL
files in `import_manifest.json` with explicit dimension, metric and normalization.
Use positive batch/repetition counts, new external database/output paths, and
an external process timeout. Capture stderr even if result storage fails.
Index the complete payloads and their hashes after the process exits.

The Rust batch baseline explicitly uses the existing doc-hidden host-binding
`Operation` adapter, not an idiomatic application batch API. Its API interval
includes record conversion and the owned typed result, while adapter setup,
JSON reporting, observations and result IO are outside. Python API intervals
include Python result construction. Ready-wall intervals include database
creation, import and readiness queries with reporting, but exclude initial
fixture preparation. These intervals are not a reliable decomposition of
conversion, engine and result costs. Do not infer missing components by subtraction.

Rust counters and nanoseconds use exact decimal strings in tool evidence.
Native allocator/RSS snapshots overlap and do not attribute bytes exclusively
to the engine. The optional compile-time `COFFERDB_CORE_SOURCE_INPUT_SHA256`
appears only as a declared build hint, or null: it does not verify the compiled
source. Record the actual clean source, lock, tool/binary hashes and build log
separately. `COFFERDB_TOOL_FUNCTIONAL_CONTROL_ONLY=1` disables Rust clocks and
observations for a tiny functional control; such output is explicitly labelled
and must not be used as performance evidence. Neither driver uploads results
or turns compilation, mock controls or a successful case into release acceptance.

## Observe host progress during an active SDK owner

`gil_probe.py` provides `observe_call` for a separately selected and reviewed
workload. It does not import the SDK or choose, run or approve a database
experiment. `check_gil_probe.py` runs only fake-owner controls:

```sh
python3.12 -B tools/check_gil_probe.py
```

Pass an actual operation, a same-owner `stats` probe, the SDK `BusyError` class,
an independent-owner probe and a `ResultJournal`. The function returns the
completed journal call record (whose payload contains the full result) and an
observation. The independent probe's full return is included in the observation.
Only busy windows that enclose actual Python progress and independent-owner
work before the operation finishes qualify. Start events and successful polls
alone never qualify. A short operation with no qualifying window is
`inconclusive_no_active_window`; it is neither a demonstrated GIL failure nor
proof of release. These owner observations do not identify exclusive native
phases or establish a response-time SLA. Input conversion and result projection
can still hold the GIL.

Observers sleep 1 ms by default and retain at most 4096 samples. Reaching the
sample bound makes the attempt incomplete rather than silently dropping proof.
Observer failure propagates after preserving the completed operation result;
an original operation error also propagates, without retry. Polling can race
owner acquisition: a main-call `BusyError` is a failed attempt, not permission
to replay a write. Callbacks must not write to the journal. Use fresh owners
and outputs per attempt and an external process timeout. A bounded join cannot
terminate a blocked Python thread; on observer timeout stop the process and do
not reuse the owners. No default scale or search/import/export matrix is
implied. Freeze actual workloads, callback identities, source/package hashes,
platforms, timeouts and analysis before formal use.

## Supervise paired workload groups

`paired_workload.py` adds an outer supervisor for the existing Python worker and
Rust `workload-driver`. It imports no SDK in the parent process. Prepare and
review both a schema-1 Python workload plan and a separate pair envelope. A
matching hash identifies bytes; it is not protocol approval or execution
permission. The pair envelope has exactly these fields:

- `schema_version`: integer `1`; `plan_sha256`: the original workload plan hash.
- `global_budget_seconds` and `language_budget_seconds`: positive cumulative
  budgets. The language budget must equal the workload plan's total budget.
- `python_sha256`, `rust_sha256`, `environment_sha256`: selected executable and
  environment-file hashes. Preserve the Python executable's invocation path
  inside its virtual environment; resolving that symlink would lose the venv.
- `tool_sha256`: the exact `paired_workload.py`, `python_workload.py`,
  `result_journal.py` and `memory_observer.py` hashes.
- `rust_build_receipt`: `binary_sha256`, `source_input_sha256`, `engine_version`
  and `build_evidence_sha256`, backed by the separately selected build-evidence
  file. This declaration and its evidence need independent provenance review;
  matching declared hashes alone do not attest to a compiler invocation.

The environment JSON fixes `system`, `release`, `machine` and a `python` object
with the complete `version` string, `implementation`, `soabi`, pointer `bits`,
`gil_disabled` and `gil_enabled`. The supervisor checks the actual host and runs
a bounded, SDK-free identity probe through the selected interpreter. Workers
then independently verify the installed package against the actual wheel.
Freeze additional image, hardware, resource, dependency and build identities in
the enclosing evaluation protocol. The environment object is not a complete
platform qualification by itself.

```sh
python3.12 -B tools/paired_workload.py \
  --protocol /tmp/selected-plan.json --protocol-sha256 PLAN_SHA256 \
  --envelope /tmp/pair-envelope.json --envelope-sha256 ENVELOPE_SHA256 \
  --inputs /tmp/workload-inputs --wheel /tmp/artifacts/selected.whl \
  --python /tmp/selected-venv/bin/python --rust /tmp/selected-workload-driver \
  --environment /tmp/selected-environment.json \
  --build-evidence /tmp/selected-rust-build.json --output /tmp/new-paired-attempt
python3.12 -B tools/check_paired_workload.py
```

The checker runs fake workers, synthetic journal controls and small real child
processes; it never imports or calls a database. It is not workload acceptance.
The supervisor uses external round 1 Python/Rust, round 2 Rust/Python, alternating
thereafter. Each language completes the listed profile/scale/mode group before
switching; these are not adjacent per-case pairs. Derived Python subplans retain
all selected profiles/scales/modes and set only `rounds=1`; external and internal
round numbers are recorded separately. Python-only lifecycle, retained results
and ownership work remain separate from the Rust write/query process coverage.
Do not compare different stage totals as pure binding overhead.

One overall monotonic clock and each language's accumulated elapsed time survive
all subplans. Each Rust logical unit shares one deadline across write, query,
validation and cleanup. Remaining global/language/unit time bounds each child.
The first process, evidence or budget failure stops the entire schedule; later
units remain `Not Run`. There is no retry or resume-in-place. Preserve failed
attempts, all complete payloads and original streams before separately approving
a new attempt. A zero child exit does not establish completion: the supervisor
checks actual call inventories, payload envelopes and import counts. It hashes
Rust payloads without rewriting the original files. It does not calculate score
equivalence, quality metrics or an independent scoring oracle.

Only the fresh POSIX process group created for each invocation receives cleanup
signals. Timeout, cancellation and leftover descendants are failures. Detached
processes that deliberately escape that group are outside this mechanism; the
selected workers must not daemonize. Cleanup can need bounded additional time
after a deadline and never authorizes the next workload. Input hashes and frozen
artifacts/tools are checked around each unit; these reads warm OS caches and
are supervision overhead, outside the reported API/ready-wall intervals. No
cold-OS-cache or exclusive memory attribution is implied. Loader injection and
functional-control overrides are rejected. Outputs must be new and external to
inputs, source and the selected wheel directory. Final execution completeness
is separate from protocol approval, numerical comparison and product acceptance.

Executable invocation paths and their resolved targets are both bound, including
virtual-environment symlinks; a target or byte change stops the attempt. The
final `complete` flag is provisional until this supervisor exits zero after
checking receipt IO against the same remaining deadline. Retain the final
`SUPERVISOR_EXIT_RECEIPT` stdout record and process exit as well as `run.json`.
Failure to commit the final receipt revokes completion; actual child returns
remain available even when a later clock, receipt or cleanup fails. Final
closeout accounting is emitted in the exit receipt rather than silently
restarting a budget for reporting.
