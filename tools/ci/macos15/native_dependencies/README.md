# macOS 15 declared native dependency inspection

`native_closure.py` reads the ordinary arm64 slice of Mach-O executables,
libraries, and extension bundles without executing or loading them. It derives
the non-system library inputs needed by the existing macOS 15 runtime sandbox.
It also checks that those inputs and the complete selected standard-library
file sets have not changed before a later command.

The default command is inert:

```sh
python3 -I -S -B native_closure.py
```

It does not inspect the host, read an input bundle, import a product, create an
output, install software, or launch a child process. Explicit `inspect` and
`fresh` commands require a real Darwin arm64 host running macOS 15, with the
same OS version as the bound observation snapshot.

## Inputs and results

The `--helper-root` argument identifies the existing `dynamic_inputs` directory.
The collector and plan modules are checked against their fixed SHA-256 values
before their source is compiled. Their bounded file reader and complete
standard-library index are reused. This module does not replace the original
build, virtual-environment, sandbox, process-supervision, or plan validators.

An `inspect` bundle is a JSON object with exactly these keys:

```json
{
  "config": {"path": "/absolute/config.json", "sha256": "<64 hex digits>"},
  "snapshot": {"path": "/absolute/snapshot.json", "sha256": "<64 hex digits>"},
  "index": {"path": "/absolute/readonly-index.json", "sha256": "<64 hex digits>"},
  "evidence_reference": "/absolute/native-report.json"
}
```

Run it explicitly with the bundle's actual raw-byte SHA-256:

```sh
python3 -I -S -B native_closure.py inspect \
  --helper-root /absolute/tools/ci/macos15/dynamic_inputs \
  --inputs /absolute/inspect-inputs.json \
  --inputs-sha256 ACTUAL_BUNDLE_SHA256
```

All referenced documents must match their actual raw-byte hashes. The config
must pin this validator's exact source SHA. The snapshot and index retain the
original ordered CPython 3.12, 3.13, and 3.14 identities and source/config
bindings. Each complete standard-library tree is reindexed before its native
graph is examined; additional files are changes too.

Successful inspection prints an amended snapshot, a detailed static graph, and
the certificate shape consumed by the existing plan instantiator. The amended
snapshot fills each runtime's `runtime_files` with exact non-system files
outside its indexed standard library. The caller must save the exact returned
objects and original process output, bind their digests, and establish their
provenance. A JSON field alone does not authenticate an actual observation.
The evidence reference is a destination chosen by that caller; this tool does
not write it or certify that it has already been saved.

A `fresh` bundle has exactly three file references: `snapshot`, `index`, and
`report`, each with `path` and `sha256`. The snapshot and report are the actual
inspection outputs. It reindexes every complete standard-library set, rebuilds
and compares each native graph, and checks all seven selected tool identities.
Missing tools, changed bytes, changed indexed/native file modes, writable
selected tools, changed native aliases, or a formerly absent
weak dependency becoming available cause failure. The validator source must
still match the report. A successful fresh result describes the observation at
that point in time, not an atomic filesystem snapshot or an authorization to
run a command. A caller must invoke it at each relevant launch boundary; merely
running it once at workflow startup is insufficient.

## Dependency scope

Inspection starts at each selected interpreter and every indexed `.so` and
`.dylib` in its standard library. It handles thin and big-endian fat Mach-O
containers, ordinary arm64 slices, macOS minimum-version commands, dylib load
commands, and loader/executable/run-path substitutions. A required missing
dependency, ambiguous run-path target, unsupported required dyld command,
malformed image, or incompatible minimum OS stops inspection.
Embedded dyld environment declarations and unmodeled legacy library or
sub-framework lookup rules are rejected. An executable's declared dynamic
loader must be `/usr/lib/dyld`, which is recorded as an OS dependency.
Symlink aliases of native dependencies are resolved to exact regular target files and resolved
again during `fresh`. The existing standard-library reader refuses links in its
indexed trees.

Declared `/usr/lib`, `/System/Library/Frameworks`, and
`/System/Library/PrivateFrameworks` dependencies are recorded as inputs supplied
by the observed OS and dyld shared cache. They are not given invented standalone
file hashes. A system install name in a graph does not prove that a native
process successfully loaded it; actual sandboxed execution supplies that
evidence later.

The certificate's `complete` field means the **declared static graph** of the
selected interpreters and indexed extensions was resolved within this scope.
It does not prove all possible programmatic `dlopen` calls. The runtime's exact
file permissions and actual fixed driver execution must still constrain and
exercise dynamic loading. It also does not prove that a built SDK extension was
loaded, that installation succeeded, or that any product/platform acceptance
passed. Returned `sdk_executed` and `product_acceptance` fields remain false.

The parser follows Apple's [Mach-O load command definitions](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h)
and [dyld path rules](https://github.com/apple-oss-distributions/dyld/blob/main/doc/man/man1/dyld.1).
It deliberately rejects unresolved or ambiguous cases instead of granting a
directory or guessing fallback search paths.

## Bounds and development controls

Per-runtime native inspection allows at most 512 files, 2,048 graph states,
64 dependency levels, 64 inherited run paths, 128 MiB per image, and 2 GiB of
native reads. At most 64 exact extra native files may be passed to the existing
runtime plan. The three-runtime inspection/fresh operation shares a 180-second
deadline; existing standard-library indexing bounds remain in force. Failures
raise an exception and return a nonzero process status. There are no retries,
repairs, fallback downloads, or subprocesses. The workflow must also impose its
own bounded supervision and retain stdout/stderr on failure.

Run only the small local development controls with:

```sh
python3 -I -S -B test_native_closure.py \
  --helper-root /absolute/tools/ci/macos15/dynamic_inputs
```

The controls use synthetic Mach-O bytes and temporary miniature file trees.
They check malformed/fat images, OS/architecture rejection, run-path ambiguity,
required/weak dependencies, aliases, whole-index additions, byte/mode/tool
drift, shortened graph rejection, and inert defaults. Process spawning and
network connections are forbidden by tripwires. These controls do not execute
native bytes or substitute for a real macOS 15 build/install/runtime run.
