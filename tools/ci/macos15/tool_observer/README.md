# Existing macOS build-tool observation

`tool_observer.py` fills the seven initially empty tool rows in an actual
Mac15 interpreter snapshot. It reads explicitly selected existing files and
captures four version commands with the unchanged process supervisor. It does
not search PATH, acquire dependencies, install software, build wheels, or run
the SDK. The default `verify` action performs no input or platform IO.

An explicit observation uses the pinned runtime bridge and original helpers:

```sh
python3 -I -S -B tool_observer.py observe \
  --runtime-bridge /absolute/runtime_bridge.py \
  --helper-root /absolute/tools/ci/macos15 \
  --bundle /absolute/observation-inputs.json \
  --bundle-sha256 ACTUAL_INPUT_SHA256 \
  --output /absolute/owned-private-parent/new-observation
```

The bundle has exactly `snapshot`, `selected_tools`, and `checkout` fields.
`snapshot` is a `{path, sha256}` reference to the original collector snapshot.
`checkout` is the selected source path. `selected_tools` maps exactly `cargo`,
`rustc`, `maturin`, `sandbox`, `compiler`, `prepare`, and `builder` to literal
absolute file paths. All ancestors and files must be free of symbolic links;
files must be bounded, regular, singly linked and not group/world writable.
Executables must be executable. Source scripts must be
`checkout/tools/prepare_source.py` and `checkout/tools/build_python.py`.
The sandbox path is `/usr/bin/sandbox-exec`.

Select actual toolchain `cargo` and `rustc` binaries, not resolved rustup
multiplexer paths. An isolated HOME is used; this tool does not configure
rustup or infer a default toolchain. Explicit caller selection, clean source
verification and dependency acquisition belong to the enclosing workflow.
Hashing the named scripts does not establish the checkout's Git identity; the
existing preparation/build steps must still verify that identity.

The four commands are exactly the selected cargo, rustc, maturin and clang
executables followed by `--version`. The first three selected filenames must
remain `cargo`, `rustc`, and `maturin`. Raw version outputs, actual command exit,
EOF and process-group cleanup receipts are retained. The stdout, stderr and sentinel status streams are
checked against the original supervisor path, size and digest seals immediately
after capture and again at finalization; changed streams are never re-signed. Maturin must report
`1.10.2`, as required by the existing builder. Rust and clang versions are
recorded rather than predicted or silently constrained to a new release.
Sandbox identity is its file digest and actual OS snapshot; the two source
scripts are identified by their file digests, without inventing version output.

Each command has at most 30 seconds, 64 KiB of stdout plus stderr, and the
original separate 16 KiB sentinel status-stream bound. The observation
has a cooperative 180-second deadline including input reads and final checks.
The original deadline starts at explicit observer entry, before bound input
loading, and is checked again after final output writes. All selected file digests, the base interpreter and pinned maintenance code
are rechecked before each launch. Bundle and snapshot bindings are checked
before observation, each launch and finalization. The original supervisor
provides process-group cleanup and raw capture. The outer workflow must also
supervise the entire observer; an in-progress filesystem read is not separately
preemptible here. No timeout restarts the observation.

Successful output includes a new `SNAPSHOT.json`, with only its `tools` field
changed, and `RESULT.json`, binding both snapshot identities, selected tools,
actual versions and capture receipts. Failures retain the observed prefix and
`HOLD.json`. A result file written before a late final check remains a candidate;
only successful actual tool exit with no HOLD can complete the observation.
Earlier output is not reused or overwritten. The original
collector files and their observation digest remain unchanged; this separate
report carries the tool observations. It does not prove tool native-dependency
completeness, sandbox capability, a successful build or platform acceptance.
The later native observation and fresh-launch gates remain required.

The output is new and below an existing caller-owned `0700` parent, separate
from source, interpreter, stdlib and maintenance inputs. Runtime installation
and the business driver are not part of this command.

Development controls use miniature files and substituted version captures:

```sh
python3 -I -S -B test_tool_observer.py \
  --helper-root /absolute/tools/ci/macos15 \
  --runtime-bridge /absolute/runtime_bridge.py
```

They cover exact version/receipt binding, drift, timeout, host and path failures,
no retry, retained failure prefixes and inert default behavior. The fixture
sandbox path is mapped to a tiny owned file; no system tool, native image,
SDK, network or actual subprocess is executed by these controls.
