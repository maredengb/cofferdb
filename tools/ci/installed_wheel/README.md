# Exact offline wheel installation and driver connection

`install_driver.py` connects an actual built wheel to an existing business
verification driver. It bootstraps with the selected venv's bundled ensurepip,
installs the fixed pip 25.3 wheel, checks the actual installer, installs the SDK
and checks installed files before importing the SDK, then calls that driver's
unchanged `main()` interface. It does not define a new search, scoring, cost or
acceptance oracle. The default `verify` action has no filesystem/platform probes,
installation, SDK import or subprocess.

The explicit `run` arguments are:

- `--runtime-bridge`: the fixed, digest-pinned runtime bridge source.
- `--wheel`, `--wheel-sha256`, `--native-sha256`: actual builder outputs.
- `--source-metadata`, `--source-input-sha256`, `--sdk-version`, `--engine-version`:
  actual prepared-source identity. This version selects SDK/engine `0.1.1`.
- `--pip-wheel`: the fixed local upstream pip 25.3 wheel.
- `--venv`: the actual new venv used to invoke this process with `-I -B`.
- `--driver`, `--driver-sha256`: the selected reviewed business driver bytes.
- `--context`, `--context-sha256`: exact non-artifact driver context.
- `--fixtures`: the driver's fixed local source/provenance fixture directory.
- `--work`: a new business-driver output path below an existing private work root.

The context object has exactly `attempt_id`, `cell_id`,
`authorization_reference`, `preflight_evidence`, `budget_reference`, and
`files_sha256`. The first five are actual nonempty caller references; the last
is the independently frozen derived-fixture digest map. This component does not
invent approvals, expected query results, fixture hashes or budget references.
The outer execution controls determine whether these selected objects may run;
a populated JSON context alone grants no execution capability.

The interpreter must be ordinary 64-bit CPython 3.12, 3.13 or 3.14 in the selected
isolated venv, with bytecode generation disabled and initially empty
site-packages. Native Mac15 arm64 or glibc Linux x86_64 is selected from actual
runtime facts. Wheel ABI/platform filenames and internal tags must match.
This observation does not prove the full OS matrix or minimum glibc baseline.

Before installation the component verifies the exact archive digest, native
member digest, prepared `_source.json` bytes, complete wheel RECORD, distribution
name/version, ABI tag, absence of runtime dependencies and unexpected relocated
payload/startup files. Archive members are bounded and paths cannot escape.
The fixed source builder remains responsible for complete source/license and
native-binary inspection; these are not replaced by a wheel filename.

The four commands use the same venv interpreter:

```text
python -I -B -m ensurepip --default-pip
python -I -B -m pip --isolated install --no-index --no-deps --no-compile \
  --only-binary=:all: --disable-pip-version-check --no-input --no-cache-dir \
  --force-reinstall EXACT_LOCAL_PIP_25_3_WHEEL
python -I -B -c 'import json,pip;print(json.dumps({"version":pip.__version__,"path":pip.__file__},sort_keys=True))'
python -I -B -m pip --isolated install --no-index --no-deps --no-compile \
  --only-binary=:all: --disable-pip-version-check --no-input --no-cache-dir EXACT_LOCAL_WHEEL
```

No shell, new process group, alternate installer or source build is used.
These children retain the enclosing sandbox and process group. Mac execution
must occur inside the original runner's byte-identical profile that already
passed compiler execution/read and network-denial probes. Linux requires its
separately selected clean runtime container. The enclosing bounded supervisor
remains responsible for process-group exit, raw capture and final acceptance of
this command's actual exit; this adapter does not certify those properties.

The adapter has a 240-second cooperative budget within the original runtime
command/whole limits, not an extension to them. Installer subprocess timeouts
use the remaining original adapter budget. The same deadline is checked again
after command records and raw streams are opened, immediately before spawning;
expired time is never clamped into permission to start a process. The same
budget is checked after native import/file verification, before executing the
driver module, and again before entering its `main()`. The outer runner can impose a shorter
remaining time and must supervise SDK/driver calls and any blocked IO as well.
First failure stops without retry. It does not claim the SDK driver is
independently interruptible from inside the same Python process.

After pip succeeds, every installed package file and immutable dist-info payload
must match the wheel. Only pip's `INSTALLER`, `REQUESTED` and exact local
`direct_url.json` additions are accepted; installed RECORD must cover every
package/metadata file exactly once with matching size/digest, except its own
unhashed entry. Symlinks, unexpected package files and changed native/source
bytes fail before the SDK import.

The driver contract's `dist_info` digest represents the canonical inventory of
immutable wheel metadata, excluding RECORD. The actual pip-rewritten RECORD is
verified separately and recorded with its own digest and complete installed
file map. The other three identities represent the wheel bytes, actual native
member and `_source.json` bytes. No digest is substituted for another identity.
The selected import specification, actual loaded package/native paths and
reported SDK/engine versions are checked against those files.

The resulting actual contract is written into the new sibling directory
`WORK-installation`, then the selected business driver's `main()` receives
`--fixtures`, `--contract`, `--contract-sha256` and the still-new `--work` path.
Its business assertions remain unchanged. Input and installed-file identities
are checked again after it returns. A successful component result explicitly
retains `product_acceptance: false` and requires a successful outer capture.

Evidence includes exact installer argv, raw output, actual return codes,
installed identities, generated driver contract, and a result or HOLD. A result
written just before a late final deadline failure remains provisional and is
accompanied by HOLD. Do not interpret a file's existence as successful tool exit.
The component never removes a partial installation or overwrites an old attempt.

Development controls use tiny synthetic zip members and fake installers,
SDK modules and drivers. No synthetic native bytes are executed and no SDK is
imported. They exercise the complete adapter call chain with those substitutions,
as well as altered records/native bytes, wrong loaded paths, symlinks, installer
failure and inert default behavior:

```sh
python3 -I -S -B test_install_driver.py --runtime-bridge /absolute/runtime_bridge.py
```

These controls are development evidence, not actual pip compatibility, sandbox
capability, final wheel installation or platform/whole-version acceptance.

## Fixed pip installer

The runtime also requires `--pip-wheel`, pointing to the exact upstream
`pip-25.3-py3-none-any.whl` (1,778,622 bytes, SHA256
`9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd`).
The enclosing job supplies this file before entering the offline runtime;
the adapter never downloads it. Add its path and digest to the runtime's
read-only external inputs.

The four commands share the original 240-second adapter budget: ensurepip
bootstrap, offline replacement with that exact pip wheel, an isolated actual
pip version/path probe, and then the SDK wheel installation. They preserve
the enclosing process group. All pip wheel members other than the rewritten
RECORD are compared to the installed bytes before the SDK command. The actual
pip probe must report version 25.3 from this new venv. A failure stops before
SDK installation; installing pip and the SDK in one transaction is not used.
The bootstrap version is recorded in its raw output and is not described as
the SDK installer. Original failures remain separate from this implementation.
