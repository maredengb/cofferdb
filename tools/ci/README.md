# Supervised macOS 15 maintainer commands

These optional helpers capture source preparation, native wheel builds, empty
virtual-environment creation and externally supplied installation checks on
macOS 15 arm64. SDK users do not need them. The helpers use Python's standard
library and the existing [source and wheel tools](../README.md).

Keep `macos15/runner.py`, `macos15/probe.py`, `macos15/workflow_build.py` and
`macos15/venv_prepare.py` together. The wrappers verify the sibling runner's
exact bytes before loading it. Plans select `q_process_supervisor.py` by its
absolute path and SHA-256. The supervisor uses its own actual file location;
it does not require a separate checkout or an internal project directory.

## Inspect a plan

The `verify` and `preview` modes validate a supplied plan and display the
commands or sandbox profile without launching those commands. For example:

```sh
python3.12 -I -S -B tools/ci/macos15/workflow_build.py preview \
  --plan /absolute/path/to/build-plan.json \
  --plan-sha256 <actual-plan-sha256>
```

`venv_prepare.py` accepts the same inspection options. `runner.py` validates a
runtime plan and produces a hash of its sandbox profile. Each entry defaults
to `verify`. The CLI help lists its arguments. Plans use canonical absolute
paths and exact file hashes; they must be generated for the selected machine
and inputs. The tools do not discover or install missing runtimes, dependencies
or an installation test driver.

## Execute the selected inputs

The explicit `run` mode additionally requires
`--approved-plan-sha256 <actual-plan-sha256>` to match the supplied plan. This
argument records the caller's plan selection; it does not grant OS permissions
or provide an authentication mechanism. Actual execution also verifies the
macOS 15 arm64 host and the selected files before starting commands.

- `workflow_build.py` calls the existing source preparer on a clean product
  checkout at a fixed commit, then the wheel builder with the three selected
  CPython interpreters. Its total deadline is 3,900 seconds, with at most
  300 seconds for preparation and 3,600 seconds for the build. Cargo dependencies
  must already be available for locked, offline resolution.
- `venv_prepare.py` creates three new empty virtual environments with copies
  of CPython 3.12, 3.13 and 3.14, without pip or the SDK. It verifies their paths,
  interpreter hashes and private directory identities within a shared
  300-second deadline. Existing directories are rejected.
- `runner.py` applies an explicit runtime sandbox to a selected environment.
  It first checks the actual OS/runtime and verifies that compiler execution,
  compiler reads and the selected numeric outbound connection are denied.
  It then runs the externally supplied, hash-bound installation driver under
  that same profile. The unit shares a 300-second deadline. The driver and its
  complete inputs are separate required assets.

The runtime sandbox permits the declared interpreter, standard library and
specific runtime dependencies, with writes restricted to the selected new
virtual environment and work directory. The supervising interpreter remains
outside the environment whose packages are being installed. The build helper
uses the build toolchain; the runtime sandbox's compiler restrictions apply
to the installation checks.

## Read the captured result

`q_process_supervisor.py` captures both original output streams and retains an
owned process-group leader through cleanup. Supplied commands must not detach
from that group. The group supervisor is not an OS sandbox. Its receipts report
the actual command exit, output completeness and process-group cleanup.

Outputs use new, separate directories. Failures preserve available outputs and
stop the sequence without an automatic retry. Output limits monitor observed
bytes; they are not filesystem quotas. Check the final receipt, original logs,
actual exit and cleanup together. A completed build, created environment or
captured installation command does not by itself establish package acceptance.
The installation driver's assertions and the tested artifacts determine the
coverage of that run.

These files do not install or enable a GitHub workflow, upload an artifact or
publish a package. An actual CI job must supply its fixed source, runtime and
dependency identities, plans and installation driver.
