# macOS 15 package job

`job.py` connects fixed public dependency preparation to the existing build and
runtime workflow. Its default `verify` action is inert. The companion
`macos15-package.yml` is installed at `.github/workflows/macos15-package.yml`. It has only a manual dispatch trigger. The
caller supplies a reviewed public commit and an execution-record reference;
these strings are recorded, not treated as proof of permission or acceptance.

The job selects the existing public repository by numeric ID, requires the
selected commit to equal the dispatch SHA and refuses a rerun attempt. Checkout
uses the already selected pinned checkout action with credential persistence
disabled. The standard `macos-15` runner is checked at runtime for native arm64
and actual macOS 15. No Python, Rust or Xcode installation is attempted. The
existing Framework CPython 3.12 starts the entry; the established collector
independently observes the three actual supported runtimes later.

The existing workflow's 7200-second parent timer encloses dependency preparation,
observation, builds, installations, business calls and final checks. One small
pinned entry is read before installing that existing timer. No new supervisor
or independent restart clock is introduced. Preparation has a smaller 1800-second
limit within it. The GitHub step/job additionally have finite 122/130-minute
limits for termination and evidence upload; these do not grant more product
execution time. A timeout or lost cleanup receipt remains incomplete.

Preparation uses the existing Q supervisor and explicit child environments:

1. Observe the current Apple developer directory with `xcode-select --print-path`.
   Select and hash real Git/clang files inside that installation. This avoids
   Apple's multiply hard-linked `/usr/bin` dispatch wrappers. Read the exact
   checkout root, clean status including untracked files, origin, commit and tree.
   Cargo configuration in the checkout, binding path or ancestor directories
   stops preparation before network access.
2. Select the already installed `stable-aarch64-apple-darwin` cargo/rustc pair,
   preserving its actual hashes. Observe `rustc -vV` and require native arm64 and
   Rust 1.80 or newer. Missing tools stop; rustup shims or downloads are not used.
3. Download one fixed pip 25.3 wheel and one fixed maturin 1.10.2 macOS wheel from
   their declared official HTTPS URLs. Each curl command allows 180 seconds,
   no redirect and no retry, and is checked against exact length and SHA256.
   pip remains a local runtime input. Maturin's complete archive, RECORD and
   license metadata are retained; only its exact binary is extracted. Actual
   selected Rust and maturin binaries must contain ordinary arm64 Mach-O slices
   compatible with this host. This static check is not a complete compiler
   dependency-closure claim; subsequent actual commands must still succeed.
4. In a new empty Cargo home, run the fixed two-lock `cargo vendor --locked`
   recipe once, at most 1200 seconds. Both locks select the existing 19 public
   registry packages. Audit every vendored source member and package checksum,
   preserving notices, checksum files and the complete file index. Write the
   explicit source replacement config rather than executing vendor stdout.
5. Execute both `cargo metadata --frozen` commands under a system sandbox that
   denies network access, at most 120 seconds each. Recheck clean source,
   tool identities and complete vendor bytes. The offline environment is kept
   for the unchanged original build; the strict compiler/network-denying runtime
   profile remains unchanged.
6. Create three exact driver contexts from this real run/attempt, the earliest
   parent INTENT, completed preparation record and the fixed public fixture.
   Actual wheel/native/source/venv values remain the runtime bridge's existing
   whole-token placeholders until the real builder and venv results supply them.
   The original installer now uses the fixed pip 25.3 input before SDK installation.
   Each runtime command keeps its existing 240-second limit within 300 seconds.
7. Invoke the existing workflow with the same parent budget. Recheck original
   preparation stream seals, tool/context bytes and vendor contents at completion.
   Read back the clean public source again in a bounded 120-second check within
   the original parent deadline.
   A generated RESULT remains provisional until actual successful job exit and
   retrieved artifact checks.

The new output is separate from checkout and toolchain inputs, with owned 0700
subdirectories. Files are never copied into an existing output. Failure preserves
the prefix and attempts to write HOLD; there is no automatic retry, cache cleanup,
package publication or repository mutation. The artifact action runs on failure
as well as success, retaining raw evidence, dependency inputs/licenses, exact
wheels and runtime work. It excludes compiler build targets, Cargo registry cache
and disposable installed venvs. Installed-file identities and rewritten RECORD
checks are retained in runtime evidence. Artifact retention is 14 days; evidence
must be retrieved and preserved before expiry.

This job covers three actual Mac platform installation/business cells only.
It does not itself decide matrix acceptance, scoring/cost results, Linux status,
human quickstart results or release permission. The precise source, dependency,
execution and upload scope must be reviewed before dispatch.

Local development controls substitute every external command and native/platform
operation. They validate real input-contract generation and locked-vendor audit
on tiny synthetic bytes, ordering, first failure, original raw seals, deadlines,
source/config drift and inert defaults. The candidate recipe's separate six
archive/path fixtures are also retained. None execute SDK, build, download,
install or claim compatibility with macOS 15:

```sh
python3.12 -I -S -B test_job.py --product /absolute/product --installer /absolute/installed_wheel
python3.12 -I -S -B check_fixtures.py
```

After the final source readback and enclosing result write, the job invokes the
retained workflow verifier again. It uses the original wheel, document and raw
stream identities, and rechecks the original vendor index and maturin archive.
It performs no additional subprocess or installation and fails the job if any
of those output bytes changed.

The three actual context cell IDs are N01, N02 and N03 for CPython 3.12, 3.13
and 3.14 respectively. Offline assertion mappings can refer to the same actual
cell and attempt; they do not cause an additional installation or a new context.
