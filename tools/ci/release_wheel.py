"""Build and test version 0.1.4 release wheels without publishing them."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile

VERSION = "0.1.4"
OLD_CRATE = (
    "https://static.crates.io/crates/cofferdb/cofferdb-0.1.3.crate",
    "2292053a3182dd0924fb15c82fe00d62b40c154f1754d465d4c4a88d2d980b6b",
)
PIP = (
    "https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl",
    "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd",
)
MATURIN_MAC = (
    "https://files.pythonhosted.org/packages/4a/85/1d1b64dbb6518ee633bfde8787e251ae59428818fea7a6bdacb8008a09bd/maturin-1.10.2-py3-none-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl",
    "7fbd997c5347649ee7987bd05a92bd5b8b07efa4ac3f8bcbf6196e07eb573d89",
)
EXPECTED = {"last_commit_id": 1, "document_count": 1, "chunk_count": 2}
PRIVATE_DATA = re.compile(rb"(?:/Users/|/private/(?:tmp|var)/|/home/)(?!runner/|runneradmin/)[^\x00\s]+")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, error):
    if not condition:
        raise ValueError(error)


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def actions_identity():
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id is None:
        return None
    require(run_id.isdecimal() and os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
            "use a new workflow dispatch after a failed attempt")
    return {"run_id": run_id, "attempt": 1}


def stage(work, value):
    (work / "stage").write_text(value + "\n", encoding="utf-8")


def download(pin, directory):
    url, checksum = pin
    path = directory / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as output:
        shutil.copyfileobj(response, output)
    require(sha(path) == checksum, "upstream package digest differs")
    return path


def source_identity(repo, commit):
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "full source commit required")
    require(run("git", "rev-parse", "HEAD", cwd=repo) == commit, "checkout differs from source SHA")
    require(not run("git", "status", "--porcelain", "--untracked-files=all", cwd=repo), "checkout is dirty")
    return {"commit": commit, "tree": run("git", "rev-parse", "HEAD^{tree}", cwd=repo)}


def environment(pythons, architecture, versions):
    machine = {"X64": "x86_64", "ARM64": "arm64"}[architecture]
    require(platform.machine() == run("uname", "-m") == machine, "host architecture differs")
    probe = """import json, platform, sys, sysconfig
print(json.dumps(dict(implementation=sys.implementation.name, version=platform.python_version(),
                      system=platform.system(), machine=platform.machine(), bits=64 if sys.maxsize > 2**32 else 32,
                      suffix=sysconfig.get_config_var('EXT_SUFFIX'), gil=not sysconfig.get_config_var('Py_GIL_DISABLED'))))"""
    runtimes = [json.loads(run(python, "-I", "-c", probe)) for python in pythons]
    require([row["version"] for row in runtimes] == versions, "interpreter micro versions differ")
    system = platform.system()
    for row, abi in zip(runtimes, ("312", "313", "314"), strict=True):
        require(row["implementation"] == "cpython" and row["bits"] == 64 and row["gil"],
                "regular 64-bit CPython required")
        require(row["machine"] == machine and row["system"] == system, "interpreter architecture differs")
        require(row["suffix"].startswith(f".cpython-{abi}-"), "native extension ABI differs")
    if system == "Darwin":
        version = run("sw_vers", "-productVersion")
        require(version.startswith("15."), "macOS 15.x required")
        require(run("sysctl", "-in", "sysctl.proc_translated") in ("", "0"), "translated execution")
    else:
        version = run("getconf", "GNU_LIBC_VERSION")
        require(version == "glibc 2.28" and machine == "x86_64", "glibc 2.28 x86_64 required")
    image = os.environ.get("ImageVersion") or os.environ.get("RELEASE_IMAGE", "")
    require(re.fullmatch(r"[A-Za-z0-9._:/@-]+", image), "runner image identity required")
    return {"system": system, "version": version, "machine": machine, "image": image, "python": runtimes}


def check_rust(expected_host):
    rust = run("rustc", "-vV").splitlines()
    cargo = run("cargo", "--version")
    require("release: 1.97.1" in rust and cargo.startswith("cargo 1.97.1 "), "Rust/Cargo 1.97.1 required")
    require(f"host: {expected_host}" in rust, "Rust host target differs")


def install(python, venv, pip, wheel, record):
    require(sha(wheel) == record["sha256"], "wheel digest differs before installation")
    run(python, "-I", "-m", "venv", venv)
    executable = venv / "bin/python"
    install_args = ("-I", "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", "--only-binary=:all:")
    run(executable, *install_args, pip)
    require(run(executable, "-I", "-c", "import pip; print(pip.__version__)") == "25.3", "pip version differs")
    run(executable, *install_args, wheel)
    return executable


def inspect_installed(digest):
    import cofferdb
    from cofferdb import _native

    prefix = Path(sys.prefix).resolve()
    require(sys.prefix != sys.base_prefix, "fresh virtual environment required")
    for module in (cofferdb, _native):
        require(Path(module.__file__).resolve().is_relative_to(prefix), "module did not load from the venv")
    require(sha(Path(_native.__file__)) == digest, "installed native file digest differs")
    require(
        (importlib.metadata.version("cofferdb"), cofferdb.__version__, cofferdb.engine_version)
        == (VERSION, VERSION, VERSION),
        "distribution, SDK or engine version differs",
    )
    require(not sysconfig.get_config_var("Py_GIL_DISABLED"), "installed interpreter is free-threaded")


def read_fixture(path, evidence, directory, kind):
    from cofferdb import Database

    oracle = json.loads(evidence.read_text(encoding="utf-8"))
    require(oracle["stats"] == EXPECTED and oracle["chunk"] == "wal#commit", "unexpected old fixture oracle")
    require(oracle["kind"] == kind and oracle["actions"] == actions_identity(), "unexpected fixture origin")
    require(sha(path) == oracle["sha256"], "old fixture digest differs")
    directory.mkdir()
    fresh = directory / "restored.coffer"
    shutil.copyfile(path, fresh)
    with Database.open(str(fresh), read_only=True) as db:
        stats = db.stats()
        require(all(stats[key] == value for key, value in EXPECTED.items()), "old fixture stats differ")
        notes = db.collection("notes")
        hit = notes.search("write ahead", top_k=1)["results"][0]
        chunk = notes.get_chunk(hit["chunk_id"])["chunk"]
        require(chunk["chunk_external_id"] == oracle["chunk"], "old fixture query differs")
    require(sha(fresh) == oracle["sha256"], "read-only fixture changed")


def check_egress(path):
    require(path.is_file() and not path.is_symlink(), "only regular artifact files may be uploaded")
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            require(not archive.comment and all(not row.comment and not row.extra for row in members),
                    "unexpected wheel archive metadata")
            payloads = [(row.filename.encode(), archive.read(row)) for row in members]
    else:
        payloads = [(path.name.encode(), path.read_bytes())]
    require(not any(PRIVATE_DATA.search(payload) for pair in payloads for payload in pair),
            "artifact contains a private filesystem path")


def release_build(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import build_python as builder

    output, repo = args.work.resolve(), args.repo.resolve()
    stage(output, "source-and-host")
    source_identity(repo, args.commit)
    host = environment(args.python, args.arch, args.versions)
    mac = host["system"] == "Darwin"
    check_rust(
        {"arm64": "aarch64-apple-darwin", "x86_64": "x86_64-apple-darwin"}[host["machine"]]
        if mac else "x86_64-unknown-linux-gnu"
    )
    if mac:
        require(run("xcodebuild", "-version").splitlines()[0] == "Xcode 16.0", "Xcode must be 16.0")
        require(run("xcrun", "--sdk", "macosx", "--show-sdk-version") == "15.0", "SDK must be 15.0")
        require(os.environ.get("MACOSX_DEPLOYMENT_TARGET") == "15.0", "deployment target must be 15.0")
    maturin = args.maturin
    stage(output, "build-inputs")
    if mac:
        toolkit = output / "build-tools"
        run(args.python[0], "-I", "-m", "venv", toolkit)
        run(toolkit / "bin/pip", "--isolated", "install", "--no-index", "--no-deps", download(MATURIN_MAC, output))
        maturin = str(toolkit / "bin/maturin")
    run("cargo", "fetch", "--locked", "--manifest-path", repo / "bindings/python/Cargo.toml")
    source = output / "source"
    run(args.python[0], "-I", repo / "tools/prepare_source.py", "--commit", args.commit, source)
    stage(output, "native-build")
    result = builder.build(source, output / "built", args.python, maturin)
    if mac:
        require(all(row["binary"]["build_sdk_version"] == "15.0" for row in result["artifacts"]),
                "actual wheel SDK differs")
    write_json(output / "build-environment.json", host)
    if not args.build_only:
        accept_build(repo, args.commit, host, output, output, args.python, args.fixture)


def accept_build(repo, commit, host, build, output, pythons, fixture=None):
    stage(output, "install")
    identity = source_identity(repo, commit)
    build_host = json.loads((build / "build-environment.json").read_text(encoding="utf-8"))
    require(build_host == host, "build and installation environments differ")
    result = json.loads((build / "built/build-manifest.json").read_text(encoding="utf-8"))
    for key, value in identity.items():
        require(result["source"]["source"][key] == value, "prepared wheel input source differs")
    pip = download(PIP, output)
    artifact = output / "release-wheels"
    artifact.mkdir()
    rows = []
    for python, build_record in zip(pythons, result["artifacts"], strict=True):
        abi = f"cp3{build_record['runtime']['minor']}"
        wheel = build / "built" / abi / build_record["wheel"]
        executable = install(python, output / f"venv-{abi}", pip, wheel, build_record)
        run(executable, "-I", "-B", Path(__file__), "inspect", build_record["native_sha256"], cwd=output)
        # A separate process runs the full installed suite, including the write/query/reopen lifecycle.
        suite = subprocess.run(
            [executable, "-I", "-B", "-m", "unittest", "discover", "-s", repo / "bindings/python/tests"],
            cwd=output, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        (output / f"tests-{abi}.log").write_text(suite.stdout + suite.stderr, encoding="utf-8")
        require(suite.returncode == 0, "installed test suite failed")
        counts = re.search(r"^Ran (\d+) tests? in ", suite.stderr, re.MULTILINE)
        require(counts and int(counts[1]) > 0, "installed test discovery returned no tests")
        if fixture:
            run(executable, "-I", "-B", Path(__file__), "inspect", build_record["native_sha256"],
                "--fixture", fixture, "--fixture-kind", "old_database", "--copy-to", output / f"old-{abi}", cwd=output)
        record = {name: build_record[name] for name in ("wheel", "sha256", "native_sha256", "binary")}
        record.update(python=build_record["runtime"]["version"], installed_suite_exit_code=0,
                      installed_tests=int(counts[1]),
                      old_database_checked=bool(fixture))
        staged = artifact / wheel.name
        shutil.copyfile(wheel, staged)
        check_egress(staged)
        rows.append(record)
    write_json(artifact / "result.json", {
        **identity, "source_input_sha256": result["source"]["source_input_sha256"], "environment": host,
        "acceptance_script_sha256": sha(Path(__file__)), "actions": actions_identity(),
        "rust": result["rustc"], "cargo": result["cargo"], "maturin": result["maturin"], "pip": "25.3",
        "artifacts": rows,
    })
    (artifact / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['wheel']}\n" for row in rows), encoding="utf-8",
    )
    for name in ("result.json", "SHA256SUMS"):
        check_egress(artifact / name)


def make_fixtures(work):
    archive = download(OLD_CRATE, work)
    imported = work / "cofferdb-0.1.3"
    run("tar", "-xzf", archive, "-C", work)
    check_rust("x86_64-unknown-linux-gnu")
    run("cargo", "build", "--locked", "--bin", "coffer", cwd=imported)
    coffer = imported / "target/debug/coffer"
    old_dir, snapshot_dir = work / "old", work / "snapshot"
    old_dir.mkdir()
    snapshot_dir.mkdir()
    # Keep generated lock files in the work directory, outside both artifacts.
    old, snapshot = work / "old.coffer", work / "snapshot.coffer"
    run(coffer, "init", old)
    manifest = "examples/data/markdown/import_manifest.json"
    run(coffer, "import", old, "--manifest", manifest, "--json", cwd=imported)
    stats = json.loads(run(coffer, "stats", old, "--json"))["result"]
    require(all(stats[key] == value for key, value in EXPECTED.items()), "v0.1.3 fixture stats differ")
    found = json.loads(run(coffer, "search", old, "--collection", "notes", "write ahead", "--json"))["result"]
    require(found["results"][0]["chunk_external_id"] == "wal#commit", "v0.1.3 fixture query differs")
    run(coffer, "snapshot", old, snapshot)
    fixtures = ((old, old_dir, "old_database"), (snapshot, snapshot_dir, "standalone_snapshot"))
    for fixture, directory, kind in fixtures:
        staged = directory / "fixture.coffer"
        shutil.copyfile(fixture, staged)
        evidence = directory / "fixture.json"
        write_json(evidence, {"stats": EXPECTED, "chunk": "wal#commit", "sha256": sha(staged),
                              "kind": kind, "actions": actions_identity()})
        check_egress(staged)
        check_egress(evidence)


def restore(args):
    host = environment(args.python, "X64", args.versions)
    require(host["system"] == "Darwin", "native Intel macOS recovery required")
    work = args.work.resolve()
    manifest = json.loads((args.wheels / "result.json").read_text(encoding="utf-8"))
    require(manifest["commit"] == args.commit, "restore input is from a different commit")
    require(manifest["actions"] == actions_identity(), "restore input is from a different workflow run")
    require(sha(Path(__file__)) == manifest["acceptance_script_sha256"], "restore assertion differs from build")
    require(manifest["environment"]["machine"] == "x86_64", "restore input is not Intel")
    records = manifest["artifacts"]
    require(len(records) == 3, "restore needs all three Intel wheels")
    pip = download(PIP, work)
    for python, version, record in zip(args.python, args.versions, records, strict=True):
        abi = f"cp{version.split('.')[0]}{version.split('.')[1]}"
        wheel = f"cofferdb-{VERSION}-{abi}-{abi}-macosx_15_0_x86_64.whl"
        require(record["wheel"] == wheel and record["python"] == version, "restore wheel or Python differs")
        executable = install(python, work / f"venv-{abi}", pip, args.wheels / wheel, record)
        run(executable, "-I", "-B", Path(__file__), "inspect", record["native_sha256"],
            "--fixture", args.snapshot, "--fixture-kind", "standalone_snapshot",
            "--copy-to", work / f"restore-{abi}", cwd=work)
    output = work / "verified"
    output.mkdir()
    report = output / "restore.json"
    write_json(report, {
        "commit": args.commit, "environment": host, "snapshot_sha256": sha(args.snapshot / "fixture.coffer"),
        "actions": actions_identity(),
        "wheels": {row["wheel"]: row["sha256"] for row in records},
        "restore_exit_code": 0,
    })
    check_egress(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build")
    build.add_argument("--repo", type=Path, required=True)
    build.add_argument("--arch", required=True, choices=["X64", "ARM64"])
    build.add_argument("--fixture", type=Path)
    build.add_argument("--maturin", default="maturin")
    build.add_argument("--build-only", action="store_true")
    accept = actions.add_parser("accept")
    accept.add_argument("--repo", type=Path, required=True)
    accept.add_argument("--built", type=Path, required=True)
    fixtures = actions.add_parser("fixtures")
    fixtures.add_argument("work", type=Path)
    recovery = actions.add_parser("restore")
    recovery.add_argument("--wheels", type=Path, required=True)
    recovery.add_argument("--snapshot", type=Path, required=True)
    for action in (build, accept, recovery):
        action.add_argument("--python", type=Path, nargs=3, required=True)
        action.add_argument("--versions", nargs=3, required=True)
        action.add_argument("--commit", required=True)
        action.add_argument("work", type=Path)
    inspect = actions.add_parser("inspect")
    inspect.add_argument("native_sha256")
    inspect.add_argument("--fixture", type=Path)
    inspect.add_argument("--fixture-kind", choices=["old_database", "standalone_snapshot"])
    inspect.add_argument("--copy-to", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        release_build(args)
    elif args.action == "accept":
        host = environment(args.python, "X64", args.versions)
        require(host["system"] == "Linux", "separate acceptance requires baseline Linux")
        accept_build(args.repo.resolve(), args.commit, host, args.built.resolve(), args.work.resolve(), args.python)
    elif args.action == "fixtures":
        make_fixtures(args.work)
    elif args.action == "restore":
        restore(args)
    else:
        inspect_installed(args.native_sha256)
        if args.fixture:
            read_fixture(args.fixture / "fixture.coffer", args.fixture / "fixture.json",
                         args.copy_to, args.fixture_kind)


if __name__ == "__main__":
    main()
