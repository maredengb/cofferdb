"""Build and test release wheels and check the Linux ARM64 source crate without publishing them."""
import argparse
from contextlib import chdir
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile

VERSION = "0.2.0"
OLD_CRATE = (
    "https://static.crates.io/crates/cofferdb/cofferdb-0.1.4.crate",
    "3a4f00ab817349206b3cc4b3e11e80b0b475ff6e4b895c12838967ee853039e8",
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
# Independent expectations for the bundled Markdown fixture and its legacy snapshot.
MODEL = "synthetic-demo-2d-v1"
CHECKPOINT_QUERY = {"model_name": MODEL, "dim": 2, "vector": [0, 1]}
EXPECTED_RECOVERY = {
    "collection": {
        "name": "notes", "embedding_dim": 2, "embedding_model": MODEL, "distance_metric": "cosine",
        "vector_index": {"type": "exact"}, "vectors_normalized": False,
        "tokenizer": {"name": "default_en", "profile": "latin_code_default", "language_hint": "en",
                      "version": "0.1", "config": {"case_fold": "lower", "split": "unicode_punctuation",
                                                    "stop_words": "default_en"}},
    },
    "embeddings": [
        {"chunk_external_id": "wal#checkpoint", "model_name": MODEL, "dim": 2, "vector": [0, 1]},
        {"chunk_external_id": "wal#commit", "model_name": MODEL, "dim": 2, "vector": [1, 0]},
    ],
    "vector_hits": [["wal#checkpoint", 1.0], ["wal#commit", 0.0]],
}
PRIVATE_DATA = re.compile(rb"(?:/Users/|/private/(?:tmp|var)/|/home/)(?!runner/|runneradmin/)[^\x00\s]+")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, error):
    if not condition:
        raise ValueError(error)


def run(*args, cwd=None, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def actions_identity():
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id is None:
        return None
    require(run_id.isdecimal() and os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
            "use a new workflow dispatch after a failed attempt")
    return {"run_id": run_id, "attempt": 1}


def stage(work, value, abi=""):
    (work / "stage").write_text(f"{value} {abi}\n", encoding="utf-8")


def download(pin, directory):
    url, checksum = pin
    path = directory / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as output:
        shutil.copyfileobj(response, output)
    require(sha(path) == checksum, "upstream package digest differs")
    return path


def source_identity(repo, commit):
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "full source commit required")
    origin = run("git", "remote", "get-url", "origin", cwd=repo).removesuffix(".git")
    require(origin in ("https://github.com/maredengb/cofferdb", "git@github.com:maredengb/cofferdb"),
            "unexpected product repository")
    require(run("git", "rev-parse", "HEAD", cwd=repo) == commit, "checkout differs from source SHA")
    require(not run("git", "status", "--porcelain", "--untracked-files=all", cwd=repo), "checkout is dirty")
    return {"commit": commit, "tree": run("git", "rev-parse", "HEAD^{tree}", cwd=repo)}


def environment(pythons, architecture, versions):
    system = platform.system()
    require(system in {"Darwin", "Linux"}, "unsupported build system")
    machine = "x86_64" if architecture == "X64" else {"Darwin": "arm64", "Linux": "aarch64"}[system]
    require(os.environ.get("RUNNER_ARCH", architecture) == architecture, "runner architecture differs")
    require(platform.machine() == run("uname", "-m") == machine, "host architecture differs")
    probe = """import json, platform, sys, sysconfig
print(json.dumps(dict(implementation=sys.implementation.name, version=platform.python_version(),
                      system=platform.system(), machine=platform.machine(), bits=64 if sys.maxsize > 2**32 else 32,
                      suffix=sysconfig.get_config_var('EXT_SUFFIX'), gil=not sysconfig.get_config_var('Py_GIL_DISABLED'))))"""
    runtimes = [json.loads(run(python, "-I", "-c", probe)) for python in pythons]
    require([row["version"] for row in runtimes] == versions, "interpreter micro versions differ")
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
        require(version == "glibc 2.28", "release acceptance requires the glibc 2.28 baseline")
    image = os.environ.get("ImageVersion", "") if system == "Darwin" else os.environ.get("RELEASE_IMAGE", "")
    if system == "Linux":
        require(re.fullmatch(r"quay\.io/pypa/manylinux_2_28_" + machine + r"@sha256:[0-9a-f]{64}", image),
                "pinned architecture-specific manylinux image required")
    require(re.fullmatch(r"[A-Za-z0-9._:/@-]+", image), "runner image identity required")
    return {"system": system, "version": version, "machine": machine, "image": image, "runner_image": os.environ.get("ImageVersion", ""),
            "actions": actions_identity(), "python": runtimes}


def check_rust(expected_host):
    rust = run("rustc", "-vV").splitlines()
    cargo = run("cargo", "--version")
    require("release: 1.97.1" in rust and cargo.startswith("cargo 1.97.1 "), "Rust/Cargo 1.97.1 required")
    require(f"host: {expected_host}" in rust, "Rust host target differs")
    return {"host": expected_host, "rustc": run("rustc", "--version"), "cargo": cargo}


def install(python, work, abi, pip, wheel, record):
    stage(work, "wheel-digest", abi)
    require(sha(wheel) == record["sha256"], "wheel digest differs before installation")
    stage(work, "venv", abi)
    venv = work / f"venv-{abi}"
    run(python, "-I", "-m", "venv", venv)
    executable = venv / "bin/python"
    install_args = ("-I", "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", "--only-binary=:all:")
    stage(work, "pip-install", abi)
    run(executable, *install_args, pip)
    require(run(executable, "-I", "-c", "import pip; print(pip.__version__)") == "25.3", "pip version differs")
    stage(work, "wheel-install", abi)
    run(executable, *install_args, wheel)
    return executable


MAINTENANCE_TESTS = {"test_snapshot_restore", "test_rebuild_reopen", "test_compact_generations",
                     "test_snapshot_publish_race", "test_maintenance_rejections"}


def run_maintenance_guide(executable, repo):
    guide = (repo / "docs/python.md").read_text(encoding="utf-8")
    sections = {section.splitlines()[0]: section for section in guide.split("\n## ")[1:]}
    examples = []
    for heading, count in (("Write, search, reopen and check", 1), ("Check and restore a snapshot", 2)):
        blocks = re.findall(r"^```python\n(.*?)^```$", sections[heading], re.MULTILINE | re.DOTALL)
        require(len(blocks) == count, "Python maintenance guide examples incomplete")
        examples.extend(blocks)
    with tempfile.TemporaryDirectory() as empty:
        run(executable, "-I", "-B", "-c", "\n".join(examples), cwd=empty, env={**os.environ, "PATH": ""})


def check_guide_links(readme):
    for guide in ("rust", "python"):
        url = f"https://github.com/maredengb/cofferdb/blob/v{VERSION}/docs/{guide}.md"
        require(f"]({url})" in readme, f"package README missing the {guide} guide for {VERSION}")


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
    check_guide_links(importlib.metadata.metadata("cofferdb").get_payload())
    require(not sysconfig.get_config_var("Py_GIL_DISABLED"), "installed interpreter is free-threaded")


def fixture_oracle(directory, kind):
    require({p.name for p in directory.iterdir()} == {"fixture.coffer", "fixture.json"},
            "unexpected compatibility input files")
    oracle = json.loads((directory / "fixture.json").read_bytes())
    require(oracle["stats"] == EXPECTED and oracle["chunk"] == "wal#commit", "unexpected old fixture oracle")
    require(oracle["kind"] == kind and oracle["actions"] == actions_identity(), "unexpected fixture origin")
    require(oracle["crate"] == {"url": OLD_CRATE[0], "sha256": OLD_CRATE[1]}, "unexpected old source crate")
    require(sha(directory / "fixture.coffer") == oracle["sha256"], "old fixture digest differs")
    return oracle


def verify_recovered_contents(collections, export, results):
    require(len(collections) == 1, "restored collection count differs")
    config = {key: collections[0][key] for key in EXPECTED_RECOVERY["collection"]}
    embeddings = [json.loads(line) for line in (export / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()]
    observed = {
        "collection": config,
        "embeddings": sorted(embeddings, key=lambda row: row["chunk_external_id"]),
        "vector_hits": [[row["chunk_external_id"], row["vector_score"]] for row in results],
    }
    for field, expected in EXPECTED_RECOVERY.items():
        require(observed[field] == expected, f"restored {field} differs")
    return observed


def cli_recovered_contents(coffer, database):
    # Generate the query independently of the snapshot and exported vectors.
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        query = work / "query.json"
        write_json(query, CHECKPOINT_QUERY)
        collections = json.loads(run(coffer, "list-collections", database, "--json"))["result"]["collections"]
        results = json.loads(run(coffer, "vector-search", database, "--collection", "notes", "--embedding",
                                 query, "--top-k", "2", "--json"))["result"]["results"]
        export = work / "export"
        run(coffer, "export", database, export, "--collection", "notes", "--json")
        return verify_recovered_contents(collections, export, results)


def read_fixture(source, directory, kind):
    from cofferdb import Database, check

    oracle = fixture_oracle(source, kind)
    directory.mkdir()
    fresh = directory / "fixture.coffer"
    shutil.copyfile(source / "fixture.coffer", fresh)
    with chdir(directory):
        checked = check(fresh.name)
    require(checked == oracle["cli_check"], "installed Python check differs from CLI on identical fixture bytes")
    with Database.open(str(fresh), read_only=True) as db:
        stats = db.stats()
        require(all(stats[key] == value for key, value in EXPECTED.items()), "old fixture stats differ")
        notes = db.collection("notes")
        hit = notes.search("write ahead", top_k=1)["results"][0]
        chunk = notes.get_chunk(hit["chunk_id"])["chunk"]
        require(chunk["chunk_external_id"] == oracle["chunk"], "old fixture query differs")
        with tempfile.TemporaryDirectory() as temporary:
            export = Path(temporary) / "export"
            notes.export_jsonl(str(export))
            results = notes.search(embedding=CHECKPOINT_QUERY, top_k=2)["results"]
            recovered = verify_recovered_contents(db.list_collections(), export, results)
    require(sha(fresh) == oracle["sha256"], "read-only fixture changed")
    return {"sha256": sha(fresh), "stats": {key: stats[key] for key in EXPECTED},
            "chunk": chunk["chunk_external_id"], "cli_check": checked, "recovered_contents": recovered}


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
    triple = "aarch64" if host["machine"] == "arm64" else host["machine"]
    tools = check_rust(triple + ("-apple-darwin" if mac else "-unknown-linux-gnu"))
    if mac:
        tools["xcode"] = run("xcodebuild", "-version").splitlines()[0]
        tools["sdk"] = run("xcrun", "--sdk", "macosx", "--show-sdk-version")
        require(tools["xcode"] == "Xcode 16.0" and tools["sdk"] == "15.0", "Xcode 16.0 and SDK 15.0 required")
        require(os.environ.get("MACOSX_DEPLOYMENT_TARGET") == "15.0", "deployment target must be 15.0")
    else:
        tools["rust_channel_sha256"] = sha(output / "rust-channel.toml")
    maturin = args.maturin
    stage(output, "build-inputs")
    if mac:
        toolkit = output / "build-tools"
        run(args.python[0], "-I", "-m", "venv", toolkit)
        run(toolkit / "bin/pip", "--isolated", "install", "--no-index", "--no-deps", download(MATURIN_MAC, output))
        maturin = str(toolkit / "bin/maturin")
    for manifest in (repo / "Cargo.toml", repo / "bindings/python/Cargo.toml"):
        run("cargo", "fetch", "--locked", "--manifest-path", manifest)
    source = output / "source"
    prepare_env = {**os.environ, "CARGO_TARGET_DIR": str(output / "prepare-target")}
    run(args.python[0], "-I", "-B", repo / "tools/prepare_source.py", "--commit", args.commit, source,
        env=prepare_env)
    stage(output, "native-build")
    result = builder.build(source, output / "built", args.python, maturin)
    if mac:
        require(all(row["binary"]["build_sdk_version"] == "15.0" for row in result["artifacts"]),
                "actual wheel SDK differs")
    # Keep the CLI outside PATH and the wheel. Only the explicit comparison uses it.
    run("cargo", "build", "--locked", "--offline", "--bin", "coffer", cwd=repo, env=prepare_env)
    cli = output / "comparison-coffer"
    shutil.copy2(output / "prepare-target/debug/coffer", cli)
    tools["comparison_cli_sha256"] = sha(cli)
    write_json(output / "build-environment.json", host)
    write_json(output / "build-tools.json", tools)
    if not args.build_only:
        accept_build(repo, args.commit, host, output, output, args.python, args.fixture)


def accept_build(repo, commit, host, build, output, pythons, fixture=None):
    stage(output, "acceptance-inputs")
    identity = source_identity(repo, commit)
    build_host = json.loads((build / "build-environment.json").read_text(encoding="utf-8"))
    require(build_host == host, "build and installation environments differ")
    result = json.loads((build / "built/build-manifest.json").read_text(encoding="utf-8"))
    for key, value in identity.items():
        require(result["source"]["source"][key] == value, "prepared wheel input source differs")
    tools = json.loads((build / "build-tools.json").read_bytes())
    cli = build / "comparison-coffer"
    require(sha(cli) == tools["comparison_cli_sha256"], "comparison CLI transfer differs")
    stage(output, "pip-download")
    pip = download(PIP, output)
    artifact = output / "release-wheels"
    artifact.mkdir()
    rows = []
    for python, build_record in zip(pythons, result["artifacts"], strict=True):
        expected_runtime = host["python"][len(rows)]
        require(build_record["runtime"]["version"] == expected_runtime["version"], "wheel interpreter version differs")
        abi = f"cp3{build_record['runtime']['minor']}"
        tag = ("macosx_15_0_" if host["system"] == "Darwin" else "manylinux_2_28_") + host["machine"]
        require(build_record["wheel"] == f"cofferdb-{VERSION}-{abi}-{abi}-{tag}.whl", "wheel name differs")
        wheel = build / "built" / abi / build_record["wheel"]
        executable = install(python, output, abi, pip, wheel, build_record)
        stage(output, "wheel-import", abi)
        run(executable, "-I", "-B", Path(__file__).resolve(), "inspect", build_record["native_sha256"], cwd=output)
        # A separate process runs the full installed suite, including the write/query/reopen lifecycle.
        stage(output, "installed-tests", abi)
        suite = subprocess.run(
            [executable, "-I", "-B", "-m", "unittest", "discover", "-v", "-s", repo / "bindings/python/tests"],
            cwd=output, env={**os.environ, "PATH": "", "COFFER_TEST_CLI": ""},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        (output / f"tests-{abi}.log").write_text(suite.stdout + suite.stderr, encoding="utf-8")
        require(suite.returncode == 0, "installed test suite failed")
        counts = re.search(r"^Ran (\d+) tests? in ", suite.stderr, re.MULTILINE)
        require(counts and int(counts[1]) > 0, "installed test discovery returned no tests")
        require(not re.search(r"skipped=[1-9]", suite.stderr), "installed suite skipped required coverage")
        check_cases = re.findall(r"^test_\w+ \(test_check\.CheckTests\.[^)]+\).*? \.\.\. ok$", suite.stderr, re.MULTILINE)
        require(check_cases, "installed path-based check suite did not pass")
        maintenance = sorted(re.findall(r"^(test_\w+) \(test_maintenance\.MaintenanceTests\.[^)]+\).*? \.\.\. ok$", suite.stderr, re.MULTILINE))
        require(set(maintenance) >= MAINTENANCE_TESTS, "installed maintenance evidence incomplete")
        run_maintenance_guide(executable, repo)
        # The installed package has already passed the suite without a CLI on PATH.
        comparison = json.loads(run(executable, "-I", "-B", repo / "bindings/python/tests_native/compare_cli.py",
                                    cli, cwd=output))
        observed = None
        if fixture:
            stage(output, "old-database", abi)
            observed = json.loads(run(
                executable, "-I", "-B", Path(__file__).resolve(), "inspect", build_record["native_sha256"],
                "--fixture", fixture, "--fixture-kind", "old_database", "--copy-to", output / f"old-{abi}", cwd=output))
        record = {name: build_record[name] for name in ("wheel", "sha256", "native_sha256", "binary")}
        record.update(versions=[VERSION, VERSION, VERSION], installed_native_verified=True,
                      python=build_record["runtime"]["version"], installed_suite_exit_code=0,
                      installed_tests=int(counts[1]), check_tests=len(check_cases), maintenance=maintenance,
                      cli_comparison=comparison, comparison_cli_sha256=tools["comparison_cli_sha256"],
                      old_database=observed)
        staged = artifact / wheel.name
        stage(output, "artifact-check", abi)
        shutil.copyfile(wheel, staged)
        check_egress(staged)
        rows.append(record)
    stage(output, "acceptance-report")
    write_json(artifact / "result.json", {
        **identity, "source_input_sha256": result["source"]["source_input_sha256"], "environment": host,
        "job": os.environ.get("GITHUB_JOB"), "build_tools": tools,
        "acceptance_script_sha256": sha(Path(__file__)), "actions": actions_identity(),
        "linux_script_sha256": sha(Path(__file__).with_name("linux_release.sh")) if host["system"] == "Linux" else None,
        "rust": result["rustc"], "cargo": result["cargo"], "maturin": result["maturin"], "pip": "25.3",
        "artifacts": rows,
    })
    (artifact / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['wheel']}\n" for row in rows), encoding="utf-8",
    )
    for name in ("result.json", "SHA256SUMS"):
        check_egress(artifact / name)


def check_database(coffer, path):
    stats = json.loads(run(coffer, "stats", path, "--json"))["result"]
    require(all(stats[key] == value for key, value in EXPECTED.items()), "sample database stats differ")
    found = json.loads(run(coffer, "search", path, "--collection", "notes", "write ahead", "--json"))["result"]
    require(found["results"][0]["chunk_external_id"] == "wal#commit", "sample query differs")
    report = json.loads(run(coffer, "check", path, "--verify-indexes", "--json"))["result"]
    require(report["status"] == "ok", "database integrity check failed")


def check_crate(args):
    work, repo = args.work.resolve(), args.repo.resolve()
    identity = source_identity(repo, args.commit)
    require(platform.system() == "Linux" and platform.machine() == run("uname", "-m") == "aarch64"
            and os.environ.get("RUNNER_ARCH", "ARM64") == "ARM64", "native Linux ARM64 required")
    check_rust("aarch64-unknown-linux-gnu")
    build_env = {**os.environ, "CARGO_TARGET_DIR": str(work / "target")}
    run("cargo", "package", "--locked", cwd=repo, env=build_env)
    name = f"cofferdb-{VERSION}"
    archive = work / "target/package" / (name + ".crate")
    # Cargo uses a zero timestamp and the package basename as its only gzip header field.
    header = b"\x1f\x8b\x08\x08\x00\x00\x00\x00\x02\xff" + archive.name.encode("ascii") + b"\x00"
    require(archive.read_bytes().startswith(header), "unexpected crate gzip header")
    expected = set(run("cargo", "package", "--list", "--locked", cwd=repo, env=build_env).splitlines())
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            require(path.parts[:1] == (name,) and ".." not in path.parts and member.isfile()
                    and member.uid == member.gid == 0 and not (member.uname or member.gname or member.pax_headers),
                    "unexpected crate member or archive metadata")
            relative = path.relative_to(name).as_posix()
            require(relative not in names, "duplicate crate member")
            source_file = repo / ("Cargo.toml" if relative == "Cargo.toml.orig" else relative)
            expected_mode = 0o755 if source_file.is_file() and source_file.stat().st_mode & 0o111 else 0o644
            require(member.mode == expected_mode and member.mtime == 1153704088, "unexpected crate member attributes")
            names.add(relative)
            payload = package.extractfile(member).read()
            if relative == ".cargo_vcs_info.json":
                vcs = json.loads(payload)
                require(vcs["git"] == {"sha1": args.commit} and vcs["path_in_vcs"] == "", "crate source differs")
            elif relative == "Cargo.toml":
                normalized = tomllib.loads(payload.decode())["package"]
                require(normalized["version"] == VERSION and normalized["license"] == "Apache-2.0"
                        and not PRIVATE_DATA.search(payload), "normalized crate manifest differs")
            else:
                source = repo / ("Cargo.toml" if relative == "Cargo.toml.orig" else relative)
                require(source.is_file() and not source.is_symlink() and source.read_bytes() == payload,
                        "crate differs from reviewed source")
        require(names == expected, "crate members differ from Cargo package list")
        package.extractall(work, filter="data")
    unpacked = work / name
    readme = (unpacked / "README.md").read_text(encoding="utf-8")
    require(f'cofferdb = "={VERSION}"' in readme, "crate README dependency differs")
    check_guide_links(readme)
    for language in ("Rust", "Python"):
        guide = unpacked / "docs" / f"{language.lower()}.md"
        heading = f"# {language} {VERSION} programming guide"
        require(guide.is_file() and guide.read_text(encoding="utf-8").splitlines()[:1] == [heading],
                f"crate missing the {language} guide for {VERSION}")
    maintenance_suites = ["maintenance_files", "maintenance_process"]
    for suite in maintenance_suites:
        log = run("cargo", "test", "--locked", "--features", "native-test-support",
                  "--test", suite, cwd=unpacked, env=build_env)
        require(re.search(r"test result: ok\. [1-9]\d* passed; 0 failed; 0 ignored;", log),
                f"native maintenance suite missing or incomplete: {suite}")
    example = json.loads(run("cargo", "run", "--locked", "--example", "embedded", cwd=unpacked, env=build_env))
    require(example["results"][0]["chunk_external_id"] == "wal#commit", "embedded Rust example query differs")
    run("cargo", "build", "--locked", "--bin", "coffer", cwd=unpacked, env=build_env)
    coffer = work / "target/debug/coffer"
    require("AArch64" in run("readelf", "-h", coffer), "CLI architecture differs")
    oracle = fixture_oracle(args.fixture, "old_database")
    observed_check = json.loads(run(coffer, "check", "fixture.coffer", "--json", cwd=args.fixture))["result"]
    require(observed_check == oracle["cli_check"] and sha(args.fixture / "fixture.coffer") == oracle["sha256"],
            "native ARM64 CLI check differs on the shared compatibility fixture")
    snapshot = work / "backup.coffer"
    with tempfile.TemporaryDirectory(dir=work) as original:
        db = Path(original) / "notes.coffer"
        run(coffer, "init", db)
        run(coffer, "import", db, "--manifest", unpacked / "examples/data/markdown/import_manifest.json", "--json")
        check_database(coffer, db)
        run(coffer, "snapshot", db, snapshot)
    restore_dir = work / "restored"
    restore_dir.mkdir()
    copy = restore_dir / "restored.coffer"
    shutil.copyfile(snapshot, copy)
    check_database(coffer, copy)
    recovered = cli_recovered_contents(coffer, copy)
    require(sha(copy) == sha(snapshot), "restored database bytes changed")
    require(source_identity(repo, args.commit) == identity, "checkout changed during crate verification")
    output = work / "release-crate"
    output.mkdir()
    shutil.copyfile(archive, output / archive.name)
    require(sha(output / archive.name) == sha(archive), "staged crate differs")
    report = output / "result.json"
    write_json(report, {**identity, "actions": actions_identity(), "job": os.environ.get("GITHUB_JOB"), "rust": run("rustc", "--version"),
               "host": "aarch64-unknown-linux-gnu", "runner_image": os.environ.get("ImageVersion", ""),
               "system": platform.release(), "glibc": run("getconf", "GNU_LIBC_VERSION"),
               "crate": archive.name, "sha256": sha(archive), "cli_sha256": sha(coffer),
               "fixture_sha256": oracle["sha256"], "cli_check": observed_check,
               "recovered_contents": recovered, "maintenance_suites": maintenance_suites, "snapshot_sha256": sha(snapshot), "stats": EXPECTED, "chunk": "wal#commit"})
    check_egress(report)


def make_fixtures(work, repo, commit):
    identity = source_identity(repo, commit)
    archive = download(OLD_CRATE, work)
    imported = work / "cofferdb-0.1.4"
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
    check_database(coffer, old)
    run(coffer, "snapshot", old, snapshot)
    current_target = work / "candidate-target"
    run("cargo", "build", "--locked", "--bin", "coffer", "--target-dir", current_target, cwd=repo)
    current_cli = current_target / "debug/coffer"
    fixtures = ((old, old_dir, "old_database"), (snapshot, snapshot_dir, "standalone_snapshot"))
    for fixture, directory, kind in fixtures:
        staged = directory / "fixture.coffer"
        shutil.copyfile(fixture, staged)
        before_check = sha(staged)
        cli_check = json.loads(run(current_cli, "check", staged.name, "--json", cwd=directory))["result"]
        require(cli_check["status"] == "ok" and sha(staged) == before_check, "candidate CLI fixture check failed")
        evidence = directory / "fixture.json"
        write_json(evidence, {**identity, "crate": {"url": OLD_CRATE[0], "sha256": sha(archive)},
                              "job": os.environ.get("GITHUB_JOB"), "machine": platform.machine(),
                              "runner_image": os.environ.get("ImageVersion", ""),
                              "stats": EXPECTED, "chunk": "wal#commit", "sha256": before_check, "cli_check": cli_check,
                              "kind": kind, "actions": actions_identity()})
        check_egress(staged)
        check_egress(evidence)


def restore(args):
    work = args.work.resolve()
    stage(work, "restore-inputs")
    host = environment(args.python, args.arch, args.versions)
    target = (host["system"], host["machine"])
    tags = {("Darwin", "x86_64"): "macosx_15_0_x86_64", ("Linux", "aarch64"): "manylinux_2_28_aarch64"}
    require(target in tags, "native Intel macOS or Linux ARM64 recovery required")
    manifest = json.loads((args.wheels / "result.json").read_text(encoding="utf-8"))
    require(manifest["commit"] == args.commit, "restore input is from a different commit")
    require(manifest["actions"] == actions_identity(), "restore input is from a different workflow run")
    require(sha(Path(__file__)) == manifest["acceptance_script_sha256"], "restore assertion differs from build")
    require(manifest["environment"]["system"] == host["system"] and
            manifest["environment"]["machine"] == host["machine"], "restore input platform differs")
    if host["system"] == "Linux":
        require(manifest["environment"]["image"] == host["image"], "restore image differs")
    records = manifest["artifacts"]
    require(len(records) == 3, "restore needs all three platform wheels")
    stage(work, "pip-download")
    pip = download(PIP, work)
    restored = {}
    for python, version, record in zip(args.python, args.versions, records, strict=True):
        abi = f"cp{version.split('.')[0]}{version.split('.')[1]}"
        wheel = f"cofferdb-{VERSION}-{abi}-{abi}-{tags[target]}.whl"
        require(record["wheel"] == wheel and record["python"] == version, "restore wheel or Python differs")
        executable = install(python, work, abi, pip, args.wheels / wheel, record)
        stage(work, "snapshot-check", abi)
        restored[wheel] = json.loads(run(
            executable, "-I", "-B", Path(__file__).resolve(), "inspect", record["native_sha256"],
            "--fixture", args.snapshot, "--fixture-kind", "standalone_snapshot", "--copy-to", work / f"restore-{abi}", cwd=work))
    stage(work, "restore-report")
    output = work / "verified"
    output.mkdir()
    report = output / "restore.json"
    write_json(report, {
        "commit": args.commit, "environment": host, "snapshot_sha256": sha(args.snapshot / "fixture.coffer"),
        "actions": actions_identity(), "tree": manifest["tree"], "job": os.environ.get("GITHUB_JOB"), "restored": restored,
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
    accept.add_argument("--arch", choices=["X64", "ARM64"], default="X64")
    accept.add_argument("--fixture", type=Path)
    crate = actions.add_parser("crate")
    crate.add_argument("work", type=Path)
    crate.add_argument("--repo", type=Path, required=True)
    crate.add_argument("--commit", required=True)
    crate.add_argument("--fixture", type=Path, required=True)
    fixtures = actions.add_parser("fixtures")
    fixtures.add_argument("work", type=Path)
    fixtures.add_argument("--repo", type=Path, required=True)
    fixtures.add_argument("--commit", required=True)
    recovery = actions.add_parser("restore")
    recovery.add_argument("--arch", choices=["X64", "ARM64"], default="X64")
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
        host = environment(args.python, args.arch, args.versions)
        require(host["system"] == "Linux", "separate acceptance requires baseline Linux")
        accept_build(args.repo.resolve(), args.commit, host, args.built.resolve(), args.work.resolve(), args.python, args.fixture)
    elif args.action == "crate":
        check_crate(args)
    elif args.action == "fixtures":
        make_fixtures(args.work, args.repo, args.commit)
    elif args.action == "restore":
        restore(args)
    else:
        inspect_installed(args.native_sha256)
        if args.fixture:
            print(json.dumps(read_fixture(args.fixture, args.copy_to, args.fixture_kind)))


if __name__ == "__main__":
    main()
