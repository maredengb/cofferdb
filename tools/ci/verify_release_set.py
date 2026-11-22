"""Verify the exact crate, wheels and recovery reports produced by one release run."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from release_wheel import MAINTENANCE_TESTS, EXPECTED, EXPECTED_RECOVERY, VERSION, actions_identity, fixture_oracle, require, sha, source_identity


def read(path):
    return json.loads(path.read_bytes())


def verify_environment(host, group, machine, run):
    linux = group.startswith("linux-")
    require(host["system"] == ("Linux" if linux else "Darwin") and host["machine"] == machine
            and host["actions"] == run and host["runner_image"], "runner or system differs")
    if linux:
        require(host["version"] == "glibc 2.28" and
                host["image"].startswith(f"quay.io/pypa/manylinux_2_28_{machine}@sha256:"), "Linux baseline differs")
    else:
        require(host["version"].startswith("15.") and host["image"] == host["runner_image"], "macOS baseline differs")
    for minor, runtime in zip((12, 13, 14), host["python"], strict=True):
        require(runtime["system"] == host["system"] and runtime["machine"] == machine
                and runtime["implementation"] == "cpython" and runtime["bits"] == 64 and runtime["gil"]
                and runtime["version"].startswith(f"3.{minor}.") and runtime["suffix"].startswith(f".cpython-3{minor}-"),
                "CPython environment differs")


def verify(directory, source_sha, tree):
    platforms = {
        "linux-aarch64": ("manylinux_2_28_aarch64", "aarch64"),
        "linux-x86_64": ("manylinux_2_28_x86_64", "x86_64"),
        "macos-arm64": ("macosx_15_0_arm64", "arm64"),
        "macos-x86_64": ("macosx_15_0_x86_64", "x86_64"),
    }
    prefix = "cofferdb-" + VERSION.replace(".", "") + "-"
    reports = {}
    prepared = set()
    run = actions_identity()
    require(run is not None, "GitHub run identity required")
    fixture_groups = {"cofferdb-014-old-database": "old_database", "cofferdb-014-snapshot": "standalone_snapshot"}
    expected_groups = {prefix+n for n in (*platforms, "crate", "arm-snapshot-acceptance", "intel-snapshot-acceptance")}
    require({p.name for p in directory.iterdir()} == expected_groups | fixture_groups.keys(), "artifact groups differ")
    oracles = {}
    for name, kind in fixture_groups.items():
        oracle = fixture_oracle(directory / name, kind)
        require(oracle["commit"] == source_sha and oracle["tree"] == tree and oracle["job"] == "fixtures"
                and oracle["machine"] == "x86_64" and oracle["runner_image"], "compatibility sample source differs")
        oracles[kind] = {field: oracle[field] for field in ("sha256", "stats", "chunk", "cli_check")}
        oracles[kind]["recovered_contents"] = EXPECTED_RECOVERY
    for group, (tag, machine) in platforms.items():
        root = directory / (prefix + group)
        report = read(root / "result.json")
        require(report["commit"] == source_sha and report["tree"] == tree and report["actions"] == run,
                "wheel group source or run differs")
        host = report["environment"]
        verify_environment(host, group, machine, run)
        linux = group.startswith("linux-")
        tools = report["build_tools"]
        triple = machine + "-unknown-linux-gnu" if linux else ("aarch64" if machine == "arm64" else machine) + "-apple-darwin"
        require(report["job"] == ("linux" if linux else "macos") and tools["host"] == triple and
                report["rust"] == tools["rustc"] and report["cargo"] == tools["cargo"]
                and report["maturin"] == "1.10.2" and report["pip"] == "25.3", "wheel build tools differ")
        if linux:
            require(tools["rust_channel_sha256"] == "03569b1886ceb5c05276b50c8431ab111de944cd6140fe1fa7d821dd8e0f29cf",
                    "Rust distribution differs")
        else:
            require(tools["xcode"] == "Xcode 16.0" and tools["sdk"] == "15.0", "macOS SDK differs")
        require(report["acceptance_script_sha256"] == sha(Path(__file__).with_name("release_wheel.py")),
                "wheel check script differs")
        if linux:
            require(report["linux_script_sha256"] == sha(Path(__file__).with_name("linux_release.sh")), "Linux driver differs")
        digest = report["source_input_sha256"]
        prepared.add(digest)
        wheel_hashes = {}
        for minor, row in zip((12, 13, 14), report["artifacts"], strict=True):
            abi = f"cp3{minor}"
            name = f"cofferdb-{VERSION}-{abi}-{abi}-{tag}.whl"
            require(row["wheel"] == name and sha(root / name) == row["sha256"], "wheel bytes differ")
            require(row["installed_native_verified"] and row["versions"] == [VERSION] * 3 and
                    row["installed_suite_exit_code"] == 0 and row["installed_tests"] >= row["check_tests"] > 0
                    and row["old_database"] == oracles["old_database"]
                    and row["python"] == host["python"][minor - 12]["version"], "installed evidence incomplete")
            require(set(row["maintenance"]) >= MAINTENANCE_TESTS, "missing per-wheel maintenance results")
            commands = ["check", "compact", "rebuild-index", "repair", "snapshot"]
            require(row["cli_comparison"] == {"human_commands": commands, "json_commands": commands,
                    "restored": ["python", "cli"], "missing_collection": "collection_not_found"}
                    and len(row["comparison_cli_sha256"]) == 64
                    and row["comparison_cli_sha256"] == tools["comparison_cli_sha256"],
                    "installed wheel CLI comparison or snapshot recovery differs")
            with zipfile.ZipFile(root / name) as archive:
                source = json.loads(archive.read("cofferdb/_source.json"))
                require(source["source"]["commit"] == source_sha and source["source"]["tree"] == tree
                        and source["source_input_sha256"] == digest, "wheel source differs")
                native, = [n for n in archive.namelist() if n.startswith("cofferdb/_native.") and n.endswith(".so")]
                require(hashlib.sha256(archive.read(native)).hexdigest() == row["native_sha256"], "native bytes differ")
            wheel_hashes[name] = row["sha256"]
        require({p.name for p in root.iterdir()} == {*wheel_hashes, "result.json", "SHA256SUMS"}, "unexpected wheel asset")
        expected_sums = "".join(f"{value}  {name}\n" for name, value in wheel_hashes.items())
        require((root / "SHA256SUMS").read_bytes() == expected_sums.encode(), "wheel checksum file differs")
        reports[group] = wheel_hashes
    require(len(prepared) == 1, "prepared source differs across platforms")
    for name, group in (("arm", "linux-aarch64"), ("intel", "macos-x86_64")):
        archive = directory / (prefix + name + "-snapshot-acceptance")
        require({p.name for p in archive.iterdir()} == {"restore.json"}, "unexpected recovery asset")
        receipt = read(archive / "restore.json")
        verify_environment(receipt["environment"], group, platforms[group][1], run)
        built_host = read(directory / (prefix + group) / "result.json")["environment"]
        require(receipt["environment"]["python"] == built_host["python"] and
                (name != "arm" or receipt["environment"]["image"] == built_host["image"]), "recovery runtime differs")
        require(receipt["actions"] == run and receipt["commit"] == source_sha and receipt["tree"] == tree
                and receipt["job"] == name + "-restore" and receipt["restore_exit_code"] == 0
                and receipt["wheels"] == reports[group] and receipt["snapshot_sha256"] == oracles["standalone_snapshot"]["sha256"]
                and receipt["restored"] == dict.fromkeys(reports[group], oracles["standalone_snapshot"]),
                "snapshot recovery receipt differs")
    crate_dir = directory / (prefix + "crate")
    crate = crate_dir / f"cofferdb-{VERSION}.crate"
    require({p.name for p in crate_dir.iterdir()} == {crate.name, "result.json"}, "unexpected crate asset")
    evidence = read(crate_dir / "result.json")
    require(evidence["actions"] == run and evidence["commit"] == source_sha and evidence["tree"] == tree
            and evidence["job"] == "arm-crate" and evidence["runner_image"]
            and evidence["host"] == "aarch64-unknown-linux-gnu" and evidence["stats"] == EXPECTED
            and evidence["fixture_sha256"] == oracles["old_database"]["sha256"]
            and evidence["cli_check"] == oracles["old_database"]["cli_check"]
            and evidence["recovered_contents"] == EXPECTED_RECOVERY
            and evidence["maintenance_suites"] == ["maintenance_files", "maintenance_process"]
            and evidence["chunk"] == "wal#commit" and evidence["crate"] == crate.name
            and evidence["sha256"] == sha(crate), "crate verification receipt differs")
    return {name: digest for group in reports.values() for name, digest in group.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    args = parser.parse_args()
    source = source_identity(args.repo, args.commit)
    verified = verify(args.directory, args.commit, source["tree"])
    with args.checksums.open("x", encoding="ascii", newline="\n") as output:
        for name, digest in sorted(verified.items()):
            output.write(f"{digest}  {name}\n")
    print("Verified the source crate, twelve installed wheels and both standalone restorations.")
