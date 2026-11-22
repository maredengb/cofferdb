"""Compare an installed SDK with its matching CLI, each on its own fixed view."""

import argparse
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

import cofferdb


def run(cli, *args, json_output=True):
    command = [cli, *map(str, args)]
    if json_output:
        command.append("--json")
    output = subprocess.check_output(command, text=True)
    if not json_output:
        return output
    envelope = json.loads(output)
    assert envelope["command"] == args[0], args[0]
    return envelope["result"]


def compare(cli, old_cli=None, old_python=None):
    fixtures = Path(__file__).resolve().parents[1] / "tests/test_maintenance.py"
    setup = runpy.run_path(str(fixtures))["setup_notes"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        main = root / "main"
        with cofferdb.Database.create(str(main)) as db:
            setup(db)
            commit = int(db.stats()["last_commit_id"])
            bak = db.snapshot(str(root / "python"))
        human_commands = []
        for command, arguments, field in (
            ("check", (), "status"),
            ("repair", ("--truncate-tail",), "repaired"),
            ("rebuild-index", (), "index_suffix_status"),
            ("compact", (), "after_file_size_bytes"),
            ("snapshot", (root / "human-backup",), "package"),
        ):
            source = root / command
            shutil.copyfile(main, source)
            lines = run(cli, command, source, *arguments, json_output=False).splitlines()
            assert lines[0] == command and any(line.startswith(f"  {field}: ") for line in lines[1:]), command
            human_commands.append(command)
        json_commands = []

        def checked(command, *args):
            result = run(cli, command, *args)
            json_commands.append(command)
            return result

        check = checked("check", main, "--verify-indexes")
        assert check == cofferdb.check(str(main))
        repair = checked("repair", main, "--truncate-tail")
        assert repair["action"] == "truncate_tail" and repair["repaired"] is False
        assert repair["bytes_removed"] == 0
        assert repair["before_file_size_bytes"] == repair["after_file_size_bytes"] == main.stat().st_size
        backup = checked("snapshot", main, root / "cli")
        assert all(Path(receipt["source_path"]).samefile(main) for receipt in (bak, backup))
        assert Path(bak["output_path"]).samefile(root / "python")
        assert Path(backup["output_path"]).samefile(root / "cli")
        package = backup["package"]
        for key, value in package.items():
            if key in ("last_valid_commit", "critical_end_offset", "size_bytes"):
                value = int(value)
            assert bak[key] == value, key
        for command, method in (
            ("rebuild-index", "rebuild_index"),
            ("compact", "compact"),
        ):
            for name in ("cli-copy", "python-copy"):
                shutil.copyfile(main, root / name)
            cli_result = checked(command, root / "cli-copy")
            with cofferdb.Database.open(str(root / "python-copy")) as db:
                result = getattr(db, method)()
            for summary in (cli_result, result):
                assert type(summary["duration_ms"]) is int and summary["duration_ms"] >= 0
            for k, v in result.items():
                if k != "duration_ms":
                    assert v == cli_result[k], (command, k, v, cli_result[k])
        with cofferdb.Database.open(str(main)) as db:
            try:
                db.rebuild_index("missing")
            except cofferdb.CofferError as error:
                missing = error.code
            else:
                raise AssertionError("missing collection accepted")
        rejected = subprocess.run(
            [cli, "rebuild-index", str(main), "--collection", "missing", "--json"],
            capture_output=True,
            text=True,
        )
        assert (
            rejected.returncode
            and json.loads(rejected.stderr)["error"]["code"] == missing
        )
        snapshots = [(root / "python", commit), (root / "cli", commit)]
        if old_cli:
            legacy = root / "legacy-main"
            setup_code = "import runpy,sys,cofferdb; assert cofferdb.__version__ != sys.argv[3]; db=cofferdb.Database.create(sys.argv[2]); runpy.run_path(sys.argv[1])['setup_notes'](db); print(db.stats()['last_commit_id']); db.close()"
            legacy_commit = int(
                subprocess.check_output(
                    [
                        old_python,
                        "-I",
                        "-c",
                        setup_code,
                        str(fixtures),
                        str(legacy),
                        cofferdb.__version__,
                    ],
                    text=True,
                )
            )
            run(old_cli, "snapshot", legacy, root / "legacy")
            legacy.unlink()
            snapshots.append((root / "legacy", legacy_commit))
        main.unlink()
        for snapshot, expected in snapshots:
            with tempfile.TemporaryDirectory() as isolated:
                copied = Path(isolated) / "only.coffer"
                shutil.copyfile(snapshot, copied)
                subprocess.run(
                    [sys.executable, "-I", str(fixtures), str(copied), str(expected)],
                    cwd=isolated,
                    check=True,
                )

        return {"human_commands": sorted(human_commands), "json_commands": sorted(json_commands),
                "restored": ["python", "cli"] + (["legacy"] if old_cli else []),
                "missing_collection": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("--old-cli", type=Path)
    parser.add_argument("--old-python", type=Path)
    args = parser.parse_args()
    if bool(args.old_cli) != bool(args.old_python):
        parser.error("supply both legacy inputs together")
    print(json.dumps(compare(args.cli.resolve(), args.old_cli, args.old_python), sort_keys=True))
