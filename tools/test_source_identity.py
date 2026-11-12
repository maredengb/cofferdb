"""Offline integration checks for the current package repository and Git source."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import tomllib
import unittest

SPEC = importlib.util.spec_from_file_location("prepare_source", Path(__file__).with_name("prepare_source.py"))
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)
REPOSITORY = "https://github.com/maredengb/cofferdb"


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cofferdb-source-identity-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.git("init", "-q")
        (self.root / "source.txt").write_text("Synthetic public source fixture\n")
        self.git("add", "source.txt")
        self.git("commit", "-qm", "Synthetic source fixture")
        self.commit = self.git("rev-parse", "HEAD")
        self.git("remote", "add", "origin", REPOSITORY + ".git")

    def git(self, *args):
        return subprocess.check_output([
            "git", "-c", "user.name=Synthetic Fixture",
            "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
            "-c", "core.hooksPath=/dev/null", "-C", str(self.root), *args],
            stderr=subprocess.DEVNULL, text=True).strip()

    def test_current_transport_forms_bind_exact_clean_commit_and_tree(self):
        for remote in (REPOSITORY, REPOSITORY + ".git", "git@github.com:maredengb/cofferdb.git"):
            with self.subTest(remote=remote):
                self.git("remote", "set-url", "origin", remote)
                self.assertEqual(source.source_identity(self.root, self.commit, REPOSITORY),
                    {"commit": self.commit, "tree": self.git("rev-parse", "HEAD^{tree}"),
                     "kind": "git_checkout"})

    def test_legacy_or_foreign_origin_does_not_match_current_package(self):
        for remote in ("https://github.com/maredengb/coffered.git",
                       "https://github.com/other/cofferdb.git"):
            with self.subTest(remote=remote):
                self.git("remote", "set-url", "origin", remote)
                with self.assertRaisesRegex(ValueError, "origin does not match"):
                    source.source_identity(self.root, self.commit, REPOSITORY)

    def test_dirty_or_different_commit_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, "not HEAD"):
            source.source_identity(self.root, "f" * 40, REPOSITORY)
        (self.root / "untracked.txt").write_text("Synthetic extra input\n")
        with self.assertRaisesRegex(ValueError, "must be clean"):
            source.source_identity(self.root, self.commit, REPOSITORY)

    def test_engine_binding_and_wheel_metadata_use_same_repository(self):
        product = Path(__file__).resolve().parents[1]
        engine = tomllib.loads((product / "Cargo.toml").read_text())["package"]
        binding = tomllib.loads((product / "bindings/python/Cargo.toml").read_text())["package"]
        wheel = tomllib.loads((product / "pyproject.toml").read_text())["project"]
        self.assertEqual(engine["repository"], REPOSITORY)
        self.assertEqual(binding["repository"], REPOSITORY)
        self.assertEqual(wheel["urls"]["Source"], REPOSITORY)
        self.assertEqual(engine["version"], binding["version"])
        self.assertEqual(engine["version"], wheel["version"])


if __name__ == "__main__":
    unittest.main()
