"""Prepare a self-contained, source-identified CofferDB build directory.

Run with Python 3.12+ and the locked Cargo dependencies available locally.
This copies source files, never Git history, and does not publish anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import tomllib


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("source paths must remain inside the source tree")
    return Path(*path.parts)


def read_source(root: Path, relative: Path) -> bytes:
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("source preparation does not follow symbolic links")
    if not path.is_file():
        raise ValueError(f"missing source file: {relative.as_posix()}")
    return path.read_bytes()


def command(arguments: list[str], cwd: Path) -> str:
    return subprocess.check_output(arguments, cwd=cwd, text=True).strip()


def license_inventory(root: Path) -> dict:
    binding = Path("bindings/python")
    lock_data = read_source(root, binding / "Cargo.lock")
    lock = tomllib.loads(lock_data.decode("utf-8"))
    inventory = json.loads(read_source(root, binding / "licenses/inventory.json"))
    if inventory["cargo_lock_sha256"] != digest(lock_data):
        raise ValueError("license inventory does not identify the current binding Cargo.lock")
    packages = {(row["name"], row["version"], row["source"])
                for row in lock["package"] if row.get("source", "").startswith("registry+")}
    declared = {(row["name"], row["version"], row["source"])
                for row in inventory["dependencies"]}
    if len(declared) != len(inventory["dependencies"]) or packages != declared:
        raise ValueError("license inventory does not cover the exact locked registry dependencies")
    for row in inventory["dependencies"]:
        for name, expected in row["license_files"].items():
            if digest(read_source(root, binding / checked_relative(name))) != expected:
                raise ValueError("an upstream license notice does not match its recorded hash")
    return inventory


def source_identity(root: Path, commit: str | None, repository: str) -> dict:
    if commit is None:
        return {"commit": None, "tree": None, "kind": "unversioned_export"}
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("--commit must be a complete lowercase Git SHA-1")
    actual_root = Path(command(["git", "rev-parse", "--show-toplevel"], root)).resolve()
    if actual_root != root:
        raise ValueError("versioned source must be the Git repository root")
    if command(["git", "status", "--porcelain", "--untracked-files=all"], root):
        raise ValueError("versioned source must be clean, including untracked files")
    if command(["git", "rev-parse", "HEAD"], root) != commit:
        raise ValueError("the requested source commit is not HEAD")
    remote = command(["git", "remote", "get-url", "origin"], root)
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.removeprefix("git@github.com:")
    if remote.removesuffix(".git") != repository.removesuffix(".git"):
        raise ValueError("source origin does not match the package repository")
    return {"commit": commit, "tree": command(["git", "rev-parse", "HEAD^{tree}"], root),
            "kind": "git_checkout"}


def source_paths(root: Path, cargo: str) -> list[Path]:
    paths = {Path("pyproject.toml")}
    for relative_manifest in (Path("Cargo.toml"), Path("bindings/python/Cargo.toml")):
        # An absolute manifest and a neutral cwd prevent unrelated ancestor
        # workspace configuration from changing the selected product inputs.
        listed = command([cargo, "package", "--list", "--locked", "--offline", "--allow-dirty",
                          "--manifest-path", str(root / relative_manifest)], Path(tempfile.gettempdir()))
        for name in listed.splitlines():
            relative = relative_manifest.parent / checked_relative(name)
            if relative.name in {"Cargo.toml.orig", ".cargo_vcs_info.json"}:
                continue  # Cargo-generated archive metadata, not a source input.
            read_source(root, relative)
            paths.add(relative)
    # Cargo's Git-free file discovery omits hidden files. Preserve the explicit
    # source-distribution additions from the canonical packaging configuration.
    project = tomllib.loads(read_source(root, Path("pyproject.toml")).decode("utf-8"))
    for item in project["tool"]["maturin"].get("include", []):
        if isinstance(item, str):
            pattern, formats = item, ["sdist", "wheel"]
        else:
            pattern, formats = item["path"], item.get("format", ["sdist", "wheel"])
        if isinstance(formats, str):
            formats = [formats]
        if "sdist" not in formats:
            continue
        checked_relative(pattern)
        matched = False
        for path in root.glob(pattern):
            if path.is_symlink():
                raise ValueError("source preparation does not follow symbolic links")
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            read_source(root, relative)
            paths.add(relative)
            matched = True
        if not matched:
            raise ValueError("an explicit source-distribution include has no regular files")
    return sorted(paths)


def prepare(root: Path, output: Path, cargo: str, commit: str | None) -> dict:
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory")
    # Resolve parents too: a symlink must not conceal an output inside source.
    if output.resolve().is_relative_to(root):
        raise ValueError("output must be outside the source tree")
    project = tomllib.loads(read_source(root, Path("pyproject.toml")).decode("utf-8"))
    engine = tomllib.loads(read_source(root, Path("Cargo.toml")).decode("utf-8"))["package"]
    binding = tomllib.loads(read_source(root, Path("bindings/python/Cargo.toml")).decode("utf-8"))["package"]
    if project["project"]["version"] != binding["version"] or binding["version"] != engine["version"]:
        raise ValueError("SDK, native binding and engine versions must match")
    repository = engine["repository"]
    origin = source_identity(root, commit, repository)
    inventory = license_inventory(root)
    paths = source_paths(root, cargo)
    source_name = Path("bindings/python/python/cofferdb/_source.json")
    if source_name in paths or (root / source_name).exists():
        raise ValueError("prepare from checkout sources, not an already prepared source export")
    inputs = {}
    for relative in paths:
        data = read_source(root, relative)
        mode = (root / relative).stat().st_mode & 0o777
        inputs[relative.as_posix()] = {"sha256": digest(data), "size": len(data), "mode": mode}
    metadata = {
        "schema_version": 1,
        "sdk_version": binding["version"],
        "engine_version": engine["version"],
        "repository": repository,
        "source": origin,
        "source_input_sha256": digest(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()),
        "binding_cargo_lock_sha256": inventory["cargo_lock_sha256"],
        "engine_cargo_lock_sha256": digest(read_source(root, Path("Cargo.lock"))),
        "source_license_inventory_sha256": digest(read_source(root, Path("bindings/python/licenses/inventory.json"))),
    }
    output.mkdir(parents=True)
    for relative in paths:
        data = read_source(root, relative)
        if digest(data) != inputs[relative.as_posix()]["sha256"]:
            raise ValueError("source changed during preparation; preserve this incomplete output")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        shutil.copymode(root / relative, destination)
    source_file = output / source_name
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # The complete source-input map is source material. Its hash deliberately
    # excludes itself and the generated metadata to avoid a self-reference.
    (output / "SOURCE-INPUTS.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--commit", help="exact clean product-repository HEAD, when available")
    args = parser.parse_args()
    try:
        metadata = prepare(args.source, args.output, args.cargo, args.commit)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"source preparation failed: {error}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
