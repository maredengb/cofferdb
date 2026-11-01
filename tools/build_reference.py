"""Build offline Rust/Python references from prepared source and its installed wheel.

Install the pinned documentation requirements first. Builds and raw logs remain
outside the archive. No package or documentation is uploaded by this command.
"""

from __future__ import annotations

import argparse
import gzip
import html
from html.parser import HTMLParser
import importlib.metadata
import importlib.resources
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from urllib.parse import unquote, urlsplit, urlunsplit

import cofferdb
import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor

from prepare_source import checked_relative, digest, read_source


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attributes):
        for key, value in attributes:
            if value is not None and key == "id":
                self.ids.add(value)
            if value is not None and key in {"href", "src"}:
                self.links.append(value)


class LocalLinks(Extension):
    def __init__(self, relative, rendered):
        self.relative = relative
        self.rendered = rendered
        super().__init__()

    def extendMarkdown(self, parser):
        relative, rendered = self.relative, self.rendered

        class Rewrite(Treeprocessor):
            def run(self, root):
                for element in root.iter():
                    link = element.get("href")
                    if not link:
                        continue
                    url = urlsplit(link)
                    if url.scheme or url.netloc or not url.path:
                        continue
                    target = posixpath.normpath(posixpath.join(relative.parent.as_posix(), url.path))
                    if target in rendered:
                        element.set("href", urlunsplit(("", "", str(Path(url.path).with_suffix(".html")),
                                                        url.query, url.fragment)))

        parser.treeprocessors.register(Rewrite(parser), "local-markdown-links", 5)


def page(title, body):
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            "<style>body{max-width:76rem;margin:2rem auto;padding:0 1rem;font:17px/1.55 system-ui}"
            "pre{overflow:auto;background:#f3f5f7;padding:1rem}table{border-collapse:collapse}"
            "td,th{border:1px solid #bbb;padding:.4rem;text-align:left}a{color:#174aa0}</style>"
            f"<body>{body}</body></html>\n")


def run(arguments, cwd, logs, name, environment=None):
    env = dict(os.environ)
    if environment:
        env.update(environment)
    with (logs / f"{name}.stdout").open("wb") as stdout, (logs / f"{name}.stderr").open("wb") as stderr:
        subprocess.run(arguments, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=True)


def rust_reference(source, target, destination, cargo, logs, name):
    version = tomllib.loads((source / "Cargo.toml").read_text())["package"]["version"]
    env = {"CARGO_TARGET_DIR": str(target),
           "CARGO_ENCODED_RUSTDOCFLAGS": "\x1f".join(["-Dwarnings", f"--remap-path-prefix={source}=source"])}
    run([cargo, "doc", "--locked", "--offline", "--no-deps", "--lib", "--manifest-path",
         str(source / "Cargo.toml")], logs, logs, name, env)
    shutil.copytree(target / "doc", destination)
    crate = destination / "cofferdb"
    if any((crate / item).exists() for item in ("binding", "cli")):
        raise ValueError("doc-hidden implementation modules appeared in the public reference")
    symbols = []
    for path in sorted(crate.glob("*.html")):
        match = re.fullmatch(r"(struct|enum|type|fn|trait|constant)\.(.+)\.html", path.name)
        if not match:
            continue
        owner = f"cofferdb::{match[2]}"
        symbols.append({"name": owner, "kind": match[1], "page": f"cofferdb/{path.name}"})
        parser = Elements()
        parser.feed(path.read_text(encoding="utf-8"))
        for anchor in sorted(parser.ids):
            if anchor.startswith(("method.", "structfield.", "variant.", "associatedconstant.", "associatedtype.")):
                symbols.append({"name": owner + "::" + anchor.split(".", 1)[1],
                                "kind": anchor.split(".", 1)[0],
                                "page": f"cofferdb/{path.name}#{anchor}"})
    if not symbols:
        raise ValueError("rustdoc produced no supported symbols")
    (destination / "index.html").write_text(
        page(f"CofferDB Rust {version}", f"<h1>CofferDB Rust {html.escape(version)}</h1>"
             "<p><a href='cofferdb/index.html'>Crate API reference</a></p>"
             "<p><a href='public-symbols.json'>Public symbol inventory</a></p>"), encoding="utf-8")
    result = {"version": version, "symbols": symbols,
              "scope": "Rendered public rustdoc items, including derived/trait methods; doc-hidden items excluded"}
    (destination / "public-symbols.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def python_symbols(inventory):
    content = ["<h1>Installed Python public symbols</h1><p>Generated from the actual installed "
               "package and its packaged type stub. Type hints do not perform runtime coercion.</p><ul>"]
    for name in inventory["symbols"]:
        content.append(f"<li><a href='#{html.escape(name)}'>{html.escape(name)}</a></li>")
    content.append("</ul>")
    for name, value in inventory["symbols"].items():
        content.append(f"<section id='{html.escape(name)}'><h2>{html.escape(name)}</h2>")
        content.append("<pre>" + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre></section>")
    return page("CofferDB Python symbols", "\n".join(content))


def check_local_files(root):
    checked = 0
    for path in sorted(root.rglob("*.html")):
        parser = Elements()
        parser.feed(path.read_text(encoding="utf-8"))
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                raise ValueError(f"missing or escaping local reference: {path.relative_to(root)} -> {link}")
            checked += 1
    return checked


def copy_notices(public, tools, logs):
    notices = public / "third-party-notices"
    notices.mkdir()
    records = {}
    for name, version in tools.items():
        distribution = importlib.metadata.distribution(name)
        for relative in distribution.files or ():
            metadata = any(part.endswith((".dist-info", ".egg-info")) for part in relative.parts[:-1])
            named_notice = relative.name.lower().startswith(("license", "licence", "copying", "notice"))
            if not metadata or ("licenses" not in relative.parts and not named_notice):
                continue
            if "__pycache__" in relative.parts or relative.suffix.lower() in {".py", ".pyc", ".pyo", ".so", ".pyd", ".dll"}:
                continue
            target = notices / "python" / f"{name}-{version}" / checked_relative(str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            data = Path(distribution.locate_file(relative)).read_bytes()
            target.write_bytes(data)
            records[target.relative_to(public).as_posix()] = digest(data)
    sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], cwd=logs, text=True).strip())
    origin = sysroot / "share/doc/rust"
    if not (origin / "COPYRIGHT.html").is_file() or not (origin / "licenses").is_dir():
        raise ValueError("selected Rust toolchain copyright and license materials are required for rustdoc assets")
    target = notices / "rustdoc"
    target.mkdir()
    shutil.copyfile(origin / "COPYRIGHT.html", target / "COPYRIGHT.html")
    shutil.copytree(origin / "licenses", target / "licenses")
    (notices / "README.txt").write_text(
        "Available upstream notices from the selected installed documentation distributions.\n"
        "Only distribution metadata notices are copied; package code and caches are excluded.\n"
        "Rust's complete upstream inventory includes components beyond these generated pages;\n"
        "the presence of a license text alone does not mean every listed component is bundled.\n"
        "The SDK's separate source dependency inventory is in source/bindings/python/licenses/.\n")
    return records


def build(source, legacy, output, cargo, legacy_commit):
    source, legacy = source.resolve(strict=True), legacy.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink() or any(output.resolve().is_relative_to(root) for root in (source, legacy)):
        raise ValueError("output must be a new directory outside both source trees")
    source_file = Path("bindings/python/python/cofferdb/_source.json")
    metadata = json.loads(read_source(source, source_file))
    actual_source = json.loads(importlib.resources.files("cofferdb").joinpath("_source.json").read_text())
    if actual_source != metadata or cofferdb.__version__ != metadata["sdk_version"] or cofferdb.engine_version != metadata["engine_version"]:
        raise ValueError("installed wheel and prepared source identity do not match")
    inputs = json.loads(read_source(source, Path("SOURCE-INPUTS.json")))
    if digest(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()) != metadata["source_input_sha256"]:
        raise ValueError("source input map does not match package metadata")
    for relative, expected in inputs.items():
        if digest(read_source(source, checked_relative(relative))) != expected["sha256"]:
            raise ValueError("prepared input changed after source identity was recorded")
    requirements = read_source(source, Path("bindings/python/requirements-docs.txt")).decode()
    tools = {}
    for line in requirements.splitlines():
        if line and not line.startswith("#"):
            name, expected = line.split("==")
            tools[name] = importlib.metadata.version(name)
            if tools[name] != expected:
                raise ValueError(f"documentation tool version mismatch: {name}")
    if tomllib.loads((legacy / "Cargo.toml").read_text())["package"]["version"] != "0.1.0":
        raise ValueError("legacy source must be Rust 0.1.0")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=legacy, text=True).strip()
    if head != legacy_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=legacy):
        raise ValueError("legacy reference source is not the requested clean commit")
    output.mkdir(parents=True)
    logs = output / "build-logs"
    logs.mkdir()
    version = metadata["sdk_version"]
    public = output / f"cofferdb-{version}-reference"
    public.mkdir()
    copied = public / "source"
    for name in list(inputs) + ["SOURCE-INPUTS.json", source_file.as_posix()]:
        relative = checked_relative(name)
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(read_source(source, relative))
    current = rust_reference(source, output / "rust-current-target", public / f"rust-{version}", cargo, logs, "rust-current")
    previous = rust_reference(legacy, output / "rust-legacy-target", public / "rust-0.1.0", cargo, logs, "rust-legacy")
    run([sys.executable, "-m", "sphinx", "-n", "-W", "-b", "html", "-d", str(output / "sphinx-doctrees"),
         str(source / "bindings/python/docs"), str(public / f"python-{version}")], logs, logs, "sphinx")
    run([sys.executable, str(source / "bindings/python/tools/api_inventory.py"),
         str(public / "python-public-symbols.json")], logs, logs, "python-inventory")
    inventory = json.loads((public / "python-public-symbols.json").read_text())
    (public / "python-public-symbols.html").write_text(python_symbols(inventory), encoding="utf-8")
    documents = [Path("README.md"), Path("bindings/python/README.md")]
    documents += sorted(Path("docs") / path.name for path in (copied / "docs").glob("*.md"))
    rendered = {name.as_posix() for name in documents}
    for relative in documents:
        body = markdown.markdown(read_source(copied, relative).decode("utf-8"),
                                 extensions=["tables", "fenced_code", "toc", LocalLinks(relative, rendered)])
        (copied / relative.with_suffix(".html")).write_text(page(relative.stem, body), encoding="utf-8")
    body = (f"<h1>CofferDB {html.escape(version)} offline reference</h1>"
            "<p>Experimental API documentation. See the matching release record for artifact availability and package-specific validation results. Generated references do not imply full version or platform acceptance.</p>"
            "<ul><li><a href='source/docs/README.html'>Versioned programming guides</a></li>"
            f"<li><a href='rust-{version}/cofferdb/index.html'>Rust {html.escape(version)} API</a></li>"
            "<li><a href='rust-0.1.0/cofferdb/index.html'>Rust 0.1.0 API, generated from its own source</a></li>"
            f"<li><a href='python-{version}/index.html'>Python {html.escape(version)} API and types</a></li>"
            "<li><a href='python-public-symbols.html'>Actual Python symbol inventory</a></li>"
            "<li><a href='reference-manifest.json'>Source and build identities</a></li></ul>"
            "<p>Local navigation, source, styles and search assets are included. External upstream/standard-library links require a connection.</p>")
    (public / "index.html").write_text(page("CofferDB offline reference", body), encoding="utf-8")
    notices = copy_notices(public, tools, logs)
    manifest = {"schema_version": 1, "source": metadata, "legacy_rust_commit": head,
                "documentation_tools": tools, "python_version": sys.version.split()[0],
                "rustc": subprocess.check_output(["rustc", "--version"], cwd=logs, text=True).strip(),
                "rust_public_items": len(current["symbols"]), "legacy_rust_public_items": len(previous["symbols"]),
                "python_exports": len(inventory["symbols"]), "documentation_notice_hashes": notices,
                "status": "development_reference_not_version_acceptance"}
    (public / "reference-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["checked_local_file_links"] = check_local_files(public.resolve())
    manifest["link_check_scope"] = "Static HTML file targets; external URLs, JavaScript-generated links and arbitrary fragments not certified"
    manifest["files"] = {str(p.relative_to(public)): digest(p.read_bytes()) for p in sorted(public.rglob("*"))
                         if p.is_file() and p.name != "reference-manifest.json"}
    (public / "reference-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output / f"cofferdb-{version}-reference.tar.gz"
    def normalized_header(info):
        if not (info.isfile() or info.isdir()):
            raise ValueError("the reference archive contains a non-regular source entry")
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.pax_headers = {}
        return info

    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
            bundle.add(public, arcname=public.name, filter=normalized_header)
    (output / "SHA256SUMS").write_text(f"{digest(archive.read_bytes())}  {archive.name}\n")
    return {"archive": archive.name, "sha256": digest(archive.read_bytes()),
            "checked_local_file_links": manifest["checked_local_file_links"], "source": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rust-v010-source", type=Path, required=True)
    parser.add_argument("--rust-v010-commit", default="6498bd8730973c6c247b831c34fc25299ac51f30")
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.rust_v010_source, args.output, args.cargo, args.rust_v010_commit), indent=2))


if __name__ == "__main__":
    main()
