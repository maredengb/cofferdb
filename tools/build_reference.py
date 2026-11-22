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
        self.links = []
        self.redirect = False

    def handle_starttag(self, tag, attributes):
        if tag == "meta" and dict(attributes).get("http-equiv", "").lower() == "refresh":
            self.redirect = True
        for key, value in attributes:
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


class RustMembers(Elements):
    """Read rustdoc's field, variant, inherent impl and trait-declaration sections."""

    def __init__(self):
        super().__init__()
        self.section = ""
        self.members = {}

    def handle_starttag(self, tag, attributes):
        super().handle_starttag(tag, attributes)
        attrs = dict(attributes)
        anchor = attrs.get("id", "")
        classes = attrs.get("class", "").split()
        if tag == "h2" and "section-header" in classes:
            self.section = anchor
            return
        if not anchor or "trait-impl" in classes:
            return
        parts = anchor.split(".")
        kind, components = parts[0], parts[1:]
        if self.section == "variants":
            if tag == "span" and len(parts) == 4 and kind == "variant" and parts[2] == "field":
                kind, components = "variantfield", [parts[1], parts[3]]
            elif not (tag == "section" and len(parts) == 2 and kind == "variant" and kind in classes):
                return
        elif self.section == "fields":
            if not (tag == "span" and len(parts) == 2 and kind == "structfield" and kind in classes):
                return
        elif self.section in (
            "implementations", "required-methods", "provided-methods",
            "required-associated-types", "provided-associated-types",
            "required-associated-consts", "provided-associated-consts",
        ):
            if tag != "section" or len(parts) != 2 or kind not in (
                "method", "tymethod", "associatedtype", "associatedconstant"
            ):
                return
        else:
            # Standard traits, blanket implementations and dereferenced types have
            # rustdoc member anchors of their own. They do not belong to this declaration.
            return
        # A numeric suffix disambiguates an HTML fragment, never a Rust identifier.
        names = [re.sub(r"-[0-9]+$", "", part) for part in components]
        if all(part.isidentifier() or part.isdecimal() for part in names):
            self.members[anchor] = kind, "::".join(names)


def rust_symbols(crate):
    """Index actual public module pages. Private module paths contain only redirects."""
    symbols = []
    item_page = re.compile(r"(struct|enum|union|type|fn|trait|traitalias|constant|static|macro|attr|derive)\.(.+)\.html")
    for index in sorted(crate.rglob("index.html")):
        module = index.parent
        owner = "::".join((crate.name, *module.relative_to(crate).parts))
        module_page = Elements()
        module_page.feed(index.read_text(encoding="utf-8"))
        if module_page.redirect:
            continue
        symbols.append({"name": owner, "kind": "crate" if module == crate else "mod",
                        "page": index.relative_to(crate.parent).as_posix()})
        for path in sorted(module.glob("*.html")):
            match = item_page.fullmatch(path.name)
            if not match:
                continue
            parser = RustMembers()
            rendered = path.read_text(encoding="utf-8")
            parser.feed(rendered)
            # Re-exports may also leave compatibility redirects in public modules.
            if parser.redirect:
                continue
            name = owner + "::" + match[2]
            page = path.relative_to(crate.parent).as_posix()
            symbols.append({"name": name, "kind": match[1], "page": page})
            for anchor, (kind, member) in sorted(parser.members.items()):
                symbols.append({"name": name + "::" + member, "kind": kind,
                                "page": f"{page}#{anchor}"})
    if not symbols:
        raise ValueError("rustdoc produced no crate reference")
    return symbols


def link_offline_guide(reference, guide, title, marker):
    """Add a local, version-matched guide link to a generated API page."""
    rendered = reference.read_text(encoding="utf-8")
    if marker not in rendered:
        raise ValueError(f"missing guide navigation insertion point in {reference.name}")
    href = html.escape(os.path.relpath(guide, reference.parent).replace(os.sep, "/"))
    navigation = f"<p><a href='{href}'>{html.escape(title)}</a></p>"
    replacement = navigation + marker if marker == "</main>" else marker + navigation
    reference.write_text(rendered.replace(marker, replacement, 1), encoding="utf-8")


def rust_reference(source, target, destination, cargo, logs, name, *, require_docs=False):
    version = tomllib.loads((source / "Cargo.toml").read_text())["package"]["version"]
    flags = ["-Dwarnings", f"--remap-path-prefix={source}=source"]
    if require_docs:
        flags.append("-Dmissing_docs")
    env = {"CARGO_TARGET_DIR": str(target), "CARGO_ENCODED_RUSTDOCFLAGS": "\x1f".join(flags)}
    run([cargo, "doc", "--locked", "--offline", "--no-deps", "--lib", "--manifest-path",
         str(source / "Cargo.toml")], logs, logs, name, env)
    shutil.copytree(target / "doc", destination)
    crate = destination / "cofferdb"
    if any((crate / item).exists() for item in ("binding", "cli")):
        raise ValueError("doc-hidden implementation modules appeared in the public reference")
    symbols = rust_symbols(crate)
    guide = destination.parent / "source/docs/rust.html"
    if require_docs:
        pages = {item["page"] for item in symbols if "#" not in item["page"]}
        for path in sorted(pages):
            link_offline_guide(destination / path, guide, "Rust programming guide", "</main>")
    (destination / "index.html").write_text(
        page(f"CofferDB Rust {version}", f"<h1>CofferDB Rust {html.escape(version)}</h1>"
             "<p><a href='cofferdb/index.html'>Crate API reference</a></p>"
             "<p><a href='public-symbols.json'>Public symbol inventory</a></p>"), encoding="utf-8")
    if require_docs:
        link_offline_guide(destination / "index.html", guide, "Rust programming guide", "</h1>")
    result = {"version": version, "symbols": symbols,
              "scope": "Public rustdoc items, fields, variants, inherent members and trait declarations; trait implementations and doc-hidden items excluded"}
    (destination / "public-symbols.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def python_symbols(inventory):
    esc = html.escape
    version = esc(inventory["sdk_version"])
    content = ["<h1>Installed Python public symbols</h1><p>Generated from the installed wheel and its "
               f"packaged type declarations. Browse the <a href='python-{version}/api.html'>native API</a>, "
               f"<a href='python-{version}/data.html'>data types</a> and "
               "<a href='source/docs/python.html'>Python guide</a> "
               "for complete workflows.</p><ul>"]
    for name in inventory["symbols"]:
        content.append(f"<li><a href='#{esc(name)}'>{esc(name)}</a></li>")
    content.append("</ul>")

    def description(value):
        return f"<div style='white-space:pre-wrap'>{esc(value)}</div>"

    def callable_doc(name, value):
        content.append(f"<pre>{esc(name + value['declared_signature'])}</pre>")
        content.append(description(value["doc"]))

    for name, value in inventory["symbols"].items():
        content.append(f"<section id='{esc(name)}'><h2>{esc(name)}</h2>")
        if value["kind"] == "native_function":
            callable_doc(name, value)
        else:
            content.append(description(value["doc"]))
        if value["kind"] == "type_alias":
            content.append(f"<pre>{esc(value['definition'])}</pre>")
        if value["kind"] == "version":
            content.append(f"<p>Value: <code>{esc(value['value'])}</code></p>")
        for member, method in value.get("methods", {}).items():
            full_name = f"{name}.{member}"
            content.append(f"<h3 id='{esc(full_name)}'>{esc(full_name)}</h3>")
            callable_doc(full_name, method)
        fields = value.get("fields", value.get("declared_attributes", {}))
        if fields:
            content.append("<table><thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead><tbody>")
            docs = value.get("field_docs", value.get("attribute_docs", {}))
            for key, annotation in fields.items():
                optional = " (optional key)" if key in value.get("optional", ()) else ""
                content.append(f"<tr id='{esc(name)}.{esc(key)}'><td><code>{esc(key)}</code>{optional}</td>"
                               f"<td><code>{esc(annotation)}</code></td><td>{esc(docs[key])}</td></tr>")
            content.append("</tbody></table>")
        content.append("</section>")
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
    current = rust_reference(source, output / "rust-current-target", public / f"rust-{version}", cargo, logs,
                             "rust-current", require_docs=True)
    previous = rust_reference(legacy, output / "rust-legacy-target", public / "rust-0.1.0", cargo, logs, "rust-legacy")
    run([sys.executable, "-m", "sphinx", "-n", "-W", "-b", "html", "-d", str(output / "sphinx-doctrees"),
         str(source / "bindings/python/docs"), str(public / f"python-{version}")], logs, logs, "sphinx")
    for topic in ("index", "api", "data", "contracts"):
        link_offline_guide(public / f"python-{version}" / f"{topic}.html",
                           public / "source/docs/python.html", "Python programming guide", "</h1>")
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
            "<p>API documentation. See the matching release record for artifact availability and package-specific validation results. Generated references do not imply full version or platform acceptance.</p>"
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
