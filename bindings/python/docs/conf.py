"""Build the reference from the matching installed native package."""

import cofferdb

project = "CofferDB Python"
copyright = "2026, CofferDB contributors"
release = cofferdb.__version__
version = release
extensions = ["sphinx.ext.autodoc"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
html_theme = "alabaster"
html_title = f"CofferDB Python {release}"
html_last_updated_fmt = None
html_copy_source = True
nitpicky = True
rst_prolog = f".. |engine_version| replace:: {cofferdb.engine_version}\n"


def resolve_recursive_alias(app, env, node, content):
    # Sphinx 8 renders recursive ForwardRef values with a class role. Link the
    # actual alias target instead; unresolved references still remain errors.
    if node.get("refdomain") == "py" and node.get("reftarget") == "JsonValue":
        return env.domains.python_domain.resolve_xref(
            env, node["refdoc"], app.builder, "type", "JsonValue", node, content,
        )
    return None


def setup(app):
    app.connect("missing-reference", resolve_recursive_alias)

