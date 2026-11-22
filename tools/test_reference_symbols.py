"""Run with the documentation environment: python tools/test_reference_symbols.py."""

import tempfile
import unittest
from pathlib import Path

from build_reference import rust_symbols


class RustSymbolTests(unittest.TestCase):
    def test_only_owned_members_and_individual_fields_are_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory) / "cofferdb"
            crate.mkdir()
            (crate / "index.html").write_text("<main></main>")
            (crate / "enum.Index.html").write_text("""
                <h2 id="variants" class="section-header">Variants</h2>
                <section id="variant.Ann" class="variant"></section>
                <div id="variant.Ann.fields" class="sub-variant">
                  <span id="variant.Ann.field.params" class="section-header"></span>
                </div>
                <section id="variant.Value" class="variant"></section>
                <span id="variant.Value.field.0" class="section-header"></span>
                <h2 id="implementations" class="section-header">Implementations</h2>
                <section id="method.as_str" class="method"></section>
                <h2 id="trait-implementations" class="section-header">Trait Implementations</h2>
                <section id="method.fmt" class="method trait-impl"></section>
                <h2 id="blanket-implementations" class="section-header">Blanket Implementations</h2>
                <section id="associatedtype.Error-1" class="associatedtype trait-impl"></section>
            """)
            (crate / "struct.Row.html").write_text("""
                <h2 id="fields" class="section-header">Fields</h2>
                <span id="structfield.result" class="structfield section-header"></span>
            """)
            (crate / "trait.Read.html").write_text("""
                <h2 id="required-methods" class="section-header">Required Methods</h2>
                <section id="tymethod.read" class="method"></section>
            """)
            members = {(s["name"], s["kind"], s["page"].split("#")[1])
                       for s in rust_symbols(crate) if "#" in s["page"]}
        self.assertEqual(members, {
            ("cofferdb::Index::Ann", "variant", "variant.Ann"),
            ("cofferdb::Index::Ann::params", "variantfield", "variant.Ann.field.params"),
            ("cofferdb::Index::Value", "variant", "variant.Value"),
            ("cofferdb::Index::Value::0", "variantfield", "variant.Value.field.0"),
            ("cofferdb::Index::as_str", "method", "method.as_str"),
            ("cofferdb::Row::result", "structfield", "structfield.result"),
            ("cofferdb::Read::read", "tymethod", "tymethod.read"),
        })


if __name__ == "__main__":
    unittest.main()
