"""Describe the actual installed package and its packaged public type hints."""

import argparse
import ast
import importlib.resources
import inspect
import json
from pathlib import Path
from typing import get_type_hints, is_typeddict

import cofferdb


def inventory():
    stub = ast.parse(importlib.resources.files("cofferdb").joinpath("__init__.pyi").read_text())
    declared_attributes = {
        node.name: {item.target.id: ast.unparse(item.annotation)
                    for item in node.body if isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)}
        for node in stub.body if isinstance(node, ast.ClassDef)
    }
    declared_methods = {
        node.name: {
            item.name: {
                "signature": "(" + ast.unparse(item.args) + ") -> " + ast.unparse(item.returns),
                "staticmethod": any(isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                                     for decorator in item.decorator_list),
            }
            for item in node.body if isinstance(item, ast.FunctionDef) and item.returns is not None
        }
        for node in stub.body if isinstance(node, ast.ClassDef)
    }
    symbols = {}
    for name in sorted(cofferdb.__all__):
        value = getattr(cofferdb, name)
        if is_typeddict(value):
            symbols[name] = {
                "kind": "TypedDict",
                "fields": {key: inspect.formatannotation(item)
                           for key, item in get_type_hints(value, include_extras=True).items()},
                "required": sorted(value.__required_keys__),
                "optional": sorted(value.__optional_keys__),
                "doc": inspect.getdoc(value),
            }
        elif name in ("Database", "Collection"):
            methods = {}
            for member_name in dir(value):
                if member_name.startswith("_") and member_name not in ("__enter__", "__exit__"):
                    continue
                member = getattr(value, member_name)
                if callable(member):
                    methods[member_name] = {"signature": str(inspect.signature(member)),
                                            "doc": inspect.getdoc(member)}
            symbols[name] = {"kind": "native_class", "methods": methods,
                             "declared_methods": declared_methods.get(name, {}),
                             "doc": inspect.getdoc(value)}
        elif inspect.isclass(value) and issubclass(value, BaseException):
            symbols[name] = {"kind": "exception", "bases": [base.__name__ for base in value.__bases__],
                             "declared_attributes": declared_attributes.get(name, {}),
                             "doc": inspect.getdoc(value)}
        elif name in ("__version__", "engine_version"):
            symbols[name] = {"kind": "version", "value": value}
        else:
            symbols[name] = {"kind": "type_alias", "definition": inspect.formatannotation(value)}
    return {"schema_version": 1, "sdk_version": cofferdb.__version__,
            "engine_version": cofferdb.engine_version, "symbols": symbols}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inventory(), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
