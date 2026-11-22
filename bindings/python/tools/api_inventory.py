"""Validate and describe the installed API and the public typing files in its wheel."""

import argparse
import ast
import importlib.resources
import inspect
import json
from pathlib import Path
from typing import get_type_hints, is_typeddict

import cofferdb


def require_doc(value, name):
    if not value or not value.strip():
        raise ValueError(f"Missing public API documentation for {name}")
    return inspect.cleandoc(value)


def attribute_docs(node, lines, owner):
    """Read Sphinx attribute comments immediately above annotated declarations."""
    result = {}
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        name = item.target.id
        if name.startswith("_") and name != "__version__":
            continue
        index = item.lineno - 2
        comments = []
        while index >= 0 and lines[index].lstrip().startswith("#:"):
            comments.append(lines[index].lstrip()[2:].strip())
            index -= 1
        result[name] = require_doc(" ".join(reversed(comments)), f"{owner}.{name}")
    return result


def check_names(actual, expected, owner):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"{owner} mismatch: missing {missing}, unexpected {extra}")


def typed_signature(function):
    return "(" + ast.unparse(function.args) + ") -> " + ast.unparse(function.returns)


def native_callable(member, declaration, name):
    """Check call shape, including keyword-only parameters and concrete defaults."""
    signature = inspect.signature(member)
    args = declaration.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    declared = [(arg.arg, inspect.Parameter.POSITIONAL_ONLY if i < len(args.posonlyargs)
                 else inspect.Parameter.POSITIONAL_OR_KEYWORD, defaults[i])
                for i, arg in enumerate(positional)]
    if args.vararg:
        declared.append((args.vararg.arg, inspect.Parameter.VAR_POSITIONAL, None))
    declared.extend((arg.arg, inspect.Parameter.KEYWORD_ONLY, default)
                    for arg, default in zip(args.kwonlyargs, args.kw_defaults))
    if args.kwarg:
        declared.append((args.kwarg.arg, inspect.Parameter.VAR_KEYWORD, None))
    parameters = list(signature.parameters.values())
    if [parameter.name for parameter in parameters] != [label for label, _, _ in declared]:
        raise ValueError(f"{name}: native and stub parameter names or order differ")
    for parameter, (label, kind, default) in zip(parameters, declared):
        if parameter.kind != kind:
            raise ValueError(f"{name}.{label}: native and stub parameter kinds differ")
        if default is None:
            matches = parameter.default is inspect.Parameter.empty
        else:
            declared_default = ast.literal_eval(default)
            matches = parameter.default is not inspect.Parameter.empty and (
                declared_default is Ellipsis or
                type(parameter.default) is type(declared_default) and parameter.default == declared_default)
        if not matches:
            raise ValueError(f"{name}.{label}: native and stub defaults differ")
    return {"signature": str(signature), "declared_signature": typed_signature(declaration),
            "doc": require_doc(member.__doc__, name),
            "declared_doc": require_doc(ast.get_docstring(declaration), f"{name} (stub)")}


def inventory():
    package = importlib.resources.files("cofferdb")
    stub_source = package.joinpath("__init__.pyi").read_text(encoding="utf-8")
    types_source = package.joinpath("types.py").read_text(encoding="utf-8")
    stub, types = ast.parse(stub_source), ast.parse(types_source)
    classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}
    functions = {node.name: node for node in stub.body if isinstance(node, ast.FunctionDef)}
    versions = attribute_docs(stub, stub_source.splitlines(), "cofferdb")
    aliases = attribute_docs(types, types_source.splitlines(), "cofferdb.types")
    reexports = {item.asname for node in stub.body if isinstance(node, ast.ImportFrom)
                 for item in node.names if item.asname == item.name}
    check_names(cofferdb.__all__, reexports | classes.keys() | functions.keys() | versions.keys(),
                "Python runtime and stub exports")
    typed_fields = {}
    type_lines = types_source.splitlines()
    for node in types.body:
        if not isinstance(node, ast.ClassDef) or node.name not in reexports:
            continue
        fields = {}
        for base in node.bases:
            if isinstance(base, ast.Name):
                fields.update(typed_fields.get(base.id, {}))
        fields.update(attribute_docs(node, type_lines, node.name))
        typed_fields[node.name] = fields

    symbols = {}
    for name in sorted(cofferdb.__all__):
        value = getattr(cofferdb, name)
        if is_typeddict(value):
            hints = get_type_hints(value, include_extras=True)
            field_docs = typed_fields[name]
            check_names(field_docs, hints, name)
            symbols[name] = {
                "kind": "TypedDict",
                "fields": {key: inspect.formatannotation(item) for key, item in hints.items()},
                "field_docs": field_docs,
                "required": sorted(value.__required_keys__), "optional": sorted(value.__optional_keys__),
                "doc": require_doc(value.__dict__.get("__doc__"), name),
            }
        elif name in functions:
            symbols[name] = {"kind": "native_function", **native_callable(value, functions[name], name)}
        elif name in classes:
            node = classes[name]
            symbol = {"doc": require_doc(value.__dict__.get("__doc__"), name),
                      "declared_doc": require_doc(ast.get_docstring(node), f"{name} (stub)")}
            if issubclass(value, BaseException):
                symbol.update(kind="exception", bases=[base.__name__ for base in value.__bases__],
                              declared_attributes={item.target.id: ast.unparse(item.annotation)
                                                   for item in node.body if isinstance(item, ast.AnnAssign)},
                              attribute_docs=attribute_docs(node, stub_source.splitlines(), name))
            else:
                members = {key: getattr(value, key) for key in dir(value)
                           if not key.startswith("_") or key in ("__enter__", "__exit__")}
                declarations = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}
                check_names(members, declarations, name)
                methods = {}
                declared_methods = {}
                for key, member in members.items():
                    declaration = declarations[key]
                    static = isinstance(inspect.getattr_static(value, key), staticmethod)
                    if static != any(isinstance(d, ast.Name) and d.id == "staticmethod"
                                     for d in declaration.decorator_list):
                        raise ValueError(f"{name}.{key}: native and stub binding kinds differ")
                    methods[key] = native_callable(member, declaration, f"{name}.{key}")
                    declared_methods[key] = {"signature": methods[key]["declared_signature"],
                                             "staticmethod": static}
                symbol.update(kind="native_class", methods=methods, declared_methods=declared_methods)
            symbols[name] = symbol
        elif name in versions:
            symbols[name] = {"kind": "version", "value": value, "doc": versions[name]}
        else:
            symbols[name] = {"kind": "type_alias", "definition": inspect.formatannotation(value),
                             "doc": require_doc(aliases.get(name), name)}
    return {"schema_version": 2, "sdk_version": cofferdb.__version__,
            "engine_version": cofferdb.engine_version, "symbols": symbols}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inventory(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
