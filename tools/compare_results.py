"""Compare nonempty scored search responses without deleting or sorting fields.

Only declared computed-score paths admit numerical tolerance. This utility checks
comparison and consistency, not retrieval correctness or release acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

FLOAT_PATHS = (
    'results.*.bm25_score', 'results.*.vector_score',
    'results.*.final_score', 'results.*.metadata_boost',
    'results.*.normalized_scores.bm25', 'results.*.normalized_scores.vector',
    'results.*.normalized_scores.metadata_boost',
    'results.*.explain.scores.bm25_raw', 'results.*.explain.scores.bm25_norm',
    'results.*.explain.scores.vector_raw', 'results.*.explain.scores.vector_norm',
    'results.*.explain.scores.metadata_boost', 'results.*.explain.scores.final_score',
    'results.*.explain.final_score', 'results.*.explain.normalization.bm25_max',
    'results.*.explain.bm25.terms.*.raw_contribution',
    'results.*.explain.bm25.terms.*.avg_chunk_length',
    'results.*.explain.matched_term_details.*.bm25_contribution',
)
DUPLICATES = (
    ('final_score', 'explain.final_score', 'explain.scores.final_score'),
    ('bm25_score', 'explain.scores.bm25_raw'),
    ('vector_score', 'explain.scores.vector_raw'),
    ('normalized_scores.bm25', 'explain.scores.bm25_norm'),
    ('normalized_scores.vector', 'explain.scores.vector_norm'),
    ('metadata_boost', 'normalized_scores.metadata_boost', 'explain.scores.metadata_boost'),
)
NULLABLE = {'bm25_score', 'vector_score', 'explain.scores.bm25_raw',
            'explain.scores.vector_raw', 'normalized_scores.vector',
            'explain.scores.vector_norm', 'explain.normalization.bm25_max'}


def number(value):
    return type(value) in (int, float, Decimal)


def finite(value):
    return value.is_finite() if type(value) is Decimal else (type(value) is int or math.isfinite(value))


def exact(a, b):
    if number(a) and number(b):
        return a == b
    return type(a) is type(b) and a == b


def get(value, path):
    for key in path.split('.'):
        if type(value) is not dict or key not in value:
            raise ValueError('missing object path: ' + path)
        value = value[key]
    return value


def expand(value, pattern):
    """Resolve stars only through arrays; missing fields never disappear."""
    parts = pattern.split('.')
    def visit(node, offset):
        if offset == len(parts):
            return [node]
        key = parts[offset]
        if key == '*':
            if type(node) is not list:
                raise ValueError('wildcard requires array: ' + pattern)
            return [leaf for child in node for leaf in visit(child, offset + 1)]
        if type(node) is not dict or key not in node:
            raise ValueError('missing object path: ' + pattern)
        return visit(node[key], offset + 1)
    return visit(value, 0)


def validate(response, no_match_ids):
    rows = get(response, 'results')
    if type(rows) is not list or not rows:
        raise ValueError('this comparison requires a nonempty scored response')
    text_present = get(response, 'query_context.text_present')
    if type(text_present) is not bool:
        raise ValueError('text_present must be boolean')
    seen = set()
    for row in rows:
        identity = get(row, 'chunk_id')
        if type(identity) is not str or identity in seen:
            raise ValueError('unique string chunk IDs required')
        seen.add(identity)
        for group in DUPLICATES:
            values = [get(row, path) for path in group]
            if any(not exact(values[0], value) for value in values[1:]):
                raise ValueError('inconsistent duplicate score: ' + group[0])
        minimum = get(row, 'explain.normalization.bm25_min')
        if not number(minimum) or minimum != 0:
            raise ValueError('bm25_min must be exact numeric zero')
        for pattern in FLOAT_PATHS:
            relative = pattern.removeprefix('results.*.')
            for value in expand(row, relative):
                if value is None and relative in NULLABLE:
                    continue
                if not number(value) or not finite(value):
                    raise ValueError('invalid computed score: ' + relative)
        if not text_present or identity in no_match_ids:
            raw = get(row, 'bm25_score')
            if not text_present:
                if raw is not None:
                    raise ValueError('absent text requires null raw BM25')
            elif not number(raw) or raw != 0:
                raise ValueError('no-match raw BM25 must be exact zero')
            norm = get(row, 'normalized_scores.bm25')
            if not number(norm) or norm != 0:
                raise ValueError('missing/no-match text requires exact zero norm')
    if not set(no_match_ids) <= seen:
        raise ValueError('no-match oracle IDs missing from results')


def compare(left, right, *, required_paths, no_match_ids=()):
    """Return coverage only after complete validation and ordered comparison.

    required_paths comes from the predeclared protocol, not observed mismatches.
    Empty optional term arrays remain visible as zero coverage in the report.
    """
    if (type(required_paths) not in (list, tuple) or not required_paths
            or any(type(p) is not str or p not in FLOAT_PATHS for p in required_paths)
            or len(set(required_paths)) != len(required_paths)):
        raise ValueError('explicit unique computed-score coverage paths required')
    if (type(no_match_ids) not in (list, tuple) or any(type(x) is not str for x in no_match_ids)
            or len(set(no_match_ids)) != len(no_match_ids)):
        raise ValueError('invalid frozen no-match oracle IDs')
    validate(left, no_match_ids)
    validate(right, no_match_ids)
    coverage = {pattern: 0 for pattern in FLOAT_PATHS}
    array_step = object()
    tokens = {tuple(array_step if part == '*' else part for part in pattern.split('.')): pattern
              for pattern in FLOAT_PATHS}
    def walk(a, b, path):
        label = repr(tuple('*' if p is array_step else p for p in path))
        score_path = tokens.get(path)
        # Integers remain integers in JSON parsing; bool is never a number.
        if number(a) and number(b):
            if not finite(a) or not finite(b):
                raise ValueError('nonfinite number: ' + label)
            if score_path is not None:
                coverage[score_path] += 1
                if a == b:
                    return
                left_number, right_number = Fraction(a), Fraction(b)
                if abs(left_number - right_number) <= Fraction(1, 10**12) * (1 + max(abs(left_number), abs(right_number))):
                    return
            elif a == b:
                return
            raise ValueError('numeric difference: ' + label)
        if type(a) is not type(b):
            raise ValueError('type difference: ' + label)
        if type(a) is dict:
            if set(a) != set(b):
                raise ValueError('object key difference: ' + label)
            for key in a:
                walk(a[key], b[key], (*path, key))
        elif type(a) is list:
            if len(a) != len(b):
                raise ValueError('array length difference: ' + label)
            for x, y in zip(a, b):
                walk(x, y, (*path, array_step))
        elif a != b:
            raise ValueError('value difference: ' + label)
    walk(left, right, ())
    if any(coverage[path] == 0 for path in required_paths):
        raise ValueError('required numerical coverage was not exercised')
    return {'comparison_equal': True, 'numerical_coverage': coverage,
            'required_paths': list(required_paths), 'semantic_acceptance': False}


def load(path):
    raw = Path(path).read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=Decimal)
    return value, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--require-path', action='append', required=True)
    parser.add_argument('--no-match-id', action='append', default=[])
    args = parser.parse_args()
    left, left_sha = load(args.left)
    right, right_sha = load(args.right)
    report = compare(left, right, required_paths=args.require_path, no_match_ids=args.no_match_id)
    report.update(left_sha256=left_sha, right_sha256=right_sha)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
