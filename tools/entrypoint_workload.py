"""Run frozen Python entrypoints with one cumulative POSIX supervision budget.

This tool supervises execution and preserves streams. It does not infer semantic
acceptance from exit zero, approve a protocol, install packages, or run a matrix.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import paired_workload as paired
import python_workload as worker


def load(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('entrypoint plan SHA differs')
    plan = json.loads(raw)
    keys = {'schema_version', 'inputs_sha256', 'python', 'python_target',
            'python_sha256', 'entries', 'unit_budget_seconds', 'total_budget_seconds'}
    if type(plan) is not dict or set(plan) != keys or type(plan['schema_version']) is not int or plan['schema_version'] != 1:
        raise ValueError('unknown entrypoint plan schema')
    for name in ('unit_budget_seconds', 'total_budget_seconds'):
        paired.positive(plan[name], name)
    inputs = plan['inputs_sha256']
    if type(inputs) is not dict or not inputs:
        raise ValueError('frozen inputs are required')
    for name, digest in inputs.items():
        if not Path(name).is_absolute() or type(digest) is not str or len(digest) != 64:
            raise ValueError('input path/hash is invalid')
    for name in ('python', 'python_target'):
        if not Path(plan[name]).is_absolute():
            raise ValueError('absolute executable invocation and target required')
    entries = plan['entries']
    if type(entries) is not list or not entries:
        raise ValueError('explicit nonempty entrypoint list required')
    ids = set()
    for row in entries:
        if type(row) is not dict or set(row) != {'id', 'argv'}:
            raise ValueError('invalid entrypoint row')
        name, argv = row['id'], row['argv']
        if (type(name) is not str or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name)
                or name in ids):
            raise ValueError('invalid or duplicate entrypoint ID')
        ids.add(name)
        if (type(argv) is not list or len(argv) < 4 or any(type(a) is not str for a in argv)
                or argv[:3] != [plan['python'], '-I', '-B']
                or not Path(argv[3]).is_absolute() or argv[3] not in inputs):
            raise ValueError('entrypoints must use isolated nonoptimized Python and a frozen script')
    required = [Path(__file__).resolve(), Path(paired.__file__).resolve(),
                Path(worker.__file__).resolve()]
    for name in ('result_journal.py', 'memory_observer.py'):
        required.append(Path(worker.__file__).resolve().parent / name)
    if any(str(p) not in inputs for p in required):
        raise ValueError('supervisor dependency hashes are required')
    return plan


def execute(plan, output):
    if os.name != 'posix':
        raise ValueError('POSIX process groups are required')
    if any(k.startswith(('LD_', 'DYLD_')) for k in os.environ):
        raise ValueError('remove loader injection before execution')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('output must be new')
    for path in plan['inputs_sha256']:
        if output.resolve().is_relative_to(Path(path).resolve().parent):
            raise ValueError('output must be outside frozen input directories')
    bindings = {'python': {'invocation': plan['python'], 'resolved': plan['python_target'],
                           'sha256': plan['python_sha256']}}
    paired.verify_frozen(plan['inputs_sha256'])
    paired.verify_invocations(bindings)
    output.mkdir(parents=True)
    rows = [dict(row, state='Not Run') for row in plan['entries']]
    report = {'kind': 'ENTRYPOINT_EXECUTION_NOT_ACCEPTANCE', 'complete': False,
              'completion_scope': 'provisional until exit zero plus final stdout receipt',
              'plan': plan, 'entries': rows}
    budget = paired.Budget(plan['total_budget_seconds'], plan['total_budget_seconds'])
    env = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME'}}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        worker.write_json(output / 'run.json', report)
        for row in rows:
            started, deadline = budget.deadline('python', plan['unit_budget_seconds'])
            row['state'] = 'running'
            errors = []
            try:
                paired.verify_frozen(plan['inputs_sha256'])
                paired.verify_invocations(bindings)
                worker.write_json(output / 'run.json', report)
                row['process'] = paired.run_process(row['argv'], output / row['id'], deadline,
                                                    cwd=output, env=env)
                paired.verify_frozen(plan['inputs_sha256'])
                paired.verify_invocations(bindings)
            except BaseException as error:
                errors.append(error)
            row['state'] = 'incomplete' if errors else 'completed'
            try:
                worker.write_json(output / 'run.json', report)
            except BaseException as error:
                errors.append(error)
            try:
                last_accounted = budget.finish('python', started, deadline)
            except BaseException as error:
                errors.append(error)
            if errors:
                row['state'] = 'incomplete'
                if len(errors) == 1:
                    raise errors[0]
                raise BaseExceptionGroup('entrypoint and supervision failed', errors)
        report['complete'] = True
        worker.write_json(output / 'run.json', report)
        closeout_started = budget.finish('python', last_accounted, deadline)
        print(json.dumps({'event': 'ENTRYPOINT_EXIT_RECEIPT',
                          'run_json_sha256': worker.sha(output / 'run.json'),
                          'elapsed_through_run_receipt_seconds': budget.used['python'],
                          'semantic_acceptance': False}), flush=True)
        # Printing and evidence checks share the original final deadline too.
        budget.finish('python', closeout_started, deadline)
        return report
    except BaseException as error:
        report['complete'] = False
        for row in rows:
            if row['state'] == 'running': row['state'] = 'incomplete'
        if rows and rows[-1]['state'] == 'completed': rows[-1]['state'] = 'incomplete'
        report['failure'] = worker.failure_record(error)
        try:
            worker.write_json(output / 'run.json', report)
        except BaseException as receipt_error:
            raise BaseExceptionGroup('entrypoint and failure receipt failed', [error, receipt_error]) from None
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(load(args.plan, args.plan_sha256), args.output)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # An independent healthy stream must retain known child outcomes when
        # both process and aggregate receipt storage fail. Never replay work.
        worker.report_failure(error, sys.stderr)
        raise
