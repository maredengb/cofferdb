"""Prepare only three new empty venvs under the pinned existing supervisor.

No pip, dependency or SDK installation. Default preview never launches Python.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import stat
import time
import types
import hashlib

HERE = Path(__file__).absolute().parent
RUNNER_SHA = 'f984c5a04572522efa2b5c422b854a71f827124449be353a0c83b2e57a1dc835'
def runner_bytes():
    selected = HERE / 'runner.py'
    if any(item.is_symlink() for item in [selected, *selected.parents]):
        raise ValueError('Runner path link')
    fd = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not (stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and before.st_size <= 131072 and not before.st_mode & 0o022):
            raise ValueError('Bounded protected runner source')
        with os.fdopen(fd, 'rb', closefd=False) as stream: raw = stream.read(131073)
        after = os.fstat(fd)
        signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(raw) != before.st_size or signature(before) != signature(after) or hashlib.sha256(raw).hexdigest() != RUNNER_SHA:
            raise ValueError('Frozen runner bytes must match before compile')
        return raw
    finally:
        os.close(fd)
r = types.ModuleType('mac_venv_shared'); r.__file__ = str(HERE / 'runner.py')
exec(compile(runner_bytes(), r.__file__, 'exec', dont_inherit=True), r.__dict__)


def validate(plan):
    r.need(type(plan) is dict and set(plan) == {'schema_version', 'kind', 'supervisor', 'runtimes', 'evidence'}
           and type(plan['schema_version']) is int and plan['schema_version'] == 1
           and plan['kind'] == 'macos15_three_empty_venvs_1', 'Exact empty-venv plan')
    r.need(type(plan['supervisor']) is dict and set(plan['supervisor']) == {'path', 'sha256'}
           and plan['supervisor']['sha256'] == r.SUPERVISOR_SHA, 'Pinned existing supervisor')
    r.path(plan['supervisor']['path']); evidence = r.path(plan['evidence'])
    r.need(type(plan['runtimes']) is list and len(plan['runtimes']) == 3, 'Exactly original three ABI venvs')
    destinations = []
    for row, minor in zip(plan['runtimes'], [12, 13, 14]):
        r.need(type(row) is dict and set(row) == {'minor', 'path', 'sha256', 'venv'}
               and type(row['minor']) is int and row['minor'] == minor, 'Ordered base interpreter identity')
        source = r.path(row['path']); dest = r.path(row['venv']); r.checksum(row['sha256'])
        r.need(not r.beneath(source, dest) and not r.beneath(dest, source)
               and not r.beneath(evidence, dest) and not r.beneath(dest, evidence), 'Separate new venv and inputs')
        destinations.append(dest)
    r.need(all(not r.beneath(a, b) and not r.beneath(b, a) for i, a in enumerate(destinations)
               for b in destinations[i + 1:]), 'Disjoint exact venv roots')
    r.need(all(not r.beneath(r.path(plan['supervisor']['path']), root)
               and not r.beneath(HERE, root) for root in [*destinations, evidence]),
           'Maintenance helpers outside new mutable roots')
    return plan


def commands(plan):
    validate(plan)
    return [[row['path'], '-I', '-S', '-B', '-m', 'venv', '--without-pip', '--copies', row['venv']]
            for row in plan['runtimes']]


def read_plan(path, expected):
    r.checksum(expected); p = r.path(path); r.no_link(p); raw = p.read_bytes()
    r.need(r.digest(raw) == expected, 'Frozen venv plan SHA')
    return validate(r.parsed(raw))


def actual(plan):
    r.row(plan['supervisor'])
    for row in plan['runtimes']:
        r.row({k: row[k] for k in ['path', 'sha256']}, executable=True)


def binding(row, original):
    root = r.path(row['venv']); r.no_link(root); info = root.lstat()
    r.need(stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == original
           and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
           'Only the same newly owned private0700 venv root; never repair chmod')
    invoke = root / 'bin/python'; real = invoke.resolve(strict=True)
    r.need(r.beneath(real, root), '--copies venv target must remain inside the new root')
    r.row({'path': str(real), 'sha256': row['sha256']}, executable=True)
    cfg = root / 'pyvenv.cfg'; r.no_link(cfg)
    info = cfg.stat(); r.need(stat.S_ISREG(info.st_mode) and info.st_size <= 16384, 'Bounded venv config')
    with cfg.open('rb') as stream: raw = stream.read(16385)
    r.need(len(raw) == info.st_size and b'include-system-site-packages = false' in raw,
                                'Private empty venv config')
    values = dict(line.split(' = ', 1) for line in raw.decode('utf-8', 'strict').splitlines()
                  if ' = ' in line)
    r.need(values.get('version', '').split('.')[:2] == ['3', str(row['minor'])],
           'New venv actual version must match selected minor')
    return {'minor': row['minor'], 'venv': str(root), 'invocation': str(invoke), 'real': str(real),
            'base': row['path'],
            'sha256': row['sha256'], 'pyvenv_cfg_sha256': r.digest(raw),
            'pip_or_SDK_installed_by_this_command': False}


def run(path, expected, approved):
    r.need(approved == expected, 'Exact separate empty-venv plan approval required')
    plan = read_plan(path, expected); host = r.host_gate(); actual(plan)
    # Refuse all existing roots before creating evidence or invoking any command.
    for row in plan['runtimes']:
        dest = r.path(row['venv']); r.no_link(dest, missing=True)
        r.need(not os.path.lexists(dest), 'Venv root already exists')
        parent = dest.parent.stat()
        r.need(parent.st_uid == os.getuid() and stat.S_IMODE(parent.st_mode) == 0o700, 'Private venv parent')
    supervisor = r.load_supervisor(plan)
    started = time.monotonic(); deadline = started + 300
    evidence = r.private_new(plan['evidence'])
    r.save(evidence / 'INTENT.json', {'plan_sha256': expected, 'deadline_monotonic': deadline, 'actual_host': host,
           'no_pip': True, 'no_SDK': True, 'whole_seconds': 300})
    records, result_bindings = [], []
    environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C',
                   'HOME': str(evidence), 'TMPDIR': str(evidence)}
    try:
        for row, argv in zip(plan['runtimes'], commands(plan)):
            actual(plan); r.need(read_plan(path, expected) == plan, 'Venv plan drift')
            r.need(time.monotonic() + 3 < deadline, 'Venv chain deadline')
            root = r.private_new(row['venv']); info = root.lstat(); original = (info.st_dev, info.st_ino)
            request = evidence / ('cp3' + str(row['minor']) + '-command.json')
            r.save(request, {'schema_version': 1, 'argv': argv, 'cwd': str(evidence)})
            receipt = supervisor.capture(str(request), r.digest(request.read_bytes()),
                python=plan['runtimes'][0]['path'], environment=environment,
                output=str(evidence / ('cp3' + str(row['minor']) + '-capture')),
                deadline=min(deadline, time.monotonic() + 120), supervisor_sha256=r.SUPERVISOR_SHA,
                raw_limit=2 * 1024 * 1024, cleanup_reserve=2.0)
            r.captured_ok(receipt); records.append(receipt)
            result_bindings.append(binding(row, original))
        actual(plan); r.need(read_plan(path, expected) == plan and time.monotonic() < deadline, 'Final venv gate')
        result = {'status': 'THREE_EMPTY_VENVS_PREPARED_NOT_INSTALLED', 'bindings': result_bindings,
                  'records': records, 'plan_sha256': expected, 'product_acceptance': False}
        r.save(evidence / 'RESULT.json', result); return result
    except BaseException as error:
        try:
            r.save(evidence / 'HOLD.json', {'status': 'HOLD_EMPTY_VENV_NO_RETRY',
                   'error_type': type(error).__name__, 'plan_sha256': expected})
        except BaseException:
            pass
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', choices=['verify', 'preview', 'run'], default='verify')
    p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True)
    p.add_argument('--approved-plan-sha256'); a = p.parse_args()
    if a.action == 'run': result = run(a.plan, a.plan_sha256, a.approved_plan_sha256)
    else:
        plan = read_plan(a.plan, a.plan_sha256)
        result = {'status': 'EMPTY_VENV_COMMANDS_PREVIEW_NOT_EXECUTED', 'commands': commands(plan),
                  'product_acceptance': False}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
