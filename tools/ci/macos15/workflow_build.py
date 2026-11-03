"""Capture existing Mac prepare/build commands; default verify never executes them."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import types
import stat

HERE = Path(__file__).absolute().parent
RUNNER_SHA = 'cffc0a3a5e229daa99b231178ab226e2bcfad76cbf353a8c86c17e3a02a1502e'
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
r = types.ModuleType('mac_runtime_shared'); r.__file__ = str(HERE / 'runner.py')
exec(compile(runner_bytes(), r.__file__, 'exec', dont_inherit=True), r.__dict__)
FIELDS = {'schema_version', 'kind', 'checkout', 'public_commit', 'prepare', 'builder',
          'pythons', 'cargo', 'rustc', 'maturin', 'cargo_home', 'work', 'evidence', 'supervisor'}


def validate(value):
    r.need(type(value) is dict and set(value) == FIELDS
           and type(value['schema_version']) is int and value['schema_version'] == 1
           and value['kind'] == 'macos15_three_abi_supervised_build_1', 'Exact build plan schema')
    r.need(type(value['public_commit']) is str and len(value['public_commit']) == 40
           and all(c in '0123456789abcdef' for c in value['public_commit']), 'Fixed public commit')
    checkout, work, evidence = [r.path(value[k]) for k in ['checkout', 'work', 'evidence']]
    r.need(not any(r.beneath(a, b) or r.beneath(b, a) for a, b in
           [(checkout, work), (checkout, evidence), (work, evidence)]), 'Separate source/work/evidence')
    r.path(value['cargo_home'])
    for key, relative in [('prepare', 'tools/prepare_source.py'), ('builder', 'tools/build_python.py')]:
        row = value[key]
        r.need(type(row) is dict and set(row) == {'path', 'sha256'}
               and row['path'] == str(checkout / relative), 'Existing product maintenance script path')
        r.checksum(row['sha256'])
    for key in ['cargo', 'rustc', 'maturin', 'supervisor']:
        row = value[key]; r.need(type(row) is dict and set(row) == {'path', 'sha256'}, 'Tool identity')
        r.path(row['path']); r.checksum(row['sha256'])
    r.need(not any(r.beneath(r.path(value['supervisor']['path']), root)
                  or r.beneath(HERE, root) for root in [work, evidence]), 'Maintenance helpers outside build outputs')
    r.need(value['supervisor']['sha256'] == r.SUPERVISOR_SHA, 'Pinned existing supervisor')
    r.need(type(value['pythons']) is list and len(value['pythons']) == 3, 'Three original native ABIs')
    for row, minor in zip(value['pythons'], [12, 13, 14]):
        r.need(type(row) is dict and set(row) == {'path', 'sha256', 'minor'}
               and type(row['minor']) is int and row['minor'] == minor, 'Ordered CP312/313/314')
        r.path(row['path']); r.checksum(row['sha256'])
    return value


def commands(plan):
    validate(plan)
    python = plan['pythons'][0]['path']; work = Path(plan['work'])
    prepare = [python, '-I', '-S', '-B', plan['prepare']['path'], str(work / 'source'),
               '--source', plan['checkout'], '--commit', plan['public_commit'], '--cargo', plan['cargo']['path']]
    build = [python, '-I', '-S', '-B', plan['builder']['path'], str(work / 'wheels'),
             '--source', str(work / 'source')]
    for row in plan['pythons']:
        build += ['--python', row['path']]
    build += ['--maturin', plan['maturin']['path']]
    return [('prepare', prepare, 300), ('build', build, 3600)]


def read_plan(path, expected):
    r.checksum(expected); p = r.path(path); r.no_link(p); raw = p.read_bytes()
    r.need(r.digest(raw) == expected, 'Frozen build plan raw SHA')
    return validate(r.parsed(raw))


def actual(plan):
    for key in ['prepare', 'builder', 'supervisor']:
        r.row(plan[key])
    for key in ['cargo', 'rustc', 'maturin']:
        r.row(plan[key], executable=True)
    for row in plan['pythons']:
        r.row({k: row[k] for k in ['path', 'sha256']}, executable=True, host_role='python')
    for key in ['checkout', 'cargo_home']:
        p = r.path(plan[key]); r.no_link(p); r.need(p.is_dir(), 'Selected existing input directory')


def run(path, expected, approved):
    r.need(approved == expected, 'Exact separate build-plan approval required')
    plan = read_plan(path, expected); host = r.host_gate(); actual(plan)
    # Existing source helpers verify actual clean checkout/commit and original
    # normal-GIL native ABI/notice/dependency contracts; no substitute assertions.
    supervisor = r.load_supervisor(plan)
    started = time.monotonic(); deadline = started + 3900
    evidence = r.private_new(plan['evidence'])
    r.save(evidence / 'INTENT.json', {'plan_sha256': expected, 'started_monotonic': started, 'actual_host': host,
           'deadline_monotonic': deadline, 'prepare_seconds': 300, 'build_seconds': 3600,
           'product_acceptance': False})
    work = r.private_new(plan['work'])
    environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'HOME': str(work),
                   'TMPDIR': str(work), 'CARGO_HOME': plan['cargo_home'], 'CARGO_NET_OFFLINE': 'true',
                   'CARGO': plan['cargo']['path'], 'RUSTC': plan['rustc']['path'],
                   'MACOSX_DEPLOYMENT_TARGET': '15.0'}
    records = []
    try:
        for name, argv, seconds in commands(plan):
            actual(plan); r.need(read_plan(path, expected) == plan, 'Build plan drift')
            r.need(time.monotonic() + 3 < deadline, 'Build chain deadline before launch')
            request_path = evidence / (name + '-command.json')
            r.save(request_path, {'schema_version': 1, 'argv': argv, 'cwd': plan['checkout']})
            raw = request_path.read_bytes()
            receipt = supervisor.capture(str(request_path), r.digest(raw),
                python=plan['pythons'][0]['path'], environment=environment,
                output=str(evidence / (name + '-capture')), supervisor_sha256=r.SUPERVISOR_SHA,
                raw_limit=16 * 1024 * 1024, cleanup_reserve=2.0,
                deadline=min(deadline, time.monotonic() + seconds))
            r.captured_ok(receipt); records.append(receipt)
        actual(plan); r.need(read_plan(path, expected) == plan and time.monotonic() < deadline, 'Final build gate')
        result = {'status': 'EXISTING_THREE_ABI_BUILD_CAPTURED_NOT_ACCEPTANCE',
                  'plan_sha256': expected, 'product_acceptance': False, 'records': records}
        r.save(evidence / 'RESULT.json', result); return result
    except BaseException as error:
        try:
            r.save(evidence / 'HOLD.json', {'status': 'HOLD_MAC_BUILD', 'error_type': type(error).__name__,
                   'product_acceptance': False, 'plan_sha256': expected})
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
        result = {'status': 'MAC_BUILD_COMMANDS_PREVIEW_NOT_EXECUTED', 'commands': commands(plan),
                  'product_acceptance': False, 'supervisor_cross_macos': 'HOLD_UNTESTED'}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
