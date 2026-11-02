"""Attach fresh input checks to each original Mac15 supervisor capture.

Default verification performs no IO. An explicit step uses the original
validators, sandbox, and bounded supervisor without changing their files.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import types

BRIDGE_SHA = 'f60fdc6217fbf705291b6781630b46921d4c0e278166fd498fa32b2e68285b4f'
NATIVE_SHA = 'f334fb3f04257a4d20e664ca524b972478a0d4dccebd838f16b71711dbb620dc'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def load_bridge(filename):
    """Bootstrap only the fixed, inert bridge module from protected bytes."""
    path = Path(filename)
    need(path.is_absolute() and str(path) == filename and '..' not in path.parts
         and not any(item.is_symlink() for item in (path, *path.parents)), 'Literal bridge path required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 131072
             and not info.st_mode & 0o022, 'Bounded protected bridge source required')
        raw = os.read(fd, 131073)
        need(len(raw) == info.st_size and digest(raw) == BRIDGE_SHA, 'Pinned runtime bridge source required')
    finally:
        os.close(fd)
    module = types.ModuleType('launch_runtime_bridge'); module.__file__ = filename
    exec(compile(raw, filename, 'exec', dont_inherit=True), module.__dict__)
    return module


def load_native(bridge, filename):
    raw = bridge.read_file(filename, 131072)
    need(digest(raw) == NATIVE_SHA, 'Pinned native graph source required')
    module = types.ModuleType('launch_native_closure'); module.__file__ = filename
    exec(compile(raw, filename, 'exec', dont_inherit=True), module.__dict__)
    return module


def plan_input_binding(step, plan, snapshot, helper_root, helper_hashes):
    """The graph checked now must describe the command that will run next."""
    root = Path(helper_root)
    expected_paths = {name: str(root.parent / name if name == 'q_process_supervisor.py' else root / name)
                      for name in helper_hashes if name != 'dynamic_inputs/dynamic_plan.py'}
    need(snapshot['helper_paths'] == expected_paths, 'Selected helper paths differ from observed inputs')
    supervisor = {'path': snapshot['helper_paths']['q_process_supervisor.py'],
                  'sha256': helper_hashes['q_process_supervisor.py']}
    need(plan['supervisor'] == supervisor, 'Plan supervisor differs from observed input')
    runtimes = snapshot['runtimes']
    observed = [{'minor': runtime['minor'], **runtime['binary']} for runtime in runtimes]
    if step == 'build':
        need(plan['pythons'] == observed and plan['public_commit'] == snapshot['source_commit'],
             'Build interpreter/source differs from fresh snapshot')
        for name in ('prepare', 'builder', 'cargo', 'rustc', 'maturin'):
            need(plan[name] == snapshot['tools'][name], 'Build tool differs from fresh snapshot: ' + name)
    elif step == 'venvs':
        actual = [{key: row[key] for key in ('minor', 'path', 'sha256')} for row in plan['runtimes']]
        need(actual == observed, 'Venv interpreter differs from fresh snapshot')
    elif step == 'runtime':
        matches = [runtime for runtime in runtimes if runtime['minor'] == plan['python']['minor']]
        need(len(matches) == 1, 'One fresh runtime ABI required')
        runtime = matches[0]
        need(plan['python']['base'] == runtime['binary']['path']
             and plan['python']['sha256'] == runtime['binary']['sha256']
             and plan['stdlib'] == runtime['stdlib'] and plan['runtime_files'] == runtime['runtime_files']
             and plan['dependencies'] == [], 'Runtime closure differs from fresh snapshot')
        for name in ('sandbox', 'compiler'):
            need(plan[name] == snapshot['tools'][name], 'Runtime tool differs from fresh snapshot: ' + name)
        need(plan['probe_sha256'] == helper_hashes['probe.py'], 'Runtime probe differs from fixed source')
    else:
        raise ValueError('Unknown plan binding step')


def current_inputs(bridge, native, helper_root, filename, expected, step, plan):
    """Re-read exact documents and code, then the complete observed inputs."""
    raw = bridge.read_file(filename, 2 ** 20)
    need(digest(raw) == expected, 'Fresh input bundle changed')
    bundle = bridge.decode(raw)
    need(type(bundle) is dict and set(bundle) == {'snapshot', 'index', 'report'}, 'Exact fresh input bundle')
    values = {}
    for key, row in bundle.items():
        bridge.identity(row)
        value = bridge.read_file(row['path'], 16 * 1024 ** 2)
        need(digest(value) == row['sha256'], 'Fresh bound document changed')
        values[key] = bridge.decode(value)
    plan_input_binding(step, plan, values['snapshot'], helper_root, bridge.HELPERS)
    need(digest(bridge.read_file(bridge.__file__, 131072)) == BRIDGE_SHA
         and digest(bridge.read_file(native.__file__, 131072)) == NATIVE_SHA
         and values['report']['validator_sha256'] == NATIVE_SHA, 'Fresh entry code changed')
    root = bridge.path(helper_root)
    for name, expected_sha in bridge.HELPERS.items():
        selected = root.parent / name if name == 'q_process_supervisor.py' else root / name
        need(digest(bridge.read_file(str(selected), 131072)) == expected_sha, 'Existing helper changed')
    native.same_host(values['snapshot'])
    observer, _ = native.helpers(str(root / 'dynamic_inputs'))
    result = native.fresh(values['snapshot'], values['index'], values['report'], observer)
    # Original byte references must still agree after the full observation.
    need(digest(bridge.read_file(filename, 2 ** 20)) == expected, 'Fresh bundle changed during observation')
    for row in bundle.values():
        need(digest(bridge.read_file(row['path'], 16 * 1024 ** 2)) == row['sha256'],
             'Fresh document changed during observation')
    return result


class Guard:
    def __init__(self, before, record, maximum_calls):
        self.before = before
        self.record = record
        self.maximum_calls = maximum_calls
        self.calls = 0
        self.supervisor_loads = 0

    def wrap(self, capture):
        def guarded(request, expected, **arguments):
            need(self.calls < self.maximum_calls, 'Unexpected extra helper launch')
            deadline = arguments.get('deadline')
            need(type(deadline) in (int, float) and time.monotonic() + 3 < deadline,
                 'Original command deadline exhausted before fresh gate')
            started = time.monotonic()
            observation = self.before()
            need(time.monotonic() + 3 < deadline, 'Original command deadline exhausted during fresh gate')
            self.record(self.calls, {'status': 'FRESH_BEFORE_ORIGINAL_CAPTURE', 'request_sha256': expected,
                'request_path': request, 'started_monotonic': started, 'finished_monotonic': time.monotonic(),
                'original_deadline_monotonic': deadline, 'observation': observation, 'product_acceptance': False})
            self.calls += 1
            # Forward the exact original deadline and arguments. Gate time
            # consumes the existing budget; it never extends a stage.
            return capture(request, expected, **arguments)
        return guarded


@contextmanager
def guarded_supervisor(shared_runner, guard):
    original = shared_runner.load_supervisor
    def load(plan):
        need(guard.supervisor_loads == 0, 'One original supervisor per helper invocation')
        supervisor = original(plan)
        guard.supervisor_loads += 1
        return types.SimpleNamespace(capture=guard.wrap(supervisor.capture))
    shared_runner.load_supervisor = load
    try:
        yield
    finally:
        shared_runner.load_supervisor = original


def execute(step, plan_path, plan_sha256, approved, fresh_path, fresh_sha256,
            gate_evidence, helper_root, bridge, native):
    need(step in ('build', 'venvs', 'runtime'), 'Known original step required')
    need(approved == plan_sha256, 'Exact selected plan permission binding required')
    for value in (plan_sha256, fresh_sha256): bridge.checksum(value)
    modules = bridge.load_helpers(helper_root)
    selected = {'build': 'workflow_build.py', 'venvs': 'venv_prepare.py', 'runtime': 'runner.py'}[step]
    helper = modules[selected]
    runner = helper if step == 'runtime' else helper.r
    read_plan = helper.validate if step == 'runtime' else helper.read_plan
    plan = read_plan(plan_path, plan_sha256)
    runner.host_gate()
    before = lambda: current_inputs(bridge, native, helper_root, fresh_path, fresh_sha256, step, plan)
    before()
    destination = bridge.path(gate_evidence)
    protected = [bridge.path(v) for v in (plan_path, fresh_path, helper_root, bridge.__file__, native.__file__)]
    for row in bridge.decode(bridge.read_file(fresh_path, 2 ** 20)).values():
        bridge.identity(row)
        protected.append(bridge.path(row['path']))
    mutable = [bridge.path(plan[k]) for k in ('work', 'venv', 'evidence') if k in plan]
    if step == 'venvs': mutable += [bridge.path(row['venv']) for row in plan['runtimes']]
    need(all(not runner.beneath(destination, other) and not runner.beneath(other, destination)
             for other in [*protected, *mutable]), 'Separate fresh observation evidence required')
    need(all(not runner.beneath(item, output) for item in protected for output in mutable),
         'Fresh documents and maintenance code cannot be runtime-writable')
    evidence = runner.private_new(str(destination))
    runner.save(evidence / 'INTENT.json', {'kind': 'mac15_original_launch_fresh_gate_1', 'step': step,
        'plan_sha256': plan_sha256, 'fresh_inputs_sha256': fresh_sha256,
        'bridge_sha256': BRIDGE_SHA, 'native_validator_sha256': NATIVE_SHA, 'product_acceptance': False})
    count = 3 if step == 'venvs' else 2
    def record(number, value): runner.save(evidence / ('launch-%02d.json' % number), value)
    guard = Guard(before, record, count)
    try:
        with guarded_supervisor(runner, guard):
            result = helper.run(plan_path, plan_sha256, approved)
        need(guard.calls == count and guard.supervisor_loads == 1, 'Original helper capture count differs')
        final_observation = before()
        summary = {'status': 'ORIGINAL_STEP_CAPTURED_WITH_FRESH_INPUTS_NOT_ACCEPTANCE', 'step': step,
                   'captures': guard.calls, 'original_result': result, 'final_observation': final_observation,
                   'product_acceptance': False}
        runner.save(evidence / 'RESULT.json', summary)
        return summary
    except BaseException as error:
        try:
            runner.save(evidence / 'HOLD.json', {'status': 'HOLD_FRESH_OR_ORIGINAL_STEP',
                'step': step, 'captures_entered': guard.calls, 'error_type': type(error).__name__,
                'product_acceptance': False, 'automatic_retry': False})
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'run'), nargs='?', default='verify')
    for key in ('step', 'plan', 'plan-sha256', 'approved-plan-sha256', 'fresh-inputs', 'fresh-inputs-sha256',
                'gate-evidence', 'helper-root', 'runtime-bridge', 'native-validator'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.action == 'verify':
        result = {'status': 'LAUNCH_BRIDGE_NOT_EXECUTED', 'product_acceptance': False}
    else:
        bridge = load_bridge(args.runtime_bridge)
        native = load_native(bridge, args.native_validator)
        result = execute(args.step, args.plan, args.plan_sha256, args.approved_plan_sha256,
                         args.fresh_inputs, args.fresh_inputs_sha256, args.gate_evidence,
                         args.helper_root, bridge, native)
    print(canonical(result).decode(), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
