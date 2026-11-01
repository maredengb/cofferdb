"""Connect observed Mac15 inputs to existing bounded build and runtime helpers."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import time
import threading
import types

PINS = {
    'runtime_bridge/runtime_bridge.py': 'e4a4978671957b18dc92fbd1cd8ea1d928ff46db2b88c36062d8b80f231fc4c4',
    'dynamic_inputs/dynamic_collect.py': '15b1354b782b51e99a01bfa563b004e5c38573ad41674259bc882db84dd1cc1c',
    'native_dependencies/native_closure.py': '71011b78c440b1563b96ab0935c7528df5e776ab6915fa5c1baa76a10c55dbb4',
    'launch_bridge/launch_bridge.py': '6982447a5523d18264555cd378d4ace7e181b03d8a9720d8e45b8ae3082b7c55',
    'tool_observer/tool_observer.py': 'ee22e33ced5febcd8020211b3c1b50a29bba31706ed0b4bcd767d1ab775a6554',
}
INPUT_MANIFEST = '4ec41541f34f51b594ed8a97eb99c7a599f436541fe81d3eb552b41bb14cc9d7'
WHOLE_SECONDS = 7200


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def bootstrap(root):
    """Load the fixed file reader first, without using an import search path."""
    root = Path(root)
    need(root.is_absolute() and '..' not in root.parts
         and not any(p.is_symlink() for p in (root, *root.parents)), 'Literal helper root required')
    selected = root / 'runtime_bridge/runtime_bridge.py'
    fd = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 131072
             and not info.st_mode & 0o022, 'Protected bounded bridge required')
        raw = os.read(fd, 131073)
        need(len(raw) == info.st_size and sha(raw) == PINS['runtime_bridge/runtime_bridge.py'],
             'Runtime bridge identity differs')
    finally:
        os.close(fd)
    module = types.ModuleType('workflow_runtime_bridge'); module.__file__ = str(selected)
    exec(compile(raw, str(selected), 'exec', dont_inherit=True), module.__dict__)
    modules = {'runtime': module}
    for key, relative in [('collector', 'dynamic_inputs/dynamic_collect.py'),
                          ('native', 'native_dependencies/native_closure.py'),
                          ('launch', 'launch_bridge/launch_bridge.py'),
                          ('tools', 'tool_observer/tool_observer.py')]:
        path = root / relative; raw = module.read_file(str(path), 131072)
        need(sha(raw) == PINS[relative], 'Component identity differs: ' + relative)
        item = types.ModuleType('workflow_' + key); item.__file__ = str(path)
        exec(compile(raw, str(path), 'exec', dont_inherit=True), item.__dict__)
        modules[key] = item
    modules['originals'] = module.load_helpers(str(root))
    return modules


class WholeDeadline(TimeoutError):
    pass


@contextmanager
def whole_budget(clock=time.monotonic):
    """One parent timer, including reads; child captures retain their own budgets."""
    need(threading.current_thread() is threading.main_thread(), 'Parent timer requires the main thread')
    need(signal.SIGALRM not in signal.pthread_sigmask(signal.SIG_BLOCK, []),
         'Inherited SIGALRM must not be blocked')
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'Existing real-time timer must not be replaced')
    previous = signal.getsignal(signal.SIGALRM)
    started = clock(); deadline = started + WHOLE_SECONDS
    expired = False
    def expire(_number, _frame):
        nonlocal expired
        expired = True
        raise WholeDeadline('Mac workflow whole deadline')
    def check():
        need(not expired and clock() < deadline, 'Mac workflow whole deadline exhausted')
    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, WHOLE_SECONDS)
        yield {'started': started, 'deadline': deadline, 'check': check}
        check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def input_document(b, filename, expected):
    b.checksum(expected); raw = b.read_file(filename, 2 ** 20)
    need(sha(raw) == expected, 'Workflow input bytes changed')
    value = b.decode(raw)
    fields = {'schema_version', 'kind', 'source_commit', 'repository_id', 'checkout', 'cargo_home',
              'selected_tools', 'driver_contract', 'output'}
    need(type(value) is dict and set(value) == fields and type(value['schema_version']) is int
         and value['schema_version'] == 1 and value['kind'] == 'mac15_complete_workflow_inputs_1',
         'Exact workflow input shape required')
    need(type(value['repository_id']) is int and value['repository_id'] == 1422246026,
         'Selected repository identity differs')
    need(type(value['source_commit']) is str and len(value['source_commit']) == 40
         and all(c in '0123456789abcdef' for c in value['source_commit'])
         and value['source_commit'] != '0' * 40, 'Fixed public source required')
    for key in ('checkout', 'cargo_home', 'output'): b.path(value[key])
    b.identity(value['driver_contract'])
    return value


def github_identity(value):
    need(os.environ.get('GITHUB_REPOSITORY_ID') == str(value['repository_id'])
         and os.environ.get('GITHUB_REPOSITORY') in ('maredengb/coffered', 'maredengb/cofferdb')
         and os.environ.get('GITHUB_SHA') == value['source_commit']
         and os.environ.get('GITHUB_RUN_ID', '').isdigit()
         and os.environ.get('GITHUB_RUN_ATTEMPT', '').isdigit(), 'Actual selected GitHub job identity differs')
    return {name: os.environ[name] for name in ('GITHUB_REPOSITORY_ID', 'GITHUB_REPOSITORY',
                                               'GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT')}


def pipeline(value, modules, root, budget, recheck, *, retain_check=None):
    """Actual functions produce each next input; no fabricated success receipts."""
    b, c, n, launch, tools = [modules[key] for key in ('runtime', 'collector', 'native', 'launch', 'tools')]
    originals = modules['originals']; runner = originals['runner.py']; plan = originals['dynamic_inputs/dynamic_plan.py']
    runner.host_gate(); job = github_identity(value); budget['check'](); recheck()
    output = b.path(value['output'])
    protected = [b.path(value['checkout']), b.path(value['cargo_home']), b.path(root),
                 b.path(value['driver_contract']['path'])]
    need(all(not runner.beneath(output, p) and not runner.beneath(p, output) for p in protected),
         'Workflow output overlaps input')
    out = runner.private_new(str(output))
    entered = []; completed = []; refs = {}; captures = []
    def read(ref, bound=16 * 1024 ** 2):
        raw = b.read_file(ref['path'], bound); need(sha(raw) == ref['sha256'], 'Original output changed')
        return b.decode(raw)
    def reference(path, expected):
        b.checksum(expected)
        ref = {'path': str(path), 'sha256': expected}
        read(ref); refs[str(path)] = ref
        return ref
    def write(name, data):
        path = out / name
        expected = sha(runner.canonical(data))
        runner.save(path, data)
        return reference(path, expected)
    def fresh():
        budget['check'](); recheck()
        for name, expected in PINS.items():
            need(sha(b.read_file(str(Path(root) / name), 131072)) == expected, 'Component changed')
        budget['check']()
    def step(name, action):
        fresh(); entered.append(name)
        result = action(); budget['check'](); fresh(); completed.append(name)
        return result
    try:
        write('INTENT.json', {'kind': value['kind'], 'source_commit': value['source_commit'], 'job': job,
              'started_monotonic': budget['started'], 'deadline_monotonic': budget['deadline'],
              'product_acceptance': False, 'automatic_retry': False})
        for name in ('data', 'work', 'evidence', 'venvs', 'gates'):
            (out / name).mkdir(mode=0o700)
        driver = read(value['driver_contract'], 2 ** 20)
        need(sha(canonical(driver)) == value['driver_contract']['sha256'], 'Canonical external driver contract required')
        # No external driver path can borrow writable output permissions.
        need(type(driver) is dict and set(driver) == {'runtime_drivers', 'network_probe', 'command_seconds'}
             and type(driver['runtime_drivers']) is list and len(driver['runtime_drivers']) == 3,
             'Complete external driver contract required')
        placeholders = ('wheel', 'wheel_sha256', 'native_sha256', 'venv', 'source_metadata',
                        'source_input_sha256', 'sdk_version', 'engine_version', 'smoke_work')
        for item, minor in zip(driver['runtime_drivers'], (12, 13, 14)):
            # Structural only: these tokens are not recorded as actual artifacts.
            b.runtime_driver(item, {'{' + key + '}': 'uninstantiated' for key in placeholders}, [], minor)
            for row in [{'path': item['path'], 'sha256': item['sha256']}, *item['inputs']]:
                b.identity(row); runner.row(row)
                need(not runner.beneath(b.path(row['path']), out), 'External driver input overlaps mutable output')
        configuration = {'schema_version': 1, 'kind': 'mac15_dynamic_input_config_1',
            'source_repository': 'maredengb/coffered', 'source_commit': value['source_commit'],
            'helper_sha256': dict(plan.HELPERS), 'actions': dict(plan.ACTIONS),
            'python_providers': list(plan.PROVIDERS), 'selection_method': plan.METHOD,
            'native_closure_gate': 'independent-exact-closure-input-v1',
            'driver_contract_sha256': value['driver_contract']['sha256'],
            'candidate_manifest_sha256': INPUT_MANIFEST,
            'native_closure_validator_sha256': PINS['native_dependencies/native_closure.py']}
        plan.configuration(configuration); plan.verify_candidate_manifest(INPUT_MANIFEST)
        config_ref = write('data/config.json', configuration)
        layout = {'checkout': value['checkout'], 'cargo_home': value['cargo_home'],
                  'work_parent': str(out / 'work'), 'evidence_parent': str(out / 'evidence'),
                  'venv_parent': str(out / 'venvs')}
        layout_ref = write('data/layout.json', layout)
        collected = step('collect', lambda: c.collect(configuration, str(Path(root).parent), str(out / 'collect')))
        collect_snapshot_ref = reference(out / 'collect/SNAPSHOT.json', collected['snapshot_sha256'])
        snapshot = read(collect_snapshot_ref)
        index_ref = reference(out / 'collect/READONLY-INDEX.json', snapshot['readonly_index_sha256'])
        reference(out / 'collect/OBSERVATIONS.json', snapshot['observations_sha256'])
        index = read(index_ref)
        need(snapshot['config_sha256'] == config_ref['sha256']
             and snapshot['readonly_index_sha256'] == index_ref['sha256']
             and snapshot['run_identity'] == {'run_id': job['GITHUB_RUN_ID'],
                 'attempt': job['GITHUB_RUN_ATTEMPT'], 'nonce': 'collect'}, 'Collector job/input binding differs')
        def collected_inputs():
            fresh(); need(read(collect_snapshot_ref) == snapshot and read(index_ref) == index,
                          'Original collector outputs changed')
        observed_tools = step('tools', lambda: tools.observe(b, root, snapshot, value['selected_tools'], value['checkout'],
                                           str(out / 'tools'), input_check=collected_inputs))
        reference(out / 'tools/RESULT.json', sha(runner.canonical(observed_tools)))
        tool_snapshot_ref = reference(out / 'tools/SNAPSHOT.json', observed_tools['snapshot_sha256'])
        tool_snapshot = read(tool_snapshot_ref)
        native_ref = {'path': str(out / 'data/native-report.json')}
        native = step('native', lambda: n.inspect(configuration, tool_snapshot, index,
            PINS['native_dependencies/native_closure.py'], native_ref['path'], c, plan))
        snapshot_ref = write('data/snapshot.json', native['snapshot'])
        report_ref = write('data/native-report.json', native['report'])
        closure_ref = write('data/native-closure.json', native['closure'])
        need(native['report_sha256'] == report_ref['sha256']
             and native['closure']['snapshot_sha256'] == snapshot_ref['sha256'], 'Actual native outputs differ')
        fresh_ref = write('data/fresh-inputs.json', {'snapshot': snapshot_ref, 'index': index_ref, 'report': report_ref})
        initial = step('derive', lambda: plan.derive(configuration, native['snapshot'], layout, driver, native['closure']))
        need(set(initial['plans']) == {'build', 'empty_venvs'} and initial['missing'] == [
             'actual new empty-venv bindings and actual wheel/driver inputs before runtime plan instantiation'],
             'Earlier technical inputs are missing')
        build = initial['plans']['build']; venvs = initial['plans']['empty_venvs']
        originals['workflow_build.py'].validate(build); originals['venv_prepare.py'].validate(venvs)
        def original(stage, name, original_plan):
            ref = write('data/' + name + '-plan.json', original_plan)
            observed = step(name, lambda: launch.execute(stage, ref['path'], ref['sha256'], ref['sha256'],
                fresh_ref['path'], fresh_ref['sha256'], str(out / 'gates' / name), root, b, n))
            result_ref = reference(Path(original_plan['evidence']) / 'RESULT.json',
                                   sha(runner.canonical(observed['original_result'])))
            result = read(result_ref)
            need(observed['status'] == 'ORIGINAL_STEP_CAPTURED_WITH_FRESH_INPUTS_NOT_ACCEPTANCE'
                 and observed['original_result'] == result and result['plan_sha256'] == ref['sha256'],
                 'Original step result differs from actual invocation')
            for receipt in result['records']:
                runner.captured_ok(receipt); captures.append(receipt)
            return result_ref
        build_result = original('build', 'build', build)
        empty_result = original('venvs', 'venvs', venvs)
        # The builder prints the exact manifest bytes into its originally sealed stdout.
        build_stdout = read(build_result)['records'][1]['raw_streams']['stdout']
        manifest_raw = b.read_file(build_stdout['path'], 16 * 1024 ** 2)
        need(len(manifest_raw) == build_stdout['size'] and sha(manifest_raw) == build_stdout['sha256'],
             'Original builder stdout changed')
        b.decode(manifest_raw)
        manifest_ref = reference(Path(build['work']) / 'wheels/build-manifest.json', build_stdout['sha256'])
        bundle = {'schema_version': 1, 'kind': 'mac15_runtime_bridge_inputs_1', 'config': config_ref,
                  'snapshot': snapshot_ref, 'layout': layout_ref, 'driver_contract': value['driver_contract'],
                  'native_closure': closure_ref, 'build_result': build_result,
                  'build_manifest': manifest_ref,
                  'empty_venvs_result': empty_result}
        write('data/runtime-inputs.json', bundle)
        actual = step('runtime-plans', lambda: b.instantiate(bundle, originals))
        need(actual['status'] == 'RUNTIME_PLANS_DERIVED_NOT_EXECUTED'
             and set(actual['plans']) == {'cp312', 'cp313', 'cp314'}, 'Three actual runtime plans required')
        runtime_results = {}
        for minor in (12, 13, 14):
            name = 'cp3' + str(minor); entry = actual['plans'][name]
            need(sha(runner.canonical(entry['plan'])) == entry['plan_sha256'], 'Derived runtime digest differs')
            runtime_results[name] = original('runtime', name, entry['plan'])
        def verify_outputs():
            fresh()
            for receipt in [*observed_tools['records'], *captures]:
                runner.captured_ok(receipt)
                need(set(receipt['raw_streams']) == {'stdout', 'stderr', 'status'}, 'Original three stream seals required')
                for row in receipt['raw_streams'].values():
                    raw = b.read_file(row['path'], 16 * 1024 ** 2)
                    need(len(raw) == row['size'] and sha(raw) == row['sha256'], 'Original captured output changed')
            for ref in refs.values():
                read(ref)
            need(len(captures) == 11, 'Two build, three venv, and six runtime captures required')
            for artifact in actual['observed']:
                need(sha(b.read_file(artifact['wheel'])) == artifact['wheel_sha256'], 'Final wheel changed')
            budget['check']()
        verify_outputs()
        result = {'status': 'MAC15_BUILD_AND_THREE_RUNTIME_CELLS_COMPLETED_NOT_ACCEPTANCE',
            'source_commit': actual['source_commit'], 'source_tree': actual['source_tree'],
            'actual_artifacts': actual['observed'], 'runtime_results': runtime_results,
            'entered_steps': entered, 'completed_steps': completed, 'captured_commands': len(captures),
            'product_acceptance': False, 'completion_requires_successful_job_exit': True}
        write('RESULT.json', result); verify_outputs()
        if retain_check is not None:
            retain_check(verify_outputs)
        return result
    except BaseException as error:
        try:
            runner.save(out / 'HOLD.json', {'status': 'HOLD_MAC15_WORKFLOW', 'error_type': type(error).__name__,
                'entered_steps': entered, 'completed_steps': completed, 'automatic_retry': False,
                'product_acceptance': False, 'interrupted_process_cleanup_requires_original_receipts': True})
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('verify', 'run'), default='verify')
    for key in ('inputs', 'inputs-sha256', 'helper-root'): parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.action == 'verify':
        print('{"status":"MAC15_WORKFLOW_NOT_EXECUTED","product_acceptance":false}'); return 0
    with whole_budget() as budget:
        modules = bootstrap(args.helper_root); b = modules['runtime']
        value = input_document(b, args.inputs, args.inputs_sha256)
        def recheck():
            need(input_document(b, args.inputs, args.inputs_sha256) == value, 'Workflow inputs changed')
            row = value['driver_contract']
            need(sha(b.read_file(row['path'], 2 ** 20)) == row['sha256'], 'External driver contract changed')
        result = pipeline(value, modules, args.helper_root, budget, recheck)
        budget['check']()
        print(canonical(result).decode(), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
