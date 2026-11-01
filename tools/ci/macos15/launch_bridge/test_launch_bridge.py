"""Launch ordering/failure controls; capture is replaced, no child is started."""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
s = importlib.util.spec_from_file_location('launch_bridge_test', HERE / 'launch_bridge.py')
l = importlib.util.module_from_spec(s); s.loader.exec_module(l)
BRIDGE = HELPER_ROOT = NATIVE_PATH = None


def receipt():
    return {'state': 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE', 'actual_command_exit_code': 0,
            'process_group_gone': True, 'raw_eof': {'stdout': True, 'stderr': True},
            'unread_tail_possible': False, 'failure_reason': None}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cofferdb-launch-fixture-', dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.popen = patch.object(subprocess, 'Popen', side_effect=AssertionError('NO CHILD PROCESS'))
        self.popen.start(); self.addCleanup(self.popen.stop)
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('NO NETWORK'))
        self.network.start(); self.addCleanup(self.network.stop)

    def test_default_is_inert(self):
        args = type('Args', (), {'action': 'verify'})()
        with patch.object(l.argparse.ArgumentParser, 'parse_args', return_value=args), \
             patch.object(l, 'load_bridge', side_effect=AssertionError('NO INPUT READ')), \
             patch('sys.stdout', new=io.StringIO()) as stream:
            self.assertEqual(l.main(), 0)
        self.assertIn('NOT_EXECUTED', stream.getvalue())

    def test_each_capture_rechecks_before_forwarding_original_deadline(self):
        events = []; deadline = time.monotonic() + 30
        guard = l.Guard(lambda: events.append('fresh') or {'checked': True},
                        lambda i, value: events.append(('record', i)), 2)
        def capture(*args, **kw):
            events.append(('capture', args, kw)); return receipt()
        wrapped = guard.wrap(capture)
        for number in range(2):
            self.assertEqual(wrapped('/fixture/request', 'f' * 64, deadline=deadline, python='/fixture/python'), receipt())
        self.assertEqual([item if isinstance(item, str) else item[0] for item in events],
                         ['fresh', 'record', 'capture', 'fresh', 'record', 'capture'])
        self.assertEqual(events[-1][2]['deadline'], deadline)
        self.assertEqual(guard.calls, 2)

    def test_fresh_failure_and_record_failure_prevent_capture(self):
        for gate, record in ((lambda: (_ for _ in ()).throw(ValueError('changed')), lambda *_: None),
                             (lambda: {}, lambda *_: (_ for _ in ()).throw(OSError('record failed')))):
            called = []
            guard = l.Guard(gate, record, 1)
            with self.assertRaises((ValueError, OSError)):
                guard.wrap(lambda *_a, **_k: called.append(True))('/fixture/request', 'a' * 64, deadline=time.monotonic() + 30)
            self.assertEqual(called, [])

    def test_fresh_elapsed_time_consumes_original_deadline(self):
        clock = [100.0]; called = []
        def gate(): clock[0] = 109.0; return {}
        guard = l.Guard(gate, lambda *_: None, 1)
        with patch.object(l.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ValueError, 'during fresh'):
                guard.wrap(lambda *_a, **_k: called.append(True))('/fixture/request', 'a' * 64, deadline=110.0)
        self.assertEqual(called, [])

    def test_extra_launch_is_rejected(self):
        guard = l.Guard(lambda: {}, lambda *_: None, 1)
        capture = guard.wrap(lambda *_a, **_k: receipt())
        capture('/fixture/request', 'a' * 64, deadline=time.monotonic() + 30)
        with self.assertRaisesRegex(ValueError, 'extra helper launch'):
            capture('/fixture/request2', 'b' * 64, deadline=time.monotonic() + 30)

    def test_loader_restored_after_success_or_failure(self):
        for fail in (False, True):
            capture = lambda *_a, **_k: receipt()
            original = lambda plan: types.SimpleNamespace(capture=capture)
            runner = types.SimpleNamespace(load_supervisor=original)
            guard = l.Guard(lambda: {}, lambda *_: None, 1)
            try:
                with l.guarded_supervisor(runner, guard):
                    self.assertIsNot(runner.load_supervisor, original)
                    runner.load_supervisor({}).capture('/fixture/request', 'f' * 64, deadline=time.monotonic() + 30)
                    if fail: raise ValueError('fixture failure')
            except ValueError:
                self.assertTrue(fail)
            self.assertIs(runner.load_supervisor, original)

    def test_second_supervisor_load_rejected(self):
        runner = types.SimpleNamespace(load_supervisor=lambda plan: types.SimpleNamespace(capture=lambda *_a, **_k: receipt()))
        with l.guarded_supervisor(runner, l.Guard(lambda: {}, lambda *_: None, 2)):
            runner.load_supervisor({})
            with self.assertRaisesRegex(ValueError, 'One original'): runner.load_supervisor({})

    def file(self, name, data=b'fixture', executable=False):
        p = self.root / name; p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        p.write_bytes(data); p.chmod(0o700 if executable else 0o600)
        return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest()}

    def build_case(self, fail_at=None):
        modules = BRIDGE.load_helpers(HELPER_ROOT)
        helper = modules['workflow_build.py']; r = helper.r
        checkout = self.root / 'checkout'; checkout.mkdir(mode=0o700)
        cargo_home = self.root / 'cargo-home'; cargo_home.mkdir(mode=0o700)
        plan = {'schema_version': 1, 'kind': 'macos15_three_abi_supervised_build_1',
                'checkout': str(checkout), 'public_commit': '1' * 40,
                'prepare': self.file('checkout/tools/prepare_source.py'), 'builder': self.file('checkout/tools/build_python.py'),
                'pythons': [{'minor': n, **self.file('input/python' + str(n), executable=True)} for n in (12, 13, 14)],
                'cargo': self.file('input/cargo', executable=True), 'rustc': self.file('input/rustc', executable=True),
                'maturin': self.file('input/maturin', executable=True), 'cargo_home': str(cargo_home),
                'work': str(self.root / 'work'), 'evidence': str(self.root / 'original-evidence'),
                'supervisor': {'path': str(Path(HELPER_ROOT).parent / 'q_process_supervisor.py'), 'sha256': r.SUPERVISOR_SHA}}
        row = self.file('plan.json', r.canonical(plan))
        fresh = self.file('fresh.json', l.canonical({k: self.file(k + '.json', b'{}') for k in ('snapshot', 'index', 'report')}))
        gates = []; captures = []
        def gate(*_a, **_k):
            gates.append(len(gates))
            if len(gates) == fail_at: raise ValueError('fixture readonly drift')
            return {'status': 'FIXTURE_FRESH_NOT_REAL_PLATFORM'}
        def capture(path, expected, **kw):
            captures.append(json.loads(Path(path).read_bytes()))
            return receipt()
        original = lambda plan: types.SimpleNamespace(capture=capture)
        r.load_supervisor = original
        with patch.object(BRIDGE, 'load_helpers', return_value=modules), patch.object(r, 'host_gate', return_value={}), \
             patch.object(l, 'current_inputs', side_effect=gate):
            args = ('build', row['path'], row['sha256'], row['sha256'], fresh['path'], fresh['sha256'],
                    str(self.root / 'gates'), HELPER_ROOT, BRIDGE, types.SimpleNamespace(__file__=NATIVE_PATH))
            if fail_at:
                with self.assertRaisesRegex(ValueError, 'readonly drift'): l.execute(*args)
                result = None
            else: result = l.execute(*args)
        self.assertIs(r.load_supervisor, original)
        return result, gates, captures

    def test_original_build_loop_has_gate_before_both_launches(self):
        result, gates, captures = self.build_case()
        self.assertEqual(len(gates), 4)  # entry, prepare, build, final
        self.assertEqual(len(captures), 2)
        self.assertIn('prepare_source.py', captures[0]['argv'][4])
        self.assertIn('build_python.py', captures[1]['argv'][4])
        self.assertEqual(result['captures'], 2)
        self.assertTrue((self.root / 'gates/launch-01.json').is_file())

    def test_second_launch_drift_preserves_prefix_and_stops(self):
        result, gates, captures = self.build_case(fail_at=3)
        self.assertEqual(len(captures), 1)
        self.assertTrue((self.root / 'original-evidence/HOLD.json').is_file())
        self.assertTrue((self.root / 'gates/HOLD.json').is_file())
        self.assertFalse((self.root / 'gates/RESULT.json').exists())

    def test_changed_or_unprotected_bridge_code_rejected_before_compile(self):
        row = self.file('not-bridge.py', b'raise RuntimeError("must never run")\n')
        with self.assertRaisesRegex(ValueError, 'Pinned'): l.load_bridge(row['path'])
        Path(row['path']).chmod(0o666)
        with self.assertRaisesRegex(ValueError, 'protected'): l.load_bridge(row['path'])


    def bindings(self):
        root = Path(HELPER_ROOT)
        paths = {name: str(root.parent / name if name == 'q_process_supervisor.py' else root / name)
                 for name in BRIDGE.HELPERS if name != 'dynamic_inputs/dynamic_plan.py'}
        runtimes = [{'minor': n, 'binary': {'path': '/fixture/cp3' + str(n), 'sha256': str(n)[0] * 64},
                     'stdlib': '/fixture/stdlib3' + str(n), 'runtime_files': []} for n in (12, 13, 14)]
        tools = {name: {'path': '/fixture/' + name, 'sha256': 'c' * 64}
                 for name in ('prepare', 'builder', 'cargo', 'rustc', 'maturin', 'sandbox', 'compiler')}
        snapshot = {'helper_paths': paths, 'runtimes': runtimes, 'tools': tools, 'source_commit': '1' * 40}
        supervisor = {'path': paths['q_process_supervisor.py'], 'sha256': BRIDGE.HELPERS['q_process_supervisor.py']}
        build = {'supervisor': supervisor, 'pythons': [{'minor': x['minor'], **x['binary']} for x in runtimes],
                 'public_commit': snapshot['source_commit'], **{name: tools[name] for name in ('prepare', 'builder', 'cargo', 'rustc', 'maturin')}}
        venvs = {'supervisor': supervisor, 'runtimes': [{**row, 'venv': '/fixture/venv' + str(row['minor'])} for row in build['pythons']]}
        runtime = {'supervisor': supervisor, 'python': {'minor': 12, 'base': runtimes[0]['binary']['path'],
                   'sha256': runtimes[0]['binary']['sha256']}, 'stdlib': runtimes[0]['stdlib'],
                   'runtime_files': [], 'dependencies': [], 'sandbox': tools['sandbox'], 'compiler': tools['compiler'],
                   'probe_sha256': BRIDGE.HELPERS['probe.py']}
        return snapshot, {'build': build, 'venvs': venvs, 'runtime': runtime}

    def test_matching_plan_inputs_bind_all_original_step_types(self):
        snapshot, plans = self.bindings()
        for step, plan in plans.items(): l.plan_input_binding(step, plan, snapshot, HELPER_ROOT, BRIDGE.HELPERS)

    def test_unrelated_valid_runtime_cannot_borrow_fresh_graph(self):
        snapshot, plans = self.bindings()
        edits = [lambda p: p['python'].update(base='/fixture/other-python'),
                 lambda p: p.update(stdlib='/fixture/other-stdlib'),
                 lambda p: p.update(runtime_files=[{'path': '/fixture/unverified.dylib', 'sha256': 'd' * 64}]),
                 lambda p: p.update(dependencies=['/fixture/wider']),
                 lambda p: p['compiler'].update(path='/fixture/other-compiler')]
        for edit in edits:
            plan = copy.deepcopy(plans['runtime']); edit(plan)
            with self.assertRaisesRegex(ValueError, 'differs'):
                l.plan_input_binding('runtime', plan, snapshot, HELPER_ROOT, BRIDGE.HELPERS)

    def test_build_source_tool_or_python_mismatch_cannot_borrow_fresh(self):
        snapshot, plans = self.bindings()
        edits = [lambda p: p.update(public_commit='2' * 40),
                 lambda p: p['pythons'][1].update(path='/fixture/other-python'),
                 lambda p: p['maturin'].update(sha256='b' * 64)]
        for edit in edits:
            plan = copy.deepcopy(plans['build']); edit(plan)
            with self.assertRaisesRegex(ValueError, 'differs'):
                l.plan_input_binding('build', plan, snapshot, HELPER_ROOT, BRIDGE.HELPERS)

    def test_venv_or_helper_path_mismatch_cannot_borrow_fresh(self):
        snapshot, plans = self.bindings(); plans['venvs']['runtimes'][0]['path'] = '/fixture/other-python'
        with self.assertRaisesRegex(ValueError, 'differs'):
            l.plan_input_binding('venvs', plans['venvs'], snapshot, HELPER_ROOT, BRIDGE.HELPERS)
        snapshot['helper_paths']['runner.py'] = '/fixture/other-helper'
        with self.assertRaisesRegex(ValueError, 'differ'):
            l.plan_input_binding('runtime', plans['runtime'], snapshot, HELPER_ROOT, BRIDGE.HELPERS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper-root', required=True)
    parser.add_argument('--runtime-bridge', required=True)
    parser.add_argument('--native-validator', required=True)
    args, rest = parser.parse_known_args()
    BRIDGE = l.load_bridge(args.runtime_bridge); HELPER_ROOT = args.helper_root; NATIVE_PATH = args.native_validator
    unittest.main(argv=[__file__, *rest])
