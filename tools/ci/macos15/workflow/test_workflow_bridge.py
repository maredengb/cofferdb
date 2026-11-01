"""Miniature orchestration controls. Process/network/SDK operations are replaced."""
import argparse
import copy
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
spec = importlib.util.spec_from_file_location('workflow_fixture', HERE / 'workflow_bridge.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
ROOT = None


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cofferdb-workflow-control-', dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(subprocess, 'Popen', side_effect=AssertionError('REAL PROCESS FORBIDDEN')).start()
        patch.object(socket.socket, 'connect', side_effect=AssertionError('NETWORK FORBIDDEN')).start()
        # Runtime, native, original validators and derivation are real source modules.
        s = importlib.util.spec_from_file_location('fixture_bridge', Path(ROOT) / 'runtime_bridge/runtime_bridge.py')
        self.b = importlib.util.module_from_spec(s); s.loader.exec_module(self.b)
        self.originals = self.b.load_helpers(ROOT); self.r = self.originals['runner.py']
        self.d = self.originals['dynamic_inputs/dynamic_plan.py']
        self.helper = self.root / 'helper'; self.helper.mkdir()
        for relative in t.PINS:
            p = self.helper / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('synthetic-source')
        self.pins = {key: t.sha((self.helper / key).read_bytes()) for key in t.PINS}
        patch.object(t, 'PINS', self.pins).start()
        self.checkout = self.root / 'checkout'; (self.checkout / 'tools').mkdir(parents=True)
        self.cargo = self.root / 'cargo'; self.cargo.mkdir()
        self.tools = {}
        for name in ('cargo', 'rustc', 'maturin', 'sandbox', 'compiler', 'prepare', 'builder'):
            path = self.checkout / 'tools' / ('prepare_source.py' if name == 'prepare' else 'build_python.py') if name in ('prepare', 'builder') else self.root / 'tools' / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'synthetic tool'); self.tools[name] = str(path)
        self.driver = self.root / 'driver.py'; self.driver.write_text('synthetic; never executed')
        driver_row = {'path': str(self.driver), 'sha256': t.sha(self.driver.read_bytes())}
        self.contract = self.root / 'contract.json'
        self.contract.write_bytes(t.canonical({'runtime_drivers': [dict(driver_row, minor=n, arguments=['{wheel}', '{wheel_sha256}'], inputs=[], no_detach_reviewed=True) for n in (12, 13, 14)],
                                               'network_probe': {'address': '1.1.1.1', 'port': 443}, 'command_seconds': 240}))
        self.value = {'schema_version': 1, 'kind': 'mac15_complete_workflow_inputs_1', 'source_commit': '1' * 40,
                      'repository_id': 1422246026, 'checkout': str(self.checkout), 'cargo_home': str(self.cargo),
                      'selected_tools': self.tools, 'driver_contract': {'path': str(self.contract), 'sha256': t.sha(self.contract.read_bytes())},
                      'output': str(self.root / 'output')}
        self.events = []; self.actuals = []
        self.job = {'GITHUB_REPOSITORY_ID': '1422246026', 'GITHUB_REPOSITORY': 'maredengb/coffered',
                    'GITHUB_SHA': '1' * 40, 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1'}
        patch.dict(os.environ, self.job).start()
        patch.object(self.r, 'host_gate', return_value={'synthetic': True}).start()
        patch.object(self.d, 'verify_candidate_manifest').start()
        self.modules = {'runtime': self.b, 'originals': self.originals,
            'collector': types.SimpleNamespace(collect=self.collect), 'tools': types.SimpleNamespace(observe=self.observe),
            'native': types.SimpleNamespace(inspect=self.native), 'launch': types.SimpleNamespace(execute=self.execute)}
        patch.object(self.b, 'instantiate', side_effect=self.instantiate).start()

    def write(self, path, value):
        path = Path(path); path.write_bytes(t.canonical(value)); return path

    def collect(self, config, helpers, output):
        self.events.append('collect'); out = Path(output); out.mkdir(mode=0o700)
        self.assertEqual(helpers, str(self.helper.parent))
        runtimes = []
        for minor in (12, 13, 14):
            binary = self.root / ('python3.' + str(minor)); binary.write_bytes(b'synthetic')
            runtimes.append({'minor': minor, 'micro': 1, 'releaselevel': 'final', 'implementation': 'cpython',
                'free_threaded': False, 'binary': {'path': str(binary), 'sha256': t.sha(b'synthetic')},
                'stdlib': str(self.root / ('stdlib3.' + str(minor))), 'stdlib_index_sha256': t.sha(t.canonical([])), 'runtime_files': []})
        snapshot = {'schema_version': 1, 'kind': 'observed_mac15_technical_inputs_1', 'config_sha256': t.sha(t.canonical(config)),
            'source_commit': config['source_commit'], 'os': {'system': 'Darwin', 'machine': 'arm64', 'version': '15.1'},
            'runtimes': runtimes, 'tools': dict.fromkeys(self.tools),
            'helper_paths': {name: str(self.helper.parent / name if name == 'q_process_supervisor.py' else self.helper / name) for name in self.d.HELPERS},
            'readonly_index_sha256': t.sha(t.canonical([])), 'native_closure_status': 'DECLARED_FILES_INDEXED_REQUIRES_INDEPENDENT_CLOSURE',
            'run_identity': {'run_id': '123', 'attempt': '1', 'nonce': 'collect'}, 'observations_sha256': t.sha(t.canonical([]))}
        self.write(out / 'SNAPSHOT.json', snapshot); self.write(out / 'READONLY-INDEX.json', [])
        self.write(out / 'OBSERVATIONS.json', [])
        return {'status': 'INPUT_MISSING', 'snapshot_sha256': t.sha(t.canonical(snapshot))}

    def observe(self, b, root, snapshot, selected, checkout, output, input_check):
        self.events.append('tools'); input_check(); snapshot = copy.deepcopy(snapshot)
        snapshot['tools'] = {k: {'path': p, 'sha256': t.sha(Path(p).read_bytes())} for k, p in selected.items()}
        out = Path(output); out.mkdir(mode=0o700); self.write(out / 'SNAPSHOT.json', snapshot)
        records = []
        for number in range(4):
            rows = {}
            for stream in ('stdout', 'stderr', 'status'):
                path = out / ('tool-%d-%s.raw' % (number, stream)); path.write_bytes(b'synthetic version output')
                rows[stream] = {'path': str(path), 'sha256': t.sha(path.read_bytes()), 'size': path.stat().st_size}
            records.append({'state': 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE', 'actual_command_exit_code': 0,
                'process_group_gone': True, 'raw_eof': {'stdout': True, 'stderr': True}, 'unread_tail_possible': False,
                'failure_reason': None, 'raw_streams': rows})
        report = {'status': 'EXISTING_TOOL_VERSIONS_OBSERVED_NOT_ACCEPTANCE',
                  'snapshot_sha256': t.sha(t.canonical(snapshot)), 'records': records}
        self.write(out / 'RESULT.json', report)
        return report

    def native(self, config, snapshot, index, validator, reference, observer, plan):
        self.events.append('native')
        report = {'synthetic_graph': True}
        return {'snapshot': snapshot, 'report': report, 'report_sha256': t.sha(t.canonical(report)),
                'closure': {'kind': 'independent_exact_native_closure_1', 'validator_sha256': validator,
                    'snapshot_sha256': t.sha(t.canonical(snapshot)), 'readonly_index_sha256': t.sha(t.canonical(index)),
                    'complete': True, 'evidence_reference': reference}}

    def execute(self, stage, path, digest, approved, fresh, fresh_sha, gate, root, b, native):
        self.events.append(stage); self.assertEqual(digest, approved)
        self.assertEqual(t.sha(Path(path).read_bytes()), digest); self.assertEqual(t.sha(Path(fresh).read_bytes()), fresh_sha)
        plan = json.loads(Path(path).read_bytes()); out = Path(plan['evidence']); out.mkdir(mode=0o700)
        records = []
        for i in range(3 if stage == 'venvs' else 2):
            rows = {}
            for stream in ('stdout', 'stderr', 'status'):
                p = out / ('%d-%s.raw' % (i, stream)); p.write_bytes(b'synthetic raw')
                rows[stream] = {'path': str(p), 'size': p.stat().st_size, 'sha256': t.sha(p.read_bytes())}
            records.append({'state': 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE', 'actual_command_exit_code': 0,
                            'process_group_gone': True, 'raw_eof': {'stdout': True, 'stderr': True},
                            'unread_tail_possible': False, 'failure_reason': None, 'raw_streams': rows})
        if stage == 'build':
            p = Path(plan['work']) / 'wheels'; p.mkdir(parents=True)
            manifest = self.write(p / 'build-manifest.json', {'synthetic_manifest': True})
            row = records[1]['raw_streams']['stdout']; Path(row['path']).write_bytes(manifest.read_bytes())
            row.update({'sha256': t.sha(manifest.read_bytes()), 'size': manifest.stat().st_size})
        result = {'plan_sha256': digest, 'records': records}
        self.write(out / 'RESULT.json', result)
        return {'status': 'ORIGINAL_STEP_CAPTURED_WITH_FRESH_INPUTS_NOT_ACCEPTANCE', 'original_result': result}

    def instantiate(self, bundle, modules):
        self.events.append('runtime-plans'); self.assertIs(modules, self.originals)
        self.assertEqual(set(bundle), {'schema_version', 'kind', *self.b.DOCUMENTS})
        for name in self.b.DOCUMENTS:
            self.assertEqual(t.sha(Path(bundle[name]['path']).read_bytes()), bundle[name]['sha256'])
        layout = json.loads(Path(bundle['layout']['path']).read_bytes()); plans = {}; artifacts = []
        for minor in (12, 13, 14):
            name = 'cp3' + str(minor); plan = {'evidence': layout['evidence_parent'] + '/runtime-' + name}
            plans[name] = {'plan': plan, 'plan_sha256': t.sha(self.r.canonical(plan))}
            wheel = Path(layout['work_parent']) / (name + '.whl'); wheel.write_bytes(b'not a wheel; fixture only')
            artifacts.append({'minor': minor, 'wheel': str(wheel), 'wheel_sha256': t.sha(wheel.read_bytes())})
        return {'status': 'RUNTIME_PLANS_DERIVED_NOT_EXECUTED', 'plans': plans, 'observed': artifacts,
                'source_commit': '1' * 40, 'source_tree': '2' * 40}

    def run_pipeline(self, recheck=lambda: None, retain_check=None):
        return t.pipeline(self.value, self.modules, str(self.helper),
            {'started': 0, 'deadline': 7200, 'check': lambda: None}, recheck, retain_check=retain_check)

    def test_whole_chain_uses_actual_next_outputs_and_original_initial_plan_validators(self):
        result = self.run_pipeline()
        self.assertEqual(self.events, ['collect', 'tools', 'native', 'build', 'venvs', 'runtime-plans', 'runtime', 'runtime', 'runtime'])
        self.assertEqual(result['captured_commands'], 11); self.assertFalse(result['product_acceptance'])
        self.assertTrue(result['completion_requires_successful_job_exit'])

    def test_failed_build_preserves_prefix_and_never_creates_venvs(self):
        def failure(*args): self.events.append('failed-build'); raise ValueError('synthetic build failure')
        self.modules['launch'].execute = failure
        with self.assertRaisesRegex(ValueError, 'build failure'): self.run_pipeline()
        self.assertEqual(self.events[-1], 'failed-build'); self.assertNotIn('venvs', self.events)
        hold = json.loads((self.root / 'output/HOLD.json').read_bytes())
        self.assertIn('build', hold['entered_steps']); self.assertNotIn('build', hold['completed_steps'])

    def test_no_second_runtime_after_failed_first_cell(self):
        original = self.execute
        def execute(stage, *args):
            if stage == 'runtime': self.events.append('failed-runtime'); raise ValueError('synthetic runtime failure')
            return original(stage, *args)
        self.modules['launch'].execute = execute
        with self.assertRaisesRegex(ValueError, 'runtime failure'): self.run_pipeline()
        self.assertEqual(self.events.count('failed-runtime'), 1)

    def test_wrong_github_source_stops_before_creating_output(self):
        with patch.dict(os.environ, {'GITHUB_SHA': '9' * 40}), self.assertRaisesRegex(ValueError, 'job identity'):
            self.run_pipeline()
        self.assertFalse((self.root / 'output').exists())

    def test_changed_component_stops_before_collection(self):
        (self.helper / 'runtime_bridge/runtime_bridge.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Component changed'): self.run_pipeline()
        self.assertEqual(self.events, [])

    def test_bad_driver_placeholders_stop_before_collect(self):
        data = json.loads(self.contract.read_bytes()); data['runtime_drivers'][0]['arguments'] = ['{wrong}']
        self.contract.write_bytes(t.canonical(data)); self.value['driver_contract']['sha256'] = t.sha(self.contract.read_bytes())
        with self.assertRaisesRegex(ValueError, 'placeholder'): self.run_pipeline()
        self.assertEqual(self.events, [])

    def test_original_raw_stream_tampering_is_not_resealed(self):
        original = self.execute; last_raw = []
        def execute(stage, *args):
            result = original(stage, *args)
            if stage == 'build': last_raw.append(result['original_result']['records'][0]['raw_streams']['stdout']['path'])
            if stage == 'runtime': Path(last_raw[0]).write_bytes(b'changed')
            return result
        self.modules['launch'].execute = execute
        with self.assertRaisesRegex(ValueError, 'captured output changed'): self.run_pipeline()

    def test_written_layout_retains_expected_bytes_before_first_readback(self):
        save = self.r.save
        def corrupt(path, data):
            save(path, data)
            if path.name == 'layout.json': path.write_bytes(path.read_bytes()[:-1] + b' ')
        with patch.object(self.r, 'save', side_effect=corrupt), self.assertRaisesRegex(ValueError, 'Original output changed'):
            self.run_pipeline()
        self.assertEqual(self.events, [])

    def test_collector_original_snapshot_sha_is_required(self):
        def corrupt(*args):
            result = self.collect(*args); path = Path(args[2]) / 'SNAPSHOT.json'
            path.write_bytes(path.read_bytes()[:-1] + b' ')
            return result
        self.modules['collector'].collect = corrupt
        with self.assertRaisesRegex(ValueError, 'Original output changed'): self.run_pipeline()
        self.assertEqual(self.events, ['collect'])

    def test_tool_original_snapshot_sha_is_required(self):
        def corrupt(*args, **kwargs):
            result = self.observe(*args, **kwargs); path = Path(args[5]) / 'SNAPSHOT.json'
            path.write_bytes(path.read_bytes()[:-1] + b' ')
            return result
        self.modules['tools'].observe = corrupt
        with self.assertRaisesRegex(ValueError, 'Original output changed'): self.run_pipeline()
        self.assertEqual(self.events, ['collect', 'tools'])

    def test_original_result_not_resealed_at_first_readback(self):
        def corrupt(stage, *args):
            result = self.execute(stage, *args)
            plan = json.loads(Path(args[0]).read_bytes()); path = Path(plan['evidence']) / 'RESULT.json'
            path.write_bytes(path.read_bytes()[:-1] + b' ')
            return result
        self.modules['launch'].execute = corrupt
        with self.assertRaisesRegex(ValueError, 'Original output changed'): self.run_pipeline()
        self.assertNotIn('venvs', self.events)

    def test_build_manifest_bound_to_original_builder_stdout(self):
        def corrupt(stage, *args):
            result = self.execute(stage, *args)
            if stage == 'build':
                plan = json.loads(Path(args[0]).read_bytes()); path = Path(plan['work']) / 'wheels/build-manifest.json'
                path.write_bytes(path.read_bytes()[:-1] + b' ')
            return result
        self.modules['launch'].execute = corrupt
        with self.assertRaisesRegex(ValueError, 'Original output changed'): self.run_pipeline()
        self.assertNotIn('runtime-plans', self.events)

    def test_terminal_result_write_drift_never_returns_completion(self):
        save = self.r.save
        for target in ('raw', 'wheel', 'layout', 'manifest', 'final', 'tool'):
            with self.subTest(target=target):
                self.value['output'] = str(self.root / ('output-' + target)); output = Path(self.value['output'])
                def corrupt(path, data):
                    save(path, data)
                    if path == output / 'RESULT.json':
                        if target == 'raw':
                            result = json.loads((output / 'evidence/build/RESULT.json').read_bytes())
                            selected = Path(result['records'][0]['raw_streams']['stdout']['path'])
                        elif target == 'wheel': selected = Path(data['actual_artifacts'][0]['wheel'])
                        elif target == 'layout': selected = output / 'data/layout.json'
                        elif target == 'manifest': selected = output / 'work/build/wheels/build-manifest.json'
                        elif target == 'tool': selected = output / 'tools/tool-0-stdout.raw'
                        else: selected = path
                        selected.write_bytes(selected.read_bytes()[:-1] + b' ')
                with patch.object(self.r, 'save', side_effect=corrupt), self.assertRaises(ValueError): self.run_pipeline()
                self.assertTrue((output / 'HOLD.json').is_file())

    def test_retained_check_uses_original_seals_after_pipeline_return(self):
        for target in ('raw', 'wheel', 'layout', 'manifest', 'final', 'tool'):
            with self.subTest(target=target):
                checks = []; self.value['output'] = str(self.root / ('retained-' + target))
                output = Path(self.value['output'])
                result = self.run_pipeline(retain_check=checks.append)
                self.assertEqual(len(checks), 1); checks[0]()
                events = list(self.events)
                if target == 'raw':
                    receipt = json.loads((output / 'evidence/build/RESULT.json').read_bytes())
                    selected = Path(receipt['records'][0]['raw_streams']['stdout']['path'])
                elif target == 'wheel': selected = Path(result['actual_artifacts'][0]['wheel'])
                elif target == 'layout': selected = output / 'data/layout.json'
                elif target == 'manifest': selected = output / 'work/build/wheels/build-manifest.json'
                elif target == 'tool': selected = output / 'tools/tool-0-stdout.raw'
                else: selected = output / 'RESULT.json'
                selected.write_bytes(selected.read_bytes()[:-1] + b' ')
                with self.assertRaises(ValueError): checks[0]()
                self.assertEqual(self.events, events)

    def test_inherited_blocked_alarm_stops_before_body_without_signal_mutation(self):
        with patch.object(t.signal, 'pthread_sigmask', return_value={t.signal.SIGALRM}) as mask, \
             patch.object(t.signal, 'signal') as change, patch.object(t.signal, 'setitimer') as arm:
            with self.assertRaisesRegex(ValueError, 'must not be blocked'):
                with t.whole_budget(): self.fail('body must not run')
            mask.assert_called_once_with(t.signal.SIG_BLOCK, [])
            change.assert_not_called(); arm.assert_not_called()

    def test_whole_budget_latches_expiry_and_restores_handlers(self):
        now = [0.0]; handlers = {}; calls = []
        with patch.object(t.signal, 'getitimer', return_value=(0.0, 0.0)), \
             patch.object(t.signal, 'getsignal', return_value='previous'), \
             patch.object(t.signal, 'signal', side_effect=lambda n, h: handlers.update({n: h})), \
             patch.object(t.signal, 'setitimer', side_effect=lambda *args: calls.append(args)):
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                with t.whole_budget(clock=lambda: now[0]) as budget:
                    with self.assertRaises(t.WholeDeadline): handlers[t.signal.SIGALRM](0, None)
                    budget['check']()
        self.assertEqual(handlers[t.signal.SIGALRM], 'previous'); self.assertEqual(calls[-1][1], 0)

    def test_existing_alarm_is_never_replaced(self):
        with patch.object(t.signal, 'getitimer', return_value=(2.0, 0.0)), patch.object(t.signal, 'signal') as change:
            with self.assertRaisesRegex(ValueError, 'must not be replaced'):
                with t.whole_budget(): pass
            change.assert_not_called()

    def test_default_has_no_inputs_host_alarm_or_execution(self):
        with patch('sys.argv', ['workflow_bridge.py']), patch.object(t, 'bootstrap', side_effect=AssertionError('NO IO')), \
             patch.object(t, 'whole_budget', side_effect=AssertionError('NO TIMER')), patch('sys.stdout', new=io.StringIO()) as output:
            self.assertEqual(t.main(), 0)
        self.assertIn('NOT_EXECUTED', output.getvalue())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--helper-root', required=True)
    args, rest = parser.parse_known_args(); ROOT = args.helper_root
    unittest.main(argv=['test_workflow_bridge.py', *rest])
