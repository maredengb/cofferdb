"""Miniature filesystem controls for runtime plan binding; no actual SDK or host probe."""
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
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('selected_runtime_bridge', HERE / 'runtime_bridge.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
parser = argparse.ArgumentParser()
parser.add_argument('--helper-root', default=str(HERE))
ARGS, REST = parser.parse_known_args()


class Tests(unittest.TestCase):
    def setUp(self):
        self.process = patch.object(subprocess, 'Popen', side_effect=AssertionError('No process in controls'))
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('No network in controls'))
        self.process.start()
        self.network.start()
        self.addCleanup(self.process.stop)
        self.addCleanup(self.network.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='cofferdb-runtime-bridge-', dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.modules = m.load_helpers(ARGS.helper_root)
        self.r = self.modules['runner.py']
        self.d = self.modules['dynamic_inputs/dynamic_plan.py']
        # The unchanged runtime helper's system-file readback is a boundary,
        # not a fake Mac15 execution. All new bridge file checks remain real.
        self.actual = patch.object(self.r, 'actual_inputs', return_value=None)
        self.actual_mock = self.actual.start()
        self.addCleanup(self.actual.stop)
        self.docs = self.fixture()
        self.bundle = {'schema_version': 1, 'kind': 'mac15_runtime_bridge_inputs_1'}
        self.refresh()

    def write(self, path, raw, mode=0o644):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(mode)
        return {'path': str(path), 'sha256': m.sha(raw)}

    def row(self, name, executable=False):
        return self.write(self.root / name, ('fixture:' + name).encode(), 0o755 if executable else 0o644)

    def fixture(self):
        helper = Path(ARGS.helper_root)
        driver_row = self.row('immutable/driver.py')
        driver = {'runtime_drivers': [{'minor': minor, **driver_row,
                    'arguments': ['--wheel', '{wheel}', '--sha256', '{wheel_sha256}',
                                  '--source', '{source_metadata}', '--work', '{smoke_work}'],
                    'inputs': [], 'no_detach_reviewed': True} for minor in (12, 13, 14)],
                  'network_probe': {'address': '1.1.1.1', 'port': 443}, 'command_seconds': 120}
        config = {'schema_version': 1, 'kind': 'mac15_dynamic_input_config_1',
                  'source_repository': 'maredengb/coffered', 'source_commit': '1' * 40,
                  'helper_sha256': self.d.HELPERS, 'actions': self.d.ACTIONS,
                  'python_providers': self.d.PROVIDERS, 'selection_method': self.d.METHOD,
                  'native_closure_gate': 'independent-exact-closure-input-v1',
                  'driver_contract_sha256': self.d.sha(self.d.raw(driver)),
                  'candidate_manifest_sha256': '4ec41541f34f51b594ed8a97eb99c7a599f436541fe81d3eb552b41bb14cc9d7',
                  'native_closure_validator_sha256': '9' * 64}
        runtimes = []
        for minor in (12, 13, 14):
            stdlib = self.root / ('immutable/stdlib/cp3' + str(minor))
            stdlib.mkdir(parents=True)
            runtimes.append({'minor': minor, 'micro': 99, 'releaselevel': 'final',
                             'implementation': 'cpython', 'free_threaded': False,
                             'binary': self.row('immutable/base/cp3' + str(minor), True),
                             'stdlib': str(stdlib), 'stdlib_index_sha256': 'a' * 64, 'runtime_files': []})
        layout = {key: str(self.root / value) for key, value in (
            ('checkout', 'checkout'), ('cargo_home', 'cargo-home'), ('work_parent', 'work'),
            ('evidence_parent', 'evidence'), ('venv_parent', 'venvs'))}
        for value in layout.values():
            Path(value).mkdir(mode=0o700)
        tools = {key: self.row('immutable/tools/' + key, True) for key in ('cargo', 'rustc', 'maturin', 'compiler')}
        # The frozen schema rejects a temporary-path compiler. Its actual
        # system file read is stubbed; no compiler is invoked or probed here.
        tools['compiler'] = {'path': '/Library/Developer/CommandLineTools/usr/bin/clang', 'sha256': '8' * 64}
        tools['sandbox'] = {'path': '/usr/bin/sandbox-exec', 'sha256': '7' * 64}
        tools['prepare'] = self.row('checkout/tools/prepare_source.py')
        tools['builder'] = self.row('checkout/tools/build_python.py')
        snapshot = {'schema_version': 1, 'kind': 'observed_mac15_technical_inputs_1',
                    'config_sha256': self.d.sha(self.d.raw(config)), 'source_commit': config['source_commit'],
                    'os': {'system': 'Darwin', 'machine': 'arm64', 'version': '15.99'},
                    'runtimes': runtimes, 'tools': tools,
                    'helper_paths': {name: str(helper.parent / name if name == 'q_process_supervisor.py' else helper / name)
                                     for name in self.d.HELPERS},
                    'readonly_index_sha256': 'b' * 64,
                    'native_closure_status': 'DECLARED_FILES_INDEXED_REQUIRES_INDEPENDENT_CLOSURE',
                    'run_identity': {'run_id': 'fixture', 'attempt': '1', 'nonce': 'miniature'},
                    'observations_sha256': 'c' * 64}
        closure = {'kind': 'independent_exact_native_closure_1', 'snapshot_sha256': self.d.sha(self.d.raw(snapshot)),
                   'readonly_index_sha256': snapshot['readonly_index_sha256'], 'complete': True,
                   'evidence_reference': 'synthetic-local-control-only',
                   'validator_sha256': config['native_closure_validator_sha256']}
        plans = self.d.derive(config, snapshot, layout, driver, closure)['plans']

        def receipt(argv, cwd):
            return {'state': 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE',
                    'actual_command_exit_code': 0, 'process_group_gone': True,
                    'raw_eof': {'stdout': True, 'stderr': True}, 'unread_tail_possible': False,
                    'failure_reason': None, 'supervisor_sha256': self.d.HELPERS['q_process_supervisor.py'],
                    'request_sha256': m.sha(self.r.canonical({'schema_version': 1, 'argv': argv, 'cwd': cwd})), 'argv': argv}

        build = plans['build']
        empty = plans['empty_venvs']
        build_result = {'status': 'EXISTING_THREE_ABI_BUILD_CAPTURED_NOT_ACCEPTANCE',
                        'plan_sha256': m.sha(self.r.canonical(build)), 'product_acceptance': False,
                        'records': [receipt(argv, build['checkout']) for _, argv, _ in self.modules['workflow_build.py'].commands(build)]}
        venv_result = {'status': 'THREE_EMPTY_VENVS_PREPARED_NOT_INSTALLED',
                       'plan_sha256': m.sha(self.r.canonical(empty)), 'product_acceptance': False,
                       'records': [receipt(argv, empty['evidence']) for argv in self.modules['venv_prepare.py'].commands(empty)],
                       'bindings': []}
        metadata = {'sdk_version': '0.1.1', 'engine_version': '0.1.1',
                    'repository': 'https://github.com/maredengb/coffered',
                    'source_input_sha256': 'd' * 64,
                    'source': {'kind': 'git_checkout', 'commit': '1' * 40, 'tree': '2' * 40}}
        self.write(Path(build['work']) / 'source/bindings/python/python/cofferdb/_source.json', m.canonical(metadata))
        artifacts = []
        for selected, row in zip(runtimes, empty['runtimes']):
            minor = row['minor']
            venv = Path(row['venv'])
            venv.mkdir(mode=0o700)
            real = venv / 'bin/python'
            self.write(real, Path(row['path']).read_bytes(), 0o755)
            cfg = b'include-system-site-packages = false\nversion = 3.' + str(minor).encode() + b'.99\n'
            self.write(venv / 'pyvenv.cfg', cfg)
            (venv / ('lib/python3.' + str(minor)) / 'site-packages').mkdir(parents=True)
            venv_result['bindings'].append({'minor': minor, 'venv': str(venv), 'invocation': str(real),
                'real': str(real), 'base': row['path'], 'sha256': row['sha256'],
                'pyvenv_cfg_sha256': m.sha(cfg), 'pip_or_SDK_installed_by_this_command': False})
            native = b'FAKE_NATIVE_BYTES_NOT_A_MACHO_OR_SDK'
            filename = 'cofferdb-0.1.1-cp3%d-cp3%d-macosx_15_0_arm64.whl' % (minor, minor)
            target = Path(build['work']) / 'wheels' / ('cp3' + str(minor)) / filename
            target.parent.mkdir(parents=True)
            with zipfile.ZipFile(target, 'w') as wheel:
                wheel.writestr('cofferdb/_native.cpython-3%d-darwin.so' % minor, native)
                wheel.writestr('cofferdb/_source.json', m.canonical(metadata))
            artifacts.append({'wheel': filename, 'sha256': m.sha(target.read_bytes()),
                'native_sha256': m.sha(native), 'source_input_sha256': metadata['source_input_sha256'],
                'runtime': {'implementation': 'cpython', 'major': 3, 'minor': minor, 'version': '3.%d.99' % minor,
                            'system': 'Darwin', 'machine': 'arm64', 'pointer_bits': 64, 'free_threaded': False}})
        return {'config': config, 'snapshot': snapshot, 'layout': layout, 'driver_contract': driver,
                'native_closure': closure, 'build_result': build_result,
                'build_manifest': {'schema_version': 1, 'kind': 'native_wheel_build_not_platform_acceptance',
                    'source': metadata, 'builder_sha256': tools['builder']['sha256'], 'maturin': '1.10.2', 'artifacts': artifacts},
                'empty_venvs_result': venv_result}

    def refresh(self):
        destinations = {'build_result': self.root / 'evidence/build/RESULT.json',
                        'empty_venvs_result': self.root / 'evidence/empty-venvs/RESULT.json',
                        'build_manifest': self.root / 'work/build/wheels/build-manifest.json'}
        for name in m.DOCUMENTS:
            target = destinations.get(name, self.root / 'inputs' / (name + '.json'))
            # Existing builder output uses indented JSON, not canonical JSON.
            self.bundle[name] = self.write(target, (json.dumps(self.docs[name], indent=2) + '\n').encode())

    def run_bridge(self):
        return m.instantiate(self.bundle, self.modules)

    def test_three_abi_plans_bind_actual_files_without_execution(self):
        before = sorted(str(p) for p in self.root.rglob('*'))
        result = self.run_bridge()
        self.assertEqual(result['status'], 'RUNTIME_PLANS_DERIVED_NOT_EXECUTED')
        self.assertEqual(set(result['plans']), {'cp312', 'cp313', 'cp314'})
        self.assertFalse(result['sdk_executed'])
        self.assertFalse(result['product_acceptance'])
        self.assertTrue(result['fresh_native_closure_gate_required_before_execution'])
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))
        self.assertEqual(self.actual_mock.call_count, 3)
        for minor in (12, 13, 14):
            value = result['plans']['cp3' + str(minor)]
            self.assertEqual(value['plan_sha256'], m.sha(self.r.canonical(value['plan'])))
            self.assertEqual(value['plan']['dependencies'], [])

    def test_failed_or_boolean_exit_cannot_become_plan(self):
        for code in (1, -15, False, None):
            self.docs['build_result']['records'][0]['actual_command_exit_code'] = code
            self.refresh()
            with self.assertRaisesRegex(ValueError, 'HOLD_CAPTURE'):
                self.run_bridge()

    def test_missing_capture_or_cleanup_rejected(self):
        for key, value in [('process_group_gone', False), ('raw_eof', {'stdout': True, 'stderr': False}),
                           ('unread_tail_possible', True)]:
            original = copy.deepcopy(self.docs['build_result']['records'][0])
            self.docs['build_result']['records'][0][key] = value
            self.refresh()
            with self.assertRaises(ValueError):
                self.run_bridge()
            self.docs['build_result']['records'][0] = original

    def test_command_identity_or_plan_mismatch_rejected(self):
        self.docs['build_result']['records'][0]['argv'].append('--different')
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Capture command'):
            self.run_bridge()

    def test_partial_or_reordered_wheels_rejected(self):
        self.docs['build_manifest']['artifacts'].reverse()
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            self.run_bridge()

    def test_changed_actual_wheel_rejected(self):
        record = self.docs['build_manifest']['artifacts'][0]
        wheel = self.root / 'work/build/wheels/cp312' / record['wheel']
        wheel.write_bytes(wheel.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Actual wheel bytes'):
            self.run_bridge()

    def test_old_or_private_source_rejected(self):
        self.docs['build_manifest']['source']['source']['kind'] = 'unversioned_export'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Final clean public source'):
            self.run_bridge()

    def test_wrong_raw_result_path_rejected(self):
        self.bundle['build_result'] = self.write(self.root / 'other-result.json', m.canonical(self.docs['build_result']))
        with self.assertRaisesRegex(ValueError, 'Result path'):
            self.run_bridge()

    def test_nonempty_venv_rejected(self):
        self.write(self.root / 'venvs/cp312/lib/python3.12/site-packages/startup.pth', b'not empty')
        with self.assertRaisesRegex(ValueError, 'must still be empty'):
            self.run_bridge()

    def test_changed_copy_or_cfg_rejected(self):
        (self.root / 'venvs/cp312/bin/python').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Base/copy interpreter'):
            self.run_bridge()

    def test_changed_venv_config_rejected(self):
        (self.root / 'venvs/cp312/pyvenv.cfg').write_bytes(b'include-system-site-packages = true\n')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            self.run_bridge()

    def test_venv_copy_redirected_outside_root_rejected(self):
        selected = self.root / 'venvs/cp312/bin/python'
        selected.unlink()
        selected.symlink_to(self.docs['empty_venvs_result']['bindings'][0]['base'])
        with self.assertRaisesRegex(ValueError, 'resolves inside'):
            self.run_bridge()

    def test_source_metadata_change_rejected(self):
        selected = self.root / 'work/build/source/bindings/python/python/cofferdb/_source.json'
        selected.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'prepared source metadata'):
            self.run_bridge()

    def test_wrong_native_even_with_matching_archive_digest_rejected(self):
        record = self.docs['build_manifest']['artifacts'][0]
        selected = self.root / 'work/build/wheels/cp312' / record['wheel']
        with zipfile.ZipFile(selected, 'w') as wheel:
            wheel.writestr('cofferdb/_native.cpython-312-darwin.so', b'other native')
            wheel.writestr('cofferdb/_source.json', m.canonical(self.docs['build_manifest']['source']))
        record['sha256'] = m.sha(selected.read_bytes())
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Native wheel member'):
            self.run_bridge()

    def test_unbound_or_unknown_argument_placeholder_rejected(self):
        template = self.docs['driver_contract']['runtime_drivers'][0]
        incomplete = copy.deepcopy(template)
        incomplete['arguments'] = ['--wheel', '{wheel}']
        with self.assertRaisesRegex(ValueError, 'actual wheel'):
            m.runtime_driver(incomplete, {'{wheel}': '/some/wheel'}, [], 12)
        bad = copy.deepcopy(template)
        bad['arguments'].append('--prefix={venv}')
        with self.assertRaisesRegex(ValueError, 'embedded'):
            m.runtime_driver(bad, {'{wheel}': '/some/wheel', '{wheel_sha256}': 'a' * 64,
                                 '{source_metadata}': '/some/source', '{smoke_work}': '/some/work'}, [], 12)

    def test_conflicting_wheel_driver_input_rejected(self):
        template = copy.deepcopy(self.docs['driver_contract']['runtime_drivers'][0])
        template['inputs'] = [{'path': '/some/wheel', 'sha256': 'a' * 64}]
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            m.runtime_driver(template, {'{wheel}': '/some/wheel', '{wheel_sha256}': 'b' * 64,
                                 '{source_metadata}': '/some/source', '{smoke_work}': '/some/work'},
                             [{'path': '/some/wheel', 'sha256': 'b' * 64}], 12)

    def test_existing_runtime_output_rejected(self):
        (self.root / 'work/runtime-cp312').mkdir()
        with self.assertRaisesRegex(ValueError, 'output already exists'):
            self.run_bridge()

    def test_duplicate_json_and_boolean_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            m.decode(b'{"x":1,"x":2}')
        self.bundle['schema_version'] = True
        with self.assertRaisesRegex(ValueError, 'Exact bridge input'):
            self.run_bridge()

    def test_linked_and_world_writable_input_rejected(self):
        selected = self.root / 'inputs/config.json'
        selected.chmod(0o666)
        with self.assertRaisesRegex(ValueError, 'Protected bounded'):
            self.run_bridge()
        selected.chmod(0o644)
        alias = self.root / 'linked.json'
        alias.symlink_to(selected)
        with self.assertRaises(OSError):
            m.read_file(str(alias))

    def test_default_verify_is_inert(self):
        args = argparse.Namespace(action='verify')
        with patch.object(m.argparse.ArgumentParser, 'parse_args', return_value=args), \
                patch.object(m, 'read_file', side_effect=AssertionError('No file read')), \
                patch.object(m, 'load_helpers', side_effect=AssertionError('No helper load')), \
                patch('sys.stdout', new=io.StringIO()) as output:
            self.assertEqual(m.main(), 0)
            self.assertIn('RUNTIME_BRIDGE_NOT_EXECUTED', output.getvalue())


if __name__ == '__main__':
    unittest.main(argv=[__file__, *REST])
