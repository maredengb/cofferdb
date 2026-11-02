"""Tiny tool/version fixtures; no actual process, tool probe, or network."""
import argparse
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('tool_observer_controls', HERE / 'tool_observer.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
BRIDGE = HELPER = None


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='cofferdb-tool-observer-fixture-', dir='/private/tmp')
        self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.b = t.load_bridge(BRIDGE); self.h = self.b.load_helpers(HELPER); self.r = self.h['runner.py']
        self.checkout = self.root / 'source'; (self.checkout / 'tools').mkdir(parents=True)
        self.tools = self.root / 'selected'; self.tools.mkdir()
        self.selected = {}
        for name in t.KEYS:
            p = self.tools / name
            if name in ('prepare', 'builder'):
                p = self.checkout / 'tools' / ('prepare_source.py' if name == 'prepare' else 'build_python.py')
            p.write_bytes(('fixture-' + name).encode()); p.chmod(0o755)
            self.selected[name] = str(p)
        sandbox_fixture = self.selected['sandbox']; self.selected['sandbox'] = '/usr/bin/sandbox-exec'
        original_reader = self.b.read_file
        self.reader = lambda p, bound=self.b.MAX_FILE: original_reader(sandbox_fixture if p == '/usr/bin/sandbox-exec' else p, bound)
        original_access = os.access
        self.addCleanup(patch.stopall)
        patch.object(self.b, 'read_file', side_effect=self.reader).start()
        patch.object(os, 'access', side_effect=lambda p, mode: original_access(sandbox_fixture if p == '/usr/bin/sandbox-exec' else p, mode)).start()
        patch.object(subprocess, 'Popen', side_effect=AssertionError('NO PROCESS')).start()
        patch.object(socket.socket, 'connect', side_effect=AssertionError('NO NETWORK')).start()
        patch.object(t.platform, 'system', return_value='Darwin').start()
        patch.object(t.platform, 'machine', return_value='arm64').start()
        patch.object(t.platform, 'mac_ver', return_value=('15.7.1', (), '')).start()
        root = Path(HELPER)
        self.snapshot = {'schema_version': 1, 'kind': 'observed_mac15_technical_inputs_1',
                         'os': {'system': 'Darwin', 'machine': 'arm64', 'version': '15.7.1', 'ImageOS': 'macos15'},
                         'tools': dict.fromkeys(t.KEYS), 'runtimes': [],
                         'helper_paths': {name: str(root.parent / name if name == 'q_process_supervisor.py' else root / name)
                                          for name in self.b.HELPERS if name != 'dynamic_inputs/dynamic_plan.py'}}
        for minor in (12, 13, 14):
            p = self.root / ('python' + str(minor)); p.write_bytes(b'fixture-python'); p.chmod(0o755)
            stdlib = self.root / ('stdlib' + str(minor)); stdlib.mkdir()
            self.snapshot['runtimes'].append({'minor': minor, 'binary': {'path': str(p), 'sha256': t.sha(p.read_bytes())}, 'stdlib': str(stdlib)})
        self.parent = self.root / 'evidence'; self.parent.mkdir(mode=0o700); self.out = self.parent / 'new'
        self.now = 0; self.calls = []; self.failure = None; self.after_capture = None
        patch.object(self.b, 'load_helpers', return_value=self.h).start()
        patch.object(self.r, 'load_supervisor', return_value=types.SimpleNamespace(capture=self.capture)).start()

    def capture(self, request_file, request_sha, **kwargs):
        request = json.loads(Path(request_file).read_bytes()); self.calls.append((request, kwargs))
        name = next(k for k in t.QUERIES if self.selected[k] == request['argv'][0])
        output = Path(kwargs['output']); output.mkdir(mode=0o700)
        versions = {'cargo': b'cargo 1.97.1 (fixture)\n', 'rustc': b'rustc 1.97.1 (fixture)\n',
                    'maturin': b'maturin 1.10.2\n', 'compiler': b'Apple clang version 17.0.0 (fixture)\nTarget: arm64\n'}
        (output / 'stdout.raw').write_bytes(versions[name]); (output / 'stderr.raw').write_bytes(b'')
        (output / 'status.raw').write_bytes(b'{"event":"spawned","pid":12345}\n{"event":"completed","exit_code":0}\n')
        seals = {stream: {'path': str(output / (stream + '.raw')), 'size': len((output / (stream + '.raw')).read_bytes()),
                          'sha256': t.sha((output / (stream + '.raw')).read_bytes())} for stream in ('stdout', 'stderr', 'status')}
        if self.failure == name:
            raise ValueError('fixture failed capture')
        if self.after_capture:
            self.after_capture(name)
        return {'state': 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE', 'actual_command_exit_code': 0,
                'process_group_gone': True, 'raw_eof': {'stdout': True, 'stderr': True},
                'unread_tail_possible': False, 'failure_reason': None, 'argv': request['argv'],
                'request_sha256': request_sha, 'supervisor_sha256': self.r.SUPERVISOR_SHA, 'raw_streams': seals}

    def run_observer(self, **kw):
        return t.observe(self.b, HELPER, self.snapshot, self.selected, str(self.checkout), str(self.out), clock=lambda: self.now, **kw)

    def test_four_bound_versions_update_only_tools(self):
        result = self.run_observer(); self.assertEqual(len(self.calls), 4)
        updated = json.loads((self.out / 'SNAPSHOT.json').read_bytes())
        original = copy.deepcopy(updated); original['tools'] = dict.fromkeys(t.KEYS)
        self.assertEqual(original, self.snapshot); self.assertFalse(result['product_acceptance'])
        self.assertEqual(result['versions']['maturin'], 'maturin 1.10.2')
        self.assertTrue(all(x[1]['deadline'] == 30 for x in self.calls))
        self.assertTrue(all(x[1]['raw_limit'] == 65536 for x in self.calls))
        self.assertTrue(all(x[0]['argv'][1:] == ['--version'] for x in self.calls))

    def test_failed_second_query_preserves_prefix_no_retry(self):
        self.failure = 'rustc'
        with self.assertRaisesRegex(ValueError, 'fixture failed'): self.run_observer()
        self.assertEqual(len(self.calls), 2); self.assertTrue((self.out / 'cargo-capture/stdout.raw').exists())
        self.assertEqual(json.loads((self.out / 'HOLD.json').read_bytes())['completed_version_queries'], ['cargo'])
        self.assertFalse((self.out / 'RESULT.json').exists())

    def test_tool_drift_stops_next_capture(self):
        self.after_capture = lambda name: Path(self.selected['maturin']).write_bytes(b'changed') if name == 'cargo' else None
        with self.assertRaisesRegex(ValueError, 'bytes changed'): self.run_observer()
        self.assertEqual(len(self.calls), 1)

    def test_expired_after_last_capture_has_no_success(self):
        def expire(name):
            if name == 'compiler': self.now = 181
        self.after_capture = expire
        with self.assertRaisesRegex(ValueError, 'deadline'): self.run_observer()
        self.assertEqual(len(self.calls), 4); self.assertFalse((self.out / 'RESULT.json').exists())

    def test_bound_input_drift_stops_before_next_launch(self):
        def check():
            if self.calls: raise ValueError('Observation inputs changed')
        with self.assertRaisesRegex(ValueError, 'inputs changed'): self.run_observer(input_check=check)
        self.assertEqual(len(self.calls), 1)

    def test_wrong_host_creates_no_output(self):
        with patch.object(t.platform, 'mac_ver', return_value=('27.0', (), '')):
            with self.assertRaisesRegex(ValueError, 'MAC15'): self.run_observer()
        self.assertFalse(self.out.exists()); self.assertEqual(self.calls, [])

    def test_helper_path_wrong_stops_before_launch(self):
        self.snapshot['helper_paths']['runner.py'] += '-other'
        with self.assertRaisesRegex(ValueError, 'helper paths'): self.run_observer()
        self.assertEqual(self.calls, [])

    def test_rustup_resolved_name_rejected(self):
        self.selected['cargo'] = str(self.tools / 'rustup')
        with self.assertRaisesRegex(ValueError, 'shim target'): self.run_observer()
        self.assertEqual(self.calls, [])

    def test_wrong_capture_identity_rejected(self):
        original = self.capture
        def bad(*args, **kw):
            rec = original(*args, **kw); rec['argv'] = ['not-selected']; return rec
        with patch.object(self.r, 'load_supervisor', return_value=types.SimpleNamespace(capture=bad)):
            with self.assertRaisesRegex(ValueError, 'receipt differs'): self.run_observer()
        self.assertEqual(len(self.calls), 1)

    def test_output_overlap_rejected(self):
        self.out = self.checkout / 'new'
        with self.assertRaisesRegex(ValueError, 'overlaps'): self.run_observer()
        self.assertFalse(self.out.exists())

    def test_symlink_tool_rejected(self):
        p = Path(self.selected['rustc']); p.unlink(); p.symlink_to(self.selected['cargo'])
        with self.assertRaises(OSError): self.run_observer()
        self.assertEqual(self.calls, [])

    def test_tool_mode_rejected(self):
        Path(self.selected['cargo']).chmod(0o777)
        with self.assertRaisesRegex(ValueError, 'Protected'): self.run_observer()

    def test_exact_maturin_and_bounded_versions(self):
        for name, raw in [('maturin', b'maturin 1.10.3'), ('cargo', b'rustup 1.28.2'), ('compiler', b'gcc 14'), ('rustc', b'x' * 65537)]:
            with self.subTest(name=name), self.assertRaises(ValueError): t.version(name, raw)

    def test_stdout_replaced_after_original_seal_rejected(self):
        def replace(name):
            if name == 'cargo':
                (self.out / 'cargo-capture/stdout.raw').write_bytes(b'cargo 9.9.9 (fixture)\n')
        self.after_capture = replace
        with self.assertRaisesRegex(ValueError, 'original seal'): self.run_observer()
        self.assertEqual(len(self.calls), 1)

    def test_prior_stderr_changed_before_final_rejected(self):
        def replace(name):
            if name == 'compiler':
                (self.out / 'cargo-capture/stderr.raw').write_bytes(b'changed')
        self.after_capture = replace
        with self.assertRaisesRegex(ValueError, 'original seal'): self.run_observer()
        self.assertEqual(len(self.calls), 4); self.assertFalse((self.out / 'RESULT.json').exists())

    def test_missing_original_seal_rejected(self):
        original = self.capture
        def bad(*args, **kw):
            rec = original(*args, **kw); del rec['raw_streams']; return rec
        with patch.object(self.r, 'load_supervisor', return_value=types.SimpleNamespace(capture=bad)):
            with self.assertRaisesRegex(ValueError, 'seals required'): self.run_observer()
        self.assertEqual(len(self.calls), 1)

    def test_initial_input_time_consumes_original_budget(self):
        def slow(): self.now = 181
        with self.assertRaisesRegex(ValueError, 'deadline'): self.run_observer(input_check=slow)
        self.assertEqual(self.calls, []); self.assertFalse(self.out.exists())

    def test_late_result_write_preserves_candidate_and_holds(self):
        original = self.r.save
        def slow(p, value):
            original(p, value)
            if p.name == 'RESULT.json': self.now = 181
        with patch.object(self.r, 'save', side_effect=slow):
            with self.assertRaisesRegex(ValueError, 'after result writes'): self.run_observer()
        self.assertTrue((self.out / 'RESULT.json').exists())
        self.assertTrue((self.out / 'HOLD.json').exists())
        self.assertTrue(json.loads((self.out / 'RESULT.json').read_bytes())['completion_requires_successful_tool_exit'])

    def test_fixture_streams_match_fixed_supervisor_contract(self):
        source = self.b.read_file(str(Path(HELPER).parent / 'q_process_supervisor.py'), 131072)
        self.assertEqual(t.sha(source), self.r.SUPERVISOR_SHA)
        tree = ast.parse(source)
        fields = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(x, ast.Tuple) and
                    [getattr(y, 'id', None) for y in x.elts] == ['streams', 'counts'] for x in node.targets):
                fields.append(set(ast.literal_eval(node.value.elts[1])))
        self.assertEqual(fields, [{'stdout', 'stderr', 'status'}])
        result = self.run_observer()
        for receipt in result['records']:
            self.assertEqual(set(receipt['raw_streams']), fields[0])

    def test_status_stream_change_rejected(self):
        def replace(name):
            if name == 'cargo': (self.out / 'cargo-capture/status.raw').write_bytes(b'changed')
        self.after_capture = replace
        with self.assertRaisesRegex(ValueError, 'original seal'): self.run_observer()
        self.assertEqual(len(self.calls), 1)

    def test_default_verify_inert(self):
        with patch('sys.argv', ['tool_observer.py']), patch.object(t, 'load_bridge', side_effect=AssertionError('NO LOAD')), patch('sys.stdout', new=io.StringIO()) as output:
            self.assertEqual(t.main(), 0)
        self.assertIn('NOT_EXECUTED', output.getvalue()); self.assertEqual(self.calls, [])

    def installed_python_fixture(self, mode=0o775):
        """Map only a provider's logical policy path to owned fixture storage.

        Keep the real row/host_file/host_protection/host_limits and fd-based
        reads, digest, X_OK and NOFOLLOW checks. No installed file is changed.
        """
        fixture = Path(self.snapshot['runtimes'][0]['binary']['path'])
        fixture.chmod(mode)
        provider = '/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python'
        self.assertTrue(self.r.python_provider(provider))
        self.assertFalse(self.r.python_provider(str(fixture)))
        limits = self.r.host_limits
        def mapped_limits(value, role, private_root=None):
            selected = provider if str(value) == str(fixture) else value
            return limits(selected, role, private_root)
        patch.object(self.r, 'host_limits', side_effect=mapped_limits).start()
        return fixture

    def test_provider_775_uses_real_policy_at_initial_loop_and_final_readback(self):
        self.installed_python_fixture()
        with patch.object(self.r, 'row', wraps=self.r.row) as rows:
            result = self.run_observer()
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(rows.call_count, 6)  # Initial + four pre-launch reads + final.
        self.assertTrue(all(call.kwargs == {'executable': True, 'host_role': 'python'} for call in rows.call_args_list))
        self.assertFalse(result['product_acceptance'])

    def test_same_775_file_is_still_rejected_as_ordinary_input(self):
        fixture = self.installed_python_fixture()
        with self.assertRaisesRegex(ValueError, 'Protected bounded regular input'):
            self.r.row({'path': str(fixture), 'sha256': t.sha(fixture.read_bytes())}, executable=True)
        self.assertEqual(self.calls, [])

    def test_unknown_provider_775_gets_no_exception(self):
        fixture = Path(self.snapshot['runtimes'][0]['binary']['path']); fixture.chmod(0o775)
        self.assertFalse(self.r.python_provider(str(fixture)))
        with self.assertRaisesRegex(ValueError, 'Protected installed-file metadata'):
            self.run_observer()
        self.assertEqual(self.calls, []); self.assertFalse(self.out.exists())

    def test_unknown_provider_755_preserves_existing_fallback(self):
        fixture = Path(self.snapshot['runtimes'][0]['binary']['path']); fixture.chmod(0o755)
        self.assertFalse(self.r.python_provider(str(fixture)))
        self.assertFalse(self.run_observer()['product_acceptance'])
        self.assertEqual(len(self.calls), 4)

    def test_provider_777_is_rejected_before_capture(self):
        self.installed_python_fixture(0o777)
        with self.assertRaisesRegex(ValueError, 'Protected installed-file metadata'): self.run_observer()
        self.assertEqual(self.calls, []); self.assertFalse(self.out.exists())

    def test_provider_wrong_digest_is_rejected_before_capture(self):
        self.installed_python_fixture()
        self.snapshot['runtimes'][0]['binary']['sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'File bytes changed'): self.run_observer()
        self.assertEqual(self.calls, []); self.assertFalse(self.out.exists())

    def test_provider_without_execute_permission_is_rejected(self):
        self.installed_python_fixture(0o664)
        with self.assertRaisesRegex(ValueError, 'Installed executable unavailable'): self.run_observer()
        self.assertEqual(self.calls, []); self.assertFalse(self.out.exists())

    def test_provider_link_is_rejected_before_capture(self):
        fixture = self.installed_python_fixture()
        target = fixture.with_name('owned-target'); target.write_bytes(fixture.read_bytes()); target.chmod(0o775)
        fixture.unlink(); fixture.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'): self.run_observer()
        self.assertEqual(self.calls, []); self.assertFalse(self.out.exists())

    def test_provider_policy_is_rechecked_before_next_capture(self):
        fixture = self.installed_python_fixture()
        self.after_capture = lambda name: fixture.chmod(0o777) if name == 'cargo' else None
        with self.assertRaisesRegex(ValueError, 'Protected installed-file metadata'): self.run_observer()
        self.assertEqual(len(self.calls), 1); self.assertTrue((self.out / 'HOLD.json').exists())
        self.assertFalse((self.out / 'RESULT.json').exists())

    def test_provider_policy_is_rechecked_after_last_capture(self):
        fixture = self.installed_python_fixture()
        self.after_capture = lambda name: fixture.chmod(0o777) if name == 'compiler' else None
        with self.assertRaisesRegex(ValueError, 'Protected installed-file metadata'): self.run_observer()
        self.assertEqual(len(self.calls), 4); self.assertTrue((self.out / 'HOLD.json').exists())
        self.assertFalse((self.out / 'RESULT.json').exists())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--helper-root', required=True); p.add_argument('--runtime-bridge', required=True)
    a, rest = p.parse_known_args(); HELPER = a.helper_root; BRIDGE = a.runtime_bridge
    unittest.main(argv=['test_tool_observer.py', *rest])
