"""Local orchestration controls: no real subprocess, download, build or SDK."""
import argparse
import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import types
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('job_tests', HERE / 'job.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
PRODUCT = None
INSTALLER = None
WORKFLOW = None


def load(path):
    spec = importlib.util.spec_from_file_location('selected', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='cofferdb-mac-job-control-')
        self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(subprocess, 'Popen', side_effect=AssertionError('NO PROCESS')).start()
        patch.object(socket.socket, 'connect', side_effect=AssertionError('NO NETWORK')).start()
        self.checkout = self.root / 'public'; self.helper = self.checkout / 'tools/ci/macos15'
        shutil.copytree(PRODUCT / 'tools/ci/macos15', self.helper)
        if WORKFLOW is not None:
            shutil.copyfile(WORKFLOW / 'workflow_bridge.py', self.helper / 'workflow/workflow_bridge.py')
        shutil.copyfile(PRODUCT / 'tools/ci/q_process_supervisor.py', self.helper.parent / 'q_process_supervisor.py')
        for name in ('Cargo.lock', 'bindings/python/Cargo.lock'):
            target = self.checkout / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((PRODUCT / name).read_bytes())
        directory = self.checkout / 'tools/ci/installed_wheel'; directory.mkdir()
        shutil.copyfile(INSTALLER / 'install_driver.py', directory / 'install_driver.py')
        shutil.copytree(PRODUCT / 'tools/ci/installed_wheel/business', directory / 'business')
        shutil.copytree(PRODUCT / 'examples/data/real_corpus', self.checkout / 'examples/data/real_corpus')
        self.w = t.workflow_loader(str(self.helper)); self.modules = self.w.bootstrap(str(self.helper))
        self.b = self.modules['runtime']; self.r = self.modules['originals']['runner.py']
        patch.object(self.r, 'host_gate', return_value={}).start()
        patch.object(self.modules['native'], 'macho', return_value={'fixture_only': True}).start()
        self.rustup = self.root / 'rustup'; tools = self.rustup / 'toolchains/stable-aarch64-apple-darwin/bin'; tools.mkdir(parents=True)
        for name in ('cargo', 'rustc'):
            p = tools / name; p.write_bytes(b'NONEXECUTED SYNTHETIC TOOL'); p.chmod(0o500)
        self.value = {'checkout': str(self.checkout), 'source_commit': 'a' * 40, 'execution_reference': 'review-fixture-only',
                      'rustup_home': str(self.rustup), 'output': str(self.root / 'job')}
        self.env = {'GITHUB_REPOSITORY_ID': '1422246026', 'GITHUB_REPOSITORY': 'maredengb/coffered',
                    'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1'}
        patch.dict(os.environ, self.env).start()
        self.budget = {'started': time.monotonic(), 'deadline': time.monotonic() + 7200, 'check': lambda: None}
        self.events = []; self.fail = None; self.wheel = self.synthetic_wheel()
        pip = dict(t.PIP, bytes=3, sha256=t.sha(b'pip')); patch.object(t, 'PIP', pip).start()
        original_module = t.module
        def selected(raw, filename, name):
            m = original_module(raw, filename, name)
            if name == 'selected_input_preparation': m.MATURIN.update(bytes=len(self.wheel), sha256=t.sha(self.wheel))
            return m
        patch.object(t, 'module', side_effect=selected).start()
        developer = self.root / 'developer'; developer.mkdir()
        for name in ('git', 'clang'):
            p = developer / name; p.write_bytes(b'SYNTHETIC DEVELOPER TOOL'); p.chmod(0o500)
        patch.object(t, 'developer_tools', return_value={'developer_directory': 'synthetic', 'git': {'path': str(developer / 'git'), 'sha256': t.sha((developer / 'git').read_bytes())}, 'compiler': {'path': str(developer / 'clang'), 'sha256': t.sha((developer / 'clang').read_bytes())}}).start()
        self.original_capture = t.Captures
        owner = self
        class FakeCapture:
            def __init__(self, b, r, supervisor, python, root, budget, deadline, fresh):
                self.records = []; self.deadline = deadline; self.fresh = fresh
            def verify(self): pass
            def run(self, name, argv, cwd, environment, seconds):
                self.fresh(); owner.events.append(name)
                if owner.fail == name: raise ValueError('synthetic command failure')
                self.records.append({'synthetic': True, 'name': name})
                if name.endswith('-root'): return str(owner.checkout).encode()
                if name.endswith('-head'): return b'a' * 40
                if name.endswith('-tree'): return b'b' * 40
                if name.endswith('-origin'): return b'https://github.com/maredengb/coffered.git'
                if name.endswith('-status'): return b''
                if name == 'rust-version': return b'rustc 1.97.1 (fixture)\nhost: aarch64-apple-darwin\n'
                if name == 'pip-download': (owner.root / 'job/inputs' / t.PIP['name']).write_bytes(b'pip')
                if name == 'download': (owner.root / 'job/maturin.whl').write_bytes(owner.wheel)
                if name == 'vendor': owner.vendor_fixture(owner.root / 'job/vendor')
                if name.startswith('metadata-'):
                    owner.assertEqual(argv[:2], ['/usr/bin/sandbox-exec', '-f'])
                    owner.assertEqual(environment['CARGO_NET_OFFLINE'], 'true')
                    return b'{"packages":[],"resolve":{}}'
                return b''
        patch.object(t, 'Captures', FakeCapture).start()
        def pipeline(value, modules, root, budget, recheck, *, retain_check=None):
            recheck(); self.events.append('existing-pipeline')
            contract = json.loads(Path(value['driver_contract']['path']).read_bytes())
            for item, minor in zip(contract['runtime_drivers'], (12, 13, 14)):
                self.modules['runtime'].runtime_driver(item, {"{" + key + "}": 'fixture' for key in
                    ('wheel', 'wheel_sha256', 'native_sha256', 'venv', 'source_metadata', 'source_input_sha256',
                     'sdk_version', 'engine_version', 'smoke_work')}, [], minor)
                self.assertIn('--pip-wheel', item['arguments'])
                context_ref = next(x for x in item['inputs'] if x['path'].endswith('-context.json'))
                context = json.loads(Path(context_ref['path']).read_bytes())
                self.assertEqual(context['cell_id'], 'N%02d' % (minor - 11))
                self.assertEqual(context['budget_reference'], t.sha((self.root / 'job/INTENT.json').read_bytes()))
                self.assertEqual(context['preflight_evidence'], t.sha((self.root / 'job/inputs/PREPARED.json').read_bytes()))
            retain_check(lambda: owner.events.append('retained-verifier'))
            return {'synthetic_only': True, 'product_acceptance': False}
        patch.object(self.w, 'pipeline', side_effect=pipeline).start()

    def synthetic_wheel(self):
        files = {'maturin-1.10.2.data/scripts/maturin': b'SYNTHETIC ONLY', 'maturin-1.10.2.dist-info/LICENSE': b'license'}
        record = 'maturin-1.10.2.dist-info/RECORD'; out = io.StringIO(); writer = csv.writer(out)
        for name, raw in files.items():
            writer.writerow([name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode(), len(raw)])
        writer.writerow([record, '', '']); files[record] = out.getvalue().encode(); target = io.BytesIO()
        with zipfile.ZipFile(target, 'w') as z:
            for name, raw in files.items(): z.writestr(name, raw)
        return target.getvalue()

    def vendor_fixture(self, directory):
        directory.mkdir(); packages = {}
        for name in ('Cargo.lock', 'bindings/python/Cargo.lock'):
            for row in tomllib.loads((self.checkout / name).read_text())['package']:
                if 'source' in row: packages[row['name'], row['version']] = row['checksum']
        for (name, version), checksum in packages.items():
            root = directory / (name + '-' + version); root.mkdir()
            data = ('[package]\nname="%s"\nversion="%s"\nlicense="MIT"\n' % (name, version)).encode()
            (root / 'Cargo.toml').write_bytes(data)
            (root / '.cargo-checksum.json').write_bytes(t.canonical({'package': checksum, 'files': {'Cargo.toml': t.sha(data)}}))

    def execute(self): return t.execute(self.value, self.w, self.modules, str(self.helper), self.budget)

    def test_substituted_complete_chain_and_real_contract(self):
        self.execute()
        self.assertLess(self.events.index('initial-source-status'), self.events.index('pip-download'))
        self.assertLess(self.events.index('metadata-1'), self.events.index('existing-pipeline'))
        self.assertEqual(self.events.count('vendor'), 1)
        result = json.loads((self.root / 'job/RESULT.json').read_bytes())
        self.assertFalse(result['product_acceptance'])

    def test_compiler_role_survives_immutable_fresh_reads(self):
        compiler = str(self.root / 'developer/clang')
        original_read = self.b.read_file; original_host = self.b.host_file; roles = []
        def strict(path, *args, **kwargs):
            if str(path) == compiler:
                raise AssertionError('Compiler must not re-enter ordinary size gate')
            return original_read(path, *args, **kwargs)
        def installed(path, **kwargs):
            if str(path) == compiler: roles.append(kwargs['role'])
            return original_host(path, **kwargs)
        with patch.object(self.b, 'read_file', side_effect=strict), patch.object(self.b, 'host_file', side_effect=installed):
            self.execute()
        self.assertGreater(len(roles), 1)
        self.assertEqual(set(roles), {'compiler'})

    def test_initial_source_failure_prevents_download(self):
        self.fail = 'initial-source-status'
        with self.assertRaisesRegex(ValueError, 'synthetic command'): self.execute()
        self.assertNotIn('pip-download', self.events); self.assertNotIn('existing-pipeline', self.events)
        self.assertTrue((self.root / 'job/HOLD.json').exists())

    def test_vendor_failure_stops_and_preserves_prefix(self):
        self.fail = 'vendor'
        with self.assertRaises(ValueError): self.execute()
        self.assertEqual(self.events.count('vendor'), 1); self.assertNotIn('metadata-0', self.events)
        self.assertTrue((self.root / 'job/maturin/archive.whl').is_file())

    def test_metadata_failure_prevents_product_execution(self):
        self.fail = 'metadata-0'
        with self.assertRaises(ValueError): self.execute()
        self.assertNotIn('metadata-1', self.events); self.assertNotIn('existing-pipeline', self.events)

    def test_public_identity_rejects_retry_or_different_sha(self):
        for key, changed in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_SHA', 'b' * 40), ('GITHUB_REPOSITORY_ID', '1')]:
            env = {**self.env, key: changed}
            with self.assertRaises(ValueError): t.public_identity(self.value, env)

    def test_ancestor_cargo_config_stops_before_vendor(self):
        path = self.checkout / '.cargo'; path.mkdir(); (path / 'config.toml').write_text('bad')
        with self.assertRaisesRegex(ValueError, 'Cargo configuration'): self.execute()
        self.assertNotIn('pip-download', self.events)

    def test_missing_stable_toolchain_never_installs_rust(self):
        path = self.rustup / 'toolchains/stable-aarch64-apple-darwin'
        path.rename(path.with_name('different-aarch64-apple-darwin'))
        with self.assertRaisesRegex(ValueError, 'stable native Rust'): self.execute()
        self.assertNotIn('pip-download', self.events)

    def test_rust_version_requires_msrv_and_native_host(self):
        for raw in (b'rustc 1.79.0 (old)\nhost: aarch64-apple-darwin\n', b'rustc 1.97.1 (new)\nhost: x86_64-apple-darwin\n'):
            with self.assertRaises(ValueError): t.rust_version(raw)

    def test_final_result_drift_is_hold(self):
        original = self.r.save
        def save(path, value):
            original(path, value)
            if path == self.root / 'job/RESULT.json': path.write_bytes(path.read_bytes() + b' ')
        with patch.object(self.r, 'save', side_effect=save):
            with self.assertRaisesRegex(ValueError, 'Original job record'): self.execute()
        self.assertTrue((self.root / 'job/HOLD.json').exists())

    def test_outer_result_write_rechecks_vendor(self):
        original = self.r.save
        def save(path, value):
            original(path, value)
            if path == self.root / 'job/RESULT.json':
                selected = next((self.root / 'job/vendor').glob('*/Cargo.toml'))
                selected.write_bytes(selected.read_bytes() + b' ')
        with patch.object(self.r, 'save', side_effect=save):
            with self.assertRaisesRegex(ValueError, 'Final vendor drift'): self.execute()
        self.assertTrue((self.root / 'job/HOLD.json').exists())

    def test_outer_result_write_rechecks_maturin_archive(self):
        original = self.r.save
        def save(path, value):
            original(path, value)
            if path == self.root / 'job/RESULT.json':
                selected = self.root / 'job/maturin/archive.whl'
                selected.chmod(0o600)  # Deliberate owner mutation of this test fixture.
                selected.write_bytes(selected.read_bytes() + b' ')
        with patch.object(self.r, 'save', side_effect=save):
            with self.assertRaisesRegex(ValueError, 'Final maturin archive drift'): self.execute()
        self.assertTrue((self.root / 'job/HOLD.json').exists())

    def test_outer_result_write_invokes_original_retained_verifier(self):
        original = self.r.save
        pipeline = self.w.pipeline.side_effect
        artifact = self.root / 'late-wheel.whl'; artifact.write_bytes(b'original wheel')
        expected = t.sha(artifact.read_bytes())
        def composed(*args, retain_check):
            def check(): t.need(t.sha(artifact.read_bytes()) == expected, 'Original final wheel changed')
            result = pipeline(*args, retain_check=lambda _unused: retain_check(check))
            return result
        def save(path, value):
            original(path, value)
            if path == self.root / 'job/RESULT.json': artifact.write_bytes(b'replaced wheel')
        with patch.object(self.w, 'pipeline', side_effect=composed), patch.object(self.r, 'save', side_effect=save):
            with self.assertRaisesRegex(ValueError, 'Original final wheel changed'): self.execute()
        self.assertTrue((self.root / 'job/HOLD.json').exists())

    def test_capture_keeps_producer_sha_and_checks_deadline(self):
        evidence = self.root / 'captures'; evidence.mkdir()
        def fake_capture(path, expected, **kw):
            request = json.loads(Path(path).read_bytes()); target = Path(kw['output']); target.mkdir()
            rows = {}
            for name in ('stdout', 'stderr', 'status'):
                p = target / (name + '.raw'); p.write_bytes(b'fixture' if name == 'stdout' else b'')
                rows[name] = {'path': str(p), 'size': p.stat().st_size, 'sha256': t.sha(p.read_bytes())}
            return {'state': 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE', 'actual_command_exit_code': 0,
                    'process_group_gone': True, 'raw_eof': {'stdout': True, 'stderr': True}, 'unread_tail_possible': False,
                    'failure_reason': None, 'argv': request['argv'], 'request_sha256': expected,
                    'supervisor_sha256': self.r.SUPERVISOR_SHA, 'raw_streams': rows}
        cap = self.original_capture(self.b, self.r, types.SimpleNamespace(capture=fake_capture), sys.executable,
                                   evidence, self.budget, time.monotonic() + 60, lambda: None)
        self.assertEqual(cap.run('one', ['fixture'], self.root, {}, 10), b'fixture')
        (evidence / 'one/stdout.raw').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'Original preparation'): cap.verify()
        cap.deadline = time.monotonic() - 1
        with self.assertRaisesRegex(ValueError, 'deadline before command'): cap.run('two', ['fixture'], self.root, {}, 10)
        self.assertFalse((evidence / 'two.json').exists())

    def test_capture_rejects_stdout_changed_by_final_freshness_check(self):
        evidence = self.root / 'captures'; evidence.mkdir()
        def fake_capture(path, expected, **kw):
            request = json.loads(Path(path).read_bytes()); target = Path(kw['output']); target.mkdir()
            rows = {}
            for name in ('stdout', 'stderr', 'status'):
                p = target / (name + '.raw'); p.write_bytes(b'original' if name == 'stdout' else b'')
                rows[name] = {'path': str(p), 'size': p.stat().st_size, 'sha256': t.sha(p.read_bytes())}
            return {'state': 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE', 'actual_command_exit_code': 0,
                    'process_group_gone': True, 'raw_eof': {'stdout': True, 'stderr': True}, 'unread_tail_possible': False,
                    'failure_reason': None, 'argv': request['argv'], 'request_sha256': expected,
                    'supervisor_sha256': self.r.SUPERVISOR_SHA, 'raw_streams': rows}
        checks = 0
        def freshness():
            nonlocal checks
            checks += 1
            if checks == 2: (evidence / 'one/stdout.raw').write_bytes(b'replaced')
        cap = self.original_capture(self.b, self.r, types.SimpleNamespace(capture=fake_capture), sys.executable,
                                   evidence, self.budget, time.monotonic() + 60, freshness)
        with self.assertRaisesRegex(ValueError, 'Consumed preparation stdout changed'):
            cap.run('one', ['fixture'], self.root, {}, 10)
        self.assertEqual(checks, 2)

    def test_default_is_inert(self):
        with patch.object(t, 'workflow_loader', side_effect=AssertionError('NO READ')), patch.object(sys, 'argv', ['job.py']), patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(t.main(), 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--product', required=True); parser.add_argument('--installer', required=True)
    parser.add_argument('--workflow')
    args, rest = parser.parse_known_args(); PRODUCT = Path(args.product); INSTALLER = Path(args.installer)
    WORKFLOW = Path(args.workflow) if args.workflow else None
    unittest.main(argv=['test_job.py', *rest])
