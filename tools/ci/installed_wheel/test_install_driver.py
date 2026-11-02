"""Tiny wheel/install/driver contract controls; every subprocess and SDK load is substituted."""
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
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('install_driver_tests', HERE / 'install_driver.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
BRIDGE = None


def digest_field(data):
    return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()


def record(files, target):
    output = io.StringIO(); writer = csv.writer(output, lineterminator='\n')
    for name, raw in sorted(files.items()):
        writer.writerow([name, '', ''] if name == target else [name, digest_field(raw), str(len(raw))])
    return output.getvalue().encode()


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='cofferdb-wheel-driver-fixture-', dir='/private/tmp')
        self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(subprocess, 'Popen', side_effect=AssertionError('NO PROCESS')).start()
        patch.object(socket.socket, 'connect', side_effect=AssertionError('NO NETWORK')).start()
        self.b = t.load_bridge(BRIDGE)
        self.inputs = self.root / 'inputs'; self.inputs.mkdir()
        self.venv = self.root / 'venv'; self.site = self.venv / 'lib/python3.12/site-packages'; self.site.mkdir(parents=True)
        self.parent = self.root / 'work'; self.parent.mkdir(mode=0o700)
        self.dist = 'cofferdb-0.1.1.dist-info/'; self.native = 'cofferdb/_native.cpython-312-darwin.so'
        self.tag = 'cp312-cp312-macosx_15_0_arm64'
        self.source = {'sdk_version': '0.1.1', 'engine_version': '0.1.1', 'source_input_sha256': 'a' * 64}
        self.source_raw = t.canonical(self.source)
        self.files = {'cofferdb/__init__.py': b'# synthetic package; never imported\n',
                      'cofferdb/_source.json': self.source_raw, self.native: b'SYNTHETIC NATIVE; NEVER EXECUTED',
                      self.dist + 'METADATA': b'Metadata-Version: 2.4\nName: cofferdb\nVersion: 0.1.1\n\n',
                      self.dist + 'WHEEL': ('Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: ' + self.tag + '\n\n').encode(),
                      self.dist + 'RECORD': b''}
        self.files[self.dist + 'RECORD'] = record(self.files, self.dist + 'RECORD')
        self.wheel = self.inputs / ('cofferdb-0.1.1-' + self.tag + '.whl'); self.write_wheel()
        self.source_path = self.inputs / 'source.json'; self.source_path.write_bytes(self.source_raw)
        self.driver = self.inputs / 'driver.py'
        self.driver.write_text('def main():\n    return 0\n')
        self.context = self.inputs / 'context.json'
        self.context.write_bytes(t.canonical({**{key: 'synthetic fixture' for key in
            ('attempt_id', 'cell_id', 'authorization_reference', 'preflight_evidence', 'budget_reference')}, 'files_sha256': {'tiny.json': 'b' * 64}}))
        self.fixtures = self.inputs / 'fixtures'; self.fixtures.mkdir()
        self.args = types.SimpleNamespace(runtime_bridge=BRIDGE, wheel=str(self.wheel), wheel_sha256=t.sha(self.wheel.read_bytes()),
            native_sha256=t.sha(self.files[self.native]), source_metadata=str(self.source_path), source_input_sha256='a' * 64,
            sdk_version='0.1.1', engine_version='0.1.1', venv=str(self.venv), driver=str(self.driver), driver_sha256=t.sha(self.driver.read_bytes()),
            context=str(self.context), context_sha256=t.sha(self.context.read_bytes()), fixtures=str(self.fixtures), work=str(self.parent / 'smoke'))
        self.pip_wheel = self.inputs / t.PIP_NAME
        self.pip_files = {'pip/__init__.py': b'__version__ = \"25.3\"\n', 'pip-25.3.dist-info/METADATA': b'Name: pip\nVersion: 25.3\n', 'pip-25.3.dist-info/RECORD': b''}
        with zipfile.ZipFile(self.pip_wheel, 'w') as archive:
            for name, raw in self.pip_files.items(): archive.writestr(name, raw)
        patch.object(t, 'PIP_SHA', t.sha(self.pip_wheel.read_bytes())).start()
        patch.object(t, 'PIP_BYTES', self.pip_wheel.stat().st_size).start()
        self.args.pip_wheel = str(self.pip_wheel)
        self.calls = []; self.loads = []

    def write_wheel(self):
        with zipfile.ZipFile(self.wheel, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in self.files.items(): archive.writestr(name, raw)

    def payload(self):
        return t.wheel_payload(self.wheel.read_bytes(), self.args.wheel_sha256, self.args.native_sha256,
            self.source_raw, self.args.source_input_sha256, '0.1.1', '0.1.1', self.tag)

    def install_fixture(self):
        actual = copy.deepcopy(self.files)
        actual[self.dist + 'INSTALLER'] = b'pip\n'; actual[self.dist + 'REQUESTED'] = b''
        actual[self.dist + 'direct_url.json'] = t.canonical({'url': self.wheel.as_uri(), 'archive_info': {'hashes': {'sha256': self.args.wheel_sha256}}})
        actual[self.dist + 'RECORD'] = record(actual, self.dist + 'RECORD')
        for name, raw in actual.items():
            p = self.site / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
        return actual

    def identity(self):
        return t.installed_identity(self.b, self.site, self.payload(), str(self.wheel), self.args.wheel_sha256)

    def fake_process(self, argv, **kw):
        self.calls.append((argv, kw))
        if len(self.calls) == 2:
            for name, raw in self.pip_files.items():
                path = self.site / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        if len(self.calls) == 3:
            kw['stdout'].write(t.canonical({'version': '25.3', 'path': str(self.site / 'pip/__init__.py')}))
        if len(self.calls) == 4: self.install_fixture()
        return types.SimpleNamespace(returncode=0)

    def fake_import(self, name):
        self.loads.append(name)
        if name == 'cofferdb':
            return types.SimpleNamespace(__file__=str(self.site / 'cofferdb/__init__.py'), __version__='0.1.1', engine_version='0.1.1')
        if name == 'cofferdb._native': return types.SimpleNamespace(__file__=str(self.site / self.native))
        raise AssertionError('unexpected import')

    def run_execute(self, process=None, load=None):
        original_find = t.PathFinder.find_spec
        def find(name, path=None, target=None):
            if name == 'cofferdb': return types.SimpleNamespace(origin=str(self.site / 'cofferdb/__init__.py'))
            return original_find(name, path, target)
        with patch.object(t, 'runtime', return_value=(self.site, self.tag)), \
             patch.object(t.subprocess, 'run', side_effect=process or self.fake_process), \
             patch.object(t.PathFinder, 'find_spec', side_effect=find), \
             patch.object(t.importlib, 'import_module', side_effect=load or self.fake_import):
            return t.execute(self.args)

    def test_wheel_and_installed_complete_byte_identity(self):
        self.install_fixture(); result = self.identity()
        for row in result['identity'].values(): self.assertEqual(row['expected_sha256'], row['observed_sha256'])
        self.assertEqual(result['installed_files_sha256'][self.native], self.args.native_sha256)
        self.assertNotEqual(result['installed_record_sha256'], t.sha(self.files[self.dist + 'RECORD']))

    def test_whole_chain_substituted_install_and_import(self):
        result = self.run_execute()
        self.assertEqual(result['status'], 'EXACT_OFFLINE_INSTALL_AND_DRIVER_COMPLETED')
        self.assertFalse(result['product_acceptance']); self.assertEqual(self.loads, ['cofferdb', 'cofferdb._native'])
        self.assertEqual([row[0] for row in self.calls], t.install_commands(sys.executable, str(self.wheel), str(self.pip_wheel)))
        self.assertTrue(all('start_new_session' not in row[1] and 'shell' not in row[1] for row in self.calls))
        self.assertTrue(all(0 < row[1]['timeout'] <= 240 for row in self.calls))
        contract = json.loads((self.parent / 'smoke-installation/driver-contract.json').read_bytes())
        self.assertEqual(contract['identity'], result['identity']['identity'])

    def test_installer_failure_stops_no_sdk_load(self):
        def failed(argv, **kw): self.calls.append((argv, kw)); return types.SimpleNamespace(returncode=1)
        with self.assertRaisesRegex(ValueError, 'installer failed'): self.run_execute(process=failed)
        self.assertEqual(len(self.calls), 1); self.assertEqual(self.loads, [])
        self.assertTrue((self.parent / 'smoke-installation/HOLD.json').exists())

    def test_installed_native_mismatch_stops_before_sdk(self):
        def wrong(argv, **kw):
            value = self.fake_process(argv, **kw)
            if len(self.calls) == 4: (self.site / self.native).write_bytes(b'x')
            return value
        with self.assertRaisesRegex(ValueError, 'Installed bytes'): self.run_execute(process=wrong)
        self.assertEqual(self.loads, [])

    def test_wrong_loaded_native_path_rejected(self):
        def wrong(name):
            module = self.fake_import(name)
            if name.endswith('_native'): module.__file__ = '/synthetic-other.so'
            return module
        with self.assertRaisesRegex(ValueError, 'Loaded SDK'): self.run_execute(load=wrong)

    def test_record_payload_corruption_rejected(self):
        self.install_fixture(); (self.site / (self.dist + 'RECORD')).write_bytes(b'wrong,,\n')
        with self.assertRaisesRegex(ValueError, 'RECORD complete set'): self.identity()

    def test_unexpected_pth_rejected(self):
        self.install_fixture(); (self.site / 'cofferdb/extra.pth').write_bytes(b'import fake')
        with self.assertRaisesRegex(ValueError, 'member set'): self.identity()

    def test_wrong_installed_url_rejected(self):
        self.install_fixture(); p = self.site / (self.dist + 'direct_url.json')
        p.write_bytes(t.canonical({'url': 'https://example.invalid/other.whl', 'archive_info': {'hashes': {'sha256': self.args.wheel_sha256}}}))
        with self.assertRaisesRegex(ValueError, 'origin differs'): self.identity()

    def test_installed_symlink_rejected(self):
        self.install_fixture(); p = self.site / 'cofferdb/__init__.py'; p.unlink(); p.symlink_to(self.source_path)
        with self.assertRaisesRegex(ValueError, 'symbolic link'): self.identity()

    def test_wheel_record_mismatch_rejected(self):
        self.files['cofferdb/__init__.py'] = b'altered'; self.write_wheel(); self.args.wheel_sha256 = t.sha(self.wheel.read_bytes())
        with self.assertRaisesRegex(ValueError, 'RECORD payload'): self.payload()

    def test_wheel_native_or_tag_mismatch_rejected(self):
        self.args.native_sha256 = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'Native payload'): self.payload()
        self.args.native_sha256 = t.sha(self.files[self.native]); self.tag = 'cp313-cp313-macosx_15_0_arm64'
        with self.assertRaisesRegex(ValueError, 'tag'): self.payload()

    def test_duplicate_archive_member_rejected(self):
        with zipfile.ZipFile(self.wheel, 'a') as archive:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); archive.writestr('cofferdb/__init__.py', b'duplicate')
        self.args.wheel_sha256 = t.sha(self.wheel.read_bytes())
        with self.assertRaisesRegex(ValueError, 'unique'): self.payload()

    def test_nonempty_venv_stops_before_install(self):
        (self.site / 'prior.txt').write_bytes(b'prior')
        with self.assertRaisesRegex(ValueError, 'empty'): self.run_execute()
        self.assertEqual(self.calls, [])

    def test_context_missing_stops_before_install(self):
        self.context.write_bytes(b'{}'); self.args.context_sha256 = t.sha(self.context.read_bytes())
        with self.assertRaisesRegex(ValueError, 'context'): self.run_execute()
        self.assertEqual(self.calls, [])

    def test_wheel_drift_before_install_stops(self):
        self.wheel.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Wheel digest'): self.run_execute()
        self.assertEqual(self.calls, [])

    def test_external_driver_failure_is_hold(self):
        self.driver.write_text('def main():\n    return 1\n'); self.args.driver_sha256 = t.sha(self.driver.read_bytes())
        with self.assertRaisesRegex(ValueError, 'business driver failed'): self.run_execute()
        self.assertTrue((self.parent / 'smoke-installation/HOLD.json').exists())

    def test_deadline_after_command_record_prevents_installer_launch(self):
        now = [0.0]; original = t.save
        def save(path, value):
            original(path, value)
            if Path(path).name == 'command-0.json': now[0] = 241.0
        with patch.object(t.time, 'monotonic', side_effect=lambda: now[0]), patch.object(t, 'save', side_effect=save):
            with self.assertRaisesRegex(ValueError, 'deadline before process launch'): self.run_execute()
        self.assertEqual(self.calls, []); self.assertEqual(self.loads, [])
        self.assertTrue((self.parent / 'smoke-installation/HOLD.json').exists())

    def test_deadline_after_stream_open_prevents_installer_launch(self):
        now = [0.0]; original = Path.open
        def opened(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            if path.name == 'command-0.stderr.raw': now[0] = 241.0
            return stream
        with patch.object(t.time, 'monotonic', side_effect=lambda: now[0]), patch.object(Path, 'open', opened):
            with self.assertRaisesRegex(ValueError, 'deadline before process launch'): self.run_execute()
        self.assertEqual(self.calls, []); self.assertEqual(self.loads, [])
        self.assertTrue((self.parent / 'smoke-installation/HOLD.json').exists())

    def test_deadline_after_native_import_prevents_business_driver(self):
        now = [0.0]; marker = self.root / 'driver-entered'
        self.driver.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("entered")\ndef main():\n    return 0\n')
        self.args.driver_sha256 = t.sha(self.driver.read_bytes())
        def load(name):
            module = self.fake_import(name)
            if name == 'cofferdb._native': now[0] = 241.0
            return module
        with patch.object(t.time, 'monotonic', side_effect=lambda: now[0]):
            with self.assertRaisesRegex(ValueError, 'budget exhausted'): self.run_execute(load=load)
        self.assertFalse(marker.exists())
        self.assertTrue((self.parent / 'smoke-installation/HOLD.json').exists())

    def test_deadline_after_driver_module_prevents_main_entry(self):
        now = [0.0]; marker = self.root / 'driver-main-entered'
        self.driver.write_text('import time\ntime.monotonic()\ndef main():\n    from pathlib import Path\n    Path(' + repr(str(marker)) + ').write_text("entered")\n    return 0\n')
        self.args.driver_sha256 = t.sha(self.driver.read_bytes())
        def clock():
            # The loaded driver is identified by its real code filename.
            if sys._getframe(1).f_code.co_filename == str(self.driver): now[0] = 241.0
            return now[0]
        with patch.object(t.time, 'monotonic', new=clock):
            with self.assertRaisesRegex(ValueError, 'budget exhausted'): self.run_execute()
        self.assertFalse(marker.exists())

    def test_wrong_pip_wheel_stops_before_any_process(self):
        self.pip_wheel.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'Fixed pip'): self.run_execute()
        self.assertEqual(self.calls, [])

    def test_wrong_actual_pip_version_prevents_sdk_install(self):
        def wrong(argv, **kw):
            value = self.fake_process(argv, **kw)
            if len(self.calls) == 3:
                kw['stdout'].seek(0); kw['stdout'].truncate()
                kw['stdout'].write(t.canonical({'version': '26.0', 'path': str(self.site / 'pip/__init__.py')}))
            return value
        with self.assertRaisesRegex(ValueError, 'Actual installer'): self.run_execute(process=wrong)
        self.assertEqual(len(self.calls), 3); self.assertEqual(self.loads, [])

    def test_pip_member_tampering_prevents_sdk_install(self):
        def wrong(argv, **kw):
            value = self.fake_process(argv, **kw)
            if len(self.calls) == 3: (self.site / 'pip/__init__.py').write_bytes(b'changed')
            return value
        with self.assertRaisesRegex(ValueError, 'Installed pip member'): self.run_execute(process=wrong)
        self.assertEqual(len(self.calls), 3); self.assertEqual(self.loads, [])

    def test_default_is_inert(self):
        with patch('sys.argv', ['install_driver.py']), patch.object(t, 'load_bridge', side_effect=AssertionError('NO READ')), patch('sys.stdout', new=io.StringIO()) as output:
            self.assertEqual(t.main(), 0)
        self.assertIn('NOT_EXECUTED', output.getvalue())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--runtime-bridge', required=True)
    args, rest = parser.parse_known_args(); BRIDGE = args.runtime_bridge
    unittest.main(argv=['test_install_driver.py', *rest])
