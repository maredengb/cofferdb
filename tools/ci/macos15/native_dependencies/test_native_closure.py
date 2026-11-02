"""Tiny synthetic Mach-O and file graphs, never executed as native code."""
import argparse
import json
import copy
import importlib.util
import io
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('native_closure_test_target', HERE / 'native_closure.py')
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
OBSERVER = PLAN = None


def text_command(command, text, header_size):
    value = text.encode() + b'\0'
    size = (header_size + len(value) + 7) // 8 * 8
    return struct.pack('<III', command, size, header_size) + b'\0' * (header_size - 12) + value + b'\0' * (size - header_size - len(value))


def image(deps=(), rpaths=(), minimum=(11, 0, 0), cpu=0x0100000c, kind=6, extra=()):
    commands = [struct.pack('<6I', 0x32, 24, 1, minimum[0] << 16 | minimum[1] << 8 | minimum[2], 15 << 16, 0)]
    commands += [text_command(command, name, 24) for command, name in deps]
    commands += [text_command(0x8000001c, path, 12) for path in rpaths]
    commands += list(extra)
    return struct.pack('<8I', 0xfeedfacf, cpu, 0, kind, len(commands), sum(map(len, commands)), 0, 0) + b''.join(commands)


def config():
    return {'schema_version': 1, 'kind': 'mac15_dynamic_input_config_1',
            'source_repository': 'maredengb/coffered', 'source_commit': '1' * 40,
            'helper_sha256': dict(PLAN.HELPERS), 'actions': dict(PLAN.ACTIONS),
            'python_providers': list(PLAN.PROVIDERS), 'selection_method': PLAN.METHOD,
            'native_closure_gate': 'independent-exact-closure-input-v1',
            'driver_contract_sha256': None, 'candidate_manifest_sha256': 'f' * 64,
            'native_closure_validator_sha256': 'e' * 64}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cofferdb-native-graph-fixture-', dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.popen = patch.object(subprocess, 'Popen', side_effect=AssertionError('NO PROCESS EXECUTION'))
        self.popen.start(); self.addCleanup(self.popen.stop)
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('NO NETWORK'))
        self.network.start(); self.addCleanup(self.network.stop)
        self.config = config()
        self.snapshot = {'schema_version': 1, 'kind': 'observed_mac15_technical_inputs_1',
                         'config_sha256': n.sha(n.canonical(self.config)), 'source_commit': self.config['source_commit'],
                         'os': {'system': 'Darwin', 'machine': 'arm64', 'version': '15.7.1'},
                         'runtimes': [], 'tools': {}, 'readonly_index_sha256': ''}
        self.index = []
        for minor in (12, 13, 14):
            provider = self.root / ('cp3' + str(minor))
            stdlib = provider / 'stdlib'; stdlib.mkdir(parents=True)
            self.put(stdlib / 'tiny.py', b'x = 1\n')
            self.put(provider / 'lib.dylib', image(deps=[(0xc, '/usr/lib/libSystem.B.dylib')]))
            self.put(provider / 'python3', image(kind=2, deps=[(0xc, '@loader_path/lib.dylib')]))
            (provider / 'python3').chmod(0o755)
            self.put(stdlib / 'extension.so', image(kind=8, deps=[(0xc, '@loader_path/../lib.dylib')]))
            rows = OBSERVER.index_tree(stdlib, time.monotonic() + 5)
            self.index += rows
            self.snapshot['runtimes'].append({'minor': minor, 'micro': 1, 'releaselevel': 'final',
                'implementation': 'cpython', 'free_threaded': False, 'binary': self.file_row(provider / 'python3'),
                'stdlib': str(stdlib), 'stdlib_index_sha256': n.sha(n.canonical(rows)), 'runtime_files': []})
        self.snapshot['readonly_index_sha256'] = n.sha(n.canonical(self.index))
        for name in ('cargo', 'rustc', 'maturin', 'sandbox', 'compiler', 'prepare', 'builder'):
            selected = self.root / 'tools' / name
            self.put(selected, ('fixture tool ' + name).encode())
            if name == 'compiler': selected.chmod(0o755)
            self.snapshot['tools'][name] = self.file_row(selected)

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data); path.chmod(0o644)

    def file_row(self, path):
        return {'path': str(path), 'sha256': n.sha(path.read_bytes())}

    def inspect(self):
        return n.inspect(self.config, self.snapshot, self.index, 'e' * 64,
                         str(self.root / 'native-evidence.json'), OBSERVER, PLAN)

    def test_default_verify_no_host_read_or_process(self):
        args = type('Args', (), {'action': 'verify'})()
        with patch.object(n.argparse.ArgumentParser, 'parse_args', return_value=args), \
             patch.object(n.platform, 'system', side_effect=AssertionError('NO HOST READ')), \
             patch.object(n, 'helpers', side_effect=AssertionError('NO HELPER READ')), \
             patch('sys.stdout', new=io.StringIO()) as stream:
            self.assertEqual(n.main(), 0)
        self.assertIn('NOT_EXECUTED', stream.getvalue())

    def test_non15_host_rejected_before_inputs(self):
        args = type('Args', (), {'action': 'inspect'})()
        with patch.object(n.argparse.ArgumentParser, 'parse_args', return_value=args), \
             patch.object(n.platform, 'system', return_value='Darwin'), \
             patch.object(n.platform, 'machine', return_value='arm64'), \
             patch.object(n.platform, 'mac_ver', return_value=('27.0', (), '')):
            with self.assertRaisesRegex(ValueError, 'MAC15'): n.main()

    def test_parse_thin_dylib_rpaths_and_minimum(self):
        result = n.macho(image(deps=[(0xc, '@rpath/lib.dylib')], rpaths=['@loader_path']), [15, 0, 0])
        self.assertEqual(result['minimum_macos'], [11, 0, 0])
        self.assertEqual(result['dependencies'], [{'name': '@rpath/lib.dylib', 'kind': 'required'}])

    def test_fat32_and_fat64_select_actual_arm64(self):
        arm = image(); intel = image(cpu=0x01000007)
        for magic, width in ((0xcafebabe, 20), (0xcafebabf, 32)):
            start = 8 + 2 * width
            fmt = '>IIIII' if width == 20 else '>IIQQII'
            a = (0x01000007, 3, start, len(intel), 0)
            b = (0x0100000c, 0, start + len(intel), len(arm), 0)
            table = struct.pack(fmt, *(a if width == 20 else (*a, 0))) + struct.pack(fmt, *(b if width == 20 else (*b, 0)))
            self.assertEqual(n.macho(struct.pack('>II', magic, 2) + table + intel + arm, [15, 0, 0])['minimum_macos'], [11, 0, 0])

    def test_fat_out_of_bounds_and_duplicate_arm_fail(self):
        data = image()
        bad = struct.pack('>II', 0xcafebabe, 1) + struct.pack('>IIIII', 0x0100000c, 0, 9999, len(data), 0) + data
        with self.assertRaisesRegex(ValueError, 'bounds'): n.macho(bad, [15, 0, 0])
        table = struct.pack('>IIIII', 0x0100000c, 0, 48, len(data), 0)
        table += struct.pack('>IIIII', 0x0100000c, 0, 48 + len(data), len(data), 0)
        with self.assertRaisesRegex(ValueError, 'Exactly one'): n.macho(struct.pack('>II', 0xcafebabe, 2) + table + data * 2, [15, 0, 0])

    def test_newer_os_other_arch_and_required_commands_fail(self):
        for raw in (image(minimum=(16, 0, 0)), image(cpu=0x01000007), image(extra=[struct.pack('<II', 0x80009999, 8)])):
            with self.assertRaises(ValueError): n.macho(raw, [15, 0, 0])

    def test_malformed_commands_and_truncation_fail(self):
        for raw in (b'\xcf\xfa\xed\xfe', image()[:-1], image(extra=[struct.pack('<II', 0xc, 8)])):
            with self.assertRaises(ValueError): n.macho(raw, [15, 0, 0])

    def test_embedded_dyld_environment_cannot_bypass_resolution_model(self):
        raw = image(kind=2, extra=[text_command(0x27, 'DYLD_LIBRARY_PATH=/fixture/override', 12)])
        with self.assertRaisesRegex(ValueError, 'dependency-affecting'): n.macho(raw, [15, 0, 0])

    def test_system_dynamic_loader_recorded_custom_loader_rejected(self):
        raw = image(kind=2, extra=[text_command(0xe, '/usr/lib/dyld', 12)])
        self.assertIn({'name': '/usr/lib/dyld', 'kind': 'system_dynamic_loader'}, n.macho(raw, [15, 0, 0])['dependencies'])
        raw = image(kind=2, extra=[text_command(0xe, '/fixture/custom-dyld', 12)])
        with self.assertRaisesRegex(ValueError, 'Non-system'): n.macho(raw, [15, 0, 0])

    def test_full_three_graphs_derive_external_files_and_fresh(self):
        result = self.inspect()
        self.assertEqual(len(result['report']['graphs']), 3)
        for runtime in result['snapshot']['runtimes']:
            self.assertEqual(len(runtime['runtime_files']), 1)
            self.assertTrue(runtime['runtime_files'][0]['path'].endswith('/lib.dylib'))
        self.assertFalse(result['report']['arbitrary_dynamic_loads_verified'])
        self.assertFalse(result['product_acceptance'])
        self.assertEqual(n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)['status'], 'DECLARED_INPUTS_FRESH_NOT_EXECUTED')

    def test_system_cache_names_are_os_inputs_not_fake_hashes(self):
        graph = self.inspect()['report']['graphs'][0]
        edges = [e for e in graph['edges'] if e['name'] == '/usr/lib/libSystem.B.dylib']
        self.assertTrue(edges)
        self.assertTrue(all(e['status'] == 'system_shared_cache_or_OS_file' for e in edges))
        self.assertFalse(any(r['path'].startswith('/usr/lib') for r in graph['files']))

    def test_rpath_unique_and_ambiguous_resolution(self):
        one = self.root / 'one'; two = self.root / 'two'
        self.put(one / 'lib.so', image()); self.put(two / 'lib.so', image())
        base = str(one / 'python')
        self.assertEqual(n.resolve_dependency('@rpath/lib.so', base, base, [str(one)])[1], str(one / 'lib.so'))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            n.resolve_dependency('@rpath/lib.so', base, base, [str(one), str(two)])

    def test_exact_loader_and_executable_path_expansion(self):
        self.assertEqual(n.expanded('@loader_path', '/fixture/lib/a', '/fixture/bin/python'), '/fixture/lib')
        self.assertEqual(n.expanded('@executable_path/../lib/b', '/fixture/other/a', '/fixture/bin/python'), '/fixture/lib/b')
        with self.assertRaises(ValueError): n.expanded('relative.dylib', '/fixture/a', '/fixture/python')

    def test_required_missing_and_weak_absence(self):
        lib = self.root / 'cp312/lib.dylib'
        self.put(lib, image(deps=[(0xc, '@loader_path/missing.dylib')]))
        with self.assertRaisesRegex(ValueError, 'Required native'): self.inspect()
        self.put(lib, image(deps=[(0x80000018, '@loader_path/missing.dylib')]))
        result = self.inspect()
        self.assertIn('weak_dependency_absent', [e['status'] for e in result['report']['graphs'][0]['edges']])
        self.put(lib.parent / 'missing.dylib', image())
        with self.assertRaisesRegex(ValueError, 'graph changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_native_symlink_alias_is_exactly_resolved_and_rechecked(self):
        lib = self.root / 'cp312/lib.dylib'; target = self.root / 'cp312/versioned.dylib'
        lib.rename(target); lib.symlink_to(target.name)
        result = self.inspect()
        self.assertEqual(result['snapshot']['runtimes'][0]['runtime_files'][0]['path'], str(target))
        other = self.root / 'cp312/other.dylib'; self.put(other, target.read_bytes())
        lib.unlink(); lib.symlink_to(other.name)
        with self.assertRaisesRegex(ValueError, 'graph changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_selected_base_keeps_its_interpreter_role_in_native_graph(self):
        calls = []; original = OBSERVER.native_bytes
        def record(value, bound, **kwargs):
            calls.append((value, kwargs.get('python_base')))
            return original(value, bound, **kwargs)
        with patch.object(OBSERVER, 'native_bytes', side_effect=record):
            self.inspect()
        for runtime in self.snapshot['runtimes']:
            base = runtime['binary']['path']
            self.assertIn((base, base), calls)

    def test_stdlib_addition_or_byte_mutation_fails_fresh(self):
        result = self.inspect()
        self.put(self.root / 'cp312/stdlib/late.py', b'late')
        with self.assertRaisesRegex(ValueError, 'stdlib changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)
        (self.root / 'cp312/stdlib/late.py').unlink()
        self.put(self.root / 'cp312/stdlib/tiny.py', b'x = 2\n')
        with self.assertRaisesRegex(ValueError, 'stdlib changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_external_native_mode_bytes_and_tool_drift_fail(self):
        result = self.inspect(); lib = self.root / 'cp312/lib.dylib'; old = lib.read_bytes()
        self.put(lib, image(minimum=(12, 0, 0)))
        with self.assertRaisesRegex(ValueError, 'graph changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)
        self.put(lib, old); lib.chmod(0o666)
        with self.assertRaisesRegex(ValueError, 'Protected'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)
        lib.chmod(0o644); self.put(self.root / 'tools/cargo', b'changed')
        with self.assertRaisesRegex(ValueError, 'tool identity or protection changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_tool_becoming_group_writable_rejected(self):
        result = self.inspect()
        (self.root / 'tools/builder').chmod(0o666)
        with self.assertRaisesRegex(ValueError, 'tool identity or protection changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_last_tool_read_expiring_budget_cannot_sign_fresh(self):
        result = self.inspect(); clock = [1000.0]; original = OBSERVER.bytes_
        def read(path, bound):
            raw = original(path, bound)
            if path == str(self.root / 'tools/builder'): clock[0] = 1181.0
            return raw
        with patch.object(n.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(OBSERVER, 'bytes_', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'tool identity or protection changed'):
                n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_host_identity_preserves_real_collector_image_metadata(self):
        self.snapshot['os'].update(ImageOS='macos15', ImageVersion='fixture-image')
        with patch.object(n.platform, 'system', return_value='Darwin'), \
             patch.object(n.platform, 'machine', return_value='arm64'), \
             patch.object(n.platform, 'mac_ver', return_value=('15.7.1', (), '')):
            n.same_host(self.snapshot)
            self.snapshot['os']['version'] = '15.7.0'
            with self.assertRaisesRegex(ValueError, 'host OS changed'): n.same_host(self.snapshot)

    def test_index_digest_foreign_path_and_runtime_order_rejected(self):
        self.snapshot['readonly_index_sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'index identity'): self.inspect()
        self.snapshot['readonly_index_sha256'] = n.sha(n.canonical(self.index))
        self.snapshot['runtimes'].reverse()
        with self.assertRaisesRegex(ValueError, 'Ordered'): self.inspect()
        self.snapshot['runtimes'].reverse()
        row = copy.deepcopy(self.index[0]); row['path'] = str(self.root / 'foreign.py')
        self.index.append(row); self.snapshot['readonly_index_sha256'] = n.sha(n.canonical(self.index))
        with self.assertRaisesRegex(ValueError, 'outside'): self.inspect()

    def test_config_and_snapshot_binding_rejected(self):
        self.config['native_closure_validator_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'binding differs'): self.inspect()

    def test_graph_deletion_cannot_shorten_fresh_check(self):
        result = self.inspect(); result['report']['graphs'].pop()
        with self.assertRaisesRegex(ValueError, 'count'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_graph_file_deletion_cannot_hide_actual_dependency(self):
        result = self.inspect(); result['report']['graphs'][0]['files'].pop()
        with self.assertRaisesRegex(ValueError, 'graph changed'):
            n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def test_no_missing_tool_or_empty_tool_map_signs_fresh(self):
        for tools in ({}, {k: None for k in self.snapshot['tools']}):
            self.snapshot['tools'] = tools
            result = self.inspect()
            with self.assertRaises(ValueError): n.fresh(result['snapshot'], self.index, result['report'], OBSERVER)

    def framework_graph_fixture(self):
        versions = self.root / 'Framework/Versions'; base = versions / '3.12'
        stdlib = base / 'lib/python3.12'; config_dir = stdlib / 'config-3.12-darwin'
        config_dir.mkdir(parents=True)
        core = base / 'Python'; self.put(core, image(deps=[(0xc, '/usr/lib/libSystem.B.dylib')]))
        binary = base / 'bin/python3.12'; self.put(binary, image(kind=2))
        for suffix in ('a', 'dylib'):
            (config_dir / ('libpython3.12.' + suffix)).symlink_to('../../../Python')
        owner = (stdlib.stat().st_uid, stdlib.stat().st_gid)
        with patch.object(OBSERVER, 'FRAMEWORK_VERSIONS', versions), patch.object(OBSERVER, 'FRAMEWORK_ALIAS_OWNER', owner), patch.object(OBSERVER, 'native_protection'):
            index = OBSERVER.index_tree(stdlib, time.monotonic() + 5)
        runtime = {'minor': 12, 'binary': self.file_row(binary), 'stdlib': str(stdlib), 'stdlib_index_sha256': n.sha(n.canonical(index))}
        return versions, runtime, index, core, owner
    def test_framework_aliases_use_one_canonical_native_seed(self):
        versions, runtime, index, core, owner = self.framework_graph_fixture()
        read = []
        def reader(path, bound):
            read.append(path); return OBSERVER.bytes_(path, bound)
        with patch.object(n, 'FRAMEWORK_VERSIONS', versions), patch.object(n, 'FRAMEWORK_ALIAS_OWNER', owner):
            graph = n.inspect_runtime(runtime, index, [15, 0, 0], reader, time.monotonic() + 5)
        self.assertEqual(read.count(str(core)), 1)
        self.assertEqual(len(graph['aliases']), 2)
        self.assertTrue(all(target == str(core) for target in graph['aliases'].values()))
        self.assertIn(str(core), [row['path'] for row in graph['runtime_files']])
        self.assertFalse(any('/config-' in path for path in read))
    def test_framework_native_target_bytes_or_alias_record_change_rejected(self):
        versions, runtime, index, core, owner = self.framework_graph_fixture()
        with patch.object(n, 'FRAMEWORK_VERSIONS', versions), patch.object(n, 'FRAMEWORK_ALIAS_OWNER', owner):
            bad = json.loads(json.dumps(index)); bad[0]['alias']['text'] = '../../../Other'
            runtime['stdlib_index_sha256'] = n.sha(n.canonical(bad))
            with self.assertRaisesRegex(ValueError, 'alias target'):
                n.inspect_runtime(runtime, bad, [15, 0, 0], OBSERVER.bytes_, time.monotonic() + 5)
            runtime['stdlib_index_sha256'] = n.sha(n.canonical(index))
            self.put(core, image(deps=[(0xc, '/usr/lib/libobjc.A.dylib')]))
            with self.assertRaisesRegex(ValueError, 'Framework target changed'):
                n.inspect_runtime(runtime, index, [15, 0, 0], OBSERVER.bytes_, time.monotonic() + 5)
    def test_native_graph_deadline_and_read_bound_fail_closed(self):
        runtime = self.snapshot['runtimes'][0]
        with self.assertRaisesRegex(ValueError, 'deadline'):
            n.inspect_runtime(runtime, self.index, [15, 0, 0], OBSERVER.bytes_, time.monotonic() - 1)
        with patch.object(n, 'MAX_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                n.inspect_runtime(runtime, self.index, [15, 0, 0], OBSERVER.bytes_, time.monotonic() + 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper-root', required=True)
    args, rest = parser.parse_known_args()
    OBSERVER, PLAN = n.helpers(args.helper_root)
    unittest.main(argv=[__file__, *rest])
