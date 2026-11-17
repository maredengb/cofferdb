"""Installed-file role controls; no SDK, external tool, network or hosted job."""
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('host_runner', Path(__file__).with_name('runner.py'))
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class HostFiles(unittest.TestCase):
    def info(self, *, mode=0o775, size=152096, links=1, uid=None):
        return types.SimpleNamespace(st_mode=stat.S_IFREG | mode, st_size=size,
                                     st_nlink=links, st_uid=os.getuid() if uid is None else uid)

    def test_installed_python_775_is_narrow_and_still_single_link(self):
        source = '/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python'
        self.assertEqual(r.host_protection(source, self.info(), 'python'), 128 * 1024**2)
        for info in (self.info(mode=0o777), self.info(links=2), self.info(uid=987654), self.info(size=129*1024**2)):
            with self.assertRaises(ValueError): r.host_protection(source, info, 'python')
        for value in ('/private/tmp/Python', '/Library/Frameworks/Other.framework/Versions/3.12/Python'):
            with self.assertRaises(ValueError): r.host_protection(value, self.info(), 'python')

    def test_only_declared_clang_receives_larger_bound(self):
        for value in ('/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang',
                      '/Library/Developer/CommandLineTools/usr/bin/clang'):
            self.assertEqual(r.host_protection(value, self.info(mode=0o755, size=264942528), 'compiler'),512*1024**2)
            for info in (self.info(mode=0o775, size=264942528),self.info(mode=0o755,size=513*1024**2)):
                with self.assertRaises(ValueError): r.host_protection(value, info, 'compiler')
        with self.assertRaises(ValueError):
            r.host_protection('/private/tmp/clang',self.info(mode=0o755,size=264942528),'compiler')
        with self.assertRaises(ValueError):
            r.host_protection('/Library/Developer/CommandLineTools/usr/bin/clang',self.info(mode=0o755,size=264942528),'native')

    def test_private_copy_775_needs_owned_0700_root_and_exact_digest(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            root=Path(folder);(root/'bin').mkdir();p=root/'bin/python';p.write_bytes(b'copy');p.chmod(0o775)
            row={'path':str(p),'sha256':hashlib.sha256(b'copy').hexdigest()}
            self.assertEqual(r.row(row, executable=True, host_role='private_python',private_root=str(root)),p)
            with self.assertRaisesRegex(ValueError,'File bytes changed'):
                r.row(dict(row,sha256='a'*64),host_role='private_python',private_root=str(root))
            with self.assertRaises(ValueError):r.row(row)
            root.chmod(0o750)
            with self.assertRaises(ValueError):r.host_file(str(p),role='private_python',private_root=str(root))

    def test_no_link_or_in_read_mutation_is_accepted(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            p=Path(folder)/'file';p.write_bytes(b'original');p.chmod(0o600)
            alias=p.with_name('alias');alias.symlink_to(p)
            with self.assertRaises(ValueError):r.host_file(str(alias),role='native')
            real_read=os.read;changed=False
            def mutate(fd,n):
                nonlocal changed
                data=real_read(fd,n)
                if not changed:
                    changed=True;p.write_bytes(b'changed!')
                return data
            with patch.object(r.os,'read',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'changed during read'):r.host_file(str(p),role='native')

    def test_native_exception_does_not_apply_to_source_or_data(self):
        root='/Library/Frameworks/Python.framework/Versions/3.12/'
        self.assertEqual(r.host_protection(root+'lib/libpython.dylib',self.info(mode=0o664),'native'),128*1024**2)
        for tail in ('lib/config.json','lib/test.py'):
            with self.assertRaises(ValueError):r.host_protection(root+tail,self.info(),'native')

    def test_bundled_tcl_tk_images_keep_native_metadata_protection(self):
        paths = [f'/Library/Frameworks/Python.framework/Versions/3.{minor}/Frameworks/'
                 f'{name}.framework/Versions/8.6/{name}'
                 for minor in (12, 13, 14) for name in ('Tcl', 'Tk')]
        paths += [f'/Library/Frameworks/Python.framework/Versions/3.14/Frameworks/'
                  f'{name}.framework/Versions/9.0/{name}' for name in ('Tcl', 'Tk')]
        self.assertEqual(set(paths), r.FRAMEWORK_NATIVE_IMAGES)
        for value in paths:
            with self.subTest(value=value):
                for mode in (0o575, 0o775):
                    self.assertEqual(r.host_protection(value, self.info(mode=mode, size=4664640, uid=0),
                                                       'native'), 128 * 1024**2)
                for info in (self.info(mode=0o577), self.info(mode=0o4575),
                             self.info(links=2), self.info(uid=987654),
                             self.info(size=129 * 1024**2)):
                    with self.assertRaises(ValueError): r.host_protection(value, info, 'native')
                info = self.info(mode=0o575); info.st_mode = stat.S_IFDIR | 0o575
                with self.assertRaises(ValueError): r.host_protection(value, info, 'native')
                with self.assertRaises(ValueError): r.host_protection(value, self.info(), 'python')

    def test_tcl9_runtime_row_reaches_input_checks_and_profile_as_literal(self):
        tcl = '/Library/Frameworks/Python.framework/Versions/3.14/Frameworks/Tcl.framework/Versions/9.0/Tcl'
        native = {'path': tcl, 'sha256': 'a' * 64}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            root = Path(folder); root.chmod(0o700)
            binary = root / 'python'; binary.write_bytes(b'fixture')
            plan = {'sandbox': {'path': '/usr/bin/sandbox-exec'},
                    'supervisor': {'path': '/fixture/supervisor'},
                    'compiler': {'path': '/fixture/clang'},
                    'python': {'invocation': str(binary), 'real': str(binary),
                               'base': str(binary), 'sha256': 'b' * 64},
                    'venv': str(root), 'stdlib': str(root), 'dependencies': [],
                    'runtime_files': [native], 'probe_sha256': 'c' * 64,
                    'driver': {'path': '/fixture/driver', 'sha256': 'd' * 64, 'inputs': []},
                    'work': str(root)}
            with patch.object(r, 'row') as checked:
                r.actual_inputs(plan)
            self.assertIn(((native,), {'host_role': 'native'}),
                          [(call.args, call.kwargs) for call in checked.call_args_list])
            with patch.object(r, 'schema', return_value=plan):
                rendered = r.profile(plan)
            self.assertIn('(literal "' + tcl + '")', rendered)
            self.assertNotIn('(subpath "' + tcl + '")', rendered)
            self.assertIn('(allow file-read* (literal "/"))', rendered)
            self.assertNotIn('(subpath "/")', rendered)
            self.assertIn('(literal "/fixture/clang")', rendered)
            self.assertIsNone(r.framework_app_binary(str(binary)))

    def test_framework_launcher_is_one_exec_and_read_literal(self):
        base = '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12'
        app = '/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python'
        self.assertEqual(str(r.framework_app_binary(base)), app)
        homebrew = ('/opt/homebrew/Cellar/python@3.12/3.12.16/Frameworks/Python.framework/'
                    'Versions/3.12/bin/python3.12')
        self.assertEqual(r.framework_app_binary(homebrew).name, 'Python')
        self.assertIsNone(r.framework_app_binary(
            '/Users/runner/hostedtoolcache/Python/3.12.10/arm64/bin/python3.12'))
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            root = Path(folder); root.chmod(0o700)
            binary = root / 'python'; binary.write_bytes(b'fixture')
            plan = {'sandbox': {'path': '/usr/bin/sandbox-exec'},
                    'supervisor': {'path': '/fixture/supervisor'},
                    'compiler': {'path': '/Applications/Xcode_16.4.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'},
                    'python': {'invocation': str(binary), 'real': str(binary),
                               'base': base, 'sha256': 'b' * 64, 'minor': 12},
                    'venv': str(root), 'stdlib': '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12',
                    'dependencies': [], 'runtime_files': [], 'probe_sha256': 'c' * 64,
                    'driver': {'path': '/fixture/driver', 'sha256': 'd' * 64, 'inputs': []},
                    'work': str(root)}
            with patch.object(r, 'schema', return_value=plan):
                rendered = r.profile(plan)
        self.assertIn('(literal "' + app + '")', rendered)
        self.assertEqual(rendered.count('(literal "' + app + '")'), 2)
        self.assertNotIn('(subpath "' + app + '")', rendered)
        self.assertIn('(literal "' + plan['compiler']['path'] + '")', rendered)
        data = next(line for line in rendered.splitlines()
                    if line.startswith('(allow file-read-data '))
        self.assertIn('(literal "/")', data)
        self.assertIn('(literal "/private")', data)
        self.assertNotIn('(subpath ', data)
        self.assertNotIn('(literal "/usr/bin")', data)
        self.assertIn('(subpath "/usr/bin")', rendered)

    def test_bundled_framework_name_exception_is_exactly_eight_canonical_paths(self):
        root = '/Library/Frameworks/Python.framework/Versions/3.13/'
        fixed = root + 'Frameworks/Tcl.framework/Versions/8.6/Tcl'
        rejected = [
            fixed.replace('/8.6/', '/9.0/'), fixed.replace('/8.6/', '/8.6.14/'),
            fixed.replace('/3.13/', '/3.12/').replace('/8.6/', '/9.0/'),
            fixed.replace('/3.13/', '/3.14/').replace('/8.6/', '/9.1/'),
            fixed.replace('/3.13/', '/3.14/').replace('/8.6/', '/9.0/').replace('/Tcl.framework/', '/Tk.framework/').replace('/9.0/Tcl', '/9.1/Tk'),
            fixed.replace('/8.6/', '/Current/'), fixed.replace('/Tcl.framework/', '/Other.framework/'),
            fixed.replace('/8.6/Tcl', '/8.6/Tk'), fixed.replace('/Frameworks/Tcl.', '/lib/Frameworks/Tcl.'),
            fixed.replace('/Library/Frameworks/', '/opt/homebrew/Cellar/python@3.13/3.13.9/Frameworks/'),
            '/Users/runner/hostedtoolcache/Python/3.13.9/arm64/Frameworks/Tcl.framework/Versions/8.6/Tcl',
            root + 'Frameworks/Tcl.framework/Versions/8.6/Unknown',
            root + 'private-data', root + 'Frameworks/Tcl.framework/Versions/8.6/../8.6/Tcl',
        ]
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.host_protection(value, self.info(mode=0o775), 'native')

        # Other Python versions do not receive the six-path Framework exception;
        # host_limits applies its generic non-provider policy to them instead.
        for minor in (11, 15):
            value = fixed.replace('/3.13/', f'/3.{minor}/')
            with self.subTest(value=value):
                self.assertNotIn(value, r.FRAMEWORK_NATIVE_IMAGES)
                self.assertFalse(r.python_provider(value))
                self.assertEqual(r.host_limits(value, 'native'), (128 * 1024**2, 0o022))
                self.assertEqual(r.host_protection(value, self.info(mode=0o755), 'native'),
                                 128 * 1024**2)
                with self.assertRaisesRegex(ValueError, r'"protected_mode": false'):
                    r.host_protection(value, self.info(mode=0o775), 'native')

    def test_native_rejection_names_only_fixed_framework_alias_candidates(self):
        fixed = '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/config-3.12-darwin/libpython3.12.a'
        with self.assertRaisesRegex(ValueError, r'config-3\.12-darwin/libpython3\.12\.a'):
            r.host_limits(fixed, 'native')
        unknown = '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/private-data.a'
        with self.assertRaisesRegex(ValueError, 'Declared native file required') as caught:
            r.host_limits(unknown, 'native')
        self.assertNotIn('private-data', str(caught.exception))

    def test_native_rejection_reports_only_safe_provider_label_and_open_fd_metadata(self):
        value = '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12'
        info = self.info(mode=0o755, size=123456, links=1, uid=987654)
        with self.assertRaisesRegex(ValueError, 'Declared native file required') as caught:
            r.host_protection(value, info, 'native')
        message = str(caught.exception)
        self.assertIn('framework/3.12/bin/python3.12', message)
        self.assertIn('"mode":"0755"', message)
        self.assertIn('"links":1', message)
        self.assertIn('"bytes":123456', message)
        self.assertIn('"owner":"other"', message)
        self.assertNotIn('/Library/Frameworks', message)
        self.assertNotIn('987654', message)

    def test_native_archive_candidate_rejection_reports_fixed_label_with_fd_metadata(self):
        value = ('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/'
                 'config-3.12-darwin/libpython3.12.a')
        with self.assertRaisesRegex(ValueError, 'Declared native file required') as caught:
            r.host_protection(value, self.info(mode=0o644, size=12345), 'native')
        message = str(caught.exception)
        self.assertIn('config-3.12-darwin/libpython3.12.a', message)
        self.assertIn('"provider":"framework/3.12"', message)
        self.assertIn('"mode":"0644"', message)
        self.assertNotIn('/Library/Frameworks', message)

    def test_native_rejection_hashes_unrecognized_provider_path_name(self):
        value = '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/private-data.a'
        with self.assertRaisesRegex(ValueError, 'Declared native file required') as caught:
            r.host_protection(value, self.info(mode=0o644, size=2345, links=2), 'native')
        message = str(caught.exception)
        self.assertIn('"provider":"framework/3.12"', message)
        self.assertIn('"suffix":".a"', message)
        self.assertIn('"links":2', message)
        self.assertIn('"bytes":2345', message)
        self.assertNotIn('private-data', message)
        self.assertNotIn('/Library/Frameworks', message)


if __name__=='__main__':unittest.main()
