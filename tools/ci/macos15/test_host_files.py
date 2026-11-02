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


if __name__=='__main__':unittest.main()
