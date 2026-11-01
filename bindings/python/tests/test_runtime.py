"""Declared runtime guard checks; these do not replace native wheel smoke."""

import types
import unittest
from unittest.mock import patch

from cofferdb._runtime import check_runtime


class RuntimeGuardTests(unittest.TestCase):
    def check(self, *, os_name="darwin", machine="arm64", version=(3, 12),
              gil=True, interpreter="cpython", mac="15.0", libc="glibc 2.28"):
        with (
            patch("cofferdb._runtime.sys.implementation", types.SimpleNamespace(name=interpreter)),
            patch("cofferdb._runtime.sys.version_info", version),
            patch("cofferdb._runtime.sys.platform", os_name),
            patch("cofferdb._runtime.sysconfig.get_config_var", return_value=0 if gil else 1),
            patch("cofferdb._runtime.platform.machine", return_value=machine),
            patch("cofferdb._runtime.platform.mac_ver", return_value=(mac, (), machine)),
            patch("cofferdb._runtime.os.confstr", return_value=libc),
        ):
            check_runtime()

    def test_declared_six_combinations_pass_the_guard(self):
        for version in [(3, 12), (3, 13), (3, 14)]:
            for os_name, machine in [("darwin", "arm64"), ("linux", "x86_64")]:
                with self.subTest(version=version, os=os_name):
                    self.check(version=version, os_name=os_name, machine=machine)

    def test_unsupported_runtimes_are_rejected(self):
        for override in [
            {"version": (3, 11)}, {"version": (3, 15)}, {"gil": False},
            {"interpreter": "pypy"}, {"machine": "x86_64"},
            {"os_name": "win32"}, {"mac": "10.15"}, {"mac": "11.7"},
            {"mac": "12.7"}, {"mac": "13.7"}, {"mac": "14.9"}, {"mac": ""},
            {"os_name": "linux", "machine": "aarch64"},
            {"os_name": "linux", "machine": "x86_64", "libc": "glibc 2.27"},
            {"os_name": "linux", "machine": "x86_64", "libc": "musl 1.2"},
            {"os_name": "linux", "machine": "x86_64", "libc": None},
        ]:
            with self.subTest(override=override), self.assertRaisesRegex(ImportError, "regular-GIL"):
                self.check(**override)


if __name__ == "__main__":
    unittest.main()
