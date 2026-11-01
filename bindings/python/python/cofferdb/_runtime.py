"""Reject runtimes outside this package's declared native support matrix."""

import os
import platform
import sys
import sysconfig

_SUPPORTED = (
    "CofferDB requires regular-GIL CPython 3.12, 3.13 or 3.14 on "
    "macOS 15+ arm64 or Linux x86_64 with glibc 2.28+. "
    "See the versioned installation and compatibility guide."
)


def check_runtime() -> None:
    """Check declared runtime prerequisites; this is not an installation test."""
    if sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise ImportError(_SUPPORTED)
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise ImportError(_SUPPORTED)
    if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled():
        raise ImportError(_SUPPORTED)
    machine = platform.machine()
    if sys.platform == "darwin":
        if machine != "arm64":
            raise ImportError(_SUPPORTED)
        try:
            version = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
        except ValueError:
            raise ImportError(_SUPPORTED) from None
        if version < (15, 0):
            raise ImportError(_SUPPORTED)
        return
    if sys.platform != "linux" or machine != "x86_64":
        raise ImportError(_SUPPORTED)
    try:
        implementation, version = os.confstr("CS_GNU_LIBC_VERSION").split()
        baseline = tuple(int(part) for part in version.split(".")[:2])
    except (AttributeError, OSError, TypeError, ValueError):
        raise ImportError(_SUPPORTED) from None
    if implementation != "glibc" or baseline < (2, 28):
        raise ImportError(_SUPPORTED)
