"""Reject runtimes outside this package's declared native support matrix."""

import os
import platform
import sys
import sysconfig

# This message serves the same support gate on both platforms. Keep its versions
# and architectures aligned with the branches below and the wheel support matrix.
_SUPPORTED = (
    "CofferDB requires regular-GIL CPython 3.12, 3.13 or 3.14 on "
    "macOS 15+ on arm64 or x86_64, or Linux aarch64 or x86_64 with glibc 2.28+. "
    "See the versioned installation and compatibility guide."
)


def check_runtime() -> None:
    """Check declared runtime prerequisites; this is not an installation test."""
    # Run these checks before importing the native module so an unsupported
    # interpreter sees the package error before the extension loader tries to
    # resolve its binary.
    if sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise ImportError(_SUPPORTED)
    # A free-threaded build can run with its GIL enabled. Check both the
    # build configuration and the active GIL before loading the extension.
    if sys.maxsize <= 2**32 or sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise ImportError(_SUPPORTED)
    if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled():
        raise ImportError(_SUPPORTED)
    # Compare the architecture reported by the running interpreter against the
    # wheel architectures. macOS and Linux use different names for the same ARM
    # architecture.
    machine = platform.machine()
    if sys.platform == "darwin":
        if machine not in {"arm64", "x86_64"}:
            raise ImportError(_SUPPORTED)
        # The wheel's deployment target is expressed in macOS product versions.
        # Darwin kernel releases use a different numbering scheme, so compare the
        # product version reported to this interpreter.
        try:
            version = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
        except ValueError:
            raise ImportError(_SUPPORTED) from None
        if version < (15, 0):
            raise ImportError(_SUPPORTED)
        return
    if sys.platform != "linux" or machine not in {"aarch64", "x86_64"}:
        raise ImportError(_SUPPORTED)
    # Linux wheel compatibility depends on the running libc as well as the
    # machine type. Treat an unavailable glibc version as an unsupported host.
    try:
        implementation, version = os.confstr("CS_GNU_LIBC_VERSION").split()
        baseline = tuple(int(part) for part in version.split(".")[:2])
    # The system query can fail or return an unexpected shape. Normalize those
    # failures to the same public ImportError as a version outside the support
    # matrix.
    except (AttributeError, OSError, TypeError, ValueError):
        raise ImportError(_SUPPORTED) from None
    if implementation != "glibc" or baseline < (2, 28):
        raise ImportError(_SUPPORTED)
