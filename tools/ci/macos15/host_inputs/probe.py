"""Read bounded host-file metadata without executing the selected tools."""
import json
import os
from pathlib import Path
import platform
import resource
import re
import signal
import stat
import sys

MAX_FILE = 128 * 1024 * 1024
MAX_ROWS = 20


def inspect_file(selected):
    """Mirror the regular-input metadata gate; never read or execute file bytes."""
    try:
        selected = Path(selected).resolve(strict=True)
        directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        opened = None
        try:
            for part in selected.parent.parts[1:]:
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory)
                os.close(directory)
                directory = following
            opened = os.open(selected.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
            info = os.fstat(opened)
            checks = {'regular': stat.S_ISREG(info.st_mode), 'single_link': info.st_nlink == 1,
                      'within_size': info.st_size <= MAX_FILE,
                      'protected_mode': not bool(info.st_mode & 0o022)}
            return {'status': 'OBSERVED_METADATA_ONLY', 'mode': oct(stat.S_IMODE(info.st_mode)),
                    'links': info.st_nlink, 'bytes': info.st_size, 'checks': checks,
                    'original_gate_would_accept': all(checks.values())}
        finally:
            if opened is not None:
                os.close(opened)
            os.close(directory)
    except OSError as error:
        # Paths and exception text may contain account information.
        return {'status': 'UNAVAILABLE', 'errno': error.errno,
                'error_type': type(error).__name__}
    except RuntimeError:
        return {'status': 'UNAVAILABLE', 'error_type': 'PathResolutionError'}


def fixed_inputs():
    rows = [('bootstrap_python', Path(sys.executable))]
    for minor in ('3.12', '3.13', '3.14'):
        rows.append(('framework_' + minor, Path('/Library/Frameworks/Python.framework/Versions')
                     / minor / 'bin' / ('python' + minor)))
    rows.extend((name, Path('/usr/bin') / name) for name in ('xcode-select', 'curl', 'sandbox-exec'))
    for label, root, compiler in (
            ('xcode', Path('/Applications/Xcode.app/Contents/Developer'),
             'Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'),
            ('clt', Path('/Library/Developer/CommandLineTools'), 'usr/bin/clang')):
        rows.extend(((label + '_git', root / 'usr/bin/git'), (label + '_clang', root / compiler)))
    rust = Path.home() / '.rustup/toolchains/stable-aarch64-apple-darwin/bin'
    rows.extend(('rust_' + name, rust / name) for name in ('cargo', 'rustc'))
    return rows


def main():
    if len(sys.argv) != 1:
        raise ValueError('This probe accepts no paths or commands')
    if not (platform.system() == 'Darwin' and platform.machine() == 'arm64'
            and platform.mac_ver()[0].split('.')[0] == '15'
            and sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12)):
        raise ValueError('Native macOS 15 CPython 3.12 required')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    signal.alarm(30)
    rows = fixed_inputs()
    if len(rows) > MAX_ROWS:
        raise ValueError('Metadata row bound')
    image = {}
    for key in ('ImageOS', 'ImageVersion'):
        value = os.environ.get(key, '')
        if not re.fullmatch(r'[A-Za-z0-9._-]{0,80}', value):
            raise ValueError('Unexpected image identity')
        image[key] = value
    result = {'schema_version': 1, 'status': 'HOST_METADATA_NOT_PRODUCT_VALIDATION',
              'product_acceptance': False,
              'image': image,
              'host': {'system': platform.system(), 'version': platform.mac_ver()[0],
                       'architecture': platform.machine(), 'python': list(sys.version_info[:3])},
              'files': {label: inspect_file(path) for label, path in rows}}
    raw = json.dumps(result, sort_keys=True, separators=(',', ':'))
    if len(raw.encode()) > 65536:
        raise ValueError('Metadata output bound')
    print(raw, flush=True)


if __name__ == '__main__':
    main()
