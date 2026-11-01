"""Observe bounded standard-library metadata without reading file contents."""
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import stat
import sys
import time

ROOT = Path('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12')
CORE = Path('/Library/Frameworks/Python.framework/Versions/3.12/Python')
PUBLIC_NAMES = frozenset(('config-3.12-darwin/libpython3.12.a',
                          'config-3.12-darwin/libpython3.12.dylib'))
MAX_ENTRIES = 24000
MAX_DEPTH = 64
MAX_REJECTS = 16
MAX_OUTPUT = 65536
MAX_FILE = 32 * 2**20


class Stop(ValueError):
    pass


def signature(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def attributes(value):
    kind = ('regular' if stat.S_ISREG(value.st_mode) else
            'directory' if stat.S_ISDIR(value.st_mode) else
            'symlink' if stat.S_ISLNK(value.st_mode) else 'other')
    return {'type': kind, 'mode': oct(stat.S_IMODE(value.st_mode)),
            'links': value.st_nlink, 'bytes': value.st_size,
            'uid': value.st_uid, 'gid': value.st_gid}


def checks(value):
    return {'regular': stat.S_ISREG(value.st_mode),
            'single_link': value.st_nlink == 1, 'within_size': value.st_size <= MAX_FILE}


def directory(selected):
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(selected).parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def fixed_core_metadata():
    fd = directory(CORE.parent)
    try:
        before = os.stat(CORE.name, dir_fd=fd, follow_symlinks=False)
        after = os.stat(CORE.name, dir_fd=fd, follow_symlinks=False)
        if signature(before) != signature(after):
            raise Stop('fixed_target_drift')
        return {'role': 'framework_core', 'path': str(CORE), 'metadata': attributes(before)}
    finally:
        os.close(fd)


def observe(root, emit, *, seconds=30):
    """Inspect names/stat only; unknown entry/link text never enters public output."""
    deadline = time.monotonic() + seconds
    counts = {'entries': 0, 'rejected': 0, 'undisclosed_names': 0, 'undisclosed_targets': 0}
    parent = directory(Path(root).parent)
    fd = None
    def gate():
        if time.monotonic() >= deadline:
            raise Stop('deadline')
    def stable(current, before, above, leaf):
        if (signature(before) != signature(os.fstat(current)) or
                signature(before) != signature(os.stat(leaf, dir_fd=above, follow_symlinks=False))):
            raise Stop('directory_drift')
    def walk(current, before, above, leaf, prefix, depth):
        gate()
        if depth > MAX_DEPTH:
            raise Stop('depth_bound')
        stable(current, before, above, leaf)
        with os.scandir(current) as scan:
            for item in scan:
                gate()
                counts['entries'] += 1
                if counts['entries'] > MAX_ENTRIES:
                    raise Stop('entry_bound')
                before_entry = item.stat(follow_symlinks=False)
                relative = prefix + item.name
                if stat.S_ISDIR(before_entry.st_mode):
                    child = os.open(item.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=current)
                    try:
                        opened = os.fstat(child)
                        if signature(opened) != signature(before_entry):
                            raise Stop('directory_replacement')
                        walk(child, opened, current, item.name, relative + '/', depth + 1)
                    finally:
                        os.close(child)
                elif not all(checks(before_entry).values()):
                    if counts['rejected'] >= MAX_REJECTS:
                        raise Stop('rejection_bound')
                    counts['rejected'] += 1
                    disclosed = relative in PUBLIC_NAMES
                    row = {'event': 'rejected_entry_metadata', 'ordinal': counts['rejected'],
                           'scan_ordinal': counts['entries'], 'depth': depth,
                           'metadata': attributes(before_entry), 'checks': checks(before_entry),
                           'path_disclosed': disclosed}
                    if disclosed:
                        row['relative_path'] = relative
                    else:
                        counts['undisclosed_names'] += 1
                    if disclosed and stat.S_ISLNK(before_entry.st_mode):
                        # Read only the link itself. Never resolve or open arbitrary targets.
                        target = os.readlink(item.name, dir_fd=current)
                        row['link_text_disclosed'] = disclosed and target == '../../../Python'
                        if not row['link_text_disclosed']:
                            counts['undisclosed_targets'] += 1
                        if row['link_text_disclosed']:
                            row['link_text'] = target
                            row['fixed_target'] = fixed_core_metadata()
                    after_entry = os.stat(item.name, dir_fd=current, follow_symlinks=False)
                    if signature(before_entry) != signature(after_entry):
                        raise Stop('entry_drift')
                    emit(row)
        stable(current, before, above, leaf)
    try:
        fd = os.open(Path(root).name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        initial = os.fstat(fd)
        emit({'event': 'stdlib_metadata_intent', 'root_role': 'framework_cp312_stdlib',
              'gate': {'regular': True, 'single_link': True, 'max_file_bytes': MAX_FILE},
              'scan_order': 'os.scandir_unspecified', 'file_contents_read': False,
              'sdk_executed': False})
        walk(fd, initial, parent, Path(root).name, '', 0)
        named = directory(root)
        try:
            if signature(os.fstat(named)) != signature(initial):
                raise Stop('named_root_drift')
        finally:
            os.close(named)
        emit({'event': 'stdlib_metadata_complete', **counts,
              'all_rejected_paths_disclosed': counts['undisclosed_names'] == 0,
              'all_requested_metadata_disclosed': counts['undisclosed_names'] == counts['undisclosed_targets'] == 0,
              'product_acceptance': False})
        return counts
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def main():
    if len(sys.argv) != 1:
        return 2
    if not (platform.system() == 'Darwin' and platform.machine() == 'arm64'
            and platform.mac_ver()[0].split('.')[0] == '15'
            and sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12)):
        return 2
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Stop('deadline')))
    signal.alarm(30)
    written = 0
    def emit(row):
        nonlocal written
        raw = (json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode()
        if written + len(raw) > MAX_OUTPUT:
            raise Stop('output_bound')
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        written += len(raw)
    try:
        image = {key: os.environ.get(key, '') for key in ('ImageOS', 'ImageVersion')}
        if (image['ImageOS'] not in ('macos15', 'macos15-arm64') or
                not re.fullmatch(r'[0-9]{8}\.[0-9]{1,8}(?:\.[0-9]{1,4})?', image['ImageVersion'])):
            raise Stop('image_identity')
        version = platform.mac_ver()[0]
        if not re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,2}', version):
            raise Stop('os_identity')
        emit({'event': 'host_identity', 'image': image,
              'python': list(sys.version_info[:3]), 'system': platform.system(),
              'architecture': platform.machine(), 'os_version': version,
              'host_instance_observation_only': True, 'product_acceptance': False})
        counts = observe(ROOT, emit)
        return 2 if counts['undisclosed_names'] or counts['undisclosed_targets'] else 0
    except Stop as error:
        try:
            emit({'event': 'stdlib_metadata_hold', 'reason': error.args[0], 'product_acceptance': False})
        except Stop:
            pass
        return 2
    except OSError as error:
        emit({'event': 'stdlib_metadata_hold', 'reason': 'os_error', 'errno': error.errno,
              'product_acceptance': False})
        return 2
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
