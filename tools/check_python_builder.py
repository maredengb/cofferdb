"""Check source and wheel rejection paths without loading the native extension.

Supply a prepared source directory and one actual matching wheel. The command
retains malformed copies and its checks in a new external output directory.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import zipfile

import build_python as builder


def check(source, wheel, output, python):
    source = source.resolve(strict=True)
    wheel = wheel.resolve(strict=True)
    output = output.absolute()
    if (output.exists() or output.is_symlink() or output.resolve().is_relative_to(source)
            or wheel.is_relative_to(output.resolve())):
        raise ValueError('checks require a new output directory outside the source and input wheel')
    inputs, metadata = builder.source_inputs(source)
    runtime = builder.interpreter(python)
    original = wheel.read_bytes()
    output.mkdir(parents=True)
    records = []

    def record(value):
        records.append(value)
        (output / 'checks.json').write_text(json.dumps({
            'kind':'source_and_artifact_controls_without_engine_execution',
            'builder_sha256':builder.sha(Path(builder.__file__).read_bytes()),
            'checker_sha256':builder.sha(Path(__file__).read_bytes()),
            'source_input_sha256':metadata['source_input_sha256'],
            'wheel_sha256':builder.sha(original), 'runtime':runtime, 'checks':records,
        }, indent=2) + '\n')

    def refusal(name, action):
        try:
            action()
        except (ValueError, OSError) as error:
            record({'case':name, 'refused':True, 'error_type':type(error).__name__, 'message':str(error)})
        else:
            record({'case':name, 'refused':False})
            raise AssertionError('malformed input unexpectedly accepted: ' + name)

    clean = output / 'clean'
    clean.mkdir()
    result = builder.artifact(wheel, metadata, runtime, clean, [], inputs)
    record({'case':'original-actual-wheel', 'accepted':True, 'artifact':result})
    with zipfile.ZipFile(io.BytesIO(original)) as bundle:
        contents = {name:bundle.read(name) for name in bundle.namelist() if not name.endswith('/')}
    record_name = next(name for name in contents if name.endswith('.dist-info/RECORD'))
    native_name = 'cofferdb/_native' + runtime['extension_suffix']

    def artifact_case(name, mutation, resign=True):
        case = output / name
        case.mkdir()
        data = dict(contents)
        mutation(data)
        if resign:
            stream = io.StringIO()
            writer = csv.writer(stream, lineterminator='\n')
            for path, payload in data.items():
                digest = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode()
                writer.writerow([path, '', ''] if path == record_name else [path, digest, str(len(payload))])
            if record_name in data:
                data[record_name] = stream.getvalue().encode()
        target = case / wheel.name
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path, payload in data.items():
                bundle.writestr(path, payload)
        refusal(name, lambda:builder.artifact(target, metadata, runtime, case, [], inputs))

    # Every regular member is necessary: source modules, typing, generated
    # identity, native module, distribution metadata and each original notice.
    for index, name in enumerate(sorted(contents)):
        artifact_case(f'missing-member-{index:03d}', lambda data, name=name:data.pop(name))
    artifact_case('renamed-native', lambda data:data.__setitem__('cofferdb/wrong_native.so', data.pop(native_name)))
    artifact_case('wrong-native-abi', lambda data:data.__setitem__('cofferdb/_native.invalid.so', data.pop(native_name)))
    artifact_case('native-outside-package', lambda data:data.__setitem__('wrong_native.so', data.pop(native_name)))
    artifact_case('extra-root-module', lambda data:data.__setitem__('unlisted.py', b'# unexpected package member\n'))
    artifact_case('extra-package-module', lambda data:data.__setitem__('cofferdb/unlisted.py', b'# unexpected package member\n'))
    artifact_case('changed-python-resigned', lambda data:data.__setitem__('cofferdb/types.py', data['cofferdb/types.py'] + b'\n# changed\n'))
    artifact_case('changed-python-stale-record', lambda data:data.__setitem__('cofferdb/types.py', data['cofferdb/types.py'] + b'\n# changed\n'), False)
    license_name = next(name for name in contents if name.endswith('.dist-info/licenses/LICENSE'))
    artifact_case('changed-source-license', lambda data:data.__setitem__(license_name, data[license_name] + b'\nchanged\n'))

    for name, filename in (
        ('filename-wrong-version', wheel.name.replace('cofferdb-'+metadata['sdk_version']+'-', 'cofferdb-9.9.9-')),
        ('filename-wrong-distribution', wheel.name.replace('cofferdb-', 'different-', 1)),
        ('filename-extra-prefix', 'extra-' + wheel.name),
    ):
        case = output / name
        case.mkdir()
        target = case / filename
        target.write_bytes(original)
        refusal(name, lambda target=target, case=case:builder.artifact(target, metadata, runtime, case, [], inputs))

    metadata_name = next(name for name in contents if name.endswith('.dist-info/METADATA'))
    artifact_case('duplicate-version-header', lambda data:data.__setitem__(
        metadata_name, b'Version: 9.9.9\n' + data[metadata_name]))
    artifact_case('duplicate-name-header', lambda data:data.__setitem__(
        metadata_name, b'Name: different\n' + data[metadata_name]))

    if runtime['system'] == 'Darwin':
        def changed_target(data, field, value):
            native = bytearray(data[native_name])
            if struct.unpack_from('<I', native, 0)[0] != 0xfeedfacf:
                raise ValueError('the control requires the actual thin arm64 Mach-O')
            count = struct.unpack_from('<I', native, 16)[0]
            offset, changed = 32, 0
            for _ in range(count):
                command, size = struct.unpack_from('<II', native, offset)
                if size < 8 or offset + size > len(native):
                    raise ValueError('invalid Mach-O load command in the control input')
                if command == 0x32:
                    assert struct.unpack_from('<I', native, offset+8)[0] == 1
                    assert struct.unpack_from('<I', native, offset+12)[0] == 15 << 16
                    struct.pack_into('<I', native, offset+field, value)
                    changed += 1
                offset += size
            assert changed == 1
            data[native_name] = bytes(native)

        # Only inspect these copies; never import or execute patched binaries.
        for name, version in [('minos-15-0-1', (15 << 16) | 1),
                              ('minos-15-1', (15 << 16) | (1 << 8)),
                              ('minos-14-6', (14 << 16) | (6 << 8)),
                              ('minos-16-0', 16 << 16)]:
            artifact_case(name, lambda data, version=version:changed_target(data, 12, version))
        for name, platform in [('platform-ios', 2), ('platform-catalyst', 6),
                               ('platform-ios-simulator', 7), ('platform-unknown', 0)]:
            artifact_case(name, lambda data, platform=platform:changed_target(data, 8, platform))
        artifact_case('build-sdk-unavailable', lambda data:changed_target(data, 16, 0))

        def changed_header(data, offset, value):
            native = bytearray(data[native_name])
            struct.pack_into('<I', native, offset, value)
            data[native_name] = bytes(native)

        for name, offset, value in [('header-executable', 12, 2),
                                    ('header-wrong-cpu', 4, 0x01000007),
                                    ('header-too-many-commands', 16, 0xffffffff),
                                    ('header-truncated-command-range', 20, 0xffffffff),
                                    ('header-empty-command-range', 20, 0)]:
            artifact_case(name, lambda data, offset=offset, value=value:changed_header(data, offset, value))

    for name in ('unlisted-cargo-config', 'unlisted-rust-source', 'unlisted-symlink', 'changed-source-mode'):
        copied = output / name
        shutil.copytree(source, copied)
        if name == 'unlisted-cargo-config':
            path = copied / '.cargo/config.toml'
            path.parent.mkdir(exist_ok=True)
            path.write_text('[profile.release]\nopt-level = 2\ncodegen-units = 1\n')
        elif name == 'unlisted-rust-source':
            (copied / 'unlisted.rs').write_text('// unregistered build input\n')
        elif name == 'unlisted-symlink':
            (copied / 'unlisted-link').symlink_to(source, target_is_directory=True)
        else:
            path = copied / 'Cargo.toml'
            path.chmod(path.stat().st_mode ^ 0o100)
        refusal(name, lambda copied=copied:builder.source_inputs(copied))

    if builder.source_inputs(source) != (inputs, metadata) or wheel.read_bytes() != original:
        raise AssertionError('an original check input changed')
    return {'checks':len(records), 'engine_loaded':False, 'output':str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.wheel, args.output, args.python)))
