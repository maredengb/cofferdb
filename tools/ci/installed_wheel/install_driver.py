"""Install an exact local wheel offline, verify installed bytes, then call a fixed driver."""
import argparse
import base64
import copy
import csv
from email.parser import BytesParser
import hashlib
import importlib
from importlib.machinery import PathFinder
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
import sysconfig
import time
import types
import zipfile

BRIDGE_SHA = 'e4a4978671957b18dc92fbd1cd8ea1d928ff46db2b88c36062d8b80f231fc4c4'
MAX_FILE = 128 * 1024 ** 2


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def load_bridge(filename):
    p = Path(filename)
    need(p.is_absolute() and str(p) == filename and '..' not in p.parts
         and not any(x.is_symlink() for x in (p, *p.parents)), 'Literal fixed bridge path')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 131072
             and not info.st_mode & 0o022, 'Protected bridge source')
        raw = os.read(fd, 131073)
        need(len(raw) == info.st_size and sha(raw) == BRIDGE_SHA, 'Pinned bridge source')
    finally:
        os.close(fd)
    module = types.ModuleType('install_driver_bridge'); module.__file__ = filename
    exec(compile(raw, filename, 'exec', dont_inherit=True), module.__dict__)
    return module


def member(name):
    p = PurePosixPath(name)
    need(type(name) is str and name and not name.startswith('/') and '\\' not in name
         and str(p) == name and '..' not in p.parts and all(32 <= ord(x) < 127 for x in name),
         'Unsafe wheel or record path')
    return name


def wheel_payload(raw, wheel_sha, native_sha, source_raw, source_input_sha, sdk_version, engine_version, tag):
    """Verify the actual immutable payload before invoking any installer or SDK."""
    need(sha(raw) == wheel_sha, 'Wheel digest differs')
    source = json.loads(source_raw)
    need(source.get('source_input_sha256') == source_input_sha and source.get('sdk_version') == sdk_version
         and source.get('engine_version') == engine_version, 'Source metadata differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(len(infos) <= 2048 and len({x.filename for x in infos}) == len(infos)
             and sum(x.file_size for x in infos) <= MAX_FILE, 'Bounded unique wheel members')
        files = {}
        for info in infos:
            name = info.filename
            if info.is_dir():
                member(name.rstrip('/')); continue
            member(name)
            mode = info.external_attr >> 16
            need(not stat.S_ISLNK(mode) and (stat.S_IFMT(mode) in (0, stat.S_IFREG))
                 and not name.endswith(('.pth', '.pyc')) and not name.startswith(('bin/', 'scripts/')),
                 'Unexpected executable startup or wheel member type')
            files[name] = archive.read(info)
    dist = 'cofferdb-' + sdk_version + '.dist-info/'
    need(all(name.startswith(('cofferdb/', dist)) for name in files), 'Wheel has unrelated or relocated payload')
    need({'cofferdb/__init__.py', 'cofferdb/_source.json', dist + 'METADATA', dist + 'WHEEL', dist + 'RECORD'} <= set(files),
         'Complete Python/distribution/source payload required')
    need(files['cofferdb/_source.json'] == source_raw, 'Wheel source bytes differ from prepared source')
    natives = [name for name in files if name.endswith('.so')]
    need(len(natives) == 1 and natives[0].startswith('cofferdb/_native.')
         and sha(files[natives[0]]) == native_sha, 'Native payload differs')
    metadata = BytesParser().parsebytes(files[dist + 'METADATA'])
    wheel = BytesParser().parsebytes(files[dist + 'WHEEL'])
    need(metadata.get_all('Name') == ['cofferdb'] and metadata.get_all('Version') == [sdk_version]
         and not metadata.get_all('Requires-Dist') and wheel.get_all('Tag') == [tag], 'Distribution identity/tag/dependencies differ')
    rows = list(csv.reader(io.StringIO(files[dist + 'RECORD'].decode('utf-8'))))
    need(all(len(row) == 3 for row in rows) and len({row[0] for row in rows}) == len(rows)
         and {row[0] for row in rows} == set(files), 'Wheel RECORD set differs')
    for name, digest, size in rows:
        if name == dist + 'RECORD':
            need(digest == size == '', 'Wheel RECORD self entry differs')
        else:
            expected = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b'=').decode()
            need(digest == expected and size == str(len(files[name])), 'Wheel RECORD payload differs')
    return {'files': files, 'dist': dist, 'native': natives[0], 'source': source}


def file_set(directory, limit=2048):
    """Read bounded names without following directory or file links."""
    root = Path(directory); pending = [(root, 0)]; names = set(); entries = 0
    while pending:
        current, depth = pending.pop(); need(depth <= 32, 'Installed directory depth bound')
        with os.scandir(current) as iterator:
            for item in iterator:
                entries += 1; need(entries <= limit, 'Installed entry bound')
                need(not item.is_symlink(), 'Installed symbolic link')
                info = item.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append((Path(item.path), depth + 1))
                else:
                    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Installed regular files only')
                    names.add(Path(item.path).relative_to(root).as_posix())
                need(len(names) + len(pending) <= limit, 'Installed member bound')
    return names


def installed_identity(b, site, payload, wheel_path, wheel_sha):
    """Compare actual files and installer RECORD before importing cofferdb."""
    site = b.path(str(site)); dist = payload['dist']; files = payload['files']
    pkg_names = {'cofferdb/' + n for n in file_set(site / 'cofferdb')}
    dist_names = {dist + n for n in file_set(site / dist.rstrip('/'))}
    extras = {dist + name for name in ('INSTALLER', 'REQUESTED', 'direct_url.json')}
    need(pkg_names | dist_names == set(files) | extras, 'Installed package/metadata member set differs')
    actual = {}; total = 0
    for name in sorted(pkg_names | dist_names):
        bound = len(files[name]) if name in files and name != dist + 'RECORD' else 1024 * 1024
        actual[name] = b.read_file(str(site / name), bound)
        total += len(actual[name]); need(total <= MAX_FILE + 1024 * 1024, 'Installed total byte bound')
    record = dist + 'RECORD'
    for name, data in files.items():
        if name != record:
            need(actual[name] == data, 'Installed bytes differ from wheel: ' + name)
    need(actual[dist + 'INSTALLER'] == b'pip\n' and actual[dist + 'REQUESTED'] == b'', 'Installer metadata differs')
    direct = b.decode(actual[dist + 'direct_url.json'])
    need(type(direct) is dict and set(direct) == {'url', 'archive_info'}
         and direct['url'] == Path(wheel_path).as_uri() and type(direct['archive_info']) is dict,
         'Installed local wheel origin differs')
    archive = direct['archive_info']
    need(set(archive) in ({'hash', 'hashes'}, {'hashes'}) and archive['hashes'] == {'sha256': wheel_sha}
         and archive.get('hash', 'sha256=' + wheel_sha) == 'sha256=' + wheel_sha, 'Installer archive digest differs')
    rows = list(csv.reader(io.StringIO(actual[record].decode('utf-8'))))
    need(all(len(row) == 3 for row in rows) and len({row[0] for row in rows}) == len(rows)
         and {row[0] for row in rows} == set(actual), 'Installed RECORD complete set differs')
    for name, digest, size in rows:
        member(name)
        if name == record:
            need(digest == size == '', 'Installed RECORD self entry differs')
        else:
            expected = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(actual[name]).digest()).rstrip(b'=').decode()
            need(digest == expected and size == str(len(actual[name])), 'Installed RECORD digest or size differs')
    expected_dist = {name: sha(data) for name, data in files.items() if name.startswith(dist) and name != record}
    observed_dist = {name: sha(actual[name]) for name in expected_dist}
    need(expected_dist == observed_dist, 'Installed immutable distribution payload differs')
    identity = {
        'wheel': {'reference': str(wheel_path), 'expected_sha256': wheel_sha, 'observed_sha256': wheel_sha},
        'native': {'reference': str(site / payload['native']), 'expected_sha256': sha(files[payload['native']]), 'observed_sha256': sha(actual[payload['native']])},
        'source': {'reference': str(site / 'cofferdb/_source.json'), 'expected_sha256': sha(files['cofferdb/_source.json']), 'observed_sha256': sha(actual['cofferdb/_source.json'])},
        'dist_info': {'reference': str(site / dist.rstrip('/')) + ' (immutable wheel metadata inventory; RECORD checked separately)',
                      'expected_sha256': sha(canonical(expected_dist)), 'observed_sha256': sha(canonical(observed_dist))}}
    return {'identity': identity, 'installed_record_sha256': sha(actual[record]), 'immutable_distribution_files': expected_dist,
            'installed_files_sha256': {name: sha(data) for name, data in actual.items()}}


def runtime(venv):
    root = Path(venv)
    need(sys.implementation.name == 'cpython' and sys.version_info.major == 3 and sys.version_info.minor in (12, 13, 14)
         and not sysconfig.get_config_var('Py_GIL_DISABLED') and sys.maxsize > 2 ** 32, 'Ordinary supported CPython required')
    need(sys.flags.isolated == 1 and sys.dont_write_bytecode and Path(sys.prefix) == root
         and sys.prefix != sys.base_prefix, 'Selected isolated new venv required')
    pure = Path(sysconfig.get_path('purelib')); plat = Path(sysconfig.get_path('platlib'))
    need(pure == plat and root in pure.parents and pure.is_dir(), 'Selected venv site-packages required')
    need(not any(p.is_symlink() for p in (root, pure, *pure.parents)), 'Literal venv installation path')
    pair = (platform.system(), platform.machine())
    if pair == ('Darwin', 'arm64'):
        need(platform.mac_ver()[0].split('.')[0] == '15', 'Actual Mac15 required')
        platform_tag = 'macosx_15_0_arm64'
    else:
        need(pair == ('Linux', 'x86_64') and platform.libc_ver()[0] == 'glibc', 'Native glibc Linux required')
        platform_tag = 'manylinux_2_28_x86_64'
    return pure, 'cp3%d-cp3%d-%s' % (sys.version_info.minor, sys.version_info.minor, platform_tag)


def save(path, value):
    raw = canonical(value)
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


PIP_NAME = 'pip-25.3-py3-none-any.whl'
PIP_SHA = '9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd'
PIP_BYTES = 1778622
PIP_PROBE = 'import json,pip;print(json.dumps({"version":pip.__version__,"path":pip.__file__},sort_keys=True))'


def pip_identity(b, site, raw):
    need(len(raw) == PIP_BYTES and sha(raw) == PIP_SHA, 'Fixed pip 25.3 wheel identity')
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) <= 2048 and sum(x.file_size for x in archive.infolist()) <= 64 * 1024 ** 2,
             'Bounded pip archive')
        for name in names:
            member(name)
            need(name.startswith(('pip/', 'pip-25.3.dist-info/')), 'Unexpected pip member root')
            if name.endswith('/') or name == 'pip-25.3.dist-info/RECORD': continue
            expected = archive.read(name)
            actual = b.read_file(str(site / name), 16 * 1024 ** 2)
            need(actual == expected, 'Installed pip member differs from fixed wheel')
            files[name] = sha(actual)
    need('pip/__init__.py' in files and 'pip-25.3.dist-info/METADATA' in files, 'Complete installed pip package')
    need(sorted(x.name for x in site.glob('pip-*.dist-info')) == ['pip-25.3.dist-info'], 'Ambiguous pip distribution')
    return {'version': '25.3', 'wheel_sha256': PIP_SHA, 'installed_members_sha256': sha(canonical(files))}


def install_commands(python, wheel, pip_wheel):
    pip_install = [python, '-I', '-B', '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps',
                   '--no-compile', '--only-binary=:all:', '--disable-pip-version-check', '--no-input', '--no-cache-dir']
    return [[python, '-I', '-B', '-m', 'ensurepip', '--default-pip'],
            [*pip_install, '--force-reinstall', pip_wheel],
            [python, '-I', '-B', '-c', PIP_PROBE],
            [*pip_install, wheel]]


def execute(args):
    started = time.monotonic(); deadline = started + 240
    b = load_bridge(args.runtime_bridge)
    for value in (args.wheel_sha256, args.native_sha256, args.source_input_sha256, args.driver_sha256, args.context_sha256):
        b.checksum(value)
    for value in (args.sdk_version, args.engine_version):
        need(value == '0.1.1', 'Selected version differs')
    site, tag = runtime(args.venv)
    need(not list(site.iterdir()), 'Original runtime must have empty site-packages')
    wheel = b.path(args.wheel); work = b.path(args.work); state = work.with_name(work.name + '-installation')
    need(wheel.name == 'cofferdb-' + args.sdk_version + '-' + tag + '.whl', 'Wheel filename differs from actual runtime')
    need(work.parent.is_dir() and not work.exists() and not state.exists(), 'New installation/driver output required')
    source_raw = b.read_file(args.source_metadata, b.MAX_JSON); wheel_raw = b.read_file(args.wheel)
    pip_raw = b.read_file(args.pip_wheel, PIP_BYTES)
    need(b.path(args.pip_wheel).name == PIP_NAME and len(pip_raw) == PIP_BYTES and sha(pip_raw) == PIP_SHA, 'Fixed pip 25.3 wheel identity')
    driver_raw = b.read_file(args.driver, 131072); context_raw = b.read_file(args.context, b.MAX_JSON)
    need(sha(driver_raw) == args.driver_sha256 and sha(context_raw) == args.context_sha256, 'External driver/context differs')
    payload = wheel_payload(wheel_raw, args.wheel_sha256, args.native_sha256, source_raw,
                            args.source_input_sha256, args.sdk_version, args.engine_version, tag)
    context = b.decode(context_raw)
    fields = {'attempt_id', 'cell_id', 'authorization_reference', 'preflight_evidence', 'budget_reference', 'files_sha256'}
    need(type(context) is dict and set(context) == fields and type(context['files_sha256']) is dict,
         'Exact caller driver context required')
    need(all(type(context[key]) is str and context[key].strip() for key in fields - {'files_sha256'}), 'Actual caller context required')
    for name, digest in context['files_sha256'].items(): member(name); b.checksum(digest)
    inputs = [b.path(x) for x in (args.wheel, args.source_metadata, args.driver, args.context, args.runtime_bridge, args.fixtures, args.pip_wheel)]
    need(all(p != root and root not in p.parents and p not in root.parents for p in inputs for root in (work, state, b.path(args.venv))),
         'Mutable outputs overlap frozen inputs')
    need(not any(p.is_symlink() for p in (work.parent, *work.parent.parents)), 'Literal output parent')
    state.mkdir(mode=0o700)
    save(state / 'INTENT.json', {'wheel_sha256': args.wheel_sha256, 'driver_sha256': args.driver_sha256,
         'context_sha256': args.context_sha256, 'pip_wheel_sha256': PIP_SHA, 'started_monotonic': started, 'deadline_monotonic': deadline,
         'product_acceptance': False})
    def fresh():
        for p, digest, bound in [(args.wheel, args.wheel_sha256, MAX_FILE), (args.driver, args.driver_sha256, 131072),
                                 (args.context, args.context_sha256, b.MAX_JSON), (args.source_metadata, sha(source_raw), b.MAX_JSON),
                                 (args.runtime_bridge, BRIDGE_SHA, 131072), (args.pip_wheel, PIP_SHA, PIP_BYTES)]:
            need(sha(b.read_file(p, bound)) == digest, 'Frozen installation input changed')
        need(time.monotonic() + 2 < deadline, 'Installation/driver budget exhausted')
    try:
        installer = None
        for index, argv in enumerate(install_commands(sys.executable, args.wheel, args.pip_wheel)):
            fresh()
            if index == 3:
                need(installer is not None and pip_identity(b, site, pip_raw) == installer, 'Fixed installer required before SDK command')
                fresh()
            save(state / ('command-%d.json' % index), {'argv': argv, 'cwd': str(state)})
            # Preserve the enclosing process group: no shell, detaching or new session.
            with (state / ('command-%d.stdout.raw' % index)).open('xb') as stdout, (state / ('command-%d.stderr.raw' % index)).open('xb') as stderr:
                remaining = deadline - time.monotonic() - 2
                need(remaining > 0, 'Installation deadline before process launch')
                completed = subprocess.run(argv, cwd=state, stdout=stdout, stderr=stderr, check=False,
                                           timeout=remaining)
            save(state / ('command-%d.exit.json' % index), {'returncode': completed.returncode})
            need(completed.returncode == 0, 'Offline installer failed')
            if index == 2:
                observed = b.decode(b.read_file(str(state / 'command-2.stdout.raw'), 65536))
                need(observed == {'version': '25.3', 'path': str(site / 'pip/__init__.py')}, 'Actual installer version/path differs')
                installer = pip_identity(b, site, pip_raw)
                save(state / 'PIP-IDENTITY.json', installer)
        fresh(); actual = installed_identity(b, site, payload, args.wheel, args.wheel_sha256)
        spec = PathFinder.find_spec('cofferdb', sys.path)
        need(spec is not None and spec.origin == str(site / 'cofferdb/__init__.py')
             and not any(name == 'cofferdb' or name.startswith('cofferdb.') for name in sys.modules), 'Actual import origin differs or SDK already loaded')
        contract = {'schema_version': 1, **copy.deepcopy(context), 'sdk_version': args.sdk_version,
                    'engine_version': args.engine_version, 'identity': actual['identity']}
        contract_path = state / 'driver-contract.json'; save(contract_path, contract)
        save(state / 'INSTALLED-IDENTITY.json', actual)
        fresh()
        package = importlib.import_module('cofferdb'); native = importlib.import_module('cofferdb._native')
        need(package.__file__ == str(site / 'cofferdb/__init__.py') and native.__file__ == str(site / payload['native'])
             and package.__version__ == args.sdk_version and package.engine_version == args.engine_version, 'Loaded SDK/native/version differs')
        need(installed_identity(b, site, payload, args.wheel, args.wheel_sha256) == actual, 'Installed files changed at import')
        fresh()
        driver = types.ModuleType('selected_installed_package_driver'); driver.__file__ = args.driver
        code = compile(driver_raw, args.driver, 'exec', dont_inherit=True)
        fresh(); exec(code, driver.__dict__); fresh()
        previous = sys.argv
        try:
            sys.argv = [args.driver, '--fixtures', args.fixtures, '--contract', str(contract_path),
                        '--contract-sha256', sha(canonical(contract)), '--work', args.work]
            status = driver.main()
        finally:
            sys.argv = previous
        need(type(status) is int and status == 0, 'External business driver failed')
        fresh(); need(installed_identity(b, site, payload, args.wheel, args.wheel_sha256) == actual, 'Installed files changed after driver')
        result = {'status': 'EXACT_OFFLINE_INSTALL_AND_DRIVER_COMPLETED', 'product_acceptance': False,
                  'identity': actual, 'installer': installer, 'driver_sha256': args.driver_sha256, 'driver_contract_sha256': sha(canonical(contract)),
                  'completion_requires_successful_outer_capture': True}
        save(state / 'RESULT.json', result); need(time.monotonic() < deadline, 'Final installation deadline')
        return result
    except BaseException as error:
        try: save(state / 'HOLD.json', {'status': 'HOLD_INSTALLED_WHEEL_DRIVER', 'error_type': type(error).__name__, 'product_acceptance': False})
        except BaseException: pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['verify', 'run'], default='verify')
    for key in ('runtime-bridge', 'wheel', 'wheel-sha256', 'native-sha256', 'source-metadata', 'source-input-sha256',
                'sdk-version', 'engine-version', 'pip-wheel', 'venv', 'driver', 'driver-sha256', 'context', 'context-sha256', 'fixtures', 'work'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.action == 'verify':
        print('{"status":"INSTALLED_WHEEL_DRIVER_NOT_EXECUTED","product_acceptance":false}'); return 0
    print(canonical(execute(args)).decode(), end=''); return 0


if __name__ == '__main__':
    raise SystemExit(main())
