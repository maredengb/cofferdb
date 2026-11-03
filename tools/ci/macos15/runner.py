"""Frozen macOS15 runtime sandbox renderer; default validation never launches anything.

Explicit run is still subject to separate approval. Capture is delegated to the
existing pinned q_process_supervisor, not a replacement process runner.
"""
import argparse
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import stat
import time

SUPERVISOR_SHA = '39d6c8842e1a9c972fd911feb4e3c483d1eb6947fc7291b65ac6f8a3bc778e51'
HERE = Path(__file__).absolute().parent
FORBIDDEN = ('/Library/Developer', '/Applications', '/opt/homebrew/Cellar/llvm',
             '/opt/homebrew/Cellar/rust', '/usr/local/lib/rustlib', '/usr/bin', '/bin', '/sbin', '/usr/sbin')
SYSTEM_READ = ('/usr/lib', '/System/Library/Frameworks', '/System/Library/PrivateFrameworks')
RAW_LIMIT = 2 * 1024 * 1024
FIELDS = {'schema_version', 'kind', 'python', 'stdlib', 'dependencies', 'runtime_files', 'venv', 'work',
          'evidence', 'sandbox', 'supervisor', 'probe_sha256', 'compiler', 'driver',
          'network_probe', 'whole_seconds', 'command_seconds'}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False, separators=(',', ':')) + '\n').encode()


def unique(items):
    out = {}
    for key, value in items:
        need(key not in out, 'Duplicate JSON field')
        out[key] = value
    return out


def parsed(raw):
    need(len(raw) <= 128 * 1024, 'Plan byte bound')
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def checksum(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 required')


def path(value):
    need(type(value) is str and '\0' not in value and '\n' not in value and '\r' not in value,
         'Literal path required')
    p = Path(value)
    need(p.is_absolute() and str(p) == value and not value.startswith('//')
         and '..' not in p.parts and value != '/', 'Canonical absolute non-root path')
    return p


def beneath(p, root):
    return p == root or p.is_relative_to(root)


def no_link(p, missing=False):
    for item in reversed((p, *p.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            need(missing, 'Fixed path missing')
            return
        need(not stat.S_ISLNK(info.st_mode), 'Data path symlink')
        if item != p:
            need(stat.S_ISDIR(info.st_mode), 'Ancestor not directory')


def python_provider(value):
    text = str(path(str(value)))
    patterns = (
        r'/Library/Frameworks/Python[.]framework/Versions/3[.](?:12|13|14)/',
        r'/opt/homebrew/Cellar/python@3[.](?:12|13|14)/3[.](?:12|13|14)[.][0-9]+(?:_[0-9]+)?/Frameworks/Python[.]framework/Versions/3[.](?:12|13|14)/',
        r'/Users/runner/hostedtoolcache/Python/3[.](?:12|13|14)[.][0-9]+/arm64/',
    )
    return any(re.match(pattern, text) for pattern in patterns)


def public_native_candidate(p):
    """Return a fixed, non-secret relative label for known Framework aliases."""
    for minor in (12, 13, 14):
        version = f'3.{minor}'
        expected = (Path('/Library/Frameworks/Python.framework/Versions') / version /
                    'lib' / ('python' + version) / f'config-{version}-darwin')
        if p.parent == expected and p.name in (f'libpython{version}.a', f'libpython{version}.dylib'):
            return f'config-{version}-darwin/{p.name}'
    return None


def provider_diagnostic_identity(p):
    """Return a fixed provider label and a hash of its path, never an arbitrary path."""
    text = str(p)
    for minor in (12, 13, 14):
        version = f'3.{minor}'
        framework_root = Path('/Library/Frameworks/Python.framework/Versions') / version
        try:
            rel = p.relative_to(framework_root)
        except ValueError:
            rel = None
        if rel is not None:
            if (rel.parts == ('bin', f'python{version}') or
                    rel.parts == ('Resources', 'Python.app', 'Contents', 'MacOS', 'Python')):
                return f'framework/{version}', f'framework/{version}/' + '/'.join(rel.parts)
            return f'framework/{version}', None
        homebrew = re.match(
            rf'^/opt/homebrew/Cellar/python@{re.escape(version)}/{re.escape(version)}[.][0-9]+(?:_[0-9]+)?/Frameworks/Python[.]framework/Versions/{re.escape(version)}(?:/(.*))?$',
            text)
        if homebrew:
            rel = tuple((homebrew.group(1) or '').split('/'))
            if rel in (('bin', f'python{version}'),
                       ('Resources', 'Python.app', 'Contents', 'MacOS', 'Python')):
                return f'homebrew/{version}', f'homebrew/{version}/' + '/'.join(rel)
            return f'homebrew/{version}', None
        hosted = re.match(
            rf'^/Users/runner/hostedtoolcache/Python/{re.escape(version)}[.][0-9]+/arm64/(.*)$',
            text)
        if hosted:
            return f'hostedtoolcache/{version}/arm64', None
    return None


def native_rejection_detail(p, info):
    if info is None:
        return ''
    identity = provider_diagnostic_identity(p)
    if identity is None:
        return ''
    provider, known_label = identity
    details = {
        'provider': provider,
        'path_sha256': digest(str(p).encode('utf-8')),
        'regular': stat.S_ISREG(info.st_mode),
        'mode': format(stat.S_IMODE(info.st_mode), '04o'),
        'links': info.st_nlink,
        'bytes': info.st_size,
        'owner': 'root' if info.st_uid == 0 else 'current' if info.st_uid == os.getuid() else 'other',
    }
    safe_label = public_native_candidate(p) or known_label
    if safe_label is not None:
        details['known_path'] = safe_label
    if p.suffix.lower() in ('.a', '.dylib', '.so'):
        details['suffix'] = p.suffix.lower()
    else:
        details['suffix'] = 'none' if not p.suffix else 'other'
    return ' (' + json.dumps(details, sort_keys=True, separators=(',', ':')) + ')'


def host_limits(value, role, private_root=None, file_info=None):
    """Explicit installed-file roles; ordinary evidence never uses this policy."""
    p = path(str(value))
    if role in ('python', 'native'):
        if not python_provider(p):
            return 128 * 1024 ** 2, 0o022
        if role == 'python':
            need(p.name in ('Python', 'python3.12', 'python3.13', 'python3.14'),
                 'Actual provider interpreter required')
        else:
            label = public_native_candidate(p)
            detail = native_rejection_detail(p, file_info)
            if label and not detail:
                detail = f' (fixed Framework candidate: {label})'
            if not (p.name == 'Python' or p.suffix in ('.so', '.dylib')):
                raise ValueError('Declared native file required' + detail)
        return 128 * 1024 ** 2, 0o002
    if role == 'compiler':
        selected = (re.fullmatch(r'/Applications/Xcode[A-Za-z0-9_.-]*[.]app/Contents/Developer/Toolchains/XcodeDefault[.]xctoolchain/usr/bin/clang(?:-[0-9]+)?', str(p))
             or re.fullmatch(r'/Library/Developer/CommandLineTools/usr/bin/clang(?:-[0-9]+)?', str(p)))
        return (512 if selected else 128) * 1024 ** 2, 0o022
    need(role == 'private_python' and private_root is not None, 'Explicit installed-file role required')
    root = path(private_root); no_link(root); info = root.stat()
    need(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700
         and p.parent == root / 'bin' and p.name in ('python', 'python3', 'python3.12', 'python3.13', 'python3.14'),
         'Copied Python must belong to bound private0700 venv')
    return 128 * 1024 ** 2, 0o002


def host_protection(value, info, role, private_root=None):
    bound, forbidden_mode = host_limits(value, role, private_root, info)
    checks = {'regular': stat.S_ISREG(info.st_mode), 'single_link': info.st_nlink == 1,
              'within_size': info.st_size <= bound,
              'protected_mode': not bool(info.st_mode & (forbidden_mode | 0o7000)),
              'owner': info.st_uid in (0, os.getuid())}
    need(all(checks.values()), 'Protected installed-file metadata rejected: ' + json.dumps(
        {'role': role, 'mode': oct(stat.S_IMODE(info.st_mode)), 'links': info.st_nlink,
         'bytes': info.st_size, 'checks': checks}, sort_keys=True))
    return bound


def host_file(value, *, role, private_root=None):
    """Bounded NOFOLLOW read, with fd and named-file identity checked before/after."""
    p = path(value); no_link(p)
    private_signature = None
    if role == 'private_python':
        root_info = path(private_root).stat()
        private_signature = (root_info.st_dev, root_info.st_ino, root_info.st_mode, root_info.st_uid, root_info.st_gid)
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY); opened = None
    try:
        for part in p.parent.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        opened = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(opened); bound = host_protection(p, before, role, private_root)
        if role in ('python', 'private_python', 'compiler'):
            need(os.access(p, os.X_OK), 'Installed executable unavailable')
        signature = lambda x: (x.st_dev, x.st_ino, x.st_mode, x.st_uid, x.st_gid,
                               x.st_nlink, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        chunks, size = [], 0
        while size <= bound:
            block = os.read(opened, min(65536, bound + 1 - size))
            if not block: break
            chunks.append(block); size += len(block)
        need(size == before.st_size and signature(before) == signature(os.fstat(opened))
             and signature(before) == signature(os.stat(p.name, dir_fd=directory, follow_symlinks=False)),
             'Installed file changed during read')
        host_protection(p, os.fstat(opened), role, private_root)
        if private_signature is not None:
            root_info = path(private_root).stat()
            need(private_signature == (root_info.st_dev, root_info.st_ino, root_info.st_mode, root_info.st_uid, root_info.st_gid),
                 'Private venv root changed during read')
        no_link(p)
        need(signature(before) == signature(p.lstat()), 'Installed path changed during read')
        return b''.join(chunks)
    finally:
        if opened is not None: os.close(opened)
        os.close(directory)


def row(value, *, executable=False, host_role=None, private_root=None):
    need(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact file identity row')
    p = path(value['path']); checksum(value['sha256'])
    if host_role is not None:
        need(digest(host_file(str(p), role=host_role, private_root=private_root)) == value['sha256'], 'File bytes changed')
        return p
    no_link(p)
    info = p.stat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 128 * 1024 * 1024
         and not info.st_mode & 0o022, 'Protected bounded regular input')
    need(digest(p.read_bytes()) == value['sha256'], 'File bytes changed')
    if executable:
        need(os.access(p, os.X_OK), 'Executable unavailable')
    return p


def schema(plan):
    """Structural validation and SBPL safety; no host capability probes or IO."""
    need(type(plan) is dict and set(plan) == FIELDS and type(plan['schema_version']) is int
         and plan['schema_version'] == 1 and plan['kind'] == 'macos15_python_runtime_isolation_1',
         'Exact Mac runtime schema')
    for key in ('sandbox', 'supervisor', 'compiler'):
        need(type(plan[key]) is dict and set(plan[key]) == {'path', 'sha256'}, 'Identity row')
        path(plan[key]['path']); checksum(plan[key]['sha256'])
    need(plan['sandbox']['path'] == '/usr/bin/sandbox-exec', 'Only system sandbox-exec')
    need(plan['supervisor']['sha256'] == SUPERVISOR_SHA, 'Reviewed supervisor identity required')
    p = plan['python']
    need(type(p) is dict and set(p) == {'invocation', 'real', 'base', 'sha256', 'minor'}
         and type(p['minor']) is int and p['minor'] in (12, 13, 14), 'Python identity/minor row')
    invocation, real, base = path(p['invocation']), path(p['real']), path(p['base']); checksum(p['sha256'])
    venv, work, evidence = [path(plan[k]) for k in ('venv', 'work', 'evidence')]
    need(beneath(invocation, venv), 'Runtime invocation must belong to selected new venv')
    need(all(not beneath(a, b) and not beneath(b, a) for a, b in
             [(venv, work), (venv, evidence), (work, evidence)]), 'Separate venv/work/evidence')
    need(type(plan['dependencies']) is list and len(plan['dependencies']) <= 32,
         'Bounded exact dependency read roots')
    need(type(plan['runtime_files']) is list and len(plan['runtime_files']) <= 64,
         'Bounded exact runtime shared-library files')
    for value in plan['runtime_files']:
        need(type(value) is dict and set(value) == {'path', 'sha256'}, 'Runtime file identity')
        path(value['path']); checksum(value['sha256'])
    roots = [path(plan['stdlib'])] + [path(v) for v in plan['dependencies']]
    need(len(set(roots)) == len(roots), 'Duplicate read root')
    forbidden = [Path(v) for v in FORBIDDEN]
    compiler = path(plan['compiler']['path'])
    tool_parts = {'.cargo', '.rustup', 'rustlib', 'SDKs', 'CommandLineTools', 'Xcode.app', 'llvm'}
    for root in roots:
        need(len(root.parts) >= 4 and not any(beneath(root, deny) or beneath(deny, root)
                 for deny in forbidden) and not beneath(compiler, root), 'Broad/compiler read root refused')
        need(not tool_parts.intersection(root.parts)
             and str(root) not in {'/opt/homebrew/Cellar', '/usr/local/Cellar', '/Users', '/private'},
             'Toolchain/cache or package-manager aggregate read root refused')
        need(not any(beneath(a, root) or beneath(root, a) for a in [work, evidence]),
             'Read roots must not overlap work/evidence')
    for value in [real, base, *[path(v['path']) for v in plan['runtime_files']]]:
        need(value != compiler and not any(beneath(value, deny) for deny in forbidden)
             and not tool_parts.intersection(value.parts), 'Runtime file cannot grant toolchain access')
    need(not any(beneath(compiler, a) or beneath(base, a) for a in [venv, work, evidence])
         and not any(beneath(real, a) for a in [work, evidence])
         and not any(beneath(compiler, Path(s)) for s in SYSTEM_READ), 'Compiler outside allowed runtime')
    need(not any(beneath(compiler, Path(v)) for v in ['/dev', '/private/tmp']),
         'Actual fixed compiler location required')
    d = plan['driver']
    need(type(d) is dict and set(d) == {'path', 'sha256', 'arguments', 'inputs', 'no_detach_reviewed'},
         'External reviewed driver contract')
    path(d['path']); checksum(d['sha256'])
    need(d['no_detach_reviewed'] is True, 'External driver must be reviewed against detaching')
    need(type(d['arguments']) is list and all(type(a) is str and '\0' not in a
         for a in d['arguments']) and sum(len(a.encode()) for a in d['arguments']) < 32768,
         'Bounded literal driver arguments')
    need(type(d['inputs']) is list and len(d['inputs']) <= 64, 'Bounded frozen wheel/smoke inputs')
    for value in d['inputs']:
        need(type(value) is dict and set(value) == {'path', 'sha256'}, 'External input identity')
        path(value['path']); checksum(value['sha256'])
    for value in [path(d['path']), *[path(v['path']) for v in d['inputs']]]:
        need(not any(beneath(value, deny) for deny in forbidden)
             and not tool_parts.intersection(value.parts) and value != compiler,
             'External driver/data cannot grant compiler/SDK reads')
    immutable_files = [path(plan['supervisor']['path']), HERE / 'runner.py', HERE / 'probe.py',
                       path(d['path']), *[path(v['path']) for v in d['inputs']],
                       *[path(v['path']) for v in plan['runtime_files']]]
    need(all(not beneath(value, mutable) for value in immutable_files for mutable in [venv, work]),
         'Trusted maintenance/driver/data inputs cannot reside in writable runtime roots')
    checksum(plan['probe_sha256'])
    net = plan['network_probe']
    need(type(net) is dict and set(net) == {'address', 'port'}
         and type(net['port']) is int and 1 <= net['port'] <= 65535
         and type(net['address']) is str, 'Network probe target')
    need(ipaddress.IPv4Address(net['address']).is_global, 'Fixed numeric external IPv4 probe')
    need(type(plan['whole_seconds']) is int and plan['whole_seconds'] == 300
         and type(plan['command_seconds']) is int and 5 <= plan['command_seconds']
         <= plan['whole_seconds'], 'Explicit finite local deadlines')
    return plan


def sbpl(value):
    # JSON string escaping is a quoted SBPL string for these literal paths;
    # backslash/control/nonASCII rejected to avoid parser interpretation gaps.
    need(all(32 <= ord(c) < 127 and c not in '\\' for c in value), 'ASCII literal SBPL paths')
    return json.dumps(value)


def profile(plan):
    schema(plan)
    read_files = [plan['python']['invocation'], plan['python']['real'], str(HERE / 'probe.py'),
                  '/System/Library/CoreServices/SystemVersion.plist',
                  plan['driver']['path']] + [r['path'] for r in plan['driver']['inputs']] + [
                  r['path'] for r in plan['runtime_files']]
    read_roots = list(SYSTEM_READ) + [plan['stdlib']] + plan['dependencies'] + [plan['venv']]
    ancestors = sorted({str(parent) for v in read_files + read_roots + [plan['work']]
                        for parent in Path(v).parents})
    lines = ['(version 1)', '(deny default)', '(deny network*)', '(allow process-fork)',
             '(allow sysctl-read)', '(allow file-read-metadata ' + ' '.join(
                '(literal ' + sbpl(v) + ')' for v in ancestors) + ')',
             '(allow process-exec ' + ' '.join('(literal ' + sbpl(v) + ')'
               for v in sorted(set([plan['python']['invocation'], plan['python']['real']]))) + ')',
             '(allow file-read* ' + ' '.join('(literal ' + sbpl(v) + ')'
               for v in sorted(set(read_files + ['/dev/null', '/dev/urandom', '/dev/random'])) ) + ')',
             '(allow file-read* ' + ' '.join('(subpath ' + sbpl(v) + ')'
               for v in sorted(set(read_roots + [plan['work']]))) + ')',
             '(allow file-write* ' + ' '.join('(subpath ' + sbpl(plan[k]) + ')'
               for k in ['venv', 'work']) + ')',
             '(deny file-read* ' + ' '.join('(subpath ' + sbpl(v) + ')'
               for v in FORBIDDEN) + ' (literal ' + sbpl(plan['compiler']['path']) + '))']
    return '\n'.join(lines) + '\n'


def validate(plan_path, expected):
    checksum(expected); p = path(plan_path); no_link(p)
    raw = p.read_bytes(); need(digest(raw) == expected, 'Frozen plan bytes differ')
    plan = schema(parsed(raw)); profile(plan)
    need(not any(beneath(p, path(plan[k])) for k in ['venv', 'work']), 'Plan cannot be runtime-writable')
    return plan


def actual_inputs(plan):
    # Never called by verify/preview. Existing venv is newly provisioned by an
    # independently approved step; this runtime cannot bootstrap an interpreter.
    for key in ['sandbox', 'supervisor', 'compiler']:
        row(plan[key], executable=key != 'supervisor', host_role='compiler' if key == 'compiler' else None)
    need(Path(plan['python']['invocation']).resolve(strict=True) == path(plan['python']['real']),
         'Venv Python target changed')
    row({'path': plan['python']['real'], 'sha256': plan['python']['sha256']}, executable=True, host_role='private_python', private_root=plan['venv'])
    row({'path': plan['python']['base'], 'sha256': plan['python']['sha256']}, executable=True, host_role='python')
    for key in ['venv', 'stdlib']:
        p = path(plan[key]); no_link(p); need(p.is_dir(), 'Runtime read directory missing')
    venv = path(plan['venv']); info = venv.stat()
    need(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700, 'Dedicated private venv')
    for value in plan['dependencies']:
        p = path(value); no_link(p); need(p.is_dir(), 'Dependency read directory missing')
    for value in plan['runtime_files']:
        row(value, host_role='native')
    row({'path': str(HERE / 'probe.py'), 'sha256': plan['probe_sha256']})
    row({k: plan['driver'][k] for k in ['path', 'sha256']})
    for value in plan['driver']['inputs']:
        row(value)


def private_new(value):
    p = path(value); no_link(p, missing=True); need(not os.path.lexists(p), 'New path exists')
    parent = p.parent; info = parent.stat()
    need(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700, 'Owned private parent')
    p.mkdir(mode=0o700)
    return p


def save(p, value):
    raw = canonical(value)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def host_gate():
    version = platform.mac_ver()
    need(platform.system() == 'Darwin' and platform.machine() == 'arm64'
         and version[0].split('.')[0] == '15', 'HOLD_REQUIRES_MACOS15_ARM64')
    info = os.uname()
    image = {key: os.environ.get(key) for key in ['ImageOS', 'ImageVersion', 'RUNNER_ARCH']}
    need(all(value is None or type(value) is str and len(value.encode()) <= 1024
             for value in image.values()), 'Bounded runner image metadata')
    return {'platform_mac_ver': version,
            'os_uname': {key: getattr(info, key) for key in ['sysname', 'nodename', 'release', 'version', 'machine']},
            'runner_image_environment': image}


def load_supervisor(plan):
    p = row(plan['supervisor'], executable=False)
    spec = importlib.util.spec_from_file_location('selected_mac_process_supervisor', p)
    module = importlib.util.module_from_spec(spec)
    # Load only the SHA-bound maintenance supervisor, not SDK/driver code.
    raw = p.read_bytes(); need(digest(raw) == SUPERVISOR_SHA, 'Supervisor changed at load')
    exec(compile(raw, str(p), 'exec', dont_inherit=True), module.__dict__)
    return module


def captured_ok(receipt):
    need(receipt.get('state') == 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE'
         and receipt.get('actual_command_exit_code') == 0
         and type(receipt.get('actual_command_exit_code')) is int
         and receipt.get('process_group_gone') is True
         and type(receipt.get('raw_eof')) is dict
         and set(receipt['raw_eof']) == {'stdout', 'stderr'}
         and all(v is True for v in receipt['raw_eof'].values())
         and receipt.get('unread_tail_possible') is False
         and receipt.get('failure_reason') is None, 'HOLD_CAPTURE_OR_CLEANUP')


def run(plan_path, expected, approved):
    need(approved == expected, 'Separate exact plan approval required')
    plan = validate(plan_path, expected)
    host = host_gate()  # On non15, no sandbox command/probe/directory mutation.
    actual_inputs(plan)
    supervisor = load_supervisor(plan)
    started = time.monotonic(); deadline = started + plan['whole_seconds']
    evidence = private_new(plan['evidence'])
    save(evidence / 'INTENT.json', {'plan_sha256': expected, 'started_monotonic': started, 'actual_host': host,
         'deadline_monotonic': deadline, 'no_product_assertions_added': True})
    work = private_new(plan['work'])
    profile_path = evidence / 'runtime.sb'
    fd = os.open(profile_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(profile(plan).encode()); stream.flush(); os.fsync(stream.fileno())
    environment = {'PATH': '/nonexistent', 'LANG': 'C', 'LC_ALL': 'C', 'HOME': str(work),
                   'TMPDIR': str(work), 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
    records = []
    try:
        probe_argv = [plan['python']['invocation'], '-I', '-B', str(HERE / 'probe.py'),
            '--compiler', plan['compiler']['path'], '--address', plan['network_probe']['address'],
            '--port', str(plan['network_probe']['port']), '--expected-minor', str(plan['python']['minor'])]
        driver_argv = [plan['python']['invocation'], '-I', '-B', plan['driver']['path'],
                       *plan['driver']['arguments']]
        for name, target in [('probes', probe_argv), ('install-smoke', driver_argv)]:
            # Every command uses the same byte-identical profile; driver may
            # start only after all three denial probes pass with clean capture.
            actual_inputs(plan)
            need(validate(plan_path, expected) == plan, 'Plan drift')
            need(profile_path.read_text() == profile(plan), 'Profile drift')
            need(time.monotonic() + 3 < deadline, 'Whole deadline exhausted before launch')
            request = {'schema_version': 1, 'argv': [plan['sandbox']['path'], '-f',
                       str(profile_path), *target], 'cwd': str(work)}
            request_path = evidence / (name + '-command.json'); save(request_path, request)
            raw = request_path.read_bytes()
            receipt = supervisor.capture(str(request_path), digest(raw),
                # Sentinel stays outside the test venv: installing a wheel may
                # add .pth startup hooks there. Never load those unsandboxed.
                python=plan['python']['base'], environment=environment,
                output=str(evidence / (name + '-capture')),
                deadline=min(deadline, time.monotonic() + plan['command_seconds']),
                supervisor_sha256=SUPERVISOR_SHA, raw_limit=RAW_LIMIT, cleanup_reserve=2.0)
            captured_ok(receipt); records.append(receipt)
            if name == 'probes':
                stdout = (evidence / 'probes-capture/stdout.raw').read_bytes()
                facts = parsed(stdout)
                need(type(facts) is dict and facts.get('status') == 'MAC_SANDBOX_PROBES_PASSED'
                     and set(facts.get('facts', {})) == {'compiler_exec', 'compiler_read', 'network_outbound'}
                     and all(v.get('denied') is True and type(v.get('errno')) is int
                             and v['errno'] in (1, 13) for v in facts['facts'].values()),
                     'HOLD_SANDBOX_CAPABILITY_PROBE')
        actual_inputs(plan)
        need(validate(plan_path, expected) == plan and profile_path.read_text() == profile(plan)
             and time.monotonic() < deadline, 'Final input/deadline gate')
        result = {'status': 'MAC_RUNTIME_COMMANDS_CAPTURED_NOT_ACCEPTANCE',
                  'plan_sha256': expected, 'profile_sha256': digest(profile_path.read_bytes()),
                  'scope': 'MacCP' + str(plan['python']['minor']) + ' original install/smoke; offline representatives only on approved CP312 plan',
                  'product_acceptance': False, 'records': records}
        save(evidence / 'RESULT.json', result)
        return result
    except BaseException as error:
        try:
            save(evidence / 'HOLD.json', {'status': 'HOLD_MAC_RUNTIME', 'error_type': type(error).__name__,
                 'plan_sha256': expected, 'product_acceptance': False})
        except BaseException:
            pass  # Preserve the original failure; existing supervisor raw stays intact.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'preview', 'run'], nargs='?', default='verify')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--approved-plan-sha256')
    args = parser.parse_args()
    if args.action == 'run':
        result = run(args.plan, args.plan_sha256, args.approved_plan_sha256)
    else:
        plan = validate(args.plan, args.plan_sha256)
        result = {'status': 'MAC_PROFILE_PREVIEW_NOT_EXECUTED', 'profile': profile(plan),
                  'profile_sha256': digest(profile(plan).encode()), 'product_acceptance': False,
                  'runtime_capability': 'HOLD_UNPROBED', 'supervisor_cross_macos': 'HOLD_UNTESTED'}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
