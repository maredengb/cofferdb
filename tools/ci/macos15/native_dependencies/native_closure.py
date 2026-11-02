"""Inspect declared Mac15 interpreter native dependencies without loading them.

This validates the static Mach-O dependency graph of each interpreter and its
indexed extension modules. System shared-cache dependencies are recorded as OS
inputs. Arbitrary runtime dlopen behavior is outside that graph and must remain
bounded by the runtime driver and sandbox. Default verify performs no IO.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import struct
import time
import types

COLLECTOR_SHA = '8ded06566d4854818cef66e1f9db488a7410f85da5cb00d83791839bad0e6cae'
PLAN_SHA = 'd98c070430f7154565c9389eabfe874e5081932119e16c5c9cc7ee61e96e70a3'
FRAMEWORK_VERSIONS = Path('/Library/Frameworks/Python.framework/Versions')
FRAMEWORK_ALIAS_OWNER = (0, 80)
SYSTEM_ROOTS = ('/usr/lib', '/System/Library/Frameworks', '/System/Library/PrivateFrameworks')
MACH_MAGICS = (b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf')
LOADS = {0xc: 'required', 0x80000018: 'weak', 0x8000001f: 'reexport',
         0x80000023: 'upward', 0x20: 'lazy'}
MAX_NATIVE_FILES = 512
MAX_GRAPH_STATES = 2048
MAX_BYTES = 2 * 1024 ** 3
MAX_FILE = 128 * 1024 ** 2


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def literal(value):
    need(type(value) is str and value and all(32 <= ord(c) < 127 for c in value)
         and '\\' not in value, 'Literal native path required')
    return value


def absolute(value):
    literal(value)
    path = Path(value)
    need(path.is_absolute() and str(path) == value and '..' not in path.parts
         and not value.startswith('//') and value != '/', 'Canonical absolute path required')
    return path


def system_path(value):
    path = absolute(value)
    return any(path.is_relative_to(Path(root)) for root in SYSTEM_ROOTS)


def version(value):
    return [value >> 16, value >> 8 & 255, value & 255]


def arm64_slice(raw):
    need(len(raw) >= 4, 'Truncated native magic')
    if raw[:4] == MACH_MAGICS[0]:
        return raw
    need(raw[:4] in MACH_MAGICS[1:], 'Unsupported native image magic')
    need(len(raw) >= 8, 'Truncated fat header')
    count = struct.unpack_from('>I', raw, 4)[0]
    width = 32 if raw[:4] == MACH_MAGICS[2] else 20
    need(0 < count <= 64 and 8 + count * width <= len(raw), 'Bounded fat architecture table')
    candidates = []
    regions = []
    for index in range(count):
        position = 8 + index * width
        if width == 32:
            cpu, subtype, start, length, alignment, reserved = struct.unpack_from('>IIQQII', raw, position)
            need(reserved == 0, 'Unsupported fat reserved field')
        else:
            cpu, subtype, start, length, alignment = struct.unpack_from('>IIIII', raw, position)
        need(alignment <= 31 and start >= 8 + count * width and length >= 32
             and start % (1 << alignment) == 0 and start + length <= len(raw), 'Invalid fat slice bounds')
        need(all(start + length <= old_start or old_end <= start for old_start, old_end in regions),
             'Overlapping fat slices')
        regions.append((start, start + length))
        if cpu == 0x0100000c and subtype & 0x00ffffff == 0:
            candidates.append(raw[start:start + length])
    need(len(candidates) == 1, 'Exactly one ordinary arm64 slice required')
    return candidates[0]


def macho(raw, maximum_os):
    """Parse load commands only; never execute or import a native image."""
    data = arm64_slice(raw)
    need(len(data) >= 32, 'Truncated arm64 header')
    magic, cpu, subtype, kind, count, size, flags, reserved = struct.unpack_from('<8I', data)
    need(magic == 0xfeedfacf and cpu == 0x0100000c and subtype & 0x00ffffff == 0
         and kind in (2, 6, 8) and reserved == 0, 'Ordinary arm64 executable/library/bundle required')
    need(count <= 4096 and size <= 4 * 1024 ** 2 and 32 + size <= len(data), 'Bounded load commands')
    offset, end = 32, 32 + size
    dependencies, rpaths, targets = [], [], []

    def string(position, length, minimum):
        relative = struct.unpack_from('<I', data, position + 8)[0]
        need(minimum <= relative < length, 'Invalid load-command string offset')
        suffix = data[position + relative:position + length]
        need(b'\0' in suffix, 'Unterminated load-command string')
        return literal(suffix.split(b'\0', 1)[0].decode('ascii', 'strict'))

    for _ in range(count):
        need(offset + 8 <= end, 'Truncated load command')
        command, length = struct.unpack_from('<II', data, offset)
        need(length >= 8 and length % 8 == 0 and offset + length <= end, 'Invalid load command size')
        if command in LOADS:
            need(length >= 24, 'Truncated dylib command')
            dependencies.append({'name': string(offset, length, 24), 'kind': LOADS[command]})
        elif command == 0x8000001c:
            need(length >= 16, 'Truncated rpath command')
            rpaths.append(string(offset, length, 12))
        elif command == 0x32:
            need(length >= 24, 'Truncated platform command')
            target, minimum, sdk, tools = struct.unpack_from('<4I', data, offset + 8)
            need(target == 1 and length == 24 + 8 * tools, 'Non-macOS or malformed platform command')
            targets.append(version(minimum))
        elif command == 0x24:
            need(length == 16, 'Malformed minimum macOS command')
            targets.append(version(struct.unpack_from('<I', data, offset + 8)[0]))
        elif command == 0xe:  # LC_LOAD_DYLINKER
            need(length >= 16 and kind == 2, 'Invalid dynamic loader declaration')
            loader = string(offset, length, 12)
            need(loader == '/usr/lib/dyld', 'Non-system dynamic loader unsupported')
            dependencies.append({'name': loader, 'kind': 'system_dynamic_loader'})
        elif command in (0x6, 0x7, 0xf, 0x10, 0x12, 0x13, 0x14, 0x15, 0x27):
            # Legacy linked-library/sub-framework rules and embedded dyld
            # environment can alter lookup outside the modeled path rules.
            raise ValueError('Unsupported dependency-affecting load command')
        elif command & 0x80000000 and command not in (0x80000022, 0x80000028, 0x80000033, 0x80000034):
            # Unknown required dyld commands must be handled deliberately.
            raise ValueError('Unsupported required dyld load command')
        offset += length
    need(offset == end and len(targets) == 1 and targets[0] <= maximum_os,
         'Missing/ambiguous or unsupported minimum macOS')
    need(len(dependencies) <= 128 and len(rpaths) <= 32, 'Bounded native dependencies/rpaths')
    return {'filetype': kind, 'minimum_macos': targets[0], 'dependencies': dependencies, 'rpaths': rpaths}


def expanded(name, loader, executable):
    literal(name)
    for token, parent in (('@loader_path', Path(loader).parent), ('@executable_path', Path(executable).parent)):
        if name == token:
            return str(absolute(str(parent)))
        if name.startswith(token + '/'):
            return str(absolute(os.path.normpath(str(parent / name[len(token) + 1:]))))
    return str(absolute(name))


def resolve_dependency(name, loader, executable, rpaths):
    if name.startswith('@rpath/'):
        tail = name[len('@rpath/'):]
        need(tail and not tail.startswith('/'), 'Invalid rpath-relative dependency')
        candidates = [str(absolute(os.path.normpath(str(Path(root) / tail)))) for root in rpaths]
    else:
        candidates = [expanded(name, loader, executable)]
    resolved = []
    for candidate in candidates:
        if system_path(candidate):
            resolved.append((candidate, candidate, 'system_shared_cache_or_OS_file'))
        elif os.path.lexists(candidate):
            real = str(Path(candidate).resolve(strict=True))
            absolute(real)
            resolved.append((candidate, real, 'regular_file'))
    # Refuse ambiguous alternatives rather than guessing a dyld fallback or
    # granting a directory. Several spellings of the same exact file are fine.
    need(len({row[1] for row in resolved}) <= 1, 'Ambiguous native dependency resolution')
    return resolved[0] if resolved else None


def inspect_runtime(runtime, index, maximum_os, read_file, deadline, *, protection=None):
    base = runtime['binary']['path']
    absolute(base)
    stdlib = absolute(runtime['stdlib'])
    rows = [row for row in index if absolute(row['path']).is_relative_to(stdlib)]
    need(sha(canonical(rows)) == runtime['stdlib_index_sha256'], 'Runtime index digest differs')
    need(len({row['path'] for row in rows}) == len(rows), 'Duplicate readonly index path')
    original, fixed_aliases, targets = {}, {}, {}
    for row in rows:
        if row.get('kind') != 'framework_install_alias':
            need(set(row) == {'path', 'bytes', 'mode', 'sha256'}, 'Ordinary index row shape')
            original[row['path']] = row
            continue
        version = f"3.{runtime['minor']}"
        root = FRAMEWORK_VERSIONS / version
        allowed = {str(root / 'lib' / ('python' + version) / f'config-{version}-darwin' / ('libpython' + version + '.' + suffix)) for suffix in ('a', 'dylib')}
        need(runtime['minor'] in (12, 13, 14) and stdlib == root / 'lib' / ('python' + version)
             and row['path'] in allowed and set(row) == {'path', 'kind', 'bytes', 'mode', 'sha256', 'alias'},
             'Fixed Framework alias index required')
        alias = row['alias']
        need(type(alias) is dict and set(alias) == {'target', 'text', 'identity', 'target_identity'}
             and alias['target'] == str(root / 'Python') and alias['text'] == '../../../Python', 'Fixed Framework alias target required')
        link, target = alias['identity'], alias['target_identity']
        need(type(link) is list and type(target) is list and len(link) == len(target) == 9
             and all(type(v) is int for v in link + target), 'Framework alias metadata required')
        need(stat.S_ISLNK(link[2]) and link[3:7] == [*FRAMEWORK_ALIAS_OWNER, 1, 15]
             and stat.S_ISREG(target[2]) and target[5] == 1 and target[6] == row['bytes']
             and stat.S_IMODE(target[2]) == row['mode'] and 0 <= row['bytes'] <= 32 * 2**20,
             'Framework alias metadata differs')
        checksum(row['sha256'])
        target_path = alias['target']
        if target_path in targets:
            old = targets[target_path]
            need(all(old[key] == row[key] for key in ('sha256', 'bytes', 'mode'))
                 and old['alias']['target_identity'] == target, 'Framework aliases disagree')
        targets[target_path] = row
        fixed_aliases[row['path']] = target_path
    cache, aliases, edges, states = {}, dict(fixed_aliases), [], set()
    read_bytes = 0

    def load(value):
        nonlocal read_bytes
        need(time.monotonic() < deadline, 'Native closure deadline')
        if value in cache:
            return cache[value]
        need(len(cache) < MAX_NATIVE_FILES, 'Native file count bound')
        data = read_file(value, MAX_FILE)
        read_bytes += len(data)
        need(read_bytes <= MAX_BYTES, 'Native read byte budget')
        info = os.stat(value, follow_symlinks=False)
        if protection is None:
            need(stat.S_ISREG(info.st_mode) and not info.st_mode & 0o022, 'Protected native file required')
        else:
            protection(value, info)
        digest = sha(data)
        if value == base:
            need(digest == runtime['binary']['sha256'], 'Interpreter identity changed')
        if value in targets:
            expected = targets[value]
            identity = [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            need(digest == expected['sha256'] and len(data) == expected['bytes']
                 and stat.S_IMODE(info.st_mode) == expected['mode']
                 and identity == expected['alias']['target_identity'], 'Indexed Framework target changed')
        if value in original:
            need(digest == original[value]['sha256'] and len(data) == original[value]['bytes']
                 and stat.S_IMODE(info.st_mode) == original[value]['mode'], 'Indexed native file changed')
        result = {'path': value, 'sha256': digest, 'bytes': len(data), 'mode': stat.S_IMODE(info.st_mode),
                  'image': macho(data, maximum_os)}
        cache[value] = result
        return result

    root_image = load(base)
    root_rpaths = tuple(expanded(value, base, base) for value in root_image['image']['rpaths'])
    seeds = [base, *targets] + [row['path'] for row in rows if 'kind' not in row and row['path'].endswith(('.so', '.dylib'))]
    queue = [(seed, root_rpaths, 0) for seed in seeds]
    while queue:
        value, inherited, depth = queue.pop(0)
        need(depth <= 64 and len(states) < MAX_GRAPH_STATES and time.monotonic() < deadline,
             'Native graph depth/state/deadline bound')
        state = (value, inherited)
        if state in states:
            continue
        states.add(state)
        observed = load(value)
        own = [expanded(item, value, base) for item in observed['image']['rpaths']]
        rpaths = tuple(dict.fromkeys([*own, *inherited]))
        need(len(rpaths) <= 64, 'Inherited run-path bound')
        for entry in observed['image']['dependencies']:
            resolved = resolve_dependency(entry['name'], value, base, rpaths)
            edge = {'loader': value, 'name': entry['name'], 'kind': entry['kind'], 'rpaths': list(rpaths)}
            if resolved is None:
                need(entry['kind'] == 'weak', 'Required native dependency missing')
                edge['status'] = 'weak_dependency_absent'
                # Absence is an input too: fresh checking resolves this exact edge again.
                edge['resolved'] = None
            else:
                spelling, real, scope = resolved
                edge.update(status=scope, spelling=spelling, resolved=real)
                if scope == 'regular_file':
                    aliases[spelling] = real
                    queue.append((real, rpaths, depth + 1))
            edges.append(edge)
    files = sorted(cache.values(), key=lambda row: row['path'])
    runtime_files = [{'path': row['path'], 'sha256': row['sha256']} for row in files
                     if row['path'] != base and not Path(row['path']).is_relative_to(stdlib)]
    need(len(runtime_files) <= 64, 'Runtime runner native file-list bound')
    return {'minor': runtime['minor'], 'base': base, 'stdlib': str(stdlib), 'files': files,
            'runtime_files': runtime_files, 'aliases': aliases, 'edges': edges,
            'scope': 'declared interpreter and indexed .so/.dylib static dependencies; no arbitrary dlopen proof'}


def helpers(root):
    root = absolute(root)
    sources = {}
    for name, expected in (('dynamic_plan.py', PLAN_SHA), ('dynamic_collect.py', COLLECTOR_SHA)):
        selected = root / name
        need(not any(part.is_symlink() for part in (selected, *selected.parents)), 'Linked helper refused')
        fd = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 131072,
                 'Bounded ordinary helper required')
            raw = os.read(fd, 131073)
            need(len(raw) == info.st_size and sha(raw) == expected, 'Pinned existing observer required')
            sources[name] = raw
        finally:
            os.close(fd)
    selected = root / 'dynamic_collect.py'
    module = types.ModuleType('native_closure_observer')
    module.__file__ = str(selected)
    exec(compile(sources['dynamic_collect.py'], str(selected), 'exec', dont_inherit=True), module.__dict__)
    return module, module.load_instantiator()


def checksum(value):
    import re
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) and value != '0' * 64, 'SHA256 required')


def snapshot_os(snapshot):
    need(snapshot['os']['system'] == 'Darwin' and snapshot['os']['machine'] == 'arm64', 'Native Mac snapshot required')
    numbers = snapshot['os']['version'].split('.')
    need(1 <= len(numbers) <= 3 and all(part.isascii() and part.isdecimal() and len(part) <= 3
                                       for part in numbers), 'Actual OS version required')
    maximum_os = [int(part) for part in numbers] + [0] * (3 - len(numbers))
    need(maximum_os[0] == 15, 'Mac15 snapshot required')
    return maximum_os


def input_set(snapshot, index):
    need(type(index) is list and len(index) <= 72000
         and sha(canonical(index)) == snapshot['readonly_index_sha256'], 'Complete readonly index identity differs')
    need(type(snapshot['runtimes']) is list and len(snapshot['runtimes']) == 3, 'Exactly three runtimes required')
    paths = []
    for runtime, minor in zip(snapshot['runtimes'], (12, 13, 14)):
        need(type(runtime['minor']) is int and runtime['minor'] == minor, 'Ordered native runtime required')
        absolute(runtime['binary']['path'])
        checksum(runtime['binary']['sha256'])
        root = absolute(runtime['stdlib'])
        need(not any(root == old or root in old.parents or old in root.parents for old in paths),
             'Distinct stdlib trees required')
        paths.append(root)
    need(all(sum(absolute(row['path']).is_relative_to(root) for root in paths) == 1 for row in index),
         'Index contains files outside selected stdlibs')


def observed_runtime(runtime, index, maximum_os, observer, deadline):
    base = runtime['binary']['path']
    return inspect_runtime(runtime, index, maximum_os,
        lambda value, bound: observer.native_bytes(value, bound, python_base=base), deadline,
        protection=lambda value, info: observer.native_protection(value, info, python_base=base))


def inspect(config, snapshot, index, validator_sha256, evidence_reference, observer, plan):
    plan.configuration(config)
    checksum(validator_sha256)
    absolute(evidence_reference)
    need(config['native_closure_validator_sha256'] == validator_sha256
         and snapshot['config_sha256'] == sha(canonical(config))
         and snapshot['source_commit'] == config['source_commit'], 'Validator/config/source binding differs')
    maximum_os = snapshot_os(snapshot)
    input_set(snapshot, index)
    deadline = time.monotonic() + 180
    graphs = []
    amended = json.loads(json.dumps(snapshot))
    for number, minor in enumerate((12, 13, 14)):
        runtime = snapshot['runtimes'][number]
        need(type(runtime['minor']) is int and runtime['minor'] == minor, 'Ordered native runtime required')
        # Re-index exact file sets, not just the old paths: additions must fail too.
        actual = observer.index_tree(Path(runtime['stdlib']), deadline)
        need(sha(canonical(actual)) == runtime['stdlib_index_sha256'], 'Fresh complete stdlib index differs')
        graph = observed_runtime(runtime, index, maximum_os, observer, deadline)
        graphs.append(graph)
        amended['runtimes'][number]['runtime_files'] = graph['runtime_files']
    report = {'schema_version': 1, 'kind': 'mac15_declared_static_native_graph_1',
              'validator_sha256': validator_sha256, 'input_snapshot_sha256': sha(canonical(snapshot)),
              'output_snapshot_sha256': sha(canonical(amended)), 'readonly_index_sha256': snapshot['readonly_index_sha256'],
              'graphs': graphs, 'system_dependency_scope': 'declared system install names, provided by the observed OS/dyld shared cache',
              'arbitrary_dynamic_loads_verified': False, 'sdk_loaded': False}
    certificate = {'kind': 'independent_exact_native_closure_1', 'validator_sha256': validator_sha256,
                   'snapshot_sha256': sha(canonical(amended)), 'readonly_index_sha256': snapshot['readonly_index_sha256'],
                   'complete': True, 'evidence_reference': evidence_reference}
    need(time.monotonic() < deadline, 'Native closure final deadline')
    return {'status': 'DECLARED_NATIVE_GRAPH_INSPECTED_NOT_EXECUTED', 'snapshot': amended,
            'report': report, 'report_sha256': sha(canonical(report)), 'closure': certificate,
            'sdk_executed': False, 'product_acceptance': False}


def fresh(snapshot, index, report, observer):
    """Recheck the full file set, native aliases/dependency resolution and tools."""
    need(report['output_snapshot_sha256'] == sha(canonical(snapshot))
         and report['readonly_index_sha256'] == sha(canonical(index)) == snapshot['readonly_index_sha256'],
         'Fresh gate graph/index/snapshot binding differs')
    input_set(snapshot, index)
    maximum_os = snapshot_os(snapshot)
    need(report['kind'] == 'mac15_declared_static_native_graph_1' and len(report['graphs']) == 3,
         'Fresh graph kind/count differs')
    deadline = time.monotonic() + 180
    for runtime, graph in zip(snapshot['runtimes'], report['graphs']):
        need(runtime['minor'] == graph['minor'] and runtime['binary']['path'] == graph['base']
             and runtime['stdlib'] == graph['stdlib']
             and runtime['runtime_files'] == graph['runtime_files'], 'Fresh graph runtime differs')
        need(sha(canonical(observer.index_tree(Path(runtime['stdlib']), deadline))) == runtime['stdlib_index_sha256'],
             'Fresh complete stdlib changed')
        need(observed_runtime(runtime, index, maximum_os, observer, deadline) == graph,
             'Fresh native dependency graph changed')
    need(type(snapshot['tools']) is dict
         and set(snapshot['tools']) == {'cargo', 'rustc', 'maturin', 'sandbox', 'compiler', 'prepare', 'builder'},
         'Exact selected tool set required')
    for name, row in snapshot['tools'].items():
        need(row is not None, 'Missing selected tool: ' + name)
        need(time.monotonic() < deadline, 'Fresh input deadline')
        tool_raw = observer.compiler_bytes(row['path']) if name == 'compiler' else observer.bytes_(row['path'], MAX_FILE)
        info = os.stat(row['path'], follow_symlinks=False)
        need(time.monotonic() < deadline and stat.S_ISREG(info.st_mode) and not info.st_mode & 0o022
             and sha(tool_raw) == row['sha256'], 'Fresh tool identity or protection changed: ' + name)
    need(time.monotonic() < deadline, 'Fresh final deadline')
    return {'status': 'DECLARED_INPUTS_FRESH_NOT_EXECUTED', 'snapshot_sha256': sha(canonical(snapshot)),
            'report_sha256': sha(canonical(report)), 'sdk_executed': False, 'product_acceptance': False}


def same_host(snapshot):
    # The collector also records image metadata in os; it is not part of the
    # three host identity fields and must not make every real snapshot fail.
    expected = {'system': platform.system(), 'machine': platform.machine(), 'version': platform.mac_ver()[0]}
    need(all(snapshot['os'].get(key) == value for key, value in expected.items()), 'Observed host OS changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['verify', 'inspect', 'fresh'], default='verify')
    parser.add_argument('--inputs')
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--helper-root')
    args = parser.parse_args()
    if args.action == 'verify':
        print(canonical({'status': 'NATIVE_CLOSURE_NOT_EXECUTED', 'sdk_executed': False}).decode(), end='')
        return 0
    need(platform.system() == 'Darwin' and platform.machine() == 'arm64'
         and platform.mac_ver()[0].split('.')[0] == '15', 'HOLD_REQUIRES_MAC15_ARM64')
    observer, plan = helpers(args.helper_root)
    raw = observer.bytes_(absolute(args.inputs), 2 ** 20)
    checksum(args.inputs_sha256)
    need(sha(raw) == args.inputs_sha256, 'Exact input bundle bytes required')
    bundle = plan.decode(raw)
    required = {'config', 'snapshot', 'index', 'evidence_reference'} if args.action == 'inspect' else {'snapshot', 'index', 'report'}
    need(type(bundle) is dict and set(bundle) == required, 'Exact native input bundle')
    documents = {}
    for name in required - {'evidence_reference'}:
        row = plan.row(bundle[name])
        value = observer.bytes_(row['path'], 16 * 1024 ** 2)
        need(sha(value) == row['sha256'], 'Bound native input changed')
        documents[name] = plan.decode(value)
    same_host(documents['snapshot'])
    validator_sha = sha(observer.bytes_(str(Path(__file__).absolute()), 131072))
    if args.action == 'inspect':
        absolute(bundle['evidence_reference'])
        result = inspect(documents['config'], documents['snapshot'], documents['index'], validator_sha,
                         bundle['evidence_reference'], observer, plan)
    else:
        need(documents['report']['validator_sha256'] == validator_sha, 'Native validator code differs')
        result = fresh(documents['snapshot'], documents['index'], documents['report'], observer)
    print(canonical(result).decode(), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
