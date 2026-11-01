"""Bind completed Mac15 builds and empty virtual environments to runtime plans.

The default action is inert. Instantiation reads bounded, digest-bound inputs;
it never invokes interpreters, installs packages, or runs the resulting plans.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import types
import zipfile


HELPERS = {
    'runner.py': 'f984c5a04572522efa2b5c422b854a71f827124449be353a0c83b2e57a1dc835',
    'probe.py': 'c0cdbf1ce7271f7585bba59d58e51a2f22fc5e9c2a49eeb82d87ab2725b90c10',
    'workflow_build.py': 'cbb6b8f55254022eb31434af9becee0be753c953bbe86bb60e99c9fa8e84b303',
    'venv_prepare.py': 'b82c9d0c15776cd200c95d52c03e34f2f555e55995bbf84571b6a2ec74b1fa71',
    'q_process_supervisor.py': '39d6c8842e1a9c972fd911feb4e3c483d1eb6947fc7291b65ac6f8a3bc778e51',
    'dynamic_inputs/dynamic_plan.py': 'c1b3a635f0b621018416b71dd6770891784624670c89ef99fd174092523ef010',
}
DOCUMENTS = ('config', 'snapshot', 'layout', 'driver_contract', 'native_closure',
             'build_result', 'build_manifest', 'empty_venvs_result')
MAX_JSON = 2 * 1024 * 1024
MAX_FILE = 128 * 1024 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(',', ':')) + '\n').encode()


def checksum(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value)
         and value != '0' * 64, 'Exact SHA256 required')
    return value


def path(value):
    need(type(value) is str and all(32 <= ord(c) < 127 for c in value)
         and '\\' not in value, 'Literal ASCII path required')
    selected = Path(value)
    need(selected.is_absolute() and str(selected) == value and value != '/'
         and not value.startswith('//') and '..' not in selected.parts,
         'Canonical absolute path required')
    return selected


def signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_file(value, bound=MAX_FILE):
    selected = path(value)
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
        before = os.fstat(opened)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_size <= bound and not before.st_mode & 0o022,
             'Protected bounded regular input required')
        chunks, size = [], 0
        while size <= bound:
            chunk = os.read(opened, min(65536, bound + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        need(size == before.st_size and signature(before) == signature(os.fstat(opened))
             and signature(before) == signature(os.stat(selected.name, dir_fd=directory,
                                                         follow_symlinks=False)),
             'Input changed during read')
        return b''.join(chunks)
    finally:
        if opened is not None:
            os.close(opened)
        os.close(directory)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def identity(value):
    need(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact file identity')
    path(value['path'])
    checksum(value['sha256'])
    return value


def bound_json(value):
    identity(value)
    raw = read_file(value['path'], MAX_JSON)
    need(sha(raw) == value['sha256'], 'Bound input digest differs')
    return decode(raw)


def load_helpers(root):
    """Load only unchanged, pinned maintenance code; no SDK or driver imports."""
    root = path(root)
    sources = {}
    for name, expected in HELPERS.items():
        selected = root.parent / name if name == 'q_process_supervisor.py' else root / name
        raw = read_file(str(selected), 131072)
        need(sha(raw) == expected, 'Existing helper digest differs: ' + name)
        sources[name] = (selected, raw)
    loaded = {}
    for name in ('runner.py', 'workflow_build.py', 'venv_prepare.py', 'dynamic_inputs/dynamic_plan.py'):
        selected, raw = sources[name]
        module = types.ModuleType('runtime_bridge_' + name.replace('/', '_'))
        module.__file__ = str(selected)
        exec(compile(raw, str(selected), 'exec', dont_inherit=True), module.__dict__)
        loaded[name] = module
    return loaded


def successful_records(result, status, plan, commands, r):
    need(type(result) is dict and result.get('status') == status
         and result.get('product_acceptance') is False
         and result.get('plan_sha256') == sha(r.canonical(plan)),
         'Completed helper result does not bind its exact plan')
    records = result.get('records')
    need(type(records) is list and len(records) == len(commands), 'Complete capture record set required')
    for receipt, (argv, cwd) in zip(records, commands):
        r.captured_ok(receipt)
        request = {'schema_version': 1, 'argv': argv, 'cwd': cwd}
        need(receipt.get('supervisor_sha256') == HELPERS['q_process_supervisor.py']
             and receipt.get('request_sha256') == sha(r.canonical(request))
             and receipt.get('argv') == argv, 'Capture command or supervisor identity differs')


def actual_wheel(filename, record, metadata):
    """Recheck the bytes already inspected by the successful pinned builder."""
    raw = read_file(str(filename))
    checksum(record.get('sha256'))
    checksum(record.get('native_sha256'))
    need(sha(raw) == record['sha256'], 'Actual wheel bytes differ from build record')
    import io
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        names = bundle.namelist()
        need(len(names) == len(set(names)) and len(names) <= 2048
             and sum(info.file_size for info in bundle.infolist()) <= MAX_FILE,
             'Bounded unique wheel member set required')
        native = [name for name in names if name.startswith('cofferdb/_native.') and name.endswith('.so')]
        need(len(native) == 1 and 'cofferdb/_source.json' in names, 'Native/source wheel members missing')
        need(sha(bundle.read(native[0])) == record['native_sha256'], 'Native wheel member identity differs')
        need(decode(bundle.read('cofferdb/_source.json')) == metadata, 'Wheel source metadata differs')


def actual_binding(value, expected, r):
    fields = {'minor', 'venv', 'invocation', 'real', 'base', 'sha256',
              'pyvenv_cfg_sha256', 'pip_or_SDK_installed_by_this_command'}
    need(type(value) is dict and set(value) == fields and type(value['minor']) is int
         and value['minor'] == expected['minor'] and value['venv'] == expected['venv']
         and value['base'] == expected['path'] and value['sha256'] == expected['sha256']
         and value['pip_or_SDK_installed_by_this_command'] is False, 'Actual empty venv binding differs')
    root = path(value['venv'])
    r.no_link(root)
    info = root.stat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
         and stat.S_IMODE(info.st_mode) == 0o700, 'Same private venv root required')
    need(value['invocation'] == str(root / 'bin/python'), 'Venv invocation differs')
    r.no_link(root / 'bin')
    real = path(value['real'])
    need(real.is_relative_to(root) and Path(value['invocation']).resolve(strict=True) == real,
         'Copied venv Python no longer resolves inside its root')
    need(sha(read_file(str(real))) == value['sha256']
         and sha(read_file(value['base'])) == value['sha256'], 'Base/copy interpreter changed')
    checksum(value['pyvenv_cfg_sha256'])
    cfg = read_file(str(root / 'pyvenv.cfg'), 16384)
    need(sha(cfg) == value['pyvenv_cfg_sha256'], 'Venv configuration changed')
    values = dict(line.split(' = ', 1) for line in cfg.decode('utf-8').splitlines() if ' = ' in line)
    need(values.get('include-system-site-packages') == 'false'
         and values.get('version', '').split('.')[:2] == ['3', str(value['minor'])], 'Venv is not isolated')
    packages = root / ('lib/python3.' + str(value['minor'])) / 'site-packages'
    r.no_link(packages)
    need(packages.is_dir() and not any(packages.iterdir()), 'New venv site-packages must still be empty')
    return {key: value[key] for key in ('invocation', 'real', 'base', 'sha256', 'minor')}


def runtime_driver(template, replacements, required_inputs, minor):
    need(type(template) is dict and set(template) == {'minor', 'path', 'sha256', 'arguments',
                                                     'inputs', 'no_detach_reviewed'}
         and type(template['minor']) is int and template['minor'] == minor,
         'Ordered exact runtime driver template required')
    result = {key: template[key] for key in template if key != 'minor'}
    identity({key: result[key] for key in ('path', 'sha256')})
    need(result['no_detach_reviewed'] is True and type(result['arguments']) is list
         and len(result['arguments']) <= 256, 'Reviewed bounded literal driver required')
    arguments = []
    used = set()
    for argument in result['arguments']:
        need(type(argument) is str and '\0' not in argument, 'Literal argument required')
        if argument in replacements:
            used.add(argument)
            arguments.append(replacements[argument])
        else:
            need('{' not in argument and '}' not in argument, 'Unknown or embedded argument placeholder')
            arguments.append(argument)
    need({'{wheel}', '{wheel_sha256}'} <= used, 'Driver must bind the actual wheel and its digest')
    need(type(result['inputs']) is list and len(result['inputs']) <= 62, 'Bounded exact driver inputs')
    inputs = {}
    for row in result['inputs'] + required_inputs:
        identity(row)
        need(row['path'] not in inputs or inputs[row['path']] == row, 'Conflicting driver input identity')
        inputs[row['path']] = row
    result['arguments'] = arguments
    result['inputs'] = [inputs[name] for name in sorted(inputs)]
    return result


def instantiate(bundle, modules):
    fields = {'schema_version', 'kind', *DOCUMENTS}
    need(type(bundle) is dict and set(bundle) == fields and type(bundle['schema_version']) is int
         and bundle['schema_version'] == 1 and bundle['kind'] == 'mac15_runtime_bridge_inputs_1',
         'Exact bridge input schema required')
    docs = {name: bound_json(bundle[name]) for name in DOCUMENTS}
    r = modules['runner.py']
    b = modules['workflow_build.py']
    v = modules['venv_prepare.py']
    d = modules['dynamic_inputs/dynamic_plan.py']
    d.verify_candidate_manifest(docs['config']['candidate_manifest_sha256'])
    initial = d.derive(docs['config'], docs['snapshot'], docs['layout'],
                       docs['driver_contract'], docs['native_closure'])
    need(set(initial['plans']) == {'build', 'empty_venvs'} and initial['missing'] == [
        'actual new empty-venv bindings and actual wheel/driver inputs before runtime plan instantiation'],
        'Earlier technical inputs remain incomplete')
    build, empty = initial['plans']['build'], initial['plans']['empty_venvs']
    b.validate(build)
    v.validate(empty)
    # Filenames and command receipts bind results to the actual initial plans.
    need(bundle['build_result']['path'] == build['evidence'] + '/RESULT.json'
         and bundle['build_manifest']['path'] == build['work'] + '/wheels/build-manifest.json'
         and bundle['empty_venvs_result']['path'] == empty['evidence'] + '/RESULT.json',
         'Result path is not the selected helper output')
    successful_records(docs['build_result'], 'EXISTING_THREE_ABI_BUILD_CAPTURED_NOT_ACCEPTANCE', build,
                       [(argv, build['checkout']) for _, argv, _ in b.commands(build)], r)
    successful_records(docs['empty_venvs_result'], 'THREE_EMPTY_VENVS_PREPARED_NOT_INSTALLED', empty,
                       [(argv, empty['evidence']) for argv in v.commands(empty)], r)
    manifest = docs['build_manifest']
    need(type(manifest) is dict and type(manifest.get('schema_version')) is int
         and manifest['schema_version'] == 1 and manifest.get('kind') == 'native_wheel_build_not_platform_acceptance'
         and manifest.get('builder_sha256') == build['builder']['sha256']
         and manifest.get('maturin') == '1.10.2', 'Selected builder result required')
    metadata = manifest['source']
    need(type(metadata) is dict and metadata.get('sdk_version') == '0.1.1'
         and metadata.get('engine_version') == '0.1.1'
         and metadata.get('repository') == 'https://github.com/maredengb/coffered',
         'Source SDK/engine/repository identity differs')
    source = metadata.get('source')
    need(type(source) is dict and set(source) == {'kind', 'commit', 'tree'}
         and source['kind'] == 'git_checkout' and source['commit'] == build['public_commit']
         and type(source['tree']) is str and re.fullmatch('[0-9a-f]{40}', source['tree'])
         and source['tree'] != '0' * 40, 'Final clean public source identity required')
    checksum(metadata.get('source_input_sha256'))
    source_path = build['work'] + '/source/bindings/python/python/cofferdb/_source.json'
    source_raw = read_file(source_path, MAX_JSON)
    need(decode(source_raw) == metadata, 'Actual prepared source metadata changed')
    artifacts = manifest.get('artifacts')
    bindings = docs['empty_venvs_result'].get('bindings')
    need(type(artifacts) is list and len(artifacts) == 3
         and type(bindings) is list and len(bindings) == 3, 'Exactly three actual wheels and venvs required')
    plans, observed = {}, []
    for index, minor in enumerate((12, 13, 14)):
        record = artifacts[index]
        runtime = record.get('runtime', {})
        selected = docs['snapshot']['runtimes'][index]
        need(type(runtime) is dict and type(runtime.get('minor')) is int and runtime['minor'] == minor
             and type(runtime.get('major')) is int and runtime['major'] == 3
             and runtime.get('implementation') == 'cpython' and runtime.get('system') == 'Darwin'
             and runtime.get('machine') == 'arm64' and runtime.get('free_threaded') is False
             and runtime.get('pointer_bits') == 64
             and runtime.get('version') == '3.%d.%d' % (minor, selected['micro']),
             'Actual artifact runtime differs from selected ordinary native interpreter')
        filename = 'cofferdb-0.1.1-cp3%d-cp3%d-macosx_15_0_arm64.whl' % (minor, minor)
        need(record.get('wheel') == filename and record.get('source_input_sha256') == metadata['source_input_sha256'],
             'Artifact filename/source differs')
        wheel = Path(build['work']) / 'wheels' / ('cp3' + str(minor)) / filename
        actual_wheel(wheel, record, metadata)
        python = actual_binding(bindings[index], empty['runtimes'][index], r)
        work = docs['layout']['work_parent'] + '/runtime-cp3' + str(minor)
        evidence = docs['layout']['evidence_parent'] + '/runtime-cp3' + str(minor)
        replacements = {'{wheel}': str(wheel), '{wheel_sha256}': record['sha256'],
                        '{native_sha256}': record['native_sha256'], '{venv}': bindings[index]['venv'],
                        '{source_metadata}': source_path, '{source_input_sha256}': metadata['source_input_sha256'],
                        '{sdk_version}': metadata['sdk_version'], '{engine_version}': metadata['engine_version'],
                        '{smoke_work}': work + '/smoke'}
        driver = runtime_driver(docs['driver_contract']['runtime_drivers'][index], replacements,
            [{'path': str(wheel), 'sha256': record['sha256']}, {'path': source_path, 'sha256': sha(source_raw)}], minor)
        plan = {'schema_version': 1, 'kind': 'macos15_python_runtime_isolation_1',
                'python': python, 'stdlib': selected['stdlib'], 'dependencies': [],
                'runtime_files': selected['runtime_files'], 'venv': bindings[index]['venv'],
                'work': work, 'evidence': evidence, 'sandbox': docs['snapshot']['tools']['sandbox'],
                'supervisor': build['supervisor'], 'probe_sha256': HELPERS['probe.py'],
                'compiler': docs['snapshot']['tools']['compiler'], 'driver': driver,
                'network_probe': docs['driver_contract']['network_probe'], 'whole_seconds': 300,
                'command_seconds': docs['driver_contract']['command_seconds']}
        r.schema(plan)
        r.profile(plan)
        r.actual_inputs(plan)
        for destination in (work, evidence):
            r.no_link(path(destination), missing=True)
            need(not os.path.lexists(destination), 'New runtime output already exists')
        plans['cp3' + str(minor)] = {'plan': plan, 'plan_sha256': sha(r.canonical(plan))}
        observed.append({'minor': minor, 'wheel': str(wheel), 'wheel_sha256': record['sha256'],
                         'pyvenv_cfg_sha256': bindings[index]['pyvenv_cfg_sha256']})
    # Re-read all bound documents at the end. This is a bounded observation,
    # not a promise of an atomic filesystem snapshot or a fresh runtime gate.
    for name in DOCUMENTS:
        need(bound_json(bundle[name]) == docs[name], 'Bound document changed during derivation')
    return {'status': 'RUNTIME_PLANS_DERIVED_NOT_EXECUTED', 'plans': plans, 'observed': observed,
            'input_sha256': {name: bundle[name]['sha256'] for name in DOCUMENTS},
            'source_commit': source['commit'], 'source_tree': source['tree'],
            'sdk_executed': False, 'product_acceptance': False,
            'fresh_native_closure_gate_required_before_execution': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['verify', 'instantiate'], default='verify')
    parser.add_argument('--inputs')
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--helper-root')
    args = parser.parse_args()
    if args.action == 'verify':
        result = {'status': 'RUNTIME_BRIDGE_NOT_EXECUTED', 'sdk_executed': False}
    else:
        checksum(args.inputs_sha256)
        raw = read_file(args.inputs, MAX_JSON)
        need(sha(raw) == args.inputs_sha256, 'Bridge input root digest differs')
        result = instantiate(decode(raw), load_helpers(args.helper_root))
    print(canonical(result).decode(), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
