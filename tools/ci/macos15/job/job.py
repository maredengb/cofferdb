"""Prepare fixed public inputs and run the existing macOS 15 package workflow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import types

WORKFLOW_SHA = '94d03971098cd379acd5f8a9062439c3f959eddbdced07d8872e5515f76f51ea'
PREPARER_SHA = '99a394b4c5c247800f525aa716091872fd82ac3c2bb93168c3c92374f6419daf'
INSTALLER_SHA = 'bb2899ac427808253bb5c201ae3fbb06f95bee74d14b330569e41ec989b00d8a'
BUSINESS_SHA = '0db442b087a9a1b69dab32263f60ddc799c7a7ec5b3a3735fd73c0b4e6483152'
PIP = {'name': 'pip-25.3-py3-none-any.whl', 'bytes': 1778622,
       'sha256': '9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd',
       'url': 'https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl'}
PREPARE_SECONDS = 1800
OFFLINE_PROFILE = '(version 1)\n(allow default)\n(deny network*)\n'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def module(raw, filename, name):
    value = types.ModuleType(name); value.__file__ = str(filename)
    exec(compile(raw, str(filename), 'exec', dont_inherit=True), value.__dict__)
    return value


def workflow_loader(root):
    """One bounded pinned entry read before its existing whole-budget context."""
    filename = Path(root) / 'workflow/workflow_bridge.py'
    need(filename.is_absolute() and '..' not in filename.parts
         and not any(x.is_symlink() for x in (filename, *filename.parents)), 'Literal helper path')
    fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd); raw = os.read(fd, 131073)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not info.st_mode & 0o022
             and len(raw) == info.st_size <= 131072 and sha(raw) == WORKFLOW_SHA, 'Pinned workflow entry')
    finally:
        os.close(fd)
    return module(raw, filename, 'selected_package_workflow')


def public_identity(value, environment):
    need(re.fullmatch('[0-9a-f]{40}', value['source_commit']) and value['source_commit'] != '0' * 40,
         'Exact selected public commit')
    need(environment.get('GITHUB_REPOSITORY_ID') == '1422246026'
         and environment.get('GITHUB_REPOSITORY') in ('maredengb/coffered', 'maredengb/cofferdb')
         and environment.get('GITHUB_SHA') == value['source_commit']
         and re.fullmatch('[1-9][0-9]*', environment.get('GITHUB_RUN_ID', ''))
         and environment.get('GITHUB_RUN_ATTEMPT') == '1', 'Selected public job or first attempt differs')
    need(type(value['execution_reference']) is str
         and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._:/#=-]{0,255}', value['execution_reference']),
         'Literal reviewed execution reference required')
    return {key: environment[key] for key in ('GITHUB_REPOSITORY_ID', 'GITHUB_REPOSITORY',
            'GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT')}


def cargo_config_gate(checkout):
    """Cargo searches cwd ancestors, not only the selected manifest directory."""
    roots = {Path(checkout), Path(checkout) / 'bindings', Path(checkout) / 'bindings/python',
             *Path(checkout).parents}
    for root in roots:
        for leaf in ('config', 'config.toml'):
            need(not os.path.lexists(root / '.cargo' / leaf), 'Unexpected ancestor Cargo configuration')


def tree_index(b, root, *, max_entries=10000, max_bytes=1024 ** 3, check=lambda: None):
    rows = []; size = 0; entries = 0
    def walk(folder):
        nonlocal entries, size
        with os.scandir(folder) as scan:
            for item in scan:
                check(); entries += 1
                need(entries <= max_entries and not item.is_symlink(), 'Bounded real input tree')
                if item.is_dir(follow_symlinks=False):
                    walk(Path(item.path))
                else:
                    raw = b.read_file(item.path); size += len(raw)
                    need(size <= max_bytes, 'Input tree byte budget')
                    rows.append({'path': str(Path(item.path).relative_to(root)), 'bytes': len(raw),
                                 'sha256': sha(raw)})
    walk(Path(root)); check()
    return sorted(rows, key=lambda x: x['path'])


def rust_pair(preparer, rustup_home):
    rows = preparer.installed_toolchains(rustup_home)
    need(len(rows) <= 32, 'Installed toolchain listing bound')
    selected = [x for x in rows if Path(x['directory']).name == 'stable-aarch64-apple-darwin']
    need(len(selected) == 1 and 'unusable' not in selected[0], 'Existing stable native Rust pair required')
    return selected[0]


def rust_version(raw):
    lines = raw.decode('utf-8').strip().splitlines()
    need(len(raw) <= 65536 and len(lines) <= 64 and 'host: aarch64-apple-darwin' in lines, 'Native arm64 Rust host required')
    first = lines[0]
    match = re.fullmatch(r'rustc (\d+)\.(\d+)\.(\d+) .*', first)
    need(match is not None and tuple(map(int, match.groups())) >= (1, 80, 0), 'Rust MSRV 1.80 not met')
    return first


class Captures:
    """Use Q directly; retain original receipts and raw stream identities."""
    def __init__(self, b, r, supervisor, python, root, budget, deadline, fresh):
        self.b, self.r, self.supervisor = b, r, supervisor
        self.python, self.root, self.budget = python, root, budget
        self.deadline, self.fresh = deadline, fresh
        self.records = []

    def verify(self):
        for item in self.records:
            self.r.captured_ok(item['receipt'])
            for row in item['receipt']['raw_streams'].values():
                raw = self.b.read_file(row['path'], 16 * 1024 ** 2)
                need(len(raw) == row['size'] and sha(raw) == row['sha256'], 'Original preparation stream changed')

    def run(self, name, argv, cwd, environment, seconds):
        self.budget['check'](); self.fresh()
        need(time.monotonic() + 3 < self.deadline, 'Input preparation deadline before command')
        request = {'schema_version': 1, 'argv': argv, 'cwd': str(cwd)}
        raw = self.r.canonical(request); request_path = self.root / (name + '.json')
        self.r.save(request_path, request)
        need(self.b.read_file(str(request_path)) == raw, 'Request changed after save')
        receipt = self.supervisor.capture(str(request_path), sha(raw), python=self.python,
            environment=environment, output=str(self.root / name), supervisor_sha256=self.r.SUPERVISOR_SHA,
            deadline=min(self.deadline, self.budget['deadline'], time.monotonic() + seconds),
            raw_limit=16 * 1024 ** 2, cleanup_reserve=2.0)
        # Even an unsuccessful receipt is retained in Q's original evidence directory.
        self.r.captured_ok(receipt)
        need(receipt.get('argv') == argv and receipt.get('request_sha256') == sha(raw)
             and receipt.get('supervisor_sha256') == self.r.SUPERVISOR_SHA
             and set(receipt['raw_streams']) == {'stdout', 'stderr', 'status'}, 'Original command binding differs')
        self.records.append({'name': name, 'receipt': receipt}); self.verify()
        self.budget['check'](); self.fresh()
        need(time.monotonic() < self.deadline, 'Input preparation deadline after command')
        row = receipt['raw_streams']['stdout']
        raw = self.b.read_file(row['path'], 16 * 1024 ** 2)
        need(len(raw) == row['size'] and sha(raw) == row['sha256'],
             'Consumed preparation stdout changed')
        self.budget['check']()
        need(time.monotonic() < self.deadline, 'Input preparation deadline after read')
        return raw


def developer_tools(raw, b):
    developer = raw.decode('utf-8').strip()
    need(re.fullmatch(r'/Applications/Xcode(?:[A-Za-z0-9_.-]*)\.app/Contents/Developer', developer)
         or developer == '/Library/Developer/CommandLineTools', 'Explicit existing Apple developer directory')
    root = b.path(developer)
    candidates = {'git': root / 'usr/bin/git',
                  'compiler': root / ('usr/bin/clang' if developer.endswith('CommandLineTools')
                                      else 'Toolchains/XcodeDefault.xctoolchain/usr/bin/clang')}
    result = {'developer_directory': developer}
    for name, invocation in candidates.items():
        real = invocation.resolve(strict=True)
        need(real.is_relative_to(root), 'Developer tool alias escapes selected installation')
        data = b.host_file(str(real), role='compiler') if name == 'compiler' else b.read_file(str(real))
        need(os.access(real, os.X_OK), 'Developer tool unavailable')
        result[name] = {'path': str(real), 'sha256': sha(data)}
    return result


def source_gate(captures, value, environment, label, git_tool):
    checkout = value['checkout']
    command = [git_tool, '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null']
    queries = [('root', ['rev-parse', '--show-toplevel']), ('head', ['rev-parse', 'HEAD']),
               ('tree', ['rev-parse', 'HEAD^{tree}']), ('origin', ['remote', 'get-url', 'origin']),
               ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]
    result = {name: captures.run(label + '-' + name, command + argv, checkout,
                                environment, 20).decode('utf-8').strip() for name, argv in queries}
    need(result['root'] == checkout and result['head'] == value['source_commit']
         and re.fullmatch('[0-9a-f]{40}', result['tree'])
         and result['origin'] in ('https://github.com/maredengb/coffered',
                                  'https://github.com/maredengb/coffered.git')
         and result['status'] == '', 'Fixed clean public checkout required')
    cargo_config_gate(checkout)
    return result


def driver_contract(b, r, root, inputs, job, execution_reference, preflight_ref, budget_ref):
    """Bind original public assertions; actual artifact tokens stay uninstantiated."""
    installer = root / 'tools/ci/installed_wheel/install_driver.py'
    driver = installer.parent / 'business/driver.py'
    fixture_root = root / 'examples/data/real_corpus'
    driver_raw = b.read_file(str(driver), 131072)
    need(sha(driver_raw) == BUSINESS_SHA and sha(b.read_file(str(installer), 131072)) == INSTALLER_SHA,
         'Reviewed installed-package code differs')
    business = module(driver_raw, driver, 'selected_fixture_preparation')
    # Importing this pinned module does not import the SDK or invoke its main.
    provenance = b.decode(b.read_file(str(fixture_root / 'provenance.json'), 2 ** 20))
    names = ['provenance.json', 'code/documents.jsonl', 'code/chunks.jsonl', 'code/embeddings.jsonl']
    names += sorted(x for x in provenance['files_sha256'] if x.startswith('source_snapshot/'))
    fixture_rows = []
    for name in names:
        need('..' not in Path(name).parts and not Path(name).is_absolute(), 'Literal fixture member')
        raw = b.read_file(str(fixture_root / name))
        fixture_rows.append({'path': str(fixture_root / name), 'sha256': sha(raw)})
    _documents, _chunks, _embeddings, derived = business.fixture(fixture_root)
    bridge_path = root / 'tools/ci/macos15/runtime_bridge/runtime_bridge.py'
    pip_path = inputs / PIP['name']
    need(len(b.read_file(str(pip_path))) == PIP['bytes'] and sha(b.read_file(str(pip_path))) == PIP['sha256'], 'Frozen local pip input')
    common = [{'path': str(pip_path), 'sha256': PIP['sha256']}, {'path': str(driver), 'sha256': BUSINESS_SHA},
              {'path': str(bridge_path), 'sha256': sha(b.read_file(str(bridge_path), 131072))}, *fixture_rows]
    templates = []
    for minor in (12, 13, 14):
        context = {'attempt_id': job['GITHUB_RUN_ID'] + '-' + job['GITHUB_RUN_ATTEMPT'],
                   'cell_id': 'N%02d' % (minor - 11), 'authorization_reference': execution_reference,
                   'preflight_evidence': preflight_ref['sha256'], 'budget_reference': budget_ref['sha256'],
                   'files_sha256': business.hashes(derived)}
        context_path = inputs / ('cp3%d-context.json' % minor)
        r.save(context_path, context); context_ref = {'path': str(context_path), 'sha256': sha(canonical(context))}
        need(b.read_file(str(context_path), 2 ** 20) == canonical(context), 'Original context write changed')
        args = ['run', '--pip-wheel', str(pip_path), '--runtime-bridge', str(bridge_path), '--wheel', '{wheel}', '--wheel-sha256', '{wheel_sha256}',
                '--native-sha256', '{native_sha256}', '--source-metadata', '{source_metadata}',
                '--source-input-sha256', '{source_input_sha256}', '--sdk-version', '{sdk_version}',
                '--engine-version', '{engine_version}', '--venv', '{venv}', '--driver', str(driver),
                '--driver-sha256', BUSINESS_SHA, '--context', str(context_path), '--context-sha256', context_ref['sha256'],
                '--fixtures', str(fixture_root), '--work', '{smoke_work}']
        templates.append({'minor': minor, 'path': str(installer), 'sha256': INSTALLER_SHA,
                          'arguments': args, 'inputs': [*common, context_ref], 'no_detach_reviewed': True})
    return {'runtime_drivers': templates, 'network_probe': {'address': '1.1.1.1', 'port': 443},
            'command_seconds': 240}


def execute(value, w, modules, helper_root, budget):
    b = modules['runtime']; r = modules['originals']['runner.py']
    r.host_gate(); job = public_identity(value, os.environ)
    need(sys.version_info[:2] == (3, 12) and sys.implementation.name == 'cpython', 'Bootstrap CPython 3.12 required')
    for key in ('checkout', 'rustup_home', 'output'): b.path(value[key]); r.no_link(Path(value[key]), missing=key == 'output')
    checkout = Path(value['checkout']); out = Path(value['output'])
    need(not r.beneath(out, checkout) and not r.beneath(checkout, out)
         and not r.beneath(out, Path(value['rustup_home'])), 'Independent job output required')
    need(Path(helper_root) == checkout / 'tools/ci/macos15', 'Helpers must belong to selected checkout')
    need(not out.exists(), 'New job output required')
    # RUNNER_TEMP belongs to this ephemeral job. Do not change its permissions.
    need(out.parent.stat().st_uid == os.getuid(), 'Owned job temporary parent')
    out.mkdir(mode=0o700)
    for name in ('evidence', 'inputs', 'home', 'cargo'):
        (out / name).mkdir(mode=0o700)
    refs = {}
    def saved(name, data):
        expected = sha(canonical(data)); target = out / name; r.save(target, data)
        need(sha(b.read_file(str(target), 16 * 1024 ** 2)) == expected, 'Original job record changed')
        refs[str(target)] = expected
        return {'path': str(target), 'sha256': expected}
    budget_ref = saved('INTENT.json', {'job': job, 'source_commit': value['source_commit'],
        'execution_reference': value['execution_reference'], 'started_monotonic': budget['started'],
        'deadline_monotonic': budget['deadline'], 'whole_seconds': 7200, 'prepare_seconds': PREPARE_SECONDS,
        'automatic_retry': False, 'product_acceptance': False})
    local = Path(__file__).parent
    preparer_raw = b.read_file(str(local / 'prepare_inputs.py'), 131072)
    need(sha(preparer_raw) == PREPARER_SHA, 'Input recipe identity differs')
    preparer = module(preparer_raw, local / 'prepare_inputs.py', 'selected_input_preparation')
    python = str(Path(sys.executable).resolve(strict=True))
    immutable = {python: sha(b.host_file(python, role='python')), str(local / 'prepare_inputs.py'): PREPARER_SHA,
                 str(local / 'job.py'): sha(b.read_file(str(local / 'job.py'), 131072)),
                 str(Path(helper_root) / 'workflow/workflow_bridge.py'): WORKFLOW_SHA}
    immutable_roles = {python: 'python'}
    for system in ('/usr/bin/xcode-select', '/usr/bin/curl', '/usr/bin/sandbox-exec'):
        immutable[system] = sha(b.read_file(system))
    def fresh():
        budget['check'](); public_identity(value, os.environ)
        for filename, expected in immutable.items():
            need(sha(b.host_file(filename, role=immutable_roles[filename]) if filename in immutable_roles else b.read_file(filename)) == expected, 'Selected job code/tool changed')
        for filename, expected in refs.items():
            need(sha(b.read_file(filename, 16 * 1024 ** 2)) == expected, 'Original job record changed')
        budget['check']()
    supervisor = r.load_supervisor({'supervisor': {'path': str(Path(helper_root).parent / 'q_process_supervisor.py'),
                                                  'sha256': r.SUPERVISOR_SHA}})
    cap = Captures(b, r, supervisor, python, out / 'evidence', budget,
                   min(budget['deadline'], time.monotonic() + PREPARE_SECONDS), fresh)
    environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'HOME': str(out / 'home'),
                   'TMPDIR': str(out / 'home'), 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
    try:
        developer = developer_tools(cap.run('developer-directory', ['/usr/bin/xcode-select', '--print-path'],
                                             out / 'home', environment, 20), b)
        saved('inputs/developer-tools.json', developer)
        immutable_roles[developer['compiler']['path']] = 'compiler'
        immutable.update({developer[key]['path']: developer[key]['sha256'] for key in ('git', 'compiler')})
        source = source_gate(cap, value, environment, 'initial-source', developer['git']['path'])
        pair = rust_pair(preparer, value['rustup_home']); saved('inputs/rust-selection.json', pair)
        immutable.update({pair[key]['path']: pair[key]['sha256'] for key in ('cargo', 'rustc')})
        observed_version = rust_version(cap.run('rust-version', [pair['rustc']['path'], '-vV'],
                                               out / 'home', environment, 30))
        recipe = preparer.command_recipe(value['checkout'], pair['cargo']['path'], pair['rustc']['path'],
            str(out / 'cargo'), str(out / 'vendor'), str(out / 'home'), str(out / 'maturin.whl'))
        saved('inputs/recipe.json', recipe)
        pip_path = out / 'inputs' / PIP['name']
        pip_argv = ['/usr/bin/curl', '--fail', '--silent', '--show-error', '--proto', '=https',
                    '--max-time', '180', '--max-filesize', str(PIP['bytes']), '--output', str(pip_path), PIP['url']]
        cap.run('pip-download', pip_argv, checkout, environment, 180)
        pip_raw = b.read_file(str(pip_path), PIP['bytes'])
        need(len(pip_raw) == PIP['bytes'] and sha(pip_raw) == PIP['sha256'], 'Fixed pip download differs')
        immutable[str(pip_path)] = PIP['sha256']
        saved('inputs/pip-input.json', PIP)
        for key in ('download', 'vendor'):
            cap.run(key, recipe[key]['argv'], checkout,
                    {**recipe['environment_network_prepare'], 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'},
                    recipe[key]['seconds'])
            if key == 'download':
                extracted = preparer.extract_maturin(str(out / 'maturin.whl'), str(out / 'maturin'))
                immutable[extracted['path']] = extracted['sha256']
                host_version = list(map(int, modules['native'].platform.mac_ver()[0].split('.')))
                host_version += [0] * (3 - len(host_version))
                for tool in (pair['cargo']['path'], pair['rustc']['path'], extracted['path']):
                    modules['native'].macho(b.read_file(tool), host_version)
                saved('inputs/maturin.json', extracted)
        cargo_config_gate(checkout)
        with (out / 'cargo/config.toml').open('xb') as stream:
            stream.write(recipe['config_bytes_utf8'].encode())
        (out / 'cargo/config.toml').chmod(0o600)
        vendor = preparer.vendor_audit(value['checkout'], str(out / 'vendor'))
        saved('inputs/vendor-audit.json', vendor)
        vendor_index = tree_index(b, out / 'vendor', check=budget['check'])
        saved('inputs/vendor-index.json', vendor_index)
        config_sha = sha(recipe['config_bytes_utf8'].encode()); immutable[str(out / 'cargo/config.toml')] = config_sha
        profile_path = out / 'inputs/offline.sb'
        with profile_path.open('xb') as stream: stream.write(OFFLINE_PROFILE.encode())
        profile_path.chmod(0o600); immutable[str(profile_path)] = sha(OFFLINE_PROFILE.encode())
        for index, item in enumerate(recipe['offline_metadata']):
            stdout = cap.run('metadata-' + str(index), ['/usr/bin/sandbox-exec', '-f', str(profile_path), *item['argv']],
                            checkout, recipe['environment_offline'], item['seconds'])
            metadata = b.decode(stdout)
            need(type(metadata.get('packages')) is list and metadata.get('resolve') is not None,
                 'Complete offline Cargo metadata required')
        need(source_gate(cap, value, environment, 'final-source', developer['git']['path']) == source, 'Source changed during preparation')
        need(tree_index(b, out / 'vendor', check=budget['check']) == vendor_index, 'Vendor changed during preparation')
        cap.verify(); fresh()
        preflight = saved('inputs/PREPARED.json', {'source': source, 'rust_version': observed_version,
            'maturin': extracted, 'vendor_index_sha256': sha(canonical(vendor_index)), 'config_sha256': config_sha,
            'records': cap.records, 'input_preparation_completed': True, 'sdk_tested': False})
        contract = driver_contract(b, r, checkout, out / 'inputs', job, value['execution_reference'], preflight, budget_ref)
        contract_ref = saved('inputs/driver-contract.json', contract)
        for driver in contract['runtime_drivers']:
            for row in driver['inputs']:
                immutable[row['path']] = row['sha256']
        need(time.monotonic() < cap.deadline, 'Input preparation deadline before build')
        selected = {key: pair[key]['path'] for key in ('cargo', 'rustc')}
        selected.update({'maturin': extracted['path'], 'sandbox': '/usr/bin/sandbox-exec', 'compiler': developer['compiler']['path'],
                         'prepare': str(checkout / 'tools/prepare_source.py'), 'builder': str(checkout / 'tools/build_python.py')})
        workflow_input = {'schema_version': 1, 'kind': 'mac15_complete_workflow_inputs_1', 'source_commit': value['source_commit'],
                          'repository_id': 1422246026, 'checkout': value['checkout'], 'cargo_home': str(out / 'cargo'),
                          'selected_tools': selected, 'driver_contract': contract_ref, 'output': str(out / 'workflow')}
        saved('inputs/workflow-inputs.json', workflow_input)
        def recheck():
            fresh(); cap.verify()
        final_checks = []
        result = w.pipeline(workflow_input, modules, helper_root, budget, recheck,
                            retain_check=final_checks.append)
        need(len(final_checks) == 1 and callable(final_checks[0]), 'Original workflow verifier required')
        post = Captures(b, r, supervisor, python, out / 'evidence', budget,
                        min(budget['deadline'], time.monotonic() + 120), fresh)
        need(source_gate(post, value, environment, 'post-build-source', developer['git']['path']) == source, 'Final public source drift')
        saved('inputs/FINAL-SOURCE.json', {'source': source, 'records': post.records})
        def verify_outputs():
            fresh(); cap.verify(); post.verify()
            need(sha(b.read_file(str(out / 'maturin/archive.whl'))) == preparer.MATURIN['sha256'], 'Final maturin archive drift')
            need(tree_index(b, out / 'vendor', check=budget['check']) == vendor_index, 'Final vendor drift')
            final_checks[0]()
        verify_outputs()
        saved('RESULT.json', {'status': 'MAC15_JOB_COMPLETED_NOT_ACCEPTANCE', 'source_commit': value['source_commit'],
              'source_tree': source['tree'], 'workflow_result': result, 'product_acceptance': False,
              'requires_successful_job_exit_and_artifact_retrieval': True})
        verify_outputs()
        return result
    except BaseException as error:
        try: r.save(out / 'HOLD.json', {'status': 'HOLD_MAC15_JOB', 'error_type': type(error).__name__,
                                      'automatic_retry': False, 'product_acceptance': False})
        except BaseException: pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('verify', 'run'), default='verify')
    for name in ('checkout', 'source-commit', 'execution-reference', 'rustup-home', 'output'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.action == 'verify':
        print('{"status":"MAC15_JOB_NOT_EXECUTED","product_acceptance":false}'); return 0
    root = str(Path(args.checkout) / 'tools/ci/macos15'); w = workflow_loader(root)
    with w.whole_budget() as budget:
        modules = w.bootstrap(root)
        execute(vars(args), w, modules, root, budget)
        budget['check']()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
