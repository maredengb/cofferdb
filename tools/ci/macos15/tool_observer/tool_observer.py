"""Observe explicitly selected existing tools; the default action is inert."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import time
import types

BRIDGE_SHA = '3909795ef8a29ce68e9d5eb2314306174139f242cff6e7b42bf31853ad130b88'
KEYS = ('cargo', 'rustc', 'maturin', 'sandbox', 'compiler', 'prepare', 'builder')
QUERIES = ('cargo', 'rustc', 'maturin', 'compiler')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_bridge(filename):
    p = Path(filename)
    need(p.is_absolute() and str(p) == filename and '..' not in p.parts
         and not any(x.is_symlink() for x in (p, *p.parents)), 'Literal fixed bridge path')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size <= 131072
             and not st.st_mode & 0o022, 'Protected bridge source')
        raw = os.read(fd, 131073)
        need(len(raw) == st.st_size and sha(raw) == BRIDGE_SHA, 'Pinned bridge source')
    finally:
        os.close(fd)
    m = types.ModuleType('tool_observer_bridge'); m.__file__ = filename
    exec(compile(raw, filename, 'exec', dont_inherit=True), m.__dict__)
    return m


def version(name, raw):
    """Record a bounded actual version response, not a predicted tool hash."""
    need(len(raw) <= 65536, 'Version output bound')
    text = raw.decode('utf-8', errors='strict').strip()
    need(text and '\0' not in text and len(text.splitlines()) <= 64, 'Version response shape')
    first = text.splitlines()[0]
    if name in ('cargo', 'rustc'):
        need(re.fullmatch(name + r' \d+\.\d+\.\d+[^\r\n]*', first), 'Selected Rust tool response differs')
    elif name == 'maturin':
        need(text == 'maturin 1.10.2', 'Existing builder requires maturin 1.10.2')
    elif name == 'compiler':
        need(re.match(r'(?:Apple )?clang version \d+\.', first), 'Selected clang response differs')
    else:
        raise ValueError('No version query for this tool')
    return text


def selected_rows(b, selected, checkout):
    need(type(selected) is dict and set(selected) == set(KEYS), 'Exact seven selected tool paths')
    checkout = b.path(checkout)
    need(selected['sandbox'] == '/usr/bin/sandbox-exec', 'Existing system sandbox path')
    for key, leaf in [('prepare', 'prepare_source.py'), ('builder', 'build_python.py')]:
        need(selected[key] == str(checkout / 'tools' / leaf), 'Existing checkout maintenance script')
    for key in ('cargo', 'rustc', 'maturin'):
        # In particular, never resolve a rustup shim and then execute it as rustup.
        need(b.path(selected[key]).name == key, 'Select actual named tool binary, not a shim target')
    rows = {}
    for key in KEYS:
        raw = b.host_file(selected[key], role='compiler') if key == 'compiler' else b.read_file(selected[key])
        if key not in ('prepare', 'builder'):
            need(os.access(selected[key], os.X_OK), 'Selected executable unavailable')
        rows[key] = {'path': selected[key], 'sha256': sha(raw)}
    return rows


def host_matches(snapshot):
    need(platform.system() == 'Darwin' and platform.machine() == 'arm64'
         and platform.mac_ver()[0].split('.')[0] == '15', 'HOLD_REQUIRES_MAC15_ARM64')
    observed = snapshot['os']
    need(observed.get('system') == 'Darwin' and observed.get('machine') == 'arm64'
         and observed.get('version') == platform.mac_ver()[0], 'Snapshot host differs')


def sealed_streams(b, evidence, name, receipt):
    seals = receipt.get('raw_streams')
    need(type(seals) is dict and set(seals) == {'stdout', 'stderr', 'status'}, 'Original stream seals required')
    output = {}
    for stream in ('stdout', 'stderr', 'status'):
        row = seals[stream]; selected = str(evidence / (name + '-capture') / (stream + '.raw'))
        need(type(row) is dict and set(row) == {'path', 'size', 'sha256'}
             and row['path'] == selected and type(row['size']) is int
             and 0 <= row['size'] <= 65536, 'Exact original stream seal required')
        need(stream != 'status' or row['size'] <= 16384, 'Original control stream bound')
        b.checksum(row['sha256']); raw = b.read_file(selected, 65536)
        need(len(raw) == row['size'] and sha(raw) == row['sha256'], 'Captured stream differs from original seal')
        output[stream] = raw
    need(len(output['stdout']) + len(output['stderr']) <= 65536, 'Original aggregate raw bound')
    return output


def observe(b, helper_root, snapshot, selected, checkout, output, *, clock=time.monotonic, input_check=lambda: None, started=None):
    """Four fixed version commands under the existing supervisor; no install/build."""
    started = clock() if started is None else started
    deadline = started + 180
    host_matches(snapshot)
    input_check()
    need(clock() + 3 < deadline, 'Tool observation deadline after initial input')
    need(snapshot.get('kind') == 'observed_mac15_technical_inputs_1'
         and snapshot.get('schema_version') == 1 and type(snapshot.get('schema_version')) is int,
         'Actual collector snapshot required')
    need(snapshot['tools'] == dict.fromkeys(KEYS), 'Tool observation requires the unfilled original snapshot')
    root = b.path(helper_root)
    helpers = b.load_helpers(helper_root); r = helpers['runner.py']
    expected_paths = {name: str(root.parent / name if name == 'q_process_supervisor.py' else root / name)
                      for name in b.HELPERS if name != 'dynamic_inputs/dynamic_plan.py'}
    need(snapshot['helper_paths'] == expected_paths, 'Observed helper paths differ')
    need([x['minor'] for x in snapshot['runtimes']] == [12, 13, 14], 'Three ordered observed runtimes')
    python = snapshot['runtimes'][0]['binary']; r.row(python, executable=True, host_role='python')
    rows = selected_rows(b, selected, checkout)
    out = b.path(output)
    immutable = [b.path(checkout), b.path(b.__file__), root, root.parent / 'q_process_supervisor.py']
    immutable += [b.path(x['path']) for x in rows.values()]
    immutable += [b.path(x['binary']['path']) for x in snapshot['runtimes']]
    immutable += [b.path(x['stdlib']) for x in snapshot['runtimes']]
    need(all(not r.beneath(out, p) and not r.beneath(p, out) for p in immutable), 'Output overlaps observed inputs')
    supervisor_row = {'path': expected_paths['q_process_supervisor.py'], 'sha256': r.SUPERVISOR_SHA}
    supervisor = r.load_supervisor({'supervisor': supervisor_row})
    need(clock() + 3 < deadline, 'Tool observation deadline after initial readback')
    evidence = r.private_new(str(out)); work = evidence / 'work'; work.mkdir(mode=0o700)
    r.save(evidence / 'INTENT.json', {'snapshot_sha256': sha(b.canonical(snapshot)), 'selected_tools': rows,
           'started_monotonic': started, 'deadline_monotonic': deadline, 'product_acceptance': False})
    records = []; versions = {}
    environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'HOME': str(work), 'TMPDIR': str(work)}
    try:
        for name in QUERIES:
            input_check()
            need(clock() + 3 < deadline, 'Tool observation deadline before launch')
            need(selected_rows(b, selected, checkout) == rows, 'Selected tool bytes changed')
            r.row(python, executable=True, host_role='python')
            need(sha(b.read_file(b.__file__, 131072)) == BRIDGE_SHA, 'Bridge changed')
            b.load_helpers(helper_root)
            need(clock() + 3 < deadline, 'Tool observation deadline after readback')
            request = {'schema_version': 1, 'argv': [rows[name]['path'], '--version'], 'cwd': str(work)}
            request_path = evidence / (name + '-command.json'); r.save(request_path, request)
            receipt = supervisor.capture(str(request_path), sha(request_path.read_bytes()),
                python=python['path'], environment=environment, output=str(evidence / (name + '-capture')),
                supervisor_sha256=r.SUPERVISOR_SHA, raw_limit=65536, cleanup_reserve=2.0,
                deadline=min(deadline, clock() + 30))
            r.captured_ok(receipt)
            need(receipt.get('argv') == request['argv']
                 and receipt.get('request_sha256') == sha(r.canonical(request))
                 and receipt.get('supervisor_sha256') == r.SUPERVISOR_SHA, 'Version command receipt differs')
            versions[name] = version(name, sealed_streams(b, evidence, name, receipt)['stdout'])
            need(clock() < deadline, 'Tool observation deadline after captured output')
            records.append(receipt)
        input_check()
        need(selected_rows(b, selected, checkout) == rows, 'Final tool bytes changed')
        r.row(python, executable=True, host_role='python'); host_matches(snapshot)
        for name, receipt in zip(QUERIES, records):
            sealed_streams(b, evidence, name, receipt)
        need(clock() < deadline, 'Tool observation deadline after final readback')
        updated = copy.deepcopy(snapshot); updated['tools'] = rows
        report = {'status': 'EXISTING_TOOL_VERSIONS_OBSERVED_NOT_ACCEPTANCE',
                  'input_snapshot_sha256': sha(b.canonical(snapshot)), 'snapshot_sha256': sha(b.canonical(updated)),
                  'tools': rows, 'versions': versions, 'records': records, 'product_acceptance': False,
                  'sandbox_version_basis': snapshot['os'], 'scripts_version_basis': 'exact file bytes',
                  'tool_dependencies_or_sdk_capability_proven': False,
                  'completion_requires_successful_tool_exit': True}
        r.save(evidence / 'SNAPSHOT.json', updated); r.save(evidence / 'RESULT.json', report)
        need(clock() < deadline, 'Tool observation deadline after result writes')
        return report
    except BaseException as error:
        try:
            r.save(evidence / 'HOLD.json', {'status': 'HOLD_TOOL_OBSERVATION', 'error_type': type(error).__name__,
                   'completed_version_queries': list(versions), 'product_acceptance': False})
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['verify', 'observe'], default='verify')
    for key in ('runtime-bridge', 'helper-root', 'bundle', 'bundle-sha256', 'output'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.action == 'verify':
        print('{"status":"TOOL_OBSERVER_NOT_EXECUTED","product_acceptance":false}'); return 0
    started = time.monotonic()
    b = load_bridge(args.runtime_bridge)
    raw = b.read_file(args.bundle, b.MAX_JSON); need(sha(raw) == b.checksum(args.bundle_sha256), 'Bundle SHA differs')
    bundle = b.decode(raw)
    need(type(bundle) is dict and set(bundle) == {'snapshot', 'selected_tools', 'checkout'}, 'Exact observation bundle')
    snapshot = b.bound_json(bundle['snapshot'])
    def input_check():
        need(b.read_file(args.bundle, b.MAX_JSON) == raw and b.bound_json(bundle['snapshot']) == snapshot, 'Observation inputs changed')
    result = observe(b, args.helper_root, snapshot, bundle['selected_tools'], bundle['checkout'], args.output, input_check=input_check, started=started)
    print(b.canonical(result).decode(), end=''); return 0


if __name__ == '__main__':
    raise SystemExit(main())
