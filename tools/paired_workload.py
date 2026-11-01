"""Supervise declared Python/Rust workload groups with cumulative budgets.

A plan hash binds bytes, not approval. No SDK is imported in this supervisor.
Complete execution is not numerical equivalence, a performance win or acceptance.
"""
from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import python_workload as python_worker
from result_journal import failure


class ProcessFailure(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.process_record = record
        # Shared failure reporting preserves this known CHILD invocation state;
        # it never infers an SDK transaction outcome from a process exit.
        self.call_record = {'kind':'supervised_child_process', 'record':record}
        self.call_returned = record.get('exit_code') is not None
        self.call_result_available = True
        self.call_result = record


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError('positive integer required: ' + name)
    return value


class Budget:
    """One overall clock and one accumulated ledger for each language."""
    def __init__(self, overall, per_language, clock=time.monotonic):
        self.overall = positive(overall, 'overall budget')
        self.per_language = positive(per_language, 'language budget')
        self.clock = clock
        self.start = clock()
        self.used = {'python': 0.0, 'rust': 0.0}

    def deadline(self, language, unit_seconds):
        now = self.clock()
        remaining = min(self.start + self.overall - now,
                        self.per_language - self.used[language],
                        positive(unit_seconds, 'unit budget'))
        if remaining <= 0:
            raise TimeoutError('cumulative workload budget exhausted before next unit')
        return now, now + remaining

    def finish(self, language, started, deadline):
        now = self.clock()
        self.used[language] += now - started
        if now > deadline:
            raise TimeoutError('unit including supervision and evidence exceeded remaining budget')
        return now


def schedule(plan):
    for external_round in range(1, positive(plan['rounds'], 'rounds') + 1):
        languages = ('python', 'rust') if external_round % 2 else ('rust', 'python')
        for language in languages:
            for profile, scale, mode in itertools.product(plan['profiles'], plan['scales'], plan['modes']):
                yield {'external_round': external_round, 'language': language,
                       'profile': profile, 'scale': scale, 'mode': mode, 'internal_round': 0,
                       'id': f'r{external_round}-{language}-{profile}-{scale}-{mode}'}


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_group(process, record, grace=1.0):
    """Signal only the new process group created for this invocation."""
    errors = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if group_exists(process.pid):
                os.killpg(process.pid, sig)
                record['cleanup_signals'].append(signal.Signals(sig).name)
        except BaseException as error:
            errors.append(error)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
        except BaseException as error:
            errors.append(error)
        # TERM must be followed by KILL when descendants remain after leader exit.
        if not group_exists(process.pid):
            break
        if sig == signal.SIGTERM:
            time.sleep(grace)
    try:
        process.wait(timeout=grace)
    except BaseException as error:
        errors.append(error)
    if errors:
        raise BaseExceptionGroup('failed to clean the owned process group', errors)


def run_process(argv, output, deadline, *, cwd, env):
    output = Path(output)
    output.mkdir()
    record = {'argv': list(argv), 'cwd': str(cwd), 'started_monotonic': time.monotonic(),
              'exit_code': None, 'timed_out': False, 'cleanup_signals': [],
              'status': 'starting', 'pid': None}
    python_worker.write_json(output / 'process.json', record)
    process = None
    errors = []
    try:
        with (output / 'stdout').open('xb') as stdout, (output / 'stderr').open('xb') as stderr:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('budget exhausted before child start')
                process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                record.update(pid=process.pid, status='running')
                python_worker.write_json(output / 'process.json', record)
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired as error:
                    record['timed_out'] = True
                    errors.append(error)
                record['exit_code'] = process.returncode
                if process.returncode is not None and group_exists(process.pid):
                    errors.append(RuntimeError('child leader exited while its process group remains'))
                if process.returncode is not None and process.returncode != 0:
                    errors.append(RuntimeError('child returned nonzero'))
            except BaseException as error:
                errors.append(error)
            finally:
                if process is not None:
                    try:
                        stop_group(process, record)
                    except BaseException as error:
                        errors.append(error)
                    record['exit_code'] = process.returncode
    except BaseException as error:
        errors.append(error)
    finally:
        record['finished_monotonic'] = None
        try:
            record['finished_monotonic'] = time.monotonic()
        except BaseException as error:
            errors.append(error)
        if (record['finished_monotonic'] is not None
                and record['finished_monotonic'] > deadline and not record['timed_out']):
            errors.append(TimeoutError('child supervision exceeded its unit deadline'))
        record['status'] = 'failed' if errors else 'completed'
        if errors:
            try:
                record['errors'] = [failure(error) for error in list(errors)]
            except BaseException as error:
                errors.append(error)
                record['diagnostic_formatting_failed'] = True
        try:
            python_worker.write_json(output / 'process.json', record)
        except BaseException as error:
            errors.append(error)
    if errors:
        record['status'] = 'failed'
        error = ProcessFailure('first failed process; no retry', record)
        raise error from BaseExceptionGroup('process and cleanup errors', errors)
    return record


def labels_for_import(mode, facts, batch_rows):
    if mode == 'manifest':
        return ['import-manifest']
    return [f'batch-{field}-{i}' for field in ('documents', 'chunks', 'embeddings')
            for i in range(math.ceil(facts[field + '.jsonl']['rows'] / batch_rows))]


def expected_labels(plan, unit, facts, queries, *, rust_reader=False):
    ids = [q['id'] for q in queries if q['profile'] == unit['profile']]
    if rust_reader:
        return [label for q in ids for label in
                ([f'new-handle-{q}'] + [f'warmup-{q}-{i}' for i in range(plan['warmup_repeats'])]
                 + [f'warm-{q}-{i}' for i in range(plan['query_repeats'])])]
    labels = labels_for_import(unit['mode'], facts['files'][str(unit['scale'])], plan['batch_rows'])
    labels += [f'import-ready-{q}' for q in ids] + ['import-stats']
    if unit['language'] == 'rust':
        return labels
    for q in ids:
        labels += [f'new-handle-{q}'] + [f'warmup-{q}-{i}' for i in range(plan['warmup_repeats'])]
        labels += [f'warm-{q}-{i}' for i in range(plan['query_repeats'])]
        labels += [f'retained-{q}-{i}' for i in range(plan['retain_repeats'])]
    if unit['scale'] == plan['lifecycle_scale']:
        for i in range(plan['lifecycle_cycles']):
            labels += [f'lifecycle-search-{i}', f'lifecycle-stats-{i}']
        labels += labels_for_import(unit['mode'], facts['files']['ownership'], plan['batch_rows'])
        labels += ['ownership-vector-query', 'ownership-stats', 'ownership-long-read']
    return labels


def regular(path, root):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('evidence file escapes its declared directory')
    return path


def rust_receipt(output, labels, identity):
    events = regular(output / 'events.jsonl', output)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    if not rows or rows[0].get('event') != 'identity' or rows[-1].get('event') != 'case_complete':
        raise ValueError('Rust journal is not complete')
    actual = rows[0]['identity']
    if any(actual.get(k) != v for k, v in identity.items()):
        raise ValueError('Rust journal has different runtime/request identity')
    started = [(r.get('sequence'), r.get('label')) for r in rows if r.get('event') == 'started']
    completed = [r for r in rows if r.get('event') == 'completed']
    if (started != [(r.get('sequence'), r.get('label')) for r in completed]
            or [r.get('label') for r in completed] != labels
            or [r.get('sequence') for r in completed] != [str(i) for i in range(1, len(labels)+1)]):
        raise ValueError('Rust call inventory is incomplete, reordered or duplicated')
    verified = []
    for index, row in enumerate(completed, 1):
        reference = row.get('payload')
        if (row.get('outcome') != 'ok' or type(reference) is not dict
                or set(reference) != {'relative_path', 'bytes'}
                or reference['relative_path'] != f'payloads/{index:08}.json'
                or type(reference['bytes']) is not str or not reference['bytes'].isdigit()
                or type(row.get('elapsed_ns')) is not str or not row['elapsed_ns'].isdigit()):
            raise ValueError('Rust result lacks a complete matching success envelope')
        path = regular(output / reference['relative_path'], output)
        if path.stat().st_size != int(reference['bytes']):
            raise ValueError('Rust result length differs')
        value = json.loads(path.read_bytes())
        if type(value) is not dict or 'ok' not in value or not set(value) <= {'ok', 'operation_outcome'}:
            raise ValueError('Rust result envelope is invalid')
        verified.append({'label': row['label'], 'path': reference['relative_path'],
                         'sha256': python_worker.sha(path), 'bytes': path.stat().st_size})
    return {'journal_sha256': python_worker.sha(events), 'payloads': verified}


def verify_counts(output, facts):
    rows = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    stats = [r for r in rows if r.get('event') == 'completed' and r.get('label') == 'import-stats']
    if len(stats) != 1:
        raise ValueError('missing unique import stats')
    value = json.loads((output / stats[0]['payload']['relative_path']).read_bytes())['ok']
    for field, filename in [('document_count','documents.jsonl'), ('chunk_count','chunks.jsonl'),
                            ('embedding_count','embeddings.jsonl')]:
        if type(value[field]) is not int or value[field] != facts[filename]['rows']:
            raise ValueError('actual import count differs from the frozen input')


def validate_python(output, case, plan, labels):
    receipt = python_worker.case_receipt(output, case, plan)
    rows = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    if [r['label'] for r in rows if r['event'] == 'completed'] != labels:
        raise ValueError('Python worker omitted or duplicated a declared API call')
    if case[1] == plan['lifecycle_scale']:
        for event in ('expected_refusal', 'lifecycle_before_refusal', 'lifecycle_closed'):
            if [r.get('cycle') for r in rows if r['event'] == event] != list(range(plan['lifecycle_cycles'])):
                raise ValueError('Python lifecycle inventory is incomplete')
        if sum(r['event'] == 'long_result_valid_after_close' for r in rows) != 1:
            raise ValueError('missing owned long content after close')
    return receipt


def load_inputs(plan_path, plan_sha, envelope_path, envelope_sha, inputs, wheel, python, rust, environment, build_evidence):
    plan = python_worker.protocol(plan_path, plan_sha)
    if python_worker.sha(envelope_path) != envelope_sha:
        raise ValueError('pair envelope SHA differs')
    envelope = json.loads(envelope_path.read_bytes())
    required = {'schema_version', 'plan_sha256', 'global_budget_seconds', 'language_budget_seconds',
                'python_sha256', 'rust_sha256', 'environment_sha256', 'tool_sha256', 'rust_build_receipt'}
    if type(envelope) is not dict or set(envelope) != required or type(envelope['schema_version']) is not int or envelope['schema_version'] != 1:
        raise ValueError('unsupported pair envelope')
    if envelope['plan_sha256'] != plan_sha:
        raise ValueError('pair envelope references another plan')
    for field in ('global_budget_seconds', 'language_budget_seconds'):
        positive(envelope[field], field)
    for path, field in [(python,'python_sha256'), (rust,'rust_sha256'), (environment,'environment_sha256')]:
        if path.is_symlink() and path not in (python,):
            raise ValueError('declared artifact must be a regular file')
        if python_worker.sha(path.resolve(strict=True)) != envelope[field]:
            raise ValueError('actual selected artifact differs: ' + field)
    tools = {p.name:python_worker.sha(p) for p in
             [Path(__file__), Path(python_worker.__file__), Path(__file__).with_name('result_journal.py'),
              Path(__file__).with_name('memory_observer.py')]}
    if envelope['tool_sha256'] != tools:
        raise ValueError('actual paired tools differ from selected envelope')
    receipt = envelope['rust_build_receipt']
    if (type(receipt) is not dict or receipt.get('binary_sha256') != envelope['rust_sha256']
            or receipt.get('source_input_sha256') != plan['source_input_sha256']
            or receipt.get('engine_version') != plan['engine_version']
            or not isinstance(receipt.get('build_evidence_sha256'), str)
            or len(receipt['build_evidence_sha256']) != 64
            or python_worker.sha(build_evidence) != receipt['build_evidence_sha256']):
        raise ValueError('Rust build receipt is not bound to this binary/source')
    # This is a reviewed provenance declaration, not compiler attestation.
    observed = json.loads(environment.read_bytes())
    if any(observed.get(k) != v for k, v in {'system':platform.system(), 'release':platform.release(),
                                           'machine':platform.machine()}.items()):
        raise ValueError('actual host differs from selected environment')
    code = ('import sys,sysconfig,struct,json; print(json.dumps({'
            '"version":sys.version,"implementation":sys.implementation.name,'
            '"soabi":sysconfig.get_config_var("SOABI"),'
            '"bits":struct.calcsize("P")*8,'
            '"gil_disabled":bool(sysconfig.get_config_var("Py_GIL_DISABLED")),'
            '"gil_enabled":getattr(sys,"_is_gil_enabled",lambda:True)()}))')
    clean_env = {k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME'}}
    if any(k.startswith(('LD_', 'DYLD_')) for k in clean_env):
        raise ValueError('remove loader injection before interpreter identity probe')
    probe = subprocess.run([str(python), '-I', '-B', '-c', code],
                           env=clean_env, capture_output=True, check=True, timeout=10)
    runtime = json.loads(probe.stdout)
    if (observed.get('python') != runtime or runtime['implementation'] != 'cpython'
            or runtime['bits'] != 64 or runtime['gil_disabled'] or not runtime['gil_enabled']):
        raise ValueError('actual interpreter differs from selected regular-GIL environment')
    if envelope['language_budget_seconds'] != plan['total_timeout_seconds']:
        raise ValueError('cumulative language budget must match the selected workload plan')
    if python_worker.sha(wheel) != plan['wheel_sha256']:
        raise ValueError('selected Python wheel differs')
    facts = python_worker.input_files(inputs, plan)
    queries = json.loads((inputs / 'scale-queries.json').read_bytes())
    if any(not [q for q in queries if q['profile'] == profile] for profile in plan['profiles']):
        raise ValueError('a selected profile has no requests')
    return plan, envelope, facts, queries


def verify_frozen(files):
    for path, expected in files.items():
        if python_worker.sha(Path(path)) != expected:
            raise ValueError('frozen execution input/tool changed: ' + path)


def verify_invocations(bindings):
    for item in bindings.values():
        path = Path(item['invocation'])
        if (str(path.resolve(strict=True)) != item['resolved']
                or python_worker.sha(path) != item['sha256']):
            raise ValueError('selected executable invocation/target changed')


def execute(plan, envelope, facts, queries, *, inputs, wheel, python, rust, output,
            frozen_files=None, input_documents=None, invocation_bindings=None):
    if os.name != 'posix':
        raise ValueError('paired supervisor requires POSIX process groups')
    if any(k.startswith(('LD_', 'DYLD_')) for k in os.environ) or os.environ.get('COFFERDB_TOOL_FUNCTIONAL_CONTROL_ONLY'):
        raise ValueError('remove loader injection and functional-control overrides before measurement')
    output = python_worker.checked_output(output, inputs)
    if output.resolve().is_relative_to(wheel.parent.resolve()):
        raise ValueError('output must be outside the wheel directory')
    output.mkdir(parents=True)
    derived = dict(plan, rounds=1)
    python_worker.write_json(output / 'python-subplan.json', derived)
    derived_sha = python_worker.sha(output / 'python-subplan.json')
    frozen_files = frozen_files or {}
    invocation_bindings = invocation_bindings or {}
    units = list(schedule(plan))
    report = {'kind':'PAIRED_EXECUTION_NOT_ACCEPTANCE', 'complete':False,
              'plan':plan, 'envelope':envelope, 'derived_plan_sha256':derived_sha,
              'frozen_file_sha256':frozen_files, 'invocation_bindings':invocation_bindings,
              'units':[dict(u, state='Not Run', stages=[]) for u in units],
              'source_scope':'SDK/wheel/source and external supervisor tools are separately frozen',
              'comparison_scope':'different language process/stage coverage; no pure-binding subtraction'}
    python_worker.write_json(output / 'run.json', report)
    verify_frozen(frozen_files)
    verify_invocations(invocation_bindings)
    for name, path in (input_documents or {}).items():
        if Path(name).name != name:
            raise ValueError('execution document name must be local')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != frozen_files[str(path.absolute())]:
            raise ValueError('selected execution document changed')
        with (output / name).open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    env = {k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME'}}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    budget = Budget(envelope['global_budget_seconds'], envelope['language_budget_seconds'])
    try:
        for row in report['units']:
            language = row['language']
            started, deadline = budget.deadline(language, plan['case_timeout_seconds'])
            row['state'] = 'running'
            row['deadline_monotonic'] = deadline
            python_worker.write_json(output / 'run.json', report)
            folder = output / row['id']
            folder.mkdir()
            unit_errors = []
            try:
                verify_frozen(frozen_files)
                verify_invocations(invocation_bindings)
                python_worker.input_files(inputs, plan)
                if language == 'python':
                    case = (row['profile'], row['scale'], row['mode'], 0)
                    result = folder / 'result'
                    argv = [str(python), '-B', str(Path(python_worker.__file__).resolve()),
                            '--worker', '-'.join(map(str,case)), '--protocol', str(output/'python-subplan.json'),
                            '--protocol-sha256', derived_sha, '--inputs', str(inputs),
                            '--wheel', str(wheel), '--output', str(result)]
                    row['stages'].append(run_process(argv, folder/'process', deadline, cwd=output, env=env))
                    row['evidence'] = validate_python(result, case, derived,
                                                     expected_labels(plan,row,facts,queries))
                    verify_counts(result, facts['files'][str(row['scale'])])
                else:
                    database = folder / 'data.coffer'
                    collection = json.loads((inputs/str(row['scale'])/'import_manifest.json').read_bytes())['collection']
                    common_identity = {'engine_version':plan['engine_version'], 'profile':row['profile'],
                                       'functional_control_only':False}
                    for stage in ('write','query'):
                        result = folder / (stage+'-result')
                        if stage == 'write':
                            argv = [str(rust),'write',str(database),str(inputs/str(row['scale'])),
                                    str(inputs/'scale-queries.json'),str(result),row['profile'],row['mode'],str(plan['batch_rows'])]
                            identity = dict(common_identity, mode=row['mode'])
                        else:
                            before = python_worker.sha(database)
                            argv = [str(rust),str(database),collection,str(inputs/'scale-queries.json'),
                                    str(result),row['profile'],str(plan['warmup_repeats']),str(plan['query_repeats'])]
                            identity = dict(common_identity, database=str(database))
                        process_record = run_process(argv, folder/(stage+'-process'), deadline, cwd=output, env=env)
                        row['stages'].append(process_record)
                        row[stage+'_evidence'] = rust_receipt(result,
                            expected_labels(plan,row,facts,queries,rust_reader=stage=='query'),identity)
                        if stage == 'write':
                            verify_counts(result, facts['files'][str(row['scale'])])
                        elif python_worker.sha(database) != before:
                            raise ValueError('Rust read-only query changed database bytes')
                verify_frozen(frozen_files)
                verify_invocations(invocation_bindings)
                python_worker.input_files(inputs, plan)
            except BaseException as error:
                unit_errors.append(error)
            row['state'] = 'incomplete' if unit_errors else 'completed'
            try:
                python_worker.write_json(output / 'run.json', report)
            except BaseException as error:
                unit_errors.append(error)
            try:
                last_accounted = budget.finish(language, started, deadline)
            except BaseException as error:
                unit_errors.append(error)
            report['language_elapsed_seconds'] = dict(budget.used)
            if unit_errors:
                row['state'] = 'incomplete'
                if len(unit_errors) == 1:
                    raise unit_errors[0]
                raise BaseExceptionGroup('workload unit and supervision failed', unit_errors)
        report['complete'] = True
        report['completion_scope'] = 'provisional until supervisor exits zero after final receipt budget check'
        python_worker.write_json(output / 'run.json', report)
        # Charge the final receipt IO to the last unit/language under the SAME
        # deadline. Failure revokes complete, rather than resetting a timer.
        budget.finish(language, last_accounted, deadline)
        report['language_elapsed_seconds_after_closeout'] = dict(budget.used)
        return report
    except BaseException as error:
        if report['complete'] and report['units']:
            report['units'][-1]['state'] = 'incomplete'
        report['complete'] = False
        report['failure'] = python_worker.failure_record(error)
        if isinstance(error, ProcessFailure):
            report['failed_process'] = error.process_record
        for row in report['units']:
            if row['state'] == 'running':
                row['state'] = 'incomplete'
        report['language_elapsed_seconds'] = dict(budget.used)
        try:
            python_worker.write_json(output / 'run.json', report)
        except BaseException as evidence_error:
            raise BaseExceptionGroup('workload and final receipt failed', [error,evidence_error]) from None
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','envelope','inputs','wheel','python','rust','environment','build-evidence','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--envelope-sha256',required=True)
    args = p.parse_args()
    values = load_inputs(args.protocol,args.protocol_sha256,args.envelope,args.envelope_sha256,
                         args.inputs,args.wheel,args.python,args.rust,args.environment,args.build_evidence)
    plan, envelope = values[:2]
    # Keep approved expected hashes, never re-baseline by hashing changed files.
    frozen_files = {str(args.protocol.absolute()):args.protocol_sha256,
                    str(args.envelope.absolute()):args.envelope_sha256,
                    str(args.environment.absolute()):envelope['environment_sha256'],
                    str(args.build_evidence.absolute()):envelope['rust_build_receipt']['build_evidence_sha256'],
                    str(args.python.resolve()):envelope['python_sha256'],
                    str(args.python.absolute()):envelope['python_sha256'],
                    str(args.rust.absolute()):envelope['rust_sha256'],
                    str(args.wheel.absolute()):plan['wheel_sha256']}
    for name, expected in envelope['tool_sha256'].items():
        frozen_files[str(Path(__file__).resolve().with_name(name))] = expected
    bindings = {name:{'invocation':str(path.absolute()), 'resolved':str(path.resolve(strict=True)),
                      'sha256':envelope[name+'_sha256']}
                for name,path in [('python',args.python),('rust',args.rust)]}
    documents = {'selected-protocol.json':args.protocol, 'selected-envelope.json':args.envelope,
                 'selected-environment.json':args.environment, 'selected-rust-build.json':args.build_evidence}
    report = execute(*values,inputs=args.inputs.resolve(),wheel=args.wheel.resolve(),
            python=args.python.absolute(),rust=args.rust.absolute(),output=args.output.absolute(),
            frozen_files=frozen_files,input_documents=documents,invocation_bindings=bindings)
    print(json.dumps({'kind':'SUPERVISOR_EXIT_RECEIPT', 'complete':report['complete'],
                      'language_elapsed_seconds_after_closeout':report['language_elapsed_seconds_after_closeout'],
                      'run_json_sha256':python_worker.sha(args.output/'run.json')},sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if isinstance(error, ProcessFailure):
            print(json.dumps({'process_record':error.process_record},ensure_ascii=True),file=sys.stderr)
        python_worker.report_failure(error,sys.stderr)
        raise
