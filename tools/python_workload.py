"""Run declared Python workloads with complete results and explicit observations.

A protocol hash binds the supplied plan; it is not evidence of review or
acceptance. Obtain review before a formal run. This tool never uploads results.
Run supervision uses a separate process and a first-error stop policy.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.resources
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
import zipfile

from memory_observer import ProcessMemoryObserver
from result_journal import ResultJournal, failure

PROFILES = ('expanded', 'balanced', 'constrained')
SCALES = (1000, 10000, 100000)
FIELDS = {
    'schema_version', 'source_input_sha256', 'wheel_sha256',
    'sdk_version', 'engine_version', 'input_manifest_sha256',
    'profiles', 'scales', 'modes', 'rounds', 'query_repeats',
    'warmup_repeats', 'batch_rows', 'retain_repeats', 'lifecycle_cycles',
    'lifecycle_scale', 'trace_python', 'case_timeout_seconds',
    'total_timeout_seconds',
    'tool_sha256',
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, indent=2) + '\n').encode()
    temporary = path.with_name(path.name + '.incomplete')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def protocol(path, expected):
    if sha(path) != expected:
        raise ValueError('protocol hash differs from the selected plan')
    value = json.loads(Path(path).read_bytes())
    if (type(value) is not dict or set(value) != FIELDS
            or type(value['schema_version']) is not int or value['schema_version'] != 1):
        raise ValueError('unsupported workload protocol schema')
    for name in ('source_input_sha256', 'wheel_sha256', 'input_manifest_sha256'):
        item = value[name]
        if type(item) is not str or len(item) != 64 or any(c not in '0123456789abcdef' for c in item):
            raise ValueError('invalid SHA256 field: ' + name)
    for name, supported in [('profiles', PROFILES), ('scales', SCALES),
                            ('modes', ('manifest', 'batch'))]:
        items = value[name]
        expected_type = int if name == 'scales' else str
        if (type(items) is not list or not items or any(type(item) is not expected_type for item in items)
                or len(set(items)) != len(items)
                or any(item not in supported for item in items)):
            raise ValueError('invalid workload selection: ' + name)
    for name in ('rounds', 'query_repeats', 'warmup_repeats', 'batch_rows',
                 'retain_repeats', 'lifecycle_cycles', 'case_timeout_seconds',
                 'total_timeout_seconds'):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError('positive integer required: ' + name)
    if (type(value['lifecycle_scale']) is not int or value['lifecycle_scale'] not in value['scales']
            or type(value['trace_python']) is not bool):
        raise ValueError('invalid lifecycle or observation selection')
    for field in ('sdk_version', 'engine_version'):
        if type(value[field]) is not str or not value[field]:
            raise ValueError('version string required: ' + field)
    tools = [Path(__file__), Path(sys.modules[ResultJournal.__module__].__file__),
             Path(sys.modules[ProcessMemoryObserver.__module__].__file__)]
    actual_tools = {tool.name:sha(tool) for tool in tools}
    if value['tool_sha256'] != actual_tools:
        raise ValueError('actual workload tools differ from the selected plan')
    return value


def input_files(root, plan):
    root = root.resolve(strict=True)
    if sha(root / 'input-manifest.json') != plan['input_manifest_sha256']:
        raise ValueError('prepared input manifest differs from the plan')
    facts = json.loads((root / 'input-manifest.json').read_bytes())
    expected = {'1000', '10000', '100000', 'ownership'}
    if set(facts['files']) != expected:
        raise ValueError('input manifest must declare the four prepared cases')
    names = {'documents.jsonl', 'chunks.jsonl', 'embeddings.jsonl', 'import_manifest.json'}
    for case, files in facts['files'].items():
        if set(files) != names:
            raise ValueError('prepared case has an unexpected file set')
        for name, identity in files.items():
            path = root / case / name
            if (path.is_symlink() or not path.resolve().is_relative_to(root)
                    or path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']):
                raise ValueError('prepared input changed: ' + case + '/' + name)
        manifest = json.loads((root / case / 'import_manifest.json').read_bytes())
        if any(manifest.get(field) != field + '.jsonl' for field in ('documents', 'chunks', 'embeddings')):
            raise ValueError('import manifest paths must name the inspected case inputs')
    for name, field in [('scale-queries.json', 'scale_query_sha256'),
                        ('ownership-queries.json', 'ownership_query_sha256')]:
        path = root / name
        if path.is_symlink() or sha(path) != facts[field]:
            raise ValueError('prepared query changed: ' + name)
    return facts


def installed_sdk(plan, wheel):
    if sha(wheel) != plan['wheel_sha256']:
        raise ValueError('selected wheel differs from the plan')
    import cofferdb
    source = json.loads(importlib.resources.files(cofferdb).joinpath('_source.json').read_text())
    if (cofferdb.__version__ != plan['sdk_version']
            or cofferdb.engine_version != plan['engine_version']
            or source['source_input_sha256'] != plan['source_input_sha256']):
        raise ValueError('actual installed SDK/engine/source differs from the plan')
    directory = Path(cofferdb.__file__).resolve().parent
    distribution = importlib.metadata.distribution('cofferdb')
    if Path(distribution.locate_file('cofferdb')).resolve() != directory:
        raise ValueError('imported package differs from its installed distribution')
    with zipfile.ZipFile(wheel) as bundle:
        for name in bundle.namelist():
            if name.startswith('cofferdb/') and not name.endswith('/'):
                actual = directory / name.removeprefix('cofferdb/')
                if actual.is_symlink() or actual.read_bytes() != bundle.read(name):
                    raise ValueError('installed package differs from actual wheel: ' + name)
        if json.loads(bundle.read('cofferdb/_source.json')) != source:
            raise ValueError('wheel and installed static source differ')
    return cofferdb, source


def batches(path, size):
    with path.open(encoding='utf-8') as stream:
        iterator = (json.loads(line) for line in stream)
        while rows := list(itertools.islice(iterator, size)):
            yield rows


def query_kwargs(row):
    return {name: row[name] for name in
            ('text', 'embedding', 'filter', 'weights', 'metadata_boost', 'top_k', 'profile')
            if name in row}


def import_case(sdk, database, folder, collection_name, mode, batch_rows, journal):
    if mode == 'manifest':
        journal.measure('import-manifest', lambda: database.import_jsonl(str(folder / 'import_manifest.json')))
        return database.collection(collection_name)
    manifest = json.loads((folder / 'import_manifest.json').read_bytes())
    config = {
        'name': collection_name, 'embedding_dim': manifest['embedding_dim'],
        'distance_metric': manifest['distance_metric'],
        'embedding_model': manifest['embedding_model'],
        'vectors_normalized': manifest['vectors_normalized'],
    }
    if 'tokenizer' in manifest:
        config['tokenizer'] = manifest['tokenizer']
    collection = database.create_collection(config)
    for field, filename in [('documents', 'documents.jsonl'), ('chunks', 'chunks.jsonl'),
                            ('embeddings', 'embeddings.jsonl')]:
        for index, rows in enumerate(batches(folder / filename, batch_rows)):
            journal.measure('batch-' + field + '-' + str(index),
                            lambda rows=rows, field=field: collection.write_batch(**{field: rows}))
    return collection


def failure_record(error):
    record = {'error': failure(error), 'known_calls': []}
    pending, seen = [error], set()
    for item in pending:
        if id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, 'call_record'):
            value = {'call_returned': item.call_returned, 'record': item.call_record,
                     'result_available': item.call_result_available}
            if item.call_result_available:
                value['result'] = item.call_result
            record['known_calls'].append(value)
        pending.extend(link for link in (item.__cause__, item.__context__) if link is not None)
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
    return record


def record_fatal(journal, error):
    record = failure_record(error)
    try:
        payload = journal.payload(record)
        journal.append({'event': 'case_failure', 'payload': payload, 'complete': False})
    except BaseException as evidence_error:
        raise BaseExceptionGroup('workload and final evidence retention failed', [error, evidence_error]) from None


def report_failure(error, stream):
    # The supervising process already owns this separate output channel. A
    # payload-directory failure must not discard a successful call's known value.
    stream.write('\nCOFFERDB_WORKLOAD_FAILURE_JSON ')
    for fragment in json.JSONEncoder(sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).iterencode(failure_record(error)):
        stream.write(fragment)
    stream.write('\n')
    stream.flush()


def stored(journal, row):
    return json.loads((journal.output / row['payload']['relative_path']).read_bytes())['ok']


def check_counts(value, facts):
    for field, filename in [('document_count', 'documents.jsonl'),
                            ('chunk_count', 'chunks.jsonl'), ('embedding_count', 'embeddings.jsonl')]:
        if value[field] != facts[filename]['rows']:
            raise AssertionError('normal import count differs: ' + field)


def ownership_case(sdk, root, profile, mode, plan, journal):
    folder = root / 'ownership'
    manifest = json.loads((folder / 'import_manifest.json').read_bytes())
    path = journal.output / 'ownership.coffer'
    with (folder / 'chunks.jsonl').open(encoding='utf-8') as stream:
        expected = json.loads(next(stream))
    request = next(row for row in json.loads((root / 'ownership-queries.json').read_bytes())
                   if row.get('profile') == profile)
    with sdk.Database.create(str(path), config={'resource_profile': profile}) as database:
        collection = import_case(sdk, database, folder, manifest['collection'],
                                 mode, plan['batch_rows'], journal)
        journal.measure('ownership-vector-query', lambda: collection.search(**query_kwargs(request)))
        row = journal.measure('ownership-stats', database.stats)
        check_counts(stored(journal, row), json.loads((root / 'input-manifest.json').read_bytes())['files']['ownership'])
        journal.measure('ownership-long-read',
                        lambda: collection.get_chunk_by_external_id(expected['chunk_external_id']), retain=True)
        del collection
    del database
    actual = journal.retained[-1]
    if (actual['chunk']['text'].encode('utf-8') != expected['text'].encode('utf-8')
            or actual['chunk']['source_span'] != expected['source_span']
            or actual['document']['source_uri'] != 'synthetic://cost/owner'):
        raise AssertionError('retained full content/source differs after closing the database')
    del actual, expected
    journal.append({'event': 'long_result_valid_after_close', 'observation': journal.observe()})
    journal.release_retained()


def run_case(sdk, root, plan, case, output, observer):
    profile, scale, mode, repeat = case
    journal = ResultJournal(output, {
        'case': {'profile': profile, 'scale': scale, 'mode': mode, 'round': repeat},
        'source_input_sha256': plan['source_input_sha256'],
        'wheel_sha256': plan['wheel_sha256'], 'python': sys.version,
        'protocol': plan,
        'platform': platform.platform(),
        'cache_scope': 'OS page cache uncontrolled; reopened handle and same-handle warm samples are labelled',
        'timing_scope': 'API call intervals exclude fixture parsing, observations and result IO; ready wall includes them',
        'memory_scope': 'overlapping Python/process allocator/RSS snapshots; not engine-only attribution',
    }, observer)
    path = output / 'data.coffer'
    folder = root / str(scale)
    manifest = json.loads((folder / 'import_manifest.json').read_bytes())
    queries = [row for row in json.loads((root / 'scale-queries.json').read_bytes()) if row['profile'] == profile]
    try:
        ready_started = time.perf_counter_ns()
        with sdk.Database.create(str(path), config={'resource_profile': profile}) as database:
            collection = import_case(sdk, database, folder, manifest['collection'],
                                     mode, plan['batch_rows'], journal)
            # Include all declared routes before calling the imported indexes ready.
            for row in queries:
                journal.measure('import-ready-' + row['id'],
                                lambda row=row: collection.search(**query_kwargs(row)))
            journal.append({'event': 'import_ready', 'wall_ns': time.perf_counter_ns() - ready_started,
                            'includes_observation_and_result_io': True,
                            'observation': journal.observe()})
            row = journal.measure('import-stats', database.stats)
            facts = json.loads((root / 'input-manifest.json').read_bytes())['files'][str(scale)]
            check_counts(stored(journal, row), facts)
            del collection
        closed_hash = sha(path)
        for row in queries:
            with sdk.Database.open(str(path), read_only=True, options={'resource_profile': profile}) as database:
                collection = database.collection(manifest['collection'])
                journal.measure('new-handle-' + row['id'],
                                lambda: collection.search(**query_kwargs(row)))
                for index in range(plan['warmup_repeats']):
                    journal.measure('warmup-' + row['id'] + '-' + str(index),
                                    lambda: collection.search(**query_kwargs(row)))
                for index in range(plan['query_repeats']):
                    journal.measure('warm-' + row['id'] + '-' + str(index),
                                    lambda: collection.search(**query_kwargs(row)))
                for index in range(plan['retain_repeats']):
                    journal.measure('retained-' + row['id'] + '-' + str(index),
                                    lambda: collection.search(**query_kwargs(row)), retain=True)
                del collection
            journal.append({'event': 'closed_with_results_retained', 'observation': journal.observe()})
            journal.release_retained()
            del database
        if sha(path) != closed_hash:
            raise AssertionError('read-only query phases changed the database bytes')
        if scale == plan['lifecycle_scale']:
            with (folder / 'documents.jsonl').open(encoding='utf-8') as stream:
                first_document = json.loads(next(stream))
            for index in range(plan['lifecycle_cycles']):
                with sdk.Database.open(str(path), options={'resource_profile': profile}) as database:
                    collection = database.collection(manifest['collection'])
                    journal.measure('lifecycle-search-' + str(index),
                                    lambda: collection.search(**query_kwargs(queries[0])))
                    before = database.stats()
                    document_before = collection.get_document_by_external_id(first_document['external_id'])
                    journal.append({'event': 'lifecycle_before_refusal', 'cycle': index,
                                    'payload': journal.payload({'ok':before})})
                    try:
                        collection.write_batch(documents=[first_document])
                    except sdk.CofferError as error:
                        if error.code != 'duplicate_external_id' or error.operation_outcome != {'state': 'not_committed', 'confirmed_committed_batches': 0, 'confirmed_committed_records': 0}:
                            raise
                        journal.append({'event': 'expected_refusal', 'cycle': index,
                                        'payload': journal.payload({'error': failure(error)})})
                    else:
                        raise AssertionError('duplicate write unexpectedly succeeded')
                    row = journal.measure('lifecycle-stats-' + str(index), database.stats)
                    after = stored(journal, row)
                    if (after != before or collection.get_document_by_external_id(first_document['external_id']) != document_before):
                        raise AssertionError('rejected duplicate changed data or commit progress')
                    del collection
                del database
                gc.collect()
                journal.append({'event': 'lifecycle_closed', 'cycle': index,
                                'observation': journal.observe()})
            ownership_case(sdk, root, profile, mode, plan, journal)
        journal.append({'event': 'case_complete', 'observation': journal.observe()})
    except BaseException as error:
        record_fatal(journal, error)
        raise
    finally:
        journal.close()


def case_receipt(output, case, plan):
    events = output / 'events.jsonl'
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    expected_case = dict(zip(('profile', 'scale', 'mode', 'round'), case))
    if (not rows or rows[0]['event'] != 'identity'
            or rows[0]['identity']['case'] != expected_case
            or rows[0]['identity']['protocol'] != plan
            or rows[-1]['event'] != 'case_complete'):
        raise ValueError('worker exit lacks its complete matching case journal')
    started = [row for row in rows if row['event'] == 'started']
    completed = [row for row in rows if row['event'] == 'completed']
    if (not started or [(row['sequence'], row['label']) for row in started]
            != [(row['sequence'], row['label']) for row in completed]
            or any(row.get('outcome') != 'ok' or row.get('call_outcome') != 'returned'
                   or type(row.get('elapsed_ns')) is not int or row['elapsed_ns'] < 0
                   or 'measurement_error' in row for row in completed)
            or any(row['event'] == 'case_failure' for row in rows)):
        raise ValueError('worker journal contains an incomplete or failed API call')
    required_payload_events = {'completed': 'ok', 'lifecycle_before_refusal': 'ok',
                               'expected_refusal': 'error'}
    if any(row['event'] in required_payload_events and 'payload' not in row for row in rows):
        raise ValueError('worker event is missing its complete payload')
    verified = {}
    for row in rows:
        if 'payload' not in row:
            continue
        item = row['payload']
        if (type(item) is not dict or set(item) != {'relative_path', 'sha256', 'bytes'}
                or type(item['sha256']) is not str or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])
                or type(item['bytes']) is not int or item['bytes'] < 1
                or item['relative_path'] != 'payloads/' + item['sha256'] + '.json'):
            raise ValueError('invalid complete payload reference in worker journal')
        relative = Path(item['relative_path'])
        target = output / relative
        if relative.is_absolute() or '..' in relative.parts or target.is_symlink() or not target.resolve().is_relative_to(output.resolve()):
            raise ValueError('worker payload escapes its case directory')
        expected = (item['sha256'], item['bytes'])
        if relative.as_posix() in verified and verified[relative.as_posix()]['identity'] != expected:
            raise ValueError('inconsistent content address in worker journal')
        if relative.as_posix() not in verified:
            if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
                raise ValueError('complete worker payload differs from its recorded identity')
            value = json.loads(target.read_bytes())
            if type(value) is not dict:
                raise ValueError('worker payload lacks its result or error envelope')
            verified[relative.as_posix()] = {'identity': expected, 'fields': set(value)}
        required = required_payload_events.get(row['event'])
        if required is not None and verified[relative.as_posix()]['fields'] != {required}:
            raise ValueError('worker payload envelope differs from its event outcome')
    return {'journal_sha256': sha(events), 'api_calls': len(completed),
            'complete_payloads': len(verified)}


def checked_output(output, inputs):
    output = output.absolute()
    if (output.exists() or output.is_symlink()
            or output.resolve().is_relative_to(inputs.resolve())
            or output.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError('choose a new output outside the input and source trees')
    return output


def supervise(plan_path, expected, inputs, wheel, output):
    started = time.monotonic()
    plan = protocol(plan_path, expected)
    inputs = inputs.resolve(strict=True)
    output = checked_output(output, inputs)
    # Verify inputs before starting any database work. Children repeat this readback.
    facts = input_files(inputs, plan)
    if sha(wheel) != plan['wheel_sha256']:
        raise ValueError('selected wheel differs from the plan')
    output.mkdir(parents=True)
    source = Path(__file__).resolve()
    receipt = {'kind': 'DECLARED_PYTHON_WORKLOAD', 'protocol_sha256': expected,
               'protocol': plan, 'input_manifest': facts,
               'tool_sha256': plan['tool_sha256'], 'cases': [], 'complete': False}
    write_json(output / 'run.json', receipt)
    (output / 'protocol.json').write_bytes(plan_path.read_bytes())
    for case in itertools.product(plan['profiles'], plan['scales'], plan['modes'], range(plan['rounds'])):
        name = '-'.join(map(str, case))
        remaining = plan['total_timeout_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            receipt['failure'] = {'kind': 'total_timeout', 'next_case': list(case)}
            write_json(output / 'run.json', receipt)
            raise TimeoutError('overall workload budget exhausted; partial results retained')
        command = [sys.executable, '-B', str(source), '--worker', name,
                   '--protocol', str(output / 'protocol.json'), '--protocol-sha256', expected,
                   '--inputs', str(inputs), '--wheel', str(wheel.resolve()), '--output', str(output / name)]
        env = {key: value for key, value in os.environ.items() if key not in {'PYTHONPATH', 'PYTHONHOME'}}
        receipt['active_case'] = list(case)
        write_json(output / 'run.json', receipt)
        with (output / (name + '.stdout')).open('xb') as stdout, (output / (name + '.stderr')).open('xb') as stderr:
            try:
                result = subprocess.run(command, cwd=output, env=env, stdout=stdout, stderr=stderr,
                                        timeout=min(remaining, plan['case_timeout_seconds']))
                row = {'case': list(case), 'exit_code': result.returncode, 'timed_out': False}
            except subprocess.TimeoutExpired:
                row = {'case': list(case), 'exit_code': None, 'timed_out': True}
            except BaseException as error:
                receipt['failure'] = failure(error)
                write_json(output / 'run.json', receipt)
                raise
        receipt['cases'].append(row)
        receipt['active_case'] = None
        write_json(output / 'run.json', receipt)
        if row['timed_out'] or row['exit_code'] != 0:
            raise RuntimeError('first failed case; no retry, subsequent cases not executed: ' + name)
        try:
            row['evidence'] = case_receipt(output / name, case, plan)
        except BaseException as error:
            receipt['failure'] = failure(error)
            write_json(output / 'run.json', receipt)
            raise
        protocol(plan_path, expected)
    receipt['complete'] = True
    write_json(output / 'run.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.output = checked_output(args.output, args.inputs)
        if args.worker:
            plan = protocol(args.protocol, args.protocol_sha256)
            input_files(args.inputs, plan)
            sdk, _ = installed_sdk(plan, args.wheel)
            profile, scale, mode, repeat = args.worker.split('-')
            case = (profile, int(scale), mode, int(repeat))
            if (profile not in plan['profiles'] or case[1] not in plan['scales']
                    or mode not in plan['modes'] or not 0 <= case[3] < plan['rounds']):
                raise ValueError('worker case is not declared in the plan')
            if plan['trace_python']:
                tracemalloc.start()
            observer = ProcessMemoryObserver()
            run_case(sdk, args.inputs, plan, case, args.output, observer.observe)
        else:
            supervise(args.protocol, args.protocol_sha256, args.inputs, args.wheel, args.output)
    except BaseException as error:
        try:
            report_failure(error, sys.stderr)
        except BaseException:
            # If both storage and stderr fail, keep the original nonzero
            # failure. No complete evidence or uncommitted write is inferred.
            pass
        raise


if __name__ == '__main__':
    main()
