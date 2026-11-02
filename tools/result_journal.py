"""Engine-independent record retention for a prospective matched workload.

This module does not import a database or authorize an evaluation. Call timers
exclude observers, result serialization and journaling. Those activities can
still affect later samples; any real protocol must declare them explicitly.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback


def failure(error):
    value = {'type':type(error).__name__, 'message':str(error),
             'traceback':''.join(traceback.format_exception(error))}
    fields = ('code', 'message', 'detail', 'recoverable', 'suggested_action', 'operation_outcome', 'sdk_reason')
    value['structured_fields'] = {name:getattr(error, name) for name in fields if hasattr(error, name)}
    # Cleanup groups and allocation failures can retain a core error deeper in
    # the chain. Keep each unique exception and all its links, including cycles.
    pending, identities, nodes = [error], {id(error):0}, []

    def reference(item):
        if item is None:
            return None
        key = id(item)
        if key not in identities:
            identities[key] = len(pending)
            pending.append(item)
        return identities[key]

    for item in pending:
        nodes.append({'type':type(item).__name__, 'message':str(item),
                      'structured_fields':{name:getattr(item, name) for name in fields if hasattr(item, name)},
                      'cause':reference(item.__cause__), 'context':reference(item.__context__),
                      'suppress_context':item.__suppress_context__,
                      'members':[reference(child) for child in item.exceptions]
                          if isinstance(item, BaseExceptionGroup) else None})
    value['exception_graph'] = {'root':0, 'nodes':nodes}
    return value


class ResultJournal:
    def __init__(self, output, identity, observer, clock=time.perf_counter_ns):
        self.output = Path(output)
        self.output.mkdir()
        (self.output / 'payloads').mkdir()
        self.stream = (self.output / 'events.jsonl').open('x', encoding='utf-8')
        self.observer = observer
        self.clock = clock
        self.sequence = 0
        self.retained = []
        try:
            self.append({'event':'identity', 'identity':identity,
                         'timing_scope':'callable only; observers, serialization and journal IO excluded',
                         'memory_scope':'raw overlapping snapshots; no exclusive engine allocation attribution'})
        except BaseException:
            self.stream.close()
            raise

    def append(self, row):
        self.stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def observe(self):
        try:
            return {'ok':self.observer()}
        except Exception as error:
            return {'unavailable':failure(error)}

    def payload(self, value):
        digest = hashlib.sha256()
        size = 0
        handle, filename = tempfile.mkstemp(prefix='unfinished-', dir=self.output / 'payloads')
        temporary = Path(filename)
        # A serialization/IO failure deliberately leaves this partial file.
        with os.fdopen(handle, 'wb') as stream:
            for fragment in json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False).iterencode(value):
                data = fragment.encode('utf-8')
                digest.update(data)
                size += len(data)
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        name = digest.hexdigest() + '.json'
        destination = temporary.parent / name
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = hashlib.sha256()
            with destination.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):
                    existing.update(block)
            if existing.hexdigest() != digest.hexdigest() or destination.stat().st_size != size:
                raise ValueError('existing payload does not match its content address')
        temporary.unlink()  # Only this invocation's temporary link is removed.
        return {'relative_path':'payloads/' + name, 'sha256':digest.hexdigest(), 'bytes':size}

    def measure(self, label, call, retain=False):
        self.sequence += 1
        sequence = self.sequence
        before = self.observe()
        self.append({'event':'started', 'sequence':sequence, 'label':label, 'before':before})
        row = {'event':'completed', 'sequence':sequence, 'label':label,
               'elapsed_ns':None, 'result_retained_by_harness':False,
               'exception_retained_for_propagation':False}
        errors = []
        result = None
        caught = None
        released = False
        started = self.clock()
        try:
            result = call()
        except BaseException as error:
            caught = error
        row['call_outcome'] = 'returned' if caught is None else 'raised'
        row['exception_retained_for_propagation'] = caught is not None

        # Every post-call stage is independent: a clock or observer failure
        # must not prevent the one attempt to preserve a known call result.
        try:
            row['elapsed_ns'] = self.clock() - started
            if row['elapsed_ns'] < 0:
                raise ValueError('monotonic clock moved backwards')
        except BaseException as error:
            errors.append(('clock', error))
        try:
            row['after_call'] = self.observe()
        except BaseException as error:
            errors.append(('after_call_observer', error))
        try:
            row['payload'] = self.payload({'ok':result} if caught is None else {'error':failure(caught)})
        except BaseException as error:
            errors.append(('payload', error))
        if retain and caught is None:
            try:
                self.retained.append(result)
                row['result_retained_by_harness'] = True
            except BaseException as error:
                errors.append(('retention', error))
        if not errors:
            result = None
            released = True
            try:
                gc.collect()
                row['after_release_or_retention'] = self.observe()
            except BaseException as error:
                errors.append(('after_release_observer', error))
        else:
            row['after_release_or_retention'] = {'not_taken':'post-call tool failure; known result held for propagation'}
        row['result_retained_for_error_propagation'] = caught is None and not released
        row['outcome'] = ('evidence_error' if any(stage in ('payload', 'retention') for stage, _ in errors)
                          else 'measurement_error' if errors else 'error' if caught is not None else 'ok')
        try:
            for stage, error in errors:
                key = 'evidence_error' if stage in ('payload', 'retention') else 'measurement_error'
                row.setdefault(key, failure(error))
            if errors:
                row['tool_failures'] = [{'stage':stage, 'error':failure(error)} for stage, error in errors]
            self.append(row)
        except BaseException as error:
            # Do not retry a possibly partially written event or a callable.
            errors.append(('journal', error))
            row['outcome'] = 'evidence_error'
        if errors:
            original = ([caught] if caught is not None else []) + [error for _, error in errors]
            error = original[0] if len(original) == 1 else BaseExceptionGroup('call or evidence retention failed', original)
            # If disk persistence failed, the supervisor can recover the exact
            # value without rerunning the operation. Otherwise call_record
            # locates the complete persisted payload, even after value release.
            error.call_returned = caught is None
            error.call_result_available = caught is None and not released
            if error.call_result_available:
                error.call_result = result
            error.call_record = row
            if len(original) == 1:
                # Re-raising this existing exception must preserve its cause
                # and context. Only the newly created group suppresses context.
                raise error
            raise error from None
        if caught is not None:
            raise caught
        return row

    def release_retained(self):
        count = len(self.retained)
        self.retained.clear()
        gc.collect()
        self.append({'event':'retained_released', 'objects':count, 'observation':self.observe()})

    def close(self):
        self.stream.close()
