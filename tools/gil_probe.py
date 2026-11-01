"""Bounded host-progress observations for a separately reviewed SDK call.

This module neither imports an SDK nor selects a workload. A busy rejection is
an active-owner observation, not a native-phase timestamp or a response SLA.
Use an external process timeout: Python cannot terminate a stuck observer thread.
"""
from __future__ import annotations

import threading
import time


class ProbeError(RuntimeError):
    """An observation failed; completed SDK results remain in the journal."""


def observe_call(label, call, busy_probe, busy_error, independent_probe, journal,
                 *, interval_seconds=0.001, join_seconds=5.0, sample_limit=4096):
    """Return the journal call record and host-progress observations.

    ``busy_probe`` must use the SAME owner as ``call``. It can race the operation
    before acquisition; a main-call BusyError is a real failed attempt, not a
    reason to retry. ``independent_probe`` must use a separate owner. Neither
    callback may mutate the journal. Ordinary successful polling is not proof.
    """
    if (isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float))
            or not 0 < interval_seconds <= 1
            or isinstance(join_seconds, bool) or not isinstance(join_seconds, (int, float))
            or not 0 < join_seconds <= 60
            or type(sample_limit) is not int or not 1 <= sample_limit <= 100000):
        raise ValueError('invalid bounded observer parameters')
    if not isinstance(busy_error, type) or not issubclass(busy_error, Exception):
        raise TypeError('busy_error must be an SDK exception class')
    started, done, ready = threading.Event(), threading.Event(), threading.Event()
    samples, errors = [], []
    counters = {'busy_rejections': 0, 'successful_polls': 0, 'dropped_samples': 0}
    bounds = {}

    def observer():
        ready.set()
        started.wait()
        try:
            while not done.wait(interval_seconds):
                before = time.perf_counter_ns()
                try:
                    busy_probe()
                except busy_error:
                    rejected = time.perf_counter_ns()
                    counters['busy_rejections'] += 1
                    active_before = not done.is_set()
                    # Actual Python instructions run in this thread, followed by
                    # an independent native call. Preserve both activity checks.
                    marker = sum(range(16))
                    independent = independent_probe()
                    # Save a known independent return before any later clock
                    # or observation can fail. Missing timing never qualifies.
                    row = {'poll_started_ns': before, 'busy_rejected_ns': rejected,
                           'host_progress_finished_ns': None,
                           'active_before_independent': active_before,
                           'active_after_independent': None,
                           'python_marker': marker, 'independent_result': independent}
                    if len(samples) < sample_limit:
                        samples.append(row)
                    else:
                        counters['dropped_samples'] += 1
                    row['host_progress_finished_ns'] = time.perf_counter_ns()
                    row['active_after_independent'] = not done.is_set()
                else:
                    counters['successful_polls'] += 1
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=observer, name='cofferdb-host-progress', daemon=True)
    worker.start()
    if not ready.wait(join_seconds):
        # Wake a late observer so it does not wait forever on the start event.
        done.set()
        started.set()
        worker.join(join_seconds)
        raise ProbeError('observer did not reach its ready barrier; stop this attempt')

    def invoke():
        bounds['call_started_ns'] = time.perf_counter_ns()
        started.set()
        try:
            return call()
        finally:
            # Observation failures must never replace a returned SDK value or
            # its original exception before ResultJournal receives it.
            try:
                bounds['call_finished_ns'] = time.perf_counter_ns()
            except BaseException as error:
                errors.append(error)
            try:
                done.set()
            except BaseException as error:
                errors.append(error)

    result = None
    operation_error = None
    try:
        result = journal.measure(label, invoke)
    except BaseException as error:
        operation_error = error
    finally:
        for signal in (done, started):
            try:
                signal.set()
            except BaseException as error:
                errors.append(error)
        interrupted_join = False
        try:
            worker.join(join_seconds)
        except BaseException as error:
            interrupted_join = True
            errors.append(error)
    try:
        alive = worker.is_alive()
    except BaseException as error:
        alive = True
        errors.append(error)
    # An interrupted join can make is_alive unreliable on some CPython builds.
    # Treat the graph as potentially mutable even if is_alive returns False.
    alive = alive or interrupted_join
    if alive:
        errors.append(ProbeError('observer join timed out; stop the process, do not reuse owners'))
    # A live worker still owns samples/counters. Do not read a concurrently
    # mutating graph as a complete record; this attempt is incomplete.
    if alive:
        observation = {'label': label, 'status': 'incomplete_observer_timeout',
                       'bounds': dict(bounds), 'samples': None, 'counters': None}
    else:
        qualified = [row for row in samples if row['active_before_independent']
                     and row['active_after_independent']
                     and bounds.get('call_started_ns', 0) <= row['poll_started_ns']
                     <= row['busy_rejected_ns'] <= row['host_progress_finished_ns']
                     <= bounds.get('call_finished_ns', -1)]
        status = ('call_failed' if operation_error else
                  'observation_failed' if errors else
                  'incomplete_sample_limit' if counters['dropped_samples'] else
                  'host_progress_observed' if qualified else 'inconclusive_no_active_window')
        observation = {'label': label, 'status': status, 'bounds': dict(bounds),
                       'samples': samples, 'counters': counters,
                       'qualified_samples': len(qualified),
                       'python_switch_interval_seconds': __import__('sys').getswitchinterval(),
                       'interval_seconds': interval_seconds, 'join_seconds': join_seconds,
                       'sample_limit': sample_limit,
                       'scope': 'same-owner busy window; not exclusive native phase or latency SLA'}
        if counters['dropped_samples']:
            errors.append(ProbeError('sample limit reached; attempt is incomplete, no automatic retry'))
    if errors:
        from result_journal import failure
        try:
            observation['observer_errors'] = [failure(error) for error in list(errors)]
        except BaseException as error:
            errors.append(error)
    try:
        reference = journal.payload(observation)
        journal.append({'event': 'gil_observation', 'label': label, 'payload': reference,
                        'status': observation['status']})
    except BaseException as error:
        errors.append(error)
    if operation_error:
        if errors:
            raise BaseExceptionGroup('SDK call and host observation failed',
                                     [operation_error, *errors]) from None
        raise operation_error
    if errors:
        raise BaseExceptionGroup('host observation failed; inspect the completed call journal', errors)
    return result, observation
