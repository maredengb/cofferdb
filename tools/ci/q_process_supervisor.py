"""Capture one frozen command in an owned process group; no acceptance verdict.

The separate controller authorizes the command and supplies the earliest
identity/language/stage deadline. This component never selects SDK queries.
A small sentinel remains the unreaped group leader until the final kill, so
cleanup cannot signal a recycled leader PID after an early command exit.
Known readers must not detach into a different session; this is no OS sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time
import traceback


def require(value, message):
    if not value:
        raise ValueError(message)


def plain(value, *, missing=False):
    selected = Path(value)
    require(type(value) is str and value.startswith('/') and not value.startswith('//')
            and str(selected) == value and '..' not in selected.parts, 'Canonical absolute path required')
    for p in reversed((selected, *selected.parents)):
        try:
            info = p.lstat()
        except FileNotFoundError:
            require(missing, 'Missing fixed path')
            return selected
        require(not stat.S_ISLNK(info.st_mode), 'Unexpected path link')
        if p != selected:
            require(stat.S_ISDIR(info.st_mode), 'Non-directory ancestor')
    return selected


def regular(value):
    selected = plain(value)
    require(stat.S_ISREG(selected.lstat().st_mode), 'Regular input required')
    return selected


def sha_file(selected):
    digest = hashlib.sha256()
    with selected.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def request(value, expected_sha256):
    selected = regular(value)
    require(selected.stat().st_size <= 128 * 1024, 'Command request exceeds byte limit')
    raw = selected.read_bytes()
    require(len(raw) <= 128 * 1024 and hashlib.sha256(raw).hexdigest() == expected_sha256,
            'Frozen command request bytes differ')
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, 'Duplicate command request key')
            result[key] = item
        return result
    document = json.loads(raw, object_pairs_hook=pairs)
    require(type(document) is dict and set(document) == {'schema_version', 'argv', 'cwd'}
            and type(document['schema_version']) is int and document['schema_version'] == 1,
            'Exact frozen command schema required')
    argv = document['argv']
    require(type(argv) is list and argv and all(type(a) is str and '\x00' not in a for a in argv)
            and argv[0].startswith('/') and sum(len(a.encode('utf-8')) + 1 for a in argv) <= 64 * 1024,
            'Bounded literal argv required')
    require(plain(document['cwd']).is_dir(), 'Existing canonical working directory required')
    return document


def exclusive_json(selected, value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
    fd = os.open(selected, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        require(stream.write(raw) == len(raw), 'Incomplete process receipt write')
        stream.flush()
        os.fsync(stream.fileno())


def sentinel(request_path, request_sha, status_fd):
    # Caught handlers keep the leader alive through TERM. exec resets caught
    # dispositions for the real command, unlike SIG_IGN which would persist.
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    def event(value):
        raw = (json.dumps(value, allow_nan=False) + '\n').encode()
        require(len(raw) < 4096 and os.write(status_fd, raw) == len(raw), 'Control event write failed')
    try:
        document = request(request_path, request_sha)
        child = subprocess.Popen(document['argv'], cwd=document['cwd'], stdin=subprocess.DEVNULL,
                                 close_fds=True)
        event({'event': 'spawned', 'pid': child.pid})
        exit_code = child.wait()
        event({'event': 'completed', 'exit_code': exit_code})
    except BaseException:
        traceback.print_exc()
        event({'event': 'sentinel_failure'})
    finally:
        os.close(status_fd)
        sys.stdout.flush(); sys.stderr.flush()
        os.close(1); os.close(2)
    while True:
        signal.pause()  # Only the parent performs final group kill and reap.


class CaptureFailed(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('Frozen command capture failed: ' + str(receipt['failure_reason']))


def capture(request_path, request_sha256, *, python, environment, output,
            deadline, supervisor_sha256, raw_limit=16 * 1024 * 1024, cleanup_reserve=2.0):
    """The caller binds Python/links/environment and approves the exact command.

    raw_limit is a monitored observed-prefix limit, not a disk/cgroup quota.
    A read can exceed it by at most one 4096-byte block. No observed bytes are
    discarded; unread tails after overflow are explicitly reported incomplete.
    """
    previous, pending, restoration_errors = {}, [], []
    result, failure = None, None
    def record_signal(signum, _):
        pending.append(signum)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, record_signal)
        result = _capture(request_path, request_sha256, python=python, environment=environment,
                          output=output, deadline=deadline, supervisor_sha256=supervisor_sha256,
                          raw_limit=raw_limit, cleanup_reserve=cleanup_reserve, pending=pending)
    except BaseException as error:
        failure = error
    finally:
        # Keep deferral through raw hashes and receipt retention, not just reap.
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except BaseException as error:
                restoration_errors.append(error)
    receipt = result if result is not None else getattr(failure, 'receipt', None)
    if receipt is not None:
        finished = time.monotonic()
        handover_reason = ('interrupted' if pending else
                           'signal_handler_restore_failed' if restoration_errors else
                           'deadline_including_signal_restore_exceeded' if finished >= deadline else None)
        receipt['signals_received'] = list(pending)
        receipt['finished_monotonic_after_signal_restore'] = finished
        receipt['elapsed_seconds_after_signal_restore'] = finished - receipt['started_monotonic']
        for error in restoration_errors:
            receipt['infrastructure_errors'].append({'phase': 'signal_handler_restore',
                    'type': type(error).__name__, 'message': str(error)})
        if handover_reason is not None:
            receipt['state'] = 'PROCESS_CAPTURE_FAILED'
            receipt['failure_reason'] = receipt['failure_reason'] or handover_reason
            try:
                exclusive_json(Path(output) / 'handover-receipt.json', receipt)
            except BaseException as error:
                receipt['infrastructure_errors'].append({'phase': 'handover_receipt_retention',
                        'type': type(error).__name__, 'message': str(error)})
                restoration_errors.append(error)
        if failure is not None or handover_reason is not None:
            receipt['state'] = 'PROCESS_CAPTURE_FAILED'
            if restoration_errors:
                errors = ([failure] if failure is not None else []) + restoration_errors
                raise CaptureFailed(receipt) from BaseExceptionGroup('Capture and signal handover failures', errors)
            if failure is not None:
                raise failure
            raise CaptureFailed(receipt)
        return receipt
    if restoration_errors:
        errors = ([failure] if failure is not None else []) + restoration_errors
        raise BaseExceptionGroup('Capture and signal handover failures', errors)
    if failure is not None:
        raise failure
    raise RuntimeError('Capture returned no facts')


def _capture(request_path, request_sha256, *, python, environment, output,
             deadline, supervisor_sha256, raw_limit, cleanup_reserve, pending):
    started = time.monotonic()
    document = request(request_path, request_sha256)
    source = regular(str(Path(__file__)))
    require(sha_file(source) == supervisor_sha256, 'Frozen supervisor source differs')
    require(type(deadline) in (float, int) and math.isfinite(deadline)
            and type(cleanup_reserve) in (float, int) and 0.05 <= cleanup_reserve <= 10
            and started + cleanup_reserve < deadline, 'Positive bound including cleanup required')
    require(type(raw_limit) is int and 0 < raw_limit <= 16 * 1024 * 1024, 'Bounded raw capture limit required')
    require(type(python) is str and python.startswith('/') and '\x00' not in python, 'Frozen Python invocation required')
    require(type(environment) is dict and all(type(k) is str and type(v) is str
            and '\x00' not in k + v and '=' not in k for k, v in environment.items()), 'Literal child environment required')
    require(not any(k.startswith(('LD_', 'DYLD_', 'PYTHON')) or k == '__PYVENV_LAUNCHER__'
                    for k in environment), 'Startup injection environment must be cleared')
    target = plain(output, missing=True)
    require(target.parent.is_dir() and not os.path.lexists(target), 'New owned capture directory required')
    target.mkdir(mode=0o700)
    cache = target / 'sentinel-cache-must-remain-absent'
    argv = [python, '-I', '-B', '-X', 'pycache_prefix=' + str(cache), str(source),
            '_sentinel', request_path, request_sha256]
    streams, counts = {}, {'stdout': 0, 'stderr': 0, 'status': 0}
    child, status_read, status_write = None, None, None
    pipes, raw_fds = [], []
    selector, status_buffer, events = selectors.DefaultSelector(), b'', []
    reason, infrastructure_errors, group_gone = None, [], False
    raw_eof, overflow = {'stdout': False, 'stderr': False}, False
    command_exit, actual_pid = None, None
    def cleanup_operation(operation, phase, category):
        nonlocal reason
        try:
            operation()
        except BaseException as error:
            reason = reason or category
            infrastructure_errors.append({'phase': phase, 'type': type(error).__name__,
                                          'message': str(error)})
    def pump(timeout, *, raw_allowed=True):
        nonlocal status_buffer, command_exit, actual_pid, overflow, reason
        for key, _ in selector.select(max(0.0, timeout)):
            name = key.data
            if (not raw_allowed or overflow) and name != 'status':
                selector.unregister(key.fileobj); key.fileobj.close()
                continue
            block = os.read(key.fd, 4096)
            if not block:
                selector.unregister(key.fileobj); key.fileobj.close()
                if name in raw_eof: raw_eof[name] = True
                continue
            require(streams[name].write(block) == len(block), 'Incomplete observed raw write')
            counts[name] += len(block)
            if name == 'status':
                require(counts[name] <= 16 * 1024, 'Sentinel control stream exceeds limit')
                status_buffer += block
                while b'\n' in status_buffer:
                    line, status_buffer = status_buffer.split(b'\n', 1)
                    value = json.loads(line)
                    if not events:
                        require(set(value) == {'event', 'pid'} and value['event'] == 'spawned'
                                and type(value['pid']) is int and value['pid'] > 0, 'Invalid initial control event')
                        actual_pid = value['pid']
                    else:
                        require(len(events) == 1 and set(value) == {'event', 'exit_code'}
                                and value['event'] == 'completed' and type(value['exit_code']) is int,
                                'Invalid completion control event')
                        command_exit = value['exit_code']
                    events.append(value)
            elif counts['stdout'] + counts['stderr'] > raw_limit:
                overflow = True
                reason = reason or 'raw_output_limit'
    try:
        for name in counts:
            fd = os.open(target / (name + '.raw'), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            raw_fds.append(fd)
            streams[name] = os.fdopen(fd, 'wb')
            raw_fds.remove(fd)
        status_read, status_write = os.pipe()
        argv.append(str(status_write))
        # All validation/output setup consumes the same budget. Never start a
        # fresh command after its cutoff, even if the original start was early.
        if pending:
            reason = 'interrupted_before_spawn'
            raise InterruptedError('Signal received before command start')
        if time.monotonic() >= deadline - cleanup_reserve:
            reason = 'deadline_before_spawn'
            raise TimeoutError('Command budget exhausted before process start')
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env=environment, cwd=document['cwd'], close_fds=True,
                                 pass_fds=(status_write,), start_new_session=True, bufsize=0)
        pipes.extend((child.stdout, child.stderr))
        os.close(status_write); status_write = None
        status_stream = os.fdopen(status_read, 'rb', buffering=0); status_read = None
        pipes.append(status_stream)
        for name, stream in (('stdout', child.stdout), ('stderr', child.stderr), ('status', status_stream)):
            selector.register(stream, selectors.EVENT_READ, name)
        cutoff = deadline - cleanup_reserve
        while reason is None and command_exit is None:
            if pending: reason = 'interrupted'
            elif time.monotonic() >= cutoff: reason = 'command_deadline_for_cleanup'
            elif not selector.get_map(): reason = 'sentinel_incomplete'
            else: pump(min(0.05, cutoff - time.monotonic()))
    except BaseException as error:
        reason = reason or 'supervisor_error'
        infrastructure_errors.append({'type': type(error).__name__, 'message': str(error),
                                      'traceback': traceback.format_exc()})
    finally:
        for fd in (status_write, status_read, *raw_fds):
            if fd is not None:
                cleanup_operation(lambda fd=fd: os.close(fd), 'unassigned_fd_close', 'descriptor_cleanup_failed')
        if child is not None:
            # No poll()/wait() before the LAST signal. The sentinel is either
            # alive or an unreaped zombie, so its PID cannot be recycled here.
            try:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                soft_end = min(deadline, time.monotonic() + min(0.2, cleanup_reserve / 4))
                while selector.get_map() and time.monotonic() < soft_end:
                    pump(min(0.02, soft_end - time.monotonic()), raw_allowed=not overflow)
            except BaseException as error:
                reason = reason or 'cleanup_capture_error'
                infrastructure_errors.append({'type': type(error).__name__, 'message': str(error)})
            finally:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                except BaseException as error:
                    reason = reason or 'final_group_kill_failed'
                    infrastructure_errors.append({'type': type(error).__name__, 'message': str(error)})
            try:
                child.wait(timeout=max(0.01, min(1.0, deadline - time.monotonic())))
                tail_end = min(deadline, time.monotonic() + 0.5)
                while selector.get_map() and time.monotonic() < tail_end:
                    pump(min(0.02, tail_end - time.monotonic()), raw_allowed=not overflow)
                while time.monotonic() < tail_end:
                    try: os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        group_gone = True
                        break
                    time.sleep(0.01)
            except BaseException as error:
                reason = reason or 'cleanup_failed'
                infrastructure_errors.append({'type': type(error).__name__, 'message': str(error)})
        try:
            keys = list(selector.get_map().values())
        except BaseException as error:
            keys = []
            reason = reason or 'selector_cleanup_failed'
            infrastructure_errors.append({'phase': 'selector_get_map', 'type': type(error).__name__,
                                          'message': str(error)})
        for key in keys:
            cleanup_operation(lambda key=key: selector.unregister(key.fileobj),
                              'selector_unregister', 'selector_cleanup_failed')
        for stream in pipes:
            cleanup_operation(stream.close, 'pipe_close', 'descriptor_cleanup_failed')
        cleanup_operation(selector.close, 'selector_close', 'selector_cleanup_failed')
        for name, stream in streams.items():
            cleanup_operation(stream.flush, name + '_flush', 'raw_retention_failed')
            cleanup_operation(lambda stream=stream: os.fsync(stream.fileno()),
                              name + '_fsync', 'raw_retention_failed')
            cleanup_operation(stream.close, name + '_close', 'raw_retention_failed')
    if pending: reason = reason or 'interrupted'
    if command_exit != 0: reason = reason or 'actual_command_failed_or_incomplete'
    if not group_gone: reason = reason or 'process_group_not_gone'
    if not all(raw_eof.values()) or status_buffer: reason = reason or 'incomplete_raw_tail'
    if os.path.lexists(cache): reason = reason or 'sentinel_cache_appeared'
    seals = {}
    for name in counts:
        try:
            selected = regular(str(target / (name + '.raw')))
            seals[name] = {'path': str(selected), 'size': selected.stat().st_size, 'sha256': sha_file(selected)}
            require(seals[name]['size'] == counts[name], 'Retained raw size differs from observed complete writes')
        except BaseException as error:
            reason = reason or 'raw_seal_failed'
            infrastructure_errors.append({'type': type(error).__name__, 'message': str(error)})
    try:
        require(sha_file(regular(request_path)) == request_sha256 and sha_file(source) == supervisor_sha256,
                'Fixed request/supervisor input changed after command')
    except BaseException as error:
        reason = reason or 'input_changed_after_command'
        infrastructure_errors.append({'type': type(error).__name__, 'message': str(error)})
    finished = time.monotonic()
    if finished >= deadline: reason = reason or 'deadline_including_retention_exceeded'
    receipt = {'state': 'PROCESS_CAPTURE_FACTS_ONLY_NOT_ACCEPTANCE',
               'failure_reason': reason, 'request_sha256': request_sha256, 'supervisor_sha256': supervisor_sha256,
               'argv': document['argv'], 'sentinel_argv': argv, 'sentinel_pid_and_pgid': child.pid if child else None,
               'actual_command_pid': actual_pid, 'actual_command_exit_code': command_exit,
               'sentinel_exit_code': child.returncode if child else None, 'process_group_gone': group_gone,
               'started_monotonic': started, 'finished_monotonic_before_receipt': finished,
               'elapsed_seconds_before_receipt': finished - started, 'deadline_monotonic': deadline,
               'raw_streams': seals, 'raw_eof': raw_eof, 'observed_raw_bytes': counts,
               'raw_limit_monitor_bytes': raw_limit, 'unread_tail_possible': not all(raw_eof.values()),
               'control_events': events, 'signals_received': list(pending),
               'infrastructure_errors': infrastructure_errors,
               'authorization_and_result_semantics_checked': False}
    try:
        exclusive_json(target / 'process-receipt.json', receipt)
    except BaseException as error:
        receipt['state'] = 'PROCESS_CAPTURE_FAILED'
        receipt['failure_reason'] = receipt['failure_reason'] or 'process_receipt_retention_failed'
        receipt['infrastructure_errors'].append({'phase': 'process_receipt_retention',
                'type': type(error).__name__, 'message': str(error)})
        raise CaptureFailed(receipt) from error
    if time.monotonic() >= deadline or pending:
        receipt['state'], receipt['failure_reason'] = ('PROCESS_CAPTURE_FAILED',
                reason or ('interrupted' if pending else 'receipt_deadline_exceeded'))
        try:
            exclusive_json(target / 'late-receipt.json', receipt)
        except BaseException as error:
            receipt['infrastructure_errors'].append({'phase': 'late_receipt_retention',
                    'type': type(error).__name__, 'message': str(error)})
            raise CaptureFailed(receipt) from error
    if receipt['failure_reason'] is not None:
        receipt['state'] = 'PROCESS_CAPTURE_FAILED'
        raise CaptureFailed(receipt)
    receipt['state'] = 'PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE'
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', '_sentinel'))
    parser.add_argument('request')
    parser.add_argument('request_sha256')
    parser.add_argument('status_fd', nargs='?', type=int)
    args = parser.parse_args()
    if args.action == '_sentinel':
        require(type(args.status_fd) is int and args.status_fd > 2, 'Dedicated parent status descriptor required')
        sentinel(args.request, args.request_sha256, args.status_fd)
    else:
        value = request(args.request, args.request_sha256)
        print(json.dumps({'state': 'COMMAND_STATIC_VALIDATED_NOT_EXECUTED', 'argv': value['argv'], 'cwd': value['cwd']}))


if __name__ == '__main__':
    main()
