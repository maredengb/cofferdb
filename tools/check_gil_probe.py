"""No-engine controls for host-progress barriers and result/error retention."""
from pathlib import Path
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from gil_probe import observe_call
from result_journal import ResultJournal


class Busy(Exception):
    pass


class ProbeControls(unittest.TestCase):
    def journal(self, root):
        return ResultJournal(Path(root) / 'journal', {'fake_sdk': True}, lambda: {})

    def test_active_owner_and_independent_progress(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            progressed = threading.Event()
            def call():
                if not progressed.wait(2):
                    raise AssertionError('fake observer made no progress')
                # Keep the fake owner active through the observer timestamp.
                time.sleep(0.01)
                return {'full_result': 'river\u0000河', 'integer': '18446744073709551615'}
            def busy():
                raise Busy('fake active owner')
            def independent():
                progressed.set()
                return {'independent': True}
            try:
                result, observation = observe_call('fake-active', call, busy, Busy, independent, journal)
                payload = json.loads((Path(root)/'journal'/result['payload']['relative_path']).read_bytes())
                self.assertEqual(payload['ok']['full_result'], 'river\u0000河')
                self.assertEqual(observation['status'], 'host_progress_observed')
                self.assertGreater(observation['qualified_samples'], 0)
            finally:
                journal.close()

    def test_successful_polls_are_not_progress_proof(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            try:
                _, row = observe_call('no-busy', lambda: time.sleep(0.02), lambda: {}, Busy,
                                      lambda: {}, journal)
                self.assertEqual(row['status'], 'inconclusive_no_active_window')
                self.assertEqual(row['qualified_samples'], 0)
            finally:
                journal.close()

    def test_original_error_not_retried(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            calls = []
            original = ValueError('original failure')
            def call():
                calls.append(True)
                raise original
            try:
                with self.assertRaises(ValueError) as caught:
                    observe_call('bad-call', call, lambda: {}, Busy, lambda: {}, journal)
                self.assertIs(caught.exception, original)
                self.assertEqual(len(calls), 1)
            finally:
                journal.close()

    def test_observer_failure_preserves_successful_result(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            observed = threading.Event()
            def busy():
                observed.set()
                raise ValueError('probe fault')
            def call():
                self.assertTrue(observed.wait(2))
                return {'committed': True, 'body': 'retained'}
            try:
                with self.assertRaises(BaseExceptionGroup):
                    observe_call('observer-failure', call, busy, Busy, lambda: {}, journal)
            finally:
                journal.close()
            events = [json.loads(line) for line in (Path(root)/'journal/events.jsonl').read_text().splitlines()]
            completed = next(e for e in events if e['event'] == 'completed')
            payload = json.loads((Path(root)/'journal'/completed['payload']['relative_path']).read_bytes())
            self.assertEqual(payload['ok'], {'committed':True,'body':'retained'})
            self.assertEqual(events[-1]['status'], 'observation_failed')

    def test_complete_call_not_counted_after_it_finishes(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            release, entered = threading.Event(), threading.Event()
            def busy():
                raise Busy()
            def independent():
                entered.set()
                release.wait(1)
                return {}
            def call():
                self.assertTrue(entered.wait(2))
                release.set()
                return {}
            try:
                _, row = observe_call('after-completion', call, busy, Busy, independent, journal)
                # Every accepted sample still has to lie inside actual bounds;
                # scheduler ordering can allow the independent probe first.
                for sample in row['samples']:
                    if sample['host_progress_finished_ns'] > row['bounds']['call_finished_ns']:
                        self.assertFalse(sample['active_after_independent'])
            finally:
                release.set()
                journal.close()

    def test_end_clock_failure_keeps_returned_value(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            returned = threading.Event()
            original_clock = time.perf_counter_ns
            original_thread = threading.get_ident()
            failed = []
            def clock():
                if threading.get_ident() == original_thread and returned.is_set() and not failed:
                    failed.append(True)
                    raise OSError('end clock unavailable')
                return original_clock()
            def call():
                returned.set()
                return {'committed': True, 'body': 'known successful result'}
            try:
                with patch('gil_probe.time.perf_counter_ns', side_effect=clock):
                    with self.assertRaises(BaseExceptionGroup):
                        observe_call('end-clock', call, lambda: {}, Busy, lambda: {}, journal)
            finally:
                journal.close()
            events = [json.loads(line) for line in (Path(root)/'journal/events.jsonl').read_text().splitlines()]
            completed = next(e for e in events if e['event'] == 'completed')
            payload = json.loads((Path(root)/'journal'/completed['payload']['relative_path']).read_bytes())
            self.assertEqual(payload['ok'], {'committed':True,'body':'known successful result'})
            self.assertEqual(events[-1]['status'], 'observation_failed')

    def test_interrupted_join_preserves_original_operation_error(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            original = ValueError('original operation outcome')
            interruption = KeyboardInterrupt('join interrupted')
            def call():
                raise original
            try:
                with patch('gil_probe.threading.Thread.join', side_effect=interruption):
                    with self.assertRaises(BaseExceptionGroup) as caught:
                        observe_call('join-interrupted', call, lambda: {}, Busy, lambda: {}, journal)
                self.assertIn(original, caught.exception.exceptions)
                self.assertIn(interruption, caught.exception.exceptions)
            finally:
                journal.close()
                for worker in threading.enumerate():
                    if worker.name == 'cofferdb-host-progress':
                        worker.join(1)
            events = [json.loads(line) for line in (Path(root)/'journal/events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['status'], 'incomplete_observer_timeout')

    def test_independent_return_survives_its_end_clock_failure(self):
        with tempfile.TemporaryDirectory() as root:
            journal = self.journal(root)
            returned = threading.Event()
            original_clock = time.perf_counter_ns
            original_thread = threading.get_ident()
            failed = []
            def clock():
                if threading.get_ident() != original_thread and returned.is_set() and not failed:
                    failed.append(True)
                    raise OSError('independent end clock unavailable')
                return original_clock()
            def call():
                self.assertTrue(returned.wait(2))
                return {'main': 'complete'}
            def busy():
                raise Busy()
            def independent():
                returned.set()
                return {'independent': 'known full value'}
            try:
                with patch('gil_probe.time.perf_counter_ns', side_effect=clock):
                    with self.assertRaises(BaseExceptionGroup):
                        observe_call('independent-clock', call, busy, Busy, independent, journal)
            finally:
                journal.close()
            events = [json.loads(line) for line in (Path(root)/'journal/events.jsonl').read_text().splitlines()]
            observation = json.loads((Path(root)/'journal'/events[-1]['payload']['relative_path']).read_bytes())
            self.assertEqual(observation['status'], 'observation_failed')
            self.assertEqual(observation['qualified_samples'], 0)
            self.assertEqual(observation['samples'][0]['independent_result'],
                             {'independent': 'known full value'})
            self.assertIsNone(observation['samples'][0]['host_progress_finished_ns'])

    def test_invalid_limits_rejected_without_call(self):
        for options in [{'interval_seconds':0}, {'interval_seconds':float('nan')},
                        {'join_seconds':float('inf')}, {'sample_limit':True}, {'sample_limit':0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                observe_call('invalid', lambda: self.fail('unexpected call'), lambda: {}, Busy,
                             lambda: {}, None, **options)


if __name__ == '__main__':
    unittest.main()
