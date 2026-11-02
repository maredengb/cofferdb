"""No database, quality workload or performance acceptance is executed here."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import weakref
from unittest.mock import patch
import io

from result_journal import ResultJournal, failure

CONTROL_ROOT = None


class JournalControls(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='journal-controls-', dir=CONTROL_ROOT))
        self.journals = []

    def tearDown(self):
        for journal in self.journals:
            journal.close()

    def journal(self, name='records', ticks=(100, 125), observer=lambda:{'raw':17}):
        iterator = iter(ticks)
        journal = ResultJournal(self.root/name, {'kind':'NO_ENGINE_CONTROL'}, observer, clock=lambda:next(iterator))
        self.journals.append(journal)
        return journal

    def rows(self, journal):
        return [json.loads(line) for line in (journal.output/'events.jsonl').read_text().splitlines()]

    def value(self, journal, row):
        path = journal.output/row['payload']['relative_path']
        data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), row['payload']['sha256'])
        self.assertEqual(len(data), row['payload']['bytes'])
        return json.loads(data)

    def test_elapsed_excludes_observer_and_full_result_is_retained(self):
        journal = self.journal()
        value = {'text':'Jade 中 🙂\0\n'*1000, 'span':{'end':'18446744073709551615'}, 'scores':[0.1, 0.123456789]}
        row = journal.measure('full-value', lambda:value)
        self.assertEqual(row['elapsed_ns'], 25)
        self.assertEqual(self.value(journal, row), {'ok':value})
        self.assertEqual([r['event'] for r in self.rows(journal)], ['identity','started','completed'])

    def test_repeated_identical_results_keep_one_complete_payload(self):
        journal = self.journal(ticks=(1,2,3,4))
        a = journal.measure('first', lambda:{'hits':[{'id':'a','rank':1,'score':0.5}]})
        b = journal.measure('second', lambda:{'hits':[{'id':'a','rank':1,'score':0.5}]})
        self.assertEqual(a['payload'], b['payload'])
        self.assertEqual(len(list((journal.output/'payloads').glob('*.json'))), 1)
        self.assertEqual(self.value(journal, b)['ok']['hits'][0]['rank'], 1)

    def test_core_error_and_confirmed_progress_persist_before_propagation(self):
        class ControlledError(Exception):
            code='controlled';message='controlled failure';detail={'actual':'18446744073709551615'}
            recoverable=False;suggested_action=None
            operation_outcome={'state':'committed','confirmed_committed_records':3}
        journal = self.journal()
        error = ControlledError('controlled failure')
        def call():
            raise error
        with self.assertRaises(ControlledError) as caught:
            journal.measure('error', call)
        self.assertIs(caught.exception, error)
        row = self.rows(journal)[-1]
        self.assertEqual(row['outcome'], 'error')
        stored = self.value(journal,row)['error']['structured_fields']
        self.assertEqual(stored, {name:getattr(error,name) for name in
            ['code','message','detail','recoverable','suggested_action','operation_outcome']})

    def test_serialization_failure_is_not_reported_as_a_success(self):
        journal = self.journal()
        with self.assertRaises(TypeError):
            journal.measure('unencodable', lambda:{'value':object()})
        row = self.rows(journal)[-1]
        self.assertEqual(row['outcome'], 'evidence_error')
        self.assertNotIn('payload',row)
        self.assertTrue(list((journal.output/'payloads').glob('unfinished-*')))

    def test_observer_failure_is_explicit_and_does_not_mask_result(self):
        def observer():
            raise OSError('controlled unsupported metric')
        journal = self.journal(observer=observer)
        row = journal.measure('metric-unavailable', lambda:7)
        self.assertEqual(row['after_call']['unavailable']['type'],'OSError')
        self.assertEqual(self.value(journal,row), {'ok':7})

    def test_negative_clock_result_is_preserved_then_stops(self):
        journal = self.journal(ticks=(100,99))
        with self.assertRaisesRegex(ValueError, 'backwards'):
            journal.measure('bad-clock',lambda:7)
        row = self.rows(journal)[-1]
        self.assertEqual(row['outcome'],'measurement_error')
        self.assertEqual(row['elapsed_ns'],-1)
        self.assertIn('measurement_error',row)
        self.assertEqual(self.value(journal,row),{'ok':7})

    def test_retained_results_and_release_are_distinct_events(self):
        class Values(list):
            pass
        references = []
        def call():
            value = Values([1,2,3])
            references.append(weakref.ref(value))
            return value
        journal = self.journal()
        row = journal.measure('retained',call,retain=True)
        self.assertTrue(row['result_retained_by_harness'])
        self.assertIsNotNone(references[0]())
        journal.release_retained()
        self.assertIsNone(references[0]())
        self.assertEqual(self.rows(journal)[-1]['event'],'retained_released')

    def test_existing_output_and_corrupt_content_address_are_not_overwritten(self):
        journal = self.journal(ticks=(1,2,3,4))
        first = journal.measure('first',lambda:17)
        target = journal.output/first['payload']['relative_path']
        target.write_bytes(b'controlled preexisting corruption')
        with self.assertRaisesRegex(ValueError,'content address'):
            journal.measure('second',lambda:17)
        self.assertEqual(target.read_bytes(),b'controlled preexisting corruption')
        self.assertEqual(self.rows(journal)[-1]['outcome'],'evidence_error')
        with self.assertRaises(FileExistsError):
            ResultJournal(journal.output,{},lambda:{})

    def test_journal_failure_keeps_the_original_error_and_progress(self):
        class ControlledError(Exception):
            operation_outcome={'state':'committed','confirmed_committed_records':3}
        error = ControlledError('committed')
        journal = self.journal()
        original = journal.append
        def append(row):
            if row['event'] == 'completed':
                raise OSError('controlled full journal volume')
            original(row)
        journal.append = append
        def call():
            raise error
        with self.assertRaises(BaseExceptionGroup) as caught:
            journal.measure('error-and-journal-failure',call)
        self.assertIs(caught.exception.exceptions[0],error)
        self.assertIsInstance(caught.exception.exceptions[1],OSError)
        payload = next((journal.output/'payloads').glob('*.json'))
        stored = json.loads(payload.read_bytes())['error']['structured_fields']
        self.assertEqual(stored['operation_outcome'],error.operation_outcome)
        self.assertEqual(self.rows(journal)[-1]['event'],'started')

    def test_clock_failure_after_call_retains_the_original_signal(self):
        error = KeyboardInterrupt('controlled call interruption')
        calls = 0
        def clock():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError('controlled clock allocation failure')
            return 17
        journal = self.journal()
        journal.clock = clock
        def call():
            raise error
        with self.assertRaises(BaseExceptionGroup) as caught:
            journal.measure('clock-failed-after-call',call)
        self.assertIs(caught.exception.exceptions[0],error)
        self.assertIsInstance(caught.exception.exceptions[1],MemoryError)
        row = self.rows(journal)[-1]
        self.assertEqual(row['outcome'],'measurement_error')
        self.assertEqual(self.value(journal,row)['error']['type'],'KeyboardInterrupt')

    def test_clock_failure_after_success_preserves_complete_known_value(self):
        journal = self.journal()
        value = {'operation_outcome':{'state':'committed','confirmed_committed_records':3},
                 'rows':[{'text':'Jade 中 🙂\0','value':18446744073709551615}]}
        ticks = iter((17, MemoryError('controlled end clock failure')))
        count = 0
        def clock():
            item = next(ticks)
            if isinstance(item,BaseException):
                raise item
            return item
        def call():
            nonlocal count
            count += 1
            return value
        journal.clock = clock
        with self.assertRaises(MemoryError) as caught:
            journal.measure('returned-before-clock-failure',call)
        self.assertEqual(count,1)
        self.assertTrue(caught.exception.call_returned)
        self.assertTrue(caught.exception.call_result_available)
        self.assertIs(caught.exception.call_result,value)
        row = self.rows(journal)[-1]
        self.assertEqual(row['outcome'],'measurement_error')
        self.assertEqual(row['call_outcome'],'returned')
        self.assertEqual(self.value(journal,row),{'ok':value})

    def test_post_call_observer_signal_preserves_success_and_signal(self):
        signal = KeyboardInterrupt('controlled post-call observer signal')
        observations = 0
        def observer():
            nonlocal observations
            observations += 1
            if observations == 2:
                raise signal
            return {'raw':17}
        journal = self.journal(observer=observer)
        value = {'committed':3,'text':'Jade 中'}
        with self.assertRaises(KeyboardInterrupt) as caught:
            journal.measure('observer-signal',lambda:value)
        self.assertIs(caught.exception,signal)
        self.assertTrue(signal.call_returned)
        self.assertIs(signal.call_result,value)
        self.assertEqual(self.value(journal,signal.call_record),{'ok':value})
        self.assertEqual(signal.call_record['outcome'],'measurement_error')

    def test_payload_failure_reports_actual_harness_retention(self):
        class Values(list):
            pass
        for retain in (False,True):
            with self.subTest(retain=retain):
                journal = self.journal(name='retained-'+str(retain))
                reference = []
                def call():
                    value = Values([{'state':'committed','rows':3}])
                    reference.append(weakref.ref(value))
                    return value
                (journal.output/'payloads').rename(journal.output/'preserved-empty-payloads')
                with self.assertRaises(FileNotFoundError) as caught:
                    journal.measure('payload-failed',call,retain=retain)
                row = self.rows(journal)[-1]
                self.assertEqual(row['outcome'],'evidence_error')
                self.assertEqual(row['result_retained_by_harness'],bool(journal.retained))
                self.assertEqual(len(journal.retained),int(retain))
                self.assertTrue(caught.exception.call_returned)
                self.assertTrue(caught.exception.call_result_available)
                self.assertIs(caught.exception.call_result,reference[0]())

    def test_payload_and_clock_and_journal_faults_do_not_repeat_the_call(self):
        journal = self.journal()
        clock_error = MemoryError('clock')
        payload_error = OSError('payload')
        journal_error = KeyboardInterrupt('journal')
        ticks = iter((17,clock_error))
        def clock():
            item = next(ticks)
            if isinstance(item,BaseException): raise item
            return item
        journal.clock = clock
        count = 0
        value = {'operation_outcome':{'state':'committed','confirmed_committed_records':3}}
        def call():
            nonlocal count
            count += 1
            return value
        original = journal.append
        def append(row):
            if row['event']=='completed': raise journal_error
            original(row)
        journal.append = append
        with patch.object(journal,'payload',side_effect=payload_error) as payload:
            with self.assertRaises(BaseExceptionGroup) as caught:
                journal.measure('combined-tool-fault',call)
        self.assertEqual(count,1)
        self.assertEqual(payload.call_count,1)
        self.assertEqual(caught.exception.exceptions,(clock_error,payload_error,journal_error))
        self.assertIs(caught.exception.call_result,value)
        self.assertTrue(caught.exception.call_returned)
        self.assertEqual(caught.exception.call_record['outcome'],'evidence_error')
        self.assertEqual(self.rows(journal)[-1]['event'],'started')

    def test_journal_failure_after_release_delivers_complete_payload_reference(self):
        journal = self.journal()
        original = journal.append
        def append(row):
            if row['event']=='completed': raise OSError('controlled journal failure')
            original(row)
        journal.append = append
        value = {'operation_outcome':{'state':'committed'},'rows':list(range(30))}
        with self.assertRaises(OSError) as caught:
            journal.measure('journal-fault-after-payload',lambda:value)
        self.assertTrue(caught.exception.call_returned)
        self.assertFalse(caught.exception.call_result_available)
        self.assertEqual(caught.exception.call_record['outcome'],'evidence_error')
        self.assertEqual(self.value(journal,caught.exception.call_record),{'ok':value})

    def test_retention_allocation_failure_does_not_claim_harness_reference(self):
        class CannotRetain(list):
            def append(self,value): raise MemoryError('controlled retention allocation failure')
        journal = self.journal()
        journal.retained = CannotRetain()
        value = {'rows':[1,2,3]}
        with self.assertRaises(MemoryError) as caught:
            journal.measure('retain-allocation',lambda:value,retain=True)
        self.assertFalse(caught.exception.call_record['result_retained_by_harness'])
        self.assertEqual(journal.retained,[])
        self.assertIs(caught.exception.call_result,value)
        self.assertEqual(self.value(journal,caught.exception.call_record),{'ok':value})

    def test_after_release_signal_preserves_persisted_result(self):
        count = 0
        def observer():
            nonlocal count
            count += 1
            if count == 3: raise KeyboardInterrupt('controlled release observer signal')
            return {'raw':17}
        journal = self.journal(observer=observer)
        with self.assertRaises(KeyboardInterrupt) as caught:
            journal.measure('release-observer',lambda:{'text':'complete returned body'})
        self.assertTrue(caught.exception.call_returned)
        self.assertFalse(caught.exception.call_result_available)
        self.assertEqual(self.value(journal,caught.exception.call_record),{'ok':{'text':'complete returned body'}})
        self.assertEqual(self.rows(journal)[-1]['outcome'],'measurement_error')

    def test_single_post_call_fault_preserves_its_existing_cause(self):
        for stage in ('clock','observer','payload','journal','retention'):
            with self.subTest(stage=stage):
                journal=self.journal(name='cause-'+stage)
                cause=ValueError('controlled underlying cause')
                error=KeyboardInterrupt('controlled observer fault') if stage=='observer' else OSError('controlled '+stage+' fault')
                error.__cause__=cause
                ticks=iter((17,19))
                clock_calls=0
                observations=0
                def clock():
                    nonlocal clock_calls
                    clock_calls+=1
                    if stage=='clock' and clock_calls==2:raise error
                    return next(ticks)
                def observer():
                    nonlocal observations
                    observations+=1
                    if stage=='observer' and observations==2:raise error
                    return {'raw':17}
                journal.clock=clock
                journal.observer=observer
                if stage=='payload':
                    def payload(value):raise error
                    journal.payload=payload
                if stage=='journal':
                    append=journal.append
                    def fail_completed(row):
                        if row['event']=='completed':raise error
                        append(row)
                    journal.append=fail_completed
                if stage=='retention':
                    class CannotRetain(list):
                        def append(self,value):raise error
                    journal.retained=CannotRetain()
                with self.assertRaises(type(error)) as caught:
                    journal.measure('known-cause',lambda:{'complete':'中'},retain=stage=='retention')
                self.assertIs(caught.exception,error)
                self.assertIs(error.__cause__,cause)
                graph=failure(error)['exception_graph']
                self.assertEqual(len(graph['nodes']),2)
                self.assertEqual(graph['nodes'][graph['nodes'][0]['cause']]['message'],str(cause))

    def test_failed_identity_write_closes_its_new_stream(self):
        stream = io.StringIO()
        with patch('result_journal.Path.open',return_value=stream), \
             patch.object(ResultJournal,'append',side_effect=TypeError('controlled identity failure')):
            with self.assertRaises(TypeError):
                ResultJournal(self.root/'identity-failure',{},lambda:{})
        self.assertTrue(stream.closed)



class Contract(unittest.TestCase):
    def test_actual_sdk_operation_outcome_and_reason_are_complete(self):
        class Signal(KeyboardInterrupt):
            sdk_reason='interrupted_after_commit'
            operation_outcome={'state':'committed','confirmed_committed_batches':2,'confirmed_committed_records':513,'last_confirmed_commit_id':18446744073709551615,'last_confirmed_commit_sequence_no':7}
        ticks=iter([10,20])
        folder=Path(tempfile.mkdtemp(dir=CONTROL_ROOT))/'journal'
        journal=ResultJournal(folder,{},lambda:{},clock=lambda:next(ticks))
        try:
            error=Signal()
            def call(): raise error
            with self.assertRaises(Signal): journal.measure('signal',call)
            row=json.loads((folder/'events.jsonl').read_text().splitlines()[-1])
            payload=json.loads((folder/row['payload']['relative_path']).read_text())
            fields=payload['error']['structured_fields']
            self.assertEqual(fields['operation_outcome'],error.operation_outcome)
            self.assertEqual(fields['sdk_reason'],error.sdk_reason)
        finally: journal.close()
    def test_cleanup_group_and_nested_core_error_keep_every_field(self):
        class Core(Exception):
            code='controlled_core';message='完整错误';detail={'actual':18446744073709551615}
            recoverable=False;suggested_action=None
        core=Core('generic SDK string')
        memory=MemoryError('conversion allocation')
        memory.__cause__=core
        memory.operation_outcome={'state':'committed','confirmed_committed_records':513}
        signal=KeyboardInterrupt('body interrupted')
        signal.sdk_reason='signal_after_call'
        group=BaseExceptionGroup('body and cleanup',[signal,memory])
        value=failure(group)
        graph=value['exception_graph'];nodes=graph['nodes']
        self.assertEqual(graph['root'],0)
        self.assertEqual(len(nodes),4)
        self.assertEqual([nodes[n]['type'] for n in nodes[0]['members']],['KeyboardInterrupt','MemoryError'])
        stored_memory=nodes[nodes[0]['members'][1]]
        self.assertEqual(stored_memory['structured_fields']['operation_outcome'],memory.operation_outcome)
        stored_core=nodes[stored_memory['cause']]['structured_fields']
        self.assertEqual(stored_core,{name:getattr(core,name) for name in ['code','message','detail','recoverable','suggested_action']})
        self.assertEqual(json.loads(json.dumps(value)),value)
    def test_exception_cycles_are_links_not_lost_nodes_or_recursion(self):
        first=RuntimeError('first');second=ValueError('second')
        first.__cause__=second;second.__context__=first
        graph=failure(first)['exception_graph']
        self.assertEqual(len(graph['nodes']),2)
        self.assertEqual(graph['nodes'][0]['cause'],1)
        self.assertEqual(graph['nodes'][1]['context'],0)


if __name__ == '__main__':
    import argparse
    import sys
    import result_journal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new external control directory; original failures are retained')
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('choose a new output directory outside the source tree')
    output.mkdir(parents=True)
    CONTROL_ROOT = output
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (JournalControls, Contract))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'kind':'ENGINE_INDEPENDENT_RESULT_RETENTION_CONTROLS',
              'checks':result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
              'successful':result.wasSuccessful(), 'engine_loaded':False,
              'component_sha256':hashlib.sha256(Path(result_journal.__file__).read_bytes()).hexdigest(),
              'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'checks.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    raise SystemExit(0 if result.wasSuccessful() else 1)
