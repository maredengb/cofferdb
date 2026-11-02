"""No-engine controls for declared schedules, budgets, supervision and receipts."""
from pathlib import Path
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import paired_workload as pair
from result_journal import ResultJournal


class PairedControls(unittest.TestCase):
    def test_rounds_group_languages_without_resetting_internal_round(self):
        plan = {'rounds':3,'profiles':['expanded','balanced','constrained'],
                'scales':[1000,10000,100000],'modes':['manifest','batch']}
        rows = list(pair.schedule(plan))
        self.assertEqual(len(rows),108)
        self.assertEqual(len({r['id'] for r in rows}),108)
        self.assertEqual([rows[i]['language'] for i in (0,18,36,54,72,90)],
                         ['python','rust','rust','python','python','rust'])
        self.assertEqual([rows[i]['external_round'] for i in (0,36,72)],[1,2,3])
        self.assertTrue(all(r['internal_round']==0 for r in rows))
        self.assertEqual((rows[0]['profile'],rows[0]['scale'],rows[0]['mode']),
                         ('expanded',1000,'manifest'))

    def test_accumulated_language_and_global_budgets(self):
        clock = [0.0]
        budget = pair.Budget(20,8,lambda:clock[0])
        start, deadline = budget.deadline('python',5)
        self.assertEqual(deadline,5)
        clock[0]=4;budget.finish('python',start,deadline)
        clock[0]=10
        start, deadline = budget.deadline('python',9)
        self.assertEqual(deadline,14)  # Only four language seconds remain.
        clock[0]=14;budget.finish('python',start,deadline)
        with self.assertRaises(TimeoutError):budget.deadline('python',1)
        start, deadline = budget.deadline('rust',10)
        self.assertEqual(deadline,20)  # Global, not a new language timer.
        clock[0]=21
        with self.assertRaises(TimeoutError):budget.finish('rust',start,deadline)
        self.assertEqual(budget.used['rust'],7)

    def run_child(self, code, root, seconds=5):
        env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME'}}
        return pair.run_process([sys.executable,'-I','-B','-c',code],Path(root)/'process',
                                time.monotonic()+seconds,cwd=root,env=env)

    def test_real_child_preserves_both_streams_and_exit(self):
        with tempfile.TemporaryDirectory() as root:
            row=self.run_child('import sys; print("known result"); print("known diagnostic",file=sys.stderr)',root)
            self.assertEqual(row['exit_code'],0)
            self.assertEqual(row['status'],'completed')
            self.assertIn('known result',(Path(root)/'process/stdout').read_text())
            self.assertIn('known diagnostic',(Path(root)/'process/stderr').read_text())

    def test_nonzero_is_one_failed_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(pair.ProcessFailure) as caught:
                self.run_child('import sys;print("partial result");sys.exit(7)',root)
            self.assertEqual(caught.exception.process_record['exit_code'],7)
            self.assertIn('partial result',(Path(root)/'process/stdout').read_text())
            self.assertEqual(len(list(Path(root).iterdir())),1)

    def test_timeout_terminates_owned_child_and_descendant(self):
        with tempfile.TemporaryDirectory() as root:
            code=('import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
                  'print(p.pid,flush=True);time.sleep(30)')
            with self.assertRaises(pair.ProcessFailure) as caught:
                self.run_child(code,root,seconds=0.15)
            row=caught.exception.process_record
            self.assertTrue(row['timed_out'])
            self.assertIn('SIGTERM',row['cleanup_signals'])
            self.assertIsNotNone(row['exit_code'])
            # kill(pid,0) includes zombies. A signalled descendant must never
            # continue doing work; /proc can distinguish Linux zombie state.
            pid=int((Path(root)/'process/stdout').read_text().strip())
            status=Path('/proc')/str(pid)/'stat'
            if status.exists():self.assertEqual(status.read_text().split()[2],'Z')
            else:
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:self.fail('descendant remains live after process-group cleanup')

    def test_log_context_close_failure_preserves_exit_and_original_error(self):
        real_open = Path.open
        class FailingClose:
            def __init__(self, handle): self.handle = handle
            def __enter__(self): return self.handle
            def __exit__(self, *args):
                self.handle.close()
                raise OSError('controlled log close failure')
        for stream in ('stdout', 'stderr'):
            for exit_code in (0, 7):
                with self.subTest(stream=stream, exit_code=exit_code), tempfile.TemporaryDirectory() as root:
                    def opening(path, *args, **kwargs):
                        handle = real_open(path, *args, **kwargs)
                        return FailingClose(handle) if path.name == stream and args == ('xb',) else handle
                    with patch.object(Path, 'open', opening):
                        with self.assertRaises(pair.ProcessFailure) as caught:
                            self.run_child(f'import sys;print("known return 河");sys.exit({exit_code})', root)
                    record = caught.exception.process_record
                    self.assertEqual(record['exit_code'], exit_code)
                    self.assertEqual(record['status'], 'failed')
                    persisted = json.loads((Path(root)/'process/process.json').read_text())
                    self.assertEqual(persisted['exit_code'], exit_code)
                    self.assertEqual(persisted['status'], 'failed')
                    diagnostic = json.dumps(pair.failure(caught.exception), ensure_ascii=False)
                    self.assertIn('controlled log close failure', diagnostic)
                    if exit_code: self.assertIn('child returned nonzero', diagnostic)
                    self.assertIn('known return 河', (Path(root)/'process/stdout').read_text())

    def rust_fixture(self, root, *, mode='batch'):
        output=Path(root)/'rust';output.mkdir();(output/'payloads').mkdir()
        labels=['import-stats'];value={'ok':{'chunk_count':1,'document_count':1,'embedding_count':1}}
        data=(json.dumps(value)+'\n').encode();(output/'payloads/00000001.json').write_bytes(data)
        rows=[{'event':'identity','identity':{'engine_version':'0.1.1','profile':'balanced','mode':mode,'functional_control_only':False}},
              {'event':'started','sequence':'1','label':'import-stats'},
              {'event':'completed','sequence':'1','label':'import-stats','elapsed_ns':'10','outcome':'ok',
               'payload':{'relative_path':'payloads/00000001.json','bytes':str(len(data))}},
              {'event':'case_complete'}]
        (output/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return output,rows,labels

    def test_rust_full_payload_and_counts_are_independently_indexed(self):
        with tempfile.TemporaryDirectory() as root:
            output,rows,labels=self.rust_fixture(root)
            receipt=pair.rust_receipt(output,labels,{'engine_version':'0.1.1','functional_control_only':False})
            self.assertEqual(len(receipt['payloads']),1)
            pair.verify_counts(output,{f+'.jsonl':{'rows':1} for f in ['documents','chunks','embeddings']})
            with self.assertRaises(ValueError):pair.rust_receipt(output,labels+['missing'],{})
            (output/'payloads/00000001.json').unlink()
            with self.assertRaises(ValueError):pair.rust_receipt(output,labels,{})

    def test_rust_payload_escape_and_fake_success_rejected(self):
        for variant in ('escape','wrong_length','error','wrong_identity'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as root:
                output,rows,labels=self.rust_fixture(root)
                if variant=='escape':rows[2]['payload']['relative_path']='../outside.json'
                elif variant=='wrong_length':rows[2]['payload']['bytes']='1'
                elif variant=='error':rows[2]['outcome']='error'
                else:rows[0]['identity']['functional_control_only']=True
                (output/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):
                    pair.rust_receipt(output,labels,{'functional_control_only':False})

    def test_python_complete_marker_cannot_hide_missing_calls(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'python';plan={'lifecycle_scale':1000,'lifecycle_cycles':1}
            case=('balanced',10000,'batch',0)
            journal=ResultJournal(output,{'case':dict(zip(('profile','scale','mode','round'),case)),
                                         'protocol':plan},lambda:{})
            journal.measure('one-call',lambda:{'full':'return'})
            journal.append({'event':'case_complete'});journal.close()
            pair.validate_python(output,case,plan,['one-call'])
            with self.assertRaises(ValueError):pair.validate_python(output,case,plan,['one-call','omitted'])

    def test_known_unit_error_survives_budget_finish_failure(self):
        # Check execute's error aggregation without launching any product/SDK.
        with tempfile.TemporaryDirectory() as root:
            inputs=Path(root)/'inputs';inputs.mkdir(); wheel=Path(root)/'wheels/a.whl';wheel.parent.mkdir()
            plan={'rounds':1,'profiles':['balanced'],'scales':[1000],'modes':['batch'],
                  'case_timeout_seconds':5,'engine_version':'0.1.1','batch_rows':2,
                  'warmup_repeats':1,'query_repeats':1}
            envelope={'global_budget_seconds':10,'language_budget_seconds':10}
            original=pair.ProcessFailure('known child error',{'exit_code':7})
            with patch.object(pair.python_worker,'input_files',return_value={}), \
                 patch.object(pair,'run_process',side_effect=original), \
                 patch.object(pair.Budget,'finish',side_effect=TimeoutError('finish expired')):
                with self.assertRaises(BaseExceptionGroup) as caught:
                    pair.execute(plan,envelope,{},[],inputs=inputs,wheel=wheel,
                                 python=Path(sys.executable),rust=Path('/not/run'),output=Path(root)/'output')
            self.assertIn(original,caught.exception.exceptions)
            report=json.loads((Path(root)/'output/run.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual([r['state'] for r in report['units']],['incomplete','Not Run'])

    def test_frozen_expected_hash_is_not_rebased_to_new_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'binary';path.write_bytes(b'original')
            expected=pair.python_worker.sha(path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):pair.verify_frozen({str(path):expected})

    def test_rust_write_and_query_share_one_unit_deadline(self):
        # Real schedule/execution control flow, fake child and journal validators.
        # The second language unit must not acquire a fresh budget per stage.
        with tempfile.TemporaryDirectory() as root:
            inputs=Path(root)/'inputs';(inputs/'1000').mkdir(parents=True)
            (inputs/'1000/import_manifest.json').write_text('{"collection":"notes"}')
            wheel=Path(root)/'wheel/a.whl';wheel.parent.mkdir()
            plan={'rounds':1,'profiles':['balanced'],'scales':[1000],'modes':['manifest'],
                  'case_timeout_seconds':5,'engine_version':'0.1.1','batch_rows':2,
                  'warmup_repeats':1,'query_repeats':1}
            env={'global_budget_seconds':30,'language_budget_seconds':30}
            calls=[]
            def child(argv,output,deadline,**kwargs):
                calls.append((argv,deadline))
                if 'write' in argv:(Path(argv[2])).write_bytes(b'closed database')
                return {'exit_code':0}
            with patch.object(pair.python_worker,'input_files',return_value={}), \
                 patch.object(pair,'run_process',side_effect=child), \
                 patch.object(pair,'validate_python',return_value={}), \
                 patch.object(pair,'expected_labels',return_value=[]), \
                 patch.object(pair,'verify_counts'), \
                 patch.object(pair,'rust_receipt',return_value={}):
                report=pair.execute(plan,env,{"files":{"1000":{}}},[],inputs=inputs,wheel=wheel,
                                    python=Path(sys.executable),rust=Path('/fake/rust'),output=Path(root)/'output')
            self.assertTrue(report['complete'])
            self.assertEqual(len(calls),3)
            self.assertEqual(calls[1][1],calls[2][1])
            self.assertEqual([r['state'] for r in report['units']],['completed','completed'])
            self.assertIn('--worker',calls[0][0])
            self.assertNotIn('-I',calls[0][0])  # Sibling maintainer modules must remain importable.

    def test_final_process_receipt_failure_keeps_known_exit(self):
        with tempfile.TemporaryDirectory() as root:
            original=pair.python_worker.write_json
            def write(path,value):
                if path.name=='process.json' and value.get('status')=='completed':
                    raise OSError('final receipt unavailable')
                return original(path,value)
            with patch.object(pair.python_worker,'write_json',side_effect=write):
                with self.assertRaises(pair.ProcessFailure) as caught:
                    self.run_child('print("known returned child result")',root)
            self.assertEqual(caught.exception.process_record['exit_code'],0)
            self.assertEqual(caught.exception.process_record['status'],'failed')
            self.assertTrue(caught.exception.call_result_available)
            self.assertIn('known returned child result',(Path(root)/'process/stdout').read_text())

    def test_end_clock_failure_preserves_actual_child_exit(self):
        for exit_code in (0,7):
            with self.subTest(exit_code=exit_code),tempfile.TemporaryDirectory() as root:
                original_clock=time.monotonic
                injected=[]
                def clock():
                    frame=sys._getframe(1)
                    while frame and frame.f_code.co_name!='run_process':
                        frame=frame.f_back
                    row=frame.f_locals.get('record',{}) if frame else {}
                    if (frame is not None and row.get('exit_code')==exit_code
                            and 'finished_monotonic' in row and not injected):
                        injected.append(True)
                        raise OSError('final monotonic unavailable')
                    return original_clock()
                with patch.object(pair.time,'monotonic',side_effect=clock):
                    with self.assertRaises(pair.ProcessFailure) as caught:
                        self.run_child(f'import sys;print("known child return",flush=True);sys.exit({exit_code})',root)
                row=caught.exception.process_record
                self.assertEqual(row['exit_code'],exit_code)
                self.assertIsNone(row['finished_monotonic'])
                self.assertEqual(row['status'],'failed')
                self.assertTrue(caught.exception.call_result_available)
                self.assertEqual(json.loads((Path(root)/'process/process.json').read_text())['exit_code'],exit_code)

    def test_invocation_alias_and_original_target_both_bound(self):
        with tempfile.TemporaryDirectory() as root:
            original=Path(root)/'original';other=Path(root)/'other';alias=Path(root)/'python'
            original.write_bytes(b'original bytes');other.write_bytes(b'new bytes');alias.symlink_to(original)
            binding={'python':{'invocation':str(alias),'resolved':str(original.resolve()),
                               'sha256':pair.python_worker.sha(original)}}
            pair.verify_invocations(binding)
            alias.unlink();alias.symlink_to(other)
            with self.assertRaises(ValueError):pair.verify_invocations(binding)
            # Even identical bytes at a different target are a changed binding.
            other.write_bytes(original.read_bytes())
            with self.assertRaises(ValueError):pair.verify_invocations(binding)

    def fake_execute_receipt_case(self,root,*,writer_effect=None,budget_clock=None):
        inputs=Path(root)/'inputs';(inputs/'1000').mkdir(parents=True)
        (inputs/'1000/import_manifest.json').write_text('{"collection":"notes"}')
        wheel=Path(root)/'wheel/a.whl';wheel.parent.mkdir()
        plan={'rounds':1,'profiles':['balanced'],'scales':[1000],'modes':['manifest'],
              'case_timeout_seconds':5,'engine_version':'0.1.1','batch_rows':2,
              'warmup_repeats':1,'query_repeats':1}
        env={'global_budget_seconds':1 if budget_clock else 30,'language_budget_seconds':1 if budget_clock else 30}
        def child(argv,output,deadline,**kwargs):
            if 'write' in argv:Path(argv[2]).write_bytes(b'closed database')
            return {'exit_code':0}
        from contextlib import ExitStack
        real_budget=pair.Budget
        with ExitStack() as stack:
            stack.enter_context(patch.object(pair.python_worker,'input_files',return_value={}))
            stack.enter_context(patch.object(pair,'run_process',side_effect=child))
            stack.enter_context(patch.object(pair,'validate_python',return_value={}))
            stack.enter_context(patch.object(pair,'expected_labels',return_value=[]))
            stack.enter_context(patch.object(pair,'verify_counts'))
            stack.enter_context(patch.object(pair,'rust_receipt',return_value={}))
            if writer_effect:stack.enter_context(patch.object(pair.python_worker,'write_json',side_effect=writer_effect))
            if budget_clock:stack.enter_context(patch.object(pair,'Budget',side_effect=lambda a,b:real_budget(a,b,clock=budget_clock)))
            return pair.execute(plan,env,{'files':{'1000':{}}},[],inputs=inputs,wheel=wheel,
                                python=Path(sys.executable),rust=Path('/fake/rust'),output=Path(root)/'output')

    def test_final_complete_write_failure_revokes_complete(self):
        with tempfile.TemporaryDirectory() as root:
            original=pair.python_worker.write_json;injected=[]
            def write(path,value):
                if value.get('complete') is True and not injected:
                    injected.append(True);raise OSError('complete receipt unavailable')
                return original(path,value)
            with self.assertRaises(OSError):self.fake_execute_receipt_case(root,writer_effect=write)
            report=json.loads((Path(root)/'output/run.json').read_text())
            self.assertFalse(report['complete']);self.assertIn('failure',report)

    def test_unit_receipt_write_failure_marks_unit_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            original=pair.python_worker.write_json;injected=[]
            def write(path,value):
                if (value.get('units') and value['units'][0]['state']=='completed'
                        and not injected):
                    injected.append(True);raise OSError('unit receipt unavailable')
                return original(path,value)
            with self.assertRaises(OSError):self.fake_execute_receipt_case(root,writer_effect=write)
            report=json.loads((Path(root)/'output/run.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual([r['state'] for r in report['units']],['incomplete','Not Run'])

    def test_final_receipt_io_counts_against_original_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            original=pair.python_worker.write_json;clock=[0.0]
            def write(path,value):
                if value.get('complete') is True:clock[0]=2.0
                return original(path,value)
            with self.assertRaises(TimeoutError):
                self.fake_execute_receipt_case(root,writer_effect=write,budget_clock=lambda:clock[0])
            report=json.loads((Path(root)/'output/run.json').read_text())
            self.assertFalse(report['complete'])
            self.assertGreater(report['language_elapsed_seconds']['rust'],1)

    def test_unit_receipt_io_counts_against_language_budget(self):
        with tempfile.TemporaryDirectory() as root:
            original=pair.python_worker.write_json;clock=[0.0];injected=[]
            def write(path,value):
                if (value.get('units') and value['units'][0]['state']=='completed'
                        and not injected):
                    injected.append(True);clock[0]=2.0
                return original(path,value)
            with self.assertRaises(TimeoutError):
                self.fake_execute_receipt_case(root,writer_effect=write,budget_clock=lambda:clock[0])
            report=json.loads((Path(root)/'output/run.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual([r['state'] for r in report['units']],['incomplete','Not Run'])
            self.assertGreater(report['language_elapsed_seconds']['python'],1)

    def test_expected_call_inventory_includes_python_only_work(self):
        plan={'warmup_repeats':1,'query_repeats':2,'retain_repeats':1,'lifecycle_cycles':2,
              'lifecycle_scale':1000,'batch_rows':256}
        facts={'files':{k:{f+'.jsonl':{'rows':1} for f in ['documents','chunks','embeddings']}
                        for k in ['1000','ownership']}}
        queries=[{'id':'q','profile':'balanced'}]
        unit={'language':'python','profile':'balanced','scale':1000,'mode':'batch'}
        py=pair.expected_labels(plan,unit,facts,queries)
        ru=pair.expected_labels(plan,dict(unit,language='rust'),facts,queries)
        self.assertIn('ownership-long-read',py)
        self.assertEqual(py.count('lifecycle-search-1'),1)
        self.assertNotIn('ownership-long-read',ru)
        self.assertEqual(len(ru),5)


if __name__=='__main__':
    unittest.main()
