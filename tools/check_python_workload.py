"""Execution-tool controls with a fake SDK; no database engine is imported."""
from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
import types
import unittest
import sys
from unittest.mock import patch

import python_workload as tool
from result_journal import ResultJournal

OUTPUT_ROOT = None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plan():
    source = Path(tool.__file__).parent
    return dict(schema_version=1, source_input_sha256='1'*64, wheel_sha256='2'*64,
                input_manifest_sha256='3'*64, sdk_version='0.1.1', engine_version='0.1.1',
                profiles=['balanced'], scales=[1000], modes=['manifest'], rounds=1,
                query_repeats=1, warmup_repeats=1, batch_rows=1, retain_repeats=1,
                lifecycle_cycles=2, lifecycle_scale=1000, trace_python=False,
                case_timeout_seconds=60, total_timeout_seconds=120,
                tool_sha256={name:tool.sha(source/name) for name in
                             ['python_workload.py','result_journal.py','memory_observer.py']})


def fixtures(root):
    root.mkdir()
    facts = {'files': {}}
    for name in ['1000','10000','100000','ownership']:
        directory = root/name
        directory.mkdir()
        doc = dict(external_id='sample', metadata={}, title='Jade',
                   source_uri='synthetic://cost/owner' if name=='ownership' else 'synthetic://cost/sample')
        chunk = dict(document_external_id='sample', chunk_external_id='long' if name=='ownership' else 'sample',
                     ordinal=0, text='Jade 中 leaves.\n'*100, metadata={},
                     source_span={'start':0,'end':len(('Jade 中 leaves.\n'*100).encode()),'line_start':1,'line_end':100})
        vector = dict(chunk_external_id=chunk['chunk_external_id'], dim=2, model_name='controlled',
                      vector=[0.25,0.75])
        records = {}
        for filename,row in [('documents.jsonl',doc),('chunks.jsonl',chunk),('embeddings.jsonl',vector)]:
            data=(json.dumps(row,ensure_ascii=False)+'\n').encode()
            (directory/filename).write_bytes(data)
            records[filename]={'sha256':digest(data),'bytes':len(data),'rows':1}
        manifest=dict(collection='controlled', embedding_dim=2, embedding_model='controlled',
                      distance_metric='cosine', vectors_normalized=False,
                      documents='documents.jsonl',chunks='chunks.jsonl',embeddings='embeddings.jsonl')
        data=json.dumps(manifest).encode()
        (directory/'import_manifest.json').write_bytes(data)
        records['import_manifest.json']={'sha256':digest(data),'bytes':len(data)}
        facts['files'][name]=records
    queries=[]
    for profile in tool.PROFILES:
        for name in ['keyword','vector','hybrid','hybrid-filtered']:
            queries.append(dict(id=name,profile=profile,text='jade',
                                embedding={'model_name':'controlled','dim':2,'vector':[0.25,0.75]},
                                filter={'doc.project':9} if name=='hybrid-filtered' else None,top_k=10))
    for name,field,rows in [('scale-queries.json','scale_query_sha256',queries),
                             ('ownership-queries.json','ownership_query_sha256',queries)]:
        data=json.dumps(rows).encode();(root/name).write_bytes(data);facts[field]=digest(data)
    (root/'input-manifest.json').write_text(json.dumps(facts))
    return facts


def fake_sdk():
    class Error(Exception):
        code='duplicate_external_id';message='controlled duplicate';detail={'external_id':'sample'}
        recoverable=False;suggested_action=None
        operation_outcome={'state':'not_committed','confirmed_committed_batches':0,'confirmed_committed_records':0}
    sdk=types.SimpleNamespace(CofferError=Error,active=0,registry={},search_failure=None,
                              close_failure=None,refusal_outcome=None,searches=[])
    class Collection:
        def __init__(self,db):
            self.db=db
        def write_batch(self,*,documents=(),chunks=(),embeddings=(),on_conflict='error'):
            state=self.db.state
            if any(row['external_id'] in state['docs'] for row in documents):
                error=Error('controlled duplicate')
                if sdk.refusal_outcome is not None:error.operation_outcome=sdk.refusal_outcome
                raise error
            for row in documents:state['docs'][row['external_id']]=copy.deepcopy(row)
            for row in chunks:state['chunks'][row['chunk_external_id']]=copy.deepcopy(row)
            for row in embeddings:state['embeddings'][row['chunk_external_id']]=copy.deepcopy(row)
            state['commit']+=1
            self.db.path.write_text(str(state['commit']))
            return dict(documents_inserted=len(documents),chunks_inserted=len(chunks),
                        embeddings_inserted=len(embeddings),warnings=[])
        def search(self,text=None,*,embedding=None,filter=None,weights=None,metadata_boost=None,top_k=10,profile=None):
            if sdk.search_failure:raise sdk.search_failure
            sdk.searches.append(dict(text=text,embedding=embedding,filter=filter,top_k=top_k,profile=profile))
            return {'results':[{'chunk_external_id':next(iter(self.db.state['chunks'])),
                                'score':0.123456789,'span_end':'18446744073709551615'}],
                    'profile':profile,'input_text':text,'input_filter':filter,'explain':{'complete':True}}
        def get_document_by_external_id(self,external_id):
            return copy.deepcopy(self.db.state['docs'][external_id])
        def get_chunk_by_external_id(self,external_id):
            row=copy.deepcopy(self.db.state['chunks'][external_id])
            return {'chunk':row,'document':self.get_document_by_external_id(row['document_external_id'])}
    class Database:
        def __init__(self,path,profile):
            self.path=Path(path);self.state=sdk.registry[str(self.path)];self.profile=profile
            self.closed=False;sdk.active+=1
        @staticmethod
        def create(path,*,config=None):
            sdk.registry[path]={'docs':{},'chunks':{},'embeddings':{},'commit':0}
            Path(path).write_bytes(b'control')
            return Database(path,config['resource_profile'])
        @staticmethod
        def open(path,*,read_only=False,options=None):
            return Database(path,options['resource_profile'])
        def __enter__(self):return self
        def __exit__(self,kind,error,tb):
            self.closed=True;sdk.active-=1
            if sdk.close_failure:
                if error is not None:raise BaseExceptionGroup('controlled body and close',[error,sdk.close_failure])
                raise sdk.close_failure
            return False
        def create_collection(self,config):
            sdk.collection_config=copy.deepcopy(config)
            return Collection(self)
        def collection(self,name):return Collection(self)
        def stats(self):
            return dict(document_count=len(self.state['docs']),chunk_count=len(self.state['chunks']),
                        embedding_count=len(self.state['embeddings']),last_commit_id=self.state['commit'])
        def import_jsonl(self,path,*,on_conflict='error'):
            collection=Collection(self)
            for field,name in [('documents','documents.jsonl'),('chunks','chunks.jsonl'),('embeddings','embeddings.jsonl')]:
                for rows in tool.batches(Path(path).parent/name,1):collection.write_batch(**{field:rows})
            return {'documents_imported':len(self.state['docs']),'chunks_imported':len(self.state['chunks']),
                    'embeddings_imported':len(self.state['embeddings']),'warnings':[]}
    sdk.Database=Database
    return sdk


class Controls(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='python-workload-no-engine-',dir=OUTPUT_ROOT))
        self.inputs=self.root/'inputs';self.facts=fixtures(self.inputs)
        self.plan=plan();self.plan['input_manifest_sha256']=tool.sha(self.inputs/'input-manifest.json')
        self.wheel=self.root/'actual.whl';self.wheel.write_bytes(b'controlled-not-an-installable-wheel')
        self.plan['wheel_sha256']=tool.sha(self.wheel)
    def protocol(self,value=None):
        path=self.root/'plan.json';path.write_text(json.dumps(self.plan if value is None else value))
        return path,tool.sha(path)
    def rows(self,path):
        return [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    def test_both_paths_keep_results_counts_refusals_and_owned_long_content(self):
        for mode in ('manifest','batch'):
            sdk=fake_sdk();case=('balanced',1000,mode,0);output=self.root/mode
            tool.run_case(sdk,self.inputs,self.plan,case,output,lambda:{'raw':17})
            receipt=tool.case_receipt(output,case,self.plan)
            self.assertGreater(receipt['api_calls'],10)
            self.assertEqual(sdk.active,0)
            events=self.rows(output)
            self.assertEqual(sum(row['event']=='expected_refusal' for row in events),2)
            self.assertTrue(any(row['event']=='long_result_valid_after_close' for row in events))
            for row in events:
                if row.get('label')=='new-handle-keyword' and row['event']=='completed':
                    payload=json.loads((output/row['payload']['relative_path']).read_bytes())
                    self.assertEqual(payload['ok']['results'][0]['span_end'],'18446744073709551615')
                    self.assertEqual(payload['ok']['results'][0]['score'],0.123456789)
    def test_unexpected_committed_refusal_is_not_accepted(self):
        sdk=fake_sdk();sdk.refusal_outcome={'state':'committed','confirmed_committed_records':1}
        output=self.root/'bad-outcome'
        with self.assertRaises(sdk.CofferError):
            tool.run_case(sdk,self.inputs,self.plan,('balanced',1000,'manifest',0),output,lambda:{})
        self.assertEqual(sdk.active,0)
        row=self.rows(output)[-1]
        self.assertEqual(row['event'],'case_failure')
        error=json.loads((output/row['payload']['relative_path']).read_bytes())['error']
        self.assertEqual(error['structured_fields']['operation_outcome'],sdk.refusal_outcome)

    def test_direct_import_preserves_the_explicit_tokenizer(self):
        folder=self.inputs/'1000'
        path=folder/'import_manifest.json'
        manifest=json.loads(path.read_bytes())
        tokenizer={'name':'default_en','version':'0.1','language_hint':'en',
                   'profile':'latin_code_default','config':{'case_fold':'lower',
                   'split':'unicode_punctuation','stop_words':'default_en'}}
        manifest['tokenizer']=tokenizer
        path.write_text(json.dumps(manifest))
        sdk=fake_sdk()
        journal=ResultJournal(self.root/'tokenizer',{},lambda:{})
        try:
            with sdk.Database.create(str(self.root/'tokenizer.coffer'),config={'resource_profile':'balanced'}) as database:
                tool.import_case(sdk,database,folder,'controlled','batch',1,journal)
            self.assertEqual(sdk.collection_config['tokenizer'],tokenizer)
        finally:
            journal.close()
    def test_body_and_cleanup_exceptions_remain_in_the_journal(self):
        sdk=fake_sdk();sdk.search_failure=RuntimeError('controlled body');sdk.close_failure=MemoryError('controlled close')
        output=self.root/'two-failures'
        with self.assertRaises(BaseExceptionGroup):
            tool.run_case(sdk,self.inputs,self.plan,('balanced',1000,'manifest',0),output,lambda:{})
        row=self.rows(output)[-1]
        error=json.loads((output/row['payload']['relative_path']).read_bytes())['error']
        self.assertEqual({n['type'] for n in error['exception_graph']['nodes']},{'RuntimeError','MemoryError','ExceptionGroup'})
        self.assertEqual(sdk.active,0)
    def test_modified_or_missing_full_payload_does_not_make_a_successful_case(self):
        sdk=fake_sdk();case=('balanced',1000,'manifest',0);output=self.root/'payload-drift'
        tool.run_case(sdk,self.inputs,self.plan,case,output,lambda:{})
        row=next(row for row in self.rows(output) if row['event']=='completed')
        target=output/row['payload']['relative_path'];target.write_bytes(b'controlled corruption')
        with self.assertRaisesRegex(ValueError,'payload'):
            tool.case_receipt(output,case,self.plan)
        self.assertEqual(target.read_bytes(),b'controlled corruption')
    def test_protocol_rejects_hash_tool_boolean_and_unknown_selections(self):
        path,sha=self.protocol()
        self.assertEqual(tool.protocol(path,sha),self.plan)
        with self.assertRaises(ValueError):tool.protocol(path,'0'*64)
        for field,value in [('rounds',True),('schema_version',True),('trace_python',0),
                            ('scales',[True]),('profiles',[{}]),('profiles',['balanced','balanced']),
                            ('modes',['quiet']),('lifecycle_scale',True),('tool_sha256',{})]:
            changed=dict(self.plan);changed[field]=value;path,sha=self.protocol(changed)
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):tool.protocol(path,sha)
    def test_input_hash_or_manifest_escape_stops_before_execution(self):
        self.assertEqual(tool.input_files(self.inputs,self.plan),self.facts)
        path=self.inputs/'1000/documents.jsonl';original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'changed'):tool.input_files(self.inputs,self.plan)
        self.assertNotEqual(path.read_bytes(),original)

    def test_output_guard_rejects_existing_input_and_source_locations(self):
        for path in (self.inputs,self.inputs/'new-output',
                     Path(tool.__file__).resolve().parent/'controlled-new-output'):
            with self.subTest(path=path),self.assertRaises(ValueError):
                tool.checked_output(path,self.inputs)
        self.assertEqual(tool.checked_output(self.root/'new-output',self.inputs),
                         self.root/'new-output')
    def supervise(self,run):
        self.plan['modes']=['manifest','batch']
        path,sha=self.protocol()
        output=self.root/'supervised'
        with patch.object(tool.subprocess,'run',side_effect=run):
            return tool.supervise(path,sha,self.inputs,self.wheel,output)
    def test_supervisor_stops_at_first_failure_and_keeps_both_logs(self):
        calls=[]
        def run(command,**kw):calls.append(command);return subprocess.CompletedProcess(command,7)
        with self.assertRaises(RuntimeError):self.supervise(run)
        receipt=json.loads((self.root/'supervised/run.json').read_text())
        self.assertFalse(receipt['complete']);self.assertEqual(len(calls),1)
        self.assertEqual(receipt['cases'][0]['exit_code'],7)
        self.assertTrue(list((self.root/'supervised').glob('*.stdout')))
        self.assertTrue(list((self.root/'supervised').glob('*.stderr')))
    def test_supervisor_timeout_keeps_partial_manifest_without_retry(self):
        calls=[]
        def run(command,**kw):
            calls.append(command);raise subprocess.TimeoutExpired(command,kw['timeout'])
        with self.assertRaises(RuntimeError):self.supervise(run)
        receipt=json.loads((self.root/'supervised/run.json').read_text())
        self.assertFalse(receipt['complete']);self.assertTrue(receipt['cases'][0]['timed_out'])
        self.assertEqual(len(calls),1)
    def test_exit_zero_without_complete_matching_evidence_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.supervise(lambda command,**kw:subprocess.CompletedProcess(command,0))
        receipt=json.loads((self.root/'supervised/run.json').read_text())
        self.assertFalse(receipt['complete']);self.assertIn('failure',receipt)
        self.assertEqual(len(receipt['cases']),1)

    def test_supervisor_rejects_one_or_all_omitted_complete_payloads(self):
        self.plan['modes']=['manifest','batch']
        path,selected_hash=self.protocol()
        for omission in ('one','all'):
            calls=[]
            def run(command,**kw):
                calls.append(command)
                output=Path(command[command.index('--output')+1])
                parts=command[command.index('--worker')+1].split('-')
                case=(parts[0],int(parts[1]),parts[2],int(parts[3]))
                tool.run_case(fake_sdk(),self.inputs,self.plan,case,output,lambda:{'disabled':True})
                events=self.rows(output)
                original=(output/'events.jsonl').read_bytes()
                (output/'preserved-original-events.jsonl').write_bytes(original)
                if omission=='one':
                    uses=Counter(row['payload']['relative_path'] for row in events if 'payload' in row)
                    row=next(row for row in events if row['event']=='completed'
                             and uses[row['payload']['relative_path']]==1)
                    reference=row.pop('payload')
                    (output/reference['relative_path']).rename(output/'preserved-omitted-payload.json')
                else:
                    for row in events:row.pop('payload',None)
                    (output/'payloads').rename(output/'preserved-omitted-payloads')
                (output/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
                return subprocess.CompletedProcess(command,0)
            destination=self.root/('omitted-'+omission)
            with patch.object(tool.subprocess,'run',side_effect=run),\
                 patch.object(tool.platform,'platform',return_value='NO_ENGINE_CONTROL'),\
                 self.subTest(omission=omission),self.assertRaisesRegex(ValueError,'payload'):
                tool.supervise(path,selected_hash,self.inputs,self.wheel,destination)
            receipt=json.loads((destination/'run.json').read_text())
            self.assertFalse(receipt['complete']);self.assertEqual(len(calls),1)
            self.assertEqual(len(receipt['cases']),1);self.assertIn('failure',receipt)

    def test_real_worker_storage_failure_exports_known_return_to_stderr(self):
        path,selected_hash=self.protocol()
        output=self.root/'worker-storage-failure'
        state=self.root/'worker-storage-state.json'
        child=r'''
from pathlib import Path
import json,sys
sys.path.insert(0,sys.argv[1])
import python_workload as tool
from check_python_workload import fake_sdk
from result_journal import ResultJournal
sdk=fake_sdk();calls={'imports':0}
original=sdk.Database.import_jsonl
def imported(self,path,*,on_conflict='error'):
    calls['imports']+=1
    result=original(self,path,on_conflict=on_conflict)
    result['retention_probe']={'marker':'KNOWN_RETURN 中\0','exact':'18446744073709551615'}
    return result
sdk.Database.import_jsonl=imported
original_init=ResultJournal.__init__
original_payload=ResultJournal.payload
def initialized(self,*args,**kwargs):
    original_init(self,*args,**kwargs,clock=lambda:0)
def payload(self,value):
    if not getattr(self,'controlled_broken_payload',False) and 'ok' in value:
        self.controlled_broken_payload=True
        (self.output/'payloads').rename(self.output/'preserved-empty-payloads')
    return original_payload(self,value)
ResultJournal.__init__=initialized
ResultJournal.payload=payload
tool.installed_sdk=lambda *args:(sdk,{})
tool.ProcessMemoryObserver.__init__=lambda self:None
tool.ProcessMemoryObserver.observe=lambda self:{'disabled':True}
tool.time.perf_counter_ns=lambda:0
state=Path(sys.argv[2])
sys.argv=['controlled-worker',*sys.argv[3:]]
try:
    tool.main()
except BaseException:
    state.write_text(json.dumps({'import_calls':calls['imports'],'active_handles':sdk.active,
                                'sdk_or_engine_loaded':'cofferdb' in sys.modules}))
    raise
'''
        command=[sys.executable,'-B','-c',child,str(Path(tool.__file__).parent),str(state),
                 '--worker','balanced-1000-manifest-0','--protocol',str(path),
                 '--protocol-sha256',selected_hash,'--inputs',str(self.inputs),
                 '--wheel',str(self.wheel),'--output',str(output)]
        result=subprocess.run(command,cwd=self.root,capture_output=True,text=True,timeout=20)
        (self.root/'worker-storage.stdout').write_text(result.stdout)
        (self.root/'worker-storage.stderr').write_text(result.stderr)
        self.assertNotEqual(result.returncode,0)
        prefix='COFFERDB_WORKLOAD_FAILURE_JSON '
        report=json.loads(next(line.removeprefix(prefix) for line in result.stderr.splitlines()
                               if line.startswith(prefix)))
        self.assertEqual(len(report['known_calls']),1)
        known=report['known_calls'][0]
        self.assertTrue(known['call_returned']);self.assertTrue(known['result_available'])
        self.assertEqual(known['result']['documents_imported'],1)
        self.assertEqual(known['result']['retention_probe'],
                         {'marker':'KNOWN_RETURN 中\0','exact':'18446744073709551615'})
        self.assertEqual(json.loads(state.read_text()),
                         {'import_calls':1,'active_handles':0,'sdk_or_engine_loaded':False})
        self.assertFalse(any(row['event']=='case_complete' for row in self.rows(output)))
    def test_supervisor_requires_and_checks_the_actual_child_journal(self):
        def run(command,**kw):
            folder=Path(command[command.index('--output')+1])
            parts=command[command.index('--worker')+1].split('-')
            case=(parts[0],int(parts[1]),parts[2],int(parts[3]))
            identity={'case':dict(zip(('profile','scale','mode','round'),case)),'protocol':self.plan}
            journal=ResultJournal(folder,identity,lambda:{})
            try:
                journal.measure('controlled-not-an-engine-call',lambda:{'complete':'中'})
                journal.append({'event':'case_complete'})
            finally:journal.close()
            return subprocess.CompletedProcess(command,0)
        receipt=self.supervise(run)
        self.assertTrue(receipt['complete']);self.assertEqual(len(receipt['cases']),2)
        self.assertTrue(all(row['evidence']['complete_payloads']==1 for row in receipt['cases']))
    def test_parent_interruption_records_the_active_case(self):
        def run(command,**kw):raise KeyboardInterrupt('controlled stop')
        with self.assertRaises(KeyboardInterrupt):self.supervise(run)
        receipt=json.loads((self.root/'supervised/run.json').read_text())
        self.assertFalse(receipt['complete']);self.assertIsNotNone(receipt['active_case'])
        self.assertEqual(receipt['failure']['type'],'KeyboardInterrupt')

    def test_final_failure_keeps_known_success_nested_in_cleanup_group(self):
        signal=KeyboardInterrupt('controlled observer interruption')
        signal.call_returned=True
        signal.call_result_available=True
        signal.call_result={'documents_inserted':1,'body':'中 🙂'}
        signal.call_record={'event':'completed','call_outcome':'returned','outcome':'measurement_error'}
        cleanup=OSError('controlled close failure')
        error=BaseExceptionGroup('body and cleanup',[signal,cleanup])
        output=self.root/'known-success'
        journal=ResultJournal(output,{},lambda:{})
        try:
            tool.record_fatal(journal,error)
            row=self.rows(output)[-1]
            value=json.loads((output/row['payload']['relative_path']).read_bytes())
            self.assertEqual(row['event'],'case_failure')
            self.assertEqual(value['known_calls'],[{'call_returned':True,'record':signal.call_record,
                                                   'result_available':True,'result':signal.call_result}])
            self.assertIn('KeyboardInterrupt',{node['type'] for node in value['error']['exception_graph']['nodes']})
        finally:journal.close()

    def test_final_failure_record_io_error_keeps_original_signal(self):
        journal=ResultJournal(self.root/'final-io',{},lambda:{})
        error=KeyboardInterrupt('controlled workload interruption')
        error.operation_outcome={'state':'committed','confirmed_committed_records':1}
        (journal.output/'payloads').rename(journal.output/'preserved-payloads')
        try:
            with self.assertRaises(BaseExceptionGroup) as caught:
                tool.record_fatal(journal,error)
            self.assertIs(caught.exception.exceptions[0],error)
            self.assertIsInstance(caught.exception.exceptions[1],FileNotFoundError)
        finally:journal.close()

    def test_receipt_rejects_invalid_clock_values_in_success_rows(self):
        sdk=fake_sdk();case=('balanced',1000,'manifest',0);output=self.root/'clock-receipt'
        tool.run_case(sdk,self.inputs,self.plan,case,output,lambda:{})
        rows=self.rows(output)
        index=next(i for i,row in enumerate(rows) if row['event']=='completed')
        original=rows[index]['elapsed_ns']
        for value in (None,True,-1):
            with self.subTest(value=value):
                rows[index]['elapsed_ns']=value
                (output/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaisesRegex(ValueError,'incomplete or failed'):
                    tool.case_receipt(output,case,self.plan)
        rows[index]['elapsed_ns']=original
        (output/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.assertGreater(tool.case_receipt(output,case,self.plan)['api_calls'],0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    output=args.output.absolute()
    if (output.exists() or output.is_symlink()
            or output.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        parser.error('choose a new output directory outside the source tree')
    output.mkdir(parents=True)
    OUTPUT_ROOT=output
    unittest.main(argv=[sys.argv[0]],verbosity=2)
