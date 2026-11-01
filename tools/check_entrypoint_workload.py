"""SDK-free controls for the external entrypoint supervisor candidate."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import paired_workload as paired
import python_workload as worker
spec = importlib.util.spec_from_file_location('entrypoints', Path(__file__).with_name('entrypoint_workload.py'))
entrypoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoints)


class EntryControls(unittest.TestCase):
    def plan(self, root, codes):
        inputs = root / 'inputs';inputs.mkdir()
        files = [Path(entrypoints.__file__).resolve(), Path(paired.__file__).resolve(),
                 Path(worker.__file__).resolve(), Path(worker.__file__).resolve().parent/'result_journal.py',
                 Path(worker.__file__).resolve().parent/'memory_observer.py']
        rows = []
        for i, code in enumerate(codes):
            p=inputs/f'step{i}.py';p.write_text(code);files.append(p)
            rows.append({'id':f'step{i}','argv':[sys.executable,'-I','-B',str(p)]})
        return {'schema_version':1,'inputs_sha256':{str(p):worker.sha(p) for p in files},
                'python':sys.executable,'python_target':str(Path(sys.executable).resolve()),
                'python_sha256':worker.sha(Path(sys.executable)),'entries':rows,
                'unit_budget_seconds':5,'total_budget_seconds':15}

    def test_strict_plan_and_one_real_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.plan(root,['import sys;print("full result 河");print("diagnostic",file=sys.stderr)'])
            p=root/'inputs/plan.json';worker.write_json(p,plan)
            self.assertEqual(entrypoints.load(p,worker.sha(p)),plan)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                result=entrypoints.execute(plan,root/'output')
            self.assertTrue(result['complete']);self.assertEqual(result['entries'][0]['state'],'completed')
            self.assertIn('ENTRYPOINT_EXIT_RECEIPT',out.getvalue())
            self.assertIn('full result 河',(root/'output/step0/stdout').read_text())
            self.assertIn('diagnostic',(root/'output/step0/stderr').read_text())
            plan['schema_version']=True;worker.write_json(p,plan)
            with self.assertRaises(ValueError):entrypoints.load(p,worker.sha(p))

    def test_nonzero_preserves_known_exit_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.plan(root,['import sys;print("known");sys.exit(7)','raise AssertionError("never")'])
            with self.assertRaises(paired.ProcessFailure):entrypoints.execute(plan,root/'output')
            report=json.loads((root/'output/run.json').read_text())
            self.assertFalse(report['complete']);self.assertEqual([r['state'] for r in report['entries']],['incomplete','Not Run'])
            self.assertIn('known',(root/'output/step0/stdout').read_text())
            self.assertFalse((root/'output/step1').exists())
            self.assertEqual(report['failure']['known_calls'][0]['result']['exit_code'],7)

    def test_cli_stderr_retains_known_child_when_two_receipts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.plan(root,['print("known child exit zero")','raise AssertionError("never")'])
            path=root/'inputs/plan.json';worker.write_json(path,plan)
            bootstrap = (
                'import sys,runpy;from pathlib import Path;'
                'sys.path.insert(0,'+repr(str(Path(worker.__file__).resolve().parent))+');'
                'import python_workload as worker;original=worker.write_json\n'
                'def writing(path,value):\n'
                ' if path.name=="process.json" and value.get("status") in ("completed","failed"):raise OSError("terminal process receipt failed")\n'
                ' if path.name=="run.json" and "failure" in value:raise OSError("outer failure receipt failed")\n'
                ' return original(path,value)\n'
                'worker.write_json=writing\n'
                'sys.argv=["entrypoint_workload.py","--plan",'+repr(str(path))+',"--plan-sha256",'+repr(worker.sha(path))+',"--output",'+repr(str(root/'output'))+']\n'
                'runpy.run_path('+repr(str(Path(entrypoints.__file__).resolve()))+',run_name="__main__")\n')
            child=subprocess.run([sys.executable,'-I','-B','-c',bootstrap],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(child.returncode,0)
            marker='COFFERDB_WORKLOAD_FAILURE_JSON '
            line=next(line for line in child.stderr.splitlines() if line.startswith(marker))
            result=json.loads(line[len(marker):])
            self.assertEqual(result['known_calls'][0]['result']['exit_code'],0)
            self.assertIn('terminal process receipt failed',line)
            self.assertIn('outer failure receipt failed',line)
            self.assertIn('known child exit zero',(root/'output/step0/stdout').read_text())
            self.assertFalse((root/'output/step1').exists())
            self.assertIsNone(json.loads((root/'output/step0/process.json').read_text())['exit_code'])

    def test_changed_input_stops_before_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.plan(root,['print("must not run")'])
            Path(plan['entries'][0]['argv'][3]).write_text('print("changed")')
            with self.assertRaises(ValueError):entrypoints.execute(plan,root/'output')
            self.assertFalse((root/'output').exists())

    def test_final_receipt_error_revokes_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.plan(root,['print("done")']);real=worker.write_json;fault=OSError('controlled final receipt')
            def writing(path,value):
                if path.name=='run.json' and value.get('complete'):raise fault
                return real(path,value)
            with patch.object(worker,'write_json',writing),self.assertRaises(OSError) as caught:
                entrypoints.execute(plan,root/'output')
            self.assertIs(caught.exception,fault)
            report=json.loads((root/'output/run.json').read_text());self.assertFalse(report['complete']);self.assertEqual(report['entries'][0]['state'],'incomplete')

    def test_final_stdout_budget_remains_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.plan(root,['print("not actually called")']);clock=[0.0]
            real_budget=paired.Budget
            factory=lambda a,b:real_budget(a,b,lambda:clock[0])
            real_print=print
            def printing(*args,**kwargs):clock[0]=20;real_print(*args,**kwargs)
            with patch.object(entrypoints.paired,'Budget',side_effect=factory),patch.object(entrypoints.paired,'run_process',return_value={'exit_code':0}),patch('builtins.print',printing),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(TimeoutError):
                entrypoints.execute(plan,root/'output')
            report=json.loads((root/'output/run.json').read_text());self.assertFalse(report['complete'])


if __name__=='__main__':unittest.main()
