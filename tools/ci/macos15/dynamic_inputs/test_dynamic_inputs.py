"""Only own miniature files and pure fixture contracts; no Mac/platform probing."""
import importlib.util
import io
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).absolute().parent

def load(name):
    s=importlib.util.spec_from_file_location('dynamic_fixture_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('dynamic_plan');c=load('dynamic_collect')
def config():
    return {'schema_version':1,'kind':'mac15_dynamic_input_config_1','source_repository':'maredengb/coffered','source_commit':'1'*40,
      'helper_sha256':dict(m.HELPERS),'actions':dict(m.ACTIONS),'python_providers':list(m.PROVIDERS),'selection_method':m.METHOD,
      'native_closure_gate':'independent-exact-closure-input-v1','driver_contract_sha256':None,
      'candidate_manifest_sha256':'f'*64,'native_closure_validator_sha256':None}
def row(path):return {'path':path,'sha256':'a'*64}
def snapshot(p):
    return {'schema_version':1,'kind':'observed_mac15_technical_inputs_1','config_sha256':m.sha(m.raw(p)),'source_commit':p['source_commit'],
      'os':{'system':'Darwin','machine':'arm64','version':'15.9.9'},'runtimes':[{'minor':n,'micro':99,'releaselevel':'final',
        'implementation':'cpython','free_threaded':False,'binary':row('/fixture/base/cp3'+str(n)),'stdlib':'/fixture/stdlib/cp3'+str(n),
        'stdlib_index_sha256':'b'*64,'runtime_files':[]} for n in [12,13,14]],
      'tools':{k:None for k in ['cargo','rustc','maturin','sandbox','compiler','prepare','builder']},
      'helper_paths':{k:'/fixture/helpers/'+k for k in m.HELPERS},'readonly_index_sha256':'c'*64,
      'native_closure_status':'DECLARED_FILES_INDEXED_REQUIRES_INDEPENDENT_CLOSURE',
      'run_identity':{'run_id':'1','attempt':'1','nonce':'fixture-run-abcdef'},'observations_sha256':'d'*64}
def layout():return {'checkout':'/fixture/checkout','cargo_home':'/fixture/cargo-home','work_parent':'/fixture/work',
    'evidence_parent':'/fixture/evidence','venv_parent':'/fixture/venvs'}

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cofferdb-mac-dynamic-fixture-',dir='/private/tmp');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.popen=patch.object(subprocess,'Popen',side_effect=AssertionError('REAL PROCESS FORBIDDEN'));self.popen.start();self.addCleanup(self.popen.stop)
        self.connect=patch.object(socket.socket,'connect',side_effect=AssertionError('REAL NETWORK FORBIDDEN'));self.connect.start();self.addCleanup(self.connect.stop)
    def test_config_has_only_fixed_existing_sources_and_actions(self):
        m.configuration(config());p=config();p['python_providers'].append('/Users');
        with self.assertRaises(m.Hold):m.configuration(p)
    def test_action_alias_or_latest_is_rejected(self):
        p=config();p['actions']['checkout']='main'
        with self.assertRaises(m.Hold):m.configuration(p)
    def test_configuration_contains_only_technical_inputs(self):
        p=config();p['unexpected']={}
        with self.assertRaises(m.Hold):m.configuration(p)
    def test_public_source_matches_fixed_repository(self):
        for alias in ['maredengb/cofferdb','maredengb/coffered']:
            p=config();p['source_repository']=alias;self.assertIs(m.configuration(p),p)
        p=config();p['source_repository']='other/cofferdb'
        with self.assertRaises(m.Hold):m.configuration(p)
    def test_missing_driver_and_native_never_sdk(self):
        p=config();result=m.derive(p,snapshot(p),layout())
        self.assertEqual(result['status'],'INPUT_MISSING');self.assertFalse(result['sdk_executed']);self.assertEqual(result['plans'],{})
        self.assertIn('exact external runtime driver contract',result['missing'])
    def test_actual_patch_can_vary_without_fake_preknown_sha(self):
        for patchlevel in [1,99]:
            p=config();s=snapshot(p);s['runtimes'][0]['micro']=patchlevel
            self.assertEqual(m.derive(p,s,layout())['status'],'INPUT_MISSING')
    def test_free_threaded_bool_and_major_scope_rejected(self):
        p=config();s=snapshot(p);s['runtimes'][0]['free_threaded']=True
        with self.assertRaises(m.Hold):m.derive(p,s,layout())
        s=snapshot(p);s['runtimes'][0]['minor']=True
        with self.assertRaises(m.Hold):m.derive(p,s,layout())
    def test_mutated_snapshot_source_or_index_cannot_bind(self):
        p=config();s=snapshot(p);s['source_commit']='2'*40
        with self.assertRaises(m.Hold):m.derive(p,s,layout())
    def test_pure_plans_bind_technical_inputs_and_never_execute(self):
        p=config();driver={'runtime_drivers':[{}, {}, {}],
            'network_probe':{'address':'1.1.1.1','port':443},'command_seconds':120};p['driver_contract_sha256']=m.sha(m.raw(driver));p['native_closure_validator_sha256']='e'*64
        s=snapshot(p)
        for key in s['tools']:s['tools'][key]=row('/fixture/tools/'+key)
        s['tools']['prepare']=row('/fixture/checkout/tools/prepare_source.py')
        s['tools']['builder']=row('/fixture/checkout/tools/build_python.py')
        closure={'kind':'independent_exact_native_closure_1','validator_sha256':'e'*64,'snapshot_sha256':m.sha(m.raw(s)),
            'readonly_index_sha256':s['readonly_index_sha256'],'complete':True,'evidence_reference':'fixture-validation-evidence'}
        result=m.derive(p,s,layout(),driver,closure)
        self.assertEqual(set(result['plans']),{'build','empty_venvs'});self.assertEqual(result['status'],'INPUT_MISSING')
        self.assertIn('actual new empty-venv',result['missing'][0])
        self.assertFalse(result['sdk_executed']);self.assertEqual(set(result),{'status','config_sha256','snapshot_sha256','missing','plans','sdk_executed','runtime_profile_relaxed'})
    def test_source_query_is_fixed_no_sdk(self):
        self.assertIn('sysconfig',c.QUERY);self.assertNotIn('pip',c.QUERY);self.assertNotIn('cofferdb',c.QUERY);self.assertNotIn('subprocess',c.QUERY)
    def test_default_collect_never_calls_host_gate(self):
        with patch.object(c,'host',side_effect=AssertionError('HOST PROBE FORBIDDEN')),patch.object(c.argparse.ArgumentParser,'parse_args',
                return_value=type('A',(),{'action':'verify'})()),patch('sys.stdout',new=io.StringIO()) as out:
            self.assertEqual(c.main(),0);self.assertIn('NOT_RUN',out.getvalue())
    def test_non15_host_stops_before_evidence_or_query(self):
        p=config()
        with patch.object(c,'load_instantiator',return_value=m),patch.object(m,'verify_candidate_manifest'),patch.object(c.platform,'system',return_value='Darwin'),patch.object(c.platform,'machine',return_value='arm64'),\
             patch.object(c.platform,'mac_ver',return_value=('27.0',(),'')):
            with self.assertRaisesRegex(ValueError,'MAC15'):c.collect(p,'/fixture/helpers',str(self.root/'never-created'))
        self.assertFalse((self.root/'never-created').exists())
    def test_stdlib_full_small_index_and_no_links(self):
        root=self.root/'stdlib';root.mkdir();(root/'one.py').write_bytes(b'pass\n')
        rows=c.index_tree(root,c.time.monotonic()+5);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['sha256'],m.sha(b'pass\n'))
        (root/'escape').symlink_to(self.root/'outside')
        with self.assertRaises(ValueError):c.index_tree(root,c.time.monotonic()+5)
    def framework_stdlib_fixture(self, minor=12):
        versions = self.root / 'Framework/Versions'
        base = versions / f'3.{minor}'
        root = base / 'lib' / f'python3.{minor}'
        config = root / f'config-3.{minor}-darwin'
        config.mkdir(parents=True)
        target = base / 'Python'; target.write_bytes(b'fixture native bytes'); target.chmod(0o755)
        for suffix in ('a', 'dylib'):
            (config / f'libpython3.{minor}.{suffix}').symlink_to('../../../Python')
        return versions, root, config, target
    def framework_index(self, versions, root):
        def protection(path, info, **kwargs):
            if not c.stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o002:
                raise ValueError('fixture protected native target')
        with patch.object(c, 'FRAMEWORK_VERSIONS', versions), patch.object(c, 'FRAMEWORK_ALIAS_OWNER', (root.stat().st_uid, root.stat().st_gid)), patch.object(c, 'native_protection', side_effect=protection):
            return c.index_tree(root, c.time.monotonic() + 5)
    def test_exact_framework_aliases_and_target_are_indexed_and_fresh(self):
        for minor in (12, 13, 14):
            versions, root, config, target = self.framework_stdlib_fixture(minor)
            target.chmod(0o775)
            rows = self.framework_index(versions, root)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row['kind'] == 'framework_install_alias' and row['alias']['target'] == str(target) for row in rows))
            self.assertEqual(rows, self.framework_index(versions, root))
            target.write_bytes(b'changed native bytes')
            self.assertNotEqual(rows, self.framework_index(versions, root))
    def test_framework_unknown_alias_and_wrong_target_rejected(self):
        versions, root, config, target = self.framework_stdlib_fixture()
        alias = config / 'libpython3.12.a'
        alias.unlink(); alias.symlink_to('../../../Other')
        with self.assertRaises(ValueError): self.framework_index(versions, root)
        alias.unlink(); alias.symlink_to('../../../Python')
        (config / 'unknown.dylib').symlink_to('../../../Python')
        with self.assertRaises(ValueError): self.framework_index(versions, root)
    def test_framework_target_symlink_hardlink_and_writable_rejected(self):
        versions, root, config, target = self.framework_stdlib_fixture()
        target.chmod(0o777)
        with self.assertRaises(ValueError): self.framework_index(versions, root)
        target.chmod(0o755); hard = target.parent / 'hard'; c.os.link(target, hard)
        with self.assertRaises(ValueError): self.framework_index(versions, root)
        hard.unlink(); target.unlink(); target.symlink_to('Other')
        with self.assertRaises((ValueError, OSError)): self.framework_index(versions, root)
    def test_framework_alias_outside_fixed_root_and_cross_version_rejected(self):
        versions, root, config, target = self.framework_stdlib_fixture()
        with patch.object(c, 'FRAMEWORK_VERSIONS', self.root / 'elsewhere'):
            with self.assertRaises(ValueError): c.index_tree(root, c.time.monotonic() + 5)
        alias = config / 'libpython3.12.a'; alias.unlink(); alias.symlink_to('../../../Python')
        alias.rename(config / 'libpython3.13.a')
        with self.assertRaises(ValueError): self.framework_index(versions, root)
    def test_framework_alias_replacement_during_target_read_rejected(self):
        versions, root, config, target = self.framework_stdlib_fixture()
        original = c.os.read
        def changed(fd, count):
            value = original(fd, count)
            if value and not changed.done:
                changed.done = True
                alias = config / 'libpython3.12.a'; alias.unlink(); alias.symlink_to('../../../Other')
            return value
        changed.done = False
        with patch.object(c.os, 'read', side_effect=changed):
            with self.assertRaises(ValueError): self.framework_index(versions, root)
    def provider_fixture(self):
        cache=self.root/'hostedtoolcache/Python'
        framework=self.root/'Python.framework/Versions'
        cache.mkdir(parents=True);framework.mkdir(parents=True)
        binary=framework/'3.12/bin/python3.12'
        binary.parent.mkdir(parents=True);binary.write_bytes(b'fixture, never executable')
        alias=cache/'3.12.99/arm64/bin/python3.12'
        alias.parent.mkdir(parents=True)
        return cache,framework,binary,alias

    def test_declared_provider_alias_is_deduplicated_at_earliest_priority(self):
        cache,framework,binary,alias=self.provider_fixture();alias.symlink_to(binary)
        with patch.object(c,'load_instantiator',return_value=type('Plan',(),{'PROVIDERS':[str(cache),str(framework)]})):
            observed=[]
            self.assertEqual(c.candidates(12,observed),[(0,str(binary))])
            self.assertEqual([x['target_provider'] for x in observed],[1,1])
            self.assertEqual(c.candidates(13),[])

    def test_same_provider_regular_binary_remains_eligible(self):
        cache,framework,binary,alias=self.provider_fixture()
        with patch.object(c,'load_instantiator',return_value=type('Plan',(),{'PROVIDERS':[str(framework)]})):
            self.assertEqual(c.candidates(12),[(0,str(binary))])

    def test_external_and_sibling_prefix_targets_are_rejected_without_path_leak(self):
        cache,framework,binary,alias=self.provider_fixture()
        outside=Path(str(framework)+'-unapproved')/'python3.12'
        outside.parent.mkdir();outside.write_bytes(b'never execute')
        alias.symlink_to(outside)
        with patch.object(c,'load_instantiator',return_value=type('Plan',(),{'PROVIDERS':[str(cache),str(framework)]})):
            with self.assertRaisesRegex(ValueError,'outside fixed providers') as caught:c.candidates(12)
        self.assertNotIn(str(outside),str(caught.exception))
        self.assertIn(c.hashlib.sha256(c.os.fsencode(str(outside))).hexdigest(),str(caught.exception))

    def test_missing_alias_is_not_an_accepted_interpreter(self):
        cache,framework,binary,alias=self.provider_fixture();alias.symlink_to(self.root/'missing')
        with patch.object(c,'load_instantiator',return_value=type('Plan',(),{'PROVIDERS':[str(cache)]})):
            self.assertEqual(c.candidates(12),[])

    def test_link_loop_cannot_become_an_accepted_interpreter(self):
        cache,framework,binary,alias=self.provider_fixture();alias.symlink_to(alias)
        with patch.object(c,'load_instantiator',return_value=type('Plan',(),{'PROVIDERS':[str(cache)]})):
            try:result=c.candidates(12)
            except (OSError,RuntimeError):return
            self.assertEqual(result,[])

    def test_directory_drift_during_collection_is_hold(self):
        root=self.root/'changing';root.mkdir();(root/'one.py').write_bytes(b'1');original=c.os.read;changed=False
        def read(fd,n):
            nonlocal changed
            data=original(fd,n)
            if data and not changed:(root/'late.py').write_bytes(b'2');changed=True
            return data
        with patch.object(c.os,'read',side_effect=read),self.assertRaisesRegex(ValueError,'directory drift'):
            c.index_tree(root,c.time.monotonic()+5)

    def test_provider_parent_replacement_cannot_sign_old_named_path(self):
        provider=self.root/'provider';provider.mkdir();root=provider/'stdlib';root.mkdir();(root/'one.py').write_bytes(b'old')
        original=c.os.read;changed=False
        def read(fd,n):
            nonlocal changed
            data=original(fd,n)
            if data and not changed:
                provider.rename(self.root/'old-provider');provider.mkdir();(provider/'stdlib').mkdir();(provider/'stdlib/one.py').write_bytes(b'new')
                changed=True
            return data
        with patch.object(c.os,'read',side_effect=read),self.assertRaisesRegex(ValueError,'named ancestor drift'):
            c.index_tree(root,c.time.monotonic()+5)
        self.assertTrue(changed)
    def test_collect_cli_rejects_noncanonical_configuration_before_capture(self):
        p=config();selected=self.root/'config.json';selected.write_bytes(b' '+m.raw(p))
        args=type('A',(),{'action':'collect','config':str(selected),'helper_root':'/fixture/helpers','output':str(self.root/'never')})()
        with patch.object(c,'load_instantiator',return_value=m),patch.object(c.argparse.ArgumentParser,'parse_args',return_value=args),patch.object(c,'collect',side_effect=AssertionError('CAPTURE FORBIDDEN')):
            with self.assertRaises(m.Hold):c.main()
        self.assertFalse((self.root/'never').exists())
    def test_build_maintenance_paths_must_match_existing_checkout_contract(self):
        for key in ['prepare','builder']:
            p=config();s=snapshot(p);s['tools'][key]=row('/fixture/arbitrary/'+key)
            with self.assertRaisesRegex(m.Hold,'existing checkout maintenance path'):m.derive(p,s,layout())
    def test_fresh_roots_cannot_overlap_checkout_cargo_or_helpers(self):
        for forbidden in ['/fixture/checkout/new','/fixture/cargo-home/new','/fixture/helpers','/fixture']:
            p=config();l=layout();l['work_parent']=forbidden
            with self.assertRaises(m.Hold):m.derive(p,snapshot(p),l)

    def test_mutable_roots_cannot_overlap_observed_runtime(self):
        p=config();l=layout();l['work_parent']='/fixture/stdlib'
        with self.assertRaisesRegex(m.Hold,'observed runtime/tool'):m.derive(p,snapshot(p),l)

if __name__=='__main__':unittest.main()
