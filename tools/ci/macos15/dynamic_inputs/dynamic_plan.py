"""Derive exact technical plans from observed Mac15 inputs; never execute them."""
import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
import re

ACTIONS={'checkout':'11d5960a326750d5838078e36cf38b85af677262','upload_artifact':'cf430e030ddbb5b0abf93d22962f4752f3646cd9'}
HELPERS={'runner.py':'f984c5a04572522efa2b5c422b854a71f827124449be353a0c83b2e57a1dc835',
 'probe.py':'c0cdbf1ce7271f7585bba59d58e51a2f22fc5e9c2a49eeb82d87ab2725b90c10',
 'workflow_build.py':'cbb6b8f55254022eb31434af9becee0be753c953bbe86bb60e99c9fa8e84b303',
 'venv_prepare.py':'b82c9d0c15776cd200c95d52c03e34f2f555e55995bbf84571b6a2ec74b1fa71',
 'q_process_supervisor.py':'39d6c8842e1a9c972fd911feb4e3c483d1eb6947fc7291b65ac6f8a3bc778e51'}
PROVIDERS=['/Users/runner/hostedtoolcache/Python','/Library/Frameworks/Python.framework/Versions',
 '/opt/homebrew/Cellar/python@3.12','/opt/homebrew/Cellar/python@3.13','/opt/homebrew/Cellar/python@3.14']
METHOD='preinstalled-provider-priority-highest-final-patch-v1'
class Hold(ValueError):pass
def need(ok,why):
    if not ok:raise Hold(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def raw(value):return (json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(',',':'))+'\n').encode()
def checksum(v):return type(v) is str and re.fullmatch('[0-9a-f]{64}',v) and v!='0'*64
def commit(v):return type(v) is str and re.fullmatch('[0-9a-f]{40}',v) and v!='0'*40
def unique(pairs):
    d={}
    for k,v in pairs:need(k not in d,'duplicate JSON');d[k]=v
    return d
def decode(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(Hold('nonfinite JSON')))
def path(v):
    need(type(v) is str and v.startswith('/') and str(Path(v))==v and '..' not in Path(v).parts
        and '\x00' not in v and '\n' not in v and v!='/','canonical path');return Path(v)
def configuration(p):
    fields={'schema_version','kind','source_repository','source_commit','helper_sha256','actions','python_providers',
      'selection_method','native_closure_gate','driver_contract_sha256','candidate_manifest_sha256','native_closure_validator_sha256'}
    need(type(p) is dict and set(p)==fields and type(p['schema_version']) is int and p['schema_version']==1
        and p['kind']=='mac15_dynamic_input_config_1','configuration shape')
    need(p['source_repository'] in ('maredengb/cofferdb','maredengb/coffered') and commit(p['source_commit']),'fixed public source identity')
    need(p['helper_sha256']==HELPERS and p['actions']==ACTIONS,'fixed existing helper/action bytes')
    need(p['python_providers']==PROVIDERS and p['selection_method']==METHOD,'fixed preinstalled selection')
    need(p['native_closure_gate']=='independent-exact-closure-input-v1','fixed closure method')
    need(checksum(p['candidate_manifest_sha256']),'collector/instantiator package identity')
    need(p['native_closure_validator_sha256'] is None or checksum(p['native_closure_validator_sha256']),'independent native validator code identity or missing')
    need(p['driver_contract_sha256'] is None or checksum(p['driver_contract_sha256']),'driver external identity or missing')
    return p

def row(r):
    need(type(r) is dict and set(r)=={'path','sha256'} and checksum(r['sha256']),'exact file row');path(r['path']);return r

def derive(p,s,layout,driver=None,closure=None):
    """Require independent snapshot and closure evidence; JSON is not its own provenance."""
    configuration(p);ps=sha(raw(p))
    fields={'schema_version','kind','config_sha256','source_commit','os','runtimes','tools','helper_paths','readonly_index_sha256',
      'native_closure_status','run_identity','observations_sha256'}
    need(type(s) is dict and set(s)==fields and type(s['schema_version']) is int and s['schema_version']==1
        and s['kind']=='observed_mac15_technical_inputs_1' and s['config_sha256']==ps and s['source_commit']==p['source_commit'],'snapshot binding')
    need(s['os'].get('system')=='Darwin' and s['os'].get('machine')=='arm64' and str(s['os'].get('version','')).split('.')[0]=='15','actual15 arm64')
    need(type(s['run_identity']) is dict and set(s['run_identity'])=={'run_id','attempt','nonce'}
        and all(type(v) is str and v for v in s['run_identity'].values()),'one job/run/attempt identity')
    need(checksum(s['readonly_index_sha256']) and checksum(s['observations_sha256']),'actual index/raw identities')
    need(set(s['helper_paths'])==set(HELPERS),'exact adopted helper paths')
    for v in s['helper_paths'].values():path(v)
    need(type(layout) is dict and set(layout)=={'checkout','cargo_home','work_parent','evidence_parent','venv_parent'},'exact new path layout')
    for v in layout.values():path(v)
    roots=[path(layout[k]) for k in ['work_parent','evidence_parent','venv_parent']]
    need(all(a!=b and a not in b.parents and b not in a.parents for n,a in enumerate(roots) for b in roots[n+1:]),'distinct fresh roots')
    immutable=[path(layout['checkout']),path(layout['cargo_home'])]+[path(v) for v in s['helper_paths'].values()]
    need(all(a!=b and a not in b.parents and b not in a.parents for a in roots for b in immutable),'fresh roots overlap checkout/cargo/maintenance inputs')
    need(type(s['runtimes']) is list and len(s['runtimes'])==3,'three actual runtimes')
    runtimes=[]
    for x,n in zip(s['runtimes'],[12,13,14]):
        need(type(x) is dict and set(x)=={'minor','micro','releaselevel','implementation','free_threaded','binary','stdlib','stdlib_index_sha256','runtime_files'},'runtime fields')
        need(type(x['minor']) is int and x['minor']==n and type(x['micro']) is int and x['micro']>=0
            and x['releaselevel']=='final' and x['implementation']=='cpython' and x['free_threaded'] is False,'ordinary native final CP3minor')
        row(x['binary']);path(x['stdlib']);need(checksum(x['stdlib_index_sha256']),'complete observed stdlib index')
        need(type(x['runtime_files']) is list and len(x['runtime_files'])<=64,'bounded runtime files')
        for f in x['runtime_files']:row(f)
        runtimes.append(x)
    need(type(s['tools']) is dict and set(s['tools'])=={'cargo','rustc','maturin','sandbox','compiler','prepare','builder'},'selected tools')
    missing=[]
    for k,v in s['tools'].items():
        if v is None:missing.append('tool:'+k)
        else:row(v)
    for key,relative in [('prepare','tools/prepare_source.py'),('builder','tools/build_python.py')]:
        if s['tools'][key] is not None:
            need(s['tools'][key]['path']==str(path(layout['checkout'])/relative),'exact existing checkout maintenance path')
    inputs=[path(x['binary']['path']) for x in runtimes]+[path(x['stdlib']) for x in runtimes]+[path(x['path']) for x in s['tools'].values() if x is not None]
    need(all(a!=v and a not in v.parents and v not in a.parents for a in roots for v in inputs),'mutable roots overlap observed runtime/tool inputs')
    # An independent fixed closure input is mandatory. A directory hash is not a dependency-completeness proof.
    if p['native_closure_validator_sha256'] is None:missing.append('fixed native closure validator code')
    if closure is None:missing.append('independent native closure evidence')
    else:
        need(type(closure) is dict and set(closure)=={'kind','snapshot_sha256','readonly_index_sha256','complete','evidence_reference','validator_sha256'}
            and closure['kind']=='independent_exact_native_closure_1' and closure['validator_sha256']==p['native_closure_validator_sha256'] and closure['snapshot_sha256']==sha(raw(s))
            and closure['readonly_index_sha256']==s['readonly_index_sha256'] and closure['complete'] is True
            and type(closure['evidence_reference']) is str and closure['evidence_reference'],'independent closure exact binding')
        need(s['native_closure_status']=='DECLARED_FILES_INDEXED_REQUIRES_INDEPENDENT_CLOSURE','snapshot is not a closure-completeness certificate')
    if driver is None or p['driver_contract_sha256'] is None:missing.append('exact external runtime driver contract')
    else:
        need(sha(raw(driver))==p['driver_contract_sha256'],'fixed external driver contract')
        need(type(driver) is dict and set(driver)=={'runtime_drivers','network_probe','command_seconds'}
            and type(driver['runtime_drivers']) is list and len(driver['runtime_drivers'])==3,'driver external schema')
    plans={}
    if not missing:
        supervisor={'path':s['helper_paths']['q_process_supervisor.py'],'sha256':HELPERS['q_process_supervisor.py']}
        py=[{'minor':x['minor'],**x['binary']} for x in runtimes]
        plans['build']={'schema_version':1,'kind':'macos15_three_abi_supervised_build_1','checkout':layout['checkout'],
          'public_commit':p['source_commit'],'prepare':s['tools']['prepare'],'builder':s['tools']['builder'],'pythons':py,
          **{k:s['tools'][k] for k in ['cargo','rustc','maturin']},'cargo_home':layout['cargo_home'],
          'work':layout['work_parent']+'/build','evidence':layout['evidence_parent']+'/build','supervisor':supervisor}
        plans['empty_venvs']={'schema_version':1,'kind':'macos15_three_empty_venvs_1','supervisor':supervisor,
          'runtimes':[{**x,'venv':layout['venv_parent']+'/cp3'+str(x['minor'])} for x in py],
          'evidence':layout['evidence_parent']+'/empty-venvs'}
        # Runtime plan identities depend on the ACTUAL new empty-venv binding/actual wheel+driver input.
        # Never invent those identities before venv creation/build happened.
        missing.append('actual new empty-venv bindings and actual wheel/driver inputs before runtime plan instantiation')
    return {'status':'INPUT_MISSING' if missing else 'EXACT_PLANS_DERIVED_NOT_EXECUTED','config_sha256':ps,
      'snapshot_sha256':sha(raw(s)),'missing':missing,'plans':plans,
      'sdk_executed':False,'runtime_profile_relaxed':False}

def read_json_file(value):
    selected=path(value);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in selected.parent.parts[1:]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=child
        child=os.open(selected.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        try:
            before=os.fstat(child);need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=2**20,'bounded JSON input')
            data=os.read(child,2**20+1);after=os.fstat(child)
            need(len(data)==before.st_size and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'JSON input changed')
            value=decode(data);need(raw(value)==data,'canonical raw JSON required for exact file SHA references');return value
        finally:os.close(child)
    finally:os.close(fd)

def verify_candidate_manifest(expected):
    need(checksum(expected),'candidate manifest external identity')
    root=Path(__file__).absolute().parent
    # This is a consistency gate; the caller must externally pin the entry too.
    selected=root/'MANIFEST.json';value=read_json_file(str(selected));data=raw(value);need(sha(data)==expected,'configuration package mismatch');need(set(value)=={'schema_version','kind','files'} and type(value['schema_version']) is int and value['schema_version']==1,'candidate manifest schema')
    need([v['path'] for v in value['files']]==['dynamic_collect.py','dynamic_plan.py','test_dynamic_inputs.py','README.md'],'exact dynamic candidate inputset')
    for item in value['files']:
        need(type(item) is dict and set(item)=={'path','bytes','sha256'} and type(item['bytes']) is int and 0<item['bytes']<=131072 and checksum(item['sha256']),'candidate file row')
        fd=os.open(root/item['path'],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size==item['bytes'],'candidate file identity')
            raw_bytes=os.read(fd,131073);need(len(raw_bytes)==item['bytes'] and sha(raw_bytes)==item['sha256'],'candidate file bytes')
        finally:os.close(fd)
    return expected

def main():
    a=argparse.ArgumentParser();a.add_argument('action',nargs='?',choices=['verify','instantiate'],default='verify')
    for n in ['config','snapshot','layout','driver','closure']:a.add_argument('--'+n)
    v=a.parse_args()
    if v.action=='verify':print(raw({'status':'DYNAMIC_PLAN_INSTANTIATOR_NOT_EXECUTED','sdk_executed':False}).decode(),end='');return 0
    def read(n):return read_json_file(n)
    pvalue=read(v.config);configuration(pvalue);verify_candidate_manifest(pvalue['candidate_manifest_sha256'])
    result=derive(pvalue,read(v.snapshot),read(v.layout),read(v.driver) if v.driver else None,read(v.closure) if v.closure else None)
    print(raw(result).decode(),end='');return 0 if result['status']!='INPUT_MISSING' else 2
if __name__=='__main__':raise SystemExit(main())
