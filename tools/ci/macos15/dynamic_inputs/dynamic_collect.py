"""SDK-before environment observation. Default verify never probes this host.

Collect on Mac15 only; no SDK/build/install calls.
Native dependency completeness remains an independent missing input.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import time
import types

INSTANTIATOR_SHA='d98c070430f7154565c9389eabfe874e5081932119e16c5c9cc7ee61e96e70a3'
HERE=Path(__file__).absolute().parent

def load_instantiator():
    selected=HERE/'dynamic_plan.py'
    fd=os.open(selected,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        st=os.fstat(fd)
        if not(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size<=65536):raise ValueError('instantiator source type')
        data=os.read(fd,65537)
        if len(data)!=st.st_size or hashlib.sha256(data).hexdigest()!=INSTANTIATOR_SHA:raise ValueError('instantiator SHA before compile')
    finally:os.close(fd)
    m=types.ModuleType('selected_dynamic_instantiator');m.__file__=str(selected)
    exec(compile(data,str(selected),'exec',dont_inherit=True),m.__dict__);return m

QUERY='''import json,os,platform,sys,sysconfig
print(json.dumps({'schema_version':1,'implementation':sys.implementation.name,'major':sys.version_info.major,'minor':sys.version_info.minor,'micro':sys.version_info.micro,'releaselevel':sys.version_info.releaselevel,'free_threaded':bool(sysconfig.get_config_var('Py_GIL_DISABLED')),'system':platform.system(),'machine':platform.machine(),'os_version':platform.mac_ver()[0],'executable':os.path.realpath(sys.executable),'stdlib':os.path.realpath(sysconfig.get_path('stdlib'))},sort_keys=True))
'''
MAX_FILES=24000;MAX_FILE=32*2**20;MAX_BYTES=2**30

def signature(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def directory(p):
    p=Path(p);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in p.parts[1:]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=child
        return fd
    except BaseException:os.close(fd);raise

def bytes_(p,bound=131072):
    p=Path(p);parent=directory(p.parent);fd=None
    try:
        fd=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent);before=os.fstat(fd)
        if not(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=bound):raise ValueError('bounded regular input')
        chunks=[];total=0
        while total<=bound:
            b=os.read(fd,min(65536,bound+1-total))
            if not b:break
            chunks.append(b);total+=len(b)
        if total!=before.st_size or signature(before)!=signature(os.fstat(fd)):raise ValueError('input changed')
        return b''.join(chunks)
    finally:
        if fd is not None:os.close(fd)
        os.close(parent)

def installed_policy():
    selected = HERE.parent / 'runner.py'; raw = bytes_(selected, 131072)
    expected = load_instantiator().HELPERS['runner.py']
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('Installed reader identity')
    module = types.ModuleType('observed_installed_policy'); module.__file__ = str(selected)
    exec(compile(raw, str(selected), 'exec', dont_inherit=True), module.__dict__)
    return module


def native_bytes(value, bound=128*1024**2, *, python_base=None):
    # Runtime graph members remain bounded at the original native-file limit.
    data = installed_policy().host_file(str(value), role='python' if str(value) == python_base else 'native')
    if len(data) > bound: raise ValueError('Native file bound')
    return data


def native_protection(value, info, *, python_base=None):
    return installed_policy().host_protection(str(value), info, 'python' if str(value) == python_base else 'native')


def compiler_bytes(value):
    return installed_policy().host_file(str(value), role='compiler')


def gate(deadline):
    if time.monotonic()>=deadline:raise ValueError('collection deadline')

def index_tree(root,deadline):
    """Observe complete bounded regular stdlib set, not a native dependency claim."""
    root=Path(root);parent=directory(root.parent);fd=None;rows=[];total=0;entries=0
    try:
        fd=os.open(root.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);initial=os.fstat(fd)
        def stable(current,before,above,leaf):
            if signature(before)!=signature(os.fstat(current)) or signature(before)!=signature(os.stat(leaf,dir_fd=above,follow_symlinks=False)):
                raise ValueError('stdlib directory drift')
        def walk(current,before,above,leaf,prefix,depth):
            nonlocal total,entries
            if depth>64:raise ValueError('stdlib depth')
            stable(current,before,above,leaf)
            with os.scandir(current) as it:
                for item in it:
                    gate(deadline);entries+=1
                    if entries>MAX_FILES:raise ValueError('stdlib entries')
                    info=item.stat(follow_symlinks=False);name=prefix+item.name
                    if stat.S_ISDIR(info.st_mode):
                        child=os.open(item.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=current)
                        try:walk(child,os.fstat(child),current,item.name,name+'/',depth+1)
                        finally:os.close(child)
                    else:
                        if not(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_size<=MAX_FILE):raise ValueError('stdlib link/type/file bound')
                        total+=info.st_size
                        if total>MAX_BYTES:raise ValueError('stdlib byte budget')
                        child=os.open(item.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=current)
                        try:
                            before_file=os.fstat(child)
                            if signature(before_file)!=signature(info):raise ValueError('stdlib file replacement')
                            h=hashlib.sha256();size=0
                            while True:
                                gate(deadline);chunk=os.read(child,65536)
                                if not chunk:break
                                size+=len(chunk)
                                if size>MAX_FILE:raise ValueError('stdlib file grew')
                                h.update(chunk)
                            if size!=info.st_size or signature(info)!=signature(os.fstat(child)) or signature(info)!=signature(os.stat(item.name,dir_fd=current,follow_symlinks=False)):
                                raise ValueError('stdlib file drift')
                            rows.append({'path':str(root/name),'bytes':size,'mode':stat.S_IMODE(info.st_mode),'sha256':h.hexdigest()})
                        finally:os.close(child)
            stable(current,before,above,leaf)
        walk(fd,initial,parent,root.name,'',0)
        current=directory(root)
        try:
            if signature(os.fstat(current))!=signature(initial):raise ValueError('stdlib named ancestor drift')
        finally:os.close(current)
        return sorted(rows,key=lambda r:r['path'])
    finally:
        if fd is not None:os.close(fd)
        os.close(parent)

def candidates(minor, observations=None):
    """Bounded deterministic names; no PATH search, setup-python/download/Homebrew invocation."""
    result={}
    providers=tuple(Path(source) for source in load_instantiator().PROVIDERS)
    for priority,root in enumerate(providers):
        source=str(root)
        if not root.exists():continue
        fd=directory(root)
        try:
            with os.scandir(fd) as it:
                names=[]
                for item in it:
                    if len(names)>=128:raise ValueError('provider listing bound')
                    names.append(item.name)
        finally:os.close(fd)
        if source.endswith('Python.framework/Versions'):
            names=[n for n in names if n=='3.'+str(minor)]
            paths=[root/n/'bin'/('python3.'+str(minor)) for n in names]
        elif source.endswith('hostedtoolcache/Python'):
            names=[n for n in names if __import__('re').fullmatch('3\\.'+str(minor)+r'\.\d+',n)]
            paths=[root/n/'arm64/bin'/('python3.'+str(minor)) for n in names]
        elif source.endswith('python@3.'+str(minor)):
            names=[n for n in names if __import__('re').fullmatch('3\\.'+str(minor)+r'\.\d+(?:_\d+)?',n)]
            paths=[root/n/'Frameworks/Python.framework/Versions'/('3.'+str(minor))/'bin'/('python3.'+str(minor)) for n in names]
        else:paths=[]
        for p in paths:
            if not p.exists():continue
            real=p.resolve(strict=True)
            if not any(allowed in real.parents for allowed in providers):
                # Avoid exposing an unexpected external path in public job logs.
                identity=hashlib.sha256(os.fsencode(str(real))).hexdigest()
                if observations is not None:observations.append({'minor':minor,'source_priority':priority,
                    'candidate':str(p),'target_class':'outside_fixed_providers','target_sha256':identity})
                raise ValueError('interpreter target outside fixed providers: minor=%d source=%d target_sha256=%s'
                                 % (minor,priority,identity))
            # Cache and framework entries can name the same installed binary.
            # Query it once, preserving the original earliest-provider priority.
            name=str(real);result[name]=min(priority,result.get(name,priority))
            if observations is not None:observations.append({'minor':minor,'source_priority':priority,
                'candidate':str(p),'target':name,'target_provider':next(i for i,r in enumerate(providers) if r in real.parents)})
    if len(result)>8:raise ValueError('too many candidate interpreters for bounded queries')
    return sorted((priority,name) for name,priority in result.items())

def host():
    if platform.system()!='Darwin' or platform.machine()!='arm64' or platform.mac_ver()[0].split('.')[0]!='15':raise ValueError('HOLD_REQUIRES_MAC15_ARM64')

def collect(p,helper_root,output):
    m=load_instantiator();m.configuration(p);m.verify_candidate_manifest(p['candidate_manifest_sha256']);host()
    helper_root=m.path(helper_root);out=m.path(output);parent=directory(out.parent)
    try:
        st=os.fstat(parent)
        if st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)!=0o700:raise ValueError('private new evidence parent')
        os.mkdir(out.name,0o700,dir_fd=parent)
    finally:os.close(parent)
    def save(name,value):
        raw=m.raw(value)
        if len(raw)>8*2**20:raise ValueError('metadata output bound')
        fd=os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    paths={name:str(helper_root/('macos15/'+name if name!='q_process_supervisor.py' else name)) for name in m.HELPERS}
    for name,expected in m.HELPERS.items():
        if m.sha(bytes_(paths[name]))!=expected:raise ValueError('fixed adopted helper bytes')
    supervisor=types.ModuleType('selected_original_Q');supervisor.__file__=paths['q_process_supervisor.py']
    executable=bytes_(supervisor.__file__)
    if m.sha(executable)!=m.HELPERS['q_process_supervisor.py']:raise ValueError('Q changed at load')
    exec(compile(executable,supervisor.__file__,'exec',dont_inherit=True),supervisor.__dict__)
    deadline=time.monotonic()+180;observations=[];runtimes=[];all_rows=[];save('INTENT.json',{'sdk_executed':False,'config_sha256':m.sha(m.raw(p)),'deadline_monotonic':deadline})
    for minor in [12,13,14]:
        queried=[];candidate_observations=[]
        try:selected_candidates=candidates(minor,candidate_observations)
        finally:save('CANDIDATES-%d.json'%minor,candidate_observations)
        for number,(priority,python) in enumerate(selected_candidates):
            gate(deadline);binary_sha=m.sha(bytes_(python,MAX_FILE));request=out/('query-%d-%d.json'%(minor,number))
            save(request.name,{'schema_version':1,'argv':[python,'-I','-S','-B','-c',QUERY],'cwd':str(out)})
            receipt=supervisor.capture(str(request),m.sha(bytes_(request)),python=python,environment={'PATH':'/nonexistent','LANG':'C','LC_ALL':'C','HOME':str(out),'TMPDIR':str(out)},
                output=str(out/('query-%d-%d-capture'%(minor,number))),deadline=min(deadline,time.monotonic()+20),raw_limit=65536,cleanup_reserve=2.0,supervisor_sha256=m.HELPERS['q_process_supervisor.py'])
            if not(receipt.get('state')=='PROCESS_CAPTURE_COMPLETE_NOT_ACCEPTANCE' and type(receipt.get('actual_command_exit_code')) is int and receipt['actual_command_exit_code']==0
                and receipt.get('process_group_gone') is True and receipt.get('raw_eof')=={'stdout':True,'stderr':True} and receipt.get('unread_tail_possible') is False and receipt.get('failure_reason') is None):raise ValueError('query raw/cleanup not confirmed')
            stdout=bytes_(out/('query-%d-%d-capture/stdout.raw'%(minor,number)),65536);fact=m.decode(stdout)
            if not(fact.get('implementation')=='cpython' and fact.get('major')==3 and type(fact.get('minor')) is int and fact['minor']==minor
                and type(fact.get('micro')) is int and fact['micro']>=0 and fact.get('releaselevel')=='final' and fact.get('free_threaded') is False
                and fact.get('system')=='Darwin' and fact.get('machine')=='arm64' and str(fact.get('os_version','')).split('.')[0]=='15'
                and fact.get('executable')==python and m.sha(bytes_(python,MAX_FILE))==binary_sha):raise ValueError('actual ordinary native interpreter mismatch')
            queried.append((priority,fact['micro'],python,binary_sha,fact));observations.append({'priority':priority,'fact':fact,'raw_sha256':m.sha(stdout),'binary_sha256':binary_sha})
        if not queried:raise ValueError('INPUT_MISSING CP3'+str(minor)+' allowed preinstalled provider')
        priority=min(x[0] for x in queried);chosen=[x for x in queried if x[0]==priority];micro=max(x[1] for x in chosen);chosen=[x for x in chosen if x[1]==micro]
        if len(chosen)!=1:raise ValueError('ambiguous same patch provider')
        _,_,python,binary_sha,fact=chosen[0];stdlib=m.path(fact['stdlib'])
        if not any(Path(provider) in stdlib.parents for provider in m.PROVIDERS):raise ValueError('stdlib outside allowed provider')
        rows=index_tree(stdlib,deadline);all_rows.extend(rows)
        runtimes.append({'minor':minor,'micro':micro,'releaselevel':'final','implementation':'cpython','free_threaded':False,
          'binary':{'path':python,'sha256':binary_sha},'stdlib':str(stdlib),'stdlib_index_sha256':m.sha(m.raw(rows)),'runtime_files':[]})
    save('READONLY-INDEX.json',all_rows);save('OBSERVATIONS.json',observations)
    snapshot={'schema_version':1,'kind':'observed_mac15_technical_inputs_1','config_sha256':m.sha(m.raw(p)),'source_commit':p['source_commit'],
      'os':{'system':'Darwin','machine':'arm64','version':platform.mac_ver()[0],'ImageOS':os.environ.get('ImageOS'),'ImageVersion':os.environ.get('ImageVersion')},
      'runtimes':runtimes,'tools':{k:None for k in ['cargo','rustc','maturin','sandbox','compiler','prepare','builder']},'helper_paths':paths,
      'readonly_index_sha256':m.sha(m.raw(all_rows)),'native_closure_status':'DECLARED_FILES_INDEXED_REQUIRES_INDEPENDENT_CLOSURE',
      'run_identity':{'run_id':os.environ.get('GITHUB_RUN_ID',''),'attempt':os.environ.get('GITHUB_RUN_ATTEMPT',''),'nonce':out.name},
      'observations_sha256':m.sha(m.raw(observations))}
    save('SNAPSHOT.json',snapshot)
    return {'status':'INPUT_MISSING','snapshot_sha256':m.sha(m.raw(snapshot)),
      'missing':['independent exact native dependency completeness evidence','fixed build tool bytes and runtime driver/actual wheel/venv bindings'],
      'sdk_executed':False,'sdk_automatically_invoked':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('action',nargs='?',choices=['verify','collect'],default='verify')
    for n in ['config','helper-root','output']:p.add_argument('--'+n)
    a=p.parse_args();m=load_instantiator()
    if a.action=='verify':print(m.raw({'status':'MAC15_DYNAMIC_COLLECTOR_NOT_RUN','actual_environment_probed':False,'sdk_executed':False}).decode(),end='');return 0
    result=collect(m.read_json_file(a.config),a.helper_root,a.output)
    print(m.raw(result).decode(),end='');return 2
if __name__=='__main__':raise SystemExit(main())
