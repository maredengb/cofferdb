"""Build source-identified native wheels with the selected local interpreters.

Use a prepared source export and maturin 1.10.2. This command creates a new
external output directory, retains complete build logs and never uploads files.
It is a build and artifact inspection tool, not runtime platform acceptance.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import zipfile
import tomllib
from email.parser import BytesParser


def sha(data):
    return hashlib.sha256(data).hexdigest()


def macho_target(data):
    # Read the binary target, rather than inferring its OS from an arm64 tag.
    if len(data) < 32:
        raise ValueError('native Mach-O header is truncated')
    magic, cpu, _, kind, count, size, _, _ = struct.unpack_from('<8I', data)
    if magic != 0xfeedfacf or cpu != 0x0100000c or kind != 6:
        raise ValueError('native Mach-O must be a thin arm64 dynamic library')
    end = 32 + size
    if end > len(data) or count > size // 8:
        raise ValueError('native Mach-O load command range is invalid')
    offset, builds = 32, []
    for _ in range(count):
        if offset + 8 > end:
            raise ValueError('native Mach-O load command is truncated')
        command, length = struct.unpack_from('<II', data, offset)
        if length < 8 or length % 8 or offset + length > end:
            raise ValueError('native Mach-O load command size is invalid')
        if command == 0x32:  # LC_BUILD_VERSION
            if length < 24:
                raise ValueError('native Mach-O build version is truncated')
            target, minimum, sdk, tools = struct.unpack_from('<4I', data, offset + 8)
            if length != 24 + tools * 8:
                raise ValueError('native Mach-O build tool range is invalid')
            builds.append((target, minimum, sdk))
        offset += length
    if offset != end or len(builds) != 1:
        raise ValueError('native Mach-O must have one complete build version')
    target, minimum, sdk = builds[0]
    if target != 1 or minimum != 15 << 16 or sdk == 0:  # PLATFORM_MACOS
        raise ValueError('native Mach-O platform/deployment/SDK differs from the macOS wheel contract')

    def version(value):
        parts = [value >> 16, (value >> 8) & 255, value & 255]
        return '.'.join(map(str, parts if parts[2] else parts[:2]))

    return {'platform_id':target, 'platform_name':'macos',
            'minimum_macos':version(minimum), 'build_sdk_version':version(sdk)}


def capture(arguments, cwd=None, environment=None):
    return subprocess.check_output(arguments, cwd=cwd, env=environment, text=True).strip()


def tool_identity(arguments, environment):
    executable = shutil.which(arguments[0], path=environment.get('PATH'))
    if executable is None:
        raise ValueError('required build tool is unavailable: ' + arguments[0])
    result = subprocess.run(arguments, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('could not record build tool identity: ' + arguments[0])
    return {'command':arguments, 'executable':str(Path(executable).resolve()),
            'stdout':result.stdout, 'stderr':result.stderr}


def cargo_configs(source, cargo_home):
    # Cargo searches ancestor .cargo directories and CARGO_HOME. These files
    # are environment inputs, not part of the prepared product source identity.
    candidates = {cargo_home / name for name in ('config', 'config.toml')}
    for directory in (source, *source.parents):
        candidates.update(directory / '.cargo' / name for name in ('config', 'config.toml'))
    return {str(path):{'sha256':sha(path.read_bytes()), 'size':path.stat().st_size}
            for path in sorted(candidates) if path.is_file()}


def native_toolchain(environment):
    compiler = shlex.split(environment.get('CC') or 'cc')
    result = {'compiler_driver':tool_identity(compiler + ['--version'], environment)}
    if platform.system() == 'Darwin':
        selector = environment.get('SDKROOT') or 'macosx'
        sdk_path = capture(['xcrun','--sdk',selector,'--show-sdk-path'], environment=environment)
        sdk_version = capture(['xcrun','--sdk',selector,'--show-sdk-version'], environment=environment)
        environment['SDKROOT'] = sdk_path
        linker = capture(['xcrun','--find','ld'], environment=environment)
        result.update(sdk={'path':sdk_path, 'version':sdk_version},
                      system_linker=tool_identity([linker, '-v'], environment))
    else:
        result.update(sdk=None, system_linker=tool_identity(['ld', '--version'], environment))
    result['scope'] = ('Native compiler driver and system linker inventory; explicit Rust linker flags, '
                       'target selectors and Cargo configuration are recorded separately.')
    return result


def source_inputs(source):
    for relative in ['SOURCE-INPUTS.json','bindings/python/python/cofferdb/_source.json']:
        current=source
        for part in Path(relative).parts:
            current/=part
            if current.is_symlink():
                raise ValueError('prepared metadata cannot be a symbolic link')
    inputs = json.loads((source / 'SOURCE-INPUTS.json').read_text())
    metadata = json.loads((source / 'bindings/python/python/cofferdb/_source.json').read_text())
    if sha(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()) != metadata['source_input_sha256']:
        raise ValueError('prepared source input map and package identity differ')
    actual_files = set()
    pending = [source]
    while pending:
        for path in pending.pop().iterdir():
            if path.is_symlink():
                raise ValueError('prepared source cannot contain symbolic links')
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                actual_files.add(path.relative_to(source).as_posix())
            else:
                raise ValueError('prepared source must contain only directories and regular files')
    expected_files = set(inputs) | {'SOURCE-INPUTS.json', 'bindings/python/python/cofferdb/_source.json'}
    if actual_files != expected_files:
        raise ValueError('prepared source file set differs from its input map: '
                         + json.dumps({'missing':sorted(expected_files-actual_files),
                                       'unexpected':sorted(actual_files-expected_files)}))
    for name, row in inputs.items():
        path = Path(name)
        if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in name:
            raise ValueError('prepared input escapes the source tree')
        current = source
        for part in path.parts:
            current /= part
            if current.is_symlink():
                raise ValueError('prepared source inputs cannot be symbolic links')
        data = current.read_bytes()
        if sha(data) != row['sha256'] or len(data) != row['size'] or current.stat().st_mode & 0o777 != row['mode']:
            raise ValueError('prepared source changed: ' + name)
    engine=tomllib.loads((source/'Cargo.toml').read_text())['package']['version']
    binding=tomllib.loads((source/'bindings/python/Cargo.toml').read_text())['package']['version']
    sdk=tomllib.loads((source/'pyproject.toml').read_text())['project']['version']
    if engine!=binding or engine!=metadata['engine_version'] or binding!=sdk or sdk!=metadata['sdk_version']:
        raise ValueError('static source and actual engine/binding/SDK versions differ')
    if sha((source/'Cargo.lock').read_bytes())!=metadata['engine_cargo_lock_sha256'] or sha((source/'bindings/python/Cargo.lock').read_bytes())!=metadata['binding_cargo_lock_sha256']:
        raise ValueError('static source and actual dependency locks differ')
    return inputs, metadata


def interpreter(python):
    code = ('import json,platform,sys,sysconfig; print(json.dumps(dict('
            'implementation=sys.implementation.name, major=sys.version_info.major,'
            'minor=sys.version_info.minor, version=platform.python_version(),'
            'system=platform.system(), machine=platform.machine(), libc=platform.libc_ver(),'
            'pointer_bits=64 if sys.maxsize>2**32 else 32,'
            'extension_suffix=sysconfig.get_config_var("EXT_SUFFIX"),'
            'free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")))))')
    result = json.loads(capture([str(python), '-I', '-c', code]))
    if (result['implementation'] != 'cpython' or result['major'] != 3 or result['minor'] not in {12,13,14}
            or result['free_threaded'] or result['pointer_bits'] != 64):
        raise ValueError('select regular 64-bit CPython 3.12, 3.13 or 3.14')
    pair = (result['system'], result['machine'])
    if pair not in {('Darwin','arm64'), ('Linux','x86_64')}:
        raise ValueError('select native macOS arm64 or glibc Linux x86_64 interpreters')
    if result['system']=='Linux' and result['libc'][0]!='glibc':
        raise ValueError('the Linux wheel baseline requires glibc')
    if pair != (platform.system(), platform.machine()):
        raise ValueError('this builder requires native host interpreters')
    suffix = result['extension_suffix']
    if not isinstance(suffix, str) or not suffix.startswith('.') or not suffix.endswith('.so') or '/' in suffix or '\\' in suffix:
        raise ValueError('selected interpreter has no supported native extension suffix')
    return result


def artifact(wheel, metadata, runtime, output, private_prefixes, inputs):
    tag = f"cp3{runtime['minor']}-cp3{runtime['minor']}-"
    platform_tag = 'macosx_15_0_arm64' if runtime['system']=='Darwin' else 'manylinux_2_28_x86_64'
    expected_filename = 'cofferdb-' + metadata['sdk_version'] + '-' + tag + platform_tag + '.whl'
    if wheel.name != expected_filename:
        raise ValueError('wheel filename differs from the source distribution/version and selected runtime/platform')
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or bundle.testzip() is not None:
            raise ValueError('wheel has duplicate or corrupt members')
        regular={info.filename for info in bundle.infolist() if not info.is_dir()}
        expected_python = {
            name.removeprefix('bindings/python/python/')
            for name in inputs if name.startswith('bindings/python/python/cofferdb/')
        }
        expected_native = 'cofferdb/_native' + runtime['extension_suffix']
        expected_package = expected_python | {'cofferdb/_source.json', expected_native}
        actual_package = {name for name in regular if name.startswith('cofferdb/')}
        if actual_package != expected_package:
            raise ValueError('wheel package members differ from the prepared Python/typing files and selected native module: '
                             + json.dumps({'missing':sorted(expected_package-actual_package),
                                           'unexpected':sorted(actual_package-expected_package)}))
        distribution = 'cofferdb-' + metadata['sdk_version'] + '.dist-info/'
        expected_licenses = {
            distribution + 'licenses/' + name: row['sha256']
            for name, row in inputs.items()
            if name == 'LICENSE' or name.startswith('bindings/python/licenses/')
        }
        expected_members = expected_package | set(expected_licenses) | {
            distribution + name for name in ('WHEEL', 'METADATA', 'RECORD')
        }
        if regular != expected_members:
            raise ValueError('wheel file set differs from its complete package, metadata and source notices: '
                             + json.dumps({'missing':sorted(expected_members-regular),
                                           'unexpected':sorted(regular-expected_members)}))
        for name, expected in expected_licenses.items():
            if sha(bundle.read(name)) != expected:
                raise ValueError('wheel source license differs from its prepared original: ' + name)
        record_names=[name for name in regular if name.endswith('.dist-info/RECORD')]
        if len(record_names)!=1:
            raise ValueError('wheel must have one file integrity record')
        records=list(csv.reader(io.StringIO(bundle.read(record_names[0]).decode('utf-8'))))
        if any(len(row)!=3 for row in records) or len({row[0] for row in records})!=len(records) or {row[0] for row in records}!=regular:
            raise ValueError('wheel RECORD must cover every regular member exactly once')
        for name,digest,size in records:
            if name==record_names[0]:
                if digest or size:
                    raise ValueError('wheel RECORD cannot hash itself')
                continue
            data=bundle.read(name)
            expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            if digest!=expected or size!=str(len(data)):
                raise ValueError('wheel RECORD hash or size differs from its payload: '+name)
        if json.loads(bundle.read('cofferdb/_source.json')) != metadata:
            raise ValueError('actual wheel source identity differs from prepared source')
        wheel_metadata=[name for name in names if name.endswith('.dist-info/WHEEL')]
        package_metadata=[name for name in names if name.endswith('.dist-info/METADATA')]
        if len(wheel_metadata)!=1 or len(package_metadata)!=1:
            raise ValueError('wheel must have one distribution metadata directory')
        wheel_fields=BytesParser().parsebytes(bundle.read(wheel_metadata[0]))
        package_fields=BytesParser().parsebytes(bundle.read(package_metadata[0]))
        if wheel_fields.get_all('Tag') != [tag+platform_tag]:
            raise ValueError('internal WHEEL tags differ from the selected runtime/platform')
        if package_fields.get_all('Name') != ['cofferdb'] or package_fields.get_all('Version') != [metadata['sdk_version']]:
            raise ValueError('distribution name/version differ from source identity')
        inventory_name=[name for name in names if name.endswith('/licenses/bindings/python/licenses/inventory.json')]
        if len(inventory_name)!=1:
            raise ValueError('wheel lacks its exact dependency notice inventory')
        inventory=json.loads(bundle.read(inventory_name[0]))
        if sha(bundle.read(inventory_name[0]))!=metadata['source_license_inventory_sha256']:
            raise ValueError('wheel dependency notice inventory differs from source identity')
        for dependency in inventory['dependencies']:
            for relative,expected in dependency['license_files'].items():
                matches=[name for name in names if name.endswith('/bindings/python/'+relative)]
                if len(matches)!=1 or sha(bundle.read(matches[0]))!=expected:
                    raise ValueError('wheel dependency notice differs from its recorded original')
        native = []
        for name in names:
            if name.startswith('/') or '..' in Path(name).parts or '\\' in name:
                raise ValueError('wheel member escapes its archive')
            data = bundle.read(name)
            if name.startswith('cofferdb/') and name!='cofferdb/_source.json' and not name.endswith('.so') and not name.endswith('/'):
                relative='bindings/python/python/'+name
                if relative not in inputs or sha(data)!=inputs[relative]['sha256']:
                    raise ValueError('wheel Python/typing payload differs from its prepared source: '+name)
            if any(prefix.encode() in data for prefix in private_prefixes):
                raise ValueError('wheel retains a selected absolute build path: '+name)
            if name.endswith('.so'):
                native.append((name, data))
        if len(native) != 1 or native[0][0] != expected_native:
            raise ValueError('expected only the selected native extension at ' + expected_native)
        inspection = output / (wheel.stem+'.native')
        inspection.write_bytes(native[0][1])
        if runtime['system'] == 'Darwin':
            target = macho_target(native[0][1])
            architecture = capture(['lipo','-archs',str(inspection)])
            loads = capture(['otool','-l',str(inspection)])
            libraries = capture(['otool','-L',str(inspection)]).splitlines()[1:]
            own_id = capture(['otool','-D',str(inspection)]).splitlines()[1].strip()
            libraries = [line.strip().split(' (',1)[0] for line in libraries]
            minimum_versions = re.findall(r'^\s+minos\s+([0-9.]+)\s*$', loads, re.MULTILINE)
            if architecture != 'arm64' or minimum_versions != [target['minimum_macos']] or 'cmd LC_RPATH' in loads:
                raise ValueError('native Mach-O architecture/deployment/rpath differs from the wheel contract')
            if not libraries or libraries[0] != own_id or any(not name.startswith(('/usr/lib/','/System/Library/')) for name in libraries[1:]):
                raise ValueError('Mach-O uses a non-system runtime dependency')
            sdk_versions = re.findall(r'^\s+sdk\s+([0-9.]+)\s*$', loads, re.MULTILINE)
            if sdk_versions != [target['build_sdk_version']]:
                raise ValueError('Mach-O SDK inspection differs from its actual build version')
            binary = {**target, 'architecture':architecture, 'self_image_id':own_id,
                      'runtime_dependencies':libraries[1:], 'runtime_rpath':False}
        else:
            header = capture(['readelf','-h',str(inspection)])
            versions = capture(['readelf','--version-info',str(inspection)])
            dynamic = capture(['readelf','-d',str(inspection)])
            required = sorted({tuple(map(int, match)) for match in re.findall(r'GLIBC_(\d+)\.(\d+)', versions)})
            if 'Advanced Micro Devices X86-64' not in header or any(version>(2,28) for version in required):
                raise ValueError('ELF architecture or required GLIBC exceeds the wheel contract')
            if '(RPATH)' in dynamic or '(RUNPATH)' in dynamic:
                raise ValueError('ELF retains a runtime library search path')
            binary = {'architecture':'x86_64','required_glibc_symbol_versions':['.'.join(map(str,v)) for v in required],
                      'needed_libraries':re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic), 'runtime_rpath':False}
        return {'wheel':wheel.name,'sha256':sha(wheel.read_bytes()),'native_sha256':sha(native[0][1]),
                'source_input_sha256':metadata['source_input_sha256'],'runtime':runtime,'binary':binary,
                'scope':'build and static artifact inspection; actual target-system installation remains separate'}


def build(source, output, pythons, maturin):
    source = source.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(source):
        raise ValueError('output must be a new directory outside the prepared source')
    initial, metadata = source_inputs(source)
    if capture([maturin,'--version']) != 'maturin 1.10.2':
        raise ValueError('maturin 1.10.2 is required')
    selected = [(python.resolve(strict=True),interpreter(python)) for python in pythons]
    if len({runtime['minor'] for _,runtime in selected}) != len(selected):
        raise ValueError('do not select the same interpreter minor twice')
    if platform.system() == 'Darwin' and os.environ.get('MACOSX_DEPLOYMENT_TARGET','15.0') != '15.0':
        raise ValueError('macOS deployment target must be exactly 15.0')
    output.mkdir(parents=True)
    env = dict(os.environ)
    env['CARGO_NET_OFFLINE'] = 'true'
    if platform.system() == 'Darwin':
        env['MACOSX_DEPLOYMENT_TARGET'] = '15.0'
    target = output/'cargo-target'
    env['CARGO_TARGET_DIR'] = str(target)
    rustc=env.get('RUSTC','rustc')
    cargo=env.get('CARGO','cargo')
    sysroot = capture([rustc,'--print','sysroot'])
    cargo_home = Path(env.get('CARGO_HOME',str(Path.home()/'.cargo'))).resolve()
    configuration = cargo_configs(source, cargo_home)
    toolchain = native_toolchain(env)
    flags = env.get('CARGO_ENCODED_RUSTFLAGS')
    flags = flags.split('\x1f') if flags else shlex.split(env.get('RUSTFLAGS',''))
    flags += [f'--remap-path-prefix={source}=cofferdb-'+metadata['sdk_version'],
              f'--remap-path-prefix={cargo_home}=cargo-dependencies',
              f'--remap-path-prefix={sysroot}=rust-toolchain']
    if platform.system() == 'Darwin':
        flags += ['-C',f'link-arg=-Wl,-oso_prefix,{target}/']
    env.pop('RUSTFLAGS',None)
    env['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join(flags)
    selectors = ('RUSTC','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','RUSTUP_TOOLCHAIN',
                 'CARGO_BUILD_TARGET','CARGO_HOME','CARGO_TARGET_DIR',
                 'CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER','CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER',
                 'CC','CXX','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','SDKROOT','DEVELOPER_DIR','MACOSX_DEPLOYMENT_TARGET')
    context = {'source':metadata, 'compiler_flags':flags, 'native_toolchain':toolchain,
               'cargo_configuration_files':configuration,
               'rustc_verbose':capture([rustc,'-vV'], environment=env),
               'cargo_verbose':capture([cargo,'-vV'], environment=env),
               'selected_environment':{name:env.get(name) for name in selectors}}
    (output/'build-context.json').write_text(json.dumps(context,indent=2)+'\n')
    records = []
    for python,runtime in selected:
        env['PYO3_PYTHON']=str(python)
        abi = 'cp3'+str(runtime['minor']); destination=output/abi
        command = [maturin,'build','--release','--locked','--offline','--verbose','-i',str(python),'--out',str(destination)]
        if runtime['system'] == 'Linux':
            command += ['--compatibility','manylinux_2_28']
        with (output/(abi+'.stdout')).open('wb') as stdout, (output/(abi+'.stderr')).open('wb') as stderr:
            result = subprocess.run(command,cwd=source,env=env,stdout=stdout,stderr=stderr)
        if result.returncode:
            raise RuntimeError(f'{abi} build exited {result.returncode}; retain its output and logs')
        files = list(destination.glob('*.whl'))
        if len(files)!=1:
            raise ValueError('one actual wheel is required for each selected runtime')
        record = artifact(files[0],metadata,runtime,output,[str(source),str(cargo_home),sysroot,str(target)],initial)
        record['command'] = command
        records.append(record)
        (output/'partial-build-manifest.json').write_text(json.dumps({'completed':records,'source':metadata},indent=2)+'\n')
    final, final_metadata = source_inputs(source)
    if final!=initial or final_metadata!=metadata:
        raise ValueError('prepared source changed during the build; retain these outputs')
    if cargo_configs(source, cargo_home) != configuration:
        raise ValueError('Cargo environment configuration changed during the build; retain these outputs')
    result={'schema_version':1,'kind':'native_wheel_build_not_platform_acceptance','source':metadata,
            'rustc':capture([rustc,'--version']),'cargo':capture([cargo,'--version']),
            'builder_sha256':sha(Path(__file__).read_bytes()),
            'maturin':'1.10.2','host':platform.platform(),
            'build_context':context,
            'artifacts':records}
    (output/'build-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--python',type=Path,action='append',required=True)
    parser.add_argument('--maturin',default='maturin')
    args=parser.parse_args()
    try:
        result=build(args.source,args.output,args.python,args.maturin)
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        parser.exit(1,f'wheel build failed: {error}\n')
    print(json.dumps(result,indent=2))
