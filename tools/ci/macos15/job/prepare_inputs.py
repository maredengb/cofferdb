"""Inert candidate helpers. External commands belong to the existing Q supervisor."""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tomllib
import zipfile

MATURIN = {
    'url': 'https://files.pythonhosted.org/packages/4a/85/1d1b64dbb6518ee633bfde8787e251ae59428818fea7a6bdacb8008a09bd/maturin-1.10.2-py3-none-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl',
    'bytes': 17072583,
    'sha256': '7fbd997c5347649ee7987bd05a92bd5b8b07efa4ac3f8bcbf6196e07eb573d89',
}
LOCKS = {'Cargo.lock': 'a4c423f24a1d64eafab755678b0caca220c5d86be04c4b684383a90fae6afc72',
         'bindings/python/Cargo.lock': '80e68d86ca3f2949f18847bcc6d16d7257828173a7d33d12e7fd6ac1a8e4b452'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path, limit=256 * 1024 * 1024):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Literal absolute path')
    need(not any(x.is_symlink() for x in [p, *p.parents]), 'No symbolic links')
    s = p.stat()
    need(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and not s.st_mode & 0o022
         and s.st_size <= limit, 'Protected bounded regular file')
    raw = p.read_bytes()
    need(len(raw) == s.st_size, 'Stable file size')
    return raw


def installed_toolchains(rustup_home):
    """Read-only candidates, never selects a default or executes rustup/shims."""
    home = Path(rustup_home)
    need(home.is_absolute() and '..' not in home.parts, 'Explicit original rustup home')
    need(not any(x.is_symlink() for x in [home, *home.parents]), 'No linked rustup home')
    result = []
    for directory in sorted((home / 'toolchains').iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        row = {'directory': str(directory)}
        try:
            for name in ['cargo', 'rustc']:
                p = directory / 'bin' / name
                raw = regular(p)
                need(os.access(p, os.X_OK), 'Executable tool')
                row[name] = {'path': str(p), 'sha256': digest(raw)}
        except (ValueError, OSError) as error:
            row['unusable'] = str(error)
        result.append(row)
    return result


def extract_maturin(wheel, output):
    """Explicit local extraction only; retains archive plus all RECORD/license files."""
    raw = regular(wheel, MATURIN['bytes'])
    need(len(raw) == MATURIN['bytes'] and digest(raw) == MATURIN['sha256'], 'Fixed Mac wheel')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        need(len(infos) <= 256 and sum(x.file_size for x in infos) <= 128 * 1024 * 1024,
             'Bounded wheel expansion')
        names = [x.filename for x in infos]
        need(len(names) == len(set(names)), 'Unique wheel members')
        for x in infos:
            p = PurePosixPath(x.filename)
            need(not p.is_absolute() and '..' not in p.parts and '\\' not in x.filename,
                 'Safe wheel member')
            need(not stat.S_ISLNK(x.external_attr >> 16), 'No archive links')
        records = [n for n in names if n.endswith('.dist-info/RECORD')]
        need(len(records) == 1, 'One RECORD')
        listed = {}
        for n, h, size in csv.reader(io.StringIO(z.read(records[0]).decode())):
            need(n not in listed and n in names, 'Unique existing RECORD member')
            listed[n] = h
            if n == records[0]:
                need(h == '' and size == '', 'Unsigned RECORD self row')
                continue
            b = z.read(n)
            expected = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode()
            need(h == expected and size == str(len(b)), 'Wheel RECORD digest and size')
        need(set(listed) == {n for n in names if not n.endswith('/')}, 'Complete wheel RECORD')
        binaries = [n for n in names if n in {
            'maturin/maturin', 'maturin-1.10.2.data/scripts/maturin'}]
        need(len(binaries) == 1, 'One supported exact binary member; otherwise HOLD')
        dst = Path(output)
        need(dst.is_absolute() and '..' not in dst.parts, 'Explicit new extraction output')
        need(not any(x.is_symlink() for x in [dst.parent, *dst.parent.parents]), 'No output links')
        dst.mkdir(mode=0o700)  # exclusive; never reuse a prefix after failure
        (dst / 'archive.whl').write_bytes(raw)
        (dst / 'archive.whl').chmod(0o400)
        b = z.read(binaries[0]); (dst / 'maturin').write_bytes(b); (dst / 'maturin').chmod(0o500)
        # Preserve complete metadata, including RECORD, license and notices; no install/pip.
        for name in names:
            if '.dist-info/' in name and not name.endswith('/'):
                p = dst / 'metadata' / name
                p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                p.write_bytes(z.read(name)); p.chmod(0o400)
        return {'path': str(dst / 'maturin'), 'sha256': digest(b),
                'wheel_sha256': digest(raw), 'binary_member': binaries[0],
                'version_pending_existing_tool_observer': True}


def locked_packages(checkout):
    root = Path(checkout)
    packages = {}
    for name, expected in LOCKS.items():
        raw = regular(root / name, 4 * 1024 * 1024)
        need(digest(raw) == expected, 'Changed final locks require revised candidate')
        for row in tomllib.loads(raw.decode()).get('package', []):
            if 'source' not in row:
                continue
            need(row['source'] == 'registry+https://github.com/rust-lang/crates.io-index'
                 and 'checksum' in row, 'Only frozen public registry inputs')
            key = (row['name'], row['version'])
            need(key not in packages or packages[key] == row['checksum'], 'Consistent package checksums')
            packages[key] = row['checksum']
    need(len(packages) == 19, 'Expected 19 frozen public packages')
    return packages


def vendor_audit(checkout, vendor):
    expected = locked_packages(checkout); seen = {}; files = []
    for package in sorted(Path(vendor).iterdir()):
        need(not package.is_symlink() and package.is_dir(), 'Only real package directories')
        manifest = tomllib.loads(regular(package / 'Cargo.toml').decode())['package']
        key = (manifest['name'], manifest['version'])
        need(key in expected and key not in seen, 'Exact unique locked package')
        c = json.loads(regular(package / '.cargo-checksum.json').decode())
        need(c['package'] == expected[key], 'Locked registry checksum')
        actual = set()
        for p in package.rglob('*'):
            need(not p.is_symlink(), 'No vendor links')
            if p.is_file() and p.relative_to(package).as_posix() != '.cargo-checksum.json':
                actual.add(p.relative_to(package).as_posix())
        need(actual == set(c['files']), 'Complete vendor file inventory')
        for name, value in sorted(c['files'].items()):
            relative = PurePosixPath(name)
            need(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
                 'Safe vendor file name')
            raw = regular(package / name)
            need(digest(raw) == value, 'Vendor source SHA')
            files.append({'path': str((package / name).relative_to(vendor)), 'sha256': value})
        seen[key] = {'name': key[0], 'version': key[1], 'checksum': c['package'],
                     'license': manifest.get('license'), 'license_file': manifest.get('license-file')}
    need(set(seen) == set(expected), 'All 19 packages present')
    return {'packages': list(seen.values()), 'files': files, 'sdk_tested': False}


def command_recipe(checkout, cargo, rustc, cargo_home, vendor, private_home, wheel):
    """Data only. Outer Q must validate source identity, execute once, retain raw receipts."""
    paths = [Path(x) for x in [checkout, cargo, rustc, cargo_home, vendor, private_home, wheel]]
    need(all(p.is_absolute() and '..' not in p.parts for p in paths), 'Explicit absolute paths')
    locked_packages(checkout)
    for p in [cargo, rustc]:
        regular(p); need(os.access(p, os.X_OK), 'Executable regular tool')
    base = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'HOME': private_home,
            'TMPDIR': private_home, 'CARGO_HOME': cargo_home, 'CARGO': cargo, 'RUSTC': rustc,
            'CARGO_NET_RETRY': '0'}
    config = '[source.crates-io]\nreplace-with = "cofferdb-vendor"\n[source.cofferdb-vendor]\ndirectory = ' + json.dumps(vendor) + '\n'
    manifests = [str(Path(checkout) / x) for x in ['Cargo.toml', 'bindings/python/Cargo.toml']]
    return {'environment_network_prepare': dict(base, CARGO_NET_OFFLINE='false'),
            'environment_offline': dict(base, CARGO_NET_OFFLINE='true'),
            'download': {'argv': ['/usr/bin/curl', '--fail', '--silent', '--show-error',
                        '--proto', '=https', '--max-time', '180', '--max-filesize', str(MATURIN['bytes']),
                        '--output', wheel, MATURIN['url']], 'seconds': 180},
            'vendor': {'argv': [cargo, 'vendor', '--locked', '--no-delete', '--versioned-dirs',
                       '--manifest-path', manifests[0], '--sync', manifests[1], vendor], 'seconds': 1200},
            'config_path': str(Path(cargo_home) / 'config.toml'), 'config_bytes_utf8': config,
            'offline_metadata': [{'argv': [cargo, 'metadata', '--frozen', '--format-version', '1',
                                 '--manifest-path', manifest], 'seconds': 120} for manifest in manifests],
            'capture_contract': 'Existing Q only; one attempt per command, raw receipts, fail/timeout stops; total 1800s',
            'ready_for_execution': False}


if __name__ == '__main__':
    print('INPUT_PREPARATION_CANDIDATE_NOT_EXECUTED')
