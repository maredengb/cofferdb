"""Local adversarial fixtures only; no subprocess, network, SDK or real tool invocation."""
import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('candidate', Path(__file__).with_name('prepare_inputs.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Fixtures(unittest.TestCase):
    def setUp(self):
        # resolve /private/tmp alias only for fixture placement; candidate never resolves supplied paths.
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.root = Path(self.tmp.name)
        self.saved = dict(m.MATURIN)

    def tearDown(self):
        m.MATURIN.clear(); m.MATURIN.update(self.saved); self.tmp.cleanup()

    def wheel(self, bad=False, extra=False):
        files = {'maturin-1.10.2.data/scripts/maturin': b'not executable: fixture bytes',
                 'maturin-1.10.2.dist-info/licenses/LICENSE': b'fixture license'}
        record = 'maturin-1.10.2.dist-info/RECORD'; stream = io.StringIO(); w = csv.writer(stream)
        for n, b in files.items():
            h = base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode()
            w.writerow([n, 'sha256=' + ('incorrect' if bad else h), str(len(b))])
        w.writerow([record, '', '']); files[record] = stream.getvalue().encode()
        if extra: files['unlisted'] = b'extra'
        p = self.root / 'input.whl'
        with zipfile.ZipFile(p, 'w') as z:
            for n, b in files.items(): z.writestr(n, b)
        p.chmod(0o600); raw = p.read_bytes()
        m.MATURIN.update(bytes=len(raw), sha256=m.digest(raw))
        return p

    def test_archive_and_record_license_retained(self):
        p = self.wheel(); out = self.root / 'new'; receipt = m.extract_maturin(p, out)
        self.assertEqual((out/'archive.whl').read_bytes(), p.read_bytes())
        self.assertEqual((out/'metadata/maturin-1.10.2.dist-info/licenses/LICENSE').read_bytes(), b'fixture license')
        self.assertEqual(receipt['sha256'], m.digest((out/'maturin').read_bytes()))
        with self.assertRaises(FileExistsError): m.extract_maturin(p, out)

    def test_wrong_record_stops_before_output(self):
        p = self.wheel(bad=True)
        with self.assertRaises(ValueError): m.extract_maturin(p, self.root/'new')
        self.assertFalse((self.root/'new').exists())

    def test_unlisted_member_stops(self):
        p = self.wheel(extra=True)
        with self.assertRaises(ValueError): m.extract_maturin(p, self.root/'new')

    def test_wrong_frozen_archive_stops(self):
        p = self.wheel(); m.MATURIN['sha256'] = '0'*64
        with self.assertRaises(ValueError): m.extract_maturin(p, self.root/'new')

    def test_symlink_rejected(self):
        p = self.root/'real'; p.write_bytes(b'fixture'); p.chmod(0o600)
        link = self.root/'link'; link.symlink_to(p)
        with self.assertRaises(ValueError): m.regular(link)

    def test_toolchains_are_not_shims(self):
        home = self.root/'rustup'; binary = home/'toolchains/fixture-aarch64-apple-darwin/bin'
        binary.mkdir(parents=True)
        for name in ['cargo', 'rustc']:
            p = binary/name; p.write_bytes(b'fixture'); p.chmod(0o500)
        rows = m.installed_toolchains(home)
        self.assertEqual(len(rows), 1); self.assertNotIn('unusable', rows[0])
        (binary/'cargo').unlink(); (binary/'cargo').symlink_to(binary/'rustc')
        self.assertIn('unusable', m.installed_toolchains(home)[0])


if __name__ == '__main__': unittest.main()
