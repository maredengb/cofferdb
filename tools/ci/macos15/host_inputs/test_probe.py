"""Local metadata controls; no hosted job, external tool or SDK execution."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('probe.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class MetadataTests(unittest.TestCase):
    def test_distinguishes_link_and_permission_rejections_without_changing_inputs(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            path = Path(folder) / 'file'; path.write_bytes(b'input'); path.chmod(0o600)
            self.assertTrue(p.inspect_file(path)['original_gate_would_accept'])
            alias = Path(folder) / 'alias'; os.link(path, alias)
            row = p.inspect_file(path)
            self.assertEqual(row['links'], 2)
            self.assertFalse(row['checks']['single_link'])
            self.assertTrue(row['checks']['protected_mode'])
            alias.unlink(); path.chmod(0o620)
            row = p.inspect_file(path)
            self.assertFalse(row['checks']['protected_mode'])
            self.assertTrue(row['checks']['single_link'])
            self.assertEqual(path.stat().st_mode & 0o777, 0o620)
            self.assertEqual(path.read_bytes(), b'input')

    def test_missing_input_does_not_disclose_path(self):
        row = p.inspect_file('/private/tmp/absent-private-account-input-xyz')
        self.assertEqual(row['status'], 'UNAVAILABLE')
        self.assertNotIn('private', str(row))

    def test_fixed_selection_and_output_labels(self):
        rows = p.fixed_inputs()
        self.assertLessEqual(len(rows), p.MAX_ROWS)
        self.assertEqual(len(rows), len({name for name, _ in rows}))
        self.assertTrue(all(path.is_absolute() for _, path in rows))


if __name__ == '__main__':
    unittest.main()
