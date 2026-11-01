"""Own miniature metadata fixtures only; no host probe or product execution."""
import importlib.util
import io
from types import SimpleNamespace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).with_name('probe.py')
SPEC = importlib.util.spec_from_file_location('stdlib_metadata_fixture', FILE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='stdlib-metadata-controls-', dir='/private/tmp')
        self.root = Path(self.tmp.name) / 'stdlib'
        self.root.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def test_regular_files_are_never_opened_or_read(self):
        (self.root / 'plain.py').write_bytes(b'not imported or executed')
        rows = []
        with patch.object(m.os, 'read', side_effect=AssertionError('file bytes forbidden')):
            result = m.observe(self.root, rows.append)
        self.assertEqual(result['rejected'], 0)
        self.assertEqual(rows[-1]['event'], 'stdlib_metadata_complete')
        self.assertFalse(rows[0]['file_contents_read'])

    def test_fixed_framework_alias_records_lstat_and_only_fixed_target(self):
        folder = self.root / 'config-3.12-darwin'
        folder.mkdir()
        name = folder / 'libpython3.12.dylib'
        name.symlink_to('../../../Python')
        rows = []
        target = {'role': 'framework_core', 'metadata': {'type': 'regular'}}
        with patch.object(m, 'fixed_core_metadata', return_value=target) as fixed:
            m.observe(self.root, rows.append)
        observed = next(row for row in rows if row['event'] == 'rejected_entry_metadata')
        self.assertEqual(observed['relative_path'], 'config-3.12-darwin/libpython3.12.dylib')
        self.assertEqual(observed['metadata']['type'], 'symlink')
        self.assertFalse(observed['checks']['regular'])
        self.assertEqual(observed['link_text'], '../../../Python')
        self.assertEqual(observed['fixed_target'], target)
        fixed.assert_called_once_with()

    def test_unknown_names_and_external_link_text_are_not_disclosed(self):
        hidden = 'fixture-private-account@example.invalid'
        (self.root / hidden).symlink_to('/private/fixture-only/not-for-public-output')
        rows = []
        with patch.object(m, 'fixed_core_metadata', side_effect=AssertionError('external target forbidden')):
            counts = m.observe(self.root, rows.append)
        text = json.dumps(rows)
        self.assertNotIn(hidden, text)
        self.assertNotIn('/private/fixture-only', text)
        self.assertEqual(counts['undisclosed_names'], 1)
        self.assertFalse(rows[-1]['all_rejected_paths_disclosed'])

    def test_named_alias_with_unexpected_target_is_not_followed_or_printed(self):
        folder = self.root / 'config-3.12-darwin'
        folder.mkdir()
        (folder / 'libpython3.12.a').symlink_to('/private/fixture-only/target')
        rows = []
        with patch.object(m, 'fixed_core_metadata', side_effect=AssertionError('target forbidden')):
            counts = m.observe(self.root, rows.append)
        self.assertEqual(counts['undisclosed_targets'], 1)
        self.assertFalse(rows[-1]['all_requested_metadata_disclosed'])
        self.assertNotIn('/private/fixture-only/target', json.dumps(rows))
        row = next(row for row in rows if row['event'] == 'rejected_entry_metadata')
        self.assertFalse(row['link_text_disclosed'])

    def test_hardlink_and_size_predicates_are_independent(self):
        source = self.root / 'source'
        source.write_bytes(b'x')
        os.link(source, self.root / 'alias')
        with (self.root / 'large').open('wb') as stream:
            stream.truncate(m.MAX_FILE + 1)
        rows = []
        m.observe(self.root, rows.append)
        bad = [row for row in rows if row['event'] == 'rejected_entry_metadata']
        self.assertEqual(len(bad), 3)
        self.assertEqual(sum(not row['checks']['single_link'] for row in bad), 2)
        self.assertEqual(sum(not row['checks']['within_size'] for row in bad), 1)
        self.assertTrue(all(row['checks']['regular'] for row in bad))

    def test_directory_symlink_is_not_traversed(self):
        other = Path(self.tmp.name) / 'outside'
        other.mkdir()
        (other / 'unread').write_bytes(b'not read')
        (self.root / 'linked').symlink_to(other, target_is_directory=True)
        rows = []
        counts = m.observe(self.root, rows.append)
        self.assertEqual(counts['entries'], 1)
        self.assertEqual(counts['rejected'], 1)
        self.assertNotIn('unread', json.dumps(rows))

    def test_entry_and_depth_bounds_stop_without_completion(self):
        (self.root / 'child').mkdir()
        (self.root / 'child' / 'plain').write_bytes(b'x')
        for limits, reason in [({'MAX_ENTRIES': 1}, 'entry_bound'), ({'MAX_DEPTH': 0}, 'depth_bound')]:
            rows = []
            with patch.multiple(m, **limits), self.assertRaisesRegex(m.Stop, reason):
                m.observe(self.root, rows.append)
            self.assertFalse(any(row['event'] == 'stdlib_metadata_complete' for row in rows))

    def test_rejection_bound_preserves_earlier_rows(self):
        (self.root / 'one').symlink_to('absent')
        (self.root / 'two').symlink_to('absent')
        rows = []
        with patch.object(m, 'MAX_REJECTS', 1), self.assertRaisesRegex(m.Stop, 'rejection_bound'):
            m.observe(self.root, rows.append)
        self.assertEqual(sum(row['event'] == 'rejected_entry_metadata' for row in rows), 1)

    def test_deadline_stops_before_scan(self):
        with self.assertRaisesRegex(m.Stop, 'deadline'):
            m.observe(self.root, lambda _: None, seconds=-1)

    def test_untrusted_image_text_never_appears_in_output(self):
        buffer = io.BytesIO()
        with patch.object(m.sys, 'argv', ['probe.py']), patch.object(m.sys, 'version_info', (3, 12, 10)), \
             patch.object(m.platform, 'system', return_value='Darwin'), \
             patch.object(m.platform, 'machine', return_value='arm64'), \
             patch.object(m.platform, 'mac_ver', return_value=('15.8.2', (), '')), \
             patch.object(m.resource, 'setrlimit'), patch.object(m.signal, 'signal'), \
             patch.object(m.signal, 'alarm'), patch.object(m.sys, 'stdout', SimpleNamespace(buffer=buffer)), \
             patch.dict(m.os.environ, {'ImageOS': 'fixture-private-account', 'ImageVersion': '20261028.0489.1'}), \
             patch.object(m, 'observe', side_effect=AssertionError('scan forbidden')):
            self.assertEqual(m.main(), 2)
        self.assertNotIn(b'fixture-private-account', buffer.getvalue())
        self.assertIn(b'image_identity', buffer.getvalue())

    def test_no_path_arguments_are_accepted(self):
        with patch.object(m.sys, 'argv', ['probe.py', '/not/accepted']):
            self.assertEqual(m.main(), 2)


if __name__ == '__main__':
    unittest.main()
