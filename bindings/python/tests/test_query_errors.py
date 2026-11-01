"""Installed-package regression for the shared core's top-k diagnostics."""
from pathlib import Path
import tempfile
import unittest

from cofferdb import CofferError, Database


class QueryErrorTests(unittest.TestCase):
    def test_representable_top_k_rejection_preserves_complete_core_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with Database.create(str(Path(directory) / 'notes.coffer')) as db:
                notes = db.create_collection({'name': 'notes', 'embedding_dim': 2,
                                              'distance_metric': 'cosine'})
                notes.write_batch(documents=[{'external_id': 'log'}], chunks=[{
                    'document_external_id': 'log', 'chunk_external_id': 'leaf',
                    'ordinal': 0, 'text': 'Jade leaves grow beside a quiet river.',
                }])
                committed = db.stats()['last_commit_id']
                for top_k in [0, 9007199254740992, 18446744073709551615]:
                    with self.subTest(top_k=top_k):
                        with self.assertRaises(CofferError) as caught:
                            notes.search('jade', top_k=top_k)
                        error = caught.exception
                        self.assertEqual(error.code, 'invalid_search_params')
                        self.assertEqual(error.message, 'top_k must be a positive integer in the supported exact JSON range')
                        self.assertEqual(error.detail, {'field': 'top_k', 'min': '1',
                            'max': '9007199254740991', 'actual': str(top_k)})
                        self.assertFalse(error.recoverable)
                        self.assertIsNone(error.suggested_action)
                        self.assertEqual(db.stats()['last_commit_id'], committed)
                result = notes.search('jade', top_k=9007199254740991)
                self.assertEqual(result['top_k'], 9007199254740991)
                self.assertEqual([row['chunk_external_id'] for row in result['results']], ['leaf'])
                self.assertEqual(db.stats()['last_commit_id'], committed)


if __name__ == '__main__':
    unittest.main()
