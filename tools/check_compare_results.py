"""No-engine controls for strict scored-response comparison."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import compare_results as subject


def fixture():
    row = {'chunk_id': 'one', 'final_score': 0.5, 'bm25_score': 1.0,
           'vector_score': None, 'metadata_boost': 0.0,
           'normalized_scores': {'bm25': 1.0, 'vector': None, 'metadata_boost': 0.0},
           'explain': {'final_score': 0.5,
                       'scores': {'final_score': 0.5, 'bm25_raw': 1.0, 'bm25_norm': 1.0,
                                  'vector_raw': None, 'vector_norm': None, 'metadata_boost': 0.0},
                       'normalization': {'bm25_min': 0, 'bm25_max': 1.0},
                       'bm25': {'terms': [{'raw_contribution': 1.0, 'avg_chunk_length': 2.0}], 'b': 0.75},
                       'matched_term_details': [{'bm25_contribution': 1.0}]}}
    return {'results': [row], 'query_context': {'text_present': True}, 'unknown': 42}


class Controls(unittest.TestCase):
    def run_pair(self, a, b, **kwargs):
        return subject.compare(a, b, required_paths=['results.*.final_score'], **kwargs)

    def test_tolerated_score_and_exact_duplicates(self):
        a = fixture(); b = copy.deepcopy(a)
        for path in [('final_score',), ('explain', 'final_score'), ('explain', 'scores', 'final_score')]:
            node = b['results'][0]
            for key in path[:-1]: node = node[key]
            node[path[-1]] += 1e-13
        self.assertTrue(self.run_pair(a, b)['comparison_equal'])
        b['results'][0]['explain']['final_score'] += 1e-14
        with self.assertRaises(ValueError): self.run_pair(a, b)

    def test_not_every_number_is_tolerated(self):
        for transform in [lambda b: b.update(unknown=42 + 1e-13),
                          lambda b: b['results'][0]['explain']['bm25'].update(b=0.75 + 1e-13),
                          lambda b: b.update(unknown=True),
                          lambda b: b.update(unknown=None)]:
            a=fixture(); b=copy.deepcopy(a); transform(b)
            with self.assertRaises(ValueError): self.run_pair(a,b)

    def test_large_integer_and_null(self):
        a=fixture(); a['unknown']=2**63+1; b=copy.deepcopy(a); b['unknown']-=1
        with self.assertRaises(ValueError): self.run_pair(a,b)
        b=copy.deepcopy(a); b['results'][0]['vector_score']=0
        with self.assertRaises(ValueError): self.run_pair(a,b)

    def test_key_and_array_shape(self):
        a=fixture()
        for value in [{}, {'0': a['results'][0]}, []]:
            b=copy.deepcopy(a); b['results']=value
            with self.assertRaises(ValueError): self.run_pair(a,b)
        b=copy.deepcopy(a); b['results'][0]['explain']['bm25']['terms']={}
        with self.assertRaises(ValueError): self.run_pair(a,b)
        b=copy.deepcopy(a); del b['unknown']
        with self.assertRaises(ValueError): self.run_pair(a,b)

    def test_missing_coverage_and_nonfinite(self):
        a=fixture()
        with self.assertRaises(ValueError): subject.compare(a,a,required_paths=['results.*.vector_score'])
        for value in [float('nan'), float('inf'), True]:
            b=copy.deepcopy(a); b['results'][0]['explain']['normalization']['bm25_max']=value
            with self.assertRaises(ValueError): self.run_pair(b,b)
        b=copy.deepcopy(a); b['unknown']=float('inf')
        with self.assertRaises(ValueError): self.run_pair(b,b)

    def test_no_match_and_absent_text_exactness(self):
        a=fixture()
        with self.assertRaises(ValueError): self.run_pair(a,a,no_match_ids=['one'])
        row=a['results'][0];row['bm25_score']=row['explain']['scores']['bm25_raw']=0
        row['normalized_scores']['bm25']=row['explain']['scores']['bm25_norm']=0
        self.run_pair(a,a,no_match_ids=['one'])
        with self.assertRaises(ValueError): self.run_pair(a,a,no_match_ids=['absent'])
        a['query_context']['text_present']=False
        with self.assertRaises(ValueError): self.run_pair(a,a)
        row['bm25_score']=row['explain']['scores']['bm25_raw']=None
        self.run_pair(a,a)
        row['explain']['normalization']['bm25_min']=1e-20
        with self.assertRaises(ValueError): self.run_pair(a,a)

    def test_literal_keys_cannot_impersonate_array_paths(self):
        a=fixture(); a['results.*.final_score']=0.5
        b=copy.deepcopy(a); b['results.*.final_score']+=1e-13
        with self.assertRaises(ValueError): self.run_pair(a,b)

    def test_json_duplicate_keys_and_constants(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'input.json'
            for raw in ['{"a":1,"a":2}', '{"a":NaN}']:
                path.write_text(raw)
                with self.assertRaises(ValueError): subject.load(path)

    def test_exact_decimal_json_and_underflow(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'input.json'
            def decoded(literal):
                path.write_text(literal)
                return subject.load(path)[0]
            for first,second in [('9007199254740992.0','9007199254740993.0'),
                                 ('0.75','0.75000000000000001'), ('0','1e-999')]:
                a=fixture(); b=copy.deepcopy(a)
                a['unknown']=decoded(first); b['unknown']=decoded(second)
                with self.assertRaises(ValueError): self.run_pair(a,b)
            a=fixture(); b=copy.deepcopy(a);a['unknown']=decoded('0');b['unknown']=decoded('0.0')
            self.run_pair(a,b)
            a['unknown']=b['unknown']=decoded('1e999');self.run_pair(a,b)
            a=fixture(); row=a['results'][0]
            row['bm25_score']=row['explain']['scores']['bm25_raw']=decoded('1e-999')
            row['normalized_scores']['bm25']=row['explain']['scores']['bm25_norm']=0
            with self.assertRaises(ValueError): self.run_pair(a,a,no_match_ids=['one'])

    def test_order_is_exact(self):
        a=fixture(); row=copy.deepcopy(a['results'][0]); row['chunk_id']='two';a['results'].append(row)
        b=copy.deepcopy(a); b['results'].reverse()
        with self.assertRaises(ValueError): self.run_pair(a,b)


if __name__ == '__main__': unittest.main()
