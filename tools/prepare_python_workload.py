"""Prepare deterministic cost inputs only; no database or timed product runs.

The 100k scale JSONL bytes reproduce the existing public reconstruction recipe.
The 1k and 10k cases are new deterministic document-prefix subsets, not rewritten
historical measurements. Synthetic vectors measure mechanics, not relevance.
"""
from pathlib import Path
import argparse
import hashlib
import json


def wire(row):
    return (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_case(directory, documents, chunks, embeddings, identity, model='synthetic-lifecycle-v1'):
    directory.mkdir()
    rows = {'documents.jsonl': documents, 'chunks.jsonl': chunks, 'embeddings.jsonl': embeddings}
    facts = {}
    for name, records in rows.items():
        digest = hashlib.sha256(); count = 0; size = 0
        with (directory/name).open('xb') as stream:
            for row in records:
                data = wire(row); stream.write(data); digest.update(data); count += 1; size += len(data)
        facts[name] = {'rows': count, 'bytes': size, 'sha256': digest.hexdigest()}
    manifest = {'schema_version': '0.1', 'dataset_id': identity, 'collection': 'lifecycle',
                'embedding_dim': 768, 'embedding_model': model, 'distance_metric': 'cosine',
                'vectors_normalized': False, 'documents': 'documents.jsonl', 'chunks': 'chunks.jsonl',
                'embeddings': 'embeddings.jsonl',
                'hashes': {name.removesuffix('.jsonl')+'_sha256': row['sha256'] for name,row in facts.items()}}
    (directory/'import_manifest.json').write_bytes(wire(manifest))
    facts['import_manifest.json'] = {'sha256': hashlib.sha256(wire(manifest)).hexdigest(), 'bytes': len(wire(manifest))}
    return facts


def documents(order):
    for document in order:
        revision = 5 if document < 89 else 4
        yield {'external_id': f'doc-{document}', 'metadata': {'project': document % 20, 'revision': revision},
               'source_uri': f'project/{document}/revision/{revision}', 'title': f'Storage project {document}'}


def chunks(order):
    for document in order:
        revision = 5 if document < 89 else 4
        for ordinal in range(100):
            yield {'chunk_external_id': f'chunk-{document*100+ordinal}', 'document_external_id': f'doc-{document}',
                   'metadata': {'project': document % 20, 'revision': revision}, 'ordinal': ordinal, 'source_span': None,
                   'text': f'Project {document} revision {revision} task {ordinal}: verify durability recovery checksum and searchable index generation after document replacement.'}


def embeddings(order):
    for document in order:
        revision = 5 if document < 89 else 4
        vector = [0]*768;vector[(document+revision)%768] = 1
        for ordinal in range(100):
            yield {'chunk_external_id': f'chunk-{document*100+ordinal}', 'dim': 768,
                   'model_name': 'synthetic-lifecycle-v1', 'vector': vector}


def prepare(output):
    output = output.absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('choose a new output directory outside the source tree')
    output.mkdir(parents=True)
    order = [*range(89, 1000), *range(89)]
    facts = {}
    for count in [1000,10000,100000]:
        selected = order[:count//100]
        facts[str(count)] = write_case(output/str(count), documents(selected), chunks(selected), embeddings(selected),
                                      'v011-cost-prefix-'+str(count))
        assert facts[str(count)]['chunks.jsonl']['rows'] == count
        assert facts[str(count)]['embeddings.jsonl']['rows'] == count
    historical = {'documents.jsonl':'bd71819fbe4789e2645136fa40a10ec11981b99a26b38eb499038020e700ef36',
                  'chunks.jsonl':'6b09b63481b568572ecf543498bbdc5f1b946a41ef10e0fb8f09c6e19def61d1',
                  'embeddings.jsonl':'bcab207bb7d976e95f99d95970d2a180c1fdf8b0a6489d348d7ac1fc38d20329'}
    assert {name:facts['100000'][name]['sha256'] for name in historical} == historical
    basis = [0.0]*768;basis[93] = 1.0
    modes = [{'id':mode, 'profile':profile,'top_k':10,
              'text':None if mode=='vector' else '89',
              'embedding':None if mode=='keyword' else {'model_name':'synthetic-lifecycle-v1','dim':768,'vector':basis},
              'filter':{'doc.project':9} if mode=='hybrid-filtered' else None}
             for profile in ['expanded','balanced','constrained']
             for mode in ['keyword','vector','hybrid','hybrid-filtered']]
    (output/'scale-queries.json').write_bytes(wire(modes))
    # A separate body/large-vector case does not alter the scale reconstruction.
    owner_doc = [{'external_id':'owner','title':'Synthetic cost ownership sample','source_uri':'synthetic://cost/owner','metadata':{}}]
    long_lines = 32768
    text = 'Jade leaves grow beside a quiet river.\n' * long_lines
    owner_chunks = [{'document_external_id':'owner','chunk_external_id':'long','ordinal':0,'text':text,'metadata':{},
                     'source_span':{'start':0,'end':len(text.encode()),'line_start':1,'line_end':long_lines}}]
    owner_chunks += [{'document_external_id':'owner','chunk_external_id':f'vector-{i}','ordinal':i+1,
                      'text':f'Synthetic large-vector ownership record {i}.','metadata':{}} for i in range(256)]
    owner_model = 'synthetic-v011-ownership-constant-v2'
    owner_vectors = [{'chunk_external_id':f'vector-{i}','model_name':owner_model,'dim':768,
                      'vector':[0.125]*768} for i in range(256)]
    facts['ownership'] = write_case(output/'ownership', owner_doc, owner_chunks, owner_vectors,
                                   'v011-cost-ownership-v4', owner_model)
    owner_queries = [{'id':'long-body', 'op':'get', 'external_id':'long'}]
    owner_queries += [{'id':'vector-ownership-'+profile, 'op':'search', 'profile':profile,
                       'text':'ownership', 'embedding':{'model_name':owner_model,'dim':768,'vector':[0.125]*768},
                       'top_k':10, 'filter':None}
                      for profile in ['expanded','balanced','constrained']]
    (output/'ownership-queries.json').write_bytes(wire(owner_queries))
    result={'kind':'COST_INPUT_PREPARATION_ONLY','files':facts,
            'scale_query_sha256':hashlib.sha256((output/'scale-queries.json').read_bytes()).hexdigest(),
            'ownership_query_sha256':hashlib.sha256((output/'ownership-queries.json').read_bytes()).hexdigest(),
            'recipe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'long_body_bytes':len(text.encode()),'ownership_vector_records':256,
            'synthetic_inputs_are_semantic_quality_evidence':False,
            'historical_100k_jsonl_hashes_match':True,'database_execution_or_performance_measurement':False}
    (output/'input-manifest.json').write_bytes(wire(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(prepare(parser.parse_args().output),indent=2))
