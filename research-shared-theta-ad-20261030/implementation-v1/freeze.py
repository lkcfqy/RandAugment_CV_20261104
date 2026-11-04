import importlib.metadata,json,time
from stheta.protocol import WORK,ASSETS,SPEC,REVISION,WEIGHT_SHA,save_json,file_sha,METHOD_NAMES,BASELINE_NAMES,SHUFFLES
from stheta.methods import permutations

if __name__=='__main__':
    assert json.loads((WORK/'reports/synthetic-tests.json').read_text())['status']=='passed'
    assert json.loads((WORK/'reports/transformers-equivalence.json').read_text())['status']=='passed'
    assert json.loads((WORK/'reports/encoder-cross-version.json').read_text())['status']=='passed'
    assert json.loads((WORK/'split-manifest.json').read_text())['status']=='ready'
    before=json.loads((WORK/'pre-data-manifest.json').read_text())
    assert before['protocol']==json.loads(json.dumps(SPEC)) and before['data_code_sha256']==file_sha(WORK/'stheta/data.py')
    split=json.loads((WORK/'split-manifest.json').read_text())
    cal={c['cluster_id'] for c in split['calibration']};assert len(cal)==64
    assert set(split['references'])=={'11','22','33'}
    maps={}
    for seed,refs in split['references'].items():
        ids=[c['cluster_id'] for c in refs];assert len(ids)==len(set(ids))==64 and not cal.intersection(ids)
        assert refs==sorted(refs,key=lambda c:(c['canonical']['pixelhash'],c['canonical']['path']))
        maps[seed]={str(p):permutations(ids,p).tolist() for p in SHUFFLES}
    save_json(WORK/'shuffle-mappings.json',maps)
    files=[{'path':str(p.relative_to(WORK)),'sha256':file_sha(p)} for p in sorted(WORK.rglob('*.py')) if not any(part.startswith('.') or part=='__pycache__' for part in p.relative_to(WORK).parts)]
    frozen={'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':SPEC,'code_files':files,
      'split_manifest_sha256':file_sha(WORK/'split-manifest.json'),'pre_data_manifest_sha256':file_sha(WORK/'pre-data-manifest.json'),
      'asset_file_manifest_sha256':file_sha(ASSETS/'reports/file-manifest.jsonl'),'model_revision':REVISION,'model_sha256':WEIGHT_SHA,
      'environment':{p:importlib.metadata.version(p) for p in ['torch','transformers','safetensors','Pillow','numpy','scipy','ImageHash','scikit-image','scikit-learn']},
      'formal_scoring_authorized':False,'preflight_source_selection':'seed11 references, first calibration cluster; only training normal images',
      'MPS_preflight_allowed_seconds':300,'test_model_forwards_allowed':0,'prohibited_preflight_outputs':['real AUROC','real TPR','real theta winner distributions']}
    frozen['method_names']=METHOD_NAMES;frozen['external_baseline_names']=BASELINE_NAMES
    frozen['shuffle_mappings_sha256']=file_sha(WORK/'shuffle-mappings.json')
    save_json(WORK/'frozen-manifest.json',frozen)
    print(json.dumps({'files':len(files),'manifest_sha256':file_sha(WORK/'frozen-manifest.json'),'formal_scoring_authorized':False}))
