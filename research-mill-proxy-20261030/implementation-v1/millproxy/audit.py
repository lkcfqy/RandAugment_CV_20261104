"""Strict record completeness, safe numerical caches, and execution release checks."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .core import TRAIN_IDS,TEST_IDS,LEVELS,SEEDS,finite,readonly

def expected_forward_keys():
    keys=[]
    for split,ids in [('train',TRAIN_IDS),('test',TEST_IDS)]:
        for scene in ids:
            for level in (10,*LEVELS):keys.append((split,scene,'real',level,None))
    for method in ('A','B'):
        for scene in TEST_IDS:
            for level in LEVELS:
                for seed in SEEDS:keys.append(('test',scene,method,level,seed))
    return tuple(keys)

def validate_records(records):
    expected=set(expected_forward_keys())
    if set(records)!=expected:raise ValueError(f'Incomplete/unknown forward identities: missing={len(expected-set(records))}, extra={len(set(records)-expected)}')
    for key,record in records.items():
        if set(record)!={'psnr','delta_e76','input_sha256'}:raise ValueError('Unexpected or missing metric record fields')
        finite([record['psnr'],record['delta_e76']])
        h=record['input_sha256']
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Invalid image hash')
    return True

def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def array_sha(a):
    a=np.ascontiguousarray(a)
    header=json.dumps({'dtype':a.dtype.str,'shape':a.shape},sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'\0'+a.tobytes()).hexdigest()

def save_arrays(stem,arrays,metadata):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):raise ValueError('Object/pickle cache forbidden')
    for value in arrays.values():finite(value)
    tmp=stem.with_suffix('.npz.partial')
    with tmp.open('wb') as f:np.savez(f,**arrays)
    tmp.replace(stem.with_suffix('.npz'))
    info={'metadata':metadata,'file_sha256':file_sha(stem.with_suffix('.npz')),'arrays':{k:{'shape':list(np.asarray(v).shape),'dtype':np.asarray(v).dtype.str,'sha256':array_sha(v)} for k,v in arrays.items()}}
    stem.with_suffix('.json').write_text(json.dumps(info,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return info

def load_arrays(stem):
    stem=Path(stem);info=json.loads(stem.with_suffix('.json').read_text());path=stem.with_suffix('.npz')
    if file_sha(path)!=info['file_sha256']:raise ValueError('Cache file hash mismatch')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(info['arrays']):raise ValueError('Cache key mismatch')
        arrays={k:np.array(archive[k],copy=True) for k in archive.files}
    for k,a in arrays.items():
        rec=info['arrays'][k]
        if list(a.shape)!=rec['shape'] or a.dtype.str!=rec['dtype'] or array_sha(a)!=rec['sha256']:raise ValueError('Array identity mismatch')
        finite(a);readonly(a)
    return arrays,info['metadata']

def resource_bound(t_spent,t_forward,t_cell,t_post):
    values=np.array([t_spent,t_forward,t_cell,t_post],float)
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid timing estimate')
    return float(t_spent+1.25*(326*t_forward+32*t_cell+t_post))

def code_manifest():
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'millproxy').glob('*.py'))+list((root/'tests').glob('*.py'))
    paths += [root/name for name in ('run_oracles.py','freeze.py','PROTOCOL.md','protocol.json') if (root/name).is_file()]
    return {str(p.relative_to(root)):file_sha(p) for p in sorted(paths)}

def current_code_hash():
    return hashlib.sha256(json.dumps(code_manifest(),sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()

def require_release(receipt,freeze_sha):
    if freeze_sha!=current_code_hash():raise ValueError('Current code/protocol differs from reviewed freeze')
    if receipt.get('freeze_sha256')!=freeze_sha or receipt.get('static_review') is not True or receipt.get('protocol_confirmed') is not True:raise ValueError('Frozen implementation awaits static review and protocol confirmation')
    for field,options in [('synthetic_degeneracy',('variance_1e-12',)),('absolute_ranking_eligibility',('absolute_R',)),('unidentifiable_policy',('global',))]:
        if receipt.get(field) not in options:raise ValueError('Unresolved protocol field: '+field)
    if receipt.get('c_anchor_confirmed') is not True:raise ValueError('C_anchor control pending confirmation')
    return True
