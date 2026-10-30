"""Seal inputs, scripts, and all random numbers before model inference."""
import ast,datetime,hashlib,json
from pathlib import Path
import numpy as np
import torch
from audit_math import self_tests

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'mammoth-derpp-baseline-20261029'
SEED=20261029

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def array_hash(value):
    a=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def state_fingerprints(c):
    return {'model':{n:array_hash(v) for n,v in c['model'].items()},
            'buffer':{n:array_hash(v) if isinstance(v,(torch.Tensor,np.ndarray)) else v for n,v in c['buffer'].items()},
            'optimizer_steps':c['optimizer_hook_steps'],'task':c['task'],'completed_epoch':c['completed_epoch']}

def savej(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')

def main():
    for name in ['prepare.py','infer.py','analyze.py','audit_math.py','verify.py','supervise.py']:
        ast.parse((ROOT/name).read_text(),filename=name)
    assert not (ROOT/'protocol-freeze.json').exists()
    manifest=json.loads((BASE/'MANIFEST.json').read_text())
    cp=BASE/'run-002/task2-final.pt'
    expected=next(v['sha256'] for v in manifest['files'] if v['path']=='run-002/task2-final.pt')
    assert sha(cp)==expected
    c=torch.load(cp,map_location='cpu',weights_only=False)
    assert c['optimizer_hook_steps']==6300 and c['buffer']['active']==500 and c['buffer']['seen']==200000
    assert set(c['buffer'])=={'examples','labels','logits','rows','arrivals','seen','active'}
    fingerprint=state_fingerprints(c)
    savej(ROOT/'input-state-fingerprints.json',fingerprint)
    tests=self_tests();savej(ROOT/'preflight-math-tests.json',tests)
    params=np.zeros((2,8,500,3),np.int16)
    for group in range(2):
        for view in range(8):
            r=np.random.default_rng(SEED+800000+100*group+view)
            params[group,view,:,:2]=r.integers(0,9,size=(500,2))
            params[group,view,:,2]=r.integers(0,2,size=500)
    r=np.random.default_rng(SEED+700001);calib=np.zeros((160,32,3),np.int16)
    calib[:,:,:2]=r.integers(0,9,size=(160,32,2));calib[:,:,2]=r.integers(0,2,size=(160,32))
    uniforms=np.random.default_rng(SEED+700000).random((160,500))
    np.savez_compressed(ROOT/'frozen-random-plan.npz',score_parameters=params,
                        calibration_parameters=calib,calibration_uniforms=uniforms)
    names=['PROTOCOL_ZH.md','prepare.py','infer.py','analyze.py','audit_math.py','verify.py','supervise.py',
           'input-state-fingerprints.json','preflight-math-tests.json','frozen-random-plan.npz']
    external={str(cp):expected,str(BASE/'baseline_v2.py'):sha(BASE/'baseline_v2.py')}
    for p in sorted((BASE/'sources').iterdir()):
        if p.is_file():external[str(p)]=sha(p)
    external[str(BASE/'MANIFEST.json')]=sha(BASE/'MANIFEST.json')
    savej(ROOT/'protocol-freeze.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'seed':SEED,'checkpoint_steps':6300,'files':{n:sha(ROOT/n) for n in names},
          'external_readonly_inputs':external,'model_forwards_before_freeze':0,
          'scientific_scope':'New frozen-model reference-extension Gate A; no branches, no A2ER reproduction'})
    print(json.dumps({'frozen':True,'checkpoint_sha256':expected,'protocol_sha256':sha(ROOT/'PROTOCOL_ZH.md'),
                      'math_tests':tests,'forward_count':0},indent=2))

if __name__=='__main__':main()
