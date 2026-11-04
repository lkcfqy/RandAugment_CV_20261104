from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib,json,os,time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=_portable_project_path(__file__, '')
PREFLIGHT=PROJECT/'augmentation-selection-cifar100-preflight-20261102'
MODELS={'dinov2_small':384,'deit_tiny':192}
POLICIES=['clean','weak','ra']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def save_npz(p,**arrays):
    p=Path(p);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def durable_array(p,array):
    p=Path(p);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def finish_memmap(mm,tmp,final):
    mm.flush()
    with Path(tmp).open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,final)
def canonical_heads():
    return [{'index':i,'episode':e,'B':B,'policy':p,'fold':f}
            for i,(e,B,p,f) in enumerate((e,B,p,f) for e in range(12) for B in [8,16] for p in POLICIES for f in [0,1,2,3,-1])]

def validate_joint_gate(root,protocol_hash,identity_hash,dimensions=MODELS):
    root=Path(root);answer={}
    for model,dim in dimensions.items():
        d=root/model;s=read(d/'FIT_SEAL.json')
        assert s['status']=='FIT_SEALED' and s['protocol_sha256']==protocol_hash and s['identity_sha256']==identity_hash
        assert s['head_count']==360 and len(s['choices'])==24
        assert {(c['episode'],c['B']) for c in s['choices']}=={(e,B) for e in range(12) for B in [8,16]}
        for filename,h in s['files'].items():assert sha(d/filename)==h,(model,filename)
        assert read(d/'HEAD_IDENTITIES.json')==canonical_heads()
        W=np.load(d/'W.npy',mmap_mode='r',allow_pickle=False);b=np.load(d/'b.npy',mmap_mode='r',allow_pickle=False)
        assert W.shape==(360,dim,100) and b.shape==(360,100) and W.dtype==b.dtype==np.float64
        assert np.isfinite(W).all() and np.isfinite(b).all()
        for c in s['choices']:
            assert c['selected_index']==int(np.argmax(c['cv_correct']))
            expected=[h['index'] for p in POLICIES for h in canonical_heads() if (h['episode'],h['B'],h['policy'],h['fold'])==(c['episode'],c['B'],p,-1)]
            assert c['full_head_indices']==expected
        answer[model]=sha(d/'FIT_SEAL.json')
    return answer

class Guard:
    def __init__(self):
        self.limits=read(ROOT/'LIMITS.json');self.started=time.monotonic()
        self.allowance=self.limits['deadline_epoch']-time.time()
    def remaining(self):return min(self.allowance-(time.monotonic()-self.started),self.limits['deadline_epoch']-time.time())
    def check(self,reserve=180):
        if self.remaining()<reserve:raise RuntimeError('Cumulative deadline / save-and-exit reserve reached')
        size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+self.limits['preflight_bytes']
        if size>500*2**20-20*2**20:raise RuntimeError('500MiB cumulative artifact limit/reserve')
        return size
