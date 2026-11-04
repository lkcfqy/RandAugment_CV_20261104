"""Atomic state/manifest I/O shared by local acceptance and the remote worker."""
import hashlib,json,os
from pathlib import Path

def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def fsync_dir(p):
    fd=os.open(str(p),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.part')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fsync_dir(p.parent)

def canonical_hash(value):
    import torch
    h=hashlib.sha256()
    def go(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous()
            h.update(b'T');h.update(str(t.dtype).encode());h.update(json.dumps(list(t.shape)).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'D')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):go(k);go(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(b'L');h.update(str(len(x)).encode())
            for y in x:go(y)
        else:h.update(json.dumps(x,sort_keys=True,allow_nan=False).encode());h.update(b'|')
    go(value);return h.hexdigest()

def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def save_checkpoint(path,state,identity):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),'Immutable checkpoint already exists'
    frozen=cpu_tree(state);content_hash=canonical_hash(frozen)
    payload={'schema':1,'identity':identity,'state':frozen,'state_content_sha256':content_hash}
    tmp=p.with_name(p.name+'.part')
    with tmp.open('wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fsync_dir(p.parent)
    receipt={'schema':1,'kind':'checkpoint','path':str(p),'filename':p.name,'bytes':p.stat().st_size,
             'sha256':sha_file(p),'state_content_sha256':content_hash,'identity':identity,
             'phase':state['phase'],'step':state['step'],'epoch':state['epoch'],'cursor':state['cursor']}
    atomic_json(p.with_suffix('.receipt.json'),receipt)
    return receipt

def validate_checkpoint(path,receipt,expected_identity):
    import torch
    p=Path(path)
    assert p.stat().st_size==receipt['bytes'],'Checkpoint byte count mismatch'
    assert sha_file(p)==receipt['sha256'],'Checkpoint file hash mismatch'
    assert receipt['identity']==expected_identity,'Checkpoint receipt identity mismatch'
    data=torch.load(p,map_location='cpu',weights_only=True)
    assert data['schema']==1 and data['identity']==expected_identity
    assert data['state_content_sha256']==canonical_hash(data['state'])==receipt['state_content_sha256']
    s=data['state']
    for field in ('model','optimizer','scheduler','rng','permutation','step','epoch','cursor','presentations','phase','normalization'):
        assert field in s, 'Missing complete-state field '+field
    assert (s['step'],s['epoch'],s['cursor'],s['phase'])==(receipt['step'],receipt['epoch'],receipt['cursor'],receipt['phase'])
    assert set(s['rng'])=={'python','numpy','torch_cpu','torch_cuda'}
    assert sha_file(p)==receipt['sha256'],'Checkpoint reopen hash mismatch'
    return data
