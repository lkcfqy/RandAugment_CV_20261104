"""Small local-only, frozen-feature pilot primitives."""
from portable_dev.paths import project_path as _portable_project_path
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["VECLIB_MAXIMUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
os.environ["OPENBLAS_NUM_THREADS"]="4"
from pathlib import Path
import hashlib,json,time,io,resource
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

ROOT=Path(__file__).resolve().parent
PROJECT=_portable_project_path(__file__, '')
ZIP=PROJECT/"research-frofa-pilot-20261030/assets/EuroSAT_RGB.zip"
MODEL=PROJECT/"research-shared-theta-ad-20261030/models/dinov2-small"
SALT="augmentation-selection-eurosat-dinov2-20261102-v1"
POLICIES=["clean","weak","ra"]
LAMBDA=.001
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(20261102)
torch.set_grad_enabled(False)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2),b""): h.update(b)
    return h.hexdigest()

def key(*parts):
    return hashlib.sha256((SALT+"|"+"|".join(map(str,parts))).encode()).hexdigest()

def rng_seed(*parts): return int(key(*parts)[:16],16)&((1<<63)-1)

def atomic_json(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
    with tmp.open("w") as f:
        json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def atomic_npz(name,**arrays):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def guard(protocol):
    if time.time()>protocol["deadline_epoch"]: raise RuntimeError("hard wall-clock deadline")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss>protocol["max_rss_bytes"]: raise RuntimeError("RSS budget exceeded")
    size=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    if size>protocol["max_artifact_bytes"]-20*1024**2: raise RuntimeError("artifact limit / reserve")
    return rss,size

CLEAN=T.Compose([T.Resize(256,interpolation=IM.BICUBIC,antialias=True),T.CenterCrop(224)])
WEAK=T.Compose([T.RandomResizedCrop(224,scale=(.5,1.),ratio=(.75,4/3),interpolation=IM.BICUBIC,antialias=True),T.RandomHorizontalFlip(.5)])
RA=T.RandAugment(num_ops=2,magnitude=9,num_magnitude_bins=31,interpolation=IM.BILINEAR,fill=[124,116,104])
TENSOR=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

def make_views(img,identity,all_views=True):
    clean=TENSOR(CLEAN(img))
    if not all_views: return [clean]
    weak=[];strong=[]
    for v in range(4):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(rng_seed("geometry",identity,v))
            w=WEAK(img)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(rng_seed("randaugment",identity,v))
            a=RA(w.copy())
        weak.append(TENSOR(w));strong.append(TENSOR(a))
    return [clean]+weak+strong

def read_image(z,entry):
    im=Image.open(io.BytesIO(z.read(entry["member"]))).convert("RGB")
    arr=np.asarray(im)
    assert im.size==(64,64) and hashlib.sha256(arr.tobytes()).hexdigest()==entry["rgb_sha256"]
    return im

def load_model():
    from transformers import Dinov2Config,Dinov2Model
    from safetensors.torch import load_file
    cfg=Dinov2Config.from_json_file(str(MODEL/"config.json"));cfg._attn_implementation="eager"
    model=Dinov2Model(cfg)
    state=load_file(str(MODEL/"model.safetensors"),device="cpu")
    status=model.load_state_dict(state,strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    model=model.cpu().float().eval().requires_grad_(False)
    return model

def state_hash(model):
    h=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):
        h.update(n.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()

def encode(model,tensors,batch=16):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(tensors),batch):
            x=torch.stack(tensors[i:i+batch])
            f=model(pixel_values=x).last_hidden_state[:,0,:]
            assert torch.isfinite(f).all()
            f=torch.nn.functional.normalize(f,p=2,dim=-1,eps=1e-12)
            out.append(f.numpy().copy())
    return np.concatenate(out,axis=0)

def policy_features(cached,policy):
    if policy=="clean": return np.repeat(cached[:,0:1,:],4,axis=1)
    if policy=="weak": return cached[:,1:5,:]
    if policy=="ra": return cached[:,5:9,:]
    raise ValueError(policy)

def ridge(x,y,classes=10,lam=LAMBDA,view_weights=None):
    # Objective mean_i sum_v w_iv * ||x_iv W+b-onehot(y_i)||^2 + lam*||W||^2.
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.int64)
    n,v,d=x.shape
    vw=np.full((n,v),1/v) if view_weights is None else np.asarray(view_weights,dtype=np.float64)
    assert vw.shape==(n,v) and np.all(vw>=0) and np.allclose(vw.sum(1),1,atol=1e-14)
    weights=torch.from_numpy((vw/n).reshape(-1))
    X=torch.from_numpy(x.reshape(-1,d).copy())
    Y=torch.nn.functional.one_hot(torch.from_numpy(np.repeat(y,v)),num_classes=classes).to(torch.float64)
    xm=(X*weights[:,None]).sum(0);ym=(Y*weights[:,None]).sum(0)
    xc=X-xm;yc=Y-ym
    G=(xc.T*weights)@xc
    C=(xc.T*weights)@yc
    W=torch.linalg.solve(G+lam*torch.eye(d,dtype=torch.float64),C)
    b=ym-xm@W
    assert torch.isfinite(W).all() and torch.isfinite(b).all()
    return W.numpy(),b.numpy()

def choose(values): return int(np.argmax(np.asarray(values)))

def scores(x,W,b): return np.asarray(x,dtype=np.float64)@W+b

