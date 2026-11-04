import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from .protocol import ASSETS,WEIGHT_SHA,THETAS,file_sha

EPS=1e-8
MEAN=np.array([.485,.456,.406],dtype=np.float32)
STD=np.array([.229,.224,.225],dtype=np.float32)

def linearize(c):
    c=np.asarray(c,dtype=np.float32)
    return np.where(c<=np.float32(.04045),c/np.float32(12.92),((c+np.float32(.055))/np.float32(1.055))**np.float32(2.4)).astype(np.float32)

def srgb(u):
    u=np.asarray(u,dtype=np.float32)
    return np.where(u<=np.float32(.0031308),np.float32(12.92)*u,np.float32(1.055)*np.maximum(u,0)**np.float32(1/2.4)-np.float32(.055)).astype(np.float32)

def transform(rgb,theta):
    if tuple(theta)==(1.,1.):return rgb
    a,b=map(np.float32,theta);L=linearize(rgb);mu=L.mean(axis=(-3,-2),keepdims=True,dtype=np.float32)
    return srgb(np.clip(a*(b*L+(1-b)*mu),0,1))

def color_stats(rgb):
    L=linearize(rgb);q=np.quantile(L,[.25,.5,.75],axis=(-3,-2),method='linear').astype(np.float32)
    raw=q[2]-q[0];return q[1],np.maximum(raw,np.float32(1e-4)),raw<np.float32(1e-4)

def color_target(ref_rgbs):
    stats=[color_stats(x) for x in ref_rgbs]
    return np.median(np.stack([s[0] for s in stats]),axis=0).astype(np.float32),np.median(np.stack([s[1] for s in stats]),axis=0).astype(np.float32),int(sum(s[2].sum() for s in stats))

def color_normalize(rgb,target):
    m,s,floor=color_stats(rgb);mbar,sbar=target[:2]
    return srgb(np.clip(mbar+sbar*(linearize(rgb)-m)/s,0,1)),int(floor.sum())

def load_encoder():
    path=ASSETS/'models/dinov2-small'
    assert file_sha(path/'model.safetensors')==WEIGHT_SHA
    original=torch.load
    def reject(*args,**kwargs):raise RuntimeError('pickle checkpoint loading forbidden')
    torch.load=reject
    try:
        model,info=AutoModel.from_pretrained(str(path),trust_remote_code=False,local_files_only=True,use_safetensors=True,
             dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    finally:torch.load=original
    assert isinstance(model,Dinov2Model) and all(not info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    assert model.config._attn_implementation=='eager'
    return model.to('cpu').eval().requires_grad_(False)

def image_tensor(rgb_batch):
    x=np.stack([(x-MEAN)/STD for x in rgb_batch]).transpose(0,3,1,2).copy()
    return torch.from_numpy(x)

@torch.inference_mode()
def encode(model,rgbs,device='cpu',batch_size=8,deadline_check=None):
    result=[]
    for i in range(0,len(rgbs),batch_size):
        if deadline_check:deadline_check()
        tensor=image_tensor(rgbs[i:i+batch_size]).to(device=device,dtype=torch.float32)
        z=model(pixel_values=tensor,return_dict=True).last_hidden_state[:,1:,:]
        assert z.shape[1:]==(256,384)
        result.append(F.normalize(z,p=2,dim=-1,eps=EPS).cpu())
    return torch.cat(result,dim=0)
