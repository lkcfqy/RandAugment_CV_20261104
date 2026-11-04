from portable_dev.paths import project_path as _portable_project_path
import os,sys
from pathlib import Path
P=Path(__file__).resolve().parent
R=_portable_project_path(__file__, '')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='',PYTORCH_ENABLE_MPS_FALLBACK='0')
sys.path.insert(0,str(R/'research-view-route-20261030/deps'))
import json,hashlib,math,random,time,resource,contextlib,datetime
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from PIL import Image
import timm
from timm.data.random_erasing import RandomErasing

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(n):return json.loads((P/n).read_text())
def sha(q):
    h=hashlib.sha256()
    with Path(q).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def th(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def statehash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(tuple(a.shape)).encode()+b'\0');h.update(a.numpy().tobytes())
    return h.hexdigest()
def dump(n,x):
    q=P/n;q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_name(q.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(q)
def npz(n,**arrays):
    q=P/n;q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_name(q.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    tmp.replace(q);return sha(q)
def check_budget(proto):
    if time.time()>=proto['deadline_epoch']:raise RuntimeError('wall deadline reached')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>proto['host_RSS_stop_bytes']:raise RuntimeError('RSS cap reached')
    total=sum(q.stat().st_size for q in P.rglob('*') if q.is_file())
    if total>proto['new_artifact_limit_bytes']-5*1024**2:raise RuntimeError('artifact cap reserved margin reached')
    return total
@contextlib.contextmanager
def seeded(py_seed,torch_seed):
    pr=random.getstate();tr=torch.get_rng_state()
    random.seed(py_seed);torch.manual_seed(torch_seed)
    try:yield
    finally:random.setstate(pr);torch.set_rng_state(tr)

PRE=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
def load_image(row):
    assert sha(row['path'])==row['sha256']
    with Image.open(row['path']) as im:x=PRE(im.convert('RGB'))
    assert x.shape==(3,224,224) and torch.isfinite(x).all()
    return x
def correlated(pixel):
    # Rank permutation keeps every channel's original float32 values bit-for-bit.
    kernel=torch.tensor([[0,1,0],[1,4,1],[0,1,0]],dtype=torch.float64)/8
    y=F.conv2d(F.pad(pixel.double().unsqueeze(0),(1,1,1,1),mode='reflect'),kernel[None,None].repeat(3,1,1,1),groups=3)[0]
    out=torch.empty_like(pixel)
    for c in range(3):
        order=torch.argsort(y[c].flatten(),stable=True);out[c].flatten()[order]=torch.sort(pixel[c].flatten(),stable=True).values
    assert all(torch.equal(torch.sort(out[c].flatten()).values,torch.sort(pixel[c].flatten()).values) for c in range(3))
    return out
def variants(x,row):
    top,left,h,w=row['rectangle_tlhw'];result={'none':x.clone()}
    for mode in ('pixel','rand'):
        with seeded(row['rectangle_seed'],row['noise_seed']):
            result[mode]=RandomErasing(probability=1.,min_area=.2,max_area=.2,min_aspect=1.,max_aspect=1.,mode=mode,min_count=1,max_count=1,device='cpu')(x.clone())
        outside=torch.ones_like(x,dtype=torch.bool);outside[:,top:top+h,left:left+w]=False
        assert torch.equal(result[mode][outside],x[outside])
        assert not torch.equal(result[mode][:,top:top+h,left:left+w],x[:,top:top+h,left:left+w])
    pix=result['pixel'][:,top:top+h,left:left+w]
    corr=correlated(pix);result['matched_correlated']=result['pixel'].clone();result['matched_correlated'][:,top:top+h,left:left+w]=corr
    # Full-image marginal match as well as erased-rectangle match, no clipping.
    for c in range(3):assert torch.equal(torch.sort(result['pixel'][c].flatten()).values,torch.sort(result['matched_correlated'][c].flatten()).values)
    return result
def var(a):return float(a.double().var(unbiased=False))
def rms(a):return float(a.double().square().mean().sqrt())
def lag_corr(x):
    vals=[]
    for c in range(3):
        a=x[c,:,:-1].double().flatten();b=x[c,:,1:].double().flatten();a=a-a.mean();b=b-b.mean();d=(a.square().mean()*b.square().mean()).sqrt()
        vals.append(float((a*b).mean()/d) if d>0 else None)
    return vals
def input_stats(x,row):
    t,l,h,w=row['rectangle_tlhw'];e=x[:,t:t+h,l:l+w]
    return {'flat_mean':float(x.double().mean()),'flat_var':var(x),'channel_mean':[float(v.double().mean()) for v in x],
      'channel_var':[var(v) for v in x],'erase_flat_var':var(e),'erase_channel_mean':[float(v.double().mean()) for v in e],
      'erase_channel_var':[var(v) for v in e],'erase_lag1_correlations':lag_corr(e),'input_tensor_sha256':th(x)}
def ln_derivative(z,p,gamma,eps):
    z=z.double();p=p.double();g=gamma.double();a=z-z.mean(-1,keepdim=True);q=p-p.mean(-1,keepdim=True)
    s=(a.square().mean(-1,keepdim=True)+eps).sqrt()
    return g*(q/s-a*(a*q).mean(-1,keepdim=True)/s.pow(3))
def ln_statistics(patches,pos,norm,row,patch_size):
    z=patches.double()+pos.double();g=norm.weight.detach().double();b=norm.bias.detach().double();eps=norm.eps
    derivative=ln_derivative(z,pos,g,eps)
    h=1e-4
    fd=(F.layer_norm(z+h*pos.double(),(z.shape[-1],),g,b,eps)-F.layer_norm(z-h*pos.double(),(z.shape[-1],),g,b,eps))/(2*h)
    rel=float((derivative-fd).norm()/(derivative.norm()+1e-30));maxabs=float((derivative-fd).abs().max())
    assert rel<1e-5 and maxabs<1e-5,(rel,maxabs)
    n=int(math.sqrt(patches.shape[0]));t,l,hh,ww=row['rectangle_tlhw'];full=[];none=[]
    for iy in range(n):
        for ix in range(n):
            y0,x0=iy*patch_size,ix*patch_size;y1,x1=y0+patch_size,x0+patch_size
            full.append(y0>=t and y1<=t+hh and x0>=l and x1<=l+ww)
            none.append(y1<=t or y0>=t+hh or x1<=l or x0>=l+ww)
    masks={'all':torch.ones(n*n,dtype=torch.bool),'fully_erased':torch.tensor(full),'unaffected':torch.tensor(none)}
    pervar=patches.double().var(-1,unbiased=False);perS=derivative.square().mean(-1).sqrt()
    out={'patch_flat_mean':float(patches.double().mean()),'patch_flat_var':var(patches),'patch_spatial_var_mean_D':float(patches.double().var(0,unbiased=False).mean()),
         'position_flat_var':var(pos),'ln_fd_relative_error':rel,'ln_fd_max_abs':maxabs,'regions':{}}
    for label,mask in masks.items():
        assert mask.any()
        out['regions'][label]={'tokens':int(mask.sum()),'mean_feature_var':float(pervar[mask].mean()),'LN_PE_sensitivity_RMS':rms(derivative[mask])}
    return out,pervar.numpy(),perS.numpy()

def load_model(name,proto):
    torch.manual_seed(20261102);spec=proto['models'][name];assert sha(spec['path'])==spec['sha256'];start=time.monotonic()
    if name=='deit_tiny':
        model=timm.create_model('deit_tiny_patch16_224.fb_in1k',pretrained=False,num_classes=1000)
        raw=torch.load(spec['path'],map_location='cpu',weights_only=True)
        state=raw['model'];status=model.load_state_dict(state,strict=True)
        assert not status.missing_keys and not status.unexpected_keys
        norm=model.blocks[0].norm1;patch_module=model.patch_embed;pos_module=model.pos_drop
        position=model.pos_embed.detach().clone();patch_size=16
        assert isinstance(model.norm_pre,torch.nn.Identity)
    else:
        from transformers import Dinov2Config,Dinov2Model
        from safetensors.torch import load_file
        cfg=Dinov2Config.from_json_file(str(Path(spec['path']).parent/'config.json'));cfg._attn_implementation='eager'
        model=Dinov2Model(cfg);state=load_file(spec['path'],device='cpu');status=model.load_state_dict(state,strict=True)
        assert not status.missing_keys and not status.unexpected_keys
        norm=model.encoder.layer[0].norm1;patch_module=model.embeddings.patch_embeddings;pos_module=model.embeddings
        position=model.embeddings.interpolate_pos_encoding(torch.zeros(1,257,384),224,224).detach().clone();patch_size=14
    assert all(torch.isfinite(v).all() for v in state.values())
    del state
    model=model.to('cpu').eval().requires_grad_(False)
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
    return model,norm,patch_module,pos_module,position,patch_size,{'strict_missing':[],'strict_unexpected':[],'load_seconds':time.monotonic()-start,'parameters':sum(v.numel() for v in model.parameters()),'state_sha256':statehash(model),'weight_sha256':spec['sha256'],'all_parameters_frozen':True}

def outputs(model,name,x):
    if name=='deit_tiny':
        f=model.forward_features(x);return f[:,0],model.forward_head(f)
    y=model(pixel_values=x,return_dict=True);return y.last_hidden_state[:,0],None
