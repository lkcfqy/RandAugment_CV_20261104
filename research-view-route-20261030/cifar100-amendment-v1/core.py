"""Frozen FP32 MPS implementation. All data/augmentation draws have explicit keys."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(BASE/'deps'))
import contextlib,hashlib,json,math,random,time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image,ImageOps
import timm

torch.set_num_threads(1)
DEVICE=torch.device('mps')
MEAN=(.485,.456,.406);STD=(.229,.224,.225)
WEAK=T.Compose([T.RandomResizedCrop(224,scale=(.8,1.),ratio=(.75,4/3),interpolation=InterpolationMode.BICUBIC),T.RandomHorizontalFlip(.5)])
NORM=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
EVAL=T.Compose([T.Resize((224,224),interpolation=InterpolationMode.BICUBIC),NORM])
RA=[T.RandAugment(num_ops=2,magnitude=m,num_magnitude_bins=31,interpolation=InterpolationMode.BICUBIC,fill=(124,116,104)) for m in range(10)]
RECIPES=['B1','B2','B3','B4','B5','B6','C1','C2']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def key(domain,*parts):
    s='pets-route-v1|rng-v1|'+domain+'|'+ '|'.join(map(str,parts))
    return int.from_bytes(hashlib.sha256(s.encode('utf-8')).digest()[:8],'big')&((1<<63)-1)
@contextlib.contextmanager
def cpu_rng(seed):
    old=torch.get_rng_state();torch.random.default_generator.manual_seed(seed)
    try:yield
    finally:torch.set_rng_state(old)
def tensorhash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        v=v.detach().cpu().contiguous();h.update(k.encode()+b'\0'+str(v.dtype).encode()+b'\0'+str(tuple(v.shape)).encode()+b'\0');h.update(v.numpy().tobytes())
    return h.hexdigest()
def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x
def save_pt(path,payload):
    torch.mps.synchronize();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(cpu_tree(payload),tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    digest=sha(tmp);loaded=torch.load(tmp,map_location='cpu',weights_only=True)
    assert sha(tmp)==digest
    if isinstance(loaded,dict) and 'model' in loaded:assert tensorhash(loaded['model'])==tensorhash(payload['model'])
    del loaded;tmp.replace(path)
    dump(path.with_suffix(path.suffix+'.sha256.json'),{'path':str(path.relative_to(ROOT)),'sha256':digest,'bytes':path.stat().st_size,'safe_readback':True})
    return digest

def model(seed=11,initial=None):
    assert torch.backends.mps.is_available(),'MPS not available; no CPU fallback'
    with cpu_rng(key('modelinit',seed)):
        m=timm.create_model('deit_tiny_patch16_224.fb_in1k',pretrained=False,num_classes=1000,
                            drop_rate=0.,pos_drop_rate=0.,proj_drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,patch_drop_rate=0.)
    raw=torch.load(BASE/'assets'/'deit_tiny_patch16_224-a1311bcf.pth',map_location='cpu',weights_only=True)
    assert 'model' in raw and isinstance(raw['model'],dict)
    result=m.load_state_dict(raw['model'],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert m.num_features==192 and m.num_classes==1000
    pretrained_hash=tensorhash(m.state_dict())
    m.reset_classifier(100)
    with cpu_rng(key('headinit',seed)):
        torch.nn.init.normal_(m.head.weight,0.,.01);torch.nn.init.zeros_(m.head.bias)
    if initial is not None:m.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['model'],strict=True)
    assert all(p.dtype==torch.float32 for p in m.parameters())
    assert all(not isinstance(mod,torch.nn.Dropout) or mod.p==0 for mod in m.modules())
    assert all(not hasattr(mod,'drop_prob') or mod.drop_prob==0 for mod in m.modules())
    m=m.to(DEVICE);torch.mps.synchronize()
    if not (ROOT/'model-validation.json').exists():
        dump(ROOT/'model-validation.json',{'status':'passed','safe_load':True,'strict_1000_class_load':True,'pretrained_state_sha256':pretrained_hash,
             'checkpoint_sha256':sha(BASE/'assets'/'deit_tiny_patch16_224-a1311bcf.pth'),'parameters_after_head_replacement':sum(p.numel() for p in m.parameters()),
             'feature_dim':192,'head_classes':100,'dropout_and_stochastic_depth':0,'dtype':'float32','device':'mps','timm':timm.__version__,
             'torch':torch.__version__,'torchvision':torchvision.__version__,'pretrained_cfg':m.pretrained_cfg})
    return m

class Data:
    def __init__(self):
        self.splits=json.loads((ROOT/'splits.json').read_text())
        self.rows={r['image_id']:r for r in json.loads((ROOT/'data-manifest.json').read_text())}
        self.order_cache={}
        self.images=np.load(ROOT/'data'/'original_rgb_uint8.npy',mmap_mode='r',allow_pickle=False)
        assert self.images.shape==(60000,32,32,3) and self.images.dtype==np.uint8
    def image(self,ident):
        return Image.fromarray(self.images[self.rows[ident]['raw_array_index']])
    def ids(self,seed,update):
        support=self.splits['support'][str(seed)];steps=len(support)//32;epoch,batch=divmod(update,steps)
        k=(seed,epoch)
        if k not in self.order_cache:
            gen=torch.Generator().manual_seed(key('data',seed,epoch))
            self.order_cache[k]=torch.randperm(len(support),generator=gen).tolist()
        return [support[i] for i in self.order_cache[k][batch*32:(batch+1)*32]]
    def batch(self,seed,update,kind,cache=None,warmup=0):
        ids=self.ids(seed,update);weak=[];strong=[];magnitudes=[]
        for slot,ident in enumerate(ids):
            with cpu_rng(key('weak',seed,update,slot,ident)):w=WEAK(self.image(ident))
            if kind=='ent':
                mag=0 if update<warmup else min(9,max(0,math.floor(9*cache[ident]+.5)))
                if update>=warmup:
                    with cpu_rng(key('ra',seed,update,slot,ident)):w=RA[mag](w.copy())
                magnitudes.append(mag)
            weak.append(NORM(w))
            if kind in ['dual','candidate','random','head']:
                with cpu_rng(key('ra',seed,update,slot,ident)):s=RA[9](w.copy())
                strong.append(NORM(s))
        w=torch.stack(weak).to(DEVICE);s=torch.stack(strong).to(DEVICE) if strong else None
        labels=torch.tensor([self.rows[x]['class_id'] for x in ids],device=DEVICE,dtype=torch.long)
        return ids,w,s,labels,magnitudes
    def eval_batch(self,ids):
        x=torch.stack([EVAL(self.image(i)) for i in ids]).to(DEVICE)
        y=torch.tensor([self.rows[i]['class_id'] for i in ids],device=DEVICE,dtype=torch.long)
        return x,y

class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,factor):ctx.factor=factor;return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return g*ctx.factor,None
def features(m,x):return m.forward_head(m.forward_features(x),pre_logits=True)
def losses(m,w,s,y,kind,route=0):
    if kind in ['weak','ent']:
        z=m(w);return F.cross_entropy(z,y),z
    if kind=='dual':
        z=m(torch.cat((w,s),0));return F.cross_entropy(z,torch.cat((y,y))),z
    if kind=='head':
        with torch.no_grad():f=features(m,torch.cat((w,s),0))
        z=m.head(f);return F.cross_entropy(z,torch.cat((y,y))),z
    assert kind in ['candidate','random']
    if kind=='candidate' or route==0:
        fw=ScaleGrad.apply(features(m,w),2.)
        with torch.no_grad():fs=features(m,s)
    else:
        with torch.no_grad():fw=features(m,w)
        fs=ScaleGrad.apply(features(m,s),2.)
    z=m.head(torch.cat((fw,fs),0));return F.cross_entropy(z,torch.cat((y,y))),z
def optimizer(m,low=False,head_only=False,lp=False):
    if lp:return torch.optim.AdamW(m.head.parameters(),lr=.003,weight_decay=0.,betas=(.9,.999),eps=1e-8,foreach=False)
    groups={}
    for name,p in m.named_parameters():
        if head_only and not name.startswith('head.'):continue
        lr=1e-3 if name.startswith('head.') else (1e-5 if low else 1e-4)
        wd=.01 if p.ndim>=2 else 0.
        groups.setdefault((lr,wd),[]).append(p)
    return torch.optim.AdamW([{'params':v,'lr':lr,'base_lr':lr,'weight_decay':wd} for (lr,wd),v in groups.items()],betas=(.9,.999),eps=1e-8,foreach=False)
def multiplier(step,n):
    warm=math.ceil(.05*n)
    return step/warm if step<=warm else .1+.9*(1+math.cos(math.pi*(step-warm)/(n-warm)))/2
def set_lr(opt,step,n):
    scale=multiplier(step,n)
    for g in opt.param_groups:g['lr']=g['base_lr']*scale
def update(m,opt,data,seed,update_index,kind,cache,warmup,local_step,n,forced_route=None):
    ids,w,s,y,mag=data.batch(seed,update_index,kind,cache,warmup)
    gen=torch.Generator().manual_seed(key('routing',seed,update_index))
    route=int(torch.rand((),generator=gen).item()<.5) if forced_route is None else forced_route
    set_lr(opt,local_step,n);opt.zero_grad(set_to_none=True)
    loss,z=losses(m,w,s,y,kind,route)
    if kind=='ent':
        with torch.no_grad():
            p=torch.softmax(z.detach(),dim=-1);entropy=-(p*torch.log_softmax(z.detach(),dim=-1)).sum(-1)
            q=(1-entropy/math.log(100)).clamp(0,1).cpu().tolist()
        for ident,qi in zip(ids,q):cache[ident]=qi
    loss.backward();opt.step();value=float(loss.detach().cpu())
    if not math.isfinite(value):raise RuntimeError('VALIDATION: nonfinite training loss')
    return {'loss':value,'route':route if kind=='random' else None,'magnitudes':mag}

@torch.no_grad()
def predict(m,data,ids,path=None):
    old=m.training;m.eval();outs=[];labels=[];started=time.perf_counter()
    for start in range(0,len(ids),32):
        x,y=data.eval_batch(ids[start:start+32]);outs.append(m(x).cpu());labels.append(y.cpu())
    torch.mps.synchronize();z=torch.cat(outs);y=torch.cat(labels)
    assert torch.isfinite(z).all()
    result={'ids':ids,'logits':z,'labels':y,'correct':int((z.argmax(-1)==y).sum()),'count':len(ids),'seconds':time.perf_counter()-started}
    if path:save_pt(path,result)
    m.train(old);return result
