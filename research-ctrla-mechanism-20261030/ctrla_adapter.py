"""Audited local adapter for ControlAugment e77a4c1 (MIT; see THIRD_PARTY_LICENSE).

CPU Pillow transforms are the official preprocessing backend. All model operations
in this experiment require MPS, with fallback disabled by the entrypoint.
"""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import importlib.util, math, random, time, os, json, hashlib, sys
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode

ROOT=_portable_project_path(__file__, 'research-ctrla-mechanism-20261030')
SOURCE=ROOT/'sources'/'ControlAugment-e77a4c19a3db893c120fc18b774fa290e826592b'/'src'
def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,SOURCE/filename)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
official_aug=module('official_ctrla_control','augmentations_CtrlA_Control.py')
official_model=module('official_ctrla_model','model_lib.py')
official_ctrl=module('official_ctrla_utils','CtrlA_utils.py')
OPS=list(official_aug.ControlAugment()._augmentation_space())
SIGNED=np.array(list(official_aug.ControlAugment()._augmentation_space().values()))
AFFINE=set(OPS[:6])

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,path)

def original_op(img,name,magnitude):
    return official_aug._apply_op(img,name,float(magnitude),InterpolationMode.BILINEAR,None)

def inverse_matrix(name,m):
    """Output-to-input Pillow matrix, always in original 32x32 coordinates."""
    center=[16.,16.];angle=0.;translate=[0,0];scale=1.;shear=[0.,0.]
    if name=='TranslateX':translate=[int(m*16),0]
    elif name=='TranslateY':translate=[0,int(m*16)]
    elif name=='ShearX':center=[0.,0.];shear=[45*m,0.]
    elif name=='ShearY':center=[0.,0.];shear=[0.,45*m]
    elif name=='Scale':scale=1+.5*m
    elif name=='Rotate':angle=-60*m # Pillow.rotate is opposite to affine angle.
    else:raise ValueError(name)
    vals=TF._get_inverse_affine_matrix(center,angle,translate,scale,shear)
    return np.array([vals[:3],vals[3:],[0.,0.,1.]],dtype=np.float64)

def matrix_render(img,inv,size=None):
    return img.transform(size or img.size,Image.Transform.AFFINE,tuple(inv[:2].reshape(-1)),Image.Resampling.BILINEAR,fillcolor=(0,0,0))

def _translate(x,y):return np.array([[1.,0.,x],[0.,1.,y],[0.,0.,1.]])

def render_block(img,names,mags,mode):
    if len(names)==1:
        return original_op(img,names[0],mags[0])
    if mode=='S':
        for n,m in zip(names,mags):img=original_op(img,n,m)
        return img
    invs=[inverse_matrix(n,float(m)) for n,m in zip(names,mags)]
    # Enclose every intermediate support including a conservative interpolation
    # halo. Extent is computed from ORIGINAL matrices, never enlarged magnitudes.
    lo=np.array([0.,0.]);hi=np.array([32.,32.]);alllo=lo.copy();allhi=hi.copy()
    for inv in invs:
        corners=np.array([[lo[0]-2,lo[1]-2,1],[hi[0]+2,lo[1]-2,1],[hi[0]+2,hi[1]+2,1],[lo[0]-2,hi[1]+2,1]])
        dst=corners@np.linalg.inv(inv).T
        lo=dst[:,:2].min(0)-2;hi=dst[:,:2].max(0)+2
        alllo=np.minimum(alllo,lo);allhi=np.maximum(allhi,hi)
    origin=np.floor(alllo).astype(int)-4
    end=np.ceil(allhi).astype(int)+4
    size=tuple((end-origin).tolist())
    if max(size)>1024:raise RuntimeError('Unexpected expanded canvas extent')
    padded=Image.new('RGB',size,(0,0,0));padded.paste(img,tuple((-origin).tolist()))
    if mode=='E':
        for inv in invs:
            local=_translate(*(-origin))@inv@_translate(*origin)
            padded=matrix_render(padded,local)
    elif mode=='F':
        total=np.eye(3)
        # inverse(T_last ... T_first) = inv_first ... inv_last
        for inv in invs:total=total@inv
        local=_translate(*(-origin))@total@_translate(*origin)
        padded=matrix_render(padded,local)
    else:raise ValueError(mode)
    x,y=(-origin).tolist();return padded.crop((x,y,x+32,y+32))

def render(img,op_indices,mags,mode='S'):
    assert img.size==(32,32)
    j=0
    while j<len(op_indices):
        name=OPS[int(op_indices[j])]
        if name in AFFINE:
            k=j+1
            while k<len(op_indices) and OPS[int(op_indices[k])] in AFFINE:k+=1
            img=render_block(img,[OPS[int(o)] for o in op_indices[j:k]],mags[j:k],mode);j=k
        else:
            img=original_op(img,name,mags[j]);j+=1
    return img

def tilted_quantile(q,alpha):
    # F(t)=(1-alpha)t+alpha*t*t, stable inverse; alpha=0 returns q.
    q=np.asarray(q);alpha=np.asarray(alpha)
    return np.where(alpha==0,q,2*q/((1-alpha)+np.sqrt((1-alpha)**2+4*alpha*q)))

def strengths(order,quantiles,signs,gamma,alpha,coefficient=1.):
    g=np.asarray(gamma)[order];a=np.asarray(alpha)[order]
    return coefficient*g*tilted_quantile(quantiles,a)*np.where(SIGNED[order],signs,1)

def base_image(arr,flip,crop):
    if flip:arr=arr[:,::-1]
    arr=np.pad(arr,((4,4),(4,4),(0,0)),mode='edge')
    top,left=map(int,crop)
    return Image.fromarray(np.ascontiguousarray(arr[top:top+32,left:left+32]))

def tensor_image(img):
    return torch.from_numpy(np.asarray(img).copy().transpose(2,0,1)).float().div_(255.)

def make_model(device='mps'):
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; CPU fallback prohibited')
    # Memory-layout-only compatibility adaptation: MPS max-pool backward fails
    # for the pinned source's channels-last layout on this installed runtime.
    # Architecture, values, parameters and optimizer semantics are unchanged.
    return official_model.airbench_net().to(memory_format=torch.contiguous_format).to(device)

def lr_at(epoch,horizon=20):
    # Exactly the pinned source schedule with epoch_max set to 20 actual epochs.
    return .5*.05*(1+math.cos(math.pi*(epoch-1)/(horizon+1)))

def new_optimizer(model,epoch):
    # Reset momentum every actual epoch as in the pinned implementation.
    return torch.optim.SGD(model.parameters(),lr=lr_at(epoch),momentum=.9,nesterov=True,weight_decay=2.5e-4)

def update_xi(xi,kappa,phase):
    if phase==1:return xi,0.
    delta=(1-xi)*(kappa-2.)
    if abs(delta)<.005:delta=.005*np.sign(delta)
    elif abs(delta)>.1:delta=.1*np.sign(delta)
    return float(np.clip(xi+delta,.5,.99)),float(delta)

class Budget:
    def __init__(self):
        self.path=ROOT/'budget-state.json'
        if self.path.exists():self.state=json.loads(self.path.read_text())
        else:
            now=time.time();self.state={'started_unix':now,'deadline_unix':now+8*3600,'train_images':0,'inference_images':0,'train_cap':1840000,'inference_cap':8000000,'colab_cu':0,'formal_compute_begins':'first model benchmark; all later elapsed time included'};self.save()
    def check(self,train=0,inference=0):
        if time.time()>self.state['deadline_unix']:raise RuntimeError('HARD_WALL_BUDGET_EXHAUSTED')
        if self.state['train_images']+train>self.state['train_cap']:raise RuntimeError('TRAIN_EXPOSURE_BUDGET_EXHAUSTED')
        if self.state['inference_images']+inference>self.state['inference_cap']:raise RuntimeError('INFERENCE_BUDGET_EXHAUSTED')
        self.state['train_images']+=train;self.state['inference_images']+=inference
        if train or inference:self.save() # charge each attempted batch before model work
    def save(self):
        self.state['updated_unix']=time.time();self.state['elapsed_wall_seconds']=time.time()-self.state['started_unix'];write_json(self.path,self.state)

def normalize(x,mean,std):
    return (x-torch.as_tensor(mean,device=x.device)[None,:,None,None])/torch.as_tensor(std,device=x.device)[None,:,None,None]
