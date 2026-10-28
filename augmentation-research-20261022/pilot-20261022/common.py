import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.utils import parametrize
from torchvision.models import vit_b_16
from torchvision import transforms as T
from torchvision.transforms import functional as TF,InterpolationMode

ROOT=Path(__file__).resolve().parent
PLAN_PATH=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/candidate-data-plan.json')
MEAN=(.485,.456,.406);STD=(.229,.224,.225)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(path,v):
    with Path(path).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def keyseed(*parts):return int(hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()[:15],16)

def weak_pil(im,source_id,epoch,seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(keyseed('weak-v1',seed,epoch,source_id))
        i,j,h,w=T.RandomResizedCrop.get_params(im,scale=(.8,1.),ratio=(.9,1.1))
        out=TF.resized_crop(im,i,j,h,w,(224,224),InterpolationMode.BILINEAR,antialias=True)
        if torch.rand(())<.5:out=TF.hflip(out)
    return out

def augmented_pil(im,source_id,epoch,seed,aug):
    assert aug in {'weak','ra'},aug
    out=weak_pil(im,source_id,epoch,seed)
    if aug=='ra':
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(keyseed('RA-v1',seed,epoch,source_id))
            out=T.RandAugment(num_ops=2,magnitude=9,num_magnitude_bins=31,
                interpolation=InterpolationMode.BILINEAR,fill=[124,116,104])(out)
    return out

def tensor(im):return TF.normalize(TF.to_tensor(im),MEAN,STD)
def eval_tensor(im):return tensor(TF.center_crop(TF.resize(im,256,InterpolationMode.BILINEAR,antialias=True),224))

class QVLoRA(nn.Module):
    def __init__(self,width,rank,seed):
        super().__init__()
        g=torch.Generator().manual_seed(seed)
        self.aq=nn.Parameter(torch.randn(rank,width,generator=g)*.01)
        self.bq=nn.Parameter(torch.zeros(width,rank))
        self.av=nn.Parameter(torch.randn(rank,width,generator=g)*.01)
        self.bv=nn.Parameter(torch.zeros(width,rank))
    def forward(self,w):
        q=self.bq@self.aq;v=self.bv@self.av
        return w+torch.cat((q,torch.zeros_like(q),v),dim=0)

def build_model(layout,seed):
    assert layout in {'front','back'},layout
    torch.manual_seed(keyseed('model',seed))
    m=vit_b_16(weights=None)
    state=torch.load(ROOT/'data/vit_b_16-c867db91.pth',map_location='cpu',weights_only=True)
    m.load_state_dict(state,strict=True);del state
    torch.manual_seed(keyseed('head',seed))
    m.heads.head=nn.Linear(768,37)
    for p in m.parameters():p.requires_grad_(False)
    blocks=[0,1,2] if layout=='front' else [9,10,11]
    for slot,i in enumerate(blocks):
        # Same random A initialization by slot, not placement, B starts zero.
        p=QVLoRA(768,4,keyseed('adapter-slot',seed,slot))
        parametrize.register_parametrization(m.encoder.layers[i].self_attention,'in_proj_weight',p)
    for p in m.heads.parameters():p.requires_grad_(True)
    count=sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert count==65317,count
    return m

def optimizer(model):
    adapters=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('heads.')]
    heads=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('heads.')]
    return torch.optim.AdamW([{'params':adapters,'lr':.0003,'base_lr':.0003},
        {'params':heads,'lr':.001,'base_lr':.001}],weight_decay=.01)

def set_lr(opt,step,total):
    warm=max(1,int(.1*total))
    factor=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/(total-warm)))
    for group in opt.param_groups:group['lr']=group['base_lr']*factor

def setup_mps():
    try:os.nice(10)
    except OSError:pass
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available()
    torch.mps.set_per_process_memory_fraction(.1)

def pace(start):
    torch.mps.synchronize();active=time.perf_counter()-start
    time.sleep(2*active)
    return active

def validated_rows(manifest):
    rows=manifest['rows']
    assert rows and len({r['source_id'] for r in rows})==len(rows)
    candidates={r['source_id']:r for r in json.loads(PLAN_PATH.read_text())['rows']}
    old_root=PLAN_PATH.parent.parent/'causal-augmentation-research-20261020'
    old_ids={p.stem for p in old_root.rglob('*.jpg')}
    assert len(old_ids)==177
    for r in rows:
        assert r['candidate_partition'] in {'train','development'}
        source=candidates[r['source_id']]
        assert source['candidate_partition']==r['candidate_partition']
        assert source['class_id']==r['class_id'] and source['species']==r['species']
        assert r['source_id'] not in old_ids
        assert isinstance(r['class_id'],int) and 0<=r['class_id']<37
        assert isinstance(r['duplicate_cluster_id'],str) and r['duplicate_cluster_id']
        assert len(r['image_sha256'])==64
    for partition in ('train','development'):
        assert {r['class_id'] for r in rows if r['candidate_partition']==partition}==set(range(37))
    train_clusters={r['duplicate_cluster_id'] for r in rows if r['candidate_partition']=='train'}
    dev_clusters={r['duplicate_cluster_id'] for r in rows if r['candidate_partition']=='development'}
    assert not (train_clusters & dev_clusters)
    assert len({r['duplicate_cluster_id'] for r in rows})==len(rows),'Known duplicate clusters must all be removed.'
    return rows
