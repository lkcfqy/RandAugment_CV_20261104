"""Bounded synthetic-input compatibility/timing probe, not a research experiment."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
import gc
import json
from pathlib import Path
import resource
import time
from datetime import datetime, timezone
import torch
from torch import nn
from torch.nn.utils import parametrize
from torchvision.models import vit_b_16

class QVLoRA(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.aq=nn.Parameter(torch.randn(rank,width)*0.01)
        self.bq=nn.Parameter(torch.zeros(width,rank))
        self.av=nn.Parameter(torch.randn(rank,width)*0.01)
        self.bv=nn.Parameter(torch.zeros(width,rank))
    def forward(self, original):
        q=self.bq@self.aq
        v=self.bv@self.av
        return original+torch.cat((q,torch.zeros_like(q),v),dim=0)

def attach(model, mode):
    if mode=='full':
        return
    for p in model.parameters():p.requires_grad_(False)
    if mode=='front':layers,rank=range(3),4
    elif mode=='back':layers,rank=range(9,12),4
    else:layers,rank=range(12),1
    for i in layers:
        att=model.encoder.layers[i].self_attention
        parametrize.register_parametrization(att,'in_proj_weight',QVLoRA(768,rank))
    for p in model.heads.parameters():p.requires_grad_(True)

def main():
    try:os.nice(10)
    except OSError:pass
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt={'at_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Synthetic-input compatibility and timing only; no dataset, pretrained weight, scientific result or CUDA.',
        'torch':torch.__version__,'MPS_built':torch.backends.mps.is_built(),
        'MPS_available':torch.backends.mps.is_available(),'CUDA_available':torch.cuda.is_available(),
        'cpu_threads':torch.get_num_threads(),'cpu_count':os.cpu_count(),
        'MPS_fallback_enabled':False,'input_shape':[2,3,224,224],
        'new_dependencies':False,'cases':[]}
    if not receipt['MPS_available']:
        receipt['status']='mps_not_available_probe_stopped'
    else:
        torch.mps.set_per_process_memory_fraction(0.10)
        receipt['per_process_memory_fraction']=0.10
        t0=time.perf_counter()
        for mode in ('front','back','spread','full'):
            torch.manual_seed(7201)
            model=vit_b_16(weights=None,num_classes=37)
            nn.init.normal_(model.heads.head.weight,std=0.01)
            attach(model,mode)
            model=model.to('mps').train()
            opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.001,momentum=0.9)
            x=torch.randn(2,3,224,224,device='mps')
            y=torch.tensor([0,1],device='mps')
            before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad and 'heads' not in n and ('.bq' in n or mode=='full') and ('encoder_layer_0.' in n or 'encoder_layer_9.' in n)}
            times=[]
            loss_values=[]
            for step in range(4):
                torch.mps.synchronize();ts=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(x),y)
                loss.backward();opt.step()
                torch.mps.synchronize()
                times.append(time.perf_counter()-ts)
                loss_values.append(float(loss.detach().cpu()))
            after=dict(model.named_parameters())
            changed={n:float((after[n].detach()-v).abs().max().cpu()) for n,v in before.items()}
            receipt['cases'].append({'mode':mode,'device':str(next(model.parameters()).device),
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'trainable_backbone_parameters':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('heads.')),
                'all_step_seconds':times,'steady_step_seconds_mean':sum(times[1:])/3,
                'loss_values_not_a_scientific_result':loss_values,
                'sampled_backbone_parameter_changes':changed,
                'all_losses_finite':all(torch.isfinite(torch.tensor(v)) for v in loss_values),
                'MPS_current_allocated_bytes':torch.mps.current_allocated_memory(),
                'MPS_driver_allocated_bytes':torch.mps.driver_allocated_memory()})
            del model,opt,x,y,before,after
            gc.collect();torch.mps.empty_cache()
        receipt['wall_seconds_including_builds']=time.perf_counter()-t0
        receipt['peak_RSS_bytes_Mac']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt['status']='bounded_mps_forward_backward_update_completed'
    out=Path(__file__).resolve().parent/'mps-vit-probe.json'
    with out.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
