"""Strict safetensors CPU load of reviewed pinned CIDNet source, zero forwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

START=time.monotonic()
START_CPU=time.process_time()
started=datetime.now(timezone.utc).isoformat()
ROOT=Path(sys.argv[1]).resolve()
SRC=ROOT/'vendor/HVI-CIDNet'
SOURCE_FILES=['net/CIDNet.py','net/HVI_transform.py','net/LCA.py','net/transformer_utils.py']
tree={x['path']:x for x in json.loads((ROOT/'sources/hvi_tree.json').read_text())['tree']}
source_checks=[]
for rel in SOURCE_FILES:
    b=(SRC/rel).read_bytes()
    git_blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    assert git_blob==tree[rel]['sha'], 'Pinned source mismatch: '+rel
    source_checks.append(dict(path=rel,git_blob=git_blob,sha256=hashlib.sha256(b).hexdigest()))
config=json.loads((ROOT/'models/config.json').read_text())
assert config==dict(channels=[36,36,72,144],heads=[1,2,4,8],norm=False)
with (ROOT/'models/model.safetensors').open('rb') as f:
    sha=hashlib.file_digest(f,'sha256').hexdigest()
assert sha=='2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894'

import torch
from safetensors.torch import load_file
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_default_device('cpu')
sys.path.insert(0,str(SRC))
from net.CIDNet import CIDNet

state=load_file(str(ROOT/'models/model.safetensors'),device='cpu')
assert all(t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in state.values())
with torch.device('cpu'):
    model=CIDNet(**config)
defaults={name:getattr(model.trans,name) for name in ['alpha_s','alpha','gated','gated2','this_k']}
load_result=model.load_state_dict(state,strict=True)
assert not load_result.missing_keys and not load_result.unexpected_keys
assert len(model.state_dict())==len(state)
assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
model.trans.alpha_s=1.0
model.trans.alpha=1.0  # eval_hf.py calls this CLI argument alpha_i
model.trans.gated=True
model.trans.gated2=True
model.eval()
model.requires_grad_(False)
assert all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
assert not any(m.training for m in model.modules())
forwards={'count':0}
def forbid_forward(module,args):
    forwards['count']+=1
    raise RuntimeError('This asset verification run prohibits every model forward')
handle=model.register_forward_pre_hook(forbid_forward)

contract={
 'status':'FROZEN_FOR_FUTURE_REVIEWED_PROTOCOL_NOT_EXECUTED',
 'model_config':config,'input_channels':'RGB','input_dtype':'float32',
 'input_range':[0,1],'input_layout':'NCHW','native_width':600,'native_height':400,
 'spatial_divisor':8,'resize':False,'padding_for_native_panel':False,'imagenet_normalization':False,
 'alpha_s':1.0,'alpha_i':1.0,'gamma':1.0,'gated':True,'gated2':True,
 'use_GT_mean':False,'any_gt_mean_correction':False,'gt_available_to_model':False,
 'output_clamp_before_future_metrics':[0.0,1.0],'model_eval':True,'parameters_frozen':True,
 'parameter_paths':{
  'alpha_s':'model.trans.alpha_s', 'alpha_i':'model.trans.alpha',
  'gated':'model.trans.gated','gated2':'model.trans.gated2',
  'gamma':'External input exponent immediately before model(input ** gamma); not a model attribute',
  'use_GT_mean':'External measurement option in measure.py:119 (store_true default False); do not invoke this script',
  'GT_mean_correction':'measure.py:91-94; excluded completely from future evaluation',
  'density_k':'model.trans.density_k loaded from checkpoint; preserve learned value',
  'this_k':'Runtime cache set by HVIT from density_k; not a free hyperparameter'
 },
 'source_default_note':'RGB_HVI defaults alpha_s=1.3, alpha=1.0, gated=False, gated2=False; eval_hf overrides both coefficients to 1.0 and both gates to True',
 'no_execution_of':['eval_hf.py','eval.py','measure.py','repository installation scripts'],
}
(ROOT/'reports/inference_contract.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow','einops','safetensors','huggingface_hub']}
report=dict(status='STRICT_CPU_LOAD_PASSED_ZERO_FORWARD',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,python_executable=sys.executable,package_versions=versions,source_checks=source_checks,source_commit=json.loads((ROOT/'reports/source_pins.json').read_text())['hvi_github_commit'],checkpoint_sha256=sha,config=config,strict=True,missing_keys=load_result.missing_keys,unexpected_keys=load_result.unexpected_keys,prefix_remapping=False,state_tensor_count=len(state),state_numel=sum(v.numel() for v in state.values()),parameter_numel=sum(p.numel() for p in model.parameters()),tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),all_checkpoint_tensors_finite=True,all_loaded_tensors_equal=True,devices=sorted({str(p.device) for p in model.parameters()}),dtypes=sorted({str(p.dtype) for p in model.parameters()}),source_defaults=defaults,effective_transform={name:getattr(model.trans,name) for name in ['alpha_s','alpha','gated','gated2']},density_k=float(model.trans.density_k.detach().item()),eval_mode=True,parameters_frozen=True,safetensors_only=True,torch_load_called=False,unsafe_pickle=False,allowlist_changes=False,network_during_model_loading=False,gt_images_opened_by_loader=0,model_forward_count=forwards['count'],metrics_computed=0,gpu_seconds=0,wall_seconds=time.monotonic()-START,process_cpu_seconds=time.process_time()-START_CPU,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(ROOT/'reports/model_cpu_verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
(ROOT/'reports/state_manifest.json').write_text(json.dumps([dict(key=k,shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size()) for k,v in sorted(state.items())],indent=2)+'\n')
print(json.dumps(report,indent=2))
