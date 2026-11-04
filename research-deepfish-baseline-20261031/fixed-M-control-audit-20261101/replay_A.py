"""Replay only official A data transforms; compare the actual archived per-op trace."""
from portable_dev.paths import project_path as _portable_project_path
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import copy,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import cv2,mmcv
from mmengine.dataset import DefaultSampler
from mmengine.registry import init_default_scope
from mmdet.datasets import CocoDataset
from mmdet.registry import TRANSFORMS

R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
W=Path(__file__).resolve().parent
D=R/'recovery-v5-20261031/payload/data'
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def plain(v):
 if hasattr(v,'tensor'):v=v.tensor
 if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 if isinstance(v,np.generic):return v.item()
 return v
def rng():
 s=np.random.get_state();n=[s[0],s[1].tolist(),s[2],s[3],s[4]]
 return {'numpy':hashlib.sha256(canonical(n)).hexdigest(),'python':hashlib.sha256(canonical(random.getstate())).hexdigest(),'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
def snapshot(results):return {k:plain(results[k]) for k in ('img_shape','scale','scale_factor','flip','flip_direction','homography_matrix','gt_bboxes','gt_bboxes_labels','gt_ignore_flags') if k in results}
def norm(x):
 if isinstance(x,list):return [norm(y) for y in x]
 if not isinstance(x,dict):return x
 z={k:norm(v) for k,v in x.items()}
 for k in ['classes','scale','crop_size']:
  if isinstance(z.get(k),list):z[k]=tuple(z[k])
 if 'scales' in z:z['scales']=[tuple(v) for v in z['scales']]
 return z

torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(0);mmcv.use_backend('cv2');init_default_scope('mmdet')
cfg=json.loads((R/'recovery-v5-20261031/payload/config_A.json').read_text())
dataset=CocoDataset(ann_file='annotations/support5.json',data_root=str(D)+'/',data_prefix={'img':'images/'},metainfo={'classes':('fish',)},filter_cfg={'filter_empty_gt':False,'min_size':0},return_classes=True,pipeline=[])
sampler=DefaultSampler(dataset,shuffle=True,seed=20261031,round_up=False)
history=json.loads((R/'recovery-v5-20261031/gpu-run/artifacts/formal_A_trace.json').read_text())
steps=[];colors=[]
def wrap(op,label):
 if op.__class__.__name__=='RandomChoice':
  for b,branch in enumerate(op.transforms):branch.transforms=[wrap(x,label+'.branch'+str(b)+'.'+str(i)) for i,x in enumerate(branch.transforms)]
 def call(results):
  before=rng();before_data=snapshot(results)
  is_color=op.__class__.__name__=='Color'
  if is_color:
   image=results['img'].copy();state=np.random.get_state();local=np.random.RandomState(0);local.set_state(state);local.uniform(0,1);expected=local.get_state()
  result=op(results);assert result is not None
  if is_color:
   actual=np.random.get_state();assert np.array_equal(image,result['img']) and image.dtype==np.uint8
   assert actual[0]==expected[0] and np.array_equal(actual[1],expected[1]) and actual[2:]==expected[2:]
   colors.append({'identity_exact':True,'exactly_one_numpy_uniform_draw':True})
  steps.append({'op':op.__class__.__name__,'path':label,'before':before_data,'after':snapshot(result),'rng_before':before,'rng_after':rng()})
  return result
 return call
ops=[wrap(TRANSFORMS.build(norm(t)),str(i)) for i,t in enumerate(cfg['train_dataloader']['dataset']['pipeline'][0]['transforms'])]
random.seed(20261031);np.random.seed(20261031);torch.manual_seed(20261031)
records=[];checks=[];i=0;start=time.monotonic()
for epoch in range(40):
 sampler.set_epoch(epoch)
 for index in sampler:
  steps.clear();data=dataset.get_data_info(index);image_id=data['img_id']
  for op in ops:data=op(data)
  sample=data['data_samples'];record={'image_id':image_id,'steps':copy.deepcopy(steps),'packed_gt':plain(sample.gt_instances.bboxes),'packed_labels':plain(sample.gt_instances.labels),'input_shape':list(data['inputs'].shape)}
  old=history[i];assert all(record[k]==old[k] for k in ['image_id','packed_gt','packed_labels','input_shape'])
  assert len(record['steps'])==len(old['steps'])
  for j,(a,b) in enumerate(zip(record['steps'],old['steps'])):
   assert all(a[k]==b[k] for k in ['op','path','before','after'])
   for side in ['rng_before','rng_after']:
    assert a[side]['numpy']==b[side]['numpy'] and a[side]['python']==b[side]['python']
   assert a['rng_before']['torch_cpu']==a['rng_after']['torch_cpu'] and b['rng_before']['torch_cpu']==b['rng_after']['torch_cpu']
   checks.append({'iteration':i+1,'operation_index':j,'geometry_GT_and_numpy_python_absolute_RNG_match':True,'torch_CPU_no_draw_in_both':True})
  records.append(record);i+=1
assert i==len(colors)==200
(W/'A_REPLAY_TRACE.json').write_bytes(canonical(records))
(W/'A_REPLAY_CHECKS.json').write_bytes(canonical(checks))
summary={'status':'PASS','PID':os.getpid(),'input_images':5,'iterations':200,'operation_checks':len(checks),'Color_identity_and_exact_single_numpy_draw_checks':len(colors),'all_geometry_GT_branch_paths_numpy_python_RNG_match':True,'Torch_CPU_rng_consumption_unchanged_per_operator':True,'absolute_Torch_CPU_state_reconstructed':False,'reason':'Historical model initialization and DataLoader iterator seeds are absent from this model-free replay. Only their hashes were archived; no state is invented from a hash.','GPU_CUDA_rng_reconstructed':False,'model_calls':0,'training_updates':0,'wall_seconds':time.monotonic()-start,'source_config_sha256':hashlib.sha256((R/'recovery-v5-20261031/payload/config_A.json').read_bytes()).hexdigest(),'historical_trace_sha256':hashlib.sha256((R/'recovery-v5-20261031/gpu-run/artifacts/formal_A_trace.json').read_bytes()).hexdigest()}
(W/'A_REPLAY_VERIFICATION.json').write_bytes(canonical(summary));print(json.dumps(summary),flush=True)
