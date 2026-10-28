import time
START=time.perf_counter()
import os,sys,json,hashlib,resource,traceback,importlib.util,math
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];PILOT=ROOT/'augmentation-research-20261022/pilot-20261022'
LOG=(OUT/'events.jsonl').open('x',buffering=1)
counts={'training_forward_images':0,'inference_forward_images':0,'training_forward_calls':0,'inference_forward_calls':0,'backward_calls':0,'optimizer_updates':0,'decoded_image_reads':0}
compute_wall=0.;compute_cpu=0.;steps=[];infer=[];progress={'phase':'startup','inflight':None}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
def event(kind,**kw):
 LOG.write(json.dumps(dict(event=kind,wall=time.perf_counter()-START,counts=dict(counts),progress=dict(progress),peak_RSS_bytes=rss(),**kw),ensure_ascii=False,allow_nan=False)+'\n')
def guard():
 if (OUT/'STOP.request').exists():raise RuntimeError('explicit_stop_marker')
 if time.perf_counter()-START>=48:raise RuntimeError('soft_wall_stop')
 if rss()>3221225472:raise RuntimeError('own_peak_RSS_stop')
 if compute_wall>=2.5 and compute_wall>2*compute_cpu:raise RuntimeError('own_compute_contention_stop')
def main():
 global compute_wall,compute_cpu
 assert sys.platform=='darwin'
 n=os.getpriority(os.PRIO_PROCESS,0)
 if n<10:os.nice(10-n)
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 cfg=json.loads((OUT/'protocol.json').read_text())
 assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol-lock.json').read_text())['sha256']
 for name,h in json.loads((OUT/'code-lock.json').read_text())['sha256'].items():assert sha(OUT/name)==h
 assert sha(OUT/'selection.json')==cfg['selection_sha256']
 for rel,h in cfg['input_sha256'].items():assert sha(ROOT/rel)==h
 assert sha(PILOT/'data/vit_b_16-c867db91.pth')==cfg['weight_sha256']
 rows=json.loads((OUT/'selection.json').read_text());assert len(rows)==16 and all(r['candidate_partition']=='train' for r in rows)
 event('inputs_verified',source_ids=[r['source_id'] for r in rows]);guard()
 import torch
 from torch import nn
 from PIL import Image
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert torch.get_num_threads()==torch.get_num_interop_threads()==1
 spec=importlib.util.spec_from_file_location('throughput_common',PILOT/'common.py')
 common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
 event('imports_complete',torch_version=torch.__version__);guard()
 t=time.perf_counter();model=common.build_model('front',20261024).to('cpu').train();opt=common.optimizer(model)
 assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters())
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==65317
 assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules())
 assert all(m.p==0 for m in model.modules() if isinstance(m,nn.Dropout))
 assert all(m.dropout==0 for m in model.modules() if isinstance(m,nn.MultiheadAttention))
 before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
 event('model_ready',model_build_seconds=time.perf_counter()-t,trainable=65317);guard()
 paths=[]
 for r in rows:
  p=PILOT/'data/images'/(r['source_id']+'.jpg');assert p.resolve().is_relative_to(PILOT/'data/images')
  assert sha(p)==r['image_sha256'];paths.append(p)
 def tensors(index,clean=False):
  result=[]
  for r,p in zip(rows,paths):
   with Image.open(p) as f:im=f.convert('RGB');im.load()
   counts['decoded_image_reads']+=1
   assert im.size==(r['width'],r['height'])
   if index==0 and not clean:assert hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()==r['decoded_RGB_sha256']
   imtensor=common.eval_tensor(im) if clean else common.tensor(common.augmented_pil(im,r['source_id'],index,20261024,'ra'))
   result.append(imtensor)
  return torch.stack(result),torch.tensor([r['class_id'] for r in rows],dtype=torch.long)
 stop='max_updates'
 for step in range(5):
  guard()
  if steps and time.perf_counter()-START+max(s['inclusive_seconds'] for s in steps)*1.6+3>=48:
   stop='predicted_next_update_would_exceed_soft_cap';break
  progress.update(phase='training',inflight=None,logical_step=step+1)
  step_start=time.perf_counter();prep_start=time.perf_counter();x,y=tensors(step);prep=time.perf_counter()-prep_start
  common.set_lr(opt,step,1110);opt.zero_grad(set_to_none=True)
  fw=0.;bw=0.;ce=0.;cpu_start=time.process_time();compute_start=time.perf_counter()
  for start in range(0,16,4):
   guard();progress.update(inflight='training_forward',micro_start=start);event('forward_start')
   t=time.perf_counter();z=model(x[start:start+4]);loss=nn.functional.cross_entropy(z,y[start:start+4],reduction='sum')/16
   fw+=time.perf_counter()-t;counts['training_forward_images']+=4;counts['training_forward_calls']+=1
   assert bool(torch.isfinite(loss));ce+=float(loss.detach())
   progress['inflight']='backward';event('backward_start')
   t=time.perf_counter();loss.backward();bw+=time.perf_counter()-t;counts['backward_calls']+=1
   progress['inflight']=None;del z,loss;event('micro_complete')
  grads=[p.grad for p in model.parameters() if p.requires_grad]
  assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
  assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
  progress['inflight']='optimizer';t=time.perf_counter();opt.step();opt_seconds=time.perf_counter()-t
  counts['optimizer_updates']+=1;progress['inflight']=None
  cpu=time.process_time()-cpu_start;active=time.perf_counter()-compute_start;compute_cpu+=cpu;compute_wall+=active
  row={'step':step+1,'warmup':step==0,'images':16,'decode_and_augment_seconds':prep,'forward_seconds':fw,'backward_seconds':bw,'optimizer_seconds':opt_seconds,'compute_wall_seconds':active,'compute_cpu_seconds':cpu,'inclusive_seconds':time.perf_counter()-step_start,'mean_CE_for_finite_check_only':ce,'peak_RSS_bytes':rss()}
  steps.append(row);event('update_complete',timing=row);del x,y;guard()
 if steps and time.perf_counter()-START+max(s['forward_seconds']+s['decode_and_augment_seconds'] for s in steps)*1.5+2<48:
  progress.update(phase='training_set_inference',inflight=None);model.eval();t=time.perf_counter();x,y=tensors(0,True);prep=time.perf_counter()-t;del y
  with torch.inference_mode():
   for start in range(0,16,4):
    guard();progress.update(inflight='inference_forward',micro_start=start);event('inference_start')
    t=time.perf_counter();cpu=time.process_time();z=model(x[start:start+4]);elapsed=time.perf_counter()-t;cput=time.process_time()-cpu
    assert bool(torch.isfinite(z).all());counts['inference_forward_images']+=4;counts['inference_forward_calls']+=1
    compute_wall+=elapsed;compute_cpu+=cput;progress['inflight']=None
    infer.append({'images':4,'forward_seconds':elapsed,'cpu_seconds':cput});event('inference_complete');del z
  inference={'clean_decode_prepare_seconds':prep,'batches':infer};del x
 else:inference={'not_measured':'remaining_time_gate'}
 delta=max(float((p.detach()-before[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad)
 assert delta>0 and len(steps)>=2
 progress.update(phase='complete',inflight=None)
 result={'status':'completed_throughput_only','stop_reason':stop,'worker_wall_seconds':time.perf_counter()-START,'worker_cpu_seconds':time.process_time(),'counts':counts,'steps':steps,'training_set_inference':inference,'peak_RSS_bytes':rss(),'trainable_max_abs_update':delta,'checkpoint_saved':False,'scientific_accuracy_compared':False,'device':'cpu','intraop_threads':1,'interop_threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0),'model_forwards_are_image_presentations':True,'GPU_calls':0,'development_images_read':0,'sealed_read':False}
 dump('result.json',result);event('complete');print(json.dumps({'status':result['status'],'updates':counts['optimizer_updates'],'forward_images':counts['training_forward_images']+counts['inference_forward_images'],'wall':result['worker_wall_seconds'],'peak_RSS_bytes':rss()}),flush=True)
try:main()
except BaseException as ex:
 dump('failure.json',{'status':'incomplete_not_scientific_negative','error':repr(ex),'traceback':traceback.format_exc(),'worker_wall_seconds':time.perf_counter()-START,'worker_cpu_seconds':time.process_time(),'counts':counts,'steps':steps,'inference_batches':infer,'progress':progress,'peak_RSS_bytes':rss()})
 event('stopped',error=repr(ex));raise
finally:LOG.close()
