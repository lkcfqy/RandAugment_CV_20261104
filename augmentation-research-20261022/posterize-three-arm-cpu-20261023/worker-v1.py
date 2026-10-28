import time
PROCESS_START=time.perf_counter()
import os,sys,json,hashlib,resource,traceback,importlib.util,signal,math,gc,io
from pathlib import Path
from collections import Counter
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];PILOT=ROOT/'augmentation-research-20261022/pilot-20261022';ATTEMPT=OUT/sys.argv[1]
CFG=json.loads((OUT/'protocol.json').read_text());BUDGET=json.loads((OUT/'budget.json').read_text())
LOG=(ATTEMPT/'events.jsonl').open('x',buffering=1)
COUNTS={'training_forward_images':0,'dev_forward_images':0,'training_forward_calls':0,'dev_forward_calls':0,'backward_calls':0,'optimizer_updates':0,'train_image_reads':0,'dev_image_reads':0,'parity_image_reads':0}
PROGRESS={'phase':'initialization','arm':None,'epoch':None,'logical_step':None,'micro_completed':0,'inflight':None}
MODEL=None;OPT=None;ARM_OUT=None;stop_signal=False;compute_wall=0.;compute_cpu=0.;RECENT=[];RESULTS={};CURRENT_UPDATES=0
TRACE_SHA=None;INITIAL_SHA=None;FROZEN_SHA=None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
def event(kind,**kw):
 rec=dict(event=kind,stage_elapsed=time.monotonic()-BUDGET['started_monotonic'],worker_elapsed=time.perf_counter()-PROCESS_START,counts=dict(COUNTS),progress=dict(PROGRESS),peak_RSS_bytes=rss(),**kw)
 LOG.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
 return rec
class StopRun(Exception):pass
def request_stop(signum,frame):
 global stop_signal
 stop_signal=True
signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
def guard():
 if stop_signal or (OUT/'STOP.request').exists():raise StopRun('explicit_or_supervisor_stop')
 if time.monotonic()>=BUDGET['deadline_monotonic']-15:raise StopRun('absolute_budget_save_margin')
 if rss()>CFG['RSS_stop_bytes']:raise StopRun('observed_RSS_threshold')
 if compute_wall>=2.5 and compute_wall>2*compute_cpu:raise StopRun('own_compute_contention_cumulative')
 if len(RECENT)>=8 and sum(a for a,b in RECENT)>2*sum(b for a,b in RECENT):raise StopRun('own_compute_contention_recent_8_microbatches')
def measured(w,c):
 global compute_wall,compute_cpu
 compute_wall+=w;compute_cpu+=c;RECENT.append((w,c))
 if len(RECENT)>8:RECENT.pop(0)
def param_digest(model,trainable):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if bool(p.requires_grad)!=trainable:continue
  h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().numpy().tobytes())
 return h.hexdigest()
def checkpoint(tag):
 if MODEL is None:return None
 import torch
 target=(ARM_OUT if ARM_OUT is not None else ATTEMPT)/('checkpoint-'+tag+'.pt')
 assert not target.exists()
 state={'trainable_state':{n:p.detach().clone() for n,p in MODEL.named_parameters() if p.requires_grad},'optimizer_state':None if OPT is None else OPT.state_dict(),'partial_gradients':{n:None if p.grad is None else p.grad.detach().clone() for n,p in MODEL.named_parameters() if p.requires_grad},'progress':dict(PROGRESS),'counts':dict(COUNTS),'arm_updates':CURRENT_UPDATES,'torch_CPU_rng_state':torch.get_rng_state(),'seed':CFG['seed'],'trace_sha256':TRACE_SHA,'initial_parameter_sha256':INITIAL_SHA,'frozen_parameter_sha256':FROZEN_SHA,'protocol_sha256':sha(OUT/'protocol.json'),'budget_sha256':sha(OUT/'budget.json'),'resume_automatically':False,'stage_elapsed':time.monotonic()-BUDGET['started_monotonic']}
 with target.open('xb') as f:torch.save(state,f)
 return {'path':str(target.relative_to(ROOT)),'sha256':sha(target)}
def main():
 global MODEL,OPT,ARM_OUT,CURRENT_UPDATES,TRACE_SHA,INITIAL_SHA,FROZEN_SHA
 assert sys.platform=='darwin'
 priority=os.getpriority(os.PRIO_PROCESS,0)
 if priority<10:os.nice(10-priority)
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol-lock.json').read_text())['sha256']
 for name,h in json.loads((OUT/'code-lock.json').read_text())['sha256'].items():assert sha(OUT/name)==h
 for rel,h in CFG['input_sha256'].items():assert sha(ROOT/rel)==h,rel
 assert sha(OUT/'SCREEN_SPEC_FROZEN_ZH.md')==CFG['frozen_spec_sha256']
 assert sha(ROOT/CFG['previous_ledger'])==CFG['previous_ledger_sha256']
 assert sha(PILOT/'data/vit_b_16-c867db91.pth')==CFG['base_checkpoint_sha256']
 event('source_locks_verified');guard()
 import torch
 import numpy as np
 from torch import nn
 from PIL import Image
 from torchvision import transforms as T
 from torchvision.transforms import functional as TF,InterpolationMode
 from torchvision.transforms import autoaugment as AA
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert torch.get_num_threads()==torch.get_num_interop_threads()==1
 spec=importlib.util.spec_from_file_location('posterize_frozen_common',PILOT/'common.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
 rows=json.loads((PILOT/'fit-manifest.json').read_text())['rows'];assert len(rows)==1180 and all(r['candidate_partition'] in ('train','development') for r in rows)
 assert len({r['source_id'] for r in rows})==1180
 train=[r for r in rows if r['candidate_partition']=='train'];dev=[r for r in rows if r['candidate_partition']=='development'];assert len(train)==591 and len(dev)==589
 assert {r['class_id'] for r in train}=={r['class_id'] for r in dev}==set(range(37))
 assert not ({r['duplicate_cluster_id'] for r in train}&{r['duplicate_cluster_id'] for r in dev})
 mapping={r['source_id']:r for r in rows};train_ids={r['source_id'] for r in train};decoded_verified=set()
 def load_image(r,kind):
  p=PILOT/'data/images'/(r['source_id']+'.jpg');assert p.resolve().is_relative_to(PILOT/'data/images')
  b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==r['image_sha256'],r['source_id']
  with Image.open(io.BytesIO(b)) as f:im=f.convert('RGB');im.load()
  assert im.size==(r['width'],r['height'])
  if r['source_id'] not in decoded_verified:
   assert hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()==r['decoded_RGB_sha256'];decoded_verified.add(r['source_id'])
  COUNTS[kind+'_image_reads']+=1
  return im
 PROGRESS['phase']='preflight_trajectories';traces=[];lookup={};orders=[];op_counts={a:Counter() for a in CFG['arms']};target_slots=0;target_views=0
 ra=T.RandAugment(num_ops=2,magnitude=9,num_magnitude_bins=31,interpolation=InterpolationMode.BILINEAR,fill=CFG['fill'])
 space=ra._augmentation_space(31,(224,224));names=list(space);assert names==CFG['pool'];survivors=[s for s in names if s!='Posterize']
 def draw_named(name):
  mags,signed=space[name];m=float(mags[9].item()) if mags.ndim>0 else 0.
  if signed and torch.randint(2,(1,)):m*=-1.
  return {'name':name,'magnitude':m}
 for epoch in range(5):
  order=np.random.default_rng(common.keyseed('order-v1',CFG['seed'],epoch)).permutation(len(train)).tolist();ids=[train[i]['source_id'] for i in order];orders.append(ids)
  for sid in ids:
   guard();r=mapping[sid]
   dummy=Image.new('L',(r['width'],r['height']))
   with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(common.keyseed('weak-v1',CFG['seed'],epoch,sid));i,j,h,w=T.RandomResizedCrop.get_params(dummy,scale=(.8,1.),ratio=(.9,1.1));flip=bool(torch.rand(())<.5)
   del dummy
   with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(common.keyseed('RA-v1',CFG['seed'],epoch,sid));b=[draw_named(names[int(torch.randint(len(names),(1,)).item())]) for slot in range(2)]
   z=[];rr=[];hits=0
   for slot,op in enumerate(b):
    if op['name']=='Posterize':
     hits+=1;z.append({'name':'Identity','magnitude':0.})
     with torch.random.fork_rng(devices=[]):
      torch.random.default_generator.manual_seed(common.keyseed('nm-removal-v1',CFG['seed'],epoch,sid,slot,'replacement'));rr.append(draw_named(survivors[int(torch.randint(13,(1,)).item())]))
    else:z.append(dict(op));rr.append(dict(op))
   item={'epoch':epoch,'source_id':sid,'weak':{'crop':[i,j,h,w],'flip':flip},'B':b,'Z':z,'R':rr};assert len(b)==len(z)==len(rr)==2
   for k in range(2):
    if b[k]['name']!='Posterize':assert z[k]==rr[k]==b[k]
    else:assert z[k]=={'name':'Identity','magnitude':0.} and rr[k]['name'] in survivors
   for a in CFG['arms']:op_counts[a].update(o['name'] for o in item[a])
   target_slots+=hits;target_views+=bool(hits);traces.append(item);lookup[(epoch,sid)]=item
 assert len(traces)==2955 and target_slots==op_counts['B']['Posterize'] and op_counts['Z']['Posterize']==op_counts['R']['Posterize']==0
 dump(OUT/'orders.json',orders)
 with (OUT/'trajectories.jsonl').open('x') as f:
  for item in traces:f.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
 TRACE_SHA=sha(OUT/'trajectories.jsonl')
 def render(im,item,arm):
  i,j,h,w=item['weak']['crop'];im=TF.resized_crop(im,i,j,h,w,(224,224),InterpolationMode.BILINEAR,antialias=True)
  if item['weak']['flip']:im=TF.hflip(im)
  for op in item[arm]:im=AA._apply_op(im,op['name'],op['magnitude'],interpolation=InterpolationMode.BILINEAR,fill=CFG['fill'])
  return im
 selected=json.loads((ROOT/CFG['prior_selection']).read_text());assert len(selected)==16 and all(r['source_id'] in train_ids for r in selected)
 parity=[]
 for epoch in (0,4):
  for r in selected:
   guard();sid=r['source_id'];r=mapping[sid];im=load_image(r,'parity');rep=render(im,lookup[(epoch,sid)],'B');original=common.augmented_pil(im,sid,epoch,CFG['seed'],'ra');assert rep.size==original.size and rep.mode==original.mode and rep.tobytes()==original.tobytes(),(sid,epoch)
   parity.append({'epoch':epoch,'source_id':sid,'exact_RGB_equal':True,'RGB_sha256':hashlib.sha256(rep.tobytes()).hexdigest()})
   del im,rep,original
 dump(OUT/'preflight.json',{'status':'passed','trace_sha256':TRACE_SHA,'orders_sha256':sha(OUT/'orders.json'),'trace_rows':2955,'total_slot_draws':5910,'actual_target_slots':target_slots,'actual_target_views':target_views,'op_counts':{a:dict(c) for a,c in op_counts.items()},'B_pixel_parity':parity,'all_structural_checks_passed':True,'train_count':591,'dev_count':589,'model_forwards':0,'stage_elapsed':time.monotonic()-BUDGET['started_monotonic']})
 event('preflight_passed',target_slots=target_slots,target_views=target_views,parity_views=32,trace_sha256=TRACE_SHA);guard()
 for arm in CFG['arms']:
  ARM_OUT=OUT/arm;ARM_OUT.mkdir();CURRENT_UPDATES=0;PROGRESS.update(phase='model_build',arm=arm,epoch=0,logical_step=0,micro_completed=0,inflight=None)
  MODEL=common.build_model('front',CFG['seed']).to('cpu').train();OPT=common.optimizer(MODEL)
  assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in MODEL.parameters())
  assert sum(p.numel() for p in MODEL.parameters() if p.requires_grad)==65317 and not OPT.state
  assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in MODEL.modules())
  assert all(m.p==0 for m in MODEL.modules() if isinstance(m,nn.Dropout)) and all(m.dropout==0 for m in MODEL.modules() if isinstance(m,nn.MultiheadAttention))
  initial=param_digest(MODEL,True);frozen=param_digest(MODEL,False)
  if INITIAL_SHA is None:INITIAL_SHA=initial;FROZEN_SHA=frozen
  assert initial==INITIAL_SHA and frozen==FROZEN_SHA
  dump(ARM_OUT/'initialization.json',{'initial_trainable_sha256':initial,'frozen_sha256':frozen,'trainable':65317,'empty_optimizer':True,'protocol_sha256':sha(OUT/'protocol.json'),'trace_sha256':TRACE_SHA})
  event('arm_initialized',initial_trainable_sha256=initial);guard();arm_start=time.monotonic()
  step_log=(ARM_OUT/'steps.jsonl').open('x',buffering=1)
  try:
   for epoch,ids in enumerate(orders):
    epoch_ce=0.;seen=0;trajectory_digest=hashlib.sha256()
    for batch_start in range(0,591,16):
     guard();PROGRESS.update(phase='training',epoch=epoch+1,logical_step=CURRENT_UPDATES+1,micro_completed=0,inflight=None)
     tick=time.perf_counter();chosen=ids[batch_start:batch_start+16];bs=len(chosen);assert bs==16 or(batch_start==576 and bs==15)
     xx=[]
     for sid in chosen:
      im=load_image(mapping[sid],'train');pic=render(im,lookup[(epoch,sid)],arm);trajectory_digest.update(sid.encode());trajectory_digest.update(pic.tobytes());xx.append(common.tensor(pic));del im,pic
     x=torch.stack(xx);del xx;y=torch.tensor([mapping[sid]['class_id'] for sid in chosen],dtype=torch.long)
     common.set_lr(OPT,CURRENT_UPDATES,1110);OPT.zero_grad(set_to_none=True);loss_value=0.
     for micro in range(0,bs,4):
      guard();n=min(4,bs-micro);PROGRESS['inflight']='forward';event('micro_start',batch_start=batch_start,micro_start=micro,images=n)
      w0=time.perf_counter();c0=time.process_time();z=MODEL(x[micro:micro+n]);loss=nn.functional.cross_entropy(z,y[micro:micro+n],reduction='sum')/bs
      COUNTS['training_forward_calls']+=1;COUNTS['training_forward_images']+=n;assert bool(torch.isfinite(loss));loss_value+=float(loss.detach())
      PROGRESS['inflight']='backward';event('backward_start');loss.backward();COUNTS['backward_calls']+=1;measured(time.perf_counter()-w0,time.process_time()-c0)
      PROGRESS['micro_completed']=micro+n;PROGRESS['inflight']=None;del loss,z;event('micro_complete')
     grads=[p.grad for p in MODEL.parameters() if p.requires_grad];assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads);assert all(p.grad is None for p in MODEL.parameters() if not p.requires_grad)
     PROGRESS['inflight']='optimizer';OPT.step();CURRENT_UPDATES+=1;COUNTS['optimizer_updates']+=1;PROGRESS['inflight']=None
     assert COUNTS['optimizer_updates']<=555 and COUNTS['training_forward_images']<=8865 and COUNTS['backward_calls']<=2220
     row={'arm':arm,'epoch':epoch+1,'step':CURRENT_UPDATES,'examples':bs,'mean_CE':loss_value,'adapter_lr':OPT.param_groups[0]['lr'],'head_lr':OPT.param_groups[1]['lr'],'inclusive_seconds':time.perf_counter()-tick,'stage_elapsed':time.monotonic()-BUDGET['started_monotonic'],'peak_RSS_bytes':rss()};step_log.write(json.dumps(row)+'\n');event('update_complete',step_seconds=row['inclusive_seconds'],mean_CE=loss_value)
     epoch_ce+=loss_value*bs;seen+=bs;del x,y,grads
     guard()
    assert seen==591 and CURRENT_UPDATES==(epoch+1)*37
    cp=checkpoint('epoch%02d'%(epoch+1));dump(ARM_OUT/('epoch%02d.json'%(epoch+1)),{'epoch':epoch+1,'updates':CURRENT_UPDATES,'source_presentations':seen,'training_mean_CE':epoch_ce/seen,'ordered_rendered_RGB_sha256':trajectory_digest.hexdigest(),'order_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'checkpoint':cp,'stage_elapsed':time.monotonic()-BUDGET['started_monotonic']});event('epoch_complete',training_mean_CE=epoch_ce/seen,checkpoint=cp)
  finally:step_log.close()
  assert CURRENT_UPDATES==185;assert param_digest(MODEL,False)==FROZEN_SHA
  PROGRESS.update(phase='development_evaluation',epoch=5,logical_step=185,micro_completed=0,inflight=None);MODEL.eval();records=[]
  with (ARM_OUT/'development-predictions.jsonl').open('x',buffering=1) as fout:
   with torch.inference_mode():
    for start in range(0,589,4):
     guard();batch=dev[start:start+4];xs=[]
     for r in batch:
      im=load_image(r,'dev');xs.append(common.eval_tensor(im));del im
     x=torch.stack(xs);del xs;PROGRESS['inflight']='dev_forward';event('dev_forward_start',dev_start=start)
     w0=time.perf_counter();c0=time.process_time();z=MODEL(x);COUNTS['dev_forward_calls']+=1;COUNTS['dev_forward_images']+=len(batch);measured(time.perf_counter()-w0,time.process_time()-c0);assert bool(torch.isfinite(z).all())
     lp=torch.log_softmax(z.to(torch.float64),dim=1);pred=z.argmax(1).tolist()
     for i,r in enumerate(batch):
      row={'source_id':r['source_id'],'class_id':r['class_id'],'prediction':pred[i],'correct':pred[i]==r['class_id'],'NLL':-float(lp[i,r['class_id']]),'logits':z[i].tolist()};fout.write(json.dumps(row)+'\n');records.append(row)
     PROGRESS['inflight']=None;PROGRESS['micro_completed']=start+len(batch);event('dev_batch_complete');del z,x,lp
  assert len(records)==589;correct=sum(r['correct'] for r in records)
  summary={'arm':arm,'seed':CFG['seed'],'epochs':5,'updates':185,'training_presentations':2955,'dev_count':589,'dev_correct':correct,'dev_accuracy':correct/589,'dev_NLL':sum(r['NLL'] for r in records)/589,'arm_wall_seconds':time.monotonic()-arm_start,'initial_trainable_sha256':INITIAL_SHA,'frozen_before_sha256':FROZEN_SHA,'frozen_after_sha256':param_digest(MODEL,False),'final_trainable_sha256':param_digest(MODEL,True),'trace_sha256':TRACE_SHA,'prediction_sha256':sha(ARM_OUT/'development-predictions.jsonl'),'peak_RSS_bytes':rss()};assert summary['frozen_after_sha256']==FROZEN_SHA;dump(ARM_OUT/'summary.json',summary);RESULTS[arm]=summary;event('arm_complete',correct=correct,accuracy=correct/589)
  if arm=='B' and correct<531:
   dump(OUT/'result.json',{'status':'baseline_readiness_unmet','classification':'unresolved','baseline_min_correct':531,'summaries':RESULTS,'counts':COUNTS,'stage_elapsed':time.monotonic()-BUDGET['started_monotonic'],'worker_cpu_seconds':time.process_time(),'unrun_arms':['Z','R'],'reason':'Predeclared B readiness gate; no seed/epoch repair'});return
  MODEL=None;OPT=None;ARM_OUT=None;gc.collect();guard()
 assert set(RESULTS)=={'B','Z','R'} and COUNTS['optimizer_updates']==555 and COUNTS['training_forward_images']==8865 and COUNTS['dev_forward_images']==1767
 byarm={a:{r['source_id']:r for r in [json.loads(s) for s in (OUT/a/'development-predictions.jsonl').read_text().splitlines()]} for a in CFG['arms']}
 patterns={format(i,'03b'):0 for i in range(8)};joint=[]
 for r in dev:
  sid=r['source_id'];p=''.join('1' if byarm[a][sid]['correct'] else '0' for a in CFG['arms']);patterns[p]+=1;joint.append({'source_id':sid,'class_id':r['class_id'],'B':byarm['B'][sid]['correct'],'Z':byarm['Z'][sid]['correct'],'R':byarm['R'][sid]['correct'],'pattern_BZR':p})
 dz=RESULTS['Z']['dev_correct']-RESULTS['B']['dev_correct'];dr=RESULTS['R']['dev_correct']-RESULTS['B']['dev_correct'];hr=dr-dz
 classification='positive_observed_reversal' if (dz>=3 and dr<=-3) or(dz<=-3 and dr>=3) else ('negative_same_direction_screen' if(dz>=3 and dr>=3) or(dz<=-3 and dr<=-3) else 'unresolved_below_direction_threshold')
 dump(OUT/'joint-correctness.json',joint)
 PROGRESS.update(phase='complete',arm=None,inflight=None)
 final={'status':'all_three_arms_completed','classification':classification,'summaries':RESULTS,'baseline_readiness_passed':RESULTS['B']['dev_correct']>=531,'contrast_gate_count':3,'gZ_count':dz,'gR_count':dr,'h_count':hr,'gZ':dz/589,'gR':dr/589,'h':hr/589,'patterns_BZR':patterns,'counts':COUNTS,'stage_elapsed':time.monotonic()-BUDGET['started_monotonic'],'worker_wall_seconds':time.perf_counter()-PROCESS_START,'worker_cpu_seconds':time.process_time(),'compute_wall_seconds':compute_wall,'compute_cpu_seconds':compute_cpu,'peak_RSS_bytes':rss(),'training_seed_count':1,'exploratory_development_only':True,'original_30_epoch_claim_tested':False,'new_algorithm_or_novelty_claimed':False,'GPU_calls':0,'sealed_read':False}
 dump(OUT/'result.json',final);event('complete',classification=classification);print(json.dumps({'classification':classification,'correct':{a:RESULTS[a]['dev_correct'] for a in CFG['arms']},'stage_elapsed':final['stage_elapsed']}),flush=True)
try:main()
except BaseException as ex:
 try:cp=checkpoint('stopped')
 except BaseException as cp_error:cp={'save_error':repr(cp_error)}
 failure={'status':'stopped_unresolved','error':repr(ex),'type':type(ex).__name__,'traceback':traceback.format_exc(),'counts':COUNTS,'progress':PROGRESS,'completed_arm_summaries':RESULTS,'checkpoint':cp,'stage_elapsed':time.monotonic()-BUDGET['started_monotonic'],'worker_wall_seconds':time.perf_counter()-PROCESS_START,'worker_cpu_seconds':time.process_time(),'compute_wall_seconds':compute_wall,'compute_cpu_seconds':compute_cpu,'peak_RSS_bytes':rss(),'automatic_retry':False}
 dump(ATTEMPT/'failure.json',failure);event('stopped',reason=repr(ex),checkpoint=cp);raise
finally:LOG.close()
