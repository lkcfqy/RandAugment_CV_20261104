"""One isolated CPU training phase; original official train_step and frozen A/M pipelines."""
from portable_dev.paths import project_path as _portable_project_path
import argparse,copy,hashlib,importlib.util,json,os,pathlib,random,resource,shutil,sys,time,traceback
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
F=R/'fixed-M-execution-20261101/payload'
sys.path.insert(0,str(F))
from candidate_worker import normalize_config
from candidate_control import validate_optimizer_groups
from contracts import canonical,validate_pretrained_key_schema

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 data=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
 tmp=path.with_name(path.name+'.partial')
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 assert path.read_bytes()==data
def tree(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
def state_hash(state):
 h=hashlib.sha256();size=0
 for name,t in sorted(state.items()):
  b=t.detach().contiguous().numpy().tobytes()
  h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(b);size+=len(b)
 return {'state_sha256':h.hexdigest(),'tensor_count':len(state),'tensor_bytes':size}
last_storage_check=0
def guard(extra=0):
 global last_storage_check
 context=json.loads((P/'RUN_STARTED.json').read_text())
 assert time.monotonic()<context['deadline_monotonic'],'Two-hour total runtime reached'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=12*1024**3,'CPU process peak RSS exceeded12GiB'
 assert shutil.disk_usage(P).free>=80*1024**2+extra,'Filesystem margin reached'
 if extra or time.monotonic()-last_storage_check>30:
  assert tree(R)+tree(P)+extra<=8*1024**3-80*1024**2,'Research plus staging8GiB margin reached'
  last_storage_check=time.monotonic()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','formal']);parser.add_argument('arm',choices=['A','M']);args=parser.parse_args()
 phase=args.phase+'_'+args.arm;out=P/phase;out.mkdir(exist_ok=False)
 protocol=json.loads((P/'PROTOCOL.json').read_text());frozen=json.loads((P/'CODE_PINS.json').read_text())
 for name,pin in frozen.items():assert sha(P/name)==pin,'Execution source changed: '+name
 began=time.monotonic();report={'status':'STARTED','phase':args.phase,'arm':args.arm,'PID':os.getpid(),'seed':20261031,'formal_updates_planned':200 if args.phase=='formal' else 0,'completed_updates':0,'dev_test_access':False,'GPU_MPS_upload_CU_new':0}
 def save(stage):
  report.update(stage=stage,elapsed_seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  atomic(out/'RESULT.json',report);print(json.dumps({k:report[k] for k in ['stage','arm','completed_updates','elapsed_seconds','peak_RSS_bytes']}),flush=True)
 try:
  guard();import numpy as np,cv2,torch,mmcv,mmdet
  from mmdet.registry import MODELS
  from mmdet.utils import register_all_modules
  from mmdet.datasets import CocoDataset
  from mmengine.config import Config
  from mmengine.dataset import DefaultSampler,pseudo_collate
  from mmengine.optim import build_optim_wrapper
  torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(0)
  torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
  assert torch.version.cuda is None and torch.get_num_threads()==4 and cv2.getNumThreads()==1
  register_all_modules(init_default_scope=True);mmcv.use_backend('cv2')
  assert mmcv.adjust_color.__globals__['imread_backend']=='cv2'
  site=pathlib.Path(mmdet.__file__).parent.parent
  for n,h in protocol['official_implementation_sha256'].items():assert sha(site/n)==h,n
  for a in protocol['asset_pins']:
   file=R/a['path'];assert file.stat().st_size==a['bytes'] and sha(file)==a['sha256']
  random.seed(20261031);np.random.seed(20261031);torch.random.default_generator.manual_seed(20261031)
  conf=F/('config_A_reference.json' if args.arm=='A' else 'config_M.json')
  assert sha(conf)==protocol['config_sha256'][args.arm]
  cfg=Config(normalize_config(json.loads(conf.read_text())))
  cfg['model']['language_model']['name']=str(R/'assets/bert-base-uncased')
  start=time.monotonic();model=MODELS.build(cfg['model']);model.init_weights();model.to(torch.device('cpu'))
  source=torch.load(R/protocol['asset_pins'][0]['path'],map_location='cpu',weights_only=True)
  assert isinstance(source,dict) and all(isinstance(t,torch.Tensor) for t in source.values())
  assert all(bool(torch.isfinite(t).all()) for t in source.values() if t.is_floating_point())
  incompatible=model.load_state_dict(source,strict=False)
  schema=validate_pretrained_key_schema(incompatible.missing_keys,incompatible.unexpected_keys)
  state=model.state_dict()
  for name in set(state)&set(source):assert torch.equal(state[name],source[name]),name
  for name in schema['log_scale_keys']:assert state[name].item()==0 and tuple(state[name].shape)==(1,)
  assert tuple(state['dn_query_generator.label_embedding.weight'].shape)==(1,256)
  initial=state_hash(state);del source,state
  assert all(t.device.type=='cpu' for t in list(model.parameters())+list(model.buffers()))
  trainable=[(n,t) for n,t in model.named_parameters() if t.requires_grad]
  names=hashlib.sha256(canonical([n for n,t in trainable])).hexdigest()
  assert names=='62f7c4fa11c27bb72c9d472e1885d9a2ff1cd02f74523f7a53ea334770c4fe3a'
  optim=build_optim_wrapper(model,cfg['optim_wrapper']);assert not optim.optimizer.state
  groups=validate_optimizer_groups(model,optim.optimizer)
  gh=hashlib.sha256(canonical(groups)).hexdigest()
  assert gh=='6b6d9e546744922c8a013ee76b91f647fe185e345a4048fb7bd8a5727b5308f0'
  report.update(model_load_seconds=time.monotonic()-start,initial=initial,trainable_names_sha256=names,trainable_tensors=len(trainable),trainable_parameters=sum(t.numel() for n,t in trainable),optimizer_groups_sha256=gh,optimizer_initial_state_empty=True,official_key_schema=schema)
  atomic(out/'OPTIMIZER_GROUPS.json',groups)
  spec=importlib.util.spec_from_file_location('frozen_observer',R/'ets-pipeline-dryrun-20261101/dryrun.py');observer_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer_module)
  assert sha(pathlib.Path(spec.origin))==protocol['observer_source_sha256']
  observer_module.HERE=out
  assert json.loads(conf.read_text())['train_dataloader']['dataset']['pipeline'][0]['transforms']==observer_module.PROTO['arms'][args.arm]
  data=R/'recovery-v5-20261031/payload/data'
  support=json.loads((data/'annotations/support5.json').read_text());ann={v['image_id']:v['id'] for v in support['annotations']}
  assert sha(data/'annotations/support5.json')==protocol['support_sha256']
  for pin in observer_module.PROTO['support_images']:assert sha(data/'images'/pin['file_name'])==pin['sha256']
  ds=CocoDataset(ann_file='annotations/support5.json',data_root=str(data)+'/',data_prefix={'img':'images/'},metainfo={'classes':('fish',)},filter_cfg={'filter_empty_gt':False,'min_size':0},return_classes=True,pipeline=[])
  assert len(ds)==5
  sampler=DefaultSampler(ds,shuffle=True,seed=20261031,round_up=False);order=[]
  for epoch in range(40):sampler.set_epoch(epoch);order.extend(list(sampler))
  expected=json.loads((F/'reference_A_expected.json').read_text())['anchor_order']
  assert [ds.get_data_info(i)['img_id'] for i in order]==expected
  observer=observer_module.Observer();pipeline=observer_module.make_pipeline(args.arm,observer)
  initial_rng=observer_module.rng()
  report['initial_RNG']=initial_rng
  if args.arm=='M' and args.phase=='formal':
   a=json.loads((P/'formal_A/RESULT.json').read_text())
   assert a['status']=='TRAINING_COMPLETE'
   for k in ['initial','initial_RNG','trainable_names_sha256','optimizer_groups_sha256','optimizer_initial_state_empty']:
    assert report[k]==a[k],'CPU A/M control mismatch: '+k
  selected=protocol['preflight_iterations'][args.arm] if args.phase=='preflight' else list(range(1,201))
  model.train();save('MODEL_AND_CONTROLS_VERIFIED')
  forward_orig=model._run_forward;backward_orig=optim.backward;step_orig=optim.step;timings={}
  def forward(*a,**kw):
   t=time.monotonic();value=forward_orig(*a,**kw);timings['forward_loss_seconds']=time.monotonic()-t;return value
  def backward(loss,**kw):
   assert bool(torch.isfinite(loss).all()),'Nonfinite loss before backward'
   t=time.monotonic();value=backward_orig(loss,**kw);timings['backward_seconds']=time.monotonic()-t;return value
  def step(**kw):
   t=time.monotonic();gradients=[p.grad for n,p in trainable if p.grad is not None]
   assert len(gradients)==896 and all(g.device.type=='cpu' and bool(torch.isfinite(g).all()) for g in gradients),'Nonfinite/missing gradient'
   timings['gradient_check_seconds']=time.monotonic()-t;timings['gradient_tensors']=len(gradients)
   t=time.monotonic();value=step_orig(**kw);timings['optimizer_seconds']=time.monotonic()-t;return value
  model._run_forward=forward;optim.backward=backward;optim.step=step
  exposure=[];steps=[];pipeline_matches=[];(out/'trajectory').mkdir()
  for iteration,index in enumerate(order[:max(selected)],1):
   guard();info=ds.get_data_info(index);observer.begin(iteration,info['img_id'],ann[info['img_id']],(info['height'],info['width']))
   before=observer_module.rng();t=time.monotonic();item=pipeline(info);pipe_seconds=time.monotonic()-t
   sig=observer_module.signature(item)
   ref=json.loads((R/f'ets-pipeline-dryrun-20261101/reference_{args.arm}/{iteration:03}.json').read_text())
   matches=all(sig[k]==ref['signature'][k] for k in ['pixel_sha256','shape_CHW','dtype','gt_sha256','metadata_sha256'])
   # CPU model RNG is separate from geometry RNG; no historical CUDA state equality is claimed.
   assert matches and all(before[k]==ref['RNG_before'][k] for k in ['python','numpy']),'Frozen data trajectory differs'
   pipeline_matches.append(matches)
   visible=[s for s in observer.sources if observer_module.nonempty(s)]
   row={'iteration':iteration,'epoch':(iteration-1)//5,'anchor_image_id':info['img_id'],'RNG_before_pipeline':before,'signature':sig,'pipeline_seconds':pipe_seconds,'events':copy.deepcopy(observer.events),'final_GT_lineage':copy.deepcopy(observer.boxes),'final_source_regions':copy.deepcopy(observer.sources),'selected_source_occurrences':len(observer.sources),'final_nonempty_source_occurrences':len(visible),'final_unique_source_count':len({s['image_id'] for s in visible}),'removed_GT_total':sum(e['discarded_GT_count'] for e in observer.events)}
   if iteration in selected:
    if args.phase=='preflight':assert len(item['data_samples'].gt_instances)>0,'Positive GT probe required'
    timings={};batch=pseudo_collate([item]);row['RNG_before_update']=observer_module.rng()
    t=time.monotonic();log=model.train_step(batch,optim);seconds=time.monotonic()-t
    assert all(bool(torch.isfinite(v).all()) if isinstance(v,torch.Tensor) else np.isfinite(v) for v in log.values())
    report['completed_updates']+=1
    assert all(float(v['step'])==report['completed_updates'] for v in optim.optimizer.state.values())
    row.update(updated=True,train_step_seconds=seconds,all_loss_gradients_finite=True,step_timings=dict(timings),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,RNG_after_update=observer_module.rng())
    steps.append({'iteration':iteration,'input_shape':sig['shape_CHW'],'GT_count':len(item['data_samples'].gt_instances),'pipeline_seconds':pipe_seconds,'train_step_seconds':seconds,**timings,'peak_RSS_bytes':row['peak_RSS_bytes'],'all_loss_gradients_finite':True})
    del batch,log
   atomic(out/'trajectory'/f'{iteration:03}.json',row);exposure.append(row)
   del item,row
   if iteration in selected and (args.phase=='preflight' or iteration%10==0):save('TRAINING')
  report.update(steps=steps,pipeline_reference_matches=sum(pipeline_matches),anchor_order=[r['anchor_image_id'] for r in exposure],source_exposure={'anchor_count':len(exposure),'selected_occurrences':sum(r['selected_source_occurrences'] for r in exposure),'visible_occurrences':sum(r['final_nonempty_source_occurrences'] for r in exposure),'GT_presentations':sum(len(r['signature']['gt']['gt_instances']['boxes']) for r in exposure),'empty_GT_inputs':sum(not r['signature']['gt']['gt_instances']['boxes'] for r in exposure),'GT_removed':sum(r['removed_GT_total'] for r in exposure),'Mosaic_applied':sum(e.get('applied',False) for r in exposure for e in r['events'] if e['type']=='CachedMosaic'),'MixUp_applied':sum(e.get('applied',False) for r in exposure for e in r['events'] if e['type']=='CachedMixUp')})
  if args.phase=='formal':
   assert report['completed_updates']==200 and len(exposure)==200
   guard(extra=700*1024**2);t=time.monotonic();state=model.state_dict()
   assert all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point())
   receipt=state_hash(state);checkpoint=out/'final_model.pth';temporary=out/'final_model.pth.partial'
   with temporary.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
   os.replace(temporary,checkpoint)
   loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
   assert state_hash(loaded)==receipt and all(torch.equal(state[n],loaded[n]) for n in state)
   assert len(loaded)==908;del loaded,state
   report['checkpoint']={'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':sha(checkpoint),'load':receipt,'weights_only':True,'complete_state_verified':True,'save_and_verify_seconds':time.monotonic()-t}
   atomic(out/'CHECKPOINT_RECEIPT.json',{'load':receipt,**report['checkpoint']})
   report['status']='TRAINING_COMPLETE'
  else:report['status']='PREFLIGHT_COMPLETE'
  save('COMPLETE')
 except BaseException as exc:
  report.update(status='INCOMPLETE',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());save('FAILED');raise

if __name__=='__main__':main()
