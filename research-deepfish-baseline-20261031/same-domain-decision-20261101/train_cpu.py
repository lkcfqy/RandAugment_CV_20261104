"""Three frozen CPU arms; same official initialization and per-source mean control."""
from portable_dev.paths import project_path as _portable_project_path
import argparse,copy,hashlib,importlib.util,json,os,pathlib,random,resource,shutil,sys,time,traceback
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
F=R/'fixed-M-execution-20261101/payload'
sys.path.insert(0,str(F))
from candidate_worker import normalize_config
from candidate_control import validate_optimizer_groups
from contracts import canonical,validate_pretrained_key_schema
from mean_update import source_mean_update

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 b=json.dumps(v,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n';tmp=p.with_name(p.name+'.partial')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);assert p.read_bytes()==b

def tree(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def state_hash(state):
 h=hashlib.sha256();size=0
 for n,t in sorted(state.items()):
  b=t.detach().contiguous().numpy().tobytes();h.update(json.dumps([n,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(b);size+=len(b)
 return {'state_sha256':h.hexdigest(),'tensor_count':len(state),'tensor_bytes':size}
last_storage=0
def guard(extra=0):
 global last_storage
 ctx=read(P/'RUN_STARTED.json');assert time.monotonic()<ctx['deadline_monotonic'],'Three-hour wall limit'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=12*1024**3,'12GiB peak RSS limit'
 assert shutil.disk_usage(P).free>80*1024**2+extra,'Filesystem margin'
 if extra or time.monotonic()-last_storage>30:
  assert tree(R)+tree(P)+extra<=12*1024**3-80*1024**2,'12GiB research+staging margin';last_storage=time.monotonic()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','formal']);parser.add_argument('arm',choices=['A1','M1','C0']);args=parser.parse_args();out=P/(args.phase+'_'+args.arm);out.mkdir(exist_ok=False)
 proto=read(P/'PROTOCOL.json');began=time.monotonic();report={'status':'STARTED','phase':args.phase,'arm':args.arm,'PID':os.getpid(),'seed':20261031,'completed_updates':0,'completed_source_forwards':0,'formal_updates_planned':200 if args.phase=='formal' else 0,'dev_test_access':False,'GPU_MPS_upload_CU_new':0}
 def save(stage):
  report.update(stage=stage,elapsed_seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic(out/'RESULT.json',report);print(json.dumps({k:report[k] for k in ['stage','arm','completed_updates','completed_source_forwards','elapsed_seconds','peak_RSS_bytes']}),flush=True)
 try:
  for name,h in read(P/'CODE_PINS.json').items():assert sha(P/name)==h,name
  guard();import numpy as np,cv2,torch,mmcv,mmdet
  from mmdet.registry import MODELS
  from mmdet.utils import register_all_modules
  from mmdet.datasets import CocoDataset
  from mmengine.config import Config
  from mmengine.dataset import DefaultSampler,pseudo_collate
  from mmengine.optim import build_optim_wrapper
  torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(0);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
  assert torch.version.cuda is None and cv2.getNumThreads()==1
  register_all_modules(init_default_scope=True);mmcv.use_backend('cv2');assert mmcv.adjust_color.__globals__['imread_backend']=='cv2'
  site=pathlib.Path(mmdet.__file__).parent.parent
  for n,h in proto['official_implementation_sha256'].items():assert sha(site/n)==h,n
  for pin in proto['asset_pins']:
   file=R/pin['path'];assert file.stat().st_size==pin['bytes'] and sha(file)==pin['sha256']
  random.seed(20261031);np.random.seed(20261031);torch.random.default_generator.manual_seed(20261031)
  recipe='M' if args.arm=='M1' else 'A';conf=F/('config_M.json' if recipe=='M' else 'config_A_reference.json');assert sha(conf)==proto['config_sha256'][recipe]
  cfg=Config(normalize_config(read(conf)));assert cfg['param_scheduler']==[];cfg['model']['language_model']['name']=str(R/'assets/bert-base-uncased')
  t=time.monotonic();model=MODELS.build(cfg['model']);model.init_weights();model.to(torch.device('cpu'))
  source=torch.load(R/proto['asset_pins'][0]['path'],map_location='cpu',weights_only=True);assert isinstance(source,dict) and all(isinstance(t,torch.Tensor) for t in source.values());assert all(bool(torch.isfinite(t).all()) for t in source.values() if t.is_floating_point())
  inc=model.load_state_dict(source,strict=False);schema=validate_pretrained_key_schema(inc.missing_keys,inc.unexpected_keys);state=model.state_dict()
  for n in set(state)&set(source):assert torch.equal(state[n],source[n]),n
  for n in schema['log_scale_keys']:assert state[n].item()==0 and tuple(state[n].shape)==(1,)
  initial=state_hash(state);del source,state
  assert all(t.device.type=='cpu' for t in list(model.parameters())+list(model.buffers()))
  trainable=[(n,t) for n,t in model.named_parameters() if t.requires_grad];names=hashlib.sha256(canonical([n for n,t in trainable])).hexdigest();optim=build_optim_wrapper(model,cfg['optim_wrapper']);assert not optim.optimizer.state
  groups=validate_optimizer_groups(model,optim.optimizer);gh=hashlib.sha256(canonical(groups)).hexdigest();lr=[(g['lr'],g['weight_decay']) for g in optim.optimizer.param_groups]
  assert names=='62f7c4fa11c27bb72c9d472e1885d9a2ff1cd02f74523f7a53ea334770c4fe3a' and gh=='6b6d9e546744922c8a013ee76b91f647fe185e345a4048fb7bd8a5727b5308f0'
  report.update(model_load_seconds=time.monotonic()-t,initial=initial,trainable_names_sha256=names,trainable_tensors=len(trainable),trainable_parameters=sum(t.numel() for n,t in trainable),optimizer_groups_sha256=gh,optimizer_initial_state_empty=True,official_key_schema=schema)
  atomic(out/'OPTIMIZER_GROUPS.json',groups)
  spec=importlib.util.spec_from_file_location('frozen_observer',R/'ets-pipeline-dryrun-20261101/dryrun.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o);assert sha(pathlib.Path(spec.origin))==proto['observer_source_sha256']
  data=P/'data' if args.arm!='C0' else R/'recovery-v5-20261031/payload/data';annfile='annotations/support1.json' if args.arm!='C0' else 'annotations/support5.json';assert sha(data/annfile)==proto['support_sha256']['new' if args.arm!='C0' else 'old'];shot=read(data/annfile);aid={a['image_id']:a['id'] for a in shot['annotations']}
  ds=CocoDataset(ann_file=annfile,data_root=str(data)+'/',data_prefix={'img':'images/'},metainfo={'classes':('fish',)},filter_cfg={'filter_empty_gt':False,'min_size':0},return_classes=True,pipeline=[]);assert len(ds)==5;index={ds.get_data_info(i)['img_id']:i for i in range(5)}
  pins=read(P/'SUPPORT_FORMAT_CHECK.json') if args.arm!='C0' else o.PROTO['support_images']
  for pin in pins:assert sha(data/'images'/pin['file_name'])==pin['sha256']
  sampler=DefaultSampler(ds,shuffle=True,seed=20261031,round_up=False);order=[]
  for epoch in range(40):sampler.set_epoch(epoch);order.extend(list(sampler))
  ledger=read(P/'C0_FROZEN_SOURCE_LEDGER.json');assert sha(P/'C0_FROZEN_SOURCE_LEDGER.json')==proto['C0_source_ledger_sha256']
  source_groups=[r['source_image_ids'] for r in ledger] if args.arm=='C0' else [[ds.get_data_info(i)['img_id']] for i in order]
  obs=o.Observer();pipeline=o.make_pipeline(recipe,obs);report['initial_RNG']=o.rng();historical=read(R/'cpu-paired-AM-20261101/formal_A/RESULT.json')
  for k in ['initial','initial_RNG','trainable_names_sha256','optimizer_groups_sha256','optimizer_initial_state_empty']:assert report[k]==historical[k],'Historical CPU A0 control mismatch: '+k
  report['same_initial_controls_as_A0_M0']=True;atomic(out/'INITIAL_CONTROL_CHECK.json',{k:report[k] for k in ['initial','initial_RNG','trainable_names_sha256','optimizer_groups_sha256','optimizer_initial_state_empty']})
  refs={pathlib.Path(x['path']).name:x for x in read(P/f'DATA_REFERENCE_MANIFEST_{args.arm}.json')}
  selected=proto['preflight_iterations'][args.arm] if args.phase=='preflight' else list(range(1,201));model.train();save('MODEL_AND_CONTROLS_VERIFIED');(out/'trajectory').mkdir();steps=[];occ=0;exposure=[];counts={'clip':0,'optimizer':0,'zero_grad':0,'schedule_boundary':0}
  for attr,key in [('_clip_grad','clip'),('step','optimizer'),('zero_grad','zero_grad')]:
   original=getattr(optim,attr)
   def counted(*a,_original=original,_key=key,**kw):counts[_key]+=1;return _original(*a,**kw)
   setattr(optim,attr,counted)
  def finite_grad():
   g=[p.grad for n,p in trainable if p.grad is not None];assert len(g)==896 and all(x.device.type=='cpu' and bool(torch.isfinite(x).all()) for x in g),'Invalid accumulated gradient'
  def boundary():
   counts['schedule_boundary']+=1;assert [(g['lr'],g['weight_decay']) for g in optim.optimizer.param_groups]==lr,'Constant LR/decay changed'
  for u,ids in enumerate(source_groups[:max(selected)],1):
   guard();refpath=P/f'data_reference_{args.arm}/{u:03}.json';assert sha(refpath)==refs[refpath.name]['sha256'];ref=read(refpath);assert ids==ref['source_image_ids'];row={'update':u,'k':len(ids),'source_image_ids':ids,'sources':[],'RNG_before_update_sources':o.rng()}
   def batches():
    nonlocal occ
    for j,iid in enumerate(ids):
     guard();occ+=1;info=ds.get_data_info(index[iid]);obs.begin(occ,iid,aid[iid],(info['height'],info['width']));before=o.rng();t=time.monotonic();item=pipeline(info);sec=time.monotonic()-t;sig=o.signature(item);reference=ref['sources'][j]
     assert all(sig[k]==reference['signature'][k] for k in ['pixel_sha256','shape_CHW','dtype','gt_sha256','metadata_sha256']) and all(before[k]==reference['RNG_before'][k] for k in ['python','numpy']),'Frozen data trajectory mismatch'
     s={'source_image_id':iid,'annotation_id':aid[iid],'RNG_before_pipeline':before,'RNG_before_source_forward':o.rng(),'signature':sig,'pipeline_seconds':sec,'events':copy.deepcopy(obs.events),'final_GT_lineage':copy.deepcopy(obs.boxes),'final_source_regions':copy.deepcopy(obs.sources),'reference_match':True};row['sources'].append(s)
     yield pseudo_collate([item])
     s['RNG_after_source']=o.rng();del item
   if u in selected:
    t=time.monotonic()
    if args.arm=='C0':
     result=source_mean_update(model,optim,batches(),len(ids),finite_grad,boundary)
    else:
     batches_iter=batches();batch=next(batches_iter)
     # Same official A/M train_step as historical pair; check gradients before its single step.
     original_step=optim.step
     def checked(**kw):finite_grad();return original_step(**kw)
     optim.step=checked;logs=model.train_step(batch,optim);optim.step=original_step
     assert all(bool(torch.isfinite(v).all()) if isinstance(v,torch.Tensor) else np.isfinite(v) for v in logs.values());boundary()
     try:next(batches_iter);raise AssertionError('Expected one anchor batch')
     except StopIteration:pass
     result={'source_count':1,'mean_loss':float(logs['loss']),'update_seconds':time.monotonic()-t};del batch,logs,batches_iter
    seconds=time.monotonic()-t;report['completed_updates']+=1;report['completed_source_forwards']+=len(ids)
    assert all(float(v['step'])==report['completed_updates'] for v in optim.optimizer.state.values());assert all(n==report['completed_updates'] for n in counts.values())
    row.update(updated=True,train_step_seconds=seconds,mean_update=result,all_loss_gradients_finite=True,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,RNG_after_update=o.rng());steps.append({'update':u,'k':len(ids),'train_step_seconds':seconds,'max_input_shape':max((s['signature']['shape_CHW'] for s in row['sources']),key=lambda x:x[1]*x[2]),'positive_GT_sources':sum(bool(s['signature']['gt']['gt_instances']['boxes']) for s in row['sources']),'peak_RSS_bytes':row['peak_RSS_bytes']})
    if args.phase=='preflight':assert steps[-1]['positive_GT_sources']>0
   else:
    for batch in batches():del batch
   atomic(out/'trajectory'/f'{u:03}.json',row);exposure.append(row)
   if u in selected and (args.phase=='preflight' or u%5==0):save('TRAINING')
  report.update(steps=steps,update_call_counts=counts,scheduler_objects=0,source_groups=[r['source_image_ids'] for r in exposure],source_exposure={'source_forwards':report['completed_source_forwards'],'selected_occurrences':sum(len(s['final_source_regions']) for r in exposure for s in r['sources']),'GT_presentations':sum(len(s['signature']['gt']['gt_instances']['boxes']) for r in exposure for s in r['sources']),'empty_GT_inputs':sum(not s['signature']['gt']['gt_instances']['boxes'] for r in exposure for s in r['sources']),'Mosaic_applied':sum(e.get('applied',False) for r in exposure for s in r['sources'] for e in s['events'] if e['type']=='CachedMosaic'),'MixUp_applied':sum(e.get('applied',False) for r in exposure for s in r['sources'] for e in s['events'] if e['type']=='CachedMixUp')})
  if args.phase=='formal':
   assert report['completed_updates']==200 and report['completed_source_forwards']==(659 if args.arm=='C0' else 200);guard(extra=700*1024**2);t=time.monotonic();state=model.state_dict();assert all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point());receipt=state_hash(state);ck=out/'final_model.pth';temp=out/'final_model.pth.partial'
   with temp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
   os.replace(temp,ck);loaded=torch.load(ck,map_location='cpu',weights_only=True);assert state_hash(loaded)==receipt and all(torch.equal(state[n],loaded[n]) for n in state);assert len(loaded)==908;del loaded,state
   report['checkpoint']={'path':str(ck),'bytes':ck.stat().st_size,'sha256':sha(ck),'load':receipt,'weights_only':True,'complete_state_verified':True,'save_and_verify_seconds':time.monotonic()-t};atomic(out/'CHECKPOINT_RECEIPT.json',report['checkpoint']);report['status']='TRAINING_COMPLETE'
  else:report['status']='PREFLIGHT_COMPLETE'
  save('COMPLETE')
 except BaseException as exc:
  report.update(status='INCOMPLETE',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());save('FAILED');raise
if __name__=='__main__':main()
