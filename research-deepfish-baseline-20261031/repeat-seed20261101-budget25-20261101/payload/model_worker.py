"""Future isolated official-Runner phases. Importing this file imports no model framework."""
import argparse,copy,hashlib,json,math,os,pathlib,random,sys,time
from contracts import ContractError,ResourceUndetermined,canonical,validate_context,verify_exposures,validate_pretrained_key_schema,repeat_training_seed
from lifecycle import atomic_json
from image_contracts import require_bgr_uint8,verify_color_contract
ROOT=pathlib.Path(__file__).resolve().parent
def file_sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
 return h.hexdigest()
def percentile90(values):return sorted(values)[max(0,math.ceil(.9*len(values))-1)]
def normalize_config(config):
 """Restore only tuple-valued official config fields lost during JSON encoding."""
 def restore(value):
  if isinstance(value,list):return [restore(x) for x in value]
  if not isinstance(value,dict):return value
  result={k:restore(v) for k,v in value.items()}
  for key in ('classes','scale','crop_size'):
   if isinstance(result.get(key),list):result[key]=tuple(result[key])
  if 'scales' in result:result['scales']=[tuple(x) if isinstance(x,list) else x for x in result['scales']]
  return result
 return restore(config)
def worker(context_path,phase):
 ctx=json.loads(pathlib.Path(context_path).read_text());validate_context(ctx,time.time())
 if time.monotonic()>=ctx['work_deadline_monotonic']:raise ResourceUndetermined('Worker deadline exhausted.')
 if os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')!='1' or os.environ.get('HF_HUB_OFFLINE')!='1':raise ContractError('Safe-load and offline environment required.')
 out=pathlib.Path(ctx['artifact_dir']);kind,arm=phase.split('_');start=time.monotonic()
 if kind not in ('preflight','formal','evaluate') or arm not in ('A','B'):raise ContractError('Unknown phase.')
 cfg=normalize_config(json.loads((ROOT/('config_'+arm+'.json')).read_text()));seed=repeat_training_seed(cfg)
 freeze=None
 if kind!='preflight':
  freeze=json.loads((out/'U_FREEZE.json').read_text())
  if freeze['status']!='FROZEN' or freeze['dev_metrics_seen'] or freeze['preflight_state_reused'] or freeze['updates']!=200 or freeze.get('seed')!=seed:raise ContractError('Invalid fixed-U repeat freeze.')
 # All imports below occur only in the separately authorized future worker.
 import numpy as np
 import torch
 import mmcv,mmdet
 from mmengine.config import Config
 from mmengine.hooks import Hook
 from mmengine.runner import Runner
 from mmdet.registry import TRANSFORMS
 from mmdet.utils import register_all_modules
 from mmdet.datasets.transforms import Color
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ResourceUndetermined('Exactly one CUDA GPU is required.')
 if torch.__version__.split('+')[0]!='2.1.0' or torch.version.cuda!='11.8' or mmcv.__version__!='2.1.0' or mmdet.__version__!='3.3.0':raise ContractError('Candidate core versions differ.')
 pins=json.loads((ROOT/'implementation_pins.json').read_text());site=pathlib.Path(mmdet.__file__).parent.parent
 for name,expected in pins['file_sha256'].items():
  if file_sha(site/name)!=expected:raise ContractError('Official implementation file differs: '+name)
 mmcv.use_backend('cv2')
 if mmcv.adjust_color.__globals__.get('imread_backend')!='cv2':raise ContractError('Color backend must be cv2.')
 color_gate=verify_color_contract(Color)
 register_all_modules(init_default_scope=True)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 cfg['work_dir']=str(out/(phase+'_logs'))
 cfg['model']['language_model']['name']=str(ROOT/'assets/bert-base-uncased')
 cfg['train_dataloader']['dataset']['data_root']=str(ROOT/'data')+'/'
 cfg['test_dataloader']['dataset']['data_root']=str(ROOT/'data')+'/'
 cfg['test_evaluator']['ann_file']=str(ROOT/'data/annotations/dev64.json')
 original=ROOT/'assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
 expected_asset=json.loads((ROOT/'asset_pins.json').read_text())['assets'][0]
 if original.stat().st_size!=expected_asset['bytes'] or file_sha(original)!=expected_asset['sha256']:raise ContractError('Original checkpoint hash changed.')
 source_state=torch.load(original,map_location='cpu',weights_only=True)
 if not isinstance(source_state,dict) or not all(isinstance(v,torch.Tensor) for v in source_state.values()):raise ContractError('Checkpoint must contain only a tensor state dict.')
 if not all(bool(torch.isfinite(v).all()) for v in source_state.values() if v.is_floating_point()):raise ContractError('Nonfinite source checkpoint.')
 cfg['load_from']=str(original)
 total=15 if kind=='preflight' else freeze['updates'];cfg['train_cfg']['max_epochs']=total//5
 def rng():
  s=np.random.get_state();n=[s[0],s[1].tolist(),s[2],s[3],s[4]]
  return {'numpy':hashlib.sha256(canonical(n)).hexdigest(),'python':hashlib.sha256(canonical(random.getstate())).hexdigest(),'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
 def plain(v):
  if hasattr(v,'tensor'):v=v.tensor
  if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,(list,tuple)):return [plain(x) for x in v]
  if isinstance(v,np.generic):return v.item()
  return v
 def snapshot(results):return {k:plain(results[k]) for k in ('img_shape','scale','scale_factor','flip','flip_direction','homography_matrix','gt_bboxes','gt_bboxes_labels','gt_ignore_flags') if k in results}
 @TRANSFORMS.register_module()
 class AuditedPipeline:
  def __init__(self,transforms):
   self.steps=[]
   def wrap(op,label):
    # Instrument existing branch Compose leaves; do not reimplement random geometry.
    if op.__class__.__name__=='RandomChoice':
     for b,branch in enumerate(op.transforms):branch.transforms=[wrap(x,label+'.branch'+str(b)+'.'+str(i)) for i,x in enumerate(branch.transforms)]
    def call(results):
     before=rng();before_data=snapshot(results)
     if op.__class__.__name__=='Color':require_bgr_uint8(results['img'],'cv2')
     result=op(results)
     if result is None:raise ContractError('Unexpected sample refetch would change source exposure.')
     self.steps.append({'op':op.__class__.__name__,'path':label,'before':before_data,'after':snapshot(result),'rng_before':before,'rng_after':rng()})
     return result
    return call
   self.pipeline=[wrap(TRANSFORMS.build(t),str(i)) for i,t in enumerate(transforms)]
  def __call__(self,results):
   self.steps=[];image_id=results['img_id']
   for t in self.pipeline:results=t(results)
   sample=results['data_samples'];record={'image_id':image_id,'steps':copy.deepcopy(self.steps),'packed_gt':plain(sample.gt_instances.bboxes),'packed_labels':plain(sample.gt_instances.labels)}
   sample.set_metainfo({'paired_audit':record});return results
 if kind=='evaluate':
  cfg['test_evaluator']['outfile_prefix']=str(out/(phase+'_predictions'))
  pair=json.loads((out/'PAIR_CHECK_OK.json').read_text())
  if pair['freeze_sha256']!=file_sha(out/'U_FREEZE.json') or pair['updates_per_arm']!=total:raise ContractError('Final-pair gate is missing or stale.')
  report=json.loads((out/('formal_'+arm+'.json')).read_text());final=out/report['final_checkpoint']['path']
  if file_sha(final)!=report['final_checkpoint']['sha256']:raise ContractError('Fixed final checkpoint digest mismatch.')
  final_state=torch.load(final,map_location='cpu',weights_only=True)
  if not isinstance(final_state,dict) or not all(isinstance(v,torch.Tensor) for v in final_state.values()):raise ContractError('Final checkpoint is not tensor-only.')
  del final_state,source_state;cfg['load_from']=str(final)
  runner=Runner.from_cfg(Config(cfg));metrics=runner.test();atomic_json(out/(phase+'.json'),{'status':'COMPLETE','updates':total,'checkpoint_sha256':file_sha(final),'dev_images':64,'dev_boxes':187,'metrics':{k:float(v) for k,v in metrics.items()},'official_test_used':False});return
 records=[];step_times=[];report={'phase':phase,'seed':seed,'color_gate':color_gate,'preflight_state_reused':False,'loss_and_gradients_finite':True}
 def state_digest(state):
  h=hashlib.sha256()
  for name,value in sorted(state.items()):
   t=value.detach().cpu().contiguous();h.update(canonical([name,str(t.dtype),list(t.shape)]));h.update(t.numpy().tobytes())
  return h.hexdigest()
 class AuditHook(Hook):
  priority='VERY_HIGH'
  def before_train(self,runner):
   state=runner.model.state_dict();missing=set(state)-set(source_state);extra=set(source_state)-set(state)
   schema=validate_pretrained_key_schema(missing,extra)
   if cfg['model']['bbox_head']['contrastive_cfg']['log_scale']!=0.0 or cfg['model']['language_model']['add_pooling_layer'] is not False:raise ContractError('Frozen official configuration changed.')
   if tuple(state['dn_query_generator.label_embedding.weight'].shape)!=(1,256):raise ContractError('Unexpected denoising embedding shape.')
   for key in schema['log_scale_keys']:
    if tuple(state[key].shape)!=(1,) or state[key].dtype!=torch.float32 or state[key].item()!=0.0 or not runner.model.get_parameter(key).requires_grad:raise ContractError('Official contrastive log_scale initialization differs.')
   for key in set(state)&set(source_state):
    if state[key].shape!=source_state[key].shape or not torch.equal(state[key].detach().cpu(),source_state[key]):raise ContractError('Pretrained tensor not loaded exactly: '+key)
   positional='language_model.language_backbone.body.model.embeddings.position_ids'
   if not torch.equal(source_state[positional],torch.arange(512).reshape(1,512)):raise ContractError('Unexpected legacy positional buffer.')
   for suffix,shape in [('weight',(768,768)),('bias',(768,))]:
    if tuple(source_state['language_model.language_backbone.body.model.pooler.dense.'+suffix].shape)!=shape:raise ContractError('Unexpected unused official BERT pooler shape.')
   if runner.optim_wrapper.optimizer.state:raise ContractError('Optimizer must start empty.')
   trainable=[n for n,p in runner.model.named_parameters() if p.requires_grad]
   report.update(initial_state_sha256=state_digest(state),trainable_names_sha256=hashlib.sha256(canonical(trainable)).hexdigest(),optimizer_initial_state_empty=True,missing_keys=sorted(missing),official_zero_log_scale_keys=schema['log_scale_keys'],ignored_checkpoint_tensors=sorted(extra),state_tensor_bytes=sum(t.numel()*t.element_size() for t in state.values()),init_seconds=time.monotonic()-start)
   atomic_json(out/(phase+'_initialization.json'),dict(report,status='INITIALIZATION_VERIFIED'))
   original_step=runner.optim_wrapper.step
   def checked_step(**kwargs):
    flags=[torch.isfinite(p.grad).all() for p in runner.model.parameters() if p.grad is not None]
    if not flags or not bool(torch.stack(flags).all()):raise ContractError('Nonfinite or absent gradients.')
    return original_step(**kwargs)
   runner.optim_wrapper.step=checked_step;torch.cuda.reset_peak_memory_stats();self.last=time.monotonic()
  def before_train_iter(self,runner,batch_idx,data_batch=None):
   if time.monotonic()>=ctx['work_deadline_monotonic']:raise ResourceUndetermined('Training work deadline.')
   sample=data_batch['data_samples'][0];self.current=copy.deepcopy(sample.metainfo['paired_audit']);self.current['cuda_rng_before_update']=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
   self.current['input_shape']=list(data_batch['inputs'][0].shape)
   torch.cuda.synchronize()
  def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
   torch.cuda.synchronize();now=time.monotonic();step_times.append(now-self.last);self.last=now
   if outputs and any(not math.isfinite(float(v)) for v in outputs.values()):raise ContractError('Nonfinite training output.')
   self.current['rng_after_update']=dict(rng(),torch_cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
   records.append(self.current);atomic_json(out/(phase+'_progress.json'),{'phase':phase,'completed_optimizer_updates':len(records),'dev_metrics_seen':False})
 runner=Runner.from_cfg(Config(cfg));runner.register_hook(AuditHook());runner.train()
 if runner.iter!=total:raise ContractError('Formal/preflight update count mismatch.')
 report['exposures']=verify_exposures([r['image_id'] for r in records],total);report['peak_cuda_bytes']=torch.cuda.max_memory_allocated();report['actual_updates']=runner.iter
 atomic_json(out/(phase+'_trace.json'),records)
 if kind=='preflight':
  if len(step_times)!=15:raise ContractError('Preflight must contain exactly 15 optimizer updates.')
  report.update(warmup_updates=5,timed_updates=10,p90_update_seconds=percentile90(step_times[5:]),update_seconds=step_times)
  inference_cfg=copy.deepcopy(cfg['test_dataloader']);inference_cfg['dataset']['ann_file']='annotations/support5.json'
  loader=Runner.build_dataloader(inference_cfg,seed=seed);runner.model.eval();times=[]
  with torch.no_grad():
   for batch in loader:
    if time.monotonic()>=ctx['work_deadline_monotonic']:raise ResourceUndetermined('Inference timing deadline.')
    torch.cuda.synchronize();t=time.monotonic();prediction=runner.model.test_step(batch);torch.cuda.synchronize();times.append(time.monotonic()-t);del prediction
  if len(times)!=5:raise ContractError('Profile inference uses five support images only.')
  report.update(p90_predict_seconds=percentile90(times),model_outputs_saved=False,preflight_checkpoint_saved=False,dev_metrics_computed=False)
 else:
  final=out/('final_'+arm+'_U'+str(total)+'.pth');temporary=final.with_suffix('.tmp');state={k:v.detach().cpu() for k,v in runner.model.state_dict().items()}
  with temporary.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
  os.replace(temporary,final);report['final_checkpoint']={'path':final.name,'bytes':final.stat().st_size,'sha256':file_sha(final)}
 report['status']='COMPLETE';atomic_json(out/(phase+'.json'),report)
 # Process termination discards all preflight model, optimizer and RNG state.
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context',required=True);p.add_argument('--phase',choices=[s+'_'+a for s in ('preflight','formal','evaluate') for a in 'AB'],required=True);a=p.parse_args();worker(a.context,a.phase)
if __name__=='__main__':main()
