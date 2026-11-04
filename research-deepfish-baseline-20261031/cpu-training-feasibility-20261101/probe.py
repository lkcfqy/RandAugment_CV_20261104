"""Exactly one support image / one official CPU train_step. No dev/test/persistence of weights."""
from portable_dev.paths import project_path as _portable_project_path
import copy,hashlib,json,os,pathlib,random,resource,sys,time,traceback
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
F=R/'fixed-M-execution-20261101/payload'
sys.path.insert(0,str(F))
from candidate_worker import normalize_config
from candidate_control import validate_optimizer_groups
from contracts import canonical,validate_pretrained_key_schema

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
report=dict(status='STARTING',seed=20261031,image_id=2244,device='cpu',model_instances=0,
 formal_training=False,dev_test_loaded=False,optimizer_updates=0,loss_values_disclosed=False)
began=time.monotonic()
def save(stage):
 report.update(stage=stage,elapsed_seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (P/'RESULT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:report[k] for k in ['stage','elapsed_seconds','peak_RSS_bytes','optimizer_updates']}),flush=True)

try:
 import numpy as np,cv2,torch,mmcv
 from mmdet.registry import MODELS,DATASETS
 from mmdet.utils import register_all_modules
 from mmengine.optim import build_optim_wrapper
 from mmengine.dataset import pseudo_collate
 from mmengine.config import Config
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(0)
 torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
 assert torch.version.cuda is None
 register_all_modules(init_default_scope=True);mmcv.use_backend('cv2')
 random.seed(20261031);np.random.seed(20261031);torch.random.default_generator.manual_seed(20261031)
 config_path=F/'config_M.json';assert sha(config_path)=='72f991e8d27439ab9732eb21ef30365e0bb91ecb1386c65785b9871d09a8a6aa'
 cfg=Config(normalize_config(json.loads(config_path.read_text())))
 cfg['model']['language_model']['name']=str(R/'assets/bert-base-uncased')
 pin=json.loads((F/'asset_pins.json').read_text())['assets'][0];weight=R/pin['path']
 assert weight.stat().st_size==pin['bytes'] and sha(weight)==pin['sha256']
 for asset in json.loads((F/'asset_pins.json').read_text())['assets'][1:]:
  f=R/asset['path'];assert f.stat().st_size==asset['bytes'] and sha(f)==asset['sha256']
 report.update(torch_version=torch.__version__,config_sha256=sha(config_path),weight_sha256=pin['sha256'],threads=4,interop=1,OpenCV_threads=cv2.getNumThreads())
 save('ASSETS_AND_CPU_ENV_VERIFIED')
 t=time.monotonic();model=MODELS.build(cfg['model']);report['model_instances']=1;model.init_weights();model.to(torch.device('cpu'))
 source=torch.load(weight,map_location='cpu',weights_only=True)
 assert isinstance(source,dict) and all(isinstance(v,torch.Tensor) for v in source.values())
 assert all(bool(torch.isfinite(v).all()) for v in source.values() if v.is_floating_point())
 incompatible=model.load_state_dict(source,strict=False)
 schema=validate_pretrained_key_schema(incompatible.missing_keys,incompatible.unexpected_keys)
 state=model.state_dict()
 for name in set(state)&set(source):assert torch.equal(state[name],source[name]),name
 for name in schema['log_scale_keys']:assert state[name].item()==0.0 and tuple(state[name].shape)==(1,)
 assert tuple(state['dn_query_generator.label_embedding.weight'].shape)==(1,256)
 del source,state
 assert all(v.device.type=='cpu' for v in list(model.parameters())+list(model.buffers()))
 trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 frozen=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]
 names_hash=hashlib.sha256(canonical([n for n,p in trainable])).hexdigest()
 assert names_hash=='62f7c4fa11c27bb72c9d472e1885d9a2ff1cd02f74523f7a53ea334770c4fe3a'
 report.update(model_load_seconds=time.monotonic()-t,trainable_tensors=len(trainable),trainable_parameters=sum(p.numel() for n,p in trainable),frozen_tensors=len(frozen),frozen_parameters=sum(p.numel() for n,p in frozen),trainable_names_sha256=names_hash,official_key_schema=schema)
 (P/'PARAMETERS.json').write_text(json.dumps([dict(name=n,shape=list(p.shape),requires_grad=p.requires_grad,numel=p.numel()) for n,p in model.named_parameters()],indent=2)+'\n')
 optim=build_optim_wrapper(model,cfg['optim_wrapper']);assert not optim.optimizer.state
 groups=validate_optimizer_groups(model,optim.optimizer);report['optimizer_groups_sha256']=hashlib.sha256(canonical(groups)).hexdigest()
 model.train()
 data=copy.deepcopy(cfg['train_dataloader']['dataset']);data['data_root']=str(R/'recovery-v5-20261031/payload/data')+'/'
 data['pipeline']=data['pipeline'][0]['transforms'];dataset=DATASETS.build(data)
 assert len(dataset)==5
 index=next(i for i in range(5) if dataset.get_data_info(i)['img_id']==2244)
 t=time.monotonic();item=dataset[index];report['pipeline_seconds']=time.monotonic()-t
 report.update(input_shape=list(item['inputs'].shape),GT_count=len(item['data_samples'].gt_instances),sample_fetches=1,cache_state='first frozen M input; empty initial caches; Mosaic/MixUp warmup not complete')
 batch=pseudo_collate([item]);save('READY_ONE_CPU_STEP')
 original_forward=model._run_forward;original_backward=optim.backward;original_step=optim.step
 def forward(*a,**kw):
  t=time.monotonic()
  try:return original_forward(*a,**kw)
  finally:report['forward_loss_seconds']=time.monotonic()-t;save('FORWARD_FINISHED')
 def backward(*a,**kw):
  t=time.monotonic()
  try:return original_backward(*a,**kw)
  finally:report['backward_seconds']=time.monotonic()-t;save('BACKWARD_FINISHED')
 def step(**kw):
  t=time.monotonic();gradients=[p.grad for n,p in trainable if p.grad is not None]
  assert gradients and all(g.device.type=='cpu' and bool(torch.isfinite(g).all()) for g in gradients)
  report.update(gradient_tensors=len(gradients),all_gradients_finite=True,gradient_check_seconds=time.monotonic()-t)
  t=time.monotonic();result=original_step(**kw);report['optimizer_seconds']=time.monotonic()-t;report['optimizer_updates']+=1;return result
 model._run_forward=forward;optim.backward=backward;optim.step=step
 t=time.monotonic();log_vars=model.train_step(batch,optim);report['full_train_step_seconds']=time.monotonic()-t
 assert all(bool(torch.isfinite(v).all()) if isinstance(v,torch.Tensor) else np.isfinite(v) for v in log_vars.values())
 assert report['optimizer_updates']==1 and optim.optimizer.state
 assert all(float(v['step'])==1.0 for v in optim.optimizer.state.values())
 report.update(status='ONE_CPU_FORWARD_BACKWARD_OPTIMIZER_STEP_PASS',all_loss_components_finite=True,optimizer_state_entries=len(optim.optimizer.state),all_optimizer_steps_one=True,trained_checkpoint_saved=False)
 save('COMPLETE')
except BaseException as error:
 report.update(status='CPU_STEP_BLOCKED',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
 save('FAILED');raise
