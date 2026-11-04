"""Pure guards for one fixed M candidate; importing this file runs no model."""
from contracts import ContractError

EXPERIMENT_ID='deepfish-fixed-M-seed20261031-U200'

def validate_candidate_context(ctx):
 if ctx.get('experiment_id')!=EXPERIMENT_ID or ctx.get('fixed_M_model_run_authorized') is not True:
  raise ContractError('Fixed M has no execution GO. Local preparation is not authorization to run.')

def validate_runtime_versions(lock_path):
 import importlib.metadata
 import sys
 if sys.version_info[:3]!=(3,10,19):raise ContractError('Historical Python3.10.19 runtime required.')
 rows=[]
 for line in lock_path.read_text().splitlines():
  if not line or line.startswith('#'):continue
  name,version=line.split('==',1)
  actual=importlib.metadata.version(name)
  if actual!=version:raise ContractError('Historical runtime distribution differs: '+name)
  rows.append({'name':name,'version':actual})
 return rows

def validate_initial_report(report,expected):
 for key in ('initial_state_sha256','trainable_names_sha256','state_tensor_bytes','optimizer_initial_state_empty'):
  if report.get(key)!=expected[key]:raise ContractError('Historical A initialization/control mismatch: '+key)

def expected_optimizer_values(name):
 # Official DefaultOptimWrapperConstructor: first longest matching custom key.
 for key,lr_mult,decay_mult in [('absolute_pos_embed',1.0,0.0),('backbone',0.1,1.0)]:
  if key in name:return 1e-4*lr_mult,1e-4*decay_mult
 return 1e-4,1e-4

def validate_optimizer_groups(model,optimizer):
 by_id={id(p):n for n,p in model.named_parameters()}
 rows=[]
 for group in optimizer.param_groups:
  if tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group.get('amsgrad',False):raise ContractError('AdamW defaults differ from pinned historical implementation.')
  for parameter in group['params']:
   name=by_id[id(parameter)];lr,decay=expected_optimizer_values(name)
   if group['lr']!=lr or group['weight_decay']!=decay:raise ContractError('Effective optimizer group differs: '+name)
   rows.append({'name':name,'lr':group['lr'],'weight_decay':group['weight_decay'],'requires_grad':parameter.requires_grad})
 return rows

def validate_first_update_rng(record,expected):
 if record['cuda_rng_before_update']!=expected['first_update_cuda_rng_sha256']:raise ContractError('First CUDA RNG differs from historical A.')
 if record['steps'][0]['rng_before']['torch_cpu']!=expected['first_pipeline_torch_CPU_rng_sha256']:raise ContractError('First pipeline CPU Torch RNG differs from historical A.')
 # NumPy after M is intentionally different; no M/A geometry or exposure equality gate.
