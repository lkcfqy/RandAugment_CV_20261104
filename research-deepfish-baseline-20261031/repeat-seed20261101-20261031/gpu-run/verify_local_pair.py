"""Local-only safe checkpoint and paired-data audit before dev is released."""
import argparse,hashlib,json,pathlib,sys
BASE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'payload'))
from contracts import ContractError,verify_initializations,verify_paired_traces,verify_exposures,canonical
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def verify_pair(root):
 import torch
 root=pathlib.Path(root);reports=[json.loads((root/f'formal_{a}.json').read_text()) for a in 'AB'];verify_initializations(reports)
 traces=[json.loads((root/f'formal_{a}_trace.json').read_text()) for a in 'AB'];paired=verify_paired_traces(*traces);checks=[]
 for arm,r,trace in zip('AB',reports,traces):
  if r.get('seed')!=20261101 or r['actual_updates']!=200:raise ContractError('Local final pair seed/U mismatch.')
  exposures=verify_exposures([x['image_id'] for x in trace],200)
  recorded=dict(r['exposures'],counts={int(k):v for k,v in r['exposures']['counts'].items()})
  if exposures!=recorded:raise ContractError('Local final exposure receipt mismatch.')
  ck=r['final_checkpoint']
  if ck['path']!=f'final_{arm}_U200.pth':raise ContractError('Unexpected fixed-final path.')
  p=root/ck['path']
  if p.stat().st_size!=ck['bytes'] or sha(p)!=ck['sha256']:raise ContractError('Local final checkpoint hash mismatch.')
  state=torch.load(p,map_location='cpu',weights_only=True)
  if not isinstance(state,dict) or len(state)!=908 or not all(isinstance(x,torch.Tensor) for x in state.values()):raise ContractError('Local final checkpoint is not the expected tensor-only state dict.')
  if not all(bool(torch.isfinite(x).all()) for x in state.values() if x.is_floating_point()):raise ContractError('Local final checkpoint contains nonfinite values.')
  checks.append(dict(arm=arm,bytes=ck['bytes'],sha256=ck['sha256'],tensor_keys=len(state),all_floating_finite=True,weights_only=True,map_location='cpu'));del state
 return dict(status='LOCAL_PAIR_VERIFIED',seed=20261101,formal_updates_per_arm=200,paired_trace_sha256=paired,initial_state_sha256=reports[0]['initial_state_sha256'],checkpoints=checks,model_instantiated=False,dev_evaluated_before_this_receipt=False,torch_version=torch.__version__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifacts');p.add_argument('receipt');a=p.parse_args();r=verify_pair(a.artifacts)
 from external_watchdog import atomic
 atomic(pathlib.Path(a.receipt),r);print(json.dumps({'status':r['status'],'seed':r['seed'],'paired_trace_sha256':r['paired_trace_sha256']}))
