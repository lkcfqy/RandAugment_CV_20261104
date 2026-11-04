"""Safe complete checkpoint return receipt; no model construction or evaluation."""
import argparse,hashlib,json,pathlib,sys
BASE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'payload'))
from contracts import ContractError,canonical,verify_exposures
from candidate_control import validate_initial_report,validate_first_update_rng

def verify(root):
 import torch
 torch.set_num_threads(4)
 root=pathlib.Path(root);expected=json.loads((BASE/'payload/reference_A_expected.json').read_text())
 report=json.loads((root/'formal_M.json').read_text())
 pre=json.loads((root/'preflight_M.json').read_text())
 for row in (pre,report):validate_initial_report(row,expected)
 if report['seed']!=20261031 or report['actual_updates']!=200:raise ContractError('Seed/U mismatch')
 trace=json.loads((root/'formal_M_trace.json').read_text())
 if [x['image_id'] for x in trace]!=expected['anchor_order']:raise ContractError('Anchor schedule mismatch')
 verify_exposures([x['image_id'] for x in trace],200);validate_first_update_rng(trace[0],expected)
 ck=report['final_checkpoint'];p=root/ck['path']
 if ck['path']!='final_M_U200.pth' or p.stat().st_size!=ck['bytes']:raise ContractError('Wrong final checkpoint')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 if h.hexdigest()!=ck['sha256']:raise ContractError('Returned checkpoint hash differs')
 state=torch.load(p,map_location='cpu',weights_only=True)
 if not isinstance(state,dict) or len(state)!=908 or not all(isinstance(t,torch.Tensor) for t in state.values()):raise ContractError('Complete tensor-only state required')
 digest=hashlib.sha256();nbytes=0
 for name,t in sorted(state.items()):
  if t.device.type!='cpu' or (t.is_floating_point() and not bool(torch.isfinite(t).all())):raise ContractError('Nonfinite/nonCPU saved tensor')
  data=t.detach().contiguous().numpy().tobytes();digest.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode());digest.update(data);nbytes+=len(data)
 return dict(status='LOCAL_M_CHECKPOINT_VERIFIED',seed=20261031,actual_updates=200,
  checkpoint=ck,load=dict(state_sha256=digest.hexdigest(),tensor_count=908,tensor_bytes=nbytes,weights_only=True,finite=True),
  initial_state_sha256=report['initial_state_sha256'],historical_A_initialization_matched=True,
  historical_anchor_order_matched=True,model_instantiated=False,dev_evaluated=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifacts');p.add_argument('receipt');a=p.parse_args()
 result=verify(a.artifacts)
 from external_watchdog import atomic
 atomic(pathlib.Path(a.receipt),result);print(json.dumps({'status':result['status'],'checkpoint_sha256':result['checkpoint']['sha256']}))
