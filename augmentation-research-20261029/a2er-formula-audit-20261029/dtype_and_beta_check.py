"""Clarify dtype promotion and mirror-space beta gradient after the main audit."""
import json
from pathlib import Path
import numpy as np
import torch
from formula_audit import FixedVisitHarness,outputs_for

root=Path(__file__).resolve().parent
rows=[]
for dtype in [torch.float16,torch.float32,torch.float64]:
    current=torch.ones((32,2),dtype=dtype)
    stored=torch.zeros((32,2),dtype=torch.float32)
    delta=(current-stored).abs().pow(2).sum(-1)/2
    row={'current_output_dtype':str(dtype),'stored_target_dtype':str(stored.dtype),'actual_residual_dtype':str(delta.dtype),'quantile_dtype':str(torch.quantile(delta,.5).dtype)}
    try:row['direct_quantile_same_input_dtype']=float(torch.quantile(torch.ones(32,dtype=dtype),.5))
    except RuntimeError as e:row['direct_quantile_error']=str(e)
    rows.append(row)
engine=FixedVisitHarness(qold=.5)
h=torch.tensor(outputs_for(np.array([.5]*24+[8.]*8)),dtype=torch.float32,requires_grad=True)
losses=iter([1.,3.])
def loss_func(x,**kwargs):return h.sum()*0.+next(losses)
def der_func(x,**kwargs):return {'der':h[x[:,0].long()]}
loss=engine.engine(loss_func,der_func);loss.backward()
beta_grad=float(engine.engine.beta_real_space.grad)
assert beta_grad==-2.
result={'dtypes':rows,'beta_test':{'fifo_loss':1.,'replay_loss':3.,'beta_real_gradient':beta_grad,'expected':-2.,'parameter_updated':False},'purpose':'post-run clarification of actual precision and detached mirror-space gradient; no changes to frozen main audit or official code'}
(root/'results/dtype-beta-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
