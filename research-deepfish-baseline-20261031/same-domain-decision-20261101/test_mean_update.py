import copy,hashlib,json,pathlib,torch
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mean_update import source_mean_update
P=pathlib.Path(__file__).resolve().parent
torch.set_num_threads(4);torch.set_num_interop_threads(1)
class Fixture(BaseModel):
 def __init__(self):
  super().__init__();self.layer=torch.nn.Linear(3,1,dtype=torch.float64)
 def forward(self,inputs,data_samples=None,mode='loss'):
  y=self.layer(inputs)
  return {'loss_main':((y-data_samples)**2).mean(),'loss_aux':[y.square().mean()*.17,y.mean()*.03]}
def build():
 torch.manual_seed(91);m=Fixture();o=OptimWrapper(torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.0001),clip_grad={'max_norm':.1,'norm_type':2});return m,o
def samples(k):
 g=torch.Generator().manual_seed(47)
 return [{'inputs':torch.randn(2,3,generator=g,dtype=torch.float64),'data_samples':torch.randn(2,1,generator=g,dtype=torch.float64)} for _ in range(k)]
results=[]
for k in [1,2,4,5,8]:
 model,optim=build();ref,refoptim=build();batches=samples(k);calls={'clip':0,'optimizer':0,'zero':0,'boundary':0};grad={}
 for attr,key in [('_clip_grad','clip'),('step','optimizer'),('zero_grad','zero')]:
  original=getattr(optim,attr)
  def counted(*a,_f=original,_key=key,**kw):calls[_key]+=1;return _f(*a,**kw)
  setattr(optim,attr,counted)
 def inspect():
  grad.update({n:p.grad.clone() for n,p in model.named_parameters()})
 def boundary():calls['boundary']+=1
 out=source_mean_update(model,optim,iter(batches),k,inspect,boundary)
 losses=[ref.parse_losses(ref._run_forward(ref.data_preprocessor(b,True),mode='loss'))[0] for b in batches]
 target=torch.stack(losses).mean();assert abs(out['mean_loss']-float(target))<1e-12
 refoptim.backward(target)
 for name,p in ref.named_parameters():torch.testing.assert_close(grad[name],p.grad,rtol=1e-12,atol=1e-12)
 refoptim.step();refoptim.zero_grad()
 for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
 assert calls=={'clip':1,'optimizer':1,'zero':1,'boundary':1}
 if k==1:
  standard,so=build();standard.train_step(batches[0],so)
  for a,b in zip(model.parameters(),standard.parameters()):assert torch.equal(a,b)
 results.append({'k':k,'mean_loss_and_unclipped_grad_match_explicit_mean':True,'updated_parameters_match':True,'counts':calls,'k1_official_train_step_bitwise_equal':True if k==1 else None})
ledger=json.loads((P/'C0_FROZEN_SOURCE_LEDGER.json').read_text());assert [r['update'] for r in ledger]==list(range(1,201));assert sum(r['k'] for r in ledger)==659;assert all(r['k']==len(r['source_image_ids'])==len(r['sources_in_stored_order']) for r in ledger)
for r in ledger:assert [s['image_id'] for s in r['sources_in_stored_order']]==r['source_image_ids']
# All200 update-level call counts with every frozen multiset, using only synthetic data.
m,o=build();counts={'clip':0,'step':0,'boundary':0};clip=o._clip_grad;step=o.step
def counted_clip():counts['clip']+=1;return clip()
def counted_step(**kw):counts['step']+=1;return step(**kw)
def boundary():counts['boundary']+=1
o._clip_grad=counted_clip;o.step=counted_step
for r in ledger:source_mean_update(m,o,iter(samples(r['k'])),r['k'],schedule_boundary=boundary)
assert counts=={'clip':200,'step':200,'boundary':200};assert all(float(v['step'])==200 for v in o.optimizer.state.values());assert o._inner_count==659
report={'status':'PASS','synthetic_only':True,'fixture_cases':results,'full_ledger_occurrences':659,'full_ledger_updates':200,'all200_call_counts':counts,'actual_scheduler_objects':0,'constant_LR_boundary_checks':200,'source_ledger_sha256':hashlib.sha256((P/'C0_FROZEN_SOURCE_LEDGER.json').read_bytes()).hexdigest()}
(P/'TEST_RESULTS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','cases':len(results),'full_counts':counts,'source_backward_calls':659}))
