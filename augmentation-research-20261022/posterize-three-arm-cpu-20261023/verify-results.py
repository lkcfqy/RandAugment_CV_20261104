import time
START=time.perf_counter();CPU=time.process_time()
from pathlib import Path
import json,hashlib,math,csv
ROOT=Path.cwd().resolve();OUT=Path(__file__).resolve().parent
CFG=json.loads((OUT/'protocol.json').read_text());BUDGET=json.loads((OUT/'budget.json').read_text())
assert time.monotonic()<BUDGET['deadline_monotonic']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
assert not (OUT/'independent-verification.json').exists()
result=json.loads((OUT/'result.json').read_text());watch=json.loads((OUT/'attempt-001/watchdog.json').read_text());pre=json.loads((OUT/'preflight.json').read_text())
assert result['status']=='all_three_arms_completed' and watch['exit_code']==0 and watch['hard_stop_reason'] is None
assert watch['child_inclusive_wall_seconds']<3600 and result['stage_elapsed']<3600
assert not list(OUT.glob('attempt-002*'))
assert sha(OUT/'trajectories.jsonl')==pre['trace_sha256']
assert sha(OUT/'orders.json')==pre['orders_sha256']
for rel,h in json.loads((OUT/'frozen-delivery-manifest.json').read_text()).items():assert sha(ROOT/rel)==h,rel
prior=ROOT/'augmentation-research-20261022/posterize-cpu-budget-20261023'
old_checks=[]
for rel,h in json.loads((prior/'delivery-manifest.json').read_text()).items():
 assert (str(prior.relative_to(ROOT)) in rel or rel=='CURRENT_CHECKPOINT_20261023_POSTERIZE_CPU_BUDGET.json') and (ROOT/rel).resolve().is_relative_to(ROOT)
 assert sha(ROOT/rel)==h;old_checks.append(rel)
for rel,h in CFG['input_sha256'].items():assert sha(ROOT/rel)==h
assert sha(ROOT/CFG['previous_ledger'])==CFG['previous_ledger_sha256']
rows=json.loads((ROOT/'augmentation-research-20261022/pilot-20261022/fit-manifest.json').read_text())['rows'];assert all(r['candidate_partition'] in ('train','development') for r in rows)
train=[r for r in rows if r['candidate_partition']=='train'];dev=[r for r in rows if r['candidate_partition']=='development'];trainids={r['source_id'] for r in train};devids=[r['source_id'] for r in dev];labels={r['source_id']:r['class_id'] for r in dev};assert len(train)==591 and len(dev)==589
orders=json.loads((OUT/'orders.json').read_text());assert len(orders)==5 and all(len(v)==591 and set(v)==trainids for v in orders)
tr=[json.loads(s) for s in (OUT/'trajectories.jsonl').read_text().splitlines()];assert len(tr)==2955
for epoch in range(5):assert [r['source_id'] for r in tr if r['epoch']==epoch]==orders[epoch]
targetslots=0
for r in tr:
 assert len(r['B'])==len(r['Z'])==len(r['R'])==2
 for i,op in enumerate(r['B']):
  if op['name']=='Posterize':
   targetslots+=1;assert r['Z'][i]=={'name':'Identity','magnitude':0.} and r['R'][i]['name'] in CFG['pool'] and r['R'][i]['name']!='Posterize'
  else:assert r['Z'][i]==r['R'][i]==op
assert targetslots==432 and targetslots==pre['actual_target_slots']
assert len(pre['B_pixel_parity'])==32 and all(x['exact_RGB_equal'] for x in pre['B_pixel_parity'])
byarm={};summaries={};steps={};max_nll_error=0.;checkpoints=[]
for arm in ['B','Z','R']:
 path=OUT/arm;summary=json.loads((path/'summary.json').read_text());init=json.loads((path/'initialization.json').read_text());s=[json.loads(v) for v in (path/'steps.jsonl').read_text().splitlines()]
 assert len(s)==185 and [x['step'] for x in s]==list(range(1,186))
 assert sum(x['examples'] for x in s)==2955
 for epoch in range(1,6):
  sub=s[(epoch-1)*37:epoch*37];assert len(sub)==37 and sum(x['examples'] for x in sub)==591 and all(x['epoch']==epoch for x in sub)
  for j,row in enumerate(sub):assert row['examples']==(15 if j==36 else 16)
  ep=json.loads((path/('epoch%02d.json'%epoch)).read_text());assert ep['order_sha256']==hashlib.sha256('\n'.join(orders[epoch-1]).encode()).hexdigest()
  cp=ROOT/ep['checkpoint']['path'];assert cp.resolve().is_relative_to(OUT) and sha(cp)==ep['checkpoint']['sha256'];checkpoints.append(str(cp.relative_to(ROOT)))
 assert all(math.isfinite(x['mean_CE']) and x['mean_CE']>=0 for x in s)
 for step,row in enumerate(s):
  factor=(step+1)/111 if step<111 else .5*(1+math.cos(math.pi*(step-111)/(1110-111)))
  assert math.isclose(row['adapter_lr'],.0003*factor,rel_tol=1e-13) and math.isclose(row['head_lr'],.001*factor,rel_tol=1e-13)
 pred=[json.loads(v) for v in (path/'development-predictions.jsonl').read_text().splitlines()];assert [x['source_id'] for x in pred]==devids
 for r in pred:
  assert r['class_id']==labels[r['source_id']] and len(r['logits'])==37 and all(math.isfinite(v) for v in r['logits'])
  z=r['logits'];arg=max(range(37),key=z.__getitem__);assert arg==r['prediction'];assert r['correct']==(arg==r['class_id'])
  mx=max(z);nll=mx-z[r['class_id']]+math.log(math.fsum(math.exp(v-mx) for v in z));err=abs(nll-r['NLL']);max_nll_error=max(max_nll_error,err);assert err<1e-10
 count=sum(r['correct'] for r in pred);assert count==summary['dev_correct'] and math.isclose(count/589,summary['dev_accuracy'],abs_tol=1e-14)
 assert math.isclose(math.fsum(r['NLL'] for r in pred)/589,summary['dev_NLL'],abs_tol=1e-12)
 assert summary['prediction_sha256']==sha(path/'development-predictions.jsonl') and init['initial_trainable_sha256']==summary['initial_trainable_sha256'] and init['empty_optimizer']
 assert summary['frozen_before_sha256']==summary['frozen_after_sha256'] and summary['trace_sha256']==pre['trace_sha256']
 byarm[arm]={r['source_id']:r for r in pred};summaries[arm]=summary;steps[arm]=s
assert len({s['initial_trainable_sha256'] for s in summaries.values()})==1 and len({s['frozen_before_sha256'] for s in summaries.values()})==1
assert len({s['final_trainable_sha256'] for s in summaries.values()})==3
for a in ['Z','R']:
 assert [(r['examples'],r['adapter_lr'],r['head_lr']) for r in steps[a]]==[(r['examples'],r['adapter_lr'],r['head_lr']) for r in steps['B']]
patterns={format(i,'03b'):0 for i in range(8)};joint=[]
for sid in devids:
 pattern=''.join(str(int(byarm[a][sid]['correct'])) for a in ['B','Z','R']);patterns[pattern]+=1
 joint.append({'source_id':sid,'class_id':labels[sid],**{a+'_prediction':byarm[a][sid]['prediction'] for a in ['B','Z','R']},**{a+'_correct':int(byarm[a][sid]['correct']) for a in ['B','Z','R']},**{a+'_NLL':byarm[a][sid]['NLL'] for a in ['B','Z','R']},'pattern_BZR':pattern})
assert patterns==result['patterns_BZR']
with (OUT/'paired-development-results.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(joint[0]));writer.writeheader();writer.writerows(joint)
dz=summaries['Z']['dev_correct']-summaries['B']['dev_correct'];dr=summaries['R']['dev_correct']-summaries['B']['dev_correct']
label='positive_observed_reversal' if(dz>=3 and dr<=-3)or(dz<=-3 and dr>=3)else('negative_same_direction_screen' if(dz>=3 and dr>=3)or(dz<=-3 and dr<=-3)else'unresolved_below_direction_threshold')
assert label==result['classification'] and dz==result['gZ_count'] and dr==result['gR_count'] and dr-dz==result['h_count']
assert result['counts']==watch['last_event']['counts']
assert result['counts']['optimizer_updates']==555 and result['counts']['backward_calls']==2220 and result['counts']['training_forward_images']==8865 and result['counts']['dev_forward_images']==1767
pairwise={}
for a,b in [('B','Z'),('B','R'),('Z','R')]:
 pairwise[a+'_'+b]={'both_correct':sum(byarm[a][sid]['correct'] and byarm[b][sid]['correct'] for sid in devids),'a_only_correct':sum(byarm[a][sid]['correct'] and not byarm[b][sid]['correct'] for sid in devids),'b_only_correct':sum(not byarm[a][sid]['correct'] and byarm[b][sid]['correct'] for sid in devids),'both_wrong':sum(not byarm[a][sid]['correct'] and not byarm[b][sid]['correct'] for sid in devids),'prediction_label_disagreements':sum(byarm[a][sid]['prediction']!=byarm[b][sid]['prediction'] for sid in devids)}
verification={'status':'passed','method':'Independent stdlib recomputation from saved logits/labels and frozen metadata; no model rerun','classification':label,'correct_counts':{a:summaries[a]['dev_correct'] for a in ['B','Z','R']},'gZ_count':dz,'gR_count':dr,'h_count':dr-dz,'patterns_BZR':patterns,'pairwise':pairwise,'max_per_image_NLL_recomputation_error':max_nll_error,'all_initial_trainables_identical':True,'all_base_parameters_preserved_by_before_after_hash':True,'all_LR_sequences_and_logical_batch_sizes_identical':True,'saved_checkpoints_verified':checkpoints,'target_slots':targetslots,'prior_stage_manifest_files_unchanged':old_checks,'new_model_forwards':0,'new_backward_calls':0,'new_updates':0,'worker_wall_seconds':time.perf_counter()-START,'worker_cpu_seconds':time.process_time()-CPU,'stage_elapsed_at_verification':time.monotonic()-BUDGET['started_monotonic'],'images_read':0,'sealed_data_read':False}
assert verification['stage_elapsed_at_verification']<3600
dump('independent-verification.json',verification)
print(json.dumps({'status':'passed','correct_counts':verification['correct_counts'],'classification':label,'pairwise':pairwise,'worker_wall_seconds':verification['worker_wall_seconds'],'checkpoints':len(checkpoints)}))
