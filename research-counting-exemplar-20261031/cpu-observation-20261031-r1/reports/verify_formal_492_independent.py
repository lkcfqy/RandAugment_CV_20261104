from portable_dev.paths import project_path as _portable_project_path
import time
START=time.monotonic()
import csv, hashlib, io, json, math, os, statistics, tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
P=_portable_project_path(__file__, 'research-counting-exemplar-20261031/cpu-observation-20261031-r1')
A=P/'formal/artifacts'; O=P/'formal/output'; R=P/'reports'
EXPECTED='649a171fb5bd1d659aa43675e072024ad6c5aa7fba75b0efefa70d481517a79b'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def exact(v):return [v.numerator,v.denominator]
def atom(p,data):
 if isinstance(data,dict):data=json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
 if isinstance(data,str):data=data.encode()
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:
  tmp=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def need(ok,msg):
 if not ok:raise AssertionError(msg)
controller=read(P/'formal/controller_report.json')
need(controller['status']=='COMPLETE_492_PENDING_BOX_REVIEW' and controller['returncode']==0,'terminal complete status')
need(sha(P/'FREEZE.json')==EXPECTED,'freeze SHA')
freeze=read(P/'FREEZE.json'); frozen_n=0
for group in ('files','referenced_assets'):
 for rel,item in freeze[group].items():
  p=P/rel;need(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'frozen asset: '+rel);frozen_n+=1
index=read(A/'artifact_index.json')['files']
for rel,item in index.items():
 for root in (A,O):
  p=root/rel;need(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'artifact: '+str(p))
need((A/'artifact_index.json').read_bytes()==(O/'artifact_index.json').read_bytes(),'index mirror')
completion=read(A/'completion.json');need(completion['complete'] is True and completion['formal_completed']==492,'completion')
e=read(A/'engineering/status.json');c=e['counters']
need(all(c[k]==492 for k in ('formal_attempts','formal_completed','root_forward_entered','root_forward_reserved','root_forward_returned')),'forward counters')
need(all(c[k]==0 for k in ('GPU_operation_attempts','network_attempts','pickle_attempts','training_steps')),'resource counters')
need(e['strict_loading']['strict'] and e['strict_loading']['all_values_equal'] and not e['strict_loading']['missing_keys'] and not e['strict_loading']['unexpected_keys'],'strict loading')
need(sha(A/'engineering/status.json')==controller['engineering_sha256'],'engineering binding')
queries=read(P/'science/queries_frozen.json')['queries'];labels=read(P/'science/labels_frozen.json')['pairs'];selected=read(P/'science/selection_frozen.json')['images']
need(len(queries)==492 and len(labels)==47 and len(selected)==94,'frozen denominator')
for saved,original in [('queries','queries_frozen.json'),('labels','labels_frozen.json'),('selection','selection_frozen.json')]:
 need(read(A/'inputs'/f'{saved}.json')==read(P/'science'/original),'saved input '+saved)
need(read(A/'inputs/protocol.json')==read(P/'execution_contract.json'),'current execution contract')
qmap={q['query_id']:q for q in queries};smap={s['filename']:s for s in selected};lmap={p['pair_id']:p for p in labels}
paths=sorted((A/'results').glob('*.json'));need([p.name for p in paths]==[f'{i:04d}.json' for i in range(492)],'contiguous result set')
rows=[read(p) for p in paths];need(len({r['query_id'] for r in rows})==492 and {r['query_id'] for r in rows}==set(qmap),'exact unique query IDs')
seen_images=set();grouped={};durations={m:[] for m in ('F','C','V')};saturated=[];predicted_boxes=0
for n,(row,q) in enumerate(zip(rows,queries)):
 need(row['query_id']==q['query_id'] and row['query_sha256']==q['query_sha256'] and row['input_contract']==q,'query order/input '+str(n))
 payload={k:v for k,v in q.items() if k!='query_sha256'}
 need(hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==q['query_sha256'],'query digest')
 src=smap[q['filename']];lab=lmap[q['pair_id']];target=q['target'];other='B' if target=='A' else 'A';mode=q['mode']
 need(row['original_annotation_identity']==src and all(row[k]==q[k] for k in ('pair_id','filename','mode','target')),'annotation identity')
 need(lab['eligible'] and mode in ('F','C','V') and target in ('A','B'),'roster/mode')
 need(q['exemplar_boxes']==src[target+'_boxes_xyxy'] and q['negative_exemplar_boxes']==[src[other+'_boxes_xyxy']] and len(q['exemplar_boxes'])==3 and len(q['negative_exemplar_boxes'][0])==3,'six fixed boxes')
 txt=lab['frozen_input_texts'];need(q['text']==(txt['F_'+target] if mode=='F' else txt['C'] if mode=='C' else '') and q['negative_texts']==[txt['F_'+other] if mode=='F' else ''],'FCV native texts')
 truth=next(x for x in lab['selected_images'] if x['filename']==q['filename'])
 need(truth['A_count']==src['A_count'] and truth['B_count']==src['B_count'] and q['target_count_from_official_annotation']==truth[target+'_count'] and q['negative_count_from_official_annotation']==truth[other+'_count'],'frozen truth')
 if q['filename'] not in seen_images:
  need(sha(P/'science/images'/q['filename'])==q['image_sha256'],'image bytes');seen_images.add(q['filename'])
 pred=row['prediction'];count=pred['count'];need(isinstance(count,int) and not isinstance(count,bool) and 0<=count<=900,'pred count')
 need(count==len(pred['boxes'])==len(pred['scores']),'prediction lengths')
 need(all(len(b)==4 and all(math.isfinite(v) for v in b) for b in pred['boxes']) and all(math.isfinite(v) and v>.23 for v in pred['scores']),'finite saved predictions/confidence')
 v=row['validity'];need(all(v[k] for k in ('active_logits_finite','coordinates_finite','padding_exactly_negative_infinity','sigmoid_scores_finite')),'native finite/masked values')
 need(row['box_format']=='normalized_cx_cy_w_h' and row['saturated_ge_800']==(count>=800),'box/saturation identity')
 attempt=read(A/'attempts'/f'{n+1:04d}.json');need(attempt['status']=='COMPLETE' and attempt['label']==q['query_id'] and attempt['result_path']==f'results/{n:04d}.json','attempt binding')
 grouped.setdefault(q['filename'],{}).setdefault(mode,{})[target]=count;durations[mode].append(row['complete_query_seconds_before_persist']);predicted_boxes+=count
 if count>=800:saturated.append(q['query_id'])
need(len(seen_images)==82,'exact eligible image count')
summary=read(A/'science/summary.json');remote={p['pair_id']:p for p in summary['pairs']}
pair_rows=[];diffs=[];mode_means={m:[] for m in ('F','C','V')};qualifying=[]
for lab in labels:
 pid=lab['pair_id'];entry={'pair_id':pid,'status':'EVALUATED' if lab['eligible'] else 'NA','reason':lab['reason'],'frozen_input_texts':lab['frozen_input_texts'],'review_note_zh':lab.get('review_note_zh'),'images':lab['selected_images']}
 if not lab['eligible']:
  need(remote[pid]['status']=='NA' and remote[pid]['mean_error'] is None and remote[pid]['reason']==lab['reason'],'NA retained')
  entry.update(mean_error=None,mean_error_exact=None,qualifies=None);pair_rows.append(entry);continue
 per_mode={m:[] for m in ('F','C','V')};images=[]
 for im in lab['selected_images']:
  truthA,truthB=im['A_count'],im['B_count'];need(truthA+truthB>0,'positive denominator');preds=grouped[im['filename']];errors={}
  for m in per_mode:
   need(set(preds[m])=={'A','B'},'both targets');err=Fraction(abs(preds[m]['A']-truthA)+abs(preds[m]['B']-truthB),truthA+truthB);per_mode[m].append(err);errors[m]=exact(err)
  images.append({'filename':im['filename'],'truth':{'A':truthA,'B':truthB},'prediction_counts':preds,'error_exact':errors})
 means={m:sum(vals,Fraction(0))/2 for m,vals in per_mode.items()};difference=min(means['C'],means['V'])-means['F'];diffs.append(difference)
 qualifies=means['F']<=Fraction(1,10) and means['C']>=Fraction(3,10) and means['V']>=Fraction(3,10)
 if qualifies:qualifying.append(pid)
 for m in means:mode_means[m].append(means[m])
 need(remote[pid]['mean_error_exact']=={m:exact(x) for m,x in means.items()} and remote[pid]['qualifies']==qualifies and remote[pid]['oracle_minus_F_exact']==exact(difference),'independent pair metrics')
 entry.update(images=images,mean_error={m:float(x) for m,x in means.items()},mean_error_exact={m:exact(x) for m,x in means.items()},oracle_minus_F=float(difference),qualifies=qualifies);pair_rows.append(entry)
need(len(diffs)==41 and len(pair_rows)==47,'all pairs')
G=sum(diffs,Fraction(0))/41;gateG=G>=Fraction(1,10);gateN=len(qualifying)>=10
need(summary['G_exact']==exact(G) and summary['qualifying_pairs']==len(qualifying) and summary['gate_G']==gateG and summary['gate_pair_count']==gateN and summary['numeric_investment_gate']==(gateG and gateN),'independent exact gates')
need(summary['saturated_query_ids']==saturated,'saturation set')
processes={}
for name,pid in [('controller',23941),('worker',23943),('caffeinate',23944)]:
 try:os.kill(pid,0);state='alive'
 except ProcessLookupError:state='exited'
 processes[name]={'pid':pid,'state':state};need(state=='exited','process must exit: '+name)
mode_average={m:float(sum(v,Fraction(0))/41) for m,v in mode_means.items()}
report={'status':'COMPLETE_492_INDEPENDENTLY_VERIFIED_GATES_NOT_MET' if not(gateG and gateN) else 'COMPLETE_492_NUMERIC_GATES_MET_REQUIRES_BOX_REVIEW','verified_at_UTC':datetime.now(timezone.utc).isoformat(),'freeze_sha256':EXPECTED,'frozen_files_and_assets_verified':frozen_n,'indexed_artifacts_verified_in_each_of_two_copies':len(index),'unique_queries':492,'eligible_pairs':41,'eligible_images':82,'all_reported_pairs':47,'NA_pairs':6,'mean_pair_error':mode_average,'G':float(G),'G_exact':exact(G),'G_threshold':.1,'G_gate':gateG,'qualifying_pair_count':len(qualifying),'qualifying_pair_required':10,'qualifying_pair_ids':qualifying,'pair_count_gate':gateN,'numeric_investment_gate':gateG and gateN,'predicted_boxes_preserved':predicted_boxes,'saturated_query_ids':saturated,'independent_metric_implementation':'This script recomputes from frozen labels using Fraction; does not import production metrics.py.','metric_formula':'E(image,mode)=(abs(pred_A-truth_A)+abs(pred_B-truth_B))/(truth_A+truth_B); each pair=mean(two images); G=mean_41(min(pair_C,pair_V)-pair_F). Count gate: F<=.10 and C>=.30 and V>=.30, at least 10 pairs.','oracle_envelope_uses_GT':True,'deployable_selector':False,'points_used_as_localization_GT':False,'qualitative_box_review':'NOT_TRIGGERED_BY_JOINT_NUMERIC_GATE; all predicted boxes retained; no correct-target-selection claim made.','scope':'Frozen CountGD++ HF/app basic path on Mac CPU FP32, threads2/1, .23,800/max1333, no adaptive cropping or pseudo-exemplars; no CUDA/paper/all-method equivalence.','runtime':e['runtime'],'resources':{'new_CU':0,'new_GPU':0,'network_attempts':0,'training_steps':0,'peak_RSS_bytes':controller['peak_RSS_bytes'],'formal_wall_seconds':controller['this_compute_wall_seconds'],'pre_formal_cumulative_seconds':controller['prior_compute_seconds'],'formal_terminal_cumulative_seconds':controller['cumulative_compute_seconds'],'hard_cap_seconds':7200,'actual_loading_gate':e['compute_gate']},'processes':processes,'query_wall_before_final_persistence':{m:{'min':min(v),'median':statistics.median(v),'max':max(v),'sum':sum(v)} for m,v in durations.items()},'pairs':pair_rows,'remaining_blockers':[],'limitations':['Engineering failures on MPS/CUDA are not scientific negative results.','No new contribution established by this fixed observation; no universal absence claim.','Annotations supply counts and exemplar identities; placeholder point lists cannot validate localization.','Numerical gates were not met. No after-effect sample removal, changed names, changed thresholds, or follow-up model run.']}
atom(R/'FORMAL_492_RESULT.json',report)
sio=io.StringIO();writer=csv.writer(sio);writer.writerow(['pair_id','status','F_mean_error','C_mean_error','V_mean_error','oracle_minus_F','qualifies','reason'])
for p in pair_rows:
 m=p['mean_error'];writer.writerow([p['pair_id'],p['status'],m['F'] if m else '',m['C'] if m else '',m['V'] if m else '',p.get('oracle_minus_F',''),p['qualifies'] if m else '',p['reason']])
atom(R/'ALL_47_PAIRS.csv',sio.getvalue())
verification_wall=time.monotonic()-START
ledger={'prior_cumulative_compute_seconds':controller['prior_compute_seconds'],'formal_controller_wall_seconds':controller['this_compute_wall_seconds'],'cumulative_at_formal_exit_seconds':controller['cumulative_compute_seconds'],'post_run_independent_verification_and_report_wall_seconds':verification_wall,'cumulative_including_independent_verification_seconds':controller['cumulative_compute_seconds']+verification_wall,'remaining_of_7200_seconds':7200-controller['cumulative_compute_seconds']-verification_wall,'new_CU':0,'new_GPU':0,'network_calls':0,'note':'Includes frozen file/model hashing, both persisted copies, independent Fraction recomputation, process checks and JSON/CSV fsync; no new model execution.'}
atom(R/'FORMAL_492_BUDGET_LEDGER.json',ledger)
print(json.dumps({'status':report['status'],'G':report['G'],'qualifying_pairs':len(qualifying),'mean_pair_error':mode_average,'verified_files':frozen_n,'indexed_files':len(index),'predicted_boxes_preserved':predicted_boxes,'verification_wall_seconds':verification_wall,'processes':processes,'ledger':ledger},ensure_ascii=False,indent=2))
