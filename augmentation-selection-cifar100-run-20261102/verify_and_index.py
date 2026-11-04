"""Verify saved arrays and frozen contracts; no model import or new fitting."""
from portable_dev.paths import project_path as _portable_project_path
import os
for v in ['VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[v]='4'
import json,hashlib,math,datetime,csv
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
PROJECT=_portable_project_path(__file__, '');PRE=PROJECT/'augmentation-selection-cifar100-preflight-20261102'
def load(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def near(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=0),(a,b)
models={'dinov2_small':384,'deit_tiny':192};policies=['clean','weak','ra'];short=['S','W','R']
r=load(P/'RESULTS.json');run=load(P/'RUN_RECEIPT.json');limits=load(P/'LIMITS.json');proto=load(PRE/'PROTOCOL.json');ids=load(PRE/'IDENTITIES.json');gate=load(P/'JOINT_QUERY_GATE.json')
assert run['status']=='COMPLETE' and run['model_processes_exited'] and r['status']=='COMPLETE_BOTH_ENCODERS' and len(r['all48_settings'])==48
assert sha(PRE/'PROTOCOL.json')==limits['protocol_sha256'] and sha(PRE/'IDENTITIES.json')==limits['identity_sha256']
manifest=load(PRE/'MANIFEST_SHA256.json')
if 'files' in manifest:manifest=manifest['files']
for entry in manifest:
 assert sha(PRE/entry['path'])==entry['sha256'] and (PRE/entry['path']).stat().st_size==entry['bytes'],entry['path']
freeze=load(P/'CODE_FREEZE.json');fix=load(P/'PREQUERY_NUMERIC_FIX.json')
for name,h in freeze['code'].items():
 if name=='analyze_results.py':
  assert sha(P/'analyze_results.before_integer_comparison.py.txt')==h==fix['old_analysis_sha256']
  assert sha(P/name)==fix['new_analysis_sha256']
 else:assert sha(P/name)==h,name
assert sha(P/'exact_accounting.py')==fix['helper_sha256'] and fix['utc_epoch']<gate['utc_epoch']
plot=load(P/'PLOT_PLAN_PIN.json');assert sha(P/'render_results.py')==plot['script_sha256'] and plot['utc_epoch']<gate['utc_epoch']
assert sha(PROJECT/'research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy')==proto['data_and_identity_sha256']['npy']
for m,v in proto['model_sources'].items():assert sha(PROJECT/v['path'])==v['sha256']
for m,v in proto['source_code_and_environment'].items():
 src=PROJECT/('augmentation-selection-pilot-20261102' if m=='dinov2_small' else 'augmentation-selection-deit-20261102')
 assert sha(src/'common.py')==limits['common_sha256'][m]
 for code in v['environment']['sources'].values():assert sha(Path(code['path']))==code['sha256']
support=ids['support_union'];query=ids['query'];q_y=np.array([x['label'] for x in query]);N=len(query)
assert len(support)==8751 and N==9990 and set(q_y)==set(range(100))
assert {x['id'] for x in support}.isdisjoint(x['id'] for x in query)
assert {x['rgb_hash'] for x in support}.isdisjoint(x['rgb_hash'] for x in query)
for e in ids['episodes']:
 rank=np.array(e['rank_16']);y=np.array(e['labels_16']);fold=np.array(e['fold_16'])
 for B in [8,16]:
  mask=rank<B;assert np.array_equal(np.bincount(y[mask],minlength=100),np.full(100,B))
  for f in range(4):assert np.array_equal(np.bincount(y[mask&(fold==f)],minlength=100),np.full(100,B//4))
receipts=[];stages=[];all_preds=0;all_oof=0
for m,D in models.items():
 d=P/m;fit=load(d/'FIT_SEAL.json');qs=load(d/'QUERY_SEAL.json');heads=load(d/'HEAD_IDENTITIES.json')
 assert sha(d/'FIT_SEAL.json')==gate['fit_seal_sha256'][m] and fit['utc_epoch']<gate['utc_epoch']<qs['utc_epoch']
 assert qs['joint_gate_sha256']==sha(P/'JOINT_QUERY_GATE.json')
 assert fit['model_state_before']==fit['model_state_after']==qs['model_state_before']==qs['model_state_after']
 for seal in [fit,qs]:
  assert seal['protocol_sha256']==limits['protocol_sha256'] and seal['identity_sha256']==limits['identity_sha256']
  for name,h in seal['files'].items():assert sha(d/name)==h
 assert len(fit['choices'])==24 and len(heads)==360
 W=np.load(d/'W.npy',mmap_mode='r',allow_pickle=False);bias=np.load(d/'b.npy',mmap_mode='r',allow_pickle=False)
 assert W.shape==(360,D,100) and bias.shape==(360,100) and W.dtype==bias.dtype==np.float64
 assert np.isfinite(W).all() and np.isfinite(bias).all()
 pred=np.load(d/'QUERY_PREDICTIONS.npy',mmap_mode='r',allow_pickle=False)
 assert pred.shape==(360,N) and pred.dtype==np.uint8 and int(pred.max())<100
 correct=(pred==q_y[None,:]).sum(1);all_preds+=pred.size
 with np.load(d/'CV_PREDICTIONS.npz',allow_pickle=False) as cv:
  assert len(cv['head_index'])==86400;all_oof+=len(cv['head_index'])
  vcorrect={h:int((cv['prediction'][cv['head_index']==h]==cv['label'][cv['head_index']==h]).sum()) for h in set(cv['head_index'].tolist())}
 for row in [x for x in r['all48_settings'] if x['encoder']==m]:
  full=[];fold=[];vc=[]
  for policy in policies:
   hh=[h for h in heads if (h['episode'],h['B'],h['policy'])==(row['episode'],row['B'],policy)]
   full.append(int(correct[next(h['index'] for h in hh if h['fold']==-1)]));ff=[h['index'] for h in hh if h['fold']>=0]
   fold.append(sum(int(correct[h]) for h in ff));vc.append(sum(vcorrect[h] for h in ff))
  assert full==row['full_query_correct'] and vc==row['cv_correct'] and fold==[sum(x) for x in row['fold_query_correct']]
  selected=int(np.argmax(vc));assert selected==row['selected_index'];near(row['regret_pp'],100*(max(full)-full[selected])/N)
  stages.append({'encoder':m,'B':row['B'],'episode':row['episode'],'cv_correct':vc,'fold_query_correct_summed':fold,'full_query_correct':full,'selected':row['selected'],'wrong_full':row['wrong_full'],'same_pair_fold_inferior':row['same_pair_fold_inferior']})
 for B in [8,16]:
  rr=[x for x in stages if x['encoder']==m and x['B']==B];s=r['summaries'][f'{m}:B{B}']
  oracle=sum(max(x['full_query_correct']) for x in rr);constant=max(sum(x['full_query_correct'][j] for x in rr) for j in range(3))
  selected=sum(x['full_query_correct'][int(np.argmax(x['cv_correct']))] for x in rr)
  mincost=oracle-sum(max(x['full_query_correct'][j] for j,v in enumerate(x['cv_correct']) if v==max(x['cv_correct'])) for x in rr)
  maxcost=oracle-sum(min(x['full_query_correct'][j] for j,v in enumerate(x['cv_correct']) if v==max(x['cv_correct'])) for x in rr)
  near(s['H_pp'],100*(oracle-constant)/(12*N));near(s['C_pp'],100*(oracle-selected)/(12*N))
  near(s['tie_bounds']['mean_regret_bounds'][0],100*mincost/(12*N));near(s['tie_bounds']['mean_regret_bounds'][1],100*maxcost/(12*N))
  assert s['C_gt_H']==(oracle-selected>oracle-constant) and s['Cmin_gt_H']==(mincost>oracle-constant)
 receipts += [load(d/'FIT_RECEIPT.json'),load(d/'QUERY_RECEIPT.json')]
assert all_preds==7192800 and all_oof==172800
assert sum(x['head_fits'] for x in receipts)==720 and sum(x['image_forwards'] for x in receipts)==177498
assert all(x['all_argmax_matches'] and x['new_forwards']==x['new_fits']==0 for x in r['checks'].values())
assert not list(P.rglob('*.inprogress*')) and not (P/'STOP_RECEIPT.json').exists()
assert all((P/f'{m}_{phase}.stderr.log').stat().st_size==0 for m in models for phase in ['fit','query'])
for name,expected in [('ALL_HEAD_METRICS.csv',720),('ALL_FOLD_COUNTS.csv',576)]:
 with (P/name).open() as f:assert len(list(csv.DictReader(f)))==expected

def rank_string(v):return ' > '.join(' = '.join(short[j] for j,x in enumerate(v) if x==val) for val in sorted(set(v),reverse=True))
lines=['# 全部48设置的三阶段排序','','S=标准预处理，W=弱增强，R=弱增强后RandAugment。CV总正确数分母100B；折模型查询总正确数分母4×9990；全标签模型分母9990。相等以`=`保留；不新增选择规则。','折查询和全标签查询均是封存选择后的事后解释；训练集规模和图像改变，不能纯因果拆分验证噪声与重训练效应。','', '|编码器|B|episode|CV排序 / 正确数S,W,R|折查询排序 / 正确数S,W,R|全标签排序 / 正确数S,W,R|原选择|全标签错选|同对折阶段已差|','|:---|---:|---:|:---|:---|:---|:---|:---|:---|']
for x in stages:
 def cell(k):return rank_string(x[k])+' / '+','.join(map(str,x[k]))
 lines.append('|'+ '|'.join([x['encoder'],str(x['B']),str(x['episode']),cell('cv_correct'),cell('fold_query_correct_summed'),cell('full_query_correct'),x['selected'],str(x['wrong_full']),str(x['same_pair_fold_inferior'])])+'|')
(P/'STAGE_RANKINGS.md').write_text('\n'.join(lines)+'\n');save('STAGE_RANKINGS.json',stages)
save('FINAL_VERIFICATION.json',{'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_and_identity_unchanged':True,'preflight_manifest_files_verified':len(manifest),'core_code_unchanged_except_documented_prequery_analysis_fix':True,'data_weights_and_implementation_hashes_verified':True,'all100_classes_nested_and_fold_counts_verified':True,'support_query_identity_and_rgb_disjoint':True,'both_fit_seals_precede_joint_gate_and_both_queries_follow_it':True,'formal_image_forwards':177498,'formal_head_fits':720,'choices':48,'query_decisions_verified_from_saved_predictions':all_preds,'oof_decisions_verified':all_oof,'all_query_and_oof_argmax_reconstruction_passed_in_joint_analysis':True,'group_H_C_tie_bounds_independently_recomputed':True,'stage_count_records':len(stages),'metric_rows':720,'fold_rows':576,'incomplete_files':0,'model_process_exit_evidence':'supervisor joined all4 workers and analysis, exit_code0; no process-wide inspection used','peak_worker_rss_bytes':max(x['peak_rss_bytes'] for x in receipts),'worker_wall_seconds_sum':sum(x['wall_seconds'] for x in receipts),'new_forwards':0,'new_fits':0,'new_CU':0})
print('PASS: hashes, identities, all results, chronology and resource counts',flush=True)
