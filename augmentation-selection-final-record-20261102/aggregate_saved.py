"""Final evidence-only synthesis. Standard library only; no feature/model/fit calls."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib,json,csv,math,itertools,datetime,time
ROOT=Path(__file__).resolve().parent
PROJECT=_portable_project_path(__file__, '')
START='2026-11-02T06:54:48.315461+00:00'
DIRS={
'pilot':'augmentation-selection-pilot-20261102',
'deit':'augmentation-selection-deit-20261102',
'synthesis':'augmentation-selection-synthesis-20261102',
'preflight':'augmentation-selection-cifar100-preflight-20261102',
'cifar':'augmentation-selection-cifar100-run-20261102'}
EXPECTED={
'pilot':'68c2a6c48374b6064fe2b2784d00970c3dbceb8cc4a29f29a827e6d022b101bc',
'deit':'eb1088df3f80b7f9fdabb600e166ba163475161586a038f75ca6402923ed7276',
'synthesis':'980d70e169bc8751459bc9a802e367fba9da7b02f617bbab47cf4474d17980d9',
'preflight':'7198fbf0ec29a38c21dcba60b35128851b3f318303919af10eca2fc5c7a3b8d4',
'cifar':'3f1d4a75015791e78ecf0f0697355a3aba0f0e3b31b0d3f3dfbc3a9bc6f0f6ee'}
POL=['clean','weak','ra'];ALIAS={'clean':'S','weak':'W','ra':'R'}
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def close(a,b):assert math.isclose(a,b,abs_tol=1e-9,rel_tol=0),(a,b)
def winners(a):return [j for j,v in enumerate(a) if v==max(a)]
def dist(a):return {'all12':a,'mean':sum(a)/len(a),'min':min(a),'max':max(a),'positive':sum(x>0 for x in a),'equal':sum(x==0 for x in a),'negative':sum(x<0 for x in a)}
verified={};receipts={}
for tag,name in DIRS.items():
 p=PROJECT/name;assert sha(p/'MANIFEST_SHA256.json')==EXPECTED[tag]
 mf=read(p/'MANIFEST_SHA256.json');entries=mf['files']
 if isinstance(entries,dict):entries=[{'path':k,**v} if isinstance(v,dict) else {'path':k,'sha256':v} for k,v in entries.items()]
 for x in entries:
  assert sha(p/x['path'])==x['sha256'],(tag,x['path'])
  if 'bytes' in x:assert (p/x['path']).stat().st_size==x['bytes']
 verified[tag]={'directory':name,'manifest_sha256':EXPECTED[tag],'manifest_files_checked':len(entries),'task_receipt_sha256':sha(p/'TASK_RECEIPT.json')}
 receipts[tag]=read(p/'TASK_RECEIPT.json')
a=read(PROJECT/DIRS['synthesis']/'SYNTHESIS.json');b=read(PROJECT/DIRS['cifar']/'RESULTS.json')
rows=[]
for dataset,originals,N,K in [('EuroSAT',a['all_48_settings'],1000,10),('CIFAR-100',b['all48_settings'],9990,100)]:
 assert len(originals)==48
 for r in originals:
  cv=r['cv_correct'];full=r['full_query_correct'];fold=r['fold_query_correct'];sel=POL.index(r['selected']);cvset=winners(cv);fullset=winners(full);fs=[sum(z) for z in fold]
  assert sel==cvset[0] and cv==[sum(z) for z in r['cv_fold_correct']]
  acc=[100*x/N for x in full];dw=100*(full[sel]-full[1])/N;dr=100*(full[sel]-full[2])/N;reg=100*(max(full)-full[sel])/N
  close(r['selected_minus_weak_pp'],dw);close(r['selected_minus_ra_pp'],dr);close(r['regret_pp'],reg)
  wrong=sel not in fullset;same=wrong and any(fs[sel]<fs[i] for i in fullset)
  if dataset=='EuroSAT':
   assert wrong==r['selected_wrong_full'] and (same==(wrong and r['fold_inferior_to_any_full_optimum']))
  else:assert wrong==r['wrong_full'] and same==(wrong and r['same_pair_fold_inferior'])
  rows.append({'dataset':dataset,'encoder':r['encoder'],'B':r['B'],'episode':r['episode'],'classes':K,'total_support_labels':K*r['B'],'cv_validation_per_fold':K*r['B']//4,'query_n':N,'cv_correct':cv,'cv_fold_correct':r['cv_fold_correct'],'fold_query_correct':fold,'full_query_correct':full,'full_accuracy':acc,'selected_index':sel,'selected':r['selected'],'cv_best_indices':cvset,'fold_query_best_indices':winners(fs),'full_query_best_indices':fullset,'selected_accuracy':acc[sel],'selected_minus_weak_pp':dw,'selected_minus_ra_pp':dr,'regret_pp':reg,'wrong_full':wrong,'same_pair_fold_inferior_wrong':same})
assert len(rows)==96 and len({(x['dataset'],x['encoder'],x['B'],x['episode']) for x in rows})==96
summaries={}
for dataset in ['EuroSAT','CIFAR-100']:
 for model in ['dinov2_small','deit_tiny']:
  for B in [8,16]:
   key=f'{dataset}:{model}:B{B}';rr=sorted([x for x in rows if (x['dataset'],x['encoder'],x['B'])==(dataset,model,B)],key=lambda x:x['episode'])
   assert [x['episode'] for x in rr]==list(range(12));N=rr[0]['query_n'];den=12*N;scale=100/den
   fixed=[sum(x['full_query_correct'][j] for x in rr) for j in range(3)];oracle=sum(max(x['full_query_correct']) for x in rr);selected=sum(x['full_query_correct'][x['selected_index']] for x in rr)
   H=oracle-max(fixed);C=oracle-selected;assert C-H==max(fixed)-selected
   eligible=[x['cv_best_indices'] for x in rr];assignments=[]
   for combo in itertools.product(*eligible):
    q=sum(x['full_query_correct'][j] for x,j in zip(rr,combo));assignments.append({'choice_indices':list(combo),'mean_accuracy':q*scale,'regret_pp':(oracle-q)*scale,'minus_weak_pp':(q-fixed[1])*scale,'minus_ra_pp':(q-fixed[2])*scale})
   low=min(x['regret_pp'] for x in assignments);high=max(x['regret_pp'] for x in assignments)
   best_tie=sum(max(x['full_query_correct'][j] for j in x['cv_best_indices']) for x in rr);Cmin=oracle-best_tie
   wrong=sum(x['wrong_full'] for x in rr);same=sum(x['same_pair_fold_inferior_wrong'] for x in rr)
   ss={'dataset':dataset,'encoder':model,'B':B,'query_n':N,'classes':rr[0]['classes'],'fixed_mean':dict(zip(POL,[x*scale for x in fixed])),'best_constant_expost':[POL[j] for j in winners(fixed)],'selected_mean':selected*scale,'selected_counts':{p:sum(x['selected']==p for x in rr) for p in POL},'H_pp':H*scale,'C_pp':C*scale,'CminusH_pp':(C-H)*scale,'H_correct_sum':H,'C_correct_sum':C,'Cmin_correct_sum':Cmin,'C_gt_H':C>H,'Cmin_gt_H':Cmin>H,'cv_tie_episodes':[x['episode'] for x in rr if len(x['cv_best_indices'])>1],'tie_assignment_count':len(assignments),'tie_assignments':assignments,'tie_bounds':{'mean_accuracy':[min(x['mean_accuracy'] for x in assignments),max(x['mean_accuracy'] for x in assignments)],'regret_pp':[low,high],'minus_weak_pp':[min(x['minus_weak_pp'] for x in assignments),max(x['minus_weak_pp'] for x in assignments)],'minus_ra_pp':[min(x['minus_ra_pp'] for x in assignments),max(x['minus_ra_pp'] for x in assignments)]},'wrong_full_count':wrong,'same_pair_fold_inferior_wrong_count':same,'stage_majority':None if wrong==0 else same*2>wrong,'distributions':{k:dist([x[k] for x in rr]) for k in ['selected_minus_weak_pp','selected_minus_ra_pp','regret_pp']}}
   source=(a if dataset=='EuroSAT' else b)['summaries'][f'{model}:B{B}']
   if dataset=='EuroSAT':
    close(ss['H_pp'],source['oracle_space_vs_expost_best_constant_mean_pp']);close(ss['C_pp'],source['distributions']['regret_pp']['mean']);assert same==source['full_wrong_and_same_pair_fold_inferior'];bounds=source['cv_tie_sensitivity']['bounds']['regret_pp']
   else:
    close(ss['H_pp'],source['H_pp']);close(ss['C_pp'],source['C_pp']);assert same==source['same_pair_fold_inferior_wrong_count'];bounds=source['tie_bounds']['mean_regret_bounds']
   for x,y in zip([low,high],bounds):close(x,y)
   for p in POL:close(ss['fixed_mean'][p],source['fixed_mean_accuracy' if dataset=='EuroSAT' else 'fixed_mean'][p])
   summaries[key]=ss
budget={f'{d}:{m}':{'H8':summaries[f'{d}:{m}:B8']['H_pp'],'H16':summaries[f'{d}:{m}:B16']['H_pp'],'decreases':summaries[f'{d}:{m}:B16']['H_correct_sum']<summaries[f'{d}:{m}:B8']['H_correct_sum']} for d in ['EuroSAT','CIFAR-100'] for m in ['dinov2_small','deit_tiny']}
assert len(summaries)==8 and all(s['C_gt_H'] and s['Cmin_gt_H'] for s in summaries.values())
assert sum(len(s['cv_tie_episodes']) for s in summaries.values())==4
assert sum(not s['stage_majority'] for s in summaries.values())==1
assert sum(not x['decreases'] for x in budget.values())==1
sourcefiles={f'{DIRS["synthesis"]}/SYNTHESIS.json':sha(PROJECT/DIRS['synthesis']/'SYNTHESIS.json'),f'{DIRS["cifar"]}/RESULTS.json':sha(PROJECT/DIRS['cifar']/'RESULTS.json')}
save('COMBINED_RESULTS.json',{'status':'COMPLETE_96_SETTINGS_8_GROUPS','policy_order':POL,'all96_settings':rows,'summaries':summaries,'budget_descriptions':budget,'source_result_sha256':sourcefiles,'all_source_result_agreement':True,'no_new_effects':True,'new_model_forwards':0,'new_head_fits':0,'new_CU':0})
with (ROOT/'ALL_96_SETTINGS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
 for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
flat=[]
for key,s in summaries.items():flat.append({'group':key,**{k:v for k,v in s.items() if k not in ['tie_assignments','distributions']},'CV_minus_W':s['distributions']['selected_minus_weak_pp']['mean'],'CV_minus_R':s['distributions']['selected_minus_ra_pp']['mean']})
with (ROOT/'ALL_8_GROUPS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader()
 for row in flat:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
md=['# 全部96个设置','','S/W/R=标准/弱增强/弱增强后RandAugment；每一行都是已有冻结结果。没有跨数据集合并准确率。','', '|数据集|编码器|B|episode|S|W|R|CV选择|CV−W|CV−R|regret|','|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|']
for x in rows:
 z=x['full_accuracy'];md.append(f"|{x['dataset']}|{x['encoder']}|{x['B']}|{x['episode']}|{z[0]:.4f}|{z[1]:.4f}|{z[2]:.4f}|{ALIAS[x['selected']]}|{x['selected_minus_weak_pp']:+.4f}|{x['selected_minus_ra_pp']:+.4f}|{x['regret_pp']:.4f}|")
(ROOT/'ALL_96_SETTINGS.md').write_text('\n'.join(md)+'\n')
life=[]
for tag in DIRS:
 t=receipts[tag];key='known_task_processes_exited' if tag in ['pilot','deit','synthesis'] else 'known_benchmark_processes_exited' if tag=='preflight' else 'model_processes_exited'
 assert t[key] is True
 if tag in ['pilot','deit']:assert t['main_exec_session_exit_code']==0
 assert t['new_CU']==0
 life.append({'task':DIRS[tag],'recorded_status':t['status'],'exit_evidence_field':key,'exit_evidence_value':True,'receipt':str(Path(DIRS[tag])/'TASK_RECEIPT.json'),'receipt_sha256':verified[tag]['task_receipt_sha256'],'current_state':'COMPLETE_NO_PENDING_EXECUTION'})
save('SOURCE_INDEX.json',{'verified_archives':verified,'source_result_sha256':sourcefiles,'all_source_files_verified':sum(x['manifest_files_checked'] for x in verified.values()),'source_mutations':0})
save('LIFECYCLE.json',{'status':'RESEARCH_LINE_CLOSED','tasks':life,'running_research_jobs':0,'pending_experiment_commands':[],'scheduled_research_actions_created':0,'fourth_round_started':False,'historical_plan_disposition':{'augmentation-selection-synthesis-20261102/CROSS_DATASET_PLAN_NOT_EXECUTED.md':'HISTORICAL; superseded by authorized completed full-class CIFAR100 protocol, not an unstarted job','augmentation-selection-cifar100-preflight-20261102/PROTOCOL.json':'Historical preflight status retained; all integration/execution completed in cifar100-run archive'},'verification_scope':'Task-local exit receipts and joined child processes; no OS-wide process claim or ps/sysctl query','new_CU':0,'presentation_changes':0})
save('VALIDATION.json',{'status':'PASS','source_manifests_verified':len(verified),'source_manifest_files_verified':sum(x['manifest_files_checked'] for x in verified.values()),'unique_settings':96,'complete_groups':8,'episodes_per_group':12,'raw_integer_counts_match_original_metrics':True,'original_choices_and_ties_preserved':True,'all_H_C_and_tie_bounds_match_sources':True,'algebra_checked_with_integer_counts':True,'cv_tied_settings':4,'stage_counterexample_groups':1,'budget_counterexample_model_dataset_pairs':1,'new_predictions_or_features':0,'new_model_calls':0,'new_training':0,'new_CU':0,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps({'status':'PASS','settings':len(rows),'groups':len(summaries),'source_files_verified':sum(x['manifest_files_checked'] for x in verified.values()),'all_C_gt_H':True,'no_new_model_or_head_runs':True},indent=2))
