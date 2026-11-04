"""Offline arithmetic from archived r5 receipts; no runtime, network, or model use."""
from portable_dev.paths import project_path as _portable_project_path
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
RESEARCH=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
RUN=RESEARCH/'recovery-v5-20261031'
def read(rel):return json.loads((RUN/rel).read_text())
c=read('gpu-run/controller_report.json');ctx=read('gpu-run/artifacts/run_context.json');result=read('reports/RESULT.json')
prior_cu=result['resources']['cumulative_CU_conservative_upper_ledger'];prior_gpu=result['resources']['cumulative_allocated_GPU_seconds_upper'];rate=c['live_rate']
assert abs(prior_cu-1.1001052668)<1e-9 and abs(prior_gpu-3225.2736)<1e-3 and rate==1.15
start=ctx['allocation_started_epoch'];last=start;phases=[]
for r in c['stage_receipts']:
 phases.append({'stage':r['phase'],'start_epoch':last,'end_epoch':r['completed_epoch'],'seconds':r['completed_epoch']-last});last=r['completed_epoch']
phases.append({'stage':'release_observed','start_epoch':last,'end_epoch':c['allocation_release_observed_epoch'],'seconds':c['allocation_release_observed_epoch']-last})
total=sum(x['seconds'] for x in phases);assert abs(total-c['allocation_wall_seconds_upper'])<1e-6
common=sum(x['seconds'] for x in phases if x['stage'] in ['preflight_A','preflight_B'])
pair=sum(x['seconds'] for x in phases if x['stage'] in ['formal_A','formal_B','evaluate_A','evaluate_B'])
tail=sum(x['seconds'] for x in phases if x['stage'] in ['final','release_observed'])
assert abs(common+pair+tail-total)<1e-6
remaining_cu=2-prior_cu;remaining_gpu=7200-prior_gpu;cu_seconds=remaining_cu/rate*3600
base={'one_new_cold_runtime':total,'two_pairs_shared_runtime':common+2*pair+tail,'two_pairs_separate_cold_runtimes':2*total}
def scenario(seconds):
 cu=seconds*rate/3600
 return {'additional_seconds':seconds,'additional_minutes':seconds/60,'additional_CU_at_historical_rate':cu,'cumulative_CU':prior_cu+cu,'cumulative_GPU_seconds':prior_gpu+seconds,'fits_CU':cu<=remaining_cu,'fits_GPU_time':seconds<=remaining_gpu}
estimates={name:{'r5_based_without_extra_margin':scenario(t),'planning_25percent_plus180seconds':scenario(1.25*t+180)} for name,t in base.items()}
checkpoint_bytes=sum(x['bytes'] for x in result['final_checkpoints']);artifact_bytes=result['persistence']['total_final_artifact_bytes']
current_bytes=sum(p.stat().st_size for p in RESEARCH.rglob('*') if p.is_file())
source_paths=['gpu-run/controller_report.json','gpu-run/artifacts/run_context.json','reports/RESULT.json','gpu-run/formal_A_download_receipt.json','gpu-run/formal_B_download_receipt.json','gpu-run/artifacts/formal_A.stdout.log','gpu-run/artifacts/formal_B.stdout.log','payload/config_A.json','payload/config_B.json','gpu-run/persistence_probe.json']
data={'status':'COST_ESTIMATE_ONLY_NO_EXECUTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exploratory_after_first_result':True,'original_seed':20261031,'proposed_additional_seed_order':[20261101,20261102],'recommended_additional_paired_seeds_by_compute_budget':1,'formal_U_fixed':200,'per_seed_arms':['A_factor1.0','B_factor0.8'],'prior_CU':prior_cu,'prior_GPU_seconds':prior_gpu,'remaining_CU':remaining_cu,'remaining_GPU_seconds':remaining_gpu,'historical_T4_CU_per_hour':rate,'remaining_seconds_limited_by_CU_at_historical_rate':cu_seconds,'stage_timings':phases,'r5_total_seconds':total,'r5_CU_increment':c['cumulative_CU_upper']-c['prior_cu'],'r5_shared_prefix_seconds_including_preflights':common,'r5_per_pair_train_persist_evaluate_seconds':pair,'r5_final_persist_release_seconds':tail,'planning_buffer_rule':'1.25 * receipt-based full stage time + 180 seconds reserve; engineering allowance, not a confidence bound; no efficacy input','scenarios':estimates,'two_pairs_optimistic_zero_setup_and_zero_tail_with_same_buffer':scenario(1.25*(2*pair)+180),'maximum_rate_for_one_conservative_pair_CU_per_hour':remaining_cu*3600/(1.25*total+180),'storage':{'current_research_bytes':current_bytes,'current_research_GiB':current_bytes/1024**3,'original_directory_cap_bytes':4*1024**3,'one_pair_checkpoint_bytes':checkpoint_bytes,'one_pair_full_artifact_reference_bytes':artifact_bytes,'project_after_one_pair_reference_bytes':current_bytes+artifact_bytes,'project_after_one_pair_reference_GiB':(current_bytes+artifact_bytes)/1024**3,'excess_over_original_cap_one_pair_bytes':max(0,current_bytes+artifact_bytes-4*1024**3),'project_after_two_pairs_reference_bytes':current_bytes+2*artifact_bytes,'storage_policy_resolution_needed':current_bytes+artifact_bytes>4*1024**3},'new_GPU_allocations':0,'new_model_calls':0,'new_uploads':0,'new_installs':0,'new_CU':0,'scientific_config_files_changed':False,'source_sha256':{p:hashlib.sha256((RUN/p).read_bytes()).hexdigest() for p in source_paths}}
(ROOT/'COST_ESTIMATE.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:data[k] for k in ['remaining_CU','remaining_GPU_seconds','remaining_seconds_limited_by_CU_at_historical_rate','r5_total_seconds','r5_CU_increment','r5_shared_prefix_seconds_including_preflights','r5_per_pair_train_persist_evaluate_seconds','r5_final_persist_release_seconds','scenarios','two_pairs_optimistic_zero_setup_and_zero_tail_with_same_buffer','maximum_rate_for_one_conservative_pair_CU_per_hour','storage']},indent=2))
