"""Bounded local paired CPU lifecycle; no install, network, cloud or accelerator calls."""
from portable_dev.paths import project_path as _portable_project_path
import datetime,hashlib,json,os,pathlib,subprocess,sys,time,traceback
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 data=json.dumps(v,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n';temp=p.with_name(p.name+'.partial')
 with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def tree(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def estimate(pre_a,pre_m,elapsed,proto):
 rule=proto['time_rule'];scale=rule['geometry_size_multiplier']
 a=max(x['train_step_seconds']+x['pipeline_seconds'] for x in pre_a['steps'])
 m=max(x['train_step_seconds']+x['pipeline_seconds'] for x in pre_m['steps'])
 training=200*(a+m)*scale
 inference=128*rule['reference_max_CPU_forward_seconds']
 storage=2*rule['checkpoint_save_and_full_verification_allowance_per_arm_seconds']
 overhead=rule['other_initialization_data_logging_and_release_allowance_seconds']
 future=rule['safety_multiplier']*(training+inference+storage+overhead)
 rss=rule['safety_multiplier']*(max(pre_a['peak_RSS_bytes'],pre_m['peak_RSS_bytes'])*scale+700*1024**2)
 return {'A_max_update_and_pipeline_seconds':a,'M_max_update_and_pipeline_seconds':m,'geometry_upper_bound_factor':scale,'training_seconds_before_margin':training,'inference_seconds_before_margin':inference,'save_verify_seconds_before_margin':storage,'other_seconds_before_margin':overhead,'safety_multiplier':rule['safety_multiplier'],'estimated_future_seconds':future,'preflight_elapsed_seconds':elapsed,'estimated_total_seconds':elapsed+future,'conservative_peak_RSS_bytes':rss,'time_pass':elapsed+future<7200,'memory_pass':rss<=12*1024**3}
def eval_prepare():
 e=P/'evaluation';old=json.loads((R/'cpu-cross-input-diagnostic-20261101/RUN_PROTOCOL.json').read_text())
 run={k:old[k] for k in ['original_protocol_sha256','core_runtime_versions','dev_annotation','environment_lock_sha256','official_implementation_sha256','image_order']}
 run.update(authorized_CPU_paired_AM=True,code_sha256={n:sha(e/n) for n in ['formal_cpu.py','diagnostic_contracts.py']},models=[],identity_map={},expected_identities=[],CPU_trained_pair=True)
 for arm in ['A','M']:
  record=json.loads((P/f'formal_{arm}/RESULT.json').read_text());assert record['status']=='TRAINING_COMPLETE' and record['completed_updates']==200
  ck=record['checkpoint'];path=pathlib.Path(ck['path']);assert path.stat().st_size==ck['bytes'] and sha(path)==ck['sha256']
  config=R/'fixed-M-execution-20261101/payload'/('config_A_reference.json' if arm=='A' else 'config_M.json')
  spec={'seed':20261031,'arm':arm,'checkpoint':str(path),'bytes':ck['bytes'],'sha256':ck['sha256'],'config':str(config),'config_sha256':sha(config),'state_receipt':str(P/f'formal_{arm}/CHECKPOINT_RECEIPT.json')};run['models'].append(spec)
  for image_id in old['image_order']:
   key=f'20261031_{arm}_c100_{image_id}';reference=old['identity_map'][f'20261031_A_c100_{image_id}']
   run['identity_map'][key]={**reference,'seed':20261031,'arm':arm,'checkpoint_sha256':ck['sha256'],'config_sha256':spec['config_sha256']};run['expected_identities'].append(key)
 assert len(set(run['expected_identities']))==128
 for folder in ['receipts','predictions','metrics']:(e/folder).mkdir(exist_ok=True)
 atomic(e/'RUN_PROTOCOL.json',run);atomic(e/'RUN_STARTED.json',json.loads((P/'RUN_STARTED.json').read_text()))

def main():
 assert not (P/'RUN_STARTED.json').exists(),'Refuse to restart an already begun paired run'
 frozen=json.loads((P/'CODE_PINS.json').read_text())
 for n,h in frozen.items():assert sha(P/n)==h,n
 proto=json.loads((P/'PROTOCOL.json').read_text());start=time.monotonic()
 ctx={'start_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_monotonic':start,'deadline_monotonic':start+7200,'PID':os.getpid(),'preflight_deadline_monotonic':start+600}
 atomic(P/'RUN_STARTED.json',ctx)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(P/'mplconfig'),HF_HOME=str(P/'hf_home'),XDG_CACHE_HOME=str(P/'cache'))
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
 phases=[]
 def execute(label,script,args,deadline):
  remaining=deadline-time.monotonic();assert remaining>0,'Wall deadline reached before '+label
  t=time.monotonic();row={'label':label,'started_seconds':t-start};phases.append(row);atomic(P/'PHASES.json',phases)
  with (P/(label+'.stdout.log')).open('xb') as log:
   child=subprocess.Popen([sys.executable,'-B',str(script),*args],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
   row['pid']=child.pid;atomic(P/'PHASES.json',phases)
   try:code=child.wait(timeout=remaining)
   except subprocess.TimeoutExpired:
    child.kill();code=child.wait();row['timeout_killed']=True
   row.update(exit_code=code,elapsed_seconds=time.monotonic()-t,child_exited=True);atomic(P/'PHASES.json',phases)
  if code!=0:raise RuntimeError('Phase failed: '+label)
 try:
  for arm in ['A','M']:execute('preflight_'+arm,P/'train_cpu.py',['preflight',arm],ctx['preflight_deadline_monotonic'])
  a=json.loads((P/'preflight_A/RESULT.json').read_text());m=json.loads((P/'preflight_M/RESULT.json').read_text())
  assert a['status']==m['status']=='PREFLIGHT_COMPLETE' and a['completed_updates']==1 and m['completed_updates']==2
  gate=estimate(a,m,time.monotonic()-start,proto)
  gate.update(current_research_plus_staging_bytes=tree(R)+tree(P),estimated_new_models_bytes=2*692175962,other_artifact_allowance_bytes=100*1024**2)
  gate['storage_pass']=gate['current_research_plus_staging_bytes']+gate['estimated_new_models_bytes']+gate['other_artifact_allowance_bytes']<=8*1024**3-80*1024**2
  gate['status']='PASS' if all(gate[k] for k in ['time_pass','memory_pass','storage_pass']) else 'RESOURCE_GATE_STOP'
  atomic(P/'PREFLIGHT_GATE.json',gate);print(json.dumps(gate),flush=True)
  if gate['status']!='PASS':
   atomic(P/'FINAL_STATUS.json',{'status':'RESOURCE_GATE_STOP','formal_updates_A':0,'formal_updates_M':0,'elapsed_seconds':time.monotonic()-start,'GPU_MPS_upload_CU_new':0});return
  # Subprocess boundaries discard every preflight tensor, optimizer and RNG state.
  for arm in ['A','M']:execute('formal_'+arm,P/'train_cpu.py',['formal',arm],ctx['deadline_monotonic'])
  eval_prepare()
  for index in range(2):execute('evaluation_'+str(index),P/'evaluation/formal_cpu.py',['infer','--index',str(index)],ctx['deadline_monotonic'])
  execute('metrics',P/'evaluation/formal_cpu.py',['metrics'],ctx['deadline_monotonic'])
  execute('metric_recompute',P/'evaluation/formal_cpu.py',['verify'],ctx['deadline_monotonic'])
  atomic(P/'FINAL_STATUS.json',{'status':'COMPLETE','formal_updates_A':200,'formal_updates_M':200,'dev_predictions':128,'elapsed_seconds':time.monotonic()-start,'all_child_processes_exited':all(v.get('child_exited') for v in phases),'GPU_MPS_upload_CU_new':0})
 except BaseException as exc:
  atomic(P/'FINAL_STATUS.json',{'status':'INCOMPLETE','error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'all_child_processes_exited':all(v.get('child_exited') for v in phases),'GPU_MPS_upload_CU_new':0});raise

if __name__=='__main__':main()
