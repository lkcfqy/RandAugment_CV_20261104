"""One bounded local CPU lifecycle; no network, accelerator, upload or install."""
from portable_dev.paths import project_path as _portable_project_path
import datetime,hashlib,json,os,pathlib,subprocess,sys,time,traceback,shutil
P=pathlib.Path(__file__).resolve().parent;R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 b=json.dumps(v,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n';tmp=p.with_name(p.name+'.partial')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def tree(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def estimate(pre,elapsed):
 scalar={a:max(s['train_step_seconds']/s['k'] for s in p['steps']) for a,p in pre.items()};shape=800/750
 training=(200*scalar['A1']+200*scalar['M1']+659*scalar['C0'])*shape
 inference=192*2.256413041992346;save=3*60;other=180;remaining=1.25*(training+inference+save+other);rss=1.25*(max(x['peak_RSS_bytes'] for x in pre.values())*shape+700*1024**2)
 b=tree(R)+tree(P);needed=3*692247962+100*1024**2
 return {'per_source_measured_seconds':scalar,'geometry_factor':shape,'training_seconds_before_margin':training,'inference_seconds_before_margin':inference,'save_verify_seconds_before_margin':save,'other_seconds_before_margin':other,'safety_multiplier':1.25,'measured_preflight_seconds':elapsed,'estimated_remaining_seconds':remaining,'estimated_total_seconds':elapsed+remaining,'conservative_peak_RSS_bytes':rss,'current_research_plus_staging_bytes':b,'new_models_plus_remaining_artifacts_allowance':needed,'filesystem_free_bytes':shutil.disk_usage(P).free,'time_pass':elapsed+remaining<10800,'memory_pass':rss<=12*1024**3,'storage_pass':b+needed+80*1024**2<12*1024**3,'filesystem_pass':shutil.disk_usage(P).free>needed+2*1024**3}
def eval_prepare():
 e=P/'evaluation';old=read(R/'cpu-cross-input-diagnostic-20261101/RUN_PROTOCOL.json');run={k:old[k] for k in ['original_protocol_sha256','core_runtime_versions','dev_annotation','environment_lock_sha256','official_implementation_sha256','image_order']}
 run.update(authorized_CPU_three_arms=True,code_sha256={n:sha(e/n) for n in ['formal_cpu.py','diagnostic_contracts.py']},models=[],identity_map={},expected_identities=[],CPU_trained_all_three=True)
 # This barrier verifies allthree before a single new dev inference.
 records={a:read(P/f'formal_{a}/RESULT.json') for a in ['A1','M1','C0']}
 for a,r in records.items():assert r['status']=='TRAINING_COMPLETE' and r['completed_updates']==200 and r['checkpoint']['complete_state_verified']
 for arm,r in records.items():
  ck=r['checkpoint'];path=pathlib.Path(ck['path']);assert path.stat().st_size==ck['bytes'] and sha(path)==ck['sha256'];config=R/'fixed-M-execution-20261101/payload'/('config_M.json' if arm=='M1' else 'config_A_reference.json');spec={'seed':20261031,'arm':arm,'checkpoint':str(path),'bytes':ck['bytes'],'sha256':ck['sha256'],'config':str(config),'config_sha256':sha(config),'state_receipt':str(P/f'formal_{arm}/CHECKPOINT_RECEIPT.json')};run['models'].append(spec)
  for iid in old['image_order']:
   key=f'20261031_{arm}_c100_{iid}';ref=old['identity_map'][f'20261031_A_c100_{iid}'];run['identity_map'][key]={**ref,'arm':arm,'checkpoint_sha256':ck['sha256'],'config_sha256':spec['config_sha256']};run['expected_identities'].append(key)
 assert len(set(run['expected_identities']))==192
 for f in ['receipts','predictions','metrics']:(e/f).mkdir(exist_ok=True)
 atomic(e/'RUN_PROTOCOL.json',run);atomic(e/'RUN_STARTED.json',read(P/'RUN_STARTED.json'));atomic(P/'ALL_THREE_CHECKPOINTS_BEFORE_DEV.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_sha256':{a:r['checkpoint']['sha256'] for a,r in records.items()},'dev_predictions_at_barrier':len(list((e/'predictions').glob('*.json')))})
def main():
 assert not (P/'RUN_STARTED.json').exists(),'No duplicate lifecycle start'
 for n,h in read(P/'CODE_PINS.json').items():assert sha(P/n)==h,n
 assert read(P/'TEST_RESULTS.json')['status']=='PASS';start=time.monotonic();ctx={'start_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_monotonic':start,'deadline_monotonic':start+10800,'PID':os.getpid(),'preflight_deadline_monotonic':start+600};atomic(P/'RUN_STARTED.json',ctx)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(P/'mplconfig'),HF_HOME=str(P/'hf_home'),XDG_CACHE_HOME=str(P/'cache'))
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
 phases=[]
 def execute(label,script,args,deadline):
  remain=deadline-time.monotonic();assert remain>0;began=time.monotonic();row={'label':label,'started_seconds':began-start};phases.append(row);atomic(P/'PHASES.json',phases)
  with (P/(label+'.stdout.log')).open('xb') as log:
   child=subprocess.Popen([sys.executable,'-B',str(script),*args],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT);row['pid']=child.pid;atomic(P/'PHASES.json',phases)
   try:code=child.wait(timeout=remain)
   except subprocess.TimeoutExpired:child.kill();code=child.wait();row['timeout_killed']=True
   except BaseException:child.kill();child.wait();row['child_exited']=True;atomic(P/'PHASES.json',phases);raise
   row.update(exit_code=code,elapsed_seconds=time.monotonic()-began,child_exited=True);atomic(P/'PHASES.json',phases)
  assert code==0,'Phase failed: '+label
 try:
  for arm in ['A1','M1','C0']:execute('preflight_'+arm,P/'train_cpu.py',['preflight',arm],ctx['preflight_deadline_monotonic'])
  pre={a:read(P/f'preflight_{a}/RESULT.json') for a in ['A1','M1','C0']};assert all(v['status']=='PREFLIGHT_COMPLETE' for v in pre.values());gate=estimate(pre,time.monotonic()-start);gate['status']='PASS' if all(gate[k] for k in ['time_pass','memory_pass','storage_pass','filesystem_pass']) else 'RESOURCE_GATE_STOP';atomic(P/'PREFLIGHT_GATE.json',gate);print(json.dumps(gate),flush=True)
  if gate['status']!='PASS':atomic(P/'FINAL_STATUS.json',{'status':'RESOURCE_GATE_STOP','actual_formal_updates':{'A1':0,'M1':0,'C0':0},'elapsed_seconds':time.monotonic()-start,'all_child_processes_exited':True,'GPU_MPS_upload_CU_new':0});return
  for arm in ['A1','M1','C0']:execute('formal_'+arm,P/'train_cpu.py',['formal',arm],ctx['deadline_monotonic'])
  eval_prepare()
  for i in range(3):execute('evaluation_'+str(i),P/'evaluation/formal_cpu.py',['infer','--index',str(i)],ctx['deadline_monotonic'])
  execute('metrics',P/'evaluation/formal_cpu.py',['metrics'],ctx['deadline_monotonic']);execute('metric_recompute',P/'evaluation/formal_cpu.py',['verify'],ctx['deadline_monotonic'])
  atomic(P/'FINAL_STATUS.json',{'status':'COMPLETE','actual_formal_updates':{'A1':200,'M1':200,'C0':200},'dev_predictions':192,'elapsed_seconds':time.monotonic()-start,'all_child_processes_exited':all(x.get('child_exited') for x in phases),'GPU_MPS_upload_CU_new':0})
 except BaseException as exc:
  atomic(P/'FINAL_STATUS.json',{'status':'INCOMPLETE','error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'all_child_processes_exited':all(x.get('child_exited') for x in phases),'GPU_MPS_upload_CU_new':0});raise
if __name__=='__main__':main()
