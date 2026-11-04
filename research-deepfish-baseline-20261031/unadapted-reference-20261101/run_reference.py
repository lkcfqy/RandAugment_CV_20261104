import pathlib,json,os,sys,subprocess,time,datetime,traceback
P=pathlib.Path(__file__).resolve().parent
assert not (P/'RUN_STARTED.json').exists(),'No duplicate unadapted run'
def atomic(p,obj):
 b=json.dumps(obj,indent=2,sort_keys=True).encode()+b'\n';tmp=p.with_name(p.name+'.partial')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
start=time.monotonic();ctx={'start_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_monotonic':start,'deadline_monotonic':start+600,'PID':os.getpid()};atomic(P/'RUN_STARTED.json',ctx)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(P/'mplconfig'),HF_HOME=str(P/'hf_home'),XDG_CACHE_HOME=str(P/'cache'))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
phases=[]
try:
 for label,args in [('inference64',['infer','--index','0']),('metrics',['metrics']),('recompute',['verify'])]:
  row={'label':label,'started_seconds':time.monotonic()-start};phases.append(row);atomic(P/'PHASES.json',phases)
  with (P/(label+'.stdout.log')).open('xb') as f:
   child=subprocess.Popen([sys.executable,'-B',str(P/'formal_cpu.py'),*args],cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT);row['PID']=child.pid;atomic(P/'PHASES.json',phases)
   try:code=child.wait(timeout=max(.001,ctx['deadline_monotonic']-time.monotonic()))
   except subprocess.TimeoutExpired:child.kill();code=child.wait();row['timed_out']=True
   except BaseException:child.kill();child.wait();raise
  row.update(exit_code=code,elapsed_seconds=time.monotonic()-start-row['started_seconds'],child_exited=True);atomic(P/'PHASES.json',phases)
  assert code==0,'Failed phase '+label
 atomic(P/'FINAL_STATUS.json',{'status':'COMPLETE','dev_forwards':64,'training_updates':0,'elapsed_seconds':time.monotonic()-start,'all_children_exited':all(x['child_exited'] for x in phases),'new_GPU_MPS_install_download_upload_CU':0})
except BaseException as e:
 atomic(P/'FINAL_STATUS.json',{'status':'INCOMPLETE','error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'all_completed_children_exited':all(x.get('child_exited',False) for x in phases),'training_updates':0,'new_GPU_MPS_install_download_upload_CU':0});raise
