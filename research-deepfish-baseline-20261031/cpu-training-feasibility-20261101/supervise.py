"""Ten-minute total task deadline; one completed training step maximum."""
import datetime,json,os,pathlib,subprocess,sys,time
P=pathlib.Path(__file__).resolve().parent
limits=json.loads((P/'LIMITS.json').read_text())
deadline=datetime.datetime.fromisoformat(limits['start_UTC']).timestamp()+600
remaining=deadline-time.time()
if remaining<=0:raise RuntimeError('Ten-minute task budget expired before model probe')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(P/'mplconfig'),HF_HOME=str(P/'hf_home'),XDG_CACHE_HOME=str(P/'cache'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
started=time.time()
try:
 with (P/'probe.stdout.log').open('xb') as log:
  process=subprocess.run([sys.executable,'-B',str(P/'probe.py')],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
 outcome=dict(status='CHILD_EXITED',exit_code=process.returncode,seconds=time.time()-started)
except subprocess.TimeoutExpired:
 outcome=dict(status='TEN_MINUTE_DEADLINE_CHILD_KILLED',exit_code=None,seconds=time.time()-started)
(P/'PROCESS_EXIT.json').write_text(json.dumps(outcome,indent=2)+'\n');print(json.dumps(outcome))
