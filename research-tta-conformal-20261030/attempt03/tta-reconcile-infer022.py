from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import datetime,json,os,re,subprocess,time
r=_portable_project_path(__file__, 'research-tta-conformal-20261030/attempt03')
receipt=r/'INFERENCE_CONNECTION_RECOVERY.json'
assert not receipt.exists()
record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'One bounded idempotent recovery of infer022 inside the same alive session; no new allocation, no training, no changed inference setting.', 'evidence':{'readonly_cli_status':'existing T4 session IDLE; last completed scientific exec infer021','readonly_kernel_probe':{'engine_exists':True,'epoch':80,'benchmark_steps':0,'formal_steps':25040,'formal_presentations':3200000,'inference_views':112640,'infer022_done':False,'infer022_array':False}},'safety':'Engine.infer_chunk returns an existing done marker before running; requires exact preceding inference count; kernel requests run serially. A late duplicate request returns the done marker without model execution.','max_additional_recovery_submissions':1,'original_pending_child_pid':17357,'new_allocations':0,'current_allocation_start':1793391776.5344,'prior_GPU_seconds':1652.7684237957,'hard_GPU_seconds':7200,'wall_deadline':1793404407.1513782}
record['cumulative_gpu_seconds_before_recovery']=1652.7684237957+time.time()-1793391776.5344
assert record['cumulative_gpu_seconds_before_recovery']<6000 and time.time()<1793404407.1513782-3600
receipt.write_text(json.dumps(record,indent=2)+'\n')
args=['/Users/lkc/.local/share/google-colab-cli/bin/python',str(r.parent/'colab_guard.py'),'exec','--session','tta-conformal-final-20261030','--file',str(r/'commands/infer022.py'),'--timeout','60']
env=os.environ.copy();env['COLAB_CLI_ASSIGN_RETRIES']='0'
t=time.time()
try:
 c=subprocess.run(args,capture_output=True,text=True,timeout=90,env=env)
 record['exit_code']=c.returncode;record['stdout']=c.stdout;record['stderr']=c.stderr
except subprocess.TimeoutExpired as e:
 record['exit_code']=124;record['stdout']='';record['stderr']='Bounded reconciliation CLI timeout; remote outcome unknown. No further recovery submissions.'
for field in ('stdout','stderr'):
 record[field]=re.sub(r'ya29\.[A-Za-z0-9_.-]+','[REDACTED]',record[field])
 record[field]=re.sub(r'(?i)Bearer\s+[^\s]+','Bearer [REDACTED]',record[field])
record['duration_seconds']=time.time()-t;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('exit_code','stdout','stderr','duration_seconds','finished_utc')}),flush=True)
