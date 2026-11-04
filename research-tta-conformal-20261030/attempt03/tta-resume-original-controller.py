from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import datetime,json,os,signal,subprocess,time,hashlib
r=_portable_project_path(__file__, 'research-tta-conformal-20261030/attempt03')
path=Path('/tmp/tta-infer022-recovered.done.json')
cmd=['/Users/lkc/.local/share/google-colab-cli/bin/python',str(r.parent/'colab_guard.py'),'download','--session','tta-conformal-final-20261030','/content/tta_attempt03/output/infer022.done.json',str(path)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
assert p.returncode==0,p.stderr
value=json.loads(path.read_text());prior=json.loads((r/'commits/infer021.done.json').read_text())
assert value['success'] and value['action']=='infer022' and value['identity']==prior['identity']
assert value['audit']=={'benchmark_steps':0,'benchmark_presentations':0,'formal_steps':25040,'formal_presentations':3200000,'inference_views':117760}
assert value['logits']['row_start']==11264 and value['logits']['row_end']==11776
assert value['logits']['checkpoint_sha256']==json.loads((r/'commits/formal080.receipt.json').read_text())['sha256']
record=json.loads((r/'INFERENCE_CONNECTION_RECOVERY.json').read_text())
record['marker_verified_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record['marker_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();record['confirmed_audit']=value['audit']
result=subprocess.run(['ps','-p','17357','-o','ppid=,command='],capture_output=True,text=True)
expected='/Users/lkc/.local/share/google-colab-cli/bin/python '+str(r.parent/'colab_guard.py')+' exec --session tta-conformal-final-20261030 --file '+str(r/'commands/infer022.py')+' --timeout 600'
line=result.stdout.strip()
if line:
 parent,actual=line.split(None,1)
 assert parent=='16874' and actual==expected, 'PID identity mismatch; no signal sent'
 os.kill(17357,signal.SIGTERM)
 record['stale_local_cli_terminated']=True
 record['termination_reason']='Recovered completion marker validated. Only the old local client is ended; original controller reconciles marker and continues, no remote stop/kernel interrupt.'
else:record['stale_local_cli_terminated']=False
(r/'INFERENCE_CONNECTION_RECOVERY.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'marker_verified':True,'audit':value['audit'],'stale_client_terminated':record['stale_local_cli_terminated'],'no_new_allocation':True}),flush=True)
