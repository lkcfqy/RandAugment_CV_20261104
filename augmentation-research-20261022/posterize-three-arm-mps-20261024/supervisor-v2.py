import os,sys,time,json,subprocess,signal,hashlib
from pathlib import Path
from datetime import datetime,timezone
D=Path(__file__).resolve().parent
b=json.loads((D/'budget.json').read_text())
assert time.time()<b['readiness_deadline_epoch']
py=json.loads((D/'environment-spec.json').read_text())['python']
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
log=(D/'attempt-002/stdout.log').open('x',buffering=1)
p=subprocess.Popen([py,'-B','-u',str(D/'worker-v2.py'),'attempt-002'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
with (D/'attempt-002/launch.json').open('x') as f:json.dump({'PID':p.pid,'at_utc':datetime.now(timezone.utc).isoformat(),'phase_start_utc':b['start_utc'],'deadline_utc':b['deadline_utc'],'command':[py,'-B','-u',str(D/'worker-v2.py'),'attempt-002'],'supervisor_pid':os.getpid(),'worker_sha256':hashlib.sha256((D/'worker-v2.py').read_bytes()).hexdigest(),'signal_scope':'only owned worker PID'},f,indent=2)
print('SUPERVISOR_WORKER_PID',p.pid,flush=True)
term=False;kill=False;pos=0
while p.poll() is None:
    now=time.time()
    if now>=b['deadline_epoch']-30 and not term:
        p.send_signal(signal.SIGTERM);term=True;print('DEADLINE_SAVE_SIGNAL',flush=True)
    if now>=b['deadline_epoch'] and not kill:
        p.kill();kill=True;print('HARD_DEADLINE_KILL',flush=True)
    with (D/'attempt-002/stdout.log').open() as f:
        f.seek(pos);s=f.read();pos=f.tell()
    if s:print(s,end='',flush=True)
    time.sleep(1)
log.close()
with (D/'attempt-002/stdout.log').open() as f:f.seek(pos);s=f.read()
if s:print(s,end='',flush=True)
with (D/'attempt-002/process-exit.json').open('x') as f:json.dump({'returncode':p.returncode,'at_utc':datetime.now(timezone.utc).isoformat(),'phase_elapsed':time.time()-b['start_epoch'],'deadline_term_sent':term,'deadline_kill_sent':kill},f,indent=2)
print('PROCESS_EXIT',p.returncode,flush=True)
sys.exit(p.returncode if p.returncode>=0 else 1)
