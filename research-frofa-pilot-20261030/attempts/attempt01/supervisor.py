import os,sys,time,json,subprocess,signal,hashlib
from pathlib import Path
from runtime import ROOT,P,save,utc
def main():
    seal=json.loads((ROOT/'source-seal.json').read_text())
    for rel,digest in seal.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise SystemExit('source seal mismatch '+rel)
    ledger=ROOT/'ledger.json'
    if ledger.exists() and json.loads(ledger.read_text()).get('started'):raise SystemExit('A budget already started; inspect before any continuation, never reset.')
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'deps'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0',HF_HOME=str(ROOT/'hf-cache'),HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    attempts=[]
    for script in ['bootstrap.py','pilot.py']:
        with (ROOT/(script+'.log')).open('xb') as log:
            process=subprocess.Popen([sys.executable,'-B',str(ROOT/script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            attempts.append({'script':script,'pid':process.pid,'start_utc':utc()});save(ROOT/'supervisor.json',attempts)
            reason=None
            while process.poll() is None:
                if ledger.exists():
                    state=json.loads(ledger.read_text())
                    if state.get('started'):
                        elapsed=max(time.time()-state['start_epoch'],time.monotonic()-state['start_mono'])
                        cap=P['limits']['wall_seconds'] if state.get('preflight_passed') else P['limits']['preflight_seconds']
                        if elapsed>=cap:
                            reason='supervisor_deadline_'+str(cap);os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=3)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                            save(ROOT/'STOP.json',{'reason':reason,'elapsed_seconds':elapsed,'resource_stop':True,'utc':utc()});break
                time.sleep(.5)
            process.wait();attempts[-1].update(exit_code=process.returncode,end_utc=utc(),termination_reason=reason);save(ROOT/'supervisor.json',attempts)
            if process.returncode!=0:return process.returncode
    return 0
if __name__=='__main__':raise SystemExit(main())
