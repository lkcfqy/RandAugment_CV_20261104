"""Time the inference subprocess including imports and shutdown."""
from pathlib import Path
import datetime,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with (ROOT/'process.log').open('x') as f:
    proc=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'infer.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    try:code=proc.wait(timeout=660);timeout=False
    except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15);code=proc.returncode;timeout=True
result={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'whole_child_seconds':time.perf_counter()-t,'exit_code':code,'timeout_terminated':timeout,
        'scope':'Inference imports/source checks/shadow calibration/8+8 scoring/saved outputs/shutdown. No optimizer.'}
with (ROOT/'process-timing.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2));sys.exit(code)
