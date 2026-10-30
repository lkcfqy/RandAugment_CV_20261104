"""Independent whole-child wall timer, includes imports and final output."""
import datetime,json,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
start=time.perf_counter();wall=time.time();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (p/'process.log').open('x',buffering=1) as f:
    child=subprocess.Popen([sys.executable,'-u',str(p/'diagnose.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in child.stdout:
        f.write(line);print(line,end='',flush=True)
    code=child.wait()
r={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'whole_child_perf_seconds':time.perf_counter()-start,'whole_child_wall_seconds':time.time()-wall,'exit_code':code,'includes':'Python startup/imports, all read-only audit and microfit work, final serialization/printing, process shutdown'}
with (p/'process-timing.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
