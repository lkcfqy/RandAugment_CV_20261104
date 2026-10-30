"""Independent complete-child wall timer; only terminates its own timed-out child."""
import datetime,json,subprocess,sys,threading,time
from pathlib import Path
root=Path(__file__).resolve().parent
start=time.perf_counter();wall=time.time();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
timed_out=[False]
with (root/'process-v2.log').open('x',buffering=1) as log:
    child=subprocess.Popen([sys.executable,'-u',str(root/'baseline_v2.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    def stop_owned_child():
        if child.poll() is None:timed_out[0]=True;child.terminate()
    timer=threading.Timer(5460,stop_owned_child);timer.start()
    try:
        for line in child.stdout:log.write(line);print(line,end='',flush=True)
        code=child.wait()
    finally:timer.cancel()
record={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'whole_child_perf_seconds':time.perf_counter()-start,'whole_child_wall_seconds':time.time()-wall,'exit_code':code,'timeout_terminated':timed_out[0],'scope':'Includes imports, source/data checks, preflight, full training/evaluation/checkpointing, final output, child shutdown.'}
with (root/'process-timing-v2.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record),flush=True)
