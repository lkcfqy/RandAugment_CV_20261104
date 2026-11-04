"""Tiny external UTC wall guard; no inference, model, dependency or network calls."""
import datetime,json,os,pathlib,signal,time
from diagnostic_contracts import atomic_json
p=pathlib.Path(__file__).resolve().parent
ctx=json.loads((p/'RUN_STARTED.json').read_text())
started=datetime.datetime.fromisoformat(ctx['utc'])
deadline=started+datetime.timedelta(seconds=5400)
stop_at=deadline-datetime.timedelta(seconds=15)
receipt={'status':'ARMED','pid':os.getpid(),'supervisor_pid':ctx['supervisor_pid'],
         'latest_wall_end_UTC':deadline.isoformat(),'termination_trigger_UTC':stop_at.isoformat(),
         'cleanup_reserve_seconds':15,'model_calls':0,
         'note':'Additional UTC guard; inference/config/code pins unchanged. All formal processes use Python3.11 monotonic; Python3.9 monitoring offsets are not mixed.'}
atomic_json(p/'WALL_WATCHDOG.json',receipt,new=True)
while True:
    if (p/'TERMINAL.json').exists():receipt['status']='DISARMED_FORMAL_TERMINAL_PRESENT';break
    try:os.kill(ctx['supervisor_pid'],0)
    except ProcessLookupError:receipt['status']='SUPERVISOR_EXITED';break
    if datetime.datetime.now(datetime.timezone.utc)>=stop_at:
        receipt['status']='UTC_LIMIT_TERMINATION_SENT'
        atomic_json(p/'WALL_WATCHDOG.json',receipt)
        os.kill(ctx['supervisor_pid'],signal.SIGTERM)
        break
    time.sleep(1)
receipt['finished_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
atomic_json(p/'WALL_WATCHDOG.json',receipt)
print(json.dumps(receipt),flush=True)
