"""Budget controller: elapsed MPS phases are cumulative, even on worker failure."""
import json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
def put(path,obj):
    t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(path)
def tree_bytes():
    total=0
    for p in BASE.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass  # atomic checkpoint rename during a scan
    return total
def interrupted(signum,frame):raise KeyboardInterrupt(f'Controller signal {signum}')
ledgerfile=BASE/'mps-ledger.json'
ledger=json.loads(ledgerfile.read_text()) if ledgerfile.exists() else {'limit_seconds':7200,'phases':[],'cumulative_seconds':0.,'peak_new_bytes':0}
assert not ledger.get('active'),'Existing active phase must be reconciled before a restart'
deadline=datetime(2026,10,31,1,14,tzinfo=timezone.utc).timestamp()
started=time.time();mono=time.monotonic();phase={'domain':'cifar100_view_route_pilot_v1','name':sys.argv[1],'args':sys.argv[2:],'started_utc':datetime.now(timezone.utc).isoformat(),'started_epoch':started}
remaining=7200-ledger['cumulative_seconds']
assert remaining>10 and started<deadline-10
assert sys.argv[1] in ['oracle.py','pilot.py'],'Only this experiment workers are allowed'
ledger['active']=phase;put(ledgerfile,ledger)
log=ROOT/'logs'/f'{Path(sys.argv[1]).stem}-{len(ledger["phases"])+1:02d}.txt'
with log.open('w') as out:
    child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/sys.argv[1]),*sys.argv[2:]],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                           env={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'})
    phase['pid']=child.pid;phase['log']=str(log.relative_to(BASE));reason=None
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while child.poll() is None:
            elapsed=time.monotonic()-mono;size=tree_bytes()
            ledger['peak_new_bytes']=max(size,ledger['peak_new_bytes']);phase['elapsed_seconds']=elapsed;put(ledgerfile,ledger)
            # Reserve ten seconds to terminate, account and persist before either hard cap.
            if elapsed>=remaining-10:reason='MPS_120_MIN_LIMIT'
            if time.time()>=deadline-10:reason='SIX_HOUR_WALL_LIMIT'
            if size>3*1024**3:reason='THREE_GIB_DISK_LIMIT'
            if reason:break
            time.sleep(1)
    except BaseException as exc:reason=f'CONTROLLER_INTERRUPTED: {type(exc).__name__}: {exc}'
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    phase['elapsed_seconds']=time.monotonic()-mono;phase['exit_code']=child.returncode;phase['budget_stop']=reason
    phase['ended_utc']=datetime.now(timezone.utc).isoformat();ledger['cumulative_seconds']+=phase['elapsed_seconds']
    ledger['phases'].append(phase);ledger['active']=None;put(ledgerfile,ledger)
print(json.dumps(phase),flush=True)
sys.exit(child.returncode if child.returncode and child.returncode>0 else 1 if reason or child.returncode else 0)
