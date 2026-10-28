"""Bound own CPU worker wall time. No process-table queries or external actions."""
import sys,os,json,time,subprocess,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
phase=sys.argv[1]
assert phase in ('prepare','preflight','continuation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol-lock.json').read_text())['sha256']
for name,h in json.loads((OUT/'code-lock.json').read_text())['sha256'].items():assert sha(OUT/name)==h
assert not (OUT/(phase+'-watchdog.json')).exists()
assert not (OUT/(phase+'-events.jsonl')).exists(),'No retry after a started scientific worker'
limit=10.
if phase=='preflight':
    prep=json.loads((OUT/'prepare-watchdog.json').read_text())
    assert prep['exit_code']==0 and prep['stop_reason'] is None
    limit=10-prep['child_inclusive_wall_seconds'];assert limit>0

if phase=='continuation':
    ad=json.loads((OUT/'continuation-admission.json').read_text())
    assert ad['admitted'];limit=ad['remaining_worker_wall_seconds'];assert 0<limit<30
# Only this child is ever signalled; no inspection of other projects/processes.
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.perf_counter();reason=None
with (OUT/(phase+'-stdout.txt')).open('x') as stdout,(OUT/(phase+'-stderr.txt')).open('x') as stderr:
    proc=subprocess.Popen([sys.executable,'-B',str(OUT/('prepare.py' if phase=='prepare' else 'worker.py')),phase,str(limit)],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    while proc.poll() is None:
        elapsed=time.perf_counter()-start
        if elapsed>=limit or (OUT/'STOP.request').exists():
            reason='external_wall_cap' if elapsed>=limit else 'explicit_stop_marker'
            proc.kill();proc.wait();break
        time.sleep(min(.02,max(.001,limit-elapsed)))
    code=proc.wait()
elapsed=time.perf_counter()-start
records=[p.name for p in (OUT/'forward-records').glob('*.json')] if (OUT/'forward-records').exists() else []
dump(OUT/(phase+'-watchdog.json'),dict(phase=phase,exit_code=code,stop_reason=reason,limit_seconds=limit,child_inclusive_wall_seconds=elapsed,
    completed_record_count_total=len(records),completed_records_total=sorted(records),compute_interrupted_at_cap=reason=='external_wall_cap',possible_poll_and_cleanup_overhead_seconds=.1))
print(json.dumps(dict(phase=phase,exit_code=code,stop_reason=reason,elapsed=elapsed,completed_record_count_total=len(records))),flush=True)
sys.exit(0 if code==0 and reason is None else 1)
