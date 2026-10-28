import os,sys,time,json,hashlib,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol-lock.json').read_text())['sha256']
for name,h in json.loads((OUT/'code-lock.json').read_text())['sha256'].items():assert sha(OUT/name)==h
assert not (OUT/'watchdog.json').exists() and not (OUT/'events.jsonl').exists() and not (OUT/'STARTED.json').exists(),'No automatic retry'
dump('STARTED.json',{'status':'one_authorized_disposable_timing_child','limit':55,'training_budget_authorized_only_for_throughput':True})
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.perf_counter();reason=None
with (OUT/'stdout.txt').open('x') as so,(OUT/'stderr.txt').open('x') as se:
 child=subprocess.Popen([sys.executable,'-B',str(OUT/'worker.py')],cwd=ROOT,env=env,stdout=so,stderr=se)
 while child.poll() is None:
  elapsed=time.perf_counter()-start
  if elapsed>=55 or (OUT/'STOP.request').exists():
   reason='external_55_second_cap' if elapsed>=55 else 'explicit_stop_marker';child.kill();child.wait();break
  time.sleep(.02)
 code=child.wait()
wall=time.perf_counter()-start
records=[json.loads(s) for s in (OUT/'events.jsonl').read_text().splitlines()] if (OUT/'events.jsonl').exists() else []
last=records[-1] if records else None
result={'child_inclusive_wall_seconds':wall,'exit_code':code,'stop_reason':reason,'last_event':last,'possible_inflight_work':None if last is None else last['progress'].get('inflight'),'no_other_process_inspection':True,'automatic_retry':False}
dump('watchdog.json',result)
print(json.dumps({'wall_seconds':wall,'exit_code':code,'stop_reason':reason,'last_counts':None if last is None else last['counts']}),flush=True)
sys.exit(0 if code==0 and reason is None else 1)
