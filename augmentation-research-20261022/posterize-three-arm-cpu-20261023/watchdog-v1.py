import os,sys,time,json,hashlib,subprocess,datetime,signal
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol-lock.json').read_text())['sha256']
for name,h in json.loads((OUT/'code-lock.json').read_text())['sha256'].items():assert sha(OUT/name)==h
budget_file=OUT/'budget.json'
if not budget_file.exists():
 started=time.monotonic();dump(budget_file,{'started_monotonic':started,'deadline_monotonic':started+3600,'started_epoch_seconds':time.time(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_cap_seconds':3600,'cannot_reset':True,'includes_preflight_training_and_evaluation':True,'protocol_sha256':sha(OUT/'protocol.json')})
budget=json.loads(budget_file.read_text());assert budget['protocol_sha256']==sha(OUT/'protocol.json') and budget['wall_cap_seconds']==3600
assert time.monotonic()<budget['deadline_monotonic']-20
attempt=OUT/'attempt-001';attempt.mkdir(exist_ok=False)
dump(attempt/'launch.json',{'budget_sha256':sha(budget_file),'code_sha256':json.loads((OUT/'code-lock.json').read_text())['sha256'],'remaining_at_launch_seconds':budget['deadline_monotonic']-time.monotonic()})
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.perf_counter();soft=False;reason=None
with (attempt/'stdout.txt').open('x') as so,(attempt/'stderr.txt').open('x') as se:
 child=subprocess.Popen([sys.executable,'-B',str(OUT/'worker-v1.py'),attempt.name],cwd=ROOT,env=env,stdout=so,stderr=se)
 while child.poll() is None:
  now=time.monotonic()
  if (OUT/'STOP.request').exists() or now>=budget['deadline_monotonic']-10:
   if not soft:child.send_signal(signal.SIGTERM);soft=True
  if now>=budget['deadline_monotonic']:
   reason='absolute_60_minute_hard_deadline';child.kill();child.wait();break
  time.sleep(.1)
 code=child.wait()
wall=time.perf_counter()-start
records=[]
if (attempt/'events.jsonl').exists():
 for line in (attempt/'events.jsonl').read_text().splitlines():
  try:records.append(json.loads(line))
  except json.JSONDecodeError:pass
last=records[-1] if records else None
dump(attempt/'watchdog.json',{'child_inclusive_wall_seconds':wall,'stage_elapsed_at_finish':time.monotonic()-budget['started_monotonic'],'exit_code':code,'hard_stop_reason':reason,'soft_signal_sent':soft,'last_event':last,'budget_sha256':sha(budget_file),'other_processes_inspected':False,'automatic_retry':False})
print(json.dumps({'child_inclusive_wall_seconds':wall,'exit_code':code,'hard_stop_reason':reason,'last_counts':None if last is None else last['counts']}),flush=True)
sys.exit(0 if code==0 and reason is None else 1)
