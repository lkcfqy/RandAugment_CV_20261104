"""Synthetic lifecycle/budget and static CPU-computation tests; no model/network."""
import ast,copy,importlib,json,pathlib,sys,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'payload'),str(ROOT/'gpu-run')]
from budget_M import gate,PREPARATION_SECONDS,REFERENCE_BPS,REFERENCE_INIT_SECONDS,REFERENCE_UPDATE_SECONDS
from contracts import ContractError
import external_watchdog as watchdog
checks=[]
def record(name,fn):fn();checks.append(name)
ctx=dict(prior_cu=1.8088330815795528,prior_gpu_seconds=5443.899788473129,cu_per_hour_upper_bound=1.15)
def nominal():
 a=gate(ctx,0,prepare_remaining=PREPARATION_SECONDS)
 b=gate(ctx,PREPARATION_SECONDS)
 assert a['status']==b['status']=='FITS'
 assert abs(a['required_total_seconds']-2112.2549063134115)<1e-7
 assert a['required_total_seconds']==b['required_total_seconds']
 assert abs(b['projected_cumulative_CU']-2.4835811766518927)<1e-10
record('Same whole25 formula before setup and after elapsed preparation',nominal)
record('Preparation boundary rejects completeU200',lambda:(_ for _ in ()).throw(AssertionError()) if gate(ctx,595.65)['status']=='FITS' else None)
record('Slow update rejects without reducingU',lambda:(_ for _ in ()).throw(AssertionError()) if gate(ctx,PREPARATION_SECONDS,t_update=2.0)['status']=='FITS' else None)
record('Slow return rejects with complete checkpoint/release included',lambda:(_ for _ in ()).throw(AssertionError()) if gate(ctx,PREPARATION_SECONDS,bps=3000000)['status']=='FITS' else None)
def ledger():
 try:gate(dict(ctx,prior_cu=0),0)
 except ContractError:return
 raise AssertionError('Ledger reset accepted')
record('Ledger reset refused',ledger)

def release_test(denied):
 old=(watchdog.ROOT,watchdog.snapshot,watchdog.cli)
 with tempfile.TemporaryDirectory() as name:
  root=pathlib.Path(name);watchdog.ROOT=root
  (root/'allocation_receipt.json').write_text(json.dumps(dict(endpoint='synthetic-owned')))
  state={'assignments':[dict(endpoint='synthetic-owned',name='synthetic')],'active_assignments':1}
  calls=[]
  watchdog.snapshot=lambda:copy.deepcopy(state)
  def command(args,timeout):
   calls.append(args)
   if denied:raise RuntimeError('Synthetic denied release')
   state['assignments']=[];state['active_assignments']=0
  watchdog.cli=command
  first=watchdog.release_owned(dict(session='synthetic'));second=watchdog.release_owned(dict(session='synthetic'))
  assert len(calls)==1
  assert first['release']==('UNCERTAIN' if denied else 'CONFIRMED')
  assert second['release']==first['release']
 watchdog.ROOT,watchdog.snapshot,watchdog.cli=old
record('Successful owned release reconciles without duplicate request',lambda:release_test(False))
record('Denied owned release is recorded and never retried',lambda:release_test(True))

def function_source(path,name):
 s=path.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.get_source_segment(s,node)
original=pathlib.Path('/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-deepfish-baseline-20261031/cpu-cross-input-diagnostic-20261101')
def same_metrics():
 for name in ('evaluate_condition','mapped_stats','array_digest'):
  a=function_source(original/'diagnostic_contracts.py',name);b=function_source(ROOT/'cpu-evaluation/diagnostic_contracts.py',name)
  assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))
record('CPU metric computation AST identical to historicalCPU A',same_metrics)
def same_load():
 a=function_source(original/'formal_cpu.py','safe_model')
 a=a.replace("(PREP/'receipts/r1'/f\"probe_{spec['seed']}_{spec['arm']}.json\")","pathlib_path(spec['state_receipt'])")
 b=function_source(ROOT/'cpu-evaluation/formal_cpu.py','safe_model')
 assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))
record('CPU complete-state safe load unchanged except new receipt input',same_load)
def same_infer():
 a=function_source(original/'formal_cpu.py','infer').replace("'expected':128","'expected':64").replace('!=128','!=64').replace('Expected128 completed identities','Expected64 completed identities')
 a=a.replace("geometry_hash=digest(geometry);batch=pseudo_collate([packed])","geometry_hash=digest(geometry)\n                if geometry_hash!=pinned['reference_A_geometry_sha256']:raise ContractError('Geometry/GT differs from historical CPU A')\n                batch=pseudo_collate([packed])")
 b=function_source(ROOT/'cpu-evaluation/formal_cpu.py','infer')
 assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))
record('CPU inference unchanged except64 count and extra geometry equality guard',same_infer)
def static():
 for folder in ('payload','gpu-run','cpu-evaluation'):
  for f in (ROOT/folder).glob('*.py'):compile(f.read_text(),f.name,'exec')
 worker=(ROOT/'payload/candidate_worker.py').read_text()
 assert "kind=='evaluate'" not in worker and "'evaluate'" not in worker
 driver=(ROOT/'payload/driver.py').read_text()
 assert "phase('preflight_M')" in driver and "phase('formal_M')" in driver
 assert 'evaluate_' not in driver
 assert "(ROOT/'requirements_runtime_lock.txt').read_text()" in driver
record('All scripts compile; GPU schedule has no dev phase',static)
assert not any(x in sys.modules for x in ('torch','mmcv','mmdet','numpy'))
(ROOT/'reports/EXECUTION_TESTS.json').write_text(json.dumps(dict(status='PASS',checks=checks,framework_imports=0,model_calls=0,network_calls=0),indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks),model_calls=0,network_calls=0)))
