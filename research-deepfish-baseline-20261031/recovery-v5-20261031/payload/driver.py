"""One future Colab entry point. Nothing executes on import; approval gates every side effect."""
import argparse,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time,urllib.request
sys.dont_write_bytecode=True
from contracts import ContractError,NotAuthorized,ResourceUndetermined,canonical,choose_updates,validate_context,verify_paired_traces,verify_initializations
from lifecycle import atomic_bytes,atomic_json,emit_evidence,native_unassign
from transfer import persist_stage
ROOT=pathlib.Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1024**2),b''):h.update(part)
 return h.hexdigest()
def verify_payload(expected):
 p=ROOT/'MANIFEST.json'
 if not expected or sha(p)!=expected:raise ContractError('External reviewed manifest digest required.')
 m=json.loads(p.read_text());actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
 if actual!={r['path'] for r in m['files']}|{'MANIFEST.json','SHA256SUMS'}:raise ContractError('Unexpected or missing payload file.')
 for r in m['files']:
  p=ROOT/r['path']
  if p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ContractError('Payload hash mismatch: '+r['path'])
 return m
def bounded_command(argv,log,deadline,env=None):
 left=deadline-time.monotonic()
 if left<=0:raise ResourceUndetermined('Work deadline exhausted.')
 with pathlib.Path(log).open('wb') as f:
  process=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
  try:code=process.wait(timeout=left)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGKILL);process.wait();raise ResourceUndetermined('Bounded subprocess exceeded work deadline.')
 if code:raise ContractError('Subprocess failed; inspect '+pathlib.Path(log).name)
def download_assets(deadline):
 pins=json.loads((ROOT/'asset_pins.json').read_text())
 for row in pins['assets']:
  p=ROOT/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():
   if p.stat().st_size==row['bytes'] and sha(p)==row['sha256']:continue
   raise ContractError('Existing asset fails declared digest: '+row['path'])
  partial=p.with_suffix(p.suffix+'.part');size=0;h=hashlib.sha256()
  with urllib.request.urlopen(row['url'],timeout=min(40,max(.1,deadline-time.monotonic()))) as response,partial.open('xb') as f:
   while True:
    if time.monotonic()>=deadline:raise ResourceUndetermined('Asset download deadline.')
    chunk=response.read(1024**2)
    if not chunk:break
    size+=len(chunk)
    if size>row['bytes']:raise ContractError('Asset exceeds declared size.')
    f.write(chunk);h.update(chunk)
   f.flush();os.fsync(f.fileno())
  if size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ContractError('Official asset SHA256 mismatch.')
  os.replace(partial,p)
def prepare_environment(logdir,deadline):
 if shutil.disk_usage(ROOT).free<12*1024**3:raise ResourceUndetermined('Need 12 GiB free runtime disk for environment/assets/working files.')
 from bootstrap_environment import bootstrap
 environment=ROOT/'environment'
 try:result=bootstrap(environment,json.loads((ROOT/'python_bootstrap_pins.json').read_text()),(ROOT/'requirements_candidate.txt').read_text(),time.time()+deadline-time.monotonic())
 finally:
  target=logdir/'environment';target.mkdir(exist_ok=True)
  if environment.exists():
   for p in environment.iterdir():
    if p.is_file() and p.suffix in ('.json','.log','.txt'):shutil.copyfile(p,target/p.name)
 if result['primitive_gate']['devices']!=['cpu','cuda'] or result['primitive_gate']['status']!='PASS':raise ResourceUndetermined('Synthetic deterministic CUDA/compiled operator gate did not pass before weights.')
 return pathlib.Path(result['python_executable'])
def _run_owned(ctx,budget):
 verify_payload(ctx.get('expected_manifest_sha256'))
 out=pathlib.Path(ctx['artifact_dir']);out.mkdir(parents=False,exist_ok=False)
 hard=time.monotonic()+budget['hard_deadline_epoch']-time.time();work=hard-180
 ctx=dict(ctx,work_deadline_monotonic=work,hard_deadline_monotonic=hard)
 context_copy=out/'run_context.json';atomic_json(context_copy,ctx)
 native_guard=subprocess.Popen([sys.executable,'-B',str(ROOT/'lifecycle.py'),'--context',str(context_copy)],start_new_session=True)
 result={'status':'STARTED','authorization_scope':'this new DeepFish experiment only','preflight_updates_per_arm':15,'formal_updates_per_arm':None,'official_test_used':False}
 try:
  # Content-addressed stdout receipts survive a VM failure; persistent storage is also mandatory.
  atomic_json(out/'run_report.json',result);emit_evidence('run_started',result)
  sample=os.urandom(16*1024**2);t=time.monotonic();atomic_bytes(out/'write_probe.bin',sample);write_bps=len(sample)/(time.monotonic()-t);(out/'write_probe.bin').unlink();del sample
  write_bps=min(write_bps,ctx['persistence_probe']['download_bytes_per_second'])
  py=prepare_environment(out,work);download_assets(work)
  child_env=dict(os.environ,MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  def phase(name):
   bounded_command([str(py),'-B',str(ROOT/'model_worker.py'),'--context',str(context_copy),'--phase',name],out/(name+'.stdout.log'),work,child_env)
   persist_stage(out,name,ROOT,work)
  for a in 'AB':phase('preflight_'+a)
  probes=[json.loads((out/('preflight_'+a+'.json')).read_text()) for a in 'AB']
  verify_initializations(probes)
  verify_paired_traces(*[json.loads((out/('preflight_'+a+'_trace.json')).read_text()) for a in 'AB'])
  timings=dict(init_seconds=max(p['init_seconds'] for p in probes),p90_update_a=probes[0]['p90_update_seconds'],p90_update_b=probes[1]['p90_update_seconds'],p90_predict_seconds=max(p['p90_predict_seconds'] for p in probes),checkpoint_bytes=max(p['state_tensor_bytes'] for p in probes),write_bps=write_bps)
  # Four initializations remain: formal A/B, then fixed-final dev evaluation A/B.
  timing_for_rule=dict(timings,init_seconds=2*timings['init_seconds'])
  freeze=choose_updates(timing_for_rule,work-time.monotonic());freeze.update(raw_timings=timings,rule_timings=timing_for_rule,initialization_count_remaining=4,preflight_state_reused=False,dev_metrics_seen=False)
  atomic_json(out/'U_FREEZE.json',freeze);emit_evidence('U_FREEZE',freeze)
  if freeze['status']!='FROZEN':raise ResourceUndetermined('No fixed candidate U fits the measured remaining budget.')
  result['formal_updates_per_arm']=freeze['updates']
  for a in 'AB':phase('formal_'+a)
  formal=[json.loads((out/('formal_'+a+'.json')).read_text()) for a in 'AB']
  verify_initializations(probes+formal)
  paired=verify_paired_traces(*[json.loads((out/('formal_'+a+'_trace.json')).read_text()) for a in 'AB'])
  pair_receipt={'paired_trace_sha256':paired,'updates_per_arm':freeze['updates'],'final_checkpoints':[p['final_checkpoint'] for p in formal],'freeze_sha256':sha(out/'U_FREEZE.json')}
  atomic_json(out/'PAIR_CHECK_OK.json',pair_receipt)
  for a in 'AB':phase('evaluate_'+a)
  result.update(status='COMPLETE',dev_results={a:json.loads((out/('evaluate_'+a+'.json')).read_text()) for a in 'AB'},paired_trace_sha256=paired)
 except BaseException as error:
  result.update(status='RESOURCE_UNDETERMINED' if isinstance(error,ResourceUndetermined) else 'INCOMPLETE',error_type=type(error).__name__,error=str(error),scientific_conclusion_complete=False)
 finally:
  result.update(allocated_seconds=time.time()-ctx['allocation_started_epoch'],estimated_cu=ctx.get('prior_cu',0)+(time.time()-ctx['allocation_started_epoch'])*ctx['cu_per_hour_upper_bound']/3600,release_status='EXTERNAL_RECONCILIATION_REQUIRED')
  atomic_json(out/'run_report.json',result);emit_evidence('run_report',result)
  # Preserve the reserved cleanup window; the external guard releases at hard-90.
  try:persist_stage(out,'final',ROOT,hard-100)
  except BaseException as error:
   result.update(status='INCOMPLETE_PERSISTENCE',scientific_conclusion_complete=False,persistence_error_type=type(error).__name__,persistence_error=str(error))
   atomic_json(out/'run_report.json',result);emit_evidence('local_persistence_incomplete',result)
 return 0 if result['status']=='COMPLETE' else 4
def run(context_path):
 ctx=json.loads(pathlib.Path(context_path).read_text());budget=validate_context(ctx,time.time())
 if not os.environ.get('COLAB_RELEASE_TAG') and not os.environ.get('TBE_RUNTIME_ADDR'):raise ResourceUndetermined('Entry point is restricted to the future verified Colab runtime.')
 try:return _run_owned(ctx,budget)
 finally:
  # This boundary covers payload/setup failures too, after owned-runtime context validation.
  # Only the independent external controller can confirm endpoint disappearance.
  release=native_unassign()
  try:
   out=pathlib.Path(ctx['artifact_dir'])
   if out.is_dir():atomic_json(out/'native_release_result.json',release)
   emit_evidence('native_release_result',release)
  except BaseException:pass
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-context',required=True);a=p.parse_args()
 try:return run(a.run_context)
 except NotAuthorized as e:print(json.dumps({'status':'NOT_AUTHORIZED','reason':str(e)}));return 3
 except (ContractError,OSError,ValueError) as e:print(json.dumps({'status':'RESOURCE_UNDETERMINED','reason':str(e)}));return 4
if __name__=='__main__':raise SystemExit(main())
