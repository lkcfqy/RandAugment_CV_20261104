"""CPU-only dependency and download-persistence validation, with prior budget carried forward."""
import hashlib,json,os,pathlib,subprocess,sys,time,zipfile
from external_watchdog import ROOT,PYTHON,atomic,cli,snapshot,release_owned
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 name='deepfish-cpu-recovery-v1';start=time.time();before=snapshot();atomic(ROOT/'before.json',before)
 if before['assignments'] or before['active_assignments']:raise RuntimeError('Other active allocations prevent attribution')
 c={'authorization':{'upload':True,'model_run':True},'archive_sha256':'0fa965babadfaa83b9e7a83aadd81470fc76b566c118e7883ade3b0042ab2263','session':name,'deadline_epoch':start+1500,'absolute_limit_epoch':start+1800,'allocation_started_epoch':start,'prior_cu':0.056508,'prior_gpu_seconds':20.373}
 atomic(ROOT/'control.json',c);report={'status':'STARTING','prior_cu':.056508,'gpu_requested':False,'allocation_attempts':0,'new_oauth':False};attempt=False
 def run(args,timeout,label):
  env=dict(os.environ,DEEPFISH_ALLOCATION_RECEIPT=str(ROOT/'allocation_receipt.json'))
  with (ROOT/(label+'.log')).open('wb') as f:
   p=subprocess.Popen([PYTHON,'-B',str(ROOT/'colab_guard.py'),*args],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);last=0
   while p.poll() is None:
    if time.time()>c['deadline_epoch']-100:p.kill();p.wait();raise TimeoutError('CPU setup boundary')
    if time.time()-last>30 and attempt:
     current=snapshot();last=time.time();atomic(ROOT/'latest_usage.json',current)
     if current['rate_per_hour']>0:raise RuntimeError('CPU is not zero-CU; stop for revised budget')
    time.sleep(.5)
   if p.returncode:raise RuntimeError('Official CLI action failed: '+args[0])
  return (ROOT/(label+'.log')).read_text()
 try:
  with (ROOT/'watchdog.log').open('xb') as log:g=subprocess.Popen([sys.executable,'-B',str(ROOT/'external_watchdog.py'),'--parent-pid',str(os.getpid())],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+45
  while not (ROOT/'watchdog_state.json').exists() and time.time()<until:
   if g.poll() is not None:raise RuntimeError('CPU watchdog did not arm')
   time.sleep(.2)
  state=json.loads((ROOT/'watchdog_state.json').read_text());os.kill(state['pid'],0);assert time.time()-state['heartbeat_epoch']<5
  attempt=True;report['allocation_attempts']=1;run(['new','--session',name],120,'allocation')
  current=snapshot();atomic(ROOT/'allocated_usage.json',current)
  if current['rate_per_hour']!=0 or any(x['accelerator'] not in ('NONE','') for x in current['assignments']):raise RuntimeError('CPU-only zero-CU allocation was not confirmed')
  print('CPU_ALLOCATED_ZERO_RATE',flush=True)
  output=run(['exec','--session',name,'--file',str(ROOT/'remote_bootstrap.py'),'--timeout','1250'],1300,'remote_bootstrap')
  index=json.loads(next(x[len('DEEPFISH_CPU_EVIDENCE '):] for x in output.splitlines() if x.startswith('DEEPFISH_CPU_EVIDENCE ')));atomic(ROOT/'remote_evidence_index.json',index)
  received=ROOT/'received';received.mkdir(exist_ok=False);download_seconds={}
  for file,row in index.items():
   t=time.monotonic();partial=received/(file+'.part');run(['download','--session',name,'content/deepfish_cpu_bootstrap/'+file,str(partial)],90,'download_'+file);download_seconds[file]=time.monotonic()-t
   assert partial.stat().st_size==row['bytes'] and sha(partial)==row['sha256']
   with partial.open('rb') as f:os.fsync(f.fileno())
   os.replace(partial,received/file);assert sha(received/file)==row['sha256']
  with zipfile.ZipFile(received/'evidence.zip') as z:
   for name2 in z.namelist():
    assert pathlib.PurePosixPath(name2).name==name2
    (received/name2).write_bytes(z.read(name2))
  report.update(status='CPU_PREPARATION_VERIFIED' if (received/'import_gate.json').exists() else 'CPU_ENGINEERING_FAILURE',all_indexed_files_persisted_before_release=True,download_seconds=download_seconds,download_bytes_per_second=index['persistence_marker.bin']['bytes']/download_seconds['persistence_marker.bin'],marker_sha256=index['persistence_marker.bin']['sha256'],model_instantiated=False)
  if (received/'bootstrap_error.json').exists():report['bootstrap_error']=json.loads((received/'bootstrap_error.json').read_text())
 except BaseException as error:report.update(status='CPU_ENGINEERING_FAILURE',error_type=type(error).__name__,error=str(error))
 finally:
  if attempt:report.update(release_owned(c))
  else:report['release']='NOT_ALLOCATED'
  if report['release'] in ('CONFIRMED','NOT_ALLOCATED'):atomic(ROOT/'RELEASE_CONFIRMED.json',{'release':report['release']})
  report['final_sessions_usage']=snapshot();report['elapsed_cpu_allocation_wall_seconds']=time.time()-start;report['prior_cu_preserved']=.056508;atomic(ROOT/'controller_report.json',report);print(json.dumps(report,sort_keys=True),flush=True)
 return 0 if report['status']=='CPU_PREPARATION_VERIFIED' and report['release']=='CONFIRMED' else 2
if __name__=='__main__':raise SystemExit(main())
