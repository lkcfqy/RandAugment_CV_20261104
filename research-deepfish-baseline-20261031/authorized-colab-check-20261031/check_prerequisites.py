"""One bounded T4 prerequisite check; release immediately when frozen prerequisites fail."""
from portable_dev.paths import project_path as _portable_project_path
import hashlib,json,os,pathlib,subprocess,sys,time
from external_watchdog import ROOT,PYTHON,atomic,cli,snapshot,release_owned
PACKAGE=_portable_project_path(__file__, 'research-deepfish-baseline-20261031/local-runpack-20261031/deepfish_5shot_two_arm.zip')
def command(args,timeout,log):
 r=subprocess.run([PYTHON,'-B',str(ROOT/'colab_guard.py'),*args],capture_output=True,text=True,timeout=timeout,env=dict(os.environ,DEEPFISH_ALLOCATION_RECEIPT=str(ROOT/'allocation_receipt.json')))
 atomic(ROOT/(log+'.json'),{'arguments':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'finished_epoch':time.time()})
 if r.returncode:raise RuntimeError('Official CLI action failed: '+args[0])
 return r.stdout
def main():
 authorization=json.loads((ROOT/'AUTHORIZATION.json').read_text());assert authorization['authorization']=={'upload':True,'model_run':True}
 assert PACKAGE.stat().st_size==authorization['archive_bytes'] and hashlib.sha256(PACKAGE.read_bytes()).hexdigest()==authorization['archive_sha256']
 before=snapshot();atomic(ROOT/'before_allocation.json',before)
 if before['assignments'] or before['active_assignments'] or before['balance']<2:raise RuntimeError('Isolated 2 CU attribution unavailable')
 session='deepfish-5shot-'+authorization['archive_sha256'][:12];start=time.time();control=dict(authorization['authorization']);control={'authorization':control,'archive_sha256':authorization['archive_sha256'],'session':session,'allocation_started_epoch':start,'deadline_epoch':start+300,'absolute_limit_epoch':start+7200,'max_cu':2,'rate_verified':False}
 atomic(ROOT/'control.json',control)
 report={'status':'STARTING','session':session,'allocation_attempts':0,'model_executed':False,'package_uploaded':False,'remote_probe_bytes':(ROOT/'remote_environment_probe.py').stat().st_size,'dependencies_installed':False,'predictions_produced':False,'new_credentials_created':False,'credits_purchased':False,'before':before}
 allocated_attempt=False;guard=None
 try:
  with (ROOT/'watchdog.log').open('xb') as log:guard=subprocess.Popen([sys.executable,'-B',str(ROOT/'external_watchdog.py'),'--parent-pid',str(os.getpid())],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+50
  while time.time()<until and not (ROOT/'watchdog_state.json').exists():
   if guard.poll() is not None:raise RuntimeError('Independent watchdog failed before allocation')
   time.sleep(.2)
  state=json.loads((ROOT/'watchdog_state.json').read_text());os.kill(state['pid'],0)
  if state['status']!='ARMED' or time.time()-state['heartbeat_epoch']>5:raise RuntimeError('Independent watchdog not verifiably armed')
  report['external_watchdog_verified']={'pid':state['pid'],'heartbeat_epoch':state['heartbeat_epoch'],'independent_cli_snapshot_succeeded':True}
  atomic(ROOT/'ALLOCATION_CLAIM.json',{'session':session,'started_epoch':start,'archive_sha256':authorization['archive_sha256']})
  allocated_attempt=True;report['allocation_attempts']=1
  command(['new','--gpu','T4','--session',session],120,'allocation')
  current=snapshot();atomic(ROOT/'allocated_usage.json',current);receipt=json.loads((ROOT/'allocation_receipt.json').read_text());own=[a for a in current['assignments'] if a['endpoint']==receipt['endpoint']]
  rate=current['rate_per_hour']
  if len(own)!=1 or len(current['assignments'])!=1 or current['active_assignments']!=1 or own[0]['accelerator']!='T4' or not isinstance(rate,(int,float)) or rate<=0:raise RuntimeError('Live single-T4 rate and ownership could not be verified')
  control.update(rate_verified=True,cu_per_hour_upper_bound=rate,deadline_epoch=min(start+300,start+7200,start+1.95*3600/rate));atomic(ROOT/'control.json',control)
  report['live_rate_cu_per_hour']=rate;report['allocated_usage']=current
  if control['deadline_epoch']-time.time()<100:raise RuntimeError('Insufficient bounded time for prerequisite probe and release')
  print(json.dumps({'stage':'LIVE_RATE_VERIFIED','rate_cu_per_hour':rate,'elapsed_seconds':time.time()-start}),flush=True)
  output=command(['exec','--session',session,'--file',str(ROOT/'remote_environment_probe.py'),'--timeout','30'],45,'remote_environment')
  environment=json.loads(next(x[len('DEEPFISH_ENVIRONMENT '):] for x in output.splitlines() if x.startswith('DEEPFISH_ENVIRONMENT ')));atomic(ROOT/'environment.json',environment);report['environment']=environment
  missing=[]
  if not environment['python310_compatible']:missing.append('Frozen Python 3.10 candidate is absent; no interpreter-installation or alternative candidate is authorized by the frozen runtime plan')
  if not environment['persistent_mount_candidates']:missing.append('No existing durable remote mount was found; persistent destination cannot be marked verified')
  if environment['free_bytes']<12*1024**3:missing.append('Less than 12 GiB available')
  if not environment['native_management_env_present']:missing.append('Native Colab release-management environment is unavailable')
  if environment['gpu_query_returncode']:missing.append('NVIDIA runtime query failed')
  report['blockers']=missing
  if missing:report['status']='RESOURCE_UNDETERMINED_PREREQUISITES_FAILED';print(json.dumps({'stage':report['status'],'blockers':missing}),flush=True)
  else:
   report['status']='AWAITING_DURABLE_STORAGE_VERIFICATION';atomic(ROOT/'controller_report.json',report);print(json.dumps({'stage':report['status'],'environment':environment}),flush=True)
   while time.time()<control['deadline_epoch']-120 and not (ROOT/'CONTINUE_VERIFIED').exists():time.sleep(1)
   # A full continuation requires a separately verified persistent mount and driver context.
   if not (ROOT/'CONTINUE_VERIFIED').exists():report['blockers']=['Persistent storage verification not completed before bounded setup deadline'];report['status']='RESOURCE_UNDETERMINED'
   else:raise RuntimeError('Continuation must be taken over before this bounded prerequisite controller exits')
 except BaseException as error:
  report.update(status='RESOURCE_UNDETERMINED',error_type=type(error).__name__,error=str(error))
 finally:
  if allocated_attempt:
   recovery=release_owned(control);report.update(recovery)
   if recovery['release']=='CONFIRMED':atomic(ROOT/'RELEASE_CONFIRMED.json',recovery)
  else:report['release']='NOT_ALLOCATED';atomic(ROOT/'RELEASE_CONFIRMED.json',{'release':'NOT_ALLOCATED'})
  try:after=snapshot();report['final_sessions_usage']=after;atomic(ROOT/'after_release_sessions_usage.json',after)
  except Exception as e:report['final_sessions_usage_error']=type(e).__name__
  report['allocation_wall_seconds_upper']=time.time()-start if allocated_attempt else 0
  if allocated_attempt and 'live_rate_cu_per_hour' in report:report['consumption_upper_estimate_cu']=max(0,before['balance']-report.get('final_sessions_usage',before)['balance'],report['live_rate_cu_per_hour']*report['allocation_wall_seconds_upper']/3600)
  atomic(ROOT/'controller_report.json',report);print(json.dumps({k:report.get(k) for k in ('status','release','allocation_attempts','allocation_wall_seconds_upper','consumption_upper_estimate_cu','blockers','error')}),flush=True)
 return 0 if report['release']=='CONFIRMED' and not report.get('error') else 2
if __name__=='__main__':raise SystemExit(main())
