"""Original two-arm driver with verified local stage downloads and cumulative budget."""
import base64,hashlib,json,os,pathlib,re,shutil,subprocess,sys,time,zipfile
from external_watchdog import ROOT,PYTHON,atomic,cli,snapshot,release_owned,request_final_release
from local_transport import bounded_stream
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'payload'))
from contracts import complete_repeat_budget,ResourceUndetermined,MAX_CU,MAX_GPU_SECONDS
LOCAL_CPU_PYTHON='/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-readiness/.venv/bin/python'
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 receipt=json.loads((BASE/'reports/PACKAGE_RECEIPT.json').read_text());archive=BASE/receipt['archive'];assert archive.stat().st_size==receipt['archive_bytes'] and sha(archive)==receipt['archive_sha256']
 ledger=json.loads((BASE/'execution_ledger.json').read_text());assert ledger['CPU_gate_verified'] and ledger['previous_release_confirmed']
 research=pathlib.Path(ledger['research_directory']);used=sum(p.stat().st_size for p in research.rglob('*') if p.is_file())
 if used+2*1024**3>6*1024**3 or shutil.disk_usage(BASE).free<10*1024**3:raise RuntimeError('Full paired-result disk reserve unavailable')
 if ledger['seed']!=20261101 or ledger['additional_paired_seed_count']!=1 or ledger['fixed_updates']!=200 or ledger['max_cumulative_CU']!=MAX_CU or ledger['max_cumulative_GPU_seconds']!=MAX_GPU_SECONDS:raise RuntimeError('Repeat scope mismatch')
 prior=ledger['prior_CU_upper'];prior_gpu=ledger['prior_GPU_seconds_upper'];start=time.time();before=snapshot();atomic(ROOT/'before.json',before)
 if before['active_assignments'] or before['assignments'] or prior>=MAX_CU-.05:raise RuntimeError('Isolated remaining resource budget unavailable')
 name='deepfish-resume25-20261101-'+receipt['archive_sha256'][:10];remote='deepfish_resume25_'+receipt['archive_sha256'][:10]
 c={'authorization':{'upload':True,'model_run':True},'authorization_base_archive_sha256':'0fa965babadfaa83b9e7a83aadd81470fc76b566c118e7883ade3b0042ab2263','archive_sha256':receipt['archive_sha256'],'session':name,'allocation_started_epoch':start,'deadline_epoch':start+300,'absolute_limit_epoch':start+7200-prior_gpu,'prior_cu':prior,'prior_gpu_seconds':prior_gpu}
 c.update(seed=20261101,additional_paired_seed_count=1,fixed_updates=200,max_cumulative_CU=MAX_CU,max_cumulative_GPU_seconds=MAX_GPU_SECONDS)
 atomic(ROOT/'control.json',c);report={'status':'STARTING','seed':20261101,'additional_paired_seed_count':1,'post_result_exploratory_repeat':True,'prior_cu':prior,'prior_gpu_seconds':prior_gpu,'package_sha256':receipt['archive_sha256'],'allocation_attempts':0,'upload_bytes':0,'stage_receipts':[]};attempt=False;last_poll=0;max_rate=0;operation=0;stage_final_received=False
 local=ROOT/'artifacts';local.mkdir(exist_ok=False)
 def check_credit(force=False):
  nonlocal last_poll,max_rate
  if not attempt or not force and time.time()-last_poll<25:return
  s=snapshot();last_poll=time.time();max_rate=max(max_rate,s['rate_per_hour']);atomic(ROOT/'latest_usage.json',s)
  if not s['assignments'] and (ROOT/'final_download_receipt.json').exists():
   report.setdefault('allocation_release_observed_epoch',s['observed_epoch']);return
  a=json.loads((ROOT/'allocation_receipt.json').read_text()) if (ROOT/'allocation_receipt.json').exists() else None
  if a is not None and (len(s['assignments'])!=1 or s['assignments'][0]['endpoint']!=a['endpoint'] or s['assignments'][0]['accelerator']!='T4'):raise RuntimeError('Single owned T4 attribution changed')
  if max_rate>0:c['deadline_epoch']=min(c['deadline_epoch'],start+min(MAX_GPU_SECONDS-prior_gpu,(MAX_CU-prior)*3600/max_rate));atomic(ROOT/'control.json',c)
  spent=max(max_rate*(time.time()-start)/3600,max(0,before['balance']-s['balance']))
  if prior+spent>=MAX_CU-.02:raise TimeoutError('Cumulative credit boundary')
 def command(args,timeout,label,cleanup=False):
  nonlocal operation
  operation+=1;t=time.monotonic();log=ROOT/(str(operation).zfill(4)+'_'+label+'.log');lines=[]
  def line(s):
   with log.open('a') as f:f.write(s);f.flush();os.fsync(f.fileno())
   lines.append(s)
  def stop():
   if time.monotonic()-t>=timeout:return True
   if not cleanup:
    check_credit()
    if time.time()>=c['deadline_epoch']-90:return True
   return False
  result=bounded_stream([PYTHON,'-B',str(ROOT/'colab_guard.py'),*args],line,stop,env=dict(os.environ,DEEPFISH_ALLOCATION_RECEIPT=str(ROOT/'allocation_receipt.json')))
  atomic(ROOT/(str(operation).zfill(4)+'_'+label+'.json'),dict(result,arguments=args))
  if result['returncode'] or result['terminated_by_watchdog']:raise RuntimeError('CLI action incomplete: '+args[0])
  return ''.join(lines)
 def fetch_file(remote_path,path,row,label):
  path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.download')
  command(['download','--session',name,remote_path,str(temp)],90,label)
  if temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:raise ValueError('Remote download hash mismatch')
  with temp.open('rb') as f:os.fsync(f.fileno())
  os.replace(temp,path)
  if sha(path)!=row['sha256']:raise ValueError('Local readback mismatch')
 def collect_stage(data):
  nonlocal stage_final_received
  body={'phase':data['phase'],'files':data['files']};canonical=(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
  if hashlib.sha256(canonical).hexdigest()!=data['index_sha256']:raise ValueError('Stage index hash mismatch')
  if sum(r['bytes'] for r in data['files'])>2*1024**3:raise ValueError('Output reserve exceeded')
  for row in data['files']:
   rel=pathlib.PurePosixPath(row['path'])
   if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe artifact path')
   dest=local/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists() and dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']:continue
   temp=dest.with_suffix(dest.suffix+'.assembling');h=hashlib.sha256();total=0
   with temp.open('wb') as f:
    for part in row['parts']:
     if not re.fullmatch(r'[0-9a-f]{64}_[0-9]+\.bin',part['file']) or part['bytes']>8*1024**2:raise ValueError('Invalid transfer part')
     chunk=ROOT/'chunk.bin';fetch_file('content/'+remote+'/payload/transfer/'+part['file'],chunk,part,'artifact_chunk')
     b=chunk.read_bytes();f.write(b);h.update(b);total+=len(b);chunk.unlink()
    f.flush();os.fsync(f.fileno())
   if total!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Reassembled artifact mismatch')
   os.replace(temp,dest)
   if sha(dest)!=row['sha256']:raise ValueError('Atomic artifact readback mismatch')
  if data['phase']=='formal_B':
   check_credit(force=True)
   pair_receipt=ROOT/'LOCAL_PAIR_VERIFIED.json'
   audited=subprocess.run([LOCAL_CPU_PYTHON,'-B',str(ROOT/'verify_local_pair.py'),str(local),str(pair_receipt)],capture_output=True,text=True,timeout=60)
   atomic(ROOT/'local_pair_audit_execution.json',{'returncode':audited.returncode,'stdout':audited.stdout,'stderr':audited.stderr})
   if audited.returncode or not pair_receipt.exists():raise RuntimeError('Local safe final-pair audit failed before dev')
  ack={'phase':data['phase'],'index_sha256':data['index_sha256'],'all_files_atomically_persisted_and_hashed':True,'files':len(data['files']),'completed_epoch':time.time()};ackpath=ROOT/(data['phase']+'_download_receipt.json');atomic(ackpath,ack)
  command(['upload','--session',name,str(ackpath),'content/'+remote+'/payload/acks/'+data['phase']+'.json'],30,'stage_ack');report['upload_bytes']+=ackpath.stat().st_size
  report['stage_receipts'].append(ack);atomic(ROOT/'progress.json',report)
  print(json.dumps({'stage':'LOCAL_STAGE_PERSISTED','phase':data['phase'],'files':len(data['files']),'elapsed_GPU_seconds':time.time()-start}),flush=True)
  if data['phase']=='final':
   stage_final_received=True;request_final_release(ROOT)
 try:
  with (ROOT/'watchdog.log').open('xb') as f:g=subprocess.Popen([sys.executable,'-B',str(ROOT/'external_watchdog.py'),'--parent-pid',str(os.getpid())],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+45
  while not (ROOT/'watchdog_state.json').exists() and time.time()<until:
   if g.poll() is not None:raise RuntimeError('External watchdog failed')
   time.sleep(.2)
  gs=json.loads((ROOT/'watchdog_state.json').read_text());os.kill(gs['pid'],0);assert time.time()-gs['heartbeat_epoch']<5
  attempt=True;report['allocation_attempts']=1;command(['new','--gpu','T4','--session',name],120,'allocate')
  allocated=snapshot();assert len(allocated['assignments'])==1 and allocated['assignments'][0]['accelerator']=='T4' and allocated['rate_per_hour']>0
  max_rate=allocated['rate_per_hour'];c['deadline_epoch']=start+min(MAX_GPU_SECONDS-prior_gpu,(MAX_CU-prior)*3600/max_rate);atomic(ROOT/'control.json',c);report['live_rate']=max_rate;atomic(ROOT/'allocated_usage.json',allocated)
  probe=ROOT/'remote_probe.py';probe.write_text("import pathlib,hashlib,json,os\np=pathlib.Path('/content/"+remote+"');p.mkdir(exist_ok=False);b=(b'DeepFish public non-sensitive persistence marker v1\\n'*200000)[:8388608];f=(p/'marker.bin').open('wb');f.write(b);f.flush();os.fsync(f.fileno());f.close();print('DEEPFISH_PROBE '+json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}))\n")
  output=command(['exec','--session',name,'--file',str(probe),'--timeout','30'],45,'probe');report['upload_bytes']+=probe.stat().st_size;row=json.loads(next(x[len('DEEPFISH_PROBE '):] for x in output.splitlines() if x.startswith('DEEPFISH_PROBE ')))
  t=time.monotonic();fetch_file('content/'+remote+'/marker.bin',ROOT/'marker.bin',row,'marker');bps=row['bytes']/(time.monotonic()-t);proof={'remote_sha256':row['sha256'],'local_sha256':sha(ROOT/'marker.bin'),'bytes':row['bytes'],'download_bytes_per_second':bps,'verified_epoch':time.time()};atomic(ROOT/'persistence_probe.json',proof)
  check_credit(force=True)
  budget_gate=complete_repeat_budget(prior,prior_gpu,max_rate,time.time()-start,bps);atomic(ROOT/'COMPLETE_COST_GATE.json',budget_gate)
  print(json.dumps({'stage':'EARLY_FULL_COST_GATE','status':budget_gate['status'],'required_seconds':budget_gate['required_seconds'],'remaining_seconds':budget_gate['remaining_seconds'],'download_bytes_per_second':bps}),flush=True)
  if budget_gate['status']!='FITS':raise ResourceUndetermined('Measured marker bandwidth and live rate cannot fit full U200 before dependency preparation.')
  parts=[]
  with archive.open('rb') as f:
   for i in range(3):
    b=f.read(6*1024**2)
    if not b:break
    p=ROOT/('package_part'+str(i)+'.bin');p.write_bytes(b);parts.append({'file':p.name,'bytes':len(b),'sha256':sha(p)});command(['upload','--session',name,str(p),'content/'+remote+'/'+p.name],90,'upload_package');report['upload_bytes']+=len(b);p.unlink()
  context={'authorization':{'upload':True,'model_run':True},'allocation_started_epoch':start,'cu_per_hour_upper_bound':max_rate,'quote_verified':True,'external_release_guard_verified':True,'persistent_storage_verified':True,'persistence_mode':'local_cli_sha256','persistence_probe':proof,'prior_cu':prior,'prior_gpu_seconds':prior_gpu,'artifact_dir':'/content/'+remote+'/artifacts','expected_manifest_sha256':receipt['manifest_sha256']}
  launch={'remote':remote,'parts':parts,'archive_sha256':receipt['archive_sha256'],'archive_bytes':receipt['archive_bytes'],'context':context}
  launcher=ROOT/'remote_launch.py';launcher.write_text('import pathlib,json,hashlib,zipfile,subprocess,sys,os\nCONFIG='+repr(launch)+'''
base=pathlib.Path('/content')/CONFIG['remote'];marker=base/'launch_entered.json';marker.write_text(json.dumps({'status':'LAUNCH_ENTERED','archive_sha256':CONFIG['archive_sha256']}));print('DEEPFISH_LAUNCH_ENTERED '+CONFIG['archive_sha256'],flush=True);archive=base/'package.zip';h=hashlib.sha256()
with archive.open('wb') as out:
 for part in CONFIG['parts']:
  p=base/part['file'];b=p.read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'];out.write(b);h.update(b)
assert h.hexdigest()==CONFIG['archive_sha256'] and archive.stat().st_size==CONFIG['archive_bytes']
payload=base/'payload';payload.mkdir();(base/'artifacts').parent.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  path=pathlib.PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
  target=payload/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
(payload/'acks').mkdir();ctx=base/'run_context.json';ctx.write_text(json.dumps(CONFIG['context']))
p=subprocess.Popen([sys.executable,'-B',str(payload/'driver.py'),'--run-context',str(ctx)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
for line in p.stdout:print(line,end='',flush=True)
print('DEEPFISH_DRIVER_EXIT '+str(p.wait()),flush=True)
''');report['upload_bytes']+=launcher.stat().st_size
  if report['upload_bytes']>13*1024**2:raise ValueError('Input upload scope exceeded')
  print(json.dumps({'stage':'GPU_PACKAGE_UPLOADED_AND_PERSISTENCE_VERIFIED','rate':max_rate,'download_bytes_per_second':bps,'remaining_CU':MAX_CU-prior}),flush=True)
  execution_log=ROOT/'execution.stdout.log'
  def on_line(line):
   with execution_log.open('a') as f:f.write(line);f.flush();os.fsync(f.fileno())
   if line.strip().startswith('{'):
    try:failure=json.loads(line)
    except json.JSONDecodeError:failure={}
    if failure.get('status')=='CLI_FAILED':
     atomic(ROOT/'cli_failure.json',failure)
     raise RuntimeError('CLI failure received; launch is not replayed; release owned allocation and inspect exact safe diagnostics.')
   if line.startswith('DEEPFISH_EVIDENCE '):
    outer=json.loads(line[len('DEEPFISH_EVIDENCE '):]);b=base64.b64decode(outer['base64'],validate=True)
    if len(b)!=outer['bytes'] or hashlib.sha256(b).hexdigest()!=outer['sha256']:raise ValueError('Evidence frame integrity failure')
    value=json.loads(b)
    if outer['name']=='stage_ready':collect_stage(value)
    else:atomic(ROOT/(outer['name']+'.json'),value)
  def stop_execution():
   check_credit()
   return report.get('allocation_release_observed_epoch') is not None or time.time()>=c['deadline_epoch']-90
  result=bounded_stream([PYTHON,'-B',str(ROOT/'colab_guard.py'),'exec','--session',name,'--file',str(launcher),'--timeout',str(max(1,c['deadline_epoch']-time.time()-90))],on_line,stop_execution,env=dict(os.environ,PYTHONUNBUFFERED='1'))
  report['execution_transport']=result
  if not stage_final_received:raise RuntimeError('Final local artifact receipt is missing')
  saved=json.loads((local/'run_report.json').read_text());report['remote_result']=saved;report['status']='COMPLETE' if saved['status']=='COMPLETE' else saved['status'];report['all_artifacts_persisted_before_release']=True
 except BaseException as error:
  if (ROOT/'final_download_receipt.json').exists() and (local/'run_report.json').exists() and json.loads((local/'run_report.json').read_text()).get('status')=='COMPLETE':report.update(status='COMPLETE',remote_result=json.loads((local/'run_report.json').read_text()),all_artifacts_persisted_before_release=True,transport_warning=type(error).__name__+': '+str(error))
  else:report.update(status='RESOURCE_UNDETERMINED' if isinstance(error,ResourceUndetermined) else 'INCOMPLETE',error_type=type(error).__name__,error=str(error))
 finally:
  if attempt:report.update(release_owned(c))
  else:report['release']='NOT_ALLOCATED'
  if report['release'] in ('CONFIRMED','NOT_ALLOCATED'):atomic(ROOT/'RELEASE_CONFIRMED.json',{'release':report['release']})
  try:final=snapshot();report['final_sessions_usage']=final
  except Exception as error:report['final_sessions_usage_error']=type(error).__name__
  elapsed=max(0,report.get('allocation_release_observed_epoch',time.time())-start) if attempt else 0;report.update(allocation_wall_seconds_upper=elapsed,cumulative_GPU_seconds_upper=prior_gpu+elapsed,cumulative_CU_upper=prior+max(max_rate*elapsed/3600,max(0,before['balance']-report.get('final_sessions_usage',before)['balance'])),max_cumulative_CU=MAX_CU,max_cumulative_GPU_seconds=MAX_GPU_SECONDS,scientific_configuration_changed=False,scientific_changes=[],seed_relative_to_original=20261101,prior_seed=20261031,budget_revision='cumulative2CU_to2.5CU; previous repeat scientific configuration unchanged',fixed_U200=True)
  atomic(ROOT/'controller_report.json',report);print(json.dumps({k:report.get(k) for k in ['status','release','cumulative_CU_upper','cumulative_GPU_seconds_upper','error']}),flush=True)
 return 0 if report['status']=='COMPLETE' and report['release']=='CONFIRMED' else 2
if __name__=='__main__':raise SystemExit(main())
