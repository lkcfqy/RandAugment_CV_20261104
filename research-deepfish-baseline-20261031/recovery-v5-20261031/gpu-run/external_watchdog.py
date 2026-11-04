"""Independent finite external release guard; no allocation, upload or model operations."""
import argparse,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
PYTHON='/Users/lkc/.local/share/google-colab-cli/bin/python'
def atomic(path,value):
 p=pathlib.Path(path);tmp=p.with_suffix('.tmp');data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 if p.read_bytes()!=data:raise IOError('Local persistence readback mismatch')
def cli(args,timeout=20):
 r=subprocess.run([PYTHON,'-B',str(ROOT/'colab_guard.py'),*args],capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError('CLI '+args[0]+' failed: '+str(r.returncode))
 return r.stdout
def snapshot():
 s=cli(['--snapshot'],45)
 return json.loads(next(line[len('DEEPFISH_CONTROL '):] for line in s.splitlines() if line.startswith('DEEPFISH_CONTROL ')))
def release_owned(control):
 receipt=ROOT/'allocation_receipt.json';endpoint=json.loads(receipt.read_text())['endpoint'] if receipt.exists() else None;errors=[]
 def own(s):return [x for x in s['assignments'] if x['name']==control['session'] or endpoint is not None and x['endpoint']==endpoint]
 try:
  before=snapshot()
  if not own(before):return {'release':'CONFIRMED' if not before['assignments'] or endpoint is not None else 'UNCERTAIN','after_release':before,'errors':errors}
  for attempt in range(2):
   try:cli(['--release-receipt',str(receipt)] if endpoint else ['stop','--session',control['session']],20)
   except Exception as e:errors.append(type(e).__name__)
   after=snapshot()
   if not own(after):return {'release':'CONFIRMED','after_release':after,'errors':errors}
  return {'release':'UNCERTAIN','after_release':after,'errors':errors}
 except Exception as e:return {'release':'UNCERTAIN','errors':errors+[type(e).__name__]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-pid',type=int,required=True);a=p.parse_args();c=json.loads((ROOT/'control.json').read_text())
 if c['authorization']!={'upload':True,'model_run':True} or c['authorization_base_archive_sha256']!='0fa965babadfaa83b9e7a83aadd81470fc76b566c118e7883ade3b0042ab2263':raise ValueError('Guard scope mismatch')
 if c['archive_sha256']!=json.loads((ROOT.parent/'reports/PACKAGE_RECEIPT.json').read_text())['archive_sha256']:raise ValueError('Current archive receipt mismatch')
 before=snapshot();atomic(ROOT/'watchdog_state.json',{'status':'ARMED','pid':os.getpid(),'heartbeat_epoch':time.time(),'independent_snapshot':before,'deadline_epoch':c['deadline_epoch']})
 while time.time()<c['absolute_limit_epoch']:
  if (ROOT/'RELEASE_CONFIRMED.json').exists():return 0
  new=json.loads((ROOT/'control.json').read_text())
  if new['deadline_epoch']>c['absolute_limit_epoch']:raise ValueError('Deadline extension past two-hour cap')
  c=new
  try:os.kill(a.parent_pid,0);alive=True
  except ProcessLookupError:alive=False
  if not alive or time.time()>=c['deadline_epoch']-90 or (ROOT/'REQUEST_RELEASE').exists():
   result=release_owned(c);atomic(ROOT/'watchdog_release.json',dict(result,trigger='PARENT_EXITED' if not alive else 'DEADLINE_OR_REQUEST',finished_epoch=time.time()))
   if result['release']=='CONFIRMED':atomic(ROOT/'RELEASE_CONFIRMED.json',result)
   return 0 if result['release']=='CONFIRMED' else 2
  atomic(ROOT/'watchdog_state.json',{'status':'ARMED','pid':os.getpid(),'heartbeat_epoch':time.time(),'deadline_epoch':c['deadline_epoch']});time.sleep(1)
 return 2
if __name__=='__main__':raise SystemExit(main())
