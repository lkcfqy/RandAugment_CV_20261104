"""Small adaptations of prior atomic evidence, finite expiry and owned-release semantics."""
import argparse,base64,hashlib,importlib.util,json,os,pathlib,subprocess,sys,tempfile,time
from contracts import ContractError,NotAuthorized,positive,validate_context
def atomic_bytes(path,data):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',delete=False) as stream:
  temp=pathlib.Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
 os.replace(temp,path)
 actual=path.read_bytes()
 if actual!=data:raise IOError('Persistence readback mismatch')
 return {'bytes':len(data),'sha256':hashlib.sha256(actual).hexdigest()}
def atomic_json(path,value):return atomic_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def emit_evidence(name,value):
 data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
 print('DEEPFISH_EVIDENCE '+json.dumps({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}),flush=True)
def release_owned(command,snapshot,endpoint,name):
 """External controller helper only: one release request then reconciliation."""
 errors=[];request=['release-owned',endpoint] if endpoint else ['stop','--session',name]
 try:command(request,20)
 except Exception as error:errors.append('initial release: '+str(error))
 try:state=snapshot()
 except Exception as error:return {'release':'UNCERTAIN','errors':errors+['reconciliation: '+str(error)]}
 own=lambda rows:[a for a in rows if a['name']==name or endpoint is not None and a['endpoint']==endpoint]
 if own(state['assignments']) or endpoint is None and state['assignments']:return {'release':'UNCERTAIN','errors':errors,'after_release':state}
 return {'release':'CONFIRMED','errors':errors,'after_release':state}
def native_unassign():
 """Future Colab only; absence/failure never means confirmed release."""
 try:
  from google.colab import runtime
  runtime.unassign()
  return {'release':'REQUEST_RETURNED_UNCONFIRMED'}
 except BaseException as error:return {'release':'UNCERTAIN','error_type':type(error).__name__}
def validate_expiry_context(ctx,now,monotonic_now):
 budget=validate_context(ctx,now)
 hard=ctx.get('hard_deadline_monotonic');work=ctx.get('work_deadline_monotonic')
 if not positive(hard) or not positive(work) or abs((hard-monotonic_now)-(budget['hard_deadline_epoch']-now))>5 or abs(hard-work-180)>.01:raise ContractError('Expiry deadline differs from verified allocation budget.')
 return hard
def expiry_watch(context_path):
 # No Colab lookup, persistence, wait or release before this independent gate.
 ctx=json.loads(pathlib.Path(context_path).read_text());deadline=validate_expiry_context(ctx,time.time(),time.monotonic());state=pathlib.Path(ctx['artifact_dir'])/'native_expiry.json'
 try:
  spec=importlib.util.find_spec('google.colab.runtime')
  if spec is None or not os.environ.get('TBE_RUNTIME_ADDR'):raise RuntimeError('Native runtime management unavailable; external guard remains necessary.')
  atomic_json(state,{'status':'ARMED','deadline_monotonic':deadline,'runtime_source_sha256':hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest()})
  while time.monotonic()<deadline-12:time.sleep(min(1,max(0,deadline-12-time.monotonic())))
  atomic_json(state,{'status':'UNASSIGN_REQUESTED'})
  result=subprocess.run([sys.executable,'-c','from google.colab import runtime; runtime.unassign()'],timeout=max(.1,min(10,deadline-time.monotonic())),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  atomic_json(state,{'status':'REQUEST_RETURNED_UNCONFIRMED','returncode':result.returncode})
 except BaseException as error:atomic_json(state,{'status':'UNASSIGN_UNCONFIRMED','error_type':type(error).__name__})
def expiry_main():
 p=argparse.ArgumentParser();p.add_argument('--context',type=pathlib.Path,required=True);a=p.parse_args()
 try:expiry_watch(a.context);return 0
 except NotAuthorized as error:print(json.dumps({'status':'NOT_AUTHORIZED','reason':str(error)}));return 3
 except (ContractError,OSError,ValueError) as error:print(json.dumps({'status':'RESOURCE_UNDETERMINED','reason':str(error)}));return 4
if __name__=='__main__':raise SystemExit(expiry_main())
