"""Installed official Colab CLI, with interactive OAuth and assignment retries disabled."""
import hashlib,json,os,pathlib,re,signal,sys,time,traceback
os.environ['COLAB_CLI_ASSIGN_RETRIES']='0'
os.environ['COLAB_CLI_REQUEST_TIMEOUT']='15'
PREFIX='DEEPFISH_CONTROL '
def main():
 import colab_cli.auth as auth
 def no_oauth(*args,**kwargs):raise RuntimeError('Existing authentication unavailable; new OAuth disabled')
 auth._run_remote_flow=no_oauth
 from colab_cli.common import state
 args=sys.argv[1:]
 try:
  if args==['--snapshot']:
   signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('snapshot deadline')));signal.alarm(40)
   info=state.client.get_consumption_user_info();sessions,assignments=state.sync_sessions();names={s.endpoint:s.name for s in sessions.values()}
   result={'observed_epoch':time.time(),'balance':info.paid_compute_units_balance,'rate_per_hour':info.consumption_rate_hourly,'active_assignments':info.assignments_count,'assignments':[{'name':names.get(a.endpoint),'endpoint':a.endpoint,'accelerator':a.accelerator.value} for a in assignments]}
   signal.alarm(0);print('\n'+PREFIX+json.dumps(result,sort_keys=True),flush=True);return 0
  if args[:1]==['--release-receipt']:
   receipt=json.loads(pathlib.Path(args[1]).read_text());state.client.unassign(receipt['endpoint']);print('Owned release requested; reconciliation still required.',flush=True);return 0
  if args[:1]==['new']:
   receipt=pathlib.Path(os.environ['DEEPFISH_ALLOCATION_RECEIPT']);original=state.client.assign;calls=0
   def once(*a,**kw):
    nonlocal calls
    calls+=1
    if calls!=1:raise RuntimeError('Second allocation attempt forbidden')
    result=original(*a,**kw)
    with receipt.open('x') as f:json.dump({'endpoint':result.endpoint,'session_name':args[args.index('--session')+1],'accelerator':args[args.index('--gpu')+1] if '--gpu' in args else 'NONE','assigned_epoch':time.time()},f);f.flush();os.fsync(f.fileno())
    return result
   state.client.assign=once
  from colab_cli.cli import app
  app(args=args,standalone_mode=False);return 0
 except BaseException as error:
  code=getattr(error,'exit_code',None)
  if code is not None:return int(code)
  indicators=sorted(set(re.findall(r'\b(?:401|403|404|429|500|502|503|invalid_grant|invalid_client|invalid_scope|access_denied)\b',str(error))))
  message=str(error)
  known={'IOPub channel must be running to receive output','stdin channel must be running to allow input','Connection was lost.','Kernel died before replying to kernel_info','You must first start a kernel.','You must first start a kernel before requesting a client.','A kernel is already started. Shutdown it before starting a new one.'}
  safe_message=message if message in known or re.fullmatch(r"Kernel didn't respond(?: to heartbeats)? in [0-9.]+ seconds(?: and timed out)?",message) else None
  frames=[{'file':pathlib.Path(x.filename).name,'function':x.name,'line':x.lineno} for x in traceback.extract_tb(error.__traceback__)]
  print(json.dumps({'status':'CLI_FAILED','error_type':type(error).__name__,'known_safe_message':safe_message,'error_message_sha256':hashlib.sha256(message.encode()).hexdigest(),'stack_locations':frames,'safe_indicators':indicators,'interactive_oauth_enabled':False}),flush=True);return 2
if __name__=='__main__':
 code=main();sys.stdout.flush();sys.stderr.flush()
 if code:os._exit(code)  # Failed CLI startup may leave websocket threads alive.
 raise SystemExit(code)
