"""Bounded independent local reclamation using existing official CLI credentials."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from evidence import atomic_bytes
from contracts import canonical,parse_control_frame

def release_owned(command,snapshot,receipt,name):
    """A failed stop never skips reconciliation. Retry only a confirmed own endpoint."""
    errors=[];endpoint=json.loads(receipt.read_text())['endpoint'] if receipt.exists() else None
    request=['--release-receipt',str(receipt)] if endpoint else ['stop','--session',name]
    try:command(request,20)
    except Exception as error:errors.append('initial release: '+str(error))
    try:state=snapshot()
    except Exception as error:return {'release':'UNCERTAIN','errors':errors+['reconciliation: '+str(error)]}
    def own(rows):return [a for a in rows if a['name']==name or endpoint and a['endpoint']==endpoint]
    if own(state['assignments']):
        try:command(request,15)
        except Exception as error:errors.append('confirmed-active release: '+str(error))
        try:state=snapshot()
        except Exception as error:return {'release':'UNCERTAIN','errors':errors+['final reconciliation: '+str(error)]}
    if own(state['assignments']):return {'release':'UNCERTAIN','errors':errors+['owned endpoint remains active'],'after_release':state}
    if endpoint is None and state['assignments']:return {'release':'UNCERTAIN','errors':errors+['assignment without receipt cannot safely be attributed'],'after_release':state}
    return {'release':'CONFIRMED','errors':errors,'after_release':state}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--parent-pid',type=int,required=True);parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--session',required=True)
    args=parser.parse_args();run=args.run;guard=Path(__file__).with_name('colab_guard.py');python='/Users/lkc/.local/share/google-colab-cli/bin/python'
    def alive():
        try:os.kill(args.parent_pid,0);return True
        except ProcessLookupError:return False
    def command(arguments,timeout):
        remaining=args.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('watchdog finite lifetime reached')
        process=subprocess.run([python,str(guard),*arguments],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=min(timeout,remaining),env=dict(os.environ,PYTHONUNBUFFERED='1'))
        print(process.stdout,end='',flush=True)
        if process.returncode:raise RuntimeError('official CLI '+arguments[0]+' failed')
        return process.stdout
    def snapshot():return parse_control_frame(command(['--snapshot'],15))
    status={'status':'WAITING','finite_deadline_monotonic':args.deadline,'new_authentication':False}
    atomic_bytes(run/'release_watchdog.json',canonical(status))
    while time.monotonic()<args.deadline:
        if (run/'RELEASE_CONFIRMED.json').exists():return 0
        orphaned=not alive();due=time.monotonic()>=args.deadline-90
        receipt=run/'allocation_receipt.json'
        if due or orphaned and receipt.exists():
            status.update(status='RELEASING',trigger='PARENT_EXITED' if orphaned else 'ALLOCATION_DEADLINE')
            atomic_bytes(run/'release_watchdog.json',canonical(status))
            try:status.update(release_owned(command,snapshot,receipt,args.session))
            except Exception as error:status.update(release='UNCERTAIN',error_type=type(error).__name__)
            atomic_bytes(run/'release_watchdog.json',canonical(status))
            if status.get('release')=='CONFIRMED':return 0
            # One later reconciliation at the deadline, never a blind repeated unassign.
            while time.monotonic()<args.deadline-90:
                if (run/'RELEASE_CONFIRMED.json').exists():return 0
                time.sleep(min(1,max(0,args.deadline-90-time.monotonic())))
            try:
                observed=snapshot();endpoint=json.loads(receipt.read_text())['endpoint'] if receipt.exists() else None
                known=[a for a in observed['assignments'] if a['name']==args.session or endpoint and a['endpoint']==endpoint]
                if known:status.update(release_owned(command,snapshot,receipt,args.session))
                elif not observed['assignments'] or endpoint is not None:status.update(release='CONFIRMED',after_release=observed)
                else:status.update(release='UNCERTAIN',error='unattributed allocation')
            except Exception as error:status.update(release='UNCERTAIN',final_reconciliation_error=type(error).__name__)
            atomic_bytes(run/'release_watchdog.json',canonical(status));return 0 if status.get('release')=='CONFIRMED' else 2
        time.sleep(min(1,max(0,args.deadline-time.monotonic())))
    return 2

if __name__=='__main__':raise SystemExit(main())
