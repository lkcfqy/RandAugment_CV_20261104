"""Existing official Colab CLI only; interactive/new OAuth is disabled.

Snapshots deliberately expose only rate/balance/session identity, never credentials.
An allocation receipt is written immediately when the single assign call returns.
"""
import json,os,sys
from pathlib import Path
from contracts import CONTROL_PREFIX

def main():
    os.environ['COLAB_CLI_ASSIGN_RETRIES']='0'
    os.environ['COLAB_CLI_REQUEST_TIMEOUT']='15'
    import colab_cli.auth as auth
    def no_oauth(*args,**kwargs):raise RuntimeError('Existing Colab authentication required; new OAuth disabled')
    auth._run_remote_flow=no_oauth
    from colab_cli.common import state
    args=sys.argv[1:]
    try:
        if args==['--snapshot']:
            info=state.client.get_consumption_user_info();sessions,assignments=state.sync_sessions()
            names={x.endpoint:x.name for x in sessions.values()}
            data={'balance':info.paid_compute_units_balance,'rate_per_hour':info.consumption_rate_hourly,'active_assignments':info.assignments_count,'assignments':[{'name':names.get(a.endpoint),'endpoint':a.endpoint,'accelerator':a.accelerator.value} for a in assignments]}
            print('\n'+CONTROL_PREFIX+json.dumps(data,sort_keys=True),flush=True);return 0
        if args[:1]==['--release-receipt']:
            receipt=json.loads(Path(args[1]).read_text())
            state.client.unassign(receipt['endpoint'])
            print('Released the endpoint in the allocation receipt.',flush=True);return 0
        if args[:1]==['new']:
            receipt_path=Path(os.environ['CUE_ALLOCATION_RECEIPT'])
            original=state.client.assign
            calls=0
            def assign_once(*a,**kw):
                nonlocal calls
                calls+=1
                if calls!=1:raise RuntimeError('second assignment attempt forbidden')
                result=original(*a,**kw)
                data={'endpoint':result.endpoint,'session_name':args[args.index('--session')+1],'accelerator':'T4'}
                with receipt_path.open('x') as stream:
                    json.dump(data,stream);stream.flush();os.fsync(stream.fileno())
                return result
            state.client.assign=assign_once
        from colab_cli.cli import app
        app(args=args,standalone_mode=False);return 0
    except BaseException as error:
        code=getattr(error,'exit_code',None)
        if code is not None:return int(code)
        print('Colab command blocked/failed: '+type(error).__name__,file=sys.stderr,flush=True)
        return 2

if __name__=='__main__':raise SystemExit(main())
