"""Local one-allocation controller; default --check never contacts Colab.

Run requires an externally supplied explicit GO bound to FREEZE.json. No retries,
second allocation, alternative accelerator, new OAuth, or post-release retrieval.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from contracts import canonical,validate_go,credit_gate,complete_result_set,make_packet,validate_artifact_path,bound_metric_inputs
from evidence import atomic_bytes,owned_size
from transport import Collector,bounded_stream
from release_guard import release_owned

BASE=Path(__file__).resolve().parents[2]
CLI_PYTHON=Path('/Users/lkc/.local/share/google-colab-cli/bin/python')

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify_freeze(base):
    freeze_path=base/'FREEZE.json';freeze=json.loads(freeze_path.read_text())
    for rel,item in freeze['files'].items():
        path=base/validate_artifact_path(rel)
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('freeze mismatch: '+rel)
    return freeze,sha(freeze_path)

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--authorization',type=Path);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);verify_start=time.monotonic()
    freeze,freeze_sha=verify_freeze(BASE);local_prelaunch_compute=time.monotonic()-verify_start
    if not args.run:
        print(json.dumps({'status':'LOCAL_FREEZE_VERIFIED_PENDING_GO','freeze_sha256':freeze_sha,'allocation_attempts':0,'model_forwards':0,'bundle_bytes':freeze['bundle']['bytes'],'local_verification_seconds':local_prelaunch_compute},indent=2));return 0
    if args.authorization is None:raise ValueError('Explicit GO authorization file is required')
    authorization=json.loads(args.authorization.read_text());validate_go(authorization,freeze_sha)
    if owned_size(BASE.parent)+768*1024**2>6*1024**3:raise RuntimeError('local owned disk cap lacks the bounded output/transfer reserve')
    run=BASE/'run';run.mkdir(exist_ok=True);collector=Collector(run);allocation_start=None;new_attempted=False
    name='counting-fixed-'+freeze_sha[:12];remote_name='cue_'+freeze_sha[:12];receipt=run/'allocation_receipt.json'
    guard=BASE/'payload/scripts/colab_guard.py';command_index=0;last_poll=0.;initial_balance=None;max_rate=0.;report={'status':'STARTING','freeze_sha256':freeze_sha,'allocation_attempts':0,'local_prelaunch_compute_seconds':local_prelaunch_compute,'session':name,'retrieval_after_release':False,'new_authentication':False}
    abort=None;allocated=False
    def elapsed():return 0. if allocation_start is None else time.monotonic()-allocation_start
    def command(arguments,timeout=30,*,poll_credit=False,capture=True,cleanup=False):
        nonlocal command_index
        command_index+=1;start=time.monotonic();lines=[]
        environment=os.environ.copy();environment.update(PYTHONUNBUFFERED='1',CUE_ALLOCATION_RECEIPT=str(receipt))
        def stop():
            if time.monotonic()-start>=timeout:return True
            if allocation_start is not None and elapsed()>=(2700 if cleanup else 2610):return True
            if poll_credit:watch_credit()
            return abort is not None and not cleanup
        def line(value):
            collector.line(value)
            if capture:lines.append(value)
        result=bounded_stream([str(CLI_PYTHON),str(guard),*arguments],line,stop,env=environment)
        atomic_bytes(run/('command-%03d.json'%command_index),canonical({'arguments':arguments,'elapsed_allocation_seconds':elapsed(),**result}))
        if result['terminated_by_watchdog'] or result['returncode']!=0:raise RuntimeError('CLI operation failed/uncertain: '+arguments[0])
        return ''.join(lines)
    def snapshot(cleanup=False):return json.loads(command(['--snapshot'],timeout=15,cleanup=cleanup))
    def watch_credit(force=False):
        nonlocal last_poll,max_rate,abort
        if not force and time.monotonic()-last_poll<30:return
        last_poll=time.monotonic()
        try:
            state=snapshot();max_rate=max(max_rate,state['rate_per_hour'])
            spent=max(0.,initial_balance-state['balance'],max_rate*elapsed()/3600)
            gate=credit_gate(max_rate,spent,max(0.,2700-elapsed()),2.)
            own=[a for a in state['assignments'] if a['name']==name]
            if state['active_assignments']!=1 or len(own)!=1 or own[0]['accelerator']!='T4':raise RuntimeError('single owned T4 identity no longer confirmed')
            if not gate['allowed']:raise RuntimeError('actual rate/credit forecast exceeds the strict 2 CU cap')
            report['latest_usage']={'snapshot':state,'elapsed_allocation_seconds':elapsed(),'spent_upper_estimate_CU':spent,'credit_gate':gate}
            atomic_bytes(run/'usage_latest.json',canonical(report['latest_usage']))
        except Exception as error:abort=str(error)
    try:
        before=snapshot();report['before_allocation']=before;initial_balance=before['balance']
        if before['active_assignments']!=0 or before['assignments']:raise RuntimeError('existing active assignments prevent isolated rate/cost attribution')
        if initial_balance<2:raise RuntimeError('less than the authorized 2 CU reserve is available')
        # Exclusive claim is deliberately never removed, including failed allocations.
        with (run/'ALLOCATION_CLAIM.json').open('x') as stream:
            json.dump({'freeze_sha256':freeze_sha,'session':name,'scope':authorization['scope']},stream);stream.flush();os.fsync(stream.fileno())
        allocation_start=time.monotonic()
        with (run/'release_watchdog.log').open('xb') as log:
            watchdog=subprocess.Popen([sys.executable,str(BASE/'payload/scripts/release_guard.py'),'--run',str(run),'--parent-pid',str(os.getpid()),'--deadline',str(allocation_start+2700),'--session',name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report['independent_release_watchdog_pid']=watchdog.pid
        new_attempted=True;report['allocation_attempts']=1
        command(['new','--gpu','T4','--session',name],timeout=180)
        allocated=True;watch_credit(force=True)
        if abort:raise RuntimeError(abort)
        mkdir_script=run/'remote_mkdir.py'
        atomic_bytes(mkdir_script,('from pathlib import Path\nPath('+repr('/content/'+remote_name)+').mkdir(exist_ok=False)\n').encode())
        command(['exec','--session',name,'--file',str(mkdir_script),'--timeout','30'],timeout=40,poll_credit=True)
        for part in freeze['bundle']['parts']:
            command(['upload','--session',name,str(BASE/part['path']),'content/'+remote_name+'/'+Path(part['path']).name],timeout=90,poll_credit=True,capture=False)
        launch={
            'remote_name':remote_name,'parts':[{'filename':Path(p['path']).name,'bytes':p['bytes'],'sha256':p['sha256']} for p in freeze['bundle']['parts']],
            'bundle_sha256':freeze['bundle']['sha256'],'bundle_bytes':freeze['bundle']['bytes'],'payload_bytes':freeze['payload_bytes'],
            'package_manifest_sha256':freeze['package_manifest_sha256'],'allocation_seconds_left':2700-elapsed(),
            'local_prelaunch_compute_seconds':local_prelaunch_compute,'previous_compute_wall_seconds':freeze['prior_compute_wall_seconds']}
        template=(BASE/'payload/scripts/launch_template.py').read_text()
        script=template.replace('__CUE_LAUNCH_CONFIG__',repr(launch));launch_file=run/'remote_launch.py';atomic_bytes(launch_file,script.encode())
        execute_timeout=max(1.,2610-elapsed())
        command(['exec','--session',name,'--file',str(launch_file),'--timeout',str(execute_timeout)],timeout=execute_timeout,poll_credit=True,capture=False)
        # Always collect while allocation still exists. Streamed files normally suffice.
        collection_start=time.monotonic();artifact_root=collector.artifacts
        remote_output='content/'+remote_name+'/payload/output/'
        index_path=artifact_root/'artifact_index.json'
        if not index_path.exists():
            index_download=run/'artifact_index.download.json';command(['download','--session',name,remote_output+'artifact_index.json',str(index_download)],timeout=30,poll_credit=True)
            collector.line(make_packet('artifact_index.json',index_download.read_bytes()))
        index=json.loads(index_path.read_text());downloaded=0
        for rel in collector.missing(index):
            item=index['files'][rel]
            if item['bytes']>8*1024**2:raise RuntimeError('unexpected oversized evidence file')
            downloaded+=item['bytes']
            if downloaded>256*1024**2:raise RuntimeError('evidence transfer reserve exceeded')
            temp=run/'retrieval.tmp';command(['download','--session',name,remote_output+rel,str(temp)],timeout=30,poll_credit=True)
            collector.line(make_packet(rel,temp.read_bytes()));temp.unlink()
        if collector.missing(index):raise RuntimeError('evidence incomplete before release')
        report['all_indexed_artifacts_persisted_before_release']=True;report['artifact_files']=len(index['files'])
        supervisor=json.loads((artifact_root/'supervisor.json').read_text());report['remote_supervisor']=supervisor
        completion=json.loads((artifact_root/'completion.json').read_text()) if (artifact_root/'completion.json').exists() else {'complete':False}
        if supervisor['status']=='WORKER_COMPLETE' and completion['complete']:
            from metrics import summarize
            queries=json.loads((BASE/'payload/science/queries_frozen.json').read_text())['queries']
            labels=json.loads((BASE/'payload/science/labels_frozen.json').read_text())['pairs']
            rows=[json.loads((artifact_root/rel).read_text()) for rel in sorted(index['files']) if rel.startswith('results/')]
            complete_result_set(rows,{q['query_id']:q['query_sha256'] for q in queries})
            grouped={}
            for row in rows:
                pred=row['prediction']
                if pred['count']!=len(pred['boxes']) or pred['count']!=len(pred['scores']):raise RuntimeError('saved boxes/count/scores disagree')
                grouped.setdefault(row['filename'],{}).setdefault(row['mode'],{})[row['target']]=pred['count']
            pairs,policy=bound_metric_inputs(grouped,labels);recomputed=summarize(pairs,policy)
            remote_summary=json.loads((artifact_root/'science/summary.json').read_text())
            if any(remote_summary[k]!=v for k,v in recomputed.items()):raise RuntimeError('local label-bound exact summary differs')
            report['status']='COMPLETE_492_PENDING_PREDICTED_BOX_REVIEW'
        else:report['status']='ENGINEERING_BLOCKED_OR_FORMAL_INCOMPLETE'
        report['local_collection_validation_wall_seconds']=time.monotonic()-collection_start
        report['compute_used_seconds_including_local_validation']=index.get('compute_used_seconds_after_artifact_hashing',supervisor['compute_used_seconds'])+report['local_collection_validation_wall_seconds']
        if report['compute_used_seconds_including_local_validation']>1800:raise RuntimeError('total compute including local validation exceeds cap')
    except BaseException as error:
        report.update(status='BLOCKED_OR_INCOMPLETE',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        # Even failed allocation/setup/precheck is released. No second assign call exists.
        if new_attempted:
            try:
                recovery=release_owned(lambda a,t:command(a,timeout=t,cleanup=True),lambda:snapshot(cleanup=True),receipt,name)
                report.update(recovery)
                if recovery['release']=='CONFIRMED':atomic_bytes(run/'RELEASE_CONFIRMED.json',canonical(recovery))
            except BaseException as error:report['release']='UNCERTAIN';report['release_error']=type(error).__name__+': '+str(error)
        else:report['release']='NOT_ALLOCATED'
        report['allocation_wall_seconds']=elapsed();report['aborted_by_guard']=abort
        atomic_bytes(run/'controller_report.json',canonical(report))
        print(json.dumps({k:report.get(k) for k in ('status','release','allocation_attempts','allocation_wall_seconds','compute_used_seconds_including_local_validation','error','release_error')},indent=2),flush=True)
    return 0 if report['status']=='COMPLETE_492_PENDING_PREDICTED_BOX_REVIEW' and report['release']=='CONFIRMED' else 2

if __name__=='__main__':raise SystemExit(main())
