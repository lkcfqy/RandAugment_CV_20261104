"""One local CPU precheck or separately authorized full observation; no services."""
import time
START=time.monotonic()
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from contracts import canonical,complete_result_set,bound_metric_inputs,compute_gate
from evidence import atomic_bytes,owned_size
from transport import Collector,bounded_stream,compute_deadline

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_freeze():
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    for group in ('files','referenced_assets'):
        for rel,item in freeze[group].items():
            p=ROOT/rel
            if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('freeze differs: '+rel)
    return freeze,sha(ROOT/'FREEZE.json')

def stop_reason(now,run_deadline,compute_deadline_at,rss,disk):
    if now>=run_deadline:return 'external phase wall reached'
    if now>=compute_deadline_at-60:return 'cumulative CPU compute boundary; 60 s finalization reserve'
    if rss>8*1024**3:return '8 GiB RSS limit'
    if disk>6*1024**3:return '6 GiB owned disk limit'
    return None

def validate_formal_go(path,freeze_sha):
    if path is None:raise ValueError('formal 492 requires separate explicit GO')
    a=json.loads(Path(path).read_text())
    if a.get('formal_allowed') is not True or a.get('freeze_sha256')!=freeze_sha or a.get('scope')!='Mac_CPU_fixed_492' or a.get('max_runs')!=1:raise ValueError('formal GO must match this freeze and one fixed CPU run')

def main(argv=None):
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group();g.add_argument('--precheck',action='store_true');g.add_argument('--formal',action='store_true');parser.add_argument('--authorization',type=Path)
    args=parser.parse_args(argv);start=START;freeze,freeze_sha=verify_freeze();plan=json.loads((ROOT/'plan.json').read_text())
    if not args.precheck and not args.formal:
        print(json.dumps({'status':'CPU_FREEZE_VERIFIED','freeze_sha256':freeze_sha,'forwards':0,'GPU':0,'compute_cap_seconds':7200}));return 0
    mode='precheck' if args.precheck else 'formal';prior=plan['prior_compute_seconds'];precheck=None
    if args.formal:
        validate_formal_go(args.authorization,freeze_sha)
        pre_report=json.loads((ROOT/'precheck/controller_report.json').read_text())
        if pre_report['status']!='PASS_CPU_REPEATABILITY' or pre_report['freeze_sha256']!=freeze_sha:raise ValueError('matching completed CPU repeatability required')
        precheck=ROOT/'precheck/artifacts/engineering/status.json'
        if sha(precheck)!=pre_report['engineering_sha256']:raise ValueError('precheck evidence changed')
        prior=pre_report['cumulative_compute_seconds']
        e=json.loads(precheck.read_text());times=[max(c['complete_query_seconds'] for c in e['precheck_calls'] if c['mode']==m) for m in ('F','C','V')]
        if not compute_gate(prior+time.monotonic()-start,30.,times,60.)['allowed']:raise RuntimeError('unchanged 2x full492 forecast exceeds the 7200 s CPU cap')
    run=ROOT/mode;run.mkdir(exist_ok=True)
    with (run/'CLAIM.json').open('x') as f:
        json.dump({'mode':mode,'freeze_sha256':freeze_sha,'prior_compute_seconds':prior,'attempt_cap':6 if args.precheck else 492},f);f.flush();os.fsync(f.fileno())
    if owned_size(ROOT.parent)+512*1024**2>6*1024**3:raise RuntimeError('owned disk lacks output reserve')
    deadline=start+7200-prior;phase_deadline=min(deadline-60,start+180) if args.precheck else deadline-60
    output=run/'output';collector=Collector(run);pid=None;caffeinate=None;peak_rss=0;reason=None
    report={'status':'RUNNING','mode':mode,'freeze_sha256':freeze_sha,'prior_compute_seconds':prior,'new_CU':0,'new_GPU':0,'retries':0}
    context={'supervisor_pid':os.getpid(),'mode':mode,'output':str(output),'previous_compute_wall_seconds':prior,'freeze_sha256':freeze_sha,'compute_deadline_monotonic':deadline,'precheck_engineering':str(precheck) if precheck else None}
    atomic_bytes(run/'context.json',canonical(context))
    def on_started(child_pid):
        nonlocal pid,caffeinate
        pid=child_pid;caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        report['worker_pid']=pid;report['caffeinate_pid']=caffeinate.pid
    def watchdog():
        nonlocal peak_rss,reason
        if pid:
            try:
                p=subprocess.run(['/bin/ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,timeout=.25)
                peak_rss=max(peak_rss,int(p.stdout.strip() or 0)*1024)
            except (ValueError,subprocess.TimeoutExpired):pass
        state=output/'worker_state.json'
        if state.exists():peak_rss=max(peak_rss,json.loads(state.read_text()).get('memory',{}).get('peak_rss_bytes',0))
        reason=stop_reason(time.monotonic(),phase_deadline,deadline,peak_rss,owned_size(ROOT.parent))
        return reason is not None
    try:
        outcome=bounded_stream([sys.executable,'-B',str(ROOT/'scripts/cpu_worker.py'),'--context',str(run/'context.json')],collector.line,watchdog,cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'),on_started=on_started)
        report.update(outcome);report['stop_reason']=reason
        with compute_deadline(min(deadline,start+180) if args.precheck else deadline):
            index_path=collector.artifacts/'artifact_index.json'
            if not index_path.exists():raise RuntimeError('worker did not persist final artifact index')
            index=json.loads(index_path.read_text())
            if collector.missing(index):raise RuntimeError('persisted artifact set is incomplete')
            e=json.loads((collector.artifacts/'engineering/status.json').read_text());report['engineering_sha256']=sha(collector.artifacts/'engineering/status.json')
            report['indexed_artifacts_complete']=True;report['indexed_files']=len(index['files'])
            if outcome['returncode']!=0 or outcome['terminated_by_watchdog']:raise RuntimeError('CPU worker failed; retain incomplete engineering evidence')
            if args.precheck:
                c=e['counters']
                if e['status']!='PASS_CPU_REPEATABILITY' or (c['root_forward_entered'],c['root_forward_returned'],c['precheck_attempts'])!=(6,6,6) or c['formal_attempts']!=0:raise RuntimeError('CPU repeatability incomplete')
                report['status']='PASS_CPU_REPEATABILITY'
            else:
                from metrics import summarize
                q=json.loads((ROOT/'science/queries_frozen.json').read_text())['queries'];labels=json.loads((ROOT/'science/labels_frozen.json').read_text())['pairs']
                rows=[json.loads((collector.artifacts/p).read_text()) for p in sorted(index['files']) if p.startswith('results/')]
                complete_result_set(rows,{x['query_id']:x['query_sha256'] for x in q});grouped={}
                for row in rows:
                    p=row['prediction']
                    if p['count']!=len(p['boxes']) or p['count']!=len(p['scores']):raise RuntimeError('saved prediction counts differ')
                    grouped.setdefault(row['filename'],{}).setdefault(row['mode'],{})[row['target']]=p['count']
                pairs,policy=bound_metric_inputs(grouped,labels);local=summarize(pairs,policy);remote=json.loads((collector.artifacts/'science/summary.json').read_text())
                if any(remote.get(k)!=v for k,v in local.items()):raise RuntimeError('independent local exact metrics differ')
                report['status']='COMPLETE_492_PENDING_BOX_REVIEW'
    except BaseException as error:report.update(status='ENGINEERING_BLOCKED_OR_INCOMPLETE',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        if caffeinate is not None:
            caffeinate.terminate()
            try:caffeinate.wait(timeout=1)
            except subprocess.TimeoutExpired:caffeinate.kill();caffeinate.wait(timeout=1)
        report['peak_RSS_bytes']=peak_rss;report['this_compute_wall_seconds']=time.monotonic()-start;report['cumulative_compute_seconds']=prior+report['this_compute_wall_seconds'];report['remaining_compute_seconds']=max(0.,7200-report['cumulative_compute_seconds'])
        if report['cumulative_compute_seconds']>7200 or args.precheck and report['this_compute_wall_seconds']>180:report['status']='ENGINEERING_BLOCKED_OR_INCOMPLETE';report['deadline_exhausted']=True
        atomic_bytes(run/'controller_report.json',canonical(report))
        print(json.dumps({k:report.get(k) for k in ('status','mode','this_compute_wall_seconds','cumulative_compute_seconds','remaining_compute_seconds','error')},indent=2))
    return 0 if report['status'] in ('PASS_CPU_REPEATABILITY','COMPLETE_492_PENDING_BOX_REVIEW') else 2

if __name__=='__main__':raise SystemExit(main())
