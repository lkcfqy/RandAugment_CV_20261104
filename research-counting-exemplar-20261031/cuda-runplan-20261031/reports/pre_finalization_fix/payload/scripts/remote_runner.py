"""Colab-only parent watchdog. Prepares once, then streams one worker to completion.

No allocation/authentication logic. Installation and transfer wall are allocation wall;
every other remote preparation/verification second is charged to the 1800 s ledger.
"""
import time
SCRIPT_START=time.monotonic()
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
from contracts import canonical
from evidence import Evidence,atomic_bytes,owned_size
from transport import bounded_stream

ROOT=Path(__file__).resolve().parents[1]
LIMIT=6*1024**3

def preparation_termination_reason(*,compute_used,allocation_left,disk_bytes,tree_rss=0):
    if compute_used>=1740:return 'preparation compute deadline reached with finalization reserve'
    if allocation_left<=90:return 'preparation allocation deadline reached with release reserve'
    if disk_bytes>LIMIT:return '6 GiB owned disk cap exceeded during preparation'
    if tree_rss>8*1024**3:return '8 GiB preparation RSS cap exceeded'
    return None

def termination_reason(*,worker_elapsed,saw_formal,compute_used,tree_rss,cuda_peak,allocation_left,disk_bytes=0):
    if allocation_left<=90:return 'allocation cleanup reserve reached'
    if compute_used>=1740:return 'worker compute boundary reached; 60 s reserved for finalization/verification'
    if not saw_formal and worker_elapsed>=180:return '180 s external precheck wall reached'
    if tree_rss>8*1024**3:return '8 GiB parent plus worker RSS guard reached'
    if cuda_peak>8*1024**3:return '8 GiB CUDA memory guard reached'
    if disk_bytes>LIMIT:return '6 GiB owned disk cap exceeded during formal output'
    return None

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def owned_bytes():return owned_size(ROOT.parent)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allocation-seconds-left',type=float,required=True);parser.add_argument('--assembly-compute-seconds',type=float,required=True)
    args=parser.parse_args();start=SCRIPT_START;excluded=0.;setup=0.;preparing=False;evidence=Evidence(ROOT/'output')
    plan=json.loads((ROOT/'plan.json').read_text());preparation={'status':'RUNNING','native_environment':{},'transfers':[],'installation_wall_seconds':0.,'download_wall_seconds':0.,'assembly_compute_seconds':args.assembly_compute_seconds,'new_authentication':False,'allocation_count':1,'retries':0}
    report={'status':'PREPARING','worker_started':False};worker_pid=None;prep_pid=None;stop_reason=None;worker_start=None;saw_formal=False
    def tree_rss(pid):
        try:return sum(int(Path(path).read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') for path in ('/proc/self/statm','/proc/%d/statm'%pid))
        except (FileNotFoundError,TypeError):return 0
    def left():return args.allocation_seconds_left-(time.monotonic()-start)
    def current_excluded():
        if not preparing:return excluded
        state_path=ROOT/'preparation_state.json'
        if not state_path.exists():return excluded
        state=json.loads(state_path.read_text());value=state['excluded_wall_seconds']
        if state['phase'] in ('INSTALL','DOWNLOAD'):value+=max(0,time.monotonic()-state['phase_started_monotonic'])
        return value
    def charged():return args.assembly_compute_seconds+time.monotonic()-start-current_excluded()
    def ensure():
        if left()<=90:raise RuntimeError('allocation cleanup reserve reached')
        if plan['previous_compute_wall_seconds']+charged()>=1740:raise RuntimeError('total compute budget exhausted by preparation')
        if owned_bytes()>LIMIT:raise RuntimeError('owned remote disk cap exceeded')
    try:
        ensure()
        preparation_start=time.monotonic();preparing=True
        def prep_watchdog():
            nonlocal stop_reason
            reason=preparation_termination_reason(compute_used=plan['previous_compute_wall_seconds']+charged(),allocation_left=left(),disk_bytes=owned_bytes(),tree_rss=tree_rss(prep_pid))
            if reason:stop_reason=reason
            return reason is not None
        def prep_started(pid):
            nonlocal prep_pid
            prep_pid=pid
        prep_outcome=bounded_stream([sys.executable,'-u',str(ROOT/'scripts/remote_prepare.py'),'--allocation-seconds-left',str(left())],lambda line:print(line,end='',flush=True),prep_watchdog,cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'),on_started=prep_started)
        excluded=current_excluded();preparing=False
        preparation=json.loads((ROOT/'output/engineering/preparation.json').read_text()) if (ROOT/'output/engineering/preparation.json').exists() else {'status':'BLOCKED','error':'preparation killed before final report'}
        report['preparation_process']=prep_outcome
        if prep_outcome['returncode']!=0 or prep_outcome['terminated_by_watchdog'] or preparation['status']!='READY':raise RuntimeError('preparation blocked: '+str(stop_reason or preparation.get('error')))
        ensure();setup=charged();preparation['setup_compute_seconds']=setup
        evidence.write('engineering/preparation.json',preparation)
        environment=os.environ.copy();environment.update(PYTHONPATH=os.pathsep.join([str(ROOT/'deps'),str(ROOT/'scripts')]),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NVIDIA_TF32_OVERRIDE='0')
        context={'supervisor_pid':os.getpid(),'setup_compute_seconds':setup,'previous_compute_wall_seconds':plan['previous_compute_wall_seconds'],
                 'compute_deadline_monotonic':start+excluded+1800-plan['previous_compute_wall_seconds']-args.assembly_compute_seconds,
                 'finalization_reserve_seconds':60,'allocation_stop_monotonic':start+args.allocation_seconds_left-90}
        atomic_bytes(ROOT/'run_context.json',canonical(context));worker_start=time.monotonic();report['worker_started']=True
        report['setup_compute_seconds']=setup;report['remaining_compute_at_worker_start']=1800-plan['previous_compute_wall_seconds']-setup
        def on_started(pid):
            nonlocal worker_pid
            worker_pid=pid
        def watchdog():
            nonlocal stop_reason,saw_formal
            elapsed=time.monotonic()-worker_start
            state_file=ROOT/'worker_state.json'
            state=json.loads(state_file.read_text()) if state_file.exists() else {'phase':'precheck'}
            saw_formal=saw_formal or state['phase'] in ('formal','complete')
            if worker_pid:
                try:
                    rss=int(Path('/proc/%d/statm'%worker_pid).read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                    parent_rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                    report['peak_tree_rss_bytes']=max(report.get('peak_tree_rss_bytes',0),rss+parent_rss)
                except FileNotFoundError:pass
            mem=state.get('memory',{})
            stop_reason=termination_reason(worker_elapsed=elapsed,saw_formal=saw_formal,compute_used=plan['previous_compute_wall_seconds']+charged(),tree_rss=report.get('peak_tree_rss_bytes',0),cuda_peak=max(mem.get('peak_cuda_allocated_bytes',0),mem.get('peak_cuda_reserved_bytes',0)),allocation_left=left(),disk_bytes=owned_bytes())
            return stop_reason is not None
        outcome=bounded_stream([sys.executable,'-u',str(ROOT/'scripts/worker.py')],lambda line:print(line,end='',flush=True),watchdog,cwd=ROOT,env=environment,on_started=on_started)
        report.update(outcome);report['status']='WORKER_COMPLETE' if outcome['returncode']==0 and not outcome['terminated_by_watchdog'] else 'BLOCKED_OR_INCOMPLETE'
        report['stop_reason']=stop_reason
    except BaseException as error:
        report.update(status='BLOCKED_OR_INCOMPLETE',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        excluded=current_excluded();preparing=False
        report['preparation']=preparation;report['allocation_stage_wall_seconds']=time.monotonic()-start
        expiry_state=ROOT/'runtime_expiry_state.json'
        report['native_runtime_expiry_guard']=json.loads(expiry_state.read_text()) if expiry_state.exists() else {'status':'NOT_YET_CONFIRMED'}
        report['charged_compute_seconds_this_allocation']=charged()
        report['compute_used_seconds']=plan['previous_compute_wall_seconds']+charged()
        report['compute_remaining_seconds']=max(0.,1800-report['compute_used_seconds'])
        report['scientific_conclusion_complete']=report.get('status')=='WORKER_COMPLETE'
        # Includes preparation logs and partial evidence even after a killed worker.
        index={}
        for path in sorted((ROOT/'output').rglob('*')):
            if path.is_file() and path.name!='artifact_index.json':
                index[str(path.relative_to(ROOT/'output'))]={'sha256':sha(path),'bytes':path.stat().st_size}
        report['compute_used_seconds']=plan['previous_compute_wall_seconds']+charged()
        report['compute_remaining_seconds']=max(0.,1800-report['compute_used_seconds'])
        evidence.write('supervisor.json',report)
        index['supervisor.json']={'sha256':sha(ROOT/'output/supervisor.json'),'bytes':(ROOT/'output/supervisor.json').stat().st_size}
        evidence.write('artifact_index.json',{'files':index,'complete_worker':report.get('status')=='WORKER_COMPLETE','compute_used_seconds_after_artifact_hashing':plan['previous_compute_wall_seconds']+charged()})
    return 0 if report.get('status')=='WORKER_COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
