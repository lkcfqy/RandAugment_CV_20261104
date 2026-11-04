"""Reviewed future entry point. No scientific work runs on import or `status`.

Parent process owns the persistent ledger and kills a worker at its remaining
wall deadline. No automatic retry, budget reset, Colab allocation, or download.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
import numpy as np
from .audit import current_code_hash,require_release,resource_bound
from .runtime import ComputeLedger,ContinuousBudget,DiskGuard,atomic_json,preflight,run_worker_bounded

IMPLEMENTATION_ROOT=Path(__file__).resolve().parents[1]

def oracle_charge_seconds():
    path=IMPLEMENTATION_ROOT/'reports/oracle_runs.json'
    runs=json.loads(path.read_text()) if path.exists() else []
    return sum(math.ceil(r['wall_seconds']) for r in runs)

def status(asset_root):
    path=Path(asset_root)/'implementation-v1/run-state/compute_ledger.json'
    return {'code_hash':current_code_hash(),'oracle_charge_seconds':oracle_charge_seconds(),'ledger':json.loads(path.read_text()) if path.exists() else None,'no_scientific_process_started':True}

def artificial_preflight_fixture():
    """Fixed manufactured full-size fixture; never fitted to or scored on test Y/GT."""
    from .core import q8,source_bins
    from .fit import AParams,BParams,nearest_nonempty
    from .synth import synthesize_a,synthesize_b
    from .diagnostics import diagnose_cell
    yy,xx=np.indices((400,600))
    x=np.stack([(xx+yy)%256,(2*xx+yy)%256,(xx+3*yy)%256]).astype(np.float64)/255
    counts=np.bincount(source_bins(x).ravel(),minlength=16)*6;fallback=nearest_nonempty(counts)
    quantiles=[]
    for b in range(16):
        n=int(counts[fallback[b]])
        # Full-size sample arrays reproduce empirical-quantile lookup footprint.
        quantiles.append(tuple(np.linspace(-2,2,n,dtype=np.float64) for _ in range(3)))
    spectrum=np.broadcast_to(np.eye(3,dtype=np.complex128),(400,600,3,3)).copy();spectrum[0,0]=0
    root=spectrum.copy();sigma=np.ones((16,3))*.03
    bp=BParams(np.broadcast_to(np.arange(256)/510,(3,256)).copy(),np.zeros((16,3)),sigma*sigma,sigma,tuple(quantiles),counts,fallback,spectrum,root)
    ap=AParams(1.,np.ones(3)*.5,np.column_stack([np.ones(3)*.03**2,np.zeros(3)]),0.)
    y=q8(.5*x+.03*np.random.Generator(np.random.PCG64(20261030)).standard_normal(x.shape))
    def cell_probe():
        aa=np.stack([synthesize_a(x,ap,'Scene7',1,seed) for seed in (11,22,33,44)])
        bb=np.stack([synthesize_b(x,bp,'Scene7',1,seed)[0] for seed in (11,22,33,44)])
        # Compute to measure full work, discard values; no task utility metric.
        diagnose_cell(x,y,aa,bp,synthetic_degeneracy='variance_1e-12')
        diagnose_cell(x,y,bb,bp,synthetic_degeneracy='variance_1e-12')
    return x,cell_probe

def validate_worker_session(job):
    """An internal worker must match its parent's live, persisted reservation."""
    mode=job.get('mode')
    if mode not in ('preflight','formal') or job.get('receipt',{}).get('scope')!=mode:raise ValueError('Worker stage/release scope mismatch')
    root=Path(job['asset_root']).resolve();run=Path(job['run_root']).resolve()
    if run!=root/'implementation-v1/run-state':raise ValueError('Worker ledger path mismatch')
    path=run/'compute_ledger.json'
    if not path.is_file() or not path.with_suffix('.lock').is_file():raise ValueError('Worker requires its parent active ledger/lock')
    active=json.loads(path.read_text()).get('active')
    fields=('kind','started_monotonic','reserved_seconds','spent_before')
    if not isinstance(active,dict) or any(job.get(k)!=active.get(k) for k in fields) or active.get('kind')!=mode:raise ValueError('Worker reservation differs from parent ledger')
    reserved=active['reserved_seconds'];spent=active['spent_before']
    if not np.isfinite([reserved,spent,active['started_monotonic']]).all() or reserved<=0 or spent<0 or reserved+spent>1800 or (mode=='preflight' and reserved>120):raise ValueError('Invalid worker reservation')

def _worker(job):
    validate_worker_session(job)
    from .model import CIDNetPredictor
    from .pipeline import PanelReader,prepare_generated,evaluate_generated
    root=Path(job['asset_root']);run=Path(job['run_root']);receipt=job['receipt'];freeze_sha=job['freeze_sha256']
    require_release(receipt,freeze_sha)
    already=time.monotonic()-job['started_monotonic'];remaining=job['reserved_seconds']-already
    budget=ContinuousBudget(remaining,prior_spent=job['spent_before']+already,disk_root=root)
    if job['mode']=='preflight':
        predictor=CIDNetPredictor(root,device=job['device'],receipt=receipt,freeze_sha=freeze_sha)
        x,cell=artificial_preflight_fixture();budget.check()
        def forward():budget.check();predictor(x);budget.check()
        def probe_cell():budget.check();cell();budget.check()
        result=preflight(forward,probe_cell,predictor.synchronize,t_spent=budget.total_spent,t_post=job['tpost'],receipt=receipt,freeze_sha=freeze_sha)
        result.update(status='PREFLIGHT_COMPLETE',freeze_sha256=freeze_sha,model_calls=predictor.calls,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,fixture='fixed artificial RGB 400x600; no real data/GT or task utility')
        if predictor.device=='mps':result['device_allocated_bytes']=predictor.torch.mps.current_allocated_memory()
        elif predictor.device=='cuda':result['device_allocated_bytes']=predictor.torch.cuda.max_memory_allocated()
        else:result['device_allocated_bytes']=0
    else:
        reader=PanelReader(root);cache=run/'generated'
        prepare_generated(reader,cache,receipt=receipt,freeze_sha=freeze_sha,budget=budget)
        predictor=CIDNetPredictor(root,device=job['device'],receipt=receipt,freeze_sha=freeze_sha)
        result=evaluate_generated(reader,cache,predictor,receipt=receipt,freeze_sha=freeze_sha,budget=budget)
        if predictor.calls!=326:raise ValueError('Formal model-call count is not exactly326')
        result.update(status='FORMAL_COMPLETE',freeze_sha256=freeze_sha,model_calls=predictor.calls,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    budget.before_write(len(json.dumps(result).encode())*2)
    atomic_json(run/(job['mode']+'_result.json'),result);budget.check()
    return result

def launch(mode,asset_root,receipt,*,device,tpost):
    if mode not in ('preflight','formal'):raise ValueError('Unknown scheduler action')
    freeze_sha=current_code_hash();require_release(receipt,freeze_sha)
    if receipt.get('scope')!=mode:raise ValueError('Release scope does not match requested stage')
    if device not in ('cpu','mps','cuda') or not np.isfinite(tpost) or tpost<=0:raise ValueError('Explicit device and positive conservative tpost required')
    root=Path(asset_root).resolve();run=root/'implementation-v1/run-state';guard=DiskGuard(root)
    preflight_path=run/'preflight_result.json'
    if mode=='formal':
        if not preflight_path.is_file():raise ValueError('Formal execution requires completed same-freeze preflight')
        pf=json.loads(preflight_path.read_text())
        if pf.get('freeze_sha256')!=freeze_sha or pf.get('status')!='PREFLIGHT_COMPLETE' or pf.get('resource_go') is not True:raise ValueError('Preflight missing, blocked or stale')
        if pf.get('device')!=device:raise ValueError('Device differs from timed preflight')
    guard.reserve(1024*1024)
    ledger=ComputeLedger(run/'compute_ledger.json',oracle_charge=oracle_charge_seconds())
    if mode=='formal':
        bound=resource_bound(ledger.spent,pf['t_forward'],pf['t_cell'],pf['t_post'])
        if bound>1800:raise TimeoutError('RESOURCE_BLOCKED: recomputed conservative bound exceeds1800s')
    session=ledger.begin(mode)
    job={**session,'mode':mode,'asset_root':str(root),'run_root':str(run),'device':device,'tpost':tpost,'receipt':receipt,'freeze_sha256':freeze_sha}
    job_path=run/(mode+'_job.json');outcome='EXECUTION_BLOCKED'
    try:
        atomic_json(job_path,job)
        # Leave one second of the original reservation for parent persistence.
        remaining=session['reserved_seconds']-(time.monotonic()-session['started_monotonic'])-1.
        if remaining<=0:raise TimeoutError('RESOURCE_BLOCKED: no remaining worker allowance')
        process=run_worker_bounded([sys.executable,'-m','millproxy.scheduler','_worker','--job',str(job_path)],remaining_seconds=remaining,cwd=IMPLEMENTATION_ROOT)
        guard.reserve(len(process['stdout'].encode())+len(process['stderr'].encode())+65536)
        (run/(mode+'_stdout.log')).write_text(process['stdout']);(run/(mode+'_stderr.log')).write_text(process['stderr'])
        outcome='RESOURCE_BLOCKED' if process['resource_blocked'] else 'PASS' if process['exit_code']==0 else 'VALIDATION_OR_EXECUTION_BLOCKED'
        result_path=run/(mode+'_result.json')
        result=json.loads(result_path.read_text()) if result_path.exists() else {'status':outcome,'scientific_negative':False}
        estimated_charge=math.ceil(time.monotonic()-session['started_monotonic'])
        result.update(parent_status=outcome,device=device,ledger_path=str(ledger.path),spent_before_final_ledger_flush=ledger.spent+estimated_charge)
        if mode=='preflight' and outcome=='PASS':
            result['T_bound']=resource_bound(ledger.spent+estimated_charge,result['t_forward'],result['t_cell'],result['t_post']);result['resource_go']=result['T_bound']<=1800
        if outcome!='PASS':result['scientific_negative']=False;result['resource_go']=False
        guard.reserve(len(json.dumps(result).encode())*2+65536);atomic_json(result_path,result)
        guard.reserve();event=ledger.finish(outcome)
    except BaseException:
        if ledger.state['active'] is not None:ledger.finish(outcome)
        raise
    result.update(parent_status=outcome,charged_event=event,total_spent_seconds=ledger.spent,remaining_seconds=ledger.remaining,device=device)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['status','preflight','formal','_worker'])
    parser.add_argument('--asset-root');parser.add_argument('--release');parser.add_argument('--device',choices=['cpu','mps','cuda'])
    parser.add_argument('--post-bound-seconds',type=float);parser.add_argument('--job')
    args=parser.parse_args()
    if args.mode=='_worker':
        if not args.job:parser.error('--job required')
        result=_worker(json.loads(Path(args.job).read_text()))
    elif args.mode=='status':
        if not args.asset_root:parser.error('--asset-root required')
        result=status(args.asset_root)
    else:
        if not all([args.asset_root,args.release,args.device,args.post_bound_seconds]):parser.error('Execution requires explicit asset root, reviewed release, device, and positive post bound')
        result=launch(args.mode,args.asset_root,json.loads(Path(args.release).read_text()),device=args.device,tpost=args.post_bound_seconds)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
