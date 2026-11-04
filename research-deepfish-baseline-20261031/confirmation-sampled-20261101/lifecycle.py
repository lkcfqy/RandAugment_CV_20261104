"""One bounded local CPU workflow; no cloud, installs or training."""
import datetime,json,os,pathlib,signal,subprocess,sys,time,traceback
from diagnostic_contracts import P,Guard,atomic_json,complete_matrix,read_execution,sha

def run():
    context=json.loads((P/'RUN_STARTED.json').read_text());guard=Guard(context)
    frozen=json.loads((P/'FROZEN.json').read_text())
    if sha(P/'RUN_PROTOCOL.json')!=frozen['run_protocol_sha256']:raise RuntimeError('Execution freeze differs')
    for path,expected in frozen['file_sha256'].items():
        if sha(P/path)!=expected:raise RuntimeError('Frozen file differs: '+path)
    lock=P/'LIFECYCLE_LOCK.json'
    with lock.open('x') as f:json.dump({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
    phases=[];child=None;started=time.monotonic();status='INCOMPLETE';reason=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    try:
        guard.check()
        for label,args in [('P0',['infer','--index','0']),('A0',['infer','--index','1']),('A1',['infer','--index','2']),('metrics',['metrics']),('recompute',['verify'])]:
            guard.check();t=time.monotonic()
            with (P/'logs'/(label+'.stdout.log')).open('ab') as out:
                child=subprocess.Popen([sys.executable,'-B',str(P/'formal_cpu.py'),*args],cwd=P,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(P/'CURRENT_PHASE.json',{'label':label,'PID':child.pid,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RUNNING'})
                try:ret=child.wait(timeout=max(.01,context['deadline_monotonic']-time.monotonic()))
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                    raise
            phases.append({'label':label,'PID':child.pid,'exit_code':ret,'child_exited':True,'elapsed_seconds':time.monotonic()-t});atomic_json(P/'PHASES.json',phases)
            print(json.dumps(phases[-1]),flush=True);child=None
            if ret!=0:raise RuntimeError('Child phase failed: '+label)
            if label=='A1':atomic_json(P/'ALL_696_COMPLETE.json',complete_matrix(read_execution()),new=True)
        status='COMPLETE'
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc);traceback.print_exc()
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        atomic_json(P/'CURRENT_PHASE.json',{'status':'EXITED','all_children_exited':True})
        final={'status':status,'reason':reason,'all_children_exited':True,'lifecycle_seconds':time.monotonic()-started,'total_workflow_seconds':time.monotonic()-context['start_monotonic'],'new_GPU_MPS_install_upload_CU':0,'training_updates':0,'expected_forwards':696,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        atomic_json(P/'FINAL_STATUS.json',final);print(json.dumps(final),flush=True)
    return 0 if status=='COMPLETE' else 1

if __name__=='__main__':sys.exit(run())
