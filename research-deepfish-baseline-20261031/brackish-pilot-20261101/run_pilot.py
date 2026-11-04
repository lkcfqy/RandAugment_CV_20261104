"""Bounded local lifecycle: discard preflight, train2, dev3, freeze decisions, confirm3."""
from contracts import *
import subprocess,sys,traceback

def estimate(pre,elapsed,proto):
    rule=proto['time_rule'];training=0;rows={};rss=0
    for arm in ('A0','A1'):
        record=pre[arm];assert record['status']=='PREFLIGHT_COMPLETE' and record['completed_updates']==1
        s=record['steps'][0];area=s['shape_CHW'][1]*s['shape_CHW'][2]
        ratio=max(1.,rule['geometry_max_area']/area)
        step=max(s['train_step_seconds']+s['pipeline_seconds'],rule['historical_A_max_step_seconds'])
        training+=200*step*ratio;rss=max(rss,record['peak_RSS_bytes']*ratio)
        rows[arm]={'actual_step_seconds':s['train_step_seconds'],'pipeline_seconds':s['pipeline_seconds'],'observed_shape':s['shape_CHW'],'forecast_step_base_seconds':step,'geometry_ratio':ratio}
    infer=proto['expected_eval_forwards']*rule['reference_forward_max_seconds']
    other=rule['model_save_verify_allowance_seconds']+rule['initialization_metric_and_bookkeeping_allowance_seconds']
    future=rule['safety_multiplier']*(training+infer+other)
    peak=rule['safety_multiplier']*(rss+700*2**20)
    return {'preflight':rows,'training_seconds_before_margin':training,'inference_seconds_before_margin':infer,'save_init_metric_bookkeeping_seconds_before_margin':other,'safety_multiplier':rule['safety_multiplier'],'elapsed_seconds':elapsed,'future_seconds_conservative':future,'total_seconds_conservative':elapsed+future,'peak_RSS_bytes_conservative':peak,'time_pass':elapsed+future<proto['wall_limit_seconds'],'memory_pass':peak<=proto['RSS_cap_bytes']}

def main():
    assert not (P/'RUN_STARTED.json').exists(),'No duplicate pilot lifecycle'
    for name,h in read(P/'CODE_PINS.json').items():assert sha(P/name)==h
    proto=read(P/'PROTOCOL.json');assert proto['status']=='FROZEN_BEFORE_ANY_MODEL_OUTPUTS_OR_UPDATES'
    origin=read(P/'LIFECYCLE_BUDGET_ORIGIN.json') if (P/'LIFECYCLE_BUDGET_ORIGIN.json').exists() else {'start_UTC':utc(),'start_epoch':time.time(),'deadline_epoch':time.time()+10800}
    context={'start_UTC':origin['start_UTC'],'start_epoch':origin['start_epoch'],'deadline_epoch':origin['deadline_epoch'],'PID':os.getpid(),'preflight_deadline_epoch':min(time.time()+600,origin['deadline_epoch']),'current_attempt_UTC':utc()}
    atomic(P/'RUN_STARTED.json',context)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(P/'mplconfig'),HF_HOME=str(P/'hf_home'),XDG_CACHE_HOME=str(P/'cache'),CUDA_VISIBLE_DEVICES='')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='4'
    phases=[]
    def execute(label,script,args,deadline):
        remaining=deadline-time.time();assert remaining>0,'Deadline before '+label
        row={'phase':label,'started_UTC':utc(),'started_epoch':time.time(),'command':[sys.executable,'-B',script,*args]};phases.append(row);atomic(P/'PHASES.json',phases)
        with (P/(label+'.stdout.log')).open('xb') as f:
            child=subprocess.Popen([sys.executable,'-B',str(P/script),*args],cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT)
            row['PID']=child.pid;atomic(P/'PHASES.json',phases)
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:child.kill();code=child.wait();row['timeout_killed']=True
            except BaseException:child.kill();child.wait();row['exception_killed']=True;raise
        row.update(exit_code=code,child_exited=True,elapsed_seconds=time.time()-row['started_epoch'],completed_UTC=utc());atomic(P/'PHASES.json',phases)
        assert code==0,'Phase failed: '+label
    try:
        for arm in ('A0','A1'):execute('preflight_'+arm,'train_worker.py',['preflight',arm],context['preflight_deadline_epoch'])
        pre={a:read(P/f'preflight_{a}/RESULT.json') for a in ('A0','A1')}
        gate=estimate(pre,time.time()-context['start_epoch'],proto)
        gate.update(current_research_plus_staging_bytes=tree(R)+tree(P),two_complete_checkpoints_bytes=2*692247962,prediction_other_output_allowance_bytes=150*2**20)
        gate['projected_project_bytes']=gate['current_research_plus_staging_bytes']+gate['two_complete_checkpoints_bytes']+gate['prediction_other_output_allowance_bytes']
        gate['storage_pass']=gate['projected_project_bytes']<=12*2**30-80*2**20
        gate['status']='PASS' if all(gate[k] for k in ['time_pass','memory_pass','storage_pass']) else 'RESOURCE_GATE_STOP'
        atomic(P/'PREFLIGHT_GATE.json',gate);print(json.dumps(gate),flush=True)
        if gate['status']!='PASS':
            atomic(P/'FINAL_STATUS.json',{'status':'RESOURCE_GATE_STOP','formal_updates_A0':0,'formal_updates_A1':0,'new_CU':0,'all_children_exited':all(r.get('child_exited') for r in phases)});return
        # Each preflight process has exited: no model, optimizer or RNG state is reused.
        for arm in ('A0','A1'):execute('formal_'+arm,'train_worker.py',['formal',arm],context['deadline_epoch'])
        for split in ('dev','confirm'):
            for arm in ('P0','A0','A1'):execute('infer_'+split+'_'+arm,'eval_worker.py',['infer',split,'--arm',arm],context['deadline_epoch'])
            execute('metrics_'+split,'eval_worker.py',['metrics',split],context['deadline_epoch'])
        atomic(P/'FINAL_STATUS.json',{'status':'COMPLETE','completed_UTC':utc(),'elapsed_seconds':time.time()-context['start_epoch'],'formal_updates_A0':200,'formal_updates_A1':200,'expected_forwards':1224,'all_children_exited':all(r.get('child_exited') for r in phases),'new_GPU_MPS_install_download_upload_CU':0})
    except BaseException as e:
        atomic(P/'FINAL_STATUS.json',{'status':'INCOMPLETE','error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'UTC':utc(),'elapsed_seconds':time.time()-context['start_epoch'],'all_completed_children_exited':all(r.get('child_exited',False) for r in phases),'new_CU':0});raise
if __name__=='__main__':main()
