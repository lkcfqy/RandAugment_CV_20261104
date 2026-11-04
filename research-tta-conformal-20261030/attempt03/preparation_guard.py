"""Only enforce the newly approved 8-minute preparation ceiling; never allocate."""
from portable_dev.paths import project_path as _portable_project_path
import datetime,json,os,subprocess,time
from pathlib import Path
R=_portable_project_path(__file__, 'research-tta-conformal-20261030/attempt03')
def write(name,obj):(R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
a=json.loads((R/'allocation-attempted.json').read_text());start=a['epoch']
write('PREPARATION_LIMIT_ADDENDUM.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'parent_approved_preparation_limit_seconds':480,'allocation_start_epoch':start,
 'preparation_deadline_epoch':start+480,'startup_prediction_gate_seconds':6300,
 'scientific_changes':[],'new_GPU_allocations_by_guard':0,
 'guard_release_trigger_before_deadline_seconds':45,
 'purpose':'Enforce latest parent8min condition without changing the already-running frozen execution source.'})
while True:
    if (R/'released.json').exists():
        write('preparation-guard-complete.json',{'result':'runtime_already_released','extra_GPU_allocations':0});break
    if (R/'performance-gate.json').exists():
        g=json.loads((R/'performance-gate.json').read_text())
        if g['passed'] and g['measured_epoch']-start<=480:
            write('preparation-guard-complete.json',{'result':'PASS','actual_preparation_seconds':g['measured_epoch']-start,
                  'deadline_seconds':480,'gate_prediction_seconds':g['cumulative_projection_seconds'],'no_stop_needed':True});break
    if time.time()-start>=435:
        command=['/Users/lkc/.local/share/google-colab-cli/bin/python',str(R.parent/'colab_guard.py'),'stop','--session','tta-conformal-final-20261030']
        env=os.environ.copy();env['COLAB_CLI_ASSIGN_RETRIES']='0';t=time.time()
        try:
            q=subprocess.run(command,capture_output=True,text=True,timeout=45,env=env)
            result={'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
        except subprocess.TimeoutExpired:result={'exit_code':124,'outcome':'uncertain; controller release reconciliation required'}
        write('preparation-guard-stop.json',{'started_epoch':t,'ended_epoch':time.time(),'reason':'Reserve45s to release within8min preparation budget',**result})
        break
    time.sleep(2)
print('Preparation guard complete',flush=True)
