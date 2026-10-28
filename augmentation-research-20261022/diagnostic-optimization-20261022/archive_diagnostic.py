"""Archive a stopped, incomplete diagnostic honestly; never resume computation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
from verify_preservation import verify
WORK=Path(__file__).resolve().parent
PROJECT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
ROOT=PROJECT/'augmentation-research-20261022/diagnostic-optimization-20261022'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    cfg=read(ROOT/'protocol.json');lock=read(ROOT/'protocol-lock.json');pre=read(ROOT/'pre-execution-lock.json')
    assert sha(ROOT/'protocol.json')==lock['protocol_sha256']==pre['protocol_sha256']
    for n,h in lock['code_sha256'].items():assert sha(ROOT/n)==h
    for row in pre['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    assert sha(ROOT/'verify_independent.py')==pre['independent_verifier_sha256']
    failure=read(ROOT/'diagnostic-failure.json');started=read(ROOT/'results/started.json')
    partial=read(ROOT/'partial-analysis.json');ind=read(WORK/'partial-audit.json');full=read(WORK/'independent-verification.json')
    assert failure['phase']=={'stage':'paired_head_and_optimizer_oracle','layout':'back','gradient_batch':0}
    assert not (ROOT/'results/completed.json').exists() and not (ROOT/'analysis.json').exists()
    assert len(list((ROOT/'results').glob('*/batch*/post-step-trainable.pt')))==5
    assert len(list((ROOT/'results').glob('*/batch*/receipt.json')))==4
    assert partial['front_back_CLS_change_ratio'] is None
    assert ind['status']=='partial_artifacts_recomputed_protocol_gate_failure_preserved' and ind['diagnostic_status']=='incomplete_not_scientific_negative'
    assert ind['artifact_failures']==[] and len(ind['protocol_gate_failures'])==1 and ind['error'] is None
    assert ind['saved_steps_executed']==5 and ind['completed_case_receipts']==4 and ind['full_front_back_S_comparison'] is None
    assert sha(WORK/'verify_partial_independent.py')==ind['script_sha256']
    assert full['status']=='failed_or_incomplete_preserve_all' and full['error']['type']=='FileNotFoundError'
    assert sha(WORK/'verify_independent.py')==full['verifier_sha256']==pre['independent_verifier_sha256']
    assert (WORK/'independent-partial-review.md').exists()
    copied=[];cpname='CURRENT_CHECKPOINT_20261022_OPTIMIZATION_DIAGNOSTIC.md'
    for p in sorted(WORK.iterdir()):
        if not p.is_file() or p.name==cpname:continue
        target=ROOT/p.name
        if target.exists():assert sha(target)==sha(p),str(target)
        else:shutil.copyfile(p,target)
        copied.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    cp=PROJECT/cpname;assert not cp.exists();shutil.copyfile(WORK/cpname,cp)
    preservation=verify();save(ROOT/'preservation-after.json',preservation)
    save(ROOT/'supplement-copy-receipt.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'files':copied,'all_SHA_match':True})
    elapsed=(datetime.fromisoformat(failure['at_utc'])-datetime.fromisoformat(started['at_utc'])).total_seconds()
    receipt={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'owned_diagnostic_session_exited_after_predeclared_pairing_failure',
        'exec_session':38426,'owned_process_PID':started['PID'],'tool_exit_code':1,'owned_session_still_running':False,
        'exit_basis':'Tool write_stdin returned terminal exit_code1; no resumed or replacement process launched.',
        'PID_absence_freshly_queried':False,'optional_ps_retried':False,'global_GPU_idle_claimed':False,
        'started_to_failure_UTC_seconds':elapsed,'exact_perf_counter_total_peak_RSS_and_driver_peak':None,
        'wall_cap_seconds':600,'extra_GPU_forward_or_optimizer_calls_after_failure':0}
    save(ROOT/'execution-exit-receipt.json',receipt)
    status={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'stopped_incomplete_diagnostic_original_gate_failure_preserved',
        'planned_reset_steps':8,'executed_reset_steps':5,'complete_case_receipts':4,'full_front_back_CLS_ratio':None,
        'scientific_negative_or_new_contribution':False,'original_protocol_code_and_tolerances_changed':False,
        'independent_partial_artifacts_recomputed':True,'original_paired_head_gate_passed':False,
        'full_frozen_verifier_completed_or_passed':False,'partial_audit_artifact_failures':0,
        'new_full_training_seeds_LR_search_test_reads':False,'fair_reference_control_proposed_only':True,
        'remaining_blocker':{'action':'Finish front/back initial CLS scale comparison','target':'back-batch0 paired head-weight gradient then remaining back probes/batches',
            'status':'stopped_by_predeclared_validity_rule','reason':'142 head-weight gradient elements outside original atol2e-8,rtol2e-5; root cause not localized; primary paired data incomplete.'},
        'pending_approval_or_uncertain_execution':False,'A100_Colab_ARiS_new_dependencies_paid_upload_publish':False,
        'model_request':cfg['model_request']}
    save(ROOT/'outcome.json',status);save(ROOT/'LATEST_STATUS.json',status)
    save(ROOT/'background-status.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'this_diagnostic_training_or_GPU_jobs_running_or_paused':0,'owned_exec_session_exit_code':1,
        'independent_CPU_artifact_audit_finished':True,'restarted_or_extra_steps':False,'scheduler_created':False,
        '8A_files_environment_or_processes_inspected':False,'global_machine_idle_claimed':False})
    rows=[{'relative_path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['delivery-manifest.json','delivery-verification.json']]
    manifest={'at_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),'files':rows,'payload_files':len(rows),
        'payload_bytes':sum(r['bytes'] for r in rows),'checkpoint':{'path':str(cp),'bytes':cp.stat().st_size,'sha256':sha(cp)},
        'diagnostic_complete':False,'original_failure_preserved':True}
    save(ROOT/'delivery-manifest.json',manifest)
    assert all((ROOT/r['relative_path']).stat().st_size==r['bytes'] and sha(ROOT/r['relative_path'])==r['sha256'] for r in rows)
    final={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'all_partial_delivery_bytes_SHA_verified_incomplete_measurement_retained',
        'payload_files':len(rows),'payload_bytes':manifest['payload_bytes'],'manifest_sha256':sha(ROOT/'delivery-manifest.json'),
        'checkpoint_sha256':sha(cp),'original2523_pilot_payloads_unchanged':preservation['original_pilot_payloads'],
        'old_proposal17_research58_other25_unchanged':[preservation['old_proposal'],preservation['old_research'],preservation['other_protected']],
        'independent_partial_audit_checks':ind['checks'],'independent_partial_artifact_failures':0,'original_protocol_gate_failures_preserved':1,
        'scientific_runs_full_training':0,'isolated_steps_executed':5,'planned8_completed':False,
        'background_GPU_or_training_jobs_for_this_diagnostic':0,'new_control_launched':False}
    save(ROOT/'delivery-verification.json',final);print(json.dumps(final,ensure_ascii=False))
if __name__=='__main__':main()
