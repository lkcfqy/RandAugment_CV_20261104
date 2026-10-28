"""Archive only a completed, independently verified pilot; never launch training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
from verify_previous import verify

WORK=Path(__file__).resolve().parent
PROJECT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
ROOT=PROJECT/'augmentation-research-20261022/pilot-20261022'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    done=read(ROOT/'results/completed.json');analysis=read(ROOT/'analysis/summary.json')
    independent=read(WORK/'independent/result-verification.json')
    definition_matched=read(WORK/'independent/result-verification-v2.json')
    exit_receipt=read(ROOT/'execution-exit-receipt.json')
    cfg=read(ROOT/'protocol.json');lock=read(ROOT/'protocol-lock.json')
    closeout=read(ROOT/'closeout-checks.json')
    assert done['status']=='all_four_cells_completed'
    assert independent['status']=='metrics_and_saved_state_audit_completed_with_strict_norm_discrepancies'
    assert independent['original_frozen_verifier_passed'] is False
    assert independent['norm_tolerances_relaxed'] is False
    assert independent['bootstrap_replicates_verified']==5000 and independent['bootstrap_max_absolute_difference']==0
    assert 'primary_I' in independent['check_log'] and 'all5000_bootstrap_replicates' in independent['check_log']
    norms=read(WORK/'independent/norm-diagnostic.json')['records']
    assert len(norms)==24 and all(r['CPU_FP32_minus_reported']==0 for r in norms)
    assert all(r['check'].endswith((':Q_delta_frobenius',':V_delta_frobenius')) for r in independent['strict_norm_discrepancies'])
    assert definition_matched['status']=='passed_definition_matched_saved_run_audit_v2_original_failure_preserved'
    assert definition_matched['original_frozen_verifier_passed'] is False and definition_matched['norm_tolerances_relaxed'] is False
    assert definition_matched['primary_I']==independent['primary_I']
    assert definition_matched['bootstrap_max_absolute_difference']==0
    assert exit_receipt['training_tool_exit_code']==0 and exit_receipt['original_training_process_still_present'] is False
    assert closeout['optional_additional_PID_ps_recheck']['status']=='denied_and_stopped_without_retry'
    assert closeout['background_state_basis']=='original_training_tool_exit_code0_and_saved_15:54:28UTC_PID_absence_receipt; no further training launched'
    assert done['protocol_sha256']==sha(ROOT/'protocol.json')==lock['protocol_sha256']
    assert independent['primary_I']==analysis['interaction_I']
    assert all(sha(ROOT/n)==h for n,h in lock['code_sha256'].items())
    # Original prefitting payloads must remain byte-identical after training.
    copy=read(ROOT/'copy-receipt.json')
    for row in copy['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    copied=[]
    for p in sorted((WORK/'independent').iterdir()):
        if not p.is_file():continue
        target=ROOT/'independent'/p.name
        if target.exists():assert sha(target)==sha(p),str(p)
        else:shutil.copyfile(p,target)
        copied.append({'path':str(target.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    for name in ['archive_results.py','verify_previous.py']:
        target=ROOT/name
        if target.exists():assert sha(target)==sha(WORK/name),name
        else:shutil.copyfile(WORK/name,target)
    preservation=verify();save(ROOT/'preservation-final.json',preservation)
    save(ROOT/'independent-copy-final.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'files':copied,
        'independent_CPU_verification_not_retraining':True,
        'original_full_verifier_passed':False,
        'metrics_and5000_bootstraps_verified':True,
        'auxiliary_norm_definition_discrepancies_preserved':len(independent['strict_norm_discrepancies']),
        'all24_same_definition_CPU_FP32_norms_match_exactly':True})
    outcome={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_four_cell_single_seed_development_pilot_and_independent_recalculation',
        'question':cfg['question'],'protocol_sha256':lock['protocol_sha256'],'epochs_per_cell':cfg['epochs'],
        'seed':cfg['seed'],'scientific_cells':4,'scientific_updates':sum(c['updates'] for c in done['cells']),
        'source_presentations':sum(c['source_presentations'] for c in done['cells']),
        'primary_I':analysis['interaction_I'],'G_front':analysis['G_front'],'G_back':analysis['G_back'],
        'paired_photo_conditional_interval':analysis['paired_stratified_photo_bootstrap_95percent_interval'],
        'engineering_gate_I_min':cfg['engineering_gate_I_min'],'engineering_gate_passed':analysis['engineering_gate_passed'],
        'novel_contribution_or_cross_seed_confirmation_established':False,
        'interpretation':'Single seed and common optimizer recipe; fixed-model source-photo interval excludes training variance. See finding-zh.md for scientific decision and limitations.',
        'additional_seeds_LR_search_test_reads_or_backup_training':0,
        'actual_device':'mps:0','MPS_fallback':False,'CUDA_used':False,
        'wall_seconds_including_resource_pause':done['wall_seconds'],'wall_cap_seconds':cfg['wall_cap_seconds'],
        'resource_pause_seconds':read(ROOT/'resource-resume-01.json')['pause_duration_seconds'],
        'CPU_concurrency_limit':'Known8A period reported by parent, ended by15:32UTC minute precision; no8A file/environment access.',
        'all_work_on_Mac':True,'new_dependencies_installed':False,'A100_Colab_ARiS_paid_upload_publish':False,
        'old_proposal_payloads_unchanged':preservation['previous_proposal_payloads_unchanged'],
        'old_research_payloads_unchanged':preservation['previous_research_payloads_unchanged'],
        'other_protected_files_unchanged':preservation['other_protected_files_unchanged'],
        'remaining_execution_permission_blocker':None,'pending_or_uncertain_actions':False,
        'optional_closeout_read_only_ps_recheck':'Denied by sandbox; stopped without retry. See closeout-checks.json. Training exit remains established by original tool and saved closeout receipt.',
        'background_training_processes_for_this_pilot':0,'model_request':cfg['model_request']}
    outcome.update(independent_metrics_and5000_bootstraps_verified=True,original_full_verifier_passed=False,
        definition_matched_saved_state_audit_v2_passed=True,definition_matched_v2_checks=definition_matched['checks_count'],
        auxiliary_QV_FP64_norm_strict_discrepancies=len(independent['strict_norm_discrepancies']),
        all24_same_definition_CPU_FP32_norms_match_exactly=True,
        norm_audit_scope='Original strict FP64-versus-reported-FP32 reduction predicate failed and is retained; identical-definition CPU FP32 replay matches all24 exactly. No training or statistical threshold changed; see independent reports.')
    save(ROOT/'outcome.json',outcome)
    background=read(ROOT/'background-status.json')
    background.update(at_utc=datetime.now(timezone.utc).isoformat(),independent_recomputation_pending=False,
        independent_CPU_recomputation_finished=True,this_pilot_training_processes_running_or_paused=0,
        scheduler_created=False,background_state_basis=closeout['background_state_basis'],
        optional_additional_PID_ps_recheck='denied_and_stopped_without_retry')
    (ROOT/'background-status.json').write_text(json.dumps(background,ensure_ascii=False,indent=2)+'\n')
    cp=PROJECT/'CURRENT_CHECKPOINT_20261022_LORA_PILOT.md'
    assert cp.exists() and (ROOT/'finding-zh.md').exists() and (ROOT/'runtime-accounting.json').exists()
    with (PROJECT/'CURRENT_CHECKPOINT_20261022_LORA_RUNNING.md').open('a') as f:
        f.write('\n## 本轮完成\n\n四格与独立复算已完成，本轮后台训练为0。当前结论、原始结果与下一步以CURRENT_CHECKPOINT_20261022_LORA_PILOT.md为准；上述运行/暂停快照作为历史记录保留。\n')
    latest={'at_utc':datetime.now(timezone.utc).isoformat(),'status':outcome['status'],
        'cells':[{'cell':c['cell'],'completed':True,'epochs':c['epochs'],'updates':c['updates']} for c in done['cells']],
        'original_PID':49366,'original_training_process_still_present':False,
        'training_tool_exit_code':0,'training_completed':True,'scientific_completed_at_utc':done['at_utc'],
        'wall_seconds':done['wall_seconds'],'elapsed_wall_seconds':done['wall_seconds'],
        'remaining_original_wall_cap_seconds':cfg['wall_cap_seconds']-done['wall_seconds'],
        'budget_reset_or_extended':False,'restart_or_extra_seed':False,
        'resource_pause_seconds':outcome['resource_pause_seconds'],
        'CPU_concurrency_limit':outcome['CPU_concurrency_limit'],
        'independent_metrics_and5000_bootstraps_verified':True,'original_full_verifier_passed':False,
        'definition_matched_saved_state_audit_v2_passed':True,
        'auxiliary_norm_definition_discrepancies_preserved':True,
        'background_state_basis':closeout['background_state_basis'],
        'optional_additional_PID_ps_recheck':'denied_and_stopped_without_retry'}
    (ROOT/'LATEST_STATUS.json').write_text(json.dumps(latest,ensure_ascii=False,indent=2)+'\n')
    rows=[{'relative_path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
          for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
          and p.name not in {'delivery-manifest.json','delivery-verification.json'}]
    manifest={'at_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),'files':rows,
        'payload_files':len(rows),'payload_bytes':sum(r['bytes'] for r in rows),
        'checkpoint':{'path':str(cp),'bytes':cp.stat().st_size,'sha256':sha(cp)},
        'historical_running_checkpoint':{'path':str(PROJECT/'CURRENT_CHECKPOINT_20261022_LORA_RUNNING.md'),
            'sha256':sha(PROJECT/'CURRENT_CHECKPOINT_20261022_LORA_RUNNING.md')},
        'excludes':'delivery-manifest.json, delivery-verification.json and any bytecode cache'}
    save(ROOT/'delivery-manifest.json',manifest)
    assert all((ROOT/r['relative_path']).stat().st_size==r['bytes'] and sha(ROOT/r['relative_path'])==r['sha256'] for r in rows)
    result={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'all_payload_bytes_SHA256_and_previous_materials_verified',
        'payload_files':len(rows),'payload_bytes':manifest['payload_bytes'],
        'manifest_sha256':sha(ROOT/'delivery-manifest.json'),'checkpoint_sha256':sha(cp),
        'all_prefitting_2412_payload_files_still_match':True,'independent_metrics_and5000_bootstraps_passed':True,
        'original_full_verifier_passed':False,'auxiliary_QV_FP64_norm_strict_discrepancies_preserved':18,
        'definition_matched_saved_state_audit_v2_passed':True,
        'all24_same_definition_CPU_FP32_norms_match_exactly':True,
        'background_training_processes_for_this_pilot':0,'scientific_runs':4,'extra_seeds':0,
        'previous_proposal_payloads_unchanged':17,'previous_research_payloads_unchanged':58,'other_protected_files_unchanged':25}
    save(ROOT/'delivery-verification.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
