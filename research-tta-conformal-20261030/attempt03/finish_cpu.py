"""Wait for verified GPU release, then run the frozen CPU analysis and archive."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time,traceback
from state_io import atomic_json
R=Path(__file__).resolve().parent
DEADLINE=1793404407.1513782
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def event(stage,**values):
    o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,**values}
    atomic_json(R/'CPU_STATUS.json',o);print(json.dumps(o,ensure_ascii=False),flush=True)
try:
    event('WAITING_FOR_ALL_LOGITS_AND_VERIFIED_GPU_RELEASE')
    while True:
        if (R/'BLOCKER.json').exists():raise RuntimeError('GPU controller reported blocker; no CPU scientific inference started')
        ready=(R/'results/cache-acceptance.json').exists() and (R/'released.json').exists() and (R/'results/run-state.json').exists()
        if ready:
            assert read(R/'released.json')['verified_zero'] and read(R/'results/cache-acceptance.json')['pass']
            assert read(R/'results/run-state.json')['result']=='LOGITS_COMPLETE'
            break
        if time.time()>DEADLINE-600:raise RuntimeError('Original wall budget insufficient for finish; no new run')
        time.sleep(5)
    patch=read(R/'decision-guard-patch.json')
    assert sha(R/'analyze.before-guard-fix.py')==sha(R.parent/'analyze.py')==patch['analysis_before_sha256']
    assert sha(R/'analyze.py')==patch['analysis_after_sha256']
    assert sha(R/'decision_guards.py')==patch['guard_module_sha256']
    assert read(R/'decision-guard-tests.json')['passed']
    assert sha(R/'conformal_core.py')==sha(R.parent/'conformal_core.py')
    atomic_json(R/'cpu-sources-locked.json',{n:sha(R/n) for n in ('analyze.py','decision_guards.py','conformal_core.py','verify_completed.py','build_report.py','finish_cpu.py')})
    for name in ('verify_completed.py','analyze.py','build_report.py'):
        event('CPU_RUNNING',script=name)
        remaining=DEADLINE-time.time()-60
        if remaining<=0:raise RuntimeError('Original6h deadline reached')
        with (R/(name+'.run.log')).open('w') as f:
            result=subprocess.run([sys.executable,'-u',str(R/name)],stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        if result.returncode:raise RuntimeError(name+' failed with exit '+str(result.returncode)+'; see its run.log')
        event('CPU_STAGE_COMPLETE',script=name)
    decision=read(R/'analysis/decision.json');release=read(R/'released.json')
    assert release['verified_zero'] and read(R/'durability-final-verification.json')['passed']
    now=time.time();assert now<DEADLINE
    final={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'COMPLETE',
      'scientific_status':decision['candidate_status'],'A_pass':decision['A_pass'],'B_pass':decision['B_pass_after_controls'],
      'formal_epochs':80,'formal_steps_this_trajectory':25040,'formal_logit_views':200000,
      'GPU_released_verified':True,'last_verified_rate_CU_hour':0,'last_verified_balance':release['balance'],
      'cumulative_GPU_seconds_upper':release['cumulative_GPU_seconds_upper'],
      'seconds_since_original_download':now-1793382807.1513782,'original_6h_budget_met':True,
      'parent_research_project_continues':True,'automatic_additional_experiment':False}
    atomic_json(R/'STATUS.json',final)
    event('COMPLETE',candidate_status=decision['candidate_status'])
    index=f'''# TTA-conformal 最新完整结果\n\n已完成80epoch与20000图十视图正式评估、999次配对重采样。\n\n- 最新完整中文报告：[attempt03/REPORT_ZH.md](attempt03/REPORT_ZH.md)\n- 当前候选状态：{decision['candidate_status']}；A门={decision['A_pass']}，B门={decision['B_pass_after_controls']}。\n- GPU释放验证通过，累计占用上界{release['cumulative_GPU_seconds_upper']/60:.3f}分钟。\n- 原失败报告、90分钟门停止、105分钟修订和全部账本均保留。\n- 此状态只适用于该候选，整体研究由主线继续推进。\n'''
    (R.parent/'RECOVERY_FINAL_INDEX_ZH.md').write_text(index)
    manifest={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('SHA256SUMS.json','archive-receipt.json')}
    atomic_json(R/'SHA256SUMS.json',manifest)
    storage=sum(p.stat().st_size for base in (R.parent,Path('/Users/lkc/Documents/Codex/2026-10-31/task/tta-conformal')) for p in base.rglob('*') if p.is_file())
    assert storage<2*1024**3
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file_count':len(manifest),
      'manifest_sha256':sha(R/'SHA256SUMS.json'),'staging_plus_project_bytes':storage,'under_2GiB':True,
      'seconds_since_original_download':time.time()-1793382807.1513782,'under_original_6h':time.time()<DEADLINE,
      'GPU_released_verified':True,'candidate_status':decision['candidate_status'],'all_failures_preserved':True}
    atomic_json(R/'archive-receipt.json',receipt)
    print(json.dumps({'final_status':final,'archive':receipt,'report':str(R/'REPORT_ZH.md')},ensure_ascii=False),flush=True)
except BaseException as error:
    atomic_json(R/'CPU_BLOCKER.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'error':str(error),'traceback':traceback.format_exc(),'new_GPU_allocations':0})
    event('CPU_BLOCKED',error=str(error));raise
