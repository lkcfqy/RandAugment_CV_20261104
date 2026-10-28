"""Add a broadened-scope proposal delivery; never modify prior results or datasets."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess

WORK=Path(__file__).resolve().parent
PROJECT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
RESEARCH=PROJECT/'causal-augmentation-research-20261020'
OUT=PROJECT/'augmentation-research-20261022'
INDEX=PROJECT/'CURRENT_CHECKPOINT_20261022_AUGMENTATION.md'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def save(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    assert not OUT.exists() and not INDEX.exists()
    assert sha(WORK/'independent-proposal-review.md')=='076913ec8fad637474fe907c03fa9561c6d5be2cdc306ed24361c485ba9c5bba'
    old_roots=[]
    for name,cp_key in [('causal-resume-20261022','external_checkpoint'),
                        ('causal-next-question-20261022','checkpoint'),
                        ('causal-mechanism-map-20261022','checkpoint')]:
        root=RESEARCH/name
        manifest=json.loads((root/'delivery-manifest.json').read_text())
        for row in manifest['files']:
            p=root/row['relative_path']
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        cp=manifest[cp_key]
        assert sha(Path(cp['path']))==cp['sha256']
        old_roots.append({'root':str(root),'payload_files':len(manifest['files']),
            'manifest_sha256':sha(root/'delivery-manifest.json'),'checkpoint':cp})
    protected=json.loads((RESEARCH/'causal-next-question-20261022/preservation-baseline.json').read_text())['protected_checks']
    for row in protected:assert sha(PROJECT/row['relative_path'])==row['sha256']
    preserve={'at_utc':datetime.now(timezone.utc).isoformat(),'previous_deliveries':old_roots,
        'previous_payload_files_unchanged':sum(x['payload_files'] for x in old_roots),
        'protected_checks':protected,'all_previous_bytes_and_sha256_match':True,
        'old_final_or_sealed_arrays_decoded':False,'8A_files_or_environment_accessed':False,
        'old_negative_claims_or_classroom_files_modified':False}
    save(WORK/'preservation-final.json',preserve)
    probe=json.loads((WORK/'mps-vit-probe.json').read_text())
    assert probe['MPS_available'] and not probe['MPS_fallback_enabled']
    assert len(probe['cases'])==4
    for r in probe['cases']:
        assert r['device']=='mps:0' and r['all_losses_finite']
        assert any(v>0 for v in r['sampled_backbone_parameter_changes'].values())
    assert {r['trainable_backbone_parameters'] for r in probe['cases'][:3]}=={36864}
    plan=json.loads((WORK/'candidate-data-plan.json').read_text())
    assert len(plan['rows'])==len({r['source_id'] for r in plan['rows']})==1776
    for part in plan['partitions']:
        selected=[r for r in plan['rows'] if r['candidate_partition']==part]
        assert len(selected)==592
        assert all(sum(r['class_id']==c for r in selected)==16 for c in range(37))
    assert sha(Path(plan['source_index']))==plan['source_index_sha256']
    old_ids={p.stem for p in RESEARCH.rglob('*.jpg')}
    assert all(r['source_id'] not in old_ids for r in plan['rows'])
    save(WORK/'readiness-verification.json',{
        'at_utc':datetime.now(timezone.utc).isoformat(),
        'MPS_forward_backward_parameter_update_cases':4,'synthetic_update_steps':16,
        'actual_MPS_device':'mps:0','CPU_fallback':False,
        'three_LoRA_layouts_have_equal_backbone_parameter_counts':True,
        'same_main_parameter_count':36864,
        'candidate_photo_ids':1776,'per_partition':592,
        'partitions_disjoint':True,'old_local_photo_ids_excluded':len(old_ids),
        'selection_used_no_new_RGB_or_outcome':True,
        'candidate_manifest_is_final_split_or_preregistration':False,
        'scientific_dataset_fits':0,'new_image_or_weight_payload_download_bytes':0})
    pg=subprocess.run(['pgrep','-f',str(PROJECT)],capture_output=True,text=True)
    assert pg.returncode in (0,1),pg.stderr
    exclude={os.getpid(),os.getppid()}
    pids=[int(x) for x in pg.stdout.splitlines() if x.strip() and int(x) not in exclude]
    lines=[]
    if pids:
        ps=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid=,pcpu=,rss=,command='],capture_output=True,text=True)
        assert ps.returncode in (0,1),ps.stderr
        lines=ps.stdout.splitlines()
    save(WORK/'background-status.json',{
        'at_utc':datetime.now(timezone.utc).isoformat(),
        'project_path_matching_processes_excluding_delivery':lines,
        'MPS_probe_process_finished':True,'metadata_network_checks_finished':True,
        'new_background_training_jobs':0,'scheduler_created':False,
        'independent_review_finished':True,'pending_or_uncertain_action':False,
        '8A_processes_or_files_inspected':False,
        'limitation':'Path-specific process check, not a claim about global CPU/GPU idleness.'})
    evidence=json.loads((WORK/'novelty-evidence.json').read_text())
    save(WORK/'outcome.json',{
        'at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'completed_primary_and_backup_proposals_with_actual_MPS_training_path_probe',
        'scope':'General data augmentation within the original RandAugment project.',
        'main':'Equal-parameter early/late QV-LoRA x fixed RandAugment clean-error interaction.',
        'main_decision':'conditional_go_to_data_QA_implementation_and_protocol_lock',
        'backup':'Identical post-CutMix augmented inputs with stale versus transported provenance-area targets, canonical-order reference.',
        'backup_decision':'reserve_only_high_prior_overlap_risk_not_parallel_training',
        'new_novelty_or_contribution_established':False,
        'dataset_training_runs':0,'actual_synthetic_MPS_parameter_update_steps':16,
        'new_images_or_weights_downloaded':False,'new_dependencies':False,
        'candidate_data_plan':'candidate-data-plan.json','candidate_split_formally_locked':False,
        'next_checkpoint':'Prepare only the primary data/label and duplicate QA plus full training implementation; freeze a completed development protocol before fitting. Do not start both questions or reuse old sealed/test data. No question/threshold tuning on old negatives.',
        'teacher_praise_or_publication_guaranteed':False,
        'old_results_classroom_and_prior_interpretations_unchanged':True,
        'all_new_work_remains_on_Mac':True,
        'A100_Colab_paid_ARIS_upload_publication':False,
        'technical_boundaries':evidence['technical_boundaries'],
        'unresolved_execution_permission_blocker_for_proposal':None,
        'sysctl_probe_remains_denied_stopped':True,
        'pending_or_uncertain_actions':False,
        'model_request':evidence['model_request']})
    OUT.mkdir()
    for p in sorted(WORK.iterdir()):
        if not p.is_file():continue
        if p.suffix=='.json':json.loads(p.read_text())
        target=OUT/p.name
        shutil.copyfile(p,target)
        assert sha(target)==sha(p)
    with INDEX.open('x') as f:
        f.write('''# RandAugment扩围研究检查点

2026-10-22 UTC。用户明确研究不限因果。仍在原RandAugment项目，旧阴性、因果研究记录、课堂与封存资料保留，不把教师评价当作结果保证。

- [主方案与唯一备选](augmentation-research-20261022/proposal-zh.md)：主推同参数预算的前/后层Q/V-LoRA × 固定RA训练收益交互；备选是同RGB的CutMix后增强来源面积目标一致性。
- [独立提案与挑战](augmentation-research-20261022/independent-proposal-review.md) / [近邻证据与访问边界](augmentation-research-20261022/novelty-evidence.json)
- [设计草案](augmentation-research-20261022/design-draft.json) / [候选数据清单](augmentation-research-20261022/candidate-data-plan.json)：排除177已有JPEG ID；训练/开发/预留各592，仅元数据，非最终测试锁。
- [实际MPS兼容计时](augmentation-research-20261022/mps-vit-probe.json) / [实现代码](augmentation-research-20261022/mps_vit_probe.py)：4布局16步真实mps:0反向更新，非数据集效果。
- [状态](augmentation-research-20261022/outcome.json) / [保留核验](augmentation-research-20261022/preservation-final.json) / [后台](augmentation-research-20261022/background-status.json)

主方案conditional GO到数据QA、实现和正式协议锁定；尚无数据集训练、未宣布原创机制或新贡献。不是只做冻结特征：适配器/完整骨干训练路径已实际检查。近邻Surgical FT、Hint-Aug、DynamicAug和CoCa直接覆盖相关动机；精确位置×RA经验差异仍待验证。备选与OA-CutMix重叠风险高，不并行开训。

下一步只推进主线：核验新公共照片/标签、重复组、增强可辨识性及实际loader/AdamW资源，再锁定完整协议。计划4核心格10epochs开发，2小时墙钟上限；完整30runs/最多30epochs/12小时累计上限，均含明确优化与停止边界。欠训练/预算耗尽记测量不足，不能变科学阴性。8A优先，单MPS进程、受限CPU/内存和计算节奏；A100仍需另问。

旧34+12+12份交付与3旧检查点、19旧回合/3课堂/3源输入SHA保持。旧J*解释v2与代理候选否决不改，不解封旧final，不重跑旧smallNORB。0新图片/权重/依赖下载，0后台训练。sysctl权限读拒绝后停；官方Pets二进制浏览不支持与308属于技术格式/重定向，固定镜像HEAD200。历史验证/403与不可访问目标不绕过。

所有资料与代码在Mac。无A100/Colab/付费/ARiS/私有上传/发布，未操作8A。Sol/max与既有Fast请求配置不变，实际后端档位回执未暴露。
''')
    rows=[{'relative_path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    m={'at_utc':datetime.now(timezone.utc).isoformat(),'root':str(OUT),'files':rows,
        'payload_files':len(rows),'payload_bytes':sum(r['bytes'] for r in rows),
        'checkpoint':{'path':str(INDEX),'bytes':INDEX.stat().st_size,'sha256':sha(INDEX)},
        'excludes':'delivery-manifest.json and delivery-verification.json created after payload hashing'}
    save(OUT/'delivery-manifest.json',m)
    v={'at_utc':datetime.now(timezone.utc).isoformat(),
        'payload_files':len(rows),'payload_bytes':m['payload_bytes'],
        'all_payload_bytes_sha256_match':all((OUT/r['relative_path']).stat().st_size==r['bytes'] and sha(OUT/r['relative_path'])==r['sha256'] for r in rows),
        'manifest_sha256':sha(OUT/'delivery-manifest.json'),'checkpoint_sha256':sha(INDEX),
        'previous_payload_files_unchanged':58,'protected_other_files_unchanged':25,
        'new_scientific_dataset_fits':0,'synthetic_MPS_update_steps':16,
        'new_background_training_jobs':0,'project_path_processes_seen':len(lines)}
    assert v['all_payload_bytes_sha256_match']
    save(OUT/'delivery-verification.json',v)
    print(json.dumps({'root':str(OUT),'checkpoint':str(INDEX),**v},ensure_ascii=False))

if __name__=='__main__':main()
