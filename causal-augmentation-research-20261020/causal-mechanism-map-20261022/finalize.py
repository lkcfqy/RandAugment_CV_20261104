"""Archive a decision map and audit evidence. Does not decode images or run models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess

WORK = Path('/Users/lkc/Documents/Codex/2026-10-22/task-2/mechanism-map-20261022')
PROJECT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
RESEARCH = PROJECT/'causal-augmentation-research-20261020'
OUT = RESEARCH/'causal-mechanism-map-20261022'
INDEX = PROJECT/'CURRENT_CHECKPOINT_20261022_MAP.md'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def main():
    assert not OUT.exists() and not INDEX.exists(), 'Never overwrite a prior delivery.'
    assert sha(WORK/'independent-review.md')=='013e8e9016e5c6565e037b38caabd29b5b01e6fd161bf624d7c1b4c93595fc73'
    baseline=json.loads((WORK/'preservation-baseline.json').read_text())
    for old in baseline['previous_deliveries']:
        root=Path(old['root'])
        assert sha(root/'delivery-manifest.json')==old['manifest_sha256']
        manifest=json.loads((root/'delivery-manifest.json').read_text())
        for row in manifest['files']:
            p=root/row['relative_path']
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        assert sha(Path(old['checkpoint']['path']))==old['checkpoint']['sha256']
    for row in baseline['protected_checks']:
        assert sha(PROJECT/row['relative_path'])==row['sha256']

    # Independent root-side recomputation of the reviewer's 64 matching pairs.
    pets=RESEARCH/'area-pairing-cpu-20261021/mechanism-transfer-pets-20261021'
    photos=json.loads((pets/'photo-fetch-receipt.json').read_text())['files']
    masks={r['source_id']:r for r in json.loads((pets/'mask-fetch-receipt.json').read_text())['files']}
    assert len(photos)==64 and len({r['source_id'] for r in photos})==64
    paired=[]
    for r in photos:
        m=masks[r['source_id']]
        local_photo=pets/'photos'/Path(r['entry']).name
        local_mask=pets/'masks'/Path(m['entry']).name
        for p,record in ((local_photo,r),(local_mask,m)):
            assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
        paired.append({'photo_id':r['source_id'],
                       'photo':{'path':str(local_photo),'bytes':r['bytes'],'sha256':r['sha256']},
                       'mask':{'path':str(local_mask),'bytes':m['bytes'],'sha256':m['sha256']}})
    inventory={'at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'root_recomputed_all128_bytes_and_sha256_match_existing_receipts',
        'photo_files':64,'paired_mask_files':64,
        'photo_bytes':sum(r['photo']['bytes'] for r in paired),
        'paired_mask_bytes':sum(r['mask']['bytes'] for r in paired),
        'new_image_download_bytes_to_reuse_these_consumed_files':0,
        'images_or_masks_decoded_this_round':False,
        'edited_label_truth_verified_this_round':False,
        'animal_identity_verified':False,'fresh_confirmation':False,
        'scope':'Consumed finite-photo-pool diagnostic route with locally procedural backgrounds.',
        'pairs':paired}
    assert inventory['photo_bytes']==6373398 and inventory['paired_mask_bytes']==172720
    save(WORK/'local-data-check.json',inventory)
    save(WORK/'preservation-final.json',{
        'at_utc':datetime.now(timezone.utc).isoformat(),
        'old34_and12_payload_files_bytes_sha256_unchanged':True,
        'old19_classroom3_and_source3_sha256_unchanged':True,
        'old_root_checkpoints_unchanged':True,
        'protected_checks':baseline['protected_checks'],
        'old_final_or_sealed_arrays_decoded':False,
        'old_jstar_or_proxy_candidate_or_memo_v2_modified':False})

    # Check named project processes, without inspecting another project's files.
    pg=subprocess.run(['pgrep','-f',str(PROJECT)],capture_output=True,text=True)
    assert pg.returncode in (0,1),pg.stderr
    ancestors={os.getpid(),os.getppid()}
    pids=[int(x) for x in pg.stdout.splitlines() if x.strip() and int(x) not in ancestors]
    process_lines=[]
    if pids:
        ps=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid=,pcpu=,rss=,command='],capture_output=True,text=True)
        assert ps.returncode in (0,1),ps.stderr
        process_lines=ps.stdout.splitlines()
    save(WORK/'background-status.json',{
        'at_utc':datetime.now(timezone.utc).isoformat(),
        'project_path_matching_processes_excluding_delivery':process_lines,
        'background_experiments_started_this_round':0,
        'new_scientific_model_or_toy_runs':0,
        'independent_review_finished':True,'scheduler_created':False,
        'pending_or_uncertain_operation':False,
        '8A_files_environment_or_processes_inspected':False,
        'limitation':'Path-name process check plus action history; no global CPU idleness claim.'})
    ledger=json.loads((WORK/'evidence-ledger.json').read_text())
    save(WORK/'outcome.json',{
        'at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'completed_three_failure_family_decision_map_training_no_go',
        'main_deliverable':'shortlist-zh.md','details':'decision-details-zh.md',
        'failure_families_ranked':3,'new_research_candidates_admitted':0,
        'new_established_contributions':0,
        'preferred_retained_direction':'Original paired cross-editor training-benefit transfer; no new project.',
        'scientific_stop':ledger['scientific_decision']['primary_missing_evidence'],
        'existing_data_audit_route_available':True,
        'minimum_additional_image_download_bytes_for_existing_route':0,
        'data_limits':'Consumed photos support exploration only; animal identities, edited-label recognizability and natural-background causal semantics not verified.',
        'training_permission_blocker':None,'new_training_protocol_frozen':False,
        'new_model_or_toy_runs':0,'new_images_weights_or_dependencies':False,
        'old_negative_and_corrected_interpretation_preserved':True,
        'blocked_reads':ledger['protected_boundaries'],
        'new_blocked_reads_stopped_no_alternates':True,
        'pending_or_uncertain_actions':False,
        'next_checkpoint':'No training queue. Retain one original cross-editor question; scientific discrimination and bounded paired data/label audit must precede a new fitting protocol. Do not unlock old final data or rerun closed designs.',
        'model_request':ledger['model_request'],
        'resources':ledger['resource_scope']})
    OUT.mkdir()
    for source in sorted(WORK.iterdir()):
        if source.is_file():
            if source.suffix=='.json':json.loads(source.read_text())
            target=OUT/source.name
            shutil.copyfile(source,target)
            assert sha(source)==sha(target)
    INDEX.write_text('''# 因果数据增强：机制与数据决策图检查点

2026-10-22 UTC。继续原RandAugment研究目录，RA为基线，尚无新贡献。本续轮是短决策图，未训练。

- [短名单](causal-augmentation-research-20261020/causal-mechanism-map-20261022/shortlist-zh.md)：按本项目可证伪性／准备成本，编辑来源捷径、多捷径覆盖、标签损伤三家族排序；非三个新pilot。
- [近邻与资格细节](causal-augmentation-research-20261020/causal-mechanism-map-20261022/decision-details-zh.md) / [来源与访问边界](causal-augmentation-research-20261020/causal-mechanism-map-20261022/evidence-ledger.json)
- [独立审查](causal-augmentation-research-20261020/causal-mechanism-map-20261022/independent-review.md) / [本机128文件独立复核](causal-augmentation-research-20261020/causal-mechanism-map-20261022/local-data-check.json)
- [状态](causal-augmentation-research-20261020/causal-mechanism-map-20261022/outcome.json) / [保留核验](causal-augmentation-research-20261020/causal-mechanism-map-20261022/preservation-final.json) / [后台](causal-augmentation-research-20261020/causal-mechanism-map-20261022/background-status.json)

下一项模型实验NO-GO：具体经验预测尚不能区分ForAug/FFR等已知解释。优先保留原配对跨编辑器问题；不要求发明新算法，不以重复已知现象作为贡献。可用旧Pets64图+64对应mask与程序背景做有限池工程/数据审计，新增图片下载0字节；不是自然背景因果或新动物确认。

旧[解释检查点](CURRENT_CHECKPOINT_20261022_CONTINUED.md)、[22头结果检查点](CURRENT_CHECKPOINT_20261022.md)、34+12份交付、旧19份与课堂3份材料SHA保持。J*修正版与代理错误候选否决不改；不解封旧final，不重跑NORB、筛阈值或大规模新方向。

本轮0模型/形式toy实验、0图片/权重/依赖下载、0后台实验。新HF README和dSprites API读取不可访问后停止；历史验证/403不换路，Places未重访。8A保持最高优先；无A100/Colab/付费/ARiS/上传/发布。Sol/max及既有Fast请求设置未改，实际后端档位回执未暴露。
''')
    rows=[{'relative_path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    manifest={'at_utc':datetime.now(timezone.utc).isoformat(),'root':str(OUT),
        'files':rows,'payload_files':len(rows),'payload_bytes':sum(x['bytes'] for x in rows),
        'checkpoint':{'path':str(INDEX),'bytes':INDEX.stat().st_size,'sha256':sha(INDEX)},
        'excludes':'delivery-manifest.json and delivery-verification.json created after payload hashing'}
    save(OUT/'delivery-manifest.json',manifest)
    verification={'at_utc':datetime.now(timezone.utc).isoformat(),
        'payload_files':len(rows),'payload_bytes':manifest['payload_bytes'],
        'all_payload_bytes_and_sha256_match':all((OUT/x['relative_path']).stat().st_size==x['bytes'] and sha(OUT/x['relative_path'])==x['sha256'] for x in rows),
        'manifest_sha256':sha(OUT/'delivery-manifest.json'),'checkpoint_sha256':sha(INDEX),
        'local_128_file_check_passed':True,'old_results_and_classroom_unchanged':True,
        'training':'no_go','new_experiments':0,'background_experiments_started':0,
        'project_processes_seen':len(process_lines)}
    assert verification['all_payload_bytes_and_sha256_match']
    save(OUT/'delivery-verification.json',verification)
    print(json.dumps({'out':str(OUT),'checkpoint':str(INDEX),**verification},ensure_ascii=False))

if __name__=='__main__':
    main()
