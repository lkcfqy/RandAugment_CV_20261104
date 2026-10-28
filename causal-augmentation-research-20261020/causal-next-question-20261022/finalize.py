"""Archive a no-go research qualification without changing the prior negative experiment."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
PROJECT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
RESEARCH = PROJECT/'causal-augmentation-research-20261020'
PREVIOUS = RESEARCH/'causal-resume-20261022'
OUT = RESEARCH/'causal-next-question-20261022'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def save(path,data):
    with path.open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    expected = {'memo-review-zh.md':'7f6e0206a2121a32c216f6188eae1d41dd7abb48976886f5d39d8a9bc9f2e34d',
                'candidate-qualification-zh.md':'4b3e656d16320bdd5047a913c13ddc2c4fb39cbb6f91ddb103603e75aa225d33'}
    for name,digest in expected.items():
        assert sha(WORK/'independent-review'/name)==digest
    old=json.loads((PREVIOUS/'delivery-manifest.json').read_text())
    assert all((PREVIOUS/r['relative_path']).stat().st_size==r['bytes'] and sha(PREVIOUS/r['relative_path'])==r['sha256'] for r in old['files'])
    old_index=Path(old['external_checkpoint']['path'])
    assert sha(old_index)==old['external_checkpoint']['sha256']
    preserved=json.loads((OUT/'preservation-baseline.json').read_text())
    for row in preserved['protected_checks']:
        assert sha(PROJECT/row['relative_path'])==row['sha256']
    copied=[]
    for source in sorted(WORK.rglob('*')):
        if not source.is_file():
            continue
        relative=source.relative_to(WORK)
        target=OUT/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert sha(target)==sha(source)
        else:
            shutil.copyfile(source,target)
        assert sha(target)==sha(source)
        copied.append({'relative_path':str(relative),'bytes':source.stat().st_size,'sha256':sha(source)})
    gate=json.loads((OUT/'prior-and-data-gate.json').read_text())
    save(OUT/'review-resolution.json',{
        'at_utc':datetime.now(timezone.utc).isoformat(),'independent_reviews_sha256':expected,
        'original_memo_sha256':sha(OUT/'interpreting-jstar-zh.md'),
        'current_memo_v2_sha256':sha(OUT/'interpreting-jstar-zh-v2.md'),
        'resolved':['Replace overstrong reject-large-effect wording with unobserved-under-fixed-setting and engineering stop.',
                    'Clarify first-order cancellation is local and not a confirmed explanation at alpha.8.',
                    'Add gradient tolerance does not establish exact-optimum sign stability for1e-5nat interaction.'],
        'old_experiment_or_threshold_changed':False,'original_memo_retained':True,
        'candidate_independent_decision':'no_go','root_agrees':True,
        'v2_full_second_review_claimed':False})
    pg=subprocess.run(['pgrep','-f',str(PROJECT)],capture_output=True,text=True)
    assert pg.returncode in (0,1),pg.stderr
    pids=[int(line) for line in pg.stdout.splitlines() if line.strip() and int(line)!=os.getpid()]
    processes=[]
    if pids:
        ps=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid=,pcpu=,rss=,command='],capture_output=True,text=True,check=True)
        processes=ps.stdout.splitlines()
    assert not processes,processes
    save(OUT/'background-status.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'project_processes_excluding_current_delivery':processes,'background_experiment_count':0,
        'new_fitting_started':False,'independent_review_finished':True,'scheduler_created':False,
        'pending_or_uncertain_action':False,'8A_project_or_files_inspected':False})
    outcome={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_memo_and_single_candidate_no_go',
        'memo_current':'interpreting-jstar-zh-v2.md','candidate_count':1,'admitted_candidates':0,
        'new_model_experiments':0,'new_toy_experiments':0,'old_smallNORB_design_repeated':False,
        'new_established_contributions':0,'new_images_or_weights_downloaded':False,'installs':False,
        'scientific_stopping_reason':gate['scientific_gate']['reason'],
        'data_gate_passed':False,'data_blockers':['Places full terms not obtained after two ordinary timeouts.',
            'No verified minimal aligned foreground-mask-donor and source-ID manifest with exact download bytes.',
            'Stored ecological labels contain author-acknowledged mapping errors; edited-label and nuisance-semantics evidence unverified.'],
        'execution_permission_blocker':None,'blocked_read':gate['data_gate']['Places_terms'],
        'historical_denied_or_verification_sources_retried':False,
        'background_experiment_count':0,
        'next_checkpoint':'Current single candidate no-go. Do not repeat oldNORB, tune its gate, queue training, download heavy data or substitute a second question. Reopening requires a distinct pre-result mechanism beyond error-placement/edit-coverage, plus public data/label readiness.',
        'model_request':gate['model_request'],'A100_paid_ARIS_private_sharing_8A':False}
    save(OUT/'outcome.json',outcome)
    save(OUT/'preservation-final.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'old34_payload_files_bytes_sha256_unchanged':True,
        'old_delivery_manifest_sha256':sha(PREVIOUS/'delivery-manifest.json'),
        'old_root_checkpoint_sha256':sha(old_index),
        'previous19_round_files_and3_classroom_assets_and3_source_inputs_unchanged':True,
        'protected_checks':preserved['protected_checks'],
        'old_final_or_sealed_arrays_decoded':False,'previous_results_recomputed':False,
        'all_new_source_copies_match':True,'copied_files':copied})
    index=PROJECT/'CURRENT_CHECKPOINT_20261022_CONTINUED.md'
    with index.open('x') as f:
        f.write('''# 因果数据增强：解释与下一问题资格检查点

2026-10-22 UTC。研究继续走因果推断×数据增强；RandAugment作基线。**上轮阴性保留，本续轮完成解释和一个新候选的资格审查，未启动新训练。**

- [当前短备忘录v2](causal-augmentation-research-20261020/causal-next-question-20261022/interpreting-jstar-zh-v2.md)：J*近零是两种分配的增强收益接近，不是增强收益为零；一阶平均抵消是局部设计性质，不能确认解释alpha=.8的实测值；.02nat是工程投入门，不是统计显著性/功效或等效性界。
- [唯一候选与停止决定](causal-augmentation-research-20261020/causal-next-question-20261022/candidate-and-stopping-decision-zh.md)：代理总体/类别准确率相同，错误位置是否改变定向背景干预相对完整RA的收益。NO-GO：已有精确代理错误结构先例，固定K又没固定真实新增反例数，尚无超出已知机制的具体预测。
- [原始先例与公共数据预检](causal-augmentation-research-20261020/causal-next-question-20261022/prior-and-data-gate.json)：BARACK/GIC/AutoBackSwap等正文可读；CUB非商业科研教育条款可读、含ImageNet重叠警告；Waterbirds作者披露三种鸟类别映射错误；Places条款页两次普通timeout后停止，最小合法配对包与标签证据未齐。
- [独立备忘录审查](causal-augmentation-research-20261020/causal-next-question-20261022/independent-review/memo-review-zh.md) / [独立候选NO-GO](causal-augmentation-research-20261020/causal-next-question-20261022/independent-review/candidate-qualification-zh.md)
- [本轮状态](causal-augmentation-research-20261020/causal-next-question-20261022/outcome.json) / [保留核验](causal-augmentation-research-20261020/causal-next-question-20261022/preservation-final.json) / [后台状态](causal-augmentation-research-20261020/causal-next-question-20261022/background-status.json)

旧[22头检查点](CURRENT_CHECKPOINT_20261022.md)与其34份交付、旧19份回合和课堂三份材料均保持字节/SHA一致。旧J*=−0.0000126745nat，条件区间[−0.0000500336,+0.0000246846]；不重跑、不改门槛、不把极小符号变成机制。

当前0项新准入、0项新模型/形式toy实验、0后台实验，无待执行协议/下载/审批或不明动作。科学增量门在先；即便补齐数据，仍需事前可区分预测。许可链、原料/主体映射、真实标签与干预有效性是另一个独立未通过门；算力不是当前阻塞。未读旧final/封存数组，8A仍最高优先。

没有A100/Colab/付费/ARiS/私有分享/发布、新依赖或8A文件操作。历史Geiping、PMC与403/受限作者入口保持停止，不重试不换路。Sol/max及既有Fast请求配置未改变，实际后端回执没有暴露。
''')
    files=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            if path.suffix=='.json':json.loads(path.read_text())
            files.append({'relative_path':str(path.relative_to(OUT)),'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'at_utc':datetime.now(timezone.utc).isoformat(),'root':str(OUT),'files':files,
        'payload_files':len(files),'payload_bytes':sum(v['bytes'] for v in files),
        'checkpoint':{'path':str(index),'bytes':index.stat().st_size,'sha256':sha(index)},
        'excludes':'delivery-manifest.json and delivery-verification.json created after payload hashing'}
    save(OUT/'delivery-manifest.json',manifest)
    assert all((OUT/r['relative_path']).stat().st_size==r['bytes'] and sha(OUT/r['relative_path'])==r['sha256'] for r in files)
    verification={'at_utc':datetime.now(timezone.utc).isoformat(),'all_payload_bytes_and_sha256_match':True,
        'payload_files':len(files),'payload_bytes':manifest['payload_bytes'],'manifest_sha256':sha(OUT/'delivery-manifest.json'),
        'checkpoint_sha256':sha(index),'old_results_and_classroom_unchanged':True,'new_experiments':0,
        'candidate_decision':'no_go','background_experiment_count':0}
    save(OUT/'delivery-verification.json',verification)
    print(json.dumps({'out':str(OUT),'checkpoint':str(index),**verification},ensure_ascii=False))

if __name__=='__main__':
    main()
