"""Deliver verified new records and a current checkpoint; preserve all earlier research."""
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
OUT = RESEARCH/'causal-resume-20261022'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

def save(path,data):
    with path.open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def write(path,text):
    with path.open('x') as f:
        f.write(text)

def main():
    source_review = WORK/'independent-review'
    expected_review = {'verify_saved_run.py':'7ad407087c5baba6e39a544ff4800c31d5a237147de6542400fd55cad4793a6c',
        'result-verification.json':'7deb9f9764d8e0f46ab84342d1efcfc428b156bb078fa2b0062d222ec5ff596a',
        'lock-sequence-review.json':'ff8673f276469ab86111fef616d71655e489989f891606605542d159e3d73acb',
        'result-review-zh.md':'d0e7448e743451f27315fa6c436c2a35d09d61be6ee3004db2c242acd6378101'}
    for name,digest in expected_review.items():
        assert sha(source_review/name) == digest
    for source in sorted(source_review.iterdir()):
        target = OUT/'independent-review'/source.name
        if target.exists():
            assert sha(source) == sha(target)
        else:
            shutil.copyfile(source,target)
            assert sha(source) == sha(target)
    shutil.copyfile(__file__,OUT/'closeout.py')
    r = json.loads((OUT/'results.json').read_text())
    verification = json.loads((OUT/'independent-review/result-verification.json').read_text())
    sequence = json.loads((OUT/'independent-review/lock-sequence-review.json').read_text())
    assert verification['status'] == 'independent_saved_output_recomputation_passed'
    assert sequence['records_strictly_ordered']
    assert r['screen_passed'] is False and r['gates']['numerical_all22_passed'] and r['gates']['capability_passed']
    lock = json.loads((OUT/'protocol-lock.json').read_text())
    assert sha(OUT/'protocol.json') == lock['protocol_sha256']
    assert sha(OUT/'experiment.py') == lock['code_sha256']
    baseline = json.loads((OUT/'preservation-baseline.json').read_text())
    protected = []
    for row in baseline['previous_round_files']+baseline['classroom_assets']:
        path = PROJECT/row['relative_path']
        matched = path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        protected.append({'relative_path':row['relative_path'],'bytes':path.stat().st_size,'sha256':sha(path),'unchanged':matched})
        assert matched
    p = json.loads((OUT/'protocol.json').read_text())
    for name,digest in p['input_sha256'].items():
        path = Path(p['source_root'])/name
        assert sha(path) == digest
        protected.append({'relative_path':str(path.relative_to(PROJECT)),'sha256':digest,'unchanged':True})
    save(OUT/'preservation-final.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'protected_checks':protected,
        'all_unchanged':True,'prior_round_payload_files':19,'classroom_assets':3,'no_old_final_arrays_decoded':True,
        'no_8A_access':True})
    proc = subprocess.run(['ps','-axo','pid=,pcpu=,rss=,command='],text=True,capture_output=True,check=True)
    matches = []
    for line in proc.stdout.splitlines():
        items = line.strip().split(None,3)
        if len(items)==4 and str(PROJECT) in items[3] and int(items[0]) != os.getpid():
            matches.append({'pid':int(items[0]),'cpu_percent':float(items[1]),'RSS_KiB':int(items[2]),'command':items[3]})
    background = {'at_utc':datetime.now(timezone.utc).isoformat(),'project_matching_processes_excluding_current_closeout':matches,
        'background_experiment_count':len(matches),'experiment_exec_exit_code':0,'independent_reviewer_finished':True,
        'post_turn_scheduler_created':False,'pending_permission_or_uncertain_action':False,
        'scope':'Project path matching only; no inspection of8A files/process commands or assertion that all other Mac work stopped.'}
    save(OUT/'background-status.json',background)
    assert not matches, 'Unexpected project process: report before final delivery'
    outcome = {'at_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_verified_negative_development_screen',
        'mainline_questions_proposed':1,'independent_questions_proposed':1,'independent_proposal_admitted':False,
        'executed_frozen_screens':1,'new_readouts':22,'new_established_contributions':0,
        'primary_J_star':r['primary_J_star'],'gates':r['gates'],'screen_passed':False,
        'independent_recomputation_passed':True,'old_45_readouts_rerun':False,
        'actual_scientific_device':'single-thread CPU','scientific_internal_wall_seconds':r['wall_seconds'],
        'peak_RSS_bytes_Mac':r['peak_RSS_bytes_Mac'],'MPS_used':'preflight64x64 device smoke test only; no scientific MPS work',
        'next_checkpoint':'Fixed qB/qS×weak-pool claim closes. No parameter variation, new-data download or training job queued. Research line stays open for an independently motivated question with unused objects and label/measurement evidence.',
        'remaining_execution_blocker':None,
        'literature_gaps':['Geiping OpenReview paper body unknown after historical browser verification; not retried.',
            'PMC10923583 paper body unknown after this-round browser verification; stopped without alternate route.'],
        'background_experiment_count':0,'all_prior_results_and_classroom_assets_preserved':True,
        'requested_model_settings':p['requested_model']}
    save(OUT/'outcome.json',outcome)

    mean_risks = {kind:{arm:sum(r['risks'][kind+str(k)][arm] for k in range(5))/5 for arm in ('clean','weak')} for kind in ('B','S')}
    risk_table = '| 权重分配 | clean LL | weak LL | weak−clean |\n|---|---:|---:|---:|\n'
    risk_table += f"| U：均匀，描述参照 | {r['risks']['U']['clean']:.9f} | {r['risks']['U']['weak']:.9f} | {r['delta_U']['mean']:.9f} |\n"
    for kind,label in [('B','B：类内对齐，五配置均值'),('S','S：对象间抵消，五配置均值')]:
        risk_table += f"| {label} | {mean_risks[kind]['clean']:.9f} | {mean_risks[kind]['weak']:.9f} | {r['delta_'+kind+'_mean']['mean']:.9f} |\n"
    config_table = '| k | ΔB | ΔS | J* |\n|---|---:|---:|---:|\n'
    for row in r['configs']:
        config_table += f"| {row['k']} | {row['delta_B']:.9f} | {row['delta_S']:.9f} | {row['J_star']:.9f} |\n"
    report = '''# 因果数据增强恢复：联合条件关联的固定开发检验

2026-10-22 UTC。**Mac上已重新跑通；22个新读出全部收敛、独立复算一致，预设交互门未通过，0项新增已验证贡献。** 研究线仍是因果推断×数据增强，RandAugment作为基线。这个结果关闭当前固定权重/弱池设置，不宣告整个研究结束，也不把随机增强本身称作因果方法。

先核对的最新检查点是旧 `independent-design-challenge-20261021/finding-zh.md`：45模型的主量−0.004294nat、区间[−0.009271,+0.000683]、两边weak优于strong而无预设翻转。本轮未重训或重算那45头；19份旧交付及三份课堂PDF/HTML在开始、结束均逐字节/SHA保持一致。项目不含Git仓库/分支，在项目、子目录及检查过的父链未找到AGENTS。既有关闭表（亮度/背景配对、面积/Equalize、过滤源权重、CER外推、几何评分、立体crop/fill、抽样耦合及旧O/V）保持原判定。

本轮主线只提出一个联合条件关联问题；独立角色只提出一个真实方位覆盖与增强互补/冗余问题，后者因目前增量不足而否决，未训练。主线有条件准入一次开发测量；技术准入与原创性分开。设计与审查见 [协议](protocol.json)、[结果前独立审查](independent-review/mainline-qualification-zh.md) 及 [另一提案否决](independent-review/proposal-gate-zh.md)。

| 变量 | 实际含义与本轮边界 |
|---|---|
| O、Y | O是物理玩具，Y是其五类类别；O是独立来源单位。标签类别不由软件权重干预改变。 |
| A、E、L、X | 方位/仰角/光照条件及图像。依赖原装置控制与元数据；改变方位可同时改变遮挡和明暗，光照编号不是校准剂量。 |
| Q | 在既有固定图池中改变训练损失权重的可执行软件处理；本轮比较类内对齐B与同类对象间抵消S。 |
| G、θ | G是固定弱二维增强核，θ是同目标定义拟合的冻结表征分类头；观察其开发风险。没有恢复自然世界完整SCM或识别新的自然do效应。 |

精确问题是：同样15源物体、540独特照片、每物体36条件；类别/物体质量、类别内每单因素边际及无条件完整(A,E,L)联合均相同，图片权重直方图也匹配时，仅将A×L的权重关联在同类对象间对齐或抵消，是否使weak相对clean的风险差改变至少+.02nat？所有图始终有正权重，**没有制造支持缺失**。初稿U对比B存在普通权重集中度替代解释；在看任何新特征值/结果之前，独立审查触发同集中度S控制，并将U留为描述参照。没有改写旧已完成协议。

系数t=(−1,−.5,0,.5,1)，五个标签旋转配置。S矩阵、旧源物体ordinal及实际映射都锁定并用Fraction核验。每个B/S配置的五种图权重各108张，Kish值同为4500/11≈409.09；该数只是图权重集中度，不是独立对象数。每物体跨五配置的系数多重集相同，单配置内不必相同；S仍有对象×条件关联。可识别量因此限于这种有限分配处理与增强核的总交互，不能独指像素shortcut机制。各配置分别拟合后平均风险，没有先混合成均匀权重。

沿用旧低档五家族池：clean质量1/3，旋转±5°、横/纵切变±.05、横/纵平移±9px各1/15；它不是完整canonical RA。二维操作沿用物体类别标签，并假设仍可辨识；没有盲审逐张认证。原物体类别保持、旧裁切代理量小都不能替代语义可辨识性认证，也不能把二维仿射称作真实三维转动或重新打光。

精确枚举增强期望，float64、同零初值、λ=.01，LBFGS上限300迭代/450评估且梯度∞≤1e−5。U两头、B/S五配置各两头，共22头。原图数/总损失质量与求解精度公平；clean540行、weak5940行的枚举成本不同，实际优化迭代/耗时也不冒称完全相同。没有新增图像、特征或编码器调用。协议锁于12:40:35 UTC，22头全部冻结于12:40:48 UTC，随后才解码source-bank中的development_seen；不打开含final的evaluation-features归档。内部时间、哈希与静态顺序吻合，不冒称第三方预注册。

全部使用已消费的10个开发物体，每类2个、每物体36视图。先物体内平均，再类平衡；5个配置、360视图与22个处理臂不增加独立物体数。主要量J*=mean_k[(LLweak−LLclean)_B−(LLweak−LLclean)_S]。预锁门为J*≥+.02nat，且1024种类别内配对有放回对象重采样的95%稳定区间下界>0。

'''
    report += risk_table+'\n'+config_table+'\n'
    report += f'''主交互 **J*={r['primary_J_star']['mean']:.12f}nat**，对象稳定区间 **[{r['primary_J_star']['object_stability_interval95'][0]:.12f}, {r['primary_J_star']['object_stability_interval95'][1]:+.12f}]**。量级及区间两门均未通过，方向也不支持原正向预测。两种分配下weak相对clean的条件开发收益均约.0107nat；这项描述不替换主量，也不构成新的算法贡献。五个配置完整保留，不挑局部配置或类别再检验。

数值及测量能力门通过：22头最大梯度∞约1.54e−6；Uclean源准确率100%，B/S的clean开发准确率平均分别96.28%/96.44%。本次阴性不是未能学会分类。区间只反映这10个旧开发物体在固定源15物体/预训练表征/学习器下的重采样稳定性，不含新源对象或数据集总体不确定性；很窄也不能证明所有对象、完整RA或端到端训练等效/无效。

独立程序未导入主实验、未优化模型，从冻结输出重算22个加权源目标/梯度、开发logits/逐图LL/逐物体指标、所有配置和1024种重采样。logits最大差0，逐图LL最大差1.78e−15，源目标最大差5.55e−17；结论一致。见 [独立复算回执](independent-review/result-verification.json)、[独立脚本](independent-review/verify_saved_run.py)、[锁与时序核对](independent-review/lock-sequence-review.json)。脚本保留实际执行版本，输出采用独占创建；再次复核应使用新的本地输出目录，保留现有回执。

设备实测为Apple M5 Pro、64GiB、18逻辑CPU，既有隔离Python3.14.7、torch2.14.1、torchvision0.29.1、NumPy2.5.4。MPS可用，预检确实在mps:0执行64×64矩阵运算，含传输同步约.09748秒，CPU差0；**科学实验为单线程CPU**，程序内部科学阶段{r['wall_seconds']:.6f}秒、峰RSS{r['peak_RSS_bytes_Mac']/2**20:.4f}MiB，未将其冒充含解释器导入的完整端到端耗时。独立复算约.212秒。CUDA不可用；没有A100/Colab/ARIS、新依赖或8A访问。官方ResNet18权重SHA重核一致，现有Pets6份修复/9份亮度检查点仅列清单，本轮未加载推理。

查新限于精确问题的定向原文审查：[Wiles](https://arxiv.org/html/2110.11328v2)的B.2/D/E已研究smallNORB移位与增强，向量/复合属性框架可容纳联合因素；[NvN](https://arxiv.org/html/2210.01302)的C.3.2已有多干扰与调整；[Whac-A-Mole](https://arxiv.org/html/2212.04825v2)的2.1/3/5已有多捷径及增强放大风险，其UrbanCars采用分别相关且给定Y独立的捷径。独立审查另核读 [Jeon](https://arxiv.org/html/2501.04288v1) 的多属性变化/AA/RA。实读实验中未见当前严格集中度匹配的全套对照，这只支持一次有限测量，不证明原创；已知多捷径/联合依赖原理不作为新意。

实际查新边界仍在：历史 [Geiping原文](https://openreview.net/pdf?id=TMs-EYm4_ms) 的浏览器验证页没有正文，本轮0次重试/换路；本轮 [PMC原文入口](https://pmc.ncbi.nlm.nih.gov/articles/PMC10923583/) 也返回验证页后停止，检索片段没有替代全文认证。精确覆盖均未知，没有等待中下载或不明操作，均不是自动审批拒绝。硬件sysctl在普通沙箱首次读取失败，按原目标经host escalation批准后成功，已解除；其余执行没有权限阻塞。

下一检查点：当前固定命题按失败门关闭，不追加alpha、矩阵、幅度、类别、指标或旧对象变体。全50个smallNORB物体及旧Pets资料保持已消费身份，旧final/封存内容本轮未解码。确认需要未参与选题的新对象、足够对象级精度、事先规定的标签/测量审核及具体近邻增量；本次没有信号支持立即扩大本设置。后续问题应由独立的采集/机制证据给出可区分预测，再决定新的独立验证，不能用这轮开发输出挑有利测试。项目研究线开放，当前没有已锁定待跑任务。

交付包含协议、代码、22头、全部开发预测/结果、独立审查与复算、保留核验及后台状态。本项目后台实验进程0，未创建任何后续调度；无待批或结果不明操作。请求设置为GPT‑6.1‑Sol/max/Fast，独立spawn显式传入Sol/max；工具未暴露Fast或实际后端档位回执，不声称已验证每轮实际档位。课堂原论文/对比/真实图片可视化材料全部保留，8A仍优先。
'''
    write(OUT/'finding-zh.md',report)
    index = PROJECT/'CURRENT_CHECKPOINT_20261022.md'
    write(index,'''# RandAugment / 因果数据增强：当前检查点

2026-10-22 UTC。优先读本文件，再读旧CURRENT_EVIDENCE_STATUS_20261021.md的历史更正。仍走因果推断×数据增强；RandAugment是基线。**本轮22个CPU读出与独立复算完成，预设材料性交互未支持，0项新增已验证贡献。**

本轮问题：同样图/物体/条件分布及权重集中程度，将方位×光照关联在同类对象间对齐或抵消，是否改变固定弱增强池相对clean的收益。主J*=−0.000012674471nat，条件对象稳定区间[−0.000050033588,+0.000024684646]，未过预设+.02nat及区间门。22头收敛、测量能力通过；保持开发/确认区别，不外推完整RA或自然do效果。

- [本轮报告](causal-augmentation-research-20261020/causal-resume-20261022/finding-zh.md)
- [锁定协议](causal-augmentation-research-20261020/causal-resume-20261022/protocol.json) / [执行代码](causal-augmentation-research-20261020/causal-resume-20261022/experiment.py)
- [结果](causal-augmentation-research-20261020/causal-resume-20261022/results.json) / [独立复算](causal-augmentation-research-20261020/causal-resume-20261022/independent-review/result-verification.json)
- [状态与下一步](causal-augmentation-research-20261020/causal-resume-20261022/outcome.json) / [后台进程](causal-augmentation-research-20261020/causal-resume-20261022/background-status.json)
- [旧45头阴性检查点](causal-augmentation-research-20261020/independent-design-challenge-20261021/finding-zh.md)：主交互−.004294nat、旧预设翻转失败，19份交付保持字节/SHA一致；未重跑或改旧指标。

M5 Pro/64GiB/18逻辑CPU；MPS真实小矩阵预检成功，科学实验用既有隔离环境单线程CPU约4.968秒（程序内部科学阶段）、峰395.1MiB。无新编码/图像、依赖、A100/Colab/ARIS、8A或课堂材料修改。请求Sol/max/Fast，实际后端回执未暴露。

固定qB/qS×weak命题关闭，不追加旋钮追结果。全50个smallNORB物体都是已消费开发资料；旧final/封存数组本轮不打开。后续确认需新对象、对象级精度、标签/干预测量证据及独立动机。当前本项目后台实验0、无调度、无待执行训练或审批。

Geiping历史OpenReview验证页与本轮PMC验证页均未取得正文，保持停止、不换路，精确先例覆盖未知；没有运行权限阻塞或不明结果。原课堂PDF、两份HTML和全部旧研究结果保留；8A优先。
''')
    files = []
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            files.append({'relative_path':str(path.relative_to(OUT)),'bytes':path.stat().st_size,'sha256':sha(path)})
            if path.suffix == '.json':
                json.loads(path.read_text())
    manifest = {'at_utc':datetime.now(timezone.utc).isoformat(),'root':str(OUT),'file_count':len(files),
                'payload_bytes':sum(v['bytes'] for v in files),'files':files,
                'external_checkpoint':{'path':str(index),'bytes':index.stat().st_size,'sha256':sha(index)},
                'excludes':'delivery-manifest.json and delivery-verification.json, which are created after payload hashing'}
    save(OUT/'delivery-manifest.json',manifest)
    assert all((OUT/f['relative_path']).stat().st_size == f['bytes'] and sha(OUT/f['relative_path']) == f['sha256'] for f in files)
    delivery = {'at_utc':datetime.now(timezone.utc).isoformat(),'all_payload_bytes_and_sha256_match':True,
                'payload_files':len(files),'payload_bytes':manifest['payload_bytes'],'manifest_sha256':sha(OUT/'delivery-manifest.json'),
                'external_checkpoint_sha256':sha(index),'all_prior_and_classroom_checks_unchanged':True,
                'background_experiment_count':0,'independent_recomputation_passed':True,
                'scientific_effect_gate_passed':False,'new_established_contributions':0}
    save(OUT/'delivery-verification.json',delivery)
    print(json.dumps({'out':str(OUT),'checkpoint':str(index),**delivery},ensure_ascii=False))

if __name__ == '__main__':
    main()
