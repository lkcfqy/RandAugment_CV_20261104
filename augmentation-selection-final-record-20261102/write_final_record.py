"""Project conclusion and evidence index from the already completed synthesis."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
PROJECT=_portable_project_path(__file__, '')
r=json.loads((P/'COMBINED_RESULTS.json').read_text());idx=json.loads((P/'SOURCE_INDEX.json').read_text());life=json.loads((P/'LIFECYCLE.json').read_text())
label={'dinov2_small':'DINOv2-S','deit_tiny':'DeiT-T'};pol={'clean':'S','weak':'W','ra':'R'}
md=['# 固定标签预算下增强选择：本轮最终研究结论','','本轮完成两数据集、两冻结编码器、两标签预算下的 96 个设置、8 个组。**获得了可复算的经验记录，尚未得到可靠论文贡献。** 在这个固定三策略池中，CV 的均值未超过每组事后均值最高的固定策略；但它能胜过部分事前固定基线，且哪一种固定策略较强依模型和数据集变化。这不支持“CV 普遍无效”、新的误差机制或新算法收益。','',
'这项工作的可复用部分是：保持标签预算、候选策略和选择规则不变，完整保存所有固定策略、CV 选择、查询预测、并列敏感性及折到全标签的排序。所有负结果和未复现描述均保留；耗时不是贡献证据。','',
'令 Q(e,p) 为支持集 e 下策略 p 的全标签头查询准确率，O(e)=max_p Q(e,p)。H=mean_e O(e)−max_p mean_e Q(e,p)，C=mean_e[O(e)−Q(e,CV)]。于是：','',
'`C − H = max_p mean_e Q(e,p) − mean_e Q(e,CV)`。','',
'所以 C>H 只是“CV 低于本组事后最佳固定均值”的等价表述，不能包装成新机制或定理。H 和并列上下界都用到了查询标签；事后最佳固定策略也不是事先已知、可直接部署的选择规则。实际有信息量的是它们的规模、完整逐例分布和适用范围。','',
'S=标准预处理；W=普通 RRC+flip；R=W 后固定 RandAugment。B=每类8或16标签，全部组均包含12套预先固定支持集。表内差值单位 pp；“折阶段已差”比较的是 CV 所选与最终最佳之间的同一对策略。','',
'|数据集 / 编码器 / B|事后最佳固定|CV−W|CV−R|H|C|全部并列候选 C 范围|折阶段已差/全标签错选|','|:---|:---:|---:|---:|---:|---:|:---|:---:|']
for key in [f'{d}:{m}:B{b}' for d in ['EuroSAT','CIFAR-100'] for m in ['dinov2_small','deit_tiny'] for b in [8,16]]:
 s=r['summaries'][key]
 lo,hi=s['tie_bounds']['regret_pp'];d=s['distributions'];name=f"{s['dataset']} / {label[s['encoder']]} / {s['B']}"
 md.append(f"|{name}|{'/'.join(pol[x] for x in s['best_constant_expost'])}|{d['selected_minus_weak_pp']['mean']:+.4f}|{d['selected_minus_ra_pp']['mean']:+.4f}|{s['H_pp']:.4f}|{s['C_pp']:.4f}|{lo:.4f}–{hi:.4f}|{s['same_pair_fold_inferior_wrong_count']}/{s['wrong_full_count']}|")
md+=['',
'八组均 C>H，给予查询标签来处理全部 CV 并列时仍如此。但 CIFAR-100 的 H 只有0.0033–0.0484 pp，三策略平均差异并不等于按支持集切换策略的额外空间。CIFAR-100 的 RA 较弱，因此四组 CV−RA 为正；CV−W 三组为正、一组为负。EuroSAT 上 DINO 的 W 均值较高、DeiT 的 R 较高；CIFAR-100 四组均为 S。不能据此选择一个跨任务“总是最好”的固定策略。','',
'保留的边界和反证：','',
'- **并列敏感性完整保留。** EuroSAT DINO/B8 的 episode4、7 是 W/R 并列，4种完整组合使 C 在0.6917–0.8583 pp；EuroSAT DeiT/B16 的 episode6、9 是 S/W 并列，4种组合使 C 在0.6417–1.2417 pp。后一组 CV−W 可从−0.55变为+0.05，但最有利端点仍低于固定R 0.55 pp。其它6组无最优CV并列。原S>W>R规则未改变，最有利端点不是推荐规则。',
'- **阶段描述不稳定。** EuroSAT四组以及CIFAR三个组中，多数全标签错选在同对折查询上已落后；CIFAR DeiT/B8仅3/7，未满足事前多数条件，其余4个错误在折查询时反而更高。不能把所有错误归为验证噪声，也不能把任意第三策略的差异算作同一对比较。',
'- **预算描述未完全复现。** H随B从8增到16，EuroSAT两模型和CIFAR DINO下降；CIFAR DeiT却由0.0033增至0.0142 pp。EuroSAT DeiT的C随预算增加，其他三对下降，因此没有统一的预算因果结论。',
'- **依赖与混杂保留。** 两编码器共享各数据集查询和支持身份；支持集相交、B嵌套、EuroSAT图块可能相关。96设置不是96个独立试验。EuroSAT每折验证20/40图，CIFAR-100是200/400图；类别数、总标签量、分辨率和域一同变化，不能称纯域效应。两数据集均已有历史使用，属于探索性；没有独立确认、伪独立显著性或跨任务普适性声明。',
'',
'最窄可支持陈述是：**在已见的EuroSAT/CIFAR-100、DINOv2-S/DeiT-T、固定三个输入增强策略、固定ridge和这批支持集下，四折CV未超过各组事后最佳固定均值；收益依参照策略而变，按支持集切换的潜在空间有限，错误出现的阶段也不一致。** 这是一条有边界的经验观察，不是“首次发现验证噪声、标签成本或训练规模失配”。','',
'与已有工作的关系及发表前仍缺的证据：','',
'|已有工作|已知问题/方法与本轮边界|仍缺的证据（不是待执行计划）|','|:---|:---|:---|',
'|[Evaluating the Evaluators](https://arxiv.org/abs/2307.02732)，TMLR 2024|少样本性能估计与模型选择已是研究对象；一次CV错选或验证/查询排序不同不是新发现。|需要说明本观察相对已知评估问题增加何种可推广知识，并在未参与探索的任务上前瞻检验。|',
'|[CLAP](https://openaccess.thecvf.com/content/CVPR2024/html/Silva-Rodriguez_A_Closer_Look_at_the_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2024_paper.html)，CVPR 2024|少样本适配中验证集及任务特定调参成本已有讨论；本轮是冻结DINO/DeiT的ridge，不是CLAP复现。|需在对齐总标签、计算预算和选择对象后，与可部署的强固定/选择方法比较；不能只比一个较弱RA基线。|',
'|[VEST](https://arxiv.org/html/2506.04713v4)|通过检索OOD与ID表现选择CLIP微调checkpoint/layer；本轮无外部检索、无编码器微调，选择的是像素增强。|需证明无外部数据条件下的经验边界确有额外价值；若作性能比较须明确资源与任务差异，不能直接对比论文榜单。|',
'|[FroFA](https://arxiv.org/abs/2403.10519)，CVPR 2024|冻结表示上的增强已有强基础；本轮在输入像素上做增强，接口不同不自动构成创新，也未重启旧高斯替身实验。|需有真实冻结表示/输入增强的公平强对照及稳健效应，超出“模型偏好不同增强”的已有常识。|',
'',
'进一步谈发表还需前瞻保留且未用于设计的任务/数据；把类别数和总验证量等因素拆开；对相关支持/查询采用有效的不确定性设计；验证一个可证伪、可迁移且有实践意义的预测或改进。当前两数据集、两编码器、单一ridge正则和三策略池不足以完成这些论证。这些是证据缺口，不是承诺的后续实验；当前没有第四轮或其它待启动任务。论文阅读范围沿用已归档记录，本步没有新查新或下载。','',
'全部96行、8组完整固定策略均值、选择频率、各组12个配对差值、4个并列设置的全部组合和阶段原始计数见 [ALL_96_SETTINGS.md](ALL_96_SETTINGS.md)、[ALL_8_GROUPS.csv](ALL_8_GROUPS.csv)、[COMBINED_RESULTS.json](COMBINED_RESULTS.json)。原协议、实现、逐例预测、图表、测试及生命周期见 [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md)。本步只整理已有结果，未运行模型/提取特征/拟合/下载/上传/安装或新增CU；RandAugment已有演讲未修改。']
(P/'CONCLUSION_ZH.md').write_text('\n'.join(md)+'\n')
# Complete tie sensitivity table and assignments are deliberately not compressed into a best endpoint.
t=['# 八组完整并列敏感性','','原S>W>R为主结果。下表及JSON保留所有最优CV候选组合；端点需要查询标签，不是部署规则。','', '|组|并列episode|完整组合数|CV准确率范围|CV−W范围|CV−R范围|C范围|','|:---|:---|---:|:---|:---|:---|:---|']
for key in [f'{d}:{m}:B{b}' for d in ['EuroSAT','CIFAR-100'] for m in ['dinov2_small','deit_tiny'] for b in [8,16]]:
 s=r['summaries'][key]
 def fmt(k):return '–'.join(f'{x:.4f}' for x in s['tie_bounds'][k])
 t.append('|'+ '|'.join([key,str(s['cv_tie_episodes']),str(s['tie_assignment_count']),fmt('mean_accuracy'),fmt('minus_weak_pp'),fmt('minus_ra_pp'),fmt('regret_pp')])+'|')
t+=['','每种完整组合的12项选择、准确率、两基线差值与C都保留在COMBINED_RESULTS.json的tie_assignments；未更改任何选择。']
(P/'ALL_TIE_SENSITIVITY.md').write_text('\n'.join(t)+'\n')
e=['# 证据索引与结束状态','','本记录汇合5个只读来源档案，验证其367个清单文件的SHA256/字节，96个唯一设置与原结果逐项一致。原档案未覆盖或改写。所有原精度特征、分类头和逐例预测仍在原目录；这里不重复复制大资产。','',
'|本目录文件|内容|','|:---|:---|',
'|[CONCLUSION_ZH.md](CONCLUSION_ZH.md)|可向用户交代的有限结论、8组核心表、反证及已有工作边界|',
'|[ALL_96_SETTINGS.csv](ALL_96_SETTINGS.csv) / [MD](ALL_96_SETTINGS.md)|96个设置的全部策略准确率、选择、配对差、regret；CSV含每折整数计数|',
'|[ALL_8_GROUPS.csv](ALL_8_GROUPS.csv)|8组完整固定策略均值、事后最佳、选择频率、H/C、并列界和阶段统计|',
'|[COMBINED_RESULTS.json](COMBINED_RESULTS.json)|全部原始整数统计、各组12个配对分布、每一种并列组合及预算描述|',
'|[ALL_TIE_SENSITIVITY.md](ALL_TIE_SENSITIVITY.md)|并列处理的完整上下界，原规则仍为主结果|',
'|[SOURCE_INDEX.json](SOURCE_INDEX.json) / [VALIDATION.json](VALIDATION.json)|来源哈希、367文件核验、原始数值一致、整数恒等式检查|',
'|[LIFECYCLE.json](LIFECYCLE.json)|各任务退出回执及当前无运行/待启动实验的状态|',
'|aggregate_saved.py / write_final_record.py|仅标准库的复算和文档生成实现；不导入模型库|',
'',
'原始证据：','',
'|档案|关键入口与用途|完整清单SHA256|','|:---|:---|:---|']
purpose={
'pilot':'EuroSAT DINO：PROTOCOL、SELECTION_SEAL、HEADS、QUERY_PREDICTIONS、POSTRUN_VERIFICATION、RUN/TASK_RECEIPT',
'deit':'EuroSAT DeiT：同类完整文件；SECOND_ENCODER_CONTRACT_TEST验证共享视图及必要预处理差异',
'synthesis':'EuroSAT保存预测综合：全部并列组合、错误设置、576折计数、三候选混淆修正、5项合成测试',
'preflight':'CIFAR100资格/协议/身份/计时、VEST阅读边界；历史“尚未执行”状态由后续完成记录接续',
'cifar':'CIFAR100全部结果：JOINT_QUERY_GATE、两FIT/QUERY_SEAL、720头、全部预测、阶段排序、FINAL_VERIFICATION'}
for tag,v in idx['verified_archives'].items():
 d=v['directory'];e.append(f"|[{d}](../{d}/REPORT_ZH.md)|{purpose[tag]}|`{v['manifest_sha256']}`|")
e+=['',
'阶段和图表可直接追溯到 [EuroSAT全部错误](../augmentation-selection-synthesis-20261102/ALL_WRONG_CHOICES.md)、[EuroSAT完整并列](../augmentation-selection-synthesis-20261102/ALL_TIE_ASSIGNMENTS.md)、[EuroSAT逐例图](../augmentation-selection-synthesis-20261102/episode_differences.png)、[CIFAR全部阶段排序](../augmentation-selection-cifar100-run-20261102/STAGE_RANKINGS.md)、[CIFAR逐例图](../augmentation-selection-cifar100-run-20261102/paired_differences.png)。图表不省略不利设置。','',
'已结束任务及生命周期：','',
'|任务|原任务记录状态|退出依据|当前状态|','|:---|:---|:---|:---|']
for x in life['tasks']:
 e.append(f"|{x['task']}|{x['recorded_status']}|[{x['exit_evidence_field']}=true](../{x['receipt']})|已结束，无待启动步骤|")
e+=['',
'EuroSAT两次主执行session均有exit_code=0；CIFAR监督进程等待全部4个模型工作进程与分析进程退出，正式执行exit_code=0。预检和只读综合同样有已退出回执。本结论限定于这条研究线的已启动任务，不是整机进程扫描；没有调用此前被拒的ps/sysctl。','',
'旧CROSS_DATASET_PLAN_NOT_EXECUTED.md是当时历史计划，已由随后冻结并完成的全类CIFAR100方案接续；不再是待运行任务。预检档案内的“待集成”字段也只是历史状态。LIFECYCLE.json明确当前running_research_jobs=0、pending_experiment_commands=[]、fourth_round_started=false；没有创建新的调度任务。','',
'成本事实：本线累计1440个真实ridge头、96次选择、195194个正式图视图前向，另有320个预检/计时前向；未更新编码器。保存7912800个查询决策和190080个OOF决策。每个原任务的耗时及资源以其回执为准；CIFAR完整75.56分钟已经包含准备，不再把其预检21分钟重复累加。本线新增CU=0。此次最终整理新增模型、特征、拟合、下载、安装、上传和CU均为0，产物大小/时间记入本目录TASK_RECEIPT。','',
'当前研究线已关闭，无剩余实验计划启动。没有把新结果换进已有RandAugment演讲。结论可供2026-11-02韩国19:00汇报使用；这不是提醒或自动发布任务。']
(P/'EVIDENCE_INDEX.md').write_text('\n'.join(e)+'\n')
(P/'HANDOFF_ZH.md').write_text('''本轮增强选择研究已结束。两数据集、两编码器、两预算共96个设置完整完成，所有固定策略、并列情况和不利结果均可复算。我们得到的有限观察是：CV在这批设置上没有超过各组事后最佳固定均值，但能胜过部分预先固定策略；最佳固定策略随模型和数据变化。H/C关系是代数恒等式，不是新机制；部分阶段和预算描述也未复现。

这些结果目前不足以称为可靠论文贡献。已有少样本验证、标签成本和冻结特征增强工作覆盖了主要先验；还缺未见任务上的前瞻确认、对齐标签与验证量的强对照，以及超出已有认识的可迁移结论。本轮不再加实验，所有本线计算任务均有退出回执，无后台训练或待启动第四轮；新增云额度为0。已有RandAugment演讲保持原样。

证据入口：CONCLUSION_ZH.md、ALL_96_SETTINGS.csv、ALL_8_GROUPS.csv、COMBINED_RESULTS.json、EVIDENCE_INDEX.md、LIFECYCLE.json。
''')
print('wrote final conclusion, evidence index, complete tie table and handoff')
