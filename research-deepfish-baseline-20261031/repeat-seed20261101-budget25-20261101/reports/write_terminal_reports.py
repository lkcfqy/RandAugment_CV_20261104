"""Summarize frozen saved results; standard-library CPU only, no model calls."""
from portable_dev.paths import project_path as _portable_project_path
import csv,datetime,hashlib,json,pathlib
W=pathlib.Path(__file__).resolve().parents[1]
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
D=W/'reports'; E=W/'error-analysis/reports'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
old=read(R/'error-analysis-20261031/reports/OVERALL.json');new=read(E/'OVERALL.json')
summary=read(E/'SUMMARY.json');controller=read(W/'gpu-run/controller_report.json');transport=read(D/'FINAL_TRANSPORT_VERIFICATION.json');pair=read(W/'gpu-run/LOCAL_PAIR_VERIFIED.json');release=read(D/'FINAL_RELEASE_SNAPSHOT.json');verify=read(E/'VERIFICATION.json')
assert controller['status']=='COMPLETE' and controller['release']=='CONFIRMED'
assert release['active_assignments']==0 and release['assignments']==[] and release['rate_per_hour']==0
assert verify['status']=='PASS' and verify['check_count']==69
assert transport['all_files_rehashed'] and transport['all_frozen_runtime_pins_unchanged']
metrics=list(csv.DictReader((D/'TWO_SEED_METRICS.csv').open()));groups=list(csv.DictReader((D/'TWO_SEED_GROUP_SCALE_COMPARISON.csv').open()))
formal={a:read(W/'gpu-run/artifacts'/f'formal_{a}.json') for a in 'AB'}
assert all(m['actual_updates']==200 and m['optimizer_initial_state_empty'] and not m['preflight_state_reused'] and set(m['exposures']['counts'].values())=={40} for m in formal.values())
assert not pair['dev_evaluated_before_this_receipt']
assert controller['cumulative_CU_upper']<=2.5 and controller['cumulative_GPU_seconds_upper']<=7200
stages=[];previous=0
for line in (D/'controller.stdout.log').read_text().splitlines():
 try:q=json.loads(line)
 except json.JSONDecodeError:continue
 if q.get('stage')=='LOCAL_STAGE_PERSISTED':
  elapsed=q['elapsed_GPU_seconds'];stages.append({'phase':q['phase'],'elapsed_seconds':elapsed,'increment_seconds':elapsed-previous,'includes_local_hash_and_transfer':True});previous=elapsed
stages.append({'phase':'final_ack_to_release_observed','elapsed_seconds':controller['allocation_wall_seconds_upper'],'increment_seconds':controller['allocation_wall_seconds_upper']-previous,'includes_release_reconciliation':True})
cost={
 'utc':now,'status':'ACTUAL_COMPLETE_RELEASED','prior_CU_upper':controller['prior_cu'],'prior_GPU_seconds_upper':controller['prior_gpu_seconds'],
 'attempt_CU_upper':controller['cumulative_CU_upper']-controller['prior_cu'],'attempt_GPU_seconds_upper':controller['allocation_wall_seconds_upper'],
 'cumulative_CU_upper':controller['cumulative_CU_upper'],'cumulative_GPU_seconds_upper':controller['cumulative_GPU_seconds_upper'],
 'cap_CU':2.5,'cap_GPU_seconds':7200,'live_T4_CU_per_hour':controller['live_rate'],'stages':stages,
 'early_gate':read(W/'gpu-run/COMPLETE_COST_GATE.json'),'post_preflight_gate':read(W/'gpu-run/U_FREEZE.json'),
 'final_release_snapshot':release,'rate_clock_is_conservative_bound_not_invoice':True,
 'scope':'All preparation, preflight, formal training, evaluation, full return, release; prior failed attempts carry forward.'}
dump(D/'COST_ACTUAL.json',cost)
primary=[x for x in metrics if x['metric']=='primary_AP']
mean={k:sum(float(v[k]) for v in primary)/2 for k in ['A_points','B_points','delta_points']}
result={
 'status':'COMPLETE_RELEASED_CPU_VERIFIED','utc':now,
 'scientific_status':'post-result exploratory paired seed repeat; no demonstrated augmentation benefit or reliable new contribution',
 'completed_seeds':[20261031,20261101],'additional_seed_count':1,'third_seed_run':False,
 'previous_failed_repeat':{'seed':20261101,'preflight_updates_per_arm':15,'planned_formal_updates_per_arm':200,'actual_formal_updates_per_arm':0,'dev_evaluations':0,'counts_as_scientific_seed':False},
 'current_repeat':{'seed':20261101,'support_images':5,'support_boxes':5,'dev_images':64,'dev_boxes':187,'dev_prefix_groups':8,'planned_formal_updates_per_arm':200,'actual_formal_updates_per_arm':200,'formal_exposures_per_source_per_arm':40,'preflight_updates_per_arm':15,'preflight_state_reused':False,'Color_factors':{'A':1.0,'B':0.8},'pair_verified_before_dev':True,'fixed_final_checkpoint':True,'official_test_used_for_training_selection_or_current_model_evaluation':False},
 'primary_metric':'COCO bbox AP IoU0.50:0.05:0.95 maxDets100',
 'reported_metric_caveat':'Original AP50/AP75 and scale AP are maxDets1000; saved predictions300/image. Uniform100/300 recomputation is labeled post-hoc diagnosis, not replacement.',
 'two_seed_full_metrics':{'20261031':old,'20261101':new},'descriptive_two_seed_primary_mean_points':mean,
 'conclusions':[
  'B-A primary AP is -1.6045839418438401 and -1.6873008558065894 points for the two fixed seeds.',
  'The first seed AP50-up/AP75-down pattern did not replicate: the second seed has AP50 -2.6807108092898546 points and AP75 +0.050308793773118365 points at the original caps.',
  'Second-seed geometry coverage at IoU0.75 loses19 GT and gains2; AP75 itself barely changes, so matching counts are not an AP mechanism decomposition.',
  'Four of eight prefix-group AP delta signs flip; two seeds on one correlated dev set do not establish cross-episode or cross-scene reliability.',
  'No new training, tuning, threshold selection, relabeling, test evaluation, or third seed is implied.'
 ],
 'assets':{'package_bytes':(W/'deepfish_5shot_two_arm.zip').stat().st_size,'package_sha256':sha(W/'deepfish_5shot_two_arm.zip'),'uploaded_total_bytes_including_bootstrap':controller['upload_bytes'],'final_transport_files':transport['file_count'],'final_transport_bytes':transport['total_bytes'],'final_index_sha256':transport['index_sha256'],'checkpoints':pair['checkpoints'],'initial_state_sha256':pair['initial_state_sha256'],'paired_trace_sha256':pair['paired_trace_sha256']},
 'checks':{'pre_run_non_model_tests':58,'independent_review':'PASS','independent_non_model_tests':24,'final_CPU_analysis_checks':69,'all_frozen_runtime_pins_unchanged':True,'all_70_returned_artifacts_rehashed':True},
 'cost':cost,'remaining_execution_blockers':[],
 'limits':['Same official5-shot episode, same64dev/187GT; only training randomness was repeated.','Dev added187 labels and is prefix-isolated from support, not proven scene/video isolated.','Original dataset detection-label derivation and prefix semantics remain bounded documented uncertainties.','Historic QA viewed3 test images; test was not used for training selection or this model evaluation.','N=2, correlated frames, one fixed factor/short training length; no significance or novelty claim.'],
 'input_sha256':{str(p.relative_to(W)) if p.is_relative_to(W) else str(p):sha(p) for p in [D/'FINAL_TRANSPORT_VERIFICATION.json',D/'FINAL_RELEASE_SNAPSHOT.json',E/'OVERALL.json',E/'VERIFICATION.json',R/'error-analysis-20261031/reports/OVERALL.json']}}
dump(D/'RESULT.json',result)
newtable=[]
for key,title in [('primary_AP','主 AP@100'),('original_AP50','原 AP50@1000'),('original_AP75','原 AP75@1000'),('original_AR100','AR@100'),('original_AR300','AR@300')]:
 x=next(v for v in metrics if v['seed']=='20261101' and v['metric']==key);newtable.append(f"| {title} | {float(x['A_points']):.4f} | {float(x['B_points']):.4f} | {float(x['delta_points']):+.4f} |")
grouptable=[]
for x in groups:
 if x['group']=='ALL':continue
 grouptable.append(f"| {x['group']} | {x['GT']} | {float(x['20261031_delta_AP_100'])*100:+.2f} | {float(x['20261101_A_AP_100'])*100:.2f} | {float(x['20261101_B_AP_100'])*100:.2f} | {float(x['20261101_delta_AP_100'])*100:+.2f} |")
scaletable=[]
for x in groups:
 if x['group']=='ALL' and x['area']!='all':
  scaletable.append(f"| {x['area']} | {x['GT']} | {float(x['20261031_delta_AP_100'])*100:+.4f} | {float(x['20261101_A_AP_100'])*100:.4f} | {float(x['20261101_B_AP_100'])*100:.4f} | {float(x['20261101_delta_AP_100'])*100:+.4f} |")
report='''固定颜色增强的第二种子复核已完成，两臂均为 U=200，全部权重、预测和日志已回传，GPU 已释放。主 AP 在两个种子中均降低；目前没有得到可靠的新贡献。

这是看过首轮结果后决定的一次探索性重复检查。唯一新增种子为 20261101；此前同种子的预算门停止尝试只有预检、正式更新 0、dev 结果 0，不计作一个科学结果。本次计划和实际均为每臂 200 次正式更新，每张 support 曝光 40 次，预检每臂 15 次另计并丢弃状态。

| 种子 | A：Color 1.0 主 AP | B：Color 0.8 主 AP | B−A（百分点） |
|---|---:|---:|---:|
| 20261031 | 26.7624 | 25.1578 | −1.6046 |
| 20261101 | 29.9651 | 28.2778 | −1.6873 |

两种子的描述性均值 A=28.3637、B=26.7178、差 −1.6459；N=2 不支持显著性或普遍可靠性判断。两轮使用同一官方 5-shot episode 和固定 64 张 dev（187 框、8 前缀组），只检验训练随机性。

新种子的完整整体指标如下。主 AP 使用 maxDets=100；原 AP50/AP75 使用 maxDets=1000，实际每图仅保存 300 框。AR100 与 AR300 单列。统一 maxDets=100/300 的后验辅助复算、原 12 项指标和全量 PR 均保留，未用辅助口径替换主指标。

| 指标 | A | B | B−A（百分点） |
|---|---:|---:|---:|
'''+ '\n'.join(newtable)+'''

首轮 AP50 +2.3968、AP75 −3.4749；本轮 AP50 −2.6807、AP75 +0.0503。前一次“宽松匹配改善、严格匹配变差”的指标形态没有重复，不能解释为已证明的颜色机制。

新种子按固定规则分析全部 38,400 个预测与 187 个 GT：IoU 0.75 的几何合格候选丢失 19 个目标、增加 2 个，top300 COCO 匹配数从 103 降到 86；同时 AP75 几乎不变，说明排序与几何覆盖需要一起看，不能用单个阈值计数代替 AP。8 组中本轮 3 组上升、5 组下降，有 4 组相对首轮反转。六张改善/退化/不变图由事先固定规则选择，详见下方错误分析。

执行沿用已成功的单 T4、Python 3.10.19、Torch 2.1.0+cu118、MMCV 2.1/MMDetection 3.3 配置。A/B Color(prob=1,level=0) 的因子分别为 1.0/0.8；cv2/uint8/BGR、标准几何、优化器参数与更新数不变。实测 A 逐像素恒等，两臂随机数消费相同；source 顺序、几何、GT、RNG 轨迹一致。两臂从原权重和空优化器分别初始化，完整 checkpoint 在 Mac 以 CPU weights_only=True 检查和配对通过后才评估 dev。未用 loss/dev/test 选 U、阈值或 checkpoint。

本轮增量为 0.5510888406 CU、1725.1477 GPU 秒（28.75 分钟）；历史失败与准备消耗完整结转，最终累计 1.8088330816 CU、5443.8998 GPU 秒（90.73 分钟），低于 2.5 CU/120 分钟上限。分配后第 16.75 秒、上传与依赖准备前的带宽/费率门已通过；8 MiB 回取实测 4.45 MB/s，完整剩余估计 2611.66 秒，对应可用 3464.50 秒。预检后 U200 门再次通过。全部 70 个远端结果文件共 1,400,239,201 字节完成原子落盘及 SHA256 核验；本机 CPU 分析 69 项检查通过。最终独立快照显示 active_assignments=0、assignments=[]、rate_per_hour=0；回收后未再调用模型或分配 GPU。

证据只适用于这个短程、固定颜色因子的同一 support/dev 探索。dev 使用额外 187 个标签；前缀组隔离不能称新场景，前缀语义和检测框派生来源的既有不确定性保留。历史 QA 查看过 3 张 test 图；官方 test 未用于训练选择或本次模型评估。没有第三个种子、改标签或继续调参。

当前可测假设是：固定 Color 0.8 对不同目标的排序和几何覆盖影响不一致，简单的整体 AP50/AP75 解释不能稳定跨训练随机性成立。现有证据不足以支持将它发展为新方法或论文贡献；当前授权的唯一重复检查已经完成，无执行阻塞，也不自动开启后续实验。

- [两轮机器结果与完整口径](reports/RESULT.json)
- [两轮指标 CSV](reports/TWO_SEED_METRICS.csv)
- [全量错误分析与六张选图](error-analysis/ERROR_ANALYSIS_ZH.md)
- [成本分解与实时门记录](reports/COST_ACTUAL.json)
- [70 文件回传核验](reports/FINAL_TRANSPORT_VERIFICATION.json)、[释放快照](reports/FINAL_RELEASE_SNAPSHOT.json)、[CPU 69 项核验](error-analysis/reports/VERIFICATION.json)
- [最终归档清单](FINAL_ARCHIVE_SHA256.json)（不含清单自身及其回执；历史 PRE_RUN 清单保留为运行前快照）
'''
(W/'RESULT_ZH.md').write_text(report)
err='''第二个种子 20261101 的错误分析已完成。仅使用已保存预测、固定 dev64 原图与原标注，CPU 复算；不新训练、不改阈值、不改标签。完整沿用首轮的匹配、尺度、曲线与确定选图规则。主 AP 29.9651→28.2778（−1.6873 个百分点）；整体数值及原 maxDets 混合口径见 [终态报告](../RESULT_ZH.md)。

全量前缀组结果如下，均为 AP@100、百分点。每组固定 8 图；组 AP 不是整体 AP 的可加贡献，也不是独立视频/场景效果。4 组方向随种子反转。

| 前缀组 | GT | 首轮差值 | 本轮 A | 本轮 B | 本轮差值 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(grouptable)+'''

COCO 原尺度区间和原 area 均保留；small 仅 8 个目标，结果尤其不稳。

| 尺度 | GT | 首轮差值 | 本轮 A | 本轮 B | 本轮差值 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(scaletable)+'''

没有置信度阈值搜索。相同类别预测按框间 IoU≥0.5、再按固定索引规则一一配对，得到 7,256 对；每臂 11,944 个未配对保留。它们只是几何邻近候选，不代表相同 query/鱼身份。

| 固定预算/IoU | A 匹配 GT | B 匹配 GT | A 未匹配 GT | B 未匹配 GT |
|---|---:|---:|---:|---:|
| top100 / .50 | 151 | 148 | 36 | 39 |
| top100 / .75 | 94 | 85 | 93 | 102 |
| top300 / .50 | 158 | 152 | 29 | 35 |
| top300 / .75 | 103 | 86 | 84 | 101 |

全部 300 候选中的逐 GT 最大 IoU 差：中位数 −0.02595、均值 −0.03806。IoU .50 的合格几何候选丢失 10、增加 5；.75 丢失 19、增加 2。B 有 1 个 GT 在 .50 有候选却因 COCO 一一竞争未匹配。对两臂都有 .50 候选的 148 个 GT，首个候选图内排序 35 个变差、28 个改善、85 个相同；.75 的 84 个共有目标相应为 17/22/45。几何上界和匹配计数不是 AP 的因果分解。

相对现有标注，top300/.50 的未匹配预测从 19,042 到 19,048，其中重复候选 1770→982、定位近失 2789→3143、低标注重叠候选 14483→14923。首轮重复候选反而增加，因此不能归纳为稳定的“颜色导致重复框”机制。这些类别不是可靠的真实背景/漏标判定，每图固定 300 框也会产生很大的低分候选计数。保留 [全部 IoU 档](reports/ALL_IOU_BINS.csv)、[101 点 PR](reports/FULL_INTERPOLATED_PR_CURVES.csv) 和 [原始排序 TP/FP 曲线](reports/FULL_RAW_PR_STEPS.npz)，不只依据 .50/.75。

六张图严格按逐图 AP@100 差值选取最大改善 2 张、最大退化 2 张、其余绝对变化最小 2 张；并列 image_id 排序。每图同样展示 top10，加每个 GT 的最大 IoU 候选，仅用于阅读，指标使用完整候选。已查看六张双面板图，观察如下：

- [541，+26.25](overlays/image_541_improvement.png)：左下边缘的大斑点鱼，最佳框 IoU 0.982→0.959，并未改善；达到 .75 的首框从第 8 位升到第 2 位，分数 .0089→.1267。这张图的提升与较好框被排到前面相符。
- [564，+24.67](overlays/image_564_improvement.png)：同一前缀中的大鱼，A 首框 IoU 约 .740、B 约 .757；第一个 .75 合格候选从第 60 位到第 1 位，B 第 2 位还有 IoU 约 .921 的框。两个改善图来自同一组，不能当作独立现象。
- [1479，−18.99](overlays/image_1479_regression.png)：根系前的两条已标鱼，两臂均排第 1/2；主鱼首框 IoU 约 .946→.845，另一条 .701→.628，说明排名相近仍可能在较严格 IoU 档损失。叶片状物体两臂都有高分框。右上小浅色物体疑似另一鱼但无 GT，保留疑点，不修标注或扣除误检。
- [1236，−13.53](overlays/image_1236_regression.png)：水中枝干旁的小鱼，GT4196 最佳 IoU .781→.726，失去 .75 合格候选；GT4197 .684→.547，虽然图内排序 73→31 仍定位变差。右边缘 GT4195 首个 .75 候选从第 1 到第 5；本例并非所有对象一致退化。
- [735，0.00](overlays/image_735_unchanged.png)：左侧黑鱼首框均第 1，IoU .868→.866，分数 .169→.779；逐图 AP 均 80，分数幅值变化不等于图内 AP 变化。
- [1481，0.00](overlays/image_1481_unchanged.png)：主鱼均第 1，另一遮挡小目标 GT5089 两臂都没有 IoU .5 候选；逐图 AP 均 45.45。下方叶片候选分数 .719→.819，仍不可据此称为新增真实鱼目标。

GT4195/4660 原 xywh、area 和右侧 0.5 像素越界全部原样保留；评估保持官方先 resize 后加载标签路径。1479 的疑似未标物体、1481 小目标可见性等只作原标注疑点，不改变任何数字。除这六张当轮选图外，继承 renderer 生成的旧种子放大图明确改名为 historical_seed20261031_detail_crops.png，旧 rank 标签不用于本轮解释；[更正记录](reports/IMPLEMENTATION_NOTES.json) 保留原输出元数据。

以上更支持“排序和几何覆盖的影响异质、解释对种子敏感”这个后续可测假设，未证明原因或创新。只有同一 episode 的两个训练种子、相关的 8 组 dev、固定一个因子与 U200；不得据此宣称跨场景泛化或可靠提升。

[完整 8组×尺度表](reports/ALL_SUBGROUP_DELTAS.csv)、[187 GT 诊断](reports/GT_DIAGNOSTICS.csv)、[38,400 预测诊断](reports/ALL_PREDICTION_DIAGNOSTICS.csv)、[所有跨臂配对](reports/CROSS_ARM_PAIRS.csv)、[两轮分组对照](../reports/TWO_SEED_GROUP_SCALE_COMPARISON.csv)、[69 项核验](reports/VERIFICATION.json)。
'''
(W/'error-analysis/ERROR_ANALYSIS_ZH.md').write_text(err)
(W/'RUN_HANDOFF.md').write_text('REAL TERMINAL COMPLETE — DO NOT RELAUNCH\n\nOnly authorized new seed20261101 completed both formal arms200 and both64dev evaluations. Preflight states discarded; full pair safely persisted before dev. CPU analysis69checks PASS. All70 remote artifacts rehashed. GPU released with independent active_assignments0/assignments[]/rate0 snapshot. Cumulative1.8088330815795528CU/5443.899788473129GPU seconds. Both-seed primary AP deltas -1.6045839418438401 and -1.6873008558065894points; no demonstrated contribution. No thirdseed, threshold tuning, new labels, model call or GPU allocation is authorized by this completion. Read RESULT_ZH.md and reports/RESULT.json. Final archive manifest and external receipt record project persistence.\n')
with (W/'progress.md').open('a') as f:f.write('\n'+now+' CPU terminal: analyze/render/verify exit0;69checks PASS; all6 frozen overlays inspected. Same-seed paired result complete, primaryAP A29.965066269154512 B28.27776541334792;B-A -1.6873008558065894points. Historical firstpair retained. No further GPU/model calls. Reports prepared; ancillary inherited crop metadata explicitly corrected and cache warning recorded without target retry. Final incremental archive next.\n')
print(json.dumps({'written':['RESULT_ZH.md','error-analysis/ERROR_ANALYSIS_ZH.md','reports/RESULT.json','reports/COST_ACTUAL.json'],'status':'COMPLETE_RELEASED_CPU_VERIFIED','no_GPU_or_model_calls':True}))
