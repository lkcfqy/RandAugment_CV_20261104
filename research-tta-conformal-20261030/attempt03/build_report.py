import hashlib,json,os,datetime
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(R/'matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A=R/'analysis';D=R/'results'
point=json.loads((A/'all-point-estimates.json').read_text())
ci=json.loads((A/'simultaneous-intervals.json').read_text())
intervals=ci['intervals'];decision=json.loads((A/'decision.json').read_text())
tuning=json.loads((A/'tuning-only-selections.json').read_text())
split=json.loads((D/'split-summary.json').read_text())
state=json.loads((D/'run-state.json').read_text())
release=json.loads((R/'released.json').read_text())
perf=json.loads((R/'performance-gate.json').read_text())
env=json.loads((D/'environment.json').read_text())['environment']
mathcheck=json.loads((R.parent/'math-acceptance.json').read_text())
TS=(.5,.75,1,1.5,2,3,5)
colors={'00':'#245ea8','10':'#7e4a99','01':'#bf7b19','11':'#bd3f3f'}
arm_labels={'00':'Original rank + original spectrum','10':'TTA rank + original spectrum',
            '01':'Original rank + TTA spectrum','11':'TTA rank + TTA spectrum'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.facecolor':'white','axes.titleweight':'bold'})
fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
for arm in ('00','10','01','11'):
    vals=[point[f'aps|{t:g}|{arm}|0.1']['M'] for t in TS]
    axes[0].plot(TS,vals,'o-',color=colors[arm],label=f'h{arm}')
axes[0].set_xscale('log');axes[0].set_xticks(TS,labels=[str(t) for t in TS]);axes[0].set_xlabel('Temperature (fixed grid)')
axes[0].set_ylabel('Mean set size (labels)');axes[0].set_title('APS: all four arms');axes[0].legend(ncol=2,frameon=False)
for j,(name,color) in enumerate((('Rp','#245ea8'),('Rq','#bd3f3f'),('J','#5a6570'))):
    items=[intervals[f'aps|{t:g}:{name}'] for t in TS]
    y=np.array([i['estimate'] for i in items]);err=np.array([y[k]-i['simultaneous95'][0] for k,i in enumerate(items)])
    x=np.arange(len(TS))+(j-1)*.18
    axes[1].errorbar(x,y,yerr=err,fmt='o',capsize=2,color=color,label=name,markersize=4)
axes[1].axhline(0,color='#777',lw=.8);axes[1].axhspan(-.25,.25,color='#e8e8e8',alpha=.8)
axes[1].set_xticks(np.arange(len(TS)),labels=[str(t) for t in TS]);axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Set-size difference (labels)');axes[1].set_title('Paired effects: simultaneous 95% CI');axes[1].legend(frameon=False)
for arm in ('00','11'):
    rs=[point[f'aps|{t:g}|{arm}|0.1'] for t in TS]
    axes[2].plot([r['Dhat'] for r in rs],[r['M'] for r in rs],'o-',color=colors[arm],label=f'h{arm}')
    for t,r in zip(TS,rs):axes[2].annotate(f'{t:g}',(r['Dhat'],r['M']),xytext=(3,4),textcoords='offset points',fontsize=8)
axes[2].set_xlabel('Class-coverage Dhat (probability squared)');axes[2].set_ylabel('Mean set size (labels)')
axes[2].set_title('Finite temperature grid (all points)');axes[2].legend(frameon=False)
fig.suptitle('CIFAR-100 internal heldout | one seed-0 ResNet20 | alpha=0.10',fontsize=14)
fig.savefig(A/'mechanism-diagnostic.png',dpi=180)
fig.savefig(A/'mechanism-diagnostic.svg')
plt.close(fig)

def fmt(x,n=4):return '不可估' if x is None else f'{x:.{n}f}'
def cint(name,level=95):
    r=intervals[name]
    if not r['estimable']:return '不可估'
    lo,hi=r[f'simultaneous{level}'];return f"{r['estimate']:.3f} [{lo:.3f}, {hi:.3f}]"
def table(rows,head):
    return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
rows=[]
for arm in ('00','10','01','11'):
    r=point[f'aps|1|{arm}|0.1']
    rows.append([f'h{arm}',fmt(r['M']),f"{r['coverage']*100:.2f}%",fmt(r['Dhat'],6),fmt(r['threshold'],6)])
main=table(rows,['臂','平均集合大小 M','覆盖率','Dhat','校准阈值'])
effects=table([[f'{t:g}',cint(f'aps|{t:g}:Rp'),cint(f'aps|{t:g}:Rq'),cint(f'aps|{t:g}:J')] for t in TS],
              ['T','Rp及95%同时CI','Rq及95%同时CI','J及95%同时CI'])
controlrows=[]
for m in ('raps','saps','rankonly'):
    t=tuning['temperature_selection'][m]['sources']['q']['T']
    controlrows.append([m,str(tuning['params'][m]),f'{t:g}',cint(f'{m}|{t:g}:Rp'),cint(f'{m}|{t:g}:Rq'),cint(f'{m}|{t:g}:J')])
controls=table(controlrows,['评分','共享参数','tuning选q温度','Rp及95%同时CI','Rq及95%同时CI','J及95%同时CI'])
counts=point['aps|1|00|0.1']['class_n']
bootstrap=json.loads((A/'bootstrap-design.json').read_text())
baseline=.0
raw=np.load(D/'logits-fp32.npy',mmap_mode='r')
y=np.load(D/'heldout-labels.npy')
acc_p=float(np.mean(raw[10000:,4,:].argmax(1)==y[10000:]))
q=np.load(D/'mean-logits-f64.npy',mmap_mode='r')
acc_q=float(np.mean(q[10000:].argmax(1)==y[10000:]))
qts=decision['B_Tq_star'];pts=decision['B_Tp_star']
equiv=[]
for t in TS:
    for name in ('Rp','Rq','J'):
        rr=intervals[f'aps|{t:g}:{name}']
        if rr['estimable'] and rr['simultaneous90'][0]>=-.25 and rr['simultaneous90'][1]<=.25:equiv.append(f'{name}@T={t:g}')
title='发现预定审查信号，需主线审查，不能直接声称新贡献' if decision['candidate_status']=='REVIEW_SIGNAL' else '关闭本候选：未达到预定新机制继续门'
reasonA=[]
if not decision['A_opposite_rank_effects']:reasonA.append('Rp/Rq未达到预定异号且两侧均超过0.25标签的同时区间要求')
if not decision['A_coverage_pass']:reasonA.append('至少一臂覆盖率同时下界未达到89%')
if decision['A_numeric_saturation']:reasonA.append('存在严格数值饱和标记')
if any(x['concentrated'] for x in decision['A_concentration'].values()):reasonA.append('贡献集中于top1%异常样本')
reasonB=[]
if not decision['B_tuning_selection_feasible']:reasonB.append('至少一个预定参数或温度的tuning覆盖/类覆盖约束不可行')
for field,label in [('gain_pass','至少一个原输出温度的缩集优势不超过同时下界0.25'),('D_pass','类覆盖Dhat可比性未全部证实'),('coverage_pass','边际覆盖及1pp容差未全部证实')]:
    if not all(c[field] for c in decision['B_all_7_comparators']):reasonB.append(label)
if not decision['B_restored_gain_pass']:reasonB.append('恢复原排序后的实质优势消失未证实')
if not decision['B_attenuation_pass']:reasonB.append('恢复原排序造成至少0.25标签衰减未证实')
if not all(c['mechanism_beyond_control_established'] for c in decision['negative_controls']):reasonB.append('未建立超出SAPS/rankonly已知机制的优势')
alloc=json.loads((R/'cli-ledger/0003-allocate.json').read_text())
gpu_seconds=release['attempt_GPU_seconds_upper']
cumulative_gpu=release['cumulative_GPU_seconds_upper']
wall=datetime.datetime.fromisoformat(release['utc']).timestamp()-1793382807.1513782
estimated_cu=cumulative_gpu/3600*1.15
storage=sum(p.stat().st_size for base in (R.parent,Path('/Users/lkc/Documents/Codex/2026-10-31/task/tta-conformal')) for p in base.rglob('*') if p.is_file())
usage_receipts=[json.loads(p.read_text()) for p in (R/'cli-ledger').glob('*released-usage.json')]
assert len(usage_receipts)==1 and release['verified_zero']
release['usage_output']=usage_receipts[0]['stdout'];release['sessions_output']=release['sessions_stdout']
qualification=json.loads((R.parent/'attempt02/commits/benchmark_restore.done.json').read_text())['audit']
assert state['current_allocation_audit']=={'benchmark_steps':0,'benchmark_presentations':0,'formal_steps':25040,'formal_presentations':3200000,'inference_views':200000}
assert qualification['benchmark_steps']==200 and qualification['formal_steps']==0
budget={'outcome':decision['candidate_status'],'formal_trajectory':'attempt03_formal_designated',
 'prior_GPU_seconds_upper':1652.7684237957,'current_GPU_seconds_upper':gpu_seconds,'cumulative_GPU_seconds_upper':cumulative_gpu,
 'actual_current_rate_CU_hour':1.15,'cumulative_rate_based_CU_upper_estimate':estimated_cu,
 'original_balance':41.94,'final_balance':release['balance'],'rounded_balance_difference':round(41.94-release['balance'],2),
 'startup_prediction_seconds':perf['cumulative_projection_seconds'],'startup_gate_seconds':6300,'running_hard_cap_seconds':7200,'CU_cap':6,
 'original_download_start_epoch':1793382807.1513782,'original_wall_deadline_epoch':1793404407.1513782,
 'first_attempt_formal_steps_min':23475,'first_attempt_formal_steps_charged_upper':25040,
 'current_formal_steps':25040,'cumulative_formal_steps_known_min':48515,'cumulative_formal_steps_charged_upper':50080,
 'cumulative_formal_presentations_known_min':6200000,'cumulative_formal_presentations_charged_upper':6400000,
 'acceptance_updates_all_allocations':400,'acceptance_presentations_all_allocations':51200,
 'all_training_updates_charged_upper':50480,'all_training_presentations_charged_upper':6451200,
 'inference_views_known_min':210240,'inference_views_charged_upper':410240,
 'actual_current_allocation_audit':state['current_allocation_audit'],'qualification_from_previous_allocation':qualification,
 'total_allocation_count':3,'formal_training_trajectories_including_lost_first':2,
 'released_session_count':0,'released_rate_CU_hour':0,'released_verified':True,'storage_staging_plus_project_bytes':storage,
 'new_OAuth':False,'new_credentials':False,'purchases':False,'private_history_uploaded':False,'Mac_GPU_used':False,'commit_or_push':False,
 'no_additional_experiment_authorized':True}
(R/'RESOURCE_LEDGER.json').write_text(json.dumps(budget,ensure_ascii=False,indent=2)+'\n')
text=f'''# TTA-conformal 机制诊断：{title}

本次完成了独立CIFARResNet20的80 epoch训练、20k图像十视图缓存、四臂全部7温度/4评分/2alpha的评估与999次配对分组重采样。结果仅针对固定seed0这一模型及内部heldout，不能推广成多个模型/训练seed结论。候选结束不表示整体研究停止；没有自动追加本候选实验。

## 主结果与停止决定

预定A门：**{'通过' if decision['A_pass'] else '未通过'}**。{'；'.join(reasonA) if reasonA else '达到预定信号标准，但仍需独立主线审查。'}。

预定B主门：**{'通过' if decision['B_primary_frontier_and_restore_pass'] else '未通过'}**；计入反证后的B门：**{'通过' if decision['B_pass_after_controls'] else '未通过'}**。{'；'.join(reasonB) if reasonB else '达到有限温度网格内预定标准。'}。

APS、T=1、alpha=.1：

{main}

Rp=M00-M10，Rq=M01-M11，J=Rq-Rp；正值表示替换rank后缩集，单位均为标签数。四臂都独立calibrate，共用每样本U。下面的区间控制整个预注册检验族，不是逐格普通95%区间。

{effects}

90%同时区间完整位于[-.25,.25]的局部等效项：{', '.join(equiv) if equiv else '无'}。其余不能仅因“不显著”写成等效或无作用。

![机制诊断图](analysis/mechanism-diagnostic.png)

图中保留所有温度点。灰带为±0.25标签；右图只是有限网格点，不代表连续温度族前沿。描述性top1准确率：原输出{acc_p*100:.2f}%，十视图均值{acc_q*100:.2f}%；这不是官方test benchmark。

## 调参、温度前沿与负控

RAPS/SAPS参数仅用2k pseudocal+2k validation在原输出T1选择，四臂/温度/alpha共享；完整候选含失败项见 [tuning-only-selections.json](analysis/tuning-only-selections.json)。正式cal只估阈值，eval没有选择参数/温度/子群。APS tuning选T_p*={pts:g}、T_q*={qts:g}；B仍比较全部7个原输出温度，不在eval丢掉不可比或不确定点。

{controls}

SAPS使用原论文Eq10；rankonly=rank-1+U。RAPS按预定诊断公式，正则项不随机，不能混称原论文所有随机化实现的精确复现。SAPS/rankonly的原输出tuning温度与q温度不同的点若不在预注册同时检验族中，其不确定性会阻止B升级，未事后扩大检验族或选更好点。负控排除同时要求参数/温度tuning可行、点估计可比、现有同时CI端点给出的覆盖/D可比性成立及size优势下界超过0.25；不可比或不确定均不得升级。缺失Dhat传播为不可估。没有扩充CI族。逐项门判据见 [decision.json](analysis/decision.json)；预先静态反例测试见 [decision-guard-tests.json](decision-guard-tests.json)，补丁与前后哈希见 [decision-guard-patch.json](decision-guard-patch.json)。

alpha=.05完整结果仅作敏感性，未用于救题。所有224个组合保留于 [CSV](analysis/all-point-estimates.csv) 和 [含逐类覆盖的JSON](analysis/all-point-estimates.json)。

## 数据、数值和推断验收

- 公开CIFAR100官方Python包MD5匹配 `eb9058c3a382ffc7106e4002c42a8d85`；SHA256 `{split['archive_sha256']}`。数据来源和引用见 [SOURCES.md](../SOURCES.md)，未标MIT许可。
- 60k图像发现{split['duplicate_group_count']}个精确重复组、{split['duplicate_images']}张图；{split['cross_origin_duplicate_groups']}组跨官方train/test。按训练前标签盲分组规则分配，cross-role duplicate=0；train40k、tuning4k、cal6k、eval10k均精确满足。图像ID、原来源、内容SHA、角色见 [split manifest](results/split-manifest.jsonl)。近重复未检测。
- CPU数学验收{len(mathcheck['checks'])}项全部通过，涵盖退化恒等、温度/置换、同谱集合大小、rankonly、精确次序统计量与尾部/空集政策，见 [math-acceptance.json](../math-acceptance.json)。
- FP32 logits按image/view/class保存，baseline为中心原图view4；GPU输出流SHA与读回SHA一致，逐样本均值验收通过。后续评分均float64，允许空集、不补top1，稳定classID并列与donor rank保留。细节见 [cache acceptance](results/cache-acceptance.json) 与 [数值诊断](analysis/numerical-diagnostics.json)。
- 999次cal/eval分别重采样重复组、同次跨臂配对、重新校准；cal有{bootstrap['cal_group_count']}组，eval有{bootstrap['eval_group_count']}组。预注册族共{ci['family_size']}个维度，95% max-|t|临界值{ci['critical95']:.4f}，90%为{ci['critical90']:.4f}。没有独立重复训练。置信区间是近似bootstrap，同时控制不是有限样本精确保证。
- eval逐类n范围{min(counts)}–{max(counts)}，不足以给每类1pp精度保证；逐类Wilson95%区间只是描述性而非100类同时保证。任何类n<2会标不可估，本次D不可估的variant-resample数为{ci['invalid_D_variant_resamples']}。Dhat负值保留。

## 可靠性修复、资源与失败账本

本轮是第三次分配、第二条正式训练轨迹，前一次恢复分配仅完成验收、正式训练为0。科学设置、评分公式、阈值、304维CI族和重采样方法均未改变。正式统计前主线静态审计修复了参数/温度不可行漏阻断、负控可比性漏检查及Dhat不可估时的异常。原90分钟预测门停止记录保留，之后主线/Pro批准105分钟启动门，120分钟硬限、6CU、原连续6小时和2GiB未变。分配前完整累计预测100.37分钟；实际准备48.61秒，通过8分钟准备限制，临正式开跑累计预测{perf['cumulative_projection_seconds']/60:.2f}分钟。105是启动门，未被用作运行中停止线。

实际GPU为{env['gpu']}，现有官方CLI0.7.5及已有授权；费率1.15CU/小时。本次分配至释放核验上界{gpu_seconds/60:.3f}分钟，三次累计上界{cumulative_gpu/60:.3f}分钟。按费率估计累计{estimated_cu:.3f}CU；余额41.94→{release['balance']:.2f}，显示余额差约{41.94-release['balance']:.2f}CU，不当作精确计费。

本次正式25040步、320万呈现、200000views；本次额外验收0步0views。前两次验收共400步/51200呈现（参考与50步重放双方均计账），已耗尽验收上限。第一轮最后确认23475正式步，未知尾段仍按完整25040步/320万呈现上计。累计正式上界50080步/640万呈现；含验收共50480步/6451200呈现；推理保守上界410240views。没有排除失败计算，没有用验收模型续训。

正式每5epoch保存完整模型、优化器momentum、手动cosine状态、Python/NumPy/Torch CPU/全部CUDA RNG、step/epoch/采样状态和归一化；epoch0及5–80共17个检查点全部下载、fsync、文件SHA及完整状态重新打开校验。下一块恢复RNG在模型/优化器构造和加载之后，原GPU randperm/增强流程保持不变。正式40个512图logit块（末块32图）均逐块验证image/role/source/view/class/dtype/最终checkpoint身份和哈希，再推进下一块。全部正式logits取回后立即释放GPU，后续组装均值缓存与统计在本机CPU。

推理第23块曾出现单个CLI调用不返回。只读状态/内核探针确认会话仍存活、epoch80与112640views、该块文件及完成标记均不存在后，在同一会话对原幂等函数执行一次有界恢复；完成标记确认117760views且匹配最终模型后，才结束卡住的旧本地CLI客户端。原控制器读取完成标记后继续剩余分块。没有新分配或重复模型计算；等待时间全部计入累计GPU预算。旧CLI的-15退出码是这一主动客户端清理，不是GPU中止。内部连接卡顿原因未确定，证据及恢复命令见 [INFERENCE_CONNECTION_RECOVERY.json](INFERENCE_CONNECTION_RECOVERY.json)。

释放后：`{release['sessions_output'].strip()}`，额度回执：

```
{release['usage_output'].strip()}
```

GPU释放后的完整本地状态和轨迹复核见 [durability-final-verification.json](durability-final-verification.json) 与 [trajectory-continuity.json](trajectory-continuity.json)。首次下载至本轮释放{wall/60:.2f}分钟；统计及最终归档仍计入原23:53:27UTC墙钟截止，最终时长/目录字节计数见 `archive-receipt.json`。累计账本见 [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json)。

全部失败分支保留：第一轮使用detached Popen，75epoch后远端分配消失，缺少中途持久化导致正式产物未取回；平台内部原因未知，不能把本地pruned当平台根因。第二次分配恢复验收通过，96.16分钟预测超过当时90分钟门，主动释放，未训练正式模型。[第二轮报告](../attempt02/REPORT_ZH.md)保留原预测细目及首次审批拒绝/随后直接授权证据解锁的记录。第三次分配经透明门修订后执行本次完整正式轨迹；旧报告和各次CLI账本均保留。

完整版本链：[原协议](../PROTOCOL_ZH.md) → [第一次恢复修订](../attempt02/AMENDMENT_APPROVED.json) → [105分钟门修订](AMENDMENT_APPROVED.json)；[完整成本表](PREALLOCATION_COST.json)与[8分钟准备限制](PREPARATION_LIMIT_ADDENDUM.json)均归档。没有新OAuth、持久凭据、购买、MacGPU、commit/push或上传历史私密项目。迟到第一轮产物不得按效果挑选：本次正式轨迹已按训练前规定指定采用。

## 解释边界

已有 [TTA-conformal论文](https://arxiv.org/html/2505.22764v1) 已讨论rank改善与概率平滑；[TS4CP/ConfTS](https://arxiv.org/abs/2402.05806) 和 [SAPS](https://arxiv.org/html/2310.06430v2) 已覆盖温度/rank机制。因此普通缩集、普通rank收益和四格分解不能独自作为新贡献。本次没有复用其未经核实许可的研究代码。

这是内部heldout重分割，不是官方test成绩；组原子分配使图像级抽样不再完全独立，cluster bootstrap用于反映已知重复依赖。Dhat的经典无偏修正在独立同分布类内观测下成立；在重复组存在时应视为常规修正估计而非宣称严格无偏。单模型、一个增强协议、有限温度族、保守同时CI及有限样本限制了功效。未通过预定门只关闭此候选，不证明所有TTA机制不存在。

完整协议：[PROTOCOL_ZH.md](../PROTOCOL_ZH.md)；训练前哈希：[protocol-lock.json](../protocol-lock.json)；所有产物的SHA256见 `SHA256SUMS.json`。若无新指示，本候选不追加seed、模型或视图。
'''
(R/'REPORT_ZH.md').write_text(text)
print(json.dumps({'report':str(R/'REPORT_ZH.md'),'candidate_status':decision['candidate_status'],'current_gpu_minutes':gpu_seconds/60,'cumulative_gpu_minutes':cumulative_gpu/60,'estimated_cumulative_CU':estimated_cu},ensure_ascii=False))
