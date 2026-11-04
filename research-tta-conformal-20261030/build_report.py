import hashlib,json,os
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
perf=json.loads((D/'performance-gate.json').read_text())
env=json.loads((D/'environment.json').read_text())
mathcheck=json.loads((R/'math-acceptance.json').read_text())
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
if not decision['B_tuning_selection_feasible']:reasonB.append('tuning覆盖/类覆盖约束下不存在可行温度')
for field,label in [('gain_pass','至少一个原输出温度的缩集优势不超过同时下界0.25'),('D_pass','类覆盖Dhat可比性未全部证实'),('coverage_pass','边际覆盖及1pp容差未全部证实')]:
    if not all(c[field] for c in decision['B_all_7_comparators']):reasonB.append(label)
if not decision['B_restored_gain_pass']:reasonB.append('恢复原排序后的实质优势消失未证实')
if not decision['B_attenuation_pass']:reasonB.append('恢复原排序造成至少0.25标签衰减未证实')
if not all(c['mechanism_beyond_control_established'] for c in decision['negative_controls']):reasonB.append('未建立超出SAPS/rankonly已知机制的优势')
alloc=json.loads((R/'cli-ledger/allocate-t4.json').read_text())
stop=json.loads((R/'cli-ledger/stop-complete.json').read_text())
gpu_seconds=stop['finished_epoch']-alloc['started_epoch']
wall=release['released_epoch']-next(json.loads(line)['allocation_elapsed_seconds']+alloc['started_epoch'] for line in (D/'events.jsonl').read_text().splitlines() if json.loads(line)['event']=='download_started')
estimated_cu=gpu_seconds/3600*1.15
storage=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
budget={'allocation_start_utc':alloc['started_utc'],'stop_finished_epoch':stop['finished_epoch'],
        'gpu_seconds_upper_from_request_start':gpu_seconds,'actual_rate_cu_per_hour':1.15,'rate_based_cu_upper_estimate':estimated_cu,
        'budget_cu_limit':6,'budget_gpu_seconds_limit':7200,'download_to_release_seconds':wall,
        'preallocation_balance':41.94,'postrelease_usage':release['usage_output'],
        'postrelease_sessions':release['sessions_output'],'local_staging_bytes':storage,
        'formal_steps':state['formal_steps'],'benchmark_steps':state['benchmark_steps'],
        'formal_presentations':state['formal_presentations'],'benchmark_presentations':state['benchmark_presentations'],
        'inference_views':state['inference_views'],'repeated_training_runs':1,
        'private_history_uploaded':False,'new_oauth_started':False,'Mac_GPU_used':False}
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

SAPS使用原论文Eq10；rankonly=rank-1+U。RAPS按预定诊断公式，正则项不随机，不能混称原论文所有随机化实现的精确复现。SAPS/rankonly的原输出tuning温度与q温度不同的点若不在预注册同时检验族中，其不确定性会阻止B升级，未事后扩大检验族或选更好点。逐项门判据见 [decision.json](analysis/decision.json)。

alpha=.05完整结果仅作敏感性，未用于救题。所有224个组合保留于 [CSV](analysis/all-point-estimates.csv) 和 [含逐类覆盖的JSON](analysis/all-point-estimates.json)。

## 数据、数值和推断验收

- 公开CIFAR100官方Python包MD5匹配 `eb9058c3a382ffc7106e4002c42a8d85`；SHA256 `{split['archive_sha256']}`。数据来源和引用见 [SOURCES.md](SOURCES.md)，未标MIT许可。
- 60k图像发现{split['duplicate_group_count']}个精确重复组、{split['duplicate_images']}张图；{split['cross_origin_duplicate_groups']}组跨官方train/test。按训练前标签盲分组规则分配，cross-role duplicate=0；train40k、tuning4k、cal6k、eval10k均精确满足。图像ID、原来源、内容SHA、角色见 [split manifest](results/split-manifest.jsonl)。近重复未检测。
- CPU数学验收{len(mathcheck['checks'])}项全部通过，涵盖退化恒等、温度/置换、同谱集合大小、rankonly、精确次序统计量与尾部/空集政策，见 [math-acceptance.json](math-acceptance.json)。
- FP32 logits按image/view/class保存，baseline为中心原图view4；GPU输出流SHA与读回SHA一致，逐样本均值验收通过。后续评分均float64，允许空集、不补top1，稳定classID并列与donor rank保留。细节见 [cache acceptance](results/cache-acceptance.json) 与 [数值诊断](analysis/numerical-diagnostics.json)。
- 999次cal/eval分别重采样重复组、同次跨臂配对、重新校准；cal有{bootstrap['cal_group_count']}组，eval有{bootstrap['eval_group_count']}组。预注册族共{ci['family_size']}个维度，95% max-|t|临界值{ci['critical95']:.4f}，90%为{ci['critical90']:.4f}。没有独立重复训练。置信区间是近似bootstrap，同时控制不是有限样本精确保证。
- eval逐类n范围{min(counts)}–{max(counts)}，不足以给每类1pp精度保证；逐类Wilson95%区间只是描述性而非100类同时保证。任何类n<2会标不可估，本次D不可估的variant-resample数为{ci['invalid_D_variant_resamples']}。Dhat负值保留。

## 资源与交付

实际GPU为{env['gpu']}，官方CLI0.7.5，现有授权正常，未启动新OAuth。T4费率1.15CU/小时。分配请求起至stop完成共{gpu_seconds/60:.2f}分钟，按费率保守估计{estimated_cu:.3f}CU，均低于120分钟/6CU。性能门最初保守预测{perf['predicted_total_gpu_seconds']/60:.2f}分钟；未降低80epochs/模型/视图配置。正式25040步+测速200步=25240步，呈现3,200,000+25,600=3,225,600，推理205,120views。

产物完成后先取回并验证bundle SHA，再释放GPU，CPU统计未占用GPU。释放后服务器报告：`{release['sessions_output'].strip()}`。释放后额度/费率：

```
{release['usage_output'].strip()}
```

账户余额只有两位小数，余额差不是精确计费。首次下载到释放{wall/60:.2f}分钟；最终归档墙钟和目录字节计数由 `archive-receipt.json` 记录。资源账本见 [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json)。没有Mac MPS、commit/push、购买或上传私密历史项目。仅本次独立脚本进入本人Colab。

执行失败分支完整保留：普通沙箱读取CLI子命令帮助时曾因CLI写自身日志受限失败，之后以已授权正常日志/缓存权限解决，没有新OAuth；创建项目归档目录时执行服务一次断开，核实目录不存在后安全恢复。18:02:39 UTC发出的Jupyter只读状态查询在180秒后超时(exit124)，先核对会话和usage，再通过官方CLI文件下载读取日志，确认训练仍在进行，切换监控通道继续；没有重启模型、重复训练或新增推理。相关原始回执均在 `cli-ledger/`，监控日志在 `monitor/`。

## 解释边界

已有 [TTA-conformal论文](https://arxiv.org/html/2505.22764v1) 已讨论rank改善与概率平滑；[TS4CP/ConfTS](https://arxiv.org/abs/2402.05806) 和 [SAPS](https://arxiv.org/html/2310.06430v2) 已覆盖温度/rank机制。因此普通缩集、普通rank收益和四格分解不能独自作为新贡献。本次没有复用其未经核实许可的研究代码。

这是内部heldout重分割，不是官方test成绩；组原子分配使图像级抽样不再完全独立，cluster bootstrap用于反映已知重复依赖。Dhat的经典无偏修正在独立同分布类内观测下成立；在重复组存在时应视为常规修正估计而非宣称严格无偏。单模型、一个增强协议、有限温度族、保守同时CI及有限样本限制了功效。未通过预定门只关闭此候选，不证明所有TTA机制不存在。

完整协议：[PROTOCOL_ZH.md](PROTOCOL_ZH.md)；训练前哈希：[protocol-lock.json](protocol-lock.json)；所有产物的SHA256见 `SHA256SUMS.json`。若无新指示，本候选不追加seed、模型或视图。
'''
(R/'REPORT_ZH.md').write_text(text)
print(json.dumps({'report':str(R/'REPORT_ZH.md'),'candidate_status':decision['candidate_status'],'gpu_minutes':gpu_seconds/60,'estimated_CU':estimated_cu},ensure_ascii=False))
