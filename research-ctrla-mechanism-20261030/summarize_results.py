import os
os.environ['MPLCONFIGDIR']='/tmp/ctrla-matplotlib-cache'
from ctrla_adapter import *
import csv,subprocess,time
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def metric(x,scale=100):
    if x is None:return 'invalid denominator'
    return f"{x['estimate']*scale:.3f} [{x['ci95'][0]*scale:.3f}, {x['ci95'][1]*scale:.3f}]"

def savefig(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    status=json.loads((ROOT/'frozen-stage-status.json').read_text());assert status['status']=='complete'
    results=[json.loads((ROOT/f'seed{s}-frozen-results.json').read_text()) for s in (0,1)]
    trains=[json.loads((ROOT/f'seed{s}-training-summary.json').read_text()) for s in (0,1)]
    budget=json.loads((ROOT/'budget-state.json').read_text());env=json.loads((ROOT/'environment-preflight.json').read_text())
    aggregate=[];operations=[];integrity=[]
    for r in results:
        seed=r['seed'];challenge=np.load(ROOT/'data'/'audit_challenges.npz')['order']
        active=np.any((challenge[:,:,:-1]<6)&(challenge[:,:,1:]<6),axis=2)
        for renderer in 'EF':
            s=np.load(ROOT/'raw'/f'seed{seed}-audit-raw-N5-S-full.npz')
            f=np.load(ROOT/'raw'/f'seed{seed}-audit-raw-N5-{renderer}-full.npz')
            assert np.array_equal(s['prediction'][~active],f['prediction'][~active])
            assert np.allclose(s['ce'][~active],f['ce'][~active],atol=1e-6,rtol=0)
        integrity.append({'seed':seed,'no_adjacent_affine_challenges':int((~active).sum()),'no_intervention_predictions_and_CE_identical':True,'model_BN_unchanged':r['frozen_model_and_BN_bitwise_unchanged']})
        for family,values in r['families'].items():
            for renderer,v in values['renderers'].items():
                agg=v['mechanism']['aggregate'];match=v['audit_CE_match']
                row={'seed':seed,'family':family,'renderer':renderer,'coefficient':v['coefficient'],'error':agg['error']['estimate'],'error_ci_low':agg['error']['ci95'][0],'error_ci_high':agg['error']['ci95'][1],'E':agg['E']['estimate'],'E_ci_low':agg['E']['ci95'][0],'E_ci_high':agg['E']['ci95'][1],'new_wrong':agg['new_wrong']['estimate'],'corrected_wrong':agg['corrected_wrong']['estimate'],'relative_CE_difference':match['estimate'],'relative_CE_ci_low':match['ci95'][0],'relative_CE_ci_high':match['ci95'][1],'audit_CE_match_accepted':match['accepted']}
                aggregate.append(row)
                for op in v['mechanism']['by_operation']:
                    out={'seed':seed,'family':family,'renderer':renderer,'operation':op['operation'],'weight':op['weight'],'ratio_valid':op['ratio_valid'],'prefix_accuracy':op['prefix_accuracy'],'full_accuracy':op['full_accuracy'],'base_accuracy':op['base_accuracy'],'single_accuracy':op['single_accuracy']}
                    for key in ('conditional_R','single_R','ROR_gap','E','new_wrong','corrected_wrong'):
                        obj=op[key];out[key]=obj['estimate'] if obj else '';out[key+'_ci_low']=obj['ci95'][0] if obj else '';out[key+'_ci_high']=obj['ci95'][1] if obj else ''
                    operations.append(out)
    for name,rows in [('aggregate-metrics.csv',aggregate),('operation-metrics.csv',operations)]:
        with (ROOT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(ROOT/'result-integrity.json',integrity)
    # Original-image ROR intervals also retain all151 views within each image.
    from frozen_diagnostics import bootstrap_weights
    ror_rows=[];ror_curves={};w=bootstrap_weights(2000)
    for seed in (0,1):
        raw=np.load(ROOT/'raw'/f'seed{seed}-epoch10-audit-official-ror.npz')
        correct=(raw['prediction']==raw['target'][None,:]).astype(float)
        acc=correct.mean(1);bacc=w@correct.T;bror=bacc/bacc[:,0,None]
        ror=acc/acc[0];bounds=np.quantile(bror,[.025,.975],axis=0)
        ror_curves[seed]=(ror,bounds)
        for op in range(15):
            for level in range(11):
                cond=0 if level==0 else 1+10*op+level-1
                ror_rows.append({'seed':seed,'operation':OPS[op],'strength':level/10,'accuracy':acc[cond],'ROR':ror[cond],'ROR_ci_low':bounds[0,cond],'ROR_ci_high':bounds[1,cond]})
    with (ROOT/'official-ror-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ror_rows[0]));writer.writeheader();writer.writerows(ror_rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'S':'#4b5563','E':'#d97706','F':'#2563eb'}
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for t in trains:
        h=t['history'];epochs=[v['epoch'] for v in h]
        axs[0].plot(epochs,[v['controller_accuracy']*100 for v in h],marker='o',label=f"seed {t['seed']}")
        axs[1].plot(epochs,[v['train_ce'] for v in h],label=f"seed {t['seed']} online train")
        axs[1].plot(epochs,[v['controller_ce'] for v in h],ls='--',label=f"seed {t['seed']} clean controller")
    axs[0].set(xlabel='Actual epoch (92,000 images)',ylabel='Controller accuracy (%)',title='Common checkpoint learning health');axs[1].set(xlabel='Actual epoch',ylabel='Cross entropy',title='Online train and epoch-end eval losses')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    savefig(fig,'training-health')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,r in zip(axs,results):
        for j,(family,label) in enumerate([('raw','Original ASD'),('same_parameter_matched','Shared S coefficient'),('renderer_rematched','Renderer rematched')]):
            for k,renderer in enumerate('SEF'):
                m=r['families'][family]['renderers'][renderer]['mechanism']['aggregate']['error'];p=m['estimate']*100;lo,hi=np.array(m['ci95'])*100
                ax.errorbar(j+(k-1)*.18,p,yerr=[[p-lo],[hi-p]],fmt='o',color=colors[renderer],capsize=3,label=renderer if j==0 else None)
        ax.set_xticks(range(3),['Original ASD','Shared S scale','Own CE scales'],rotation=15);ax.set(ylabel='Audit error (%)',title=f"seed {r['seed']}; image-cluster 95% CI");ax.legend();ax.grid(axis='y',alpha=.2)
    savefig(fig,'rendering-error')
    fig,axs=plt.subplots(1,2,figsize=(11,6),sharey=True)
    for ax,r in zip(axs,results):
        values=r['families']['raw']['paired_rendering_differences']['S_minus_F'];rows=values['by_operation'];xs=np.array([x['estimate'] for x in rows])*100;lo=np.array([x['ci95'][0] for x in rows])*100;hi=np.array([x['ci95'][1] for x in rows])*100
        ax.errorbar(xs,np.arange(15),xerr=[xs-lo,hi-xs],fmt='o',capsize=2,color=colors['F']);ax.axvline(0,color='gray',lw=1);ax.axvline(2,color='#dc2626',ls='--',lw=1)
        ax.set_yticks(range(15),OPS);ax.invert_yaxis();ax.set(xlabel='error(S) - error(F) (percentage points)',title=f"seed {r['seed']}: aggregate {metric(values['aggregate'])} pp");ax.grid(axis='x',alpha=.2)
    fig.suptitle('Original ASD: positive favors fusion; 2pp gate applies to aggregate only')
    savefig(fig,'paired-rendering-gain')
    fig,axs=plt.subplots(1,2,figsize=(11,6),sharey=True)
    for ax,r in zip(axs,results):
        for k,renderer in enumerate('SEF'):
            rows=r['families']['raw']['renderers'][renderer]['mechanism']['by_operation'];xs=np.array([x['E']['estimate'] for x in rows])*100;lo=np.array([x['E']['ci95'][0] for x in rows])*100;hi=np.array([x['E']['ci95'][1] for x in rows])*100
            ax.errorbar(xs,np.arange(15)+(k-1)*.15,xerr=[xs-lo,hi-xs],fmt='o',ms=3,capsize=2,color=colors[renderer],label=renderer)
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(15),OPS);ax.invert_yaxis();ax.set(xlabel='Error interaction E (percentage points)',title=f"seed {r['seed']}; original ASD");ax.legend();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Conditional interaction is descriptive; Ctrl-A does not promise E = 0')
    savefig(fig,'conditional-interaction')
    fig,axs=plt.subplots(3,5,figsize=(15,9),sharex=True,layout='constrained')
    for op,ax in enumerate(axs.flat):
        for seed in (0,1):
            curve,bounds=ror_curves[seed];ix=np.r_[0,np.arange(1+op*10,1+(op+1)*10)]
            ax.plot(np.linspace(0,1,11),curve[ix],label=f'seed {seed}')
            ax.fill_between(np.linspace(0,1,11),bounds[0,ix],bounds[1,ix],alpha=.15)
        ax.set_title(OPS[op]);ax.axhline(1,color='gray',lw=.5);ax.grid(alpha=.2)
    axs[0,0].legend();fig.supxlabel('Normalized strength');fig.supylabel('Official raw-image single-operation ROR');savefig(fig,'official-ror')
    matching_counts={r['seed']:sum(v['audit_CE_match']['accepted'] for v in r['families']['renderer_rematched']['renderers'].values()) for r in results}
    lines=['# Ctrl-A 冻结机制筛选实测报告','',f"完成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",'',f"资源门结论：**{status['decision']}**。未启动在线四臂；冻结实验没有检验后续全局反馈是否补偿。",'',
           f"同参数F相对S的改善为 seed0 {metric(results[0]['resource_gate'])} pp、seed1 {metric(results[1]['resource_gate'])} pp；均为小幅正向结果，但未达到预设+2pp资源门。这不是零效应结论。分别重新匹配后，完整audit区间验收为 seed0 {matching_counts[0]}/3、seed1 {matching_counts[1]}/3 通过；未通过者不能声称已排除平均难度差异。",'',
           '## 共同 checkpoint 与预算','',
           '| seed | 初始 controller acc | epoch10 controller acc | epoch10 clean CE | 实际 epoch | 原46k曝光 |','|---|---:|---:|---:|---:|---:|']
    for t in trains:lines.append(f"| {t['seed']} | {t['initial_controller']['accuracy']*100:.2f}% | {t['final_controller']['accuracy']*100:.2f}% | {t['final_controller']['ce']:.4f} | 10 | 20 |")
    lines+=['',f"模型后端为 MPS / float32，Apple M5 Pro、20 GPU 核、64 GB；CPU Pillow 只负责官方预处理。真实训练 {budget['train_images']:,} 张（上限1,840,000），推理账本 {budget['inference_images']:,} image-forwards（上限8,000,000），另有2,008个合成训练/梯度前向。从首次完整模型测速到最后阶段账本的连续墙钟 {budget['elapsed_wall_seconds']/60:.2f} 分钟（硬上限480分钟）。Colab **0 CU**。",'',
           '## 预注册主资源门','',
           'Δ_err = error(S) − error(F)，原 ASD、N5、同参数；正数表示F改善。95% CI按原图配对重采样，保留其全部15视图。每个 seed 都须点估计≥+2pp且CI下界>0；此门只用于资源分配。','',
           '| seed | Δ_err pp [95% CI] | 该seed过门 |','|---|---:|---|']
    for r in results:lines.append(f"| {r['seed']} | {metric(r['resource_gate'])} | {r['resource_gate']['seed_pass']} |")
    lines+=['','## N2 冻结参照','',
           '| seed | renderer | N2错误率% [95% CI] | 平均CE |','|---|---|---:|---:|']
    for r in results:
        for renderer,v in r['N2'].items():lines.append(f"| {r['seed']} | {renderer} | {metric(v['error'])} | {v['mean_ce']:.6f} |")
    lines+=['','## 冻结错误率、交互与难度匹配','',
           '表中为按官方末端操作概率1/15汇总。原始参数的三种渲染同参数；shared系数来自S的strength-match选择并共用于S/E/F；rematched分别调幅，不能称几何相同。','',
           '| seed | family | renderer | 系数 | 错误率% [CI] | E pp [CI] | 相对N2 CE差% [CI] | audit匹配合格 |','|---|---|---|---:|---:|---:|---:|---|']
    for r in results:
        for fam,vals in r['families'].items():
            for renderer,v in vals['renderers'].items():
                agg=v['mechanism']['aggregate'];match=v['audit_CE_match']
                lines.append(f"| {r['seed']} | {fam} | {renderer} | {v['coefficient']:.5f} | {metric(agg['error'])} | {metric(agg['E'])} | {metric(match)} | {match['accepted']} |")
    lines+=['','匹配合格要求 audit 相对CE差的点估计和完整95%CI均在±5%内。未通过的比较不能用来排除总体难度。匹配只约束平均CE，不保证损失分布、各类难度或语义保真相同。原始N2只是同一冻结模型的难度参照，未训练额外N2模型。','',
           '完整 conditional R、同底图 single R、ROR gap、E、新增错/纠正错及每项CI见 operation-metrics.csv 和各 seed JSON；低分母明确标记无效，不挑最坏组合。官方原始图 ROR 独立保存为151个条件，配对CI见official-ror-metrics.csv，防止把基础crop/flip上下文当成组合效应。这里的条件比值按固定抽样的前缀上下文分布汇总，不是每种固定组合的穷举结果。','',
           '## 审计、故障与限制','',
           '19项预训练测试通过；另有2-worker spawn冒烟测试。矩阵原32×32坐标、单操作、整数平移/逆操作、边界、顺序、光度屏障、分层数据与统计方向均有测试。首次MPS channels-last反向失败后仅改连续布局，CPU/MPS梯度相对L2差约2.02e-6；第二项故障是动态模块的spawn序列化，真实训练曝光0时修复。两次失败与恢复回执均保留，硬时限未重置。',
           '冻结前后模型参数和BN状态逐项字节一致。无相邻仿射块的挑战上S/E/F预测及CE一致，见result-integrity.json。推理读取方式0/6 workers逐字节哈希一致，改0 workers减少启停开销，见reader-benchmark.json。源代码未修改，历史tracked文件保全检查另见preservation-after.json。',
           '这是缩短机制实验：airbench94结构，两个seed各10实际epoch，非论文最终精度复现。代码提交e77a4c1早于v2；新分层不代表独立确认。图像bootstrap不代表训练seed不确定性。覆盖率不代表语义保真；E/F的prefix是裁回后的读出，full在连续块内保留渲染状态，细节见PRE_RUN_NOTES_ZH.md。扩展画布也改变边界填充参与插值的方式；Pillow uint8量化保留，不能把所有像素变化都归为纯粹信息恢复。非零交互不说明Ctrl-A论文承诺被推翻。',
           '官方test批次未解包、未实例化、未评估；公开归档含test成员的压缩字节但从未读取其图像/标签。数据、权重、原始预测、日志和第三方源码受本目录.gitignore保护；未commit/push或上传。','',
           '## 来源与运行','',
           '- Ctrl-A v2： https://arxiv.org/html/2603.21819v2 （2026-10-26）。',
           '- 官方代码： https://github.com/jbcdfm/ControlAugment/tree/e77a4c19a3db893c120fc18b774fa290e826592b （MIT，版本固定）。',
           '- Lazy Resampling 的融合重采样先例： https://monai.readthedocs.io/en/stable/lazy_resampling.html 及 https://github.com/KCL-BMEIS/LazyResampling 。单纯仿射融合获益不是新贡献。',
           '- CIFAR-10 官方下载与训练成员校验： asset-manifest.json；源码hash： source-manifest.json；接口差异： patch-manifest.json。','',
           '执行入口：`../.venv/bin/python -B run_stages.py`（本次已完成，不应直接重复执行）；独立阶段为`train_common.py`、`frozen_diagnostics.py`；汇总为`summarize_results.py`。协议在`protocol.json`与`protocol-lock.json`，采样在`sampling-manifest.json`，实际命令与文件hash在`run-manifest.json`及恢复/最终清单。所有运行均需当前获准且可见MPS的本机执行环境。','',
           '## 图与原始结果','',
           '- figures/training-health.png：学习健康度。','- figures/rendering-error.png：三组渲染错误率。','- figures/paired-rendering-gain.png：两个seed各操作的同参数收益。','- figures/conditional-interaction.png：各操作E及CI。','- figures/official-ror.png：原始图单操作ROR。','- raw/*.npz：全部固定样本预测、标签、CE；weights/*.pt：完整epoch模型/BN、optimizer、schedule、RNG、ξ/Γ/α。','']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines))
    (ROOT/'LATEST_STATUS_ZH.md').write_text(f"# 已完成\n\n{status['decision']}；两个共同checkpoint及冻结诊断完成，未启动在线四臂。\n\n完整结果见 RESULTS_ZH.md，原始预测/CE见 raw/，各操作与CI见CSV和各seed JSON。\n\n训练{budget['train_images']:,}张；推理{budget['inference_images']:,} forwards；连续墙钟{budget['elapsed_wall_seconds']/60:.2f}分钟；Colab 0CU。\n")
    progress=json.loads((ROOT/'PROGRESS.json').read_text());progress.update(completed_utc=datetime.now(timezone.utc).isoformat(),status='complete',decision=status['decision'],budget=budget,current_blocker=None,active_stage=None);write_json(ROOT/'PROGRESS.json',progress)
    previous=json.loads((ROOT/'tracked-preservation-before.json').read_text());project=ROOT.parent
    changed=[p for p,h in previous.items() if not (project/p).is_file() or sha256(project/p)!=h]
    source=json.loads((ROOT/'source-manifest.json').read_text());changed_source=[p for p,h in source['source_files'].items() if sha256(ROOT/p)!=h]
    lock=json.loads((ROOT/'protocol-lock.json').read_text())
    preserved={'historical_tracked_files_checked':len(previous),'changed_historical_tracked_files':changed,'changed_upstream_source_files':changed_source,'protocol_hash_unchanged':sha256(ROOT/'protocol.json')==lock['protocol_sha256'],'sampling_hash_unchanged':sha256(ROOT/'sampling-manifest.json')==lock['sampling_manifest_sha256'],'git_status':subprocess.check_output(['git','status','--short'],cwd=project,text=True)}
    write_json(ROOT/'preservation-after.json',preserved)
    assert not changed and not changed_source and preserved['protocol_hash_unchanged'] and preserved['sampling_hash_unchanged']
    files={str(p.relative_to(ROOT)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ('sources','data','__pycache__')) and p.name!='deliverable-manifest.json'}
    write_json(ROOT/'deliverable-manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':files,'decision':status['decision'],'budget':budget,'preservation':preserved,'raw_predictions_local_only':True})
    print(json.dumps({'decision':status['decision'],'gates':[r['resource_gate'] for r in results],'budget':budget,'report':str(ROOT/'RESULTS_ZH.md')},indent=2),flush=True)

if __name__=='__main__':main()
