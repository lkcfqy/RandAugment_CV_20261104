"""Pre-query-written rendering plan: every final setting, no new model or choice."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=json.loads((ROOT/'RESULTS.json').read_text());assert s['status']=='COMPLETE_BOTH_ENCODERS'
groups=[('dinov2_small',8),('dinov2_small',16),('deit_tiny',8),('deit_tiny',16)]
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for ax,(model,B) in zip(axes.flat,groups):
    rows=[r for r in s['all48_settings'] if r['encoder']==model and r['B']==B];assert len(rows)==12
    for key,label,color,marker,dx in [('selected_minus_weak_pp','CV minus fixed weak','#0072B2','o',-.08),('selected_minus_ra_pp','CV minus fixed RA','#D55E00','^',.08)]:
        ax.scatter([r['episode']+dx for r in rows],[r[key] for r in rows],label=label,c=color,marker=marker,s=32)
    ax.axhline(0,c='#444444',lw=1);ax.set_title(f'{model}, B={B}/class');ax.set_xticks(range(12));ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
for ax in axes[:,0]:ax.set_ylabel('Paired query accuracy difference (pp)')
for ax in axes[1]:ax.set_xlabel('Support episode')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,fontsize=9)
fig.suptitle('CIFAR-100: all48 frozen settings')
fig.text(.5,.015,'All100 classes; previously used data. Shared query and overlapping supports; no independence claim.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.90));fig.savefig(ROOT/'paired_differences.png',dpi=170);fig.savefig(ROOT/'paired_differences.svg');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4))
for i,(model,B) in enumerate(groups):
    r=s['summaries'][f'{model}:B{B}']
    for offset,k,color,label in [(-.23,'H_pp','#0072B2','Oracle opportunity H'),(0,'C_pp','#D55E00','Original CV regret C'),(.23,None,'#E6B566','Best possible tie regret C_min')]:
        value=r[k] if k else r['tie_bounds']['mean_regret_bounds'][0]
        ax.bar(i+offset,value,width=.22,color=color,label=label if i==0 else None)
ax.set_xticks(range(4),['DINO / B8','DINO / B16','DeiT / B8','DeiT / B16']);ax.set_ylabel('Percentage points');ax.grid(axis='y',alpha=.2)
ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.15),ncol=3);ax.spines[['top','right']].set_visible(False)
fig.text(.5,.01,'H and tie bounds use query labels for diagnosis; they are not deployable choices.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(ROOT/'opportunity_cost.png',dpi=170);fig.savefig(ROOT/'opportunity_cost.svg');plt.close(fig)

md=['# Complete tied-CV candidate records','','Original C>W>R tie rule remains primary. All eligible per-setting candidates are retained; aggregate bounds exactly cover the full Cartesian product without exponential duplicate output.','',
    '| Encoder | B | Episode | Original choice | Candidate | CV correct | Query accuracy | Query regret |','|:---|---:|---:|:---|:---|---:|---:|---:|']
for r in s['all48_settings']:
    if len(r['cv_best'])>1:
        for candidate in r['cv_best']:
            j=['clean','weak','ra'].index(candidate);a=r['full_accuracy']
            md.append(f"| {r['encoder']} | {r['B']} | {r['episode']} | {r['selected']} | {candidate} | {r['cv_correct'][j]} | {a[j]:.4f} | {max(a)-a[j]:.4f} |")
md+=['','All candidates, original selections and class-count-scale differences remain in RESULTS.json.']
(ROOT/'ALL_CV_TIES.md').write_text('\n'.join(md)+'\n')
