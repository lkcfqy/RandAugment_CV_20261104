"""Descriptive rendering of all saved results; no model or estimator."""
from pathlib import Path
import os, json
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

s=json.loads((ROOT/'SYNTHESIS.json').read_text())
groups=[('dinov2_small',8,'DINOv2 / B8'),('dinov2_small',16,'DINOv2 / B16'),('deit_tiny',8,'DeiT / B8'),('deit_tiny',16,'DeiT / B16')]
rows=s['all_48_settings']
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for ax,(model,B,label) in zip(axes.flat,groups):
    rr=[r for r in rows if r['encoder']==model and r['B']==B]
    for key,title,color,marker,off in [('selected_minus_weak_pp','CV minus weak','#0072B2','o',-.16),('selected_minus_ra_pp','CV minus RA','#D55E00','^',0),('ra_minus_weak_pp','Fixed RA minus weak','#009E73','s',.16)]:
        ax.scatter([r['episode']+off for r in rr],[r[key] for r in rr],label=title,c=color,marker=marker,s=28)
    ax.axhline(0,c='#555555',lw=1);ax.set_title(label);ax.set_xticks(range(12));ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
handles,labels=axes[0,1].get_legend_handles_labels()
fig.legend(handles,labels,fontsize=8,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3)
for ax in axes[:,0]:ax.set_ylabel('Paired query accuracy difference (pp)')
for ax in axes[1]:ax.set_xlabel('Support episode')
fig.suptitle('All episodes: fixed-policy gaps and CV selection differences')
fig.text(.5,.015,'Shared query / overlapping supports. Descriptive values, no independent-trial inference.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.90))
fig.savefig(ROOT/'episode_differences.png',dpi=170);fig.savefig(ROOT/'episode_differences.svg')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
for i,(model,B,label) in enumerate(groups):
    d=s['summaries'][f'{model}:B{B}'];rr=[r for r in rows if r['encoder']==model and r['B']==B]
    axes[0].bar(i-.25,d['oracle_space_vs_expost_best_constant_mean_pp'],width=.23,color='#0072B2',label='Oracle gain over ex-post best constant' if i==0 else None)
    axes[0].bar(i,d['distributions']['regret_pp']['mean'],width=.23,color='#D55E00',label='Original CV regret' if i==0 else None)
    axes[0].bar(i+.25,d['cv_tie_sensitivity']['bounds']['regret_pp'][0],width=.23,color='#E6B566',label='Most favorable tie assignment regret' if i==0 else None)
    for r in rr:
        color={'clean':'#555555','weak':'#0072B2','ra':'#D55E00'}[r['selected']]
        axes[1].scatter(i+(r['episode']-5.5)*.035,r['regret_pp'],c=color,s=34)
for ax in axes:
    ax.set_xticks(range(4),[g[2].replace(' / ','\n') for g in groups]);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Mean accuracy difference (pp)');axes[0].set_title('Opportunity and cost accounting');axes[0].legend(fontsize=7)
axes[1].set_title('All 12 episode regrets per setting');axes[1].set_ylabel('Query oracle minus CV-selected accuracy (pp)')
for p,c in [('clean','#555555'),('weak','#0072B2'),('ra','#D55E00')]:axes[1].scatter([],[],c=c,label='Selected '+p,s=30)
axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
fig.text(.5,.012,'Oracle and best-constant references use query labels; they are not deployment choices or confidence bounds.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,.86,1));fig.savefig(ROOT/'opportunity_regret.png',dpi=180);fig.savefig(ROOT/'opportunity_regret.svg');plt.close(fig)

md=['# All 48 settings','', 'Accuracy and differences in percent / pp. C=clean, W=weak, R=RA. OOF is total correct from the four held-out folds. Every setting is included.','']
for model,B,label in groups:
    md += [f'## {label}','','| Episode | C | W | R | OOF C/W/R correct | Choice | CV−W | CV−R | R−W | Regret | Full-query best set |','|---:|---:|---:|---:|:---|:---|---:|---:|---:|---:|:---|']
    for r in rows:
        if r['encoder']==model and r['B']==B:
            a=r['full_query_accuracy']
            md.append(f"| {r['episode']} | {a[0]:.1f} | {a[1]:.1f} | {a[2]:.1f} | {'/'.join(map(str,r['cv_correct']))} / {r['cv_total']} | {r['selected']} | {r['selected_minus_weak_pp']:+.1f} | {r['selected_minus_ra_pp']:+.1f} | {r['ra_minus_weak_pp']:+.1f} | {r['regret_pp']:.1f} | {','.join(r['full_query_best_set'])} |")
    md += ['']
(ROOT/'ALL_EPISODES.md').write_text('\n'.join(md)+'\n')

md=['# All incorrect full-query choices: per-fold evidence','', 'Post-result descriptive analysis. Each row compares the actual CV choice to one full-query optimum; all tied optima are retained. CV validation denominator per fold is 20 (B8) or 40 (B16). Query denominator is 1000 per fold. Margins are integer correct-count differences, selected minus reference. These folds are dependent and are not separate statistical trials.','', '| Encoder/B/episode | Selected → reference | CV selected counts [4 folds] | CV reference counts [4 folds] | CV margin | Fold-query margins [4 folds] | Full-query regret pp |','|:---|:---|:---|:---|---:|:---|---:|']
for r in rows:
    si=['clean','weak','ra'].index(r['selected'])
    for ref in r['wrong_selection_references']:
        oi=['clean','weak','ra'].index(ref['query_optimal_policy'])
        md.append(f"| {r['encoder']}/{r['B']}/{r['episode']} | {r['selected']} → {ref['query_optimal_policy']} | {r['cv_fold_correct'][si]} | {r['cv_fold_correct'][oi]} | {ref['cv_total_margin_selected_minus_optimum']} | {ref['fold_query_margins_selected_minus_optimum']} | {r['regret_pp']:.1f} |")
(ROOT/'ALL_WRONG_CHOICES.md').write_text('\n'.join(md)+'\n')

md=['# Complete CV-tie candidate sensitivity','', 'Original tie rule remains clean > weak > RA. All query-labelled choices below are diagnostic bounds, not a replacement policy. No candidates or assignments are omitted.','']
for t in s['all_cv_tied_settings']:
    md += [f"## {t['encoder']} B={t['B']}, episode {t['episode']}",'',f"Original choice: {t['original_choice']}. Tied OOF correct: {t['tied_cv_correct']}/{t['cv_total']}.",'','| Candidate | CV correct by fold | Fold-query mean % | Full-query % | Regret pp |','|:---|:---|---:|---:|---:|']
    for a in t['all_candidates']:md.append(f"| {a['policy']} | {a['cv_fold_correct']} | {a['fold_query_mean_accuracy']:.3f} | {a['full_query_accuracy']:.1f} | {a['regret_pp']:.1f} |")
    md += ['']
for key,d in s['summaries'].items():
    sensitivity=d['cv_tie_sensitivity'];episodes=[i for i,x in enumerate(sensitivity['eligible_indices']) if len(x)>1]
    md += [f'## All assignments: {key}','',f'Variable episodes: {episodes}. Other choices remain exactly the original unique CV best.','', '| Choices at variable episodes | Mean accuracy | CV−weak | CV−RA | Mean regret | C/W/R selection counts |','|:---|---:|---:|---:|---:|:---|']
    for a in sensitivity['all_assignments']:
        choices=[s['policy_order'][a['choice_indices'][e]] for e in episodes]
        counts=[a['selected_counts'][p] for p in s['policy_order']]
        md.append(f"| {choices} | {a['accuracy_percent']:.3f} | {a['minus_weak_pp']:+.3f} | {a['minus_ra_pp']:+.3f} | {a['regret_pp']:.3f} | {counts} |")
    md += ['',f"Marginal frequency bounds (not all jointly realizable): {sensitivity['frequency_bounds']}",'']
(ROOT/'ALL_TIE_ASSIGNMENTS.md').write_text('\n'.join(md)+'\n')
