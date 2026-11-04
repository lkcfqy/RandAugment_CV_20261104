"""Post-result integer-count analysis. Numpy only; no model or head fitting."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
from datetime import datetime, timezone
import collections, csv, hashlib, itertools, json, time
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = _portable_project_path(__file__, '')
SOURCES = {'dinov2_small': PROJECT/'augmentation-selection-pilot-20261102',
           'deit_tiny': PROJECT/'augmentation-selection-deit-20261102'}
POLICIES = ['clean', 'weak', 'ra']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def winners(values):
    values = np.asarray(values)
    return np.flatnonzero(values == values.max()).tolist()

def count_stats(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'min': float(a.min()), 'max': float(a.max()),
            'sd_descriptive': float(a.std(ddof=1)) if len(a)>1 else 0.,
            'all': a.tolist(), 'positive': int((a>0).sum()),
            'zero': int((a==0).sum()), 'negative': int((a<0).sum())}

def paired_stage_flags(selected, fold_totals, full_totals):
    oracle=winners(full_totals)
    return {'selected_wrong_full':selected not in oracle,
            'selected_wrong_fold':selected not in winners(fold_totals),
            'fold_inferior_to_any_full_optimum':any(fold_totals[selected]<fold_totals[i] for i in oracle),
            'fold_inferior_to_all_full_optima':all(fold_totals[selected]<fold_totals[i] for i in oracle)}

def tie_enumeration(cv, full, query_n):
    cv, full = np.asarray(cv), np.asarray(full)
    assert cv.ndim == 2 and cv.shape == full.shape and cv.shape[1] == 3
    eligible = [winners(x) for x in cv]
    assignments = []
    oracle = full.max(1)
    fixed_mean = full.mean(0) * 100/query_n
    for choice in itertools.product(*eligible):
        picked = full[np.arange(len(full)), choice]
        assignments.append({'choice_indices': list(choice),
                            'accuracy_percent': float(picked.mean()*100/query_n),
                            'minus_weak_pp': float((picked-full[:,1]).mean()*100/query_n),
                            'minus_ra_pp': float((picked-full[:,2]).mean()*100/query_n),
                            'regret_pp': float((oracle-picked).mean()*100/query_n),
                            'selected_counts': {p:choice.count(i) for i,p in enumerate(POLICIES)}})
    return {'eligible_indices':eligible,'assignment_count':len(assignments),
            'all_assignments': assignments,
            'bounds': {k:[min(x[k] for x in assignments),max(x[k] for x in assignments)]
                       for k in ['accuracy_percent','minus_weak_pp','minus_ra_pp','regret_pp']},
            'frequency_bounds':{p:[min(x['selected_counts'][p] for x in assignments),
                                    max(x['selected_counts'][p] for x in assignments)] for p in POLICIES},
            'fixed_mean_accuracy':dict(zip(POLICIES,fixed_mean.tolist()))}

def main():
    started=time.perf_counter(); all_rows=[]; fold_rows=[]; pair_rows=[]; summaries={}; inputs={}; tie_rows=[]
    for model, source in SOURCES.items():
        heads=json.loads((source/'HEAD_IDENTITIES.json').read_text())
        seal=json.loads((source/'SELECTION_SEAL.json').read_text())
        old=json.loads((source/'SUMMARY.json').read_text())
        run=json.loads((source/'RUN_RECEIPT.json').read_text())
        assert run['status']=='COMPLETE' and run['query_start_utc']>seal['utc']
        filenames=['HEAD_IDENTITIES.json','SELECTION_SEAL.json','SUMMARY.json','RUN_RECEIPT.json','CV_PREDICTIONS.npz','QUERY_PREDICTIONS.npz','IDENTITIES.json','QUERY_IDENTITIES.json','CODE_FREEZE.json','PROTOCOL.json','MANIFEST_SHA256.json']
        inputs[model]={f:{'sha256':sha(source/f),'bytes':(source/f).stat().st_size} for f in filenames}
        assert inputs[model]['CV_PREDICTIONS.npz']['sha256']==seal['cv_predictions_sha256']
        assert inputs[model]['QUERY_PREDICTIONS.npz']['sha256']==run['query_predictions_sha256']
        with np.load(source/'CV_PREDICTIONS.npz',allow_pickle=False) as z:
            cvhead=z['head_index']; cvlabel=z['label']; cvprediction=z['prediction']; cvid=z['support_union_index']
        with np.load(source/'QUERY_PREDICTIONS.npz',allow_pickle=False) as z:
            pred=z['prediction']; label=z['label']; qi=z['query_index']; hi=z['head_index']
        assert pred.shape==(360,1000) and np.array_equal(hi,np.arange(360))
        qcorrect=(pred==label[None,:]).sum(1)
        index={(h['episode'],h['B'],h['policy'],h['fold']):h['index'] for h in heads}
        model_rows=[]
        for B in [8,16]:
            rows=[]
            for episode in range(12):
                cv=np.zeros((3,4),dtype=int); fq=np.zeros((3,4),dtype=int); full=np.zeros(3,dtype=int)
                for pi,policy in enumerate(POLICIES):
                    full[pi]=qcorrect[index[(episode,B,policy,-1)]]
                    heldout=[]
                    for fold in range(4):
                        idx=index[(episode,B,policy,fold)]; mask=cvhead==idx
                        assert mask.sum()==B*10//4
                        cv[pi,fold]=int((cvprediction[mask]==cvlabel[mask]).sum())
                        fq[pi,fold]=int(qcorrect[idx]); heldout.extend(cvid[mask].tolist())
                        fold_rows.append({'encoder':model,'B':B,'episode':episode,'policy':policy,'fold':fold,
                                          'validation_correct':int(cv[pi,fold]),'validation_total':int(mask.sum()),
                                          'query_correct':int(fq[pi,fold]),'query_total':len(label)})
                    assert len(set(heldout))==B*10
                totals=cv.sum(1); ftotal=fq.sum(1)
                choice=next(c for c in seal['choices'] if c['episode']==episode and c['B']==B)
                assert choice['cv_correct']==totals.tolist()
                selected=choice['selected_index']; assert selected==winners(totals)[0]
                oracle=winners(full); foldbest=winners(ftotal); tied=winners(totals)
                full_acc=full*100/len(label); fold_acc=ftotal*100/(4*len(label))
                row={'encoder':model,'B':B,'episode':episode,'selected':POLICIES[selected],
                     'cv_correct':totals.tolist(),'cv_total':B*10,'cv_fold_correct':cv.tolist(),
                     'fold_query_correct':fq.tolist(),'query_total':len(label),'full_query_correct':full.tolist(),
                     'full_query_accuracy':full_acc.tolist(),'fold_query_mean_accuracy':fold_acc.tolist(),
                     'cv_best_set':[POLICIES[i] for i in tied],
                     'fold_query_best_set':[POLICIES[i] for i in foldbest],
                     'full_query_best_set':[POLICIES[i] for i in oracle],
                     'selected_accuracy':float(full_acc[selected]),
                     'selected_minus_weak_pp':float((full[selected]-full[1])*100/len(label)),
                     'selected_minus_ra_pp':float((full[selected]-full[2])*100/len(label)),
                     'ra_minus_weak_pp':float((full[2]-full[1])*100/len(label)),
                     'regret_pp':float((full.max()-full[selected])*100/len(label)),
                     'selected_wrong_full':selected not in oracle,
                     'selected_wrong_fold':selected not in foldbest,
                     'fold_choice_wrong_full':foldbest[0] not in oracle,
                     'fold_query_choice_full_regret_pp':float((full.max()-full[foldbest[0]])*100/len(label)),
                     'cv_margin_to_full_optimum_min_correct':int(totals[selected]-max(totals[oracle])),
                     'wrong_selection_references':[]}
                row.update(paired_stage_flags(selected,ftotal,full))
                if selected not in oracle:
                    for oi in oracle:
                        diff=cv[selected]-cv[oi]
                        qdiff=fq[selected]-fq[oi]
                        row['wrong_selection_references'].append({
                            'query_optimal_policy':POLICIES[oi],
                            'cv_total_margin_selected_minus_optimum':int(totals[selected]-totals[oi]),
                            'cv_fold_margins_selected_minus_optimum':diff.tolist(),
                            'validation_folds_positive_equal_negative':[int((diff>0).sum()),int((diff==0).sum()),int((diff<0).sum())],
                            'fold_query_margins_selected_minus_optimum':qdiff.tolist(),
                            'query_folds_positive_equal_negative':[int((qdiff>0).sum()),int((qdiff==0).sum()),int((qdiff<0).sum())]})
                original=next(r for r in old['all_settings'] if r['episode']==episode and r['B']==B)
                for pi,p in enumerate(POLICIES):assert abs(original['full_'+p]-full_acc[pi])<1e-10
                for k in ['selected_accuracy','selected_minus_weak_pp','selected_minus_ra_pp']:
                    assert abs(row[k]-original[k])<1e-10
                assert abs(row['regret_pp']-original['oracle_regret_pp'])<1e-10
                for pi,pj in itertools.combinations(range(3),2):
                    pair_rows.append({'encoder':model,'B':B,'episode':episode,'policy_left':POLICIES[pi],
                                      'policy_right':POLICIES[pj],'cv_correct_left_minus_right':int(totals[pi]-totals[pj]),
                                      'fold_query_correct_sum_left_minus_right':int(ftotal[pi]-ftotal[pj]),
                                      'full_query_correct_left_minus_right':int(full[pi]-full[pj])})
                if len(tied)>1:
                    tie_rows.append({'encoder':model,'B':B,'episode':episode,'original_choice':POLICIES[selected],
                                     'tied_cv_correct':int(totals.max()),'cv_total':B*10,
                                     'all_candidates':[{'policy':POLICIES[i],'full_query_accuracy':float(full_acc[i]),
                                                        'fold_query_mean_accuracy':float(fold_acc[i]),
                                                        'regret_pp':float(full_acc.max()-full_acc[i]),
                                                        'cv_fold_correct':cv[i].tolist()} for i in tied]})
                rows.append(row);model_rows.append(row);all_rows.append(row)
            full=np.array([r['full_query_correct'] for r in rows]);cv=np.array([r['cv_correct'] for r in rows]);oracle=full.max(1)
            fixed_mean=full.mean(0)*100/len(label);strongest=winners(full.sum(0))
            s={'fixed_mean_accuracy':dict(zip(POLICIES,fixed_mean.tolist())),
               'selected_mean_accuracy':float(np.mean([r['selected_accuracy'] for r in rows])),
               'selected_counts':{p:sum(r['selected']==p for r in rows) for p in POLICIES},
               'full_query_optimal_membership_counts':{p:sum(p in r['full_query_best_set'] for r in rows) for p in POLICIES},
               'unique_full_query_optimal_counts':{p:sum(r['full_query_best_set']==[p] for r in rows) for p in POLICIES},
               'full_query_best_set_counts':dict(collections.Counter('|'.join(r['full_query_best_set']) for r in rows)),
               'expost_best_constant_policy':[POLICIES[i] for i in strongest],
               'oracle_mean_accuracy':float(oracle.mean()*100/len(label)),
               'oracle_space_vs_fixed':{p:count_stats((oracle-full[:,i])*100/len(label)) for i,p in enumerate(POLICIES)},
               'oracle_space_vs_expost_best_constant_mean_pp':float((oracle.mean()-full.mean(0).max())*100/len(label)),
               'distributions':{k:count_stats([r[k] for r in rows]) for k in ['selected_minus_weak_pp','selected_minus_ra_pp','ra_minus_weak_pp','regret_pp','cv_margin_to_full_optimum_min_correct']},
               'wrong_full_count':sum(r['selected_wrong_full'] for r in rows),
               'wrong_fold_count':sum(r['selected_wrong_fold'] for r in rows),
               'fold_choice_wrong_full_count':sum(r['fold_choice_wrong_full'] for r in rows),
               'full_wrong_and_fold_correct':sum(r['selected_wrong_full'] and not r['selected_wrong_fold'] for r in rows),
               'full_wrong_and_fold_wrong':sum(r['selected_wrong_full'] and r['selected_wrong_fold'] for r in rows),
               'full_wrong_and_same_pair_fold_inferior':sum(r['selected_wrong_full'] and r['fold_inferior_to_any_full_optimum'] for r in rows),
               'full_wrong_without_same_pair_fold_inferiority':sum(r['selected_wrong_full'] and not r['fold_inferior_to_any_full_optimum'] for r in rows),
               'regret_given_wrong':count_stats([r['regret_pp'] for r in rows if r['selected_wrong_full']]),
               'regret_by_selected_policy':{p:{'selected_episodes':[r['episode'] for r in rows if r['selected']==p],
                    'wrong_count':sum(r['selected']==p and r['selected_wrong_full'] for r in rows),
                    'sum_regret_pp':float(sum(r['regret_pp'] for r in rows if r['selected']==p)),
                    'contribution_to_mean_regret_pp':float(sum(r['regret_pp'] for r in rows if r['selected']==p)/12)} for p in POLICIES},
               'cv_tie_sensitivity':tie_enumeration(cv,full,len(label))}
            for pi,p in enumerate(POLICIES):
                lhs=s['selected_mean_accuracy']-s['fixed_mean_accuracy'][p]
                rhs=s['oracle_space_vs_fixed'][p]['mean']-s['distributions']['regret_pp']['mean']
                assert abs(lhs-rhs)<1e-10
            summaries[f'{model}:B{B}']=s
    assert len(all_rows)==48 and len(fold_rows)==576 and len(pair_rows)==144
    assert inputs['dinov2_small']['IDENTITIES.json']['sha256']==inputs['deit_tiny']['IDENTITIES.json']['sha256']
    assert inputs['dinov2_small']['QUERY_IDENTITIES.json']['sha256']==inputs['deit_tiny']['QUERY_IDENTITIES.json']['sha256']
    output={'status':'COMPLETE_SAVED_PREDICTION_ANALYSIS','utc':datetime.now(timezone.utc).isoformat(),
            'seconds':time.perf_counter()-started,'policy_order':POLICIES,'input_hashes':inputs,
            'all_48_settings':all_rows,'summaries':summaries,'all_cv_tied_settings':tie_rows,
            'new_model_forwards':0,'new_head_fits':0,'new_CU':0,'archived_main_results_matched':True,
            'inference_scope':'Descriptive conditional on shared query and overlapping supports; no confidence intervals or independent-trial claim.'}
    (ROOT/'SYNTHESIS.json').write_text(json.dumps(output,indent=2)+'\n')
    for name, table in [('FOLD_CORRECT_COUNTS.csv',fold_rows),('PAIRWISE_RANK_COUNTS.csv',pair_rows)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    compact=[]
    for r in all_rows:
        z={k:v for k,v in r.items() if not isinstance(v,(list,dict))}
        for i,p in enumerate(POLICIES):
            z['cv_'+p]=r['cv_correct'][i];z['full_'+p]=r['full_query_accuracy'][i]
            z['fold_query_'+p]=r['fold_query_mean_accuracy'][i]
        compact.append(z)
    with (ROOT/'ALL_EPISODES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
    print(json.dumps({k:{'fixed':v['fixed_mean_accuracy'],'choice':v['selected_counts'],
                        'oracle_space_best_constant':v['oracle_space_vs_expost_best_constant_mean_pp'],
                        'regret_mean':v['distributions']['regret_pp']['mean'],
                        'regret_given_wrong':v['regret_given_wrong'],'regret_by_policy':v['regret_by_selected_policy'],
                        'tie_bounds':v['cv_tie_sensitivity']['bounds'],
                        'wrong_full_fold_correct':v['full_wrong_and_fold_correct'],
                        'wrong_full_fold_wrong':v['full_wrong_and_fold_wrong']}
                      for k,v in summaries.items()},indent=2))

if __name__=='__main__':main()
