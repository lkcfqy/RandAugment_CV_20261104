"""Predeclared CPU analysis of saved training-only CLS features. No model import."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def rms(a):return float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))
def main():
    cfg=read(ROOT/'protocol.json');done=read(ROOT/'results/completed.json');lock=read(ROOT/'protocol-lock.json')
    assert done['status']=='all8_reset_one_step_cases_completed'
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
    assert done['protocol_sha256']==lock['protocol_sha256']
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in lock['code_sha256'].items())
    rows=[];baselines={};scales={};pooled={};all_resolved=True
    for layout in ['front','back']:
        with np.load(ROOT/'results'/layout/'initial-features.npz',allow_pickle=False) as d:
            f0=d['features'].astype(np.float64);rep=d['repeat_features'].astype(np.float64)
        floors=[max(rms(rep[j]-f0[j]),np.finfo(np.float32).eps*rms(f0[j])) for j in range(2)]
        baselines[layout]={'probe_RMS':[rms(f0[j]) for j in range(2)],'repeat_noise_RMS':[rms(rep[j]-f0[j]) for j in range(2)],'floors':floors}
        pooled[layout]=[];scales[layout]=[]
        for batch in range(4):
            folder=ROOT/'results'/layout/f'batch{batch}'
            with np.load(folder/'features.npz',allow_pickle=False) as d:f1=d['post_step_features'].astype(np.float64)
            receipt=read(folder/'receipt.json')
            joint_weight=float(np.sqrt(sum(r['CPU_FP64_frobenius']**2 for r in receipt['merged_QV_updates'])))
            s=[rms(f1[j]-f0[j]) for j in range(2)];sp=rms(f1-f0);resolved=[s[j]>10*floors[j] for j in range(2)]
            all_resolved=all_resolved and all(resolved)
            row={'layout':layout,'gradient_batch':batch,'probe0_S':s[0],'probe1_S':s[1],'pooled_S':sp,
                'relative_pooled_S':sp/rms(f0),'probe0_resolved':resolved[0],'probe1_resolved':resolved[1],
                'merged_QV_joint_Frobenius':joint_weight,'S_per_merged_QV_Frobenius':sp/joint_weight}
            rows.append(row);scales[layout].append(s);pooled[layout].append(sp)
    front=np.asarray(pooled['front']);back=np.asarray(pooled['back'])
    positive=bool(np.all(front>0) and np.all(back>0) and np.all(np.asarray(scales['front'])>0) and np.all(np.asarray(scales['back'])>0))
    ratios=front/back if positive else None
    probe_ratios=np.asarray(scales['front'])/np.asarray(scales['back']) if positive else None
    consistent=bool(all_resolved and positive and (np.all(probe_ratios>1) or np.all(probe_ratios<1)))
    variation={l:{'all4_pooled_S':pooled[l],'mean_S':float(np.mean(pooled[l])),
        'sample_SD_S':float(np.std(pooled[l],ddof=1)),
        'CV_S':float(np.std(pooled[l],ddof=1)/np.mean(pooled[l])) if np.mean(pooled[l])>0 else None,
        'sample_SD_logS':float(np.std(np.log(pooled[l]),ddof=1)) if min(pooled[l])>0 else None,
        'range_S':[float(np.min(pooled[l])),float(np.max(pooled[l]))]} for l in ['front','back']}
    summary={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_train_only_initial_one_step_scale_diagnostic',
        'rows':rows,'baseline_precision':baselines,'all8_case_two_probe_changes_resolved':all_resolved,
        'paired_front_over_back_ratios':ratios.tolist() if positive else None,'paired_per_probe_ratios':probe_ratios.tolist() if positive else None,
        'geometric_mean_front_over_back':float(np.exp(np.mean(np.log(ratios)))) if positive else None,
        'paired_ratio_range':[float(ratios.min()),float(ratios.max())] if positive else None,
        'batch_variation':variation,'consistent_direction_all4_batches_and_both_probes':consistent,
        'absolute_mean_log_ratio_divided_by_larger_within_layout_log_SD':float(abs(np.mean(np.log(ratios)))/max(v['sample_SD_logS'] for v in variation.values())) if positive and max(v['sample_SD_logS'] for v in variation.values())>0 else None,
        'statistical_scope':'Descriptive across4 fixed gradient batches at one common initialization. No p-value, cross-seed/population CI, animal independence or full-training inference.',
        'new_augmentation_interaction_or_novel_contribution_established':False,
        'original_pilot_engineering_gate_changed':False,'full_training_LR_search_test_reads_launched':False,
        'finish_decision':'Stop this diagnostic after8 reset steps. Any fair revised control is a new estimand/protocol, not retroactive repair or automatic continuation.'}
    with (ROOT/'analysis.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with (ROOT/'cases.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'status':summary['status'],'all_resolved':all_resolved,'front_over_back_ratios':summary['paired_front_over_back_ratios'],
        'geometric_mean_ratio':summary['geometric_mean_front_over_back'],'consistent':consistent,'batch_variation':variation},ensure_ascii=False))

if __name__=='__main__':main()
