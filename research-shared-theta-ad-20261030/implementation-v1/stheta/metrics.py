"""Pure score metrics. Imported only for synthetic tests or future authorized scoring."""
import numpy as np
from .methods import baseline_names
from .protocol import METHOD_NAMES,BASELINE_NAMES,query_manifest_records

def auroc(labels,scores):
    y=np.asarray(labels,dtype=int);s=np.asarray(scores,dtype=float)
    assert np.isfinite(s).all() and set(np.unique(y))=={0,1}
    # Average ranks implement half credit for ties; stable ordering is incidental.
    order=np.argsort(s,kind='stable');rank=np.empty(len(s),float)
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:j+=1
        rank[order[i:j]]=(i+1+j)/2;i=j
    p=y.sum();n=len(y)-p
    return float((rank[y==1].sum()-p*(p+1)/2)/(p*n))

def threshold(scores):
    s=np.asarray(scores);assert s.shape==(64,) and np.isfinite(s).all()
    return float(np.sort(s)[61])

def positive(scores,tau):return np.asarray(scores)>tau

def validate_complete_input(scores_by_seed,query_records,split_manifest):
    def require(ok,message):
        if not ok:raise ValueError(message)
    require(isinstance(scores_by_seed,dict) and set(scores_by_seed)=={'11','22','33'},'Exactly seeds 11/22/33 required')
    require(split_manifest.get('status')=='ready','Frozen split manifest must be ready')
    require(len(split_manifest.get('calibration',[]))==64 and len(split_manifest.get('tests',[]))==962,'Expected manifest must contain cal64/test962')
    expected=query_manifest_records(split_manifest)
    require(isinstance(query_records,list) and len(query_records)==1026,'Exactly 1026 query records required')
    required={'query_id','path','kind','domain','label','pixelhash'}
    require(all(isinstance(r,dict) and set(r)==required for r in query_records),'Query record schema mismatch')
    require(all(type(r['label']) is int and r['label'] in (0,1) for r in query_records),'Labels must be integer 0/1')
    require(len({r['query_id'] for r in query_records})==1026,'Duplicate query IDs')
    require(len({r['path'] for r in query_records})==1026,'Duplicate query paths')
    require(len({r['pixelhash'] for r in query_records})==1026,'Duplicate query pixel identity')
    require(query_records==expected,'IDs, order, labels, paths or pixels differ from frozen manifest')
    require(all(r['kind']=='calibration' and r['domain']=='calibration' and r['label']==0 for r in query_records[:64]),'Calibration must be 64 normal items')
    tests=query_records[64:]
    require(all(r['kind']=='test' for r in tests),'Test split marker mismatch')
    for domain,label,count in [('same',0,230),('same',1,459),('illumination',0,75),('illumination',1,198)]:
        require(sum(r['domain']==domain and r['label']==label for r in tests)==count,'Domain/label count mismatch')
    for seed,methods in scores_by_seed.items():
        require(isinstance(methods,dict) and set(methods)==set(METHOD_NAMES),'Exactly all59 named methods required for seed '+seed)
        require(set(baseline_names(methods))==set(BASELINE_NAMES),'All51 external baselines required')
        for method,values in methods.items():
            a=np.asarray(values)
            require(a.shape==(1026,) and a.dtype.kind in 'fiu','Score shape/type mismatch for '+seed+'/'+method)
            require(bool(np.isfinite(a).all()),'Nonfinite scores for '+seed+'/'+method)
    return tests

def summarize(scores_by_seed,query_records,split_manifest):
    """Formal-only future output; no CI or selection of a replacement primary."""
    test_records=validate_complete_input(scores_by_seed,query_records,split_manifest)
    seed_metrics={};thresholds={}
    for seed,scores in scores_by_seed.items():
        seed_metrics[seed]={};thresholds[seed]={}
        for name,values in scores.items():
            values=np.asarray(values);assert values.shape==(1026,)
            tau=threshold(values[:64]);thresholds[seed][name]=tau
            metrics={}
            for domain in ('same','illumination'):
                ix=np.array([r['domain']==domain for r in test_records]);y=np.array([r['label'] for r in test_records])[ix];s=values[64:][ix]
                metrics[domain]={'AUROC':auroc(y,s),'TPR':float(positive(s[y==1],tau).mean()),'FPR':float(positive(s[y==0],tau).mean())}
            metrics['eqAUROC']=(metrics['same']['AUROC']+metrics['illumination']['AUROC'])/2
            seed_metrics[seed][name]=metrics
    seeds=list(seed_metrics);names=list(seed_metrics[seeds[0]]);baselines=baseline_names(names)
    average=lambda name,domain:float(np.mean([seed_metrics[s][name]['eqAUROC'] if domain=='eq' else seed_metrics[s][name][domain]['AUROC'] for s in seeds]))
    eq_best=max(sorted(baselines),key=lambda n:average(n,'eq'));il_best=max(sorted(baselines),key=lambda n:average(n,'illumination'))
    same_best=max(('identity','pool9'),key=lambda n:average(n,'same'))
    eq_delta=average('primary','eq')-average(eq_best,'eq')
    il_delta=average('primary','illumination')-average(il_best,'illumination')
    same_delta=average('primary','same')-average(same_best,'same')
    each=[seed_metrics[s]['primary']['eqAUROC']-seed_metrics[s][eq_best]['eqAUROC'] for s in seeds]
    shuffle_delta=average('primary','eq')-average('shuffle/9001','eq')
    gates={'eq_ge_02':eq_delta>=.02,'illumination_ge_03':il_delta>=.03,'same_drop_le_01':same_delta>=-.01,'each_seed_eq_strictly_positive':all(x>0 for x in each),'primary_shuffle9001_ge_01':shuffle_delta>=.01}
    shuffle_sensitivity={str(p):{'mean_eqAUROC':average(f'shuffle/{p}','eq'),'primary_minus_shuffle_mean_eq':average('primary','eq')-average(f'shuffle/{p}','eq'),'primary_gate':p==9001} for p in (9001,9002,9003,9004,9005)}
    return {'methods':seed_metrics,'thresholds':thresholds,'investment_baseline_envelope':{'eq':eq_best,'illumination':il_best,'same_identity_or_pool':same_best},'deltas':{'eq':eq_delta,'illumination':il_delta,'same':same_delta,'per_seed_eq':each,'shuffle9001_eq':shuffle_delta},'gates':gates,'continue':all(gates.values()),'shuffle_mapping_sensitivity':shuffle_sensitivity,'CI':None,'no_target_FPR_guarantee':True,'mechanism_cannot_rescue_failed_primary':True}
