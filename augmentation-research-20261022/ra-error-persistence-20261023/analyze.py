"""Deterministic cached-prediction decomposition, single CPU, no torch/model/image IO."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import time,signal
START=time.perf_counter()
def cap(*args):raise TimeoutError('60-second numeric budget')
signal.signal(signal.SIGALRM,cap);signal.alarm(60)
import csv,hashlib,json,resource,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PILOT=ROOT/'augmentation-research-20261022/pilot-20261022'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,data):
    with (HERE/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(name,rows):
    with (HERE/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def stats(values):
    a=np.asarray(values,dtype=float);n=len(a)
    if not n:return {'N':0,'mean':None,'median':None,'min':None,'max':None,'positive':0,'negative':0,'zero':0,'leave_one_out_mean_min':None,'leave_one_out_mean_max':None}
    loo=(a.sum()-a)/(n-1) if n>1 else None
    return {'N':n,'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max()),'positive':int((a>1e-12).sum()),'negative':int((a< -1e-12).sum()),'zero':int((abs(a)<=1e-12).sum()),'leave_one_out_mean_min':None if loo is None else float(loo.min()),'leave_one_out_mean_max':None if loo is None else float(loo.max())}
def groups(w,r):
    return {'both_correct':w['correct']&r['correct'],'repaired':~w['correct']&r['correct'],'regressed':w['correct']&~r['correct'],'both_wrong':~w['correct']&~r['correct']}
def subset(w,r,mask):
    n=int(mask.sum());d=r['nll']-w['nll'];dp=r['pmax']-w['pmax']
    return {'N':n,'NLL_delta':stats(d[mask]),'NLL_contribution_all589':float(d[mask].sum()/589),'weak_NLL_mean':None if not n else float(w['nll'][mask].mean()),'ra_NLL_mean':None if not n else float(r['nll'][mask].mean()),'pmax_delta':stats(dp[mask]),'ptrue_delta':stats((r['ptrue']-w['ptrue'])[mask]),'margin_delta':stats((r['margin']-w['margin'])[mask]),'same_prediction_count':int(((w['pred']==r['pred'])&mask).sum()),'final_correctness_counts':{k:int((m&mask).sum()) for k,m in groups(w,r).items()}}
def direction(s):
    d=s['NLL_delta']['mean'];c=s['pmax_delta']['mean'];loo_d=s['NLL_delta']['leave_one_out_mean_min'];loo_c=s['pmax_delta']['leave_one_out_mean_min']
    return {'nonempty':s['N']>0,'mean_NLL_increases':d is not None and d>1e-12,'mean_max_confidence_increases':c is not None and c>1e-12,'both_directional_predictions_hold':d is not None and c is not None and d>1e-12 and c>1e-12,'both_signs_survive_every_one_photo_deletion':loo_d is not None and loo_c is not None and loo_d>1e-12 and loo_c>1e-12}
def main():
    cfg=json.loads((HERE/'protocol.json').read_text());lock=json.loads((HERE/'protocol-lock.json').read_text())
    assert sha(HERE/'protocol.json')==lock['protocol_sha256']
    assert sha(HERE/'analyze.py')==json.loads((HERE/'code-lock.json').read_text())['sha256']
    for p,h in cfg['input_and_preservation_sha256'].items():assert sha(ROOT/p)==h,p
    manifest=json.loads((PILOT/'fit-manifest.json').read_text());class_names=manifest['class_names']
    dev={r['source_id']:r['class_id'] for r in manifest['rows'] if r['candidate_partition']=='development'}
    assert len(dev)==589 and len(class_names)==37
    ids=np.array(sorted(dev));labels=np.array([dev[s] for s in ids]);N=len(ids);idx=np.arange(N)
    caches={};receipt=[]
    for cell in ['front-weak','front-ra','back-weak','back-ra']:
        for epoch in cfg['epochs']:
            path=PILOT/'results'/cell/f'development-epoch{epoch:03d}.npz'
            with np.load(path,allow_pickle=False) as arr:
                src=arr['source_ids'].astype(str);assert len(src)==len(set(src))==N and set(src)==set(ids)
                order=np.argsort(src);y=arr['labels'][order];z=arr['logits'][order].astype(np.float64);saved_pred=arr['predictions'][order]
            assert z.shape==(589,37) and np.array_equal(y,labels) and np.isfinite(z).all()
            shifted=z-z.max(1,keepdims=True);logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True));p=np.exp(logp);pred=np.argmax(z,1)
            assert np.array_equal(saved_pred,pred)
            competitors=z.copy();competitors[idx,labels]=-np.inf
            caches[cell,epoch]={'z':z,'pred':pred,'correct':pred==labels,'nll':-logp[idx,labels],'ptrue':p[idx,labels],'pmax':p.max(1),'margin':z[idx,labels]-competitors.max(1)}
            receipt.append({'cell':cell,'epoch':epoch,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shape':list(z.shape),'labels_and_ids_match_dev_manifest':True,'saved_predictions_match_argmax':True})
    initial=caches['front-weak',0]['z']
    init_max=max(float(np.max(abs(q['z']-initial))) for (cell,e),q in caches.items() if e==0)
    assert init_max<=1e-5
    old=json.loads((PILOT/'analysis/summary.json').read_text())
    for cell in ['front-weak','front-ra','back-weak','back-ra']:
        q=caches[cell,30];assert int((~q['correct']).sum())==old['cells'][cell]['errors']
        assert abs(float(q['nll'].mean())-old['cells'][cell]['NLL'])<1e-12
    per_photo=[];per_class=[];confusion_rows=[];pair_results={};closure_max=0.
    for layout in ['front','back']:
        by_epoch={};group_masks={}
        for epoch in cfg['epochs']:
            w=caches[layout+'-weak',epoch];r=caches[layout+'-ra',epoch];gg=groups(w,r);group_masks[epoch]=gg
            assert np.all(sum(m.astype(int) for m in gg.values())==1)
            d=r['nll']-w['nll'];summaries={k:subset(w,r,m) for k,m in gg.items()}
            summed=sum(s['NLL_contribution_all589'] for s in summaries.values());expected=float(r['nll'].mean()-w['nll'].mean());error=abs(summed-expected);closure_max=max(closure_max,error);assert error<1e-12
            positive=float(d[d>0].sum()/N);negative=float(d[d<0].sum()/N);assert abs(positive+negative-expected)<1e-12
            error_change=int((~r['correct']).sum()-(~w['correct']).sum());assert error_change==int(gg['regressed'].sum()-gg['repaired'].sum())
            summary={'weak_errors':int((~w['correct']).sum()),'ra_errors':int((~r['correct']).sum()),'error_count_change_RA_minus_weak':error_change,'weak_NLL':float(w['nll'].mean()),'ra_NLL':float(r['nll'].mean()),'NLL_delta':expected,'positive_NLL_budget':positive,'negative_NLL_budget':negative,'all_photo':subset(w,r,np.ones(N,dtype=bool)),'groups':summaries}
            if epoch==0:assert abs(expected)<1e-12 and error_change==0
            by_epoch[str(epoch)]=summary
            for i,sid in enumerate(ids):
                group=next(k for k,m in gg.items() if m[i])
                row={'pair':layout,'epoch':epoch,'source_id':sid,'label':int(labels[i]),'class_name':class_names[str(labels[i])],'group':group,'weak_pred':int(w['pred'][i]),'ra_pred':int(r['pred'][i]),'weak_pred_name':class_names[str(w['pred'][i])],'ra_pred_name':class_names[str(r['pred'][i])]}
                for key in ['nll','ptrue','pmax','margin']:
                    row['weak_'+key]=float(w[key][i]);row['ra_'+key]=float(r[key][i]);row['delta_'+key]=float(r[key][i]-w[key][i])
                per_photo.append(row)
        w=caches[layout+'-weak',30];r=caches[layout+'-ra',30];w20=caches[layout+'-weak',20];r20=caches[layout+'-ra',20]
        frozen={name:subset(w,r,mask) for name,mask in group_masks[20].items()}
        samewrong=(~w20['correct'])&(w20['pred']==r20['pred'])&(w20['pred']==w['pred'])&(w20['pred']==r['pred'])
        strict=subset(w,r,samewrong)
        persist={}
        for final_group in ['repaired','regressed','both_wrong']:
            final_mask=group_masks[30][final_group]
            persist[final_group]={'final_count':int(final_mask.sum()),'count_already_same_group_by_epoch':{str(e):int((final_mask&group_masks[e][final_group]).sum()) for e in cfg['epochs']},'same_group_all_5_10_20_30':int(np.logical_and.reduce([group_masks[e][final_group] for e in [5,10,20,30]]).sum())}
        both_final=by_epoch['30']['groups']['both_wrong'];anchored=frozen['both_wrong']
        pair_results[layout]={'by_epoch':by_epoch,'anchor20_groups_evaluated_at30':frozen,'strict_same_wrong_label_all_20_30':strict,'endpoint_membership_persistence':persist,'primary_direction':direction(both_final),'anchor20_direction':direction(anchored),'strict_same_wrong_direction':direction(strict),'interpretation':'Conditional descriptions of fixed trained predictions; no position interaction, independent seeds, pixel perturbations or calibration claim.'}
        for c in range(37):
            mask=labels==c;ss=subset(w,r,mask)
            per_class.append({'pair':layout,'class_id':c,'class_name':class_names[str(c)],'N':int(mask.sum()),'weak_errors':int((~w['correct'][mask]).sum()),'ra_errors':int((~r['correct'][mask]).sum()),'NLL_delta':ss['NLL_delta']['mean'],'NLL_contribution_all589':ss['NLL_contribution_all589'],'repaired':ss['final_correctness_counts']['repaired'],'regressed':ss['final_correctness_counts']['regressed'],'both_wrong':ss['final_correctness_counts']['both_wrong'],'weak_false_positives':int(((w['pred']==c)&~mask).sum()),'ra_false_positives':int(((r['pred']==c)&~mask).sum())})
            for pred in range(37):
                a=int((mask&(w['pred']==pred)).sum());b=int((mask&(r['pred']==pred)).sum())
                confusion_rows.append({'pair':layout,'true_class':c,'true_name':class_names[str(c)],'pred_class':pred,'pred_name':class_names[str(pred)],'weak_count':a,'ra_count':b,'delta_RA_minus_weak':b-a})
    final={'status':'completed_exploratory_cache_analysis','at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':lock['protocol_sha256'],'primary_pair':'front','N':N,'classes':37,'pairs':pair_results,'primary_explanation_status':'necessary_direction_survives_but_not_causal_or_novelty_confirmation' if pair_results['front']['primary_direction']['both_directional_predictions_hold'] else 'necessary_direction_fails_for_primary_pair_do_not_rescue_with_other_pair_epoch_or_metric','decomposition_max_abs_closure_error':closure_max,'initial_logits_max_abs_difference':init_max,'all_input_hashes_unchanged':all(sha(ROOT/p)==h for p,h in cfg['input_and_preservation_sha256'].items()),'resources':{'wall_seconds_including_imports_before_write':time.perf_counter()-START,'peak_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'python':sys.version,'numpy':np.__version__,'cpu_thread_env':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},'numeric_invocation':1,'training_runs':0,'model_forwards':0,'gpu_calls':0,'images_read':0,'new_random_draws':0},'no_new_training_authorized_by_result':True}
    assert final['all_input_hashes_unchanged']
    table('per-photo.csv',per_photo);table('per-class.csv',per_class);table('confusion.csv',confusion_rows);dump('input-receipt.json',receipt);dump('result.json',final)
    print(json.dumps({'status':final['primary_explanation_status'],'pairs':{k:{'endpoint_groups':q['by_epoch']['30']['groups'],'frozen20_both_wrong_at30':q['anchor20_groups_evaluated_at30']['both_wrong'],'strict_same_wrong':q['strict_same_wrong_label_all_20_30'],'directions':q['primary_direction'],'anchor_directions':q['anchor20_direction'],'persistence':q['endpoint_membership_persistence']} for k,q in pair_results.items()},'resources':final['resources'],'closure_error':closure_max},ensure_ascii=False,indent=2));signal.alarm(0)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if not (HERE/'failure.json').exists():dump('failure.json',{'type':type(e).__name__,'message':str(e),'wall_seconds':time.perf_counter()-START})
        raise
