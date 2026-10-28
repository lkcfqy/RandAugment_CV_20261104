"""Independent Decimal recomputation from saved logits; no torch/model calls."""
import time
START=time.perf_counter()
import os,json,hashlib,math,csv,statistics,resource
from pathlib import Path
from decimal import Decimal,localcontext
from collections import Counter
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
nice=os.getpriority(os.PRIO_PROCESS,0)
if nice<10:os.nice(10-nice)
def guard():
    assert time.perf_counter()-START<15,'Postprocessing 15s budget exhausted'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4294967296

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvout(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

assert sha(OUT/'verify-and-analyze.py')==load(OUT/'analysis-code-lock.json')['sha256']
protocol=load(OUT/'protocol.json')
assert sha(OUT/'protocol.json')==load(OUT/'protocol-lock.json')['sha256']
assert sha(OUT/'selection.json')==protocol['selection_sha256']
for name,h in load(OUT/'code-lock.json')['sha256'].items():assert sha(OUT/name)==h
protected={}
for rel,h in protocol['input_and_preservation_sha256'].items():
    got=sha(ROOT/rel);assert got==h,rel;protected[rel]=got;guard()
rows=load(OUT/'selection.json');assert len(rows)==37
pre=load(OUT/'preflight-result.json');con=load(OUT/'continuation-result.json')
pw=load(OUT/'preflight-watchdog.json');cw=load(OUT/'continuation-watchdog.json')
assert pre['completed_forwards']==4 and con['completed_forwards']==144
assert all(w['exit_code']==0 and w['stop_reason'] is None for w in (pw,cw))
combined=pw['child_inclusive_wall_seconds']+cw['child_inclusive_wall_seconds']
assert combined<120 and pw['child_inclusive_wall_seconds']<30
assert pre['peak_RSS_bytes']<4294967296 and con['peak_RSS_bytes']<4294967296
assert all(r['device']=='cpu' and r['intraop_threads']==r['interop_threads']==1 and r['nice']>=10 for r in (pre,con))
assert pre['frozen_sha256_before']==pre['frozen_sha256_after']==con['frozen_sha256_before']==con['frozen_sha256_after']
for phase,expected in [('preflight',4),('continuation',144)]:
    events=[json.loads(s) for s in (OUT/(phase+'-events.jsonl')).read_text().splitlines()]
    for e in ('forward_start','forward_saved'):assert sum(x['event']==e for x in events)==expected
    assert events[-1]['event']=='complete'
from PIL import Image
records={};receipts={};images={};maxerr=0.;checks=0;flat=[];pairs=[];traces=[]
metric_names=['p_true','p_fixed_rival','fixed_rival_margin','p_dynamic_rival','dynamic_rival_margin','max_confidence','nll']
files=list((OUT/'forward-records').glob('*.json'));assert len(files)==148
for p in files:
    r=load(p);key=(r['source_id'],r['arm'],r['view']);assert key not in records;records[key]=r;receipts[str(p.relative_to(OUT))]=sha(p)
assert set(records)=={(r['source_id'],a,v) for r in rows for a in ('front-weak','front-ra') for v in ('weak','ra')}
class_names={r['class_id']:r['class_name'] for r in rows}
for row in rows:
    sid=row['source_id'];y=row['class_id'];pairpath=OUT/'views'/sid/'pair.json';pair=load(pairpath)
    receipts[str(pairpath.relative_to(OUT))]=sha(pairpath)
    assert pair['source']==row and pair['original_uninstrumented_RA_pixel_parity'] is True and len(pair['operations'])==2
    assert pair['N']==2 and pair['M']==9 and pair['epoch']==0 and pair['training_seed']==20261022
    assert sha(ROOT/row['image_path'])==row['image_sha256']
    pics={}
    for view in ('source','weak','ra'):
        p=pairpath.parent/(view+'.png');h=pair['image_hashes'][view]
        assert sha(p)==h['file_sha256'];images[str(p.relative_to(OUT))]=h['file_sha256']
        with Image.open(p) as im:
            assert im.mode=='RGB' and list(im.size)==h['size']
            assert hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()==h['decoded_RGB_sha256']
            pics[view]=im.tobytes()
        if view=='source':assert h['decoded_RGB_sha256']==row['decoded_RGB_sha256']
    # Integer byte arithmetic independently checks the recorded strength.
    aa=pics['weak'];bb=pics['ra'];assert len(aa)==len(bb)==224*224*3
    mae=sum(abs(a-b) for a,b in zip(aa,bb))/len(aa)
    changed=sum(aa[i:i+3]!=bb[i:i+3] for i in range(0,len(aa),3))/(224*224)
    assert abs(mae-pair['pixel_MAE_0_255'])<1e-12 and changed==pair['pixel_changed_fraction']
    for num,op in enumerate(pair['operations']):traces.append(dict(source_id=sid,class_id=y,slot=num,N=2,M=9,epoch=0,weak_seed=row['weak_seed'],ra_seed=row['ra_seed'],operation=op['operation'],signed_magnitude=op['signed_magnitude'],effective=json.dumps(op['effective'],sort_keys=True),pixel_MAE_0_255=mae,pixel_changed_fraction=changed))
    zw=records[(sid,'front-weak','weak')]['logits'];j=max([i for i in range(37) if i!=y],key=lambda i:zw[i])
    for arm in ('front-weak','front-ra'):
        ownz=records[(sid,arm,'weak')]['logits'];ownj=max([i for i in range(37) if i!=y],key=lambda i:ownz[i])
        for view in ('weak','ra'):
            rec=records[(sid,arm,view)];z=rec['logits'];m=rec['metrics']
            assert len(z)==37 and all(math.isfinite(v) for v in z) and rec['class_id']==y
            assert rec['input_tensor_sha256']==pair['tensor_sha256'][view]
            pred=sorted(range(37),key=lambda i:(-z[i],i))[0]
            dyn=sorted((i for i in range(37) if i!=y),key=lambda i:(-z[i],i))[0]
            assert m['fixed_rival']==j and m['dynamic_rival']==dyn and m['prediction']==pred and m['correct']==(pred==y)
            assert rec['own_weak_fixed_rival']==ownj
            with localcontext() as ctx:
                ctx.prec=60
                zz=[Decimal.from_float(v) for v in z];mx=max(zz);ee=[(v-mx).exp() for v in zz];den=sum(ee);prob=[v/den for v in ee]
                independent=dict(p_true=float(prob[y]),p_fixed_rival=float(prob[j]),fixed_rival_margin=float(zz[y]-zz[j]),p_dynamic_rival=float(prob[dyn]),dynamic_rival_margin=float(zz[y]-zz[dyn]),max_confidence=float(prob[pred]),nll=float(mx-zz[y]+den.ln()))
                for k,v in independent.items():
                    err=abs(v-m[k]);maxerr=max(maxerr,err);checks+=1;assert err<1e-12,(sid,arm,view,k,err)
                assert abs(rec['own_weak_fixed_rival_probability']-float(prob[ownj]))<1e-12
                assert abs(rec['own_weak_fixed_rival_margin']-float(zz[y]-zz[ownj]))<1e-12
                checks+=2
            flat.append(dict(source_id=sid,true_class=class_names[y],checkpoint=arm,view=view,fixed_rival_class=class_names[j],dynamic_rival_class=class_names[dyn],prediction_class=class_names[pred],prediction_correct=pred==y,**{k:m[k] for k in metric_names}))
        weak=records[(sid,arm,'weak')];aug=records[(sid,arm,'ra')];mw=weak['metrics'];ma=aug['metrics']
        item=dict(source_id=sid,true_class=class_names[y],checkpoint=arm,fixed_rival_class=class_names[j],weak_prediction=class_names[mw['prediction']],ra_prediction=class_names[ma['prediction']],weak_correct=mw['correct'],ra_correct=ma['correct'],prediction_changed=mw['prediction']!=ma['prediction'],dynamic_rival_changed=mw['dynamic_rival']!=ma['dynamic_rival'],weak_dynamic_rival=class_names[mw['dynamic_rival']],ra_dynamic_rival=class_names[ma['dynamic_rival']],own_weak_fixed_rival=class_names[ownj],pixel_MAE_0_255=mae,pixel_changed_fraction=changed,operations=json.dumps(pair['operations'],ensure_ascii=False,sort_keys=True))
        for k in metric_names:item.update({k+'_weak':mw[k],k+'_ra':ma[k],'delta_'+k:ma[k]-mw[k]})
        item['delta_own_weak_fixed_rival_margin']=aug['own_weak_fixed_rival_margin']-weak['own_weak_fixed_rival_margin']
        pairs.append(item)
    guard()
summary={}
for arm in ('front-weak','front-ra'):
    subset=[r for r in pairs if r['checkpoint']==arm]
    stats={}
    for k in metric_names+['own_weak_fixed_rival_margin']:
        values=[r['delta_'+k] for r in subset]
        stats['delta_'+k]=dict(mean=statistics.fmean(values),median=statistics.median(values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),positive=sum(v>0 for v in values),min=min(values),max=max(values))
    patterns=Counter(('correct' if r['weak_correct'] else 'wrong')+'_to_'+('correct' if r['ra_correct'] else 'wrong') for r in subset)
    summary[arm]=dict(n=37,paired_stats=stats,weak_correct=sum(r['weak_correct'] for r in subset),ra_correct=sum(r['ra_correct'] for r in subset),prediction_changed=sum(r['prediction_changed'] for r in subset),dynamic_rival_changed=sum(r['dynamic_rival_changed'] for r in subset),transitions=dict(patterns),
        changed_predictions=[{k:r[k] for k in ('source_id','true_class','weak_prediction','ra_prediction','weak_correct','ra_correct','delta_p_true','delta_fixed_rival_margin')} for r in subset if r['prediction_changed']],
        counterexamples_to_individual_true_probability_drop=[r['source_id'] for r in subset if r['delta_p_true']>=0],
        counterexamples_to_individual_fixed_margin_drop=[r['source_id'] for r in subset if r['delta_fixed_rival_margin']>=0])
probe=summary['front-weak']['paired_stats']
probe_pass=probe['delta_p_true']['mean']<0 and probe['delta_fixed_rival_margin']['mean']<0
paired_difference={k:statistics.fmean([next(r for r in pairs if r['source_id']==row['source_id'] and r['checkpoint']=='front-ra')['delta_'+k]-next(r for r in pairs if r['source_id']==row['source_id'] and r['checkpoint']=='front-weak')['delta_'+k] for row in rows]) for k in metric_names}
result=dict(scope='Exploratory fixed 37 training images, one original epoch0 draw, two existing front checkpoints; not independent validation or semantic certification',complete=True,unique_images=37,unique_forwards=148,primary_fixed_directional_probe_survives=probe_pass,summary=summary,descriptive_front_ra_minus_front_weak_paired_response=paired_difference,
    observed_operator_counts=dict(Counter(t['operation'] for t in traces)),resource=dict(preflight_child_inclusive_wall_seconds=pw['child_inclusive_wall_seconds'],continuation_child_inclusive_wall_seconds=cw['child_inclusive_wall_seconds'],combined_child_inclusive_wall_seconds=combined,peak_RSS_bytes=max(pre['peak_RSS_bytes'],con['peak_RSS_bytes']),combined_forward_wall_seconds=con['forward_wall_seconds'],combined_forward_cpu_seconds=con['forward_cpu_seconds'],training_runs=0,neural_GPU_forwards=0),
    interpretation_limits=['Scores alone cannot certify loss of breed semantics.','No manipulation of operation type or magnitude: no operator-causal attribution.','Training images and prior development exploration: not held-out preregistered confirmation.','Two checkpoints from one seed are not independent retraining replications.','Original back-pair dev confidence counterexample remains open and unchanged.'],
    previous_secondary_back_counterexample=dict(shared_wrong_n=32,mean_NLL_delta=0.03080015,median_NLL_delta=-0.04000993,max_confidence_delta=-0.0001478412,status='Retained original exploratory development result; not recomputed here'))
csvout('per-view.csv',flat);csvout('paired.csv',pairs);csvout('operations.csv',traces)
dump(OUT/'result.json',result)
dump(OUT/'verification.json',dict(status='passed',independent_formula='Decimal.from_float with precision60 exp/logsum, no torch calls',absolute_tolerance=1e-12,max_metric_absolute_error=maxerr,numeric_metric_checks=checks,verified_forward_records=148,verified_pairs=37,verified_saved_PNGs=111,protected_source_and_previous_files=len(protected),full_logits_retained=True,original_augmentation_pixel_parity_all=True,all_selected_training_whitelist=True,worker_forward_start_counts={'preflight':4,'continuation':144},combined_child_inclusive_wall_seconds=combined,postprocessing_wall_seconds=time.perf_counter()-START,postprocessing_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,additional_model_forwards=0,source_hashes=protected,forward_and_pair_sha256=receipts,image_sha256=images))
print(json.dumps({'primary_probe_survives':probe_pass,'resource':result['resource'],'summary':summary,'descriptive_front_ra_minus_front_weak_paired_response':paired_difference,'verification':{'max_metric_absolute_error':maxerr,'checks':checks,'seconds':time.perf_counter()-START}},indent=2))
