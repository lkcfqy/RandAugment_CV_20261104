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
rows=load(OUT/'selection.json');assert len(rows)==16
pre=load(OUT/'preflight-result.json');con=load(OUT/'continuation-result.json')
pw=load(OUT/'preflight-watchdog.json');cw=load(OUT/'continuation-watchdog.json')
assert pre['completed_forwards']==8 and con['completed_forwards']==56
assert all(w['exit_code']==0 and w['stop_reason'] is None for w in (pw,cw))
prep=load(OUT/'prepare-watchdog.json')
assert prep['exit_code']==0 and prep['stop_reason'] is None
combined=prep['child_inclusive_wall_seconds']+pw['child_inclusive_wall_seconds']+cw['child_inclusive_wall_seconds']
assert combined<30 and prep['child_inclusive_wall_seconds']+pw['child_inclusive_wall_seconds']<10
assert pre['peak_RSS_bytes']<4294967296 and con['peak_RSS_bytes']<4294967296
assert all(r['device']=='cpu' and r['intraop_threads']==r['interop_threads']==1 and r['nice']>=10 for r in (pre,con))
assert pre['frozen_sha256_before']==pre['frozen_sha256_after']==con['frozen_sha256_before']==con['frozen_sha256_after']
for phase,expected in [('preflight',8),('continuation',56)]:
    events=[json.loads(s) for s in (OUT/(phase+'-events.jsonl')).read_text().splitlines()]
    for e in ('forward_start','forward_saved'):assert sum(x['event']==e for x in events)==expected
    assert events[-1]['event']=='complete'
from PIL import Image
records={};receipts={};images={};maxerr=0.;checks=0;flat=[];pairs=[];traces=[]
metric_names=['p_true','p_fixed_rival','fixed_rival_margin','p_dynamic_rival','dynamic_rival_margin','max_confidence','nll']
files=list((OUT/'forward-records').glob('*.json'));assert len(files)==64
for p in files:
    r=load(p);key=(r['source_id'],r['arm'],r['view']);assert key not in records;records[key]=r;receipts[str(p.relative_to(OUT))]=sha(p)
assert set(records)=={(r['source_id'],a,v) for r in rows for a in ('front-weak','front-ra') for v in ('weak','ra')}
manifest=load(ROOT/'augmentation-research-20261022/pilot-20261022/fit-manifest.json')
class_names={r['class_id']:r['source_id'].rsplit('_',1)[0] for r in manifest['rows']}
assert len(class_names)==37
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
            flat.append(dict(source_id=sid,group=row['group'],true_class=class_names[y],checkpoint=arm,view=view,fixed_rival_class=class_names[j],dynamic_rival_class=class_names[dyn],prediction_class=class_names[pred],prediction_correct=pred==y,**{k:m[k] for k in metric_names}))
        weak=records[(sid,arm,'weak')];aug=records[(sid,arm,'ra')];mw=weak['metrics'];ma=aug['metrics']
        item=dict(source_id=sid,group=row['group'],true_class=class_names[y],checkpoint=arm,fixed_rival_class=class_names[j],weak_prediction=class_names[mw['prediction']],ra_prediction=class_names[ma['prediction']],weak_correct=mw['correct'],ra_correct=ma['correct'],prediction_changed=mw['prediction']!=ma['prediction'],dynamic_rival_changed=mw['dynamic_rival']!=ma['dynamic_rival'],weak_dynamic_rival=class_names[mw['dynamic_rival']],ra_dynamic_rival=class_names[ma['dynamic_rival']],own_weak_fixed_rival=class_names[ownj],pixel_MAE_0_255=mae,pixel_changed_fraction=changed,operations=json.dumps(pair['operations'],ensure_ascii=False,sort_keys=True))
        for k in metric_names:item.update({k+'_weak':mw[k],k+'_ra':ma[k],'delta_'+k:ma[k]-mw[k]})
        item['delta_own_weak_fixed_rival_margin']=aug['own_weak_fixed_rival_margin']-weak['own_weak_fixed_rival_margin']
        pairs.append(item)
    guard()

# Reconstruct the preselected class-matched cohorts from the prior clean cache table.
oldrows=list(csv.DictReader((ROOT/'augmentation-research-20261022/ra-error-persistence-20261023/per-photo.csv').open()))
e20=[r for r in oldrows if r['pair']=='front' and r['epoch']=='20'];assert len(e20)==589
wr=[r for r in e20 if r['group']=='both_wrong' and r['weak_pred']==r['ra_pred']];co=[r for r in e20 if r['group']=='both_correct']
cl=sorted({int(r['label']) for r in wr}&{int(r['label']) for r in co})[:8]
expected=[min(r['source_id'] for r in pool if int(r['label'])==c) for c in cl for pool in (wr,co)]
assert [r['source_id'] for r in rows]==expected
base=load(OUT/'cached-clean-baseline.json')['rows'];assert len(base)==64
baseline={(r['source_id'],r['arm'],r['epoch']):r for r in base};assert len(baseline)==64
# Check actual original NPZ values; no forward or new views.
import numpy as np
for arm in ('front-weak','front-ra'):
    for epoch in (20,30):
        path=ROOT/('augmentation-research-20261022/pilot-20261022/results/'+arm+'/development-epoch%03d.npz'%epoch)
        with np.load(path,allow_pickle=False) as data:
            ids=list(data['source_ids'])
            for row in rows:
                b=baseline[(row['source_id'],arm,epoch)];idx=ids.index(row['source_id'])
                assert b['logits']==data['logits'][idx].tolist() and b['class_id']==int(data['labels'][idx])
                z=b['logits'];y=b['class_id'];pred=max(range(37),key=lambda k:z[k])
                assert pred==b['metrics']['prediction']
                with localcontext() as ctx:
                    ctx.prec=60;zz=[Decimal.from_float(x) for x in z];mx=max(zz);ee=[(x-mx).exp() for x in zz];den=sum(ee)
                    for key,val in [('p_true',float(ee[y]/den)),('max_confidence',float(ee[pred]/den)),('nll',float(mx-zz[y]+den.ln()))]:
                        err=abs(b['metrics'][key]-val);maxerr=max(maxerr,err);assert err<1e-12;checks+=1
                old=next(r for r in oldrows if r['pair']=='front' and int(r['epoch'])==epoch and r['source_id']==row['source_id'])
                prefix='weak' if arm=='front-weak' else 'ra'
                assert int(old[prefix+'_pred'])==pred
                assert abs(float(old[prefix+'_ptrue'])-b['metrics']['p_true'])<1e-12
                assert abs(float(old[prefix+'_pmax'])-b['metrics']['max_confidence'])<1e-12
        guard()
for rel,h in load(OUT/'trajectory-lock.json')['sha256'].items():assert sha(OUT/rel)==h
assert sha(OUT/'visual-review-before-inference.json')==load(OUT/'visual-review-lock.json')['sha256']
assert (OUT/'visual-review-before-inference.json').stat().st_mtime_ns<min(p.stat().st_mtime_ns for p in files)
assert (OUT/'trajectory-lock.json').stat().st_mtime_ns<min(p.stat().st_mtime_ns for p in files)
linked=[]
for row in rows:
    sid=row['source_id'];rec=dict(source_id=sid,group=row['group'],class_name=class_names[row['class_id']])
    for arm,short in [('front-weak','weak_trained'),('front-ra','ra_trained')]:
        b=baseline[(sid,arm,30)]['metrics'];w=records[(sid,arm,'weak')]['metrics'];a=records[(sid,arm,'ra')]['metrics']
        pair=next(x for x in pairs if x['source_id']==sid and x['checkpoint']==arm)
        for name,m in [('clean30',b),('weak_view',w),('ra_view',a)]:
            rec[short+'_'+name+'_prediction']=class_names[m['prediction']]
            rec[short+'_'+name+'_correct']=m['correct']
            rec[short+'_'+name+'_ptrue']=m['p_true']
            rec[short+'_'+name+'_pmax']=m['max_confidence']
        rec[short+'_clean_to_weak_prediction_changed']=b['prediction']!=w['prediction']
        rec[short+'_weak_to_ra_prediction_changed']=w['prediction']!=a['prediction']
        for key in ('p_true','fixed_rival_margin'):
            rec[short+'_delta_'+key]=pair['delta_'+key]
            rec[short+'_abs_delta_'+key]=abs(pair['delta_'+key])
    rec['both_clean30_wrong']=all(not baseline[(sid,a,30)]['metrics']['correct'] for a in ('front-weak','front-ra'))
    rec['all4_new_views_wrong']=all(not records[(sid,a,v)]['metrics']['correct'] for a in ('front-weak','front-ra') for v in ('weak','ra'))
    rec['all4_new_views_same_prediction']=len({records[(sid,a,v)]['metrics']['prediction'] for a in ('front-weak','front-ra') for v in ('weak','ra')})==1
    rec['ra_trained_abs_ptrue_response_smaller']=rec['ra_trained_abs_delta_p_true']<rec['weak_trained_abs_delta_p_true']
    rec['ra_trained_abs_margin_response_smaller']=rec['ra_trained_abs_delta_fixed_rival_margin']<rec['weak_trained_abs_delta_fixed_rival_margin']
    rec['predeclared_joint_counterexample']=row['group']=='epoch20_shared_wrong' and rec['both_clean30_wrong'] and rec['all4_new_views_wrong'] and rec['ra_trained_abs_ptrue_response_smaller'] and rec['ra_trained_abs_margin_response_smaller']
    linked.append(rec)
summary={}
for group in ('epoch20_shared_wrong','epoch20_both_correct'):
    result={}
    group_linked=[x for x in linked if x['group']==group]
    for arm,short in [('front-weak','weak_trained'),('front-ra','ra_trained')]:
        subset=[r for r in pairs if r['checkpoint']==arm and r['group']==group];assert len(subset)==8
        stats={}
        for key in metric_names+['own_weak_fixed_rival_margin']:
            vv=[r['delta_'+key] for r in subset]
            stats[key]={'mean_signed':statistics.fmean(vv),'median_signed':statistics.median(vv),'mean_absolute':statistics.fmean(abs(x) for x in vv),'negative':sum(x<0 for x in vv),'positive':sum(x>0 for x in vv),'zero':sum(x==0 for x in vv)}
        result[arm]={'n':8,'clean30_correct':sum(x[short+'_clean30_correct'] for x in group_linked),'weak_correct':sum(x['weak_correct'] for x in subset),'ra_correct':sum(x['ra_correct'] for x in subset),'clean_to_weak_prediction_changed':sum(x[short+'_clean_to_weak_prediction_changed'] for x in group_linked),'weak_to_ra_prediction_changed':sum(x['prediction_changed'] for x in subset),'weak_to_ra_transitions':dict(Counter(('correct' if x['weak_correct'] else 'wrong')+'_to_'+('correct' if x['ra_correct'] else 'wrong') for x in subset)),'paired_metrics':stats}
    result['joint_counts']={key:sum(x[key] for x in group_linked) for key in ['both_clean30_wrong','all4_new_views_wrong','all4_new_views_same_prediction','ra_trained_abs_ptrue_response_smaller','ra_trained_abs_margin_response_smaller','predeclared_joint_counterexample']}
    summary[group]=result
examples=[r['source_id'] for r in linked if r['predeclared_joint_counterexample']]
prior=protocol['prior_paired_training_stage']
resource_summary={'this_stage_prepare_seconds':prep['child_inclusive_wall_seconds'],'this_stage_preflight_seconds':pw['child_inclusive_wall_seconds'],'this_stage_continuation_seconds':cw['child_inclusive_wall_seconds'],'this_stage_total_child_inclusive_seconds':combined,'this_stage_forwards':64,'previous_stage_total_child_inclusive_seconds':prior['child_inclusive_wall_seconds'],'previous_stage_forwards':148,'paired_stages_cumulative_child_inclusive_seconds':prior['child_inclusive_wall_seconds']+combined,'paired_stages_cumulative_forwards':212,'scope':'Only these two paired CPU probe stages; earlier training/cache generation remains in original records and is not included or reset.','peak_RSS_bytes':max(pre['peak_RSS_bytes'],con['peak_RSS_bytes']),'forward_wall_seconds':con['forward_wall_seconds'],'forward_cpu_seconds':con['forward_cpu_seconds'],'new_training_runs':0,'GPU_forwards':0}
result={'at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'status':'completed_exploratory','n':16,'summary':summary,'predeclared_joint_counterexample_count':len(examples),'predeclared_joint_counterexample_ids':examples,'resource':resource_summary,'prior_art_candidate_decision':'Archive generic sensitivity/semantic-proxy candidate; existing work overlaps and finite diagnostic cannot establish new mechanism or algorithm. No further sample/seed/strength expansion.','limits':protocol['inference_limits'],'all_counterexamples_retained':True,'visual_review_before_new_scores':True}
csvout('per-view.csv',flat);csvout('paired.csv',pairs);csvout('operations.csv',traces);csvout('linked-clean-and-views.csv',linked)
dump(OUT/'result.json',result)
dump(OUT/'compute-ledger.json',resource_summary)
dump(OUT/'verification.json',{'status':'passed','formula':'Independent Decimal60 from original saved logits plus original NPZ equality','numeric_metric_checks':checks,'max_abs_error':maxerr,'tolerance':1e-12,'forward_count':64,'cached_baseline_records_verified':64,'saved_PNGs_verified':48,'trajectories_frozen_before_inference':True,'visual_review_record_before_inference':True,'preserved_input_and_old_files':len(protected),'postprocess_wall_seconds':time.perf_counter()-START,'postprocess_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'additional_model_forwards':0,'input_and_preservation_sha256':protected,'record_and_pair_sha256':receipts,'saved_image_sha256':images})
print(json.dumps({'resources':resource_summary,'counterexample_ids':examples,'group_summary':{g:{a:{k:v for k,v in d.items() if k!='paired_metrics'} for a,d in s.items()} for g,s in summary.items()},'linked':[{k:r[k] for k in ['source_id','group','weak_trained_clean30_prediction','weak_trained_weak_view_prediction','weak_trained_ra_view_prediction','ra_trained_clean30_prediction','ra_trained_weak_view_prediction','ra_trained_ra_view_prediction','ra_trained_abs_ptrue_response_smaller','ra_trained_abs_margin_response_smaller','predeclared_joint_counterexample']} for r in linked],'verification':{'checks':checks,'max_error':maxerr,'seconds':time.perf_counter()-START}},indent=2))
