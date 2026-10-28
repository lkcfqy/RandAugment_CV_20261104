"""Independent pure-Python scalar oracle for the fixed cache analysis."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import time,signal
START=time.perf_counter()
def cap(*args):raise TimeoutError('60-second verification budget')
signal.signal(signal.SIGALRM,cap);signal.alarm(60)
import csv,hashlib,json,math,resource,statistics
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PILOT=ROOT/'augmentation-research-20261022/pilot-20261022'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((HERE/'protocol.json').read_text());res=json.loads((HERE/'result.json').read_text())
assert sha(HERE/'protocol.json')==json.loads((HERE/'protocol-lock.json').read_text())['protocol_sha256']
assert sha(HERE/'verify.py')==json.loads((HERE/'verification-code-lock.json').read_text())['sha256']
assert sha(HERE/'analyze.py')==json.loads((HERE/'code-lock.json').read_text())['sha256']
for name,digest in cfg['input_and_preservation_sha256'].items():assert sha(ROOT/name)==digest
manifest=json.loads((PILOT/'fit-manifest.json').read_text());dev={r['source_id']:int(r['class_id']) for r in manifest['rows'] if r['candidate_partition']=='development'};ids=sorted(dev);N=len(ids);assert N==589
cache={}
for cell in ['front-weak','front-ra','back-weak','back-ra']:
    for e in cfg['epochs']:
        with np.load(PILOT/'results'/cell/f'development-epoch{e:03d}.npz',allow_pickle=False) as a:
            tuples=list(zip(a['source_ids'].tolist(),a['labels'].tolist(),a['logits'].tolist(),a['predictions'].tolist()))
        records={}
        for sid,y,z,pred_saved in tuples:
            assert sid in dev and y==dev[sid]
            maximum=max(z);weights=[math.exp(x-maximum) for x in z];norm=math.fsum(weights);prob=[x/norm for x in weights]
            pred=max(range(len(z)),key=z.__getitem__);assert pred==pred_saved
            records[sid]={'pred':pred,'correct':pred==y,'nll':math.log(norm)+maximum-z[y],'ptrue':prob[y],'pmax':max(prob),'margin':z[y]-max(x for j,x in enumerate(z) if j!=y)}
        assert sorted(records)==ids;cache[cell,e]=records
names=['both_correct','repaired','regressed','both_wrong']
def group(w,r):return names[(0 if w['correct'] else 1)+(0 if r['correct'] else 2)]
def stat(values):
    n=len(values)
    if n==0:return {'N':0,'mean':None,'median':None,'min':None,'max':None,'positive':0,'negative':0,'zero':0,'leave_one_out_mean_min':None,'leave_one_out_mean_max':None}
    total=math.fsum(values);loo=[math.fsum(values[:i]+values[i+1:])/(n-1) for i in range(n)] if n>1 else []
    return {'N':n,'mean':total/n,'median':statistics.median(values),'min':min(values),'max':max(values),'positive':sum(v>1e-12 for v in values),'negative':sum(v< -1e-12 for v in values),'zero':sum(abs(v)<=1e-12 for v in values),'leave_one_out_mean_min':min(loo) if loo else None,'leave_one_out_mean_max':max(loo) if loo else None}
def sub(w,r,members):
    n=len(members);d={key:[r[s][key]-w[s][key] for s in members] for key in ['nll','pmax','ptrue','margin']}
    return {'N':n,'NLL_delta':stat(d['nll']),'NLL_contribution_all589':math.fsum(d['nll'])/N,'weak_NLL_mean':math.fsum(w[s]['nll'] for s in members)/n if n else None,'ra_NLL_mean':math.fsum(r[s]['nll'] for s in members)/n if n else None,'pmax_delta':stat(d['pmax']),'ptrue_delta':stat(d['ptrue']),'margin_delta':stat(d['margin']),'same_prediction_count':sum(w[s]['pred']==r[s]['pred'] for s in members),'final_correctness_counts':{k:sum(group(w[s],r[s])==k for s in members) for k in names}}
checks=0;maxerror=0.
def compare(a,b,path=''):
    global checks,maxerror
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        error=abs(a-b);assert error<1e-12,(path,a,b,error);checks+=1;maxerror=max(maxerror,error)
    else:assert a==b,(path,a,b);checks+=1
with (HERE/'per-photo.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==5890 and len({(r['pair'],r['epoch'],r['source_id']) for r in rows})==5890
for row in rows:
    layout=row['pair'];e=int(row['epoch']);sid=row['source_id'];w=cache[layout+'-weak',e][sid];r=cache[layout+'-ra',e][sid]
    assert int(row['label'])==dev[sid] and row['group']==group(w,r)
    assert int(row['weak_pred'])==w['pred'] and int(row['ra_pred'])==r['pred']
    for key in ['nll','pmax','ptrue','margin']:
        compare(float(row['weak_'+key]),w[key]);compare(float(row['ra_'+key]),r[key]);compare(float(row['delta_'+key]),r[key]-w[key])
for layout in ['front','back']:
    q=res['pairs'][layout]
    for e in cfg['epochs']:
        w=cache[layout+'-weak',e];r=cache[layout+'-ra',e];actual=q['by_epoch'][str(e)]
        for k in names:compare(actual['groups'][k],sub(w,r,[s for s in ids if group(w[s],r[s])==k]),f'{layout}/{e}/{k}')
        compare(actual['all_photo'],sub(w,r,ids))
        compare(actual['weak_errors'],sum(not w[s]['correct'] for s in ids));compare(actual['ra_errors'],sum(not r[s]['correct'] for s in ids))
        compare(actual['error_count_change_RA_minus_weak'],actual['ra_errors']-actual['weak_errors'])
        compare(actual['weak_NLL'],math.fsum(w[s]['nll'] for s in ids)/N);compare(actual['ra_NLL'],math.fsum(r[s]['nll'] for s in ids)/N)
        delta=[r[s]['nll']-w[s]['nll'] for s in ids]
        compare(actual['NLL_delta'],math.fsum(delta)/N);compare(actual['positive_NLL_budget'],math.fsum(d for d in delta if d>0)/N);compare(actual['negative_NLL_budget'],math.fsum(d for d in delta if d<0)/N)
    w=cache[layout+'-weak',30];r=cache[layout+'-ra',30];w20=cache[layout+'-weak',20];r20=cache[layout+'-ra',20]
    for k in names:compare(q['anchor20_groups_evaluated_at30'][k],sub(w,r,[s for s in ids if group(w20[s],r20[s])==k]))
    strict=[s for s in ids if not w20[s]['correct'] and w20[s]['pred']==r20[s]['pred']==w[s]['pred']==r[s]['pred']]
    compare(q['strict_same_wrong_label_all_20_30'],sub(w,r,strict))
    for k in ['repaired','regressed','both_wrong']:
        final=[s for s in ids if group(w[s],r[s])==k];expected={'final_count':len(final),'count_already_same_group_by_epoch':{str(e):sum(group(cache[layout+'-weak',e][s],cache[layout+'-ra',e][s])==k for s in final) for e in cfg['epochs']},'same_group_all_5_10_20_30':sum(all(group(cache[layout+'-weak',e][s],cache[layout+'-ra',e][s])==k for e in [5,10,20,30]) for s in ids)}
        compare(q['endpoint_membership_persistence'][k],expected)
    for field,summary in [('primary_direction',q['by_epoch']['30']['groups']['both_wrong']),('anchor20_direction',q['anchor20_groups_evaluated_at30']['both_wrong']),('strict_same_wrong_direction',q['strict_same_wrong_label_all_20_30'])]:
        a=summary['NLL_delta'];b=summary['pmax_delta']
        expected={'nonempty':summary['N']>0,'mean_NLL_increases':a['mean'] is not None and a['mean']>1e-12,'mean_max_confidence_increases':b['mean'] is not None and b['mean']>1e-12,'both_directional_predictions_hold':a['mean'] is not None and b['mean'] is not None and a['mean']>1e-12 and b['mean']>1e-12,'both_signs_survive_every_one_photo_deletion':a['leave_one_out_mean_min'] is not None and b['leave_one_out_mean_min'] is not None and a['leave_one_out_mean_min']>1e-12 and b['leave_one_out_mean_min']>1e-12}
        compare(q[field],expected)
with (HERE/'per-class.csv').open() as f:classes=list(csv.DictReader(f))
for row in classes:
    layout=row['pair'];c=int(row['class_id']);w=cache[layout+'-weak',30];r=cache[layout+'-ra',30];members=[s for s in ids if dev[s]==c];ss=sub(w,r,members)
    compare(int(row['N']),len(members));compare(float(row['NLL_delta']),ss['NLL_delta']['mean']);compare(float(row['NLL_contribution_all589']),ss['NLL_contribution_all589'])
    for k in ['repaired','regressed','both_wrong']:compare(int(row[k]),ss['final_correctness_counts'][k])
    for arm,cc in [('weak',w),('ra',r)]:compare(int(row[arm+'_errors']),sum(not cc[s]['correct'] for s in members));compare(int(row[arm+'_false_positives']),sum(dev[s]!=c and cc[s]['pred']==c for s in ids))
with (HERE/'confusion.csv').open() as f:confusions=list(csv.DictReader(f))
assert len(confusions)==2*37*37
for row in confusions:
    layout=row['pair'];c=int(row['true_class']);p=int(row['pred_class'])
    for arm in ['weak','ra']:compare(int(row[arm+'_count']),sum(dev[s]==c and cache[layout+'-'+arm,30][s]['pred']==p for s in ids))
    compare(int(row['delta_RA_minus_weak']),int(row['ra_count'])-int(row['weak_count']))
assert res['primary_explanation_status'].startswith('necessary_direction_survives')==res['pairs']['front']['primary_direction']['both_directional_predictions_hold']
for name,digest in cfg['input_and_preservation_sha256'].items():assert sha(ROOT/name)==digest
out={'status':'passed','at_utc':datetime.now(timezone.utc).isoformat(),'method':'Scalar Python math.exp/fsum/log and statistics.median from20 raw NPZ inputs; independently derives correctness groups, frozen memberships, all moments/deletion bounds,37-class and37x37 confusion tables; no import of analyze.py. Same agent, not independent reviewer.','per_photo_rows':len(rows),'scalar_or_exact_checks':checks,'max_absolute_error':maxerror,'class_rows':len(classes),'confusion_rows':len(confusions),'preserved_input_files':len(cfg['input_and_preservation_sha256']),'resources':{'wall_seconds_including_imports_before_write':time.perf_counter()-START,'peak_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_threads':1,'training_runs':0,'model_forwards':0,'gpu_calls':0},'checked_output_hashes':{name:sha(HERE/name) for name in ['result.json','per-photo.csv','per-class.csv','confusion.csv','analyze.py','verify.py']}}
with (HERE/'verification.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(out,ensure_ascii=False,indent=2));signal.alarm(0)
