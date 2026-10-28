"""Separate arithmetic verification; NumPy is used only to read the NPZ."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import csv,json,math,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
start=time.perf_counter()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text())
result=json.loads((HERE/'result.json').read_text())
records=list(csv.DictReader((HERE/'per-photo.csv').open()))
mapping=json.loads((HERE/'group-mapping.json').read_text())
with np.load(ROOT/protocol['primary_input'],allow_pickle=False) as d:
    logits=d['logits'].tolist();labels=d['labels'].tolist();ids=d['source_ids'].astype(str).tolist()
rows_by_id={r['source_id']:r for r in records}
assert len(rows_by_id)==589 and set(rows_by_id)==set(ids)
C=37;gamma=2/math.log(C);tol=1e-12
maximum={k:0. for k in ('score_original','score_oracle','score_fixed','score_rebatched','score_onehot','score_row_permuted_restored','score_class_renamed')}
checked=0

def probabilities(row):
    maximum=max(row);v=[math.exp(float(t)-maximum) for t in row];denom=math.fsum(v)
    return [q/denom for q in v]
def cosine_score(p,target):
    dot=math.fsum(a*b for a,b in zip(p,target))
    dp=max(math.sqrt(math.fsum(a*a for a in p)),1e-8)
    dt=max(math.sqrt(math.fsum(b*b for b in target)),1e-8)
    return (dot/(dp*dt))**gamma
for group in mapping:
    idx=group['indices'];by=[int(labels[i]) for i in idx];perm=group['row_permutation'];rename=group['old_to_new_class']
    assert sorted(perm)==list(range(len(idx))) and sorted(rename)==list(range(C))
    for i in idx:
        r=rows_by_id[ids[i]];p=probabilities(logits[i]);onehot=[float(k==labels[i]) for k in range(C)]
        correct=cosine_score(p,onehot)
        original=cosine_score(p,by) if len(by)==C else correct
        rowperm=cosine_score(p,[by[k] for k in perm]) if len(by)==C else correct
        rp=[0.]*C
        for old,new in enumerate(rename):rp[new]=p[old]
        renamed=cosine_score(rp,[rename[k] for k in by]) if len(by)==C else correct
        expected={'score_original':original,'score_oracle':correct,'score_fixed':correct,'score_rebatched':correct,'score_onehot':correct,'score_row_permuted_restored':rowperm,'score_class_renamed':renamed}
        assert int(r['label'])==labels[i] and int(r['group'])==group['group'] and int(r['batch_size'])==len(idx)
        for name,value in expected.items():
            delta=abs(float(r[name])-value);maximum[name]=max(maximum[name],delta);assert delta<=tol,(ids[i],name,delta);checked+=1
collisions=[r for r in records if int(r['batch_size'])==C]
for key,col in [('primary_original_vs_oracle','score_oracle'),('same_rows_rebatched','score_rebatched'),('row_permutation','score_row_permuted_restored'),('class_renaming','score_class_renamed')]:
    v=sorted(abs(float(r['score_original'])-float(r[col])) for r in collisions)
    point=.95*(len(v)-1);lo=int(point);q=v[lo]+(point-lo)*(v[min(lo+1,len(v)-1)]-v[lo])
    m=(v[(len(v)-1)//2]+v[len(v)//2])/2
    values={'count':len(v),'mean_abs':math.fsum(v)/len(v),'median_abs':m,'p95_abs':q,'max_abs':max(v),'count_above_tolerance':sum(x>tol for x in v)}
    for field,value in values.items():assert abs(value-result[key][field])<=tol,(key,field)
for relative,h in result['input_hashes'].items():assert sha(ROOT/relative)==h,relative
for f,h in json.loads((HERE/'code-lock.json').read_text())['files'].items():assert sha(HERE/f)==h
assert sha(HERE/'protocol.json')==json.loads((HERE/'protocol-lock.json').read_text())['protocol_sha256']
receipt={'status':'passed','at_utc':datetime.now(timezone.utc).isoformat(),'method':'Pure-Python math.fsum cosine of explicit vectors; NumPy only loads NPZ. Separate implementation, same agent, not independent reviewer.','saved_values_checked':checked,'records':len(records),'maximum_absolute_differences':maximum,'aggregate_statistics_recomputed':True,'all_recorded_input_and_code_hashes_unchanged':True,'wall_seconds':time.perf_counter()-start,'source_files':{p.name:sha(p) for p in [HERE/'result.json',HERE/'per-photo.csv',HERE/'group-mapping.json',Path(__file__)]}}
with (HERE/'verification.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
