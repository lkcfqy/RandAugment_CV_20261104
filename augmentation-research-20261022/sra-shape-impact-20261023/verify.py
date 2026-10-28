"""Separate arithmetic and RNG-trace verification; never imports impact.py or source AST."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import time,signal
START=time.perf_counter()
def limit(*args):raise TimeoutError('60-second verification cap')
signal.signal(signal.SIGALRM,limit);signal.alarm(60)
import collections,csv,hashlib,json,math,resource
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text())
CFG=read('protocol.json');RES=read('result.json')
assert sha(HERE/'protocol.json')==read('protocol-lock.json')['protocol_sha256']
assert sha(HERE/'verify.py')==read('verification-code-lock.json')['sha256']
for f,h in read('code-lock.json')['files'].items():assert sha(HERE/f)==h
for p,h in RES['input_hashes'].items():assert sha(ROOT/p)==h,p
with np.load(ROOT/CFG['input'],allow_pickle=False) as a:
    Z=a['logits'].tolist();Y=a['labels'].tolist();IDS=a['source_ids'].tolist()
groups=json.loads((ROOT/CFG['prior_group_mapping']).read_text())
with (HERE/'scores.csv').open() as f:scores=list(csv.DictReader(f))
with (HERE/'decisions.csv').open() as f:decisions=list(csv.DictReader(f))
assert len(scores)==589 and len(decisions)==1178
assert len({r['source_id'] for r in scores})==589
by_score={r['source_id']:r for r in scores}
by_dec={(r['source_id'],int(r['depth'])):r for r in decisions}
assert len(by_dec)==1178
ATOL=CFG['tolerances']['score_atol'];PTOL=CFG['tolerances']['parameter_atol']
max_score=0.;max_arg=0.;score_checks=0;argument_checks=0;rng_checks=0
ops=['ShearX','ShearY','TranslateX','TranslateY','Rotate','Brightness','Color','Sharpness','Contrast','Solarize','Posterize','Equalize','AutoContrast','Identity']
scale={'ShearX':.3,'ShearY':.3,'TranslateX':.45,'TranslateY':.45,'Rotate':30.,'Brightness':.9,'Color':.9,'Sharpness':.9,'Contrast':.9}
photo={'Brightness','Color','Sharpness','Contrast'}
rank_summary=[];all_deltas=[];collision_dec=[];affected=set();score_order=[]
for group in groups:
    ix=group['indices'];g=group['group'];B=len(ix);labels=[Y[i] for i in ix];collision=B==37
    target_norm=math.sqrt(math.fsum(v*v for v in labels))
    torch.manual_seed(CFG['seeds']['operation_trace_torch']+g)
    np.random.seed(CFG['seeds']['operation_trace_numpy']+g)
    group_scores=[]
    for i in ix:
        sid=IDS[i];score_order.append(sid);row=by_score[sid]
        assert int(row['label'])==Y[i] and int(row['group'])==g and int(row['B'])==B
        assert (row['collision']=='True')==collision
        values=Z[i];exps=[math.exp(v-max(values)) for v in values]
        den=math.fsum(exps);p=[v/den for v in exps]
        pnorm=math.sqrt(math.fsum(v*v for v in p))
        correct=(p[Y[i]]/pnorm)**(2/math.log(37))
        original=(math.fsum(a*b for a,b in zip(p,labels))/(pnorm*target_norm))**(2/math.log(37)) if collision else correct
        for key,expected in [('original',original),('patched',correct),('oracle',correct),('prior_adapter',correct),('one_hot',correct)]:
            error=abs(float(row[key])-expected);assert error<=ATOL,(sid,key,error)
            max_score=max(max_score,error);score_checks+=1
        group_scores.append(row)
        if collision:all_deltas.append(float(row['patched'])-float(row['original']))
        for depth in range(2):
            op=ops[int(torch.randint(0,len(ops),()))]
            probability=float(np.random.uniform(1.,1.));gate=float(np.random.uniform(0.,1.))<probability
            sign=-1. if op in scale and float(np.random.uniform())>=.5 else 1.
            dec=by_dec[(sid,depth)]
            assert dec['operation']==op and (dec['applied']=='True')==gate and float(dec['probability'])==probability
            assert int(dec['group'])==g and int(dec['B'])==B and (dec['collision']=='True')==collision
            rng_checks+=1;expected_parameters=[]
            for arm in ['original','patched','oracle']:
                magnitude=float(row[arm]);assert float(dec['magnitude_'+arm])==magnitude
                if op in scale:arg=(1. if op in photo else 0.)+sign*scale[op]*magnitude
                elif op=='Solarize':arg=256.-256.*magnitude
                elif op=='Posterize':arg=4.*magnitude
                else:arg=None
                expected_parameters.append(arg)
                if arg is None:assert dec['parameter_'+arm]==''
                else:
                    error=abs(float(dec['parameter_'+arm])-arg);assert error<=PTOL,(sid,depth,arm,error)
                    max_arg=max(max_arg,error);argument_checks+=1
            delta=None if expected_parameters[0] is None else expected_parameters[1]-expected_parameters[0]
            changed=delta is not None and abs(delta)>PTOL
            assert (dec['argument_changed']=='True')==changed
            if delta is None:assert dec['parameter_delta']==''
            else:assert abs(float(dec['parameter_delta'])-delta)<=PTOL
            if collision:
                collision_dec.append(dec)
                if changed:affected.add(sid)
            else:assert not changed
    ranks=[]
    for arm in ['original','patched']:
        ordered=sorted(group_scores,key=lambda r:(float(r[arm]),r['source_id']))
        ranking={r['source_id']:k for k,r in enumerate(ordered)}
        assert all(int(r['rank_'+arm])==ranking[r['source_id']] for r in group_scores)
        ranks.append(ranking)
    inversions=ties=0
    for a in range(B):
        for b in range(a+1,B):
            da=float(group_scores[a]['original'])-float(group_scores[b]['original'])
            db=float(group_scores[a]['patched'])-float(group_scores[b]['patched'])
            if da==0 or db==0:ties+=1
            elif (da<0)!=(db<0):inversions+=1
    k=math.ceil(B/4)
    top=[{sid for sid,rank in r.items() if rank>=B-k} for r in ranks]
    spearman=1-6*sum((ranks[0][sid]-ranks[1][sid])**2 for sid in ranks[0])/(B*(B*B-1))
    reported=next(r for r in RES['ranking_per_group'] if r['group']==g)
    assert reported['pair_inversions']==inversions and reported['pair_ties']==ties
    assert reported['top_quartile_overlap']==len(top[0]&top[1])
    assert abs(reported['spearman_id_tiebroken']-spearman)<=ATOL
    if collision:rank_summary.append((inversions,B*(B-1)//2,ties,len(top[0]&top[1]),k,spearman))
assert [r['source_id'] for r in scores]==score_order
assert [(r['source_id'],int(r['depth'])) for r in decisions]==[(sid,d) for sid in score_order for d in range(2)]
assert sum(r[0] for r in rank_summary)==RES['ranking']['pair_inversions']
assert sum(r[1] for r in rank_summary)==RES['ranking']['total_pairs']
assert sum(r[2] for r in rank_summary)==RES['ranking']['pair_ties']
assert sum(r[3] for r in rank_summary)==RES['ranking']['top_quartile_overlap']
assert sum(r[4] for r in rank_summary)==RES['ranking']['top_quartile_slots']
assert abs(sum(r[5] for r in rank_summary)/len(rank_summary)-RES['ranking']['mean_group_spearman'])<=ATOL

def quantile(values,q):
    v=sorted(values);position=(len(v)-1)*q;lo=math.floor(position);hi=math.ceil(position)
    return v[lo]+(position-lo)*(v[hi]-v[lo])
def checkstats(values,expected):
    v=[abs(x) for x in values];assert len(v)==expected['n']
    if not v:assert all(expected[k] is None for k in ['mean_abs','median_abs','p95_abs','max_abs']);return
    vals={'mean_abs':math.fsum(v)/len(v),'median_abs':quantile(v,.5),'p95_abs':quantile(v,.95),'max_abs':max(v)}
    for k,x in vals.items():assert abs(x-expected[k])<=PTOL
checkstats(all_deltas,RES['collision_score_delta'])
for op in ops:
    rows=[r for r in collision_dec if r['operation']==op];reported=RES['augmentation_decisions']['per_operator'][op]
    assert reported['selected']==reported['applied']==len(rows)
    assert reported['argument_changes']==sum(r['argument_changed']=='True' for r in rows)
    checkstats([float(r['parameter_delta']) for r in rows if r['parameter_delta']!=''],reported['parameter_delta'])
assert len(affected)==RES['augmentation_decisions']['collision_samples_with_argument_change']
assert sum(r['argument_changed']=='True' for r in collision_dec)==RES['augmentation_decisions']['collision_argument_changed_slots']
assert RES['augmentation_decisions']['selected_operator_differences']==RES['augmentation_decisions']['application_gate_differences']==0
# Independent closed-form schedule arithmetic, no BatchSampler reuse.
for name,r in RES['loader_schedules'].items():
    q=(r['N']+r['world']-1)//r['world'];size=r['per_rank_full_batch'];full,rem=divmod(q,size)
    expected=collections.Counter({(size+1)//2:full})
    if rem:expected[(rem+1)//2]+=1
    expected={str(k):v for k,v in expected.items() if v}
    assert expected==r['scoring_batch_histogram']
    assert r['per_rank_sampler_size']==q and r['sampler_padding_total']==q*r['world']-r['N']
    assert r['last_loader_batch']==(rem if rem else size)
    hits=expected.get(str(r['classes']),0) if r['SRA_actually_active'] else 0
    assert r['triggering_batches']==hits and r['triggering_score_samples']==hits*r['classes']
    if name.startswith(('official_','local_')):assert hits==0
for p,h in RES['input_hashes'].items():assert sha(ROOT/p)==h
out={'status':'passed','at_utc':datetime.now(timezone.utc).isoformat(),'method':'Separate pure-Python fsum/exp score formulas, independently written RNG and argument equations, ranking recomputation and closed-form batching; does not import impact.py, adapter or official AST. Same agent, not an external reviewer.','score_rows':len(scores),'scalar_score_checks':score_checks,'score_max_abs_error':max_score,'operation_gate_rng_checks':rng_checks,'scalar_operator_parameter_checks':argument_checks,'parameter_max_abs_error':max_arg,'ranking_groups':len(groups),'collision_samples_with_argument_change':len(affected),'noncollision_argument_changes':0,'real_default_trigger_batches':0,'old_audit_inputs_and_sources_unchanged':True,'files_checked':{p:sha(HERE/p) for p in ['scores.csv','decisions.csv','result.json','verify.py','verification-code-lock.json']},'resources':{'wall_seconds_including_imports_before_write':time.perf_counter()-START,'peak_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_threads':1,'gpu_calls':0,'training_runs':0}}
with (HERE/'verification.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(out,ensure_ascii=False,indent=2));signal.alarm(0)
