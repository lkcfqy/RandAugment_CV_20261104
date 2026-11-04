"""CPU-only analysis under PROTOCOL_ZH.md; reads frozen outputs after release."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from conformal_core import arms, softmax, scores_sorted, quantile, coverage_metrics
from decision_guards import tuning_eligibility, assess_control

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results';OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
TS=(.5,.75,1.,1.5,2.,3.,5.)
ARM=('00','10','01','11')
B=999
start=time.time()

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def key(method,t,arm):return f'{method}|{t:g}|{arm}'

def wilson(success,n,z=1.959963984540054):
    p=success/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return np.stack([c-h,c+h],axis=1).tolist()

state=json.loads((DATA/'run-state.json').read_text())
assert state['result']=='LOGITS_COMPLETE'
assert state['formal_steps']==25040 and state['formal_presentations']==3200000
assert state['benchmark_steps']==200 and state['inference_views']==205120
cache=json.loads((DATA/'cache-acceptance.json').read_text());assert cache['pass']
labels=np.load(DATA/'heldout-labels.npy');image_ids=np.load(DATA/'heldout-image-ids.npy')
saved_split=np.load(DATA/'split-ids.npz')
expected_ids=np.concatenate([saved_split[k] for k in ('pseudocal','validation','cal','eval')])
assert np.array_equal(expected_ids,image_ids) and len(np.unique(image_ids))==20000
assert np.array_equal(np.load(DATA/'view-ids.npy'),np.arange(10))
raw=np.load(DATA/'logits-fp32.npy',mmap_mode='r')
pz=np.asarray(np.load(DATA/'baseline-logits-fp32.npy'),dtype=np.float64)
qz=np.asarray(np.load(DATA/'mean-logits-f64.npy'),dtype=np.float64)
assert np.array_equal(pz,raw[:,4,:])
assert np.array_equal(qz,raw.mean(1,dtype=np.float64))
u=np.random.default_rng(20261031).random(60000)[image_ids]
np.save(OUT/'shared-U.npy',u)
nc=6000;yc=labels[4000:10000];ye=labels[10000:]
probabilities={t:arms(softmax(pz,t),softmax(qz,t)) for t in TS}

def make(method,t,arm,params,sl=slice(None)):
    order,v=probabilities[t][arm]
    order=order[sl];v=v[sl]
    s=scores_sorted(v,u[sl],method=method,**params)
    ranks=np.argsort(order,axis=1,kind='stable')
    true=s[np.arange(len(s)),ranks[np.arange(len(s)),labels[sl]]]
    return s,true,v

def metrics(sizes,covered,ys,target=.9,with_classes=False):
    r=coverage_metrics(covered,ys,target=target)
    out={'M':float(np.mean(sizes)),**r}
    if with_classes and r['Dhat'] is not None:
        n=np.array(r['class_n']);succ=np.bincount(ys,weights=covered,minlength=100)
        out['class_wilson95']=wilson(succ,n)
    if not with_classes:
        out.pop('class_n',None);out.pop('class_coverage',None)
    return out

def tune_result(method,params,t=1.,arm='00'):
    s,tr,_=make(method,t,arm,params,slice(0,4000))
    th=quantile(tr[:2000],.1)
    m=metrics((s[2000:]<=th).sum(1),tr[2000:]<=th,labels[2000:4000])
    return {'params':params,'T':t,'arm':arm,'threshold':th,**m}

def violation(c):return max(.89-c,0,c-.91)

params={'aps':{},'rankonly':{}};param_grid_results={};param_feasible={}
for method,grid in (
    ('raps',[{'lam':l,'kreg':k} for l in (.001,.01,.1) for k in (1,5,10)]),
    ('saps',[{'lam':l} for l in (.02,.05,.1,.2,.4,.6)])):
    rs=[tune_result(method,p) for p in grid]
    feasible=[r for r in rs if violation(r['coverage'])==0 and r['Dhat'] is not None]
    def sortkey(r):
        return (0 if feasible else violation(r['coverage']),r['M'],r['Dhat'] if r['Dhat'] is not None else float('inf'),
                abs(r['coverage']-.9),r['params']['lam'],r['params'].get('kreg',0))
    chosen=min(feasible or rs,key=sortkey)
    params[method]=chosen['params'];param_feasible[method]=bool(feasible)
    param_grid_results[method]={'grid':rs,'chosen':chosen,'coverage_feasible':bool(feasible)}
param_feasible.update({'aps':True,'rankonly':True})
selection={}
for method,p in params.items():
    rs=[tune_result(method,p,t,a) for a in ('00','11') for t in TS]
    anchor=next(r for r in rs if r['arm']=='00' and r['T']==1.)['Dhat']
    bysource={}
    for a,source in (('00','p'),('11','q')):
        group=[r for r in rs if r['arm']==a]
        feasible=[r for r in group if violation(r['coverage'])==0 and anchor is not None and r['Dhat'] is not None and r['Dhat']<=anchor+.001]
        chosen=min(feasible or group,key=lambda r:(0 if feasible else violation(r['coverage']),r['M'],
                    r['Dhat'] if r['Dhat'] is not None else float('inf'),abs(r['coverage']-.9),abs(math.log(r['T'])),r['T']))
        bysource[source]={'T':chosen['T'],'feasible':bool(feasible),'chosen_metrics':chosen}
    selection[method]={'sources':bysource,'all_tuning_points':rs,'Dhat_anchor':anchor}
write('tuning-only-selections.json',{'params':params,'param_grids':param_grid_results,'temperature_selection':selection,
        'all_primary_comparators':list(TS),'test_used_for_selection':False})
print(json.dumps({'stage':'tuning_locked','params':params,'temperatures':{m:{s:v['T'] for s,v in r['sources'].items()} for m,r in selection.items()}}),flush=True)

point={};point_vectors={};numerical={}
for method,p in params.items():
    for t in TS:
        for a in ARM:
            k=key(method,t,a)
            s,tr,v=make(method,t,a,p,slice(4000,None))
            for alpha in (.1,.05):
                th=quantile(tr[:nc],alpha)
                sizes=(s[nc:]<=th).sum(1);cov=tr[nc:]<=th
                met=metrics(sizes,cov,ye,target=1-alpha,with_classes=True)
                rk=k+f'|{alpha:g}'
                point[rk]={'method':method,'T':t,'arm':a,'alpha':alpha,'threshold':th,
                           'empty_fraction':float(np.mean(sizes==0)),'full_fraction':float(np.mean(sizes==100)),**met}
                if alpha==.1:point_vectors[k]=(sizes,cov)
            numerical[k]={'zero_probability_count':int((v==0).sum()),
               'samples_with_score_plateau':int((np.diff(s,axis=1)==0).any(axis=1).sum()),
               'max_score':float(s.max()),'min_probability':float(v.min()),'nonfinite_count':int((~np.isfinite(s)).sum())}
write('all-point-estimates.json',point)
write('numerical-diagnostics.json',numerical)
# Strong negative control consistency is checked on real heldout outputs.
rank_reference={a:point_vectors[key('rankonly',1.,a)] for a in ARM}
for t in TS:
    for a in ARM:
        for ref,got in zip(rank_reference[a],point_vectors[key('rankonly',t,a)]):
            assert np.array_equal(ref,got), 'Rankonly changed under temperature; inspect numerical rank ties'
    for aa,bb in (('00','01'),('10','11')):
        for left,right in zip(point_vectors[key('rankonly',t,aa)],point_vectors[key('rankonly',t,bb)]):
            assert np.array_equal(left,right), 'Rankonly spectrum dependence violates negative control'
write('real-output-consistency.json',{'image_and_view_order':True,'baseline_center_equal':True,
      'mean_logit_cache_equal':True,'rankonly_temperature_invariant':True,'rankonly_spectrum_invariant':True})
fields=['method','T','arm','alpha','M','coverage','D','Dhat','threshold','empty_fraction','full_fraction']
with (OUT/'all-point-estimates.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in point.values():w.writerow({k:r[k] for k in fields})
print(json.dumps({'stage':'point_estimates_complete','count':len(point)}),flush=True)

# Draw indices once; repeated exact image groups stay intact in every draw.
hash_by_id={}
for line in (DATA/'split-manifest.jsonl').open():
    r=json.loads(line);hash_by_id[r['id']]=r['image_sha256']
def group_positions(ids):
    d=collections.defaultdict(list)
    for pos,id_ in enumerate(ids):d[hash_by_id[int(id_)]].append(pos)
    return [np.array(v,dtype=np.int32) for v in d.values()]
cg=group_positions(image_ids[4000:10000]);eg=group_positions(image_ids[10000:])
rng=np.random.default_rng(20261032)
cis=[];eis=[]
for _ in range(B):
    cis.append(np.concatenate([cg[i] for i in rng.integers(len(cg),size=len(cg))]))
    eis.append(np.concatenate([eg[i] for i in rng.integers(len(eg),size=len(eg))]))
resample_hash=hashlib.sha256()
for ci,ei in zip(cis,eis):resample_hash.update(ci.tobytes());resample_hash.update(ei.tobytes())
write('bootstrap-design.json',{'B':B,'seed':20261032,'cal_group_count':len(cg),'eval_group_count':len(eg),
      'cal_resample_n_range':[min(map(len,cis)),max(map(len,cis))],
      'eval_resample_n_range':[min(map(len,eis)),max(map(len,eis))],
      'resampling_indices_sha256':resample_hash.hexdigest(),'tuning_reselected':False,'model_retrained':False})

variants=[('aps',t,a) for t in TS for a in ARM]
variants.extend((m,selection[m]['sources']['q']['T'],a) for m in ('raps','saps','rankonly') for a in ARM)
boot={};pointstat={};invalid_D=0
for vi,(method,t,a) in enumerate(variants):
    k=key(method,t,a);s,tr,_=make(method,t,a,params[method],slice(4000,None))
    thresholds=np.array([quantile(tr[:nc][ci],.1) for ci in cis])
    output=np.empty((B,3),dtype=np.float64)
    # Reuse samplewise size/coverage where a bootstrap quantile repeats.
    for threshold in np.unique(thresholds):
        sizes=(s[nc:]<=threshold).sum(1);covered=tr[nc:]<=threshold
        for bi in np.flatnonzero(thresholds==threshold):
            ei=eis[bi];c=covered[ei];yn=ye[ei]
            n=np.bincount(yn,minlength=100);success=np.bincount(yn,weights=c,minlength=100)
            d=np.nan
            if (n>=2).all():
                rate=success/n;d=np.mean((rate-.9)**2-rate*(1-rate)/(n-1))
            else:invalid_D+=1
            output[bi]=[np.mean(sizes[ei]),np.mean(c),d]
    boot[k]=output
    pr=point[k+'|0.1'];pointstat[k]=np.array([pr['M'],pr['coverage'],pr['Dhat']],dtype=np.float64)
    print(json.dumps({'stage':'paired_bootstrap','variant':vi+1,'total':len(variants),'key':k,'elapsed_seconds':time.time()-start}),flush=True)

# Exact preregistered inferential family. Any missing D stays unestimable.
family={}
def add(name, terms):
    value=0.;b=np.zeros(B)
    for coeff,k,j in terms:
        value+=coeff*pointstat[k][j];b+=coeff*boot[k][:,j]
    family[name]=(value,b)
for method,t,a in variants:
    k=key(method,t,a)
    for j,metric in enumerate(('M','coverage','Dhat')):add(f'{k}:{metric}',[(1,k,j)])
for method,tt in [('aps',TS)]+[(m,(selection[m]['sources']['q']['T'],)) for m in ('raps','saps','rankonly')]:
    for t in tt:
        k={a:key(method,t,a) for a in ARM}
        add(f'{method}|{t:g}:Rp',[(1,k['00'],0),(-1,k['10'],0)])
        add(f'{method}|{t:g}:Rq',[(1,k['01'],0),(-1,k['11'],0)])
        add(f'{method}|{t:g}:J',[(1,k['01'],0),(-1,k['11'],0),(-1,k['00'],0),(1,k['10'],0)])
tpstar=selection['aps']['sources']['p']['T'];tqstar=selection['aps']['sources']['q']['T']
for tp in TS:
    for tq in TS:
        kp,kq=key('aps',tp,'00'),key('aps',tq,'11')
        add(f'B|{tp:g}|{tq:g}:gain',[(1,kp,0),(-1,kq,0)])
        add(f'B|{tp:g}|{tq:g}:Ddiff',[(1,kq,2),(-1,kp,2)])
        add(f'B|{tp:g}|{tq:g}:Cdiff',[(1,kq,1),(-1,kp,1)])
for tq in TS:add(f'B|{tq:g}:restored_gain',[(1,key('aps',tpstar,'00'),0),(-1,key('aps',tq,'01'),0)])
names=list(family);theta=np.array([family[n][0] for n in names]);rep=np.column_stack([family[n][1] for n in names])
sd=rep.std(0,ddof=1);valid=np.isfinite(rep).all(0)&np.isfinite(theta)
stochastic=valid&(sd>0)
max_t=np.max(np.abs((rep[:,stochastic]-theta[stochastic])/sd[stochastic]),axis=1)
c95=float(np.quantile(max_t,.95,method='higher'));c90=float(np.quantile(max_t,.9,method='higher'))
intervals={}
for j,name in enumerate(names):
    if not valid[j]:intervals[name]={'estimable':False,'reason':'nonfinite D in at least one paired resample'};continue
    if sd[j]==0 and not np.all(rep[:,j]==theta[j]):
        intervals[name]={'estimable':False,'reason':'zero bootstrap SD inconsistent with point'};continue
    intervals[name]={'estimable':True,'estimate':float(theta[j]),'bootstrap_sd':float(sd[j]),
        'simultaneous95':[float(theta[j]-c95*sd[j]),float(theta[j]+c95*sd[j])],
        'simultaneous90':[float(theta[j]-c90*sd[j]),float(theta[j]+c90*sd[j])]}
write('simultaneous-intervals.json',{'family_size':len(names),'valid_dimensions':int(valid.sum()),'critical95':c95,
        'critical90':c90,'bootstrap_replicates':B,'invalid_D_variant_resamples':invalid_D,'intervals':intervals})
np.savez_compressed(OUT/'paired-bootstrap-statistics.npz',names=np.array(names),point=theta,replicates=rep)
def bounds(name):
    i=intervals[name]
    return i.get('simultaneous95',[-math.inf,math.inf])
rp,rq=bounds('aps|1:Rp'),bounds('aps|1:Rq')
opposite=(rp[0]>.25 and rq[1]<-.25) or (rq[0]>.25 and rp[1]<-.25)
coverA=all(bounds(f'aps|1|{a}:coverage')[0]>=.89 for a in ARM)
saturated=any(numerical[f'aps|1|{a}']['samples_with_score_plateau']>0 or numerical[f'aps|1|{a}']['zero_probability_count']>0 for a in ARM)
concentration={}
for name,a,b in [('Rp','00','10'),('Rq','01','11')]:
    diff=point_vectors[key('aps',1,a)][0]-point_vectors[key('aps',1,b)][0]
    absolute=np.abs(diff);den=absolute.sum();share=float(np.sort(absolute)[-100:].sum()/den) if den else 0.
    concentration[name]={'top_one_percent_absolute_contribution_share':share,'concentrated':share>=.5}
gateA=opposite and coverA and not saturated and not any(d['concentrated'] for d in concentration.values())
comparators=[]
for tp in TS:
    g=bounds(f'B|{tp:g}|{tqstar:g}:gain');d=bounds(f'B|{tp:g}|{tqstar:g}:Ddiff');c=bounds(f'B|{tp:g}|{tqstar:g}:Cdiff')
    pc=bounds(f'aps|{tp:g}|00:coverage');qc=bounds(f'aps|{tqstar:g}|11:coverage')
    comparators.append({'Tp':tp,'gain_pass':g[0]>.25,'D_pass':d[1]<=.001,
              'coverage_pass':pc[0]>=.89 and qc[0]>=.89 and c[0]>=-.01})
restored=bounds(f'B|{tqstar:g}:restored_gain');attenuation=bounds(f'aps|{tqstar:g}:Rq')
tuning_guard=tuning_eligibility(param_feasible,selection)
selection_ok=tuning_guard['all_feasible']
gateBraw=selection_ok and invalid_D==0 and all(all(x[k] for k in ('gain_pass','D_pass','coverage_pass')) for x in comparators) and restored[1]<=.25 and attenuation[0]>.25
# Counterevidence uses only tuning-selected original output points. No new test selection.
# If a needed point is outside the preregistered simultaneous family, uncertainty
# blocks promotion; it is never smuggled into the family after looking at results.
controls=[]
for m in ('saps','rankonly'):
    tp=selection[m]['sources']['p']['T'];k=key(m,tp,'00');r=point[k+'|0.1']
    q=point[key('aps',tqstar,'11')+'|0.1']
    control_tuning=bool(param_feasible[m]) and all(selection[m]['sources'][source]['feasible'] for source in ('p','q')) and selection['aps']['sources']['q']['feasible']
    control={'method':m,'Tp_tuning':tp,'point_M':r['M'],'point_coverage':r['coverage'],'point_Dhat':r['Dhat'],
             **assess_control(r,q,intervals,k,key('aps',tqstar,'11'),control_tuning)}
    controls.append(control)
gateB=gateBraw and all(c['mechanism_beyond_control_established'] for c in controls)
decision={'A_pass':bool(gateA),'A_opposite_rank_effects':bool(opposite),'A_coverage_pass':bool(coverA),
          'A_numeric_saturation':bool(saturated),'A_concentration':concentration,
          'B_primary_frontier_and_restore_pass':bool(gateBraw),'B_pass_after_controls':bool(gateB),
          'B_tuning_selection_feasible':selection_ok,'B_tuning_guard_details':tuning_guard,'B_Tp_star':tpstar,'B_Tq_star':tqstar,
          'B_all_7_comparators':comparators,'B_restored_gain_pass':restored[1]<=.25,
          'B_attenuation_pass':attenuation[0]>.25,'negative_controls':controls,
          'candidate_status':'REVIEW_SIGNAL' if gateA or gateB else 'CLOSE_THIS_CANDIDATE',
          'new_experiment_authorized_by_this_script':False,'single_trained_model_conditional':True,
          'elapsed_cpu_analysis_seconds':time.time()-start}
write('decision.json',decision)
write('analysis-environment.json',{'numpy':np.__version__,'analysis_file_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'conformal_core_sha256':hashlib.sha256((ROOT/'conformal_core.py').read_bytes()).hexdigest(),
        'cpu_only':True,'started_epoch':start,'ended_epoch':time.time()})
print(json.dumps({'stage':'complete',**decision},ensure_ascii=False),flush=True)
