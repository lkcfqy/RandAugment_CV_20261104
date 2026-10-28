"""Arithmetic audit of saved banks and predictions; no encoder, fit or selection."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
def read(n): return json.loads((ROOT/n).read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def close(a,b):
    assert np.allclose(a,b,rtol=1e-9,atol=1e-10), (a,b)
def object_key(r): return (r['category'],r['instance'])
def full_key(r): return tuple(r[k] for k in ['category','instance','elevation','azimuth','lighting'])
def ll(logits,y): return np.logaddexp.reduce(logits,axis=1)-logits[np.arange(len(y)),y]

parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args()
p=read('protocol.json'); lock=read('protocol-lock.json'); train=read('training-results.json'); freeze=read('training-freeze.json'); pred=read('source-prediction.json')
for n,h in lock['hashes'].items(): assert sha(ROOT/n)==h,n
for n,h in freeze['files'].items(): assert sha(ROOT/n)==h,n
assert freeze['protocol_lock_SHA256']==sha(ROOT/'protocol-lock.json')
original=read('aborted-source-pass-v0/protocol.json')
assert {k:v for k,v in p.items() if k!='resources'}=={k:v for k,v in original.items() if k!='resources'}
ro=dict(original['resources']);rn=dict(p['resources']);assert ro.pop('encoder_wall_seconds_cap_per_phase')==360;assert rn.pop('encoder_wall_seconds_cap_per_phase')==480;assert ro==rn
sets=[set(map(tuple,p[n])) for n in ['source_objects','development_objects','final_objects']]
assert list(map(len,sets))==[15,10,25] and all(not sets[i]&sets[j] for i in range(3) for j in range(i))
for role,rs in p['roles'].items():
    assert len({full_key(r) for r in rs})==len(rs)
    assert set(Counter(object_key(r) for r in rs).values())=={36}
    assert {r['azimuth'] for r in rs}==set(p['angles_codes_novel' if role.endswith('novel') else 'angles_codes_seen'])
bank=np.load(ROOT/'source-feature-bank.npz'); weights=np.load(ROOT/'readouts.npz')
z=bank['clean']; ys=np.array([r['category'] for r in p['roles']['source_seen']]); yd=np.array([r['category'] for r in p['roles']['development_seen']])
close(z.mean(0),np.zeros(512));close(np.mean(z*z),1.)
arms=['clean']+[q['id'] for q in p['policies']]
max_gradient=0.;max_objective_diff=0.
for arm in arms:
    x=z if arm=='clean' else np.concatenate([z,bank[arm+'_plus'],bank[arm+'_minus']])
    y=ys if arm=='clean' else np.tile(ys,3)
    w,b=weights[arm+'_W'],weights[arm+'_b']; lg=x@w+b
    objective=ll(lg,y).mean()+.005*np.sum(w*w)
    prob=np.exp(lg-lg.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);prob[np.arange(len(y)),y]-=1
    gradient=max(np.max(np.abs(x.T@prob/len(y)+.01*w)),np.max(np.abs(prob.mean(0))))
    assert np.isfinite(objective) and gradient<=1e-5
    max_gradient=max(max_gradient,float(gradient));max_objective_diff=max(max_objective_diff,abs(float(objective)-train['models'][arm]['objective']))
    close(objective,train['models'][arm]['objective']);close(gradient,train['models'][arm]['gradient_inf'])
    for name,xv,yv in [('source_clean',z,ys),('development_seen',bank['development_seen'],yd)]:
        lgv=xv@w+b
        close(ll(lgv,yv).mean(),train['models'][arm][name+'_LL']);close(np.mean(lgv.argmax(1)==yv),train['models'][arm][name+'_accuracy'])
lookup={full_key(r):j for j,r in enumerate(p['roles']['source_seen'])}; ds=[[],[],[]]
for t,j in lookup.items():
    c,i,e,a,l=t
    for k,new in [(0,(c,i,e,a+2,l)),(1,(c,i,e+4,a,l)),(2,(c,i,e,a,3))]:
        if (k==0 and a+2 in p['angles_codes_seen']) or (k==1 and e+4 in [0,4,8]) or (k==2 and l==0): ds[k].append(z[lookup[new]]-z[j])
assert list(map(len,ds))==[450,360,270]
moments=[np.einsum('ni,nj->ij',d,d)/len(d) for d in map(np.asarray,ds)];cv=.5*moments[0]+.5*moments[1];cl=moments[2]
def cosine(a,b): return np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
score_errors=[]
for m in pred['metrics']:
    a=m['id'];delta=np.concatenate([bank[a+'_plus']-z,bank[a+'_minus']-z]);ca=delta.T@delta/len(delta)
    s=float(cosine(ca,cv)-cosine(ca,cl));score_errors.append(abs(s-m['score']));close(s,m['score']);close(np.trace(ca),m['feature_energy'])
    x=(z+bank[a+'_plus']+bank[a+'_minus'])/3;x=x-x.mean(0);y=np.eye(5)[ys];y-=y.mean(0);K=x@x.T;L=y@y.T
    close(np.vdot(K,L)/(np.linalg.norm(K)*np.linalg.norm(L)),m['label_kernel_alignment'])
assert pred['high']==sorted(pred['metrics'],key=lambda m:(-m['score'],m['id']))[0]['id']
assert pred['low']==sorted(pred['metrics'],key=lambda m:(m['score'],m['id']))[0]['id']
assert datetime.fromisoformat(pred['at_UTC'])<datetime.fromisoformat(freeze['at_UTC'])
assert freeze['new_final_pixels_read']==train['final_pixels_read']==0
receipt=read('training-download.json');raw=ROOT/'data'/p['downloads']['training']['file'];assert raw.stat().st_size==receipt['bytes']==p['downloads']['training']['content_length'];assert sha(raw)==receipt['sha256']
audit={'at_UTC':datetime.now(timezone.utc).isoformat(),'mode':'training_only' if args.training_only else 'complete_saved_result','no_encoder_or_fit_or_new_selection':True,'protocol_and_freeze_hashes_pass':True,'all_scientific_fields_unchanged_by_budget_amendment':True,'object_split_sizes':[15,10,25],'views_per_object_per_regime':36,'source_pair_counts':[450,360,270],'max_recomputed_training_gradient_inf':max_gradient,'max_objective_difference':max_objective_diff,'max_source_score_difference':max(score_errors),'model_request_vs_actual_note':'Sol/max requested; actual tier execution metadata unavailable'}
if not args.training_only:
    result=read('final-results.json'); feats=np.load(ROOT/'evaluation-features.npz'); saved=np.load(ROOT/'evaluation-predictions.npz'); grouped=np.load(ROOT/'per-object-losses.npz'); recomputed={};max_logits_difference=0.
    for role,x in feats.items():
        records=p['roles'][role];y=np.array([r['category'] for r in records]);objects=sorted(set(map(object_key,records)));a=[]
        for arm in arms:
            logits=x@weights[arm+'_W']+weights[arm+'_b'];max_logits_difference=max(max_logits_difference,float(np.max(np.abs(logits-saved[role+'__'+arm+'__logits']))));loss=ll(logits,y)
            close(loss.mean(),result['risk_table'][role][arm]['LL']);close(np.mean(logits.argmax(1)==y),result['risk_table'][role][arm]['accuracy'])
            a.append([loss[[object_key(r)==obj for r in records]].mean() for obj in objects])
        recomputed[role]=np.array(a).T;close(recomputed[role],grouped[role])
    objs=sorted(sets[2]);rng=np.random.default_rng(2026102121);draws=[]
    for c in range(5): draws.append(rng.choice(np.flatnonzero([o[0]==c for o in objs]),size=(5000,5),replace=True))
    idx=np.hstack(draws)
    def summary(v): return float(v.mean()),np.quantile(np.mean(v[idx],axis=1),[.025,.975])
    s,u=recomputed['final_seen'],recomputed['final_novel'];hi,lo=arms.index(pred['high']),arms.index(pred['low']);g=(u[:,lo]-u[:,hi])-(s[:,lo]-s[:,hi]);point,ci=summary(g)
    close(point,result['primary_policy_by_azimuth_interaction']['mean']);close(ci,result['primary_policy_by_azimuth_interaction']['paired_category_stratified_object_bootstrap95'])
    assert result['predeclared_interaction_screen_passed']==bool(point>=.01 and ci[0]>0)
    for j,arm in enumerate(arms[1:],1):
        gs=s[:,0]-s[:,j];gu=u[:,0]-u[:,j]
        for name,v in [('gain_seen',gs),('gain_novel',gu),('extra_novel_gain_interaction',gu-gs)]:
            point2,ci2=summary(v);close(point2,result['all_policy_final_object_gains'][arm][name]['mean']);close(ci2,result['all_policy_final_object_gains'][arm][name]['paired_category_stratified_object_bootstrap95'])
    close(g,(u[:,0]-u[:,hi])-(s[:,0]-s[:,hi])-(u[:,0]-u[:,lo])+(s[:,0]-s[:,lo]))
    tr=read('testing-download.json');raw=ROOT/'data'/p['downloads']['testing']['file'];assert raw.stat().st_size==tr['bytes']==p['downloads']['testing']['content_length'];assert sha(raw)==tr['sha256'];assert datetime.fromisoformat(tr['at_UTC'])>datetime.fromisoformat(freeze['at_UTC'])
    assert tr['bytes']+receipt['bytes']<=p['resources']['new_image_download_bytes_max']
    for field,name in [('code_SHA256','experiment.py'),('protocol_SHA256','protocol.json'),('training_freeze_SHA256','training-freeze.json'),('predictions_SHA256','evaluation-predictions.npz')]: assert result[field]==sha(ROOT/name)
    audit.update({'all_saved_logits_losses_and_object_aggregates_match':True,'max_saved_logits_difference':max_logits_difference,'main_effect_and_5000_object_bootstrap_match':True,'final_objects':25,'independent_unit':'physical object,paired within five fixed categories','source_training_frozen_before_testing_download':True,'new_image_bytes':tr['bytes']+receipt['bytes'],'not_interpreted_as_causal_effect_or_full_RA':True})
out=ROOT/('verification-training.json' if args.training_only else 'verification-final.json')
with out.open('x') as f: json.dump(audit,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(audit))
