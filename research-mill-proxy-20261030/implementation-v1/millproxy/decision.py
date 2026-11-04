"""Response controls, within-level distinct-scene ranking, and frozen gate order."""
import numpy as np
from .core import LEVELS,finite

def matrix(x):
    x=np.asarray(x,np.float64);finite(x)
    if x.shape!=(8,4):raise ValueError('Expected all eight test scenes by four levels')
    return x

def pair_agreement(real,pred,mask=None,multiplicity=None):
    real=finite(np.asarray(real,np.float64));pred=finite(np.asarray(pred,np.float64))
    if real.ndim!=1 or pred.shape!=real.shape:raise ValueError('Pair vector mismatch')
    n=len(real);mask=np.ones(n,bool) if mask is None else np.asarray(mask,bool)
    counts=np.ones(n,int) if multiplicity is None else np.asarray(multiplicity,int)
    if mask.shape!=(n,) or counts.shape!=(n,) or np.any(counts<0):raise ValueError('Invalid pair mask/multiplicities')
    pairs=[];weight=0.;correct=0.;scenes=set()
    for s in range(n):
        for t in range(s+1,n):
            if not(mask[s] and mask[t]) or counts[s]*counts[t]==0 or abs(real[s]-real[t])<.5:continue
            diff=pred[s]-pred[t]
            score=.5 if abs(diff)<=1e-8 else float(diff*(real[s]-real[t])>0)
            pw=int(counts[s]*counts[t]);pairs.append([s,t,score,pw]);weight+=pw;correct+=pw*score;scenes.update([s,t])
    return {'eligible_pairs':len(pairs),'covered_scenes':len(scenes),'pair_weight':weight,'agreement':correct/weight if weight else None,'sufficient':len(pairs)>=8 and len(scenes)>=6,'pairs':pairs}

def responses(test_anchor,test_real,synthetic_metrics,train_anchor,train_real,*,color):
    t=finite(np.asarray(test_anchor,np.float64));r=matrix(test_real);s=finite(np.asarray(synthetic_metrics,np.float64))
    tt=finite(np.asarray(train_anchor,np.float64));tr=finite(np.asarray(train_real,np.float64))
    if t.shape!=(8,) or s.shape!=(8,4,4) or tt.shape!=(6,) or tr.shape!=(6,4):raise ValueError('Incomplete task metrics')
    sign=1 if color else -1
    real=sign*(r-t[:,None]);syn=sign*(s.mean(2)-t[:,None]);train=sign*(tr-tt[:,None])
    return {'real':real,'synthetic':syn,'C_meanD':np.broadcast_to(train.mean(0),(8,4)).copy(),'C_anchor':sign*(tr.mean(0)[None,:]-t[:,None]),'absolute_real':r,'absolute_synthetic':s.mean(2)}

def ranking_report(real,pred):
    real=matrix(real);pred=matrix(pred)
    levels=[pair_agreement(real[:,j],pred[:,j]) for j in range(4)]
    scores=[x['agreement'] for x in levels]
    return {'levels':dict(zip(map(str,LEVELS),levels)),'all_levels_sufficient':all(x['sufficient'] for x in levels),'equal_level_agreement':float(np.mean(scores)) if all(x is not None for x in scores) else None}

def absolute_ranking(real_r,syn_s,real_d,*,eligibility):
    r=matrix(real_r);s=matrix(syn_s);d=matrix(real_d)
    if eligibility=='absolute_R':return ranking_report(r,s)
    raise ValueError('Absolute ranking must use its own |R_s-R_t| >= .5 pairs, never D pairs')

def decide(real_psnr,real_color,pred_psnr,pred_color,b_pass,*,validation_ok=True,resource_ok=True,c_anchor_confirmed=False,identifiable=None,unidentifiable_policy=None):
    if not validation_ok:return {'decision':'VALIDATION_BLOCKED','scientific_negative':False}
    if not resource_ok:return {'decision':'RESOURCE_BLOCKED','scientific_negative':False}
    if not c_anchor_confirmed:return {'decision':'PENDING_C_ANCHOR_CONFIRMATION','scientific_negative':False}
    real=matrix(real_psnr);color=matrix(real_color)
    if set(pred_psnr)!=set(pred_color) or set(pred_psnr)!={'A','B','C_anchor','C_meanD'}:raise ValueError('Incomplete methods/controls')
    p=np.asarray(b_pass)
    if p.shape!=(8,4) or not np.isin(p,[False,True]).all():raise ValueError('Invalid cell pass mask')
    p=p.astype(bool);ident=np.ones((8,4),bool) if identifiable is None else np.asarray(identifiable,bool)
    if ident.shape!=(8,4) or np.any(p&~ident):raise ValueError('Unidentifiable cell cannot pass')
    reports={}
    for method in pred_psnr:
        prediction=matrix(pred_psnr[method]);cp=matrix(pred_color[method])
        reports[method]={'mae_psnr':float(abs(prediction-real).mean()),'mae_color':float(abs(cp-color).mean()),'ranking':ranking_report(real,prediction)}
    # Report B and C_anchor on exactly the same stat-passing cells and pairs,
    # even when another proxy triggers the higher-priority CLOSE decision.
    b_reports={};anchor_reports={};same_cells={}
    for j,level in enumerate(LEVELS):
        mask=p[:,j]
        b_reports[str(level)]=pair_agreement(real[:,j],pred_psnr['B'][:,j],mask=mask)
        anchor_reports[str(level)]=pair_agreement(real[:,j],pred_psnr['C_anchor'][:,j],mask=mask)
        same_cells[str(level)]={'scene_indices':np.flatnonzero(mask).tolist(),'B_mae_psnr':float(abs(pred_psnr['B'][mask,j]-real[mask,j]).mean()) if mask.any() else None,'C_anchor_mae_psnr':float(abs(pred_psnr['C_anchor'][mask,j]-real[mask,j]).mean()) if mask.any() else None,'B_mae_color':float(abs(pred_color['B'][mask,j]-color[mask,j]).mean()) if mask.any() else None,'C_anchor_mae_color':float(abs(pred_color['C_anchor'][mask,j]-color[mask,j]).mean()) if mask.any() else None}
    common={'methods':reports,'stat_pass_endpoint_pairs':{'B':b_reports,'C_anchor':anchor_reports},'stat_pass_common_cells':same_cells,'control_note':'C_anchor and D use test GT scoring; diagnostic control, not a GT-free deployment predictor'}
    close=[]
    for method in ('A','B','C_anchor'):
        m=reports[method];r=m['ranking']
        if m['mae_psnr']<=.2 and m['mae_color']<=1 and r['all_levels_sufficient'] and r['equal_level_agreement']>=.8:close.append(method)
    if close:return common|{'decision':'CLOSE_KNOWN_PROXY_SUFFICIENT','close_methods':close}
    if not ident.all():
        if unidentifiable_policy is None:return common|{'decision':'PENDING_STAT_UNIDENTIFIABLE_POLICY'}
        if unidentifiable_policy=='global':return common|{'decision':'STAT_UNIDENTIFIABLE_NO_CONTINUE'}
        if unidentifiable_policy!='cell':raise ValueError('Unknown unidentifiable policy')
    bad=[]
    for j,level in enumerate(LEVELS):
        r=b_reports[str(level)]
        if r['sufficient'] and r['agreement']<=.6:bad.append(j)
    witness=[]
    for scene in range(8):
        if any(p[scene,j] and abs(pred_psnr['B'][scene,j]-real[scene,j])>=1 for j in bad):witness.append(scene)
    keep=len(bad)>=2 and len(witness)>=4
    return common|{'decision':'RETAIN_NARROW_MEASUREMENT_PENDING_REVIEW' if keep else 'NO_UPGRADE','Lbad':[LEVELS[j] for j in bad],'distinct_witness_scene_indices':witness,'reason':None if keep else 'Insufficient stat-passing pair coverage / disagreement / distinct-scene errors; no gate or simulator changes'}

def cluster_bootstrap(real,pred):
    real=matrix(real);pred=matrix(pred);rng=np.random.Generator(np.random.PCG64(20261030))
    maes=[];agreements=[]
    for _ in range(999):
        counts=np.bincount(rng.integers(0,8,size=8),minlength=8)
        maes.append(float(np.sum(abs(pred-real)*counts[:,None])/(8*4)))
        scores=[pair_agreement(real[:,j],pred[:,j],multiplicity=counts)['agreement'] for j in range(4)]
        agreements.append(float(np.mean(scores)) if all(x is not None for x in scores) else None)
    valid=[x for x in agreements if x is not None]
    percentile=lambda x:np.percentile(x,[2.5,97.5],method='linear').tolist() if len(x) else [None,None]
    return {'replicates':999,'seed':20261030,'mae_percentile95':percentile(maes),'agreement_percentile95':percentile(valid),'agreement_NA_replicates':sum(x is None for x in agreements),'method':'scene cluster; original distinct-scene pairs weighted n_s*n_t; no retries; not a gate'}
