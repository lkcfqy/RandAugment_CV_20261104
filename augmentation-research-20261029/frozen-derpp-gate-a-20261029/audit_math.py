"""Frozen, NumPy-only Gate A definitions. Known identities, not novelty claims."""
import numpy as np

EPS = 1e-6  # Original half-SUM epsilon 1e-5 / 10 output dimensions.

def scores(h, z):
    h=np.asarray(h,np.float64);z=np.asarray(z,np.float64)
    mean=h.mean(axis=0)
    da=.5*np.square(h-z).mean(axis=-1).mean(axis=0)
    dm=.5*np.square(mean-z).mean(axis=-1)
    var=.5*np.square(h-mean).mean(axis=-1).mean(axis=0)
    return da,dm,var

def update_weights(w, ids, residual, q):
    d=np.asarray(residual,np.float64)+EPS
    q=(1-len(ids)/len(w))*q+(len(ids)/len(w))*np.quantile(d,.5,method='linear')
    eta=np.clip(1-np.maximum(d-q,0)/max(q,np.finfo(np.float32).tiny),0,1)
    ratio=(eta*d+(1-eta)*q)/d
    w[ids]=.5*w[ids]+.5*np.sqrt(ratio)
    return float(q)

def weighted_ids(w, uniforms, count=32):
    u=np.clip(uniforms,1e-15,1-1e-15)
    return np.argsort(-(np.log(w)-np.log(-np.log(u))),kind='stable')[:count]

def rank(x):
    x=np.asarray(x);o=np.argsort(x,kind='stable');r=np.empty(len(x),float);i=0
    while i<len(o):
        j=i+1
        while j<len(o) and x[o[j]]==x[o[i]]:j+=1
        r[o[i:j]]=(i+j-1)/2;i=j
    return r

def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);x=x-x.mean();y=y-y.mean()
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>1e-15 else 0.

def spearman(x,y):return corr(rank(x),rank(y))
def probability(w):return w/w.sum()
def tv(a,b):return float(np.abs(probability(a)-probability(b)).sum()/2)

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-30 else 0.

def stats(w):
    p=probability(w)
    return {'min':float(w.min()),'max':float(w.max()),'sum':float(w.sum()),
            'entropy':float(-(p*np.log(p)).sum()),'ESS':float(1/(p@p))}

def assign(w,score,strata,rows,arrivals):
    out=w.copy()
    for s in np.unique(strata):
        ids=np.flatnonzero(strata==s)
        ids=ids[np.lexsort((ids,arrivals[ids],rows[ids],-score[ids]))]
        out[ids]=np.sort(w[ids])
    return out

def invariants(w,arms,strata):
    ans={}
    for name,x in arms.items():
        exact=np.array_equal(np.sort(w),np.sort(x));error=0.
        for s in np.unique(strata):
            mask=strata==s
            exact=exact and np.array_equal(np.sort(w[mask]),np.sort(x[mask]))
            error=max(error,abs(x[mask].sum()/x.sum()-w[mask].sum()/w.sum()))
        assert exact and error<1e-12
        ans[name]={'exact_global_and_stratum_weight_multisets':bool(exact),
                   'max_stratum_mass_error':float(error),**stats(x)}
    return ans

def rank_pairs(a,b,strata):
    discordant=0;strict=0;ties=0
    for s in np.unique(strata):
        ids=np.flatnonzero(strata==s);i,j=np.triu_indices(len(ids),1)
        da=np.sign(a[ids[i]]-a[ids[j]]);db=np.sign(b[ids[i]]-b[ids[j]])
        valid=(da!=0)&(db!=0);strict+=int(valid.sum());ties+=int((~valid).sum())
        discordant+=int(((da*db<0)&valid).sum())
    return {'discordant_pairs':discordant,'strict_pairs':strict,'tied_pairs':ties,
            'discordant_fraction':discordant/strict if strict else 0.}

def ols_r2(y,x):
    coef=np.linalg.lstsq(x,y,rcond=None)[0];pred=x@coef;den=np.square(y-y.mean()).sum()
    return float(1-np.square(y-pred).sum()/den) if den>1e-30 else 0.

def self_tests():
    rng=np.random.default_rng(72346);h=rng.normal(size=(8,17,10));z=rng.normal(size=(17,10))
    da,dm,v=scores(h,z);err=float(np.abs(da-dm-v).max());assert err<1e-12
    dc,mc,vc=scores(h-h.mean(-1,keepdims=True),z-z.mean(-1,keepdims=True))
    trans=.5*np.square(h.mean(-1)-z.mean(-1)).mean(0)
    assert np.max(np.abs(da-dc-trans))<1e-12
    zero_target=np.zeros_like(z)
    a,b,c=scores(np.repeat(zero_target[None],8,axis=0),zero_target);assert np.array_equal(a,b) and np.all(c==0)
    w=np.ones(17);q=update_weights(w,np.arange(17),np.zeros(17),0.)
    assert q==EPS and np.array_equal(w,np.ones(17))
    # Same ratio under conversion from half-mean to the original half-sum units.
    d=np.linspace(.1,3,17);w=np.ones(17);q=update_weights(w,np.arange(17),d,0.)
    ds=10*d+1e-5;qs=np.quantile(ds,.5);eta=np.clip(1-np.maximum(ds-qs,0)/qs,0,1)
    sum_unit=.5+.5*np.sqrt((eta*ds+(1-eta)*qs)/ds)
    assert np.max(np.abs(w-sum_unit))<1e-15
    strata=np.arange(17)%3;rows=np.arange(17);arr=rows.copy()
    aa=assign(w,da,strata,rows,arr);bb=assign(w,dm,strata,rows,arr)
    invariants(w,{'A':aa,'B':bb},strata)
    assert tv(np.ones(17),np.ones(17))==0 and cosine(np.zeros(17),np.ones(17))==0
    return {'all_passed':True,'max_synthetic_identity_error':err,
            'zero_residual_control':True,'half_sum_to_half_mean_units_equivalent':True,
            'matched_multisets_control':True}
