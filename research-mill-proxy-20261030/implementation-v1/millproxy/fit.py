"""A/B estimators. Public fit functions accept only six train X,Y images."""
from dataclasses import dataclass
import numpy as np
from .core import Q,finite,readonly,source_bins,validate_training_pair
from .spectral import training_spectrum,principal_root

@dataclass(frozen=True)
class AParams:
    gamma:float
    gain:np.ndarray
    beta:np.ndarray
    loss:float

@dataclass(frozen=True)
class BParams:
    lut:np.ndarray
    mu:np.ndarray
    variance:np.ndarray
    sigma:np.ndarray
    quantiles:tuple
    original_counts:np.ndarray
    fallback:np.ndarray
    spectrum:np.ndarray
    root:np.ndarray

def gamma_grid():
    grid=np.exp(np.log(.2)+np.arange(4097,dtype=np.float64)/4096*np.log(25.))
    grid[2048]=1.0
    return grid

def nnls2(n,t,v):
    n=np.asarray(n,np.float64);t=np.asarray(t,np.float64);v=np.asarray(v,np.float64)
    finite(n);finite(t);finite(v)
    if not(n.shape==t.shape==v.shape) or np.any(n<0):raise ValueError('Invalid bin summaries')
    keep=n>0;n,t,v=n[keep],t[keep],v[keep]
    if not len(n):return np.zeros(2)
    design=np.column_stack([np.ones_like(t),t])*np.sqrt(n)[:,None]
    target=v*np.sqrt(n)
    least=np.linalg.lstsq(design,target,rcond=None)[0]
    # The feasible least-squares solution is already the minimum-norm solution,
    # including rank-one designs. Otherwise a two-variable optimum lies on an axis.
    if np.all(least>=0):return least
    candidates=[np.zeros(2),np.array([max(0.,np.sum(n*v)/np.sum(n)),0.])]
    den=np.sum(n*t*t)
    candidates.append(np.array([0.,max(0.,np.sum(n*t*v)/den) if den else 0.]))
    return min(candidates,key=lambda b:(float(np.sum(n*(v-b[0]-b[1]*t)**2)),float(b@b)))

def fit_a(x,y):
    x,y=validate_training_pair(x,y)
    codes=np.rint(x*255).astype(np.int64)
    grid=gamma_grid();powers=(np.arange(256,dtype=np.float64)[None,:]/255)**grid[:,None]
    gains=np.zeros((4097,3));losses=np.zeros(4097)
    for c in range(3):
        cc=codes[:,c].ravel();yy=y[:,c].ravel()
        n=np.bincount(cc,minlength=256).astype(np.float64)
        sy=np.bincount(cc,weights=yy,minlength=256)
        numerator=powers@sy;denominator=(powers*powers)@n
        gains[:,c]=np.clip(np.divide(numerator,denominator,out=np.zeros(4097),where=denominator!=0),0,2)
        losses+=np.sum(yy*yy)-2*gains[:,c]*numerator+gains[:,c]**2*denominator
    losses/=x.size
    eligible=np.flatnonzero(losses<=losses.min()+1e-14)
    winner=min(eligible,key=lambda k:(abs(np.log(grid[k])),grid[k]))
    gamma=float(grid[winner]);gain=gains[winner].copy()
    m=gain[None,:,None,None]*x**gamma
    residual=y-m;t=np.clip(m,0,1);beta=[]
    for c in range(3):
        tt=t[:,c].ravel();ee=residual[:,c].ravel();bins=np.minimum(15,np.floor(tt*16).astype(int))
        n=[];means=[];variances=[]
        for b in range(16):
            mask=bins==b
            if mask.any():
                e=ee[mask];center=e[0]+np.mean(e-e[0]);n.append(int(mask.sum()));means.append(tt[mask].mean());variances.append(np.mean((e-center)**2))
        beta.append(nnls2(n,means,variances))
    return AParams(gamma,readonly(gain),readonly(np.asarray(beta)),float(np.mean((y-m)**2)))

def monotone_lut(counts,sum_y):
    n=np.asarray(counts,np.float64);sy=np.asarray(sum_y,np.float64)
    if n.shape!=(256,) or sy.shape!=(256,) or np.any(n<0):raise ValueError('Invalid 256-node sufficient statistics')
    ids=np.flatnonzero(n>0)
    if not len(ids):raise ValueError('No observed LUT node')
    blocks=[]
    for j in ids:
        blocks.append([int(j),int(j),float(n[j]),float(sy[j])])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    fitted=np.empty(len(ids),np.float64)
    for lo,hi,weight,total in blocks:fitted[(ids>=lo)&(ids<=hi)]=np.clip(total/weight,0,1)
    return np.interp(np.arange(256),ids,fitted)

def nearest_nonempty(counts):
    ids=np.flatnonzero(np.asarray(counts)>0)
    if not len(ids):raise ValueError('No observed source bin')
    return np.array([ids[np.argmin(abs(ids-b))] for b in range(16)],np.int64)

def lut_query(lut,x):
    x=np.asarray(x,np.float64)
    return np.stack([np.interp(x[...,c,:,:],np.arange(256)/255,lut[c]) for c in range(3)],axis=-3)

def b_fields(x,p):
    bins=source_bins(x)
    mean=lut_query(p.lut,x)+np.moveaxis(p.mu[bins],-1,-3)
    sigma=np.moveaxis(p.sigma[bins],-1,-3)
    return mean,sigma,bins

def fit_b(x,y):
    x,y=validate_training_pair(x,y);codes=np.rint(x*255).astype(int)
    lut=np.stack([monotone_lut(np.bincount(codes[:,c].ravel(),minlength=256),np.bincount(codes[:,c].ravel(),weights=y[:,c].ravel(),minlength=256)) for c in range(3)])
    bins=source_bins(x);counts=np.bincount(bins.ravel(),minlength=16);fallback=nearest_nonempty(counts)
    residual=y-lut_query(lut,x);mu=np.zeros((16,3));variance=np.zeros((16,3));sigma=np.zeros((16,3));quantiles=[[None]*3 for _ in range(16)]
    z=np.empty_like(residual)
    for b in np.flatnonzero(counts):
        mask=bins==b
        for c in range(3):
            vals=residual[:,c][mask]
            # Shifted mean is the same estimator, but represents an exactly
            # constant empirical residual as an exactly zero point mass.
            mu[b,c]=vals[0]+np.mean(vals-vals[0]);centered=vals-mu[b,c]
            variance[b,c]=np.mean(centered*centered);sigma[b,c]=np.sqrt(max(variance[b,c],Q*Q/12))
            standardized=centered/sigma[b,c];z[:,c][mask]=standardized
            quantiles[b][c]=readonly(np.sort(standardized))
    for b in np.flatnonzero(counts==0):
        donor=fallback[b];mu[b]=mu[donor];variance[b]=variance[donor];sigma[b]=sigma[donor]
        quantiles[b]=quantiles[donor].copy()
    spectrum=training_spectrum(z);root=principal_root(spectrum)
    return BParams(*(readonly(a) for a in (lut,mu,variance,sigma)),tuple(tuple(row) for row in quantiles),readonly(counts),readonly(fallback),readonly(spectrum),readonly(root))
