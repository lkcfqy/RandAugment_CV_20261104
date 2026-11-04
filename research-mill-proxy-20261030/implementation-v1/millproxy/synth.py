"""Synthetic images consume test X, frozen parameters, and domain-separated RNG only."""
import numpy as np
from .core import q8,rgb,rng_for
from .fit import b_fields
from .spectral import conjugate_field,real_ifft,orbit_project

def synthesize_a(x,p,scene,level,seed):
    x=rgb(x,quantized=True)
    if x.ndim!=3:raise ValueError('One source image required')
    m=p.gain[:,None,None]*x**p.gamma;t=np.clip(m,0,1)
    variance=p.beta[:,0,None,None]+p.beta[:,1,None,None]*t
    if np.any(variance<0):raise ValueError('Negative A variance')
    eta=rng_for('A',scene,level,seed,'gaussian').standard_normal(x.shape,dtype=np.float64)
    return q8(m+np.sqrt(variance)*eta)

def rank_map_channel(u,bins,quantiles,priority):
    out=np.empty_like(u,dtype=np.float64);uf=u.ravel();bf=bins.ravel();pf=priority.ravel();of=out.ravel()
    for b in range(16):
        ids=np.flatnonzero(bf==b);n=len(ids)
        if not n:continue
        order=np.lexsort((pf[ids],uf[ids]));q=quantiles[b];nt=len(q)
        # Exact rational ceil(nt * (j-.5)/n), avoiding float-boundary ambiguity.
        j=np.arange(1,n+1,dtype=np.int64)
        indices=(nt*(2*j-1)+2*n-1)//(2*n)-1
        of[ids[order]]=q[indices]
    return out

def synthesize_b(x,p,scene,level,seed):
    x=rgb(x,quantized=True)
    if x.ndim!=3 or p.root.shape[:2]!=x.shape[-2:]:raise ValueError('Source and frozen spectrum dimensions differ')
    mean,sigma,bins=b_fields(x,p)
    eta=rng_for('B',scene,level,seed,'gaussian').standard_normal(x.shape,dtype=np.float64)
    eta_f=np.fft.fft2(eta,norm='ortho')
    v0=conjugate_field(np.einsum('hwab,bhw->ahw',p.root,eta_f));u=real_ifft(v0)
    priorities=[rng_for('B',scene,level,seed,'ties-'+c).permutation(bins.size).reshape(bins.shape) for c in 'RGB']
    for _ in range(10):
        projected=orbit_project(v0,np.fft.fft2(u,norm='ortho'));u=real_ifft(projected)
        u=np.stack([rank_map_channel(u[c],bins,tuple(p.quantiles[b][c] for b in range(16)),priorities[c]) for c in range(3)])
    result=q8(mean+sigma*u)
    fallback_fraction=float(np.mean(p.original_counts[bins]==0))
    return result,{'iterations':10,'last_operation':'rank_map_then_Q8','empty_train_bin_fallback_fraction':fallback_fraction,'single_draw_spectrum_rank':'at most one; not the empirical training cross-spectrum'}
