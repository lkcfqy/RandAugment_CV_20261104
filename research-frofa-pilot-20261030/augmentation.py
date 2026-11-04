"""Batched channel prefix action of the already validated exact fixture."""
import numpy as np
from fixture_controls import StructuredChannel,independent_marginals,normalize_channel
def action(u,d,q,sqrtw,low,high,mu,z):
    p=sqrtw*z
    ps=np.concatenate((np.zeros(p.shape[:-1]+(1,)),np.cumsum(p,axis=-1)),axis=-1)
    rs=np.concatenate((np.zeros(p.shape[:-1]+(1,)),np.cumsum(p*q,axis=-1)),axis=-1)
    take=lambda arr,idx:np.take_along_axis(arr,idx,axis=-1)
    result=d*(-u*take(ps,low)+take(rs,high)-take(rs,low)+(1-u)*(ps[...,-1:]-take(ps,high)))-mu*ps[...,-1:]
    return result
class SupportControls:
    def __init__(self,x,a,gaussian=True,budget=None):
        self.x=np.asarray(x,dtype=np.float32);self.a=a
        self.packed=[];self.gaussian=gaussian
        for sample in self.x:
            xc=sample.T.astype(np.float64);lo=xc.min(axis=1,keepdims=True);d=xc.max(axis=1,keepdims=True)-lo+1e-8;u=(xc-lo)/d
            item={'u':u,'d':d,'lo':lo}
            if gaussian:
                parts=[StructuredChannel(u[c],float(d[c,0]),a) for c in range(u.shape[0])]
                J=max(p.q.size for p in parts);C,T=u.shape
                q=np.zeros((C,J));sq=np.zeros((C,J));low=np.zeros((C,T),np.int32);high=low.copy();mu=np.zeros((C,T));var=mu.copy()
                for c,p in enumerate(parts):
                    k=len(p.q);q[c,:k]=p.q;sq[c,:k]=p.sqrtw;low[c]=p.low;high[c]=p.high;mu[c]=p.mu
                    emu,var[c]=independent_marginals(u[c],float(d[c,0]),a)
                    if np.max(np.abs(emu-mu[c])/d[c,0])>1e-10:raise RuntimeError('analytic marginal mismatch')
                item.update(q=q,sqrtw=sq,low=low,high=high,mu=mu,var=var)
            self.packed.append(item)
            if budget:budget.check()
    def batch(self,indices,arm,rng):
        if arm=='clean':return np.array(self.x[indices],copy=True)
        output=np.empty((len(indices),)+self.x.shape[1:],np.float32)
        for j,i in enumerate(indices):
            p=self.packed[int(i)];u,d=p['u'],p['d']
            if arm in ('C','E'):
                size=(u.shape[0],1) if arm=='C' else u.shape
                delta=d*(np.clip(u+rng.uniform(-self.a,self.a,size),0.,1.)-u)
            elif arm=='GC':
                delta=p['mu']+action(u,d,p['q'],p['sqrtw'],p['low'],p['high'],p['mu'],rng.standard_normal(p['q'].shape))
            elif arm=='GE':delta=p['mu']+np.sqrt(p['var'])*rng.standard_normal(u.shape)
            else:raise ValueError(arm)
            # Gaussian controls are not clamped or renormalized.
            output[j]=(self.x[int(i)].astype(np.float64)+delta.T).astype(np.float32)
        return output
    @property
    def bytes(self):return sum(v.nbytes for p in self.packed for v in p.values())+self.x.nbytes
