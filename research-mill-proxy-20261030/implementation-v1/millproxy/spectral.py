"""Full two-sided orthonormal FFT, matrix roots and common-phase orbits."""
import numpy as np
from .core import finite

def frequency_pairs(h,w):
    flat=np.arange(h*w).reshape(h,w)
    partner=flat[np.ix_((-np.arange(h))%h,(-np.arange(w))%w)].ravel()
    ids=np.arange(h*w)
    return ids[ids<partner],partner[ids<partner],ids[ids==partner]

def conjugate_field(v):
    v=np.array(v,dtype=np.complex128,copy=True)
    if v.ndim!=3 or v.shape[0]!=3:raise ValueError('Expected [3,H,W] Fourier field')
    a,b,self_ids=frequency_pairs(*v.shape[-2:]);f=v.reshape(3,-1)
    f[:,b]=f[:,a].conj();f[:,self_ids]=f[:,self_ids].real;f[:,0]=0
    return v

def principal_root(spectrum):
    s=np.asarray(spectrum,dtype=np.complex128)
    finite(s)
    if s.ndim!=4 or s.shape[-2:]!=(3,3):raise ValueError('Expected [H,W,3,3] cross spectrum')
    s=(s+s.swapaxes(-1,-2).conj())*.5
    eig,vec=np.linalg.eigh(s)
    trace=np.trace(s,axis1=-2,axis2=-1).real
    if np.any(eig[...,0] < -1e-10*np.maximum(trace,1e-12)):raise ValueError('Materially negative cross-spectrum eigenvalue')
    eig=np.maximum(eig,0)
    root=(vec*np.sqrt(eig)[...,None,:])@vec.swapaxes(-1,-2).conj()
    a,b,self_ids=frequency_pairs(*s.shape[:2]);f=root.reshape(-1,3,3)
    f[b]=f[a].conj();f[self_ids]=f[self_ids].real;f[0]=0
    return root

def training_spectrum(z):
    z=np.asarray(z,dtype=np.float64);finite(z)
    if z.ndim!=4 or z.shape[:2]!=(6,3):raise ValueError('Six RGB standardized training images required')
    s=np.zeros((*z.shape[-2:],3,3),np.complex128)
    for image in z:
        image=image-image[:,:1,:1]
        image=image-image.mean(axis=(-2,-1),keepdims=True)
        f=np.fft.fft2(image,norm='ortho');f[:,0,0]=0
        s+=np.einsum('ahw,bhw->hwab',f,f.conj())/6
    s=(s+s.swapaxes(-1,-2).conj())*.5;s[0,0]=0
    return s

def real_ifft(v):
    finite(v)
    out=np.fft.ifft2(v,norm='ortho')
    tolerance=1e-10*max(1.,float(np.max(np.abs(out.real))))
    if float(np.max(np.abs(out.imag)))>tolerance:raise ValueError('IFFT imaginary residual exceeds protocol tolerance')
    return out.real

def orbit_project(v0,w):
    v0=np.asarray(v0,np.complex128);w=np.asarray(w,np.complex128)
    finite(v0);finite(w)
    if v0.shape!=w.shape or v0.shape[0]!=3:raise ValueError('Orbit shape mismatch')
    a,b,self_ids=frequency_pairs(*v0.shape[-2:]);v=v0.reshape(3,-1);ww=w.reshape(3,-1);p=np.zeros_like(v)
    norm=np.sum(abs(v[:,a])**2,axis=0)
    inner=np.sum(v[:,a].conj()*ww[:,a],axis=0)
    phase=np.ones(inner.shape,np.complex128)
    nz=abs(inner)!=0;phase[nz]=inner[nz]/abs(inner[nz])
    p[:,a]=v[:,a]*phase;p[:,a[norm==0]]=0;p[:,b]=p[:,a].conj()
    for k in self_ids:
        if k==0:continue
        vv=v[:,k].real
        if np.sum(vv*vv)==0:continue
        inner_real=float(np.dot(vv,ww[:,k].real))
        p[:,k]=vv*(-1 if inner_real<0 else 1)
    return p.reshape(v0.shape)
