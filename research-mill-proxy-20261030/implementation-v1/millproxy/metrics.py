"""Floating output metrics, evaluated only after model output clamp [0,1]."""
import numpy as np
from .core import rgb

XYZ=np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]],np.float64)
WHITE=np.array([.95047,1,1.08883],np.float64)

def psnr(p,g):
    p=rgb(p);g=rgb(g)
    if p.shape!=g.shape:raise ValueError('Metric image shape mismatch')
    mse=float(np.mean((p-g)**2))
    if not np.isfinite(mse) or mse<=0:raise ValueError('Zero/nonfinite MSE is a validation failure')
    return float(-10*np.log10(mse))

def lab(x):
    x=rgb(x)
    linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    xyz=np.einsum('ab,bhw->ahw',XYZ,linear)/WHITE[:,None,None]
    delta=6/29
    f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta*delta)+4/29)
    return np.stack([116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2])])

def delta_e76(p,g):
    p=rgb(p);g=rgb(g)
    if p.shape!=g.shape:raise ValueError('Metric image shape mismatch')
    return float(np.sqrt(np.sum((lab(p)-lab(g))**2,axis=0)).mean())
