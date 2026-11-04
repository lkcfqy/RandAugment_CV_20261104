"""Per-scene/level diagnostics on final Q8 images, using frozen B fields for A and B."""
import numpy as np
from .core import Q,finite,rgb
from .fit import b_fields

PAIRS=((0,1),(0,2),(1,2))

def ks2(x,y):
    x=np.sort(finite(np.asarray(x,np.float64).ravel()));y=np.sort(finite(np.asarray(y,np.float64).ravel()))
    if not len(x) or not len(y):raise ValueError('KS requires nonempty samples')
    values=np.sort(np.concatenate([x,y]))
    # Integer CDF numerators avoid e.g. 1-.95 rounding just above the .05 gate.
    nx=np.searchsorted(x,values,side='right');ny=np.searchsorted(y,values,side='right')
    numerator=np.max(abs(nx*len(y)-ny*len(x)))
    return float(numerator/(len(x)*len(y)))

def bin_errors(real,draws,bins):
    real=finite(np.asarray(real,np.float64));draws=finite(np.asarray(draws,np.float64));bins=np.asarray(bins)
    if draws.shape!=(4,*real.shape) or real.shape[0]!=3 or bins.shape!=real.shape[-2:]:raise ValueError('Four matched residual draws required')
    if bins.min()<0 or bins.max()>15:raise ValueError('Source bin out of range')
    emean=np.zeros(3);evar=np.zeros(3)
    for b in range(16):
        mask=bins==b;n=int(mask.sum())
        if not n:continue
        r=real[:,mask];s=draws[:,:,mask];w=n/bins.size
        rm=r[:,0]+(r-r[:,:1]).mean(-1);sm=s[:,:,0]+(s-s[:,:,:1]).mean(-1)
        rv=np.mean((r-rm[:,None])**2,axis=-1);sv=np.mean((s-sm[:,:,None])**2,axis=-1)
        emean+=w*np.mean(abs(sm-rm),axis=0)
        evar+=w*np.mean(abs(sv-rv)/np.maximum(rv,Q*Q),axis=0)
    return {'Emean':float(emean.max()),'Evar':float(evar.max())}

def band_ids(h,w):
    edges=np.linspace(-.5,.5,17)
    row=np.clip(np.searchsorted(edges,np.fft.fftfreq(h),side='right')-1,0,15)
    col=np.clip(np.searchsorted(edges,np.fft.fftfreq(w),side='right')-1,0,15)
    return (row[:,None]*16+col[None,:]).ravel()

def spectral_summary(u):
    u=finite(np.asarray(u,np.float64));centered=u-u[:,:1,:1]
    centered=centered-centered.mean(axis=(-2,-1),keepdims=True)
    f=np.fft.fft2(centered,norm='ortho');f[:,0,0]=0;flat=f.reshape(3,-1)
    ids=band_ids(*u.shape[-2:])
    power=np.stack([np.bincount(ids,weights=abs(flat[c])**2,minlength=256) for c in range(3)])
    cross=[]
    for a,c in PAIRS:
        products=flat[a]*flat[c].conj()
        cross.append(np.bincount(ids,weights=products.real,minlength=256)+1j*np.bincount(ids,weights=products.imag,minlength=256))
    variances=np.mean(centered*centered,axis=(-2,-1))
    corr=[]
    for a,c in PAIRS:
        den=np.sqrt(variances[a]*variances[c])
        corr.append(np.mean(centered[a]*centered[c])/den if den>0 else np.nan)
    return {'power':power,'cross':np.stack(cross),'variance':variances,'correlation':np.array(corr)}

def diagnose_residuals(real_u,draw_u,bins,*,synthetic_degeneracy,real_residual=None,draw_residual=None):
    # Optional residual aliases support dimensionless mathematical oracles. The
    # public image entry point below always supplies physical encoded-RGB residuals.
    real_u=finite(np.asarray(real_u,np.float64));draw_u=finite(np.asarray(draw_u,np.float64))
    if draw_u.shape!=(4,*real_u.shape):raise ValueError('Exactly four draws required')
    if synthetic_degeneracy not in ('zero_only','variance_1e-12'):raise ValueError('Synthetic degeneracy threshold needs protocol confirmation')
    result=bin_errors(real_u if real_residual is None else real_residual,draw_u if draw_residual is None else draw_residual,bins)
    result['EKS']=float(max(np.mean([ks2(real_u[c],u[c]) for u in draw_u]) for c in range(3)))
    real=spectral_summary(real_u);n=real_u.shape[-2]*real_u.shape[-1]
    rpower=real['power'];weight_den=np.array([np.sqrt(rpower[a]*rpower[c]).sum() for a,c in PAIRS])
    if np.any(real['variance']<=1e-12) or np.any(weight_den<=1e-12*n):
        return result|{'status':'STAT_UNIDENTIFIABLE','identifiable':False,'pass':False,'EPSD':None,'Ecor':None,'Ecoh':None,'Ecross':None}
    rnorm=rpower/rpower.sum(axis=1,keepdims=True)
    psd_errors=[];corr_errors=[];coh_errors=[]
    threshold=1e-12 if synthetic_degeneracy=='variance_1e-12' else 0.
    for u in draw_u:
        syn=spectral_summary(u);spower=syn['power'];degenerated=syn['variance']<=threshold
        pe=[]
        for c in range(3):
            total=spower[c].sum()
            pe.append(2. if total<=threshold*n else float(abs(spower[c]/total-rnorm[c]).sum()))
        psd_errors.append(pe)
        corr_errors.append(2. if np.any(degenerated) else float(np.max(abs(syn['correlation']-real['correlation']))))
        ce=[]
        for k,(a,c) in enumerate(PAIRS):
            rd=np.sqrt(rpower[a]*rpower[c]);sd=np.sqrt(spower[a]*spower[c]);weights=rd/weight_den[k]
            active=weights>0;diff=np.zeros(256);usable=active&(sd>1e-12)
            diff[active&~usable]=2.
            diff[usable]=abs(syn['cross'][k,usable]/sd[usable]-real['cross'][k,usable]/rd[usable])
            ce.append(float(np.sum(weights*diff)))
        coh_errors.append(ce)
    result['EPSD']=float(np.mean(psd_errors,axis=0).max())
    result['Ecor']=float(np.mean(corr_errors));result['Ecoh']=float(np.mean(coh_errors,axis=0).max())
    result['Ecross']=max(result['Ecor'],result['Ecoh'])
    passed=result['Emean']<=2*Q and result['Evar']<=.15 and result['EKS']<=.05 and result['EPSD']<=.15 and result['Ecross']<=.05
    return result|{'status':'STAT_IDENTIFIABLE','identifiable':True,'pass':bool(passed)}

def diagnose_cell(x,real_y,final_draws,bparams,*,synthetic_degeneracy):
    x=rgb(x,quantized=True);real_y=rgb(real_y,quantized=True);draws=rgb(final_draws,quantized=True)
    if real_y.shape!=x.shape or draws.shape!=(4,*x.shape):raise ValueError('Cell image identity/shape mismatch')
    mean,sigma,bins=b_fields(x,bparams)
    real_r=real_y-mean;draw_r=draws-mean
    return diagnose_residuals(real_r/sigma,draw_r/sigma,bins,synthetic_degeneracy=synthetic_degeneracy,real_residual=real_r,draw_residual=draw_r)
