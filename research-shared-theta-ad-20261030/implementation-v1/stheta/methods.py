"""Exhaustive distances and all 59 frozen method readouts; no dataset I/O."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .protocol import THETA_IDS,SHUFFLES,SPEC,digest
from .encoder import EPS

def lower_mean(x,k,axis=0):return np.sort(x,axis=axis).take(np.arange(k),axis=axis).mean(axis=axis)
def upper_mean(x,k=13,axis=0):
    x=np.asarray(x);return np.sort(x,axis=axis).take(np.arange(x.shape[axis]-k,x.shape[axis]),axis=axis).mean(axis=axis)

def permutations(ref_ids,seed,theta_ids=THETA_IDS):
    return np.array([[0]+sorted(range(1,len(theta_ids)),key=lambda t:(digest(f'stheta-v1/perm/{seed}/{rid}/{theta_ids[t]}'),t)) for rid in ref_ids],dtype=np.int64)

def permute_cache(A,permutation):
    # New shared label j contains each reference's independently permuted original view.
    return np.take_along_axis(A,permutation[None,:,:],axis=2)

def shared_scores(A,trim=204,top=13):
    d=A.min(axis=1)  # P,T
    t=int(np.argmin(lower_mean(d,trim,axis=0)))
    return float(upper_mean(d[:,t],top)),t

def original_winners(A,argq):
    """Global tie priority theta, canonical ref, row-major patch, for id and pool."""
    P,R,T=A.shape;rows=np.arange(P)
    ref0=A[:,:,0].argmin(axis=1)
    slot=A.transpose(0,2,1).reshape(P,T*R).argmin(axis=1)
    t,r=slot//R,slot%R
    return {'identity':(np.zeros(P,dtype=np.int64),ref0,argq[rows,ref0,0]),
            'pool9':(t,r,argq[rows,r,t])}

def fit_projections(bank):
    """Return float64 fit, float32 inference bases; never center inference vectors."""
    x=np.asarray(bank,dtype=np.float32);R,T,Q,D=x.shape
    total=R*(T-1)*Q;sumd=np.zeros(D,np.float64);sums=np.zeros((D,D),np.float64)
    with threadpool_limits(limits=4):
        for ref in x:
            delta=(ref[1:].astype(np.float64)-ref[0].astype(np.float64)).reshape(-1,D)
            sumd+=delta.sum(axis=0);sums+=delta.T@delta
        mu=sumd/total;S0=sums/total;Sc=S0-np.outer(mu,mu)
        bases={};evidence={}
        for name,matrix in [('S0',S0),('Sc',Sc)]:
            values,vectors=np.linalg.eigh((matrix+matrix.T)*.5)
            order=np.argsort(values,kind='stable')[::-1][:min(32,D)]
            u=vectors[:,order]
            for j in range(u.shape[1]):
                if u[np.argmax(np.abs(u[:,j])),j]<0:u[:,j]*=-1
            bases[name]=u.astype(np.float32)
            evidence[name]={'eigenvalues':values[order],'basis_float64':u,'matrix':matrix}
    return bases,{'mu':mu,'n_deltas':total,**evidence}

@torch.inference_mode()
def distance_cache(query,bank,bases=None,group_chunk=32,ranks=(8,16,32),deadline_check=None):
    """A[p,r,t] and row-major q winner. Full dot products reused for pre projections.

    bank is [R,T,Q,D]; q groups are never truncated. Every projected query is
    searched against the entire transformed identity/pool bank, not old winners.
    Returned projected winners use flattening [theta,ref,q].
    """
    assert query.dtype==bank.dtype==torch.float32
    P,D=query.shape;R,T,Q,_=bank.shape;device=query.device
    assert bank.device==device and group_chunk>=1
    flat=bank.permute(1,0,2,3).reshape(T*R*Q,D)
    A=torch.empty((P,T*R),device=device,dtype=torch.float32)
    arg=torch.empty((P,T*R),device=device,dtype=torch.int32)
    qnorm=(query*query).sum(-1,keepdim=True)
    projection={};prepared={}
    for name,u in (bases or {}).items():
        u=torch.as_tensor(u,dtype=torch.float32,device=device)
        qu=query@u;bu=flat@u
        for rank in ranks:
            if rank>u.shape[1]:continue
            key=f'{name}/r{rank}';qn=(qnorm-qu[:,:rank].square().sum(-1,keepdim=True)).clamp_min(0)
            denq=qn.sqrt().clamp_min(EPS)
            prepared[key]=(qu[:,:rank],bu[:,:rank],qn,denq)
            projection[key]={mode:(torch.full((P,),float('inf'),device=device),torch.full((P,),-1,device=device,dtype=torch.int64)) for mode in ('identity','pool9')}
    for first in range(0,T*R,group_chunk):
        if deadline_check:deadline_check()
        last=min(first+group_chunk,T*R);lo,hi=first*Q,last*Q;b=flat[lo:hi]
        bnorm=b.square().sum(-1)[None,:]
        dot=query@b.T
        dist=(qnorm+bnorm-2*dot).clamp_min(0)
        values,indices=dist.reshape(P,last-first,Q).min(-1)
        A[:,first:last]=values;arg[:,first:last]=indices.to(torch.int32)
        for key,(qu,bu,qn,denq) in prepared.items():
            pub=bu[lo:hi];bn=(bnorm-pub.square().sum(-1)[None,:]).clamp_min(0)
            denb=bn.sqrt().clamp_min(EPS)
            projected_dot=dot-qu@pub.T
            pd=(qn/denq.square()+bn/denb.square()-2*projected_dot/(denq*denb)).clamp_min(0)
            for mode in ('identity','pool9'):
                end=min(hi,R*Q) if mode=='identity' else hi
                if end<=lo:continue
                v,ix=pd[:,:end-lo].min(-1);ix=ix.to(torch.int64)+lo
                old,oldix=projection[key][mode];better=v<old
                projection[key][mode]=(torch.where(better,v,old),torch.where(better,ix,oldix))
    projected={k:{mode:{'distances':d.cpu().numpy(),'indices':i.cpu().numpy()} for mode,(d,i) in modes.items()} for k,modes in projection.items()}
    return A.reshape(P,T,R).permute(0,2,1).cpu().numpy(),arg.reshape(P,T,R).permute(0,2,1).cpu().numpy().astype(np.uint16),projected

@torch.inference_mode()
def post_projection_scores(query,bank,A,argq,bases,ranks=(8,16,32),top=13):
    scores={};winners=original_winners(A,argq)
    for mode,(t,r,q) in winners.items():
        matched=bank[torch.as_tensor(r,device=bank.device),torch.as_tensor(t,device=bank.device),torch.as_tensor(q.astype(np.int64),device=bank.device)]
        residual=query-matched
        for name,u in bases.items():
            u=torch.as_tensor(u,dtype=torch.float32,device=query.device)
            for rank in ranks:
                if rank>u.shape[1]:continue
                uk=u[:,:rank];v=residual-(residual@uk)@uk.T
                ds=v.square().sum(-1).cpu().numpy()
                scores[f'projection/{name}/r{rank}/{mode}/post_sq']=float(upper_mean(ds,top))
                scores[f'projection/{name}/r{rank}/{mode}/post_sqrt']=float(upper_mean(np.sqrt(ds),top))
    return scores

def fit_bow(bank):
    x=np.asarray(bank[:,0],dtype=np.float32).reshape(-1,bank.shape[-1])
    with threadpool_limits(limits=4):
        model=KMeans(**SPEC['reference_selection']['kmeans']).fit(x)
    centers=model.cluster_centers_.astype(np.float32)
    return centers,bow_histograms(bank,centers)

def bow_histograms(features,centers):
    features=np.asarray(features,dtype=np.float32);shape=features.shape[:-2];P,D=features.shape[-2:]
    side=int(np.sqrt(P));assert side*side==P
    flat=features.reshape(-1,D)
    labels=np.empty(len(flat),np.int32)
    for start in range(0,len(flat),8192):
        b=flat[start:start+8192]
        ds=(b*b).sum(-1)[:,None]+(centers*centers).sum(-1)[None,:]-2*b@centers.T
        labels[start:start+len(b)]=ds.argmin(-1)
    label_grid=labels.reshape(-1,P)
    yy,xx=np.meshgrid(np.arange(side),np.arange(side),indexing='ij')
    cells=((np.floor(5*(yy+.5)/side).astype(int)*5)+np.floor(5*(xx+.5)/side).astype(int)).ravel()
    hist=np.empty((len(label_grid),25,len(centers)),np.float32)
    for i,lab in enumerate(label_grid):
        h=np.bincount(cells*len(centers)+lab,minlength=25*len(centers)).reshape(25,len(centers)).astype(np.float32)
        h+=np.float32(1e-6);hist[i]=h/h.sum(-1,keepdims=True)
    return hist.reshape(*shape,25,len(centers))

def bow_costs(ref_hist,query_hist):
    kl=(ref_hist*(np.log(ref_hist)-np.log(query_hist))).sum(-1)
    return lower_mean(kl,20,axis=-1)

def selected_reference_scores(A,cost,selector,top=13):
    R,T=cost.shape;theta=cost.argmin(axis=1)
    best=cost[np.arange(R),theta]
    ordering=np.argsort(best,kind='stable')
    scores={}
    for K in (1,4,10):
        if K>R:continue
        refs=ordering[:K]
        selected=A[:,refs,theta[refs]].min(axis=1)
        pooled=A[:,refs,:].min(axis=(1,2))
        scores[f'refselect/{selector}/K{K}/selected_view']=float(upper_mean(selected,top))
        scores[f'refselect/{selector}/K{K}/pool9']=float(upper_mean(pooled,top))
    return scores

def cache_scores(A,ref_ids,query_features=None,bow_centers=None,ref_hist=None,trim=204,top=13,diagnostics=None):
    P,R,T=A.shape
    scores={'identity':float(upper_mean(A[:,:,0].min(axis=1),top)),
            'pool9':float(upper_mean(A.min(axis=(1,2)),top))}
    scores['primary'],primary_theta=shared_scores(A,trim,top)
    d=A.min(axis=1)
    control_theta=int(d.mean(axis=0).argmin())
    scores['selector/all256']=float(upper_mean(d[:,control_theta],top))
    scores['selector/min_top13']=float(upper_mean(d,top,axis=0).min())
    if diagnostics is not None:
        diagnostics.update({'primary':primary_theta,'selector/all256':control_theta,'selector/min_top13':int(upper_mean(d,top,axis=0).argmin())})
    for seed in SHUFFLES:
        shuffled=permute_cache(A,permutations(ref_ids,seed,THETA_IDS[:T]))
        assert np.array_equal(np.sort(A,axis=2),np.sort(shuffled,axis=2))
        assert np.array_equal(A[:,:,0],shuffled[:,:,0])
        assert np.array_equal(A.min(axis=(1,2)),shuffled.min(axis=(1,2)))
        scores[f'shuffle/{seed}'],theta=shared_scores(shuffled,trim,top)
        if diagnostics is not None:diagnostics[f'shuffle/{seed}']=theta
    scores.update(selected_reference_scores(A,lower_mean(A,trim,axis=0),'trimmed_chamfer',top))
    if bow_centers is not None:
        h=bow_histograms(query_features,bow_centers)
        scores.update(selected_reference_scores(A,bow_costs(ref_hist,h),'spatial_bow_KL',top))
    return scores

def all_readouts(query,bank,A,argq,projected,bases,ref_ids,centers,ref_hist,diagnostics=None):
    scores=cache_scores(A,ref_ids,query.cpu().numpy(),centers,ref_hist,diagnostics=diagnostics)
    scores.update(post_projection_scores(query,bank,A,argq,bases))
    for key,modes in projected.items():
        for mode,record in modes.items():scores[f'projection/{key}/{mode}/pre_sq']=float(upper_mean(record['distances']))
    return scores

def baseline_names(names):
    return sorted(n for n in names if n in ('identity','pool9','robust_color') or n.startswith(('refselect/','projection/')))
