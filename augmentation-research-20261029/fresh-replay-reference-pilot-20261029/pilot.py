#!/usr/bin/env python3
"""Predeclared snapshot fresh-replay reference pilot; never an A2ER reproduction."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_FAST_MATH']='0'
import argparse,copy,datetime,hashlib,json,math,resource,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED=20261029
DEADLINE=datetime.datetime.fromisoformat('2026-10-29T19:11:01+00:00').timestamp()
PROJECT=Path('/Users/lkc/Downloads/projects/RandAugment_CV_20261104')
DATA=PROJECT/'data/cifar100-official-20261029'
MEAN=torch.tensor([.5071,.4867,.4408]).view(1,3,1,1)
STD=torch.tensor([.2675,.2565,.2761]).view(1,3,1,1)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return jsonable(x.tolist())
    if isinstance(x,np.generic):return jsonable(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def savej(path,x):Path(path).write_text(json.dumps(jsonable(x),indent=2)+'\n')
def ensure_time(reserve=0):
    if time.time()+reserve>=DEADLINE:raise TimeoutError('Predeclared 19:11:01 UTC hard deadline / reserve reached')
def sync():torch.mps.synchronize()
def memory():return {'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory()}


class Block(nn.Module):
    def __init__(self,ci,co,stride=1):
        super().__init__()
        self.c1=nn.Conv2d(ci,co,3,stride,1,bias=False);self.g1=nn.GroupNorm(8,co)
        self.c2=nn.Conv2d(co,co,3,1,1,bias=False);self.g2=nn.GroupNorm(8,co)
        self.skip=nn.Identity() if ci==co and stride==1 else nn.Sequential(nn.Conv2d(ci,co,1,stride,bias=False),nn.GroupNorm(8,co))
    def forward(self,x):return F.relu(self.g2(self.c2(F.relu(self.g1(self.c1(x)))))+self.skip(x))
class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,16,3,1,1,bias=False),nn.GroupNorm(8,16),nn.ReLU())
        self.blocks=nn.Sequential(Block(16,16),Block(16,16),Block(16,32,2),Block(32,32),Block(32,64,2),Block(64,64))
        self.head=nn.Linear(64,10)
    def forward(self,x):return self.head(self.blocks(self.stem(x)).mean(dim=(2,3)))


def augment(images,seed=None,params=None):
    n=len(images)
    if params is None:
        rng=np.random.default_rng(seed)
        params=np.column_stack([rng.integers(0,9,n),rng.integers(0,9,n),rng.random(n)<.5]).astype(np.int64)
    padded=F.pad(images.float()/255,(4,4,4,4))
    crops=[]
    for i,(top,left,flip) in enumerate(params):
        a=padded[i,:,int(top):int(top)+32,int(left):int(left)+32]
        crops.append(a.flip(-1) if flip else a)
    return (torch.stack(crops)-MEAN)/STD,params
def clean(images):return (images.float()/255-MEAN)/STD
def schedule(n,steps,seed):
    rng=np.random.default_rng(seed);chunks=[];total=0
    while total<steps*32:
        order=rng.permutation(n);chunks.append(order);total+=n
    return np.concatenate(chunks)[:steps*32].reshape(steps,32)
def weighted_ids(weights,seed,k=32):
    u=np.random.default_rng(seed).random(len(weights));u=np.clip(u,1e-15,1-1e-15)
    score=np.log(weights)-np.log(-np.log(u))
    return np.argsort(-score,kind='stable')[:k]
def update_weights(w,ids,delta,q):
    d=np.asarray(delta,dtype=np.float64)+1e-5
    q=(1-len(ids)/len(w))*q+len(ids)/len(w)*np.quantile(d,.5,method='linear')
    eta=np.clip(1-np.maximum(d-q,0)/max(q,np.finfo(np.float32).tiny),0,1)
    ratio=(eta*d+(1-eta)*q)/d
    w[ids]=.5*w[ids]+.5*np.sqrt(ratio)
    return q
def rank(x):
    x=np.asarray(x);order=np.argsort(x,kind='stable');r=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        r[order[i:j]]=(i+j-1)/2;i=j
    return r
def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-15 else 0.
def spearman(a,b):return corr(rank(a),rank(b))
def tv(a,b):return float(np.abs(a/a.sum()-b/b.sum()).sum()/2)
def stats(w):
    p=w/w.sum();return {'min':float(w.min()),'max':float(w.max()),'sum':float(w.sum()),'entropy':float(-(p*np.log(p)).sum()),'ess':float(1/(p@p))}
def assign(w,score,strata,rows,random_seed=None):
    out=w.copy();rng=np.random.default_rng(random_seed) if random_seed is not None else None
    for s in np.unique(strata):
        ids=np.flatnonzero(strata==s)
        if rng is not None:out[ids]=w[rng.permutation(ids)]
        else:
            ids=ids[np.lexsort((rows[ids],-score[ids]))]
            out[ids]=np.sort(w[ids])
    return out
def invariants(native,others,strata):
    detail={}
    for name,w in others.items():
        ok=np.array_equal(np.sort(native),np.sort(w));err=0.
        for s in np.unique(strata):
            ids=strata==s
            ok=ok and np.array_equal(np.sort(native[ids]),np.sort(w[ids]))
            err=max(err,abs(native[ids].sum()/native.sum()-w[ids].sum()/w.sum()))
        detail[name]={'exact_multiset':bool(ok),'max_stratum_probability_mass_error':err,**stats(w)}
        assert ok and err<1e-12
    return detail
def cpu_copy(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return copy.deepcopy(x)


@torch.no_grad()
def predict(model,images,device,seed=None):
    model.eval();out=[];parameters=[]
    # Parameter RNG is generated across the full requested set, then sliced.
    if seed is not None:
        rng=np.random.default_rng(seed);n=len(images)
        parameters=np.column_stack([rng.integers(0,9,n),rng.integers(0,9,n),rng.random(n)<.5]).astype(np.int64)
    for a in range(0,len(images),128):
        ensure_time()
        x=clean(images[a:a+128]) if seed is None else augment(images[a:a+128],params=parameters[a:a+128])[0]
        out.append(model(x.to(device)).cpu())
    return torch.cat(out),parameters
def evaluate(model,images,labels,device):
    h,_=predict(model,images,device)
    lp=h.double().log_softmax(-1);nll=-lp[torch.arange(len(labels)),labels].numpy()
    errors=(h.argmax(-1)!=labels).numpy();prob=lp.exp()[torch.arange(len(labels)),labels].numpy()
    return {'nll':float(nll.mean()),'error_rate':float(errors.mean()),'accuracy':float(1-errors.mean()),'n':len(labels)}, {'logits':h.numpy(),'nll':nll,'error':errors,'true_prob':prob}


def preflight(out):
    from torchvision.transforms import functional as TF
    checks={};rng=np.random.default_rng(99)
    raw=torch.from_numpy(rng.integers(0,256,(3,3,32,32),dtype=np.uint8))
    pars=np.array([[0,0,0],[8,8,1],[3,5,1]])
    a,_=augment(raw,params=pars);reference=[]
    for im,(top,left,flip) in zip(raw,pars):
        z=TF.crop(TF.pad(im,4,fill=0),int(top),int(left),32,32)
        if flip:z=TF.hflip(z)
        reference.append(TF.normalize(z.float()/255,MEAN.flatten().tolist(),STD.flatten().tolist()))
    checks['augmentation_max_abs']=float((a-torch.stack(reference)).abs().max());assert checks['augmentation_max_abs']==0
    w=np.linspace(.1,1,12);s=np.repeat([0,1,2],4);rows=np.arange(12)
    invariants(w,{'A':assign(w,rows,s,rows),'R':assign(w,rows,s,rows,5)},s)
    h=rng.normal(size=(8,12,10));z=rng.normal(size=(12,10));davg=.5*((h-z)**2).sum(-1).mean(0);dmean=.5*((h.mean(0)-z)**2).sum(-1)
    checks['variance_identity_error']=float(np.max(np.abs(davg-dmean-.5*((h-h.mean(0))**2).sum(-1).mean(0))));assert checks['variance_identity_error']<1e-12
    ww=np.ones(512);q=update_weights(ww,np.arange(32),np.array([.5]*24+[8.]*8),.5)
    checks['q_known_case']=q;checks['priority_low_high']=[float(ww[0]),float(ww[31])]
    assert abs(q-.500000625)<1e-12 and abs(ww[0]-1)<1e-4 and abs(ww[31]-.625)<1e-5
    torch.manual_seed(SEED)
    c=SmallResNet();m=copy.deepcopy(c).to('mps');inp=a[:2].clone();target=torch.tensor([0,1])
    co=c(inp);mo=m(inp.to('mps'));cl=F.cross_entropy(co,target);ml=F.cross_entropy(mo,target.to('mps'));cl.backward();ml.backward();sync()
    checks['cpu_mps_logits_max_abs']=float((co.detach()-mo.detach().cpu()).abs().max())
    checks['cpu_mps_gradient_max_abs']=max(float((p.grad-q.grad.cpu()).abs().max()) for p,q in zip(c.parameters(),m.parameters()))
    checks['cpu_mps_loss_difference']=abs(float(cl.detach())-float(ml.detach().cpu()))
    assert checks['cpu_mps_logits_max_abs']<2e-4 and checks['cpu_mps_gradient_max_abs']<5e-4
    checks['parameter_count']=sum(p.numel() for p in c.parameters());checks['passed']=True
    savej(out/'preflight.json',checks);del c,m;torch.mps.empty_cache()
    return checks


def run(out):
    start=time.perf_counter();summary={'started_utc':utc(),'status':'running','device':'mps','seed':SEED,'optimizer_steps':0,'hard_deadline_utc':'2026-10-29T19:11:01+00:00','scope':'Declared snapshot fresh-replay DER + residual sampling reference; NOT original A2ER reproduction.'}
    events=(out/'progress.jsonl').open('x',buffering=1)
    losses=(out/'training-losses.jsonl').open('x',buffering=1)
    def event(phase,**values):
        r={'utc':utc(),'phase':phase,**values};events.write(json.dumps(jsonable(r))+'\n');print(json.dumps(jsonable(r)),flush=True)
    def logloss(x):losses.write(json.dumps(jsonable(x))+'\n')
    try:
        assert torch.backends.mps.is_available()
        ensure_time(60)
        summary['versions']={'torch':torch.__version__,'numpy':np.__version__,'script_sha256':digest(__file__)}
        summary['preflight']=preflight(out);event('preflight_passed',**summary['preflight'])
        trainfile=DATA/'train-only/train.bin'
        receipt=json.loads((DATA/'preparation.json').read_text());assert receipt['status']=='ready_train_only'
        assert trainfile.stat().st_size==50000*3074
        records=np.memmap(trainfile,mode='r',dtype=np.uint8,shape=(50000,3074));labels=np.asarray(records[:,1])
        fit=np.load(DATA/'splits/fit_indices.npy',allow_pickle=False);dev=np.load(DATA/'splits/development_indices.npy',allow_pickle=False)
        oldrows=fit[labels[fit]<5];newrows=fit[(labels[fit]>=5)&(labels[fit]<10)]
        od=dev[labels[dev]<5];nd=dev[(labels[dev]>=5)&(labels[dev]<10)]
        assert len(oldrows)==len(newrows)==2000 and len(od)==len(nd)==500
        assert np.intersect1d(np.concatenate([oldrows,newrows]),np.concatenate([od,nd])).size==0
        def images(rows):return torch.from_numpy(np.array(records[rows,2:],copy=True).reshape(-1,3,32,32))
        ox,nx,odx,ndx=map(images,[oldrows,newrows,od,nd])
        oy,ny,ody,ndy=[torch.from_numpy(labels[r].astype(np.int64)) for r in [oldrows,newrows,od,nd]]
        bufferpos=[]
        for c,count in enumerate([103,103,102,102,102]):
            ids=np.flatnonzero(oy.numpy()==c).tolist()
            ids.sort(key=lambda i:hashlib.sha256(f'{SEED}|buffer|{int(oldrows[i])}'.encode()).digest())
            bufferpos.extend(ids[:count])
        bufferpos=np.array(bufferpos);brows=oldrows[bufferpos];bx=ox[bufferpos];by=oy[bufferpos]
        assert len(np.unique(brows))==512 and not np.intersect1d(brows,od).size
        plan1=schedule(len(ox),600,SEED+1);plan2=schedule(len(nx),224,SEED+2)
        np.savez_compressed(out/'data-and-sequences.npz',old_fit_rows=oldrows,new_fit_rows=newrows,old_dev_rows=od,new_dev_rows=nd,buffer_rows=brows,task1_batch_positions=plan1,task2_and_branch_batch_positions=plan2)
        summary['data']={'official_archive_sha256':receipt['archive_sha256'],'train_sha256':next(x['sha256'] for x in receipt['files'] if x['path']=='train-only/train.bin'),'buffer_count':512,'buffer_class_counts':np.bincount(by.numpy(),minlength=5).tolist(),'snapshot_age_step':600,'fit_dev_disjoint':True,'official_test_read':False}
        torch.manual_seed(SEED);np.random.seed(SEED)
        model=SmallResNet().to('mps');opt=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9,weight_decay=.0005)
        initial,initialraw=evaluate(model,odx,ody,'mps');summary['initial_old_dev']=initial
        np.savez_compressed(out/'initial-old-dev.npz',**initialraw)
        step_times=[];phase_t=time.perf_counter()
        for step,idx in enumerate(plan1):
            ensure_time(20);t=time.perf_counter();model.train()
            x,_=augment(ox[idx],SEED+100000+step);y=oy[idx].to('mps')
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x.to('mps')),y);assert torch.isfinite(loss)
            loss.backward();opt.step();sync();dt=time.perf_counter()-t;step_times.append(dt);summary['optimizer_steps']+=1
            logloss({'phase':'task1','step':step+1,'ce':float(loss.detach().cpu()),'seconds':dt})
            if step==15:
                med=float(np.median(step_times[4:]));estimate=(600-16+3*(160+320))*med+120
                summary['throughput']={'first16_steps_seconds':sum(step_times),'warm_median_seconds':med,'conservative_remaining_seconds':estimate,'remaining_wall_seconds':DEADLINE-time.time()}
                event('throughput',**summary['throughput']);ensure_time(estimate)
            if (step+1)%100==0:event('task1',step=step+1,ce=float(loss.detach().cpu()),elapsed_seconds=time.perf_counter()-phase_t)
        oldlearn,raw=evaluate(model,odx,ody,'mps');np.savez_compressed(out/'task1-old-dev.npz',**raw)
        summary['task1_old_dev']=oldlearn;summary['task1_wall_seconds']=time.perf_counter()-phase_t
        learned=oldlearn['accuracy']>=.5 and oldlearn['nll']<=1.5 and initial['nll']-oldlearn['nll']>=.3
        event('task1_learning_gate',passed=learned,metrics=oldlearn)
        torch.save({'model':cpu_copy(model.state_dict()),'optimizer':cpu_copy(opt.state_dict()),'step':600},out/'task1-checkpoint.pt')
        if not learned:
            summary['status']='baseline_insufficient_task1';return summary
        z,storedparams=predict(model,bx,'mps',SEED+300000);zt=z.to('mps')
        np.savez_compressed(out/'buffer-at-capture.npz',rows=brows,labels=by.numpy(),historical_targets=z.numpy(),capture_augmentation=storedparams,capture_step=np.full(512,600))
        for group in opt.param_groups:group['lr']=.02
        w=np.ones(512,dtype=np.float64);q=0.;prefixids=[];phase_t=time.perf_counter()
        def replay_step(which,step,weights,do_update=False):
            nonlocal q
            idx=plan2[step]
            ids_d=weighted_ids(weights,SEED+400000+step*2)
            ids_c=weighted_ids(weights,SEED+400001+step*2)
            xx,_=augment(nx[idx],SEED+500000+step)
            xd,_=augment(bx[ids_d],SEED+600000+step)
            xc,_=augment(bx[ids_c],SEED+700000+step)
            model.train();opt.zero_grad(set_to_none=True)
            output=model(torch.cat([xx,xd,xc]).to('mps'));hn,hd,hc=output.split(32)
            ce=F.cross_entropy(hn,ny[idx].to('mps'));cr=F.cross_entropy(hc,by[ids_c].to('mps'))
            mse=F.mse_loss(hd,zt[torch.as_tensor(ids_d,device='mps')])
            objective=ce+.5*cr+.5*mse;assert torch.isfinite(objective)
            delta=.5*((hd.detach()-zt[torch.as_tensor(ids_d,device='mps')])**2).sum(-1).cpu().numpy()
            if do_update:q=update_weights(weights,ids_d,delta,q)
            objective.backward();opt.step();sync();summary['optimizer_steps']+=1
            logloss({'phase':which,'step':step+1,'ce_new':float(ce.detach().cpu()),'ce_old':float(cr.detach().cpu()),'mse':float(mse.detach().cpu()),'objective':float(objective.detach().cpu()),'q':q})
            return ids_d,ids_c
        for step in range(160):
            ensure_time(20);ids_d,ids_c=replay_step('task2_prefix',step,w,True);prefixids.append([ids_d,ids_c])
            if (step+1)%40==0:event('task2_prefix',step=step+1,q=q,weights=stats(w),elapsed_seconds=time.perf_counter()-phase_t)
        np.savez_compressed(out/'prefix-replay-history.npz',buffer_positions=np.asarray(prefixids),weights=w,q=q)
        baselineold,rawold=evaluate(model,odx,ody,'mps');baselinenew,rawnew=evaluate(model,ndx,ndy,'mps')
        np.savez_compressed(out/'branchpoint-dev.npz',old_logits=rawold['logits'],old_nll=rawold['nll'],old_error=rawold['error'],new_logits=rawnew['logits'],new_nll=rawnew['nll'],new_error=rawnew['error'])
        summary['branchpoint']={'old':baselineold,'new':baselinenew,'q':q,'weights':stats(w),'task2_wall_seconds':time.perf_counter()-phase_t}
        checkpoint={'model':cpu_copy(model.state_dict()),'optimizer':cpu_copy(opt.state_dict()),'step':760,'buffer_rows':brows,'historical_targets':z,'weights':w.copy(),'q':q}
        torch.save(checkpoint,out/'common-checkpoint.pt')
        baselineok=all(m['accuracy']>=.35 and m['nll']<=1.8 for m in [baselineold,baselinenew])
        event('branchpoint_learning_gate',passed=baselineok,**summary['branchpoint'])
        if not baselineok:
            summary['status']='baseline_insufficient_branchpoint';return summary
        # Frozen diagnostic phase. No threshold or target update occurs.
        ensure_time(60);model.eval();q_frozen=q;clean_logits,_=predict(model,bx,'mps')
        clean_prob=clean_logits.double().softmax(-1).numpy();true_conf=clean_prob[np.arange(512),by.numpy()]
        strata=by.numpy()*2+(true_conf>=.5);cohorts=np.zeros(512,dtype=int)
        sizes={int(s):int((strata==s).sum()) for s in np.unique(strata)}
        exchangeable=sum(n for n in sizes.values() if n>=2)/512
        groups=[];diagnostics=[];scores=[]
        clean_entropy=-(clean_prob*np.log(clean_prob+1e-300)).sum(-1)
        for group in range(2):
            hs=[];pars=[]
            for view in range(8):
                h,pa=predict(model,bx,'mps',SEED+800000+group*100+view);hs.append(h.numpy());pars.append(pa)
            h=np.asarray(hs,dtype=np.float64);zz=z.numpy().astype(np.float64);mean=h.mean(0)
            davg=.5*((h-zz)**2).sum(-1).mean(0);dmean=.5*((mean-zz)**2).sum(-1)
            var=.5*((h-mean)**2).sum(-1).mean(0)
            assert np.max(np.abs(davg-dmean-var))<1e-10
            hc=h-h.mean(-1,keepdims=True);zc=zz-zz.mean(-1,keepdims=True)
            da_c=.5*((hc-zc)**2).sum(-1).mean(0);dm_c=.5*((hc.mean(0)-zc)**2).sum(-1)
            trans=.5*10*((h.mean(-1)-zz.mean(-1))**2).mean(0)
            assert np.max(np.abs(davg-da_c-trans))<1e-10
            A=assign(w,davg,strata,brows);B=assign(w,dmean,strata,brows)
            Ac=assign(w,da_c,strata,brows);Bc=assign(w,dm_c,strata,brows)
            inv=invariants(w,{'A':A,'B':B,'A_centered':Ac,'B_centered':Bc},strata)
            delta_p=B/B.sum()-A/A.sum()
            norms=np.linalg.norm(mean,axis=-1);znorms=np.linalg.norm(zz,axis=-1)
            norm_residual=.5*((np.linalg.norm(h,axis=-1)-znorms)**2).mean(0)
            cov=np.column_stack([np.ones(512),norms,znorms,true_conf,clean_entropy,mean.mean(-1)])
            coef=np.linalg.lstsq(cov,delta_p,rcond=None)[0];fitted=cov@coef
            denom=((delta_p-delta_p.mean())**2).sum();r2=1-((delta_p-fitted)**2).sum()/denom if denom>1e-30 else 0.
            dg={'TV_A_B':tv(A,B),'TV_centered_A_B':tv(Ac,Bc),'variance_mean':float(var.mean()),'variance_max':float(var.max()),'translation_residual_fraction':float(trans.sum()/max(davg.sum(),1e-30)),
                'variance_norm_spearman':spearman(var,norms),'variance_confidence_spearman':spearman(var,true_conf),'variance_entropy_spearman':spearman(var,clean_entropy),'identity_delta_covariate_OLS_R2':float(r2),'weight_invariants':inv}
            diagnostics.append(dg);groups.append({'A':A,'B':B});scores.append((davg,dmean))
            np.savez_compressed(out/f'score-views-{group+1}.npz',logits=h,augmentation_parameters=np.asarray(pars),Davg=davg,Dmean=dmean,half_view_variance=var,Davg_centered=da_c,Dmean_centered=dm_c,translation_component=trans,norm_residual=norm_residual,A=A,B=B,clean_logits=clean_logits.numpy(),confidence=true_conf,entropy=clean_entropy,strata=strata,age_cohort=cohorts)
            event('gate_A_view_set',set=group+1,diagnostics=dg)
        da=groups[0]['B']/w.sum()-groups[0]['A']/w.sum();db=groups[1]['B']/w.sum()-groups[1]['A']/w.sum()
        denom=np.linalg.norm(da)*np.linalg.norm(db);cos=float(da@db/denom) if denom>1e-30 else 0.
        sa=spearman(scores[0][0],scores[1][0]);sb=spearman(scores[0][1],scores[1][1])
        passed=exchangeable>=.8 and all(d['TV_A_B']>=.05 for d in diagnostics) and sa>=.8 and sb>=.8 and cos>=.5
        assert q==q_frozen and np.array_equal(z.numpy(),checkpoint['historical_targets'].numpy())
        summary['gate_A']={'passed':passed,'exchangeable_fraction':exchangeable,'stratum_sizes':sizes,'Davg_spearman':sa,'Dmean_spearman':sb,'signed_identity_difference_cosine':cos,'sets':diagnostics,'q_unchanged':True}
        savej(out/'gate-A.json',summary['gate_A']);event('gate_A',passed=passed,spearman_Davg=sa,spearman_Dmean=sb,identity_cosine=cos,TVs=[d['TV_A_B'] for d in diagnostics])
        if not passed:
            summary['status']='gate_A_not_met_no_branches';return summary
        # Branches: only identity assignment differs among N/A/B/R.
        armweights={'N':w.copy(),'A':groups[0]['A'],'B':groups[0]['B'],'R':assign(w,np.zeros(512),strata,brows,SEED+900000),'U':np.ones(512)}
        invariants(w,{k:v for k,v in armweights.items() if k!='U'},strata)
        np.savez_compressed(out/'branch-fixed-weights.npz',**armweights)
        arms={};all_exposures={}
        for arm in ['N','A','B','R','U']:
            ensure_time(60)
            model.load_state_dict(checkpoint['model']);opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
            q=checkpoint['q'];exposures=[];armt=time.perf_counter()
            for step in range(160,224):
                ensure_time(15);exposures.append(replay_step(arm,step,armweights[arm],False))
            aold,ro=evaluate(model,odx,ody,'mps');anew,rn=evaluate(model,ndx,ndy,'mps')
            np.savez_compressed(out/f'branch-{arm}-outputs.npz',old_logits=ro['logits'],old_nll=ro['nll'],old_error=ro['error'],new_logits=rn['logits'],new_nll=rn['nll'],new_error=rn['error'],replay_buffer_positions=np.asarray(exposures))
            arms[arm]={'old':aold,'new':anew,'wall_seconds':time.perf_counter()-armt,'weights':stats(armweights[arm]),'distinct_replay_sources':len(np.unique(exposures))};all_exposures[arm]=np.asarray(exposures)
            assert q==checkpoint['q']
            event('branch_complete',arm=arm,**arms[arm])
        comparisons={k:{'B_old_NLL_improvement':arms[k]['old']['nll']-arms['B']['old']['nll'],'B_new_NLL_worsening':arms['B']['new']['nll']-arms[k]['new']['nll']} for k in ['N','A','R','U']}
        scientific_gate=all(v['B_old_NLL_improvement']>=.01 and v['B_new_NLL_worsening']<=.01 for v in comparisons.values())
        summary['gate_B']={'passed':scientific_gate,'arms':arms,'comparisons':comparisons,'source_exposure_position_agreement_to_N':{k:float((all_exposures[k]==all_exposures['N']).mean()) for k in arms},'second_seed_run':False}
        summary['status']='single_seed_development_gate_passed' if scientific_gate else 'gate_B_not_met'
        savej(out/'gate-B.json',summary['gate_B']);return summary
    except Exception as e:
        summary['status']='execution_unresolved_time' if isinstance(e,TimeoutError) else 'execution_error'
        summary['error']=f'{type(e).__name__}: {e}';summary['traceback']=traceback.format_exc()
        event('stopped',status=summary['status'],error=summary['error'])
        return summary
    finally:
        summary['finished_utc']=utc();summary['execution_wall_seconds']=time.perf_counter()-start
        summary['memory']=memory();savej(out/'summary.json',summary)
        event('finished',status=summary['status'],optimizer_steps=summary['optimizer_steps'],wall_seconds=summary['execution_wall_seconds'])
        events.close();losses.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=run(args.output)
    print(json.dumps(jsonable(result),indent=2),flush=True)
