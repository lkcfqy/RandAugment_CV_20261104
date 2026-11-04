"""Train exactly two common continuation checkpoints; never forks online arms."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_FAST_MATH']='0'
import sys,time,random,traceback,argparse
from datetime import datetime,timezone
from ctrla_adapter import *
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

official_transforms=module('official_ctrla_transforms','transforms_utils.py')

def event(kind,**items):
    obj={'utc':datetime.now(timezone.utc).isoformat(),'event':kind,**items}
    with (ROOT/'logs'/'training-events.jsonl').open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
    print(json.dumps(obj,allow_nan=False),flush=True)

def worker_seed(worker):
    value=torch.initial_seed()%2**32;random.seed(value);np.random.seed(value);torch.set_num_threads(1)

class TrainDataset(Dataset):
    def __init__(self,gamma,alpha,mean,std):
        self.indices=np.load(ROOT/'data'/'fit_indices.npy');self.images=None;self.labels=None
        aug=official_aug.ControlAugment(Naugs=5,gamma=gamma,skew=alpha,interpolation=InterpolationMode.BILINEAR)
        self.transform=official_transforms.aug_pipeline(aug,'cifar10','modified',mean,std,0)
    def __len__(self):return 2*len(self.indices)
    def __getitem__(self,index):
        if self.images is None:self.images=np.load(ROOT/'data'/'train_images.npy',mmap_mode='r');self.labels=np.load(ROOT/'data'/'train_labels.npy',mmap_mode='r')
        original=self.indices[index%len(self.indices)]
        x=TF.to_tensor(Image.fromarray(self.images[original]))
        if index>=len(self.indices):x=TF.hflip(x)
        return self.transform(x),int(self.labels[original])

class RORDataset(Dataset):
    def __init__(self,role='controller'):
        self.indices=np.load(ROOT/'data'/f'{role}_indices.npy');self.images=None;self.labels=None
    def __len__(self):return 151*len(self.indices)
    def __getitem__(self,index):
        if self.images is None:self.images=np.load(ROOT/'data'/'train_images.npy',mmap_mode='r');self.labels=np.load(ROOT/'data'/'train_labels.npy',mmap_mode='r')
        view,local=divmod(index,len(self.indices));original=self.indices[local]
        img=Image.fromarray(self.images[original])
        if view:
            op,strength=divmod(view-1,10);m=(strength+1)/10
            if SIGNED[op]:m*=(-1)**local
            img=original_op(img,OPS[op],m)
        return tensor_image(img),int(self.labels[original])

def loader(dataset,shuffle=False,seed=0,workers=6):
    return DataLoader(dataset,batch_size=125,shuffle=shuffle,drop_last=shuffle,num_workers=workers,pin_memory=False,persistent_workers=False,worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(seed),multiprocessing_context='spawn' if workers else None)

def predict_loader(model,dl,mean,std,budget):
    preds=[];losses=[];targets=[];model.eval()
    with torch.inference_mode():
        for x,y in dl:
            budget.check(inference=len(y));z=model(normalize(x.to('mps'),mean,std));
            loss=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none')
            preds.append(z.argmax(1).cpu().numpy());losses.append(loss.cpu().numpy());targets.append(y.numpy())
    budget.save();return np.concatenate(preds),np.concatenate(losses),np.concatenate(targets)

def clean_eval(model,mean,std,budget):
    # First1000 entries of RORDataset are exactly raw clean controller images.
    ds=torch.utils.data.Subset(RORDataset(),range(1000))
    pred,loss,y=predict_loader(model,loader(ds,workers=0),mean,std,budget)
    return {'accuracy':float(np.mean(pred==y)),'ce':float(np.mean(loss))}

def ror_eval(model,mean,std,budget,seed,epoch,role='controller'):
    t=time.monotonic();pred,loss,y=predict_loader(model,loader(RORDataset(role)),mean,std,budget)
    n=1000 if role=='controller' else 2000
    acc=(pred==y).reshape(151,n).mean(1)
    np.savez_compressed(ROOT/'raw'/f'seed{seed}-epoch{epoch}-{role}-official-ror.npz',prediction=pred.reshape(151,n),loss=loss.reshape(151,n),target=y[:n],accuracy=acc)
    if acc[0]<.10:raise RuntimeError('Official ROR clean denominator <0.10')
    event('ror_complete',seed=seed,epoch=epoch,role=role,image_forwards=151*n,elapsed_seconds=time.monotonic()-t,clean_accuracy=float(acc[0]))
    return acc

def fit_statistics():
    p=ROOT/'normalization.json'
    if p.exists():return json.loads(p.read_text())
    x=np.load(ROOT/'data'/'train_images.npy',mmap_mode='r');idx=np.load(ROOT/'data'/'fit_indices.npy');s=np.zeros(3);ss=np.zeros(3);n=0
    for batch in np.array_split(idx,92):
        a=x[batch].astype(np.float64)/255;s+=a.sum((0,1,2));ss+=(a*a).sum((0,1,2));n+=len(batch)*1024
    mean=s/n;std=np.sqrt(ss/n-mean**2);result={'mean':mean.tolist(),'std':std.tolist(),'population':'fit46000 only; horizontal duplication leaves these population moments unchanged','dtype_accumulation':'float64'}
    write_json(p,result);return result

def checkpoint(model,opt,seed,epoch,gamma,alpha,xi,history,mean,std,initial,budget,phase_update_done):
    path=ROOT/'weights'/f'seed{seed}-epoch{epoch:02d}.pt'
    state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict() if opt else None,'scheduler':{'type':'pinned_explicit_cosine','horizon_actual_epochs':20,'last_epoch':epoch,'next_epoch_lr':lr_at(epoch+1),'recreate_optimizer_next_epoch':True},'seed':seed,'actual_epoch':epoch,'train_images':epoch*92000,'Gamma':gamma,'alpha':alpha,'xi':xi,'history':history,'mean':mean,'std':std,'initial_controller':initial,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'mps':torch.mps.get_rng_state()},'phase_update_done':phase_update_done,'protocol_sha256':sha256(ROOT/'protocol.json'),'sampling_manifest_sha256':sha256(ROOT/'sampling-manifest.json'),'budget':budget.state,'source_commit':'e77a4c19a3db893c120fc18b774fa290e826592b','batch_cursor':None,'resume_scope':'epoch boundary; optimizer intentionally reset next epoch; future stage not authorized'}
    # Optimizer tensors also moved to CPU for a portable, complete checkpoint.
    if state['optimizer']:
        for slot in state['optimizer']['state'].values():
            for k,v in slot.items():
                if torch.is_tensor(v):slot[k]=v.detach().cpu()
    tmp=path.with_suffix('.pt.tmp');torch.save(state,tmp);os.replace(tmp,path)
    write_json(ROOT/f'seed{seed}-latest-checkpoint.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'actual_epoch':epoch,'train_images':epoch*92000,'phase_update_done':phase_update_done})
    budget.save()

def train_seed(seed,budget,mean,std):
    if (ROOT/f'seed{seed}-training-summary.json').exists():raise RuntimeError('Completed seed exists; refuse rerun')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.mps.manual_seed(seed)
    model=make_model();initial=clean_eval(model,mean,std,budget)
    gamma=[0.]*15;alpha=[0.]*15;xi=.9;history=[];opt=None
    event('seed_start',seed=seed,initial_controller=initial)
    checkpoint(model,opt,seed,0,gamma,alpha,xi,history,mean,std,initial,budget,False)
    for epoch in range(1,11):
        budget.check();dataset=TrainDataset(gamma,alpha,mean,std)
        dl=loader(dataset,shuffle=True,seed=seed*1000+epoch)
        opt=new_optimizer(model,epoch);model.train();loss_sum=0.;correct=0;count=0;t=time.monotonic()
        for batch,(x,y) in enumerate(dl):
            budget.check(train=len(y));x=x.to('mps');y=y.to('mps')
            opt.zero_grad(set_to_none=True);z=model(x);loss=torch.nn.functional.cross_entropy(z,y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
            loss.backward();opt.step();loss_sum+=float(loss.detach().cpu())*len(y);correct+=int((z.argmax(1)==y).sum().cpu());count+=len(y)
            if (batch+1)%100==0:
                budget.save();event('batch_health',seed=seed,epoch=epoch,batch=batch+1,images=count,ce=loss_sum/count,accuracy=correct/count,elapsed_seconds=time.monotonic()-t)
        torch.mps.synchronize();train_seconds=time.monotonic()-t
        assert count==92000
        val=clean_eval(model,mean,std,budget)
        row={'epoch':epoch,'train_ce':loss_sum/count,'train_accuracy':correct/count,'controller_ce':val['ce'],'controller_accuracy':val['accuracy'],'lr':lr_at(epoch),'train_images':count,'train_seconds':train_seconds,'Gamma_used':list(gamma),'alpha_used':list(alpha),'xi_used':xi}
        history.append(row);event('epoch_complete',seed=seed,**row)
        checkpoint(model,opt,seed,epoch,gamma,alpha,xi,history,mean,std,initial,budget,False)
        if epoch==5 and val['accuracy']<.20 and val['ce']>=initial['ce']:raise RuntimeError('LEARNING_HEALTH_FAILED_AT_EPOCH5')
        if epoch==10 and (val['accuracy']<.35 or val['ce']>.90*initial['ce']):raise RuntimeError('LEARNING_HEALTH_FAILED_AT_EPOCH10')
        if epoch%5==0:
            phase=epoch//5;recent=history[-5:]
            kappa=float(np.mean([r['train_ce'] for r in recent])/np.mean([r['controller_ce'] for r in recent]))
            xi,delta=update_xi(xi,kappa,phase)
            acc=ror_eval(model,mean,std,budget,seed,epoch)
            gamma,alpha=official_ctrl.get_ASD(acc,15,xi,10,gamma,alpha)
            gamma=list(map(float,gamma));alpha=list(map(float,alpha))
            if not all(np.isfinite(gamma)) or not all(np.isfinite(alpha)):raise RuntimeError('Nonfinite ASD')
            row.update(kappa_phase=kappa,xi_next=xi,delta_xi=delta,Gamma_next=gamma,alpha_next=alpha)
            event('phase_update',seed=seed,epoch=epoch,kappa=kappa,xi=xi,Gamma=gamma,alpha=alpha)
            checkpoint(model,opt,seed,epoch,gamma,alpha,xi,history,mean,std,initial,budget,True)
    summary={'seed':seed,'status':'common_checkpoint_complete','actual_epochs':10,'fit_exposures':20,'train_images':920000,'initial_controller':initial,'final_controller':val,'history':history,'Gamma':gamma,'alpha':alpha,'xi':xi,'online_branch_started':False}
    write_json(ROOT/f'seed{seed}-training-summary.json',summary)
    del model,opt;torch.mps.empty_cache();event('seed_complete',seed=seed,final_controller=val)

def main():
    assert json.loads((ROOT/'benchmark.json').read_text())['status']=='passed'
    assert (ROOT/'asset-manifest.json').exists()
    torch.set_num_threads(6);budget=Budget();stats=fit_statistics()
    try:
        for seed in (0,1):train_seed(seed,budget,stats['mean'],stats['std'])
        write_json(ROOT/'common-stage-status.json',{'status':'complete','seeds':[0,1],'online_branch_started':False})
    except Exception as e:
        budget.save();write_json(ROOT/'common-stage-status.json',{'status':'stopped_indeterminate','error':str(e),'traceback':traceback.format_exc(),'no_mechanistic_negative_claim':True});raise

if __name__=='__main__':main()
