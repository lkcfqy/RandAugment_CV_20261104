#!/usr/bin/env python3
"""One fixed, source-backed Mammoth DER++ subset baseline. No candidate gates."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_FAST_MATH']='0'
import ast,copy,datetime,hashlib,json,math,resource,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
DATA=PROJECT/'data/cifar100-official-20261029'
SEED=20261029
MEAN=(.5071,.4867,.4408);STD=(.2675,.2565,.2761)
torch.set_num_threads(4);torch.set_num_interop_threads(1)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def savej(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sync():torch.mps.synchronize()
def memory():return {'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory()}
def cpu_copy(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return copy.deepcopy(x)


def source(path):return ROOT/'sources'/path.replace('/','__')
def get_classes():
    ns={'torch':torch,'nn':nn,'F':F,'relu':F.relu,'avg_pool2d':F.avg_pool2d,'List':List,'np':np}
    tree=ast.parse(source('backbone/__init__.py').read_text())
    base=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MammothBackbone')
    base.body=[x for x in base.body if isinstance(x,ast.FunctionDef) and x.name in {'__init__','to'}]
    exec(compile(ast.Module(body=[base],type_ignores=[]),str(source('backbone/__init__.py')),'exec'),ns)
    tree=ast.parse(source('backbone/ResNetBlock.py').read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='conv3x3' or isinstance(x,ast.ClassDef) and x.name in {'BasicBlock','ResNet'}]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source('backbone/ResNetBlock.py')),'exec'),ns)
    tree=ast.parse(source('utils/buffer.py').read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in {'BaseSampleSelection','ReservoirSampling'}]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source('utils/buffer.py')),'exec'),ns)
    return ns
CLASSES=get_classes()
def network():return CLASSES['ResNet'](CLASSES['BasicBlock'],[2,2,2,2],10,64)
def train_transform():return T.Compose([T.ToPILImage(),T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(MEAN,STD)])
def clean_batch(x):return TF.normalize(x.float()/255,MEAN,STD)


class AuditBuffer:
    """Same used Buffer semantics, extra row/arrival metadata; CPU original images."""
    def __init__(self):
        self.capacity=500;self.seen=0;self.active=0
        self.examples=torch.zeros(500,3,32,32,dtype=torch.float32)
        self.labels=torch.zeros(500,dtype=torch.int64);self.logits=torch.zeros(500,10,dtype=torch.float32)
        self.rows=np.full(500,-1,np.int64);self.arrivals=np.full(500,-1,np.int64)
        self.selector=CLASSES['ReservoirSampling'](500,'cpu')
        self.draws=[];self.admitted=[];self.incoming_rows=None;self.expected_logits=None
    def is_empty(self):return self.seen==0
    def get_data(self,size,transform,device):
        ids=np.random.choice(self.active,size=min(size,self.active),replace=False)
        self.draws.append({'slots':ids.tolist(),'source_rows':self.rows[ids].tolist(),'arrival_indices':self.arrivals[ids].tolist()})
        x=torch.stack([transform(self.examples[int(i)]) for i in ids]).to(device)
        return x,self.labels[ids].to(device),self.logits[ids].to(device)
    def add_data(self,examples,labels,logits):
        ex=examples.detach().cpu();lab=labels.detach().cpu();log=logits.detach().cpu()
        assert len(ex)==len(self.incoming_rows)
        if self.expected_logits is not None:assert torch.equal(log,self.expected_logits)
        self.admitted=[]
        for i in range(len(ex)):
            slot=self.selector(self.seen);self.seen+=1
            if slot>=0:
                self.examples[slot]=ex[i];self.labels[slot]=lab[i];self.logits[slot]=log[i]
                self.rows[slot]=self.incoming_rows[i];self.arrivals[slot]=self.seen-1
                self.admitted.append({'slot':int(slot),'row':int(self.incoming_rows[i]),'arrival_index':self.seen-1})
        self.active=min(self.capacity,self.seen)
    def snapshot(self):return {'examples':self.examples.clone(),'labels':self.labels.clone(),'logits':self.logits.clone(),'rows':self.rows.copy(),'arrivals':self.arrivals.copy(),'seen':self.seen,'active':self.active}


def make_observe(adapter):
    tree=ast.parse(source('models/derpp.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Derpp')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='observe')
    def mse(a,b):
        value=F.mse_loss(a,b);adapter.mse_values.append(value.detach().clone());return value
    ns={'F':SimpleNamespace(mse_loss=mse)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source('models/derpp.py')),'exec'),ns)
    return ns['observe']


@torch.no_grad()
def evaluate(model,images,labels):
    model.eval();hs=[]
    for a in range(0,len(images),64):hs.append(model(clean_batch(images[a:a+64]).to('mps')).cpu())
    sync();h=torch.cat(hs);lp=h.double().log_softmax(-1)
    nll=-lp[torch.arange(len(labels)),labels].numpy();pred=h.argmax(-1);err=(pred!=labels).numpy()
    metric={'n':len(labels),'nll':float(nll.mean()),'accuracy':float(1-err.mean()),'error_rate':float(err.mean()),
            'per_class':{str(int(c)):{'n':int((labels==c).sum()),'accuracy':float((pred[labels==c]==c).float().mean()),'nll':float(nll[(labels==c).numpy()].mean())} for c in labels.unique()}}
    return metric,{'logits':h.numpy(),'nll':nll,'error':err,'labels':labels.numpy()}


def preflight(out):
    started=time.perf_counter();torch.manual_seed(SEED+9000)
    c=network();m=copy.deepcopy(c).to('mps');x=torch.randn(2,3,32,32);y=torch.tensor([0,9])
    hc=c(x);hm=m(x.to('mps'));lc=F.cross_entropy(hc,y);lm=F.cross_entropy(hm,y.to('mps'))
    lc.backward();lm.backward();sync()
    difflog=float((hc.detach()-hm.detach().cpu()).abs().max())
    diffg=max(float((a.grad-b.grad.cpu()).abs().max()) for a,b in zip(c.parameters(),m.parameters()))
    statsdiff=max(float((a.detach()-m.state_dict()[n].cpu()).abs().max()) for n,a in c.state_dict().items() if 'running_' in n)
    # Actual PIL roundtrip used by standard replay; all 256 byte values checked.
    raw=torch.arange(256,dtype=torch.uint8).reshape(1,16,16).repeat(3,2,2)
    roundtrip=TF.pil_to_tensor(TF.to_pil_image(raw.float()/255))
    assert torch.equal(raw,roundtrip)
    expected_params=11173962;count=sum(a.numel() for a in c.parameters());bn=sum(isinstance(a,nn.BatchNorm2d) for a in c.modules())
    assert count==expected_params and bn==20 and difflog<2e-4 and diffg<2e-3 and statsdiff<2e-4
    result={'passed':True,'parameters':count,'batchnorm_layers':bn,'cpu_mps_logit_max_abs':difflog,'cpu_mps_gradient_max_abs':diffg,'cpu_mps_BN_running_state_max_abs':statsdiff,'raw_float_PIL_byte_roundtrip_exact':True,'optimizer_steps':0,'wall_seconds':time.perf_counter()-started}
    savej(out/'preflight.json',result);del c,m,hc,hm,lc,lm;torch.mps.empty_cache();return result


def resource_guard():
    r=subprocess.run(['memory_pressure','-Q'],capture_output=True,text=True,timeout=10)
    import re
    match=re.search(r'free percentage:\s*(\d+)%',r.stdout)
    free=int(match.group(1)) if match else None
    ps=subprocess.run(['ps','-axo','pid=,pcpu=,pmem=,comm='],capture_output=True,text=True,timeout=10)
    conflicts=[]
    for line in ps.stdout.splitlines():
        a=line.strip().split(None,3)
        if len(a)!=4:continue
        pid,cpu,mem,name=a;name=Path(name).name
        if int(pid) not in {os.getpid(),os.getppid()} and 'python' in name.lower() and float(cpu)>=50 and float(mem)>=1:
            conflicts.append({'pid':int(pid),'cpu_percent':float(cpu),'memory_percent':float(mem),'executable':name})
    result={'free_percent':free,'other_heavy_python':conflicts,'process_query_exit_code':ps.returncode}
    if free is not None and free<15:raise RuntimeError(f'Memory resource guard: {result}')
    if conflicts:raise RuntimeError(f'Concurrent training resource guard: {result}')
    return result


def main(out):
    out.mkdir(exist_ok=False);started=time.perf_counter();wall_start=time.time();deadline=wall_start+5400
    result={'status':'running','started_utc':utc(),'seed':SEED,'scope':'Independent official-preset DER++ two-task TRAIN/dev adaptation; no candidate gates','optimizer_hook_steps':0,'versions':{'python':os.sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__},'script_sha256':sha(__file__)}
    events=(out/'progress.jsonl').open('x',buffering=1);steps=(out/'steps.jsonl').open('x',buffering=1)
    def event(phase,**kw):
        r={'utc':utc(),'phase':phase,**kw};events.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
    net=None;opt=None;buffer=None;scheduler=None
    def checkpoint(name,task,epoch):
        torch.save({'model':cpu_copy(net.state_dict()),'optimizer':cpu_copy(opt.state_dict()),'scheduler':scheduler.state_dict(),'buffer':buffer.snapshot(),'task':task,'completed_epoch':epoch,'optimizer_hook_steps':result['optimizer_hook_steps'],'numpy_rng_state':np.random.get_state(),'torch_rng_state':torch.get_rng_state(),'mps_rng_state':torch.mps.get_rng_state()},out/name)
    try:
        assert torch.backends.mps.is_available();result['initial_resources']=resource_guard()
        frozen=json.loads((ROOT/'protocol-freeze-v2.json').read_text());assert all(sha(ROOT/n)==h for n,h in frozen['files'].items())
        manifest=json.loads((ROOT/'sources/manifest.json').read_text());assert all(sha(source(x['path']))==x['sha256'] for x in manifest['files'])
        result['preflight']=preflight(out);event('preflight',**result['preflight'])
        receipt=json.loads((DATA/'preparation.json').read_text());train=DATA/'train-only/train.bin'
        expected=next(x['sha256'] for x in receipt['files'] if x['path']=='train-only/train.bin');assert sha(train)==expected
        rec=np.memmap(train,mode='r',dtype=np.uint8,shape=(50000,3074));all_labels=np.asarray(rec[:,1])
        fit=np.load(DATA/'splits/fit_indices.npy',allow_pickle=False);dev=np.load(DATA/'splits/development_indices.npy',allow_pickle=False)
        fitrows=[fit[(all_labels[fit]>=lo)&(all_labels[fit]<lo+5)] for lo in [0,5]]
        devrows=[dev[(all_labels[dev]>=lo)&(all_labels[dev]<lo+5)] for lo in [0,5]]
        assert all(len(x)==2000 for x in fitrows) and all(len(x)==500 for x in devrows)
        assert not np.intersect1d(np.concatenate(fitrows),np.concatenate(devrows)).size
        def imgs(rows):return torch.from_numpy(np.array(rec[rows,2:],copy=True).reshape(-1,3,32,32))
        xs=[imgs(r) for r in fitrows];ys=[torch.tensor(all_labels[r].astype(np.int64)) for r in fitrows]
        dx=[imgs(r) for r in devrows];dy=[torch.tensor(all_labels[r].astype(np.int64)) for r in devrows]
        plans=np.stack([np.stack([np.random.default_rng(SEED+task*1000+epoch).permutation(2000) for epoch in range(50)]) for task in range(2)])
        np.savez_compressed(out/'data-and-order-plan.npz',old_fit=fitrows[0],new_fit=fitrows[1],old_dev=devrows[0],new_dev=devrows[1],permutations=plans)
        result['data']={'train_sha256':expected,'fit_per_task':2000,'dev_per_task':500,'head_classes':10,'buffer_size':500,'test_read':False,'class_names':['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle']}
        torch.manual_seed(SEED);np.random.seed(SEED);torch.mps.manual_seed(SEED)
        net=network().to('mps');buffer=AuditBuffer();transform=train_transform()
        adapter=SimpleNamespace(net=net,buffer=buffer,device=torch.device('mps'),transform=transform,args=SimpleNamespace(alpha=.3,beta=.5,minibatch_size=32),ce_values=[],mse_values=[])
        def lossfunc(h,y):
            l=F.cross_entropy(h,y);adapter.ce_values.append(l.detach().clone());return l
        adapter.loss=lossfunc;observe=make_observe(adapter)
        # First forward logits are captured only for explicit storage-order checks.
        capture=[False];captured=[]
        def capture_hook(module,args,output):
            if capture[0] and not captured:captured.append(output.detach().cpu().clone())
        hook=net.register_forward_hook(capture_hook)
        initial,raw=evaluate(net,dx[0],dy[0]);result['initial_old_dev']=initial
        np.savez_compressed(out/'initial-old-dev.npz',**raw)
        global_current=np.zeros(50000,np.int32);global_replay=np.zeros((2,50000),np.int32)
        epoch_results=[];task_evaluations=[];first_times=[]
        for task in range(2):
            opt=torch.optim.SGD(net.parameters(),lr=.03,momentum=0,weight_decay=0,nesterov=False)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[35,45],gamma=.2);adapter.opt=opt
            def on_step(*args):result['optimizer_hook_steps']+=1
            step_hook=opt.register_step_post_hook(on_step);task_start=time.perf_counter()
            for epoch in range(50):
                if time.time()>deadline:raise TimeoutError('90-minute execution budget reached')
                epoch_start=time.perf_counter();lr=opt.param_groups[0]['lr'];expected_lr=.03 if epoch<35 else .006 if epoch<45 else .0012
                assert abs(lr-expected_lr)<1e-12
                order=plans[task,epoch];lossvals=[];cevals=[];msevals=[];repvals=[];epoch_current=0
                for batch,startidx in enumerate(range(0,2000,32)):
                    if time.time()>deadline:raise TimeoutError('90-minute execution budget reached')
                    t=time.perf_counter();idx=order[startidx:startidx+32];rows=fitrows[task][idx]
                    rawx=xs[task][idx];x=torch.stack([transform(im) for im in rawx]).to('mps');y=ys[task][idx].to('mps')
                    not_aug=rawx.float()/255
                    adapter.ce_values=[];adapter.mse_values=[];buffer.draws=[];buffer.incoming_rows=rows
                    capture[0]=result['optimizer_hook_steps']<2;captured.clear();buffer.expected_logits=None
                    # add_data is called only after this forward, so link expected
                    # storage check through a narrow wrapper for the first steps.
                    original_add=buffer.add_data
                    if capture[0]:
                        def checked_add(*args,**kw):
                            assert len(captured)==1;buffer.expected_logits=captured[0];return original_add(*args,**kw)
                        buffer.add_data=checked_add
                    net.train();before=result['optimizer_hook_steps']
                    objective=observe(adapter,x,y,not_aug,epoch=epoch)
                    if capture[0]:buffer.add_data=original_add
                    sync();elapsed=time.perf_counter()-t
                    assert result['optimizer_hook_steps']==before+1 and math.isfinite(objective)
                    ce=float(adapter.ce_values[0].cpu());rep=float(adapter.ce_values[1].cpu()) if len(adapter.ce_values)>1 else 0.;mse=float(adapter.mse_values[0].cpu()) if adapter.mse_values else 0.
                    assert abs(objective-(ce+.5*rep+.3*mse))<1e-4*max(1,abs(objective))
                    global_current[rows]+=1;epoch_current+=len(rows)
                    for k,draw in enumerate(buffer.draws):np.add.at(global_replay[k],np.asarray(draw['source_rows']),1)
                    row={'task':task+1,'epoch':epoch+1,'batch':batch+1,'global_step':result['optimizer_hook_steps'],'lr':lr,'current_n':len(rows),'current_rows':rows.tolist(),'replay_draws':buffer.draws,'reservoir_admitted':buffer.admitted,'buffer_seen':buffer.seen,'loss':objective,'ce_current':ce,'ce_replay':rep,'mse':mse,'synchronized_wall_seconds':elapsed}
                    steps.write(json.dumps(row)+'\n');lossvals.append(objective);cevals.append(ce);msevals.append(mse);repvals.append(rep)
                    if result['optimizer_hook_steps']<=32:first_times.append(elapsed)
                    if result['optimizer_hook_steps']==32:
                        median=float(np.median(first_times[4:]));remaining=(6300-32)*median+120
                        result['throughput']={'first32_seconds':sum(first_times),'warm_median_step_seconds':median,'estimated_remaining_seconds':remaining,'model':'Full BN ResNet18 NF64, real replay from step2'}
                        event('throughput',**result['throughput'])
                        if time.time()+remaining>deadline:raise TimeoutError('Measured throughput predicts exceeding 90-minute budget')
                assert epoch_current==2000 and len(lossvals)==63
                bn=[int(m.num_batches_tracked.cpu()) for m in net.modules() if isinstance(m,nn.BatchNorm2d)]
                assert len(bn)==20 and min(bn)==max(bn)==3*result['optimizer_hook_steps']-2
                scheduler.step()
                er={'task':task+1,'epoch':epoch+1,'completed_global_step':result['optimizer_hook_steps'],'lr_used':lr,'lr_next':opt.param_groups[0]['lr'],'current_presentations':epoch_current,'mean_loss':float(np.mean(lossvals)),'mean_ce_current':float(np.mean(cevals)),'mean_ce_replay':float(np.mean(repvals)),'mean_mse':float(np.mean(msevals)),'wall_seconds':time.perf_counter()-epoch_start,'buffer_seen':buffer.seen,'buffer_distinct_sources':len(np.unique(buffer.rows[:buffer.active])),'buffer_class_counts':torch.bincount(buffer.labels[:buffer.active],minlength=10).tolist(),'BN_num_batches_tracked':bn[0],**memory()}
                epoch_results.append(er);savej(out/'epochs.json',epoch_results);event('epoch',**er)
                if (epoch+1)%10==0:
                    er['resources']=resource_guard()
                    savej(out/'epochs.json',epoch_results)
                    if epoch<49:checkpoint(f'safety-task{task+1}-epoch{epoch+1:02d}.pt',task+1,epoch+1)
            step_hook.remove();assert result['optimizer_hook_steps']==3150*(task+1)
            evaluation={'task':task+1,'steps_completed':result['optimizer_hook_steps'],'task_wall_seconds':time.perf_counter()-task_start}
            oldmet,oldraw=evaluate(net,dx[0],dy[0]);evaluation['old_dev']=oldmet
            np.savez_compressed(out/f'task{task+1}-old-dev.npz',**oldraw)
            if task==1:
                newmet,newraw=evaluate(net,dx[1],dy[1]);evaluation['new_dev']=newmet
                np.savez_compressed(out/'task2-new-dev.npz',**newraw)
                evaluation['combined_dev']={'n':1000,'accuracy':(oldmet['accuracy']+newmet['accuracy'])/2,'nll':(oldmet['nll']+newmet['nll'])/2}
            task_evaluations.append(evaluation);result['task_evaluations']=task_evaluations
            checkpoint(f'task{task+1}-final.pt',task+1,50);event('task_completed',**evaluation)
        hook.remove()
        assert buffer.seen==200000 and global_current.sum()==200000 and global_replay.sum()==403136
        assert np.all(global_current[np.concatenate(fitrows)]==50)
        np.savez_compressed(out/'exposure-counts.npz',current=global_current,replay_distillation=global_replay[0],replay_supervised=global_replay[1])
        result['exposure']={'current_total':int(global_current.sum()),'current_unique':int(np.count_nonzero(global_current)),'each_fit_row_presentations':50,'replay_distillation_total':int(global_replay[0].sum()),'replay_supervised_total':int(global_replay[1].sum()),'buffer_seen_observations':buffer.seen,'final_buffer_unique_sources':len(np.unique(buffer.rows)),'final_buffer_classes':torch.bincount(buffer.labels,minlength=10).tolist()}
        first=task_evaluations[0]['old_dev'];last=task_evaluations[1]
        g1=first['accuracy']>=.5 and first['nll']<=1.5 and initial['nll']-first['nll']>=.3
        g2=all(last[k]['accuracy']>=.35 and last[k]['nll']<=1.8 for k in ['old_dev','new_dev'])
        result['inherited_basic_learning_checks']={'task1':g1,'task2_old_and_new':g2,'interpretation':'Same weak operational thresholds as prior pilot; not scientific sufficiency or contribution.'}
        result['status']='completed_basic_learning_checks_passed' if g1 and g2 else 'completed_basic_learning_checks_not_met'
    except Exception as e:
        result['status']='incomplete_execution';result['error']=f'{type(e).__name__}: {e}';result['traceback']=traceback.format_exc()
        if net is not None and opt is not None and scheduler is not None and buffer is not None:
            try:checkpoint('interrupted-state.pt',task+1,epoch)
            except Exception as ee:result['checkpoint_error']=str(ee)
        event('stopped',error=result['error'],steps=result['optimizer_hook_steps'])
    finally:
        result['finished_utc']=utc();result['run_perf_seconds']=time.perf_counter()-started;result['run_wall_seconds']=time.time()-wall_start;result['memory']=memory()
        savej(out/'summary.json',result);event('finished',status=result['status'],steps=result['optimizer_hook_steps'],run_seconds=result['run_perf_seconds'])
        events.close();steps.close()
    return result

if __name__=='__main__':
    r=main(ROOT/'run-002');print(json.dumps(r,indent=2),flush=True)
