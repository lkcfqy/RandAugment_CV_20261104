#!/usr/bin/env python3
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0';os.environ['PYTORCH_MPS_FAST_MATH']='0'
import copy,datetime,hashlib,importlib.util,json,math,resource,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
OLD=PROJECT/'augmentation-research-20261029/fresh-replay-reference-pilot-20261029'
DATA=PROJECT/'data/cifar100-official-20261029'
OUT=HERE/'run-001'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def flatten(gs):return torch.cat([g.detach().float().cpu().reshape(-1) for g in gs])
def movement(a,b):
    rows=[]
    for k in a:
        x=a[k].double();y=b[k].double();d=y-x
        rows.append({'name':k,'numel':x.numel(),'l2_change':float(d.norm()),'max_abs_change':float(d.abs().max()),'initial_l2':float(x.norm()),'changed_elements':int((d!=0).sum())})
    return {'tensors':rows,'changed_tensors':sum(x['l2_change']>0 for x in rows),'total_tensors':len(rows),'total_l2_change':math.sqrt(sum(x['l2_change']**2 for x in rows))}
def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():h=model(x);loss=F.cross_entropy(h,y)
    torch.mps.synchronize()
    return {'ce':float(loss.cpu()),'accuracy':float((h.argmax(1)==y).float().mean().cpu())}


def main():
    OUT.mkdir(exist_ok=False);t0=time.perf_counter();wall0=time.time();result={'started_utc':utc(),'status':'running','purpose':'implementation health only'}
    manifest=json.loads((OLD/'RUN_MANIFEST.json').read_text())
    assert all(sha(OLD/x['path'])==x['sha256'] for x in manifest['files'])
    spec=importlib.util.spec_from_file_location('frozen_pilot',OLD/'pilot.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    # Only these verified, self-created local checkpoint files are deserialized.
    c1=torch.load(OLD/'run-001/task1-checkpoint.pt',map_location='cpu',weights_only=False)
    c2=torch.load(OLD/'run-001/common-checkpoint.pt',map_location='cpu',weights_only=False)
    torch.manual_seed(p.SEED);initial=p.SmallResNet()
    result['old_parameter_changes']={'initial_to_task1':movement(initial.state_dict(),c1['model']),'task1_to_task2':movement(c1['model'],c2['model'])}
    result['architecture']={'text':str(initial),'parameters':sum(x.numel() for x in initial.parameters()),'parameter_tensors':len(list(initial.parameters())),'all_requires_grad':all(x.requires_grad for x in initial.parameters()),'head_out_features':initial.head.out_features,'batchnorm_count':sum(isinstance(x,nn.modules.batchnorm._BatchNorm) for x in initial.modules()),'groupnorm_count':sum(isinstance(x,nn.GroupNorm) for x in initial.modules())}
    macs=[];hooks=[]
    def hook(name):
        def f(layer,args,out):
            if isinstance(layer,nn.Conv2d):m=out.numel()*(layer.in_channels//layer.groups)*layer.kernel_size[0]*layer.kernel_size[1]
            else:m=out.numel()*layer.in_features
            macs.append({'name':name,'output_shape':list(out.shape),'MACs':m})
        return f
    for n,m in initial.named_modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):hooks.append(m.register_forward_hook(hook(n)))
    with torch.no_grad():initial(torch.zeros(1,3,32,32))
    for h in hooks:h.remove()
    result['conv_linear_forward_MACs_per_image']=sum(x['MACs'] for x in macs);result['layer_MACs']=macs
    result['old_optimizer']={}
    for label,c in [('task1',c1),('task2',c2)]:
        opt=c['optimizer'];groups=[{k:v for k,v in g.items() if k!='params'} for g in opt['param_groups']]
        result['old_optimizer'][label]={'recorded_step':c['step'],'groups':groups,'parameter_states':len(opt['state']),'momentum_states':sum('momentum_buffer' in v for v in opt['state'].values()),'momentum_l2':math.sqrt(sum(float(v['momentum_buffer'].double().square().sum()) for v in opt['state'].values()))}
    seq=np.load(OLD/'run-001/data-and-sequences.npz',allow_pickle=False);buf=np.load(OLD/'run-001/buffer-at-capture.npz',allow_pickle=False);replay=np.load(OLD/'run-001/prefix-replay-history.npz',allow_pickle=False)
    raw=np.memmap(DATA/'train-only/train.bin',mode='r',dtype=np.uint8,shape=(50000,3074));labels=np.asarray(raw[:,1])
    def imgs(rows):return torch.from_numpy(np.array(raw[rows,2:],copy=True).reshape(-1,3,32,32))
    exposures={}
    for label,steps,rowskey,plankey in [('task1',600,'old_fit_rows','task1_batch_positions'),('task2',160,'new_fit_rows','task2_and_branch_batch_positions')]:
        rows=seq[rowskey];ids=rows[seq[plankey][:steps]].reshape(-1);counts=np.array([np.sum(ids==i) for i in rows])
        exposures[label]={'steps':steps,'batch_size':32,'total_presentations':len(ids),'unique_sources':len(np.unique(ids)),'epochs_equivalent':len(ids)/len(rows),'class_presentations':np.bincount(labels[ids],minlength=10).tolist(),'source_count_min':int(counts.min()),'source_count_max':int(counts.max())}
    ids=replay['buffer_positions'];assert ids.shape==(160,2,32)
    exposures['replay']={'shape':list(ids.shape),'total_presentations':int(ids.size),'unique_buffer_positions':len(np.unique(ids)),'class_presentations':np.bincount(buf['labels'][ids.reshape(-1)],minlength=10).tolist()}
    result['exposures']=exposures
    log=[json.loads(x) for x in (OLD/'run-001/training-losses.jsonl').read_text().splitlines()]
    assert len(log)==760 and sum(x['phase']=='task1' for x in log)==600 and sum(x['phase']=='task2_prefix' for x in log)==160
    prior=json.loads((OLD/'run-001/summary.json').read_text());wall=(datetime.datetime.fromisoformat(prior['finished_utc'])-datetime.datetime.fromisoformat(prior['started_utc'])).total_seconds()
    result['old_timing']={'run_perf_counter_seconds':prior['execution_wall_seconds'],'recorded_UTC_difference_seconds':wall,'task1_sum_synchronized_step_seconds':sum(x['seconds'] for x in log if x['phase']=='task1'),'task1_phase_seconds':prior['task1_wall_seconds'],'task2_phase_seconds':prior['branchpoint']['task2_wall_seconds'],'scope':'run() starts after imports; includes preflight/data/training/evaluation, excludes Python import and final serialization/printing. Each optimizer step is followed by torch.mps.synchronize(). No old external whole-process timing was captured.'}
    # Recompute exact first new-task minibatch, without optimizer update.
    model=p.SmallResNet().to('mps');model.load_state_dict(c1['model']);model.train()
    idx=seq['task2_and_branch_batch_positions'][0];newrows=seq['new_fit_rows'][idx]
    bd,bc=ids[0];bx=imgs(buf['rows'])
    xn,_=p.augment(imgs(newrows),p.SEED+500000);xd,_=p.augment(bx[bd],p.SEED+600000);xc,_=p.augment(bx[bc],p.SEED+700000)
    hn,hd,hc=model(torch.cat([xn,xd,xc]).to('mps')).split(32)
    newce=F.cross_entropy(hn,torch.tensor(labels[newrows].astype(np.int64),device='mps'))
    oldce=F.cross_entropy(hc,torch.tensor(buf['labels'][bc],device='mps'))
    mse=F.mse_loss(hd,torch.tensor(buf['historical_targets'][bd],device='mps'))
    components=[newce,.5*oldce,.5*mse];gs=[flatten(torch.autograd.grad(v,list(model.parameters()),retain_graph=True)) for v in components]
    torch.mps.synchronize()
    record=next(x for x in log if x['phase']=='task2_prefix')
    losses={'ce_new':float(newce.detach().cpu()),'ce_old':float(oldce.detach().cpu()),'mse':float(mse.detach().cpu())}
    maxerr=max(abs(losses[k]-record[k]) for k in losses);assert maxerr<2e-4
    result['first_task2_recalculation']={'losses':losses,'saved_loss_max_abs_error':maxerr,'component_order':['CE_new','.5 CE_replay','.5 MSE'],'weighted_gradient_l2':[float(g.norm()) for g in gs],
        'gradient_cosine':{f'{i}-{j}':float(gs[i]@gs[j]/(gs[i].norm()*gs[j].norm())) for i in range(3) for j in range(i+1,3)},'optimizer_steps':0}
    del model,hn,hd,hc,components,gs;torch.mps.empty_cache()
    # One predeclared micro-overfit diagnostic; no replay or augmentation.
    healthrows=[]
    for c in range(5,10):
        candidates=seq['new_fit_rows'][labels[seq['new_fit_rows']]==c].tolist()
        candidates.sort(key=lambda i:hashlib.sha256(f'health-20261029|{c}|{i}'.encode()).digest());healthrows.extend(candidates[:4])
    healthrows=np.array(healthrows);np.save(OUT/'microfit-train-rows.npy',healthrows,allow_pickle=False)
    assert not np.intersect1d(healthrows,np.r_[seq['old_dev_rows'],seq['new_dev_rows']]).size
    x=p.clean(imgs(healthrows)).to('mps');y=torch.tensor(labels[healthrows].astype(np.int64),device='mps')
    model=p.SmallResNet().to('mps');model.load_state_dict(c1['model'])
    optimizer=torch.optim.SGD(model.parameters(),lr=.02,momentum=.9,weight_decay=.0005);optimizer.load_state_dict(copy.deepcopy(c1['optimizer']))
    for group in optimizer.param_groups:group['lr']=.02
    stepcount=[0]
    def increment(*args):stepcount[0]+=1
    handle=optimizer.register_step_post_hook(increment)
    init_metrics=evaluate(model,x,y);before=p.cpu_copy(model.state_dict());records=[];grad_checks={};train_start=time.perf_counter();train_wall=time.time();success=False
    with (OUT/'microfit-steps.jsonl').open('x',buffering=1) as f:
        for step in range(300):
            if time.perf_counter()-train_start>300:break
            model.train();optimizer.zero_grad(set_to_none=True);torch.mps.synchronize();a=time.perf_counter_ns()
            h=model(x);h.retain_grad();loss=F.cross_entropy(h,y);torch.mps.synchronize();b=time.perf_counter_ns()
            loss.backward();torch.mps.synchronize();c=time.perf_counter_ns()
            if step==0:
                expected=(h.detach().softmax(-1)-F.one_hot(y,10))/20
                grad_checks={'all_parameters_require_grad':all(p.requires_grad for p in model.parameters()),'none_gradient_parameters':[n for n,p in model.named_parameters() if p.grad is None],
                    'nonzero_gradient_tensors':sum(bool(p.grad.abs().max().cpu()>0) for p in model.parameters()),'logit_gradient_max_error':float((h.grad-expected).abs().max().cpu()),
                    'per_parameter_gradient_l2':{n:float(p.grad.norm().cpu()) for n,p in model.named_parameters()},'head_target_indices':y.cpu().tolist()}
                assert not grad_checks['none_gradient_parameters'] and grad_checks['logit_gradient_max_error']<1e-6
            optimizer.step();torch.mps.synchronize();d=time.perf_counter_ns()
            r={'step':step+1,'lr':optimizer.param_groups[0]['lr'],'pre_update_ce':float(loss.detach().cpu()),'forward_seconds':(b-a)/1e9,'backward_seconds':(c-b)/1e9,'step_seconds_including_first_step_gradient_diagnostic':(d-c)/1e9,'synchronized_total_seconds':(d-a)/1e9}
            assert math.isfinite(r['pre_update_ce']);f.write(json.dumps(r)+'\n');records.append(r)
            if (step+1)%25==0:
                metric=evaluate(model,x,y);print(json.dumps({'phase':'microfit','step':step+1,**metric}),flush=True)
                if metric['accuracy']>=1.0 and metric['ce']<=.05:success=True;break
    final=evaluate(model,x,y);handle.remove()
    assert stepcount[0]==len(records)
    result['microfit']={'passed':success,'initial':init_metrics,'final':final,'optimizer_hook_steps':stepcount[0],'gradient_checks':grad_checks,'parameter_changes':movement(before,p.cpu_copy(model.state_dict())),
        'label_counts':np.bincount(labels[healthrows],minlength=10).tolist(),'training_perf_seconds':time.perf_counter()-train_start,'training_wall_seconds':time.time()-train_wall,'sum_synchronized_step_seconds':sum(x['synchronized_total_seconds'] for x in records),'median_synchronized_step_seconds':float(np.median([x['synchronized_total_seconds'] for x in records[4:]])),'replay_used':False,'augmentation_used':False,'diagnostic_train_images':20,'new_complete_prefix':False}
    result['status']='health_pass_no_definite_bug_found' if success else 'microfit_not_met_no_recipe_search'
    result['finished_utc']=utc();result['main_perf_seconds']=time.perf_counter()-t0;result['main_wall_seconds']=time.time()-wall0
    result['memory']={'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()}
    js(OUT/'diagnostic.json',result)
    print(json.dumps({'status':result['status'],'microfit':{k:v for k,v in result['microfit'].items() if k not in ['gradient_checks','parameter_changes']},'main_perf_seconds':result['main_perf_seconds'],'old_timing':result['old_timing']}),flush=True)

if __name__=='__main__':main()
