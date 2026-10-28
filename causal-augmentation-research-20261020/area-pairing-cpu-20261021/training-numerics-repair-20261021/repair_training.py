"""Disposable training-side numerical diagnosis; no validation/test or model search."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import datetime,hashlib,json,math,resource,signal,statistics,time
import torch
import torchvision
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/mechanism-transfer-pets-20261021')
WEIGHT=DATA.parent/'encoder-readiness-20261021/runtime/torch-cache/hub/checkpoints/resnet18-f37072fd.pth'

def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def number(x):
    x=float(x)
    return x if math.isfinite(x) else str(x)

def state_hash(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def bn_hash(model):
    return state_hash({name:v for name,v in model.state_dict().items() if 'running_' in name or 'num_batches_tracked' in name})

def stats(named,gradient=False):
    total=0.;maximum=0.;largest=None;finite=True;count=0
    for name,p in named:
        value=p.grad if gradient else p.detach()
        if value is None:continue
        value=value.detach();count+=value.numel()
        if not bool(torch.isfinite(value).all()):finite=False
        peak=float(value.abs().max())
        if peak>maximum or not math.isfinite(peak):maximum=peak;largest=name
        total+=float(value.double().square().sum())
    return dict(finite=finite,L2=number(math.sqrt(total)),max_abs=number(maximum),largest_peak_parameter=largest,elements=count)

def new_model(head_lr,body_lr):
    model=torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
    torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(0).get_state())
    model.fc=torch.nn.Linear(512,37,device='cpu');model.train().to('cpu')
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
    body=[p for name,p in model.named_parameters() if not name.startswith('fc.')]
    optimizer=torch.optim.SGD([dict(params=body,lr=body_lr),dict(params=model.fc.parameters(),lr=head_lr)],momentum=.9,weight_decay=1e-4)
    assert all(p.device.type=='cpu' for p in model.parameters())
    return model,optimizer

def safe_train_step(model,optimizer,x,y):
    """Minimal reusable nonzero-LR CPU update, with finite loss/gradient/parameter guards."""
    optimizer.zero_grad(set_to_none=True)
    logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y,reduction='mean')
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss before backward; step refused')
    loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('nonfinite gradient; optimizer step refused')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('nonfinite updated parameter; stop')
    return float(loss.detach())

def diagnose(name,head_lr,body_lr,steps,x,y,allow_unsafe_original=False):
    model,optimizer=new_model(head_lr,body_lr);initial_hash=state_hash(model.state_dict());initial_bn=bn_hash(model)
    first_conv=model.conv1.weight.detach().clone();first_head=model.fc.weight.detach().clone()
    groups={'body':[(n,p) for n,p in model.named_parameters() if not n.startswith('fc.')],
            'head':[(n,p) for n,p in model.named_parameters() if n.startswith('fc.')]}
    trace=[];start=time.perf_counter()
    with (ROOT/(name+'-trace.jsonl')).open('x') as stream:
        for i in range(steps):
            optimizer.zero_grad(set_to_none=True);assert all(p.grad is None for p in model.parameters())
            logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y,reduction='mean')
            row=dict(step=i,loss=number(loss.detach()),logits_min=number(logits.detach().min()),logits_max=number(logits.detach().max()),
                loss_finite=bool(torch.isfinite(loss)),parameters_before={g:stats(parts) for g,parts in groups.items()})
            if i==0:
                reference=-torch.nn.functional.log_softmax(logits,dim=1)[torch.arange(len(y)),y].mean()
                row['CE_equals_mean_negative_log_probability_abs_error']=float(abs(reference-loss).detach())
            if row['loss_finite']:
                loss.backward();row['gradients']={g:stats(parts,True) for g,parts in groups.items()}
                gradients_finite=all(v['finite'] for v in row['gradients'].values());row['gradients_finite']=gradients_finite
                if not gradients_finite and not allow_unsafe_original:row['update']='refused_nonfinite_gradient'
                else:
                    optimizer.step();row['update']='original_unchecked_SGD' if allow_unsafe_original else 'finite_guarded_SGD'
                    row['parameters_after']={g:stats(parts) for g,parts in groups.items()}
                    buffers=[v['momentum_buffer'] for v in optimizer.state.values() if 'momentum_buffer' in v]
                    row['momentum_buffers_finite']=all(bool(torch.isfinite(v).all()) for v in buffers)
            else:row['update']='stopped_nonfinite_loss_before_backward'
            trace.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if not row['loss_finite'] or (not allow_unsafe_original and not row.get('gradients_finite',True)):break
    result=dict(name=name,head_LR=head_lr,body_LR=body_lr,initial_state_sha256=initial_hash,
        BN_running_stats_before_sha256=initial_bn,BN_running_stats_unchanged=initial_bn==bn_hash(model),
        steps_recorded=len(trace),losses=[r['loss'] for r in trace],wall_seconds=time.perf_counter()-start,
        any_nonfinite_loss=any(not r['loss_finite'] for r in trace),
        any_nonfinite_gradient=any(not r.get('gradients_finite',True) for r in trace),
        any_nonfinite_parameter=any(not q['finite'] for r in trace for q in r.get('parameters_after',{}).values()),
        conv1_weight_delta_L2=number((model.conv1.weight.detach()-first_conv).double().norm()),
        head_weight_delta_L2=number((model.fc.weight.detach()-first_head).double().norm()))
    if not allow_unsafe_original:
        with torch.inference_mode():result['loss_after_detailed_steps']=float(torch.nn.functional.cross_entropy(model(x),y))
        times=[];lean_losses=[]
        for i in range(20):
            t=time.perf_counter();lean_losses.append(safe_train_step(model,optimizer,x,y));times.append(time.perf_counter()-t)
        with torch.inference_mode():result['final_training_loss']=float(torch.nn.functional.cross_entropy(model(x),y))
        result.update(lean_steps=20,lean_losses=lean_losses,mean_guarded_training_step_seconds=statistics.mean(times),
            median_guarded_training_step_seconds=statistics.median(times),guarded_training_images_per_second=8/statistics.mean(times),
            final_state_sha256=state_hash(model.state_dict()),all_final_parameters_finite=all(bool(torch.isfinite(v).all()) for v in model.parameters()),
            BN_running_stats_unchanged_after_all_steps=initial_bn==bn_hash(model),
            conv1_weight_delta_L2=float((model.conv1.weight.detach()-first_conv).double().norm()),
            head_weight_delta_L2=float((model.fc.weight.detach()-first_head).double().norm()))
    save(name+'-result.json',result)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result,trace

def main():
    try:os.nice(10)
    except OSError:pass
    signal.alarm(120);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'repair-protocol.json').read_text());preprocess=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    weight_sha=hashlib.sha256(WEIGHT.read_bytes()).hexdigest();assert weight_sha=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    labels={line.split()[0]:int(line.split()[1])-1 for line in (DATA/'archive-text-3.txt').read_text().splitlines() if line.strip() and not line.startswith('#')}
    images=[];audit=[]
    for row in protocol['photos']:
        path=DATA/'photos'/(row['source_id']+'.jpg');im=Image.open(path).convert('RGB')
        assert labels[row['source_id']]==row['class37_target']
        images.append(preprocess(im));audit.append(dict(source_id=row['source_id'],source_mode=im.mode,
            raw_minmax=[min(v[0] for v in im.getextrema()),max(v[1] for v in im.getextrema())],JPEG_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    x=torch.stack(images).to('cpu');y=torch.tensor([r['class37_target'] for r in protocol['photos']],dtype=torch.long,device='cpu')
    reconstructed=x*torch.tensor([.229,.224,.225]).reshape(1,3,1,1)+torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
    assert x.shape==(8,3,224,224) and x.dtype==torch.float32 and torch.isfinite(x).all()
    assert y.min()>=0 and y.max()<37 and y.dtype==torch.int64
    assert reconstructed.min()>=-1e-6 and reconstructed.max()<=1+1e-6
    model,_=new_model(.1,.01);assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values());del model
    save('input-and-initialization-audit.json',dict(photos=audit,tensor_shape=list(x.shape),dtype=str(x.dtype),device=str(x.device),
        normalized_minmax=[float(x.min()),float(x.max())],inverse_normalized_minmax=[float(reconstructed.min()),float(reconstructed.max())],
        targets=y.tolist(),preprocess=repr(preprocess),pretrained_weight_sha256=weight_sha,pretrained_all_finite=True,
        mean_cross_entropy_on37logits=True,new_test_or_validation_reads=0,head_seed=0,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads()))
    start=time.perf_counter()
    original,traceA=diagnose('original_high_LR',.1,.01,12,x,y,True)
    repaired,traceB=diagnose('repaired_lower_LR',.01,.001,30,x,y,False)
    assert original['initial_state_sha256']==repaired['initial_state_sha256']
    success=(not repaired['any_nonfinite_loss'] and not repaired['any_nonfinite_gradient'] and not repaired['any_nonfinite_parameter']
        and repaired['all_final_parameters_finite'] and repaired['conv1_weight_delta_L2']>0 and repaired['head_weight_delta_L2']>0
        and repaired['final_training_loss']<.7*repaired['losses'][0] and repaired['BN_running_stats_unchanged_after_all_steps'])
    result=dict(status='short_CPU_training_repair_passed' if success else 'repair_not_yet_passed',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_nonfinite_failure_reproduced=original['any_nonfinite_loss'] or original['any_nonfinite_gradient'],
        identical_initialization_and_inputs=True,only_training_configuration_change='head/body learning rates x0.1, plus finite guards; no clipping,warmup,architecture or data changes',
        initial_loss=repaired['losses'][0],loss_after30steps=repaired['loss_after_detailed_steps'],final_loss_after50steps=repaired['final_training_loss'],
        actual_nonzero_backbone_and_head_updates=True,all_repaired_losses_gradients_parameters_finite=success,
        batch=8,CPU_only=True,ARIS=False,GPU=False,A100=False,tests_or_validation_evaluations=0,
        throughput_images_per_second=repaired['guarded_training_images_per_second'],step_seconds=repaired['mean_guarded_training_step_seconds'],
        wall_seconds=time.perf_counter()-start,process_peak_RSS_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        learned_weights_saved=False,no_full_training_autostart=True,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save('repair-result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));assert success

if __name__=='__main__':main()
