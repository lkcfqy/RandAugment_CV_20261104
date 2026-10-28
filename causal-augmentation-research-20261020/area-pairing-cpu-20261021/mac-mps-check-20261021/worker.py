"""Same training-only FP32 workload on CPU or local Apple MPS, no CPU fallback."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
    PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import datetime,gc,hashlib,json,resource,signal,statistics,sys,time
import torch,torchvision
from PIL import Image

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
DATA=BASE/'mechanism-transfer-pets-20261021'
WEIGHT=BASE/'encoder-readiness-20261021/runtime/torch-cache/hub/checkpoints/resnet18-f37072fd.pth'

def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def sync(device):
    if device=='mps':torch.mps.synchronize()
def memory(device):
    result={'process_peak_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device=='mps':
        for name in ['current_allocated_memory','driver_allocated_memory','recommended_max_memory']:
            if hasattr(torch.mps,name):result[name+'_bytes']=int(getattr(torch.mps,name)())
    return result
def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def make_model(device):
    model=torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
    torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(0).get_state())
    model.fc=torch.nn.Linear(512,37,device='cpu');initial=state_hash(model)
    model.to(device)
    return model,initial
def train_mode(model):
    model.train()
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
def finite_tensors(values):
    # Single reduction/synchronization avoids per-parameter host/device roundtrips.
    flat=torch.cat([v.detach().reshape(-1) for v in values])
    ok=bool(torch.isfinite(flat).all().item());del flat
    return ok
def training_step(model,optimizer,x_cpu,y_cpu,device):
    x=x_cpu.to(device);y=y_cpu.to(device)
    optimizer.zero_grad(set_to_none=True);logits=model(x)
    loss=torch.nn.functional.cross_entropy(logits,y)
    if not bool(torch.isfinite(loss).item()):raise FloatingPointError('nonfinite loss')
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert all(v.device.type==device for v in gradients)
    if not finite_tensors(gradients):raise FloatingPointError('nonfinite gradients')
    optimizer.step()
    if not finite_tensors(list(model.parameters())):raise FloatingPointError('nonfinite parameters')
    return float(loss.detach().cpu())

def main(device):
    try:os.nice(10)
    except OSError:pass
    signal.alarm(180);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_device('cpu');torch.use_deterministic_algorithms(False)
    if device=='mps':assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    photos=json.loads((BASE/'training-numerics-repair-20261021/repair-protocol.json').read_text())['photos']
    preprocess=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    x8=torch.stack([preprocess(Image.open(DATA/'photos'/(r['source_id']+'.jpg')).convert('RGB')) for r in photos])
    y8=torch.tensor([r['class37_target'] for r in photos],dtype=torch.long)
    results=[];started=time.perf_counter()
    for batch in [8,32]:
        x_cpu=x8.repeat(batch//8,1,1,1);y_cpu=y8.repeat(batch//8)
        model,initial=make_model(device);model.eval()
        warm_started=time.perf_counter()
        with torch.inference_mode():
            for _ in range(3):output=model(x_cpu.to(device))
            sync(device);warm_seconds=time.perf_counter()-warm_started
            sync(device);t=time.perf_counter()
            for _ in range(8):output=model(x_cpu.to(device))
            sync(device);seconds=time.perf_counter()-t
            assert torch.isfinite(output).all()
            inf=dict(initial_state_sha256=initial,logits=output.cpu().tolist(),warmup_seconds=warm_seconds,
                timed_forward_count=8,timed_seconds=seconds,images_per_second=8*batch/seconds,
                input_transfer_included=True,memory_after=memory(device))
        del output,model;gc.collect()
        if device=='mps':torch.mps.empty_cache()
        model,initial=make_model(device);train_mode(model)
        optimizer=torch.optim.SGD([dict(params=model.fc.parameters(),lr=.01),
            dict(params=[p for n,p in model.named_parameters() if not n.startswith('fc.')],lr=.001)],momentum=.9,weight_decay=1e-4)
        losses=[];step_times=[];memory_samples=[];warm_started=time.perf_counter()
        for _ in range(2):losses.append(training_step(model,optimizer,x_cpu,y_cpu,device))
        sync(device);training_warm_seconds=time.perf_counter()-warm_started
        sync(device);t=time.perf_counter()
        for _ in range(8):
            step_start=time.perf_counter();losses.append(training_step(model,optimizer,x_cpu,y_cpu,device));sync(device)
            step_times.append(time.perf_counter()-step_start);memory_samples.append(memory(device))
        sync(device);training_seconds=time.perf_counter()-t
        model.eval()
        with torch.inference_mode():
            output=model(x_cpu.to(device));final_loss=float(torch.nn.functional.cross_entropy(output,y_cpu.to(device)).cpu())
            final_logits=output.cpu().tolist()
        tr=dict(initial_state_sha256=initial,warmup_updates=2,timed_updates=8,warmup_seconds=training_warm_seconds,
            losses=losses,final_loss=final_loss,logits_after10updates=final_logits,
            training_loss_decreased=final_loss<.7*losses[0],all_loss_grad_parameter_finite=True,
            input_transfer_included=True,timed_seconds=training_seconds,mean_step_seconds=statistics.mean(step_times),
            images_per_second=8*batch/training_seconds,parameter_devices=sorted(set(p.device.type for p in model.parameters())),
            gradient_devices=sorted(set(p.grad.device.type for p in model.parameters() if p.grad is not None)),
            memory_samples_after_steps=memory_samples)
        result=dict(device=device,batch=batch,inference=inf,training=tr);results.append(result)
        print(json.dumps({'device':device,'batch':batch,'inference_images_sec':inf['images_per_second'],
            'guarded_training_images_sec':tr['images_per_second'],'initial_loss':losses[0],'final_loss':final_loss,
            'warmup_inference_seconds':warm_seconds,'warmup_training_seconds':training_warm_seconds}),flush=True)
        del model,optimizer,output;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    out=dict(status='completed_same_batch_'+device+'_probe',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        device=device,results=results,wall_seconds=time.perf_counter()-started,memory_final=memory(device),
        torch=torch.__version__,torchvision=torchvision.__version__,threads=torch.get_num_threads(),
        precision='FP32',CPU_fallback_enabled=False,MPS_fast_math=False,
        deterministic_algorithms_requested=False,weight_sha256=hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_or_validation_uses=0,
        new_install_or_security_change=False,A100_or_remote_GPU=False,ARIS=False)
    save(device+'-result.json',out)

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['cpu','mps']
    try:main(sys.argv[1])
    except Exception as error:
        save(sys.argv[1]+'-failure.json',dict(status='probe_stopped_no_workaround',type=type(error).__name__,message=str(error)))
        raise
