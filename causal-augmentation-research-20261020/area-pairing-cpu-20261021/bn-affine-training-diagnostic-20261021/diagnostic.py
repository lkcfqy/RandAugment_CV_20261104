"""One predeclared training-only BN-affine control, not an augmentation-benefit experiment."""
import os
os.environ.update(OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1',
    CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0',
    PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import datetime, gc, hashlib, json, resource, signal, time
import numpy as np
from PIL import Image
import torch, torchvision
from torchvision.transforms import functional as F, InterpolationMode, TrivialAugmentWide

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
OLD=BASE/'stage-validity-pilot-20261021'
WEIGHT=BASE/'encoder-readiness-20261021/runtime/torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
DATA=BASE/'mechanism-transfer-pets-20261021/photos'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    with (ROOT/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def state_hash(values):
    h=hashlib.sha256()
    for n,v in values.items():
        h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def bn_stats(model):
    return state_hash({n:v for n,v in model.state_dict().items()
        if n.endswith(('running_mean','running_var','num_batches_tracked'))})
def bn_affine(model):
    return {n+'.'+k:v for n,m in model.named_modules()
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
        for k,v in m.named_parameters(recurse=False)}
def train_mode(model):
    model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
def tensor(im): return F.normalize(F.to_tensor(im),[.485,.456,.406],[.229,.224,.225])
def finite(values):
    flat=torch.cat([v.detach().reshape(-1) for v in values]);ok=bool(torch.isfinite(flat).all().item())
    del flat
    return ok
def evaluate(model,images,labels):
    model.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(images),8):
            parts.append(model(torch.stack([tensor(im) for im in images[start:start+8]]).to('mps')).cpu())
        z=torch.cat(parts);y=torch.tensor(labels,dtype=torch.long)
        losses=torch.nn.functional.cross_entropy(z,y,reduction='none')
        assert torch.isfinite(z).all() and torch.isfinite(losses).all()
        out=dict(mean_LL_nats=float(losses.mean()),accuracy=float((z.argmax(1)==y).float().mean()),
            logits=z.tolist(),losses=losses.tolist())
    train_mode(model)
    return out
def make_model(freeze_affine):
    m=torchvision.models.resnet18(weights=None)
    m.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
    torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(0).get_state())
    m.fc=torch.nn.Linear(512,2,device='cpu')
    initial=state_hash(m.state_dict())
    if freeze_affine:
        for p in bn_affine(m).values():p.requires_grad_(False)
    m.to('mps');train_mode(m)
    opt=torch.optim.SGD([dict(params=m.fc.parameters(),lr=.01),dict(
        params=[p for n,p in m.named_parameters() if not n.startswith('fc.') and p.requires_grad],lr=.001)],
        momentum=.9,weight_decay=1e-4)
    return m,opt,initial
def step(model,opt,x,y):
    opt.zero_grad(set_to_none=True);z=model(x.to('mps'))
    loss=torch.nn.functional.cross_entropy(z,y.to('mps'))
    if not bool(torch.isfinite(loss).item()):raise FloatingPointError('nonfinite loss')
    loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert all(v.device.type=='mps' for v in grads)
    if not finite(grads):raise FloatingPointError('nonfinite gradient')
    opt.step()
    if not finite(list(model.parameters())):raise FloatingPointError('nonfinite parameter')
    return float(loss.detach().cpu())
def timeout(signum,frame):raise TimeoutError('predeclared240secondwalllimit, no extension')

def main():
    try:os.nice(10)
    except OSError:pass
    signal.signal(signal.SIGALRM,timeout);signal.alarm(240)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    torch.use_deterministic_algorithms(False)
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    started=time.perf_counter();protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha(WEIGHT)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    prior=json.loads((OLD/'protocol.json').read_text())['train_rows']
    old_inputs=json.loads((OLD/'input-manifest.json').read_text())
    old_arm=json.loads((OLD/'finetune_TA-result.json').read_text())
    base=[];photo_hashes=[]
    for row in prior:
        path=DATA/(row['source_id']+'.jpg')
        with Image.open(path) as im:
            base.append(F.center_crop(F.resize(im.convert('RGB'),256,InterpolationMode.BILINEAR,antialias=True),[224,224]))
        photo_hashes.append(dict(source_id=row['source_id'],bytes=path.stat().st_size,sha256=sha(path)))
    actual_base=[dict(source_id=r['source_id'],RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest()) for r,im in zip(prior,base)]
    assert actual_base==old_inputs['base']
    policy=TrivialAugmentWide(num_magnitude_bins=31,interpolation=InterpolationMode.NEAREST,fill=128)
    views=[];actual_views=[]
    for epoch in range(16):
        bank=[]
        for i,row in enumerate(prior):
            key='stage-pilot-v1:'+str(epoch)+':'+row['source_id']
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')&((1<<63)-1)
            torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            im=policy(base[i]);bank.append(im)
            actual_views.append(dict(pass_index=epoch,source_id=row['source_id'],seed=seed,RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
        views.append(bank)
    assert actual_views==old_inputs['TA_views']
    orders=old_inputs['orders'];labels=[r['Y'] for r in prior]
    assert len(prior)==64 and len(orders)==16 and all(sorted(order)==list(range(64)) for order in orders)
    save('input-verification.json',dict(old_manifest_sha256=sha(OLD/'input-manifest.json'),
        all64base_and1024TA_RGB_match_old=True,source_photos=photo_hashes,
        previously_used_training_images_only=True,development_or_test_pixels_read=0))
    arms={};compute=0.
    for arm in protocol['arms']:
        model,opt,initial=make_model(arm['freeze_BN_affine'])
        assert initial==old_arm['initial_state_sha256']
        stats_before=bn_stats(model);affine_before=state_hash(bn_affine(model))
        conv_before=model.conv1.weight.detach().cpu().clone()
        checkpoints={'0':dict(train_clean=evaluate(model,base,labels),train_TA_pass0=evaluate(model,views[0],labels))}
        initial_delta=float(np.max(np.abs(np.array(checkpoints['0']['train_clean']['logits'])-
            np.array(old_arm['checkpoints']['0']['train_clean']['logits']))))
        assert initial_delta<.005
        rgb_chain=hashlib.sha256();losses=[];times=[];s=0;arm_start=time.perf_counter()
        with (ROOT/(arm['name']+'-steps.jsonl')).open('x') as trace:
            for epoch in range(16):
                for offset in range(0,64,8):
                    if time.perf_counter()-started>230 or compute>120:raise TimeoutError('fixedwall/compute cap reached')
                    ids=orders[epoch][offset:offset+8];ims=[views[epoch][i] for i in ids]
                    for i,im in zip(ids,ims):
                        rgb_chain.update((str(epoch)+':'+prior[i]['source_id']).encode());rgb_chain.update(im.tobytes())
                    x=torch.stack([tensor(im) for im in ims]);y=torch.tensor([labels[i] for i in ids],dtype=torch.long)
                    torch.mps.synchronize();t=time.perf_counter();loss=step(model,opt,x,y);torch.mps.synchronize()
                    elapsed=time.perf_counter()-t;compute+=elapsed;times.append(elapsed);losses.append(loss);s+=1
                    trace.write(json.dumps(dict(step=s,pass_index=epoch,loss=loss,guarded_compute_seconds=elapsed))+'\n');trace.flush()
                    if s in [64,128]:
                        checkpoints[str(s)]=dict(train_clean=evaluate(model,base,labels),train_TA_pass0=evaluate(model,views[0],labels))
                        print(json.dumps(dict(arm=arm['name'],step=s,clean_train_LL=checkpoints[str(s)]['train_clean']['mean_LL_nats'],
                            clean_train_accuracy=checkpoints[str(s)]['train_clean']['accuracy'],wall_seconds=time.perf_counter()-started)),flush=True)
        assert s==128 and bn_stats(model)==stats_before
        aff_after=state_hash(bn_affine(model));assert (aff_after==affine_before)==arm['freeze_BN_affine']
        assert rgb_chain.hexdigest()==old_arm['exact_training_RGB_order_sha256']
        conv_delta=float((model.conv1.weight.detach().cpu()-conv_before).double().norm());assert conv_delta>0
        params_device=sorted({p.device.type for p in model.parameters()})
        assert params_device==['mps']
        metrics=checkpoints['128']['train_clean']
        out=dict(name=arm['name'],freeze_BN_affine=arm['freeze_BN_affine'],steps=s,source_photos=64,
            training_presentations=1024,initial_state_sha256=initial,old_CPU_initial_logit_maxabs=initial_delta,
            exact_training_RGB_order_sha256=rgb_chain.hexdigest(),BN_running_stats_unchanged=True,
            BN_affine_initial_sha256=affine_before,BN_affine_final_sha256=aff_after,conv1_delta_L2=conv_delta,
            final_state_sha256=state_hash(model.state_dict()),parameter_devices=params_device,
            all_loss_grad_parameter_finite=True,training_compute_seconds=sum(times),
            mean_guarded_step_seconds=sum(times)/128,images_per_second=1024/sum(times),
            wall_seconds=time.perf_counter()-arm_start,checkpoints=checkpoints,
            passes_learning_gate=metrics['accuracy']>=.9 and metrics['mean_LL_nats']<=.2,
            underlearning_gate=metrics['accuracy']<=.8 or metrics['mean_LL_nats']>=.5,
            last32_online_TA_mean_loss=sum(losses[-32:])/32,weights_saved=False)
        save(arm['name']+'-result.json',out);arms[arm['name']]=out
        del model,opt;gc.collect();torch.mps.empty_cache()
    control=arms['affine_trainable'];frozen=arms['affine_frozen']
    if control['underlearning_gate'] and frozen['passes_learning_gate']:
        interpretation='BN_affine_update_changes_learnability_in_this_fixed_development_setting_only'
    elif not control['underlearning_gate']:
        interpretation='old_CPU_underlearning_not_reproduced_on_MPS_cannot_attribute_old_failure_to_BN_affine'
    else:
        interpretation='affine_freeze_did_not_rescue_learning_at_this_fixed_budget_no_further_tuning'
    result=dict(status='completed_single_predeclared_training_only_diagnostic',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        interpretation=interpretation,training_compute_seconds=compute,total_wall_seconds=time.perf_counter()-started,
        protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),torch=torch.__version__,torchvision=torchvision.__version__,
        device='mps',precision='FP32',CPU_fallback=False,MPS_fast_math=False,deterministic_algorithms=False,
        development_test_evaluations=0,new_downloads_or_installs=False,A100=False,ARIS=False,no_originality_or_augmentation_benefit_claim=True,
        no_additional_steps_seeds_or_tuning=True,process_peak_RSS_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sampled_MPS_tensor_memory_bytes=torch.mps.current_allocated_memory(),sampled_MPS_driver_memory_bytes=torch.mps.driver_allocated_memory(),
        final_metrics={n:dict(train_clean=a['checkpoints']['128']['train_clean'],train_TA_pass0=a['checkpoints']['128']['train_TA_pass0'],
            passes_learning_gate=a['passes_learning_gate'],underlearning_gate=a['underlearning_gate']) for n,a in arms.items()})
    save('diagnostic-results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='final_metrics'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('diagnostic-failure.json',dict(status='stopped_no_retuning',type=type(e).__name__,message=str(e)))
        raise
