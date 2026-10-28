"""Training-side runtime calibration only. No learned weights or accuracy results retained."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import datetime,hashlib,json,platform,resource,signal,statistics,time
import torch
import torchvision
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/mechanism-transfer-pets-20261021')
WEIGHT=DATA.parent/'encoder-readiness-20261021/runtime/torch-cache/hub/checkpoints/resnet18-f37072fd.pth'

def main():
    try:os.nice(10)
    except OSError:pass
    signal.alarm(90);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    meta=json.loads((ROOT/'scope-and-resource-metadata.json').read_text())
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    preprocess=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    images=[preprocess(Image.open(DATA/'photos'/(r['source_id']+'.jpg')).convert('RGB')) for r in meta['benchmark_training_photos']]
    target=torch.tensor([r['class37_target'] for r in meta['benchmark_training_photos']],dtype=torch.long,device='cpu')
    inputs=torch.stack(images).to('cpu');results=[];start=time.perf_counter()
    for batch in [8,32]:
        x=inputs.repeat(batch//8,1,1,1);y=target.repeat(batch//8)
        for mode in ['frozen_backbone','finetune_backbone']:
            model=torchvision.models.resnet18(weights=None)
            model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
            model.requires_grad_(mode=='finetune_backbone')
            torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(0).get_state())
            model.fc=torch.nn.Linear(512,37,device='cpu');model.train().to('cpu')
            for layer in model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
            body=[v for n,v in model.named_parameters() if not n.startswith('fc.') and v.requires_grad]
            # LR0 preserves weights; timing includes backward, momentum state and step overhead.
            groups=[dict(params=model.fc.parameters(),lr=0.)]
            if body:groups.append(dict(params=body,lr=0.))
            optimizer=torch.optim.SGD(groups,momentum=.9,weight_decay=1e-4)
            timings=[]
            for i in range(8):
                t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y)
                assert torch.isfinite(loss)
                loss.backward();optimizer.step()
                if i>=2:timings.append(time.perf_counter()-t)
            assert all(p.device.type=='cpu' for p in model.parameters())
            result=dict(mode=mode,batch=batch,warmup_steps=2,measured_steps=6,
                mean_step_seconds=statistics.mean(timings),median_step_seconds=statistics.median(timings),
                minimum_step_seconds=min(timings),images_per_second=batch/statistics.mean(timings),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                BN_running_stats_fixed=True,weights_and_optimizer_discarded=True,
                photo_decoding_and_augmentation_excluded_from_timing=True,
                learning_rate_zero_no_weight_drift=True,not_training_convergence_evidence=True)
            results.append(result);print(json.dumps(result),flush=True)
            del model,optimizer
    out=dict(status='CPU_runtime_calibration_completed_no_research_result',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        results=results,CPU_only=True,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        torch=torch.__version__,torchvision=torchvision.__version__,system=platform.system(),
        process_peak_RSS_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        wall_seconds=time.perf_counter()-start,benchmark_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data='eight existing original-training-side photos repeated to batch32; no heldout pixels',
        learned_weights_saved=False,validation_or_test_evaluations=0,ARIS=False,GPU=False,A100=False)
    with (ROOT/'cpu-training-timing.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2))

if __name__=='__main__':main()
