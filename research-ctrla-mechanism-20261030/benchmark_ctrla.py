import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_FAST_MATH']='0'
import time,json,subprocess,sys,traceback
from ctrla_adapter import *

def main():
    assert (ROOT/'protocol-lock.json').exists()
    budget=Budget();torch.set_num_threads(6);torch.manual_seed(991)
    if (ROOT/'benchmark.json').exists() and not (ROOT/'benchmark-attempt01.json').exists():
        (ROOT/'benchmark-attempt01.json').write_bytes((ROOT/'benchmark.json').read_bytes())
    attempted_train_steps=0
    result={'status':'running','synthetic_only':True,'real_training_images':0,'fallback':False,'dtype':'float32','batch':125}
    t0=time.monotonic()
    try:
        cpu=make_model('cpu').eval();mps=make_model('mps').eval();mps.load_state_dict(cpu.state_dict())
        small=torch.randn(4,3,32,32)
        budget.check(inference=8)
        with torch.inference_mode():
            yc=cpu(small);ym=mps(small.to('mps'));torch.mps.synchronize()
            err=float((yc-ym.cpu()).abs().max());result['cpu_mps_logits_max_abs']=err
            if not torch.allclose(yc,ym.cpu(),atol=2e-4,rtol=2e-3):raise RuntimeError('CPU-MPS eval discrepancy')
        result['model_parameters']=sum(p.numel() for p in mps.parameters())
        result['trainable_parameters']=sum(p.numel() for p in mps.parameters() if p.requires_grad)
        # Compare gradients in train mode, including BatchNorm, before timing.
        cpu.train();mps.train();target=torch.arange(4)%10
        zc=cpu(small);zm=mps(small.to('mps'))
        lc=torch.nn.functional.cross_entropy(zc,target);lm=torch.nn.functional.cross_entropy(zm,target.to('mps'))
        lc.backward();lm.backward();torch.mps.synchronize()
        gc=torch.cat([p.grad.flatten() for p in cpu.parameters() if p.grad is not None])
        gm=torch.cat([p.grad.detach().cpu().flatten() for p in mps.parameters() if p.grad is not None])
        result['cpu_mps_train_loss_abs']=abs(float(lc.detach())-float(lm.detach().cpu()))
        result['cpu_mps_gradient_relative_l2']=float(torch.linalg.vector_norm(gc-gm)/torch.linalg.vector_norm(gc))
        if result['cpu_mps_gradient_relative_l2']>.01:raise RuntimeError('CPU-MPS gradient discrepancy >1%')
        result['layout']='contiguous NCHW (documented compatibility amendment)'
        del cpu
        x=torch.rand(125,3,32,32,device='mps');y=torch.arange(125,device='mps')%10
        opt=new_optimizer(mps,1);mps.train();timings=[]
        for i in range(15):
            budget.check();torch.mps.synchronize();t=time.monotonic()
            attempted_train_steps+=1
            opt.zero_grad(set_to_none=True);z=mps(x);loss=torch.nn.functional.cross_entropy(z,y);loss.backward();opt.step();torch.mps.synchronize()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite synthetic loss')
            timings.append(time.monotonic()-t)
        result['synthetic_train_batch_seconds']=timings
        mps.eval();times=[]
        with torch.inference_mode():
            for i in range(15):
                budget.check(inference=125);torch.mps.synchronize();t=time.monotonic();mps(x);torch.mps.synchronize();times.append(time.monotonic()-t)
        result['inference_batch_seconds']=times
        result['steady_train_images_sec']=125/np.median(timings[5:]);result['steady_inference_images_sec']=125/np.median(times[5:])
        img=Image.fromarray(np.random.default_rng(1).integers(0,256,(32,32,3),dtype=np.uint8))
        aug=official_aug.ControlAugment(Naugs=5,gamma=[.6]*15,skew=[.2]*15,interpolation=InterpolationMode.BILINEAR)
        t=time.monotonic()
        for _ in range(1000):tensor_image(aug(img))
        result['single_cpu_preprocess_images_sec']=1000/(time.monotonic()-t)
        result['train_model_only_hours_for_1840000']=1840000/result['steady_train_images_sec']/3600
        result['frozen_stage_upper_image_forwards']=2*(1114000+690000+302000+11000)+1875+8
        result['inference_model_only_hours']=result['frozen_stage_upper_image_forwards']/result['steady_inference_images_sec']/3600
        result['estimate_note']='Measured model-only floor, not end-to-end promise; CPU rendering, disk, matching and checkpoint overhead are additional. Actual phase times determine continuation.'
        result['mps_allocated_bytes']=torch.mps.current_allocated_memory();result['status']='passed'
    except Exception as e:
        result.update(status='failed',error=str(e),traceback=traceback.format_exc())
    finally:
        result['elapsed_seconds']=time.monotonic()-t0;result['synthetic_timed_train_steps_attempted']=attempted_train_steps
        budget.state['synthetic_train_images_current_attempt']=attempted_train_steps*125+8
        budget.save();write_json(ROOT/'benchmark.json',result);print(json.dumps(result,indent=2),flush=True)
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
