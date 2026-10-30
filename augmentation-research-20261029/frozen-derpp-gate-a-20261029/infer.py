"""Frozen checkpoint inference and explicitly auxiliary shadow-priority calibration."""
import os,sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_FAST_MATH']='0'
sys.dont_write_bytecode=True
import datetime,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as TF
from audit_math import scores,update_weights,weighted_ids,stats
from prepare import sha,state_fingerprints,savej,array_hash

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'mammoth-derpp-baseline-20261029'
OUT=ROOT/'run-001'
spec=importlib.util.spec_from_file_location('fixed_baseline',BASE/'baseline_v2.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def view_batch(raw,params):
    out=[]
    for im,p in zip(raw,params):
        x=TF.to_pil_image(im);x=TF.pad(x,4,fill=0)
        x=TF.crop(x,int(p[0]),int(p[1]),32,32)
        if p[2]:x=TF.hflip(x)
        out.append(TF.normalize(TF.to_tensor(x),base.MEAN,base.STD))
    return torch.stack(out)

def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter();deadline=started+600
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    freeze=json.loads((ROOT/'protocol-freeze.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in freeze['files'].items())
    assert all(sha(Path(n))==h for n,h in freeze['external_readonly_inputs'].items())
    c=torch.load(BASE/'run-002/task2-final.pt',map_location='cpu',weights_only=False)
    before=state_fingerprints(c)
    assert before==json.loads((ROOT/'input-state-fingerprints.json').read_text())
    resources=base.resource_guard();assert torch.backends.mps.is_available()
    model=base.network();model.load_state_dict(c['model']);model.requires_grad_(False);model.eval();model.to('mps')
    raw=c['buffer']['examples'];z=c['buffer']['logits'].numpy().astype(np.float64)
    assert raw.shape==(500,3,32,32) and z.shape==(500,10)
    for im in raw:
        assert torch.equal(TF.to_tensor(TF.to_pil_image(im)),im)
    # Exact support/padding/flip check against a separate tensor path.
    checks=np.asarray([[0,0,0],[8,8,0],[0,8,1],[4,4,1]],np.int16)
    actual=view_batch(raw[:4],checks);reference=[]
    padded=torch.nn.functional.pad(raw[:4],(4,4,4,4),value=0)
    for i,p in enumerate(checks):
        x=padded[i,:,int(p[0]):int(p[0])+32,int(p[1]):int(p[1])+32]
        if p[2]:x=x.flip(-1)
        reference.append(TF.normalize(x,base.MEAN,base.STD))
    assert torch.equal(actual,torch.stack(reference))
    plan=np.load(ROOT/'frozen-random-plan.npz',allow_pickle=False)
    events=(OUT/'progress.jsonl').open('x',buffering=1)
    def emit(phase,**kw):
        r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,**kw}
        events.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
    def forward(x):
        if time.perf_counter()>deadline:raise TimeoutError('Frozen inference ten-minute budget exceeded')
        return model(x.to('mps')).cpu().numpy()
    try:
        with torch.inference_mode():
            clean=np.concatenate([forward(TF.normalize(raw[a:a+64],base.MEAN,base.STD)) for a in range(0,500,64)])
            torch.mps.synchronize();np.save(OUT/'clean-logits.npy',clean,allow_pickle=False)
            w=np.ones(500,np.float64);q=0.;calib_ids=[];calib_logits=[];qhist=[];whist=[];dhist=[]
            cal_start=time.perf_counter()
            for step in range(160):
                ids=weighted_ids(w,plan['calibration_uniforms'][step])
                h=forward(view_batch(raw[ids],plan['calibration_parameters'][step]))
                d=.5*np.square(h.astype(np.float64)-z[ids]).mean(-1)
                oldq=q;oldw=w[ids].copy();q=update_weights(w,ids,d,q)
                calib_ids.append(ids);calib_logits.append(h);dhist.append(d);qhist.append([oldq,q]);whist.append(np.stack([oldw,w[ids]]))
                if (step+1)%40==0:emit('shadow_calibration',batch=step+1,q=q,weights=stats(w))
            torch.mps.synchronize();cal_seconds=time.perf_counter()-cal_start
            np.savez_compressed(OUT/'shadow-calibration.npz',ids=np.asarray(calib_ids),logits=np.asarray(calib_logits),
               residual=np.asarray(dhist),q_before_after=np.asarray(qhist),w_before_after=np.asarray(whist),
               weights=w,q=np.asarray(q),visits=np.bincount(np.asarray(calib_ids).ravel(),minlength=500))
            logits=np.empty((2,8,500,10),np.float32);score_start=time.perf_counter()
            for group in range(2):
                for view in range(8):
                    params=plan['score_parameters'][group,view]
                    logits[group,view]=np.concatenate([forward(view_batch(raw[a:a+64],params[a:a+64])) for a in range(0,500,64)])
                torch.mps.synchronize();emit('score_group',group=group+1,images=4000)
            score_seconds=time.perf_counter()-score_start
            np.save(OUT/'score-logits.npy',logits,allow_pickle=False)
            # Independent CPU precision check on a predeclared first eight slots, no selection.
            cpu=base.network();cpu.load_state_dict(c['model']);cpu.requires_grad_(False);cpu.eval()
            hx=cpu(view_batch(raw[:8],plan['score_parameters'][0,0,:8])).numpy()
            dif=float(np.max(np.abs(hx-logits[0,0,:8])))
            assert dif<2e-4,('CPU/MPS precision',dif)
            for name,value in model.state_dict().items():assert array_hash(value)==before['model'][name],name
            assert state_fingerprints(c)==before
            assert all(sha(Path(n))==h for n,h in freeze['external_readonly_inputs'].items())
            assert all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters())
            result={'status':'completed_frozen_inference','started_utc':started_utc,
              'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'wall_seconds':time.perf_counter()-started,'shadow_calibration_seconds':cal_seconds,'score_seconds':score_seconds,
              'inference_images':{'clean':500,'shadow':5120,'score':8000,'CPU_precision_extra':8},
              'optimizer_steps':0,'backward_calls':0,'model_and_BN_bitwise_unchanged':True,'buffer_and_targets_unchanged':True,
              'checkpoint_and_source_files_unchanged':True,'q_frozen_during_scoring':True,'raw_PIL_roundtrip_exact_all500':True,
              'PIL_tensor_padding_crop_flip_pixel_check':True,
              'cpu_mps_first8_max_abs':dif,'shadow_weight_stats':stats(w),'shadow_final_Q':q,
              'resources':resources,'memory':base.memory(),'test_read':False,'dev_read':False}
            savej(OUT/'inference-summary.json',result);emit('finished',**result)
    except Exception as exc:
        savej(OUT/'inference-error.json',{'error':repr(exc),'traceback':traceback.format_exc(),'new_optimizer_steps':0})
        raise
    finally:events.close()

if __name__=='__main__':main()
