"""Verify saved traces, arrays, and every model checkpoint on CPU; no training."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image
import torch,torchvision
from torchvision.transforms import functional as F,InterpolationMode

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
try:os.nice(10)
except OSError:pass
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_hash(d):
    h=hashlib.sha256()
    for k,v in d.items():h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()
def ll(z,y):
    z=z.astype(np.float64);top=z.max(axis=1)
    return top+np.log(np.exp(z-top[:,None]).sum(axis=1))-z[np.arange(len(y)),y]

t=time.perf_counter();r=json.loads((ROOT/'repair-results.json').read_text());protocol=json.loads((ROOT/'protocol.json').read_text())
assert r['status']=='completed' and protocol['recipe_count']==1
assert sha(ROOT/'protocol.json')==r['protocol_sha256'] and sha(ROOT/'repair.py')==r['code_sha256']
assert sha(ROOT/'training-probe-arrays.npz')==r['arrays_sha256'];data=np.load(ROOT/'training-probe-arrays.npz',allow_pickle=False)
rows=json.loads((BASE/'stage-validity-pilot-20261021/protocol.json').read_text())['train_rows'];x=[]
for row in rows[:8]:
    with Image.open(BASE/'mechanism-transfer-pets-20261021/photos'/(row['source_id']+'.jpg')) as im:
        b=F.center_crop(F.resize(im.convert('RGB'),256,InterpolationMode.BILINEAR,antialias=True),[224,224])
        x.append(F.normalize(F.to_tensor(b),[.485,.456,.406],[.229,.224,.225]))
x=torch.stack(x);verified=0;maxloss=0.;checkpoint_checks=[];gate_checks={};maxfirst8=0.
for arm in ['base','TA']:
    a=json.loads((ROOT/(arm+'-result.json')).read_text());trace=[json.loads(l) for l in (ROOT/(arm+'-steps.jsonl')).read_text().splitlines()]
    assert len(trace)==2048 and [q['step'] for q in trace]==list(range(1,2049))
    assert all(np.isfinite(q['loss']) and q['guarded_compute_seconds']>=0 for q in trace)
    for q in trace:
        factor=.1+.9*(q['step']-1)/127 if q['step']<=128 else 1.
        assert q['head_lr']==.01*factor and q['body_lr']==.0001*factor
    old=json.loads((BASE/'training-validity-recovery-20261021'/(arm+'-result.json')).read_text())
    assert a['exact_training_RGB_order_sha256']==old['exact_training_RGB_order_sha256']
    assert a['first256_online_LL']==float(np.mean([q['loss'] for q in trace[:256]]))
    assert a['last256_online_LL']==float(np.mean([q['loss'] for q in trace[-256:]]))
    for step,metrics in a['checkpoints'].items():
        for which,key in [('clean','clean'),('fixed_TA','TA')]:
            z=data[arm+'_'+step+'_'+key+'_logits'];y=data['clean_labels' if key=='clean' else 'TA_probe_labels'];m=metrics[which]
            calc=ll(z,y);err=abs(float(calc.mean())-m['mean_LL_nats']);assert err<5e-7
            assert float((z.argmax(1)==y).mean())==m['accuracy'] and np.array_equal(z,np.array(m['logits'],dtype=np.float32))
            verified+=len(y);maxloss=max(maxloss,err)
        f=data[arm+'_'+step+'_clean_features'];fs=metrics['clean']['features']
        assert int(np.any(f!=0,axis=0).sum())==fs['active_channels']
        assert float(f.astype(np.float64).var(axis=0).mean())==fs['mean_between_sample_variance']
    for c in a['saved_model_checkpoints']:
        path=ROOT/c['file'];assert path.stat().st_size==c['bytes'] and sha(path)==c['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True);state=payload['model']
        assert payload['step']==c['step'] and payload['arm']==arm
        assert payload['protocol_sha256']==r['protocol_sha256'] and payload['code_sha256']==r['code_sha256']
        assert state_hash(state)==c['model_state_sha256']==payload['model_state_sha256']
        bn={k:v for k,v in state.items() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
        assert state_hash(bn)==a['BN_initial_sha256']==payload['BN_statistics_sha256']
        assert all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values())
        model=torchvision.models.resnet18(weights=None);model.fc=torch.nn.Linear(512,2);model.load_state_dict(state,strict=True);model.eval()
        with torch.inference_mode():z=model(x).numpy()
        expected=data[arm+'_'+str(c['step'])+'_clean_logits'][:8];delta=float(np.max(np.abs(z-expected)));assert delta<2e-4
        maxfirst8=max(maxfirst8,delta)
        if c['optimizer_saved']:
            opt=payload['optimizer'];assert len(opt['state'])==62 and [g['lr'] for g in opt['param_groups']]==[.01,.0001]
            assert all(bool(torch.isfinite(v['momentum_buffer']).all()) and v['momentum_buffer'].device.type=='cpu' for v in opt['state'].values())
        else:assert 'optimizer' not in payload
        checkpoint_checks.append(dict(file=c['file'],bytes=c['bytes'],sha256=c['sha256'],strict_model_load=True,CPU_first8_logit_maxabs=delta,optimizer_saved=c['optimizer_saved']))
    ini=a['checkpoints']['0'];fin=a['checkpoints']['2048']
    gates=dict(completed_all_finite=True,
        clean_learning=all(a['checkpoints'][s]['clean']['mean_LL_nats']<=.5*ini['clean']['mean_LL_nats'] for s in ['1536','2048']),
        own_distribution_learning=(arm=='base' or all(a['checkpoints'][s]['fixed_TA']['mean_LL_nats']<=.5*ini['fixed_TA']['mean_LL_nats'] for s in ['1536','2048'])),
        online_learning=a['last256_online_LL']<=.5*a['first256_online_LL'],
        no_old_collapse=(fin['clean']['features']['mean_between_sample_variance']>1e-4*ini['clean']['features']['mean_between_sample_variance'] and fin['clean']['unique_logit_rows']>1))
    assert gates==r['criteria'][arm]==a['success_criteria'];gate_checks[arm]=gates
assert all(all(g.values()) for g in gate_checks.values())==r['single_recipe_recovered']
out=dict(status='passed',verified_saved_predictions=verified,maximum_mean_LL_reconstruction_error=maxloss,
    training_trace_rows_verified=4096,learning_rate_schedule_verified=True,full_RGB_order_chains_match_previous=True,
    all_checkpoints=checkpoint_checks,maximum_checkpoint_CPU_first8_logit_error=maxfirst8,criteria_recomputed=gate_checks,
    device='cpu',model_forwards=len(checkpoint_checks),training_updates=0,heldout_images_read=0,wall_seconds=time.perf_counter()-t)
with (ROOT/'verification.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['all_checkpoints','criteria_recomputed']}))
