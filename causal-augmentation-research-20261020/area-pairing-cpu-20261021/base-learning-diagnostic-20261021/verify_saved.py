"""CPU-only verification of saved arrays, scalar traces, and final states; no model forward or training."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
try:os.nice(10)
except OSError:pass
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_hash(d):
    h=hashlib.sha256()
    for k,v in d.items():
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def feature_stats(a):
    return dict(active_channels=int(np.any(a!=0,axis=0).sum()),all_zero_rows=int(np.all(a==0,axis=1).sum()),
        unique_rows=len(np.unique(a,axis=0)),zero_fraction=float(np.mean(a==0)),
        mean_between_sample_variance_float64=float(a.astype(np.float64).var(axis=0).mean()))
def ll(z,y):
    z=z.astype(np.float64);top=z.max(axis=1)
    return top+np.log(np.exp(z-top[:,None]).sum(axis=1))-z[np.arange(len(y)),y]

r=json.loads((ROOT/'diagnostic-results.json').read_text());data=np.load(ROOT/'training-probe-arrays.npz',allow_pickle=False)
assert sha(ROOT/'diagnose.py')==r['files']['source_sha256']
assert sha(ROOT/'protocol.json')==r['files']['protocol_sha256']
assert sha(ROOT/'training-probe-arrays.npz')==r['files']['arrays_sha256']
assert r['status']=='completed';y=data['labels'];assert y.shape==(64,) and np.bincount(y).tolist()==[32,32]
out=dict(status='passed',model_forwards=0,updates=0,devices_used=['cpu'],source_hashes_match=True,
    individual_saved_predictions_verified=0,probe_array_checks={},state_checks={},trace_checks={})
max_ll_error=0.
for origin in ['cpu','mps']:
    arm=r['arms'][origin]
    path=ROOT/arm['checkpoint_file'];assert path.stat().st_size==arm['checkpoint_bytes'] and sha(path)==arm['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values())
    assert state_hash(state)==arm['final_state_sha256']
    bn={k:v for k,v in state.items() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    assert state_hash(bn)==arm['BN_stats_sha256']=='466c5e05e608c72e52dada1b3807495482d49ad58196c9e2723e7ff2743a5ad1'
    w=state['fc.weight'].numpy();b=state['fc.bias'].numpy();f=data[origin+'_64_features'];z=data[origin+'_64_logits']
    reconstructed=f@w.T+b;head_error=float(np.max(np.abs(reconstructed-z)));assert head_error<2e-5
    out['state_checks'][origin]=dict(hashes_match=True,all_tensors_finite=True,BN_running_statistics_unchanged=True,
        head_reconstruction_maxabs=head_error,head_weight_L2_float64=float(np.linalg.norm(w.astype(np.float64))),
        head_bias=b.tolist(),features64=feature_stats(f))
    for step,metrics in arm['checkpoints'].items():
        tag=origin+'_'+step;z=data[tag+'_logits'];f=data[tag+'_features'];error=abs(float(ll(z,y).mean())-metrics['mean_LL_nats'])
        assert error<3e-7 and float((z.argmax(1)==y).mean())==metrics['accuracy']
        fs=feature_stats(f)
        for key in ['active_channels','all_zero_rows','unique_rows','zero_fraction']:assert fs[key]==metrics['features'][key]
        out['probe_array_checks'][tag]=dict(mean_LL_float64=float(ll(z,y).mean()),accuracy=float((z.argmax(1)==y).mean()),
            saved_LL_error=error,**fs);out['individual_saved_predictions_verified']+=64;max_ll_error=max(max_ll_error,error)
    trace=[json.loads(line) for line in (ROOT/(origin+'-steps.jsonl')).read_text().splitlines()]
    old=BASE/('stage-validity-pilot-20261021/finetune_base-steps.jsonl' if origin=='cpu' else 'training-validity-recovery-20261021/base-steps.jsonl')
    previous=[json.loads(line)['loss'] for line in old.read_text().splitlines()][:64]
    assert len(trace)==64 and [t['step'] for t in trace]==list(range(1,65))
    assert [t['loss'] for t in trace]==previous
    assert all(t['all_parameters_finite'] and math.isfinite(t['loss']) for t in trace)
    assert all(s['finite'] and s['missing_gradients']==0 for t in trace for s in t['gradients'].values())
    assert all(u['actual_delta_L2']>0 for t in trace for u in t['actual_updates'].values())
    update_error=max(u['manual_SGD_parameter_maxabs'] for t in trace for u in t['actual_updates'].values());assert update_error<1e-7
    out['trace_checks'][origin]=dict(steps=64,losses_exactly_match_old=True,missing_gradients=0,
        every_step_has_finite_gradients_and_parameters=True,all_tracked_actual_updates_nonzero=True,
        manual_SGD_parameter_maxabs=update_error)
for tag,m in r['cross_device_same_final_state'].items():
    z=data[tag+'_logits'];f=data[tag+'_features'];error=abs(float(ll(z,y).mean())-m['mean_LL_nats'])
    assert error<3e-7 and float((z.argmax(1)==y).mean())==m['accuracy']
    origin=tag.split('_')[0];delta=float(np.max(np.abs(z-data[origin+'_64_logits'])))
    assert delta==m['logit_difference_from_original_device']['maxabs']
    out['probe_array_checks'][tag]=dict(mean_LL_float64=float(ll(z,y).mean()),accuracy=float((z.argmax(1)==y).mean()),
        saved_LL_error=error,same_state_cross_device_logit_maxabs=delta,**feature_stats(f))
    out['individual_saved_predictions_verified']+=64;max_ll_error=max(max_ll_error,error)
out['maximum_LL_reconstruction_error']=max_ll_error
out['notes']=['FP32 gradient norms calculated on each device in the training trace are approximate diagnostic summaries, not precise cross-device validation. same-state full-gradient comparisons use CPU NumPy float64.',
    'No current or historical held-out images loaded. Checkpoints were loaded as tensors only; no model instantiated or evaluated.',
    'The current CPU replay uses deterministic=False but matches all64 archived CPU losses exactly despite old CPU deterministic=True. This weakens that flag as an explanation for this observed early divergence.']
with (ROOT/'verification.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(status=out['status'],predictions=out['individual_saved_predictions_verified'],maximum_LL_error=max_ll_error,
    trace_matches={k:v['losses_exactly_match_old'] for k,v in out['trace_checks'].items()},state_checks=out['state_checks']),ensure_ascii=False))
