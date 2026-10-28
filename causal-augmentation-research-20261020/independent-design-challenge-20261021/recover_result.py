"""Recover the locked summary from frozen outputs only; never fit or encode."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json
import time
import numpy as np

OUT=Path(__file__).resolve().parent
start=time.perf_counter()
p=json.loads((OUT/'experiment-protocol.json').read_text())
lock=json.loads((OUT/'experiment-lock.json').read_text())
freeze=json.loads((OUT/'training-freeze.json').read_text())
source=json.loads((OUT/'source-training.json').read_text())
dev=json.loads((OUT/'development-summary.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(OUT/'experiment-protocol.json')==lock['protocol_sha256']
assert sha(OUT/'experiment.py')==lock['code_sha256']
assert sha(OUT/'readouts.npz')==freeze['readouts_sha256']
assert sha(OUT/'source-training.json')==freeze['training_sha256']
assert len(source['records'])==45 and source['all45converged']
evals=dev['models'];objects=dev['objects'];names=sorted(p['datasets'])
def vector(name,arm):return np.array(evals[name+'__'+arm]['object_LL'])
config=[]
for a in p['assignments']:
    o,v=a['datasets']['O'],a['datasets']['V']
    do=vector(o,'strong')-vector(o,'weak');dv=vector(v,'strong')-vector(v,'weak')
    config.append({'anchor':a['anchor_ordinal'],'block':a['seed_block'],'datasets':a['datasets'],
        'delta_O':float(do.mean()),'delta_V':float(dv.mean()),'C':float((do-dv).mean()),'C_per_object':(do-dv).tolist()})
delta_o=np.mean([vector(f'O{k}','strong')-vector(f'O{k}','weak') for k in range(3)],axis=0)
delta_v=np.mean([vector(f'V{k}','strong')-vector(f'V{k}','weak') for k in range(3)],axis=0)
contrast=delta_o-delta_v
assert np.allclose(contrast,np.mean([c['C_per_object'] for c in config],axis=0),rtol=0,atol=1e-12)
groups=[[i for i,o in enumerate(objects) if o[0]==c] for c in range(5)]
assert all(len(g)==2 for g in groups)
pairs=[(0,0),(0,1),(1,0),(1,1)]
bootstrap=np.array([[groups[c][j] for c,v in enumerate(choices) for j in pairs[v]] for choices in itertools.product(range(4),repeat=5)])
assert bootstrap.shape==(1024,10)
def summary(v):
    return {'mean':float(v.mean()),'object_resampling_stability_interval95':np.quantile(v[bootstrap].mean(1),[.025,.975]).tolist(),'per_object':v.tolist()}
primary=summary(contrast);C=primary['mean'];dO=float(delta_o.mean());dV=float(delta_v.mean());ci=primary['object_resampling_stability_interval95']
capability={d:float(np.mean([evals[f'{d}{k}__clean']['accuracy'] for k in range(3)])) for d in ('O','V')}
capable=all(v>=.4 for v in capability.values());sign=np.sign(C)
stable=int(sum(np.sign(c['C'])==sign for c in config)) if sign else 0
gates={'material_interaction':bool(abs(C)>=.02),'opposite_policy_order':bool(dO*dV<0),
    'each_structure_material':bool(abs(dO)>=.01 and abs(dV)>=.01),
    'object_resampling_excludes_zero':bool(ci[0]>0 or ci[1]<0),'atleast7of9_same_sign':bool(stable>=7)}
tierA=bool(capable and all(gates.values()))
winnerO='strong' if dO<0 else 'weak';winnerV='strong' if dV<0 else 'weak'
gainO=np.mean([vector(f'O{k}','clean')-vector(f'O{k}',winnerO) for k in range(3)],axis=0)
gainV=np.mean([vector(f'V{k}','clean')-vector(f'V{k}',winnerV) for k in range(3)],axis=0)
tierB=bool(tierA and gainO.mean()>=.01 and gainV.mean()>=.01)
means={d:{arm:float(np.mean([evals[name+'__'+arm]['mean_LL'] for name in names if name.startswith(d)])) for arm in ('clean','weak','strong')} for d in ('S','O','V')}
result={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'fixed_CPU_screen_complete_summary_recovered_without_retraining',
    'primary_C':primary,'delta_O':summary(delta_o),'delta_V':summary(delta_v),'acquisition_policy_risks':means,
    'validity':{'all45converged':True,'capability_clean_accuracy':capability,'capability_passed':capable},
    'effect_gates':gates,'same_direction_configurations':stable,'configurations':config,
    'tier_A_pool_flip_passed':tierA,'tier_B_operational_choice_passed':tierB,'RA_motivated_signed_flip_supported':bool(tierA and C<0),
    'winner_within_augmented_pools':{'O':winnerO,'V':winnerV},'winner_gain_over_clean':{'O':summary(gainO),'V':summary(gainV)},
    'CI_scope':'Exact1024 empirical within-class physical-object resamples, conditional stability only; not new object-population confirmation or9independent acquisition trials.',
    'all_objects_previously_explored':True,'old_final_arrays_read':False,'new_encoder_execution':False,
    'source_training_wall_seconds':source['wall_seconds'],'full_original_run_wall_seconds':None,
    'resource_scope':'Executed CPU tensors and1thread guards,120s/2GiB guards passed; final original totalwall/peakRSS numbers lost at JSON serialization and are not reconstructed.',
    'summary_recovery_seconds':time.perf_counter()-start,'protocol_sha256':lock['protocol_sha256'],'code_sha256':lock['code_sha256'],
    'recovery_code_sha256':sha(Path(__file__)),'serialization_failure_preserved':'results-serialization-failed.json.partial'}
with (OUT/'results.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    f.write('\n')
print(json.dumps({k:result[k] for k in ('status','primary_C','delta_O','delta_V','acquisition_policy_risks','validity','effect_gates','tier_A_pool_flip_passed','tier_B_operational_choice_passed','source_training_wall_seconds','summary_recovery_seconds')},ensure_ascii=False))
