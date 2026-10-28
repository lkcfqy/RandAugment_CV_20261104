"""Locked finite acquisition-allocation screen; CPU readouts, no image/encoder calls."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import resource
import time
import numpy as np

OUT = Path(__file__).resolve().parent
START = time.perf_counter()
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def budget():
    assert time.perf_counter() - START < 120, 'fixed120s wall cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2**31, 'fixed2GiB Mac peakRSS cap'


def ce(z, w, b, y):
    logits = z @ w + b
    shift = logits.max(-1)
    return np.log(np.exp(logits-shift[..., None]).sum(-1)) + shift - np.take_along_axis(logits, y[..., None], -1)[..., 0]


def fit(x, labels, weights):
    tx = torch.from_numpy(np.ascontiguousarray(x))
    ty = torch.from_numpy(labels.copy())
    tw = torch.from_numpy(weights.copy())
    assert tx.device.type == 'cpu'
    w = torch.zeros((512, 5), dtype=torch.float64, device='cpu', requires_grad=True)
    b = torch.zeros(5, dtype=torch.float64, device='cpu', requires_grad=True)
    opt = torch.optim.LBFGS([w,b], lr=1., max_iter=300, max_eval=450,
        history_size=30, tolerance_grad=1e-8, tolerance_change=1e-12,
        line_search_fn='strong_wolfe')
    history=[]
    def closure():
        budget()
        opt.zero_grad()
        objective = (torch.nn.functional.cross_entropy(tx@w+b, ty, reduction='none')*tw).sum() + .005*(w*w).sum()
        objective.backward()
        assert torch.isfinite(objective) and torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()
        history.append(float(objective.detach()))
        return objective
    opt.step(closure)
    value = float(closure().detach())
    grad = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    result={'objective':value,'gradient_infinity':grad,'closure_calls':len(history),'closure_objectives':history,
        'optimizer_iterations':int(opt.state[w]['n_iter']),'converged':grad<=1e-5}
    assert grad<=1e-5, f'fixed solver gradient gate failed: {grad}'
    return w.detach().numpy(), b.detach().numpy(), result


def run():
    p=json.loads((OUT/'experiment-protocol.json').read_text())
    lock=json.loads((OUT/'experiment-lock.json').read_text())
    assert sha(OUT/'experiment-protocol.json')==lock['protocol_sha256']
    assert sha(__file__)==lock['code_sha256']
    assert not (OUT/'results.json').exists()
    root=Path(p['source_root'])
    for name,value in p['input_sha256'].items():
        assert sha(root/name)==value, name
    old=json.loads((root/'protocol.json').read_text())
    rows=old['roles']['source_seen']
    labels=np.array([r['category'] for r in rows],dtype=np.int64)
    cache=np.load(root/p['source_feature_bank'])
    norm=np.load(root/'normalizer.npz')
    states=[r['state_id'] for r in p['policies']]
    bank=np.stack([(cache[s]*float(norm['scale'])+norm['mean'])*np.sqrt(512.) for s in states],axis=1)
    assert bank.shape==(540,21,512)
    error=float(np.max(np.abs((bank*bank).sum(2)-512.)))
    assert error<1e-6
    weights={arm:np.array([float(Fraction(r[arm])) for r in p['policies']]) for arm in ('clean','weak','strong')}
    for v in weights.values(): assert np.isclose(v.sum(),1.,rtol=0,atol=1e-14)
    models={};training=[]
    # O/V primary fits first; every single fit still uses the same fixed stopping rule.
    names=sorted(p['datasets'],key=lambda s:(s.startswith('S'),s))
    current=None
    try:
        for name in names:
            ids=np.array(p['datasets'][name],dtype=np.int64)
            assert len(ids) in (60,180)
            assert np.array_equal(np.bincount(labels[ids],minlength=5),[len(ids)//5]*5)
            for arm in ('clean','weak','strong'):
                current=name+'__'+arm
                ix=np.flatnonzero(weights[arm])
                x=bank[ids][:,ix].reshape(-1,512)
                y=np.repeat(labels[ids],len(ix))
                sample_weights=np.tile(weights[arm][ix],len(ids))/len(ids)
                assert np.isclose(sample_weights.sum(),1.)
                assert np.allclose([sample_weights[y==c].sum() for c in range(5)],[.2]*5)
                t=time.perf_counter()
                w,b,details=fit(x,y,sample_weights)
                # Independent final objective/gradient arithmetic in NumPy, no optimization.
                logits=x@w+b
                probs=np.exp(logits-logits.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True)
                probs[np.arange(len(y)),y]-=1.;probs*=sample_weights[:,None]
                numpy_grad=max(float(np.abs(x.T@probs+.01*w).max()),float(np.abs(probs.sum(0)).max()))
                numpy_obj=float(ce(x,w,b,y)@sample_weights+.005*(w*w).sum())
                assert abs(numpy_obj-details['objective'])<1e-10 and numpy_grad<=1e-5
                models[current+'__w'],models[current+'__b']=w,b
                training.append({'dataset':name,'policy':arm,'raw_photos':len(ids),'state_rows':len(y),
                    **details,'numpy_gradient_infinity':numpy_grad,'numpy_objective':numpy_obj,
                    'fit_seconds':time.perf_counter()-t})
                budget()
    except Exception as exc:
        np.savez_compressed(OUT/'partial-readouts.npz',**models)
        save('execution-failure.json',{'at_utc':now(),'during':current,'error':repr(exc),'completed_readouts':len(training),
            'source_training':training,'scientific_effect_status':'not assessed; computational/measurement failure, not falsified',
            'development_array_decoded':False,'wall_seconds':time.perf_counter()-START})
        raise
    assert len(training)==45
    np.savez_compressed(OUT/'readouts.npz',**models)
    save('source-training.json',{'at_utc':now(),'all45converged':True,'records':training,
        'cache_inverse_norm_error':error,'wall_seconds':time.perf_counter()-START})
    save('training-freeze.json',{'at_utc':now(),'readouts_sha256':sha(OUT/'readouts.npz'),
        'training_sha256':sha(OUT/'source-training.json'),'development_array_decoded':False,
        'protocol_sha256':lock['protocol_sha256'],'code_sha256':lock['code_sha256']})
    # Decode only the already-consumed development member, after all45readouts are frozen.
    zdev=(cache['development_seen']*float(norm['scale'])+norm['mean'])*np.sqrt(512.)
    dev=old['roles']['development_seen']
    ydev=np.array([r['category'] for r in dev])
    objects=sorted(set((r['category'],r['instance']) for r in dev))
    assert len(objects)==10 and not set(objects)&set(map(tuple,old['source_objects']))
    obj_indices=[np.array([i for i,r in enumerate(dev) if (r['category'],r['instance'])==o]) for o in objects]
    assert all(len(i)==36 for i in obj_indices)
    evals={};saved={}
    for name in names:
        for arm in ('clean','weak','strong'):
            key=name+'__'+arm;w,b=models[key+'__w'],models[key+'__b']
            loss=ce(zdev,w,b,ydev);prediction=(zdev@w+b).argmax(1)
            assert np.isfinite(loss).all()
            saved[key+'__LL'],saved[key+'__pred']=loss,prediction
            evals[key]={'object_LL':[float(loss[i].mean()) for i in obj_indices],
                'object_accuracy':[float(np.mean(prediction[i]==ydev[i])) for i in obj_indices],
                'mean_LL':float(loss.mean()),'accuracy':float(np.mean(prediction==ydev))}
    np.savez_compressed(OUT/'development-arrays.npz',**saved)
    save('development-summary.json',{'at_utc':now(),'objects':objects,'models':evals})
    def vector(name,arm):return np.array(evals[name+'__'+arm]['object_LL'])
    # Every acquisition assignment is retained; O3 andV3 are repeated references, not new fits.
    config=[]
    for a in p['assignments']:
        o,v=a['datasets']['O'],a['datasets']['V']
        do=vector(o,'strong')-vector(o,'weak')
        dv=vector(v,'strong')-vector(v,'weak')
        config.append({'anchor':a['anchor_ordinal'],'block':a['seed_block'],'datasets':a['datasets'],
            'delta_O':float(do.mean()),'delta_V':float(dv.mean()),'C':float((do-dv).mean()),
            'C_per_object':(do-dv).tolist()})
    delta_o=np.mean([vector(f'O{k}','strong')-vector(f'O{k}','weak') for k in range(3)],axis=0)
    delta_v=np.mean([vector(f'V{k}','strong')-vector(f'V{k}','weak') for k in range(3)],axis=0)
    contrast=delta_o-delta_v
    assert np.allclose(contrast,np.mean([c['C_per_object'] for c in config],axis=0),rtol=0,atol=1e-12)
    # All4^5 ordered resamples: two physical objects within each of five fixed categories.
    group_indices=[[i for i,o in enumerate(objects) if o[0]==c] for c in range(5)]
    assert all(len(g)==2 for g in group_indices)
    pairs=[(0,0),(0,1),(1,0),(1,1)]
    bootstrap=np.array([[group_indices[c][j] for c,choice in enumerate(choices) for j in pairs[choice]]
                        for choices in itertools.product(range(4),repeat=5)])
    assert bootstrap.shape==(1024,10)
    def summary(v):
        ci=np.quantile(v[bootstrap].mean(1),[.025,.975])
        return {'mean':float(v.mean()),'object_resampling_stability_interval95':ci.tolist(),'per_object':v.tolist()}
    primary=summary(contrast);C=primary['mean'];dO=float(delta_o.mean());dV=float(delta_v.mean())
    ci=primary['object_resampling_stability_interval95']
    capability={d:float(np.mean([evals[f'{d}{k}__clean']['accuracy'] for k in range(3)])) for d in ('O','V')}
    capable=all(v>=.4 for v in capability.values())
    sign=np.sign(C);stable=sum(np.sign(c['C'])==sign for c in config) if sign else 0
    effect_gates={'material_interaction':abs(C)>=.02,'opposite_policy_order':dO*dV<0,
        'each_structure_material':abs(dO)>=.01 and abs(dV)>=.01,
        'object_resampling_excludes_zero':ci[0]>0 or ci[1]<0,'atleast7of9_same_sign':stable>=7}
    tierA=capable and all(effect_gates.values())
    winnerO='strong' if dO<0 else 'weak';winnerV='strong' if dV<0 else 'weak'
    gainO=np.mean([vector(f'O{k}','clean')-vector(f'O{k}',winnerO) for k in range(3)],axis=0)
    gainV=np.mean([vector(f'V{k}','clean')-vector(f'V{k}',winnerV) for k in range(3)],axis=0)
    tierB=tierA and gainO.mean()>=.01 and gainV.mean()>=.01
    means={d:{arm:float(np.mean([evals[name+'__'+arm]['mean_LL'] for name in names if name.startswith(d)]))
              for arm in ('clean','weak','strong')} for d in ('S','O','V')}
    result={'at_utc':now(),'status':'fixed_CPU_development_screen_complete','primary_C':primary,
        'delta_O':summary(delta_o),'delta_V':summary(delta_v),'acquisition_policy_risks':means,
        'validity':{'all45converged':True,'capability_clean_accuracy':capability,'capability_passed':capable},
        'effect_gates':effect_gates,'same_direction_configurations':int(stable),'configurations':config,
        'tier_A_pool_flip_passed':bool(tierA),'tier_B_operational_choice_passed':bool(tierB),
        'RA_motivated_signed_flip_supported':bool(tierA and C<0),
        'winner_within_augmented_pools':{'O':winnerO,'V':winnerV},
        'winner_gain_over_clean':{'O':summary(gainO),'V':summary(gainV)},
        'CI_scope':'Exact1024 empirical within-class physical-object resamples; conditional stability on this explored data/source pool. Not a population confirmation, not9independent acquisition trials.',
        'all_objects_previously_explored':True,'old_final_arrays_read':False,'new_encoder_execution':False,
        'CPU_threads':torch.get_num_threads(),'torch_version':torch.__version__,'numpy_version':np.__version__,
        'wall_seconds':time.perf_counter()-START,'peak_RSS_bytes_Mac':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'protocol_sha256':lock['protocol_sha256'],'code_sha256':lock['code_sha256']}
    budget()
    save('results.json',result)
    print(json.dumps({k:result[k] for k in ('status','primary_C','delta_O','delta_V','acquisition_policy_risks','validity','effect_gates','tier_A_pool_flip_passed','tier_B_operational_choice_passed','wall_seconds','peak_RSS_bytes_Mac')},ensure_ascii=False))


if __name__=='__main__':
    run()
