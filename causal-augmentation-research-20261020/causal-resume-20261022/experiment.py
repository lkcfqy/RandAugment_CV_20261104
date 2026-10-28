"""One frozen development screen: class-aligned vs object-balanced joint nuisance weights."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as F
from collections import defaultdict
import hashlib
import itertools
import json
import resource
import time
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
START = time.perf_counter()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(name, data):
    with (OUT / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def budget():
    assert time.perf_counter() - START < 120, '120s whole-script scientific wall cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2**31, '2GiB Mac peakRSS cap'

def ce(x, w, b, y):
    logits = x @ w + b
    shift = logits.max(1)
    return np.log(np.exp(logits - shift[:, None]).sum(1)) + shift - logits[np.arange(len(y)), y]

def aggregation(rows, weights, keys):
    sums = defaultdict(F)
    for row, weight in zip(rows, weights):
        sums[tuple(row[key] for key in keys)] += weight
    return dict(sums)

def design(p, rows):
    n = len(rows)
    assert n == 540
    assert all(row['camera'] == 0 for row in rows)
    ordinals = {(int(c), int(instance)): j for c, values in p['object_ordinal_by_class'].items()
                for j, instance in enumerate(values)}
    assert {(r['category'],r['instance']) for r in rows} == set(ordinals)
    expected = set(itertools.product(p['factor_levels']['azimuth'], p['factor_levels']['elevation'], p['factor_levels']['lighting']))
    for obj in ordinals:
        selected = [r for r in rows if (r['category'],r['instance']) == obj]
        assert len(selected) == 36
        assert {(r['azimuth'],r['elevation'],r['lighting']) for r in selected} == expected
    t = list(map(F, p['coefficients']))
    matrix = [list(map(F, r)) for r in p['control_matrix']]
    assert all(sum(r) == 0 for r in matrix)
    assert all(sorted(r[j] for r in matrix) == sorted(t) for j in range(3))
    weights = {'U': [F(1, n)] * n}
    coefficient_map = {}
    for k in range(5):
        coefficient_map[str(k)] = []
        for kind in ('B', 'S'):
            values = []
            for row in rows:
                y, instance = row['category'], row['instance']
                a = p['factor_levels']['azimuth'].index(row['azimuth'])
                l = p['factor_levels']['lighting'].index(row['lighting'])
                sign_a = -1 if a < 3 else 1
                sign_l = -1 if l == 0 else 1
                coefficient = t[(y + k) % 5] if kind == 'B' else matrix[(y + k) % 5][ordinals[(y, instance)]]
                values.append((1 + F(4, 5) * coefficient * sign_a * sign_l) / n)
            weights[kind + str(k)] = values
        for (category, instance), j in ordinals.items():
            coefficient_map[str(k)].append({'category': category, 'instance': instance, 'ordinal': j,
                 'B': str(t[(category+k)%5]), 'S': str(matrix[(category+k)%5][j])})
    ref = weights['U']
    checks = {}
    for name, w in weights.items():
        assert sum(w) == 1 and min(w) > 0
        for keys in [('category',), ('category','instance'), ('category','azimuth'),
                     ('category','elevation'), ('category','lighting'), ('azimuth','elevation','lighting')]:
            assert aggregation(rows, w, keys) == aggregation(rows, ref, keys), (name, keys)
        if name.startswith('S'):
            assert aggregation(rows, w, ('category','azimuth','elevation','lighting')) == aggregation(rows, ref, ('category','azimuth','elevation','lighting'))
        eff = 1 / sum(v*v for v in w)
        checks[name] = {'weight_sum': str(sum(w)), 'minimum_weight': str(min(w)), 'maximum_weight': str(max(w)),
                        'Kish_image_weight_N_eff_exact': str(eff), 'Kish_image_weight_N_eff': float(eff),
                        'raw_photos': n, 'physical_objects': 15}
    for k in range(5):
        assert sorted(weights['B'+str(k)]) == sorted(weights['S'+str(k)])
    for kind in ('B', 'S'):
        assert all(sum(weights[kind+str(k)][i] for k in range(5))/5 == ref[i] for i in range(n))
    for category, instance in ordinals:
        mapped = [next(r for r in coefficient_map[str(k)] if (r['category'],r['instance']) == (category,instance)) for k in range(5)]
        assert sorted(F(r['B']) for r in mapped) == sorted(F(r['S']) for r in mapped)
    save('design-checks.json', {'at_utc': now(), 'arithmetic': 'exact rational, metadata only, before feature reads',
         'all_matching_checks_passed': True, 'checks': checks, 'actual_object_coefficient_map': coefficient_map,
         'weight_histograms_B_S_equal_each_k': True, 'per_object_coefficient_multisets_equal_across_k': True,
         'S_class_full_condition_joint_uniform': True, 'mean_k_weights_equal_U': True,
         'independence_limit': 'Kish row count is a weight concentration diagnostic, not a count of independent physical objects.'})
    return {key: np.array([float(v) for v in value]) for key,value in weights.items()}

def fit(x, y, weights):
    tx = torch.from_numpy(np.ascontiguousarray(x))
    ty = torch.from_numpy(y.copy())
    tw = torch.from_numpy(weights.copy())
    assert tx.device.type == 'cpu' and tx.dtype == torch.float64
    w = torch.zeros((512,5), dtype=torch.float64, device='cpu', requires_grad=True)
    b = torch.zeros(5, dtype=torch.float64, device='cpu', requires_grad=True)
    opt = torch.optim.LBFGS([w,b], lr=1., max_iter=300, max_eval=450, history_size=30,
                          tolerance_grad=1e-8, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    history = []
    def closure():
        budget()
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(tx@w+b,ty,reduction='none')*tw).sum() + .005*(w*w).sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()
        history.append(float(loss.detach()))
        return loss
    opt.step(closure)
    value = float(closure().detach())
    grad = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    wn, bn = w.detach().numpy(), b.detach().numpy()
    logits = x@wn+bn
    prob = np.exp(logits-logits.max(1,keepdims=True)); prob /= prob.sum(1,keepdims=True)
    prob[np.arange(len(y)), y] -= 1
    prob *= weights[:,None]
    ng = max(float(np.abs(x.T@prob+.01*wn).max()), float(np.abs(prob.sum(0)).max()))
    nv = float(ce(x,wn,bn,y)@weights + .005*(wn*wn).sum())
    assert abs(value-nv) < 1e-10 and max(grad,ng) <= 1e-5
    return wn, bn, {'objective': value, 'gradient_infinity': grad, 'numpy_objective': nv,
                     'numpy_gradient_infinity': ng, 'converged': True, 'closure_objectives': history,
                     'optimizer_iterations': int(opt.state[w]['n_iter'])}

def run():
    p = json.loads((OUT/'protocol.json').read_text())
    lock = json.loads((OUT/'protocol-lock.json').read_text())
    assert sha(OUT/'protocol.json') == lock['protocol_sha256']
    assert sha(__file__) == lock['code_sha256']
    assert not (OUT/'results.json').exists()
    root = Path(p['source_root'])
    for name, digest in p['input_sha256'].items():
        assert sha(root/name) == digest
    old = json.loads((root/'protocol.json').read_text())
    rows = old['roles']['source_seen']
    y = np.array([r['category'] for r in rows], dtype=np.int64)
    photo_weights = design(p,rows)
    cache = np.load(root/'source-feature-bank.npz')
    norm = np.load(root/'normalizer.npz')
    bank = np.stack([(cache[state['id']]*float(norm['scale'])+norm['mean'])*np.sqrt(512.) for state in p['states']], axis=1)
    assert bank.shape == (540,11,512)
    norm_error = float(np.abs((bank*bank).sum(2)-512.).max())
    assert norm_error < 1e-6
    state_weights = np.array([float(F(s['weak_weight'])) for s in p['states']])
    assert abs(state_weights.sum()-1) < 1e-14
    names = ['U'] + [kind+str(k) for k in range(5) for kind in ('B','S')]
    models, training = {}, []
    source_accuracy = {}
    current = None
    try:
        for name in names:
            for policy in ('clean','weak'):
                current = name+'__'+policy
                if policy == 'clean':
                    x = bank[:,0,:]
                    labels = y
                    wts = photo_weights[name]
                else:
                    x = bank.reshape(-1,512)
                    labels = np.repeat(y,11)
                    wts = (photo_weights[name][:,None]*state_weights[None,:]).ravel()
                t = time.perf_counter()
                w,b,details = fit(x,labels,wts)
                models[current+'__w'], models[current+'__b'] = w,b
                source_accuracy[current] = float(np.mean((bank[:,0,:]@w+b).argmax(1) == y))
                training.append({'key': current, 'state_rows': len(labels), 'fit_seconds': time.perf_counter()-t,
                                 'clean_source_accuracy': source_accuracy[current], **details})
                budget()
    except Exception as exc:
        np.savez_compressed(OUT/'partial-readouts.npz', **models)
        save('execution-failure.json', {'at_utc': now(), 'during': current, 'error': repr(exc),
             'completed_models': len(training), 'source_training': training, 'development_array_decoded': False,
             'scientific_status': 'measurement incomplete; effect not assessed'})
        raise
    assert len(training) == 22
    np.savez_compressed(OUT/'readouts.npz', **models)
    save('source-training.json', {'at_utc': now(), 'all_22_converged': True, 'records': training,
         'cache_inverse_norm_error': norm_error, 'source_fit_wall_seconds': time.perf_counter()-START})
    save('training-freeze.json', {'at_utc': now(), 'readouts_sha256': sha(OUT/'readouts.npz'),
         'source_training_sha256': sha(OUT/'source-training.json'), 'protocol_sha256': lock['protocol_sha256'],
         'code_sha256': lock['code_sha256'], 'development_array_decoded': False})
    # source-feature-bank contains this already-consumed development member; no evaluation-features archive or final-role access.
    dev = old['roles']['development_seen']
    z = (cache['development_seen']*float(norm['scale'])+norm['mean'])*np.sqrt(512.)
    yd = np.array([r['category'] for r in dev], dtype=np.int64)
    objects = sorted({(r['category'],r['instance']) for r in dev})
    assert len(objects) == 10 and not set(objects)&set(map(tuple,old['source_objects']))
    indices = [np.array([i for i,r in enumerate(dev) if (r['category'],r['instance']) == obj]) for obj in objects]
    assert all(len(i) == 36 for i in indices)
    arrays, summaries = {}, {}
    for name in names:
        for policy in ('clean','weak'):
            key = name+'__'+policy
            w,b = models[key+'__w'], models[key+'__b']
            logits = z@w+b
            loss = ce(z,w,b,yd)
            pred = logits.argmax(1)
            arrays[key+'__logits'], arrays[key+'__LL'], arrays[key+'__pred'] = logits,loss,pred
            summaries[key] = {'mean_LL': float(loss.mean()), 'accuracy': float(np.mean(pred == yd)),
                               'object_LL': [float(loss[ix].mean()) for ix in indices],
                               'object_accuracy': [float(np.mean(pred[ix] == yd[ix])) for ix in indices]}
    np.savez_compressed(OUT/'development-arrays.npz', **arrays)
    save('development-summary.json', {'at_utc': now(), 'objects': objects, 'models': summaries})
    def v(name, policy):
        return np.array(summaries[name+'__'+policy]['object_LL'])
    configs, contrasts = [], []
    for k in range(5):
        db = v('B'+str(k),'weak') - v('B'+str(k),'clean')
        ds = v('S'+str(k),'weak') - v('S'+str(k),'clean')
        j = db-ds
        contrasts.append(j)
        configs.append({'k': k, 'delta_B': float(db.mean()), 'delta_S': float(ds.mean()),
                        'J_star': float(j.mean()), 'J_star_per_object': j.tolist()})
    primary = np.mean(contrasts,axis=0)
    groups = [[i for i,o in enumerate(objects) if o[0] == c] for c in range(5)]
    assert all(len(g) == 2 for g in groups)
    pair = [(0,0),(0,1),(1,0),(1,1)]
    bootstrap = np.array([[groups[c][j] for c, choice in enumerate(choices) for j in pair[choice]]
                          for choices in itertools.product(range(4),repeat=5)])
    def summary(value):
        return {'mean': float(value.mean()), 'object_stability_interval95': np.quantile(value[bootstrap].mean(1),[.025,.975]).tolist(),
                'per_object': value.tolist()}
    primary_summary = summary(primary)
    capability = {'U_source_clean_accuracy': source_accuracy['U__clean'],
                  'B_development_clean_accuracy_mean': float(np.mean([summaries['B'+str(k)+'__clean']['accuracy'] for k in range(5)])),
                  'S_development_clean_accuracy_mean': float(np.mean([summaries['S'+str(k)+'__clean']['accuracy'] for k in range(5)]))}
    valid = bool(capability['U_source_clean_accuracy'] >= .8 and capability['B_development_clean_accuracy_mean'] >= .4 and capability['S_development_clean_accuracy_mean'] >= .4)
    gates = {'numerical_all22_passed': True, 'capability_passed': valid,
             'J_star_at_least_0_02': bool(primary_summary['mean'] >= .02),
             'stability_interval_lower_positive': bool(primary_summary['object_stability_interval95'][0] > 0)}
    result = {'at_utc': now(), 'status': 'completed_fixed_development_screen', 'primary_J_star': primary_summary,
              'configs': configs, 'gates': gates, 'screen_passed': bool(all(gates.values())),
              'delta_U': summary(v('U','weak')-v('U','clean')),
              'delta_B_mean': summary(np.mean([v('B'+str(k),'weak')-v('B'+str(k),'clean') for k in range(5)],axis=0)),
              'delta_S_mean': summary(np.mean([v('S'+str(k),'weak')-v('S'+str(k),'clean') for k in range(5)],axis=0)),
              'capability': capability,
              'risks': {name: {policy: summaries[name+'__'+policy]['mean_LL'] for policy in ('clean','weak')} for name in names},
              'independent_unit': '10 already-consumed physical development objects, two in each of five fixed categories',
              'uncertainty_scope': '1024 exhaustive within-category ordered object resamples; fixed source15 and fixed encoder/weights. Not fresh population confirmation.',
              'no_final_arrays_read': True, 'source_photos': 540, 'source_objects': 15, 'development_photos': 360,
              'MPS_used_for_scientific_experiment': False, 'device': 'CPU', 'CPU_threads': 1,
              'wall_seconds': time.perf_counter()-START, 'peak_RSS_bytes_Mac': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    budget()
    save('results.json', result)
    print(json.dumps({key: result[key] for key in ('status','primary_J_star','gates','screen_passed','wall_seconds','peak_RSS_bytes_Mac')}, ensure_ascii=False))

if __name__ == '__main__':
    run()
