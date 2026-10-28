"""One locked CPU development test. No model imports or original-image access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np

OUT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    with (OUT / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rng(tag):
    return np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16))


def schedule(seed, few, arm):
    r = rng(f'coupling-groups-v1:{seed}:{few}:{arm}')
    result = np.empty((3, 21, 12, 15), dtype=np.int8)
    for c in range(3):
        for b in range(12):
            if arm == 'shared':
                result[c, :, b, :] = r.permutation(21)[:, None]
            else:
                for slot in range(15):
                    result[c, :, b, slot] = r.permutation(21)
    return result.reshape(756, 15)


def probabilities(x, w, b):
    z = x @ w + b
    z -= z.max(axis=-1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(axis=-1, keepdims=True)


def losses(x, w, b, y):
    z = x @ w + b
    m = z.max(axis=-1)
    ll = np.log(np.exp(z - m[..., None]).sum(axis=-1)) + m - np.take_along_axis(z, y[..., None], -1)[..., 0]
    assert np.isfinite(ll).all()
    return ll


def update(x, y, w, b, eta):
    p = probabilities(x, w, b)
    p[np.arange(len(y)), y] -= 1.
    p /= len(y)
    return w - eta * (x.T @ p + .01 * w), b - eta * p.sum(0)


def objective(bank, ids, labels, w, b):
    q = losses(bank[ids], w, b, np.broadcast_to(labels[ids, None], (len(ids), 21)))
    per_class = [float(q[labels[ids] == c].mean()) for c in range(5)]
    return {'class_bank_LL': per_class,
            'mean_bank_LL': float(np.mean(per_class)),
            'objective': float(np.mean(per_class) + .005 * (w*w).sum())}


def run():
    start = time.perf_counter()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    lock = json.loads((OUT / 'protocol-lock.json').read_text())
    assert sha(OUT / 'protocol.json') == lock['protocol_sha256']
    assert sha(__file__) == lock['code_sha256']
    assert not (OUT / 'results.json').exists()
    source = Path(protocol['source'])
    for name, expected in protocol['inputs'].items():
        assert sha(source / name) == expected, name
    old = json.loads((source / 'protocol.json').read_text())
    records = old['roles']['source_seen']
    labels = np.array([r['category'] for r in records])
    cache = np.load(source / 'source-feature-bank.npz')
    norm = np.load(source / 'normalizer.npz')
    # Exact inverse of z=(L2_embedding-mu)/scalar; no PCA, clipping or renormalization.
    bank = np.stack([cache[s] * float(norm['scale']) + norm['mean']
                     for s in protocol['states']], axis=1) * np.sqrt(512.)
    assert bank.shape == (540, 21, 512)
    max_norm_error = float(np.max(np.abs((bank * bank).sum(2) - 512.)))
    assert max_norm_error < 1e-6, max_norm_error
    eta = 1. / (.5 * (513. + 1e-6) + .01)
    trace_checks, exposure_checks, models, source_results, plan_arrays = [], [], {}, [], {}
    for scenario in protocol['scenarios']:
        few = scenario['few_class']
        data = np.asarray(scenario['batch_source_indices'], dtype=np.int64)
        assert data.shape == (12, 15)
        assert all(len(set(row)) == 15 for row in data)
        assert all(np.array_equal(np.bincount(labels[row], minlength=5), [3]*5) for row in data)
        ids, occurrences = np.unique(data, return_counts=True)
        assert len(ids) == 156
        assert all(n == (3 if labels[i] == few else 1) for i, n in zip(ids, occurrences))
        # At theta0: centered gradient = centered_x outer (uniform_probability-onehot_y).
        # This is an arithmetic relevance test only, not a forecast of generalization.
        trace_rows = []
        for row in data:
            dx = bank[row] - bank[row].mean(1, keepdims=True)
            a = np.full((15, 5), .2)
            a[np.arange(15), labels[row]] -= 1
            shared_center = np.einsum('isd,ic->sdc', dx, a) / 15.
            shared_trace = float(np.mean(np.sum(shared_center**2, axis=(1, 2))))
            independent_trace = float(np.sum(np.mean(np.sum(dx**2, axis=2), axis=1) * np.sum(a*a, axis=1)) / 225.)
            covariance = dx.reshape(15, -1) @ dx.reshape(15, -1).T / 21.
            inner_a = a @ a.T
            difference_identity = float((np.sum(covariance * inner_a) - np.sum(np.diag(covariance) * np.diag(inner_a))) / 225.)
            assert np.isclose(shared_trace-independent_trace, difference_identity, rtol=1e-9, atol=1e-10)
            trace_rows.append({'shared': shared_trace, 'independent': independent_trace,
                               'difference': shared_trace-independent_trace,
                               'cross_covariance_identity': difference_identity})
        trace_checks.append({'few_class': few, 'theta': 'zero', 'batches': trace_rows,
                             'mean_difference': float(np.mean([v['difference'] for v in trace_rows]))})
        # Check schedules before any model fit. Every (photo,state) histogram matches.
        for seed in protocol['sampler']['seeds']:
            plans = {arm: schedule(seed, few, arm) for arm in ('shared', 'independent')}
            for arm, plan in plans.items():
                assert np.array_equal(np.sort(plan.reshape(3, 21, 12, 15), axis=1),
                                      np.broadcast_to(np.arange(21)[None, :, None, None], (3, 21, 12, 15)))
                counts = np.zeros((540, 21), dtype=np.int32)
                row_ids = np.tile(data, (63, 1))
                np.add.at(counts, (row_ids.ravel(), plan.ravel()), 1)
                expected = np.zeros_like(counts)
                expected[ids] = occurrences[:, None] * 3
                assert np.array_equal(counts, expected)
                assert plan.shape == row_ids.shape
                plan_arrays[f'few{few}_seed{seed}_{arm}'] = plan
            exposure_checks.append({'few_class': few, 'seed': seed, 'same_per_photo_state_counts': True,
                                    'same_raw_input_order': True, 'updates_each_arm': 756,
                                    'fixed_theta_total_gradient_sum_equal_by_exact_counts': True})
    # Negative controls on source features only.
    data0 = np.asarray(protocol['scenarios'][0]['batch_source_indices'])
    constant_bank = np.broadcast_to(bank[:, :1], bank.shape)
    plan_a = plan_arrays['few0_seed0_shared']
    plan_b = plan_arrays['few0_seed0_independent']
    w0, b0 = np.zeros((512, 5)), np.zeros(5)
    wa, ba, wb, bb = w0.copy(), b0.copy(), w0.copy(), b0.copy()
    for t in range(100):
        row = data0[t % 12]
        wa, ba = update(constant_bank[row, plan_a[t]], labels[row], wa, ba, eta)
        wb, bb = update(constant_bank[row, plan_b[t]], labels[row], wb, bb, eta)
    assert np.array_equal(wa, wb) and np.array_equal(ba, bb)
    save('necessary-checks.json', {'at_utc': now(), 'cache_inverse_squared_norm_max_error': max_norm_error,
        'eta': eta, 'gradient_covariance': trace_checks, 'exposure_checks': exposure_checks,
        'constant_state_or_identical_schedule_control_max_parameter_difference': 0.,
        'fixed_theta_null_does_not_rule_out_trajectory_effects': True,
        'elapsed_seconds': time.perf_counter()-start})
    np.savez_compressed(OUT / 'schedules.npz', **plan_arrays)
    # No gate demanding positive variance: its sign is unrestricted and theta0 is only one point.
    save('training-start.json', {'at_utc': now(), 'protocol_sha256': lock['protocol_sha256'],
        'code_sha256': lock['code_sha256'], 'eta': eta, 'development_features_read': False})
    for scenario in protocol['scenarios']:
        few = scenario['few_class']
        data = np.asarray(scenario['batch_source_indices'], dtype=np.int64)
        ids = np.unique(data)
        for seed in protocol['sampler']['seeds']:
            for arm in ('shared', 'independent'):
                name = f'few{few}_seed{seed}_{arm}'
                plan = plan_arrays[name]
                w, b = np.zeros((512, 5)), np.zeros(5)
                trajectory = [{'step': 0, **objective(bank, ids, labels, w, b)}]
                for step in range(756):
                    row = data[step % 12]
                    w, b = update(bank[row, plan[step]], labels[row], w, b, eta)
                    if (step+1) % 252 == 0:
                        assert np.isfinite(w).all() and np.isfinite(b).all()
                        trajectory.append({'step': step+1, **objective(bank, ids, labels, w, b)})
                    if step % 100 == 0 and time.perf_counter()-start > 120:
                        np.savez_compressed(OUT / 'partial-checkpoint.npz', w=w, b=b)
                        save('runtime-stop.json', {'at_utc': now(), 'name': name, 'step': step+1, 'reason': 'fixed120s wall cap'})
                        raise RuntimeError('fixed120s wall cap; partial saved; no device or parameter change')
                models[name+'_w'], models[name+'_b'] = w, b
                clean = losses(bank[ids, 0], w, b, labels[ids])
                source_results.append({'few_class': few, 'seed': seed, 'arm': arm, 'trajectory': trajectory,
                    'class_clean_LL': [float(clean[labels[ids]==c].mean()) for c in range(5)]})
    np.savez_compressed(OUT / 'readouts.npz', **models)
    save('source-training.json', {'at_utc': now(), 'all_source_models_finished': True,
        'models': source_results, 'elapsed_seconds': time.perf_counter()-start})
    save('training-freeze.json', {'at_utc': now(), 'readout_sha256': sha(OUT/'readouts.npz'),
        'source_results_sha256': sha(OUT/'source-training.json'), 'development_features_read': False,
        'elapsed_seconds': time.perf_counter()-start})
    # Only after all160readouts are frozen, read the already-consumed development role.
    zdev = (cache['development_seen'] * float(norm['scale']) + norm['mean']) * np.sqrt(512.)
    dev = old['roles']['development_seen']
    ydev = np.array([r['category'] for r in dev])
    objects = sorted(set((r['category'], r['instance']) for r in dev))
    assert not set(objects) & set(map(tuple, old['source_objects']))
    per_run, eval_arrays = [], {}
    for scenario in protocol['scenarios']:
        few = scenario['few_class']
        for seed in protocol['sampler']['seeds']:
            pair = {}
            for arm in ('shared', 'independent'):
                name = f'few{few}_seed{seed}_{arm}'
                w, b = models[name+'_w'], models[name+'_b']
                ll = losses(zdev, w, b, ydev)
                pred = (zdev@w+b).argmax(1)
                eval_arrays[name+'_LL'] = ll
                eval_arrays[name+'_pred'] = pred
                objll = [float(ll[[i for i,r in enumerate(dev) if (r['category'],r['instance'])==obj]].mean()) for obj in objects]
                pair[arm] = {'class_LL': [float(ll[ydev==c].mean()) for c in range(5)],
                    'class_accuracy': [float((pred[ydev==c]==c).mean()) for c in range(5)],
                    'object_LL': objll}
            delta = np.array(pair['shared']['class_LL'])-np.array(pair['independent']['class_LL'])
            common = np.delete(delta, few).mean()
            per_run.append({'few_class': few, 'seed': seed, 'arms': pair,
                'class_shared_minus_independent_LL': delta.tolist(), 'rare_delta': float(delta[few]),
                'common_delta': float(common), 'interaction': float(delta[few]-common)})
    np.savez_compressed(OUT / 'evaluation-arrays.npz', **eval_arrays)
    per_seed = [float(np.mean([r['interaction'] for r in per_run if r['seed']==s])) for s in protocol['sampler']['seeds']]
    rare_seed = [float(np.mean([r['rare_delta'] for r in per_run if r['seed']==s])) for s in protocol['sampler']['seeds']]
    common_seed = [float(np.mean([r['common_delta'] for r in per_run if r['seed']==s])) for s in protocol['sampler']['seeds']]
    boot_rng = rng('coupling-groups-v1:bootstrap:paired-schedule-seeds')
    bootstrap_ids = boot_rng.integers(0, 16, (20000, 16))
    def summary(values):
        v=np.array(values); ci=np.quantile(v[bootstrap_ids].mean(1), [.025,.975])
        return {'mean':float(v.mean()), 'CI95_schedule_conditional':ci.tolist(), 'per_schedule_seed':values}
    primary = summary(per_seed)
    mean=primary['mean'];ci=primary['CI95_schedule_conditional']
    result={'at_utc':now(), 'status':'fixed_development_experiment_complete', 'primary_interaction':primary,
        'rare_class_delta':summary(rare_seed), 'common_class_delta':summary(common_seed),
        'screen_passed':bool(abs(mean)>=.01 and (ci[0]>0 or ci[1]<0)),
        'threshold_nat':.01, 'per_scenario_class_table':[{'few_class':c,
            'interaction_mean':float(np.mean([v['interaction'] for v in per_run if v['few_class']==c])),
            'rare_delta_mean':float(np.mean([v['rare_delta'] for v in per_run if v['few_class']==c]))} for c in range(5)],
        'objects':objects, 'runs':per_run, 'wall_seconds':time.perf_counter()-start,
        'CI_interpretation':'Only paired schedule randomness conditional on fixed explored photos and21-statebank. Not new independent data or a population/object CI.',
        'not_claimed':['fullAA/RA/TA result','realminoritygroup conclusion','naturalcausalidentification','originality proven','independent confirmation','a null proves no effect'],
        'GPU':False,'new_encoder_execution':False,'new_images_read':False,'final_role_read':False,
        'protocol_sha256':lock['protocol_sha256'],'code_sha256':lock['code_sha256']}
    save('results.json',result)
    print(json.dumps({k:result[k] for k in ('status','primary_interaction','rare_class_delta','common_class_delta','screen_passed','per_scenario_class_table','wall_seconds')},ensure_ascii=False))


if __name__ == '__main__':
    run()
