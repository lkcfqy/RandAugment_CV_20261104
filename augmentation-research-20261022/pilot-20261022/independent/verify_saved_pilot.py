"""Independent CPU-only audit. Do not execute until all four cells completed.

No import of common/train/analyze, forward pass, optimization or image decoding.
Torch is imported lazily only for CPU weights_only loading of small checkpoints.
"""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import traceback
import numpy as np

ROOT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/pilot-20261022')
OUT = Path('/Users/lkc/Documents/Codex/2026-10-22/task-2/lora-ra-pilot-20261022/independent')
PROTOCOL_SHA = '9386fc378f876cf12c68ef5d7248f073d7313433d158350b4143942ac1b52a3f'
CELLS = ['front-weak', 'front-ra', 'back-weak', 'back-ra']
SIGNS = np.array([1, -1, -1, 1], dtype=np.int64)
CHECKS = []


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def close(name, actual, expected, atol=1e-10, rtol=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    check(name, a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
          and np.allclose(a, b, atol=atol, rtol=rtol))
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def save(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def summarize(z, y):
    # Independent stable formulation: logaddexp reduction, not the main exp/sum.
    z = z.astype(np.float64)
    pred = z.argmax(axis=1)
    error = (pred != y).astype(np.int64)
    nll = np.logaddexp.reduce(z, axis=1) - z[np.arange(len(y)), y]
    per_class = [float(error[y == c].mean()) for c in range(37)]
    return error, nll, {'N': len(y), 'errors': int(error.sum()),
                        'error': float(error.mean()), 'accuracy': float(1 - error.mean()),
                        'NLL': float(nll.mean()), 'balanced_error': float(np.mean(per_class)),
                        'per_class_error': per_class}


def load_cpu_state(path):
    import torch
    if not getattr(load_cpu_state, '_configured', False):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        load_cpu_state._configured = True
    value = torch.load(path, map_location='cpu', weights_only=True)
    state = {}
    for name, t in value['trainable_state'].items():
        check(str(path.name) + ':' + name + ':CPU_tensor', t.device.type == 'cpu')
        state[name] = t.detach().numpy().copy()
    return value, state


def zero_K_structure():
    code = ast.parse((ROOT / 'common.py').read_bytes())
    cls = next(n for n in code.body if isinstance(n, ast.ClassDef) and n.name == 'QVLoRA')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    expected = ast.parse('''def forward(self,w):
    q=self.bq@self.aq;v=self.bv@self.av
    return w+torch.cat((q,torch.zeros_like(q),v),dim=0)
''').body[0]
    check('locked_QV_forward_has_Q_zeroK_V_order', ast.dump(forward) == ast.dump(expected))


def main():
    # No scientific output is opened before this completion gate.
    if not (ROOT / 'results/completed.json').is_file():
        raise SystemExit('Not executed: await explicit four-cell completion notification.')
    cfg = read(ROOT / 'protocol.json')
    lock = read(ROOT / 'protocol-lock.json')
    done = read(ROOT / 'results/completed.json')
    check('expected_protocol_SHA', sha(ROOT / 'protocol.json') == PROTOCOL_SHA == lock['protocol_sha256'])
    check('all_four_completed_same_protocol', done['status'] == 'all_four_cells_completed'
          and done['protocol_sha256'] == PROTOCOL_SHA and done['extra_seeds'] == 0)
    check('all_locked_sources_unchanged', all(sha(ROOT / n) == d for n, d in lock['code_sha256'].items()))
    check('all_locked_evidence_unchanged', all(sha(ROOT / n) == d for n, d in cfg['evidence_sha256'].items()))
    check('fixed_endpoint_and_primary', cfg['epochs'] == 30 and cfg['seed'] == 20261022
          and cfg['batch'] == 16 and cfg['engineering_gate_I_min'] == .01
          and cfg['primary_metric'] == 'source-photo micro classification error on clean development images')
    manifest = read(ROOT / 'fit-manifest.json')
    check('manifest_SHA', sha(ROOT / 'fit-manifest.json') == cfg['fit_manifest_sha256'])
    dev = [r for r in manifest['rows'] if r['candidate_partition'] == 'development']
    train = [r for r in manifest['rows'] if r['candidate_partition'] == 'train']
    ids = np.array([r['source_id'] for r in dev])
    labels = np.array([r['class_id'] for r in dev], dtype=np.int64)
    check('final_counts_and_unique_development_units', len(dev) == 589 and len(train) == 591
          and len(set(ids.tolist())) == len(ids)
          and len({r['duplicate_cluster_id'] for r in manifest['rows']}) == len(manifest['rows'])
          and set(labels.tolist()) == set(range(37)))
    endpoint = cfg['epochs']
    epoch_points = [0] + cfg['evaluation_epochs']
    steps_per_epoch = math.ceil(len(train) / cfg['batch'])
    expected_updates = endpoint * steps_per_epoch
    expected_presentations = endpoint * len(train)
    check('endpoint_counts', expected_updates == cfg['data']['updates_per_cell'] == 1110
          and expected_presentations == cfg['data']['presentations_per_cell'] == 17730)
    zero_K_structure()

    per_epoch = {}
    per_cell = {}
    tensors_by_epoch = {}
    states_initial = {}
    parsed_summaries = {}
    inputs = {}
    views = {}
    frozen = {}
    feature_checks = {}
    parameter_checks = {}
    max_nll_diff = 0.0
    max_parameter_diff = 0.0
    start = read(ROOT / 'results/started.json')
    copy = read(ROOT / 'copy-receipt.json')
    dt = datetime.datetime.fromisoformat
    check('saved_record_lock_copy_start_complete_order', dt(cfg['frozen_at_utc']) < dt(lock['at_utc'])
          < dt(copy['at_utc']) < dt(start['at_utc']) < dt(done['at_utc']))
    check('actual_CPU_nice_no_fallback_receipt', start['actual_process_nice'] == 10
          and start['CPU_threads'] == 1 and not start['MPS_fallback']
          and not start['CUDA_available'] and start['protocol_sha256'] == PROTOCOL_SHA)
    check('complete_wall_cap_respected', 0 < done['wall_seconds'] <= cfg['wall_cap_seconds'])

    for cell in CELLS:
        folder = ROOT / 'results' / cell
        summary = read(folder / 'summary.json')
        parsed_summaries[cell] = summary
        layout, aug = cell.split('-')
        check(cell + ':identity', summary['cell'] == cell and summary['layout'] == layout
              and summary['augmentation'] == aug and summary['seed'] == cfg['seed']
              and summary['epochs'] == endpoint and summary['updates'] == expected_updates
              and summary['source_presentations'] == expected_presentations)
        curve_list = read(folder / 'curve.json')
        check(cell + ':all_curve_epochs', [p['epoch'] for p in curve_list] == list(range(endpoint + 1)))
        curves = {p['epoch']: p for p in curve_list}
        check(cell + ':all_declared_NPZ_only',
              {int(p.stem.rsplit('epoch', 1)[1]) for p in folder.glob('development-epoch*.npz')} == set(epoch_points))
        saved = {}
        for epoch in epoch_points:
            path = folder / ('development-epoch%03d.npz' % epoch)
            a = arrays(path)
            z = a['logits']
            check(cell + ':epoch%d:original_logits_FP32_finite' % epoch,
                  z.dtype == np.float32 and z.shape == (len(ids), 37) and np.isfinite(z).all())
            check(cell + ':epoch%d:IDs_labels_predictions' % epoch,
                  np.array_equal(a['source_ids'], ids) and np.array_equal(a['labels'], labels)
                  and np.array_equal(a['predictions'], z.argmax(axis=1)))
            e, nll, metrics = summarize(z, labels)
            reference = curves[epoch]['development']
            for key in ['error', 'balanced_error', 'NLL']:
                difference = close(cell + ':epoch%d:%s' % (epoch, key), metrics[key], reference[key])
                if key == 'NLL':
                    max_nll_diff = max(max_nll_diff, difference)
            close(cell + ':epoch%d:per_class' % epoch,
                  metrics['per_class_error'], 1 - np.array(reference['per_class_accuracy']))
            check(cell + ':epoch%d:feature_shape_finite' % epoch,
                  a['selected_features'].shape == (8, 768) and np.isfinite(a['selected_features']).all())
            saved[epoch] = a
            tensors_by_epoch[(cell, epoch)] = (e, nll)
            per_epoch.setdefault(str(epoch), {})[cell] = metrics
            inputs[str(path.relative_to(ROOT))] = sha(path)
        for key in ['error', 'balanced_error', 'NLL']:
            close(cell + ':final_summary_' + key, per_epoch[str(endpoint)][cell][key], summary['final_development'][key])
        initial, final = saved[0], saved[endpoint]
        feature_rmse = float(np.sqrt(np.mean((final['selected_features'] - initial['selected_features']) ** 2)))
        check(cell + ':features_changed', math.isfinite(feature_rmse) and feature_rmse > 0)
        close(cell + ':feature_RMSE', feature_rmse, summary['fixed_eight_feature_RMSE'], atol=1e-8, rtol=1e-6)
        feature_checks[cell] = {'RMSE': feature_rmse, 'initial_final_8_only': True}
        frozen[cell] = summary['frozen_parameter_SHA256_before']
        check(cell + ':frozen_digest_receipt', frozen[cell] == summary['frozen_parameter_SHA256_after']
              and re.fullmatch('[a-f0-9]{64}', frozen[cell]) is not None)
        opt = read(folder / 'optimizer-actual.json')
        check(cell + ':optimizer_group_sizes_LR',
              [g['parameters'] for g in opt['groups']] == [36864, 28453]
              and [g['initial_lr'] for g in opt['groups']] == [.0003, .001]
              and all(g['weight_decay'] == .01 for g in opt['groups'])
              and opt['defaults']['betas'] == [.9, .999] and opt['defaults']['eps'] == 1e-8)
        steps = [json.loads(t) for t in (folder / 'steps.jsonl').read_text().splitlines() if t]
        check(cell + ':actual_logged_updates', len(steps) == expected_updates)
        for j, s in enumerate(steps):
            epoch = j // steps_per_epoch + 1
            expected_size = min(cfg['batch'], len(train) - (j % steps_per_epoch) * cfg['batch'])
            warm = max(1, int(.1 * expected_updates))
            factor = (j + 1) / warm if j < warm else .5 * (1 + math.cos(math.pi * (j - warm) / (expected_updates - warm)))
            assert s['step'] == j + 1 and s['epoch'] == epoch and s['examples'] == expected_size
            assert math.isfinite(s['mean_CE']) and s['mean_CE'] >= 0 and s['all_gradients_finite']
            assert abs(s['adapter_lr'] - .0003 * factor) < 1e-15
            assert abs(s['head_lr'] - .001 * factor) < 1e-15
        check(cell + ':all_logged_step_counts_finite_and_schedule', True)
        check(cell + ':source_exposure_from_steps', sum(s['examples'] for s in steps) == expected_presentations)
        for epoch in range(1, endpoint + 1):
            part = steps[(epoch - 1) * steps_per_epoch:epoch * steps_per_epoch]
            mean = sum(p['mean_CE'] * p['examples'] for p in part) / len(train)
            close(cell + ':epoch%d:logged_training_CE' % epoch, mean, curves[epoch]['training_mean_CE'])
            check(cell + ':epoch%d:training_count' % epoch, curves[epoch]['examples'] == len(train))
        views[cell] = read(folder / 'view-hashes.json')
        check(cell + ':view_hash_epochs', [v['epoch'] for v in views[cell]] == list(range(endpoint)))
        check(cell + ':all_view_digests_well_formed',
              all(re.fullmatch('[a-f0-9]{64}', v[k]) is not None for v in views[cell]
                  for k in ['ordered_RGB_sha256', 'ordered_tensor_sha256', 'source_order_sha256']))

        meta0, state0 = load_cpu_state(folder / 'initial-trainable-checkpoint.pt')
        meta1, state1 = load_cpu_state(folder / 'trainable-checkpoint.pt')
        check(cell + ':checkpoint_hash_and_metadata', sha(folder / 'trainable-checkpoint.pt') == summary['trainable_checkpoint_sha256']
              and all(m['protocol_sha256'] == PROTOCOL_SHA and m['layout'] == layout
                      and m['augmentation'] == aug and m['seed'] == cfg['seed'] for m in [meta0, meta1]))
        check(cell + ':checkpoint_key_sets', set(state0) == set(state1) and len(state1) == 14)
        specification = read(folder / 'trainable-parameters.json')
        check(cell + ':trainable_numel_and_shapes', sum(a.size for a in state1.values()) == specification['total'] == 65317
              and set(p['name'] for p in specification['parameters']) == set(state1)
              and all(list(state1[p['name']].shape) == p['shape'] and state1[p['name']].size == p['numel'] for p in specification['parameters']))
        h = hashlib.sha256()
        for v in state0.values():
            h.update(v.tobytes())
        check(cell + ':initial_state_digest', h.hexdigest() == summary['initial_slot_and_head_sha256'])
        grouped = {}
        for name, a in state1.items():
            check(cell + ':' + name + ':finite_FP32', a.dtype == np.float32 and np.isfinite(a).all())
            change = float(np.abs(a - state0[name]).max())
            max_parameter_diff = max(max_parameter_diff,
                close(cell + ':' + name + ':change', change, summary['parameter_absmax_changes'][name], atol=1e-8, rtol=1e-6))
            if name.startswith('heads.'):
                check(cell + ':' + name + ':head_updated', change > 0)
            else:
                prefix, suffix = name.rsplit('.', 1)
                grouped.setdefault(prefix, {})[suffix] = (state0[name], a)
        positions = sorted(int(re.search(r'encoder_layer_(\d+)\.', p).group(1)) for p in grouped)
        check(cell + ':layout_blocks_and_QV_only', positions == ([0, 1, 2] if layout == 'front' else [9, 10, 11])
              and len(grouped) == 3 and all(set(g) == {'aq', 'bq', 'av', 'bv'} for g in grouped.values()))
        slot_states = []
        effective = {}
        for prefix in sorted(grouped, key=lambda p: int(re.search(r'encoder_layer_(\d+)\.', p).group(1))):
            g = grouped[prefix]
            check(cell + ':' + prefix + ':r4_shapes_and_B0',
                  all(g[k][1].shape == shape for k, shape in [('aq', (4, 768)), ('av', (4, 768)), ('bq', (768, 4)), ('bv', (768, 4))])
                  and np.count_nonzero(g['bq'][0]) == 0 and np.count_nonzero(g['bv'][0]) == 0)
            q = g['bq'][1].astype(np.float64) @ g['aq'][1].astype(np.float64)
            v = g['bv'][1].astype(np.float64) @ g['av'][1].astype(np.float64)
            norms = {'Q_delta_frobenius': float(np.linalg.norm(q)), 'V_delta_frobenius': float(np.linalg.norm(v)),
                     'Bq_frobenius': float(np.linalg.norm(g['bq'][1].astype(np.float64))),
                     'Bv_frobenius': float(np.linalg.norm(g['bv'][1].astype(np.float64)))}
            check(cell + ':' + prefix + ':effective_QV_and_B_nonzero', all(math.isfinite(t) and t > 0 for t in norms.values()))
            for key, val in norms.items():
                close(cell + ':' + prefix + ':' + key, val, summary['effective_Q_V_updates'][prefix][key], atol=2e-7, rtol=2e-5)
            check(cell + ':' + prefix + ':K_receipt', summary['effective_Q_V_updates'][prefix]['K_delta_exact_zero_by_construction'])
            effective[prefix] = norms
            slot_states.append({k: value[0] for k, value in g.items()})
        states_initial[cell] = {'slots': slot_states,
                                'head_weight': state0['heads.head.weight'], 'head_bias': state0['heads.head.bias'],
                                'initial_logits': initial['logits'], 'initial_features': initial['selected_features']}
        parameter_checks[cell] = effective
        check(cell + ':MPS_max_observation_contains_all_saved_evaluations',
              summary['max_observed_MPS_driver_bytes'] >= max(c['development']['max_observed_MPS_driver_bytes']
                  for c in curve_list if 'development' in c))
        per_cell[cell] = per_epoch[str(endpoint)][cell]

    reference = states_initial[CELLS[0]]
    for cell in CELLS:
        state = states_initial[cell]
        check(cell + ':common_initial_head_exact', np.array_equal(state['head_weight'], reference['head_weight'])
              and np.array_equal(state['head_bias'], reference['head_bias']))
        check(cell + ':common_A_B_by_slot_exact',
              all(np.array_equal(a[k], b[k]) for a, b in zip(state['slots'], reference['slots']) for k in ['aq', 'av', 'bq', 'bv']))
        close(cell + ':common_initial_logits', state['initial_logits'], reference['initial_logits'], atol=1e-5, rtol=0)
        close(cell + ':common_initial_features', state['initial_features'], reference['initial_features'], atol=1e-5, rtol=0)
    for aug in ['weak', 'ra']:
        check(aug + ':all_RGB_normalized_tensor_source_hashes_paired', views['front-' + aug] == views['back-' + aug])
        check(aug + ':same_layout_frozen_base_initial_digests', frozen['front-' + aug] == frozen['front-weak']
              and frozen['back-' + aug] == frozen['back-weak'])
    check('source_order_same_across_all_four', all([v['source_order_sha256'] for v in views[c]]
          == [v['source_order_sha256'] for v in views[CELLS[0]]] for c in CELLS))
    check('summary_list_matches_completed', {s['cell']: s for s in done['cells']} == parsed_summaries)

    errors = np.column_stack([tensors_by_epoch[(c, endpoint)][0] for c in CELLS])
    nll = np.column_stack([tensors_by_epoch[(c, endpoint)][1] for c in CELLS])
    contribution = errors @ SIGNS
    I = float(contribution.mean())
    Gf = float(np.mean(errors[:, 0] - errors[:, 1]))
    Gb = float(np.mean(errors[:, 2] - errors[:, 3]))
    bootcfg = cfg['bootstrap']
    check('frozen_bootstrap_settings', bootcfg['replicates'] == 5000 and bootcfg['seed'] == 73102
          and bootcfg['quantiles'] == [.025, .975] and bootcfg['method'] == 'numpy.quantile linear')
    rng = np.random.Generator(np.random.PCG64(73102))
    bootstrap = np.zeros(5000, dtype=np.float64)
    for breed in range(37):
        values = contribution[labels == breed]
        draw = rng.integers(0, values.size, size=(5000, values.size))
        bootstrap += np.take(values, draw).sum(axis=1, dtype=np.int64) / labels.size
    ci = np.quantile(bootstrap, [.025, .975], method='linear')
    analysis = read(ROOT / 'analysis/summary.json')
    stored = arrays(ROOT / 'analysis/paired-errors-and-bootstrap.npz')
    check('analysis_input_NPZ_hashes', all(analysis['input_npz_sha256'][c]
          == inputs['results/%s/development-epoch030.npz' % c] for c in CELLS))
    check('analysis_saved_error_matrix_contributions', np.array_equal(stored['source_ids'], ids)
          and np.array_equal(stored['labels'], labels) and np.array_equal(stored['errors'], errors)
          and np.array_equal(stored['interaction_contributions'], contribution))
    bootstrap_max_diff = close('all5000_bootstrap_replicates', bootstrap, stored['bootstrap_interactions'], atol=1e-15, rtol=0)
    close('NLL_array', nll, stored['NLL'])
    close('primary_I', I, analysis['interaction_I'])
    close('G_front', Gf, analysis['G_front'])
    close('G_back', Gb, analysis['G_back'])
    close('bootstrap_quantiles', ci, analysis['paired_stratified_photo_bootstrap_95percent_interval'], atol=1e-15, rtol=0)
    close('NLL_secondary_interaction', float(np.mean(nll @ SIGNS)), analysis['I_NLL_secondary'])
    check('engineering_gate_and_RA_direction_flags', analysis['engineering_gate_passed'] == bool(I >= .01)
          and analysis['RA_improves_front'] == bool(Gf > 0) and analysis['RA_improves_back'] == bool(Gb > 0)
          and not analysis['gate_is_significance_power_or_equivalence'])
    for cell in CELLS:
        for key in ['N', 'errors', 'error', 'accuracy', 'NLL', 'balanced_error']:
            close('analysis:' + cell + ':' + key, per_cell[cell][key], analysis['cells'][cell][key])
    save_arrays_path = OUT / 'independently-recomputed-arrays.npz'
    if save_arrays_path.exists():
        raise FileExistsError(save_arrays_path)
    np.savez_compressed(save_arrays_path, source_ids=ids, labels=labels, errors=errors, NLL=nll,
                        interaction_contributions=contribution, bootstrap_interactions=bootstrap)
    result = {'status': 'passed_independent_saved_run_verification',
              'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'protocol_sha256': PROTOCOL_SHA,
              'protocol_lock_sha256': sha(ROOT / 'protocol-lock.json'),
              'checks_count': len(CHECKS), 'checks_passed': CHECKS,
              'all_saved_development_epochs_recomputed': per_epoch, 'final_cells': per_cell,
              'G_front': Gf, 'G_back': Gb, 'primary_I': I,
              'primary_I_percentage_points': I * 100, 'bootstrap_interval': ci.tolist(),
              'bootstrap_replicates_verified': 5000, 'bootstrap_max_absolute_difference': bootstrap_max_diff,
              'max_NLL_summary_difference': max_nll_diff, 'max_parameter_change_difference': max_parameter_diff,
              'effective_QV_parameter_checks': parameter_checks, 'feature_change_checks': feature_checks,
              'engineering_gate_I_min': .01, 'engineering_gate_passed': bool(I >= .01),
              'no_CI_lower_bound_gate': True, 'single_seed_common_recipe_only': True,
              'not_a_novelty_or_general_mechanism_conclusion': True,
              'source_presentations_per_cell': expected_presentations, 'updates_per_cell': expected_updates,
              'all_source_RGB_tensor_hashes_paired': True,
              'timeline': {'protocol': cfg['frozen_at_utc'], 'lock': lock['at_utc'], 'copy': copy['at_utc'],
                           'started': start['at_utc'], 'completed': done['at_utc']},
              'wall_seconds': done['wall_seconds'], 'input_NPZ_SHA256': inputs,
              'independent_arrays_SHA256': sha(save_arrays_path),
              'verification_boundary': 'Recomputed metrics/5000 bootstrap and CPU arithmetic on saved trainable states; checked frozen-base/gradient/hash receipts and locked code. No independent ViT forward, gradients replay, base-state reconstruction, optimizer replay or retraining.',
              'CPU_threads': 1, 'new_GPU_or_forward_or_training_used': False,
              'reserved_confirmation_pixels_or_old_test_arrays_read': False}
    save(OUT / 'result-verification.json', result)
    print(json.dumps({'status': result['status'], 'I': I, 'interval': ci.tolist(),
                      'gate_passed': result['engineering_gate_passed'], 'checks': len(CHECKS),
                      'output_sha256': sha(OUT / 'result-verification.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        failure = {'status': 'independent_verification_failed_not_retrained', 'error': repr(exc),
                   'traceback': traceback.format_exc(), 'checks_passed': CHECKS,
                   'script_sha256': sha(__file__), 'protocol_sha256_expected': PROTOCOL_SHA}
        path = OUT / 'result-verification-failed.json'
        if not path.exists():
            save(path, failure)
        raise
