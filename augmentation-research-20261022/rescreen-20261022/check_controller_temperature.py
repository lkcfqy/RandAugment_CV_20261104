"""CPU-only replay of existing development logits. Does not import model code."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'cpu-feasibility-plan.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stable_mis(z, y, temp, gamma):
    # cos(softmax(z/T), e_y) == sqrt(softmax(2*z/T)_y).
    relative = 2 * (z - z[np.arange(len(y)), y, None]) / temp
    relative[np.arange(len(y)), y] = -np.inf
    other_max = relative.max(axis=1)
    log_other = other_max + np.log(np.exp(relative - other_max[:, None]).sum(axis=1))
    log_score = -0.5 * gamma * np.logaddexp(0.0, log_other)
    return np.exp(log_score), log_score

def rank(values, ids):
    order = np.lexsort((ids, values))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    return ranks

def summary(values):
    return {str(q): float(np.quantile(values, q, method='linear')) for q in [0, .5, .9, .95, 1]}

def main():
    started = time.perf_counter()
    cfg = json.loads(PLAN.read_text())
    source = Path(cfg['source_project']) / cfg['source_npz']
    manifest_path = Path(cfg['source_project']) / 'augmentation-research-20261022/pilot-20261022/fit-manifest.json'
    input_before = {str(p): sha(p) for p in [source, manifest_path, PLAN, Path(__file__)]}
    rows = json.loads(manifest_path.read_text())['rows']
    dev = {r['source_id']: r['class_id'] for r in rows if r['candidate_partition'] == 'development'}
    with np.load(source, allow_pickle=False) as arr:
        z = arr['logits'].astype(np.float64)
        y = arr['labels'].copy()
        ids = arr['source_ids'].astype(str)
        pred = arr['predictions'].copy()
    assert z.shape == (589, 37) and len(y) == len(ids) == 589
    assert np.isfinite(z).all() and len(set(ids)) == len(ids)
    assert set(ids) == set(dev)
    assert all(dev[s] == int(c) for s, c in zip(ids, y))
    assert np.array_equal(z.argmax(axis=1), pred)
    gamma = 2.0 / math.log(z.shape[1])
    scores, log_scores, ranks, matched = {}, {}, {}, {}
    checks = {}
    for t in [0.5, 1.0, 2.0]:
        scores[t], log_scores[t] = stable_mis(z, y, t, gamma)
        assert np.isfinite(scores[t]).all() and np.all((scores[t] > 0) & (scores[t] <= 1))
        soft = np.exp((z - z.max(axis=1, keepdims=True)) / t)
        soft /= soft.sum(axis=1, keepdims=True)
        direct = (soft[np.arange(len(y)), y] / np.linalg.norm(soft, axis=1)) ** gamma
        gap = float(np.max(np.abs(scores[t] - direct)))
        assert gap < 1e-12
        assert np.array_equal((z / t).argmax(axis=1), pred)
        ranks[t] = rank(log_scores[t], ids)
        checks[str(t)] = {'formula_max_abs_gap': gap, 'argmax_identical': True,
                          'exact_log_score_ties': len(y) - len(np.unique(log_scores[t]))}
    sorted_base = np.sort(scores[1.0])
    for t in scores:
        matched[t] = sorted_base[ranks[t]]
        assert np.array_equal(np.sort(matched[t]), sorted_base)
    contrasts = {}
    for a,b in [(1.,2.),(.5,2.)]:
        raw_diff = np.abs(scores[a] - scores[b])
        rank_diff = np.abs(ranks[a] - ranks[b]) / (len(y)-1)
        matched_diff = np.abs(matched[a] - matched[b])
        inversions = 0
        for i in range(len(y)-1):
            inversions += int(np.sum((ranks[a][i] - ranks[a][i+1:]) * (ranks[b][i] - ranks[b][i+1:]) < 0))
        contrasts[f'{a:g}_vs_{b:g}'] = {
            'raw_MIS_absolute_change_quantiles': summary(raw_diff),
            'raw_MIS_mean_absolute_change': float(raw_diff.mean()),
            'normalized_rank_displacement_quantiles': summary(rank_diff),
            'fraction_moving_at_least_one_decile': float(np.mean(rank_diff >= .1)),
            'pairwise_inversion_fraction': inversions/(len(y)*(len(y)-1)/2),
            'matched_MIS_absolute_change_quantiles': summary(matched_diff),
            'matched_MIS_mean_absolute_change': float(matched_diff.mean()),
            'sorted_magnitude_multiset_identical': True,
        }
    main_result = contrasts['1_vs_2']
    gate = (main_result['raw_MIS_absolute_change_quantiles']['0.95'] >= .1 and
            main_result['fraction_moving_at_least_one_decile'] >= .1)
    assert all(sha(p) == value for p,value in input_before.items())
    elapsed = time.perf_counter()-started
    assert elapsed < cfg['resources']['max_seconds']
    with (HERE/'controller-temperature-per-photo.csv').open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['source_id','label','prediction']+[f'{kind}_T{t:g}' for t in scores for kind in ['MIS','log_MIS','rank','matched_MIS']])
        for i,s in enumerate(ids):
            writer.writerow([s,int(y[i]),int(pred[i])]+[v for t in scores for v in [float(scores[t][i]),float(log_scores[t][i]),int(ranks[t][i]),float(matched[t][i])]])
    result = {
        'status': 'completed_necessary_condition_only',
        'at_utc': datetime.now(timezone.utc).isoformat(),
        'source_kind': 'previously_consumed_development_logits_from_one_non_SRA_trained_model',
        'photographs': len(y), 'classes': z.shape[1], 'existing_errors': int(np.sum(pred != y)),
        'epsilon': 2.0, 'gamma': gamma,
        'temperature_mean_MIS': {str(t): float(s.mean()) for t,s in scores.items()},
        'formula_and_prediction_checks': checks, 'contrasts': contrasts,
        'preregistered_engineering_gate_passed': bool(gate),
        'decision': 'eligible_for_one_paired_image_CPU_diagnostic_only' if gate else 'do_not_advance_this_candidate_to_images_or_training',
        'input_hashes_before_and_after_equal': input_before,
        'wall_seconds': elapsed, 'CPU_threads_env': 1,
        'training_runs': 0, 'images_decoded': 0, 'GPU_calls': 0,
        'limitations': [
            'MIS sensitivity is implied by the published formula; it is not a new theorem.',
            'No temperature was fitted or shown to improve calibration.',
            'No SRA model was trained; no augmentation harm or training benefit was measured.',
            'Existing development data are exploratory and not new confirmation.',
            'No conditional photo bootstrap is used as training-seed evidence.'
        ]
    }
    with (HERE/'controller-temperature-result.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
