"""One CPU metadata mechanism experiment. Standard library; no model or images."""
import argparse
import bisect
import csv
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

SEED = 20261021
HASHES = {
    'fg_bg_ratios_val.json': 'e32c61f46f06627f524c1579e5523f14f57d08dce37acecd7c81fb4fdef72e4e',
    'hf_val_indices.json': 'ce1abaf2263f63b4e71c7ba8b3c6ef566042bfa588fc9bdd01f1c8f7ecbbf715',
}


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_metadata(folder):
    raw = {}
    receipt = {}
    for name, digest in HASHES.items():
        data = (folder / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        raw[name] = json.loads(data)
        receipt[name] = {'bytes': len(data), 'sha256': digest}
    groups = defaultdict(list)
    indices = raw['hf_val_indices.json']
    for key, ratio in raw['fg_bg_ratios_val.json'].items():
        p = Path(key)
        cls = p.parent.name
        source = cls + '/' + p.stem
        assert source in indices and 0 <= ratio <= 1
        groups[cls].append((source, float(ratio), indices[source]))
    assert len(groups) == 1000
    return groups, receipt


def split(groups, classes, seed):
    bg_raw, foreground_raw = {}, {}
    for cls in classes:
        ordered = sorted(groups[cls], key=lambda r: h('split:' + str(seed) + ':' + r[0]))
        cut = len(ordered) // 2
        bg_raw[cls] = ordered[:cut]
        foreground_raw[cls] = ordered[cut:]
    n_foreground = min(len(foreground_raw[c]) for c in classes)
    assert n_foreground > 0
    foreground = {c: foreground_raw[c][:n_foreground] for c in classes}
    background = {c: [r for r in bg_raw[c] if r[1] < .8] for c in classes}
    empty = [c for c in classes if not background[c]]
    if empty:
        raise ValueError('No eligible background support for: ' + ','.join(empty))
    bg_ids = {r[0] for c in classes for r in bg_raw[c]}
    fg_ids = {r[0] for c in classes for r in foreground[c]}
    assert not (bg_ids & fg_ids)
    return background, foreground, bg_raw, foreground_raw


def mutual_information(rows):
    n = len(rows)
    marginal = [sum(row[j] for row in rows) / n for j in range(n)]
    value = 0.0
    for row in rows:
        assert abs(sum(row) - 1) < 1e-10 and min(row) >= -1e-14
        value += sum(p * math.log2(p / marginal[j])
                     for j, p in enumerate(row) if p > 0) / n
    assert value >= -1e-12
    return max(0.0, value), marginal


def kernel(background, foreground, classes, width, keep_rows=False):
    nc = len(classes)
    inv = round(1 / width)
    normal_bins = round(.8 * inv)
    def bin_id(ratio):
        return normal_bins if ratio >= .8 else int(Decimal(str(ratio)) * inv)
    base_bin = [0.0] * normal_bins
    counts_bg = []
    counts_fg = []
    for cls in classes:
        cb = Counter(bin_id(r[1]) for r in background[cls])
        cf = Counter(bin_id(r[1]) for r in foreground[cls])
        counts_bg.append(cb)
        counts_fg.append(cf)
        for b in range(normal_bins):
            base_bin[b] += cb[b] / len(background[cls]) / nc
    bin_source = []
    for b in range(normal_bins):
        bin_source.append([counts_bg[j][b] / len(background[c]) / nc / base_bin[b]
                           if base_bin[b] else 1 / nc
                           for j, c in enumerate(classes)])
    bin_source.append([1 / nc] * nc)
    fg_weights = [[counts_fg[j][b] / len(foreground[c])
                   for b in range(normal_bins + 1)] for j, c in enumerate(classes)]
    fg_bin = [sum(w[b] for w in fg_weights) / nc for b in range(normal_bins + 1)]
    rows = []
    for weights in fg_weights:
        row = [sum(weights[b] * bin_source[b][j] for b in range(normal_bins + 1))
               for j in range(nc)]
        rows.append(row)
    mi, source_marginal = mutual_information(rows)
    matched_control_mi, control_marginal = mutual_information([source_marginal] * nc)
    global_control_mi, _ = mutual_information([[1 / nc] * nc] * nc)
    assert max(abs(a - b) for a, b in zip(control_marginal, source_marginal)) < 1e-12
    assert matched_control_mi < 1e-12 and global_control_mi < 1e-12
    fallback = fg_bin[-1] + sum(fg_bin[b] for b in range(normal_bins) if not base_bin[b])
    bg_bin_matched = [fg_bin[b] + fallback * base_bin[b] if base_bin[b] else 0.0
                      for b in range(normal_bins)]
    assert abs(sum(bg_bin_matched) - 1) < 1e-12
    # Verify the image-level marginal formula, not just class margins.
    image_marginal_by_class = []
    for cls in classes:
        total = 0.0
        for image in background[cls]:
            b = bin_id(image[1])
            total += (fg_bin[b] / base_bin[b] + fallback) / nc / len(background[cls])
        image_marginal_by_class.append(total)
    margin_error = max(abs(a - b) for a, b in zip(image_marginal_by_class, source_marginal))
    assert margin_error < 1e-12
    result = {
        'class_count': nc, 'bin_width': width,
        'foreground_per_class': len(foreground[classes[0]]),
        'foreground_count': sum(map(len, foreground.values())),
        'eligible_background_count': sum(map(len, background.values())),
        'mi_bits': mi,
        'matched_image_marginal_random_control_mi_bits': matched_control_mi,
        'global_class_balanced_random_control_mi_bits': global_control_mi,
        'same_source_class_probability': sum(rows[i][i] for i in range(nc)) / nc,
        'random_control_same_source_class_probability': 1 / nc,
        'area_bin_match_probability': 1 - fallback,
        'matched_marginal_random_area_bin_match_probability': sum(
            fg_bin[b] * bg_bin_matched[b] for b in range(normal_bins)),
        'global_class_balanced_random_area_bin_match_probability': sum(
            fg_bin[b] * base_bin[b] for b in range(normal_bins)),
        'fallback_probability': fallback,
        'source_marginal_tv_from_uniform': sum(abs(p - 1 / nc) for p in source_marginal) / 2,
        'image_marginal_verification_max_error': margin_error,
        'foreground_bin_masses': fg_bin,
        'background_base_bin_masses': base_bin,
    }
    if keep_rows:
        result['_rows'] = rows
        result['_source_marginal'] = source_marginal
    return result


def csv_payload(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path,
                        default=Path(__file__).parent.parent / 'filter-audit-20261021' / 'metadata')
    args = parser.parse_args()
    root = Path(__file__).parent
    outputs = ['results.json', 'primary-split.csv', 'primary-policy-kernel.csv']
    assert all(not (root / x).exists() for x in outputs), 'Preserving existing results'
    try:
        os.nice(10)
    except OSError:
        pass
    wall0, cpu0 = time.perf_counter(), time.process_time()
    assert abs(mutual_information([[1.0, 0.0], [0.0, 1.0]])[0] - 1) < 1e-12
    assert mutual_information([[.5, .5], [.5, .5]])[0] < 1e-12
    groups, provenance = load_metadata(args.metadata_dir)
    classes = sorted(groups, key=lambda c: h('cohort:' + str(SEED) + ':' + c))[:100]
    bg, fg, bg_raw, fg_raw = split(groups, classes, SEED)
    primary = kernel(bg, fg, classes, .1, keep_rows=True)
    conditional_rows = primary.pop('_rows')
    matched_marginal = primary.pop('_source_marginal')
    print(json.dumps({'phase': 'primary_complete', 'mi_bits': primary['mi_bits']}, ensure_ascii=False), flush=True)
    all_fg = [r for c in classes for r in fg[c]]
    shuffled = list(all_fg)
    rng = random.Random(SEED + 9900)
    null = []
    n = len(fg[classes[0]])
    for trial in range(99):
        rng.shuffle(shuffled)
        perm_fg = {c: shuffled[j*n:(j+1)*n] for j, c in enumerate(classes)}
        r = kernel(bg, perm_fg, classes, .1)
        assert max(abs(a-b) for a,b in zip(r['foreground_bin_masses'], primary['foreground_bin_masses'])) < 1e-12
        null.append(r['mi_bits'])
    print(json.dumps({'phase': 'permutations_complete', 'n': len(null)}, ensure_ascii=False), flush=True)
    robustness = []
    for seed in range(SEED + 1, SEED + 5):
        b, f, _, _ = split(groups, classes, seed)
        r = kernel(b, f, classes, .1)
        r['split_seed'] = seed
        robustness.append(r)
    widths = [kernel(bg, fg, classes, width) for width in (.05, .2)]
    all_classes = sorted(groups)
    b_all, f_all, _, _ = split(groups, all_classes, SEED)
    all_class_point = kernel(b_all, f_all, all_classes, .1)
    ordered = sorted(null)
    rank_p = (1 + sum(x >= primary['mi_bits'] for x in null)) / 100
    passed = primary['mi_bits'] >= .01 and primary['mi_bits'] > max(null)
    split_rows = []
    for cls in classes:
        used_f = {r[0] for r in fg[cls]}
        for fold, images in [('background', bg_raw[cls]), ('foreground', fg_raw[cls])]:
            for source, ratio, index in images:
                split_rows.append(dict(source_id=source, source_class=cls,
                                       source_hf_validation_index=index, ratio=ratio,
                                       fold=fold, used_in_primary=int((fold=='background' and ratio<.8) or (fold=='foreground' and source in used_f))))
    kernel_rows = [dict(foreground_source_class=y, background_source_class=c,
                        area_match_conditional_probability=conditional_rows[i][j],
                        matched_image_marginal_random_probability=matched_marginal[j],
                        global_class_uniform_random_probability=1/len(classes))
                   for i,y in enumerate(classes) for j,c in enumerate(classes)]
    result = {
        'experiment': 'Area-only matching versus image-marginal-preserving random pairing',
        'status': 'completed_metadata_mechanism_only',
        'preregistered_signal_passed': passed,
        'not_visual_or_causal_model_evidence': True,
        'hypothesis_scope': 'Dependence of nominal foreground source class and removed-object source class under a proposed geometry selector, not ForAug default behavior',
        'seed': SEED, 'primary_classes': classes, 'input_provenance': provenance,
        'primary': primary,
        'permutation_reference': {'count':99, 'values_bits':null, 'max_bits':max(null),
                                  'mean_bits':sum(null)/len(null), 'rank_fraction':rank_p,
                                  'interpretation':'Reference conditional on this finite metadata pool, not a natural-population confidence interval'},
        'other_split_points': robustness, 'bin_width_sensitivity_points': widths,
        'all_1000_class_point': all_class_point,
        'integrity': {'foreground_background_source_overlap':0,
                      'equal_foreground_count_per_class':True,
                      'controls_match_induced_background_image_marginal':True,
                      'sparse_monte_carlo_plugin_MI_not_used':True,
                      'no_hyperparameter_selection_on_result':True},
        'execution': {'python':sys.version, 'executable':sys.executable,
                      'dependency':'Python standard library only', 'cpu_workers':1,
                      'wall_seconds':time.perf_counter()-wall0,
                      'cpu_seconds':time.process_time()-cpu0,
                      'GPU_or_model_inference':False, 'image_payloads_read':False,
                      'data_downloads':0, 'ARIS_used':False,
                      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'protocol_sha256':hashlib.sha256((root/'experiment.md').read_bytes()).hexdigest()},
    }
    blobs = {'results.json':(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode(),
             'primary-split.csv':csv_payload(split_rows,list(split_rows[0])),
             'primary-policy-kernel.csv':csv_payload(kernel_rows,list(kernel_rows[0]))}
    for name,payload in blobs.items():
        with (root/name).open('xb') as f:
            f.write(payload)
    print(json.dumps({'status':result['status'],'signal_passed':passed,
                      'primary_mi_bits':primary['mi_bits'],'permutation_max_bits':max(null),
                      'rank_fraction':rank_p,'all_class_mi_bits':all_class_point['mi_bits'],
                      'wall_seconds':result['execution']['wall_seconds'],
                      'cpu_seconds':result['execution']['cpu_seconds']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
