"""Exact pre-result design review on a synthetic index grid, no training/data IO."""
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json

rows = list(product(range(5), range(3), range(6), range(3), range(2)))
t = list(map(F, [-1, '-1/2', 0, '1/2', 1]))
C = [list(map(F, row)) for row in [
    [-1, 0, 1], ['-1/2', '1/2', 0], [0, 1, -1],
    ['1/2', -1, '1/2'], [1, '-1/2', '-1/2']]]
q0 = [F(1, 540)] * len(rows)

def u(row):
    return 1 if ((row[2] // 3 + row[4]) % 2 == 0) else -1

qB = [[F(1, 540) * (1 + F(4, 5) * t[(r[0]+k)%5] * u(r))
       for r in rows] for k in range(5)]
qS = [[F(1, 540) * (1 + F(4, 5) * C[(r[0]+k)%5][r[1]] * u(r))
       for r in rows] for k in range(5)]

def marginal(q, indices):
    d = defaultdict(F)
    for r, w in zip(rows, q):
        d[tuple(r[i] for i in indices)] += w
    return dict(d)

indices = {
    'class': [0], 'physical_object': [0, 1],
    'class_azimuth': [0, 2], 'class_elevation': [0, 3],
    'class_lighting': [0, 4], 'object_azimuth': [0, 1, 2],
    'object_elevation': [0, 1, 3], 'object_lighting': [0, 1, 4],
    'unconditional_full_nuisance_joint': [2, 3, 4],
    'class_azimuth_elevation': [0, 2, 3],
    'class_elevation_lighting': [0, 3, 4],
}
checks = {key: all(marginal(q, ix) == marginal(q0, ix) for q in qB+qS)
          for key, ix in indices.items()}
checks.update({
    'C_row_sums_zero': all(sum(row) == 0 for row in C),
    'C_each_column_t_permutation': all(sorted(C[y][j] for y in range(5)) == sorted(t)
                                      for j in range(3)),
    'C_all_coefficient_counts_three_each': Counter(x for row in C for x in row) == Counter({x: 3 for x in t}),
    'normalization_and_positive_support': all(sum(q) == 1 and min(q) > 0 for q in qB+qS),
    'qS_class_full_nuisance_joint_equal_q0': all(marginal(q,[0,2,3,4]) == marginal(q0,[0,2,3,4]) for q in qS),
    'qB_class_azimuth_lighting_changed': all(marginal(q,[0,2,4]) != marginal(q0,[0,2,4]) for q in qB),
    'per_configuration_global_weight_histograms_equal': all(Counter(b) == Counter(s) for b,s in zip(qB,qS)),
    'each_arm_configuration_mean_q0_pointwise': all(sum(q[i] for q in arm)/5 == q0[i] for arm in [qB,qS] for i in range(540)),
    'each_object_coefficient_multiset_matched_across_configurations': all(
        sorted(t[(y+k)%5] for k in range(5)) == sorted(C[(y+k)%5][j] for k in range(5))
        for y in range(5) for j in range(3)),
    'kish_concentration_matches': all(1/sum(w*w for w in q) == F(4500,11) for q in qB+qS),
})
assert all(checks.values()), checks

report = {
    'kind': 'pre_result_exact_index_grid_review_v2',
    'training_ran': False, 'project_arrays_read': False,
    'actual_row_alignment_verified': False,
    'azimuth_partition': 'rank 0,1,2 versus rank 3,4,5',
    'lighting_partition': 'rank 0 versus rank 1',
    'C': [list(map(str,row)) for row in C],
    'checks': checks,
    'distinct_source_images': 540, 'distinct_source_objects': 15,
    'development_objects_as_reported_not_read': 10,
    'qB_qS_weight_histogram': {str(k): v for k,v in sorted(Counter(qB[0]).items())},
    'kish_image_weight_concentration': '4500/11',
    'maximum_weight_ratio': '9',
    'gate': 'conditional_go_one_development_pilot_original_contribution_not_established',
    'allowed_estimand': 'mean_k[(LLweak-LLclean)_qB - (LLweak-LLclean)_qS] on predesignated consumed development objects',
    'remaining_limits': [
        'qS induces physical-object by condition dependence; it is not an absence of all nuisance association.',
        'qB/qS per-object weight histograms differ within a configuration; they match over all five fixed configurations.',
        'Both arms average to q0 before fitting. Fit separate heads first; average risks afterwards.',
        'No conclusion of pixel shortcut use, physical do effects, or full RandAugment benefit follows from the primary interaction alone.',
        'Actual source object ordinal mapping must come from the pre-existing locked source_objects order.',
    ],
}
out = Path(__file__).with_name('matched-concentration-verification.json')
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
