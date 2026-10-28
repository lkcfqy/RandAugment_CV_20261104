"""Exact algebraic review only: no project arrays, images, models or training."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path

rows = list(product(range(5), range(3), range(6), range(3), range(2)))
t = [F(-1), F(-1, 2), F(0), F(1, 2), F(1)]
q0 = [F(1, 540)] * len(rows)

def cell(row):
    y, obj, az, el, li = row
    return 1 if ((az // 3 + li) % 2 == 0) else -1

qs = [[F(1, 540) * (1 + F(4, 5) * t[(r[0] + k) % 5] * cell(r))
       for r in rows] for k in range(5)]

def aggregate(weights, indices):
    out = defaultdict(F)
    for row, w in zip(rows, weights):
        out[tuple(row[i] for i in indices)] += w
    return dict(out)

matching = {
    'class': [0], 'physical_object': [0, 1],
    'class_azimuth': [0, 2], 'class_elevation': [0, 3],
    'class_lighting': [0, 4], 'object_azimuth': [0, 1, 2],
    'object_elevation': [0, 1, 3], 'object_lighting': [0, 1, 4],
    'unconditional_azimuth_elevation_lighting': [2, 3, 4],
    'class_azimuth_elevation': [0, 2, 3],
    'class_elevation_lighting': [0, 3, 4],
}
checks = {key: all(aggregate(q, idx) == aggregate(q0, idx) for q in qs)
          for key, idx in matching.items()}
checks['normalization'] = all(sum(q) == 1 for q in qs)
checks['full_positive_support'] = all(min(q) > 0 for q in qs)
checks['mean_configuration_equals_q0_pointwise'] = all(
    sum(q[i] for q in qs) / 5 == q0[i] for i in range(540))
checks['label_azimuth_lighting_is_changed'] = all(
    aggregate(q, [0, 2, 4]) != aggregate(q0, [0, 2, 4]) for q in qs)
assert all(checks.values()), checks

neff = [1 / sum(w*w for w in q) for q in qs]
assert len(set(neff)) == 1
assert neff[0] == F(4500, 11)
yu = defaultdict(F)
for r, w in zip(rows, qs[0]):
    yu[(r[0], cell(r))] += w
info = sum(float(p) * math.log(float(p / F(1, 10))) for p in yu.values())
metadata_accuracy = sum(max(yu[(y, u)] for y in range(5)) for u in [-1, 1])

report = {
    'kind': 'algebraic_necessary_condition_review_not_array_validation',
    'no_project_arrays_images_or_models_read': True,
    'rows_assumed': '5 classes x 3 objects x 6 azimuths x 3 elevations x 2 lights',
    'azimuth_partition': 'rank 0,1,2 versus rank 3,4,5',
    'lighting_partition': 'rank 0 versus rank 1',
    'alpha': '4/5', 't': list(map(str, t)),
    'checks': checks,
    'q_min': str(min(qs[0])), 'q_max': str(max(qs[0])),
    'q0_kish_image_weight_neff': '540',
    'qk_kish_image_weight_neff': str(neff[0]),
    'qk_kish_image_weight_neff_float': float(neff[0]),
    'relative_weight_second_moment': '33/25',
    'metadata_joint_cue_u_only_bayes_accuracy': str(metadata_accuracy),
    'metadata_joint_cue_u_only_bayes_accuracy_float': float(metadata_accuracy),
    'metadata_label_mutual_information_nats': info,
    'cautions': [
        'No change to support: all 540 source rows retain positive weight.',
        'Distinct images and total/object/marginal loss masses match; loss weights do not.',
        'Kish value is a weight concentration diagnostic, not independent object count or literal lost images.',
        'Average weighted objective equals q0 before fitting; each qk must be fitted separately.',
        'Metadata cue predictivity does not establish cue recovery from frozen image features.',
        'Correct actual cache metadata/row alignment remains to be verified by the execution code.',
    ],
}
target = Path(__file__).with_name('weight-design-verification.json')
target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(report, indent=2, ensure_ascii=False))
