"""Prepare a metadata-only stratified audit sample; no image or model access."""
import csv
import io
import json
from pathlib import Path
import random


def prepare(root, seed=20261021, per_stratum=50):
    ratios = json.loads((root/'metadata/fg_bg_ratios_val.json').read_text())
    indices = json.loads((root/'metadata/hf_val_indices.json').read_text())
    strata = {k: [] for k in ['eligible_far_lt_0.6', 'eligible_near_0.6_to_lt_0.8',
                              'excluded_near_0.8_to_lt_0.9', 'excluded_far_ge_0.9']}
    for key, ratio in sorted(ratios.items()):
        group = ('eligible_far_lt_0.6' if ratio < .6 else
                 'eligible_near_0.6_to_lt_0.8' if ratio < .8 else
                 'excluded_near_0.8_to_lt_0.9' if ratio < .9 else
                 'excluded_far_ge_0.9')
        strata[group].append((key, ratio))
    if sum(map(len, strata.values())) != len(ratios):
        raise ValueError('Strata do not partition ratio-key population')
    rng = random.Random(seed)
    selected = []
    designs = []
    for name, pool in strata.items():
        n = min(per_stratum, len(pool))
        designs.append({'stratum': name, 'population': len(pool), 'sample': n,
                        'inclusion_probability': n/len(pool) if pool else None})
        for key, ratio in rng.sample(pool, n):
            p = Path(key)
            original = p.parent.name+'/'+p.stem
            if original not in indices:
                raise ValueError('Unmatched source index')
            selected.append({'source_key': key, 'source_image_key': original,
                             'source_imagenet_class': p.parent.name,
                             'source_hf_validation_index': indices[original],
                             'ratio': ratio, 'eligible_strict_lt_0_8': int(ratio < .8),
                             'stratum': name, 'inclusion_probability': n/len(pool),
                             'image_availability': 'unverified'})
    # Randomize review order; private design table is separate from annotation form.
    rng.shuffle(selected)
    for i, row in enumerate(selected, 1):
        row['audit_id'] = 'BGVAL_'+str(i).zfill(4)
    if len({r['source_key'] for r in selected}) != len(selected):
        raise ValueError('Duplicate sample key')
    return selected, {
        'seed': seed, 'population': '49751 validation ratio keys after earlier mask selection',
        'population_size': len(ratios), 'sample_size': len(selected), 'strata': designs,
        'labels': 'No human review or image verification executed',
        'scope': 'Background reconstruction validity audit, not augmented-image false rejection',
        'weighting': 'Stratified without-replacement; use inverse inclusion probabilities for descriptive population ratios after human review; uncertainty requires design-aware treatment',
        'rights': 'ImageNet/ForNet image access must be resolved before obtaining/displaying images',
    }


def csv_bytes(rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


if __name__ == '__main__':
    root = Path(__file__).parent
    selected, design = prepare(root)
    key_fields = ['audit_id'] + [k for k in selected[0] if k != 'audit_id']
    forms = [dict(audit_id=r['audit_id'], human_background_validity='', human_scene='',
                  uncertainty_reason='', reviewer_id='', original_and_repaired_images_seen='')
             for r in selected]
    outputs = {'background-audit-sample-key.csv': csv_bytes(selected, key_fields),
               'background-audit-annotation-template.csv': csv_bytes(forms, list(forms[0])),
               'background-audit-design.json': (json.dumps(design, ensure_ascii=False, indent=2)+'\n').encode()}
    for name in outputs:
        if (root/name).exists():
            raise SystemExit('Preserving existing '+name)
    for name, payload in outputs.items():
        with (root/name).open('xb') as f:
            f.write(payload)
    print(json.dumps(design, ensure_ascii=False))
