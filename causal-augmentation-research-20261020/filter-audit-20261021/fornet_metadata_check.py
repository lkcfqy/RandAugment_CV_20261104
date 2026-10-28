"""Recompute eligibility from pinned public JSON only; never loads images/models."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def compute(root):
    ratios = json.loads((root / 'metadata/fg_bg_ratios_val.json').read_text())
    indices = json.loads((root / 'metadata/hf_val_indices.json').read_text())
    totals = Counter(k.split('/')[0] for k in indices)
    by_class = defaultdict(list)
    ids = set()
    unmatched = []
    for key, value in ratios.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid ratio: ' + key)
        p = Path(key)
        source = p.parent.name + '/' + p.stem
        if source in ids:
            raise ValueError('Duplicate source image: ' + source)
        ids.add(source)
        if source not in indices:
            unmatched.append(key)
        by_class[p.parent.name].append(value)
    classes = []
    for cls, total in sorted(totals.items()):
        values = by_class[cls]
        kept = sum(v < 0.8 for v in values)
        classes.append(dict(source_imagenet_class=cls, source_index_rows=total,
                            ratio_rows=len(values), missing_ratio=total-len(values),
                            metadata_eligible_keys=kept,
                            metadata_excluded_keys=len(values)-kept,
                            eligibility_fraction=kept/len(values) if values else None))
    excluded = sum(v >= 0.8 for v in ratios.values())
    summary = dict(
        scope='Validation ratio-key eligibility after earlier mask selection; archive/image usability unverified',
        source_revision='dd6b027a1a01b00fea442276668a54a697eb838b',
        source_index_rows=len(indices), source_index_unique_values=len(set(indices.values())),
        ratio_rows=len(ratios), ratio_unique_source_ids=len(ids),
        ratio_keys_not_in_source_index=unmatched,
        source_rows_missing_ratio=len(set(indices)-ids),
        source_class_count=len(classes), source_index_counts_per_class=dict(Counter(totals.values())),
        threshold=0.8, comparison='ratio < 0.8',
        metadata_eligible_background_keys=len(ratios)-excluded,
        metadata_excluded_background_keys=excluded,
        excluded_fraction=excluded/len(ratios), exact_threshold_rows=sum(v == 0.8 for v in ratios.values()),
        zero_eligible_classes=[x['source_imagenet_class'] for x in classes if x['metadata_eligible_keys'] == 0],
        ten_lowest_eligibility_classes=sorted(classes, key=lambda x: x['eligibility_fraction'] if x['eligibility_fraction'] is not None else -1)[:10],
        interpretation_limits=[
            'ImageNet source class is not actual background scene truth',
            'Ratio keys do not establish actual archive files, masks or semantic validity',
            'Missing ratio entries cannot individually be attributed to segmentation failure without logs',
            'No false rejection, causal effect or downstream performance estimated',
            'Same-class loader fallback can reintroduce a background if a class has zero eligible backgrounds; none found in these metadata',
        ])
    return summary, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    summary, classes = compute(args.root)
    print(json.dumps({'summary': summary, 'per_source_class': classes}, ensure_ascii=False))
