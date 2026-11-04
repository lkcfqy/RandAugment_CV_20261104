"""Independent, stdlib-only verification. Reads annotations and bytes, never pixels/code."""
from pathlib import Path
import csv, json, hashlib, collections, math, datetime, shutil

P = Path(__file__).resolve().parent
D = P / 'original_v4'
O = P / 'reports'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

manifest = json.loads((O / 'EXTRACTED_FILE_SHA256.json').read_text())
assert len(manifest) == 15193
for row in manifest:
    f = D / row['path']
    assert f.is_file() and not f.is_symlink()
    assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], row['path']
assert sum(r['bytes'] for r in manifest) == 197144979
assert sha(P / 'assets/brackish-dataset-v4.zip') == '95de9551dd3792e4c690b59ea8dfe9c4ab59dae4c7ad32d994bd6820a75630da'
videos = [r for r in manifest if r['path'].endswith('.avi')]
assert len(videos) == len({r['sha256'] for r in videos}) == 89

all_frames = list(csv.DictReader((O / 'ALL_FRAME_QUALIFICATION.csv').open()))
assert len(all_frames) == len({r['filename'] for r in all_frames}) == 15084
frames = {r['filename']: r for r in all_frames}
co = collections.defaultdict(collections.Counter)
aa = collections.defaultdict(collections.Counter)
category_counts = collections.Counter()
outside, outside_frames, bad = [], set(), []
split_sets = {}
for s in ('train', 'valid', 'test'):
    split_sets[s] = set((D / (s + '.txt')).read_text().splitlines())
    x = json.loads((D / f'annotations/annotations_COCO/{s}_groundtruth.json').read_text())
    ims = {i['id']: i for i in x['images']}
    cats = {c['name']: c['id'] for c in x['categories']}
    for a in x['annotations']:
        im = ims[a['image_id']]
        n = im['file_name']
        assert n in split_sets[s] and im['width'] == 960 and im['height'] == 540
        b = tuple(a['bbox'])
        co[n][(a['category_id'], *b)] += 1
        category_counts[a['category_id']] += 1
        xx, yy, w, h = b
        if not all(math.isfinite(float(v)) for v in (*b, a['area'])) or w <= 0 or h <= 0 or a['area'] != w*h or a.get('iscrowd', 0) or a.get('ignore', 0):
            bad.append([s, a['id']])
        if xx < 0 or yy < 0 or xx+w > 960 or yy+h > 540:
            outside.append([s, a['id']]); outside_frames.add(n)
    with (D / f'annotations/annotations_AAU/{s}.csv').open() as f:
        for a in csv.DictReader(f, delimiter=';'):
            xx, yy, rr, bb = [float(a[k]) for k in ('Upper left corner X','Upper left corner Y','Lower right corner X','Lower right corner Y')]
            aa[a['Filename']][(cats[a['Annotation tag']], xx, yy, rr-xx, bb-yy)] += 1
assert dict(co) == dict(aa)
assert sum(category_counts.values()) == 35565 and not bad
assert len(outside) == 742 and len(outside_frames) == 557
assert len(set.union(*split_sets.values())) == sum(map(len, split_sets.values())) == 14674

qualified = {r['filename'] for r in all_frames if r['eligible_frame_metadata'] == 'True'}
assert len(qualified) == 14518
assert sum(sum(co[n].values()) for n in qualified) == 34265
assert sum(sum(v for k,v in co[n].items() if k[0] in (1,2)) for n in qualified) == 12976
assert sum(not co[n] for n in qualified) == 2230
assert sum(not any(k[0] in (1,2) for k in co[n]) for n in qualified) == 8783
tails = [r for r in all_frames if r['coverage'] == 'outside_declared_annotation_coverage']
assert len(tails) == 566
assert sum(bool(r['published_splits']) for r in tails) == 156
assert sum(sum(co[r['filename']].values()) for r in tails) == 1300
assert sum(int(r['YOLO_rows']) for r in tails) == 0

blocks = {}
for label in ('SUPPORT', 'DEVELOPMENT', 'FUTURE_CONFIRMATION'):
    rows = list(csv.DictReader((O / ('PROPOSED_' + label + '_POOL_FRAMES.csv')).open()))
    names = {r['filename'] for r in rows}
    dates = {r['date'] for r in rows}
    clips = {r['video'] for r in rows}
    assert len(rows) == len(names) and names <= qualified
    for n in names:
        assert rows and frames[n]['coverage'] == 'within_declared_annotation_coverage'
    blocks[label] = (names, dates, clips)
assert set.union(*(x[0] for x in blocks.values())) == qualified
for a in blocks:
    for b in blocks:
        if a < b:
            assert all(not (blocks[a][k] & blocks[b][k]) for k in range(3))
strict = [n for n in blocks['SUPPORT'][0] if sum(co[n].values()) == 1 and next(iter(co[n]))[0] in (1,2)]
assert len(strict) == 742 and len({frames[n]['video'] for n in strict}) == 7

result = {
    'status': 'PASS', 'verified_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'original_files_verified_count': len(manifest), 'original_file_bytes': sum(r['bytes'] for r in manifest),
    'original_files_unchanged': True, 'archive_SHA256_verified': True,
    'AVI_exact_byte_duplicate_groups': 0, 'AVI_actual_frame_counts_or_pixels_not_decoded': True,
    'AAU_COCO_exact_category_bbox_multiset_agreement': True, 'COCO_annotations': 35565,
    'hard_structural_invalid_or_ignore_crowd_records': bad,
    'out_of_bounds_boxes_retained_unmodified': len(outside), 'out_of_bounds_affected_frames': len(outside_frames),
    'split_frames': 14674, 'qualified_frames': len(qualified), 'qualified_boxes': 34265,
    'qualified_fish_boxes': 12976, 'qualified_zero_any_annotation_frames': 2230,
    'qualified_zero_fish_target_frames': 8783, 'declared_coverage_tail_frames_excluded': len(tails),
    'tail_frames_in_official_split': 156, 'COCO_AAU_tail_boxes_excluded_from_proposal': 1300,
    'date_video_filename_disjoint_proposed_pools': True,
    'strict_one_total_fish_box_support_candidates': len(strict),
    'strict_one_total_fish_box_support_candidate_videos': 7,
    'model_calls': 0, 'training_steps': 0, 'video_pixel_decodes': 0,
    'upstream_code_executed': False, 'installs': 0, 'uploads_bytes': 0, 'new_GPU_MPS_Colab_CU': 0,
    'free_disk_bytes': shutil.disk_usage(P).free,
    'note': 'Coverage/empty status is publisher annotation metadata, not independent visual completeness proof.'
}
(O / 'VERIFICATION.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
