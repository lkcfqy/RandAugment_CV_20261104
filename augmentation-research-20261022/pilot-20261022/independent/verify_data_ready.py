"""Read-only source/copy audit. Standard library only; no model or image decode."""
import ast
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import zlib

WORK = Path('/Users/lkc/Documents/Codex/2026-10-22/task-2/lora-ra-pilot-20261022')
PROJECT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/pilot-20261022')
PLAN = PROJECT.parent / 'candidate-data-plan.json'
OLD = PROJECT.parent.parent / 'causal-augmentation-research-20261020'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def main():
    checks = []

    def check(name, ok):
        checks.append({'check': name, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(name)

    manifest = read(WORK / 'fit-manifest.json')
    plan = read(PLAN)
    qa = read(WORK / 'qa/source-qa.json')
    decisions = read(WORK / 'qa/duplicate-adjudication.json')
    visual = read(WORK / 'qa/visual-review.json')
    geometry = read(WORK / 'qa/geometry-visibility.json')
    cfg = read(WORK / 'protocol.json')
    lock = read(WORK / 'protocol-lock.json')
    pre = read(WORK / 'preflight.json')
    copy = read(PROJECT / 'copy-receipt.json')
    rows, excluded = manifest['rows'], manifest['excluded_rows']
    candidate = {r['source_id']: r for r in plan['rows']}
    source = {r['source_id']: r for r in qa['rows']}
    reserved = {k for k, v in candidate.items() if v['candidate_partition'] == 'reserved_confirmation'}
    old_ids = {p.stem for p in OLD.rglob('*.jpg')}
    known_excluded = {'Bombay_132', 'Bombay_107', 'keeshond_103', 'keeshond_167'}
    check('old_ID_count177_filenames_only', len(old_ids) == 177)
    check('candidate_IDs_unique1776', len(candidate) == len(plan['rows']) == 1776)
    check('QA_rows_exactly_train_and_development_candidates', set(source) == set(candidate) - reserved)
    check('all_candidate_IDs_disjoint_old177', not set(candidate) & old_ids)
    check('final_kept1180_unique', len(rows) == len({r['source_id'] for r in rows}) == 1180)
    check('four_whole_duplicate_members_excluded_no_replacement',
          {r['source_id'] for r in excluded} == known_excluded and len(excluded) == 4
          and manifest['replacement_samples'] == 0)
    check('final_order_is_QA_order_minus_four', rows == [dict(r, duplicate_cluster_id='unique-source:' + r['source_id'])
          for r in qa['rows'] if r['source_id'] not in known_excluded])
    counts = collections.Counter((r['candidate_partition'], r['class_id']) for r in rows)
    partition_counts = collections.Counter(r['candidate_partition'] for r in rows)
    check('final_partition_counts591_589', dict(partition_counts) == {'train': 591, 'development': 589})
    check('all37_classes_retained_each_partition',
          set(counts) == {(p, c) for p in ['train', 'development'] for c in range(37)})
    check('post_exclusion_imbalanced_counts_expected', counts[('train', 18)] == 15
          and counts[('development', 7)] == 14 and counts[('development', 18)] == 15
          and all(v == 16 for k, v in counts.items()
                  if k not in {('train', 18), ('development', 7), ('development', 18)}))
    for r in rows:
        c = candidate[r['source_id']]
        check_fields = ['candidate_partition', 'class_id', 'species', 'image_entry', 'mask_entry']
        assert all(r[k] == c[k] for k in check_fields)
        assert r['source_id'] not in reserved and r['source_id'] not in old_ids
    check('kept_metadata_matches_source_candidates', True)
    check('all_kept_duplicate_clusters_unique', len({r['duplicate_cluster_id'] for r in rows}) == 1180)
    expected_pairs = {tuple(sorted((p['a'], p['b']))) for p in qa['dhash_distance4_candidate_pairs']}
    actual_pairs = {tuple(sorted((p['a'], p['b']))) for p in decisions['pairs']}
    check('all_duplicate_candidates_adjudicated_same_capture', expected_pairs == actual_pairs
          and len(actual_pairs) == 2 and all(p['decision'] == 'same_capture' for p in decisions['pairs']))
    check('no_other_QA_issues_or_exact_groups', qa['issues'] == [] and qa['exact_decoded_duplicate_groups'] == [])
    check('manifest_source_hashes_match', manifest['source_QA_sha256'] == sha(WORK / 'qa/source-qa.json')
          and manifest['duplicate_decisions_sha256'] == sha(WORK / 'qa/duplicate-adjudication.json')
          and manifest['candidate_plan_sha256'] == sha(PLAN))
    expected_ids = set(source)
    check('only1184_selected_members_stored',
          {p.stem for p in (WORK / 'data/images').glob('*.jpg')} == expected_ids
          and {p.stem for p in (WORK / 'data/masks').glob('*.png')} == expected_ids)
    payload_count = payload_bytes = 0
    for sid, r in source.items():
        for folder, suffix, key, digest_key in [('images', '.jpg', 'image_zip_metadata', 'image_sha256'),
                                               ('masks', '.png', 'mask_zip_metadata', 'mask_sha256')]:
            b = (WORK / 'data' / folder / (sid + suffix)).read_bytes()
            info = candidate[sid][key]
            assert len(b) == info['bytes'], (sid, key, 'size')
            assert zlib.crc32(b) == info['crc32'], (sid, key, 'crc')
            assert hashlib.sha256(b).hexdigest() == r[digest_key], (sid, key, 'sha')
            payload_count += 1
            payload_bytes += len(b)
    check('all2368_stored_source_payloads_size_CRC_SHA_match', payload_count == 2368)
    check('visual_reviews_and_links_closed', visual['status'] == 'completed_before_task_fit'
          and not visual['breed_semantics_or_all_epoch_preservation_certified']
          and all(sha(WORK / v['path']) == v['sha256'] for v in visual['review_records'].values()))
    check('empty_mask_failure_preserved', (WORK / 'qa-attempt01/failure.json').is_file()
          and (WORK / 'qa-before-empty-mask-diagnostic.py').is_file())
    unavailable = {r['source_id'] for r in geometry['rows'] if not r['foreground_proxy_available']}
    check('geometry_unavailable_proxy_correctly_identified', unavailable == {'saint_bernard_15'}
          and all(r['weak_original_foreground_retained_fraction'] is None
                  and r['RA_to_weak_foreground_pixel_count_ratio'] is None
                  for r in geometry['rows'] if r['source_id'] in unavailable))
    check('all_three_empty_masks_retained_as_classification_sources',
          {'miniature_pinscher_14', 'saint_bernard_108', 'saint_bernard_15'} <= {r['source_id'] for r in rows})
    license_receipt = read(WORK / 'data/license-receipt.json')
    html = (WORK / 'data/official-pets-page.html').read_bytes()
    check('official_license_evidence_hash_and_link', hashlib.sha256(html).hexdigest() == license_receipt['sha256']
          and b'https://creativecommons.org/licenses/by-sa/4.0/' in html)
    check('resource_actual_two_layouts_succeeded', pre['status'] == 'completed_resource_only_calibration'
          and {c['layout'] for c in pre['cases']} == {'front', 'back'} and len(pre['cases']) == 2
          and all(c['device'] == 'mps:0' and c['trainable_parameters'] == 65317 for c in pre['cases'])
          and pre['augmentation_restores_CPU_and_preserves_MPS_RNG'] and not pre['MPS_fallback']
          and pre['CPU_threads'] == 1 and pre['actual_process_nice'] == 10 and pre['MPS_memory_fraction'] == .1)
    estimate = sum(sum(c['step_seconds'][1:]) / 4 for c in pre['cases']) * 2 * math.ceil(591 / 16) * 30 * 3
    check('resource_estimate_independently_recomputed', abs(estimate - pre['estimated_30ep_four_cell_seconds_compute_plus_pacing']) < 1e-9)
    check('resource_rule_selects30_epochs_without_task_outcomes', estimate * 1.25 <= 7200 and cfg['epochs'] == 30
          and cfg['resource_decision']['estimated_seconds_with_25percent_overhead'] == estimate * 1.25)
    check('formal_protocol_hash_matches_report_and_lock', sha(WORK / 'protocol.json') == lock['protocol_sha256']
          == '9386fc378f876cf12c68ef5d7248f073d7313433d158350b4143942ac1b52a3f')
    check('protocol_data_and_resource_hashes_match', cfg['fit_manifest_sha256'] == sha(WORK / 'fit-manifest.json')
          == pre['fit_manifest_sha256'] and cfg['data']['train_photos'] == 591 and cfg['data']['development_photos'] == 589)
    check('locked_sources_and_evidence_SHA_match',
          all(sha(WORK / n) == h for n, h in lock['code_sha256'].items())
          and all(sha(WORK / n) == h for n, h in cfg['evidence_sha256'].items()))
    check('actual_pretrained_weight_hash_matches_protocol', sha(WORK / 'data/vit_b_16-c867db91.pth') == cfg['weight_sha256'])
    for entry in copy['files']:
        rel = Path(entry['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        for base in [WORK, PROJECT]:
            p = base / rel
            assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], (str(base), str(rel))
    check('all2412_attested_prefit_copies_independently_hashed', len(copy['files']) == copy['file_count'] == 2412
          and sum(f['bytes'] for f in copy['files']) == copy['bytes'] == 483359685)
    started = read(PROJECT / 'results/started.json')
    timeline = {'manifest': manifest['at_utc'], 'visual': visual['at_utc'], 'protocol': cfg['frozen_at_utc'],
                'lock': lock['at_utc'], 'copy': copy['at_utc'], 'training_start_metadata_only': started['at_utc']}
    parse = datetime.datetime.fromisoformat
    check('QA_manifest_then_protocol_lock_copy_then_fit',
          parse(timeline['visual']) < parse(timeline['manifest']) < parse(timeline['protocol'])
          < parse(timeline['lock']) < parse(timeline['copy']) < parse(timeline['training_start_metadata_only']))
    check('started_receipt_has_same_protocol_and_nice_cpu_settings', started['protocol_sha256'] == lock['protocol_sha256']
          and started['actual_process_nice'] == 10 and started['CPU_threads'] == 1 and not started['MPS_fallback'])
    for n in ['common.py', 'train.py', 'preflight.py', 'qa.py', 'make_fit_manifest.py', 'prepare_protocol.py', 'analyze.py']:
        ast.parse((WORK / n).read_bytes())
    result = {'status': 'passed_data_QA_resource_lock_and_copy_audit',
              'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'checks': checks,
              'fitting_root': str(PROJECT), 'fit_manifest_sha256': sha(WORK / 'fit-manifest.json'),
              'protocol_sha256': lock['protocol_sha256'], 'protocol_lock_sha256': sha(WORK / 'protocol-lock.json'),
              'copy_receipt_sha256': sha(PROJECT / 'copy-receipt.json'),
              'counts': {'train': 591, 'development': 589, 'reserved_candidates_not_opened': 592,
                         'excluded_whole_duplicate_members': 4, 'replacement': 0,
                         'payload_members_verified': payload_count, 'payload_bytes_verified': payload_bytes,
                         'copied_files_verified': 2412, 'copied_bytes_verified': 483359685,
                         'retained_training_geometry_proxies_available': 590,
                         'retained_training_geometry_proxy_unavailable': 1},
              'resource': {'status': pre['status'], 'steady_step_seconds_by_layout': {c['layout']: c['steady_mean_seconds'] for c in pre['cases']},
                           'max_observed_MPS_driver_bytes': max(c['max_observed_driver_MPS_bytes'] for c in pre['cases']),
                           'estimated_30epoch_compute_plus_pacing_seconds': estimate,
                           'with_predeclared_25percent_margin_seconds': estimate * 1.25,
                           'actual_complete_training_walltime_guaranteed': False},
              'timeline_from_saved_records': timeline,
              'time_verification_limit': 'Saved internal timestamps and SHA/copy consistency; not independent cryptographic timestamp attestation.',
              'minimal_source_changes_accepted': {
                  'train.py': 'Only actual_process_nice/Pillow/CUDA_available receipt fields added after the static closure; locked source re-read and hashes checked.',
                  'preflight.py': 'Use final validated train rows591 and actual ceil(N/16) estimate; preserve dummy labels, two layouts, five paced updates, cap and RNG assertions; add actual nice.',
                  'make_fit_manifest.py': 'Pre-result exact/near decision coverage, connected clusters, remove all confirmed/uncertain members, preserve source order, no replacement; actual manifest independently reconstructed.',
                  'qa.py': 'Empty-source-mask diagnostic changes ratios to unavailable proxy without deleting classification source; failure and original code preserved.'},
              'remaining_limits': ['Finite fixed source-photo pool; unknown animal identity and ImageNet pretraining overlap.',
                                   '37 predetermined visual examples, only36 retained after duplicate deletion; not expert breed certification or all-epoch preservation.',
                                   'Three all-background trimaps remain; no mask supervision and no mask-based classification deletion.',
                                   'Header-margin network buffers may include adjacent bytes; no reserved member extraction/storage/decoding/scoring is claimed.',
                                   'Single common LR/seed development pilot; no cross-seed or general mechanism inference.'],
              'permission_events_preserved_not_retried': ['Prior denied global py_compile cache write', 'Prior denied nice mutation for first CPU QA PID'],
              'current_review_is_permission_blocked': False,
              'scientific_metrics_or_training_curves_inspected': False,
              'new_GPU_or_training_or_download_started_by_reviewer': False,
              'reserved_confirmation_pixels_or_old_test_arrays_read': False,
              'resource_pending': False,
              'scientific_completion_and_result_verification_pending': True}
    out = WORK / 'independent/data-ready-review.json'
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'sha256': sha(out), 'file': str(out)}))


if __name__ == '__main__':
    main()
