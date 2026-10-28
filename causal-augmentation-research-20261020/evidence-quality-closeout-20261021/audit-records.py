"""Read-only audit of saved record chronology; no model or pixel processing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020')
STUDY = ROOT / 'area-pairing-cpu-20261021'
OUT = Path(__file__).resolve().parent
sources = {}
checks = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = json.loads(path.read_text())
    sources[str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return data


def check_recorded_hash(record, field, target, expected):
    actual = sha(target)
    checks.append({'record': str(record.relative_to(ROOT)), 'field': field,
                   'target': str(target.relative_to(ROOT)), 'recorded': expected,
                   'current': actual, 'matches': actual == expected})


def timeline(branch, filenames):
    rows = []
    for filename in filenames:
        path = STUDY / branch / filename
        data = read(path)
        content_time = data.get('at_UTC', data.get('at_utc'))
        rows.append({'file': str(path.relative_to(ROOT)), 'recorded_at': content_time,
                     'archive_mtime_UTC': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                     'bytes': path.stat().st_size, 'sha256': sha(path)})
    parsed = [datetime.fromisoformat(r['recorded_at']) for r in rows]
    return {'records': rows, 'recorded_order_strictly_increasing': all(a < b for a, b in zip(parsed, parsed[1:]))}


chronology = {
    'brightness': timeline('brightness-coupling-training-20261021', ['protocol.json', 'protocol-lock.json', 'training-freeze.json', 'heldout-fetch-receipt.json', 'final-results.json']),
    'filter': timeline('train-filter-sourceweight-20261021', ['protocol.json', 'training-freeze.json', 'test-fetch-receipt.json', 'test-results.json']),
    'equalize': timeline('equalize-pairing-cpu-20261021', ['protocol.json', 'development-freeze.json', 'heldout-results.json'])
}

b = STUDY / 'brightness-coupling-training-20261021'
lock = read(b / 'protocol-lock.json')
for name, digest in lock['hashes'].items():
    check_recorded_hash(b / 'protocol-lock.json', '/hashes/' + name, b / name, digest)
freeze = read(b / 'training-freeze.json')
for field, target in [('protocol_lock_sha256', 'protocol-lock.json'), ('training_results_sha256', 'training-results.json')]:
    check_recorded_hash(b / 'training-freeze.json', '/' + field, b / target, freeze[field])
fetch = read(b / 'heldout-fetch-receipt.json')
check_recorded_hash(b / 'heldout-fetch-receipt.json', '/training_freeze_sha256', b / 'training-freeze.json', fetch['training_freeze_sha256'])
final = read(b / 'final-results.json')
for field, target in [('protocol_sha256', 'protocol.json'), ('code_sha256', 'experiment.py'), ('training_freeze_sha256', 'training-freeze.json')]:
    check_recorded_hash(b / 'final-results.json', '/' + field, b / target, final[field])
verification = read(b / 'verification.json')
for field, target in [('final_results_sha256', 'final-results.json'), ('training_results_sha256', 'training-results.json'), ('code_sha256', 'experiment.py')]:
    check_recorded_hash(b / 'verification.json', '/' + field, b / target, verification[field])

b = STUDY / 'train-filter-sourceweight-20261021'
freeze = read(b / 'training-freeze.json')
for field, target in [('protocol_sha256', 'protocol.json'), ('training_code_sha256', 'train_probe.py'), ('fetch_code_sha256', 'fetch_fresh_test.py'), ('models_sha256', 'trained-models.json'), ('teachers_sha256', 'teacher-models.json'), ('training_weights_sha256', 'training-weights.json'), ('training_manifest_sha256', 'train-image-manifest.json')]:
    check_recorded_hash(b / 'training-freeze.json', '/' + field, b / target, freeze[field])
fetch = read(b / 'test-fetch-receipt.json')
check_recorded_hash(b / 'test-fetch-receipt.json', '/training_freeze_sha256', b / 'training-freeze.json', fetch['training_freeze_sha256'])

b = STUDY / 'equalize-pairing-cpu-20261021'
freeze = read(b / 'development-freeze.json')
for field, target in [('protocol_sha256', 'protocol.json'), ('code_sha256', 'run_equalize_audit.py'), ('development_results_sha256', 'fit-results.json')]:
    check_recorded_hash(b / 'development-freeze.json', '/' + field, b / target, freeze[field])
heldout = read(b / 'heldout-results.json')
for field, target in [('protocol_sha256', 'protocol.json'), ('code_sha256', 'run_equalize_audit.py')]:
    check_recorded_hash(b / 'heldout-results.json', '/' + field, b / target, heldout[field])

# Verify threshold consistency using previously saved aggregate fields only.
bp = read(STUDY / 'brightness-coupling-training-20261021/protocol.json')
br = read(STUDY / 'brightness-coupling-training-20261021/final-results.json')
fp = read(STUDY / 'train-filter-sourceweight-20261021/protocol.json')
fr = read(STUDY / 'train-filter-sourceweight-20261021/test-results.json')
ep = read(STUDY / 'equalize-pairing-cpu-20261021/protocol.json')
er = read(STUDY / 'equalize-pairing-cpu-20261021/heldout-results.json')
screens = {
    'brightness': {'gate': bp['candidate_screen'], 'saved': br['candidate_screen'],
                   'relative_20percent_comparison_matches': (br['relative_primary_reduction'] >= bp['candidate_screen']['min_relative_primary_reduction']) == br['candidate_screen']['relative_reduction_at_least20percent'],
                   'screen_passed': br['passes_large_effect_pilot_screen']},
    'filter': {'gate': fp['screen'], 'saved': fr['gates'],
               'minimum_gain_comparison_matches': (fr['primary_benefit_LL_matched_source_control_minus_filtered'] >= fp['screen']['primary_benefit_min_nats']) == fr['gates']['primary_gain_at_least001'],
               'screen_passed': fr['screen_passed']},
    'equalize': {'gate': ep['primary_screen'], 'saved': er['gates'],
                 'minimum_harm_comparison_matches': (er['primary_delta_LL_global_minus_cross'] >= ep['primary_screen']['delta_mean_nats_min']) == er['gates']['mean_harm_at_least_001'],
                 'screen_passed': er['screen_passed']}
}

result = {
    'at_UTC': datetime.now(timezone.utc).isoformat(),
    'scope': 'Existing small JSON/source record hashes and saved aggregate threshold checks only; no inference, image decoding, binary model/array loading, bootstrap rerun or training.',
    'chronology': chronology,
    'record_hash_checks': checks,
    'all_recorded_hash_checks_match': all(c['matches'] for c in checks),
    'all_recorded_chronologies_ordered': all(v['recorded_order_strictly_increasing'] for v in chronology.values()),
    'screens': screens,
    'evidence_limitations': [
        'Internal record timestamps and present hash chains are consistent with recorded freeze-before-fetch order, not independent external timestamp or public preregistration proof.',
        'Archive modification times are later than internal execution timestamps and must not be treated as execution order.',
        'This audit does not reconstruct unlogged analyst decisions or prove no prior exposure beyond the available records.',
        'Binary checkpoints and arrays were not independently rehashed in this round; historical verification records remain separate evidence.',
        'Practical pilot screens are not equivalence margins, power guarantees, or familywise error controls for adaptive project-wide exploration.'
    ],
    'sources': sources
}
output = OUT / 'chronology-audit.json'
with output.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'output': str(output), 'hash_checks': len(checks), 'all_hashes_match': result['all_recorded_hash_checks_match'], 'all_orders_match': result['all_recorded_chronologies_ordered'], 'threshold_comparisons_match': all(next(v[k] for k in v if k.endswith('_comparison_matches')) for v in screens.values())}, ensure_ascii=False))
