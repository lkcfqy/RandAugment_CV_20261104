"""Read-only, standard-library verification. No model/tensor deserialization.

This script refuses to read predictions until result.json confirms all arms ended.
The only output file is explicitly within the review workspace.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import zipfile

RUN = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/posterize-three-arm-mps-20261024')
REVIEW = Path('/Users/lkc/Documents/Codex/2026-10-24/task-9/mps_integrity_review')
ROOT = RUN.parents[1]
ARMS = ('B', 'Z', 'R')
hashes = {}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    v = h.hexdigest()
    hashes[str(path)] = v
    return v


def close(a, b, tol=1e-11):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a-b) <= tol * max(1.0, abs(a), abs(b)), (a, b)


if not (RUN/'result.json').exists():
    raise SystemExit('PENDING: all-arm result.json absent; predictions not read.')
result = read_json(RUN/'result.json')
assert result['status'] == 'all_three_arms_completed'
assert set(result['summaries']) == set(ARMS)
assert all((RUN/a/'summary.json').exists() for a in ARMS)
if not (RUN/'attempt-002/process-exit.json').exists():
    raise SystemExit('PENDING: process exit receipt absent; predictions not read.')

cfg = read_json(RUN/'protocol.json')
budget = read_json(RUN/'budget.json')
assert digest(RUN/'protocol.json') == read_json(RUN/'protocol-lock.json')['sha256']
for name, expected in read_json(RUN/'code-lock-v2.json')['sha256'].items():
    assert digest(RUN/name) == expected, name
for name, expected in cfg['input_sha256'].items():
    assert digest(ROOT/name) == expected, name
assert digest(RUN.parent/'pilot-20261022/data/vit_b_16-c867db91.pth') == cfg['base_checkpoint_sha256']
launch = read_json(RUN/'attempt-002/launch.json')
assert launch['worker_sha256'] == hashes[str(RUN/'worker-v2.py')]
assert launch['phase_start_utc'] == budget['start_utc'] and launch['deadline_utc'] == budget['deadline_utc']
assert budget['start_utc'] == '2026-10-24T12:43:56.683066+00:00'
assert budget['deadline_utc'] == '2026-10-24T14:13:56.683066+00:00'
assert budget['deadline_epoch'] - budget['start_epoch'] == 5400
assert budget['training_stop_epoch'] == budget['deadline_epoch'] - 600
assert budget['readiness_deadline_epoch'] == budget['start_epoch'] + 900

manifest = read_json(RUN.parent/'pilot-20261022/fit-manifest.json')
train = [r for r in manifest['rows'] if r['candidate_partition'] == 'train']
dev = [r for r in manifest['rows'] if r['candidate_partition'] == 'development']
assert len(manifest['rows']) == len({r['source_id'] for r in manifest['rows']}) == 1180
assert len(train) == 591 and len(dev) == 589
assert len({r['duplicate_cluster_id'] for r in manifest['rows']}) == 1180
assert {r['class_id'] for r in train} == {r['class_id'] for r in dev} == set(range(37))
truth = {r['source_id']: r['class_id'] for r in dev}
train_ids = {r['source_id'] for r in train}
assert not train_ids & set(truth)

orders = read_json(RUN/'orders.json')
trace = read_jsonl(RUN/'trajectories.jsonl')
preflight = read_json(RUN/'preflight.json')
trace_hash = digest(RUN/'trajectories.jsonl')
assert trace_hash == preflight['trace_sha256']
random_evidence = read_json(REVIEW/'random-stream-evidence.json')
assert random_evidence['status'] == 'all_random_streams_recomputed_exactly'
assert random_evidence['trace_sha256'] == trace_hash
assert digest(RUN/'orders.json') == preflight['orders_sha256']
assert len(orders) == 30 and all(len(o) == 591 and set(o) == train_ids for o in orders)
assert len(trace) == 17730
assert [(t['epoch'], t['source_id']) for t in trace] == [(e, sid) for e, order in enumerate(orders) for sid in order]
op_counts = {a: Counter() for a in ARMS}
target_slots = target_views = 0
for t in trace:
    assert len(t['B']) == len(t['Z']) == len(t['R']) == 2
    n = 0
    for b, z, r in zip(t['B'], t['Z'], t['R']):
        if b['name'] == 'Posterize':
            n += 1
            assert z == {'name': 'Identity', 'magnitude': 0.0}
            assert r['name'] in cfg['pool'] and r['name'] != 'Posterize'
        else:
            assert b == z == r
    target_slots += n
    target_views += bool(n)
    for a in ARMS:
        op_counts[a].update(v['name'] for v in t[a])
assert target_slots == preflight['actual_target_slots']
assert target_views == preflight['actual_target_views']
assert {a: dict(c) for a, c in op_counts.items()} == preflight['op_counts']

readiness = read_json(RUN/'readiness.json')
start = read_json(RUN/'formal-training-start.json')
assert readiness['status'] == 'GO' and readiness['updates'] == 9
assert readiness['train_presentations'] == sum(r['batch'] for r in readiness['rows']) == 141
assert readiness['phase_elapsed'] < 900 and start['phase_elapsed'] < 900
assert start['protocol_sha256'] == hashes[str(RUN/'protocol.json')]
assert start['readiness_sha256'] == digest(RUN/'readiness.json')
estimate = max(r['seconds'] for r in readiness['rows'][3:]) * 3330 * 1.35 + sum(r['seconds'] for r in readiness['rows'][:3]) + 180
close(estimate, readiness['predicted_training_and_overhead_seconds'])
assert estimate < readiness['available_before_600s_reserve']

initial_hashes = set()
frozen_hashes = set()
metrics = {}
predictions = {}
checkpoint_receipts = []
epoch_receipts = []
for arm in ARMS:
    d = RUN/arm
    init = read_json(d/'initialization.json')
    summary = read_json(d/'summary.json')
    assert summary == result['summaries'][arm]
    assert summary['arm'] == arm and summary['seed'] == 20261024
    assert summary['epochs'] == 30 and summary['updates'] == 1110
    assert summary['training_presentations'] == 17730 and summary['dev_count'] == 589
    assert init['trace_sha256'] == summary['trace_sha256'] == trace_hash
    assert init['initial_trainable_sha256'] == summary['initial_trainable_sha256'] == readiness['initial_trainable_sha256']
    assert init['frozen_sha256'] == summary['frozen_before_sha256'] == summary['frozen_after_sha256'] == readiness['frozen_sha256']
    assert init['empty_optimizer'] is True and init['trainable'] == 65317
    initial_hashes.add(init['initial_trainable_sha256'])
    frozen_hashes.add(init['frozen_sha256'])
    steps = read_jsonl(d/'steps.jsonl')
    assert len(steps) == 1110 and [s['step'] for s in steps] == list(range(1, 1111))
    assert sum(s['examples'] for s in steps) == 17730
    for j, s in enumerate(steps):
        epoch, pos = divmod(j, 37)
        assert s['arm'] == arm and s['epoch'] == epoch + 1
        assert s['examples'] == (15 if pos == 36 else 16)
        assert math.isfinite(s['mean_CE']) and s['mean_CE'] >= 0
        factor = (j+1)/111 if j < 111 else 0.5*(1+math.cos(math.pi*(j-111)/999))
        close(s['adapter_lr'], 0.0003 * factor)
        close(s['head_lr'], 0.001 * factor)
        assert 0 <= s['stage_elapsed'] < 4800
    for e, order in enumerate(orders, 1):
        ep_path = d/('epoch%02d.json' % e)
        ep = read_json(ep_path)
        assert ep['epoch'] == e and ep['updates'] == e * 37 and ep['source_presentations'] == 591
        assert ep['order_sha256'] == hashlib.sha256('\n'.join(order).encode()).hexdigest()
        close(ep['training_mean_CE'], sum(s['mean_CE']*s['examples'] for s in steps[(e-1)*37:e*37])/591)
        cp = ep['checkpoint']
        if e in (5, 10, 20, 30):
            path = ROOT/cp['path']
            assert path == d/('checkpoint-epoch%02d.pt' % e)
            assert digest(path) == cp['sha256']
            with zipfile.ZipFile(path) as zf:
                assert zf.testzip() is None
                entries = zf.namelist()
                assert any(n.endswith('/data.pkl') for n in entries)
                assert any('/data/' in n for n in entries)
            checkpoint_receipts.append({'arm': arm, 'epoch': e, 'path': str(path), 'sha256': cp['sha256'], 'bytes': path.stat().st_size, 'zip_crc_valid': True, 'tensors_deserialized': False})
        else:
            assert cp is None
        epoch_receipts.append({'arm': arm, 'epoch': e, 'updates': ep['updates'], 'order_sha256': ep['order_sha256'], 'worker_rendered_RGB_sha256': ep['ordered_rendered_RGB_sha256']})
        digest(ep_path)
    assert len(list(d.glob('epoch[0-9][0-9].json'))) == 30
    assert len(list(d.glob('checkpoint-*.pt'))) == 4
    lines = read_jsonl(d/'development-predictions.jsonl')
    assert len(lines) == len({r['source_id'] for r in lines}) == 589
    assert [r['source_id'] for r in lines] == [r['source_id'] for r in dev]
    computed = {}
    nlls = []
    for r in lines:
        sid = r['source_id']
        assert r['class_id'] == truth[sid]
        logits = r['logits']
        assert len(logits) == 37 and all(math.isfinite(v) for v in logits)
        pred = max(range(37), key=logits.__getitem__)
        correct = pred == truth[sid]
        assert pred == r['prediction'] and type(r['correct']) is bool and correct == r['correct']
        m = max(logits)
        nll = m + math.log(math.fsum(math.exp(v-m) for v in logits)) - logits[truth[sid]]
        close(nll, r['NLL'])
        nlls.append(nll)
        computed[sid] = correct
    count = sum(computed.values())
    mean_nll = math.fsum(nlls)/589
    assert count == summary['dev_correct']
    close(count/589, summary['dev_accuracy'])
    close(mean_nll, summary['dev_NLL'])
    assert digest(d/'development-predictions.jsonl') == summary['prediction_sha256']
    digest(d/'summary.json')
    digest(d/'steps.jsonl')
    digest(d/'initialization.json')
    predictions[arm] = computed
    metrics[arm] = {'correct': count, 'n': 589, 'accuracy': count/589, 'mean_NLL_from_logits': mean_nll}
assert len(initial_hashes) == len(frozen_hashes) == 1

patterns = {format(i, '03b'): 0 for i in range(8)}
joint = []
for sid, label in truth.items():
    p = ''.join(str(int(predictions[a][sid])) for a in ARMS)
    patterns[p] += 1
    joint.append({'source_id': sid, 'class_id': label, **{a: predictions[a][sid] for a in ARMS}, 'pattern_BZR': p})
assert patterns == result['patterns_BZR']
assert joint == read_json(RUN/'joint-correctness.json')
paired = {}
for alt, base in [('Z', 'B'), ('R', 'B'), ('R', 'Z')]:
    corrected = sum(predictions[alt][s] and not predictions[base][s] for s in truth)
    harmed = sum(predictions[base][s] and not predictions[alt][s] for s in truth)
    delta = metrics[alt]['correct'] - metrics[base]['correct']
    assert corrected - harmed == delta
    paired[alt+'-'+base] = {'corrected': corrected, 'new_errors': harmed, 'delta_correct': delta, 'delta_percentage_points': 100*delta/589}
dz = paired['Z-B']['delta_correct']
dr = paired['R-B']['delta_correct']
if abs(dz) >= 3 and abs(dr) >= 3:
    classification = 'observed_reversal' if dz*dr < 0 else 'same_direction'
else:
    classification = 'unresolved_below_direction_threshold'
assert classification == result['classification']
assert (dz, dr, dr-dz) == (result['gZ_count'], result['gR_count'], result['h_count'])
for key, count in [('gZ', dz), ('gR', dr), ('h', dr-dz)]:
    close(result[key], count/589)
assert result['contrast_gate_count'] == 3

expected_counts = {'training_forward_images': 53190, 'dev_forward_images': 1767, 'training_forward_calls': 3330, 'dev_forward_calls': 111, 'backward_calls': 3330, 'optimizer_updates': 3330, 'train_image_reads': 53331, 'dev_image_reads': 1767, 'parity_image_reads': 48}
assert result['counts'] == expected_counts
events = read_jsonl(RUN/'attempt-002/events.jsonl')
event_counts = Counter(e['event'] for e in events)
for name, expected in {'source_locks_verified': 1, 'environment': 1, 'preflight_passed': 1, 'arm_initialized': 3, 'micro_start': 3330, 'backward_start': 3330, 'micro_complete': 3330, 'update_complete': 3330, 'epoch_complete': 90, 'dev_forward_start': 111, 'dev_batch_complete': 111, 'arm_complete': 3, 'complete': 1}.items():
    assert event_counts[name] == expected, (name, event_counts[name])
assert 'stopped' not in event_counts
assert events[-1]['event'] == 'complete' and events[-1]['counts'] == expected_counts
assert events[0]['event'] == 'source_locks_verified' and all(v == 0 for v in events[0]['counts'].values())
environment = [e for e in events if e['event'] == 'environment'][0]
assert environment['device'] == 'mps' and environment['fallback'] is False
for arm in ARMS:
    arm_events = [e for e in events if e['progress']['arm'] == arm]
    names = Counter(e['event'] for e in arm_events)
    assert names['arm_initialized'] == names['arm_complete'] == 1
    assert names['update_complete'] == names['backward_start'] == 1110
    assert names['epoch_complete'] == 30 and names['dev_batch_complete'] == 37
    first_dev = next(i for i, e in enumerate(arm_events) if e['event'] == 'dev_forward_start')
    assert sum(e['event'] == 'update_complete' for e in arm_events[:first_dev]) == 1110
assert all(0 <= e['stage_elapsed'] < 5400 for e in events)
assert all(a['stage_elapsed'] <= b['stage_elapsed'] for a, b in zip(events, events[1:]))
exit_receipt = read_json(RUN/'attempt-002/process-exit.json')
assert exit_receipt['returncode'] == 0
assert exit_receipt['deadline_term_sent'] is False and exit_receipt['deadline_kill_sent'] is False
assert 0 <= result['stage_elapsed'] <= exit_receipt['phase_elapsed'] < 5400
assert datetime.fromisoformat(exit_receipt['at_utc']).timestamp() < budget['deadline_epoch']
first_failure = read_json(RUN/'attempt-001/failure.json')
assert all(v == 0 for v in first_failure['counts'].values())
assert first_failure['checkpoint'] is None and first_failure['completed_arm_summaries'] == {}
assert first_failure['compute_wall_seconds'] == 0

for path in [RUN/'result.json', RUN/'joint-correctness.json', RUN/'preflight.json', RUN/'formal-training-start.json', RUN/'clock-incident.json', RUN/'attempt-001/failure.json', RUN/'attempt-001/process-exit.json', RUN/'attempt-002/events.jsonl', RUN/'attempt-002/launch.json', RUN/'attempt-002/process-exit.json']:
    digest(path)
evidence = {
    'verified_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'all_deterministic_checks_passed',
    'run': str(RUN),
    'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'classification': classification,
    'metrics_from_saved_logits': metrics,
    'paired_changes': paired,
    'patterns_BZR': patterns,
    'formal_counts': expected_counts,
    'benchmark_updates': 9,
    'benchmark_forward_images': 141,
    'total_optimizer_updates_including_benchmark': 3339,
    'total_backward_calls_including_benchmark': 3339,
    'total_model_forward_presentations_including_benchmark': 55098,
    'parity_image_reads_without_model_forward': 48,
    'total_image_reads_including_benchmark_and_parity': 55146,
    'stage_elapsed_seconds': result['stage_elapsed'],
    'exit_receipt': exit_receipt,
    'trace': {'sha256': trace_hash, 'rows': len(trace), 'target_slots': target_slots, 'target_views': target_views, 'op_counts': {a: dict(c) for a, c in op_counts.items()}},
    'independent_random_stream_recomputation': random_evidence,
    'checkpoint_checks': checkpoint_receipts,
    'epoch_receipts': epoch_receipts,
    'event_counts': dict(event_counts),
    'audited_input_hashes': hashes,
    'limits': ['No raw images read or rendered.', 'No model inference/training performed.', 'No tensor deserialization; checkpoint verification is bytes/hash/ZIP CRC only.', 'Parameter and rendered-pixel digests are worker-produced receipts, not independently recomputed.', 'One development seed; no inferential or generalization guarantee.'],
}
(REVIEW/'verification-evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
print(json.dumps({k: evidence[k] for k in ('status', 'classification', 'metrics_from_saved_logits', 'paired_changes', 'stage_elapsed_seconds', 'exit_receipt')}, indent=2))
