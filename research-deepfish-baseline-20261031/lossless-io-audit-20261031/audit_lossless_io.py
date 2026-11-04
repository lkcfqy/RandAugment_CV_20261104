"""Standalone CPU file audit; imports no model implementation or cloud API."""
from __future__ import annotations
from portable_dev.paths import project_path as _portable_project_path

import collections
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import torch

ROOT = _portable_project_path(__file__, 'research-deepfish-baseline-20261031')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
ORIGINAL_SHA = '822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3'
EXPECTED = {
    'A': '5a2a8d4955c0a91641c9e66ac879740077fc4640f06379e850640a46da2268f0',
    'B': 'd14239f71925ac4f813eb80c8241ce7ca1670d1017bcf047ecae994327870778',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def raw_view(tensor):
    # Reinterpreting bytes is not numeric conversion: preserves signed zero and NaN payloads.
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def raw_hash(tensor):
    return hashlib.sha256(memoryview(raw_view(tensor).numpy())).hexdigest()


def exact_tensor(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(raw_view(a), raw_view(b))


def meta(tensor):
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(), layout=str(tensor.layout),
                requires_grad=tensor.requires_grad, tensor_bytes=tensor.numel() * tensor.element_size())


def content_state_digest(state):
    # Matches the saved experiment's state_digest convention, including key/dtype/shape.
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(canonical([key, str(tensor.dtype), list(tensor.shape)]))
        h.update(memoryview(raw_view(tensor).numpy()))
    return h.hexdigest()


def complete_state_digest(state):
    # Also commits to container/key order and the observed storage metadata.
    h = hashlib.sha256(canonical({'format': 'dense-independent-state-v1',
                                  'container': type(state).__name__, 'key_order': list(state)}))
    for key, tensor in state.items():
        h.update(canonical([key, meta(tensor)]))
        h.update(memoryview(raw_view(tensor).numpy()))
    return h.hexdigest()


def safe_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert isinstance(state, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                          for k, v in state.items()), 'Tensor state dict required'
    assert all(t.device.type == 'cpu' and t.layout == torch.strided and not t.is_quantized
               and not t.is_conj() and not t.is_neg() for t in state.values())
    return state


def validate_observed_target(state):
    assert type(state) is dict and not getattr(state, '__dict__', {})
    assert len(state) == 908
    assert all(t.is_contiguous() and t.storage_offset() == 0 and not t.requires_grad
               and t.untyped_storage().nbytes() == t.numel() * t.element_size()
               for t in state.values()), 'Unexpected target layout; refusing lossy normalization'
    assert len({t.untyped_storage().data_ptr() for t in state.values()}) == len(state), \
        'Unexpected storage aliasing; do not silently change it'


def save_torch(path, value):
    with path.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())


def reconstruct(package, base, actual_base_sha):
    assert package['format'] == 'exact-whole-tensor-reference-v1'
    assert actual_base_sha == package['base_file_sha256'] == ORIGINAL_SHA
    assert package['base_content_state_sha256'] == content_state_digest(base)
    refs = package['reference_keys']
    changed = package['changed_tensors']
    assert not set(refs) & set(changed)
    assert set(refs) | set(changed) == set(package['target_key_order'])
    assert len(package['target_key_order']) == len(set(package['target_key_order']))
    result = {}
    for key in package['target_key_order']:
        if key in refs:
            tensor = base[key]
            assert raw_hash(tensor) == refs[key]['tensor_sha256']
        else:
            tensor = changed[key]
        # Clone to preserve the target's observed unique, zero-offset storage per tensor.
        result[key] = tensor.detach().clone(memory_format=torch.contiguous_format)
        assert meta(result[key]) == package['target_metadata'][key]
    validate_observed_target(result)
    assert content_state_digest(result) == package['target_content_state_sha256']
    assert complete_state_digest(result) == package['target_complete_state_sha256']
    return result


def audit_arm(arm, base, base_digest):
    path = ROOT / f'recovery-v5-20261031/gpu-run/artifacts/final_{arm}_U200.pth'
    source_sha = sha_file(path)
    assert source_sha == EXPECTED[arm]
    start = time.perf_counter()
    state = safe_state(path)
    validate_observed_target(state)
    records, refs, changed = [], {}, {}
    groups = collections.defaultdict(collections.Counter)
    for key, tensor in state.items():
        original = base.get(key)
        same = original is not None and exact_tensor(tensor, original)
        status = 'identical_to_original' if same else ('new_key' if original is None else 'changed')
        record = dict(key=key, status=status, **meta(tensor), tensor_sha256=raw_hash(tensor),
                      original_tensor_sha256=raw_hash(original) if original is not None else None)
        records.append(record)
        group = groups[key.split('.')[0]]
        group[status + '_keys'] += 1
        group[status + '_bytes'] += record['tensor_bytes']
        if same:
            refs[key] = {'tensor_sha256': record['tensor_sha256']}
        else:
            changed[key] = tensor
    target_content = content_state_digest(state)
    target_complete = complete_state_digest(state)
    package = dict(format='exact-whole-tensor-reference-v1',
                   base_file_sha256=ORIGINAL_SHA, base_content_state_sha256=base_digest,
                   target_source_file_sha256=source_sha, target_container='dict',
                   target_key_order=list(state), target_metadata={k: meta(t) for k, t in state.items()},
                   target_content_state_sha256=target_content,
                   target_complete_state_sha256=target_complete,
                   reference_keys=refs, changed_tensors=changed,
                   non_tensor_state={}, optimizer_state_present_in_source=False,
                   byteorder=sys.byteorder)
    timings = {'compare_and_hash_seconds': time.perf_counter() - start}
    # Temporary files are new derived duplicates only; original checkpoints are read-only.
    with tempfile.TemporaryDirectory(prefix=f'exact_rebuild_{arm}_', dir=OUT) as directory:
        temp = Path(directory)
        diff = temp / f'delta_{arm}.pth'
        full = temp / f'rebuilt_{arm}.pth'
        start = time.perf_counter()
        save_torch(diff, package)
        timings['package_write_seconds'] = time.perf_counter() - start
        package_bytes, package_sha = diff.stat().st_size, sha_file(diff)
        del package
        start = time.perf_counter()
        loaded = torch.load(diff, map_location='cpu', weights_only=True)
        rebuilt = reconstruct(loaded, base, sha_file(SOURCE))
        timings['package_reload_and_reconstruct_verify_seconds'] = time.perf_counter() - start
        assert list(rebuilt) == list(state)
        assert all(exact_tensor(rebuilt[k], state[k]) and meta(rebuilt[k]) == meta(state[k]) for k in state)
        start = time.perf_counter()
        save_torch(full, rebuilt)
        timings['full_checkpoint_write_seconds'] = time.perf_counter() - start
        full_bytes, full_sha = full.stat().st_size, sha_file(full)
        del rebuilt, loaded
        gc.collect()
        start = time.perf_counter()
        readback = safe_state(full)
        validate_observed_target(readback)
        assert list(readback) == list(state)
        verified = 0
        for key, tensor in state.items():
            assert exact_tensor(readback[key], tensor)
            assert meta(readback[key]) == meta(tensor)
            assert raw_hash(readback[key]) == raw_hash(tensor)
            verified += 1
        rebuilt_content, rebuilt_complete = content_state_digest(readback), complete_state_digest(readback)
        assert rebuilt_content == target_content and rebuilt_complete == target_complete
        timings['full_reload_and_all_tensor_verification_seconds'] = time.perf_counter() - start
        del readback
    assert sha_file(path) == source_sha
    counts = collections.Counter(record['status'] for record in records)
    bytes_by_status = collections.Counter()
    for record in records:
        bytes_by_status[record['status']] += record['tensor_bytes']
    report = dict(status='EXACT_STATE_RECONSTRUCTION_VERIFIED', arm=arm,
                  source_path=str(path.relative_to(ROOT)), source_bytes=path.stat().st_size,
                  source_sha256=source_sha, base_sha256=ORIGINAL_SHA,
                  counts=dict(counts), bytes_by_status=dict(bytes_by_status),
                  total_tensor_bytes=sum(bytes_by_status.values()), groups=dict(groups),
                  package_bytes=package_bytes, package_sha256=package_sha,
                  package_savings_vs_source_bytes=path.stat().st_size - package_bytes,
                  package_savings_fraction=1 - package_bytes / path.stat().st_size,
                  reconstructed_file_bytes=full_bytes, reconstructed_file_sha256=full_sha,
                  original_content_state_sha256=target_content,
                  rebuilt_content_state_sha256=rebuilt_content,
                  original_complete_state_sha256=target_complete,
                  rebuilt_complete_state_sha256=rebuilt_complete,
                  tensors_verified_bitwise=verified, key_order_and_storage_metadata_verified=True,
                  all_target_storages_unique=True, original_type='dict',
                  source_has_optimizer_or_non_tensor_state=False,
                  safe_cpu_loads_only=True, model_instantiated=False,
                  raw_requires_grad_flag_not_a_freezing_claim=True,
                  complete_state_hash_includes_dtype_shape_stride_offset_layout_and_order=True,
                  full_torch_zip_file_hash_need_not_match_across_serializer_versions=True,
                  temporary_generated_files_removed_after_validation=True,
                  original_checkpoint_preserved=True, timings=timings, tensors=records)
    write_json(OUT / f'ARM_{arm}_AUDIT.json', report)
    print(json.dumps({k: report[k] for k in ['arm', 'status', 'counts', 'bytes_by_status',
                     'package_bytes', 'package_savings_vs_source_bytes',
                     'tensors_verified_bitwise', 'original_complete_state_sha256']}), flush=True)
    del state, changed
    gc.collect()
    return report


def main():
    torch.set_num_threads(4)
    assert exact_tensor(torch.tensor([0.0]), torch.tensor([0.0]))
    assert not exact_tensor(torch.tensor([0.0]), torch.tensor([-0.0])), 'Signed-zero regression'
    assert not exact_tensor(torch.tensor([1], dtype=torch.int32), torch.tensor([1], dtype=torch.int64))
    assert shutil.disk_usage(OUT).free >= 8 * 1024**3
    assert sha_file(SOURCE) == ORIGINAL_SHA and SOURCE.stat().st_size == 691901857
    base = safe_state(SOURCE)
    base_digest = content_state_digest(base)
    reports = [audit_arm(arm, base, base_digest) for arm in 'AB']
    assert sha_file(SOURCE) == ORIGINAL_SHA
    repeat = ROOT / 'repeat-seed20261101-20261031'
    freeze_path = repeat / 'gpu-run/U_FREEZE.json'
    ledger_path = repeat / 'gpu-run/controller_report.json'
    freeze = json.loads(freeze_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    t = freeze['raw_timings']
    remaining_cu = 2 - ledger['cumulative_CU_upper']
    remaining_gpu = 7200 - ledger['cumulative_GPU_seconds_upper']
    available_seconds = min(remaining_gpu, remaining_cu * 3600 / ledger['live_rate'])
    max_package_bytes = max(r['package_bytes'] for r in reports)
    non_io_remaining = (4 * t['init_seconds'] + 1.5 * 200 * (t['p90_update_a'] + t['p90_update_b'])
                        + 1.5 * 128 * t['p90_predict_seconds'] + 120)
    measured_setup = ledger['remote_result']['allocated_seconds']
    new_io = 4 * max_package_bytes / t['write_bps']
    modeled_full_seconds = measured_setup + non_io_remaining + new_io + 180
    # This lower bound omits metadata entirely and transfers only non-identical tensor bytes.
    minimal_changed = max(r['bytes_by_status']['changed'] + r['bytes_by_status']['new_key'] for r in reports)
    ideal_io = 4 * minimal_changed / t['write_bps']
    ideal_full = measured_setup + non_io_remaining + ideal_io + 180
    summary = dict(status='NO_MATERIAL_SAVINGS_FOR_WHOLE_TENSOR_REFERENCES',
                   utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   python=sys.version, torch_version=torch.__version__, device='cpu',
                   source_file_sha256=ORIGINAL_SHA, source_content_state_sha256=base_digest,
                   source_tensor_count=len(base), original_only_keys=sorted(set(base)-set(reports[0]['tensors'][i]['key'] for i in range(len(reports[0]['tensors'])))),
                   arms=[{k: r[k] for k in r if k != 'tensors'} for r in reports],
                   local_generated_duplicates_removed=True, all_original_assets_unchanged=True,
                   no_dependency_install=True, no_network=True, no_GPU_allocation=True,
                   no_model_calls=True, no_cloud_runner_or_scientific_config_changes=True,
                   no_execution_GO=True, synthetic_byte_comparison_checks=3,
                   saved_requires_grad_false_is_detach_not_freeze=True,
                   cost=dict(formula_unchanged='4*init + 1.5*200*(p90_A+p90_B) + 1.5*128*p90_predict + 4*transfer_bytes/bps + 120; measured cold preparation + 180 release reserve',
                             remaining_CU=remaining_cu, remaining_GPU_seconds=remaining_gpu,
                             last_saved_rate=ledger['live_rate'], available_seconds=available_seconds,
                             measured_cold_prepare_seconds=measured_setup, release_reserve_seconds=180,
                             saved_bandwidth_bytes_per_second=t['write_bps'], original_tensor_bytes=t['checkpoint_bytes'],
                             measured_max_package_bytes=max_package_bytes,
                             measured_package_IO_seconds=new_io,
                             estimated_full_seconds_with_package=modeled_full_seconds,
                             estimated_CU_with_package=modeled_full_seconds*ledger['live_rate']/3600,
                             shortfall_seconds=modeled_full_seconds-available_seconds,
                             fits_remaining_CU_and_GPU_time=modeled_full_seconds<=available_seconds,
                             unmeasured_future_encoding_reconstruction_overhead_excluded_not_assumed_zero=True,
                             ideal_changed_bytes_without_metadata=minimal_changed,
                             ideal_savings_vs_original_tensor_IO_seconds=4*(t['checkpoint_bytes']-minimal_changed)/t['write_bps'],
                             ideal_full_seconds_without_metadata=ideal_full,
                             even_ideal_changed_only_cost_fits=ideal_full<=available_seconds,
                             freeze_sha256=sha_file(freeze_path), ledger_sha256=sha_file(ledger_path)),
                   scope_limit='No claim about every possible lossless compression method; this audit tests whole unchanged tensor references plus all complete changed/new tensors only.')
    assert not summary['cost']['fits_remaining_CU_and_GPU_time']
    assert not summary['cost']['even_ideal_changed_only_cost_fits']
    write_json(OUT / 'SUMMARY.json', summary)
    print(json.dumps({'status': summary['status'], 'cost': summary['cost']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
