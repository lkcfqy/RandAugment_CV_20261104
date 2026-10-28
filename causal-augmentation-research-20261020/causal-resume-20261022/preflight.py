"""Read-only research inputs plus a tiny local MPS capability check; write new run only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time

PROJECT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
RESEARCH = PROJECT / 'causal-augmentation-research-20261020'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / 'preflight.py')
    start = time.perf_counter()
    receipt = json.loads((RESEARCH / 'independent-design-challenge-20261021/delivery-receipt.json').read_text())
    old_checks = []
    for row in receipt['files']:
        path = RESEARCH / 'independent-design-challenge-20261021' / row['name']
        digest = sha(path)
        old_checks.append({'relative_path': str(path.relative_to(PROJECT)), 'bytes': path.stat().st_size,
                           'sha256': digest, 'matches_previous_receipt': digest == row['sha256'] and path.stat().st_size == row['bytes']})
    assets = []
    for name in ['RandAugment_NeurIPS_2020.pdf', 'randaugment-animation.html', 'augmentation-comparison.html']:
        path = PROJECT / name
        assets.append({'relative_path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    files = []
    for name in ['source-feature-bank.npz', 'normalizer.npz', 'readouts.npz', 'data']:
        path = RESEARCH / 'norb-view-transfer-pilot-20261021' / name
        if path.is_dir():
            files.extend({'relative_path': str(p.relative_to(PROJECT)), 'bytes': p.stat().st_size,
                          'decoded': False} for p in sorted(path.iterdir()) if p.is_file())
        else:
            files.append({'relative_path': str(path.relative_to(PROJECT)), 'bytes': path.stat().st_size,
                          'decoded': False})
    protected = {'at_utc': datetime.now(timezone.utc).isoformat(),
                 'previous_round_files': old_checks, 'classroom_assets': assets,
                 'previous_round_all_19_match': len(old_checks) == 19 and all(r['matches_previous_receipt'] for r in old_checks),
                 'final_role_arrays_or_pixels_decoded': False, 'data_file_inventory': files,
                 'note': 'Hashes check preservation only. No prior scientific calculation rerun; final arrays not opened.'}
    assert protected['previous_round_all_19_match']
    save(out / 'preservation-baseline.json', protected)

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    profile = subprocess.run(['system_profiler', 'SPHardwareDataType', 'SPDisplaysDataType', '-json'], capture_output=True, text=True, check=True)
    hardware = json.loads(profile.stdout)
    raw_cpu = hardware.get('SPHardwareDataType', [{}])[0]
    raw_gpu = hardware.get('SPDisplaysDataType', [{}])[0]
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'executable': sys.executable,
              'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
              'hardware': {k: raw_cpu.get(k) for k in ('machine_name', 'machine_model', 'chip_type', 'physical_memory', 'number_processors')},
              'GPU_model': raw_gpu.get('sppci_model'), 'logical_cpus': os.cpu_count(),
              'load_average': list(os.getloadavg()),
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy', 'Pillow')},
              'MPS_built': bool(torch.backends.mps.is_built()), 'MPS_available': bool(torch.backends.mps.is_available()),
              'CUDA_available': bool(torch.cuda.is_available()),
              'MPS_scientific_training_or_inference': False,
              'CPU_threads': torch.get_num_threads(), 'CPU_interop_threads': torch.get_num_interop_threads(),
              'new_dependencies': False, '8A_paths_read': False}
    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64) / 4096
    cpu = x @ x.T
    if result['MPS_available']:
        t = time.perf_counter()
        y = x.to('mps')
        z = y @ y.T
        torch.mps.synchronize()
        result['MPS_smoke_test'] = {'device': str(z.device), 'shape': list(z.shape),
             'wall_seconds_including_transfer_and_sync': time.perf_counter() - t,
             'max_absolute_error_vs_CPU': float((z.cpu() - cpu).abs().max()),
             'purpose': 'device capability only; not a scientific experiment or throughput benchmark'}
        del y, z
        torch.mps.empty_cache()
    ps = subprocess.run(['ps', '-axo', 'pid=,pcpu=,rss=,command='], capture_output=True, text=True, check=True)
    project_processes = []
    for line in ps.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4 and str(PROJECT) in fields[3]:
            project_processes.append({'pid': int(fields[0]), 'cpu_percent': float(fields[1]), 'RSS_KiB': int(fields[2]),
                                      'command': fields[3]})
    result['project_processes_at_preflight'] = project_processes
    result['wall_seconds'] = time.perf_counter() - start
    result['peak_RSS_bytes_Mac'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save(out / 'environment.json', result)
    print(json.dumps({'out': str(out), 'previous_round_all_19_match': True, 'MPS_available': result['MPS_available'],
                      'MPS_smoke_test': result.get('MPS_smoke_test'), 'packages': result['packages'],
                      'wall_seconds': result['wall_seconds'], 'peak_RSS_bytes_Mac': result['peak_RSS_bytes_Mac']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
