#!/usr/bin/env python3
"""Bounded CPU/MPS environment acceptance; synthetic data, no optimizer updates."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

# Explicitly prevent an unsupported MPS operation from silently running on CPU.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTORCH_MPS_FAST_MATH'] = '0'

PROTOCOL = {
    'purpose': 'hardware and environment acceptance, not research training',
    'seed': 20261029,
    'dtype': 'float32; CPU float64 reference also checked',
    'cpu_threads': 4,
    'cpu_interop_threads': 1,
    'optimizer_steps': 0,
    'external_data_or_weights': False,
    'model_input_shape': [8, 3, 32, 32],
    'model': 'Conv2d(3,8,3,pad=1), ReLU, AvgPool2d(2), Conv2d(8,16,3,pad=1), ReLU, AdaptiveAvgPool2d(1), Flatten, Linear(16,10)',
    'model_mode': 'eval; autograd enabled; no BatchNorm or Dropout',
    'model_repeats_per_device': 4,
    'timing': 'first compute plus 3 subsequent computes; synchronized MPS; setup separately; not a throughput benchmark',
    'thresholds': {
        'exact_tensor': {'atol': 0.0, 'rtol': 0.0},
        'matmul': {'atol': 2e-5, 'rtol': 2e-4},
        'model_logits': {'atol': 2e-5, 'rtol': 2e-4},
        'model_loss': {'atol': 1e-5, 'rtol': 1e-5},
        'model_gradient': {'atol': 2e-5, 'rtol': 2e-3},
    },
    'memory': 'Darwin process peak RSS in bytes; MPS allocator/driver snapshots are not peak GPU memory; unified memory values must not be summed',
}


def command_value(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as exc:
        return {'unavailable': str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new result filename')
    started = time.perf_counter()
    report = {
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol': PROTOCOL,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': {'version': sys.version, 'executable': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix},
        'platform': {'system': platform.system(), 'macos': platform.mac_ver()[0], 'architecture': platform.machine()},
        'hardware': {key: command_value(['/usr/sbin/sysctl', '-n', key]) for key in ['hw.memsize', 'hw.logicalcpu', 'machdep.cpu.brand_string']},
        'checks': {}, 'memory_snapshots': [], 'warnings': [],
    }
    try:
        import torch
        import torchvision
        import numpy
        import PIL

        torch.set_num_threads(PROTOCOL['cpu_threads'])
        torch.set_num_interop_threads(PROTOCOL['cpu_interop_threads'])
        torch.manual_seed(PROTOCOL['seed'])
        report['versions'] = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': numpy.__version__, 'pillow': PIL.__version__}
        report['module_locations'] = {'torch': torch.__file__, 'torchvision': torchvision.__file__}
        report['mps'] = {'built': bool(torch.backends.mps.is_built()), 'available': bool(torch.backends.mps.is_available()), 'cpu_fallback': os.environ['PYTORCH_ENABLE_MPS_FALLBACK'], 'fast_math': os.environ['PYTORCH_MPS_FAST_MATH']}
        report['checks']['isolated_environment'] = sys.prefix != sys.base_prefix and all(str(Path(v)).startswith(sys.prefix + '/') for v in report['module_locations'].values())
        report['checks']['version_pair'] = torch.__version__ == '2.13.0' and torchvision.__version__ == '0.28.0'
        report['checks']['mps_built_and_available'] = report['mps']['built'] and report['mps']['available']

        def sync(device):
            if device == 'mps':
                torch.mps.synchronize()

        def memory(label):
            snap = {'label': label, 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            rss = command_value(['/bin/ps', '-o', 'rss=', '-p', str(os.getpid())])
            snap['process_current_rss_bytes'] = int(rss) * 1024 if isinstance(rss, str) and rss.isdigit() else rss
            if report['mps']['available']:
                torch.mps.synchronize()
                for key in ['current_allocated_memory', 'driver_allocated_memory', 'recommended_max_memory']:
                    try:
                        snap['mps_' + key + '_bytes'] = int(getattr(torch.mps, key)())
                    except Exception as exc:
                        snap['mps_' + key + '_bytes'] = {'unavailable': str(exc)}
            report['memory_snapshots'].append(snap)

        def compare(actual, reference, tolerance):
            actual = actual.detach().cpu().to(torch.float64)
            reference = reference.detach().cpu().to(torch.float64)
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
            result = dict(tolerance, shape=list(reference.shape), finite=finite)
            if finite:
                difference = actual - reference
                allowed = tolerance['atol'] + tolerance['rtol'] * reference.abs()
                result.update({
                    'pass': bool((difference.abs() <= allowed).all()),
                    'max_abs_error': difference.abs().max().item(),
                    'rmse': difference.square().mean().sqrt().item(),
                    'relative_l2_error': (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item(),
                })
            else:
                result.update({'pass': False, 'max_abs_error': None, 'rmse': None, 'relative_l2_error': None})
            return result

        memory('after_import')
        # CPU extension call catches a common torch/torchvision ABI mismatch.
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]])
        keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5).tolist()
        report['torchvision_extension'] = {'operation': 'CPU nms', 'expected_indices': [0, 2], 'actual_indices': keep}
        report['checks']['torchvision_cpu_extension'] = keep == [0, 2]
        devices = ['cpu'] + (['mps'] if report['mps']['available'] else [])
        generator = torch.Generator(device='cpu').manual_seed(PROTOCOL['seed'])
        a = torch.randn(64, 64, generator=generator, dtype=torch.float32) / 8
        b = torch.randn(64, 64, generator=generator, dtype=torch.float32) / 8
        reference_mm = a.double() @ b.double()
        exact = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 64
        report['tensor_checks'] = {}
        for device in devices:
            t0 = time.perf_counter()
            ad, bd, xd = a.to(device), b.to(device), exact.to(device)
            sync(device)
            setup = time.perf_counter() - t0
            t0 = time.perf_counter()
            exact_out, mm_out = xd * 2 + 1, ad @ bd
            sync(device)
            elapsed = time.perf_counter() - t0
            checks = {
                'setup_seconds': setup, 'compute_seconds': elapsed,
                'exact_arithmetic': compare(exact_out, exact * 2 + 1, PROTOCOL['thresholds']['exact_tensor']),
                'matmul_vs_cpu_float64': compare(mm_out, reference_mm, PROTOCOL['thresholds']['matmul']),
            }
            report['tensor_checks'][device] = checks
            report['checks'][device + '_tensor'] = checks['exact_arithmetic']['pass'] and checks['matmul_vs_cpu_float64']['pass']
        memory('after_tensor_checks')

        base = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(16, 10),
        ).eval()
        inputs = torch.randn(*PROTOCOL['model_input_shape'], generator=generator, dtype=torch.float32)
        labels = torch.arange(PROTOCOL['model_input_shape'][0]) % 10
        report['model_parameter_count'] = sum(p.numel() for p in base.parameters())
        outputs = {}
        report['model_runs'] = {}
        for name, device, dtype, repeats in [('cpu_float64_reference', 'cpu', torch.float64, 1)] + [(d, d, torch.float32, PROTOCOL['model_repeats_per_device']) for d in devices]:
            t0 = time.perf_counter()
            model = copy.deepcopy(base).to(device=device, dtype=dtype)
            x = inputs.to(device=device, dtype=dtype).detach().clone().requires_grad_(True)
            y = labels.to(device)
            sync(device)
            setup = time.perf_counter() - t0
            before = [p.detach().clone() for p in model.parameters()]
            durations = []
            for repeat in range(repeats):
                model.zero_grad(set_to_none=True)
                x.grad = None
                sync(device)
                t0 = time.perf_counter()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                sync(device)
                durations.append(time.perf_counter() - t0)
                if repeat == 0:
                    outputs[name] = {
                        'logits': logits.detach().cpu().clone(), 'loss': loss.detach().cpu().clone(),
                        'parameter_gradient': torch.cat([p.grad.detach().cpu().reshape(-1) for p in model.parameters()]),
                        'input_gradient': x.grad.detach().cpu().clone(),
                    }
            unchanged = all(torch.equal(p.detach(), old) for p, old in zip(model.parameters(), before))
            finite = all(bool(torch.isfinite(t).all()) for t in outputs[name].values())
            report['checks'][name + '_finite_and_no_update'] = unchanged and finite
            report['model_runs'][name] = {'device': device, 'dtype': str(dtype), 'setup_seconds': setup, 'forward_backward_seconds': durations, 'loss': outputs[name]['loss'].item(), 'finite': finite, 'parameters_unchanged': unchanged, 'optimizer_steps': 0}
            memory('after_model_' + name)
            del model, x, y, before, logits, loss
        pairs = [('cpu', 'cpu_float64_reference')]
        if 'mps' in devices:
            pairs += [('mps', 'cpu_float64_reference'), ('mps', 'cpu')]
        report['model_comparisons'] = {}
        for actual, reference in pairs:
            key = actual + '_vs_' + reference
            checks = {}
            for field in ['logits', 'loss', 'parameter_gradient', 'input_gradient']:
                category = 'model_gradient' if 'gradient' in field else 'model_' + field
                checks[field] = compare(outputs[actual][field], outputs[reference][field], PROTOCOL['thresholds'][category])
            report['model_comparisons'][key] = checks
            report['checks'][key] = all(c['pass'] for c in checks.values())
        memory('completed')
        report['all_passed'] = all(report['checks'].values())
    except Exception as exc:
        report['all_passed'] = False
        report['fatal_error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    report['total_wall_seconds'] = time.perf_counter() - started
    report['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'result': str(args.output), 'all_passed': report['all_passed'], 'mps': report.get('mps'), 'checks': report['checks'], 'seconds': report['total_wall_seconds'], 'fatal_error': report.get('fatal_error')}, indent=2))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
