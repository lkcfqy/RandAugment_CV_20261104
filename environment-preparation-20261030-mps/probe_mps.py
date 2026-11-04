"""One small synthetic MPS forward computation; no project code, data, or models."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import platform
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise SystemExit('Refusing to overwrite an existing MPS receipt.')

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTORCH_MPS_FAST_MATH'] = '0'
started = time.monotonic()
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'purpose': 'small synthetic MPS tensor readiness only; not experiment reproduction',
    'python': sys.version, 'executable': sys.executable,
    'platform': platform.platform(), 'architecture': platform.machine(),
    'mps_cpu_fallback': False, 'mps_fast_math': False,
    'model_or_dataset_downloads': False, 'training': False,
    'tensor_shape': [16, 16], 'dtype': 'float32',
    'operation': 'x @ x.T + 0.25; x = arange(256).reshape(16,16)/128 - 1',
    'comparison_atol': 1e-5, 'comparison_rtol': 1e-5,
    'wall_time_limit_seconds': 45,
}
try:
    import torch
    import torchvision
    torch.set_num_threads(1)
    receipt.update(torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                   torch_path=torch.__file__, torchvision_path=torchvision.__file__,
                   mps_is_built=torch.backends.mps.is_built(),
                   mps_is_available=torch.backends.mps.is_available())
    if not receipt['mps_is_built'] or not receipt['mps_is_available']:
        raise RuntimeError('MPS backend is not built or not available in this process.')
    with torch.inference_mode():
        x_cpu = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 128 - 1
        expected = x_cpu @ x_cpu.T + 0.25
        x_mps = x_cpu.to('mps')
        torch.mps.synchronize()
        operation_started = time.monotonic()
        actual_mps = x_mps @ x_mps.T + 0.25
        torch.mps.synchronize()
        receipt['synchronized_operation_seconds'] = time.monotonic() - operation_started
        receipt['result_device'] = str(actual_mps.device)
        actual = actual_mps.cpu()
        receipt['all_finite'] = bool(torch.isfinite(actual).all())
        receipt['max_abs_error_vs_cpu'] = float((actual - expected).abs().max())
        receipt['allclose_cpu'] = bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        receipt['current_mps_allocated_bytes'] = torch.mps.current_allocated_memory()
        if actual_mps.device.type != 'mps' or not receipt['all_finite'] or not receipt['allclose_cpu']:
            raise RuntimeError('MPS output failed device, finite-value, or CPU comparison check.')
    receipt['status'] = 'passed'
except Exception as exc:
    receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
finally:
    receipt['elapsed_seconds'] = time.monotonic() - started
    with output.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
if receipt['status'] != 'passed':
    raise SystemExit(1)
