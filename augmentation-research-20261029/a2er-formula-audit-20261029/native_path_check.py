#!/usr/bin/env python3
"""Author buffer's repeated-input path; deterministic arithmetic, no training."""
import datetime
import hashlib
import json
import time
import numpy as np
import torch
from formula_audit import CORE, ROOT


def main():
    t = time.perf_counter()
    output = ROOT / 'results/native-path-check.json'
    assert not output.exists()
    b = CORE['ReplayBuffer'](keys=['x', 'der'], capacity=16, batch_size=16,
                             select='priority', remove='reservoir', verbose=False)
    for i in range(16):
        b.append(x=np.array([i, i/16, (i % 4)/8], dtype=np.float32),
                 der=np.array([i/32, -i/64], dtype=np.float32), priority=1.0)
    expected_x = np.stack(b.buffer['x'].tolist())
    expected_z = np.stack(b.buffer['der'].tolist())
    groups, checks = [], []
    for seed in [20261029, 20261030]:
        np.random.seed(seed)
        inputs, targets, ids, outputs = [], [], [], []
        for v in range(8):
            (data,), (index,) = b.sample()
            order = np.argsort(index)
            x = data['x'][order]
            z = data['der'][order]
            inputs.append(x.numpy().copy()); targets.append(z.numpy().copy())
            ids.append(index.tolist())
            # A fixed arithmetic map only illustrates the repeated-input result.
            outputs.append(torch.stack([x[:, 1] + x[:, 2], x[:, 1] - x[:, 2]], dim=1).numpy())
        h = np.stack(outputs).astype(np.float64)
        z = expected_z.astype(np.float64)
        davg = (0.5 * ((h-z)**2).sum(axis=-1)).mean(axis=0)
        dmean = 0.5 * ((h.mean(axis=0)-z)**2).sum(axis=-1)
        variance = 0.5 * ((h-h.mean(axis=0))**2).sum(axis=-1).mean(axis=0)
        group = {'seed': seed, 'visits': ids,
                 'input_max_abs_difference': float(np.max(np.abs(np.stack(inputs)-expected_x))),
                 'target_max_abs_difference': float(np.max(np.abs(np.stack(targets)-expected_z))),
                 'Davg': davg.tolist(), 'Dmean': dmean.tolist(),
                 'half_view_variance': variance.tolist(),
                 'Davg_Dmean_max_abs': float(np.max(np.abs(davg-dmean)))}
        for key in ['input_max_abs_difference', 'target_max_abs_difference', 'Davg_Dmean_max_abs']:
            checks.append({'name': f'{seed}_{key}_zero', 'passed': group[key] == 0})
        checks.append({'name': f'{seed}_view_variance_zero', 'passed': bool(np.all(variance == 0))})
        groups.append(group)
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'device': 'cpu', 'optimizer_steps': 0, 'neural_network_runs': 0,
              'scope': 'Released sample buffer + deterministic arithmetic; not unpublished image benchmark reproduction.',
              'groups': groups, 'checks': checks, 'all_passed': all(x['passed'] for x in checks),
              'core_wall_seconds': time.perf_counter()-t}
    with output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k,v in result.items() if k not in {'groups', 'checks'}}, indent=2))
    assert result['all_passed']


if __name__ == '__main__':
    main()
