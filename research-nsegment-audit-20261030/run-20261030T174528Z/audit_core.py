"""Independent, minimal, inspected reimplementation; never imports upstream source.

Reference: NSegment transforms.py 2541-2570, commit 5c66795b38ffd351f2d3363bc4eb3e5c3b580fac.
Upstream MIT notice is preserved in sources/LICENSE. NumPy/OpenCV + stdlib only.
"""
import hashlib
import json
import os
import platform
import resource
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def checkpoint(stage, **kwargs):
    row = dict(utc=utc(), stage=stage, **kwargs)
    with (ROOT / 'checkpoints.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
    print(json.dumps(row, ensure_ascii=False), flush=True)


def save_json(name, obj):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def guard(stage, seconds):
    deadline = min(time.time() + seconds, PROTOCOL['deadline_epoch'])
    def monitor():
        while True:
            reason = None
            if peak_rss() >= PROTOCOL['budgets']['memory_bytes']:
                reason = 'memory_limit'
            elif time.time() >= deadline:
                reason = 'deadline'
            if reason:
                checkpoint('budget_stop', running_stage=stage, reason=reason,
                           peak_rss_bytes=peak_rss())
                os._exit(124)
            time.sleep(0.2)
    threading.Thread(target=monitor, daemon=True).start()
    return deadline


def state_record(state):
    return dict(algorithm=state[0], keys=state[1].tolist(), position=int(state[2]),
                has_gauss=int(state[3]), cached_gaussian=float(state[4]))


def state_restore(record):
    return (record['algorithm'], np.array(record['keys'], dtype=np.uint32),
            record['position'], record['has_gauss'], record['cached_gaussian'])


def state_hash(state):
    return hashlib.sha256(json.dumps(state_record(state), sort_keys=True).encode()).hexdigest()


def array_hash(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def gate(prob):
    # Upstream 2549-2551: do not optimize away p=1; comparison is >, not >=.
    value = float(np.random.rand())
    return value, not (value > prob)


def make_maps(shape, alpha_sigma_list, random_state=None):
    # Upstream 2557-2559; deterministic audits supply the already-existing argument.
    if random_state is None:
        random_state = np.random.RandomState(None)
    before = state_hash(random_state.get_state())
    # Upstream 2560: retain randint even for a singleton.
    choice = int(random_state.randint(0, len(alpha_sigma_list)))
    alpha, sigma = alpha_sigma_list[choice]
    after_choice = state_hash(random_state.get_state())
    # Upstream 2561-2563: float64 uniform x, then y, both drawn before either blur.
    dx = alpha * (2 * random_state.rand(*shape) - 1)
    dy = alpha * (2 * random_state.rand(*shape) - 1)
    after_field = state_hash(random_state.get_state())
    # Upstream 2564-2565: retain (0,0) auto kernel and default border.
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    # Upstream 2566-2568: integer coordinate mesh, add float64, clip, cast to float32.
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
    return dict(x=x, y=y, dx=dx, dy=dy, map_x=map_x, map_y=map_y,
                alpha=int(alpha), sigma=int(sigma), choice=choice,
                local_before=before, local_after_choice=after_choice,
                local_after_field=after_field)


def remap(segment, maps, identity=False):
    # Upstream 2569: actual nearest-neighbor remap. C substitutes identity maps only.
    if identity:
        map_x, map_y = maps['x'].astype(np.float32), maps['y'].astype(np.float32)
    else:
        map_x, map_y = maps['map_x'], maps['map_y']
    return cv2.remap(segment, map_x, map_y, interpolation=cv2.INTER_NEAREST)


def recover_donors(maps):
    # Separate X/Y ramps have exact float32 values at both audited sizes.
    return remap(maps['x'].astype(np.float32), maps), remap(maps['y'].astype(np.float32), maps)


def check_uint8(maps, donor_x, donor_y):
    # Four uint8 channels encode two uint16 coordinate components; never CV_32S.
    x, y = maps['x'], maps['y']
    packed = np.empty(x.shape + (4,), dtype=np.uint8)
    packed[..., 0], packed[..., 1] = x & 255, x >> 8
    packed[..., 2], packed[..., 3] = y & 255, y >> 8
    sampled = remap(packed, maps)
    rx = sampled[..., 0].astype(np.uint16) + (sampled[..., 1].astype(np.uint16) << 8)
    ry = sampled[..., 2].astype(np.uint16) + (sampled[..., 3].astype(np.uint16) << 8)
    assert np.array_equal(rx, donor_x), 'uint8 X differs from float32 ramp'
    assert np.array_equal(ry, donor_y), 'uint8 Y differs from float32 ramp'
    return True


def region_counts(changed, radius):
    h, w = changed.shape
    r = radius
    views = {'interior': changed[r:h-r, r:w-r],
             'corner_tl': changed[:r, :r], 'corner_tr': changed[:r, w-r:],
             'corner_bl': changed[h-r:, :r], 'corner_br': changed[h-r:, w-r:]}
    out = {}
    for key, view in views.items():
        out[key + '_n'] = int(view.size)
        out[key + '_changed'] = int(np.count_nonzero(view))
        out[key + '_rate'] = out[key + '_changed'] / view.size
    edge_n = changed.size - sum(out[k + '_n'] for k in views)
    edge_changed = int(np.count_nonzero(changed)) - sum(out[k + '_changed'] for k in views)
    out.update(edge_n=edge_n, edge_changed=edge_changed, edge_rate=edge_changed / edge_n)
    return out


def environment():
    return dict(utc=utc(), python=sys.version, executable=sys.executable,
                platform=platform.platform(), machine=platform.machine(),
                numpy=np.__version__, opencv=cv2.__version__,
                cv2_threads=cv2.getNumThreads(), opencl_active=cv2.ocl.useOpenCL(),
                default_border=int(cv2.BORDER_DEFAULT),
                reflect101=int(cv2.BORDER_REFLECT_101),
                relevant_env={k: os.environ.get(k) for k in
                              ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']})
