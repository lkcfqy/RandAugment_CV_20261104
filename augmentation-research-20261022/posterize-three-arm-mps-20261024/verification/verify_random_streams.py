"""Recompute CPU RNG draws from manifest metadata only. No model/image loading."""
from pathlib import Path
import hashlib, json, math
from datetime import datetime, timezone
import numpy as np
import torch

RUN = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/posterize-three-arm-mps-20261024')
REVIEW = Path('/Users/lkc/Documents/Codex/2026-10-24/task-9/mps_integrity_review')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cfg = json.loads((RUN/'protocol.json').read_text())
rows = json.loads((RUN.parent/'pilot-20261022/fit-manifest.json').read_text())['rows']
train = [r for r in rows if r['candidate_partition'] == 'train']
mapping = {r['source_id']: r for r in train}
orders = json.loads((RUN/'orders.json').read_text())
trace = [json.loads(s) for s in (RUN/'trajectories.jsonl').read_text().splitlines()]


def seed(*parts):
    return int(hashlib.sha256(':'.join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def generator(*parts):
    return torch.Generator(device='cpu').manual_seed(seed(*parts))


def randint(g, hi):
    return int(torch.randint(hi, (1,), generator=g, device='cpu').item())


def draw_weak(row, g):
    height, width = row['height'], row['width']
    log_ratio = torch.log(torch.tensor((0.9, 1.1), device='cpu'))
    crop = None
    for _ in range(10):
        area = height * width * torch.empty(1, device='cpu').uniform_(0.8, 1.0, generator=g).item()
        ratio = torch.exp(torch.empty(1, device='cpu').uniform_(log_ratio[0], log_ratio[1], generator=g)).item()
        w, h = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
        if 0 < w <= width and 0 < h <= height:
            crop = [randint(g, height-h+1), randint(g, width-w+1), h, w]
            break
    if crop is None:
        ratio = width / height
        if ratio < 0.9:
            w, h = width, round(width/0.9)
        elif ratio > 1.1:
            h, w = height, round(height*1.1)
        else:
            w, h = width, height
        crop = [(height-h)//2, (width-w)//2, h, w]
    flip = bool(torch.rand((), generator=g, device='cpu') < 0.5)
    return {'crop': crop, 'flip': flip}


# Match the locked torchvision float32 magnitude construction, using only scalars.
lin = lambda end: float(torch.linspace(0.0, end, 31, device='cpu')[9].item())
magnitude = {name: 0.0 for name in cfg['pool']}
for name in ('ShearX', 'ShearY'):
    magnitude[name] = lin(0.3)
for name in ('TranslateX', 'TranslateY'):
    magnitude[name] = lin(150.0 / 331.0 * 224)
magnitude['Rotate'] = lin(30.0)
for name in ('Brightness', 'Color', 'Contrast', 'Sharpness'):
    magnitude[name] = lin(0.9)
magnitude['Posterize'] = float((8 - (torch.arange(31, device='cpu') / 7.5).round().int())[9].item())
magnitude['Solarize'] = float(torch.linspace(255.0, 0.0, 31, device='cpu')[9].item())
signed = {'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness', 'Color', 'Contrast', 'Sharpness'}
survivors = [x for x in cfg['pool'] if x != 'Posterize']


def draw(g, pool):
    name = pool[randint(g, len(pool))]
    m = magnitude[name]
    if name in signed and randint(g, 2):
        m = -m
    return {'name': name, 'magnitude': m}


for epoch in range(30):
    perm = np.random.default_rng(seed('order-v1', 20261024, epoch)).permutation(591)
    assert orders[epoch] == [train[int(i)]['source_id'] for i in perm], epoch

replacements = 0
for item in trace:
    epoch, sid = item['epoch'], item['source_id']
    weak = draw_weak(mapping[sid], generator('weak-v1', 20261024, epoch, sid))
    assert weak == item['weak'], ('weak', epoch, sid)
    g = generator('RA-v1', 20261024, epoch, sid)
    baseline = [draw(g, cfg['pool']), draw(g, cfg['pool'])]
    assert baseline == item['B'], ('baseline', epoch, sid)
    for slot, op in enumerate(baseline):
        if op['name'] == 'Posterize':
            replacement = draw(generator('nm-removal-v1', 20261024, epoch, sid, slot, 'replacement'), survivors)
            assert replacement == item['R'][slot], ('replacement', epoch, sid, slot)
            assert item['Z'][slot] == {'name': 'Identity', 'magnitude': 0.0}
            replacements += 1
        else:
            assert item['Z'][slot] == item['R'][slot] == op

out = {'status': 'all_random_streams_recomputed_exactly', 'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'orders': 30, 'weak_views': len(trace), 'baseline_slots': 2*len(trace), 'replacement_slots': replacements, 'seed': 20261024, 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'model_loading_or_forward_or_backward': False, 'image_reads': 0, 'protocol_sha256': hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest(), 'trace_sha256': hashlib.sha256((RUN/'trajectories.jsonl').read_bytes()).hexdigest(), 'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(REVIEW/'random-stream-evidence.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
