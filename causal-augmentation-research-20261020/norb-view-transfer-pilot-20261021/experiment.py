"""One locked smallNORB frozen-feature training-value pilot. CPU only."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import gzip
import hashlib
import json
import math
import struct
import time
import urllib.request
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
RESEARCH = PROJECT / 'causal-augmentation-research-20261020'
META = RESEARCH / 'official-intervention-data-gate-20261021'
RUNTIME = RESEARCH / 'area-pairing-cpu-20261021/encoder-readiness-20261021/runtime'
WEIGHTS = RUNTIME / 'torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
BASE_URL = 'https://cs.nyu.edu/~yann/data/norb-v1.0-small/'
SEEN = [0, 2, 4, 6, 8, 10]
NOVEL = [18, 20, 22, 24, 26, 28]
ELEV = [0, 4, 8]
LIGHT = [0, 3]
POLICIES = [
    {'id': 'rotate_05', 'family': 'rotate', 'magnitude': 5.0},
    {'id': 'rotate_15', 'family': 'rotate', 'magnitude': 15.0},
    {'id': 'shearX_005', 'family': 'shearX', 'magnitude': 0.05},
    {'id': 'shearX_015', 'family': 'shearX', 'magnitude': 0.15},
    {'id': 'shearY_005', 'family': 'shearY', 'magnitude': 0.05},
    {'id': 'shearY_015', 'family': 'shearY', 'magnitude': 0.15},
    {'id': 'translateX_09', 'family': 'translateX', 'magnitude': 9},
    {'id': 'translateX_18', 'family': 'translateX', 'magnitude': 18},
    {'id': 'translateY_09', 'family': 'translateY', 'magnitude': 9},
    {'id': 'translateY_18', 'family': 'translateY', 'magnitude': 18},
]
ENCODE_SECONDS = 0.0


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def progress(phase, **kwargs):
    rec = {'at_UTC': now(), 'phase': phase, **kwargs}
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def metadata(split):
    pre = 'smallnorb-5x46789' if split == 'training' else 'smallnorb-5x01235'
    result = {}
    for kind in ['cat', 'info']:
        p = META / 'metadata' / f'{pre}x9x18x6x2x96x96-{split}-{kind}.mat.gz'
        raw = gzip.decompress(p.read_bytes())
        magic, ndim = struct.unpack_from('<ii', raw)
        dims = struct.unpack_from('<' + 'i' * max(3, ndim), raw, 8)
        offset = 8 + 4 * max(3, ndim)
        assert magic == 0x1E3D4C54 and dims[0] == 24300
        result[kind] = np.frombuffer(raw, dtype='<i4', offset=offset).copy()
    cats = result['cat']
    infos = result['info'].reshape(24300, 4)
    return [{'row': j, 'category': int(c), 'instance': int(i), 'elevation': int(e),
             'azimuth': int(a), 'lighting': int(l), 'camera': 0}
            for j, (c, (i, e, a, l)) in enumerate(zip(cats, infos))]


def key(r):
    return tuple(r[k] for k in ['category', 'instance', 'elevation', 'azimuth', 'lighting'])


def prepare():
    assert not (OUT / 'protocol.json').exists()
    train, test = metadata('training'), metadata('testing')
    units = sorted({(r['category'], r['instance']) for r in train})
    source, dev = [], []
    for c in range(5):
        ordered = sorted([u for u in units if u[0] == c], key=lambda u: hashlib.sha256(f'norb-view-v1|{u[0]}|{u[1]}'.encode()).hexdigest())
        source.extend(ordered[:3]); dev.extend(ordered[3:])
    roles = {}
    for name, records, objects, angles in [
        ('source_seen', train, source, SEEN), ('source_novel', train, source, NOVEL),
        ('development_seen', train, dev, SEEN), ('development_novel', train, dev, NOVEL),
        ('final_seen', test, sorted({(r['category'], r['instance']) for r in test}), SEEN),
        ('final_novel', test, sorted({(r['category'], r['instance']) for r in test}), NOVEL)]:
        roles[name] = sorted([r for r in records if (r['category'], r['instance']) in objects and r['azimuth'] in angles and r['elevation'] in ELEV and r['lighting'] in LIGHT], key=key)
    assert [len(roles[n]) for n in roles] == [540, 540, 360, 360, 900, 900]
    http = json.loads((META / 'official-http-metadata.json').read_text())
    downloads = {('training' if '-training-' in r['file'] else 'testing'): r for r in http['heads'] if '-dat.' in r['file']}
    weight_hash = sha(WEIGHTS)
    assert weight_hash.startswith('f37072fd'), 'existing official ResNet18 V1 hash prefix changed'
    p = {'at_UTC': now(), 'question': 'Can a source physical-view versus light displacement alignment predict a 2D policy by novel-azimuth interaction on held-out physical objects?',
         'scope': 'Fixed ImageNet ResNet18 representation; actual regularized multinomial readout training. Not end-to-end learning, full AA/RA/TA, a natural-world do effect, or an established novelty claim.',
         'source_objects': source, 'development_objects': dev, 'final_objects': sorted({(r['category'], r['instance']) for r in test}),
         'roles': roles, 'angles_codes_seen': SEEN, 'angles_codes_novel': NOVEL,
         'object_selection': 'fixed SHA256 ordering within each category; official test objects never used to fit/choose predictor or policies',
         'metadata_exposure': 'All public cat/info metadata was already read in the previous source gate. This is not a claim labels were never seen. Final pixel/classification outcomes do not enter score construction, choices or tuning.',
         'policies': POLICIES, 'policy_law': 'For each source photo: identity, plus and minus each weight1/3. Exact enumeration of this stochastic expected readout objective, no Monte Carlo pseudo-replicates.',
         'preprocess': 'camera0 grayscale->RGB repeat, bilinear resize96to224 with no crop; affine/rotate at224 using PIL/torchvision bilinear, fillRGB128; ImageNet normalization; ResNet18 eval avgpool512; per-image L2 normalization.',
         'feature_transform': 'Source-clean mean and scalar per-coordinate RMS only; same transform for all arms/roles.',
         'geometry_metric': {'definition': 'UNCENTERED displacement second moments. Cv=.5 mean outer(Daz)+.5 mean outer(Delevation), source-only allowed views, objects equally weighted; Cl=mean outer(Dlighting). Ca=mean outer(plus-clean and minus-clean). s=cosF(Ca,Cv)-cosF(Ca,Cl).',
                             'steps': 'azimuth20degrees,elevation20degrees; light0to3 categorical contrast, not matched physical dose',
                             'rank_choice': 'highest and lowest s over ten prechosen policies; numerical tie broken by lexical ID; need score gap>1e-6',
                             'causal_limit': 'Cl is specificity comparator, not a deconfounding adjustment; viewing change includes shading/occlusion; Cv includes elevation while novel test changes azimuth only.'},
         'known_source_predictors': ['centered label-kernel alignment of per-photo mean(identity,plus,minus) features, Dao-inspired rather than exact implementation replication', 'mean squared feature displacement', 'mean raw RGB pixel MSE'],
         'source_matched_secondary': 'Among different families, abs CKA difference<=.01 and feature/pixel energy ratios<=1.5, choose pair with largest s gap. No matched pair does not change primary. This is an additional specificity check, not causal isolation.',
         'readout': {'loss': 'mean CE over uniformly weighted source photos and finite policy views +lambda/2||W||F²; unpenalized bias', 'lambda': .01, 'dtype': 'float64', 'initialization': 'zero', 'LBFGS_max_iter': 300, 'max_eval': 450, 'history': 30, 'line_search': 'strong_wolfe', 'all_arms_same_objective_definition': True, 'convergence_gradient_infinity_max': 1e-5},
         'validity_stop': {'baseline_source_accuracy_min': .8, 'baseline_development_seen_accuracy_min': .4, 'reason': 'task-specific measurement readiness, not a universal accuracy or scientific significance threshold', 'all_readouts_finite_and_converged': True, 'no_parameter_search_if_failed': True},
         'primary': {'formula': 'Gamma=(LLlow_finalNovel-LLhigh_finalNovel)-(LLlow_finalSeen-LLhigh_finalSeen), average within each physical object then balanced categories',
                     'name': 'policy_by_azimuth_interaction', 'prediction': 'Gamma>0', 'practical_point_gate_nats': .01, 'paired_object_bootstrap95_lower_positive_gate': True,
                     'interpretation': 'clean cancels. Positive interaction alone is not augmentation benefit; separately report each policy versus clean in all cells.'},
         'uncertainty': '5000 paired category-stratified resamples of25 physical objects (5/category), RNG2026102121; complete seen+novel views travel together. Conditional on source15 objects, fixed pretrained encoder, deterministic augmentation bank and fitted convex readouts. No view or arm pseudo-replicates; no training/source-population uncertainty.',
         'increment_assessment': 'Report all10 policy interactions and descriptive Spearman(s,G), CKA/G, feature-energy/G and pixel-MSE/G. A positive primary alone is not incremental predictor value; source-matched contrast (if any) and known-predictor overlap limit claims. No policy-bank population inference or exhaustive novelty claim.',
         'access_order': 'protocol/code/review locks -> training gzip only -> source features/scores + developmentSeen readiness -> fit all readouts and freeze -> testing gzip -> fixed evaluation once; final-results refuses overwrite',
         'stop_after_result': 'No extra policies, strengths, objects, models or tuning based on any heldout result. Negative/uncertain results retained; no-effect not inferred from failed screen.',
         'resources': {'CPU_threads': 1, 'batch': 32, 'encoder_wall_seconds_cap_per_phase': 480, 'whole_release_compressed_bytes_max': 262778180, 'new_image_download_bytes_max': 262696005, 'GPU': False, 'A100': False, 'ARIS': False, 'automation': False, 'external_models': False, '8A_access': False, 'software_install': False},
         'downloads': downloads, 'weights_path': str(WEIGHTS), 'weights_SHA256': weight_hash,
         'metadata_record_SHA256': sha(META / 'metadata-factorial-check.json'),
         'model_request': 'GPT-6.1-Sol/max preferred; two independent reviewers requested this; actual execution tier receipt unavailable.'}
    save('protocol.json', p)
    lock = {'at_UTC': now(), 'hashes': {name: sha(OUT / name) for name in ['protocol.json', 'experiment.py', 'prior-review.json', 'design-review.json']}, 'new_image_pixels_read': 0, 'prior_gate': 'conditional one-pilot go; not originality proven', 'root_review_disposition': 'Adopted second-moment definition, equal axis weights, absolute-gain reporting and known KTA/energy controls. Ten policies fixed before pixels (five families x two strengths), root implementation choice beyond initial six-policy discussion.'}
    save('protocol-lock.json', lock)
    progress('protocol_locked', source_objects=len(source), development_objects=len(dev), final_objects=25, policies=len(POLICIES))


def check_lock():
    lock = read('protocol-lock.json')
    for name, digest in lock['hashes'].items():
        assert sha(OUT / name) == digest, name


def download(split):
    check_lock()
    if split == 'testing':
        freeze = read('training-freeze.json')
        assert freeze['validity_passed'] and freeze['new_final_pixels_read'] == 0
    row = read('protocol.json')['downloads'][split]
    data_dir = OUT / 'data'; data_dir.mkdir(exist_ok=True)
    path = data_dir / row['file']; assert not path.exists()
    req = urllib.request.Request(row['url'], headers={'User-Agent': 'NORBLockedResearchPilot/1.0'})
    start = time.perf_counter(); n = 0; h = hashlib.sha256(); next_log = 8 * 1024 * 1024
    with urllib.request.urlopen(req, timeout=30) as r, path.open('xb') as f:
        assert r.status == 200 and int(r.headers['Content-Length']) == row['content_length']
        while True:
            chunk = r.read(min(1024 * 1024, row['content_length'] - n + 1))
            if not chunk:
                break
            n += len(chunk); assert n <= row['content_length']
            f.write(chunk); h.update(chunk)
            if n >= next_log:
                progress('download', split=split, bytes=n, expected=row['content_length'])
                next_log += 8 * 1024 * 1024
            assert time.perf_counter() - start < 900, 'download wall cap'
    assert n == row['content_length']
    save(f'{split}-download.json', {'at_UTC': now(), 'url': row['url'], 'bytes': n, 'sha256': h.hexdigest(), 'wall_seconds': time.perf_counter()-start, 'image_pixels_decoded': False, 'published_SHA_comparison': 'not available; recorded local TLS-source fingerprint, length checked; gzip CRC checked at later extraction'})
    progress('download_complete', split=split, bytes=n, seconds=time.perf_counter()-start)


def extract(split, role_names):
    p = read('protocol.json'); path = OUT / 'data' / p['downloads'][split]['file']
    receipt = read(f'{split}-download.json'); assert path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256']
    targets = {}
    arrays = {}
    for role in role_names:
        arrays[role] = np.empty((len(p['roles'][role]), 96, 96), dtype=np.uint8)
        for j, r in enumerate(p['roles'][role]):
            targets.setdefault(r['row'], []).append((role, j))
    count = 0
    with gzip.open(path, 'rb') as f:
        header = f.read(24)
        assert struct.unpack('<6i', header) == (0x1E3D4C55, 4, 24300, 2, 96, 96)
        for row in range(24300):
            raw = f.read(18432); assert len(raw) == 18432
            if row in targets:
                image = np.frombuffer(raw, dtype=np.uint8, count=9216).reshape(96, 96)
                for role, j in targets[row]:
                    arrays[role][j] = image
                    count += 1
        assert f.read(1) == b'', 'extra pixels or bad CRC'
    assert count == sum(len(x) for x in arrays.values())
    progress('extracted', split=split, roles=role_names, images=count, camera=0, gzip_CRC_and_dimensions_passed=True)
    return arrays


def get_encoder():
    import torch
    import torchvision
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert sha(WEIGHTS) == read('protocol.json')['weights_SHA256']
    model = torchvision.models.resnet18(weights=None)
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True); model.fc = torch.nn.Identity()
    model.eval().to('cpu')
    return model, {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__, 'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'torch_interop': torch.get_num_interop_threads()}


def transform(img, policy, sign):
    from torchvision.transforms import functional as TF, InterpolationMode
    if policy is None:
        return img
    mag = float(policy['magnitude']) * sign
    family = policy['family']
    if family == 'rotate':
        return TF.rotate(img, mag, interpolation=InterpolationMode.BILINEAR, fill=[128, 128, 128])
    translate = [0, 0]; shear = [0., 0.]
    if family == 'translateX': translate[0] = int(mag)
    if family == 'translateY': translate[1] = int(mag)
    if family == 'shearX': shear[0] = math.degrees(math.atan(mag))
    if family == 'shearY': shear[1] = math.degrees(math.atan(mag))
    return TF.affine(img, angle=0., translate=translate, scale=1., shear=shear, interpolation=InterpolationMode.BILINEAR, fill=[128, 128, 128])


def encode(model, images, label, policy=None, sign=0):
    import torch
    from PIL import Image
    global ENCODE_SECONDS
    start = time.perf_counter(); features = []; pixel_mse = []
    mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
    for offset in range(0, len(images), 32):
        batch = []
        for gray in images[offset:offset+32]:
            original = Image.fromarray(gray).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            altered = transform(original, policy, sign)
            a = np.asarray(altered, dtype=np.float32) / 255.
            if policy is not None:
                pixel_mse.append(float(np.mean((a - np.asarray(original, dtype=np.float32) / 255.)**2)))
            batch.append(np.transpose(a, (2, 0, 1)))
        t = (torch.from_numpy(np.stack(batch)) - mean) / std
        with torch.inference_mode():
            v = model(t).numpy().astype(np.float64)
        assert v.shape[1] == 512 and np.isfinite(v).all()
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        features.append(v)
        elapsed = ENCODE_SECONDS + time.perf_counter() - start
        assert elapsed < 480, 'phase encoder time cap; stop without changing recipe'
    ENCODE_SECONDS += time.perf_counter() - start
    result = np.concatenate(features)
    progress('encoded', label=label, images=len(images), phase_encoder_seconds=ENCODE_SECONDS)
    return result, (float(np.mean(pixel_mse)) if pixel_mse else 0.)


def cka(x, labels):
    x = x - x.mean(0)
    y = np.eye(5)[labels]; y -= y.mean(0)
    numerator = np.sum((x.T @ y)**2)
    denom = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(numerator / denom)


def moment(d):
    return d.T @ d / len(d)


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    assert denom > 1e-12
    return float(np.sum(a * b) / denom)


def source_geometry(z, records):
    lookup = {key(r): j for j, r in enumerate(records)}
    az, el, light = [], [], []
    for r in records:
        c, i, e, a, l = key(r); j = lookup[key(r)]
        if a + 2 in SEEN: az.append(z[lookup[(c, i, e, a+2, l)]] - z[j])
        if e + 4 in ELEV: el.append(z[lookup[(c, i, e+4, a, l)]] - z[j])
        if l == 0: light.append(z[lookup[(c, i, e, a, 3)]] - z[j])
    assert [len(az), len(el), len(light)] == [450, 360, 270]
    return .5 * moment(np.stack(az)) + .5 * moment(np.stack(el)), moment(np.stack(light))


def losses(z, w, b, labels):
    logits = z @ w + b
    shift = logits.max(1)
    ll = np.log(np.exp(logits - shift[:, None]).sum(1)) + shift - logits[np.arange(len(labels)), labels]
    return logits, ll


def fit_model(x, labels):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(x))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    w = torch.zeros((512, 5), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1., max_iter=300, max_eval=450, history_size=30, tolerance_grad=1e-8, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    calls = 0
    def closure():
        nonlocal calls
        opt.zero_grad()
        objective = torch.nn.functional.cross_entropy(x @ w + b, y) + .005 * w.square().sum()
        objective.backward(); calls += 1
        assert torch.isfinite(objective) and torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()
        return objective
    opt.step(closure)
    value = float(closure().detach())
    grad = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    assert grad <= 1e-5, f'nonconverged fixed objective: {grad}'
    return w.detach().numpy(), b.detach().numpy(), {'objective': value, 'gradient_inf': grad, 'closure_calls': calls, 'converged': True}


def fit():
    check_lock(); assert not (OUT / 'training-freeze.json').exists()
    start = time.perf_counter(); p = read('protocol.json')
    imgs = extract('training', ['source_seen', 'development_seen'])
    model, versions = get_encoder()
    f0, _ = encode(model, imgs['source_seen'], 'source-clean')
    fd, _ = encode(model, imgs['development_seen'], 'development-seen-clean')
    mu = f0.mean(0); scale = float(np.sqrt(np.mean((f0-mu)**2))); assert scale > 1e-8
    z0 = (f0-mu)/scale; zd = (fd-mu)/scale
    labels = np.array([r['category'] for r in p['roles']['source_seen']])
    dev_labels = np.array([r['category'] for r in p['roles']['development_seen']])
    cv, cl = source_geometry(z0, p['roles']['source_seen'])
    bank = {'clean': z0, 'development_seen': zd}
    metrics = []
    for policy in POLICIES:
        plus, pm = encode(model, imgs['source_seen'], policy['id']+'+', policy, 1)
        minus, mm = encode(model, imgs['source_seen'], policy['id']+'-', policy, -1)
        zp, zm = (plus-mu)/scale, (minus-mu)/scale
        bank[policy['id']+'_plus'] = zp; bank[policy['id']+'_minus'] = zm
        ca = moment(np.concatenate([zp-z0, zm-z0]))
        metrics.append({'id': policy['id'], 'family': policy['family'], 'score': cosine(ca, cv)-cosine(ca, cl),
                        'view_alignment': cosine(ca, cv), 'light_alignment': cosine(ca, cl),
                        'feature_energy': float(np.trace(ca)), 'pixel_MSE': (pm+mm)/2,
                        'label_kernel_alignment': cka((z0+zp+zm)/3, labels)})
    order = sorted(metrics, key=lambda m: (-m['score'], m['id']))
    high, low = order[0]['id'], sorted(metrics, key=lambda m: (m['score'], m['id']))[0]['id']
    gap = order[0]['score']-order[-1]['score']
    matches = []
    for a in metrics:
        for b in metrics:
            if a['score'] <= b['score'] or a['family'] == b['family']:
                continue
            ratio_f = max(a['feature_energy'], b['feature_energy'])/min(a['feature_energy'], b['feature_energy'])
            ratio_p = max(a['pixel_MSE'], b['pixel_MSE'])/min(a['pixel_MSE'], b['pixel_MSE'])
            if abs(a['label_kernel_alignment']-b['label_kernel_alignment']) <= .01 and ratio_f <= 1.5 and ratio_p <= 1.5:
                matches.append({'high': a['id'], 'low': b['id'], 'score_gap': a['score']-b['score'], 'CKA_abs_difference': abs(a['label_kernel_alignment']-b['label_kernel_alignment']), 'feature_energy_ratio': ratio_f, 'pixel_MSE_ratio': ratio_p})
    matches.sort(key=lambda q: (-q['score_gap'], q['high'], q['low']))
    prediction = {'at_UTC': now(), 'metrics': metrics, 'clean_label_kernel_alignment': cka(z0, labels), 'high': high, 'low': low, 'score_gap': gap, 'matched_secondary': matches[0] if matches else None,
                  'uses_final_pixels_or_predictions': False, 'uses_development_outcomes': False, 'definition': p['geometry_metric'], 'new_augmentation_predictor_originality_established': False}
    save('source-prediction.json', prediction)
    assert gap > 1e-6, 'no numerical geometry contrast; stop before final data'
    np.savez_compressed(OUT / 'source-feature-bank.npz', **bank)
    np.savez(OUT / 'normalizer.npz', mean=mu, scale=np.array(scale))
    fitted = {}; fit_info = {}
    for arm in ['clean'] + [x['id'] for x in POLICIES]:
        x = z0 if arm == 'clean' else np.concatenate([z0, bank[arm+'_plus'], bank[arm+'_minus']])
        y = labels if arm == 'clean' else np.tile(labels, 3)
        w, b, details = fit_model(x, y)
        fitted[arm+'_W'] = w; fitted[arm+'_b'] = b
        pred0, ll0 = losses(z0, w, b, labels)
        predd, lld = losses(zd, w, b, dev_labels)
        fit_info[arm] = {**details, 'source_clean_LL': float(ll0.mean()), 'source_clean_accuracy': float(np.mean(pred0.argmax(1)==labels)),
                         'development_seen_LL': float(lld.mean()), 'development_seen_accuracy': float(np.mean(predd.argmax(1)==dev_labels))}
        progress('readout_fitted', arm=arm, **details)
    np.savez(OUT / 'readouts.npz', **fitted)
    valid = fit_info['clean']['source_clean_accuracy'] >= .8 and fit_info['clean']['development_seen_accuracy'] >= .4
    results = {'at_UTC': now(), 'models': fit_info, 'baseline_measurement_validity_passed': valid,
               'versions': versions, 'encoder_seconds': ENCODE_SECONDS, 'wall_seconds': time.perf_counter()-start, 'source_physical_objects': 15, 'development_physical_objects': 10, 'final_pixels_read': 0}
    save('training-results.json', results)
    save('training-freeze.json', {'at_UTC': now(), 'validity_passed': valid, 'new_final_pixels_read': 0,
         'protocol_lock_SHA256': sha(OUT/'protocol-lock.json'), 'files': {n: sha(OUT/n) for n in ['source-prediction.json','source-feature-bank.npz','normalizer.npz','readouts.npz','training-results.json']},
         'rank_or_recipe_changed_after_development_results': False})
    progress('training_frozen', validity=valid, high=high, low=low, encoder_seconds=ENCODE_SECONDS, wall_seconds=time.perf_counter()-start)


def rank(v):
    order = np.argsort(v, kind='stable'); result = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i+1
        while j < len(v) and v[order[j]] == v[order[i]]: j += 1
        result[order[i:j]] = (i+j-1)/2; i = j
    return result


def spearman(a, b):
    a, b = rank(np.asarray(a)), rank(np.asarray(b))
    a -= a.mean(); b -= b.mean(); denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else None


def evaluate():
    check_lock(); assert not (OUT/'final-results.json').exists(), 'no repeat final evaluation'
    freeze = read('training-freeze.json'); assert freeze['validity_passed']
    for n,h in freeze['files'].items(): assert sha(OUT/n) == h
    start = time.perf_counter(); p = read('protocol.json'); prediction = read('source-prediction.json')
    rest = extract('training', ['source_novel', 'development_novel'])
    rest.update(extract('testing', ['final_seen', 'final_novel']))
    model, versions = get_encoder()
    normalizer = np.load(OUT/'normalizer.npz'); mu, scale = normalizer['mean'], float(normalizer['scale'])
    bank = np.load(OUT/'source-feature-bank.npz')
    feature = {'source_seen': bank['clean'], 'development_seen': bank['development_seen']}
    for role, images in rest.items():
        f, _ = encode(model, images, role); feature[role] = (f-mu)/scale
    np.savez_compressed(OUT/'evaluation-features.npz', **feature)
    readouts = np.load(OUT/'readouts.npz')
    arms = ['clean'] + [q['id'] for q in POLICIES]
    aggregate = {}; saved = {}; object_loss = {}
    for role, z in feature.items():
        records = p['roles'][role]; y = np.array([r['category'] for r in records])
        objects = sorted({(r['category'],r['instance']) for r in records})
        array = np.empty((len(objects),len(arms)))
        aggregate[role] = {}
        for j, arm in enumerate(arms):
            logits, ll = losses(z, readouts[arm+'_W'], readouts[arm+'_b'], y)
            saved[role+'__'+arm+'__logits'] = logits
            aggregate[role][arm] = {'LL': float(ll.mean()), 'accuracy': float(np.mean(logits.argmax(1)==y))}
            for k, ob in enumerate(objects):
                mask = np.array([(r['category'],r['instance'])==ob for r in records]); assert mask.sum()==36
                array[k,j] = ll[mask].mean()
        object_loss[role] = array
    np.savez_compressed(OUT/'evaluation-predictions.npz', **saved)
    units = sorted({(r['category'],r['instance']) for r in p['roles']['final_seen']})
    rng = np.random.default_rng(2026102121)
    idx = np.concatenate([rng.choice([i for i,u in enumerate(units) if u[0]==c], size=(5000,5), replace=True) for c in range(5)], axis=1)
    def summary(v):
        ci = np.quantile(v[idx].mean(1),[.025,.975])
        return {'mean': float(v.mean()), 'paired_category_stratified_object_bootstrap95': ci.tolist()}
    by_policy = {}
    s = object_loss['final_seen']; u = object_loss['final_novel']
    for j, arm in enumerate(arms[1:],1):
        gS = s[:,0]-s[:,j]; gU = u[:,0]-u[:,j]
        by_policy[arm] = {'gain_seen': summary(gS), 'gain_novel': summary(gU), 'extra_novel_gain_interaction': summary(gU-gS)}
    hi = arms.index(prediction['high']); lo = arms.index(prediction['low'])
    gamma = (u[:,lo]-u[:,hi])-(s[:,lo]-s[:,hi]); primary = summary(gamma)
    gate = primary['mean']>=.01 and primary['paired_category_stratified_object_bootstrap95'][0]>0
    matched = prediction['matched_secondary']; matched_result = None
    if matched:
        mh, ml = arms.index(matched['high']), arms.index(matched['low'])
        matched_result = {**matched, 'interaction': summary((u[:,ml]-u[:,mh])-(s[:,ml]-s[:,mh]))}
    g = [by_policy[q['id']]['extra_novel_gain_interaction']['mean'] for q in prediction['metrics']]
    correlations = {k: spearman([q[k] for q in prediction['metrics']],g) for k in ['score','label_kernel_alignment','feature_energy','pixel_MSE']}
    result = {'at_UTC': now(), 'status': 'one_fixed_final_evaluation_complete', 'risk_table': aggregate,
              'high_policy': prediction['high'], 'low_policy': prediction['low'], 'primary_policy_by_azimuth_interaction': primary,
              'predeclared_interaction_screen_passed': gate, 'high_novel_gain': by_policy[prediction['high']]['gain_novel'],
              'high_positive_novel_gain_point_observed': by_policy[prediction['high']]['gain_novel']['mean']>0,
              'all_policy_final_object_gains': by_policy, 'source_matched_secondary': matched_result,
              'descriptive_policy_bank_Spearman_not_independent_policy_population_inference': correlations,
              'final_physical_objects': 25, 'final_original_views': 1800, 'bootstrap_unit': 'paired physical object,5/category; fixed source15/encoder/bank/convex readouts',
              'claim': 'Only a conditional policy-by-azimuth interaction is tested. Gamma alone cannot mean positive augmentation benefit, end-to-end training value, identified physical mechanism or novel contribution.',
              'novelty_or_contribution_established': False, 'model_or_parameter_retuning_after_final': False,
              'code_SHA256': sha(OUT/'experiment.py'), 'protocol_SHA256': sha(OUT/'protocol.json'), 'training_freeze_SHA256': sha(OUT/'training-freeze.json'),
              'predictions_SHA256': sha(OUT/'evaluation-predictions.npz'), 'encoder_seconds': ENCODE_SECONDS, 'wall_seconds': time.perf_counter()-start, 'versions': versions}
    save('final-results.json', result)
    np.savez(OUT/'per-object-losses.npz', **object_loss)
    save('test-consumption.json', {'at_UTC': now(), 'status': 'consumed_for_locked_view_transfer_question', 'physical_objects': units,
         'selected_final_views': 1800, 'future_use': 'These25 physical objects are not untouched object-level confirmation for a new question after this evaluation. Other views do not create new independent objects.'})
    progress('final_evaluation_complete', primary=primary, screen=gate, high_gain_novel=result['high_novel_gain'], correlations=correlations, encoder_seconds=ENCODE_SECONDS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare','download_training','fit','download_testing','evaluate'])
    arg = parser.parse_args().phase
    if arg == 'prepare': prepare()
    elif arg.startswith('download_'): download(arg.split('_',1)[1])
    elif arg == 'fit': fit()
    else: evaluate()
