"""Independent saved-output verification; never imports experiment.py or fits models."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
import resource
import time
import numpy as np

START = time.perf_counter()
ROOT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/causal-resume-20261022')
OUT = Path('/Users/lkc/Documents/Codex/2026-10-22/task-2/independent-review')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(name, root=ROOT):
    return json.loads((root/name).read_text())

comparisons = {}
def close(name, observed, expected, tolerance=1e-10):
    a, b = np.asarray(observed), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    comparisons[name] = error
    assert error <= tolerance, (name, error, tolerance)

p, lock, freeze = read('protocol.json'), read('protocol-lock.json'), read('training-freeze.json')
assert sha(ROOT/'protocol.json') == lock['protocol_sha256'] == freeze['protocol_sha256']
assert sha(ROOT/'experiment.py') == lock['code_sha256'] == freeze['code_sha256']
assert sha(ROOT/'readouts.npz') == freeze['readouts_sha256']
assert sha(ROOT/'source-training.json') == freeze['source_training_sha256']
assert freeze['development_array_decoded'] is False
oldroot = Path(p['source_root'])
for name, digest in p['input_sha256'].items():
    assert sha(oldroot/name) == digest
old = read('protocol.json', oldroot)
sr, dr = old['roles']['source_seen'], old['roles']['development_seen']
assert len(sr) == 540 and len(dr) == 360
order = {(int(c), instance): j for c, vals in p['object_ordinal_by_class'].items()
         for j, instance in enumerate(vals)}
assert all(p['object_ordinal_by_class'][str(c)] == [i for y,i in old['source_objects'] if y == c]
           for c in range(5))
want = set(product(p['factor_levels']['azimuth'], p['factor_levels']['elevation'], p['factor_levels']['lighting']))
assert all(r['camera'] == 0 for r in sr+dr)
for obj in order:
    selected = [r for r in sr if (r['category'],r['instance']) == obj]
    assert len(selected) == 36 and {(r['azimuth'],r['elevation'],r['lighting']) for r in selected} == want

t = list(map(F, p['coefficients']))
C = [list(map(F, row)) for row in p['control_matrix']]
def fraction_weights(kind, k=None):
    ws = []
    for r in sr:
        c, i = r['category'], r['instance']
        az = p['factor_levels']['azimuth'].index(r['azimuth'])
        li = p['factor_levels']['lighting'].index(r['lighting'])
        u = 1 if ((az//3 + li) % 2 == 0) else -1
        v = F(0) if kind == 'U' else (t[(c+k)%5] if kind == 'B' else C[(c+k)%5][order[(c,i)]])
        ws.append((1+F(4,5)*v*u)/540)
    return ws

names = ['U'] + [kind+str(k) for k in range(5) for kind in ['B','S']]
qf = {'U': fraction_weights('U')}
for k in range(5):
    for kind in ['B','S']:
        qf[kind+str(k)] = fraction_weights(kind,k)
def marginal(ws, keys):
    d = defaultdict(F)
    for r,w in zip(sr,ws):
        d[tuple(r[key] for key in keys)] += w
    return d
for name, ws in qf.items():
    assert sum(ws) == 1 and min(ws) > 0
    for keys in [('category',), ('category','instance'), ('category','azimuth'),
                 ('category','elevation'), ('category','lighting'),
                 ('category','instance','azimuth'), ('category','instance','elevation'),
                 ('category','instance','lighting'), ('azimuth','elevation','lighting')]:
        assert marginal(ws,keys) == marginal(qf['U'],keys)
    if name.startswith('S'):
        keys = ('category','azimuth','elevation','lighting')
        assert marginal(ws,keys) == marginal(qf['U'],keys)
    if name != 'U':
        assert 1/sum(w*w for w in ws) == F(4500,11)
for k in range(5):
    assert Counter(qf['B'+str(k)]) == Counter(qf['S'+str(k)])
for kind in ['B','S']:
    assert all(sum(qf[kind+str(k)][i] for k in range(5))/5 == F(1,540) for i in range(540))

states = [s['id'] for s in p['states']]
assert states == ['clean','rotate_05_plus','rotate_05_minus','shearX_005_plus','shearX_005_minus',
                  'shearY_005_plus','shearY_005_minus','translateX_09_plus','translateX_09_minus',
                  'translateY_09_plus','translateY_09_minus']
state_weights = np.asarray([float(F(s['weak_weight'])) for s in p['states']])
close('state_masses', state_weights, [1/3]+[1/15]*10, 0.)
q = {name: np.asarray(list(map(float,ws))) for name,ws in qf.items()}

cache, norm = np.load(oldroot/'source-feature-bank.npz'), np.load(oldroot/'normalizer.npz')
assert not any('final' in name.lower() for name in cache.files)
bank = np.stack([np.sqrt(512.)*(cache[s]*float(norm['scale'])+norm['mean']) for s in states],axis=1)
assert bank.shape == (540,11,512) and np.isfinite(bank).all()
close('source_state_squared_norm', (bank*bank).sum(2), np.full((540,11),512.),1e-6)
z = np.sqrt(512.)*(cache['development_seen']*float(norm['scale'])+norm['mean'])
assert z.shape == (360,512) and np.isfinite(z).all()
sy = np.array([r['category'] for r in sr],dtype=int)
dy = np.array([r['category'] for r in dr],dtype=int)
objects = sorted({(r['category'],r['instance']) for r in dr})
assert len(objects) == 10 and not set(objects)&set(order)
objix = [np.array([i for i,r in enumerate(dr) if (r['category'],r['instance']) == obj]) for obj in objects]
assert all(len(ix) == 36 for ix in objix)
groups = [[i for i,(c,j) in enumerate(objects) if c == y] for y in range(5)]
assert all(len(g) == 2 for g in groups)
assert all({(dr[i]['azimuth'],dr[i]['elevation'],dr[i]['lighting']) for i in ix} == want for ix in objix)

heads = np.load(ROOT/'readouts.npz')
arr = np.load(ROOT/'development-arrays.npz')
record = read('source-training.json')
train = {r['key']:r for r in record['records']}
saved = read('development-summary.json')
result = read('results.json')
close('saved_object_order', objects, saved['objects'],0.)
expected_head_keys = {name+'__'+policy+'__'+field for name in names for policy in ['clean','weak'] for field in ['w','b']}
assert set(heads.files) == expected_head_keys and len(train) == 22
saved_hashes = {name:sha(ROOT/name) for name in ['readouts.npz','source-training.json','training-freeze.json',
                 'development-arrays.npz','development-summary.json','results.json','protocol.json','experiment.py','protocol-lock.json']}
details, objects_ll = {}, {}
source_accuracy, dev_accuracy = {}, {}
for name in names:
    for policy in ['clean','weak']:
        key = name+'__'+policy
        W, b = heads[key+'__w'], heads[key+'__b']
        assert W.shape == (512,5) and b.shape == (5,) and np.isfinite(W).all() and np.isfinite(b).all()
        if policy == 'clean':
            x, labels, wts = bank[:,0,:], sy, q[name]
        else:
            x, labels = bank.reshape(5940,512), np.repeat(sy,11)
            wts = (q[name][:,None]*state_weights).reshape(5940)
        logits = x@W+b
        losses = np.logaddexp.reduce(logits,axis=1)-logits[np.arange(len(labels)),labels]
        value = float(wts@losses + .005*np.square(W).sum())
        prob = np.exp(logits-logits.max(1,keepdims=True))
        prob /= prob.sum(1,keepdims=True)
        prob[np.arange(len(labels)),labels] -= 1.
        res = prob*wts[:,None]
        gradient = max(float(np.abs(x.T@res+.01*W).max()),float(np.abs(res.sum(0)).max()))
        assert gradient <= 1e-5
        for field in ['objective','numpy_objective']:
            close(key+'__source_'+field,value,train[key][field])
        for field in ['gradient_infinity','numpy_gradient_infinity']:
            close(key+'__source_'+field,gradient,train[key][field])
        source_accuracy[key] = float(np.mean((bank[:,0,:]@W+b).argmax(1)==sy))
        close(key+'__source_accuracy',source_accuracy[key],train[key]['clean_source_accuracy'],0.)
        dlogits = z@W+b
        dloss = np.logaddexp.reduce(dlogits,axis=1)-dlogits[np.arange(len(dy)),dy]
        dpred = dlogits.argmax(1)
        for field, calc in [('logits',dlogits),('LL',dloss),('pred',dpred)]:
            close(key+'__development_'+field,calc,arr[key+'__'+field],0. if field == 'pred' else 1e-10)
        oLL = np.array([dloss[ix].mean() for ix in objix])
        oacc = np.array([np.mean(dpred[ix]==dy[ix]) for ix in objix])
        close(key+'__object_LL',oLL,saved['models'][key]['object_LL'])
        close(key+'__object_accuracy',oacc,saved['models'][key]['object_accuracy'],0.)
        close(key+'__mean_LL',float(dloss.mean()),saved['models'][key]['mean_LL'])
        close(key+'__absolute_result',float(dloss.mean()),result['risks'][name][policy])
        dev_accuracy[key] = float(np.mean(dpred==dy))
        close(key+'__mean_accuracy',dev_accuracy[key],saved['models'][key]['accuracy'],0.)
        objects_ll[key] = oLL
        details[key] = {'objective':value,'gradient_infinity':gradient,'source_accuracy':source_accuracy[key],
                        'development_LL':float(dloss.mean()),'development_accuracy':dev_accuracy[key]}

def delta(name):
    return objects_ll[name+'__weak']-objects_ll[name+'__clean']
contrasts = []
for k in range(5):
    B,S = delta('B'+str(k)),delta('S'+str(k))
    contrast = B-S
    contrasts.append(contrast)
    for field, v in [('delta_B',B.mean()),('delta_S',S.mean()),('J_star',contrast.mean()),('J_star_per_object',contrast)]:
        close('configuration'+str(k)+'__'+field,v,result['configs'][k][field])
primary = np.array(contrasts).mean(axis=0)
draws = np.array([tuple(i for pair in pairs for i in pair)
                  for pairs in product(*[list(product(g,repeat=2)) for g in groups])])
assert draws.shape == (1024,10)
def check_summary(name, value, expected):
    byclass = [value[g].mean() for g in groups]
    mean = float(np.mean(byclass))
    sample_means = value[draws].reshape(1024,5,2).mean(2).mean(1)
    interval = np.quantile(sample_means,[.025,.975])
    close(name+'__mean',mean,expected['mean'])
    close(name+'__per_object',value,expected['per_object'])
    close(name+'__interval',interval,expected['object_stability_interval95'])
    return {'mean':mean,'object_stability_interval95':interval.tolist(),'per_object':value.tolist()}
main = check_summary('primary_J_star',primary,result['primary_J_star'])
check_summary('delta_U',delta('U'),result['delta_U'])
check_summary('delta_B_mean',np.mean([delta('B'+str(k)) for k in range(5)],axis=0),result['delta_B_mean'])
check_summary('delta_S_mean',np.mean([delta('S'+str(k)) for k in range(5)],axis=0),result['delta_S_mean'])
capability = {
    'U_source_clean_accuracy':source_accuracy['U__clean'],
    'B_development_clean_accuracy_mean':float(np.mean([dev_accuracy['B'+str(k)+'__clean'] for k in range(5)])),
    'S_development_clean_accuracy_mean':float(np.mean([dev_accuracy['S'+str(k)+'__clean'] for k in range(5)])),
}
for key,val in capability.items():
    close('capability__'+key,val,result['capability'][key],0.)
gates = {'numerical_all22_passed':True,
         'capability_passed':bool(capability['U_source_clean_accuracy']>=.8 and capability['B_development_clean_accuracy_mean']>=.4 and capability['S_development_clean_accuracy_mean']>=.4),
         'J_star_at_least_0_02':bool(main['mean']>=.02),
         'stability_interval_lower_positive':bool(main['object_stability_interval95'][0]>0)}
assert gates == result['gates'] and all(gates.values()) == result['screen_passed']
assert all(sha(ROOT/name) == digest for name,digest in saved_hashes.items())
report = {
    'at_utc':datetime.now(timezone.utc).isoformat(),
    'status':'independent_saved_output_recomputation_passed',
    'training_ran':False,'encoder_calls':False,'old_final_or_sealed_arrays_read':False,
    'project_arrays_read':['prior source-feature-bank selected11 source states and development_seen','prior normalizer','new readouts','new development-arrays'],
    'project_array_paths_not_opened':['evaluation-features.npz','prior final outputs or predictions'],
    'hashes':saved_hashes,'verifier_sha256':sha(__file__),
    'actual_metadata_and_exact_weights_recomputed':True,
    'all_22_source_objectives_and_gradients_recomputed':True,
    'all_development_logits_losses_predictions_recomputed':True,
    'object_means_primary_all5_configurations_and_1024_draws_recomputed':True,
    'maximum_numerical_comparison_error':max(comparisons.values()),
    'maximum_source_gradient_infinity':max(v['gradient_infinity'] for v in details.values()),
    'primary_J_star':main,'gates':gates,'screen_passed':bool(all(gates.values())),
    'comparisons':comparisons,'model_details':details,
    'limits':['Conditional stability of10 consumed development objects; no new-object or training-population confirmation.',
              'Static order/hash record supports freeze before dev decoding, not complete historical activity forensics.',
              'Numeric agreement does not establish original contribution, semantic recognizability, or natural-world causal effects.'],
    'wall_seconds':time.perf_counter()-START,
    'peak_RSS_bytes_Mac':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
}
destination = OUT/'result-verification.json'
with destination.open('x') as f:
    json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    f.write('\n')
print(json.dumps({key:report[key] for key in ['status','maximum_numerical_comparison_error','maximum_source_gradient_infinity',
                                             'primary_J_star','gates','screen_passed','wall_seconds','peak_RSS_bytes_Mac']},ensure_ascii=False,indent=2))
