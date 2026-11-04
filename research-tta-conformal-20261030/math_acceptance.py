import hashlib
import json
from pathlib import Path
import time
import numpy as np
from conformal_core import *

start = time.monotonic()
rng = np.random.default_rng(20261030)
n, classes, nc = 360, 20, 200
z = rng.normal(size=(n, classes))
p, q = softmax(z), softmax(rng.normal(size=(n, classes)))
y = rng.integers(classes, size=n)
u = rng.random(n)
checks = []
methods = [('aps', {}), ('raps', {'lam': .01, 'kreg': 5}), ('saps', {'lam': .1}), ('rankonly', {})]
def add(name): checks.append({'name': name, 'passed': True})
def outputs(a, method, kw):
    return {k: calibrate_predict(o, v, u, y, nc, .1, method=method, **kw)[0] for k, (o, v) in a.items()}
for method, kw in methods:
    a = outputs(arms(p, p), method, kw)
    assert all(np.array_equal(a['00'], x) for x in a.values())
add('p=q: all four arms identical for every score')
for method, kw in methods:
    a = outputs(arms(p, softmax(z, 2)), method, kw)
    assert np.array_equal(a['00'], a['10']) and np.array_equal(a['01'], a['11'])
add('pure temperature: rank arms collapse')
perm = rng.permutation(classes)
for method, kw in methods:
    a = outputs(arms(p, p[:, perm]), method, kw)
    assert np.array_equal(a['00'], a['01']) and np.array_equal(a['10'], a['11'])
add('pure class permutation: spectrum arms collapse')
for method, kw in methods:
    a = arms(p, q)
    for b in ('0','1'):
        o1,v = a['0'+b]; o2,_ = a['1'+b]
        s = scores_sorted(v, u, method=method, **kw)
        threshold = float(np.median(s))
        s1,s2 = label_scores(o1,s),label_scores(o2,s)
        assert np.array_equal((s1<=threshold).sum(1),(s2<=threshold).sum(1))
add('same spectrum, threshold, U: samplewise set size identical')
base = np.linspace(2, -2, classes)
orders = np.array([rng.permutation(classes) for _ in range(n)])
for method, kw in methods[:2]:
    sets = []
    for t in (.5,.75,1,1.5,2,3,5):
        v = np.tile(softmax(base[None,:],t),(n,1))
        sets.append(calibrate_predict(orders,v,u,y,nc,.1,method=method,**kw)[0])
    assert all(np.array_equal(sets[0],s) for s in sets)
add('identical strictly positive spectrum over all samples: APS/RAPS invariant under temperature')
o,v=rank_spectrum(p)
s=label_scores(o,scores_sorted(v,u,method='rankonly'))
inverse=np.argsort(o,axis=1)
assert np.array_equal(s,inverse+u[:,None])
a=outputs(arms(p,q),'rankonly',{})
assert np.array_equal(a['00'],a['01']) and np.array_equal(a['10'],a['11'])
add('rankonly equals rank-1+U and is spectrum invariant')
assert quantile(np.arange(1,10),.2)==8.0
assert np.isinf(quantile(np.arange(1,10),.01))
add('finite sample ceil order statistic without interpolation; k>n gives infinity')
ties=np.tile(np.array([.4,.3,.3,0]),(3,1))
o,v=rank_spectrum(ties)
assert np.array_equal(o[0],[0,1,2,3])
donor=np.array([[3,2,1,0]]*3)
sorted_s=scores_sorted(v,np.array([0,.5,.999]))
assert np.array_equal(label_scores(donor,sorted_s)[:,3],sorted_s[:,0])
assert ((label_scores(donor,sorted_s)<=-1).sum(1)==0).all()
assert ((label_scores(donor,sorted_s)<=np.inf).sum(1)==4).all()
add('stable class ID ties, donor rank retained, empty sets allowed, infinite threshold full set')
tiny=softmax(np.array([[0,-100,-1000,-10000.]]))
ss=scores_sorted(tiny,np.array([.2]))
assert ss.dtype==np.float64 and np.isfinite(ss).all()
add('float64 tail underflow and cumulative saturation diagnosed without clipping')
report={'status':'PASS','elapsed_seconds':time.monotonic()-start,'checks':checks,
        'rankonly_definition':'rank-1+U, randomized uniform rank baseline',
        'files_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['conformal_core.py','math_acceptance.py']}}
Path('math-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
