"""Pure sampling, class and decision rules. No imports of model/runtime packages."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import json, hashlib, os, math, datetime, time, resource, shutil
P = Path(__file__).resolve().parent
R = _portable_project_path(__file__, 'research-deepfish-baseline-20261031')
ASSET = R/'brackish-asset-check-20261101'
SEED = 20261031
SALT = 'brackish-pilot-20261101-v1'
class ContractError(RuntimeError): pass
PARTIAL = {'2019-03-19_17-07-53to2019-03-19_17-08-34_1':200,
           '2019-03-19_18-01-56to2019-03-19_18-02-13_1':100}
def read(p): return json.loads(Path(p).read_text())
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=canonical(x)+b'\n';tmp=p.with_name(p.name+'.partial')
    with tmp.open('wb') as f: f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    assert p.read_bytes()==b
def cap_indices(n,cap=8):
    assert n>0 and cap>0
    m=min(cap,n)
    return [((2*j+1)*n)//(2*m) for j in range(m)]
def support_index(n,shared,episode):
    assert n>0 and episode in (0,1)
    if shared:
        assert n>=2
        return ((2*episode+1)*n)//4
    return n//2
def video_sort_key(v): return (hashlib.sha256((SALT+'|'+v).encode()).hexdigest(),v)
def covered(v,n): return n>=1 and (v not in PARTIAL or n<=PARTIAL[v])
def mapped_fish(annotations,image_id):
    out=[]
    for a in annotations:
        if a['category_id'] in (1,2):
            out.append({k:a[k] for k in ['bbox','area','iscrowd']})
            out[-1].update(id=len(out),image_id=image_id,category_id=1)
    return out
def choose(dev_p0,dev_a):
    assert all(math.isfinite(x) and 0<=x<=1 for x in (dev_p0,dev_a))
    return 'A' if dev_a>dev_p0 else 'P0'
def decision(dev_p0,dev_a,confirm_p0,confirm_a):
    chosen=choose(dev_p0,dev_a);delta=confirm_a-confirm_p0
    return {'chosen':chosen,'dev_A_minus_P0':dev_a-dev_p0,'confirm_A_minus_P0':delta,
            'selected_minus_P0':delta if chosen=='A' else 0.,
            'harmful_selection_mismatch':chosen=='A' and delta<0,
            'missed_gain':chosen=='P0' and delta>0,
            'equal_confirm_outcome':delta==0}
def tree(p):
    return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file() and not f.is_symlink())
class Guard:
    def __init__(self): self.last=0;self.research_bytes=0
    def check(self,extra=0):
        if (P/'RUN_STARTED.json').exists():
            assert time.time()<read(P/'RUN_STARTED.json')['deadline_epoch'],'Three-hour wall limit'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=12*2**30,'12GiB RSS limit'
        if time.monotonic()-self.last>30:
            self.research_bytes=tree(R);self.last=time.monotonic()
        assert self.research_bytes+tree(P)+extra<=12*2**30-80*2**20,'Project cap / retained80MiB margin'
        assert shutil.disk_usage(P).free>extra+80*2**20,'Filesystem margin'
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
