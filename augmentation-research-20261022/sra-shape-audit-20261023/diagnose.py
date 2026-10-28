"""Bounded CPU-only audit. All outputs use exclusive creation; no training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import time,signal
START=time.perf_counter()
def timeout_handler(signum,frame):
    raise TimeoutError('60-second preregistered wall cap')
signal.signal(signal.SIGALRM,timeout_handler)
signal.alarm(60)
import ast,csv,hashlib,inspect,json,math,resource,textwrap,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mis_score import mis_score

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj):
    with (HERE/name).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def original_score(ori,y):
    if y.shape[-1] != ori.shape[-1]:
        y = torch.zeros(ori.shape).to(y.device).scatter_(-1, y[..., None], 1.0)
    cos_sim = F.cosine_similarity(F.softmax(ori.detach(), dim=-1), y) ** (2 / np.log(y.shape[-1]))
    return cos_sim

def oracle(z,y):
    # Independent NumPy implementation of the published one-hot cosine score.
    p=np.exp(z-z.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    return (p[np.arange(len(y)),y]/np.sqrt(np.square(p).sum(axis=1)))**(2/math.log(z.shape[1]))
def gap(a,b): return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def summary(v,tol):
    v=np.asarray(v)
    return {'count':len(v),'mean_abs':float(np.mean(np.abs(v))),'median_abs':float(np.median(np.abs(v))),'p95_abs':float(np.quantile(np.abs(v),.95)),'max_abs':float(np.max(np.abs(v))),'count_above_tolerance':int(np.sum(np.abs(v)>tol))}

def main():
    cfg=json.loads((HERE/'protocol.json').read_text())
    lock=json.loads((HERE/'protocol-lock.json').read_text())
    assert sha(HERE/'protocol.json')==lock['protocol_sha256']
    code_lock=json.loads((HERE/'code-lock.json').read_text())
    for name,h in code_lock['files'].items(): assert sha(HERE/name)==h
    evidence=json.loads((HERE/'evidence-ledger.json').read_text())
    excerpt=textwrap.dedent('\n'.join(evidence['source_excerpt_for_primary']['lines']))
    src_ast=ast.parse(excerpt).body
    local_ast=ast.parse(textwrap.dedent(inspect.getsource(original_score))).body[0].body[:2]
    assert [ast.dump(n) for n in src_ast]==[ast.dump(n) for n in local_ast], 'source expression mismatch'
    source=ROOT/cfg['primary_input'];manifest=ROOT/cfg['manifest']
    monitored=[source,manifest,HERE/'protocol.json',HERE/'protocol-lock.json',HERE/'code-lock.json',HERE/'evidence-ledger.json',HERE/'mis_score.py',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in monitored}
    with np.load(source,allow_pickle=False) as data:
        z=data['logits'].astype(np.float64);y=data['labels'].astype(np.int64);ids=data['source_ids'].astype(str);pred=data['predictions']
    dev={r['source_id']:r['class_id'] for r in json.loads(manifest.read_text())['rows'] if r['candidate_partition']=='development'}
    assert z.shape==(589,37) and len(set(ids))==589 and set(ids)==set(dev)
    assert all(dev[s]==int(k) for s,k in zip(ids,y)) and np.array_equal(z.argmax(1),pred) and np.isfinite(z).all()
    n,c=z.shape;tol=cfg['tolerances']['float64_atol']
    order=np.argsort(ids,kind='stable')
    order=order[np.random.default_rng(cfg['seeds']['row_order']).permutation(n)]
    transform_rng=np.random.default_rng(cfg['seeds']['within_batch_and_class_permutations'])
    whole=original_score(torch.from_numpy(z),torch.from_numpy(y)).numpy()
    expected=oracle(z,y)
    assert gap(whole,expected)<=tol
    rows=[];groups=[];diagnostic_meta=[];checks={'AST_matches_pinned_source':True,'whole_noncollision_max_gap':gap(whole,expected)}
    control_max={'repair_vs_oracle':0.,'rebatch_vs_oracle':0.,'onehot_vs_oracle':0.,'repaired_row_permutation':0.,'repaired_class_renaming':0.}
    for group,start in enumerate(range(0,n,c)):
        ix=order[start:start+c];bz=z[ix];by=y[ix];b=len(ix)
        tz=torch.from_numpy(bz);ty=torch.from_numpy(by)
        original=original_score(tz,ty).numpy();correct=oracle(bz,by);fixed=mis_score(tz,ty).numpy()
        onehot=original_score(tz,F.one_hot(ty,c).to(tz.dtype)).numpy()
        rebatch=np.concatenate([original_score(tz[j:j+18],ty[j:j+18]).numpy() for j in range(0,b,18)])
        perm=transform_rng.permutation(b);inverse=np.argsort(perm)
        permuted=original_score(tz[perm],ty[perm]).numpy()[inverse]
        fixed_permuted=mis_score(tz[perm],ty[perm]).numpy()[inverse]
        rename=transform_rng.permutation(c)
        rz=np.empty_like(bz);rz[:,rename]=bz
        ry=rename[by]
        renamed=original_score(torch.from_numpy(rz),torch.from_numpy(ry)).numpy()
        fixed_renamed=mis_score(torch.from_numpy(rz),torch.from_numpy(ry)).numpy()
        for name,value in [('repair_vs_oracle',gap(fixed,correct)),('rebatch_vs_oracle',gap(rebatch,correct)),('onehot_vs_oracle',gap(onehot,correct)),('repaired_row_permutation',gap(fixed_permuted,fixed)),('repaired_class_renaming',gap(fixed_renamed,fixed))]:
            control_max[name]=max(control_max[name],value);assert value<=tol,(name,value)
        groups.append({'group':group,'batch_size':b,'collision':b==c,'original_vs_oracle':summary(original-correct,tol),'original_rebatch_delta':summary(original-rebatch,tol),'original_row_permutation_delta':summary(original-permuted,tol),'original_class_renaming_delta':summary(original-renamed,tol),'original_finite':bool(np.isfinite(original).all())})
        diagnostic_meta.append({'group':group,'indices':ix.tolist(),'row_permutation':perm.tolist(),'old_to_new_class':rename.tolist()})
        for j,k in enumerate(ix):
            rows.append({'source_id':str(ids[k]),'label':int(y[k]),'group':group,'batch_size':b,'score_original':float(original[j]),'score_oracle':float(correct[j]),'score_fixed':float(fixed[j]),'score_rebatched':float(rebatch[j]),'score_onehot':float(onehot[j]),'score_row_permuted_restored':float(permuted[j]),'score_class_renamed':float(renamed[j])})
    synthetic=[];rng=np.random.default_rng(cfg['seeds']['synthetic_controls']);invalid_count=0
    for dtype in [torch.float32,torch.float64]:
        atol=cfg['tolerances']['float32_atol'] if dtype==torch.float32 else tol
        for cls in cfg['controls']['classes']:
            for batch in [cls-1,cls,cls+1]:
                tz=torch.tensor(rng.normal(size=(batch,cls)),dtype=dtype)
                ty=torch.tensor(rng.integers(0,cls,size=batch),dtype=torch.int64)
                onehot=F.one_hot(ty,cls).to(dtype)
                soft=torch.tensor(rng.dirichlet(np.ones(cls),size=batch),dtype=dtype)
                corr=oracle(tz.numpy().astype(np.float64),ty.numpy())
                fixed=mis_score(tz,ty).numpy();orig=original_score(tz,ty).numpy()
                encoded=original_score(tz,onehot).numpy()
                assert gap(fixed,corr)<=atol and gap(encoded,corr)<=atol
                assert gap(mis_score(tz,soft).numpy(),original_score(tz,soft).numpy())<=atol
                assert gap(mis_score(tz,onehot).numpy(),fixed)<=atol
                if batch!=cls: assert gap(orig,corr)<=atol
                # The alternative unpowered caller retains its existing formula.
                unpowered=F.cosine_similarity(F.softmax(tz,-1),onehot).numpy()
                assert gap(mis_score(tz,ty,class_exponent=False).numpy(),unpowered)<=atol
                synthetic.append({'dtype':str(dtype),'classes':cls,'batch':batch,'collision':batch==cls,'original_oracle_max_gap':gap(orig,corr),'repair_oracle_max_gap':gap(fixed,corr),'negative_controls_pass':True})
        tz=torch.randn(3,4,generator=torch.Generator().manual_seed(20261025),dtype=dtype);ty=torch.tensor([0,1,2]);soft=F.one_hot(ty,4).to(dtype)
        cases=[(tz,ty.to(dtype)),(tz,ty[:,None]),(tz,torch.tensor([0,1,4])),(tz,-soft),(tz,soft*.5),(tz,soft*float('nan')),(tz*float('nan'),ty)]
        for badz,bady in cases:
            try: mis_score(badz,bady)
            except (ValueError,TypeError): invalid_count+=1
            else: raise AssertionError('Invalid input was silently accepted')
    collisions=[r for r in rows if r['batch_size']==c];remainder=[r for r in rows if r['batch_size']!=c]
    deltas=lambda field:np.array([r['score_original']-r[field] for r in collisions])
    checks.update(control_max);checks['invalid_inputs_rejected']=invalid_count
    checks['input_hashes_unchanged']=all(sha(ROOT/p)==h for p,h in hashes.items())
    assert checks['input_hashes_unchanged']
    result={'status':'completed','at_utc':datetime.now(timezone.utc).isoformat(),'source_kind':'existing_non_SRA_Pets_development_logits; not SRA performance reproduction','N':n,'C':c,'collision_groups':len(collisions)//c,'collision_samples':len(collisions),'noncollision_samples':len(remainder),'primary_original_vs_oracle':summary(deltas('score_oracle'),tol),'same_rows_rebatched':summary(deltas('score_rebatched'),tol),'row_permutation':summary(deltas('score_row_permuted_restored'),tol),'class_renaming':summary(deltas('score_class_renamed'),tol),'remainder_original_vs_oracle':summary([r['score_original']-r['score_oracle'] for r in remainder],tol),'controls':checks,'synthetic_cases':synthetic,'groups':groups,'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},'resources':{'wall_seconds_including_imports':time.perf_counter()-START,'peak_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'device':'cpu','training_runs':0,'model_forward_passes':0,'image_decodes':0,'gpu_calls':0},'input_hashes':hashes,'limitations':['No estimate of downstream classification harm.','Fixed cached logits isolate the score function, not the full train-mode network.','Default CIFAR score batches 128/40 do not collide with 10/100 classes.','No claim of novel scoring theory or original-paper result invalidity.','Existing exploratory development records are not new confirmation data.','No claim about behavior on the authors original PyTorch 1.13.1 runtime.']}
    result['defect_reproduced']=result['primary_original_vs_oracle']['max_abs']>tol
    result['decision']='Keep the locally repaired adapter and regression audit; no full training justified by this edge case.' if result['defect_reproduced'] else 'Do not claim this defect reproduced.'
    with (HERE/'per-photo.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump('group-mapping.json',diagnostic_meta)
    dump('result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','synthetic_cases','input_hashes','versions')},ensure_ascii=False,indent=2))
    signal.alarm(0)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if not (HERE/'failure.json').exists():dump('failure.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'type':type(e).__name__,'error':str(e),'wall_seconds':time.perf_counter()-START})
        raise
