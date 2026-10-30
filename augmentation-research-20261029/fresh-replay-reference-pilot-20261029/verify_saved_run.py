"""Post-run integrity and independent metric recalculation; no model execution."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run-001'
DATA=ROOT.parents[1]/'data/cifar100-official-20261029'
f=json.loads((ROOT/'protocol-freeze.json').read_text())
checks={'frozen_protocol_and_script_unchanged':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in f['files'].items())}
records=[json.loads(x) for x in (RUN/'training-losses.jsonl').read_text().splitlines()]
assert len(records)==760
phase_stats={}
for phase,n in [('task1',600),('task2_prefix',160)]:
    selected=[x for x in records if x['phase']==phase]
    assert len(selected)==n and [x['step'] for x in selected]==list(range(1,n+1))
    phase_stats[phase]={'count':n,'last32_means':{k:float(np.mean([x[k] for x in selected[-32:]])) for k in selected[-1] if k not in ['phase','step']}}
    assert all(np.isfinite(v) for x in selected for k,v in x.items() if k!='phase')
seq=np.load(RUN/'data-and-sequences.npz',allow_pickle=False)
log=np.load(RUN/'branchpoint-dev.npz',allow_pickle=False)
raw=np.memmap(DATA/'train-only/train.bin',dtype=np.uint8,mode='r',shape=(50000,3074))
checks['fit_dev_disjoint']=not np.intersect1d(np.r_[seq['old_fit_rows'],seq['new_fit_rows']],np.r_[seq['old_dev_rows'],seq['new_dev_rows']]).size
checks['buffer_dev_disjoint']=not np.intersect1d(seq['buffer_rows'],np.r_[seq['old_dev_rows'],seq['new_dev_rows']]).size
metrics={}
for tag in ['old','new']:
    rows=seq[tag+'_dev_rows'];y=raw[rows,1].astype(int);h=log[tag+'_logits'].astype(float)
    m=h.max(1);lse=m+np.log(np.exp(h-m[:,None]).sum(1));nll=lse-h[np.arange(len(y)),y];pred=h.argmax(1)
    err=float(np.max(np.abs(nll-log[tag+'_nll'])))
    assert err<1e-12
    metrics[tag]={'nll':float(nll.mean()),'accuracy':float(np.mean(pred==y)),'saved_nll_max_abs_error':err,
                  'predicted_class_counts':np.bincount(pred,minlength=10).tolist(),'true_class_counts':np.bincount(y,minlength=10).tolist(),
                  'per_class_accuracy':{int(c):float(np.mean(pred[y==c]==c)) for c in np.unique(y)}}
checks['gate_A_not_run']=not (RUN/'gate-A.json').exists()
checks['branches_not_run']=not list(RUN.glob('branch-*-outputs.npz'))
assert all(checks.values())
result={'checks':checks,'all_passed':True,'phase_statistics':phase_stats,'independent_metrics':metrics,'new_optimizer_steps':0,'new_model_forwards':0}
with (RUN/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
