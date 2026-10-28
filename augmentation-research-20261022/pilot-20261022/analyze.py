"""Paired fixed-model development analysis; no fitting, no reserved data."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
CELLS=('front-weak','front-ra','back-weak','back-ra')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    cfg=json.loads((ROOT/'protocol.json').read_text());epoch=cfg['epochs']
    lock=json.loads((ROOT/'protocol-lock.json').read_text())
    assert lock['protocol_sha256']==sha(ROOT/'protocol.json')
    assert lock['code_sha256']['analyze.py']==sha(ROOT/'analyze.py')
    manifest=json.loads((ROOT/'fit-manifest.json').read_text())
    assert cfg['fit_manifest_sha256']==sha(ROOT/'fit-manifest.json')
    expected=[r for r in manifest['rows'] if r['candidate_partition']=='development']
    expected_ids=np.array([r['source_id'] for r in expected]);expected_labels=np.array([r['class_id'] for r in expected])
    assert len(set(expected_ids))==len(expected_ids) and set(expected_labels)==set(range(37))
    done=json.loads((ROOT/'results/completed.json').read_text())
    assert done['status']=='all_four_cells_completed'
    assert done['protocol_sha256']==sha(ROOT/'protocol.json')
    out=ROOT/'analysis';out.mkdir(exist_ok=False)
    arrays={};metrics={};errors=[];nlls=[];ids=None;labels=None;inputs={}
    for cell in CELLS:
        path=ROOT/f'results/{cell}/development-epoch{epoch:03d}.npz'
        with np.load(path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        z=a['logits'].astype(np.float64);y=a['labels'];source=a['source_ids']
        assert z.dtype==np.float64 and np.isfinite(z).all() and z.shape==(len(y),37)
        if ids is None:ids=source;labels=y
        assert np.array_equal(ids,source) and np.array_equal(labels,y)
        assert np.array_equal(source,expected_ids) and np.array_equal(y,expected_labels)
        pred=np.argmax(z,axis=1);err=(pred!=y).astype(np.int64)
        assert np.array_equal(pred,a['predictions'])
        centered=z-z.max(axis=1,keepdims=True)
        nll=np.log(np.exp(centered).sum(1))-centered[np.arange(len(y)),y]
        metrics[cell]={'N':len(y),'errors':int(err.sum()),'error':float(err.mean()),
            'accuracy':float(1-err.mean()),'NLL':float(nll.mean()),
            'balanced_error':float(np.mean([err[y==c].mean() for c in range(37)]))}
        saved=json.loads((ROOT/f'results/{cell}/summary.json').read_text())['final_development']
        assert abs(metrics[cell]['error']-saved['error'])<1e-12
        assert abs(metrics[cell]['NLL']-saved['NLL'])<1e-10
        errors.append(err);nlls.append(nll);inputs[cell]=sha(path)
    e=np.stack(errors,axis=1);l=np.stack(nlls,axis=1)
    contrast=np.array([1,-1,-1,1]);d=e@contrast
    I=float(d.mean());Gf=float((e[:,0]-e[:,1]).mean());Gb=float((e[:,2]-e[:,3]).mean())
    bootstrap_cfg=cfg['bootstrap'];rng=np.random.default_rng(bootstrap_cfg['seed'])
    b=np.zeros(bootstrap_cfg['replicates'],dtype=np.float64)
    for c in range(37):
        group=d[labels==c]
        draws=rng.integers(0,len(group),size=(len(b),len(group)))
        b+=group[draws].sum(axis=1)/len(d)
    ci=np.quantile(b,[.025,.975],method='linear')
    np.savez_compressed(out/'paired-errors-and-bootstrap.npz',source_ids=ids,labels=labels,
        errors=e,NLL=l,interaction_contributions=d,bootstrap_interactions=b)
    with (out/'per-image.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['source_id','class_id',*[x+'_error' for x in CELLS],'I_contribution'])
        writer.writerows([[sid,int(y),*map(int,err),int(di)] for sid,y,err,di in zip(ids,labels,e,d)])
    with (out/'per-class.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class_id','N',*CELLS,'I'])
        for c in range(37):
            selected=labels==c;means=e[selected].mean(0)
            writer.writerow([c,int(selected.sum()),*map(float,means),float(d[selected].mean())])
    summary={'status':'single_seed_development_only','seed':cfg['seed'],'endpoint_epoch':epoch,
        'metric':'source-photo micro classification error on fixed clean development pool',
        'cells':metrics,'G_front':Gf,'G_back':Gb,'interaction_I':I,
        'front_minus_back_weak_error':float((e[:,0]-e[:,2]).mean()),
        'front_minus_back_RA_error':float((e[:,1]-e[:,3]).mean()),
        'I_NLL_secondary':float((l@contrast).mean()),
        'paired_stratified_photo_bootstrap_95percent_interval':ci.tolist(),
        'bootstrap':bootstrap_cfg,'bootstrap_array_sha256':sha(out/'paired-errors-and-bootstrap.npz'),
        'uncertainty_scope':'Conditional sensitivity to source-photo sampling at fixed observed breed counts and four fixed trained models. No training-seed variability, animal-identity guarantee, population CI or confirmatory inference.',
        'engineering_gate_I_min':cfg['engineering_gate_I_min'],
        'engineering_gate_passed':bool(I>=cfg['engineering_gate_I_min']),
        'gate_is_significance_power_or_equivalence':False,
        'RA_improves_front':bool(Gf>0),'RA_improves_back':bool(Gb>0),
        'input_npz_sha256':inputs,'reserved_confirmation_read':False,'extra_seeds':0}
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
