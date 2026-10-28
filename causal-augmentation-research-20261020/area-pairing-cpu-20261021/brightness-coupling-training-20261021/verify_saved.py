"""Read-only numerical/file verification after the single locked evaluation."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import datetime,hashlib,importlib.util,json
import numpy as np
import torch,torchvision
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('locked_experiment',ROOT/'experiment.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

def ll(z,y):
    z=z.astype(np.float64);v=z-z.max(1,keepdims=True)
    return np.log(np.exp(v).sum(1))-v[np.arange(len(y)),y]

def main():
    E.configure();E.checklock();p=E.load('protocol.json');tr=E.load('training-results.json');out=E.load('final-results.json');freeze=E.load('training-freeze.json')
    assert E.sha(ROOT/'training-results.json')==freeze['training_results_sha256']
    train=np.load(ROOT/'training-predictions.npz');test=np.load(ROOT/'heldout-predictions.npz');errs=[];trace_count=0;CPU=[]
    _,ims,_,maps,_=E.training_data();x=torch.stack([E.tensor(E.render(a,m,'shared',1,1)) for a,m in zip(ims[:8],maps[:8])])
    for arm in E.ARMS:
        steps=[json.loads(s) for s in (ROOT/(arm+'-steps.jsonl')).read_text().splitlines()];assert len(steps)==2016;trace_count+=len(steps)
        for i,r in enumerate(steps,1):
            fac=.1+.9*(i-1)/127 if i<=128 else 1.
            assert r['step']==i and r['epoch']==(i-1)//8 and np.isfinite(r['loss']) and abs(r['head_lr']-.01*fac)<1e-15 and abs(r['body_lr']-.0001*fac)<1e-15
        assert abs(np.mean([r['loss'] for r in steps[:256]])-tr['arms'][arm]['first256_online_LL'])<1e-15
        assert abs(np.mean([r['loss'] for r in steps[-512:]])-tr['arms'][arm]['last512_online_LL'])<1e-15
        for step,s in tr['arms'][arm]['stages'].items():
            for part,kind in [('clean','clean'),('probe','own_probe')]:
                z=train[arm+'_'+step+'_'+part];y=train['clean_labels' if part=='clean' else 'probe_labels']
                errs.append(abs(ll(z,y).mean()-s[kind]['mean_LL_nats']));assert abs((z.argmax(1)==y).mean()-s[kind]['accuracy'])<1e-12
        for ck in tr['arms'][arm]['checkpoints']:
            assert E.sha(ROOT/ck['file'])==ck['sha256'] and (ROOT/ck['file']).stat().st_size==ck['bytes']
            saved=torch.load(ROOT/ck['file'],map_location='cpu',weights_only=True)
            assert E.D.state_hash(saved['model'])==ck['state_sha256'] and saved['protocol_sha256']==E.sha(ROOT/'protocol.json') and saved['code_sha256']==E.sha(ROOT/'experiment.py')
            assert all(bool(torch.isfinite(v).all()) for v in saved['model'].values())
            m=torchvision.models.resnet18(weights=None);m.fc=torch.nn.Linear(512,2);m.load_state_dict(saved['model'],strict=True);m.eval()
            assert E.D.bn_stats(m)==tr['arms'][arm]['BN_initial_sha256']
            with torch.inference_mode():z=m(x).numpy()
            dz=float(np.max(np.abs(z-train[arm+'_'+str(saved['step'])+'_clean'][:8])));assert dz<1e-4
            CPU.append(dict(file=ck['file'],CPU_first8_train_logit_maxabs=dz))
            if saved['step']==2016:
                assert len(saved['optimizer']['param_groups'])==2 and len(saved['optimizer']['state'])==62
            del m,saved
        scale=float(np.abs(train[arm+'_2016_clean'][:,1]-train[arm+'_2016_clean'][:,0]).mean())
        assert scale==tr['arms'][arm]['normalizer']
        for kind in ['original','guarded','foreground_only','background_swap']:
            z=test[arm+'_'+kind];y=np.repeat(test['labels'],4) if kind=='background_swap' else test['labels']
            errs.append(abs(ll(z,y).mean()-out['arms'][arm][kind]['mean_LL_nats']));assert (z.argmax(1)==y).mean()==out['arms'][arm][kind]['accuracy']
        margin=np.diff(test[arm+'_background_swap'].astype(float).reshape(48,4,2),axis=2)[:,:,0]
        s=margin.std(1)/scale;assert np.array_equal(s,test[arm+'_sensitivity'])
    A=test['shared_sensitivity']-test['semantic_independent_sensitivity'];B=test['shifted_independent_sensitivity']-test['semantic_independent_sensitivity']
    rng=np.random.default_rng(2026102117);idx=np.concatenate([rng.choice(np.flatnonzero(test['labels']==k),(5000,24),replace=True) for k in [0,1]],axis=1)
    assert A.mean()==out['primary_A_mean'] and B.mean()==out['secondary_specificity_B_mean']
    assert np.array_equal(np.quantile(A[idx].mean(1),[.025,.975]),out['A_bootstrap95'])
    assert np.array_equal(np.quantile(B[idx].mean(1),[.025,.975]),out['B_bootstrap95'])
    assert max(errs)<1e-6 and all(out['candidate_screen'].values())==out['passes_large_effect_pilot_screen']
    ids={r['source_id'] for r in p['test_rows']};assert len(ids)==48 and not ids.intersection(p['excluded_historical_photo_or_mask_IDs'])
    fetch=E.load('heldout-fetch-receipt.json');assert len(fetch['files'])==96
    for r in fetch['files']:
        sub,ext=('heldout-photos','.jpg') if r['kind']=='image' else ('heldout-masks','.png');path=ROOT/sub/(r['source_id']+ext)
        assert path.stat().st_size==r['bytes'] and E.sha(path)==r['sha256']
    result=dict(status='saved_outputs_verified',at_UTC=E.now(),traces_checked=trace_count,loss_recompute_max_abs=max(errs),checkpoints_strict_loaded=len(CPU),
        checkpoint_CPU_checks=CPU,finite_all_saved_parameters=True,BN_statistics_fixed=True,all96heldout_file_hashes_checked=True,
        metrics_and_paired_bootstrap_exactly_recomputed=True,heldout_no_additional_inference=True,CPU_training_probe_only=True,
        shared_sensitivity_denominator_nonzero=out['arms']['shared']['mean_normalized_sensitivity']>0,
        final_results_sha256=E.sha(ROOT/'final-results.json'),training_results_sha256=E.sha(ROOT/'training-results.json'),code_sha256=E.sha(ROOT/'experiment.py'))
    E.save('verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
