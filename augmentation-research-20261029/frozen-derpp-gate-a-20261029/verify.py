"""Independent verification of saved Gate A records; no new model forward."""
import datetime,json
from pathlib import Path
import numpy as np
import torch
from prepare import sha,savej,state_fingerprints

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'mammoth-derpp-baseline-20261029'
OUT=ROOT/'run-001'

def main():
    f=json.loads((ROOT/'protocol-freeze.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in f['files'].items())
    assert all(sha(Path(n))==h for n,h in f['external_readonly_inputs'].items())
    c=torch.load(BASE/'run-002/task2-final.pt',map_location='cpu',weights_only=False)
    assert state_fingerprints(c)==json.loads((ROOT/'input-state-fingerprints.json').read_text())
    z=c['buffer']['logits'].numpy().astype(np.float64)
    plan=np.load(ROOT/'frozen-random-plan.npz',allow_pickle=False)
    cal=np.load(OUT/'shadow-calibration.npz',allow_pickle=False)
    # Check fixed streams, including independent score groups, without inference.
    for g in range(2):
        for v in range(8):
            rng=np.random.default_rng(20261029+800000+100*g+v)
            expected=np.column_stack([rng.integers(0,9,(500,2)),rng.integers(0,2,500)])
            assert np.array_equal(expected,plan['score_parameters'][g,v])
    assert np.array_equal(np.random.default_rng(20261029+700000).random((160,500)),plan['calibration_uniforms'])
    w=np.ones(500);Qsum=0.;max_werror=0.
    # Independently recompute in original half-SUM units, not the implementation's half-MEAN.
    for b in range(160):
        u=np.clip(plan['calibration_uniforms'][b],1e-15,1-1e-15)
        ids=np.argsort(-(np.log(w)-np.log(-np.log(u))),kind='stable')[:32]
        assert np.array_equal(ids,cal['ids'][b])
        h=cal['logits'][b].astype(np.float64)
        Dsum=np.square(h-z[ids]).sum(-1)/2
        assert np.max(np.abs(Dsum/10-cal['residual'][b]))<1e-12
        assert np.max(np.abs(w[ids]-cal['w_before_after'][b,0]))<1e-13
        assert abs(Qsum/10-cal['q_before_after'][b,0])<1e-13
        d=Dsum+1e-5;Qsum=(468*Qsum+32*np.median(d))/500
        eta=np.maximum(0,np.minimum(1,2-d/Qsum))
        ratio=(eta*d+(1-eta)*Qsum)/d
        w[ids]=(w[ids]+np.sqrt(ratio))/2
        error=np.max(np.abs(w[ids]-cal['w_before_after'][b,1]));max_werror=max(max_werror,float(error))
        assert error<1e-13 and abs(Qsum/10-cal['q_before_after'][b,1])<1e-13
    assert np.max(np.abs(w-cal['weights']))<1e-13
    assert np.array_equal(np.bincount(cal['ids'].ravel(),minlength=500),cal['visits'])
    logits=np.load(OUT/'score-logits.npy',allow_pickle=False).astype(np.float64)
    saved=np.load(OUT/'scored-arrays.npz',allow_pickle=False)
    result=json.loads((OUT/'gate-A.json').read_text())
    assert logits.shape==(2,8,500,10) and np.isfinite(logits).all()
    max_score_error=0.;tvs=[]
    for g in range(2):
        h=logits[g];mean=h.sum(0)/8
        avg=(np.square(h).mean(0)-2*z*mean+np.square(z)).sum(-1)/20
        dm=np.square(mean-z).sum(-1)/20
        var=(np.square(h).mean(0)-np.square(mean)).sum(-1)/20
        for name,value in [('Davg',avg),('Dmean',dm),('V',var)]:
            error=float(np.max(np.abs(value-saved[f'g{g+1}_raw_{name}'])))
            max_score_error=max(max_score_error,error);assert error<1e-12
        A=saved[f'g{g+1}_raw_A'];B=saved[f'g{g+1}_raw_B'];weights=cal['weights']
        for group in np.unique(saved['strata']):
            ids=np.flatnonzero(saved['strata']==group)
            ordered_weights=np.sort(weights[ids])
            for score,arm in [(saved[f'g{g+1}_raw_Davg'],A),(saved[f'g{g+1}_raw_Dmean'],B)]:
                order=ids[np.lexsort((ids,saved['arrivals'][ids],saved['rows'][ids],-score[ids]))]
                assert np.array_equal(arm[order],ordered_weights)
        tv=float(np.abs(A/A.sum()-B/B.sum()).sum()/2);tvs.append(tv)
        assert abs(tv-result['groups'][g]['modes']['raw']['TV_A_B'])<1e-15
    inf=json.loads((OUT/'inference-summary.json').read_text())
    assert inf['optimizer_steps']==0 and inf['backward_calls']==0
    assert inf['model_and_BN_bitwise_unchanged'] and inf['buffer_and_targets_unchanged']
    proc=json.loads((ROOT/'process-timing.json').read_text());assert proc['exit_code']==0 and not proc['timeout_terminated']
    assert result['branches_executed']==0
    out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_passed':True,
         'checkpoint_and_frozen_files_unchanged':True,'independent_shadow_sampling_and_half_sum_replay':True,
         'shadow_weight_max_abs_error':max_werror,'independent_moment_score_max_abs_error':max_score_error,
         'raw_TVs':tvs,'complete_slots':500,'independent_groups':2,'views_per_group':8,
         'model_forwards_in_verification':0,'optimizer_steps':0}
    savej(OUT/'independent-verification.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
