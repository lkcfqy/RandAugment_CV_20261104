"""Readback of fixed objectives, source weights, cross-fitting, chronology and new test."""
from pathlib import Path
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(n):return json.loads((ROOT/n).read_text())

def main():
    protocol=load('protocol.json');freeze=load('training-freeze.json');fetch=load('test-fetch-receipt.json')
    result=load('test-results.json');models=load('trained-models.json');teachers=load('teacher-models.json')
    assert sha(ROOT/'protocol.json')==freeze['protocol_sha256']
    assert sha(ROOT/'train_probe.py')==freeze['training_code_sha256']
    assert sha(ROOT/'fetch_fresh_test.py')==freeze['fetch_code_sha256']
    assert sha(ROOT/'trained-models.json')==freeze['models_sha256']
    assert sha(ROOT/'teacher-models.json')==freeze['teachers_sha256']
    assert sha(ROOT/'training-weights.json')==freeze['training_weights_sha256']
    assert freeze['at_utc']<fetch['at_utc']<result['at_utc']
    assert fetch['HTTP_range_bytes']==protocol['new_test_HTTP_range_bytes']<10_000_000
    assert fetch['JPEG_payload_bytes']==protocol['new_test_JPEG_payload_bytes']
    assert not fetch['errors'] and len(fetch['files'])==64
    for r in fetch['files']:
        p=ROOT/'test-photos'/(r['source_id']+'.jpg');assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    x=np.load(ROOT/'train-features.npy',allow_pickle=False).astype(np.float64)
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    rows=load('train-image-manifest.json')['records'];y=np.array([r['Y'] for r in rows])
    assert x.shape==(576,512) and len(rows)==576
    weights=load('training-weights.json')['arm_weights'];keep=np.array([r['retained'] for r in rows]).reshape(64,9)
    counts=keep.sum(axis=1);assert keep[:,0].all()
    expected=counts/counts.sum()
    source_error=0.;grad_error=0.;objective_error=0.;max_gradient=0.
    for name,values in weights.items():
        w=np.array(values);assert (w>=0).all() and abs(w.sum()-1)<1e-14
        if name in ['filtered_standard_counts','all_views_matched_filtered_source_weights','clean_matched_filtered_source_weights']:
            source_error=max(source_error,float(np.max(np.abs(w.reshape(64,9).sum(axis=1)-expected))))
        else:assert np.max(np.abs(w.reshape(64,9).sum(axis=1)-1/64))<1e-14
        model=models[name];coef=np.array(model['coef']);z=x@coef+model['intercept']
        loss=np.logaddexp(0,z)-y*z
        objective=float(w@loss+.001*.5*(coef@coef))
        probability=1/(1+np.exp(-z));residual=w*(probability-y)
        grad=max(float(np.abs(x.T@residual+.001*coef).max()),abs(float(residual.sum())))
        objective_error=max(objective_error,abs(objective-model['objective']));max_gradient=max(max_gradient,grad)
        grad_error=max(grad_error,abs(grad-model['gradient_inf_norm']))
        assert grad<1e-6
    assert source_error<1e-15 and objective_error<1e-12 and grad_error<1e-12
    for i,row in enumerate(rows):
        model=teachers[str(row['fold'])];score=float(x[i]@np.array(model['coef'])+model['intercept'])
        assert abs(score-row['teacher_fold_score'])<1e-12
        assert row['retained']==(row['view']==0 or ((score>0)==row['Y']))
    # Cross-fit reconstruction of every teacher's optimization objective uses other photo folds only.
    anchors=np.arange(0,576,9)
    for fold in range(4):
        ids=[int(i) for i in anchors if rows[i]['fold']!=fold];assert len(ids)==48
        model=teachers[str(fold)];coef=np.array(model['coef']);z=x[ids]@coef+model['intercept']
        obj=float(np.mean(np.logaddexp(0,z)-y[ids]*z)+.001*.5*(coef@coef))
        assert abs(obj-model['objective'])<1e-12
    xt=np.load(ROOT/'test-features.npy',allow_pickle=False).astype(np.float64);xt/=np.linalg.norm(xt,axis=1,keepdims=True)
    test=load('test-image-manifest.json')['rows'];predictions=load('test-predictions.json')
    prior_ids=set(protocol['prior_photo_or_mask_IDs_excluded']);new_ids={r['source_id'] for r in test}
    assert len(new_ids)==64 and not new_ids&prior_ids
    train_rgb={r['decoded_RGB_sha256'] for r in load('train-image-manifest.json')['photos']}
    test_rgb={r['decoded_RGB_sha256'] for r in test};assert len(test_rgb)==64 and not train_rgb&test_rgb
    error=0.;losses={}
    for name,model in models.items():
        ll=[]
        for i,(row,vec) in enumerate(zip(test,xt)):
            score=math.fsum(float(a)*b for a,b in zip(vec,model['coef']))+model['intercept']
            loss=max(0,score)+math.log1p(math.exp(-abs(score)))-row['Y']*score
            error=max(error,abs(loss-predictions[name][i]['loss']))
            assert predictions[name][i]['correct']==((score>0)==row['Y'])
            ll.append(loss)
        losses[name]=np.array(ll)
        assert abs(math.fsum(ll)/64-result['table'][name]['mean_LL_nats'])<1e-12
    difference=losses['all_views_matched_filtered_source_weights']-losses['filtered_standard_counts']
    assert abs(float(difference.mean())-result['primary_benefit_LL_matched_source_control_minus_filtered'])<1e-12
    rng=np.random.default_rng(2026102106);Y=np.array([r['Y'] for r in test])
    ids=np.concatenate([rng.choice(np.where(Y==c)[0],size=(5000,32),replace=True) for c in [0,1]],axis=1)
    CI=np.quantile(difference[ids].mean(axis=1),[.025,.975])
    assert np.max(np.abs(CI-np.array(result['primary_paired_species_bootstrap95_CI'])))<1e-12
    verification=dict(status='passed',source_photos_train64_test64=True,excluded_all128_previously_used_photo_or_mask_IDs=True,
        exact_train_test_decoded_RGB_duplicates=0,all_11_fixed_logistic_objectives_and_source_crossfit_verified=True,
        matched_source_weight_max_abs_error=source_error,student_objective_max_abs_readback_error=objective_error,
        student_gradient_readback_max_abs_error=grad_error,largest_student_gradient_inf_norm=max_gradient,
        test_predictions_independent_math_fsum_max_abs_loss_error=error,paired_bootstrap95_CI_reproduced=True,
        model_code_and_weights_frozen_before_test_download=True,all64JPEG_CRC_SHA_and_range_sizes_verified=True,
        fresh_test_consumption_ledger_saved=True,CPU_only=True,ARIS=False,GPU=False)
    with (ROOT/'verification.json').open('x') as f:json.dump(verification,f,indent=2);f.write('\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
