"""Independent readback of fixed controls, exact old pixels, pairing and saved readouts."""
from pathlib import Path
from collections import defaultdict,Counter
import datetime
import hashlib
import json
import math
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/mechanism-transfer-pets-20261021')
PROJECT=DATA.parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(name):return json.loads((ROOT/name).read_text())

def independent_score(row,model):
    norm=math.sqrt(math.fsum(float(v)*float(v) for v in row))
    z=[float(v)/norm for v in row]
    centers=model['centers']
    return 10*math.fsum(z[i]*(centers[1][i]-centers[0][i]) for i in range(512))

def independent_metrics(rows,x,model):
    losses=[];correct=[]
    for r,vector in zip(rows,x):
        score=independent_score(vector,model);y=r['Y']
        losses.append(max(0.,score)+math.log1p(math.exp(-abs(score)))-y*score)
        correct.append(int((score>0)==bool(y)))
    return math.fsum(losses)/len(losses),math.fsum(correct)/len(correct)

def main():
    old=json.loads((DATA/'delivery-verification.json').read_text())
    for r in old['files']:
        p=DATA/r['relative_path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    old_ready=DATA.parent/'encoder-readiness-20261021'
    ready=json.loads((old_ready/'delivery-verification.json').read_text())
    for r in ready['files']:
        p=old_ready/r['relative_path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    cal=load('calibration-result.json');freeze=load('measurement-freeze.json')
    assert cal['passed'] and all(cal['gates'].values())
    assert sha(ROOT/'calibration-result.json')==freeze['calibration_result_sha256']
    assert sha(ROOT/'source-train-calibration-plan.json')==freeze['calibration_plan_sha256']
    ca,ma=load('calibration-encoder-audit.json'),load('main-encoder-audit.json')
    assert ca['weight_sha256']==ma['weight_sha256']==freeze['encoder_audit']['weight_sha256']
    assert ca['code_sha256']==ma['code_sha256']==sha(ROOT/'run_encoder_round.py')
    assert ca['preprocess']==ma['preprocess'] and ca['inference_device']==ma['inference_device']=='cpu'
    assert cal['at_utc']<=freeze['at_utc']<ma['at_utc']
    assert ca['threads']==ma['threads']==1 and ca['interop_threads']==ma['interop_threads']==1
    frame_counts={};opaque_checks=0;frame_hash_checks=0
    for mode in ['calibration','main']:
        manifest=load(mode+'-frames.json');rows=manifest['records']
        with (ROOT/(mode+'-frames.rgba')).open('rb') as f:
            for i,row in enumerate(rows):
                pixels=f.read(65536)
                assert len(pixels)==65536 and row['index']==i
                assert hashlib.sha256(pixels).hexdigest()==row['RGBA_sha256']
                assert all(v==255 for v in pixels[3::4]);opaque_checks+=1;frame_hash_checks+=1
            assert not f.read(1)
        assert sha(ROOT/(mode+'-frames.rgba'))==load(mode+'-encoder-audit.json')['input_frames_sha256']
        assert sha(ROOT/(mode+'-features.npy'))==load(mode+'-encoder-audit.json')['feature_sha256']
        frame_counts[mode]=len(rows)
    rows=load('main-frames.json')['records']
    x=np.load(ROOT/'main-features.npy',allow_pickle=False)
    assert x.shape==(8704,512) and np.isfinite(x).all()
    groups=defaultdict(list)
    for r in rows:groups[(r['case'],r['layout'])].append(r)
    for group in groups.values():
        assert len({r['actual_background_id'] for r in group})==len(group)
        assert sum(r['coupled'] for r in group)==1
        assert Counter(r['C'] for r in group)=={0:len(group)//2,1:len(group)//2}
        for field in ['foreground_id','T','cropped','visible_alpha_fraction','retained_fraction_of_T','Y']:
            assert len({r[field] for r in group})==1
    fit_ids={r['foreground_id'] for r in rows if r['split']=='fit'}|{r['actual_background_id'] for r in rows if r['split']=='fit'}
    heldout_ids={r['foreground_id'] for r in rows if r['split']=='heldout'}|{r['actual_background_id'] for r in rows if r['split']=='heldout'}
    assert len(fit_ids)==len(heldout_ids)==32 and not fit_ids&heldout_ids
    for layout in ['ordinary','protected']:
        for split in ['fit','heldout']:
            independent=[r for r in rows if r['layout']==layout and r['split']==split]
            coupled=[r for r in independent if r['coupled']]
            for field in ['foreground_id','actual_background_id','Y']:
                a,b=Counter(r[field] for r in coupled),Counter(r[field] for r in independent)
                assert all(a[k]*len(independent)==b[k]*len(coupled) for k in set(a)|set(b))
    # Exact readback of32 previously saved QA pixel rasters, independent PNG decoder.
    render=json.loads((DATA/'render-receipt.json').read_text())
    photo_plan=json.loads((DATA/'pixel-photo-plan.json').read_text())['rows']
    qa_count=0;guard_count=0
    raw=np.memmap(ROOT/'main-frames.rgba',mode='r',dtype=np.uint8,shape=(len(rows),128,128,4))
    lookup={(r['case'],r['layout'],r['actual_background_id']):r for r in rows}
    cases={r['case']:r for r in rows}
    for q in render['QA']:
        reference=cases[q['case']]
        if q['variant']=='coupled':actual=reference['base_background_id']
        else:
            base=next(r for r in photo_plan if r['source_id']==reference['base_background_id'])
            actual=next(r['source_id'] for r in photo_plan if r['role']=='background' and r['split']=='heldout' and r['macroblock']==reference['block'] and r['species']!=base['species'])
        r=lookup[q['case'],q['layout'],actual]
        expected=np.asarray(Image.open(DATA/q['file']).convert('RGBA'))
        assert np.array_equal(expected,raw[r['index']]),q
        qa_count+=1
    for case in sorted(set(q['case'] for q in render['QA'])):
        group=groups[case,'protected']
        guard=np.asarray(Image.open(DATA/'qa'/('case'+str(case)+'-guard.png')).convert('L'))>0
        ref=raw[group[0]['index']]
        for r in group[1:]:
            assert np.array_equal(raw[r['index']][guard],ref[guard]);guard_count+=1
    models=load('round2-models.json');results=load('round2-results.json')
    max_loss_error=0.;max_accuracy_error=0.
    for layout in ['protected','ordinary']:
        for trained,model in models[layout].items():
            for policy in ['coupled','independent']:
                ids=[i for i,r in enumerate(rows) if r['layout']==layout and r['split']=='heldout' and (policy=='independent' or r['coupled'])]
                loss,acc=independent_metrics([rows[i] for i in ids],x[ids],model)
                saved=results['layouts'][layout]['overall']['table'][trained][policy]
                max_loss_error=max(max_loss_error,abs(loss-saved['mean_log_loss_nats']))
                max_accuracy_error=max(max_accuracy_error,abs(acc-saved['accuracy']))
    assert max_loss_error<1e-12 and max_accuracy_error==0
    audit=dict(status='passed',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        old_negative_study_files_bytes_and_SHA256_unchanged=len(old['files']),
        original_readiness_proposal_files_unchanged=len(ready['files']),
        gate_and_freeze_preceded_main=True,encoder_code_and_weight_unchanged_after_freeze=True,
        CPU_single_thread_confirmed=True,frame_counts=frame_counts,
        per_frame_hashes_verified=frame_hash_checks,fully_opaque_frames=opaque_checks,
        exact_old_QA_RGBA_images=qa_count,protected_guard_pixel_comparisons=guard_count,
        exact_C_balance_in_all_independent_cases=True,foreground_T_and_background_marginals_preserved=True,
        source_fit_and_heldout_photos_disjoint=True,calibration_only_original_fit_side=True,
        independent_math_fsum_LL_readback_max_abs_error=max_loss_error,accuracy_readback_error=max_accuracy_error,
        no_independence_claim_over_augmented_frames=True,ARIS=False,GPU=False)
    with (ROOT/'round2-verification.json').open('x') as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
