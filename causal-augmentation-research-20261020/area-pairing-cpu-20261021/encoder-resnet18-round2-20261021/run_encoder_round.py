"""Fixed CPU-only frozen encoder measurement; no backbone training or model search."""
import os
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1",
                  CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
READY=BASE/'encoder-readiness-20261021'
RUNTIME=READY/'runtime'
os.environ['TORCH_HOME']=str(RUNTIME/'torch-cache')
import numpy as np
import torch
import torchvision
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def feature_extraction(mode):
    started=time.perf_counter();cpu=time.process_time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_device('cpu');torch.use_deterministic_algorithms(True)
    receipt=json.loads((RUNTIME/'weight-download-receipt.json').read_text())
    path=RUNTIME/'torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
    assert path.stat().st_size==46830571 and sha(path)==receipt['sha256']
    assert receipt['sha256']=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','pillow','filelock',
        'fsspec','Jinja2','MarkupSafe','mpmath','networkx','setuptools','sympy','typing_extensions']}
    assert {k:versions[k] for k in ['torch','torchvision','numpy','pillow']}=={
        'torch':'2.14.1','torchvision':'0.29.1','numpy':'2.5.4','pillow':'13.0.0'}
    model=torchvision.models.resnet18(weights=None)
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);del state
    model.fc=torch.nn.Identity();model.eval().requires_grad_(False).to('cpu')
    assert all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters())
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    preprocess=weights.transforms()
    desc=json.loads((ROOT/(mode+'-frames.json')).read_text())
    rows=desc['records'];rawpath=ROOT/(mode+'-frames.rgba')
    assert rawpath.stat().st_size==len(rows)*65536
    raw=np.memmap(rawpath,dtype=np.uint8,mode='r',shape=(len(rows),65536))
    features=np.empty((len(rows),512),dtype=np.float32)
    last_update=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            end=min(start+8,len(rows))
            batch=torch.stack([preprocess(Image.frombytes('RGBA',(128,128),raw[i].tobytes()).convert('RGB'))
                for i in range(start,end)]).to('cpu')
            assert batch.device.type=='cpu'
            out=model(batch)
            assert out.shape==(end-start,512) and out.device.type=='cpu'
            features[start:end]=out.numpy()
            elapsed=time.perf_counter()-started
            if time.perf_counter()-last_update>=15 or end==len(rows):
                print(json.dumps({'phase':mode,'images_done':end,'total':len(rows),'wall_seconds':elapsed}),flush=True)
                last_update=time.perf_counter()
            if mode=='main' and elapsed>600:raise TimeoutError('Approved main CPU time budget exceeded; preserve outputs and report')
    assert np.isfinite(features).all()
    with (ROOT/(mode+'-features.npy')).open('xb') as f:np.save(f,features,allow_pickle=False)
    audit=dict(mode=mode,at_utc=now(),python=sys.version,executable=sys.executable,versions=versions,
        weight_url=receipt['url'],weight_sha256=receipt['sha256'],preprocess=repr(preprocess),
        encoder='frozen torchvision ResNet18 IMAGENET1K_V1, fc replaced with Identity, avgpool512',
        parameter_device_types=sorted(set(p.device.type for p in model.parameters())),
        inference_device='cpu',threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        no_backbone_training=True,no_T_C_mask_path_as_features=True,ARIS=False,GPU=False,A100=False,
        input_frames_sha256=sha(rawpath),input_manifest_sha256=sha(ROOT/(mode+'-frames.json')),
        feature_sha256=sha(ROOT/(mode+'-features.npy')),code_sha256=sha(Path(__file__)),
        wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu)
    save(mode+'-encoder-audit.json',audit)
    return rows,features,audit


def normalized(x):
    x=np.asarray(x,dtype=np.float64)
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    assert np.all(norms>0)
    return x/norms


def fit(rows,features,label='Y'):
    z=normalized(features);y=np.array([r[label] for r in rows])
    centers=normalized(np.stack([z[y==c].mean(axis=0) for c in [0,1]]))
    return {'centers':centers.tolist(),'class_counts':[int((y==c).sum()) for c in [0,1]],
            'logit_scale':10.,'dimension':512,'input':'RGB encoder features only'}


def predictions(features,model):
    logits=10*normalized(features)@np.array(model['centers'],dtype=np.float64).T
    return logits[:,1]-logits[:,0]


def metrics(rows,features,model,label='Y'):
    y=np.array([r[label] for r in rows]);scores=predictions(features,model)
    loss=np.logaddexp(0,scores)-y*scores;correct=(scores>0)==y
    return {'records':len(rows),'mean_log_loss_nats':float(loss.mean()),'accuracy':float(correct.mean())}


def select(rows,x,predicate):
    ids=[i for i,r in enumerate(rows) if predicate(r)]
    return [rows[i] for i in ids],x[ids]


def calibration():
    rows,x,audit=feature_extraction('calibration')
    plan=json.loads((ROOT/'source-train-calibration-plan.json').read_text())
    train,tx=select(rows,x,lambda r:r['kind']=='scaled_neutral' and r['partition']=='prototype_fit')
    test,bx=select(rows,x,lambda r:r['kind']=='scaled_neutral' and r['partition']=='calibration')
    intact,ax=select(rows,x,lambda r:r['kind']=='intact_neutral')
    assert len(train)==192 and len(test)==64 and len(intact)==4
    assert sorted(r['case'] for r in train)==plan['prototype_fit_neutral_cases']
    assert sorted(r['case'] for r in test)==plan['calibration_neutral_cases']
    assert not {r['foreground_id'] for r in train}&{r['foreground_id'] for r in test}
    assert all(r['split']=='fit' for r in rows)
    model=fit(train,tx)
    a=metrics(intact,ax,model);b=metrics(test,bx,model)
    by_source={}
    for source in sorted(set(r['foreground_id'] for r in test)):
        ss,sx=select(test,bx,lambda r:r['foreground_id']==source)
        by_source[source]=metrics(ss,sx,model)
    gates={'A_at_least_3_of_4':a['accuracy']>=.75,'B_source_mean_at_least_075':
        np.mean([m['accuracy'] for m in by_source.values()])>=.75,
        'B_each_source_at_least_050':all(m['accuracy']>=.5 for m in by_source.values())}
    passed=all(gates.values())
    pred=[]
    for ss,sx in [(intact,ax),(test,bx)]:
        for row,score in zip(ss,predictions(sx,model)):
            pred.append({'index':row['index'],'kind':row['kind'],'case':row.get('case'),
                'foreground_id':row['foreground_id'],'Y':row['Y'],'logit_dog_minus_cat':float(score),
                'prediction':int(score>0),'cropped':row.get('cropped'),'T':row.get('T'),
                'visible_alpha_fraction':row.get('visible_alpha_fraction')})
    result={'status':'measurement_gate_passed' if passed else 'measurement_gate_failed_main_not_run',
        'at_utc':now(),'positive_control_A':a,'positive_control_B':b,'B_per_source':by_source,
        'gates':{k:bool(v) for k,v in gates.items()},'passed':passed,
        'prototype_fit_sources':12,'calibration_sources':4,'source_disjoint':True,
        'only_original_fit_side_sources_used':True,'no_external_heldout_used':True,
        'fixed_head':'L2-normalized 512 features; normalized class means; logits10*cosine',
        'predictions':pred,'main_test_run':False,'no_tuning_after_calibration':True}
    save('calibration-model.json',model);save('calibration-result.json',result)
    if passed:
        save('measurement-freeze.json',dict(at_utc=now(),status='fixed_calibration_gates_passed_before_main',
            calibration_result_sha256=sha(ROOT/'calibration-result.json'),
            calibration_model_sha256=sha(ROOT/'calibration-model.json'),
            calibration_plan_sha256=sha(ROOT/'source-train-calibration-plan.json'),
            encoder_audit=audit,prototype_head='L2 normalize -> class mean -> L2 normalize; logits10*cosine',
            main_training_sources='all original16 fit foreground photos, both policy-specific heads; frozen encoder',
            external_heldout_previously_used_by_old_weak_model=True,main_screen='positive deltaLL and DID in all4 blocks'))
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},ensure_ascii=False,indent=2),flush=True)


def compare(rows,x,models):
    table={}
    for trained,model in models.items():
        table[trained]={}
        for policy in ['coupled','independent']:
            rr,xx=select(rows,x,lambda r:policy=='independent' or r['coupled'])
            table[trained][policy]=metrics(rr,xx,model)
    effects={}
    for trained in models:
        c,i=table[trained]['coupled'],table[trained]['independent']
        effects[trained+'_trained_delta_LL_independent_minus_coupled']=i['mean_log_loss_nats']-c['mean_log_loss_nats']
        effects[trained+'_trained_accuracy_drop_coupled_minus_independent']=c['accuracy']-i['accuracy']
    effects['difference_in_differences_LL']=effects['coupled_trained_delta_LL_independent_minus_coupled']-effects['independent_trained_delta_LL_independent_minus_coupled']
    return dict(table=table,effects=effects)


def main_test():
    freeze=json.loads((ROOT/'measurement-freeze.json').read_text())
    assert freeze['status']=='fixed_calibration_gates_passed_before_main'
    assert sha(ROOT/'calibration-result.json')==freeze['calibration_result_sha256']
    rows,x,audit=feature_extraction('main')
    assert len(rows)==8704
    result={};saved={}
    for layout in ['protected','ordinary']:
        sub,sx=select(rows,x,lambda r:r['layout']==layout)
        train,tx=select(sub,sx,lambda r:r['split']=='fit')
        test,vx=select(sub,sx,lambda r:r['split']=='heldout')
        models={}
        for policy in ['coupled','independent']:
            rr,xx=select(train,tx,lambda r:policy=='independent' or r['coupled'])
            models[policy]=fit(rr,xx)
        blocks={}
        for block in range(4):
            rr,xx=select(test,vx,lambda r:r['block']==block);blocks[str(block)]=compare(rr,xx,models)
        strata={}
        for cropped in [False,True]:
            rr,xx=select(test,vx,lambda r:r['cropped']==cropped);strata[str(cropped)]=compare(rr,xx,models) if rr else None
        by_class={}
        for y in [0,1]:
            rr,xx=select(test,vx,lambda r:r['Y']==y);by_class[str(y)]=compare(rr,xx,models)
        delta=[blocks[str(b)]['effects']['coupled_trained_delta_LL_independent_minus_coupled'] for b in range(4)]
        did=[blocks[str(b)]['effects']['difference_in_differences_LL'] for b in range(4)]
        result[layout]=dict(overall=compare(test,vx,models),heldout_source_disjoint_macroblocks=blocks,
            crop_strata=strata,per_class=by_class,positive_delta_LL_blocks=sum(d>0 for d in delta),
            positive_DID_blocks=sum(d>0 for d in did),cautious_pilot_screen=all(d>0 for d in delta+did))
        saved[layout]=models
        print(json.dumps({'layout':layout,'effects':result[layout]['overall']['effects'],'block_delta_LL':delta,'block_DID':did}),flush=True)
    out=dict(status='completed_exploratory_frozen_ResNet18_round2',at_utc=now(),CPU_only=True,
        ARIS=False,GPU=False,A100=False,primary_layout='protected',layouts=result,
        freeze_sha256=sha(ROOT/'measurement-freeze.json'),encoder_audit_sha256=sha(ROOT/'main-encoder-audit.json'),
        unique_RGB_frames=len(rows),external_heldout_macroblocks=4,
        limitations=['Mechanism migration on Oxford-IIIT Pet, not ForNet reconstruction or natural OOD evidence.',
            'Only16 external heldout foreground and16 background source photographs; composite frames are dependent.',
            'External heldout already inspected by old weak model; new round is exploratory.',
            'ImageNet pretraining overlap with Oxford photos cannot be ruled out.',
            'Protected neutral guard is an explicit renderer control, absent from original ForAug.',
            'A positive pilot screen would not establish novelty or a research contribution.'])
    save('round2-models.json',saved);save('round2-results.json',out)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['calibration','main']
    try:os.nice(10)
    except OSError:pass
    signal.alarm(650 if sys.argv[1]=='main' else 240)
    if sys.argv[1]=='calibration':calibration()
    else:main_test()
