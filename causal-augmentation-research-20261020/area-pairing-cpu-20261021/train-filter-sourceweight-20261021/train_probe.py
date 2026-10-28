"""One training-augmentation value test with source-weight controls and fresh raw test photos."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from collections import Counter
import datetime,hashlib,inspect,json,signal,sys,time
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F,TrivialAugmentWide,InterpolationMode

ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/mechanism-transfer-pets-20261021')
RUNTIME=DATA.parent/'encoder-readiness-20261021/runtime'
MEAN=[.485,.456,.406];STD=[.229,.224,.225]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    torch.use_deterministic_algorithms(True)

def model_encoder():
    path=RUNTIME/'torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
    assert sha(path)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    model=torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    model.fc=torch.nn.Identity();model.eval().requires_grad_(False).to('cpu')
    assert all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters())
    return model

def tensor(image):return F.normalize(F.to_tensor(image),MEAN,STD)

def base_image(image):
    return F.center_crop(F.resize(image,256,InterpolationMode.BILINEAR,antialias=True),[224,224])

def encode(images):
    model=model_encoder();start=time.perf_counter();result=np.empty((len(images),512),dtype=np.float32)
    with torch.inference_mode():
        for i in range(0,len(images),8):
            batch=torch.stack([tensor(im) for im in images[i:i+8]]).to('cpu')
            assert batch.device.type=='cpu'
            result[i:i+len(batch)]=model(batch).numpy()
    assert np.isfinite(result).all()
    return result,time.perf_counter()-start

def normalize(x):
    x=x.astype(np.float64);norm=np.linalg.norm(x,axis=1,keepdims=True)
    assert (norm>0).all()
    return x/norm

def fit_logistic(x,y,weight):
    # Fixed objective; no data-dependent class weighting or regularization search.
    t0=time.perf_counter();weight=np.asarray(weight,dtype=np.float64);weight=weight/weight.sum()
    X=torch.tensor(x,dtype=torch.float64,device='cpu');Y=torch.tensor(y,dtype=torch.float64,device='cpu')
    W=torch.tensor(weight,dtype=torch.float64,device='cpu')
    coef=torch.zeros(512,dtype=torch.float64,device='cpu',requires_grad=True)
    intercept=torch.zeros((),dtype=torch.float64,device='cpu',requires_grad=True)
    optimizer=torch.optim.LBFGS([coef,intercept],lr=1,max_iter=200,tolerance_grad=1e-9,
        tolerance_change=1e-12,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        logits=X@coef+intercept
        loss=(W*torch.nn.functional.binary_cross_entropy_with_logits(logits,Y,reduction='none')).sum()+.001*.5*coef.square().sum()
        loss.backward();calls+=1
        return loss
    optimizer.step(closure);objective=float(closure().detach())
    grad=float(max(coef.grad.abs().max(),intercept.grad.abs()))
    assert grad<1e-6,('optimizer_gradient_gate_failed',grad)
    return dict(coef=coef.detach().numpy().tolist(),intercept=float(intercept.detach()),
        L2_lambda=.001,objective=objective,gradient_inf_norm=grad,closure_calls=calls,
        wall_seconds=time.perf_counter()-t0,converged=True)

def predict(x,model):return x@np.asarray(model['coef'])+model['intercept']

def loss_values(x,y,model):
    z=predict(x,model);return np.logaddexp(0,z)-y*z

def random_view(base,source_id,view,policy):
    seed=int.from_bytes(hashlib.sha256(('train-filter-TA-v1:'+source_id+':'+str(view)).encode()).digest()[:8],'little')&((1<<63)-1)
    g=torch.Generator(device='cpu').manual_seed(seed)
    torch.random.set_rng_state(g.get_state())
    image=policy(base)
    # Replay the same CPU RNG draws solely to record the official sampled operation.
    space=policy._augmentation_space(policy.num_magnitude_bins)
    index=int(torch.randint(len(space),(1,),generator=g).item());op=list(space)[index]
    magnitudes,signed=space[op]
    if magnitudes.ndim>0:
        level=int(torch.randint(len(magnitudes),(1,),dtype=torch.long,generator=g).item());mag=float(magnitudes[level].item())
    else:level=None;mag=0.
    if signed and int(torch.randint(2,(1,),generator=g).item()):mag*=-1
    return image,dict(seed=seed,operation=op,magnitude_level=level,magnitude_value=mag)

def train():
    start=time.perf_counter();protocol=json.loads((ROOT/'protocol.json').read_text())
    assert not (ROOT/'test-photos').exists(),'Fresh test pixels must not be acquired before freeze'
    source_rows=protocol['train_rows'];assert len(source_rows)==64
    policy=TrivialAugmentWide(num_magnitude_bins=31,interpolation=InterpolationMode.NEAREST,fill=128)
    official=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    images=[];records=[];photo_audit=[]
    for source in source_rows:
        path=DATA/'photos'/(source['source_id']+'.jpg')
        original=Image.open(path).convert('RGB');base=base_image(original)
        assert torch.equal(tensor(base),official(original))
        photo_audit.append(dict(source_id=source['source_id'],JPEG_sha256=sha(path),
            decoded_RGB_sha256=hashlib.sha256(original.tobytes()).hexdigest(),width=original.width,height=original.height))
        for view in range(9):
            image,meta=(base,dict(operation='CleanAnchor',seed=None,magnitude_level=None,magnitude_value=None)) if view==0 else random_view(base,source['source_id'],view,policy)
            images.append(image)
            records.append(dict(source_id=source['source_id'],Y=source['Y'],fold=source['fold'],view=view,
                index=len(records),RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest(),**meta))
    features,enc_seconds=encode(images);x=normalize(features)
    with (ROOT/'train-features.npy').open('xb') as f:np.save(f,features,allow_pickle=False)
    anchors=np.arange(0,576,9);labels=np.array([r['Y'] for r in records]);folds=np.array([r['fold'] for r in records])
    teacher_models={};pred=np.zeros(576,dtype=np.float64)
    for fold in range(4):
        ids=anchors[folds[anchors]!=fold];teacher=fit_logistic(x[ids],labels[ids],np.ones(len(ids)))
        out=np.where(folds==fold)[0];pred[out]=predict(x[out],teacher);teacher_models[str(fold)]=teacher
    keep=((pred>0)==labels).reshape(64,9);keep[:,0]=True
    counts=keep.sum(axis=1);assert (counts>=1).all()
    weights={};anchor_weight=np.zeros(576);anchor_weight[anchors]=1/64
    weights['clean_equal_source']=anchor_weight
    weights['all_views_equal_source']=np.ones(576)/576
    weights['filtered_standard_counts']=keep.astype(float).ravel()/counts.sum()
    weights['all_views_matched_filtered_source_weights']=np.repeat(counts,9)/(9*counts.sum())
    weights['filtered_equal_source']=(keep/(64*counts[:,None])).ravel()
    clean_weight=np.zeros(576);clean_weight[anchors]=counts/counts.sum()
    weights['clean_matched_filtered_source_weights']=clean_weight
    fallback=keep.astype(float)/576;fallback[:,0]+=(9-counts)/576
    weights['filtered_fallback_to_clean_fixed_count']=fallback.ravel()
    assert list(weights)==protocol['arms']
    standard_source=weights['filtered_standard_counts'].reshape(64,9).sum(axis=1)
    matched_source=weights['all_views_matched_filtered_source_weights'].reshape(64,9).sum(axis=1)
    assert np.max(np.abs(standard_source-matched_source))<1e-15
    models={}
    for name,weight in weights.items():
        assert abs(weight.sum()-1)<1e-14 and (weight>=0).all()
        models[name]=fit_logistic(x,labels,weight)
    for i,row in enumerate(records):row.update(teacher_fold_score=float(pred[i]),retained=bool(keep.ravel()[i]))
    save('train-image-manifest.json',dict(records=records,photos=photo_audit,
        no_op_equals_official_V1_tensor_checks=64,source_crossfit=True))
    npweights={k:v.tolist() for k,v in weights.items()}
    save('training-weights.json',dict(arm_weights=npweights,retained_per_source=[dict(source_id=s['source_id'],Y=s['Y'],retained=int(counts[i])) for i,s in enumerate(source_rows)]))
    save('teacher-models.json',teacher_models);save('trained-models.json',models)
    clean_x=x[anchors];clean_y=labels[anchors]
    diagnostics={name:dict(clean_train_LL=float(loss_values(clean_x,clean_y,model).mean()),
        clean_train_accuracy=float(((predict(clean_x,model)>0)==clean_y).mean()),
        weighted_source_ESS=float(1/(weight.reshape(64,9).sum(axis=1)**2).sum()),
        class_weight_mass=[float(weight[labels==y].sum()) for y in [0,1]],
        optimization_gradient_inf_norm=models[name]['gradient_inf_norm']) for (name,model),weight in zip(models.items(),weights.values())}
    save('training-results.json',dict(status='completed_all_fixed_training_arms_before_test_acquisition',at_utc=now(),
        source_photos=64,total_views=576,retained_views=int(keep.sum()),retained_count_min=int(counts.min()),retained_count_max=int(counts.max()),
        rejected_views=int((~keep).sum()),crossfit_teacher_count=4,trained_student_readouts=7,
        source_weight_matching_max_error=float(np.max(np.abs(standard_source-matched_source))),
        encoder_seconds=enc_seconds,wall_seconds=time.perf_counter()-start,diagnostics=diagnostics,
        all_objectives_gradient_gate_passed=True,fresh_test_downloaded=False,CPU_only=True,ARIS=False,GPU=False))
    save('training-freeze.json',dict(at_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),
        training_code_sha256=sha(Path(__file__)),fetch_code_sha256=sha(ROOT/'fetch_fresh_test.py'),
        models_sha256=sha(ROOT/'trained-models.json'),teachers_sha256=sha(ROOT/'teacher-models.json'),
        training_weights_sha256=sha(ROOT/'training-weights.json'),training_features_sha256=sha(ROOT/'train-features.npy'),
        training_manifest_sha256=sha(ROOT/'train-image-manifest.json'),
        all_11_logistic_optimizations_converged=True,fresh_test_pixels_never_acquired=True,
        torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,
        official_TA_forward_source_sha256=hashlib.sha256(inspect.getsource(TrivialAugmentWide.forward).encode()).hexdigest(),
        official_V1_preprocess=repr(official),encoder_device='cpu',threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        encoder_weight_sha256='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',
        backbone_trained=False,no_test_or_validation_hyperparameter_search=True))
    print((ROOT/'training-results.json').read_text(),flush=True)

def evaluate():
    start=time.perf_counter();protocol=json.loads((ROOT/'protocol.json').read_text());freeze=json.loads((ROOT/'training-freeze.json').read_text())
    for key,name in [('protocol_sha256','protocol.json'),('training_code_sha256','train_probe.py'),('models_sha256','trained-models.json'),('training_weights_sha256','training-weights.json')]:assert sha(ROOT/name)==freeze[key]
    assert not (ROOT/'test-results.json').exists()
    receipt=json.loads((ROOT/'test-fetch-receipt.json').read_text());assert not receipt['errors'] and len(receipt['files'])==64
    assert freeze['at_utc']<receipt['at_utc']
    downloaded={r['source_id']:r for r in receipt['files']}
    train_manifest=json.loads((ROOT/'train-image-manifest.json').read_text())
    train_RGB={r['decoded_RGB_sha256'] for r in train_manifest['photos']}
    images=[];rows=[];hashes=set()
    for source in protocol['fresh_test_rows']:
        path=ROOT/'test-photos'/(source['source_id']+'.jpg');check=downloaded[source['source_id']]
        assert sha(path)==check['sha256'] and path.stat().st_size==check['bytes']
        original=Image.open(path).convert('RGB');digest=hashlib.sha256(original.tobytes()).hexdigest()
        assert digest not in train_RGB and digest not in hashes,'Exact decoded photo duplication; stop without test interpretation'
        hashes.add(digest);image=base_image(original);images.append(image)
        rows.append(dict(source_id=source['source_id'],Y=source['Y'],breed_id=source['breed_id'],
            JPEG_sha256=check['sha256'],decoded_RGB_sha256=digest,width=original.width,height=original.height,
            model_RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    features,enc_seconds=encode(images);x=normalize(features);y=np.array([r['Y'] for r in rows])
    with (ROOT/'test-features.npy').open('xb') as f:np.save(f,features,allow_pickle=False)
    models=json.loads((ROOT/'trained-models.json').read_text());table={};predictions={};losses={}
    for name,model in models.items():
        zz=predict(x,model);ll=loss_values(x,y,model);losses[name]=ll
        table[name]=dict(original_test_photos=len(rows),mean_LL_nats=float(ll.mean()),accuracy=float(((zz>0)==y).mean()))
        predictions[name]=[dict(source_id=r['source_id'],Y=int(y[i]),score=float(zz[i]),loss=float(ll[i]),correct=bool((zz[i]>0)==y[i])) for i,r in enumerate(rows)]
    filt='filtered_standard_counts';match='all_views_matched_filtered_source_weights';clean='clean_equal_source'
    delta=losses[match]-losses[filt]
    rng=np.random.default_rng(2026102106)
    samples=np.concatenate([rng.choice(np.where(y==c)[0],size=(5000,32),replace=True) for c in [0,1]],axis=1)
    boot=delta[samples].mean(axis=1);CI=np.quantile(boot,[.025,.975])
    gains={name:float((losses[name]-losses[filt]).mean()) for name in models if name!=filt}
    gates=dict(primary_gain_at_least001=float(delta.mean())>=.01,
        paired_bootstrap95_lower_positive=float(CI[0])>0,filtered_better_than_clean=float(losses[filt].mean())<float(losses[clean].mean()))
    out=dict(status='completed_once_on64_fresh_original_photos',at_utc=now(),table=table,
        primary_benefit_LL_matched_source_control_minus_filtered=float(delta.mean()),primary_paired_species_bootstrap95_CI=CI.tolist(),
        gains_LL_other_arm_minus_filtered=gains,gates=gates,screen_passed=all(gates.values()),
        fresh_test_used_for_hypothesis_or_hyperparameters=False,test_is_unaugmented=True,
        test_source_photos=64,exact_train_test_RGB_duplicates=0,encoder_seconds=enc_seconds,
        wall_seconds=time.perf_counter()-start,model_fit_after_test=False,CPU_only=True,ARIS=False,GPU=False,
        interpretation='Frozen-feature logistic training only, not end-to-end representation learning or causal identification. Bootstrap covers fresh-photo sampling conditional on the fixed64training photos and one augmentation draw bank; animal identity and pretraining independence unknown.')
    save('test-image-manifest.json',dict(rows=rows));save('test-predictions.json',predictions);save('test-results.json',out)
    save('test-consumption.json',dict(status='consumed_for_this_fixed_question',at_utc=now(),source_ids=[r['source_id'] for r in rows],
        future_use='Do not claim these64photos are untouched confirmation for a new hypothesis or hyperparameter choice.',
        training_freeze_sha256=sha(ROOT/'training-freeze.json'),test_results_sha256=sha(ROOT/'test-results.json')))
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['train','test']
    try:os.nice(10)
    except OSError:pass
    signal.alarm(180);setup()
    if sys.argv[1]=='train':train()
    else:evaluate()
