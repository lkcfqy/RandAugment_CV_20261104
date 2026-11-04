"""Freeze assets and metadata before any diagnostic model forward."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib, json, datetime, random, math, subprocess

P=Path(__file__).resolve().parent
R=_portable_project_path(__file__, '')
SALT='pe-erasing-diagnostic-20261102-v1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def key(*s):return int.from_bytes(hashlib.sha256((SALT+'|'+'|'.join(map(str,s))).encode()).digest()[:8],'big')&((1<<63)-1)
def dump(n,x): (P/n).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
assert not (P/'PROTOCOL.json').exists()
pool=sorted((R/'research-counting-exemplar-20261031/data/images').glob('*.jpg'))
assert len(pool)==94
selected=sorted(pool,key=lambda p:(key('image',p.name),p.name))[:32]
images=[]
for q in selected:
    rng=random.Random(key('rectangle',q.name));rng.random();a=rng.uniform(.2,.2)*224**2;ratio=math.exp(rng.uniform(0,0))
    h=round(math.sqrt(a*ratio));w=round(math.sqrt(a/ratio));top=rng.randint(0,224-h);left=rng.randint(0,224-w)
    images.append({'id':q.stem,'filename':q.name,'path':str(q),'sha256':sha(q),'bytes':q.stat().st_size,
                   'rectangle_tlhw':[top,left,h,w],'rectangle_seed':key('rectangle',q.name),'noise_seed':key('noise',q.name)})
weights=[('deit_tiny',R/'research-view-route-20261030/assets/deit_tiny_patch16_224-a1311bcf.pth'),
         ('dinov2_small',R/'research-shared-theta-ad-20261030/models/dinov2-small/model.safetensors')]
models={n:{'path':str(q),'bytes':q.stat().st_size,'sha256':sha(q)} for n,q in weights}
assert models['deit_tiny']['sha256'].startswith('a1311bcf')
assert models['dinov2_small']['sha256']=='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
started=datetime.datetime(2026,11,2,2,45,tzinfo=datetime.timezone.utc)  # conservative task-wide budget origin
proto={'status':'FROZEN_BEFORE_DIAGNOSTIC_MODEL_OUTPUTS','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'budget_start_UTC':started.isoformat(),'deadline_epoch':started.timestamp()+7200,'wall_limit_seconds':7200,
 'new_artifact_limit_bytes':200*1024**2,'host_RSS_stop_bytes':12*1024**3,'device':'cpu','torch_threads':4,'interop_threads':1,
 'image_pool':'94 already-local public PairTally images; reuse pixels only, not counting experiments/labels; correlated pairs possible',
 'selection_rule':'32 smallest SHA256-derived keys of fixed salt|image|filename; no image content/label/effect selection',
 'salt':SALT,'images':images,'models':models,'modes':['none','pixel','rand','matched_correlated'],
 'preprocessing':'PIL RGB -> torchvision Resize(shorter=256,bicubic) -> CenterCrop224 -> ToTensor -> Normalize ImageNet mean/std; no input clipping after erasing',
 'mean':[.485,.456,.406],'std':[.229,.224,.225],
 'erasing':{'implementation':'installed official timm1.0.22 RandomErasing','probability':1.,'min_area':.2,'max_area':.2,'min_aspect':1.,'max_aspect':1.,'min_count':1,'max_count':1,'actual_rectangle_hw':[100,100],'actual_area_fraction':10000/50176,'interpretation':'conditional on one erasure; not a training frequency recommendation'},
 'matched_correlated':'For each pixel-mode erased channel, reflect-pad then smooth with fixed cross kernel [[0,1,0],[1,4,1],[0,1,0]]/8; assign original sorted pixel values in smoothed rank order. Exact per-channel multiset (thus all scalar moments) retained. Artificial diagnostic control, not proposed augmentation.',
 'alphas':[.99,1.,1.01],'output_fd_halfstep':.01,'secondary_fd_halfstep_first_image_only':.005,
 'statistics':{'paper_input':'population variance ddof0 of all C,H,W entries of one normalized image, flattened; also spatial per-channel variance reported separately',
 'paper_tokens':'population variance ddof0 of all N,D patch projection entries, excluding CLS, real projection bias retained',
 'LN_axis':'per-token feature-dimension D variance; report all patch tokens, fully erased patch subset and unaffected subset separately; CLS excluded from primary',
 'sensitivity':'absolute RMS over patch tokens/features of d LN(X+alpha P)/dalpha at alpha1, including learned gamma; compare analytic to central finite difference h1e-4 in float64 for every model/image/mode',
 'scalar_proxy_prediction':'relative LN sensitivity approx sqrt(VI_none/VI_mode), evaluated descriptively, not claimed an exact theorem',
 'output':'final normalized CLS representation, cosine change and RMS relative change; symmetric PE finite-difference response norm / representation norm; DeiT1000 logits top1/top5 also descriptive',
 'ranking':'top5 cosine neighbours among all32 clean final CLS representations excluding same image; no class labels or accuracy'},
 'decision_gate':{'scope':'heuristic engineering relevance thresholds, not statistical significance or training benefit',
 'ordinary_gate_both_models':'Across pixel/rand vs none, median absolute log error of scalar-proxy relative first-LN sensitivity >=log1.10 AND pixel-vs-rand sensitivity ordering disagreement >=25% of32 images (proxy or actual relative difference<1% counts as no mismatch).',
 'matched_gate_both_models':'Median absolute log token-variance ratio and first-LN sensitivity ratio matched/pixel each >=log1.10, with per-channel multiset identity verified.',
 'output_gate_both_models':'Median absolute log full-output PE-sensitivity ratio matched/pixel >=log1.10 AND (median cosine distance of matched vs pixel CLS >=.01 OR median clean-neighbour top5 Jaccard distance >=.20).',
 'outcome':'Only if every gate holds for both models consider a future training proposal; otherwise close this probe. No new training is authorized. Covariance differences alone are insufficient.'},
 'limits':['frozen perturbation diagnosis only, not train/test accuracy or augmentation efficacy','PairTally is not ImageNet; dataset normalization does not imply per-image variance1','DINOv2 uses its official fixed224 positional interpolation; no resolution change across conditions','paper final AAAI PDF inaccessible; preprint definitions verified, exact final page not independently verified','linear covariance identity W Sigma W^T is established algebra, not novelty'],
 'new_weights_download_install_upload_GPU_MPS_Colab_CU':0,'expected_unique_model_image_mode_identities':256,'expected_core_forwards':768,
 'extra_identity_forwards':2,'extra_output_fd_test_forwards':16}
dump('PROTOCOL.json',proto)
dump('ASSETS.json',{'images_count':32,'images':images,'models':models,'source_pins':{
 'PairTally_asset_manifest':sha(R/'research-counting-exemplar-20261031/reports/asset_manifest.json'),
 'DeiT_download_receipts':sha(R/'research-view-route-20261030/sources/receipts.json'),
 'DINOv2_verification':sha(R/'research-shared-theta-ad-20261030/reports/model-cpu-verification.json'),
 'DINOv2_config':sha(R/'research-shared-theta-ad-20261030/models/dinov2-small/config.json'),
 'timm_erasing':sha(R/'research-view-route-20261030/deps/timm/data/random_erasing.py')}})
print(json.dumps({'status':proto['status'],'model_hashes':models,'selected_images':len(images),'protocol_sha256':sha(P/'PROTOCOL.json'),'deadline_UTC':datetime.datetime.fromtimestamp(proto['deadline_epoch'],datetime.timezone.utc).isoformat()}))
