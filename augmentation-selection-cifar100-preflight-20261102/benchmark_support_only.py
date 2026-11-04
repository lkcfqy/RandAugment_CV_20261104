"""Bounded support-image throughput, synthetic ridge and persistence checks only."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,hashlib,importlib.util,resource
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parent
PROJECT=_portable_project_path(__file__, '')
name=sys.argv[1];assert name in ['dinov2_small','deit_tiny']
source=PROJECT/('augmentation-selection-pilot-20261102' if name=='dinov2_small' else 'augmentation-selection-deit-20261102')
spec=importlib.util.spec_from_file_location('frozen_common',source/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
np,torch,Image=c.np,c.torch,c.Image
c.SALT='augmentation-selection-cifar100-20261102-v1'
identity=json.loads((ROOT/'IDENTITIES.json').read_text());audit=json.loads((ROOT/'ASSET_AUDIT.json').read_text())
assert audit['status']=='PASS' and c.sha(ROOT/'IDENTITIES.json')==audit['identities_sha256']
raw=np.load(PROJECT/'research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy',mmap_mode='r',allow_pickle=False)
lookup={r['id']:r for r in identity['support_union']}
rows=[lookup[i] for i in identity['timing_support_ids']]
assert len(rows)==8 and all(r['id'].startswith('train/') for r in rows)
t=time.perf_counter();tensors=[];viewhash=[]
for r in rows:
    image=Image.fromarray(raw[r['index']].copy())
    v=c.make_views(image,r['rgb_hash'])
    tensors.extend(v)
    viewhash.append({'id':r['id'],'clean':hashlib.sha256(v[0].numpy().tobytes()).hexdigest(),
                     'weak_ra':[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in v[1:]]})
transform_seconds=time.perf_counter()-t
t=time.perf_counter();model=c.load_model();load_seconds=time.perf_counter()-t
expected_state={'dinov2_small':'64fa8e4e6a2af982552f7d63246f8eeb61e9c02980689f3d109747fee783ebef',
                'deit_tiny':'2f4ccb35d484075cf3e1197bcf34e4966998552c260e153c94bbd321c7ee3e6f'}
before=c.state_hash(model);assert before==expected_state[name]
assert not model.training and not any(p.requires_grad for p in model.parameters())
batch=[tensors[i%len(tensors)] for i in range(16)]
warm=c.encode(model,batch);assert np.isfinite(warm).all()
times=[]
for i in range(5):
    batch=[tensors[(16*(i+1)+j)%len(tensors)] for j in range(16)]
    t=time.perf_counter();out=c.encode(model,batch);times.append(time.perf_counter()-t)
    assert np.isfinite(out).all()
after=c.state_hash(model);assert before==after
dimension=384 if name=='dinov2_small' else 192
assert warm.shape==(16,dimension)
del model,out,warm,tensors,batch

# All ridge inputs below are artificial; no CIFAR labels or real features are used.
rng=np.random.default_rng(20261102)
x=rng.normal(size=(1600,4,dimension)).astype(np.float32)
x/=np.linalg.norm(x,axis=-1,keepdims=True);y=np.arange(1600)%100
fit_times=[]
for _ in range(4):
    t=time.perf_counter();W,b=c.ridge(x,y,classes=100);fit_times.append(time.perf_counter()-t)
assert W.shape==(dimension,100) and b.shape==(100,)
q=rng.normal(size=(9990,dimension)).astype(np.float32);q/=np.linalg.norm(q,axis=-1,keepdims=True)
t=time.perf_counter();blocks=[]
for start in range(0,len(q),256):blocks.append(c.scores(q[start:start+256],W,b).argmax(1).astype(np.uint8))
score_seconds=time.perf_counter()-t;prediction=np.concatenate(blocks)
assert np.array_equal(prediction,c.scores(q,W,b).argmax(1))

# Exact native-precision persistence; score recomputation uses the same fixed blocks.
saved=ROOT/(name+'_synthetic_persistence.npz')
t=time.perf_counter();np.savez(saved,feature=q,W=W,b=b,prediction=prediction);write_seconds=time.perf_counter()-t
with np.load(saved,allow_pickle=False) as z:
    assert np.array_equal(z['feature'],q) and np.array_equal(z['W'],W) and np.array_equal(z['b'],b)
    re=np.concatenate([c.scores(z['feature'][i:i+256],z['W'],z['b']).argmax(1).astype(np.uint8) for i in range(0,len(q),256)])
    assert np.array_equal(re,z['prediction'])

# Duplicate identical views preserve equal original weight for a 100-class fixture.
fixture=rng.normal(size=(100,1,dimension)).astype(np.float32);fy=np.arange(100)
Wa,ba=c.ridge(fixture,fy,classes=100);Wb,bb=c.ridge(np.repeat(fixture,4,axis=1),fy,classes=100)
assert np.allclose(Wa,Wb,atol=1e-11,rtol=1e-10) and np.allclose(ba,bb,atol=1e-11,rtol=1e-10)
result={'status':'PASS_NO_QUERY_UTILITY','encoder':name,'utc':datetime.now(timezone.utc).isoformat(),
        'source_common_sha256':c.sha(source/'common.py'),'source_weight_sha256':audit['models'][name]['sha256'],
        'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'device':'cpu',
        'timing_originals':8,'originals_all_from_support':True,'model_image_forwards':96,'real_head_fits':0,'query_image_forwards':0,
        'transform_seconds_8_originals':transform_seconds,'load_seconds':load_seconds,'warmup_images':16,'batch_size':16,
        'timed_batch_seconds':times,'timed_batch_p90_seconds':float(np.percentile(times,90)),
        'initial_state_hash':before,'final_state_hash':after,'feature_dimension':dimension,
        'synthetic_ridge_shape':[1600,4,dimension,100],'synthetic_ridge_fit_seconds_including_warmup':fit_times,
        'synthetic_ridge_timed_p90_seconds':float(np.percentile(fit_times[1:],90)),
        'synthetic_ridge_calls':6,'synthetic_query_score_one_head_seconds':score_seconds,'synthetic_query_count':9990,
        'chunk_size':256,'all_synthetic_block_predictions_match_dense':True,'native_precision_persistence_exact':True,
        'synthetic_persistence_bytes':saved.stat().st_size,'synthetic_persistence_write_seconds':write_seconds,
        'synthetic_persistence_sha256':c.sha(saved),'unit_original_weight_duplicate_view_test':'PASS',
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'views':viewhash,'new_CU':0}
(ROOT/(name+'_BENCHMARK.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='views'},indent=2))
