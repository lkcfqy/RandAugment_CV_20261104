"""CPU-only invariant checks, synthetic tiny images; no model computation."""
from core import *
import shutil,tempfile
from PIL.PngImagePlugin import PngInfo
import manifest

results={}
# Independent key scopes must preserve the caller's CPU RNG and each other's views.
pixels=np.arange(64*80*3,dtype=np.uint8).reshape(64,80,3);im=Image.fromarray(pixels)
old=torch.get_rng_state().clone()
def views(with_ra):
    with cpu_rng(key('weak',11,0,0,'fixture')):w=WEAK(im.copy())
    s=None
    if with_ra:
        with cpu_rng(key('ra',11,0,0,'fixture')):s=RA[9](w.copy())
    return w.tobytes(),None if s is None else s.tobytes()
a=views(True);views(False);b=views(True)
assert a==b and torch.equal(old,torch.get_rng_state())
assert len({key(d,11,0,0,'fixture') for d in ['data','weak','ra','routing','headinit']})==5
results['rng_branch_independence_and_restore']=True
d=Data.__new__(Data);d.splits={'support':{'11':[f'id{i:04d}' for i in range(592)]}};d.order_cache={}
first=sum([d.ids(11,u) for u in range(18)],[])
assert len(first)==576 and len(set(first))==576 and all(len(d.ids(11,u))==32 for u in range(100))
assert first!=sum([d.ids(11,u) for u in range(18,36)],[])
results['shuffle_592_drop_last_18_batches']=True
for n in [128,129,192,512,1000]:
    w=math.ceil(.05*n)
    assert abs(multiplier(1,n)-1/w)<1e-12 and multiplier(w,n)==1 and abs(multiplier(n,n)-.1)<1e-12
results['schedule_endpoints']=True
space=RA[9]._augmentation_space(31,(224,224))
results['native_randaugment_pool']={op:{'signed':signed,'shape':list(vals.shape)} for op,(vals,signed) in space.items()}
results['ent_old_cache_magnitudes']=[min(9,max(0,math.floor(9*q+.5))) for q in [0.,.05,.1,.5,.95,1.]]
assert results['ent_old_cache_magnitudes']==[0,0,1,5,9,9]

# Synthetic split fixture: class balance survives exact/pixel/cross-split/conflict exclusions.
fixture=ROOT/'fixtures'/'manifest-case';assert not fixture.exists()
(fixture/'data'/'images').mkdir(parents=True);(fixture/'data'/'annotations').mkdir()
ann={'trainval':[],'test':[]}
def add(ident,cls,split,pixel=None,meta=False):
    if pixel is None:
        v=key('fixture-pixels',ident)
        pixel=np.full((8,8,3),[(v>>i)&255 for i in [0,8,16]],dtype=np.uint8)
        pixel[0,0]=[cls, len(ann[split])%256,split=='test']
    p=fixture/'data'/'images'/f'{ident}.jpg';pnginfo=PngInfo()
    if meta:pnginfo.add_text('different_file_bytes','identical_rgb')
    Image.fromarray(pixel).save(p,format='PNG',pnginfo=pnginfo)
    ann[split].append(f'{ident} {cls} 1 {cls}')
    return pixel
for cls in range(1,38):
    for j in range(35):add(f'c{cls:02d}_{j:03d}',cls,'trainval')
    for j in range(2):add(f'c{cls:02d}_test{j}',cls,'test')
def pix(ident):
    with Image.open(fixture/'data'/'images'/f'{ident}.jpg') as im:return np.array(im.convert('RGB'))
add('c01_test_duplicate',1,'test',pix('c01_000'))
add('c02_zzz_duplicate',2,'trainval',pix('c02_000'))
add('c03_zzz_pixel_duplicate',3,'trainval',pix('c03_000'),meta=True)
add('c05_label_conflict',5,'trainval',pix('c04_000'))
add('c06_test_zduplicate',6,'test',pix('c06_test0'))
for split,lines in ann.items():(fixture/'data'/'annotations'/f'{split}.txt').write_text('\n'.join(lines)+'\n')
manifest.ROOT=fixture;manifest.main()
rows={r['image_id']:r for r in json.loads((fixture/'data-manifest.json').read_text())}
assert not rows['c01_000']['included'] and rows['c01_test_duplicate']['included']
assert rows['c02_000']['included'] and not rows['c02_zzz_duplicate']['included']
assert rows['c03_000']['file_sha256']!=rows['c03_zzz_pixel_duplicate']['file_sha256']
assert rows['c03_000']['rgb_dimensions_sha256']==rows['c03_zzz_pixel_duplicate']['rgb_dimensions_sha256']
assert not rows['c03_zzz_pixel_duplicate']['included']
assert rows['c04_000']['exclusion_reason']==rows['c05_label_conflict']['exclusion_reason']=='label_conflict'
assert rows['c06_test0']['included'] and not rows['c06_test_zduplicate']['included']
summary=json.loads((fixture/'data-validation.json').read_text());assert summary['validation_count']==592
assert all(v==592 for v in summary['support_per_seed'].values())
results['synthetic_manifest_dedup_and_split']=summary
dump(ROOT/'implementation-fixtures.json',{'status':'passed','cpu_only':True,'real_dataset_images':0,'model_computation':False,'checks':results,
     'source_hashes':{p:sha(ROOT/p) for p in ['core.py','manifest.py','pilot.py','fixtures.py']}})
print('CPU_IMPLEMENTATION_FIXTURES_PASSED',flush=True)
