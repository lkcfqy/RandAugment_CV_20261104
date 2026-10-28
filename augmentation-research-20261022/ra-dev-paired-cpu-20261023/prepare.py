"""Frozen, CPU-only paired-view probe. Never trains or reads reserved data."""
import time
START=time.perf_counter()
import os,sys,json,hashlib,math,resource,traceback,importlib.util
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PILOT=ROOT/'augmentation-research-20261022/pilot-20261022'
PHASE=sys.argv[1]
LIMIT=float(sys.argv[2])
assert PHASE=='prepare'
LOG=(OUT/(PHASE+'-events.jsonl')).open('x',buffering=1)
COUNT=0
FORWARD_WALL=0.
FORWARD_CPU=0.

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def event(kind,**kw):
    LOG.write(json.dumps(dict(event=kind,worker_wall_seconds=time.perf_counter()-START,**kw),ensure_ascii=False,allow_nan=False)+'\n')
    print(kind,kw.get('source_id',''),kw.get('arm',''),kw.get('view',''),flush=True)

def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def guard():
    if (OUT/'STOP.request').exists():raise RuntimeError('Explicit stop marker')
    if time.perf_counter()-START>=LIMIT:raise RuntimeError('Worker wall budget exhausted')
    if rss()>4294967296:raise RuntimeError('Peak RSS exceeds 4GiB')
    if FORWARD_WALL>=2 and FORWARD_WALL>2*FORWARD_CPU:raise RuntimeError('Own forward wall/CPU contention gate triggered')

def effective(op,m):
    if op in ('ShearX','ShearY'):return {'shear_angle_degrees':math.degrees(math.atan(m)),'center':[0,0]}
    if op in ('TranslateX','TranslateY'):return {'translation_pixels':int(m),'axis':op[-1]}
    if op=='Rotate':return {'rotation_degrees':m}
    if op in ('Brightness','Color','Contrast','Sharpness'):return {'factor':1+m}
    if op=='Posterize':return {'bits':int(m)}
    if op=='Solarize':return {'threshold':m}
    return {'magnitude_ignored':True}

def metrics(z,y,j):
    mx=max(z);ex=[math.exp(v-mx) for v in z];den=math.fsum(ex);p=[v/den for v in ex]
    rival=max((k for k in range(37) if k!=y),key=lambda k:z[k])
    pred=max(range(37),key=lambda k:z[k])
    return dict(p_true=p[y],fixed_rival=j,p_fixed_rival=p[j],fixed_rival_margin=z[y]-z[j],
        dynamic_rival=rival,p_dynamic_rival=p[rival],dynamic_rival_margin=z[y]-z[rival],
        max_confidence=p[pred],prediction=pred,correct=pred==y,nll=mx-z[y]+math.log(den))

def main():
    assert PHASE=='prepare' and LIMIT<=10
    n=os.getpriority(os.PRIO_PROCESS,0)
    if n<10:os.nice(10-n)
    event('startup',nice=os.getpriority(os.PRIO_PROCESS,0))
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol-lock.json').read_text())['sha256']
    assert sha(OUT/'selection.json')==p['selection_sha256']
    for name,h in json.loads((OUT/'code-lock.json').read_text())['sha256'].items():assert sha(OUT/name)==h
    for rel,h in p['input_and_preservation_sha256'].items():assert sha(ROOT/rel)==h,rel;guard()
    rows=json.loads((OUT/'selection.json').read_text());assert len(rows)==16
    manifest=json.loads((PILOT/'fit-manifest.json').read_text());dev={r['source_id']:r for r in manifest['rows'] if r['candidate_partition']=='development'}
    for r in rows:
        for k in ('class_id','species','image_sha256','decoded_RGB_sha256','width','height'):assert dev[r['source_id']][k]==r[k]
    import torch
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    from torchvision.transforms import autoaugment as AA
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('paired_probe_original_common',PILOT/'common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    event('imports_complete')
    pairs={};preptimes=[]
    for r in rows:
        t=time.perf_counter();sid=r['source_id'];path=ROOT/r['image_path']
        assert sha(path)==r['image_sha256']
        with Image.open(path) as f:
            im=f.convert('RGB');im.load()
        decoded=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        assert decoded==r['decoded_RGB_sha256'] and im.size==(r['width'],r['height'])
        assert common.keyseed('weak-v1',20261022,0,sid)==r['weak_seed']
        assert common.keyseed('RA-v1',20261022,0,sid)==r['ra_seed']
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(r['weak_seed'])
            ci,cj,ch,cw=common.T.RandomResizedCrop.get_params(im,scale=(.8,1.),ratio=(.9,1.1))
            flip=bool(torch.rand(())<.5)
        weak_trace=dict(top=ci,left=cj,height=ch,width=cw,output_size=[224,224],horizontal_flip=flip,seed=r['weak_seed'])
        weak=common.augmented_pil(im,sid,0,20261022,'weak')
        tracedweak=common.TF.resized_crop(im,ci,cj,ch,cw,(224,224),common.InterpolationMode.BILINEAR,antialias=True)
        if flip:tracedweak=common.TF.hflip(tracedweak)
        assert weak.tobytes()==tracedweak.tobytes()
        ops=[];original_apply=AA._apply_op
        def capture(image,op_name,magnitude,interpolation,fill):
            ops.append(dict(operation=op_name,signed_magnitude=float(magnitude),effective=effective(op_name,float(magnitude)),interpolation=str(interpolation),fill=fill))
            return original_apply(image,op_name,magnitude,interpolation,fill)
        AA._apply_op=capture
        try:aug=common.augmented_pil(im,sid,0,20261022,'ra')
        finally:AA._apply_op=original_apply
        ref=common.augmented_pil(im,sid,0,20261022,'ra')
        assert len(ops)==2 and aug.mode==ref.mode and aug.size==ref.size and aug.tobytes()==ref.tobytes()
        directory=OUT/'views'/sid;directory.mkdir(parents=True,exist_ok=False)
        hashes={}
        for name,pic in (('source',im),('weak',weak),('ra',aug)):
            pic.save(directory/(name+'.png'))
            hashes[name]=dict(file_sha256=sha(directory/(name+'.png')),decoded_RGB_sha256=hashlib.sha256(str(pic.size).encode()+pic.tobytes()).hexdigest(),size=list(pic.size))
        xs={name:common.tensor(pic).contiguous() for name,pic in (('weak',weak),('ra',aug))}
        th={name:hashlib.sha256(x.numpy().tobytes()).hexdigest() for name,x in xs.items()}
        assert all(x.device.type=='cpu' and x.dtype==torch.float32 and tuple(x.shape)==(3,224,224) for x in xs.values())
        a=np.asarray(weak).astype(np.float64);b=np.asarray(aug).astype(np.float64)
        elapsed=time.perf_counter()-t;preptimes.append(elapsed)
        pair=dict(source=r,weak_trace=weak_trace,dataset_attribution='Existing Oxford-IIIT Pet project development whitelist; local evidence only',N=2,M=9,magnitude_bins=31,epoch=0,training_seed=20261022,operations=ops,original_uninstrumented_RA_pixel_parity=True,
            image_hashes=hashes,tensor_sha256=th,pixel_MAE_0_255=float(np.abs(a-b).mean()),pixel_changed_fraction=float(np.any(a!=b,axis=2).mean()),preparation_wall_seconds=elapsed,
            semantic_label_preservation_verified=False)
        dump(directory/'pair.json',pair)
        pairs[sid]=dict(tensors=xs,tensor_sha256=th)
        event('pair_prepared',source_id=sid,seconds=elapsed);guard()
    # Existing clean predictions: explicit preprocessing distinction; no new forward.
    baseline=[]
    for epoch in (20,30):
        for arm in ('front-weak','front-ra'):
            path=PILOT/'results'/arm/('development-epoch%03d.npz'%epoch)
            with np.load(path,allow_pickle=False) as cache:
                ids=list(cache['source_ids']);assert len(ids)==589
                for r in rows:
                    idx=ids.index(r['source_id']);z=cache['logits'][idx].tolist();y=int(cache['labels'][idx]);assert y==r['class_id']
                    j=max((i for i in range(37) if i!=y),key=lambda i:z[i])
                    baseline.append(dict(source_id=r['source_id'],group=r['group'],class_id=y,arm=arm,epoch=epoch,logits=z,metrics=metrics(z,y,j),source_cache=str(path.relative_to(ROOT))))
    dump(OUT/'cached-clean-baseline.json',dict(input_transform='Original eval_tensor:resize256bilinear,centercrop224;not new weak randomcrop/flip',rows=baseline))
    # Diagnostic contact sheets retain aspect ratio; source PNGs remain untouched.
    sheet_paths=[]
    for s in range(4):
        sheet=Image.new('RGB',(936,4*264),(245,245,245));draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(rows[4*s:4*s+4]):
            y0=k*264;draw.text((6,y0+3),r['source_id']+' | '+r['group'],fill='black')
            for col,view in enumerate(('source','weak','ra')):
                draw.text((col*312+6,y0+20),view,fill='black')
                with Image.open(OUT/'views'/r['source_id']/(view+'.png')) as im:
                    thumb=ImageOps.contain(im,(300,224));sheet.paste(thumb,(col*312+(312-thumb.width)//2,y0+38))
        path=OUT/('contact-sheet-%d.png'%s);sheet.save(path);sheet_paths.append(path.name);guard()
    locked={}
    for r in rows:
        directory=OUT/'views'/r['source_id']
        for filename in ('pair.json','source.png','weak.png','ra.png'):
            path=directory/filename;locked[str(path.relative_to(OUT))]=sha(path)
    locked['cached-clean-baseline.json']=sha(OUT/'cached-clean-baseline.json')
    for name in sheet_paths:locked[name]=sha(OUT/name)
    dump(OUT/'trajectory-lock.json',dict(frozen_before_any_new_model_forward=True,sha256=locked,rows=16,seed=20261022,epoch=0))
    guard()
    dump(OUT/'prepare-result.json',dict(status='complete',completed_forwards=0,prepared_pairs=16,worker_wall_seconds=time.perf_counter()-START,peak_RSS_bytes=rss(),trajectory_lock_sha256=sha(OUT/'trajectory-lock.json')))
    event('complete',forwards=0)
try:main()
except BaseException as error:
    record=dict(status='stopped',phase=PHASE,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),completed_forwards=COUNT,
        worker_wall_seconds=time.perf_counter()-START,forward_wall_seconds=FORWARD_WALL,forward_cpu_seconds=FORWARD_CPU,peak_RSS_bytes=rss())
    dump(OUT/(PHASE+'-failure.json'),record);event('stopped',error=str(error));raise
finally:LOG.close()
