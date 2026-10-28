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
assert PHASE in ('preflight','continuation')
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
    global COUNT,FORWARD_WALL,FORWARD_CPU
    assert sys.platform=='darwin','RSS convention frozen for this Mac'
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice<10:os.nice(10-nice)
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    event('startup',phase=PHASE,nice=os.getpriority(os.PRIO_PROCESS,0),limit=LIMIT)
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol-lock.json').read_text())['sha256']
    assert sha(OUT/'selection.json')==protocol['selection_sha256']
    code_lock=json.loads((OUT/'code-lock.json').read_text())
    for name,h in code_lock['sha256'].items():assert sha(OUT/name)==h,name
    verified={}
    for rel,h in protocol['input_and_preservation_sha256'].items():
        got=sha(ROOT/rel);assert got==h,rel;verified[rel]=got;guard()
    allrows=json.loads((OUT/'selection.json').read_text())
    manifest=json.loads((PILOT/'fit-manifest.json').read_text())
    whitelist={r['source_id']:r for r in manifest['rows'] if r['candidate_partition']=='development'}
    assert len(whitelist)==589 and len(allrows)==16
    assert [r['source_id'] for r in allrows]==protocol['selection_ids']
    for r in allrows:
        original=whitelist[r['source_id']]
        for key in ('class_id','species','candidate_partition','image_sha256','decoded_RGB_sha256','width','height'):
            assert r[key]==original[key],(r['source_id'],key)
        assert ROOT/r['image_path']==PILOT/'data/images'/(r['source_id']+'.jpg')
    prep=json.loads((OUT/'prepare-result.json').read_text())
    assert prep['completed_forwards']==0 and prep['prepared_pairs']==16
    assert sha(OUT/'trajectory-lock.json')==prep['trajectory_lock_sha256']
    trajectory=json.loads((OUT/'trajectory-lock.json').read_text())
    assert trajectory['frozen_before_any_new_model_forward']
    for rel,h in trajectory['sha256'].items():assert sha(OUT/rel)==h,rel
    if PHASE=='preflight':
        assert LIMIT<=10
        rows=allrows[:2];max_forwards=8
    else:
        admission=json.loads((OUT/'continuation-admission.json').read_text())
        assert admission['admitted'] is True and LIMIT<=admission['remaining_worker_wall_seconds']
        prior=json.loads((OUT/'preflight-result.json').read_text())
        assert prior['completed_forwards']==8
        FORWARD_WALL=prior['forward_wall_seconds'];FORWARD_CPU=prior['forward_cpu_seconds']
        rows=allrows[2:];max_forwards=56
    dump(OUT/(PHASE+'-input-receipt.json'),dict(verified=verified,selection_sha256=sha(OUT/'selection.json'),development_whitelist_count=589,selected_source_ids=[r['source_id'] for r in rows],sealed_read=False))
    guard()
    import torch
    import numpy as np
    from PIL import Image
    from torchvision.transforms import autoaugment as AA
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.get_num_threads()==torch.get_num_interop_threads()==1
    spec=importlib.util.spec_from_file_location('paired_probe_original_common',PILOT/'common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    event('imports_complete',torch=torch.__version__)
    guard()
    pairs={};preptimes=[]
    for r in rows:
        t=time.perf_counter();sid=r['source_id'];directory=OUT/'views'/sid
        metadata=json.loads((directory/'pair.json').read_text());assert metadata['source']==r
        xs={}
        for view in ('weak','ra'):
            with Image.open(directory/(view+'.png')) as im:xs[view]=common.tensor(im).contiguous()
        for view,x in xs.items():
            assert x.device.type=='cpu' and x.dtype==torch.float32 and tuple(x.shape)==(3,224,224)
            assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==metadata['tensor_sha256'][view]
        pairs[sid]=dict(tensors=xs,tensor_sha256=metadata['tensor_sha256'])
        preptimes.append(time.perf_counter()-t);guard()
    model_start=time.perf_counter()
    model=common.build_model('front',20261022)
    params=dict(model.named_parameters());trainkeys={k for k,v in params.items() if v.requires_grad}
    assert sum(params[k].numel() for k in trainkeys)==65317
    def frozen_digest():
        h=hashlib.sha256()
        for name,p in model.named_parameters():
            if name in trainkeys:continue
            h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    frozen=frozen_digest()
    for arm in ('front-weak','front-ra'):
        summary=json.loads((PILOT/'results'/arm/'summary.json').read_text())
        assert frozen==summary['frozen_parameter_SHA256_before']==summary['frozen_parameter_SHA256_after']
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
    model_build_seconds=time.perf_counter()-model_start
    event('model_verified',frozen_sha256=frozen,model_build_seconds=model_build_seconds);guard()
    recdir=OUT/'forward-records';recdir.mkdir(exist_ok=True)
    jrefs={};ownrefs={};times=[];restores=[]
    for arm in ('front-weak','front-ra'):
        cpath=PILOT/'results'/arm/'trainable-checkpoint.pt'
        ck=torch.load(cpath,map_location='cpu',weights_only=True)
        assert ck['layout']=='front' and ck['augmentation']==arm.split('-')[1] and ck['seed']==20261022
        assert ck['protocol_sha256']==sha(PILOT/'protocol.json')
        state=ck['trainable_state'];assert set(state)==trainkeys
        with torch.no_grad():
            for name,v in state.items():
                assert v.device.type=='cpu' and v.dtype==params[name].dtype and v.shape==params[name].shape
                params[name].copy_(v);assert torch.equal(params[name],v)
        del ck,state
        restores.append(dict(arm=arm,checkpoint_sha256=sha(cpath),restored_parameter_count=len(trainkeys),restored_scalar_count=65317,exact_values_verified=True))
        event('checkpoint_restored',arm=arm);guard()
        for r in rows:
            sid=r['source_id'];y=r['class_id'];pair=pairs[sid]
            for view in ('weak','ra'):
                guard();assert COUNT<max_forwards
                x=pair['tensors'][view]
                assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==pair['tensor_sha256'][view]
                event('forward_start',source_id=sid,arm=arm,view=view,number=COUNT+1)
                t=time.perf_counter();cpu=time.process_time()
                with torch.inference_mode():out=model(x.unsqueeze(0))
                cpu_elapsed=time.process_time()-cpu;elapsed=time.perf_counter()-t
                COUNT+=1;FORWARD_WALL+=elapsed;FORWARD_CPU+=cpu_elapsed;times.append(elapsed)
                assert out.device.type=='cpu' and tuple(out.shape)==(1,37) and bool(torch.isfinite(out).all())
                z=out[0].tolist()
                if view=='weak':
                    j=max((k for k in range(37) if k!=y),key=lambda k:z[k])
                    ownrefs[(arm,sid)]=j
                    if arm=='front-weak':jrefs[sid]=j
                m=metrics(z,y,jrefs[sid]);own=metrics(z,y,ownrefs[(arm,sid)])
                rec=dict(phase=PHASE,source_id=sid,group=r['group'],class_id=y,class_name=r['class_name'],arm=arm,view=view,
                    logits=z,metrics=m,own_weak_fixed_rival=ownrefs[(arm,sid)],own_weak_fixed_rival_probability=own['p_fixed_rival'],own_weak_fixed_rival_margin=own['fixed_rival_margin'],
                    input_tensor_sha256=pair['tensor_sha256'][view],wall_seconds=elapsed,process_cpu_seconds=cpu_elapsed,peak_RSS_bytes=rss())
                dump(recdir/(sid+'__'+arm+'__'+view+'.json'),rec)
                event('forward_saved',source_id=sid,arm=arm,view=view,number=COUNT,seconds=elapsed);guard()
    final_frozen=frozen_digest();assert final_frozen==frozen
    assert COUNT==max_forwards
    guard()
    result=dict(status='complete',phase=PHASE,completed_forwards=COUNT,total_completed_forwards_including_preflight=COUNT+(8 if PHASE=='continuation' else 0),
        worker_wall_seconds=time.perf_counter()-START,forward_wall_seconds=FORWARD_WALL,forward_cpu_seconds=FORWARD_CPU,
        single_forward_wall_seconds=times,pair_preparation_wall_seconds=preptimes,model_build_seconds=model_build_seconds,
        peak_RSS_bytes=rss(),device='cpu',dtype='float32',intraop_threads=1,interop_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),
        model_eval=True,all_parameters_frozen=True,frozen_sha256_before=frozen,frozen_sha256_after=final_frozen,checkpoint_restores=restores,training_runs=0,neural_GPU_forwards=0)
    dump(OUT/(PHASE+'-result.json'),result)
    event('complete',forwards=COUNT,seconds=result['worker_wall_seconds'])

try:main()
except BaseException as error:
    record=dict(status='stopped',phase=PHASE,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),completed_forwards=COUNT,
        worker_wall_seconds=time.perf_counter()-START,forward_wall_seconds=FORWARD_WALL,forward_cpu_seconds=FORWARD_CPU,peak_RSS_bytes=rss())
    dump(OUT/(PHASE+'-failure.json'),record);event('stopped',error=str(error));raise
finally:LOG.close()
