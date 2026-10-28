"""Fixed two-policy training-side recovery. No held-out evaluation or hyperparameter search."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',
    CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import collections,datetime,gc,hashlib,importlib.util,json,math,resource,signal,time
import numpy as np
from PIL import Image,ImageDraw,ImageStat
import torch,torchvision
from torchvision.transforms import functional as F,InterpolationMode,TrivialAugmentWide
from torchvision.transforms.autoaugment import _apply_op

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
HELPER=BASE/'bn-affine-training-diagnostic-20261021/diagnostic.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='7e2eaf44e41c887eee545541d68d4d7bc8f1276cc601231c1b777243de336c5d'
spec=importlib.util.spec_from_file_location('verified_training_helpers',HELPER)
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
DATA=BASE/'mechanism-transfer-pets-20261021';OLD=BASE/'stage-validity-pilot-20261021'
GEOMETRIC={'ShearX','ShearY','TranslateX','TranslateY','Rotate'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def seed_for(epoch,source):return int.from_bytes(hashlib.sha256(('stage-pilot-v1:'+str(epoch)+':'+source).encode()).digest()[:8],'little')&((1<<63)-1)
def augment(image,epoch,source,policy,space):
    seed=seed_for(epoch,source);state=torch.Generator(device='cpu').manual_seed(seed).get_state()
    torch.random.set_rng_state(state)
    op=list(space)[int(torch.randint(len(space),(1,)).item())];magnitudes,signed=space[op]
    index=int(torch.randint(len(magnitudes),(1,),dtype=torch.long).item()) if magnitudes.ndim>0 else None
    magnitude=float(magnitudes[index].item()) if index is not None else 0.
    negative=bool(torch.randint(2,(1,)).item()) if signed else False
    if negative:magnitude*=-1
    torch.random.set_rng_state(state);out=policy(image)
    return out,dict(source_id=source,pass_index=epoch,seed=seed,operation=op,magnitude_bin=index,magnitude=magnitude,negative=negative)
def risk(im,mask,info,base_area):
    transformed=_apply_op(mask,info['operation'],info['magnitude'],InterpolationMode.NEAREST,fill=0) if info['operation'] in GEOMETRIC else mask
    area=transformed.histogram()[255];extrema=im.getextrema();std=ImageStat.Stat(im).stddev
    return dict(foreground_pixels=area,base_foreground_pixels=base_area,
        foreground_retention=area/base_area if base_area else None,
        channel_std_RGB255=std,max_channel_spatial_std_RGB255=max(std),
        channel_dynamic_ranges_RGB255=[b-a for a,b in extrema],exact_constant=all(a==b for a,b in extrema))
def timeout(signum,frame):raise TimeoutError('fixed600secondwalllimit; no extension')

def main():
    try:os.nice(10)
    except OSError:pass
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu');torch.use_deterministic_algorithms(False)
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    start=time.perf_counter();protocol=json.loads((ROOT/'protocol.json').read_text())
    rows=json.loads((OLD/'protocol.json').read_text())['train_rows'];labels=[r['Y'] for r in rows]
    assert collections.Counter(labels)=={0:32,1:32}
    species={}
    for line in (DATA/'archive-text-3.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields=line.split();species[fields[0]]=int(fields[2])-1
    assert all(species[r['source_id']]==r['Y'] for r in rows)
    old_input=json.loads((OLD/'input-manifest.json').read_text())
    old_views={(r['pass_index'],r['source_id']):r for r in old_input['TA_views']}
    base=[];masks=[];areas=[];photos=[]
    for row in rows:
        path=DATA/'photos'/(row['source_id']+'.jpg');maskpath=DATA/'masks'/(row['source_id']+'.png')
        with Image.open(path) as original,Image.open(maskpath) as rawmask:
            assert original.size==rawmask.size
            base.append(F.center_crop(F.resize(original.convert('RGB'),256,InterpolationMode.BILINEAR,antialias=True),[224,224]))
            array=np.asarray(rawmask);assert set(np.unique(array))<={1,2,3}
            binary=Image.fromarray(np.where(array==1,255,0).astype(np.uint8))
            mask=F.center_crop(F.resize(binary,256,InterpolationMode.NEAREST),[224,224]);masks.append(mask);areas.append(mask.histogram()[255])
        photos.append(dict(source_id=row['source_id'],Y=row['Y'],photo_sha256=sha(path),mask_sha256=sha(maskpath)))
    actual_base=[dict(source_id=r['source_id'],RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest()) for r,im in zip(rows,base)]
    assert actual_base==old_input['base']
    xbase=torch.stack([D.tensor(im) for im in base]);yall=torch.tensor(labels,dtype=torch.long)
    inverse=xbase*torch.tensor([.229,.224,.225])[None,:,None,None]+torch.tensor([.485,.456,.406])[None,:,None,None]
    assert bool(torch.isfinite(xbase).all()) and float(inverse.min())>=-1e-6 and float(inverse.max())<=1+1e-6
    rng=np.random.default_rng(2026102102);orders=[rng.permutation(64).tolist() for _ in range(256)]
    assert orders[:16]==old_input['orders']
    policy=TrivialAugmentWide(num_magnitude_bins=31,interpolation=InterpolationMode.NEAREST,fill=128)
    space=policy._augmentation_space(31)
    probe=[];probe_y=[];probe_rows=[]
    for epoch in range(4):
        for i,row in enumerate(rows):
            im,info=augment(base[i],epoch,row['source_id'],policy,space)
            rgb=hashlib.sha256(im.tobytes()).hexdigest();assert rgb==old_views[(epoch,row['source_id'])]['RGB_sha256']
            probe.append(im);probe_y.append(labels[i]);probe_rows.append(dict(**info,Y=labels[i],RGB_sha256=rgb,**risk(im,masks[i],info,areas[i])))
    save('probe-manifest.json',probe_rows)
    save('input-audit.json',dict(source_photos=photos,labels_agree_official_species_metadata=True,class_counts=[32,32],
        input_shape=list(xbase.shape),dtype=str(xbase.dtype),normalized_min=float(xbase.min()),normalized_max=float(xbase.max()),
        inverse_normalization_min=float(inverse.min()),inverse_normalization_max=float(inverse.max()),
        loss='mean cross_entropy on two raw logits and int64 targets, no sigmoid/softmax before loss',
        base_RGB_all_match_old=True,first16orders_match_old=True,fixed_TA_probe_matches_old=True,
        mask_labels='1 foreground;2 background;3 unknown excluded; resize/crop nearest; same sampled geometry only',
        official_metadata_sha256=sha(DATA/'archive-text-3.txt'),helper_sha256=sha(HELPER),
        official_autoaugment_source_sha256=sha(Path(torchvision.transforms.autoaugment.__file__))))
    models={};optimizers={};arms={};traces={};global_compute=0.;timings={n:[] for n in ['base','TA']};losses={n:[] for n in ['base','TA']}
    for name in ['base','TA']:
        model,opt,initial=D.make_model(False)
        assert initial==json.loads((OLD/'finetune_TA-result.json').read_text())['initial_state_sha256']
        parameter_ids=[id(p) for group in opt.param_groups for p in group['params']]
        assert len(parameter_ids)==len(set(parameter_ids)) and set(parameter_ids)=={id(p) for p in model.parameters()}
        assert all(p.requires_grad for p in model.parameters())
        models[name]=model;optimizers[name]=opt
        arms[name]=dict(initial_state_sha256=initial,BN_initial_sha256=D.bn_stats(model),
            checkpoints={'0':dict(train_clean=D.evaluate(model,base,labels),fixed_train_TA_probe=D.evaluate(model,probe,probe_y))})
        traces[name]=(ROOT/(name+'-steps.jsonl')).open('x')
    assert arms['base']['checkpoints']['0']==arms['TA']['checkpoints']['0']
    counts=collections.Counter();collisions={};view_rows=[];examples_geometry={};examples_std={}
    rgb_chain=hashlib.sha256();base_chain=hashlib.sha256();risk_file=(ROOT/'TA-risk-views.jsonl').open('x')
    checkpoints={128,256,512,1024,1536,2048}
    for block in range(16):
        for name in ['base','TA']:
            model=models[name];opt=optimizers[name]
            for epoch in range(block*16,(block+1)*16):
                for offset in range(0,64,8):
                    if time.perf_counter()-start>580 or global_compute>480:raise TimeoutError('predeclared wall/compute cap, no extension')
                    ids=orders[epoch][offset:offset+8]
                    if name=='base':
                        x=xbase[ids]
                        for i in ids:
                            base_chain.update((str(epoch)+':'+rows[i]['source_id']).encode());base_chain.update(base[i].tobytes())
                    else:
                        ims=[]
                        for i in ids:
                            im,info=augment(base[i],epoch,rows[i]['source_id'],policy,space)
                            rgb=hashlib.sha256(im.tobytes()).hexdigest()
                            if epoch<16:assert rgb==old_views[(epoch,rows[i]['source_id'])]['RGB_sha256']
                            info.update(Y=labels[i],RGB_sha256=rgb,**risk(im,masks[i],info,areas[i]));ims.append(im)
                            rgb_chain.update((str(epoch)+':'+rows[i]['source_id']).encode());rgb_chain.update(im.tobytes())
                            counts[info['operation']]+=1;collisions.setdefault(rgb,[0,0])[labels[i]]+=1;view_rows.append(info)
                            risk_file.write(json.dumps(info)+'\n')
                            key=rows[i]['source_id'];ret=info['foreground_retention'];sd=info['max_channel_spatial_std_RGB255']
                            if ret is not None and (key not in examples_geometry or ret<examples_geometry[key][0]):examples_geometry[key]=(ret,i,im.copy(),info)
                            if key not in examples_std or sd<examples_std[key][0]:examples_std[key]=(sd,i,im.copy(),info)
                        x=torch.stack([D.tensor(im) for im in ims])
                    y=yall[ids];torch.mps.synchronize();t=time.perf_counter();loss=D.step(model,opt,x,y);torch.mps.synchronize()
                    elapsed=time.perf_counter()-t;global_compute+=elapsed;timings[name].append(elapsed);losses[name].append(loss)
                    step=len(losses[name]);traces[name].write(json.dumps(dict(step=step,pass_index=epoch,loss=loss,guarded_compute_seconds=elapsed))+'\n')
                traces[name].flush();risk_file.flush()
            step=len(losses[name])
            if step in checkpoints:
                arms[name]['checkpoints'][str(step)]=dict(train_clean=D.evaluate(model,base,labels),fixed_train_TA_probe=D.evaluate(model,probe,probe_y))
        step=len(losses['TA']);assert step==len(losses['base'])==(block+1)*128
        print(json.dumps(dict(step_per_arm=step,base_online_last128_LL=float(np.mean(losses['base'][-128:])),
            TA_online_last128_LL=float(np.mean(losses['TA'][-128:])),
            clean_train_LL={n:arms[n]['checkpoints'].get(str(step),{}).get('train_clean',{}).get('mean_LL_nats') for n in ['base','TA']},
            wall_seconds=time.perf_counter()-start)),flush=True)
    for f in traces.values():f.close()
    risk_file.close()
    assert len(view_rows)==16384 and sum(counts.values())==16384
    for name in ['base','TA']:
        model=models[name];a=arms[name]
        assert D.bn_stats(model)==a['BN_initial_sha256']
        a.update(steps=2048,source_photos=64,presentations=16384,BN_running_stats_unchanged=True,
            BN_affine_trainable=True,final_state_sha256=D.state_hash(model.state_dict()),all_loss_grad_parameter_finite=True,
            actual_parameter_devices=sorted({p.device.type for p in model.parameters()}),training_compute_seconds=sum(timings[name]),
            block128_online_mean_LL=[float(np.mean(losses[name][s:s+128])) for s in range(0,2048,128)],
            exact_training_RGB_order_sha256=base_chain.hexdigest() if name=='base' else rgb_chain.hexdigest(),weights_saved=False)
        save(name+'-result.json',a)
    assert set(arms['base']['checkpoints'])==set(arms['TA']['checkpoints'])=={'0','128','256','512','1024','1536','2048'}
    mixed={h:v for h,v in collisions.items() if all(v)};floor=sum(sum(v)*(-sum((n/sum(v))*math.log(n/sum(v)) for n in v if n)) for v in collisions.values())/16384
    def quantiles(values):return {str(q):float(np.quantile(values,q)) for q in [0,.01,.05,.5,.95,1]}
    retention=[r['foreground_retention'] for r in view_rows if r['foreground_retention'] is not None]
    risk_summary=dict(TA_presentations=16384,operation_counts=dict(counts),
        foreground_retention_quantiles=quantiles(retention),foreground_retention_eq0_count=sum(v==0 for v in retention),
        foreground_retention_below_0_1_count=sum(v<.1 for v in retention),
        max_channel_std_RGB255_quantiles=quantiles([r['max_channel_spatial_std_RGB255'] for r in view_rows]),
        exact_constant_count=sum(r['exact_constant'] for r in view_rows),
        cross_label_exact_RGB_collision_hashes=len(mixed),cross_label_exact_collision_views=sum(sum(v) for v in mixed.values()),
        empirical_fixed_library_cross_entropy_lower_bound_nats=floor,
        interpretation='pixel/mask risk proxies only; no semantic readability labels; exact-collision bound applies to one fixed model on this empirical library, not changing online models')
    save('risk-summary.json',risk_summary)
    chosen=sorted(examples_geometry.values(),key=lambda x:x[0])[:6]+sorted(examples_std.values(),key=lambda x:x[0])[:6]
    sheet=Image.new('RGB',(800,6*240),'white');draw=ImageDraw.Draw(sheet);selected=[]
    for j,(value,i,im,info) in enumerate(chosen):
        col=j%2;row=j//2;x=col*400;y=row*240
        sheet.paste(base[i].resize((192,192)),(x,y));sheet.paste(im.resize((192,192)),(x+196,y))
        draw.text((x,y+194),rows[i]['source_id']+' Y='+str(labels[i]),fill='black')
        draw.text((x,y+207),info['operation']+' m='+format(info['magnitude'],'.3f'),fill='black')
        draw.text((x,y+220),'FG='+format(info['foreground_retention'] or 0,'.3f')+' std='+format(info['max_channel_spatial_std_RGB255'],'.2f'),fill='black')
        selected.append(info)
    sheet.save(ROOT/'training-risk-examples.png');save('training-risk-examples.json',selected)
    result=dict(status='completed_fixed2048step_base_TA_training_recovery',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        main_wall_seconds=time.perf_counter()-start,training_compute_seconds=global_compute,main_wall_excludes_Python_library_imports=True,
        protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),helper_sha256=sha(HELPER),
        torch=torch.__version__,torchvision=torchvision.__version__,device='mps',precision='FP32',CPU_fallback=False,MPS_fast_math=False,
        previously_used_training_photos=64,development_or_test_photos_or_evaluations=0,matched_updates=2048,matched_presentations_per_arm=16384,
        no_new_data_or_software=True,no_originality_or_generalization_claim=True,no_universal_training_accuracy_gate=True,
        no_best_checkpoint_selection=True,no_more_updates_seeds_or_tuning=True,A100=False,ARIS=False,
        process_peak_RSS_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        final_metrics={n:arms[n]['checkpoints']['2048'] for n in ['base','TA']})
    save('recovery-results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='final_metrics'},indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('recovery-failure.json',dict(status='stopped_no_retuning',type=type(e).__name__,message=str(e)))
        raise
