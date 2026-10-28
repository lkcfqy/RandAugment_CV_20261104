"""One fixed conservative training repair. No held-out data or adaptive tuning."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
    PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import collections,datetime,gc,hashlib,importlib.util,json,re,resource,signal,subprocess,time,traceback
import numpy as np
from PIL import Image
import torch,torchvision
from torchvision.transforms import functional as F,InterpolationMode,TrivialAugmentWide

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
OLD=BASE/'stage-validity-pilot-20261021';PREVIOUS=BASE/'training-validity-recovery-20261021'
DATA=BASE/'mechanism-transfer-pets-20261021'
HELPER=PREVIOUS/'recovery.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='e1df399d3c26cd792f7c99a3c293e16db081ad3f514f6a169055972bfbffb8d5'
spec=importlib.util.spec_from_file_location('old_recovery_helpers',HELPER);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);D=R.D
MODELS={};OPTS={};STEPS={};TRACES={};CHECKPOINTS={};CAPS={};ARRAYS={};RESOURCES=[]
START=None;COMPUTE=0.

class FixedStop(RuntimeError):pass
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lr_factor(step):return 0.1+0.9*(step-1)/127 if step<=128 else 1.
def sync():torch.mps.synchronize()
def cpu_opt(opt):
    raw=opt.state_dict();out={'param_groups':raw['param_groups'],'state':{}}
    for k,v in raw['state'].items():out['state'][k]={n:(a.detach().cpu().clone() if torch.is_tensor(a) else a) for n,a in v.items()}
    return out
def checkpoint(arm,tag,optimizer=False):
    m=MODELS[arm];state={n:v.detach().cpu().clone() for n,v in m.state_dict().items()}
    payload=dict(arm=arm,step=STEPS[arm],model=state,model_state_sha256=D.state_hash(state),
        protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),BN_statistics_sha256=D.bn_stats(m))
    if optimizer:payload['optimizer']=cpu_opt(OPTS[arm])
    path=ROOT/(arm+'-'+tag+'.pt');assert not path.exists();torch.save(payload,path)
    CHECKPOINTS[arm].append(dict(step=STEPS[arm],file=path.name,bytes=path.stat().st_size,sha256=sha(path),
        model_state_sha256=payload['model_state_sha256'],optimizer_saved=optimizer))
def resources(step):
    r=subprocess.run(['/usr/bin/memory_pressure','-Q'],capture_output=True,text=True,timeout=3)
    match=re.search(r'System-wide memory free percentage:\s*(\d+)%',r.stdout)
    free=int(match.group(1)) if r.returncode==0 and match else None
    RESOURCES.append(dict(step_per_arm=step,memory_free_percent=free,query_returncode=r.returncode,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if free is not None and free<15:raise FixedStop('memoryfree<15%;yieldto8Apriority')
def guards():
    if time.perf_counter()-START>540 or COMPUTE>480:raise FixedStop('fixedwallorcomputecap;noextension')
def feature_summary(f):
    return dict(active_channels=int(np.any(f!=0,axis=0).sum()),all_zero_rows=int(np.all(f==0,axis=1).sum()),
        unique_rows=len(np.unique(f,axis=0)),zero_fraction=float(np.mean(f==0)),
        mean_between_sample_variance=float(f.astype(np.float64).var(axis=0).mean()))
class PoolCapture:
    def __init__(self,m):self.enabled=False;self.value=None;self.handle=m.avgpool.register_forward_hook(self.hook)
    def hook(self,module,args,out):
        if self.enabled:self.value=out.detach().flatten(1).cpu().numpy().copy()
def evaluate(arm,x,y,name):
    m=MODELS[arm];c=CAPS[arm];m.eval();c.enabled=name=='clean';zs=[];features=[]
    with torch.inference_mode():
        for offset in range(0,len(y),8):
            z=m(x[offset:offset+8].to('mps')).cpu();zs.append(z)
            if c.enabled:features.append(c.value.copy())
    z=torch.cat(zs);ls=torch.nn.functional.cross_entropy(z,y,reduction='none')
    assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(ls).all())
    key=arm+'_'+str(STEPS[arm])+'_'+name;ARRAYS[key+'_logits']=z.numpy().copy()
    out=dict(mean_LL_nats=float(ls.mean()),accuracy=float((z.argmax(1)==y).float().mean()),
        unique_logit_rows=int(torch.unique(z,dim=0).shape[0]),losses=ls.tolist(),logits=z.tolist())
    if features:
        f=np.concatenate(features);ARRAYS[key+'_features']=f;out['features']=feature_summary(f)
    c.enabled=False;D.train_mode(m);assert m.training and all(not b.training for b in m.modules() if isinstance(b,torch.nn.modules.batchnorm._BatchNorm))
    return out
def timeout(signum,frame):raise FixedStop('hard600secondalarm')

def main():
    global START,COMPUTE
    try:os.nice(10)
    except OSError:pass
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu');torch.use_deterministic_algorithms(False)
    assert torch.backends.mps.is_available();START=time.perf_counter();signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    protocol=json.loads((ROOT/'protocol.json').read_text());assert protocol['recipe_count']==1
    resources(0)
    rows=json.loads((OLD/'protocol.json').read_text())['train_rows'];ys=[r['Y'] for r in rows]
    assert len(rows)==64 and collections.Counter(ys)=={0:32,1:32};selected={r['source_id'] for r in rows}
    species={}
    for line in (DATA/'archive-text-1.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            v=line.split()
            if v[0] in selected:species[v[0]]=int(v[2])-1
    assert all(species[r['source_id']]==r['Y'] for r in rows)
    old=json.loads((OLD/'input-manifest.json').read_text());oldviews={(v['pass_index'],v['source_id']):v for v in old['TA_views']}
    base=[];photo_hashes=[]
    for r in rows:
        path=DATA/'photos'/(r['source_id']+'.jpg')
        with Image.open(path) as im:base.append(F.center_crop(F.resize(im.convert('RGB'),256,InterpolationMode.BILINEAR,antialias=True),[224,224]))
        photo_hashes.append(dict(source_id=r['source_id'],Y=r['Y'],sha256=sha(path)))
    assert [dict(source_id=r['source_id'],RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest()) for r,im in zip(rows,base)]==old['base']
    xbase=torch.stack([D.tensor(im) for im in base]);yall=torch.tensor(ys,dtype=torch.long)
    rng=np.random.default_rng(2026102102);orders=[rng.permutation(64).tolist() for _ in range(256)];assert orders[:16]==old['orders']
    policy=TrivialAugmentWide(num_magnitude_bins=31,interpolation=InterpolationMode.NEAREST,fill=128);space=policy._augmentation_space(31)
    probe=[];probe_rows=[]
    for epoch in range(4):
        for r,im in zip(rows,base):
            aug,info=R.augment(im,epoch,r['source_id'],policy,space);rgb=hashlib.sha256(aug.tobytes()).hexdigest()
            assert rgb==oldviews[(epoch,r['source_id'])]['RGB_sha256'];probe.append(D.tensor(aug));probe_rows.append(dict(**info,Y=r['Y'],RGB_sha256=rgb))
    xprobe=torch.stack(probe);yprobe=yall.repeat(4);del probe
    save('probe-manifest.json',probe_rows)
    save('input-audit.json',dict(training_rows=photo_hashes,class_counts=[32,32],labels_match_official=True,
        base_RGB_match_old=True,first16orders_match_old=True,probe_matches_old=True,metadata_sha256=sha(DATA/'archive-text-1.txt'),
        recovery_helper_sha256=sha(HELPER),old_input_manifest_sha256=sha(OLD/'input-manifest.json'),
        heldout_or_development_images_read=0,training_subset='previouslydesignatedcustom64sources;notanofficialtrainval-onlysplit'))
    ARMS={};LOSSES={};CHAINS={n:hashlib.sha256() for n in ['base','TA']};OPCOUNTS=collections.Counter()
    for arm in ['base','TA']:
        m,opt,initial=D.make_model(False);assert initial=='ec9897fa3d388d789d326977f28f7814ed3b474c083763c2468e46744999fe32'
        MODELS[arm]=m;OPTS[arm]=opt;STEPS[arm]=0;CHECKPOINTS[arm]=[];CAPS[arm]=PoolCapture(m);LOSSES[arm]=[]
        pids=[id(p) for g in opt.param_groups for p in g['params']];assert len(pids)==len(set(pids)) and set(pids)=={id(p) for p in m.parameters()}
        ARMS[arm]=dict(initial_state_sha256=initial,BN_initial_sha256=D.bn_stats(m),checkpoints={'0':dict(clean=evaluate(arm,xbase,yall,'clean'),fixed_TA=evaluate(arm,xprobe,yprobe,'TA'))})
        TRACES[arm]=(ROOT/(arm+'-steps.jsonl')).open('x')
    assert ARMS['base']['checkpoints']['0']==ARMS['TA']['checkpoints']['0']
    eval_steps={128,256,512,1024,1536,2048};compute_per_arm=dict(base=0.,TA=0.)
    for block in range(16):
        for arm in ['base','TA']:
            for epoch in range(block*16,(block+1)*16):
                for offset in range(0,64,8):
                    guards();ids=orders[epoch][offset:offset+8]
                    if arm=='base':ims=[base[i] for i in ids];x=xbase[ids]
                    else:
                        ims=[]
                        for i in ids:
                            aug,info=R.augment(base[i],epoch,rows[i]['source_id'],policy,space);ims.append(aug);OPCOUNTS[info['operation']]+=1
                            if epoch<16:assert hashlib.sha256(aug.tobytes()).hexdigest()==oldviews[(epoch,rows[i]['source_id'])]['RGB_sha256']
                        x=torch.stack([D.tensor(im) for im in ims])
                    for i,im in zip(ids,ims):CHAINS[arm].update((str(epoch)+':'+rows[i]['source_id']).encode());CHAINS[arm].update(im.tobytes())
                    step=STEPS[arm]+1;factor=lr_factor(step);OPTS[arm].param_groups[0]['lr']=.01*factor;OPTS[arm].param_groups[1]['lr']=.0001*factor
                    sync();t=time.perf_counter();loss=D.step(MODELS[arm],OPTS[arm],x,yall[ids]);sync();elapsed=time.perf_counter()-t
                    COMPUTE+=elapsed;compute_per_arm[arm]+=elapsed;STEPS[arm]=step;LOSSES[arm].append(loss)
                    TRACES[arm].write(json.dumps(dict(step=step,pass_index=epoch,loss=loss,head_lr=.01*factor,body_lr=.0001*factor,guarded_compute_seconds=elapsed))+'\n')
                TRACES[arm].flush()
            step=STEPS[arm]
            if step in eval_steps:
                ARMS[arm]['checkpoints'][str(step)]=dict(clean=evaluate(arm,xbase,yall,'clean'),fixed_TA=evaluate(arm,xprobe,yprobe,'TA'))
                if step in [128,1024,2048]:checkpoint(arm,'step'+str(step),optimizer=step==2048)
                keys=[k for k in ARMS[arm]['checkpoints'] if k!='0'];initial=ARMS[arm]['checkpoints']['0']['clean']
                if len(keys)>=2:
                    recent=[ARMS[arm]['checkpoints'][k]['clean'] for k in keys[-2:]]
                    collapsed=all(v['mean_LL_nats']>=.95*initial['mean_LL_nats'] and v['features']['mean_between_sample_variance']<=1e-4*initial['features']['mean_between_sample_variance'] for v in recent)
                    if collapsed:raise FixedStop(arm+'recurrentseverecollapseat'+str(step))
        assert STEPS['base']==STEPS['TA']==(block+1)*128
        s=STEPS['base'];latest={n:ARMS[n]['checkpoints'].get(str(s)) for n in ['base','TA']}
        print(json.dumps(dict(step_per_arm=s,online_last128_LL={n:float(np.mean(LOSSES[n][-128:])) for n in ['base','TA']},
            clean_LL={n:v['clean']['mean_LL_nats'] if v else None for n,v in latest.items()},
            clean_accuracy={n:v['clean']['accuracy'] if v else None for n,v in latest.items()},main_wall_seconds=time.perf_counter()-START)),flush=True)
        if s%512==0:resources(s)
    gates={}
    for arm in ['base','TA']:
        a=ARMS[arm];initial=a['checkpoints']['0'];final=a['checkpoints']['2048']
        assert D.bn_stats(MODELS[arm])==a['BN_initial_sha256'];assert len(LOSSES[arm])==2048
        old_result=json.loads((PREVIOUS/(arm+'-result.json')).read_text());assert CHAINS[arm].hexdigest()==old_result['exact_training_RGB_order_sha256']
        gates[arm]=dict(completed_all_finite=True,
            clean_learning=all(a['checkpoints'][s]['clean']['mean_LL_nats']<=.5*initial['clean']['mean_LL_nats'] for s in ['1536','2048']),
            own_distribution_learning=(arm=='base' or all(a['checkpoints'][s]['fixed_TA']['mean_LL_nats']<=.5*initial['fixed_TA']['mean_LL_nats'] for s in ['1536','2048'])),
            online_learning=float(np.mean(LOSSES[arm][-256:]))<=.5*float(np.mean(LOSSES[arm][:256])),
            no_old_collapse=(final['clean']['features']['mean_between_sample_variance']>1e-4*initial['clean']['features']['mean_between_sample_variance'] and final['clean']['unique_logit_rows']>1))
        a.update(updates=2048,presentations=16384,all_loss_gradient_parameter_finite=True,BN_running_statistics_unchanged=True,
            first256_online_LL=float(np.mean(LOSSES[arm][:256])),last256_online_LL=float(np.mean(LOSSES[arm][-256:])),
            block128_online_LL=[float(np.mean(LOSSES[arm][i:i+128])) for i in range(0,2048,128)],
            training_compute_seconds=compute_per_arm[arm],exact_training_RGB_order_sha256=CHAINS[arm].hexdigest(),RGB_order_matches_old=True,
            saved_model_checkpoints=CHECKPOINTS[arm],success_criteria=gates[arm])
        save(arm+'-result.json',a);TRACES[arm].close()
    ARRR=dict(ARRAYS,clean_labels=yall.numpy(),TA_probe_labels=yprobe.numpy());np.savez_compressed(ROOT/'training-probe-arrays.npz',**ARRR)
    passed=all(all(g.values()) for g in gates.values())
    result=dict(status='completed',single_recipe_recovered=passed,criteria=gates,
        finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_wall_seconds=time.perf_counter()-START,
        guarded_training_compute_seconds=COMPUTE,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_version=torch.__version__,torchvision_version=torchvision.__version__,device='mps',precision='FP32',CPU_fallback=False,
        steps_per_arm=STEPS,protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),
        arrays_sha256=sha(ROOT/'training-probe-arrays.npz'),resource_checks=RESOURCES,TA_operation_counts=dict(OPCOUNTS),
        no_best_checkpoint_selection=True,no_heldout_data=True,no_further_recipe_or_seed=True,no_generalization_or_contribution_claim=True,
        final_metrics={n:ARMS[n]['checkpoints']['2048'] for n in ['base','TA']})
    save('repair-results.json',result);signal.alarm(0)
    print(json.dumps({k:v for k,v in result.items() if k not in ['final_metrics','resource_checks','TA_operation_counts']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        signal.alarm(0)
        for arm in MODELS:
            try:checkpoint(arm,'stopped-step'+str(STEPS[arm]),optimizer=True)
            except Exception:pass
        for f in TRACES.values():
            try:f.flush();f.close()
            except Exception:pass
        save('stopped.json',dict(status='stopped_no_retuning',type=type(e).__name__,reason=str(e),traceback=traceback.format_exc(),
            steps=STEPS,saved_checkpoints=CHECKPOINTS,resource_checks=RESOURCES,wall_seconds=None if START is None else time.perf_counter()-START))
        raise
