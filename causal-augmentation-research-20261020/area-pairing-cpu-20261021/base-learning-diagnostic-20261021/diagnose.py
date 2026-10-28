"""Bounded, training-only diagnostic of the previously observed base collapse."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',
    CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import collections,copy,datetime,gc,hashlib,importlib.util,json,resource,signal,time,traceback
import numpy as np
from PIL import Image
import torch,torchvision
from torchvision.transforms import functional as F,InterpolationMode

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
OLD=BASE/'stage-validity-pilot-20261021'
REC=BASE/'training-validity-recovery-20261021'
DATA=BASE/'mechanism-transfer-pets-20261021'
WEIGHT=BASE/'encoder-readiness-20261021/runtime/torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
HELPER=BASE/'bn-affine-training-diagnostic-20261021/diagnostic.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='7e2eaf44e41c887eee545541d68d4d7bc8f1276cc601231c1b777243de336c5d'
spec=importlib.util.spec_from_file_location('verified_helpers',HELPER)
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
TRACK=['conv1.weight','layer4.1.conv2.weight','layer4.1.bn2.weight','layer4.1.bn2.bias','fc.weight','fc.bias']
# Full conv2 gradients are included in full gradient comparisons; small actual-update probes avoid redundant transfers.
SMALL=[n for n in TRACK if n!='layer4.1.conv2.weight']
DEEP={1,8,11,16,24,26,28,32,40,48,56,64}
PROBES={0,8,16,24,32,48,64}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    with (ROOT/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sync(device):
    if device=='mps':torch.mps.synchronize()
def summary(a):
    a=np.asarray(a,dtype=np.float64)
    return dict(shape=list(a.shape),finite=bool(np.isfinite(a).all()),minimum=float(a.min()),maximum=float(a.max()),
        L2=float(np.linalg.norm(a.ravel())),zero_fraction=float(np.mean(a==0)),mean=float(a.mean()),std=float(a.std()))
def feature_summary(a):
    a=np.asarray(a)
    return dict(**summary(a),all_zero_rows=int(np.all(a==0,axis=1).sum()),
        active_channels=int(np.any(a!=0,axis=0).sum()),mean_between_sample_variance=float(np.var(a,axis=0).mean()),
        unique_rows=int(np.unique(a,axis=0).shape[0]))
def flat_gradient_stats(named):
    out={}
    for group in ['body','head']:
        ps=[p for n,p in named.items() if n.startswith('fc.')==(group=='head')]
        gs=[p.grad.detach().flatten() for p in ps if p.grad is not None]
        if gs:
            a=torch.cat(gs)
            out[group]=dict(L2=float(a.norm().cpu()),maxabs=float(a.abs().max().cpu()),
                zero_fraction=float((a==0).float().mean().cpu()),finite=bool(torch.isfinite(a).all().item()),
                missing_gradients=sum(p.grad is None for p in ps),total_tensors=len(ps))
        else:out[group]=dict(missing_gradients=len(ps),total_tensors=len(ps))
    return out
def model_factory(device,state=None,opt_state=None):
    m=torchvision.models.resnet18(weights=None)
    if state is None:
        m.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
        torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(0).get_state())
        m.fc=torch.nn.Linear(512,2,device='cpu')
    else:
        m.fc=torch.nn.Linear(512,2,device='cpu');m.load_state_dict(state,strict=True)
    m.to(device);D.train_mode(m)
    opt=torch.optim.SGD([dict(params=m.fc.parameters(),lr=.01),dict(params=[p for n,p in m.named_parameters() if not n.startswith('fc.')],lr=.001)],
        momentum=.9,weight_decay=1e-4)
    if opt_state is not None:opt.load_state_dict(copy.deepcopy(opt_state))
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in m.parameters()}
    assert all(p.requires_grad for p in m.parameters())
    return m,opt
def cpu_state(model):return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
def cpu_optimizer(opt):
    a=copy.deepcopy(opt.state_dict())
    for state in a['state'].values():
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=v.detach().cpu().clone()
    return a

class Capture:
    def __init__(self,m):
        self.deep=False;self.pool=None;self.stages={};self.pre_relu=[]
        self.handles=[m.avgpool.register_forward_hook(self.pooled)]
        for name in ['layer1','layer2','layer3','layer4']:
            self.handles.append(getattr(m,name).register_forward_hook(self.stage(name)))
        self.handles.append(m.layer4[1].relu.register_forward_pre_hook(self.before_relu))
    def reset(self,deep=False):self.deep=deep;self.stages={};self.pre_relu=[]
    def pooled(self,module,args,out):self.pool=out.detach().flatten(1).cpu().numpy().copy()
    def stage(self,name):
        def hook(module,args,out):
            if self.deep:self.stages[name]=summary(out.detach().cpu().numpy())
        return hook
    def before_relu(self,module,args):
        if self.deep:
            # Copy before the in-place ReLU; never return or mutate a replacement tensor.
            a=args[0].detach().cpu().numpy().copy();s=summary(a)
            s['positive_fraction']=float(np.mean(a>0));self.pre_relu.append(s)
    def close(self):
        for h in self.handles:h.remove()

def probe(m,device,cap,xall,yall,tag,arrays):
    m.eval();zs=[];fs=[];deep=None
    with torch.inference_mode():
        for i in range(0,64,8):
            cap.reset(deep=i==0)
            z=m(xall[i:i+8].to(device)).cpu();zs.append(z);fs.append(cap.pool.copy())
            if i==0:deep=dict(stages=cap.stages,last_block_relu_inputs=cap.pre_relu)
    z=torch.cat(zs);f=np.concatenate(fs);ls=torch.nn.functional.cross_entropy(z,yall,reduction='none')
    arrays[tag+'_features']=f;arrays[tag+'_logits']=z.numpy().copy()
    D.train_mode(m)
    return dict(mean_LL_nats=float(ls.mean()),accuracy=float((z.argmax(1)==yall).float().mean()),
        features=feature_summary(f),logits=summary(z.numpy()),unique_logit_rows=int(torch.unique(z,dim=0).shape[0]),
        maxabs_logit_minus_head_bias=float((z-m.fc.bias.detach().cpu()).abs().max()),
        first8_deep=deep,model_train_after=bool(m.training),all_BN_eval_after=all(not b.training for b in m.modules() if isinstance(b,torch.nn.modules.batchnorm._BatchNorm)))

def compare_arrays(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b
    return dict(maxabs=float(np.max(np.abs(d))),L2=float(np.linalg.norm(d.ravel())),
        relative_L2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(a.ravel()),1e-30)),
        cosine=float(np.dot(a.ravel(),b.ravel())/max(np.linalg.norm(a.ravel())*np.linalg.norm(b.ravel()),1e-30)))

def same_state_check(state,opt_state,x,y,label):
    outputs={}
    for device in ['cpu','mps']:
        m,opt=model_factory(device,state,opt_state);named=dict(m.named_parameters())
        opt.zero_grad(set_to_none=True);z=m(x.to(device));loss=torch.nn.functional.cross_entropy(z,y.to(device));loss.backward()
        grads={n:p.grad.detach().cpu().numpy().copy() for n,p in named.items()}
        opt.step();sync(device)
        params={n:p.detach().cpu().numpy().copy() for n,p in named.items()}
        outputs[device]=dict(loss=float(loss.detach().cpu()),logits=z.detach().cpu().numpy().copy(),grads=grads,params=params)
        del m,opt;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    a,b=outputs['cpu'],outputs['mps'];gradient={};updates={}
    for n in a['grads']:
        gradient[n]=compare_arrays(a['grads'][n],b['grads'][n])
        original=state[n].numpy()
        updates[n]=compare_arrays(a['params'][n]-original,b['params'][n]-original)
    flat_a=np.concatenate([a['grads'][n].ravel() for n in a['grads']]);flat_b=np.concatenate([b['grads'][n].ravel() for n in b['grads']])
    out=dict(state_origin=label,model_state_sha256=D.state_hash(state),CPU_loss=a['loss'],MPS_loss=b['loss'],
        logits=compare_arrays(a['logits'],b['logits']),full_gradient=compare_arrays(flat_a,flat_b),
        gradients_by_parameter=gradient,updates_by_parameter=updates)
    del outputs,flat_a,flat_b;gc.collect()
    return out

def timeout(signum,frame):raise TimeoutError('predeclared240secondwallcap; no extension')
def main():
    try:os.nice(10)
    except OSError:pass
    signal.signal(signal.SIGALRM,timeout);signal.alarm(240)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu');torch.use_deterministic_algorithms(False)
    assert torch.backends.mps.is_available() and torch.backends.mps.is_built()
    start=time.perf_counter();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha(WEIGHT)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    rows=json.loads((OLD/'protocol.json').read_text())['train_rows'];ys=[r['Y'] for r in rows]
    assert len(rows)==64 and collections.Counter(ys)=={0:32,1:32}
    species={}
    for line in (DATA/'archive-text-1.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            p=line.split()
            if p[0] in {r['source_id'] for r in rows}:species[p[0]]=int(p[2])-1
    assert all(species[r['source_id']]==r['Y'] for r in rows)
    old=json.loads((OLD/'input-manifest.json').read_text());base=[];photos=[];reference_max=0.
    official=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    for r in rows:
        path=DATA/'photos'/(r['source_id']+'.jpg')
        with Image.open(path) as im:
            rgb=im.convert('RGB');b=F.center_crop(F.resize(rgb,256,InterpolationMode.BILINEAR,antialias=True),[224,224])
            reference_max=max(reference_max,float((D.tensor(b)-official(rgb)).abs().max()));base.append(b)
        photos.append(dict(source_id=r['source_id'],Y=r['Y'],photo_sha256=sha(path)))
    assert [dict(source_id=r['source_id'],RGB_sha256=hashlib.sha256(b.tobytes()).hexdigest()) for r,b in zip(rows,base)]==old['base']
    assert reference_max==0
    xall=torch.stack([D.tensor(im) for im in base]);yall=torch.tensor(ys,dtype=torch.long)
    orders=old['orders'][:8];batches=[order[i:i+8] for order in orders for i in range(0,64,8)]
    assert len(batches)==64 and all(sorted(o)==list(range(64)) for o in orders)
    assert all(torch.equal(xall[ids],torch.stack([D.tensor(base[i]) for i in ids])) for ids in batches)
    save('input-audit.json',dict(source_photos=photos,labels_match_official=True,class_counts=[32,32],
        normalized_shape=list(xall.shape),normalized_min=float(xall.min()),normalized_max=float(xall.max()),
        official_preprocess_maxabs=reference_max,prestack_and_on_the_fly_exact=True,base_RGB_match_archive=True,
        x_dtype=str(xall.dtype),y_dtype=str(yall.dtype),batches=batches,no_heldout_pixels_read=True,
        training_split='Existing custom64source training subset; not official trainval-only',
        source_code_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'protocol.json'),helper_sha256=sha(HELPER)))
    arrays={};arms={};snapshots={};checks={}
    initial,_=model_factory('cpu');initial_state=cpu_state(initial);initial_hash=D.state_hash(initial_state)
    assert initial_hash=='ec9897fa3d388d789d326977f28f7814ed3b474c083763c2468e46744999fe32'
    del initial
    checks['0']=same_state_check(initial_state,None,xall[batches[0]],yall[batches[0]],'shared_initial')
    save('same-state-0.json',checks['0'])
    for device in ['cpu','mps']:
        model,opt=model_factory(device,initial_state);cap=Capture(model);named=dict(model.named_parameters())
        bn_before=D.bn_stats(model);traces=[];checkpoints={};compute=0.
        checkpoints['0']=probe(model,device,cap,xall,yall,device+'_0',arrays)
        with (ROOT/(device+'-steps.jsonl')).open('x') as trace:
            for step,ids in enumerate(batches,1):
                if time.perf_counter()-start>225:raise TimeoutError('predeclaredwallcapapproaching')
                x=xall[ids].to(device);y=yall[ids].to(device);cap.reset(deep=step in DEEP)
                before={n:named[n].detach().cpu().clone() for n in SMALL}
                momentum={n:opt.state[named[n]].get('momentum_buffer',None) for n in SMALL}
                momentum={n:None if v is None else v.detach().cpu().clone() for n,v in momentum.items()}
                sync(device);t=time.perf_counter();opt.zero_grad(set_to_none=True)
                z=model(x);loss=torch.nn.functional.cross_entropy(z,y);assert torch.isfinite(loss)
                loss.backward();gstats=flat_gradient_stats(named);assert all(s['finite'] for s in gstats.values())
                fstats=feature_summary(cap.pool);deep=dict(stages=cap.stages,last_block_relu_inputs=cap.pre_relu)
                grad_small={n:named[n].grad.detach().cpu().clone() for n in SMALL}
                named_grads={n:summary(named[n].grad.detach().cpu().numpy()) for n in TRACK}
                opt.step();assert D.finite(list(model.parameters()));sync(device);elapsed=time.perf_counter()-t;compute+=elapsed
                updates={}
                for n in SMALL:
                    after=named[n].detach().cpu();lr=.01 if n.startswith('fc.') else .001
                    grad=grad_small[n]+1e-4*before[n]
                    velocity=grad if momentum[n] is None else .9*momentum[n]+grad
                    expected=before[n]-lr*velocity
                    updates[n]=dict(actual_delta_L2=float((after-before[n]).double().norm()),
                        actual_delta_maxabs=float((after-before[n]).abs().max()),manual_SGD_parameter_maxabs=float((after-expected).abs().max()))
                row=dict(step=step,ids=ids,Y=[ys[i] for i in ids],loss=float(loss.detach().cpu()),
                    forward_features=fstats,gradients=gstats,named_gradients=named_grads,actual_updates=updates,
                    all_parameters_finite=True,guarded_instrumented_seconds=elapsed,**deep)
                trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush();traces.append(row)
                if device=='mps' and step==24:snapshots['mps24']=(cpu_state(model),cpu_optimizer(opt))
                if step in PROBES:
                    checkpoints[str(step)]=probe(model,device,cap,xall,yall,device+'_'+str(step),arrays)
                    p=checkpoints[str(step)]
                    print(json.dumps(dict(device=device,step=step,train_LL=p['mean_LL_nats'],train_accuracy=p['accuracy'],
                        active_features=p['features']['active_channels'],feature_variance=p['features']['mean_between_sample_variance'],
                        wall_seconds=time.perf_counter()-start)),flush=True)
        assert D.bn_stats(model)==bn_before
        final=cpu_state(model);checkpoint=ROOT/(device+'-step64-state.pt');torch.save(final,checkpoint)
        old_path=OLD/'finetune_base-steps.jsonl' if device=='cpu' else REC/'base-steps.jsonl'
        archived=[json.loads(l)['loss'] for l in old_path.read_text().splitlines()][:64]
        losses=[r['loss'] for r in traces];deltas=[abs(a-b) for a,b in zip(losses,archived)]
        arms[device]=dict(initial_state_sha256=initial_hash,BN_running_stats_unchanged=True,BN_stats_sha256=bn_before,
            optimizer_parameter_coverage_correct=True,all_parameters_trainable=True,updates=64,presentations=512,
            guarded_instrumented_compute_seconds=compute,checkpoints=checkpoints,
            archived_first64_loss_maxabs=max(deltas),archived_loss_differences=deltas,
            final_state_sha256=D.state_hash(final),checkpoint_file=checkpoint.name,checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha(checkpoint))
        save(device+'-result.json',arms[device]);cap.close();del model,opt,cap,named;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    s,o=snapshots.pop('mps24');checks['24']=same_state_check(s,o,xall[batches[24]],yall[batches[24]],'MPS_main_step24')
    save('same-state-24.json',checks['24']);del s,o
    cross={}
    for origin,target in [('mps','cpu'),('cpu','mps')]:
        state=torch.load(ROOT/(origin+'-step64-state.pt'),map_location='cpu',weights_only=True)
        m,opt=model_factory(target,state);cap=Capture(m);tag=origin+'_state64_on_'+target
        cross[tag]=probe(m,target,cap,xall,yall,tag,arrays)
        cross[tag]['logit_difference_from_original_device']=compare_arrays(arrays[tag+'_logits'],arrays[origin+'_64_logits'])
        cross[tag]['feature_difference_from_original_device']=compare_arrays(arrays[tag+'_features'],arrays[origin+'_64_features'])
        cap.close();del m,opt,cap,state;gc.collect()
        if target=='mps':torch.mps.empty_cache()
    np.savez_compressed(ROOT/'training-probe-arrays.npz',**arrays,labels=yall.numpy())
    save('diagnostic-results.json',dict(status='completed',started_UTC=started,finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_version=torch.__version__,torchvision_version=torchvision.__version__,numpy_version=np.__version__,
        MPS_available=torch.backends.mps.is_available(),cpu_threads=torch.get_num_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        main_updates_per_arm=64,additional_same_state_diagnostic_updates_per_device=2,arms=arms,cross_device_same_final_state=cross,
        files=dict(source_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'protocol.json'),arrays_sha256=sha(ROOT/'training-probe-arrays.npz')),
        limitations=protocol['interpretation_limit']))
    signal.alarm(0);print(json.dumps(dict(status='completed',wall_seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if not (ROOT/'failure.json').exists():save('failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
        raise
