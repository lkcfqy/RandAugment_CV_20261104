"""One pre-fixed 4cell development diagnostic. No independent-test claim or policy search."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import datetime,hashlib,importlib.util,json,resource,signal,time
import numpy as np
from PIL import Image
import torch,torchvision
from torchvision.transforms import functional as F,InterpolationMode,TrivialAugmentWide

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
REPAIR=BASE/'training-numerics-repair-20261021/repair_training.py'
spec=importlib.util.spec_from_file_location('repaired_CPU_step',REPAIR);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
DATA=BASE/'mechanism-transfer-pets-20261021'
DEV=BASE/'train-filter-sourceweight-20261021/test-photos'
MEAN=[.485,.456,.406];STD=[.229,.224,.225]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def base_image(path):
    im=Image.open(path).convert('RGB')
    return F.center_crop(F.resize(im,256,InterpolationMode.BILINEAR,antialias=True),[224,224])
def tensor(im):return F.normalize(F.to_tensor(im),MEAN,STD)
def training_mode(model):
    model.train()
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
def body_hash(model):return R.state_hash({n:v for n,v in model.state_dict().items() if not n.startswith('fc.')})

def make_model(finetune):
    model=torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(R.WEIGHT,map_location='cpu',weights_only=True),strict=True)
    model.requires_grad_(finetune)
    torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(0).get_state())
    model.fc=torch.nn.Linear(512,2,device='cpu');training_mode(model)
    groups=[dict(params=model.fc.parameters(),lr=.01)]
    if finetune:groups.append(dict(params=[p for n,p in model.named_parameters() if not n.startswith('fc.')],lr=.001))
    optimizer=torch.optim.SGD(groups,momentum=.9,weight_decay=1e-4)
    assert all(p.device.type=='cpu' for p in model.parameters())
    return model,optimizer

def evaluate(model,images,labels):
    model.eval();logits=[]
    with torch.inference_mode():
        for start in range(0,len(images),8):
            x=torch.stack([tensor(im) for im in images[start:start+8]]).to('cpu')
            logits.append(model(x))
        z=torch.cat(logits);y=torch.tensor(labels,dtype=torch.long,device='cpu')
        ll=torch.nn.functional.cross_entropy(z,y,reduction='none')
        assert torch.isfinite(z).all() and torch.isfinite(ll).all()
        out=dict(mean_LL_nats=float(ll.mean()),accuracy=float((z.argmax(dim=1)==y).float().mean()),
            logits=z.tolist(),losses=ll.tolist())
    training_mode(model)
    return out

def main():
    try:os.nice(10)
    except OSError:pass
    signal.alarm(600);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu');torch.use_deterministic_algorithms(True)
    started=time.perf_counter();deadline=started+590
    protocol=json.loads((ROOT/'protocol.json').read_text());train=protocol['train_rows'];dev=protocol['development_rows']
    assert sha(R.WEIGHT)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    base=[base_image(DATA/'photos'/(r['source_id']+'.jpg')) for r in train]
    dev_images=[base_image(DEV/(r['source_id']+'.jpg')) for r in dev]
    train_y=[r['Y'] for r in train];dev_y=[r['Y'] for r in dev]
    rng=np.random.default_rng(2026102102);orders=[rng.permutation(64).tolist() for _ in range(16)]
    policy=TrivialAugmentWide(num_magnitude_bins=31,interpolation=InterpolationMode.NEAREST,fill=128)
    views=[];manifest=[]
    for epoch in range(16):
        epoch_views=[]
        for i,source in enumerate(train):
            key='stage-pilot-v1:'+str(epoch)+':'+source['source_id']
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')&((1<<63)-1)
            torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            im=policy(base[i]);epoch_views.append(im)
            manifest.append(dict(pass_index=epoch,source_id=source['source_id'],seed=seed,RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
        views.append(epoch_views)
    save('input-manifest.json',dict(orders=orders,TA_views=manifest,
        base=[dict(source_id=r['source_id'],RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest()) for r,im in zip(train,base)],
        development_previously_used=True,prior_data_receipt_sha256=sha(BASE/'train-filter-sourceweight-20261021/test-fetch-receipt.json')))
    arms={};common_init=None
    for cell in protocol['cells']:
        name=cell['name'];model,optimizer=make_model(cell['finetune'])
        initial=R.state_hash(model.state_dict());body_before=body_hash(model);BN_before=R.bn_hash(model)
        if common_init is None:common_init=initial
        else:assert initial==common_init
        conv_before=model.conv1.weight.detach().clone();head_before=model.fc.weight.detach().clone()
        checkpoints={};training=[];image_chain=hashlib.sha256();step=0;arm_start=time.perf_counter();train_seconds=0.
        checkpoints['0']=dict(train_clean=evaluate(model,base,train_y),development_clean=evaluate(model,dev_images,dev_y))
        with (ROOT/(name+'-steps.jsonl')).open('x') as trace:
            for epoch in range(16):
                for offset in range(0,64,8):
                    if time.perf_counter()>deadline:raise TimeoutError('pre-fixed600secondCPU budget; no extension')
                    ids=orders[epoch][offset:offset+8]
                    ims=[views[epoch][i] if cell['TA'] else base[i] for i in ids]
                    for i,im in zip(ids,ims):
                        image_chain.update((str(epoch)+':'+train[i]['source_id']).encode());image_chain.update(im.tobytes())
                    x=torch.stack([tensor(im) for im in ims]).to('cpu');y=torch.tensor([train_y[i] for i in ids],dtype=torch.long,device='cpu')
                    t=time.perf_counter();loss=R.safe_train_step(model,optimizer,x,y);elapsed=time.perf_counter()-t
                    train_seconds+=elapsed;step+=1;training.append(loss)
                    trace.write(json.dumps(dict(step=step,pass_index=epoch,loss=loss,guarded_compute_seconds=elapsed))+'\n');trace.flush()
                    if step in [64,128]:
                        checkpoints[str(step)]=dict(train_clean=evaluate(model,base,train_y),development_clean=evaluate(model,dev_images,dev_y))
                    if step%32==0:print(json.dumps({'arm':name,'step':step,'loss':loss,'wall_seconds':time.perf_counter()-started}),flush=True)
        body_after=body_hash(model);BN_after=R.bn_hash(model)
        assert step==128 and BN_before==BN_after
        if not cell['finetune']:assert body_before==body_after
        else:assert body_before!=body_after
        out=dict(name=name,finetune=cell['finetune'],TA=cell['TA'],steps=step,source_photos=64,training_presentations=1024,
            initial_state_sha256=initial,body_before_sha256=body_before,body_after_sha256=body_after,
            BN_running_stats_sha256=BN_before,BN_running_stats_unchanged=True,
            conv1_delta_L2=float((model.conv1.weight.detach()-conv_before).double().norm()),
            head_delta_L2=float((model.fc.weight.detach()-head_before).double().norm()),
            exact_training_RGB_order_sha256=image_chain.hexdigest(),all_loss_grad_and_parameter_checks_passed=True,
            training_compute_seconds=train_seconds,mean_guarded_step_seconds=train_seconds/128,
            training_images_per_second=1024/train_seconds,wall_seconds=time.perf_counter()-arm_start,
            checkpoints=checkpoints,final_state_sha256=R.state_hash(model.state_dict()),weights_saved=False)
        save(name+'-result.json',out);arms[name]=out;del model,optimizer
    assert arms['frozen_base']['exact_training_RGB_order_sha256']==arms['finetune_base']['exact_training_RGB_order_sha256']
    assert arms['frozen_TA']['exact_training_RGB_order_sha256']==arms['finetune_TA']['exact_training_RGB_order_sha256']
    effects={}
    for step in ['0','64','128']:
        scores={name:arm['checkpoints'][step]['development_clean']['mean_LL_nats'] for name,arm in arms.items()}
        Gf=scores['frozen_base']-scores['frozen_TA'];Gt=scores['finetune_base']-scores['finetune_TA']
        effects[step]=dict(G_frozen=Gf,G_finetune=Gt,D=Gt-Gf)
    def judge(value):return 'positive' if value>.02 else 'negative' if value<-.02 else 'indeterminate'
    last=effects['128'];jf,jt=judge(last['G_frozen']),judge(last['G_finetune'])
    interpretation='indeterminate_at_fixed_short_development_budget'
    if jf==jt and jf!='indeterminate':interpretation='same_direction_in_this_development_run_only'
    elif jf!='indeterminate' and jt!='indeterminate':interpretation='opposite_directions_in_this_development_run_only'
    result=dict(status='completed_single_fixed_CPU_stage_pilot',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='measurement applicability development diagnosis, not originality or independent generalization evidence',
        effects_by_fixed_checkpoint=effects,final_judgements=dict(frozen=jf,finetune=jt),interpretation=interpretation,
        final_metrics={name:arm['checkpoints']['128']['development_clean'] for name,arm in arms.items()},
        arm_training_compute_seconds={name:arm['training_compute_seconds'] for name,arm in arms.items()},
        total_guarded_training_compute_seconds=sum(arm['training_compute_seconds'] for arm in arms.values()),
        total_wall_seconds=time.perf_counter()-started,CPU_budget_seconds=600,threads=torch.get_num_threads(),
        process_peak_RSS_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),reused_safe_step_code_sha256=sha(REPAIR),
        initial_state_sha256=common_init,all4arms_matched_training_sample_budget=True,
        all4arms_finite_and_BNstats_fixed=True,training_RGB_exactly_matched_within_policy_across_stages=True,
        development_source_photos=64,development_already_used_previously=True,no_independent_test_claim=True,
        no_new_data_or_software=True,CPU_only=True,GPU=False,ARIS=False,A100=False,
        long_plan_remains_paused=True,no_weights_saved=True,no_more_steps_seeds_or_tuning=True)
    save('pilot-results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='final_metrics'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('pilot-failure.json',dict(status='stopped_no_retuning',type=type(error).__name__,error=str(error),CPU_only=True,GPU=False))
        raise
