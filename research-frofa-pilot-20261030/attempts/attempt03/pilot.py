"""Locked real-feature pilot. All model execution is local and budgeted."""
import os,sys,json,time,hashlib,io,zipfile,gc,traceback,inspect,math,collections
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import t as student_t
from safetensors.torch import save_file,load_file
from transformers import SiglipVisionModel,SiglipVisionConfig,SiglipImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead,SiglipPreTrainedModel
from types import SimpleNamespace
from runtime import ROOT,P,Budget,Stop,save,sha,utc
from augmentation import SupportControls,action
from fixture_controls import nodes_weights,perturb

B=Budget();ARMS=P['arms'];DEVICE=torch.device('mps')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise Stop('MPS fallback must be explicitly disabled')
if not torch.backends.mps.is_available():raise Stop('MPS unavailable; CPU training fallback forbidden')
torch.mps.set_per_process_memory_fraction(min(1.,10*1024**3/torch.mps.recommended_max_memory()))

def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def inspect_libraries():
    import transformers,safetensors,augmentation
    sources={}
    for module in [inspect.getmodule(SiglipMultiheadAttentionPoolingHead),inspect.getmodule(SiglipVisionConfig),inspect.getmodule(SiglipImageProcessor),augmentation]:
        path=Path(module.__file__);sources[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
    config=SiglipVisionConfig.from_pretrained(ROOT/'assets/siglip',local_files_only=True,trust_remote_code=False)
    expected={'hidden_size':768,'intermediate_size':3072,'num_attention_heads':12,'image_size':224,'patch_size':16,'hidden_act':'gelu_pytorch_tanh','layer_norm_eps':1e-6,'attention_dropout':0.0}
    for key,value in expected.items():
        if getattr(config,key)!=value:raise Stop('unexpected SigLIP config '+key)
    source=inspect.getsource(SiglipMultiheadAttentionPoolingHead.forward)
    if 'self.attention(probe, hidden_state, hidden_state)[0]' not in source or 'residual + self.mlp(hidden_state)' not in source:raise Stop('MAP source architecture mismatch')
    save(ROOT/'library-source-lock.json',{'sources':sources,'MAP_forward':source,'vision_config':config.to_dict(),'transformers':transformers.__version__,'safetensors':safetensors.__version__,'torch':torch.__version__,'numpy':np.__version__,'mps_fallback':False})
    return config

def manifest():
    B.phase('decode_deduplicate_manifest');start=time.perf_counter();seen={};duplicates=[];all_rows=[]
    with zipfile.ZipFile(ROOT/'assets/EuroSAT_RGB.zip') as z:
        names=sorted(n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg')) and not n.startswith('__MACOSX/'))
        for i,name in enumerate(names):
            image=Image.open(io.BytesIO(z.read(name))).convert('RGB');array=np.asarray(image)
            if array.shape!=(64,64,3):raise Stop('unexpected decoded RGB shape '+name)
            label=Path(name).parent.name;digest=hashlib.sha256(array.tobytes()).hexdigest()
            row={'member':name,'class':label,'rgb_sha256':digest}
            all_rows.append(row)
            if digest in seen:
                if seen[digest]['class']!=label:raise Stop('cross-label decoded RGB duplicate '+digest)
                duplicates.append({'kept':seen[digest]['member'],'duplicate':name,'rgb_sha256':digest})
            else:seen[digest]=row
            if i%1000==0:B.check();B.event('decode_progress',images=i,total=len(names))
    classes=sorted(set(r['class'] for r in seen.values()))
    if len(classes)!=10:raise Stop('expected ten classes')
    selected=[]
    for label_index,label in enumerate(classes):
        rows=[dict(r) for r in seen.values() if r['class']==label]
        for row in rows:row['sort_hash']=hashlib.sha256(b'frofa-es-v1\0'+bytes.fromhex(row['rgb_sha256'])).hexdigest()
        rows.sort(key=lambda r:(r['sort_hash'],r['rgb_sha256']))
        if len(rows)<500:raise Stop('insufficient unique images '+label)
        for rank,row in enumerate(rows[:500]):
            row.update(label=label_index,class_rank=rank,index=len(selected),split='trainpool' if rank<100 else ('validation' if rank<200 else 'test'),calibration=rank<10)
            selected.append(row)
    save(ROOT/'decoded-inventory.json',all_rows);save(ROOT/'deduplication.json',{'duplicates':duplicates,'unique':len(seen),'decoded':len(names)})
    save(ROOT/'manifest.json',{'classes':classes,'rows':selected,'protocol_sha256':sha(ROOT/'protocol.json')})
    save(ROOT/'manifest-lock.json',{'sha256':sha(ROOT/'manifest.json'),'count':len(selected),'decode_seconds':time.perf_counter()-start})
    B.timing('cpu_decode_deduplicate_wall',time.perf_counter()-start);B.event('manifest_locked',images=len(selected),unique=len(seen),within_label_duplicates=len(duplicates),sha256=sha(ROOT/'manifest.json'))
    return selected

def image_batch(z,rows):
    return [Image.open(io.BytesIO(z.read(row['member']))).convert('RGB') for row in rows]

def encoder_call(model,pixels,device,category):
    B.reserve('encoder_images',len(pixels));B.reserve('encoder_'+category+'_images',len(pixels));B.check(torch)
    if device.type=='mps':torch.mps.synchronize()
    start=time.perf_counter();cpu=time.process_time()
    with torch.inference_mode():output=model(pixel_values=pixels.to(device)).last_hidden_state
    if device.type=='mps':torch.mps.synchronize()
    elapsed=time.perf_counter()-start;B.timing('encoder_'+category+'_sync_wall',elapsed);B.timing('encoder_'+category+'_cpu_process',time.process_time()-cpu)
    result=output.detach().cpu().numpy().copy();del output
    if result.dtype!=np.float32 or result.shape[1:]!=(196,768) or not np.isfinite(result).all():raise Stop('encoder feature shape/dtype/finite failure')
    B.check(torch);B.flush();return result,elapsed

def cache_features(rows):
    B.phase('encoder_parity_and_cache');config_path=ROOT/'assets/siglip'
    processor=SiglipImageProcessor.from_pretrained(config_path,local_files_only=True,trust_remote_code=False)
    model=SiglipVisionModel.from_pretrained(config_path,local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager',dtype=torch.float32)
    model.eval().requires_grad_(False);model.vision_model.use_head=False;del model.vision_model.head
    if any(p.requires_grad for p in model.parameters()):raise Stop('encoder is not frozen')
    dev=[r for r in rows if r['class_rank']<2]
    with zipfile.ZipFile(ROOT/'assets/EuroSAT_RGB.zip') as z:
        pixels=processor(images=image_batch(z,dev),return_tensors='pt')['pixel_values']
        cpu_outputs=[]
        for start in range(0,20,4):
            out,_=encoder_call(model,pixels[start:start+4],torch.device('cpu'),'parity_cpu');cpu_outputs.append(out)
        cpu_outputs=np.concatenate(cpu_outputs);model.to(DEVICE)
        mps_outputs,_=encoder_call(model,pixels,DEVICE,'parity_mps')
        error=np.linalg.norm((mps_outputs.astype(np.float64)-cpu_outputs).reshape(20,-1),axis=1)/np.linalg.norm(cpu_outputs.astype(np.float64).reshape(20,-1),axis=1)
        save(ROOT/'encoder-parity.json',{'relative_l2':error.tolist(),'maximum':float(error.max()),'threshold':1e-4,'image_ids':[r['rgb_sha256'] for r in dev],'passed':bool(np.all(error<=1e-4))})
        B.event('encoder_parity',maximum_relative_l2=float(error.max()),passed=bool(np.all(error<=1e-4)))
        if np.any(error>1e-4):raise Stop('encoder CPU/MPS parity threshold failed')
        del pixels,cpu_outputs,mps_outputs
        B.check(torch,disk=True)
        path=ROOT/'cache/features.fp32.npy';path.parent.mkdir(exist_ok=True)
        cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(5000,196,768))
        hashes=[];timings=[]
        for start in range(0,len(rows),32):
            selected=rows[start:start+32];pre=time.perf_counter();pixels=processor(images=image_batch(z,selected),return_tensors='pt')['pixel_values'];B.timing('cpu_image_preprocess_wall',time.perf_counter()-pre)
            out,seconds=encoder_call(model,pixels,DEVICE,'cache_mps');timings.append({'n':len(out),'seconds':seconds})
            cache[start:start+len(out)]=out
            hashes.extend(hashlib.sha256(v.tobytes()).hexdigest() for v in out)
            del pixels,out
            if start%320==0:B.event('cache_progress',completed=min(start+32,5000),total=5000);cache.flush();B.check(torch,disk=True)
        cache.flush();del cache
        read=np.load(path,mmap_mode='r',allow_pickle=False)
        for i,row in enumerate(read):
            if hashlib.sha256(row.tobytes()).hexdigest()!=hashes[i]:raise Stop('FP32 cache readback mismatch '+str(i))
        save(ROOT/'cache-lock.json',{'path':'cache/features.fp32.npy','sha256':sha(path),'row_sha256':hashes,'shape':list(read.shape),'dtype':str(read.dtype),'finite_all':True,'readback_bitwise':True,'manifest_sha256':sha(ROOT/'manifest.json'),'encoder_timing':timings})
    del model,processor;gc.collect();torch.mps.empty_cache();B.check(torch,disk=True);B.event('cache_verified',rows=5000,bytes=path.stat().st_size,sha256=sha(path));return read

class Head(torch.nn.Module):
    def __init__(self,config):
        super().__init__();self.pool=SiglipMultiheadAttentionPoolingHead(config);self.classifier=torch.nn.Linear(768,10)
        self.pool.apply(lambda module:SiglipPreTrainedModel._init_weights(SimpleNamespace(config=config),module))
        torch.nn.init.normal_(self.classifier.weight,std=768**-0.5);torch.nn.init.zeros_(self.classifier.bias)
    def forward(self,x):return self.classifier(self.pool(x))

def new_head(config,seed):
    torch.manual_seed(seed);head=Head(config)
    if head.pool.attention.dropout!=0 or head.pool.mlp.activation_fn.__class__.__name__ not in ('PytorchGELUTanh','GELUTanh'):
        # Activation name is evidence; config itself is the definitive pinned tanh-GELU contract.
        if config.hidden_act!='gelu_pytorch_tanh' or head.pool.attention.dropout!=0:raise Stop('head activation/dropout mismatch')
    digest=state_hash(head.state_dict());return head,digest

def support_indices(rows,seed):
    if seed==0:return np.array([r['index'] for r in rows if r['calibration']],np.int64)
    rng=np.random.default_rng(seed);indices=[]
    for label in range(10):
        candidates=[r['index'] for r in rows if r['label']==label and 10<=r['class_rank']<100]
        indices.extend(rng.choice(candidates,size=10,replace=False).tolist())
    return np.array(indices,np.int64)

def orders(seed):
    rng=np.random.default_rng(10000+seed);batches=[]
    for epoch in range(80):
        shuffled=[rng.permutation(np.arange(c*10,(c+1)*10)) for c in range(10)]
        for step in range(5):
            batch=np.concatenate([r[step*2:step*2+2] for r in shuffled]);batches.append(rng.permutation(batch))
    return np.stack(batches)

def moment_preflight(cache,rows):
    B.phase('real_feature_analytic_moment_preflight');checks=[]
    indices=[next(r['index'] for r in rows if r['label']==label and r['class_rank']==0) for label in (0,1)]
    for a in (0.1,1.):
        support=SupportControls(np.asarray(cache[indices]),a,True,B)
        for image,p in enumerate(support.packed):
            for c in (0,1,2,3):
                u=p['u'][c];d=float(p['d'][c,0]);q,w,_=nodes_weights(u,a);g=perturb(u[None,:],d,q[:,None]);mu=w@g;fac=(g-mu).T*np.sqrt(w);cov=fac@fac.T
                J=p['q'].shape[1]
                implicit=action(u[None,:],np.array([[d]]),p['q'][c:c+1],p['sqrtw'][c:c+1],p['low'][c:c+1],p['high'][c:c+1],p['mu'][c:c+1],np.eye(J)).T
                estimates={'mu_scaled':float(np.max(np.abs(p['mu'][c]-mu)/d)),
                           'cov_scaled':float(np.max(np.abs(implicit@implicit.T-cov)/(d*d))),
                           'GE_var_scaled':float(np.max(np.abs(p['var'][c]-np.diag(cov))/(d*d))),
                           'rounded_factor_cov_scaled':float(np.max(np.abs(implicit.astype(np.float32).astype(np.float64)@implicit.astype(np.float32).astype(np.float64).T-cov)/(d*d)))}
                passed=all(v<=1e-5 for v in estimates.values());checks.append({'a':a,'image_id':rows[indices[image]]['rgb_sha256'],'channel':c,'tokens':196,'nodes':J,'d':d,**estimates,'passed':passed})
                if not passed:save(ROOT/'real-moment-checks.json',checks);raise Stop('real-feature analytic moment check failed')
        del support;gc.collect();B.check(torch)
    save(ROOT/'real-moment-checks.json',{'checks':checks,'passed':True,'exact_object':'continuous conditional float64 moment operator; FP32 feature transfer rounding recorded, no finite-precision bitwise Gaussian moment claim','no_MC_fitting':True})
    B.event('real_moment_checks_passed',cases=len(checks),maximum=max(max(v for k,v in row.items() if k.endswith('_scaled')) for row in checks))

def run_fit(config,controls,seed,arm,lr,wd,nupdates=400,phase='confirm',a_index=0):
    head,init_hash=new_head(config,999 if phase=='preflight' else seed);head.to(DEVICE);head.train()
    decay=[p for p in head.parameters() if p.ndim>=2];nodecay=[p for p in head.parameters() if p.ndim<2]
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':wd},{'params':nodecay,'weight_decay':0.}],lr=lr,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    order=orders(seed);order_hash=hashlib.sha256(order.tobytes()).hexdigest();rng=np.random.default_rng(np.random.SeedSequence([20261030,{'preflight':0,'calibration':1,'confirm':2}[phase],seed,ARMS.index(arm),a_index]))
    labels=np.repeat(np.arange(10),10);times=[];losses=[];B.event('fit_start',phase=phase,seed=seed,arm=arm,lr=lr,wd=wd,a=controls.a,updates=nupdates,init_sha256=init_hash,batch_order_sha256=order_hash)
    for step in range(1,nupdates+1):
        B.reserve('updates',1);B.reserve('training_exposures',20);B.reserve(phase+'_training_exposures',20);B.reserve(arm+'_training_exposures',20)
        if phase=='preflight':B.reserve('preflight_updates',1)
        if arm!='clean':B.reserve('augmentation_views',20);B.reserve(arm+'_augmentation_views',20)
        B.check(torch);torch.mps.synchronize();start=time.perf_counter();cpu=time.process_time()
        batch=order[step-1];t0=time.perf_counter();array=controls.batch(batch,arm,rng);B.timing('cpu_augmentation_wall',time.perf_counter()-t0)
        gpu_start=time.perf_counter();features=torch.from_numpy(array).to(DEVICE);target=torch.from_numpy(labels[batch]).to(DEVICE)
        rate=lr*(step/20 if step<=20 else .5*(1+math.cos(math.pi*(step-20)/380)))
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True);logits=head(features);loss=torch.nn.functional.cross_entropy(logits,target,label_smoothing=0.);loss.backward();optimizer.step();torch.mps.synchronize()
        elapsed=time.perf_counter()-start;gpu_elapsed=time.perf_counter()-gpu_start;value=float(loss.item());times.append(elapsed);losses.append(value)
        B.timing('head_training_mps_sync_wall',gpu_elapsed);B.timing('training_cpu_process',time.process_time()-cpu)
        if not np.isfinite(value):raise Stop('nonfinite head loss')
        del features,target,logits,loss,array
        if step%40==0:B.event('fit_progress',phase=phase,seed=seed,arm=arm,step=step,loss=value,mean_step_seconds=float(np.mean(times[-40:])));B.check(torch,disk=True)
    optimizer.zero_grad(set_to_none=True);del optimizer;head.eval()
    if nupdates==400:B.reserve('complete_fits',1)
    info={'phase':phase,'seed':seed,'arm':arm,'lr':lr,'wd':wd,'a':controls.a,'updates':nupdates,'initial_sha256':init_hash,'batch_order_sha256':order_hash,'step_seconds':times,'losses':losses,'final_sha256':state_hash(head.state_dict())}
    B.flush();B.event('fit_complete',phase=phase,seed=seed,arm=arm,updates=nupdates,mean_step_seconds=float(np.mean(times)))
    return head,info

def evaluate(head,cache,indices,category,labels=None):
    head.eval();predictions=[];times=[]
    with torch.inference_mode():
        for start in range(0,len(indices),20):
            chosen=indices[start:start+20];B.reserve('eval_images',len(chosen));B.reserve(category+'_eval_images',len(chosen));B.check(torch);torch.mps.synchronize();t0=time.perf_counter();cpu=time.process_time()
            features=torch.from_numpy(np.array(cache[chosen],copy=True)).to(DEVICE);logits=head(features);torch.mps.synchronize();times.append(time.perf_counter()-t0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist());B.timing('head_evaluation_mps_sync_wall',times[-1]);B.timing('evaluation_cpu_process',time.process_time()-cpu)
            del features,logits
    predictions=np.array(predictions,np.int64)
    result={'count':len(indices),'batch_seconds':times}
    if labels is not None:result.update(correct=int(np.sum(predictions==labels[indices])),accuracy_percent=float(100*np.mean(predictions==labels[indices])))
    return result,predictions

def preflight(config,cache,rows):
    moment_preflight(cache,rows);B.phase('fixed_200_update_resource_preflight');si=support_indices(rows,0);support=np.array(cache[si]);measure=[];prep=[];last=None
    for ai,a in enumerate((.1,1.)):
        t0=time.perf_counter();controls=SupportControls(support,a,True,B);prep.append(time.perf_counter()-t0);B.check(torch);B.event('preflight_controls_ready',a=a,bytes=controls.bytes,seconds=prep[-1])
        for arm in ARMS:
            head,info=run_fit(config,controls,0,arm,.001,.1,20,'preflight',ai);measure.append(info)
            if ai==1 and arm=='GE':last=head
            else:del head;gc.collect();torch.mps.empty_cache()
        del controls;gc.collect()
    validation=np.array([r['index'] for r in rows if r['split']=='validation'],np.int64)
    eval_info,_=evaluate(last,cache,validation[:200],'preflight',None);del last;gc.collect();torch.mps.empty_cache()
    per_arm={arm:max(float(np.quantile(row['step_seconds'][10:],.9)) for row in measure if row['arm']==arm) for arm in ARMS}
    fit_counts={'clean':10,'C':8,'E':6,'GC':6,'GE':6};estimated=sum(400*fit_counts[arm]*per_arm[arm] for arm in ARMS)+6*max(prep)+(96000/20)*float(np.quantile(eval_info['batch_seconds'],.9))+180
    remaining=14400-B.elapsed();passed=estimated<=.75*remaining and B.elapsed()<2700
    result={'passed':passed,'elapsed_seconds':B.elapsed(),'remaining_wall_seconds':remaining,'projected_remaining_seconds':estimated,'allowed_projection_seconds':.75*remaining,'slow_p90_step_seconds':per_arm,'support_prep_seconds':prep,'evaluation_timing':eval_info,'measurements':measure,'preflight_weights_discarded':True}
    save(ROOT/'preflight.json',result);B.event('resource_preflight_result',passed=passed,estimate_seconds=estimated,allowed_seconds=.75*remaining,elapsed_seconds=B.elapsed())
    if not passed:raise Stop('resource projection gate failed; recipe unchanged')
    B.s['preflight_passed']=True;B.phase('preflight_passed');return result

def calibrate(config,cache,rows):
    B.phase('calibration_six_complete_fits');si=support_indices(rows,0);x=np.array(cache[si]);val=np.array([r['index'] for r in rows if r['split']=='validation'],np.int64);labels=np.array([r['label'] for r in rows]);results=[]
    controls=SupportControls(x,.1,False,B)
    for lr in (.0003,.001):
        for wd in (.01,.1):
            head,info=run_fit(config,controls,0,'clean',lr,wd,400,'calibration');ev,_=evaluate(head,cache,val,'calibration',labels);info['validation']=ev;results.append(info);save(ROOT/f'fits/calibration-clean-{lr}-{wd}.json',info);del head;gc.collect();torch.mps.empty_cache()
    best=sorted(results,key=lambda r:(-r['validation']['accuracy_percent'],r['lr'],-r['wd']))[0];lr,wd=best['lr'],best['wd'];del controls
    cres=[]
    for ai,a in enumerate((.1,1.)):
        controls=SupportControls(x,a,False,B);head,info=run_fit(config,controls,0,'C',lr,wd,400,'calibration',ai);ev,_=evaluate(head,cache,val,'calibration',labels);info['validation']=ev;cres.append(info);save(ROOT/f'fits/calibration-C-{a}.json',info);del head,controls;gc.collect();torch.mps.empty_cache()
    best_a=sorted(cres,key=lambda r:(-r['validation']['accuracy_percent'],r['a']))[0]['a'];chosen={'lr':lr,'wd':wd,'a':best_a,'calibration_results':results+cres,'calibration_support':si.tolist(),'calibration_weights_discarded':True}
    save(ROOT/'hyperparameters-locked.json',chosen);B.event('calibration_locked',lr=lr,wd=wd,a=best_a);return chosen

def confirmation(config,cache,rows,hp):
    B.phase('confirmation_30_complete_fits');models=[]
    for seed in range(1,7):
        si=support_indices(rows,seed);t0=time.perf_counter();controls=SupportControls(np.array(cache[si]),hp['a'],True,B);B.timing('confirmation_support_preparation_wall',time.perf_counter()-t0);B.event('episode_controls_ready',seed=seed,bytes=controls.bytes,prep_seconds=time.perf_counter()-t0)
        baseline=None
        for arm in ARMS:
            head,info=run_fit(config,controls,seed,arm,hp['lr'],hp['wd'],400,'confirm',0 if hp['a']==.1 else 1)
            pairing=(info['initial_sha256'],info['batch_order_sha256'])
            if baseline is None:baseline=pairing
            elif pairing!=baseline:raise Stop('episode initialization/order pairing failure')
            info['support_indices']=si.tolist();info['support_ids']=[rows[i]['rgb_sha256'] for i in si]
            path=ROOT/f'heads/episode-{seed}-{arm}.safetensors';path.parent.mkdir(exist_ok=True);save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},path)
            info.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path));save(ROOT/f'fits/episode-{seed}-{arm}.json',info);models.append(info);del head;gc.collect();torch.mps.empty_cache();B.check(torch,disk=True)
        del controls;gc.collect();B.event('episode_complete',seed=seed)
    save(ROOT/'all-confirmation-models-frozen.json',{'utc':utc(),'models':[{'seed':m['seed'],'arm':m['arm'],'checkpoint':m['checkpoint'],'sha256':m['checkpoint_sha256'],'final_state_sha256':m['final_sha256']} for m in models],'count':len(models),'before_any_test_metric':True});return models

def final_test(config,cache,rows,models):
    if len(models)!=30:raise Stop('all 30 models required before test')
    B.phase('single_final_test_pass');indices=np.array([r['index'] for r in rows if r['split']=='test'],np.int64);labels=np.array([r['label'] for r in rows]);results=[]
    for m in models:
        head,_=new_head(config,m['seed']);head.load_state_dict(load_file(ROOT/m['checkpoint']),strict=True)
        if state_hash(head.state_dict())!=m['final_sha256']:raise Stop('frozen final model hash mismatch')
        head.to(DEVICE);ev,preds=evaluate(head,cache,indices,'test',labels);row={'seed':m['seed'],'arm':m['arm'],**ev};results.append(row);save(ROOT/f'test/episode-{m["seed"]}-{m["arm"]}.json',row);np.save(ROOT/f'test/episode-{m["seed"]}-{m["arm"]}-pred.npy',preds,allow_pickle=False);del head;gc.collect();torch.mps.empty_cache();B.event('final_test_progress',models_completed=len(results),total=30)
    return results

def statistics(results):
    table={seed:{r['arm']:r['accuracy_percent'] for r in results if r['seed']==seed} for seed in range(1,7)}
    differences={name:[] for name in ['I','dC','dE','g','h']}
    for seed,r in table.items():
        differences['I'].append((r['C']-r['E'])-(r['GC']-r['GE']));differences['dC'].append(r['GC']-r['C']);differences['dE'].append(r['GE']-r['E']);differences['g'].append(r['C']-r['clean']);differences['h'].append(r['C']-r['E'])
    estimates={}
    for name,values in differences.items():
        mean=float(np.mean(values));se=float(np.std(values,ddof=1)/np.sqrt(6));q95=float(student_t.ppf(.95,5));qbon=float(student_t.ppf(1-.05/6,5))
        estimates[name]={'episode_differences_pp':values,'mean_pp':mean,'se_pp':se,'one_sided_95_lower_pp':mean-q95*se,'CI90_pp':[mean-q95*se,mean+q95*se],'Bonferroni_family95_CI_pp':[mean-qbon*se,mean+qbon*se]}
    phenomenon=all(estimates[k]['mean_pp']>=1 and estimates[k]['one_sided_95_lower_pp']>0 for k in ['g','h'])
    equivalent=all(estimates[k]['CI90_pp'][0]>=-.5 and estimates[k]['CI90_pp'][1]<=.5 for k in ['I','dC','dE'])
    insufficient=any(estimates[k]['Bonferroni_family95_CI_pp'][0]>.5 or estimates[k]['Bonferroni_family95_CI_pp'][1]<-.5 for k in ['I','dC','dE'])
    decision='phenomenon_gate_failed_stop' if not phenomenon else ('phenomenon_and_equivalence_passed_separate_review_required' if equivalent else ('local_insufficiency_stop' if insufficient else 'undetermined_stop'))
    return {'accuracy_percent_by_episode':table,'estimates':estimates,'df':5,'phenomenon_passed':phenomenon,'equivalence_passed':equivalent,'local_insufficiency':insufficient,'decision':decision,'scope':'conditional on this fixed pool/test, six paired episodes; no test bootstrap, spatial OOD or novelty claim; no Q continuation'}

def archive_metrics(result):
    save(ROOT/'RESULTS.json',result);B.check(torch,disk=True);B.s['phase']='completed';B.s['elapsed_seconds']=B.elapsed();B.s['finished_utc']=utc();B.flush()
    lines=['# Frozen-feature pilot result','',f"Decision: {result['decision']}",f"Continuous elapsed: {B.elapsed():.3f} seconds",'',json.dumps(result,indent=2), '', 'No automatic continuation; scope and resource protocol remain fixed.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(part in {'deps','wheels','hf-cache','__pycache__'} for part in p.relative_to(ROOT).parts) and p.name not in {'artifact-hashes.json','supervisor.json'} and not p.name.endswith('.log')}
    save(ROOT/'artifact-hashes.json',files);B.event('pilot_complete',decision=result['decision'],elapsed_seconds=B.elapsed());B.check(torch)

def main():
    try:
        B.check(torch,disk=True);config=inspect_libraries();rows=manifest();cache=cache_features(rows);preflight(config,cache,rows);B.event('full_training_authorized_by_preflight',fits=36)
        hp=calibrate(config,cache,rows);models=confirmation(config,cache,rows,hp);results=final_test(config,cache,rows,models);result=statistics(results);archive_metrics(result)
    except Exception as e:
        save(ROOT/'STOP.json',{'utc':utc(),'phase':B.s['phase'],'reason':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':B.elapsed(),'counts':B.s['counts'],'interpretation':'Resource/implementation stop; partial pairs descriptive only. No absence-of-effect/equivalence claim.'});B.s['phase']='stopped';B.flush();raise
if __name__=='__main__':main()
