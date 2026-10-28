"""Serial four-cell representation-learning pilot; requires a frozen protocol lock."""
from common import *
from datetime import datetime,timezone
import gc,resource
RUN_START=None
WALL_CAP=7200
PROGRESS={'stage':'startup','cell':None,'epoch':0,'updates':0}

def guard():
    if (ROOT/'STOP.request').exists():raise RuntimeError('User/resource stop requested; preserve partial results.')
    if RUN_START is not None and time.perf_counter()-RUN_START>WALL_CAP:
        raise RuntimeError('Budget exhausted: incomplete measurement, not scientific negative.')

def features(model,x):
    x=model._process_input(x)
    x=torch.cat((model.class_token.expand(x.shape[0],-1,-1),x),dim=1)
    return model.encoder(x)[:,0]

@torch.no_grad()
def evaluate(model,images,rows,batch_size=16):
    model.eval();logits=[];selected_features=[];active=0.;max_driver=0
    for start in range(0,len(rows),batch_size):
        guard()
        batch=rows[start:start+batch_size]
        x=torch.stack([eval_tensor(images[r['source_id']]) for r in batch]).to('mps')
        torch.mps.synchronize();t=time.perf_counter()
        f=features(model,x);z=model.heads(f)
        logits.append(z.detach().cpu().numpy())
        if start==0:selected_features=f[:8].detach().cpu().numpy()
        active+=pace(t)
        max_driver=max(max_driver,torch.mps.driver_allocated_memory());guard()
    model.train()
    z=np.concatenate(logits);labels=np.array([r['class_id'] for r in rows],dtype=np.int64)
    assert np.isfinite(z).all(),'Nonfinite development logits.'
    pred=z.argmax(1);acc_by_class=[float(np.mean(pred[labels==c]==c)) for c in range(37)]
    shifted=z.astype(np.float64)-z.max(axis=1,keepdims=True)
    ll=-shifted[np.arange(len(labels)),labels]+np.log(np.exp(shifted).sum(1))
    return {'logits':z,'labels':labels,'source_ids':np.array([r['source_id'] for r in rows]),
        'predictions':pred,'selected_features':selected_features}, {
        'error':float(np.mean(pred!=labels)),'balanced_error':1-float(np.mean(acc_by_class)),
        'NLL':float(np.mean(ll)),'per_class_accuracy':acc_by_class,'MPS_active_seconds':active,
        'max_observed_MPS_driver_bytes':max_driver}

def save_arrays(path,d):np.savez_compressed(path,**d)

def parameter_digest(model,trainable=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad!=trainable:continue
        h.update(name.encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    global RUN_START,WALL_CAP
    setup_mps()
    protocol=ROOT/'protocol.json';lock=json.loads((ROOT/'protocol-lock.json').read_text())
    assert sha(protocol)==lock['protocol_sha256']
    for name,digest in lock['code_sha256'].items():assert sha(ROOT/name)==digest,name
    cfg=json.loads(protocol.read_text())
    assert cfg['seed']==20261022 and isinstance(cfg['epochs'],int) and cfg['epochs'] in (10,30)
    assert cfg['batch']==16 and cfg['wall_cap_seconds']==7200
    assert sorted(map(tuple,cfg['cell_order']))==sorted([('front','weak'),('front','ra'),('back','weak'),('back','ra')])
    assert cfg['epochs'] in cfg['evaluation_epochs'] and all(0<e<=cfg['epochs'] for e in cfg['evaluation_epochs'])
    assert cfg['primary_metric']=='source-photo micro classification error on clean development images'
    assert cfg['optimizer']['adapter_lr']==.0003 and cfg['optimizer']['head_lr']==.001 and cfg['optimizer']['weight_decay']==.01
    assert cfg['architecture']['trainable_total']==65317
    RUN_START=time.perf_counter();WALL_CAP=cfg['wall_cap_seconds'];guard()
    assert sha(ROOT/'data/vit_b_16-c867db91.pth')==cfg['weight_sha256']
    manifest=json.loads((ROOT/'fit-manifest.json').read_text())
    assert sha(ROOT/'fit-manifest.json')==cfg['fit_manifest_sha256']
    rows=validated_rows(manifest)
    train_rows=[r for r in rows if r['candidate_partition']=='train'];dev_rows=[r for r in rows if r['candidate_partition']=='development']
    images={}
    for r in rows:
        guard()
        p=ROOT/'data/images'/(r['source_id']+'.jpg');assert sha(p)==r['image_sha256']
        images[r['source_id']]=Image.open(p).convert('RGB').copy()
    seed=cfg['seed'];epochs=cfg['epochs'];batch=cfg['batch'];steps_per_epoch=math.ceil(len(train_rows)/batch)
    result_root=ROOT/'results';result_root.mkdir(exist_ok=False)
    run_start=RUN_START;all_summaries=[];initial_reference=None
    initial_trainable_reference=None
    dump(result_root/'started.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':lock['protocol_sha256'],'device':'mps:0','PID':os.getpid(),
        'CPU_threads':torch.get_num_threads(),'MPS_fallback':False,
        'actual_process_nice':os.getpriority(os.PRIO_PROCESS,0),
        'CUDA_available':torch.cuda.is_available(),
        'versions':{'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
                    'Pillow':__import__('PIL').__version__,
                    'numpy':np.__version__,'python':__import__('sys').version},
        'command':__import__('sys').argv})
    for layout,aug in cfg['cell_order']:
        guard();PROGRESS.update(stage='model_build',cell=f'{layout}-{aug}',epoch=0,updates=0)
        cell=f'{layout}-{aug}';out=result_root/cell;out.mkdir()
        model=build_model(layout,seed).to('mps').train();opt=optimizer(model)
        guard()
        dump(out/'optimizer-actual.json',{'defaults':opt.defaults,
            'groups':[{'initial_lr':g['lr'],'weight_decay':g['weight_decay'],
                       'parameters':sum(p.numel() for p in g['params'])} for g in opt.param_groups]})
        trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        frozen_before=parameter_digest(model)
        initial_slot_digest=hashlib.sha256()
        for value in trainable.values():initial_slot_digest.update(value.numpy().tobytes())
        initial_slot_hash=initial_slot_digest.hexdigest()
        if initial_trainable_reference is None:initial_trainable_reference=initial_slot_hash
        assert initial_slot_hash==initial_trainable_reference,'Initial adapter-slot/head tensors differ.'
        torch.save({'trainable_state':trainable,'protocol_sha256':lock['protocol_sha256'],
            'layout':layout,'augmentation':aug,'seed':seed},out/'initial-trainable-checkpoint.pt')
        dump(out/'trainable-parameters.json',{'parameters':[{'name':n,'shape':list(p.shape),'numel':p.numel()} for n,p in model.named_parameters() if p.requires_grad],
            'total':sum(p.numel() for p in model.parameters() if p.requires_grad),'backbone':36864,'head':28453})
        initial,initial_metrics=evaluate(model,images,dev_rows,batch)
        guard()
        if initial_reference is None:initial_reference=initial['logits'].copy()
        assert np.max(np.abs(initial['logits']-initial_reference))<1e-5
        save_arrays(out/'development-epoch000.npz',initial)
        initial_feat=initial['selected_features'].copy()
        curves=[{'epoch':0,'development':initial_metrics}];view_hashes=[];active=0.;cell_start=time.perf_counter()
        actual_updates=0;actual_presentations=0;max_driver=initial_metrics['max_observed_MPS_driver_bytes']
        for epoch in range(epochs):
            guard();PROGRESS.update(stage='training',epoch=epoch+1)
            order=np.random.default_rng(keyseed('order-v1',seed,epoch)).permutation(len(train_rows))
            view_digest=hashlib.sha256();tensor_digest=hashlib.sha256();order_digest=hashlib.sha256()
            loss_sum=0.;seen=0;gradmax=0.
            for start in range(0,len(order),batch):
                guard()
                selected=[train_rows[i] for i in order[start:start+batch]]
                batch_tensors=[]
                for r in selected:
                    im=augmented_pil(images[r['source_id']],r['source_id'],epoch,seed,aug)
                    view_digest.update(r['source_id'].encode());view_digest.update(im.tobytes())
                    image_tensor=tensor(im)
                    tensor_digest.update(r['source_id'].encode());tensor_digest.update(image_tensor.numpy().tobytes())
                    order_digest.update((r['source_id']+'\n').encode())
                    batch_tensors.append(image_tensor)
                x=torch.stack(batch_tensors).to('mps');y=torch.tensor([r['class_id'] for r in selected],device='mps')
                step=epoch*steps_per_epoch+start//batch;set_lr(opt,step,epochs*steps_per_epoch)
                torch.mps.synchronize();t=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(x),y)
                assert torch.isfinite(loss),f'{cell} nonfinite loss'
                loss.backward()
                grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                assert len(grads)==len(trainable),'Missing trainable gradient.'
                assert all(torch.isfinite(g).all() for g in grads),f'{cell} nonfinite gradients'
                gradmax=max(gradmax,max(float(g.detach().abs().max().cpu()) for g in grads))
                opt.step();active+=pace(t)
                actual_updates+=1;actual_presentations+=len(selected)
                PROGRESS['updates']=actual_updates
                max_driver=max(max_driver,torch.mps.driver_allocated_memory())
                value=float(loss.detach().cpu());loss_sum+=value*len(selected);seen+=len(selected)
                with (out/'steps.jsonl').open('a') as f:
                    f.write(json.dumps({'step':step+1,'epoch':epoch+1,'examples':len(selected),
                        'mean_CE':value,'adapter_lr':opt.param_groups[0]['lr'],
                        'head_lr':opt.param_groups[1]['lr'],'all_gradients_finite':True})+'\n')
                guard()
            view_hashes.append({'epoch':epoch,'ordered_RGB_sha256':view_digest.hexdigest(),
                'ordered_tensor_sha256':tensor_digest.hexdigest(),'source_order_sha256':order_digest.hexdigest()})
            curve={'epoch':epoch+1,'training_mean_CE':loss_sum/seen,'examples':seen,'gradient_absmax':gradmax}
            if epoch+1 in cfg['evaluation_epochs']:
                PROGRESS['stage']='evaluation'
                arrays,metrics=evaluate(model,images,dev_rows,batch)
                max_driver=max(max_driver,metrics['max_observed_MPS_driver_bytes'])
                save_arrays(out/f'development-epoch{epoch+1:03d}.npz',arrays)
                curve['development']=metrics
                if epoch+1==epochs:
                    final_features=arrays['selected_features']
            curves.append(curve)
            with (out/'curve.json').open('w') as f:json.dump(curves,f,indent=2)
            print(json.dumps({'cell':cell,'epoch':epoch+1,'train_CE':curve['training_mean_CE'],
                'development':curve.get('development',{}),'cell_wall_seconds':time.perf_counter()-cell_start}),flush=True)
        final_state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        changes={n:float((final_state[n]-trainable[n]).abs().max()) for n in trainable}
        assert any(v>0 for n,v in changes.items() if not n.startswith('heads.'))
        assert all(v>0 for n,v in changes.items() if n.startswith('heads.')),'Head unchanged.'
        frozen_after=parameter_digest(model)
        assert frozen_before==frozen_after,'Frozen base parameters changed.'
        effective={}
        for name,module in model.named_modules():
            if isinstance(module,QVLoRA):
                q=(module.bq@module.aq).detach().cpu();v=(module.bv@module.av).detach().cpu()
                effective[name]={'Q_delta_frobenius':float(q.norm()),'V_delta_frobenius':float(v.norm()),
                    'Bq_frobenius':float(module.bq.detach().norm().cpu()),'Bv_frobenius':float(module.bv.detach().norm().cpu()),
                    'K_delta_exact_zero_by_construction':True}
                assert effective[name]['Q_delta_frobenius']>0 and effective[name]['V_delta_frobenius']>0
        feature_rmse=float(np.sqrt(np.mean((final_features-initial_feat)**2)))
        assert feature_rmse>0 and math.isfinite(feature_rmse)
        assert actual_updates==epochs*steps_per_epoch and actual_presentations==epochs*len(train_rows)
        guard();PROGRESS['stage']='checkpoint'
        torch.save({'trainable_state':final_state,'protocol_sha256':lock['protocol_sha256'],
            'layout':layout,'augmentation':aug,'seed':seed},out/'trainable-checkpoint.pt')
        dump(out/'view-hashes.json',view_hashes)
        summary={'cell':cell,'layout':layout,'augmentation':aug,'seed':seed,'epochs':epochs,
            'updates':actual_updates,'source_presentations':actual_presentations,
            'initial_slot_and_head_sha256':initial_slot_hash,
            'frozen_parameter_SHA256_before':frozen_before,'frozen_parameter_SHA256_after':frozen_after,
            'trainable_checkpoint_sha256':sha(out/'trainable-checkpoint.pt'),
            'final_development':curves[-1]['development'],'parameter_absmax_changes':changes,
            'effective_Q_V_updates':effective,'fixed_eight_feature_RMSE':feature_rmse,
            'training_MPS_active_seconds':active,'cell_wall_seconds':time.perf_counter()-cell_start,
            'MPS_driver_bytes_at_finish':torch.mps.driver_allocated_memory(),
            'max_observed_MPS_driver_bytes':max_driver}
        dump(out/'summary.json',summary);all_summaries.append(summary)
        del model,opt,trainable,final_state,x,y;gc.collect();torch.mps.empty_cache()
        guard()
    for aug in ('weak','ra'):
        a=json.loads((result_root/f'front-{aug}/view-hashes.json').read_text())
        b=json.loads((result_root/f'back-{aug}/view-hashes.json').read_text())
        assert a==b,'Pixel trajectories differ between layouts.'
    guard();PROGRESS['stage']='complete'
    dump(result_root/'completed.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'all_four_cells_completed','protocol_sha256':lock['protocol_sha256'],
        'cells':all_summaries,'wall_seconds':time.perf_counter()-run_start,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'matched_RGB_trajectories_verified':True,'extra_seeds':0,
        'reserved_confirmation_read':False,'MPS_device':'mps:0','fallback':False})
    print('ALL_FOUR_CELLS_COMPLETED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        failure={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'incomplete_not_a_scientific_negative',
            'reason':repr(ex),'progress':PROGRESS,
            'elapsed_seconds':None if RUN_START is None else time.perf_counter()-RUN_START,
            'automatic_retry':False}
        path=ROOT/'results/incomplete.json' if (ROOT/'results').is_dir() else ROOT/'run-failure.json'
        if not path.exists():dump(path,failure)
        raise
