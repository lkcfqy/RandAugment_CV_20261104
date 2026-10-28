"""Eight controlled reset steps. Never read development/test arrays or fit a trajectory."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,time,gc,resource,traceback
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parent
PILOT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/pilot-20261022')
sys.path.insert(0,str(PILOT))
import common as C

START=None
CAP=600
ACTIVE=0.
MAX_DRIVER=0
PHASE={'stage':'before_start','layout':None,'gradient_batch':None}

def sha(p):return C.sha(p)
def save(p,v):C.dump(p,v)
def guard():
    if (ROOT/'STOP.request').exists():raise RuntimeError('Resource/user stop requested; retain partial diagnostic.')
    if START is not None and time.perf_counter()-START>CAP:raise RuntimeError('Predeclared diagnostic wall budget exhausted.')

def measured(fn):
    global ACTIVE,MAX_DRIVER
    guard();torch.mps.synchronize();t=time.perf_counter()
    value=fn();torch.mps.synchronize();dt=time.perf_counter()-t
    ACTIVE+=dt;MAX_DRIVER=max(MAX_DRIVER,torch.mps.driver_allocated_memory())
    time.sleep(2*dt);guard();return value

def features(m,x):
    x=m._process_input(x)
    x=torch.cat((m.class_token.expand(x.shape[0],-1,-1),x),dim=1)
    return m.encoder(x)[:,0]

def probe(m,batches):
    m.eval();out=[]
    with torch.no_grad():
        for x in batches:out.append(features(m,x.to('mps')).detach().cpu().numpy())
    return np.stack(out)

def trainable(m):return {n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
def state_hash(state,include_names=True):
    h=hashlib.sha256()
    for n,p in state.items():
        if include_names:h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def frozen_hash(m):
    h=hashlib.sha256()
    for n,p in m.named_parameters():
        if p.requires_grad:continue
        h.update(n.encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=np.float64)))

def main():
    global START,CAP
    cfg=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'protocol-lock.json').read_text())
    assert not (ROOT/'results').exists() and not (ROOT/'analysis.json').exists(),'No overwrite or repeated diagnostic'
    assert sha(ROOT/'protocol.json')==lock['protocol_sha256']
    for n,h in lock['code_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(PILOT/'common.py')==lock['original_common_sha256']
    assert sha(PILOT/'protocol.json')==cfg['original_pilot_protocol_sha256']
    assert sha(PILOT/'fit-manifest.json')==cfg['fit_manifest_sha256']
    assert sha(PILOT/'data/vit_b_16-c867db91.pth')==cfg['weight_sha256']
    assert sha(PILOT/'delivery-manifest.json')==cfg['protected_original_pilot_delivery_manifest_sha256']
    START=time.perf_counter();CAP=cfg['wall_cap_seconds'];guard()
    C.setup_mps()
    out=ROOT/'results';out.mkdir(exist_ok=False)
    save(out/'started.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'device':'mps:0',
        'MPS_available':torch.backends.mps.is_available(),'MPS_built':torch.backends.mps.is_built(),'CUDA_available':torch.cuda.is_available(),
        'MPS_fallback':False,'CPU_threads':torch.get_num_threads(),'actual_process_nice':os.getpriority(os.PRIO_PROCESS,0),
        'protocol_sha256':lock['protocol_sha256'],'wall_cap_seconds':CAP,
        'versions':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version},
        'global_GPU_idleness_or_current8A_process_inspection_claimed':False})
    manifest=json.loads((PILOT/'fit-manifest.json').read_text())
    allowed={r['source_id']:r for r in manifest['rows'] if r['candidate_partition']=='train'}
    expected=sorted(allowed,key=lambda sid:hashlib.sha256(('opt-diagnostic-v1|20261022|'+sid).encode()).hexdigest())[:96]
    assert [r['source_id'] for r in cfg['selected_rows']]==expected
    assert len({r['duplicate_cluster_id'] for r in cfg['selected_rows']})==96
    for role,count in [('gradient',4),('probe',2)]:
        assert sum(r['role']==role for r in cfg['selected_rows'])==count*16
        assert all(sum(r['role']==role and r['batch']==b for r in cfg['selected_rows'])==16 for b in range(count))
    grad=[[] for _ in range(4)];probes=[[] for _ in range(2)];receipts=[]
    for row in cfg['selected_rows']:
        guard();sid=row['source_id'];assert sid in allowed and row['candidate_partition']=='train'
        assert all(allowed[sid][k]==row[k] for k in ['source_id','class_id','candidate_partition','image_sha256','decoded_RGB_sha256','duplicate_cluster_id'])
        p=PILOT/'data/images'/(sid+'.jpg');assert sha(p)==row['image_sha256']
        with Image.open(p) as raw:im=raw.convert('RGB').copy()
        assert hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()==row['decoded_RGB_sha256']
        x=C.tensor(C.weak_pil(im,sid,0,cfg['model_seed'])) if row['role']=='gradient' else C.eval_tensor(im)
        h=hashlib.sha256(x.numpy().tobytes()).hexdigest()
        (grad if row['role']=='gradient' else probes)[row['batch']].append((x,row['class_id']))
        receipts.append({'source_id':sid,'role':row['role'],'batch':row['batch'],'tensor_sha256':h,'image_sha256':row['image_sha256']})
    save(out/'input-receipt.json',{'training_only':True,'photos':96,'gradient_photos':64,'probe_photos':32,'rows':receipts,
        'development_test_or_reserved_pixels_logits_or_accuracy_read':False})
    grad_x=[torch.stack([x for x,y in b]) for b in grad]
    grad_y=[torch.tensor([y for x,y in b],dtype=torch.int64) for b in grad]
    probe_x=[torch.stack([x for x,y in b]) for b in probes]
    cases=[];reference_initial_features=None;reference_slot_hash=None;paired_head_grads={}
    for layout in cfg['layouts']:
        PHASE.update(stage='model_build',layout=layout,gradient_batch=None)
        guard();folder=out/layout;folder.mkdir()
        m=C.build_model(layout,cfg['model_seed']).to('mps')
        initial=trainable(m);assert sum(t.numel() for t in initial.values())==65317
        original=torch.load(PILOT/'results'/('front-weak' if layout=='front' else 'back-weak')/'initial-trainable-checkpoint.pt',map_location='cpu',weights_only=True)['trainable_state']
        assert set(initial)==set(original) and all(torch.equal(initial[n],original[n]) for n in initial)
        slot_hash=state_hash(initial,False)
        if reference_slot_hash is None:reference_slot_hash=slot_hash
        assert slot_hash==reference_slot_hash
        torch.save({'trainable_state':initial,'layout':layout,'protocol_sha256':lock['protocol_sha256']},folder/'initial-trainable.pt')
        before_frozen=frozen_hash(m)
        f0=measured(lambda:probe(m,probe_x));f0_repeat=measured(lambda:probe(m,probe_x))
        assert np.isfinite(f0).all()
        if reference_initial_features is None:reference_initial_features=f0.copy()
        assert np.allclose(f0,reference_initial_features,**cfg['precision']['initial_front_back_allclose'])
        np.savez_compressed(folder/'initial-features.npz',features=f0,repeat_features=f0_repeat)
        for batch in range(4):
            PHASE.update(stage='case_reset',layout=layout,gradient_batch=batch)
            guard();case=f'{layout}-batch{batch}';d=folder/f'batch{batch}';d.mkdir()
            with torch.no_grad():
                for n,p in m.named_parameters():
                    if p.requires_grad:p.copy_(initial[n].to('mps'))
            assert state_hash(trainable(m))==state_hash(initial)
            opt=C.optimizer(m);C.set_lr(opt,0,1110)
            lrs={n:opt.param_groups[1 if n.startswith('heads.') else 0]['lr'] for n in initial}
            assert opt.param_groups[0]['lr']==cfg['optimizer']['adapter_actual_lr'] and opt.param_groups[1]['lr']==cfg['optimizer']['head_actual_lr']
            def step():
                m.train();opt.zero_grad(set_to_none=True)
                PHASE['stage']='gradient_forward'
                loss=torch.nn.functional.cross_entropy(m(grad_x[batch].to('mps')),grad_y[batch].to('mps'))
                save(d/'pre-step-loss.json',{'finite':bool(torch.isfinite(loss).item()),'CE_repr':repr(float(loss.detach().cpu()))})
                assert torch.isfinite(loss);PHASE['stage']='backward';loss.backward()
                gradients={n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in m.named_parameters() if p.requires_grad}
                torch.save({'gradients':gradients,'case':case,'protocol_sha256':lock['protocol_sha256']},d/'gradients.pt')
                PHASE['stage']='raw_gradient_saved_structure_check'
                assert set(gradients)==set(initial) and all(g is not None and torch.isfinite(g).all() for g in gradients.values())
                a=[n for n in gradients if n.endswith(('.aq','.av'))];b=[n for n in gradients if n.endswith(('.bq','.bv'))]
                assert len(a)==6 and len(b)==6
                assert all(torch.count_nonzero(gradients[n])==0 for n in a),'A data-gradient must be zero at B0'
                assert all(torch.count_nonzero(gradients[n])>0 for n in b),'B data-gradient unexpectedly zero'
                PHASE['stage']='single_optimizer_step';opt.step()
                updated=trainable(m)
                torch.save({'trainable_state':updated,'case':case,'protocol_sha256':lock['protocol_sha256']},d/'post-step-trainable.pt')
                PHASE['stage']='raw_post_step_saved'
                return float(loss.detach().cpu()),gradients,updated
            loss,grads,after=measured(step)
            PHASE['stage']='paired_head_and_optimizer_oracle'
            head_grads={n:g.numpy().copy() for n,g in grads.items() if n.startswith('heads.')}
            if layout=='front':paired_head_grads[batch]=head_grads
            else:assert all(np.allclose(g,paired_head_grads[batch][n],**cfg['precision']['paired_head_gradient_allclose']) for n,g in head_grads.items())
            parameter_records={};oracle_tolerance=cfg['precision']['manual_AdamW_parameter_oracle']
            for n,p0 in initial.items():
                a=p0.numpy().astype(np.float64);g=grads[n].numpy().astype(np.float64);actual=after[n].numpy().astype(np.float64);lr=lrs[n]
                expected=a*(1-lr*.01)-lr*g/(np.abs(g)+1e-8)
                assert np.allclose(expected,actual,atol=oracle_tolerance['atol'],rtol=oracle_tolerance['rtol']),n
                parameter_records[n]={'gradient_norm_CPU_FP64':norm(g),'gradient_nonzero':int(np.count_nonzero(g)),
                    'gradient_absmax':float(np.max(np.abs(g))),'gradient_fraction_abs_le_epsilon':float(np.mean(np.abs(g)<=1e-8)),
                    'parameter_delta_norm_CPU_FP64':norm(actual-a),'AdamW_FP64_oracle_max_abs_error':float(np.max(np.abs(actual-expected))),'actual_lr':lr}
            merged=[]
            for slot,i in enumerate(cfg['blocks'][layout]):
                prefix=f'encoder.layers.encoder_layer_{i}.self_attention.parametrizations.in_proj_weight.0'
                for projection in ['q','v']:
                    a=after[prefix+'.a'+projection].numpy().astype(np.float64);b=after[prefix+'.b'+projection].numpy().astype(np.float64)
                    assert np.count_nonzero(initial[prefix+'.b'+projection].numpy())==0
                    merged.append({'slot':slot,'projection':projection,'CPU_FP64_frobenius':norm(b@a)})
            PHASE['stage']='post_step_probe'
            f1=measured(lambda:probe(m,probe_x));assert np.isfinite(f1).all()
            np.savez_compressed(d/'features.npz',post_step_features=f1)
            record={'case':case,'layout':layout,'gradient_batch':batch,'reset_from_initial_sha256':state_hash(initial),'initial_slot_sha256':slot_hash,
                'optimizer_steps_in_this_fresh_state':1,'training_batch_CE_before_step':loss,'parameter_records':parameter_records,
                'merged_QV_updates':merged,'A_data_gradients_all_exact0':True,'B_data_gradients_all_nonzero':True,
                'manual_first_AdamW_parameter_oracle_passed':True,'probe_feature_shape':list(f1.shape)}
            save(d/'receipt.json',record);cases.append(record)
            print(json.dumps({'completed_case':case,'completed_reset_steps':len(cases),'of':8,'wall_seconds':time.perf_counter()-START}),flush=True)
            del opt,grads,after,f1;gc.collect();guard()
        after_frozen=frozen_hash(m);assert before_frozen==after_frozen
        save(folder/'frozen-base-receipt.json',{'before_sha256':before_frozen,'after_sha256':after_frozen,'unchanged':True,'initial_matches_original_pilot':True,'initial_slot_sha256':slot_hash})
        del m,initial,original;gc.collect();torch.mps.empty_cache();guard()
    save(out/'completed.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'status':'all8_reset_one_step_cases_completed',
        'protocol_sha256':lock['protocol_sha256'],'completed_reset_steps':len(cases),'sequential_steps_per_model':1,
        'training_images_seen_for_gradients':64,'training_only_probe_photos':32,'development_accuracy_or_test_read':False,
        'wall_seconds':time.perf_counter()-START,'wall_cap_seconds':CAP,'MPS_synchronized_host_section_seconds':ACTIVE,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_observed_MPS_driver_bytes':MAX_DRIVER,
        'actual_device':'mps:0','fallback':False,'CPU_threads':torch.get_num_threads(),'new_full_training_or_LR_search':False,
        'original_scientific_outcome_modified':False,'norm_definition':'All NEW reported norms use CPU FP64; old audit files unchanged.'})
    print('DIAGNOSTIC_8_RESET_STEPS_COMPLETED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        path=ROOT/'diagnostic-failure.json'
        if not path.exists():save(path,{'at_utc':datetime.now(timezone.utc).isoformat(),'phase':PHASE,'error':repr(e),'traceback':traceback.format_exc(),'no_automatic_retry_or_parameter_change':True})
        raise
