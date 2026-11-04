"""Isolated preflight or fixed200-update A training; no evaluation or model selection."""
from runtime_common import *
import argparse,traceback,collections
def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','formal']);parser.add_argument('arm',choices=['A0','A1']);args=parser.parse_args()
    out=P/(args.phase+'_'+args.arm);out.mkdir(exist_ok=False);started=time.monotonic();guard=Guard()
    report={'status':'STARTED','phase':args.phase,'arm':args.arm,'PID':os.getpid(),'seed':SEED,'updates_planned':200 if args.phase=='formal' else 1,'completed_updates':0,'dev_confirm_access':False,'new_CU':0}
    def save(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(out/'RESULT.json',report)
        print(json.dumps({k:report[k] for k in ['stage','arm','completed_updates','elapsed_seconds','peak_RSS_bytes']}),flush=True)
    try:
        check_code_pins();guard.check();versions=setup();reset_seed()
        import torch,numpy as np
        from mmengine.optim import build_optim_wrapper
        from mmengine.dataset import pseudo_collate
        proto=read(P/'PROTOCOL.json');model,cfg,load=build_model()
        torch.set_grad_enabled(True);model.train()
        optim=build_optim_wrapper(model,cfg['optim_wrapper']);assert not optim.optimizer.state
        trainable=[(n,t) for n,t in model.named_parameters() if t.requires_grad]
        names=digest([n for n,t in trainable]);groups=validate_optimizer_groups(model,optim.optimizer);group_hash=digest(groups)
        assert names==proto['expected_trainable_names_sha256'] and group_hash==proto['expected_optimizer_groups_sha256']
        assert load['load_state']==proto['expected_initial']
        report.update(environment=versions,load=load,initial=load['load_state'],trainable_names_sha256=names,trainable_tensors=len(trainable),trainable_parameters=sum(t.numel() for n,t in trainable),optimizer_groups_sha256=group_hash,optimizer_initial_state_empty=True)
        atomic(out/'OPTIMIZER_GROUPS.json',groups)
        group='support'+args.arm[-1];ds=dataset(group);assert len(ds)==5
        order=training_order(ds);summary=read(P/'DATA_REFERENCE_SUMMARY.json')[args.arm]
        assert order==summary['dataset_index_order']
        image_pins=read(P/'IMAGE_MANIFEST.json');shot=read(P/f'data/annotations/{group}.json');aids={a['image_id']:a['id'] for a in shot['annotations']}
        for im in shot['images']:assert sha(P/'data/images'/im['file_name'])==image_pins[im['file_name']]['image_sha256']
        o=observer();obs=o.Observer();pipeline=o.make_pipeline('A',obs)
        report['initial_RNG']=o.rng();assert report['initial_RNG']==proto['expected_initial_RNG'],'Initial CPU RNG differs from historical A'
        report['same_initial_state_optimizer_RNG_as_historical_A']=True
        refs=read(P/'DATA_REFERENCE_MANIFEST.json');selected=proto['preflight_iterations'][args.arm] if args.phase=='preflight' else 200
        initial_lr=[(g['lr'],g['weight_decay']) for g in optim.optimizer.param_groups]
        counts={'clip':0,'optimizer_step':0,'zero_grad':0,'schedule_boundary':0}
        for attr,key in [('_clip_grad','clip'),('step','optimizer_step'),('zero_grad','zero_grad')]:
            original=getattr(optim,attr)
            def counted(*a,_original=original,_key=key,**kw):
                if _key=='optimizer_step':
                    grads=[p.grad for n,p in trainable if p.grad is not None]
                    assert len(grads)==896 and all(g.device.type=='cpu' and bool(torch.isfinite(g).all()) for g in grads),'Invalid gradient'
                counts[_key]+=1;return _original(*a,**kw)
            setattr(optim,attr,counted)
        rows=[];steps=[];(out/'trajectory').mkdir();save('MODEL_AND_CONTROLS_VERIFIED')
        for u,index in enumerate(order[:selected],1):
            guard.check();info=ds.get_data_info(index);obs.begin(u,info['img_id'],aids[info['img_id']],(info['height'],info['width']))
            before=o.rng();t=time.monotonic();item=pipeline(copy.deepcopy(info));pipeline_seconds=time.monotonic()-t;sig=o.signature(item)
            refpath=P/f'data_reference_{args.arm}/{u:03}.json';assert sha(refpath)==refs[str(refpath.relative_to(P))]
            ref=read(refpath)
            assert all(sig[k]==ref['signature'][k] for k in ['pixel_sha256','shape_CHW','dtype','gt_sha256','metadata_sha256'])
            assert all(before[k]==ref['RNG_before'][k] for k in ['numpy','python'])
            row={'update':u,'dataset_index':index,'image_id':info['img_id'],'signature':sig,'RNG_before_pipeline':before,'RNG_before_update':o.rng(),'pipeline_seconds':pipeline_seconds,'events':copy.deepcopy(obs.events),'final_GT_lineage':copy.deepcopy(obs.boxes),'updated':False}
            if args.phase=='formal' or u==selected:
                if args.phase=='preflight':assert len(item['data_samples'].gt_instances)>0
                batch=pseudo_collate([item]);t=time.monotonic();losses=model.train_step(batch,optim);duration=time.monotonic()-t
                assert all(bool(torch.isfinite(v).all()) if isinstance(v,torch.Tensor) else np.isfinite(v) for v in losses.values())
                counts['schedule_boundary']+=1;report['completed_updates']+=1
                assert all(float(v['step'])==report['completed_updates'] for v in optim.optimizer.state.values())
                assert all(v==report['completed_updates'] for v in counts.values())
                assert [(g['lr'],g['weight_decay']) for g in optim.optimizer.param_groups]==initial_lr
                row.update(updated=True,train_step_seconds=duration,loss_terms={k:float(v) for k,v in losses.items()},all_loss_gradients_finite=True,RNG_after_update=o.rng())
                steps.append({'update':u,'shape_CHW':sig['shape_CHW'],'GT_count':len(item['data_samples'].gt_instances),'train_step_seconds':duration,'pipeline_seconds':pipeline_seconds,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
                del batch,losses
            atomic(out/'trajectory'/f'{u:03}.json',row);rows.append(row);del item
            if row['updated'] and (args.phase=='preflight' or u%5==0):save('TRAINING')
        report.update(steps=steps,update_call_counts=counts,scheduler_objects=0,source_exposure={'occurrences':len(rows),'per_image':dict(collections.Counter(r['image_id'] for r in rows)),'GT_presentations':sum(len(r['signature']['gt']['gt_instances']['boxes']) for r in rows),'empty_GT_inputs':sum(not r['signature']['gt']['gt_instances']['boxes'] for r in rows),'GT_removed_by_geometry':sum(sum(e['discarded_GT_count'] for e in r['events']) for r in rows)},all_data_reference_matches=True)
        if args.phase=='formal':
            assert report['completed_updates']==200 and all(n==40 for n in report['source_exposure']['per_image'].values())
            guard.check(extra=700*2**20);t=time.monotonic();state=model.state_dict()
            assert all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point())
            state_receipt=state_hash(state);checkpoint=out/'final_model.pth';tmp=checkpoint.with_name(checkpoint.name+'.partial')
            with tmp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,checkpoint)
            reread=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state_hash(reread)==state_receipt and all(torch.equal(state[n],reread[n]) for n in state) and len(reread)==908
            report['checkpoint']={'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':sha(checkpoint),'load':state_receipt,'complete_state_verified':True,'weights_only':True,'save_and_verify_seconds':time.monotonic()-t}
            atomic(out/'CHECKPOINT_RECEIPT.json',report['checkpoint']);report['status']='TRAINING_COMPLETE'
        else:report['status']='PREFLIGHT_COMPLETE'
        save('COMPLETE')
    except BaseException as e:
        report.update(status='INCOMPLETE',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());save('FAILED');raise
if __name__=='__main__':main()
