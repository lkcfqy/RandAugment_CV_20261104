"""Identical CPU inference for P0/A0/A1, all fixed identities, then persisted COCO evaluation."""
from runtime_common import *
import argparse,traceback,collections

def payload_path(split,arm,iid): return P/f'evaluation/{split}/predictions/{arm}/{iid}.json'

def load_payload(split,arm,iid):
    x=read(payload_path(split,arm,iid));value=x['payload'];assert digest(value)==x['payload_sha256']
    assert value['split']==split and value['arm']==arm and value['image_id']==iid
    assert value['annotation_sha256']==read(P/'DATA_READY.json')['annotations'][split]
    assert len(value['predictions'])==300
    return value

def infer(split,arm):
    out=P/f'evaluation/{split}';(out/'receipts').mkdir(parents=True,exist_ok=True);(out/f'predictions/{arm}').mkdir(parents=True,exist_ok=True)
    receipt=out/f'receipts/{arm}.json';started=time.monotonic();guard=Guard()
    report={'status':'STARTED','arm':arm,'split':split,'PID':os.getpid(),'completed_forwards':0,'persisted':0,'skipped_persisted':0,'forward_seconds':0.,'pipeline_seconds':0.,'persistence_seconds':0.,'model_effects_evaluated':False,'new_CU':0}
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic(receipt,report)
    try:
        check_code_pins();guard.check()
        for name in ('A0','A1'):
            trained=read(P/f'formal_{name}/RESULT.json');assert trained['status']=='TRAINING_COMPLETE' and trained['completed_updates']==200
            assert sha(P/f'formal_{name}/final_model.pth')==trained['checkpoint']['sha256']
        if split=='confirm':
            choice=read(P/'DEV_DECISIONS.json');assert choice['status']=='FROZEN_BEFORE_CONFIRMATION';report['dev_decisions_sha256']=sha(P/'DEV_DECISIONS.json')
            assert not (P/'evaluation/confirm/METRICS.json').exists(),'Confirm already evaluated'
        report['environment']=setup();reset_seed()
        import torch,numpy as np,cv2
        from mmengine.dataset import Compose,pseudo_collate
        from mmdet.datasets.transforms import Color
        from mmdet.evaluation.metrics.coco_metric import CocoMetric
        torch.set_grad_enabled(False)
        ck=None if arm=='P0' else P/f'formal_{arm}/final_model.pth'
        model,cfg,load=build_model(ck);model.eval();report['load']=load
        if ck is not None:assert load['load_state']==read(P/f'formal_{arm}/CHECKPOINT_RECEIPT.json')['load']
        count={'text':0,'dn':0};captions=[]
        def text_hook(mod,args):
            assert list(args[0])==['fish. '],repr(args[0]);count['text']+=1
            if not captions:captions.append(list(args[0]))
        def dn_hook(mod,args):count['dn']+=1;raise AssertionError('Training-only DN used in evaluation')
        model.language_model.register_forward_pre_hook(text_hook)
        model.dn_query_generator.register_forward_pre_hook(dn_hook);model.dn_query_generator.label_embedding.register_forward_pre_hook(dn_hook)
        ds=dataset(split);images=read(P/f'data/annotations/{split}.json')['images'];assert len(ds)==len(images) and ds.cat_ids==[1]
        assert [ds.get_data_info(i)['img_id'] for i in range(len(ds))]==[x['id'] for x in images]
        steps=cfg['test_dataloader']['dataset']['pipeline'];assert [x['type'] for x in steps]==['LoadImageFromFile','FixScaleResize','LoadAnnotations','PackDetInputs']
        assert steps[1]['scale']==(800,1333) and steps[1]['keep_ratio']
        loader=Compose([steps[0]]);rest=Compose(steps[1:]);color=Color(prob=1.,level=0,min_mag=1.,max_mag=1.)
        pins=read(P/'IMAGE_MANIFEST.json');report['status']='INFERRING';save()
        for index in range(len(ds)):
            guard.check(extra=2**20);info=ds.get_data_info(index);iid=info['img_id'];pin=pins[Path(info['img_path']).name]
            assert sha(info['img_path'])==pin['image_sha256']
            t=time.monotonic();loaded=loader(copy.deepcopy(info));raw=loaded['img'];instances=copy.deepcopy(loaded['instances']);colored=color(loaded)
            assert raw.dtype==np.uint8 and np.array_equal(raw,colored['img']) and colored['instances']==instances
            packed=rest(colored);sample=packed['data_samples'];gt=sample.gt_instances;boxes=gt.bboxes.tensor if hasattr(gt.bboxes,'tensor') else gt.bboxes
            geo={'input_shape':list(packed['inputs'].shape),'GT_boxes':boxes.tolist(),'GT_labels':gt.labels.tolist(),'metadata':sample.metainfo};geo_hash=digest(geo)
            if arm!='P0':assert geo_hash==load_payload(split,'P0',iid)['geometry_sha256']
            if payload_path(split,arm,iid).exists():
                old=load_payload(split,arm,iid);assert old['geometry_sha256']==geo_hash and old['checkpoint_sha256']==load['source_sha256'];report['skipped_persisted']+=1;continue
            batch=pseudo_collate([packed]);pipe_seconds=time.monotonic()-t
            t=time.monotonic()
            with torch.inference_mode():output=model.test_step(batch)
            forward_seconds=time.monotonic()-t;pred=output[0].pred_instances
            assert len(output)==1 and pred.bboxes.shape==(300,4) and pred.scores.shape==(300,) and pred.labels.shape==(300,)
            assert all(z.device.type=='cpu' and bool(torch.isfinite(z).all()) for z in [pred.bboxes,pred.scores,pred.labels])
            bboxes=pred.bboxes.detach().numpy();scores=pred.scores.detach().numpy();labels=pred.labels.detach().numpy();assert np.all(labels==0)
            rows=[{'image_id':iid,'bbox':CocoMetric.xyxy2xywh(None,bboxes[j]),'score':float(scores[j]),'category_id':1} for j in range(300)]
            value={'split':split,'arm':arm,'image_id':iid,'image_sha256':pin['image_sha256'],'annotation_sha256':read(P/'DATA_READY.json')['annotations'][split],'checkpoint_sha256':load['source_sha256'],'state_sha256':load['load_state']['state_sha256'],'config_sha256':read(P/'PROTOCOL.json')['historical_A_config_sha256'],'geometry_sha256':geo_hash,'input_shape':geo['input_shape'],'predictions':rows,'pipeline_seconds':pipe_seconds,'forward_seconds':forward_seconds,'actual_text':'fish. ','CPU_FP32':True}
            if split=='confirm':value['dev_decisions_sha256']=report['dev_decisions_sha256']
            t=time.monotonic();atomic(payload_path(split,arm,iid),{'payload':value,'payload_sha256':digest(value)});persist_seconds=time.monotonic()-t
            report['completed_forwards']+=1;report['persisted']+=1;report['forward_seconds']+=forward_seconds;report['pipeline_seconds']+=pipe_seconds;report['persistence_seconds']+=persist_seconds;report['last_image_id']=iid
            if report['persisted']%16==0:save();print(json.dumps({k:report[k] for k in ['split','arm','persisted','elapsed_seconds']}),flush=True)
            del loaded,raw,colored,packed,sample,gt,boxes,batch,output,pred,bboxes,scores,labels,rows,value
        assert report['persisted']+report['skipped_persisted']==len(ds)
        assert count=={'text':report['completed_forwards'],'dn':0}
        assert all(p.grad is None for p in model.parameters()) and state_hash(model.state_dict())==load['load_state']
        report.update(status='COMPLETE',expected_images=len(ds),model_state_unchanged=True,actual_captions=captions,forward_hook_counts=count,all_gradients_none=True);save()
    except BaseException as e:
        report.update(status='INCOMPLETE',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());save();raise

def evaluate(split):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    check_code_pins();guard=Guard();guard.check();out=P/f'evaluation/{split}';(out/'metrics').mkdir(exist_ok=True)
    path=R/'confirmation-sampled-20261101/diagnostic_contracts.py'
    assert sha(path)==read(P/'PROTOCOL.json')['historical_metric_contract_sha256']
    spec=importlib.util.spec_from_file_location('prior_metric_contract',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    source=P/f'data/annotations/{split}.json';assert sha(source)==read(P/'DATA_READY.json')['annotations'][split]
    ids=[x['id'] for x in read(source)['images']];results={};manifest=[]
    for arm in ('P0','A0','A1'):
        receipt=read(out/f'receipts/{arm}.json');assert receipt['status']=='COMPLETE' and receipt['expected_images']==len(ids)
        files=list((out/f'predictions/{arm}').glob('*.json'));assert {int(f.stem) for f in files}==set(ids) and len(files)==len(ids)
        assert not list((out/f'predictions/{arm}').glob('*.partial'))
        rows=[]
        for iid in ids:
            value=load_payload(split,arm,iid);rows.extend(value['predictions']);manifest.append({'path':str(payload_path(split,arm,iid).relative_to(P)),'sha256':sha(payload_path(split,arm,iid))})
        assert len(rows)==300*len(ids)
        prior=None
        # Re-read all persisted predictions for a second exact recomputation.
        for repeat in (0,1):
            guard.check();allrows=rows if repeat==0 else [q for iid in ids for q in load_payload(split,arm,iid)['predictions']]
            gt=COCO(str(source));dt=gt.loadRes(copy.deepcopy(allrows));ev=COCOeval(gt,dt,'bbox')
            ev.params.catIds=[1];ev.params.imgIds=ids;ev.params.maxDets=[100,300,1000];ev.params.iouThrs=np.linspace(.5,.95,10,endpoint=True)
            ev.evaluate();ev.accumulate();ev.summarize()
            arrays={k:ev.eval[k] for k in ('precision','recall','scores')}
            result={'arm':arm,'split':split,'images':len(ids),'predictions':len(allrows),'stats':[float(v) for v in ev.stats],'metrics':old.mapped_stats(ev.stats),'array_sha256':{k:old.array_digest(v) for k,v in arrays.items()},'maxDets':[100,300,1000],'iouThrs':ev.params.iouThrs.tolist()}
            if repeat==0:
                with (out/f'metrics/{arm}.npz').open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                atomic(out/f'metrics/{arm}.json',result);prior=result
            else:
                assert result==read(out/f'metrics/{arm}.json')
                with np.load(out/f'metrics/{arm}.npz',allow_pickle=False) as saved:
                    assert all(np.array_equal(saved[k],arrays[k]) for k in arrays)
        results[arm]=prior
    atomic(out/'PREDICTION_MANIFEST.json',manifest)
    atomic(out/'METRICS.json',{'status':'ALL_THREE_COMPLETE_RECOMPUTED','results':results,'prediction_manifest_sha256':digest(manifest),'UTC':utc()})
    if split=='dev':
        p0=results['P0']['metrics']['AP']['value'];choices={arm:{'P0_AP':p0,'A_AP':results[arm]['metrics']['AP']['value'],'selected_model':arm if choose(p0,results[arm]['metrics']['AP']['value'])=='A' else 'P0'} for arm in ('A0','A1')}
        assert not list((P/'evaluation/confirm').glob('predictions/*/*.json')) if (P/'evaluation/confirm').exists() else True
        atomic(P/'DEV_DECISIONS.json',{'status':'FROZEN_BEFORE_CONFIRMATION','UTC':utc(),'rule':'strict A dev AP@100>P0, tieP0, each episode independently','choices':choices,'dev_metrics_sha256':sha(out/'METRICS.json')})
    else:
        dev=read(P/'evaluation/dev/METRICS.json')['results'];decisions=read(P/'DEV_DECISIONS.json');reports={}
        for arm in ('A0','A1'):
            row=decision(dev['P0']['metrics']['AP']['value'],dev[arm]['metrics']['AP']['value'],results['P0']['metrics']['AP']['value'],results[arm]['metrics']['AP']['value'])
            assert decisions['choices'][arm]['selected_model']==(arm if row['chosen']=='A' else 'P0')
            reports[arm]=row
        atomic(P/'SCIENTIFIC_RESULT.json',{'status':'COMPLETE_FIXED_PILOT','UTC':utc(),'dev':dev,'confirm':results,'decisions':decisions,'episode_results':reports,'interpretation_limits':['one camera, date-disjoint not site-disjoint','two episodes share3videos; not independent repeats or known fish identities','confirmation120frames38fishGT; correlated frames, uncertain visual completeness','fixed existing ordinary adaptation; no new method/contribution claim'],'new_CU':0})
    print(json.dumps({'split':split,'AP_percent':{a:results[a]['metrics']['AP']['percent'] for a in results},'all12_and_arrays_recomputed':True}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['infer','metrics']);parser.add_argument('split',choices=['dev','confirm']);parser.add_argument('--arm',choices=['P0','A0','A1']);a=parser.parse_args()
    if a.action=='infer':infer(a.split,a.arm)
    else:evaluate(a.split)
