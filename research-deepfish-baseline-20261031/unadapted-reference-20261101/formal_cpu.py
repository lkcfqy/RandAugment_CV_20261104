"""CPU paired A/M evaluation; original inference core preserved. Inference, persisted evaluation, independent reread.

No training, checkpoint copies, test split, GPU/MPS, uploads, or environment installs.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from diagnostic_contracts import (P,R,PREP,FACTORS,ContractError,Guard,atomic_json,canonical,
    sha,digest,read_protocol,read_execution,identity,persist_prediction,complete_matrix,
    evaluate_condition,tree_bytes)

def normalize_config(value):
    if isinstance(value,list):return [normalize_config(x) for x in value]
    if not isinstance(value,dict):return value
    result={k:normalize_config(v) for k,v in value.items()}
    for key in ('classes','scale','crop_size'):
        if isinstance(result.get(key),list):result[key]=tuple(result[key])
    if 'scales' in result:result['scales']=[tuple(x) if isinstance(x,list) else x for x in result['scales']]
    return result

def validate_pins(run):
    for name,expected in run['code_sha256'].items():
        if sha(P/name)!=expected:raise ContractError('Execution code changed: '+name)
    if sha(PREP/'ENVIRONMENT_LOCK.json')!=run['environment_lock_sha256']:
        raise ContractError('Environment lock changed')
    if sha(run['dev_annotation']['path'])!=run['dev_annotation']['sha256']:
        raise ContractError('Original dev annotation changed')

def initial_state_hash(state):
    h=hashlib.sha256();size=0
    for name,t in sorted(state.items()):
        b=t.detach().contiguous().numpy().tobytes();h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(b);size+=len(b)
    return {'state_sha256':h.hexdigest(),'tensor_count':len(state),'tensor_bytes':size}

def safe_model(spec,report):
    import torch
    from mmengine.config import Config
    from mmdet.registry import MODELS
    began=time.monotonic();proto=json.loads((P/'PROTOCOL.json').read_text())
    for entry in json.loads((P/'INITIALIZATION_SEMANTICS.json').read_text())['static_sources']:
        if sha(pathlib_path(entry['path']))!=entry['sha256']:raise ContractError('Initialization semantics source changed')
    if sha(spec['config'])!=spec['config_sha256']:raise ContractError('Config changed')
    cfg=Config(normalize_config(json.loads(pathlib_path(spec['config']).read_text())))
    pins=json.loads((R/'cpu-paired-AM-20261101/PROTOCOL.json').read_text())['asset_pins']
    for a in pins:
        f=R/a['path']
        if f.stat().st_size!=a['bytes'] or sha(f)!=a['sha256']:raise ContractError('Official existing asset changed')
    cfg['model']['language_model']['name']=str(R/'assets/bert-base-uncased')
    if pathlib_path(spec['checkpoint']).stat().st_size!=spec['bytes'] or sha(spec['checkpoint'])!=spec['sha256']:raise ContractError('Original checkpoint changed')
    torch.set_grad_enabled(False)
    model=MODELS.build(cfg['model']);model.init_weights();model.to(torch.device('cpu'))
    source=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
    if not isinstance(source,dict) or not all(isinstance(v,torch.Tensor) for v in source.values()):raise ContractError('Tensor-only official source required')
    if len(source)!=903 or not all(bool(torch.isfinite(v).all()) for v in source.values() if v.is_floating_point()):raise ContractError('Source schema/finite check failed')
    state=model.state_dict();common=set(state)&set(source);missing=set(state)-set(source);extra=set(source)-set(state)
    logkeys=set(proto['missing_model_definitions']['fixed_identity_log_scales']);dn=proto['missing_model_definitions']['random_training_only_DN']
    if missing!=logkeys|{dn} or extra!=set(proto['extra_source_keys']) or len(common)!=900 or len(state)!=908:raise ContractError('Exact pinned original/init key schema differs')
    for name in common:
        if source[name].shape!=state[name].shape or source[name].dtype!=state[name].dtype:raise ContractError('Source tensor shape/dtype differs: '+name)
    for name in logkeys:
        if tuple(state[name].shape)!=(1,) or state[name].dtype!=torch.float32 or state[name].item()!=0:raise ContractError('Fixed identity scale differs')
    if tuple(state[dn].shape)!=(1,256):raise ContractError('Training-only DN shape differs')
    if model.language_model.language_backbone.body.model.pooler is not None:raise ContractError('Unexpected pooler construction')
    if any(b.bias is not None for b in model.bbox_head.cls_branches):raise ContractError('Unexpected contrastive class bias')
    pos=source['language_model.language_backbone.body.model.embeddings.position_ids']
    if not torch.equal(pos,torch.arange(512).reshape(1,512)):raise ContractError('Legacy position buffer differs')
    # Deliberately explicit completion, never pretend raw903 keys strict-load as908.
    completed={n:source[n] if n in common else state[n] for n in state}
    inc=model.load_state_dict(completed,strict=True)
    if inc.missing_keys or inc.unexpected_keys:raise ContractError('Completed state strict load failed')
    loaded=model.state_dict()
    for n in common:
        if not torch.equal(loaded[n],source[n]):raise ContractError('Source bytes changed: '+n)
    receipt=initial_state_hash(loaded)
    if receipt!=proto['expected_complete_initial_state']:raise ContractError('Initialization differs from A0/A1')
    counts={'dn_query_generator':0,'dn_label_embedding':0,'language_model':0};captions=[]
    def no_dn(which):
        def reject(module,args):
            counts[which]+=1;raise ContractError('Random training-only DN unexpectedly used in inference')
        return reject
    model.dn_query_generator.register_forward_pre_hook(no_dn('dn_query_generator'))
    model.dn_query_generator.label_embedding.register_forward_pre_hook(no_dn('dn_label_embedding'))
    def record_text(module,args):
        counts['language_model']+=1
        value=list(args[0]) if isinstance(args[0],(list,tuple)) else [args[0]]
        if value!=['fish. ']:raise ContractError('Frozen actual caption differs: '+repr(value))
        if value not in captions:captions.append(value)
    model.language_model.register_forward_pre_hook(record_text)
    report['load']={'seconds':time.monotonic()-began,'raw_checkpoint_sha256':spec['sha256'],'raw_checkpoint_keys':903,'direct_raw_strict_load':False,'strict_completed_in_RAM_state_load':True,'900_source_tensors_exact':True,'missing_fixed_scale_keys':sorted(logkeys),'random_training_only_key':dn,'unused_source_keys':sorted(extra),'initial':receipt,'same_initial_state_as_A0_A1':True,'weights_only':True,'all_finite':True,'no_large_checkpoint_copy':True,'passive_forward_counts':counts,'actual_caption_values':captions}
    del source,state,completed,loaded,pos
    model.eval()
    if model.training or any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):raise ContractError('CPU eval-only state required')
    return model,cfg

def pathlib_path(value):
    from pathlib import Path
    return Path(value)

def infer(index):
    began=time.monotonic();run=read_execution();validate_pins(run)
    context=json.loads((P/'RUN_STARTED.json').read_text());guard=Guard(context)
    spec=run['models'][index];label=f"{spec['seed']}_{spec['arm']}"
    receipt=P/'receipts'/('model_'+label+'.json')
    if receipt.exists():raise ContractError('Do not repeat a started model phase')
    report={'status':'STARTED','model':label,'pid':os.getpid(),'forwards_started':0,
            'forwards_completed':0,'predictions_persisted':0,'conditions_complete':[],
            'pipeline_seconds':0.,'forward_seconds':0.,'format_seconds':0.,'persistence_seconds':0.,
            'metrics_seen':False,'official_test_used':False,'new_CU':0,'uploads':0}
    atomic_json(receipt,report,new=True)
    try:
        guard.check()
        if os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')!='1' or os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':
            raise ContractError('Offline safe-load environment absent')
        import random
        import numpy as np
        import cv2
        import torch
        import mmcv,mmdet,mmengine,torchvision
        from mmengine.dataset import Compose,pseudo_collate
        from mmdet.registry import DATASETS
        from mmdet.utils import register_all_modules
        from mmdet.datasets.transforms import Color
        from mmdet.evaluation.metrics.coco_metric import CocoMetric
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
        cv2.setNumThreads(0)
        if torch.version.cuda is not None or torch.get_num_threads()!=4 or torch.get_num_interop_threads()!=1 or cv2.getNumThreads()!=1:
            raise ContractError('Frozen CPU thread/backend settings differ')
        actual_versions={'torch':torch.__version__.split('+')[0],'torchvision':torchvision.__version__.split('+')[0],
                         'mmcv':mmcv.__version__,'mmengine':mmengine.__version__,'mmdet':mmdet.__version__,
                         'numpy':np.__version__,'cv2':cv2.__version__}
        if actual_versions!=run['core_runtime_versions']:raise ContractError('Core runtime differs')
        random.seed(spec['seed']);np.random.seed(spec['seed']);torch.random.default_generator.manual_seed(spec['seed'])
        register_all_modules(init_default_scope=True);mmcv.use_backend('cv2')
        if mmcv.adjust_color.__globals__.get('imread_backend')!='cv2':raise ContractError('BGR/cv2 backend differs')
        site=pathlib_path(mmdet.__file__).parent.parent
        for name,expected in run['official_implementation_sha256'].items():
            if sha(site/name)!=expected:raise ContractError('Official implementation changed: '+name)
        report['environment']={'versions':actual_versions,'device':'cpu','torch_threads':4,
                                'interop_threads':1,'OpenCV_threads':1,'dtype':'float32','autocast':False}
        report['import_setup_seconds']=time.monotonic()-began
        model,cfg=safe_model(spec,report);report['status']='MODEL_VERIFIED';atomic_json(receipt,report)
        guard.check()
        ds_cfg=copy.deepcopy(cfg['test_dataloader']['dataset']);steps=ds_cfg.pop('pipeline')
        if [s['type'] for s in steps]!=['LoadImageFromFile','FixScaleResize','LoadAnnotations','PackDetInputs']:
            raise ContractError('Original evaluation pipeline differs')
        if steps[0]['imdecode_backend']!='cv2' or steps[0]['color_type']!='color' or steps[1]['scale']!=(800,1333) or not steps[1]['keep_ratio']:
            raise ContractError('Original image/resize contract differs')
        ds_cfg.update(ann_file=run['dev_annotation']['path'],data_root=str(R/'repeat-seed20261101-budget25-20261101/payload/data'),pipeline=[])
        dataset=DATASETS.build(ds_cfg)
        if len(dataset)!=64 or dataset.cat_ids!=[1]:raise ContractError('Frozen64dev/category mismatch')
        infos=[dataset.get_data_info(i) for i in range(64)]
        if [x['img_id'] for x in infos]!=run['image_order']:raise ContractError('Frozen dev order differs')
        loader=Compose([steps[0]]);rest=Compose(steps[1:])
        for condition,factor in FACTORS:
            color=Color(prob=1.0,level=0,min_mag=factor,max_mag=factor)
            for info in infos:
                guard.check(extra=1024**2)
                key=identity(spec['seed'],spec['arm'],condition,info['img_id'])
                if (P/'predictions'/(key+'.json')).exists():raise ContractError('Duplicate inference identity')
                pinned=run['identity_map'][key]
                if sha(info['img_path'])!=pinned['image_sha256']:raise ContractError('Image bytes changed')
                p0=time.monotonic();loaded=loader(copy.deepcopy(info));raw=loaded['img']
                if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[2]!=3:raise ContractError('uint8/BGR image required')
                old_instances=copy.deepcopy(loaded['instances']);colored=color(loaded)
                if colored['img'].dtype!=raw.dtype or colored['img'].shape!=raw.shape or colored['instances']!=old_instances:
                    raise ContractError('Color changed geometry/labels')
                if factor==1.0:
                    if not np.array_equal(colored['img'],raw):raise ContractError('A input Color not exact identity')
                else:
                    gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
                    expected=cv2.addWeighted(raw,.8,np.repeat(gray[...,None],3,axis=2),.2,0)
                    if not np.array_equal(colored['img'],expected):raise ContractError('B input differs from fixed cv2 transform')
                    del gray,expected
                packed=rest(colored);sample=packed['data_samples'];gt=sample.gt_instances
                boxes=gt.bboxes.tensor if hasattr(gt.bboxes,'tensor') else gt.bboxes
                geometry={'input_shape':list(packed['inputs'].shape),'GT_boxes':boxes.tolist(),
                          'GT_labels':gt.labels.tolist(),'metadata':sample.metainfo}
                geometry_hash=digest(geometry)
                if geometry_hash!=pinned['reference_A_geometry_sha256']:raise ContractError('Geometry/GT differs from historical CPU A')
                batch=pseudo_collate([packed])
                pipeline_seconds=time.monotonic()-p0
                report['forwards_started']+=1;report['status']='INFERRING';atomic_json(receipt,report)
                f0=time.monotonic()
                with torch.inference_mode():output=model.test_step(batch)
                forward_seconds=time.monotonic()-f0;report['forwards_completed']+=1
                guard.check(extra=1024**2)
                f1=time.monotonic()
                if len(output)!=1:raise ContractError('Batch output count differs')
                pred=output[0].pred_instances
                if pred.bboxes.shape!=(300,4) or pred.scores.shape!=(300,) or pred.labels.shape!=(300,):
                    raise ContractError('Official top300 output shape differs')
                if any(v.device.type!='cpu' or not bool(torch.isfinite(v).all()) for v in [pred.bboxes,pred.scores,pred.labels]):
                    raise ContractError('Nonfinite/nonCPU prediction')
                bboxes=pred.bboxes.detach().numpy();scores=pred.scores.detach().numpy();labels=pred.labels.detach().numpy()
                if np.any(labels!=0):raise ContractError('Unexpected class index')
                # Official3.3 helper does not use self; this performs conversion only, no metrics.
                rows=[{'image_id':info['img_id'],'bbox':CocoMetric.xyxy2xywh(None,bboxes[j]),
                       'score':float(scores[j]),'category_id':dataset.cat_ids[int(labels[j])]} for j in range(300)]
                payload={**pinned,'identity':key,'annotation_sha256':run['dev_annotation']['sha256'],
                         'environment_lock_sha256':run['environment_lock_sha256'],
                         'geometry_sha256':geometry_hash,'input_shape':list(packed['inputs'].shape),
                         'color_uint8_BGR_cv2_exact':True,'predictions':rows,
                         'pipeline_seconds':pipeline_seconds,'forward_seconds':forward_seconds}
                format_seconds=time.monotonic()-f1
                t=time.monotonic();persisted=persist_prediction(payload,run);persist_seconds=time.monotonic()-t
                if persisted['existing_exact']:raise ContractError('Unexpected prior completed identity')
                report['predictions_persisted']+=1
                for name,value in [('pipeline_seconds',pipeline_seconds),('forward_seconds',forward_seconds),('format_seconds',format_seconds),('persistence_seconds',persist_seconds)]:report[name]+=value
                report['last_identity']=key;report['last_record_sha256']=persisted['sha256']
                if report['predictions_persisted']%16==0:
                    report['elapsed_seconds']=time.monotonic()-began;atomic_json(receipt,report)
                    print(json.dumps({'model':label,'persisted':report['predictions_persisted'],'expected':64,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
                del loaded,raw,colored,packed,sample,gt,boxes,batch,output,pred,bboxes,scores,labels,rows,payload
            report['conditions_complete'].append(condition);atomic_json(receipt,report)
        if report['forwards_started']!=64 or report['forwards_completed']!=64 or report['predictions_persisted']!=64:
            raise ContractError('Expected64 completed identities per checkpoint')
        if any(p.grad is not None for p in model.parameters()):raise ContractError('Unexpected gradient')
        if initial_state_hash(model.state_dict())!=report['load']['initial']:raise ContractError('Initial state mutated during evaluation')
        if report['load']['passive_forward_counts']!={'dn_query_generator':0,'dn_label_embedding':0,'language_model':64}:raise ContractError('Unexpected forward usage')
        report['model_state_unchanged_after64']=True
        report['all_gradients_none']=True
        report.update(status='MODEL_COMPLETE',elapsed_seconds=time.monotonic()-began,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      checkpoint_copied=False,model_evaluations=0)
        atomic_json(receipt,report)
    except BaseException as exc:
        report.update(status='INCOMPLETE',reason=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-began,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(receipt,report);raise

def metrics(verify=False):
    import numpy as np
    run=read_execution();validate_pins(run);matrix=complete_matrix(run)
    results={}
    for spec in run['models']:
        result,arrays=evaluate_condition(run,spec,'c100')
        label=f"20261031_{spec['arm']}_c100";out=P/'metrics'/(label+'.json');arr=P/'metrics'/(label+'.npz')
        if verify:
            old=json.loads(out.read_text())
            for key in ['stats','metrics','array_sha256','images','predictions','maxDets','iouThrs']:
                if result[key]!=old[key]:raise ContractError('Persisted metric recomputation differs')
            with np.load(arr,allow_pickle=False) as saved:
                for key in arrays:
                    if not np.array_equal(saved[key],arrays[key]):raise ContractError('Full metric arrays differ')
        else:
            with arr.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            result['arrays_file_sha256']=sha(arr);atomic_json(out,result,new=True)
        results[spec['arm']]=result
    if verify:
        atomic_json(P/'METRIC_RECOMPUTE_VERIFIED.json',dict(status='ALL64_EXACT_FROM_PERSISTED',model_instances=0,prediction_manifest_sha256=matrix['manifest_sha256']),new=True)
    else:
        p=results['P0']['metrics']
        a0=json.loads((R/'cpu-paired-AM-20261101/evaluation/COMPARISON.json').read_text())['metrics_A']
        a1=json.loads((R/'same-domain-decision-20261101/evaluation/COMPARISON.json').read_text())['metrics_A1']
        atomic_json(P/'COMPARISON.json',dict(status='COMPLETE_UNADAPTED_REFERENCE_SAME_CPU',seed=20261031,metric='AP@100',metrics_P0=p,metrics_A0=a0,metrics_A1=a1,A0_minus_P0_pp=a0['AP']['percent']-p['AP']['percent'],A1_minus_P0_pp=a1['AP']['percent']-p['AP']['percent'],model_instances_in_metrics=0),new=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['infer','metrics','verify']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.phase=='infer':infer(a.index)
    else:metrics(a.phase=='verify')
