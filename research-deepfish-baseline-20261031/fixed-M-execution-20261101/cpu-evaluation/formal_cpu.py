"""Frozen512 CPU diagnostic. Inference, persisted evaluation, independent reread.

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

def safe_model(spec,report):
    import torch
    from mmengine.config import Config
    from mmdet.registry import MODELS
    began=time.monotonic()
    if sha(spec['config'])!=spec['config_sha256']:raise ContractError('Config changed')
    cfg=Config(normalize_config(json.loads(pathlib_path(spec['config']).read_text())))
    assets=json.loads((R/'repeat-seed20261101-budget25-20261101/payload/asset_pins.json').read_text())['assets'][1:]
    bert=R/'assets/bert-base-uncased'
    if {f.name for f in bert.iterdir() if f.is_file()}!={pathlib_path(a['path']).name for a in assets}:
        raise ContractError('Unexpected BERT files; safetensors-only assets required')
    for asset in assets:
        f=bert/pathlib_path(asset['path']).name
        if f.stat().st_size!=asset['bytes'] or sha(f)!=asset['sha256']:raise ContractError('BERT asset changed')
    cfg['model']['language_model']['name']=str(bert)
    if pathlib_path(spec['checkpoint']).stat().st_size!=spec['bytes'] or sha(spec['checkpoint'])!=spec['sha256']:
        raise ContractError('Full checkpoint changed')
    model=MODELS.build(cfg['model']);model.init_weights();model.to(torch.device('cpu'))
    checkpoint=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict) or not all(isinstance(v,torch.Tensor) for v in checkpoint.values()):
        raise ContractError('Tensor-only complete state dict required')
    state=model.state_dict()
    if set(state)!=set(checkpoint):raise ContractError('Full checkpoint/model keys differ')
    for key,source in checkpoint.items():
        if source.shape!=state[key].shape or source.dtype!=state[key].dtype or source.device.type!='cpu':
            raise ContractError('Checkpoint tensor schema differs: '+key)
        if source.is_floating_point() and not bool(torch.isfinite(source).all()):
            raise ContractError('Nonfinite checkpoint tensor')
    incompatible=model.load_state_dict(checkpoint,strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:raise ContractError('Non-strict load')
    expected=hashlib.sha256();actual=hashlib.sha256();count_bytes=0
    for key,value in sorted(model.state_dict().items()):
        source=checkpoint[key]
        if value.device.type!='cpu' or value.dtype!=source.dtype or value.shape!=source.shape:
            raise ContractError('Loaded tensor schema differs')
        header=json.dumps([key,str(source.dtype),list(source.shape)],separators=(',',':')).encode()
        a=source.detach().contiguous().numpy().tobytes();b=value.detach().contiguous().numpy().tobytes()
        if a!=b:raise ContractError('Loaded tensor bytes differ: '+key)
        expected.update(header);expected.update(a);actual.update(header);actual.update(b);count_bytes+=len(a)
    if actual.digest()!=expected.digest() or len(checkpoint)!=908:raise ContractError('Complete state hash/schema differs')
    prior=json.loads(pathlib_path(spec['state_receipt']).read_text())
    if actual.hexdigest()!=prior['load']['state_sha256']:raise ContractError('State differs from approved complete checkpoint')
    report['load']={'seconds':time.monotonic()-began,'checkpoint_sha256':spec['sha256'],
                    'state_sha256':actual.hexdigest(),'tensor_count':len(checkpoint),'tensor_bytes':count_bytes,
                    'weights_only':True,'strict':True,'all_tensor_bytes_equal':True,'finite':True,
                    'missing_keys':[],'unexpected_keys':[]}
    del checkpoint,state,source,value,a,b
    model.eval()
    if model.training or any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
        raise ContractError('CPU eval-only model required')
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
    spec=run['models'][0];result,arrays=evaluate_condition(run,spec,'c100')
    label='20261031_M_c100';out=P/'metrics'/(label+'.json');arr=P/'metrics'/(label+'.npz')
    if verify:
        old=json.loads(out.read_text())
        for key in ['stats','metrics','array_sha256','images','predictions','maxDets','iouThrs']:
            if result[key]!=old[key]:raise ContractError('Persisted metric recomputation differs')
        with np.load(arr,allow_pickle=False) as saved:
            for key in arrays:
                if not np.array_equal(saved[key],arrays[key]):raise ContractError('Full metric arrays differ')
        atomic_json(P/'METRIC_RECOMPUTE_VERIFIED.json',dict(status='ALL64_EXACT_FROM_PERSISTED',model_instances=0,prediction_manifest_sha256=matrix['manifest_sha256']),new=True)
    else:
        with arr.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        result['arrays_file_sha256']=sha(arr);atomic_json(out,result,new=True)
        prior=json.loads(pathlib_path(run['reference_A_metrics']).read_text())
        atomic_json(P/'COMPARISON.json',dict(status='COMPLETE_CPU_ONLY',seed=20261031,input_factor=1.0,metric='AP@100',A=prior['metrics']['AP']['percent'],M=result['metrics']['AP']['percent'],M_minus_A_pp=result['metrics']['AP']['percent']-prior['metrics']['AP']['percent'],metrics_A=prior['metrics'],metrics_M=result['metrics'],model_instances_in_metrics=0),new=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['infer','metrics','verify'],required=True);a=p.parse_args()
    if a.phase=='infer':infer(0)
    else:metrics(a.phase=='verify')
if __name__=='__main__':main()
