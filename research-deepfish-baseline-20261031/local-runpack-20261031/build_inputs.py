"""Local read/copy/static generation only. Does not import any upstream module."""
import ast,copy,hashlib,json,pathlib,sys,tarfile
ROOT=pathlib.Path(__file__).resolve().parent;P=ROOT/'payload';ASSET=pathlib.Path(sys.argv[1]);FOLLOW=ASSET/'followup-group-provenance-20261031'
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
evidence=json.loads((FOLLOW/'sources/mmdetection_official_evidence.json').read_text());text=evidence['files']['config']['content'];env={}
def literal(n):
 if isinstance(n,ast.Constant):return n.value
 if isinstance(n,(ast.List,ast.Tuple)):return [literal(x) for x in n.elts]
 if isinstance(n,ast.Dict):return {literal(k):literal(v) for k,v in zip(n.keys,n.values)}
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and not n.args:return {k.arg:literal(k.value) for k in n.keywords}
 if isinstance(n,ast.Name):return env[n.id]
 if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='_base_' and n.attr=='backend_args':return None
 raise ValueError('Not a static literal: '+ast.dump(n))
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
  try:env[n.targets[0].id]=literal(n.value)
  except (ValueError,KeyError):pass
model=env['model'];model['bbox_head']['num_classes']=1;model['language_model']['name']='assets/bert-base-uncased'
for arm,mag in [('A',1.0),('B',.8)]:
 train=copy.deepcopy(env['train_pipeline']);train[0].update(color_type='color',imdecode_backend='cv2');train.insert(2,dict(type='Color',prob=1.0,level=0,min_mag=mag,max_mag=mag))
 evaluate=copy.deepcopy(env['test_pipeline']);evaluate[0].update(color_type='color',imdecode_backend='cv2')
 ds=dict(type='CocoDataset',data_root='data/',ann_file='annotations/support5.json',data_prefix={'img':'images/'},metainfo={'classes':['fish']},filter_cfg={'filter_empty_gt':False,'min_size':0},return_classes=True,pipeline=[dict(type='AuditedPipeline',transforms=train)],backend_args=None)
 eds=dict(type='CocoDataset',data_root='data/',ann_file='annotations/dev64.json',data_prefix={'img':'images/'},metainfo={'classes':['fish']},test_mode=True,return_classes=True,pipeline=evaluate,backend_args=None)
 cfg=dict(default_scope='mmdet',model=model,load_from='assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth',resume=False,work_dir='outputs/arm_'+arm,randomness=dict(seed=20261031,diff_rank_seed=False,deterministic=True),env_cfg=dict(cudnn_benchmark=False,mp_cfg=dict(mp_start_method='fork',opencv_num_threads=0),dist_cfg=dict(backend='nccl')),train_dataloader=dict(batch_size=1,num_workers=0,persistent_workers=False,sampler=dict(type='DefaultSampler',shuffle=True,round_up=False),batch_sampler=dict(type='AspectRatioBatchSampler'),dataset=ds),train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=0,val_interval=99999),val_dataloader=None,val_cfg=None,val_evaluator=None,test_dataloader=dict(batch_size=1,num_workers=0,persistent_workers=False,sampler=dict(type='DefaultSampler',shuffle=False,round_up=False),dataset=eds),test_cfg={'type':'TestLoop'},test_evaluator=dict(type='CocoMetric',ann_file='data/annotations/dev64.json',metric='bbox',format_only=False,prefix='dev'),optim_wrapper=env['optim_wrapper'],param_scheduler=[],auto_scale_lr=dict(enable=False,base_batch_size=1),default_hooks=dict(timer=dict(type='IterTimerHook'),logger=dict(type='LoggerHook',interval=10),sampler_seed=dict(type='DistSamplerSeedHook'),checkpoint=None),log_processor=dict(type='LogProcessor',window_size=10,by_epoch=True),log_level='INFO',launcher='none')
 cfg['optim_wrapper'].pop('_delete_',None);write(P/('config_'+arm+'.json'),cfg)
dev=json.loads((FOLLOW/'reports/DEV_SELECTION.json').read_text());ANN=ASSET/'assets/annotation_docs/FISH/annotations';train=json.loads((ANN/'train.json').read_text());ids={i['image_id'] for i in dev['images']}
subset={k:([i for i in v if i['id'] in ids] if k=='images' else [a for a in v if a['image_id'] in ids] if k=='annotations' else v) for k,v in train.items()}
(P/'data/annotations').mkdir(parents=True,exist_ok=True);(P/'data/annotations/support5.json').write_bytes((ANN/'5_shot.json').read_bytes());(P/'data/annotations/dev64.json').write_text(json.dumps(subset,sort_keys=True,separators=(',',':'))+'\n')
chosen={r['archive_member']:r for r in dev['support']+dev['images']};done=set();(P/'data/images').mkdir(parents=True,exist_ok=True)
with tarfile.open(ASSET/'assets/FISH.tar.gz','r|gz') as tf:
 for m in tf:
  if m.name not in chosen:continue
  r=chosen[m.name];assert m.isfile() and m.name.startswith('FISH/train/') and pathlib.PurePosixPath(r['file_name']).name==r['file_name']
  data=tf.extractfile(m).read();assert len(data)==r['image_bytes'] and hashlib.sha256(data).hexdigest()==r['image_sha256'];(P/'data/images'/r['file_name']).write_bytes(data);done.add(m.name)
  if len(done)==69:break
assert done==set(chosen)
write(P/'data_selection.json',{'selection_version':dev['selection_version'],'seed':20261031,'support_ids':[r['image_id'] for r in dev['support']],'dev_ids':[r['image_id'] for r in dev['images']],'support_boxes':5,'extra_dev_images':64,'extra_dev_boxes':187,'prefix_disjoint':True,'shared_habitat_codes':['7463'],'source_train_sha256':dev['source_train_sha256'],'source_5shot_sha256':dev['source_5_shot_sha256'],'images':[dict(file=r['file_name'],sha256=r['image_sha256'],bytes=r['image_bytes']) for r in dev['support']+dev['images']]})
bert=json.loads((ASSET/'reports/EXISTING_BERT_ASSETS.json').read_text());assets=[dict(path='assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth',url='https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth',bytes=691901857,sha256='822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3')]
for r in bert['files']:
 name=pathlib.Path(r['path']).name;assets.append(dict(path='assets/bert-base-uncased/'+name,url='https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/'+name,bytes=r['size'],sha256=r['sha256']))
write(P/'asset_pins.json',{'future_official_downloads_only':True,'assets':assets})
pins={v['path']:hashlib.sha256(v['content'].encode()).hexdigest() for v in evidence['files'].values() if v['path'].startswith('mmdet/')};color=json.loads((FOLLOW/'sources/color_transform_evidence.json').read_text());pins['mmdet/datasets/transforms/colorspace.py']=hashlib.sha256(color['mmdetection']['content'].encode()).hexdigest();write(P/'implementation_pins.json',{'mmdet_version':'3.3.0','upstream_commit':'44ebd17b145c2372c4b700bfb9cb20dbd28ab64a','file_sha256':pins})
# Isolated pure-transform fixture: real official methods, no mmdet/mmcv imports or models.
# The identity decorator represents only the uncached standalone transform path;
# the actual MMCV-decorated Color is tested again in the future runtime gate.
fixture='''# Copyright (c) OpenMMLab. All rights reserved. Apache-2.0.
# Test-only pure method extraction; not part of the upload payload.
import cv2, numpy as np
from typing import Optional
from types import SimpleNamespace
_MAX_LEVEL=10
imread_backend='cv2'
cache_randomness=lambda f:f
TRANSFORMS=SimpleNamespace(register_module=lambda:lambda cls:cls)
class BaseTransform:
 def __call__(self,results):return self.transform(results)
def bgr2gray(img):return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''
src=(ASSET/'sources/Domain-RAG/mmdetection/mmdet/datasets/transforms/augment_wrappers.py').read_text()
for source,names in [(src,{'level_to_mag'}),(color['mmcv']['content'],{'adjust_color'}),(color['mmdetection']['content'],{'ColorTransform','Color'})]:
 for n in ast.parse(source).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:fixture+='\n'+ast.get_source_segment(source,n)+'\n'
 fixture+='\nmmcv=SimpleNamespace(adjust_color=adjust_color)\n' if names=={'adjust_color'} else ''
(ROOT/'test_fixtures/pure_color.py').write_text(fixture)
print(json.dumps({'images_copied':len(done),'dev_images':len(subset['images']),'dev_boxes':len(subset['annotations']),'configs':'two complete JSON dictionaries; no model execution'}))
