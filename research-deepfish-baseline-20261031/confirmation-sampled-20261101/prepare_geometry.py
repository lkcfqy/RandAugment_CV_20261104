"""Extract only frozen selected assets, then check original data pipeline without a model."""
import copy,json,os,pathlib,random,tarfile,time
from diagnostic_contracts import P,R,PREP,sha,digest,atomic_json,Guard,expected_ids
from formal_cpu import normalize_config

began=time.monotonic();guard=Guard(json.loads((P/'RUN_STARTED.json').read_text()))
selection=json.loads((P/'SELECTION.json').read_text())
targets={x['archive_member']:x for x in selection['images']};seen=set()
with tarfile.open(R/'assets/FISH.tar.gz','r|gz') as archive:
    for member in archive:
        if member.name not in targets:continue
        guard.check();entry=targets[member.name]
        if not member.isfile() or member.size!=int(entry['jpeg_bytes']):raise RuntimeError('Unexpected selected archive member')
        name=entry['file_name']
        if pathlib.Path(name).name!=name:raise RuntimeError('Unsafe selected filename')
        data=archive.extractfile(member).read()
        import hashlib
        if hashlib.sha256(data).hexdigest()!=entry['jpeg_sha256']:raise RuntimeError('Selected JPEG differs')
        output=P/'data/images'/name
        with output.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        seen.add(member.name)
if seen!=set(targets):raise RuntimeError('Selected assets missing')
extraction_seconds=time.monotonic()-began

import numpy as np,cv2,torch,mmcv
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules
from mmdet.datasets.transforms import Color
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False);cv2.setNumThreads(0)
random.seed(20261031);np.random.seed(20261031);torch.random.default_generator.manual_seed(20261031)
register_all_modules(init_default_scope=True);mmcv.use_backend('cv2')
cfgpath=R/'fixed-M-execution-20261101/payload/config_A_reference.json'
cfg=Config(normalize_config(json.loads(cfgpath.read_text())))
ds=copy.deepcopy(cfg['test_dataloader']['dataset']);steps=ds.pop('pipeline')
assert [x['type'] for x in steps]==['LoadImageFromFile','FixScaleResize','LoadAnnotations','PackDetInputs']
assert steps[1]['scale']==(800,1333) and steps[1]['keep_ratio']
ds.update(ann_file=str(P/'data/annotations.json'),data_root=str(P/'data'),pipeline=[])
dataset=DATASETS.build(ds);assert len(dataset)==232 and dataset.cat_ids==[1]
infos=[dataset.get_data_info(i) for i in range(len(dataset))]
assert [x['img_id'] for x in infos]==selection['image_ids']
loader=Compose([steps[0]]);rest=Compose(steps[1:]);color=Color(prob=1.,level=0,min_mag=1.,max_mag=1.)
geoms={}
for info in infos:
    guard.check();loaded=loader(copy.deepcopy(info));raw=loaded['img'];old_instances=copy.deepcopy(loaded['instances'])
    colored=color(loaded)
    assert raw.dtype==np.uint8 and np.array_equal(raw,colored['img']) and colored['instances']==old_instances
    packed=rest(colored);sample=packed['data_samples'];gt=sample.gt_instances
    boxes=gt.bboxes.tensor if hasattr(gt.bboxes,'tensor') else gt.bboxes
    geometry={'input_shape':list(packed['inputs'].shape),'GT_boxes':boxes.tolist(),'GT_labels':gt.labels.tolist(),'metadata':sample.metainfo}
    geoms[str(info['img_id'])]={'image_sha256':sha(info['img_path']),'geometry_sha256':digest(geometry),'input_shape':geometry['input_shape'],'GT_count':len(boxes)}
atomic_json(P/'GEOMETRY_REFERENCE.json',{'annotation_sha256':sha(P/'data/annotations.json'),'images':geoms,'model_calls':0},new=True)
run=json.loads((R/'unadapted-reference-20261101/RUN_PROTOCOL.json').read_text())
models=[]
for x in json.loads((P/'MODEL_IDENTITIES.json').read_text())['models']:
    arm=x['label'];spec={'arm':arm,'seed':20261031,'checkpoint':str(R/x['checkpoint']),'bytes':x['bytes'],'sha256':x['file_sha256'],'state_sha256':x['complete_model_state_sha256'],'config':str(cfgpath),'config_sha256':sha(cfgpath)}
    if arm!='P0':spec['state_receipt']=str(R/('cpu-paired-AM-20261101/formal_A/CHECKPOINT_RECEIPT.json' if arm=='A0' else 'same-domain-decision-20261101/formal_A1/CHECKPOINT_RECEIPT.json'))
    models.append(spec)
annotation={'path':str(P/'data/annotations.json'),'sha256':sha(P/'data/annotations.json'),'images':232,'GT':576}
run.update(authorized_CPU_sampled_confirmation=True,models=models,image_order=selection['image_ids'],image_data_root=str(P/'data'),dev_annotation=annotation,geometry_reference_sha256=sha(P/'GEOMETRY_REFERENCE.json'))
run.pop('authorized_CPU_unadapted_reference',None)
image_lookup={int(x['image_id']):x for x in selection['images']}
run['expected_identities']=expected_ids(models,selection['image_ids']);run['identity_map']={}
for spec in models:
    for image_id in selection['image_ids']:
        key=f"20261031_{spec['arm']}_c100_{image_id}"
        run['identity_map'][key]={'seed':20261031,'arm':spec['arm'],'condition':'c100','factor':1.,'image_id':image_id,'image_sha256':image_lookup[image_id]['jpeg_sha256'],'checkpoint_sha256':spec['sha256'],'config_sha256':spec['config_sha256']}
proto=json.loads((P/'PROTOCOL.json').read_text());proto['dev_annotation']=annotation
atomic_json(P/'PROTOCOL.json',proto)
run['code_sha256']={} # Final pins filled after no-model tests; execution rejects an absent freeze marker.
atomic_json(P/'RUN_PROTOCOL.json',run,new=True)
atomic_json(P/'PREPARATION_RECEIPT.json',{'status':'COMPLETE_NO_MODEL','selected_images_extracted':len(seen),'selected_JPEG_bytes':sum(int(x['jpeg_bytes']) for x in selection['images']),'geometry_images':len(geoms),'GT':sum(x['GT_count'] for x in geoms.values()),'extraction_seconds':extraction_seconds,'total_seconds':time.monotonic()-began,'models_loaded':0,'training':0,'official_test_extracted':False,'remaining_confirm_pool_extracted':False,'new_CU':0},new=True)
print(json.dumps({'status':'PREPARED_NO_MODEL','images':len(geoms),'GT':sum(x['GT_count'] for x in geoms.values()),'seconds':time.monotonic()-began}),flush=True)
