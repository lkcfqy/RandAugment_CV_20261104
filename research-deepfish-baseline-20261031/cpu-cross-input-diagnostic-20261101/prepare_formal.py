"""Freeze the authorized formal matrix without importing or invoking a model."""
import base64
import csv
import datetime
import importlib.metadata as md
import json
from diagnostic_contracts import P,R,PREP,PROTOCOL_SHA,FACTORS,METRIC_MAPPING,sha,atomic_json,expected_ids,tree_bytes

assert not (P/'RUN_PROTOCOL.json').exists()
for name in ['receipts','logs','predictions','metrics','tmp']:(P/name).mkdir(exist_ok=True)
assert sha(PREP/'PROTOCOL.json')==PROTOCOL_SHA
assert sha(PREP/'DELIVERABLE_MANIFEST.json')=='a7667c6f8bd5d59d3f592627f12250a63361b22fe972ed3ff1c36dabf23ede11'
original=json.loads((PREP/'PROTOCOL.json').read_text())
annotation=original['dev_annotation'];assert sha(annotation['path'])==annotation['sha256']
dev=json.loads(open(annotation['path']).read());image_order=[x['id'] for x in dev['images']]
assert len(image_order)==len(set(image_order))==64 and len(dev['annotations'])==187
assert len(dev['categories'])==1 and dev['categories'][0]['id']==1 and dev['categories'][0]['name']=='fish'
images={x['image_id']:x for x in original['images'] if x['split']=='dev'}
assert set(images)==set(image_order)
for image in images.values():assert sha(image['path'])==image['sha256']
environment=json.loads((PREP/'ENVIRONMENT_LOCK.json').read_text())
for name,item in environment['packages'].items():assert md.version(name)==item['version'],name
for spec in original['models']:
    assert sha(spec['config'])==spec['config_sha256']
    # Full checkpoint hashes are verified immediately before each safe model load.
    from pathlib import Path
    assert Path(spec['checkpoint']).stat().st_size==spec['bytes']

ninja=md.distribution('ninja');wheel=Path(ninja.locate_file('ninja-1.11.1.1.dist-info/WHEEL'))
raw=wheel.read_bytes();raw_sha=sha(wheel)
record={row[0]:row[1] for row in csv.reader(ninja.read_text('RECORD').splitlines())}
assert record['ninja-1.11.1.1.dist-info/WHEEL']=='sha256='+base64.urlsafe_b64encode(bytes.fromhex(raw_sha)).decode().rstrip('=')
old=json.loads((PREP/'receipts/WHEEL_METADATA_DIAGNOSIS.json').read_text())['packages']['ninja']
import hashlib
normalized=hashlib.sha256(wheel.read_text().encode()).hexdigest()
assert normalized==old['WHEEL_sha256'] and raw_sha!=normalized
atomic_json(P/'DOCUMENTATION_CORRECTIONS.json',{
    'pip_check':'The two recorded metadata-only warnings remain; no reinstallation or package metadata modification.',
    'Ninja_WHEEL_hash':{'old_ambiguous_field':'WHEEL_sha256','old_value_means':'UTF8 text hash after universal-newline normalization',
       'normalized_text_sha256':normalized,'raw_file_sha256':raw_sha,'raw_bytes':len(raw),'CRLF_count':raw.count(b'\r\n'),
       'raw_file_matches_installed_RECORD':True},
    'metric_mapping_correction':{'AP':'maxDets100','AP50_AP75_scaleAP':'maxDets1000',
        'AR100':'stats6 maxDets100','AR300':'stats7 maxDets300','AR1000':'stats8 maxDets1000',
        'scaleAR':'stats9-11 maxDets1000','all_AR1000_wording_retracted':True},
    'historical_reports_preserved_unchanged':True},new=True)

ids=expected_ids(original['models'],image_order);assert len(ids)==512
mapping={}
for spec in original['models']:
    for condition,factor in FACTORS:
        for image_id in image_order:
            from diagnostic_contracts import identity
            key=identity(spec['seed'],spec['arm'],condition,image_id)
            mapping[key]={'seed':spec['seed'],'arm':spec['arm'],'condition':condition,'factor':factor,
                          'image_id':image_id,'image_sha256':images[image_id]['sha256'],
                          'checkpoint_sha256':spec['sha256'],'config_sha256':spec['config_sha256']}
implementation=json.loads((R/'repeat-seed20261101-budget25-20261101/payload/implementation_pins.json').read_text())
site=PREP/'.venv/lib/python3.11/site-packages'
official_pins=dict(implementation['file_sha256'])
for rel in ['mmdet/models/layers/transformer/grounding_dino_layers.py','mmdet/evaluation/metrics/coco_metric.py',
            'mmcv/ops/multi_scale_deform_attn.py','mmcv/image/photometric.py','pycocotools/cocoeval.py','pycocotools/coco.py']:
    official_pins[rel]=sha(site/rel)
for rel,h in implementation['file_sha256'].items():assert sha(site/rel)==h
code_files=['diagnostic_contracts.py','formal_cpu.py','test_contracts.py','prepare_formal.py']
run={'status':'FROZEN_AUTHORIZED_BEFORE_FORMAL_CPU_EFFECTS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'authorized_full512':True,'authorization':'Trusted parent delegation: complete all512 localCPU after no-model tests; no additionalGO needed.',
     'original_protocol_sha256':PROTOCOL_SHA,'prepared_plan_sha256':sha(PREP/'FULL_DIAGNOSTIC_PLAN.json'),
     'question':original['question'],'interpretation_limit':original['interpretation_limit'],
     'models':original['models'],'dev_annotation':annotation,'image_order':image_order,'image_pins':images,
     'expected_identities':ids,'identity_map':mapping,'GT_boxes':187,
     'GT4195_4660_unchanged':[a for a in dev['annotations'] if a['id'] in (4195,4660)],
     'environment_lock_sha256':sha(PREP/'ENVIRONMENT_LOCK.json'),
     'core_runtime_versions':{'torch':'2.1.0','torchvision':'0.16.0','mmcv':'2.1.0','mmengine':'0.10.7','mmdet':'3.3.0','numpy':'1.24.3','cv2':'4.11.0'},
     'official_implementation_sha256':official_pins,'code_sha256':{name:sha(P/name) for name in code_files},
     'runtime':{'device':'cpu','torch_threads':4,'interop_threads':1,'cv2_setNumThreads':0,'required_cv2_getNumThreads':1,
                'dtype':'float32','model_eval':True,'inference_mode':True,'batch_size':1,
                'OMP_MKL_OPENBLAS_VECLIB_NUMEXPR_threads':4,'HF_offline':True,'checkpoint_weights_only':True},
     'pipeline':'cv2 uint8 BGR decode -> official fixed Color(prob1,level0,min=max1/.8) -> original FixScaleResize -> original LoadAnnotations -> PackDetInputs',
     'metric_mapping':[{'name':name,'stats_index':idx,'maxDets':mdet,'area':area} for name,idx,mdet,area in METRIC_MAPPING],
     'COCO_maxDets':[100,300,1000],'COCO_iouThrs':{'start':.5,'end':.95,'count':10},
     'primary_interaction':'I = [AP_B-AP_A]_input0.8 - [AP_B-AP_A]_input1.0, perseed, percentage points; APmaxDets100',
     'save_all_top300':True,'metric_gate':'All512 unique persisted identities and all8 conditions complete before any metrics',
     'recompute':'Separate fresh CPU process rereads persisted predictions and requires exact stats/full COCO arrays',
     'wall_limit_seconds':5400,'new_artifact_cap_bytes':100*1024**2,'research_cap_bytes':6*1024**3,
     'retain_research_margin_bytes':80*1024**2,'time_estimate_only_seconds':2887.5877703294245,
     'official_test':False,'training':False,'third_seed':False,'warmup_or_preflight_repeats':0,
     'checkpoint_or_full_image_tensor_copies':False,'GPU_MPS_Colab_upload_CU_new':0,
     'incomplete_policy':'Missing/duplicate/failed/budget-interrupted matrix is incomplete; never select a completed subset.',
     'preparation_research_bytes':tree_bytes(R)}
assert run['preparation_research_bytes']+100*1024**2 <6*1024**3-80*1024**2
atomic_json(P/'RUN_PROTOCOL.json',run,new=True)
print(json.dumps({'status':'FORMAL_PROTOCOL_FROZEN_NO_MODEL','identities':512,'protocol_sha256':sha(P/'RUN_PROTOCOL.json'),
                  'new_CU':0,'uploads':0}))
