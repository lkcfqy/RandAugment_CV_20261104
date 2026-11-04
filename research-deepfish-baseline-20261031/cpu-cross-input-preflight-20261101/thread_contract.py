"""No-model check: pin OpenCV to sequential execution with its official API.

GCD ignores positive thread limits. Compare the same support image's prepared
tensors under the measured setting and an explicitly sequential future setting.
No checkpoint is opened and no prediction is made.
"""
import copy
import hashlib
import json
import pathlib
import time

P = pathlib.Path(__file__).resolve().parent
R = P.parent
import cv2
import torch
from mmengine.dataset import Compose
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
register_all_modules(init_default_scope=True)
protocol = json.loads((P/'PROTOCOL.json').read_text())
assert hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest() == '0b4101c0e9e41904c8dc23618439fa2499815eb136ea106bd3c48ef1ed02e305'
assert json.loads((P/'receipts/r1/CPU_PROBE_TERMINAL.json').read_text())['engineering_forward_calls'] == 8
cfg = json.loads(pathlib.Path(protocol['models'][0]['config']).read_text())
ds = cfg['test_dataloader']['dataset']
steps = ds.pop('pipeline')
steps[1]['scale'] = tuple(steps[1]['scale'])
ds.update(ann_file=protocol['support_annotation']['path'],
          data_root=str(R/'repeat-seed20261101-budget25-20261101/payload/data'),pipeline=[])
ds['metainfo']['classes'] = tuple(ds['metainfo']['classes'])
dataset = DATASETS.build(ds)
assert len(dataset) == 5
info = next(dataset.get_data_info(i) for i in range(5) if dataset.get_data_info(i)['img_id'] == 1892)
records = []
prepared = {}
for request in [4,0]:
    cv2.setNumThreads(request)
    reported = cv2.getNumThreads()
    if request == 0:
        assert reported == 1, 'Sequential OpenCV API not effective; stop without model retry'
    for factor in [1.0,0.8]:
        pipeline = copy.deepcopy(steps)
        pipeline.insert(1,dict(type='Color',prob=1.0,level=0,min_mag=factor,max_mag=factor))
        pipeline = Compose(pipeline)
        began = time.monotonic()
        packed = pipeline(copy.deepcopy(info))
        elapsed = time.monotonic()-began
        gt = packed['data_samples'].gt_instances
        boxes = gt.bboxes.tensor if hasattr(gt.bboxes,'tensor') else gt.bboxes
        values = dict(inputs=packed['inputs'].numpy().tobytes(),
                      bboxes=boxes.numpy().tobytes(), labels=gt.labels.numpy().tobytes(),
                      metainfo=packed['data_samples'].metainfo)
        prepared[(request,factor)] = values
        records.append(dict(requested_cv2_threads=request,reported_cv2_threads=reported,
                            factor=factor,pipeline_seconds=elapsed,
                            input_shape=list(packed['inputs'].shape),
                            input_sha256=hashlib.sha256(values['inputs']).hexdigest(),
                            bbox_sha256=hashlib.sha256(values['bboxes']).hexdigest(),
                            labels_sha256=hashlib.sha256(values['labels']).hexdigest()))
for factor in [1.0,0.8]:
    assert prepared[(4,factor)] == prepared[(0,factor)], 'Thread control changed packed input/GT/metadata'
result = dict(status='NO_MODEL_THREAD_CONTRACT_PASS', model_instances=0, forward_calls=0,
              dev_test_images=0, image_id=1892, records=records,
              prepared_tensors_gt_metadata_byte_identical=True,
              parallel_framework=[line.strip() for line in cv2.getBuildInformation().splitlines() if 'Parallel framework' in line],
              original_measured_cv2_setting={'requested':4,'reported':records[0]['reported_cv2_threads'],
                  'positive_limit_enforcement':False},
              proposed_formal_runtime={'torch_threads':4,'interop_threads':1,'cv2_setNumThreads':0,
                  'cv2_getNumThreads':1,'cv2_parallel_regions':'sequential'},
              official_source='https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp',
              no_model_rerun_under_proposed_setting=True, full_512_authorized=False)
out = P/'receipts/THREAD_CONTRACT_CHECK.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
