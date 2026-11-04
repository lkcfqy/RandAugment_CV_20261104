"""One bounded CPU engineering probe: four checkpoints, one support image, eight passes.

No training, evaluator, dev/test image iteration, prediction persistence, or full-run mode.
"""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import time
import traceback

P = pathlib.Path(__file__).resolve().parent
R = P.parent
PAYLOAD = R / 'repeat-seed20261101-budget25-20261101/payload'
PROTOCOL_SHA = '0b4101c0e9e41904c8dc23618439fa2499815eb136ea106bd3c48ef1ed02e305'

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def research_bytes():
    return sum(x.stat().st_size for x in R.rglob('*') if x.is_file())

def normalize_config(value):
    # Same tuple restoration as the existing successful official CUDA driver.
    if isinstance(value, list):
        return [normalize_config(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {k: normalize_config(v) for k, v in value.items()}
    for key in ('classes', 'scale', 'crop_size'):
        if isinstance(result.get(key), list):
            result[key] = tuple(result[key])
    if 'scales' in result:
        result['scales'] = [tuple(x) if isinstance(x, list) else x for x in result['scales']]
    return result

def read_protocol():
    assert sha(P/'PROTOCOL.json') == PROTOCOL_SHA, 'Frozen protocol differs'
    protocol = json.loads((P/'PROTOCOL.json').read_text())
    assert protocol['probe_image']['image_id'] == 1892
    assert protocol['preflight']['maximum_forward_calls_total'] == 8
    assert len(protocol['models']) == 4
    assert json.loads((P/'receipts/OFFICIAL_IMPORT_GATE.json').read_text())['official_implementation_pins_verified']
    return protocol

def worker(index):
    started = time.monotonic()
    protocol = read_protocol()
    spec = protocol['models'][index]
    label = str(spec['seed']) + '_' + spec['arm']
    receipt = P/'receipts'/('probe_' + label + '.json')
    assert not receipt.exists(), 'No repeated engineering model runs'
    report = dict(status='STARTED', model=label, protocol_sha256=PROTOCOL_SHA,
                  forward_calls_started=0, forward_calls_completed=0,
                  dev_test_images=0, metrics_computed=False, predictions_saved=False,
                  GPU_MPS_Colab_actions=0, uploads=0, conditions=[])
    write_json(receipt, report)
    try:
        assert os.environ['HF_HUB_OFFLINE'] == '1' and os.environ['TRANSFORMERS_OFFLINE'] == '1'
        assert os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] == '1'
        import random
        import numpy as np
        import cv2
        import torch
        import mmcv
        import mmdet
        from mmengine.dataset import Compose, pseudo_collate
        from mmdet.registry import MODELS, DATASETS
        from mmdet.utils import register_all_modules
        from mmdet.datasets.transforms import Color

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        cv2.setNumThreads(4)
        assert torch.version.cuda is None and torch.get_num_threads() == 4
        assert torch.get_num_interop_threads() == 1
        random.seed(spec['seed'])
        np.random.seed(spec['seed'])
        # Seed only the CPU generator; torch.manual_seed also visits other backends.
        torch.random.default_generator.manual_seed(spec['seed'])
        register_all_modules(init_default_scope=True)
        mmcv.use_backend('cv2')
        assert mmcv.adjust_color.__globals__.get('imread_backend') == 'cv2'
        pins = json.loads((PAYLOAD/'implementation_pins.json').read_text())
        site = pathlib.Path(mmdet.__file__).parent.parent
        for name, expected in pins['file_sha256'].items():
            assert sha(site/name) == expected, name
        assert sha(spec['config']) == spec['config_sha256']
        cfg = normalize_config(json.loads(pathlib.Path(spec['config']).read_text()))
        bert = R/'assets/bert-base-uncased'
        assets = json.loads((PAYLOAD/'asset_pins.json').read_text())['assets'][1:]
        assert {p.name for p in bert.iterdir() if p.is_file()} == {pathlib.PurePosixPath(a['path']).name for a in assets}
        for asset in assets:
            f = bert/pathlib.PurePosixPath(asset['path']).name
            assert f.stat().st_size == asset['bytes'] and sha(f) == asset['sha256']
        cfg['model']['language_model']['name'] = str(bert)
        report['environment'] = dict(torch=torch.__version__, mmcv=mmcv.__version__,
                                     mmdet=mmdet.__version__, cv2=cv2.__version__,
                                     torch_threads=4, interop_threads=1, cv2_threads=cv2.getNumThreads(),
                                     device='cpu', dtype='float32', autocast=False,
                                     deterministic_algorithms=True)
        report['status'] = 'BUILDING_MODEL'
        write_json(receipt, report)
        load_start = time.monotonic()
        assert pathlib.Path(spec['checkpoint']).stat().st_size == spec['bytes']
        assert sha(spec['checkpoint']) == spec['sha256']
        model = MODELS.build(cfg['model'])
        model.init_weights()
        model.to(torch.device('cpu'))
        checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
        assert isinstance(checkpoint, dict) and all(isinstance(t, torch.Tensor) for t in checkpoint.values())
        model_state = model.state_dict()
        assert set(model_state) == set(checkpoint), 'Complete checkpoint/model keys differ'
        for key in model_state:
            assert model_state[key].dtype == checkpoint[key].dtype
            assert model_state[key].shape == checkpoint[key].shape
            assert checkpoint[key].device.type == 'cpu'
            if checkpoint[key].is_floating_point():
                assert bool(torch.isfinite(checkpoint[key]).all()), 'Nonfinite checkpoint tensor'
        incompatible = model.load_state_dict(checkpoint, strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        expected_state = hashlib.sha256()
        actual_state = hashlib.sha256()
        state_bytes = 0
        # Compare bytes (including signed zero), not only floating-point equality.
        for key, actual in sorted(model.state_dict().items()):
            source = checkpoint[key]
            assert actual.device.type == 'cpu' and actual.dtype == source.dtype and actual.shape == source.shape
            header = json.dumps([key, str(source.dtype), list(source.shape)], separators=(',', ':')).encode()
            source_bytes = source.detach().contiguous().numpy().tobytes()
            loaded_bytes = actual.detach().contiguous().numpy().tobytes()
            assert source_bytes == loaded_bytes, 'Loaded tensor differs: ' + key
            expected_state.update(header); expected_state.update(source_bytes)
            actual_state.update(header); actual_state.update(loaded_bytes)
            state_bytes += len(source_bytes)
        assert expected_state.digest() == actual_state.digest()
        report['load'] = dict(seconds=time.monotonic()-load_start, checkpoint_sha256=spec['sha256'],
                              state_sha256=actual_state.hexdigest(), state_tensor_count=len(checkpoint),
                              state_tensor_bytes=state_bytes, strict=True, all_tensors_byte_identical=True,
                              all_checkpoint_values_finite=True, weights_only=True,
                              missing_keys=[], unexpected_keys=[])
        del checkpoint, model_state, source, actual, source_bytes, loaded_bytes
        model.eval()
        assert not model.training
        assert all(t.device.type == 'cpu' for t in list(model.parameters()) + list(model.buffers()))

        image = protocol['probe_image']
        assert sha(image['path']) == image['sha256']
        support = protocol['support_annotation']
        assert sha(support['path']) == support['sha256']
        ds_cfg = copy.deepcopy(cfg['test_dataloader']['dataset'])
        original_pipeline = ds_cfg.pop('pipeline')
        assert [x['type'] for x in original_pipeline] == ['LoadImageFromFile','FixScaleResize','LoadAnnotations','PackDetInputs']
        assert original_pipeline[0]['imdecode_backend'] == 'cv2'
        assert original_pipeline[1]['scale'] == (800,1333) and original_pipeline[1]['keep_ratio']
        ds_cfg.update(ann_file=support['path'], data_root=str(PAYLOAD/'data'), pipeline=[])
        dataset = DATASETS.build(ds_cfg)
        assert len(dataset) == 5
        selected = [dataset.get_data_info(i) for i in range(5) if dataset.get_data_info(i)['img_id'] == image['image_id']]
        assert len(selected) == 1
        info = selected[0]
        assert pathlib.Path(info['img_path']).resolve() == pathlib.Path(image['path']).resolve()
        base = Compose([original_pipeline[0]])(copy.deepcopy(info))
        raw = base['img']
        assert raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 3
        rng_start = np.random.get_state()
        transformed = []
        rng_states = []
        for factor in [1.0,0.8]:
            np.random.set_state(rng_start)
            sample = Color(prob=1.0,level=0,min_mag=factor,max_mag=factor)(copy.deepcopy(base))
            assert sample['img'].dtype == np.uint8 and sample['img'].shape == raw.shape
            assert sample['instances'] == base['instances']
            transformed.append(sample['img'])
            s = np.random.get_state()
            rng_states.append((s[0], s[1].tobytes(), s[2], s[3], s[4]))
        assert np.array_equal(transformed[0], raw)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        expected = cv2.addWeighted(raw,0.8,np.repeat(gray[...,None],3,axis=2),0.2,0)
        assert np.array_equal(transformed[1], expected)
        assert rng_states[0] == rng_states[1]
        np.random.set_state(rng_start); np.random.rand()
        s = np.random.get_state()
        assert rng_states[0] == (s[0],s[1].tobytes(),s[2],s[3],s[4])
        report['image_contract'] = dict(image_id=image['image_id'], image_sha256=image['sha256'],
                                       raw_shape=list(raw.shape), raw_dtype='uint8', channel_order='BGR',
                                       backend='cv2', A_pixel_identity=True, B_exact_fixed_blend=True,
                                       equal_rng_consumption=True, one_numpy_draw=True,
                                       Color_preserves_geometry_and_instances=True,
                                       support_annotation_sha256=support['sha256'])
        reference_geometry = None
        for factor in [1.0,0.8]:
            steps = copy.deepcopy(original_pipeline)
            steps.insert(1,dict(type='Color',prob=1.0,level=0,min_mag=factor,max_mag=factor))
            pipeline = Compose(steps)
            p0 = time.monotonic()
            packed = pipeline(copy.deepcopy(info))
            sample = packed['data_samples']
            gt = sample.gt_instances
            boxes = gt.bboxes.tensor if hasattr(gt.bboxes,'tensor') else gt.bboxes
            geometry = dict(input_shape=list(packed['inputs'].shape),
                            gt_boxes=boxes.numpy().tobytes(), labels=gt.labels.numpy().tobytes(),
                            metainfo=sample.metainfo)
            if reference_geometry is None:
                reference_geometry = geometry
            else:
                assert geometry == reference_geometry, 'Color condition changed geometry, labels, or metadata'
            batch = pseudo_collate([packed])
            pipeline_seconds = time.monotonic()-p0
            report['forward_calls_started'] += 1
            report['status'] = 'FORWARD_STARTED'
            write_json(receipt, report)
            f0 = time.monotonic()
            with torch.inference_mode():
                output = model.test_step(batch)
            forward_seconds = time.monotonic()-f0
            assert len(output) == 1
            pred = output[0].pred_instances
            checks = {'boxes':pred.bboxes, 'scores':pred.scores, 'labels':pred.labels}
            shapes = {k:list(v.shape) for k,v in checks.items()}
            assert pred.bboxes.shape == (300,4) and pred.scores.shape == (300,) and pred.labels.shape == (300,)
            assert all(v.device.type == 'cpu' for v in checks.values())
            assert all(bool(torch.isfinite(v).all()) for v in checks.values())
            report['forward_calls_completed'] += 1
            report['conditions'].append(dict(factor=factor, pipeline_seconds=pipeline_seconds,
                forward_seconds=forward_seconds, pipeline_plus_forward_seconds=pipeline_seconds+forward_seconds,
                output_shapes=shapes, output_finite=True, output_device='cpu', prediction_values_retained=False))
            report['status'] = 'FORWARD_FINISHED'
            write_json(receipt, report)
            # Do not persist, print, match to GT, or summarize prediction values.
            del pred, checks, output, packed, batch, sample, gt, boxes
        assert sha(support['path']) == support['sha256']
        report['image_contract']['packed_geometry_labels_metadata_equal'] = True
        report['status'] = 'ENGINEERING_PASS'
        report['total_process_seconds'] = time.monotonic()-started
        report['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(receipt, report)
        print(json.dumps({'model':label,'status':report['status'],
                          'forward_calls':report['forward_calls_completed'],
                          'load_seconds':report['load']['seconds'],
                          'condition_seconds':[x['pipeline_plus_forward_seconds'] for x in report['conditions']],
                          'peak_RSS_bytes':report['peak_RSS_bytes']}),flush=True)
    except BaseException as exc:
        report.update(status='STOPPED', reason=type(exc).__name__+': '+str(exc),
                      total_process_seconds=time.monotonic()-started,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(receipt, report)
        traceback.print_exc()
        raise

def supervisor():
    protocol = read_protocol()
    marker = P/'receipts/CPU_PROBE_START.json'
    assert not marker.exists(), 'No repeated preflight execution'
    write_json(marker,dict(status='STARTED', protocol_sha256=PROTOCOL_SHA,
                           script_sha256=sha(__file__), maximum_forward_calls=8,
                           full_512_run_authorized=False))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               TORCH_FORCE_WEIGHTS_ONLY_LOAD='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               MPLCONFIGDIR=str(P/'.cache/matplotlib'),XDG_CACHE_HOME=str(P/'.cache'),
               HF_HOME=str(P/'.cache/huggingface'),TMPDIR=str(P/'tmp'))
    began = time.monotonic()
    models = []
    proc = None
    result = dict(status='RUNNING', uploads=0, new_CU=0, new_GPU_seconds=0,
                  full_512_forward_calls=0, dev_test_predictions=0)
    try:
        for i,spec in enumerate(protocol['models']):
            label = str(spec['seed'])+'_'+spec['arm']
            with (P/'logs'/('probe_'+label+'.log')).open('w') as log:
                child_start = time.monotonic()
                proc = subprocess.Popen([sys.executable,'-B',__file__,'--model-index',str(i)],
                                        cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if time.monotonic()-child_start > 300 or time.monotonic()-began > 900:
                        raise TimeoutError('Frozen engineering time limit reached')
                    if research_bytes() > 6*1024**3-80*1024**2:
                        raise RuntimeError('Research directory disk guard reached')
                    time.sleep(1)
                assert proc.returncode == 0, 'Worker stopped: '+label
            report = json.loads((P/'receipts'/('probe_'+label+'.json')).read_text())
            assert report['status'] == 'ENGINEERING_PASS' and report['forward_calls_completed'] == 2
            models.append(report)
            print(json.dumps({'model':label,'status':report['status'],
                              'process_seconds':report['total_process_seconds']}),flush=True)
        slowest = max(c['pipeline_plus_forward_seconds'] for m in models for c in m['conditions'])
        loads = sum(m['load']['seconds'] for m in models)
        result.update(status='ENGINEERING_PREFLIGHT_COMPLETE_NOT_FULL_DIAGNOSIS',
                      models=len(models), engineering_forward_calls=8,
                      model_load_seconds_total=loads, slowest_probe_pipeline_plus_forward_seconds=slowest,
                      planning_512_seconds=1.5*loads+512*2*slowest+600,
                      planning_model_load_seconds=1.5*loads,
                      planning_forward_seconds=512*2*slowest,
                      planning_persistence_evaluation_reserve_seconds=600,
                      planning_is_measured_full_run=False,
                      maximum_peak_RSS_bytes=max(m['peak_RSS_bytes'] for m in models))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        result.update(status='STOPPED', reason=type(exc).__name__+': '+str(exc),
                      completed_model_probes=len(models), planning_512_seconds=None)
        raise
    finally:
        result.update(actual_supervisor_seconds=time.monotonic()-began,research_bytes=research_bytes())
        write_json(P/'receipts/CPU_PROBE_TERMINAL.json', result)
        print(json.dumps(result),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-index',type=int,choices=range(4))
    args = parser.parse_args()
    if args.model_index is None:supervisor()
    else:worker(args.model_index)
