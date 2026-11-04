"""Read-only source/annotation audit. Never import or execute upstream code."""
from portable_dev.paths import project_path as _portable_project_path
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = _portable_project_path(__file__, 'research-deepfish-baseline-20261031')
ETS = HERE / 'official_ets'
checks = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    checks.append({'name': name, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(name)


def static_assignments(path):
    """Support literals, dict(), references and +; report unknowns explicitly."""
    ns = {}

    def value(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return copy.deepcopy(ns.get(n.id, {'UNRESOLVED': n.id}))
        if isinstance(n, (ast.List, ast.Tuple)):
            return [value(x) for x in n.elts]
        if isinstance(n, ast.Dict):
            return {value(k): value(v) for k, v in zip(n.keys, n.values)}
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'dict' and not n.args:
            return {k.arg: value(k.value) for k in n.keywords}
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -value(n.operand)
        if isinstance(n, ast.Attribute) and ast.dump(n) == ast.dump(ast.parse('_base_.backend_args', mode='eval').body):
            return None  # Both referenced dataset bases declare backend_args=None.
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            left, right = value(n.left), value(n.right)
            if type(left) is type(right) and isinstance(left, (str, list, int, float)):
                return left + right
        return {'UNRESOLVED_AST': ast.dump(n)}

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ns[target.id] = value(node.value)
    return ns


cfg_path = ROOT / 'repeat-seed20261101-budget25-20261101/payload/config_A.json'
a = json.loads(cfg_path.read_text())
a_pipeline = a['train_dataloader']['dataset']['pipeline'][0]['transforms']
default = static_assignments(HERE / 'official_mmdet_swint_config.py')
ets_base = static_assignments(ETS / 'mmdetection/configs/_base_/datasets/CDFSOD_detection_few-shot.py')
ets_top = static_assignments(ETS / 'mmdetection/configs/grounding_dino/CDFSOD/GroudingDINO-few-shot.py')
ets_b = static_assignments(ETS / 'mmdetection/configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py')
check('A geometry equals official MMDetection default', a_pipeline[3:] == default['train_pipeline'][2:])
check('A explicit identity Color', a_pipeline[2] == {'level': 0, 'max_mag': 1.0, 'min_mag': 1.0, 'prob': 1.0, 'type': 'Color'})
check('ETS shared flip equals A', ets_base['train_pipeline'][4] == a_pipeline[3])
check('ETS shared resize/crop and packing equal A', ets_base['train_pipeline'][6:] == a_pipeline[4:])
check('ETS model config has no active pipeline override', 'train_pipeline' not in ets_top and 'train_dataloader' not in ets_top)
check('ETS base leaves data_root unset', 'data_root' not in ets_base)
check('ETS base leaves metainfo unset', 'metainfo' not in ets_base)
check('ETS top is 15 epoch schedule', ets_top['train_cfg'] == {'max_epochs': 15, 'type': 'EpochBasedTrainLoop', 'val_interval': 5})
official_model = copy.deepcopy(default['model'])
official_model['bbox_head']['num_classes'] = 1
official_model['language_model']['name'] = 'assets/bert-base-uncased'
check('A model differs from official config only by class count and local BERT path', a['model'] == official_model)
official_optim = copy.deepcopy(default['optim_wrapper'])
official_optim.pop('_delete_')
check('A optimizer equals official default after removing merge directive', a['optim_wrapper'] == official_optim)

explicit = copy.deepcopy(ets_base['train_pipeline'])
explicit[0].update(color_type='color', imdecode_backend='cv2')
explicit[2].update(center_ratio_range=[0.5, 1.5], bbox_clip_border=True, max_cached_images=40, random_pop=True)
explicit[3].update(hue_delta=5, saturation_delta=30, value_delta=30)
explicit[4].update(direction='horizontal')
explicit[5].update(flip_ratio=0.5, max_iters=15, bbox_clip_border=True, random_pop=True)
explicit[6].update(prob=[0.5, 0.5])
selected = {}
for i in (1, 2, 3):
    file = f'configs/5-shot-dataset{i}.py'
    ns = static_assignments(ETS / file)
    selected[file] = {'train_cfg': ns['train_cfg'], 'train_dataloader': ns['train_dataloader'], 'load_from': ns.get('load_from'), 'model_backbone': ns['model']['backbone']}

out = {
    'method': 'stdlib AST and JSON only; no eval, exec, upstream imports, model or transform calls',
    'format_note': 'Static JSON evidence, not an executable MMEngine config. Tuple constructor requirements must be preserved in a future Python config.',
    'A_pipeline': a_pipeline, 'official_MMDetection_pipeline': default['train_pipeline'],
    'ETS_active_fixed_template_pipeline': ets_base['train_pipeline'],
    'ETS_fixed_template_with_explicit_defaults': explicit,
    'ETS_training_template': {k: ets_top.get(k) for k in ['train_cfg', 'param_scheduler', 'optim_wrapper', 'default_hooks']},
    'ETS_SwinB_overrides': ets_b,
    'ETS_data_root_and_metainfo_unresolved': True,
    'selected_submission_configs': selected,
}
(HERE / 'PIPELINES_STATIC.json').write_text(json.dumps(out, indent=2) + '\n')

inputs = [cfg_path, ROOT / 'repeat-seed20261101-budget25-20261101/payload/model_worker.py', ROOT / 'repeat-seed20261101-budget25-20261101/gpu-run/artifacts/formal_A.json', ROOT / 'assets/annotation_docs/FISH/annotations/train.json', ROOT / 'assets/annotation_docs/FISH/annotations/5_shot.json', ROOT / 'repeat-seed20261101-budget25-20261101/payload/data/annotations/support5.json', ROOT / 'followup-group-provenance-20261031/reports/DEV_AND_BOX_CONTRACT_ZH.md', ROOT / 'followup-group-provenance-20261031/reports/DECODED_CONTRACT_SAMPLES.json', ROOT / 'followup-group-provenance-20261031/preview/support_contract_sheet.jpg', ROOT / 'cpu-cross-input-diagnostic-20261101/DELIVERABLE_MANIFEST.json', ROOT / 'repeat-seed20261101-budget25-20261101/FINAL_ARCHIVE_SHA256.json']
inp = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in inputs]
(HERE / 'LOCAL_INPUT_MANIFEST.json').write_text(json.dumps(inp, indent=2) + '\n')
check('Closed CPU archive manifest unchanged', sha(ROOT / 'cpu-cross-input-diagnostic-20261101/DELIVERABLE_MANIFEST.json') == 'cf6f3d1af3dbda3843b12ed9c41a5547f7da822bc6bb35b73bec520b47d737b6')
support = json.loads((HERE / 'SUPPORT_COMPLETENESS.json').read_text())
train_data = json.loads((ROOT / support['inputs']['train']['relative_path']).read_text())
shot_path = ROOT / support['inputs']['used_support']['relative_path']
shot_data = json.loads(shot_path.read_text())
for row in support['rows']:
    actual_train = [x for x in train_data['annotations'] if x['image_id'] == row['image_id']]
    actual_shot = [x for x in shot_data['annotations'] if x['image_id'] == row['image_id']]
    check(f"Support evidence recomputed from original files: {row['image_id']}", actual_train == row['train_annotations'] == actual_shot == row['shot_annotations'])
check('Used support bytes equal official shot', support['used_support_equals_official_shot_bytes'])
check('All five train annotation records retained', support['all_support_train_annotations_preserved'])
check('Support image identities match full train', all(x['train_image_record_matches'] for x in support['rows']))
check('Support counts are 1/1 in every image', all(x['train_annotation_count'] == x['shot_annotation_count'] == 1 for x in support['rows']))
for cls, row in json.loads((HERE / 'TRANSFORM_STATIC_COMPARISON.json').read_text())['classes'].items():
    check(f'ETS/MMDetection class AST identical: {cls}', row['equal_including_docstrings'])
formal = json.loads((ROOT / 'repeat-seed20261101-budget25-20261101/gpu-run/artifacts/formal_A.json').read_text())
check('Existing A actual U200 and forty exposures per image', formal['actual_updates'] == 200 and set(formal['exposures']['counts'].values()) == {40})
(HERE / 'STATIC_VERIFICATION.json').write_text(json.dumps({'status': 'PASS', 'checks': checks, 'model_calls': 0, 'upstream_code_executions': 0, 'installs': 0, 'uploads': 0, 'GPU_allocations': 0, 'new_CU': 0}, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'checks': len(checks), 'model_calls': 0}))
