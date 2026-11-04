"""Local pure guard checks. No model framework, installation or remote action."""
import copy
import importlib
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / 'payload'
sys.path.insert(0, str(P))
control = importlib.import_module('candidate_control')
worker = importlib.import_module('candidate_worker')
ContractError = control.ContractError
expected = json.loads((P / 'reference_A_expected.json').read_text())
tests = []

def check(name, function):
    function()
    tests.append({'name': name, 'passed': True})

def rejects(function):
    try:
        function()
    except ContractError:
        return
    raise AssertionError('Expected ContractError')

check('No GO rejects worker before framework imports', lambda: rejects(
    lambda: worker.worker(P / 'run_context.example.json', 'preflight_M')))
report = {k: expected[k] for k in (
    'initial_state_sha256', 'trainable_names_sha256',
    'state_tensor_bytes', 'optimizer_initial_state_empty')}
check('Exact historical initialization receipt accepted',
      lambda: control.validate_initial_report(report, expected))
for key in report:
    changed = dict(report)
    changed[key] = None
    check('Mismatch rejected: ' + key, lambda changed=changed:
          rejects(lambda: control.validate_initial_report(changed, expected)))

def optimizer_check():
    names = ['head.weight', 'backbone.weight', 'backbone.absolute_pos_embed']
    parameters = [types.SimpleNamespace(requires_grad=True) for _ in names]
    fake = types.SimpleNamespace(named_parameters=lambda: zip(names, parameters))
    groups = []
    for name, parameter in zip(names, parameters):
        lr, decay = control.expected_optimizer_values(name)
        groups.append(dict(params=[parameter], lr=lr, weight_decay=decay,
                           betas=(.9, .999), eps=1e-8, amsgrad=False))
    optimizer = types.SimpleNamespace(param_groups=groups)
    rows = control.validate_optimizer_groups(fake, optimizer)
    assert [(x['lr'], x['weight_decay']) for x in rows] == [
        (1e-4, 1e-4), (1e-5, 1e-4), (1e-4, 0.0)]
    groups[1]['lr'] = 1e-4
    rejects(lambda: control.validate_optimizer_groups(fake, optimizer))
check('Optimizer longest-key precedence and wrong group rejection', optimizer_check)

def tuple_check():
    fields = ('classes', 'scale', 'crop_size', 'img_scale', 'ratio_range',
              'center_ratio_range', 'pad_val')
    normalized = worker.normalize_config({k: [1, 2] for k in fields})
    assert all(normalized[k] == (1, 2) for k in fields)
check('Official tuple constructor fields restored', tuple_check)

def rng_check():
    record = dict(cuda_rng_before_update=expected['first_update_cuda_rng_sha256'],
                  steps=[dict(rng_before=dict(
                      torch_cpu=expected['first_pipeline_torch_CPU_rng_sha256']))])
    control.validate_first_update_rng(record, expected)
    bad = copy.deepcopy(record)
    bad['cuda_rng_before_update'] = '0' * 64
    rejects(lambda: control.validate_first_update_rng(bad, expected))
check('First-update historical RNG identity and mismatch rejection', rng_check)

def static_check():
    for f in P.glob('*.py'):
        compile(f.read_text(), f.name, 'exec')
    source = (P / 'candidate_worker.py').read_text()
    assert "expected['anchor_order'][runner.iter]" in source
    assert "validate_runtime_versions(ROOT/'requirements_runtime_lock.txt')" in source
    assert len(expected['anchor_order']) == 200
check('All packaged scripts compile and fixed runtime/anchor guards are present', static_check)

frameworks = [n for n in sys.modules if n.split('.')[0] in
              {'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'transformers'}]
assert not frameworks, frameworks
result = dict(status='PASS', tests=tests, framework_imports=0, model_calls=0,
              candidate_runtime_executed=False, test_source='test_controls.py')
(ROOT / 'NO_MODEL_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status='PASS', tests=len(tests), framework_imports=0,
                      model_calls=0)))
