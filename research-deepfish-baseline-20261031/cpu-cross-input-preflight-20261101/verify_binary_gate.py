"""Record an unmodified wheel-metadata diagnosis, then one import-only gate."""
from portable_dev.paths import project_path as _portable_project_path
import base64
import csv
import hashlib
import importlib.metadata as md
import json
import pathlib
import platform
import struct
import sys
import time
import traceback
from email.parser import Parser

P = _portable_project_path(__file__, 'research-deepfish-baseline-20261031/cpu-cross-input-preflight-20261101')
R = P.parent
start = time.monotonic()

def dump(name, value):
    dest = P / 'receipts' / name
    assert not dest.exists(), 'Refuse overwriting gate evidence'
    dest.write_text(json.dumps(value, indent=2) + '\n')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def architectures(path):
    with path.open('rb') as f:
        data = f.read(4096)
    if data[:4] == b'\xcf\xfa\xed\xfe':
        return [hex(struct.unpack_from('<I', data, 4)[0])]
    if data[:4] == b'\xca\xfe\xba\xbe':
        n = struct.unpack_from('>I', data, 4)[0]
        return [hex(struct.unpack_from('>I', data, 8 + 20*i)[0]) for i in range(n)]
    raise RuntimeError('Unrecognized Mach-O header')

try:
    from pip._internal.operations.check import create_package_set_from_installed, check_package_set
    from pip._internal.utils.compatibility_tags import get_supported
    from pip._vendor.packaging.tags import parse_tag
    from pip._vendor.packaging.utils import parse_wheel_filename

    packages, parsing_problems = create_package_set_from_installed()
    missing, conflicting = check_package_set(packages)
    assert not parsing_problems and not missing and not conflicting, 'Actual dependency graph failure'
    supported = set(get_supported())
    reports = [json.loads((P/'receipts'/x).read_text()) for x in ['core_install.json', 'build_tools_install.json']]
    installs = {item['metadata']['name'].lower(): item for report in reports for item in report['install']}
    evidence = {'pip_check_exit_code': 1, 'pip_check_original_log': 'logs/pip_check.log',
                'missing_dependencies': missing, 'version_conflicts': conflicting,
                'metadata_parse_problems': parsing_problems, 'interpreter': sys.version,
                'machine': platform.machine(), 'packages': {}, 'binaries': []}
    for name in ['torch', 'ninja', 'torchvision', 'mmcv']:
        dist = md.distribution(name)
        wheel = dist.read_text('WHEEL')
        parsed = Parser().parsestr(wheel).get_all('Tag', [])
        raw_tags = [line[5:] for line in wheel.splitlines() if line.startswith('Tag: ')]
        download = installs[name]['download_info']
        filename = pathlib.PurePosixPath(download['url']).name
        _, _, _, filename_tags = parse_wheel_filename(filename)
        evidence['packages'][name] = {
            'version': dist.version, 'WHEEL_text': wheel,
            'WHEEL_sha256': hashlib.sha256(wheel.encode()).hexdigest(),
            'parsed_header_tags': parsed, 'raw_tag_lines': raw_tags,
            'embedded_parsed_tag_compatible': any(not parse_tag(t).isdisjoint(supported) for t in parsed),
            'filename_tags_compatible': not filename_tags.isdisjoint(supported),
            'download_info': download,
        }
    for name, relative in [
        ('torch','torch/_C.cpython-311-darwin.so'),
        ('torch','torch/lib/libtorch_cpu.dylib'),
        ('torch','torch/lib/libc10.dylib'),
        ('ninja','ninja/data/bin/ninja'),
        ('mmcv','mmcv/_ext.cpython-311-darwin.so'),
    ]:
        dist = md.distribution(name)
        path = pathlib.Path(dist.locate_file(relative))
        record = {row[0]: row[1] for row in csv.reader(dist.read_text('RECORD').splitlines())}
        actual = sha(path)
        expected = record[relative]
        assert expected == 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip('='), relative
        archs = architectures(path)
        assert '0x100000c' in archs, 'ARM64 binary absent'
        evidence['binaries'].append({'file': relative, 'sha256': actual,
                                     'installed_RECORD_matches': True, 'MachO_cpu_types': archs})
    evidence['conclusion_before_import'] = (
        'pip check remains FAILED and is retained unchanged. Its two warnings are explained by '
        'Torch ARM64 filename/binary with an embedded x86_64 Tag, and Ninja Tag lines after a blank '
        'header terminator. All dependency requirements pass. Native compatibility is not yet proven.'
    )
    evidence['validation_revision_before_model_or_effect'] = (
        'No packages, WHEEL metadata, model code or operator code are changed. One official import-only '
        'gate is used to distinguish malformed metadata from actual binary incompatibility. '
        'Any native import failure stops this implementation path; no build/fork/version fallback.'
    )
    dump('WHEEL_METADATA_DIAGNOSIS.json', evidence)
    print('METADATA_EVIDENCE_RECORDED_NO_MODEL', flush=True)

    import torch
    import torchvision
    import mmcv
    import mmengine
    import mmdet
    from mmcv.ops import MultiScaleDeformableAttention
    assert torch.__version__.split('+')[0] == '2.1.0'
    assert torchvision.__version__.split('+')[0] == '0.16.0'
    assert mmcv.__version__ == '2.1.0' and mmengine.__version__ == '0.10.7' and mmdet.__version__ == '3.3.0'
    assert torch.version.cuda is None and platform.machine() == 'arm64'
    pins = json.loads((R/'repeat-seed20261101-budget25-20261101/payload/implementation_pins.json').read_text())
    site = pathlib.Path(mmdet.__file__).parent.parent
    for name, expected in pins['file_sha256'].items():
        assert sha(site/name) == expected, name
    result = {'status':'IMPORT_ONLY_PASS_WITH_RECORDED_WHEEL_METADATA_DEFECTS',
              'seconds':time.monotonic()-start, 'torch':torch.__version__,
              'torchvision':torchvision.__version__, 'mmcv':mmcv.__version__,
              'mmengine':mmengine.__version__, 'mmdet':mmdet.__version__,
              'arch':platform.machine(), 'CUDA_build':torch.version.cuda,
              'official_implementation_pins_verified':True, 'native_source_changes':0,
              'pip_check_pass':False, 'model_instances':0, 'operator_calls':0,
              'GPU_MPS_Colab_actions':0, 'uploads':0}
    dump('OFFICIAL_IMPORT_GATE.json', result)
    print(json.dumps(result), flush=True)
except BaseException as exc:
    result = {'status':'STOPPED_NO_BINARY_FALLBACK', 'exception':type(exc).__name__,
              'reason':str(exc), 'seconds':time.monotonic()-start, 'model_instances':0,
              'GPU_MPS_Colab_actions':0, 'uploads':0}
    dump('OFFICIAL_IMPORT_GATE_FAILURE.json', result)
    traceback.print_exc()
    print(json.dumps(result), flush=True)
    raise
