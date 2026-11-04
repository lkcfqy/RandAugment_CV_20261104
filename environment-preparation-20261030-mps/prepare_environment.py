"""Create the explicitly authorized new local environment; never run project code."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

PROJECT = _portable_project_path(__file__, '')
PREP = PROJECT / 'environment-preparation-20261030-mps'
VENV = PROJECT / '.venv'
WORK = Path('/Users/lkc/Documents/Codex/2026-10-30/task-2')
HISTORY = PROJECT / 'environment-preparation-20261029'

def git(*args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(PROJECT), *args], text=True).strip()

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, value):
    (PREP / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def verify_official(item):
    url = f"https://pypi.org/pypi/{item['name']}/{item['version']}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    files = [f for f in data['urls'] if f['filename'] == item['filename']]
    if len(files) != 1:
        raise RuntimeError(f"No unique official wheel: {item['filename']}")
    wheel = files[0]
    assert not wheel['yanked'], item['filename']
    assert wheel['digests']['sha256'] == item['sha256'], item['filename']
    assert wheel['size'] == item['bytes'], item['filename']
    assert wheel['url'] == item['url'], item['filename']
    assert wheel['url'].startswith('https://files.pythonhosted.org/'), item['filename']
    return {**item, 'metadata_url': url, 'requires_python': data['info']['requires_python'], 'official_metadata_matched': True}

if os.path.lexists(PREP) or os.path.lexists(VENV):
    raise SystemExit('Preparation directory or .venv already exists; refusing to replace it.')
if git('status', '--porcelain', '--untracked-files=all'):
    raise SystemExit('Checkout is not clean; inspect existing work before proceeding.')
if sys.version_info[:2] != (3, 14):
    raise SystemExit('This historical dependency lock targets Python 3.14.')

source_manifest = json.loads((HISTORY / 'sources/wheel-manifest.json').read_text())
with ThreadPoolExecutor(max_workers=4) as pool:
    verified_sources = list(pool.map(verify_official, source_manifest))
print('Official PyPI metadata and historical wheel hashes matched:', len(verified_sources), flush=True)

tracked = git('ls-files', '-z').split('\0')
tracked_hashes = {name: digest(PROJECT / name) for name in tracked if name}
included = json.loads((PROJECT / 'backup-20261030/INCLUDED_PROJECT_FILES.json').read_text())['files']
mismatches = [item['path'] for item in included if tracked_hashes.get(item['path']) != item['sha256'] or (PROJECT / item['path']).stat().st_size != item['bytes']]
if mismatches:
    raise SystemExit(f'Historical included-file manifest failed: {mismatches}')

PREP.mkdir()
(PREP / '.gitignore').write_text('wheelhouse/\n__pycache__/\n')
(PREP / 'logs').mkdir()
shutil.copyfile(__file__, PREP / 'prepare_environment.py')
shutil.copyfile(WORK / 'probe_randaugment_mps.py', PREP / 'probe_mps.py')
shutil.copyfile(HISTORY / 'requirements-macos-arm64-py314.lock', PREP / 'requirements-macos-arm64-py314.lock')
save('official-wheel-verification.json', verified_sources)
save('tracked-sha256-before.json', tracked_hashes)
save('repository-before.json', {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'project': str(PROJECT), 'head': git('rev-parse', 'HEAD'),
    'branch': git('branch', '--show-current'), 'remote': git('remote', 'get-url', 'origin'),
    'shallow': git('rev-parse', '--is-shallow-repository'),
    'tracked_files': len(tracked_hashes), 'included_manifest_verified': len(included),
    'tracked_sha256_manifest': 'tracked-sha256-before.json',
    'base_python': sys.executable, 'python_version': sys.version,
})
print('Historical included-file hashes verified:', len(included), flush=True)

env = {k: v for k, v in os.environ.items() if not k.startswith(('PIP_', 'PYTHON'))}
env['PIP_CONFIG_FILE'] = '/dev/null'
started = time.monotonic()
subprocess.run([sys.executable, '-I', '-m', 'venv', str(VENV)], check=True, timeout=60)
python = str(VENV / 'bin/python')
pip = [python, '-I', '-m', 'pip', '--disable-pip-version-check', '--no-input']
lock = str(PREP / 'requirements-macos-arm64-py314.lock')
wheelhouse = PREP / 'wheelhouse'
steps = [
    ('download', [*pip, 'download', '--no-cache-dir', '--only-binary=:all:', '--require-hashes', '--index-url', 'https://pypi.org/simple', '-r', lock, '-d', str(wheelhouse)]),
    ('install', [*pip, 'install', '--no-index', '--only-binary=:all:', '--require-hashes', '--find-links', str(wheelhouse), '--report', str(PREP / 'pip-install-report.json'), '-r', lock]),
    ('check', [*pip, 'check']),
]
receipts = []
for name, cmd in steps:
    print('STEP', name, flush=True)
    with (PREP / 'logs' / f'pip-{name}.log').open('w') as log:
        cp = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
    receipts.append({'step': name, 'exit_code': cp.returncode, 'log': f'logs/pip-{name}.log'})
    save('installation-steps.json', receipts)
    cp.check_returncode()
    if name == 'download':
        actual = sorted(p.name for p in wheelhouse.glob('*.whl'))
        assert actual == sorted(item['filename'] for item in verified_sources)
        for item in verified_sources:
            assert digest(wheelhouse / item['filename']) == item['sha256'], item['filename']
        print('Downloaded wheel hashes verified:', len(actual), flush=True)
freeze = subprocess.check_output([*pip, 'freeze', '--all'], env=env, text=True)
(PREP / 'installed-freeze.txt').write_text(freeze)
save('installation-result.json', {'status': 'passed', 'venv': str(VENV), 'python': python, 'elapsed_seconds': time.monotonic() - started, 'wheel_count': len(verified_sources)})
print('ISOLATED_INSTALLATION_PASSED', python, flush=True)
