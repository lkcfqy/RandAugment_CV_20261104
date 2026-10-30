#!/usr/bin/env python3
"""Rebuild the pinned environment offline into a NEW directory; no model run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    origin = json.loads((root / 'environment-origin.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--python', default=origin['python_base_executable'])
    args = parser.parse_args()
    target = args.target.expanduser().absolute()
    if target.exists():
        parser.error('target already exists; no files will be overwritten')
    info = json.loads(subprocess.check_output([
        args.python, '-c',
        'import sys,platform,json; print(json.dumps({"version":sys.version,"machine":platform.machine(),"system":platform.system()}))',
    ], text=True))
    if info['version'] != origin['python_version'] or info['machine'] != 'arm64' or info['system'] != 'Darwin':
        parser.error('requires the recorded CPython 3.14.10 conda-forge build on macOS arm64; see environment-origin.json')
    for wheel in json.loads((root / 'sources/wheel-manifest.json').read_text()):
        name = wheel['filename']
        if Path(name).name != name:
            raise ValueError('invalid wheel filename')
        path = root / 'wheelhouse' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != wheel['sha256']:
            raise ValueError('wheel hash mismatch: ' + name)
    subprocess.run([args.python, '-m', 'venv', '--copies', str(target)], check=True)
    python = str(target / 'bin/python')
    pip_version = subprocess.check_output([python, '-c', 'from importlib.metadata import version; print(version("pip"))'], text=True).strip()
    if pip_version != '26.2.2':
        raise RuntimeError('unexpected ensurepip version; new environment retained for inspection, no package install performed')
    pip = [python, '-m', 'pip', '--isolated', '--disable-pip-version-check', '--no-cache-dir']
    subprocess.run(pip + ['install', '--no-index', '--only-binary=:all:', '--find-links', str(root / 'wheelhouse'), '--require-hashes', '-r', str(root / 'requirements-macos-arm64-py314.lock')], check=True)
    subprocess.run(pip + ['check'], check=True)
    print('Rebuilt environment:', target)
    print('No data/weights downloaded and no acceptance or research model executed.')


if __name__ == '__main__':
    main()
