"""Archive completed results only; no numerical fixture imports or execution."""
from portable_dev.paths import project_path as _portable_project_path
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

STAGE = Path('/Users/lkc/Documents/Codex/2026-10-30/task-5/research-frofa-fixture-20261030')
PROJECT = _portable_project_path(__file__, '')
TARGET = PROJECT/'research-frofa-fixture-20261030'
EXCLUDE = {'SHA256SUMS', 'archive-verification.json'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    raw = (json.dumps(data, indent=2, ensure_ascii=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(raw)


def check_history():
    before = json.loads((STAGE/'preservation-before.json').read_text())
    inventory = {}
    for base, dirs, files in os.walk(PROJECT):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', TARGET.name}
                         and not (Path(base)/d).is_symlink())
        for name in sorted(files):
            p = Path(base)/name
            st = p.lstat()
            inventory[str(p.relative_to(PROJECT))] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, mode=st.st_mode)
    changed = [p for p, state in before['files'].items() if inventory.get(p) != state]
    added_outside_target = sorted(set(inventory)-set(before['files']))
    status = subprocess.check_output(['git', '--no-optional-locks', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=PROJECT, text=True)
    filtered = ''.join(line+'\n' for line in status.splitlines() if not line[3:].startswith(TARGET.name+'/'))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    index_sha = digest(PROJECT/'.git/index')
    result = dict(checked_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                  baseline_files=len(before['files']), historical_metadata_changed=changed,
                  added_files_outside_target=added_outside_target,
                  git_head_unchanged=head == before['git_head'],
                  git_index_unchanged=index_sha == before['git_index_sha256'],
                  prior_git_status_unchanged=filtered == before['git_status'],
                  preservation_check='size, mtime_ns and mode for all historical files; HEAD, index hash and prior git status; historical content was not rehashed')
    result['passed'] = not changed and not added_outside_target and all(result[k] for k in
                       ['git_head_unchanged', 'git_index_unchanged', 'prior_git_status_unchanged'])
    if not result['passed']:
        raise RuntimeError('Historical preservation check failed: '+json.dumps(result))
    return result


def payload():
    paths = sorted(p for p in STAGE.rglob('*') if p.is_file() and p.name not in EXCLUDE)
    if any(p.is_symlink() for p in paths):
        raise RuntimeError('Refuse symlink payload')
    return {str(p.relative_to(STAGE)): digest(p) for p in paths}


def prepare():
    result = check_history()
    write_new(STAGE/'preservation-after.json', result)
    entries = payload()
    with (STAGE/'SHA256SUMS').open('x') as f:
        for rel, sha in entries.items():
            f.write(sha+'  '+rel+'\n')
    print(json.dumps(dict(prepared=True, payload_files=len(entries),
                          manifest_sha256=digest(STAGE/'SHA256SUMS'), history=result), indent=2))


def copy_verified():
    entries = {}
    for line in (STAGE/'SHA256SUMS').read_text().splitlines():
        sha, rel = line.split('  ', 1)
        entries[rel] = sha
    if entries != payload():
        raise RuntimeError('Stage payload differs from prepared manifest')
    check_history()
    TARGET.mkdir(exist_ok=True)
    if TARGET.is_symlink():
        raise RuntimeError('Refuse symlink target')
    paths = list(entries)+['SHA256SUMS']
    copied, existing_equal = [], []
    for rel in paths:
        source, dest = STAGE/rel, TARGET/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.is_symlink() or digest(dest) != digest(source):
                raise RuntimeError('Refuse overwrite of different existing file: '+str(dest))
            existing_equal.append(rel)
        else:
            with source.open('rb') as src, dest.open('xb') as out:
                shutil.copyfileobj(src, out)
            copied.append(rel)
    target_payload = {str(p.relative_to(TARGET)): digest(p) for p in sorted(TARGET.rglob('*'))
                      if p.is_file() and p.name not in EXCLUDE}
    if target_payload != entries:
        raise RuntimeError('Destination payload mismatch')
    if digest(STAGE/'SHA256SUMS') != digest(TARGET/'SHA256SUMS'):
        raise RuntimeError('Destination manifest mismatch')
    result = dict(verified_utc=datetime.datetime.now(datetime.UTC).isoformat(), source=str(STAGE),
                  destination=str(TARGET), payload_files=len(entries), copied_files=len(copied),
                  already_identical_files=len(existing_equal), overwritten_files=0,
                  payload_bytes=sum((STAGE/rel).stat().st_size for rel in entries),
                  all_payload_sha256_equal=True, manifest_sha256=digest(STAGE/'SHA256SUMS'),
                  source_and_destination_file_sets_equal=True, preservation_after_copy=check_history(),
                  excludes=['SHA256SUMS (hash recorded here)', 'archive-verification.json (self-referential)'])
    write_new(STAGE/'archive-verification.json', result)
    write_new(TARGET/'archive-verification.json', result)
    if digest(STAGE/'archive-verification.json') != digest(TARGET/'archive-verification.json'):
        raise RuntimeError('Verification record copy mismatch')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        prepare()
    elif sys.argv[1:] == ['--copy']:
        copy_verified()
    else:
        raise SystemExit('Use --prepare then --copy; does not execute mathematical fixture.')
