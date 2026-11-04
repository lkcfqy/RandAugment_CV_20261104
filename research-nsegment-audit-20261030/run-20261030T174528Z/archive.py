"""Create and verify an immutable local deliverable; no upload, no VCS actions."""
import hashlib
import json
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN='run-20261030T174528Z'
DEST=ROOT/'delivery'
DEST.mkdir(exist_ok=False)
payload=DEST/RUN
payload.mkdir()
excluded={'.venv','__pycache__','delivery'}
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(part in excluded for part in p.relative_to(ROOT).parts)]
for p in files:
    rel=p.relative_to(ROOT)
    target=payload/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
manifest=[]
for p in sorted(payload.rglob('*')):
    if p.is_file():
        data=p.read_bytes()
        manifest.append(dict(path=str(p.relative_to(payload)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(payload/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=DEST/('archive-20261030T174528Z.tar.gz')
with tarfile.open(archive,'x:gz') as tar:
    tar.add(payload,arcname=RUN)
with tarfile.open(archive,'r:gz') as tar:
    actual={m.name:m for m in tar.getmembers() if m.isfile()}
    assert set(actual)=={RUN+'/'+r['path'] for r in manifest}|{RUN+'/SHA256SUMS.json'}
    for row in manifest:
        data=tar.extractfile(actual[RUN+'/'+row['path']]).read()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    assert tar.extractfile(RUN+'/SHA256SUMS.json').read()==(payload/'SHA256SUMS.json').read_bytes()
record=dict(status='staging_archive_verified',utc=datetime.now(timezone.utc).isoformat(),
            run=RUN,files=len(manifest)+1,archive=archive.name,archive_bytes=archive.stat().st_size,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256((payload/'SHA256SUMS.json').read_bytes()).hexdigest(),
            session_elapsed_seconds=time.time()-json.loads((ROOT/'protocol.json').read_text())['start_epoch'])
(DEST/'STAGING_ARCHIVE_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
