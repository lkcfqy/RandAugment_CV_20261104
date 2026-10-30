"""Retrieve bounded public author metadata/source only. Never execute it."""
import datetime
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent / 'sources' / 'car'
ROOT.mkdir(parents=True, exist_ok=True)
def fetch(url, cap):
    with urllib.request.urlopen(url, timeout=25) as response:
        body = response.read(cap + 1)
    if len(body) > cap:
        raise ValueError('bounded retrieval exceeded byte cap')
    return body
def save(name, body):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError('refusing to replace existing source: ' + name)
    else:
        path.write_bytes(body)

raw = fetch('https://zenodo.org/api/records/23642547', 1000000)
d = json.loads(raw)
save('record-23642547.json', raw)
latest_raw = fetch(d['links']['latest'], 1000000)
latest = json.loads(latest_raw)
save('latest-record-observed.json', latest_raw)
print(json.dumps({'pinned_id':d['id'], 'version':d['metadata'].get('version'), 'latest_id':latest.get('id'), 'latest_version':latest.get('metadata',{}).get('version'), 'latest_date':latest.get('metadata',{}).get('publication_date')}, ensure_ascii=False))
for item in d['files']:
    if item['key'] not in {'README.md', 'CAR_code.zip', 'MANIFEST_sha256.txt'}:
        continue
    body = fetch(item['links']['self'], 2000000)
    kind, expected = item['checksum'].split(':')
    assert kind == 'md5' and hashlib.md5(body).hexdigest() == expected
    save(item['key'], body)
    print(json.dumps({'file':item['key'],'bytes':len(body),'md5_verified':True,'sha256':hashlib.sha256(body).hexdigest()}))
    if item['key'].endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            listing = [{'path':v.filename,'bytes':v.file_size} for v in z.infolist()]
            save('code-file-index.json', json.dumps(listing,indent=2).encode())
            for entry in z.infolist():
                p = Path(entry.filename)
                if entry.is_dir() or p.is_absolute() or '..' in p.parts:
                    continue
                if entry.file_size > 500000 or p.suffix not in {'.py','.md','.txt','.yml','.yaml'}:
                    continue
                if entry.external_attr >> 16 & 0o170000 == 0o120000:
                    continue
                save('code/' + p.as_posix(), z.read(entry))
            print(json.dumps({'archive_entries':len(listing),'extracted_only_small_text_source':True}))
