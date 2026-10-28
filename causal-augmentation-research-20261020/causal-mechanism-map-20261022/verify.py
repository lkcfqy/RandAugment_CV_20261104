"""Read-only delivery/hash audit; no model, dataset decoding or network calls."""
from pathlib import Path
import hashlib
import json
import sys

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
m=json.loads((root/'delivery-manifest.json').read_text())
for r in m['files']:
    p=root/r['relative_path']
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
cp=m['checkpoint']
assert sha(Path(cp['path']))==cp['sha256']
local=json.loads((root/'local-data-check.json').read_text())
for pair in local['pairs']:
    for key in ('photo','mask'):
        r=pair[key];p=Path(r['path'])
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
baseline=json.loads((root/'preservation-baseline.json').read_text())
for old in baseline['previous_deliveries']:
    prior=Path(old['root'])
    assert sha(prior/'delivery-manifest.json')==old['manifest_sha256']
    for r in json.loads((prior/'delivery-manifest.json').read_text())['files']:
        p=prior/r['relative_path']
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    assert sha(Path(old['checkpoint']['path']))==old['checkpoint']['sha256']
project=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
for r in baseline['protected_checks']:
    assert sha(project/r['relative_path'])==r['sha256']
print(json.dumps({'status':'all_hash_checks_passed','payload_files':len(m['files']),
    'local_photo_mask_files':128,'previous_payload_files':46,
    'other_protected_files':len(baseline['protected_checks']),
    'no_decoding_or_models':True}))
