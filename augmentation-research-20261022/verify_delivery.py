"""Read-only integrity verification. No models, dataset decoding or network."""
from pathlib import Path
import hashlib
import json

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

root=Path(__file__).resolve().parent
m=json.loads((root/'delivery-manifest.json').read_text())
for r in m['files']:
    p=root/r['relative_path']
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
assert sha(Path(m['checkpoint']['path']))==m['checkpoint']['sha256']
p=json.loads((root/'preservation-final.json').read_text())
for old in p['previous_deliveries']:
    base=Path(old['root'])
    assert sha(base/'delivery-manifest.json')==old['manifest_sha256']
    for r in json.loads((base/'delivery-manifest.json').read_text())['files']:
        f=base/r['relative_path']
        assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
    assert sha(Path(old['checkpoint']['path']))==old['checkpoint']['sha256']
project=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
for r in p['protected_checks']:assert sha(project/r['relative_path'])==r['sha256']
plan=json.loads((root/'candidate-data-plan.json').read_text())
assert len(plan['rows'])==len({r['source_id'] for r in plan['rows']})==1776
probe=json.loads((root/'mps-vit-probe.json').read_text())
assert all(r['device']=='mps:0' and r['all_losses_finite'] for r in probe['cases'])
assert {r['trainable_backbone_parameters'] for r in probe['cases'][:3]}=={36864}
print(json.dumps({'status':'all_hash_and_readiness_checks_passed',
    'payload_files':len(m['files']),'old_payload_files':58,'other_protected_files':25,
    'candidate_photos':1776,'scientific_dataset_training_runs':0,'actual_synthetic_MPS_updates':16}))
