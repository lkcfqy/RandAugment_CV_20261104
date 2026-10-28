"""Read-only hashes, never decode old arrays or change scientific artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,importlib.util
PILOT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/pilot-20261022')
EXPECTED='86b3d30c1d9f57d03a6bfb9363b26d13f93ccf50e4a08f6d8224190606f7aa38'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verify():
    assert sha(PILOT/'delivery-manifest.json')==EXPECTED
    manifest=json.loads((PILOT/'delivery-manifest.json').read_text())
    for row in manifest['files']:
        p=PILOT/row['relative_path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    assert sha(manifest['checkpoint']['path'])==manifest['checkpoint']['sha256']
    assert sha(manifest['historical_running_checkpoint']['path'])==manifest['historical_running_checkpoint']['sha256']
    spec=importlib.util.spec_from_file_location('protected_old_verifier',PILOT/'verify_previous.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);prior=m.verify()
    return {'at_utc':datetime.now(timezone.utc).isoformat(),'status':'all_original_pilot_and_prior_research_classroom_bytes_SHA_match',
        'original_pilot_payloads':len(manifest['files']),'original_pilot_manifest_sha256':EXPECTED,
        'original_norm_audit_failure_and_v2_unchanged':True,'original_root_checkpoints_unchanged':True,
        'old_proposal':prior['previous_proposal_payloads_unchanged'],'old_research':prior['previous_research_payloads_unchanged'],
        'other_protected':prior['other_protected_files_unchanged'],'old_arrays_or_images_decoded':False,
        '8A_files_env_or_processes_inspected':False}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False))
