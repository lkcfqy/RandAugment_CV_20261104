"""Read-only SHA verification of prior research/classroom artifacts, no decoding."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
PROJECT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
PROPOSAL=PROJECT/'augmentation-research-20261022'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify():
    report={'at_utc':datetime.now(timezone.utc).isoformat(),'old_files':[],'decoded_old_arrays_or_images':False,
            '8A_files_or_environment_accessed':False,'protected_classroom_modified':False}
    proposal_manifest=json.loads((PROPOSAL/'delivery-manifest.json').read_text())
    prior=json.loads((PROPOSAL/'preservation-final.json').read_text())
    for row in proposal_manifest['files']:
        p=PROPOSAL/row['relative_path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    cp=proposal_manifest['checkpoint'];assert sha(cp['path'])==cp['sha256']
    report['previous_proposal_payloads_unchanged']=len(proposal_manifest['files'])
    report['proposal_manifest_sha256']=sha(PROPOSAL/'delivery-manifest.json')
    report['previous_checkpoint_sha256']=sha(cp['path'])
    count=0
    for old in prior['previous_deliveries']:
        base=Path(old['root']);assert sha(base/'delivery-manifest.json')==old['manifest_sha256']
        for row in json.loads((base/'delivery-manifest.json').read_text())['files']:
            p=base/row['relative_path']
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
            count+=1
        assert sha(old['checkpoint']['path'])==old['checkpoint']['sha256']
    report['previous_research_payloads_unchanged']=count
    for row in prior['protected_checks']:
        p=PROJECT/row['relative_path']
        assert sha(p)==row['sha256'],str(p)
        if 'bytes' in row:assert p.stat().st_size==row['bytes'],str(p)
        report['old_files'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':row['sha256'],'unchanged':True})
    report['other_protected_files_unchanged']=len(report['old_files'])
    report['all_checks_passed']=True
    return report

if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False))
