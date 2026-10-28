"""Copy independent pre-result evidence into the new run and create its exclusive lock."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

WORK = Path(__file__).resolve().parent
OUT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/causal-resume-20261022')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not (OUT/'results.json').exists() and not (OUT/'source-training.json').exists()
    assert sha(OUT/'protocol.json') == 'ef8a096b874c86a204fe02d93d595c1f7825a53f0a829f1f92e96288ccafa4c3'
    assert sha(OUT/'experiment.py') == '88f80cdffe3d5afdebd203980e5676f67940d8c18375210a7bb3bfbc2004b048'
    review_source = WORK/'independent-review'
    assert (review_source/'static-review.json').exists() and (review_source/'static-review-addendum.json').exists()
    review_out = OUT/'independent-review'
    review_out.mkdir(exist_ok=False)
    records = []
    for source in sorted(review_source.iterdir()):
        assert source.is_file()
        target = review_out/source.name
        shutil.copyfile(source,target)
        assert sha(source) == sha(target)
        records.append({'name':source.name, 'bytes':target.stat().st_size, 'sha256':sha(target)})
    for name in ('access-record.json','freeze_run.py'):
        assert not (OUT/name).exists()
        shutil.copyfile(WORK/name,OUT/name)
    lock = {'at_utc':datetime.now(timezone.utc).isoformat(), 'protocol_sha256':sha(OUT/'protocol.json'),
            'code_sha256':sha(OUT/'experiment.py'), 'independent_pre_result_files':records,
            'code_change_before_lock':'Add camera0 and per-object complete Cartesian metadata assertions, independently checked; scientific definition unchanged.',
            'new_source_features_or_new_development_values_read':False,
            'independent_design_gate':'conditional_go_once_development_only; established contribution count0',
            'not_third_party_registration':True, 'prior_results_and_classroom_assets_preserved':True}
    with (OUT/'protocol-lock.json').open('x') as f:
        json.dump(lock,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'lock_sha256':sha(OUT/'protocol-lock.json'),'review_files':len(records),'at_utc':lock['at_utc']}))

if __name__ == '__main__':
    main()
