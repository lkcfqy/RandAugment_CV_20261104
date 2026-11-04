"""Read local public text snapshots only; never import or execute upstream code."""
from pathlib import Path, PurePosixPath
from collections import Counter
import csv
import hashlib
import json
import time


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parent
    api = json.loads((root / 'hf_repo_api.json').read_text())
    rows = list(csv.DictReader((root / 'hf_Train_metadata.csv').open()))
    names = {x['rfilename'] for x in api['siblings']}
    expected_columns = ['image_file_name', 'mask_file_name', 'mask_rgb_file_name', 'domain', 'source_dataset']
    checks = []

    def check(name, value):
        assert value, name
        checks.append(name)

    check('fixed_hf_revision', api['sha'] == '225bb99a7a70588dd37e0e7e4e566e6be20ea82f')
    check('fixed_csv_sha256', hashlib.sha256((root / 'hf_Train_metadata.csv').read_bytes()).hexdigest() == 'c48d62ffc8eeb69833be2c133ba4fac590897b4c217f2f6c6b718ba2e81be30b')
    check('five_metadata_columns', list(rows[0]) == expected_columns)
    check('3321_unique_rows', len(rows) == len({tuple(x.values()) for x in rows}) == 3321)
    check('domain_counts', Counter(x['domain'] for x in rows) == {'Urban': 1766, 'Rural': 1555})
    check('all_source_dataset_loveda', {x['source_dataset'] for x in rows} == {'LoveDA'})
    for column in expected_columns[:3]:
        values = [x[column] for x in rows]
        check(column + '_unique', len(set(values)) == 3321)
        check(column + '_safe_existing_paths', all(not PurePosixPath(s).is_absolute() and '..' not in PurePosixPath(s).parts and 'Train/' + s in names for s in values))
        check(column + '_domain_and_stem_agreement', all(PurePosixPath(x[column]).parts[0] == x['domain'] and PurePosixPath(x[column]).stem == PurePosixPath(x['image_file_name']).stem for x in rows))
    check('three_pngs_per_pair', sum(s.endswith('.png') for s in names) == 9963)
    tree = json.loads((root / 'github_tree.json').read_text())
    check('github_tree_complete', tree.get('truncated') is False)
    index = json.loads((root / 'source_index.json').read_text())
    for entry in index:
        blob = (root / entry['file']).read_bytes()
        git_hash = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
        check('source_blob:' + entry['repository'] + '/' + entry['path'], git_hash == entry['git_blob_sha1'] and hashlib.sha256(blob).hexdigest() == entry['sha256'])
    result = {'status': 'PASS', 'checks_passed': len(checks), 'checks': checks, 'seconds': time.perf_counter() - start, 'external_code_executed': False, 'image_mask_weight_bytes_read': 0}
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
