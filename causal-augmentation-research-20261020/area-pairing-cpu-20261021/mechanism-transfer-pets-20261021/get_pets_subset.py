"""Fixed metadata-only sampling and bounded downloads of licensed Oxford Pets."""
import concurrent.futures
import hashlib
import json
import os
import pathlib
import sys
from remote_zip import RemoteZip, save_json, ROOT


def plan():
    index = json.loads((ROOT / 'zip-index-zip64.json').read_text())
    entries = index['entries']
    selected = []
    for split, filename in [('fit', 'archive-text-3.txt'), ('heldout', 'archive-text-2.txt')]:
        pool = []
        for line in (ROOT / filename).read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            name, breed, species, breed_index = line.split()
            image, mask = 'images/' + name + '.jpg', 'annotations/trimaps/' + name + '.png'
            if image not in entries or mask not in entries:
                continue
            if entries[image]['bytes'] > 512000 or entries[mask]['bytes'] > 512000:
                continue
            pool.append(dict(source_id=name, species=int(species) - 1, breed=int(breed) - 1,
                             split=split, image_entry=image, mask_entry=mask,
                             order_hash=hashlib.sha256(('pets-scale-transfer-v1:' + name).encode()).hexdigest()))
        for species in [0, 1]:
            group = sorted((x for x in pool if x['species'] == species), key=lambda x:x['order_hash'])
            assert len(group) >= 64
            for i, row in enumerate(group[:64]):
                row['candidate_role'] = 'foreground' if i < 16 else 'background'
                selected.append(row)
    assert len(selected) == len({x['source_id'] for x in selected}) == 256
    masks = sum(30 + len(x['mask_entry'].encode()) + 512 + entries[x['mask_entry']]['compressed_bytes'] for x in selected)
    result = dict(dataset='Oxford-IIIT Pet', official_license='CC BY-SA 4.0, current publisher page',
                  official_license_url='https://www.robots.ox.ac.uk/~vgg/data/pets/',
                  mirror_revision=index['revision'], candidate_masks=256, mask_range_response_bytes=masks,
                  final_target_photos=128, photo_limit_bytes_each=512000,
                  selection='fixed hash, two species, official photo trainval/test splits, no image/model scores',
                  sample_unit='source photograph, not generated composites', rows=selected)
    save_json('subset-plan.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))


def fetch(kind):
    try: os.nice(10)
    except OSError: pass
    index = json.loads((ROOT / 'zip-index-zip64.json').read_text())
    if kind == 'mask':
        plan_name = 'mask-pilot-plan.json' if (ROOT / 'mask-pilot-plan.json').exists() else 'subset-plan.json'
        rows = json.loads((ROOT / plan_name).read_text())['rows']
        directory, key = ROOT / 'masks', 'mask_entry'
    else:
        rows = json.loads((ROOT / 'pixel-photo-plan.json').read_text())['rows']
        directory, key = ROOT / 'photos', 'image_entry'
    directory.mkdir(exist_ok=False)
    def worker(row):
        reader = RemoteZip(index['revision'], index['entries'])
        result = dict(source_id=row['source_id'], entry=row[key], kind=kind)
        try:
            data = reader.member_fast(row[key], 512000)
            extension = '.png' if kind == 'mask' else '.jpg'
            path = directory / (row['source_id'] + extension)
            assert path.parent == directory
            with path.open('xb') as f: f.write(data)
            result.update(status='CRC_and_size_verified', bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          range_requests=reader.receipts)
        except Exception as e:
            result.update(status='failed_no_retry', error_type=type(e).__name__, error=str(e), range_requests=reader.receipts)
        return result
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for i, row in enumerate(executor.map(worker, rows), 1):
            results.append(row)
            if i % 32 == 0: print(json.dumps({'download_kind':kind,'finished':i,'total':len(rows)}), flush=True)
    receipt = dict(status='bounded_subset_fetch_completed', kind=kind, requested=len(rows),
                   succeeded=sum(x['status']=='CRC_and_size_verified' for x in results),
                   range_response_bytes=sum(q['bytes'] for x in results for q in x['range_requests']),
                   archive_downloaded=False, new_software=False, accounts=False, files=results)
    save_json(kind + '-fetch-receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 1: plan()
    elif sys.argv[1] in ['mask', 'photo']: fetch(sys.argv[1])
    else: raise ValueError('Use no argument for metadata plan, mask or photo for bounded fetch')
