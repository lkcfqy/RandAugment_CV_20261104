"""Approved small public JPEG subset, only after all training decisions are frozen."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime,hashlib,json
from remote_zip import RemoteZip

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    freeze=json.loads((ROOT/'training-freeze.json').read_text())
    assert freeze['protocol_sha256']==sha(ROOT/'protocol.json')
    assert freeze['training_code_sha256']==sha(ROOT/'train_probe.py')
    assert freeze['fetch_code_sha256']==sha(Path(__file__))
    assert freeze['models_sha256']==sha(ROOT/'trained-models.json')
    assert freeze['all_11_logistic_optimizations_converged']
    assert not (ROOT/'test-results.json').exists()
    directory=ROOT/'test-photos';directory.mkdir(exist_ok=False)
    rows=protocol['fresh_test_rows'];assert len(rows)==64
    assert sum(30+len(r['entry'])+512+r['compressed_bytes'] for r in rows)==protocol['new_test_HTTP_range_bytes']<10_000_000
    def fetch(row):
        z=RemoteZip(protocol['test_archive_revision'],{row['entry']:row})
        payload=z.member_fast(row['entry'],512000)
        assert payload[:2]==b'\xff\xd8' and len(payload)==row['bytes']
        path=directory/(row['source_id']+'.jpg')
        with path.open('xb') as f:f.write(payload)
        return dict(source_id=row['source_id'],Y=row['Y'],bytes=len(payload),sha256=sha(path),
            original_crc32_verified=True,requests=z.receipts)
    files=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,r):r for r in rows}
        for future in as_completed(futures):
            try:files.append(future.result())
            except Exception as e:errors.append(dict(source_id=futures[future]['source_id'],type=type(e).__name__,message=str(e)))
    actual=sum(q['bytes'] for r in files for q in r['requests'])
    result=dict(status='all64_small_public_JPEGs_CRC_and_SHA_verified' if not errors else 'partial_fetch_stop_and_inspect',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,errors=errors,
        JPEG_payload_bytes=sum(r['bytes'] for r in files),HTTP_range_bytes=actual,
        full_archive_downloaded=False,new_account=False,license='official Oxford-IIIT Pet CC BY-SA4.0',
        official_source='https://www.robots.ox.ac.uk/~vgg/data/pets/',
        pinned_mirror='https://huggingface.co/datasets/antokun/The-Oxford-IIIT-Pet-Dataset-With-Annotations/tree/'+protocol['test_archive_revision'],
        pixel_decoding_before_training_freeze=False,training_freeze_sha256=sha(ROOT/'training-freeze.json'))
    with (ROOT/'test-fetch-receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
    assert not errors and len(files)==64

if __name__=='__main__':main()
