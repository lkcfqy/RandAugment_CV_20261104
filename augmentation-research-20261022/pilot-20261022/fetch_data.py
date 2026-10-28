"""Bounded licensed public data fetch; never fetch reserved confirmation pixels."""
from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures as cf
import hashlib,http.client,json,os,struct,threading,time,urllib.request,urllib.error,urllib.parse,zlib

ROOT=Path(__file__).resolve().parent
PROJECT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
PLAN=PROJECT/'augmentation-research-20261022/candidate-data-plan.json'
DATA=ROOT/'data'
REV='003aeb4bc80d584ef307dc24e8b38d26cf9f9e43'
URL='https://huggingface.co/datasets/antokun/The-Oxford-IIIT-Pet-Dataset-With-Annotations/resolve/'+REV+'/'+urllib.parse.quote('The Oxford-IIIT Pet Dataset With Annotations.zip')
ARCHIVE_BYTES=815141550
STOP=threading.Event()
LOCAL=threading.local()
RESOLVED=None

def digest(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def member(row,kind):
    if STOP.is_set():raise RuntimeError('Stopped after source failure.')
    entry=row['image_entry' if kind=='image' else 'mask_entry']
    e=row['image_zip_metadata' if kind=='image' else 'mask_zip_metadata']
    folder=DATA/('images' if kind=='image' else 'masks')
    path=folder/(row['source_id']+('.jpg' if kind=='image' else '.png'))
    if path.exists():
        b=path.read_bytes()
        assert len(b)==e['bytes'] and zlib.crc32(b)&0xffffffff==e['crc32']
        return {'source_id':row['source_id'],'class_id':row['class_id'],'partition':row['candidate_partition'],'kind':kind,'path':str(path),'bytes':len(b),'sha256':digest(b),'cache_reused':True}
    length=30+len(entry.encode())+512+e['compressed_bytes']
    assert length<1024*1024 and e['bytes']<=512000
    start=e['local_offset']
    for attempt in range(2):
        try:
            if not hasattr(LOCAL,'connection'):
                LOCAL.connection=http.client.HTTPSConnection(RESOLVED.hostname,timeout=30)
            connection=LOCAL.connection
            path_and_query=RESOLVED.path+('?' + RESOLVED.query if RESOLVED.query else '')
            connection.request('GET',path_and_query,headers={'User-Agent':'RandAugment-local-research/1.0','Range':f'bytes={start}-{start+length-1}','Connection':'keep-alive'})
            r=connection.getresponse()
            if r.status in (401,403):STOP.set();raise RuntimeError(f'HTTP{r.status} access refusal; stopped, no alternate')
            assert r.status==206, f'Range unsupported; status {r.status}, body not consumed'
            expected=f'bytes {start}-{start+length-1}/{ARCHIVE_BYTES}'
            assert r.getheader('Content-Range')==expected,'Content-Range mismatch; stop without reading body'
            block=r.read(length+1);assert len(block)==length
            break
        except urllib.error.HTTPError as ex:
            STOP.set()
            raise RuntimeError(f'HTTP {ex.code}; source stopped, no alternate or access retry') from ex
        except (TimeoutError,urllib.error.URLError,http.client.HTTPException,ConnectionError):
            if hasattr(LOCAL,'connection'):
                LOCAL.connection.close();del LOCAL.connection
            if attempt:STOP.set();raise
            time.sleep(.5)
    assert block[:4]==b'PK\x03\x04'
    nl,el=struct.unpack_from('<HH',block,26)
    assert el<=512 and block[30:30+nl].decode()==entry
    payload=block[30+nl+el:30+nl+el+e['compressed_bytes']]
    if e['method']==0:b=payload
    else:
        assert e['method']==8
        dec=zlib.decompressobj(-15);b=dec.decompress(payload,512001)
        assert dec.eof and not dec.unconsumed_tail
    assert len(b)==e['bytes'] and zlib.crc32(b)&0xffffffff==e['crc32']
    with path.open('xb') as f:f.write(b)
    return {'source_id':row['source_id'],'class_id':row['class_id'],'partition':row['candidate_partition'],
        'kind':kind,'path':str(path),'bytes':len(b),'sha256':digest(b),'range_bytes':length,'CRC_verified':True}

def main():
    global RESOLVED
    try:os.nice(10)
    except OSError:pass
    t0=time.perf_counter();DATA.mkdir(exist_ok=True)
    for name in ('images','masks'): (DATA/name).mkdir(exist_ok=True)
    plan=json.loads(PLAN.read_text())
    rows=[r for r in plan['rows'] if r['candidate_partition']!='reserved_confirmation']
    assert len(rows)==1184
    # Fetch only the published usage statement, without accepting any new terms.
    license_url='https://www.robots.ox.ac.uk/~vgg/data/pets/'
    if not (DATA/'license-receipt.json').exists():
        with urllib.request.urlopen(license_url,timeout=20) as r:
            b=r.read(1000000);assert r.status==200
        assert b'creativecommons.org/licenses/by-sa/4.0' in b
        (DATA/'official-pets-page.html').write_bytes(b)
        save(DATA/'license-receipt.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'url':license_url,
            'status':200,'sha256':digest(b),'bytes':len(b),'published_license':'CC BY-SA4.0',
            'use':'Local noncommercial research; source attribution and modification notes retained; no publication/upload.',
            'older_research_use_README_retained_in_existing_project':True,
            'repository_code_license_not_substituted_for_photo_license':True})
    else:
        assert digest((DATA/'official-pets-page.html').read_bytes())==json.loads((DATA/'license-receipt.json').read_text())['sha256']
    # Reuse the ordinary same-object redirect target in memory; never log its signed query.
    with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=20) as r:
        assert r.status==200 and int(r.headers['Content-Length'])==ARCHIVE_BYTES
        RESOLVED=urllib.parse.urlsplit(r.url);assert RESOLVED.scheme=='https'
    save(DATA/'connection-reuse-receipt.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'source_url':URL,'resolved_host':RESOLVED.hostname,'HEAD_status':200,'ordinary_redirect_of_same_public_object':True,
        'signed_query_stored_or_logged':False,'max_connections':2,
        'previous_fetcher_stopped_by_parent_to_reduce_connection_overhead':True,'data_or_scientific_selection_changed':False})
    receipts=[];errors=[]
    tasks=[(r,k) for r in rows for k in ('image','mask')]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(member,r,k):(r,k) for r,k in tasks}
        for f in cf.as_completed(futures):
            row,kind=futures[f]
            try:receipts.append(f.result())
            except Exception as ex:
                STOP.set();errors.append({'source_id':row['source_id'],'kind':kind,'error':repr(ex)})
                for pending in futures:pending.cancel()
            if len(receipts)%100==0 and receipts:
                print(json.dumps({'files':len(receipts),'of':len(tasks),'seconds':round(time.perf_counter()-t0,1)}),flush=True)
    if errors:
        save(DATA/'fetch-errors.json',{'receipts':receipts,'errors':errors,'stopped':True});raise RuntimeError(errors[0])
    assert len(receipts)==2368
    save(DATA/'photo-mask-receipt.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'source_url':URL,'source_revision':REV,'photos':1184,'masks':1184,
        'reserved_confirmation_fetched':False,'files':receipts,'wall_seconds':time.perf_counter()-t0})
    weights=DATA/'vit_b_16-c867db91.pth'
    if not weights.exists():
        u='https://download.pytorch.org/models/vit_b_16-c867db91.pth'
        h=hashlib.sha256();count=0
        with urllib.request.urlopen(u,timeout=30) as r, (DATA/'weights.partial').open('xb') as out:
            assert r.status==200
            while True:
                block=r.read(1048576)
                if not block:break
                count+=len(block);assert count<=346328529
                h.update(block);out.write(block)
        assert count==346328529 and h.hexdigest().startswith('c867db91')
        (DATA/'weights.partial').rename(weights)
        save(DATA/'weight-receipt.json',{'url':u,'bytes':count,'sha256':h.hexdigest(),'published_filename_hash_prefix_matched':True,'weights_only_load_required':True})
    print(json.dumps({'status':'data_and_weight_download_completed','files':len(receipts),'reserved_confirmation_fetched':False,'seconds':time.perf_counter()-t0}),flush=True)

if __name__=='__main__':main()
