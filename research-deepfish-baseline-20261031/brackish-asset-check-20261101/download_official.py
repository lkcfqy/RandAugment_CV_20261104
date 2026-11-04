"""Bounded unauthenticated download from author-linked Kaggle fixed version4."""
import pathlib,json,urllib.request,urllib.parse,urllib.error,hashlib,datetime,time,os
P=pathlib.Path(__file__).resolve().parent
META=json.loads((P/'sources/kaggle_metadata.json').read_text())
assert META['ref']=='aalborguniversity/brackish-dataset' and META['currentVersionNumber']==4 and META['licenseName']=='CC BY-SA 4.0' and META['isPrivate'] is False
assert META['totalBytes']==197144979
URL='https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/brackish-dataset?datasetVersionNumber=4'
CAP=300*1024**2
deadline=datetime.datetime(2026,11,1,19,8,49,tzinfo=datetime.timezone.utc).timestamp()
dest=P/'assets/brackish-dataset-v4.zip';partial=dest.with_suffix('.zip.partial')
assert not dest.exists() and not partial.exists()
receipt={'official_URL':URL,'fixed_dataset_version':4,'dataset_license':META['licenseName'],'publisher_totalBytes':META['totalBytes'],'authentication_or_credentials_used':False,'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'HTTP_attempts':[]}
def save(): (P/'sources/DOWNLOAD_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
save()
try:
    for attempt in range(2):
        try:
            response=urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'Brackish-metadata-audit/1.0'}),timeout=30)
            break
        except urllib.error.HTTPError as e:
            receipt['HTTP_attempts'].append({'status':e.code,'attempt':attempt+1,'executed':'HTTP request only, no archive body saved'});save()
            if attempt or e.code not in (408,429,500,502,503,504):raise
            time.sleep(1)
    with response:
        parsed=urllib.parse.urlparse(response.geturl())
        assert parsed.scheme=='https' and (parsed.hostname in ('www.kaggle.com','kaggle.com','storage.googleapis.com') or parsed.hostname.endswith('.googleapis.com'))
        headers={k:response.headers.get(k) for k in ['Content-Length','Content-Type','Content-Disposition','ETag','Last-Modified','x-goog-hash']}
        length=int(headers['Content-Length']) if headers['Content-Length'] else None
        if length is None or not 0<length<=CAP:raise RuntimeError('Archive length absent or over authorized300MiB; stop before body')
        if 'zip' not in (headers['Content-Type'] or '') and 'octet-stream' not in (headers['Content-Type'] or ''):raise RuntimeError('Not an official binary archive response')
        receipt.update(status='DOWNLOADING',headers=headers,final_origin=parsed.scheme+'://'+parsed.netloc,final_path=parsed.path,expected_download_bytes=length)
        receipt['HTTP_attempts'].append({'status':response.status,'attempt':len(receipt['HTTP_attempts'])+1});save()
        h=hashlib.sha256();md5=hashlib.md5();count=0
        with partial.open('xb') as f:
            while True:
                if time.time()>=deadline:raise TimeoutError('30minute task deadline')
                b=response.read(1024**2)
                if not b:break
                if count==0 and not b.startswith(b'PK'):raise RuntimeError('Archive magic not ZIP')
                count+=len(b)
                if count>CAP or count>length:raise RuntimeError('Advertised/raw archive cap exceeded')
                f.write(b);h.update(b);md5.update(b)
            f.flush();os.fsync(f.fileno())
        if count!=length:raise RuntimeError('Truncated or overlong download')
        etag=(headers['ETag'] or '').strip('"')
        if len(etag)==32 and all(c in '0123456789abcdefABCDEF' for c in etag):
            if md5.hexdigest()!=etag.lower():raise RuntimeError('Server ETag/MD5 mismatch')
            receipt['server_MD5_ETag_verified']=True
        else:receipt['server_MD5_ETag_verified']=False
        os.rename(partial,dest)
        receipt.update(status='COMPLETE',bytes=count,sha256=h.hexdigest(),md5=md5.hexdigest(),completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save();print(json.dumps(receipt,ensure_ascii=False),flush=True)
except BaseException as e:
    receipt.update(status='FAILED_OR_STOPPED',reason=type(e).__name__+': '+str(e),partial_bytes=partial.stat().st_size if partial.exists() else 0);save();raise
