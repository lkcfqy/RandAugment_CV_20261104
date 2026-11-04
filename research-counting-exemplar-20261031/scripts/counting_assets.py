"""Asset-only HTTP acquisition helper. Never imports repository/model code."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib, json, time, threading, os, requests
ROOT = _portable_project_path(__file__, 'research-counting-exemplar-20261031')
LEDGER = ROOT / 'reports/download_ledger.json'
LOCK = threading.Lock()
LIMIT = int(3.5 * 1024**3)
START = 1793411700  # overwritten from first local source mtime when loading
START = min(p.stat().st_mtime for p in (ROOT/'sources').glob('*') if p.is_file())
END = START + 45*60
RECORDS = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
TOTAL = sum(r.get('bytes_received', 0) for r in RECORDS)
if not RECORDS:
    for p in (ROOT/'sources').glob('*'):
        if p.is_file():
            RECORDS.append({'path':str(p.relative_to(ROOT)), 'bytes_received':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'acquired_before_ledger':True})
    TOTAL = sum(r['bytes_received'] for r in RECORDS)

def save_records():
    tmp = LEDGER.with_suffix('.tmp')
    tmp.write_text(json.dumps(RECORDS, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(LEDGER)

def get(url, relative, max_bytes=30_000_000, headers=None, expected_sha=None, allowed_status=(200,), timeout=45):
    global TOTAL
    dest = ROOT / relative
    if dest.exists():
        return dest
    if time.time() > END:
        raise RuntimeError('45-minute acquisition limit reached')
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec={'url':url,'path':relative,'headers_requested':headers or {},'started_unix':time.time(),'bytes_received':0}
    part=dest.with_suffix(dest.suffix+'.part')
    digest=hashlib.sha256()
    try:
        with requests.get(url,headers=headers,stream=True,timeout=(15, timeout)) as res:
            rec.update(status=res.status_code,headers={k:v for k,v in res.headers.items() if k.lower() in {'content-length','content-range','content-type','etag','accept-ranges','last-modified','x-linked-etag','x-linked-size','x-repo-commit','content-disposition'}},final_url=res.url.split('?')[0])
            if res.status_code not in allowed_status:
                raise RuntimeError('Unexpected HTTP '+str(res.status_code))
            length=int(res.headers.get('Content-Length','0'))
            if length>max_bytes:
                raise RuntimeError('Content-Length exceeds per-request limit; response body not read')
            with part.open('xb') as out:
                for chunk in res.iter_content(1024*1024):
                    if time.time()>END:
                        raise RuntimeError('45-minute acquisition limit reached')
                    with LOCK:
                        TOTAL += len(chunk)
                        rec['bytes_received'] += len(chunk)
                        if TOTAL>LIMIT or rec['bytes_received']>max_bytes:
                            raise RuntimeError('Download byte limit exceeded')
                    digest.update(chunk);out.write(chunk)
            rec['sha256']=digest.hexdigest()
            if expected_sha and rec['sha256'] != expected_sha:
                raise RuntimeError('SHA256 mismatch')
            part.replace(dest)
            rec['complete']=True
    except Exception as ex:
        rec['error']=str(ex)
        raise
    finally:
        rec['elapsed_seconds']=time.time()-rec['started_unix']
        with LOCK:
            RECORDS.append(rec);save_records()
    return dest

def dump(relative, value):
    p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+'\n')

if __name__=='__main__':
    import concurrent.futures
    p=json.loads((ROOT/'sources/pairtally-commit.json').read_text())['sha']
    hf='06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1'
    h=json.loads((ROOT/'sources/countgdpp-hf-revision.json').read_text())
    jobs=[(f'https://api.github.com/repos/bbvisual/PairTally_Benchmark/git/trees/{p}?recursive=1','sources/pairtally-tree.json'),(f'https://huggingface.co/api/models/nikigoli/CountGDPlusPlus/tree/{hf}?recursive=true&expand=true','sources/countgdpp-hf-tree.json'),('https://api.github.com/repos/niki-amini-naieni/CountGDPlusPlus/commits/main','sources/countgdpp-github-commit.json')]
    jobs += [(f'https://huggingface.co/nikigoli/CountGDPlusPlus/resolve/{hf}/{x["rfilename"]}','sources/countgdpp-hf/'+x['rfilename']) for x in h['siblings'] if x['rfilename'].endswith(('.py','.json','.md','.txt')) or x['rfilename']=='LICENSE']
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fs={pool.submit(get,u,r):r for u,r in jobs}
        for f in concurrent.futures.as_completed(fs):
            try:f.result()
            except Exception as ex:errors.append({'path':fs[f],'error':str(ex)})
    print(json.dumps({'downloaded_jobs':len(jobs),'errors':errors,'bytes_total':TOTAL}))
