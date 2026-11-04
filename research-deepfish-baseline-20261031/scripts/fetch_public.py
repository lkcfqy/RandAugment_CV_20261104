"""Bounded anonymous HTTPS fetch; does not execute downloaded content."""
import argparse, datetime, hashlib, json, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'reports/download_ledger.jsonl'
TOTAL_LIMIT = 2 * 1024**3
DISK_LIMIT = 4 * 1024**3

def used_downloads():
    return sum(json.loads(s).get('bytes_received', 0) for s in LEDGER.read_text().splitlines()) if LEDGER.exists() else 0

def fetch(url, rel, cap, timeout=120, head=False):
    if not url.startswith('https://'): raise ValueError('HTTPS only')
    dst=ROOT / rel
    if ROOT not in dst.resolve().parents: raise ValueError('Outside task directory')
    if dst.exists(): raise FileExistsError(dst)
    used=used_downloads()
    cap=min(cap,TOTAL_LIMIT-used)
    disk=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if cap <= 0 or disk+cap > DISK_LIMIT: raise RuntimeError('Budget exceeded')
    dst.parent.mkdir(parents=True,exist_ok=True)
    partial=dst.with_name(dst.name+'.partial')
    if partial.exists(): raise FileExistsError(partial)
    headers=dst.with_name(dst.name+'.headers')
    cmd=['/usr/bin/curl','--disable','--location','--fail','--silent','--show-error','--proto','=https','--proto-redir','=https','--connect-timeout','30','--max-time',str(timeout),'--max-filesize',str(cap),'--dump-header',str(headers),'--output',str(partial)]
    if head: cmd+=['--head']
    cmd += [url]
    result=subprocess.run(cmd, text=True, capture_output=True)
    size=partial.stat().st_size if partial.exists() else 0
    entry=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),url=url,path=rel,bytes_received=size,returncode=result.returncode,stderr=result.stderr,cap=cap,head=head)
    if partial.exists():
        entry['sha256']=hashlib.sha256(partial.read_bytes()).hexdigest() if size<16*1024**2 else filehash(partial)
    with LEDGER.open('a') as f: f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    if result.returncode: print(json.dumps(entry)); raise RuntimeError(result.stderr)
    partial.rename(dst)
    print(json.dumps(entry),flush=True)
    return dst

def filehash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('path');p.add_argument('--cap',type=int,required=True);p.add_argument('--timeout',type=int,default=120);p.add_argument('--head',action='store_true');a=p.parse_args();fetch(a.url,a.path,a.cap,a.timeout,a.head)
