import hashlib,json,os,time,datetime,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
def utc():return datetime.datetime.now(datetime.UTC).isoformat()
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n');tmp.replace(path)
def sha(path,algo='sha256'):
    h=hashlib.new(algo)
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
class Stop(RuntimeError):pass
class Budget:
    def __init__(self):
        self.path=ROOT/'ledger.json'
        self.s=json.loads(self.path.read_text()) if self.path.exists() else {'phase':'prepared','counts':{},'times':{},'peaks':{},'started':False,'preflight_passed':False,'complete_fits':0}
    def flush(self):self.s['updated_utc']=utc();save(self.path,self.s)
    def start(self):
        if not self.s['started']:
            self.s.update(started=True,start_utc=utc(),start_epoch=time.time(),start_mono=time.monotonic());self.flush()
    def elapsed(self):return max(time.time()-self.s['start_epoch'],time.monotonic()-self.s['start_mono']) if self.s['started'] else 0.
    def check(self,torch=None,disk=False):
        if self.s['started']:
            elapsed=self.elapsed()
            if elapsed>=P['limits']['wall_seconds']:raise Stop('four_hour_wall_limit')
            if not self.s['preflight_passed'] and elapsed>=P['limits']['preflight_seconds']:raise Stop('45_minute_preflight_limit')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        driver=torch.mps.driver_allocated_memory() if torch is not None and torch.backends.mps.is_available() else 0
        for name,val in [('rss_peak_bytes',rss),('mps_driver_bytes',driver),('rss_peak_plus_current_driver_bytes',rss+driver)]:self.s['peaks'][name]=max(self.s['peaks'].get(name,0),val)
        if rss+driver>=P['limits']['rss_plus_mps_driver_bytes']:raise Stop('24GiB_conservative_memory_limit')
        if disk:
            used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            self.s['peaks']['disk_bytes']=max(used,self.s['peaks'].get('disk_bytes',0))
            if used>=P['limits']['disk_bytes']:raise Stop('8GiB_new_disk_limit')
    def reserve(self,key,n):
        value=self.s['counts'].get(key,0)+n
        cap=P['limits'].get(key)
        if cap is not None and value>cap:raise Stop(key+'_limit')
        self.s['counts'][key]=value;self.flush();self.check()
    def timing(self,key,seconds):self.s['times'][key]=self.s['times'].get(key,0.)+seconds
    def phase(self,name):self.s['phase']=name;self.flush();print(json.dumps({'phase':name,'utc':utc(),'elapsed_s':self.elapsed(),'counts':self.s['counts']}),flush=True)
    def event(self,name,**data):
        row={'event':name,'utc':utc(),'elapsed_s':self.elapsed(),**data}
        with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
        print(json.dumps(row,default=str),flush=True)
