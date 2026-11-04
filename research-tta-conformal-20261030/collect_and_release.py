"""Collect only this experiment and release its T4 as soon as logits finish."""
import hashlib,json,subprocess,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent
PY='/Users/lkc/.local/share/google-colab-cli/bin/python'
start=json.loads((R/'cli-ledger/allocate-t4.json').read_text())['started_epoch']

def call(name,args):
    p=subprocess.run([PY,str(R/'cli_record.py'),name,*args],capture_output=True,text=True,timeout=190)
    receipt=R/'cli-ledger'/(name+'.json')
    if not receipt.exists():raise RuntimeError('Missing CLI receipt '+name)
    r=json.loads(receipt.read_text())
    if p.returncode or r['exit_code']:
        print(json.dumps({'step':name,'failed':True,'receipt':str(receipt)}),flush=True)
        raise RuntimeError('CLI step failed '+name)
    return r

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

index=0
(R/'monitor').mkdir(exist_ok=True)
while True:
    if time.time()-start>110*60:raise RuntimeError('Collection waiting deadline; local watchdog will release at115min')
    logfile=R/'monitor'/f'remote-process-{index:03d}.log'
    r=call(f'collector-log-{index:03d}',['download','--session','tta-conformal-20261030',
               '/content/tta_conformal_20261030/process.log',str(logfile)])
    parsed={'last_event':None,'bundle':None}
    for line in logfile.read_text().splitlines():
        try:
            obj=json.loads(line)
            if 'event' in obj:parsed['last_event']=obj
            if 'BUNDLE_READY' in obj:parsed['bundle']=obj
        except json.JSONDecodeError:pass
    if parsed['last_event'] is None:raise RuntimeError('No parseable collector status')
    print(json.dumps(parsed),flush=True)
    if parsed['bundle']:
        b=parsed['bundle'];local=R/'results.tar'
        call('download-results',['download','--session','tta-conformal-20261030',b['BUNDLE_READY'],str(local)])
        assert local.stat().st_size==b['bytes'] and sha(local)==b['sha256'],'Downloaded bundle integrity mismatch'
        # Bundle bytes verified before release; CPU extraction/analysis does not hold GPU.
        call('stop-complete',['stop','--session','tta-conformal-20261030'])
        sessions=call('sessions-after-stop',['sessions'])
        usage=call('usage-after-stop',['usage'])
        assert 'tta-conformal-20261030' not in sessions['stdout'],'Candidate session still active'
        rec={'released_epoch':time.time(),'sessions_output':sessions['stdout'],'usage_output':usage['stdout'],
             'bundle_sha256':b['sha256'],'bundle_bytes':b['bytes'],'run_result':b['result'],
             'release_confirmed':True,'gpu_allocation_elapsed_seconds':time.time()-start}
        (R/'released.json').write_text(json.dumps(rec,indent=2)+'\n')
        out=R/'results';out.mkdir(exist_ok=False)
        with tarfile.open(local,'r') as tf:
            members=tf.getmembers()
            assert all(m.isfile() and '/' not in m.name and '\\' not in m.name and m.name not in ('.','..') for m in members)
            for member in members:
                with tf.extractfile(member) as source,(out/member.name).open('wb') as dest:
                    while chunk:=source.read(1024*1024):dest.write(chunk)
        hashes=json.loads((out/'artifact-hashes.json').read_text())
        assert all(sha(out/name)==digest for name,digest in hashes.items()),'Artifact SHA256 verification failed'
        (R/'collection-verified.json').write_text(json.dumps({'verified':True,'file_count':len(hashes),**rec},indent=2)+'\n')
        print(json.dumps({'COLLECTED_AND_RELEASED':True,**rec}),flush=True)
        break
    index+=1
    if index%10==0:
        usage=call(f'usage-during-{index:03d}',['usage'])
        print(json.dumps({'usage':usage['stdout']}),flush=True)
    time.sleep(30)
