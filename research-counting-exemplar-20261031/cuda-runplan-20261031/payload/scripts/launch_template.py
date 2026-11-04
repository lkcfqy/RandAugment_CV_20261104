"""Template executed by the official Colab CLI only after explicit local GO.

The controller replaces __CUE_LAUNCH_CONFIG__ with frozen, JSON-compatible data.
"""
import time
SCRIPT_START=time.monotonic()
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath

def main(config):
    start=SCRIPT_START;root=Path('/content')/config['remote_name'];payload=root/'payload'
    payload.mkdir(parents=True,exist_ok=False)
    def check_deadlines():
        used=config['previous_compute_wall_seconds']+config['local_prelaunch_compute_seconds']+time.monotonic()-start
        if used>=1740:raise RuntimeError('assembly/import compute deadline reached with finalization reserve')
        if config['allocation_seconds_left']-(time.monotonic()-start)<=90:raise RuntimeError('assembly allocation release reserve reached')
    archive=root/'payload.tar.gz';h=hashlib.sha256();total=0
    with archive.open('xb') as output:
        for part in config['parts']:
            check_deadlines()
            path=root/part['filename'];data=path.read_bytes()
            if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:raise RuntimeError('uploaded package part hash mismatch')
            h.update(data);total+=len(data);output.write(data)
        output.flush();os.fsync(output.fileno())
    if h.hexdigest()!=config['bundle_sha256'] or total!=config['bundle_bytes']:raise RuntimeError('uploaded package bundle hash mismatch')
    extracted=0
    with tarfile.open(archive,'r:gz') as bundle:
        for member in bundle:
            check_deadlines()
            rel=PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts or '\\' in member.name:raise RuntimeError('unexpected package member')
            extracted+=member.size
            if extracted>config['payload_bytes'] or extracted>6*1024**3:raise RuntimeError('package extraction exceeds frozen bound')
            destination=payload/str(rel);destination.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,destination.open('xb') as target:
                while data:=source.read(1024**2):target.write(data)
    manifest=json.loads((payload/'package_files.json').read_text())
    if hashlib.sha256((payload/'package_files.json').read_bytes()).hexdigest()!=config['package_manifest_sha256']:raise RuntimeError('package file manifest differs from freeze')
    for rel,entry in manifest['files'].items():
        check_deadlines()
        path=payload/rel
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if path.stat().st_size!=entry['bytes'] or actual!=entry['sha256']:raise RuntimeError('package file identity mismatch: '+rel)
    assembly=time.monotonic()-start;left=config['allocation_seconds_left']-assembly
    if left<=60:raise RuntimeError('insufficient allocation wall after package assembly')
    expiry=subprocess.Popen([sys.executable,str(payload/'scripts/runtime_expiry.py'),'--deadline',str(time.monotonic()+left),'--state',str(payload/'runtime_expiry_state.json')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    environment=os.environ.copy();environment.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    # Reader in the notebook streams a child synchronously; no detached background job.
    process=subprocess.Popen([sys.executable,'-u',str(payload/'scripts/remote_runner.py'),'--allocation-seconds-left',str(left),'--assembly-compute-seconds',str(assembly+config['local_prelaunch_compute_seconds'])],cwd=payload,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:print(line,end='',flush=True)
        result=process.wait()
    finally:process.stdout.close()
    print('CUE_REMOTE_EXIT='+str(result),flush=True)

if __name__=='__main__':main(__CUE_LAUNCH_CONFIG__)
