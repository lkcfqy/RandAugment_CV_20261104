"""Template executed by the official Colab CLI only after explicit local GO.

The controller replaces {'remote_name': 'cue_915d27011f9f', 'parts': [{'filename': 'part-000.bin', 'bytes': 8388608, 'sha256': 'b295e26e3dbf862552029127e4c5c8bfa6256ad5a1c371e2d4907d2f38304e0a'}, {'filename': 'part-001.bin', 'bytes': 8388608, 'sha256': '309531dffd2e32478ce1d203249de52ec48a114bf4a3b1f658562a4cf23fbc23'}, {'filename': 'part-002.bin', 'bytes': 8388608, 'sha256': '42e53b8bdf4fc1dfa7a4ed923c9a4df26cf573a642bf243fa34c8702e4048a7b'}, {'filename': 'part-003.bin', 'bytes': 8388608, 'sha256': '98582ccd863d3941964bdf28cbb9e34ed9772748c8a141625fc2b264689eb288'}, {'filename': 'part-004.bin', 'bytes': 8388608, 'sha256': 'be61506c9f89e4b99ed6b6eb87fc574fbeb4c5bb0d05ba5e7c074e465bb5125b'}, {'filename': 'part-005.bin', 'bytes': 8388608, 'sha256': '4ddde0ba98fc8cefec26975dffab6e7a62366a7142ae8182063f7c71e43a5847'}, {'filename': 'part-006.bin', 'bytes': 8388608, 'sha256': '5574af689c10921a7aa33576386882739078c1847de686dbee0c10ea9a5c5285'}, {'filename': 'part-007.bin', 'bytes': 8388608, 'sha256': '0e2a0111959a9664d1ebf01e52bcf5e75e159fbb89475d8f6b108b71fc06e905'}, {'filename': 'part-008.bin', 'bytes': 8388608, 'sha256': '05aef98ed0721736bffa39bbb9e2278da2a6e59fa8938540523f71aa32744ea1'}, {'filename': 'part-009.bin', 'bytes': 8388608, 'sha256': '5cd730e769dc7590c7b71e53407a8cff100d3030957970d4658094105ca5c23c'}, {'filename': 'part-010.bin', 'bytes': 8388608, 'sha256': 'a1dc39bff6cfe696a778c7360f862ddf62907372df4ee6c273ed78d7f6df34aa'}, {'filename': 'part-011.bin', 'bytes': 8388608, 'sha256': '49cd02fae39cb8b246f31a9629cc1b572a0ceffa7f84117fb7f56282a783ab7b'}, {'filename': 'part-012.bin', 'bytes': 206353, 'sha256': 'b33eccae13fc761bfa38a7b6dc7c71255d532bb872eee9602bee205761f998c3'}], 'bundle_sha256': 'eba33cd408517c8ef04dcd91311f2da4b65f513aa2f98e7030757694c169907e', 'bundle_bytes': 100869649, 'payload_bytes': 104086344, 'package_manifest_sha256': '08fa6c3b9621b231d6a72f27982833375b2e9ee5c37de2e3fce30760f28faf08', 'allocation_seconds_left': 2646.5999492499977, 'local_prelaunch_compute_seconds': 0.11172545800218359, 'previous_compute_wall_seconds': 54.91578541700437} with frozen, JSON-compatible data.
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

if __name__=='__main__':main({'remote_name': 'cue_915d27011f9f', 'parts': [{'filename': 'part-000.bin', 'bytes': 8388608, 'sha256': 'b295e26e3dbf862552029127e4c5c8bfa6256ad5a1c371e2d4907d2f38304e0a'}, {'filename': 'part-001.bin', 'bytes': 8388608, 'sha256': '309531dffd2e32478ce1d203249de52ec48a114bf4a3b1f658562a4cf23fbc23'}, {'filename': 'part-002.bin', 'bytes': 8388608, 'sha256': '42e53b8bdf4fc1dfa7a4ed923c9a4df26cf573a642bf243fa34c8702e4048a7b'}, {'filename': 'part-003.bin', 'bytes': 8388608, 'sha256': '98582ccd863d3941964bdf28cbb9e34ed9772748c8a141625fc2b264689eb288'}, {'filename': 'part-004.bin', 'bytes': 8388608, 'sha256': 'be61506c9f89e4b99ed6b6eb87fc574fbeb4c5bb0d05ba5e7c074e465bb5125b'}, {'filename': 'part-005.bin', 'bytes': 8388608, 'sha256': '4ddde0ba98fc8cefec26975dffab6e7a62366a7142ae8182063f7c71e43a5847'}, {'filename': 'part-006.bin', 'bytes': 8388608, 'sha256': '5574af689c10921a7aa33576386882739078c1847de686dbee0c10ea9a5c5285'}, {'filename': 'part-007.bin', 'bytes': 8388608, 'sha256': '0e2a0111959a9664d1ebf01e52bcf5e75e159fbb89475d8f6b108b71fc06e905'}, {'filename': 'part-008.bin', 'bytes': 8388608, 'sha256': '05aef98ed0721736bffa39bbb9e2278da2a6e59fa8938540523f71aa32744ea1'}, {'filename': 'part-009.bin', 'bytes': 8388608, 'sha256': '5cd730e769dc7590c7b71e53407a8cff100d3030957970d4658094105ca5c23c'}, {'filename': 'part-010.bin', 'bytes': 8388608, 'sha256': 'a1dc39bff6cfe696a778c7360f862ddf62907372df4ee6c273ed78d7f6df34aa'}, {'filename': 'part-011.bin', 'bytes': 8388608, 'sha256': '49cd02fae39cb8b246f31a9629cc1b572a0ceffa7f84117fb7f56282a783ab7b'}, {'filename': 'part-012.bin', 'bytes': 206353, 'sha256': 'b33eccae13fc761bfa38a7b6dc7c71255d532bb872eee9602bee205761f998c3'}], 'bundle_sha256': 'eba33cd408517c8ef04dcd91311f2da4b65f513aa2f98e7030757694c169907e', 'bundle_bytes': 100869649, 'payload_bytes': 104086344, 'package_manifest_sha256': '08fa6c3b9621b231d6a72f27982833375b2e9ee5c37de2e3fce30760f28faf08', 'allocation_seconds_left': 2646.5999492499977, 'local_prelaunch_compute_seconds': 0.11172545800218359, 'previous_compute_wall_seconds': 54.91578541700437})
