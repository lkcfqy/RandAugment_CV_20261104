"""Template executed by the official Colab CLI only after explicit local GO.

The controller replaces {'remote_name': 'cue_dc6ce7deb6a8', 'parts': [{'filename': 'part-000.bin', 'bytes': 8388608, 'sha256': '525a67049ace250b5828897f794dc7c7bb4c9be4e4d0e441b3ec7f7ef685c2b6'}, {'filename': 'part-001.bin', 'bytes': 8388608, 'sha256': 'ce5280271c6a07d11c00a8defd91b8453c34364cdc768f88c9edcef0ea374b84'}, {'filename': 'part-002.bin', 'bytes': 8388608, 'sha256': 'a98fc1b5471810340de1e65ca63db142054f96fa96bfc7acaf9603038b9ba6a4'}, {'filename': 'part-003.bin', 'bytes': 8388608, 'sha256': 'b4b63e8cc7dd6dc5024dfc404fd845e33f10de44f366f3e5294edf73a2c42a8c'}, {'filename': 'part-004.bin', 'bytes': 8388608, 'sha256': 'ab94d30c64df0e372bc734b2c4e31ee7513bfa9078c20e584ef0de723d88d9ec'}, {'filename': 'part-005.bin', 'bytes': 8388608, 'sha256': 'acc76b0bbd731741850bf340f7683371615e5500328c53de7aa8cb82fe079197'}, {'filename': 'part-006.bin', 'bytes': 8388608, 'sha256': '7f2f398f82cf971ac9384975f605d62afc684911ce05fd0c4c792e28f01f5289'}, {'filename': 'part-007.bin', 'bytes': 8388608, 'sha256': 'a9a23d1b58c092e8ba682eba834a356dcec87c44db665c1f16e34b3a0fc60e7d'}, {'filename': 'part-008.bin', 'bytes': 8388608, 'sha256': '43f014443b4bb0569087c21765183dd6a576b491f241e6a4668c2509b527ad14'}, {'filename': 'part-009.bin', 'bytes': 8388608, 'sha256': '0a4f63a70bb356d75fae1139f79a60e0161fee70de6a72432fb2ce3c4b64037b'}, {'filename': 'part-010.bin', 'bytes': 8388608, 'sha256': '38e76893a777392ac25986385a46ac0b41f1ccf3fc9e550d4c4fa8197c0f8fe4'}, {'filename': 'part-011.bin', 'bytes': 8388608, 'sha256': 'f7b39208b813b87aa629e94afddb710b3228d20a77e7b7f9d9e03032aa4c31c3'}, {'filename': 'part-012.bin', 'bytes': 208039, 'sha256': '246e9becd6c3d1ec5696ec6f577e7539911f6a74e78cebc3319170f584309c3b'}], 'bundle_sha256': '4512328d1a5ab63d2bb9465aeae1741bd6989ddd97d6683740619d023aec4efd', 'bundle_bytes': 100871335, 'payload_bytes': 104091365, 'package_manifest_sha256': '84ec75b4e478c0d4c36129b0e63810ef6bb5fe44d573d0df56c6b904eebe3442', 'allocation_seconds_left': 2630.7508413749892, 'local_prelaunch_compute_seconds': 0.10814029100583866, 'previous_compute_wall_seconds': 259.6864732820104} with frozen, JSON-compatible data.
"""
import time
SCRIPT_START=time.monotonic()
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath

def main(config):
    start=SCRIPT_START;root=Path('/content')/config['remote_name'];payload=root/'payload'
    payload.mkdir(parents=True,exist_ok=False)
    def check_deadlines():
        if time.monotonic()-start>=60:raise RuntimeError('60 s preparation compute cap reached during assembly')
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

if __name__=='__main__':main({'remote_name': 'cue_dc6ce7deb6a8', 'parts': [{'filename': 'part-000.bin', 'bytes': 8388608, 'sha256': '525a67049ace250b5828897f794dc7c7bb4c9be4e4d0e441b3ec7f7ef685c2b6'}, {'filename': 'part-001.bin', 'bytes': 8388608, 'sha256': 'ce5280271c6a07d11c00a8defd91b8453c34364cdc768f88c9edcef0ea374b84'}, {'filename': 'part-002.bin', 'bytes': 8388608, 'sha256': 'a98fc1b5471810340de1e65ca63db142054f96fa96bfc7acaf9603038b9ba6a4'}, {'filename': 'part-003.bin', 'bytes': 8388608, 'sha256': 'b4b63e8cc7dd6dc5024dfc404fd845e33f10de44f366f3e5294edf73a2c42a8c'}, {'filename': 'part-004.bin', 'bytes': 8388608, 'sha256': 'ab94d30c64df0e372bc734b2c4e31ee7513bfa9078c20e584ef0de723d88d9ec'}, {'filename': 'part-005.bin', 'bytes': 8388608, 'sha256': 'acc76b0bbd731741850bf340f7683371615e5500328c53de7aa8cb82fe079197'}, {'filename': 'part-006.bin', 'bytes': 8388608, 'sha256': '7f2f398f82cf971ac9384975f605d62afc684911ce05fd0c4c792e28f01f5289'}, {'filename': 'part-007.bin', 'bytes': 8388608, 'sha256': 'a9a23d1b58c092e8ba682eba834a356dcec87c44db665c1f16e34b3a0fc60e7d'}, {'filename': 'part-008.bin', 'bytes': 8388608, 'sha256': '43f014443b4bb0569087c21765183dd6a576b491f241e6a4668c2509b527ad14'}, {'filename': 'part-009.bin', 'bytes': 8388608, 'sha256': '0a4f63a70bb356d75fae1139f79a60e0161fee70de6a72432fb2ce3c4b64037b'}, {'filename': 'part-010.bin', 'bytes': 8388608, 'sha256': '38e76893a777392ac25986385a46ac0b41f1ccf3fc9e550d4c4fa8197c0f8fe4'}, {'filename': 'part-011.bin', 'bytes': 8388608, 'sha256': 'f7b39208b813b87aa629e94afddb710b3228d20a77e7b7f9d9e03032aa4c31c3'}, {'filename': 'part-012.bin', 'bytes': 208039, 'sha256': '246e9becd6c3d1ec5696ec6f577e7539911f6a74e78cebc3319170f584309c3b'}], 'bundle_sha256': '4512328d1a5ab63d2bb9465aeae1741bd6989ddd97d6683740619d023aec4efd', 'bundle_bytes': 100871335, 'payload_bytes': 104091365, 'package_manifest_sha256': '84ec75b4e478c0d4c36129b0e63810ef6bb5fe44d573d0df56c6b904eebe3442', 'allocation_seconds_left': 2630.7508413749892, 'local_prelaunch_compute_seconds': 0.10814029100583866, 'previous_compute_wall_seconds': 259.6864732820104})
