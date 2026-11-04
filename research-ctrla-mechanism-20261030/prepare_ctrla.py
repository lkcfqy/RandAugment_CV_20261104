from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib, json, os, platform, subprocess, sys, tarfile, time, urllib.request
from datetime import datetime, timezone

ROOT=_portable_project_path(__file__, '')
OUT=ROOT/'research-ctrla-mechanism-20261030'
COMMIT='e77a4c19a3db893c120fc18b774fa290e826592b'
if OUT.exists():
    raise SystemExit('Target exists: inspect before continuing; no overwrite.')
OUT.mkdir()
for name in ('sources','data','weights','logs','raw','figures','tests'):
    (OUT/name).mkdir()
(OUT/'.gitignore').write_text('data/\nweights/\nlogs/\nraw/\nsources/\n__pycache__/\n*.pyc\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
baseline={p:sha(ROOT/p) for p in tracked if p and (ROOT/p).is_file()}
(OUT/'tracked-preservation-before.json').write_text(json.dumps(baseline,indent=2)+'\n')
(OUT/'repository-before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT).decode())
url=f'https://codeload.github.com/jbcdfm/ControlAugment/tar.gz/{COMMIT}'
archive=OUT/'sources'/'ControlAugment.tar.gz'
started=time.monotonic()
with urllib.request.urlopen(url,timeout=60) as response, archive.open('xb') as f:
    while chunk:=response.read(1024*1024): f.write(chunk)
with tarfile.open(archive) as tar:
    members=tar.getmembers()
    if any(m.issym() or m.islnk() or not (m.isfile() or m.isdir()) for m in members):
        raise RuntimeError('Unexpected source archive member')
    tar.extractall(OUT/'sources',filter='data')
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'source_url':url,'commit':COMMIT,'archive_sha256':sha(archive),'download_seconds':time.monotonic()-started,'source_files':{str(p.relative_to(OUT)):sha(p) for p in (OUT/'sources'/f'ControlAugment-{COMMIT}').rglob('*') if p.is_file()},'paper':'https://arxiv.org/html/2603.21819v2','paper_date':'2026-10-26','code_date':'2026-10-06','code_not_v2_exact_reproduction':True,'test_data_loaded':False,'colab_cu':0}
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_FAST_MATH']='0'
import torch, torchvision
receipt={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'platform':platform.platform(),'mps_available':torch.backends.mps.is_available(),'cpu_fallback':False}
if receipt['mps_available']:
    a=torch.arange(256,dtype=torch.float32).reshape(16,16)/128-1
    b=a.to('mps'); t=time.monotonic(); actual=b@b.T;torch.mps.synchronize()
    receipt.update(device=str(actual.device),seconds=time.monotonic()-t,max_abs_error=float((actual.cpu()-a@a.T).abs().max()))
receipt['hardware']=subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True)
# Avoid persisting serial identifiers in hardware report.
receipt['hardware']='\n'.join(line for line in receipt['hardware'].splitlines() if not any(s in line for s in ('Serial Number','Hardware UUID','Provisioning UDID')))
(OUT/'environment-preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'source_sha256':manifest['archive_sha256'],'mps':receipt},indent=2),flush=True)
