"""Bounded project-only official binary environment; never instantiate a model."""
from portable_dev.paths import project_path as _portable_project_path
import pathlib,json,hashlib,urllib.request,tarfile,subprocess,os,time,shutil,datetime
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
P=R/'cpu-cross-input-preflight-20261101'; P.mkdir(exist_ok=True)
for n in ['logs','receipts','downloads','wheels','tmp','.cache']:(P/n).mkdir(exist_ok=True)
CAP=6*1024**3;STOP_BYTES=CAP-80*1024**2;start=time.monotonic()
def dump(n,d):(P/'receipts'/n).write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def disk():return sum(x.stat().st_size for x in R.rglob('*') if x.is_file())
def checkdisk():
 b=disk()
 if b>STOP_BYTES:raise RuntimeError(f'research disk guard reached: {b}')
 return b
E=os.environ.copy()
for k in list(E):
 if k.startswith('PIP_'):E.pop(k,None)
E.update(PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_CACHE_DIR='1',PIP_INDEX_URL='https://pypi.org/simple',XDG_CACHE_HOME=str(P/'.cache'),TMPDIR=str(P/'tmp'),HF_HOME=str(P/'.cache/huggingface'),MPLCONFIGDIR=str(P/'.cache/matplotlib'),BUILD_CUDA_EXTENSIONS='0',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',TOKENIZERS_PARALLELISM='false')
def run(label,args,timeout=600):
 before=time.monotonic();log=P/'logs'/f'{label}.log'
 with log.open('w') as f:
  proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=E,cwd=P)
  reason=None
  try:
   while proc.poll() is None:
    if time.monotonic()-before>timeout:reason='stage timeout';raise RuntimeError(reason)
    if time.monotonic()-start>1200:reason='preparation1200sec limit';raise RuntimeError(reason)
    checkdisk();time.sleep(2)
  except BaseException:
   proc.terminate()
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   dump(label+'.json',{'status':'STOPPED','reason':reason,'returncode':proc.returncode,'seconds':time.monotonic()-before});raise
 rec={'status':'PASS' if proc.returncode==0 else 'FAILED','returncode':proc.returncode,'seconds':time.monotonic()-before,'research_bytes':checkdisk(),'command':args}
 dump(label+'.json',rec);print(json.dumps({'stage':label,**{k:rec[k] for k in ['status','seconds','research_bytes']}}),flush=True)
 if proc.returncode:raise RuntimeError(label+' failed; see local log; no automatic native build/fork fallback')
def download(url,path,max_bytes,expected=None):
 if path.exists():
  if expected and sha(path)==expected:return
  raise RuntimeError('Refuse overwrite existing unverified download '+str(path))
 tmp=path.with_suffix(path.suffix+'.partial');h=hashlib.sha256();size=0
 with urllib.request.urlopen(url,timeout=45) as response,tmp.open('wb') as f:
  for b in iter(lambda:response.read(1024**2),b''):
   size+=len(b)
   if size>max_bytes:raise RuntimeError('bounded download exceeded')
   f.write(b);h.update(b)
 if expected and h.hexdigest()!=expected:raise RuntimeError('downloadSHA mismatch')
 os.replace(tmp,path);return {'url':url,'bytes':size,'sha256':h.hexdigest()}
try:
 before=checkdisk();catalog=P/'downloads/uv0.9.5-python-catalog.json'
 download('https://raw.githubusercontent.com/astral-sh/uv/0.9.5/crates/uv-python/download-metadata.json',catalog,5*1024**2,'fe7014cebbe034c8cbf17c79cda78872774378ff66cad78077f27d289e6e7862')
 data=json.loads(catalog.read_text());entry=data['cpython-3.11.14-darwin-aarch64-none'];assert entry['os']=='darwin' and entry['arch']['family']=='aarch64'
 assert entry['url'].startswith('https://github.com/astral-sh/python-build-standalone/releases/download/20251014/')
 archive=P/'downloads/cpython3.11.14-arm64.tar.gz';info=download(entry['url'],archive,80*1024**2,entry['sha256'])
 pyhome=P/'.python';assert not pyhome.exists(),'Refuse duplicate bootstrap';pyhome.mkdir()
 with tarfile.open(archive,'r:gz') as tf:
  members=tf.getmembers();assert sum(m.size for m in members)<500*1024**2
  for m in members:
   pure=pathlib.PurePosixPath(m.name);assert not pure.is_absolute() and '..' not in pure.parts and pure.parts[0]=='python'
  tf.extractall(pyhome,filter='data')
 py=pyhome/'python/bin/python3.11'
 dump('python_source.json',{'entry':entry,'catalog_sha256':sha(catalog),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'install_path':str(pyhome),'source':'Astral official python-build-standalone pinned UV0.9.5 catalogue','system_environment_modified':False})
 run('python_identity',[str(py),'-B','-c','import sys,platform;assert sys.version_info[:3]==(3,11,14);assert platform.machine()=="arm64";print(sys.version);print(platform.machine())'],30)
 run('create_venv',[str(py),'-B','-m','venv',str(P/'.venv')],90)
 vp=P/'.venv/bin/python'
 run('build_tools',[str(vp),'-B','-m','pip','install','--index-url','https://pypi.org/simple','--no-cache-dir','--only-binary=:all:','--report',str(P/'receipts/build_tools_install.json'),'pip==26.2.2','setuptools==69.5.1','wheel==0.48.0','ninja==1.11.1.1'],300)
 wheel_src=R/'cpu-feasibility-20261101/sources/mmcv-2.1.0-cp311-universal2.inspection-only.whl.bin';assert sha(wheel_src)=='6bc8018e724dc53dbea36fe4db5777da25cd0b67a912b2e802d2cba7b4ce3894'
 wheel=P/'wheels/mmcv-2.1.0-cp311-cp311-macosx_10_9_universal2.whl';shutil.copy2(wheel_src,wheel)
 old=R/'cpu-feasibility-20261101/requirements_mac_cpu_candidate.txt';lines=[]
 for line in old.read_text().splitlines():
  if line.startswith('#'):continue
  if line.startswith('mmcv @ '):line='mmcv @ '+wheel.as_uri()+'#sha256='+sha(wheel)
  lines.append(line)
 lines.append('ninja==1.11.1.1');req=P/'requirements_installed_candidate.txt';req.write_text('\n'.join(lines)+'\n')
 run('core_dependencies',[str(vp),'-B','-m','pip','install','--index-url','https://pypi.org/simple','--no-cache-dir','--only-binary=:all:','--no-binary=fairscale','--no-build-isolation','--report',str(P/'receipts/core_install.json'),'-r',str(req)],900)
 run('pip_check',[str(vp),'-B','-m','pip','check'],60)
 run('pip_freeze',[str(vp),'-B','-m','pip','freeze','--all'],60)
 # Verify official implementation and CPU extension import only; no model/operator calls.
 code='''import torch,torchvision,mmcv,mmengine,mmdet,platform,json,pathlib,hashlib
from mmcv.ops import MultiScaleDeformableAttention
assert torch.__version__.split('+')[0]=='2.1.0' and torchvision.__version__.split('+')[0]=='0.16.0'
assert mmcv.__version__=='2.1.0' and mmengine.__version__=='0.10.7' and mmdet.__version__=='3.3.0'
assert torch.version.cuda is None and platform.machine()=='arm64'
pins=json.loads(pathlib.Path(PINS).read_text());site=pathlib.Path(mmdet.__file__).parent.parent
for name,expected in pins['file_sha256'].items():assert hashlib.sha256((site/name).read_bytes()).hexdigest()==expected,name
print(json.dumps(dict(torch=torch.__version__,torchvision=torchvision.__version__,mmcv=mmcv.__version__,mmengine=mmengine.__version__,mmdet=mmdet.__version__,arch=platform.machine(),cuda_build=torch.version.cuda,official_file_pins_verified=True,model_instances=0,operator_calls=0)))
'''.replace('PINS',repr(str(R/'repeat-seed20261101-budget25-20261101/payload/implementation_pins.json')))
 run('official_import_gate',[str(vp),'-B','-c',code],90)
 dump('ENVIRONMENT_TERMINAL.json',{'status':'PREPARED_IMPORT_GATE_PASS','seconds':time.monotonic()-start,'research_bytes_before':before,'research_bytes_after':checkdisk(),'native_operator_build':False,'fairscale_cuda_extensions':False,'model_instances':0,'GPU_MPS_Colab_actions':0,'uploads':0})
 print('ENVIRONMENT_PREPARED_NO_MODEL_RUN',flush=True)
except BaseException as e:
 dump('ENVIRONMENT_TERMINAL.json',{'status':'STOPPED','reason':type(e).__name__+': '+str(e),'seconds':time.monotonic()-start,'research_bytes':disk(),'model_instances':0,'GPU_MPS_Colab_actions':0,'uploads':0,'no_source_patch_or_fork_retry':True})
 raise
