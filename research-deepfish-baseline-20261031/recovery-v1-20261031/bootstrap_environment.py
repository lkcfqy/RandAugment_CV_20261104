"""SHA-pinned Python/uv bootstrap and official-source dependency install, no model construction."""
import hashlib,json,os,pathlib,subprocess,sys,tarfile,time,urllib.request,zipfile
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,value):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');b=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
 with q.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(q,p);assert p.read_bytes()==b
def bootstrap(root,pins,requirements,deadline):
 root=pathlib.Path(root);root.mkdir(parents=True,exist_ok=True);log=root/'bootstrap.log';evidence=[]
 def run(args,name):
  with (root/(name+'.log')).open('wb') as f:
   subprocess.run(args,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time.time()),env=dict(os.environ,UV_PYTHON_DOWNLOADS='never',UV_NO_CONFIG='1',PIP_CONFIG_FILE=os.devnull))
 def download(row,name):
  p=root/name;h=hashlib.sha256();size=0
  with urllib.request.urlopen(row['url'],timeout=40) as src,p.open('xb') as f:
   while True:
    if time.time()>deadline:raise TimeoutError('Bootstrap deadline')
    b=src.read(1024**2)
    if not b:break
    size+=len(b)
    if size>100*1024**2:raise ValueError('Bootstrap asset unexpectedly exceeds 100 MiB')
    h.update(b);f.write(b)
   f.flush();os.fsync(f.fileno())
  if h.hexdigest()!=row['sha256'] or row.get('bytes',size)!=size:raise ValueError('Bootstrap SHA/size mismatch')
  evidence.append({'url':row['url'],'path':name,'bytes':size,'sha256':h.hexdigest()});return p
 wheel=download(pins['uv'],pins['uv']['filename'])
 with zipfile.ZipFile(wheel) as z:
  names=[n for n in z.namelist() if pathlib.PurePosixPath(n).name=='uv'];assert len(names)==1
  uv=root/'uv';uv.write_bytes(z.read(names[0]));uv.chmod(0o755)
 archive=download(pins['python'],'cpython-3.10.19.tar.gz')
 with tarfile.open(archive,'r:gz') as t:t.extractall(root/'python_dist',filter='data')
 candidate=root/'python_dist/python/bin/python3.10';version=subprocess.check_output([str(candidate),'-B','-c','import platform;print(platform.python_version())'],text=True,timeout=10).strip();assert version=='3.10.19'
 run([str(uv),'venv','--python',str(candidate),'--no-managed-python','--seed','--default-index','https://pypi.org/simple',str(root/'runtime_env')],'venv')
 py=root/'runtime_env/bin/python';req=root/'requirements.txt';req.write_text(requirements)
 # Only published PyPI/PyTorch/OpenMMLab sources, including build dependencies.
 run([str(py),'-m','pip','--isolated','install','--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cu117','--find-links','https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html','--only-binary=mmcv','--report',str(root/'install_report.json'),'-r',str(req)],'install')
 run([str(py),'-m','pip','check'],'pip_check');run([str(py),'-m','pip','freeze','--all'],'pip_freeze')
 result={'status':'ENVIRONMENT_INSTALLED','python_version':version,'python_executable':str(py),'interpreter_executable_sha256':sha(candidate),'uv_version':subprocess.check_output([str(uv),'--version'],text=True).strip(),'uv_executable_sha256':sha(uv),'downloads':evidence,'requirements_sha256':sha(req),'install_report_sha256':sha(root/'install_report.json'),'model_constructed':False}
 atomic_json(root/'bootstrap_result.json',result);return result
