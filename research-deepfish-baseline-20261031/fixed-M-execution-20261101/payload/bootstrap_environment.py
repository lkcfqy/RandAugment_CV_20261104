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
   subprocess.run(args,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time.time()),env=dict(os.environ,UV_PYTHON_DOWNLOADS='never',UV_NO_CONFIG='1',MPLBACKEND='Agg',PIP_CONFIG_FILE=os.devnull))
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
 run([str(py),'-m','pip','--isolated','install','--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cu118','--find-links','https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html','--only-binary=mmcv','--report',str(root/'install_report.json'),'-r',str(req)],'install')
 run([str(py),'-B','-c','from pkg_resources import packaging; from torch.utils.cpp_extension import CUDA_HOME; print(\"TORCH_CPP_EXTENSION_IMPORT_PASS\")'],'torch_cpp_extension_gate');
 report=json.loads((root/'install_report.json').read_text());setuptools=[x for x in report['install'] if x['metadata']['name'].lower()=='setuptools'];assert len(setuptools)==1 and setuptools[0]['metadata']['version']=='69.5.1' and setuptools[0]['download_info']['archive_info']['hashes']['sha256']=='c636ac361bc47580504644275c9ad802c50415c7522212252c033bd15f301f32'

 core_wheel_hashes={'torch': 'a81b554184492005543ddc32e96469f9369d778dedd195d73bda9bed407d6589', 'torchvision': '033712f65d45afe806676c4129dfe601ad1321d9e092df62b15847c02d4061dc', 'mmcv': '82a3eb54106f643dace1f9bd4d96e96899a0bb2360a5bb8e2f252a36d95051d9', 'mmengine': '262ac976a925562f78cd5fd14dd1bc9b680ed0aa81f0d85b723ef782f99c54ee', 'mmdet': '2e23e291281ac57e7dccf8678e957da45fbe560ce78a1f5ded6afeccd3730f17'}
 observed={x['metadata']['name'].lower():x['download_info']['archive_info'].get('hashes',{}).get('sha256') for x in report['install']}
 assert all(observed.get(k)==v for k,v in core_wheel_hashes.items()),'Core wheel differs from CPU-verified official artifact'
 historical=json.loads(pathlib.Path(__file__).with_name('historical_install_artifacts.json').read_text())['artifacts']
 assert all(observed.get(x['name'].lower())==x['sha256'] for x in historical if x['sha256'] is not None),'Known historical runtime download artifact differs'
 run([str(py),'-m','pip','check'],'pip_check');run([str(py),'-m','pip','freeze','--all'],'pip_freeze')
 run([str(py),'-B','-c',"import json,torch\nfrom mmcv.ops import nms\nassert torch.__version__=='2.1.0+cu118' and torch.version.cuda=='11.8'\ntorch.use_deterministic_algorithms(True)\ndevices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])\nfor device in devices:\n for indices in ([],[64],[0,64,899]):\n  x=torch.zeros((900,4),dtype=torch.float32,device=device);idx=torch.tensor(indices,dtype=torch.int64,device=device);x[idx]=1.0\n  expected=torch.zeros((900,4),dtype=torch.float32)\n  for i in indices:expected[i].fill_(1.0)\n  assert torch.equal(x.cpu(),expected)\n keep=nms(torch.tensor([[0.,0.,5.,5.],[0.,0.,5.,5.]],device=device),torch.tensor([.9,.8],device=device),.5)[1].cpu().tolist();assert keep==[0]\nassert torch.are_deterministic_algorithms_enabled()\nprint(json.dumps({'status':'PASS','devices':devices,'torch':torch.__version__,'deterministic_scalar_index_assignment':True,'compiled_nms':True,'model_instantiated':False}))\n"],'primitive_gate')
 result={'status':'ENVIRONMENT_INSTALLED','python_version':version,'python_executable':str(py),'interpreter_executable_sha256':sha(candidate),'uv_version':subprocess.check_output([str(uv),'--version'],text=True).strip(),'uv_executable_sha256':sha(uv),'downloads':evidence,'requirements_sha256':sha(req),'install_report_sha256':sha(root/'install_report.json'),'primitive_gate':json.loads((root/'primitive_gate.log').read_text().splitlines()[-1]),'model_constructed':False}
 atomic_json(root/'bootstrap_result.json',result);return result
