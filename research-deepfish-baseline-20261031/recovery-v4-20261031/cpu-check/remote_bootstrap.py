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
 run([str(py),'-m','pip','check'],'pip_check');run([str(py),'-m','pip','freeze','--all'],'pip_freeze')
 run([str(py),'-B','-c',"import json,torch\nfrom mmcv.ops import nms\nassert torch.__version__=='2.1.0+cu118' and torch.version.cuda=='11.8'\ntorch.use_deterministic_algorithms(True)\ndevices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])\nfor device in devices:\n for indices in ([],[64],[0,64,899]):\n  x=torch.zeros((900,4),dtype=torch.float32,device=device);idx=torch.tensor(indices,dtype=torch.int64,device=device);x[idx]=1.0\n  expected=torch.zeros((900,4),dtype=torch.float32)\n  for i in indices:expected[i].fill_(1.0)\n  assert torch.equal(x.cpu(),expected)\n keep=nms(torch.tensor([[0.,0.,5.,5.],[0.,0.,5.,5.]],device=device),torch.tensor([.9,.8],device=device),.5)[1].cpu().tolist();assert keep==[0]\nassert torch.are_deterministic_algorithms_enabled()\nprint(json.dumps({'status':'PASS','devices':devices,'torch':torch.__version__,'deterministic_scalar_index_assignment':True,'compiled_nms':True,'model_instantiated':False}))\n"],'primitive_gate')
 result={'status':'ENVIRONMENT_INSTALLED','python_version':version,'python_executable':str(py),'interpreter_executable_sha256':sha(candidate),'uv_version':subprocess.check_output([str(uv),'--version'],text=True).strip(),'uv_executable_sha256':sha(uv),'downloads':evidence,'requirements_sha256':sha(req),'install_report_sha256':sha(root/'install_report.json'),'model_constructed':False}
 atomic_json(root/'bootstrap_result.json',result);return result

PINS={'uv': {'url': 'https://files.pythonhosted.org/packages/9b/83/a0bdf4abf86ede79b427778fe27e2b4a022c98a7a8ea1745dcd6c6561f17/uv-0.9.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 'bytes': 21332544, 'sha256': '6507bbbcd788553ec4ad5a96fa19364dc0f58b023e31d79868773559a83ec181', 'filename': 'uv-0.9.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'}, 'python': {'name': 'cpython', 'arch': {'family': 'x86_64', 'variant': None}, 'os': 'linux', 'libc': 'gnu', 'major': 3, 'minor': 10, 'patch': 19, 'prerelease': '', 'url': 'https://github.com/astral-sh/python-build-standalone/releases/download/20251014/cpython-3.10.19%2B20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz', 'sha256': 'c0b09b744293f2aad85b1ef84544f8a7ba383675b29d1f7efd1e96bb9984399f', 'variant': None, 'build': '20251014'}, 'uv_version': '0.9.5', 'python_catalog_url': 'https://raw.githubusercontent.com/astral-sh/uv/0.9.5/crates/uv-python/download-metadata.json', 'python_catalog_sha256': 'fe7014cebbe034c8cbf17c79cda78872774378ff66cad78077f27d289e6e7862'}
REQUIREMENTS='# Single future Linux x86_64 CPython 3.10 candidate; no local installation.\n# Driver supplies only PyPI, official PyTorch cu118 and OpenMMLab sources.\ntorch==2.1.0+cu118\ntorchvision==0.16.0+cu118\nmmcv==2.1.0\nmmengine==0.10.7\nmmdet==3.3.0\ntransformers==4.34.0\ntokenizers==0.14.1\nhuggingface-hub==0.17.3\nsafetensors==0.5.3\nfairscale==0.4.13\nnumpy==1.24.3\nPillow==11.1.0\nopencv-python==4.11.0.86\nscipy==1.13.1\npycocotools==2.0.8\nshapely==2.0.7\nmatplotlib==3.9.4\nsix==1.17.0\nterminaltables==3.1.10\ntqdm==4.65.2\naddict==2.4.0\npackaging==24.2\nPyYAML==6.0.2\ntermcolor==3.0.1\nyapf==0.43.0\nregex==2024.11.6\nrequests==2.28.2\n# Resolver, transitive lock, pip check and CUDA ABI remain future runtime gates.\n\n# Torch 2.0 cpp_extension imports pkg_resources.packaging.\nsetuptools==69.5.1\n'
IMAGE_CONTRACTS='"""Synthetic uint8/BGR Color gate; imports no detector or model."""\ndef require_bgr_uint8(img,backend):\n import numpy as np\n if backend!=\'cv2\' or img.dtype!=np.uint8 or img.ndim!=3 or img.shape[2]!=3:raise ValueError(\'Only explicitly declared cv2 uint8 BGR is allowed.\')\ndef verify_color_contract(Color):\n import cv2,numpy as np\n image=(np.arange(17*19*3,dtype=np.uint16)%256).astype(np.uint8).reshape(17,19,3)\n require_bgr_uint8(image,\'cv2\');saved=np.random.get_state()\n def state():\n  x=np.random.get_state();return (x[0],x[1].tobytes(),x[2],x[3],x[4])\n try:\n  np.random.seed(1729);a=Color(prob=1.0,level=0,min_mag=1.0,max_mag=1.0)({\'img\':image.copy()})[\'img\'];ra=state()\n  np.random.seed(1729);b=Color(prob=1.0,level=0,min_mag=.8,max_mag=.8)({\'img\':image.copy()})[\'img\'];rb=state()\n  np.random.seed(1729);np.random.rand();expected_rng=state()\n  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);expected=cv2.addWeighted(image,.8,np.repeat(gray[...,None],3,axis=2),.2,0)\n  result=dict(identity_exact=bool(np.array_equal(a,image)),rng_equal=ra==rb,exactly_one_numpy_draw=ra==expected_rng,b_matches_cv2_bgr_blend=bool(np.array_equal(b,expected)),cv2_version=cv2.__version__,numpy_version=np.__version__)\n  if not all(result[k] for k in (\'identity_exact\',\'rng_equal\',\'exactly_one_numpy_draw\',\'b_matches_cv2_bgr_blend\')):raise ValueError(\'Color contract failed: \'+str(result))\n  require_bgr_uint8(a,\'cv2\');require_bgr_uint8(b,\'cv2\');return result\n finally:np.random.set_state(saved)\n'
IMPLEMENTATION_PINS={'file_sha256': {'mmdet/__init__.py': '2cdba8435317cb84202c3b138e71465ed99688f39433e51e5424a363da922d79', 'mmdet/datasets/coco.py': 'e267a8af63cd6cf0a0966f3b2e6b14831d04b48d9e8f6018dc53e54dece29b76', 'mmdet/datasets/transforms/colorspace.py': 'c0288547217e983a3ce110fd96d8e9defd9d80d566931a9f5362320ed7ec3e1c', 'mmdet/datasets/transforms/formatting.py': '14f3d8dbf9189d7b97e47a9453b97aad67aa3cfd66f69a5c2cd8e330649ba16a', 'mmdet/models/detectors/grounding_dino.py': 'a79d59ca49fe66734273a6df4a338aeb1505a2ea54fc1acd164e6ef638ae3692', 'mmdet/models/layers/transformer/dino_layers.py': 'efd6c26addba00a7120bb88621d0eb956a8a681ea0e84515b7c0df0e92a4fb22'}, 'mmdet_version': '3.3.0', 'upstream_commit': '44ebd17b145c2372c4b700bfb9cb20dbd28ab64a'}
IMPORT_GATE="import sys,json,pathlib,hashlib\nimport torch,mmcv,mmdet,mmengine,numpy,cv2\nfrom mmdet.datasets.transforms import Color\nfrom image_contracts import verify_color_contract\nmmcv.use_backend('cv2')\nassert torch.__version__=='2.1.0+cu118' and mmcv.__version__=='2.1.0' and mmdet.__version__=='3.3.0'\nsite=pathlib.Path(mmdet.__file__).parent.parent\npins=json.loads(pathlib.Path('implementation_pins.json').read_text())\nfor name,h in pins['file_sha256'].items():assert hashlib.sha256((site/name).read_bytes()).hexdigest()==h,name\nfrom mmcv.ops import nms\nkeep=nms(torch.tensor([[0.,0.,5.,5.],[0.,0.,5.,5.]]),torch.tensor([.9,.8]),.5)[1].tolist();assert keep==[0]\nresult={'status':'PASS','python':sys.version,'torch':torch.__version__,'mmcv':mmcv.__version__,'mmdet':mmdet.__version__,'mmengine':mmengine.__version__,'numpy':numpy.__version__,'cv2':cv2.__version__,'cuda_available':torch.cuda.is_available(),'official_color':verify_color_contract(Color),'compiled_cpu_nms_synthetic':True,'implementation_source_hashes_match':True,'model_instantiated':False}\npathlib.Path('import_gate.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))\n"
root=pathlib.Path('/content/deepfish_cpu_bootstrap');root.mkdir(exist_ok=False);os.chdir(root)
try:
 print('DEEPFISH_CPU_STAGE '+json.dumps({'stage':'BOOTSTRAP_STARTED'}),flush=True)
 result=bootstrap(root,PINS,REQUIREMENTS,time.time()+1200)
 (root/'image_contracts.py').write_text(IMAGE_CONTRACTS);atomic_json(root/'implementation_pins.json',IMPLEMENTATION_PINS);(root/'import_gate.py').write_text(IMPORT_GATE)
 with (root/'import_gate.log').open('wb') as f:subprocess.run([result['python_executable'],'-B',str(root/'import_gate.py')],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=90,env=dict(os.environ,TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',MPLBACKEND='Agg'))
 print('DEEPFISH_CPU_STAGE '+json.dumps({'stage':'IMPORT_GATE_PASSED'}),flush=True)
except BaseException as error:
 atomic_json(root/'bootstrap_error.json',{'status':'FAILED','error_type':type(error).__name__,'error':str(error),'model_instantiated':False})
finally:
 marker=root/'persistence_marker.bin';data=(b'DeepFish public non-sensitive persistence marker v1\n'*200000)[:8*1024**2]
 with marker.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert marker.read_bytes()==data
 files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.log','.txt') and p.name not in ('evidence_index.json',)]
 with zipfile.ZipFile(root/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.name)
 index={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (root/'evidence.zip',marker)}
 atomic_json(root/'evidence_index.json',index)
 print('DEEPFISH_CPU_EVIDENCE '+json.dumps(index,sort_keys=True),flush=True)
