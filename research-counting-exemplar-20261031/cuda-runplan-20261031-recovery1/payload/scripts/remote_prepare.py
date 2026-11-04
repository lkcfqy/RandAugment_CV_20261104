"""Single isolated preparation child; no learned forward. Parent enforces wall/RSS/disk."""
import argparse,hashlib,importlib.metadata as metadata,json,os,platform,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
from contracts import canonical
from evidence import Evidence,atomic_bytes,owned_size
ROOT=Path(__file__).resolve().parents[1]
LIMIT=6*1024**3

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def owned_bytes():return owned_size(ROOT.parent)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allocation-seconds-left',type=float,required=True);args=parser.parse_args()
    start=time.monotonic();excluded=0.;evidence=Evidence(ROOT/'output');plan=json.loads((ROOT/'plan.json').read_text())
    preparation={'status':'RUNNING','native_environment':{},'transfers':[],'installation_wall_seconds':0.,'download_wall_seconds':0.,'new_authentication':False,'allocation_count':1,'retries':0}
    def left():return args.allocation_seconds_left-(time.monotonic()-start)
    def phase(name):atomic_bytes(ROOT/'preparation_state.json',canonical({'phase':name,'phase_started_monotonic':time.monotonic(),'excluded_wall_seconds':excluded}))
    def ensure():
        if left()<=90:raise RuntimeError('allocation cleanup reserve reached')
        if owned_bytes()>LIMIT:raise RuntimeError('owned remote disk cap exceeded')
    phase('COMPUTE')
    try:
        ensure()
        if sys.platform!='linux' or platform.machine()!='x86_64' or sys.version_info<(3,12):raise RuntimeError('frozen wheel set requires Linux x86_64 CPython >=3.12')
        import torch,torchvision
        if torch.version.cuda is None:raise RuntimeError('native Torch is not a CUDA build; GPU Torch replacement is forbidden')
        preparation['native_environment']={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'torch_cuda_build':torch.version.cuda,'native_torch_file':torch.__file__,'native_torchvision_file':torchvision.__file__}
        native_versions={name:metadata.version(name) for name in ('torch','torchvision')}
        packages=json.loads((ROOT/'dependency_manifest.json').read_text())['packages']
        for pkg in packages:
            path=ROOT/'wheels'/pkg['filename']
            if path.stat().st_size!=pkg['bytes'] or sha(path)!=pkg['sha256']:raise RuntimeError('frozen dependency wheel mismatch')
        phase('INSTALL');install_start=time.monotonic()
        pip=subprocess.run([sys.executable,'-m','pip','install','--target',str(ROOT/'deps'),'--no-index','--no-deps','--require-hashes','--only-binary=:all:','--find-links',str(ROOT/'wheels'),'-r',str(ROOT/'requirements-fixed.lock')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=min(180,max(1,left()-90)))
        install_wall=time.monotonic()-install_start;excluded+=install_wall;phase('COMPUTE');preparation['installation_wall_seconds']=install_wall
        (ROOT/'output').mkdir(exist_ok=True);atomic_bytes(ROOT/'output/install.log',pip.stdout.encode())
        if pip.returncode:raise RuntimeError('isolated fixed-wheel installation failed; see install.log')
        sys.path.insert(0,str(ROOT/'deps'))
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.markers import default_environment
        env=default_environment();env['extra']='';versions={}
        for pkg in packages:
            actual=metadata.version(pkg['name']);versions[pkg['name']]=actual
            if actual!=pkg['version'] or not SpecifierSet(pkg['requires_python']).contains(platform.python_version()):raise RuntimeError('fixed dependency/Python version mismatch: '+pkg['name'])
            for raw in pkg['requires_dist']:
                req=Requirement(raw)
                if req.marker and not req.marker.evaluate(env):continue
                actual_dep=metadata.version(req.name);versions[req.name]=actual_dep
                if not req.specifier.contains(actual_dep,prereleases=True):raise RuntimeError('native dependency incompatible: '+raw)
        for name in ('torch','torchvision','numpy','scipy','Pillow','matplotlib','PyYAML','requests','packaging','regex','filelock','tqdm','fsspec','typing_extensions'):
            versions[name]=metadata.version(name)
        if any(metadata.version(k)!=v for k,v in native_versions.items()):raise RuntimeError('native GPU Torch/vision version changed')
        preparation['resolved_versions']=versions;preparation['gpu_torch_reinstalled']=False
        evidence.write('engineering/preparation.json',preparation)
        weights=json.loads((ROOT/'manifests/model_identity.json').read_text())['weight_assets']
        projected=plan['download_bytes_before_allocation']+plan['bundle_transfer_budget_bytes']+sum(x['bytes'] for x in weights)+plan['result_download_reserve_bytes']
        if projected>plan['download_cap_bytes']:raise RuntimeError('cumulative transfer cap forecast exceeded')
        for asset in weights:
            ensure();path=ROOT/asset['path'];path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise RuntimeError('unexpected preexisting weight; one transfer per asset only')
            if owned_bytes()+asset['bytes']>LIMIT:raise RuntimeError('owned disk forecast exceeded')
            if not asset['url'].startswith('https://huggingface.co/'):raise RuntimeError('only fixed official HF weight URL allowed')
            phase('DOWNLOAD');begin=time.monotonic();count=0;temporary=path.with_suffix('.partial')
            request=urllib.request.Request(asset['url'],headers={'User-Agent':'CountGD-fixed-observation/1.0'})
            with urllib.request.urlopen(request,timeout=min(60,max(1,left()-90))) as response,temporary.open('xb') as stream:
                if response.status!=200:raise RuntimeError('weight download status not 200')
                final=urllib.parse.urlsplit(response.url)
                if final.scheme!='https' or not (final.hostname=='huggingface.co' or final.hostname.endswith(('.huggingface.co','.hf.co'))):raise RuntimeError('unexpected weight distribution redirect')
                while True:
                    if left()<=90:raise RuntimeError('allocation download reserve reached')
                    chunk=response.read(1024**2)
                    if not chunk:break
                    count+=len(chunk)
                    if count>asset['bytes']:raise RuntimeError('weight response exceeds frozen size')
                    stream.write(chunk)
                stream.flush();os.fsync(stream.fileno())
            elapsed=time.monotonic()-begin;excluded+=elapsed;phase('COMPUTE');preparation['download_wall_seconds']+=elapsed
            # Deliberate separate hash pass: verification CPU wall counts as computation.
            if count!=asset['bytes'] or sha(temporary)!=asset['sha256']:raise RuntimeError('official safetensors size/hash mismatch')
            os.replace(temporary,path)
            preparation['transfers'].append({'path':asset['path'],'bytes':count,'sha256':asset['sha256'],'download_wall_seconds':elapsed})
            evidence.write('engineering/preparation.json',preparation)
        ensure();preparation.update(status='READY',owned_disk_bytes=owned_bytes(),forecast_transfer_bytes=projected)
    except BaseException as error:
        preparation.update(status='BLOCKED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        # The parent freezes an in-progress download/install exclusion on failure.
        evidence.write('engineering/preparation.json',preparation)
    return 0 if preparation['status']=='READY' else 2

if __name__=='__main__':raise SystemExit(main())
