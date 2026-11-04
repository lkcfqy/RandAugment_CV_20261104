"""Prepare isolated overlay from existing packages + verified official PyPI wheels."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import requests,sys,json,hashlib,time,subprocess,importlib.metadata as metadata,os
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name,parse_wheel_filename
ROOT=_portable_project_path(__file__, 'research-counting-exemplar-20261031')
PHASE=ROOT/'cpu-readiness';ENV=PHASE/'.venv'
for d in ['sources','wheelhouse','reports','scripts','logs']:(PHASE/d).mkdir(parents=True,exist_ok=True)
START=time.time();downloaded=0;http=[]
def write(name,value):
    (PHASE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def fetch(url,dest,maxsize):
    global downloaded
    p=PHASE/dest
    if p.exists():return p
    started=time.time();h=hashlib.sha256();n=0
    with requests.get(url,stream=True,timeout=(15,60)) as resp:
        resp.raise_for_status()
        if int(resp.headers.get('Content-Length',0))>maxsize:raise RuntimeError('Body exceeds limit')
        with p.open('xb') as f:
            for chunk in resp.iter_content(1024*1024):
                n+=len(chunk);downloaded+=len(chunk)
                if n>maxsize or downloaded>500*1024**2:raise RuntimeError('Dependency download budget exceeded')
                f.write(chunk);h.update(chunk)
    http.append({'url':url,'path':dest,'bytes':n,'sha256':h.hexdigest(),'seconds':time.time()-started});write('reports/dependency_downloads.json',http)
    return p
installed={}
for d in metadata.distributions():
    name=canonicalize_name(d.metadata['Name'])
    if name not in installed:installed[name]=d
requested={'transformers':'4.49.0','tokenizers':'0.21.4','yapf':'0.40.1','timm':None,'addict':None,'termcolor':None,'opencv-python-headless':None,'pycocotools':None}
new={};queue=list(requested.items());reuse=set();constraints=[]
while queue:
    name,version=queue.pop(0);name=canonicalize_name(name)
    if name in new:continue
    url='https://pypi.org/pypi/'+name+'/'+(version+'/' if version else '')+'json'
    p=fetch(url,'sources/pypi-'+name+'-'+(version or 'latest')+'.json',5_000_000);data=json.loads(p.read_text());version=data['info']['version']
    if data['info'].get('requires_python'):assert SpecifierSet(data['info']['requires_python']).contains('.'.join(str(x) for x in sys.version_info[:3]))
    new[name]={'version':version,'metadata':data}
    for line in data['info'].get('requires_dist') or []:
        r=Requirement(line)
        if r.marker and not r.marker.evaluate({'extra':''}):continue
        dep=canonicalize_name(r.name);constraints.append((name,dep,str(r.specifier)))
        if dep in new:
            assert r.specifier.contains(new[dep]['version']), (name,line,new[dep]['version'])
        elif dep in requested:queue.append((dep,requested[dep]))
        elif dep in installed and r.specifier.contains(installed[dep].version):reuse.add(dep)
        else:queue.append((dep,None))
for src,name,spec in constraints:
    v=new[name]['version'] if name in new else installed[name].version
    assert SpecifierSet(spec).contains(v),(src,name,spec,v)
tag_rank={t:i for i,t in enumerate(sys_tags())};wheels=[]
for name,item in new.items():
    options=[]
    for f in item['metadata']['urls']:
        if f['packagetype']!='bdist_wheel' or f.get('yanked'):continue
        _,_,_,tags=parse_wheel_filename(f['filename']);rank=min((tag_rank[t] for t in tags if t in tag_rank),default=10**9)
        if rank<10**9:options.append((rank,f['filename'],f))
    if not options:raise RuntimeError('No compatible binary wheel: '+name+' '+item['version'])
    f=min(options,key=lambda x:x[:2])[2]
    assert f['url'].startswith('https://files.pythonhosted.org/')
    p=fetch(f['url'],'wheelhouse/'+f['filename'],150_000_000)
    actual=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert actual==f['digests']['sha256']
    wheels.append({'name':name,'version':item['version'],'filename':f['filename'],'url':f['url'],'bytes':p.stat().st_size,'sha256':actual})
write('reports/new_wheel_manifest.json',wheels)
if not ENV.exists():subprocess.run([sys._base_executable,'-m','venv',str(ENV)],check=True)
site=ENV/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages';links=[]
# Reuse existing packages by selective links, excluding overridden distributions.
# The fresh environment has exactly one active transformers/tokenizers version.
for name,d in sorted(installed.items()):
    if name in new or name=='pip':continue
    base=Path(d.locate_file('')).resolve()
    if not d.files:raise RuntimeError('No file manifest for reuse '+name)
    tops=sorted({p.parts[0] for p in d.files if p.parts and p.parts[0] not in ['..','__pycache__']})
    for top in tops:
        source=base/top;target=site/top
        if not source.exists():continue
        if target.is_symlink():assert target.resolve()==source.resolve();continue
        if target.exists():continue
        target.symlink_to(source,target_is_directory=source.is_dir());links.append({'distribution':name,'version':d.version,'target':str(target),'source':str(source)})
write('reports/reused_package_links.json',links)
cmd=[str(ENV/'bin/python'),'-m','pip','install','--no-index','--no-deps','--ignore-installed']+[str(PHASE/'wheelhouse'/x['filename']) for x in wheels]
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_NO_CACHE_DIR':'1'}
run=subprocess.run(cmd,env=env,text=True,capture_output=True);(PHASE/'logs/pip-install.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode()
run=subprocess.run([str(ENV/'bin/python'),'-m','pip','check'],env=env,text=True,capture_output=True);(PHASE/'logs/pip-check.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode()
write('reports/environment_preparation.json',{'started_unix':START,'finished_unix':time.time(),'base_python':sys._base_executable,'new_environment':str(ENV),'new_packages':[{k:x[k] for k in ['name','version','sha256']} for x in wheels],'reused_distributions':sorted(set(x['distribution'] for x in links)),'network_response_body_bytes':downloaded,'only_official_pypi_binary_wheels':True,'source_builds':0,'old_environment_updates':0,'model_forward_calls':0,'gpu_operations':0,'pip_check':run.stdout.strip()})
print(json.dumps({'new_packages':{x['name']:x['version'] for x in wheels},'reused_distributions':len(set(x['distribution'] for x in links)),'download_bytes':downloaded,'pip_check':run.stdout.strip()}))
