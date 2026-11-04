"""Prepare verified official assets; never execute downloaded research code."""
import hashlib, importlib.metadata as im, json, os, shutil, subprocess, sys, tarfile, time
import urllib.request, urllib.parse, zipfile
from datetime import datetime, timezone
from pathlib import Path
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parent
LIMIT = 3 * 1024**3
DEADLINE = datetime(2026,10,31,1,14,tzinfo=timezone.utc).timestamp()
SOURCES = ROOT/'sources'
for name in ['sources','assets','data','logs','wheelhouse','deps']:
    (ROOT/name).mkdir(exist_ok=True)
receipts=json.loads((SOURCES/'receipts.json').read_text()) if (SOURCES/'receipts.json').exists() else []

def js(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def check():
    if time.time() >= DEADLINE: raise RuntimeError('RESOURCE: six-hour wall limit')
    n=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if n > LIMIT: raise RuntimeError(f'RESOURCE: disk cap {n}')
    return n

def fetch(url, path, expected=None, algorithm='sha256', cap=900_000_000):
    assert url.startswith('https://')
    check()
    for old in receipts:
        if old['url']==url and path.exists() and old['sha256']==hashlib.file_digest(path.open('rb'),'sha256').hexdigest():
            print('VERIFIED_EXISTING',str(path.relative_to(ROOT)),flush=True);return path
    req=urllib.request.Request(url,headers={'User-Agent':'pets-view-route-local-research/1.0'})
    started=time.time()
    with urllib.request.urlopen(req,timeout=90) as resp, path.open('wb') as out:
        final=resp.url
        if not final.startswith('https://'): raise RuntimeError('Non-HTTPS redirect')
        length=resp.headers.get('Content-Length')
        if length and int(length)>cap: raise RuntimeError('Asset exceeds declared cap')
        count=0
        while b:=resp.read(1024**2):
            out.write(b); count+=len(b)
            if count>cap: raise RuntimeError('Asset exceeds cap')
            if time.time()>DEADLINE: raise RuntimeError('Wall limit')
    sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    digest=hashlib.file_digest(path.open('rb'),algorithm).hexdigest()
    if expected and not digest.startswith(expected): raise RuntimeError(f'Checksum mismatch: {path}')
    r={'url':url,'resolved_url':final,'path':str(path.relative_to(ROOT)),'bytes':count,'sha256':sha,
       'verified_algorithm':algorithm,'expected':expected,'actual':digest,'seconds':time.time()-started,
       'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    receipts.append(r);js(SOURCES/'receipts.json',receipts)
    print(json.dumps(r),flush=True);check()
    return path

def metadata(package,version=None):
    url=f'https://pypi.org/pypi/{package}/'+(f'{version}/' if version else '')+'json'
    p=SOURCES/f'pypi-{package}-{version or "current"}.json'
    fetch(url,p,cap=20_000_000)
    return json.loads(p.read_text())

def dependencies():
    pins={'timm':'1.0.22','huggingface-hub':'0.36.0'}
    queue=[Requirement('timm==1.0.22')]
    selected={};wheels=[];supported=set(sys_tags())
    existing={canonicalize_name(d.metadata['Name']):d.version for d in im.distributions()}
    while queue:
        req=queue.pop(0)
        if req.marker and not req.marker.evaluate({'extra':''}): continue
        name=canonicalize_name(req.name)
        if name in selected:
            assert selected[name]['version'] in req.specifier,(req,selected[name])
            continue
        if name in existing and existing[name] in req.specifier:
            selected[name]={'version':existing[name],'source':'existing-environment'};continue
        meta=metadata(name,pins.get(name))
        version=meta['info']['version'];assert version in req.specifier,(name,version,req)
        candidates=[]
        for u in meta['urls']:
            if u['packagetype']!='bdist_wheel' or u.get('yanked'):continue
            _,_,_,tags=parse_wheel_filename(u['filename'])
            if supported.intersection(tags): candidates.append(u)
        if not candidates: raise RuntimeError(f'No compatible official wheel: {name} {version}')
        u=sorted(candidates,key=lambda x:x['filename'])[0]
        assert urllib.parse.urlparse(u['url']).hostname=='files.pythonhosted.org'
        p=fetch(u['url'],ROOT/'wheelhouse'/u['filename'],u['digests']['sha256'],cap=100_000_000)
        wheels.append(str(p));selected[name]={'version':version,'source':'official-pypi','wheel':u['filename'],'sha256':u['digests']['sha256']}
        queue.extend(Requirement(x) for x in meta['info'].get('requires_dist') or [])
    js(ROOT/'dependency-lock.json',selected)
    command=[sys.executable,'-B','-m','pip','install','--no-input','--no-deps','--no-index','--no-compile','--target',str(ROOT/'deps'),*wheels]
    r=subprocess.run(command,capture_output=True,text=True)
    (ROOT/'logs'/'pip-install.txt').write_text(r.stdout+'\n'+r.stderr)
    r.check_returncode()
    print('DEPENDENCIES_READY',flush=True)

def assets(model_only=False):
    api=fetch('https://api.github.com/repos/facebookresearch/deit/commits/main',SOURCES/'deit-commit.json',cap=5_000_000)
    rev=json.loads(api.read_text())['sha']
    for name in ['README_deit.md','LICENSE']:
        fetch(f'https://raw.githubusercontent.com/facebookresearch/deit/{rev}/{name}',SOURCES/f'deit-{name}',cap=1_000_000)
    hf=fetch('https://huggingface.co/api/models/timm/deit_tiny_patch16_224.fb_in1k',SOURCES/'timm-model-api.json',cap=2_000_000)
    hrev=json.loads(hf.read_text())['sha']
    for name in ['README.md','config.json']:
        fetch(f'https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k/resolve/{hrev}/{name}',SOURCES/f'timm-{name}',cap=2_000_000)
    url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
    assert url in (SOURCES/'deit-README_deit.md').read_text()
    fetch(url,ROOT/'assets'/'deit_tiny_patch16_224-a1311bcf.pth','a1311bcf',cap=30_000_000)
    if model_only:return
    fetch('https://www.robots.ox.ac.uk/~vgg/data/pets/',SOURCES/'pets-official.html',cap=2_000_000)
    # Official torchvision 0.28.0 ships these exact URLs and published MD5s.
    tv=Path(im.distribution('torchvision').locate_file('torchvision/datasets/oxford_iiit_pet.py'))
    shutil.copyfile(tv,SOURCES/'torchvision-0.28.0-oxford_iiit_pet.py')
    for name,md5 in [('annotations','95a8c909bbe2e81eed6a22bccdf3f68f'),('images','5c4f3ee8e5d25df40f4fd59a7f44e54c')]:
        url=f'https://www.robots.ox.ac.uk/~vgg/data/pets/data/{name}.tar.gz'
        assert url in tv.read_text() and md5 in tv.read_text()
        archive=fetch(url,ROOT/'assets'/f'{name}.tar.gz',md5,'md5')
        with tarfile.open(archive,'r:gz') as tar:
            members=[]
            for m in tar.getmembers():
                p=Path(m.name)
                if p.is_absolute() or '..' in p.parts: raise RuntimeError('Unsafe archive member')
                if not m.isfile():continue
                if name=='images' and p.suffix.lower()=='.jpg' and p.parts[0]=='images': members.append(m)
                if name=='annotations' and str(p) in ['annotations/trainval.txt','annotations/test.txt','annotations/list.txt','annotations/README']:members.append(m)
            if check()+sum(m.size for m in members)>LIMIT: raise RuntimeError('Extraction exceeds disk budget')
            for m in members:
                target=ROOT/'data'/m.name;target.parent.mkdir(exist_ok=True,parents=True)
                with tar.extractfile(m) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
        # Only task-created checked archives are removed; receipts preserve source and digest.
        archive.unlink();check()
    js(ROOT/'asset-preparation.json',{'status':'complete','deit_revision':rev,'timm_hf_revision':hrev,
       'data_license':'CC-BY-SA-4.0; original image copyrights retained','model_license':'Apache-2.0',
       'disk_bytes':check(),'archives_deleted_after_verified_extraction':True})

if __name__=='__main__':
    if len(sys.argv)==1:dependencies()
    assets(model_only='model-only' in sys.argv)
