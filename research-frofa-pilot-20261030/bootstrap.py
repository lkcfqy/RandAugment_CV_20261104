"""Official source downloads, checksum verification and isolated wheel installation."""
import hashlib,json,sys,urllib.request,importlib.metadata,subprocess,time
from pathlib import Path
from packaging.requirements import Requirement
from packaging.version import Version,InvalidVersion
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename,canonicalize_name
from runtime import ROOT,P,Budget,Stop,save,sha
B=Budget();MAN=json.loads((ROOT/'downloads.json').read_text()) if (ROOT/'downloads.json').exists() else []
def fetch(url,dest=None,limit=1024**3):
    if dest is not None and Path(dest).exists():
        previous=[r for r in MAN if r['path']==str(Path(dest).relative_to(ROOT)) and r['url']==url]
        if previous and sha(dest)==previous[-1]['sha256']:
            B.event('reuse_verified_download',path=str(Path(dest).relative_to(ROOT)));return Path(dest)
        raise Stop('existing unverified download requires inspection '+str(dest))
    B.start();B.check()
    request=urllib.request.Request(url,headers={'User-Agent':'local-frozen-feature-pilot/1.0'})
    data=[];size=0;h=hashlib.sha256();f=None
    if dest is not None:
        dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise Stop('refuse duplicate download '+str(dest))
        f=dest.open('xb')
    try:
        with urllib.request.urlopen(request,timeout=45) as response:
            final=response.url
            while True:
                remaining=P['limits']['download_bytes']-B.s['counts'].get('download_bytes',0)
                if remaining<=0:raise Stop('download byte budget exhausted')
                block=response.read(min(1024*1024,remaining))
                if not block:break
                B.reserve('download_bytes',len(block));size+=len(block)
                if size>limit:raise Stop('per-download byte limit')
                h.update(block)
                if f is None:data.append(block)
                else:f.write(block)
    finally:
        if f is not None:f.close()
    MAN.append({'url':url,'resolved_host':urllib.parse.urlparse(final).hostname,'bytes':size,'sha256':h.hexdigest(),'path':None if dest is None else str(dest.relative_to(ROOT))})
    save(ROOT/'downloads.json',MAN);B.check(disk=True)
    return b''.join(data) if dest is None else dest
def pypi(name,version=None):
    url='https://pypi.org/pypi/'+name+('/'+version if version else '')+'/json'
    return json.loads(fetch(url,limit=32*1024*1024))
def dependencies():
    if (ROOT/'dependency-lock.json').exists() and (ROOT/'deps').exists():
        lock=json.loads((ROOT/'dependency-lock.json').read_text())
        installed={canonicalize_name(d.metadata['Name']):d.version for d in importlib.metadata.distributions(path=[str(ROOT/'deps')])}
        if all(installed.get(name)==item['version'] and sha(ROOT/'wheels'/item['wheel'])==item['sha256'] for name,item in lock['isolated'].items()):
            B.event('reuse_verified_isolated_dependencies');return
        raise Stop('existing isolated dependency lock mismatch')
    queue=[Requirement('transformers=='+P['dependencies']['transformers']),Requirement('safetensors=='+P['dependencies']['safetensors'])]
    chosen={};existing={};constraints={};wheels=[];tags=list(sys_tags());rank={t:i for i,t in enumerate(tags)}
    while queue:
        req=queue.pop(0)
        if req.marker and not req.marker.evaluate({'extra':''}):continue
        name=canonicalize_name(req.name)
        constraints[name]=SpecifierSet(','.join(filter(None,[str(constraints.get(name,'')),str(req.specifier)])))
        spec=constraints[name]
        if name in chosen:
            if Version(chosen[name]['version']) not in spec:raise Stop('dependency constraint conflict '+name)
            continue
        try:version=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:version=None
        if version is not None and Version(version) in spec:
            existing[name]=version;continue
        index=pypi(name)
        candidates=[]
        for text in index['releases']:
            try:parsed=Version(text)
            except InvalidVersion:continue
            if not parsed.is_prerelease and parsed in spec:candidates.append(parsed)
        candidates.sort(reverse=True)
        if not candidates:raise Stop('no official compatible dependency '+name)
        version=str(candidates[0]);metadata=pypi(name,version)
        options=[]
        for file in metadata['urls']:
            if file['packagetype']!='bdist_wheel' or file.get('yanked'):continue
            if file.get('requires_python') and Version('.'.join(map(str,sys.version_info[:3]))) not in SpecifierSet(file['requires_python']):continue
            _,_,_,wheel_tags=parse_wheel_filename(file['filename'])
            matches=[rank[t] for t in wheel_tags if t in rank]
            if matches:options.append((min(matches),file['filename'],file))
        if not options:raise Stop('no compatible binary wheel for '+name+' '+version)
        file=sorted(options,key=lambda t:(t[0],t[1]))[0][2]
        path=ROOT/'wheels'/file['filename'];fetch(file['url'],path,128*1024*1024)
        if sha(path)!=file['digests']['sha256']:raise Stop('PyPI wheel hash mismatch '+name)
        chosen[name]={'version':version,'wheel':file['filename'],'url':file['url'],'sha256':file['digests']['sha256'],'requires_dist':metadata['info']['requires_dist']}
        wheels.append(path)
        queue.extend(Requirement(r) for r in (metadata['info']['requires_dist'] or []))
    save(ROOT/'dependency-lock.json',{'existing':existing,'isolated':chosen})
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--target',str(ROOT/'deps')]+[str(p) for p in wheels]
    with (ROOT/'dependency-install.log').open('xb') as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,2700-B.elapsed()))
    if completed.returncode:raise Stop('isolated dependency installation failed')
    B.event('dependencies_ready',packages={k:v['version'] for k,v in chosen.items()});B.check(disk=True)
def assets():
    base='https://huggingface.co/'+P['assets']['model']+'/resolve/'+P['assets']['revision']+'/'
    for name in P['assets']['files']:
        path=ROOT/'assets/siglip'/name;fetch(base+name,path)
        if name=='model.safetensors' and sha(path)!=P['assets']['model_sha256']:raise Stop('model SHA256 mismatch')
        B.event('asset_ready',asset_name=name,bytes=path.stat().st_size)
    zip_path=ROOT/'assets/EuroSAT_RGB.zip';fetch(P['assets']['dataset_url'],zip_path,150*1024*1024)
    if sha(zip_path,'md5')!=P['assets']['dataset_md5']:raise Stop('EuroSAT MD5 mismatch')
    docs={
      'eurosat-record.json':'https://zenodo.org/api/records/7711810',
      'eurosat-MIT.txt':'https://raw.githubusercontent.com/phelber/EuroSAT/master/LICENSE',
      'Apache-2.0.txt':'https://www.apache.org/licenses/LICENSE-2.0.txt',
      'Sentinel_Data_Legal_Notice.pdf':'https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice'}
    for name,url in docs.items():fetch(url,ROOT/'licenses'/name,16*1024*1024)
    B.event('all_assets_verified',dataset_md5=sha(zip_path,'md5'),model_sha256=sha(ROOT/'assets/siglip/model.safetensors'))
def main():
    try:
        B.phase('bootstrap_dependencies');dependencies();B.phase('bootstrap_assets');assets();B.phase('assets_ready');return 0
    except Exception as e:
        import traceback
        save(ROOT/'STOP.json',{'phase':B.s['phase'],'reason':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':B.elapsed(),'resource_stop':True});B.s['phase']='stopped';B.flush();raise
if __name__=='__main__':main()
