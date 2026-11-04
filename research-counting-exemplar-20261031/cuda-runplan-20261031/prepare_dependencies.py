"""Local preparation only; obtain official Linux wheels, never install or allocate."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import json,hashlib,urllib.request,time

ROOT=_portable_project_path(__file__, 'research-counting-exemplar-20261031')
PLAN=ROOT/'cuda-runplan-20261031'
WHEELS=PLAN/'payload/wheels'
META=PLAN/'sources'
for p in [WHEELS,META,PLAN/'reports']:p.mkdir(parents=True,exist_ok=True)
rows=[];downloaded=0
metadata=[]
for p in sorted((ROOT/'cpu-readiness/sources').glob('pypi-*.json')):
    d=json.loads(p.read_text());metadata.append((d,str(p.relative_to(ROOT))))
for name,version in [('huggingface-hub','0.36.2'),('safetensors','0.8.0'),('hf-xet','1.8.0')]:
    url=f'https://pypi.org/pypi/{name}/{version}/json'
    with urllib.request.urlopen(url,timeout=30) as r:body=r.read(4*1024*1024)
    downloaded+=len(body);p=META/f'pypi-{name}-{version}.json';p.write_bytes(body)
    metadata.append((json.loads(body),str(p.relative_to(ROOT))))
for d,source in metadata:
    candidates=[]
    for row in d['urls']:
        f=row['filename']
        compatible='none-any.whl' in f or ('manylinux' in f and 'x86_64' in f and 'abi3' in f)
        if row['packagetype']=='bdist_wheel' and compatible and not row.get('yanked',False):candidates.append(row)
    if not candidates:raise RuntimeError('No portable official wheel '+d['info']['name'])
    chosen=min(candidates,key=lambda x:x['size'])
    p=WHEELS/chosen['filename'];expected=chosen['digests']['sha256']
    if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    else:
        assert chosen['url'].startswith('https://files.pythonhosted.org/')
        sha=hashlib.sha256();size=0
        with urllib.request.urlopen(chosen['url'],timeout=30) as response,p.with_suffix('.part').open('wb') as output:
            while True:
                b=response.read(1024*1024)
                if not b:break
                size+=len(b);downloaded+=len(b)
                if size>chosen['size']:raise RuntimeError('wheel size exceeded metadata')
                sha.update(b);output.write(b)
        assert size==chosen['size'] and sha.hexdigest()==expected
        p.with_suffix('.part').replace(p)
    rows.append({'name':d['info']['name'],'version':d['info']['version'],'filename':chosen['filename'],'url':chosen['url'],'sha256':expected,'bytes':chosen['size'],'requires_python':d['info'].get('requires_python'),'requires_dist':d['info'].get('requires_dist') or [],'metadata_source':source})
    print(json.dumps({'prepared':d['info']['name'],'version':d['info']['version'],'bytes':chosen['size']}),flush=True)
(PLAN/'payload/requirements-fixed.lock').write_text('\n'.join(f"{x['name']}=={x['version']} --hash=sha256:{x['sha256']}" for x in rows)+'\n')
report={'packages':rows,'downloaded_response_body_bytes':downloaded,'wheel_bytes':sum(x['bytes'] for x in rows),'old_download_ledger_bytes':1529818810,'new_download_ledger_bytes':1529818810+downloaded,'remote_weight_copy_bytes_planned':1378009960,'download_cap_bytes':int(3.5*1024**3),'local_installs':0,'remote_installs':0,'GPU_allocations':0,'generated_unix':time.time()}
assert report['new_download_ledger_bytes']+report['remote_weight_copy_bytes_planned']<report['download_cap_bytes']
(PLAN/'reports/dependency_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
(PLAN/'payload/dependency_manifest.json').write_text(json.dumps({'packages':rows},indent=2)+'\n')
print(json.dumps({'wheel_bytes':report['wheel_bytes'],'downloaded_bytes':downloaded,'packages':len(rows),'installs':0,'allocations':0}),flush=True)
