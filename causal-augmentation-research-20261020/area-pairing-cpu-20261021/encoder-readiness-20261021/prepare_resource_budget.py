"""Read only official PyPI metadata and write a proposed, fully pinned CPU budget.

This script DOES NOT download wheels/weights, install packages, or alter environments.
"""
from pathlib import Path
import datetime
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version, InvalidVersion

ROOT=Path(__file__).resolve().parent
META=ROOT/'metadata'
ENV=default_environment();ENV['extra']=''
TAGS={tag:i for i,tag in enumerate(sys_tags())}
UA={'User-Agent':'CPU-encoder-resource-audit/1.0'}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=20) as response:
        body=response.read(2097153)
        assert response.status==200 and len(body)<=2097152
    return body


def metadata(name,version):
    path=META/(name+'-'+version+'.json')
    if path.exists():return json.loads(path.read_text())
    body=fetch('https://pypi.org/pypi/'+name+'/'+version+'/json')
    with path.open('xb') as f:f.write(body)
    return json.loads(body)


def wheel(data):
    choices=[]
    for row in data['urls']:
        if row.get('yanked') or not row['filename'].endswith('.whl'):continue
        _,_,_,tags=parse_wheel_filename(row['filename'])
        scores=[TAGS[tag] for tag in tags if tag in TAGS]
        if scores:choices.append((min(scores),row))
    assert choices,'No compatible binary wheel for '+data['info']['name']
    return min(choices,key=lambda x:x[0])[1]


def choose(name,requirements):
    body=fetch('https://pypi.org/rss/project/'+name+'/releases.xml')
    with (META/(name+'-releases.xml.txt')).open('xb') as f:f.write(body)
    versions=[]
    for node in ET.fromstring(body).findall('./channel/item'):
        try:v=Version(node.findtext('title'))
        except InvalidVersion:continue
        if not v.is_prerelease and not v.is_devrelease and all(v in req for req in requirements):versions.append(v)
    assert versions,'No compatible stable RSS release for '+name
    for version in sorted(set(versions),reverse=True):
        data=metadata(name,str(version))
        if data['info'].get('requires_python') and Version(ENV['python_full_version']) not in SpecifierSet(data['info']['requires_python']):continue
        try:wheel(data)
        except AssertionError:continue
        return data
    raise ValueError('No stable compatible wheel for '+name)


def main():
    pins={'torch':'2.14.1','torchvision':'0.29.1','numpy':'2.5.4','pillow':'13.0.0'}
    constraints={name:[SpecifierSet('=='+v)] for name,v in pins.items()}
    queue=list(pins);resolved={};ignored=[]
    while queue:
        name=queue.pop(0)
        if name in resolved:
            assert all(Version(resolved[name]['info']['version']) in spec for spec in constraints[name])
            continue
        data=metadata(name,pins[name]) if name in pins else choose(name,constraints[name])
        version=Version(data['info']['version'])
        assert all(version in spec for spec in constraints[name])
        assert not data['info'].get('requires_python') or Version(ENV['python_full_version']) in SpecifierSet(data['info']['requires_python'])
        resolved[name]=data
        for text in data['info'].get('requires_dist') or []:
            req=Requirement(text)
            if req.marker is not None and not req.marker.evaluate(ENV):
                ignored.append(dict(parent=name,requirement=text));continue
            dependency=canonicalize_name(req.name)
            assert not req.url and not req.extras,'Unexpected additional source/extras'
            constraints.setdefault(dependency,[]).append(req.specifier)
            if dependency not in queue:queue.append(dependency)
    for name,data in resolved.items():
        for text in data['info'].get('requires_dist') or []:
            req=Requirement(text)
            if req.marker is None or req.marker.evaluate(ENV):
                dependency=canonicalize_name(req.name)
                assert dependency in resolved and Version(resolved[dependency]['info']['version']) in req.specifier
    rows=[]
    for name,data in sorted(resolved.items()):
        item=wheel(data)
        rows.append(dict(package=name,version=data['info']['version'],filename=item['filename'],bytes=item['size'],
                         sha256=item['digests']['sha256'],url=item['url'],requires_python=data['info'].get('requires_python'),
                         license_expression=data['info'].get('license_expression'),license=data['info'].get('license')))
    head=json.loads((ROOT/'resnet18-weight-HEAD.json').read_text())
    result=dict(status='proposal_only_no_install_or_weight_download',as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                interpreter=ENV,packages=rows,all_active_dependency_constraints_satisfied=True,
                no_Linux_CUDA_packages_selected=True,excluded_marked_requirements=ignored,
                total_wheel_download_bytes=sum(x['bytes'] for x in rows),weight=head,
                total_download_payload_bytes=sum(x['bytes'] for x in rows)+head['bytes'],
                cache_reuse_assumed=False,package_installation_executed=False,weight_payload_downloaded=False,
                architecture='existing CPython3.14.7 / macOS arm64; binary wheels only')
    with (ROOT/'proposed-resource-budget.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    with (ROOT/'proposed-requirements-lock.txt').open('x') as f:
        f.write('# Proposal only. Requires explicit approval before installing. CPython3.14 macOS arm64.\n')
        for row in rows:f.write(row['package']+' @ '+row['url']+' --hash=sha256:'+row['sha256']+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['excluded_marked_requirements','interpreter']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
