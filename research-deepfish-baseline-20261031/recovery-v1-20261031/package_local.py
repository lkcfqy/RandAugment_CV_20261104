"""Exact-member public-data packaging and privacy audit; never uploads or runs a model."""
import ast,hashlib,json,pathlib,re,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
TOP={'bootstrap_environment.py','python_bootstrap_pins.json','transfer.py','config_A.json','config_B.json','driver.py','model_worker.py','contracts.py','lifecycle.py','image_contracts.py','requirements_candidate.txt','asset_pins.json','implementation_pins.json','data_selection.json','README.md','ATTRIBUTION.md','run_context.example.json','LICENSE-MMDETECTION'}
# Frozen against official 5_shot.json and the original train records selected by
# DEV_SELECTION.json; never regenerated from payload annotations during build.
FROZEN_ANNOTATION_CANONICAL_SHA256={'support5.json':'cdd5cd9cb7619fc3fe3a7ae05404270aa06e7e7397e3a6b2def7bb2efa685703','dev64.json':'35f7c48c6040722f860037b0a1216502ff7616f5199d609a2bac1a147b49b730'}
def digest(data):return hashlib.sha256(data).hexdigest()
def check_text(text):
 for pattern in [r'/Users/',r'/home/',r'gh[pousr]_[A-Za-z0-9]{20,}',r'sk-[A-Za-z0-9_-]{20,}',r'hf_[A-Za-z0-9]{20,}',r'AKIA[A-Z0-9]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'Bearer\s+[A-Za-z0-9_-]{20,}']:
  if re.search(pattern,text):raise ValueError('Potential private path/credential pattern: '+pattern)
def audit_payload(root):
 from PIL import Image
 root=pathlib.Path(root);selection=json.loads((root/'data_selection.json').read_text());expected={r['file']:r for r in selection['images']}
 allowed=TOP|{'data/annotations/support5.json','data/annotations/dev64.json'}|{'data/images/'+x for x in expected}|{'MANIFEST.json','SHA256SUMS'}
 files=[p for p in root.rglob('*') if p.is_file()];names={str(p.relative_to(root)) for p in files}
 if names-allowed or (allowed-{'MANIFEST.json','SHA256SUMS'})-names:raise ValueError('Payload allowlist mismatch: '+str(sorted(names^allowed)))
 private=[];image_total=0
 for p in files:
  if p.is_symlink():raise ValueError('Symlink forbidden')
  rel=str(p.relative_to(root))
  if p.suffix=='.jpg':
   image_total+=1;data=p.read_bytes();r=expected[p.name]
   if len(data)!=r['bytes'] or digest(data)!=r['sha256']:raise ValueError('Original public image hash mismatch')
   with Image.open(p) as image:
    if image.size!=(1920,1080) or image.format!='JPEG':raise ValueError('Image header contract')
    if image.getexif() or image.info.get('comment') or any('xmp' in k.lower() for k in image.info):private.append(p.name)
  else:
   text=p.read_text();check_text(text)
   if p.suffix=='.py':ast.parse(text)
   if p.suffix=='.json':json.loads(text)
 if image_total!=69 or private:raise ValueError('Unexpected image count or embedded metadata: '+str(private))
 support=json.loads((root/'data/annotations/support5.json').read_text());dev=json.loads((root/'data/annotations/dev64.json').read_text())
 for name,data in [('support5.json',support),('dev64.json',dev)]:
  canonical=(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
  if digest(canonical)!=FROZEN_ANNOTATION_CANONICAL_SHA256[name]:raise ValueError('Frozen annotation digest changed: '+name)
 for key,data in [('support_ids',support),('dev_ids',dev)]:
  if sorted(selection[key])!=sorted(row['id'] for row in data['images']):raise ValueError('Frozen selected-record membership changed: '+key)
 if set(expected)!={row['file_name'] for data in (support,dev) for row in data['images']}:raise ValueError('Image membership differs from annotations')
 edges={row['id']:row for row in dev['annotations']}
 for annotation_id,bbox,area in [(4195,[1867.5,117.5,53.0,117.0],6201.0),(4660,[1779.5,85.5,141.0,83.0],11703.0)]:
  if edges[annotation_id]['bbox']!=bbox or edges[annotation_id]['area']!=area:raise ValueError('Original edge annotation changed')
 if (len(support['images']),len(support['annotations']),len(dev['images']),len(dev['annotations']))!=(5,5,64,187):raise ValueError('Data budget changed')
 if {i['file_name'] for i in support['images']}&{i['file_name'] for i in dev['images']}:raise ValueError('Support/dev overlap')
 return {'images':image_total,'support_images':5,'support_boxes':5,'extra_dev_images':64,'extra_dev_boxes':187,'private_metadata_images':private,'all_image_hashes_match':True,'frozen_annotation_digests_match':True,'selected_record_membership_matches':True,'original_edge_boxes_retained':True,'file_allowlist_passed':True,'no_private_paths_or_credential_patterns':True,'official_test_included':False,'weights_repositories_or_old_experiments_included':False,'manifest_payload_file_count':len(names-{'MANIFEST.json','SHA256SUMS'})}
def build():
 root=ROOT/'payload';audit=audit_payload(root);rows=[]
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.name not in ('MANIFEST.json','SHA256SUMS'):rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())})
 manifest={'format_version':1,'scope':'fixed DeepFish 5-shot A/B local preparation','files':rows}
 m=root/'MANIFEST.json';m.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 sums=[(r['path'],r['sha256']) for r in rows]+[('MANIFEST.json',digest(m.read_bytes()))]
 (root/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(sums)))
 archive=ROOT/'deepfish_5shot_two_arm.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file():
    info=zipfile.ZipInfo(str(p.relative_to(root)),date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
 if archive.stat().st_size>13*1024**2:raise ValueError('Upload archive exceeds 13 MiB; no silent data reduction')
 expected={str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
 with zipfile.ZipFile(archive) as z:
  if set(z.namelist())!=set(expected) or len(z.namelist())!=len(expected):raise ValueError('ZIP membership mismatch')
  for name,h in expected.items():
   if digest(z.read(name))!=h:raise ValueError('ZIP round-trip hash mismatch')
 report={'status':'LOCAL_PACKAGE_VERIFIED','archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive.read_bytes()),'archive_limit_bytes':13*1024**2,'manifest_sha256':digest(m.read_bytes()),'sha256sums_sha256':digest((root/'SHA256SUMS').read_bytes()),'zip_members':len(expected),'uncompressed_payload_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'zip_roundtrip_verified':True,'privacy_audit':audit,'uploaded':False,'model_executed':False,'dependencies_installed':False,'resource_status':'AUTHORIZED_PENDING_CUDA_PREFLIGHT','pending_command':'python -B driver.py --run-context /content/deepfish-run-context.json'}
 (ROOT/'reports').mkdir(exist_ok=True);(ROOT/'reports/PACKAGE_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n')
 return report
if __name__=='__main__':
 # Kept below the function so unit tests only import audit functions.
 report=build();print(json.dumps(report,indent=2))
