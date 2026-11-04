import pathlib,json,hashlib,zipfile,subprocess,sys,os
CONFIG={'remote': 'deepfish_ab_18620499a5', 'parts': [{'file': 'package_part0.bin', 'bytes': 6291456, 'sha256': '650eae737d70c5c0ef0974bbe198bd524c72b388ab5897d10ae4833076dfd854'}, {'file': 'package_part1.bin', 'bytes': 5385370, 'sha256': '3c9eed97cbb8b533de6500b8f70825c6fc92cf939154fcaa0c985a6ea5ea082e'}], 'archive_sha256': '18620499a527d929a15bc05a239d6fa7e3d1f2d57aaccd91689710b17c50c439', 'archive_bytes': 11676826, 'context': {'authorization': {'upload': True, 'model_run': True}, 'allocation_started_epoch': 1793454745.3815339, 'cu_per_hour_upper_bound': 1.15, 'quote_verified': True, 'external_release_guard_verified': True, 'persistent_storage_verified': True, 'persistence_mode': 'local_cli_sha256', 'persistence_probe': {'remote_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'local_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'bytes': 8388608, 'download_bytes_per_second': 4415451.618314741, 'verified_epoch': 1793454760.721165}, 'prior_cu': 0.06980278659629822, 'prior_gpu_seconds': 20.373, 'artifact_dir': '/content/deepfish_ab_18620499a5/artifacts', 'expected_manifest_sha256': '35fe1ddfd783deab60164aab60b0083dd75c1aa8bdb45967d4992265bda06f0d'}}
base=pathlib.Path('/content')/CONFIG['remote'];archive=base/'package.zip';h=hashlib.sha256()
with archive.open('wb') as out:
 for part in CONFIG['parts']:
  p=base/part['file'];b=p.read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'];out.write(b);h.update(b)
assert h.hexdigest()==CONFIG['archive_sha256'] and archive.stat().st_size==CONFIG['archive_bytes']
payload=base/'payload';payload.mkdir();(base/'artifacts').parent.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  path=pathlib.PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
  target=payload/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
(payload/'acks').mkdir();ctx=base/'run_context.json';ctx.write_text(json.dumps(CONFIG['context']))
p=subprocess.Popen([sys.executable,'-B',str(payload/'driver.py'),'--run-context',str(ctx)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
for line in p.stdout:print(line,end='',flush=True)
print('DEEPFISH_DRIVER_EXIT '+str(p.wait()),flush=True)
