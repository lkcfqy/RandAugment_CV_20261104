import pathlib,json,hashlib,zipfile,subprocess,sys,os
CONFIG={'remote': 'deepfish_ab_f5b909f8e2', 'parts': [{'file': 'package_part0.bin', 'bytes': 6291456, 'sha256': 'df33973b23e9a2988d4704458a48404a06e7883bdced350204e68967a9cd0757'}, {'file': 'package_part1.bin', 'bytes': 5385700, 'sha256': 'b404bbb264215f6e41a119486d6ff3eb49ed084c5d3198d71f0d48f5bca2ab94'}], 'archive_sha256': 'f5b909f8e26326063c5ae250884842a4e1d71da56c9a9b0f45980713cd95166c', 'archive_bytes': 11677156, 'context': {'authorization': {'upload': True, 'model_run': True}, 'allocation_started_epoch': 1793455495.06963, 'cu_per_hour_upper_bound': 1.15, 'quote_verified': True, 'external_release_guard_verified': True, 'persistent_storage_verified': True, 'persistence_mode': 'local_cli_sha256', 'persistence_probe': {'remote_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'local_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'bytes': 8388608, 'download_bytes_per_second': 3597610.4085893077, 'verified_epoch': 1793455511.483704}, 'prior_cu': 0.2116088636362023, 'prior_gpu_seconds': 464.287675951004, 'artifact_dir': '/content/deepfish_ab_f5b909f8e2/artifacts', 'expected_manifest_sha256': 'fb3f9124841e9c1d514d60f79214dc03d8ee2afe0402a1b8011f352c337e75ad'}}
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
