import pathlib,json,hashlib,zipfile,subprocess,sys,os
CONFIG={'remote': 'deepfish_ab_fafab1e349', 'parts': [{'file': 'package_part0.bin', 'bytes': 6291456, 'sha256': '4ffe1046722de12dde5d662ece965e24f0065d634d9ef501ae29fc9544eb9fe2'}, {'file': 'package_part1.bin', 'bytes': 5386158, 'sha256': '630e164bc1001b5a6f9eacb8d1015ee36501eed19bfd196fb67cc015a918e0bf'}], 'archive_sha256': 'fafab1e3494703df035e4c5720dd9fbf06b93fe9406278bc8988e7775c613ad2', 'archive_bytes': 11677614, 'context': {'authorization': {'upload': True, 'model_run': True}, 'allocation_started_epoch': 1793456636.942037, 'cu_per_hour_upper_bound': 1.15, 'quote_verified': True, 'external_release_guard_verified': True, 'persistent_storage_verified': True, 'persistence_mode': 'local_cli_sha256', 'persistence_probe': {'remote_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'local_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'bytes': 8388608, 'download_bytes_per_second': 4481728.847282624, 'verified_epoch': 1793456651.924607}, 'prior_cu': 0.3172815439545578, 'prior_gpu_seconds': 795.0891099910737, 'artifact_dir': '/content/deepfish_ab_fafab1e349/artifacts', 'expected_manifest_sha256': '96e685d9dc08f59fc0055710b9360eaf57d3b15d1b6a4b6924348aa5a1db6c21'}}
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
