import pathlib,json,hashlib,zipfile,subprocess,sys,os
CONFIG={'remote': 'deepfish_repeat_392b5c9e3f', 'parts': [{'file': 'package_part0.bin', 'bytes': 6291456, 'sha256': 'fcb0d37ebf148c78cfb8df00d25125433df7a41a96d1a50f0b2ad6e844a05880'}, {'file': 'package_part1.bin', 'bytes': 5388184, 'sha256': '2649b8b1f4eab2507523ec4b093ba78f0ca02a0dfc364906e9cff8aa8f3cac58'}], 'archive_sha256': '392b5c9e3f8675c2d06e67b9e2a1633300fd42765dc6c2148752734cd25af6c9', 'archive_bytes': 11679640, 'context': {'authorization': {'upload': True, 'model_run': True}, 'allocation_started_epoch': 1793463261.398519, 'cu_per_hour_upper_bound': 1.15, 'quote_verified': True, 'external_release_guard_verified': True, 'persistent_storage_verified': True, 'persistence_mode': 'local_cli_sha256', 'persistence_probe': {'remote_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'local_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'bytes': 8388608, 'download_bytes_per_second': 2354867.827203123, 'verified_epoch': 1793463280.4864619}, 'prior_cu': 1.1001052667892774, 'prior_gpu_seconds': 3225.2735856513978, 'artifact_dir': '/content/deepfish_repeat_392b5c9e3f/artifacts', 'expected_manifest_sha256': 'e9eebbfe8c59dff9e0672af6e0b4fa95aee6cd8248ca60072159a1e8073a5005'}}
base=pathlib.Path('/content')/CONFIG['remote'];marker=base/'launch_entered.json';marker.write_text(json.dumps({'status':'LAUNCH_ENTERED','archive_sha256':CONFIG['archive_sha256']}));print('DEEPFISH_LAUNCH_ENTERED '+CONFIG['archive_sha256'],flush=True);archive=base/'package.zip';h=hashlib.sha256()
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
