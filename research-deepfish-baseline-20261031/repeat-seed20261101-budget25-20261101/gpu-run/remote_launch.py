import pathlib,json,hashlib,zipfile,subprocess,sys,os
CONFIG={'remote': 'deepfish_resume25_5fa8916620', 'parts': [{'file': 'package_part0.bin', 'bytes': 6291456, 'sha256': 'fcbc8231836aa5f419b3ecdecb9127ec442e9487b54732bca32c4e6296d972d8'}, {'file': 'package_part1.bin', 'bytes': 5388943, 'sha256': '45148142c98cd649032a19bb787a030b46bdf232eb9700862bf40be514d88a63'}], 'archive_sha256': '5fa891662011ddb919ff0d390cd9de7e874d1afcdbc97756761b8abeb28c7350', 'archive_bytes': 11680399, 'context': {'authorization': {'upload': True, 'model_run': True}, 'allocation_started_epoch': 1793508870.168802, 'cu_per_hour_upper_bound': 1.15, 'quote_verified': True, 'external_release_guard_verified': True, 'persistent_storage_verified': True, 'persistence_mode': 'local_cli_sha256', 'persistence_probe': {'remote_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'local_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'bytes': 8388608, 'download_bytes_per_second': 4453855.0958829, 'verified_epoch': 1793508885.232935}, 'prior_cu': 1.2577442410187722, 'prior_gpu_seconds': 3718.752113674164, 'artifact_dir': '/content/deepfish_resume25_5fa8916620/artifacts', 'expected_manifest_sha256': '0072c8409541bedb69d5ba732896c5a5809e91407709963fd0975c19ebdfba63'}}
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
