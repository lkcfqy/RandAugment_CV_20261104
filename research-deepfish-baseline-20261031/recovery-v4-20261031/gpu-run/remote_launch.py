import pathlib,json,hashlib,zipfile,subprocess,sys,os
CONFIG={'remote': 'deepfish_ab_dd122a9217', 'parts': [{'file': 'package_part0.bin', 'bytes': 6291456, 'sha256': '990f0191c48e7db135b1a9b73c9b8618ca7956531546762f1326d689b93342e6'}, {'file': 'package_part1.bin', 'bytes': 5387325, 'sha256': '0b98efe6a37c67eed033168158edd6fafd3701abd1a17d5587464e4d99aac72c'}], 'archive_sha256': 'dd122a9217321602d79f352a0165980b0b893a1bd1f824ddc64cdc68d2a10cd4', 'archive_bytes': 11678781, 'context': {'authorization': {'upload': True, 'model_run': True}, 'allocation_started_epoch': 1793457882.330535, 'cu_per_hour_upper_bound': 1.15, 'quote_verified': True, 'external_release_guard_verified': True, 'persistent_storage_verified': True, 'persistence_mode': 'local_cli_sha256', 'persistence_probe': {'remote_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'local_sha256': '127a2501e0268090b342668546bf7a0b4e903ee1dc3188454b57195c852bde3f', 'bytes': 8388608, 'download_bytes_per_second': 5337472.80465392, 'verified_epoch': 1793457896.910679}, 'prior_cu': 0.4724195386433071, 'prior_gpu_seconds': 1260.3443497161866, 'artifact_dir': '/content/deepfish_ab_dd122a9217/artifacts', 'expected_manifest_sha256': '587e82969a9be46c9ee31e9fa683564d07829eec041bd21f104f223030cf7243'}}
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
