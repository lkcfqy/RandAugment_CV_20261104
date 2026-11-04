import tarfile, pathlib, json, hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
for name in ['Domain-RAG','CDFSOD-benchmark']:
 sha=json.loads((ROOT/'sources'/f'{name}-commit.json').read_text())['sha']
 archive=ROOT/'sources'/f'{name}-{sha}.tar.gz'; out=ROOT/'sources'/name
 rows=[]; total=0
 with tarfile.open(archive,'r:gz') as tf:
  for m in tf:
   p=pathlib.PurePosixPath(m.name)
   if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk() or not (m.isdir() or m.isfile()): raise ValueError('Unsafe archive member '+m.name)
   total+=m.size
   if total>200*1024**2: raise ValueError('Source uncompressed budget')
   if not m.isfile(): continue
   dst=out.joinpath(*p.parts[1:]); dst.parent.mkdir(parents=True,exist_ok=True)
   b=tf.extractfile(m).read(); dst.write_bytes(b)
   rows.append(dict(path=str(dst.relative_to(ROOT)),size=len(b),sha256=hashlib.sha256(b).hexdigest()))
 (ROOT/'reports'/f'{name}-source-manifest.json').write_text(json.dumps(dict(commit=sha,archive=str(archive.relative_to(ROOT)),files=rows,total_bytes=total),indent=2))
 print(name,sha,len(rows),total)
