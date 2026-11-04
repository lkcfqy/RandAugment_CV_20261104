"""Tar inventory and hashes. Copy only annotation/docs; never extract links or execute code."""
import collections,hashlib,json,pathlib,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
archive=ROOT/'assets/FISH.tar.gz'
rows=[];seen=set();total=0;notes=[]
with tarfile.open(archive,'r|gz') as tf:
 for m in tf:
  path=pathlib.PurePosixPath(m.name)
  if path.is_absolute() or '..' in path.parts or m.issym() or m.islnk() or not (m.isfile() or m.isdir()): raise ValueError('Unsafe member '+m.name)
  if m.name in seen: raise ValueError('Duplicate member '+m.name)
  seen.add(m.name);total+=m.size
  if len(seen)>30000 or total>2*1024**3 or m.size>100*1024**2: raise ValueError('Uncompressed safety limit')
  row={'name':m.name,'size':m.size,'type':'file' if m.isfile() else 'directory'}
  if m.isfile():
   h=hashlib.sha256();keep=path.suffix.lower()=='.json' or path.name.lower().startswith(('license','readme','copying','notice'))
   chunks=[] if keep else None
   f=tf.extractfile(m)
   while True:
    b=f.read(1024**2)
    if not b:break
    h.update(b)
    if keep:chunks.append(b)
   row['sha256']=h.hexdigest()
   if keep:
    dst=ROOT/'assets/annotation_docs'/path;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b''.join(chunks));row['copied_to']=str(dst.relative_to(ROOT));notes.append(m.name)
  rows.append(row)
(ROOT/'reports/FISH_archive_members.json').write_text(json.dumps(rows,indent=2))
summary={'safe_member_checks_passed':True,'file_count':sum(r['type']=='file' for r in rows),'member_count':len(rows),'uncompressed_bytes':total,'copied_annotations_or_docs':notes,'extensions':dict(collections.Counter(pathlib.PurePosixPath(r['name']).suffix.lower() for r in rows if r['type']=='file')),'full_images_extracted':False}
(ROOT/'reports/FISH_archive_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
