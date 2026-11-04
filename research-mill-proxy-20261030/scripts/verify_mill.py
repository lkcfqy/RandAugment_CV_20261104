"""Archive integrity, identity, fixed panel, and exact duplicate audit; no metrics."""
import collections
import csv
import hashlib
import io
import json
import re
import resource
import stat
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from PIL import Image

ROOT=Path(sys.argv[1]).resolve()
START=time.monotonic()
started=datetime.now(timezone.utc).isoformat()
ARCHIVE=ROOT/'downloads/MILLs-DSLR.zip'
assert hashlib.file_digest(ARCHIVE.open('rb'),'sha256').hexdigest() == 'cf14eaf7ac287a30822fd626cba76b255f39e416b95bcc089e48c7aa91503112'

def write_json(rel,obj):
    p=ROOT/rel
    p.write_text(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    return hashlib.sha256(p.read_bytes()).hexdigest()

rows=[]
scene_files=collections.defaultdict(dict)
expected_missing='MILLs-DSLR/test/input/Test13-9.png'
with ZipFile(ARCHIVE) as z:
    infos=z.infolist()
    names=[i.filename for i in infos]
    assert len(names)==len(set(names)), 'Duplicate archive paths'
    total=sum(i.file_size for i in infos)
    assert total <= 512*1024**2
    for i in infos:
        p=PurePosixPath(i.filename)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
        assert p.parts[0]=='MILLs-DSLR'
        assert not stat.S_ISLNK(i.external_attr >> 16), 'Archive symlink'
        assert not i.flag_bits & 1, 'Encrypted member'
        assert i.file_size <= 2*1024**2, 'Oversized member'
        assert i.file_size <= max(1,i.compress_size)*100, 'Compression bomb ratio'
        if i.is_dir(): continue
        assert len(p.parts)==4 and p.parts[1] in ('train','validation','test') and p.parts[2] in ('input','gt') and p.suffix=='.png'
        split,kind=p.parts[1:3]
        if kind=='input':
            match=re.fullmatch(r'(.+)-(10|[1-9])\.png',p.name)
            assert match, p.name
            scene,level=match.group(1),int(match.group(2))
        else:
            scene,level=p.stem,None
        key=(split,scene)
        tag='GT' if level is None else 'L'+str(level)
        assert tag not in scene_files[key]
        scene_files[key][tag]=i.filename
        rows.append(dict(archive_path=i.filename,split=split,scene_id=scene,kind=kind,level=level,archive_bytes=i.file_size,zip_crc32=f'{i.CRC:08x}',tag=tag))
    train_scenes=sorted(s for split,s in scene_files if split=='train')
    test_scenes=sorted(s for split,s in scene_files if split=='test')
    validation_scenes=sorted(s for split,s in scene_files if split=='validation')
    assert (len(train_scenes),len(validation_scenes),len(test_scenes))==(30,12,8)
    assert not set(train_scenes)&set(test_scenes)
    ranks=sorted((hashlib.sha256(('MILL-proxy-v1/'+s).encode('utf-8')).hexdigest(),s) for s in train_scenes)
    chosen=[s for h,s in ranks[:6]]
    panel_paths=set()
    for split,scenes in [('train',chosen),('test',test_scenes)]:
        for s in scenes:
            for tag in ('L10','L1','L3','L5','L7','GT'):
                panel_paths.add(scene_files[(split,s)][tag])
    assert len(panel_paths)==84 and expected_missing not in panel_paths
    missing=[]
    for key,files in sorted(scene_files.items()):
        for tag in ['GT']+['L'+str(i) for i in range(1,11)]:
            if tag not in files:
                split,s=key
                missing.append(f'MILLs-DSLR/{split}/input/{s}-{tag[1:]}.png')
    assert missing==[expected_missing], missing
    for row in sorted(rows,key=lambda x:x['archive_path']):
        # ZipFile.read verifies CRC; validation pixels are never decoded.
        content=z.read(row['archive_path'])
        assert content[:8]==b'\x89PNG\r\n\x1a\n' and content[12:16]==b'IHDR'
        w,h,bits,color,comp,filt,interlace=struct.unpack('>IIBBBBB',content[16:29])
        row.update(width=w,height=h,png_bit_depth=bits,png_color_type=color,png_interlace=interlace,raw_sha256=hashlib.sha256(content).hexdigest(),decoded_rgb_sha256=None,decoded=False,extracted=row['archive_path'] in panel_paths)
        assert (w,h,bits,color)==(600,400,8,2), (row['archive_path'],w,h,bits,color)
        if row['split'] in ('train','test'):
            with Image.open(io.BytesIO(content)) as im:
                im.load()
                assert im.size==(w,h) and im.mode=='RGB'
                rgb=im.tobytes()
                # Fixed explicit domain and dimensions, then row-major RGB8 bytes.
                row['decoded_rgb_sha256']=hashlib.sha256(b'RGB8\0'+struct.pack('>II',w,h)+rgb).hexdigest()
                row['decoded']=True
        if row['extracted']:
            dest=ROOT/'data'/row['archive_path']
            assert dest.resolve().is_relative_to((ROOT/'data').resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists(): assert hashlib.sha256(dest.read_bytes()).hexdigest()==row['raw_sha256']
            else: dest.write_bytes(content)

rows.sort(key=lambda r:r['archive_path'])
by_path={r['archive_path']:r for r in rows}
panel=[]
for split,scenes in [('train',chosen),('test',test_scenes)]:
    for scene in scenes:
        for tag in ('L10','L1','L3','L5','L7','GT'):
            row=by_path[scene_files[(split,scene)][tag]]
            panel.append({k:row[k] for k in ['split','scene_id','level','tag','archive_path','raw_sha256','decoded_rgb_sha256','width','height','png_bit_depth']}|{'role':'bright_gt' if tag=='GT' else 'anchor' if tag=='L10' else 'real_light','local_path':'data/'+row['archive_path']})
panel_hash=write_json('reports/panel.json',panel)
manifest_hash=write_json('reports/archive_manifest.json',rows)
with (ROOT/'reports/panel.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(panel[0]));writer.writeheader();writer.writerows(panel)
write_json('reports/train_scene_selection.json',[dict(rank=i+1,scene_id=s,selection_sha256=h,selected=i<6) for i,(h,s) in enumerate(ranks)])
write_json('reports/scene_manifest.json',[dict(split=split,scene_id=s,files=files,selected=(split=='train' and s in chosen) or split=='test') for (split,s),files in sorted(scene_files.items())])
duplicate_audit={}
for field in ('raw_sha256','decoded_rgb_sha256'):
    groups=collections.defaultdict(list)
    for row in rows:
        if row['split'] in ('train','test'): groups[row[field]].append(row)
    cross=[dict(sha256=ha,paths=[x['archive_path'] for x in group]) for ha,group in groups.items() if {x['split'] for x in group}=={'train','test'}]
    within=[dict(sha256=ha,paths=[x['archive_path'] for x in group]) for ha,group in groups.items() if len(group)>1]
    duplicate_audit[field]=dict(train_test_exact_duplicates=cross,all_train_test_duplicate_groups=within)
write_json('reports/duplicate_audit.json',duplicate_audit)
summary=dict(status='ASSET_PREPARATION_ONLY',archive_entries=len(infos),png_count=len(rows),uncompressed_archive_bytes=total,split_counts={s:dict(scenes=len({r['scene_id'] for r in rows if r['split']==s}),input=sum(r['split']==s and r['kind']=='input' for r in rows),gt=sum(r['split']==s and r['kind']=='gt' for r in rows)) for s in ['train','validation','test']},selected_train_scenes=chosen,test_scenes=test_scenes,panel_images=len(panel),panel_sha256=panel_hash,archive_manifest_sha256=manifest_hash,panel_extracted_bytes=sum(r['archive_bytes'] for r in rows if r['extracted']),all_image_dimensions=[600,400],png_bits_per_channel=8,png_color_type=2,decoded_train_test_images=sum(r['decoded'] for r in rows),validation_pixels_decoded=0,missing_known_bad=[expected_missing],unexpected_missing=[],extraction_safety='All entries checked before extraction; no traversal, symlinks, duplicate paths, encryption or size violations; every file CRC verified; only fixed panel extracted',raw_hash_definition='SHA256 of original PNG bytes, not camera RAW sensor files',decoded_hash_definition='SHA256(b"RGB8\\0" + struct.pack(">II",width,height) + row_major_RGB_uint8_bytes)',scene_id_definition='Remove only final -<1..10> from input stems, preserve _a/_b; GT stem must match exactly; split is archive directory, never filename prefix',cross_split_raw_duplicate_count=len(duplicate_audit['raw_sha256']['train_test_exact_duplicates']),cross_split_decoded_duplicate_count=len(duplicate_audit['decoded_rgb_sha256']['train_test_exact_duplicates']),started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-START,process_cpu_seconds=time.process_time(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_forward_count=0,metrics_computed=0,gpu_seconds=0)
write_json('reports/data_verification.json',summary)
(ROOT/'reports/panel.sha256').write_text(panel_hash+'  panel.json\n')
print(json.dumps(summary,indent=2))
