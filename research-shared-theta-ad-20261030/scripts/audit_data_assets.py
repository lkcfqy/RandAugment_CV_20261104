"""Integrity, dimensions, source metadata and exact-duplicate audit; no model/scientific metrics."""
import collections,concurrent.futures,hashlib,json,pathlib,struct,time
from PIL import Image,ExifTags
ROOT=pathlib.Path(__file__).resolve().parents[1]
START=time.monotonic()
assert json.loads((ROOT/'reports/data-acquisition.json').read_text())['status']=='complete'
ENTRIES=json.loads((ROOT/'sources/aebad-selected-manifest.json').read_text())['entries']
WANTED_TAGS={'DateTime','DateTimeOriginal','DateTimeDigitized','Make','Model','BodySerialNumber','ImageUniqueID','ImageDescription','DocumentName','Artist','Software','Orientation'}
def audit(entry):
    path=ROOT/'data'/entry['path'];raw=path.read_bytes()
    assert len(raw)==entry['uncompressed_bytes']
    split,label,domain,name=pathlib.PurePosixPath(entry['path']).parts[2:]
    with Image.open(path) as im:
        im.load()
        fields={}
        exif=im.getexif()
        for tag,value in exif.items():
            tag_name=ExifTags.TAGS.get(tag,str(tag))
            if tag_name in WANTED_TAGS:fields[tag_name]=str(value)
        if 34665 in exif:
            for tag,value in exif.get_ifd(34665).items():
                tag_name=ExifTags.TAGS.get(tag,str(tag))
                if tag_name in WANTED_TAGS:fields[tag_name]=str(value)
        record={'path':entry['path'],'split':split,'label':label,'domain':domain,'bytes':len(raw),'sha256_raw':hashlib.sha256(raw).hexdigest(),'width':im.width,'height':im.height,'format':im.format,'mode':im.mode,'info_keys':sorted(im.info.keys()),'source_metadata':fields,'zip_crc32':entry['crc32']}
        # Actual camera files can be JPEG with a .png suffix; Pillow detects bytes.
        if split!='ground_truth':
            rgb=im.convert('RGB')
            record['sha256_decoded_rgb']=hashlib.sha256(struct.pack('>II',im.width,im.height)+rgb.tobytes()).hexdigest()
        return record
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for i,record in enumerate(pool.map(audit,ENTRIES)):
        records.append(record)
        if (i+1)%200==0:print(json.dumps({'files_checked':i+1,'seconds':round(time.monotonic()-START,2)}),flush=True)
(ROOT/'reports/file-manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
images=[x for x in records if x['split']!='ground_truth'];masks=[x for x in records if x['split']=='ground_truth']
assert len(images)==1483 and len(masks)==657
counts=collections.Counter((x['split'],x['domain'],'normal' if x['label']=='good' else 'anomaly') for x in images)
assert sum(v for k,v in counts.items() if k[0]=='train')==521
for key,num in [(('test','same','normal'),230),(('test','same','anomaly'),459),(('test','illumination','normal'),75),(('test','illumination','anomaly'),198)]:assert counts[key]==num
mask_map={(x['label'],x['domain'],pathlib.PurePosixPath(x['path']).name):x for x in masks}
missing_masks=[];mask_dim_mismatches=[]
for x in images:
    if x['split']=='test' and x['label']!='good':
        key=(x['label'],x['domain'],pathlib.PurePosixPath(x['path']).name);m=mask_map.get(key)
        if m is None:missing_masks.append(x['path'])
        elif (m['width'],m['height'])!=(x['width'],x['height']):mask_dim_mismatches.append({'image':x['path'],'image_size':[x['width'],x['height']],'mask_size':[m['width'],m['height']]})
def duplicates(hashkey):
    groups=collections.defaultdict(list)
    for x in images:groups[x[hashkey]].append(x)
    all_groups=[{'sha256':h,'paths':[x['path'] for x in g],'splits':sorted(set(x['split'] for x in g)),'domains':sorted(set(x['domain'] for x in g))} for h,g in groups.items() if len(g)>1]
    return {'all_duplicate_groups':all_groups,'cross_train_test_groups':[g for g in all_groups if set(g['splits'])=={'train','test'}],'within_train_groups':[g for g in all_groups if g['splits']==['train']],'within_test_groups':[g for g in all_groups if g['splits']==['test']]}
dups={'raw_file':duplicates('sha256_raw'),'decoded_rgb':duplicates('sha256_decoded_rgb'),'decoded_hash_definition':'SHA256(big-endian uint32 width || uint32 height || Pillow.convert(RGB).tobytes()); no resizing, EXIF orientation transform, or model computation','limitations':'Exact equality only. No claim about near-duplicates, shared physical blade identities, collection independence, or entity-disjoint splits.'}
(ROOT/'reports/duplicates.json').write_text(json.dumps(dups,indent=2))
metadata_counts=collections.Counter();metadata_values=collections.defaultdict(set);info_keys=collections.Counter()
for x in images:
    info_keys.update(x['info_keys'])
    for k,v in x['source_metadata'].items():metadata_counts[k]+=1;metadata_values[k].add(v)
meta={'image_info_key_counts':dict(info_keys),'recognized_source_field_counts':dict(metadata_counts),'recognized_field_unique_counts':{k:len(v) for k,v in metadata_values.items()},'recognized_field_values':{k:sorted(v) for k,v in metadata_values.items()},'official_entity_or_collection_batch_mapping_found':False,'basis':['Official repository tree contains code/config/documentation and visual examples, no entity/batch mapping','Full ZIP central directory contains images, masks, folders, macOS metadata; no CSV/JSON/entity mapping manifest','Official paper section 3.2 defines train/test domains and camera positions but supplies no per-image physical-blade identity or capture-session/batch mapping'],'do_not_infer':['IMG_ numeric filenames are file identifiers, not documented independent physical blade IDs','Folder names background/view/illumination/same are domain labels, not independent entity or collection-batch IDs','Camera metadata or timestamps, if present, do not establish distinct physical specimens'],'independent_entity_generalization':'not established from official released metadata; leave unresolved for protocol review'}
(ROOT/'reports/source-metadata-audit.json').write_text(json.dumps(meta,indent=2))
summary={'status':'passed file/decode/count checks; metadata independence unresolved','image_count':len(images),'mask_count':len(masks),'counts':{'/'.join(k):v for k,v in sorted(counts.items())},'image_dimensions':dict(collections.Counter(f"{x['width']}x{x['height']}" for x in images)),'mask_dimensions':dict(collections.Counter(f"{x['width']}x{x['height']}" for x in masks)),'image_formats':dict(collections.Counter(x['format'] for x in images)),'image_modes':dict(collections.Counter(x['mode'] for x in images)),'mask_formats':dict(collections.Counter(x['format'] for x in masks)),'missing_masks':missing_masks,'mask_dimension_mismatches':mask_dim_mismatches,'raw_cross_train_test_duplicate_groups':len(dups['raw_file']['cross_train_test_groups']),'rgb_cross_train_test_duplicate_groups':len(dups['decoded_rgb']['cross_train_test_groups']),'raw_all_duplicate_groups':len(dups['raw_file']['all_duplicate_groups']),'rgb_all_duplicate_groups':len(dups['decoded_rgb']['all_duplicate_groups']),'model_computations':0,'human_image_visualizations':0,'files_changed_or_removed_based_on_content':0,'elapsed_seconds':time.monotonic()-START}
(ROOT/'reports/data-integrity-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
