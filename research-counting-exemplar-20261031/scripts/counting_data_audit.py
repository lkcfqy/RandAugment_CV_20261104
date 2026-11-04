"""Metadata/ZIP-header audit and deterministic selection only; no models."""
from counting_assets import ROOT, dump
import json, hashlib, collections, io, zipfile, struct, math, csv
r=ROOT/'sources/pairtally/dataset/pairtally_dataset/annotations'
a=json.loads((r/'pairtally_annotations_simple.json').read_text())
m=json.loads((r/'image_metadata.json').read_text())
intra=json.loads((r/'pairtally_annotations_intra_simple.json').read_text())
inter=json.loads((r/'pairtally_annotations_inter_simple.json').read_text())
split=json.loads((r/'pairtally_splits_simple.json').read_text())['test']
bbx=json.loads((r/'bbx_anno_valid.json').read_text())
assert set(a)==set(m)==set(split) and len(split)==len(set(split))==681
assert set(intra).isdisjoint(inter) and set(intra)|set(inter)==set(a)
errors=[];groups=collections.defaultdict(list)
def rect(b):
    assert len(b)==4 and all(len(p)==2 for p in b)
    x1,y1=b[0];x2,y2=b[2]
    assert b==[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    assert all(math.isfinite(x) for p in b for x in p) and x2>x1>=0 and y2>y1>=0
    return [x1,y1,x2,y2]
for filename,d in m.items():
    item=a[filename]; pair_id='_'.join([d['super_category'],d['test_type'],d['positive_code'],d['negative_code']])
    assert filename.startswith(pair_id+'_')
    assert (filename in intra)==(d['test_type']=='INTRA')
    assert len(item['points'])==d['positive_count'] and len(item['negative_points'])==d['negative_count']
    assert item['positive_prompt']==d['positive_prompt'] and item['negative_prompt']==d['negative_prompt']
    assert d['original_filename'] in bbx
    assert len(item['box_examples_coordinates'])==len(item['negative_box_exemples_coordinates'])==3
    A=[rect(b) for b in item['box_examples_coordinates']];B=[rect(b) for b in item['negative_box_exemples_coordinates']]
    assert A==bbx[d['original_filename']]['pos'] and B==bbx[d['original_filename']]['neg']
    assert all(len(p)==2 and all(math.isfinite(x) for x in p) for f in ['points','negative_points'] for p in item[f])
    entry={'filename':filename,'pair_id':pair_id,'original_filename':d['original_filename'],'selection_sha256':hashlib.sha256(('PairTally-observation-v1/'+filename).encode('utf-8')).hexdigest(),'A_prompt_source':item['positive_prompt'],'B_prompt_source':item['negative_prompt'],'A_count':len(item['points']),'B_count':len(item['negative_points']),'A_boxes_xyxy':A,'B_boxes_xyxy':B,'source_annotation_sha256':hashlib.sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'original_names':[d['components']['obj1_name'],d['components']['obj2_name']]}
    groups[pair_id].append(entry)
assert len(groups)==97
pairs=[];selected=[]
for key,entries in sorted(groups.items()):
    entries.sort(key=lambda x:(x['selection_sha256'],x['filename']))
    is_intra=m[entries[0]['filename']]['test_type']=='INTRA'
    pairs.append({'pair_id':key,'test_type':'INTRA' if is_intra else 'INTER','image_count':len(entries),'source_prompt_pairs':sorted(set((e['A_prompt_source'],e['B_prompt_source']) for e in entries)),'ranked_candidates':entries,'selected_filenames':[e['filename'] for e in entries[:2]] if is_intra else []})
    if is_intra:selected.extend(entries[:2])
assert len(selected)==94
cd=(ROOT/'sources/pairtally-zip-central.bin').read_bytes();eocd=struct.unpack_from('<4s4H2IH',cd,cd.rfind(b'PK\x05\x06'));cd_start=eocd[6]
assert eocd[1:3]==(0,0) and eocd[3]==eocd[4]==682
with zipfile.ZipFile(io.BytesIO(cd)) as z:
    infos=sorted(z.infolist(),key=lambda x:x.header_offset)
    listing=[]
    for i,x in enumerate(infos):
        assert not x.filename.startswith('/') and '..' not in x.filename.split('/') and '\\' not in x.filename
        assert ((x.external_attr>>16)&0o170000)!=0o120000 and not (x.flag_bits&1)
        assert x.compress_type in (0,8) and x.file_size <30_000_000
        offset=x.header_offset+cd_start
        end=(infos[i+1].header_offset+cd_start if i+1<len(infos) else cd_start)-1
        listing.append({'filename':x.filename,'basename':x.filename.rsplit('/',1)[-1],'is_dir':x.is_dir(),'uncompressed_bytes':x.file_size,'compressed_bytes':x.compress_size,'compression_method':x.compress_type,'crc32':f'{x.CRC:08x}','header_offset':offset,'range_end':end,'flag_bits':x.flag_bits})
assert len(set(x['filename'] for x in listing))==len(listing)
files=[x for x in listing if not x['is_dir']]
assert len(files)==681 and len(set(x['basename'] for x in files))==681
assert set(x['basename'] for x in files)==set(a)
byname={x['basename']:x for x in files}
for x in selected:x['zip_entry']=byname[x['filename']]
dump('reports/archive_manifest.json',{'archive_size':cd_start+eocd[5]+22,'full_archive_downloaded':False,'full_archive_sha256':None,'reason_no_full_hash':'Only central directory and selected byte ranges are downloaded; per-image SHA256 and ZIP CRC32 verified separately.','central_directory_sha256':hashlib.sha256(cd).hexdigest(),'entries':listing})
dump('reports/pair_inventory.json',pairs)
selection={'rule':"Per INTRA pair, sort SHA256(UTF8('PairTally-observation-v1/' + complete official simplified image filename including .jpg)); take first two. Ties by full filename.",'source_pairtally_commit':'4a6dbcf916a75090f0c505ace9d3ee8ad5960cb1','pair_key':'super_category + test_type + positive_code + negative_code (ordered official A/B).','selected_image_count':len(selected),'selected_pair_count':47,'all_selected_are_pre_effect':True,'image_or_effect_based_exclusion':False,'common_name_status':'Separate candidate table, not yet frozen; image selection and all six box identities frozen.','images':selected}
dump('reports/selection_frozen.json',selection)
(ROOT/'reports/selection_frozen.sha256').write_text(hashlib.sha256((ROOT/'reports/selection_frozen.json').read_bytes()).hexdigest()+'  selection_frozen.json\n')
with (ROOT/'reports/selection_frozen.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['pair_id','filename','selection_sha256','A_prompt_source','B_prompt_source','A_count','B_count']);w.writeheader();w.writerows({k:x[k] for k in w.fieldnames} for x in selected)
dump('reports/metadata_audit.json',{'all_images':681,'intra_images':313,'inter_images':368,'intra_pairs':47,'inter_pairs':50,'all_counts_equal_point_lengths':True,'all_six_boxes_present':True,'all_box_polygons_axis_aligned_valid':True,'all_six_boxes_equal_original_filename_bbx_table':True,'original_filename_mapping_unique':len(set(x['original_filename'] for x in m.values()))==len(m),'selected_image_count':94,'max_queries_under_three_modes_and_AB':564,'all_intra_pair_size_distribution':dict(collections.Counter(len(v) for k,v in groups.items() if '_INTRA_' in k)),'selected_equal_count_images':[x['filename'] for x in selected if x['A_count']==x['B_count']],'errors':errors})
print(json.dumps({'pairs':97,'selected':94,'selected_compressed_bytes':sum(x['zip_entry']['compressed_bytes'] for x in selected),'metadata_checks':'pass','archive_entries':len(listing)}))
