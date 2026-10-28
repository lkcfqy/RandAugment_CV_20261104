"""Metadata-only candidate split and byte budget; does not fetch/decode any image."""
from pathlib import Path
import hashlib
import json

PROJECT=Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
RESEARCH=PROJECT/'causal-augmentation-research-20261020'
SOURCE=RESEARCH/'area-pairing-cpu-20261021/mechanism-transfer-pets-20261021'
OUT=Path(__file__).resolve().parent
index=json.loads((SOURCE/'zip-index-zip64.json').read_text())
entries=index['entries']
old_ids={p.stem for p in RESEARCH.rglob('*.jpg')}
rows=[]
counts={}
for line in (SOURCE/'archive-text-3.txt').read_text().splitlines():
    if not line or line.startswith('#'):continue
    name,breed,species,breed_index=line.split()
    if name in old_ids:continue
    key='images/'+name+'.jpg'
    if key not in entries:continue
    # Bound the proposal's per-photo transfer budget using stored ZIP metadata.
    if entries[key]['bytes']>512000:continue
    rows.append({'source_id':name,'class_id':int(breed)-1,'species':int(species)-1,
        'image_entry':key,'mask_entry':'annotations/trimaps/'+name+'.png',
        'order_hash':hashlib.sha256(('augmentation-location-candidate-v1:'+name).encode()).hexdigest()})
selected=[]
for c in range(37):
    pool=sorted([r for r in rows if r['class_id']==c],key=lambda r:r['order_hash'])
    counts[str(c)]=len(pool)
    assert len(pool)>=48,(c,len(pool))
    for i,r in enumerate(pool[:48]):
        r['candidate_partition']=('train','development','reserved_confirmation')[i//16]
        r['image_zip_metadata']=entries[r['image_entry']]
        r['mask_zip_metadata']=entries[r['mask_entry']]
        selected.append(r)
partitions={}
for part in ('train','development','reserved_confirmation'):
    group=[r for r in selected if r['candidate_partition']==part]
    partitions[part]={'photos':len(group),'photos_per_class':16,
        'image_uncompressed_bytes':sum(r['image_zip_metadata']['bytes'] for r in group),
        'image_compressed_payload_bytes':sum(r['image_zip_metadata']['compressed_bytes'] for r in group),
        'mask_compressed_payload_bytes':sum(r['mask_zip_metadata']['compressed_bytes'] for r in group)}
result={'status':'metadata_candidate_only_not_a_fitted_result_or_final_test_lock',
    'question':'Equal-parameter early versus late attention LoRA × RandAugment training benefit.',
    'source_revision':index['revision'],'source_archive_bytes':index['archive_bytes'],
    'source_index':str(SOURCE/'zip-index-zip64.json'),
    'source_index_sha256':hashlib.sha256((SOURCE/'zip-index-zip64.json').read_bytes()).hexdigest(),
    'official_split_used':'trainval only; no old final/test arrays or image pixels opened',
    'old_local_JPEG_ids_excluded':len(old_ids),'minimum_eligible_photos_per_class':min(counts.values()),
    'eligible_per_class':counts,'partitions':partitions,
    'new_payload_downloaded':False,'RGB_or_masks_decoded':False,
    'selection_uses_model_scores_or_old_outcomes':False,
    'before_training_required':'Check provenance, exact and near duplicates, target visibility and augmentation implementation; freeze final split and protocol before fitting.',
    'unit':'source photograph or pre-defined duplicate cluster, not augmented view or known animal identity',
    'pretraining_overlap_verified_absent':False,
    'network_overhead_or_range_request_time_in_payload_budget':False,'rows':selected}
p=OUT/'candidate-data-plan.json'
with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ('old_local_JPEG_ids_excluded','minimum_eligible_photos_per_class','partitions','new_payload_downloaded')},ensure_ascii=False))
