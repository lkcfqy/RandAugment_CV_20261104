"""Measure future input categories without copying a training bundle, uploading, or changing labels."""
import hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];ASSET=pathlib.Path(sys.argv[1])
ANN=ASSET/'assets/annotation_docs/FISH/annotations'
dev=json.loads((ROOT/'reports/DEV_SELECTION.json').read_text());train=json.loads((ANN/'train.json').read_text());ids={r['image_id'] for r in dev['images']}
subset={'images':[i for i in train['images'] if i['id'] in ids],'annotations':[a for a in train['annotations'] if a['image_id'] in ids],'categories':train['categories']}
encoded=(json.dumps(subset,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
bert=json.loads((ASSET/'reports/EXISTING_BERT_ASSETS.json').read_text())
rows=[{'category':'69 original JPEGs: support5 + dev64','bytes':dev['total_raw_jpeg_bytes_support_and_dev'],'status':'archive bytes measured; only13 QA originals copied'},
 {'category':'official 5_shot.json','bytes':(ANN/'5_shot.json').stat().st_size,'status':'existing unchanged'},
 {'category':'dev COCO subset JSON','bytes':len(encoded),'sha256_if_materialized':hashlib.sha256(encoded).hexdigest(),'status':'measured in memory only; original records selected unchanged, not written as training labels'},
 {'category':'original converted Swin-T checkpoint','bytes':(ASSET/'assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth').stat().st_size,'status':'existing verified public asset'},
 {'category':'BERT local asset candidate, including license and README','bytes':sum(r['size'] for r in bert['files']),'status':'existing seven files; offline model construction not verified'}]
report={'upload_authorized':False,'uploaded':False,'new_experiment_not_covered_by_old_counting_upload_permission':True,'categories':rows,'exact_data_bytes_before_config_manifest_overhead':sum(r['bytes'] for r in rows[:3]),'exact_full_input_bytes_before_code_config_manifest_overhead':sum(r['bytes'] for r in rows),'planned_local_config_manifest_overhead_ceiling_bytes':1024**2,'public_implementation':'open-mmlab/mmdetection@44ebd17b145c2372c4b700bfb9cb20dbd28ab64a; fetch in a separately authorized environment, not full Domain-RAG upload','excluded':['official test images/labels','full FISH.tar.gz','unselected train images/labels','old counting research files/results','credentials','Flux/retrieval assets','another MM-Grounding-DINO checkpoint']}
(ROOT/'reports/PLANNED_INPUT_SIZES.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
