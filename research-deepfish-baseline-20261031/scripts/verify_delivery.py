"""Verify delivery integrity only; no library/model loads and no network."""
import ast,collections,datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
checks=[]
def check(label,ok):
 checks.append({'check':label,'passed':bool(ok)})
 if not ok:raise AssertionError(label)
m=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
for a in [m['data'],m['weight']]:
 p=ROOT/a.get('file',a.get('path'));check('exact_size:'+p.name,p.stat().st_size==a['bytes']);check('sha256:'+p.name,digest(p)==a['sha256'])
for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
check('own_script_syntax_memory_only',True)
for name in ['Domain-RAG','CDFSOD-benchmark']:
 d=json.loads((ROOT/'reports'/f'{name}-source-manifest.json').read_text())
 errors=[]
 for r in d['files']:
  p=ROOT/r['path']
  if p.stat().st_size!=r['size'] or digest(p)!=r['sha256']:errors.append(r['path'])
 check('source_snapshot_unchanged:'+name,not errors)
archive_rows=json.loads((ROOT/'reports/FISH_archive_members.json').read_text());names=[pathlib.PurePosixPath(x['name']).as_posix() for x in archive_rows]
check('unique_normalized_archive_names',len(names)==len(set(names)))
annroot=ROOT/'assets/annotation_docs/FISH/annotations'
for s,v in m['data']['splits'].items():
 p=annroot/(s+'.json');d=json.loads(p.read_text());check('annotation_hash:'+s,digest(p)==v['annotation_file_sha256']);check('annotation_counts:'+s,len(d['images'])==v['images'] and len(d['annotations'])==v['annotations'])
 for im in d['images']:
  name='FISH/'+('test' if s=='test' else 'train')+'/'+im['file_name']
  if name not in set(names):raise AssertionError(name)
check('all_annotation_images_in_archive',True)
files={r['name']:r for r in archive_rows if r['type']=='file'}
for p in (ROOT/'preview/originals').rglob('*.jpg'):
 member=str(p.relative_to(ROOT/'preview/originals'));check('preview_exact_bytes:'+member,digest(p)==files[member]['sha256'])
wt=json.loads((ROOT/'reports/WEIGHT_CHECK.json').read_text());check('recorded_safe_weights_inspection',wt['status']=='inspected' and wt['weights_only'] and wt['all_finite'] and wt['tensor_count']==903 and not wt['unsafe_fallback'] and not wt['model_instantiated'])
configs=json.loads((ROOT/'reports/static_resolved_configs.json').read_text())
for shot in [1,5,10]:
 c=configs[f'static_FISH_{shot}shot_swin_t']['config'];check(f'static_FISH_{shot}_constraints',c['train_cfg']['max_epochs']==12 and c['train_dataloader']['batch_size']==4 and c['train_dataloader']['dataset']['ann_file']==f'annotations/{shot}_shot.json' and c['model']['backbone']['embed_dims']==96)
check('no_partially_downloaded_data_or_weights',not list((ROOT/'assets').glob('*.partial')))
ledger=[json.loads(x) for x in (ROOT/'reports/download_ledger.jsonl').read_text().splitlines()];dl=sum(x['bytes_received'] for x in ledger)
check('local_recorded_download_payload_under_2GiB',dl<2*1024**3)
# 32 MiB reserve is conservative for the small connector requests used in this audit, not measured network wire bytes.
check('download_payload_plus_32MiB_metadata_reserve_under_2GiB',dl+32*1024**2<2*1024**3)
disk=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());check('task_directory_under_4GiB',disk<4*1024**3)
for name in ['README_ZH.md','DATA_MANIFEST.md','reports/CONFIG_AUDIT_ZH.md','reports/WEIGHT_CHECK_ZH.md','reports/NEXT_CUDA_CHECK_ZH.md','reports/VISUAL_REVIEW_ZH.md','reports/OPERATION_LOG_ZH.md']:
 check('report_present:'+name,(ROOT/name).is_file())
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'asset_delivery_integrity_verified','checks':checks,'checks_count':len(checks),'recorded_curl_payload_bytes':dl,'recorded_curl_payload_GiB':dl/1024**3,'directory_file_bytes':disk,'directory_file_GiB':disk/1024**3,'model_execution':False,'training':False,'GPU':False,'paper_reproduction_claim':False}
(ROOT/'reports/FINAL_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
