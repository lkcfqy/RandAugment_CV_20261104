"""Final local integrity and data-contract verification; no upstream imports, models or test semantics."""
import ast,collections,datetime,hashlib,json,math,pathlib,re,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];ASSET=pathlib.Path(sys.argv[1]);checks=[]
def check(name,ok,detail=None):
 checks.append({'check':name,'pass':bool(ok),'detail':detail})
 if not ok:raise AssertionError(name)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(1024**2),b''):h.update(part)
 return h.hexdigest()
old=json.loads((ASSET/'DELIVERY_MANIFEST.json').read_text());errors=[]
for row in old['files']:
 p=ASSET/row['path']
 if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:errors.append(row['path'])
check('all_previous_delivery_files_unchanged_by_byte_hash',not errors,{'files':len(old['files']),'errors':errors,'test_semantic_content_not_parsed':True})
dev=json.loads((ROOT/'reports/DEV_SELECTION.json').read_text());stats=json.loads((ROOT/'reports/GROUP_AUDIT.json').read_text())
ann=ASSET/'assets/annotation_docs/FISH/annotations';train=json.loads((ann/'train.json').read_text());support=json.loads((ann/'5_shot.json').read_text())
check('source_train_hash',sha(ann/'train.json')==dev['source_train_sha256'])
check('source_5shot_hash',sha(ann/'5_shot.json')==dev['source_5_shot_sha256'])
images={i['id']:i for i in train['images']};anns=collections.defaultdict(list)
for a in train['annotations']:anns[a['image_id']].append(a)
pref=lambda name:re.sub(r'_f\d+\.jpg$','',name)
dev_ids={i['image_id'] for i in dev['images']};sup_ids={i['id'] for i in support['images']}
check('fixed_official_5shot_ids',sup_ids=={930,1892,1900,2244,2363})
check('dev_count_64_and_unique',len(dev['images'])==len(dev_ids)==64)
check('dev_box_budget_187',sum(len(anns[i]) for i in dev_ids)==187)
check('support_box_budget_5',sum(len(anns[i]) for i in sup_ids)==5)
check('dev_support_id_disjoint',not(dev_ids&sup_ids))
check('prefix_disjoint',not({pref(images[i]['file_name']) for i in dev_ids}&{pref(images[i]['file_name']) for i in sup_ids}))
check('file_hash_disjoint',not({i['image_sha256'] for i in dev['images']}&{i['image_sha256'] for i in dev['support']}))
check('shared_habitat_reported',dev['dev_support_habitat_overlap']==['7463'])
eligible={pref(i['file_name']) for i in train['images']}-{pref(i['file_name']) for i in support['images']}
expected=sorted(eligible,key=lambda p:(hashlib.sha256(('deepfish-prefix-dev-v1\n'+p).encode()).hexdigest(),p))[:8]
check('chosen_groups_are_first_eight_fixed_hashes',[g['prefix'] for g in dev['groups']]==expected)
check('eight_distinct_frames_per_group',all(len(g['selected_image_ids'])==len(set(g['selected_image_ids']))==8 for g in dev['groups']))
check('manifest_annotations_reference_original_records',all(sorted(a['id'] for a in anns[i['image_id']])==i['annotation_ids'] for i in dev['images']+dev['support']))
check('numeric_contract_finite_positive',all(all(math.isfinite(x) for x in a['bbox']+[a['area']]) and a['bbox'][2]>0 and a['bbox'][3]>0 and abs(a['area']-a['bbox'][2]*a['bbox'][3])<1e-5 and a['category_id']==1 for i in dev_ids|sup_ids for a in anns[i]))
check('declared_selected_dimensions',all((images[i]['width'],images[i]['height'])==(1920,1080) for i in dev_ids|sup_ids))
geo=json.loads((ROOT/'reports/SELECTED_GEOMETRY_CHECK.json').read_text());check('boundary_exceptions_exact',[(a['annotation_id'],a['max_boundary_excess_px']) for a in geo['boundary_excesses']]==[(4195,0.5),(4660,0.5)])
decoded=json.loads((ROOT/'reports/DECODED_CONTRACT_SAMPLES.json').read_text());check('13_samples_37_boxes_decoded',len(decoded)==13 and sum(i['boxes'] for i in decoded)==37)
check('decoded_original_files_match_published_hashes',all(sha(ROOT/'preview/originals'/i['file_name'])==i['sha256'] for i in decoded))
check('no_test_sample_decoded',all(i['role'] in ('support','dev') for i in decoded))
for shot,prefix_n,img_n,box_n,hab_n,hab_img_n,hab_box_n in [(1,45,3531,12369,19,3517,12355),(5,43,3305,12122,17,2831,10533),(10,39,3018,11645,14,2663,10792)]:
 s=stats['shots'][str(shot)];a=s['remaining_train_excluding_support_prefixes'];b=s['remaining_train_excluding_support_habitats']
 check(f'{shot}shot_remaining_counts',(a['prefix_groups'],a['images'],a['boxes'],b['habitat_codes'],b['images'],b['boxes'])==(prefix_n,img_n,box_n,hab_n,hab_img_n,hab_box_n))
template=(ROOT/'templates/two_arm_config.py.in').read_text();ast.parse(template)
check('template_has_no_official_test_json','annotations/test.json' not in template and 'annotations/dev64.json' in template)
check('template_suppresses_training_validation','val_dataloader = None' in template and 'val_evaluator = None' in template)
check('template_disables_general_checkpoint_hook','default_hooks = dict(checkpoint=None)' in template)
render=lambda mag,wd:template.replace('__TOTAL_UPDATES__','25').replace('__COLOR_MAG__',str(mag)).replace('__WORK_DIR__',repr(wd))
trees=[ast.parse(render(mag,wd)) for mag,wd in [(1.0,'work/A'),(0.8,'work/B')]]
assignments=[]
for t in trees:
 assignments.append({n.targets[0].id:ast.dump(n.value) for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)})
diff=[k for k in assignments[0] if assignments[0][k]!=assignments[1][k]]
check('two_rendered_template_differences_only_color_and_directory',set(diff)=={'color_mag','work_dir'},diff)
parsed=[]
for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text());parsed.append(p.name)
check('local_audit_scripts_parse_without_execution',True,parsed)
json_count=0
for p in ROOT.rglob('*.json'):
 if p.name not in ('DELIVERY_MANIFEST.json','FINAL_VERIFICATION.json'):json.loads(p.read_text());json_count+=1
check('json_artifacts_parse',True,json_count)
sizes=json.loads((ROOT/'reports/PLANNED_INPUT_SIZES.json').read_text());check('planned_bytes_consistent',sizes['exact_data_bytes_before_config_manifest_overhead']==11783016 and sizes['exact_full_input_bytes_before_code_config_manifest_overhead']==1144854702)
fetch=json.loads((ROOT/'reports/PUBLIC_FETCH_LOG.json').read_text());direct=sum(x.get('bytes',0) for x in fetch);upper=1484698643+direct+64*1024**2
check('download_budget_with_64MiB_metadata_allowance',upper<2*1024**3,{'previous_recorded_payload':1484698643,'new_direct_http_payload':direct,'metadata_allowance':64*1024**2,'conservative_total':upper,'limit':2*1024**3})
old_bytes=sum(p.stat().st_size for p in ASSET.rglob('*') if p.is_file() and ROOT not in p.parents)
new_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
check('research_disk_budget',old_bytes+new_bytes+1024**2<4*1024**3,{'prior_bytes':old_bytes,'new_bytes_before_final_manifest':new_bytes,'limit':4*1024**3})
out={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','checks':checks,'checks_passed':len(checks),'execution_limits':{'upstream_code_executed':False,'model_instantiated':False,'gpu_used':False,'dependencies_installed':False,'data_uploaded':False,'test_semantic_content_reopened':False},'not_verified':['runtime config resolution','complete dependency solving/imports','CUDA ABI, model construction, throughput or efficacy','all-image annotation completeness','detection annotation derivation or full prefix-video-site mapping']}
(ROOT/'reports/FINAL_VERIFICATION.json').write_text(json.dumps(out,indent=2))
manifest=[]
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and p.name!='DELIVERY_MANIFEST.json':manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
(ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps({'verified_utc':out['verified_utc'],'file_count':len(manifest),'file_bytes_without_manifest':sum(p['bytes'] for p in manifest),'files':manifest},indent=2))
print(json.dumps({'status':'PASS','checks_passed':len(checks),'previous_files_hash_checked':len(old['files']),'new_files_hashed':len(manifest),'new_file_bytes_without_manifest':sum(p['bytes'] for p in manifest)},indent=2))
