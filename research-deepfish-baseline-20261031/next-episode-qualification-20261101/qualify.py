"""Metadata-only second-episode qualification. No pixels, predictions, model or network."""
from portable_dev.paths import project_path as _portable_project_path
import collections,csv,datetime,hashlib,json,math,pathlib,re,os
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
ANN=R/'assets/annotation_docs/FISH/annotations'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(P/name).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def csvout(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prefix(im):
 m=re.fullmatch(r'(.+)_f(\d+)\.jpg',im['file_name']);assert m;return m.group(1)
def habitat(im):return im['file_name'].split('_')[0]
def frame(im):return int(re.search(r'_f(\d+)\.jpg$',im['file_name']).group(1))
freeze=read(P/'FREEZE.json');assert sha(P/'PROTOCOL.md')==freeze['protocol_sha256']
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(freeze['deadline_UTC'])
train=read(ANN/'train.json');old=read(ANN/'5_shot.json');dev=read(R/'followup-group-provenance-20261031/reports/DEV_SELECTION.json');members=read(R/'reports/FISH_archive_members.json');by_member={x['name']:x for x in members}
by_id={i['id']:i for i in train['images']};anns=collections.defaultdict(list)
for a in train['annotations']:anns[a['image_id']].append(a)
assert len(by_id)==3596 and len(train['annotations'])==12434
old_ids={i['id'] for i in old['images']};old_ps={prefix(i) for i in old['images']};dev_ps={g['prefix'] for g in dev['groups']};assert len(old_ps)==3 and len(dev_ps)==8 and not old_ps&dev_ps
used_ps=old_ps|dev_ps;used_h={x.split('_')[0] for x in used_ps};all_ps={prefix(i) for i in train['images']}
qa=read(R/'reports/PREVIEW_PLAN.json');qa_ps={prefix(x['image']) for x in qa};qa_train_ids={x['image']['id'] for x in qa if x['split']!='test'};qa_test=[{'split':'test','image_id':x['image']['id'],'file_name':x['image']['file_name'],'prefix':prefix(x['image'])} for x in qa if x['split']=='test']
# The later13-image QA consists only of the original5 support and fixed8 dev prefixes.
qa_ps|={i['prefix'] for i in dev['images']}|old_ps
qa_all_h={x.split('_')[0] for x in qa_ps}
archive_jsons=sorted(x['name'] for x in members if x['name'].endswith('.json'))
assert archive_jsons==['FISH/annotations/10_shot.json','FISH/annotations/1_shot.json','FISH/annotations/5_shot.json','FISH/annotations/test.json','FISH/annotations/train.json']
source_lists=[];official5=[]
for f in sorted(ANN.glob('*_shot.json'),key=lambda x:str(x.relative_to(R))):
 d=read(f);si={i['id']:i for i in d['images']};sa=collections.defaultdict(list)
 for a in d['annotations']:sa[a['image_id']].append(a)
 comparisons=[]
 for iid,im in si.items():
  ta=sorted(anns[iid],key=lambda a:a['id']);ss=sorted(sa[iid],key=lambda a:a['id']);missing=sorted({a['id'] for a in ta}-{a['id'] for a in ss});extra=sorted({a['id'] for a in ss}-{a['id'] for a in ta})
  comparisons.append({'image_id':iid,'file_name':im['file_name'],'prefix':prefix(im),'habitat_code':habitat(im),'train_annotation_ids':[a['id'] for a in ta],'shot_annotation_ids':[a['id'] for a in ss],'missing_train_annotation_ids':missing,'extra_shot_annotation_ids':extra,'all_annotation_fields_equal':ta==ss,'image_fields_equal':by_id[iid]==im})
 count5=len(si)==5 and len(d['annotations'])==5 and all(len(sa[i])==len(anns[i])==1 for i in si)
 complete=all(x['all_annotation_fields_equal'] and x['image_fields_equal'] for x in comparisons)
 ps={prefix(i) for i in si.values()};base_ok=count5 and complete and not ps&dev_ps and not set(si)&old_ids;strict_ok=base_ok and not ps&old_ps
 row={'path':str(f.relative_to(R)),'sha256':sha(f),'images':len(si),'boxes':len(d['annotations']),'is_complete_5image_5box_list':count5 and complete,'dev_prefix_overlap':sorted(ps&dev_ps),'old_support_image_overlap':sorted(set(si)&old_ids),'old_support_prefix_overlap':sorted(ps&old_ps),'strict_eligible':strict_ok,'relaxed_eligible':base_ok,'comparisons':comparisons};source_lists.append(row)
 if count5:official5.append(row)
write('OFFICIAL_LIST_QUALIFICATION.json',{'archive_json_members':archive_jsons,'official_lists_in_path_order':source_lists,'strict_qualified_5shot_lists':[x['path'] for x in official5 if x['strict_eligible']],'relaxed_qualified_5shot_lists':[x['path'] for x in official5 if x['relaxed_eligible']],'no_subset_of_10shot_or_union_of_1shot_used':True})
# Inventory prepared support copies by filename/hash; copies are not additional official episodes.
copies=[]
for base,dirs,files in os.walk(R):
 dirs[:]=[d for d in dirs if d not in {'.venv','.python','.git','__pycache__'}]
 for fn in files:
  if fn=='support5.json':
   f=pathlib.Path(base)/fn;copies.append({'path':str(f.relative_to(R)),'sha256':sha(f),'same_as_official_original_5shot':sha(f)==sha(ANN/'5_shot.json')})
write('PREPARED_SUPPORT_COPIES.json',sorted(copies,key=lambda x:x['path']))
assert not any(x['strict_eligible'] or x['relaxed_eligible'] for x in official5)
namespace=freeze['hash_namespace'];qualified=[];image_checks=[]
for iid,im in sorted(by_id.items()):
 aa=anns[iid];single=len(aa)==1;valid=single and aa[0]['category_id']==1 and aa[0].get('iscrowd',0)==0 and aa[0].get('ignore',0)==0
 base=valid and prefix(im) not in dev_ps and iid not in old_ids
 strict=base and prefix(im) not in old_ps
 key=hashlib.sha256((namespace+str(iid)).encode()).hexdigest()
 row={'image_id':iid,'file_name':im['file_name'],'prefix':prefix(im),'habitat_code':habitat(im),'train_box_count':len(aa),'one_fish_noncrowd_nonignore':bool(valid),'excluded_dev_prefix':prefix(im) in dev_ps,'excluded_old_support_image':iid in old_ids,'excluded_old_support_prefix':prefix(im) in old_ps,'strict_eligible':bool(strict),'relaxed_eligible':bool(base),'selection_sha256':key};image_checks.append(row)
 if strict:qualified.append(row)
csvout('ALL_TRAIN_IMAGE_QUALIFICATION.csv',image_checks)
assert len(qualified)>=5,'Strict tier cannot supply5; stop and report rather than change definition'
qualified.sort(key=lambda x:(x['selection_sha256'],x['image_id']));selected=qualified[:5];csvout('STRICT_ELIGIBLE_SINGLE_BOX_IMAGES.csv',qualified)
def enrich(row):
 iid=row['image_id'];im=by_id[iid];aa=anns[iid];m=by_member['FISH/train/'+im['file_name']];a=aa[0];x,y,w,h=a['bbox'];assert all(math.isfinite(v) for v in [x,y,w,h,a['area']]) and w>0 and h>0
 return {**row,'annotation_ids':[a['id'] for a in aa],'annotations_copied_from_full_train_without_changes':aa,'complete_relative_to_train':True,'visual_annotation_completeness':'not checked; no candidate pixels read','habitat_code_evidence':'official original slice_df_reg uses first filename token as habitat selector','habitat_name_GPS_deployment_video_mapping':'unknown','frame_number':frame(im),'image_size':[im['width'],im['height']],'archive_member':m['name'],'image_bytes':m['size'],'image_sha256_from_existing_archive_audit':m['sha256'],'bbox_boundary_overhang_px':max(0,-x,-y,x+w-im['width'],y+h-im['height']),'area_matches_xywh':abs(a['area']-w*h)<1e-5,'same_image_previously_in_train_QA':iid in qa_train_ids,'prefix_previously_in_any_train_or_test_QA':prefix(im) in qa_ps,'habitat_code_overlaps_old_support_or_dev':habitat(im) in used_h}
selected=[enrich(x) for x in selected];new_ps={x['prefix'] for x in selected};new_h={x['habitat_code'] for x in selected}
assert len({x['image_id'] for x in selected})==5 and sum(len(x['annotation_ids']) for x in selected)==5
old_hashes={by_member['FISH/train/'+i['file_name']]['sha256'] for i in old['images']};dev_hashes={i['image_sha256'] for i in dev['images']};new_hashes={x['image_sha256_from_existing_archive_audit'] for x in selected};assert len(new_hashes)==5 and not new_hashes&(old_hashes|dev_hashes)
write('PROPOSED_SELF_CONSTRUCTED_EPISODE.json',{'status':'METADATA_QUALIFIED_SELF_CONSTRUCTED_PROPOSAL_NOT_OFFICIAL_EPISODE','selector':freeze,'strict_candidate_images':len(qualified),'selected':selected,'image_count':5,'box_count':5,'prefix_count':len(new_ps),'habitat_code_count':len(new_h),'original_support_and_dev_filename_byte_hash_overlap':False,'active_support_files_modified':False,'new_COCO_training_file_written':False})
def inventory(excluded_p=None,excluded_h=None):
 keep=[im for im in train['images'] if prefix(im) not in (excluded_p or set()) and habitat(im) not in (excluded_h or set())]
 return {'prefix_count':len({prefix(im) for im in keep}),'habitat_code_count':len({habitat(im) for im in keep}),'images':len(keep),'boxes':sum(len(anns[im['id']]) for im in keep),'prefixes':sorted({prefix(im) for im in keep}),'habitat_codes':sorted({habitat(im) for im in keep})}
remaining={'train_total':inventory(),'exclude_current_support_and_dev_prefixes':inventory(used_ps),'exclude_current_used_and_proposed_support_prefixes':inventory(used_ps|new_ps),'also_exclude_all_previously_QA_seen_prefixes':inventory(used_ps|new_ps|qa_ps),'stronger_exclude_used_and_proposed_habitat_codes':inventory(excluded_h=used_h|new_h),'stronger_also_exclude_QA_habitat_codes':inventory(excluded_h=used_h|new_h|qa_all_h),'definitions':{'unused_for_effects':'no model training or dev-effect evaluation using that prefix in the current task','strict_QA_unseen':'also no prior QA image in train/test with that prefix','prefix_semantics':'file grouping only; independent video/site not established','habitat_code_semantics':'official numeric selector, not proven independent GPS/site/session'},'test_QA_disclosure':qa_test,'official_test_model_effects_used':False,'official_test_json_or_pixels_opened_this_check':False}
write('CONFIRMATION_POOL_INVENTORY.json',remaining)
prefix_rows=[]
for ps in sorted(all_ps):
 ims=[i for i in train['images'] if prefix(i)==ps]
 prefix_rows.append({'prefix':ps,'habitat_code':habitat(ims[0]),'images':len(ims),'boxes':sum(len(anns[i['id']]) for i in ims),'single_box_images':sum(len(anns[i['id']])==1 for i in ims),'current_training_prefix':ps in old_ps,'current_dev_prefix':ps in dev_ps,'proposed_support_prefix':ps in new_ps,'previous_QA_prefix_including_test':ps in qa_ps,'unused_by_current_and_proposed_train_dev':ps not in used_ps|new_ps,'strict_QA_unseen_confirmation_prefix':ps not in used_ps|new_ps|qa_ps,'habitat_code_used_by_current_or_proposed':habitat(ims[0]) in used_h|new_h,'frame_min':min(frame(i) for i in ims),'frame_max':max(frame(i) for i in ims)})
csvout('ALL_46_PREFIX_QUALIFICATION.csv',prefix_rows)
phases=read(R/'cpu-paired-AM-20261101/PHASES.json');end=read(R/'cpu-paired-AM-20261101/FINAL_STATUS.json');phase={x['label']:x['elapsed_seconds'] for x in phases};current_bytes=sum(f.stat().st_size for f in R.rglob('*') if f.is_file());new_checkpoint_bytes=2*692247962;cap=8*1024**3;margin=80*1024**2
cost={'observed_one_complete_pair_seconds':end['elapsed_seconds'],'observed_one_complete_pair_minutes':end['elapsed_seconds']/60,'updates_per_arm':200,'arms':2,'both_64dev_evaluated':True,'existing_environment_reused_no_install_time_in_this_runtime':True,'phase_seconds':phase,'phase_minutes':{k:v/60 for k,v in phase.items()},'supervision_gap_seconds':end['elapsed_seconds']-sum(phase.values()),'new_episode_runtime_not_measured':True,'changed_support_requires_new_A_and_M_not_reusing_old_A_as_control':True,'two_full_checkpoint_bytes_observed':new_checkpoint_bytes,'research_current_bytes':current_bytes,'research_cap_bytes':cap,'retained_margin_bytes':margin,'usable_free_bytes_before_new_metadata':cap-margin-current_bytes,'minimum_storage_shortfall_for_two_weights_alone':max(0,new_checkpoint_bytes-(cap-margin-current_bytes)),'additional_logs_predictions_metadata_also_required':True,'no_old_results_deleted_or_storage_limit_changed':True}
write('COST_FACTS.json',cost)
inputs=[ANN/'train.json',ANN/'5_shot.json',R/'reports/FISH_archive_members.json',R/'reports/PREVIEW_PLAN.json',R/'followup-group-provenance-20261031/reports/DEV_SELECTION.json',R/'followup-group-provenance-20261031/sources/official_metadata_evidence.json',R/'cpu-paired-AM-20261101/PHASES.json',R/'cpu-paired-AM-20261101/FINAL_STATUS.json']
write('INPUT_PINS.json',[{'path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in inputs])
now=datetime.datetime.now(datetime.timezone.utc);assert now<datetime.datetime.fromisoformat(freeze['deadline_UTC'])
write('VERIFICATION.json',{'status':'PASS','finished_UTC':now.isoformat(),'metadata_only':True,'candidate_images_read':0,'new_predictions_or_model_effects_read':0,'downloads_installs_uploads_new_weights_CU':0,'selected5_unique_image_IDs_and5_unique_bytes':True,'selected5_boxes_complete_relative_to_train':True,'selected_not_in_dev_or_old_support_prefixes':True,'hash_order_rule_reproduced':True,'no_official_split_or_active_support_file_modified':True,'official_test_no_new_access':True})
print(json.dumps({'selected':[{k:x[k] for k in ['image_id','file_name','prefix','habitat_code','annotation_ids','selection_sha256','prefix_previously_in_any_train_or_test_QA']} for x in selected],'strict_candidates':len(qualified),'remaining':{k:{a:b for a,b in v.items() if a in ['prefix_count','habitat_code_count','images','boxes']} for k,v in remaining.items() if isinstance(v,dict) and 'prefix_count' in v},'cost':cost},indent=2))
