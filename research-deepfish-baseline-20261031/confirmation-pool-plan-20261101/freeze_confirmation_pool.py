"""Metadata-only eligibility/protocol freeze; deliberately contains no model runner."""
from portable_dev.paths import project_path as _portable_project_path
import collections, csv, datetime, hashlib, json, os, pathlib, re, shutil, time

START = time.monotonic()
R = _portable_project_path(__file__, 'research-deepfish-baseline-20261031')
W = pathlib.Path('/Users/lkc/Documents/Codex/2026-10-31/task-8')
O = W / 'deepfish-confirmation-pool-plan-20261101'
O.mkdir(exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def dump(name,x): (O/name).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
pins={}
def read(rel):
    p=R/rel; b=p.read_bytes(); pins[rel]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}; return json.loads(b)
def table(name,rows):
    with (O/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def pref(name):return re.fullmatch(r'(.+)_f(\d+)\.jpg',name).group(1)
def frame(name):return int(re.fullmatch(r'(.+)_f(\d+)\.jpg',name).group(2))
inv=read('next-episode-qualification-20261101/CONFIRMATION_POOL_INVENTORY.json')
prior=inv['also_exclude_all_previously_QA_seen_prefixes']
dump('ELIGIBILITY_RULE_FROZEN_BEFORE_EXTRACTION.json',{
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'basis':'Reuse existing strict QA-unseen prefix inventory, all eligible train images; no visual or effect selection.',
 'prior_prefixes':prior['prefixes'],'expected_images':2014,'expected_boxes':7114,
 'order':'ascending original train image ID',
 'duplicate_rule':'Exclude any candidate sharing exact cached JPEG SHA256 with excluded train images or historical test QA. Retain and disclose internal candidate duplicate groups, if any; do not call them independent.',
 'near_frame_rule':'Exclude entire support/dev/known QA prefixes. Within retained prefixes include all frames, with complete frame-gap metadata; no new distance cutoff.',
 'pixels_opened':False,'models_run':False,'formal_evaluation_authorized':False})
train=read('assets/annotation_docs/FISH/annotations/train.json')
s0=read('assets/annotation_docs/FISH/annotations/5_shot.json')
s1=read('same-domain-decision-20261101/data/annotations/support1.json')
dev=read('repeat-seed20261101-budget25-20261101/payload/data/annotations/dev64.json')
preview=read('reports/PREVIEW_PLAN.json')
members=read('reports/FISH_archive_members.json')
audit=read('followup-group-provenance-20261031/reports/GROUP_AUDIT.json')
read('followup-group-provenance-20261031/sources/official_metadata_evidence.json')
for path,expected in [('assets/annotation_docs/FISH/annotations/train.json','d7bbd549b111df65fa76c33533dbdb8dfa396f688ad2b42509dd8c28596744a4'),('reports/FISH_archive_members.json','8fc32b1152b26152fcbcffe196ba6de09261004b4309ee18a7651bcbb779c540')]:assert pins[path]['sha256']==expected
mi={x['name']:x for x in members if x.get('sha256')}
ann=collections.defaultdict(list)
for a in train['annotations']:ann[a['image_id']].append(a)
ims={x['id']:x for x in train['images']}
assert len(ims)==3596 and len(train['annotations'])==12434
sets={'support0_prefix':{pref(x['file_name']) for x in s0['images']},'support1_prefix':{pref(x['file_name']) for x in s1['images']},'dev_prefix':{pref(x['file_name']) for x in dev['images']},'historical_QA_prefix':{pref(x['image']['file_name']) for x in preview}}
excluded_prefix=set.union(*sets.values())
all_prefix={pref(x['file_name']) for x in train['images']}
assert all_prefix-excluded_prefix==set(prior['prefixes'])
used_codes={p.split('_')[0] for p in excluded_prefix}
used_hashes=collections.defaultdict(list)
for im in train['images']:
    if pref(im['file_name']) in excluded_prefix:
        member='FISH/train/'+im['file_name'];used_hashes[mi[member]['sha256']].append(member)
for x in preview:
    if x['split']=='test':used_hashes[mi[x['member']]['sha256']].append(x['member'])
rows=[];selected=[];cross=[]
for im in sorted(train['images'],key=lambda x:x['id']):
    p=pref(im['file_name']); member='FISH/train/'+im['file_name'];m=mi[member]
    reasons=[k for k,ss in sets.items() if p in ss]
    if not reasons and m['sha256'] in used_hashes:
        reasons.append('exact_JPEG_duplicate_of_excluded');cross.append({'image_id':im['id'],'matches':used_hashes[m['sha256']]})
    ar=sorted(ann[im['id']],key=lambda a:a['id'])
    row={'image_id':im['id'],'file_name':im['file_name'],'prefix':p,'habitat_code':p.split('_')[0],'frame':frame(im['file_name']),'width':im['width'],'height':im['height'],'box_count':len(ar),'eligible':not reasons,'exclusion_reasons':';'.join(reasons),'archive_member':member,'jpeg_bytes':m['size'],'jpeg_sha256':m['sha256'],'annotation_ids':';'.join(str(a['id']) for a in ar),'annotations_sha256':hashlib.sha256(canon(ar)).hexdigest()}
    rows.append(row)
    if not reasons:selected.append(row)
ids=[x['image_id'] for x in selected]; idset=set(ids)
assert len(ids)==2014 and sum(x['box_count'] for x in selected)==7114, 'Objective identity change: stop before asserting prior pool.'
assert not idset&({x['id'] for x in s0['images']}|{x['id'] for x in s1['images']}|{x['id'] for x in dev['images']})
full_ann=[a for a in train['annotations'] if a['image_id'] in idset]
assert len({a['id'] for a in full_ann})==len(full_ann)
table('ALL_TRAIN_QUALIFICATION.csv',rows);table('CONFIRMATION_IMAGES.csv',selected)
dump('CONFIRMATION_IDS.json',{'split_namespace':'official train IDs','order':'ascending image_id','image_ids':ids,'annotation_ids':sorted(a['id'] for a in full_ann),'images':len(ids),'annotations':len(full_ann),'prefixes':prior['prefixes']})
dump('confirmation_annotations.json',{'images':[ims[i] for i in ids],'annotations':full_ann,'categories':train['categories']})
dump('EXCLUSIONS.json',{'excluded_images':len(rows)-len(selected),'excluded_boxes':sum(x['box_count'] for x in rows if not x['eligible']),'excluded_prefixes_by_reason':{k:sorted(v) for k,v in sets.items()},'counts_by_reason_overlap_allowed':{k:sum(k in x['exclusion_reasons'].split(';') for x in rows) for k in sets},'test_QA_disclosure':inv['test_QA_disclosure'],'test_not_opened_this_check':True,'all_46_partition':sorted(all_prefix)})
dup=collections.defaultdict(list)
for x in selected:dup[x['jpeg_sha256']].append(x['image_id'])
groups=[];gaps=[]
for p in sorted(prior['prefixes']):
    rs=sorted([x for x in selected if x['prefix']==p],key=lambda x:x['frame'])
    ds=[b['frame']-a['frame'] for a,b in zip(rs,rs[1:])]
    groups.append({'prefix':p,'habitat_code':p.split('_')[0],'habitat_code_seen_in_excluded_prefixes':p.split('_')[0] in used_codes,'images':len(rs),'boxes':sum(x['box_count'] for x in rs),'frame_min':rs[0]['frame'],'frame_max':rs[-1]['frame'],'adjacent_frame_pairs_delta1':ds.count(1),'min_positive_frame_gap':min(ds) if ds else None,'jpeg_bytes':sum(x['jpeg_bytes'] for x in rs)})
    gaps.extend({'prefix':p,'image_id_left':a['image_id'],'image_id_right':b['image_id'],'frame_left':a['frame'],'frame_right':b['frame'],'frame_delta':b['frame']-a['frame']} for a,b in zip(rs,rs[1:]))
table('PREFIX_INVENTORY.csv',groups);table('WITHIN_PREFIX_FRAME_GAPS.csv',gaps)
identity={'eligible_exact_duplicates_with_excluded':cross,'eligible_internal_exact_JPEG_duplicate_groups':[v for v in dup.values() if len(v)>1],'method':'Use existing archive member SHA256 manifest, not new image reads or near-duplicate visual hashes. Byte inequality does not establish visual independence.','same_prefix_adjacent_frame_pairs_delta1':sum(x['adjacent_frame_pairs_delta1'] for x in groups),'prefix_count':29,'habitat_codes':prior['habitat_codes'],'codes_overlapping_exposed_prefixes':sorted(set(prior['habitat_codes'])&used_codes),'known_semantics':audit['definitions'],'independent_video_site_habitat_claim':False,'cross_prefix_near_duplicate_or_video_mapping':'unknown; no reliable mapping available in existing provenance','stronger_unseen_numeric_codes_inventory_only':inv['stronger_also_exclude_QA_habitat_codes'],'stronger_subset_used_for_selection':False}
dump('IDENTITY_AND_GROUP_EVIDENCE.json',identity)

u='unadapted-reference-20261101';h='cpu-paired-AM-20261101';s='same-domain-decision-20261101'
specs=[('P0',u,'receipts/model_20261031_P0.json','assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth'),('A0',h,'evaluation/receipts/model_20261031_A.json',h+'/formal_A/final_model.pth'),('A1',s,'evaluation/receipts/model_20261031_A1.json',s+'/formal_A1/final_model.pth')]
run=read(u+'/RUN_PROTOCOL.json');oldproto=read(u+'/PROTOCOL.json');verification=read(u+'/FINAL_VERIFICATION.json')
phase=read(u+'/PHASES.json')
models=[];costrows=[];pred_max=0;pred_means=[]
for label,folder,receiptpath,checkpoint in specs:
    d=read(folder+'/'+receiptpath);lo=d['load'];state=lo['initial']['state_sha256'] if label=='P0' else lo['state_sha256'];cksha=lo['raw_checkpoint_sha256'] if label=='P0' else lo['checkpoint_sha256']
    assert d['forwards_completed']==64 and d['status']=='MODEL_COMPLETE' and d['official_test_used']==False
    assert d['environment']==read(u+'/receipts/model_20261031_P0.json')['environment']
    # CPU byte hashing only: no checkpoint deserialization/model initialization.
    assert sha(R/checkpoint)==cksha
    models.append({'label':label,'checkpoint':checkpoint,'bytes':(R/checkpoint).stat().st_size,'file_sha256':cksha,'complete_model_state_sha256':state,'tensor_count':908,'tensor_bytes':691880148,'reference_receipt':folder+'/'+receiptpath,'strict_loading_semantics':oldproto['strict_loading'] if label=='P0' else 'weights_only=True, strict=True, no missing/unexpected; all908 tensors byte-exact to complete checkpoint','seed':20261031,'formal_training_updates':0 if label=='P0' else 200})
    fixed=d['import_setup_seconds']+lo['seconds']; variable=d['elapsed_seconds']-fixed
    pred_dir=R/folder/('predictions' if label=='P0' else 'evaluation/predictions')
    arm='A' if label=='A0' else label
    sizes=[p.stat().st_size for p in pred_dir.glob('20261031_'+arm+'_c100_*.json')]
    assert len(sizes)==64
    pred_max=max(pred_max,max(sizes));pred_means.append(sum(sizes)/64)
    costrows.append({'model':label,'measured_images':64,'measured_elapsed_seconds':d['elapsed_seconds'],'measured_import_load_seconds':fixed,'measured_forward_seconds':d['forward_seconds'],'measured_pipeline_seconds':d['pipeline_seconds'],'measured_persistence_seconds':d['persistence_seconds'],'measured_peak_RSS_bytes':d['peak_RSS_bytes'],'measured_prediction_bytes':sum(sizes),'mean_prediction_bytes':sum(sizes)/64,'projected_all_images_seconds':fixed+variable*len(ids)/64,'projection_formula':'one import/load + (elapsed - import/load) * 2014/64; no repeated model loads'})
dump('MODEL_IDENTITIES.json',{'models':models,'P0_same_initial_state_as_A0_A1':run['baseline_initial_state'],'P0_raw_file_has_903_keys_active_source_900_model_908':True,'P0_fixed_scales_zero':7,'P0_training_only_random_DN_unused_inference':verification['DN_runtime_calls']==0,'P0_no_random_new_classifier':True,'weight_hashes_recomputed_without_deserialization':True})
for name,expect in run['code_sha256'].items():assert sha(R/u/name)==expect
config='fixed-M-execution-20261101/payload/config_A_reference.json'
assert sha(R/config)==run['models'][0]['config_sha256']
dump('IMPLEMENTATION_PINS.json',{'existing_code_paths_and_sha256':{u+'/'+k:v for k,v in run['code_sha256'].items()},'official_implementation_sha256':run['official_implementation_sha256'],'runtime_versions':run['core_runtime_versions'],'environment_lock_sha256':run['environment_lock_sha256'],'config':config,'config_sha256':run['models'][0]['config_sha256'],'actual_inference_loop_AST_equality_previously_verified':verification['same_actual_inference_loop_AST'],'no_new_runner_implemented':True})
metrics_map={'AP':'maxDets100','AP50':'maxDets1000','AP75':'maxDets1000','APsmall_medium_large':'maxDets1000','AR100':'maxDets100','AR300':'maxDets300','AR1000':'maxDets1000','ARsmall_medium_large':'maxDets1000'}
dump('EVALUATION_PROTOCOL.json',{'status':'FROZEN_PLAN_ONLY_PENDING_PARENT_GO','question':'Do existing CPU-trained A0/A1 gains over unadapted P0 after fixed5images/U200 persist on previously unexplored eligible train images? No new algorithm or M branch.','models_fixed':['P0','A0','A1'],'model_order':['P0','A0','A1'],'image_order':'ascending original train imageID, all2014','unique_forwards':6042,'input_condition':'original, Color(prob=1,level=0,min_mag=1,max_mag=1) identity after cv2 uint8 BGR decode before FixScaleResize(800,1333), then original LoadAnnotations and PackDetInputs','actual_prompt':'fish. ','max_queries_saved_per_image':300,'threshold_or_NMS_change':False,'CPU':{'torch_threads':4,'interop_threads':1,'opencv_threads':1,'dtype':'float32','autocast':False,'eval':True,'inference_mode':True,'models_resident_simultaneously':1},'loading':'MODEL_IDENTITIES.json; weights_only; complete state hashes; no unsafe fallback','primary':'pool AP IoU.50:.05:.95 maxDets100; paired differences A0-P0 and A1-P0 in percentage points; neither selects the other','descriptive':'all12 original COCO metrics and complete 29-prefix AP@100/differences, including negative/zero/no-valid-GT results; no favorable group selection','metric_mapping':metrics_map,'required_complete_identities':'all6042 unique model/image records durably saved+hash-verified before efficacy evaluation; incomplete matrix cannot substitute a completed subset','metric_recompute':'recompute from persisted predictions with pinned pycocotools; preserve precision/recall/scores arrays; model processes exited before sequential metric jobs','GT':'full original train annotations retained verbatim (xywh,area,category,crowd/ignore); no relabel/clipping or extra training labels','validity_limits':['not independent scenes/videos/habitats','prefixes are filename groups; 8 of14 numeric habitat selectors overlap exposed prefixes','frames within prefix correlated; no image-iid significance tests or independent-sample claim','protocol planned after exploratory A model selection, before any pool model output; not original preregistration','official test excluded entirely; 3test images previously QA, their prefixes excluded'],'failure_handling':'objective corrupt/invalid assets or budget interruption -> incomplete and report; do not replace/select samples, change strength/prompt/checkpoint, or reduce sample count','formal_run_GO':False,'new_train_install_upload_GPU_MPS_Colab_CU':0})

measured_metric=max(x['elapsed_seconds'] for x in phase if x['label'] in ['metrics','recompute'])
# Both aggregate and disjoint-prefix sums process the same N images; two passes include full recomputation.
metric_projected=measured_metric*(len(ids)/64)*3*2*2
forward_projected=sum(x['projected_all_images_seconds'] for x in costrows)
prep_allowance=240;manifest_allowance=60
base=forward_projected+metric_projected+prep_allowance+manifest_allowance
peak=max(x['measured_peak_RSS_bytes'] for x in costrows)
inputbytes=sum(x['jpeg_bytes'] for x in selected)
output_linear=len(ids)*sum(pred_means)
output_budget=int(1.25*pred_max*6042+32*1024**2)
research_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
disk=shutil.disk_usage(R)
cost={'status':'projection_from_same_CPU64dev_not_a_full_pool_measurement','models':costrows,'images_per_model':2014,'forwards':6042,'input_dimensions_metadata':dict(collections.Counter(str((x['width'],x['height'])) for x in selected)),'forward_and_record_projected_seconds':forward_projected,'metrics_and_full_recompute_projected_seconds':metric_projected,'metric_projection_formula':'max(observed64dev metrics,recompute) * (2014/64) *3models *2[aggregate+all disjoint prefixes] *2[compute+recompute]; variable GT density/size and perprefix overhead unmeasured','planning_only_archive_extract_verify_allowance_seconds':prep_allowance,'planning_only_final_manifest_allowance_seconds':manifest_allowance,'base_seconds':base,'with_25pct_margin_seconds':base*1.25,'with_25pct_margin_hours':base*1.25/3600,'three_model_loads_only':True,'observed_max_RSS_bytes':peak,'inference_memory_estimate_bytes':int(peak*1.25+1024**3),'metric_memory_unmeasured_planning_GiB':6,'proposed_hard_RSS_GiB':12,'memory_method':'one model/one metric job at a time, stream perimage records; 6GiB planning allowance is not measured fullpool peak','prediction_bytes_linear_projection':output_linear,'prediction_bytes_observed_max_per_image':pred_max,'outputs_budget_bytes':output_budget,'outputs_budget_formula':'1.25*max observed per-image JSON bytes*6042 +32MiB for metric arrays, provenance, manifests/logs; no checkpoint copies or rendered images','all_pool_JPEG_bytes_if_future_extracted_once':inputbytes,'future_total_added_bytes_allowance':inputbytes+output_budget,'current_research_bytes':research_bytes,'research_cap_bytes':12*1024**3,'retain_margin_bytes':80*1024**2,'free_under_research_cap_after_allowances':12*1024**3-80*1024**2-research_bytes-inputbytes-output_budget,'filesystem_available_bytes':disk.free,'uncertainties':['sustained multi-hour Mac load/thermal throughput','all-pool GT density and COCO Python allocations','archive extraction/hash and group-evaluation time allowances not newly benchmarked','future orchestration needs minimal matrix-size/path parameter adaptation; no new runner today'],'exceeds_previous_three_hour_run_cap':base*1.25>3*3600,'no_subset_or_budget_change_authorized':True}
dump('COST_ESTIMATE.json',cost)
dump('INPUT_PINS.json',pins)
dump('VERIFICATION.json',{'status':'PASS_METADATA_PLAN_ONLY','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'images':len(ids),'annotations':len(full_ann),'prefixes':len(groups),'excluded_images':len(rows)-len(selected),'full_original_annotations_retained':all(a in train['annotations'] for a in full_ann),'all_original_train_images_partitioned':len(rows)==len(ims),'candidate_pixels_read':0,'models_loaded_or_called':0,'new_predictions':0,'test_JSON_or_pixels_opened':False,'existing_archive_SHA_metadata_used':True,'cross_used_exact_duplicates':len(cross),'internal_exact_duplicate_groups':sum(len(v)>1 for v in dup.values()),'known_QA_prefixes_excluded':True,'reference_model_processes_previously_exited':verification['processes_exited'],'new_CU':0,'metadata_script_seconds':time.monotonic()-START,'connection_event':'Two read-command creation failures before recovery; exec-server transport disconnected then executor key changed. No failed command executed; connected environment restored and original receipts readable. No policy denial.'})
print(json.dumps({'staging':str(O),'images':len(ids),'boxes':len(full_ann),'prefixes':len(groups),'exact_duplicate_groups':identity['eligible_internal_exact_JPEG_duplicate_groups'],'adjacent_pairs':identity['same_prefix_adjacent_frame_pairs_delta1'],'overlap_codes':identity['codes_overlapping_exposed_prefixes'],'cost':cost},ensure_ascii=False))
