"""Seal the complete local result; no model execution."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import json,hashlib,datetime,time
P=Path(__file__).resolve().parent
PRE=_portable_project_path(__file__, 'augmentation-selection-cifar100-preflight-20261102')
def load(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
run=load(P/'RUN_RECEIPT.json');lim=load(P/'LIMITS.json');ver=load(P/'FINAL_VERIFICATION.json')
assert run['status']=='COMPLETE' and ver['status']=='PASS'
assert time.time()<lim['deadline_epoch']
assert not (P/'MANIFEST_SHA256.json').exists() and not (P/'TASK_RECEIPT.json').exists()
pre_bytes=sum(p.stat().st_size for p in PRE.rglob('*') if p.is_file())
assert pre_bytes==27992579
payload=sum(p.stat().st_size for p in P.rglob('*') if p.is_file())
receipt={'status':'COMPLETE_ALL_FROZEN_SETTINGS','prepared_from_utc':'2026-11-02T05:36:42Z','sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cumulative_wall_seconds_at_seal':time.time()-lim['original_preparation_start_epoch'],'cumulative_wall_limit_seconds':5400,'hard_task_deadline_utc':'2026-11-02T07:06:42Z','global_research_deadline_utc':'2026-11-02T10:00:00Z','budget_reset':False,'dataset':'CIFAR-100 all100 fine classes','query_originals':9990,'support_union_originals':8751,'episodes':12,'per_class_B':[8,16],'formal_head_fits':720,'choices_jointly_sealed_before_query':48,'formal_image_forwards':177498,'preflight_image_forwards':192,'preflight_synthetic_ridge_calls':12,'all_query_predictions_saved_and_reprojected':7192800,'all_oof_predictions_saved_and_reprojected':172800,'final_verification':'PASS','same_model_state_hash_before_after':True,'peak_model_worker_rss_bytes':ver['peak_worker_rss_bytes'],'model_processes_exited':True,'preflight_artifact_bytes':pre_bytes,'formal_payload_bytes_excluding_this_receipt_and_manifest':payload,'combined_payload_MiB_before_final_metadata':(payload+pre_bytes)/2**20,'artifact_limit_bytes':524288000,'artifact_limit_headroom_before_final_metadata_bytes':524288000-payload-pre_bytes,'model_weights_or_raw_images_copied':False,'new_installs':0,'new_downloads':0,'new_uploads':0,'new_GPU_MPS_Colab_allocations':0,'new_CU':0,'permission_denials':0,'presentation_changes':0,'science_changes_after_query':0,'missing_settings':0,'fourth_round_started':False,'preflight_manifest_sha256':sha(PRE/'MANIFEST_SHA256.json'),'protocol_sha256':lim['protocol_sha256'],'identity_sha256':lim['identity_sha256'],'results_sha256':sha(P/'RESULTS.json'),'outcome':'C>H and tie-bound excess in all4; DeiTB8 stage-majority and DeiT budget-space decrease falsified. Limited empirical record, not reliable paper contribution.','remaining_blockers':[]}
assert payload+pre_bytes+1024**2<524288000
save('TASK_RECEIPT.json',receipt)
files=[{'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(P.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json']
save('MANIFEST_SHA256.json',{'schema':1,'excludes':['MANIFEST_SHA256.json'],'files':files})
for e in files:assert sha(P/e['path'])==e['sha256'] and (P/e['path']).stat().st_size==e['bytes']
actual=sum(p.stat().st_size for p in P.rglob('*') if p.is_file());assert actual+pre_bytes<=524288000
print(json.dumps({'status':'SEALED','file_count_including_manifest':len(files)+1,'formal_bytes':actual,'combined_bytes':actual+pre_bytes,'combined_MiB':(actual+pre_bytes)/2**20,'cumulative_wall_minutes_at_seal':receipt['cumulative_wall_seconds_at_seal']/60,'manifest_sha256':sha(P/'MANIFEST_SHA256.json')},indent=2))
