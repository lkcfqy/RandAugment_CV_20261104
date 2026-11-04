from counting_assets import ROOT, dump, START
from pathlib import Path
import json,hashlib,time,collections,struct,math
import safetensors

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def blobsha(p):
    b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
checks=[]
for treefile,subdir,kind in [('sources/pairtally-tree.json','sources/pairtally','github'),('sources/countgdpp-github-tree.json','sources/countgdpp-github','github'),('sources/countgdpp-hf-tree-all.json','sources/countgdpp-hf','hf'),('sources/bert-hf-tree.json','models/bert-base-uncased','hf')]:
    tree=json.loads((ROOT/treefile).read_text());tree=tree['tree'] if kind=='github' else tree
    bypath={x['path']:x for x in tree}
    for p in sorted((ROOT/subdir).rglob('*')):
        if not p.is_file():continue
        rel=str(p.relative_to(ROOT/subdir));entry=bypath[rel]
        expected=entry.get('lfs',{}).get('oid',entry.get('sha',entry.get('oid')))
        actual=sha(p) if entry.get('lfs') else blobsha(p)
        checks.append({'path':str(p.relative_to(ROOT)),'expected_git_blob_or_lfs_oid':expected,'actual':actual,'matches':actual==expected})
assert all(x['matches'] for x in checks)
dump('reports/source_blob_verification.json',{'files_checked':len(checks),'all_match':True,'files':checks})
bert=ROOT/'models/bert-base-uncased/model.safetensors'
with bert.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];bh=json.loads(f.read(n))
with safetensors.safe_open(str(bert),framework='numpy') as f:
    assert set(f.keys())==set(bh)-{'__metadata__'}
    for k in f.keys():assert f.get_slice(k).get_shape()==bh[k]['shape']
dump('reports/bert_asset_verification.json',{'revision':'86b5e0934494bd15c9632b12f734a8a67f723594','weight_sha256':sha(bert),'weight_bytes':bert.stat().st_size,'physical_tensor_count':len(bh)-1,'bert_tensors':sum(k.startswith('bert.') for k in bh),'pretraining_head_tensors':sum(k.startswith('cls.') for k in bh),'safe_metadata_open':True,'model_constructed':False,'note':'Official legacy base BERT contains LayerNorm gamma/beta keys and 7 cls pretraining-head tensors. Standard transformers BertModel.from_pretrained handles legacy naming and ignores pretraining head. Main CountGD++ checkpoint has complete final BertModel embeddings/encoder/pooler keys; final strict load still required.','tensor_header':bh})

images=json.loads((ROOT/'reports/image_integrity.json').read_text())['images'];selected=json.loads((ROOT/'reports/selection_frozen.json').read_text())['images']
dup={}
for key in ['sha256','decoded_rgb_sha256']:
    g=collections.defaultdict(list)
    for x in images:g[x[key]].append(x['filename'])
    dup[key]=[v for v in g.values() if len(v)>1]
box_duplicates=[]
for x in selected:
    bs=x['A_boxes_xyxy']+x['B_boxes_xyxy']
    if len(set(tuple(b) for b in bs))!=6:box_duplicates.append(x['filename'])
dump('reports/selected_data_summary.json',{'images':94,'pairs':47,'all_selected_six_boxes_in_bounds':all(not x['box_oob'] for x in images),'all_selected_crc_verified':all(x['crc_verified'] for x in images),'raw_or_rgb_duplicate_groups':dup,'duplicate_box_images':box_duplicates,'image_byte_total':sum(x['bytes'] for x in images),'selected_source_counts_min':min(min(x['A_count'],x['B_count']) for x in selected),'selected_source_counts_max':max(max(x['A_count'],x['B_count']) for x in selected),'all_images_have_no_exif_orientation':all(x['exif_orientation'] is None for x in images),'point_geometry_warning':'Synthetic placeholder points; see point_semantics.json. Counts only.','visual_audit_scope':'8 fixed images in 4 flagged pairs; images/boxes not selected or excluded based on model effects. Other pairs have structural box-to-class mapping verified, but exhaustive semantic recount not claimed.'})
pins={'pairtally':{'repository':'https://github.com/bbvisual/PairTally_Benchmark','commit':'4a6dbcf916a75090f0c505ace9d3ee8ad5960cb1','tree_sha':'2b52597b0c6007d744ff52d70054ecce211b25dd','license':'CC BY 4.0; README says for academic research purposes; complete official LICENSE retained','archive_url':'https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view','archive_bytes':152874125,'archive_full_sha256':None,'range_download_only':True},'countgdpp':{'github':'https://github.com/niki-amini-naieni/CountGDPlusPlus','github_commit':'2ecbc4331d6730d017f52a4a97bbcbae7d5d714a','hf':'https://huggingface.co/nikigoli/CountGDPlusPlus','hf_revision':'06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1','license':'MIT root; copied GroundingDINO/DETR component Apache-2.0 notices preserved in source headers','weight_sha256':'94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5','weight_bytes':937560192},'bert':{'hf':'https://huggingface.co/google-bert/bert-base-uncased','revision':'86b5e0934494bd15c9632b12f734a8a67f723594','license':'Apache-2.0','weight_sha256':'68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3','weight_bytes':440449768}}
dump('reports/source_pins.json',pins)
assets=[]
for d in ['sources','models','data','scripts']:
    for p in sorted((ROOT/d).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:assets.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
dump('reports/asset_manifest.json',{'assets':assets,'full_archive_not_acquired':True,'full_archive_sha256_not_claimed':True})
critical=['sources/pairtally/dataset/pairtally_dataset/annotations/pairtally_annotations_simple.json','sources/pairtally/dataset/pairtally_dataset/annotations/image_metadata.json','sources/pairtally/dataset/pairtally_dataset/annotations/bbx_anno_valid.json','reports/selection_frozen.json','reports/image_integrity.json','reports/state_manifest.json','models/model.safetensors','models/bert-base-uncased/model.safetensors']
(ROOT/'reports/CRITICAL_SHA256SUMS.txt').write_text('\n'.join(sha(ROOT/p)+'  '+p for p in critical)+'\n')
ledger=json.loads((ROOT/'reports/download_ledger.json').read_text());total=sum(x.get('bytes_received',0) for x in ledger)
disk=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());allocated=sum(p.stat().st_blocks*512 for p in ROOT.rglob('*') if p.is_file())
resource={'download_response_body_bytes':total,'web_lookup_metadata_reserve_bytes':5*1024**2,'download_conservative_upper_bytes':total+5*1024**2,'download_limit_bytes':int(3.5*1024**3),'logical_new_disk_bytes_at_measurement':disk,'allocated_new_disk_bytes_at_measurement':allocated,'final_report_manifest_disk_reserve_bytes':10*1024**2,'new_disk_conservative_upper_bytes':max(disk,allocated)+10*1024**2,'disk_limit_bytes':6*1024**3,'started_unix':START,'measured_unix':time.time(),'elapsed_wall_minutes':(time.time()-START)/60,'asset_work_limit_minutes':45,'http_records':len(ledger),'http_record_errors':[x for x in ledger if x.get('error')],'forward_calls':0,'repository_model_constructed':0,'repository_python_modules_executed':0,'gpu_operations':0,'colab_allocations':0,'training_steps':0,'installed_packages':0,'new_environments':0,'torch_load_calls':0,'trust_remote_code_calls':0,'new_oauth':0,'purchases':0,'git_commit_push':0,'auto_review_denials':0,'uncertain_actions':0,'permission_note':'Initial sandbox DNS resolution unavailable. Approved require_escalated asset acquisition succeeded; no approval rejection, login or purchase.'}
assert resource['download_conservative_upper_bytes']<resource['download_limit_bytes'] and resource['new_disk_conservative_upper_bytes']<resource['disk_limit_bytes'] and resource['elapsed_wall_minutes']<45
dump('reports/resource_ledger.json',resource)
print(json.dumps({'verified_source_files':len(checks),'asset_files':len(assets),'download_GiB':total/1024**3,'new_disk_GiB':disk/1024**3,'wall_minutes':resource['elapsed_wall_minutes'],'selection_sha256':sha(ROOT/'reports/selection_frozen.json')},ensure_ascii=False))
