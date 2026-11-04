"""Audit saved outputs and write provenance/resource ledger; no model imports."""
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(sys.argv[1]).resolve()
WORKSPACE=Path('/Users/lkc/Documents/Codex/2026-10-31/task-6')
def read(rel):return json.loads((ROOT/rel).read_text())
def save(rel,obj):(ROOT/rel).write_text(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

data=read('reports/data_verification.json');model=read('reports/model_cpu_verification.json');pins=read('reports/source_pins.json')
panel=read('reports/panel.json');manifest=read('reports/archive_manifest.json');logs=read('logs/downloads.json')
assert sha(ROOT/'reports/panel.json')==data['panel_sha256']
assert sha(ROOT/'reports/archive_manifest.json')==data['archive_manifest_sha256']
assert len(panel)==84 and len({x['local_path'] for x in panel})==84
assert sum(x['role']=='bright_gt' for x in panel)==14
assert sum(x['role']=='anchor' for x in panel)==14
assert sum(x['role']=='real_light' for x in panel)==56
for x in panel:assert sha(ROOT/x['local_path'])==x['raw_sha256']
with ZipFile(ROOT/'downloads/MILLs-DSLR.zip') as z:
    ns=set(z.namelist())
    gt_train=sorted(Path(x).stem for x in ns if x.startswith('MILLs-DSLR/train/gt/') and x.endswith('.png'))
    chosen=sorted(gt_train,key=lambda s:hashlib.sha256(('MILL-proxy-v1/'+s).encode()).digest())[:6]
    assert chosen==data['selected_train_scenes']
    assert 'Scene10_a' in gt_train and 'Scene10_b' in gt_train
    assert 'Scene23_a' in gt_train and 'Scene23_b' in gt_train
    assert 'MILLs-DSLR/test/input/Test13-9.png' not in ns
    assert all(x['archive_path'] in ns for x in panel)
    for x in panel:
        stem=Path(x['archive_path']).stem
        scene=stem if x['role']=='bright_gt' else stem.rpartition('-')[0]
        assert scene==x['scene_id']
assert read('sources/mill_hf_metadata.json')['sha']==pins['data_revision']
assert read('sources/hvi_hf_metadata.json')['sha']==pins['model_revision']
for tree_path,target,expected in [('sources/mill_hf_tree.json','MILLs-DSLR.zip',pins['dataset_sha256']),('sources/hvi_hf_tree.json','model.safetensors',pins['model_sha256'])]:
    record=next(x for x in read(tree_path) if x['path']==target)
    assert record['lfs']['oid']==expected
assert model['strict'] and model['devices']==['cpu'] and not model['missing_keys'] and not model['unexpected_keys']
assert data['model_forward_count']==model['model_forward_count']==0
assert data['metrics_computed']==model['metrics_computed']==0
assert data['cross_split_raw_duplicate_count']==data['cross_split_decoded_duplicate_count']==0
snapshots=[]
for rec in logs:
    if rec['status'] not in ('downloaded','verified'):continue
    p=ROOT/rec['path']
    assert p.is_file() and sha(p)==rec['sha256']
    snapshots.append({k:rec[k] for k in ['url','path','bytes_received','sha256','status']})
save('reports/source_manifest.json',snapshots)
save('reports/license_review.json',{
 'MILL_data':{'declaration':'MIT','basis':'sources/MILL_DATA_CARD.md frontmatter and sources/mill_hf_metadata.json cardData','fixed_revision':pins['data_revision']},
 'MILL_repository_code':{'license_assessed':False,'research_code_downloaded':False,'imported':False,'executed':False,'MIT_from_data_not_inferred':True},
 'HVI_code':{'declaration':'MIT','copyright':'2024 Fediory Feng','snapshot':'sources/HVI_LICENSE','copied_notice':'vendor/HVI-CIDNet/LICENSE','commit':pins['hvi_github_commit']},
 'HVI_weights':{'declaration':'MIT','basis':'sources/HVI_MODEL_CARD.md frontmatter','revision':pins['model_revision']},
 'source_review':{'reviewed_in_full_before_import':['net/CIDNet.py','net/HVI_transform.py','net/LCA.py','net/transformer_utils.py'],'git_blob_verification':True,'repository_install_scripts_executed':False,'evaluation_scripts_executed':False},
 'new_dependency':{'name':'einops','version':'0.6.1','basis':'author requirements.txt and PyPI JSON','registry':'https://pypi.org','wheel_sha256':'99149e46cc808956b174932fe563d920db4d6e5dadb8c6ecdaa7483b7ef7cfc3','installation':'offline wheel --no-index --no-deps in fresh candidate venv'},
 'local_instructions':{'AGENTS_md_found':False,'checkout_agents_skills_found':False,'searched':['project root and ancestors','task workspace and ancestors','/Users/lkc/Downloads/projects (AGENTS.md/SKILL.md)','/Users/lkc/Documents/Codex (AGENTS.md/SKILL.md)']},
 'approvals':{'denied_actions':[],'pending_actions':[],'uncertain_actions':[]}
})
now=datetime.now(timezone.utc)
logical=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
allocated=sum(p.stat().st_blocks*512 for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
network=sum(x.get('bytes_received',0) for x in logs)
start=WORKSPACE.stat().st_birthtime
ledger={
 'scope':'Asset preparation and strict CPU loading only; no research measurements',
 'workspace_created_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
 'ledger_recorded_utc':now.isoformat(),
 'elapsed_since_workspace_creation_seconds':now.timestamp()-start,
 'wall_limit_seconds':25*60,
 'download_body_bytes_exact_logged':network,
 'connector_web_inspection_reserve_bytes':5*1024**2,
 'download_bytes_conservative_accounted':network+5*1024**2,
 'download_limit_bytes':500*1024**2,
 'disk_logical_bytes_before_ledger':logical,
 'disk_allocated_bytes_before_ledger':allocated,
 'disk_conservative_peak_upper_bound_bytes':allocated+(ROOT/'downloads/MILLs-DSLR.zip').stat().st_size+5*1024**2,
 'disk_peak_bound_note':'Actual peak was not sampled; upper bound adds a full archive-size margin and 5 MiB metadata margin to final allocated disk. Original dependency paths are read-only and were not copied.',
 'disk_peak_limit_bytes':int(1.5*1024**3),
 'download_request_wall_seconds_sum':sum(x['wall_seconds'] for x in logs),
 'data_verification_wall_seconds':data['wall_seconds'],
 'data_verification_cpu_seconds':data['process_cpu_seconds'],
 'data_verification_max_rss_bytes':data['max_rss_bytes'],
 'model_verification_wall_seconds':model['wall_seconds'],
 'model_verification_cpu_seconds':model['process_cpu_seconds'],
 'model_verification_max_rss_bytes':model['max_rss_bytes'],
 'other_process_cpu_time':'Not individually instrumented; whole-task wall bound retained',
 'gpu_seconds':0,'cuda_used':False,'mps_used':False,'colab_allocations':0,'model_forwards':0,'research_metrics':0,
 'validation_pixels_decoded':0,'credentials_created':False,'new_service_logins':0,'purchases':0,'private_data_uploads':0,'pushes_or_publications':0,
 'failed_network_requests':[{'url':x['url'],'error':x['error'],'attempt':x['attempt']} for x in logs if x['status']=='failed'],
 'retry_note':'Same HTTPS origin only; TLS checking never disabled. README 404 resolved using observed official tree filename case. No authorization denials.'
}
assert ledger['elapsed_since_workspace_creation_seconds']<ledger['wall_limit_seconds']
assert ledger['download_bytes_conservative_accounted']<ledger['download_limit_bytes']
assert ledger['disk_conservative_peak_upper_bound_bytes']<ledger['disk_peak_limit_bytes']
save('reports/resource_ledger.json',ledger)
save('reports/final_audit.json',dict(status='PASS_ASSETS_ONLY',checked_utc=now.isoformat(),panel_files=84,full_manifest_files=len(manifest),independent_selection_and_pairing_rederived=True,all_panel_files_rehashed=True,all_download_snapshots_rehashed=True,download_records=len(snapshots),model_strict_load_evidence_passed=True,model_forwards=0,metrics=0,gpu_seconds=0,asset_blockers=[],pending=['Upstream mathematical protocol and independent review','Future authorized forward/runtime/numeric verification'],unknown=['Physical semantics of author _a/_b suffixes beyond complete official scene identifier','Per-image raw exposure and lux metadata not present in this subset']))
(ROOT/'.gitignore').write_text('.asset-venv/\ndownloads/\ndata/\nmodels/model.safetensors\n**/__pycache__/\n*.pyc\n')
(ROOT/'README.md').write_text('MILL proxy candidate: asset preparation complete; no model forward or research metrics executed.\n\nSee [完整资产核验报告](reports/ASSET_REPORT_ZH.md), [固定 84 图 panel](reports/panel.json), [CPU 严格加载证据](reports/model_cpu_verification.json), and [资源账本](reports/resource_ledger.json).\n\nThe full mathematical protocol remains with the parent review. This directory does not authorize or launch experiments.\n')
files=[]
for folder in ['scripts','reports','sources','vendor']:
    for p in sorted((ROOT/folder).rglob('*')):
        if p.is_file() and '__pycache__' not in str(p) and p.name!='deliverables.sha256':files.append(f'{sha(p)}  {p.relative_to(ROOT)}')
(ROOT/'reports/deliverables.sha256').write_text('\n'.join(files)+'\n')
print(json.dumps({'audit':'PASS_ASSETS_ONLY','panel_sha256':data['panel_sha256'],'download_MiB':network/1024**2,'budgeted_download_MiB':ledger['download_bytes_conservative_accounted']/1024**2,'disk_allocated_MiB':allocated/1024**2,'conservative_disk_peak_MiB':ledger['disk_conservative_peak_upper_bound_bytes']/1024**2,'wall_minutes':ledger['elapsed_since_workspace_creation_seconds']/60,'gpu':0,'forwards':0,'blockers':[]},indent=2))
